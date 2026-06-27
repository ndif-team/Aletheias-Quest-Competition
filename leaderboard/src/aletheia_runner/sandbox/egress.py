"""Egress allowlist via seccomp user-notification on connect().

The sandboxed child installs a seccomp filter that turns every ``connect()`` into
a user-notification and hands the listener fd to a supervisor in the trusted
parent. The supervisor reads the target's ``sockaddr``, and allows the connect
only to: AF_UNIX, loopback, or an IP that currently resolves from an allowed
hostname (e.g. ``api.ndif.us``; ``pypi.org`` etc. during the install phase).
Everything else gets EPERM.

Caveats (documented, acceptable for semi-trusted participants): this filters TCP
``connect()`` only — UDP ``sendto`` (incl. DNS) is not filtered, so UDP exfil is
a residual risk; and address-allowlisting has an inherent TOCTOU window. It is a
strong default-deny egress control, not a hardware boundary.
"""

from __future__ import annotations

import array
import ctypes
import http.client
import os
import select
import socket
import ssl
import struct
import tempfile
import threading
import time

# x86_64 syscall / seccomp constants.
SYS_connect = 42
SYS_seccomp = 317
SYS_process_vm_readv = 310
PR_SET_NO_NEW_PRIVS = 38
SECCOMP_SET_MODE_FILTER = 1
SECCOMP_FILTER_FLAG_NEW_LISTENER = 8

BPF_LD = 0x00
BPF_W = 0x00
BPF_ABS = 0x20
BPF_JMP = 0x05
BPF_JEQ = 0x10
BPF_RET = 0x06
AUDIT_ARCH_X86_64 = 0xC000003E
SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_USER_NOTIF = 0x7FC00000
SECCOMP_RET_KILL_PROCESS = 0x80000000
OFF_NR = 0
OFF_ARCH = 4

# ioctls (computed from _IOWR('!', n, sizeof(struct))).
SECCOMP_IOCTL_NOTIF_RECV = 0xC0502100      # _IOWR('!',0, seccomp_notif[80])
SECCOMP_IOCTL_NOTIF_SEND = 0xC0182101      # _IOWR('!',1, seccomp_notif_resp[24])
SECCOMP_IOCTL_NOTIF_ID_VALID = 0x40082102  # _IOW('!',2, __u64)
SECCOMP_USER_NOTIF_FLAG_CONTINUE = 1

AF_UNIX, AF_INET, AF_INET6 = 1, 2, 10
EPERM = 1


class _SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8),
                ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(_SockFilter))]


# struct seccomp_notif { u64 id; u32 pid; u32 flags; seccomp_data data(64) }
class _SeccompData(ctypes.Structure):
    _fields_ = [("nr", ctypes.c_int32), ("arch", ctypes.c_uint32),
                ("instruction_pointer", ctypes.c_uint64),
                ("args", ctypes.c_uint64 * 6)]


class _SeccompNotif(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint64), ("pid", ctypes.c_uint32),
                ("flags", ctypes.c_uint32), ("data", _SeccompData)]


class _SeccompNotifResp(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint64), ("val", ctypes.c_int64),
                ("error", ctypes.c_int32), ("flags", ctypes.c_uint32)]


class _Iovec(ctypes.Structure):
    _fields_ = [("base", ctypes.c_void_p), ("len", ctypes.c_size_t)]


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


# --- child side: install the connect() notifier ---------------------------

def install_connect_notifier() -> int:
    """Install a filter that user-notifies on connect(); returns the listener fd.

    Call in the child (preexec). NO_NEW_PRIVS must be settable.
    """
    prog = (_SockFilter * 7)(
        _SockFilter(BPF_LD | BPF_W | BPF_ABS, 0, 0, OFF_ARCH),
        _SockFilter(BPF_JMP | BPF_JEQ, 1, 0, AUDIT_ARCH_X86_64),  # x86_64? skip kill
        _SockFilter(BPF_RET, 0, 0, SECCOMP_RET_KILL_PROCESS),
        _SockFilter(BPF_LD | BPF_W | BPF_ABS, 0, 0, OFF_NR),
        _SockFilter(BPF_JMP | BPF_JEQ, 0, 1, SYS_connect),        # connect? notify : allow
        _SockFilter(BPF_RET, 0, 0, SECCOMP_RET_USER_NOTIF),
        _SockFilter(BPF_RET, 0, 0, SECCOMP_RET_ALLOW),
    )
    fprog = _SockFprog(len(prog), prog)
    libc = _libc()
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "NO_NEW_PRIVS failed")
    fd = libc.syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER,
                      SECCOMP_FILTER_FLAG_NEW_LISTENER, ctypes.byref(fprog))
    if fd < 0:
        raise OSError(ctypes.get_errno(), "seccomp(NEW_LISTENER) failed")
    return fd


def send_fd(sock: socket.socket, fd: int) -> None:
    sock.sendmsg([b"x"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                           array.array("i", [fd]))])


def recv_fd(sock: socket.socket) -> int:
    fds = array.array("i")
    _, ancdata, _, _ = sock.recvmsg(1, socket.CMSG_LEN(fds.itemsize))
    for level, typ, data in ancdata:
        if level == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS:
            fds.frombytes(data[:len(data) - (len(data) % fds.itemsize)])
            return fds[0]
    raise OSError("no fd received")


# --- egress: loopback-only seccomp gate + a hostname CONNECT proxy ----------

def _loopback_only(family: int, ip: str, port: int = 0) -> bool:
    """The child may connect only to the loopback CONNECT proxy (or AF_UNIX).

    Everything external must go through the proxy (which filters by hostname),
    so a notebook that ignores the proxy env simply can't connect — fail-closed.
    """
    if family == AF_UNIX:
        return True
    if family == AF_INET and ip.startswith("127."):
        return True
    if family == AF_INET6 and ip in ("::1", "::ffff:127.0.0.1"):
        return True
    return False


class _CertAuthority:
    """Ephemeral CA that mints per-host leaf certs so the proxy can terminate TLS
    for MITM'd hosts. Generated fresh per run; the private key never leaves the
    parent. The child trusts it via an injected CA bundle (see runner._run)."""

    def __init__(self):
        import datetime

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID

        self._x509, self._hashes, self._ser = x509, hashes, serialization
        self._ec, self._NameOID = ec, NameOID
        self._key = ec.generate_private_key(ec.SECP256R1())
        now = datetime.datetime.now(datetime.timezone.utc)
        self._nb = now - datetime.timedelta(hours=1)
        self._na = now + datetime.timedelta(days=2)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,
                                             "Aletheia Egress Proxy CA")])
        self._cert = (x509.CertificateBuilder()
                      .subject_name(name).issuer_name(name)
                      .public_key(self._key.public_key())
                      .serial_number(x509.random_serial_number())
                      .not_valid_before(self._nb).not_valid_after(self._na)
                      .add_extension(x509.BasicConstraints(ca=True, path_length=0),
                                     critical=True)
                      .add_extension(x509.KeyUsage(
                          digital_signature=False, content_commitment=False,
                          key_encipherment=False, data_encipherment=False,
                          key_agreement=False, key_cert_sign=True, crl_sign=True,
                          encipher_only=False, decipher_only=False), critical=True)
                      .sign(self._key, hashes.SHA256()))
        self.cert_pem = self._cert.public_bytes(serialization.Encoding.PEM)
        self._dir = tempfile.mkdtemp(prefix="aletheia-mitm-")
        self._ctx: dict[str, ssl.SSLContext] = {}

    def server_context(self, host: str) -> ssl.SSLContext:
        ctx = self._ctx.get(host)
        if ctx is not None:
            return ctx
        x509, ec, ser = self._x509, self._ec, self._ser
        from cryptography.x509.oid import ExtendedKeyUsageOID
        key = ec.generate_private_key(ec.SECP256R1())
        leaf = (x509.CertificateBuilder()
                .subject_name(x509.Name([x509.NameAttribute(
                    self._NameOID.COMMON_NAME, host)]))
                .issuer_name(self._cert.subject)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(self._nb).not_valid_after(self._na)
                .add_extension(x509.SubjectAlternativeName([x509.DNSName(host)]),
                               critical=False)
                .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
                               critical=False)
                .sign(self._key, self._hashes.SHA256()))
        pem = (leaf.public_bytes(ser.Encoding.PEM)
               + key.private_bytes(ser.Encoding.PEM,
                                   ser.PrivateFormat.TraditionalOpenSSL,
                                   ser.NoEncryption()))
        path = os.path.join(self._dir, host.replace("/", "_") + ".pem")
        with open(path, "wb") as f:
            f.write(pem)
        c = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        c.load_cert_chain(path)
        self._ctx[host] = c
        return c

    def close(self) -> None:
        import shutil
        shutil.rmtree(self._dir, ignore_errors=True)


# Request methods that can only READ. Everything else (POST/PUT/PATCH/DELETE — the
# verbs every HF upload path uses) is refused on MITM'd hosts, so a notebook can
# fetch model configs but can't push the private eval inputs to a repo it controls.
READ_METHODS = frozenset({"GET", "HEAD"})
# Hop-by-hop / host headers not forwarded upstream.
_HOP = frozenset({"connection", "proxy-connection", "keep-alive", "transfer-encoding",
                  "upgrade", "te", "trailer", "host"})


class AllowProxy:
    """A loopback HTTP CONNECT proxy that allowlists egress by **hostname suffix**.

    Two policies per allowed host:
      * **tunnel** (default, e.g. NDIF) — blind-tunnel TCP after CONNECT, no TLS
        interception;
      * **mitm** (``mitm_suffixes``, e.g. huggingface.co) — terminate TLS with a
        minted leaf cert and forward only ``READ_METHODS`` requests upstream,
        refusing writes. This closes the one bulk exfil channel an allowed,
        team-writable host (HF) would otherwise leave open.

    Filtering by name (not IP) is CDN-proof; the proxy resolves DNS itself so the
    child needs none. Runs in the trusted parent (unsandboxed).
    """

    def __init__(self, allowed_suffixes: list[str], mitm_suffixes: list[str] | None = None,
                 allowed_methods: frozenset[str] = READ_METHODS, log: list | None = None):
        self.allowed = [s.lower().lstrip(".") for s in allowed_suffixes]
        self.mitm = [s.lower().lstrip(".") for s in (mitm_suffixes or [])]
        self.allowed_methods = frozenset(m.upper() for m in allowed_methods)
        self.log = log
        self._ca = _CertAuthority() if self.mitm else None
        self._stop = threading.Event()
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(128)
        self.port = self._srv.getsockname()[1]

    @property
    def ca_cert_pem(self) -> bytes | None:
        """PEM of the MITM CA the child must trust (None when nothing is MITM'd)."""
        return self._ca.cert_pem if self._ca else None

    def host_ok(self, host: str) -> bool:
        h = host.lower().rstrip(".")
        return any(h == s or h.endswith("." + s) for s in self.allowed)

    def _is_mitm(self, host: str) -> bool:
        h = host.lower().rstrip(".")
        return any(h == s or h.endswith("." + s) for s in self.mitm)

    def start(self) -> int:
        threading.Thread(target=self._serve, daemon=True).start()
        return self.port

    def stop(self) -> None:
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass
        if self._ca is not None:
            self._ca.close()

    def _serve(self) -> None:
        self._srv.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except (socket.timeout, OSError):
                continue
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(30)
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = conn.recv(4096)
                if not chunk or len(buf) > 16384:
                    conn.close()
                    return
                buf += chunk
            parts = buf.split(b"\r\n", 1)[0].decode("latin1", "replace").split()
            if len(parts) < 2 or parts[0].upper() != "CONNECT":
                conn.sendall(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
                conn.close()
                return
            host, _, port_s = parts[1].partition(":")
            port = int(port_s) if port_s.isdigit() else 443
            if not self.host_ok(host):
                if self.log is not None:
                    self.log.append(("deny", host, port))
                conn.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
                conn.close()
                return
            if self._is_mitm(host):
                self._mitm(conn, host, port)
                return
            try:
                up = socket.create_connection((host, port), timeout=30)
            except OSError:
                conn.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                conn.close()
                return
            if self.log is not None:
                self.log.append(("allow", host, port))
            conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            self._pipe(conn, up)
        except OSError:
            try:
                conn.close()
            except OSError:
                pass

    # --- MITM path: terminate TLS, forward only read methods upstream ----------

    def _mitm(self, conn: socket.socket, host: str, port: int) -> None:
        if self.log is not None:
            self.log.append(("mitm", host, port))
        try:
            conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            tls = self._ca.server_context(host).wrap_socket(conn, server_side=True)
        except (OSError, ssl.SSLError):
            try:
                conn.close()
            except OSError:
                pass
            return
        try:
            self._filter_one(tls, host, port)
        finally:
            try:
                tls.close()
            except OSError:
                pass

    def _filter_one(self, tls: ssl.SSLSocket, host: str, port: int) -> None:
        """Read one request off the terminated TLS stream, refuse non-read methods,
        forward the rest upstream. One request per connection (we answer with
        ``Connection: close``), which keeps framing simple and is plenty for HF."""
        tls.settimeout(30)
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = tls.recv(65536)
            if not chunk or len(buf) > (1 << 20):
                return
            buf += chunk
        head, _, body = buf.partition(b"\r\n\r\n")
        lines = head.split(b"\r\n")
        parts = lines[0].decode("latin1", "replace").split(" ")
        if len(parts) < 2:
            tls.sendall(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
            return
        method, path = parts[0].upper(), parts[1]
        headers = []
        for ln in lines[1:]:
            k, sep, v = ln.partition(b":")
            if sep:
                headers.append((k.decode("latin1").strip(), v.decode("latin1").strip()))
        hmap = {k.lower(): v for k, v in headers}
        try:
            clen = int(hmap.get("content-length", "0") or "0")
        except ValueError:
            clen = 0
        # Drain a bounded request body before responding, so the client finishes
        # sending and reads our reply cleanly (a denied POST otherwise gets a reset
        # that surfaces as a vague connection error instead of a 403). Huge uploads
        # past the cap just get reset — still blocked.
        want = clen if method in self.allowed_methods else min(clen, 1 << 20)
        while len(body) < want:
            chunk = tls.recv(65536)
            if not chunk:
                break
            body += chunk
        if method not in self.allowed_methods:
            if self.log is not None:
                self.log.append(("deny-method", host, method))
            tls.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n"
                        b"Connection: close\r\n\r\n")
            return
        if self.log is not None:
            self.log.append(("allow-method", host, method))
        self._relay_upstream(tls, host, port, method, path, headers, body)

    def _relay_upstream(self, tls, host, port, method, path, headers, body) -> None:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
        fwd = {k: v for k, v in headers if k.lower() not in _HOP}
        fwd["Host"] = host
        up = None
        try:
            up = http.client.HTTPSConnection(host, port, timeout=60, context=ctx)
            up.request(method, path, body=body or None, headers=fwd)
            resp = up.getresponse()
            # http.client decodes Transfer-Encoding (chunked) but NOT Content-Encoding,
            # so `data` is the entity body exactly as the client should receive it.
            data = b"" if method == "HEAD" else resp.read()
        except OSError:
            try:
                tls.sendall(b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n")
            except OSError:
                pass
            if up is not None:
                up.close()
            return
        # Re-emit with our own framing: keep every header except the upstream's
        # body-framing ones, then set one correct Content-Length. HEAD carries no
        # body but must keep the upstream's Content-Length (it's the file size the
        # client reads for metadata); GET gets the length of the bytes we send.
        out = bytearray(f"HTTP/1.1 {resp.status} {resp.reason}\r\n".encode("latin1"))
        upstream_clen = None
        for k, v in resp.getheaders():
            kl = k.lower()
            if kl == "content-length":
                upstream_clen = v
                continue
            if kl in ("connection", "transfer-encoding"):
                continue
            out += f"{k}: {v}\r\n".encode("latin1")
        clen = upstream_clen if method == "HEAD" and upstream_clen is not None else str(len(data))
        out += f"Content-Length: {clen}\r\n".encode("latin1")
        out += b"Connection: close\r\n\r\n"
        try:
            tls.sendall(bytes(out) + data)
        except OSError:
            pass
        finally:
            up.close()

    @staticmethod
    def _pipe(a: socket.socket, b: socket.socket) -> None:
        for s in (a, b):
            s.settimeout(None)
        try:
            while True:
                r, _, _ = select.select([a, b], [], [], 300)
                if not r:
                    break
                for s in r:
                    data = s.recv(65536)
                    if not data:
                        return
                    (b if s is a else a).sendall(data)
        except OSError:
            pass
        finally:
            for s in (a, b):
                try:
                    s.close()
                except OSError:
                    pass


# --- supervisor (parent side) ---------------------------------------------

def _read_sockaddr(pid: int, addr: int, length: int) -> bytes:
    """Read the target's sockaddr via /proc/<pid>/mem (parent has ptrace access)."""
    length = max(0, min(length, 128))
    if length == 0:
        return b""
    try:
        with open(f"/proc/{pid}/mem", "rb", 0) as f:
            f.seek(addr)
            return f.read(length)
    except OSError:
        return b""


def _parse(sa: bytes) -> tuple[int, str, int]:
    if len(sa) < 2:
        return -1, "", 0
    family = struct.unpack_from("H", sa, 0)[0]
    if family == AF_INET and len(sa) >= 8:
        port = struct.unpack_from("!H", sa, 2)[0]
        return family, socket.inet_ntop(socket.AF_INET, sa[4:8]), port
    if family == AF_INET6 and len(sa) >= 24:
        port = struct.unpack_from("!H", sa, 2)[0]
        return family, socket.inet_ntop(socket.AF_INET6, sa[8:24]), port
    return family, "", 0


def serve(listener_fd: int, allow, stop: threading.Event,
          log: list | None = None) -> None:
    """Handle connect() notifications until ``stop`` is set. Closes the fd.

    ``allow`` is a callable ``(family, ip, port) -> bool`` (e.g. _loopback_only).
    """
    libc = _libc()
    try:
        while not stop.is_set():
            r, _, _ = select.select([listener_fd], [], [], 0.2)
            if not r:
                continue
            notif = _SeccompNotif()
            if libc.ioctl(listener_fd, SECCOMP_IOCTL_NOTIF_RECV,
                          ctypes.byref(notif)) != 0:
                continue
            family, ip, port = _parse(_read_sockaddr(
                notif.pid, notif.data.args[1], int(notif.data.args[2])))
            ok = allow(family, ip, port)
            if log is not None:
                log.append((family, ip, port, ok))

            resp = _SeccompNotifResp(id=notif.id)
            # Re-check the notification is still valid before answering.
            valid = ctypes.c_uint64(notif.id)
            if libc.ioctl(listener_fd, SECCOMP_IOCTL_NOTIF_ID_VALID,
                          ctypes.byref(valid)) != 0:
                continue
            if ok:
                resp.flags = SECCOMP_USER_NOTIF_FLAG_CONTINUE
            else:
                resp.error = -EPERM
            libc.ioctl(listener_fd, SECCOMP_IOCTL_NOTIF_SEND, ctypes.byref(resp))
    finally:
        try:
            os.close(listener_fd)
        except OSError:
            pass


# --- self-test ------------------------------------------------------------

def _selftest() -> dict:
    """Child installs the notifier and tries loopback (allow) + 8.8.8.8 (deny)."""
    import errno
    import json

    parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        parent_sock.close()
        os.close(r)
        try:
            fd = install_connect_notifier()
            send_fd(child_sock, fd)
            os.close(fd)
            res = {}
            for name, target in [("loopback", "127.0.0.1"), ("public", "8.8.8.8")]:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(3)
                try:
                    s.connect((target, 9))
                    res[name] = "allowed"
                except OSError as e:
                    res[name] = ("denied" if e.errno == errno.EPERM
                                 else f"allowed:{errno.errorcode.get(e.errno, e.errno)}")
                s.close()
            os.write(w, json.dumps(res).encode())
        except BaseException as e:  # noqa: BLE001
            os.write(w, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode())
        os._exit(0)

    child_sock.close()
    os.close(w)
    stop = threading.Event()

    def supervise():
        fd = recv_fd(parent_sock)
        serve(fd, _loopback_only, stop)
    threading.Thread(target=supervise, daemon=True).start()
    buf = b""
    while True:
        c = os.read(r, 4096)
        if not c:
            break
        buf += c
    os.waitpid(pid, 0)
    stop.set()
    return json.loads(buf.decode())


if __name__ == "__main__":
    import json
    print(json.dumps(_selftest()))
