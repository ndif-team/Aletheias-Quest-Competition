"""seccomp-bpf syscall blocklist (raw BPF via ctypes, no libseccomp dependency).

Installs a filter that returns EPERM for a denylist of dangerous syscalls and
allows everything else. This shrinks the kernel attack surface and closes
cross-process vectors (ptrace, process_vm_readv) that Landlock can't — together
with Landlock denying /proc, that blocks reading another process's memory/env.

Unprivileged (needs only PR_SET_NO_NEW_PRIVS). x86_64 only; on other arches the
filter kills the process via the arch guard. Apply in the child before exec.
"""

from __future__ import annotations

import ctypes

PR_SET_NO_NEW_PRIVS = 38
SYS_seccomp = 317
SECCOMP_SET_MODE_FILTER = 1

# Classic BPF opcodes.
BPF_LD = 0x00
BPF_W = 0x00
BPF_ABS = 0x20
BPF_ALU = 0x04
BPF_AND = 0x50
BPF_JMP = 0x05
BPF_JEQ = 0x10
BPF_K = 0x00
BPF_RET = 0x06

AUDIT_ARCH_X86_64 = 0xC000003E
SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ERRNO = 0x00050000
EPERM = 1

# seccomp_data: { int nr; __u32 arch; __u64 ip; __u64 args[6]; }
OFF_NR = 0
OFF_ARCH = 4
OFF_ARG0 = 16          # args[0] low 32 bits (x86_64 little-endian)
OFF_ARG1 = 24          # args[1] low 32 bits

# Deny creating internet datagram sockets so a notebook can't exfiltrate the
# (private) eval inputs over UDP/DNS — the TCP connect() gate forces all TCP
# through the loopback CONNECT proxy, but unconnected UDP sendto bypasses it, and
# the child needs no UDP of its own (the proxy resolves DNS for it).
SYS_socket = 41
AF_INET, AF_INET6 = 2, 10
SOCK_DGRAM = 2
SOCK_TYPE_MASK = 0xF   # low bits hold the type; SOCK_CLOEXEC/NONBLOCK are higher

# x86_64 syscall numbers to deny (none are used by normal Python/ML code).
DEFAULT_DENY = {
    "ptrace": 101, "process_vm_readv": 310, "process_vm_writev": 311, "kcmp": 312,
    "kexec_load": 246, "kexec_file_load": 320,
    "init_module": 175, "finit_module": 313, "delete_module": 176, "create_module": 174,
    "mount": 165, "umount2": 166, "move_mount": 429, "pivot_root": 155, "chroot": 161,
    "open_tree": 428, "fsopen": 430, "fsconfig": 431, "fsmount": 432, "mount_setattr": 442,
    "setns": 308, "unshare": 272,
    "bpf": 321, "perf_event_open": 298, "userfaultfd": 323,
    "add_key": 248, "request_key": 249, "keyctl": 250,
    "reboot": 169, "swapon": 167, "swapoff": 168, "acct": 163, "quotactl": 179,
    "settimeofday": 164, "clock_settime": 227, "adjtimex": 159, "_sysctl": 156,
}


class _SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8),
                ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(_SockFilter))]


def _build_program(deny_nrs: list[int]) -> "ctypes.Array":
    stmts = [
        # Guard: only run for x86_64; otherwise kill the process.
        _SockFilter(BPF_LD | BPF_W | BPF_ABS, 0, 0, OFF_ARCH),
        _SockFilter(BPF_JMP | BPF_JEQ | BPF_K, 1, 0, AUDIT_ARCH_X86_64),
        _SockFilter(BPF_RET | BPF_K, 0, 0, SECCOMP_RET_KILL_PROCESS),
        # Load the syscall number.
        _SockFilter(BPF_LD | BPF_W | BPF_ABS, 0, 0, OFF_NR),
    ]
    # For each denied syscall: if nr == x, return EPERM; else skip the RET.
    for nr in deny_nrs:
        stmts.append(_SockFilter(BPF_JMP | BPF_JEQ | BPF_K, 0, 1, nr))
        stmts.append(_SockFilter(BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ERRNO | EPERM))

    # socket(domain, type, ...) arg-filter: EPERM iff domain in {AF_INET,AF_INET6}
    # and (type & 0xF) == SOCK_DGRAM. A still holds the syscall nr here. Jump
    # targets are relative to the 9-instruction block below (EPERM at +7, ALLOW +8).
    stmts += [
        _SockFilter(BPF_JMP | BPF_JEQ | BPF_K, 0, 7, SYS_socket),    # +0 not socket -> ALLOW
        _SockFilter(BPF_LD | BPF_W | BPF_ABS, 0, 0, OFF_ARG0),       # +1 A = domain
        _SockFilter(BPF_JMP | BPF_JEQ | BPF_K, 1, 0, AF_INET),       # +2 inet  -> +4
        _SockFilter(BPF_JMP | BPF_JEQ | BPF_K, 0, 4, AF_INET6),      # +3 inet6 -> +4 else ALLOW
        _SockFilter(BPF_LD | BPF_W | BPF_ABS, 0, 0, OFF_ARG1),       # +4 A = type
        _SockFilter(BPF_ALU | BPF_AND | BPF_K, 0, 0, SOCK_TYPE_MASK),# +5 A &= 0xF
        _SockFilter(BPF_JMP | BPF_JEQ | BPF_K, 0, 1, SOCK_DGRAM),    # +6 dgram -> EPERM else ALLOW
        _SockFilter(BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ERRNO | EPERM),  # +7
        _SockFilter(BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ALLOW),       # +8
    ]
    return (_SockFilter * len(stmts))(*stmts)


def install_blocklist(deny: dict[str, int] | None = None) -> int:
    """Install the seccomp blocklist on the current process. Returns #denied."""
    deny = DEFAULT_DENY if deny is None else deny
    prog = _build_program(list(deny.values()))
    fprog = _SockFprog(len(prog), prog)

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
    rc = libc.syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, 0, ctypes.byref(fprog))
    if rc != 0:
        raise OSError(ctypes.get_errno(), "seccomp(SET_MODE_FILTER) failed")
    return len(deny)


# --- self-test ------------------------------------------------------------

def _selftest() -> dict:
    import os
    import socket
    results: dict[str, str] = {}
    n = install_blocklist()
    results["installed_denied_count"] = str(n)

    # getpid (allowed) still works.
    results["getpid_allowed"] = "ok" if os.getpid() > 0 else "FAILED"

    # ptrace(PTRACE_TRACEME=0) must now return EPERM.
    libc = ctypes.CDLL(None, use_errno=True)
    rc = libc.ptrace(0, 0, 0, 0)
    err = ctypes.get_errno()
    results["ptrace_blocked"] = ("EPERM (expected)" if rc == -1 and err == EPERM
                                 else f"NOT BLOCKED (rc={rc}, errno={err})")

    def _sock(family, typ):
        try:
            socket.socket(family, typ).close()
            return "ALLOWED"
        except OSError as e:
            return f"DENIED({e.errno})"

    # UDP/UDP6 internet sockets must be denied (no DNS/UDP exfil path).
    results["udp_inet_blocked"] = _sock(socket.AF_INET, socket.SOCK_DGRAM)
    results["udp_inet6_blocked"] = _sock(socket.AF_INET6, socket.SOCK_DGRAM)
    # TCP (for the loopback proxy) and AF_UNIX datagram (local) must still work.
    results["tcp_inet_allowed"] = _sock(socket.AF_INET, socket.SOCK_STREAM)
    results["unix_dgram_allowed"] = _sock(socket.AF_UNIX, socket.SOCK_DGRAM)
    return results


if __name__ == "__main__":
    import json
    import os

    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            out = {"ok": True, "results": _selftest()}
        except BaseException as e:  # noqa: BLE001
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        os.write(w, json.dumps(out).encode())
        os._exit(0)
    os.close(w)
    buf = b""
    while True:
        c = os.read(r, 4096)
        if not c:
            break
        buf += c
    os.waitpid(pid, 0)
    print(buf.decode())
