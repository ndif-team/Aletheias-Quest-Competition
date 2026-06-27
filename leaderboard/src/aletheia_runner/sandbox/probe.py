"""Capability probe: what unprivileged sandboxing primitives work *here*?

Run inside the target environment (e.g. the HF Space) to discover which
confinement mechanisms an unprivileged process can actually use, since managed
container platforms often block some via their own seccomp profile.

Each risky attempt runs in a forked child so installing a seccomp filter (or a
failed syscall) can't damage the caller. Output is JSON.

    python -m aletheia_runner.sandbox.probe
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import struct
import sys

# x86_64 constants (the Space is x86_64).
PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2
SYS_seccomp = 317
SECCOMP_SET_MODE_FILTER = 1
SECCOMP_FILTER_FLAG_NEW_LISTENER = 8
SECCOMP_RET_ALLOW = 0x7FFF0000
SYS_landlock_create_ruleset = 444
SYS_unshare = 272
CLONE_NEWUSER = 0x10000000
CLONE_NEWNET = 0x40000000
CLONE_NEWNS = 0x00020000


class SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8),
                ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]


class SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(SockFilter))]


def _allow_all_fprog() -> SockFprog:
    prog = (SockFilter * 1)(SockFilter(0x06, 0, 0, SECCOMP_RET_ALLOW))  # BPF_RET|BPF_K
    return SockFprog(1, prog)


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def _run_in_child(fn) -> dict:
    """Run fn() in a forked child; return {ok, detail}. fn returns a detail str
    or raises; an OSError-style errno can be attached."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(r)
        try:
            detail = fn() or ""
            payload = {"ok": True, "detail": detail}
        except BaseException as e:  # noqa: BLE001
            payload = {"ok": False, "detail": f"{type(e).__name__}: {e}"}
        try:
            os.write(w, json.dumps(payload).encode())
        finally:
            os._exit(0)
    os.close(w)
    buf = b""
    while True:
        chunk = os.read(r, 4096)
        if not chunk:
            break
        buf += chunk
    os.close(r)
    _, status = os.waitpid(pid, 0)
    try:
        out = json.loads(buf.decode())
    except Exception:
        return {"ok": False, "detail": f"child died (status {status})"}
    return out


# --- individual capability tests (each run in a child) --------------------

def _t_no_new_privs():
    libc = _libc()
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(NO_NEW_PRIVS) failed")
    return "PR_SET_NO_NEW_PRIVS=1"


def _t_seccomp_filter():
    libc = _libc()
    libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
    fprog = _allow_all_fprog()
    res = libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(fprog), 0, 0)
    if res != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_SECCOMP filter rejected")
    return "installed allow-all seccomp-bpf filter"


def _t_seccomp_user_notif():
    libc = _libc()
    libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
    fprog = _allow_all_fprog()
    fd = libc.syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER,
                      SECCOMP_FILTER_FLAG_NEW_LISTENER, ctypes.byref(fprog))
    if fd < 0:
        raise OSError(ctypes.get_errno(), "seccomp(NEW_LISTENER) failed")
    os.close(fd)
    return f"got user-notification listener fd={fd}"


def _t_landlock():
    libc = _libc()
    abi = libc.syscall(SYS_landlock_create_ruleset, 0, 0, 1)  # flags=VERSION
    if abi < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset(VERSION) failed")
    return f"Landlock ABI version {abi}"


def _t_userns():
    libc = _libc()
    if libc.syscall(SYS_unshare, CLONE_NEWUSER) != 0:
        raise OSError(ctypes.get_errno(), "unshare(CLONE_NEWUSER) failed")
    return "unprivileged user namespace OK"


def _t_netns_via_userns():
    libc = _libc()
    if libc.syscall(SYS_unshare, CLONE_NEWUSER | CLONE_NEWNET | CLONE_NEWNS) != 0:
        raise OSError(ctypes.get_errno(), "unshare(USER|NET|NS) failed")
    return "user+net+mount namespaces OK (bwrap-style sandbox viable)"


def _static_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "arch": platform.machine(),
        "kernel": platform.release(),
        "uid": os.getuid(),
        "gid": os.getgid(),
    }
    try:
        status = open("/proc/self/status").read()
        for key in ("Seccomp", "NoNewPrivs", "CapEff", "CapBnd"):
            for line in status.splitlines():
                if line.startswith(key + ":"):
                    info[key] = line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        info["unprivileged_userns_clone"] = \
            open("/proc/sys/kernel/unprivileged_userns_clone").read().strip()
    except OSError:
        info["unprivileged_userns_clone"] = "(absent)"
    info["tools"] = {t: bool(shutil.which(t)) for t in ("bwrap", "nsjail", "runsc")}
    return info


def probe() -> dict:
    tests = {
        "no_new_privs": _t_no_new_privs,
        "seccomp_bpf_filter": _t_seccomp_filter,
        "seccomp_user_notif": _t_seccomp_user_notif,
        "landlock": _t_landlock,
        "user_namespace": _t_userns,
        "net_mount_userns": _t_netns_via_userns,
    }
    return {
        "info": _static_info(),
        "capabilities": {name: _run_in_child(fn) for name, fn in tests.items()},
    }


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2))
