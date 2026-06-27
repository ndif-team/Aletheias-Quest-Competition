"""Minimal Landlock (LSM) filesystem confinement via raw syscalls (ctypes).

Confines the current process (and its children) to: read+execute on a set of
paths, read+write on another set, and **no access to anything else** — including
``/proc`` (so a same-uid child can't read the parent's ``/proc/<pid>/environ``
to steal a token). Landlock is unprivileged and irreversible once applied, so
call :meth:`LandlockSandbox.apply` in the child right before exec.

The Space supports Landlock ABI 6 (see space-sandbox-capabilities). No-op with a
clear error if Landlock is unavailable.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass, field

# x86_64 syscall numbers.
SYS_landlock_create_ruleset = 444
SYS_landlock_add_rule = 445
SYS_landlock_restrict_self = 446
PR_SET_NO_NEW_PRIVS = 38

LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1

# Filesystem access-right bits.
A_EXECUTE = 1 << 0
A_WRITE_FILE = 1 << 1
A_READ_FILE = 1 << 2
A_READ_DIR = 1 << 3
A_REMOVE_DIR = 1 << 4
A_REMOVE_FILE = 1 << 5
A_MAKE_CHAR = 1 << 6
A_MAKE_DIR = 1 << 7
A_MAKE_REG = 1 << 8
A_MAKE_SOCK = 1 << 9
A_MAKE_FIFO = 1 << 10
A_MAKE_BLOCK = 1 << 11
A_MAKE_SYM = 1 << 12
A_REFER = 1 << 13
A_TRUNCATE = 1 << 14
A_IOCTL_DEV = 1 << 15

RO_ACCESS = A_EXECUTE | A_READ_FILE | A_READ_DIR
# Rights that apply to a regular file (the rest are directory-only → EINVAL on a file).
FILE_RIGHTS = A_EXECUTE | A_WRITE_FILE | A_READ_FILE | A_TRUNCATE | A_IOCTL_DEV


class LandlockUnavailable(RuntimeError):
    pass


class _RulesetAttr(ctypes.Structure):
    # v1 layout (8 bytes); accepted on all ABIs >= 1.
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1  # packed: u64 + s32 = 12 bytes, no padding
    _fields_ = [("allowed_access", ctypes.c_uint64),
                ("parent_fd", ctypes.c_int32)]


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def abi_version() -> int:
    libc = _libc()
    v = libc.syscall(SYS_landlock_create_ruleset, None, 0,
                     LANDLOCK_CREATE_RULESET_VERSION)
    if v < 0:
        raise LandlockUnavailable(f"landlock unavailable (errno {ctypes.get_errno()})")
    return v


def _fs_mask(abi: int) -> int:
    mask = 0x1FFF  # ABI 1: bits 0..12
    if abi >= 2:
        mask |= A_REFER
    if abi >= 3:
        mask |= A_TRUNCATE
    if abi >= 5:
        mask |= A_IOCTL_DEV
    return mask


@dataclass
class LandlockSandbox:
    ro_paths: list[str] = field(default_factory=list)
    rw_paths: list[str] = field(default_factory=list)

    def apply(self) -> int:
        """Create + enforce the ruleset on this process. Returns the ABI used."""
        libc = _libc()
        abi = abi_version()
        full = _fs_mask(abi)

        attr = _RulesetAttr(handled_access_fs=full)
        ruleset_fd = libc.syscall(SYS_landlock_create_ruleset,
                                  ctypes.byref(attr), ctypes.sizeof(attr), 0)
        if ruleset_fd < 0:
            raise LandlockUnavailable(
                f"create_ruleset failed (errno {ctypes.get_errno()})")

        try:
            for path in self.ro_paths:
                self._allow(libc, ruleset_fd, path, RO_ACCESS & full)
            for path in self.rw_paths:
                self._allow(libc, ruleset_fd, path, full)  # full rights within scratch

            if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
                raise LandlockUnavailable(
                    f"PR_SET_NO_NEW_PRIVS failed (errno {ctypes.get_errno()})")
            if libc.syscall(SYS_landlock_restrict_self, ruleset_fd, 0) != 0:
                raise LandlockUnavailable(
                    f"restrict_self failed (errno {ctypes.get_errno()})")
        finally:
            os.close(ruleset_fd)
        return abi

    @staticmethod
    def _allow(libc, ruleset_fd: int, path: str, access: int) -> None:
        if not os.path.exists(path):
            return  # tolerate absent runtime dirs across images
        # Directory-only rights on a regular file are rejected with EINVAL.
        if not os.path.isdir(path):
            access &= FILE_RIGHTS
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        try:
            pb = _PathBeneathAttr(allowed_access=access, parent_fd=fd)
            rc = libc.syscall(SYS_landlock_add_rule, ruleset_fd,
                              LANDLOCK_RULE_PATH_BENEATH, ctypes.byref(pb), 0)
            if rc != 0:
                raise LandlockUnavailable(
                    f"add_rule({path}) failed (errno {ctypes.get_errno()})")
        finally:
            os.close(fd)


def default_system_ro_paths() -> list[str]:
    """System dirs a Python process needs to read/exec (across common images)."""
    import sys
    # /etc is granted wholesale: it's read-only config (certs, resolv.conf,
    # os-release, debian_version, ...) that runtimes/pip probe; our secrets live
    # in env (scrubbed) and runner code in /app, not /etc.
    paths = ["/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc"]
    for p in {sys.prefix, sys.base_prefix, os.path.dirname(sys.executable)}:
        paths.append(p)
    return [p for p in paths if os.path.exists(p)]


def essential_dev_paths() -> list[str]:
    """Device nodes a Python runtime needs (granted read+write)."""
    devs = ["/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom"]
    return [d for d in devs if os.path.exists(d)]


# --- self-test ------------------------------------------------------------

def _selftest(ro_dir: str, rw_dir: str, secret_path: str | None = None) -> dict:
    """Apply the sandbox, then probe what is/ isn't allowed. Run in a child."""
    results: dict[str, str] = {}
    abi = LandlockSandbox(
        ro_paths=default_system_ro_paths() + [ro_dir],
        rw_paths=[rw_dir],
    ).apply()
    results["abi"] = str(abi)

    # Exfil vector: a secret (e.g. the eval labels) sitting on the parent's disk
    # OUTSIDE every granted path must be unreadable from the confined child.
    if secret_path:
        try:
            with open(secret_path) as f:
                f.read()
            results["read_secret_outside"] = "ALLOWED [LEAK!]"
        except OSError as e:
            results["read_secret_outside"] = f"DENIED ({e.errno}) (expected)"

    ro_file = os.path.join(ro_dir, "data.txt")
    try:
        with open(ro_file) as f:
            f.read()
        results["read_ro_file"] = "ALLOWED (expected)"
    except OSError as e:
        results["read_ro_file"] = f"DENIED ({e.errno}) [unexpected]"

    try:
        with open(os.path.join(ro_dir, "tamper.txt"), "w") as f:
            f.write("x")
        results["write_ro_dir"] = "ALLOWED [unexpected!]"
    except OSError as e:
        results["write_ro_dir"] = f"DENIED ({e.errno}) (expected)"

    try:
        with open(os.path.join(rw_dir, "out.txt"), "w") as f:
            f.write("ok")
        results["write_rw_dir"] = "ALLOWED (expected)"
    except OSError as e:
        results["write_rw_dir"] = f"DENIED ({e.errno}) [unexpected]"

    # Token-theft vector: reading another process's environ via /proc.
    try:
        with open(f"/proc/{os.getppid()}/environ", "rb") as f:
            f.read(1)
        results["read_parent_environ"] = "ALLOWED [LEAK!]"
    except OSError as e:
        results["read_parent_environ"] = f"DENIED ({e.errno}) (expected)"

    return results


if __name__ == "__main__":
    import json
    import sys
    import tempfile

    ro = tempfile.mkdtemp(prefix="ll-ro-")
    rw = tempfile.mkdtemp(prefix="ll-rw-")
    with open(os.path.join(ro, "data.txt"), "w") as f:
        f.write("readable")
    # A "labels" secret in its own dir, granted to NEITHER ro nor rw.
    secret_dir = tempfile.mkdtemp(prefix="ll-secret-")
    secret = os.path.join(secret_dir, "labels.csv")
    with open(secret, "w") as f:
        f.write("id,deceptive\n0,1\n")

    # Run the actual test in a child so the irreversible restriction is contained.
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            out = {"ok": True, "results": _selftest(ro, rw, secret)}
        except BaseException as e:  # noqa: BLE001
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        os.write(w, json.dumps(out).encode())
        os._exit(0)
    os.close(w)
    data = b""
    while True:
        c = os.read(r, 4096)
        if not c:
            break
        data += c
    os.waitpid(pid, 0)
    print(data.decode())
