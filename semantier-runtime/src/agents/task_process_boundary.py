"""Linux task process filesystem boundary; cwd/prompt rules are not containment.

Only explicit task capabilities are writable. A private PID namespace contains
CLI-started gateways and tools. Network access is the configured provider's
transport, not a claim of network isolation.
"""

from pathlib import Path
import shutil


def isolated_command(command, *, writable, readonly, cwd, readonly_overlays=()):
    binary = shutil.which("bwrap")
    if not binary:
        raise RuntimeError("task filesystem isolation requires bubblewrap")
    writable = tuple(Path(p).resolve(strict=True) for p in writable)
    readonly = tuple(Path(p).absolute() for p in readonly)
    interpreter = Path(command[0]).absolute()
    interpreter_roots = []
    while interpreter.is_symlink():
        target = interpreter.readlink()
        interpreter = target if target.is_absolute() else interpreter.parent / target
        interpreter_roots.append(interpreter.parent.parent)
    readonly += tuple(interpreter_roots)
    cwd = Path(cwd).resolve(strict=True)
    if not any(cwd == p or cwd.is_relative_to(p) for p in writable):
        raise ValueError("task cwd must belong to an explicit writable capability")
    result = [
        binary,
        "--unshare-all",
        "--share-net",
        "--die-with-parent",
        "--new-session",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    for path in (
        "/usr",
        "/bin",
        "/lib",
        "/lib64",
        "/etc/ssl",
        "/etc/resolv.conf",
        "/etc/hosts",
        "/etc/nsswitch.conf",
        "/etc/passwd",
        "/etc/group",
    ):
        if Path(path).exists():
            result += ["--ro-bind", path, path]
    for path in readonly:
        result += ["--ro-bind", str(path), str(path)]
    for path in writable:
        # Reject broad capabilities even if supplied by a programming error.
        if str(path) in {"/", "/home", "/home/chris", "/tmp", "/usr", "/etc"}:
            raise ValueError("overbroad task capability")
        result += ["--bind", str(path), str(path)]
    for overlay in readonly_overlays:
        if Path(overlay).is_symlink():
            raise ValueError("read-only overlay cannot be a symlink")
        path = Path(overlay).resolve(strict=True)
        if not path.is_file() or not any(path.is_relative_to(root) for root in writable):
            raise ValueError("read-only overlay must be a file inside a writable capability")
        result += ["--ro-bind", str(path), str(path)]
    return result + ["--chdir", str(cwd), "--", *map(str, command)]
