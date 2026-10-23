"""Local product bootstrap for the standalone Hermes CLI/Desktop entrypoint.

Product extras select shipped capabilities, never the SaaS host lifecycle.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

# Only the exact stock plugin shipped before standalone separation may be upgraded.
# Unknown user plugin edits remain user-owned.
_LEGACY_STOCK = {'provider.py': 'aa35695018a23ac9402c8fe18f0b4be2c0dc7b8dd42b8147087ae9eb814ab07e', '__init__.py': '0660b2640450eedfc4b92eef718b352550ffcea8eae541a540ae1e43be0d1b19', 'plugin.yaml': '517c6aa6c19c1866b136d2ec005056b51e800e1357d6d4028680e91e6849124e'}


def bootstrap_standalone_runtime() -> None:
    from pm.features import runtime_distribution_extras
    if runtime_distribution_extras() is None:
        return
    from gateway.execution_boundary import current_execution_boundary, get_execution_boundary_provider
    if current_execution_boundary() is not None or get_execution_boundary_provider() is not None:
        raise RuntimeError("standalone bootstrap cannot initialize a managed host boundary")
    from hermes_constants import get_hermes_home, get_default_hermes_root
    from pm.paths import repo_root
    from pm.filesystem import lock_fd, durable_write_bytes
    home = get_hermes_home().resolve()
    owner = get_default_hermes_root().resolve()
    if home != owner and home.parent != owner / "profiles":
        raise ValueError("active home is outside the standalone profile owner")
    source = repo_root() / "semantier-runtime/src/plugins/sensevoice_stt"
    names = tuple(_LEGACY_STOCK)
    if not all((source / name).is_file() for name in names):
        raise RuntimeError("packaged SenseVoice plugin source is missing")
    owner.mkdir(parents=True, exist_ok=True)
    lock = owner / ".standalone-bootstrap.lock"
    if lock.is_symlink():
        raise ValueError("standalone bootstrap lock must not be a link")
    with lock.open("a+b") as stream:
        if not lock_fd(stream.fileno(), wait=True):
            raise RuntimeError("standalone bootstrap lock unavailable")
        target = home / "plugins/sensevoice_stt"
        if (target.is_symlink() or getattr(target, "is_junction", lambda: False)()
                or not target.resolve().is_relative_to(home)):
            raise ValueError("standalone plugin escapes active home")
        if target.exists():
            known = all((target / n).is_file() and not (target / n).is_symlink()
                and hashlib.sha256((target / n).read_bytes()).hexdigest() in
                    {_LEGACY_STOCK[n], hashlib.sha256((source / n).read_bytes()).hexdigest()}
                for n in names)
            if not known:
                logging.getLogger(__name__).info("Preserving user-owned SenseVoice plugin at %s", target)
                return
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            data = (source / name).read_bytes()
            if not (target / name).is_file() or (target / name).read_bytes() != data:
                durable_write_bytes(target / name, data)
