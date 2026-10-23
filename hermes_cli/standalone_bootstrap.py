"""Local product bootstrap for the standalone Hermes CLI/Desktop entrypoint.

Product extras select shipped capabilities, never the SaaS host lifecycle.
"""
from __future__ import annotations

import hashlib
import logging
import tempfile
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
    from pm.filesystem import lock_fd
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
        if not target.exists():
            return  # Fresh installs use the immutable bundled backend.
        # Legacy full installs also carry unchanged documentation/requirements.
        # Unknown extra files or links are user customizations; keep their override.
        known_hashes = {
            **{name: {digest, hashlib.sha256((source / name).read_bytes()).hexdigest()}
               for name, digest in _LEGACY_STOCK.items()},
            "README.md": {"28beb3be46aedeec58130faba044937788f6ec6bcd17c5cf8550661d7fcebeaf"},
            "requirements.txt": {"393b307853f7c7f8a6e23c278bd5c9ac40220c624f475456ff0531521cf6b519"},
        }
        # Stock0.0.9 was copied into the home but never admitted for discovery.
        known_hashes["provider.py"].add("27c6b29aac6f1e3c88ae3bca459701880bb2c355105a12cdca08f2af5bd40a7f")
        known_hashes["plugin.yaml"].add("6f9b9b32023ed14eefaf81203b4849240a308656283025c50c594aba88c4e671")
        known = all((target / name).is_file() for name in names)
        for child in target.iterdir():
            if child.is_symlink() or getattr(child, "is_junction", lambda: False)():
                known = False
            elif child.name == "__pycache__" and child.is_dir():
                known &= all(p.is_file() and not p.is_symlink() and p.suffix == ".pyc"
                             for p in child.iterdir())
            else:
                known &= (child.is_file() and child.name in known_hashes
                          and hashlib.sha256(child.read_bytes()).hexdigest() in known_hashes[child.name])
        if not known:
            logging.getLogger(__name__).info("Preserving user-owned SenseVoice plugin at %s", target)
            return
        # Retain exact old bytes outside discovery, never rewrite activation or
        # user settings. A unique owned directory prevents replacement on retry.
        retained = Path(tempfile.mkdtemp(prefix=".sensevoice-stock-", dir=home))
        try:
            target.rename(retained / "sensevoice_stt")
        except BaseException:
            retained.rmdir()
            raise
