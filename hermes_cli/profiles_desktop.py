"""Resolve a Desktop launch through the existing local profile lifecycle owner."""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
import os
import stat
from pathlib import Path

from hermes_cli import profiles
from hermes_constants import (
    get_default_hermes_root, named_profile_is_deleted,
    reset_profile_owner_root_override, set_profile_owner_root_override,
)

DESKTOP_FALLBACK = "default"

def _ensure_semantier_desktop_soul(home: Path) -> None:
    """Change only uncustomized upstream identity in the Semantier Desktop host."""
    from hermes_cli.default_soul import DEFAULT_SOUL_MD, is_legacy_template_soul
    soul = home / "SOUL.md"
    if soul.is_symlink() or getattr(soul, "is_junction", lambda: False)():
        return  # Linked identity may be deliberate user configuration.
    if soul.exists():
        if not soul.is_file():
            return
        try:
            existing = soul.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            return
        if existing.strip() != DEFAULT_SOUL_MD.strip() and not is_legacy_template_soul(existing):
            return
    from pm.paths import repo_root
    canonical = (repo_root() / "semantier-runtime/src/prompts/agents/SOUL.md").read_text(encoding="utf-8-sig")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=home,
                                     prefix=".semantier-soul-", delete=False) as stream:
        stream.write(canonical)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.chmod(temporary, 0o600)
        temporary.replace(soul)
    finally:
        temporary.unlink(missing_ok=True)


def _owner_role_path(owner: Path) -> Path:
    namespace = owner / 'profiles'
    marker = namespace / '.desktop-owner-root.json'
    if any(path.is_symlink() or profiles._junction_target(str(path)) for path in (namespace, marker)):
        raise ValueError('Desktop owner role escapes its owner')
    return marker


def _read_owner_role(owner: Path) -> dict | None:
    marker = _owner_role_path(owner)
    try:
        try:
            info = marker.stat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(info.st_mode) or info.st_size > 4096:
            raise ValueError('Invalid desktop owner role file')
        value = json.loads(marker.read_text(encoding='utf-8-sig'))
        if (not isinstance(value, dict) or set(value) != {'schema', 'role'}
                or type(value['schema']) is not int or value['schema'] != 1
                or value['role'] not in {'runtime_owner', 'user_profile'}):
            raise ValueError('Invalid desktop owner role')
        return value
    except (OSError, ValueError) as error:
        raise RuntimeError('Desktop owner role requires recovery') from error


def _write_owner_role(owner: Path, role: str) -> None:
    marker = _owner_role_path(owner)
    marker.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.desktop-owner-root-', dir=marker.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump({'schema': 1, 'role': role}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
    finally:
        Path(temporary).unlink(missing_ok=True)


def desktop_owner_is_runtime_only(owner: Path) -> bool:
    role = _read_owner_role(owner)
    return bool(role and role['role'] == 'runtime_owner')


def default_profile_is_discoverable(owner: Path) -> bool:
    # Desktop's owner directory is infrastructure, not a second empty profile.
    # A real default identity (including old chats/credentials) remains visible.
    if desktop_owner_is_runtime_only(owner):
        return False
    return owner.is_dir()


def claim_default_profile(owner: Path) -> None:
    """Explicit profile selection can claim the host root without changing its data."""
    with profiles._profile_lifecycle_lock(owner):
        if desktop_owner_is_runtime_only(owner):
            _write_owner_role(owner, 'user_profile')


def _validate_profile(owner: Path, name: str) -> bool:
    home = profiles.get_profile_dir(name)
    if name == "default" and (not profiles.named_profile_has_identity(home)
                              or desktop_owner_is_runtime_only(owner)):
        return False
    if home.is_symlink() or profiles._junction_target(str(home)):
        raise ValueError("Profile path escapes its owner")
    namespace = owner if name == "default" else owner / "profiles"
    if not home.resolve().is_relative_to(namespace.resolve()):
        raise ValueError("Profile path escapes its owner")
    try:
        home.stat()
    except FileNotFoundError:
        # Explicit recovery never recreates a removed named identity.
        return False
    if name != "default" and named_profile_is_deleted(home):
        raise ValueError("Profile is tombstoned; explicit recovery required")
    identity_exists = profiles.profile_exists(name)
    if not home.is_dir() or not identity_exists:
        raise ValueError("Profile has no valid local identity")
    # Validate existing bytes, never load defaults and write them back over corruption.
    from hermes_cli.config import require_readable_config_before_write
    require_readable_config_before_write(home / "config.yaml")
    return True

def resolve_desktop_startup_profile(
    owner_root: Path, requested: str | None = None, *, explicit: bool = False,
) -> dict[str, str | bool]:
    owner = Path(owner_root).expanduser().resolve()
    token = set_profile_owner_root_override(owner)
    try:
        if get_default_hermes_root().resolve() != owner:
            raise PermissionError("Profile owner conflicts with governed host binding")
        namespace = owner / "profiles"
        if (namespace.is_symlink() or profiles._junction_target(str(namespace))
                or not namespace.resolve().is_relative_to(owner)):
            raise ValueError("Profile namespace escapes its owner")
        with profiles._profile_lifecycle_lock(owner):
            # Validate old metadata before promoting the former Desktop home.
            _read_owner_role(owner)
            if explicit and requested and requested != "default":
                if not _validate_profile(owner, requested):
                    raise ValueError(f"Explicit profile '{requested}' is missing")
                return {"profile": requested, "recovered": False}
            from hermes_cli.profiles_desktop_migration import migrate_local_to_default
            migrated = migrate_local_to_default(owner)
            from hermes_cli.config import require_readable_config_before_write
            require_readable_config_before_write(owner / "config.yaml")
            owner.mkdir(parents=True, exist_ok=True)
            _ensure_semantier_desktop_soul(owner)
            _write_owner_role(owner, "user_profile")
            profiles.set_active_profile("default")
            return {"profile": "default", "recovered": migrated or bool(requested and requested != "default")}

    finally:
        reset_profile_owner_root_override(token)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-root", required=True, type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--explicit", action="store_true")
    args = parser.parse_args()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = resolve_desktop_startup_profile(args.owner_root, args.profile, explicit=args.explicit)
    except (OSError, ValueError, RuntimeError) as error:
        # Config parser tracebacks may include secret-bearing source lines. The
        # private caller needs the category, never raw configuration or a key.
        result = {"error": f"Local profile unavailable ({type(error).__name__}); check its identity, permissions or configuration."}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
