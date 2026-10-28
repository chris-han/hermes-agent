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
    reset_profile_owner_root_override, set_profile_owner_root_override, profile_tombstone_path,
)

DESKTOP_FALLBACK = "semantier-local"
_RETIRED_INTERNAL_PROFILE = "tender-runtime-expert"


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
    return owner.is_dir() and (
        profiles.named_profile_has_identity(owner)
        or not profiles.named_profile_is_live(owner / 'profiles' / DESKTOP_FALLBACK)
    )


def claim_default_profile(owner: Path) -> None:
    """Explicit profile selection can claim the host root without changing its data."""
    with profiles._profile_lifecycle_lock(owner):
        if desktop_owner_is_runtime_only(owner):
            _write_owner_role(owner, 'user_profile')


def _validate_profile(owner: Path, name: str, *, retiring: bool = False) -> bool:
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
        # A fully removed ordinary reference can resolve another identity;
        # its retirement marker remains intact. ensure_profile still refuses
        # to recreate a retired fallback, and explicit requests fail below.
        return False
    if name != "default" and named_profile_is_deleted(home) and not retiring:
        raise ValueError("Profile is tombstoned; explicit recovery required")
    identity_exists = profiles.named_profile_has_identity(home) if retiring else profiles.profile_exists(name)
    if not home.is_dir() or not identity_exists:
        raise ValueError("Profile has no valid local identity")
    # Validate existing bytes, never load defaults and write them back over corruption.
    from hermes_cli.config import require_readable_config_before_write
    require_readable_config_before_write(home / "config.yaml")
    return True



def _hermes_process_candidate(info: dict) -> bool:
    argv = info.get("cmdline") or []
    if not argv:
        name = (info.get("name") or "").lower()
        if name.startswith(("python", "hermes", "semantier")):
            raise RuntimeError("Cannot establish potential Hermes process ownership")
        return False
    direct_core = len(argv) > 1 and Path(argv[1]).name in {"run_agent.py", "cli.py"}
    return profiles._is_hermes_argv(argv) or direct_core


def _require_profile_unowned(home: Path) -> None:
    """Read-only retirement guard, including interactive CLI/TUI processes."""
    import psutil
    from gateway.status import _is_gateway_runtime_lock_active_strict

    if _is_gateway_runtime_lock_active_strict(home / "gateway.lock"):
        raise RuntimeError("Legacy internal profile is still owned by a running gateway")
    current = psutil.Process()
    username = current.username().casefold()
    own_chain = {current.pid, *(parent.pid for parent in current.parents())}
    for process in psutil.process_iter(["pid", "name", "username", "cmdline"]):
        try:
            if process.pid in own_chain or not _hermes_process_candidate(process.info):
                continue
            if process.info["username"] is None:
                raise RuntimeError("Cannot establish Hermes process owner")
            if process.info["username"].casefold() != username:
                continue
            argv = process.info["cmdline"] or []
            environment = process.environ()
            bound_home = next((v for k, v in environment.items() if k.upper() == "HERMES_HOME"), "")
            selected = (profiles._is_hermes_argv(argv) and
                        _RETIRED_INTERNAL_PROFILE in profiles._argv_profile_selectors(argv)) if argv else False
            if (bound_home and Path(bound_home).resolve() == home.resolve()) or selected:
                raise RuntimeError("Legacy internal profile is owned by a running process")
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except (psutil.AccessDenied, AttributeError) as error:
            raise RuntimeError("Cannot establish legacy profile process ownership") from error


def _retirement_marker(owner: Path) -> tuple[Path, Path | None]:
    """The existing tombstone also journals our atomic, resumable directory move."""
    marker = profile_tombstone_path(owner / "profiles" / _RETIRED_INTERNAL_PROFILE)
    for directory in (marker.parent, owner / ".desktop-retired-profiles"):
        if (directory.is_symlink() or profiles._junction_target(str(directory))
                or not directory.resolve().is_relative_to(owner)):
            raise ValueError("Profile retirement namespace escapes its owner")
    if marker.is_symlink() or profiles._junction_target(str(marker)):
        raise ValueError("Profile retirement marker escapes its owner")
    if not marker.exists():
        return marker, None
    try:
        record = json.loads(marker.read_text(encoding="utf-8-sig"))
        relative = Path(record["backup"])
        if record["kind"] != "desktop-internal-profile-retirement-v1" or relative.is_absolute():
            raise ValueError("Not a Desktop retirement marker")
        target = owner / relative
        retained = owner / ".desktop-retired-profiles"
        if (target.name != _RETIRED_INTERNAL_PROFILE or target.parent.parent != retained
                or target.is_symlink() or profiles._junction_target(str(target))
                or target.parent.is_symlink() or profiles._junction_target(str(target.parent))
                or not target.resolve().is_relative_to(retained.resolve())):
            raise ValueError("Invalid retained profile path")
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError("Profile is retired; explicit recovery required") from error
    return marker, target


def _retire_internal_desktop_profile(owner: Path, selected: str | None) -> bool:
    """Retain every old profile byte outside discovery; never purge or merge data."""
    from hermes_cli.config import atomic_config_write

    legacy = owner / "profiles" / _RETIRED_INTERNAL_PROFILE
    marker, target = _retirement_marker(owner)
    if not legacy.exists() and not legacy.is_symlink():
        if target is not None:
            if not target.is_dir():
                raise RuntimeError("Retained legacy profile backup is missing; explicit recovery required")
            if profiles.get_active_profile(owner) == _RETIRED_INTERNAL_PROFILE:
                profiles.ensure_profile(owner, DESKTOP_FALLBACK)
                profiles.set_active_profile(DESKTOP_FALLBACK)
        return False
    _validate_profile(owner, _RETIRED_INTERNAL_PROFILE, retiring=target is not None)
    _require_profile_unowned(legacy)
    if target is not None and target.exists():
        raise RuntimeError("Retained backup already exists; inspect the newly recreated internal profile")
    local = owner / "profiles" / DESKTOP_FALLBACK
    newly_created = not local.exists()
    if selected == _RETIRED_INTERNAL_PROFILE:
        # Transfer only this presentation preference, never SaaS plugin activation.
        # The existing locale owner decides whether the saved value is supported.
        previous = profiles._load_yaml_dict(legacy / "config.yaml") or {}
        display = previous.get("display")
        language = display.get("language") if isinstance(display, dict) else None
        profiles.ensure_profile(owner, DESKTOP_FALLBACK)
        _validate_profile(owner, DESKTOP_FALLBACK)
        if newly_created and isinstance(language, str) and language:
            atomic_config_write(local / "config.yaml", {"display": {"language": language}})
    if target is None:
        retained = owner / ".desktop-retired-profiles"
        retained.mkdir(exist_ok=True)
        batch = Path(tempfile.mkdtemp(prefix="tender-runtime-expert-", dir=retained))
        target = batch / _RETIRED_INTERNAL_PROFILE
        marker.parent.mkdir(exist_ok=True)
        record = {"kind": "desktop-internal-profile-retirement-v1", "backup": str(target.relative_to(owner))}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=marker.parent, delete=False) as stream:
            json.dump(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        try:
            temporary.replace(marker)
        finally:
            temporary.unlink(missing_ok=True)
    # A failed/interrupted move keeps the journal and original bytes. Restart
    # resumes that exact move rather than treating our own marker as a deletion.
    _require_profile_unowned(legacy)
    legacy.rename(target)
    if profiles.get_active_profile(owner) == _RETIRED_INTERNAL_PROFILE:
        profiles.ensure_profile(owner, DESKTOP_FALLBACK)
        profiles.set_active_profile(DESKTOP_FALLBACK)
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
        selected = requested
        if not selected:
            try:
                selected = (owner / "active_profile").read_text(encoding="utf-8-sig").strip() or None
            except FileNotFoundError:
                pass
        with profiles._profile_lifecycle_lock(owner):
            # Record known-empty provenance before host bootstrap adds config,
            # SOUL and empty databases. Never reclassify legacy root data.
            role = _read_owner_role(owner)
            if role is None and not profiles.named_profile_has_identity(owner):
                _write_owner_role(owner, 'runtime_owner')
            elif selected is None and role and role['role'] == 'user_profile':
                # The canonical sticky-default representation is an absent
                # active_profile file. Only a deliberate recorded claim can
                # distinguish it from fresh desktop infrastructure.
                selected = 'default'
            if selected == 'default' and explicit:
                claim_default_profile(owner)
            retired = _retire_internal_desktop_profile(owner, selected)
            if selected == _RETIRED_INTERNAL_PROFILE and (retired or named_profile_is_deleted(
                    owner / "profiles" / _RETIRED_INTERNAL_PROFILE)):
                selected = DESKTOP_FALLBACK
                explicit = False
            if selected:
                if _validate_profile(owner, selected):
                    return {"profile": selected, "recovered": retired}
                if explicit:
                    raise ValueError(f"Explicit profile '{selected}' is missing")
            # Same lifecycle owner handles both first launch and every restart/update.
            home = profiles.ensure_profile(owner, DESKTOP_FALLBACK)
            _validate_profile(owner, DESKTOP_FALLBACK)
            return {"profile": home.name, "recovered": selected != DESKTOP_FALLBACK or retired}
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
