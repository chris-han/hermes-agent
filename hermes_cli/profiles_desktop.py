"""Resolve a Desktop launch through the existing local profile lifecycle owner."""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from hermes_cli import profiles
from hermes_constants import (
    get_default_hermes_root, named_profile_is_deleted,
    reset_profile_owner_root_override, set_profile_owner_root_override,
)

DESKTOP_FALLBACK = "semantier-local"


def _validate_profile(owner: Path, name: str) -> bool:
    home = profiles.get_profile_dir(name)
    if home.is_symlink() or getattr(home, "is_junction", lambda: False)():
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
    if name != "default" and named_profile_is_deleted(home):
        raise ValueError("Profile is tombstoned; explicit recovery required")
    if not home.is_dir() or not profiles.profile_exists(name):
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
        if (namespace.is_symlink() or getattr(namespace, "is_junction", lambda: False)()
                or not namespace.resolve().is_relative_to(owner)):
            raise ValueError("Profile namespace escapes its owner")
        selected = requested
        if not selected:
            try:
                selected = (owner / "active_profile").read_text(encoding="utf-8").strip() or None
            except FileNotFoundError:
                pass
        if selected:
            if _validate_profile(owner, selected):
                return {"profile": selected, "recovered": False}
            if explicit:
                raise ValueError(f"Explicit profile '{selected}' is missing")
        # ensure_profile owns locking, tombstones, path/owner validation and creation policy.
        home = profiles.ensure_profile(owner, DESKTOP_FALLBACK)
        _validate_profile(owner, DESKTOP_FALLBACK)
        return {"profile": home.name, "recovered": selected != DESKTOP_FALLBACK}
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
