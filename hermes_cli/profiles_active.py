"""Sticky profile selection helpers shared by profile lifecycle operations."""

import contextlib


def retarget_active_profile(old: str, new: str, message: str) -> None:
    """If the sticky profile is *old*, point it at *new* and print *message*. Never raises."""
    from hermes_cli import profiles

    with contextlib.suppress(Exception):
        if profiles.get_active_profile() == old:
            profiles.set_active_profile(new)
            print(message)
