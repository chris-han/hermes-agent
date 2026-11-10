"""Read-only Chromium selection shared by browser launchers and settings."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pm

logger = logging.getLogger(__name__)


def system_browsers() -> list[dict[str, str]]:
    """Discover executables only; never inspect or copy a user's browser profile."""
    from hermes_cli.browser_connect import _BROWSERS, chromium_executable as find
    return [{"key": browser.key, "executable": binary}
            for browser in _BROWSERS if (binary := find(browser.key))]


def chromium_executable(*, allow_override: bool = True) -> str | None:
    """Explicit choice, installed built-in, then discovered system fallback."""
    override = os.environ.get("AGENT_BROWSER_EXECUTABLE_PATH") if allow_override else None
    if allow_override:
        from hermes_cli.config import load_config
        browser = load_config().get("browser", {})
        choice = browser.get("local_browser", "auto") if isinstance(browser, dict) else "auto"
        if choice not in {"", "auto", "builtin"}:
            from hermes_cli.browser_connect import chromium_executable as find
            explicit = find(choice)
            if not explicit:
                raise RuntimeError(f"Selected system browser '{choice}' is unavailable; choose another browser in Settings.")
            return explicit
    if override:
        return override
    installed = pm.installed_package("chromium")
    if installed and installed.binary is not None and Path(installed.binary).is_file():
        return str(installed.binary)
    if not allow_override:
        return None
    discovered = system_browsers()
    if discovered:
        logger.warning("Built-in Chromium unavailable; using system browser %s with an isolated profile", discovered[0]["key"])
        return discovered[0]["executable"]
    logger.warning("Built-in Chromium unavailable and no supported system browser found")
    return None
