from __future__ import annotations

from pathlib import Path
from typing import Callable

CronHomeProvider = Callable[[], list[Path]]

_cron_home_provider: CronHomeProvider | None = None


def replace_cron_home_provider(
    provider: CronHomeProvider | None,
) -> CronHomeProvider | None:
    """Replace the optional runtime-owned cron home provider."""
    global _cron_home_provider
    previous = _cron_home_provider
    _cron_home_provider = provider
    return previous


def get_cron_home_provider() -> CronHomeProvider | None:
    return _cron_home_provider
