"""Optional host-owned cron persistence; standalone Hermes retains its file store."""
from typing import Any

_provider: Any = None


def register_storage_provider(provider: Any) -> None:
    global _provider
    _provider = provider


def get_storage_provider() -> Any:
    return _provider
