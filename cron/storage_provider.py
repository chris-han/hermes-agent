"""Optional host-owned cron persistence; standalone Hermes retains its file store."""
from typing import Any

_provider: Any = None


def register_storage_provider(provider: Any) -> None:
    global _provider
    _provider = provider


def get_storage_provider(*,required: bool=False) -> Any:
    if required and _provider is None:
        from hermes_state_backend import postgres_enabled
        if postgres_enabled():
            raise RuntimeError('PostgreSQL cron storage requires a registered host provider; standalone file fallback is unavailable')
    return _provider
