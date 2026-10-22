from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


class SemantierCronHomeDiscoveryError(RuntimeError):
    pass


def _iter_workspace_hermes_homes() -> list[Path]:
    try:
        from agents.workspace_session_logs import iter_workspace_hermes_homes

        from runtime_paths import _workspaces_root

        return list(iter_workspace_hermes_homes(_workspaces_root()))
    except (ImportError, AttributeError):
        module_path = Path(__file__).with_name("workspace_session_logs.py")
        spec = importlib.util.spec_from_file_location(
            "_semantier_workspace_session_logs_for_cron_provider",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise SemantierCronHomeDiscoveryError(
                f"Semantier cron workspace discovery failed: cannot load {module_path}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        iterator = getattr(module, "iter_workspace_hermes_homes", None)
        if not callable(iterator):
            raise SemantierCronHomeDiscoveryError(
                "Semantier cron workspace discovery failed: workspace iterator is unavailable"
            )
        from runtime_paths import _workspaces_root

        return list(iterator(_workspaces_root()))


def semantier_cron_homes() -> list[Path]:
    """Return governed workspace Hermes homes for Semantier-owned cron ticks."""
    from storage.hosted_postgres import enabled

    eligible = _postgres_cron_homes() if enabled() else None
    return [
        Path(home).expanduser().resolve()
        for home in _iter_workspace_hermes_homes()
        if eligible is None or Path(home).expanduser().resolve() in eligible
    ]


def _postgres_cron_homes() -> set[Path]:
    """Read allocation authority; workspace files never allocate cron profiles."""
    import psycopg
    from hermes_state_postgresql import SessionScope
    from agents.workspace_storage_provisioning import registered_workspace_scope
    from runtime_paths import _workspaces_root

    mappings = json.loads(os.environ.get("HERMES_STATE_SCOPE_MAP", "{}"))
    if not isinstance(mappings, dict):
        raise ValueError("Invalid governed PostgreSQL session home mapping")
    scopes = {}
    for home, value in mappings.items():
        path = Path(home).expanduser().resolve()
        if (str(path) != home or not isinstance(value, dict)
                or set(value) != {"tenant_id", "workspace_id", "source_store_id"}):
            raise ValueError("Invalid governed PostgreSQL session home mapping")
        scopes[path] = SessionScope(**value)

    dsn = os.environ.get("HERMES_DATABASE_URL")
    if not dsn:
        raise SemantierCronHomeDiscoveryError(
            "HERMES_DATABASE_URL required for PostgreSQL cron discovery"
        )
    with psycopg.connect(dsn) as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        rows = connection.execute("""SELECT b.user_id,b.workspace_id,b.tenant_id,
            b.source_store_id,b.schema_name,b.hermes_home,u.payload_json
            FROM semantier_identity.workspace_runtime_bindings b
            LEFT JOIN semantier_identity.users u ON u.user_id=b.user_id""").fetchall()
    root = _workspaces_root().expanduser().resolve()
    for row in rows:
        scope = registered_workspace_scope(row, root)
        home = Path(row[5])
        if home in scopes and scopes[home] != scope:
            raise ValueError("Governed PostgreSQL namespace binding conflict")
        scopes[home] = scope
    return {home for home in scopes if home.parent == root}


def register_semantier_cron_home_provider() -> None:
    """Bind the Semantier composition to its authoritative Core cron store."""
    from cron.storage_provider import register_storage_provider
    from agents.cron_storage_provider import SemantierCronStorageProvider

    register_storage_provider(SemantierCronStorageProvider())


def ensure_semantier_cron_storage_provider() -> None:
    """Bind direct Core API entrypoints without replacing another host's provider."""
    from cron.storage_provider import get_storage_provider
    from agents.cron_storage_provider import SemantierCronStorageProvider

    provider = get_storage_provider()
    if provider is None:
        register_semantier_cron_home_provider()
    elif not isinstance(provider, SemantierCronStorageProvider):
        raise RuntimeError("Semantier cron storage binding is unavailable: another provider is registered")
