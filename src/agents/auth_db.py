from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from runtime_paths import sqlite_db_path

_LOGGER = logging.getLogger(__name__)
_SQLITE3_HEADER = b"SQLite format 3\x00"


def auth_db_path() -> Path:
    override = os.environ.get("SEMANTIER_AUTH_DB_PATH")
    if override:
        path = Path(override).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    return sqlite_db_path("auth.db")


def _is_valid_sqlite_file(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        if path.stat().st_size == 0:
            return True
        with path.open("rb") as handle:
            return handle.read(len(_SQLITE3_HEADER)) == _SQLITE3_HEADER
    except OSError:
        return False


def _quarantine_invalid_auth_db(path: Path) -> Path | None:
    if _is_valid_sqlite_file(path):
        return None
    suffix_index = 1
    while True:
        candidate = path.with_name(f"{path.name}.corrupt.{suffix_index}")
        if not candidate.exists():
            break
        suffix_index += 1
    path.replace(candidate)
    _LOGGER.warning("Quarantined invalid auth DB at %s to %s", path, candidate)
    return candidate


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(auth_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def ensure_auth_db(*, json_loader: Callable[[str], Any] | None = None) -> None:
    path = auth_db_path()
    _quarantine_invalid_auth_db(path)

    try:
        _initialize_auth_db()
    except sqlite3.DatabaseError:
        quarantined = _quarantine_invalid_auth_db(path)
        if quarantined is None:
            raise
        _initialize_auth_db()

    if json_loader is not None:
        _import_legacy_json_if_needed(json_loader=json_loader)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _initialize_auth_db() -> None:
    conn = _connect()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS organizations (
                organization_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS organization_events (
                event_id TEXT PRIMARY KEY,
                organization_id TEXT,
                created_at TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_organization_events_org_created
            ON organization_events(organization_id, created_at);

            CREATE TABLE IF NOT EXISTS gateway_correlations (
                correlation_key TEXT PRIMARY KEY,
                platform TEXT,
                owner_user_id TEXT,
                status TEXT,
                updated_at TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_gateway_correlations_platform
            ON gateway_correlations(platform, status, owner_user_id);

            CREATE TABLE IF NOT EXISTS weixin_runtime_accounts (
                account_id TEXT PRIMARY KEY,
                owner_user_id TEXT,
                owner_workspace_id TEXT,
                external_user_id TEXT,
                runtime_session_state TEXT,
                runtime_session_updated_at TEXT,
                saved_at TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_weixin_runtime_accounts_owner
            ON weixin_runtime_accounts(owner_user_id, owner_workspace_id);

            CREATE TABLE IF NOT EXISTS weixin_sync_state (
                account_id TEXT PRIMARY KEY,
                get_updates_buf TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS weixin_context_tokens (
                account_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                context_token TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(account_id, user_id)
            );

            CREATE TABLE IF NOT EXISTS gateway_pairing_pending (
                scope TEXT NOT NULL,
                platform TEXT NOT NULL,
                code TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_name TEXT,
                created_at REAL NOT NULL,
                PRIMARY KEY(scope, platform, code)
            );
            CREATE INDEX IF NOT EXISTS idx_gateway_pairing_pending_scope_platform
            ON gateway_pairing_pending(scope, platform);

            CREATE TABLE IF NOT EXISTS gateway_pairing_approved (
                scope TEXT NOT NULL,
                platform TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_name TEXT,
                approved_at REAL NOT NULL,
                PRIMARY KEY(scope, platform, user_id)
            );
            CREATE INDEX IF NOT EXISTS idx_gateway_pairing_approved_scope_platform
            ON gateway_pairing_approved(scope, platform);

            CREATE TABLE IF NOT EXISTS gateway_pairing_limits (
                scope TEXT NOT NULL,
                key TEXT NOT NULL,
                value REAL NOT NULL,
                PRIMARY KEY(scope, key)
            );

            CREATE TABLE IF NOT EXISTS weixin_login_states (
                state TEXT PRIMARY KEY,
                status TEXT,
                created_at REAL,
                updated_at REAL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feishu_link_states (
                state TEXT PRIMARY KEY,
                status TEXT,
                created_at REAL,
                updated_at REAL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feishu_bot_configs (
                owner_user_id TEXT PRIMARY KEY,
                owner_workspace_id TEXT,
                app_id TEXT,
                domain TEXT,
                saved_at TEXT,
                updated_at TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_feishu_bot_configs_workspace
            ON feishu_bot_configs(owner_workspace_id);

            CREATE TABLE IF NOT EXISTS user_settings (
                user_id TEXT PRIMARY KEY,
                settings_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS workspace_configs (
                workspace_id TEXT PRIMARY KEY,
                config_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def _count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
    return int(row["count"]) if row is not None else 0


def _import_legacy_json_if_needed(*, json_loader: Callable[[str], Any]) -> None:
    conn = _connect()
    try:
        table_specs = (
            ("users", "dict", json_loader("users")),
            ("organizations", "dict", json_loader("organizations")),
            ("organization_events", "list", json_loader("organization_events")),
            ("gateway_correlations", "list", json_loader("gateway_correlations")),
            ("weixin_login_states", "dict", json_loader("weixin_login_states")),
            ("feishu_link_states", "dict", json_loader("feishu_link_states")),
        )
        for table, shape, payload in table_specs:
            if _count(conn, table) > 0:
                continue
            if payload is None:
                continue
            if shape == "dict" and isinstance(payload, dict):
                if table == "users":
                    rows = [
                        (
                            str(key),
                            json.dumps(value, ensure_ascii=False, sort_keys=True),
                            str(value.get("updated_at") or ""),
                        )
                        for key, value in payload.items()
                        if isinstance(value, dict)
                    ]
                    conn.executemany(
                        "INSERT OR REPLACE INTO users(user_id, payload_json, updated_at) VALUES (?, ?, ?)",
                        rows,
                    )
                elif table == "organizations":
                    rows = [
                        (
                            str(key),
                            json.dumps(value, ensure_ascii=False, sort_keys=True),
                            str(value.get("updated_at") or ""),
                        )
                        for key, value in payload.items()
                        if isinstance(value, dict)
                    ]
                    conn.executemany(
                        "INSERT OR REPLACE INTO organizations(organization_id, payload_json, updated_at) VALUES (?, ?, ?)",
                        rows,
                    )
                elif table == "weixin_login_states":
                    rows = [
                        (
                            str(key),
                            str(value.get("status") or ""),
                            float(value.get("created_at") or 0.0),
                            float(value.get("updated_at") or 0.0),
                            json.dumps(value, ensure_ascii=False, sort_keys=True),
                        )
                        for key, value in payload.items()
                        if isinstance(value, dict)
                    ]
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO weixin_login_states
                        (state, status, created_at, updated_at, payload_json)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        rows,
                    )
                elif table == "feishu_link_states":
                    rows = [
                        (
                            str(key),
                            str(value.get("status") or ""),
                            float(value.get("created_at") or 0.0),
                            float(value.get("updated_at") or 0.0),
                            json.dumps(value, ensure_ascii=False, sort_keys=True),
                        )
                        for key, value in payload.items()
                        if isinstance(value, dict)
                    ]
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO feishu_link_states
                        (state, status, created_at, updated_at, payload_json)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        rows,
                    )
            elif shape == "list" and isinstance(payload, list):
                if table == "organization_events":
                    rows = [
                        (
                            str(value.get("event_id") or ""),
                            str(value.get("organization_id") or ""),
                            str(value.get("created_at") or ""),
                            json.dumps(value, ensure_ascii=False, sort_keys=True),
                        )
                        for value in payload
                        if isinstance(value, dict)
                    ]
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO organization_events
                        (event_id, organization_id, created_at, payload_json)
                        VALUES (?, ?, ?, ?)
                        """,
                        rows,
                    )
                elif table == "gateway_correlations":
                    rows = []
                    for idx, value in enumerate(payload):
                        if not isinstance(value, dict):
                            continue
                        corr_key = (
                            str(value.get("platform") or ""),
                            str(value.get("owner_user_id") or ""),
                            str(value.get("linked_account_id") or ""),
                            str(value.get("linked_external_user_id") or ""),
                            str(idx),
                        )
                        rows.append(
                            (
                                "|".join(corr_key),
                                str(value.get("platform") or ""),
                                str(value.get("owner_user_id") or ""),
                                str(value.get("status") or ""),
                                str(value.get("updated_at") or ""),
                                json.dumps(value, ensure_ascii=False, sort_keys=True),
                            )
                        )
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO gateway_correlations
                        (correlation_key, platform, owner_user_id, status, updated_at, payload_json)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        rows,
                    )
        conn.commit()
    finally:
        conn.close()


def load_users() -> dict[str, dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute("SELECT user_id, payload_json FROM users").fetchall()
        return {
            str(row["user_id"]): json.loads(str(row["payload_json"]))
            for row in rows
        }
    finally:
        conn.close()


def save_users(users: dict[str, Any]) -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM users")
        conn.executemany(
            "INSERT INTO users(user_id, payload_json, updated_at) VALUES (?, ?, ?)",
            [
                (
                    str(key),
                    json.dumps(value, ensure_ascii=False, sort_keys=True),
                    str(value.get("updated_at") or ""),
                )
                for key, value in users.items()
                if isinstance(value, dict)
            ],
        )
        conn.commit()
    finally:
        conn.close()


def load_organizations() -> dict[str, dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute("SELECT organization_id, payload_json FROM organizations").fetchall()
        return {
            str(row["organization_id"]): json.loads(str(row["payload_json"]))
            for row in rows
        }
    finally:
        conn.close()


def save_organizations(organizations: dict[str, Any]) -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM organizations")
        conn.executemany(
            "INSERT INTO organizations(organization_id, payload_json, updated_at) VALUES (?, ?, ?)",
            [
                (
                    str(key),
                    json.dumps(value, ensure_ascii=False, sort_keys=True),
                    str(value.get("updated_at") or ""),
                )
                for key, value in organizations.items()
                if isinstance(value, dict)
            ],
        )
        conn.commit()
    finally:
        conn.close()


def load_organization_events() -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT payload_json FROM organization_events ORDER BY created_at, event_id"
        ).fetchall()
        return [json.loads(str(row["payload_json"])) for row in rows]
    finally:
        conn.close()


def save_organization_events(events: list[dict[str, Any]]) -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM organization_events")
        conn.executemany(
            """
            INSERT INTO organization_events(event_id, organization_id, created_at, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    str(value.get("event_id") or ""),
                    str(value.get("organization_id") or ""),
                    str(value.get("created_at") or ""),
                    json.dumps(value, ensure_ascii=False, sort_keys=True),
                )
                for value in events
                if isinstance(value, dict)
            ],
        )
        conn.commit()
    finally:
        conn.close()


def load_gateway_correlations() -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT payload_json FROM gateway_correlations ORDER BY platform, owner_user_id, correlation_key"
        ).fetchall()
        return [json.loads(str(row["payload_json"])) for row in rows]
    finally:
        conn.close()


def save_gateway_correlations(correlations: list[dict[str, Any]]) -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM gateway_correlations")
        rows = []
        for idx, value in enumerate(correlations):
            if not isinstance(value, dict):
                continue
            corr_key = (
                str(value.get("platform") or ""),
                str(value.get("owner_user_id") or ""),
                str(value.get("linked_account_id") or ""),
                str(value.get("linked_external_user_id") or ""),
                str(idx),
            )
            rows.append(
                (
                    "|".join(corr_key),
                    str(value.get("platform") or ""),
                    str(value.get("owner_user_id") or ""),
                    str(value.get("status") or ""),
                    str(value.get("updated_at") or ""),
                    json.dumps(value, ensure_ascii=False, sort_keys=True),
                )
            )
        conn.executemany(
            """
            INSERT INTO gateway_correlations
            (correlation_key, platform, owner_user_id, status, updated_at, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def _weixin_runtime_account_row(value: dict[str, Any]) -> tuple[str, str, str, str, str, str, str, str] | None:
    account_id = str(value.get("account_id") or "").strip()
    if not account_id:
        return None
    saved_at = str(value.get("saved_at") or "").strip() or _utc_now_iso()
    return (
        account_id,
        str(value.get("owner_user_id") or ""),
        str(value.get("owner_workspace_id") or value.get("workspace_id") or ""),
        str(value.get("external_user_id") or value.get("user_id") or ""),
        str(value.get("runtime_session_state") or ""),
        str(value.get("runtime_session_updated_at") or ""),
        saved_at,
        json.dumps({**value, "saved_at": saved_at}, ensure_ascii=False, sort_keys=True),
    )


def _decode_weixin_runtime_account_payload(payload_json: str, *, account_id: str) -> dict[str, Any]:
    payload = json.loads(payload_json)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Weixin runtime account payload must be a JSON object: account_id={account_id}"
        )
    return payload


def load_weixin_runtime_accounts() -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT account_id, payload_json FROM weixin_runtime_accounts ORDER BY owner_workspace_id, account_id"
        ).fetchall()
        return [
            _decode_weixin_runtime_account_payload(
                str(row["payload_json"]),
                account_id=str(row["account_id"] or ""),
            )
            for row in rows
        ]
    finally:
        conn.close()


def get_weixin_runtime_account(account_id: str) -> dict[str, Any] | None:
    normalized = account_id.strip()
    if not normalized:
        return None
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT payload_json FROM weixin_runtime_accounts WHERE account_id=?",
            (normalized,),
        ).fetchone()
        return (
            _decode_weixin_runtime_account_payload(
                str(row["payload_json"]),
                account_id=normalized,
            )
            if row is not None
            else None
        )
    finally:
        conn.close()


def save_weixin_runtime_account(account: dict[str, Any]) -> None:
    row = _weixin_runtime_account_row(account)
    if row is None:
        raise ValueError("weixin runtime account requires account_id")
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO weixin_runtime_accounts
            (account_id, owner_user_id, owner_workspace_id, external_user_id,
             runtime_session_state, runtime_session_updated_at, saved_at, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
        conn.commit()
    finally:
        conn.close()


def update_weixin_runtime_account_state(
    account_id: str,
    *,
    runtime_session_state: str,
    runtime_session_updated_at: str,
    runtime_session_error: str | None = None,
) -> dict[str, Any] | None:
    account = get_weixin_runtime_account(account_id)
    if account is None:
        return None
    account["runtime_session_state"] = runtime_session_state
    account["runtime_session_updated_at"] = runtime_session_updated_at
    if runtime_session_error:
        account["runtime_session_error"] = runtime_session_error
    else:
        account.pop("runtime_session_error", None)
    save_weixin_runtime_account(account)
    return account


def _feishu_bot_config_row(value: dict[str, Any]) -> tuple[str, str, str, str, str, str, str] | None:
    owner_user_id = str(
        value.get("owner_user_id") or value.get("manager_user_id") or ""
    ).strip()
    if not owner_user_id:
        return None
    saved_at = str(value.get("saved_at") or "").strip() or _utc_now_iso()
    updated_at = str(value.get("updated_at") or "").strip() or saved_at
    payload = {
        **value,
        "manager_user_id": str(value.get("manager_user_id") or owner_user_id),
        "owner_user_id": owner_user_id,
        "saved_at": saved_at,
        "updated_at": updated_at,
    }
    return (
        owner_user_id,
        str(value.get("owner_workspace_id") or value.get("workspace_id") or ""),
        str(value.get("app_id") or ""),
        str(value.get("domain") or ""),
        saved_at,
        updated_at,
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
    )


def _decode_feishu_bot_config_payload(payload_json: str, *, owner_user_id: str) -> dict[str, Any]:
    payload = json.loads(payload_json)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Feishu bot config payload must be a JSON object: owner_user_id={owner_user_id}"
        )
    return payload


def get_feishu_bot_config_for_user(owner_user_id: str) -> dict[str, Any] | None:
    normalized = owner_user_id.strip()
    if not normalized:
        return None
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT payload_json FROM feishu_bot_configs WHERE owner_user_id=?",
            (normalized,),
        ).fetchone()
        return (
            _decode_feishu_bot_config_payload(
                str(row["payload_json"]),
                owner_user_id=normalized,
            )
            if row is not None
            else None
        )
    finally:
        conn.close()


def get_feishu_bot_config_for_workspace(owner_workspace_id: str) -> dict[str, Any] | None:
    normalized = owner_workspace_id.strip()
    if not normalized:
        return None
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT owner_user_id, payload_json FROM feishu_bot_configs
            WHERE owner_workspace_id=?
            ORDER BY updated_at DESC, owner_user_id
            LIMIT 1
            """,
            (normalized,),
        ).fetchone()
        return (
            _decode_feishu_bot_config_payload(
                str(row["payload_json"]),
                owner_user_id=str(row["owner_user_id"] or ""),
            )
            if row is not None
            else None
        )
    finally:
        conn.close()


def list_feishu_bot_configs() -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT owner_user_id, payload_json FROM feishu_bot_configs
            ORDER BY updated_at DESC, owner_user_id
            """
        ).fetchall()
        configs: list[dict[str, Any]] = []
        for row in rows:
            configs.append(
                _decode_feishu_bot_config_payload(
                    str(row["payload_json"]),
                    owner_user_id=str(row["owner_user_id"] or ""),
                )
            )
        return configs
    finally:
        conn.close()


def save_feishu_bot_config(config: dict[str, Any]) -> None:
    row = _feishu_bot_config_row(config)
    if row is None:
        raise ValueError("feishu bot config requires owner_user_id")
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO feishu_bot_configs
            (owner_user_id, owner_workspace_id, app_id, domain, saved_at, updated_at, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
        conn.commit()
    finally:
        conn.close()


def delete_feishu_bot_config_for_user(owner_user_id: str) -> None:
    normalized = owner_user_id.strip()
    if not normalized:
        return
    conn = _connect()
    try:
        conn.execute(
            "DELETE FROM feishu_bot_configs WHERE owner_user_id=?",
            (normalized,),
        )
        conn.commit()
    finally:
        conn.close()


def load_weixin_login_states() -> dict[str, dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute("SELECT state, payload_json FROM weixin_login_states").fetchall()
        return {str(row["state"]): json.loads(str(row["payload_json"])) for row in rows}
    finally:
        conn.close()


def get_user_settings_for_user(user_id: str) -> dict[str, Any]:
    normalized = user_id.strip()
    if not normalized:
        return {}
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT settings_json FROM user_settings WHERE user_id=?",
            (normalized,),
        ).fetchone()
        if row is None:
            return {}
        payload = json.loads(str(row["settings_json"]))
        return payload if isinstance(payload, dict) else {}
    finally:
        conn.close()


def upsert_user_settings_for_user(
    user_id: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    normalized = user_id.strip()
    if not normalized:
        raise ValueError("user_id required")
    current = get_user_settings_for_user(normalized)
    next_settings = {**current, **updates}
    updated_at = _utc_now_iso()
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO user_settings(user_id, settings_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                settings_json=excluded.settings_json,
                updated_at=excluded.updated_at
            """,
            (
                normalized,
                json.dumps(next_settings, ensure_ascii=False, sort_keys=True),
                updated_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return next_settings


def get_workspace_config(workspace_id: str) -> dict[str, Any]:
    normalized = str(workspace_id or "").strip()
    if not normalized:
        return {}
    conn = _connect()
    try:
        if not _table_exists(conn, "workspace_configs"):
            return {}
        row = conn.execute(
            "SELECT config_json FROM workspace_configs WHERE workspace_id=?",
            (normalized,),
        ).fetchone()
        if row is None:
            return {}
        payload = json.loads(str(row["config_json"]))
        return payload if isinstance(payload, dict) else {}
    finally:
        conn.close()


def save_workspace_config(
    workspace_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    normalized = str(workspace_id or "").strip()
    if not normalized:
        raise ValueError("workspace_id required")
    if not isinstance(config, dict):
        raise ValueError("workspace config must be a JSON object")
    _initialize_auth_db()
    updated_at = _utc_now_iso()
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO workspace_configs(workspace_id, config_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(workspace_id) DO UPDATE SET
                config_json=excluded.config_json,
                updated_at=excluded.updated_at
            """,
            (
                normalized,
                json.dumps(config, ensure_ascii=False, sort_keys=True),
                updated_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return config


def save_weixin_login_states(states: dict[str, Any]) -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM weixin_login_states")
        conn.executemany(
            """
            INSERT INTO weixin_login_states(state, status, created_at, updated_at, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    str(key),
                    str(value.get("status") or ""),
                    float(value.get("created_at") or 0.0),
                    float(value.get("updated_at") or 0.0),
                    json.dumps(value, ensure_ascii=False, sort_keys=True),
                )
                for key, value in states.items()
                if isinstance(value, dict)
            ],
        )
        conn.commit()
    finally:
        conn.close()


def load_feishu_link_states() -> dict[str, dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute("SELECT state, payload_json FROM feishu_link_states").fetchall()
        return {str(row["state"]): json.loads(str(row["payload_json"])) for row in rows}
    finally:
        conn.close()


def save_feishu_link_states(states: dict[str, Any]) -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM feishu_link_states")
        conn.executemany(
            """
            INSERT INTO feishu_link_states(state, status, created_at, updated_at, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    str(key),
                    str(value.get("status") or ""),
                    float(value.get("created_at") or 0.0),
                    float(value.get("updated_at") or 0.0),
                    json.dumps(value, ensure_ascii=False, sort_keys=True),
                )
                for key, value in states.items()
                if isinstance(value, dict)
            ],
        )
        conn.commit()
    finally:
        conn.close()
