"""Lightweight auth-db compatibility layer for repo-only test/runtime flows."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any


def _state_root() -> Path:
    raw = (
        os.getenv("SEMANTIER_LOCAL_STATE_DIR")
        or os.getenv("HERMES_HOME")
        or str(Path.home() / ".hermes-local")
    )
    root = Path(raw).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _auth_db_path() -> Path:
    raw = os.getenv("SEMANTIER_AUTH_DB_PATH")
    if raw:
        return Path(raw).expanduser().resolve()
    return _state_root() / "auth.db"


def _connect_auth_db() -> sqlite3.Connection:
    db_path = _auth_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
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
        """
    )
    return conn


def ensure_auth_db() -> Path:
    conn = _connect_auth_db()
    try:
        conn.commit()
    finally:
        conn.close()
    return _auth_db_path()


def load_weixin_runtime_accounts() -> list[dict[str, Any]]:
    conn = _connect_auth_db()
    try:
        rows = conn.execute(
            """
            SELECT payload_json
            FROM weixin_runtime_accounts
            ORDER BY COALESCE(saved_at, ''), account_id
            """
        ).fetchall()
    finally:
        conn.close()

    accounts: list[dict[str, Any]] = []
    for row in rows:
        payload = json.loads(str(row["payload_json"]))
        if isinstance(payload, dict):
            accounts.append(payload)
    return accounts


def save_weixin_runtime_account(account: dict[str, Any]) -> None:
    payload = dict(account)
    payload.setdefault("saved_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    account_id = str(payload.get("account_id") or "").strip()
    row = (
        account_id,
        str(payload.get("owner_user_id") or ""),
        str(payload.get("owner_workspace_id") or ""),
        str(payload.get("external_user_id") or payload.get("user_id") or ""),
        str(payload.get("runtime_session_state") or ""),
        str(payload.get("runtime_session_updated_at") or ""),
        str(payload.get("saved_at") or ""),
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
    )
    conn = _connect_auth_db()
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


def get_weixin_runtime_account(account_id: str) -> dict[str, Any] | None:
    normalized = str(account_id or "").strip()
    if not normalized:
        return None

    conn = _connect_auth_db()
    try:
        row = conn.execute(
            "SELECT payload_json FROM weixin_runtime_accounts WHERE account_id=?",
            (normalized,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None
    payload = json.loads(str(row["payload_json"]))
    return dict(payload) if isinstance(payload, dict) else None
