"""Platform operations authority and explicit, non-resetting identity bootstrap."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from agents import auth_db

OPS_LOGIN = "semantier-ops"
OPS_USER_ID = "semantier-platform-ops"
OPS_BOOTSTRAP = "operations-admin.v1"


def is_operations_admin(user: dict[str, Any] | None) -> bool:
    return bool(user and user.get("platform_role") == "operations_admin"
                and not user.get("disabled") and user.get("status", "active") == "active")


def bootstrap_operations_admin(*, password: str) -> dict[str, str]:
    """Explicit operator action. Never reset/elevate an existing identity."""
    if not 8 <= len(password) <= 256 or password != password.strip():
        raise ValueError("Initial password must contain 8–256 characters without outer whitespace")
    from agents.gateway_identity import hash_local_password, ensure_workspace_paths

    auth_db.ensure_auth_db()
    with auth_db.identity_write_transaction():
        users = auth_db.load_users()
        matches = [u for u in users.values() if u.get("user_id") == OPS_USER_ID
                   or str(u.get("password_login_name", "")).lower() == OPS_LOGIN]
        if matches:
            if len(matches) != 1 or matches[0].get("user_id") != OPS_USER_ID or matches[0].get("ops_bootstrap") != OPS_BOOTSTRAP or str(matches[0].get("password_login_name", "")).lower() != OPS_LOGIN:
                raise ValueError("Operations bootstrap identity collision; no identity was elevated")
            user = matches[0]
            status = "preserved"
        else:
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            user = {
                "user_id": OPS_USER_ID, "workspace_id": OPS_USER_ID,
                "workspace_slug": "user-" + OPS_USER_ID, "name": OPS_LOGIN,
                "password_login_name": OPS_LOGIN, "password_hash": hash_local_password(password),
                "auth_provider": "password", "platform_role": "operations_admin",
                "ops_bootstrap": OPS_BOOTSTRAP, "profile_completed_at": now,
                "created_at": now, "updated_at": now,
            }
            if auth_db.is_postgresql():
                from agents.auth_db_postgres import provision_personal_signup
                provision_personal_signup(user)
            else:
                auth_db.insert_password_user(user)
            status = "created"
    ensure_workspace_paths(user["workspace_id"])
    return {"status": status, "login": OPS_LOGIN, "user_id": user["user_id"]}


def operations_snapshot() -> dict[str, Any]:
    """Current inventory only; do not manufacture historical funnel events."""
    users = list(auth_db.load_users().values())
    customers = [u for u in users if not u.get("platform_role") and not u.get("demo_account")]
    return {
        "scope": "platform", "generated_at": datetime.now(timezone.utc).isoformat(),
        "inventory": {
            "customer_accounts": len(customers),
            "profile_completed_accounts": sum(bool(u.get("profile_completed_at")) for u in customers),
            "excluded_operator_accounts": sum(bool(u.get("platform_role")) for u in users),
            "excluded_demo_accounts": sum(bool(u.get("demo_account")) and not u.get("platform_role") for u in users),
        },
        "inventory_basis": "current_identity_records_not_historical_conversion",
        "journey_coverage": [
            {"surface": surface, "status": "not_instrumented", "visitors": None, "conversions": None}
            for surface in ("landing", "workspace", "windows", "payment", "task_reuse")
        ],
    }
