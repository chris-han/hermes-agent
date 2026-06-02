"""
DM Pairing System

Code-based approval flow for authorizing new users on messaging platforms.
Instead of static allowlists with user IDs, unknown users receive a one-time
pairing code that the bot owner approves via the CLI.

Security features (based on OWASP + NIST SP 800-63-4 guidance):
  - 8-char codes from 32-char unambiguous alphabet (no 0/O/1/I)
  - Cryptographic randomness via secrets.choice()
  - 1-hour code expiry
  - Max 3 pending codes per platform
  - Rate limiting: 1 request per user per 10 minutes
  - Lockout after 5 failed approval attempts (1 hour)
  - Codes are never logged to stdout

Storage: SQLite auth database. Pairing JSON files are not used for new state.
"""

import json
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_dir


# Unambiguous alphabet -- excludes 0/O, 1/I to prevent confusion
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
CODE_LENGTH = 8

# Timing constants
CODE_TTL_SECONDS = 3600             # Codes expire after 1 hour
RATE_LIMIT_SECONDS = 600            # 1 request per user per 10 minutes
LOCKOUT_SECONDS = 3600              # Lockout duration after too many failures

# Limits
MAX_PENDING_PER_PLATFORM = 3        # Max pending codes per platform
MAX_FAILED_ATTEMPTS = 5             # Failed approvals before lockout

_IMPORT_TIME_PAIRING_DIR = get_hermes_dir("platforms/pairing", "pairing")
# Backward-compatible module attribute used by older tests/callers that patch
# the pairing location directly. Runtime resolution should prefer the current
# HERMES_HOME unless this value has been explicitly monkeypatched away from the
# import-time default.
PAIRING_DIR = _IMPORT_TIME_PAIRING_DIR


def _resolve_pairing_dir() -> Path:
    current = PAIRING_DIR
    if current != _IMPORT_TIME_PAIRING_DIR:
        return current
    return get_hermes_dir("platforms/pairing", "pairing")


def _resolve_auth_db_path(auth_db_path: Path | str | None = None) -> Path:
    if auth_db_path is not None:
        return Path(auth_db_path).expanduser().resolve()
    explicit = os.environ.get("SEMANTIER_AUTH_DB_PATH", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    raise RuntimeError(
        "PairingStore requires an explicit auth DB path. "
        "Pass auth_db_path=... or set SEMANTIER_AUTH_DB_PATH."
    )


class PairingStore:
    """
    Manages pairing codes and approved user lists.

    Pairing rows are scoped by pairing_dir so tests and workspace-local stores
    stay isolated even when they share the same auth database.
    """

    def __init__(
        self,
        pairing_dir: Path | None = None,
        auth_db_path: Path | str | None = None,
    ):
        self._pairing_dir = Path(pairing_dir) if pairing_dir is not None else _resolve_pairing_dir()
        self._scope = str(self._pairing_dir.expanduser().resolve())
        self._db_path = _resolve_auth_db_path(auth_db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # Protects all read-modify-write cycles. The gateway runs multiple
        # platform adapters concurrently in threads sharing one PairingStore.
        self._lock = threading.RLock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
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
                """
            )

    def _pending_path(self, platform: str) -> Path:
        return self._pairing_dir / f"{platform}-pending.json"

    def _approved_path(self, platform: str) -> Path:
        return self._pairing_dir / f"{platform}-approved.json"

    def _rate_limit_path(self) -> Path:
        return self._pairing_dir / "_rate_limits.json"

    def _load_json(self, path: Path) -> dict:
        path = Path(path)
        name = path.name
        with self._connect() as conn:
            if name == "_rate_limits.json":
                rows = conn.execute(
                    "SELECT key, value FROM gateway_pairing_limits WHERE scope=?",
                    (self._scope,),
                ).fetchall()
                return {str(row["key"]): float(row["value"]) for row in rows}
            if name.endswith("-pending.json"):
                platform = name[: -len("-pending.json")]
                rows = conn.execute(
                    """
                    SELECT code, user_id, user_name, created_at
                    FROM gateway_pairing_pending
                    WHERE scope=? AND platform=?
                    """,
                    (self._scope, platform),
                ).fetchall()
                return {
                    str(row["code"]): {
                        "user_id": str(row["user_id"]),
                        "user_name": str(row["user_name"] or ""),
                        "created_at": float(row["created_at"]),
                    }
                    for row in rows
                }
            if name.endswith("-approved.json"):
                platform = name[: -len("-approved.json")]
                rows = conn.execute(
                    """
                    SELECT user_id, user_name, approved_at
                    FROM gateway_pairing_approved
                    WHERE scope=? AND platform=?
                    """,
                    (self._scope, platform),
                ).fetchall()
                return {
                    str(row["user_id"]): {
                        "user_name": str(row["user_name"] or ""),
                        "approved_at": float(row["approved_at"]),
                    }
                    for row in rows
                }
        return {}

    def _save_json(self, path: Path, data: dict) -> None:
        path = Path(path)
        name = path.name
        with self._connect() as conn:
            if name == "_rate_limits.json":
                conn.execute(
                    "DELETE FROM gateway_pairing_limits WHERE scope=?",
                    (self._scope,),
                )
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO gateway_pairing_limits(scope, key, value)
                    VALUES (?, ?, ?)
                    """,
                    [
                        (self._scope, str(key), float(value))
                        for key, value in data.items()
                    ],
                )
                return
            if name.endswith("-pending.json"):
                platform = name[: -len("-pending.json")]
                conn.execute(
                    "DELETE FROM gateway_pairing_pending WHERE scope=? AND platform=?",
                    (self._scope, platform),
                )
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO gateway_pairing_pending
                    (scope, platform, code, user_id, user_name, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            self._scope,
                            platform,
                            str(code).upper().strip(),
                            str(info.get("user_id") or ""),
                            str(info.get("user_name") or ""),
                            float(info.get("created_at") or 0.0),
                        )
                        for code, info in data.items()
                        if isinstance(info, dict)
                    ],
                )
                return
            if name.endswith("-approved.json"):
                platform = name[: -len("-approved.json")]
                conn.execute(
                    "DELETE FROM gateway_pairing_approved WHERE scope=? AND platform=?",
                    (self._scope, platform),
                )
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO gateway_pairing_approved
                    (scope, platform, user_id, user_name, approved_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            self._scope,
                            platform,
                            str(user_id),
                            str(info.get("user_name") or ""),
                            float(info.get("approved_at") or 0.0),
                        )
                        for user_id, info in data.items()
                        if isinstance(info, dict)
                    ],
                )

    # ----- Approved users -----

    def is_approved(self, platform: str, user_id: str) -> bool:
        """Check if a user is approved (paired) on a platform."""
        approved = self._load_json(self._approved_path(platform))
        return user_id in approved

    def list_approved(self, platform: str = None) -> list:
        """List approved users, optionally filtered by platform."""
        results = []
        platforms = [platform] if platform else self._all_platforms("approved")
        for p in platforms:
            approved = self._load_json(self._approved_path(p))
            for uid, info in approved.items():
                results.append({"platform": p, "user_id": uid, **info})
        return results

    def _approve_user(self, platform: str, user_id: str, user_name: str = "") -> None:
        """Add a user to the approved list. Must be called under self._lock."""
        approved = self._load_json(self._approved_path(platform))
        approved[user_id] = {
            "user_name": user_name,
            "approved_at": time.time(),
        }
        self._save_json(self._approved_path(platform), approved)

    def revoke(self, platform: str, user_id: str) -> bool:
        """Remove a user from the approved list. Returns True if found."""
        path = self._approved_path(platform)
        with self._lock:
            approved = self._load_json(path)
            if user_id in approved:
                del approved[user_id]
                self._save_json(path, approved)
                return True
        return False

    # ----- Pending codes -----

    def generate_code(
        self, platform: str, user_id: str, user_name: str = ""
    ) -> Optional[str]:
        """
        Generate a pairing code for a new user.

        Returns the code string, or None if:
          - User is rate-limited (too recent request)
          - Max pending codes reached for this platform
          - User/platform is in lockout due to failed attempts
        """
        with self._lock:
            self._cleanup_expired(platform)

            # Check lockout
            if self._is_locked_out(platform):
                return None

            # Check rate limit for this specific user
            if self._is_rate_limited(platform, user_id):
                return None

            # Check max pending
            pending = self._load_json(self._pending_path(platform))
            if len(pending) >= MAX_PENDING_PER_PLATFORM:
                return None

            # Generate cryptographically random code
            code = "".join(secrets.choice(ALPHABET) for _ in range(CODE_LENGTH))

            # Store pending request
            pending[code] = {
                "user_id": user_id,
                "user_name": user_name,
                "created_at": time.time(),
            }
            self._save_json(self._pending_path(platform), pending)

            # Record rate limit
            self._record_rate_limit(platform, user_id)

            return code

    def approve_code(self, platform: str, code: str) -> Optional[dict]:
        """
        Approve a pairing code. Adds the user to the approved list.

        Returns {user_id, user_name} on success, None if code is
        invalid/expired OR the platform is currently locked out after
        ``MAX_FAILED_ATTEMPTS`` failed approvals (#10195). Callers can
        disambiguate with ``_is_locked_out(platform)``.
        """
        with self._lock:
            self._cleanup_expired(platform)
            code = code.upper().strip()

            # Lockout check — must run before the pending lookup so a
            # valid code (e.g. one already sitting in pending) cannot be
            # accepted once the lockout fires. Without this, the lockout
            # only blocks `generate_code`, not `approve_code` — nullifying
            # the brute-force protection for any code already issued.
            if self._is_locked_out(platform):
                return None

            pending = self._load_json(self._pending_path(platform))
            if code not in pending:
                self._record_failed_attempt(platform)
                return None

            entry = pending.pop(code)
            self._save_json(self._pending_path(platform), pending)

            # Add to approved list
            self._approve_user(platform, entry["user_id"], entry.get("user_name", ""))

            return {
                "user_id": entry["user_id"],
                "user_name": entry.get("user_name", ""),
            }

    def list_pending(self, platform: str = None) -> list:
        """List pending pairing requests, optionally filtered by platform."""
        results = []
        platforms = [platform] if platform else self._all_platforms("pending")
        for p in platforms:
            self._cleanup_expired(p)
            pending = self._load_json(self._pending_path(p))
            for code, info in pending.items():
                age_min = int((time.time() - info["created_at"]) / 60)
                results.append({
                    "platform": p,
                    "code": code,
                    "user_id": info["user_id"],
                    "user_name": info.get("user_name", ""),
                    "age_minutes": age_min,
                })
        return results

    def clear_pending(self, platform: str = None) -> int:
        """Clear all pending requests. Returns count removed."""
        with self._lock:
            count = 0
            platforms = [platform] if platform else self._all_platforms("pending")
            for p in platforms:
                pending = self._load_json(self._pending_path(p))
                count += len(pending)
                self._save_json(self._pending_path(p), {})
        return count

    # ----- Rate limiting and lockout -----

    def _is_rate_limited(self, platform: str, user_id: str) -> bool:
        """Check if a user has requested a code too recently."""
        limits = self._load_json(self._rate_limit_path())
        key = f"{platform}:{user_id}"
        last_request = limits.get(key, 0)
        return (time.time() - last_request) < RATE_LIMIT_SECONDS

    def _record_rate_limit(self, platform: str, user_id: str) -> None:
        """Record the time of a pairing request for rate limiting."""
        limits = self._load_json(self._rate_limit_path())
        key = f"{platform}:{user_id}"
        limits[key] = time.time()
        self._save_json(self._rate_limit_path(), limits)

    def _is_locked_out(self, platform: str) -> bool:
        """Check if a platform is in lockout due to failed approval attempts."""
        limits = self._load_json(self._rate_limit_path())
        lockout_key = f"_lockout:{platform}"
        lockout_until = limits.get(lockout_key, 0)
        return time.time() < lockout_until

    def _record_failed_attempt(self, platform: str) -> None:
        """Record a failed approval attempt. Triggers lockout after MAX_FAILED_ATTEMPTS."""
        limits = self._load_json(self._rate_limit_path())
        fail_key = f"_failures:{platform}"
        fails = limits.get(fail_key, 0) + 1
        limits[fail_key] = fails
        if fails >= MAX_FAILED_ATTEMPTS:
            lockout_key = f"_lockout:{platform}"
            limits[lockout_key] = time.time() + LOCKOUT_SECONDS
            limits[fail_key] = 0  # Reset counter
            print(f"[pairing] Platform {platform} locked out for {LOCKOUT_SECONDS}s "
                  f"after {MAX_FAILED_ATTEMPTS} failed attempts", flush=True)
        self._save_json(self._rate_limit_path(), limits)

    # ----- Cleanup -----

    def _cleanup_expired(self, platform: str) -> None:
        """Remove expired pending codes."""
        path = self._pending_path(platform)
        pending = self._load_json(path)
        now = time.time()
        expired = [
            code for code, info in pending.items()
            if (now - info["created_at"]) > CODE_TTL_SECONDS
        ]
        if expired:
            for code in expired:
                del pending[code]
            self._save_json(path, pending)

    def _all_platforms(self, suffix: str) -> list:
        """List all platforms that have data of a given suffix."""
        table = (
            "gateway_pairing_approved"
            if suffix == "approved"
            else "gateway_pairing_pending"
            if suffix == "pending"
            else ""
        )
        if not table:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT platform FROM {table} WHERE scope=? ORDER BY platform",
                (self._scope,),
            ).fetchall()
            return [str(row["platform"]) for row in rows]
