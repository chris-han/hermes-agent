import os
import sqlite3
from pathlib import Path

from runtime_paths import sqlite_db_path


class IdempotencyStore:
    def __init__(self, path: str | None = None):
        from storage.hosted_postgres import enabled, HostedStorageUnavailable
        if enabled():
            raise HostedStorageUnavailable('legacy idempotency CLI is local-only; hosted runtime uses the governed idempotency owner')
        raw_path = path or os.getenv(
            "SEMANTIER_IDEMPOTENCY_STORE",
            str(sqlite_db_path("idempotency_store.db", ".semantier/idempotency_store.db")),
        )

        self._path = Path(raw_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self._path)

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS idempotency_keys (
                key TEXT PRIMARY KEY
            )
            """
        )
        self._conn.commit()

    # ----------------------------
    # Core semantics
    # ----------------------------

    def exists(self, key: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM idempotency_keys WHERE key = ?",
            (key,),
        )
        return cur.fetchone() is not None

    def register(self, key: str):
        self._conn.execute(
            "INSERT INTO idempotency_keys (key) VALUES (?)",
            (key,),
        )
        self._conn.commit()

    # ----------------------------
    # Lifecycle management
    # ----------------------------

    def close(self):
        if getattr(self, "_conn", None):
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        # best-effort cleanup only
        try:
            self.close()
        except Exception:
            pass
