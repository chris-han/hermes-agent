import sqlite3
import os
from pathlib import Path

from runtime_paths import sqlite_db_path


class ApprovalStore:
    def __init__(self, path: str | None = None):
        from storage.hosted_postgres import enabled, HostedStorageUnavailable
        if enabled():
            raise HostedStorageUnavailable('legacy approval CLI is local-only; hosted runtime uses the governed approval owner')
        raw_path = path or os.getenv(
            "SEMANTIER_APPROVAL_STORE",
            str(sqlite_db_path("approval.db", ".semantier/approval.db"))
        )
        self._path = Path(raw_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self._path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS approvals (
                idempotency_key TEXT,
                approver TEXT,
                PRIMARY KEY (idempotency_key, approver)
            )
            """
        )
        self._conn.commit()

    def add_approval(self, key: str, approver: str):
        self._conn.execute(
            "INSERT OR IGNORE INTO approvals VALUES (?, ?)",
            (key, approver),
        )
        self._conn.commit()

    def count_approvals(self, key: str) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM approvals WHERE idempotency_key=?",
            (key,),
        )
        return cur.fetchone()[0]
    
    def has_approval(self, key: str) -> bool:
        return self.count_approvals(key) > 0
    
    def list_approvers(self, key: str) -> list[str]:
        cur = self._conn.execute(
            "SELECT approver FROM approvals WHERE idempotency_key=? ORDER BY approver",
            (key,),
        )
        return [row[0] for row in cur.fetchall()]
