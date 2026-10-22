"""Unified approval store for multi-sig authorization."""

import sqlite3
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone
from storage.types import ApprovalRecord
from runtime_paths import sqlite_db_path
from storage.hosted_postgres import enabled, connect_authority, require_tables, sql_query, begin_write


class UnifiedApprovalStore:
    """Unified approval store for multi-signature gates.
    
    Consolidates:
    - src/approval.py (ApprovalStore)
    
    Provides:
    - Multi-sig approval gating
    - Approval tracking and audit trail
    """
    
    def __init__(self, db_path: str = str(sqlite_db_path("approvals.db", ".semantier/approvals.db"))):
        self.db_path = Path(db_path)
        if enabled():
            self.conn = connect_authority("standalone", self.db_path)
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.db_path))
        self._init_schema()
    
    def _init_schema(self):
        """Initialize schema."""
        if enabled():
            require_tables(self.conn, 'approvals')
            self.conn.commit()
            return
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS approvals (
            id TEXT PRIMARY KEY,
            idempotency_key TEXT NOT NULL,
            approver TEXT NOT NULL,
            approved_at TEXT NOT NULL,
            metadata_json TEXT,
            UNIQUE(idempotency_key, approver)
        )
        """)
        
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_approvals_idempotency_key
        ON approvals (idempotency_key)
        """)
        
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_approvals_approver
        ON approvals (approver)
        """)
        
        self.conn.commit()
    
    def add_approval(
        self,
        key: str,
        approver: str,
        metadata: Optional[dict] = None,
    ) -> ApprovalRecord:
        """Record an approval."""
        import json
        import uuid
        
        now = datetime.now(timezone.utc).isoformat()
        approval_id = str(uuid.uuid4())
        
        self.conn.execute(sql_query("""
        INSERT OR IGNORE INTO approvals (id, idempotency_key, approver, approved_at, metadata_json)
        VALUES (?, ?, ?, ?, ?)
        """, """
        INSERT INTO approvals (id, idempotency_key, approver, approved_at, metadata_json)
        VALUES (%s, %s, %s, %s, %s)
         ON CONFLICT DO NOTHING
"""), (
            approval_id,
            key,
            approver,
            now,
            json.dumps(metadata or {})
        ))
        
        self.conn.commit()
        
        return ApprovalRecord(
            idempotency_key=key,
            approver=approver,
            approved_at=now,
            metadata=metadata or {},
        )
    
    def count_approvals(self, key: str) -> int:
        """Count approvals for a key."""
        cursor = self.conn.execute(
            sql_query("SELECT COUNT(*) FROM approvals WHERE idempotency_key = ?", "SELECT COUNT(*) FROM approvals WHERE idempotency_key = %s"),
            (key,)
        )
        return cursor.fetchone()[0]
    
    def has_approval(self, key: str) -> bool:
        """Check if key has at least one approval."""
        return self.count_approvals(key) > 0
    
    def list_approvers(self, key: str) -> List[str]:
        """List all approvers for a key."""
        cursor = self.conn.execute(
            sql_query("SELECT DISTINCT approver FROM approvals WHERE idempotency_key = ? ORDER BY approver", "SELECT DISTINCT approver FROM approvals WHERE idempotency_key = %s ORDER BY approver"),
            (key,)
        )
        return [row[0] for row in cursor.fetchall()]
    
    def get_approvals(self, key: str) -> List[ApprovalRecord]:
        """Get all approval records for a key."""
        import json
        
        cursor = self.conn.execute(sql_query("""
        SELECT idempotency_key, approver, approved_at, metadata_json
        FROM approvals
        WHERE idempotency_key = ?
        ORDER BY approved_at ASC
        """, """
        SELECT idempotency_key, approver, approved_at, metadata_json
        FROM approvals
        WHERE idempotency_key = %s
        ORDER BY approved_at ASC
        """), (key,))
        
        return [
            ApprovalRecord(
                idempotency_key=row[0],
                approver=row[1],
                approved_at=row[2],
                metadata=json.loads(row[3]) if row[3] else {},
            )
            for row in cursor.fetchall()
        ]
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# Module-level singleton for backward compatibility
_default_store: Optional[UnifiedApprovalStore] = None


def get_approval_store(path: str = str(sqlite_db_path("approvals.db", ".semantier/approvals.db"))) -> UnifiedApprovalStore:
    """Get or create default approval store instance."""
    global _default_store
    if _default_store is None:
        _default_store = UnifiedApprovalStore(db_path=path)
    return _default_store
