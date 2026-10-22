"""Unified idempotency store consolidating fragmented implementations."""

import sqlite3
from pathlib import Path
from typing import Optional
from storage.types import IdempotencyKey
from runtime_paths import sqlite_db_path
from storage.hosted_postgres import enabled, connect_authority, require_tables, sql_query, begin_write


class UnifiedIdempotencyStore:
    """Unified idempotency store.
    
    Consolidates:
    - src/idempotency.py (IdempotencyStore)
    - src/eos/idempotency_store.py
    
    Ensures deterministic replay and no duplicate commits.
    """
    
    def __init__(self, db_path: str = str(sqlite_db_path("idempotency.db", ".semantier/idempotency.db"))):
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
            require_tables(self.conn, 'idempotency_keys')
            self.conn.commit()
            return
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS idempotency_keys (
            key TEXT PRIMARY KEY,
            org_id TEXT,
            intent_hash TEXT,
            registered_at TEXT NOT NULL
        )
        """)
        
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_idempotency_org_intent
        ON idempotency_keys (org_id, intent_hash)
        """)
        
        self.conn.commit()
    
    def exists(self, key: str) -> bool:
        """Check if idempotency key has been seen."""
        cursor = self.conn.execute(
            sql_query("SELECT 1 FROM idempotency_keys WHERE key = ?", "SELECT 1 FROM idempotency_keys WHERE key = %s"),
            (key,)
        )
        return cursor.fetchone() is not None
    
    def register(self, key: str) -> None:
        """Register an idempotency key."""
        from datetime import datetime, timezone
        
        now = datetime.now(timezone.utc).isoformat()
        
        # Parse org_id and intent_hash if key is in IdempotencyKey format
        org_id = None
        intent_hash = None
        
        if ":" in key:
            try:
                id_key = IdempotencyKey.from_string(key)
                org_id = id_key.org_id
                intent_hash = id_key.intent_hash
            except Exception:
                pass
        
        self.conn.execute(sql_query("""
        INSERT OR IGNORE INTO idempotency_keys (key, org_id, intent_hash, registered_at)
        VALUES (?, ?, ?, ?)
        """, """
        INSERT INTO idempotency_keys (key, org_id, intent_hash, registered_at)
        VALUES (%s, %s, %s, %s)
         ON CONFLICT DO NOTHING
"""), (key, org_id, intent_hash, now))
        
        self.conn.commit()
    
    def seen(self, org_id: str, intent_hash: str) -> bool:
        """Check if org_id + intent_hash has been seen.
        
        This is the alternate API from eos/idempotency_store.py.
        """
        cursor = self.conn.execute(
            sql_query("SELECT 1 FROM idempotency_keys WHERE org_id = ? AND intent_hash = ?", "SELECT 1 FROM idempotency_keys WHERE org_id = %s AND intent_hash = %s"),
            (org_id, intent_hash)
        )
        return cursor.fetchone() is not None
    
    def commit(self, org_id: str, intent_hash: str) -> None:
        """Register org_id + intent_hash.
        
        This is the alternate API from eos/idempotency_store.py.
        """
        key = IdempotencyKey(org_id=org_id, intent_hash=intent_hash).to_string()
        self.register(key)
    
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
_default_store: Optional[UnifiedIdempotencyStore] = None


def get_idempotency_store(path: str = str(sqlite_db_path("idempotency.db", ".semantier/idempotency.db"))) -> UnifiedIdempotencyStore:
    """Get or create default idempotency store instance."""
    global _default_store
    if _default_store is None:
        _default_store = UnifiedIdempotencyStore(db_path=path)
    return _default_store
