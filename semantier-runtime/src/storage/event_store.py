"""Unified event store consolidating ExecutionEventStore, event_store, sqlite_event_store."""

import json
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from storage.types import EventRecord, ExecutionEventType
from eos.artifact_hashing import canonical_json, content_hash
from runtime_paths import sqlite_db_path
from storage.hosted_postgres import enabled, connect_authority, require_tables, sql_query, begin_write


def utc_now() -> str:
    """Get ISO8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class UnifiedEventStore:
    """Unified event store replacing all fragmented implementations.
    
    Consolidates:
    - ExecutionEventStore (src/events.py)
    - event_store (src/eos/event_store.py)
    - sqlite_event_store (src/eos/sqlite_event_store.py)
    
    Features:
    - Content-addressable event hashing (Merkle chain)
    - Deterministic canonical JSON
    - Per-org event isolation
    - Projection versioning
    - Replay support
    """
    
    def __init__(self, db_path: str = str(sqlite_db_path("events.db", ".semantier/events.db"))):
        self.db_path = Path(db_path)
        if enabled():
            self.conn = connect_authority("standalone", self.db_path)
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.db_path))
        self._init_schema()
    
    def _init_schema(self):
        """Initialize unified schema."""
        if enabled():
            require_tables(self.conn, 'events', 'projection_versions')
            self.conn.commit()
            return
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            org_id TEXT,
            stream_id TEXT,
            correlation_id TEXT,
            occurred_at TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            
            -- EOS extensions
            ontology_version TEXT,
            projection_version INTEGER,
            content_hash TEXT,
            prior_event_hash TEXT,
            signature TEXT,
            
            created_at TEXT NOT NULL
        )
        """)

        event_columns = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(events)").fetchall()
        }
        if "stream_id" not in event_columns:
            self.conn.execute("ALTER TABLE events ADD COLUMN stream_id TEXT")
            self.conn.execute(
                "UPDATE events SET stream_id = COALESCE(stream_id, org_id, 'global')"
            )
        if "prior_event_hash" not in event_columns:
            self.conn.execute("ALTER TABLE events ADD COLUMN prior_event_hash TEXT")
            if "prev_event_hash" in event_columns:
                self.conn.execute(
                    "UPDATE events SET prior_event_hash = prev_event_hash WHERE prior_event_hash IS NULL"
                )
        
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_org_id
        ON events (org_id)
        """)
        
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_stream_id
        ON events (stream_id)
        """)
        
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_correlation_id
        ON events (correlation_id)
        """)
        
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_occurred_at
        ON events (occurred_at)
        """)
        
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_event_type
        ON events (event_type)
        """)

        # Projection versioning table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS projection_versions (
            projection_id TEXT PRIMARY KEY,
            org_id TEXT NOT NULL,
            ontology_version TEXT NOT NULL,
            version INTEGER NOT NULL,
            semantic_projection_input_json TEXT,
            semantic_scope_json TEXT,
            bindings_json TEXT,
            constraint_layer_json TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(org_id, ontology_version, version)
        )
        """)
        
        self.conn.commit()
    
    def append_event(
        self,
        event_id: str,
        event_type: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        org_id: Optional[str] = None,
        stream_id: Optional[str] = None,
        ontology_version: Optional[str] = None,
        projection_version: Optional[int] = None,
        signature: Optional[str] = None,
    ) -> EventRecord:
        """Append event with automatic content hashing."""
        
        # Compute content hash of payload
        event_content_hash = content_hash(payload)
        resolved_stream_id = stream_id or org_id or "global"
        
        # Serialize the stream head read with its append across PG workers.
        if enabled():
            begin_write(self.conn)
        # Get previous event hash for same stream.
        prev_hash = self._lookup_previous_hash(resolved_stream_id)
        
        occurred_at = utc_now()
        
        self.conn.execute(sql_query("""
        INSERT INTO events (
            event_id,
            event_type,
            org_id,
            stream_id,
            correlation_id,
            occurred_at,
            payload_json,
            ontology_version,
            projection_version,
            content_hash,
            prior_event_hash,
            signature,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, """
        INSERT INTO events (
            event_id,
            event_type,
            org_id,
            stream_id,
            correlation_id,
            occurred_at,
            payload_json,
            ontology_version,
            projection_version,
            content_hash,
            prior_event_hash,
            signature,
            created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """), (
            event_id,
            event_type,
            org_id,
            resolved_stream_id,
            correlation_id,
            occurred_at,
            canonical_json(payload),
            ontology_version,
            projection_version,
            event_content_hash,
            prev_hash,
            signature,
            utc_now(),
        ))
        
        self.conn.commit()
        
        return EventRecord(
            event_id=event_id,
            event_type=event_type,
            occurred_at=occurred_at,
            correlation_id=correlation_id or event_id,
            payload=payload,
            org_id=org_id,
            stream_id=resolved_stream_id,
            ontology_version=ontology_version,
            projection_version=projection_version,
            content_hash=event_content_hash,
            prior_event_hash=prev_hash,
            signature=signature,
        )

    def _lookup_previous_hash(self, stream_id: str) -> Optional[str]:
        cursor = self.conn.execute(
            sql_query("""
            SELECT content_hash
            FROM events
            WHERE stream_id = ?
            ORDER BY occurred_at DESC, created_at DESC, event_id DESC
            LIMIT 1
            """, """
            SELECT content_hash
            FROM events
            WHERE stream_id = %s
            ORDER BY occurred_at DESC, created_at DESC, event_id DESC
            LIMIT 1
            """),
            (stream_id,),
        )
        row = cursor.fetchone()
        return row[0] if row else None
    
    def get_event(self, event_id: str) -> Optional[EventRecord]:
        """Retrieve event by ID."""
        cursor = self.conn.execute(
            sql_query("""
            SELECT event_id, event_type, org_id, stream_id, correlation_id, occurred_at,
                   payload_json, ontology_version, projection_version,
                   content_hash, prior_event_hash, signature
            FROM events WHERE event_id = ?
            """, """
            SELECT event_id, event_type, org_id, stream_id, correlation_id, occurred_at,
                   payload_json, ontology_version, projection_version,
                   content_hash, prior_event_hash, signature
            FROM events WHERE event_id = %s
            """),
            (event_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        
        return EventRecord(
            event_id=row[0],
            event_type=row[1],
            org_id=row[2],
            stream_id=row[3],
            correlation_id=row[4],
            occurred_at=row[5],
            payload=json.loads(row[6]),
            ontology_version=row[7],
            projection_version=row[8],
            content_hash=row[9],
            prior_event_hash=row[10],
            signature=row[11],
        )
    
    def list_events(
        self,
        org_id: Optional[str] = None,
        event_type: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[EventRecord]:
        """List events with optional filtering."""
        query = "SELECT event_id, event_type, org_id, stream_id, correlation_id, occurred_at, payload_json, ontology_version, projection_version, content_hash, prior_event_hash, signature FROM events WHERE 1=1"
        params = []
        
        if org_id:
            query += sql_query(" AND org_id = ?", " AND org_id = %s")
            params.append(org_id)
        
        if event_type:
            query += sql_query(" AND event_type = ?", " AND event_type = %s")
            params.append(event_type)
        
        if correlation_id:
            query += sql_query(" AND correlation_id = ?", " AND correlation_id = %s")
            params.append(correlation_id)
        
        query += sql_query(" ORDER BY occurred_at ASC LIMIT ?", " ORDER BY occurred_at ASC LIMIT %s")
        params.append(limit)
        
        cursor = self.conn.execute(query, params)
        rows = cursor.fetchall()
        
        return [
            EventRecord(
                event_id=row[0],
                event_type=row[1],
                org_id=row[2],
                stream_id=row[3],
                correlation_id=row[4],
                occurred_at=row[5],
                payload=json.loads(row[6]),
                ontology_version=row[7],
                projection_version=row[8],
                content_hash=row[9],
                prior_event_hash=row[10],
                signature=row[11],
            )
            for row in rows
        ]
    
    def replay_chain(self, org_id: str) -> List[EventRecord]:
        """Get full event chain for replay (in order)."""
        return self.list_events(org_id=org_id, limit=10000)

    def verify_chain(self, org_id: str) -> Dict[str, Any]:
        """Verify per-org append-only hash chaining."""
        events = self.replay_chain(org_id)
        expected_prev_hash = None
        for event in events:
            recalculated_hash = content_hash(event.payload)
            if event.content_hash != recalculated_hash:
                return {
                    "status": "HASH_MISMATCH",
                    "event_id": event.event_id,
                    "stored_hash": event.content_hash,
                    "recalculated_hash": recalculated_hash,
                }
            if event.prior_event_hash != expected_prev_hash:
                return {
                    "status": "CHAIN_BROKEN",
                    "event_id": event.event_id,
                    "expected_prior_event_hash": expected_prev_hash,
                    "actual_prior_event_hash": event.prior_event_hash,
                }
            expected_prev_hash = event.content_hash
        return {"status": "OK", "event_count": len(events)}
    
    def version_projection(
        self,
        org_id: str,
        ontology_version: str,
        semantic_projection_input: Dict[str, Any],
        projection_output: Dict[str, Any],
    ) -> int:
        """Store a versioned projection. Returns version number."""
        if enabled():
            begin_write(self.conn)
        cursor = self.conn.execute(
            sql_query("""
            SELECT MAX(version) FROM projection_versions
            WHERE org_id = ? AND ontology_version = ?
            """, """
            SELECT MAX(version) FROM projection_versions
            WHERE org_id = %s AND ontology_version = %s
            """),
            (org_id, ontology_version)
        )
        row = cursor.fetchone()
        next_version = (row[0] or 0) + 1
        
        projection_id = f"proj_{org_id}_{ontology_version}_{next_version}"
        
        self.conn.execute(sql_query("""
        INSERT INTO projection_versions (
            projection_id,
            org_id,
            ontology_version,
            version,
            semantic_projection_input_json,
            semantic_scope_json,
            bindings_json,
            constraint_layer_json,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, """
        INSERT INTO projection_versions (
            projection_id,
            org_id,
            ontology_version,
            version,
            semantic_projection_input_json,
            semantic_scope_json,
            bindings_json,
            constraint_layer_json,
            created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """), (
            projection_id,
            org_id,
            ontology_version,
            next_version,
            json.dumps(semantic_projection_input),
            json.dumps(projection_output.get("semantic_scope", {})),
            json.dumps(projection_output.get("bindings", {})),
            json.dumps(projection_output.get("constraint_layer", {})),
            utc_now(),
        ))
        
        self.conn.commit()
        
        return next_version
    
    def get_projection(
        self,
        org_id: str,
        ontology_version: str,
        version: int,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a specific projection version."""
        cursor = self.conn.execute(sql_query("""
        SELECT semantic_scope_json, bindings_json, constraint_layer_json
        FROM projection_versions
        WHERE org_id = ? AND ontology_version = ? AND version = ?
        """, """
        SELECT semantic_scope_json, bindings_json, constraint_layer_json
        FROM projection_versions
        WHERE org_id = %s AND ontology_version = %s AND version = %s
        """), (org_id, ontology_version, version))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return {
            "semantic_scope": json.loads(row[0]),
            "bindings": json.loads(row[1]),
            "constraint_layer": json.loads(row[2]),
        }
    
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
_default_store: Optional[UnifiedEventStore] = None


def get_event_store(path: str = str(sqlite_db_path("events.db", ".semantier/events.db"))) -> UnifiedEventStore:
    """Get or create default event store instance."""
    global _default_store
    if _default_store is None:
        _default_store = UnifiedEventStore(db_path=path)
    return _default_store
