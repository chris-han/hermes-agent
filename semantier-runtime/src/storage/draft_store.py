"""Semantic draft store for interactive semantic completion per PRD.

This enables the Weixin → Semantier semantic completion flow where
the system interactively completes missing semantic requirements
before validation and execution.
"""

import sqlite3
import json
import uuid
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timezone
from storage.types import SemanticDraft, SemanticDraftStatus
from runtime_paths import sqlite_db_path
from storage.hosted_postgres import enabled, connect_authority, require_tables, sql_query, begin_write


def utc_now() -> str:
    """Get ISO8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class DraftStore:
    """Semantic draft store for in-flight multi-turn semantic completion.
    
    Per PRD:
    - Maintains draft event per conversation
    - Detects required but missing fields
    - Enables interactive completion
    - Binds evidence to draft
    - Only validates when draft is complete
    """
    
    def __init__(self, db_path: str = str(sqlite_db_path("drafts.db", ".semantier/drafts.db"))):
        self.db_path = Path(db_path)
        if enabled():
            self.conn = connect_authority("standalone", self.db_path)
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.db_path))
        self._init_schema()
    
    def _init_schema(self):
        """Initialize schema, then run migration for new columns."""
        if enabled():
            require_tables(self.conn, 'semantic_drafts', 'draft_messages', 'draft_evidence')
            self.conn.commit()
            return
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS semantic_drafts (
            conversation_id TEXT PRIMARY KEY,
            org_id TEXT NOT NULL,
            tags_json TEXT,
            amount TEXT,
            currency TEXT,
            counterparty TEXT,
            invoice_ref TEXT,
            evidence_refs_json TEXT,
            missing_fields_json TEXT,
            status TEXT NOT NULL,
            payload_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """)

        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_drafts_org_id
        ON semantic_drafts (org_id)
        """)

        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_drafts_status
        ON semantic_drafts (status)
        """)

        # User messages and queries during semantic completion
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS draft_messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES semantic_drafts(conversation_id)
        )
        """)

        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_draft_messages_conversation
        ON draft_messages (conversation_id)
        """)

        # Evidence attached to drafts (e.g., invoice uploads)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS draft_evidence (
            evidence_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            evidence_type TEXT NOT NULL,
            reference TEXT NOT NULL,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES semantic_drafts(conversation_id)
        )
        """)

        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_draft_evidence_conversation
        ON draft_evidence (conversation_id)
        """)

        self.conn.commit()
        self._migrate_schema()

    def _migrate_schema(self):
        """Idempotently add FR7/FR8/FR9 columns to existing databases."""
        cursor = self.conn.execute("PRAGMA table_info(semantic_drafts)")
        existing = {row[1] for row in cursor.fetchall()}
        migrations = [
            ("pins_json",               "TEXT"),
            ("gate_mode",               "TEXT NOT NULL DEFAULT 'tiered'"),
            ("projection_context_hash", "TEXT"),
            ("projection_status",       "TEXT"),
        ]
        for col, col_def in migrations:
            if col not in existing:
                self.conn.execute(
                    f"ALTER TABLE semantic_drafts ADD COLUMN {col} {col_def}"
                )
        self.conn.commit()
    
    def create_draft(
        self,
        conversation_id: str,
        org_id: str,
        initial_payload: Optional[dict] = None,
    ) -> SemanticDraft:
        """Create a new semantic draft for a conversation."""
        now = utc_now()
        
        draft = SemanticDraft(
            conversation_id=conversation_id,
            org_id=org_id,
            status=SemanticDraftStatus.INCOMPLETE,
            payload=initial_payload or {},
            created_at=now,
            updated_at=now,
        )
        
        self.conn.execute(sql_query("""
        INSERT INTO semantic_drafts (
            conversation_id,
            org_id,
            tags_json,
            amount,
            currency,
            counterparty,
            invoice_ref,
            evidence_refs_json,
            missing_fields_json,
            status,
            payload_json,
            created_at,
            updated_at,
            pins_json,
            gate_mode,
            projection_context_hash,
            projection_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, """
        INSERT INTO semantic_drafts (
            conversation_id,
            org_id,
            tags_json,
            amount,
            currency,
            counterparty,
            invoice_ref,
            evidence_refs_json,
            missing_fields_json,
            status,
            payload_json,
            created_at,
            updated_at,
            pins_json,
            gate_mode,
            projection_context_hash,
            projection_status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """), (
            draft.conversation_id,
            draft.org_id,
            json.dumps(draft.tags),
            draft.amount,
            draft.currency,
            draft.counterparty,
            draft.invoice_ref,
            json.dumps(draft.evidence_refs),
            json.dumps(draft.missing_fields),
            draft.status.value,
            json.dumps(draft.payload),
            draft.created_at,
            draft.updated_at,
            json.dumps(draft.pins),
            draft.gate_mode,
            draft.projection_context_hash,
            draft.projection_status,
        ))
        
        self.conn.commit()
        
        return draft
    
    def get_draft(self, conversation_id: str) -> Optional[SemanticDraft]:
        """Retrieve a draft by conversation ID."""
        cursor = self.conn.execute(sql_query("""
        SELECT conversation_id, org_id, tags_json, amount, currency, counterparty,
               invoice_ref, evidence_refs_json, missing_fields_json, status,
               payload_json, created_at, updated_at,
               pins_json, gate_mode, projection_context_hash, projection_status
        FROM semantic_drafts
        WHERE conversation_id = ?
        """, """
        SELECT conversation_id, org_id, tags_json, amount, currency, counterparty,
               invoice_ref, evidence_refs_json, missing_fields_json, status,
               payload_json, created_at, updated_at,
               pins_json, gate_mode, projection_context_hash, projection_status
        FROM semantic_drafts
        WHERE conversation_id = %s
        """), (conversation_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return SemanticDraft(
            conversation_id=row[0],
            org_id=row[1],
            tags=json.loads(row[2]) if row[2] else [],
            amount=row[3],
            currency=row[4],
            counterparty=row[5],
            invoice_ref=row[6],
            evidence_refs=json.loads(row[7]) if row[7] else [],
            missing_fields=json.loads(row[8]) if row[8] else [],
            status=SemanticDraftStatus(row[9]),
            payload=json.loads(row[10]) if row[10] else {},
            created_at=row[11],
            updated_at=row[12],
            pins=json.loads(row[13]) if row[13] else {},
            gate_mode=row[14] if row[14] else "tiered",
            projection_context_hash=row[15],
            projection_status=row[16],
        )
    
    def update_draft(self, draft: SemanticDraft) -> SemanticDraft:
        """Update a draft."""
        draft.updated_at = utc_now()
        
        self.conn.execute(sql_query("""
        UPDATE semantic_drafts SET
            tags_json = ?,
            amount = ?,
            currency = ?,
            counterparty = ?,
            invoice_ref = ?,
            evidence_refs_json = ?,
            missing_fields_json = ?,
            status = ?,
            payload_json = ?,
            updated_at = ?,
            pins_json = ?,
            gate_mode = ?,
            projection_context_hash = ?,
            projection_status = ?
        WHERE conversation_id = ?
        """, """
        UPDATE semantic_drafts SET
            tags_json = %s,
            amount = %s,
            currency = %s,
            counterparty = %s,
            invoice_ref = %s,
            evidence_refs_json = %s,
            missing_fields_json = %s,
            status = %s,
            payload_json = %s,
            updated_at = %s,
            pins_json = %s,
            gate_mode = %s,
            projection_context_hash = %s,
            projection_status = %s
        WHERE conversation_id = %s
        """), (
            json.dumps(draft.tags),
            draft.amount,
            draft.currency,
            draft.counterparty,
            draft.invoice_ref,
            json.dumps(draft.evidence_refs),
            json.dumps(draft.missing_fields),
            draft.status.value,
            json.dumps(draft.payload),
            draft.updated_at,
            json.dumps(draft.pins),
            draft.gate_mode,
            draft.projection_context_hash,
            draft.projection_status,
            draft.conversation_id,
        ))
        
        self.conn.commit()
        
        return draft
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> dict:
        """Add a message to draft conversation history."""
        message_id = str(uuid.uuid4())
        timestamp = utc_now()
        
        self.conn.execute(sql_query("""
        INSERT INTO draft_messages (message_id, conversation_id, role, content, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """, """
        INSERT INTO draft_messages (message_id, conversation_id, role, content, timestamp)
        VALUES (%s, %s, %s, %s, %s)
        """), (message_id, conversation_id, role, content, timestamp))
        
        self.conn.commit()
        
        return {
            "message_id": message_id,
            "role": role,
            "content": content,
            "timestamp": timestamp,
        }
    
    def get_messages(self, conversation_id: str) -> List[dict]:
        """Get all messages for a conversation."""
        cursor = self.conn.execute(sql_query("""
        SELECT message_id, role, content, timestamp
        FROM draft_messages
        WHERE conversation_id = ?
        ORDER BY timestamp ASC
        """, """
        SELECT message_id, role, content, timestamp
        FROM draft_messages
        WHERE conversation_id = %s
        ORDER BY timestamp ASC
        """), (conversation_id,))
        
        return [
            {
                "message_id": row[0],
                "role": row[1],
                "content": row[2],
                "timestamp": row[3],
            }
            for row in cursor.fetchall()
        ]
    
    def add_evidence(
        self,
        conversation_id: str,
        evidence_type: str,
        reference: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Add evidence to a draft (e.g., uploaded invoice)."""
        evidence_id = str(uuid.uuid4())
        
        self.conn.execute(sql_query("""
        INSERT INTO draft_evidence (evidence_id, conversation_id, evidence_type, reference, metadata_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """, """
        INSERT INTO draft_evidence (evidence_id, conversation_id, evidence_type, reference, metadata_json, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        """), (
            evidence_id,
            conversation_id,
            evidence_type,
            reference,
            json.dumps(metadata or {}),
            utc_now(),
        ))
        
        self.conn.commit()
        
        # Also update draft's evidence_refs
        draft = self.get_draft(conversation_id)
        if draft:
            draft.evidence_refs.append(evidence_id)
            self.update_draft(draft)
        
        return evidence_id
    
    def get_evidence(self, conversation_id: str) -> List[dict]:
        """Get all evidence for a conversation."""
        cursor = self.conn.execute(sql_query("""
        SELECT evidence_id, evidence_type, reference, metadata_json, created_at
        FROM draft_evidence
        WHERE conversation_id = ?
        ORDER BY created_at ASC
        """, """
        SELECT evidence_id, evidence_type, reference, metadata_json, created_at
        FROM draft_evidence
        WHERE conversation_id = %s
        ORDER BY created_at ASC
        """), (conversation_id,))
        
        return [
            {
                "evidence_id": row[0],
                "evidence_type": row[1],
                "reference": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "created_at": row[4],
            }
            for row in cursor.fetchall()
        ]
    
    def list_drafts(
        self,
        org_id: str,
        status: Optional[SemanticDraftStatus] = None,
        limit: int = 100,
    ) -> List[SemanticDraft]:
        """List drafts for an org."""
        query = sql_query("SELECT conversation_id, org_id, tags_json, amount, currency, counterparty, invoice_ref, evidence_refs_json, missing_fields_json, status, payload_json, created_at, updated_at FROM semantic_drafts WHERE org_id = ?", "SELECT conversation_id, org_id, tags_json, amount, currency, counterparty, invoice_ref, evidence_refs_json, missing_fields_json, status, payload_json, created_at, updated_at FROM semantic_drafts WHERE org_id = %s")
        params = [org_id]
        
        if status:
            query += sql_query(" AND status = ?", " AND status = %s")
            params.append(status.value)
        
        query += sql_query(" ORDER BY updated_at DESC LIMIT ?", " ORDER BY updated_at DESC LIMIT %s")
        params.append(limit)
        
        cursor = self.conn.execute(query, params)
        
        return [
            SemanticDraft(
                conversation_id=row[0],
                org_id=row[1],
                tags=json.loads(row[2]) if row[2] else [],
                amount=row[3],
                currency=row[4],
                counterparty=row[5],
                invoice_ref=row[6],
                evidence_refs=json.loads(row[7]) if row[7] else [],
                missing_fields=json.loads(row[8]) if row[8] else [],
                status=SemanticDraftStatus(row[9]),
                payload=json.loads(row[10]) if row[10] else {},
                created_at=row[11],
                updated_at=row[12],
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
_default_store: Optional[DraftStore] = None


def get_draft_store(path: str = str(sqlite_db_path("drafts.db", ".semantier/drafts.db"))) -> DraftStore:
    """Get or create default draft store instance."""
    global _default_store
    if _default_store is None:
        _default_store = DraftStore(db_path=path)
    return _default_store
