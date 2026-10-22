from storage import hosted_postgres as hosted
import sqlite3
import json
from datetime import datetime, timezone
import eos.signer as signer

from eos.artifact_hashing import content_hash
from runtime_paths import sqlite_db_path


class SemantierEventStore:

    def __init__(self, db_path=None):
        if hosted.enabled():
            # Hosted Merkle/replay routes share the governed EOS owner. An
            # explicit path must be mapped as EOS; it cannot open local files.
            from eos.db import get_conn
            self.conn = get_conn(db_path)
            hosted.require_tables(self.conn, 'semantier_events', 'legacy_projection_versions', 'merkle_batches', 'merkle_proofs')
            self.conn.commit()
            return
        db_path = db_path or str(sqlite_db_path("semantier.db", "semantier.db"))
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _init_schema(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS semantier_events (
            event_id TEXT PRIMARY KEY,
            org_id TEXT NOT NULL,
            intent_id TEXT,
            signal_ref TEXT,
            workflow_ref TEXT,
            rea_ref TEXT,
            ontology_version TEXT,
            projection_version INTEGER,
            policy_json TEXT,
            justification_json TEXT,
            execution_json TEXT,
            effects_json TEXT,
            content_hash TEXT,
            prev_event_hash TEXT,
            signature TEXT,
            created_at TEXT NOT NULL
        )
        """)

        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS projection_versions (
            org_id TEXT NOT NULL,
            ontology_version TEXT NOT NULL,
            projection_version INTEGER NOT NULL,
            semantic_projection_input_json TEXT,
            semantic_scope_json TEXT,
            bindings_json TEXT,
            constraint_layer_json TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (org_id, ontology_version, projection_version)
        )
        """)

        self.conn.commit()

        # Merkle batch / proofs tables for external audit
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS merkle_batches (
            batch_id TEXT PRIMARY KEY,
            root_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)

        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS merkle_proofs (
            event_id TEXT NOT NULL,
            batch_id TEXT NOT NULL,
            proof_json TEXT NOT NULL,
            PRIMARY KEY(event_id, batch_id)
        )
        """)

        self.conn.commit()

    def version_projection(self, org_id, ontology_version, semantic_projection_input, projection_output):
        if hosted.is_postgres(self.conn):
            hosted.begin_write(self.conn)
        cursor = self.conn.execute(
            hosted.sql_query("SELECT MAX(projection_version) FROM projection_versions WHERE org_id=? AND ontology_version=?", 'SELECT MAX(projection_version) FROM legacy_projection_versions WHERE org_id=%s AND ontology_version=%s'),
            (org_id, ontology_version)
        )
        row = cursor.fetchone()
        next_v = (row[0] or 0) + 1

        self.conn.execute(hosted.sql_query("""
        INSERT INTO projection_versions (
            org_id, ontology_version, projection_version,
            semantic_projection_input_json,
            semantic_scope_json,
            bindings_json,
            constraint_layer_json,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, '\n        INSERT INTO legacy_projection_versions (\n            org_id, ontology_version, projection_version,\n            semantic_projection_input_json,\n            semantic_scope_json,\n            bindings_json,\n            constraint_layer_json,\n            created_at\n        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)\n        '), (
            org_id,
            ontology_version,
            next_v,
            json.dumps(semantic_projection_input),
            json.dumps(projection_output.get("semantic_scope")),
            json.dumps(projection_output.get("bindings")),
            json.dumps(projection_output.get("constraint_layer")),
            datetime.now(timezone.utc).isoformat()
        ))

        self.conn.commit()

        return next_v

    def get_projection(self, org_id, ontology_version, projection_version):
        cursor = self.conn.execute(
            hosted.sql_query("""
            SELECT semantic_scope_json, bindings_json, constraint_layer_json
            FROM projection_versions
            WHERE org_id=? AND ontology_version=? AND projection_version=?
            """, '\n            SELECT semantic_scope_json, bindings_json, constraint_layer_json\n            FROM legacy_projection_versions\n            WHERE org_id=%s AND ontology_version=%s AND projection_version=%s\n            '),
            (org_id, ontology_version, projection_version)
        )
        row = cursor.fetchone()
        if not row:
            return None

        return {
            "semantic_scope": json.loads(row[0]),
            "bindings": json.loads(row[1]),
            "constraint_layer": json.loads(row[2])
        }

    def append_event(self, event):
        if hosted.is_postgres(self.conn):
            hosted.begin_write(self.conn)
        payload = {
            "org_id": event["org_id"],
            "ontology_version": event["ontology_version"],
            "projection_version": event["projection_version"],
            "policy": event.get("policy", {}),
            "justification": event["justification"],
            "execution": event["execution"],
            "effects": event["effects"]
        }

        event_content_hash = content_hash(payload)

        # determine prev_event_hash for chain integrity if not provided
        prev_hash = event.get("prev_event_hash")
        if not prev_hash:
            cur = self.conn.execute(
                hosted.sql_query("SELECT content_hash FROM semantier_events WHERE org_id=? ORDER BY created_at DESC LIMIT 1", 'SELECT content_hash FROM semantier_events WHERE org_id=%s ORDER BY created_at DESC LIMIT 1'),
                (event["org_id"],)
            )
            row = cur.fetchone()
            prev_hash = row[0] if row and row[0] else None

        # compute signature over content_hash when signer configured
        signature = event.get("signature")
        if not signature:
            sig = signer.sign(event_content_hash)
            if sig:
                signature = sig

        self.conn.execute(hosted.sql_query("""
        INSERT INTO semantier_events (
            event_id, org_id,
            intent_id, signal_ref, workflow_ref, rea_ref,
            ontology_version, projection_version,
            policy_json,
            justification_json, execution_json, effects_json,
            content_hash, prev_event_hash, signature, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, '\n        INSERT INTO semantier_events (\n            event_id, org_id,\n            intent_id, signal_ref, workflow_ref, rea_ref,\n            ontology_version, projection_version,\n            policy_json,\n            justification_json, execution_json, effects_json,\n            content_hash, prev_event_hash, signature, created_at\n        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n        '), (
            event["event_id"],
            event["org_id"],
            event.get("intent_id"),
            event.get("signal_ref"),
            event.get("workflow_ref"),
            event.get("rea_ref"),
            event["ontology_version"],
            event["projection_version"],
            json.dumps(event.get("policy", {})),
            json.dumps(event["justification"]),
            json.dumps(event["execution"]),
            json.dumps(event["effects"]),
            event_content_hash,
            prev_hash,
            signature,
            datetime.now(timezone.utc).isoformat()
        ))

        self.conn.commit()

        return {**event, "content_hash": event_content_hash, "prev_event_hash": prev_hash, "signature": signature}

    # --- Merkle batch helpers ---
    def create_merkle_batch(self, batch_id: str, event_ids: list[str]):
        if hosted.is_postgres(self.conn):
            hosted.begin_write(self.conn)
        # fetch content_hashes in order
        hashes = []
        for eid in event_ids:
            e = self.get_event(eid)
            if not e:
                raise ValueError(f"event not found: {eid}")
            hashes.append(e["content_hash"])

        import eos.merkle as merkle_mod
        root, proof_map = merkle_mod.build_merkle_tree(hashes)

        self.conn.execute(
            hosted.sql_query("INSERT OR REPLACE INTO merkle_batches (batch_id, root_hash, created_at) VALUES (?, ?, ?)", 'INSERT INTO merkle_batches (batch_id, root_hash, created_at) VALUES (%s, %s, %s) ON CONFLICT(batch_id) DO UPDATE SET root_hash=excluded.root_hash,created_at=excluded.created_at'),
            (batch_id, root or "", datetime.now(timezone.utc).isoformat())
        )

        for eid, h in zip(event_ids, hashes):
            proof = proof_map.get(h, [])
            self.conn.execute(
                hosted.sql_query("INSERT OR REPLACE INTO merkle_proofs (event_id, batch_id, proof_json) VALUES (?, ?, ?)", 'INSERT INTO merkle_proofs (event_id, batch_id, proof_json) VALUES (%s, %s, %s) ON CONFLICT(event_id,batch_id) DO UPDATE SET proof_json=excluded.proof_json'),
                (eid, batch_id, json.dumps(proof))
            )

        self.conn.commit()

    def get_merkle_root(self, batch_id: str) -> str | None:
        cur = self.conn.execute(hosted.sql_query("SELECT root_hash FROM merkle_batches WHERE batch_id = ?", 'SELECT root_hash FROM merkle_batches WHERE batch_id = %s'), (batch_id,))
        row = cur.fetchone()
        return row[0] if row else None

    def get_merkle_proof(self, event_id: str, batch_id: str) -> list | None:
        cur = self.conn.execute(hosted.sql_query("SELECT proof_json FROM merkle_proofs WHERE event_id = ? AND batch_id = ?", 'SELECT proof_json FROM merkle_proofs WHERE event_id = %s AND batch_id = %s'), (event_id, batch_id))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def verify_chain(self, org_id: str) -> dict:
        cursor = self.conn.execute(
            hosted.sql_query("""
            SELECT event_id, org_id, ontology_version, projection_version,
                   policy_json, justification_json, execution_json, effects_json,
                   content_hash, prev_event_hash
            FROM semantier_events
            WHERE org_id = ?
            ORDER BY created_at ASC
            """, '\n            SELECT event_id, org_id, ontology_version, projection_version,\n                   policy_json, justification_json, execution_json, effects_json,\n                   content_hash, prev_event_hash\n            FROM semantier_events\n            WHERE org_id = %s\n            ORDER BY created_at ASC\n            '),
            (org_id,),
        )
        rows = cursor.fetchall()

        expected_prev_hash = None
        for row in rows:
            payload = {
                "org_id": row[1],
                "ontology_version": row[2],
                "projection_version": row[3],
                "policy": json.loads(row[4]) if row[4] else {},
                "justification": json.loads(row[5]) if row[5] else {},
                "execution": json.loads(row[6]) if row[6] else {},
                "effects": json.loads(row[7]) if row[7] else [],
            }
            recalculated_hash = content_hash(payload)
            if row[8] != recalculated_hash:
                return {
                    "status": "HASH_MISMATCH",
                    "event_id": row[0],
                    "stored_hash": row[8],
                    "recalculated_hash": recalculated_hash,
                }
            if row[9] != expected_prev_hash:
                return {
                    "status": "CHAIN_BROKEN",
                    "event_id": row[0],
                    "expected_prev_event_hash": expected_prev_hash,
                    "actual_prev_event_hash": row[9],
                }
            expected_prev_hash = row[8]

        return {"status": "OK", "event_count": len(rows)}

    def replay_event(self, event_id):
        event = self.get_event(event_id)
        if not event:
            return None

        proj = self.get_projection(
            event["org_id"],
            event["ontology_version"],
            event["projection_version"]
        )

        return {
            "ontology_version": event["ontology_version"],
            "projection_version": event["projection_version"],
            "policy": event["policy"],
            "semantic_scope": proj["semantic_scope"],
            "bindings": proj["bindings"],
            "constraint_layer": proj["constraint_layer"],
            "justification": event["justification"],
            "execution": event["execution"],
            "effects": event["effects"],
            "content_hash": event["content_hash"],
            "prev_event_hash": event.get("prev_event_hash"),
            "signature": event.get("signature")
        }

    def get_event(self, event_id):
        cursor = self.conn.execute(
            hosted.sql_query("SELECT * FROM semantier_events WHERE event_id = ?", 'SELECT * FROM semantier_events WHERE event_id = %s'),
            (event_id,)
        )

        row = cursor.fetchone()
        if not row:
            return None

        return {
            "event_id": row[0],
            "org_id": row[1],
            "intent_id": row[2],
            "signal_ref": row[3],
            "workflow_ref": row[4],
            "rea_ref": row[5],
            "ontology_version": row[6],
            "projection_version": row[7],
            "policy": json.loads(row[8]) if row[8] else {},
            "justification": json.loads(row[9]) if row[9] else {},
            "execution": json.loads(row[10]) if row[10] else {},
            "effects": json.loads(row[11]) if row[11] else [],
            "content_hash": row[12],
            "prev_event_hash": row[13],
            "signature": row[14],
            "created_at": row[15]
        }

