from storage import hosted_postgres as hosted
import json
import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from runtime_paths import sqlite_db_path

DB_PATH = sqlite_db_path("context.db", ".semantier/context.db", "context.db")


def _get_conn():
	if hosted.enabled():
		raise hosted.HostedStorageUnavailable("legacy context owner requires explicit hosted migration")
	DB_PATH.parent.mkdir(parents=True, exist_ok=True)
	conn = sqlite3.connect(str(DB_PATH))
	conn.row_factory = sqlite3.Row
	return conn


def init_db() -> None:
	with _get_conn() as conn:
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS contexts (
				org_id TEXT NOT NULL,
				ctx_hash TEXT NOT NULL,
				context_json TEXT NOT NULL,
				actor_json TEXT NOT NULL,
				signature TEXT,
				created_at TEXT NOT NULL,
				PRIMARY KEY (org_id, ctx_hash)
			)
			"""
		)


class MockSigner:
	def sign(self, ctx_hash: str) -> str:
		return f"signed::{ctx_hash}"


class ContextEnvelope:
	def __init__(self, org_id: str, context: Dict[str, Any], actor: Dict[str, Any]):
		self.org_id = org_id
		self.context = context
		self.actor = actor
		# deterministic fingerprint
		ctx_s = json.dumps(self.context, sort_keys=True, separators=(",", ":"))
		actor_s = json.dumps(self.actor, sort_keys=True, separators=(",", ":"))
		base = f"{self.org_id}|{ctx_s}|{actor_s}"
		self.hash = hashlib.sha256(base.encode("utf-8")).hexdigest()
		self.signature: Optional[str] = None

	def sign(self, signer: Any) -> None:
		self.signature = signer.sign(self.hash)


def store_context(envelope: ContextEnvelope) -> None:
	with _get_conn() as conn:
		conn.execute(
			"INSERT OR REPLACE INTO contexts (org_id, ctx_hash, context_json, actor_json, signature, created_at) VALUES (?, ?, ?, ?, ?, ?)",
			(
				envelope.org_id,
				envelope.hash,
				json.dumps(envelope.context, sort_keys=True),
				json.dumps(envelope.actor, sort_keys=True),
				envelope.signature,
				datetime.now(timezone.utc).isoformat(),
			),
		)


def load_context(org_id: str, ctx_hash: str) -> Optional[Dict[str, Any]]:
	with _get_conn() as conn:
		row = conn.execute(
			"SELECT context_json FROM contexts WHERE org_id = ? AND ctx_hash = ?",
			(org_id, ctx_hash),
		).fetchone()
	if not row:
		return None
	return json.loads(row["context_json"])


def validate_context_against_ontology(context: Dict[str, Any], ontology: Dict[str, Any]) -> Dict[str, Any]:
	# very small validator: accept if product_type present in ontology.entities
	entities = ontology.get("entities") if ontology else None
	if entities and context.get("product_type") in entities:
		return {"status": "ACCEPT"}
	return {"status": "REJECT"}
