from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agents.knowledge_builder_evaluator import (
    EVALUATION_CASE_TYPES,
    EVALUATION_OUTCOMES,
    compute_metrics,
    evaluate_example,
)
from eos.artifact_hashing import canonical_json, content_hash as compute_content_hash
from eos.db import init_db
from runtime_paths import sqlite_db_path
from storage.hosted_postgres import enabled,connect_authority,require_tables,sql_query,begin_write
from storage.transaction_scope import transaction_scope,StoreMutationLock
from storage.knowledge_builder_store import KnowledgeBuilderStore


def _db_path() -> Path:
    return Path(os.environ.get("SEMANTIER_EOS_DB_PATH", str(sqlite_db_path("eos.db", ".semantier/eos.db"))))


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_json(value: Any) -> str:
    return canonical_json(value if value is not None else {})


def _loads_json(value: str) -> Any:
    return json.loads(value) if value else None


def _hydrate(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    for field in (
        "example_ids_json",
        "expected_runtime_effect_json",
        "source_anchor_refs_json",
        "metrics_json",
        "matched_path_json",
        "excluded_path_json",
        "error_labels_json",
    ):
        if field in payload:
            payload[field.removesuffix("_json")] = _loads_json(payload[field])
    return payload


class KnowledgeBuilderEvaluationStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path) if db_path else _db_path()
        if enabled():
            self._conn=connect_authority("eos",path)
            require_tables(self._conn,'knowledge_builder_evaluation_datasets','knowledge_builder_evaluation_examples','knowledge_builder_evaluation_results','knowledge_builder_evaluation_runs')
            self._conn.commit()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            prior_db_path = os.environ.get("SEMANTIER_EOS_DB_PATH")
            os.environ["SEMANTIER_EOS_DB_PATH"] = str(path)
            try:
                init_db()
            finally:
                if prior_db_path is None:
                    os.environ.pop("SEMANTIER_EOS_DB_PATH", None)
                else:
                    os.environ["SEMANTIER_EOS_DB_PATH"] = prior_db_path
            self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = StoreMutationLock(self._conn)
        self._builder = KnowledgeBuilderStore(db_path=path)

    def create_dataset(
        self,
        *,
        organization_id: str,
        workspace_id: str | None,
        discovery_run_id: str | None,
        domain: str,
        name: str,
        description: str,
        examples: list[dict[str, Any]],
        created_by: str,
    ) -> dict[str, Any]:
        if discovery_run_id and self._builder.get_discovery_run(discovery_run_id) is None:
            raise ValueError(f"knowledge builder discovery run not found: {discovery_run_id}")
        if not examples:
            raise ValueError("evaluation examples are required")
        dataset_id = f"kbed_{uuid4().hex}"
        now = _now_iso()
        example_rows = [
            self._prepare_example(dataset_id=dataset_id, example=example, created_at=now)
            for example in examples
        ]
        row = {
            "evaluation_dataset_id": dataset_id,
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "discovery_run_id": discovery_run_id,
            "domain": domain,
            "name": name,
            "description": description,
            "example_ids": [example["evaluation_example_id"] for example in example_rows],
            "governance_state": "CURATED",
            "created_by": created_by,
            "created_at": now,
        }
        row["content_hash"] = compute_content_hash(canonical_json(row))
        with self._lock:
            self._conn.execute(
                sql_query("""
                INSERT INTO knowledge_builder_evaluation_datasets (
                  evaluation_dataset_id, organization_id, workspace_id, discovery_run_id,
                  domain, name, description, example_ids_json, governance_state,
                  created_by, created_at, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO knowledge_builder_evaluation_datasets (\n                  evaluation_dataset_id, organization_id, workspace_id, discovery_run_id,\n                  domain, name, description, example_ids_json, governance_state,\n                  created_by, created_at, content_hash\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    dataset_id,
                    organization_id,
                    workspace_id,
                    discovery_run_id,
                    domain,
                    name,
                    description,
                    _as_json(row["example_ids"]),
                    row["governance_state"],
                    created_by,
                    now,
                    row["content_hash"],
                ),
            )
            for example in example_rows:
                self._insert_example(example)
            self._conn.commit()
        dataset = self.get_dataset(dataset_id)
        assert dataset is not None
        return dataset

    def create_tender_uat_dataset(
        self,
        *,
        organization_id: str,
        workspace_id: str | None,
        discovery_run_id: str | None,
        created_by: str,
    ) -> dict[str, Any]:
        examples = [
            {
                "case_type": "positive",
                "input_text": "本项目要求唯一授权经销商参与。",
                "expected_outcome": "match",
                "expected_relation_type": "allowed_context_for",
                "source_anchor_refs": [],
                "reviewer_notes": "Positive restrictive supplier wording.",
            },
            {
                "case_type": "negative",
                "input_text": "投标文件应填写唯一编号。",
                "expected_outcome": "exclude",
                "expected_relation_type": "not_same_as",
                "source_anchor_refs": [],
                "reviewer_notes": "Identifier-only wording is not exclusivity.",
            },
            {
                "case_type": "ambiguous",
                "input_text": "Account may mean platform account or bank account.",
                "expected_outcome": "ambiguous",
                "expected_relation_type": "conflicts_with",
                "source_anchor_refs": [],
                "reviewer_notes": "Ambiguous account neighborhood.",
            },
            {
                "case_type": "known_false_positive",
                "input_text": "唯一编号用于资料归档。",
                "expected_outcome": "exclude",
                "expected_relation_type": "not_same_as",
                "source_anchor_refs": [],
                "reviewer_notes": "Known false positive guard.",
            },
            {
                "case_type": "known_false_negative",
                "input_text": "供应商须提供独家授权证明。",
                "expected_outcome": "match",
                "expected_relation_type": "allowed_context_for",
                "source_anchor_refs": [],
                "reviewer_notes": "Known false negative guard.",
            },
            {
                "case_type": "escalation_required",
                "input_text": "该条款 requires review because context is incomplete.",
                "expected_outcome": "escalate",
                "expected_relation_type": "exception_to",
                "source_anchor_refs": [],
                "reviewer_notes": "Escalation-required case.",
            },
        ]
        return self.create_dataset(
            organization_id=organization_id,
            workspace_id=workspace_id,
            discovery_run_id=discovery_run_id,
            domain="tender_compliance",
            name="Tender semantic neighborhood UAT",
            description="UAT examples for 唯一授权 versus 唯一编号 before promotion.",
            examples=examples,
            created_by=created_by,
        )

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        dataset = _hydrate(
            self._conn.execute(
                sql_query("SELECT * FROM knowledge_builder_evaluation_datasets WHERE evaluation_dataset_id = ?", 'SELECT * FROM knowledge_builder_evaluation_datasets WHERE evaluation_dataset_id = %s'),
                (dataset_id,),
            ).fetchone()
        )
        if dataset is None:
            return None
        dataset["examples"] = self.list_examples(dataset_id)
        return dataset

    def list_examples(self, dataset_id: str) -> list[dict[str, Any]]:
        return [
            _hydrate(row)
            for row in self._conn.execute(
                sql_query("""
                SELECT * FROM knowledge_builder_evaluation_examples
                WHERE evaluation_dataset_id = ?
                ORDER BY created_at, evaluation_example_id
                """, '\n                SELECT * FROM knowledge_builder_evaluation_examples\n                WHERE evaluation_dataset_id = %s\n                ORDER BY created_at, evaluation_example_id\n                '),
                (dataset_id,),
            ).fetchall()
        ]

    def create_evaluation_run(
        self,
        *,
        evaluation_dataset_id: str,
        discovery_run_id: str | None,
        created_by: str,
    ) -> dict[str, Any]:
        dataset = self.get_dataset(evaluation_dataset_id)
        if dataset is None:
            raise ValueError(f"evaluation dataset not found: {evaluation_dataset_id}")
        run_id = f"kber_{uuid4().hex}"
        now = _now_iso()
        candidate_snapshot = {"evaluation_dataset_id": evaluation_dataset_id}
        if discovery_run_id:
            run = self._builder.get_discovery_run(discovery_run_id)
            if run is None:
                raise ValueError(f"knowledge builder discovery run not found: {discovery_run_id}")
            candidate_snapshot = {"run": run, "source": self._builder.get_source(run["source_id"])}
        elif dataset.get("discovery_run_id"):
            discovery_run_id = str(dataset["discovery_run_id"])
            run = self._builder.get_discovery_run(discovery_run_id)
            if run is None:
                raise ValueError(f"knowledge builder discovery run not found: {discovery_run_id}")
            candidate_snapshot = {"run": run, "source": self._builder.get_source(run["source_id"])}
        snapshot_hash = compute_content_hash(canonical_json(candidate_snapshot))
        result_rows = []
        for example in dataset["examples"]:
            result_rows.append(
                self._prepare_result(
                    evaluation_run_id=run_id,
                    example=example,
                    evaluated=evaluate_example(example=example, graph=None),
                    created_at=now,
                )
            )
        metrics = compute_metrics(result_rows)
        row = {
            "evaluation_run_id": run_id,
            "evaluation_dataset_id": evaluation_dataset_id,
            "discovery_run_id": discovery_run_id,
            "candidate_snapshot_hash": snapshot_hash,
            "run_status": "completed",
            "metrics": metrics,
            "authority_notice": "evaluation results are non-authoritative until governed promotion and activation",
            "created_by": created_by,
            "started_at": now,
            "completed_at": now,
        }
        row["content_hash"] = compute_content_hash(canonical_json(row))
        with self._lock:
            self._conn.execute(
                sql_query("""
                INSERT INTO knowledge_builder_evaluation_runs (
                  evaluation_run_id, evaluation_dataset_id, discovery_run_id,
                  candidate_snapshot_hash, run_status, metrics_json, authority_notice,
                  created_by, started_at, completed_at, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO knowledge_builder_evaluation_runs (\n                  evaluation_run_id, evaluation_dataset_id, discovery_run_id,\n                  candidate_snapshot_hash, run_status, metrics_json, authority_notice,\n                  created_by, started_at, completed_at, content_hash\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    run_id,
                    evaluation_dataset_id,
                    discovery_run_id,
                    snapshot_hash,
                    "completed",
                    _as_json(metrics),
                    row["authority_notice"],
                    created_by,
                    now,
                    now,
                    row["content_hash"],
                ),
            )
            for result in result_rows:
                self._insert_result(result)
            self._conn.commit()
        evaluation = self.get_evaluation_run(run_id)
        assert evaluation is not None
        return evaluation

    def get_evaluation_run(self, evaluation_run_id: str) -> dict[str, Any] | None:
        run = _hydrate(
            self._conn.execute(
                sql_query("SELECT * FROM knowledge_builder_evaluation_runs WHERE evaluation_run_id = ?", 'SELECT * FROM knowledge_builder_evaluation_runs WHERE evaluation_run_id = %s'),
                (evaluation_run_id,),
            ).fetchone()
        )
        if run is None:
            return None
        run["results"] = [
            _hydrate(row)
            for row in self._conn.execute(
                sql_query("""
                SELECT * FROM knowledge_builder_evaluation_results
                WHERE evaluation_run_id = ?
                ORDER BY created_at, evaluation_result_id
                """, '\n                SELECT * FROM knowledge_builder_evaluation_results\n                WHERE evaluation_run_id = %s\n                ORDER BY created_at, evaluation_result_id\n                '),
                (evaluation_run_id,),
            ).fetchall()
        ]
        return run

    def get_evaluation_run_scoped(
        self,
        evaluation_run_id: str,
        *,
        organization_id: str,
        workspace_id: str | None,
    ) -> dict[str, Any] | None:
        owner = self._conn.execute(
            sql_query("""
            SELECT d.organization_id, d.workspace_id
            FROM knowledge_builder_evaluation_runs r
            JOIN knowledge_builder_evaluation_datasets d
              ON d.evaluation_dataset_id = r.evaluation_dataset_id
            WHERE r.evaluation_run_id = ?
            """, '\n            SELECT d.organization_id, d.workspace_id\n            FROM knowledge_builder_evaluation_runs r\n            JOIN knowledge_builder_evaluation_datasets d\n              ON d.evaluation_dataset_id = r.evaluation_dataset_id\n            WHERE r.evaluation_run_id = %s\n            '),
            (evaluation_run_id,),
        ).fetchone()
        if owner is None:
            return None
        if owner["organization_id"] != organization_id or owner["workspace_id"] != workspace_id:
            return None
        return self.get_evaluation_run(evaluation_run_id)

    def update_result_rating(
        self,
        *,
        evaluation_result_id: str,
        human_rating: str,
        explanation_acceptance: str,
        error_labels: list[str],
        reviewer_notes: str | None,
    ) -> dict[str, Any]:
        if human_rating not in {"pass", "fail", "needs_review"}:
            raise ValueError("human_rating must be pass, fail, or needs_review")
        if explanation_acceptance not in {"accepted", "rejected", "needs_review"}:
            raise ValueError("explanation_acceptance must be accepted, rejected, or needs_review")
        existing = _hydrate(
            self._conn.execute(
                sql_query("SELECT * FROM knowledge_builder_evaluation_results WHERE evaluation_result_id = ?", 'SELECT * FROM knowledge_builder_evaluation_results WHERE evaluation_result_id = %s'),
                (evaluation_result_id,),
            ).fetchone()
        )
        if existing is None:
            raise ValueError(f"evaluation result not found: {evaluation_result_id}")
        now = _now_iso()
        content = {
            **existing,
            "human_rating": human_rating,
            "explanation_acceptance": explanation_acceptance,
            "error_labels": error_labels,
            "reviewer_notes": reviewer_notes,
            "updated_at": now,
        }
        content_hash = compute_content_hash(canonical_json(content))
        with self._lock:
            self._conn.execute(
                sql_query("""
                UPDATE knowledge_builder_evaluation_results
                SET human_rating = ?, explanation_acceptance = ?, error_labels_json = ?,
                    reviewer_notes = ?, updated_at = ?, content_hash = ?
                WHERE evaluation_result_id = ?
                """, '\n                UPDATE knowledge_builder_evaluation_results\n                SET human_rating = %s, explanation_acceptance = %s, error_labels_json = %s,\n                    reviewer_notes = %s, updated_at = %s, content_hash = %s\n                WHERE evaluation_result_id = %s\n                '),
                (
                    human_rating,
                    explanation_acceptance,
                    _as_json(error_labels),
                    reviewer_notes,
                    now,
                    content_hash,
                    evaluation_result_id,
                ),
            )
            self._conn.commit()
        return _hydrate(
            self._conn.execute(
                sql_query("SELECT * FROM knowledge_builder_evaluation_results WHERE evaluation_result_id = ?", 'SELECT * FROM knowledge_builder_evaluation_results WHERE evaluation_result_id = %s'),
                (evaluation_result_id,),
            ).fetchone()
        )

    def _prepare_example(
        self,
        *,
        dataset_id: str,
        example: dict[str, Any],
        created_at: str,
    ) -> dict[str, Any]:
        case_type = str(example.get("case_type") or example.get("caseType") or "")
        expected_outcome = str(example.get("expected_outcome") or example.get("expectedOutcome") or "")
        if case_type not in EVALUATION_CASE_TYPES:
            raise ValueError("unsupported evaluation case_type")
        if expected_outcome not in EVALUATION_OUTCOMES:
            raise ValueError("unsupported expected_outcome")
        row = {
            "evaluation_example_id": f"kbee_{uuid4().hex}",
            "evaluation_dataset_id": dataset_id,
            "case_type": case_type,
            "input_text": str(example.get("input_text") or example.get("inputText") or ""),
            "expected_outcome": expected_outcome,
            "expected_relation_type": example.get("expected_relation_type") or example.get("expectedRelationType"),
            "expected_runtime_effect": example.get("expected_runtime_effect") or example.get("expectedRuntimeEffect") or {},
            "source_anchor_refs": example.get("source_anchor_refs") or example.get("sourceAnchorRefs") or [],
            "reviewer_notes": example.get("reviewer_notes") or example.get("reviewerNotes"),
            "created_at": created_at,
        }
        row["content_hash"] = compute_content_hash(canonical_json(row))
        return row

    def _prepare_result(
        self,
        *,
        evaluation_run_id: str,
        example: dict[str, Any],
        evaluated: dict[str, Any],
        created_at: str,
    ) -> dict[str, Any]:
        row = {
            "evaluation_result_id": f"kberes_{uuid4().hex}",
            "evaluation_run_id": evaluation_run_id,
            "evaluation_example_id": example["evaluation_example_id"],
            "expected_outcome": example["expected_outcome"],
            "actual_outcome": evaluated["actual_outcome"],
            "matched_path": evaluated["matched_path"],
            "excluded_path": evaluated["excluded_path"],
            "human_rating": None,
            "ai_assisted_rating": evaluated["ai_assisted_rating"],
            "explanation_acceptance": evaluated["explanation_acceptance"],
            "error_labels": evaluated["error_labels"],
            "reviewer_notes": None,
            "created_at": created_at,
            "updated_at": created_at,
        }
        row["content_hash"] = compute_content_hash(canonical_json(row))
        return row

    def _insert_example(self, row: dict[str, Any]) -> None:
        self._conn.execute(
            sql_query("""
            INSERT INTO knowledge_builder_evaluation_examples (
              evaluation_example_id, evaluation_dataset_id, case_type, input_text,
              expected_outcome, expected_relation_type, expected_runtime_effect_json,
              source_anchor_refs_json, reviewer_notes, created_at, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, '\n            INSERT INTO knowledge_builder_evaluation_examples (\n              evaluation_example_id, evaluation_dataset_id, case_type, input_text,\n              expected_outcome, expected_relation_type, expected_runtime_effect_json,\n              source_anchor_refs_json, reviewer_notes, created_at, content_hash\n            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n            '),
            (
                row["evaluation_example_id"],
                row["evaluation_dataset_id"],
                row["case_type"],
                row["input_text"],
                row["expected_outcome"],
                row["expected_relation_type"],
                _as_json(row["expected_runtime_effect"]),
                _as_json(row["source_anchor_refs"]),
                row["reviewer_notes"],
                row["created_at"],
                row["content_hash"],
            ),
        )

    def _insert_result(self, row: dict[str, Any]) -> None:
        self._conn.execute(
            sql_query("""
            INSERT INTO knowledge_builder_evaluation_results (
              evaluation_result_id, evaluation_run_id, evaluation_example_id,
              expected_outcome, actual_outcome, matched_path_json, excluded_path_json,
              human_rating, ai_assisted_rating, explanation_acceptance,
              error_labels_json, reviewer_notes, created_at, updated_at, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, '\n            INSERT INTO knowledge_builder_evaluation_results (\n              evaluation_result_id, evaluation_run_id, evaluation_example_id,\n              expected_outcome, actual_outcome, matched_path_json, excluded_path_json,\n              human_rating, ai_assisted_rating, explanation_acceptance,\n              error_labels_json, reviewer_notes, created_at, updated_at, content_hash\n            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n            '),
            (
                row["evaluation_result_id"],
                row["evaluation_run_id"],
                row["evaluation_example_id"],
                row["expected_outcome"],
                row["actual_outcome"],
                _as_json(row["matched_path"]),
                _as_json(row["excluded_path"]),
                row["human_rating"],
                row["ai_assisted_rating"],
                row["explanation_acceptance"],
                _as_json(row["error_labels"]),
                row["reviewer_notes"],
                row["created_at"],
                row["updated_at"],
                row["content_hash"],
            ),
        )
