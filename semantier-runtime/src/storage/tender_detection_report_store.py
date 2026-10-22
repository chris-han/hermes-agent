from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agents.knowledge_builder_evaluator import compute_anchored_occurrence_metrics
from eos.artifact_hashing import canonical_json, content_hash as compute_content_hash
from eos.db import init_db
from eos.generic_replay_envelope_store import (
    SENSITIVE_REPORT_TYPE,
    GenericReplayEnvelopeStore,
)
from runtime_paths import sqlite_db_path
from storage.hosted_postgres import enabled,connect_authority,require_tables,sql_query,begin_write
from storage.transaction_scope import transaction_scope,StoreMutationLock
from storage.tender_detection_run_store import TenderDetectionRunStore


def _db_path() -> Path:
    return Path(os.environ.get("SEMANTIER_EOS_DB_PATH", str(sqlite_db_path("eos.db", ".semantier/eos.db"))))


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _loads_json(value: str) -> Any:
    return json.loads(value) if value else None


class TenderDetectionReportStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path) if db_path else _db_path()
        if enabled():
            self._conn=connect_authority("eos",path)
            require_tables(self._conn,'tender_detection_reports')
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
        self._db_path = path
        self._run_store = TenderDetectionRunStore(db_path=path)

    def create_report(self, *, run_id: str, created_by: str) -> dict[str, Any]:
        run = self._run_store.get_run(run_id)
        if run is None:
            raise ValueError(f"tender detection run not found: {run_id}")
        report_id = f"tdrep_{uuid4().hex}"
        replay_binding_id = f"sensitive_labeling_report:{report_id}"
        prior_report_id = self._latest_report_id(run_id)
        report = self._build_report_payload(
            report_id=report_id,
            run=run,
            prior_labeling_report_id=prior_report_id,
        )
        replay_binding = self._create_report_replay_binding(
            replay_binding_id=replay_binding_id,
            report=report,
            run=run,
        )
        now = _now_iso()
        row = {
            "report_id": report_id,
            "run_id": run_id,
            "organization_id": run["organization_id"],
            "report_json": canonical_json(report),
            "replay_event_id": replay_binding_id,
            "replay_binding_ref": replay_binding_id,
            "prior_labeling_report_id": prior_report_id,
            "created_by": created_by,
            "created_at": now,
            "content_hash": compute_content_hash(canonical_json(report)),
        }
        with self._lock:
            self._conn.execute(
                sql_query("""
                INSERT INTO tender_detection_reports (
                  report_id, run_id, organization_id, report_json, replay_event_id,
                  replay_binding_ref, prior_labeling_report_id, created_by, created_at,
                  content_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO tender_detection_reports (\n                  report_id, run_id, organization_id, report_json, replay_event_id,\n                  replay_binding_ref, prior_labeling_report_id, created_by, created_at,\n                  content_hash\n                )\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    row["report_id"],
                    row["run_id"],
                    row["organization_id"],
                    row["report_json"],
                    row["replay_event_id"],
                    row["replay_binding_ref"],
                    row["prior_labeling_report_id"],
                    row["created_by"],
                    row["created_at"],
                    row["content_hash"],
                ),
            )
            self._conn.commit()
        return {**row, "report": report, "replay_binding": replay_binding}

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            sql_query("SELECT * FROM tender_detection_reports WHERE report_id = ?", 'SELECT * FROM tender_detection_reports WHERE report_id = %s'),
            (report_id,),
        ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["report"] = _loads_json(payload["report_json"])
        payload["replay_binding"] = GenericReplayEnvelopeStore(
            db_path=self._db_path
        ).verify_typed_replay_binding(
            replay_binding_id=payload["replay_binding_ref"],
            expected_binding_type=SENSITIVE_REPORT_TYPE,
            expected_organization_id=payload["organization_id"],
        )
        return payload

    def build_replay_bundle(
        self,
        *,
        run_id: str,
        include_audit_details: bool,
    ) -> dict[str, Any]:
        run = self._run_store.get_run(run_id)
        if run is None:
            raise ValueError(f"tender detection run not found: {run_id}")
        reports = self._conn.execute(
            sql_query("""
            SELECT * FROM tender_detection_reports
            WHERE run_id = ?
            ORDER BY created_at DESC
            """, '\n            SELECT * FROM tender_detection_reports\n            WHERE run_id = %s\n            ORDER BY created_at DESC\n            '),
            (run_id,),
        ).fetchall()
        report_payloads = []
        replay_bindings = []
        for row in reports:
            report = dict(row)
            report["report"] = _loads_json(report["report_json"])
            report_payloads.append(report)
            binding = GenericReplayEnvelopeStore(db_path=self._db_path).verify_typed_replay_binding(
                replay_binding_id=report["replay_binding_ref"],
                expected_binding_type=SENSITIVE_REPORT_TYPE,
                expected_organization_id=run["organization_id"],
                expected_workspace_id=run.get("workspace_id"),
            )
            if binding.get("status") == "OK":
                # Preserve the public replay-bundle shape used before typed
                # replay envelopes while retaining the verified envelope and
                # its nested payload for new consumers.
                projected_binding = {**binding, **(binding.get("payload") or {})}
                projected_binding.setdefault(
                    "projection_context_ref",
                    projected_binding.get("activation_set_snapshot_id"),
                )
                projected_binding.setdefault(
                    "projection_context_hash",
                    projected_binding.get("resolved_activation_set_hash"),
                )
                replay_bindings.append(projected_binding)
        bundle = {
            "run": self._redact_run(run, include_audit_details=include_audit_details),
            "reports": report_payloads,
            "replay_bindings": replay_bindings if include_audit_details else [],
            "replay_source": "persisted_tender_detection_tables",
        }
        bundle["bundle_hash"] = compute_content_hash(canonical_json(bundle))
        return bundle

    def _latest_report_id(self, run_id: str) -> str | None:
        row = self._conn.execute(
            sql_query("""
            SELECT report_id FROM tender_detection_reports
            WHERE run_id = ?
            ORDER BY created_at DESC, report_id DESC
            LIMIT 1
            """, '\n            SELECT report_id FROM tender_detection_reports\n            WHERE run_id = %s\n            ORDER BY created_at DESC, report_id DESC\n            LIMIT 1\n            '),
            (run_id,),
        ).fetchone()
        return str(row["report_id"]) if row else None

    def _create_report_replay_binding(
        self,
        *,
        replay_binding_id: str,
        report: dict[str, Any],
        run: dict[str, Any],
    ) -> dict[str, Any]:
        project_metadata = run.get("project_metadata") or {}
        payload = {
            "report_id": report["report_id"],
            "prior_labeling_report_id": report.get("prior_labeling_report_id"),
            "detection_run_id": run["run_id"],
            "detection_replay_binding_ref": f"sensitive_labeling_detection:{run['run_id']}",
            "knowledge_source_compilation_run_id": project_metadata.get(
                "knowledge_source_compilation_run_id"
            ),
            "knowledge_source_id": project_metadata.get("knowledge_source_id"),
            "knowledge_source_hash": project_metadata.get("knowledge_source_hash"),
            "evaluation_run_id": project_metadata.get("evaluation_run_id"),
            "evaluation_dataset_id": project_metadata.get("evaluation_dataset_id"),
            "gold_label_version": project_metadata.get("gold_label_version"),
            "target_document_hash": run["source_document_hash"],
            "target_source_hash": project_metadata.get("target_source_hash"),
            "target_normalized_document_artifact_ref": project_metadata.get(
                "target_normalized_document_artifact_ref"
            ),
            "target_normalized_document_artifact_hash": project_metadata.get(
                "target_normalized_document_artifact_hash"
            ),
            "activation_set_snapshot_id": run["activation_set_snapshot_id"],
            "resolved_activation_set_hash": run["resolved_activation_set_hash"],
            "rule_version_refs": project_metadata.get("rule_version_refs") or [],
            "rule_version_set_hash": project_metadata.get("rule_version_set_hash"),
            "semantic_profile_ids": project_metadata.get("semantic_profile_ids") or [],
            "finding_refs": [
                {
                    "finding_id": finding["finding_id"],
                    "content_hash": finding["content_hash"],
                    "target_anchor_ref": finding.get("target_anchor_ref"),
                    "start_offset": finding.get("start_offset"),
                    "end_offset": finding.get("end_offset"),
                    "triggered_rule_version_refs": finding.get(
                        "triggered_rule_version_refs"
                    )
                    or [],
                    "source_anchor_refs": finding.get("source_anchor_refs") or [],
                }
                for finding in run.get("findings", [])
            ],
            "finding_set_hash": compute_content_hash(
                canonical_json(
                    [
                        {
                            "finding_id": finding["finding_id"],
                            "content_hash": finding["content_hash"],
                        }
                        for finding in run.get("findings", [])
                    ]
                )
            ),
            "metrics_hash": compute_content_hash(report.get("metrics") or {}),
            "metric_implementation_version": (
                (report.get("metrics") or {}).get("metric_schema_version")
                or "count_metrics.v1"
            ),
            "lineage_graph_hash": compute_content_hash(report.get("lineage") or {}),
            "report_content_hash": compute_content_hash(canonical_json(report)),
            "report_schema_version": "sensitive_labeling_report.v1",
        }
        return GenericReplayEnvelopeStore(db_path=self._db_path).create_typed_replay_binding(
            replay_binding_id=replay_binding_id,
            binding_type=SENSITIVE_REPORT_TYPE,
            binding_schema_version="sensitive_labeling_report_binding.v1",
            subject_type="sensitive_labeling_report",
            subject_id=report["report_id"],
            organization_id=run["organization_id"],
            workspace_id=run.get("workspace_id"),
            payload=payload,
        )

    def _build_report_payload(
        self,
        *,
        report_id: str,
        run: dict[str, Any],
        prior_labeling_report_id: str | None,
    ) -> dict[str, Any]:
        dispositions_by_finding = {
            disposition["finding_id"]: disposition for disposition in run["dispositions"]
        }
        open_issues = []
        resolved_issues = []
        modification_history = []
        for finding in run["findings"]:
            disposition = dispositions_by_finding.get(finding["finding_id"])
            item = {
                "finding_id": finding["finding_id"],
                "issue_type": finding["issue_type"],
                "matched_text": finding["matched_text"],
                "judgment_basis": finding["judgment_basis"],
                "source_anchor_refs": finding["source_anchor_refs"],
                "triggered_rule_version_id": finding["triggered_rule_version_id"],
                "resolver_evidence_ref": finding["resolver_evidence_ref"],
                "suggested_replacement": finding["suggested_replacement"],
                "suggestion_label": "AI-assisted recommendation",
                "final_user_decision": disposition["disposition"] if disposition else "open",
            }
            if disposition:
                resolved_issues.append(item)
                modification_history.append(
                    {
                        "finding_id": finding["finding_id"],
                        "disposition": disposition["disposition"],
                        "edited_replacement": disposition["edited_replacement"],
                        "rejection_rationale": disposition["rejection_rationale"],
                        "decided_by": disposition["decided_by"],
                        "decided_at": disposition["decided_at"],
                    }
                )
            else:
                open_issues.append(item)
        project_metadata = run.get("project_metadata") or {}
        gold_labels = project_metadata.get("gold_labels") or []
        metrics = (
            compute_anchored_occurrence_metrics(
                target_document_hash=run["source_document_hash"],
                findings=run["findings"],
                gold_labels=gold_labels,
                dispositions=run["dispositions"],
            )
            if isinstance(gold_labels, list) and gold_labels
            else {
                "finding_count": len(run["findings"]),
                "open_issue_count": len(open_issues),
                "resolved_issue_count": len(resolved_issues),
            }
        )
        return {
            "report_id": report_id,
            "run_id": run["run_id"],
            "tender_document_id": run["tender_document_id"],
            "source_document_hash": run["source_document_hash"],
            "open_issues": open_issues,
            "resolved_issues": resolved_issues,
            "modification_history": modification_history,
            "rule_versions": sorted(
                {finding["triggered_rule_version_id"] for finding in run["findings"]}
            ),
            "lineage": {
                "finding_to_rule_refs": [
                    {
                        "finding_id": finding["finding_id"],
                        "triggered_rule_version_refs": finding.get(
                            "triggered_rule_version_refs"
                        )
                        or [
                            {
                                "rule_version_id": finding[
                                    "triggered_rule_version_id"
                                ]
                            }
                        ],
                    }
                    for finding in run["findings"]
                ],
                "finding_to_source_anchor_refs": [
                    {
                        "finding_id": finding["finding_id"],
                        "source_anchor_refs": finding.get("source_anchor_refs") or [],
                    }
                    for finding in run["findings"]
                ],
            },
            "metrics": metrics,
            "resolver_snapshot": {
                "activation_set_snapshot_id": run["activation_set_snapshot_id"],
                "activation_resolver_policy_version": run[
                    "activation_resolver_policy_version"
                ],
                "resolved_activation_set_hash": run["resolved_activation_set_hash"],
            },
            "rerun_linkage": {
                "parent_run_id": run["parent_run_id"],
                "root_run_id": run["root_run_id"],
            },
            "prior_labeling_report_id": prior_labeling_report_id,
        }

    def _redact_run(self, run: dict[str, Any], *, include_audit_details: bool) -> dict[str, Any]:
        if include_audit_details:
            return run
        redacted = dict(run)
        redacted.pop("activation_set_snapshot_id", None)
        redacted.pop("activation_resolver_policy_version", None)
        redacted.pop("resolved_activation_set_hash", None)
        for finding in redacted.get("findings", []):
            finding.pop("triggered_rule_version_id", None)
            finding.pop("source_anchor_refs", None)
            finding.pop("resolver_evidence_ref", None)
        return redacted
