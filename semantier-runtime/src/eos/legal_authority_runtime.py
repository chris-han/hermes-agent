"""Deterministic legal authority runtime over activated typed artifacts."""

from __future__ import annotations

from storage import hosted_postgres as hosted

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eos.artifact_hashing import content_hash
from eos.db import get_conn, now_iso
from eos.legal_corpus_store import LegalCorpusStore


LEGAL_RUNTIME_NO_ACTIVE_BUNDLE = "LEGAL_RUNTIME_NO_ACTIVE_BUNDLE"
LEGAL_RUNTIME_NO_PINNED_CLAIMS = "LEGAL_RUNTIME_NO_PINNED_CLAIMS"
LEGAL_RUNTIME_NO_APPLICABLE_AUTHORITY = "LEGAL_RUNTIME_NO_APPLICABLE_AUTHORITY"
LEGAL_RUNTIME_CONFLICTING_AUTHORITY = "LEGAL_RUNTIME_CONFLICTING_AUTHORITY"
LEGAL_RUNTIME_MISSING_FACTS = "LEGAL_RUNTIME_MISSING_FACTS"
LEGAL_RUNTIME_REPLAY_PIN_MISSING = "LEGAL_RUNTIME_REPLAY_PIN_MISSING"
LEGAL_RUNTIME_RENDERER_CONTRACT_UNSUPPORTED = "LEGAL_RUNTIME_RENDERER_CONTRACT_UNSUPPORTED"
LEGAL_RUNTIME_NO_ACTIVE_ISSUE_MODEL = "LEGAL_RUNTIME_NO_ACTIVE_ISSUE_MODEL"
LEGAL_RUNTIME_UNRESOLVED_MATERIAL_ISSUES = "LEGAL_RUNTIME_UNRESOLVED_MATERIAL_ISSUES"

LEGAL_RUNTIME_REF = "legal_authority_runtime.v1"
LEGAL_RESOLUTION_RULES_VERSION = "legal_resolution_rules.v1"
LEGAL_RELEASE_POLICY_VERSION = "legal_release_policy.v1"

LEGAL_DISPOSITIONS = {
    "REQUIRED",
    "NOT_REQUIRED",
    "CONDITIONAL",
    "MISSING_FACTS",
    "UNRESOLVED",
    "ESCALATE",
}

LEGAL_RELEASE_DECISIONS = {
    "ALLOW_DEFINITIVE",
    "ALLOW_CONDITIONAL",
    "REQUIRE_CLARIFICATION",
    "ESCALATE",
    "BLOCK",
}

SUPPORTED_RENDERER_CONTRACTS = {"legal_renderer_contract.v1"}


@dataclass(frozen=True)
class NormalizedFactSet:
    facts: dict[str, Any]
    fact_states: dict[str, str]
    provenance: dict[str, Any]
    fact_set_hash: str


@dataclass(frozen=True)
class LegalQueryIR:
    operation: str
    issue_model_id: str
    jurisdiction: str
    as_of_date: str
    requested_remedy: str | None
    fact_set_hash: str
    governed_defaults: dict[str, Any]
    query_hash: str


class LegalAuthorityRuntime:
    """Resolve legal advice from activated claim candidates and exact pins only."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else None

    def execute(
        self,
        *,
        organization_id: str,
        operation: str,
        issue_model_id: str,
        jurisdiction: str,
        as_of_date: str,
        facts: dict[str, Any],
        requested_remedy: str | None = None,
        governed_defaults: dict[str, Any] | None = None,
        renderer_contract_version: str = "legal_renderer_contract.v1",
    ) -> dict[str, Any]:
        fact_set = normalize_fact_set(facts=facts, provenance={"organization_id": organization_id})
        query_ir = build_query_ir(
            operation=operation,
            issue_model_id=issue_model_id,
            jurisdiction=jurisdiction,
            as_of_date=as_of_date,
            requested_remedy=requested_remedy,
            fact_set=fact_set,
            governed_defaults=governed_defaults or {},
        )
        if renderer_contract_version not in SUPPORTED_RENDERER_CONTRACTS:
            return self._closed_result(
                query_ir=query_ir,
                fact_set=fact_set,
                reason_code=LEGAL_RUNTIME_RENDERER_CONTRACT_UNSUPPORTED,
                release_decision="BLOCK",
                disposition="ESCALATE",
                renderer_contract_version=renderer_contract_version,
            )
        pins = self._active_pins(organization_id=organization_id)
        if pins is None:
            return self._closed_result(
                query_ir=query_ir,
                fact_set=fact_set,
                reason_code=LEGAL_RUNTIME_NO_ACTIVE_BUNDLE,
                release_decision="BLOCK",
                disposition="ESCALATE",
            )
        issue_model = self._active_issue_model(
            organization_id=organization_id,
            issue_model_id=issue_model_id,
            as_of_date=as_of_date,
        )
        if issue_model is None:
            return self._closed_result(
                query_ir=query_ir,
                fact_set=fact_set,
                pins=pins,
                reason_code=LEGAL_RUNTIME_NO_ACTIVE_ISSUE_MODEL,
                release_decision="BLOCK",
                disposition="ESCALATE",
            )
        pins = {**pins, **issue_model}
        result = self._execute_with_pins(
            query_ir=query_ir,
            fact_set=fact_set,
            pins=pins,
            renderer_contract_version=renderer_contract_version,
        )
        self._record_execution_manifest(result["legal_execution_manifest"])
        return result

    def _execute_with_pins(
        self,
        *,
        query_ir: LegalQueryIR,
        fact_set: NormalizedFactSet,
        pins: dict[str, Any],
        renderer_contract_version: str,
    ) -> dict[str, Any]:
        claims = self._pinned_claims(pins=pins, query_ir=query_ir)
        if not claims:
            return self._closed_result(
                query_ir=query_ir,
                fact_set=fact_set,
                pins=pins,
                reason_code=LEGAL_RUNTIME_NO_APPLICABLE_AUTHORITY,
                release_decision="ESCALATE",
                disposition="ESCALATE",
            )

        missing = _missing_material_facts(claims, fact_set.facts)
        if missing:
            return self._closed_result(
                query_ir=query_ir,
                fact_set=fact_set,
                pins=pins,
                claims=claims,
                reason_code=LEGAL_RUNTIME_MISSING_FACTS,
                release_decision="REQUIRE_CLARIFICATION",
                disposition="MISSING_FACTS",
                missing_facts=missing,
            )

        applicable = [
            claim
            for claim in claims
            if _conditions_match(claim["candidate"].get("conditions", {}), fact_set.facts)
            and not _exception_matches(claim["candidate"].get("exceptions", []), fact_set.facts)
        ]
        if not applicable:
            return self._closed_result(
                query_ir=query_ir,
                fact_set=fact_set,
                pins=pins,
                claims=claims,
                reason_code=LEGAL_RUNTIME_NO_APPLICABLE_AUTHORITY,
                release_decision="ESCALATE",
                disposition="ESCALATE",
            )

        governing_claims = _governing_claims(applicable)
        dispositions = {_claim_disposition(claim) for claim in governing_claims}
        if len(dispositions) != 1:
            return self._closed_result(
                query_ir=query_ir,
                fact_set=fact_set,
                pins=pins,
                claims=governing_claims,
                reason_code=LEGAL_RUNTIME_CONFLICTING_AUTHORITY,
                release_decision="ESCALATE",
                disposition="UNRESOLVED",
            )

        disposition = dispositions.pop()
        issue_trace = _legal_issue_evaluation_trace(
            query_ir=query_ir,
            fact_set=fact_set,
            advice_claim_refs=[claim["candidate_id"] for claim in governing_claims],
            reason_code=None,
            disposition=disposition,
            issue_model=pins.get("issue_model"),
            missing_facts=[],
        )
        if issue_trace["blocked_definitive_release"]:
            return self._closed_result(
                query_ir=query_ir,
                fact_set=fact_set,
                pins=pins,
                claims=governing_claims,
                reason_code=LEGAL_RUNTIME_UNRESOLVED_MATERIAL_ISSUES,
                release_decision="ESCALATE",
                disposition="UNRESOLVED",
                issue_trace=issue_trace,
            )
        release_decision = (
            "ALLOW_CONDITIONAL" if disposition == "CONDITIONAL" else "ALLOW_DEFINITIVE"
        )
        advice = _advice_object(
            query_ir=query_ir,
            fact_set=fact_set,
            pins=pins,
            claims=governing_claims,
            disposition=disposition,
            reason_code=None,
            missing_facts=[],
        )
        release = _release_decision(
            release_decision=release_decision,
            reason_code=None,
            renderer_contract_version=renderer_contract_version,
            advice_hash=advice["advice_hash"],
        )
        return _result(
            query_ir=query_ir,
            fact_set=fact_set,
            advice=advice,
            release=release,
            pins=pins,
            issue_trace=issue_trace,
        )

    def replay(
        self,
        *,
        execution_manifest: dict[str, Any],
        facts: dict[str, Any],
        renderer_contract_version: str = "legal_renderer_contract.v1",
    ) -> dict[str, Any]:
        required = {
            "organization_id",
            "operation",
            "issue_model_id",
            "jurisdiction",
            "as_of_date",
            "authority_bundle_version_id",
            "authority_bundle_hash",
            "source_registry_snapshot_id",
            "source_registry_snapshot_hash",
            "knowledge_context_version_id",
            "knowledge_context_hash",
            "ontology_version_ref",
            "resolution_rules_version_ref",
            "issue_model_version_id",
            "issue_model_hash",
            "release_policy_version_ref",
            "release_policy_hash",
            "runtime_ref",
            "code_revision",
            "evidence_hashes",
            "query_hash",
            "advice_hash",
            "release_hash",
            "manifest_hash",
        }
        missing = sorted(required - set(execution_manifest))
        if missing:
            raise ValueError(f"{LEGAL_RUNTIME_REPLAY_PIN_MISSING}: {', '.join(missing)}")
        fact_set = normalize_fact_set(
            facts=facts,
            provenance={"organization_id": str(execution_manifest["organization_id"])},
        )
        query_ir = build_query_ir(
            operation=str(execution_manifest["operation"]),
            issue_model_id=str(execution_manifest["issue_model_id"]),
            jurisdiction=str(execution_manifest["jurisdiction"]),
            as_of_date=str(execution_manifest["as_of_date"]),
            requested_remedy=execution_manifest.get("requested_remedy"),
            fact_set=fact_set,
            governed_defaults=dict(execution_manifest.get("governed_defaults") or {}),
        )
        if renderer_contract_version not in SUPPORTED_RENDERER_CONTRACTS:
            raise ValueError(LEGAL_RUNTIME_RENDERER_CONTRACT_UNSUPPORTED)
        pins = self._manifest_pins(execution_manifest=execution_manifest)
        result = self._execute_with_pins(
            query_ir=query_ir,
            fact_set=fact_set,
            pins=pins,
            renderer_contract_version=renderer_contract_version,
        )
        manifest = result["legal_execution_manifest"]
        pinned_keys = required - {"manifest_hash"}
        if any(manifest[key] != execution_manifest[key] for key in pinned_keys):
            raise ValueError(LEGAL_RUNTIME_REPLAY_PIN_MISSING)
        if manifest["manifest_hash"] != execution_manifest["manifest_hash"]:
            raise ValueError(LEGAL_RUNTIME_REPLAY_PIN_MISSING)
        return result

    def _connect(self) -> sqlite3.Connection:
        if hosted.enabled():
            return get_conn(self._db_path)
        if self._db_path is None:
            return get_conn()
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _active_pins(self, *, organization_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                  a.*,
                  b.bundle_hash,
                  b.lifecycle_state AS bundle_state,
                  s.source_version_ids_json,
                  s.snapshot_hash,
                  k.claim_version_refs_json,
                  k.content_hash AS knowledge_context_hash,
                  k.ontology_version_ref,
                  b.validation_result_ref,
                  b.replay_dry_run_hash
                FROM legal_activation_events a
                JOIN legal_authority_bundle_versions b
                  ON b.authority_bundle_version_id = a.authority_bundle_version_id
                JOIN legal_source_registry_snapshots s
                  ON s.snapshot_id = a.source_registry_snapshot_id
                JOIN legal_knowledge_context_versions k
                  ON k.knowledge_context_version_id = a.knowledge_context_version_id
                WHERE a.activation_performed = 1
                  AND b.lifecycle_state = 'ACTIVE'
                ORDER BY a.created_at DESC
                LIMIT 25
                """
            ).fetchall()
            for candidate in row:
                source_version_ids = json.loads(candidate["source_version_ids_json"])
                owner = conn.execute(
                    """
                    SELECT COUNT(DISTINCT src.organization_id)
                    FROM legal_source_versions v
                    JOIN legal_sources src ON src.source_id = v.source_id
                    WHERE v.version_id IN ({})
                    """.format(",".join(hosted.sql_query("?", '%s') for _ in source_version_ids)),
                    tuple(source_version_ids),
                ).fetchone()[0]
                if owner and self._snapshot_belongs_to_org(conn, source_version_ids, organization_id):
                    runtime_pins = self._trusted_runtime_pins(conn=conn)
                    evidence_hashes = _activation_evidence_hashes(
                        activation=candidate,
                        snapshot={
                            "snapshot_hash": candidate["snapshot_hash"],
                        },
                        context={
                            "content_hash": candidate["knowledge_context_hash"],
                        },
                        bundle={
                            "bundle_hash": candidate["bundle_hash"],
                            "validation_result_ref": candidate["validation_result_ref"],
                            "replay_dry_run_hash": candidate["replay_dry_run_hash"],
                        },
                    )
                    return {
                        "activation_event_id": candidate["activation_event_id"],
                        "source_registry_snapshot_id": candidate["source_registry_snapshot_id"],
                        "source_registry_snapshot_hash": candidate["snapshot_hash"],
                        "knowledge_context_version_id": candidate["knowledge_context_version_id"],
                        "knowledge_context_hash": candidate["knowledge_context_hash"],
                        "ontology_version_ref": candidate["ontology_version_ref"],
                        "resolution_rules_version_ref": runtime_pins["resolution_rules_version_ref"],
                        "authority_bundle_version_id": candidate["authority_bundle_version_id"],
                        "authority_bundle_hash": candidate["bundle_hash"],
                        "release_policy_version_ref": runtime_pins["release_policy_version_ref"],
                        "release_policy_hash": runtime_pins["release_policy_hash"],
                        "runtime_ref": runtime_pins["runtime_ref"],
                        "code_revision": runtime_pins["code_revision"],
                        "evidence_hashes": {
                            **evidence_hashes,
                            **runtime_pins["runtime_artifact_hashes"],
                        },
                        "source_version_ids": source_version_ids,
                        "claim_refs": json.loads(candidate["claim_version_refs_json"] or "[]"),
                    }
        return None

    def _active_issue_model(
        self,
        *,
        organization_id: str,
        issue_model_id: str,
        as_of_date: str,
    ) -> dict[str, Any] | None:
        as_of = _parse_as_of(as_of_date)
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("""
                SELECT m.*
                FROM legal_issue_model_versions m
                JOIN legal_issue_model_events e
                  ON e.issue_model_version_id = m.issue_model_version_id
                 AND e.event_type = 'ACTIVATE'
                 AND e.to_state = 'ACTIVE'
                WHERE m.organization_id = ?
                  AND m.issue_model_id = ?
                  AND m.lifecycle_state = 'ACTIVE'
                ORDER BY e.created_at DESC, m.issue_model_version_id DESC
                """, """
                SELECT m.*
                FROM legal_issue_model_versions m
                JOIN legal_issue_model_events e
                  ON e.issue_model_version_id = m.issue_model_version_id
                 AND e.event_type = 'ACTIVATE'
                 AND e.to_state = 'ACTIVE'
                WHERE m.organization_id = %s
                  AND m.issue_model_id = %s
                  AND m.lifecycle_state = 'ACTIVE'
                ORDER BY e.created_at DESC, m.issue_model_version_id DESC
                """),
                (organization_id, issue_model_id),
            ).fetchall()
        for row in rows:
            effective_from = _parse_as_of(row["effective_from"])
            effective_to = _parse_as_of(row["effective_to"]) if row["effective_to"] else None
            if as_of < effective_from:
                continue
            if effective_to is not None and as_of >= effective_to:
                continue
            model = json.loads(row["model_json"])
            if content_hash(model) != row["model_hash"]:
                continue
            model = {
                **model,
                "issue_model_version_id": row["issue_model_version_id"],
                "model_hash": row["model_hash"],
            }
            return {
                "issue_model_version_id": row["issue_model_version_id"],
                "issue_model_hash": row["model_hash"],
                "issue_model": model,
                "ontology_version_ref": row["ontology_version_ref"],
                "resolution_rules_version_ref": row["resolution_rules_version_ref"],
            }
        return None

    def _manifest_pins(self, *, execution_manifest: dict[str, Any]) -> dict[str, Any]:
        with self._connect() as conn:
            stored_manifest = conn.execute(
                hosted.sql_query("""
                SELECT manifest_json
                FROM legal_execution_manifests
                WHERE manifest_hash = ?
                """, """
                SELECT manifest_json
                FROM legal_execution_manifests
                WHERE manifest_hash = %s
                """),
                (execution_manifest["manifest_hash"],),
            ).fetchone()
            if stored_manifest is None:
                raise ValueError(LEGAL_RUNTIME_REPLAY_PIN_MISSING)
            stored_payload = json.loads(stored_manifest["manifest_json"])
            comparable_stored = {
                key: value
                for key, value in stored_payload.items()
                if key != "executed_at"
            }
            comparable_submitted = {
                key: value
                for key, value in execution_manifest.items()
                if key != "executed_at"
            }
            if comparable_stored != comparable_submitted:
                raise ValueError(LEGAL_RUNTIME_REPLAY_PIN_MISSING)
            snapshot = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_source_registry_snapshots
                WHERE snapshot_id = ? AND snapshot_hash = ?
                """, """
                SELECT *
                FROM legal_source_registry_snapshots
                WHERE snapshot_id = %s AND snapshot_hash = %s
                """),
                (
                    execution_manifest["source_registry_snapshot_id"],
                    execution_manifest["source_registry_snapshot_hash"],
                ),
            ).fetchone()
            context = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_knowledge_context_versions
                WHERE knowledge_context_version_id = ? AND content_hash = ?
                """, """
                SELECT *
                FROM legal_knowledge_context_versions
                WHERE knowledge_context_version_id = %s AND content_hash = %s
                """),
                (
                    execution_manifest["knowledge_context_version_id"],
                    execution_manifest["knowledge_context_hash"],
                ),
            ).fetchone()
            bundle = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_authority_bundle_versions
                WHERE authority_bundle_version_id = ? AND bundle_hash = ?
                """, """
                SELECT *
                FROM legal_authority_bundle_versions
                WHERE authority_bundle_version_id = %s AND bundle_hash = %s
                """),
                (
                    execution_manifest["authority_bundle_version_id"],
                    execution_manifest["authority_bundle_hash"],
                ),
            ).fetchone()
            issue_model_row = conn.execute(
                hosted.sql_query("""
                SELECT m.*
                FROM legal_issue_model_versions m
                JOIN legal_issue_model_events e
                  ON e.issue_model_version_id = m.issue_model_version_id
                 AND e.event_type = 'ACTIVATE'
                 AND e.to_state = 'ACTIVE'
                WHERE m.issue_model_version_id = ?
                  AND m.model_hash = ?
                """, """
                SELECT m.*
                FROM legal_issue_model_versions m
                JOIN legal_issue_model_events e
                  ON e.issue_model_version_id = m.issue_model_version_id
                 AND e.event_type = 'ACTIVATE'
                 AND e.to_state = 'ACTIVE'
                WHERE m.issue_model_version_id = %s
                  AND m.model_hash = %s
                """),
                (
                    execution_manifest["issue_model_version_id"],
                    execution_manifest["issue_model_hash"],
                ),
            ).fetchone()
            activation = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_activation_events
                WHERE source_registry_snapshot_id = ?
                  AND knowledge_context_version_id = ?
                  AND authority_bundle_version_id = ?
                  AND activation_performed = 1
                ORDER BY created_at DESC
                LIMIT 1
                """, """
                SELECT *
                FROM legal_activation_events
                WHERE source_registry_snapshot_id = %s
                  AND knowledge_context_version_id = %s
                  AND authority_bundle_version_id = %s
                  AND activation_performed = 1
                ORDER BY created_at DESC
                LIMIT 1
                """),
                (
                    execution_manifest["source_registry_snapshot_id"],
                    execution_manifest["knowledge_context_version_id"],
                    execution_manifest["authority_bundle_version_id"],
                ),
            ).fetchone()
            resolution_rules = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_runtime_pin_artifacts
                WHERE artifact_ref = ?
                  AND artifact_type = 'resolution_rules'
                  AND lifecycle_state = 'ACTIVE'
                """, """
                SELECT *
                FROM legal_runtime_pin_artifacts
                WHERE artifact_ref = %s
                  AND artifact_type = 'resolution_rules'
                  AND lifecycle_state = 'ACTIVE'
                """),
                (execution_manifest["resolution_rules_version_ref"],),
            ).fetchone()
            release_policy = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_runtime_pin_artifacts
                WHERE artifact_ref = ?
                  AND artifact_type = 'release_policy'
                  AND artifact_hash = ?
                  AND lifecycle_state = 'ACTIVE'
                """, """
                SELECT *
                FROM legal_runtime_pin_artifacts
                WHERE artifact_ref = %s
                  AND artifact_type = 'release_policy'
                  AND artifact_hash = %s
                  AND lifecycle_state = 'ACTIVE'
                """),
                (
                    execution_manifest["release_policy_version_ref"],
                    execution_manifest["release_policy_hash"],
                ),
            ).fetchone()
            runtime_artifact = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_runtime_pin_artifacts
                WHERE artifact_ref = ?
                  AND artifact_type = 'runtime'
                  AND lifecycle_state = 'ACTIVE'
                """, """
                SELECT *
                FROM legal_runtime_pin_artifacts
                WHERE artifact_ref = %s
                  AND artifact_type = 'runtime'
                  AND lifecycle_state = 'ACTIVE'
                """),
                (execution_manifest["runtime_ref"],),
            ).fetchone()
            code_revision = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_runtime_pin_artifacts
                WHERE artifact_ref = ?
                  AND artifact_type = 'code_revision'
                  AND lifecycle_state = 'ACTIVE'
                """, """
                SELECT *
                FROM legal_runtime_pin_artifacts
                WHERE artifact_ref = %s
                  AND artifact_type = 'code_revision'
                  AND lifecycle_state = 'ACTIVE'
                """),
                (f"code_revision:{execution_manifest['code_revision']}",),
            ).fetchone()
        if (
            snapshot is None
            or context is None
            or bundle is None
            or issue_model_row is None
            or activation is None
            or resolution_rules is None
            or release_policy is None
            or runtime_artifact is None
            or code_revision is None
        ):
            raise ValueError(LEGAL_RUNTIME_REPLAY_PIN_MISSING)
        source_version_ids = json.loads(snapshot["source_version_ids_json"])
        issue_model = json.loads(issue_model_row["model_json"])
        if content_hash(issue_model) != execution_manifest["issue_model_hash"]:
            raise ValueError(LEGAL_RUNTIME_REPLAY_PIN_MISSING)
        issue_model = {
            **issue_model,
            "issue_model_version_id": issue_model_row["issue_model_version_id"],
            "model_hash": issue_model_row["model_hash"],
        }
        evidence_hashes = _activation_evidence_hashes(
            activation=activation,
            snapshot=snapshot,
            context=context,
            bundle=bundle,
        )
        release_policy_payload = json.loads(release_policy["payload_json"])
        runtime_payload = json.loads(runtime_artifact["payload_json"])
        code_revision_payload = json.loads(code_revision["payload_json"])
        runtime_evidence_hashes = {
            "resolution_rules": resolution_rules["artifact_hash"],
            "release_policy": release_policy["artifact_hash"],
            "runtime": runtime_artifact["artifact_hash"],
            "code_revision": code_revision["artifact_hash"],
        }
        expected_evidence_hashes = {**evidence_hashes, **runtime_evidence_hashes}
        if expected_evidence_hashes != execution_manifest["evidence_hashes"]:
            raise ValueError(LEGAL_RUNTIME_REPLAY_PIN_MISSING)
        return {
            "activation_event_id": activation["activation_event_id"],
            "source_registry_snapshot_id": snapshot["snapshot_id"],
            "source_registry_snapshot_hash": snapshot["snapshot_hash"],
            "knowledge_context_version_id": context["knowledge_context_version_id"],
            "knowledge_context_hash": context["content_hash"],
            "ontology_version_ref": context["ontology_version_ref"],
            "resolution_rules_version_ref": resolution_rules["artifact_ref"],
            "authority_bundle_version_id": bundle["authority_bundle_version_id"],
            "authority_bundle_hash": bundle["bundle_hash"],
            "issue_model_version_id": issue_model_row["issue_model_version_id"],
            "issue_model_hash": issue_model_row["model_hash"],
            "issue_model": issue_model,
            "release_policy_version_ref": release_policy_payload["release_policy_version_ref"],
            "release_policy_hash": release_policy["artifact_hash"],
            "runtime_ref": runtime_payload["runtime_ref"],
            "code_revision": code_revision_payload["code_revision"],
            "evidence_hashes": expected_evidence_hashes,
            "source_version_ids": source_version_ids,
            "claim_refs": json.loads(context["claim_version_refs_json"] or "[]"),
        }

    def _trusted_runtime_pins(self, *, conn: sqlite3.Connection) -> dict[str, Any]:
        store = LegalCorpusStore(self._db_path)
        release_policy_payload = {"release_policy_version_ref": LEGAL_RELEASE_POLICY_VERSION}
        runtime_payload = {"runtime_ref": LEGAL_RUNTIME_REF}
        code_revision_payload = {"code_revision": _code_revision()}
        resolution_rules = store.ensure_runtime_pin_artifact(
            artifact_ref=LEGAL_RESOLUTION_RULES_VERSION,
            artifact_type="resolution_rules",
            payload={"resolution_rules_version_ref": LEGAL_RESOLUTION_RULES_VERSION},
        )
        release_policy = store.ensure_runtime_pin_artifact(
            artifact_ref=LEGAL_RELEASE_POLICY_VERSION,
            artifact_type="release_policy",
            payload=release_policy_payload,
        )
        runtime = store.ensure_runtime_pin_artifact(
            artifact_ref=LEGAL_RUNTIME_REF,
            artifact_type="runtime",
            payload=runtime_payload,
        )
        code_revision = store.ensure_runtime_pin_artifact(
            artifact_ref=f"code_revision:{code_revision_payload['code_revision']}",
            artifact_type="code_revision",
            payload=code_revision_payload,
        )
        return {
            "resolution_rules_version_ref": resolution_rules["artifact_ref"],
            "release_policy_version_ref": release_policy_payload["release_policy_version_ref"],
            "release_policy_hash": release_policy["artifact_hash"],
            "runtime_ref": runtime_payload["runtime_ref"],
            "code_revision": code_revision_payload["code_revision"],
            "runtime_artifact_hashes": {
                "resolution_rules": resolution_rules["artifact_hash"],
                "release_policy": release_policy["artifact_hash"],
                "runtime": runtime["artifact_hash"],
                "code_revision": code_revision["artifact_hash"],
            },
        }

    def _record_execution_manifest(self, manifest: dict[str, Any]) -> None:
        with self._connect() as conn:
            existing = conn.execute(
                hosted.sql_query("""
                SELECT manifest_json
                FROM legal_execution_manifests
                WHERE manifest_hash = ?
                """, """
                SELECT manifest_json
                FROM legal_execution_manifests
                WHERE manifest_hash = %s
                """),
                (manifest["manifest_hash"],),
            ).fetchone()
            if existing is not None:
                return
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_execution_manifests (
                  manifest_hash, manifest_json, created_at
                ) VALUES (?, ?, ?)
                """, """
                INSERT INTO legal_execution_manifests (
                  manifest_hash, manifest_json, created_at
                ) VALUES (%s, %s, %s)
                """),
                (
                    manifest["manifest_hash"],
                    json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                    now_iso(),
                ),
            )

    def _snapshot_belongs_to_org(
        self,
        conn: sqlite3.Connection,
        source_version_ids: list[str],
        organization_id: str,
    ) -> bool:
        if not source_version_ids:
            return False
        rows = conn.execute(
            """
            SELECT DISTINCT src.organization_id
            FROM legal_source_versions v
            JOIN legal_sources src ON src.source_id = v.source_id
            WHERE v.version_id IN ({})
            """.format(",".join(hosted.sql_query("?", '%s') for _ in source_version_ids)),
            tuple(source_version_ids),
        ).fetchall()
        return {row[0] for row in rows} == {organization_id}

    def _pinned_claims(self, *, pins: dict[str, Any], query_ir: LegalQueryIR) -> list[dict[str, Any]]:
        claim_refs = pins["claim_refs"]
        if not claim_refs:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.*
                FROM legal_semantic_candidates c
                JOIN legal_certification_events e
                  ON e.object_ref = c.candidate_id
                WHERE c.candidate_id IN ({})
                  AND c.version_id IN ({})
                  AND c.candidate_type = 'claim'
                  AND c.review_state = 'APPROVED'
                  AND e.object_type = 'LegalClaimCandidate'
                  AND e.decision = 'CERTIFY'
                """.format(
                    ",".join(hosted.sql_query("?", '%s') for _ in claim_refs),
                    ",".join(hosted.sql_query("?", '%s') for _ in pins["source_version_ids"]),
                ),
                tuple(claim_refs + pins["source_version_ids"]),
            ).fetchall()
        claims = []
        for row in rows:
            candidate = json.loads(row["candidate_json"])
            if not _claim_matches_query(candidate, query_ir):
                continue
            if not _effective_on(candidate, query_ir.as_of_date):
                continue
            claims.append(
                {
                    "candidate_id": row["candidate_id"],
                    "candidate_hash": row["candidate_hash"],
                    "candidate": candidate,
                    "source_span": json.loads(row["source_span_json"]),
                }
            )
        return sorted(claims, key=lambda claim: claim["candidate_id"])

    def _closed_result(
        self,
        *,
        query_ir: LegalQueryIR,
        fact_set: NormalizedFactSet,
        reason_code: str,
        release_decision: str,
        disposition: str,
        pins: dict[str, Any] | None = None,
        claims: list[dict[str, Any]] | None = None,
        missing_facts: list[str] | None = None,
        renderer_contract_version: str = "legal_renderer_contract.v1",
        issue_trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        advice = _advice_object(
            query_ir=query_ir,
            fact_set=fact_set,
            pins=pins,
            claims=claims or [],
            disposition=disposition,
            reason_code=reason_code,
            missing_facts=missing_facts or [],
        )
        release = _release_decision(
            release_decision=release_decision,
            reason_code=reason_code,
            renderer_contract_version=renderer_contract_version,
            advice_hash=advice["advice_hash"],
        )
        return _result(
            query_ir=query_ir,
            fact_set=fact_set,
            advice=advice,
            release=release,
            pins=pins or {},
            issue_trace=issue_trace,
        )


def normalize_fact_set(*, facts: dict[str, Any], provenance: dict[str, Any]) -> NormalizedFactSet:
    normalized = {str(key): value for key, value in sorted((facts or {}).items())}
    fact_states = {
        key: _fact_state(value)
        for key, value in normalized.items()
    }
    payload = {
        "facts": normalized,
        "fact_states": fact_states,
        "provenance": provenance,
    }
    return NormalizedFactSet(
        facts=normalized,
        fact_states=fact_states,
        provenance=provenance,
        fact_set_hash=content_hash(payload),
    )


def _fact_state(value: Any) -> str:
    if value in (None, ""):
        return "MISSING"
    if isinstance(value, (list, tuple, set)) and len(value) != 1:
        return "CONFLICTING"
    if isinstance(value, dict) and value.get("ambiguous") is True:
        return "AMBIGUOUS"
    return "ASSERTED"


def build_query_ir(
    *,
    operation: str,
    issue_model_id: str,
    jurisdiction: str,
    as_of_date: str,
    requested_remedy: str | None,
    fact_set: NormalizedFactSet,
    governed_defaults: dict[str, Any],
) -> LegalQueryIR:
    _parse_as_of(as_of_date)
    payload = {
        "operation": operation,
        "issue_model_id": issue_model_id,
        "jurisdiction": jurisdiction,
        "as_of_date": as_of_date,
        "requested_remedy": requested_remedy,
        "fact_set_hash": fact_set.fact_set_hash,
        "governed_defaults": governed_defaults,
    }
    return LegalQueryIR(
        operation=operation,
        issue_model_id=issue_model_id,
        jurisdiction=jurisdiction,
        as_of_date=as_of_date,
        requested_remedy=requested_remedy,
        fact_set_hash=fact_set.fact_set_hash,
        governed_defaults=governed_defaults,
        query_hash=content_hash(payload),
    )


def _parse_as_of(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("as_of_date must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _claim_matches_query(candidate: dict[str, Any], query_ir: LegalQueryIR) -> bool:
    return (
        str(candidate.get("operation") or "") == query_ir.operation
        and str(candidate.get("issue_model_id") or "") == query_ir.issue_model_id
        and str(candidate.get("jurisdiction") or "") == query_ir.jurisdiction
    )


def _effective_on(candidate: dict[str, Any], as_of_date: str) -> bool:
    as_of = _parse_as_of(as_of_date)
    effective_from = candidate.get("effective_from")
    effective_to = candidate.get("effective_to")
    if effective_from and as_of < _parse_as_of(str(effective_from)):
        return False
    if effective_to and as_of >= _parse_as_of(str(effective_to)):
        return False
    return True


def _missing_material_facts(claims: list[dict[str, Any]], facts: dict[str, Any]) -> list[str]:
    required: set[str] = set()
    for claim in claims:
        required.update(str(key) for key in claim["candidate"].get("conditions", {}))
    return sorted(key for key in required if facts.get(key) in (None, ""))


def _conditions_match(conditions: dict[str, Any], facts: dict[str, Any]) -> bool:
    for key, expected in conditions.items():
        observed = facts.get(key)
        if isinstance(observed, (list, tuple, set)):
            if len(observed) != 1:
                return False
            observed = next(iter(observed))
        if isinstance(observed, dict) and observed.get("ambiguous") is True:
            return False
        if isinstance(expected, list):
            if observed not in expected:
                return False
        elif observed != expected:
            return False
    return True


def _exception_matches(exceptions: list[dict[str, Any]], facts: dict[str, Any]) -> bool:
    for exception in exceptions:
        conditions = exception.get("conditions")
        if isinstance(conditions, dict) and _conditions_match(conditions, facts):
            return True
    return False


def _claim_disposition(claim: dict[str, Any]) -> str:
    disposition = str(claim["candidate"].get("disposition") or "UNRESOLVED")
    if disposition not in LEGAL_DISPOSITIONS:
        return "UNRESOLVED"
    return disposition


def _governing_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(claims, key=lambda claim: (_authority_rank(claim), -_specificity(claim)))
    if not ranked:
        return []
    best_rank = _authority_rank(ranked[0])
    same_rank = [claim for claim in ranked if _authority_rank(claim) == best_rank]
    best_specificity = max(_specificity(claim) for claim in same_rank)
    return [claim for claim in same_rank if _specificity(claim) == best_specificity]


def _authority_rank(claim: dict[str, Any]) -> int:
    try:
        return int(claim["candidate"].get("authority_rank", 100))
    except (TypeError, ValueError):
        return 100


def _specificity(claim: dict[str, Any]) -> int:
    conditions = claim["candidate"].get("conditions", {})
    exceptions = claim["candidate"].get("exceptions", [])
    return len(conditions if isinstance(conditions, dict) else {}) + len(
        exceptions if isinstance(exceptions, list) else []
    )


def _advice_object(
    *,
    query_ir: LegalQueryIR,
    fact_set: NormalizedFactSet,
    pins: dict[str, Any] | None,
    claims: list[dict[str, Any]],
    disposition: str,
    reason_code: str | None,
    missing_facts: list[str],
) -> dict[str, Any]:
    citations = [
        {
            "candidate_id": claim["candidate_id"],
            "candidate_hash": claim["candidate_hash"],
            "source_span": claim["source_span"],
        }
        for claim in claims
    ]
    clarification_objects = _clarification_objects(
        missing_facts=missing_facts,
        fact_states=fact_set.fact_states,
    )
    knowledge_gap_candidates = _knowledge_gap_candidates(
        reason_code=reason_code,
        query_ir=query_ir,
        pins=pins,
    )
    payload = {
        "schema_version": "legal_advice_object.v1",
        "disposition": disposition,
        "reason_code": reason_code,
        "query_hash": query_ir.query_hash,
        "fact_set_hash": fact_set.fact_set_hash,
        "authority_bundle_version_id": (pins or {}).get("authority_bundle_version_id"),
        "authority_bundle_hash": (pins or {}).get("authority_bundle_hash"),
        "claim_refs": [claim["candidate_id"] for claim in claims],
        "missing_facts": missing_facts,
        "clarification_objects": clarification_objects,
        "knowledge_gap_candidates": knowledge_gap_candidates,
        "citations": citations,
    }
    return {**payload, "advice_hash": content_hash(payload)}


def _clarification_objects(
    *,
    missing_facts: list[str],
    fact_states: dict[str, str],
) -> list[dict[str, Any]]:
    objects = [
        {
            "schema_version": "legal_clarification_object.v1",
            "fact_key": fact_key,
            "reason": "MISSING_MATERIAL_FACT",
        }
        for fact_key in missing_facts
    ]
    for fact_key, state in sorted(fact_states.items()):
        if state not in {"CONFLICTING", "AMBIGUOUS"}:
            continue
        objects.append(
            {
                "schema_version": "legal_clarification_object.v1",
                "fact_key": fact_key,
                "reason": f"{state}_FACT",
            }
        )
    return objects


def _knowledge_gap_candidates(
    *,
    reason_code: str | None,
    query_ir: LegalQueryIR,
    pins: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if reason_code not in {
        LEGAL_RUNTIME_NO_ACTIVE_BUNDLE,
        LEGAL_RUNTIME_NO_APPLICABLE_AUTHORITY,
    }:
        return []
    payload = {
        "schema_version": "legal_knowledge_gap_candidate.v1",
        "reason_code": reason_code,
        "operation": query_ir.operation,
        "issue_model_id": query_ir.issue_model_id,
        "jurisdiction": query_ir.jurisdiction,
        "as_of_date": query_ir.as_of_date,
        "authority_bundle_version_id": (pins or {}).get("authority_bundle_version_id"),
    }
    return [{**payload, "candidate_hash": content_hash(payload)}]


def _activation_evidence_hashes(
    *,
    activation: sqlite3.Row | dict[str, Any],
    snapshot: sqlite3.Row | dict[str, Any],
    context: sqlite3.Row | dict[str, Any],
    bundle: sqlite3.Row | dict[str, Any],
) -> dict[str, str]:
    return {
        **json.loads(activation["content_hashes_json"] or "{}"),
        "source_registry_snapshot": snapshot["snapshot_hash"],
        "knowledge_context": context["content_hash"],
        "authority_bundle": bundle["bundle_hash"],
        "validation_result": bundle["validation_result_ref"],
        "replay_dry_run": bundle["replay_dry_run_hash"],
    }


def _release_decision(
    *,
    release_decision: str,
    reason_code: str | None,
    renderer_contract_version: str,
    advice_hash: str,
) -> dict[str, Any]:
    if release_decision not in LEGAL_RELEASE_DECISIONS:
        raise ValueError("invalid legal release decision")
    payload = {
        "schema_version": "legal_release_decision.v1",
        "decision": release_decision,
        "reason_code": reason_code,
        "renderer_contract_version": renderer_contract_version,
        "advice_hash": advice_hash,
    }
    return {**payload, "release_hash": content_hash(payload)}


def _result(
    *,
    query_ir: LegalQueryIR,
    fact_set: NormalizedFactSet,
    advice: dict[str, Any],
    release: dict[str, Any],
    pins: dict[str, Any],
    issue_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    issue_trace = issue_trace or _legal_issue_evaluation_trace(
        query_ir=query_ir,
        fact_set=fact_set,
        advice_claim_refs=advice["claim_refs"],
        reason_code=advice["reason_code"],
        disposition=advice["disposition"],
        issue_model=pins.get("issue_model"),
        missing_facts=advice["missing_facts"],
    )
    manifest = {
        "schema_version": "legal_execution_manifest.v1",
        "organization_id": fact_set.provenance["organization_id"],
        "operation": query_ir.operation,
        "issue_model_id": query_ir.issue_model_id,
        "issue_model_version_id": pins.get("issue_model_version_id"),
        "issue_model_hash": pins.get("issue_model_hash"),
        "jurisdiction": query_ir.jurisdiction,
        "as_of_date": query_ir.as_of_date,
        "requested_remedy": query_ir.requested_remedy,
        "governed_defaults": query_ir.governed_defaults,
        "query_hash": query_ir.query_hash,
        "fact_set_hash": fact_set.fact_set_hash,
        "source_registry_snapshot_id": pins.get("source_registry_snapshot_id"),
        "source_registry_snapshot_hash": pins.get("source_registry_snapshot_hash"),
        "knowledge_context_version_id": pins.get("knowledge_context_version_id"),
        "knowledge_context_hash": pins.get("knowledge_context_hash"),
        "ontology_version_ref": pins.get("ontology_version_ref"),
        "resolution_rules_version_ref": pins.get("resolution_rules_version_ref"),
        "authority_bundle_version_id": advice["authority_bundle_version_id"],
        "authority_bundle_hash": advice["authority_bundle_hash"],
        "release_policy_version_ref": pins.get("release_policy_version_ref"),
        "release_policy_hash": pins.get("release_policy_hash"),
        "runtime_ref": pins.get("runtime_ref") or LEGAL_RUNTIME_REF,
        "code_revision": pins.get("code_revision") or _code_revision(),
        "evidence_hashes": pins.get("evidence_hashes") or {},
        "advice_hash": advice["advice_hash"],
        "release_hash": release["release_hash"],
        "executed_at": now_iso(),
    }
    return {
        "schema_version": "legal_runtime_result.v1",
        "normalized_fact_set": {
            "facts": fact_set.facts,
            "fact_states": fact_set.fact_states,
            "fact_set_hash": fact_set.fact_set_hash,
        },
        "legal_query_ir": {
            "operation": query_ir.operation,
            "issue_model_id": query_ir.issue_model_id,
            "jurisdiction": query_ir.jurisdiction,
            "as_of_date": query_ir.as_of_date,
            "requested_remedy": query_ir.requested_remedy,
            "governed_defaults": query_ir.governed_defaults,
            "query_hash": query_ir.query_hash,
        },
        "legal_issue_evaluation_trace": issue_trace,
        "authority_resolution_trace": {
            "schema_version": "authority_resolution_trace.v1",
            "claim_refs": advice["claim_refs"],
            "reason_code": advice["reason_code"],
            "trace_hash": content_hash(
                {
                    "claim_refs": advice["claim_refs"],
                    "reason_code": advice["reason_code"],
                    "query_hash": query_ir.query_hash,
                }
            ),
        },
        "legal_advice_object": advice,
        "release_decision": release,
        "rendered_advice": {
            "schema_version": "rendered_advice.v1",
            "renderer_contract_version": release["renderer_contract_version"],
            "source": "legal_advice_object",
            "render_payload_hash": content_hash(
                {
                    "advice_hash": advice["advice_hash"],
                    "release_hash": release["release_hash"],
                }
            ),
        },
        "legal_execution_manifest": {
            **manifest,
            "manifest_hash": content_hash(
                {key: value for key, value in manifest.items() if key != "executed_at"}
            ),
        },
        "canonical_result_hash": content_hash(
            {
                "query_ir": query_ir.__dict__,
                "issue_trace_hash": issue_trace["trace_hash"],
                "advice_hash": advice["advice_hash"],
                "release_hash": release["release_hash"],
            }
        ),
    }


def _legal_issue_evaluation_trace(
    *,
    query_ir: LegalQueryIR,
    fact_set: NormalizedFactSet,
    advice_claim_refs: list[str],
    reason_code: str | None,
    disposition: str,
    issue_model: dict[str, Any] | None,
    missing_facts: list[str],
) -> dict[str, Any]:
    material_facts = sorted(
        {
            *missing_facts,
            *[
                key
                for key, state in fact_set.fact_states.items()
                if state in {"CONFLICTING", "AMBIGUOUS"}
            ],
        }
    )
    if issue_model and issue_model.get("required_issues"):
        issues = _evaluate_governed_issues(
            issue_model=issue_model,
            advice_claim_refs=advice_claim_refs,
            material_facts=material_facts,
            reason_code=reason_code,
        )
    else:
        issues = [
            {
                "issue_id": "ACTIVE_AUTHORITY_LOOKUP",
                "status": "UNRESOLVED" if reason_code else "RESOLVED",
                "claim_refs": advice_claim_refs,
                "material": True,
            },
            {
                "issue_id": "MATERIAL_FACTS",
                "status": "UNRESOLVED" if material_facts else "RESOLVED",
                "fact_refs": material_facts,
                "material": True,
            },
            {
                "issue_id": "RELEASE_ELIGIBILITY",
                "status": (
                    "RESOLVED"
                    if disposition in {"REQUIRED", "NOT_REQUIRED", "CONDITIONAL"}
                    else "UNRESOLVED"
                ),
                "disposition": disposition,
                "material": True,
            },
        ]
    payload = {
        "schema_version": "legal_issue_evaluation_trace.v1",
        "issue_model_id": query_ir.issue_model_id,
        "issue_model_version_id": (issue_model or {}).get("issue_model_version_id"),
        "issue_model_hash": (issue_model or {}).get("model_hash"),
        "query_hash": query_ir.query_hash,
        "issues": issues,
        "blocked_definitive_release": any(
            issue["status"] == "UNRESOLVED" and issue.get("material", True)
            for issue in issues
        ),
    }
    return {**payload, "trace_hash": content_hash(payload)}


def _evaluate_governed_issues(
    *,
    issue_model: dict[str, Any],
    advice_claim_refs: list[str],
    material_facts: list[str],
    reason_code: str | None,
) -> list[dict[str, Any]]:
    resolved: dict[str, str] = {}
    claim_refs = set(advice_claim_refs)
    issues: list[dict[str, Any]] = []
    for issue in issue_model.get("required_issues", []):
        issue_id = str(issue.get("issue_id") or "")
        dependencies = [str(dep) for dep in issue.get("dependencies", [])]
        dependency_blocked = any(resolved.get(dep) != "RESOLVED" for dep in dependencies)
        required_facts = [str(fact) for fact in issue.get("required_facts", [])]
        unresolved_facts = sorted(set(required_facts) & set(material_facts))
        required_claim_sets = issue.get("required_claim_sets") or []
        claim_set_satisfied = True
        if required_claim_sets:
            claim_set_satisfied = any(
                set(str(ref) for ref in claim_set).issubset(claim_refs)
                for claim_set in required_claim_sets
                if isinstance(claim_set, list)
            )
        status = "RESOLVED"
        unresolved_reason = None
        if reason_code:
            status = "UNRESOLVED"
            unresolved_reason = reason_code
        elif dependency_blocked:
            status = "UNRESOLVED"
            unresolved_reason = "DEPENDENCY_UNRESOLVED"
        elif unresolved_facts:
            status = "UNRESOLVED"
            unresolved_reason = "MATERIAL_FACT_UNRESOLVED"
        elif not claim_set_satisfied:
            status = "UNRESOLVED"
            unresolved_reason = "REQUIRED_CLAIM_SET_UNRESOLVED"
        resolved[issue_id] = status
        issues.append(
            {
                "issue_id": issue_id,
                "status": status,
                "dependencies": dependencies,
                "required_claim_sets": required_claim_sets,
                "required_facts": required_facts,
                "unresolved_facts": unresolved_facts,
                "material": bool(issue.get("material", True)),
                "unresolved_reason": unresolved_reason,
            }
        )
    return issues


def _code_revision() -> str:
    return os.environ.get("SEMANTIER_RUNTIME_CODE_REVISION", "unknown")
