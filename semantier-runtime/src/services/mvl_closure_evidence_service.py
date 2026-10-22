"""W9 MVL closure evidence manifest generation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from eos.artifact_hashing import canonical_json, content_hash
from semantier.knowledge_feature_packs.semantica_adapter import SEMANTICA_COMMIT
from storage.graph_schema_discovery_store import GraphSchemaDiscoveryStore

from .graph_schema_llm_evidence_runner import (
    PROVIDER_CONTRACT_VERSION,
    SKILL_MAJOR_VERSION,
)


MVL_CLOSURE_EVIDENCE_SCHEMA_VERSION = "mvl_closure_evidence_manifest.v1"
PLAN_REF = "docs/plans/2026-08-19-mvl-implementation-sequence-and-contracts-v1.md"
MVL_E2E_FLAG_CRITERIA_REPORT_SCHEMA_VERSION = "mvl_e2e_flag_criteria_report.v1"
MVL_E2E_FLAG_POLICY_REPORT_SCHEMA_VERSION = "mvl_e2e_flag_policy_report.v1"
DOCUMENT_INFERENCE_FLAG = "DOCUMENT_INFERENCE_REVIEW_E2E"
GRAPH_SCHEMA_FLAG = "GRAPH_SCHEMA_LEARNING_LOOP_E2E"
E2E_POLICY_FLAGS = frozenset({DOCUMENT_INFERENCE_FLAG, GRAPH_SCHEMA_FLAG})
E2E_POLICY_ACTION_STATUSES = frozenset({"pass", "incomplete", "blocked", "fail"})
WORKSPACE_CLOSURE_MANIFEST_PREFIX = "artifacts/mvl-closure/evidence-manifests/"
PENDING_RELEASE_GATE_PREFIX = "pending:mvl-release-gate/"
LLM_PROPOSAL_EVIDENCE_PREFIX = "llm_proposal_evidence:"
LLM_PROPOSAL_ARTIFACT_PREFIX = "artifacts/knowledge-graph-building/llm-proposals/"
HISTORICAL_LLM_PROPOSAL_ARTIFACT_PREFIX = (
    "artifacts/graph-schema-discovery/llm-proposals/"
)
HISTORICAL_LLM_PROPOSAL_SKILL_MAJOR_VERSION = "graph-schema-discovery@1"
LLM_PROPOSAL_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "provider",
        "model",
        "semantica_commit",
        "skill_major_version",
        "provider_contract_version",
        "ontology",
    }
)
RELEASE_COMMAND_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "workspace_id",
        "command_key",
        "command",
        "cwd",
        "returncode",
        "status",
        "stdout",
        "stderr",
        "recorded_at",
        "content_hash",
    }
)

GateStatus = Literal["pass", "fail", "blocked", "incomplete"]
COMMAND_RESULT_STATUSES = frozenset({"passed", "failed", "blocked", "skipped"})
RELEASE_COMMAND_EVIDENCE_STATUSES = frozenset({"passed", "failed", "skipped"})
COMMAND_KEYS = frozenset(
    {
        "backend_focused_regression",
        "frontend_focused_regression",
        "frontend_typecheck",
        "plan_spec_sync",
        "doc_semantic_alignment",
        "graph_schema_learning_loop_e2e",
        "docx_pdf_viewer_e2e",
    }
)
REQUIRED_GATE_IDS = (
    "backend_focused_regression",
    "frontend_focused_regression",
    "frontend_typecheck",
    "plan_spec_sync",
    "doc_semantic_alignment",
    "graph_schema_learning_loop_e2e",
    "open_source_docx_pdf_viewer_e2e",
    "document_inference_e2e_flag_criteria",
    "e2e_release_flag_policy",
    "llm_proposal_evidence",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _command_status(value: str | None) -> GateStatus:
    if value == "passed":
        return "pass"
    if value == "failed":
        return "fail"
    if value == "blocked":
        return "blocked"
    return "incomplete"


@dataclass(frozen=True)
class MvlClosureEvidenceWrite:
    manifest: dict[str, Any]
    path: Path


class MvlClosureEvidenceService:
    """Build and write neutral closure manifests for W9.

    The manifest is evidence, not certification. It records which gates have
    proof and which remain blocked or incomplete so release runs can close W9
    without relying on unstructured terminal logs.
    """

    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        if not workspace_id:
            raise ValueError("workspace_id is required")
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_id = workspace_id
        self.artifact_root = (
            self.workspace_root
            / "workspaces"
            / workspace_id
            / "artifacts"
            / "mvl-closure"
            / "evidence-manifests"
        )

    def build_manifest(
        self,
        *,
        command_results: dict[str, str],
        env: dict[str, str | None],
        llm_evidence_ref: str | None = None,
        e2e_flag_criteria_report: dict[str, Any] | None = None,
        e2e_flag_policy_report: dict[str, Any] | None = None,
        command_evidence_refs: dict[str, str] | None = None,
        generated_at: str | None = None,
        manifest_id: str = "mvl-closure-current",
    ) -> dict[str, Any]:
        self._require_safe_artifact_id(manifest_id, label="manifest_id")
        self._require_command_result_statuses(command_results)
        generated_at = generated_at or _utc_now()
        self._require_aware_timestamp(generated_at, label="generated_at")
        command_evidence_refs = self._verified_command_evidence_refs(
            manifest_id=manifest_id,
            command_results=command_results,
            command_evidence_refs=command_evidence_refs or {},
        )
        e2e_flag_criteria_report = self._verified_e2e_flag_criteria_report(
            e2e_flag_criteria_report
        )
        e2e_flag_policy_report = self._verified_e2e_flag_policy_report(
            e2e_flag_policy_report,
            criteria_report=e2e_flag_criteria_report,
        )
        llm_evidence_ref = self._verified_llm_evidence_ref(llm_evidence_ref)
        gates = [
            self._command_gate(
                "backend_focused_regression",
                command_results,
                "backend_focused_regression",
                "Focused backend regression bundle.",
                command_evidence_refs,
            ),
            self._command_gate(
                "frontend_focused_regression",
                command_results,
                "frontend_focused_regression",
                "Focused frontend Vitest suite.",
                command_evidence_refs,
            ),
            self._command_gate(
                "frontend_typecheck",
                command_results,
                "frontend_typecheck",
                "Hermes Workspace TypeScript typecheck.",
                command_evidence_refs,
            ),
            self._command_gate(
                "plan_spec_sync",
                command_results,
                "plan_spec_sync",
                "Governed plan/spec synchronization checker.",
                command_evidence_refs,
            ),
            self._command_gate(
                "doc_semantic_alignment",
                command_results,
                "doc_semantic_alignment",
                "Governed document semantic-alignment checker.",
                command_evidence_refs,
            ),
            self._playwright_gate(
                command_results.get("graph_schema_learning_loop_e2e"),
                evidence_ref=command_evidence_refs.get("graph_schema_learning_loop_e2e"),
            ),
            self._open_source_document_viewer_gate(
                env=env,
                doc_e2e_status=command_results.get("docx_pdf_viewer_e2e"),
                evidence_ref=command_evidence_refs.get("docx_pdf_viewer_e2e"),
            ),
            self._e2e_flag_criteria_gate(e2e_flag_criteria_report),
            self._e2e_flag_policy_gate(e2e_flag_policy_report),
            self._llm_gate(llm_evidence_ref),
        ]
        blockers = [
            gate["diagnostic"]
            for gate in gates
            if gate["status"] in {"blocked", "incomplete"} and gate.get("diagnostic")
        ]
        body = {
            "schema_version": MVL_CLOSURE_EVIDENCE_SCHEMA_VERSION,
            "manifest_id": manifest_id,
            "plan_ref": PLAN_REF,
            "workspace_id": self.workspace_id,
            "generated_at": generated_at,
            "overall_status": self._overall_status(gates),
            "gates": gates,
            "acceptance_gates": self._acceptance_gate_summary(gates),
            "blockers": blockers,
            "artifact_refs": {
                "manifest_root": str(self.artifact_root.relative_to(self.workspace_root)),
            },
        }
        body["content_hash"] = f"sha256:{content_hash(canonical_json(body))}"
        return body

    def write_manifest(self, manifest: dict[str, Any]) -> MvlClosureEvidenceWrite:
        self._verify_manifest_for_write(manifest)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        digest = str(manifest["content_hash"]).removeprefix("sha256:")[:16]
        path = self.artifact_root / f"{manifest['manifest_id']}-{digest}.json"
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return MvlClosureEvidenceWrite(manifest=manifest, path=path)

    def _verify_manifest_for_write(self, manifest: dict[str, Any]) -> None:
        if not isinstance(manifest, dict):
            raise ValueError("MVL closure evidence manifest must be an object")
        if manifest.get("schema_version") != MVL_CLOSURE_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("MVL closure evidence manifest schema_version is invalid")
        manifest_id = manifest.get("manifest_id")
        if not isinstance(manifest_id, str) or not manifest_id.strip():
            raise ValueError("MVL closure evidence manifest manifest_id is required")
        self._require_safe_artifact_id(
            manifest_id,
            label="manifest_id",
        )
        if manifest.get("plan_ref") != PLAN_REF:
            raise ValueError("MVL closure evidence manifest plan_ref mismatch")
        if manifest.get("workspace_id") != self.workspace_id:
            raise ValueError("MVL closure evidence manifest workspace_id mismatch")
        generated_at = manifest.get("generated_at")
        if not isinstance(generated_at, str):
            raise ValueError("generated_at must be timezone-aware UTC ISO-8601")
        self._require_aware_timestamp(generated_at, label="generated_at")
        gates = manifest.get("gates")
        if not isinstance(gates, list) or not gates:
            raise ValueError("MVL closure evidence manifest gates are required")
        self._require_manifest_gate_contract(gates)
        expected_status = self._overall_status(gates)
        if manifest.get("overall_status") != expected_status:
            raise ValueError("MVL closure evidence manifest overall_status mismatch")
        expected_acceptance_gates = self._acceptance_gate_summary(gates)
        if manifest.get("acceptance_gates") != expected_acceptance_gates:
            raise ValueError("MVL closure evidence manifest acceptance_gates mismatch")
        expected_blockers = [
            gate["diagnostic"]
            for gate in gates
            if gate["status"] in {"blocked", "incomplete"} and gate.get("diagnostic")
        ]
        if manifest.get("blockers") != expected_blockers:
            raise ValueError("MVL closure evidence manifest blockers mismatch")
        expected_artifact_refs = {
            "manifest_root": str(self.artifact_root.relative_to(self.workspace_root)),
        }
        if manifest.get("artifact_refs") != expected_artifact_refs:
            raise ValueError("MVL closure evidence manifest artifact_refs mismatch")
        expected_hash = self._content_hash_excluding_record_hash(manifest)
        if manifest.get("content_hash") != expected_hash:
            raise ValueError("MVL closure evidence manifest content_hash mismatch")

    @classmethod
    def _require_manifest_gate_contract(cls, gates: list[dict[str, Any]]) -> None:
        gate_ids: list[str] = []
        for gate in gates:
            if not isinstance(gate, dict):
                raise ValueError("MVL closure evidence manifest gate is invalid")
            gate_id = gate.get("gate_id")
            status = gate.get("status")
            if not isinstance(gate_id, str) or not gate_id.strip():
                raise ValueError("MVL closure evidence manifest gate_id is required")
            cls._require_manifest_gate_fields(gate_id=gate_id, gate=gate)
            if status not in {"pass", "fail", "blocked", "incomplete"}:
                raise ValueError("MVL closure evidence manifest gate status is invalid")
            description = gate.get("description")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(
                    "MVL closure evidence manifest gate description is required"
                )
            evidence_ref = gate.get("evidence_ref")
            if status == "pass" and (
                not isinstance(evidence_ref, str) or not evidence_ref.strip()
            ):
                raise ValueError(
                    "MVL closure evidence manifest passing gate missing evidence_ref"
                )
            if status == "pass" and gate.get("diagnostic") is not None:
                raise ValueError(
                    "MVL closure evidence manifest passing gate must not carry diagnostic"
                )
            if status == "pass" and not cls._valid_gate_evidence_ref(
                gate_id=gate_id,
                evidence_ref=evidence_ref,
            ):
                raise ValueError(
                    f"MVL closure evidence manifest {gate_id} evidence_ref is invalid"
                )
            diagnostic = gate.get("diagnostic")
            if status != "pass" and (
                not isinstance(diagnostic, str) or not diagnostic.strip()
            ):
                raise ValueError(
                    "MVL closure evidence manifest non-pass gate missing diagnostic"
                )
            if status != "pass" and gate.get("evidence_ref") is not None:
                raise ValueError(
                    "MVL closure evidence manifest non-pass gate must not carry evidence_ref"
                )
            gate_ids.append(gate_id)
        if gate_ids != list(REQUIRED_GATE_IDS):
            raise ValueError("MVL closure evidence manifest gate order/coverage mismatch")

    @staticmethod
    def _require_manifest_gate_fields(
        *,
        gate_id: str,
        gate: dict[str, Any],
    ) -> None:
        base_fields = {"gate_id", "status", "description", "evidence_ref", "diagnostic"}
        extra_fields_by_gate = {
            "document_inference_e2e_flag_criteria": {"criteria"},
            "e2e_release_flag_policy": {"policy"},
        }
        allowed_fields = base_fields | extra_fields_by_gate.get(gate_id, set())
        unexpected = sorted(set(gate) - allowed_fields)
        if unexpected:
            raise ValueError(
                f"MVL closure evidence manifest {gate_id} has unexpected gate fields: "
                f"{unexpected}"
            )
        if gate_id == "document_inference_e2e_flag_criteria" and not isinstance(
            gate.get("criteria"),
            dict,
        ):
            raise ValueError(
                "MVL closure evidence manifest document_inference_e2e_flag_criteria "
                "criteria is required"
            )
        if gate_id == "e2e_release_flag_policy" and not isinstance(
            gate.get("policy"),
            dict,
        ):
            raise ValueError(
                "MVL closure evidence manifest e2e_release_flag_policy policy is required"
            )

    @classmethod
    def _valid_gate_evidence_ref(cls, *, gate_id: str, evidence_ref: str) -> bool:
        command_gate_refs = {
            "backend_focused_regression": "command:backend_focused_regression",
            "frontend_focused_regression": "command:frontend_focused_regression",
            "frontend_typecheck": "command:frontend_typecheck",
            "plan_spec_sync": "command:plan_spec_sync",
            "doc_semantic_alignment": "command:doc_semantic_alignment",
            "graph_schema_learning_loop_e2e": "command:graph_schema_learning_loop_e2e",
            "open_source_docx_pdf_viewer_e2e": "command:docx_pdf_viewer_e2e",
        }
        if gate_id in command_gate_refs:
            return evidence_ref == command_gate_refs[
                gate_id
            ] or cls._valid_workspace_ref(
                evidence_ref,
                prefix="artifacts/mvl-closure/release-command-results/",
            )
        if gate_id == "document_inference_e2e_flag_criteria":
            prefix = "e2e_flag_criteria_report:"
            return evidence_ref.startswith(prefix) and cls._valid_sha256_ref(
                evidence_ref.removeprefix(prefix)
            )
        if gate_id == "e2e_release_flag_policy":
            prefix = "e2e_flag_policy_report:"
            return evidence_ref.startswith(prefix) and cls._valid_sha256_ref(
                evidence_ref.removeprefix(prefix)
            )
        if gate_id == "llm_proposal_evidence":
            return evidence_ref.startswith(
                LLM_PROPOSAL_EVIDENCE_PREFIX
            ) and cls._safe_ref_id(
                evidence_ref.removeprefix(LLM_PROPOSAL_EVIDENCE_PREFIX)
            )
        return False

    @staticmethod
    def _valid_workspace_ref(value: str, *, prefix: str) -> bool:
        if not value.startswith(prefix):
            return False
        suffix = value.removeprefix(prefix)
        path = Path(value)
        return (
            bool(suffix.strip())
            and suffix.endswith(".json")
            and not path.is_absolute()
            and "\\" not in value
            and ".." not in path.parts
        )

    @staticmethod
    def _safe_ref_id(value: str) -> bool:
        path = Path(value)
        return (
            bool(value.strip())
            and not path.is_absolute()
            and "/" not in value
            and "\\" not in value
            and ".." not in path.parts
        )

    @staticmethod
    def _require_safe_artifact_id(value: str, *, label: str) -> None:
        if not isinstance(value, str):
            raise ValueError(
                f"MVL closure evidence manifest {label} must be a safe artifact id"
            )
        path = Path(value)
        if (
            not value.strip()
            or path.is_absolute()
            or "/" in value
            or "\\" in value
            or ".." in path.parts
        ):
            raise ValueError(
                f"MVL closure evidence manifest {label} must be a safe artifact id"
            )

    @staticmethod
    def _require_aware_timestamp(value: str, *, label: str) -> None:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError(f"{label} must be timezone-aware UTC ISO-8601")

    @staticmethod
    def _require_command_result_statuses(command_results: dict[str, str]) -> None:
        if not isinstance(command_results, Mapping):
            raise ValueError("MVL closure command results must be a mapping")
        invalid_keys = [
            key
            for key in command_results
            if not isinstance(key, str) or not key.strip()
        ]
        if invalid_keys:
            raise ValueError("MVL closure command result keys are invalid")
        unknown = sorted(set(command_results) - COMMAND_KEYS)
        if unknown:
            raise ValueError(f"MVL closure command result keys are unknown: {unknown}")
        invalid = {
            key: value
            for key, value in command_results.items()
            if value not in COMMAND_RESULT_STATUSES
        }
        if invalid:
            raise ValueError(f"MVL closure command result statuses are invalid: {invalid}")

    def _verified_command_evidence_refs(
        self,
        *,
        manifest_id: str,
        command_results: dict[str, str],
        command_evidence_refs: dict[str, str],
    ) -> dict[str, str]:
        if not isinstance(command_evidence_refs, Mapping):
            raise ValueError("release command evidence refs must be a mapping")
        invalid_keys = [
            key
            for key in command_evidence_refs
            if not isinstance(key, str) or not key.strip()
        ]
        if invalid_keys:
            raise ValueError("release command evidence keys are invalid")
        invalid_refs = [
            key
            for key, artifact_ref in command_evidence_refs.items()
            if not isinstance(artifact_ref, str) or not artifact_ref.strip()
        ]
        if invalid_refs:
            raise ValueError("release command evidence refs are invalid")
        unknown = sorted(set(command_evidence_refs) - COMMAND_KEYS)
        if unknown:
            raise ValueError(f"unknown release command evidence keys: {unknown}")
        verified: dict[str, str] = {}
        for command_key, artifact_ref in command_evidence_refs.items():
            path = self._resolve_workspace_artifact_ref(
                artifact_ref,
                prefix="artifacts/mvl-closure/release-command-results/",
                label="command evidence ref",
            )
            record = json.loads(path.read_text(encoding="utf-8"))
            self._require_release_command_record_contract(record)
            expected_hash = self._content_hash_excluding_record_hash(record)
            if record.get("content_hash") != expected_hash:
                raise ValueError("release command evidence content_hash mismatch")
            if record.get("schema_version") != "mvl_release_command_result.v1":
                raise ValueError("release command evidence schema_version is invalid")
            if record.get("workspace_id") != self.workspace_id:
                raise ValueError("release command evidence workspace_id mismatch")
            if record.get("manifest_id") != manifest_id:
                raise ValueError("release command evidence manifest_id mismatch")
            if record.get("command_key") != command_key:
                raise ValueError("release command evidence command_key mismatch")
            self._require_command_payload(record.get("command"))
            self._require_command_cwd_is_workspace_relative(record)
            self._require_aware_timestamp(
                str(record.get("recorded_at") or ""),
                label="recorded_at",
            )
            expected_status = command_results.get(command_key)
            if record.get("status") != expected_status:
                raise ValueError("release command evidence status mismatch")
            self._require_command_returncode_matches_status(record)
            verified[command_key] = artifact_ref
        return verified

    @staticmethod
    def _require_release_command_record_contract(record: Any) -> None:
        if not isinstance(record, dict):
            raise ValueError("release command evidence record must be an object")
        missing = sorted(RELEASE_COMMAND_RESULT_FIELDS - set(record))
        if missing:
            raise ValueError(f"release command evidence missing fields: {missing}")
        unexpected = sorted(set(record) - RELEASE_COMMAND_RESULT_FIELDS)
        if unexpected:
            raise ValueError(f"release command evidence has unexpected fields: {unexpected}")
        if not isinstance(record.get("stdout"), str) or not isinstance(
            record.get("stderr"),
            str,
        ):
            raise ValueError("release command evidence stdout/stderr are invalid")
        if record.get("status") not in RELEASE_COMMAND_EVIDENCE_STATUSES:
            raise ValueError("release command evidence status is invalid")

    @staticmethod
    def _require_command_returncode_matches_status(record: dict[str, Any]) -> None:
        returncode = record.get("returncode")
        status = record.get("status")
        if not isinstance(returncode, int) or isinstance(returncode, bool):
            raise ValueError("release command evidence returncode is invalid")
        if status in {"passed", "skipped"} and returncode != 0:
            raise ValueError("release command evidence returncode/status mismatch")
        if status == "failed" and returncode == 0:
            raise ValueError("release command evidence returncode/status mismatch")

    @staticmethod
    def _require_command_payload(command: Any) -> None:
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(arg, str) or not arg.strip() for arg in command)
        ):
            raise ValueError("release command evidence command is invalid")

    @staticmethod
    def _require_command_cwd_is_workspace_relative(record: dict[str, Any]) -> None:
        cwd = record.get("cwd")
        if not isinstance(cwd, str) or not cwd.strip():
            raise ValueError("release command evidence cwd is invalid")
        path = Path(cwd)
        if path.is_absolute() or "\\" in cwd or ".." in path.parts:
            raise ValueError("release command evidence cwd must be workspace-relative")

    def _verified_e2e_flag_criteria_report(
        self,
        report: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if report is None:
            return None
        self._verify_content_hash(report, label="E2E flag criteria report")
        if report.get("schema_version") != MVL_E2E_FLAG_CRITERIA_REPORT_SCHEMA_VERSION:
            raise ValueError("E2E flag criteria report schema_version is invalid")
        if report.get("workspace_id") != self.workspace_id:
            raise ValueError("E2E flag criteria report workspace_id mismatch")
        if report.get("plan_ref") != PLAN_REF:
            raise ValueError("E2E flag criteria report plan_ref mismatch")
        if report.get("flag") != DOCUMENT_INFERENCE_FLAG:
            raise ValueError("E2E flag criteria report flag mismatch")
        self._require_e2e_flag_criteria_shape(report)
        return report

    @classmethod
    def _require_e2e_flag_criteria_shape(cls, report: dict[str, Any]) -> None:
        record_refs = report.get("record_refs")
        release_candidate_refs = report.get("release_candidate_refs")
        record_hashes = report.get("record_hashes")
        manifest_refs = report.get("manifest_refs")
        consecutive_pass_count = report.get("consecutive_pass_count")
        required_consecutive_passes = report.get("required_consecutive_passes")
        eligible = report.get("eligible_to_remove_flag")
        recommendation = report.get("recommendation")
        if not isinstance(record_refs, list):
            raise ValueError("E2E flag criteria report record_refs are invalid")
        if not isinstance(release_candidate_refs, list):
            raise ValueError(
                "E2E flag criteria report release_candidate_refs are required"
            )
        if not isinstance(record_hashes, list):
            raise ValueError("E2E flag criteria report record_hashes are required")
        if not isinstance(manifest_refs, list):
            raise ValueError("E2E flag criteria report manifest_refs are required")
        if (
            not isinstance(consecutive_pass_count, int)
            or isinstance(consecutive_pass_count, bool)
            or consecutive_pass_count < 0
        ):
            raise ValueError("E2E flag criteria report consecutive_pass_count is invalid")
        if (
            not isinstance(required_consecutive_passes, int)
            or isinstance(required_consecutive_passes, bool)
            or required_consecutive_passes < 1
        ):
            raise ValueError("E2E flag criteria report required_consecutive_passes is invalid")
        if not isinstance(eligible, bool):
            raise ValueError("E2E flag criteria report eligible_to_remove_flag is invalid")
        if (
            len(record_refs) != len(release_candidate_refs)
            or len(record_refs) != len(record_hashes)
            or len(record_refs) != len(manifest_refs)
        ):
            raise ValueError("E2E flag criteria report record evidence lengths mismatch")
        if any(
            not isinstance(value, str) or not cls._safe_ref_id(value)
            for value in record_refs
        ):
            raise ValueError("E2E flag criteria report record_refs are invalid")
        if len(set(record_refs)) != len(record_refs):
            raise ValueError("E2E flag criteria report record_refs contain duplicates")
        if any(
            not isinstance(value, str) or not cls._safe_ref_id(value)
            for value in release_candidate_refs
        ):
            raise ValueError("E2E flag criteria report release_candidate_refs are invalid")
        if len(set(release_candidate_refs)) != len(release_candidate_refs):
            raise ValueError(
                "E2E flag criteria report release_candidate_refs contain duplicates"
            )
        if any(
            not isinstance(value, str) or not cls._valid_sha256_ref(value)
            for value in record_hashes
        ):
            raise ValueError("E2E flag criteria report record_hashes are invalid")
        if len(set(record_hashes)) != len(record_hashes):
            raise ValueError(
                "E2E flag criteria report record_hashes contain duplicates"
            )
        if any(
            not isinstance(value, str)
            or not cls._valid_e2e_flag_criteria_manifest_ref(value)
            for value in manifest_refs
        ):
            raise ValueError("E2E flag criteria report manifest_refs are invalid")
        if len(set(manifest_refs)) != len(manifest_refs):
            raise ValueError(
                "E2E flag criteria report manifest_refs contain duplicates"
            )
        for record_ref, manifest_ref in zip(record_refs, manifest_refs, strict=True):
            if manifest_ref.startswith(PENDING_RELEASE_GATE_PREFIX) and (
                manifest_ref != f"{PENDING_RELEASE_GATE_PREFIX}{record_ref}"
            ):
                raise ValueError(
                    "E2E flag criteria report pending manifest_refs must match record_refs"
                )
        expected_eligible = consecutive_pass_count >= required_consecutive_passes
        if eligible is not expected_eligible:
            raise ValueError("E2E flag criteria report eligibility/count mismatch")
        expected_recommendation = (
            "remove_transitional_flag" if eligible else "keep_transitional_flag"
        )
        if recommendation != expected_recommendation:
            raise ValueError("E2E flag criteria report recommendation mismatch")

    @staticmethod
    def _valid_e2e_flag_criteria_manifest_ref(value: str) -> bool:
        if value.startswith(PENDING_RELEASE_GATE_PREFIX):
            run_id = value.removeprefix(PENDING_RELEASE_GATE_PREFIX)
            path = Path(run_id)
            return (
                bool(run_id.strip())
                and not path.is_absolute()
                and "/" not in run_id
                and "\\" not in run_id
                and ".." not in path.parts
            )
        if not value.startswith(WORKSPACE_CLOSURE_MANIFEST_PREFIX):
            return False
        suffix = value.removeprefix(WORKSPACE_CLOSURE_MANIFEST_PREFIX)
        if not suffix.strip() or not suffix.endswith(".json"):
            return False
        path = Path(value)
        return (
            not path.is_absolute()
            and "\\" not in value
            and ".." not in path.parts
        )

    @staticmethod
    def _valid_sha256_ref(value: str) -> bool:
        digest = value.removeprefix("sha256:")
        return (
            value.startswith("sha256:")
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest)
        )

    def _resolve_workspace_artifact_ref(
        self,
        artifact_ref: str,
        *,
        prefix: str,
        label: str,
    ) -> Path:
        if not artifact_ref.startswith(prefix):
            raise ValueError(f"{label} must be a workspace {prefix.rstrip('/')} artifact")
        suffix = artifact_ref.removeprefix(prefix)
        if not suffix.strip() or not suffix.endswith(".json"):
            raise ValueError(f"{label} must be a JSON artifact")
        workspace_root = (self.workspace_root / "workspaces" / self.workspace_id).resolve()
        expected_root = (workspace_root / prefix).resolve()
        path = (workspace_root / artifact_ref).resolve()
        if not path.is_relative_to(expected_root):
            raise ValueError(f"{label} must stay inside {prefix.rstrip('/')}")
        return path

    def _verified_e2e_flag_policy_report(
        self,
        report: dict[str, Any] | None,
        *,
        criteria_report: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if report is None:
            return None
        self._verify_content_hash(report, label="E2E flag policy report")
        self._require_e2e_flag_policy_report_fields(report)
        if report.get("schema_version") != MVL_E2E_FLAG_POLICY_REPORT_SCHEMA_VERSION:
            raise ValueError("E2E flag policy report schema_version is invalid")
        if report.get("workspace_id") != self.workspace_id:
            raise ValueError("E2E flag policy report workspace_id mismatch")
        if report.get("plan_ref") != PLAN_REF:
            raise ValueError("E2E flag policy report plan_ref mismatch")
        self._require_e2e_flag_policy_shape(
            report,
            criteria_report=criteria_report,
        )
        if (
            criteria_report is not None
            and report.get("criteria_report_hash") != criteria_report.get("content_hash")
        ):
            raise ValueError("E2E flag policy report criteria_report_hash mismatch")
        return report

    @staticmethod
    def _require_e2e_flag_policy_report_fields(report: dict[str, Any]) -> None:
        expected = {
            "schema_version",
            "workspace_id",
            "plan_ref",
            "overall_status",
            "criteria_report_hash",
            "actions",
            "content_hash",
        }
        unexpected = sorted(set(report) - expected)
        if unexpected:
            raise ValueError(
                f"E2E flag policy report has unexpected fields: {unexpected}"
            )

    def _verified_llm_evidence_ref(self, llm_evidence_ref: str | None) -> str | None:
        if llm_evidence_ref is None:
            return None
        if not isinstance(llm_evidence_ref, str):
            raise ValueError("llm_evidence_ref must use llm_proposal_evidence:<evidence_id>")
        if not llm_evidence_ref.startswith(LLM_PROPOSAL_EVIDENCE_PREFIX):
            raise ValueError("llm_evidence_ref must use llm_proposal_evidence:<evidence_id>")
        evidence_id = llm_evidence_ref.removeprefix(LLM_PROPOSAL_EVIDENCE_PREFIX)
        if not self._safe_ref_id(evidence_id):
            raise ValueError("llm_evidence_ref evidence_id must be a safe evidence id")
        evidence = GraphSchemaDiscoveryStore(
            self.workspace_root,
            workspace_id=self.workspace_id,
        ).get_llm_proposal_evidence(evidence_id)
        mismatches = []
        if evidence.get("status") != "passed":
            mismatches.append("status")
        if evidence.get("semantica_commit") != SEMANTICA_COMMIT:
            mismatches.append("semantica_commit")
        proposal_ref = str(evidence.get("proposal_ref") or "")
        current_provenance = (
            evidence.get("skill_major_version") == SKILL_MAJOR_VERSION
            and proposal_ref.startswith(LLM_PROPOSAL_ARTIFACT_PREFIX)
        )
        historical_provenance = (
            evidence.get("skill_major_version")
            == HISTORICAL_LLM_PROPOSAL_SKILL_MAJOR_VERSION
            and proposal_ref.startswith(HISTORICAL_LLM_PROPOSAL_ARTIFACT_PREFIX)
        )
        if not (current_provenance or historical_provenance):
            mismatches.append("skill_major_version")
        if evidence.get("provider_contract_version") != PROVIDER_CONTRACT_VERSION:
            mismatches.append("provider_contract_version")
        if not self._valid_sha256_ref(str(evidence.get("proposal_hash") or "")):
            mismatches.append("proposal_hash")
        if not (current_provenance or historical_provenance):
            mismatches.append("proposal_ref")
        if not mismatches:
            self._verify_llm_proposal_artifact(
                evidence,
                prefix=(
                    LLM_PROPOSAL_ARTIFACT_PREFIX
                    if current_provenance
                    else HISTORICAL_LLM_PROPOSAL_ARTIFACT_PREFIX
                ),
            )
        if mismatches:
            raise ValueError(
                "llm_evidence_ref does not identify qualifying §13 LLM evidence: "
                f"{mismatches}"
            )
        return llm_evidence_ref

    def _verify_llm_proposal_artifact(
        self, evidence: dict[str, Any], *, prefix: str
    ) -> None:
        proposal_ref = str(evidence["proposal_ref"])
        path = self._resolve_workspace_artifact_ref(
            proposal_ref,
            prefix=prefix,
            label="LLM proposal artifact ref",
        )
        proposal = json.loads(path.read_text(encoding="utf-8"))
        self._require_llm_proposal_artifact_contract(proposal)
        actual_hash = f"sha256:{content_hash(proposal)}"
        if actual_hash != evidence.get("proposal_hash"):
            raise ValueError("llm_evidence_ref proposal artifact hash mismatch")
        expected = {
            "schema_version": "semantica_llm_ontology_proposal.v1",
            "run_id": evidence.get("run_id"),
            "provider": evidence.get("provider"),
            "model": evidence.get("model"),
            "semantica_commit": evidence.get("semantica_commit"),
            "skill_major_version": evidence.get("skill_major_version"),
            "provider_contract_version": evidence.get("provider_contract_version"),
        }
        mismatches = [
            key
            for key, value in expected.items()
            if proposal.get(key) != value
        ]
        ontology = proposal.get("ontology")
        if not self._llm_proposal_ontology_is_valid(ontology):
            mismatches.append("ontology")
        if mismatches:
            raise ValueError(
                "llm_evidence_ref proposal artifact metadata mismatch: "
                f"{mismatches}"
            )

    @staticmethod
    def _llm_proposal_ontology_is_valid(ontology: Any) -> bool:
        if not isinstance(ontology, dict):
            return False
        classes = ontology.get("classes")
        if not isinstance(classes, list) or not classes:
            return False
        seen_classes: set[str] = set()
        for class_record in classes:
            if not isinstance(class_record, dict):
                return False
            name = class_record.get("name")
            if not isinstance(name, str) or not name.strip():
                return False
            normalized_name = name.strip()
            if normalized_name in seen_classes:
                return False
            seen_classes.add(normalized_name)
        properties = ontology.get("properties", [])
        if not isinstance(properties, list):
            return False
        seen_properties: set[str] = set()
        for property_record in properties:
            if not isinstance(property_record, dict):
                return False
            name = property_record.get("name")
            if not isinstance(name, str) or not name.strip():
                return False
            normalized_name = name.strip()
            if normalized_name in seen_properties:
                return False
            seen_properties.add(normalized_name)
        return True

    @staticmethod
    def _require_llm_proposal_artifact_contract(proposal: Any) -> None:
        if not isinstance(proposal, dict):
            raise ValueError("llm_evidence_ref proposal artifact must be an object")
        missing = sorted(LLM_PROPOSAL_ARTIFACT_FIELDS - set(proposal))
        if missing:
            raise ValueError(
                f"llm_evidence_ref proposal artifact missing fields: {missing}"
            )
        unexpected = sorted(set(proposal) - LLM_PROPOSAL_ARTIFACT_FIELDS)
        if unexpected:
            raise ValueError(
                f"llm_evidence_ref proposal artifact has unexpected fields: {unexpected}"
            )

    @classmethod
    def _require_e2e_flag_policy_shape(
        cls,
        report: dict[str, Any],
        *,
        criteria_report: dict[str, Any] | None,
    ) -> None:
        overall_status = report.get("overall_status")
        if overall_status not in {"pass", "incomplete"}:
            raise ValueError("E2E flag policy report overall_status is invalid")
        actions = report.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("E2E flag policy report actions are required")
        actions_by_flag: dict[str, dict[str, Any]] = {}
        duplicate_flags: list[str] = []
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError("E2E flag policy report action is invalid")
            cls._require_e2e_flag_policy_action_fields(action)
            flag = action.get("flag")
            status = action.get("status")
            if flag not in E2E_POLICY_FLAGS:
                raise ValueError("E2E flag policy report action flag is invalid")
            if status not in E2E_POLICY_ACTION_STATUSES:
                raise ValueError("E2E flag policy report action status is invalid")
            if flag in actions_by_flag:
                duplicate_flags.append(str(flag))
            actions_by_flag[str(flag)] = action
        if duplicate_flags:
            raise ValueError(
                f"E2E flag policy report actions contain duplicate flags: "
                f"{sorted(duplicate_flags)}"
            )
        if set(actions_by_flag) != E2E_POLICY_FLAGS:
            raise ValueError("E2E flag policy report actions do not cover required flags")
        cls._require_e2e_flag_policy_action_semantics(
            actions_by_flag,
            criteria_report=criteria_report,
        )
        action_statuses = {action["status"] for action in actions_by_flag.values()}
        if overall_status == "pass" and action_statuses != {"pass"}:
            raise ValueError("E2E flag policy report pass status has non-pass actions")
        if overall_status == "incomplete" and action_statuses == {"pass"}:
            raise ValueError("E2E flag policy report incomplete status has only pass actions")

    @staticmethod
    def _require_e2e_flag_policy_action_fields(action: dict[str, Any]) -> None:
        expected = {
            "flag",
            "status",
            "current_enabled",
            "criteria_eligible",
            "required_action",
            "diagnostic",
        }
        unexpected = sorted(set(action) - expected)
        if unexpected:
            raise ValueError(
                f"E2E flag policy report action has unexpected fields: {unexpected}"
            )

    @classmethod
    def _require_e2e_flag_policy_action_semantics(
        cls,
        actions_by_flag: dict[str, dict[str, Any]],
        *,
        criteria_report: dict[str, Any] | None,
    ) -> None:
        document_action = actions_by_flag[DOCUMENT_INFERENCE_FLAG]
        graph_action = actions_by_flag[GRAPH_SCHEMA_FLAG]
        criteria_eligible = (
            criteria_report.get("eligible_to_remove_flag") is True
            if criteria_report is not None
            else document_action.get("criteria_eligible") is True
        )
        cls._require_policy_action_boolean(
            document_action,
            field="current_enabled",
            flag=DOCUMENT_INFERENCE_FLAG,
        )
        cls._require_policy_action_boolean(
            document_action,
            field="criteria_eligible",
            flag=DOCUMENT_INFERENCE_FLAG,
        )
        cls._require_policy_action_boolean(
            graph_action,
            field="current_enabled",
            flag=GRAPH_SCHEMA_FLAG,
        )
        cls._require_policy_action_boolean(
            graph_action,
            field="criteria_eligible",
            flag=GRAPH_SCHEMA_FLAG,
        )
        if document_action.get("criteria_eligible") is not criteria_eligible:
            raise ValueError("E2E flag policy report document criteria_eligible mismatch")
        if graph_action.get("criteria_eligible") is not True:
            raise ValueError("E2E flag policy report graph criteria_eligible mismatch")
        expected_document_status = (
            "pass"
            if document_action.get("current_enabled") is True and criteria_eligible
            else "incomplete"
        )
        expected_graph_status = (
            "pass" if graph_action.get("current_enabled") is True else "incomplete"
        )
        if document_action.get("status") != expected_document_status:
            raise ValueError("E2E flag policy report document action status mismatch")
        if graph_action.get("status") != expected_graph_status:
            raise ValueError("E2E flag policy report graph action status mismatch")
        expected_document_required_action = (
            "remove_flag_or_keep_enabled"
            if criteria_eligible
            else "keep_transitional_flag_and_collect_clean_release_passes"
        )
        if document_action.get("required_action") != expected_document_required_action:
            raise ValueError("E2E flag policy report document required_action mismatch")
        if (
            graph_action.get("required_action")
            != "enable_in_release_job_until_e2e_is_permanently_default_on"
        ):
            raise ValueError("E2E flag policy report graph required_action mismatch")
        cls._require_policy_action_diagnostic(document_action)
        cls._require_policy_action_diagnostic(graph_action)

    @staticmethod
    def _require_policy_action_boolean(
        action: dict[str, Any],
        *,
        field: str,
        flag: str,
    ) -> None:
        if not isinstance(action.get(field), bool):
            raise ValueError(f"E2E flag policy report {flag} {field} is invalid")

    @staticmethod
    def _require_policy_action_diagnostic(action: dict[str, Any]) -> None:
        diagnostic = action.get("diagnostic")
        if action.get("status") == "pass" and diagnostic is not None:
            raise ValueError("E2E flag policy report pass action has diagnostic")
        if action.get("status") != "pass" and (
            not isinstance(diagnostic, str) or not diagnostic.strip()
        ):
            raise ValueError("E2E flag policy report non-pass action requires diagnostic")

    @staticmethod
    def _verify_content_hash(record: dict[str, Any], *, label: str) -> None:
        expected_hash = MvlClosureEvidenceService._content_hash_excluding_record_hash(
            record
        )
        if record.get("content_hash") != expected_hash:
            raise ValueError(f"{label} content_hash mismatch")

    @staticmethod
    def _content_hash_excluding_record_hash(record: dict[str, Any]) -> str:
        hashable = {key: value for key, value in record.items() if key != "content_hash"}
        return f"sha256:{content_hash(canonical_json(hashable))}"

    @staticmethod
    def _overall_status(gates: list[dict[str, Any]]) -> GateStatus:
        statuses = {gate["status"] for gate in gates}
        if "fail" in statuses:
            return "fail"
        if "blocked" in statuses:
            return "blocked"
        if "incomplete" in statuses:
            return "incomplete"
        return "pass"

    @classmethod
    def _acceptance_gate_summary(cls, gates: list[dict[str, Any]]) -> dict[str, Any]:
        by_id = {gate["gate_id"]: gate for gate in gates}
        real_integration_gate = cls._combine_gates(
            by_id,
            gate_ids=(
                "backend_focused_regression",
                "frontend_focused_regression",
                "frontend_typecheck",
                "plan_spec_sync",
                "doc_semantic_alignment",
            ),
        )
        webui_uat_gate = cls._combine_gates(
            by_id,
            gate_ids=(
                "open_source_docx_pdf_viewer_e2e",
                "document_inference_e2e_flag_criteria",
                "e2e_release_flag_policy",
                "graph_schema_learning_loop_e2e",
            ),
        )
        llm_gate = by_id.get("llm_proposal_evidence", {})
        return {
            "real_integration": {
                "result": cls._acceptance_result(real_integration_gate["status"]),
                "manifest_ref": real_integration_gate["evidence_ref"],
                "diagnostic": real_integration_gate["diagnostic"],
            },
            "live_llm": {
                "result": cls._acceptance_result(llm_gate.get("status")),
                "evidence_ref": llm_gate.get("evidence_ref"),
                "diagnostic": llm_gate.get("diagnostic"),
            },
            "webui_uat": {
                "result": cls._acceptance_result(webui_uat_gate["status"]),
                "manifest_ref": webui_uat_gate["evidence_ref"],
                "diagnostic": webui_uat_gate["diagnostic"],
            },
        }

    @staticmethod
    def _acceptance_result(status: str | None) -> str:
        if status == "pass":
            return "PASS"
        if status == "fail":
            return "FAIL"
        if status == "blocked":
            return "BLOCKED"
        return "INCOMPLETE"

    @staticmethod
    def _combine_gates(
        by_id: dict[str, dict[str, Any]],
        *,
        gate_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        selected = [
            by_id.get(gate_id, {"gate_id": gate_id, "status": "incomplete"})
            for gate_id in gate_ids
        ]
        status = MvlClosureEvidenceService._overall_status(selected)
        failing = [
            gate
            for gate in selected
            if gate.get("status") in {"fail", "blocked", "incomplete"}
        ]
        return {
            "status": status,
            "evidence_ref": (
                ",".join(str(gate.get("evidence_ref")) for gate in selected)
                if status == "pass"
                else None
            ),
            "diagnostic": (
                None
                if status == "pass"
                else "; ".join(
                    str(gate.get("diagnostic") or f"{gate.get('gate_id')} has not passed")
                    for gate in failing
                )
            ),
        }

    @staticmethod
    def _command_gate(
        gate_id: str,
        command_results: dict[str, str],
        command_key: str,
        description: str,
        command_evidence_refs: dict[str, str],
    ) -> dict[str, Any]:
        status = _command_status(command_results.get(command_key))
        return {
            "gate_id": gate_id,
            "status": status,
            "description": description,
            "evidence_ref": (
                command_evidence_refs.get(command_key) or f"command:{command_key}"
                if status == "pass"
                else None
            ),
            "diagnostic": None if status == "pass" else f"{command_key} has not passed",
        }

    @staticmethod
    def _playwright_gate(
        status_value: str | None,
        *,
        evidence_ref: str | None = None,
    ) -> dict[str, Any]:
        if status_value == "passed":
            status: GateStatus = "pass"
            diagnostic = None
        elif status_value == "failed":
            status = "fail"
            diagnostic = "graph-schema-learning-loop Playwright E2E failed"
        elif status_value == "skipped":
            status = "incomplete"
            diagnostic = "graph-schema-learning-loop Playwright E2E is still gated/skipped"
        else:
            status = "incomplete"
            diagnostic = "graph-schema-learning-loop Playwright E2E has not run"
        return {
            "gate_id": "graph_schema_learning_loop_e2e",
            "status": status,
            "description": "Full no-mock graph-schema learning-loop Playwright E2E.",
            "evidence_ref": (
                evidence_ref or "command:graph_schema_learning_loop_e2e"
                if status == "pass"
                else None
            ),
            "diagnostic": diagnostic,
        }

    @staticmethod
    def _open_source_document_viewer_gate(
        *,
        env: dict[str, str | None],
        doc_e2e_status: str | None,
        evidence_ref: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(env, Mapping):
            raise ValueError("MVL closure evidence manifest env must be a mapping")
        if doc_e2e_status == "passed":
            status: GateStatus = "pass"
            diagnostic = None
        elif doc_e2e_status == "failed":
            status = "fail"
            diagnostic = "DOCX/PDF open-source viewer E2E failed"
        else:
            status = "incomplete"
            diagnostic = "DOCX/PDF viewer E2E has not passed"
        return {
            "gate_id": "open_source_docx_pdf_viewer_e2e",
            "status": status,
            "description": "Open-source unified DOCX/PDF source-evidence viewer E2E.",
            "evidence_ref": (
                evidence_ref or "command:docx_pdf_viewer_e2e"
                if status == "pass"
                else None
            ),
            "diagnostic": diagnostic,
        }

    @staticmethod
    def _llm_gate(llm_evidence_ref: str | None) -> dict[str, Any]:
        if llm_evidence_ref:
            return {
                "gate_id": "llm_proposal_evidence",
                "status": "pass",
                "description": "Current or qualifying prior §13 live LLM proposal evidence.",
                "evidence_ref": llm_evidence_ref,
                "diagnostic": None,
            }
        return {
            "gate_id": "llm_proposal_evidence",
            "status": "blocked",
            "description": "Current or qualifying prior §13 live LLM proposal evidence.",
            "evidence_ref": None,
            "diagnostic": "No qualifying §13 live/prior LLM proposal evidence is recorded",
        }

    @staticmethod
    def _e2e_flag_criteria_gate(report: dict[str, Any] | None) -> dict[str, Any]:
        if report and report.get("eligible_to_remove_flag") is True:
            return {
                "gate_id": "document_inference_e2e_flag_criteria",
                "status": "pass",
                "description": "Removal/default-on criteria for DOCUMENT_INFERENCE_REVIEW_E2E.",
                "evidence_ref": f"e2e_flag_criteria_report:{report.get('content_hash')}",
                "diagnostic": None,
                "criteria": {
                    "recommendation": report.get("recommendation"),
                    "consecutive_pass_count": report.get("consecutive_pass_count"),
                    "required_consecutive_passes": report.get("required_consecutive_passes"),
                    "record_refs": report.get("record_refs", []),
                },
            }
        consecutive = int((report or {}).get("consecutive_pass_count") or 0)
        required = int((report or {}).get("required_consecutive_passes") or 3)
        return {
            "gate_id": "document_inference_e2e_flag_criteria",
            "status": "incomplete",
            "description": "Removal/default-on criteria for DOCUMENT_INFERENCE_REVIEW_E2E.",
            "evidence_ref": None,
            "diagnostic": (
                "DOCUMENT_INFERENCE_REVIEW_E2E removal/default-on criteria "
                f"not met ({consecutive}/{required} consecutive clean release-candidate passes)"
            ),
            "criteria": {
                "recommendation": (report or {}).get("recommendation", "keep_transitional_flag"),
                "consecutive_pass_count": consecutive,
                "required_consecutive_passes": required,
                "record_refs": (report or {}).get("record_refs", []),
            },
        }

    @staticmethod
    def _e2e_flag_policy_gate(report: dict[str, Any] | None) -> dict[str, Any]:
        if report and report.get("overall_status") == "pass":
            return {
                "gate_id": "e2e_release_flag_policy",
                "status": "pass",
                "description": "Release flag policy for transitional MVL E2E gates.",
                "evidence_ref": f"e2e_flag_policy_report:{report.get('content_hash')}",
                "diagnostic": None,
                "policy": {
                    "criteria_report_hash": report.get("criteria_report_hash"),
                    "actions": report.get("actions", []),
                },
            }
        actions = (report or {}).get("actions", [])
        diagnostics = [
            str(action.get("diagnostic"))
            for action in actions
            if action.get("diagnostic")
        ]
        return {
            "gate_id": "e2e_release_flag_policy",
            "status": "incomplete",
            "description": "Release flag policy for transitional MVL E2E gates.",
            "evidence_ref": None,
            "diagnostic": (
                "; ".join(diagnostics)
                if diagnostics
                else "Release E2E flag policy has not passed"
            ),
            "policy": {
                "criteria_report_hash": (report or {}).get("criteria_report_hash"),
                "actions": actions,
            },
        }
