"""Governed human UAT signoff artifacts for MVL closure."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from eos.artifact_hashing import canonical_json, content_hash
from services.mvl_closure_evidence_service import (
    MVL_CLOSURE_EVIDENCE_SCHEMA_VERSION,
    PLAN_REF as TECHNICAL_MVL_PLAN_REF,
)


MVL_HUMAN_UAT_SIGNOFF_SCHEMA_VERSION = "mvl_human_uat_signoff.v1"
UAT_PLAN_REF = "docs/plans/2026-08-26-mvl-human-uat-v1.md"
UatOutcome = Literal["pass", "pass_with_known_deviation", "fail"]
GateResult = Literal["PASS", "FAIL"]
CORE_GATE_IDS = {
    "source_binding",
    "graph_versioning",
    "human_semantic_review",
    "feedback_attribution",
    "no_mock",
}
REQUIRED_VISUALIZATION_IDS = ("kg", "ontology", "embedding", "temporal")
PASS_EVIDENCE_REF_REQUIREMENTS = {
    "command:graph_schema_learning_loop_e2e": "graph-schema learning-loop E2E",
    "command:docx_pdf_viewer_e2e": "DOCX/PDF viewer E2E",
    "llm_proposal_evidence:": "§13 LLM proposal evidence",
}
REQUIRED_COMMAND_EVIDENCE_REFS = frozenset(
    {
        "command:graph_schema_learning_loop_e2e",
        "command:docx_pdf_viewer_e2e",
    }
)
UAT_OUTCOME_RESULTS = {
    "pass": "PASS",
    "pass_with_known_deviation": "CONDITIONAL",
    "fail": "FAIL",
}


@dataclass(frozen=True)
class MvlHumanUatSignoffWrite:
    signoff: dict[str, Any]
    path: Path
    artifact_ref: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MvlHumanUatSignoffService:
    """Write and verify content-addressed human UAT signoff artifacts."""

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
            / "human-uat"
        )

    def build_signoff(
        self,
        *,
        signoff_id: str,
        signer_actor_id: str,
        outcome: UatOutcome,
        evidence_refs: list[str],
        notes: str,
        run_id: str,
        superproject_commit: str,
        hermes_agent_commit: str,
        hermes_workspace_commit: str,
        semantica_sha: str,
        real_integration_gate_result: GateResult,
        live_llm_result: GateResult,
        playwright_result: GateResult,
        browser_console_clean: bool,
        cross_workspace_isolation: GateResult,
        persistence_restart: GateResult,
        required_visualizations: dict[str, GateResult],
        known_deviations: list[dict[str, Any]] | None = None,
        signed_at: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(signoff_id, str):
            self._require_nonblank_strings(signoff_id=signoff_id)
        self._require_safe_artifact_id(signoff_id, label="signoff_id")
        self._require_nonblank_strings(
            signer_actor_id=signer_actor_id,
            outcome=outcome,
            notes=notes,
            run_id=run_id,
            superproject_commit=superproject_commit,
            hermes_agent_commit=hermes_agent_commit,
            hermes_workspace_commit=hermes_workspace_commit,
            semantica_sha=semantica_sha,
        )
        if outcome not in {"pass", "pass_with_known_deviation", "fail"}:
            raise ValueError("human UAT outcome is invalid")
        self._validate_evidence_refs(evidence_refs)
        signed_at = signed_at or _utc_now()
        self._require_utc_timestamp(signed_at, label="signed_at")
        deviations = known_deviations if known_deviations is not None else []
        self._validate_known_deviations_list(deviations)
        if outcome == "pass" and deviations:
            raise ValueError("pass UAT signoff cannot include known_deviations")
        if outcome == "pass_with_known_deviation" or deviations:
            self._validate_known_deviations(deviations)
        self._validate_acceptance_results(
            outcome=outcome,
            real_integration_gate_result=real_integration_gate_result,
            live_llm_result=live_llm_result,
            playwright_result=playwright_result,
            browser_console_clean=browser_console_clean,
            cross_workspace_isolation=cross_workspace_isolation,
            persistence_restart=persistence_restart,
            required_visualizations=required_visualizations,
        )
        self._validate_pass_evidence_refs(outcome=outcome, evidence_refs=evidence_refs)
        result = UAT_OUTCOME_RESULTS[outcome]
        uat_signoff = {
            "feature": "SemanticaMemoryProvider + Canonical Schema Discovery",
            "result": result,
            "run_id": run_id,
            "executed_at": signed_at,
            "tester": signer_actor_id,
            "superproject_commit": superproject_commit,
            "hermes_agent_commit": hermes_agent_commit,
            "hermes_workspace_commit": hermes_workspace_commit,
            "semantica_sha": semantica_sha,
            "real_integration_gate_result": real_integration_gate_result,
            "live_llm_result": live_llm_result,
            "playwright_result": playwright_result,
            "browser_console_clean": browser_console_clean,
            "cross_workspace_isolation": cross_workspace_isolation,
            "persistence_restart": persistence_restart,
            "required_visualizations": {
                key: required_visualizations[key] for key in REQUIRED_VISUALIZATION_IDS
            },
            "open_deviations": deviations,
            "evidence_refs": evidence_refs,
        }
        signoff = {
            "schema_version": MVL_HUMAN_UAT_SIGNOFF_SCHEMA_VERSION,
            "signoff_id": signoff_id,
            "workspace_id": self.workspace_id,
            "plan_ref": UAT_PLAN_REF,
            "signer_actor_id": signer_actor_id,
            "signed_at": signed_at,
            "outcome": outcome,
            "evidence_refs": evidence_refs,
            "known_deviations": deviations,
            "notes": notes,
            "uat_signoff": uat_signoff,
        }
        signoff["content_hash"] = f"sha256:{content_hash(canonical_json(signoff))}"
        return signoff

    def write_signoff(self, signoff: dict[str, Any]) -> MvlHumanUatSignoffWrite:
        self.verify_signoff(signoff)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        digest = str(signoff["content_hash"]).removeprefix("sha256:")[:16]
        artifact_ref = f"artifacts/mvl-closure/human-uat/{signoff['signoff_id']}-{digest}.json"
        path = self.workspace_root / "workspaces" / self.workspace_id / artifact_ref
        path.write_text(
            json.dumps(signoff, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return MvlHumanUatSignoffWrite(
            signoff=signoff,
            path=path,
            artifact_ref=artifact_ref,
        )

    def load_verified_ref(self, artifact_ref: str) -> dict[str, Any]:
        path = self._resolve_workspace_artifact_ref(
            artifact_ref,
            prefix="artifacts/mvl-closure/human-uat/",
            label="human UAT ref",
        )
        signoff = json.loads(path.read_text(encoding="utf-8"))
        self.verify_signoff(signoff)
        return signoff

    def load_closure_manifest_evidence_refs(self, artifact_ref: str) -> list[str]:
        path = self._resolve_workspace_artifact_ref(
            artifact_ref,
            prefix="artifacts/mvl-closure/evidence-manifests/",
            label="closure manifest ref",
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return self.evidence_refs_from_closure_manifest(
            manifest,
            closure_manifest_ref=artifact_ref,
        )

    def evidence_refs_from_closure_manifest(
        self,
        closure_manifest: dict[str, Any],
        *,
        closure_manifest_ref: str | None = None,
    ) -> list[str]:
        if not isinstance(closure_manifest, dict):
            raise ValueError("closure manifest must be an object")
        if closure_manifest.get("schema_version") != MVL_CLOSURE_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("closure manifest schema_version is invalid")
        if closure_manifest.get("workspace_id") != self.workspace_id:
            raise ValueError("closure manifest workspace_id mismatch")
        if closure_manifest.get("plan_ref") != TECHNICAL_MVL_PLAN_REF:
            raise ValueError("closure manifest plan_ref mismatch")
        expected_hash = self._content_hash_excluding_record_hash(closure_manifest)
        if closure_manifest.get("content_hash") != expected_hash:
            raise ValueError("closure manifest content_hash mismatch")
        refs: list[str] = []
        if closure_manifest_ref:
            self._resolve_workspace_artifact_ref(
                closure_manifest_ref,
                prefix="artifacts/mvl-closure/evidence-manifests/",
                label="closure manifest ref",
            )
            refs.append(closure_manifest_ref)
        refs.append(str(closure_manifest["content_hash"]))
        gates = closure_manifest.get("gates")
        if not isinstance(gates, list):
            raise ValueError("closure manifest requires gates")
        self._require_non_uat_gates_ready_for_signoff(gates)
        for gate in gates:
            if not isinstance(gate, dict):
                raise ValueError("closure manifest gates must be objects")
            if gate.get("status") != "pass":
                continue
            evidence_ref = gate.get("evidence_ref")
            if not evidence_ref:
                continue
            if not isinstance(evidence_ref, str):
                raise ValueError("closure manifest gate evidence_ref must be a string")
            refs.extend(ref.strip() for ref in evidence_ref.split(",") if ref.strip())
            command_alias = self._canonical_command_alias_for_gate(str(gate.get("gate_id")))
            if command_alias:
                refs.append(command_alias)
        return list(dict.fromkeys(refs))

    @staticmethod
    def _require_non_uat_gates_ready_for_signoff(gates: list[Any]) -> None:
        not_ready = [
            {
                "gate_id": gate.get("gate_id"),
                "status": gate.get("status"),
            }
            for gate in gates
            if isinstance(gate, dict)
            and gate.get("gate_id") != "human_uat"
            and gate.get("status") != "pass"
        ]
        if not_ready:
            raise ValueError(
                "closure manifest is not ready for human UAT signoff; "
                f"non-UAT gates are not passing: {not_ready}"
            )

    @staticmethod
    def _canonical_command_alias_for_gate(gate_id: str) -> str | None:
        aliases = {
            "graph_schema_learning_loop_e2e": "command:graph_schema_learning_loop_e2e",
            "open_source_docx_pdf_viewer_e2e": "command:docx_pdf_viewer_e2e",
        }
        return aliases.get(gate_id)

    def verify_signoff(self, signoff: dict[str, Any]) -> None:
        if not isinstance(signoff, dict):
            raise ValueError("human UAT signoff must be an object")
        self._require_signoff_fields(signoff)
        if signoff.get("schema_version") != MVL_HUMAN_UAT_SIGNOFF_SCHEMA_VERSION:
            raise ValueError("human UAT signoff schema_version is invalid")
        if signoff.get("workspace_id") != self.workspace_id:
            raise ValueError("human UAT signoff workspace_id mismatch")
        if signoff.get("plan_ref") != UAT_PLAN_REF:
            raise ValueError("human UAT signoff plan_ref mismatch")
        self._require_nonblank_strings(
            signoff_id=signoff.get("signoff_id"),
            signer_actor_id=signoff.get("signer_actor_id"),
            outcome=signoff.get("outcome"),
            notes=signoff.get("notes"),
        )
        self._require_safe_artifact_id(
            str(signoff.get("signoff_id") or ""),
            label="signoff_id",
        )
        if signoff.get("outcome") not in UAT_OUTCOME_RESULTS:
            raise ValueError("human UAT outcome is invalid")
        evidence_refs = signoff.get("evidence_refs")
        self._validate_evidence_refs(evidence_refs)
        self._require_utc_timestamp(signoff.get("signed_at"), label="signed_at")
        outcome = signoff.get("outcome")
        deviations = signoff.get("known_deviations")
        self._validate_known_deviations_list(deviations)
        if outcome == "pass" and deviations:
            raise ValueError("pass UAT signoff cannot include known_deviations")
        if outcome == "pass_with_known_deviation" or deviations:
            self._validate_known_deviations(deviations)
        self._verify_uat_signoff_block(signoff)
        self._validate_pass_evidence_refs(
            outcome=str(outcome or ""),
            evidence_refs=evidence_refs,
        )
        expected_hash = self._content_hash_excluding_record_hash(signoff)
        if signoff.get("content_hash") != expected_hash:
            raise ValueError("human UAT signoff content_hash mismatch")

    @staticmethod
    def _require_signoff_fields(signoff: dict[str, Any]) -> None:
        expected = {
            "schema_version",
            "signoff_id",
            "workspace_id",
            "plan_ref",
            "signer_actor_id",
            "signed_at",
            "outcome",
            "evidence_refs",
            "known_deviations",
            "notes",
            "uat_signoff",
            "content_hash",
        }
        unexpected = sorted(set(signoff) - expected)
        if unexpected:
            raise ValueError(
                f"human UAT signoff has unexpected fields: {unexpected}"
            )

    @staticmethod
    def _content_hash_excluding_record_hash(record: dict[str, Any]) -> str:
        hashable = {key: value for key, value in record.items() if key != "content_hash"}
        return f"sha256:{content_hash(canonical_json(hashable))}"

    @staticmethod
    def _validate_known_deviations(deviations: Any) -> None:
        if not deviations:
            raise ValueError("pass_with_known_deviation requires known_deviations")
        MvlHumanUatSignoffService._validate_known_deviations_list(deviations)
        gate_ids: list[str] = []
        for deviation in deviations:
            unexpected = sorted(
                set(deviation) - {"gate_id", "non_semantic", "rationale"}
            )
            if unexpected:
                raise ValueError(
                    f"known deviation has unexpected fields: {unexpected}"
                )
            gate_id = deviation.get("gate_id")
            if not isinstance(gate_id, str) or not gate_id.strip():
                raise ValueError("known deviation gate_id must be a safe evidence id")
            if not MvlHumanUatSignoffService._safe_ref_id(gate_id):
                raise ValueError("known deviation gate_id must be a safe evidence id")
            gate_ids.append(gate_id)
            if gate_id in CORE_GATE_IDS:
                raise ValueError(f"known deviation may not waive core semantic gate: {gate_id}")
            if deviation.get("non_semantic") is not True:
                raise ValueError("known deviations must be marked non_semantic=true")
            rationale = deviation.get("rationale")
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError("known deviations require rationale")
        if len(set(gate_ids)) != len(gate_ids):
            raise ValueError("known deviation gate_id values must be unique")

    @staticmethod
    def _validate_known_deviations_list(deviations: Any) -> None:
        if not isinstance(deviations, list) or any(
            not isinstance(deviation, dict) for deviation in deviations
        ):
            raise ValueError("known_deviations must be a list of objects")

    @staticmethod
    def _validate_evidence_refs(evidence_refs: Any) -> None:
        if not isinstance(evidence_refs, list) or not evidence_refs:
            raise ValueError("human UAT signoff requires evidence_refs")
        if any(not isinstance(ref, str) or not ref.strip() for ref in evidence_refs):
            raise ValueError("human UAT signoff evidence_refs must be nonempty strings")
        if len(set(evidence_refs)) != len(evidence_refs):
            raise ValueError("human UAT signoff evidence_refs must be unique")

    @classmethod
    def _verify_uat_signoff_block(cls, signoff: dict[str, Any]) -> None:
        block = signoff.get("uat_signoff")
        if not isinstance(block, dict):
            raise ValueError("human UAT signoff requires uat_signoff block")
        cls._require_uat_signoff_block_fields(block)
        required_strings = (
            "feature",
            "result",
            "run_id",
            "executed_at",
            "tester",
            "superproject_commit",
            "hermes_agent_commit",
            "hermes_workspace_commit",
            "semantica_sha",
        )
        cls._require_nonblank_strings(
            **{field: block.get(field) for field in required_strings}
        )
        expected_result = UAT_OUTCOME_RESULTS.get(str(signoff.get("outcome") or ""))
        if block.get("result") != expected_result:
            raise ValueError("uat_signoff result must match outcome")
        if block.get("tester") != signoff.get("signer_actor_id"):
            raise ValueError("uat_signoff tester must match signer_actor_id")
        if block.get("executed_at") != signoff.get("signed_at"):
            raise ValueError("uat_signoff executed_at must match signed_at")
        if block.get("evidence_refs") != signoff.get("evidence_refs"):
            raise ValueError("uat_signoff evidence_refs must match top-level evidence_refs")
        if block.get("open_deviations") != (signoff.get("known_deviations") or []):
            raise ValueError("uat_signoff open_deviations must match known_deviations")
        cls._require_utc_timestamp(block.get("executed_at"), label="executed_at")
        cls._validate_acceptance_results(
            outcome=str(signoff.get("outcome") or ""),
            real_integration_gate_result=str(block.get("real_integration_gate_result") or ""),
            live_llm_result=str(block.get("live_llm_result") or ""),
            playwright_result=str(block.get("playwright_result") or ""),
            browser_console_clean=block.get("browser_console_clean"),
            cross_workspace_isolation=str(block.get("cross_workspace_isolation") or ""),
            persistence_restart=str(block.get("persistence_restart") or ""),
            required_visualizations=block.get("required_visualizations"),
        )

    @staticmethod
    def _require_uat_signoff_block_fields(block: dict[str, Any]) -> None:
        expected = {
            "feature",
            "result",
            "run_id",
            "executed_at",
            "tester",
            "superproject_commit",
            "hermes_agent_commit",
            "hermes_workspace_commit",
            "semantica_sha",
            "real_integration_gate_result",
            "live_llm_result",
            "playwright_result",
            "browser_console_clean",
            "cross_workspace_isolation",
            "persistence_restart",
            "required_visualizations",
            "open_deviations",
            "evidence_refs",
        }
        unexpected = sorted(set(block) - expected)
        if unexpected:
            raise ValueError(
                f"human UAT signoff uat_signoff has unexpected fields: {unexpected}"
            )

    @classmethod
    def _validate_acceptance_results(
        cls,
        *,
        outcome: str,
        real_integration_gate_result: str,
        live_llm_result: str,
        playwright_result: str,
        browser_console_clean: Any,
        cross_workspace_isolation: str,
        persistence_restart: str,
        required_visualizations: Any,
    ) -> None:
        result_fields = {
            "real_integration_gate_result": real_integration_gate_result,
            "live_llm_result": live_llm_result,
            "playwright_result": playwright_result,
            "cross_workspace_isolation": cross_workspace_isolation,
            "persistence_restart": persistence_restart,
        }
        invalid = {
            field: value for field, value in result_fields.items() if value not in {"PASS", "FAIL"}
        }
        if invalid:
            raise ValueError(f"human UAT signoff has invalid acceptance results: {sorted(invalid)}")
        if not isinstance(browser_console_clean, bool):
            raise ValueError("human UAT signoff requires browser_console_clean boolean")
        if not isinstance(required_visualizations, dict):
            raise ValueError("human UAT signoff requires required_visualizations")
        unexpected_visualizations = sorted(
            set(required_visualizations) - set(REQUIRED_VISUALIZATION_IDS)
        )
        if unexpected_visualizations:
            raise ValueError(
                "human UAT signoff has unexpected required visualizations: "
                f"{unexpected_visualizations}"
            )
        missing_visualizations = [
            key
            for key in REQUIRED_VISUALIZATION_IDS
            if required_visualizations.get(key) not in {"PASS", "FAIL"}
        ]
        if missing_visualizations:
            raise ValueError(
                f"human UAT signoff missing required visualizations: {missing_visualizations}"
            )
        if outcome in {"pass", "pass_with_known_deviation"}:
            failed = [
                field for field, value in result_fields.items()
                if value != "PASS"
            ]
            failed.extend(
                f"required_visualizations.{key}"
                for key in REQUIRED_VISUALIZATION_IDS
                if required_visualizations[key] != "PASS"
            )
            if browser_console_clean is not True:
                failed.append("browser_console_clean")
            if failed:
                raise ValueError(f"human UAT pass cannot override failed mandatory gates: {failed}")

    @staticmethod
    def _validate_pass_evidence_refs(*, outcome: str, evidence_refs: list[str]) -> None:
        if outcome not in {"pass", "pass_with_known_deviation"}:
            return
        if any(not isinstance(ref, str) or not ref.strip() for ref in evidence_refs):
            raise ValueError("human UAT pass evidence refs must be nonempty strings")
        if len(set(evidence_refs)) != len(evidence_refs):
            raise ValueError("human UAT pass evidence refs must be unique")
        missing = [
            description
            for prefix, description in PASS_EVIDENCE_REF_REQUIREMENTS.items()
            if not any(ref.startswith(prefix) for ref in evidence_refs)
        ]
        if missing:
            raise ValueError(f"human UAT pass missing required evidence refs: {missing}")
        command_alias_prefixes = tuple(f"{ref}/" for ref in REQUIRED_COMMAND_EVIDENCE_REFS)
        malformed_command_refs = [
            ref
            for ref in evidence_refs
            if ref.startswith(command_alias_prefixes)
        ]
        if malformed_command_refs:
            raise ValueError(
                "human UAT pass command evidence refs must use exact command aliases"
            )
        invalid_llm_refs = [
            ref
            for ref in evidence_refs
            if ref.startswith("llm_proposal_evidence:")
            and not MvlHumanUatSignoffService._safe_ref_id(
                ref.removeprefix("llm_proposal_evidence:")
            )
        ]
        if invalid_llm_refs:
            raise ValueError("human UAT pass LLM evidence refs must use safe evidence ids")

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
    def _require_utc_timestamp(value: str, *, label: str) -> None:
        if not isinstance(value, str):
            raise ValueError(f"{label} must be timezone-aware UTC ISO-8601")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError(f"{label} must be timezone-aware UTC ISO-8601")

    @staticmethod
    def _require_safe_artifact_id(value: str, *, label: str) -> None:
        path = Path(value)
        if (
            not value.strip()
            or path.is_absolute()
            or "/" in value
            or "\\" in value
            or ".." in path.parts
        ):
            raise ValueError(f"human UAT {label} must be a safe artifact id")

    def _resolve_workspace_artifact_ref(
        self,
        artifact_ref: str,
        *,
        prefix: str,
        label: str,
    ) -> Path:
        if not isinstance(artifact_ref, str):
            raise ValueError(f"{label} must be a workspace artifact ref string")
        if not artifact_ref.startswith(prefix):
            raise ValueError(f"{label} must be a workspace {prefix.rstrip('/')} artifact ref")
        suffix = artifact_ref.removeprefix(prefix)
        if not suffix.strip() or not suffix.endswith(".json"):
            raise ValueError(f"{label} must be a JSON artifact ref")
        workspace_root = (self.workspace_root / "workspaces" / self.workspace_id).resolve()
        expected_root = (workspace_root / prefix).resolve()
        path = (workspace_root / artifact_ref).resolve()
        if not path.is_relative_to(expected_root):
            raise ValueError(f"{label} must stay inside {prefix.rstrip('/')}")
        return path

    @staticmethod
    def _require_nonblank_strings(**values: Any) -> None:
        missing = sorted(
            name
            for name, value in values.items()
            if not isinstance(value, str) or not value.strip()
        )
        if missing:
            raise ValueError(f"human UAT signoff missing fields: {missing}")
