"""Parent reality-manifest handoff for MVL acceptance gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eos.artifact_hashing import canonical_json, content_hash
from .mvl_closure_evidence_service import MVL_CLOSURE_EVIDENCE_SCHEMA_VERSION, PLAN_REF


MVL_REALITY_HANDOFF_SCHEMA_VERSION = "mvl_reality_handoff_manifest.v1"
ACCEPTANCE_PLAN_REF = (
    "docs/plans/2026-08-19-mvl-implementation-sequence-and-contracts-v1.md"
)
ACCEPTANCE_GATE_FIELDS = {
    "real_integration": ("manifest_ref",),
    "live_llm": ("evidence_ref",),
    "webui_uat": ("manifest_ref",),
}
HANDOFF_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "plan_ref",
        "acceptance_plan_ref",
        "workspace_id",
        "closure_manifest_ref",
        "closure_manifest_hash",
        "overall_status",
        "acceptance_gates",
        "reality_certification_ready",
        "content_hash",
    }
)
ACCEPTANCE_GATE_RESULTS = frozenset({"PASS", "FAIL", "BLOCKED", "INCOMPLETE"})
CLOSURE_OVERALL_STATUSES = frozenset({"pass", "fail", "blocked", "incomplete"})


@dataclass(frozen=True)
class MvlRealityHandoffWrite:
    manifest: dict[str, Any]
    path: Path
    artifact_ref: str


class MvlRealityHandoffService:
    """Write the parent reality-manifest handoff required by acceptance gates."""

    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        if not workspace_id:
            raise ValueError("workspace_id is required")
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_id = workspace_id
        self.artifact_ref = "artifacts/semantica-memory-provider/reality-manifest.json"
        self.path = self.workspace_root / "workspaces" / workspace_id / self.artifact_ref

    def build_manifest(
        self,
        *,
        closure_manifest: dict[str, Any],
        closure_manifest_ref: str,
    ) -> dict[str, Any]:
        if not self._valid_closure_manifest_ref(closure_manifest_ref):
            raise ValueError("closure_manifest_ref must be a workspace closure evidence artifact")
        self._verify_closure_manifest(closure_manifest)
        closure_manifest_hash = str(closure_manifest.get("content_hash") or "")
        acceptance_gates = closure_manifest.get("acceptance_gates")
        if not isinstance(acceptance_gates, dict):
            raise ValueError("closure manifest missing acceptance_gates")
        self._require_acceptance_gate_summary(acceptance_gates)
        manifest = {
            "schema_version": MVL_REALITY_HANDOFF_SCHEMA_VERSION,
            "plan_ref": PLAN_REF,
            "acceptance_plan_ref": ACCEPTANCE_PLAN_REF,
            "workspace_id": self.workspace_id,
            "closure_manifest_ref": closure_manifest_ref,
            "closure_manifest_hash": closure_manifest_hash,
            "overall_status": closure_manifest.get("overall_status"),
            "acceptance_gates": acceptance_gates,
            "reality_certification_ready": (
                closure_manifest.get("overall_status") == "pass"
                and all(gate.get("result") == "PASS" for gate in acceptance_gates.values())
            ),
        }
        manifest["content_hash"] = f"sha256:{content_hash(canonical_json(manifest))}"
        return manifest

    def _verify_closure_manifest(self, closure_manifest: dict[str, Any]) -> None:
        if not isinstance(closure_manifest, dict):
            raise ValueError("closure manifest must be an object")
        if closure_manifest.get("schema_version") != MVL_CLOSURE_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("closure manifest schema_version is invalid")
        if closure_manifest.get("workspace_id") != self.workspace_id:
            raise ValueError("closure manifest workspace_id mismatch")
        if closure_manifest.get("plan_ref") != PLAN_REF:
            raise ValueError("closure manifest plan_ref mismatch")
        if closure_manifest.get("overall_status") not in CLOSURE_OVERALL_STATUSES:
            raise ValueError("closure manifest overall_status is invalid")
        closure_manifest_hash = str(closure_manifest.get("content_hash") or "")
        if not closure_manifest_hash.startswith("sha256:"):
            raise ValueError("closure manifest content_hash must be sha256-pinned")
        hashable = {
            key: value
            for key, value in closure_manifest.items()
            if key != "content_hash"
        }
        expected_hash = f"sha256:{content_hash(canonical_json(hashable))}"
        if closure_manifest_hash != expected_hash:
            raise ValueError("closure manifest content_hash mismatch")

    @staticmethod
    def _require_acceptance_gate_summary(acceptance_gates: dict[str, Any]) -> None:
        expected = set(ACCEPTANCE_GATE_FIELDS)
        actual = set(acceptance_gates)
        missing = sorted(expected - actual)
        if missing:
            raise ValueError(f"closure manifest acceptance_gates missing fields: {missing}")
        unexpected = sorted(actual - expected)
        if unexpected:
            raise ValueError(
                f"closure manifest acceptance_gates has unknown fields: {unexpected}"
            )
        for gate_id, required_pass_fields in ACCEPTANCE_GATE_FIELDS.items():
            gate = acceptance_gates[gate_id]
            if not isinstance(gate, dict):
                raise ValueError(
                    f"closure manifest acceptance_gates.{gate_id} must be an object"
                )
            allowed_fields = {"result", "diagnostic", *required_pass_fields}
            unexpected_fields = sorted(set(gate) - allowed_fields)
            if unexpected_fields:
                raise ValueError(
                    "closure manifest acceptance_gates."
                    f"{gate_id} has unexpected fields: {unexpected_fields}"
                )
            result = gate.get("result")
            if result not in ACCEPTANCE_GATE_RESULTS:
                raise ValueError(
                    f"closure manifest acceptance_gates.{gate_id}.result is invalid"
                )
            diagnostic = gate.get("diagnostic")
            if result == "PASS" and diagnostic is not None:
                raise ValueError(
                    f"closure manifest acceptance_gates.{gate_id} pass has diagnostic"
                )
            if result != "PASS" and (
                not isinstance(diagnostic, str) or not diagnostic.strip()
            ):
                raise ValueError(
                    f"closure manifest acceptance_gates.{gate_id} non-pass requires diagnostic"
                )
            if result != "PASS":
                invalid_refs = sorted(
                    field
                    for field in required_pass_fields
                    if gate.get(field) is not None
                    and not MvlRealityHandoffService._valid_acceptance_gate_ref_field(
                        gate_id=gate_id,
                        field=field,
                        value=str(gate.get(field) or ""),
                    )
                )
                if invalid_refs:
                    raise ValueError(
                        "closure manifest acceptance_gates."
                        f"{gate_id} non-pass refs are invalid: {invalid_refs}"
                    )
            if result == "PASS":
                missing_refs = sorted(
                    field
                    for field in required_pass_fields
                    if not str(gate.get(field) or "").strip()
                )
                if missing_refs:
                    raise ValueError(
                        "closure manifest acceptance_gates."
                        f"{gate_id} missing passing refs: {missing_refs}"
                    )
                MvlRealityHandoffService._require_passing_gate_ref_shapes(
                    gate_id=gate_id,
                    gate=gate,
                )

    @staticmethod
    def _require_passing_gate_ref_shapes(
        *,
        gate_id: str,
        gate: dict[str, Any],
    ) -> None:
        if gate_id == "live_llm":
            evidence_ref = str(gate.get("evidence_ref") or "")
            if not MvlRealityHandoffService._valid_llm_evidence_ref(evidence_ref):
                raise ValueError(
                    "closure manifest acceptance_gates.live_llm evidence_ref is invalid"
                )
            return
        manifest_ref = str(gate.get("manifest_ref") or "")
        refs = [ref.strip() for ref in manifest_ref.split(",") if ref.strip()]
        if not refs or any(
            not MvlRealityHandoffService._valid_acceptance_manifest_ref(ref)
            for ref in refs
        ):
            raise ValueError(
                f"closure manifest acceptance_gates.{gate_id} manifest_ref is invalid"
            )

    @staticmethod
    def _valid_acceptance_gate_ref_field(
        *,
        gate_id: str,
        field: str,
        value: str,
    ) -> bool:
        if gate_id == "live_llm" and field == "evidence_ref":
            return MvlRealityHandoffService._valid_llm_evidence_ref(value)
        if field == "manifest_ref":
            refs = [ref.strip() for ref in value.split(",") if ref.strip()]
            return bool(refs) and all(
                MvlRealityHandoffService._valid_acceptance_manifest_ref(ref)
                for ref in refs
            )
        return False

    @staticmethod
    def _valid_acceptance_manifest_ref(value: str) -> bool:
        if value.startswith("command:"):
            command_key = value.removeprefix("command:")
            return MvlRealityHandoffService._safe_ref_id(command_key)
        if value.startswith("e2e_flag_criteria_report:") or value.startswith(
            "e2e_flag_policy_report:"
        ):
            _kind, digest = value.split(":", 1)
            return MvlRealityHandoffService._valid_sha256_ref(digest)
        if MvlRealityHandoffService._valid_workspace_artifact_ref(
            value,
            prefix="artifacts/mvl-closure/human-uat/",
        ):
            return True
        return MvlRealityHandoffService._valid_workspace_artifact_ref(
            value,
            prefix="artifacts/mvl-closure/release-command-results/",
        )

    @staticmethod
    def _valid_llm_evidence_ref(value: str) -> bool:
        prefix = "llm_proposal_evidence:"
        if not value.startswith(prefix):
            return False
        return MvlRealityHandoffService._safe_ref_id(value.removeprefix(prefix))

    @staticmethod
    def _valid_workspace_artifact_ref(value: str, *, prefix: str) -> bool:
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
    def _valid_sha256_ref(value: str) -> bool:
        digest = value.removeprefix("sha256:")
        return (
            value.startswith("sha256:")
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest)
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

    def write_manifest(self, manifest: dict[str, Any]) -> MvlRealityHandoffWrite:
        self._verify_handoff_manifest(manifest)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return MvlRealityHandoffWrite(
            manifest=manifest,
            path=self.path,
            artifact_ref=self.artifact_ref,
        )

    def _verify_handoff_manifest(self, manifest: dict[str, Any]) -> None:
        if not isinstance(manifest, dict):
            raise ValueError("MVL reality handoff manifest must be an object")
        self._require_handoff_manifest_fields(manifest)
        if manifest.get("schema_version") != MVL_REALITY_HANDOFF_SCHEMA_VERSION:
            raise ValueError("MVL reality handoff manifest schema_version is invalid")
        if manifest.get("plan_ref") != PLAN_REF:
            raise ValueError("MVL reality handoff manifest plan_ref mismatch")
        if manifest.get("acceptance_plan_ref") != ACCEPTANCE_PLAN_REF:
            raise ValueError("MVL reality handoff manifest acceptance_plan_ref mismatch")
        if manifest.get("workspace_id") != self.workspace_id:
            raise ValueError("MVL reality handoff manifest workspace_id mismatch")
        closure_manifest_ref = manifest.get("closure_manifest_ref")
        if not isinstance(closure_manifest_ref, str):
            raise ValueError(
                "MVL reality handoff manifest closure_manifest_ref is invalid"
            )
        if not self._valid_closure_manifest_ref(closure_manifest_ref):
            raise ValueError(
                "MVL reality handoff manifest closure_manifest_ref is invalid"
            )
        closure_manifest_hash = manifest.get("closure_manifest_hash")
        if not isinstance(closure_manifest_hash, str):
            raise ValueError(
                "MVL reality handoff manifest closure_manifest_hash is invalid"
            )
        if not self._valid_sha256_ref(closure_manifest_hash):
            raise ValueError(
                "MVL reality handoff manifest closure_manifest_hash is invalid"
            )
        if manifest.get("overall_status") not in CLOSURE_OVERALL_STATUSES:
            raise ValueError("MVL reality handoff manifest overall_status is invalid")
        acceptance_gates = manifest.get("acceptance_gates")
        if not isinstance(acceptance_gates, dict):
            raise ValueError("MVL reality handoff manifest missing acceptance_gates")
        self._require_acceptance_gate_summary(acceptance_gates)
        expected_ready = (
            manifest.get("overall_status") == "pass"
            and all(gate.get("result") == "PASS" for gate in acceptance_gates.values())
        )
        if manifest.get("reality_certification_ready") is not expected_ready:
            raise ValueError(
                "MVL reality handoff manifest reality_certification_ready mismatch"
            )
        hashable = {
            key: value
            for key, value in manifest.items()
            if key != "content_hash"
        }
        expected_hash = f"sha256:{content_hash(canonical_json(hashable))}"
        if manifest.get("content_hash") != expected_hash:
            raise ValueError("MVL reality handoff manifest content_hash mismatch")

    @staticmethod
    def _require_handoff_manifest_fields(manifest: dict[str, Any]) -> None:
        missing = sorted(HANDOFF_MANIFEST_FIELDS - set(manifest))
        if missing:
            raise ValueError(f"MVL reality handoff manifest missing fields: {missing}")
        unexpected = sorted(set(manifest) - HANDOFF_MANIFEST_FIELDS)
        if unexpected:
            raise ValueError(
                f"MVL reality handoff manifest has unexpected fields: {unexpected}"
            )

    @staticmethod
    def _valid_closure_manifest_ref(value: str) -> bool:
        if not isinstance(value, str):
            return False
        prefix = "artifacts/mvl-closure/evidence-manifests/"
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

    def write_from_closure(
        self,
        *,
        closure_manifest: dict[str, Any],
        closure_manifest_ref: str,
    ) -> MvlRealityHandoffWrite:
        return self.write_manifest(
            self.build_manifest(
                closure_manifest=closure_manifest,
                closure_manifest_ref=closure_manifest_ref,
            )
        )
