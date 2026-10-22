"""Preflight assembly for MVL closure evidence manifests."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from semantier.knowledge_feature_packs.semantica_adapter import SEMANTICA_COMMIT
from storage.graph_schema_discovery_store import GraphSchemaDiscoveryStore

from .graph_schema_llm_evidence_runner import (
    PROVIDER_CONTRACT_VERSION,
    SKILL_MAJOR_VERSION,
)
from .mvl_closure_evidence_service import (
    COMMAND_KEYS,
    MvlClosureEvidenceService,
    MvlClosureEvidenceWrite,
)
from .mvl_e2e_flag_criteria_service import MvlE2eFlagCriteriaService


class MvlClosurePreflight:
    """Assemble a current closure manifest from explicit checks plus governed state."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        workspace_id: str,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if environ is not None and not isinstance(environ, Mapping):
            raise ValueError("closure preflight environ must be a mapping")
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_id = workspace_id
        self.environ = environ if environ is not None else os.environ
        self.manifest_service = MvlClosureEvidenceService(
            self.workspace_root,
            workspace_id=workspace_id,
        )
        self.discovery_store = GraphSchemaDiscoveryStore(
            self.workspace_root,
            workspace_id=workspace_id,
        )
        self.e2e_flag_criteria_service = MvlE2eFlagCriteriaService(
            self.workspace_root,
            workspace_id=workspace_id,
        )

    def build(
        self,
        *,
        command_results: dict[str, str],
        manifest_id: str = "mvl-closure-current",
        e2e_flag_criteria_report: dict[str, Any] | None = None,
        llm_evidence_ref: str | None = None,
        command_evidence_refs: dict[str, str] | None = None,
        generated_at: str | None = None,
    ) -> dict[str, Any]:
        self.manifest_service._require_command_result_statuses(command_results)  # noqa: SLF001
        self.manifest_service._require_safe_artifact_id(  # noqa: SLF001
            manifest_id,
            label="manifest_id",
        )
        resolved_llm_evidence_ref = self._verified_llm_evidence_ref(llm_evidence_ref)
        criteria_report = (
            self.e2e_flag_criteria_service.evaluate()
            if e2e_flag_criteria_report is None
            else e2e_flag_criteria_report
        )
        flag_policy_report = self.e2e_flag_criteria_service.evaluate_release_flag_policy(
            report=criteria_report,
            env=dict(self.environ),
        )
        return self.manifest_service.build_manifest(
            manifest_id=manifest_id,
            command_results=command_results,
            env={},
            llm_evidence_ref=resolved_llm_evidence_ref,
            e2e_flag_criteria_report=criteria_report,
            e2e_flag_policy_report=flag_policy_report,
            command_evidence_refs=command_evidence_refs,
            generated_at=generated_at,
        )

    def write(self, **kwargs: Any) -> MvlClosureEvidenceWrite:
        return self.manifest_service.write_manifest(self.build(**kwargs))

    def _qualifying_llm_evidence_ref(self) -> str | None:
        evidence = self.discovery_store.latest_qualifying_llm_evidence(
            semantica_commit=SEMANTICA_COMMIT,
            skill_major_version=SKILL_MAJOR_VERSION,
            provider_contract_version=PROVIDER_CONTRACT_VERSION,
            require_reviewed=True,
        )
        if evidence is None:
            return None
        return f"llm_proposal_evidence:{evidence['evidence_id']}"

    def _verified_llm_evidence_ref(self, llm_evidence_ref: str | None) -> str | None:
        if llm_evidence_ref is None:
            return self._qualifying_llm_evidence_ref()
        if not isinstance(llm_evidence_ref, str):
            raise ValueError("llm_evidence_ref must use llm_proposal_evidence:<evidence_id>")
        prefix = "llm_proposal_evidence:"
        if not llm_evidence_ref.startswith(prefix):
            raise ValueError("llm_evidence_ref must use llm_proposal_evidence:<evidence_id>")
        evidence_id = llm_evidence_ref.removeprefix(prefix)
        self._require_safe_ref_id(evidence_id, label="llm_evidence_ref evidence_id")
        evidence = self.discovery_store.get_llm_proposal_evidence(evidence_id)
        mismatches = []
        if evidence.get("status") != "passed":
            mismatches.append("status")
        if evidence.get("semantica_commit") != SEMANTICA_COMMIT:
            mismatches.append("semantica_commit")
        if evidence.get("skill_major_version") != SKILL_MAJOR_VERSION:
            mismatches.append("skill_major_version")
        if evidence.get("provider_contract_version") != PROVIDER_CONTRACT_VERSION:
            mismatches.append("provider_contract_version")
        if mismatches:
            raise ValueError(
                "llm_evidence_ref does not identify qualifying §13 LLM evidence: "
                f"{mismatches}"
            )
        return llm_evidence_ref

    @staticmethod
    def _require_safe_ref_id(value: str, *, label: str) -> None:
        if not isinstance(value, str):
            raise ValueError(f"{label} must be a safe evidence id")
        path = Path(value)
        if (
            not value.strip()
            or path.is_absolute()
            or "/" in value
            or "\\" in value
            or ".." in path.parts
        ):
            raise ValueError(f"{label} must be a safe evidence id")


