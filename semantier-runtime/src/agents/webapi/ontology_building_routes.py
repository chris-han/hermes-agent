"""Authenticated read seam for content-addressed ontology build manifests."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Literal
import asyncio

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.route_policy import RouteAuthorizationClass
from agents.webapi.deps import require_authenticated_context, require_route_authorization
from contracts.business_answers_receivables import ReceivablesReportV1
from contracts.ontology_admission import (
    AdmitOntologyCommandV1,
    OntologyAdmissionDecisionV1,
    OntologyReleaseBundleV1,
    FailureAttributionV1,
    LearningEvidenceV1,
)
from contracts.normative_schema import NormativeSchemaContractV1, SchemaAdequacyReportV1
from contracts.normative_semantics import NormativeSemanticCandidateSetV1
from contracts.ontology_building import (
    ArtifactPinV1,
    CandidateGraphDiffProjectionV1,
    CandidateGraphLineageProjectionV1,
    ExploratoryGraphSnapshotV1,
    OntologyBuildManifestV1,
    OntologyOperation,
    CompetencyContractV1,
)
from contracts.ontology_validation import (
    RuntimeSemanticEvaluationBundleV1,
)
from contracts.ontology_workbench_read_model import (
    OntologyComparisonBaselineV1,
    OntologyWorkbenchReadModelV1,
)
from contracts.ontology_audit import OntologyAuditExportV1
from contracts.validation_contract import ValidationContractV1
from contracts.ontology_quality import QualityProfileV1, QualityEvaluationRunV1
from agents.webapi.knowledge_source_retirement_routes import _eos_path
from storage.knowledge_builder_store import KnowledgeBuilderStore
from storage.graph_intent_brief_store import GraphIntentBriefStore
from services.ontology_build_start_service import OntologyBuildStartRequest, OntologyBuildStartService
from services.ontology_admission_service import OntologyAdmissionService
from services.ontology_audit_service import OntologyAuditService
from services.epistemic_readiness_service import EpistemicReadinessService
from services.bounded_autonomy_transition_service import BoundedAutonomyTransitionService
from services.knowledge_use_orchestration_service import KnowledgeUseOrchestrationService
from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path


from contracts.ontology_seed_state import OntologySeedStateV1, OntologySeedIgnoreReceiptV1, ArtifactPinV1 as SeedPin
from contracts.ontology_seed_application import SeedSelectionReceiptV1, SeedApplyReceiptV1
from contracts.ontology_seed_alignment import AlignmentDecisionV1, SourceExtensionV1, OntologySeedComparisonV1
from contracts.canonical_source_ir import CanonicalSourceDocument
from services.ontology_seed_application_service import OntologySeedApplicationService, GovernedSeedActor
from services.ontology_hub_service import OntologyHubService
from contracts.ontology_catalog import CatalogProviderError
from eos.artifact_hashing import canonical_json
from services.ontology_seed_discovery_service import OntologySeedDiscoveryService, ExternalCatalogBinding
from services.ontology_catalog_provider import HuggingFaceOntologyCatalogConfig, HuggingFaceOntologyCatalogProvider
from services.ontology_paired_comparison_service import OntologyPairedComparisonService
from services.ontology_certification_comparison_service import (
    OntologyCertificationComparisonService,
)


router = APIRouter(tags=["ontology-building"])
ROUTE_ONTOLOGY_BUILDS = "/api/ontology-builds"
ROUTE_ONTOLOGY_BUILD = "/api/ontology-builds/{build_ref}"
ROUTE_ONTOLOGY_ADMIT = "/api/ontology-builds/{build_ref}/admit"
ROUTE_ONTOLOGY_WORKBENCH = "/api/ontology-builds/{build_ref}/workbench"
ROUTE_ONTOLOGY_AUDIT = "/api/ontology-builds/{build_ref}/audit"
ROUTE_ONTOLOGY_COMPARISONS = "/api/ontology-builds/{build_ref}/comparisons"
ROUTE_ONTOLOGY_COMPARISON = "/api/ontology-builds/{build_ref}/comparisons/{comparison_ref}"


class OntologyComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    baseline_ref: str = Field(min_length=1)
    baseline_hash: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    successor_hash: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    suite_ref: str | None = Field(default=None, min_length=1)
    suite_hash: str | None = Field(default=None, pattern=r"^sha256:[a-f0-9]{64}$")

    @model_validator(mode="after")
    def paired_suite(self):
        if (self.suite_ref is None) != (self.suite_hash is None):
            raise ValueError("suite ref and hash must be supplied together")
        return self


def _comparison_projection(view) -> dict:
    bundle = view.evaluator_bundle
    return {"comparison_ref": view.comparison_ref, "comparison_hash": view.comparison_hash,
        **view.result.model_dump(mode="json"),
        "evaluator": {"evaluator_ref": bundle.evaluator_ref, "evaluator_hash": bundle.evaluator_hash,
            "policy": bundle.policy.model_dump(mode="json"), "policy_hash": bundle.policy_hash},
        "suite": view.suite.model_dump(mode="json") if view.suite else None}


def _comparison_service_for_build(*, store: OntologyBuildStore, organization_id: str,
        workspace_id: str, build_ref: str):
    manifest = store.get(organization_id=organization_id, workspace_id=workspace_id,
        artifact_ref=build_ref, contract_type=OntologyBuildManifestV1)
    if manifest is None:
        raise LookupError("comparison successor build not found")
    if any(pin.artifact_kind == "ontology_certification_readiness" for pin in manifest.child_artifacts):
        return OntologyCertificationComparisonService()
    return OntologyPairedComparisonService()


@router.post(ROUTE_ONTOLOGY_COMPARISONS)
def evaluate_ontology_comparison(build_ref: str, body: OntologyComparisonRequest, request: Request) -> dict:
    ctx = require_route_authorization(request, authorization_class=RouteAuthorizationClass.TENANT_MEMBER,
                                      route_hint=ROUTE_ONTOLOGY_COMPARISONS)
    if not ctx.organization_id or not ctx.workspace_id:
        raise HTTPException(status_code=409, detail="Workspace organization context is required")
    store = OntologyBuildStore(_store_path(ctx.workspace_root))
    try:
        service = _comparison_service_for_build(
            store=store, organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id), build_ref=build_ref,
        )
        view = service.evaluate(store=store,
            organization_id=str(ctx.organization_id), workspace_id=str(ctx.workspace_id),
            baseline_ref=body.baseline_ref, baseline_hash=body.baseline_hash,
            successor_ref=build_ref, successor_hash=body.successor_hash,
            suite_ref=body.suite_ref, suite_hash=body.suite_hash)
        return _comparison_projection(view)
    except (LookupError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="Comparison inputs not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        store.close()


@router.get(ROUTE_ONTOLOGY_COMPARISON)
def get_ontology_comparison(build_ref: str, comparison_ref: str, request: Request,
        comparison_hash: str = Query(pattern=r"^sha256:[a-f0-9]{64}$")) -> dict:
    ctx = require_route_authorization(request, authorization_class=RouteAuthorizationClass.TENANT_MEMBER,
                                      route_hint=ROUTE_ONTOLOGY_COMPARISON)
    if not ctx.organization_id or not ctx.workspace_id:
        raise HTTPException(status_code=409, detail="Workspace organization context is required")
    store = OntologyBuildStore(_store_path(ctx.workspace_root))
    try:
        service = _comparison_service_for_build(
            store=store, organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id), build_ref=build_ref,
        )
        view = service.get(store=store,
            organization_id=str(ctx.organization_id), workspace_id=str(ctx.workspace_id),
            comparison_ref=comparison_ref, comparison_hash=comparison_hash)
        if view.result.report.subject.successor_manifest.artifact_ref != build_ref:
            raise LookupError("comparison belongs to another successor")
        return _comparison_projection(view)
    except (LookupError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="Comparison not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        store.close()


class OntologyAdmitRequest(BaseModel):
    """The caller names an actor only; readiness remains server-owned."""

    model_config = ConfigDict(extra="forbid")
    actor_ref: str = Field(min_length=1)


class OntologyAdmitResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: OntologyAdmissionDecisionV1
    release_bundle: OntologyReleaseBundleV1


def _store_path(workspace_root: str | Path) -> Path:
    return ontology_build_store_path(workspace_root)


@router.post(ROUTE_ONTOLOGY_BUILDS, response_model=OntologyBuildManifestV1)
def start_ontology_build(body: OntologyBuildStartRequest, request: Request) -> OntologyBuildManifestV1:
    ctx = require_route_authorization(request, authorization_class=RouteAuthorizationClass.TENANT_MEMBER,
                                      route_hint=ROUTE_ONTOLOGY_BUILDS)
    if not ctx.organization_id or not ctx.workspace_id:
        raise HTTPException(status_code=409, detail="Workspace organization context is required")
    registry = KnowledgeBuilderStore(_eos_path(), workspace_root=ctx.workspace_root)
    store = OntologyBuildStore(_store_path(ctx.workspace_root))
    try:
        intent_store = GraphIntentBriefStore(ctx.workspace_root, organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id), source_registry=registry)
        return OntologyBuildStartService().start(command=body, intent_store=intent_store, build_store=store)
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="GraphIntent not found") from exc
    except ValueError as exc:
        status = 404 if "graph intent not found" in str(exc) else 409
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    finally:
        store.close()
        registry.close()


@router.get(ROUTE_ONTOLOGY_BUILD, response_model=OntologyBuildManifestV1)
def get_ontology_build(build_ref: str, request: Request) -> OntologyBuildManifestV1:
    ctx = require_authenticated_context(request)
    if not ctx.organization_id or not ctx.workspace_id:
        raise HTTPException(status_code=409, detail="Workspace organization context is required")
    store = OntologyBuildStore(_store_path(ctx.workspace_root))
    try:
        manifest = store.get(
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id),
            artifact_ref=build_ref,
            contract_type=OntologyBuildManifestV1,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="Ontology build not found") from exc
    finally:
        store.close()
    if manifest is None:
        raise HTTPException(status_code=404, detail="Ontology build not found")
    return manifest


def _child_pin(manifest: OntologyBuildManifestV1, kind: str) -> ArtifactPinV1:
    pins = [pin for pin in manifest.child_artifacts if pin.artifact_kind == kind]
    if len(pins) != 1:
        raise HTTPException(status_code=409, detail=f"Ontology build requires one {kind} pin")
    return pins[0]


_WORKBENCH_ARTIFACTS: dict[str, tuple[type[BaseModel], str, str, bool]] = {
    "receivables_report": (ReceivablesReportV1, "artifact_ref", "artifact_hash", False),
    "exploratory_graph_snapshot": (ExploratoryGraphSnapshotV1, "snapshot_ref", "snapshot_hash", False),
    "quality_profile": (QualityProfileV1, "profile_ref", "profile_hash", False),
    "candidate_graph_lineage_projection": (CandidateGraphLineageProjectionV1, "lineage_ref", "lineage_hash", False),
    "candidate_graph_diff_projection": (CandidateGraphDiffProjectionV1, "diff_ref", "diff_hash", False),
    "ontology_seed_comparison": (OntologySeedComparisonV1, "comparison_ref", "comparison_hash", False),
    "quality_evaluation_run": (QualityEvaluationRunV1, "run_ref", "run_hash", False),
    "competency_contract": (CompetencyContractV1, "competency_contract_ref", "competency_contract_hash", False),
    "normative_schema_contract": (NormativeSchemaContractV1, "schema_ref", "schema_hash", False),
    "schema_adequacy_report": (SchemaAdequacyReportV1, "report_ref", "report_hash", False),
    "normative_semantic_candidate_set": (NormativeSemanticCandidateSetV1, "candidate_set_ref", "candidate_set_hash", False),
    "validation_contract": (ValidationContractV1, "validation_contract_ref", "validation_contract_hash", False),
    "runtime_semantic_evaluation_bundle": (RuntimeSemanticEvaluationBundleV1, "bundle_ref", "bundle_hash", True),
    "learning_evidence": (LearningEvidenceV1, "learning_evidence_ref", "learning_evidence_hash", True),
    "failure_attribution": (FailureAttributionV1, "attribution_ref", "attribution_hash", True),
}


def _comparison_baseline_projection(
    *, store: OntologyBuildStore, manifest: OntologyBuildManifestV1,
    organization_id: str, workspace_id: str,
) -> OntologyComparisonBaselineV1 | None:
    if not any(
        pin.artifact_kind == "ontology_certification_readiness"
        for pin in manifest.child_artifacts
    ):
        return None
    current = manifest
    seen = {current.build_ref}
    while current.predecessor_build_ref is not None:
        predecessor = store.get(
            organization_id=organization_id,
            workspace_id=workspace_id,
            artifact_ref=current.predecessor_build_ref,
            contract_type=OntologyBuildManifestV1,
        )
        if predecessor is None or predecessor.build_hash != current.predecessor_build_hash:
            raise ValueError("certification comparison baseline lineage is invalid")
        if predecessor.build_ref in seen:
            raise ValueError("certification comparison baseline lineage contains a cycle")
        seen.add(predecessor.build_ref)
        current = predecessor
    pins = tuple(
        pin for pin in current.child_artifacts
        if pin.artifact_kind == "exploratory_graph_snapshot"
    )
    if len(pins) != 1:
        raise ValueError("certification comparison baseline requires one snapshot")
    snapshot, _ = store.get_verified_exploratory_snapshot(
        organization_id=organization_id,
        workspace_id=workspace_id,
        pin=pins[0],
    )
    return OntologyComparisonBaselineV1(
        build_ref=current.build_ref,
        build_hash=current.build_hash,
        snapshot_ref=snapshot.snapshot_ref,
        snapshot_hash=snapshot.snapshot_hash,
    )


def _resolve_workbench_artifacts(
    *, store: OntologyBuildStore, manifest: OntologyBuildManifestV1,
    organization_id: str, workspace_id: str,
) -> dict[str, BaseModel | tuple[BaseModel, ...] | None]:
    """Resolve only exact manifest pins; never substitute a latest artifact."""

    resolved: dict[str, BaseModel | tuple[BaseModel, ...] | None] = {}
    for kind, (contract_type, ref_field, hash_field, repeated) in _WORKBENCH_ARTIFACTS.items():
        pins = tuple(pin for pin in manifest.child_artifacts if pin.artifact_kind == kind)
        if not repeated and len(pins) > 1:
            raise HTTPException(status_code=409, detail=f"Ontology build has ambiguous {kind} pins")
        values: list[BaseModel] = []
        for pin in pins:
            artifact = store.get(
                organization_id=organization_id,
                workspace_id=workspace_id,
                artifact_ref=pin.artifact_ref,
                contract_type=contract_type,
            )
            if artifact is None or (
                getattr(artifact, ref_field) != pin.artifact_ref
                or getattr(artifact, hash_field) != pin.artifact_hash
            ):
                raise HTTPException(status_code=409, detail=f"Pinned {kind} is missing or invalid")
            values.append(artifact)
        resolved[kind] = tuple(values) if repeated else (values[0] if values else None)
    return resolved


def _workbench_epistemic_projection(*, manifest: OntologyBuildManifestV1, artifacts: dict):
    admit_gate = next((gate for gate in manifest.gates if gate.operation is OntologyOperation.ADMIT), None)
    structural_ready = bool(
        admit_gate is not None and admit_gate.decision == "PROCEED" and not admit_gate.blocker_refs
    )
    reason_codes = () if structural_ready else ("ADMISSION_GATE_NOT_READY",)
    readiness = EpistemicReadinessService.derive(
        quality_evaluation=artifacts["quality_evaluation_run"],
        quality_profile=artifacts["quality_profile"],
        structural_ready=structural_ready,
        structural_reason_codes=reason_codes,
    )
    transition = BoundedAutonomyTransitionService.recommend(
        readiness=readiness,
        actor_authorized=True,
        intent_requires_transition=False,
        exploration_only=True,
    )
    return readiness, transition


@router.get(ROUTE_ONTOLOGY_AUDIT, response_model=OntologyAuditExportV1)
def export_ontology_audit(build_ref: str, request: Request) -> OntologyAuditExportV1:
    ctx = require_authenticated_context(request)
    if not ctx.organization_id or not ctx.workspace_id:
        raise HTTPException(status_code=409, detail="Workspace organization context is required")
    store = OntologyBuildStore(_store_path(ctx.workspace_root))
    try:
        return OntologyAuditService().export(
            store=store, organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id), build_ref=build_ref,
        )
    except (LookupError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="Ontology build not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        store.close()


@router.get(ROUTE_ONTOLOGY_WORKBENCH, response_model=OntologyWorkbenchReadModelV1)
def get_ontology_workbench(build_ref: str, request: Request) -> OntologyWorkbenchReadModelV1:
    ctx = require_authenticated_context(request)
    if not ctx.organization_id or not ctx.workspace_id:
        raise HTTPException(status_code=409, detail="Workspace organization context is required")
    organization_id, workspace_id = str(ctx.organization_id), str(ctx.workspace_id)
    store = OntologyBuildStore(_store_path(ctx.workspace_root))
    try:
        manifest = store.get(
            organization_id=organization_id, workspace_id=workspace_id,
            artifact_ref=build_ref, contract_type=OntologyBuildManifestV1,
        )
        if manifest is None:
            raise HTTPException(status_code=404, detail="Ontology build not found")
        artifacts = _resolve_workbench_artifacts(
            store=store, manifest=manifest, organization_id=organization_id,
            workspace_id=workspace_id,
        )
        readiness, transition = _workbench_epistemic_projection(manifest=manifest, artifacts=artifacts)
        return OntologyWorkbenchReadModelV1(
            read_model_ref=f"ontology-workbench:{manifest.build_ref}:{manifest.build_hash}",
            organization_id=organization_id,
            workspace_id=workspace_id,
            manifest=manifest,
            receivables_verification=artifacts["receivables_report"],
            exploratory_snapshot=artifacts["exploratory_graph_snapshot"],
            quality_profile=artifacts["quality_profile"],
            lineage=artifacts["candidate_graph_lineage_projection"],
            diff=artifacts["candidate_graph_diff_projection"],
            comparison_baseline=_comparison_baseline_projection(
                store=store,
                manifest=manifest,
                organization_id=organization_id,
                workspace_id=workspace_id,
            ),
            seed_comparison=artifacts["ontology_seed_comparison"],
            competency_contract=artifacts["competency_contract"],
            normative_schema=artifacts["normative_schema_contract"],
            schema_adequacy=artifacts["schema_adequacy_report"],
            semantic_candidate=artifacts["normative_semantic_candidate_set"],
            validation_contract=artifacts["validation_contract"],
            quality_evaluation=artifacts["quality_evaluation_run"],
            runtime_evaluations=artifacts["runtime_semantic_evaluation_bundle"],
            learning_evidence=artifacts["learning_evidence"],
            failure_attributions=artifacts["failure_attribution"],
            generated_at=datetime.now(timezone.utc),
            epistemic_readiness=readiness,
            transition_recommendation=transition,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="Ontology build not found") from exc
    finally:
        store.close()


@router.post(ROUTE_ONTOLOGY_ADMIT, response_model=OntologyAdmitResponse)
def admit_ontology_build(
    build_ref: str,
    body: OntologyAdmitRequest,
    request: Request,
) -> OntologyAdmitResponse:
    ctx = require_route_authorization(
        request,
        authorization_class=RouteAuthorizationClass.TENANT_ADMIN,
        route_hint=ROUTE_ONTOLOGY_ADMIT,
    )
    if body.actor_ref != ctx.user_id:
        raise HTTPException(status_code=403, detail="Actor ref does not match governed identity")
    if not ctx.organization_id or not ctx.workspace_id:
        raise HTTPException(status_code=409, detail="Workspace organization context is required")
    store = OntologyBuildStore(_store_path(ctx.workspace_root))
    try:
        try:
            organization_id, workspace_id = str(ctx.organization_id), str(ctx.workspace_id)
            manifest = store.get(
                organization_id=organization_id, workspace_id=workspace_id,
                artifact_ref=build_ref, contract_type=OntologyBuildManifestV1,
            )
            if manifest is None:
                raise KeyError("Ontology build not found")
            artifacts = _resolve_workbench_artifacts(
                store=store, manifest=manifest, organization_id=organization_id,
                workspace_id=workspace_id,
            )
            readiness, _ = _workbench_epistemic_projection(manifest=manifest, artifacts=artifacts)
            captured = {}

            def _admit() -> str:
                decision, release_bundle = OntologyAdmissionService().admit_persisted(
                    store=store, organization_id=organization_id, workspace_id=workspace_id,
                    command=AdmitOntologyCommandV1(build_ref=build_ref, actor_ref=body.actor_ref),
                    actor_authorized=True, decided_at=datetime.now(timezone.utc),
                )
                captured["decision"] = decision
                captured["release_bundle"] = release_bundle
                return decision.decision_ref

            orchestration = KnowledgeUseOrchestrationService.execute(
                readiness=readiness, actor_authorized=True, activate_requested=False,
                admit=_admit,
            )
            if orchestration.admission_receipt_ref is None:
                raise ValueError(
                    "admission epistemic readiness unresolved: "
                    + orchestration.transition.best_next_action
                )
            return OntologyAdmitResponse(
                decision=captured["decision"], release_bundle=captured["release_bundle"],
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Ontology build not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="Ontology build not found") from exc
    finally:
        store.close()


ROUTE_SEED_STATE = "/api/ontology-builds/{build_ref}/seed-state"
ROUTE_SEED_SEARCH = "/api/ontology-builds/{build_ref}/seed-search"


class OntologySeedSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_state_ref: str | None = None
    expected_state_hash: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def paired_state(self):
        if (self.expected_state_ref is None) != (self.expected_state_hash is None):
            raise ValueError("expected state ref/hash must be paired")
        return self


class OntologySeedStateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    build_ref: str
    build_hash: str
    search_status: str
    state: OntologySeedStateV1 | None


def _seed_context(request: Request, route: str):
    ctx = require_route_authorization(request, authorization_class=RouteAuthorizationClass.TENANT_MEMBER,
                                      route_hint=route)
    if not ctx.organization_id or not ctx.workspace_id:
        raise HTTPException(status_code=409, detail="Workspace organization context is required")
    return ctx


@router.get(ROUTE_SEED_STATE, response_model=OntologySeedStateResponse)
def get_ontology_seed_state(build_ref: str, request: Request):
    ctx = _seed_context(request, ROUTE_SEED_STATE)
    store = OntologyBuildStore(_store_path(ctx.workspace_root))
    try:
        manifest = store.get(organization_id=str(ctx.organization_id), workspace_id=str(ctx.workspace_id),
                             artifact_ref=build_ref, contract_type=OntologyBuildManifestV1)
        if manifest is None:
            raise HTTPException(status_code=404, detail="Ontology build not found")
        state = store.get_seed_state_head(organization_id=str(ctx.organization_id),
                                         workspace_id=str(ctx.workspace_id), build_ref=build_ref)
        if state is not None and state.build_hash != manifest.build_hash:
            raise ValueError("seed state does not bind exact build")
        return OntologySeedStateResponse(build_ref=build_ref, build_hash=manifest.build_hash,
            search_status=state.search_status if state else "not_started", state=state)
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="Ontology build not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        store.close()


async def _complete_seed_search(*, store_path: Path, organization_id: str, workspace_id: str,
                                build_ref: str, state_ref: str, state_hash: str):
    # Configuration grants catalog mechanics only; it never supplies tenant authority.
    repos = tuple(item.strip() for item in os.environ.get("SEMANTIER_ONTOLOGY_CATALOG_REPOSITORIES", "").split(",") if item.strip())
    provider = None
    try:
        if repos:
            provider = HuggingFaceOntologyCatalogProvider(HuggingFaceOntologyCatalogConfig(allowed_dataset_repo_ids=repos))
        await OntologySeedDiscoveryService().run(
            store_path=store_path, organization_id=organization_id, workspace_id=workspace_id,
            build_ref=build_ref, searching_state_ref=state_ref, searching_state_hash=state_hash,
            external_providers=(ExternalCatalogBinding(provider_id="huggingface", provider=provider),) if provider else (),
        )
    except (ValueError, LookupError, PermissionError):
        # A superseded or interrupted search leaves its recorded state for explicit retry.
        logging.getLogger(__name__).exception("Ontology seed search did not publish completion")
    finally:
        if provider is not None:
            await provider.aclose()


@router.post(ROUTE_SEED_SEARCH, response_model=OntologySeedStateV1, status_code=202)
def search_ontology_seeds(build_ref: str, body: OntologySeedSearchRequest,
                          request: Request, background_tasks: BackgroundTasks):
    ctx = _seed_context(request, ROUTE_SEED_SEARCH)
    path = _store_path(ctx.workspace_root)
    store = OntologyBuildStore(path)
    try:
        state = OntologySeedDiscoveryService().begin(
            store=store, organization_id=str(ctx.organization_id), workspace_id=str(ctx.workspace_id),
            build_ref=build_ref, expected_state_ref=body.expected_state_ref,
            expected_state_hash=body.expected_state_hash, created_at=datetime.now(timezone.utc),
        )
        background_tasks.add_task(_complete_seed_search, store_path=path,
            organization_id=str(ctx.organization_id), workspace_id=str(ctx.workspace_id),
            build_ref=build_ref, state_ref=state.state_ref, state_hash=state.state_hash)
        return state
    except (PermissionError, LookupError) as exc:
        raise HTTPException(status_code=404, detail="Ontology build not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        store.close()


ROUTE_SEED_IGNORE = "/api/ontology-builds/{build_ref}/seed-ignore"


class OntologySeedIgnoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_state_ref: str = Field(min_length=1)
    expected_state_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    recommendation_ref: str = Field(min_length=1)
    recommendation_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    reason_code: str = Field(default="USER_DECLINED", min_length=1, max_length=128)


class OntologySeedIgnoreResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: OntologySeedStateV1
    receipt: OntologySeedIgnoreReceiptV1


@router.post(ROUTE_SEED_IGNORE, response_model=OntologySeedIgnoreResponse)
def ignore_ontology_seed(build_ref: str, body: OntologySeedIgnoreRequest, request: Request):
    ctx = _seed_context(request, ROUTE_SEED_IGNORE)
    store = OntologyBuildStore(_store_path(ctx.workspace_root))
    try:
        manifest = store.get(organization_id=str(ctx.organization_id), workspace_id=str(ctx.workspace_id),
                             artifact_ref=build_ref, contract_type=OntologyBuildManifestV1)
        if manifest is None:
            raise HTTPException(status_code=404, detail="Ontology build not found")
        current = store.get_seed_state_head(organization_id=str(ctx.organization_id),
                                           workspace_id=str(ctx.workspace_id), build_ref=build_ref)
        if current is None or current.build_hash != manifest.build_hash or (
            current.state_ref, current.state_hash) != (body.expected_state_ref, body.expected_state_hash):
            raise ValueError("Ignore expected state is stale")
        matches = [item for item in current.visible_recommendations if (
            item.recommendation_ref, item.recommendation_hash) == (body.recommendation_ref, body.recommendation_hash)]
        if len(matches) != 1:
            raise ValueError("Ignore requires one exact visible recommendation pin")
        state, receipt = OntologySeedDiscoveryService().ignore(
            store=store, organization_id=str(ctx.organization_id), workspace_id=str(ctx.workspace_id),
            build_ref=build_ref, recommendation_ref=body.recommendation_ref,
            expected_state_ref=body.expected_state_ref, expected_state_hash=body.expected_state_hash,
            reason_code=body.reason_code, ignored_at=datetime.now(timezone.utc),
        )
        return OntologySeedIgnoreResponse(state=state, receipt=receipt)
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="Ontology build not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        store.close()


ROUTE_SEED_SELECT = "/api/ontology-builds/{build_ref}/seed-select"
ROUTE_SEED_INPUTS = "/api/ontology-builds/{build_ref}/seed-inputs"
ROUTE_SEED_ALIGN = "/api/ontology-builds/{build_ref}/seed-align"


class OntologySeedSelectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_state_ref: str = Field(min_length=1)
    expected_state_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    request_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
    mode: Literal["recommendation", "external"]
    recommendation_ref: str | None = None
    recommendation_hash: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    external_ref: str | None = None

    @model_validator(mode="after")
    def exact_selection(self):
        if self.mode == "recommendation":
            if not self.recommendation_ref or self.recommendation_hash is None or self.external_ref is not None:
                raise ValueError("recommendation selection requires only exact recommendation pins")
        elif not self.external_ref or self.recommendation_ref is not None or self.recommendation_hash is not None:
            raise ValueError("external selection requires only an inspected external ref")
        return self


class OntologySeedSelectResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: OntologySeedStateV1
    receipt: SeedSelectionReceiptV1


def _seed_actor(ctx):
    if not ctx.user_id:
        raise HTTPException(status_code=409, detail="Governed actor identity is required")
    return GovernedSeedActor(organization_id=str(ctx.organization_id), workspace_id=str(ctx.workspace_id),
                             actor_ref=str(ctx.user_id))


def _expected_seed(body):
    return SeedPin(artifact_ref=body.expected_state_ref, artifact_hash=body.expected_state_hash)


def _import_catalog_selection(path, actor, metadata, payload):
    store = OntologyBuildStore(path)
    try:
        return OntologyHubService().import_payload(payload=payload, metadata=metadata,
            organization_id=actor.organization_id, workspace_id=actor.workspace_id, store=store)
    finally:
        store.close()


@router.post(ROUTE_SEED_SELECT, response_model=OntologySeedSelectResponse)
async def select_ontology_seed(build_ref: str, body: OntologySeedSelectRequest, request: Request):
    ctx = _seed_context(request, ROUTE_SEED_SELECT)
    actor = _seed_actor(ctx)
    path = _store_path(ctx.workspace_root)
    store = OntologyBuildStore(path)
    provider = None
    service = OntologySeedApplicationService()
    try:
        if body.mode == "recommendation":
            selected = service.select_registry_seed(store=store, context=actor, build_ref=build_ref,
                expected_state=_expected_seed(body), request_id=body.request_id,
                recommendation=SeedPin(artifact_ref=body.recommendation_ref, artifact_hash=body.recommendation_hash))
        else:
            prior = store.get_seed_request(organization_id=actor.organization_id, workspace_id=actor.workspace_id,
                actor_ref=actor.actor_ref, operation="select_seed", request_id=body.request_id)
            if prior is not None:
                receipt, _ = prior
                if receipt.mode != "manual_external" or receipt.external_candidate_ref != body.external_ref:
                    raise ValueError("request id belongs to a different seed selection")
                snapshot_pin = SeedPin(artifact_ref=receipt.imported_snapshot.artifact_ref, artifact_hash=receipt.imported_snapshot.artifact_hash)
                profile_pin = SeedPin(artifact_ref=receipt.imported_profile.artifact_ref, artifact_hash=receipt.imported_profile.artifact_hash)
            else:
                candidate = service.prepare_external_selection(store=store, context=actor,
                    build_ref=build_ref, expected_state=_expected_seed(body), external_ref=body.external_ref)
                repos = tuple(item.strip() for item in os.environ.get("SEMANTIER_ONTOLOGY_CATALOG_REPOSITORIES", "").split(",") if item.strip())
                if candidate.upstream_id not in repos:
                    raise ValueError("external catalog is not enabled for import")
                provider = HuggingFaceOntologyCatalogProvider(HuggingFaceOntologyCatalogConfig(allowed_dataset_repo_ids=repos))
                metadata = await provider.metadata(body.external_ref)
                payload = await provider.fetch(body.external_ref)
                imported = await asyncio.to_thread(_import_catalog_selection, path, actor, metadata, payload)
                snapshot_pin = SeedPin(artifact_ref=imported.snapshot.snapshot_ref, artifact_hash=imported.snapshot.snapshot_hash)
                profile_pin = SeedPin(artifact_ref=imported.profile.profile_ref, artifact_hash=imported.profile.profile_hash)
            selected = service.record_imported_seed_selection(store=store, context=actor, build_ref=build_ref,
                expected_state=_expected_seed(body), request_id=body.request_id,
                imported_snapshot=snapshot_pin, imported_profile=profile_pin,
                inspected_external_candidate_ref=body.external_ref)
        return OntologySeedSelectResponse(state=selected.state, receipt=selected.receipt)
    except (PermissionError, LookupError) as exc:
        raise HTTPException(status_code=404, detail="Ontology build or seed not found") from exc
    except CatalogProviderError as exc:
        raise HTTPException(status_code=502, detail="Ontology catalog import failed; the source-first build remains available") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        store.close()
        if provider is not None:
            await provider.aclose()


@router.get(ROUTE_SEED_INPUTS)
def inspect_ontology_seed(build_ref: str, selection_ref: str, selection_hash: str, request: Request):
    ctx = _seed_context(request, ROUTE_SEED_INPUTS)
    store = OntologyBuildStore(_store_path(ctx.workspace_root))
    try:
        inputs = OntologySeedApplicationService().inspect_selected_seed(store=store, context=_seed_actor(ctx),
            build_ref=build_ref, selection_receipt=SeedPin(artifact_ref=selection_ref, artifact_hash=selection_hash))
        canonical = CanonicalSourceDocument.model_validate(inputs['input_envelope'].canonical_source_ir)
        blocks = {block.block_id: block for block in canonical.blocks}
        locators = {}
        for mapping in canonical.source_map.mappings:
            if mapping.source_locators:
                for block_id in mapping.block_ids:
                    locators.setdefault(block_id, mapping.source_locators[0])
        source_definitions = []
        for node in inputs['baseline_snapshot'].nodes:
            block_id = node.node_id.removeprefix('source-block:')
            if block_id not in blocks or block_id not in locators:
                continue
            content = blocks[block_id].content
            quote = content if isinstance(content, str) else canonical_json(content)
            if not quote.strip():
                continue
            source_definitions.append(dict(node_id=node.node_id, label=node.label, node_kind=node.node_kind,
                canonical_quote=dict(block_id=block_id, evidence_ref=f"evidence:canonical-block:{block_id.removeprefix('sha256:')}",
                    source_locator=locators[block_id].model_dump(mode='json'), quote=quote)))
        return dict(build_ref=build_ref, selection_ref=selection_ref, selection_hash=selection_hash,
            baseline_snapshot_ref=inputs['baseline_snapshot'].snapshot_ref,
            baseline_snapshot_hash=inputs['baseline_snapshot'].snapshot_hash,
            seed_snapshot_ref=inputs['seed_snapshot'].snapshot_ref, seed_snapshot_hash=inputs['seed_snapshot'].snapshot_hash,
            source_definitions=source_definitions,
            seed_inventory=[item.model_dump(mode='json') for item in inputs['seed_snapshot'].canonical_inventory],
            seed_profile=inputs['seed_profile'].model_dump(mode='json'))
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="Ontology build or seed not found") from exc
    except (ValueError, LookupError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        store.close()


class OntologySeedAlignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_state_ref: str = Field(min_length=1)
    expected_state_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    selection_ref: str = Field(min_length=1)
    selection_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    request_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
    decisions: tuple[AlignmentDecisionV1, ...] = Field(min_length=1, max_length=128)
    extensions: tuple[SourceExtensionV1, ...] = Field(default=(), max_length=128)


class OntologySeedAlignResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: OntologySeedStateV1
    receipt: SeedApplyReceiptV1
    manifest: OntologyBuildManifestV1 | None


@router.post(ROUTE_SEED_ALIGN, response_model=OntologySeedAlignResponse)
def align_ontology_seed(build_ref: str, body: OntologySeedAlignRequest, request: Request):
    ctx = _seed_context(request, ROUTE_SEED_ALIGN)
    store = OntologyBuildStore(_store_path(ctx.workspace_root))
    try:
        applied = OntologySeedApplicationService().apply_selected_seed(store=store, context=_seed_actor(ctx),
            build_ref=build_ref, expected_state=_expected_seed(body), request_id=body.request_id,
            selection_receipt=SeedPin(artifact_ref=body.selection_ref, artifact_hash=body.selection_hash),
            decisions=body.decisions, extensions=body.extensions)
        return OntologySeedAlignResponse(state=applied.state, receipt=applied.receipt, manifest=applied.manifest)
    except (PermissionError, LookupError) as exc:
        raise HTTPException(status_code=404, detail="Ontology build or seed not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        store.close()


__all__ = [
    "OntologyAdmitRequest",
    "OntologyAdmitResponse",
    "ROUTE_ONTOLOGY_ADMIT",
    "ROUTE_ONTOLOGY_AUDIT",
    "ROUTE_ONTOLOGY_WORKBENCH",
    "ROUTE_ONTOLOGY_BUILD",
    "get_ontology_build",
    "router",
]
