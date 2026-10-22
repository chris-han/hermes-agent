"""Pure retained verification for controlled-trial build projections."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping

from contracts.graph_intent_brief import GraphIntentBrief
from contracts.ontology_artifact_integrity import (
    ExploratoryGraphSnapshotIntegrityV1,
    verify_exploratory_snapshot_integrity,
)
from contracts.ontology_build_inputs import OntologyBuildInputEnvelopeV1
from contracts.ontology_building import (
    ArtifactPinV1,
    CanonicalEvidenceAdequacyAssessmentV1,
    CompetencyContractV1,
    ExploratoryGraphSnapshotV1,
    OntologyBuildManifestV1,
    OntologyOperation,
)
from contracts.normative_semantics import NormativeSemanticCandidateSetV1
from contracts.ontology_paired_evaluation import (
    ArtifactPayloadPinV1,
    PairedEvaluationSubjectV1,
)
from contracts.ontology_validation import CompiledRulePlanV1
from contracts.runtime_context_reconstruction import CapturedSourceV1
from contracts.runtime_learning_trial import (
    AdvisoryCompilerInputsV1,
    AdvisoryTrialBuildInputsV1,
    AdvisoryTrialVariantV1,
)
from contracts.validation_contract import ValidationContractV1
from eos.artifact_hashing import content_hash

from .common import (
    PinKey,
    _expect_dependencies,
    _expect_identity,
    _payload,
    _pin_key,
    _require_supplied_source,
    _validate_closure,
)
from .compiler import (
    _candidate_value,
    _input_value,
    _model_payload,
    _plan_value,
    _resolve,
    _trial_receipt_value,
    _validation_value,
)
from .trial import _validate_receipt_relationship


_PROJECTION_PROFILE = "advisory_trial_projection.v1"
_PROJECTION_VERSION = "runtime_learning_trial_paired_build_v1"
_PROJECTION_KINDS = (
    "canonical_evidence_adequacy_assessment",
    "advisory_trial_build_inputs",
    "exploratory_graph_snapshot",
    "graph_intent_brief",
    "competency_contract",
    "normative_semantic_candidate_set",
    "validation_contract",
    "compiled_rule_plan",
)
_ORIGIN_KINDS = (
    "canonical_evidence_adequacy_assessment",
    "ontology_build_inputs",
    "canonical_evidence_boundary",
    "exploratory_graph_snapshot",
    "graph_intent_brief",
    "source_identity",
    "canonical_source_ir",
    "ontology_source_first_compiler",
)


def _pin(kind: str, ref: str, digest: str) -> ArtifactPinV1:
    return ArtifactPinV1(
        artifact_kind=kind,
        artifact_ref=ref,
        artifact_hash=digest,
    )


def _payload_pin(value: Any, ref_field: str, hash_field: str) -> ArtifactPayloadPinV1:
    return ArtifactPayloadPinV1.of(value, ref_field, hash_field)


def _as_pin(pin: ArtifactPayloadPinV1, kind: str) -> ArtifactPinV1:
    return _pin(kind, pin.artifact_ref, pin.artifact_hash)


def _resolve_payload_pin(
    closure: Mapping[PinKey, CapturedSourceV1],
    pin: ArtifactPayloadPinV1,
    *,
    kind: str,
    contract: type[Any],
    label: str,
) -> tuple[CapturedSourceV1, Any]:
    source = _resolve(closure, _as_pin(pin, kind))
    value = _model_payload(source, contract, label)
    ref_field, hash_field = {
        "ontology_build_manifest": ("build_ref", "build_hash"),
        "exploratory_graph_snapshot": ("snapshot_ref", "snapshot_hash"),
    }[kind]
    if (getattr(value, ref_field), getattr(value, hash_field)) != (
        pin.artifact_ref,
        pin.artifact_hash,
    ) or pin.payload_hash != "sha256:" + content_hash(value.model_dump(mode="json")):
        raise ValueError(f"{label} full payload pin mismatch")
    return source, value


def _build_input_value(source: CapturedSourceV1) -> AdvisoryTrialBuildInputsV1:
    value = _model_payload(source, AdvisoryTrialBuildInputsV1, "trial build inputs")
    _expect_identity(
        source,
        kind="advisory_trial_build_inputs",
        ref=value.input_ref,
        digest=value.input_hash,
    )
    return value


def _manifest_children(
    source: CapturedSourceV1,
    manifest: OntologyBuildManifestV1,
    closure: Mapping[PinKey, CapturedSourceV1],
    expected_kinds: tuple[str, ...],
) -> dict[str, CapturedSourceV1]:
    kinds = tuple(pin.artifact_kind for pin in manifest.child_artifacts)
    if kinds != expected_kinds or len(
        set(_pin_key(pin) for pin in manifest.child_artifacts)
    ) != len(manifest.child_artifacts):
        raise ValueError("ontology build manifest child set or order mismatch")
    expected_dependencies = list(manifest.child_artifacts)
    if manifest.predecessor_build_ref is not None:
        expected_dependencies.append(
            _pin(
                "ontology_build_manifest",
                manifest.predecessor_build_ref,
                manifest.predecessor_build_hash,
            )
        )
    _expect_dependencies(source, tuple(expected_dependencies), closure)
    return {
        pin.artifact_kind: _resolve(closure, pin) for pin in manifest.child_artifacts
    }


def _origin_context(
    build_input: AdvisoryTrialBuildInputsV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> tuple[
    CapturedSourceV1,
    OntologyBuildManifestV1,
    CapturedSourceV1,
    ExploratoryGraphSnapshotV1,
    dict[str, CapturedSourceV1],
]:
    origin_manifest_source, origin_manifest = _resolve_payload_pin(
        closure,
        build_input.origin_manifest,
        kind="ontology_build_manifest",
        contract=OntologyBuildManifestV1,
        label="origin manifest",
    )
    origin_snapshot_source, origin_snapshot = _resolve_payload_pin(
        closure,
        build_input.origin_snapshot,
        kind="exploratory_graph_snapshot",
        contract=ExploratoryGraphSnapshotV1,
        label="origin snapshot",
    )
    children = _manifest_children(
        origin_manifest_source, origin_manifest, closure, _ORIGIN_KINDS
    )
    origin_input = _model_payload(
        children["ontology_build_inputs"], OntologyBuildInputEnvelopeV1, "origin inputs"
    )
    origin_graph_source, graph_payload = _payload(children["graph_intent_brief"])
    brief = GraphIntentBrief.from_dict(graph_payload)
    if (
        origin_manifest.assurance_profile != "A0_EXPLORATORY"
        or origin_manifest.predecessor_build_ref is not None
        or origin_manifest.predecessor_build_hash is not None
        or origin_snapshot.change_cause != "initial_exploration"
        or origin_snapshot.predecessor_snapshot_ref is not None
        or origin_snapshot.predecessor_snapshot_hash is not None
        or children["exploratory_graph_snapshot"].source
        != origin_snapshot_source.source
        or children["ontology_build_inputs"].source
        != _pin(
            "ontology_build_inputs",
            origin_input.envelope_ref,
            origin_input.envelope_hash,
        )
        or origin_manifest_source.source
        != _pin(
            "ontology_build_manifest",
            origin_manifest.build_ref,
            origin_manifest.build_hash,
        )
        or origin_snapshot_source.source
        != _pin(
            "exploratory_graph_snapshot",
            origin_snapshot.snapshot_ref,
            origin_snapshot.snapshot_hash,
        )
        or origin_graph_source.source
        != _pin("graph_intent_brief", brief.graph_intent_id, brief.graph_intent_hash)
        or origin_input.build_ref != origin_manifest.build_ref
        or origin_snapshot.build_ref != origin_manifest.build_ref
        or origin_input.graph_intent != graph_payload
        or brief.as_dict() != graph_payload
        or brief.review.state != "accepted"
    ):
        raise ValueError(
            "trial build input origin is not the exact source-first A0 build"
        )
    return (
        origin_manifest_source,
        origin_manifest,
        origin_snapshot_source,
        origin_snapshot,
        children,
    )


def _native_build_ref(build_input: AdvisoryTrialBuildInputsV1) -> str:
    basis = {
        "producer_profile": _PROJECTION_PROFILE,
        "producer_version": _PROJECTION_VERSION,
        "organization_id": build_input.organization_id,
        "workspace_id": build_input.workspace_id,
        "inputs": _pin(
            "advisory_trial_build_inputs",
            build_input.input_ref,
            build_input.input_hash,
        ).model_dump(mode="json"),
    }
    return "ontology-build:advisory-trial:" + content_hash(basis)


def _validate_trial_projection_snapshot_value(
    *,
    snapshot: ExploratoryGraphSnapshotV1,
    build_input: AdvisoryTrialBuildInputsV1,
    predecessor: ExploratoryGraphSnapshotV1,
    origin: ExploratoryGraphSnapshotV1,
    integrity: ExploratoryGraphSnapshotIntegrityV1,
) -> None:
    snapshot_basis = snapshot.model_dump(
        mode="json", exclude={"snapshot_ref", "snapshot_hash"}
    )
    digest = content_hash(snapshot_basis)
    unchanged_fields = (
        "source_identity_ref",
        "source_hash",
        "canonical_source_ir_ref",
        "canonical_source_ir_hash",
        "authority_state",
        "completeness_state",
        "nodes",
        "edges",
        "unresolved_gap_ids",
    )
    if (
        snapshot.change_cause != "controlled_trial_projection"
        or snapshot.triggering_ref != build_input.input_ref
        or snapshot.build_ref != _native_build_ref(build_input)
        or snapshot.snapshot_ref != f"exploratory-snapshot:{digest}"
        or snapshot.snapshot_hash != f"sha256:{digest}"
        or snapshot.created_at != build_input.recorded_at
        or (snapshot.organization_id, snapshot.workspace_id)
        != (build_input.organization_id, build_input.workspace_id)
        or any(
            getattr(snapshot, field) != getattr(origin, field)
            for field in unchanged_fields
        )
        or _payload_pin(predecessor, "snapshot_ref", "snapshot_hash")
        != build_input.predecessor_snapshot
        or integrity.subject
        != _pin(
            "exploratory_graph_snapshot",
            snapshot.snapshot_ref,
            snapshot.snapshot_hash,
        )
        or integrity.producer_profile != _PROJECTION_PROFILE
        or integrity.producer_version != _PROJECTION_VERSION
        or integrity.recorded_at != build_input.recorded_at
    ):
        raise ValueError("trial projection snapshot native relationship mismatch")
    verify_exploratory_snapshot_integrity(snapshot=snapshot, integrity=integrity)


def _validate_context_only_pair_values(
    *,
    origin_manifest: OntologyBuildManifestV1,
    origin_snapshot: ExploratoryGraphSnapshotV1,
    assessment: CanonicalEvidenceAdequacyAssessmentV1,
    brief: GraphIntentBrief,
    competency: CompetencyContractV1,
    compiler_inputs: AdvisoryCompilerInputsV1,
    baseline_candidate: NormativeSemanticCandidateSetV1,
    baseline_derivation: AdvisoryTrialVariantV1,
    baseline_plan: CompiledRulePlanV1,
    successor_candidate: NormativeSemanticCandidateSetV1,
    successor_derivation: AdvisoryTrialVariantV1,
    successor_plan: CompiledRulePlanV1,
    validation: ValidationContractV1,
    producer_module_bytes: bytes,
    baseline_inputs: AdvisoryTrialBuildInputsV1,
    baseline_snapshot: ExploratoryGraphSnapshotV1,
    baseline_integrity: ExploratoryGraphSnapshotIntegrityV1,
    baseline_manifest: OntologyBuildManifestV1,
    successor_inputs: AdvisoryTrialBuildInputsV1,
    successor_snapshot: ExploratoryGraphSnapshotV1,
    successor_integrity: ExploratoryGraphSnapshotIntegrityV1,
    successor_manifest: OntologyBuildManifestV1,
    subject: PairedEvaluationSubjectV1,
) -> None:
    """Validate one complete typed pair without storage or retained wrappers."""

    producer_hash = "sha256:" + sha256(producer_module_bytes).hexdigest()
    origin_manifest_pin = _payload_pin(origin_manifest, "build_ref", "build_hash")
    origin_snapshot_pin = _payload_pin(origin_snapshot, "snapshot_ref", "snapshot_hash")
    validation_pin = _pin(
        "validation_contract",
        validation.validation_contract_ref,
        validation.validation_contract_hash,
    )
    compiler_pin = _pin(
        "advisory_compiler_inputs",
        compiler_inputs.input_ref,
        compiler_inputs.input_hash,
    )
    sides = (
        (
            "BASELINE",
            baseline_inputs,
            baseline_snapshot,
            baseline_integrity,
            baseline_manifest,
            origin_manifest,
            origin_snapshot,
            baseline_candidate,
            baseline_derivation,
            baseline_plan,
        ),
        (
            "SUCCESSOR",
            successor_inputs,
            successor_snapshot,
            successor_integrity,
            successor_manifest,
            baseline_manifest,
            baseline_snapshot,
            successor_candidate,
            successor_derivation,
            successor_plan,
        ),
    )
    for (
        side,
        build_input,
        snapshot,
        integrity,
        manifest,
        predecessor_manifest,
        predecessor_snapshot,
        candidate,
        derivation,
        plan,
    ) in sides:
        if (
            build_input.side != side
            or build_input.usage != "CONTEXT_ONLY"
            or build_input.producer_ref
            != "repo:src/services/runtime_learning_trial_service.py#prepare-context-only-pair.v1"
            or build_input.producer_hash != producer_hash
            or build_input.origin_manifest != origin_manifest_pin
            or build_input.origin_snapshot != origin_snapshot_pin
            or build_input.predecessor_manifest
            != _payload_pin(predecessor_manifest, "build_ref", "build_hash")
            or build_input.predecessor_snapshot
            != _payload_pin(predecessor_snapshot, "snapshot_ref", "snapshot_hash")
            or build_input.candidate
            != _pin(
                "normative_semantic_candidate_set",
                candidate.candidate_set_ref,
                candidate.candidate_set_hash,
            )
            or build_input.derivation
            != _pin(
                "advisory_trial_variant",
                derivation.variant_ref,
                derivation.variant_hash,
            )
            or build_input.plan
            != _pin("compiled_rule_plan", plan.plan_ref, plan.plan_hash)
            or build_input.validation != validation_pin
            or build_input.compiler_inputs != compiler_pin
            or build_input.recorded_at != baseline_inputs.recorded_at
        ):
            raise ValueError("trial pair build input native relationship mismatch")
        _validate_trial_projection_snapshot_value(
            snapshot=snapshot,
            build_input=build_input,
            predecessor=predecessor_snapshot,
            origin=origin_snapshot,
            integrity=integrity,
        )
        expected_children = (
            _pin(
                "canonical_evidence_adequacy_assessment",
                assessment.assessment_ref,
                assessment.assessment_hash,
            ),
            _pin(
                "advisory_trial_build_inputs",
                build_input.input_ref,
                build_input.input_hash,
            ),
            _pin(
                "exploratory_graph_snapshot",
                snapshot.snapshot_ref,
                snapshot.snapshot_hash,
            ),
            _pin("graph_intent_brief", brief.graph_intent_id, brief.graph_intent_hash),
            _pin(
                "competency_contract",
                competency.competency_contract_ref,
                competency.competency_contract_hash,
            ),
            build_input.candidate,
            validation_pin,
            build_input.plan,
        )
        expected_gates = tuple(
            assessment.gate_for(operation) for operation in OntologyOperation
        )
        manifest_basis = {
            "build_ref": manifest.build_ref,
            "organization_id": manifest.organization_id,
            "workspace_id": manifest.workspace_id,
            "assurance_profile": manifest.assurance_profile,
            "child_artifacts": [
                item.model_dump(mode="json") for item in manifest.child_artifacts
            ],
            "gates": [item.model_dump(mode="json") for item in manifest.gates],
            "predecessor_build_ref": manifest.predecessor_build_ref,
            "predecessor_build_hash": manifest.predecessor_build_hash,
        }
        if (
            manifest.assurance_profile != "A0_EXPLORATORY"
            or manifest.build_ref != _native_build_ref(build_input)
            or manifest.child_artifacts != expected_children
            or manifest.gates != expected_gates
            or (manifest.predecessor_build_ref, manifest.predecessor_build_hash)
            != (predecessor_manifest.build_ref, predecessor_manifest.build_hash)
            or manifest.created_at != build_input.recorded_at
            or (manifest.organization_id, manifest.workspace_id)
            != (build_input.organization_id, build_input.workspace_id)
            or snapshot.build_ref != manifest.build_ref
            or manifest.build_hash != "sha256:" + content_hash(manifest_basis)
        ):
            raise ValueError("trial pair manifest native relationship mismatch")

    omission_repair = (
        baseline_derivation.variant == "OMIT_EXCEPTION"
        and successor_derivation.variant == "ADD_MISSING_FRAME"
        and successor_derivation.source == baseline_inputs.candidate
    )
    threshold_pair = (
        baseline_derivation.variant == "PROXY_THRESHOLD_20"
        and successor_derivation.variant == "PROXY_THRESHOLD_10"
        and baseline_derivation.source == successor_derivation.source
    )
    expected_subject = PairedEvaluationSubjectV1(
        baseline_manifest=_payload_pin(baseline_manifest, "build_ref", "build_hash"),
        successor_manifest=_payload_pin(successor_manifest, "build_ref", "build_hash"),
        baseline_snapshot=_payload_pin(
            baseline_snapshot, "snapshot_ref", "snapshot_hash"
        ),
        successor_snapshot=_payload_pin(
            successor_snapshot, "snapshot_ref", "snapshot_hash"
        ),
        alignment_delta=None,
    )
    if not (omission_repair or threshold_pair) or subject != expected_subject:
        raise ValueError("unsupported or mismatched trial pair projection")


def _validate_input_relations(
    build_input: AdvisoryTrialBuildInputsV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> tuple[ArtifactPinV1, ...]:
    (
        origin_manifest_source,
        origin_manifest,
        origin_snapshot_source,
        origin_snapshot,
        origin_children,
    ) = _origin_context(build_input, closure)
    predecessor_manifest_source, predecessor_manifest = _resolve_payload_pin(
        closure,
        build_input.predecessor_manifest,
        kind="ontology_build_manifest",
        contract=OntologyBuildManifestV1,
        label="predecessor manifest",
    )
    predecessor_snapshot_source, predecessor_snapshot = _resolve_payload_pin(
        closure,
        build_input.predecessor_snapshot,
        kind="exploratory_graph_snapshot",
        contract=ExploratoryGraphSnapshotV1,
        label="predecessor snapshot",
    )

    candidate_source = _resolve(closure, build_input.candidate)
    candidate, candidate_context, _ = _candidate_value(candidate_source, closure)
    receipt_source = _resolve(closure, build_input.derivation)
    receipt, _ = _trial_receipt_value(receipt_source, closure)
    _validate_receipt_relationship(receipt, closure)
    plan_source = _resolve(closure, build_input.plan)
    plan, _ = _plan_value(plan_source, closure)
    validation_source = _resolve(closure, build_input.validation)
    validation, validation_context, _ = _validation_value(validation_source, closure)
    compiler_input_source = _resolve(closure, build_input.compiler_inputs)
    compiler_context = _input_value(compiler_input_source, closure)

    origin_graph_source = origin_children["graph_intent_brief"]
    if (
        candidate_source.source != build_input.candidate
        or receipt_source.source != build_input.derivation
        or plan_source.source != build_input.plan
        or validation_source.source != build_input.validation
        or compiler_input_source.source != build_input.compiler_inputs
        or receipt.candidate != build_input.candidate
        or receipt.inputs != build_input.compiler_inputs
        or (plan.candidate_set_ref, plan.candidate_set_hash)
        != (candidate.candidate_set_ref, candidate.candidate_set_hash)
        or (plan.validation_contract_ref, plan.validation_contract_hash)
        != (validation.validation_contract_ref, validation.validation_contract_hash)
        or candidate_context.inputs != compiler_context.inputs
        or validation_context.inputs != compiler_context.inputs
        or candidate_context.competency != compiler_context.competency
    ):
        raise ValueError("trial build input compiler or trial relationship mismatch")

    origin_brief_payload = _payload(origin_graph_source)[1]
    origin_brief = GraphIntentBrief.from_dict(origin_brief_payload)
    compiler_competency = compiler_context.competency
    if (
        (build_input.organization_id, build_input.workspace_id)
        != (origin_manifest.organization_id, origin_manifest.workspace_id)
        or (build_input.organization_id, build_input.workspace_id)
        != (candidate.organization_id, candidate.workspace_id)
        or compiler_competency.graph_intent_brief_ref != origin_brief.graph_intent_id
        or compiler_competency.graph_intent_brief_hash != origin_brief.graph_intent_hash
        or compiler_context.boundary.source_identity_ref
        != origin_snapshot.source_identity_ref
        or compiler_context.boundary.source_hash != origin_snapshot.source_hash
        or compiler_context.boundary.canonical_source_ir_ref
        != origin_snapshot.canonical_source_ir_ref
        or compiler_context.boundary.canonical_source_ir_hash
        != origin_snapshot.canonical_source_ir_hash
    ):
        raise ValueError(
            "trial build input source or GraphIntent relationship mismatch"
        )

    if build_input.side == "BASELINE":
        if (
            predecessor_manifest_source.source != origin_manifest_source.source
            or predecessor_snapshot_source.source != origin_snapshot_source.source
            or receipt.variant not in {"OMIT_EXCEPTION", "PROXY_THRESHOLD_20"}
        ):
            raise ValueError("unsupported baseline trial projection lineage")
    else:
        predecessor_children = _manifest_children(
            predecessor_manifest_source,
            predecessor_manifest,
            closure,
            _PROJECTION_KINDS,
        )
        predecessor_input_source = predecessor_children["advisory_trial_build_inputs"]
        predecessor_input = _build_input_value(predecessor_input_source)
        if predecessor_input.side != "BASELINE":
            raise ValueError("successor predecessor is not its matching baseline")
        predecessor_expected = _validate_input_relations(predecessor_input, closure)
        _expect_dependencies(predecessor_input_source, predecessor_expected, closure)
        predecessor_candidate_source = _resolve(closure, predecessor_input.candidate)
        _predecessor_candidate, _predecessor_context, _ = _candidate_value(
            predecessor_candidate_source, closure
        )
        predecessor_receipt_source = _resolve(closure, predecessor_input.derivation)
        predecessor_receipt, _ = _trial_receipt_value(
            predecessor_receipt_source, closure
        )
        if (
            predecessor_input.origin_manifest != build_input.origin_manifest
            or predecessor_input.origin_snapshot != build_input.origin_snapshot
            or predecessor_manifest.build_ref != predecessor_snapshot.build_ref
            or predecessor_children["exploratory_graph_snapshot"].source
            != predecessor_snapshot_source.source
        ):
            raise ValueError("successor predecessor is not its matching baseline")
        omission_repair = (
            predecessor_receipt.variant == "OMIT_EXCEPTION"
            and receipt.variant == "ADD_MISSING_FRAME"
            and receipt.source == predecessor_input.candidate
        )
        threshold_pair = (
            predecessor_receipt.variant == "PROXY_THRESHOLD_20"
            and receipt.variant == "PROXY_THRESHOLD_10"
            and predecessor_receipt.source == receipt.source
        )
        if not (omission_repair or threshold_pair):
            raise ValueError("unsupported successor trial projection lineage")
        if predecessor_candidate_source.source != predecessor_input.candidate:
            raise ValueError("successor predecessor candidate mismatch")

    expected = {
        _pin_key(pin): pin
        for pin in (
            _as_pin(build_input.origin_manifest, "ontology_build_manifest"),
            _as_pin(build_input.origin_snapshot, "exploratory_graph_snapshot"),
            _as_pin(build_input.predecessor_manifest, "ontology_build_manifest"),
            _as_pin(build_input.predecessor_snapshot, "exploratory_graph_snapshot"),
            build_input.candidate,
            build_input.derivation,
            build_input.plan,
            build_input.validation,
            build_input.compiler_inputs,
        )
    }
    return tuple(expected.values())


def verify_advisory_trial_build_inputs_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify one intrinsic trial-build input and its declared native lineage.

    This source/closure-only check structurally binds ``producer_hash`` but cannot
    authenticate it against a retained package; full package dispatch remains a
    separate prerequisite.
    """

    closure = _validate_closure(closure)
    source = _require_supplied_source(source, closure)
    build_input = _build_input_value(source)
    expected = _validate_input_relations(build_input, closure)
    return _expect_dependencies(source, expected, closure)


def _verify_trial_projection_integrity(
    *,
    source: CapturedSourceV1,
    integrity: ExploratoryGraphSnapshotIntegrityV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> tuple[ArtifactPinV1, ...]:
    if (
        integrity.producer_profile != _PROJECTION_PROFILE
        or integrity.producer_version != _PROJECTION_VERSION
    ):
        raise ValueError("unsupported trial projection integrity producer")
    return _expect_dependencies(source, (), closure)


def _verify_trial_projection_snapshot(
    *,
    source: CapturedSourceV1,
    snapshot: ExploratoryGraphSnapshotV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> tuple[ArtifactPinV1, ...]:
    input_dependencies = tuple(
        pin
        for pin in source.dependencies
        if pin.artifact_kind == "advisory_trial_build_inputs"
    )
    integrity_dependencies = tuple(
        pin
        for pin in source.dependencies
        if pin.artifact_kind == "exploratory_graph_snapshot_integrity"
    )
    if len(input_dependencies) != 1 or len(integrity_dependencies) != 1:
        raise ValueError("trial projection snapshot dependencies are incomplete")
    input_source = _resolve(closure, input_dependencies[0])
    build_input = _build_input_value(input_source)
    predecessor_pin = _pin(
        "exploratory_graph_snapshot",
        snapshot.predecessor_snapshot_ref or "",
        snapshot.predecessor_snapshot_hash or "",
    )
    predecessor_source = _resolve(closure, predecessor_pin)
    predecessor = _model_payload(
        predecessor_source, ExploratoryGraphSnapshotV1, "predecessor snapshot"
    )
    integrity_source = _resolve(closure, integrity_dependencies[0])
    integrity = _model_payload(
        integrity_source,
        ExploratoryGraphSnapshotIntegrityV1,
        "trial projection integrity",
    )
    _origin_manifest_source, _origin_manifest, _origin_source, origin, _children = (
        _origin_context(build_input, closure)
    )
    _validate_trial_projection_snapshot_value(
        snapshot=snapshot,
        build_input=build_input,
        predecessor=predecessor,
        origin=origin,
        integrity=integrity,
    )
    if (
        input_source.source
        != _pin(
            "advisory_trial_build_inputs", build_input.input_ref, build_input.input_hash
        )
        or predecessor_source.source
        != _pin(
            "exploratory_graph_snapshot",
            predecessor.snapshot_ref,
            predecessor.snapshot_hash,
        )
        or integrity_source.source
        != _pin(
            "exploratory_graph_snapshot_integrity",
            integrity.integrity_ref,
            integrity.integrity_hash,
        )
    ):
        raise ValueError("trial projection snapshot native relationship mismatch")
    return _expect_dependencies(
        source,
        (input_source.source, predecessor_source.source, integrity_source.source),
        closure,
    )


def _verify_trial_projection_manifest(
    *,
    source: CapturedSourceV1,
    manifest: OntologyBuildManifestV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> tuple[ArtifactPinV1, ...]:
    children = _manifest_children(source, manifest, closure, _PROJECTION_KINDS)
    assessment = _model_payload(
        children["canonical_evidence_adequacy_assessment"],
        CanonicalEvidenceAdequacyAssessmentV1,
        "trial projection assessment",
    )
    build_input = _build_input_value(children["advisory_trial_build_inputs"])
    snapshot = _model_payload(
        children["exploratory_graph_snapshot"],
        ExploratoryGraphSnapshotV1,
        "trial projection snapshot",
    )
    _graph_source, graph_payload = _payload(children["graph_intent_brief"])
    brief = GraphIntentBrief.from_dict(graph_payload)
    competency = _model_payload(
        children["competency_contract"], CompetencyContractV1, "trial competency"
    )
    candidate, candidate_context, _ = _candidate_value(
        children["normative_semantic_candidate_set"], closure
    )
    validation, validation_context, _ = _validation_value(
        children["validation_contract"], closure
    )
    plan, _ = _plan_value(children["compiled_rule_plan"], closure)
    (
        _origin_source,
        origin_manifest,
        _origin_snapshot_source,
        _origin_snapshot,
        origin_children,
    ) = _origin_context(build_input, closure)
    original_assessment = _model_payload(
        origin_children["canonical_evidence_adequacy_assessment"],
        CanonicalEvidenceAdequacyAssessmentV1,
        "origin assessment",
    )
    predecessor_pin = _pin(
        "ontology_build_manifest",
        manifest.predecessor_build_ref or "",
        manifest.predecessor_build_hash or "",
    )
    predecessor_source = _resolve(closure, predecessor_pin)
    predecessor = _model_payload(
        predecessor_source, OntologyBuildManifestV1, "predecessor manifest"
    )
    expected_gates = tuple(
        assessment.gate_for(operation) for operation in OntologyOperation
    )
    scope = (manifest.organization_id, manifest.workspace_id)
    compiler_input_pin = _pin(
        "advisory_compiler_inputs",
        candidate_context.inputs.input_ref,
        candidate_context.inputs.input_hash,
    )
    if (
        manifest.assurance_profile != "A0_EXPLORATORY"
        or origin_manifest.assurance_profile != "A0_EXPLORATORY"
        or assessment != original_assessment
        or manifest.gates != expected_gates
        or manifest.gates != origin_manifest.gates
        or manifest.build_ref != _native_build_ref(build_input)
        or snapshot.build_ref != manifest.build_ref
        or snapshot.triggering_ref != build_input.input_ref
        or (snapshot.predecessor_snapshot_ref, snapshot.predecessor_snapshot_hash)
        != (
            build_input.predecessor_snapshot.artifact_ref,
            build_input.predecessor_snapshot.artifact_hash,
        )
        or manifest.created_at != build_input.recorded_at
        or snapshot.created_at != build_input.recorded_at
        or _payload_pin(predecessor, "build_ref", "build_hash")
        != build_input.predecessor_manifest
        or scope != (build_input.organization_id, build_input.workspace_id)
        or scope != (snapshot.organization_id, snapshot.workspace_id)
        or scope != (assessment.organization_id, assessment.workspace_id)
        or scope != (candidate.organization_id, candidate.workspace_id)
        or scope != (validation.organization_id, validation.workspace_id)
        or scope != (competency.organization_id, competency.workspace_id)
        or brief.as_dict() != graph_payload
        or brief.graph_intent_id != competency.graph_intent_brief_ref
        or brief.graph_intent_hash != competency.graph_intent_brief_hash
        or candidate_context.inputs != validation_context.inputs
        or compiler_input_pin != build_input.compiler_inputs
        or children["competency_contract"].source != candidate_context.inputs.competency
        or children["canonical_evidence_adequacy_assessment"].source
        != _pin(
            "canonical_evidence_adequacy_assessment",
            assessment.assessment_ref,
            assessment.assessment_hash,
        )
        or children["graph_intent_brief"].source
        != _pin("graph_intent_brief", brief.graph_intent_id, brief.graph_intent_hash)
        or children["advisory_trial_build_inputs"].source
        != _pin(
            "advisory_trial_build_inputs", build_input.input_ref, build_input.input_hash
        )
        or children["exploratory_graph_snapshot"].source
        != _pin(
            "exploratory_graph_snapshot", snapshot.snapshot_ref, snapshot.snapshot_hash
        )
        or children["normative_semantic_candidate_set"].source != build_input.candidate
        or children["validation_contract"].source != build_input.validation
        or children["compiled_rule_plan"].source != build_input.plan
        or (plan.candidate_set_ref, plan.candidate_set_hash)
        != (candidate.candidate_set_ref, candidate.candidate_set_hash)
        or (plan.validation_contract_ref, plan.validation_contract_hash)
        != (validation.validation_contract_ref, validation.validation_contract_hash)
    ):
        raise ValueError("trial projection manifest native relationship mismatch")
    basis = {
        "build_ref": manifest.build_ref,
        "organization_id": manifest.organization_id,
        "workspace_id": manifest.workspace_id,
        "assurance_profile": manifest.assurance_profile,
        "child_artifacts": [
            pin.model_dump(mode="json") for pin in manifest.child_artifacts
        ],
        "gates": [gate.model_dump(mode="json") for gate in manifest.gates],
        "predecessor_build_ref": manifest.predecessor_build_ref,
        "predecessor_build_hash": manifest.predecessor_build_hash,
    }
    if manifest.build_hash != "sha256:" + content_hash(basis):
        raise ValueError("trial projection manifest native hash mismatch")
    return _expect_dependencies(
        source, (*manifest.child_artifacts, predecessor_source.source), closure
    )


__all__ = ["verify_advisory_trial_build_inputs_v1"]
