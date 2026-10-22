"""Source-first exploratory builds over exact governed intent and source pins."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from contracts.canonical_source_ir import CanonicalSourceDocument
from contracts.evidence_location import SourceIdentity
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_for_block
from contracts.ontology_artifact_integrity import (
    create_exploratory_snapshot_integrity,
)
from contracts.ontology_build_inputs import OntologyBuildInputEnvelopeV1
from contracts.graph_intent_brief import GraphIntentBrief
from contracts.ontology_building import (
    ArtifactPinV1, CanonicalEvidenceBoundaryV1, ExploratoryGraphNodeV1, ExploratoryGraphEdgeV1,
    ExploratoryGraphSnapshotV1,
    OntologyBuildManifestV1,
)
from eos.artifact_hashing import content_hash
from services.ontology_build_coordinator import OntologyBuildCoordinator
from services.ontology_evidence_adequacy_service import OntologyEvidenceAdequacyService
from storage.graph_intent_brief_store import GraphIntentBriefStore
from storage.ontology_build_store import OntologyBuildStore

COMPILER_VERSION = "semantier_source_first_exploratory_v3"


class OntologyBuildStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    graph_intent_ref: str = Field(pattern=r"^graph-intent-[0-9a-f]{24}$")
    graph_intent_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    assurance_profile: Literal["A0_EXPLORATORY"] = "A0_EXPLORATORY"


def _pin(kind: str, ref: str, digest: str) -> ArtifactPinV1:
    return ArtifactPinV1(artifact_kind=kind, artifact_ref=ref, artifact_hash=digest)


def _verify_manifest_snapshot_integrity(
    store: OntologyBuildStore, manifest: OntologyBuildManifestV1
) -> None:
    pins = tuple(
        pin
        for pin in manifest.child_artifacts
        if pin.artifact_kind == "exploratory_graph_snapshot"
    )
    if len(pins) != 1:
        raise ValueError("ontology build manifest requires one exploratory snapshot")
    snapshot, _ = store.get_verified_exploratory_snapshot(
        organization_id=manifest.organization_id,
        workspace_id=manifest.workspace_id,
        pin=pins[0],
    )
    if snapshot.build_ref != manifest.build_ref:
        raise ValueError("exploratory snapshot does not bind its build manifest")


def _compile_source(
    canonical: CanonicalSourceDocument, brief: GraphIntentBrief, *, source_identity: SourceIdentity, build_ref: str,
    compiled_at: datetime,
) -> ExploratoryGraphSnapshotV1:
    """Project structure and reviewed intent, never infer normative facts from text."""
    binding = brief.source_binding
    mapped = {block_id for mapping in canonical.source_map.mappings for block_id in mapping.block_ids}
    blocks = sorted(canonical.blocks, key=lambda block: (block.ordinal, block.block_id))
    if not blocks or any(block.block_id not in mapped for block in blocks):
        raise ValueError("Canonical blocks require source-map evidence")
    evidence = {block.block_id: evidence_ref_for_block(
        canonical, block, source_identity=source_identity, created_at=compiled_at,
    ).evidence_ref for block in blocks}
    # Reviewed briefs retain their frozen block locators. Only the new snapshot
    # translates those locators against its exact governed source pins.
    aliases = {f"evidence:canonical-block:{block_id.removeprefix('sha256:')}": ref
               for block_id, ref in evidence.items()}
    aliases.update({ref: ref for ref in evidence.values()})
    nodes = [ExploratoryGraphNodeV1(
        node_id=f"source-block:{block.block_id}", node_kind="canonical_source_block",
        label=str(block.content) if block.content else block.block_type,
        evidence_refs=(evidence[block.block_id],),
    ) for block in blocks]
    gaps = ["gap:certification_not_started", "gap:normative_semantics_unverified"]
    for dimension in brief.material_dimensions:
        refs = tuple(aliases[ref] for ref in dimension.evidence_refs if ref in aliases)
        if not refs:
            gaps.append(f"gap:dimension_evidence:{dimension.type_id}")
            continue
        nodes.append(ExploratoryGraphNodeV1(
            node_id=f"intent-dimension:{dimension.type_id}", node_kind="reviewed_intent_dimension",
            label=dimension.description, evidence_refs=refs,
        ))
    if brief.frame_mapping:
        brief.frame_mapping.validate_source(binding, canonical)
        mapping = brief.frame_mapping.as_dict()
        members = {item['member_id']: item['span'] for item in mapping['members']}
        for frame in mapping['frames']:
            spans = [members[member_id] for member_id in frame['member_ids']]
            # Exploratory modeling decision only; full independent conditions retained.
            refs = tuple(dict.fromkeys(aliases[span['evidence_ref']] for span in [*mapping['shared_context'], *spans]))
            nodes.append(ExploratoryGraphNodeV1(
                node_id=f"intent-frame:{frame['frame_id']}", node_kind="reviewed_frame_mapping",
                label="\n".join(span['quote'] for span in spans), evidence_refs=refs,
            ))
    intent_evidence = tuple(aliases[ref] for ref in brief.evidence_refs if ref in aliases)
    # Questions are review objectives, not asserted source facts. Their backing
    # source is explicit; representability remains an unresolved gap.
    for question in brief.competency_questions:
        gaps.append(f"gap:competency:{question.question_id}")
        if intent_evidence:
            nodes.append(ExploratoryGraphNodeV1(
                node_id=f"intent-competency:{question.question_id}", node_kind="reviewed_competency_question",
                label=question.text, evidence_refs=intent_evidence,
            ))
    gaps.extend(f"gap:intent:{question.question_id}" for question in brief.unresolved_questions)
    source_node_by_evidence = {ref: f"source-block:{block_id}" for block_id, ref in evidence.items()}
    edges = tuple(ExploratoryGraphEdgeV1(
        edge_id=f"intent-evidence:{content_hash({'node': node.node_id, 'evidence': ref})}",
        source_node_id=node.node_id, target_node_id=source_node_by_evidence[ref],
        relation_type="evidenced_by", evidence_refs=(ref,),
    ) for node in nodes if node.node_kind != "canonical_source_block" for ref in node.evidence_refs)
    basis = dict(
        organization_id=binding.organization_id, workspace_id=binding.workspace_id,
        build_ref=build_ref, source_identity_ref=binding.source_identity_ref,
        source_hash=binding.source_hash, canonical_source_ir_ref=binding.canonical_source_ir_ref,
        canonical_source_ir_hash=binding.canonical_source_ir_hash,
        nodes=[node.model_dump(mode="json") for node in nodes],
        edges=[edge.model_dump(mode="json") for edge in edges],
        unresolved_gap_ids=tuple(sorted(set(gaps))), change_cause="initial_exploration",
        triggering_ref=brief.graph_intent_id,
    )
    digest = content_hash(basis)
    snapshot = ExploratoryGraphSnapshotV1(
        snapshot_ref=f"exploratory-snapshot:{digest}", snapshot_hash=f"sha256:{digest}",
        created_at=compiled_at, **basis,
    )
    return snapshot


class OntologyBuildStartService:
    def start(
        self, *, command: OntologyBuildStartRequest, intent_store: GraphIntentBriefStore,
        build_store: OntologyBuildStore,
    ) -> OntologyBuildManifestV1:
        brief = intent_store.get(command.graph_intent_ref, graph_intent_hash=command.graph_intent_hash)
        if not brief.compiler_eligible:
            raise ValueError("GraphIntent must be accepted or edited before compilation")
        binding = brief.source_binding
        identity = intent_store.source_registry.get_source_identity(
            source_identity_ref=binding.source_identity_ref, organization_id=binding.organization_id,
            workspace_id=binding.workspace_id, lineage_id=binding.lineage_id,
        )
        if identity is None:
            raise ValueError("Governed source identity is missing")
        canonical = intent_store.verify_source_binding(binding, source_identity=identity)
        compiler_bytes = Path(__file__).read_bytes()
        compiler_hash = "sha256:" + sha256(compiler_bytes).hexdigest()
        basis = dict(source_binding=binding.as_dict(), graph_intent_ref=brief.graph_intent_id,
                     graph_intent_hash=brief.graph_intent_hash, compiler_version=COMPILER_VERSION,
                     compiler_hash=compiler_hash, assurance_profile=command.assurance_profile)
        build_ref = f"ontology-build:{content_hash(basis)}"
        existing = build_store.get(organization_id=binding.organization_id, workspace_id=binding.workspace_id,
                                   artifact_ref=build_ref, contract_type=OntologyBuildManifestV1)
        if existing is not None:
            _verify_manifest_snapshot_integrity(build_store, existing)
            return existing
        source_path = (intent_store.workspace_path / identity.artifact_ref).resolve()
        workspace_path = intent_store.workspace_path.resolve()
        if workspace_path not in source_path.parents or not source_path.is_file():
            raise ValueError("source bytes must resolve inside the governed workspace")
        inputs = OntologyBuildInputEnvelopeV1.create(
            organization_id=binding.organization_id, workspace_id=binding.workspace_id,
            build_ref=build_ref, graph_intent=brief.as_dict(),
            source_identity=identity.model_dump(mode="json", by_alias=True),
            canonical_source_ir=canonical.model_dump(mode="json", by_alias=True, exclude_none=False),
            source_bytes_base64=base64.b64encode(source_path.read_bytes()).decode("ascii"),
            compiler_version=COMPILER_VERSION,
            compiler_bytes_base64=base64.b64encode(compiler_bytes).decode("ascii"),
        )
        created_at = datetime.now(timezone.utc)
        boundary_basis = dict(
            organization_id=binding.organization_id, workspace_id=binding.workspace_id,
            source_identity_ref=binding.source_identity_ref, source_hash=binding.source_hash,
            canonical_source_ir_ref=binding.canonical_source_ir_ref,
            canonical_source_ir_hash=binding.canonical_source_ir_hash,
            intended_domain=brief.represented_or_governed_subject.description,
            intended_use=brief.document_function.description, target_time=created_at.isoformat(),
            jurisdiction="unresolved", explicit_exclusions=("certification", "admission", "activation"),
        )
        digest = content_hash(boundary_basis)
        boundary = CanonicalEvidenceBoundaryV1(boundary_ref=f"canonical-boundary:{digest}",
            boundary_hash=f"sha256:{digest}", created_at=created_at, **boundary_basis)
        assessment = OntologyEvidenceAdequacyService().assess(
            boundary=boundary, source_authority_role=brief.authority_boundary.source_role,
            available_definition_regions=(), missing_normative_dependencies=(),
            search_evidence_refs=(binding.canonical_source_ir_ref,), assessed_at=created_at,
        )
        snapshot = _compile_source(canonical, brief, source_identity=identity, build_ref=build_ref, compiled_at=created_at)
        integrity = create_exploratory_snapshot_integrity(
            snapshot=snapshot,
            producer_profile="source_first_build.v1",
            producer_version=COMPILER_VERSION,
            recorded_at=created_at,
        )
        pins = (
            _pin("ontology_build_inputs", inputs.envelope_ref, inputs.envelope_hash),
            _pin("canonical_evidence_boundary", boundary.boundary_ref, boundary.boundary_hash),
            _pin("exploratory_graph_snapshot", snapshot.snapshot_ref, snapshot.snapshot_hash),
            _pin("graph_intent_brief", brief.graph_intent_id, brief.graph_intent_hash),
            _pin("source_identity", binding.source_identity_ref, "sha256:" + content_hash(identity.model_dump(mode="json", by_alias=True))),
            _pin("canonical_source_ir", binding.canonical_source_ir_ref, binding.canonical_source_ir_hash),
            _pin("ontology_source_first_compiler", COMPILER_VERSION, compiler_hash),
        )
        manifest = OntologyBuildCoordinator().project_manifest(
            build_ref=build_ref, organization_id=binding.organization_id, workspace_id=binding.workspace_id,
            assurance_profile=command.assurance_profile, assessment=assessment,
            additional_artifacts=pins, created_at=created_at,
        )
        try:
            build_store.put_many_with_integrity(
                artifacts=(inputs, boundary, assessment, snapshot, manifest),
                snapshot_integrities=(integrity,),
            )
        except ValueError:
            # A concurrent identical command may have committed the same build
            # while this command compiled. put_many rolls back its entire set.
            winner = build_store.get(organization_id=binding.organization_id, workspace_id=binding.workspace_id,
                                     artifact_ref=build_ref, contract_type=OntologyBuildManifestV1)
            if winner is None:
                raise
            _verify_manifest_snapshot_integrity(build_store, winner)
            return winner
        return manifest
