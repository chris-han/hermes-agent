"""Service wrapper for graph-schema-discovery MVL contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
import base64
from hashlib import sha256
from pathlib import Path
import json
from typing import Any

from contracts.corpus_graph_schema import (
    V2_SCHEMA_VERSION as CANDIDATE_V2_SCHEMA_VERSION,
    V3_SCHEMA_VERSION as CANDIDATE_V3_SCHEMA_VERSION,
    CorpusGraphSchemaCandidate,
)
from contracts.schema_decision import SchemaDecision
from eos.artifact_hashing import content_hash
from contracts.canonical_source_ir import (
    CanonicalSourceDocument,
    content_hash as canonical_ir_content_hash,
)
from contracts.evidence_location import EvidenceRef, SourceIdentity
from contracts.graph_intent_brief import GraphIntentBrief
from contracts.knowledge_relation_taxonomy import KnowledgeRelationType
from contracts.profile_relation_type_proposal import ProfileRelationTypeProposalV1
from contracts.source_semantic_witness import (
    SourceRoleInterpretationEvidenceV1,
    SourceSemanticWitnessSetManifestV1,
    SourceSemanticWitnessV1,
    WitnessReviewRecord,
)
from contracts.source_structure_semantic_validation import (
    SourceStructureSemanticValidationV1,
)
from semantier.knowledge_feature_packs.canonical_source_ir_adapter import (
    adapt_docx_bytes,
    canonical_elements_from_ir,
)
from semantier.knowledge_feature_packs.evidence_selector_builder import (
    build_evidence_selector,
    evidence_ref_for_block,
)
from semantier.knowledge_feature_packs.source_elements import source_elements
from semantier.semantica_context_graph.native_builder import (
    FIXED_SOURCE,
    GRAPH_ID,
    build_v0_candidate_with_result,
)
from storage.graph_schema_discovery_store import GraphSchemaDiscoveryStore
from storage.corpus_graph_schema_candidate_store import (
    CorpusGraphSchemaCandidateStore,
)
from storage.mvl_selection_store import MvlSelectionNotFound, MvlSelectionStore
from storage.schema_decision_store import SchemaDecisionStore
from storage.graph_intent_brief_store import GraphIntentBriefStore
from storage.knowledge_builder_store import KnowledgeBuilderStore
from storage.human_grounding_store import HumanGroundingStore
from semantier.knowledge_feature_packs.falkordb_backend import InMemoryGraphBackend
from storage.profile_relation_type_proposal_store import (
    ProfileRelationTypeProposalStore,
)
from storage.source_structure_semantic_validation_store import (
    SourceStructureSemanticValidationStore,
)

from .graph_schema_llm_evidence_runner import (
    GraphSchemaLlmEvidenceRunner,
    resolve_active_workspace_path,
)
from .graph_intent_discovery_service import GraphIntentDiscoveryService
from .graph_intent_compiler_policy import graph_intent_compiler_copy
from .corpus_graph_schema_profile_registry import CorpusGraphSchemaProfileRegistry
from .corpus_graph_schema_induction import (
    DEFAULT_PROFILE,
    CorpusGraphSchemaInductionKernel,
)
from .canonical_skill_registry import (
    CANONICAL_JOURNEY_REF,
    COMPILER_CAPABILITY_ID,
    resolve_historical_ref,
    select_for_new_run,
)
from .semantica_mvl_control_adapter import SemanticaMvlControlAdapter
from .semantica_schema_inducer import SemanticaSchemaInducer
from .semantier_schema_synthesizer import SemantierSchemaSynthesizer
from .semantier_schema_synthesizer import UnadmittedGraphIntentRelation
from .knowledge_relation_taxonomy_service import KnowledgeRelationTaxonomyService
from .source_semantic_witness_service import SourceSemanticWitnessService


_TRUSTED_W1_LINEAGE_REGISTRY_HASH = (
    "sha256:e1db88681aac8c048a13aaa7bcf8070c1bbab2691538582adf18bac2546a3d3b"
)


@dataclass(frozen=True)
class AcceptedWitnessGatePackage:
    """Caller input containing only source material and a trusted package pin."""

    document: CanonicalSourceDocument
    source_identity: SourceIdentity
    accepted_package_ref: str
    accepted_package_hash: str
    witness_kind: str


@dataclass(frozen=True)
class _ResolvedAcceptedWitnessGatePackage:
    """Service-resolved semantic authority for one trusted W1 package kind."""

    document: CanonicalSourceDocument
    source_identity: SourceIdentity
    graph_intent: GraphIntentBrief
    source_role_evidence: SourceRoleInterpretationEvidenceV1
    proposed_manifest: SourceSemanticWitnessSetManifestV1
    proposed_witnesses: tuple[SourceSemanticWitnessV1, ...]
    accepted_manifest: SourceSemanticWitnessSetManifestV1
    accepted_witnesses: tuple[SourceSemanticWitnessV1, ...]
    witness_kind: str


_ADJUDICATED_WITNESS_COMPILATION_AUTHORITY = object()
_ADJUDICATED_GRAPH_INTENT_COMPILATION_AUTHORITY = object()
_ADJUDICATED_GRAPH_INTENT_SUCCESSOR_AUTHORITY = object()
_PERSISTED_SUCCESSOR_WITNESS_AUTHORITY = object()
_SOURCE_STRUCTURE_SUCCESSOR_COMPILATION_AUTHORITY = object()


@dataclass(frozen=True, slots=True)
class _W1WitnessLineageRegistry:
    """Immutable repo-owned resolver for accepted W1 semantic authority."""

    repository_root: Path

    def resolve(
        self,
        *,
        accepted_package_ref: str,
        accepted_package_hash: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        _require_owned_lineage_registry_authority(self)
        if (
            _resolve_w1_lineage_registry
            is not _TRUSTED_W1_REGISTRY_IMPLEMENTATION
        ):
            raise ValueError(
                "exemplar admission requires the trusted lineage registry implementation"
            )
        return _TRUSTED_W1_REGISTRY_IMPLEMENTATION(
            repository_root=self.repository_root,
            accepted_package_ref=accepted_package_ref,
            accepted_package_hash=accepted_package_hash,
        )


def _require_owned_lineage_registry_authority(registry: object) -> None:
    if type(registry) is not _W1WitnessLineageRegistry or (
        _W1WitnessLineageRegistry.__dict__.get("resolve")
        is not _TRUSTED_W1_REGISTRY_RESOLVER_DESCRIPTOR
    ):
        raise ValueError(
            "exemplar admission requires the exact trusted lineage registry authority"
        )


def _resolve_w1_lineage_registry(
    *,
    repository_root: Path,
    accepted_package_ref: str,
    accepted_package_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    catalog_path = repository_root / (
        "src/catalogs/source_semantic_witness_lineage_registry.v1.json"
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    declared_catalog_hash = catalog.pop("content_hash", None)
    if (
        catalog.get("schema_version")
        != "source_semantic_witness_lineage_registry.v1"
        or declared_catalog_hash != _TRUSTED_W1_LINEAGE_REGISTRY_HASH
        or declared_catalog_hash != "sha256:" + content_hash(catalog)
    ):
        raise ValueError("trusted W1 witness package registry hash mismatch")
    matches = [
        item
        for item in catalog.get("packages", [])
        if item.get("accepted_package_ref") == accepted_package_ref
        and item.get("accepted_package_hash") == accepted_package_hash
    ]
    if len(matches) != 1:
        raise ValueError("trusted W1 witness package pin is not registered")
    entry = matches[0]
    proposed_path = repository_root / entry["proposed_artifact_path"]
    accepted_path = repository_root / entry["accepted_artifact_path"]
    proposed_bytes = proposed_path.read_bytes()
    proposed = json.loads(proposed_bytes)
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    proposed_declared_hash = proposed.pop("package_content_hash", None)
    accepted_declared_hash = accepted.pop("package_content_hash", None)
    if (
        proposed_declared_hash != entry["proposed_package_hash"]
        or proposed_declared_hash != "sha256:" + content_hash(proposed)
        or "sha256:" + sha256(proposed_bytes).hexdigest()
        != entry["proposed_fixture_bytes_hash"]
        or accepted_declared_hash != accepted_package_hash
        or accepted_declared_hash != "sha256:" + content_hash(accepted)
        or accepted.get("approved_proposed_package_hash")
        != proposed_declared_hash
        or accepted.get("proposed_fixture_bytes_hash")
        != entry["proposed_fixture_bytes_hash"]
    ):
        raise ValueError("trusted W1 witness package artifact hash mismatch")
    return proposed, accepted


_TRUSTED_W1_REGISTRY_IMPLEMENTATION = _resolve_w1_lineage_registry
_TRUSTED_W1_REGISTRY_RESOLVER_DESCRIPTOR = (
    _W1WitnessLineageRegistry.__dict__["resolve"]
)


def _select_terminal_w1_registry_match(
    matches: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Select the unique non-superseded package in one matching W1 lineage."""

    if not matches:
        raise ValueError(
            "GraphIntent must resolve to exactly one trusted accepted W1 package"
        )
    superseded_hashes = {
        predecessor
        for _entry, proposed, accepted in matches
        for predecessor in (
            proposed.get("predecessor_accepted_package_hash"),
            accepted.get("predecessor_accepted_package_hash"),
        )
        if isinstance(predecessor, str) and predecessor
    }
    terminal = [
        item
        for item in matches
        if item[0].get("accepted_package_hash") not in superseded_hashes
    ]
    if len(terminal) != 1:
        raise ValueError(
            "GraphIntent must resolve to exactly one terminal trusted accepted W1 package"
        )
    return terminal[0]


def _resolve_terminal_w1_hard_negative_document(
    terminal: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    matches: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> CanonicalSourceDocument:
    """Resolve a terminal package's negative source, inheriting only conserved bytes."""

    _entry, proposed, accepted = terminal
    embedded = proposed.get("hard_negative_source_document")
    if embedded is not None:
        return CanonicalSourceDocument.model_validate(embedded)

    predecessor_hash = proposed.get("predecessor_accepted_package_hash")
    conservation = proposed.get("conservation")
    if (
        not isinstance(predecessor_hash, str)
        or accepted.get("predecessor_accepted_package_hash") != predecessor_hash
        or not isinstance(conservation, dict)
        or conservation.get("classification") != "REPRESENTATION_ONLY"
        or conservation.get("source_text_changed") != 0
        or conservation.get("semantic_expectations_changed") != 0
        or conservation.get("predecessor_accepted_package_hash") != predecessor_hash
    ):
        raise ValueError("terminal W1 package lacks self-contained negative source authority")

    predecessors = [
        item
        for item in matches
        if item[0].get("accepted_package_hash") == predecessor_hash
    ]
    if len(predecessors) != 1:
        raise ValueError("terminal W1 predecessor package is missing or ambiguous")
    predecessor_proposed = predecessors[0][1]
    if (
        predecessor_proposed.get("hard_negative_source_identity")
        != proposed.get("hard_negative_source_identity")
    ):
        raise ValueError("terminal W1 negative source identity changed across conserved lineage")
    encoded = predecessor_proposed.get("hard_negative_source_bytes_base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("terminal W1 predecessor lacks negative source bytes")
    raw = base64.b64decode(encoded, validate=True)
    identity = SourceIdentity.model_validate(proposed["hard_negative_source_identity"])
    if "sha256:" + sha256(raw).hexdigest() != identity.source_hash:
        raise ValueError("terminal W1 predecessor negative source bytes hash mismatch")
    document = adapt_docx_bytes(raw, source_identity_ref=identity.source_identity_ref)
    binding = GraphIntentBrief.from_dict(proposed["hard_negative_graph_intent"]).source_binding
    if (
        document.source_identity_ref != identity.source_identity_ref
        or canonical_ir_content_hash(
            document.model_dump(mode="json", by_alias=True, exclude_none=False)
        ) != binding.canonical_source_ir_hash
    ):
        raise ValueError("terminal W1 negative source regeneration mismatch")
    return document


def _resolve_trusted_witness_gate_package(
    package: AcceptedWitnessGatePackage,
) -> _ResolvedAcceptedWitnessGatePackage:
    registry = _W1WitnessLineageRegistry(
        repository_root=Path(__file__).resolve().parents[2]
    )
    _require_owned_lineage_registry_authority(registry)
    if _resolve_w1_lineage_registry is not _TRUSTED_W1_REGISTRY_IMPLEMENTATION:
        raise ValueError(
            "exemplar admission requires the trusted lineage registry implementation"
        )
    proposed, accepted = _TRUSTED_W1_REGISTRY_IMPLEMENTATION(
        repository_root=registry.repository_root,
        accepted_package_ref=package.accepted_package_ref,
        accepted_package_hash=package.accepted_package_hash,
    )
    prefix = (
        "positive"
        if package.witness_kind == "positive_exemplar"
        else "hard_negative"
        if package.witness_kind == "hard_negative"
        else None
    )
    if prefix is None:
        raise ValueError("trusted W1 witness package kind is unsupported")
    return _ResolvedAcceptedWitnessGatePackage(
        document=package.document,
        source_identity=package.source_identity,
        graph_intent=GraphIntentBrief.from_dict(
            proposed[f"{prefix}_graph_intent"]
        ),
        source_role_evidence=(
            SourceRoleInterpretationEvidenceV1.model_validate(
                proposed[f"{prefix}_source_role_evidence"]
            )
        ),
        proposed_manifest=SourceSemanticWitnessSetManifestV1.model_validate(
            proposed[f"{prefix}_manifest"]
        ),
        proposed_witnesses=tuple(
            SourceSemanticWitnessV1.model_validate(item)
            for item in proposed[f"{prefix}_witnesses"]
        ),
        accepted_manifest=SourceSemanticWitnessSetManifestV1.model_validate(
            accepted[f"{prefix}_manifest"]
        ),
        accepted_witnesses=tuple(
            SourceSemanticWitnessV1.model_validate(item)
            for item in accepted[f"{prefix}_witnesses"]
        ),
        witness_kind=package.witness_kind,
    )


def resolve_trusted_accepted_witness_by_ref(
    witness_ref: str, *, expected_kind: str
) -> SourceSemanticWitnessV1:
    """Replay one repo-owned accepted W1 witness by its exact stable ref."""

    if expected_kind not in {"positive_exemplar", "hard_negative"}:
        raise ValueError("trusted W1 witness kind is unsupported")
    repository_root = Path(__file__).resolve().parents[2]
    catalog = json.loads((repository_root / (
        "src/catalogs/source_semantic_witness_lineage_registry.v1.json"
    )).read_text(encoding="utf-8"))
    declared_catalog_hash = catalog.pop("content_hash", None)
    if (
        declared_catalog_hash != _TRUSTED_W1_LINEAGE_REGISTRY_HASH
        or declared_catalog_hash != "sha256:" + content_hash(catalog)
    ):
        raise ValueError("trusted W1 witness package registry hash mismatch")
    prefix = "positive" if expected_kind == "positive_exemplar" else "hard_negative"
    matches: list[SourceSemanticWitnessV1] = []
    for entry in catalog.get("packages", []):
        _proposed, accepted = _TRUSTED_W1_REGISTRY_IMPLEMENTATION(
            repository_root=repository_root,
            accepted_package_ref=str(entry.get("accepted_package_ref") or ""),
            accepted_package_hash=str(entry.get("accepted_package_hash") or ""),
        )
        matches.extend(
            witness
            for item in accepted[f"{prefix}_witnesses"]
            for witness in (SourceSemanticWitnessV1.model_validate(item),)
            if witness.witness_id == witness_ref
            and witness.witness_kind == expected_kind
        )
    if len(matches) != 1:
        raise ValueError("trusted accepted W1 witness ref is missing or ambiguous")
    witness = matches[0]
    if witness.review.state not in {"accepted", "edited"}:
        raise ValueError("trusted W1 witness is not accepted/edited")
    return witness


def _resolve_trusted_witness_gate_packages_by_manifest_pins(
    *,
    positive_manifest_ref: str,
    positive_manifest_hash: str,
    hard_negative_manifest_ref: str,
    hard_negative_manifest_hash: str,
    positive_document: CanonicalSourceDocument,
) -> tuple[AcceptedWitnessGatePackage, AcceptedWitnessGatePackage]:
    """Resolve one exact accepted W1 package without caller semantic input."""

    repository_root = Path(__file__).resolve().parents[2]
    catalog_path = repository_root / (
        "src/catalogs/source_semantic_witness_lineage_registry.v1.json"
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    declared_catalog_hash = catalog.pop("content_hash", None)
    if (
        catalog.get("schema_version")
        != "source_semantic_witness_lineage_registry.v1"
        or declared_catalog_hash != _TRUSTED_W1_LINEAGE_REGISTRY_HASH
        or declared_catalog_hash != "sha256:" + content_hash(catalog)
    ):
        raise ValueError("trusted W1 witness package registry hash mismatch")
    matches: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    registry_packages = []
    for entry in catalog.get("packages", []):
        if not isinstance(entry, dict):
            raise ValueError("trusted W1 witness package registry entry is invalid")
        proposed, accepted = _TRUSTED_W1_REGISTRY_IMPLEMENTATION(
            repository_root=repository_root,
            accepted_package_ref=str(entry.get("accepted_package_ref") or ""),
            accepted_package_hash=str(entry.get("accepted_package_hash") or ""),
        )
        registry_packages.append((entry, proposed, accepted))
        positive = SourceSemanticWitnessSetManifestV1.model_validate(
            accepted["positive_manifest"]
        )
        negative = SourceSemanticWitnessSetManifestV1.model_validate(
            accepted["hard_negative_manifest"]
        )
        if (
            positive.manifest_ref == positive_manifest_ref
            and positive.content_hash == positive_manifest_hash
            and negative.manifest_ref == hard_negative_manifest_ref
            and negative.content_hash == hard_negative_manifest_hash
        ):
            matches.append((entry, proposed, accepted))
    if len(matches) != 1:
        raise ValueError(
            "trusted W1 accepted manifest pins must resolve to exactly one package"
        )
    entry, proposed, _accepted = matches[0]

    def package(prefix: str, kind: str) -> AcceptedWitnessGatePackage:
        document = (
            positive_document
            if prefix == "positive"
            else _resolve_terminal_w1_hard_negative_document(
                matches[0], registry_packages,
            )
        )
        identity = SourceIdentity.model_validate(
            proposed[f"{prefix}_source_identity"]
        )
        graph_intent = GraphIntentBrief.from_dict(
            proposed[f"{prefix}_graph_intent"]
        )
        document_source_hashes = {
            locator.source_hash
            for mapping in document.source_map.mappings
            for locator in mapping.source_locators
        }
        portable_positive = (
            prefix == "positive"
            and document_source_hashes == {identity.source_hash}
            and graph_intent.source_binding.source_identity_ref
            == identity.source_identity_ref
            and graph_intent.source_binding.source_hash == identity.source_hash
        )
        if not portable_positive and (
            document.source_identity_ref != identity.source_identity_ref
            or graph_intent.source_binding.source_identity_ref != identity.source_identity_ref
            or graph_intent.source_binding.source_hash != identity.source_hash
        ):
            raise ValueError("trusted W1 package source/GraphIntent ownership mismatch")
        return AcceptedWitnessGatePackage(
            document=document,
            source_identity=identity,
            accepted_package_ref=entry["accepted_package_ref"],
            accepted_package_hash=entry["accepted_package_hash"],
            witness_kind=kind,
        )

    return (
        package("positive", "positive_exemplar"),
        package("hard_negative", "hard_negative"),
    )


_TRUSTED_W1_MANIFEST_RESOLVER_IMPLEMENTATION = (
    _resolve_trusted_witness_gate_packages_by_manifest_pins
)


def _resolve_trusted_witness_gate_packages_for_graph_intent(
    *,
    graph_intent: GraphIntentBrief,
    positive_document: CanonicalSourceDocument,
) -> tuple[AcceptedWitnessGatePackage, AcceptedWitnessGatePackage]:
    """Resolve the unique repo-owned accepted W1 lineage for one GraphIntent."""

    repository_root = Path(__file__).resolve().parents[2]
    catalog = json.loads(
        (
            repository_root
            / "src/catalogs/source_semantic_witness_lineage_registry.v1.json"
        ).read_text(encoding="utf-8")
    )
    declared_hash = catalog.pop("content_hash", None)
    if (
        declared_hash != _TRUSTED_W1_LINEAGE_REGISTRY_HASH
        or declared_hash != "sha256:" + content_hash(catalog)
    ):
        raise ValueError("trusted W1 witness package registry hash mismatch")
    if (
        positive_document.source_identity_ref
        != graph_intent.source_binding.source_identity_ref
        or canonical_ir_content_hash(
            positive_document.model_dump(mode="json", by_alias=True, exclude_none=False)
        )
        != graph_intent.source_binding.canonical_source_ir_hash
    ):
        raise ValueError("runtime GraphIntent source ownership mismatch")
    matches: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for entry in catalog.get("packages", []):
        proposed, accepted = _TRUSTED_W1_REGISTRY_IMPLEMENTATION(
            repository_root=repository_root,
            accepted_package_ref=entry["accepted_package_ref"],
            accepted_package_hash=entry["accepted_package_hash"],
        )
        positive_identity = SourceIdentity.model_validate(
            proposed["positive_source_identity"]
        )
        if positive_identity.source_hash == graph_intent.source_binding.source_hash:
            matches.append((entry, proposed, accepted))
    terminal = _select_terminal_w1_registry_match(matches)
    entry, proposed, _accepted = terminal
    positive_identity = SourceIdentity.model_validate(proposed["positive_source_identity"])
    negative_identity = SourceIdentity.model_validate(
        proposed["hard_negative_source_identity"]
    )
    negative_document = _resolve_terminal_w1_hard_negative_document(terminal, matches)
    return (
        AcceptedWitnessGatePackage(
            document=positive_document,
            source_identity=positive_identity,
            accepted_package_ref=entry["accepted_package_ref"],
            accepted_package_hash=entry["accepted_package_hash"],
            witness_kind="positive_exemplar",
        ),
        AcceptedWitnessGatePackage(
            document=negative_document,
            source_identity=negative_identity,
            accepted_package_ref=entry["accepted_package_ref"],
            accepted_package_hash=entry["accepted_package_hash"],
            witness_kind="hard_negative",
        ),
    )


_TRUSTED_W1_GRAPH_INTENT_RESOLVER_IMPLEMENTATION = (
    _resolve_trusted_witness_gate_packages_for_graph_intent
)


def _require_owned_lineage_authority(service: object) -> None:
    if type(service) is not GraphSchemaDiscoveryService or (
        GraphSchemaDiscoveryService.__dict__.get("_resolve_witness_gate_package")
        is not _TRUSTED_WITNESS_RESOLVER_DESCRIPTOR
    ) or (
        GraphSchemaDiscoveryService.__dict__.get(
            "reconstruct_persisted_material_correction_witnesses"
        )
        is not _TRUSTED_PERSISTED_RECONSTRUCTION_DESCRIPTOR
    ) or (
        GraphSchemaDiscoveryService.__dict__.get(
            "reconstruct_material_correction_witnesses"
        )
        is not _TRUSTED_MATERIAL_RECONSTRUCTION_DESCRIPTOR
    ) or (
        GraphSchemaDiscoveryService.__dict__.get(
            "_governed_material_correction_document"
        )
        is not _TRUSTED_GOVERNED_CORRECTION_DOCUMENT_DESCRIPTOR
    ) or (
        GraphSchemaDiscoveryService.__dict__.get(
            "compile_schema_only_successor"
        )
        is not _TRUSTED_SCHEMA_ONLY_SUCCESSOR_DESCRIPTOR
    ) or (
        GraphSchemaDiscoveryService.__dict__.get(
            "resolve_registered_witness_packages"
        )
        is not _TRUSTED_REGISTERED_WITNESS_PACKAGES_DESCRIPTOR
    ) or (
        GraphSchemaDiscoveryService.__dict__.get(
            "registered_witness_selection"
        )
        is not _TRUSTED_REGISTERED_WITNESS_SELECTION_DESCRIPTOR
    ):
        raise ValueError(
            "exemplar admission requires the exact trusted lineage authority"
        )


class GraphIntentCompatibilityError(ValueError):
    """Typed fail-closed result for exact profile/intent incompatibility."""

    def __init__(self, guard_result: dict[str, Any]) -> None:
        self.guard_result = guard_result
        super().__init__(str(guard_result["reason_code"]))


class GraphSchemaDiscoveryService:
    """Thin orchestration surface for Brief v1 and decision v1 persistence."""

    __slots__ = (
        "_sealed",
        "workspace_root",
        "workspace_id",
        "store",
        "selection_store",
        "schema_decision_store",
        "candidate_store",
        "profile_relation_type_proposal_store",
        "profile_registry",
        "organization_id",
        "graph_intent_store",
    )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError(
                "governed graph schema discovery composition is immutable"
            )
        object.__setattr__(self, name, value)

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        workspace_id: str,
        profile_registry: CorpusGraphSchemaProfileRegistry | None = None,
        organization_id: str | None = None,
        source_registry: KnowledgeBuilderStore | None = None,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_id = workspace_id
        self.store = GraphSchemaDiscoveryStore(
            workspace_root, workspace_id=workspace_id
        )
        self.selection_store = MvlSelectionStore(
            workspace_root, workspace_id=workspace_id
        )
        self.schema_decision_store = SchemaDecisionStore(
            workspace_root, workspace_id=workspace_id
        )
        self.candidate_store = CorpusGraphSchemaCandidateStore(
            workspace_root, workspace_id=workspace_id
        )
        self.profile_relation_type_proposal_store = (
            ProfileRelationTypeProposalStore(
                workspace_root,
                workspace_id=workspace_id,
                organization_id=organization_id,
            )
        )
        self.profile_registry = profile_registry or CorpusGraphSchemaProfileRegistry()
        self.organization_id = organization_id
        self.graph_intent_store = (
            GraphIntentBriefStore(
                workspace_root,
                organization_id=organization_id,
                workspace_id=workspace_id,
                source_registry=source_registry
                or KnowledgeBuilderStore(workspace_root=self.workspace_root),
            )
            if isinstance(organization_id, str) and organization_id.strip()
            else None
        )
        object.__setattr__(self, "_sealed", True)

    def _law_provision_compilation_basis(self, graph_intent, positive_hash, negative_hash):
        from contracts.law_provision_witness import LawProvisionWitnessManifestV1
        from services.law_provision_witness_service import LawProvisionWitnessService
        import re

        _require_owned_lineage_authority(self)
        if self.graph_intent_store is None or (
            graph_intent.source_binding.organization_id != self.organization_id
            or graph_intent.source_binding.workspace_id != self.workspace_id
        ):
            raise ValueError("law provision source ownership mismatch")
        persisted = self.graph_intent_store.get(
            graph_intent.graph_intent_id, graph_intent_hash=graph_intent.graph_intent_hash
        )
        if persisted != graph_intent or not persisted.compiler_eligible:
            raise ValueError("law provision qualification requires exact accepted GraphIntent")
        if not all(re.fullmatch(r"sha256:[0-9a-f]{64}", pin) for pin in (positive_hash, negative_hash)):
            raise ValueError("invalid provision manifest pin")
        path = self.workspace_root / "artifacts/law-provision-witness/v1/manifests" / (positive_hash[7:] + ".json")
        if not path.resolve().is_relative_to(self.workspace_root.resolve()):
            raise ValueError("provision manifest escapes governed workspace")
        manifest = LawProvisionWitnessManifestV1.model_validate_json(path.read_text(encoding="utf-8"))
        if manifest.content_hash != positive_hash:
            raise ValueError("provision manifest hash mismatch")
        profile = CorpusGraphSchemaProfileRegistry.resolve(self.profile_registry, manifest.profile_id)
        if profile.content_hash != manifest.profile_hash:
            raise ValueError("provision profile hash mismatch")
        identity = self.graph_intent_store.source_registry.get_source_identity(
            source_identity_ref=persisted.source_binding.source_identity_ref,
            organization_id=self.organization_id, workspace_id=self.workspace_id,
            lineage_id=persisted.source_binding.lineage_id,
        )
        if identity is None:
            raise ValueError("governed law SourceIdentity missing")
        document = _TRUSTED_GOVERNED_CORRECTION_DOCUMENT_DESCRIPTOR(self, persisted)
        owner = LawProvisionWitnessService(self.workspace_root, source_identity=identity,
            document=document, graph_intent=persisted, profile_id=profile.profile_id,
            profile_hash=profile.content_hash)
        gate = LawProvisionWitnessService.qualify(owner, positive_hash, negative_hash)
        evidence = []
        for entry in manifest.entries:
            witness, _ = LawProvisionWitnessService.replay(owner, entry.witness_hash,
                review_event_ref=entry.review_event_ref)
            span = owner._load("spans", witness.span_hash)
            evidence.extend(m.evidence for m in span.members if m.use == "provision_text")
        selection = {
            "source_semantic_witness_set_ref": gate["positive_manifest_ref"],
            "source_semantic_witness_set_hash": positive_hash,
            "selected_source_semantic_witness_ref": "law-provision-witness:" + manifest.entries[0].witness_hash[7:],
            "selected_source_semantic_witness_hash": manifest.entries[0].witness_hash,
        }
        return gate, tuple(evidence), selection

    def compile_exemplar_backed_relation_decision(
        self,
        *,
        graph_intent: GraphIntentBrief,
        candidate_v3: CorpusGraphSchemaCandidate,
        witness_packages: tuple[
            AcceptedWitnessGatePackage | _ResolvedAcceptedWitnessGatePackage, ...
        ],
        taxonomy_ref: str,
        taxonomy_hash: str,
        decided_by: str,
        actor_ref: str,
        decided_at: str,
        relation_type_proposal: dict[str, Any] | None = None,
        relation_type_proposal_ref: str | None = None,
        relation_type_proposal_hash: str | None = None,
        predecessor_candidate_ref: str | None = None,
        predecessor_candidate_hash: str | None = None,
        predecessor_schema_decision_ref: str | None = None,
        predecessor_schema_decision_hash: str | None = None,
        review_ref: str = "review:w2c:exemplar-gate",
        law_provision_manifest_pins: tuple[str, str] | None = None,
        _adjudicated_authority: object | None = None,
        _allow_neutral_source_successor: bool = False,
        _relation_evidence_refs_override: tuple[EvidenceRef, ...] | None = None,
    ) -> dict[str, Any]:
        """Run accepted W1 exemplars before Candidate v4 and Decision v3."""

        if decided_by not in {"system", "user"}:
            raise ValueError("decided_by must be system or user")
        if _allow_neutral_source_successor and (
            _adjudicated_authority is not _ADJUDICATED_WITNESS_COMPILATION_AUTHORITY
        ):
            raise PermissionError("neutral source witness reuse is sealed")
        if _relation_evidence_refs_override is not None and not (
            _allow_neutral_source_successor
            and _adjudicated_authority
            is _ADJUDICATED_WITNESS_COMPILATION_AUTHORITY
        ):
            raise PermissionError("relation evidence override is sealed")
        _require_owned_lineage_authority(self)
        if law_provision_manifest_pins is not None:
            if witness_packages or _adjudicated_authority is not None or _relation_evidence_refs_override is not None:
                raise ValueError("law provision qualification cannot mix historical witness authority")
            exemplar_gate, relation_evidence_refs, witness_selection = (
                GraphSchemaDiscoveryService._law_provision_compilation_basis(
                    self, graph_intent, *law_provision_manifest_pins
                )
            )
        else:
            if _adjudicated_authority is _ADJUDICATED_WITNESS_COMPILATION_AUTHORITY:
                if not all(
                    type(package) is _ResolvedAcceptedWitnessGatePackage
                    for package in witness_packages
                ):
                    raise ValueError("adjudicated witness compilation shape mismatch")
                resolved_witness_packages = tuple(witness_packages)
            else:
                if _adjudicated_authority is not None:
                    raise PermissionError("adjudicated witness compilation is sealed")
                resolved_witness_packages = tuple(
                    _resolve_trusted_witness_gate_package(package)
                    for package in witness_packages
                )
            exemplar_gate = self._verify_accepted_witness_gate(
                resolved_witness_packages,
                graph_intent=graph_intent,
                allow_adjudicated_successor=(
                    _adjudicated_authority
                    is _ADJUDICATED_WITNESS_COMPILATION_AUTHORITY
                ),
                allow_neutral_source_successor=_allow_neutral_source_successor,
            )
            if (
                graph_intent.source_binding.workspace_id != self.workspace_id
                or (
                    self.organization_id is not None
                    and graph_intent.source_binding.organization_id
                    != self.organization_id
                )
            ):
                raise ValueError(
                    "GraphIntent source ownership mismatches discovery composition"
                )
            positive = next(
                package
                for package in resolved_witness_packages
                if package.witness_kind == "positive_exemplar"
            )
            current_identity = None
            if self.graph_intent_store is not None:
                current_identity = self.graph_intent_store.source_registry.get_source_identity(
                    source_identity_ref=graph_intent.source_binding.source_identity_ref,
                    organization_id=graph_intent.source_binding.organization_id,
                    workspace_id=graph_intent.source_binding.workspace_id,
                    lineage_id=graph_intent.source_binding.lineage_id,
                )
            current_blocks_by_text_hash: dict[str, list[Any]] = {}
            for block in positive.document.blocks:
                text_hash = "sha256:" + sha256(
                    str(block.content).encode("utf-8")
                ).hexdigest()
                current_blocks_by_text_hash.setdefault(text_hash, []).append(block)

            def current_evidence(witness: SourceSemanticWitnessV1) -> EvidenceRef:
                matches = current_blocks_by_text_hash.get(witness.exact_text_hash, [])
                rebind = None
                if not matches:
                    rebind = next((proof for proof in exemplar_gate.get("source_rebindings", [])
                        if proof["accepted_witness_ref"] == witness.witness_id
                        and proof["accepted_witness_hash"] == witness.witness_hash), None)
                    if rebind is not None:
                        matches = [block for block in positive.document.blocks
                            if block.block_id == rebind["current_block_id"]]
                if len(matches) != 1:
                    raise ValueError("runtime source does not reproduce trusted W1 evidence")
                block = matches[0]
                evidence = evidence_ref_for_block(
                    positive.document,
                    block,
                    source_identity=current_identity,
                    evidence_role="reference_example",
                    created_at=current_identity.created_at,
                )
                return evidence.model_copy(
                    update={
                        "provenance": {
                            **evidence.provenance,
                            **({"source_witness_rebinding": rebind} if rebind else {}),
                            "canonical_source_ir_ref": (
                                graph_intent.source_binding.canonical_source_ir_ref
                            ),
                            "canonical_source_ir_hash": (
                                graph_intent.source_binding.canonical_source_ir_hash
                            ),
                        }
                    }
                )
            relation_evidence_refs = (
                _relation_evidence_refs_override
                if _relation_evidence_refs_override is not None
                else tuple(
                    evidence
                    for witness in positive.accepted_witnesses
                    for evidence in witness.evidence_refs
                )
                if current_identity is None
                else tuple(
                    current_evidence(witness)
                    for witness in positive.accepted_witnesses
                )
            )
            witness_selection = {
                "source_semantic_witness_set_ref": positive.accepted_manifest.manifest_ref,
                "source_semantic_witness_set_hash": positive.accepted_manifest.content_hash,
                "selected_source_semantic_witness_ref": positive.accepted_witnesses[0].witness_id,
                "selected_source_semantic_witness_hash": positive.accepted_witnesses[0].witness_hash,
            }
        taxonomy_service = KnowledgeRelationTaxonomyService(
            repository_root=Path(__file__).resolve().parents[2],
            profile_registry=self.profile_registry,
            profile_relation_type_proposal_store=(
                self.profile_relation_type_proposal_store
            ),
        )
        predecessor_pins = (
            predecessor_candidate_ref,
            predecessor_candidate_hash,
            predecessor_schema_decision_ref,
            predecessor_schema_decision_hash,
        )
        if any(pin is None for pin in predecessor_pins) and any(
            pin is not None for pin in predecessor_pins
        ):
            raise ValueError(
                "schema correction predecessor candidate/decision ref/hash pins "
                "must be supplied together"
            )
        predecessor_candidate: CorpusGraphSchemaCandidate | None = None
        predecessor_decision: SchemaDecision | None = None
        if predecessor_candidate_ref is not None:
            predecessor_candidate = CorpusGraphSchemaCandidate.from_dict(
                self.candidate_store.get_by_id(predecessor_candidate_ref)
            )
            if (
                predecessor_candidate.schema_version
                != "corpus_graph_schema_candidate.v4"
                or predecessor_candidate.candidate_hash
                != predecessor_candidate_hash
            ):
                raise ValueError("schema correction predecessor Candidate.v4 mismatch")
            predecessor_decision = KnowledgeRelationTaxonomyService.replay_schema_decision_v3(
                taxonomy_service,
                self.schema_decision_store.get(
                    str(predecessor_schema_decision_ref),
                    taxonomy_service=taxonomy_service,
                ),
            )
            if (
                predecessor_decision.schema_decision_hash
                != predecessor_schema_decision_hash
                or predecessor_decision.candidate_binding.get("candidate_id")
                != predecessor_candidate.candidate_id
                or predecessor_decision.candidate_binding.get("candidate_hash")
                != predecessor_candidate.candidate_hash
                or predecessor_decision.graph_intent_binding
                != predecessor_candidate.graph_intent_binding
                or predecessor_decision.source_binding
                != predecessor_candidate.source_binding
            ):
                raise ValueError("schema correction predecessor Decision.v3 mismatch")
        profile_id = str(candidate_v3.compiler_provenance.get("profile_id") or "")
        profile_hash = str(
            candidate_v3.compiler_provenance.get("profile_hash") or ""
        )
        active_profile = CorpusGraphSchemaProfileRegistry.resolve(
            self.profile_registry, profile_id
        )
        if law_provision_manifest_pins is not None and (
            exemplar_gate["profile_id"] != profile_id or exemplar_gate["profile_hash"] != profile_hash
            or exemplar_gate["taxonomy_ref"] != taxonomy_ref or exemplar_gate["taxonomy_hash"] != taxonomy_hash
        ):
            raise ValueError("law provision qualification/compiler profile or taxonomy mismatch")
        if active_profile.content_hash != profile_hash:
            raise ValueError("Candidate v3 active profile hash mismatch")
        proposal_pins = (
            relation_type_proposal_ref,
            relation_type_proposal_hash,
        )
        if any(pin is None for pin in proposal_pins) and any(
            pin is not None for pin in proposal_pins
        ):
            raise ValueError("profile relation proposal ref/hash must be supplied together")
        if relation_type_proposal is not None and all(
            pin is not None for pin in proposal_pins
        ):
            raise ValueError(
                "new proposal content and persisted proposal lineage are mutually exclusive"
            )
        active_catalog_payload: dict[str, Any] | None = None
        if active_profile.knowledge_relation_catalog_hash is not None:
            active_catalog_payload = (
                CorpusGraphSchemaProfileRegistry.knowledge_relation_catalog(
                    self.profile_registry,
                    profile_id,
                    expected_profile_hash=profile_hash,
                    expected_catalog_hash=(
                        active_profile.knowledge_relation_catalog_hash
                    ),
                )
            )
        profile_relation_admission_binding: tuple[str, str] | None = None
        persisted_proposal: ProfileRelationTypeProposalV1 | None = None
        if relation_type_proposal_ref is not None:
            persisted_proposal = ProfileRelationTypeProposalV1.model_validate(
                self.profile_relation_type_proposal_store.get(
                    proposal_ref=relation_type_proposal_ref,
                    proposal_hash=str(relation_type_proposal_hash),
                    organization_id=graph_intent.source_binding.organization_id,
                )
            )
            primary = graph_intent.primary_relationship
            expected_graph_intent_binding = {
                "graph_intent_ref": graph_intent.graph_intent_id,
                "graph_intent_hash": graph_intent.graph_intent_hash,
                "relationship_ref": primary.type_id,
                "source_type_refs": list(primary.source_types),
                "target_type_refs": list(primary.target_types),
                "direction": primary.direction,
                "semantic_description": primary.semantic_description,
                "preferred_representation": (
                    graph_intent.preferred_representation.kind
                ),
                **(
                    {
                        "source_role_refs": list(primary.source_role_refs),
                        "target_role_refs": list(primary.target_role_refs),
                    }
                    if primary.source_role_refs
                    else {}
                ),
            }
            if (
                persisted_proposal.source_binding
                != graph_intent.source_binding.as_dict()
                or persisted_proposal.graph_intent_binding
                != expected_graph_intent_binding
                or persisted_proposal.evidence_refs
                != tuple(
                    sorted(
                        relation_evidence_refs,
                        key=lambda item: item.evidence_ref.encode("utf-8"),
                    )
                )
            ):
                raise ValueError(
                    "persisted profile relation proposal source binding mismatch"
                )
            predecessor = persisted_proposal.predecessor_binding
            if (
                predecessor["taxonomy_ref"] != taxonomy_ref
                or predecessor["taxonomy_hash"] != taxonomy_hash
            ):
                raise ValueError("persisted proposal taxonomy predecessor mismatch")
            predecessor_profile = CorpusGraphSchemaProfileRegistry.resolve(
                self.profile_registry, predecessor["profile_ref"]
            )
            if predecessor_profile.content_hash != predecessor["profile_hash"]:
                raise ValueError("persisted proposal profile predecessor mismatch")
            CorpusGraphSchemaProfileRegistry.knowledge_relation_catalog(
                self.profile_registry,
                predecessor_profile.profile_id,
                expected_profile_hash=predecessor["profile_hash"],
                expected_catalog_hash=predecessor[
                    "profile_relation_catalog_hash"
                ],
            )
            explicit_relation_type = persisted_proposal.proposed_relation_type
        else:
            explicit_relation_type = (
                KnowledgeRelationType.model_validate(relation_type_proposal)
                if relation_type_proposal is not None
                else None
            )
            primary_type_id = graph_intent.primary_relationship.type_id
            if explicit_relation_type is None and active_catalog_payload is not None:
                intent_catalog = CorpusGraphSchemaProfileRegistry.graph_intent_catalog(
                    self.profile_registry,
                    profile_id,
                    expected_profile_hash=profile_hash,
                    expected_catalog_hash=active_profile.graph_intent_catalog_hash,
                )
                admission_bindings = intent_catalog.get("relation_admission_bindings")
                if admission_bindings is None:
                    admission_bindings = {}
                if not isinstance(admission_bindings, dict) or any(
                    not isinstance(source_id, str)
                    or not isinstance(relation_type_id, str)
                    or not source_id
                    or not relation_type_id
                    for source_id, relation_type_id in admission_bindings.items()
                ):
                    raise ValueError("profile relation admission bindings are malformed")
                admitted_override = None
                if (
                    _adjudicated_authority
                    is _ADJUDICATED_WITNESS_COMPILATION_AUTHORITY
                    and predecessor_decision is not None
                    and len(predecessor_decision.admitted_relation_type_bindings) == 1
                ):
                    admitted_override = predecessor_decision.admitted_relation_type_bindings[
                        0
                    ]["relation_type_id"]
                exact_existing = next(
                    (
                        item
                        for item in active_catalog_payload["relationTypes"]
                        if item["relationTypeId"]
                        == (
                            admitted_override
                            or admission_bindings.get(primary_type_id)
                            or primary_type_id
                        )
                    ),
                    None,
                )
                if exact_existing is not None:
                    explicit_relation_type = KnowledgeRelationType.model_validate(
                        exact_existing
                    )
                    configured_relation_id = admission_bindings.get(primary_type_id)
                    if configured_relation_id == explicit_relation_type.relation_type_id:
                        profile_relation_admission_binding = (
                            primary_type_id,
                            configured_relation_id,
                        )
        if relation_type_proposal is not None:
            if active_catalog_payload is None:
                raise ValueError(
                    "missing predicate proposal requires an active profile relation catalog"
                )
            if (
                explicit_relation_type.definition_scope != "profile_defined"
                or explicit_relation_type.lifecycle_state != "proposed"
            ):
                raise ValueError(
                    "relation_type_proposal must be proposed and profile-defined"
                )
            if explicit_relation_type.namespace != active_catalog_payload["namespace"]:
                raise ValueError("relation_type_proposal namespace mismatches active catalog")
            if any(
                item["relationTypeId"] == explicit_relation_type.relation_type_id
                for item in active_catalog_payload["relationTypes"]
            ):
                raise ValueError(
                    "successor mapping requires exact persisted proposal lineage"
                )
        try:
            candidate_v4 = SemantierSchemaSynthesizer().bind_admitted_relations_v4(
                candidate_v3=candidate_v3,
                graph_intent=graph_intent,
                taxonomy_service=taxonomy_service,
                taxonomy_ref=taxonomy_ref,
                taxonomy_hash=taxonomy_hash,
                relation_evidence_refs=relation_evidence_refs,
                exemplar_gate=exemplar_gate,
                proposal_origin=("system" if decided_by == "system" else "human"),
                explicit_relation_type=explicit_relation_type,
                profile_ref=active_profile.profile_id,
                profile_hash=active_profile.content_hash,
                profile_relation_catalog_hash=(
                    active_profile.knowledge_relation_catalog_hash
                ),
                profile_relation_type_proposal_ref=(
                    persisted_proposal.proposal_ref
                    if persisted_proposal is not None
                    else None
                ),
                profile_relation_type_proposal_hash=(
                    persisted_proposal.proposal_hash
                    if persisted_proposal is not None
                    else None
                ),
                profile_relation_admission_binding=(
                    profile_relation_admission_binding
                ),
            )
        except UnadmittedGraphIntentRelation:
            if persisted_proposal is not None:
                raise ValueError(
                    "persisted proposal is not admitted by the active successor catalog"
                ) from None
            if explicit_relation_type is None:
                raise ValueError(
                    "missing predicate requires an explicit profile relation type proposal"
                ) from None
            primary = graph_intent.primary_relationship
            proposal = ProfileRelationTypeProposalV1.create(
                {
                    "organization_id": graph_intent.source_binding.organization_id,
                    "workspace_id": graph_intent.source_binding.workspace_id,
                    "source_binding": graph_intent.source_binding.as_dict(),
                    "graph_intent_binding": {
                        "graph_intent_ref": graph_intent.graph_intent_id,
                        "graph_intent_hash": graph_intent.graph_intent_hash,
                        "relationship_ref": primary.type_id,
                        "source_type_refs": list(primary.source_types),
                        "target_type_refs": list(primary.target_types),
                        "direction": primary.direction,
                        "semantic_description": primary.semantic_description,
                        "preferred_representation": (
                            graph_intent.preferred_representation.kind
                        ),
                        **(
                            {
                                "source_role_refs": list(primary.source_role_refs),
                                "target_role_refs": list(primary.target_role_refs),
                            }
                            if primary.source_role_refs
                            else {}
                        ),
                    },
                    "predecessor_binding": {
                        "profile_ref": active_profile.profile_id,
                        "profile_hash": active_profile.content_hash,
                        "profile_relation_catalog_hash": (
                            active_profile.knowledge_relation_catalog_hash
                        ),
                        "taxonomy_ref": taxonomy_ref,
                        "taxonomy_hash": taxonomy_hash,
                    },
                    "proposed_relation_type": explicit_relation_type,
                    "proposal_origin": (
                        "system" if decided_by == "system" else "human"
                    ),
                    "actor_ref": actor_ref,
                    "evidence_refs": relation_evidence_refs,
                    "occurred_at": decided_at,
                }
            )
            persisted_proposal = self.profile_relation_type_proposal_store.put(
                proposal,
                organization_id=graph_intent.source_binding.organization_id,
            )
            return {
                "outcome": "versioned_successor_required",
                "material": True,
                "candidate": None,
                "decision": None,
                "relation_type_proposal": persisted_proposal,
                "active_catalog_mutated": False,
            }

        if predecessor_candidate is not None:
            if (
                (
                    candidate_v4.source_binding != predecessor_candidate.source_binding
                    and not _allow_neutral_source_successor
                )
                or candidate_v4.compiler_provenance.get("taxonomy_ref")
                != predecessor_candidate.compiler_provenance.get("taxonomy_ref")
                or candidate_v4.compiler_provenance.get("taxonomy_hash")
                != predecessor_candidate.compiler_provenance.get("taxonomy_hash")
            ):
                raise ValueError("correction changed immutable admission pins")
            if (
                _adjudicated_authority
                is not _ADJUDICATED_WITNESS_COMPILATION_AUTHORITY
                and candidate_v4.graph_intent_binding
                != predecessor_candidate.graph_intent_binding
            ):
                raise ValueError(
                    "schema-only correction changed immutable GraphIntent pins"
                )
            successor_content = candidate_v4.as_dict()
            for field in ("candidate_id", "candidate_hash", "schema_version"):
                successor_content.pop(field)
            successor_content["predecessor_candidate_ref"] = (
                predecessor_candidate.candidate_id
            )
            successor_content["successor_candidate_refs"] = []
            candidate_v4 = CorpusGraphSchemaCandidate.create_v4(successor_content)

        relation_bindings = []
        for relation in candidate_v4.proposed_relation_types:
            candidate_binding = relation["relation_type_candidate"]
            fields = [
                "relation_type_id",
                "taxonomy_ref",
                "taxonomy_hash",
                "profile_ref",
                "profile_hash",
                "profile_relation_catalog_hash",
            ]
            if "profile_relation_type_proposal_ref" in candidate_binding:
                fields.extend(
                    [
                        "profile_relation_type_proposal_ref",
                        "profile_relation_type_proposal_hash",
                    ]
                )
            relation_bindings.append(
                {field: candidate_binding[field] for field in fields}
            )
        decision = taxonomy_service.create_schema_decision_v3(
            {
                "source_binding": dict(candidate_v4.source_binding or {}),
                "candidate_binding": {
                    "candidate_id": candidate_v4.candidate_id,
                    "candidate_hash": candidate_v4.candidate_hash,
                    "candidate_schema_version": candidate_v4.schema_version,
                    "gate_outcome": "accepted",
                    "operating_mode": "assisted",
                    "decided_by": decided_by,
                    "actor_ref": actor_ref,
                    "policy_ref": None,
                    "rationale_ref": None,
                    "decision_evidence_refs": [
                        exemplar_gate["positive_manifest_ref"],
                        exemplar_gate["hard_negative_manifest_ref"],
                    ],
                    "review_ref": review_ref,
                },
                "approved_schema": {
                    "node_types": [dict(item) for item in candidate_v4.proposed_node_types],
                    "relation_types": [
                        dict(item) for item in candidate_v4.proposed_relation_types
                    ],
                    "event_types": [dict(item) for item in candidate_v4.proposed_event_types],
                    "frame_types": [dict(item) for item in candidate_v4.proposed_frame_types],
                },
                "extraction_plan": [dict(item) for item in candidate_v4.extraction_plan],
                "source_evidence_refs": list(candidate_v4.source_evidence_refs),
                "semantica_binding": dict(candidate_v4.semantica_schema_evidence or {}),
                "decided_at": decided_at,
                "predecessor_schema_decision_ref": (
                    predecessor_decision.schema_decision_id
                    if predecessor_decision is not None
                    else None
                ),
                "successor_schema_decision_refs": [],
                "graph_intent_binding": dict(candidate_v4.graph_intent_binding or {}),
                "admitted_relation_type_bindings": relation_bindings,
            }
        )
        self.candidate_store.put(candidate_v4)
        if predecessor_candidate is not None:
            self.candidate_store.add_successor_link(
                predecessor_candidate_ref=predecessor_candidate.candidate_id,
                successor=candidate_v4,
            )
        self.schema_decision_store.put(decision)
        return {
            "outcome": "accepted",
            "exemplar_gate": exemplar_gate,
            "candidate": candidate_v4.as_dict(),
            "decision": decision.as_dict(),
            "witness_selection": witness_selection,
        }

    def compile_schema_only_successor(
        self,
        *,
        predecessor_candidate: CorpusGraphSchemaCandidate,
        predecessor_decision: SchemaDecision,
        candidate_delta: dict[str, Any],
        clarification_ref: str,
        decided_by: str,
        actor_ref: str,
        decided_at: str,
        operating_mode: str,
        policy_ref: str | None,
        rationale_ref: str | None,
    ) -> dict[str, Any]:
        """Compile one Candidate-v4/Decision-v3 successor under the unchanged W1 gate."""

        _require_owned_lineage_authority(self)
        if (
            predecessor_candidate.schema_version
            != "corpus_graph_schema_candidate.v4"
            or predecessor_decision.schema_version != "schema_decision.v3"
            or predecessor_decision.candidate_binding.get("candidate_id")
            != predecessor_candidate.candidate_id
            or predecessor_decision.candidate_binding.get("candidate_hash")
            != predecessor_candidate.candidate_hash
        ):
            raise ValueError("schema-only correction requires exact v4/v3 predecessors")
        if not isinstance(candidate_delta, dict) or not candidate_delta:
            raise ValueError("schema-only correction requires a candidate delta")
        graph_binding = dict(predecessor_candidate.graph_intent_binding or {})
        graph_intent = self.graph_intent_store.get(
            str(graph_binding.get("graph_intent_ref") or ""),
            graph_intent_hash=str(graph_binding.get("graph_intent_hash") or ""),
        )
        prior_gates = [
            item.get("selected_value")
            for item in predecessor_candidate.synthesis_decisions
            if item.get("decision_kind") == "exemplar_reconstruction"
        ]
        if len(prior_gates) != 1 or not isinstance(prior_gates[0], dict):
            raise ValueError("schema-only correction witness lineage is not unchanged")
        prior_gate = prior_gates[0]
        exemplar_gate = self.verify_persisted_accepted_witness_gate(
            graph_intent=graph_intent,
            positive_manifest_ref=str(prior_gate["positive_manifest_ref"]),
            positive_manifest_hash=str(prior_gate["positive_manifest_hash"]),
            hard_negative_manifest_ref=str(prior_gate["hard_negative_manifest_ref"]),
            hard_negative_manifest_hash=str(prior_gate["hard_negative_manifest_hash"]),
        )
        if len(prior_gates) != 1 or prior_gates[0] != exemplar_gate:
            raise ValueError("schema-only correction witness lineage is not unchanged")

        content = predecessor_candidate.as_dict()
        for field in ("schema_version", "candidate_id", "candidate_hash"):
            content.pop(field)
        content.update(candidate_delta)
        content["predecessor_candidate_ref"] = predecessor_candidate.candidate_id
        content["successor_candidate_refs"] = []
        content["clarification_refs"] = list(
            dict.fromkeys([*content.get("clarification_refs", []), clarification_ref])
        )
        successor = CorpusGraphSchemaCandidate.create_v4(content)
        immutable_fields = (
            "source_binding",
            "graph_intent_binding",
            "compiler_provenance",
            "proposed_relation_types",
        )
        if any(
            getattr(successor, field) != getattr(predecessor_candidate, field)
            for field in immutable_fields
        ):
            raise ValueError(
                "schema-only correction changed GraphIntent or taxonomy/profile authority"
            )

        taxonomy_service = KnowledgeRelationTaxonomyService(
            repository_root=Path(__file__).resolve().parents[2],
            profile_registry=self.profile_registry,
            profile_relation_type_proposal_store=(
                self.profile_relation_type_proposal_store
            ),
        )
        predecessor_payload = self.schema_decision_store.get(
            predecessor_decision.schema_decision_id,
            taxonomy_service=taxonomy_service,
        )
        replayed = KnowledgeRelationTaxonomyService.replay_schema_decision_v3(
            taxonomy_service, predecessor_payload
        )
        if replayed.as_dict() != predecessor_decision.as_dict():
            raise ValueError("schema-only predecessor Decision.v3 replay mismatch")
        candidate_binding = dict(predecessor_decision.candidate_binding)
        candidate_binding.update(
            {
                "candidate_id": successor.candidate_id,
                "candidate_hash": successor.candidate_hash,
                "candidate_schema_version": successor.schema_version,
                "decided_by": decided_by,
                "actor_ref": actor_ref,
                "operating_mode": operating_mode,
                "policy_ref": policy_ref,
                "rationale_ref": rationale_ref,
                "decision_evidence_refs": [
                    exemplar_gate["positive_manifest_ref"],
                    exemplar_gate["hard_negative_manifest_ref"],
                    clarification_ref,
                ],
                "review_ref": "review:w2i:schema-only-rerun",
            }
        )
        decision = taxonomy_service.create_schema_decision_v3(
            {
                "source_binding": dict(successor.source_binding or {}),
                "candidate_binding": candidate_binding,
                "approved_schema": {
                    "node_types": [dict(item) for item in successor.proposed_node_types],
                    "relation_types": [
                        dict(item) for item in successor.proposed_relation_types
                    ],
                    "event_types": [dict(item) for item in successor.proposed_event_types],
                    "frame_types": [dict(item) for item in successor.proposed_frame_types],
                },
                "extraction_plan": [dict(item) for item in successor.extraction_plan],
                "source_evidence_refs": list(successor.source_evidence_refs),
                "semantica_binding": dict(successor.semantica_schema_evidence or {}),
                "decided_at": decided_at,
                "predecessor_schema_decision_ref": (
                    predecessor_decision.schema_decision_id
                ),
                "successor_schema_decision_refs": [],
                "graph_intent_binding": dict(successor.graph_intent_binding or {}),
                "admitted_relation_type_bindings": [
                    dict(item)
                    for item in predecessor_decision.admitted_relation_type_bindings
                ],
            }
        )
        self.candidate_store.put(successor)
        self.candidate_store.add_successor_link(
            predecessor_candidate_ref=predecessor_candidate.candidate_id,
            successor=successor,
        )
        self.schema_decision_store.put(decision)
        return {
            "candidate": successor.as_dict(),
            "decision": decision.as_dict(),
            "exemplar_gate": exemplar_gate,
        }

    def compile_adjudicated_graph_intent_successor(
        self,
        *,
        predecessor_candidate: CorpusGraphSchemaCandidate,
        predecessor_decision: SchemaDecision,
        successor_graph_intent: GraphIntentBrief,
        proposed_positive_manifest: SourceSemanticWitnessSetManifestV1,
        proposed_positive_witnesses: tuple[SourceSemanticWitnessV1, ...],
        accepted_positive_manifest: SourceSemanticWitnessSetManifestV1,
        accepted_positive_witnesses: tuple[SourceSemanticWitnessV1, ...],
        proposed_hard_negative_manifest: SourceSemanticWitnessSetManifestV1,
        proposed_hard_negative_witnesses: tuple[SourceSemanticWitnessV1, ...],
        accepted_hard_negative_manifest: SourceSemanticWitnessSetManifestV1,
        accepted_hard_negative_witnesses: tuple[SourceSemanticWitnessV1, ...],
        clarification_ref: str,
        actor_ref: str,
        decided_at: str,
        _adjudication_authority: object | None = None,
        _source_structure_validation: SourceStructureSemanticValidationV1 | None = None,
        _source_structure_authority: object | None = None,
        _source_structure_positive_package: _ResolvedAcceptedWitnessGatePackage | None = None,
    ) -> dict[str, Any]:
        """Compile a GraphIntent successor only after exact 32+6 adjudication."""

        if _adjudication_authority is not _ADJUDICATED_GRAPH_INTENT_SUCCESSOR_AUTHORITY:
            raise PermissionError("adjudicated GraphIntent successor compilation is sealed")
        _require_owned_lineage_authority(self)
        if (
            predecessor_candidate.schema_version
            != "corpus_graph_schema_candidate.v4"
            or predecessor_decision.schema_version != "schema_decision.v3"
            or predecessor_decision.candidate_binding.get("candidate_id")
            != predecessor_candidate.candidate_id
            or predecessor_decision.candidate_binding.get("candidate_hash")
            != predecessor_candidate.candidate_hash
        ):
            raise ValueError(
                "GraphIntent correction requires exact v4/v3 predecessors"
            )
        predecessor_binding = dict(predecessor_candidate.graph_intent_binding or {})
        predecessor_intent = self.graph_intent_store.get(
            str(predecessor_binding.get("graph_intent_ref") or ""),
            graph_intent_hash=str(
                predecessor_binding.get("graph_intent_hash") or ""
            ),
        )
        source_changed = (
            successor_graph_intent.source_binding != predecessor_intent.source_binding
        )
        if _source_structure_validation is None:
            if _source_structure_authority is not None:
                raise PermissionError("source-structure compilation is sealed")
            direct_predecessor = successor_graph_intent.predecessor_graph_intent_ref
        else:
            if (
                _source_structure_authority
                is not _SOURCE_STRUCTURE_SUCCESSOR_COMPILATION_AUTHORITY
            ):
                raise PermissionError("source-structure compilation is sealed")
            proposed_ref = successor_graph_intent.predecessor_graph_intent_ref
            if proposed_ref is None:
                raise ValueError("source-structure GraphIntent proposal is missing")
            if successor_graph_intent.review.state != "accepted":
                raise ValueError("source-structure GraphIntent must be accepted")
            proposed_intent = self.graph_intent_store.get(proposed_ref)
            if (
                proposed_intent.review.state != "proposed"
                or proposed_intent.source_binding != successor_graph_intent.source_binding
            ):
                raise ValueError("source-structure GraphIntent proposal mismatch")
            proposed_payload = proposed_intent.as_dict()
            accepted_payload = successor_graph_intent.as_dict()
            for payload in (proposed_payload, accepted_payload):
                for field in (
                    "graph_intent_id",
                    "graph_intent_hash",
                    "review",
                    "predecessor_graph_intent_ref",
                    "successor_graph_intent_refs",
                ):
                    payload.pop(field, None)
            if proposed_payload != accepted_payload:
                raise ValueError("accepted source-structure GraphIntent changed proposal semantics")
            direct_predecessor = proposed_intent.predecessor_graph_intent_ref
            validation = _source_structure_validation
            if (
                successor_graph_intent.review.actor_ref != validation.created_by
                or validation.organization_id
                != successor_graph_intent.source_binding.organization_id
                or validation.workspace_id
                != successor_graph_intent.source_binding.workspace_id
                or validation.predecessor_graph_intent_ref
                != predecessor_intent.graph_intent_id
                or validation.predecessor_graph_intent_hash
                != predecessor_intent.graph_intent_hash
                or validation.successor_graph_intent_ref
                != proposed_intent.graph_intent_id
                or validation.successor_graph_intent_hash
                != proposed_intent.graph_intent_hash
                or validation.predecessor_source_identity_ref
                != predecessor_intent.source_binding.source_identity_ref
                or validation.successor_source_identity_ref
                != successor_graph_intent.source_binding.source_identity_ref
                or validation.predecessor_canonical_source_ir_ref
                != predecessor_intent.source_binding.canonical_source_ir_ref
                or validation.predecessor_canonical_source_ir_hash
                != predecessor_intent.source_binding.canonical_source_ir_hash
                or validation.successor_canonical_source_ir_ref
                != successor_graph_intent.source_binding.canonical_source_ir_ref
                or validation.successor_canonical_source_ir_hash
                != successor_graph_intent.source_binding.canonical_source_ir_hash
            ):
                raise ValueError("source-structure validation binding mismatch")
        if (
            direct_predecessor != predecessor_intent.graph_intent_id
            or (source_changed and _source_structure_validation is None)
            or (not source_changed and _source_structure_validation is not None)
        ):
            raise ValueError("GraphIntent successor lineage mismatch")

        def vocabulary(intent: GraphIntentBrief) -> tuple[Any, ...]:
            roles = intent.source_role_interpretation.as_dict()
            return (
                intent.document_function.type_id,
                intent.represented_or_governed_subject.type_id,
                intent.primary_relationship.type_id,
                tuple(intent.primary_relationship.source_types),
                tuple(intent.primary_relationship.target_types),
                intent.preferred_representation.kind,
                intent.preferred_representation.type_id,
                tuple(item.type_id for item in intent.material_dimensions),
                tuple((field, tuple(values)) for field, values in roles.items()),
            )

        if vocabulary(successor_graph_intent) != vocabulary(predecessor_intent):
            raise ValueError(
                "GraphIntent correction vocabulary changes require profile/schema admission"
            )

        witness_source_ref = (
            predecessor_intent.source_binding.canonical_source_ir_ref
            if _source_structure_validation is not None
            else successor_graph_intent.source_binding.canonical_source_ir_ref
        )
        canonical_path = self._resolve_canonical_source_ir_ref(witness_source_ref)
        positive_document = CanonicalSourceDocument.model_validate_json(
            canonical_path.read_text(encoding="utf-8")
        )
        prior_gate = next(
            (
                item["selected_value"]
                for item in predecessor_candidate.synthesis_decisions
                if item.get("decision_kind") == "exemplar_reconstruction"
                and isinstance(item.get("selected_value"), dict)
            ),
            None,
        )
        if prior_gate is None:
            raise ValueError("GraphIntent predecessor witness gate is missing")
        try:
            base_packages = _resolve_trusted_witness_gate_packages_by_manifest_pins(
                positive_manifest_ref=str(prior_gate["positive_manifest_ref"]),
                positive_manifest_hash=str(prior_gate["positive_manifest_hash"]),
                hard_negative_manifest_ref=str(
                    prior_gate["hard_negative_manifest_ref"]
                ),
                hard_negative_manifest_hash=str(
                    prior_gate["hard_negative_manifest_hash"]
                ),
                positive_document=positive_document,
            )
            resolved_base = {
                package.witness_kind: _resolve_trusted_witness_gate_package(package)
                for package in base_packages
            }
        except ValueError as exc:
            if "must resolve to exactly one package" not in str(exc):
                raise
            resolved_base = {
                package.witness_kind: package
                for package in self._resolve_persisted_successor_witness_packages(
                    positive_manifest_ref=str(prior_gate["positive_manifest_ref"]),
                    positive_manifest_hash=str(prior_gate["positive_manifest_hash"]),
                    hard_negative_manifest_ref=str(
                        prior_gate["hard_negative_manifest_ref"]
                    ),
                    hard_negative_manifest_hash=str(
                        prior_gate["hard_negative_manifest_hash"]
                    ),
                    positive_document=positive_document,
                )
            }

        def corrected_package(
            kind: str,
            proposed_manifest: SourceSemanticWitnessSetManifestV1,
            proposed_witnesses: tuple[SourceSemanticWitnessV1, ...],
            accepted_manifest: SourceSemanticWitnessSetManifestV1,
            accepted_witnesses: tuple[SourceSemanticWitnessV1, ...],
        ) -> _ResolvedAcceptedWitnessGatePackage:
            base = (
                _source_structure_positive_package
                if kind == "positive_exemplar"
                and _source_structure_validation is not None
                and _source_structure_positive_package is not None
                else resolved_base[kind]
            )
            package_intent = successor_graph_intent
            if (
                base.source_identity.source_identity_ref
                != successor_graph_intent.source_binding.source_identity_ref
            ):
                intent_content = successor_graph_intent.as_dict()
                for field in (
                    "graph_intent_id",
                    "graph_intent_hash",
                    "schema_version",
                ):
                    intent_content.pop(field)
                intent_content["source_binding"] = (
                    base.graph_intent.source_binding.as_dict()
                )
                intent_content["evidence_refs"] = list(base.graph_intent.evidence_refs)
                intent_content["predecessor_graph_intent_ref"] = (
                    base.graph_intent.graph_intent_id
                )
                intent_content["successor_graph_intent_refs"] = []
                package_intent = GraphIntentBrief.create(intent_content)
            return _ResolvedAcceptedWitnessGatePackage(
                document=base.document,
                source_identity=base.source_identity,
                graph_intent=package_intent,
                source_role_evidence=base.source_role_evidence,
                proposed_manifest=proposed_manifest,
                proposed_witnesses=proposed_witnesses,
                accepted_manifest=accepted_manifest,
                accepted_witnesses=accepted_witnesses,
                witness_kind=kind,
            )

        corrected_packages = (
            corrected_package(
                "positive_exemplar",
                proposed_positive_manifest,
                proposed_positive_witnesses,
                accepted_positive_manifest,
                accepted_positive_witnesses,
            ),
            corrected_package(
                "hard_negative",
                proposed_hard_negative_manifest,
                proposed_hard_negative_witnesses,
                accepted_hard_negative_manifest,
                accepted_hard_negative_witnesses,
            ),
        )
        self._verify_accepted_witness_gate(
            corrected_packages,
            graph_intent=successor_graph_intent,
            allow_adjudicated_successor=True,
            allow_neutral_source_successor=_source_structure_validation is not None,
        )
        accepted_claim_refs: list[str] = []
        source_block_evidence: dict[str, dict[str, str]] = {}
        for witness in accepted_positive_witnesses:
            for evidence in witness.evidence_refs:
                evidence_ref = evidence.evidence_ref
                block_id = str(evidence.provenance.get("source_block_id") or "")
                if not block_id:
                    raise ValueError(
                        "adjudicated witness evidence lacks a source block"
                    )
                if evidence_ref not in accepted_claim_refs:
                    accepted_claim_refs.append(evidence_ref)
                source_block_evidence[evidence_ref] = {"block_id": block_id}
        claim_paths = {
            "document_function",
            "represented_or_governed_subject",
            *(
                f"material_dimensions.{index}"
                for index, _item in enumerate(
                    successor_graph_intent.material_dimensions
                )
            ),
            *(
                f"source_role_interpretation.{field}"
                for field, values in (
                    successor_graph_intent.source_role_interpretation.as_dict().items()
                )
                if values
            ),
        }
        adjudicated_evidence = {
            "profile_id": str(predecessor_candidate.compiler_provenance["profile_id"]),
            "profile_hash": str(
                predecessor_candidate.compiler_provenance["profile_hash"]
            ),
            "catalog_hash": str(
                predecessor_candidate.compiler_provenance["catalog_hash"]
            ),
            "claim_evidence": {
                path: list(accepted_claim_refs) for path in sorted(claim_paths)
            },
            "source_block_evidence": source_block_evidence,
            "predecessor_graph_intent_hash": predecessor_intent.graph_intent_hash,
        }
        rebound_relation_evidence: tuple[EvidenceRef, ...] | None = None
        if _source_structure_validation is not None:
            if _source_structure_positive_package is not None:
                rebound_relation_evidence = tuple(
                    evidence
                    for witness in accepted_positive_witnesses
                    for evidence in witness.evidence_refs
                )
            else:
                raise ValueError(
                    "source-structure successor witness package is required"
                )
        proposal = self.induce_graph_intent_schema_candidate(
            graph_intent_ref=successor_graph_intent.graph_intent_id,
            graph_intent_hash=successor_graph_intent.graph_intent_hash,
            canonical_source_ir_ref=(
                successor_graph_intent.source_binding.canonical_source_ir_ref
            ),
            profile_id=str(predecessor_candidate.compiler_provenance["profile_id"]),
            actor_id=actor_ref,
            _adjudicated_evidence=adjudicated_evidence,
            _adjudicated_authority=_ADJUDICATED_GRAPH_INTENT_COMPILATION_AUTHORITY,
        )
        candidate_v3 = CorpusGraphSchemaCandidate.from_dict(proposal["candidate"])
        provenance = predecessor_candidate.compiler_provenance
        return self.compile_exemplar_backed_relation_decision(
            graph_intent=successor_graph_intent,
            candidate_v3=candidate_v3,
            witness_packages=corrected_packages,
            taxonomy_ref=str(provenance["taxonomy_ref"]),
            taxonomy_hash=str(provenance["taxonomy_hash"]),
            decided_by="user",
            actor_ref=actor_ref,
            decided_at=decided_at,
            predecessor_candidate_ref=predecessor_candidate.candidate_id,
            predecessor_candidate_hash=predecessor_candidate.candidate_hash,
            predecessor_schema_decision_ref=predecessor_decision.schema_decision_id,
            predecessor_schema_decision_hash=predecessor_decision.schema_decision_hash,
            review_ref=clarification_ref,
            _adjudicated_authority=_ADJUDICATED_WITNESS_COMPILATION_AUTHORITY,
            _allow_neutral_source_successor=_source_structure_validation is not None,
            _relation_evidence_refs_override=rebound_relation_evidence,
        )

    def compile_source_structure_successor(
        self,
        *,
        predecessor_candidate: CorpusGraphSchemaCandidate,
        predecessor_decision: SchemaDecision,
        successor_graph_intent: GraphIntentBrief,
        semantic_validation_ref: str,
        semantic_validation_hash: str,
        clarification_ref: str,
        actor_ref: str,
        _authority: object | None = None,
    ) -> dict[str, Any]:
        """Recompile a neutral source rebound without fabricating W1 witnesses."""

        if _authority is not _SOURCE_STRUCTURE_SUCCESSOR_COMPILATION_AUTHORITY:
            raise PermissionError("source-structure successor compilation is sealed")
        validation = SourceStructureSemanticValidationStore(
            self.workspace_root, workspace_id=self.workspace_id
        ).get(semantic_validation_ref, validation_hash=semantic_validation_hash)
        if validation.created_by != actor_ref:
            raise PermissionError("source-structure validation belongs to another actor")
        prior_gate = next(
            (
                item["selected_value"]
                for item in predecessor_candidate.synthesis_decisions
                if item.get("decision_kind") == "exemplar_reconstruction"
                and isinstance(item.get("selected_value"), dict)
            ),
            None,
        )
        if prior_gate is None:
            raise ValueError("source-structure predecessor witness gate is missing")
        predecessor_path = self._resolve_canonical_source_ir_ref(
            str(predecessor_candidate.source_binding["canonical_source_ir_ref"])
        )
        predecessor_document = CanonicalSourceDocument.model_validate_json(
            predecessor_path.read_text(encoding="utf-8")
        )
        manifest_pins = {
            "positive_manifest_ref": str(prior_gate["positive_manifest_ref"]),
            "positive_manifest_hash": str(prior_gate["positive_manifest_hash"]),
            "hard_negative_manifest_ref": str(
                prior_gate["hard_negative_manifest_ref"]
            ),
            "hard_negative_manifest_hash": str(
                prior_gate["hard_negative_manifest_hash"]
            ),
        }
        try:
            packages = _resolve_trusted_witness_gate_packages_by_manifest_pins(
                **manifest_pins,
                positive_document=predecessor_document,
            )
            resolved_packages = tuple(
                _resolve_trusted_witness_gate_package(package)
                for package in packages
            )
        except ValueError as exc:
            if "must resolve to exactly one package" not in str(exc):
                raise
            resolved_packages = self._resolve_persisted_successor_witness_packages(
                **manifest_pins,
                positive_document=predecessor_document,
            )
        resolved = {package.witness_kind: package for package in resolved_packages}
        positive_base = resolved["positive_exemplar"]
        successor_document = self._governed_material_correction_document(
            successor_graph_intent
        )
        successor_identity = self.graph_intent_store.source_registry.get_source_identity(
            source_identity_ref=successor_graph_intent.source_binding.source_identity_ref,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            lineage_id=successor_graph_intent.source_binding.lineage_id,
        )
        if successor_identity is None:
            raise ValueError("source-structure successor SourceIdentity is missing")
        new_blocks_by_content: dict[str, list[Any]] = {}
        for block in successor_document.blocks:
            new_blocks_by_content.setdefault(block.content_hash, []).append(block)

        def rebind_evidence(evidence: EvidenceRef) -> EvidenceRef:
            content_hashes = [
                item.split(":", 1)[1]
                for item in evidence.selector.structural_path
                if item.startswith("localContentHash:")
            ]
            if len(content_hashes) != 1:
                raise ValueError(
                    "source-structure evidence lacks exact content identity"
                )
            matches = new_blocks_by_content.get(content_hashes[0], [])
            if len(matches) != 1 and evidence.selector.table is not None:
                matches = [
                    block
                    for block in matches
                    if (
                        (selector := build_evidence_selector(successor_document, block)).table
                        is not None
                        and selector.table.table_ordinal
                        == evidence.selector.table.table_ordinal
                        and selector.table.row_key == evidence.selector.table.row_key
                        and selector.table.column_key
                        == evidence.selector.table.column_key
                    )
                ]
            if len(matches) != 1:
                raise ValueError(
                    "source-structure witness content mapping must be unique"
                )
            base = evidence_ref_for_block(
                successor_document,
                matches[0],
                source_identity=successor_identity,
                evidence_role=evidence.evidence_role,
                created_at=successor_identity.created_at,
            )
            return EvidenceRef.model_validate(
                base.model_copy(
                    update={
                        "provenance": {
                            **base.provenance,
                            "canonical_source_ir_ref": (
                                successor_graph_intent.source_binding.canonical_source_ir_ref
                            ),
                            "canonical_source_ir_hash": (
                                successor_graph_intent.source_binding.canonical_source_ir_hash
                            ),
                        }
                    }
                ).model_dump(mode="json", by_alias=True)
            )

        grounding = HumanGroundingStore(
            self.workspace_root / "eos.db", graph_backend=InMemoryGraphBackend()
        )
        rebound_source_role_evidence = SourceRoleInterpretationEvidenceV1.create(
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            source_identity_ref=successor_identity.source_identity_ref,
            source_hash=successor_identity.source_hash,
            canonical_source_ir_ref=(
                successor_graph_intent.source_binding.canonical_source_ir_ref
            ),
            canonical_source_ir_hash=(
                successor_graph_intent.source_binding.canonical_source_ir_hash
            ),
            field_bindings=tuple(
                binding.model_copy(
                    update={
                        "header_evidence_ref": rebind_evidence(
                            binding.header_evidence_ref
                        )
                    }
                )
                for binding in positive_base.source_role_evidence.field_bindings
            ),
        )
        discovered_rebound = SourceSemanticWitnessService().discover(
            document=successor_document,
            source_identity=successor_identity,
            graph_intent=successor_graph_intent,
            source_role_evidence=rebound_source_role_evidence,
            actor_ref=actor_ref,
            review_set_ref=(
                f"review-set:source-structure-discovery:{validation.validation_ref}"
            ),
            witness_kind="positive_exemplar",
        )
        predecessor_by_text: dict[str, list[SourceSemanticWitnessV1]] = {}
        for witness in positive_base.accepted_witnesses:
            predecessor_by_text.setdefault(witness.exact_text_hash, []).append(witness)
        rebound_proposed_witnesses = []
        for index, discovered in enumerate(discovered_rebound):
            predecessors = predecessor_by_text.get(discovered.exact_text_hash, [])
            if len(predecessors) != 1:
                raise ValueError(
                    "source-structure witness semantic mapping must be unique"
                )
            predecessor_witness = predecessors[0]
            rebound_proposed_witnesses.append(
                SourceSemanticWitnessV1.create(
                    witness_id=discovered.witness_id,
                    organization_id=self.organization_id,
                    workspace_id=self.workspace_id,
                    source_binding=discovered.source_binding,
                    witness_kind=discovered.witness_kind,
                    discovered_source_role=discovered.discovered_source_role,
                    evidence_refs=discovered.evidence_refs,
                    source_anchor_refs=discovered.source_anchor_refs,
                    exact_text_hash=discovered.exact_text_hash,
                    semantic_expectation=predecessor_witness.semantic_expectation,
                    review=WitnessReviewRecord.create(
                        state="proposed",
                        actor_ref=actor_ref,
                        review_ref_seed=(
                            f"review:source-structure-discovered:"
                            f"{validation.validation_ref}:{index}"
                        ),
                        decision_evidence_refs=discovered.evidence_refs,
                        predecessor_witness_ref=predecessor_witness.witness_id,
                    ),
                )
            )
        rebound_proposed_manifest = SourceSemanticWitnessSetManifestV1.create(
            witnesses=rebound_proposed_witnesses,
            review_set_ref=(
                f"review-set:source-structure-proposed:{validation.validation_ref}"
            ),
            predecessor_manifest_ref=positive_base.accepted_manifest.manifest_ref,
        )
        rebound_accepted_witnesses = tuple(
            witness.create_successor(
                semantic_expectation=witness.semantic_expectation,
                review=WitnessReviewRecord.create(
                    state="accepted",
                    actor_ref=actor_ref,
                    review_ref_seed=(
                        f"review:source-structure-accepted:"
                        f"{validation.validation_ref}:{index}"
                    ),
                    decision_evidence_refs=witness.evidence_refs,
                    predecessor_witness_ref=witness.witness_id,
                ),
            )
            for index, witness in enumerate(rebound_proposed_witnesses)
        )
        rebound_accepted_manifest = SourceSemanticWitnessSetManifestV1.create(
            witnesses=rebound_accepted_witnesses,
            review_set_ref=(
                f"review-set:source-structure-accepted:{validation.validation_ref}"
            ),
            predecessor_manifest_ref=rebound_proposed_manifest.manifest_ref,
        )
        for manifest, witnesses in (
            (rebound_proposed_manifest, tuple(rebound_proposed_witnesses)),
            (rebound_accepted_manifest, rebound_accepted_witnesses),
        ):
            grounding.persist_source_semantic_witness_package(
                manifest=manifest,
                witnesses=witnesses,
                artifact_ref=(
                    f"artifacts/source-semantic-witness-set/{manifest.content_hash[7:]}/"
                    "source_semantic_witness_set_manifest.v1.json"
                ),
            )
        resolved["positive_exemplar"] = replace(
            positive_base,
            document=successor_document,
            source_identity=successor_identity,
            graph_intent=successor_graph_intent,
            source_role_evidence=rebound_source_role_evidence,
            proposed_manifest=rebound_proposed_manifest,
            proposed_witnesses=tuple(rebound_proposed_witnesses),
            accepted_manifest=rebound_accepted_manifest,
            accepted_witnesses=rebound_accepted_witnesses,
        )
        positive = resolved["positive_exemplar"]
        negative = resolved["hard_negative"]
        return self.compile_adjudicated_graph_intent_successor(
            predecessor_candidate=predecessor_candidate,
            predecessor_decision=predecessor_decision,
            successor_graph_intent=successor_graph_intent,
            proposed_positive_manifest=positive.proposed_manifest,
            proposed_positive_witnesses=positive.proposed_witnesses,
            accepted_positive_manifest=positive.accepted_manifest,
            accepted_positive_witnesses=positive.accepted_witnesses,
            proposed_hard_negative_manifest=negative.proposed_manifest,
            proposed_hard_negative_witnesses=negative.proposed_witnesses,
            accepted_hard_negative_manifest=negative.accepted_manifest,
            accepted_hard_negative_witnesses=negative.accepted_witnesses,
            clarification_ref=clarification_ref,
            actor_ref=actor_ref,
            decided_at=validation.created_at,
            _adjudication_authority=_ADJUDICATED_GRAPH_INTENT_SUCCESSOR_AUTHORITY,
            _source_structure_validation=validation,
            _source_structure_authority=_SOURCE_STRUCTURE_SUCCESSOR_COMPILATION_AUTHORITY,
            _source_structure_positive_package=positive,
        )

    def reconstruct_material_correction_witnesses(
        self,
        *,
        predecessor_graph_intent: GraphIntentBrief,
        successor_graph_intent: GraphIntentBrief,
        witness_packages: tuple[
            AcceptedWitnessGatePackage | _ResolvedAcceptedWitnessGatePackage, ...
        ],
        actor_ref: str,
        _persisted_successor_authority: object | None = None,
    ) -> dict[str, Any]:
        """Reconstruct proposed witnesses without reusing predecessor acceptance."""

        _require_owned_lineage_authority(self)
        if successor_graph_intent.predecessor_graph_intent_ref != (
            predecessor_graph_intent.graph_intent_id
        ):
            raise ValueError("material correction GraphIntent predecessor mismatch")
        if successor_graph_intent.source_binding != predecessor_graph_intent.source_binding:
            raise ValueError("material correction GraphIntent source pins mismatch")
        if successor_graph_intent.graph_intent_hash == predecessor_graph_intent.graph_intent_hash:
            raise ValueError("material correction requires a distinct GraphIntent successor")
        if _persisted_successor_authority is _PERSISTED_SUCCESSOR_WITNESS_AUTHORITY:
            if not all(
                type(package) is _ResolvedAcceptedWitnessGatePackage
                for package in witness_packages
            ):
                raise ValueError("persisted successor witness package shape mismatch")
            resolved = tuple(witness_packages)
        else:
            if _persisted_successor_authority is not None:
                raise PermissionError("persisted successor witness replay is sealed")
            resolved = tuple(
                _resolve_trusted_witness_gate_package(package)
                for package in witness_packages
            )
        before_gate = self._verify_accepted_witness_gate(
            resolved,
            graph_intent=predecessor_graph_intent,
            allow_adjudicated_successor=(
                _persisted_successor_authority
                is _PERSISTED_SUCCESSOR_WITNESS_AUTHORITY
            ),
        )
        proposed_by_kind: dict[str, list[dict[str, Any]]] = {}
        proposed_manifests: dict[str, dict[str, Any]] = {}
        mismatches: list[dict[str, str]] = []
        for package in resolved:
            comparison_intent = successor_graph_intent
            if package.source_identity.source_identity_ref != (
                successor_graph_intent.source_binding.source_identity_ref
            ):
                comparison_content = successor_graph_intent.as_dict()
                for field in ("graph_intent_id", "graph_intent_hash", "schema_version"):
                    comparison_content.pop(field)
                comparison_content["source_binding"] = (
                    package.graph_intent.source_binding.as_dict()
                )
                comparison_content["evidence_refs"] = list(
                    package.graph_intent.evidence_refs
                )
                comparison_content["predecessor_graph_intent_ref"] = (
                    package.graph_intent.graph_intent_id
                )
                comparison_content["successor_graph_intent_refs"] = []
                comparison_intent = GraphIntentBrief.create(comparison_content)
            discovered = SourceSemanticWitnessService().discover(
                document=package.document,
                source_identity=package.source_identity,
                graph_intent=comparison_intent,
                source_role_evidence=package.source_role_evidence,
                actor_ref=actor_ref,
                review_set_ref=(
                    "material-correction:"
                    + successor_graph_intent.graph_intent_hash.removeprefix("sha256:")[:24]
                ),
                witness_kind=package.witness_kind,
                evidence_provenance_by_block_id={
                    str(item.evidence_refs[0].provenance["source_block_id"]): {
                        key: value
                        for key, value in item.evidence_refs[0].provenance.items()
                        if key
                        in {
                            "construction_kind",
                            "authority_promotion_allowed",
                            "lexical_overlap_positive_family_ref",
                        }
                    }
                    for item in package.proposed_witnesses
                },
            )
            accepted_by_block = {
                str(item.evidence_refs[0].provenance["source_block_id"]): item
                for item in package.accepted_witnesses
            }
            proposed_successors: list[SourceSemanticWitnessV1] = []
            for proposed in discovered:
                block_id = str(
                    proposed.evidence_refs[0].provenance["source_block_id"]
                )
                accepted = accepted_by_block.get(block_id)
                if accepted is None:
                    raise ValueError(
                        "material correction reconstruction produced an unknown block"
                    )
                if (
                    proposed.source_binding != accepted.source_binding
                    or proposed.evidence_refs != accepted.evidence_refs
                    or proposed.exact_text_hash != accepted.exact_text_hash
                ):
                    raise ValueError(
                        "material correction reconstruction source evidence mismatch"
                    )
                if proposed.semantic_expectation != accepted.semantic_expectation:
                    mismatches.append(
                        {
                            "witness_ref": accepted.witness_id,
                            "affected_semantics": "semantic_expectation",
                        }
                    )
                proposed_successors.append(
                    accepted.create_successor(
                        semantic_expectation=proposed.semantic_expectation,
                        review=WitnessReviewRecord.create(
                            state="proposed",
                            actor_ref=actor_ref,
                            review_ref_seed=(
                                "review:material-correction:"
                                + successor_graph_intent.graph_intent_hash.removeprefix(
                                    "sha256:"
                                )[:16]
                                + ":"
                                + block_id.removeprefix("block:")[:16]
                            ),
                            decision_evidence_refs=proposed.evidence_refs,
                            predecessor_witness_ref=accepted.witness_id,
                        ),
                    )
                )
            if len(discovered) != len(package.accepted_witnesses):
                raise ValueError("material correction reconstruction count mismatch")
            review_set_ref = (
                "review-set:material-correction:"
                + successor_graph_intent.graph_intent_hash.removeprefix("sha256:")[:20]
                + ":"
                + package.witness_kind
            )
            manifest = SourceSemanticWitnessSetManifestV1.create(
                witnesses=proposed_successors,
                review_set_ref=review_set_ref,
                predecessor_manifest_ref=package.accepted_manifest.manifest_ref,
            )
            proposed_by_kind[package.witness_kind] = [
                item.model_dump(mode="json", by_alias=True)
                for item in proposed_successors
            ]
            proposed_manifests[package.witness_kind] = manifest.model_dump(
                mode="json", by_alias=True
            )
        positive = proposed_by_kind.get("positive_exemplar", [])
        hard_negative = proposed_by_kind.get("hard_negative", [])
        return {
            "outcome": "pending_adjudication",
            "reason": "graph_intent_successor_requires_witness_adjudication",
            "before_gate": before_gate,
            "positive_count": len(positive),
            "hard_negative_count": len(hard_negative),
            "proposed_positive_witnesses": positive,
            "proposed_hard_negative_witnesses": hard_negative,
            "proposed_positive_manifest": proposed_manifests["positive_exemplar"],
            "proposed_hard_negative_manifest": proposed_manifests["hard_negative"],
            "material_mismatches": mismatches,
            "candidate": None,
            "decision": None,
        }

    def reconstruct_persisted_material_correction_witnesses(
        self,
        *,
        predecessor_graph_intent: GraphIntentBrief,
        successor_graph_intent: GraphIntentBrief,
        positive_manifest_ref: str,
        positive_manifest_hash: str,
        hard_negative_manifest_ref: str,
        hard_negative_manifest_hash: str,
        actor_ref: str,
    ) -> dict[str, Any]:
        """Rebuild a pending 32+6 set from the exact production W1 lineage."""

        _require_owned_lineage_authority(self)
        if (
            _resolve_trusted_witness_gate_packages_by_manifest_pins
            is not _TRUSTED_W1_MANIFEST_RESOLVER_IMPLEMENTATION
        ):
            raise ValueError(
                "material correction requires the trusted W1 manifest resolver"
            )
        document = _TRUSTED_GOVERNED_CORRECTION_DOCUMENT_DESCRIPTOR(
            self, successor_graph_intent
        )
        persisted_successor = False
        try:
            packages = _TRUSTED_W1_MANIFEST_RESOLVER_IMPLEMENTATION(
                positive_manifest_ref=positive_manifest_ref,
                positive_manifest_hash=positive_manifest_hash,
                hard_negative_manifest_ref=hard_negative_manifest_ref,
                hard_negative_manifest_hash=hard_negative_manifest_hash,
                positive_document=document,
            )
        except ValueError as exc:
            if "must resolve to exactly one package" not in str(exc):
                raise
            packages = self._resolve_persisted_successor_witness_packages(
                positive_manifest_ref=positive_manifest_ref,
                positive_manifest_hash=positive_manifest_hash,
                hard_negative_manifest_ref=hard_negative_manifest_ref,
                hard_negative_manifest_hash=hard_negative_manifest_hash,
                positive_document=document,
            )
            packages = tuple(
                replace(package, graph_intent=predecessor_graph_intent)
                if package.witness_kind == "positive_exemplar"
                else package
                for package in packages
            )
            persisted_successor = True
        return _TRUSTED_MATERIAL_RECONSTRUCTION_DESCRIPTOR(
            self,
            predecessor_graph_intent=predecessor_graph_intent,
            successor_graph_intent=successor_graph_intent,
            witness_packages=packages,
            actor_ref=actor_ref,
            _persisted_successor_authority=(
                _PERSISTED_SUCCESSOR_WITNESS_AUTHORITY
                if persisted_successor
                else None
            ),
        )

    def verify_persisted_accepted_witness_gate(
        self,
        *,
        graph_intent: GraphIntentBrief,
        positive_manifest_ref: str,
        positive_manifest_hash: str,
        hard_negative_manifest_ref: str,
        hard_negative_manifest_hash: str,
    ) -> dict[str, Any]:
        """Re-resolve and reconstruct the exact accepted witness authority."""

        _require_owned_lineage_authority(self)
        if positive_manifest_ref.startswith("law-provision-manifest:"):
            gate, _, _ = GraphSchemaDiscoveryService._law_provision_compilation_basis(
                self, graph_intent, positive_manifest_hash, hard_negative_manifest_hash
            )
            if (gate["positive_manifest_ref"] != positive_manifest_ref
                or gate["hard_negative_manifest_ref"] != hard_negative_manifest_ref):
                raise ValueError("law provision manifest ref/hash mismatch")
            return gate

        if (
            _resolve_trusted_witness_gate_packages_by_manifest_pins
            is not _TRUSTED_W1_MANIFEST_RESOLVER_IMPLEMENTATION
        ):
            raise ValueError("trusted W1 manifest resolver was replaced")
        document = _TRUSTED_GOVERNED_CORRECTION_DOCUMENT_DESCRIPTOR(
            self, graph_intent
        )
        adjudicated_successor = False
        try:
            packages = _TRUSTED_W1_MANIFEST_RESOLVER_IMPLEMENTATION(
                positive_manifest_ref=positive_manifest_ref,
                positive_manifest_hash=positive_manifest_hash,
                hard_negative_manifest_ref=hard_negative_manifest_ref,
                hard_negative_manifest_hash=hard_negative_manifest_hash,
                positive_document=document,
            )
        except ValueError as exc:
            if "must resolve to exactly one package" not in str(exc):
                raise
            packages = self._resolve_persisted_successor_witness_packages(
                positive_manifest_ref=positive_manifest_ref,
                positive_manifest_hash=positive_manifest_hash,
                hard_negative_manifest_ref=hard_negative_manifest_ref,
                hard_negative_manifest_hash=hard_negative_manifest_hash,
                positive_document=document,
            )
            packages = tuple(
                replace(package, graph_intent=graph_intent)
                if package.witness_kind == "positive_exemplar"
                else package
                for package in packages
            )
            adjudicated_successor = True
        resolved = (
            packages
            if adjudicated_successor
            else tuple(
                _TRUSTED_WITNESS_RESOLVER_DESCRIPTOR(package)
                for package in packages
            )
        )
        return self._verify_accepted_witness_gate(
            resolved,
            graph_intent=graph_intent,
            allow_adjudicated_successor=adjudicated_successor,
        )

    def _resolve_persisted_successor_witness_packages(
        self,
        *,
        positive_manifest_ref: str,
        positive_manifest_hash: str,
        hard_negative_manifest_ref: str,
        hard_negative_manifest_hash: str,
        positive_document: CanonicalSourceDocument,
    ) -> tuple[_ResolvedAcceptedWitnessGatePackage, ...]:
        if self.organization_id is None:
            raise ValueError("persisted successor witness replay requires organization")
        store = HumanGroundingStore(
            self.workspace_root / "eos.db", graph_backend=InMemoryGraphBackend()
        )

        def chain(manifest_ref: str, manifest_hash: str):
            accepted, accepted_witnesses = (
                store.replay_source_semantic_witness_package_by_ref(
                    organization_id=self.organization_id,
                    workspace_id=self.workspace_id,
                    manifest_ref=manifest_ref,
                    content_hash=manifest_hash,
                )
            )
            if accepted.predecessor_manifest_ref is None:
                raise ValueError("successor accepted manifest lacks proposed predecessor")
            proposed, proposed_witnesses = (
                store.replay_source_semantic_witness_package_by_ref(
                    organization_id=self.organization_id,
                    workspace_id=self.workspace_id,
                    manifest_ref=accepted.predecessor_manifest_ref,
                )
            )
            if proposed.predecessor_manifest_ref is None:
                raise ValueError("successor proposed manifest lacks accepted predecessor")
            baseline, _baseline_witnesses = (
                store.replay_source_semantic_witness_package_by_ref(
                    organization_id=self.organization_id,
                    workspace_id=self.workspace_id,
                    manifest_ref=proposed.predecessor_manifest_ref,
                )
            )
            return accepted, accepted_witnesses, proposed, proposed_witnesses, baseline

        positive = chain(positive_manifest_ref, positive_manifest_hash)
        negative = chain(hard_negative_manifest_ref, hard_negative_manifest_hash)
        baseline_packages = _TRUSTED_W1_MANIFEST_RESOLVER_IMPLEMENTATION(
            positive_manifest_ref=positive[4].manifest_ref,
            positive_manifest_hash=positive[4].content_hash,
            hard_negative_manifest_ref=negative[4].manifest_ref,
            hard_negative_manifest_hash=negative[4].content_hash,
            positive_document=positive_document,
        )
        base_by_kind = {
            package.witness_kind: _resolve_trusted_witness_gate_package(package)
            for package in baseline_packages
        }
        resolved: list[_ResolvedAcceptedWitnessGatePackage] = []
        for kind, successor in (
            ("positive_exemplar", positive),
            ("hard_negative", negative),
        ):
            base = base_by_kind[kind]
            resolved.append(
                replace(
                    base,
                    proposed_manifest=successor[2],
                    proposed_witnesses=successor[3],
                    accepted_manifest=successor[0],
                    accepted_witnesses=successor[1],
                )
            )
        return tuple(resolved)

    def _governed_material_correction_document(
        self, graph_intent: GraphIntentBrief
    ) -> CanonicalSourceDocument:
        path = GraphSchemaDiscoveryService._resolve_canonical_source_ir_ref(
            self, graph_intent.source_binding.canonical_source_ir_ref
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        document = CanonicalSourceDocument.model_validate(payload)
        if (
            document.source_identity_ref
            != graph_intent.source_binding.source_identity_ref
            or canonical_ir_content_hash(payload)
            != graph_intent.source_binding.canonical_source_ir_hash
        ):
            raise ValueError(
                "material correction governed CanonicalSourceIR binding mismatch"
            )
        return document

    def resolve_registered_witness_packages(
        self, graph_intent: GraphIntentBrief
    ) -> tuple[AcceptedWitnessGatePackage, AcceptedWitnessGatePackage]:
        """Resolve source-bound W1 authority without accepting browser semantics."""

        _require_owned_lineage_authority(self)
        if (
            _resolve_trusted_witness_gate_packages_for_graph_intent
            is not _TRUSTED_W1_GRAPH_INTENT_RESOLVER_IMPLEMENTATION
        ):
            raise ValueError("trusted W1 GraphIntent resolver was replaced")
        document = _TRUSTED_GOVERNED_CORRECTION_DOCUMENT_DESCRIPTOR(
            self, graph_intent
        )
        return _TRUSTED_W1_GRAPH_INTENT_RESOLVER_IMPLEMENTATION(
            graph_intent=graph_intent,
            positive_document=document,
        )

    def registered_witness_selection(
        self, graph_intent: GraphIntentBrief
    ) -> dict[str, str]:
        """Project exact accepted witness pins for a governed UI context."""

        packages = GraphSchemaDiscoveryService.resolve_registered_witness_packages(
            self, graph_intent
        )
        resolved = tuple(
            _TRUSTED_WITNESS_RESOLVER_DESCRIPTOR(package)
            for package in packages
        )
        positive = next(
            item for item in resolved if item.witness_kind == "positive_exemplar"
        )
        return {
            "sourceSemanticWitnessSetRef": positive.accepted_manifest.manifest_ref,
            "sourceSemanticWitnessSetHash": positive.accepted_manifest.content_hash,
            "selectedSourceSemanticWitnessRef": positive.accepted_witnesses[0].witness_id,
            "selectedSourceSemanticWitnessHash": positive.accepted_witnesses[0].witness_hash,
        }

    @staticmethod
    def _resolve_witness_gate_package(
        package: AcceptedWitnessGatePackage,
    ) -> _ResolvedAcceptedWitnessGatePackage:
        return _resolve_trusted_witness_gate_package(package)

    @staticmethod
    def _native_terminal_linebreak_rebind(package, witness, graph_intent):
        """Prove a parser-only terminal LF/CR difference at the same DOCX cell.

        Historical evidence is never rewritten. The prospective gate records
        both identities; source bytes, native anchor and nonterminal text must
        agree. This is not fuzzy matching or new witness acceptance.
        """
        document = package.document
        binding = graph_intent.source_binding
        if (document.source_identity_ref != binding.source_identity_ref
            or canonical_ir_content_hash(document.model_dump(mode="json", by_alias=True,
                exclude_none=False)) != binding.canonical_source_ir_hash
            or witness.source_binding.source_hash != binding.source_hash
            or package.source_identity.source_hash != binding.source_hash
            or len(witness.evidence_refs) != 1):
            return None
        evidence = witness.evidence_refs[0]
        expected = evidence.selector.exact_quote
        anchor = evidence.provenance.get("source_locator", {})
        if (not isinstance(expected, str) or not expected
            or "sha256:" + sha256(expected.encode("utf-8")).hexdigest() != witness.exact_text_hash
            or anchor.get("format") != "docx" or anchor.get("nativeProvider") != "ooxml"
            or anchor.get("sourceHash") != binding.source_hash
            or not anchor.get("nativeId") or not anchor.get("packagePart")):
            return None
        matches = []
        for mapping in document.source_map.mappings:
            for locator in mapping.source_locators:
                if (locator.native_id, locator.package_part) == (anchor["nativeId"], anchor["packagePart"]):
                    matches.append((mapping, locator))
        if len(matches) != 1:
            return None
        mapping, locator = matches[0]
        if (len(mapping.block_ids) != 1 or locator.native_provider != "ooxml"
            or locator.format != "docx" or locator.source_hash != binding.source_hash
            or locator.source_identity_ref != binding.source_identity_ref):
            return None
        blocks = [b for b in document.blocks if b.block_id == mapping.block_ids[0]]
        if len(blocks) != 1:
            return None
        block = blocks[0]
        if (block.block_type != "cell" or not isinstance(block.content, str)
            or canonical_ir_content_hash(block.content) != block.content_hash
            or block.content == expected or block.content.rstrip("\r\n") != expected):
            return None
        return dict(match_kind="SAME_BYTES_NATIVE_ANCHOR_TERMINAL_LINEBREAK",
            witness_ref=witness.witness_id, witness_hash=witness.witness_hash,
            historical_text_hash=witness.exact_text_hash, source_hash=binding.source_hash,
            current_ir_hash=binding.canonical_source_ir_hash, current_block_id=block.block_id,
            current_text_hash="sha256:" + sha256(block.content.encode("utf-8")).hexdigest(),
            native_id=locator.native_id, package_part=locator.package_part)

    @staticmethod
    def _verify_accepted_witness_gate(
        witness_packages: tuple[_ResolvedAcceptedWitnessGatePackage, ...],
        *,
        graph_intent: GraphIntentBrief,
        allow_adjudicated_successor: bool = False,
        allow_neutral_source_successor: bool = False,
    ) -> dict[str, Any]:
        by_kind = {package.witness_kind: package for package in witness_packages}
        if set(by_kind) != {"positive_exemplar", "hard_negative"}:
            raise ValueError("accepted witness gate requires positive and hard-negative packages")
        if len(by_kind["positive_exemplar"].accepted_witnesses) != 32:
            raise ValueError("accepted witness gate requires 32 positive exemplars")
        if len(by_kind["hard_negative"].accepted_witnesses) != 6:
            raise ValueError("accepted witness gate requires six hard negatives")
        positive = by_kind["positive_exemplar"]
        trusted_spine = (
            positive.graph_intent.document_function.type_id,
            positive.graph_intent.represented_or_governed_subject.type_id,
            positive.graph_intent.primary_relationship.type_id,
            positive.graph_intent.preferred_representation.kind,
            positive.graph_intent.preferred_representation.type_id,
        )
        admitted_spine = (
            graph_intent.document_function.type_id,
            graph_intent.represented_or_governed_subject.type_id,
            graph_intent.primary_relationship.type_id,
            graph_intent.preferred_representation.kind,
            graph_intent.preferred_representation.type_id,
        )
        if admitted_spine != trusted_spine:
            raise ValueError("positive witness package semantic spine mismatch")

        mismatches: list[dict[str, str]] = []
        source_rebindings = []
        for package in witness_packages:
            if len(package.proposed_witnesses) != len(package.accepted_witnesses):
                raise ValueError("witness proposal/acceptance count mismatch")
            if any(
                witness.review.state != "proposed"
                for witness in package.proposed_witnesses
            ):
                raise ValueError("witness reconstruction requires proposed predecessors")
            if any(
                witness.review.state != "accepted"
                for witness in package.accepted_witnesses
            ):
                raise ValueError("witness gate accepts only adjudicated witnesses")
            package.proposed_manifest.verify_witnesses(package.proposed_witnesses)
            package.accepted_manifest.verify_witnesses(package.accepted_witnesses)
            if (
                package.accepted_manifest.predecessor_manifest_ref
                != package.proposed_manifest.manifest_ref
            ):
                raise ValueError("accepted witness manifest predecessor mismatch")
            for proposed, accepted in zip(
                package.proposed_witnesses,
                package.accepted_witnesses,
                strict=True,
            ):
                if accepted.review.predecessor_witness_ref != proposed.witness_id:
                    raise ValueError("accepted witness predecessor mismatch")
            expectations = {
                str(witness.evidence_refs[0].provenance["source_block_id"]): (
                    witness.semantic_expectation
                )
                for witness in package.proposed_witnesses
            }
            provenance = {
                str(witness.evidence_refs[0].provenance["source_block_id"]): {
                    key: value
                    for key, value in witness.evidence_refs[0].provenance.items()
                    if key
                    in {
                        "construction_kind",
                        "authority_promotion_allowed",
                        "lexical_overlap_positive_family_ref",
                    }
                }
                for witness in package.proposed_witnesses
            }
            if package.witness_kind == "positive_exemplar":
                current_text_hashes = {
                    "sha256:" + sha256(str(block.content).encode("utf-8")).hexdigest()
                    for block in package.document.blocks
                }
                for witness in package.proposed_witnesses:
                    if witness.exact_text_hash not in current_text_hashes:
                        proof = GraphSchemaDiscoveryService._native_terminal_linebreak_rebind(
                            package, witness, graph_intent)
                        if proof is None:
                            raise ValueError(
                                "runtime source does not reproduce trusted W1 positive witness"
                            )
                        accepted = next(item for item in package.accepted_witnesses
                            if item.review.predecessor_witness_ref == witness.witness_id)
                        source_rebindings.append({**proof,
                            "accepted_witness_ref": accepted.witness_id,
                            "accepted_witness_hash": accepted.witness_hash})
                discovered = list(package.proposed_witnesses)
            else:
                discovered = SourceSemanticWitnessService().discover(
                    document=package.document,
                    source_identity=package.source_identity,
                    graph_intent=package.graph_intent,
                    source_role_evidence=package.source_role_evidence,
                    actor_ref="system:w2c-witness-reconstruction",
                    review_set_ref=package.accepted_manifest.review_set_ref,
                    expectations_by_block_id=expectations,
                    witness_kind=package.witness_kind,
                    evidence_provenance_by_block_id=provenance,
                )
            accepted_by_block = {
                str(witness.evidence_refs[0].provenance["source_block_id"]): witness
                for witness in package.accepted_witnesses
            }
            for witness in discovered:
                block_id = str(
                    witness.evidence_refs[0].provenance["source_block_id"]
                )
                accepted = accepted_by_block.get(block_id)
                if accepted is None:
                    raise ValueError(
                        "witness reconstruction produced an unknown block"
                    )
                if (
                    witness.source_binding != accepted.source_binding
                    or witness.evidence_refs != accepted.evidence_refs
                    or witness.exact_text_hash != accepted.exact_text_hash
                ):
                    raise ValueError("witness reconstruction source evidence mismatch")
                if witness.semantic_expectation != accepted.semantic_expectation:
                    mismatches.append(
                        {
                            "witness_ref": accepted.witness_id,
                            "affected_semantics": "semantic_expectation",
                        }
                    )
            if len(discovered) != len(package.accepted_witnesses):
                raise ValueError("witness reconstruction count mismatch")

        negative = by_kind["hard_negative"]
        composed = next(
            (
                witness
                for witness in positive.accepted_witnesses
                if witness.semantic_expectation.representation_kind
                == "composed_frame_set"
            ),
            None,
        )
        if composed is None and not allow_adjudicated_successor:
            raise ValueError(
                "accepted positive witnesses require a composed representative"
            )
        representative = [
            positive.accepted_witnesses[0].witness_id,
            (composed or positive.accepted_witnesses[1]).witness_id,
        ]
        return {
            "schema_version": "accepted_exemplar_gate.v1",
            "positive_manifest_ref": positive.accepted_manifest.manifest_ref,
            "positive_manifest_hash": positive.accepted_manifest.content_hash,
            "positive_count": len(positive.accepted_witnesses),
            "representative_positive_witness_refs": representative,
            "hard_negative_manifest_ref": negative.accepted_manifest.manifest_ref,
            "hard_negative_manifest_hash": negative.accepted_manifest.content_hash,
            "hard_negative_count": len(negative.accepted_witnesses),
            "hard_negative_witness_refs": [
                witness.witness_id for witness in negative.accepted_witnesses
            ],
            "material_mismatches": mismatches,
            **({"source_rebindings": source_rebindings} if source_rebindings else {}),
        }

    def start_run(
        self,
        *,
        lineage_id: str,
        experiment_id: str,
        source_ref: str,
        source_hash: str,
        skill_ref: str = CANONICAL_JOURNEY_REF,
        created_by: str,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        selected = select_for_new_run(skill_ref=skill_ref)
        if selected is None:
            raise ValueError("new run requires a selectable canonical skill")
        return self.store.create_run(
            lineage_id=lineage_id,
            experiment_id=experiment_id,
            source_ref=source_ref,
            source_hash=source_hash,
            skill_ref=selected.skill_ref,
            created_by=created_by,
            run_id=run_id,
        ).as_dict()

    def import_historical_run(
        self,
        *,
        lineage_id: str,
        experiment_id: str,
        source_ref: str,
        source_hash: str,
        skill_ref: str,
        imported_by: str,
    ) -> dict[str, Any]:
        historical = resolve_historical_ref(skill_ref)
        return self.store.import_historical_run(
            lineage_id=lineage_id,
            experiment_id=experiment_id,
            source_ref=source_ref,
            source_hash=source_hash,
            skill_ref=historical.skill_ref,
            imported_by=imported_by,
        ).as_dict()

    def start_fresh_experiment_run(
        self,
        *,
        lineage_id: str,
        prior_experiment_id: str,
        new_experiment_id: str,
        source_ref: str,
        source_hash: str,
        skill_ref: str,
        created_by: str,
    ) -> dict[str, Any]:
        if prior_experiment_id == new_experiment_id:
            raise ValueError("fresh discovery requires a new experiment_id")
        try:
            self.selection_store.active_for_experiment(new_experiment_id)
        except MvlSelectionNotFound:
            pass
        else:
            raise ValueError(
                "fresh discovery experiment must start without selected candidate state"
            )
        return self.start_run(
            lineage_id=lineage_id,
            experiment_id=new_experiment_id,
            source_ref=source_ref,
            source_hash=source_hash,
            skill_ref=skill_ref,
            created_by=created_by,
        )

    def propose_decision(self, **kwargs: Any) -> dict[str, Any]:
        return self.store.create_decision(**kwargs)

    def record_human_review(
        self,
        decision_id: str,
        *,
        state: str,
        actor_id: str,
        note: str | None = None,
        edited_candidate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_edit = None
        if state == "edited":
            if not isinstance(edited_candidate, dict):
                raise ValueError(
                    "edited review requires a complete edited candidate payload"
                )
            decision = self.store.get_decision(decision_id)
            if decision["discovery_type"] == "corpus_graph_schema_candidate":
                original_candidate = decision["proposed_semantics"][
                    "machine_readable"
                ].get("candidate")
                self._validate_edited_candidate_source_binding(
                    original_candidate, edited_candidate
                )
                normalized_edit = self._authoritative_edited_candidate(edited_candidate)
                self.candidate_store.put(
                    CorpusGraphSchemaCandidate.from_dict(normalized_edit)
                )
            else:
                normalized_edit = edited_candidate
        elif edited_candidate is not None:
            raise ValueError("edited candidate payload is only valid for edited review")
        return self.store.record_human_review(
            decision_id,
            state=state,
            actor_id=actor_id,
            note=note,
            edited_payload=normalized_edit,
        )

    def create_brief(self, **kwargs: Any) -> dict[str, Any]:
        return self.store.create_brief(**kwargs)


    def get_schema_decision(self, schema_decision_id: str) -> dict[str, Any]:
        return self.get_schema_decision_contract(schema_decision_id).as_dict()

    def get_schema_decision_contract(
        self, schema_decision_id: str
    ) -> SchemaDecision:
        taxonomy_service = KnowledgeRelationTaxonomyService(
            repository_root=Path(__file__).resolve().parents[2],
            profile_registry=self.profile_registry,
            profile_relation_type_proposal_store=(
                self.profile_relation_type_proposal_store
            ),
        )
        return self.schema_decision_store.get_contract(
            schema_decision_id,
            taxonomy_service=taxonomy_service,
        )

    def record_llm_proposal_evidence(self, **kwargs: Any) -> dict[str, Any]:
        return self.store.record_llm_proposal_evidence(**kwargs)

    def record_llm_proposal_evidence_review(
        self, evidence_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        return self.store.record_llm_proposal_evidence_review(evidence_id, **kwargs)

    def latest_qualifying_llm_evidence(self, **kwargs: Any) -> dict[str, Any] | None:
        return self.store.latest_qualifying_llm_evidence(**kwargs)

    def run_llm_proposal_evidence(self, **kwargs: Any) -> dict[str, Any]:
        return GraphSchemaLlmEvidenceRunner(
            self.workspace_root,
            workspace_id=self.workspace_id,
            store=self.store,
        ).run(**kwargs)

    def induce_corpus_schema_candidate(
        self,
        *,
        lineage_id: str,
        experiment_id: str,
        actor_id: str,
        intended_use: str,
        canonical_source_ir_ref: str | None = None,
        source_path: str | Path = FIXED_SOURCE,
        source_ref: str = "repo://docs/tender-poc-sensitive-terms",
        profile_path: str | Path | None = None,
        profile_id: str | None = None,
        competency_questions: tuple[dict[str, str], ...] = (),
        skill_ref: str | None = None,
        llm_proposal_evidence_id: str | None = None,
    ) -> dict[str, Any]:
        """Induce and persist one reviewable schema candidate without accepting it."""

        if profile_id is not None and profile_path is not None:
            raise ValueError(
                "profile_id and legacy profile_path are mutually exclusive"
            )
        compiler_capability_ref = skill_ref or COMPILER_CAPABILITY_ID
        if compiler_capability_ref != COMPILER_CAPABILITY_ID:
            raise ValueError(
                "new induction requires corpus-graph-schema-induction "
                "compiler provenance"
            )
        if profile_id is not None:
            selected_profile_path = self.profile_registry.resolve(profile_id).path
        else:
            selected_profile_path = Path(profile_path or DEFAULT_PROFILE)

        questions = self._validated_competency_questions(
            competency_questions, intended_use=intended_use
        )
        if canonical_source_ir_ref is not None:
            canonical_path = self._resolve_canonical_source_ir_ref(
                canonical_source_ir_ref
            )
            canonical_document = CanonicalSourceDocument.model_validate_json(
                canonical_path.read_text(encoding="utf-8")
            )
            canonical_elements = canonical_elements_from_ir(canonical_document)
            parsed = {
                "document_hash": canonical_elements[0]["anchor"]["document_hash"],
                "elements": canonical_elements,
            }
            provenance_canonical_refs = [canonical_source_ir_ref]
        else:
            # Bounded compatibility path for historical tests/tools. Governed
            # Studio callers must supply ``canonical_source_ir_ref``.
            parsed = source_elements(Path(source_path))
            provenance_canonical_refs = None
        structural_candidate = CorpusGraphSchemaInductionKernel(
            profile_path=selected_profile_path
        ).induce(
            parsed_document=parsed,
            source_ref=source_ref,
            intended_use=intended_use,
            competency_questions=questions,
        )
        if canonical_source_ir_ref is not None:
            synthesizer = SemantierSchemaSynthesizer()
            semantica_evidence = SemanticaSchemaInducer(
                self.workspace_root, workspace_id=self.workspace_id
            ).induce(
                semantic_network=synthesizer.semantic_network(structural_candidate),
                competency_questions=tuple(item["text"] for item in questions),
            )
            candidate = synthesizer.synthesize(
                structural_candidate=structural_candidate,
                semantica_evidence=semantica_evidence,
                lineage_id=lineage_id,
                source_ref=source_ref,
                canonical_source_ref=canonical_source_ir_ref,
                llm_proposal_evidence=(
                    self._load_reviewed_llm_proposal(llm_proposal_evidence_id)
                    if llm_proposal_evidence_id is not None
                    else None
                ),
            )
        else:
            candidate = structural_candidate
        if candidate.schema_version == "corpus_graph_schema_candidate.v2":
            self.candidate_store.put(candidate)
        selector_kinds = sorted(
            {
                str(item["source_selector"]["kind"])
                for item in candidate.semantic_role_proposals
            }
        )
        selected_representations = sorted(
            {
                str(item["selected_representation"])
                for item in candidate.representation_decisions
            }
        )
        run = self.start_run(
            lineage_id=lineage_id,
            experiment_id=experiment_id,
            source_ref=source_ref,
            source_hash=str(parsed["document_hash"]),
            skill_ref=CANONICAL_JOURNEY_REF,
            created_by=actor_id,
        )
        decision = self.propose_decision(
            run_id=run["run_id"],
            discovery_type="corpus_graph_schema_candidate",
            subject={
                "source_element_refs": list(candidate.source_evidence_refs),
                "semantic_object_ref": candidate.candidate_id,
            },
            proposed_semantics={
                "statement": "The corpus structure and semantic signals propose a reusable graph schema.",
                "machine_readable": {"candidate": candidate.as_dict()},
            },
            rationale={
                "reasoning_summary": (
                    f"Observed source selector kinds {selector_kinds}; evidence-backed "
                    f"representation decisions selected {selected_representations}."
                ),
                "basis": [
                    "source_explicit",
                    "source_structure",
                    "deterministic_inference",
                ],
                "evidence_refs": list(candidate.source_evidence_refs),
                "structural_evidence_refs": [],
                "competency_question_refs": [item["question_id"] for item in questions],
            },
            alternatives_considered=[
                {
                    "alternative": "Use a different graph representation.",
                    "disposition": "requires_human_review",
                    "reason_summary": (
                        "Observed selectors and representation evidence remain subject "
                        "to human review authority."
                    ),
                }
            ],
            confidence=0.9,
            semantica_signals={
                "llm_proposal_ref": None,
                "deterministic_proposal_ref": candidate.candidate_id,
                "deterministic_proposal_hash": candidate.candidate_hash,
                "graph_validation_refs": [],
            },
            provenance={
                "canonical_source_ir_refs": (
                    provenance_canonical_refs
                    if provenance_canonical_refs is not None
                    else list(candidate.source_evidence_refs)
                ),
                "generated_by": "corpus_graph_schema_induction",
                "model_or_engine_ref": candidate.source_structure_profile["profile_id"],
            },
        )
        return {"run": run, "candidate": candidate.as_dict(), "decision": decision}

    def induce_graph_intent_schema_candidate(
        self,
        *,
        graph_intent_ref: str,
        graph_intent_hash: str,
        canonical_source_ir_ref: str,
        profile_id: str,
        actor_id: str,
        _adjudicated_evidence: dict[str, Any] | None = None,
        _adjudicated_authority: object | None = None,
    ) -> dict[str, Any]:
        """Compile one accepted/edited manifest-backed GraphIntent into v3."""

        if self.graph_intent_store is None:
            raise ValueError("authenticated organization scope is required")
        if not isinstance(graph_intent_ref, str) or not graph_intent_ref.strip():
            raise ValueError("graphIntentRef is required")
        if not isinstance(graph_intent_hash, str) or not graph_intent_hash.startswith(
            "sha256:"
        ):
            raise ValueError("graphIntentHash must be an exact sha256 ref")
        if not isinstance(canonical_source_ir_ref, str) or not canonical_source_ir_ref:
            raise ValueError("canonicalSourceIrRef is required")

        exact = self.graph_intent_store.get(
            graph_intent_ref, graph_intent_hash=graph_intent_hash
        )
        eligible = self.graph_intent_store.require_compiler_eligible(
            graph_intent_ref
        )
        if (
            eligible.graph_intent_id != exact.graph_intent_id
            or eligible.graph_intent_hash != exact.graph_intent_hash
        ):
            raise ValueError("graphIntentRef and graphIntentHash mismatch")
        if exact.source_binding.canonical_source_ir_ref != canonical_source_ir_ref:
            raise ValueError(
                "canonicalSourceIrRef does not match the accepted GraphIntent"
            )

        profile = self.profile_registry.resolve(profile_id)
        if _adjudicated_authority is _ADJUDICATED_GRAPH_INTENT_COMPILATION_AUTHORITY:
            if not isinstance(_adjudicated_evidence, dict):
                raise ValueError("adjudicated GraphIntent evidence is required")
            proposal_evidence = dict(_adjudicated_evidence)
        else:
            if _adjudicated_authority is not None or _adjudicated_evidence is not None:
                raise PermissionError("adjudicated GraphIntent compilation is sealed")
            intent_service = GraphIntentDiscoveryService(
                self.workspace_root,
                intent_store=self.graph_intent_store,
                profile_registry=self.profile_registry,
            )
            proposal_evidence = intent_service.load_proposal_evidence(exact)
        profile_migration = False
        if (
            proposal_evidence.get("profile_id") != profile.profile_id
            or proposal_evidence.get("profile_hash") != profile.content_hash
            or proposal_evidence.get("catalog_hash")
            != profile.graph_intent_catalog_hash
        ):
            if (exact.review.state != "edited" or exact.review.reviewed_by != "user"
                or exact.predecessor_graph_intent_ref is None
                or _adjudicated_authority is not None):
                raise ValueError("profileId does not match GraphIntent evidence authority")
            self.profile_registry.require_additive_type_successor(profile.profile_id,
                predecessor_id=proposal_evidence["profile_id"],
                predecessor_hash=proposal_evidence["profile_hash"],
                predecessor_catalog_hash=proposal_evidence["catalog_hash"])
            profile_migration = True
        catalog = self.profile_registry.graph_intent_catalog(
            profile.profile_id,
            expected_profile_hash=profile.content_hash,
            expected_catalog_hash=profile.graph_intent_catalog_hash,
        )
        if _adjudicated_authority is _ADJUDICATED_GRAPH_INTENT_COMPILATION_AUTHORITY:
            if not str(proposal_evidence.get("predecessor_graph_intent_hash") or "").startswith(
                "sha256:"
            ):
                raise ValueError("adjudicated GraphIntent predecessor pin is missing")
        else:
            self._require_intent_catalog_compatibility(
                graph_intent=exact,
                profile_id=profile.profile_id,
                profile_hash=profile.content_hash,
                catalog_hash=str(profile.graph_intent_catalog_hash),
                catalog=catalog,
            )

        canonical_path = self._resolve_canonical_source_ir_ref(
            canonical_source_ir_ref
        )
        canonical_document = CanonicalSourceDocument.model_validate_json(
            canonical_path.read_text(encoding="utf-8")
        )
        reviewed_dimension_blocks = None
        if profile_migration:
            reviewed_dimension_blocks = {
                "evidence:canonical-block:" + block.block_id.removeprefix("sha256:"): block.block_id
                for block in canonical_document.blocks
            }
            if any(not dimension.evidence_refs or any(ref not in reviewed_dimension_blocks
                for ref in dimension.evidence_refs) for dimension in exact.material_dimensions):
                raise ValueError("reviewed material dimensions require exact canonical source evidence")
        canonical_elements = canonical_elements_from_ir(canonical_document)
        parsed = {
            "document_hash": canonical_elements[0]["anchor"]["document_hash"],
            "elements": canonical_elements,
        }
        questions = tuple(
            {
                "question_id": item.question_id,
                "text": item.text,
            }
            for item in exact.competency_questions
        )
        structural_candidate = CorpusGraphSchemaInductionKernel(
            profile_path=profile.path
        ).induce(
            parsed_document=parsed,
            source_ref=f"workspace://{canonical_source_ir_ref}",
            intended_use=exact.graph_intent_id,
            competency_questions=questions,
        )
        synthesizer = SemantierSchemaSynthesizer()
        semantica_evidence = SemanticaSchemaInducer(
            self.workspace_root, workspace_id=self.workspace_id
        ).induce(
            semantic_network=synthesizer.semantic_network(
                structural_candidate, graph_intent=exact
            ),
            competency_questions=tuple(
                item.text for item in exact.competency_questions
            ),
        )
        candidate = synthesizer.synthesize_v3(
            structural_candidate=structural_candidate,
            graph_intent=exact,
            semantica_evidence=semantica_evidence,
            profile_id=profile.profile_id,
            profile_hash=profile.content_hash,
            catalog_hash=str(profile.graph_intent_catalog_hash),
            profile_payload=self.profile_registry.profile_payload(
                profile.profile_id,
                expected_profile_hash=profile.content_hash,
            ),
            graph_intent_proposal_evidence=proposal_evidence,
            reviewed_dimension_blocks=reviewed_dimension_blocks,
            adjudicated_evidence_refs=(
                tuple(proposal_evidence["source_block_evidence"])
                if _adjudicated_authority
                is _ADJUDICATED_GRAPH_INTENT_COMPILATION_AUTHORITY
                else None
            ),
        )
        self.candidate_store.put(candidate)
        compiler_copy = graph_intent_compiler_copy()
        publication_identity = {
            "workspace_id": self.workspace_id,
            "candidate_ref": candidate.candidate_id,
            "graph_intent_ref": exact.graph_intent_id,
            "operation": "graph_intent_schema_compilation",
        }
        publication_digest = content_hash(publication_identity)[:32]

        run = self.start_run(
            lineage_id=exact.source_binding.lineage_id,
            experiment_id=f"graph-intent-schema-{candidate.candidate_id}",
            source_ref=structural_candidate.source_structure_profile["source_ref"],
            source_hash=str(parsed["document_hash"]),
            skill_ref=CANONICAL_JOURNEY_REF,
            created_by=actor_id,
            run_id=f"gsdr_{publication_digest}",
        )
        decision = self.propose_decision(
            run_id=run["run_id"],
            discovery_type="corpus_graph_schema_candidate",
            subject={
                "source_element_refs": list(candidate.source_evidence_refs),
                "semantic_object_ref": candidate.candidate_id,
            },
            proposed_semantics={
                "statement": compiler_copy["candidate_statement"],
                "machine_readable": {"candidate": candidate.as_dict()},
            },
            rationale={
                "reasoning_summary": compiler_copy["reasoning_summary"],
                "basis": [
                    "accepted_graph_intent",
                    "canonical_source_ir",
                    "deterministic_inference",
                ],
                "evidence_refs": list(candidate.source_evidence_refs),
                "structural_evidence_refs": list(
                    structural_candidate.source_evidence_refs
                ),
                "competency_question_refs": [
                    item.question_id for item in exact.competency_questions
                ],
                "hard_negative_question_refs": [
                    item.question_id for item in exact.hard_negative_questions
                ],
            },
            alternatives_considered=[
                item.as_dict() for item in exact.alternatives_considered
            ],
            confidence=0.9,
            semantica_signals={
                "llm_proposal_ref": None,
                "deterministic_proposal_ref": candidate.candidate_id,
                "deterministic_proposal_hash": candidate.candidate_hash,
                "graph_validation_refs": [
                    semantica_evidence["shacl_report_ref"]
                ],
            },
            provenance={
                "canonical_source_ir_refs": [canonical_source_ir_ref],
                "generated_by": "knowledge_graph_building",
                "orchestration": {"skill_ref": CANONICAL_JOURNEY_REF},
                "compiler": {
                    "capability_ref": "corpus-graph-schema-induction",
                    "candidate_schema_version": candidate.schema_version,
                    "profile_id": profile.profile_id,
                    "profile_hash": profile.content_hash,
                    "catalog_hash": profile.graph_intent_catalog_hash,
                },
                "graph_intent": {
                    "ref": exact.graph_intent_id,
                    "hash": exact.graph_intent_hash,
                },
                "semantica": {
                    "commit": semantica_evidence["semantica_commit"],
                    "input_hash": semantica_evidence["input_hash"],
                },
            },
            decision_id=f"gsdd_{publication_digest}",
            idempotent=True,
        )
        return {
            "run": self.store.get_run(run["run_id"]),
            "candidate": candidate.as_dict(),
            "decision": self.store.get_decision(decision["decision_id"]),
        }

    @staticmethod
    def _require_intent_catalog_compatibility(
        *,
        graph_intent: Any,
        profile_id: str,
        profile_hash: str,
        catalog_hash: str,
        catalog: dict[str, Any],
    ) -> None:
        admitted_types = set(map(str, catalog.get("admitted_type_ids") or []))
        admitted_relations = set(
            map(str, catalog.get("admitted_relation_ids") or [])
        )
        admitted_kinds = set(
            map(str, catalog.get("admitted_representation_kinds") or [])
        )
        primary = graph_intent.primary_relationship
        intent_type_ids = {
            graph_intent.document_function.type_id,
            graph_intent.represented_or_governed_subject.type_id,
            *primary.source_types,
            *primary.target_types,
        }
        intent_type_ids.update(
            item
            for values in graph_intent.source_role_interpretation.as_dict().values()
            for item in values
        )
        representation_type = graph_intent.preferred_representation.type_id
        if representation_type is not None:
            intent_type_ids.add(representation_type)
        intent_type_ids.update(
            item.type_id for item in graph_intent.material_dimensions
        )
        unresolved: dict[str, list[str]] = {}
        unknown_types = sorted(intent_type_ids - admitted_types)
        if unknown_types:
            unresolved["type_ids"] = unknown_types
        if primary.type_id not in admitted_relations:
            unresolved["relation_type_ids"] = [primary.type_id]
        if graph_intent.preferred_representation.kind not in admitted_kinds:
            unresolved["representation_kinds"] = [
                graph_intent.preferred_representation.kind
            ]
        if unresolved:
            raise GraphIntentCompatibilityError(
                {
                    "schema_version": "graph_intent_compiler_guard.v1",
                    "outcome": "refused",
                    "material": True,
                    "guard_id": "profile_intent_vocabulary_compatibility",
                    "reason_code": "PROFILE_INTENT_VOCABULARY_MISMATCH",
                    "affected_semantics": [
                        "represented_subject",
                        "primary_relationship",
                        "preferred_representation",
                        "material_dimensions",
                    ],
                    "unresolved_vocabulary": dict(sorted(unresolved.items())),
                    "pins": {
                        "graph_intent_ref": graph_intent.graph_intent_id,
                        "graph_intent_hash": graph_intent.graph_intent_hash,
                        "profile_id": profile_id,
                        "profile_hash": profile_hash,
                        "catalog_hash": catalog_hash,
                    },
                }
            )

    def _load_reviewed_llm_proposal(self, evidence_id: str) -> dict[str, Any]:
        evidence = self.store.get_llm_proposal_evidence(evidence_id)
        review = dict((evidence.get("diagnostics") or {}).get("review") or {})
        if evidence.get("status") != "passed" or review.get("state") != "accepted":
            raise ValueError(
                "optional LLM proposal evidence must be passed and reviewed"
            )
        proposal_ref = str(evidence.get("proposal_ref") or "")
        workspace_path = resolve_active_workspace_path(
            self.workspace_root,
            workspace_id=self.workspace_id,
        )
        root = (workspace_path / "artifacts").resolve()
        path = (workspace_path / proposal_ref).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(
                "optional LLM proposal artifact is outside the governed workspace"
            )
        proposal = json.loads(path.read_text(encoding="utf-8"))
        if f"sha256:{content_hash(proposal)}" != evidence.get("proposal_hash"):
            raise ValueError("optional LLM proposal artifact hash mismatch")
        return {**evidence, "proposal": proposal}

    def _resolve_canonical_source_ir_ref(self, artifact_ref: str) -> Path:
        """Resolve one CanonicalSourceIR inside this workspace's artifact root."""

        workspace_root = resolve_active_workspace_path(
            self.workspace_root,
            workspace_id=self.workspace_id,
        )
        governed_root = (workspace_root / "artifacts").resolve()
        candidate = Path(artifact_ref)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            relative = candidate.as_posix()
            if relative.startswith("artifacts/"):
                relative = relative.removeprefix("artifacts/")
            resolved = (governed_root / relative).resolve()
        if resolved != governed_root and governed_root not in resolved.parents:
            raise ValueError(
                "canonical_source_ir_ref must resolve inside the governed workspace artifact root"
            )
        if not resolved.is_file():
            raise ValueError(f"canonical source IR artifact not found: {artifact_ref}")
        return resolved

    @staticmethod
    def _validated_competency_questions(
        competency_questions: tuple[dict[str, str], ...], *, intended_use: str
    ) -> tuple[dict[str, str], ...]:
        if not isinstance(competency_questions, tuple):
            raise ValueError("competency_questions must be a tuple of objects")
        if not competency_questions:
            if not isinstance(intended_use, str) or not intended_use.strip():
                raise ValueError("intended_use is required")
            return (
                {
                    "question_id": f"cq:intended-use:{content_hash(intended_use)[:24]}",
                    "text": intended_use,
                },
            )

        copied: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        for item in competency_questions:
            if not isinstance(item, dict):
                raise ValueError("competency_questions items must be objects")
            if set(item) != {"question_id", "text"}:
                raise ValueError(
                    "competency_questions items must contain exactly question_id and text"
                )
            question_id = item.get("question_id")
            if not isinstance(question_id, str) or not question_id.strip():
                raise ValueError("competency_questions.question_id is required")
            try:
                question_id.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ValueError(
                    "competency_questions.question_id must be ASCII-stable"
                ) from exc
            if question_id in seen_ids:
                raise ValueError(
                    "competency_questions.question_id values must be unique"
                )
            question_text = item.get("text")
            if not isinstance(question_text, str) or not question_text.strip():
                raise ValueError("competency_questions.text is required")
            seen_ids.add(question_id)
            copied.append(dict(item))
        return tuple(copied)

    @staticmethod
    def _authoritative_edited_candidate(
        edited_candidate: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_content = dict(edited_candidate)
        candidate_content.pop("candidate_id", None)
        candidate_content.pop("candidate_hash", None)
        version = candidate_content.get("schema_version")
        if version == CANDIDATE_V3_SCHEMA_VERSION:
            return CorpusGraphSchemaCandidate.create_v3(candidate_content).as_dict()
        if version == CANDIDATE_V2_SCHEMA_VERSION:
            return CorpusGraphSchemaCandidate.create_v2(candidate_content).as_dict()
        provisional = CorpusGraphSchemaCandidate.from_dict(
            {
                **candidate_content,
                "candidate_id": "cgsc_edited_candidate_validation",
                "candidate_hash": "sha256:edited_candidate_validation",
            }
        ).as_dict()
        provisional.pop("candidate_id")
        provisional.pop("candidate_hash")
        digest = content_hash(provisional)
        return CorpusGraphSchemaCandidate.from_dict(
            {
                **provisional,
                "candidate_id": f"cgsc_{digest[:24]}",
                "candidate_hash": f"sha256:{digest}",
            }
        ).as_dict()

    @staticmethod
    def _validate_edited_candidate_source_binding(
        original_candidate: dict[str, Any],
        edited_candidate: dict[str, Any],
    ) -> None:
        if not isinstance(original_candidate, dict):
            raise ValueError("original candidate source binding is missing")
        if not isinstance(edited_candidate, dict):
            raise ValueError("edited candidate source binding is missing")
        immutable_fields = ["source_structure_profile", "source_evidence_refs"]
        if original_candidate.get("schema_version") == CANDIDATE_V3_SCHEMA_VERSION:
            immutable_fields.extend(
                [
                    "source_binding",
                    "graph_intent_binding",
                    "compiler_provenance",
                    "semantica_schema_evidence",
                    "document_understanding",
                    "semantic_role_proposals",
                    "synthesis_decisions",
                    "extraction_contract_hints",
                    "competency_questions",
                ]
            )
        if any(
            edited_candidate.get(field) != original_candidate.get(field)
            for field in immutable_fields
        ):
            raise ValueError(
                "edited candidate must preserve the immutable source binding and governed pins"
            )

    def accepted_corpus_schema_projection(self, decision_id: str) -> dict[str, Any]:
        """Project an accepted schema decision into the extraction/compiler contract."""

        decision = self.store.get_decision(decision_id)
        if decision["discovery_type"] != "corpus_graph_schema_candidate":
            raise ValueError("decision is not a corpus graph schema candidate")
        if decision["human_review"]["state"] not in {"accepted", "edited"}:
            raise ValueError("corpus graph schema candidate requires human acceptance")
        review = decision["human_review"]
        if review["state"] == "edited":
            payload = review.get("edited_payload")
            payload_hash = review.get("edited_payload_hash")
            binding = review.get("edited_payload_binding")
            if not isinstance(payload, dict) or payload_hash != (
                f"sha256:{content_hash(payload)}"
            ):
                raise ValueError(
                    "edited corpus graph schema payload is missing or tampered"
                )
            expected_binding = f"sha256:{content_hash({'decision_hash': decision['decision_hash'], 'edited_payload_hash': payload_hash})}"
            if binding != expected_binding:
                raise ValueError(
                    "edited corpus graph schema binding is missing or tampered"
                )
            authoritative = self._authoritative_edited_candidate(payload)
            if authoritative != payload:
                raise ValueError("edited corpus graph schema identity is invalid")
        else:
            payload = decision["proposed_semantics"]["machine_readable"].get(
                "candidate"
            )
        candidate = CorpusGraphSchemaCandidate.from_dict(payload)
        return {
            **candidate.extraction_contract_hints,
            "schema_candidate_id": candidate.candidate_id,
            "schema_candidate_hash": candidate.candidate_hash,
            "decision_id": decision_id,
            "review_state": decision["human_review"]["state"],
            "competency_questions": [
                dict(item) for item in candidate.competency_questions
            ],
            "event_types": [dict(item) for item in candidate.proposed_event_types],
        }

    def list_corpus_schema_candidates(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return self.store.list_decisions_by_type(
            "corpus_graph_schema_candidate", limit=limit
        )

    def run_fixed_source_v0_for_mvl(
        self,
        *,
        lineage_id: str,
        experiment_id: str,
        organization_id: str,
        actor_id: str,
        profile_ref: str,
        profile_version: str,
        source_ref: str = "repo://fixed-tender-source/phase1-001",
        skill_ref: str = CANONICAL_JOURNEY_REF,
        review_note: str = "Reviewed deterministic Semantica V0 proposal for MVL use.",
        max_decisions: int = 12,
    ) -> dict[str, Any]:
        """Run the fixed-source real Semantica deterministic V0 path and select it.

        This is the W2 MVL bridge for the repository-owned tender source. It
        deliberately composes existing native Semantica build/version surfaces
        and the thin MVL control adapter instead of adding another semantic
        proposal engine.
        """

        if max_decisions <= 0:
            raise ValueError("max_decisions must be positive")
        control = SemanticaMvlControlAdapter(
            self.workspace_root,
            organization_id=organization_id,
            workspace_id=self.workspace_id,
            domain_id=GRAPH_ID,
        )
        build = build_v0_candidate_with_result(
            control.repository,
            lineage_id=lineage_id,
            created_by_user_id=actor_id,
        )
        run = self.start_run(
            lineage_id=lineage_id,
            experiment_id=experiment_id,
            source_ref=source_ref,
            source_hash=build.source_hash,
            skill_ref=skill_ref,
            created_by=actor_id,
        )
        decisions = []
        proposal = build.proposal_result
        proposal_ref = proposal.native_artifact_ref
        proposal_hash = proposal.native_artifact_hash
        for node in proposal.graph.nodes[:max_decisions]:
            decision = self.propose_decision(
                run_id=run["run_id"],
                discovery_type="candidate_concept",
                subject={
                    "source_element_refs": self._anchor_refs(node.source_anchors),
                    "semantic_object_ref": node.node_id,
                },
                proposed_semantics={
                    "statement": "Semantica deterministic extraction proposed a candidate concept.",
                    "machine_readable": {
                        "role": node.node_type.value,
                        "canonical_label": node.canonical_label,
                        "normalized_value": node.normalized_value,
                        "display_name_zh": node.display_name_zh,
                    },
                },
                rationale={
                    "reasoning_summary": node.rationale,
                    "basis": ["source_explicit", "semantica_deterministic_extraction"],
                    "evidence_refs": self._anchor_refs(node.source_anchors),
                    "structural_evidence_refs": [],
                    "competency_question_refs": ["cq:detect-tender-sensitive-term"],
                },
                alternatives_considered=[
                    {
                        "alternative": "Treat the source span as a ReferenceExample only.",
                        "disposition": "requires_human_review_for_intent",
                        "reason_summary": "Source-role intent is Semantier-specific even when Semantica extracts the entity.",
                    }
                ],
                confidence=float(node.machine_confidence),
                semantica_signals={
                    "llm_proposal_ref": None,
                    "deterministic_proposal_ref": proposal_ref,
                    "deterministic_proposal_hash": proposal_hash,
                    "graph_validation_refs": [build.metadata.artifact_id],
                },
                provenance={
                    "canonical_source_ir_refs": self._anchor_refs(node.source_anchors),
                    "generated_by": "semantica_deterministic",
                    "model_or_engine_ref": proposal.provider_ref,
                    "semantica_commit": proposal.provider_commit,
                    "native_artifact_ref": proposal_ref,
                    "native_artifact_hash": proposal_hash,
                },
            )
            decisions.append(
                self.record_human_review(
                    decision["decision_id"],
                    state="accepted",
                    actor_id=actor_id,
                    note=review_note,
                )
            )

        brief = self.create_brief(
            run_id=run["run_id"],
            semantic_roles=[
                {
                    "role": "CandidateConcept",
                    "source_ref": source_ref,
                    "count": len(proposal.graph.nodes),
                },
                {
                    "role": "CandidateRule",
                    "source_ref": source_ref,
                    "count": len(proposal.graph.rules),
                },
            ],
            semantic_dimensions=[
                {
                    "dimension_id": GRAPH_ID,
                    "role": "domain",
                    "structure_kind": "candidate_context_graph",
                    "representation_hint": "Semantica ContextGraph",
                }
            ],
            hierarchy_edges=[],
            inheritance_policies=[],
            competency_questions=[
                {
                    "question_ref": "cq:detect-tender-sensitive-term",
                    "question": "Can the selected V0 candidate graph support tender sensitive-term detection?",
                }
            ],
        )
        graph_snapshot = control.create_native_graph_version(
            graph_payload=self._graph_payload(
                control.repository.load_graph(build.metadata.artifact_id)
            ),
            version_label=f"{experiment_id}-graph-v0",
            author="semantier-mvl@system.invalid",
            description="Reviewed deterministic Semantica graph V0 for MVL use.",
            experiment_id=experiment_id,
            actor_id=actor_id,
            source_ref=source_ref,
            source_hash=build.source_hash,
            semantic_stage="selected_for_mvl",
            control_action="USE_FOR_MVL",
            provenance_refs=[proposal_ref, brief["brief_id"]],
        )
        ontology_payload = self._ontology_payload_from_proposal(
            proposal.graph, experiment_id
        )
        ontology_snapshot = control.create_native_ontology_version(
            ontology_payload=ontology_payload,
            version_label=f"{experiment_id}-ontology-v0",
            author="semantier-mvl@system.invalid",
            description="Reviewed deterministic Semantica ontology V0 for MVL use.",
            experiment_id=experiment_id,
            actor_id=actor_id,
            source_ref=source_ref,
            source_hash=build.source_hash,
            semantic_stage="selected_for_mvl",
            control_action="USE_FOR_MVL",
            provenance_refs=[proposal_ref, brief["brief_id"]],
        )
        selection = control.select_graph_for_mvl(
            experiment_id=experiment_id,
            profile_ref=profile_ref,
            profile_version=profile_version,
            graph_artifact_id=build.metadata.artifact_id,
            ontology_semantica_version_id=f"{experiment_id}-ontology-v0",
            ontology_checksum=ontology_snapshot["checksum"],
            selected_by=actor_id,
        )
        return {
            "run": run,
            "brief": brief,
            "decisions": decisions,
            "graph_artifact": build.metadata.as_dict(),
            "graph_snapshot": graph_snapshot,
            "ontology_snapshot": ontology_snapshot,
            "selection": selection.__dict__.copy(),
            "deterministic_proposal_ref": proposal_ref,
            "deterministic_proposal_hash": proposal_hash,
        }

    @staticmethod
    def _anchor_refs(anchors: Any) -> list[str]:
        return [str(anchor.anchor_id) for anchor in anchors]

    @staticmethod
    def _graph_payload(graph: Any) -> dict[str, Any]:
        return {
            "graph_id": graph.graph_id,
            "nodes": [node.to_dict() for node in graph.nodes.values()],
            "edges": [edge.to_dict() for edge in graph.edges],
        }

    @staticmethod
    def _ontology_payload_from_proposal(
        candidate_graph: Any, experiment_id: str
    ) -> dict[str, Any]:
        classes = sorted({node.node_type.value for node in candidate_graph.nodes})
        relation_types = sorted(
            {edge.relation_type.value for edge in candidate_graph.edges}
        )
        return {
            "uri": f"urn:semantier:ontology:{GRAPH_ID}:{experiment_id}",
            "version_info": {"version": f"{experiment_id}-ontology-v0"},
            "structure": {
                "classes": classes,
                "properties": relation_types,
                "individuals": [],
                "axioms": [
                    {
                        "kind": "source_derived_candidate",
                        "node_count": len(candidate_graph.nodes),
                        "edge_count": len(candidate_graph.edges),
                        "rule_count": len(candidate_graph.rules),
                    }
                ],
            },
        }


_TRUSTED_WITNESS_RESOLVER_DESCRIPTOR = (
    GraphSchemaDiscoveryService.__dict__["_resolve_witness_gate_package"]
)
_TRUSTED_PERSISTED_RECONSTRUCTION_DESCRIPTOR = (
    GraphSchemaDiscoveryService.__dict__[
        "reconstruct_persisted_material_correction_witnesses"
    ]
)
_TRUSTED_MATERIAL_RECONSTRUCTION_DESCRIPTOR = (
    GraphSchemaDiscoveryService.__dict__["reconstruct_material_correction_witnesses"]
)
_TRUSTED_GOVERNED_CORRECTION_DOCUMENT_DESCRIPTOR = (
    GraphSchemaDiscoveryService.__dict__["_governed_material_correction_document"]
)
_TRUSTED_SCHEMA_ONLY_SUCCESSOR_DESCRIPTOR = (
    GraphSchemaDiscoveryService.__dict__["compile_schema_only_successor"]
)
_TRUSTED_REGISTERED_WITNESS_PACKAGES_DESCRIPTOR = (
    GraphSchemaDiscoveryService.__dict__["resolve_registered_witness_packages"]
)
_TRUSTED_REGISTERED_WITNESS_SELECTION_DESCRIPTOR = (
    GraphSchemaDiscoveryService.__dict__["registered_witness_selection"]
)
