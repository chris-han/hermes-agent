"""Governed canonical/profile knowledge-relation taxonomy resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from contracts.corpus_graph_schema import RelationTypeCandidate
from contracts.knowledge_relation_taxonomy import (
    KnowledgeRelationAdmission,
    KnowledgeRelationProfileCatalogV1,
    KnowledgeRelationTaxonomyV1,
    KnowledgeRelationType,
)
from services.corpus_graph_schema_profile_registry import (
    CorpusGraphSchemaProfileRegistry,
)

if TYPE_CHECKING:
    from contracts.schema_decision import SchemaDecision


DEFAULT_TAXONOMY_REF = "knowledge-relation-taxonomy:semantier.core@v1"


class KnowledgeRelationTaxonomyService:
    """Resolve exact immutable taxonomy/profile pins and admitted relation types."""

    __slots__ = (
        "_catalogs_root",
        "_profile_registry",
        "_profile_relation_type_proposal_store",
        "_repository_root",
        "_sealed",
    )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("governed taxonomy service composition is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        repository_root: str | Path | None = None,
        profile_registry: CorpusGraphSchemaProfileRegistry | None = None,
        profile_relation_type_proposal_store: object | None = None,
    ) -> None:
        if (
            profile_registry is not None
            and type(profile_registry) is not CorpusGraphSchemaProfileRegistry
        ):
            raise ValueError(
                "knowledge relation authority requires the exact governed profile registry"
            )
        if profile_relation_type_proposal_store is not None:
            from storage.profile_relation_type_proposal_store import (
                ProfileRelationTypeProposalStore,
            )

            if type(profile_relation_type_proposal_store) is not ProfileRelationTypeProposalStore:
                raise ValueError(
                    "knowledge relation authority requires the exact proposal store"
                )
        object.__setattr__(self, "_sealed", False)
        repository_root_path = (
            Path(repository_root).resolve()
            if repository_root is not None
            else Path(__file__).resolve().parents[2]
        )
        object.__setattr__(self, "_repository_root", repository_root_path)
        object.__setattr__(
            self,
            "_catalogs_root",
            (repository_root_path / "src/catalogs").resolve(),
        )
        object.__setattr__(
            self,
            "_profile_registry",
            profile_registry
            or CorpusGraphSchemaProfileRegistry(repository_root=repository_root_path),
        )
        object.__setattr__(
            self,
            "_profile_relation_type_proposal_store",
            profile_relation_type_proposal_store,
        )
        object.__setattr__(self, "_sealed", True)

    @property
    def repository_root(self) -> Path:
        return self._repository_root

    @property
    def catalogs_root(self) -> Path:
        return self._catalogs_root

    @property
    def profile_registry(self) -> CorpusGraphSchemaProfileRegistry:
        return self._profile_registry

    @property
    def profile_relation_type_proposal_store(self) -> object | None:
        return self._profile_relation_type_proposal_store

    def _require_owned_composition(self) -> None:
        if type(self) is not KnowledgeRelationTaxonomyService:
            raise ValueError("decision v3 requires the exact governed taxonomy service")
        if type(self.profile_registry) is not CorpusGraphSchemaProfileRegistry:
            raise ValueError(
                "decision v3 requires the exact governed profile registry"
            )

    def create_schema_decision_v3(self, content: dict) -> "SchemaDecision":
        """Construct Decision v3 only through this governed authority composition."""

        KnowledgeRelationTaxonomyService._require_owned_composition(self)
        from contracts.schema_decision import SchemaDecision

        return SchemaDecision._create_v3_with_governed_service(
            content,
            taxonomy_service=self,
        )

    def replay_schema_decision_v3(self, payload: dict) -> "SchemaDecision":
        """Replay Decision v3 through the same governed authority composition."""

        KnowledgeRelationTaxonomyService._require_owned_composition(self)
        from contracts.schema_decision import SchemaDecision

        return SchemaDecision._replay_v3_with_governed_service(
            payload,
            taxonomy_service=self,
        )

    def load_canonical(
        self,
        taxonomy_ref: str = DEFAULT_TAXONOMY_REF,
        *,
        expected_taxonomy_hash: str | None = None,
    ) -> KnowledgeRelationTaxonomyV1:
        matches: list[KnowledgeRelationTaxonomyV1] = []
        if not self.catalogs_root.is_dir():
            raise ValueError("governed knowledge-relation taxonomy catalog root is missing")
        for path in sorted(self.catalogs_root.glob("knowledge_relation_taxonomy.v*.json")):
            resolved = path.resolve()
            if not resolved.is_relative_to(self.catalogs_root):
                raise ValueError("taxonomy catalog path escapes governed catalog root")
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            taxonomy = KnowledgeRelationTaxonomyV1.model_validate(payload)
            if path.name != f"knowledge_relation_taxonomy.v{taxonomy.version}.json":
                raise ValueError("taxonomy catalog filename/version mismatch")
            if taxonomy.taxonomy_ref == taxonomy_ref:
                matches.append(taxonomy)
        if len(matches) != 1:
            raise ValueError(f"unknown governed taxonomy_ref: {taxonomy_ref}")
        taxonomy = matches[0]
        if (
            expected_taxonomy_hash is not None
            and taxonomy.taxonomy_hash != expected_taxonomy_hash
        ):
            raise ValueError("expected taxonomy hash mismatch")
        return taxonomy

    def resolve_taxonomy(
        self, taxonomy_ref: str, taxonomy_hash: str
    ) -> KnowledgeRelationTaxonomyV1:
        return KnowledgeRelationTaxonomyService.load_canonical(
            self,
            taxonomy_ref, expected_taxonomy_hash=taxonomy_hash
        )

    def load_profile_catalog(
        self,
        *,
        profile_ref: str,
        profile_hash: str,
        profile_relation_catalog_hash: str,
        taxonomy_ref: str,
        taxonomy_hash: str,
    ) -> KnowledgeRelationProfileCatalogV1:
        KnowledgeRelationTaxonomyService.resolve_taxonomy(
            self, taxonomy_ref, taxonomy_hash
        )
        payload = CorpusGraphSchemaProfileRegistry.knowledge_relation_catalog(
            self.profile_registry,
            profile_ref,
            expected_profile_hash=profile_hash,
            expected_catalog_hash=profile_relation_catalog_hash,
        )
        catalog = KnowledgeRelationProfileCatalogV1.model_validate(payload)
        if catalog.catalog_hash() != profile_relation_catalog_hash:
            raise ValueError("profile relation catalog canonical hash mismatch")
        if catalog.taxonomy_ref != taxonomy_ref:
            raise ValueError("profile relation catalog taxonomy ref mismatch")
        if catalog.taxonomy_hash != taxonomy_hash:
            raise ValueError("profile relation catalog taxonomy hash mismatch")
        return catalog

    def resolve_relation_type(
        self,
        *,
        relation_type_id: str,
        taxonomy_ref: str,
        taxonomy_hash: str,
        profile_ref: str | None = None,
        profile_hash: str | None = None,
        profile_relation_catalog_hash: str | None = None,
        for_new_write: bool,
        proposal_origin: Literal["system", "human"] = "system",
    ) -> KnowledgeRelationAdmission:
        # Both proposal origins deliberately traverse this identical authority path.
        if proposal_origin not in ("system", "human"):
            raise ValueError("proposal_origin must be system or human")
        taxonomy = KnowledgeRelationTaxonomyService.resolve_taxonomy(
            self, taxonomy_ref, taxonomy_hash
        )
        profile_pins = (profile_ref, profile_hash, profile_relation_catalog_hash)
        relation_type: KnowledgeRelationType | None
        if all(pin is None for pin in profile_pins):
            relation_type = KnowledgeRelationTaxonomyService._find_relation_type(
                taxonomy.relation_types, relation_type_id
            )
        elif any(pin is None for pin in profile_pins):
            raise ValueError(
                "profile-defined resolution requires profile_ref, profile_hash, and "
                "profile_relation_catalog_hash"
            )
        else:
            catalog = KnowledgeRelationTaxonomyService.load_profile_catalog(
                self,
                profile_ref=profile_ref,
                profile_hash=profile_hash,
                profile_relation_catalog_hash=profile_relation_catalog_hash,
                taxonomy_ref=taxonomy_ref,
                taxonomy_hash=taxonomy_hash,
            )
            relation_type = KnowledgeRelationTaxonomyService._find_relation_type(
                catalog.relation_types, relation_type_id
            )
        if relation_type is None:
            raise ValueError(f"unknown admitted relation_type_id: {relation_type_id}")
        if for_new_write and relation_type.lifecycle_state != "admitted":
            raise ValueError(
                f"{relation_type.lifecycle_state} relation type cannot enter a new write"
            )
        return KnowledgeRelationAdmission(
            relationType=relation_type,
            taxonomyRef=taxonomy.taxonomy_ref,
            taxonomyHash=taxonomy.taxonomy_hash,
            profileRef=profile_ref,
            profileHash=profile_hash,
            profileRelationCatalogHash=profile_relation_catalog_hash,
        )

    def resolve_candidate_admission(
        self,
        candidate: RelationTypeCandidate,
        *,
        proposal_origin: Literal["system", "human"],
    ) -> KnowledgeRelationAdmission:
        """Resolve and exactly compare a source-specific candidate for a new write."""

        admission = KnowledgeRelationTaxonomyService.resolve_relation_type(
            self,
            relation_type_id=candidate.relation_type_id,
            taxonomy_ref=candidate.taxonomy_ref,
            taxonomy_hash=candidate.taxonomy_hash,
            profile_ref=candidate.profile_ref,
            profile_hash=candidate.profile_hash,
            profile_relation_catalog_hash=candidate.profile_relation_catalog_hash,
            for_new_write=True,
            proposal_origin=proposal_origin,
        )
        relation_type = admission.relation_type
        for field_name in (
            "relation_type_id",
            "family_id",
            "definition_scope",
            "namespace",
            "representation_kind",
            "projection_shape",
            "source_type_refs",
            "target_type_refs",
            "source_role_refs",
            "target_role_refs",
            "directed",
        ):
            if getattr(candidate, field_name) != getattr(relation_type, field_name):
                raise ValueError(
                    f"resolved relation definition mismatch: {field_name}"
                )
        return admission

    def validate_candidate_proposal_lineage(
        self,
        candidate: RelationTypeCandidate,
        *,
        source_binding: dict,
        graph_intent_binding: dict,
    ) -> None:
        """Require proposal-bound candidates to replay their exact workspace artifact."""

        proposal_ref = candidate.profile_relation_type_proposal_ref
        proposal_hash = candidate.profile_relation_type_proposal_hash
        if proposal_ref is None and proposal_hash is None:
            return
        from contracts.profile_relation_type_proposal import (
            ProfileRelationTypeProposalV1,
        )
        from storage.profile_relation_type_proposal_store import (
            ProfileRelationTypeProposalStore,
        )

        store = self.profile_relation_type_proposal_store
        if type(store) is not ProfileRelationTypeProposalStore:
            raise ValueError(
                "exact persisted proposal store is required for proposal lineage"
            )
        proposal = ProfileRelationTypeProposalV1.model_validate(
            ProfileRelationTypeProposalStore.get(
                store,
                proposal_ref=str(proposal_ref),
                proposal_hash=str(proposal_hash),
                organization_id=source_binding.get("organization_id"),
            )
        )
        if proposal.source_binding != source_binding:
            raise ValueError("candidate proposal source binding mismatch")
        if (
            proposal.graph_intent_binding["graph_intent_ref"]
            != graph_intent_binding.get("graph_intent_ref")
            or proposal.graph_intent_binding["graph_intent_hash"]
            != graph_intent_binding.get("graph_intent_hash")
        ):
            raise ValueError("candidate proposal GraphIntent binding mismatch")
        if proposal.evidence_refs != candidate.evidence_refs:
            raise ValueError("candidate proposal EvidenceRefs mismatch")
        predecessor = proposal.predecessor_binding
        if (
            predecessor["taxonomy_ref"] != candidate.taxonomy_ref
            or predecessor["taxonomy_hash"] != candidate.taxonomy_hash
        ):
            raise ValueError("candidate proposal taxonomy predecessor mismatch")
        predecessor_profile = CorpusGraphSchemaProfileRegistry.resolve(
            self.profile_registry, predecessor["profile_ref"]
        )
        if predecessor_profile.content_hash != predecessor["profile_hash"]:
            raise ValueError("candidate proposal profile predecessor mismatch")
        CorpusGraphSchemaProfileRegistry.knowledge_relation_catalog(
            self.profile_registry,
            predecessor_profile.profile_id,
            expected_profile_hash=predecessor["profile_hash"],
            expected_catalog_hash=predecessor["profile_relation_catalog_hash"],
        )
        proposed = proposal.proposed_relation_type
        for field_name in (
            "relation_type_id",
            "family_id",
            "definition_scope",
            "namespace",
            "representation_kind",
            "projection_shape",
            "source_type_refs",
            "target_type_refs",
            "source_role_refs",
            "target_role_refs",
            "directed",
        ):
            if getattr(candidate, field_name) != getattr(proposed, field_name):
                raise ValueError(
                    f"candidate proposal definition mismatch: {field_name}"
                )

    @staticmethod
    def _find_relation_type(
        relation_types: tuple[KnowledgeRelationType, ...], relation_type_id: str
    ) -> KnowledgeRelationType | None:
        return next(
            (
                relation_type
                for relation_type in relation_types
                if relation_type.relation_type_id == relation_type_id
            ),
            None,
        )


__all__ = ["DEFAULT_TAXONOMY_REF", "KnowledgeRelationTaxonomyService"]
