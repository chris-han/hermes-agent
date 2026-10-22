from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Literal

from eos.artifact_hashing import content_hash

from pydantic import BaseModel, ConfigDict, Field

from contracts.graph_intent_brief import GraphIntentBrief
from contracts.ontology_building import CompetencyContractV1
from contracts.ontology_seed_query import CompetencyHypothesis, OntologySeedQuery
from contracts.ontology_seed_recommendation import (
    OntologySeedAuthorityV1, OntologySeedFitV1, OntologySeedMatchResultV1,
    SeedRecommendationV1,
)
from contracts.ontology_seeds import OntologyCapabilityProfileV1


_DIMENSION_CAPABILITY_MAP_V1 = {
    "scope": "scoped_requirement",
    "applicability": "scoped_requirement",
    "exception": "exception_structure",
    "qualification": "qualification_structure",
    "precedence": "precedence_structure",
    "time": "temporal_constraint",
    "timewindow": "temporal_constraint",
    "temporal": "temporal_constraint",
    "condition": "conditional_constraint",
    "conditionality": "conditional_constraint",
    "entitydistinction": "entity_distinction",
}


def _capability_dimension(type_id: str) -> str:
    return _DIMENSION_CAPABILITY_MAP_V1.get(type_id.casefold(), type_id)


class OntologySeedMatchPolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy_version: str = "ontology_seed_match_policy.v1"
    retrieval_limit: int = Field(default=20, ge=1)
    recommendation_limit: int = Field(default=3, ge=1)
    minimum_fit: float = Field(default=0.75, ge=0, le=1)
    supporting_minimum_fit: float = Field(default=0.25, ge=0, le=1)
    profile_revision_policy: Literal["highest_revision"] = "highest_revision"

    @property
    def policy_hash(self) -> str:
        return "sha256:" + content_hash(self.model_dump(mode="json"))


class OntologySeedDemandService:
    def derive(
        self, *, graph_intent: GraphIntentBrief,
        organization_id: str, workspace_id: str, build_ref: str,
        created_at: datetime, competency_contract: CompetencyContractV1 | None = None,
    ) -> OntologySeedQuery:
        graph_intent = GraphIntentBrief.from_dict(graph_intent.as_dict())
        if not graph_intent.compiler_eligible:
            raise ValueError("GraphIntent must be accepted or edited before seed demand derivation")
        if (
            graph_intent.source_binding.organization_id != organization_id
            or graph_intent.source_binding.workspace_id != workspace_id
        ):
            raise ValueError("GraphIntent tenant does not match query tenant")
        contract_ref = contract_hash = contract_content_hash = None
        questions = tuple(item.text for item in graph_intent.competency_questions)
        domain = graph_intent.represented_or_governed_subject.type_id
        normative = tuple(dict.fromkeys(
            _capability_dimension(item.type_id)
            for item in graph_intent.material_dimensions
        ))
        if competency_contract is not None:
            competency_contract = CompetencyContractV1.model_validate(
                competency_contract.model_dump(mode="json")
            )
            if (
                competency_contract.organization_id != organization_id
                or competency_contract.workspace_id != workspace_id
            ):
                raise ValueError("competency contract tenant does not match query tenant")
            if (
                competency_contract.graph_intent_brief_ref != graph_intent.graph_intent_id
                or competency_contract.graph_intent_brief_hash != graph_intent.graph_intent_hash
            ):
                raise ValueError("competency contract GraphIntent pins do not match")
            active_questions = tuple(
                item for item in competency_contract.questions
                if item.review_state == "confirmed" and item.criticality != "excluded"
            )
            if not active_questions:
                raise ValueError(
                    "competency contract requires at least one confirmed non-excluded question"
                )
            contract_ref = competency_contract.competency_contract_ref
            contract_hash = competency_contract.competency_contract_hash
            contract_content_hash = "sha256:" + content_hash(
                competency_contract.model_dump(mode="json")
            )
            domain = competency_contract.intended_domain
            questions = tuple(item.question for item in active_questions)
            normative = tuple(dict.fromkeys(
                dimension for item in active_questions
                for dimension in (_capability_dimension(value) for value in item.required_semantic_dimensions)
            ))
        definitions = tuple(dict.fromkeys((
            graph_intent.represented_or_governed_subject.type_id,
            *graph_intent.primary_relationship.source_types,
            *graph_intent.primary_relationship.target_types,
        )))
        relations = (graph_intent.primary_relationship.type_id,)
        preferred = graph_intent.preferred_representation
        if preferred.type_id is not None:
            if preferred.kind == "binary_relation":
                relations = tuple(dict.fromkeys((*relations, preferred.type_id)))
            else:
                definitions = tuple(dict.fromkeys((*definitions, preferred.type_id)))
        return OntologySeedQuery.create(
            organization_id=organization_id, workspace_id=workspace_id, build_ref=build_ref,
            graph_intent_ref=graph_intent.graph_intent_id,
            graph_intent_hash=graph_intent.graph_intent_hash,
            competency_contract_ref=contract_ref, competency_contract_hash=contract_hash,
            competency_contract_content_hash=contract_content_hash,
            competency_hypothesis=CompetencyHypothesis(
                domain=domain,
                intended_use=(graph_intent.document_function.type_id,),
                required_competencies=questions,
                required_definition_families=definitions,
                required_normative_dimensions=normative,
                required_relation_families=relations,
            ), created_at=created_at,
        )


@dataclass(frozen=True)
class _Scored:
    profile: OntologyCapabilityProfileV1
    fit: OntologySeedFitV1


class OntologySeedMatcher:
    def __init__(self, *, policy: OntologySeedMatchPolicyV1 | None = None) -> None:
        supplied = policy or OntologySeedMatchPolicyV1()
        self._policy = OntologySeedMatchPolicyV1.model_validate(supplied.model_dump(mode="json"))

    def match(
        self, *, query: OntologySeedQuery,
        profiles: tuple[OntologyCapabilityProfileV1, ...],
        current_graph_intent_ref: str, current_graph_intent_hash: str,
        current_competency_contract_ref: str | None = None,
        current_competency_contract_hash: str | None = None,
        current_competency_contract_content_hash: str | None = None,
    ) -> OntologySeedMatchResultV1:
        query = OntologySeedQuery.model_validate(query.model_dump(mode="json"))
        if (query.graph_intent_ref, query.graph_intent_hash) != (
            current_graph_intent_ref, current_graph_intent_hash,
        ):
            raise ValueError("query has stale GraphIntent pins")
        if (query.competency_contract_ref, query.competency_contract_hash,
            query.competency_contract_content_hash) != (
            current_competency_contract_ref, current_competency_contract_hash,
            current_competency_contract_content_hash,
        ):
            raise ValueError("query has stale competency contract pins")
        validated = tuple(
            OntologyCapabilityProfileV1.model_validate(item.model_dump(mode="json"))
            for item in profiles
        )
        for profile in validated:
            if (profile.organization_id, profile.workspace_id) != (
                query.organization_id, query.workspace_id,
            ):
                raise ValueError("ontology profile tenant does not match query tenant")
        exact_revisions: dict[tuple[str, str], OntologyCapabilityProfileV1] = {}
        for profile in validated:
            key = (profile.ontology_ref, profile.version)
            current = exact_revisions.get(key)
            if current is None or (profile.revision, profile.profile_ref) > (
                current.revision, current.profile_ref,
            ):
                exact_revisions[key] = profile
        retrieved = sorted(
            exact_revisions.values(),
            key=lambda item: (-self._retrieval_score(query, item), item.ontology_ref, item.version,
                              -item.revision, item.profile_ref)
        )[: self._policy.retrieval_limit]
        scored = [_Scored(item, self._fit(query, item)) for item in retrieved]
        scored.sort(key=lambda item: (-item.fit.coverage_ratio, -item.fit.retrieval_score,
                                      item.profile.ontology_ref, item.profile.version))
        eligible = [item for item in scored if item.fit.coverage_ratio >= self._policy.minimum_fit]
        if not eligible:
            return OntologySeedMatchResultV1(
                status="no_match", query_ref=query.query_ref, query_hash=query.query_hash,
                policy_version=self._policy.policy_version, recommendations=(),
                policy_hash=self._policy.policy_hash,
                reason_codes=("NO_REVIEWED_SEMANTIC_FIT",),
            )
        primary = eligible[0]
        selected: list[tuple[_Scored, str]] = [(primary, "PRIMARY_SEED")]
        covered = self._coverage_keys(primary.fit)
        for candidate in scored:
            if candidate is primary or len(selected) >= self._policy.recommendation_limit:
                continue
            additions = self._coverage_keys(candidate.fit) - covered
            if additions and candidate.fit.coverage_ratio >= self._policy.supporting_minimum_fit:
                selected.append((candidate, "SUPPORTING_SEED"))
                covered.update(additions)
        recommendations = tuple(
            self._recommendation(query, item.profile, item.fit, role) for item, role in selected
        )
        return OntologySeedMatchResultV1(
            status="matched", query_ref=query.query_ref, query_hash=query.query_hash,
            policy_version=self._policy.policy_version, recommendations=recommendations,
            policy_hash=self._policy.policy_hash,
            reason_codes=(),
        )

    def _fit(self, query: OntologySeedQuery, profile: OntologyCapabilityProfileV1) -> OntologySeedFitV1:
        demand = query.competency_hypothesis
        reviewed = profile.profile_provenance.state == "reviewed"
        values = lambda items: {item.value for item in items if reviewed and item.review_state == "reviewed"}
        competencies = values(profile.competency_profile.questions)
        intended_tasks = values(profile.competency_profile.intended_tasks)
        definitions = values(profile.semantic_coverage.definition_families) | values(profile.semantic_coverage.concepts)
        relations = values(profile.semantic_coverage.relation_families)
        represented_tasks = tuple(item for item in demand.intended_use if item in intended_tasks)
        missing_tasks = tuple(item for item in demand.intended_use if item not in intended_tasks)
        domain_supported = reviewed and demand.domain in {
            profile.identity.domain_primary, *profile.identity.domain_secondary,
        }
        represented_c = tuple(item for item in demand.required_competencies if item in competencies)
        represented_d = tuple(item for item in demand.required_definition_families if item in definitions)
        represented_r = tuple(item for item in demand.required_relation_families if item in relations)
        supported_dims: list[str] = []
        unsupported_dims: list[str] = []
        capabilities = profile.representation_capabilities
        for dimension in demand.required_normative_dimensions:
            claim = getattr(capabilities, dimension, None)
            if reviewed and claim is not None and claim.review_state == "reviewed" and claim.status == "supported":
                supported_dims.append(dimension)
            else:
                unsupported_dims.append(dimension)
        missing_c = tuple(item for item in demand.required_competencies if item not in competencies)
        missing_d = tuple(item for item in demand.required_definition_families if item not in definitions)
        missing_r = tuple(item for item in demand.required_relation_families if item not in relations)
        required_count = sum(map(len, (
            demand.intended_use, demand.required_competencies, demand.required_definition_families,
            demand.required_relation_families, demand.required_normative_dimensions,
        ))) + 1
        represented_count = (
            int(domain_supported) + len(represented_tasks) + len(represented_c)
            + len(represented_d) + len(represented_r) + len(supported_dims)
        )
        unresolved = []
        if not reviewed:
            unresolved.append("PROFILE_NOT_REVIEWED")
        if unsupported_dims:
            unresolved.append("UNSUPPORTED_REQUIRED_DIMENSION")
        if any(getattr(capabilities, item, None) is None for item in demand.required_normative_dimensions):
            unresolved.append("UNKNOWN_REQUIRED_DIMENSION")
        return OntologySeedFitV1(
            domain_supported=domain_supported,
            represented_intended_tasks=represented_tasks, missing_intended_tasks=missing_tasks,
            represented_competencies=represented_c, missing_competencies=missing_c,
            represented_definition_families=represented_d, missing_definition_families=missing_d,
            represented_relation_families=represented_r, missing_relation_families=missing_r,
            supported_dimensions=tuple(supported_dims), unsupported_dimensions=tuple(unsupported_dims),
            unresolved_reason_codes=tuple(unresolved),
            coverage_ratio=(represented_count / required_count if required_count else 0),
            retrieval_score=self._retrieval_score(query, profile),
        )

    def _retrieval_score(self, query: OntologySeedQuery, profile: OntologyCapabilityProfileV1) -> int:
        terms = set(re.findall(r"[a-z0-9]+", " ".join((
            query.competency_hypothesis.domain, *query.competency_hypothesis.intended_use,
        )).lower()))
        material = " ".join((profile.identity.name, profile.identity.domain_primary,
                             *profile.identity.domain_secondary)).lower()
        return sum(term in material for term in terms)

    @staticmethod
    def _coverage_keys(fit: OntologySeedFitV1) -> set[tuple[str, str]]:
        return {
            *(("competency", item) for item in fit.represented_competencies),
            *(("task", item) for item in fit.represented_intended_tasks),
            *(("definition", item) for item in fit.represented_definition_families),
            *(("relation", item) for item in fit.represented_relation_families),
            *(("dimension", item) for item in fit.supported_dimensions),
        }

    def _recommendation(self, query, profile, fit, role) -> SeedRecommendationV1:
        return SeedRecommendationV1.create(
            organization_id=query.organization_id, workspace_id=query.workspace_id,
            build_ref=query.build_ref, query_ref=query.query_ref, query_hash=query.query_hash,
            graph_intent_ref=query.graph_intent_ref, graph_intent_hash=query.graph_intent_hash,
            competency_contract_ref=query.competency_contract_ref,
            competency_contract_hash=query.competency_contract_hash,
            competency_contract_content_hash=query.competency_contract_content_hash,
            ontology_ref=profile.ontology_ref, ontology_version=profile.version,
            ontology_source_content_hash=profile.source_content_hash,
            ontology_graph_content_hash=profile.graph_content_hash,
            profile_ref=profile.profile_ref, profile_hash=profile.profile_hash,
            role=role, fit=fit,
            authority=OntologySeedAuthorityV1(
                seed_class=profile.authority.seed_class,
                authority_role=profile.authority.authority_role,
                governance_state=profile.authority.governance_state,
            ), policy_version=self._policy.policy_version, created_at=query.created_at,
            policy_hash=self._policy.policy_hash,
        )
