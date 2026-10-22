"""Semantier-owned synthesis of source facts and Semantica ontology evidence."""

from __future__ import annotations

from typing import Any

from contracts.corpus_graph_schema import CorpusGraphSchemaCandidate
from contracts.evidence_location import EvidenceRef
from contracts.graph_intent_brief import GraphIntentBrief
from contracts.knowledge_relation_taxonomy import KnowledgeRelationType
from services.knowledge_relation_taxonomy_service import (
    KnowledgeRelationTaxonomyService,
)
from .graph_intent_compiler_policy import graph_intent_compiler_copy


class UnadmittedGraphIntentRelation(ValueError):
    """The GraphIntent relation needs a governed catalog successor."""

    def __init__(self, relation_type_id: str) -> None:
        self.relation_type_id = relation_type_id
        super().__init__(f"unadmitted GraphIntent relation: {relation_type_id}")


class SemantierSchemaSynthesizer:
    """Compile product-facing candidate v2 without duplicating ontology induction."""

    def synthesize(
        self,
        *,
        structural_candidate: CorpusGraphSchemaCandidate,
        semantica_evidence: dict[str, Any],
        lineage_id: str,
        source_ref: str,
        canonical_source_ref: str,
        llm_proposal_evidence: dict[str, Any] | None = None,
    ) -> CorpusGraphSchemaCandidate:
        ontology = dict(semantica_evidence.get("ontology") or {})
        inferred_roles = {
            str(item.get("metadata", {}).get("inferred_from") or "")
            for item in ontology.get("classes", [])
        }
        role_evidence = {
            str(item["role_id"]): list(item.get("evidence_refs") or [])
            for item in structural_candidate.semantic_role_proposals
        }

        node_types = tuple(
            {
                **item,
                "graph_layer": "semantic",
                "evidence_basis": (
                    "mixed"
                    if str(item.get("derived_from_role") or "") in inferred_roles
                    else "canonical_source_ir"
                ),
            }
            for item in structural_candidate.proposed_node_types
        )
        relation_types = tuple(
            {
                **item,
                "graph_layer": "semantic",
                "evidence_basis": "semantier_synthesis",
            }
            for item in structural_candidate.proposed_relation_types
        )
        event_types = tuple(
            {**item, "graph_layer": "semantic", "evidence_basis": "mixed"}
            for item in structural_candidate.proposed_event_types
        )
        frame_types = tuple(
            {
                **item,
                "graph_layer": "semantic",
                "evidence_basis": "mixed",
                "semantica_representation": "AssociativeClass",
                "runtime_adapter": "ConstraintFrame",
            }
            for item in structural_candidate.proposed_frame_types
        )
        extraction_plan = self._extraction_plan(structural_candidate)
        synthesis_decisions = self._synthesis_decisions(
            node_types=node_types,
            relation_types=relation_types,
            event_types=event_types,
            frame_types=frame_types,
            extraction_plan=extraction_plan,
            role_evidence=role_evidence,
        )
        llm_binding: dict[str, Any] = {}
        if llm_proposal_evidence is not None:
            proposal = dict(llm_proposal_evidence.get("proposal") or {})
            llm_ontology = dict(proposal.get("ontology") or {})
            deterministic_classes = {
                str(item.get("name") or item.get("id") or "")
                for item in ontology.get("classes", [])
            }
            llm_classes = {
                str(item.get("name") or item.get("id") or "")
                for item in llm_ontology.get("classes", [])
            }
            synthesis_decisions = (
                *synthesis_decisions,
                {
                    "decision_id": "synthesis-optional-llm-disagreement",
                    "target_ref": "optional-llm-ontology-proposal",
                    "decision_kind": "proposal_disagreement",
                    "selected_value": {
                        "deterministic_only_classes": sorted(
                            deterministic_classes - llm_classes
                        ),
                        "llm_only_classes": sorted(
                            llm_classes - deterministic_classes
                        ),
                        "material": False,
                        "disposition": "advisory_only",
                    },
                    "evidence_origin": "optional_llm_proposal",
                    "evidence_refs": [llm_proposal_evidence["evidence_id"]],
                    "rationale": "Optional LLM ontology differences remain proposal evidence and do not override deterministic source observations.",
                },
            )
            llm_binding = {
                "llm_proposal_evidence_ref": llm_proposal_evidence["evidence_id"],
                "llm_proposal_ref": llm_proposal_evidence["proposal_ref"],
                "llm_proposal_hash": llm_proposal_evidence["proposal_hash"],
                "llm_provider": llm_proposal_evidence["provider"],
                "llm_model": llm_proposal_evidence["model"],
            }
        semantica_binding = {
            "semantica_commit": semantica_evidence["semantica_commit"],
            "ontology_candidate_ref": semantica_evidence["ontology_candidate_ref"],
            "ontology_candidate_hash": semantica_evidence["ontology_candidate_hash"],
            "ontology_generation_mode": semantica_evidence["generation_mode"],
            "shacl_report_ref": semantica_evidence["shacl_report_ref"],
            "shacl_report_hash": semantica_evidence["shacl_report_hash"],
            "ontology_evaluation_ref": semantica_evidence[
                "ontology_evaluation_ref"
            ],
            "ontology_evaluation": dict(
                semantica_evidence["ontology_evaluation"]
            ),
            **llm_binding,
        }
        return CorpusGraphSchemaCandidate.create_v2(
            {
                "source_binding": {
                    "lineage_id": lineage_id,
                    "source_ref": source_ref,
                    "source_hash": structural_candidate.source_structure_profile[
                        "document_hash"
                    ],
                    "canonical_source_ref": canonical_source_ref,
                },
                "semantica_schema_evidence": semantica_binding,
                "synthesis_decisions": list(synthesis_decisions),
                "document_understanding": dict(
                    structural_candidate.source_structure_profile
                ),
                "graph_layer_coverage": {
                    "semantic": True,
                    "authority": False,
                    "capability": False,
                    "lineage": True,
                },
                "semantic_role_proposals": [
                    dict(item)
                    for item in structural_candidate.semantic_role_proposals
                ],
                "proposed_node_types": list(node_types),
                "proposed_relation_types": list(relation_types),
                "proposed_event_types": list(event_types),
                "proposed_frame_types": list(frame_types),
                "representation_decisions": [
                    dict(item)
                    for item in structural_candidate.representation_decisions
                ],
                "extraction_plan": list(extraction_plan),
                "assumptions": [],
                "unresolved_ambiguities": [
                    dict(item)
                    for item in structural_candidate.unresolved_ambiguities
                ],
                "clarification_refs": [],
                "source_evidence_refs": list(
                    structural_candidate.source_evidence_refs
                ),
                "predecessor_candidate_ref": None,
                "successor_candidate_refs": [],
                "source_structure_profile": dict(
                    structural_candidate.source_structure_profile
                ),
                "intended_use": structural_candidate.intended_use,
                "extraction_contract_hints": dict(
                    structural_candidate.extraction_contract_hints
                ),
                "competency_questions": [
                    dict(item)
                    for item in structural_candidate.competency_questions
                ],
            }
        )

    def synthesize_v3(
        self,
        *,
        structural_candidate: CorpusGraphSchemaCandidate,
        graph_intent: GraphIntentBrief,
        semantica_evidence: dict[str, Any],
        profile_id: str,
        profile_hash: str,
        catalog_hash: str,
        profile_payload: dict[str, Any],
        graph_intent_proposal_evidence: dict[str, Any],
        adjudicated_evidence_refs: tuple[str, ...] | None = None,
        reviewed_dimension_blocks: dict[str, str] | None = None,
    ) -> CorpusGraphSchemaCandidate:
        """Compile accepted GraphIntent semantics into candidate v3."""

        compiler_copy = graph_intent_compiler_copy()
        intent_evidence = list(
            adjudicated_evidence_refs
            if adjudicated_evidence_refs is not None
            else graph_intent.evidence_refs
        )
        if not intent_evidence:
            raise ValueError("GraphIntent compilation requires source evidence")
        primary = graph_intent.primary_relationship.as_dict()
        representation = graph_intent.preferred_representation.as_dict()
        dimensions = [item.as_dict() for item in graph_intent.material_dimensions]
        source_roles = graph_intent.source_role_interpretation.as_dict()
        if primary["direction"] not in {
            "source_to_target",
            "target_to_source",
        }:
            raise ValueError("GraphIntent primary relationship direction is invalid")

        claim_evidence = dict(
            graph_intent_proposal_evidence.get("claim_evidence") or {}
        )
        source_block_evidence = dict(
            graph_intent_proposal_evidence.get("source_block_evidence") or {}
        )

        def claim_binding(claim_path: str) -> tuple[list[str], list[str]]:
            if reviewed_dimension_blocks is not None and claim_path.startswith("material_dimensions."):
                dimension = dimensions[int(claim_path.rsplit(".", 1)[1])]
                refs = list(dimension["evidence_refs"])
                if not refs or any(ref not in reviewed_dimension_blocks for ref in refs):
                    raise ValueError("reviewed material dimension lacks exact source grounding")
                return refs, list(dict.fromkeys(reviewed_dimension_blocks[ref] for ref in refs))
            accepted_refs = [str(item) for item in claim_evidence.get(claim_path) or []]
            block_refs = [
                str(source_block_evidence[ref]["block_id"])
                for ref in accepted_refs
                if isinstance(source_block_evidence.get(ref), dict)
                and source_block_evidence[ref].get("block_id")
            ]
            if not accepted_refs or len(block_refs) != len(accepted_refs):
                raise ValueError(
                    f"accepted GraphIntent claim lacks exact source grounding: {claim_path}"
                )
            return accepted_refs, list(dict.fromkeys(block_refs))

        def claim_path_for_type(type_id: str) -> str | None:
            if type_id == graph_intent.document_function.type_id:
                return "document_function"
            if type_id == graph_intent.represented_or_governed_subject.type_id:
                return "represented_or_governed_subject"
            for field, values in source_roles.items():
                if type_id in values:
                    return f"source_role_interpretation.{field}"
            return None

        type_ids: list[str] = []
        for value in (
            graph_intent.document_function.type_id,
            graph_intent.represented_or_governed_subject.type_id,
            *primary["source_types"],
            *primary["target_types"],
            *(item["type_id"] for item in dimensions),
            *(item for values in source_roles.values() for item in values),
        ):
            if value not in type_ids:
                type_ids.append(value)
        structural_role_by_type = {
            str(item["type_id"]): str(item["derived_from_role"])
            for item in structural_candidate.proposed_node_types
            if item.get("type_id") and item.get("derived_from_role")
        }
        node_types = tuple(
            {
                "type_id": type_id,
                "graph_layer": "semantic",
                "evidence_basis": "accepted_graph_intent",
                "intent_roles": sorted(
                    {
                        *(
                            ["relationship_source"]
                            if type_id in primary["source_types"]
                            else []
                        ),
                        *(
                            ["relationship_target"]
                            if type_id in primary["target_types"]
                            else []
                        ),
                        *(
                            ["represented_subject"]
                            if type_id
                            == graph_intent.represented_or_governed_subject.type_id
                            else []
                        ),
                    }
                ),
                **(
                    {"derived_from_role": structural_role_by_type[type_id]}
                    if type_id in structural_role_by_type
                    else {}
                ),
            }
            for type_id in type_ids
        )
        relation_types = (
            {
                **primary,
                "graph_layer": "semantic",
                "evidence_basis": "accepted_graph_intent",
                "evidence_refs": intent_evidence,
            },
        )

        representation_type = representation.get("type_id")
        slots = list(dict.fromkeys(item["type_id"] for item in dimensions))
        event_types: tuple[dict[str, Any], ...] = ()
        frame_types: tuple[dict[str, Any], ...] = ()
        if representation["kind"] == "event":
            event_types = (
                {
                    "type_id": representation_type,
                    "required_roles": slots,
                    "graph_layer": "semantic",
                    "evidence_basis": "accepted_graph_intent",
                    "evidence_refs": intent_evidence,
                },
            )
        elif representation["kind"] in {"frame", "associative_class"}:
            frame_types = (
                {
                    "type_id": representation_type,
                    "slots": slots,
                    "graph_layer": "semantic",
                    "evidence_basis": "accepted_graph_intent",
                    "semantica_representation": "AssociativeClass",
                    "runtime_adapter": "ConstraintFrame",
                    "evidence_refs": intent_evidence,
                },
            )
        representation_decisions = (
            {
                "initial_representation": "binary_triple",
                "binary_triple_sufficient": (
                    representation["kind"] == "binary_relation"
                ),
                "selected_representation": representation["kind"],
                "type_id": representation_type,
                "reason": representation["rationale"],
                "evidence_refs": intent_evidence,
            },
        )

        coverage_refs = [primary["type_id"], *(item["type_id"] for item in dimensions)]
        material_dimension_plans: list[dict[str, Any]] = []
        for index, item in enumerate(dimensions):
            type_id = str(item["type_id"])
            plan = {
                "container_ref": "graph-intent:material-dimension:" + type_id,
                "coverage_kind": "material_dimension",
                "target_ref": type_id,
                "strategy": "intent_slot",
                "evidence_refs": list(item["evidence_refs"]),
            }
            if type_id in structural_role_by_type:
                plan.update(
                    {
                        "grounding_kind": "structural_role",
                        "role_id": structural_role_by_type[type_id],
                    }
                )
            else:
                accepted_refs, block_refs = claim_binding(
                    f"material_dimensions.{index}"
                )
                plan.update(
                    {
                        "grounding_kind": "accepted_claim_evidence",
                        "accepted_evidence_refs": accepted_refs,
                        "source_block_refs": block_refs,
                    }
                )
            material_dimension_plans.append(plan)

        endpoint_grounding_plans: list[dict[str, Any]] = []
        dimension_type_ids = {str(item["type_id"]) for item in dimensions}
        for type_id in dict.fromkeys(
            [*primary["source_types"], *primary["target_types"]]
        ):
            type_id = str(type_id)
            if (
                type_id in structural_role_by_type
                or type_id in dimension_type_ids
                or type_id == representation_type
            ):
                continue
            claim_path = claim_path_for_type(type_id)
            if claim_path is None:
                raise ValueError(
                    "accepted GraphIntent relationship endpoint lacks a grounded claim: "
                    + type_id
                )
            accepted_refs, block_refs = claim_binding(claim_path)
            endpoint_grounding_plans.append(
                {
                    "container_ref": "graph-intent:endpoint:" + type_id,
                    "coverage_kind": "relationship_endpoint",
                    "target_ref": type_id,
                    "strategy": "intent_evidence_node",
                    "claim_path": claim_path,
                    "grounding_kind": "accepted_claim_evidence",
                    "accepted_evidence_refs": accepted_refs,
                    "source_block_refs": block_refs,
                    "evidence_refs": accepted_refs,
                }
            )

        representation_plans: list[dict[str, Any]] = []
        if representation["kind"] == "event":
            event_rule = dict(profile_payload.get("event_extraction") or {})
            role_type_map = {
                str(role_id): str(type_id)
                for role_id, type_id in dict(
                    profile_payload.get("node_type_by_role") or {}
                ).items()
            }
            referenced_roles = {
                str(event_rule.get("boundary_role") or ""),
                *map(str, event_rule.get("required_roles") or []),
                *map(str, event_rule.get("inherit_context_roles") or []),
                *map(str, event_rule.get("attach_following_roles") or []),
            }
            if (
                not event_rule
                or "" in referenced_roles
                or not referenced_roles <= set(role_type_map)
                or not set(slots) <= set(role_type_map.values())
            ):
                raise ValueError(
                    "exact registered profile cannot ground GraphIntent event roles"
                )
            representation_plans.append(
                {
                    "container_ref": "graph-intent:representation:" + str(
                        representation_type
                    ),
                    "coverage_kind": "preferred_representation",
                    "target_ref": representation_type,
                    "strategy": "intent_event_constructor",
                    "required_participant_types": slots,
                    "event_extraction_rule": event_rule,
                    "role_type_map": role_type_map,
                    "evidence_refs": intent_evidence,
                }
            )
        elif representation["kind"] in {"frame", "associative_class"}:
            representation_plans.append(
                {
                    "container_ref": "graph-intent:representation:" + str(
                        representation_type
                    ),
                    "coverage_kind": "preferred_representation",
                    "target_ref": representation_type,
                    "strategy": "intent_frame_constructor",
                    "grouping": "document",
                    "required_slot_types": slots,
                    "evidence_refs": intent_evidence,
                }
            )

        competency_requirements = [
            {"output_kind": "relation", "type_id": primary["type_id"]}
        ]
        if representation["kind"] in {"frame", "associative_class", "event"}:
            competency_requirements.append(
                {
                    "output_kind": (
                        "event" if representation["kind"] == "event" else "frame"
                    ),
                    "type_id": representation_type,
                }
            )
        competency_requirements.extend(
            {"output_kind": "node", "type_id": item["type_id"]}
            for item in dimensions
        )
        competency_plans = [
            {
                "container_ref": (
                    "graph-intent:competency-question:" + item.question_id
                ),
                "coverage_kind": "competency_question",
                "target_ref": item.question_id,
                "strategy": "intent_competency_check",
                "required_coverage_refs": coverage_refs,
                "requirements": competency_requirements,
                "evidence_refs": intent_evidence,
            }
            for item in graph_intent.competency_questions
        ]
        hard_negative_plans = [
            {
                "container_ref": "graph-intent:hard-negative:" + item.question_id,
                "coverage_kind": "hard_negative_guard",
                "target_ref": item.question_id,
                "strategy": "intent_distinction_guard",
                "constraint": item.distinction_to_preserve,
                "verification_mode": "assisted_only",
                "evidence_refs": intent_evidence,
            }
            for item in graph_intent.hard_negative_questions
        ]
        extraction_plan = [
            *self._extraction_plan(structural_candidate),
            {
                "container_ref": "graph-intent:primary-relationship",
                "coverage_kind": "primary_relationship",
                "target_ref": primary["type_id"],
                "strategy": "intent_relation",
                "source_types": list(primary["source_types"]),
                "target_types": list(primary["target_types"]),
                "direction": primary["direction"],
                "evidence_refs": intent_evidence,
            },
            *material_dimension_plans,
            *endpoint_grounding_plans,
            *representation_plans,
            *competency_plans,
            *hard_negative_plans,
        ]
        competency_evaluation = [
            {
                "question_id": item.question_id,
                "success_condition": item.success_condition,
                "coverage_refs": coverage_refs,
                "coverage_plan_ref": (
                    "graph-intent:competency-question:" + item.question_id
                ),
                "status": "pending_extraction",
            }
            for item in graph_intent.competency_questions
        ]
        hard_negative_evaluation = [
            {
                "question_id": item.question_id,
                "distinction_to_preserve": item.distinction_to_preserve,
                "coverage_refs": [],
                "coverage_plan_ref": (
                    "graph-intent:hard-negative:" + item.question_id
                ),
                "status": "unresolved",
            }
            for item in graph_intent.hard_negative_questions
        ]
        intent_projection = {
            "document_function": graph_intent.document_function.as_dict(),
            "represented_or_governed_subject": (
                graph_intent.represented_or_governed_subject.as_dict()
            ),
            "primary_relationship": primary,
            "preferred_representation": representation,
            "material_dimensions": dimensions,
            "source_role_interpretation": source_roles,
            "authority_boundary": graph_intent.authority_boundary.as_dict(),
            "competency_questions": [
                item.as_dict() for item in graph_intent.competency_questions
            ],
            "hard_negative_questions": [
                item.as_dict() for item in graph_intent.hard_negative_questions
            ],
            "alternatives_considered": [
                item.as_dict() for item in graph_intent.alternatives_considered
            ],
        }
        semantica_binding = {
            "semantica_commit": semantica_evidence["semantica_commit"],
            "input_hash": semantica_evidence["input_hash"],
            "ontology_candidate_ref": semantica_evidence["ontology_candidate_ref"],
            "ontology_candidate_hash": semantica_evidence["ontology_candidate_hash"],
            "ontology_generation_mode": semantica_evidence["generation_mode"],
            "shacl_report_ref": semantica_evidence["shacl_report_ref"],
            "shacl_report_hash": semantica_evidence["shacl_report_hash"],
            "ontology_evaluation_ref": semantica_evidence[
                "ontology_evaluation_ref"
            ],
            "ontology_evaluation": dict(
                semantica_evidence["ontology_evaluation"]
            ),
        }
        synthesis_decisions = (
            {
                "decision_id": "intent-primary-relationship",
                "target_ref": primary["type_id"],
                "decision_kind": "primary_relationship",
                "selected_value": dict(relation_types[0]),
                "evidence_origin": "accepted_graph_intent",
                "evidence_refs": intent_evidence,
                "rationale": primary["semantic_description"],
            },
            {
                "decision_id": "intent-preferred-representation",
                "target_ref": str(representation_type or primary["type_id"]),
                "decision_kind": "representation",
                "selected_value": dict(representation_decisions[0]),
                "evidence_origin": "accepted_graph_intent",
                "evidence_refs": intent_evidence,
                "rationale": representation["rationale"],
            },
            {
                "decision_id": "intent-extraction-plan-coverage",
                "target_ref": "extraction-plan",
                "decision_kind": "extraction_plan",
                "selected_value": {"coverage_refs": coverage_refs},
                "evidence_origin": "accepted_graph_intent",
                "evidence_refs": intent_evidence,
                "rationale": compiler_copy["extraction_plan_rationale"],
            },
        )
        binding = graph_intent.source_binding
        return CorpusGraphSchemaCandidate.create_v3(
            {
                "source_binding": binding.as_dict(),
                "graph_intent_binding": {
                    "graph_intent_ref": graph_intent.graph_intent_id,
                    "graph_intent_hash": graph_intent.graph_intent_hash,
                    "review_state": graph_intent.review.state,
                    "source_identity_ref": binding.source_identity_ref,
                },
                "compiler_provenance": {
                    "orchestration_skill_ref": "knowledge-graph-building@1.0.0",
                    "compiler_capability_ref": "corpus-graph-schema-induction",
                    "candidate_schema_version": "corpus_graph_schema_candidate.v3",
                    "profile_id": profile_id,
                    "profile_hash": profile_hash,
                    "catalog_hash": catalog_hash,
                    "semantica_commit": semantica_evidence["semantica_commit"],
                    "semantica_input_hash": semantica_evidence["input_hash"],
                },
                "semantica_schema_evidence": semantica_binding,
                "synthesis_decisions": list(synthesis_decisions),
                "document_understanding": {
                    "schema_version": "graph_intent_semantic_projection.v1",
                    "graph_intent": intent_projection,
                    "structural_evidence_summary": dict(
                        structural_candidate.source_structure_profile
                    ),
                },
                "graph_layer_coverage": {
                    "semantic": True,
                    "authority": False,
                    "capability": False,
                    "lineage": True,
                },
                "semantic_role_proposals": [
                    dict(item) for item in structural_candidate.semantic_role_proposals
                ],
                "proposed_node_types": list(node_types),
                "proposed_relation_types": list(relation_types),
                "proposed_event_types": list(event_types),
                "proposed_frame_types": list(frame_types),
                "representation_decisions": list(representation_decisions),
                "extraction_plan": extraction_plan,
                "competency_evaluation": competency_evaluation,
                "hard_negative_evaluation": hard_negative_evaluation,
                "assumptions": [
                    {"assumption_id": f"intent-assumption-{index}", "text": item}
                    for index, item in enumerate(graph_intent.assumptions)
                ],
                "unresolved_ambiguities": [
                    item.as_dict() for item in graph_intent.unresolved_questions
                ],
                "clarification_refs": list(graph_intent.clarification_refs),
                "source_evidence_refs": intent_evidence,
                "predecessor_candidate_ref": None,
                "successor_candidate_refs": [],
                "source_structure_profile": dict(
                    structural_candidate.source_structure_profile
                ),
                "extraction_contract_hints": {
                    "primary_relationship": primary,
                    "material_dimensions": dimensions,
                    "source_role_interpretation": source_roles,
                },
                "competency_questions": [
                    item.as_dict() for item in graph_intent.competency_questions
                ],
            }
        )

    def bind_admitted_relations_v4(
        self,
        *,
        candidate_v3: CorpusGraphSchemaCandidate,
        graph_intent: GraphIntentBrief,
        taxonomy_service: KnowledgeRelationTaxonomyService,
        taxonomy_ref: str,
        taxonomy_hash: str,
        relation_evidence_refs: tuple[EvidenceRef, ...],
        exemplar_gate: dict[str, Any],
        proposal_origin: str,
        explicit_relation_type: KnowledgeRelationType | None = None,
        profile_ref: str | None = None,
        profile_hash: str | None = None,
        profile_relation_catalog_hash: str | None = None,
        profile_relation_type_proposal_ref: str | None = None,
        profile_relation_type_proposal_hash: str | None = None,
        profile_relation_admission_binding: tuple[str, str] | None = None,
    ) -> CorpusGraphSchemaCandidate:
        """Bind GraphIntent meaning to existing admitted types after exemplar checks."""

        if candidate_v3.schema_version != "corpus_graph_schema_candidate.v3":
            raise ValueError("taxonomy binding requires Candidate v3 input")
        if type(taxonomy_service) is not KnowledgeRelationTaxonomyService:
            raise ValueError("Candidate v4 requires the exact governed taxonomy service")
        if not graph_intent.compiler_eligible:
            raise ValueError("taxonomy binding requires accepted or edited GraphIntent")
        if candidate_v3.graph_intent_binding != {
            "graph_intent_ref": graph_intent.graph_intent_id,
            "graph_intent_hash": graph_intent.graph_intent_hash,
            "review_state": graph_intent.review.state,
            "source_identity_ref": graph_intent.source_binding.source_identity_ref,
        }:
            raise ValueError("Candidate v3 does not bind the exact GraphIntent")
        expected_projection = {
            "document_function": graph_intent.document_function.as_dict(),
            "represented_or_governed_subject": (
                graph_intent.represented_or_governed_subject.as_dict()
            ),
            "primary_relationship": graph_intent.primary_relationship.as_dict(),
            "preferred_representation": (
                graph_intent.preferred_representation.as_dict()
            ),
            "material_dimensions": [
                item.as_dict() for item in graph_intent.material_dimensions
            ],
            "source_role_interpretation": (
                graph_intent.source_role_interpretation.as_dict()
            ),
            "authority_boundary": graph_intent.authority_boundary.as_dict(),
            "competency_questions": [
                item.as_dict() for item in graph_intent.competency_questions
            ],
            "hard_negative_questions": [
                item.as_dict() for item in graph_intent.hard_negative_questions
            ],
            "alternatives_considered": [
                item.as_dict() for item in graph_intent.alternatives_considered
            ],
        }
        if (
            candidate_v3.document_understanding.get("graph_intent")
            != expected_projection
        ):
            raise ValueError("Candidate v3 GraphIntent projection mismatch")
        if (
            exemplar_gate.get("positive_count") != 32
            or exemplar_gate.get("hard_negative_count") != 6
            or exemplar_gate.get("material_mismatches") != []
        ):
            raise ValueError("accepted exemplar gate must pass before Candidate v4")
        if not relation_evidence_refs:
            raise ValueError("Candidate v4 relation binding requires EvidenceRefs")

        content = candidate_v3.as_dict()
        content.pop("candidate_id")
        content.pop("candidate_hash")
        if explicit_relation_type is not None:
            primary = graph_intent.primary_relationship
            admitted_profile_successor = (
                explicit_relation_type.definition_scope == "profile_defined"
                and explicit_relation_type.lifecycle_state == "admitted"
                and bool(profile_ref)
                and bool(profile_hash)
                and bool(profile_relation_catalog_hash)
                and bool(profile_relation_type_proposal_ref)
                and bool(profile_relation_type_proposal_hash)
            )
            mapped_profile_relation = (
                profile_relation_admission_binding
                == (primary.type_id, explicit_relation_type.relation_type_id)
                and explicit_relation_type.definition_scope == "profile_defined"
                and explicit_relation_type.lifecycle_state == "admitted"
                and bool(profile_ref)
                and bool(profile_hash)
                and bool(profile_relation_catalog_hash)
                and graph_intent.preferred_representation.kind == "frame"
                and primary.source_types
                == (graph_intent.preferred_representation.type_id,)
                and primary.direction == "source_to_target"
            )
            if (
                not (admitted_profile_successor or mapped_profile_relation)
                and explicit_relation_type.description != primary.semantic_description
            ):
                raise ValueError(
                    "relation_type_proposal description mismatches GraphIntent"
                )
            if len(explicit_relation_type.source_type_refs) != len(
                primary.source_types
            ) or len(explicit_relation_type.target_type_refs) != len(
                primary.target_types
            ):
                raise ValueError(
                    "relation_type_proposal endpoint mapping mismatches GraphIntent"
                )
            replacements = {
                primary.type_id: explicit_relation_type.relation_type_id,
                **dict(
                    zip(
                        primary.source_types,
                        explicit_relation_type.source_type_refs,
                        strict=True,
                    )
                ),
                **dict(
                    zip(
                        primary.target_types,
                        explicit_relation_type.target_type_refs,
                        strict=True,
                    )
                ),
            }

            def replace_bound_refs(value: Any) -> Any:
                if isinstance(value, dict):
                    return {
                        key: replace_bound_refs(item) for key, item in value.items()
                    }
                if isinstance(value, list):
                    return [replace_bound_refs(item) for item in value]
                if isinstance(value, str):
                    return replacements.get(value, value)
                return value

            content = replace_bound_refs(content)
        content["compiler_provenance"] = {
            **content["compiler_provenance"],
            "candidate_schema_version": "corpus_graph_schema_candidate.v4",
            "taxonomy_ref": taxonomy_ref,
            "taxonomy_hash": taxonomy_hash,
            "profile_relation_catalog_hash": (
                profile_relation_catalog_hash
                if explicit_relation_type is not None
                else None
            ),
        }
        bound_relations: list[dict[str, Any]] = []
        for relation in content["proposed_relation_types"]:
            relation_type_id = str(relation.get("type_id") or "")
            try:
                admission = KnowledgeRelationTaxonomyService.resolve_relation_type(
                    taxonomy_service,
                    relation_type_id=relation_type_id,
                    taxonomy_ref=taxonomy_ref,
                    taxonomy_hash=taxonomy_hash,
                    profile_ref=(
                        profile_ref if explicit_relation_type is not None else None
                    ),
                    profile_hash=(
                        profile_hash if explicit_relation_type is not None else None
                    ),
                    profile_relation_catalog_hash=(
                        profile_relation_catalog_hash
                        if explicit_relation_type is not None
                        else None
                    ),
                    for_new_write=True,
                    proposal_origin=proposal_origin,
                )
            except ValueError as exc:
                if "unknown admitted relation_type_id" in str(exc):
                    raise UnadmittedGraphIntentRelation(relation_type_id) from exc
                raise
            admitted = admission.relation_type
            primary = graph_intent.primary_relationship
            if (
                primary.source_role_refs != admitted.source_role_refs
                or primary.target_role_refs != admitted.target_role_refs
            ):
                raise ValueError(
                    "GraphIntent endpoint role observations must exactly match "
                    "the admitted relation type constraints"
                )
            if explicit_relation_type is not None:
                proposed_definition = explicit_relation_type.model_dump(
                    mode="json", by_alias=True
                )
                admitted_definition = admitted.model_dump(mode="json", by_alias=True)
                proposed_definition["lifecycleState"] = "admitted"
                if admitted_definition != proposed_definition:
                    raise ValueError(
                        "admitted successor relation differs from the governed proposal"
                    )
            expected_representation = {
                "binary_relation": "binary_edge",
                "frame": "frame_role",
                "associative_class": "frame_role",
                "event": "frame_role",
            }[graph_intent.preferred_representation.kind]
            if admitted.representation_kind != expected_representation:
                raise ValueError(
                    "admitted relation representation is incompatible with GraphIntent"
                )
            relation_type_candidate = {
                        "relation_type_id": admitted.relation_type_id,
                        "family_id": admitted.family_id,
                        "definition_scope": admitted.definition_scope,
                        "namespace": admitted.namespace,
                        "representation_kind": admitted.representation_kind,
                        "projection_shape": admitted.projection_shape,
                        "source_type_refs": list(admitted.source_type_refs),
                        "target_type_refs": list(admitted.target_type_refs),
                        "source_role_refs": list(admitted.source_role_refs),
                        "target_role_refs": list(admitted.target_role_refs),
                        "directed": admitted.directed,
                        "evidence_refs": [
                            item.model_dump(mode="json", by_alias=True)
                            for item in relation_evidence_refs
                        ],
                        "taxonomy_ref": admission.taxonomy_ref,
                        "taxonomy_hash": admission.taxonomy_hash,
                        "profile_ref": admission.profile_ref,
                        "profile_hash": admission.profile_hash,
                        "profile_relation_catalog_hash": (
                            admission.profile_relation_catalog_hash
                        ),
                    }
            if profile_relation_type_proposal_ref is not None:
                relation_type_candidate.update(
                    {
                        "profile_relation_type_proposal_ref": (
                            profile_relation_type_proposal_ref
                        ),
                        "profile_relation_type_proposal_hash": (
                            profile_relation_type_proposal_hash
                        ),
                    }
                )
            bound_relations.append(
                {
                    **relation,
                    "relation_type_candidate": relation_type_candidate,
                }
            )
        content["proposed_relation_types"] = bound_relations
        content["synthesis_decisions"] = [
            *content["synthesis_decisions"],
            {
                "decision_id": "accepted-witness-exemplar-gate",
                "target_ref": "schema-decision-gate-1",
                "decision_kind": "exemplar_reconstruction",
                "selected_value": dict(exemplar_gate),
                "evidence_origin": "accepted_source_semantic_witnesses",
                "evidence_refs": [
                    exemplar_gate["positive_manifest_ref"],
                    exemplar_gate["hard_negative_manifest_ref"],
                ],
                "rationale": (
                    "Accepted positive exemplars and hard negatives were "
                    "reconstructed before relation admission."
                ),
            },
        ]
        return CorpusGraphSchemaCandidate.create_v4(content)

    @staticmethod
    def semantic_network(
        structural_candidate: CorpusGraphSchemaCandidate,
        graph_intent: GraphIntentBrief | None = None,
    ) -> dict[str, Any]:
        """Project deterministic source-role observations for Semantica induction."""

        entities: list[dict[str, Any]] = []
        for role in structural_candidate.semantic_role_proposals:
            role_id = str(role["role_id"])
            for index, evidence_ref in enumerate(role.get("evidence_refs") or []):
                entities.append(
                    {
                        "id": f"{role_id}-{index}",
                        "name": str(evidence_ref),
                        "type": role_id,
                        "source_evidence_ref": str(evidence_ref),
                    }
                )
        if not entities:
            raise ValueError("source structure produced no semantic network entities")
        entity_by_type: dict[str, str] = {}
        for entity in entities:
            entity_by_type.setdefault(str(entity["type"]), str(entity["id"]))

        relationships: list[dict[str, Any]] = []
        if graph_intent is not None:
            primary = graph_intent.primary_relationship
            if primary.direction not in {
                "source_to_target",
                "target_to_source",
            }:
                raise ValueError(
                    "GraphIntent primary relationship direction is invalid"
                )
            evidence_refs = list(graph_intent.evidence_refs)
            for type_id in (*primary.source_types, *primary.target_types):
                if type_id not in entity_by_type:
                    entity_id = f"intent-type-{type_id}"
                    entities.append(
                        {
                            "id": entity_id,
                            "name": type_id,
                            "type": type_id,
                            "source_evidence_refs": evidence_refs,
                        }
                    )
                    entity_by_type[type_id] = entity_id
            for declared_source in primary.source_types:
                for declared_target in primary.target_types:
                    source_type, target_type = declared_source, declared_target
                    if primary.direction == "target_to_source":
                        source_type, target_type = target_type, source_type
                    relationships.append(
                        {
                            "source": entity_by_type[source_type],
                            "target": entity_by_type[target_type],
                            "type": primary.type_id,
                            "direction": primary.direction,
                            "declared_source_type": declared_source,
                            "declared_target_type": declared_target,
                            "source_evidence_refs": evidence_refs,
                        }
                    )
        else:
            role_entities = {
                str(role["role_id"]): str(
                    next(
                        entity["id"]
                        for entity in entities
                        if entity["type"] == role["role_id"]
                    )
                )
                for role in structural_candidate.semantic_role_proposals
                if role.get("evidence_refs")
            }
            for relation in structural_candidate.proposed_relation_types:
                source_key = str(
                    relation.get("source_role")
                    or relation.get("source_type")
                    or ""
                )
                target_key = str(
                    relation.get("target_role")
                    or relation.get("target_type")
                    or ""
                )
                source = role_entities.get(source_key) or entity_by_type.get(source_key)
                target = role_entities.get(target_key) or entity_by_type.get(target_key)
                if source is not None and target is not None:
                    relationships.append(
                        {
                            "source": source,
                            "target": target,
                            "type": str(relation["type_id"]),
                            "source_evidence_refs": sorted(
                                {
                                    *next(
                                        entity.get("source_evidence_refs", [
                                            entity.get("source_evidence_ref")
                                        ])
                                        for entity in entities
                                        if entity["id"] == source
                                    ),
                                    *next(
                                        entity.get("source_evidence_refs", [
                                            entity.get("source_evidence_ref")
                                        ])
                                        for entity in entities
                                        if entity["id"] == target
                                    ),
                                }
                                - {None}
                            ),
                        }
                    )
        return {"entities": entities, "relationships": relationships}

    @staticmethod
    def _extraction_plan(
        candidate: CorpusGraphSchemaCandidate,
    ) -> tuple[dict[str, Any], ...]:
        plans: list[dict[str, Any]] = []
        for role in candidate.semantic_role_proposals:
            role_id = str(role["role_id"])
            selector = dict(role.get("source_selector") or {})
            plans.append(
                {
                    "container_ref": f"source-role:{role_id}",
                    "role_id": role_id,
                    "selector": selector,
                    "strategy": (
                        "table_row"
                        if selector.get("kind") in {"table_column", "combined"}
                        else "paragraph_pattern"
                    ),
                    "evidence_refs": list(role.get("evidence_refs") or []),
                }
            )
        return tuple(plans)

    @staticmethod
    def _synthesis_decisions(
        *,
        node_types: tuple[dict[str, Any], ...],
        relation_types: tuple[dict[str, Any], ...],
        event_types: tuple[dict[str, Any], ...],
        frame_types: tuple[dict[str, Any], ...],
        extraction_plan: tuple[dict[str, Any], ...],
        role_evidence: dict[str, list[str]],
    ) -> tuple[dict[str, Any], ...]:
        decisions: list[dict[str, Any]] = []
        for kind, items in (
            ("node_type", node_types),
            ("relation_type", relation_types),
            ("event_type", event_types),
            ("frame_type", frame_types),
        ):
            for item in items:
                target = str(item.get("type_id"))
                role_id = str(
                    item.get("derived_from_role")
                    or item.get("target_role")
                    or ""
                )
                decisions.append(
                    {
                        "decision_id": f"synthesis-{kind}-{target}",
                        "target_ref": target,
                        "decision_kind": kind,
                        "selected_value": dict(item),
                        "evidence_origin": (
                            "mixed"
                            if item.get("evidence_basis") == "mixed"
                            else "semantier_synthesis"
                        ),
                        "evidence_refs": role_evidence.get(role_id, []),
                        "rationale": (
                            "Normalized source structure and pinned Semantica ontology "
                            "evidence into the product-facing schema contract."
                        ),
                    }
                )
        decisions.append(
            {
                "decision_id": "synthesis-extraction-plan",
                "target_ref": "extraction-plan",
                "decision_kind": "extraction_plan",
                "selected_value": {"containers": len(extraction_plan)},
                "evidence_origin": "canonical_source_ir",
                "evidence_refs": sorted(
                    {
                        ref
                        for item in extraction_plan
                        for ref in item.get("evidence_refs", [])
                    }
                ),
                "rationale": "Compiled one deterministic plan per observed source role.",
            }
        )
        return tuple(decisions)


__all__ = ["SemantierSchemaSynthesizer"]
