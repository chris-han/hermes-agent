"""Knowledge Builder extraction orchestration shared by API, CLI, and tests."""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from contracts.knowledge_feature_pack import (
    LearningEventEnvelope,
    LearningEventType,
    MachineProposal,
    content_hash,
)
from contracts.knowledge_graph_phase1 import (
    CandidateEdge,
    CandidateNode,
    CandidateNodeType,
    CandidateRelationType,
    CandidateRule,
    SensitiveTermCandidateGraph,
)
from contracts.canonical_source_ir import CanonicalSourceDocument, ParserTrace, SourceBlock, SourceLocator, SourceMap, SourceMapping, Transformation, content_hash as ir_content_hash
from contracts.evidence_location import SourceIdentity, EvidenceRef
from contracts.knowledge_reasoning_phase1 import SemanticOccurrence, SourceAlignment, SourceAlignmentStatus
from knowledge_extraction.adapters.base import SourceExtractionAdapter
from knowledge_extraction.adapters.langextract_adapter import ExtractionProviderUnavailable, LangExtractAdapter
from knowledge_extraction.adapters.legacy_adapter import LegacyExtractionAdapter
from knowledge_extraction.assertion_grouper import group_occurrences
from knowledge_extraction.contracts import SourceExtractionRequest
from storage.knowledge_builder_store import KnowledgeBuilderStore
from storage.knowledge_extraction_store import KnowledgeExtractionStore
from storage.human_grounding_store import HumanGroundingStore
from semantier.knowledge_feature_packs.semantica_adapter import (
    SEMANTICA_COMMIT,
    InProcessSemanticaAdapter,
    SemanticaCandidateGroundingError,
    SemanticaServiceAdapter,
    build_semantica_candidate_graph,
    SemanticaCandidateGraphResult,
)
from semantier.knowledge_feature_packs.relationship_extraction import RelationshipProposer
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_for_block
from semantier.knowledge_feature_packs.reference_concept_builder import build_reference_concepts
from knowledge_extraction.quality_gates_w4 import (
    BoundaryQualityGateValidator,
    RoleClassifier,
    determine_projection_eligibility,
)


def _load_profile_payload(profile: str) -> dict[str, Any]:
    profile_path = Path(__file__).parent / "profiles" / profile / "profile.json"
    return json.loads(profile_path.read_text(encoding="utf-8"))


def _apply_boundary_quality_gates(
    graph: Any, profile_payload: dict[str, Any],
) -> tuple[Any, tuple[dict[str, Any], ...]]:
    """W4: validate each candidate node before reference projection.

    Returns ``(filtered_graph, boundary_validation)`` where:

    - ``filtered_graph`` retains only nodes that pass every hard gate
      (AD-5, IC-3).  ``needs_boundary_review`` nodes stay in the candidate
      graph but are excluded from the reference-concept projection that
      drives ``build_reference_concepts``.
    - ``boundary_validation`` is a tuple of per-node diagnostic records
      describing the gate outcome for curator review.
    """
    from contracts.knowledge_graph_phase1 import SensitiveTermCandidateGraph
    from semantier.knowledge_feature_packs.reference_concept_builder import (
        REFERENCE_NODE_TYPES,
    )

    validator = BoundaryQualityGateValidator(profile_payload)
    classifier = RoleClassifier(profile_payload)

    diagnostics: list[dict[str, Any]] = []
    kept_nodes: list[Any] = []
    kept_node_ids: set[str] = set()

    for node in graph.nodes:
        if node.node_type not in REFERENCE_NODE_TYPES:
            kept_nodes.append(node)
            kept_node_ids.add(node.node_id)
            continue

        anchor = node.source_anchors[0] if node.source_anchors else None
        if anchor is None:
            diagnostics.append({
                "node_id": node.node_id,
                "canonical_label": node.canonical_label,
                "gate_outcome": "reject",
                "reasons": ["missing_source_anchor"],
            })
            continue

        classification = classifier.classify(
            candidate_span_id=node.node_id,
            exact_text=node.canonical_label,
            source_role="canonical_sensitive_term",
            native_entities=[],
        )

        validation = validator.validate(
            candidate_span_id=node.node_id,
            exact_text=node.canonical_label,
            start_offset=int(getattr(anchor, "start_offset", 0) or 0),
            end_offset=int(getattr(anchor, "end_offset", 0) or 0),
            source_identity=str(getattr(anchor, "document_hash", "") or ""),
            semantic_role=str(
                classification.get("semantic_role")
                or "canonical_sensitive_term"
            ),
            native_entity_count=0,
            fallback_method=None,
        )
        eligibility = determine_projection_eligibility(validation)
        gate_outcome = validation.gate_outcome
        failing_gates = [
            name for name, gate in validation.gate_results.items()
            if not gate.passed and gate.severity == "hard"
        ]
        warning_gates = [
            name for name, gate in validation.gate_results.items()
            if not gate.passed and gate.severity == "warn"
        ]

        diagnostics.append({
            "node_id": node.node_id,
            "canonical_label": node.canonical_label,
            "semantic_role": classification.get("semantic_role"),
            "gate_outcome": gate_outcome,
            "review_status": eligibility.get("review_status"),
            "reason": eligibility.get("reason"),
            "failing_gates": failing_gates,
            "warning_gates": warning_gates,
        })

        if gate_outcome == "pass":
            kept_nodes.append(node)
            kept_node_ids.add(node.node_id)

    kept_edges = [
        edge for edge in graph.edges
        if edge.source_node_id in kept_node_ids and edge.target_node_id in kept_node_ids
    ]

    # pydantic models expose ``model_copy``; ``dataclasses.replace`` only
    # works on actual dataclasses.  Keep one immutable boundary so the
    # caller cannot mutate the original graph by accident.
    filtered_graph = graph.model_copy(update={
        "nodes": tuple(kept_nodes),
        "edges": tuple(kept_edges),
    })
    return filtered_graph, tuple(diagnostics)


class _CandidateOnlyBackend:
    def materialize(self, _release: dict[str, Any]) -> None:
        raise RuntimeError("candidate-only backend cannot materialize releases")


def _canonical_elements_from_governed_source(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Build deterministic paragraph anchors from governed source text."""

    text = str(source.get("source_text") or "")
    if not text:
        return []
    source_hash = str(source.get("source_hash") or content_hash(text))
    elements: list[dict[str, Any]] = []
    offset = 0
    for ordinal, line in enumerate(text.splitlines(keepends=True)):
        exact = line.rstrip("\r\n")
        start = offset
        end = start + len(exact)
        offset += len(line)
        if not exact.strip():
            continue
        anchor_id = f"{source['source_id']}:paragraph:{ordinal}"
        anchor = {
            "anchor_id": anchor_id,
            "document_hash": source_hash,
            "document_part": "paragraph",
            "paragraph_index": ordinal,
            "start_offset": start,
            "end_offset": end,
            "exact_text": exact,
            "local_text_hash": content_hash(exact),
        }
        elements.append({
            **anchor,
            "text": exact,
            "anchor": dict(anchor),
        })
    return elements


def _with_reference_list_relationships(
    graph: SensitiveTermCandidateGraph,
) -> SensitiveTermCandidateGraph:
    reference_nodes = [
        node for node in graph.nodes
        if node.node_type in {
            CandidateNodeType.TERM,
            CandidateNodeType.CONCEPT,
            CandidateNodeType.ALIAS,
            CandidateNodeType.CATEGORY,
        }
    ]
    if not reference_nodes or graph.edges:
        return graph
    anchors_by_id = {
        anchor.anchor_id: anchor
        for node in reference_nodes
        for anchor in node.source_anchors
    }
    anchors = tuple(anchors_by_id[key] for key in sorted(anchors_by_id))
    category_digest = content_hash({
        'candidate_graph_id': graph.candidate_graph_id,
        'category': 'reference_sensitive_word_list',
    })[7:23]
    category_node_id = f"reference_category_{category_digest}"
    category = CandidateNode(
        node_id=category_node_id,
        node_type=CandidateNodeType.CATEGORY,
        canonical_label="reference_sensitive_word_list",
        display_name_zh="参考敏感词清单",
        normalized_value="reference_sensitive_word_list",
        rationale="Deterministic reference-list grouping for source-list extraction.",
        source_anchors=anchors,
        extraction_run_id=graph.extraction_run_id,
        machine_proposal_id=graph.machine_proposal.proposal_id,
        machine_confidence=1.0,
    )
    existing_rule_ids = {rule.rule_id for rule in graph.rules}
    edges: list[CandidateEdge] = []
    rules: list[CandidateRule] = list(graph.rules)
    for node in reference_nodes:
        edge_digest = content_hash({
            'source': node.node_id,
            'target': category_node_id,
            'relation': 'in_category',
        })[7:23]
        edge_id = f"edge_reference_list_{edge_digest}"
        edges.append(CandidateEdge(
            edge_id=edge_id,
            source_node_id=node.node_id,
            target_node_id=category_node_id,
            relation_type=CandidateRelationType.IN_CATEGORY,
            rationale="Reference term belongs to the source-list category.",
            source_anchors=node.source_anchors,
            extraction_run_id=graph.extraction_run_id,
            machine_proposal_id=graph.machine_proposal.proposal_id,
            machine_confidence=node.machine_confidence,
        ))
        rule_digest = content_hash({
            'candidate_graph_id': graph.candidate_graph_id,
            'node_id': node.node_id,
        })[7:23]
        rule_id = f"rule_reference_term_{rule_digest}"
        if rule_id not in existing_rule_ids:
            rules.append(CandidateRule(
                rule_id=rule_id,
                family_id=1,
                condition_node_ids=(node.node_id,),
                reason_code="reference_sensitive_word_list_term",
                source_anchors=node.source_anchors,
                extraction_run_id=graph.extraction_run_id,
                machine_proposal_id=graph.machine_proposal.proposal_id,
            ))
            existing_rule_ids.add(rule_id)
    return graph.model_copy(update={
        "nodes": (*graph.nodes, category),
        "edges": tuple(edges),
        "rules": tuple(rules),
    })


class KnowledgeExtractionService:
    def __init__(
        self, *, db_path: str | Path, adapter_factory: Callable[[str, Callable[[str, str], str]], SourceExtractionAdapter] | None = None,
        artifact_writer: Callable[[str, str], str] | None = None,
        relationship_proposer: RelationshipProposer | None = None,
        relationship_feedback_refs: list[str] | tuple[str, ...] = (),
    ) -> None:
        self._db_path = Path(db_path)
        self._builder = KnowledgeBuilderStore(db_path=self._db_path)
        self._store = KnowledgeExtractionStore(self._db_path)
        self._artifact_writer = artifact_writer or governed_artifact_writer
        self._adapter_factory = adapter_factory or self._default_adapter
        self._relationship_proposer = relationship_proposer
        self._relationship_feedback_refs = tuple(relationship_feedback_refs)

    def capability(self, provider: str) -> dict[str, str]:
        return self._adapter_factory(provider, self._artifact_writer).capability()  # type: ignore[attr-defined]

    def start(
        self, *, organization_id: str, workspace_id: str | None, discovery_run_id: str,
        document_id: str, provider: str = "semantica", profile: str = "tender_sensitive_v1",
        source_id: str | None = None,
        package_id: str | None = None,
        canonical_ir_ref: str | None = None,
        source_identity_ref: str | None = None,
        parser_trace_id: str | None = None,
        provider_options: dict[str, Any] | None = None,
        source_role: str = "legacy_unknown",
        workflow_kind: str = "legacy_candidate_assertion",
    ) -> dict[str, Any]:
        canonical_elements: list[dict[str, Any]] | None = None
        canonical_ir_document: CanonicalSourceDocument | None = None
        discovery = self._builder.get_discovery_run(discovery_run_id)
        if (discovery is None or discovery["organization_id"] != organization_id or
                discovery.get("workspace_id") != workspace_id):
            raise ValueError("discovery run not found")
        if source_id is not None and discovery["source_id"] != source_id:
            raise ValueError("source_id does not match discovery run")
        source = self._builder.get_source(discovery["source_id"])
        if (source is None or source["organization_id"] != organization_id or
                source.get("workspace_id") != workspace_id):
            raise ValueError("knowledge source not found")
        if package_id:
            scope = self._store.get_knowledge_scope(package_id)
            if (scope is None or scope["organization_id"] != organization_id or
                    scope.get("workspace_id") != workspace_id):
                raise ValueError("tender package not found")
            if scope.get("discovery_run_id") != discovery_run_id:
                raise ValueError("tender package discovery run mismatch")
            package_document = next(
                (item for item in scope["documents"] if item["document_id"] == document_id), None
            )
            if package_document is None:
                raise ValueError("document not found in tender package")
            element_source_ids = {
                str(item.get("source_id")) for item in package_document.get("elements", [])
                if isinstance(item, dict) and item.get("source_id")
            }
            if len(element_source_ids) != 1:
                raise ValueError("tender package document lacks one canonical source reference")
            source = self._builder.get_source(element_source_ids.pop())
            if (source is None or source["organization_id"] != organization_id or
                    source.get("workspace_id") != workspace_id):
                raise ValueError("tender package document source not found")
            if source["source_hash"] != package_document["document_hash"]:
                raise ValueError("tender package document source hash mismatch")
            canonical_elements = list(package_document.get("elements") or [])
        if not canonical_elements and canonical_ir_ref:
            # R3-01: thin adapter that converts an IR-backed identity into the
            # canonical_elements shape the Semantica profile expects. The IR is
            # loaded once here; SourceLocator provenance flows through unchanged.
            from semantier.knowledge_feature_packs.canonical_source_ir_adapter import (
                canonical_elements_from_ir, load_canonical_source_ir,
            )
            canonical_ir_document = load_canonical_source_ir(canonical_ir_ref)
            from semantier.knowledge_feature_packs.canonical_source_ir_adapter import resolve_unique_mapping
            locator_hash = (resolve_unique_mapping(canonical_ir_document, canonical_ir_document.blocks[0].block_id).source_hash
                            if canonical_ir_document.blocks else None)
            source_hash = str(source["source_hash"])
            if locator_hash and locator_hash.removeprefix("sha256:") != source_hash.removeprefix("sha256:"):
                raise ValueError("canonical IR source hash does not match knowledge source")
            if source_identity_ref is None:
                source_identity_ref = f"src:ir:{canonical_ir_ref}"
            if parser_trace_id is None:
                parser_trace_id = f"parse:{canonical_ir_document.parser_trace.adapter_version}"
            canonical_elements = canonical_elements_from_ir(canonical_ir_document)
        if not canonical_elements and workflow_kind == "reference_graph_build":
            canonical_elements = _canonical_elements_from_governed_source(source)
        if source_identity_ref is None:
            source_identity_ref = str(source["source_ref"])
        if parser_trace_id is None:
            parser_trace_id = "parse:canonical-input"
        extraction_run_id = f"extract_{uuid4().hex}"
        adapter = self._adapter_factory(provider, self._artifact_writer)
        capability = adapter.capability()  # type: ignore[attr-defined]
        run = self._store.create_extraction_run(
            extraction_run_id=extraction_run_id,
            discovery_run_id=discovery_run_id,
            source_id=source["source_id"],
            lineage_id=source["source_id"],
            organization_id=organization_id,
            workspace_id=workspace_id,
            package_id=package_id,
            document_id=document_id,
            provider=provider,
            provider_version=capability["detail"],
            profile_ref=profile,
            source_identity_ref=source_identity_ref,
            parser_trace_id=parser_trace_id,
            source_role=source_role,
            workflow_kind=workflow_kind,
        )
        if capability["state"] == "unavailable":
            return self._store.fail_extraction_run(
                extraction_run_id, capability["detail"],
                failure_reason="provider_unavailable", provider_ref=provider,
                provider_commit=SEMANTICA_COMMIT if provider == "semantica" else None,
            )
        if provider in {"semantica", "semantica_service"}:
            return self._start_semantica(
                extraction_run_id=extraction_run_id,
                source=source,
                document_id=document_id,
                adapter=adapter,
                provider=provider,
                profile=profile,
                canonical_elements=canonical_elements,
                canonical_ir_document=canonical_ir_document,
                workflow_kind=workflow_kind,
            )
        request = SourceExtractionRequest(
            extraction_run_id=extraction_run_id,
            document_id=document_id,
            canonical_document_ref=str(source["source_ref"]),
            canonical_document_hash=str(source["source_hash"]),
            text=str(source["source_text"]),
            extraction_profile=profile,
            provider_options=provider_options or {},
        )
        try:
            result = adapter.extract(request)
        except ExtractionProviderUnavailable as exc:
            return self._store.fail_extraction_run(
                extraction_run_id, str(exc), failure_reason="provider_unavailable",
                provider_ref=provider,
            )
        proposal = MachineProposal(
            proposal_id=f"proposal_{result.raw_output_hash[7:31]}",
            provider=result.provider,
            model=result.model_ref,
            prompt_version=profile,
            prompt_hash=result.prompt_ref.rsplit(":", 1)[-1],
            parameters={"authority": "proposal_only", "provider_options_hash": content_hash(provider_options or {})},
            run_id=extraction_run_id,
            raw_output_ref=result.raw_output_ref,
            raw_output_hash=result.raw_output_hash,
            confidence=_mean_confidence(result.occurrences),
        )
        assertions = group_occurrences(
            result.occurrences,
            candidate_graph_id=f"candidate_graph_{extraction_run_id}",
            source_document_hash=str(source["source_hash"]),
            machine_proposal=proposal,
        )
        candidate_graph = _candidate_graph(
            assertions=assertions, extraction_run_id=extraction_run_id,
            source_document_hash=str(source["source_hash"]), proposal=proposal,
        )
        grounding_store = HumanGroundingStore(self._db_path, graph_backend=_CandidateOnlyBackend())
        machine_events = tuple(
            LearningEventEnvelope(
                event_id=f"machine_event_{assertion.assertion_id}", case_id=assertion.assertion_id,
                event_type=LearningEventType.MACHINE_PROPOSAL, occurred_at=datetime.now(timezone.utc),
                source_document_hash=assertion.source_document_hash,
                source_anchor_refs=tuple(anchor.anchor_id for anchor in assertion.source_anchors),
                graph_version="KG_v0", rule_version="tender_rules.v1",
                evaluation_corpus_version="tender_sensitive_v1", machine_proposal=proposal,
            ) for assertion in assertions
        )
        try:
            completed = self._store.complete_extraction_run(
            extraction_run_id=extraction_run_id,
            raw_output_ref=result.raw_output_ref,
            raw_output_hash=result.raw_output_hash,
            warnings=result.warnings,
            occurrences=result.occurrences,
                assertions=assertions, machine_events=machine_events, candidate_graph=candidate_graph,
            )
            grounding_store.ensure_materialized_empty_base(candidate_graph.candidate_graph_id)
        except Exception as exc:
            self._store.compensate_failed_completion(
                extraction_run_id, candidate_graph.candidate_graph_id, str(exc)
            )
            raise
        return completed

    def _start_semantica(
        self, *, extraction_run_id: str, source: dict[str, Any], document_id: str,
        adapter: Any, provider: str, profile: str,
        canonical_elements: list[dict[str, Any]] | None,
        canonical_ir_document: CanonicalSourceDocument | None = None,
        workflow_kind: str = "legacy_candidate_assertion",
    ) -> dict[str, Any]:
        try:
            parsed_document = _semantica_parsed_document(
                source=source, document_id=document_id, profile=profile,
                canonical_elements=canonical_elements,
            )
            result = build_semantica_candidate_graph(
                parsed_document=parsed_document,
                extraction_run_id=extraction_run_id,
                adapter=adapter,
            )
            result = _attach_canonical_evidence(result, source=source, canonical_elements=canonical_elements,
                                                organization_id=source["organization_id"],
                                                workspace_id=source.get("workspace_id"),
                                                canonical_ir_document=canonical_ir_document)
            if workflow_kind == "reference_graph_build":
                result = replace(
                    result,
                    graph=_with_reference_list_relationships(result.graph),
                )
        except SemanticaCandidateGroundingError as exc:
            proposal = exc.proposal_result
            return self._store.fail_extraction_run(
                extraction_run_id, str(exc), failure_reason="candidate_grounding_failed",
                provider_ref=proposal.provider_ref, provider_commit=proposal.provider_commit,
                native_artifact_ref=proposal.native_artifact_ref,
                native_artifact_hash=proposal.native_artifact_hash,
            )
        except Exception as exc:
            return self._store.fail_extraction_run(
                extraction_run_id, str(exc), failure_reason="provider_execution_failed",
                provider_ref=provider,
                provider_commit=SEMANTICA_COMMIT if provider == "semantica" else None,
            )
        semantica_occurrences = []
        for index, node in enumerate(result.graph.nodes):
            anchor = node.source_anchors[0]
            occurrence = SemanticOccurrence(
                occurrence_id=f"occurrence_{result.extraction_run_id}_{index}",
                extraction_run_id=result.extraction_run_id, document_id=document_id,
                source_anchor=anchor, extraction_class="entity",
                extraction_attributes={"assertion_type": "entity", "subject_text": node.canonical_label, "candidate_family_id": 1},
                alignment=SourceAlignment(provider="semantica", status=SourceAlignmentStatus.EXACT,
                    provider_start=anchor.start_offset, provider_end=anchor.end_offset),
                machine_confidence=node.machine_confidence,
                model_ref=result.graph.machine_proposal.model,
                prompt_ref=result.graph.machine_proposal.prompt_version,
                raw_output_ref=result.native_artifact_ref, extraction_index=index,
            )
            semantica_occurrences.append(occurrence)
        semantica_assertions = []
        if workflow_kind != "reference_graph_build":
            for index, occurrence in enumerate(semantica_occurrences):
                grouped = group_occurrences(
                    (occurrence,), candidate_graph_id=result.graph.candidate_graph_id,
                    source_document_hash=source["source_hash"], machine_proposal=result.graph.machine_proposal,
                )
                semantica_assertions.extend(
                    item.model_copy(update={"evidence_refs": tuple(result.graph.nodes[index].evidence_refs)})
                    for item in grouped
                )
        grounding_store = HumanGroundingStore(self._db_path, graph_backend=_CandidateOnlyBackend())
        machine_events = tuple(
            LearningEventEnvelope(
                event_id=f"machine_event_{assertion.assertion_id}",
                case_id=assertion.assertion_id,
                event_type=LearningEventType.MACHINE_PROPOSAL,
                occurred_at=datetime.now(timezone.utc),
                source_document_hash=assertion.source_document_hash,
                source_anchor_refs=tuple(anchor.anchor_id for anchor in assertion.source_anchors),
                graph_version="KG_v0",
                rule_version="tender_rules.v1",
                evaluation_corpus_version="tender_sensitive_v1",
                machine_proposal=result.graph.machine_proposal,
            )
            for assertion in semantica_assertions
        )
        # W4: run boundary quality gates BEFORE ``complete_extraction_run`` so
        # the run row is written exactly once and the row carries the W4
        # diagnostics plus the chosen ``run_status``.  This keeps the
        # ``complete_extraction_run`` contract intact for legacy callers and
        # avoids double-write races.
        boundary_validation: tuple[dict[str, Any], ...] = ()
        result_graph = result.graph
        run_status = "completed"
        if workflow_kind == "reference_graph_build":
            profile_payload = _load_profile_payload(profile)
            try:
                result_graph, boundary_validation = _apply_boundary_quality_gates(
                    result.graph, profile_payload
                )
            except Exception:
                # Gate failures must not silently drop projection.  Preserve
                # the unfiltered graph and surface an empty diagnostic so
                # the run still completes.
                result_graph = result.graph
                boundary_validation = ()
            non_pass = [d for d in boundary_validation if d["gate_outcome"] != "pass"]
            if non_pass and not all(d["gate_outcome"] == "reject" for d in non_pass):
                run_status = "completed_with_boundary_review"
        try:
            completed = self._store.complete_extraction_run(
                extraction_run_id=extraction_run_id,
                raw_output_ref=result.native_artifact_ref,
                raw_output_hash=result.native_artifact_hash,
                warnings=(
                    ("boundary_validation_required",)
                    if run_status == "completed_with_boundary_review" else ()
                ),
                occurrences=tuple(semantica_occurrences),
                assertions=tuple(semantica_assertions), machine_events=machine_events,
                candidate_graph=result.graph,
                provider_ref=result.provider_ref, provider_commit=result.provider_commit,
                native_artifact_ref=result.native_artifact_ref,
                native_artifact_hash=result.native_artifact_hash,
                boundary_validation=boundary_validation,
                run_status=run_status,
            )
            if workflow_kind == "reference_graph_build":
                concepts, expressions = build_reference_concepts(result_graph)
                self._store.put_reference_concepts(
                    extraction_run_id=extraction_run_id,
                    source_document_hash=str(source["source_hash"]),
                    candidate_graph_id=result.graph.candidate_graph_id,
                    concepts=concepts,
                    expressions=expressions,
                )
                base_graph_version = grounding_store.ensure_materialized_empty_base(
                    result.graph.candidate_graph_id
                )
                reference_machine_events = tuple(
                    LearningEventEnvelope(
                        event_id=f"machine_event_{concept.concept_id}",
                        case_id=concept.concept_id,
                        event_type=LearningEventType.MACHINE_PROPOSAL,
                        occurred_at=datetime.now(timezone.utc),
                        source_document_hash=str(source["source_hash"]),
                        source_anchor_refs=tuple(concept.source_anchor_refs),
                        graph_version=base_graph_version,
                        rule_version="tender_rules.v1",
                        evaluation_corpus_version="tender_sensitive_v1",
                        machine_proposal=result.graph.machine_proposal,
                    )
                    for concept in concepts
                )
                self._store.append_reference_concept_machine_events(
                    events=reference_machine_events,
                    concept_ids={concept.concept_id for concept in concepts},
                )
            else:
                grounding_store.ensure_materialized_empty_base(
                    result.graph.candidate_graph_id
                )
        except Exception as exc:
            self._store.compensate_failed_completion(
                extraction_run_id, result.graph.candidate_graph_id, str(exc)
            )
            raise
        return completed

    def _default_adapter(self, provider: str, writer: Callable[[str, str], str]) -> SourceExtractionAdapter:
        if provider == "semantica":
            # Production default: pinned Semantica runs in-process. Service,
            # LangExtract, and legacy providers are explicit challengers only;
            # provider failure is persisted and never triggers fallback.
            return InProcessSemanticaAdapter(
                artifact_writer=writer,
                relationship_proposer=self._relationship_proposer,
                relationship_feedback_refs=self._relationship_feedback_refs,
            )  # type: ignore[return-value]
        if provider == "semantica_service":
            service_url = os.environ.get("SEMANTICA_SERVICE_URL", "").strip()
            if not service_url:
                return _UnavailableSemanticaServiceAdapter()  # type: ignore[return-value]
            return SemanticaServiceAdapter(  # type: ignore[return-value]
                base_url=service_url, artifact_writer=writer
            )
        if provider == "legacy":
            return LegacyExtractionAdapter(artifact_writer=writer)
        if provider == "langextract":
            return LangExtractAdapter(artifact_writer=writer)
        raise ValueError(
            "provider must be semantica, semantica_service, langextract, or legacy"
        )


class _UnavailableSemanticaServiceAdapter:
    def capability(self) -> dict[str, str]:
        return {
            "provider": "semantica_service", "state": "unavailable",
            "detail": "SEMANTICA_SERVICE_URL is required",
            "executionMode": "service",
        }


def _semantica_parsed_document(
    *, source: dict[str, Any], document_id: str, profile: str,
    canonical_elements: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    text = str(source["source_text"])
    profile_path = Path(__file__).parent / "profiles" / profile / "profile.json"
    if not profile_path.is_file():
        raise ValueError(f"knowledge extraction profile not found: {profile}")
    profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
    elements = canonical_elements
    if not isinstance(elements, list) or not elements:
        raise ValueError("Semantica extraction requires governed canonical parsed elements")
    normalized = []
    for element in elements:
        if not isinstance(element, dict):
            raise ValueError("canonical parsed element is invalid")
        anchor = element.get("anchor") or {
            key: element.get(key) for key in (
                "anchor_id", "document_hash", "document_part", "paragraph_index",
                "table_index", "row_index", "cell_index", "start_offset", "end_offset",
                "exact_text", "local_text_hash",
            )
        }
        if str(anchor.get("document_hash", "")).removeprefix("sha256:") != str(source["source_hash"]).removeprefix("sha256:"):
            raise ValueError("canonical parsed element document hash mismatch")
        family_id = element.get("family_id")
        family = element.get("family")
        defaults = profile_payload.get("semantica_candidate_defaults") or {}
        if not isinstance(family_id, int):
            family_id = defaults.get("family_id")
        if not family:
            family = defaults.get("family")
        if not isinstance(family_id, int) or not family:
            raise ValueError("canonical parsed element lacks governed family metadata")
        normalized.append({"text": str(element.get("text") or anchor.get("exact_text") or ""),
                           "family_id": family_id, "family": family,
                           "anchor": anchor})
    normalized = _profile_table_term_elements(
        normalized,
        profile_payload=profile_payload,
    )
    normalized = _split_configured_delimiter_terms(
        normalized,
        profile_payload=profile_payload,
    )
    normalized = _segment_candidate_spans(
        normalized,
        profile_payload=profile_payload,
    )
    return {"document_hash": str(source["source_hash"]), "elements": normalized}


def _segment_candidate_spans(
    elements: list[dict[str, Any]], *, profile_payload: dict[str, Any]
) -> list[dict[str, Any]]:
    """W3 wiring: run ``ReferenceConceptCandidateSegmenter`` per remaining element.

    AD-2 precedence order is preserved by running this stage AFTER
    ``_split_configured_delimiter_terms``: structural units and configured
    delimiters happen first, then the segmenter is invoked only on remaining
    whole-element blocks to surface clause / protected-span proposals that
    the configured delimiters could not resolve.

    For each remaining element we synthesize a stable ``block_id`` from its
    anchor id, invoke the segmenter with the default source role
    (``canonical_sensitive_term``), and replace the element with one element
    per validated candidate span. Anchors are recomputed against the
    original element text so every child span preserves exact source
    identity and SourceLocator.

    The segmenter is imported lazily to keep pydantic out of the import
    path for callers that only use the configured_delimiter stage. When
    the segmenter is missing or raises, the element is passed through
    unchanged so the downstream W4 quality gate can flag the row as
    ``needs_boundary_review`` rather than silently promoting a whole-element
    shim (AD-3 / IC-3 step 7).
    """

    if not elements:
        return elements

    segmentation_config = profile_payload.get("segmentation") or {}
    if not segmentation_config:
        return elements

    try:
        from knowledge_extraction.segmentation import ReferenceConceptCandidateSegmenter
    except ImportError:
        return elements

    segmenter = ReferenceConceptCandidateSegmenter(profile_payload)
    source_role = "canonical_sensitive_term"

    expanded: list[dict[str, Any]] = []
    for element in elements:
        text = str(element.get("text") or "")
        anchor = dict(element.get("anchor") or {})
        block_id = str(anchor.get("anchor_id") or f"block-{id(element)}")

        if not text:
            expanded.append(element)
            continue

        if _looks_atomic(text):
            expanded.append(element)
            continue

        base_offset = int(anchor.get("start_offset") or 0)
        try:
            result = segmenter.segment(
                source_block_id=block_id,
                source_identity=str(anchor.get("document_hash") or "unknown"),
                exact_text=text,
                source_role=source_role,
                boundary_method="source_cell",
            )
        except Exception:
            expanded.append(element)
            continue

        if not result.candidate_spans:
            expanded.append(element)
            continue

        for span in result.candidate_spans:
            span_text = str(span.get("exact_text") or "")
            if not span_text:
                continue
            start = int(span.get("start_offset") or 0)
            end = int(span.get("end_offset") or 0)
            if end <= start:
                continue
            absolute_start = base_offset + start
            absolute_end = base_offset + end
            term_anchor = {
                **anchor,
                "anchor_id": f"{block_id}:segmented:{absolute_start}-{absolute_end}",
                "start_offset": absolute_start,
                "end_offset": absolute_end,
                "exact_text": span_text,
                "local_text_hash": content_hash(span_text),
            }
            expanded.append(
                {
                    "text": span_text,
                    "family_id": element.get("family_id"),
                    "family": element.get("family"),
                    "anchor": term_anchor,
                }
            )
    return expanded or elements


def _looks_atomic(text: str) -> bool:
    """Guardrail: skip the segmenter for spans that are already atomic.

    An element is considered already-atomic when it carries no configured
    list delimiter, no clause terminal token, no conjunction token, and
    is short enough that further segmentation would be synthetic. This
    is a guardrail, not a gate: W4 quality gates still classify the
    result deterministically.
    """
    if not text:
        return True
    list_delims = ("、", "；", ";", "，")
    terminals = ("。", "！", "？")
    conjunctions = ("且", "并且", "以及", "或者")
    if any(delim in text for delim in list_delims):
        return False
    if any(tok in text for tok in terminals):
        return False
    if any(tok in text for tok in conjunctions):
        return False
    if len(text) > 80:
        return False
    return True


def _split_configured_delimiter_terms(
    elements: list[dict[str, Any]], *, profile_payload: dict[str, Any]
) -> list[dict[str, Any]]:
    """Split delimiter-separated source cells into atomic candidate spans.

    This is the production path that was still collapsing a whole Chinese term
    list like "大型企业、央企、世界500强、行业龙头" into a single element. The
    profile declares those delimiters explicitly, so the runtime should apply the
    same behavior for non-table cells that represent multi-value sensitive terms.
    """

    delimiter_rules = profile_payload.get("segmentation", {}).get("delimiterRules", [])
    tokens: list[str] = []
    for rule in delimiter_rules:
        token = str(rule.get("token", "")).strip()
        if not token:
            continue
        roles = rule.get("roles", [])
        if "canonical_sensitive_term" in roles or "*" in roles:
            if token not in tokens:
                tokens.append(token)
    if not tokens:
        return elements

    pattern = "|".join(re.escape(token) for token in tokens)
    expanded: list[dict[str, Any]] = []
    for element in elements:
        text = str(element.get("text") or "")
        if not text:
            continue
        if text.count("。") or text.count("！") or text.count("？"):
            expanded.append(element)
            continue
        parts = [part.strip() for part in re.split(pattern, text) if part.strip()]
        if len(parts) <= 1:
            expanded.append(element)
            continue
        anchor = dict(element.get("anchor") or {})
        base_offset = int(anchor.get("start_offset") or 0)
        cursor = 0
        for part in parts:
            index = text.find(part, cursor)
            if index < 0:
                expanded.append(element)
                break
            start = base_offset + index
            end = start + len(part)
            term_anchor = {
                **anchor,
                "anchor_id": f"{anchor.get('anchor_id', 'source-anchor')}:term:{start}-{end}",
                "start_offset": start,
                "end_offset": end,
                "exact_text": part,
                "local_text_hash": content_hash(part),
            }
            expanded.append(
                {
                    "text": part,
                    "family_id": element.get("family_id"),
                    "family": element.get("family"),
                    "anchor": term_anchor,
                }
            )
            cursor = index + len(part)
    return expanded or elements


def _profile_table_term_elements(
    elements: list[dict[str, Any]], *, profile_payload: dict[str, Any]
) -> list[dict[str, Any]]:
    """Expand configured Markdown lexicon cells into grounded term occurrences."""
    table_profile = profile_payload.get("markdown_term_table")
    if not isinstance(table_profile, dict) or not table_profile.get("enabled"):
        return elements
    term_headers = {str(value).strip() for value in table_profile.get("term_headers", [])}
    category_headers = {
        str(value).strip() for value in table_profile.get("category_headers", [])
    }
    delimiters = [str(value) for value in table_profile.get("term_delimiters", []) if value]
    if not term_headers or not delimiters:
        raise ValueError("markdown term-table profile requires headers and delimiters")
    splitter = re.compile("|".join(re.escape(value) for value in delimiters))
    expanded: list[dict[str, Any]] = []
    for element in elements:
        text = str(element.get("text") or "")
        anchor = dict(element.get("anchor") or {})
        base_offset = int(anchor.get("start_offset") or 0)
        table_header: list[str] | None = None
        term_column: int | None = None
        category_column: int | None = None
        search_from = 0
        for raw_line in text.splitlines(keepends=True):
            line = raw_line.rstrip("\r\n")
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 2:
                table_header = None
                term_column = None
                category_column = None
                search_from += len(raw_line)
                continue
            if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                search_from += len(raw_line)
                continue
            header_term_column = next(
                (index for index, cell in enumerate(cells) if cell in term_headers), None
            )
            if header_term_column is not None:
                table_header = cells
                term_column = header_term_column
                category_column = next(
                    (index for index, cell in enumerate(cells) if cell in category_headers), None
                )
                search_from += len(raw_line)
                continue
            if table_header is None or term_column is None or term_column >= len(cells):
                search_from += len(raw_line)
                continue
            term_cell = cells[term_column]
            cell_offset = text.find(term_cell, search_from, search_from + len(raw_line))
            if cell_offset < 0:
                search_from += len(raw_line)
                continue
            family = str(element.get("family") or "sensitive term")
            if category_column is not None and category_column < len(cells):
                family = cells[category_column] or family
            within_cell = 0
            for term in splitter.split(term_cell):
                term = term.strip()
                if not term:
                    continue
                term_offset = term_cell.find(term, within_cell)
                if term_offset < 0:
                    continue
                within_cell = term_offset + len(term)
                start = base_offset + cell_offset + term_offset
                end = start + len(term)
                term_anchor = {
                    **anchor,
                    "anchor_id": f"{anchor.get('anchor_id', 'source-anchor')}:term:{start}-{end}",
                    "start_offset": start,
                    "end_offset": end,
                    "exact_text": term,
                    "local_text_hash": content_hash(term),
                }
                expanded.append(
                    {
                        "text": term,
                        "family_id": element["family_id"],
                        "family": family,
                        "anchor": term_anchor,
                    }
                )
            search_from += len(raw_line)
    return expanded or elements


def _attach_canonical_evidence(
    result: SemanticaCandidateGraphResult, *, source: dict[str, Any],
    canonical_elements: list[dict[str, Any]] | None, organization_id: str,
    workspace_id: str | None, canonical_ir_document: CanonicalSourceDocument | None = None,
) -> SemanticaCandidateGraphResult:
    """Bind provider anchors to R2 EvidenceRefs at the extraction boundary.

    When ``canonical_ir_document`` is supplied (R3-01 IR-backed adapter) the
    real IR blocks and SourceLocator namespaces are preserved so EvidenceRefs
    round-trip through the original DOCX structure unchanged.  Otherwise a
    synthetic CanonicalSourceDocument is built from the package elements.
    """
    if not canonical_elements:
        raise ValueError("Semantica extraction requires canonical evidence elements")
    canonical_hash = str(source["source_hash"])
    if not canonical_hash.startswith("sha256:"):
        canonical_hash = "sha256:" + canonical_hash
    source_identity = SourceIdentity(
        schemaVersion="semantier.source_identity.v1",
        sourceIdentityRef=str(source["source_ref"]), organizationId=organization_id,
        workspaceId=str(workspace_id or "workspace"), sourceHash=canonical_hash,
        mediaType=str(source.get("source_kind") or "text"),
        originalName=str(source["source_ref"]), artifactRef=f"knowledge-source:{source['source_id']}",
        sourceVersion=None, supersedesSourceIdentityRef=None, createdAt=datetime.now(timezone.utc),
    )
    if canonical_ir_document is not None:
        # Re-bind the IR document onto the per-run source identity so the
        # canonical hash is honoured and the artifact is lineagable.  Block
        # identity already excludes parser version (per R1), so the existing
        # blocks are reused directly.
        document = canonical_ir_document.model_copy(update={
            "sourceIdentityRef": source_identity.source_identity_ref,
        })
        evidence_by_block = {
            block.block_id: evidence_ref_for_block(document, block, source_identity=source_identity)
            for block in document.blocks
        }

        def refs(anchors):
            resolved = []
            for anchor in anchors:
                anchor_id = str(getattr(anchor, "anchor_id", "") or "")
                # SourceAnchor.anchor_id is "block_id:span:0-N" — match on the
                # block prefix so the IR block identity drives EvidenceRef.
                block_id = anchor_id.split(":span:", 1)[0]
                match = next(
                    (b for b in document.blocks if b.block_id == block_id),
                    None,
                )
                if match is not None and match.block_id in evidence_by_block:
                    resolved.append(evidence_by_block[match.block_id])
            return tuple(resolved)

        graph = result.graph.model_copy(update={
            "nodes": tuple(node.model_copy(update={"evidence_refs": refs(node.source_anchors)}) for node in result.graph.nodes),
            "edges": tuple(edge.model_copy(update={"evidence_refs": refs(edge.source_anchors)}) for edge in result.graph.edges),
            "rules": tuple(rule.model_copy(update={"evidence_refs": refs(rule.source_anchors)}) for rule in result.graph.rules),
        })
        return SemanticaCandidateGraphResult(
            graph=graph, provider_ref=result.provider_ref, provider_commit=result.provider_commit,
            extraction_run_id=result.extraction_run_id, native_artifact_ref=result.native_artifact_ref,
            native_artifact_hash=result.native_artifact_hash,
        )

    blocks: list[SourceBlock] = []
    mappings: list[SourceMapping] = []
    for ordinal, element in enumerate(canonical_elements):
        anchor = dict(element.get("anchor") or element)
        anchor_id = str(anchor.get("anchor_id") or element.get("anchor_id") or f"block:{ordinal}")
        text = str(element.get("text") or anchor.get("exact_text") or "")
        locator = SourceLocator(
            sourceIdentityRef=source_identity.source_identity_ref, sourceHash=source_identity.source_hash,
            mediaType=source_identity.media_type, nativeProvider="canonical-input", nativeId=anchor_id,
            ordinalPath=[ordinal], paragraphId=anchor_id if anchor.get("document_part", "paragraph") == "paragraph" else None,
            tableId=f"table:{anchor.get('table_index')}" if anchor.get("table_index") is not None else None,
            rowId=f"row:{anchor.get('row_index')}" if anchor.get("row_index") is not None else None,
            columnId=f"cell:{anchor.get('cell_index')}" if anchor.get("cell_index") is not None else None,
        )
        block = SourceBlock(blockId=anchor_id, blockType="cell" if anchor.get("document_part") == "table_cell" else "paragraph",
                            content=text, ordinal=ordinal, structure={"sourceAnchorId": anchor_id},
                            contentHash=ir_content_hash(text))
        blocks.append(block)
        mappings.append(SourceMapping(sourceLocators=[locator], blockIds=[block.block_id],
                                      transformation=Transformation(kind="one_to_one", transformRef=f"canonical:{anchor_id}")))
    document = CanonicalSourceDocument(
        sourceIdentityRef=source_identity.source_identity_ref, blocks=blocks, sourceMap=SourceMap(mappings=mappings),
        parserTrace=ParserTrace(provider="semantier.canonical-input", adapterVersion="r3.evidence.v1",
                                 canonicalIrVersion="canonical_source_ir.v1"),
    )
    evidence_by_anchor = {
        block.block_id: evidence_ref_for_block(document, block, source_identity=source_identity)
        for block in blocks
    }

    def refs(anchors):
        return tuple(evidence_by_anchor[anchor.anchor_id] for anchor in anchors if anchor.anchor_id in evidence_by_anchor)

    graph = result.graph.model_copy(update={
        "nodes": tuple(node.model_copy(update={"evidence_refs": refs(node.source_anchors)}) for node in result.graph.nodes),
        "edges": tuple(edge.model_copy(update={"evidence_refs": refs(edge.source_anchors)}) for edge in result.graph.edges),
        "rules": tuple(rule.model_copy(update={"evidence_refs": refs(rule.source_anchors)}) for rule in result.graph.rules),
    })
    return SemanticaCandidateGraphResult(
        graph=graph, provider_ref=result.provider_ref, provider_commit=result.provider_commit,
        extraction_run_id=result.extraction_run_id, native_artifact_ref=result.native_artifact_ref,
        native_artifact_hash=result.native_artifact_hash,
    )


def governed_artifact_writer(raw: str, digest: str) -> str:
    root_value = os.environ.get("SEMANTIER_WORKSPACE_ARTIFACTS_DIR")
    if not root_value:
        raise ValueError("governed workspace artifact root is required")
    root = Path(root_value).resolve()
    root.mkdir(parents=True, exist_ok=True)
    name = digest.removeprefix("sha256:") + ".json"
    target = (root / "knowledge-extraction" / name).resolve()
    if root not in target.parents:
        raise ValueError("extraction artifact escaped governed root")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(raw, encoding="utf-8")
    return f"artifact:knowledge-extraction/{name}"


def governed_artifact_writer_for(root: str | Path) -> Callable[[str, str], str]:
    """Bind extraction artifacts to an authenticated governed artifact root."""

    resolved_root = Path(root).resolve()

    def write(raw: str, digest: str) -> str:
        resolved_root.mkdir(parents=True, exist_ok=True)
        name = digest.removeprefix("sha256:") + ".json"
        target = (resolved_root / "knowledge-extraction" / name).resolve()
        if resolved_root not in target.parents:
            raise ValueError("extraction artifact escaped governed root")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(raw, encoding="utf-8")
        return f"artifact:knowledge-extraction/{name}"

    return write


def _mean_confidence(occurrences) -> float | None:
    values = [item.machine_confidence for item in occurrences if item.machine_confidence is not None]
    return sum(values) / len(values) if values else None


def _candidate_graph(*, assertions, extraction_run_id: str, source_document_hash: str,
                     proposal: MachineProposal) -> SensitiveTermCandidateGraph:
    nodes = []
    rules = []
    for ordinal, assertion in enumerate(assertions, 1):
        normalized = assertion.normalized_assertion
        label = str((normalized.subject or {}).get("text") or
                    (normalized.object or {}).get("text") or normalized.predicate or assertion.assertion_id)
        node_id = f"candidate_node_{assertion.assertion_id}"
        nodes.append(CandidateNode(
            node_id=node_id, node_type=CandidateNodeType.CONCEPT,
            canonical_label=label, normalized_value=label.strip().casefold(),
            rationale="Extracted assertion candidate; proposal-only until human grounding.",
            source_anchors=assertion.source_anchors, extraction_run_id=extraction_run_id,
            machine_proposal_id=proposal.proposal_id, machine_confidence=assertion.confidence,
        ))
        rules.append(CandidateRule(
            rule_id=f"rule_{ordinal}", family_id=normalized.candidate_family_id or 1,
            condition_node_ids=(node_id,), reason_code="extracted_assertion_candidate",
            source_anchors=assertion.source_anchors, extraction_run_id=extraction_run_id,
            machine_proposal_id=proposal.proposal_id,
        ))
    return SensitiveTermCandidateGraph(
        candidate_graph_id=f"candidate_graph_{extraction_run_id}",
        source_document_hash=source_document_hash, extraction_run_id=extraction_run_id,
        schema_version="sensitive_term_candidate_graph.v1", ontology_version="tender_sensitive_v1",
        machine_proposal=proposal, nodes=tuple(nodes), edges=(), rules=tuple(rules),
    )
