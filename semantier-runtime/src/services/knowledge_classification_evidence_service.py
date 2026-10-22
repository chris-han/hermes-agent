from __future__ import annotations

from dataclasses import asdict, replace
from typing import Protocol
from uuid import uuid4

from contracts.knowledge_classification import (
    ClassificationEvidenceDTO,
    ClassificationRunCreateDTO,
    ClassificationRunV1,
    ClassificationSummaryV1,
    EntityFindingV1,
    FieldFindingV1,
    OccurrenceFindingV1,
    RelationFindingV1,
    RuntimeConflictV1,
    RuntimeContextDTO,
    canonical_content_hash,
)


class KnowledgeClassificationEvidenceService(Protocol):
    def create_classification_run_idempotent(
        self,
        request: ClassificationRunCreateDTO,
        context: RuntimeContextDTO,
    ) -> ClassificationRunV1: ...

    def persist_occurrence_findings(
        self,
        run_id: str,
        findings: list[OccurrenceFindingV1],
        context: RuntimeContextDTO,
    ) -> str: ...

    def persist_field_findings(
        self,
        run_id: str,
        findings: list[FieldFindingV1],
        context: RuntimeContextDTO,
    ) -> str: ...

    def persist_entity_findings(
        self,
        run_id: str,
        findings: list[EntityFindingV1],
        context: RuntimeContextDTO,
    ) -> str: ...

    def persist_relation_findings(
        self,
        run_id: str,
        findings: list[RelationFindingV1],
        context: RuntimeContextDTO,
    ) -> str: ...

    def persist_runtime_conflicts(
        self,
        run_id: str,
        conflicts: list[RuntimeConflictV1],
        context: RuntimeContextDTO,
    ) -> str: ...

    def finalize_classification_run(
        self,
        run_id: str,
        status: str,
        context: RuntimeContextDTO,
    ) -> ClassificationRunV1: ...

    def get_classification_evidence(
        self,
        run_id: str,
        context: RuntimeContextDTO,
    ) -> ClassificationEvidenceDTO: ...


class InMemoryKnowledgeClassificationEvidenceService:
    def __init__(self) -> None:
        self._runs: dict[str, ClassificationRunV1] = {}
        self._idempotency: dict[str, str] = {}
        self._occurrences: dict[str, tuple[OccurrenceFindingV1, ...]] = {}
        self._fields: dict[str, tuple[FieldFindingV1, ...]] = {}
        self._entities: dict[str, tuple[EntityFindingV1, ...]] = {}
        self._relations: dict[str, tuple[RelationFindingV1, ...]] = {}
        self._conflicts: dict[str, tuple[RuntimeConflictV1, ...]] = {}

    def create_classification_run_idempotent(
        self,
        request: ClassificationRunCreateDTO,
        context: RuntimeContextDTO,
    ) -> ClassificationRunV1:
        if request.idempotency_key in self._idempotency:
            return self._scoped_run(self._idempotency[request.idempotency_key], context)
        run_id = str(request.metadata.get("run_id") or f"kcr_{uuid4().hex}")
        run = ClassificationRunV1(
            run_id=run_id,
            input_id=request.input_id,
            status="pending",
            source_hash=request.source_hash,
            activation_set_snapshot_id=request.activation_set_snapshot_id,
            resolved_activation_set_hash=request.resolved_activation_set_hash,
            rule_version_set_hash=request.rule_version_set_hash,
            classifier_profile_version=request.classifier_profile_version,
            ontology_version=request.ontology_version,
            content_hash=canonical_content_hash({"request": asdict(request), "context": asdict(context)}),
            metadata={
                **request.metadata,
                "organization_id": context.organization_id,
                "workspace_id": context.workspace_id,
            },
        )
        self._runs[run_id] = run
        self._idempotency[request.idempotency_key] = run_id
        return run

    def persist_occurrence_findings(
        self,
        run_id: str,
        findings: list[OccurrenceFindingV1],
        context: RuntimeContextDTO,
    ) -> str:
        self._scoped_run(run_id, context)
        self._occurrences[run_id] = tuple(findings)
        return canonical_content_hash([asdict(item) for item in findings])

    def persist_field_findings(
        self,
        run_id: str,
        findings: list[FieldFindingV1],
        context: RuntimeContextDTO,
    ) -> str:
        self._scoped_run(run_id, context)
        self._fields[run_id] = tuple(findings)
        return canonical_content_hash([asdict(item) for item in findings])

    def persist_entity_findings(
        self,
        run_id: str,
        findings: list[EntityFindingV1],
        context: RuntimeContextDTO,
    ) -> str:
        self._scoped_run(run_id, context)
        self._entities[run_id] = tuple(findings)
        return canonical_content_hash([asdict(item) for item in findings])

    def persist_relation_findings(
        self,
        run_id: str,
        findings: list[RelationFindingV1],
        context: RuntimeContextDTO,
    ) -> str:
        self._scoped_run(run_id, context)
        self._relations[run_id] = tuple(findings)
        return canonical_content_hash([asdict(item) for item in findings])

    def persist_runtime_conflicts(
        self,
        run_id: str,
        conflicts: list[RuntimeConflictV1],
        context: RuntimeContextDTO,
    ) -> str:
        self._scoped_run(run_id, context)
        self._conflicts[run_id] = tuple(conflicts)
        return canonical_content_hash([asdict(item) for item in conflicts])

    def finalize_classification_run(
        self,
        run_id: str,
        status: str,
        context: RuntimeContextDTO,
    ) -> ClassificationRunV1:
        run = self._scoped_run(run_id, context)
        evidence = self._evidence_payload(run_id)
        finalized = replace(
            run,
            status=status,
            content_hash=canonical_content_hash({"run": asdict(run), "evidence": evidence}),
        )
        self._runs[run_id] = finalized
        return finalized

    def get_classification_evidence(
        self,
        run_id: str,
        context: RuntimeContextDTO,
    ) -> ClassificationEvidenceDTO:
        run = self._scoped_run(run_id, context)
        occurrences = self._occurrences.get(run_id, ())
        fields = self._fields.get(run_id, ())
        entities = self._entities.get(run_id, ())
        relations = self._relations.get(run_id, ())
        conflicts = self._conflicts.get(run_id, ())
        evidence_payload = self._evidence_payload(run_id)
        return ClassificationEvidenceDTO(
            run=run,
            occurrences=occurrences,
            fields=fields,
            entities=entities,
            relations=relations,
            runtime_conflicts=conflicts,
            summary=ClassificationSummaryV1(
                run_id=run_id,
                occurrence_count=len(occurrences),
                field_count=len(fields),
                entity_count=len(entities),
                relation_count=len(relations),
                conflict_count=len(conflicts),
                evidence_set_hash=canonical_content_hash(evidence_payload),
            ),
        )

    def _evidence_payload(self, run_id: str) -> dict[str, object]:
        return {
            "occurrences": [asdict(item) for item in self._occurrences.get(run_id, ())],
            "fields": [asdict(item) for item in self._fields.get(run_id, ())],
            "entities": [asdict(item) for item in self._entities.get(run_id, ())],
            "relations": [asdict(item) for item in self._relations.get(run_id, ())],
            "runtime_conflicts": [asdict(item) for item in self._conflicts.get(run_id, ())],
        }

    def _scoped_run(self, run_id: str, context: RuntimeContextDTO) -> ClassificationRunV1:
        run = self._runs.get(run_id)
        if run is None:
            raise ValueError("CLASSIFICATION_RUN_NOT_FOUND")
        if run.metadata.get("organization_id") != context.organization_id:
            raise ValueError("CLASSIFICATION_RUN_NOT_FOUND")
        if (run.metadata.get("workspace_id") or None) != (context.workspace_id or None):
            raise ValueError("CLASSIFICATION_RUN_NOT_FOUND")
        return run
