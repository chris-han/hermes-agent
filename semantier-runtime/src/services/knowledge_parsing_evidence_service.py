from __future__ import annotations

from dataclasses import asdict, replace
from typing import Protocol
from uuid import uuid4

from contracts.knowledge_classification import (
    CanonicalClassificationInputV1,
    CanonicalEntityV1,
    CanonicalRecordV1,
    CanonicalRelationV1,
    CanonicalSegmentV1,
    ParseEvidenceDTO,
    ParseRunCreateDTO,
    ParseRunDTO,
    RuntimeContextDTO,
    canonical_content_hash,
)


class KnowledgeParsingEvidenceService(Protocol):
    def create_parse_run_idempotent(
        self,
        request: ParseRunCreateDTO,
        context: RuntimeContextDTO,
    ) -> ParseRunDTO: ...

    def persist_canonical_records(
        self,
        parse_run_id: str,
        records: list[CanonicalRecordV1],
        context: RuntimeContextDTO,
    ) -> str: ...

    def persist_canonical_segments(
        self,
        parse_run_id: str,
        segments: list[CanonicalSegmentV1],
        context: RuntimeContextDTO,
    ) -> str: ...

    def persist_entities_and_relations(
        self,
        parse_run_id: str,
        entities: list[CanonicalEntityV1],
        relations: list[CanonicalRelationV1],
        context: RuntimeContextDTO,
    ) -> str: ...

    def finalize_parse_run(
        self,
        parse_run_id: str,
        canonical_input: CanonicalClassificationInputV1,
        context: RuntimeContextDTO,
    ) -> ParseRunDTO: ...

    def get_canonical_classification_input(
        self,
        input_id: str,
        context: RuntimeContextDTO,
    ) -> CanonicalClassificationInputV1: ...

    def verify_parse_artifact_hashes(
        self,
        input_id: str,
        context: RuntimeContextDTO,
    ) -> bool: ...


class InMemoryKnowledgeParsingEvidenceService:
    def __init__(self) -> None:
        self._runs: dict[str, ParseRunDTO] = {}
        self._inputs: dict[str, CanonicalClassificationInputV1] = {}
        self._idempotency: dict[str, str] = {}
        self._records: dict[str, tuple[CanonicalRecordV1, ...]] = {}
        self._segments: dict[str, tuple[CanonicalSegmentV1, ...]] = {}
        self._entities: dict[str, tuple[CanonicalEntityV1, ...]] = {}
        self._relations: dict[str, tuple[CanonicalRelationV1, ...]] = {}

    def create_parse_run_idempotent(
        self,
        request: ParseRunCreateDTO,
        context: RuntimeContextDTO,
    ) -> ParseRunDTO:
        if request.idempotency_key in self._idempotency:
            return self._scoped_run(self._idempotency[request.idempotency_key], context)
        parse_run_id = str(request.metadata.get("parse_run_id") or f"kpr_{uuid4().hex}")
        input_id = str(request.metadata.get("input_id") or f"kci_{uuid4().hex}")
        run = ParseRunDTO(
            parse_run_id=parse_run_id,
            input_id=input_id,
            status="pending",
            source_hash=request.source_hash,
            content_hash=canonical_content_hash({"request": asdict(request), "context": asdict(context)}),
            metadata={
                **request.metadata,
                "organization_id": context.organization_id,
                "workspace_id": context.workspace_id,
                "source_kind": request.source_kind,
                "source_ref": request.source_ref,
                "parser_profile_version": request.parser_profile_version,
                "normalized_artifact_ref": request.normalized_artifact_ref,
                "normalized_artifact_hash": request.normalized_artifact_hash,
            },
        )
        self._runs[parse_run_id] = run
        self._idempotency[request.idempotency_key] = parse_run_id
        return run

    def persist_canonical_records(
        self,
        parse_run_id: str,
        records: list[CanonicalRecordV1],
        context: RuntimeContextDTO,
    ) -> str:
        self._scoped_run(parse_run_id, context)
        self._records[parse_run_id] = tuple(records)
        return canonical_content_hash([asdict(item) for item in records])

    def persist_canonical_segments(
        self,
        parse_run_id: str,
        segments: list[CanonicalSegmentV1],
        context: RuntimeContextDTO,
    ) -> str:
        self._scoped_run(parse_run_id, context)
        self._segments[parse_run_id] = tuple(segments)
        return canonical_content_hash([asdict(item) for item in segments])

    def persist_entities_and_relations(
        self,
        parse_run_id: str,
        entities: list[CanonicalEntityV1],
        relations: list[CanonicalRelationV1],
        context: RuntimeContextDTO,
    ) -> str:
        self._scoped_run(parse_run_id, context)
        self._entities[parse_run_id] = tuple(entities)
        self._relations[parse_run_id] = tuple(relations)
        return canonical_content_hash(
            {
                "entities": [asdict(item) for item in entities],
                "relations": [asdict(item) for item in relations],
            }
        )

    def finalize_parse_run(
        self,
        parse_run_id: str,
        canonical_input: CanonicalClassificationInputV1,
        context: RuntimeContextDTO,
    ) -> ParseRunDTO:
        run = self._scoped_run(parse_run_id, context)
        if canonical_input.organization_id != context.organization_id:
            raise ValueError("CANONICAL_INPUT_SCOPE_MISMATCH")
        if (canonical_input.workspace_id or None) != (context.workspace_id or None):
            raise ValueError("CANONICAL_INPUT_SCOPE_MISMATCH")
        finalized = replace(
            run,
            status="finalized",
            input_id=canonical_input.input_id,
            content_hash=canonical_content_hash({"run": asdict(run), "input": asdict(canonical_input)}),
        )
        self._runs[parse_run_id] = finalized
        self._inputs[canonical_input.input_id] = canonical_input
        return finalized

    def get_canonical_classification_input(
        self,
        input_id: str,
        context: RuntimeContextDTO,
    ) -> CanonicalClassificationInputV1:
        canonical_input = self._inputs.get(input_id)
        if canonical_input is None:
            raise ValueError("CANONICAL_CLASSIFICATION_INPUT_NOT_FOUND")
        if canonical_input.organization_id != context.organization_id:
            raise ValueError("CANONICAL_CLASSIFICATION_INPUT_NOT_FOUND")
        if (canonical_input.workspace_id or None) != (context.workspace_id or None):
            raise ValueError("CANONICAL_CLASSIFICATION_INPUT_NOT_FOUND")
        return canonical_input

    def get_parse_evidence(
        self,
        parse_run_id: str,
        context: RuntimeContextDTO,
    ) -> ParseEvidenceDTO:
        run = self._scoped_run(parse_run_id, context)
        return ParseEvidenceDTO(
            run=run,
            canonical_input=self.get_canonical_classification_input(run.input_id, context),
        )

    def verify_parse_artifact_hashes(
        self,
        input_id: str,
        context: RuntimeContextDTO,
    ) -> bool:
        canonical_input = self.get_canonical_classification_input(input_id, context)
        recomputed = CanonicalClassificationInputV1.build(
            input_id=canonical_input.input_id,
            organization_id=canonical_input.organization_id,
            workspace_id=canonical_input.workspace_id,
            source_kind=canonical_input.source_kind,
            source_ref=canonical_input.source_ref,
            source_hash=canonical_input.source_hash,
            normalized_artifact_ref=canonical_input.normalized_artifact_ref,
            normalized_artifact_hash=canonical_input.normalized_artifact_hash,
            parser_profile_version=canonical_input.parser_profile_version,
            schema_version=canonical_input.schema_version,
            records=canonical_input.records,
            segments=canonical_input.segments,
            entities=canonical_input.entities,
            relations=canonical_input.relations,
            graph_projection_ref=canonical_input.graph_projection_ref,
            lineage_manifest=canonical_input.lineage_manifest,
        )
        return recomputed.content_hash == canonical_input.content_hash

    def _scoped_run(self, parse_run_id: str, context: RuntimeContextDTO) -> ParseRunDTO:
        run = self._runs.get(parse_run_id)
        if run is None:
            raise ValueError("PARSE_RUN_NOT_FOUND")
        if run.metadata.get("organization_id") != context.organization_id:
            raise ValueError("PARSE_RUN_NOT_FOUND")
        if (run.metadata.get("workspace_id") or None) != (context.workspace_id or None):
            raise ValueError("PARSE_RUN_NOT_FOUND")
        return run
