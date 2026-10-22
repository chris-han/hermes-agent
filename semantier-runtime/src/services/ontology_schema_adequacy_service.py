"""Deterministic competency-to-schema representability evaluation."""

from __future__ import annotations

from datetime import datetime

from contracts.normative_schema import (
    NormativeSchemaContractV1,
    SchemaAdequacyReportV1,
    SchemaAdequacyRowV1,
)
from contracts.ontology_building import CompetencyQuestionV1
from eos.artifact_hashing import content_hash


class OntologySchemaAdequacyService:
    def verify_report(
        self,
        *,
        schema: NormativeSchemaContractV1,
        questions: tuple[CompetencyQuestionV1, ...],
        report: SchemaAdequacyReportV1,
    ) -> None:
        """Reproduce the complete report from pinned inputs before certification."""
        expected = self.evaluate(
            schema=schema,
            questions=questions,
            report_ref=report.report_ref,
            created_at=report.created_at,
        )
        if report != expected:
            raise ValueError("SCHEMA_ADEQUACY_EVIDENCE_MISMATCH")

    def evaluate(
        self,
        *,
        schema: NormativeSchemaContractV1,
        questions: tuple[CompetencyQuestionV1, ...],
        report_ref: str,
        created_at: datetime,
    ) -> SchemaAdequacyReportV1:
        slots = {slot.dimension: slot for slot in schema.slots}
        rows: list[SchemaAdequacyRowV1] = []
        for question in questions:
            required = question.required_semantic_dimensions
            missing = [
                dimension
                for dimension in required
                if dimension not in slots or slots[dimension].epistemic_status != "FOUND"
            ]
            unresolved = [
                dimension
                for dimension in missing
                if dimension in slots and slots[dimension].epistemic_status == "UNRESOLVED"
            ]
            representability = (
                "REPRESENTABLE"
                if not missing
                else "UNRESOLVED"
                if unresolved
                else "UNREPRESENTABLE"
            )
            if not question.applicability_resolved:
                representability = "UNRESOLVED"
            objects = tuple(
                object_ref
                for dimension in required
                if dimension in slots
                for object_ref in slots[dimension].schema_object_refs
            )
            rows.append(
                SchemaAdequacyRowV1(
                    competency_id=question.competency_id,
                    critical=question.criticality == "critical",
                    required_semantic_dimensions=required,
                    representing_schema_objects=objects,
                    representability=representability,
                    evidence_refs=question.evidence_refs,
                    decision_refs=(),
                    blocking_reason=(
                        "unresolved CQ dimension applicability" if not question.applicability_resolved else
                        f"missing schema dimensions: {', '.join(missing)}" if missing else None
                    ),
                    allowed_resolution_path="schema_successor" if missing else None,
                )
            )
        blocking = any(
            row.critical and row.representability != "REPRESENTABLE" for row in rows
        )
        basis = {
            "normative_schema_ref": schema.schema_ref,
            "normative_schema_hash": schema.schema_hash,
            "rows": [row.model_dump(mode="json") for row in rows],
        }
        digest = content_hash(basis)
        return SchemaAdequacyReportV1(
            report_ref=report_ref,
            report_hash=f"sha256:{digest}",
            organization_id=schema.organization_id,
            workspace_id=schema.workspace_id,
            normative_schema_ref=schema.schema_ref,
            normative_schema_hash=schema.schema_hash,
            rows=tuple(rows),
            overall_decision="FAIL" if blocking else "PASS",
            certifiable_s3_allowed=not blocking,
            created_at=created_at,
        )


__all__ = ["OntologySchemaAdequacyService"]
