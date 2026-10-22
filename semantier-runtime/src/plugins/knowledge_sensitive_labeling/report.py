from __future__ import annotations

from dataclasses import asdict

from contracts.knowledge_classification import ClassificationEvidenceDTO


def build_redaction_safe_report(evidence: ClassificationEvidenceDTO) -> dict[str, object]:
    return {
        "run_id": evidence.run.run_id,
        "input_id": evidence.run.input_id,
        "summary": asdict(evidence.summary) if evidence.summary else None,
        "finding_refs": {
            "occurrences": [item.finding_id for item in evidence.occurrences],
            "fields": [item.finding_id for item in evidence.fields],
            "entities": [item.finding_id for item in evidence.entities],
            "relations": [item.finding_id for item in evidence.relations],
        },
        "conflict_refs": [item.conflict_id for item in evidence.runtime_conflicts],
    }
