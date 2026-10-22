from __future__ import annotations

from contracts.knowledge_classification import ClassificationEvidenceDTO


def derivative_manifest(evidence: ClassificationEvidenceDTO) -> dict[str, object]:
    return {
        "source_run_id": evidence.run.run_id,
        "input_id": evidence.run.input_id,
        "derivative_replaces_input": False,
        "evidence_set_hash": evidence.summary.evidence_set_hash if evidence.summary else None,
    }
