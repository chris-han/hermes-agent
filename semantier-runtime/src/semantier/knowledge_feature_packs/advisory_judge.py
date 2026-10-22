"""Proposal-only judge critique generation for human review."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from typing import Any

from contracts.knowledge_feature_pack import LearningEventEnvelope, LearningEventType, MachineProposal
from contracts.knowledge_reasoning_phase1 import DeterministicGraphVerdict, ViolationEvidenceAssembly


def recorded_judge_critique(
    *,
    case_id: str,
    occurred_at: datetime,
    assembly: ViolationEvidenceAssembly,
    verdict: DeterministicGraphVerdict,
    structured_critique: dict[str, Any],
    artifact_writer: Any,
    prior_event_hash: str | None,
) -> LearningEventEnvelope:
    raw = json.dumps(structured_critique, ensure_ascii=False, sort_keys=True)
    digest = f"sha256:{sha256(raw.encode('utf-8')).hexdigest()}"
    proposal = MachineProposal(
        proposal_id=f"judge_{digest[7:31]}",
        provider="recorded-judge",
        model="judge:phase1-recorded",
        prompt_version="tender-graph-judge.v1",
        prompt_hash="sha256:tender-graph-judge-v1",
        parameters={"authority": "advisory_only", "temperature": 0},
        run_id=f"judge_run_{digest[7:19]}",
        raw_output_ref=artifact_writer(raw, digest),
        raw_output_hash=digest,
        confidence=structured_critique.get("confidence"),
    )
    return LearningEventEnvelope(
        event_id=f"judge_event_{digest[7:31]}",
        case_id=case_id,
        event_type=LearningEventType.JUDGE_CRITIQUE,
        occurred_at=occurred_at,
        source_document_hash=assembly.primary_source_anchors[0].document_hash,
        source_anchor_refs=tuple(anchor.anchor_id for anchor in assembly.primary_source_anchors),
        graph_version=assembly.graph_version,
        rule_version=assembly.triggered_rule_id or "unresolved",
        evaluation_corpus_version="tender_sensitive_v1",
        judge_critique=proposal,
        downstream_outcome={
            "deterministic_verdict_id": verdict.verdict_id,
            "assembly_hash": assembly.assembly_hash(),
            "advisory_only": True,
        },
        prior_event_hash=prior_event_hash,
    )

