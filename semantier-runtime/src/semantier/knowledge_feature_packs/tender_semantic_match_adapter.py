"""Candidate-only tender semantic-match adapter over current-agent structured LLM."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Sequence
from uuid import uuid4

from contracts.contextgraph_tender_semantic_match import (
    TenderSemanticMatchBatch,
    TenderSemanticMatchCandidate,
    TenderSemanticMatchSuggestion,
)
from contracts.knowledge_feature_pack import content_hash
from prompts.loader import PromptAsset, load_repo_prompt
from storage.hosted_postgres import enabled, HostedStorageUnavailable


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class TenderSemanticMatchAdapter:
    def __init__(self, llm: Any, *, db_path: str | Path | None = None) -> None:
        if db_path is not None and enabled():
            raise HostedStorageUnavailable('Tender semantic-match evaluation persistence is outside the selected hosted stores; explicit migration is required')
        self._llm = llm
        self._db_path = Path(db_path) if db_path is not None else None
        self._conn: sqlite3.Connection | None = None
        if self._db_path is not None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._init_schema()

    @staticmethod
    def default_prompt() -> PromptAsset:
        return load_repo_prompt(
            "contextgraph/tender_semantic_match_v1.md",
            allowed_namespace="contextgraph",
        )

    async def suggest(
        self,
        *,
        candidates: Sequence[TenderSemanticMatchCandidate],
        prompt: PromptAsset | None = None,
    ) -> tuple[TenderSemanticMatchSuggestion, ...]:
        prompt = prompt or self.default_prompt()
        started = monotonic()
        try:
            generated = await self._llm.generate_structured(
                instructions=prompt.text,
                input_text=_json({"candidates": [item.model_dump(mode="json") for item in candidates]}),
                json_schema=TenderSemanticMatchBatch.model_json_schema(),
                schema_name="contextgraph_tender_semantic_match_batch_v1",
                purpose="contextgraph_tender_semantic_match",
            )
            latency_ms = int((monotonic() - started) * 1000)
            parsed = TenderSemanticMatchBatch.model_validate(generated.parsed)
            self._validate_results(candidates, parsed.results)
            suggestions = tuple(
                TenderSemanticMatchSuggestion(
                    schema_version="contextgraph_tender_semantic_match_suggestion.v1",
                    suggestion_id=f"semantic_match_{uuid4().hex}",
                    finding_id=result.finding_id,
                    concept_id=by_id[result.finding_id].concept_id,
                    observed_expression=by_id[result.finding_id].observed_expression,
                    semantic_relation=result.semantic_relation,
                    meaning_match=result.meaning_match,
                    relation_match=result.relation_match,
                    polarity=result.polarity,
                    modality=result.modality,
                    qualification_context=result.qualification_context,
                    confidence=result.confidence,
                    evidence_anchor_refs=result.evidence_anchor_refs,
                    rationale=result.rationale,
                    provider=generated.provider,
                    model=generated.model,
                    agent_id=generated.agent_id,
                    prompt_ref=prompt.relative_name,
                    prompt_hash=prompt.sha256,
                    latency_ms=latency_ms,
                    token_usage=dict(generated.usage),
                    failure_reason=None,
                    grounding_status="ai_suggested",
                )
                for by_id in [{item.finding_id: item for item in candidates}]
                for result in parsed.results
            )
        except Exception as exc:
            latency_ms = int((monotonic() - started) * 1000)
            suggestions = tuple(
                TenderSemanticMatchSuggestion(
                    schema_version="contextgraph_tender_semantic_match_suggestion.v1",
                    suggestion_id=f"semantic_match_{uuid4().hex}",
                    finding_id=item.finding_id,
                    concept_id=item.concept_id,
                    observed_expression=item.observed_expression,
                    prompt_ref=prompt.relative_name,
                    prompt_hash=prompt.sha256,
                    latency_ms=latency_ms,
                    token_usage={},
                    failure_reason=str(exc),
                    grounding_status="provider_error",
                )
                for item in candidates
            )
        self._persist(suggestions)
        return suggestions

    @staticmethod
    def _validate_results(candidates: Sequence[TenderSemanticMatchCandidate], results: Sequence[Any]) -> None:
        expected = {item.finding_id: item for item in candidates}
        returned = [item.finding_id for item in results]
        if len(returned) != len(set(returned)) or set(returned) != set(expected):
            raise ValueError("semantic match result coverage mismatch")
        for result in results:
            if not set(result.evidence_anchor_refs).issubset(expected[result.finding_id].evidence_anchor_refs):
                raise ValueError("semantic match result references noncanonical evidence")

    def _init_schema(self) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS contextgraph_tender_semantic_match_suggestions (
              suggestion_id TEXT PRIMARY KEY,
              finding_id TEXT NOT NULL,
              concept_id TEXT NOT NULL,
              suggestion_json TEXT NOT NULL,
              provider TEXT,
              model TEXT,
              prompt_hash TEXT NOT NULL,
              latency_ms INTEGER NOT NULL,
              token_usage_json TEXT NOT NULL,
              failure_reason TEXT,
              grounding_status TEXT NOT NULL,
              created_at TEXT NOT NULL,
              content_hash TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def _persist(self, suggestions: tuple[TenderSemanticMatchSuggestion, ...]) -> None:
        if self._conn is None:
            return
        now = _now()
        for suggestion in suggestions:
            payload = suggestion.model_dump(mode="json")
            self._conn.execute(
                """INSERT INTO contextgraph_tender_semantic_match_suggestions (
                suggestion_id,finding_id,concept_id,suggestion_json,provider,model,
                prompt_hash,latency_ms,token_usage_json,failure_reason,grounding_status,
                created_at,content_hash
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    suggestion.suggestion_id,
                    suggestion.finding_id,
                    suggestion.concept_id,
                    _json(payload),
                    suggestion.provider,
                    suggestion.model,
                    suggestion.prompt_hash,
                    suggestion.latency_ms,
                    _json(suggestion.token_usage),
                    suggestion.failure_reason,
                    suggestion.grounding_status,
                    now,
                    content_hash(payload),
                ),
            )
        self._conn.commit()

    def list_suggestions(self) -> list[dict[str, Any]]:
        if self._conn is None:
            return []
        rows = self._conn.execute(
            """SELECT suggestion_json FROM contextgraph_tender_semantic_match_suggestions
            ORDER BY created_at, suggestion_id"""
        ).fetchall()
        return [json.loads(row["suggestion_json"]) for row in rows]
