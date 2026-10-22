"""Deterministic legal semantic candidate extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from eos.artifact_hashing import content_hash
from eos.legal_text_normalization import LegalSectionAnchor


LEGAL_CANDIDATE_TYPES = {
    "concept",
    "actor",
    "procurement_object",
    "funding_classification",
    "procurement_method",
    "stage",
    "claim",
    "condition",
    "exception",
    "threshold",
    "remedy",
    "issue_dependency",
    "date",
    "amount",
    "enumeration",
    "document_reference",
}

LEGAL_LLM_CANDIDATE_TYPES = {
    "condition",
    "claim",
    "exception",
    "issue_dependency",
}

_DATE_RE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
_AMOUNT_RE = re.compile(r"(\d+(?:\.\d+)?)(万元|元|亿元)")
_ARTICLE_REF_RE = re.compile(r"第([一二三四五六七八九十百千万零〇两\d]+)条")
_ENUM_RE = re.compile(r"[（(]([一二三四五六七八九十百千万零〇两\d]+)[）)]([^；;。]+)")
_THRESHOLD_RE = re.compile(r"(达到|超过|不低于|不少于|低于|小于|大于)(\d+(?:\.\d+)?)(万元|元|亿元)")


@dataclass(frozen=True)
class LegalSemanticCandidate:
    candidate_type: str
    candidate: dict[str, Any]
    source_span: dict[str, Any]
    extraction_method: str
    review_state: str
    candidate_hash: str
    rejection_reason: str | None = None


def extract_deterministic_candidates(section: LegalSectionAnchor) -> list[LegalSemanticCandidate]:
    text = section.text
    span_base = {
        "stable_locator": section.stable_locator,
        "section_type": section.section_type,
        "local_hash": section.local_hash,
    }
    candidates: list[LegalSemanticCandidate] = []

    for match in _DATE_RE.finditer(text):
        year, month, day = match.groups()
        candidates.append(
            _candidate(
                candidate_type="date",
                candidate={
                    "date_iso": f"{int(year):04d}-{int(month):02d}-{int(day):02d}",
                    "display_text": match.group(0),
                },
                source_span={**span_base, "char_start": match.start(), "char_end": match.end()},
            )
        )

    for match in _AMOUNT_RE.finditer(text):
        value, unit = match.groups()
        candidates.append(
            _candidate(
                candidate_type="amount",
                candidate={
                    "amount_value": float(value),
                    "amount_unit": _amount_unit(unit),
                    "display_text": match.group(0),
                },
                source_span={**span_base, "char_start": match.start(), "char_end": match.end()},
            )
        )

    for match in _ARTICLE_REF_RE.finditer(text):
        candidates.append(
            _candidate(
                candidate_type="document_reference",
                candidate={
                    "reference_type": "article",
                    "article_marker": match.group(1),
                    "display_text": match.group(0),
                },
                source_span={**span_base, "char_start": match.start(), "char_end": match.end()},
            )
        )

    for match in _ENUM_RE.finditer(text):
        candidates.append(
            _candidate(
                candidate_type="enumeration",
                candidate={
                    "enumeration_marker": match.group(1),
                    "enumeration_text": match.group(2).strip(),
                    "display_text": match.group(0),
                },
                source_span={**span_base, "char_start": match.start(), "char_end": match.end()},
            )
        )

    for match in _THRESHOLD_RE.finditer(text):
        comparator, value, unit = match.groups()
        candidates.append(
            _candidate(
                candidate_type="threshold",
                candidate={
                    "comparator": comparator,
                    "amount_value": float(value),
                    "amount_unit": _amount_unit(unit),
                    "display_text": match.group(0),
                },
                source_span={**span_base, "char_start": match.start(), "char_end": match.end()},
            )
        )

    for keyword, candidate_type in {
        "招标人": "actor",
        "投标人": "actor",
        "采购人": "actor",
        "工程建设项目": "procurement_object",
        "财政性资金": "funding_classification",
        "公开招标": "procurement_method",
        "邀请招标": "procurement_method",
        "评标": "stage",
        "投诉": "remedy",
    }.items():
        start = text.find(keyword)
        if start >= 0:
            candidates.append(
                _candidate(
                    candidate_type=candidate_type,
                    candidate={"display_text": keyword},
                    source_span={**span_base, "char_start": start, "char_end": start + len(keyword)},
                )
            )

    if "但是" in text or "除" in text:
        keyword = "但是" if "但是" in text else "除"
        start = text.find(keyword)
        candidates.append(
            _candidate(
                candidate_type="exception",
                candidate={"display_text": keyword, "scope": "requires_review"},
                source_span={**span_base, "char_start": start, "char_end": len(text)},
                review_state="UNRESOLVED",
            )
        )

    return candidates


def validate_candidate(candidate: LegalSemanticCandidate) -> LegalSemanticCandidate:
    if candidate.candidate_type not in LEGAL_CANDIDATE_TYPES:
        return _replace_review_state(candidate, "REJECTED", "unsupported_candidate_type")
    if not candidate.source_span.get("stable_locator") or not candidate.source_span.get("local_hash"):
        return _replace_review_state(candidate, "REJECTED", "missing_source_span")
    if not candidate.extraction_method:
        return _replace_review_state(candidate, "REJECTED", "missing_extraction_method")
    return candidate


def create_bounded_llm_candidate(
    *,
    candidate_type: str,
    candidate: dict[str, Any],
    source_span: dict[str, Any],
    model_ref: str,
    prompt_hash: str,
    output_text: str,
    max_source_chars: int,
    max_output_chars: int,
    confidence: float | None = None,
) -> LegalSemanticCandidate:
    candidate_type = str(candidate_type or "").strip()
    if candidate_type not in LEGAL_LLM_CANDIDATE_TYPES:
        raise ValueError("bounded LLM candidates are limited to conditional semantic types")
    if max_source_chars <= 0 or max_output_chars <= 0:
        raise ValueError("bounded LLM extraction limits are required")
    if len(str(output_text or "")) > max_output_chars:
        raise ValueError("bounded LLM output exceeds max_output_chars")
    if not str(model_ref or "").strip() or not str(prompt_hash or "").strip():
        raise ValueError("bounded LLM model_ref and prompt_hash are required")
    if not source_span.get("stable_locator") or not source_span.get("local_hash"):
        raise ValueError("bounded LLM source span is required")
    payload = {
        **candidate,
        "llm_boundary": {
            "model_ref": model_ref,
            "prompt_hash": prompt_hash,
            "output_hash": content_hash({"output_text": output_text}),
            "max_source_chars": max_source_chars,
            "max_output_chars": max_output_chars,
            "confidence": confidence,
        },
    }
    return _candidate(
        candidate_type=candidate_type,
        candidate=payload,
        source_span=source_span,
        extraction_method="bounded_llm_candidate",
        review_state="UNRESOLVED",
    )


def _candidate(
    *,
    candidate_type: str,
    candidate: dict[str, Any],
    source_span: dict[str, Any],
    extraction_method: str = "deterministic_parser",
    review_state: str = "CURATION_DRAFT",
) -> LegalSemanticCandidate:
    payload = {
        "candidate_type": candidate_type,
        "candidate": candidate,
        "source_span": source_span,
        "extraction_method": extraction_method,
    }
    return LegalSemanticCandidate(
        candidate_type=candidate_type,
        candidate=candidate,
        source_span=source_span,
        extraction_method=extraction_method,
        review_state=review_state,
        candidate_hash=content_hash(payload),
    )


def _replace_review_state(
    candidate: LegalSemanticCandidate,
    review_state: str,
    rejection_reason: str,
) -> LegalSemanticCandidate:
    return LegalSemanticCandidate(
        candidate_type=candidate.candidate_type,
        candidate=candidate.candidate,
        source_span=candidate.source_span,
        extraction_method=candidate.extraction_method,
        review_state=review_state,
        candidate_hash=candidate.candidate_hash,
        rejection_reason=rejection_reason,
    )


def _amount_unit(unit: str) -> str:
    return {"元": "CNY", "万元": "CNY_10K", "亿元": "CNY_100M"}[unit]
