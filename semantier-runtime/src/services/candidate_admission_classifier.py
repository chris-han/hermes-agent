"""Pure deterministic D1-D9 routing classifier for learning candidates.

The classifier emits non-authoritative CandidateAdmissionClassificationV1 evidence.
It never persists, admits, activates, or creates owner receipts.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

from contracts.artifact_pin import ArtifactPinV1
from contracts.candidate_admission_classification import (
    AdmissionDimension,
    CandidateAdmissionClassificationV1,
)
from contracts.change_candidate_v2 import (
    AtomicChangeProfileV1,
    ChangeCandidateV2,
    ChangeSetProfileV1,
    LearningHypothesisProfileV2,
)
from contracts.semantic_kernel import ChangeCandidate


CLASSIFIER_VERSION = "candidate-admission-d1-d9.v1"

_AUTHORITY_OWNER_PHRASES = (
    "prefer the learned skill rule over",
    "follow the skill because it is newer",
    "authoritative semantic policy",
    "override the current q0",
    "override q0",
    "provenance may be omitted",
    "backfill provenance later",
)
_ACTIVATION_PHRASES = (
    "immediately replace the active skill",
    "activate the candidate automatically",
    "may immediately replace",
    "once validation passes, activate",
)
_STRUCTURAL_MISSING_PHRASES = (
    "required evidence is incomplete",
    "structural context is missing",
    "evidence is incomplete",
    "missing required evidence",
)
_STRUCTURAL_BYPASS_PHRASES = (
    "choose false",
    "choose true",
    "proceed with the semantic label",
    "return the semantic label",
)

_SEMANTIC_RULES: tuple[tuple[AdmissionDimension, tuple[str, ...]], ...] = (
    (
        "D2",
        (
            "a = true (",
            "b = false (",
            "interpret it as true",
            "interpret it as false",
            "default to false",
            "default to true",
            "unknown should be",
            "treat every scoring formula as false",
            "always false",
        ),
    ),
    (
        "D3",
        (
            "fall inside actual price restriction",
            "expand the predicate",
            "expand the scope",
            "cover award-stage",
            "inside the predicate",
        ),
    ),
    (
        "D4",
        (
            "sufficient evidence for true",
            "sufficient evidence for false",
            "sufficient for true",
            "surrounding decision context is optional",
            "context is optional",
        ),
    ),
    (
        "D5",
        (
            "does not count as an actual restriction",
            "new defeating condition",
            "defeating patterns for actual restriction",
            "new exception",
            "new defeater",
        ),
    ),
    ("D6", ("takes precedence", "has precedence", "overrides explanatory prose")),
    ("D7", ("unknown should be reserved", "unknown means", "redefine unknown")),
)

_PROCEDURAL_MARKERS = (
    "inspect",
    "check",
    "resolve",
    "compare",
    "re-read",
    "record",
    "read all evidence",
    "evaluation order",
    "before selecting",
    "before returning",
    "before finalizing",
    "proceed through the existing governed semantic checks",
    "apply the existing governed unknown definition",
)
_PRESERVATION_MARKERS = (
    "do not change any governed definition",
    "without adding new rules",
    "existing a/b/c anchor",
    "do not change what true, false, or unknown mean",
    "existing governed",
    "same anchor-only result",
)


def _has_any(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _classifier_hash() -> str:
    payload = {
        "version": CLASSIFIER_VERSION,
        "authority_owner": _AUTHORITY_OWNER_PHRASES,
        "activation": _ACTIVATION_PHRASES,
        "structural_missing": _STRUCTURAL_MISSING_PHRASES,
        "structural_bypass": _STRUCTURAL_BYPASS_PHRASES,
        "semantic_rules": _SEMANTIC_RULES,
        "procedural": _PROCEDURAL_MARKERS,
        "preservation": _PRESERVATION_MARKERS,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


CLASSIFIER_HASH = _classifier_hash()


def _unique_pins(pins: Iterable[ArtifactPinV1]) -> tuple[ArtifactPinV1, ...]:
    seen: set[tuple[str, str, str]] = set()
    result: list[ArtifactPinV1] = []
    for pin in pins:
        key = (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)
        if key not in seen:
            seen.add(key)
            result.append(pin)
    return tuple(result)


class CandidateAdmissionClassifier:
    """Classify candidate deltas without acquiring admission authority."""

    classifier_version = CLASSIFIER_VERSION
    classifier_hash = CLASSIFIER_HASH

    def classify_text(
        self,
        *,
        candidate: ArtifactPinV1,
        candidate_text: str,
        evidence_refs: tuple[ArtifactPinV1, ...] = (),
    ) -> CandidateAdmissionClassificationV1:
        text = candidate_text.strip().lower()

        authority_dimensions: set[AdmissionDimension] = set()
        if _has_any(text, _AUTHORITY_OWNER_PHRASES):
            authority_dimensions.add("D8")
        if _has_any(text, _ACTIVATION_PHRASES):
            authority_dimensions.add("D9")
        if _has_any(text, _STRUCTURAL_MISSING_PHRASES) and _has_any(
            text, _STRUCTURAL_BYPASS_PHRASES
        ):
            authority_dimensions.update(("D4", "D8"))
        if authority_dimensions:
            return self._result(
                candidate=candidate,
                disposition="AUTHORITY_VIOLATION",
                dimensions=authority_dimensions,
                evidence_refs=evidence_refs,
                reason_code="AUTHORITY_BOUNDARY_BYPASS",
            )

        semantic_dimensions: set[AdmissionDimension] = set()
        for dimension, phrases in _SEMANTIC_RULES:
            if _has_any(text, phrases):
                semantic_dimensions.add(dimension)
        if semantic_dimensions:
            return self._result(
                candidate=candidate,
                disposition="SEMANTIC_CHANGE_REQUIRES_ADMISSION",
                dimensions=semantic_dimensions,
                evidence_refs=evidence_refs,
                reason_code="SEMANTIC_DIMENSION_CHANGED",
            )

        if _has_any(text, _PROCEDURAL_MARKERS) or _has_any(
            text, _PRESERVATION_MARKERS
        ):
            return self._result(
                candidate=candidate,
                disposition="PROCEDURAL_SAFE_TO_PROJECT",
                dimensions=set(),
                evidence_refs=evidence_refs,
                reason_code="EXECUTION_PROCEDURE_ONLY",
            )

        return self._result(
            candidate=candidate,
            disposition="INSUFFICIENT_EVIDENCE",
            dimensions=set(),
            evidence_refs=evidence_refs,
            reason_code="OPERATIVE_CHANGE_NOT_ESTABLISHED",
        )

    def classify(
        self,
        *,
        candidate: ChangeCandidate | ChangeCandidateV2,
    ) -> CandidateAdmissionClassificationV1:
        candidate_text, evidence_refs = self._candidate_payload(candidate)
        return self.classify_text(
            candidate=candidate.pin(),
            candidate_text=candidate_text,
            evidence_refs=evidence_refs,
        )

    @staticmethod
    def _candidate_payload(
        candidate: ChangeCandidate | ChangeCandidateV2,
    ) -> tuple[str, tuple[ArtifactPinV1, ...]]:
        if isinstance(candidate, ChangeCandidate):
            return candidate.delta.description, _unique_pins(
                (*candidate.delta.basis_refs, *candidate.origin.evidence)
            )

        profile = candidate.profile
        if isinstance(profile, LearningHypothesisProfileV2):
            return profile.delta.description, _unique_pins(
                (*profile.delta.basis_refs, *profile.origin.evidence)
            )
        if isinstance(profile, AtomicChangeProfileV1):
            return profile.rationale, _unique_pins(profile.basis_refs)
        if isinstance(profile, ChangeSetProfileV1):
            return profile.rationale, _unique_pins(
                (*profile.basis_refs, *profile.coherence_refs)
            )
        raise TypeError("unsupported ChangeCandidateV2 profile")

    def _result(
        self,
        *,
        candidate: ArtifactPinV1,
        disposition: str,
        dimensions: set[AdmissionDimension],
        evidence_refs: tuple[ArtifactPinV1, ...],
        reason_code: str,
    ) -> CandidateAdmissionClassificationV1:
        ordered = tuple(
            sorted(dimensions, key=lambda value: int(value.removeprefix("D")))
        )
        return CandidateAdmissionClassificationV1(
            candidate=candidate,
            disposition=disposition,
            changed_dimensions=ordered,
            classifier_version=self.classifier_version,
            classifier_hash=self.classifier_hash,
            evidence_refs=_unique_pins(evidence_refs),
            reason_code=reason_code,
        )


__all__ = [
    "CLASSIFIER_HASH",
    "CLASSIFIER_VERSION",
    "CandidateAdmissionClassifier",
]
