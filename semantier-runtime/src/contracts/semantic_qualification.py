from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


class QualificationLayer(str, Enum):
    PROJECTION = "PROJECTION"
    FRONTIER = "FRONTIER"
    CONTRACTION = "CONTRACTION"
    VERIFICATION = "VERIFICATION"
    AUTHORITY = "AUTHORITY"


LAYER_ORDER: tuple[QualificationLayer, ...] = tuple(QualificationLayer)


class QualificationState(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN_PRIMARY_PREREQUISITE_FAILED = "NOT_RUN_PRIMARY_PREREQUISITE_FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class LayerQualificationResult:
    state: QualificationState
    failure_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class QualificationGates:
    g1_structural_admissibility: bool
    g2_semantic_qualification: bool
    g3_verified_decision: bool
    g4_governed_activation_eligibility: bool


@dataclass(frozen=True)
class FiveLayerQualificationResult:
    case_id: str
    fixture_revision: str
    projection_profile: str
    frontier_contract: str
    layer_results: Mapping[QualificationLayer, LayerQualificationResult]
    primary_failure_layer: QualificationLayer | None
    failure_labels: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    replay_identity: str
    gates: QualificationGates


def normalize_evidence_refs(refs: Sequence[str]) -> tuple[str, ...]:
    return tuple(refs)
