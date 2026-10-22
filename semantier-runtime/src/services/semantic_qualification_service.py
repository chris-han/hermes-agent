from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from src.contracts.semantic_qualification import (
    LAYER_ORDER,
    FiveLayerQualificationResult,
    LayerQualificationResult,
    QualificationGates,
    QualificationLayer,
    QualificationState,
)


def reduce_five_layer_qualification(
    *,
    case_id: str,
    fixture_revision: str,
    projection_profile: str,
    frontier_contract: str,
    layer_results: Mapping[QualificationLayer, LayerQualificationResult],
    evidence_refs: Sequence[str] = (),
) -> FiveLayerQualificationResult:
    normalized: dict[QualificationLayer, LayerQualificationResult] = {}
    primary_failure: QualificationLayer | None = None

    for layer in LAYER_ORDER:
        supplied = layer_results.get(layer)
        if primary_failure is not None:
            normalized[layer] = LayerQualificationResult(
                QualificationState.NOT_RUN_PRIMARY_PREREQUISITE_FAILED
            )
            continue
        if supplied is None:
            raise ValueError(f"missing layer result before primary failure: {layer.value}")
        normalized[layer] = supplied
        if supplied.state is QualificationState.FAIL:
            primary_failure = layer

    failure_labels = tuple(
        label
        for layer in LAYER_ORDER
        for label in normalized[layer].failure_labels
        if normalized[layer].state is QualificationState.FAIL
    )
    g1 = (
        normalized[QualificationLayer.PROJECTION].state is QualificationState.PASS
        and normalized[QualificationLayer.FRONTIER].state is QualificationState.PASS
    )
    g2 = g1 and normalized[QualificationLayer.CONTRACTION].state is QualificationState.PASS
    g3 = g2 and normalized[QualificationLayer.VERIFICATION].state is QualificationState.PASS
    g4 = g3 and normalized[QualificationLayer.AUTHORITY].state is QualificationState.PASS
    gates = QualificationGates(g1, g2, g3, g4)
    refs = tuple(evidence_refs)

    replay_bytes = _serialize(
        case_id,
        fixture_revision,
        projection_profile,
        frontier_contract,
        normalized,
        primary_failure,
        failure_labels,
        refs,
        gates,
    )
    return FiveLayerQualificationResult(
        case_id=case_id,
        fixture_revision=fixture_revision,
        projection_profile=projection_profile,
        frontier_contract=frontier_contract,
        layer_results=normalized,
        primary_failure_layer=primary_failure,
        failure_labels=failure_labels,
        evidence_refs=refs,
        replay_identity=hashlib.sha256(replay_bytes).hexdigest(),
        gates=gates,
    )


def serialize_five_layer_qualification(result: FiveLayerQualificationResult) -> bytes:
    return _serialize(
        result.case_id,
        result.fixture_revision,
        result.projection_profile,
        result.frontier_contract,
        result.layer_results,
        result.primary_failure_layer,
        result.failure_labels,
        result.evidence_refs,
        result.gates,
    )


def _serialize(
    case_id,
    fixture_revision,
    projection_profile,
    frontier_contract,
    layer_results,
    primary_failure_layer,
    failure_labels,
    evidence_refs,
    gates,
) -> bytes:
    payload = {
        "case_id": case_id,
        "fixture_revision": fixture_revision,
        "projection_profile": projection_profile,
        "frontier_contract": frontier_contract,
        "layer_results": {
            layer.value: {
                "state": layer_results[layer].state.value,
                "failure_labels": list(layer_results[layer].failure_labels),
            }
            for layer in LAYER_ORDER
        },
        "primary_failure_layer": (
            primary_failure_layer.value if primary_failure_layer is not None else None
        ),
        "failure_labels": list(failure_labels),
        "evidence_refs": list(evidence_refs),
        "gates": {
            "G1": gates.g1_structural_admissibility,
            "G2": gates.g2_semantic_qualification,
            "G3": gates.g3_verified_decision,
            "G4": gates.g4_governed_activation_eligibility,
        },
    }
    return (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
