"""Deterministic assembly for the repo-owned runtime hypothesis prompt."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping
from contracts.decision_local_projection_v2 import DecisionLocalProjectionV2


PROMPT_ASSET_ID = "runtime_semantic_hypothesis.v2"
PROMPT_ASSET_PATH = "runtime_semantic_hypothesis_v2.txt"
_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / PROMPT_ASSET_PATH


@dataclass(frozen=True)
class RuntimeSemanticHypothesisPromptAsset:
    asset_id: str
    relative_name: str
    text: str
    sha256: str


@dataclass(frozen=True)
class AssembledRuntimeSemanticHypothesisPrompt:
    prompt_asset_ref: str
    prompt_asset_hash: str
    rendered_text: str
    input_text: str


def load_runtime_semantic_hypothesis_prompt() -> RuntimeSemanticHypothesisPromptAsset:
    text = _PROMPT_PATH.read_text(encoding="utf-8")
    return RuntimeSemanticHypothesisPromptAsset(
        asset_id=PROMPT_ASSET_ID,
        relative_name=PROMPT_ASSET_PATH,
        text=text,
        sha256=sha256(text.encode("utf-8")).hexdigest(),
    )


def assemble_runtime_semantic_hypothesis_prompt(
    *,
    projection: DecisionLocalProjectionV2,
    output_schema: Mapping[str, Any],
    expected_prompt_asset_hash: str | None = None,
) -> AssembledRuntimeSemanticHypothesisPrompt:
    asset = load_runtime_semantic_hypothesis_prompt()
    asset_hash = "sha256:" + asset.sha256
    if expected_prompt_asset_hash is not None and expected_prompt_asset_hash != asset_hash:
        raise ValueError("runtime hypothesis prompt hash pin mismatch")
    inputs = {
        "decision_projection_ref": projection.projection_ref,
        "decision_projection_hash": projection.projection_hash,
        "decision_input": projection.model_input_payload(),
        "schema_contract": projection_schema_contract(projection, output_schema),
    }
    rendered = asset.text
    input_text = json.dumps(
        inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n"
    return AssembledRuntimeSemanticHypothesisPrompt(
        prompt_asset_ref=PROMPT_ASSET_ID,
        prompt_asset_hash=asset_hash,
        rendered_text=rendered,
        input_text=input_text,
    )


def projection_schema_contract(projection: DecisionLocalProjectionV2, output_schema: Mapping[str, Any]) -> dict:
    """Resolve vocabulary from the exact rule retained inside the projection."""
    ref_fields = ("target_type_ref", "attribute_ref", "operator_ref", "modality_ref", "stage_ref", "consequence_ref")
    patterns = [rule["applicability_scope"]["canonical_restriction_pattern"]
                for rule in projection.governed_rules
                if rule["applicability_scope"].get("source_graph_pattern_id") == projection.canonical_pattern_ref]
    if len(patterns) != 1:
        raise ValueError("bounded projection requires one canonical rule")
    frames = [{field: frame[field] for field in sorted((*ref_fields, "value_constraint"))}
              for frame in patterns[0]["frames"]]
    return {"output_schema": dict(output_schema), "allowed_candidate_frames": frames,
            "allowed_frame_refs": {field: sorted({frame[field] for frame in frames}) for field in ref_fields}}


__all__ = [
    "AssembledRuntimeSemanticHypothesisPrompt",
    "PROMPT_ASSET_ID",
    "PROMPT_ASSET_PATH",
    "RuntimeSemanticHypothesisPromptAsset",
    "assemble_runtime_semantic_hypothesis_prompt",
    "load_runtime_semantic_hypothesis_prompt",
]
