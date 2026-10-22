"""Resolve a browser presentation selection against the packaged frozen specimen.

This prompt asset is an exact copy of the frontend educational fixture, not a
knowledge admission or arbitrary-text validator. Tests bind the two copies.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

_ASSETS = Path(__file__).resolve().parents[1] / 'prompts' / 'agents'
_SPECIMEN = json.loads((_ASSETS / 'semantic_projection_specimen.v1.json').read_text(encoding='utf-8'))
SEMANTIC_PROJECTION_POLICY = (_ASSETS / 'SEMANTIC_PROJECTION_CONTEXT.md').read_text(encoding='utf-8').strip()


def resolve_semantic_projection_context(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError('recorded semantic projection context required')
    profile = raw.get('profile')
    locale = raw.get('interfaceLocale')
    fault_id = raw.get('selectedFault')
    p = _SPECIMEN
    surface = next((x for x in p['valid_projections'] if x['profile'] == profile), None)
    drift = next((x for x in p['fault_injections'] if x['fault'] == fault_id), None)
    if not surface or locale not in ('en', 'zh') or (fault_id is not None and (not drift or profile != 'controlled_zh')):
        raise ValueError('invalid recorded specimen selection')
    ir = p['decision_ir']
    expected = {
        'schemaVersion': 'semantic_projection_context.v1', 'specimenRef': 'controlled-language-semantic-projection.v1', 'specimenVersion': p['schema_version'], 'evidenceHash': p['evidence_hash'], 'decisionId': ir['decision_id'], 'frozenSemanticHash': p['decision_semantic_hash'],
        'sourceRuleRef': ir['source_rule_frame_ref'], 'sourceRuleHash': ir['source_rule_frame_hash'], 'sourceExceptionRef': ir['source_exception_frame_ref'], 'sourceExceptionHash': ir['source_exception_frame_hash'],
        'profile': profile, 'profileSurfaceHash': surface['surface_hash'], 'selectedFault': fault_id, 'displayedText': (drift or surface)['surface_text'], 'observedSemanticHash': (drift or surface)['semantic_hash'], 'mismatches': drift['mismatches'] if drift else [], 'recordedReplay': p['replay'], 'validationScope': 'recorded_specimen_only', 'interfaceLocale': locale,
    }
    if raw != expected:
        raise ValueError('stale or modified recorded specimen context')
    return expected
