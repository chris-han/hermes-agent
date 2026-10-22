"""Derived serializer artifacts bound to real V2 and typed-frontier validators."""
from hashlib import sha256

from contracts.decision_local_projection_v2 import DecisionLocalProjectionV2
from contracts.typed_decision_model import TypedDecisionFrontierV1, TypedDecisionModelInputV1
from src.research.tender_small_dlm.context_envelope import canonical, content_hash
from src.research.tender_dlm_serializer.serializer import serialize, deserialize, verify_closure

# Categories are paths into exact inherited semantics, not newly inferred facts.
# Exceptions/qualifications/counter-evidence in this basis remain source prose.
CATEGORY_PATHS = {
    'decision_intent': ['/query', '/decision_contract/semantic_predicates'],
    'frontier': ['/typed_decision_frontier'],
    'operative_facts': ['/dlp_model_input/assertions'],
    'required_relations': ['/dlp_model_input/assertions'],
    'counter_evidence': ['/dlp_model_input/source_excerpts'],
    'exceptions': ['/dlp_model_input/source_excerpts', '/dlp_model_input/assertions'],
    'qualifications': ['/dlp_model_input/source_excerpts', '/dlp_model_input/assertions'],
    'temporal_conditions': ['/dlp_model_input/world_time_basis', '/dlp_model_input/source_excerpts'],
    'authority': ['/decision_contract/deterministic_constraints', '/source_identity'],
    'dependency_status': ['/dlp_model_input/material_dependency_resolutions', '/decision_contract/semantic_predicates'],
}


def validate_basis(original, projection):
    dlp = DecisionLocalProjectionV2.model_validate(projection)
    frontier = TypedDecisionFrontierV1.from_projection(dlp)
    expected = TypedDecisionModelInputV1.from_projection(dlp, frontier).model_dump(mode='json')
    if canonical({k: original[k] for k in expected}) != canonical(expected):
        raise ValueError('SOURCE_DLP_FRONTIER_BINDING_MISMATCH')
    if canonical(original['decision_contract']) != canonical(dlp.decision_contract.model_dump(mode='json')):
        raise ValueError('DECISION_CONTRACT_CHANGED')
    return dlp


def compile_view(original, projection):
    dlp = validate_basis(original, projection)
    body = serialize(original)
    closure = verify_closure(original, body)
    artifact = {
        'view_version': 'DecisionOptimizedDLMViewV2',
        'source_dlp_ref': dlp.projection_ref, 'source_dlp_hash': dlp.projection_hash,
        'working_context_ref': dlp.working_context.pin().artifact_ref,
        'working_context_hash': dlp.working_context.pin().artifact_hash,
        'semantic_categories': CATEGORY_PATHS,
        'condition_representation': 'exact inherited source spans and assertion labels; no new semantic extraction',
        'model_visible_body': body,
        'model_visible_sha256': sha256(body.encode()).hexdigest(),
        'replay_only_metadata': {
            'original_projection_body_hash': content_hash(projection),
            'original_input_hash': content_hash(original),
            'evidence_closure_hash': content_hash(projection['evidence']),
            'frontier_hash': original['typed_decision_frontier']['frontier_hash'],
            'world_time_hash': content_hash(original['dlp_model_input']['world_time_basis']),
        },
        'closure_hash': content_hash(closure),
    }
    artifact['content_hash'] = content_hash(artifact)
    return artifact, closure


def replay_view(artifact, original, projection):
    expected, closure = compile_view(original, projection)
    if canonical(artifact) != canonical(expected):
        raise ValueError('REPLAY_IDENTITY_FAILURE')
    # Independently run the existing deterministic input/frontier validators on
    # decoded model-visible S1, not on the original input alone.
    reconstructed = deserialize(artifact['model_visible_body'])
    before = validate_basis(original, projection)
    after = validate_basis(reconstructed, projection)
    if before != after:
        raise ValueError('DETERMINISTIC_VERIFIER_PARITY_FAILURE')
    return closure
