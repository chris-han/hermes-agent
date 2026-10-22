"""Core authorization of the bundled Meeting workflow's first installation.

Installation is exact-seed qualification, never a learned improvement claim.
Publication remains with the immutable workflow definition owner.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

from agents.gateway_identity import get_user_record
from contracts.evidence_location import EvidenceSelector
from contracts.semantic_kernel import Evidence
from contracts.workflow_execution import hash_value, workflow_definition_hash, meeting_owner_directory
from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path

EMPTY_WORKFLOW_HASH = 'sha256:' + '0' * 64
ROLE = 'workflow_initial_installation'
# Closed repo-owned contract asset is loaded with the core module at server startup.
_MANIFEST = json.loads((Path(__file__).parents[1] / 'contracts/meeting-workflow-installation.v1.json').read_text())


def manifest():
    return dict(_MANIFEST)


def _owner(ctx):
    user = get_user_record(ctx.user_id) if ctx.authenticated else None
    if (not user or user.get('workspace_id') != ctx.workspace_id
            or not ctx.organization_id):
        raise PermissionError('WORKFLOW_INSTALLATION_OWNER_REQUIRED')


def _seed(definition):
    expected = manifest()
    if (hash_value(definition) != expected['definitionArtifactHash']
            or workflow_definition_hash(definition, include_labels=False) != expected['workflowHash']):
        raise ValueError('WORKFLOW_INSTALLATION_SEED_MISMATCH')
    return expected


def _source(actor, workflow_hash):
    return f'workflow-installation:feishu-meeting-coordinator:{actor}:{workflow_hash}'


def assess(ctx, definition):
    _owner(ctx)
    expected = _seed(definition)
    directory = meeting_owner_directory(ctx)
    if (directory / 'head.json').exists() or any((directory / 'versions').glob('*.json')):
        raise ValueError('WORKFLOW_INSTALLATION_HISTORY_EXISTS')
    now = datetime.now(timezone.utc)
    source = _source(ctx.user_id, expected['workflowHash'])
    selector = EvidenceSelector(schema_version='semantier.evidence_selector.v1',
        selector_kind='structure', exact_quote=None, prefix=None, suffix=None,
        normalized_quote=None, table=None, source_element_ref=source,
        structural_path=['workflow-installation.v1', ctx.user_id, expected['workflowHash'], expected['compilerVersion']])
    evidence = Evidence.create(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
        source_ref=source, source_hash=expected['definitionArtifactHash'], selector=selector,
        selector_hash=selector.selector_hash(), evidence_role=ROLE, provenance=(),
        observed_at=now, source_time=None, recorded_at=now)
    store = OntologyBuildStore(ontology_build_store_path(ctx.workspace_root))
    try: store.put(evidence)
    finally: store.close()
    return {'assessment': evidence.pin().model_dump(mode='json'),
        'workflowHash': expected['workflowHash'], 'assessedAt': now.isoformat().replace('+00:00', 'Z'),
        'disposition': 'QUALIFIED_FOR_INITIAL_INSTALLATION'}


def verify(ctx, definition, pin, *, actor_ref=None, current_owner=True):
    if current_owner:
        _owner(ctx)
        _seed(definition)
    if not ctx.authenticated or not ctx.organization_id:
        raise PermissionError('WORKFLOW_INSTALLATION_SCOPE_REQUIRED')
    actor = ctx.user_id if current_owner else str(actor_ref or '').removeprefix('user:')
    semantic_hash = workflow_definition_hash(definition, include_labels=False)
    if pin.get('artifact_kind') != 'evidence':
        raise ValueError('WORKFLOW_INSTALLATION_ASSESSMENT_REQUIRED')
    store = OntologyBuildStore(ontology_build_store_path(ctx.workspace_root))
    try:
        evidence = store.get(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
            artifact_ref=pin.get('artifact_ref', ''), contract_type=Evidence)
    finally: store.close()
    if (not evidence or evidence.pin().model_dump(mode='json') != pin
            or evidence.evidence_role != ROLE or evidence.source_hash != hash_value(definition)
            or evidence.source_ref != _source(actor, semantic_hash)
            or evidence.selector.structural_path != ['workflow-installation.v1', actor, semantic_hash, 'workflow-execution-compiler/v2']):
        raise ValueError('WORKFLOW_INSTALLATION_ASSESSMENT_MISMATCH')
    return {'verified': True, 'assessment': pin, 'workflowHash': semantic_hash,
        'assessedAt': evidence.recorded_at.isoformat().replace('+00:00', 'Z'),
        'disposition': 'QUALIFIED_FOR_INITIAL_INSTALLATION'}
