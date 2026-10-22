"""Narrow deterministic slash adapter. Never bridges terminal or model handlers."""
from __future__ import annotations
import shlex
from urllib.parse import urlencode

from eos.artifact_hashing import content_hash


def run_review_command(context, service, message, request_id=None):
    try:
        words = shlex.split(message)
    except ValueError as exc:
        raise ValueError('INVALID_SKILLS_REVIEW_COMMAND') from exc
    if not words or words[0] != '/skills':
        raise ValueError('INVALID_SKILLS_REVIEW_COMMAND')
    if len(words) == 1:
        return {'ok': True, 'navigate_to': '/skills', 'model_ran': False}
    command = words[1]
    if command not in ('pending', 'diff', 'approve', 'reject'):
        raise ValueError('UNSUPPORTED_SKILLS_REVIEW_COMMAND')
    if command == 'pending':
        if len(words) != 2:
            raise ValueError('INVALID_SKILLS_REVIEW_COMMAND')
        pending = service._intake_owner().pending(context)
        return {'ok': True, 'objects': [service._present(item, 'pending') for item in pending],
                'navigate_to': '/learning', 'model_ran': False}
    if not 3 <= len(words) <= 4:
        raise ValueError('EXACT_PENDING_ID_REQUIRED')
    if command in ('approve', 'reject') and len(words) != 4:
        raise ValueError('EXACT_PENDING_PAYLOAD_HASH_REQUIRED')
    pending = service._intake_owner().pending(context)
    item = next((item for item in pending if item['native_id'] == words[2]), None)
    if item is None:
        raise LookupError('PENDING_NOT_FOUND')
    digest = words[3] if len(words) == 4 else item['payload_hash']
    detail = service._intake_owner().inspect_pending(context, words[2], digest)
    if command == 'diff':
        return {'ok': True, 'object': service._present(detail, 'pending'), 'model_ran': False,
            'navigate_to': '/learning?'+urlencode({'kind':'pending','ref':words[2],'payload_hash':digest})}
    # Admission is possible only for the canonical candidate explicitly imported
    # from this reviewed source revision. Native raw apply is never called.
    candidates = service._procedure_owner().list_candidates(context)
    candidates = [candidate for candidate in candidates if isinstance(candidate.get('origin'), dict)
        and candidate['origin'].get('native_id') == words[2]
        and candidate['origin'].get('payload_hash') == digest]
    if len(candidates) != 1:
        raise ValueError('EXACT_CANONICAL_REVIEW_REQUIRED')
    candidate = candidates[0]
    values = {'candidate_ref':candidate['ref'], 'candidate_hash':candidate['hash'],
        'request_id': request_id if isinstance(request_id, str) and request_id else
            'slash:'+content_hash({'command':message,'actor':context.user_id,'workspace':context.workspace_id})}
    operation = 'reject_candidate' if command == 'reject' else 'approve_candidate'
    if command == 'approve':
        if not candidate.get('assessment_ref'):
            raise ValueError('EXACT_ASSESSMENT_REQUIRED')
        values.update(assessment_ref=candidate['assessment_ref'], assessment_hash=candidate['assessment_hash'])
    result = service.act(context=context, operation=operation, values=values)
    return {'ok':True, **result, 'model_ran':False,
            'message':'Admitted only. Explicit activation and publication remain separate.' if command=='approve' else 'Rejected; source and history retained.'}


def maybe_learning_review_command(context, message, request_id=None):
    if message.split(maxsplit=1)[0:1] != ['/skills']:
        return None
    from fastapi import HTTPException
    from eos.db import init_db
    from storage.governance_store import GovernanceStore, _db_path
    from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path
    from services.learning_workbench_service import LearningWorkbenchService
    init_db(_db_path())
    store = OntologyBuildStore(ontology_build_store_path(context.workspace_root))
    governance = GovernanceStore()
    try:
        return run_review_command(context, LearningWorkbenchService(store=store,governance=governance), message, request_id)
    except PermissionError as exc:
        raise HTTPException(403,'Outside current learning scope') from exc
    except LookupError as exc:
        raise HTTPException(404,'Pending review not found') from exc
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc
    finally:
        governance.close()
        store.close()
