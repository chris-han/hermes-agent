"""Read completed GitHub provenance; native receipt HMAC alone is insufficient."""
import os
import re
import httpx
from .publication import OperationError


def verify_workflow(receipt):
    run=str(receipt.get('workflow_run_id',''))
    attempt=receipt.get('workflow_run_attempt')
    if not re.fullmatch('[1-9][0-9]{0,19}',run) or type(attempt) is not int or not 1<=attempt<=999999999:
        raise OperationError(422,'github_run_identity_required')
    token=os.environ.get('SEMANTIER_CLIENT_GITHUB_TOKEN','')
    headers={'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'}
    if token:headers['Authorization']='Bearer '+token
    with httpx.Client(timeout=15,follow_redirects=False,trust_env=False) as client:
        response=client.get(f'https://api.github.com/repos/chris-han/hermes-agent/actions/runs/{run}/attempts/{attempt}',headers=headers)
    if response.status_code!=200:raise OperationError(503,'github_provenance_unavailable')
    value=response.json()
    if value.get('status')!='completed':raise OperationError(503,'github_run_pending')
    if (value.get('conclusion')!='success' or value.get('head_sha')!=receipt['source_commit']
        or value.get('head_repository',{}).get('full_name')!='chris-han/hermes-agent'
        or not (value.get('event')=='workflow_dispatch' or (value.get('event')=='push' and value.get('head_branch')=='staging/client-release-feedback-mvl-v1'))
        or value.get('path')!='.github/workflows/semantier-windows-acceptance.yml'
        or value.get('run_attempt')!=receipt.get('workflow_run_attempt')):
        raise OperationError(422,'github_provenance_mismatch')
    return {'run_id':value['id'],'run_attempt':value['run_attempt'],'head_sha':value['head_sha'],
            'path':value['path'],'event':value['event'],'branch':value.get('head_branch'),'conclusion':'success','html_url':value['html_url']}


def reconcile_candidates(owner):
    from contracts.client_operations import ReleaseCandidate
    # Only a CI-held receipt key can make these staged inputs eligible. No head
    # is changed: an operator must still intentionally promote in Cloud.
    for path in sorted((owner.publisher.incoming/'ci-candidates').glob('*.json')):
        if path.is_symlink() or path.stat().st_size>1024*1024:continue
        try:candidate=ReleaseCandidate.model_validate_json(path.read_bytes())
        except ValueError:
            continue  # Untrusted malformed stage is never admitted; other maintenance continues.
        with owner.store.connection() as c:
            if c.execute('SELECT 1 FROM releases WHERE release_id=%s',(candidate.release_id,)).fetchone():continue
        try:owner.publisher.admit(candidate,{'principal_id':'github-actions:'+str(candidate.native_receipt.get('workflow_run_id',''))})
        except OperationError as error:
            if error.status!=503:
                owner.publisher.record_rejection(candidate,error.code,'ci-reconciler')
