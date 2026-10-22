"""State Evolution invocation/result boundary over existing producer/artifact owners."""
from contextlib import contextmanager

from contracts.state_evolution import evolution_hash, verify_evolution_plan, verify_evolution_run


def project_meeting_result(trace, plan, *, attempt_id, recorded_at, reducer_projection):
    """Normalize an owner-produced trace; this function grants no receipt trust.

    Fixture timestamps are ordering coordinates; domain time must be explicit.
    Persistence and authenticated source/producer verification are separate.
    """
    verified = verify_evolution_plan(plan)
    if verified.profile != 'WORKFLOW_ONLY':
        raise ValueError('UNSUPPORTED: Meeting projection profile')
    projected_events = reducer_projection.get('events') if isinstance(reducer_projection, dict) else None
    final_state = reducer_projection.get('finalState') if isinstance(reducer_projection, dict) else None
    workflow_outputs = {key for key in plan['sourceMap'] if key == 'workflow.run-status' or key.startswith('workflow.node-status.')}
    if (not isinstance(projected_events, list) or len(projected_events) != len(trace['steps']) or
            not isinstance(final_state, dict) or set(final_state) != workflow_outputs):
        raise ValueError('INVALID_REDUCER_PROJECTION')
    events = []
    for index, step in enumerate(trace['steps']):
        reduced = projected_events[index]
        if not isinstance(reduced, dict) or reduced.get('id') != step.get('eventId') or not isinstance(reduced.get('deltas'), list):
            raise ValueError('INVALID_REDUCER_PROJECTION')
        semantic_category = reduced.get('semanticCategory')
        if semantic_category not in ('WORKFLOW_STATE', 'OBSERVATION'):
            raise ValueError('INVALID_REDUCER_PROJECTION')
        if semantic_category == 'OBSERVATION' and reduced['deltas']:
            raise ValueError('INVALID_REDUCER_PROJECTION')
        if any(not isinstance(delta, dict) or delta.get('symbolId') not in workflow_outputs for delta in reduced['deltas']):
            raise ValueError('INVALID_REDUCER_PROJECTION')
        payload = dict(step)
        if index == 0 and isinstance(trace.get('participants'), list):
            # Keep source-owner participant identities in the sealed event envelope once;
            # per-event participant states remain on their corresponding producer events.
            payload['participantRoster'] = trace['participants']
        for envelope_field in ('semanticTime', 'microstep', 'order', 'producerOperation'):
            payload.pop(envelope_field, None)
        timestamp = payload.pop('occurredAt', None)
        if timestamp is not None:
            payload['orderingFixtureTimestamp'] = timestamp
        semantic_time = step.get('semanticTime')
        if plan['timeBasis']['kind'] == 'domain' and semantic_time is None:
            raise ValueError('UNBOUND_DOMAIN_EVENT_TIME')
        events.append({'id':step['eventId'],'kind':step['eventType'],'semanticCategory':semantic_category,
                       **({'operatorRef':step['producerOperation']} if isinstance(step.get('producerOperation'),str) else {}),
                       'semanticTime':semantic_time,'microstep':step.get('microstep',0),'order':step.get('order',index),
                       'sourceRefs':[plan['irRef'],plan['producer']],
                       'deltas':reduced['deltas'],'payload':payload})
    run = {'schemaVersion':'evolution_run.v1','profile':'WORKFLOW_ONLY','scope':plan['scope'],
           'planRef':{'artifact_kind':'evolution_execution_plan','artifact_ref':'evolution-plan:'+plan['basisHash'].split(':')[1],
                      'artifact_hash':evolution_hash(plan)},
           'basisHash':plan['basisHash'],'producer':plan['producer'],'recordedAt':recorded_at,'attemptId':attempt_id,
           'status':'SIMULATED','events':events,'series':{},'finalState':final_state,
           'diagnostics':[{'code':'OWNER_TRACE','semanticTraceHash':trace['semanticTraceHash'],
                           'claimBoundary':'SIMULATED; fixture ordering timestamps are not elapsed domain time.'}]}
    run['resultHash'] = evolution_hash({key:run[key] for key in ('status','events','series','finalState','diagnostics')})
    run['runId'] = evolution_hash(run)
    verify_evolution_run(run,plan,trusted_scope=plan['scope'])
    return run


def project_continuous_result(owner_run, plan, *, attempt_id, recorded_at):
    """Normalize one qualified deterministic SD run into a sealed EvolutionRun."""
    verified = verify_evolution_plan(plan)
    if verified.profile != 'CONTINUOUS_ONLY':
        raise ValueError('UNSUPPORTED: continuous projection profile')
    if owner_run.get('schemaVersion') != 'system-dynamics-run.v1':
        raise ValueError('INVALID_CONTINUOUS_OWNER_RUN')
    if owner_run.get('scenarioId') != plan['scenario']['id']:
        raise ValueError('CONTINUOUS_SCENARIO_MISMATCH')
    time = owner_run.get('time')
    if not isinstance(time, list) or not time:
        raise ValueError('INVALID_CONTINUOUS_OWNER_TIME')
    series = {}
    for family in ('stocks', 'flows', 'auxiliaries'):
        values = owner_run.get(family)
        if not isinstance(values, dict):
            raise ValueError('INVALID_CONTINUOUS_OWNER_SERIES')
        for ident, value in values.items():
            if ident not in plan['sourceMap'] or not isinstance(value, dict) or not isinstance(value.get('values'), list):
                raise ValueError('INVALID_CONTINUOUS_OWNER_SERIES')
            series[ident] = {**value, 'family': family}
    final_state = {}
    for ident, value in (owner_run.get('stocks') or {}).items():
        points = value.get('values') or []
        if not points:
            raise ValueError('INVALID_CONTINUOUS_OWNER_SERIES')
        final_state[ident] = {'status': 'KNOWN', 'value': points[-1]}
    diagnostics = [{
        'code': 'OWNER_NUMERICAL_RUN',
        'ownerRunId': owner_run.get('runId'),
        'engine': owner_run.get('engine'),
        'time': time,
        'ownerDiagnostics': owner_run.get('diagnostics') or [],
        'claimBoundary': 'SIMULATED: sealed deterministic System Dynamics owner output.',
    }]
    run = {
        'schemaVersion': 'evolution_run.v1', 'profile': 'CONTINUOUS_ONLY',
        'scope': plan['scope'],
        'planRef': {'artifact_kind': 'evolution_execution_plan',
                    'artifact_ref': 'evolution-plan:' + plan['basisHash'].split(':')[1],
                    'artifact_hash': evolution_hash(plan)},
        'basisHash': plan['basisHash'], 'producer': plan['producer'],
        'recordedAt': recorded_at, 'attemptId': attempt_id,
        'status': 'SIMULATED', 'events': [], 'series': series, 'finalState': final_state,
        'diagnostics': diagnostics,
    }
    if plan.get('comparisonBasisRef') is not None:
        run['comparisonBasisRef'] = plan['comparisonBasisRef']
    run['resultHash'] = evolution_hash({key: run[key] for key in ('status','events','series','finalState','diagnostics')})
    run['runId'] = evolution_hash(run)
    verify_evolution_run(run, plan, trusted_scope=plan['scope'])
    return run


def project_hybrid_result(owner_result, plan, *, attempt_id, recorded_at):
    """Normalize the one reducer+SD coordinator result into a source-mapped run."""
    verified = verify_evolution_plan(plan)
    if verified.profile != 'HYBRID_WORKFLOW_CONTINUOUS':
        raise ValueError('UNSUPPORTED: hybrid projection profile')
    if not isinstance(owner_result, dict) or owner_result.get('status') != 'SIMULATED':
        raise ValueError('INVALID_HYBRID_OWNER_RESULT')
    trajectory = owner_result.get('trajectory')
    if not isinstance(trajectory, dict) or not isinstance(trajectory.get('time'), list) or not trajectory['time']:
        raise ValueError('INVALID_HYBRID_TRAJECTORY')
    times = trajectory['time']
    if any(not isinstance(item,(int,float)) for item in times) or any(right <= left for left,right in zip(times,times[1:])):
        raise ValueError('INVALID_HYBRID_TRAJECTORY_TIME')
    series = {}
    for family in ('stocks','flows','auxiliaries'):
        rows = trajectory.get(family)
        if not isinstance(rows,dict):
            raise ValueError('INVALID_HYBRID_TRAJECTORY')
        for ident,value in rows.items():
            if ident not in plan['sourceMap'] or not isinstance(value,dict) or not isinstance(value.get('values'),list) or len(value['values']) != len(times):
                raise ValueError('INVALID_HYBRID_TRAJECTORY_SERIES')
            if any(not isinstance(point,(int,float)) for point in value['values']):
                raise ValueError('INVALID_HYBRID_TRAJECTORY_VALUE')
            series[ident] = {**value,'family':family}
    source_refs = [plan['irRef'],plan['producer'],plan['backends']['workflow'],plan['backends']['continuous']]
    events = []
    for index,item in enumerate(owner_result.get('events') or []):
        ident = item.get('id')
        semantic_time = item.get('semanticTime')
        if not isinstance(ident,str) or not isinstance(semantic_time,(int,float)) or semantic_time < plan['timeBasis']['start'] or semantic_time > plan['timeBasis']['stop']:
            raise ValueError('INVALID_HYBRID_OWNER_EVENT')
        deltas = item.get('deltas')
        if not isinstance(deltas,list) or any(delta.get('symbolId') not in plan['sourceMap'] for delta in deltas):
            raise ValueError('UNBOUND_HYBRID_EVENT_DELTA')
        category = 'MODELED_EVENT' if item.get('kind') == 'SCHEDULED_INPUT' else 'WORKFLOW_STATE'
        if category == 'MODELED_EVENT' and deltas:
            raise ValueError('INVALID_MODELED_INPUT_EVENT')
        events.append({'id':ident,'kind':item.get('kind'),'semanticCategory':category,
            'semanticTime':semantic_time,'microstep':item.get('microstep',0),'order':item.get('order',index),
            'sourceRefs':source_refs,'deltas':deltas,'payload':item.get('payload') or {}})
    workflow_state = owner_result.get('workflowProjection')
    if not isinstance(workflow_state,dict) or not isinstance(workflow_state.get('nodeStates'),dict):
        raise ValueError('INVALID_HYBRID_WORKFLOW_PROJECTION')
    final_state = {'workflow.run-status':{'status':'KNOWN','value':workflow_state['status']}}
    for ident,value in workflow_state['nodeStates'].items():
        final_state['workflow.node-status.'+ident] = {'status':'KNOWN','value':value['status']}
    for ident,value in (owner_result.get('values') or {}).items():
        if ident in plan['sourceMap'] and isinstance(value,(int,float,str,bool)):
            final_state[ident] = {'status':'KNOWN','value':value}
    diagnostics = [{'code':'HYBRID_OWNER_EXECUTION','engine':{'id':'semantier-sd-deterministic','version':'2'},
        'time':times,'claimBoundary':'SIMULATED hybrid event/continuous trajectory; no external effects.'}]
    run = {'schemaVersion':'evolution_run.v1','profile':'HYBRID_WORKFLOW_CONTINUOUS','scope':plan['scope'],
        'planRef':{'artifact_kind':'evolution_execution_plan','artifact_ref':'evolution-plan:'+plan['basisHash'].split(':')[1],
            'artifact_hash':evolution_hash(plan)},'basisHash':plan['basisHash'],'producer':plan['producer'],
        'recordedAt':recorded_at,'attemptId':attempt_id,'status':'SIMULATED','events':events,'series':series,
        'finalState':final_state,'diagnostics':diagnostics}
    run['resultHash'] = evolution_hash({key:run[key] for key in ('status','events','series','finalState','diagnostics')})
    run['runId'] = evolution_hash(run)
    verify_evolution_run(run,plan,trusted_scope=plan['scope'])
    return run


def seal_evolution_result(store, ir, plan, run, *, trusted_scope, resolved_source_pins, trusted_producer):
    """Seal independently resolved basis and producer output in the existing owner."""
    from contracts.state_evolution import EvolutionIRV1, EvolutionExecutionPlanV1, EvolutionRunV1, verify_evolution_scope
    validated_ir = EvolutionIRV1.model_validate(ir)
    verify_evolution_scope(validated_ir.scope, trusted_scope)
    verify_evolution_run(run, plan, trusted_scope=trusted_scope)
    if ir['sourcePins'] != resolved_source_pins or plan['producer'] != trusted_producer:
        raise ValueError('EVOLUTION_UNTRUSTED_SOURCE_OR_PRODUCER')
    ir_hash = evolution_hash(ir)
    if plan['irRef'] != {'artifact_kind':'evolution_ir','artifact_ref':'evolution-ir:'+ir_hash.split(':')[1],'artifact_hash':ir_hash}:
        raise ValueError('EVOLUTION_IR_BINDING_MISMATCH')
    for field in ('profile','scope','timeBasis','scenario'):
        if ir[field] != plan[field]:
            raise ValueError('EVOLUTION_IR_BINDING_MISMATCH')
    if run['planRef']['artifact_ref'] != 'evolution-plan:'+plan['basisHash'].split(':')[1]:
        raise ValueError('EVOLUTION_PLAN_REF_MISMATCH')
    store.put_many((validated_ir, EvolutionExecutionPlanV1.model_validate(plan), EvolutionRunV1.model_validate(run)))
    return run['runId']


def replay_evolution_run(store, run_id, *, trusted_scope):
    """Read sealed artifacts only; no compiler, source resolution or producer."""
    from contracts.state_evolution import EvolutionScopeV1, verify_evolution_scope
    import re
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', run_id):
        raise ValueError('INVALID_RUN_ID')

    def read(ref, schema, expected_hash=None):
        row = store.get_raw(organization_id=trusted_scope['organizationId'], workspace_id=trusted_scope['workspaceId'], artifact_ref=ref)
        if row is None:
            raise ValueError('EVOLUTION_MISSING_ARTIFACT')
        payload = row['payload']
        digest = evolution_hash(payload)
        if row['schema_version'] != schema or payload.get('schemaVersion') != schema or row['artifact_hash'] != digest or (expected_hash is not None and digest != expected_hash):
            raise ValueError('EVOLUTION_ARTIFACT_HASH_MISMATCH')
        verify_evolution_scope(EvolutionScopeV1.model_validate(payload['scope']), trusted_scope)
        return payload

    run = read('evolution-run:'+run_id.split(':')[1], 'evolution_run.v1')
    if run['runId'] != run_id:
        raise ValueError('EVOLUTION_RUN_ID_MISMATCH')
    plan = read(run['planRef']['artifact_ref'], 'evolution_execution_plan.v1', run['planRef']['artifact_hash'])
    if run['planRef']['artifact_ref'] != 'evolution-plan:'+plan['basisHash'].split(':')[1]:
        raise ValueError('EVOLUTION_PLAN_REF_MISMATCH')
    verify_evolution_run(run, plan, trusted_scope=trusted_scope)
    ir = read(plan['irRef']['artifact_ref'], 'evolution_ir.v1', plan['irRef']['artifact_hash'])
    if plan['irRef']['artifact_ref'] != 'evolution-ir:'+evolution_hash(ir).split(':')[1]:
        raise ValueError('EVOLUTION_IR_BINDING_MISMATCH')
    return {'ir':ir, 'plan':plan, 'run':run}


def reexecute_meeting_run(store, run_id, *, trusted_scope, resolved_source_pins,
                         trusted_producer, producer, normalize_workflow_trace, attempt_id, recorded_at):
    """Execute a pinned basis as a new attempt through a trusted owner adapter.

    The authenticated composition root supplies the independently resolved
    source/producer identities and an adapter bound to its governed workspace.
    Replay never calls this operation.
    """
    sealed = replay_evolution_run(store, run_id, trusted_scope=trusted_scope)
    ir, plan, previous = sealed['ir'], sealed['plan'], sealed['run']
    if plan['profile'] != 'WORKFLOW_ONLY':
        raise ValueError('UNSUPPORTED: Meeting re-execution profile')
    if ir['sourcePins'] != resolved_source_pins or plan['producer'] != trusted_producer:
        raise ValueError('EVOLUTION_UNTRUSTED_SOURCE_OR_PRODUCER')
    if not attempt_id or attempt_id == previous['attemptId']:
        raise ValueError('EVOLUTION_NEW_ATTEMPT_REQUIRED')
    trace = producer(compiled_program=plan['programs']['workflow'],
                     **plan['programs']['workflow']['scenarioInputs'])
    reducer_projection = normalize_workflow_trace(trace, plan)
    run = project_meeting_result(trace, plan, attempt_id=attempt_id, recorded_at=recorded_at,
                                 reducer_projection=reducer_projection)
    seal_evolution_result(store, ir, plan, run, trusted_scope=trusted_scope,
                          resolved_source_pins=resolved_source_pins, trusted_producer=trusted_producer)
    return run


def verify_evolution_worker_assets(directory):
    """Verify one packaged code asset inventory; it grants no semantic authority."""
    import hashlib
    import json
    from pathlib import Path
    directory=Path(directory).resolve()
    try:
        manifest=json.loads((directory/'manifest.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError) as exc:
        raise ValueError('STATE_EVOLUTION_WORKER_UNAVAILABLE') from exc
    if manifest.get('schemaVersion')!='state_evolution_worker_build.v1' or manifest.get('buildHash')!=evolution_hash({key:value for key,value in manifest.items() if key!='buildHash'}):
        raise ValueError('STATE_EVOLUTION_WORKER_MANIFEST_HASH_MISMATCH')
    files=manifest.get('files')
    if not isinstance(files,dict) or 'lib/state-evolution/worker.js' not in files:
        raise ValueError('STATE_EVOLUTION_WORKER_INVALID_MANIFEST')
    actual={str(path.relative_to(directory)).replace('\\','/') for path in directory.rglob('*') if path.is_file() and path!=directory/'manifest.json'}
    if actual!=set(files):
        raise ValueError('STATE_EVOLUTION_WORKER_ASSET_HASH_MISMATCH')
    for relative,expected in files.items():
        path=directory/relative
        if path.is_symlink() or not path.resolve().is_relative_to(directory) or relative.startswith('/') or '..' in relative.split('/'):
            raise ValueError('STATE_EVOLUTION_WORKER_ASSET_PATH_INVALID')
        if 'sha256:'+hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('STATE_EVOLUTION_WORKER_ASSET_HASH_MISMATCH')
    return manifest


def invoke_state_evolution_worker(ctx, operation, *, worker_pin=None, payload=None):
    """Invoke an exact launcher-installed backend with business credentials removed."""
    import json
    import os
    import re
    import shutil
    import subprocess
    from runtime_paths import platform_runtime_root
    if not ctx.authenticated or not ctx.organization_id or not ctx.workspace_id or not ctx.user_id:
        raise PermissionError('STATE_EVOLUTION_AUTHENTICATED_SCOPE_REQUIRED')
    platform=platform_runtime_root()
    root=platform/'state-evolution-workers'
    if not root.resolve().is_relative_to(platform.resolve()):
        raise ValueError('STATE_EVOLUTION_WORKER_ASSET_PATH_INVALID')
    if worker_pin is None:
        try:
            worker_pin=json.loads((root/'current.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError) as exc:
            raise ValueError('STATE_EVOLUTION_WORKER_UNAVAILABLE') from exc
    reference=worker_pin.get('artifact_ref','')
    if worker_pin.get('artifact_kind')!='state_evolution_worker_build' or not re.fullmatch(r'state-evolution-worker:[0-9a-f]{64}',reference):
        raise ValueError('STATE_EVOLUTION_WORKER_INVALID_PIN')
    directory=root/reference.split(':')[1]
    if not directory.resolve().is_relative_to(root.resolve()):
        raise ValueError('STATE_EVOLUTION_WORKER_ASSET_PATH_INVALID')
    manifest=verify_evolution_worker_assets(directory)
    if manifest['buildHash']!='sha256:'+reference.split(':')[1] or worker_pin.get('artifact_hash')!=evolution_hash(manifest):
        raise ValueError('STATE_EVOLUTION_WORKER_MANIFEST_HASH_MISMATCH')
    node=shutil.which('node')
    if not node:
        raise ValueError('STATE_EVOLUTION_NODE_UNAVAILABLE')
    environment={'PATH':os.environ.get('PATH',''),'LANG':'C.UTF-8'}
    version=subprocess.run([node,'--version'],env=environment,capture_output=True,text=True,check=True,timeout=5).stdout.strip()
    if version!=manifest['nodeVersion']:
        raise ValueError('STATE_EVOLUTION_NODE_VERSION_MISMATCH')
    request={'operation':operation,**(payload or {})}
    if request.get('operation') != operation:
        raise ValueError('STATE_EVOLUTION_WORKER_OPERATION_OVERRIDE')
    encoded=json.dumps(request)
    if len(encoded.encode('utf-8'))>2_000_000:
        raise ValueError('STATE_EVOLUTION_WORKER_INPUT_BUDGET_EXCEEDED')
    result=subprocess.run([node,str(directory/'lib/state-evolution/worker.js')],input=encoded,env=environment,cwd=ctx.workspace_root,capture_output=True,text=True,timeout=30)
    if result.returncode:
        raise ValueError('STATE_EVOLUTION_WORKER_FAILED: '+result.stderr[-500:])
    return json.loads(result.stdout)


def state_evolution_worker_build_pin():
    """Resolve the exact launcher-selected worker build from the platform root."""
    import json
    from runtime_paths import platform_runtime_root
    platform=platform_runtime_root()
    root=platform/'state-evolution-workers'
    if not root.resolve().is_relative_to(platform.resolve()):
        raise ValueError('STATE_EVOLUTION_WORKER_ASSET_PATH_INVALID')
    try:
        pin=json.loads((root/'current.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError) as exc:
        raise ValueError('STATE_EVOLUTION_WORKER_UNAVAILABLE') from exc
    reference=pin.get('artifact_ref','')
    import re
    if pin.get('artifact_kind')!='state_evolution_worker_build' or not re.fullmatch(r'state-evolution-worker:[0-9a-f]{64}',reference):
        raise ValueError('STATE_EVOLUTION_WORKER_INVALID_PIN')
    directory=root/reference.split(':')[1]
    if not directory.resolve().is_relative_to(root.resolve()):
        raise ValueError('STATE_EVOLUTION_WORKER_ASSET_PATH_INVALID')
    manifest=verify_evolution_worker_assets(directory)
    if manifest['buildHash']!='sha256:'+reference.split(':')[1] or pin.get('artifact_hash')!=evolution_hash(manifest):
        raise ValueError('STATE_EVOLUTION_WORKER_MANIFEST_HASH_MISMATCH')
    return pin


@contextmanager
def authenticated_state_evolution_store(ctx, *, session_ref, channel):
    """Open the existing workspace artifact owner only after governed actor/session resolution."""
    from contracts.knowledge_classification import RuntimeContextDTO
    from services.runtime_context_proposal_provider import RuntimeContextProposalProviderRegistry
    from services.runtime_context_transition_service import RuntimeAuthorityResolver
    from services.runtime_session_service import (
        GovernedRuntimeSessionResolver,
        runtime_context_dto_from_request_context,
    )
    from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path
    from storage.runtime_context_authority_store import RuntimeContextAuthorityStore

    if not ctx.authenticated or not ctx.organization_id or not ctx.workspace_id or not ctx.user_id:
        raise PermissionError('STATE_EVOLUTION_AUTHENTICATED_SCOPE_REQUIRED')
    store = OntologyBuildStore(ontology_build_store_path(ctx.workspace_root))
    try:
        authority = RuntimeAuthorityResolver(
            graph_intent_store=None,
            ontology_build_store=store,
            runtime_context_store=None,
            authority_store=RuntimeContextAuthorityStore(store),
            provider_registry=RuntimeContextProposalProviderRegistry.compose_from_packaged_resources(),
        )
        actor = RuntimeContextDTO(
            organization_id=str(ctx.organization_id),
            workspace_id=str(ctx.workspace_id),
            actor_id=str(ctx.user_id),
        )
        session_id = GovernedRuntimeSessionResolver(authority_resolver=authority).resolve(
            actor_context=actor,
            session_ref=session_ref,
            authorization_class="WORKSPACE_SIMULATION",
        )
        runtime_context = runtime_context_dto_from_request_context(
            ctx=ctx, session_id=session_id, channel=channel
        )
        scope = {
            'organizationId':runtime_context.organization_id,
            'workspaceId':runtime_context.workspace_id,
            'sessionId':runtime_context.session_id,
            'actorRef':runtime_context.actor_id,
        }
        yield store, scope, session_id
    finally:
        store.close()
