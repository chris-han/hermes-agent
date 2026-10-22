"""Read-only Meeting workflow binding verification and deterministic replay.

Definition admission/activation stays with the TypeScript definition owner. These
pure checks grant no authority and own no state; the Meeting operation store
remains the only execution/event/result/receipt writer.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eos.artifact_hashing import content_hash

WORKFLOW_ID = 'feishu-meeting-coordinator'
OWNER_REF = 'skill:feishu_meeting_coordinator'
COMPILER_VERSION = 'workflow-execution-compiler/v1'


def hash_value(value: Any) -> str:
    return 'sha256:' + content_hash(value)


def _utc(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        raise ValueError('WORKFLOW_TIMESTAMP_MUST_BE_UTC') from None


def workflow_definition_hash(definition: dict, *, include_dependency: bool = True, include_labels: bool = True) -> str:
    """The v1 semantic-basis wire contract, verified against the TS producer.

    Only ASCII machine identifiers are admitted on this runtime bridge. Historical
    v1 retains labels; compiler v2 excludes them. This is verification, never
    compilation/admission.
    """
    def records(items, fields):
        return [{key: item.get(key) for key in fields} for item in sorted(items, key=lambda item: item['id'])]

    topology = definition.get('responsibilityTopology')
    basis = {key: definition[key] for key in ('schemaVersion', 'workflowId', 'workflowVersion', 'ownerRef', 'topology')}
    basis['responsibilityTopology'] = None if not topology else {
        'participants': records(topology['participants'], ('id', 'label', 'controlBoundary', 'ref')),
        'partitions': records(topology['partitions'], ('id', 'participantRef', 'actorKind', 'label', 'roleRef')),
    }
    basis['nodes'] = records(definition['nodes'], ('id', 'kind', 'label', 'responsibilityRef', 'config', 'input', 'output', 'refs'))
    basis['edges'] = records(definition['edges'], ('id', 'source', 'target', 'transitionClass', 'conditionRef', 'config'))
    if include_dependency:
        for edge, source in zip(basis['edges'], sorted(definition['edges'], key=lambda item: item['id'])):
            dependency = source.get('dependency')
            edge['dependency'] = None if not dependency else {
                'dependencyClass': dependency['dependencyClass'],
                'contractRef': dependency.get('contractRef'),
            }
    if not include_labels:
        for node in basis['nodes']:
            node.pop('label', None)
        if basis['responsibilityTopology']:
            for records in basis['responsibilityTopology'].values():
                for record in records:
                    record.pop('label', None)
    return hash_value(basis)


def verify_meeting_binding(binding: dict, *, organization_id: str, workspace_id: str) -> dict:
    if not organization_id or not workspace_id:
        raise ValueError('WORKFLOW_SCOPE_REQUIRED')
    if binding.get('organizationId') != organization_id or binding.get('workspaceId') != workspace_id:
        raise ValueError('WORKFLOW_SCOPE_MISMATCH')
    if binding.get('schemaVersion') != 'meeting_workflow_runtime_binding.v1':
        raise ValueError('WORKFLOW_BINDING_SCHEMA_INVALID')
    definition = binding['definition']
    if definition.get('workflowId') != WORKFLOW_ID or definition.get('ownerRef') != OWNER_REF or definition.get('schemaVersion') != 'workflow_definition.v1' or definition.get('topology') != 'dag':
        raise ValueError('WORKFLOW_OWNER_IDENTITY_MISMATCH')
    for items in (definition['nodes'], definition['edges']):
        ids = [item['id'] for item in items]
        if len(ids) != len(set(ids)) or any(not re.fullmatch('[a-z][a-z0-9_]*', name) for name in ids):
            raise ValueError('WORKFLOW_MACHINE_IDENTIFIER_INVALID')
    # Historical pre-dependency snapshots retain their recorded identity. Both
    # schemes still require the complete immutable definition artifact and receipts.
    compiler_version = binding.get('execution', {}).get('compilation', {}).get('compilerVersion')
    if compiler_version not in (COMPILER_VERSION, 'workflow-execution-compiler/v2'):
        raise ValueError('WORKFLOW_EXECUTION_COMPILER_UNSUPPORTED')
    include_labels = compiler_version == COMPILER_VERSION
    valid_definition_hashes = {workflow_definition_hash(definition, include_labels=include_labels)}
    if include_labels and definition['workflowVersion'] == '1':
        valid_definition_hashes.add(workflow_definition_hash(definition, include_dependency=False))
    if binding['workflowHash'] not in valid_definition_hashes or binding['definitionArtifactHash'] != hash_value(definition):
        raise ValueError('WORKFLOW_DEFINITION_HASH_MISMATCH')
    admission, activation = binding['admission'], binding['activation']
    if admission.get('status') != 'ADMITTED' or activation.get('status') != 'ACTIVE':
        raise ValueError('WORKFLOW_OWNER_ACTIVATION_REQUIRED')
    for receipt in (admission, activation):
        if receipt.get('workflowId') != WORKFLOW_ID or receipt.get('workflowDefinitionHash') != binding['workflowHash']:
            raise ValueError('WORKFLOW_RECEIPT_BINDING_MISMATCH')
    if admission.get('basisRef') != OWNER_REF or admission['previousWorkflowHash'] != activation['previousActiveWorkflowHash'] or admission['assessmentRef'] != activation['assessmentRef'] or admission['candidateRef'] != activation['candidateRef']:
        raise ValueError('WORKFLOW_RECEIPT_BINDING_MISMATCH')
    _utc(admission['admittedAt'])
    _utc(activation['activatedAt'])
    if datetime.fromisoformat(activation['activatedAt'].replace('Z', '+00:00')) < datetime.fromisoformat(admission['admittedAt'].replace('Z', '+00:00')):
        raise ValueError('WORKFLOW_ACTIVATION_PRECEDES_ADMISSION')
    if binding['activationHash'] != hash_value(activation) or binding['activationRef'] != f"workflow-activation:{WORKFLOW_ID}:{binding['workflowHash']}":
        raise ValueError('WORKFLOW_ACTIVATION_HASH_MISMATCH')
    execution = binding['execution']
    if execution.get('schemaVersion') != 'workflow_execution_ir.v1' or execution.get('workflowHash') != binding['workflowHash'] or execution.get('workflowId') != WORKFLOW_ID or execution.get('workflowVersion') != definition['workflowVersion'] or execution.get('compilation', {}).get('compilerVersion') != compiler_version or binding['executionHash'] != hash_value(execution):
        raise ValueError('WORKFLOW_EXECUTION_HASH_MISMATCH')
    nodes = {node['id']: node for node in definition['nodes']}
    if {op['nodeRef'] for op in execution['operators']} != set(nodes) or len(execution['operators']) != len(nodes):
        raise ValueError('WORKFLOW_EXECUTION_BINDING_UNAVAILABLE')
    for op in execution['operators']:
        node = nodes[op['nodeRef']]
        kind = {'human_gate':'human', 'wait':'wait', 'effect':'effect', 'terminal':'terminal'}.get(node['kind'], 'deterministic')
        if op['operatorKind'] != kind:
            raise ValueError('WORKFLOW_EXECUTION_OPERATOR_MISMATCH')
        if kind not in {'human', 'wait', 'terminal'} and op.get('binding') != {'kind':'deterministic', 'implementationRef': f"feishu_meeting_coordinator:{node['id']}:v1"}:
            raise ValueError('WORKFLOW_EXECUTION_BINDING_UNAVAILABLE')
    controllers = {edge['edgeRef']: edge for edge in execution['controllers']}
    if len(controllers) != len(execution['controllers']) or set(controllers) != {edge['id'] for edge in definition['edges']}:
        raise ValueError('WORKFLOW_EXECUTION_CONTROLLER_MISMATCH')
    for edge in definition['edges']:
        controller = controllers[edge['id']]
        if controller['sourceRef'] != edge['source'] or controller['targetRef'] != edge['target'] or controller['controllerKind'] != edge['transitionClass'] or controller.get('conditionRef') != edge.get('conditionRef') or edge['source'] not in nodes or edge['target'] not in nodes:
            raise ValueError('WORKFLOW_EXECUTION_CONTROLLER_MISMATCH')
    return binding


def meeting_owner_directory(ctx) -> Path:
    if not ctx.authenticated:
        raise ValueError('authentication required')
    if not ctx.organization_id or not ctx.workspace_id or ctx.workspace_id == 'public':
        raise ValueError('WORKFLOW_SCOPE_REQUIRED')
    identity = json.dumps([ctx.organization_id, ctx.workspace_id, WORKFLOW_ID, OWNER_REF], ensure_ascii=False, separators=(',', ':'))
    scope = hashlib.sha256(identity.encode()).hexdigest()
    root = Path(ctx.workspace_root).resolve()
    directory = (root / '.hermes-workspace/workflow-definition-scopes' / scope / '.hermes-workspace/workflow-definitions' / WORKFLOW_ID).resolve()
    if not directory.is_relative_to(root):
        raise ValueError('WORKFLOW_PATH_OUTSIDE_WORKSPACE')
    return directory


def read_active_meeting_binding(ctx) -> dict:
    directory = meeting_owner_directory(ctx)
    try:
        head_path = (directory / 'head.json').resolve()
        if not head_path.is_relative_to(Path(ctx.workspace_root).resolve()):
            raise ValueError('WORKFLOW_PATH_OUTSIDE_WORKSPACE')
        head = json.loads(head_path.read_text())
        workflow_hash = head['contentHash']
        if not re.fullmatch('sha256:[0-9a-f]{64}', workflow_hash):
            raise ValueError('WORKFLOW_DEFINITION_HASH_MISMATCH')
        def artifact(kind):
            file = (directory / kind / (workflow_hash[7:] + '.json')).resolve()
            if not file.is_relative_to(Path(ctx.workspace_root).resolve()):
                raise ValueError('WORKFLOW_PATH_OUTSIDE_WORKSPACE')
            return json.loads(file.read_text())
        version, activation, binding = artifact('versions'), artifact('activations'), artifact('runtime-bindings')
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        raise ValueError('WORKFLOW_OWNER_ACTIVATION_REQUIRED') from None
    expected_scope = {'organizationId':ctx.organization_id, 'workspaceId':ctx.workspace_id}
    if head.get('scope') != expected_scope or version.get('scope') != expected_scope:
        raise ValueError('WORKFLOW_SCOPE_MISMATCH')
    expected_head = {**version, 'activation':activation, 'updatedAt':activation['activatedAt']}
    if head != expected_head or version.get('activation') is not None or head['definition'] != binding['definition'] or head['admission'] != binding['admission'] or activation != binding['activation'] or head['contentHash'] != binding['workflowHash']:
        raise ValueError('WORKFLOW_OWNER_RECEIPT_MISMATCH')
    if version['admission'].get('previousWorkflowHash') == 'sha256:' + '0' * 64:
        from agents.meeting_workflow_installation import verify
        verify(ctx, version['definition'], version.get('assessment') or {},
            actor_ref=version['admission'].get('actorRef'), current_owner=False)
    return verify_meeting_binding(binding, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)


def meeting_context_from_execution_boundary():
    """Resolve tool-session scope from the governed boundary and authority DB."""
    from agents.auth_session import RequestContext
    from agents.gateway_identity import get_user_organization_context, get_user_record
    from gateway.execution_boundary import current_execution_boundary
    from agents.gateway_identity import ensure_workspace_paths
    boundary = current_execution_boundary()
    if boundary is None or boundary.audit_metadata.get('authority_source') != 'semantier_authenticated_context' or not boundary.user_id or not boundary.workspace_id:
        raise ValueError('WORKFLOW_AUTHENTICATED_CONTEXT_REQUIRED')
    user = get_user_record(boundary.user_id)
    org = get_user_organization_context(boundary.user_id) or {}
    if not user or str(user.get('workspace_id') or '') != boundary.workspace_id or not org.get('organization_id'):
        raise ValueError('WORKFLOW_SCOPE_REQUIRED')
    root, runtime_home = ensure_workspace_paths(boundary.workspace_id)
    root = root.resolve()
    if boundary.paths.hermes_home is None or Path(boundary.paths.hermes_home).resolve() != runtime_home.resolve():
        raise ValueError('WORKFLOW_SCOPE_MISMATCH')
    return RequestContext(authenticated=True, user_id=boundary.user_id, organization_id=org['organization_id'], workspace_id=boundary.workspace_id, workspace_slug=boundary.workspace_id, workspace_root=root, hermes_home=root)

def authorize_meeting_requester_mutation(negotiation, requested_by_user_id, ctx=None):
    """Bind native commitment to authenticated authority, never payload identity."""
    from agents.gateway_identity import get_user_record, get_user_organization_context

    ctx = ctx or meeting_context_from_execution_boundary()
    if not ctx.authenticated:
        raise ValueError('WORKFLOW_AUTHENTICATED_CONTEXT_REQUIRED')
    user = get_user_record(ctx.user_id)
    org = get_user_organization_context(ctx.user_id) or {}
    if (not user or user.get('workspace_id') != ctx.workspace_id
            or org.get('organization_id') != ctx.organization_id):
        raise ValueError('WORKFLOW_SCOPE_MISMATCH')
    if (negotiation['workspace_id'] != ctx.workspace_id
            or negotiation.get('organization_id') != ctx.organization_id):
        raise ValueError('WORKFLOW_SCOPE_MISMATCH')
    if (ctx.user_id != negotiation['creator_user_id']
            or requested_by_user_id != ctx.user_id):
        raise PermissionError('WORKFLOW_ACTOR_IDENTITY_MISMATCH')
    return ctx


REPLAY_CONTRACT_VERSION = 'meeting-execution-replay/v1'


def replay_meeting_execution(*, binding: dict, instance_id: str, events: list[dict], receipt: dict | None = None) -> dict:
    """Pure artifact-only replay. No owner-head lookup, clock, provider or DB."""
    verify_meeting_binding(binding, organization_id=binding['organizationId'], workspace_id=binding['workspaceId'])
    if not events:
        raise ValueError('WORKFLOW_EVENT_STREAM_MISSING')
    nodes = {node['id'] for node in binding['definition']['nodes']}
    controllers = {c['edgeRef']: c for c in binding['execution']['controllers']}
    node_states = {node: {'status':'not_reached'} for node in sorted(nodes)}
    edge_states = {}
    effects = []
    interaction_events = []
    interaction_types = {
        'DECISION_FRONTIER_COMPUTED',
        'INFORMATION_ACTION_SELECTED',
        'DECISION_FRONTIER_PRESENTED',
        'VOTE_RECORDED',
        'INBOUND_ACCEPTED',
        'RESPONSIBILITY_WAKE_ACCEPTED',
        'RESPONSIBILITY_WAITING',
        'FOLLOWUP_REMINDER_SENT',
        'FOLLOWUP_RSVP_POLLED',
        'FINALIZE_STARTED',
        'CALENDAR_UPDATE_SUCCEEDED',
        'FINALIZE_COMPLETED',
    }
    decision_projection = None
    status = 'partial'
    disposition = 'partial'
    for sequence, event in enumerate(events, 1):
        if event.get('event_sequence') != sequence or event.get('negotiation_id') != instance_id:
            raise ValueError('WORKFLOW_EVENT_SEQUENCE_MISMATCH')
        _utc(event['created_at'])
        raw = event['payload_json']
        if hashlib.sha256(raw.encode()).hexdigest() != event['payload_hash']:
            raise ValueError('WORKFLOW_EVENT_PAYLOAD_HASH_MISMATCH')
        payload = json.loads(raw)
        if 'decisionProjection' in payload:
            decision_projection = payload['decisionProjection']
        if payload.get('workflowBindingHash') != hash_value(binding) or payload.get('organizationId') != binding['organizationId'] or payload.get('workspaceId') != binding['workspaceId']:
            raise ValueError('WORKFLOW_EVENT_BINDING_MISMATCH')
        if event['event_type'] in interaction_types:
            participant_user_id = payload.get('attendee_user_id')
            if event.get('actor_type') == 'participant':
                participant_user_id = event.get('actor_id')
            interaction_events.append({
                'eventId': event['event_id'],
                'eventType': event['event_type'],
                'occurredAt': event['created_at'],
                'actorType': event.get('actor_type'),
                'actorId': event.get('actor_id'),
                'participantUserId': participant_user_id,
                'messageId': payload.get('message_id'),
                'messageEventId': payload.get('message_event_id'),
                'agentProfileId': payload.get('agent_profile_id'),
                'agentSessionId': payload.get('agent_session_id'),
                'wakeKind': payload.get('wake_kind'),
                'wakeSourceRef': payload.get('wake_source_ref'),
                'responsibilityId': payload.get('responsibility_id'),
                'timerId': payload.get('timer_id'),
                'nodeRef': payload.get('nodeRef'),
            })
        if event.get('next_state'):
            status = event['next_state']
        node = payload.get('nodeRef')
        if node:
            if node not in nodes:
                raise ValueError('WORKFLOW_EVENT_NODE_UNAVAILABLE')
            node_states[node] = {'status':payload.get('nodeStatus','completed'), 'latestOccurrenceId':event['event_id'], 'latestOccurredAt':event['created_at']}
        edge = payload.get('controllerRef')
        if edge:
            controller = controllers.get(edge)
            if not controller or controller['sourceRef'] != node or controller['targetRef'] != payload.get('targetRef'):
                raise ValueError('WORKFLOW_EVENT_CONTROLLER_MISMATCH')
            selected = event['event_type'] == 'WORKFLOW_TRANSITION_SELECTED' or payload.get('transitionOutcome') == 'selected'
            if payload.get('conditionRef') != controller.get('conditionRef'):
                raise ValueError('WORKFLOW_EVENT_CONDITION_MISMATCH')
            predicates = {
                'terminal_authority_consented': status == 'consented',
                'terminal_authority_blocked': status not in {'consented','requester_decided'},
                'requester_selected_slot': status == 'requester_decided',
                'vote_or_rsvp_state_changed': event['event_type'] == 'VOTE_RECORDED',
                'wake_signal_accepted': event['event_type'] == 'RESPONSIBILITY_WAKE_ACCEPTED',
                'frontier_ready': event['event_type'] == 'DECISION_FRONTIER_PRESENTED' and bool(payload.get('decision_frontier')),
                'evidence_needed': event['event_type'] == 'INFORMATION_ACTION_SELECTED' and payload.get('sufficient_progress_disposition') == 'CONTINUE_INFORMATION',
                None: controller['controllerKind'] == 'normal',
            }
            if controller.get('conditionRef') not in predicates or payload.get('conditionSatisfied') is not selected or predicates[controller.get('conditionRef')] != selected:
                raise ValueError('WORKFLOW_EVENT_CONDITION_MISMATCH')
            edge_states[edge] = {'active':selected, 'traversed':selected}
            if not selected:
                node_states[node]['status'] = 'waiting'
                disposition = 'blocked'
            else:
                node_states[controller['targetRef']] = {'status':payload.get('targetStatus','ready')}
        if event['event_type'] == 'CALENDAR_UPDATE_SUCCEEDED':
            if not edge_states.get('authority_to_finalize',{}).get('traversed') and not edge_states.get('requester_to_finalize',{}).get('traversed'):
                raise ValueError('WORKFLOW_EFFECT_WITHOUT_SELECTED_CONTROLLER')
            terminal = controllers.get('finalize_to_terminal')
            if not terminal or terminal['sourceRef'] != 'finalize_meeting' or terminal['targetRef'] != 'terminal' or terminal.get('conditionRef'):
                raise ValueError('WORKFLOW_TERMINAL_CONTROLLER_UNAVAILABLE')
            effects.append({'attempt_id':payload['finalize_attempt_id'], 'result':payload['effectResult']})
            node_states['terminal'] = {'status':'terminal', 'latestOccurrenceId':event['event_id'], 'latestOccurredAt':event['created_at']}
            edge_states['finalize_to_terminal'] = {'active':True,'traversed':True}
            disposition = 'terminal'
    graph = {'schemaVersion':'workflow_execution_graph.v1', 'workflowId':binding['definition']['workflowId'], 'workflowVersion':binding['definition']['workflowVersion'], 'workflowHash':binding['workflowHash'], 'instanceId':instance_id, 'observedAt':events[-1]['created_at'], 'sourceCursor':str(len(events)), 'instanceStatus':disposition, 'nodeStates':node_states, 'edgeStates':edge_states, 'interactionEvents':interaction_events}
    if decision_projection is not None:
        graph['decision'] = decision_projection
    result = {'schema_version':'meeting_workflow_result.v1', 'instance_id':instance_id, 'organization_id':binding['organizationId'], 'workspace_id':binding['workspaceId'], 'execution_disposition':disposition, 'domain_status':status, 'effects':effects}
    if receipt is not None:
        expected = meeting_execution_receipt(binding=binding, instance_id=instance_id, events=events, projection={'graph':graph,'result':result})
        if receipt != expected:
            raise ValueError('WORKFLOW_RECEIPT_REPLAY_MISMATCH')
    return {'graph':graph, 'result':result}


def meeting_execution_receipt(*, binding: dict, instance_id: str, events: list[dict], projection: dict) -> dict:
    """Derive identity only; persistence/origin belongs to the native owner."""
    receipt = {
        'schema_version':'workflow_execution_receipt.v1',
        'producer_id':'meeting-coordinator-runtime', 'producer_contract_version':'meeting-definition-execution/v1',
        'workflow_id':binding['definition']['workflowId'], 'workflow_version':binding['definition']['workflowVersion'],
        'workflow_hash':binding['workflowHash'], 'workflow_activation_ref':binding['activationRef'], 'workflow_activation_hash':binding['activationHash'],
        'workflow_execution_ir_hash':binding['executionHash'], 'compiler_version':COMPILER_VERSION,
        'instance_id':instance_id, 'organization_id':binding['organizationId'], 'workspace_id':binding['workspaceId'],
        'execution_disposition':projection['result']['execution_disposition'],
        'result_ref':f'meeting-result:{instance_id}:{len(events)}', 'result_hash':hash_value(projection['result']),
        'event_stream_ref':f'meeting-events:{instance_id}:1:{len(events)}', 'event_stream_hash':hash_value(events), 'event_count':len(events),
        'graph_hash':hash_value(projection['graph']), 'replay_contract_version':REPLAY_CONTRACT_VERSION,
        'runtime_binding_refs': [op['binding']['implementationRef'] for op in binding['execution']['operators'] if op.get('binding')],
        'runtime_binding_hash':hash_value([op.get('binding') for op in binding['execution']['operators']]),
        'produced_at_utc':events[-1]['created_at'],
    }
    consumed = [json.loads(event['payload_json']).get('consumption') for event in events
                if event['event_type'] == 'SEMANTIC_BASIS_CONSUMED']
    if consumed:
        if len(consumed) != 1:
            raise ValueError('CONTINUATION_CONSUMPTION_COUNT_MISMATCH')
        receipt.update(schema_version='workflow_execution_receipt.v2',
            producer_contract_version='meeting-definition-execution/v2-continuation',
            continuation_consumption_hash=hash_value(consumed[0]))
    return {**receipt, 'receipt_hash':hash_value(receipt)}


def resolve_meeting_consumption(*, evidence, basis, working_context, expected_run, contract):
    """Artifact-only exact joins after resolving records from native owners.

    expected_run is obtained from the separately frozen comparison schedule,
    never from a supplied panel lane or receipt. No current action authority.
    """
    from contracts.decision_trajectory_projection import SemanticContinuationBasisV1
    from contracts.semantic_kernel import WorkingContext
    basis = SemanticContinuationBasisV1.model_validate(basis)
    wc = WorkingContext.model_validate(working_context)
    replay = replay_meeting_execution(binding=evidence['binding'],
        instance_id=evidence['receipt']['instance_id'], events=evidence['events'], receipt=evidence['receipt'])
    if not evidence['receipt'].get('continuation_consumption_hash'):
        raise ValueError('CONTINUATION_BINDING_MISSING')
    events = [event for event in evidence['events'] if event['event_type'] == 'SEMANTIC_BASIS_CONSUMED']
    if len(events) != 1 or events[0]['actor_type'] != 'runtime' or events[0]['actor_id'] != 'meeting-coordinator-runtime':
        raise ValueError('CONTINUATION_PRODUCER_READ_REQUIRED')
    consumed = json.loads(events[0]['payload_json'])['consumption']
    if hash_value(consumed) != evidence['receipt']['continuation_consumption_hash']:
        raise ValueError('CONTINUATION_BINDING_HASH_MISMATCH')
    for key in ('pair_id','seed','boundary','read_mode','run_id'):
        if consumed[key] != expected_run[key]:
            raise ValueError('CONTINUATION_EXPECTED_RUN_MISMATCH')
    if consumed['comparison_contract_hash'] != hash_value(contract):
        raise ValueError('CONTINUATION_COMPARISON_CONTRACT_MISMATCH')
    if consumed['reader_version'] != 'meeting-semantic-consumer/v1' or consumed['schema_version'] != 'meeting_semantic_consumption.v1':
        raise ValueError('CONTINUATION_READER_MISMATCH')
    if consumed['basis'] != basis.pin().model_dump(mode='json') or consumed['checkpoint'] != basis.checkpoint.model_dump(mode='json'):
        raise ValueError('CONTINUATION_BASIS_CHECKPOINT_MISMATCH')
    if consumed['working_context'] != wc.pin().model_dump(mode='json') or basis.working_context != wc.pin():
        raise ValueError('CONTINUATION_CONSUMED_BASIS_MISMATCH')
    command = json.loads(wc.task_goal)
    if consumed['consumed_request'] != command or consumed['consumed_request_hash'] != hash_value(command):
        raise ValueError('CONTINUATION_CONSUMED_REQUEST_MISMATCH')
    binding = evidence['binding']
    if (consumed['instance_id'] != evidence['receipt']['instance_id']
            or command['case_id'] != consumed['instance_id']
            or consumed['workflow_binding_hash'] != hash_value(binding)
            or (consumed['organization_id'],consumed['workspace_id']) != (basis.organization_id,basis.workspace_id)
            or (wc.organization_id,wc.workspace_id) != (binding['organizationId'],binding['workspaceId'])
            or consumed['actor_id'] != wc.responsibility.responsibility_bearer_id):
        raise ValueError('CONTINUATION_SCOPE_DEFINITION_MISMATCH')
    # All terminal success/effect claims come from replay. The resolved request
    # must agree with the native executed consequence, not a caller success bit.
    effects = replay['result']['effects']
    if (replay['result']['execution_disposition'] != 'terminal' or not effects
            or any(effect['result'].get('ok') is not True or effect['result'].get('event_id') != command['event'] for effect in effects)):
        raise ValueError('CONTINUATION_EFFECT_UNQUALIFIED')
    return {'consumption':consumed, 'result':replay['result'], 'graph':replay['graph']}
