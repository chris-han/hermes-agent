"""Capability-checked lowering to existing owner-native execution backends.

This module grants no authority and performs no source reads or execution.
The caller must resolve and pin source artifacts through governed owners.
"""
import copy
from contracts.state_evolution import evolution_hash

# Exact supported domain operations, not a universal BPMN interpreter.
_MEETING_OPERATIONS = {
    'observe_meeting_event': ('observe', 'scenario_input'),
    'ensure_negotiation_case': ('compute', 'initialize_negotiation'),
    'ensure_kanban_task': ('effect', 'scenario_input'),
    'ensure_followup_timer': ('effect', 'scenario_input'),
    'poll_rsvp': ('observe', 'scenario_input'),
    'collect_candidate_slots': ('compute', 'receive_feishu'),
    'collect_votes': ('wait', 'receive_feishu'),
    'form_decision_frontier': ('compute', 'planner_tick'),
    'evaluate_information_value': ('decide', 'information_actions'),
    'evaluate_terminal_authority': ('decide', 'requester_boundary'),
    'requester_decision': ('human_gate', 'present_frontier'),
    'wait_followup': ('wait', 'prepare_wait'),
    'finalize_meeting': ('effect', 'scenario_input'),
    'cancel_meeting': ('effect', 'requester_decision'),
    'terminal': ('terminal', 'terminal'),
}
_MEETING_CONTROLLERS = {
    'event_to_case': ('observe_meeting_event', 'ensure_negotiation_case', 'normal', None),
    'case_to_task': ('ensure_negotiation_case', 'ensure_kanban_task', 'normal', None),
    'task_to_timer': ('ensure_kanban_task', 'ensure_followup_timer', 'normal', None),
    'timer_to_poll': ('ensure_followup_timer', 'poll_rsvp', 'normal', None),
    'poll_to_slots': ('poll_rsvp', 'collect_candidate_slots', 'normal', None),
    'slots_to_votes': ('collect_candidate_slots', 'collect_votes', 'normal', None),
    'votes_to_frontier': ('collect_votes', 'form_decision_frontier', 'conditional', 'vote_or_rsvp_state_changed'),
    'frontier_to_information': ('form_decision_frontier', 'evaluate_information_value', 'normal', None),
    'information_to_authority': ('evaluate_information_value', 'evaluate_terminal_authority', 'conditional', 'frontier_ready'),
    'information_to_wait': ('evaluate_information_value', 'wait_followup', 'retry', 'evidence_needed'),
    'authority_to_finalize': ('evaluate_terminal_authority', 'finalize_meeting', 'conditional', 'terminal_authority_consented'),
    'authority_to_requester': ('evaluate_terminal_authority', 'requester_decision', 'escalation', 'terminal_authority_requester_decision'),
    'authority_to_wait': ('evaluate_terminal_authority', 'wait_followup', 'retry', 'terminal_authority_blocked'),
    'requester_to_finalize': ('requester_decision', 'finalize_meeting', 'conditional', 'requester_selected_slot'),
    'requester_to_cancel': ('requester_decision', 'cancel_meeting', 'conditional', 'requester_cancelled'),
    'wait_to_poll': ('wait_followup', 'poll_rsvp', 'retry', 'wake_signal_accepted'),
    'finalize_to_terminal': ('finalize_meeting', 'terminal', 'normal', None),
    'cancel_to_terminal': ('cancel_meeting', 'terminal', 'normal', None),
}
_ALLOWED_REFS = {
    'ensure_kanban_task': {'effect_contract_ref':'feishu.meeting.kanban.ensure'},
    'ensure_followup_timer': {'effect_contract_ref':'feishu.meeting.followup.ensure'},
    'evaluate_terminal_authority': {'decision_ref':'feishu.meeting.terminal_authority'},
    'requester_decision': {'authority_ref':'feishu.meeting.requester'},
    'finalize_meeting': {'effect_contract_ref':'feishu.meeting.finalize'},
    'cancel_meeting': {'effect_contract_ref':'feishu.meeting.cancel'},
}


def _indexed(rows, key, expected):
    indexed = {row[key]: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(expected):
        raise ValueError('INVALID_BINDING: incomplete or duplicate mechanism coverage')
    return indexed


def lower_meeting_program(source: dict, execution: dict, *, native_source: dict | None = None) -> dict:
    """Consume native compiled operators/controllers with complete bounded coverage.

    Scenario-input mappings denote explicitly synthetic effects/observations;
    they never authorize a real Feishu effect. Source config/semantic extensions
    outside this backend's supported surface fail before starting any actor.
    """
    if source.get('workflow_id') != 'feishu-meeting-coordinator' or source.get('owner_ref') != 'skill:feishu_meeting_coordinator':
        raise ValueError('UNSUPPORTED: Meeting source owner')
    if execution.get('schemaVersion') != 'workflow_execution_ir.v1' or execution.get('workflowId') != source['workflow_id'] or execution.get('workflowVersion') != source['workflow_version']:
        raise ValueError('INVALID_BINDING: native execution source mismatch')
    if execution.get('dependencies'):
        if native_source is None:
            raise ValueError('UNSUPPORTED: unbound dependency source')
        native_edges = {edge['id']:edge for edge in native_source['edges']}
        expected_dependencies = []
        for edge in source['edges']:
            native = native_edges.get(edge['id'])
            if native is None or (native['source'],native['target'],native['transitionClass'],native.get('conditionRef')) != (edge['source'],edge['target'],edge['transition_class'],edge.get('condition_ref')):
                raise ValueError('INVALID_BINDING: native source disagreement')
            dependency = native.get('dependency')
            if dependency:
                expected_dependencies.append({'edgeRef':edge['id'],'sourceRef':edge['source'],'targetRef':edge['target'], **dependency})
        if sorted(execution['dependencies'],key=lambda row:row['edgeRef']) != sorted(expected_dependencies,key=lambda row:row['edgeRef']):
            raise ValueError('INVALID_BINDING: dependency lowering')
    nodes = _indexed(source['nodes'], 'id', _MEETING_OPERATIONS)
    operators = _indexed(execution['operators'], 'nodeRef', _MEETING_OPERATIONS)
    edges = _indexed(source['edges'], 'id', _MEETING_CONTROLLERS)
    controllers = _indexed(execution['controllers'], 'edgeRef', _MEETING_CONTROLLERS)
    program = {'schemaVersion':'meeting_evolution_program.v1', 'operators':{}, 'controllers':{}}
    for ident, (kind, operation) in _MEETING_OPERATIONS.items():
        node = nodes[ident]
        if node['kind'] != kind or set(node) - {'id','kind','label','refs'} or node.get('refs',{}) != _ALLOWED_REFS.get(ident,{}):
            raise ValueError('UNSUPPORTED: node semantics '+ident)
        expected_kind = {'human_gate':'human','wait':'wait','effect':'effect','terminal':'terminal'}.get(kind,'deterministic')
        operator = operators[ident]
        expected_binding = {'kind':'deterministic','implementationRef':'meeting-producer:'+ident+':v1'} if expected_kind not in ('wait','human','terminal') else None
        if operator.get('binding') != expected_binding:
            raise ValueError('UNSUPPORTED: native implementation binding '+ident)
        if operator['operatorKind'] != expected_kind:
            raise ValueError('INVALID_BINDING: operator kind '+ident)
        if any(operator.get(field) for field in ('capabilityRequirementRef','decisionSpaceRef','decisionFrontierRef')):
            raise ValueError('UNSUPPORTED: additional capability or decision contract')
        program['operators'][ident] = {'operation':operation, 'operatorKind':expected_kind,
            'mappingKind':'scenario-input' if operation == 'scenario_input' else 'producer-operation'}
    for ident, expected in _MEETING_CONTROLLERS.items():
        edge = edges[ident]; controller = controllers[ident]
        actual = (edge['source'],edge['target'],edge['transition_class'],edge.get('condition_ref'))
        if actual != expected or set(edge) - {'id','source','target','transition_class','condition_ref'}:
            raise ValueError('UNSUPPORTED: controller semantics '+ident)
        lowered = (controller['sourceRef'],controller['targetRef'],controller['controllerKind'],controller.get('conditionRef'))
        if lowered != actual:
            raise ValueError('INVALID_BINDING: controller lowering '+ident)
        program['controllers'][ident] = {key:value for key,value in controller.items() if key != 'edgeRef'}
        if ident == 'wait_to_poll':
            program['controllers'][ident]['releaseOperation'] = 'timer_signal'
    program['programHash'] = evolution_hash(program)
    return program



def meeting_semantic_source(source):
    """Payload of a pinned label-free source-semantic artifact, not a file-byte hash.

    Only owner-declared display labels are removed. Semantic config, contracts,
    identifiers and condition references are retained. W0 raw source revisions
    remain separate provenance; callers seal this projection in the same store.
    """
    value = copy.deepcopy(source)
    for collection in ('nodes','edges'):
        if collection in value:
            value[collection].sort(key=lambda item:item['id'])
    for node in value.get('nodes',[]):
        node.pop('label',None)
    for collection in ('participants','partitions'):
        topology = value.get('responsibilityTopology',{})
        if collection in topology:
            topology[collection].sort(key=lambda item:item['id'])
            for item in topology[collection]:
                item.pop('label',None)
    return value


def build_meeting_evolution_binding(*, scope, source, execution, native_source, profile_seeds, compiler, backend, producer, effective_at):
    """Bind the exact repository-owned Meeting source bundle for one session scope."""
    from contracts.state_evolution import EvolutionBindingSpecV1

    if source.get('workflow_id') != 'feishu-meeting-coordinator' or source.get('owner_ref') != 'skill:feishu_meeting_coordinator':
        raise ValueError('UNSUPPORTED: Meeting source owner')
    if execution.get('workflowId') != source.get('workflow_id') or execution.get('workflowVersion') != source.get('workflow_version'):
        raise ValueError('INVALID_BINDING: native execution source mismatch')
    if set(profile_seeds) != {'requester','cooperative','conflict'}:
        raise ValueError('INVALID_BINDING: Meeting profile-seed inventory')

    def pin(kind, reference, value):
        return {'artifact_kind':kind,'artifact_ref':reference,'artifact_hash':evolution_hash(value)}

    workflow_pin=pin('workflow_definition','feishu-meeting-coordinator:workflow:'+str(source['workflow_version']),meeting_semantic_source(source))
    native_pin=pin('workflow_native_definition','feishu-meeting-coordinator:native:'+str(source['workflow_version']),meeting_semantic_source(native_source))
    execution_pin=pin('workflow_execution_ir','feishu-meeting-coordinator:execution:'+str(source['workflow_version']),execution)
    profile_pin=pin('meeting_simulation_profiles','feishu-meeting-coordinator:profiles:v1',profile_seeds)
    origin_payload={'workflow':workflow_pin,'nativeDefinition':native_pin,'nativeExecution':execution_pin,'profileSeeds':profile_pin}
    origin_pin=pin('meeting_scenario_basis','feishu-meeting-coordinator:scenario-basis:'+evolution_hash(origin_payload).split(':')[1],origin_payload)
    symbols=[]
    for ident, value_type, unit, initial in (
        ('workspace_user_count','number','person',4),
        ('stable_seed','string','dimensionless','workshop-v1'),
    ):
        symbols.append({'id':ident,'kind':'input','valueType':value_type,'unit':unit,'sourcePin':origin_pin,
            'objectId':'meeting_scenario','fieldId':ident,'effectiveAt':effective_at,
            'freshnessBasis':'authenticated-session-controlled-fixture','epistemicStatus':'ASSUMED',
            'initialValue':{'status':'KNOWN','value':initial},
            **({'range':[3,12]} if ident=='workspace_user_count' else {})})
    binding={'schemaVersion':'evolution_binding_spec.v1','scope':scope,
        'sourcePins':{'origin':origin_pin,'workflow-definition':workflow_pin,'workflow-native-definition':native_pin,
            'native-execution':execution_pin,'profile-seeds':profile_pin,
            'compiler':compiler,'workflow-backend':backend,'producer':producer},
        'symbols':symbols,'eventBindings':[],'scenarioPermissions':['workspace_user_count','stable_seed'],
        'timeBasis':{'kind':'order-only'}}
    return EvolutionBindingSpecV1.model_validate(binding).model_dump(mode='json',exclude_unset=True)

def compile_meeting_evolution(binding, source, execution, *, scenario, compiler, backend, producer, native_source=None):
    """Compile an already owner-resolved Meeting composition; never resolve trust here."""
    from contracts.state_evolution import EvolutionBindingSpecV1, EvolutionIRV1, verify_evolution_plan
    bound = EvolutionBindingSpecV1.model_validate(binding)
    for role, artifact in [('compiler',compiler),('workflow-backend',backend),('producer',producer)]:
        pin=bound.sourcePins.get(role)
        if pin is None or pin.model_dump() != artifact:
            raise ValueError('SOURCE_PIN_MISMATCH: '+role)
    for role, artifact in [('workflow-definition',source),('native-execution',execution)]:
        pin = bound.sourcePins.get(role)
        if pin is None or pin.artifact_hash != evolution_hash(meeting_semantic_source(artifact) if role == 'workflow-definition' else artifact):
            raise ValueError('SOURCE_PIN_MISMATCH: '+role)
    if native_source is not None:
        pin = bound.sourcePins.get('workflow-native-definition')
        if pin is None or pin.artifact_hash != evolution_hash(meeting_semantic_source(native_source)):
            raise ValueError('SOURCE_PIN_MISMATCH: workflow-native-definition')
    if not set(scenario['overrides']) <= set(bound.scenarioPermissions):
        raise ValueError('INVALID_SCENARIO_PERMISSIONS')
    if any(entry['symbolId'] not in bound.scenarioPermissions for entry in scenario['scheduledInputs']):
        raise ValueError('INVALID_SCENARIO_PERMISSIONS')
    program = lower_meeting_program(source, execution, native_source=native_source)
    symbols = {symbol.id:symbol for symbol in bound.symbols}
    modeled = {}
    for ident in ('workspace_user_count','stable_seed'):
        symbol = symbols.get(ident)
        if symbol is None or symbol.kind != 'input' or symbol.initialValue.status != 'KNOWN':
            raise ValueError('UNSUPPORTED: unbound Meeting scenario input '+ident)
        value = scenario['overrides'].get(ident,symbol.initialValue.value)
        symbol.validate_value(value)
        modeled[ident] = value
    if type(modeled['workspace_user_count']) is not int and not (isinstance(modeled['workspace_user_count'],float) and modeled['workspace_user_count'].is_integer()):
        raise ValueError('INVALID_BINDING: fractional Meeting population')
    if not 3 <= modeled['workspace_user_count'] <= 12 or not isinstance(modeled['stable_seed'],str) or not 1 <= len(modeled['stable_seed']) <= 128 or scenario['id'] not in tuple('ABCDEF'):
        raise ValueError('INVALID_BINDING: unsupported Meeting scenario')
    modeled['workspace_user_count'] = int(modeled['workspace_user_count'])
    profiles_pin = bound.sourcePins.get('profile-seeds')
    if profiles_pin is None:
        raise ValueError('UNSUPPORTED: unbound Meeting profile seeds')
    program['profileSeedsHash'] = profiles_pin.artifact_hash
    from services.state_evolution_clock import MeetingScenarioClock
    delay = None
    if bound.timeBasis.kind == 'domain':
        delay_symbol = symbols.get('followup_delay')
        if delay_symbol is None or delay_symbol.kind != 'input' or delay_symbol.unit != 'second' or delay_symbol.initialValue.status != 'KNOWN':
            raise ValueError('UNSUPPORTED: unbound follow-up delay')
        delay = scenario['overrides'].get('followup_delay',delay_symbol.initialValue.value)
        delay_symbol.validate_value(delay)
    clock_config = {'timeBasis':bound.timeBasis.model_dump(mode='json',exclude_unset=True),'schedule':scenario['scheduledInputs'],'followupDelay':delay}
    MeetingScenarioClock(clock_config['timeBasis'],clock_config['schedule'],followup_delay=delay)
    program['clockConfig'] = clock_config
    program['scenarioInputs'] = {**modeled,'scenario':scenario['id'],'workflow_version':source['workflow_version']}
    program['programHash'] = evolution_hash({key:value for key,value in program.items() if key != 'programHash'})
    wire = bound.model_dump(mode='json',exclude_unset=True)
    workflow_pin = wire['sourcePins']['workflow-definition']
    transitions = []
    for edge in sorted(source['edges'],key=lambda item:item['id']):
        guard = {'kind':'owner-condition','ref':edge['condition_ref'],'sourcePin':workflow_pin} if edge.get('condition_ref') else {'kind':'constant','value':True}
        transitions.append({'id':edge['id'],'source':edge['source'],'target':edge['target'],'guard':guard,'sourcePin':workflow_pin})
    ir = {'schemaVersion':'evolution_ir.v1','profile':'WORKFLOW_ONLY','scope':wire['scope'],
          'sourcePins':wire['sourcePins'],'symbols':wire['symbols'],'discreteTransitions':transitions,
          'continuousEquations':{},'eventBindings':[],'timeBasis':wire['timeBasis'],'scenario':scenario,
          'claimBoundary':'SIMULATED: controlled fixture; no external effects or requester authority granted.'}
    if 'quantityBasis' in wire:
        ir['quantityBasis'] = wire['quantityBasis']
    EvolutionIRV1.model_validate(ir)
    plan = {'schemaVersion':'evolution_execution_plan.v1','profile':'WORKFLOW_ONLY','scope':wire['scope'],
            'irRef':{'artifact_kind':'evolution_ir','artifact_ref':'evolution-ir:'+evolution_hash(ir).split(':')[1],'artifact_hash':evolution_hash(ir)},
            'compiler':compiler,'backends':{'workflow':backend},'producer':producer,
            'timeBasis':wire['timeBasis'],'scenario':scenario,'programs':{'workflow':program},
            'sourceMap':{symbol['id']:symbol['sourcePin'] for symbol in wire['symbols']}}
    for node_id in program['operators']:
        plan['sourceMap']['workflow.node-status.'+node_id] = workflow_pin
    plan['sourceMap']['workflow.run-status'] = workflow_pin
    plan['basisHash'] = evolution_hash(plan)
    verify_evolution_plan(plan)
    return ir, plan


def compile_continuous_evolution(*, scope, source, scenario_id, compiler, backend, producer, effective_at, comparison_basis_ref=None):
    """Compile one owner-resolved System Dynamics source into CONTINUOUS_ONLY State Evolution."""
    from contracts.state_evolution import EvolutionIRV1, verify_evolution_plan

    model = source.get('model') or {}
    spec = source.get('spec') or {}
    registry = source.get('registry') or {}
    if model.get('modelId') != spec.get('modelId') or registry != {
        'modelId': model.get('modelId'), 'specId': spec.get('specId')
    }:
        raise ValueError('CONTINUOUS_SOURCE_BINDING_MISMATCH')
    selected = next((item for item in spec.get('scenarios', []) if item.get('id') == scenario_id), None)
    if selected is None:
        raise ValueError('UNKNOWN_CONTINUOUS_SCENARIO')

    def pin(kind, reference, value):
        return {'artifact_kind': kind, 'artifact_ref': reference, 'artifact_hash': evolution_hash(value)}

    model_pin = pin('system_dynamics_model', 'system-dynamics-model:' + str(model['modelId']), model)
    spec_pin = pin('system_dynamics_simulation_spec', 'system-dynamics-spec:' + str(spec['specId']), spec)
    registry_pin = pin('system_dynamics_registry', 'system-dynamics-registry:' + str(model['modelId']), registry)
    origin_payload = {'structuralModel': model_pin, 'executableSpec': spec_pin, 'registry': registry_pin}
    origin_pin = pin('state_evolution_continuous_basis', 'continuous-basis:' + evolution_hash(origin_payload).split(':')[1], origin_payload)
    source_pins = {
        'origin': origin_pin,
        'structural-model': model_pin,
        'executable-spec': spec_pin,
        'registry': registry_pin,
        'compiler': compiler,
        'continuous-backend': backend,
        'producer': producer,
    }

    coordinates = (spec.get('quantityBasis') or {}).get('coordinates') or {
        **{item['id']:{'unit':item.get('unit') or spec.get('stocks',{}).get(item['id'],{}).get('unit','dimensionless')} for item in model.get('stocks',[])},
        **{item['id']:{'unit':item['unit']} for item in model.get('flows',[])},
        **{item['id']:{'unit':item.get('unit','dimensionless')} for item in model.get('auxiliaries',[])},
        **{ident:{'unit':item['unit']} for ident,item in (spec.get('parameters') or {}).items()},
    }
    stocks = spec.get('stocks') or {}
    parameters = spec.get('parameters') or {}
    flow_ids = {str(item['id']) for item in model.get('flows', [])}
    auxiliary_ids = {str(item['id']) for item in model.get('auxiliaries', [])}
    symbols = []
    for ident, coordinate in coordinates.items():
        if ident == 'time':
            continue
        unit = str(coordinate['unit'])
        if ident in stocks:
            source_pin, kind, status, value = model_pin, 'continuous', 'KNOWN', stocks[ident]['initialValue']
            epistemic = 'ASSUMED'
        elif ident in parameters:
            source_pin, kind, status, value = spec_pin, 'input', 'KNOWN', parameters[ident]['value']
            epistemic = 'ASSUMED'
        elif ident in flow_ids or ident in auxiliary_ids:
            source_pin, kind, status, value = model_pin, 'derived', 'UNKNOWN', None
            epistemic = 'DERIVED'
        else:
            raise ValueError('UNBOUND_CONTINUOUS_SYMBOL:' + ident)
        symbol = {
            'id': ident, 'kind': kind, 'valueType': 'number', 'unit': unit,
            'sourcePin': source_pin, 'objectId': str(model['modelId']), 'fieldId': ident,
            'effectiveAt': effective_at, 'freshnessBasis': 'pinned-system-dynamics-source',
            'epistemicStatus': epistemic, 'initialValue': {'status': status},
        }
        if status == 'KNOWN':
            symbol['initialValue']['value'] = value
        if ident in parameters and parameters[ident].get('range') is not None:
            symbol['range'] = parameters[ident]['range']
        symbols.append(symbol)

    scenario = {'id': scenario_id, 'overrides': dict(selected.get('parameterOverrides') or {}), 'scheduledInputs': []}
    if scenario_id != 'baseline':
        if (not isinstance(comparison_basis_ref, dict) or
                comparison_basis_ref.get('artifact_kind') != 'evolution_run' or
                not isinstance(comparison_basis_ref.get('artifact_ref'), str) or
                not comparison_basis_ref['artifact_ref'].startswith('evolution-run:') or
                not isinstance(comparison_basis_ref.get('artifact_hash'), str) or
                not comparison_basis_ref['artifact_hash'].startswith('sha256:')):
            raise ValueError('CONTINUOUS_BASELINE_BINDING_REQUIRED')
    time = spec['time']
    time_basis = {'kind': 'domain', 'unit': time['unit'], 'start': time['start'], 'stop': time['stop'], 'dt': time['dt']}
    equations = {**(spec.get('auxiliaries') or {}), **(spec.get('flowEquations') or {})}
    ir = {
        'schemaVersion': 'evolution_ir.v1', 'profile': 'CONTINUOUS_ONLY', 'scope': scope,
        'sourcePins': source_pins, 'symbols': symbols, 'discreteTransitions': [],
        'continuousEquations': equations, 'eventBindings': [], 'timeBasis': time_basis,
        'scenario': scenario, 'quantityBasis': spec.get('quantityBasis') or {'kind': 'physical'},
        'claimBoundary': str(spec.get('claimBoundary') or 'SIMULATED continuous model; no external effects.'),
    }
    EvolutionIRV1.model_validate(ir)

    source_map = {symbol['id']: symbol['sourcePin'] for symbol in symbols}
    program = {
        'operation': 'run-continuous', 'modelId': model['modelId'], 'specId': spec['specId'],
        'scenarioId': scenario_id, 'method': time['method'],
    }
    plan = {
        'schemaVersion': 'evolution_execution_plan.v1', 'profile': 'CONTINUOUS_ONLY',
        'scope': scope,
        'irRef': {'artifact_kind': 'evolution_ir', 'artifact_ref': 'evolution-ir:' + evolution_hash(ir).split(':')[1], 'artifact_hash': evolution_hash(ir)},
        'compiler': compiler, 'backends': {'continuous': backend}, 'producer': producer,
        'timeBasis': time_basis, 'scenario': scenario, 'programs': {'continuous': program},
        'sourceMap': source_map,
    }
    if scenario_id != 'baseline':
        plan['comparisonBasisRef'] = comparison_basis_ref
    plan['basisHash'] = evolution_hash(plan)
    verify_evolution_plan(plan)
    return ir, plan


def compile_hybrid_evolution(*, scope, source, scenario_id, compiler, workflow_backend, continuous_backend, producer, effective_at):
    """Lower one owner-resolved workflow+SD composition into both bounded programs."""
    from contracts.state_evolution import EvolutionIRV1, verify_evolution_plan

    workflow = source.get('workflow') or {}
    execution = source.get('execution') or {}
    model = source.get('model') or {}
    spec = source.get('spec') or {}
    registry = source.get('registry') or {}
    if (execution.get('schemaVersion') != 'workflow_execution_ir.v1' or
            execution.get('workflowId') != workflow.get('workflowId') or
            execution.get('workflowVersion') != workflow.get('workflowVersion')):
        raise ValueError('HYBRID_WORKFLOW_EXECUTION_MISMATCH')
    if model.get('modelId') != spec.get('modelId') or registry != {'modelId': model.get('modelId'), 'specId': spec.get('specId')}:
        raise ValueError('HYBRID_CONTINUOUS_SOURCE_MISMATCH')
    selected = next((item for item in spec.get('scenarios', []) if item.get('id') == scenario_id), None)
    if selected is None:
        raise ValueError('UNKNOWN_HYBRID_SCENARIO')

    def pin(kind, reference, value):
        return {'artifact_kind':kind,'artifact_ref':reference,'artifact_hash':evolution_hash(value)}

    workflow_pin = pin('workflow_definition', 'workflow:' + str(workflow['workflowId']) + ':' + str(workflow['workflowVersion']), workflow)
    execution_pin = pin('workflow_execution_ir', 'workflow-execution:' + str(workflow['workflowId']) + ':' + str(workflow['workflowVersion']), execution)
    model_pin = pin('system_dynamics_model', 'system-dynamics-model:' + str(model['modelId']), model)
    spec_pin = pin('system_dynamics_simulation_spec', 'system-dynamics-spec:' + str(spec['specId']), spec)
    registry_pin = pin('system_dynamics_registry', 'system-dynamics-registry:' + str(model['modelId']), registry)
    origin_payload = {'workflow':workflow_pin,'workflowExecution':execution_pin,'structuralModel':model_pin,'executableSpec':spec_pin,'registry':registry_pin}
    origin_pin = pin('hybrid_composition', 'hybrid-composition:' + evolution_hash(origin_payload).split(':')[1], origin_payload)
    source_pins = {'origin':origin_pin,'workflow-definition':workflow_pin,'workflow-execution':execution_pin,
                   'structural-model':model_pin,'executable-spec':spec_pin,'registry':registry_pin,
                   'compiler':compiler,'workflow-backend':workflow_backend,
                   'continuous-backend':continuous_backend,'producer':producer}

    coordinates = (spec.get('quantityBasis') or {}).get('coordinates') or {
        **{item['id']:{'unit':item.get('unit') or spec.get('stocks',{}).get(item['id'],{}).get('unit','dimensionless')} for item in model.get('stocks',[])},
        **{item['id']:{'unit':item['unit']} for item in model.get('flows',[])},
        **{item['id']:{'unit':item.get('unit','dimensionless')} for item in model.get('auxiliaries',[])},
        **{ident:{'unit':item['unit']} for ident,item in (spec.get('parameters') or {}).items()},
    }
    stocks = spec.get('stocks') or {}
    parameters = spec.get('parameters') or {}
    flow_ids = {str(item['id']) for item in model.get('flows', [])}
    auxiliary_ids = {str(item['id']) for item in model.get('auxiliaries', [])}
    symbols = []
    for ident, coordinate in coordinates.items():
        if ident == 'time':
            continue
        unit = str(coordinate['unit'])
        if ident in stocks:
            source_pin, kind, status, value, value_type, epistemic = model_pin, 'continuous', 'KNOWN', stocks[ident]['initialValue'], 'number', 'ASSUMED'
        elif ident in parameters:
            source_pin, kind, status, value, value_type, epistemic = spec_pin, 'input', 'KNOWN', parameters[ident]['value'], 'number', 'ASSUMED'
        elif ident in flow_ids or ident in auxiliary_ids:
            source_pin, kind, status, value, value_type, epistemic = model_pin, 'derived', 'UNKNOWN', None, 'number', 'DERIVED'
        else:
            raise ValueError('UNBOUND_HYBRID_SYMBOL:' + str(ident))
        item = {'id':ident,'kind':kind,'valueType':value_type,'unit':unit,'sourcePin':source_pin,
                'objectId':str(model['modelId']),'fieldId':ident,'effectiveAt':effective_at,
                'freshnessBasis':'pinned-hybrid-source','epistemicStatus':epistemic,'initialValue':{'status':status}}
        if status == 'KNOWN':
            item['initialValue']['value'] = value
        if ident in parameters and parameters[ident].get('range') is not None:
            item['range'] = parameters[ident]['range']
        symbols.append(item)
    for item in source.get('discreteSymbols', []):
        symbols.append({'id':item['id'],'kind':'discrete','valueType':item.get('valueType','number'),
            'unit':item.get('unit','dimensionless'),'sourcePin':origin_pin,'objectId':str(workflow['workflowId']),
            'fieldId':item['id'],'effectiveAt':effective_at,'freshnessBasis':'pinned-hybrid-composition',
            'epistemicStatus':'ASSUMED','initialValue':{'status':'KNOWN','value':item['initialValue']},
            **({'range':item['range']} if item.get('range') is not None else {})})
    initial_node = source.get('initialWorkflowState')
    if initial_node not in {node['id'] for node in workflow.get('nodes', [])}:
        raise ValueError('HYBRID_INITIAL_WORKFLOW_NODE_REQUIRED')
    symbols.extend([
        {'id':'workflow.run-status','kind':'discrete','valueType':'string','unit':'status','sourcePin':workflow_pin,
         'objectId':workflow['workflowId'],'fieldId':'run-status','effectiveAt':effective_at,'freshnessBasis':'workflow-reducer',
         'epistemicStatus':'DERIVED','initialValue':{'status':'KNOWN','value':'running'}},
        *[{'id':'workflow.node-status.'+node['id'],'kind':'discrete','valueType':'string','unit':'status','sourcePin':workflow_pin,
           'objectId':workflow['workflowId'],'fieldId':'node-status.'+node['id'],'effectiveAt':effective_at,
           'freshnessBasis':'workflow-reducer','epistemicStatus':'DERIVED',
           'initialValue':{'status':'KNOWN','value':'running' if node['id']==initial_node else 'not_reached'}}
          for node in workflow['nodes']],
    ])
    symbols_by_id = {item['id']:item for item in symbols}
    if len(symbols_by_id) != len(symbols):
        raise ValueError('DUPLICATE_HYBRID_SYMBOL')

    transitions_by_id = {item['edgeRef']:item for item in execution.get('controllers', [])}
    if len(transitions_by_id) != len(execution.get('controllers', [])) or set(transitions_by_id) != {edge['id'] for edge in workflow.get('edges', [])}:
        raise ValueError('HYBRID_CONTROLLER_COVERAGE_MISMATCH')
    coordinator_transitions = []
    ir_transitions = []
    for edge in workflow.get('edges', []):
        controller = transitions_by_id[edge['id']]
        if (controller.get('sourceRef'),controller.get('targetRef')) != (edge['source'],edge['target']):
            raise ValueError('HYBRID_CONTROLLER_SOURCE_MISMATCH')
        if controller.get('controllerKind') not in ('normal','conditional'):
            raise ValueError('UNSUPPORTED_HYBRID_CONTROLLER:' + str(controller.get('controllerKind')))
        condition = controller.get('conditionRef')
        guard = {'kind':'input','id':condition} if condition else {'kind':'constant','value':True}
        coordinator_transitions.append({'id':edge['id'],'source':edge['source'],'target':edge['target'],'guard':guard})
        ir_transitions.append({'id':edge['id'],'source':edge['source'],'target':edge['target'],'guard':guard,'sourcePin':workflow_pin})

    bindings = copy.deepcopy(source.get('eventBindings') or [])
    schedule = copy.deepcopy((source.get('scenarioSchedules') or {}).get(scenario_id, []))
    scenario = {'id':scenario_id,'overrides':dict(selected.get('parameterOverrides') or {}),'scheduledInputs':schedule}
    coordinator_symbols = {ident:{'kind':item['kind'],'valueType':item['valueType'],
        'initialValue':item['initialValue'].get('value'),'range':item.get('range')}
        for ident,item in symbols_by_id.items() if ident in stocks or ident in parameters or ident in flow_ids or ident in auxiliary_ids or ident in {item['id'] for item in source.get('discreteSymbols', [])}}
    coordinator_symbols['workflow.run-status'] = {'kind':'discrete','valueType':'string','initialValue':'running'}
    for node in workflow['nodes']:
        coordinator_symbols['workflow.node-status.'+node['id']] = {'kind':'discrete','valueType':'string','initialValue':'running' if node['id']==initial_node else 'not_reached'}
    coordinator_plan = {'timeBasis':{'kind':'domain','start':spec['time']['start'],'stop':spec['time']['stop'],'dt':spec['time']['dt']},
        'scenario':{'id':scenario_id,'scheduledInputs':schedule},'symbols':coordinator_symbols,
        'transitions':coordinator_transitions,'eventBindings':bindings,'initialWorkflowState':initial_node}
    time = spec['time']
    time_basis = {'kind':'domain','unit':time['unit'],'start':time['start'],'stop':time['stop'],'dt':time['dt']}
    continuous_equations = {**(spec.get('auxiliaries') or {}),**(spec.get('flowEquations') or {})}
    ir = {'schemaVersion':'evolution_ir.v1','profile':'HYBRID_WORKFLOW_CONTINUOUS','scope':scope,'sourcePins':source_pins,
        'symbols':symbols,'discreteTransitions':ir_transitions,'continuousEquations':continuous_equations,
        'eventBindings':bindings,'timeBasis':time_basis,'scenario':scenario,
        'quantityBasis':spec.get('quantityBasis') or {'kind':'physical'},
        'claimBoundary':str(spec.get('claimBoundary') or 'SIMULATED hybrid composition; no external effects.')}
    EvolutionIRV1.model_validate(ir)
    workflow_operators = {operator['nodeRef']:{'operatorKind':operator['operatorKind'],'binding':operator.get('binding')}
        for operator in execution.get('operators', [])}
    if len(workflow_operators) != len(execution.get('operators', [])) or set(workflow_operators) != {node['id'] for node in workflow.get('nodes', [])}:
        raise ValueError('HYBRID_OPERATOR_COVERAGE_MISMATCH')
    for node in workflow['nodes']:
        if workflow_operators[node['id']]['operatorKind'] not in ('wait','human','terminal') or workflow_operators[node['id']]['binding'] is not None:
            raise ValueError('UNSUPPORTED_HYBRID_OPERATOR:' + node['id'])
    source_map = {ident:item['sourcePin'] for ident,item in symbols_by_id.items()}
    workflow_program = {'operation':'run-hybrid-coordinator','workflowExecution':execution,
        'operatorOwners':{node['id']:'workflow-reducer' for node in workflow['nodes']},
        'coordinatorPlan':coordinator_plan}
    continuous_program = {'operation':'run-continuous','modelId':model['modelId'],'specId':spec['specId'],
        'scenarioId':scenario_id,'method':time['method']}
    plan = {'schemaVersion':'evolution_execution_plan.v1','profile':'HYBRID_WORKFLOW_CONTINUOUS','scope':scope,
        'irRef':{'artifact_kind':'evolution_ir','artifact_ref':'evolution-ir:'+evolution_hash(ir).split(':')[1],'artifact_hash':evolution_hash(ir)},
        'compiler':compiler,'backends':{'workflow':workflow_backend,'continuous':continuous_backend},'producer':producer,
        'timeBasis':time_basis,'scenario':scenario,'programs':{'workflow':workflow_program,'continuous':continuous_program},
        'sourceMap':source_map}
    plan['basisHash'] = evolution_hash(plan)
    verify_evolution_plan(plan)
    return ir, plan
