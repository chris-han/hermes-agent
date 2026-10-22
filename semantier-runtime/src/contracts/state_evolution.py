"""Non-authorizing State Evolution wire contracts and exact identity checks.

Source owners, producers and the governed artifact store retain authority.
These contracts never execute a workflow, solver, provider or mutation.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.artifact_pin import ArtifactPinV1
from eos.artifact_hashing import canonical_json, sha256_hex

HASH = r'^sha256:[0-9a-f]{64}$'
Profile = Literal['WORKFLOW_ONLY', 'CONTINUOUS_ONLY', 'HYBRID_WORKFLOW_CONTINUOUS']


def _number(value: int | float) -> str:
    if not math.isfinite(value):
        raise ValueError('NONFINITE_NUMBER')
    if isinstance(value, int) and abs(value) <= 2**53 - 1:
        return str(value)
    if isinstance(value, int) and int(float(value)) != value:
        raise ValueError('UNREPRESENTABLE_WIRE_INTEGER')
    value = float(value)
    if value == 0:
        return '0'
    text = repr(value)
    if 1e-6 <= abs(value) < 1e21:
        fixed = format(Decimal(text), 'f')
        return fixed.rstrip('0').rstrip('.') if '.' in fixed else fixed
    mantissa, exponent = text.split('e')
    return mantissa.removesuffix('.0') + 'e' + ('+' if int(exponent) >= 0 else '-') + str(abs(int(exponent)))


def evolution_canonical_json(value: Any) -> str:
    """ECMAScript-number bridge for the existing canonical JSON/hash owners.

Existing Python artifact hashes remain unchanged. Evolution's wire numbers
use the TS canonical serializer's finite binary64 representation, including
integer-valued floats, exponent formatting, negative zero and UTF-16 key order.
"""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode='json', exclude_unset=True)
    if value is None or isinstance(value, (str, bool)):
        return canonical_json(value)
    if isinstance(value, (int, float)):
        return _number(value)
    if isinstance(value, (tuple, list)):
        return '[' + ','.join(evolution_canonical_json(item) for item in value) + ']'
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        keys = sorted(value, key=lambda key: key.encode('utf-16-be'))
        return '{' + ','.join(canonical_json(key) + ':' + evolution_canonical_json(value[key]) for key in keys) + '}'
    raise ValueError('INVALID_JSON_VALUE')


def evolution_hash(value: Any) -> str:
    return 'sha256:' + sha256_hex(evolution_canonical_json(value))


def _utc(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError('UTC_TIME_REQUIRED')
    return value


class _Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, allow_inf_nan=False)

    @model_validator(mode='after')
    def optional_absence_is_not_explicit_null(self):
        for name in self.model_fields_set:
            if getattr(self, name) is None and not type(self).model_fields[name].is_required() and not (name == 'value' and isinstance(self, EvolutionValueV1)):
                raise ValueError('OPTIONAL_NULL_IS_NOT_ABSENCE')
        return self


class EvolutionScopeV1(_Contract):
    organizationId: str = Field(min_length=1)
    workspaceId: str = Field(min_length=1)
    sessionId: str = Field(min_length=1)
    actorRef: str = Field(min_length=1)


def verify_evolution_scope(scope: EvolutionScopeV1, trusted: dict) -> None:
    if scope.model_dump() != trusted:
        raise ValueError('EVOLUTION_SCOPE_MISMATCH')


def verify_evolution_sources(ir: EvolutionIRV1, resolved_owner_pins: dict) -> None:
    """Compare with pins independently resolved by existing source owners.

    This helper does not resolve ownership or trust pins supplied by a client.
    """
    if {role: pin.model_dump() for role, pin in ir.sourcePins.items()} != resolved_owner_pins:
        raise ValueError('EVOLUTION_SOURCE_PIN_MISMATCH')


class EvolutionValueV1(_Contract):
    status: Literal['KNOWN', 'UNKNOWN', 'ABSENT']
    value: float | bool | str | None = None

    @model_validator(mode='after')
    def preserve_absence(self):
        if ('value' in self.model_fields_set) != (self.status == 'KNOWN'):
            raise ValueError('VALUE_STATUS_MISMATCH')
        return self


class EvolutionSymbolV1(_Contract):
    id: str = Field(pattern=r'^[A-Za-z_][A-Za-z0-9_.-]*$')
    kind: Literal['discrete', 'continuous', 'input', 'derived']
    valueType: Literal['number', 'boolean', 'string']
    unit: str = Field(min_length=1)
    sourcePin: ArtifactPinV1
    objectId: str = Field(min_length=1)
    fieldId: str = Field(min_length=1)
    effectiveAt: str
    freshnessBasis: str = Field(min_length=1)
    epistemicStatus: Literal['OBSERVED', 'ASSUMED', 'DERIVED', 'UNKNOWN']
    initialValue: EvolutionValueV1
    range: list[float] | None = Field(default=None, min_length=2, max_length=2)

    @model_validator(mode='after')
    def check_source_value(self):
        _utc(self.effectiveAt)
        if self.range is not None and (self.valueType != 'number' or self.range[0] > self.range[1]):
            raise ValueError('INVALID_SYMBOL_RANGE')
        value = self.initialValue.value
        if self.initialValue.status == 'KNOWN' and value is not None:
            expected = {'number': (int, float), 'boolean': (bool,), 'string': (str,)}[self.valueType]
            if not isinstance(value, expected) or (self.valueType == 'number' and isinstance(value, bool)):
                raise ValueError('SYMBOL_VALUE_TYPE_MISMATCH')
            self.validate_value(value)
        return self

    def validate_value(self, value):
        if value is None:
            return
        expected = {'number': (int, float), 'boolean': (bool,), 'string': (str,)}[self.valueType]
        if not isinstance(value, expected) or (self.valueType == 'number' and isinstance(value, bool)):
            raise ValueError('SYMBOL_VALUE_TYPE_MISMATCH')
        evolution_canonical_json(value)
        if self.range is not None and not self.range[0] <= value <= self.range[1]:
            raise ValueError('SYMBOL_VALUE_OUT_OF_RANGE')


def _dimension(unit: str) -> dict[str, int]:
    import re
    result = {}
    parts = unit.split('/')
    if len(parts) > 2:
        raise ValueError('UNSUPPORTED_UNIT_ALGEBRA')
    for index, part in enumerate(parts):
        for atom in part.split('*'):
            if atom in ('1', 'dimensionless'):
                continue
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', atom):
                raise ValueError('UNSUPPORTED_UNIT_ALGEBRA')
            result[atom] = result.get(atom, 0) + (1 if index == 0 else -1)
    return {key: value for key, value in result.items() if value}


def _expr(expr: dict, units: dict[str, str], *, normalized=False, depth=0, budget=None):
    budget = [512] if budget is None else budget
    budget[0] -= 1
    if depth > 32 or budget[0] < 0 or not isinstance(expr, dict):
        raise ValueError('EXPRESSION_BUDGET_EXCEEDED')
    kind = expr.get('kind')
    if kind == 'constant' and set(expr) == {'kind', 'value'}:
        if isinstance(expr['value'], bool) or not isinstance(expr['value'], (int, float)):
            raise ValueError('INVALID_CONSTANT')
        evolution_canonical_json(expr['value'])
        return None
    if kind == 'ref' and set(expr) == {'kind', 'id'} and expr['id'] in units:
        return {} if normalized else _dimension(units[expr['id']])
    if kind == 'clamp' and set(expr) == {'kind', 'value', 'min', 'max'}:
        if any(isinstance(expr[key], bool) or not isinstance(expr[key], (int, float)) for key in ('min', 'max')) or not math.isfinite(expr['min']) or not math.isfinite(expr['max']) or expr['min'] > expr['max']:
            raise ValueError('INVALID_CLAMP')
        return _expr(expr['value'], units, normalized=normalized, depth=depth+1, budget=budget)
    if kind not in ('add', 'sub', 'mul', 'div', 'min', 'max') or set(expr) != {'kind', 'left', 'right'}:
        raise ValueError('UNSUPPORTED_EXPRESSION')
    left = _expr(expr['left'], units, normalized=normalized, depth=depth+1, budget=budget)
    right = _expr(expr['right'], units, normalized=normalized, depth=depth+1, budget=budget)
    if kind not in ('mul', 'div'):
        if left is not None and right is not None and left != right:
            raise ValueError('EXPRESSION_UNIT_MISMATCH')
        return left if left is not None else right
    result = dict(left or {})
    for atom, power in (right or {}).items():
        result[atom] = result.get(atom, 0) + (-power if kind == 'div' else power)
    return {key: value for key, value in result.items() if value}


class EvolutionTransitionV1(_Contract):
    id: str
    source: str
    target: str
    guard: dict[str, Any]
    sourcePin: ArtifactPinV1


class EvolutionResetV1(_Contract):
    symbolId: str
    expression: dict[str, Any]


class EvolutionWorkflowTriggerV1(_Contract):
    kind: Literal['workflow-transition']


class EvolutionThresholdTriggerV1(_Contract):
    kind: Literal['threshold']
    symbolId: str
    operator: Literal['le', 'ge']
    value: float
    direction: Literal['down', 'up']
    monotonic: Literal['decreasing', 'increasing']

    @model_validator(mode='after')
    def check_supported_direction(self):
        if (self.operator, self.direction, self.monotonic) not in (('le', 'down', 'decreasing'), ('ge', 'up', 'increasing')):
            raise ValueError('UNSUPPORTED_THRESHOLD_DIRECTION')
        return self


class EvolutionEventBindingV1(_Contract):
    eventId: str
    transitionId: str
    trigger: EvolutionWorkflowTriggerV1 | EvolutionThresholdTriggerV1
    writes: list[EvolutionResetV1] = Field(min_length=1)
    priority: int | None = None
    rearm: Literal['one-shot']


class EvolutionScheduledInputV1(_Contract):
    id: str
    time: float
    symbolId: str
    value: float | str | bool | None


class EvolutionTimeBasisV1(_Contract):
    kind: Literal['order-only', 'domain']
    unit: str | None = None
    start: float | None = None
    stop: float | None = None
    dt: float | None = None
    epoch: str | None = None

    @model_validator(mode='after')
    def check_time(self):
        if self.kind == 'order-only' and self.model_fields_set != {'kind'}:
            raise ValueError('ORDER_IS_NOT_DURATION')
        if self.kind == 'domain':
            if not self.unit or self.start is None or self.stop is None or not math.isfinite(self.start) or not math.isfinite(self.stop) or self.stop <= self.start:
                raise ValueError('INVALID_DOMAIN_TIME')
            if self.dt is not None and (not math.isfinite(self.dt) or self.dt <= 0):
                raise ValueError('INVALID_DT')
            if self.epoch is not None:
                _utc(self.epoch)
        return self


class EvolutionScenarioV1(_Contract):
    id: str
    overrides: dict[str, float | str | bool | None]
    scheduledInputs: list[EvolutionScheduledInputV1]


class EvolutionBindingSpecV1(_Contract):
    schemaVersion: Literal['evolution_binding_spec.v1']
    scope: EvolutionScopeV1
    sourcePins: dict[str, ArtifactPinV1]
    symbols: list[EvolutionSymbolV1]
    quantityBasis: dict[str, Any] | None = None
    eventBindings: list[EvolutionEventBindingV1] = []
    scenarioPermissions: list[str] = []
    timeBasis: EvolutionTimeBasisV1

    @model_validator(mode='after')
    def check_bound_symbols(self):
        pins = [pin.model_dump() for pin in self.sourcePins.values()]
        ids = {item.id for item in self.symbols}
        if len(ids) != len(self.symbols) or any(item.sourcePin.model_dump() not in pins for item in self.symbols):
            raise ValueError('INVALID_SYMBOL_SOURCE_BINDING')
        allowed = {item.id for item in self.symbols if item.kind == 'input'}
        if len(set(self.scenarioPermissions)) != len(self.scenarioPermissions) or not set(self.scenarioPermissions) <= allowed:
            raise ValueError('INVALID_SCENARIO_PERMISSIONS')
        if self.quantityBasis is not None:
            units = {item.id: item.unit for item in self.symbols}
            if self.timeBasis.unit:
                units['time'] = self.timeBasis.unit
            if self.quantityBasis.get('kind') == 'normalized-index':
                if self.quantityBasis != {'kind': 'normalized-index', 'version': '1', 'arithmetic': 'dimensionless-coordinates', 'coordinates': {key: {'unit': unit, 'referenceValue': 1} for key, unit in units.items()}} or any(isinstance(coordinate['referenceValue'], bool) for coordinate in self.quantityBasis['coordinates'].values()):
                    raise ValueError('INVALID_QUANTITY_COORDINATES')
            elif self.quantityBasis != {'kind': 'physical'}:
                raise ValueError('UNSUPPORTED_QUANTITY_BASIS')
        return self


class EvolutionIRV1(_Contract):
    schemaVersion: Literal['evolution_ir.v1']
    profile: Profile
    scope: EvolutionScopeV1
    sourcePins: dict[str, ArtifactPinV1]
    symbols: list[EvolutionSymbolV1]
    discreteTransitions: list[EvolutionTransitionV1]
    continuousEquations: dict[str, dict[str, Any]]
    eventBindings: list[EvolutionEventBindingV1]
    timeBasis: EvolutionTimeBasisV1
    scenario: EvolutionScenarioV1
    quantityBasis: dict[str, Any] | None = None
    claimBoundary: str = Field(min_length=1)

    @model_validator(mode='after')
    def validate_mechanism_closure(self):
        discrete = self.profile != 'CONTINUOUS_ONLY'
        continuous = self.profile != 'WORKFLOW_ONLY'
        if bool(self.discreteTransitions) != discrete or bool(self.continuousEquations) != continuous:
            raise ValueError('PROFILE_MECHANISM_MISMATCH')
        required = {'origin'} | ({'workflow-definition'} if discrete else set()) | ({'structural-model', 'executable-spec', 'registry'} if continuous else set())
        if not required <= self.sourcePins.keys():
            raise ValueError('MISSING_SOURCE_OWNER')
        if continuous and (self.timeBasis.kind != 'domain' or self.timeBasis.dt is None or self.quantityBasis is None):
            raise ValueError('CONTINUOUS_BASIS_REQUIRED')
        if (self.profile == 'HYBRID_WORKFLOW_CONTINUOUS') != bool(self.eventBindings):
            raise ValueError('EVENT_BINDING_PROFILE_MISMATCH')
        symbols = {item.id: item for item in self.symbols}
        if len(symbols) != len(self.symbols):
            raise ValueError('DUPLICATE_SYMBOL')
        pins = [pin.model_dump() for pin in self.sourcePins.values()]
        for item in [*self.symbols, *self.discreteTransitions]:
            if item.sourcePin.model_dump() not in pins:
                raise ValueError('UNBOUND_SOURCE_PIN')
        for ident in self.scenario.overrides:
            if ident not in symbols or symbols[ident].kind != 'input':
                raise ValueError('INVALID_SCENARIO_OVERRIDE')
            symbols[ident].validate_value(self.scenario.overrides[ident])
        scheduled_ids = set()
        for entry in self.scenario.scheduledInputs:
            if entry.id in scheduled_ids or entry.symbolId not in symbols or symbols[entry.symbolId].kind != 'input' or self.timeBasis.kind != 'domain' or not math.isfinite(entry.time) or not self.timeBasis.start <= entry.time <= self.timeBasis.stop:
                raise ValueError('INVALID_SCHEDULED_INPUT')
            symbols[entry.symbolId].validate_value(entry.value)
            scheduled_ids.add(entry.id)
        units = {item.id: item.unit for item in self.symbols}
        if self.timeBasis.unit:
            units['time'] = self.timeBasis.unit
        normalized = self.quantityBasis is not None and self.quantityBasis.get('kind') == 'normalized-index'
        if self.quantityBasis is not None:
            if normalized:
                basis = self.quantityBasis
                if set(basis) != {'kind', 'version', 'arithmetic', 'coordinates'} or basis['version'] != '1' or basis['arithmetic'] != 'dimensionless-coordinates' or basis['coordinates'] != {key: {'unit': unit, 'referenceValue': 1} for key, unit in units.items()} or any(isinstance(coordinate['referenceValue'], bool) for coordinate in basis['coordinates'].values()):
                    raise ValueError('INVALID_QUANTITY_COORDINATES')
            elif self.quantityBasis != {'kind': 'physical'}:
                raise ValueError('UNSUPPORTED_QUANTITY_BASIS')
        for ident, expr in self.continuousEquations.items():
            if ident not in symbols:
                raise ValueError('UNBOUND_EQUATION_OUTPUT')
            actual = _expr(expr, units, normalized=normalized)
            expected_unit = units[ident] + '/' + self.timeBasis.unit if symbols[ident].kind == 'continuous' else units[ident]
            if not normalized and actual is not None and actual != _dimension(expected_unit):
                raise ValueError('EXPRESSION_UNIT_MISMATCH')
        for transition in self.discreteTransitions:
            guard = transition.guard
            if guard.get('kind') == 'input':
                valid_guard = set(guard) == {'kind','id'} and guard.get('id') in symbols and symbols[guard['id']].kind == 'input'
            elif guard.get('kind') == 'constant':
                valid_guard = set(guard) == {'kind','value'} and type(guard.get('value')) is bool
            elif guard.get('kind') == 'owner-condition':
                owner = self.sourcePins.get('workflow-definition')
                valid_guard = set(guard) == {'kind','ref','sourcePin'} and isinstance(guard.get('ref'),str) and bool(guard['ref']) and owner is not None and guard['sourcePin'] == owner.model_dump()
            else:
                valid_guard = False
            if not valid_guard:
                raise ValueError('UNSUPPORTED_GUARD')
        transition_ids = {item.id for item in self.discreteTransitions}
        if len(transition_ids) != len(self.discreteTransitions):
            raise ValueError('DUPLICATE_TRANSITION')
        event_ids = set()
        for binding in self.eventBindings:
            if binding.eventId in event_ids or binding.transitionId not in transition_ids:
                raise ValueError('INVALID_EVENT_BINDING')
            event_ids.add(binding.eventId)
            if isinstance(binding.trigger, EvolutionThresholdTriggerV1):
                variable = symbols.get(binding.trigger.symbolId)
                if variable is None or variable.kind != 'continuous' or variable.valueType != 'number':
                    raise ValueError('UNBOUND_THRESHOLD_QUANTITY')
            writers = set()
            for reset in binding.writes:
                if reset.symbolId in writers or reset.symbolId not in symbols or symbols[reset.symbolId].kind not in ('input', 'continuous', 'discrete'):
                    raise ValueError('INVALID_RESET_WRITER')
                writers.add(reset.symbolId)
                actual = _expr(reset.expression, units, normalized=normalized)
                if not normalized and actual is not None and actual != _dimension(units[reset.symbolId]):
                    raise ValueError('RESET_UNIT_MISMATCH')
        # W2/W6 lower source-native controller guards. Unsupported contracts
        # cannot masquerade as compiled executable programs in this first slice.
        evolution_canonical_json(self.model_dump(mode='json', exclude_unset=True))
        return self


class EvolutionExecutionPlanV1(_Contract):
    schemaVersion: Literal['evolution_execution_plan.v1']
    profile: Profile
    scope: EvolutionScopeV1
    irRef: ArtifactPinV1
    compiler: ArtifactPinV1
    backends: dict[str, ArtifactPinV1]
    producer: ArtifactPinV1
    basisHash: str = Field(pattern=HASH)
    timeBasis: EvolutionTimeBasisV1
    scenario: EvolutionScenarioV1
    programs: dict[str, dict[str, Any]]
    sourceMap: dict[str, ArtifactPinV1]
    comparisonBasisRef: ArtifactPinV1 | None = None


class EvolutionStateDeltaV1(_Contract):
    symbolId: str
    before: EvolutionValueV1
    after: EvolutionValueV1


class EvolutionEventV1(_Contract):
    id: str
    kind: str
    semanticCategory: Literal['WORKFLOW_STATE', 'MODELED_EVENT', 'OBSERVATION'] | None = None
    operatorRef: str | None = None
    semanticTime: float | None
    microstep: int = Field(ge=0)
    order: int = Field(ge=0)
    sourceRefs: list[ArtifactPinV1]
    deltas: list[EvolutionStateDeltaV1]
    payload: dict[str, Any]
    observedAt: str | None = None


class EvolutionRunV1(_Contract):
    schemaVersion: Literal['evolution_run.v1']
    runId: str = Field(pattern=HASH)
    profile: Profile
    scope: EvolutionScopeV1
    planRef: ArtifactPinV1
    basisHash: str = Field(pattern=HASH)
    resultHash: str = Field(pattern=HASH)
    producer: ArtifactPinV1
    recordedAt: str
    attemptId: str
    comparisonBasisRef: ArtifactPinV1 | None = None
    status: Literal['SIMULATED', 'NEEDS_INPUT', 'UNSUPPORTED', 'INVALID', 'NUMERICAL_FAILURE', 'CANCELLED', 'BUDGET_EXCEEDED', 'CLIENT_PREVIEW']
    events: list[EvolutionEventV1]
    series: dict[str, Any]
    finalState: dict[str, EvolutionValueV1]
    diagnostics: list[dict[str, Any]]

    @model_validator(mode='after')
    def check_recorded_at(self):
        _utc(self.recordedAt)
        return self


def verify_evolution_plan(wire: dict) -> EvolutionExecutionPlanV1:
    plan = EvolutionExecutionPlanV1.model_validate(wire)
    required = {'workflow'} if plan.profile == 'WORKFLOW_ONLY' else {'continuous'} if plan.profile == 'CONTINUOUS_ONLY' else {'workflow', 'continuous'}
    if set(plan.backends) != required or set(plan.programs) != required:
        raise ValueError('EVOLUTION_BACKEND_PROFILE_MISMATCH')
    if plan.irRef.artifact_kind != 'evolution_ir' or plan.basisHash != evolution_hash({key: value for key, value in wire.items() if key != 'basisHash'}):
        raise ValueError('EVOLUTION_BASIS_HASH_MISMATCH')
    return plan


def verify_evolution_run(wire: dict, plan_wire: dict, *, trusted_scope: dict) -> EvolutionRunV1:
    plan = verify_evolution_plan(plan_wire)
    run = EvolutionRunV1.model_validate(wire)
    verify_evolution_scope(plan.scope, trusted_scope)
    verify_evolution_scope(run.scope, trusted_scope)
    if run.planRef.artifact_kind != 'evolution_execution_plan' or run.planRef.artifact_hash != evolution_hash(plan_wire) or run.basisHash != plan.basisHash or run.profile != plan.profile or run.producer != plan.producer or run.comparisonBasisRef != plan.comparisonBasisRef:
        raise ValueError('EVOLUTION_RUN_BINDING_MISMATCH')
    result = {key: wire[key] for key in ('status', 'events', 'series', 'finalState', 'diagnostics')}
    if run.resultHash != evolution_hash(result) or run.runId != evolution_hash({key: value for key, value in wire.items() if key != 'runId'}):
        raise ValueError('EVOLUTION_RESULT_HASH_MISMATCH')
    outputs = set(run.series) | set(run.finalState) | {delta.symbolId for event in run.events for delta in event.deltas}
    if not outputs <= plan.sourceMap.keys():
        raise ValueError('EVOLUTION_UNBOUND_OUTPUT')
    previous = None
    for event in run.events:
        if event.observedAt is not None:
            _utc(event.observedAt)
        if (plan.timeBasis.kind == 'order-only') != (event.semanticTime is None):
            raise ValueError('EVOLUTION_EVENT_TIME_BASIS_MISMATCH')
        if event.semanticTime is not None and (not math.isfinite(event.semanticTime) or not plan.timeBasis.start <= event.semanticTime <= plan.timeBasis.stop):
            raise ValueError('EVOLUTION_EVENT_TIME_OUT_OF_BOUNDS')
        coordinate = (event.semanticTime or 0, event.microstep, event.order)
        if previous is not None and coordinate <= previous:
            raise ValueError('EVOLUTION_NONMONOTONE_EVENT_ORDER')
        previous = coordinate
    return run
