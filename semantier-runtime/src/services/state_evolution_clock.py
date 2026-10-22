"""Explicit scenario clock for the bounded Meeting controlled producer."""
from datetime import datetime, timedelta, timezone
import math
from decimal import Decimal
from contracts.state_evolution import EvolutionTimeBasisV1


def _microseconds(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('INVALID_SCENARIO_CLOCK')
    scaled = Decimal(str(value))*1_000_000
    if scaled != scaled.to_integral_value():
        raise ValueError('UNSUPPORTED_TIME_PRECISION: Meeting UTC timestamps require microsecond resolution')
    return int(scaled)


def meeting_time_iso(basis, instant):
    basis = EvolutionTimeBasisV1.model_validate(basis)
    if basis.kind != 'domain' or basis.unit != 'second' or basis.epoch is None:
        raise ValueError('UNSUPPORTED: Meeting clock requires second and explicit UTC epoch')
    epoch = datetime.fromisoformat(basis.epoch.replace('Z','+00:00'))
    try:
        return (epoch+timedelta(microseconds=_microseconds(instant))).astimezone(timezone.utc).isoformat().replace('+00:00','Z')
    except OverflowError as error:
        raise ValueError('UNSUPPORTED_TIME_RANGE') from error


class MeetingScenarioClock:
    def __init__(self, basis, schedule, *, followup_delay):
        self.basis = EvolutionTimeBasisV1.model_validate(basis)
        self.semantic_time = self.basis.start if self.basis.kind == 'domain' else None
        self.phases = {}
        self._order = 0
        self._microstep = 0
        self._last_event_time = object()
        self.followup_delay = followup_delay
        if self.basis.kind == 'order-only':
            if schedule or followup_delay is not None:
                raise ValueError('UNTIMED: domain schedule not available')
            return
        if self.basis.unit != 'second' or self.basis.epoch is None:
            raise ValueError('UNSUPPORTED: Meeting clock requires second and explicit UTC epoch')
        if isinstance(followup_delay,bool) or not isinstance(followup_delay,(int,float)) or not math.isfinite(followup_delay) or followup_delay < 0:
            raise ValueError('INVALID_FOLLOWUP_DELAY')
        meeting_time_iso(self.basis,self.basis.start)
        meeting_time_iso(self.basis,self.basis.stop)
        delay_microseconds = _microseconds(followup_delay)
        for entry in schedule:
            phase = entry['value']; instant = entry['time']
            if entry['symbolId'] != 'producer_phase' or not isinstance(phase,str) or phase in self.phases or isinstance(instant,bool) or not isinstance(instant,(int,float)) or not math.isfinite(instant) or not self.basis.start <= instant <= self.basis.stop:
                raise ValueError('INVALID_SCENARIO_SCHEDULE')
            _microseconds(instant)
            self.phases[phase] = instant
        if not {'WAIT_REGISTER','TIMER_DELIVERY'} <= set(self.phases):
            raise ValueError('UNSUPPORTED: unbound wait/timer schedule')
        self.deadline = (_microseconds(self.phases['WAIT_REGISTER'])+delay_microseconds)/1_000_000
        if self.deadline > self.basis.stop:
            raise ValueError('INVALID_DEADLINE_HORIZON')
        if self.phases['TIMER_DELIVERY'] < self.deadline:
            raise ValueError('TIMER_NOT_DUE')
        ordered = ['INITIALIZE','COLLECT_ORIGINAL','COLLECT_CANDIDATES','INITIAL_FRONTIER','WAIT_REGISTER']
        if 'EARLY_TIMER_DELIVERY' in self.phases:
            ordered.append('EARLY_TIMER_DELIVERY')
            if self.phases['EARLY_TIMER_DELIVERY'] >= self.deadline:
                raise ValueError('INVALID_EARLY_TIMER_PROBE')
        ordered.extend(['TIMER_DELIVERY','FOLLOWUP','PRESENT'])
        if set(self.phases) != set(ordered):
            raise ValueError('INVALID_BINDING: incomplete producer phase schedule')
        if self.phases['INITIALIZE'] != self.basis.start:
            raise ValueError('INVALID_INITIALIZATION_TIME')
        if any(self.phases[left] > self.phases[right] for left,right in zip(ordered,ordered[1:])):
            raise ValueError('BACKWARD_SCENARIO_SCHEDULE')

    def advance_phase(self, phase):
        if self.semantic_time is None:
            return
        if phase not in self.phases:
            raise ValueError('UNSUPPORTED: unbound producer phase '+phase)
        instant = self.phases[phase]
        if instant < self.semantic_time:
            raise ValueError('BACKWARD_SCENARIO_TIME')
        self.semantic_time = instant

    def now_iso(self):
        if self.semantic_time is None:
            raise ValueError('UNTIMED: domain clock not available')
        return meeting_time_iso(self.basis,self.semantic_time)

    def deadline_iso(self):
        if self.semantic_time is None:
            raise ValueError('UNTIMED: domain deadline not available')
        return meeting_time_iso(self.basis,self.deadline)

    def timer_due(self):
        return self.semantic_time is not None and self.semantic_time >= self.deadline

    def event_coordinate(self):
        if self._last_event_time == self.semantic_time:
            self._microstep += 1
        else:
            self._microstep = 0
            self._last_event_time = self.semantic_time
        result = {'semanticTime':self.semantic_time,'microstep':self._microstep,'order':self._order}
        self._order += 1
        return result
