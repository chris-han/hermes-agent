"""DEVELOPMENT ACE given-clause mapping and independent reference scoring.

This module does not dispatch models, resolve authority, expose references to
actors, or certify legal compliance. Native execution is a separate prerequisite.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eos.artifact_hashing import content_hash

LABELS = ('Compliant', 'Non-Compliant', 'Not-Applicable')


@dataclass(frozen=True)
class AceTask:
    member_id: str
    source_component: str
    role: str
    # Immutable pairs prevent mutation after input identity is calculated.
    clauses: tuple[tuple[str, str], ...]
    scenario_text: str
    input_hash: str

    def agent_input(self) -> dict[str, Any]:
        return {'clauses': dict(self.clauses), 'scenario_text': self.scenario_text}


@dataclass(frozen=True)
class AceReference:
    member_id: str
    label: str
    reference_hash: str


def materialize_ace_member(member: dict[str, Any]) -> tuple[AceTask, AceReference]:
    """Allowlist native input fields; keep teacher/gold in the evaluator view.

    First adapter accepts exposed apparatus/development cases only. Source
    contract mapping and protected split eligibility are intentionally unproved.
    """
    if member.get('role') not in {'A', 'D'}:
        raise ValueError('only exposed apparatus/development cases are qualified')
    for key in ('member_id', 'source_component'):
        if not isinstance(member.get(key), str) or not member[key].strip():
            raise ValueError(f'{key} required')
    row = member['row']
    clauses = row.get('clauses')
    scenario = row.get('scenario_text')
    if (not isinstance(clauses, dict) or not clauses or
            any(not isinstance(k, str) or not k or not isinstance(v, str) or not v.strip()
                for k, v in clauses.items()) or not isinstance(scenario, str) or not scenario.strip()):
        raise ValueError('native clauses/scenario required')
    label = row.get('gd_tr')
    if label not in LABELS:
        raise ValueError('native reference label required; UNKNOWN is not Not-Applicable')
    actor = {'clauses': dict(clauses), 'scenario_text': scenario}
    task = AceTask(member['member_id'], member['source_component'], member['role'],
                   tuple(clauses.items()), scenario, content_hash(actor))
    reference = AceReference(task.member_id, label, content_hash({'member_id': task.member_id,
                                                                 'input_hash': task.input_hash, 'label': label}))
    return task, reference


def grade_ace_attempts(references: list[AceReference], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Score the complete assigned cohort, preserving failures and missingness.

    Exact native labels after outer whitespace removal. Actor failures count
    incorrect; genuine environment faults and not-run slots stay ungraded.
    Macro-F1 has the fixed three native classes, including zero-support classes.
    """
    expected = {r.member_id: r for r in references}
    actual = {a['member_id']: a for a in attempts}
    if (not expected or len(expected) != len(references) or len(actual) != len(attempts)
            or set(expected) != set(actual)):
        raise ValueError('complete unique assigned member identities required')
    if any(r.label not in LABELS for r in references):
        raise ValueError('invalid independent reference')
    items = []
    for r in references:
        attempt = actual[r.member_id]
        status = attempt.get('status')
        if status not in {'completed', 'actor_failure', 'environment_fault', 'not_run'}:
            raise ValueError('explicit attempt status required')
        output = attempt.get('output')
        if output is not None and not isinstance(output, str):
            raise ValueError('raw output must be text or null')
        scored = status in {'completed', 'actor_failure'}
        prediction = output.strip() if status == 'completed' and isinstance(output, str) else None
        valid_label = prediction in LABELS
        items.append({'member_id': r.member_id, 'reference_hash': r.reference_hash,
                      'reference_label': r.label, 'status': status, 'raw_output': output,
                      'prediction': prediction, 'valid_label': valid_label,
                      'correct': int(valid_label and prediction == r.label) if scored else None})
    scored_items = [x for x in items if x['correct'] is not None]
    scored = len(scored_items)
    correct = sum(x['correct'] for x in scored_items)
    class_f1 = []
    for label in LABELS:
        tp = sum(x['reference_label'] == label and x['prediction'] == label for x in scored_items)
        fp = sum(x['reference_label'] != label and x['prediction'] == label for x in scored_items)
        fn = sum(x['reference_label'] == label and x['prediction'] != label for x in scored_items)
        class_f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    n = len(references)
    f1 = sum(class_f1) / len(LABELS) if scored else None
    return {'assigned': n, 'scored': scored, 'missing': n - scored, 'correct': correct,
            'coverage': scored / n, 'accuracy': correct / n if scored == n else None,
            'macro_f1': f1 if scored == n else None,
            'scorable_accuracy': correct / scored if scored else None, 'scorable_macro_f1': f1,
            'accuracy_missingness_bounds': [correct / n, (correct + n - scored) / n],
            'invalid_outputs': sum(not x['valid_label'] for x in scored_items),
            'classification_errors': scored - correct, 'legal_critical_errors': None,
            'legal_compliance': 'UNMEASURED', 'items': items}
