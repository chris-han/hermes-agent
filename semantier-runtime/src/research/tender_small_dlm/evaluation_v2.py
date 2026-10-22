"""Opt-in strict admission for the unchanged native v1 research arithmetic.

Integrity errors are separate from semantic UNKNOWN and quality verdicts.
Descriptors establish metric definitions, never business calibration or quality.
"""
from collections.abc import Mapping
import math
from numbers import Real

from src.research.tender_small_dlm.evaluation import evaluate as legacy_evaluate
from src.research.tender_small_dlm.transforms import FAMILIES

VERSION = 'tender_small_dlm.evaluation.v2'
METRIC_NAMES = frozenset({
    'original_accuracy', 'A_recall', 'B_recall', 'macro_F1',
    'candidate_order_consistency', 'label_remap_consistency', 'paraphrase_consistency',
    'fixed_label_collapse_rate', 'frontier_correctness', 'unknown_accuracy',
    'counterfactual_pair_accuracy', 'hard_negative_gold_B_predicted_A',
    'hard_negative_accuracy', 'gold_B_predicted_A', 'positive_protection_new_errors',
    'NLL', 'Brier', 'ECE', 'selective_risk', 'selective_coverage',
})
_REQUIRED_KINDS = {'original', 'permutation', 'label_remap', 'paraphrase',
                   'frontier', 'unknown', 'counterfactual_true', *FAMILIES}


class EvaluationIntegrityError(ValueError):
    """The supplied evaluator inputs cannot be admitted for computation."""


def _require(condition, reason):
    if not condition:
        raise EvaluationIntegrityError(reason)


def _identifier(value):
    return isinstance(value, str) and bool(value.strip())


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _fixtures(fixtures):
    _require(isinstance(fixtures, (list, tuple)) and bool(fixtures), 'NONEMPTY_FIXTURES_REQUIRED')
    by_id = {}
    groups = {}
    for fixture in fixtures:
        _require(isinstance(fixture, Mapping), 'INVALID_FIXTURE')
        identity = fixture.get('fixture_id')
        _require(_identifier(identity) and identity not in by_id, 'UNIQUE_FIXTURE_IDS_REQUIRED')
        labels = fixture.get('labels')
        _require(isinstance(labels, Mapping) and bool(labels), 'INVALID_CANDIDATES')
        _require(all(_identifier(k) and isinstance(v, str) and v in {'TRUE', 'FALSE', 'UNKNOWN'}
                     for k, v in labels.items()), 'INVALID_CANDIDATES')
        _require(len(set(labels.values())) == len(labels), 'AMBIGUOUS_CANDIDATE_SEMANTICS')
        _require(isinstance(fixture.get('gold'), str) and fixture['gold'] in labels.values(), 'INVALID_GOLD')
        case, kind = fixture.get('case_id'), fixture.get('kind')
        _require(_identifier(case) and isinstance(kind, str) and kind in _REQUIRED_KINDS,
                 'INVALID_VARIANT_IDENTITY')
        group = groups.setdefault(case, set())
        _require(kind not in group, 'UNIQUE_CASE_VARIANTS_REQUIRED')
        group.add(kind)
        family = fixture.get('hard_negative_family')
        _require(family is None or _identifier(family), 'INVALID_HARD_NEGATIVE_FAMILY')
        by_id[identity] = fixture
    _require(all(kinds == _REQUIRED_KINDS for kinds in groups.values()), 'EXACT_VARIANT_COVERAGE_REQUIRED')
    return by_id


def _results(rows, fixtures, role):
    _require(isinstance(rows, (list, tuple)) and bool(rows), role + ':NONEMPTY_RESULTS_REQUIRED')
    seen = set()
    for row in rows:
        _require(isinstance(row, Mapping), role + ':INVALID_RESULT')
        identity = row.get('fixture_id')
        _require(_identifier(identity) and identity not in seen, role + ':UNIQUE_RESULT_IDS_REQUIRED')
        _require(identity in fixtures, role + ':EXACT_EVALUATION_COVERAGE_REQUIRED')
        seen.add(identity)
        _require(row.get('status') == 'COMPLETE', role + ':COMPLETE_STATUS_REQUIRED')
        probabilities = row.get('probabilities')
        _require(isinstance(probabilities, Mapping), role + ':INVALID_EVALUATION_DISTRIBUTION')
        _require(set(probabilities) == set(fixtures[identity]['labels']), role + ':INVALID_CANDIDATE_KEYS')
        selected = row.get('selected_label')
        _require(_identifier(selected) and selected in probabilities, role + ':INVALID_SELECTION')
        _require(all(_finite_number(p) and 0 <= p <= 1 for p in probabilities.values()),
                 role + ':INVALID_EVALUATION_DISTRIBUTION')
        _require(abs(sum(probabilities.values()) - 1) <= 1e-6, role + ':INVALID_EVALUATION_DISTRIBUTION')
    _require(seen == set(fixtures), role + ':EXACT_EVALUATION_COVERAGE_REQUIRED')


def _thresholds(thresholds):
    _require(isinstance(thresholds, Mapping), 'INVALID_THRESHOLDS')
    for gate, value in thresholds.items():
        _require(isinstance(gate, str) and gate.endswith(('_min', '_max'))
                 and gate.rsplit('_', 1)[0] in METRIC_NAMES, 'UNIMPLEMENTED_GATE:' + str(gate))
        _require(_finite_number(value), 'FINITE_NUMERIC_THRESHOLD_REQUIRED:' + gate)


def _identity(value):
    _require(value is None or _identifier(value), 'INVALID_DESCRIPTOR_IDENTITY')
    return value if value is not None else 'UNBOUND'


def _descriptors(fixtures, results, label_identity, frontier_identity):
    originals = [f for f in fixtures if f['kind'] == 'original']
    by_id = {r['fixture_id']: r for r in results}
    selected_ids = sorted(f['fixture_id'] for f in originals
                          if max(by_id[f['fixture_id']]['probabilities'].values()) >= .8)
    hard_negatives = [f for f in fixtures if f['gold'] == 'FALSE'
                      and (f['kind'] in FAMILIES or f['kind'] == 'original')]
    original_ids = sorted(f['fixture_id'] for f in originals)
    common = {
        'probability_target': 'candidate-semantic distribution',
        'probability_operator': 'identity temperature; supplied normalized candidate probabilities',
        'label_identity': label_identity, 'frontier_identity': frontier_identity,
        'denominator': {'population': 'original DEV source groups', 'fixture_ids': original_ids,
                        'count': len(originals), 'validity': 'all admitted COMPLETE rows; UNKNOWN retained'},
        'decision_policy': 'caller-supplied selected_label; evaluator performs no argmax or tie breaking',
        'selection_policy': 'confidence=max(candidate probabilities); correctness uses supplied selected_label',
    }
    definitions = {
        'original_accuracy': 'selected semantic label equals gold',
        'A_recall': 'TRUE true positives / gold TRUE originals; absent class is undefined',
        'B_recall': 'FALSE true positives / gold FALSE originals; absent class is undefined',
        'macro_F1': 'mean TRUE/FALSE F1; UNKNOWN counts as error; zero F1 denominator gives zero',
        'candidate_order_consistency': 'original versus permutation semantic equality',
        'label_remap_consistency': 'original versus label_remap semantic equality',
        'paraphrase_consistency': 'original versus paraphrase semantic equality',
        'fixed_label_collapse_rate': 'original versus label_remap candidate-key equality',
        'frontier_correctness': 'frontier selected semantic label equals gold',
        'unknown_accuracy': 'unknown variant selected semantic label equals gold',
        'counterfactual_pair_accuracy': 'both counterfactual_true and scoring_not_elimination correct',
        'hard_negative_gold_B_predicted_A': 'count gold FALSE hard negatives predicted TRUE',
        'hard_negative_accuracy': 'fraction gold FALSE hard negatives correct; empty population undefined',
        'gold_B_predicted_A': 'count gold FALSE originals predicted TRUE',
        'positive_protection_new_errors': 'count gold TRUE originals correct at baseline now incorrect; absent baseline uses current rows',
        'NLL': 'mean negative natural log gold-candidate probability',
        'Brier': 'summed multiclass squared error; unscaled',
        'ECE': 'top-label ten-bin expected calibration error',
        'selective_risk': 'one minus selected correctness fraction',
        'selective_coverage': 'selected originals / all originals',
    }
    descriptors = {name: dict(common, convention=convention) for name, convention in definitions.items()}
    for name, semantic in [('A_recall', 'TRUE'), ('B_recall', 'FALSE')]:
        ids = sorted(f['fixture_id'] for f in originals if f['gold'] == semantic)
        descriptors[name]['denominator'] = dict(common['denominator'], fixture_ids=ids, count=len(ids), population='gold ' + semantic + ' originals')
    for name in ('hard_negative_accuracy', 'hard_negative_gold_B_predicted_A'):
        descriptors[name]['denominator'] = dict(common['denominator'], population='gold FALSE original or hard-negative family rows',
            fixture_ids=sorted(f['fixture_id'] for f in hard_negatives), count=len(hard_negatives))
    for name in ('hard_negative_gold_B_predicted_A', 'gold_B_predicted_A', 'positive_protection_new_errors'):
        descriptors[name]['aggregation'] = 'count; no division'
    descriptors['NLL'].update(log_floor=1e-300, log_base='e')
    descriptors['Brier'].update(target='one-hot gold candidate over all candidates',
        binary_conversion='summed two-class Brier = 2 * one-event Brier only for the same binary event, polarity, probabilities and population')
    descriptors['ECE'].update(target='max candidate confidence versus supplied selected-label correctness',
        bin_count=10, bin_boundaries=[i / 10 for i in range(11)],
        bin_policy='min(9, int(confidence * 10)); left-inclusive/right-exclusive, final bin includes 1',
        event_ece='event probability versus event occurrence is a different estimand; equal scalar values do not admit comparison')
    for name in ('selective_risk', 'selective_coverage'):
        descriptors[name].update(threshold=.8, threshold_operator='>=', zero_coverage='risk=None; coverage=0')
    descriptors['selective_risk']['denominator'] = dict(common['denominator'],
        population='originals with max candidate probability >= .8', count=len(selected_ids), fixture_ids=selected_ids)
    return descriptors


def compare_metric_descriptors(left, right, metric):
    """Admit identical bound definitions only, with no implicit conversion.

A differently defined result requires an independently recorded mathematical
conversion and newly constructed matching descriptors before this check.
Scalar equality never establishes comparability or business calibration.
"""
    a, b = left.get(metric), right.get(metric)
    required = {'probability_target', 'probability_operator', 'label_identity',
                'frontier_identity', 'denominator', 'convention', 'decision_policy', 'selection_policy'}
    if not isinstance(a, Mapping) or not isinstance(b, Mapping) or not required <= a.keys() or not required <= b.keys():
        return 'NOT_COMPARABLE'
    if any(a.get(k) in (None, '', 'UNBOUND') or b.get(k) in (None, '', 'UNBOUND')
           for k in ('label_identity', 'frontier_identity')):
        return 'NOT_COMPARABLE'
    return 'COMPARABLE' if a == b else 'NOT_COMPARABLE'


def evaluate(fixtures, results, baseline, thresholds, *, label_identity=None, frontier_identity=None):
    """Validate before calling v1 once; return metadata beside its exact output."""
    by_id = _fixtures(fixtures)
    _results(results, by_id, 'RESULTS')
    if baseline is not None:
        _results(baseline, by_id, 'BASELINE')
    _thresholds(thresholds)
    labels, frontier = _identity(label_identity), _identity(frontier_identity)
    legacy = legacy_evaluate(fixtures, results, baseline, thresholds)
    return {
        'version': VERSION,
        'configuration': {'thresholds': dict(thresholds), 'label_identity': labels,
                          'frontier_identity': frontier, 'baseline': 'self' if baseline is None else 'supplied'},
        'evaluation': legacy,
        'disposition': legacy['disposition'] if thresholds else 'METRICS_ONLY',
        'metric_descriptors': _descriptors(fixtures, results, labels, frontier),
    }
