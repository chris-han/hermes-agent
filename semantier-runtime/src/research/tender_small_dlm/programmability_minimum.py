"""Research-only descendants and pretraining gate, independent of B2."""
from copy import deepcopy
import math
from src.research.tender_small_dlm.context_envelope import content_hash

KINDS = ('original', 'permutation', 'label_remap', 'paraphrase', 'frontier', 'unknown')


def _bind_research_identity(payload):
    contract = payload['decision_contract']
    contract['schema_version'] = 'tender_research_decision_contract.v1'
    contract.pop('contract_hash', None); contract.pop('contract_ref', None)
    h = content_hash(contract)
    contract.update(contract_hash=h, contract_ref='research-contract:' + h[7:])
    dlp = payload['dlp_model_input']
    dlp.update(decision_contract_ref=contract['contract_ref'], decision_contract_hash=h)
    dlp_hash = content_hash(dlp)
    frontier = payload['typed_decision_frontier']
    frontier.update(schema_version='tender_research_frontier.v1',
        decision_contract_ref=contract['contract_ref'], decision_contract_hash=h,
        dlp_hash=dlp_hash, dlp_ref='research-dlp:' + dlp_hash[7:], dlp_model_input_hash=dlp_hash)
    frontier.pop('frontier_ref', None); frontier.pop('frontier_hash', None)
    frontier.pop('frontier_policy_ref', None); frontier.pop('frontier_policy_hash', None)
    fh = content_hash(frontier)
    frontier.update(frontier_hash=fh, frontier_ref='research-frontier:' + fh[7:])
    payload.update(schema_version='tender_research_decision_input.v1', dlp_hash=dlp_hash,
        dlp_ref=frontier['dlp_ref'], dlp_model_input_hash=dlp_hash)
    payload.pop('input_hash', None); payload.pop('input_ref', None)
    ih = content_hash(payload)
    payload.update(input_hash=ih, input_ref='research-input:' + ih[7:])


def variants(case_id, group, parent, gold, assets):
    rows = []
    for kind in KINDS:
        payload = deepcopy(parent)
        labels = dict(zip('ABC', ('TRUE', 'FALSE', 'UNKNOWN')))
        if kind == 'permutation':
            labels = dict(reversed(list(labels.items())))
        if kind == 'label_remap':
            labels = dict(zip('ABC', ('FALSE', 'UNKNOWN', 'TRUE')))
        if kind == 'frontier':
            labels = {k:v for k,v in labels.items() if v in {gold, 'UNKNOWN'}}
        candidates = [{'candidate_id': k, 'semantic_value': v} for k,v in labels.items()]
        for predicates in [payload['dlp_model_input']['semantic_predicates'],
                           payload['typed_decision_frontier']['questions'],
                           payload['decision_contract']['semantic_predicates']]:
            for predicate in predicates:
                predicate['candidates'] = deepcopy(candidates)
        payload['runtime_semantic_definitions'] = assets['paraphrases' if kind == 'paraphrase' else 'definitions']
        if kind == 'unknown':
            for key in ('source_excerpts', 'assertions', 'material_dependency_resolutions'):
                payload['dlp_model_input'][key] = []
            payload['research_evidence_availability'] = 'UNAVAILABLE'
        _bind_research_identity(payload)
        rows.append({'fixture_id': case_id + ':' + kind, 'case_id': case_id,
            'source_group_id': group, 'kind': kind, 'labels': labels,
            'candidate_order': list(labels),
            'gold': 'UNKNOWN' if kind == 'unknown' else gold, 'synthetic': kind == 'unknown',
            'semantic_delta': 'required source evidence and source-bearing assertions removed' if kind == 'unknown' else kind,
            'parent_payload_hash': content_hash(parent), 'payload': payload,
            'authority': 'DERIVED_RESEARCH_ONLY_NOT_SOURCE_EVIDENCE'})
    return rows


def summarize(fixtures, results, thresholds):
    by = {r['fixture_id']:r for r in results}
    if len(by) != len(results) or set(by) != {r['fixture_id'] for r in fixtures}:
        raise ValueError('EXACT_FIXTURE_COVERAGE_REQUIRED')
    groups = {}
    valid = 0
    for fixture in fixtures:
        row = by[fixture['fixture_id']]
        probabilities = row.get('probabilities', {})
        okay = (row['status'] == 'COMPLETE' and set(probabilities) == set(fixture['labels'])
            and all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
            and abs(sum(probabilities.values()) - 1) < 1e-6
            and row.get('selected_label') in fixture['labels'])
        valid += okay
        groups.setdefault(fixture['case_id'], {})[fixture['kind']] = (fixture, row, okay)
    n = len(groups)
    if not n or any(set(g) != set(KINDS) for g in groups.values()):
        raise ValueError('EXACT_VARIANT_COVERAGE_REQUIRED')
    collapse = direction = counterfactual = unknown = 0
    consistency = {k:0 for k in ('permutation', 'label_remap', 'paraphrase')}
    original_correct = frontier_correct = 0
    for g in groups.values():
        f, r, ok = g['original']
        pred = f['labels'].get(r.get('selected_label')) if ok else None
        original_correct += pred == f['gold']
        for kind in consistency:
            ff, rr, oo = g[kind]
            consistency[kind] += ok and oo and pred == ff['labels'].get(rr.get('selected_label'))
        ff, rr, oo = g['label_remap']
        collapse += bool(ok and oo and r['selected_label'] == rr['selected_label'])
        moved = next(k for k,v in ff['labels'].items() if v == f['gold'])
        direction += bool(ok and oo and rr['probabilities'][moved] - r['probabilities'][moved] > 1e-6)
        ff, rr, oo = g['unknown']
        unknown_label = next(k for k,v in ff['labels'].items() if v == 'UNKNOWN')
        unknown += bool(oo and rr['selected_label'] == unknown_label)
        counterfactual += bool(ok and oo and rr['probabilities'][unknown_label] - r['probabilities'][unknown_label] > 1e-6)
        ff, rr, oo = g['frontier']
        frontier_correct += bool(oo and ff['labels'][rr['selected_label']] == ff['gold'])
    metrics = {'candidate_score_validity':valid/len(fixtures), 'frontier_validity':valid/len(fixtures),
        'fixed_surface_label_collapse_rate':collapse/n, 'candidate_semantic_directionality':direction/n,
        'counterfactual_directionality':counterfactual/n, 'unknown_selected_rate':unknown/n,
        'original_accuracy':original_correct/n, 'frontier_accuracy':frontier_correct/n,
        **{k+'_consistency':v/n for k,v in consistency.items()}}
    rules = [('candidate_score_validity','candidate_score_validity','min','INVALID_CANDIDATE_SCORES'),
        ('frontier_validity','frontier_validity','min','INVALID_FRONTIER'),
        ('fixed_surface_label_collapse_rate','fixed_surface_label_collapse_rate_max','max','FIXED_LABEL_COLLAPSE'),
        ('candidate_semantic_directionality','candidate_semantic_directionality_min','min','NO_SEMANTIC_DIRECTION'),
        ('counterfactual_directionality','counterfactual_directionality_min','min','NO_COUNTERFACTUAL_DIRECTION'),
        ('unknown_selected_rate','unknown_selected_rate_min','min','UNKNOWN_NOT_SELECTABLE')]
    failures = [label for metric,threshold,mode,label in rules if
        (metrics[metric] < thresholds[threshold] if mode == 'min' else metrics[metric] > thresholds[threshold])]
    gaps = [k for k in ('original_accuracy','frontier_accuracy','unknown_selected_rate',
                       'permutation_consistency','label_remap_consistency','paraphrase_consistency') if metrics[k] < .9]
    return {'metrics':metrics,'hard_failures':failures,'trainable_gaps':gaps,
            'disposition':'FAIL' if failures else 'PASS'}
