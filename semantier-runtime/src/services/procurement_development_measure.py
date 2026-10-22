"""Pure hypothetical DEVELOPMENT calculation; no source or policy authority."""
from copy import deepcopy
import hashlib
import json

PROFILE_ID = 'procurement.dev.measure.v1'
PROVENANCE = ['DEVELOPMENT', 'ASSISTANT_AUTHORED', 'NOT_QUALIFIED']
MAX_FEN = 2**63 - 1
MISSING = object()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


class Refusal(ValueError):
    def __init__(self, code, location):
        self.code, self.location = code, location
        super().__init__(code + ':' + location)


def require(condition, code, location):
    if not condition:
        raise Refusal(code, location)


def money(raw, location):
    if raw is MISSING:
        return None, 'MISSING'
    if raw is None:
        return None, 'NULL'
    if raw == 'UNKNOWN':
        return None, 'UNKNOWN'
    require(type(raw) is int, 'MONEY_FORMAT', location)
    require(0 <= raw <= MAX_FEN, 'MONEY_RANGE', location)
    return raw, 'SUPPLIED'


def evaluate(profile, case):
    """No expected outcomes, legal prose, source parsing or production owners enter here."""
    result = dict(status='REFUSED', exposure_fen=None, findings=[], ledger=[],
                  classification='ASSISTANT_SELECTED_DEVELOPMENT_ASSUMPTIONS',
                  provenance=PROVENANCE, owner_decisions={f'Q{i}': 'PENDING' for i in range(1, 7)},
                  domain_review='NOT_PERFORMED', independent_reviews=0, cash_due='NOT_EVALUATED',
                  profile_id=PROFILE_ID, profile_hash=digest(profile), input_hash=digest(case),
                  assumption_ids=['Q1A', 'Q2A', 'Q3A', 'Q4A', 'Q5A', 'Q6A'],
                  conditional_basis='HYPOTHETICAL_PROSPECTIVE_AS_IF_EFFECTIVE_ONLY')
    try:
        require(isinstance(profile, dict) and profile.get('profile_id') == PROFILE_ID,
                'UNSUPPORTED_PROFILE', 'profile')
        require(profile.get('selection') == result['classification']
                and profile.get('provenance') == PROVENANCE
                and profile.get('owner_decisions') == result['owner_decisions']
                and profile.get('domain_review') == 'NOT_PERFORMED'
                and type(profile.get('independent_reviews')) is int
                and profile['independent_reviews'] == 0,
                'FALSE_REVIEW_OR_AUTHORITY', 'profile')
        require(profile.get('purpose') == 'procurement_approval_irreversible_exposure'
                and profile.get('cohort') == 'prospective_as_if_effective'
                and profile.get('assumption_ids') == result['assumption_ids'],
                'UNSUPPORTED_PROFILE', 'profile.scope')
        require(profile.get('max_fen') == MAX_FEN and type(profile['max_fen']) is int,
                'UNSUPPORTED_PROFILE', 'profile.max_fen')
        require(isinstance(case, dict), 'INPUT_SHAPE', 'input')
        require(not ({'owner_adoption', 'authority', 'independent_reviews', 'domain_review'} & set(case))
                and case.get('judgment_assertion') == 'ASSUMED_FOR_TEST',
                'FALSE_REVIEW_OR_AUTHORITY', 'judgment_assertion')
        require(case.get('profile_id') == PROFILE_ID, 'UNSUPPORTED_PROFILE', 'profile_id')
        require(case.get('purpose') == profile['purpose'], 'UNSUPPORTED_PURPOSE', 'purpose')
        require(case.get('cohort') == profile['cohort'], 'UNSUPPORTED_COHORT', 'cohort')
        require(case.get('input_kind') == 'HYPOTHETICAL_DEVELOPMENT', 'UNSUPPORTED_COHORT', 'input_kind')
        require(set(case) <= {'case_id', 'profile_id', 'input_kind', 'purpose', 'cohort',
                'proposed_version_ref', 'knowledge_cutoff_ref', 'judgment_assertion', 'currency',
                'amount_basis', 'quoted_price_fen', 'roster_id', 'premises', 'obligations', 'observations'},
                'INPUT_SHAPE', 'input.keys')
        require(isinstance(case.get('case_id'), str) and case['case_id'], 'INPUT_SHAPE', 'case_id')
        result['case_id'] = case['case_id']
        for key, expected in [('proposed_version_ref', 'fixture://proposed-version/dev-v1'),
                              ('knowledge_cutoff_ref', 'fixture://cutoff/dev-v1')]:
            require(case.get(key) == expected and expected in profile['fixture_refs'],
                    'FIXTURE_REFERENCE', key)
        require(case.get('currency') == 'CNY' and case.get('amount_basis') == 'gross',
                'MONEY_BASIS', 'currency/amount_basis')
        quoted, quoted_state = money(case.get('quoted_price_fen', MISSING), 'quoted_price_fen')
        result['quoted_price_fen'] = quoted
        result['quoted_price_state'] = quoted_state
        raw_premises = case.get('premises', {})
        require(isinstance(raw_premises, dict), 'INPUT_SHAPE', 'premises')
        require(set(raw_premises) <= set(profile['required_premises']) | {'constitutive_condition'},
                'INPUT_SHAPE', 'premises.keys')
        result['premise_observations'] = deepcopy(raw_premises)
        result['observations'] = deepcopy(case.get('observations', {}))
        known = {}

        def premise(key, allowed):
            raw = raw_premises.get(key, MISSING)
            state = 'MISSING' if raw is MISSING else 'NULL' if raw is None else None
            if state is None:
                require(isinstance(raw, dict) and set(raw) == {'state', 'value', 'fixture_ref', 'assertion'},
                        'PREMISE_SHAPE' if isinstance(raw, dict) and 'fixture_ref' in raw else 'FIXTURE_REFERENCE',
                        'premises.' + key)
                require(raw['fixture_ref'] == 'fixture://judgment/dev-v1'
                        and raw['fixture_ref'] in profile['fixture_refs'], 'FIXTURE_REFERENCE', 'premises.' + key)
                require(raw['assertion'] == 'ASSUMED_FOR_TEST', 'FALSE_REVIEW_OR_AUTHORITY', 'premises.' + key)
                state = raw['state']
                require(state in ('SUPPLIED', 'UNKNOWN', 'CONFLICT', 'NULL', 'MISSING'),
                        'PREMISE_SHAPE', 'premises.' + key)
                if state == 'SUPPLIED':
                    require(type(raw['value']) is str and raw['value'] in allowed,
                            'PREMISE_VALUE', 'premises.' + key)
                    known[key] = raw['value']
                    return
                if state == 'CONFLICT':
                    require(isinstance(raw['value'], list) and len(raw['value']) >= 2,
                            'PREMISE_SHAPE', 'premises.' + key)
            result['findings'].append(dict(location='premises.' + key, state=state,
                                           reason='MATERIAL_PREMISE_UNRESOLVED'))

        for key, allowed in profile['required_premises'].items():
            premise(key, allowed)
        if known.get('condition_materiality') == 'CONSTITUTIVE':
            premise('constitutive_condition', ['SATISFIED'])
        # The independently frozen roster survives candidate deletion.
        roster_id = case.get('roster_id')
        require(roster_id in profile['rosters'], 'ROSTER_REFERENCE', 'roster_id')
        roster = {r['economic_id']: r for r in profile['rosters'][roster_id]}
        obligations = case.get('obligations')
        require(isinstance(obligations, list), 'INPUT_SHAPE', 'obligations')
        require(all(isinstance(o, dict) for o in obligations), 'INPUT_SHAPE', 'obligations')
        ids = [o.get('economic_id') for o in obligations]
        require(all(type(x) is str for x in ids), 'INPUT_SHAPE', 'economic_id')
        require(len(ids) == len(set(ids)), 'DUPLICATE_ECONOMIC_ITEM', 'obligations')
        require(set(ids) <= set(roster), 'ROSTER_REFERENCE', 'obligations')
        for missing in sorted(set(roster) - set(ids)):
            result['findings'].append(dict(location='obligations.' + missing, state='MISSING',
                                           reason='REQUIRED_OBLIGATION_MISSING'))
        for obligation in obligations:
            eid = obligation['economic_id']
            location = 'obligations.' + eid
            require(set(obligation) == {'economic_id', 'kind', 'partition', 'parts'}, 'INPUT_SHAPE', location)
            declared = roster[eid]
            require(obligation['kind'] == declared['kind'], 'ROSTER_REFERENCE', location + '.kind')
            partition = obligation['partition']
            require(partition in declared['partitions'], 'PARTITION_MEMBERSHIP', location)
            expected = set(declared['partitions'][partition])
            parts = obligation['parts']
            require(isinstance(parts, list) and all(isinstance(p, dict) for p in parts), 'INPUT_SHAPE', location)
            part_ids = [p.get('part_id') for p in parts]
            require(all(type(x) is str for x in part_ids), 'INPUT_SHAPE', location + '.parts')
            require(len(part_ids) == len(set(part_ids)), 'DUPLICATE_PART', location)
            require(set(part_ids) <= expected, 'PARTITION_MEMBERSHIP', location)
            for missing in sorted(expected - set(part_ids)):
                result['findings'].append(dict(location=location + '.' + missing, state='MISSING',
                                               reason='REQUIRED_PART_MISSING'))
            for part in parts:
                loc = location + '.' + part['part_id']
                require(set(part) <= {'part_id', 'amount_fen', 'fixture_ref'}, 'INPUT_SHAPE', loc)
                require(part.get('fixture_ref') == 'fixture://money/dev-v1'
                        and part['fixture_ref'] in profile['fixture_refs'], 'FIXTURE_REFERENCE', loc)
                amount, state = money(part.get('amount_fen', MISSING), loc + '.amount_fen')
                result['ledger'].append(dict(economic_id=eid, part_id=part['part_id'], amount_fen=amount,
                                             state=state, fixture_ref=part['fixture_ref']))
                if state != 'SUPPLIED':
                    result['findings'].append(dict(location=loc + '.amount_fen', state=state,
                                                   reason='REQUIRED_AMOUNT_UNRESOLVED'))
        right, liability = known.get('exit_right'), known.get('exit_liability')
        if right and liability:
            expected_liability = {'ORDINARY_EXIT': 'ENUMERATED_RESIDUAL',
                                  'NO_ORDINARY_EXIT': 'WHOLE_BASE_OBLIGATION',
                                  'FREE_ORDINARY_EXIT': 'COMPLETE_ZERO_RESIDUAL'}[right]
            require(liability == expected_liability, 'PREMISE_COMBINATION', 'exit_liability')
            if right == 'ORDINARY_EXIT':
                require(all(r['kind'] in ('residual', 'notice_consideration', 'exit_charge')
                            for r in roster.values()), 'RESIDUAL_BASIS', 'roster_id')
            if right == 'NO_ORDINARY_EXIT':
                require(all(r['kind'] == 'whole_base' for r in roster.values()), 'WHOLE_BASE_BASIS', 'roster_id')
            if right == 'FREE_ORDINARY_EXIT':
                require(all(r['kind'] == 'zero_residual' for r in roster.values()), 'ZERO_BASIS', 'roster_id')
                require(all(x['amount_fen'] in (None, 0) for x in result['ledger']), 'ZERO_BASIS', 'obligations')
        result['ledger'].sort(key=lambda x: (x['economic_id'], x['part_id']))
        result['findings'].sort(key=lambda x: x['location'])
        if result['findings']:
            result['status'] = 'UNRESOLVED'
        else:
            total = sum(x['amount_fen'] for x in result['ledger'])
            require(total <= MAX_FEN, 'MONEY_RANGE', 'exposure_fen')
            if total == 0:
                require(right == 'FREE_ORDINARY_EXIT' and liability == 'COMPLETE_ZERO_RESIDUAL'
                        and all(r['kind'] == 'zero_residual' for r in roster.values()),
                        'ZERO_BASIS', 'exposure_fen')
            result.update(status='HYPOTHETICAL_EXPOSURE', exposure_fen=total)
    except Refusal as error:
        result.update(status='REFUSED', exposure_fen=None, refusal_code=error.code,
                      refusal_location=error.location)
    return result

