"""Compare selected source, registered overlays and actual native worker map."""
from copy import deepcopy

from billing.supplier_pricing_projection import digest

PREDICATE_VERSION = 'supplier-native-registration.v9'


def reconcile_native_map(base, registrations, effective, owned_overrides=None, *, expected_base_sha256=None, required_registration_ids=None):
    if not isinstance(base, dict) or not isinstance(expected_base_sha256, str) or digest(base) != expected_base_sha256:
        return {'predicate': PREDICATE_VERSION, 'verified': False, 'reason': 'BASE_IDENTITY_UNVERIFIED'}
    if (not isinstance(registrations, dict) or not isinstance(effective, dict)
        or (owned_overrides is not None and not isinstance(owned_overrides, dict))):
        return {'predicate': PREDICATE_VERSION, 'verified': False, 'reason': 'INVALID_OBSERVATION'}
    if (not isinstance(required_registration_ids, (set, frozenset, list, tuple))
        or not all(isinstance(item, str) and item for item in required_registration_ids)):
        return {'predicate': PREDICATE_VERSION, 'verified': False, 'reason': 'REGISTRATION_INVENTORY_UNKNOWN'}
    if set(required_registration_ids).intersection(base):
        return {'predicate': PREDICATE_VERSION, 'verified': False, 'reason': 'REGISTRATION_KEY_COLLISION'}
    if not set(required_registration_ids).issubset(registrations) or not set(required_registration_ids).issubset(effective):
        return {'predicate': PREDICATE_VERSION, 'verified': False, 'reason': 'REQUIRED_REGISTRATION_MISSING'}
    overrides = owned_overrides or {}
    expected = deepcopy(base)
    for key, fields in registrations.items():
        if not isinstance(key, str) or not isinstance(fields, dict):
            return {'predicate': PREDICATE_VERSION, 'verified': False, 'reason': 'INVALID_REGISTRATION'}
        existing = expected.setdefault(key, {})
        if not isinstance(existing, dict):
            return {'predicate': PREDICATE_VERSION, 'verified': False, 'reason': 'INVALID_BASE_ENTRY'}
        permitted = overrides.get(key, {})
        if not isinstance(permitted, dict):
            return {'predicate': PREDICATE_VERSION, 'verified': False, 'reason': 'UNBOUND_OVERRIDE'}
        for field, value in fields.items():
            if not isinstance(field, str):
                return {'predicate': PREDICATE_VERSION, 'verified': False, 'reason': 'INVALID_REGISTRATION'}
            price_field = 'cost' in field.lower() or 'pric' in field.lower()
            if ((field in existing and existing[field] != value)
                or (field not in existing and value is not None and price_field)):
                claim = permitted.get(field)
                if (not isinstance(claim, dict) or claim.get('value') != value
                    or not claim.get('owner') or not claim.get('revision')):
                    return {'predicate': PREDICATE_VERSION, 'verified': False, 'reason': 'UNOWNED_OVERRIDE'}
            existing[field] = value
    for key, fields in overrides.items():
        if key not in registrations or not isinstance(fields, dict) or any(
            not isinstance(claim, dict) or not claim.get('owner') or not claim.get('revision')
            or registrations[key].get(field, object()) != claim.get('value', object())
            for field, claim in fields.items()
        ):
            return {'predicate': PREDICATE_VERSION, 'verified': False, 'reason': 'UNBOUND_OVERRIDE'}
    return {
        'predicate': PREDICATE_VERSION,
        'verified': expected == effective,
        'reason': None if expected == effective else 'UNEXPLAINED_EFFECTIVE_MAP',
        'base_sha256': digest(base),
        'registration_sha256': digest(registrations),
        'effective_sha256': digest(effective),
    }


def reconcile_native_workers(base, registrations, observations, expected_workers, owned_overrides=None, *, expected_base_sha256=None, required_registration_ids=None, expected_epochs=None):
    if (not isinstance(observations, dict) or not isinstance(expected_workers, (set, frozenset, list, tuple))
        or not expected_workers or not all(isinstance(worker, str) and worker for worker in expected_workers)
        or set(observations) != set(expected_workers)):
        return {'predicate': PREDICATE_VERSION, 'verified': False, 'reason': 'WORKER_COVERAGE_UNKNOWN'}
    if (not isinstance(expected_epochs, dict) or set(expected_epochs) != set(expected_workers)
        or not all(isinstance(epoch, str) and epoch for epoch in expected_epochs.values())):
        return {'predicate': PREDICATE_VERSION, 'verified': False, 'reason': 'WORKER_EPOCH_UNKNOWN'}
    results = {}
    for worker, observation in observations.items():
        if (not isinstance(observation, dict) or observation.get('current') is not True
            or observation.get('epoch') != expected_epochs[worker] or not isinstance(observation.get('map'), dict)):
            return {'predicate': PREDICATE_VERSION, 'verified': False, 'reason': 'WORKER_OBSERVATION_STALE'}
        results[worker] = reconcile_native_map(base, registrations, observation['map'], owned_overrides, expected_base_sha256=expected_base_sha256, required_registration_ids=required_registration_ids)
    return {'predicate': PREDICATE_VERSION, 'verified': all(row['verified'] for row in results.values()), 'workers': results}


def reconcile_native_capabilities(expected_models, capabilities, effective):
    if (not isinstance(capabilities, dict) or not isinstance(effective, dict)
        or not isinstance(expected_models, (set, frozenset, list, tuple))
        or not expected_models or not all(isinstance(model, str) and model for model in expected_models)
        or set(capabilities) != set(expected_models)):
        return {'predicate': PREDICATE_VERSION, 'verified': False, 'reason': 'CAPABILITY_COVERAGE_UNKNOWN'}
    for model in expected_models:
        expected = capabilities[model]
        actual = effective.get(model)
        if (not isinstance(expected, list) or not all(isinstance(item, str) for item in expected)
            or not isinstance(actual, dict) or actual.get('supported_openai_params') != expected):
            return {'predicate': PREDICATE_VERSION, 'verified': False, 'reason': 'CAPABILITY_MISMATCH'}
    return {'predicate': PREDICATE_VERSION, 'verified': True}
