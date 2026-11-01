"""Replay native registration intent over a supplied base in a disposable process.

This module must never be invoked in the serving process: native replay mutates
LiteLLM's global cost map. Its result is an expectation, not observed state.
"""
import hashlib
import json
import sys
from copy import deepcopy
from contextlib import redirect_stdout
from pathlib import Path


def replay_expected_state(base, snapshot):
    import litellm
    from litellm import router as native_router, utils as native_utils
    from litellm.types import router as router_types, utils as cost_types
    from litellm.types.router import Deployment
    from litellm.litellm_core_utils.get_model_cost_map import adopt_model_cost_map

    modules = (('router', native_router), ('registration', native_utils),
               ('router_types', router_types), ('cost_types', cost_types))
    versions = {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                for name, module in modules}
    if (not isinstance(base, dict) or not base or not isinstance(snapshot, dict)
        or snapshot.get('native_rule_sha256') != versions
        or snapshot.get('replay_order') != ['router', 'durable']
        or not isinstance(snapshot.get('registrations'), list)
        or not isinstance(snapshot.get('durable_registrations'), list)):
        raise ValueError('incomplete or incompatible native registration snapshot')
    adopt_model_cost_map(deepcopy(base))
    for row in snapshot['registrations']:
        params = {'model': row['model'], **row['explicit_parameters']}
        if row.get('provider') is not None:
            params['custom_llm_provider'] = row['provider']
        if row.get('base_model') is not None:
            params['base_model'] = row['base_model']
        deployment = Deployment(model_name=row['alias'], litellm_params=params,
                                model_info=row['metadata'])
        native_router.Router._register_deployment_pricing(deployment)
    for row in snapshot['durable_registrations']:
        litellm.register_model({row['key']: row['metadata']}, persist_across_reloads=False)
    return litellm.model_cost


def main():
    value = json.load(sys.stdin)
    # LiteLLM may emit provider diagnostics during registration; keep stdout
    # reserved for the bounded JSON result and do not forward diagnostic bodies.
    with open('/dev/null', 'w') as sink, redirect_stdout(sink):
        result = replay_expected_state(value['base'], value['snapshot'])
    json.dump(result, sys.stdout, allow_nan=False, separators=(',', ':'))


if __name__ == '__main__':
    main()
