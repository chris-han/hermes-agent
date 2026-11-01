"""Small proxy-admin-only process-lifetime observation for native management."""
from pathlib import Path
import os
import secrets
import hashlib
import json
import asyncio
import sys

from fastapi import HTTPException, Request
from starlette.responses import JSONResponse


def owns_native_listener(port=4000):
    """Fail closed unless this process owns the only TCP listener on the port."""
    try:
        listeners = set()
        for path in ('/proc/net/tcp', '/proc/net/tcp6'):
            for line in Path(path).read_text().splitlines()[1:]:
                fields = line.split()
                if fields[1].endswith(f':{port:04X}') and fields[3] == '0A':
                    listeners.add(fields[9])
        if len(listeners) != 1:
            return False
        owned = set()
        for descriptor in Path('/proc/self/fd').iterdir():
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, PermissionError):
                continue
            if target.startswith('socket:[') and target.endswith(']'):
                owned.add(target[8:-1])
        return listeners <= owned
    except (OSError, IndexError, ValueError):
        return False


def check_no_credentials(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or any(word in key.lower() for word in ('secret', 'password', 'credential', 'api_key', 'token_key')):
                raise ValueError('unsafe registration metadata')
            check_no_credentials(item)
    elif isinstance(value, list):
        for item in value:
            check_no_credentials(item)


def safe_router_metadata(info, model_fields):
    """Keep native routing and health extensions with known consumers."""
    if not isinstance(info, dict) or set(info) - set(model_fields) - {'mode', 'disable_background_health_check'}:
        raise ValueError('unsupported registration metadata')
    if 'mode' in info and (not isinstance(info['mode'], str) or not info['mode']):
        raise ValueError('invalid native routing mode')
    if 'disable_background_health_check' in info and not isinstance(info['disable_background_health_check'], bool):
        raise ValueError('invalid native health flag')
    check_no_credentials(info)
    return info


def safe_durable_registrations(registry, allowed_fields):
    """Copy the replay owner's ordered, schema-recognized non-secret records."""
    if not isinstance(registry, dict):
        raise ValueError('durable registration inventory unavailable')
    result = []
    for key, metadata in tuple(registry.items()):
        if not isinstance(key, str) or not key or not isinstance(metadata, dict) or set(metadata) - allowed_fields:
            raise ValueError('unsupported durable registration')
        check_no_credentials(metadata)
        # Freeze nested values and reject non-JSON objects and non-finite prices.
        copy = json.loads(json.dumps(metadata, allow_nan=False))
        result.append({'key': key, 'metadata': copy})
    return result


def native_registrations():
    """Read recognized raw Router configuration without provider credentials."""
    from litellm.proxy import proxy_server
    from litellm.types.router import CustomPricingLiteLLMParams, Deployment, ModelInfo
    from litellm.types.utils import ModelInfo as CostMapModelInfo
    from litellm import router as native_router, utils as native_utils
    from litellm.types import router as router_types, utils as cost_types

    router = proxy_server.llm_router
    if router is None:
        raise ValueError('router unavailable')
    registrations = []
    for entry in tuple(router.model_list):
        deployment = entry if isinstance(entry, Deployment) else Deployment(**entry)
        info = deployment.model_info.model_dump(mode='json', exclude_none=True)
        safe_router_metadata(info, ModelInfo.model_fields)
        params = deployment.litellm_params.model_dump(mode='json', exclude_none=True)
        pricing = {field: params[field] for field in CustomPricingLiteLLMParams.model_fields if field in params}
        check_no_credentials(pricing)
        model, alias = params.get('model'), deployment.model_name
        if not all(isinstance(value, str) and value for value in (info.get('id'), model, alias)):
            raise ValueError('invalid registration identity')
        registrations.append({'id': info['id'], 'alias': alias, 'model': model,
                              'provider': params.get('custom_llm_provider'), 'base_model': params.get('base_model'),
                              'metadata': info, 'explicit_parameters': pricing})
    if len({row['id'] for row in registrations}) != len(registrations):
        raise ValueError('duplicate registration identity')
    durable = safe_durable_registrations(native_utils._runtime_registered_model_cost,
                                        set(CostMapModelInfo.__annotations__))
    versions = {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                for name, module in (('router', native_router), ('registration', native_utils),
                                     ('router_types', router_types), ('cost_types', cost_types))}
    return {'registrations': registrations, 'durable_registrations': durable,
            'replay_order': ['router', 'durable'], 'native_rule_sha256': versions}


async def isolated_expected_state(base, snapshot):
    """Run native replay in a separate process without server credentials."""
    import litellm
    script = Path(__file__).with_name('native_expected_state.py')
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(script), stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        env={'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
             'PYTHONPATH': str(Path(litellm.__file__).resolve().parent.parent)})
    try:
        payload = json.dumps({'base': base, 'snapshot': snapshot}, allow_nan=False).encode()
        output, _ = await asyncio.wait_for(process.communicate(payload), timeout=45)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    if process.returncode or len(output) > 32_000_000:
        raise ValueError('native replay unavailable')
    value = json.loads(output)
    if not isinstance(value, dict):
        raise ValueError('invalid native replay')
    return value


class NativeRuntimeIdentityMiddleware:
    def __init__(self, app, authorize, *, listener_check=owns_native_listener, registration_reader=native_registrations, expected_runner=isolated_expected_state):
        self.app, self.authorize, self.listener_check = app, authorize, listener_check
        self.registration_reader = registration_reader
        self.expected_runner = expected_runner
        self.epoch = secrets.token_hex(32)

    async def __call__(self, scope, receive, send):
        path, method = scope.get('path'), scope.get('method')
        if scope['type'] != 'http' or (path, method) not in {('/semantier/runtime/identity', 'GET'), ('/semantier/runtime/registrations', 'GET'), ('/semantier/runtime/expected', 'POST')}:
            return await self.app(scope, receive, send)
        try:
            request = Request(scope, receive=receive)
            actor = await self.authorize(request)
            role = getattr(actor.user_role, 'value', actor.user_role)
            if role != 'proxy_admin' or not actor.user_id:
                raise HTTPException(403, 'proxy administrator required')
            if not self.listener_check():
                raise HTTPException(503, 'listener ownership unavailable')
            payload = {'epoch': self.epoch, 'listener_owned': True}
            if path == '/semantier/runtime/registrations':
                payload.update(self.registration_reader())
            elif path == '/semantier/runtime/expected':
                raw = await request.body()
                if len(raw) > 8_000_000:
                    raise HTTPException(413, 'base too large')
                body = json.loads(raw)
                if not isinstance(body, dict) or set(body) != {'base'} or not isinstance(body['base'], dict) or not body['base']:
                    raise HTTPException(422, 'base required')
                snapshot = self.registration_reader()
                expected = await self.expected_runner(body['base'], snapshot)
                if not self.listener_check() or snapshot != self.registration_reader():
                    raise ValueError('registration state changed')
                payload.update({'snapshot': snapshot, 'expected_map': expected})
            response = JSONResponse(payload, headers={'Cache-Control': 'no-store'})
        except HTTPException as error:
            response = JSONResponse({'detail': 'runtime identity unavailable'}, status_code=error.status_code)
        except Exception as error:
            # Native authentication raises ProxyException rather than HTTPException.
            # Return only its authorization status, never its possibly sensitive body.
            try:
                from litellm.proxy._types import ProxyException
                status = int(error.code) if isinstance(error, ProxyException) and str(error.code) in {'401', '403'} else 503
            except (ImportError, TypeError, ValueError):
                status = 503
            response = JSONResponse({'detail': 'runtime identity unavailable'}, status_code=status)
        return await response(scope, receive, send)
