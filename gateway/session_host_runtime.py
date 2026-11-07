"""Host-authorized transient runtime for one exact native turn."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

_current = ContextVar('host_session_runtime', default=None)


@contextmanager
def bind_host_runtime(runner, source, session_key, model, runtime, *, boundary=None, cloud_bound=False, cloud_revision="", task_context=None):
    token = _current.set((runner, source, session_key, model, dict(runtime), boundary, cloud_bound is True, cloud_revision, dict(task_context or {})))
    try:
        yield
    finally:
        _current.reset(token)


def host_runtime_for(runner, source, session_key):
    value = _current.get()
    if value is None or value[0] is not runner or value[1] is not source or value[2] != session_key:
        return None
    return value[3], dict(value[4])


def host_boundary_for(runner):
    value = _current.get()
    return value[5] if value is not None and value[0] is runner else None


def host_cloud_route(base_url, api_key):
    value = _current.get()
    if value is None or not value[6]:
        return False
    runtime = value[4]
    return (str(base_url or '').rstrip('/') == str(runtime.get('base_url') or '').rstrip('/')
            and bool(api_key) and api_key == runtime.get('api_key'))


def host_cloud_metadata(base_url, api_key):
    if not host_cloud_route(base_url, api_key):
        return None
    value = _current.get()
    return {'revision': value[7], 'task_context': dict(value[8])}


def host_inventory_scope(runner):
    factory = getattr(runner, '_host_inventory_scope', None)
    return factory() if callable(factory) else nullcontext()
