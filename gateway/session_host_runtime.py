"""Host-authorized transient runtime for one exact native turn."""
from contextlib import contextmanager
from contextvars import ContextVar

_current = ContextVar('host_session_runtime', default=None)


@contextmanager
def bind_host_runtime(runner, source, session_key, model, runtime, *, boundary=None):
    token = _current.set((runner, source, session_key, model, dict(runtime), boundary))
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
