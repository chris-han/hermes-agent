"""Host-owned cron admission; standalone Hermes has no Semantier dependency."""
import os


def require_cron_profile():
    backend = os.environ.get('SEMANTIER_STORAGE_BACKEND', 'sqlite').lower()
    if backend == 'sqlite':
        from hermes_state_backend import postgres_enabled, resolve_postgres_scope
        if postgres_enabled():
            from hermes_constants import get_hermes_home
            scope = resolve_postgres_scope(get_hermes_home())
            if scope.source_store_id.startswith('winman-native-'):
                from agents.winman_delegation import require_qualified_winman_sessiondb
                require_qualified_winman_sessiondb()
        return
    if backend != 'postgresql':
        raise RuntimeError('unsupported explicit storage backend')
    from hermes_constants import get_hermes_home
    from agents.winman_delegation import require_hosted_cron_profile
    require_hosted_cron_profile(get_hermes_home())
