"""Host-owned cron admission; standalone Hermes has no Semantier dependency."""
import os


def require_cron_profile():
    backend = os.environ.get('SEMANTIER_STORAGE_BACKEND', 'sqlite').lower()
    if backend == 'sqlite':
        return
    if backend != 'postgresql':
        raise RuntimeError('unsupported explicit storage backend')
    from hermes_constants import get_hermes_home
    from agents.winman_delegation import require_hosted_cron_profile
    require_hosted_cron_profile(get_hermes_home())
