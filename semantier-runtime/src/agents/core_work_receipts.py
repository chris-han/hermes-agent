"""Read-only typed references to existing core work; no independent receipt store."""
from agents.task_api import validate_task_context
from services.core_work_service import CoreWorkService
from services.core_scheduled_work_service import CoreScheduledWorkService


def read_receipt(context, data):
    validate_task_context(context)
    if (not isinstance(data, dict) or set(data) != {'kind', 'id'}
            or data['kind'] not in {'work', 'schedule'}
            or not isinstance(data['id'], str) or not data['id']):
        raise ValueError('expected a typed core work or schedule reference')
    service = CoreScheduledWorkService(None) if data['kind'] == 'schedule' else CoreWorkService()
    work = service.get(context, data['id'])
    return {'ok': True, 'reference': dict(data), 'work': work}
