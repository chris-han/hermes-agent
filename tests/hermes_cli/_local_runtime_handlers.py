"""Test-only transport for existing shared CLI handler regression cases.

No application mounts these URLs. The transport lets byte-download, pause/resume,
job and lifecycle tests retain their real handler/runtime behavior after retiring
production GUI HTTP exposure. Actual CLI composition is tested separately.
"""
from fastapi import APIRouter
from fastapi.testclient import TestClient


def router():
    from hermes_cli.web_routers import local_models as handlers
    result = APIRouter()
    result.add_api_route('/api/local-models/status', handlers.local_models_status, methods=['GET'])
    result.add_api_route('/api/local-models/hardware', handlers.local_models_hardware, methods=['GET'])
    result.add_api_route('/api/local-models/catalog', handlers.local_models_catalog, methods=['GET'])
    result.add_api_route('/api/local-models/runtime/install', handlers.local_models_runtime_install, methods=['POST'])
    result.add_api_route('/api/local-models/quickstart', handlers.local_models_quickstart, methods=['POST'])
    result.add_api_route('/api/local-models/activate', handlers.local_models_activate, methods=['POST'])
    def job(job_id: str):
        return handlers._job_view(handlers._JOBS[job_id])

    def pause(body: dict):
        current = handlers._RUNNING[body['job_id']]
        current['pause'].set()
        return {'paused': True}

    def resume(body: dict):
        current = handlers._RUNNING[body['job_id']]
        current['resume']()
        return {'resumed': True}

    result.add_api_route('/api/local-models/jobs/{job_id}', job, methods=['GET'])
    result.add_api_route('/api/local-models/download/pause', pause, methods=['POST'])
    result.add_api_route('/api/local-models/download/resume', resume, methods=['POST'])
    return result


def client():
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router())
    return TestClient(app)
