"""One explicitly requested managed-work execution through the public connector."""
from __future__ import annotations

import hashlib

import base64
import json
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import threading
import time
import uuid
import yaml

from agents import job_store
from agents.hermes_public_connector import PublicHermesConnector
from agents.task_service import baseline, evidence_owner, read_skill, reserve_native, resolve_task_model
from services.core_work_service import CoreWorkService, lock


def configure_profile(profile: Path, route: dict, *, scheduled=False, qualification=False) -> Path:
    """Pin both CLI and its inherited gateway to the same keyless local route."""
    path = profile / 'config.yaml'
    if path.is_symlink():
        raise ValueError('executor configuration symlink refused')
    config = yaml.safe_load(path.read_text()) or {}
    if any(key in config for key in ('model', 'custom_providers', 'fallbacks', 'agent', 'kanban')):
        raise ValueError('unexpected pre-existing model configuration')
    config['model'] = {'default': route['model'], 'provider': route['provider'], 'base_url': route['base_url']}
    config['agent'] = {'disabled_toolsets': ['kanban', 'delegation'] + (['cronjob'] if scheduled else [])}
    if qualification:
        config['updates']={'check':False}
        config['plugins']={'auto_update_check_hours':0}
        config['model']['context_length']=65536
        config['agent'].update(api_max_retries=1,auto_recovery_cycles=0)
        config['models_dev']={'url':route['base_url'].removesuffix('/v1')+'/models-dev.json'}
        # Synthetic qualification catalogue through upstream's supported cache surface.
        catalogue={'custom':{'id':'custom','models':{route['model']:{'id':route['model'],'limit':{'context':65536,'output':4096},'tool_call':True,'attachment':False}}}}
        (profile/'models_dev_cache.json').write_text(json.dumps(catalogue,sort_keys=True))
    config['kanban'] = {'dispatch_in_gateway': False}
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    verify_profile(path, route, scheduled=scheduled, qualification=qualification)
    guard = profile / 'managed-child-guard'
    guard.mkdir()
    shutil.copyfile(Path(__file__).parents[1] / 'agents/managed_child_guard.py', guard / 'sitecustomize.py')
    return path


def verify_profile(path: Path, route: dict, *, scheduled=False, qualification=False) -> None:
    if path.is_symlink():
        raise ValueError('executor configuration symlink refused')
    config = yaml.safe_load(path.read_text()) or {}
    expected_model={'default':route['model'],'provider':route['provider'],'base_url':route['base_url']}
    expected_agent={'disabled_toolsets':['kanban','delegation']+(['cronjob'] if scheduled else [])}
    if qualification:
        if config.get('updates')!={'check':False} or config.get('plugins')!={'auto_update_check_hours':0}:
            raise ValueError('qualification update checks enabled')
        expected_model['context_length']=65536
        expected_agent.update(api_max_retries=1,auto_recovery_cycles=0)
        if config.get('models_dev')!={'url':route['base_url'].removesuffix('/v1')+'/models-dev.json'}:
            raise ValueError('qualification catalogue route changed')
    if config.get('model') != expected_model:
        raise ValueError('executor route configuration changed')
    if any(key in config for key in ('custom_providers', 'fallbacks')):
        raise ValueError('executor fallback configuration refused')
    if config.get('agent') != expected_agent or config.get('kanban') != {'dispatch_in_gateway': False}:
        raise ValueError('executor management policy changed')


class ManagedWorkExecutionService:
    def __init__(self):
        self.core = CoreWorkService()
        self.running = {}
        self.lock = threading.Lock()

    @staticmethod
    def _key(context, work_id):
        return (context.organization_id, context.workspace_id, context.user_id, work_id)

    def submit(self, context, work_id, *, resolve_context, schedule_version=None):
        current = self.core.get(context, work_id)
        if current.get('schedule') and schedule_version is None:
            from agents.durable_responsibility import DurableResponsibilityError
            raise DurableResponsibilityError('use scheduled-work run-now')
        if 'execution' in current:
            return current
        row = self.core._load(context, work_id)
        instruction = row['input'].get('instruction')
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction.encode()) > 32768:
            raise ValueError('work input requires instruction (maximum 32768 bytes)')
        route = resolve_task_model(context)
        skill_pin = baseline(context)
        skill_id, body = read_skill(context, skill_pin)
        run_id = uuid.uuid4().hex
        claim = self.core.reserve_execution(context, work_id, run_id=run_id, schedule_version=schedule_version,
            skill={'pin': skill_pin, 'skill_id': skill_id, 'sha256': hashlib.sha256(body).hexdigest()})
        if claim is None:
            return self.core.get(context, work_id)
        cancelled = threading.Event()
        thread = threading.Thread(target=self._execute, args=(context, work_id, run_id, claim, instruction, route, skill_id, body, cancelled, resolve_context, schedule_version is not None), daemon=True)
        key = self._key(context, work_id)
        with self.lock:
            self.running[key] = (cancelled, thread)
        try:
            thread.start()
        except Exception:
            with self.lock:
                self.running.pop(key, None)
            raise
        return self.core.get(context, work_id)

    def stop(self, context, work_id, *, timeout=6):
        key = self._key(context, work_id)
        with self.lock:
            entry = self.running.get(key)
        if entry:
            entry[0].set()
            entry[1].join(timeout)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            row = self.core._load(context, work_id)
            reservation = row.get('execution_reservation') or {}
            if reservation.get('state') in {'stopped', 'completed'}:
                return True
            time.sleep(.05)
        return False

    def shutdown(self):
        with self.lock:
            entries = list(self.running.values())
        for event, _ in entries:
            event.set()
        for _, thread in entries:
            thread.join(6)

    def _execute(self, context, work_id, run_id, claim, instruction, route, skill_id, body, cancelled, resolve_context, scheduled=False):
        finished = threading.Event()
        key = self._key(context, work_id)
        process_status = 'launch_not_confirmed'
        reaped = False
        committed = False
        def check_authority():
            current = resolve_context()
            if self._key(current, work_id) != key or resolve_task_model(current) != route:
                raise PermissionError('execution authority or route changed')
        @contextmanager
        def launch_guard():
            check_authority()
            with lock(context, work_id):
                if cancelled.is_set():
                    raise RuntimeError('execution cancelled before launch')
                self.core.validate_execution(context, work_id, claim=claim, run_id=run_id)
                verify_profile(config_path, route, scheduled=scheduled, qualification=qualification)
                yield
        def watch():
            while not finished.wait(.1):
                try:
                    check_authority()
                    with lock(context, work_id):
                        self.core.validate_execution(context, work_id, claim=claim, run_id=run_id)
                except Exception:
                    cancelled.set()
                    return
        watcher = None
        chat_progress = bool(self.core._load(context, work_id)['input'].get('conversation_id'))
        scheduled = scheduled or chat_progress
        qualification=chat_progress and bool(os.environ.get('SEMANTIER_CHAT_QUALIFICATION_HTTP_LIMITS'))
        def progress(event):
            if not chat_progress:
                return
            event=dict(event)
            if isinstance(event.get('timestamp'),(int,float)):
                event['source_timestamp_ms']=event['timestamp']
                event['timestamp']=datetime.fromtimestamp(event['timestamp']/1000,timezone.utc).isoformat()
            with lock(context, work_id):
                self.core.validate_execution(context, work_id, claim=claim, run_id=run_id)
                with job_store._connect() as conn:
                    row = conn.execute('SELECT payload_json FROM semantier_jobs WHERE user_id=%s AND workspace_id=%s AND job_id=%s FOR UPDATE',
                        (context.user_id,context.workspace_id,work_id)).fetchone()
                    payload = json.loads(row['payload_json'])
                    events = payload.setdefault('progress_events', [])
                    if len(json.dumps(events).encode()) + len(json.dumps(event).encode()) > 1048576:
                        raise ValueError('conversation progress limit reached')
                    events.append(event)
                    conn.execute('UPDATE semantier_jobs SET payload_json=%s WHERE user_id=%s AND workspace_id=%s AND job_id=%s',
                        (json.dumps(payload,sort_keys=True),context.user_id,context.workspace_id,work_id))
        try:
            root = context.workspace_root / 'tasks' / ('work-' + run_id)
            root.mkdir(parents=True, exist_ok=False)
            workspace = root / 'files'; workspace.mkdir()
            profile = root / 'executor'; profile.mkdir()
            connector = PublicHermesConnector.from_environment(home=profile)
            connector.install(plugin_source=Path(__file__).parents[1] / 'plugins/hermes_public_observer', skill_id=skill_id, body=body,
                              observer=os.environ.get('SEMANTIER_TASK_MONITORING', 'on') != 'off', observer_optional=True)
            config_path = configure_profile(profile, route, scheduled=scheduled, qualification=qualification)
            watcher = threading.Thread(target=watch, daemon=True)
            watcher.start()
            run = connector.run(query=instruction, run_id=run_id, provider_args=('--provider', route['provider'], '--model', route['model'], '--base-url', route['base_url']), timeout=120, cancel=cancelled, on_event=progress if chat_progress else None, workspace=workspace, isolate=True, reserve=reserve_native, launch_guard=launch_guard, readonly_overlays=(config_path, profile / 'managed-child-guard/sitecustomize.py'), pinned_route=route, http_limits=json.loads(os.environ['SEMANTIER_CHAT_QUALIFICATION_HTTP_LIMITS']) if chat_progress and os.environ.get('SEMANTIER_CHAT_QUALIFICATION_HTTP_LIMITS') else None)
            reaped = True
            process_status = run.status
            if run.status == 'completed' and run.returncode == 0 and run.result is not None and not cancelled.is_set():
                result = {'run_id': run_id, 'native_result': run.result, 'returncode': run.returncode,
                          'stdout_base64': base64.b64encode(run.stdout).decode(),
                          'stderr_base64': base64.b64encode(run.stderr).decode(),
                          'gaps': list(run.gaps), 'monitoring': {'status': 'unavailable', 'reason': 'optional observation not part of terminal commit'}}
                check_authority()
                self.core.finish_execution(context, work_id, claim=claim, run_id=run_id, result=result)
                committed = True
                process_status = 'completed'
                # Optional monitoring is not result authority and cannot veto completion.
                store = None
                try:
                    check_authority()
                    store, evidence = evidence_owner()
                    monitoring = connector.attach(service=evidence, context=context, run=run, profile_id='work-' + run_id)
                    job_store.record_managed_execution_monitoring(context=context, work_id=work_id,
                        run_id=run_id, epoch=claim.lease_epoch, monitoring=monitoring)
                except Exception:
                    pass
                finally:
                    if store: store.close()
        except Exception:
            process_status = 'uncertain'
        finally:
            finished.set()
            if watcher: watcher.join(2)
            if not committed:
                try:
                    job_store.mark_managed_execution_stopped(context=context, work_id=work_id, run_id=run_id, epoch=claim.lease_epoch, process_status=process_status, reaped=reaped)
                except Exception:
                    pass  # A durable reservation remains; uncertainty never redispatches.
            with self.lock:
                self.running.pop(key, None)
