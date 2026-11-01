"""Independent task orchestration. PostgreSQL/core own outcomes and Skill custody.

Native sessions are task-local execution metadata. No transcript is accepted as
cross-task context; an idempotency replay only reads the original outcome.
"""

from __future__ import annotations
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import uuid
from urllib.parse import urlsplit

from agents import job_store
from agents.hermes_public_connector import PublicHermesConnector
from contracts.learning_workbench import SkillArtifactVersionV1
from contracts.learning_trace_evidence import DelegationBasisV1
from services.learning_trace_evidence_service import LearningTraceEvidenceService


def evidence_owner():
    from storage.ontology_build_store import OntologyBuildStore

    root = Path(os.environ["SEMANTIER_LOCAL_STATE_DIR"])
    store = OntologyBuildStore(root / "artifacts.db")
    service = LearningTraceEvidenceService(store=store)
    return store, service


def owners():
    from storage.governance_store import GovernanceStore
    from services.learned_skill_package_custody import LearnedSkillPackageCustody

    store, service = evidence_owner()
    try:
        custody = LearnedSkillPackageCustody(
            store=store,
            governance=GovernanceStore(
                Path(os.environ["SEMANTIER_LOCAL_STATE_DIR"]) / "eos.db"
            ),
        )
        return store, service, custody
    except Exception:
        store.close()
        raise


def baseline(context):
    store, service, custody = owners()
    try:
        body = (
            Path(__file__).parents[1] / "prompts/tasks/task-runtime-skill.md"
        ).read_bytes()
        raw = json.dumps(
            {
                "skill_id": "task-runtime",
                "files": [
                    {
                        "relative_path": "SKILL.md",
                        "content_base64": base64.b64encode(body).decode(),
                        "content_hash": "sha256:" + hashlib.sha256(body).hexdigest(),
                    }
                ],
            },
            sort_keys=True,
        ).encode()
        row = service.record_public(
            context=context,
            raw_source=raw,
            run_id="task-baseline",
            session_id=context.workspace_id + ":task-baseline",
            profile_id="core",
            source_kind="core_receipt",
            source_version="task-package.v1",
            source_locator="repo:task-runtime-skill",
            event_kind="package",
            sequence=0,
            payload=json.loads(raw),
            captured_at=datetime.now(timezone.utc),
        )
        return (
            custody.capture_package(
                context, row.pin().model_dump(mode="json"), "task-runtime-baseline"
            )
            .pin()
            .model_dump(mode="json")
        )
    finally:
        custody._governance.close()
        store.close()


def read_skill(context, pin):
    store, service = evidence_owner()
    try:
        scope = service.public_scope(context)
        version = store.get(
            **scope,
            artifact_ref=pin["artifact_ref"],
            contract_type=SkillArtifactVersionV1,
        )
        if (
            version.artifact_hash != pin["artifact_hash"]
            or version.owner_ref != context.user_id
        ):
            raise PermissionError("Skill pin is not owned or its hash changed")
        if len(version.files) != 1 or version.files[0].relative_path != "SKILL.md":
            raise ValueError("task mode supports a single declared Skill file")
        return version.skill_id, version.files[0].content_bytes()
    finally:
        store.close()


def resolve_task_model(context):
    from agents.auth_db import get_workspace_config

    config = get_workspace_config(context.workspace_id) or {}
    route = config.get("task_model")
    return validate_task_model(route)


def validate_task_model(route):
    if (
        not isinstance(route, dict)
        or set(route) != {"provider", "model", "base_url"}
        or not all(isinstance(v, str) and v for v in route.values())
    ):
        raise ValueError("workspace task_model configuration required")
    endpoint = urlsplit(route["base_url"])
    if (
        route["provider"] != "custom"
        or endpoint.scheme != "http"
        or endpoint.hostname != "127.0.0.1"
        or endpoint.username is not None
        or endpoint.password is not None
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError(
            "selected task MVL requires a configured keyless loopback custom endpoint; paid provider delivery is not qualified"
        )
    return route


class TaskService:
    def __init__(self):
        self.running = {}
        self.lock = threading.Lock()

    def submit(
        self, context, *, request_id, instruction, skill_pin, monitoring_basis=None, value_purpose="unknown"
    ):
        if value_purpose not in {"real", "demo", "unknown"}:
            raise ValueError("Unsupported task value purpose")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", request_id):
            raise ValueError("invalid request_id")
        if (
            not isinstance(instruction, str)
            or not instruction.strip()
            or len(instruction.encode()) > 32768
        ):
            raise ValueError("task instruction required (maximum 32768 bytes)")
        skill_id, body = read_skill(context, skill_pin)
        route = resolve_task_model(context)
        # No request can select a host path, executor profile or provider credential.
        basis = None
        if monitoring_basis is not None:
            basis = DelegationBasisV1.model_validate(monitoring_basis)
            if (
                basis.instruction_hash
                != "sha256:" + hashlib.sha256(instruction.encode()).hexdigest()
            ):
                raise ValueError("monitoring basis does not bind task instruction")
        request_hash = hashlib.sha256(
            json.dumps(
                {
                    "instruction": instruction,
                    "skill_pin": skill_pin,
                    "route": route,
                    "monitoring_basis": monitoring_basis,
                    **({"value_purpose":value_purpose} if value_purpose!="unknown" else {}),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        task_id = "task_" + request_id
        job, fresh = job_store.claim_independent_task(
            context=context,
            task_id=task_id,
            request_hash=request_hash,
            payload={
                "instruction": instruction,
                "value_purpose": value_purpose,
                "skill_pin": skill_pin,
                "skill_sha256": hashlib.sha256(body).hexdigest(),
                "model": route,
                "monitoring": "optional",
                "skill_delivery": "experimental; admission and activation not asserted",
                "monitoring_basis": monitoring_basis,
            },
        )
        if fresh:
            cancel = threading.Event()
            thread = threading.Thread(
                target=self._execute,
                args=(
                    context,
                    task_id,
                    instruction,
                    skill_id,
                    body,
                    route,
                    cancel,
                    basis,
                ),
                daemon=True,
            )
            with self.lock:
                self.running[(context.user_id, context.workspace_id, task_id)] = (
                    cancel,
                    thread,
                )
            thread.start()
        return job

    def cancel(self, context, task_id):
        with self.lock:
            entry = self.running.get((context.user_id, context.workspace_id, task_id))
        if entry:
            entry[0].set()
        return bool(entry)

    def shutdown(self):
        with self.lock:
            entries = list(self.running.values())
        for cancel, _ in entries:
            cancel.set()
        for _, thread in entries:
            thread.join(6)

    def _execute(
        self, context, task_id, instruction, skill_id, body, route, cancel, basis
    ):
        scope = dict(
            user_id=context.user_id, workspace_id=context.workspace_id, job_id=task_id
        )
        events = []
        try:
            task_root = context.workspace_root / "tasks" / task_id
            task_root.mkdir(parents=True, exist_ok=False)
            workspace = task_root / "files"
            workspace.mkdir()
            profile = task_root / "executor"
            profile.mkdir()
            connector = PublicHermesConnector.from_environment(home=profile)
            connector.install(
                plugin_source=Path(__file__).parents[1]
                / "plugins/hermes_public_observer",
                skill_id=skill_id,
                body=body,
                observer=os.environ.get("SEMANTIER_TASK_MONITORING", "on") != "off",
                observer_optional=True,
            )

            def event(row):
                if len(events) >= 500:
                    return
                events.append(row)
                # Only durable progress is exposed. A failed write stops execution.
                try:
                    job_store.update_job(
                        **scope, updates={"state": "running", "events": list(events)}
                    )
                except Exception:
                    cancel.set()
                    raise

            job_store.update_job(**scope, updates={"state": "running"})
            run = connector.run(
                query=instruction,
                run_id=uuid.uuid4().hex,
                provider_args=(
                    "--provider",
                    route["provider"],
                    "--model",
                    route["model"],
                    "--base-url",
                    route["base_url"],
                ),
                timeout=120,
                cancel=cancel,
                on_event=event,
                workspace=workspace,
                isolate=True,
                reserve=reserve_native,
            )
            success = (
                run.status == "completed"
                and run.returncode == 0
                and run.result is not None
            )
            state = (
                "completed"
                if success
                else run.status
                if run.status in {"cancelled", "timeout"}
                else "failed"
            )
            monitoring = {"status": "unavailable"}
            store = None
            try:
                store, service = evidence_owner()
                monitoring = connector.attach(
                    service=service, context=context, run=run, profile_id=task_id
                )
                if monitoring["status"] == "retained" and basis is not None:
                    assessment, receipt = service.assess_public(
                        context=context,
                        basis=basis,
                        session_id=context.workspace_id + ":connector:" + run.run_id,
                        run_id=run.run_id,
                        profile_id=task_id,
                        completion="PASS" if success else "FAIL",
                        coverage="PARTIAL",
                        surface="public-task-events",
                        recorded_at=datetime.now(timezone.utc),
                    )
                    monitoring["assessment"] = assessment.model_dump(mode="json")
                    monitoring["assessment_pin"] = receipt.pin().model_dump(mode="json")
            except Exception as exc:
                monitoring = {"status": "unavailable", "reason": type(exc).__name__}
            finally:
                if store:
                    store.close()
            job_store.update_job(
                **scope,
                updates={
                    "state": state,
                    "events": events,
                    "result": run.result,
                    "run_id": run.run_id,
                    "returncode": run.returncode,
                    "gaps": run.gaps,
                    "monitoring": monitoring,
                    "elapsed_seconds": run.elapsed_seconds,
                    "process_stdout_base64": base64.b64encode(run.stdout).decode(),
                    "process_stderr_base64": base64.b64encode(run.stderr).decode(),
                },
            )
        except Exception as exc:
            try:
                job_store.update_job(
                    **scope,
                    updates={
                        "state": "failed",
                        "error": type(exc).__name__ + ":" + str(exc)[:300],
                        "effects": "uncertain; same request will not redispatch",
                    },
                )
            except Exception:
                pass  # Existing queued/running row stays uncertain; startup marks interrupted.
        finally:
            with self.lock:
                self.running.pop((context.user_id, context.workspace_id, task_id), None)


def reserve_native():
    """Optional qualification ledger. Reserve CLI and implicit gateway before start."""
    path = os.environ.get("SEMANTIER_TASK_QUALIFICATION_LEDGER")
    if not path:
        return
    import fcntl

    with open(path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        data = json.load(f)
        limits = data.get("limits", {"native_tasks": 10, "host_starts": 24})
        if data["native_tasks"] >= limits["native_tasks"] or data["host_starts"] + 2 > limits["host_starts"]:
            raise RuntimeError("task qualification budget exhausted")
        data["native_tasks"] += 1
        data["host_starts"] += 2
        f.seek(0)
        json.dump(data, f)
        f.truncate()
        f.flush()
        os.fsync(f.fileno())
