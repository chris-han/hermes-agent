"""Native producer capture and pure eligibility projection over existing owners.

No model, candidate formation, activation, new database, or inferred skill use.
Only exact direct script invocations are captured; all other tools pass normally.
"""

from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shlex
import threading
import time

from agents.auth_session import RequestContext
from contracts.learning_workbench import SkillArtifactVersionV1
from contracts.personal_learning_observation import (
    ProcedureExecutionReceiptV1,
    PersonalLearningObservationV1,
    WorkspaceInputPinV1,
)


def result_hash(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def occurrence_id(receipt):
    return result_hash(
        json.dumps(
            [
                receipt.organization_id,
                receipt.workspace_id,
                receipt.session_id,
                receipt.task_id,
                receipt.tool_call_id,
            ],
            separators=(",", ":"),
        )
    )


def _direct_script_argv(args):
    command = args.get("command", "")
    if (
        args.get("background")
        or not isinstance(command, str)
        or any(c in command for c in ";|&<>`$\n\r")
    ):
        return None
    try:
        argv = shlex.split(command)
        if (
            len(argv) < 4
            or argv[0] not in ("python", "python3")
            or not Path(argv[1]).is_absolute()
            or Path(argv[1]).suffix != ".py"
            or argv.count("--input") != 1
            or not Path(argv[argv.index("--input") + 1]).is_absolute()
        ):
            return None
        return argv
    except (ValueError, IndexError):
        return None


def _argument_contract_hash(argv, args):
    shape = argv[2:]
    shape[shape.index("--input") + 1] = "<governed-input>"
    return result_hash(
        json.dumps(
            [shape, {k: v for k, v in args.items() if k != "command"}],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _is_upload_relative(path):
    parts = path.parts
    return (len(parts) >= 2 and parts[0] == "uploads") or (
        len(parts) >= 4 and parts[0] == "sessions" and parts[2] == "uploads"
    )


@dataclass(frozen=True)
class PreparedProcedureExecution:
    context: RequestContext
    session_id: str
    task_id: str
    tool_call_id: str
    version: SkillArtifactVersionV1
    directory: Path
    entrypoint: str
    input_pins: tuple[WorkspaceInputPinV1, ...]
    command_hash: str
    argument_contract_hash: str
    synthetic: bool
    started_at: datetime


class PersonalProcedureCaptureService:
    def __init__(self, *, store, procedure_owner):
        self.store = store
        self.procedure_owner = procedure_owner

    def prepare(self, *, context, session_id, task_id, tool_call_id, args, synthetic):
        self.procedure_owner._scope(context)
        if not session_id.startswith(context.workspace_id + ":"):
            raise PermissionError("CROSS_WORKSPACE_EXECUTION")
        argv = _direct_script_argv(args)
        if argv is None or not task_id or not tool_call_id:
            return None
        command = args["command"]
        try:
            script = Path(argv[1])
            if not script.is_absolute() or script.is_symlink():
                return None
            relative = script.resolve().relative_to(
                context.hermes_home.resolve() / "skills"
            )
            if (
                len(relative.parts) < 3
                or relative.parts[1] != "scripts"
                or script.suffix != ".py"
            ):
                return None
            skill_id = relative.parts[0]
            # Explicit --input is the bounded source-group witness, not an inferred task label.
            if argv.count("--input") != 1:
                return None
            input_path = Path(argv[argv.index("--input") + 1])
            if not input_path.is_absolute() or input_path.is_symlink():
                return None
            source_relative = input_path.resolve().relative_to(
                context.workspace_root.resolve()
            )
            if not _is_upload_relative(source_relative) or not input_path.is_file():
                return None
        except (ValueError, IndexError, OSError):
            return None
        version = self.procedure_owner.capture_baseline(context, skill_id)
        self.procedure_owner._assert_base(context, version)
        entrypoint = Path(*relative.parts[1:]).as_posix()
        if entrypoint not in {f.relative_path for f in version.files}:
            return None
        inputs = (
            WorkspaceInputPinV1(
                relative_path=source_relative.as_posix(),
                content_hash="sha256:"
                + hashlib.sha256(input_path.read_bytes()).hexdigest(),
            ),
        )
        return PreparedProcedureExecution(
            context,
            session_id,
            task_id,
            tool_call_id,
            version,
            context.hermes_home / "skills" / skill_id,
            entrypoint,
            inputs,
            result_hash(command),
            _argument_contract_hash(argv, args),
            synthetic,
            datetime.now(timezone.utc),
        )

    def complete(self, *, prepared, result, status, duration_ms=0):
        p = prepared
        try:
            payload = json.loads(result) if isinstance(result, str) else result
            code = payload.get("exit_code") if isinstance(payload, dict) else None
            if type(code) is not int:
                code = None
        except (ValueError, TypeError):
            code = None
        try:
            version_matches = (
                self.procedure_owner._files(p.directory) == p.version.files
            )
            inputs_match = all(
                "sha256:"
                + hashlib.sha256(
                    (p.context.workspace_root / pin.relative_path).read_bytes()
                ).hexdigest()
                == pin.content_hash
                for pin in p.input_pins
            )
        except OSError:
            version_matches = inputs_match = False
        receipt = ProcedureExecutionReceiptV1.create(
            organization_id=p.context.organization_id,
            workspace_id=p.context.workspace_id,
            owner_ref=p.context.user_id,
            session_id=p.session_id,
            task_id=p.task_id,
            tool_call_id=p.tool_call_id,
            skill_id=p.version.skill_id,
            entrypoint=p.entrypoint,
            predecessor=p.version.pin(),
            predecessor_verified=version_matches,
            input_pins=p.input_pins,
            inputs_verified=inputs_match,
            command_hash=p.command_hash,
            argument_contract_hash=p.argument_contract_hash,
            result_hash=result_hash(result),
            exit_code=code,
            producer_status=status or "unknown",
            synthetic=p.synthetic,
            started_at=p.started_at,
            completed_at=datetime.now(timezone.utc),
            duration_ms=max(0, int(duration_ms)),
        )
        from agents.workspace_session_logs import append_workspace_procedure_receipt

        return append_workspace_procedure_receipt(
            p.context.hermes_home, receipt=receipt
        )

    def project(
        self, *, context, receipt, messages, reviewed_targets, active_conflicts
    ):
        self.procedure_owner._scope(context)
        if (receipt.organization_id, receipt.workspace_id, receipt.owner_ref) != (
            context.organization_id,
            context.workspace_id,
            context.user_id,
        ):
            raise PermissionError("CROSS_WORKSPACE_OR_ACTOR_RECEIPT")
        if not receipt.session_id.startswith(context.workspace_id + ":"):
            raise PermissionError("CROSS_WORKSPACE_SESSION")
        reasons = []
        version = None
        if receipt.predecessor is None or not receipt.predecessor_verified:
            reasons.append("INELIGIBLE_MISSING_PREDECESSOR")
        else:
            try:
                version = self.procedure_owner._load(
                    context,
                    receipt.predecessor.artifact_ref,
                    SkillArtifactVersionV1,
                    receipt.predecessor.artifact_hash,
                )
                if version.skill_id != receipt.skill_id or receipt.entrypoint not in {
                    f.relative_path for f in version.files
                }:
                    reasons.append("INTEGRITY_BLOCKED")
            except (LookupError, ValueError, PermissionError):
                reasons.append("INELIGIBLE_MISSING_PREDECESSOR")
        results = [
            m
            for m in messages
            if m.get("role") == "tool"
            and m.get("tool_call_id") == receipt.tool_call_id
            and m.get("tool_name") == "terminal"
        ]
        message = results[0] if len(results) == 1 else None
        if receipt.exit_code is None or message is None:
            reasons.append("INELIGIBLE_MISSING_OUTCOME")
        elif result_hash(message.get("content")) != receipt.result_hash:
            reasons.append("INTEGRITY_BLOCKED")
        else:
            try:
                native_result = json.loads(message["content"])
                if (
                    type(native_result.get("exit_code")) is not int
                    or native_result["exit_code"] != receipt.exit_code
                ):
                    reasons.append("INTEGRITY_BLOCKED")
            except (ValueError, TypeError, AttributeError, KeyError):
                reasons.append("INTEGRITY_BLOCKED")
        if not receipt.input_pins or not receipt.task_id or not receipt.tool_call_id:
            reasons.append("INSUFFICIENT_ATTRIBUTION")
        if receipt.producer_status not in ("ok", "error") or (
            receipt.exit_code is not None and receipt.exit_code < 0
        ):
            reasons.append("INSUFFICIENT_ATTRIBUTION")
        requests = []
        for m in messages:
            if m.get("role") != "assistant":
                continue
            for call in m.get("tool_calls") or []:
                if call.get("id") != receipt.tool_call_id:
                    continue
                fn = call.get("function") or {}
                try:
                    args = (
                        json.loads(fn["arguments"])
                        if isinstance(fn.get("arguments"), str)
                        else fn["arguments"]
                    )
                    if fn.get("name") == "terminal" and isinstance(args, dict):
                        requests.append(args)
                except (KeyError, ValueError, TypeError):
                    pass
        if len(requests) != 1:
            reasons.append("INSUFFICIENT_ATTRIBUTION")
        elif result_hash(requests[0].get("command")) != receipt.command_hash:
            reasons.append("INTEGRITY_BLOCKED")
        else:
            try:
                argv = _direct_script_argv(requests[0])
                if argv is None:
                    raise ValueError("UNSUPPORTED_INVOCATION")
                expected_script = (
                    context.hermes_home
                    / "skills"
                    / receipt.skill_id
                    / receipt.entrypoint
                )
                input_index = argv.index("--input") + 1
                native_input = (
                    Path(argv[input_index])
                    .relative_to(context.workspace_root)
                    .as_posix()
                )
                if (
                    argv[0] not in ("python", "python3")
                    or Path(argv[1]) != expected_script
                    or len(receipt.input_pins) != 1
                    or receipt.input_pins[0].relative_path != native_input
                    or _argument_contract_hash(argv, requests[0])
                    != receipt.argument_contract_hash
                ):
                    reasons.append("INTEGRITY_BLOCKED")
            except (ValueError, IndexError, TypeError):
                reasons.append("INTEGRITY_BLOCKED")
        if not receipt.inputs_verified:
            reasons.append("INTEGRITY_BLOCKED")
        # Replay validates the producer's immutable pre/post witness, never current bytes.
        for pin in receipt.input_pins:
            relative = Path(pin.relative_path)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not _is_upload_relative(relative)
            ):
                reasons.append("INTEGRITY_BLOCKED")
        binding_complete = not reasons
        target_scope = (
            reviewed_targets.get(receipt.predecessor.artifact_hash)
            if receipt.predecessor
            else None
        )
        if target_scope in ("SEMANTIC", "AUTHORITY"):
            reasons.append("INELIGIBLE_AUTHORITY_OR_SEMANTIC_TARGET")
        elif target_scope != "PROCEDURAL":
            reasons.append("INELIGIBLE_NOT_PROCEDURAL")
        if receipt.skill_id in active_conflicts:
            reasons.append("INELIGIBLE_ACTIVE_OWNER_CONFLICT")
        if receipt.synthetic:
            reasons.append("INELIGIBLE_SYNTHETIC")
        reasons = tuple(dict.fromkeys(reasons))
        family = result_hash(
            [
                receipt.skill_id,
                receipt.entrypoint,
                receipt.predecessor.artifact_hash if receipt.predecessor else None,
                receipt.argument_contract_hash,
            ]
        )
        group = (
            result_hash(
                json.dumps(
                    [
                        family,
                        receipt.predecessor.artifact_hash
                        if receipt.predecessor
                        else None,
                        sorted(pin.content_hash for pin in receipt.input_pins),
                    ],
                    separators=(",", ":"),
                )
            )
            if receipt.input_pins
            else None
        )
        outcome = (
            "UNKNOWN"
            if not binding_complete
            else ("SUCCESS" if receipt.exit_code == 0 else "FAILURE")
        )
        return PersonalLearningObservationV1.create(
            organization_id=receipt.organization_id,
            workspace_id=receipt.workspace_id,
            owner_ref=receipt.owner_ref,
            occurrence_id=occurrence_id(receipt),
            family_id=family,
            group_id=group,
            producer_receipt=receipt.pin(),
            predecessor=receipt.predecessor,
            result_message_id=str(message["id"])
            if message and "id" in message
            else None,
            outcome=outcome,
            binding_complete=binding_complete,
            eligibility=reasons[0] if reasons else "ELIGIBLE",
            rejection_reasons=reasons,
        )

    def project_session(
        self,
        *,
        context,
        session_id,
        messages,
        reviewed_targets,
        active_conflicts,
        tool_call_ids=None,
    ):
        from agents.workspace_session_logs import list_workspace_session_trajectory

        self.procedure_owner._scope(context)
        if not session_id.startswith(context.workspace_id + ":"):
            raise PermissionError("CROSS_WORKSPACE_SESSION")
        observations = {}
        for row in list_workspace_session_trajectory(context.hermes_home, session_id):
            if row.get("record_type") != "procedure_execution_receipt":
                continue
            if (
                tool_call_ids is not None
                and row["receipt"].get("tool_call_id") not in tool_call_ids
            ):
                continue
            receipt = ProcedureExecutionReceiptV1.model_validate(row["receipt"])
            value = self.project(
                context=context,
                receipt=receipt,
                messages=messages,
                reviewed_targets=reviewed_targets,
                active_conflicts=active_conflicts,
            )
            key = value.occurrence_id
            if key in observations and observations[key] != value:
                raise ValueError("PRODUCER_OCCURRENCE_CONFLICT")
            observations[key] = value
        return tuple(observations[key] for key in sorted(observations))


# Temporary pre/post pairing only. Durable execution truth belongs to native producer logs.
_pending = {}
_pending_lock = threading.Lock()
_logger = logging.getLogger(__name__)


def _native_context(session_id):
    from gateway.execution_boundary import current_execution_boundary
    from agents.gateway_identity import get_user_record, get_user_organization_context
    from runtime_paths import workspace_root_path, workspace_hermes_home_path

    boundary = current_execution_boundary()
    if (
        boundary is None
        or not boundary.user_id
        or not boundary.workspace_id
        or boundary.audit_metadata.get("authority_source")
        != "semantier_authenticated_context"
    ):
        return None
    user = get_user_record(boundary.user_id)
    org = get_user_organization_context(boundary.user_id)
    if (
        not user
        or not org
        or user.get("workspace_id") != boundary.workspace_id
        or not org.get("organization_id")
        or not session_id.startswith(boundary.workspace_id + ":")
    ):
        return None
    home = workspace_hermes_home_path(boundary.workspace_id)
    if (
        boundary.paths.hermes_home is None
        or boundary.paths.hermes_home.resolve() != home.resolve()
    ):
        return None
    return RequestContext(
        authenticated=True,
        user_id=boundary.user_id,
        workspace_id=boundary.workspace_id,
        workspace_slug=str(user.get("workspace_slug") or boundary.workspace_id),
        workspace_root=workspace_root_path(boundary.workspace_id),
        hermes_home=home,
        organization_id=org["organization_id"],
        membership_status=org.get("membership_status") or "unassigned",
        dataset_type=org.get("dataset_type"),
    )


@contextmanager
def _native_owners(context):
    from storage.ontology_build_store import (
        OntologyBuildStore,
        ontology_build_store_path,
    )
    from storage.governance_store import GovernanceStore, _db_path
    from eos.db import init_db
    from services.learned_skill_publication_service import (
        LearnedSkillPublicationService,
    )

    init_db(_db_path())
    store = OntologyBuildStore(ontology_build_store_path(context.workspace_root))
    governance = GovernanceStore()
    try:
        yield PersonalProcedureCaptureService(
            store=store,
            procedure_owner=LearnedSkillPublicationService(
                store=store, governance=governance
            ),
        )
    finally:
        governance.close()
        store.close()


def begin_native_procedure_capture(
    *, tool_name, args, session_id="", task_id="", tool_call_id="", **_
):
    if tool_name != "terminal" or _direct_script_argv(args) is None:
        return
    try:
        from tools.terminal_tool import _session_scope

        if _session_scope().env_type != "local":
            return  # Host snapshots cannot attest bytes executed by a remote backend.
        context = _native_context(session_id)
        if context is None:
            return
        with _native_owners(context) as service:
            prepared = service.prepare(
                context=context,
                session_id=session_id,
                task_id=task_id,
                tool_call_id=tool_call_id,
                args=args,
                synthetic=(
                    os.environ.get("SEMANTIER_EPISODE_SOURCE_KIND") == "SYNTHETIC"
                    or context.dataset_type == "DEMO"
                ),
            )
        if prepared is not None:
            now = time.monotonic()
            with _pending_lock:
                for key, (_, observed) in list(_pending.items()):
                    if now - observed > 3600:
                        del _pending[key]
                if len(_pending) < 256:
                    _pending[
                        (context.workspace_id, session_id, task_id, tool_call_id)
                    ] = (prepared, now)
    except Exception as exc:
        # Capture failure never blocks ordinary low-risk work or invents an eligible record.
        _logger.warning("Procedure capture unavailable: %s", type(exc).__name__)


def finish_native_procedure_capture(
    *,
    tool_name,
    result,
    session_id="",
    task_id="",
    tool_call_id="",
    status=None,
    duration_ms=0,
    **_,
):
    if tool_name != "terminal":
        return
    try:
        context = _native_context(session_id)
        if context is None:
            return
        with _pending_lock:
            pair = _pending.pop(
                (context.workspace_id, session_id, task_id, tool_call_id), None
            )
        if pair is None:
            return
        prepared, _ = pair
        if (
            prepared.context.user_id != context.user_id
            or prepared.context.organization_id != context.organization_id
        ):
            raise PermissionError("CAPTURE_AUTHORITY_CONTEXT_CHANGED")
        with _native_owners(context) as service:
            service.complete(
                prepared=prepared, result=result, status=status, duration_ms=duration_ms
            )
    except Exception as exc:
        _logger.warning(
            "Procedure completion capture unavailable: %s", type(exc).__name__
        )
