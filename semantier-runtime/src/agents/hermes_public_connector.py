"""Opt-in public Hermes process transport; observation failures never veto tasks."""
from dataclasses import dataclass
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time
import yaml
from urllib.parse import urlsplit


@dataclass(frozen=True)
class PublicRunResult:
    run_id: str
    returncode: int | None
    status: str
    stdout: bytes
    stderr: bytes
    result: dict | None
    observations: tuple[dict, ...]
    gaps: tuple[str, ...]
    elapsed_seconds: float


def run_bounded_process(command, *, cwd, env, run_id, timeout=60, cancel=None,
                        output_limit=1024*1024, reserve=None, on_event=None, launch_guard=None):
    """Transport primitive. Tests with helper processes are not native qualification."""
    if timeout <= 0 or output_limit < 1:
        raise ValueError("INVALID_PROCESS_LIMIT")
    started=time.monotonic()
    with launch_guard() if launch_guard is not None else nullcontext():
        if reserve is not None:
            reserve()
        process=subprocess.Popen(command,cwd=cwd,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,start_new_session=True)
    buffers=[bytearray(),bytearray()]
    dropped=[False,False]
    sink_failed=threading.Event()
    def drain(pipe,index):
        pending = b""
        while raw:=pipe.read1(4096):
            remaining=max(0,output_limit-len(buffers[index]))
            buffers[index].extend(raw[:remaining])
            dropped[index] |= len(raw)>remaining
            if index == 0 and on_event is not None:
                pending += raw[:remaining]
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if isinstance(row, dict):
                        try:
                            on_event(row)
                        except Exception:
                            sink_failed.set()
    readers=[threading.Thread(target=drain,args=(pipe,i),daemon=True)
             for i,pipe in enumerate((process.stdout,process.stderr))]
    for reader in readers: reader.start()
    status="completed"
    try:
        while process.poll() is None:
            if sink_failed.is_set():
                status="event_sink_failed";break
            if cancel is not None and cancel.is_set():
                status="cancelled";break
            if time.monotonic()-started >= timeout:
                status="timeout";break
            try:process.wait(timeout=.03)
            except subprocess.TimeoutExpired:pass
    finally:
        # Reap owned descendants even when the direct process already returned.
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait(timeout=2)
        for reader in readers:reader.join(timeout=2)
        if any(reader.is_alive() for reader in readers):
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            for reader in readers:reader.join(timeout=2)
        for pipe in (process.stdout,process.stderr):pipe.close()
    gaps=["output_truncated"] if any(dropped) else []
    if sink_failed.is_set():
        status="event_sink_failed"
        gaps.append("event_sink_unavailable")
    result=None
    result_count=0
    for line in bytes(buffers[0]).splitlines():
        if not line.startswith(b"{"):continue
        try:row=json.loads(line)
        except (ValueError,UnicodeDecodeError):
            gaps.append("malformed_public_json");continue
        if isinstance(row,dict) and row.get("type")=="result":
            result_count+=1
            result=row
    if result_count>1:
        result=None
        gaps.append("ambiguous_public_result")
    return PublicRunResult(run_id,process.returncode,status,bytes(buffers[0]),
        bytes(buffers[1]),result,(),tuple(sorted(set(gaps))),time.monotonic()-started)


def normalize_public_action(event):
    """Map only an unambiguous successful public write_file report."""
    payload=event.get("payload", {})
    if event.get("hook") != "post_tool_call" or payload.get("tool_name") != "write_file" or payload.get("status") != "ok":
        return None
    args=payload.get("args")
    try:
        result=json.loads(payload.get("result", ""))
    except (TypeError, ValueError):
        return None
    if not isinstance(args, dict) or not isinstance(result, dict):
        return None
    path=args.get("path")
    if not isinstance(path, str) or not path or result.get("resolved_path") != path or not payload.get("tool_call_id"):
        return None
    return {"operation":"write", "object_ref":path, "recipient":None,
            "sequence":event.get("sequence"), "occurred_at":None}


def collect_session_observations(rows, session_id, claimed):
    """Filter raw collector rows only by an unambiguous native session."""
    if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", session_id):
        return (), ("native_session_missing_or_unsafe",)
    if session_id in claimed:
        return (), ("native_session_reused",)
    selected=[]; gaps=[]; sequences=set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("payload"),dict):
            gaps.append("malformed_observer_event"); continue
        hook=row.get("hook")
        if hook == "registration":
            continue
        native=row["payload"].get("session_id")
        if not isinstance(native,str) or not native:
            gaps.append("unscoped_observer_event"); continue
        if native != session_id:
            continue
        seq=row.get("sequence")
        if type(seq) is not int or seq in sequences:
            return (), ("observer_sequence_ambiguous",)
        sequences.add(seq); selected.append(row)
    if not any(r.get("hook")=="on_session_end" for r in selected):
        gaps.append("session_checkpoint_missing")
    return tuple(selected),tuple(sorted(set(gaps)))


class PublicHermesConnector:
    @classmethod
    def from_environment(cls, *, home):
        """Select the supported external pin without importing its implementation."""
        from agents.hermes_upstream import UpstreamInstallation
        installation = UpstreamInstallation.from_environment()
        installation.verify()
        return cls(host_root=installation.root, host_commit=installation.commit,
                   home=home, python=installation.python)

    def __init__(self, *, host_root, host_commit, home, python):
        if not re.fullmatch("[0-9a-f]{40}",host_commit):
            raise ValueError("EXACT_HOST_COMMIT_REQUIRED")
        self.host_root=Path(host_root).resolve()
        self.home=Path(home).resolve()
        self.host_commit=host_commit
        self.python=str(python)
        self.skill_id=None
        self.current_hash=None
        self._claimed_sessions=set()

    def verify_host(self):
        result=subprocess.run(["/usr/bin/git","-C",str(self.host_root),"rev-parse","HEAD"],
            capture_output=True,text=True,timeout=10)
        if result.returncode or result.stdout.strip()!=self.host_commit:
            raise ValueError("PUBLIC_HOST_SOURCE_NOT_READY")
        # A caller-supplied commit label cannot validate a patched checkout.
        dirty=subprocess.run(["/usr/bin/git","-C",str(self.host_root),"diff","--quiet","HEAD"],
            timeout=10).returncode
        untracked=subprocess.run(["/usr/bin/git","-C",str(self.host_root),"ls-files","--others","--exclude-standard"],
            capture_output=True,timeout=10)
        if dirty or untracked.returncode or untracked.stdout.strip() or not (self.host_root/"hermes_cli/main.py").is_file():
            raise ValueError("PUBLIC_HOST_SOURCE_NOT_READY")

    def _confined(self,path):
        for part in (path,*path.parents):
            if part==self.home:break
            if part.is_symlink():raise ValueError("UNOWNED_PROFILE_SYMLINK")
        return path

    def install(self, *, plugin_source, skill_id, body, observer=True, observer_optional=False):
        if not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]*",skill_id):
            raise ValueError("INVALID_SKILL_ID")
        if not isinstance(body,bytes):
            raise TypeError("exact Skill bytes required")
        marker=self._confined(self.home/".semantier-public-owned")
        self.home.mkdir(parents=True,exist_ok=True)
        if not marker.exists() and any(self.home.iterdir()):
            raise ValueError("EMPTY_OWNED_PROFILE_REQUIRED")
        marker.touch(exist_ok=True)
        if observer:
            try:
                target=self._confined(self.home/"plugins"/"semantier-public-observer")
                if target.is_symlink():
                    raise ValueError("UNOWNED_PLUGIN_PATH")
                if not target.exists():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copytree(plugin_source,target)
                elif any(p.is_symlink() for p in target.rglob("*")):
                    raise ValueError("UNOWNED_PLUGIN_PATH")
            except OSError:
                if not observer_optional:
                    raise
                observer=False
        config_path=self._confined(self.home/"config.yaml")
        if config_path.is_symlink():
            raise ValueError("UNOWNED_CONFIG_PATH")
        config=yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
        config=config or {}
        entries=config.setdefault("plugins",{})
        enabled=entries.setdefault("enabled",[])
        if observer and "semantier-public-observer" not in enabled:enabled.append("semantier-public-observer")
        config_path.write_text(yaml.safe_dump(config,sort_keys=False))
        path=self._confined(self.home/"skills"/skill_id/"SKILL.md")
        if path.parent.is_symlink() or path.is_symlink():
            raise ValueError("UNOWNED_SKILL_PATH")
        if path.exists() and self.current_hash is None:
            raise ValueError("UNOWNED_SKILL_PATH")
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()!=self.current_hash:
            raise ValueError("STALE_EXPERIMENTAL_SKILL")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(body)
        self.skill_id=skill_id
        self.current_hash=hashlib.sha256(body).hexdigest()
        return self.current_hash

    def run(self, *, query, run_id, provider_args=(), timeout=60, cancel=None, reserve=None, on_event=None, workspace=None, isolate=False, launch_guard=None, readonly_overlays=(), pinned_route=None, http_limits=None):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}",run_id):
            raise ValueError("INVALID_RUN_ID")
        self.verify_host()
        if self.skill_id is None:raise ValueError("PACKAGE_NOT_INSTALLED")
        if any(flag not in {"--provider","--model","--base-url"} for flag in provider_args[::2]) or len(provider_args)%2:
            raise ValueError("UNSUPPORTED_PROVIDER_ARGUMENT")
        env={k:os.environ[k] for k in ("PATH","LANG","SYSTEMROOT") if k in os.environ}
        env.update(HOME=str(self.home),HERMES_HOME=str(self.home),PYTHONPATH=str(self.host_root),
                   PYTHONDONTWRITEBYTECODE="1",SEMANTIER_PUBLIC_RUN_ID=run_id,
                   SEMANTIER_PUBLIC_SKILL_ID=self.skill_id)
        if pinned_route is not None:
            if not isinstance(pinned_route, dict) or set(pinned_route) != {'provider', 'model', 'base_url'}:
                raise ValueError('EXPLICIT_CHILD_ROUTE_REQUIRED')
            endpoint = urlsplit(pinned_route['base_url'])
            if (pinned_route['provider'] != 'custom' or not pinned_route['model'] or
                endpoint.scheme != 'http' or endpoint.hostname != '127.0.0.1' or
                endpoint.username is not None or endpoint.password is not None or endpoint.query or endpoint.fragment or
                tuple(provider_args) != ('--provider', pinned_route['provider'], '--model', pinned_route['model'], '--base-url', pinned_route['base_url']) or
                not isolate or len(readonly_overlays) != 2):
                raise ValueError('EXPLICIT_LOOPBACK_CHILD_ROUTE_REQUIRED')
            config, guard = (Path(value).resolve(strict=True) for value in readonly_overlays)
            if config != self.home / 'config.yaml' or guard != self.home / 'managed-child-guard/sitecustomize.py':
                raise ValueError('EXPLICIT_CHILD_GUARD_REQUIRED')
            expected_guard = Path(__file__).with_name('managed_child_guard.py').read_bytes()
            if guard.is_symlink() or guard.read_bytes() != expected_guard:
                raise ValueError('CHILD_GUARD_CHANGED')
            env['PYTHONPATH'] = str(guard.parent) + os.pathsep + str(self.host_root)
            env['OPENAI_BASE_URL'] = pinned_route['base_url']
            env['SEMANTIER_MANAGED_EXPECTED_BASE_URL'] = pinned_route['base_url']
            env['SEMANTIER_MANAGED_EXPECTED_HOME'] = str(self.home)
            env['SEMANTIER_MANAGED_CONFIG_SHA256'] = hashlib.sha256(config.read_bytes()).hexdigest()
        if http_limits is not None:
            if http_limits != {'discovery':4,'model':4} or pinned_route is None:
                raise ValueError('INVALID_QUALIFICATION_HTTP_LIMITS')
            env['SEMANTIER_MANAGED_HTTP_LIMITS']=json.dumps(http_limits,sort_keys=True)
        command=[self.python,"-m","hermes_cli.main","chat",*provider_args,
            "--skills",self.skill_id,
            "--format","stream-json","--max-turns","2","--ignore-rules","--query",query]
        if workspace is not None:
            command += ["--in", str(Path(workspace).resolve())]
        if isolate:
            if workspace is None:
                raise ValueError("TASK_WORKSPACE_REQUIRED")
            from agents.task_process_boundary import isolated_command
            # Resolve the managed interpreter's base independently of its venv.
            executable = Path(self.python).resolve()
            with executable.open("rb") as binary:
                if binary.read(4) != b"\x7fELF":
                    raise ValueError("TASK_EXECUTOR_REQUIRES_NATIVE_PYTHON_BINARY")
            base = executable.parents[1]
            command = isolated_command(command, writable=(self.home, Path(workspace)),
                readonly=(self.host_root, base), cwd=Path(workspace), readonly_overlays=readonly_overlays)
        run=run_bounded_process(command,cwd=workspace or self.host_root,env=env,run_id=run_id,
            timeout=timeout,cancel=cancel,reserve=reserve,on_event=on_event,launch_guard=launch_guard)
        observations=[];gaps=list(run.gaps)
        session=run.result.get("session_id") if isinstance(run.result,dict) else None
        root=self.home/".semantier-public"
        try:
            files=list(root.glob("*.jsonl"))
            if len(files)!=1 or files[0].is_symlink():raise ValueError("collector_identity_ambiguous")
            path=files[0]
            if path.stat().st_size>1024*1024:raise ValueError("observer_output_truncated")
            rows=[json.loads(line) for line in path.read_bytes().splitlines()]
            if sum(row.get("hook")=="registration" for row in rows)!=1:
                raise ValueError("collector_registration_missing_or_ambiguous")
            observations,selection_gaps=collect_session_observations(rows,session,self._claimed_sessions)
            gaps.extend(selection_gaps)
            coverage_path=path.with_suffix(".coverage.json")
            if coverage_path.stat().st_size>16384:raise ValueError("invalid_coverage_size")
            coverage=json.loads(coverage_path.read_bytes())
            if coverage.get("dropped") or coverage.get("errors") or coverage.get("redacted") or coverage.get("unflushed", "unknown")!=0:
                gaps.append("observer_coverage_partial")
            gaps.append("collector_coverage_not_session_scoped")
        except (OSError,ValueError,UnicodeDecodeError,TypeError,AttributeError) as exc:
            gaps.append(type(exc).__name__+":observer_unavailable")
        if isinstance(session,str) and session:
            self._claimed_sessions.add(session)
        gaps.append("observer_history_completeness_unproven")
        return PublicRunResult(run.run_id,run.returncode,run.status,run.stdout,run.stderr,
            run.result,tuple(observations),tuple(sorted(set(gaps))),run.elapsed_seconds)

    @staticmethod
    def attach(*, service, context, run, profile_id):
        """Optional core sink; preserve the native result even if attachment is denied."""
        session_id=context.workspace_id+":connector:"+run.run_id
        try:
            row=service.record_public(context=context,raw_source=run.stdout,
                run_id=run.run_id,session_id=session_id,profile_id=profile_id,
                source_kind="process_result",source_version="hermes-public-process.v1",
                source_locator="connector:"+run.run_id,event_kind="response",sequence=0,
                payload={"result":run.result,"returncode":run.returncode,
                         "collection_report":{"surface":"terminal_process_output_only", "process_status":run.status, "history_complete":False}},
                gaps=run.gaps,captured_at=datetime.now(timezone.utc))
            rows=[row]
            for index,event in enumerate(run.observations,1):
                rows.append(service.record_public(context=context,
                    raw_source=json.dumps(event,sort_keys=True).encode(),run_id=run.run_id,
                    session_id=session_id,profile_id=profile_id,source_kind="public_hook",
                    source_version="hermes-public-observer.v1-session-bound",source_locator="connector:"+run.run_id+":"+str(index),
                    event_kind="tool" if event.get("hook")=="post_tool_call" else "lifecycle",
                    sequence=index,payload={**event, "association":{"native_session_id":run.result.get("session_id") if run.result else None, "collector_run_id":event.get("run_id"), "basis":"exact_native_session"}, **({"action":normalize_public_action(event)} if normalize_public_action(event) is not None else {})},gaps=tuple(event.get("gaps",())),
                    stage="TOOL_REPORTED" if event.get("hook")=="post_tool_call" else "OBSERVED",
                    captured_at=datetime.now(timezone.utc)))
            return {"status":"retained","pins":[r.pin().model_dump(mode="json") for r in rows]}
        except Exception as exc:
            return {"status":"unavailable","reason":type(exc).__name__,"native_result_preserved":True}
