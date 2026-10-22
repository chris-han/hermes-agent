"""Ordered MVL release-gate runner for closure manifests."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from eos.artifact_hashing import canonical_json, content_hash
from semantier.knowledge_feature_packs.source_elements import source_elements
from semantier.knowledge_feature_packs.semantica_adapter import SEMANTICA_COMMIT
from .graph_schema_discovery_service import GraphSchemaDiscoveryService
from .graph_schema_llm_evidence_runner import (
    GeneratorFactory,
    GraphSchemaLlmEvidenceRunner,
    SKILL_MAJOR_VERSION,
    resolve_runtime_llm_provider_model,
)
from .mvl_closure_preflight import MvlClosurePreflight
from .mvl_e2e_flag_criteria_service import MvlE2eFlagCriteriaService, MvlE2eFlagCriteriaWrite
from .mvl_reality_handoff_service import MvlRealityHandoffService, MvlRealityHandoffWrite


CommandStatus = Literal["passed", "failed", "skipped"]
CommandExecutor = Callable[[Sequence[str], Path], "ExecutedCommand"]


BACKEND_FOCUSED_COMMAND = (
    "uv",
    "run",
    "pytest",
    "-q",
    "-c",
    "pytest.ini",
    "tests/test_graph_schema_llm_evidence_runner.py",
    "tests/test_mvl_closure_preflight.py",
    "hermes-agent/tests/agent/test_semantica_memory_provider.py",
    "tests/test_fresh_s1_discovery.py",
    "tests/test_graph_schema_skill_learning.py",
    "tests/test_graph_schema_learning_loop_backend.py",
    "tests/test_canonical_list_matcher.py",
    "tests/test_runtime_detection_snapshot.py",
    "tests/test_semantic_feedback_event_store.py",
    "tests/test_mvl_selection_and_discovery_contracts.py",
    "tests/test_mvl_closure_evidence_service.py",
    "tests/test_tender_document_review_routes.py",
    "tests/test_phase1_001_artifact_repository.py",
    "tests/test_phase1_001_native_builder.py",
    "tests/test_semantica_convergence_guards.py",
)
FRONTEND_FOCUSED_COMMAND = (
    "pnpm",
    "test",
    "src/screens/source-evidence/source-evidence-viewer.test.tsx",
    "src/server/tender-document-review.test.ts",
    "src/server/source-evidence-viewer-config.test.ts",
)
FRONTEND_TYPECHECK_COMMAND = ("pnpm", "typecheck")
PLAN_SPEC_SYNC_COMMAND = ("python", "scripts/check_plan_spec_sync.py", "check")
DOC_SEMANTIC_ALIGNMENT_COMMAND = ("python", "scripts/check_doc_semantic_alignment.py", "check")
GRAPH_SCHEMA_E2E_COMMAND = (
    "pnpm",
    "exec",
    "playwright",
    "test",
    "tests/e2e/graph-schema-learning-loop.spec.ts",
    "--reporter=line",
)
DOCX_PDF_VIEWER_E2E_COMMAND = (
    "pnpm",
    "exec",
    "playwright",
    "test",
    "tests/e2e/runtime-document-inference-review.spec.ts",
    "--reporter=line",
)
FIXED_LLM_SOURCE = Path("docs/招投标法规/POC测试敏感词汇总.docx")


@dataclass(frozen=True)
class ExecutedCommand:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class MvlReleaseGateRun:
    command_results: dict[str, CommandStatus]
    command_evidence_refs: dict[str, str]
    llm_evidence: dict[str, Any]
    manifest: dict[str, Any]
    manifest_path: Path
    e2e_flag_criteria_record: dict[str, Any]
    e2e_flag_criteria_artifact_ref: str
    reality_handoff: dict[str, Any]
    reality_handoff_path: Path
    reality_handoff_artifact_ref: str


class MvlReleaseGateRunner:
    """Run §18 checks in order and emit a closure preflight manifest."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        workspace_id: str,
        environ: Mapping[str, str] | None = None,
        executor: CommandExecutor | None = None,
        llm_generator_factory: GeneratorFactory | None = None,
    ) -> None:
        if environ is not None and not isinstance(environ, Mapping):
            raise ValueError("release gate environ must be a mapping")
        self._require_safe_artifact_id(workspace_id, label="workspace_id")
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_id = workspace_id
        self.environ = environ if environ is not None else os.environ
        self.executor = executor or self._subprocess_executor
        self.llm_generator_factory = llm_generator_factory

    def run(
        self,
        *,
        manifest_id: str = "mvl-closure-release-gate",
        llm_provider: str | None = None,
        llm_model: str | None = None,
        llm_source: str | Path = FIXED_LLM_SOURCE,
        llm_evidence_id: str | None = None,
        release_candidate_id: str | None = None,
        viewer_related_flake_count: int = 0,
    ) -> MvlReleaseGateRun:
        if (
            not isinstance(viewer_related_flake_count, int)
            or isinstance(viewer_related_flake_count, bool)
            or viewer_related_flake_count < 0
        ):
            raise ValueError("viewer_related_flake_count must be non-negative")
        self._require_safe_artifact_id(manifest_id, label="manifest_id")
        if release_candidate_id is not None:
            self._require_safe_artifact_id(
                release_candidate_id,
                label="release_candidate_id",
            )
        llm_evidence = self._capture_llm_evidence(
            provider=llm_provider,
            model=llm_model,
            source=llm_source,
            manifest_id=manifest_id,
            evidence_id=llm_evidence_id,
        )
        command_results: dict[str, CommandStatus] = {}
        command_evidence_refs: dict[str, str] = {}
        for key, command, cwd in self._ordered_commands():
            executed = self.executor(command, cwd)
            status = self._status_for(executed)
            command_results[key] = status
            command_evidence_refs[key] = self._write_command_evidence(
                manifest_id=manifest_id,
                command_key=key,
                command=command,
                cwd=cwd,
                executed=executed,
                status=status,
            )
        flag_service = MvlE2eFlagCriteriaService(
            self.workspace_root,
            workspace_id=self.workspace_id,
        )
        docx_pdf_viewer_status = self._docx_pdf_viewer_status_from_inputs(
            command_results=command_results,
            environ=self.environ,
        )
        current_flag_record = flag_service.build_record(
            run_id=manifest_id,
            release_candidate_id=release_candidate_id or manifest_id,
            docx_pdf_viewer_e2e_status=docx_pdf_viewer_status,
            viewer_related_flake_count=viewer_related_flake_count,
            manifest_ref=f"pending:mvl-release-gate/{manifest_id}",
        )
        e2e_flag_criteria_report = flag_service.evaluate(
            [*flag_service.load_records(), current_flag_record]
        )
        written = MvlClosurePreflight(
            self.workspace_root,
            workspace_id=self.workspace_id,
            environ=self.environ,
        ).write(
            manifest_id=manifest_id,
            command_results=command_results,
            command_evidence_refs=command_evidence_refs,
            e2e_flag_criteria_report=e2e_flag_criteria_report,
            llm_evidence_ref=self._current_llm_evidence_ref(llm_evidence),
        )
        flag_record = self._record_e2e_flag_criteria(
            flag_service=flag_service,
            run_id=manifest_id,
            release_candidate_id=release_candidate_id or manifest_id,
            docx_pdf_viewer_status=docx_pdf_viewer_status,
            manifest_path=written.path,
            viewer_related_flake_count=viewer_related_flake_count,
        )
        reality_handoff = self._write_reality_handoff(
            manifest=written.manifest,
            manifest_path=written.path,
        )
        return MvlReleaseGateRun(
            command_results=command_results,
            command_evidence_refs=command_evidence_refs,
            llm_evidence=llm_evidence,
            manifest=written.manifest,
            manifest_path=written.path,
            e2e_flag_criteria_record=flag_record.record,
            e2e_flag_criteria_artifact_ref=flag_record.artifact_ref,
            reality_handoff=reality_handoff.manifest,
            reality_handoff_path=reality_handoff.path,
            reality_handoff_artifact_ref=reality_handoff.artifact_ref,
        )

    def _write_command_evidence(
        self,
        *,
        manifest_id: str,
        command_key: str,
        command: Sequence[str],
        cwd: Path,
        executed: ExecutedCommand,
        status: CommandStatus,
    ) -> str:
        self._require_safe_artifact_id(manifest_id, label="manifest_id")
        self._require_safe_artifact_id(command_key, label="command_key")
        self._require_command_payload(command)
        self._require_executed_command(executed)
        self._require_command_status(status)
        cwd_ref = self._workspace_relative_cwd(cwd)
        artifact_root = (
            self.workspace_root
            / "workspaces"
            / self.workspace_id
            / "artifacts"
            / "mvl-closure"
            / "release-command-results"
        )
        body = {
            "schema_version": "mvl_release_command_result.v1",
            "manifest_id": manifest_id,
            "workspace_id": self.workspace_id,
            "command_key": command_key,
            "command": list(command),
            "cwd": cwd_ref,
            "returncode": executed.returncode,
            "status": status,
            "stdout": executed.stdout,
            "stderr": executed.stderr,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        body["content_hash"] = f"sha256:{content_hash(canonical_json(body))}"
        artifact_root.mkdir(parents=True, exist_ok=True)
        digest = str(body["content_hash"]).removeprefix("sha256:")[:16]
        path = artifact_root / f"{manifest_id}-{command_key}-{digest}.json"
        path.write_text(
            json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return path.relative_to(
            self.workspace_root / "workspaces" / self.workspace_id
        ).as_posix()

    @staticmethod
    def _require_command_payload(command: Sequence[str]) -> None:
        if not command or any(not isinstance(arg, str) or not arg.strip() for arg in command):
            raise ValueError("release command evidence command is invalid")

    @staticmethod
    def _require_executed_command(executed: ExecutedCommand) -> None:
        if not isinstance(executed.returncode, int) or isinstance(executed.returncode, bool):
            raise ValueError("release command evidence returncode is invalid")
        if not isinstance(executed.stdout, str) or not isinstance(executed.stderr, str):
            raise ValueError("release command evidence output is invalid")

    @staticmethod
    def _require_command_status(status: CommandStatus) -> None:
        if status not in {"passed", "failed", "skipped"}:
            raise ValueError("release command evidence status is invalid")

    def _workspace_relative_cwd(self, cwd: Path) -> str:
        resolved_cwd = cwd.resolve()
        if not resolved_cwd.is_relative_to(self.workspace_root):
            raise ValueError("release command cwd must be inside workspace_root")
        return resolved_cwd.relative_to(self.workspace_root).as_posix()

    @staticmethod
    def _require_safe_artifact_id(value: str, *, label: str) -> None:
        if not isinstance(value, str):
            raise ValueError(f"release gate {label} must be a safe artifact id")
        path = Path(value)
        if (
            not value.strip()
            or path.is_absolute()
            or "/" in value
            or "\\" in value
            or ".." in path.parts
        ):
            raise ValueError(f"release gate {label} must be a safe artifact id")

    def _capture_llm_evidence(
        self,
        *,
        provider: str | None,
        model: str | None,
        source: str | Path,
        manifest_id: str,
        evidence_id: str | None = None,
    ) -> dict[str, Any]:
        self._require_safe_artifact_id(manifest_id, label="manifest_id")
        if evidence_id is not None:
            self._require_safe_artifact_id(evidence_id, label="llm_evidence_id")
        source_path = self._workspace_source_path(source)
        source_text, source_hash = self._source_text(source_path)
        source_ref = f"repo://{source_path.relative_to(self.workspace_root).as_posix()}"
        service = GraphSchemaDiscoveryService(self.workspace_root, workspace_id=self.workspace_id)
        run = service.start_run(
            lineage_id=manifest_id,
            experiment_id=f"{manifest_id}-llm",
            source_ref=source_ref,
            source_hash=source_hash,
            created_by="semantier-mvl-release-gate@system.invalid",
        )
        resolved = (
            (provider, model)
            if provider and model
            else resolve_runtime_llm_provider_model(
                environ=self.environ,
                workspace_root=self.workspace_root,
            )
        )
        if resolved is None:
            return service.store.record_llm_proposal_evidence(
                run_id=run["run_id"],
                semantica_commit=SEMANTICA_COMMIT,
                skill_major_version=SKILL_MAJOR_VERSION,
                provider_contract_version="semantica-llm-ontology@1",
                provider="runtime_config",
                model="unresolved",
                status="unavailable",
                evidence_id=evidence_id,
                diagnostics={
                    "reason": "runtime LLM provider/model is not configured",
                },
            )
        provider, model = resolved
        return GraphSchemaLlmEvidenceRunner(
            self.workspace_root,
            workspace_id=self.workspace_id,
            store=service.store,
            environ=self.environ,
            generator_factory=self.llm_generator_factory,
        ).run(
            run_id=run["run_id"],
            provider=provider,
            model=model,
            source_text=source_text,
            semantica_commit=SEMANTICA_COMMIT,
            evidence_id=evidence_id,
        )

    def _workspace_source_path(self, source: str | Path) -> Path:
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = self.workspace_root / source_path
        resolved_source_path = source_path.resolve()
        if not resolved_source_path.is_relative_to(self.workspace_root):
            raise ValueError("release gate LLM source must be inside workspace_root")
        return resolved_source_path

    @staticmethod
    def _current_llm_evidence_ref(llm_evidence: dict[str, Any]) -> str | None:
        if llm_evidence.get("status") != "passed":
            return None
        return f"llm_proposal_evidence:{llm_evidence['evidence_id']}"

    def _ordered_commands(self) -> tuple[tuple[str, tuple[str, ...], Path], ...]:
        workspace = self.workspace_root / "hermes-workspace"
        return (
            ("backend_focused_regression", BACKEND_FOCUSED_COMMAND, self.workspace_root),
            ("frontend_focused_regression", FRONTEND_FOCUSED_COMMAND, workspace),
            ("frontend_typecheck", FRONTEND_TYPECHECK_COMMAND, workspace),
            ("plan_spec_sync", PLAN_SPEC_SYNC_COMMAND, self.workspace_root),
            ("doc_semantic_alignment", DOC_SEMANTIC_ALIGNMENT_COMMAND, self.workspace_root),
            ("graph_schema_learning_loop_e2e", GRAPH_SCHEMA_E2E_COMMAND, workspace),
            ("docx_pdf_viewer_e2e", DOCX_PDF_VIEWER_E2E_COMMAND, workspace),
        )

    def _record_e2e_flag_criteria(
        self,
        *,
        flag_service: MvlE2eFlagCriteriaService,
        run_id: str,
        release_candidate_id: str,
        docx_pdf_viewer_status: str,
        manifest_path: Path,
        viewer_related_flake_count: int,
    ) -> MvlE2eFlagCriteriaWrite:
        manifest_ref = manifest_path.relative_to(
            self.workspace_root / "workspaces" / self.workspace_id
        ).as_posix()
        return flag_service.write_record(
            flag_service.build_record(
                run_id=run_id,
                release_candidate_id=release_candidate_id,
                docx_pdf_viewer_e2e_status=docx_pdf_viewer_status,
                viewer_related_flake_count=viewer_related_flake_count,
                manifest_ref=manifest_ref,
            )
        )

    def _write_reality_handoff(
        self,
        *,
        manifest: dict[str, Any],
        manifest_path: Path,
    ) -> MvlRealityHandoffWrite:
        manifest_ref = manifest_path.relative_to(
            self.workspace_root / "workspaces" / self.workspace_id
        ).as_posix()
        return MvlRealityHandoffService(
            self.workspace_root,
            workspace_id=self.workspace_id,
        ).write_from_closure(
            closure_manifest=manifest,
            closure_manifest_ref=manifest_ref,
        )

    @staticmethod
    def _docx_pdf_viewer_status_from_inputs(
        *,
        command_results: dict[str, CommandStatus],
        environ: Mapping[str, str],
    ) -> str:
        status = command_results.get("docx_pdf_viewer_e2e")
        if status == "passed":
            return "passed"
        if status == "failed":
            return "failed"
        return "skipped"

    @staticmethod
    def _status_for(executed: ExecutedCommand) -> CommandStatus:
        MvlReleaseGateRunner._require_executed_command(executed)
        output = f"{executed.stdout}\n{executed.stderr}".lower()
        if executed.returncode != 0:
            return "failed"
        if MvlReleaseGateRunner._output_count_positive(output, "failed"):
            return "failed"
        if MvlReleaseGateRunner._output_count_positive(output, "skipped"):
            return "skipped"
        return "passed"

    @staticmethod
    def _output_count_positive(output: str, word: str) -> bool:
        pattern = rf"(?<!\d)([1-9]\d*)\s+{re.escape(word)}\b"
        return re.search(pattern, output) is not None

    @staticmethod
    def _source_text(path: Path) -> tuple[str, str]:
        if path.suffix.lower() == ".docx":
            parsed = source_elements(path)
            text = "\n".join(
                str(element["text"])
                for element in parsed["elements"]
                if element.get("text")
            )
            return text, parsed["document_hash"]
        data = path.read_bytes()
        return data.decode("utf-8"), f"sha256:{sha256(data).hexdigest()}"

    @staticmethod
    def _subprocess_executor(command: Sequence[str], cwd: Path) -> ExecutedCommand:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            text=True,
            capture_output=True,
        )
        return ExecutedCommand(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
