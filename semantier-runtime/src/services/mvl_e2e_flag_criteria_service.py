"""Release-candidate evidence for removing transitional MVL E2E flags."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from eos.artifact_hashing import canonical_json, content_hash
from services.mvl_closure_evidence_service import PLAN_REF


MVL_E2E_FLAG_CRITERIA_RECORD_SCHEMA_VERSION = "mvl_e2e_flag_criteria_record.v1"
MVL_E2E_FLAG_CRITERIA_REPORT_SCHEMA_VERSION = "mvl_e2e_flag_criteria_report.v1"
MVL_E2E_FLAG_POLICY_REPORT_SCHEMA_VERSION = "mvl_e2e_flag_policy_report.v1"
DOCUMENT_INFERENCE_FLAG = "DOCUMENT_INFERENCE_REVIEW_E2E"
GRAPH_SCHEMA_LEARNING_LOOP_FLAG = "GRAPH_SCHEMA_LEARNING_LOOP_E2E"
REQUIRED_CONSECUTIVE_PASSES = 3
WORKSPACE_CLOSURE_MANIFEST_PREFIX = "artifacts/mvl-closure/evidence-manifests/"
PENDING_RELEASE_GATE_PREFIX = "pending:mvl-release-gate/"
RunStatus = Literal["passed", "failed", "skipped", "blocked"]


@dataclass(frozen=True)
class MvlE2eFlagCriteriaWrite:
    record: dict[str, Any]
    path: Path
    artifact_ref: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MvlE2eFlagCriteriaService:
    """Write release-candidate run records and evaluate flag-removal readiness."""

    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        if not workspace_id:
            raise ValueError("workspace_id is required")
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_id = workspace_id
        self.artifact_root = (
            self.workspace_root
            / "workspaces"
            / workspace_id
            / "artifacts"
            / "mvl-closure"
            / "e2e-flag-criteria"
        )

    def build_record(
        self,
        *,
        run_id: str,
        release_candidate_id: str,
        docx_pdf_viewer_e2e_status: RunStatus,
        manifest_ref: str,
        viewer_related_flake_count: int = 0,
        recorded_at: str | None = None,
    ) -> dict[str, Any]:
        self._require(
            run_id=run_id,
            release_candidate_id=release_candidate_id,
            docx_pdf_viewer_e2e_status=docx_pdf_viewer_e2e_status,
            manifest_ref=manifest_ref,
        )
        self._require_safe_artifact_id(run_id, label="run_id")
        self._require_safe_artifact_id(
            release_candidate_id,
            label="release_candidate_id",
        )
        if docx_pdf_viewer_e2e_status not in {"passed", "failed", "skipped", "blocked"}:
            raise ValueError("docx_pdf_viewer_e2e_status is invalid")
        if (
            not isinstance(viewer_related_flake_count, int)
            or isinstance(viewer_related_flake_count, bool)
            or viewer_related_flake_count < 0
        ):
            raise ValueError("viewer_related_flake_count is invalid")
        self._require_manifest_ref(manifest_ref=manifest_ref, run_id=run_id)
        recorded_at = recorded_at or _utc_now()
        self._require_aware_timestamp(recorded_at)
        record = {
            "schema_version": MVL_E2E_FLAG_CRITERIA_RECORD_SCHEMA_VERSION,
            "workspace_id": self.workspace_id,
            "plan_ref": PLAN_REF,
            "flag": DOCUMENT_INFERENCE_FLAG,
            "run_id": run_id,
            "release_candidate_id": release_candidate_id,
            "docx_pdf_viewer_e2e_status": docx_pdf_viewer_e2e_status,
            "viewer_related_flake_count": viewer_related_flake_count,
            "manifest_ref": manifest_ref,
            "recorded_at": recorded_at,
        }
        record["content_hash"] = f"sha256:{content_hash(canonical_json(record))}"
        return record

    def write_record(self, record: dict[str, Any]) -> MvlE2eFlagCriteriaWrite:
        self.verify_record(record, allow_pending_manifest_ref=False)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        digest = str(record["content_hash"]).removeprefix("sha256:")[:16]
        artifact_ref = f"artifacts/mvl-closure/e2e-flag-criteria/{record['run_id']}-{digest}.json"
        path = self.workspace_root / "workspaces" / self.workspace_id / artifact_ref
        path.write_text(
            json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return MvlE2eFlagCriteriaWrite(record=record, path=path, artifact_ref=artifact_ref)

    def load_records(self) -> list[dict[str, Any]]:
        if not self.artifact_root.exists():
            return []
        records = []
        for path in sorted(self.artifact_root.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            self.verify_record(record, allow_pending_manifest_ref=False)
            records.append(record)
        return sorted(records, key=lambda item: (item["recorded_at"], item["run_id"]))

    def evaluate(self, records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        records = self.load_records() if records is None else records
        if not isinstance(records, list):
            raise ValueError("E2E flag criteria records must be a list")
        for record in records:
            self.verify_record(record)
        ordered = sorted(records, key=lambda item: (item["recorded_at"], item["run_id"]))
        self._require_unique_run_ids(ordered)
        self._require_unique_release_candidate_ids(ordered)
        consecutive = 0
        for record in reversed(ordered):
            if (
                record["docx_pdf_viewer_e2e_status"] == "passed"
                and record["viewer_related_flake_count"] == 0
            ):
                consecutive += 1
            else:
                break
        eligible = consecutive >= REQUIRED_CONSECUTIVE_PASSES
        report = {
            "schema_version": MVL_E2E_FLAG_CRITERIA_REPORT_SCHEMA_VERSION,
            "workspace_id": self.workspace_id,
            "plan_ref": PLAN_REF,
            "flag": DOCUMENT_INFERENCE_FLAG,
            "required_consecutive_passes": REQUIRED_CONSECUTIVE_PASSES,
            "consecutive_pass_count": consecutive,
            "eligible_to_remove_flag": eligible,
            "recommendation": "remove_transitional_flag" if eligible else "keep_transitional_flag",
            "record_refs": [record["run_id"] for record in ordered],
            "release_candidate_refs": [
                record["release_candidate_id"] for record in ordered
            ],
            "record_hashes": [record["content_hash"] for record in ordered],
            "manifest_refs": [record["manifest_ref"] for record in ordered],
        }
        report["content_hash"] = f"sha256:{content_hash(canonical_json(report))}"
        return report

    @staticmethod
    def _require_unique_run_ids(records: list[dict[str, Any]]) -> None:
        seen: set[str] = set()
        duplicates: list[str] = []
        for record in records:
            run_id = str(record["run_id"])
            if run_id in seen:
                duplicates.append(run_id)
            seen.add(run_id)
        if duplicates:
            raise ValueError(
                f"E2E flag criteria records contain duplicate run_id values: {duplicates}"
            )

    @staticmethod
    def _require_unique_release_candidate_ids(records: list[dict[str, Any]]) -> None:
        seen: set[str] = set()
        duplicates: list[str] = []
        for record in records:
            release_candidate_id = str(record["release_candidate_id"])
            if release_candidate_id in seen:
                duplicates.append(release_candidate_id)
            seen.add(release_candidate_id)
        if duplicates:
            raise ValueError(
                "E2E flag criteria records contain duplicate release_candidate_id "
                f"values: {duplicates}"
            )

    def evaluate_release_flag_policy(
        self,
        *,
        report: dict[str, Any] | None = None,
        env: dict[str, str | None] | None = None,
    ) -> dict[str, Any]:
        report = self.evaluate() if report is None else report
        self.verify_criteria_report(report)
        if env is None:
            env = {}
        if not isinstance(env, Mapping):
            raise ValueError("E2E flag release policy env must be a mapping")
        eligible = report.get("eligible_to_remove_flag") is True
        document_flag_enabled, document_flag_invalid = self._release_flag_state(
            env.get(DOCUMENT_INFERENCE_FLAG),
            flag=DOCUMENT_INFERENCE_FLAG,
        )
        graph_loop_flag_enabled, graph_loop_flag_invalid = self._release_flag_state(
            env.get(GRAPH_SCHEMA_LEARNING_LOOP_FLAG),
            flag=GRAPH_SCHEMA_LEARNING_LOOP_FLAG,
        )
        document_flag_passes = (
            eligible and document_flag_enabled and document_flag_invalid is None
        )
        graph_loop_flag_passes = (
            graph_loop_flag_enabled and graph_loop_flag_invalid is None
        )
        document_flag_diagnostic = None
        if not document_flag_passes:
            if document_flag_invalid is not None:
                document_flag_diagnostic = document_flag_invalid
            elif eligible:
                document_flag_diagnostic = (
                    "DOCUMENT_INFERENCE_REVIEW_E2E criteria are met "
                    "but the release flag is not enabled"
                )
            else:
                document_flag_diagnostic = (
                    "DOCUMENT_INFERENCE_REVIEW_E2E criteria are not met"
                )
        graph_loop_flag_diagnostic = None
        if not graph_loop_flag_passes:
            graph_loop_flag_diagnostic = graph_loop_flag_invalid or (
                "GRAPH_SCHEMA_LEARNING_LOOP_E2E is not enabled for the release job"
            )
        actions = [
            {
                "flag": DOCUMENT_INFERENCE_FLAG,
                "status": "pass" if document_flag_passes else "incomplete",
                "current_enabled": document_flag_enabled,
                "criteria_eligible": eligible,
                "required_action": (
                    "remove_flag_or_keep_enabled"
                    if eligible
                    else "keep_transitional_flag_and_collect_clean_release_passes"
                ),
                "diagnostic": document_flag_diagnostic,
            },
            {
                "flag": GRAPH_SCHEMA_LEARNING_LOOP_FLAG,
                "status": "pass" if graph_loop_flag_passes else "incomplete",
                "current_enabled": graph_loop_flag_enabled,
                "criteria_eligible": True,
                "required_action": "enable_in_release_job_until_e2e_is_permanently_default_on",
                "diagnostic": graph_loop_flag_diagnostic,
            },
        ]
        policy = {
            "schema_version": MVL_E2E_FLAG_POLICY_REPORT_SCHEMA_VERSION,
            "workspace_id": self.workspace_id,
            "plan_ref": PLAN_REF,
            "overall_status": (
                "pass"
                if all(action["status"] == "pass" for action in actions)
                else "incomplete"
            ),
            "criteria_report_hash": report.get("content_hash"),
            "actions": actions,
        }
        policy["content_hash"] = f"sha256:{content_hash(canonical_json(policy))}"
        return policy

    @staticmethod
    def _release_flag_state(
        value: Any,
        *,
        flag: str,
    ) -> tuple[bool, str | None]:
        if value is not None and not isinstance(value, str):
            return (
                False,
                f"{flag} has invalid release flag value {value!r}; expected '1' "
                "when enabled or empty/0 when disabled",
            )
        normalized = (value or "").strip()
        if normalized in {"", "0"}:
            return False, None
        if normalized == "1":
            return True, None
        return (
            False,
            f"{flag} has invalid release flag value {normalized!r}; expected '1' "
            "when enabled or empty/0 when disabled",
        )

    def verify_criteria_report(self, report: dict[str, Any]) -> None:
        if not isinstance(report, dict):
            raise ValueError("E2E flag criteria report must be an object")
        self._require_criteria_report_fields(report)
        if report.get("schema_version") != MVL_E2E_FLAG_CRITERIA_REPORT_SCHEMA_VERSION:
            raise ValueError("E2E flag criteria report schema_version is invalid")
        if report.get("workspace_id") != self.workspace_id:
            raise ValueError("E2E flag criteria report workspace_id mismatch")
        if report.get("plan_ref") != PLAN_REF:
            raise ValueError("E2E flag criteria report plan_ref mismatch")
        if report.get("flag") != DOCUMENT_INFERENCE_FLAG:
            raise ValueError("E2E flag criteria report flag mismatch")
        self._require_criteria_report_values(report)
        hashable = {key: value for key, value in report.items() if key != "content_hash"}
        expected_hash = f"sha256:{content_hash(canonical_json(hashable))}"
        if report.get("content_hash") != expected_hash:
            raise ValueError("E2E flag criteria report content_hash mismatch")

    @staticmethod
    def _require_criteria_report_fields(report: dict[str, Any]) -> None:
        expected = {
            "schema_version",
            "workspace_id",
            "plan_ref",
            "flag",
            "required_consecutive_passes",
            "consecutive_pass_count",
            "eligible_to_remove_flag",
            "recommendation",
            "record_refs",
            "release_candidate_refs",
            "record_hashes",
            "manifest_refs",
            "content_hash",
        }
        unexpected = sorted(set(report) - expected)
        if unexpected:
            raise ValueError(
                f"E2E flag criteria report has unexpected fields: {unexpected}"
            )

    @classmethod
    def _require_criteria_report_values(cls, report: dict[str, Any]) -> None:
        record_refs = report.get("record_refs")
        release_candidate_refs = report.get("release_candidate_refs")
        record_hashes = report.get("record_hashes")
        manifest_refs = report.get("manifest_refs")
        consecutive_pass_count = report.get("consecutive_pass_count")
        required_consecutive_passes = report.get("required_consecutive_passes")
        eligible = report.get("eligible_to_remove_flag")
        recommendation = report.get("recommendation")
        if not isinstance(record_refs, list):
            raise ValueError("E2E flag criteria report record_refs are invalid")
        if not isinstance(release_candidate_refs, list):
            raise ValueError(
                "E2E flag criteria report release_candidate_refs are required"
            )
        if not isinstance(record_hashes, list):
            raise ValueError("E2E flag criteria report record_hashes are required")
        if not isinstance(manifest_refs, list):
            raise ValueError("E2E flag criteria report manifest_refs are required")
        if (
            not isinstance(consecutive_pass_count, int)
            or isinstance(consecutive_pass_count, bool)
            or consecutive_pass_count < 0
        ):
            raise ValueError(
                "E2E flag criteria report consecutive_pass_count is invalid"
            )
        if (
            not isinstance(required_consecutive_passes, int)
            or isinstance(required_consecutive_passes, bool)
            or required_consecutive_passes < 1
        ):
            raise ValueError(
                "E2E flag criteria report required_consecutive_passes is invalid"
            )
        if not isinstance(eligible, bool):
            raise ValueError(
                "E2E flag criteria report eligible_to_remove_flag is invalid"
            )
        if (
            len(record_refs) != len(release_candidate_refs)
            or len(record_refs) != len(record_hashes)
            or len(record_refs) != len(manifest_refs)
        ):
            raise ValueError("E2E flag criteria report record evidence lengths mismatch")
        if any(
            not isinstance(value, str) or not cls._safe_artifact_id(value)
            for value in record_refs
        ):
            raise ValueError("E2E flag criteria report record_refs are invalid")
        if len(set(record_refs)) != len(record_refs):
            raise ValueError("E2E flag criteria report record_refs contain duplicates")
        if any(
            not isinstance(value, str) or not cls._safe_artifact_id(value)
            for value in release_candidate_refs
        ):
            raise ValueError(
                "E2E flag criteria report release_candidate_refs are invalid"
            )
        if len(set(release_candidate_refs)) != len(release_candidate_refs):
            raise ValueError(
                "E2E flag criteria report release_candidate_refs contain duplicates"
            )
        if any(
            not isinstance(value, str) or not cls._valid_sha256_ref(value)
            for value in record_hashes
        ):
            raise ValueError("E2E flag criteria report record_hashes are invalid")
        if len(set(record_hashes)) != len(record_hashes):
            raise ValueError(
                "E2E flag criteria report record_hashes contain duplicates"
            )
        if any(
            not isinstance(value, str)
            or not cls._valid_criteria_report_manifest_ref(value)
            for value in manifest_refs
        ):
            raise ValueError("E2E flag criteria report manifest_refs are invalid")
        if len(set(manifest_refs)) != len(manifest_refs):
            raise ValueError(
                "E2E flag criteria report manifest_refs contain duplicates"
            )
        for record_ref, manifest_ref in zip(record_refs, manifest_refs, strict=True):
            if manifest_ref.startswith(PENDING_RELEASE_GATE_PREFIX) and (
                manifest_ref != f"{PENDING_RELEASE_GATE_PREFIX}{record_ref}"
            ):
                raise ValueError(
                    "E2E flag criteria report pending manifest_refs must match "
                    "record_refs"
                )
        expected_eligible = consecutive_pass_count >= required_consecutive_passes
        if eligible is not expected_eligible:
            raise ValueError("E2E flag criteria report eligibility/count mismatch")
        expected_recommendation = (
            "remove_transitional_flag" if eligible else "keep_transitional_flag"
        )
        if recommendation != expected_recommendation:
            raise ValueError("E2E flag criteria report recommendation mismatch")

    @staticmethod
    def _valid_criteria_report_manifest_ref(value: str) -> bool:
        if value.startswith(PENDING_RELEASE_GATE_PREFIX):
            return MvlE2eFlagCriteriaService._valid_pending_manifest_ref(value)
        if not value.startswith(WORKSPACE_CLOSURE_MANIFEST_PREFIX):
            return False
        suffix = value.removeprefix(WORKSPACE_CLOSURE_MANIFEST_PREFIX)
        if not suffix.strip() or not suffix.endswith(".json"):
            return False
        path = Path(value)
        return not path.is_absolute() and "\\" not in value and ".." not in path.parts

    @staticmethod
    def _valid_sha256_ref(value: str) -> bool:
        digest = value.removeprefix("sha256:")
        return (
            value.startswith("sha256:")
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest)
        )

    @staticmethod
    def _safe_artifact_id(value: str) -> bool:
        path = Path(value)
        return (
            bool(value.strip())
            and not path.is_absolute()
            and "/" not in value
            and "\\" not in value
            and ".." not in path.parts
        )

    def verify_record(
        self,
        record: dict[str, Any],
        *,
        allow_pending_manifest_ref: bool = True,
    ) -> None:
        if not isinstance(record, dict):
            raise ValueError("E2E flag criteria record must be an object")
        self._require_record_fields(record)
        if record.get("schema_version") != MVL_E2E_FLAG_CRITERIA_RECORD_SCHEMA_VERSION:
            raise ValueError("E2E flag criteria record schema_version is invalid")
        if record.get("workspace_id") != self.workspace_id:
            raise ValueError("E2E flag criteria record workspace_id mismatch")
        if record.get("plan_ref") != PLAN_REF:
            raise ValueError("E2E flag criteria record plan_ref mismatch")
        if record.get("flag") != DOCUMENT_INFERENCE_FLAG:
            raise ValueError("E2E flag criteria record flag mismatch")
        run_id = record.get("run_id")
        release_candidate_id = record.get("release_candidate_id")
        docx_pdf_viewer_e2e_status = record.get("docx_pdf_viewer_e2e_status")
        manifest_ref = record.get("manifest_ref")
        recorded_at = record.get("recorded_at")
        self._require_string_fields(
            run_id=run_id,
            release_candidate_id=release_candidate_id,
            docx_pdf_viewer_e2e_status=docx_pdf_viewer_e2e_status,
            manifest_ref=manifest_ref,
            recorded_at=recorded_at,
        )
        self._require_safe_artifact_id(run_id, label="run_id")
        self._require_safe_artifact_id(
            release_candidate_id,
            label="release_candidate_id",
        )
        self._require_record_values(record)
        self._require_manifest_ref(
            manifest_ref=manifest_ref,
            run_id=run_id,
            allow_pending_manifest_ref=allow_pending_manifest_ref,
        )
        self._verify_closure_manifest_ref(
            manifest_ref=manifest_ref,
            docx_pdf_viewer_e2e_status=docx_pdf_viewer_e2e_status,
            allow_pending_manifest_ref=allow_pending_manifest_ref,
        )
        self._require_aware_timestamp(recorded_at)
        hashable = {key: value for key, value in record.items() if key != "content_hash"}
        expected_hash = f"sha256:{content_hash(canonical_json(hashable))}"
        if record.get("content_hash") != expected_hash:
            raise ValueError("E2E flag criteria record content_hash mismatch")

    @staticmethod
    def _require_record_fields(record: dict[str, Any]) -> None:
        expected = {
            "schema_version",
            "workspace_id",
            "plan_ref",
            "flag",
            "run_id",
            "release_candidate_id",
            "docx_pdf_viewer_e2e_status",
            "viewer_related_flake_count",
            "manifest_ref",
            "recorded_at",
            "content_hash",
        }
        unexpected = sorted(set(record) - expected)
        if unexpected:
            raise ValueError(
                f"E2E flag criteria record has unexpected fields: {unexpected}"
            )

    @staticmethod
    def _require_record_values(record: dict[str, Any]) -> None:
        if record.get("docx_pdf_viewer_e2e_status") not in {
            "passed",
            "failed",
            "skipped",
            "blocked",
        }:
            raise ValueError("E2E flag criteria record docx_pdf_viewer_e2e_status is invalid")
        flake_count = record.get("viewer_related_flake_count")
        if (
            not isinstance(flake_count, int)
            or isinstance(flake_count, bool)
            or flake_count < 0
        ):
            raise ValueError("E2E flag criteria record viewer_related_flake_count is invalid")

    @staticmethod
    def _require_aware_timestamp(value: str) -> None:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError("recorded_at must be timezone-aware UTC ISO-8601")

    @staticmethod
    def _require_manifest_ref(
        *,
        manifest_ref: str,
        run_id: str,
        allow_pending_manifest_ref: bool = True,
    ) -> None:
        if not isinstance(manifest_ref, str):
            raise ValueError(
                "E2E flag criteria manifest_ref must reference a workspace closure "
                "manifest artifact or the current pending release-gate run"
            )
        if manifest_ref.startswith(WORKSPACE_CLOSURE_MANIFEST_PREFIX):
            suffix = manifest_ref.removeprefix(WORKSPACE_CLOSURE_MANIFEST_PREFIX)
            if not suffix.strip():
                raise ValueError(
                    "E2E flag criteria manifest_ref must reference a closure "
                    "manifest artifact"
                )
            if not suffix.endswith(".json"):
                raise ValueError(
                    "E2E flag criteria manifest_ref must reference a JSON closure "
                    "manifest artifact"
                )
            path = Path(manifest_ref)
            if path.is_absolute() or "\\" in manifest_ref or ".." in path.parts:
                raise ValueError(
                    "E2E flag criteria manifest_ref must stay within closure "
                    "manifest artifacts"
                )
            return
        if (
            allow_pending_manifest_ref
            and manifest_ref == f"{PENDING_RELEASE_GATE_PREFIX}{run_id}"
        ):
            return
        if manifest_ref.startswith(PENDING_RELEASE_GATE_PREFIX):
            raise ValueError("E2E flag criteria record cannot persist a pending manifest_ref")
        raise ValueError(
            "E2E flag criteria manifest_ref must reference a workspace closure "
            "manifest artifact or the current pending release-gate run"
        )

    @staticmethod
    def _valid_pending_manifest_ref(value: str) -> bool:
        run_id = value.removeprefix(PENDING_RELEASE_GATE_PREFIX)
        path = Path(run_id)
        return (
            bool(run_id)
            and not path.is_absolute()
            and "/" not in run_id
            and "\\" not in run_id
            and ".." not in path.parts
        )

    def _verify_closure_manifest_ref(
        self,
        *,
        manifest_ref: str,
        docx_pdf_viewer_e2e_status: str,
        allow_pending_manifest_ref: bool,
    ) -> None:
        if manifest_ref.startswith(PENDING_RELEASE_GATE_PREFIX):
            if allow_pending_manifest_ref:
                return
            raise ValueError("E2E flag criteria record cannot persist a pending manifest_ref")
        path = self._resolve_closure_manifest_ref(manifest_ref)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self._verify_closure_manifest(manifest)
        if docx_pdf_viewer_e2e_status == "passed":
            gates = {gate.get("gate_id"): gate for gate in manifest.get("gates") or []}
            if gates.get("open_source_docx_pdf_viewer_e2e", {}).get("status") != "pass":
                raise ValueError(
                    "E2E flag criteria passed record requires passing "
                    "open_source_docx_pdf_viewer_e2e manifest gate"
                )

    def _verify_closure_manifest(self, manifest: dict[str, Any]) -> None:
        if manifest.get("schema_version") != "mvl_closure_evidence_manifest.v1":
            raise ValueError("E2E flag criteria closure manifest schema_version is invalid")
        if manifest.get("workspace_id") != self.workspace_id:
            raise ValueError("E2E flag criteria closure manifest workspace_id mismatch")
        if manifest.get("plan_ref") != PLAN_REF:
            raise ValueError("E2E flag criteria closure manifest plan_ref mismatch")
        hashable = {key: value for key, value in manifest.items() if key != "content_hash"}
        expected_hash = f"sha256:{content_hash(canonical_json(hashable))}"
        if manifest.get("content_hash") != expected_hash:
            raise ValueError("E2E flag criteria closure manifest content_hash mismatch")

    def _resolve_closure_manifest_ref(self, manifest_ref: str) -> Path:
        if not manifest_ref.startswith(WORKSPACE_CLOSURE_MANIFEST_PREFIX):
            raise ValueError("E2E flag criteria manifest_ref must reference a closure manifest")
        suffix = manifest_ref.removeprefix(WORKSPACE_CLOSURE_MANIFEST_PREFIX)
        if not suffix.strip():
            raise ValueError(
                "E2E flag criteria manifest_ref must reference a closure manifest artifact"
            )
        if not suffix.endswith(".json"):
            raise ValueError(
                "E2E flag criteria manifest_ref must reference a JSON closure "
                "manifest artifact"
            )
        workspace_root = self.workspace_root / "workspaces" / self.workspace_id
        path = (workspace_root / manifest_ref).resolve()
        closure_manifest_root = (
            workspace_root / "artifacts" / "mvl-closure" / "evidence-manifests"
        ).resolve()
        if not path.is_relative_to(closure_manifest_root):
            raise ValueError(
                "E2E flag criteria manifest_ref must stay within closure manifest artifacts"
            )
        return path

    @staticmethod
    def _require(**values: str) -> None:
        missing = sorted(name for name, value in values.items() if value is None)
        if missing:
            raise ValueError(f"E2E flag criteria record missing fields: {missing}")

    @staticmethod
    def _require_string_fields(**values: Any) -> None:
        invalid = sorted(
            name
            for name, value in values.items()
            if not isinstance(value, str) or not value.strip()
        )
        if invalid:
            raise ValueError(
                f"E2E flag criteria record fields must be nonblank strings: {invalid}"
            )

    @staticmethod
    def _require_safe_artifact_id(value: str, *, label: str) -> None:
        if not isinstance(value, str):
            raise ValueError(
                f"E2E flag criteria record {label} must be a safe artifact id"
            )
        path = Path(value)
        if (
            not value.strip()
            or path.is_absolute()
            or "/" in value
            or "\\" in value
            or ".." in path.parts
        ):
            raise ValueError(
                f"E2E flag criteria record {label} must be a safe artifact id"
            )
