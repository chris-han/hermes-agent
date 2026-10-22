"""Atomic append-only run artifacts for semantic retrieval and hypotheses."""
from __future__ import annotations

import json, os, re, shutil
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4
from contextlib import contextmanager, ExitStack
from storage.governed_file_io import exclusive_file_lock, sync_directory, clean_publication_temps
from dataclasses import asdict
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.observed_constraint_hypothesis import (
    ObservedConstraintHypothesisV1,
    ObservedConstraintHypothesisV2,
    ObservedConstraintHypothesisV3,
    RuntimeSemanticRetrievalSnapshotV1,
)
from contracts.decision_local_projection import DecisionLocalProjectionV1
from contracts.decision_local_projection_v2 import DecisionLocalProjectionV2
from contracts.runtime_semantic_source_region import RuntimeSemanticSourceRegionV1
from contracts.frame_match_result_v2 import FrameMatchResultDTOV2
from contracts.governed_frame_finding_v1 import GovernedFrameFindingV1
from contracts.semantic_kernel import WorkingContext
from eos.artifact_hashing import content_hash
from contracts.runtime_judgment_execution import (
    RuntimeJudgmentRequestV1, RuntimeJudgmentAttemptV1, RuntimeJudgmentResponseV1,
)

_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SHA = re.compile(r"^sha256:[0-9a-f]{64}$")
_FRAME_MATCH_WRITE_CAPABILITY = object()
_FRAME_FINDING_WRITE_CAPABILITY = object()

def _safe(value: str, field: str) -> str:
    if not isinstance(value, str) or value in {".", ".."} or _SAFE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a safe single-segment ASCII identifier")
    return value

class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["runtime_semantic_artifact_manifest.v1"]
    manifest_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    organization_id: str
    workspace_id: str
    run_id: str
    artifact_ref: str
    artifact_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    artifact_kind: Literal[
        "runtime_semantic_retrieval_snapshot.v1",
        "observed_constraint_hypothesis.v1",
        "observed_constraint_hypothesis.v2",
        "observed_constraint_hypothesis.v3",
        "decision_local_projection.v2",
    ]
    artifact_filename: Literal[
        "runtime_semantic_retrieval_snapshot.v1.json",
        "observed_constraint_hypothesis.v1.json",
        "observed_constraint_hypothesis.v2.json",
        "observed_constraint_hypothesis.v3.json",
        "decision_local_projection.v2.json",
    ]
    retrieval_snapshot_ref: str | None = None
    retrieval_snapshot_hash: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def valid(self) -> "_Manifest":
        for field in ("organization_id", "workspace_id", "run_id", "artifact_ref"):
            _safe(getattr(self, field), field)
        if (self.retrieval_snapshot_ref is None) != (self.retrieval_snapshot_hash is None):
            raise ValueError("retrieval snapshot pins must be both present or null")
        if self.artifact_kind == "runtime_semantic_retrieval_snapshot.v1":
            if self.artifact_filename != "runtime_semantic_retrieval_snapshot.v1.json" or self.retrieval_snapshot_ref is not None:
                raise ValueError("retrieval snapshot manifest shape mismatch")
        else:
            expected = f"{self.artifact_kind}.json"
            if self.artifact_filename != expected or self.retrieval_snapshot_ref is None:
                raise ValueError("hypothesis manifest shape mismatch")
        payload = self.model_dump(mode="json"); payload.pop("manifest_hash")
        if self.manifest_hash != "sha256:" + content_hash(payload):
            raise ValueError("runtime semantic manifest hash mismatch")
        return self

    @classmethod
    def create(cls, **payload: Any) -> "_Manifest":
        return cls.model_validate({**payload, "manifest_hash": "sha256:" + content_hash(payload)})


class _DecisionProjectionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["decision_local_projection_manifest.v1"]
    manifest_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    organization_id: str
    workspace_id: str
    run_id: str
    projection_ref: str
    projection_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    working_context_ref: str
    working_context_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    retrieval_snapshot_ref: str
    retrieval_snapshot_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    projection_filename: Literal["decision_local_projection.v1.json"]
    working_context_filename: Literal["working_context.v1.json"]
    source_region_ref: str | None = None
    source_region_hash: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    source_region_filename: Literal["runtime_semantic_source_region.v1.json"] | None = None

    @model_validator(mode="after")
    def valid(self) -> "_DecisionProjectionManifest":
        for field in (
            "organization_id",
            "workspace_id",
            "run_id",
            "projection_ref",
            "working_context_ref",
            "retrieval_snapshot_ref",
        ):
            _safe(getattr(self, field), field)
        source_binding = (self.source_region_ref, self.source_region_hash, self.source_region_filename)
        if any(value is not None for value in source_binding) and any(value is None for value in source_binding):
            raise ValueError("decision projection source-region manifest binding must be complete")
        payload = self.model_dump(mode="json")
        payload.pop("manifest_hash")
        if self.manifest_hash != "sha256:" + content_hash(payload):
            raise ValueError("decision projection manifest hash mismatch")
        return self

    @classmethod
    def create(cls, **payload: Any) -> "_DecisionProjectionManifest":
        return cls.model_validate(
            {**payload, "manifest_hash": "sha256:" + content_hash(payload)}
        )


class _FrameMatchManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["frame_match_result_manifest.v1"]
    manifest_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    organization_id: str
    workspace_id: str
    run_id: str
    result_ref: str
    result_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    observed_hypothesis_ref: str
    observed_hypothesis_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    artifact_filename: Literal["frame_match_result_dto.v2.json"]

    @model_validator(mode="after")
    def valid(self) -> "_FrameMatchManifest":
        for field in ("organization_id", "workspace_id", "run_id", "result_ref", "observed_hypothesis_ref"):
            _safe(getattr(self, field), field)
        payload = self.model_dump(mode="json"); payload.pop("manifest_hash")
        if self.manifest_hash != "sha256:" + content_hash(payload):
            raise ValueError("frame match result manifest hash mismatch")
        return self

    @classmethod
    def create(cls, **payload: Any) -> "_FrameMatchManifest":
        return cls.model_validate({**payload, "manifest_hash": "sha256:" + content_hash(payload)})


class _FrameFindingManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["governed_frame_finding_manifest.v1"]
    manifest_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    organization_id: str
    workspace_id: str
    run_id: str
    finding_ref: str
    finding_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    frame_match_result_ref: str
    frame_match_result_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    artifact_filename: Literal["governed_frame_finding.v1.json"]

    @model_validator(mode="after")
    def valid(self) -> "_FrameFindingManifest":
        for field in ("organization_id", "workspace_id", "run_id", "finding_ref", "frame_match_result_ref"):
            _safe(getattr(self, field), field)
        payload = self.model_dump(mode="json"); payload.pop("manifest_hash")
        if self.manifest_hash != "sha256:" + content_hash(payload):
            raise ValueError("governed frame finding manifest hash mismatch")
        return self

    @classmethod
    def create(cls, **payload: Any) -> "_FrameFindingManifest":
        return cls.model_validate({**payload, "manifest_hash": "sha256:" + content_hash(payload)})

def _write_staged_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)); stream.flush(); os.fsync(stream.fileno())

def _publish_committed_directory(target: Path, *, files: dict[str, dict[str, Any]], conflict_error: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(target.parent / f".{target.name}.publish.lock"):
        clean_publication_temps(target)
        staged = target.parent / f".{target.name}.{uuid4().hex}.tmp"
        try:
            if target.is_dir():
                if {p.name for p in target.iterdir()} != set(files) or any(
                    not (target / name).is_file() or json.loads((target / name).read_text(encoding="utf-8")) != payload
                    for name, payload in files.items()
                ):
                    raise ValueError(conflict_error)
                sync_directory(target)
                sync_directory(target.parent)
                return
            staged.mkdir()
            for name, payload in files.items(): _write_staged_json(staged / name, payload)
            sync_directory(staged)
            os.replace(staged, target)
            sync_directory(target.parent)
        finally:
            if staged.exists(): shutil.rmtree(staged)

class RuntimeSemanticHypothesisArtifactStore:
    def __init__(self, runtime_root: str | Path, *, organization_id: str, workspace_id: str) -> None:
        self.organization_id, self.workspace_id = _safe(organization_id, "organization_id"), _safe(workspace_id, "workspace_id")
        root = Path(runtime_root).resolve()
        workspace = (root if root.name == workspace_id and root.parent.name == "workspaces" else root / "workspaces" / workspace_id).resolve()
        self.root = (workspace / "runs").resolve()
        if not self.root.is_relative_to(workspace): raise ValueError("artifact root escapes workspace")

    def _source_text_dir(self, request_id, *, boundary):
        _safe(request_id, "request_id")
        workspace = self.root.parent.resolve()
        root = Path(boundary.paths.runs_root).resolve()
        if boundary.workspace_id != self.workspace_id or not root.is_relative_to(workspace):
            raise ValueError("source query boundary scope mismatch")
        target = (root / "source-text-query" / request_id).resolve()
        if not target.is_relative_to(root) or not any(target.is_relative_to(Path(p).resolve()) for p in boundary.policy.allowed_write_roots):
            raise ValueError("source query sink outside execution boundary")
        from gateway.execution_boundary import DiskIOBoundaryGuard
        return DiskIOBoundaryGuard(boundary).resolve_write_path(target)

    @contextmanager
    def source_text_attempt_lock(self, request_id, *, boundary):
        target = self._source_text_dir(request_id, boundary=boundary)
        target.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            try:
                stack.enter_context(exclusive_file_lock(target / ".execution.lock", wait=False))
            except BlockingIOError as exc:
                raise ValueError("SOURCE_TEXT_EXECUTION_IN_PROGRESS") from exc
            yield

    def write_source_text_stage(self, request_id, stage, payload, *, boundary):
        if stage not in {"start", "end"}:
            raise ValueError("invalid source query stage")
        request = payload["request"] if stage == "start" else payload["start"]["request"]
        self._own(request["organization_id"], request["workspace_id"])
        if request["request_id"] != request_id:
            raise ValueError("request identity mismatch")
        target = self._source_text_dir(request_id, boundary=boundary) / stage
        _publish_committed_directory(target, files={stage + ".json": payload},
                                     conflict_error="SOURCE_TEXT_STAGE_CONFLICT")

    def read_source_text_attempt(self, request_id, *, boundary):
        target = self._source_text_dir(request_id, boundary=boundary)
        start_path = target / "start/start.json"
        if not start_path.exists():
            if (target / "end").exists():
                raise ValueError("source query START missing")
            return None
        start = json.loads(start_path.read_text(encoding="utf-8"))
        request = start["request"]
        self._own(request["organization_id"], request["workspace_id"])
        if request["request_id"] != request_id:
            raise ValueError("source query request path mismatch")
        end_path = target / "end/end.json"
        if not end_path.exists():
            return {"start": start}
        packet = json.loads(end_path.read_text(encoding="utf-8"))
        if packet["start"] != start:
            raise ValueError("source query START mismatch")
        return packet

    def put_legal_evidence_result(self, result):
        """Retain a DEVELOPMENT evidence packet without creating a Finding."""
        from contracts.legal_evidence_recovery import LegalEvidenceResultV1
        result = LegalEvidenceResultV1.model_validate(result.model_dump(mode="json", by_alias=True))
        self._own(result.request.organization_id, result.request.workspace_id)
        _publish_committed_directory(self._dir(result.request.run_id, result.result_hash) / "legal_evidence",
            files={"legal_evidence_result.v1.json": result.model_dump(mode="json", by_alias=True)},
            conflict_error="LEGAL_EVIDENCE_RESULT_CONFLICT")
        return result

    def get_legal_evidence_result(self, *, run_id, result_hash):
        """Cold verification uses immutable packet bytes only, never source acquisition."""
        from contracts.legal_evidence_recovery import LegalEvidenceResultV1
        path = self._dir(run_id, result_hash) / "legal_evidence/legal_evidence_result.v1.json"
        result = LegalEvidenceResultV1.model_validate_json(path.read_text(encoding="utf-8"))
        self._own(result.request.organization_id, result.request.workspace_id)
        if (result.request.run_id, result.result_hash) != (run_id, result_hash):
            raise ValueError("legal evidence result path/identity mismatch")
        return result

    def prepare_judgment_execution(self, *, run_id, request, execution_profile, request_id):
        record = RuntimeJudgmentRequestV1.create(organization_id=self.organization_id,
            workspace_id=self.workspace_id, run_id=run_id, request_id=request_id,
            judgment_request=request, execution_profile=execution_profile)
        self._verify_execution(record)
        _publish_committed_directory(self._dir(run_id, record.artifact_hash) / "judgment_request",
            files={"request.v1.json": record.model_dump(mode="json")},
            conflict_error="JUDGMENT_REQUEST_CONFLICT")
        return record

    def get_judgment_execution(self, *, run_id, execution_hash):
        path = self._dir(run_id, execution_hash) / "judgment_request/request.v1.json"
        record = RuntimeJudgmentRequestV1.model_validate_json(path.read_text(encoding="utf-8"))
        if record.artifact_hash != execution_hash or record.run_id != run_id:
            raise ValueError("judgment execution identity mismatch")
        self._verify_execution(record)
        return record

    def _verify_execution(self, execution):
        self._own(execution.organization_id, execution.workspace_id)
        request = execution.judgment_request
        if set(request) != {"decision_projection_ref", "decision_projection_hash", "instructions", "input_text", "json_schema"}:
            raise ValueError("judgment execution request shape mismatch")
        projection = self.get_bounded_projection(projection_ref=request["decision_projection_ref"],
                                                projection_hash=request["decision_projection_hash"])
        if projection.retrieval_snapshot.run_id != execution.run_id:
            raise ValueError("judgment execution projection run mismatch")
        payload = json.loads(request["input_text"])
        from services.runtime_semantic_hypothesis_prompt import projection_schema_contract
        expected = dict(decision_projection_ref=projection.projection_ref,
            decision_projection_hash=projection.projection_hash, decision_input=projection.model_input_payload(),
            schema_contract=projection_schema_contract(projection, request["json_schema"]))
        if payload != expected:
            raise ValueError("judgment execution projection input mismatch")

    @contextmanager
    def judgment_execution_lock(self, execution):
        directory = self._dir(execution.run_id, execution.artifact_hash)
        directory.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            try:
                stack.enter_context(exclusive_file_lock(directory / ".execution.lock", wait=False))
            except BlockingIOError as exc:
                raise ValueError("JUDGMENT_EXECUTION_IN_PROGRESS") from exc
            yield

    def put_judgment_attempt(self, *, execution, compatibility, predecessor=None):
        retained = self.get_judgment_execution(run_id=execution.run_id, execution_hash=execution.artifact_hash)
        if retained != execution:
            raise ValueError("judgment execution binding mismatch")
        if predecessor is not None:
            prior = self.get_judgment_attempt(run_id=execution.run_id, attempt_hash=predecessor.artifact_hash)
            if prior != predecessor or prior.execution_hash != execution.artifact_hash:
                raise ValueError("judgment predecessor binding mismatch")
        compatibility = dict(compatibility)
        compatibility.setdefault("checked_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
        self._verify_compatibility(execution, compatibility)
        record = RuntimeJudgmentAttemptV1.create(organization_id=self.organization_id,
            workspace_id=self.workspace_id, run_id=execution.run_id,
            execution_ref=execution.artifact_ref, execution_hash=execution.artifact_hash,
            predecessor_attempt_ref=predecessor.artifact_ref if predecessor else None,
            predecessor_attempt_hash=predecessor.artifact_hash if predecessor else None,
            compatibility=compatibility)
        _publish_committed_directory(self._dir(record.run_id, record.artifact_hash) / "judgment_attempt",
            files={"attempt.v1.json": record.model_dump(mode="json")}, conflict_error="JUDGMENT_ATTEMPT_CONFLICT")
        return record

    def get_judgment_attempt(self, *, run_id, attempt_hash):
        path = self._dir(run_id, attempt_hash) / "judgment_attempt/attempt.v1.json"
        record = RuntimeJudgmentAttemptV1.model_validate_json(path.read_text(encoding="utf-8"))
        self._own(record.organization_id, record.workspace_id)
        if record.run_id != run_id or record.artifact_hash != attempt_hash:
            raise ValueError("judgment attempt identity mismatch")
        execution = self.get_judgment_execution(run_id=run_id, execution_hash=record.execution_hash)
        if execution.artifact_ref != record.execution_ref:
            raise ValueError("judgment attempt execution mismatch")
        self._verify_compatibility(execution, record.compatibility)
        if record.predecessor_attempt_hash is not None:
            previous = self.get_judgment_attempt(run_id=run_id, attempt_hash=record.predecessor_attempt_hash)
            if previous.artifact_ref != record.predecessor_attempt_ref or previous.execution_hash != record.execution_hash:
                raise ValueError("judgment attempt predecessor mismatch")
        return record

    def latest_judgment_attempt(self, *, execution):
        directory = self._dir(execution.run_id, execution.artifact_hash).parent
        records = []
        for path in directory.glob("*/judgment_attempt/attempt.v1.json"):
            record = RuntimeJudgmentAttemptV1.model_validate_json(path.read_text(encoding="utf-8"))
            if record.execution_hash == execution.artifact_hash:
                records.append(self.get_judgment_attempt(run_id=execution.run_id, attempt_hash=record.artifact_hash))
        parents = {record.predecessor_attempt_hash for record in records}
        heads = [record for record in records if record.artifact_hash not in parents]
        if records and len(heads) != 1:
            raise ValueError("judgment attempt lineage is branched or cyclic")
        return heads[0] if heads else None

    @staticmethod
    def _verify_compatibility(execution, receipt):
        from contracts.runtime_judgment_execution import verify_complete_request_compatibility
        verify_complete_request_compatibility(execution_profile=execution.execution_profile,
            request=execution.judgment_request, receipt=receipt)

    def put_judgment_response(self, *, execution, attempt, result):
        if self.get_judgment_execution(run_id=execution.run_id, execution_hash=execution.artifact_hash) != execution:
            raise ValueError("judgment response request mismatch")
        retained = self.get_judgment_attempt(run_id=execution.run_id, attempt_hash=attempt.artifact_hash)
        if retained != attempt or attempt.execution_hash != execution.artifact_hash or attempt.compatibility["outcome"] != "COMPATIBLE":
            raise ValueError("judgment response compatibility binding mismatch")
        payload = asdict(result)
        profile = attempt.compatibility["profile"]
        if ((result.provider, result.model) != (profile["provider"], profile["model"])
            or result.audit.get("compatibility") != attempt.compatibility):
            raise ValueError("judgment response model/request mismatch")
        record = RuntimeJudgmentResponseV1.create(organization_id=self.organization_id,
            workspace_id=self.workspace_id, run_id=execution.run_id,
            execution_ref=execution.artifact_ref, execution_hash=execution.artifact_hash,
            predecessor_attempt_ref=attempt.artifact_ref, predecessor_attempt_hash=attempt.artifact_hash,
            result=payload)
        _publish_committed_directory(self._dir(execution.run_id, execution.artifact_hash) / "judgment_response",
            files={"response.v1.json": record.model_dump(mode="json")}, conflict_error="JUDGMENT_RESPONSE_CONFLICT")
        return record

    def get_judgment_response(self, *, execution):
        retained = self.get_judgment_execution(run_id=execution.run_id, execution_hash=execution.artifact_hash)
        if execution != retained:
            raise ValueError("judgment response request mismatch")
        path = self._dir(execution.run_id, execution.artifact_hash) / "judgment_response/response.v1.json"
        if not path.exists():
            return None
        record = RuntimeJudgmentResponseV1.model_validate_json(path.read_text(encoding="utf-8"))
        self._own(record.organization_id, record.workspace_id)
        attempt = self.get_judgment_attempt(run_id=execution.run_id, attempt_hash=record.predecessor_attempt_hash)
        profile = attempt.compatibility["profile"]
        if ((record.execution_ref, record.execution_hash, record.run_id) != (execution.artifact_ref, execution.artifact_hash, execution.run_id)
            or attempt.artifact_ref != record.predecessor_attempt_ref or attempt.execution_hash != execution.artifact_hash
            or attempt.compatibility["outcome"] != "COMPATIBLE" or record.result.get("audit", {}).get("compatibility") != attempt.compatibility
            or (record.result.get("provider"), record.result.get("model")) != (profile["provider"], profile["model"])):
            raise ValueError("judgment response lineage mismatch")
        return record

    def put_retrieval_snapshot(self, snapshot: RuntimeSemanticRetrievalSnapshotV1) -> RuntimeSemanticRetrievalSnapshotV1:
        snapshot = RuntimeSemanticRetrievalSnapshotV1.model_validate(snapshot.model_dump(mode="json", by_alias=True)); self._own(snapshot.organization_id, snapshot.workspace_id)
        name = "runtime_semantic_retrieval_snapshot.v1.json"
        manifest = _Manifest.create(schema_version="runtime_semantic_artifact_manifest.v1", organization_id=self.organization_id, workspace_id=self.workspace_id, run_id=snapshot.run_id, artifact_ref=snapshot.snapshot_id, artifact_hash=snapshot.snapshot_hash, artifact_kind=snapshot.schema_version, artifact_filename=name, retrieval_snapshot_ref=None, retrieval_snapshot_hash=None)
        _publish_committed_directory(self._dir(snapshot.run_id, snapshot.snapshot_hash), files={name: snapshot.model_dump(mode="json", by_alias=True), "retrieval_snapshot.manifest.v1.json": manifest.model_dump(mode="json")}, conflict_error="RUNTIME_SEMANTIC_RETRIEVAL_SNAPSHOT_CONFLICT")
        return snapshot

    def get_retrieval_snapshot(self, *, snapshot_ref: str, snapshot_hash: str) -> RuntimeSemanticRetrievalSnapshotV1:
        directory, manifest = self._manifest(snapshot_ref, snapshot_hash, "retrieval_snapshot")
        result = RuntimeSemanticRetrievalSnapshotV1.model_validate_json((directory / manifest.artifact_filename).read_text(encoding="utf-8")); self._own(result.organization_id, result.workspace_id)
        if (result.snapshot_id, result.snapshot_hash, result.run_id) != (snapshot_ref, snapshot_hash, manifest.run_id): raise ValueError("retrieval snapshot identity/run mismatch")
        return result

    def put_bounded_projection(self, projection: DecisionLocalProjectionV2) -> DecisionLocalProjectionV2:
        """Publish the compact successor only after its original lineage is durable."""
        projection = DecisionLocalProjectionV2.model_validate_json(projection.model_dump_json())
        self._verify_bounded_projection(projection)
        snapshot = projection.retrieval_snapshot
        name = "decision_local_projection.v2.json"
        manifest = _Manifest.create(
            schema_version="runtime_semantic_artifact_manifest.v1",
            organization_id=self.organization_id, workspace_id=self.workspace_id,
            run_id=snapshot.run_id, artifact_ref=projection.projection_ref,
            artifact_hash=projection.projection_hash, artifact_kind=projection.schema_version,
            artifact_filename=name, retrieval_snapshot_ref=snapshot.snapshot_id,
            retrieval_snapshot_hash=snapshot.snapshot_hash,
        )
        _publish_committed_directory(self._dir(snapshot.run_id, projection.projection_hash),
            files={name: projection.model_dump(mode="json"),
                   "bounded_projection.manifest.v1.json": manifest.model_dump(mode="json")},
            conflict_error="BOUNDED_DECISION_PROJECTION_CONFLICT")
        return projection

    def get_bounded_projection(self, *, projection_ref: str, projection_hash: str) -> DecisionLocalProjectionV2:
        directory, manifest = self._manifest(projection_ref, projection_hash, "bounded_projection")
        projection = DecisionLocalProjectionV2.model_validate_json(
            (directory / manifest.artifact_filename).read_text(encoding="utf-8"))
        if (projection.projection_ref, projection.projection_hash, projection.retrieval_snapshot.run_id,
            projection.retrieval_snapshot.snapshot_id, projection.retrieval_snapshot.snapshot_hash) != (
            projection_ref, projection_hash, manifest.run_id,
            manifest.retrieval_snapshot_ref, manifest.retrieval_snapshot_hash):
            raise ValueError("bounded projection manifest binding mismatch")
        self._verify_bounded_projection(projection)
        return projection

    def _verify_bounded_projection(self, projection: DecisionLocalProjectionV2) -> None:
        self._own(projection.legacy.organization_id, projection.legacy.workspace_id)
        legacy, working, snapshot = self.get_decision_projection(
            projection_ref=projection.legacy.projection_ref,
            projection_hash=projection.legacy.projection_hash)
        if (legacy, working, snapshot) != (projection.legacy, projection.working_context, projection.retrieval_snapshot):
            raise ValueError("bounded projection persisted lineage mismatch")
        # Revalidate the retained rule bytes and exact authority contract. A
        # self-consistent V2 hash alone is not an authoritative rule binding.
        from services.tender_decision_local_projection_service import TenderDecisionLocalProjectionService
        rules = {rule["rule_version_id"]: rule for rule in projection.governed_rules}
        if len(rules) != len(projection.governed_rules):
            raise ValueError("duplicate bounded projection rule")
        constraints, semantic_type, family = TenderDecisionLocalProjectionService.validate_runtime_binding(
            runtime_context=projection.runtime_context, retrieval_snapshot=snapshot,
            canonical_pattern_ref=projection.canonical_pattern_ref, governed_rule_resolver=rules.get)
        predicate = projection.legacy.semantic_predicates[0]
        expected = TenderDecisionLocalProjectionService.decision_contract(
            issue_type=semantic_type, family_id=family, deterministic_constraints=constraints,
            material_requirements=predicate.material_requirements)
        if expected != projection.decision_contract:
            raise ValueError("bounded projection authority contract mismatch")

    def put_hypothesis(self, hypothesis: ObservedConstraintHypothesisV1 | ObservedConstraintHypothesisV2, *, judgment_request: dict | None = None) -> ObservedConstraintHypothesisV1 | ObservedConstraintHypothesisV2:
        model = (
            ObservedConstraintHypothesisV3
            if hypothesis.schema_version == "observed_constraint_hypothesis.v3" else ObservedConstraintHypothesisV2
            if hypothesis.schema_version == "observed_constraint_hypothesis.v2"
            else ObservedConstraintHypothesisV1
        )
        hypothesis = model.model_validate(hypothesis.model_dump(mode="json", by_alias=True)); self._own(hypothesis.organization_id, hypothesis.workspace_id)
        snapshot = self.get_retrieval_snapshot(snapshot_ref=hypothesis.retrieval_snapshot_ref, snapshot_hash=hypothesis.retrieval_snapshot_hash)
        if snapshot.run_id != hypothesis.run_id: raise ValueError("hypothesis/retrieval run binding mismatch")
        self._bind(hypothesis, snapshot)
        if hypothesis.predecessor_hypothesis_ref is not None:
            self._resolve_predecessor_chain(hypothesis.predecessor_hypothesis_ref, visited=set())
        request_files = {}
        if isinstance(hypothesis, ObservedConstraintHypothesisV3):
            self._verify_judgment_request(hypothesis, judgment_request)
            request_files["judgment_request.v1.json"] = judgment_request
        elif judgment_request is not None:
            raise ValueError("judgment request requires a projection-bound hypothesis")
        name = f"{hypothesis.schema_version}.json"
        manifest = _Manifest.create(schema_version="runtime_semantic_artifact_manifest.v1", organization_id=self.organization_id, workspace_id=self.workspace_id, run_id=hypothesis.run_id, artifact_ref=hypothesis.hypothesis_id, artifact_hash=hypothesis.hypothesis_hash, artifact_kind=hypothesis.schema_version, artifact_filename=name, retrieval_snapshot_ref=snapshot.snapshot_id, retrieval_snapshot_hash=snapshot.snapshot_hash)
        _publish_committed_directory(self._dir(hypothesis.run_id, hypothesis.hypothesis_hash), files={name: hypothesis.model_dump(mode="json", by_alias=True), "hypothesis.manifest.v1.json": manifest.model_dump(mode="json"), **request_files}, conflict_error="OBSERVED_CONSTRAINT_HYPOTHESIS_CONFLICT")
        return hypothesis

    def get_hypothesis(self, *, hypothesis_ref: str, hypothesis_hash: str) -> ObservedConstraintHypothesisV1 | ObservedConstraintHypothesisV2:
        directory, manifest = self._manifest(hypothesis_ref, hypothesis_hash, "hypothesis")
        model = (
            ObservedConstraintHypothesisV3
            if manifest.artifact_kind == "observed_constraint_hypothesis.v3" else ObservedConstraintHypothesisV2
            if manifest.artifact_kind == "observed_constraint_hypothesis.v2"
            else ObservedConstraintHypothesisV1
        )
        result = model.model_validate_json((directory / manifest.artifact_filename).read_text(encoding="utf-8")); self._own(result.organization_id, result.workspace_id)
        if (result.hypothesis_id, result.hypothesis_hash, result.run_id) != (hypothesis_ref, hypothesis_hash, manifest.run_id): raise ValueError("hypothesis identity/run mismatch")
        if (manifest.retrieval_snapshot_ref, manifest.retrieval_snapshot_hash) != (result.retrieval_snapshot_ref, result.retrieval_snapshot_hash): raise ValueError("hypothesis manifest retrieval binding mismatch")
        self._bind(result, self.get_retrieval_snapshot(snapshot_ref=result.retrieval_snapshot_ref, snapshot_hash=result.retrieval_snapshot_hash))
        if isinstance(result, ObservedConstraintHypothesisV3):
            self._verify_judgment_request(result, json.loads((directory / "judgment_request.v1.json").read_text(encoding="utf-8")))
        return result

    def get_judgment_request(self, *, hypothesis_ref: str, hypothesis_hash: str) -> dict | None:
        hypothesis = self.get_hypothesis(hypothesis_ref=hypothesis_ref, hypothesis_hash=hypothesis_hash)
        if not isinstance(hypothesis, ObservedConstraintHypothesisV3):
            return None
        request = json.loads((self._dir(hypothesis.run_id, hypothesis.hypothesis_hash)
                              / "judgment_request.v1.json").read_text(encoding="utf-8"))
        self._verify_judgment_request(hypothesis, request)
        return request

    def _verify_judgment_request(self, hypothesis: ObservedConstraintHypothesisV3, request: dict | None) -> None:
        if not isinstance(request, dict) or "sha256:" + content_hash(request) != hypothesis.judgment_request_hash:
            raise ValueError("judgment request hash mismatch")
        base_fields = {"decision_projection_ref", "decision_projection_hash", "instructions", "input_text", "json_schema"}
        execution_fields = {"execution_request_ref", "execution_request_hash", "execution_response_ref", "execution_response_hash"}
        if set(request) not in (base_fields, base_fields | execution_fields):
            raise ValueError("judgment request shape mismatch")
        if execution_fields.issubset(request):
            execution = self.get_judgment_execution(run_id=hypothesis.run_id, execution_hash=request["execution_request_hash"])
            response = self.get_judgment_response(execution=execution)
            if (execution.artifact_ref != request["execution_request_ref"] or response is None
                or (response.artifact_ref, response.artifact_hash) != (request["execution_response_ref"], request["execution_response_hash"])
                or execution.judgment_request != {key: request[key] for key in base_fields}):
                raise ValueError("judgment execution response binding mismatch")
        if (request["decision_projection_ref"], request["decision_projection_hash"]) != (
            hypothesis.decision_projection_ref, hypothesis.decision_projection_hash):
            raise ValueError("judgment projection pin mismatch")
        projection = self.get_bounded_projection(projection_ref=hypothesis.decision_projection_ref,
            projection_hash=hypothesis.decision_projection_hash)
        if (projection.retrieval_snapshot.run_id, projection.retrieval_snapshot.snapshot_id,
            projection.retrieval_snapshot.snapshot_hash) != (
            hypothesis.run_id, hypothesis.retrieval_snapshot_ref, hypothesis.retrieval_snapshot_hash):
            raise ValueError("judgment projection run/retrieval mismatch")
        source = projection.source_region.source_binding
        if (hypothesis.source_binding.source_identity_ref, hypothesis.source_binding.source_hash,
            hypothesis.source_binding.canonical_source_ir_ref, hypothesis.source_binding.canonical_source_ir_hash,
            hypothesis.evidence_refs) != (
            source.source_identity_ref, source.source_hash, source.canonical_source_ir.artifact_ref,
            source.canonical_source_ir.artifact_hash, tuple(anchor.evidence_ref for anchor in projection.source_region.evidence)):
            raise ValueError("judgment projection source/evidence mismatch")
        selected = tuple((item.ref, item.hash) for item in projection.retrieval_snapshot.ordered_material
                         if item.kind == "canonical_pattern" and item.ref == projection.canonical_pattern_ref)
        if tuple(zip(hypothesis.retrieved_pattern_refs, hypothesis.retrieved_pattern_hashes, strict=True)) != selected:
            raise ValueError("judgment projection pattern mismatch")
        payload = json.loads(request["input_text"])
        if set(payload) != {"decision_projection_ref", "decision_projection_hash", "decision_input", "schema_contract"}:
            raise ValueError("judgment request contains an ungoverned input")
        if (payload.get("decision_projection_ref"), payload.get("decision_projection_hash"), payload.get("decision_input")) != (
            projection.projection_ref, projection.projection_hash, projection.model_input_payload()):
            raise ValueError("judgment request does not consume its projection")
        from hashlib import sha256
        if "sha256:" + sha256(request["instructions"].encode("utf-8")).hexdigest() != hypothesis.prompt_asset_hash:
            raise ValueError("judgment prompt hash mismatch")
        from services.runtime_semantic_hypothesis_prompt import projection_schema_contract
        if payload.get("schema_contract") != projection_schema_contract(projection, request["json_schema"]):
            raise ValueError("judgment output schema mismatch")

    def put_decision_projection(
        self,
        *,
        run_id: str,
        projection: DecisionLocalProjectionV1,
        working_context: WorkingContext,
        retrieval_snapshot_ref: str,
        retrieval_snapshot_hash: str,
        source_region: RuntimeSemanticSourceRegionV1 | None = None,
    ) -> DecisionLocalProjectionV1:
        projection = DecisionLocalProjectionV1.model_validate(
            projection.model_dump(mode="json")
        )
        try:
            working_context = WorkingContext.model_validate(
                working_context.model_dump(mode="json")
            )
        except ValueError as exc:
            raise ValueError("decision projection working context binding mismatch") from exc
        self._own(projection.organization_id, projection.workspace_id)
        self._own(working_context.organization_id, working_context.workspace_id)
        if working_context.pin() != projection.working_context:
            raise ValueError("decision projection working context binding mismatch")
        if (
            projection.context_graph != working_context.context_graph
            or projection.ontology != working_context.ontology
            or projection.evidence != working_context.evidence
        ):
            raise ValueError("decision projection working context closure mismatch")
        if any(predicate.material_requirements for predicate in projection.semantic_predicates) and source_region is None:
            raise ValueError("material-dependent decision projection requires its exact persisted source region")
        if source_region is not None:
            source_region = RuntimeSemanticSourceRegionV1.model_validate(source_region.model_dump(mode="json"))
            if (source_region.evidence[0].evidence_ref.organization_id,
                source_region.evidence[0].evidence_ref.workspace_id) != (
                working_context.organization_id, working_context.workspace_id
            ):
                raise ValueError("decision projection source-region scope mismatch")
            if not source_region.material_dependencies:
                raise ValueError("source region does not declare the DLP material requirements")
            declared = {(item.dependency_id, item.required_evidence_role, item.min_count)
                        for item in source_region.material_dependencies}
            compiled = {(item.dependency_id, item.required_evidence_role, item.min_count)
                        for predicate in projection.semantic_predicates for item in predicate.material_requirements}
            if declared != compiled:
                raise ValueError("decision projection/source-region material requirements mismatch")
        snapshot = self.get_retrieval_snapshot(
            snapshot_ref=retrieval_snapshot_ref,
            snapshot_hash=retrieval_snapshot_hash,
        )
        if snapshot.run_id != run_id:
            raise ValueError("decision projection/retrieval run binding mismatch")
        if (
            projection.organization_id,
            projection.workspace_id,
        ) != (
            snapshot.organization_id,
            snapshot.workspace_id,
        ):
            raise ValueError("decision projection/retrieval scope mismatch")
        projection_name = "decision_local_projection.v1.json"
        working_context_name = "working_context.v1.json"
        source_region_name = "runtime_semantic_source_region.v1.json" if source_region is not None else None
        source_region_hash = (
            "sha256:" + content_hash(source_region.model_dump(mode="json")) if source_region is not None else None
        )
        manifest = _DecisionProjectionManifest.create(
            schema_version="decision_local_projection_manifest.v1",
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            run_id=run_id,
            projection_ref=projection.projection_ref,
            projection_hash=projection.projection_hash,
            working_context_ref=working_context.value_ref,
            working_context_hash=working_context.value_hash,
            retrieval_snapshot_ref=snapshot.snapshot_id,
            retrieval_snapshot_hash=snapshot.snapshot_hash,
            projection_filename=projection_name,
            working_context_filename=working_context_name,
            source_region_ref=source_region.source_region_ref if source_region is not None else None,
            source_region_hash=source_region_hash,
            source_region_filename=source_region_name,
        )
        _publish_committed_directory(
            self._dir(run_id, projection.projection_hash),
            files={
                projection_name: projection.model_dump(mode="json"),
                working_context_name: working_context.model_dump(mode="json"),
                **({source_region_name: source_region.model_dump(mode="json")}
                   if source_region is not None and source_region_name is not None else {}),
                "decision_projection.manifest.v1.json": manifest.model_dump(mode="json"),
            },
            conflict_error="DECISION_LOCAL_PROJECTION_CONFLICT",
        )
        return projection

    def get_decision_projection(
        self,
        *,
        projection_ref: str,
        projection_hash: str,
        include_source_region: bool = False,
    ) -> tuple[
        DecisionLocalProjectionV1,
        WorkingContext,
        RuntimeSemanticRetrievalSnapshotV1,
    ] | tuple[DecisionLocalProjectionV1, WorkingContext, RuntimeSemanticRetrievalSnapshotV1, RuntimeSemanticSourceRegionV1 | None]:
        _safe(projection_ref, "projection_ref")
        self._hash(projection_hash)
        paths = list(
            self.root.glob(
                f"*/runtime-semantic/{projection_hash[7:]}/decision_projection.manifest.v1.json"
            )
        )
        valid: list[tuple[Path, _DecisionProjectionManifest]] = []
        for path in paths:
            manifest = _DecisionProjectionManifest.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            self._own(manifest.organization_id, manifest.workspace_id)
            if path.parent != self._dir(manifest.run_id, manifest.projection_hash) or (
                manifest.projection_ref,
                manifest.projection_hash,
            ) != (
                projection_ref,
                projection_hash,
            ):
                raise ValueError("decision projection manifest path/identity mismatch")
            valid.append((path.parent, manifest))
        if len(valid) != 1:
            raise ValueError("committed decision projection not found or ambiguous")
        directory, manifest = valid[0]
        projection = DecisionLocalProjectionV1.model_validate_json(
            (directory / manifest.projection_filename).read_text(encoding="utf-8")
        )
        working_context = WorkingContext.model_validate_json(
            (directory / manifest.working_context_filename).read_text(encoding="utf-8")
        )
        if (
            projection.projection_ref,
            projection.projection_hash,
        ) != (
            projection_ref,
            projection_hash,
        ):
            raise ValueError("decision projection identity mismatch")
        if (
            working_context.value_ref,
            working_context.value_hash,
        ) != (
            manifest.working_context_ref,
            manifest.working_context_hash,
        ) or working_context.pin() != projection.working_context:
            raise ValueError("decision projection working context binding mismatch")
        snapshot = self.get_retrieval_snapshot(
            snapshot_ref=manifest.retrieval_snapshot_ref,
            snapshot_hash=manifest.retrieval_snapshot_hash,
        )
        if snapshot.run_id != manifest.run_id:
            raise ValueError("decision projection/retrieval run binding mismatch")
        if include_source_region:
            source_region = None
            if manifest.source_region_filename is not None:
                source_region = RuntimeSemanticSourceRegionV1.model_validate_json(
                    (directory / manifest.source_region_filename).read_text(encoding="utf-8")
                )
                source_hash = "sha256:" + content_hash(source_region.model_dump(mode="json"))
                if (source_region.source_region_ref, source_hash) != (
                    manifest.source_region_ref, manifest.source_region_hash
                ):
                    raise ValueError("decision projection source-region identity mismatch")
            elif any(predicate.material_requirements for predicate in projection.semantic_predicates):
                raise ValueError("persisted material-dependent projection is missing its source region")
            return projection, working_context, snapshot, source_region
        return projection, working_context, snapshot

    def _put_frame_match_result(self, result: FrameMatchResultDTOV2, *, capability: object) -> FrameMatchResultDTOV2:
        if capability is not _FRAME_MATCH_WRITE_CAPABILITY:
            raise PermissionError("frame match result publication requires the governed matcher")
        result = FrameMatchResultDTOV2.model_validate(result.model_dump(mode="json", by_alias=True))
        self._own(result.organization_id, result.workspace_id)
        hypothesis = self.get_hypothesis(
            hypothesis_ref=result.observed_hypothesis_ref,
            hypothesis_hash=result.observed_hypothesis_hash,
        )
        snapshot = self.get_retrieval_snapshot(
            snapshot_ref=hypothesis.retrieval_snapshot_ref,
            snapshot_hash=hypothesis.retrieval_snapshot_hash,
        )
        self._bind_match_result(result, hypothesis, snapshot)
        name = "frame_match_result_dto.v2.json"
        manifest = _FrameMatchManifest.create(
            schema_version="frame_match_result_manifest.v1",
            organization_id=self.organization_id, workspace_id=self.workspace_id,
            run_id=hypothesis.run_id, result_ref=result.result_id,
            result_hash=result.result_hash,
            observed_hypothesis_ref=hypothesis.hypothesis_id,
            observed_hypothesis_hash=hypothesis.hypothesis_hash,
            artifact_filename=name,
        )
        _publish_committed_directory(
            self._dir(hypothesis.run_id, result.result_hash),
            files={name: result.model_dump(mode="json", by_alias=True),
                   "frame_match_result.manifest.v1.json": manifest.model_dump(mode="json")},
            conflict_error="FRAME_MATCH_RESULT_CONFLICT",
        )
        return result

    def get_frame_match_result(self, *, result_ref: str, result_hash: str) -> FrameMatchResultDTOV2:
        _safe(result_ref, "result_ref"); self._hash(result_hash)
        paths = list(self.root.glob(
            f"*/runtime-semantic/{result_hash[7:]}/frame_match_result.manifest.v1.json"
        ))
        valid: list[tuple[Path, _FrameMatchManifest]] = []
        for path in paths:
            manifest = _FrameMatchManifest.model_validate_json(path.read_text(encoding="utf-8"))
            self._own(manifest.organization_id, manifest.workspace_id)
            if path.parent != self._dir(manifest.run_id, manifest.result_hash) or (
                manifest.result_ref, manifest.result_hash
            ) != (result_ref, result_hash):
                raise ValueError("frame match result manifest path/identity mismatch")
            valid.append((path.parent, manifest))
        if len(valid) != 1:
            raise ValueError("committed frame match result not found or ambiguous")
        directory, manifest = valid[0]
        result = FrameMatchResultDTOV2.model_validate_json(
            (directory / manifest.artifact_filename).read_text(encoding="utf-8")
        )
        if (result.result_id, result.result_hash) != (result_ref, result_hash):
            raise ValueError("frame match result identity mismatch")
        hypothesis = self.get_hypothesis(
            hypothesis_ref=manifest.observed_hypothesis_ref,
            hypothesis_hash=manifest.observed_hypothesis_hash,
        )
        if (result.observed_hypothesis_ref, result.observed_hypothesis_hash) != (
            manifest.observed_hypothesis_ref, manifest.observed_hypothesis_hash
        ):
            raise ValueError("frame match result manifest hypothesis binding mismatch")
        snapshot = self.get_retrieval_snapshot(
            snapshot_ref=hypothesis.retrieval_snapshot_ref,
            snapshot_hash=hypothesis.retrieval_snapshot_hash,
        )
        self._bind_match_result(result, hypothesis, snapshot)
        return result

    def _put_governed_frame_finding(self, finding: GovernedFrameFindingV1, *, capability: object) -> GovernedFrameFindingV1:
        if capability is not _FRAME_FINDING_WRITE_CAPABILITY:
            raise PermissionError("frame finding publication requires the governed decision service")
        finding = GovernedFrameFindingV1.model_validate(finding.model_dump(mode="json", by_alias=True))
        self._own(finding.organization_id, finding.workspace_id)
        result = self.get_frame_match_result(
            result_ref=finding.frame_match_result_ref,
            result_hash=finding.frame_match_result_hash,
        )
        self._bind_finding(finding, result)
        hypothesis = self.get_hypothesis(
            hypothesis_ref=result.observed_hypothesis_ref,
            hypothesis_hash=result.observed_hypothesis_hash,
        )
        name = "governed_frame_finding.v1.json"
        manifest = _FrameFindingManifest.create(
            schema_version="governed_frame_finding_manifest.v1",
            organization_id=self.organization_id, workspace_id=self.workspace_id,
            run_id=hypothesis.run_id, finding_ref=finding.finding_id,
            finding_hash=finding.finding_hash,
            frame_match_result_ref=result.result_id,
            frame_match_result_hash=result.result_hash,
            artifact_filename=name,
        )
        _publish_committed_directory(
            self._dir(hypothesis.run_id, finding.finding_hash),
            files={name: finding.model_dump(mode="json", by_alias=True),
                   "governed_frame_finding.manifest.v1.json": manifest.model_dump(mode="json")},
            conflict_error="GOVERNED_FRAME_FINDING_CONFLICT",
        )
        return finding

    def get_governed_frame_finding(self, *, finding_ref: str, finding_hash: str) -> GovernedFrameFindingV1:
        _safe(finding_ref, "finding_ref"); self._hash(finding_hash)
        paths = list(self.root.glob(
            f"*/runtime-semantic/{finding_hash[7:]}/governed_frame_finding.manifest.v1.json"
        ))
        valid: list[tuple[Path, _FrameFindingManifest]] = []
        for path in paths:
            manifest = _FrameFindingManifest.model_validate_json(path.read_text(encoding="utf-8"))
            self._own(manifest.organization_id, manifest.workspace_id)
            if path.parent != self._dir(manifest.run_id, manifest.finding_hash) or (
                manifest.finding_ref, manifest.finding_hash
            ) != (finding_ref, finding_hash):
                raise ValueError("governed frame finding manifest path/identity mismatch")
            valid.append((path.parent, manifest))
        if len(valid) != 1:
            raise ValueError("committed governed frame finding not found or ambiguous")
        directory, manifest = valid[0]
        finding = GovernedFrameFindingV1.model_validate_json(
            (directory / manifest.artifact_filename).read_text(encoding="utf-8")
        )
        result = self.get_frame_match_result(
            result_ref=manifest.frame_match_result_ref,
            result_hash=manifest.frame_match_result_hash,
        )
        if (finding.finding_id, finding.finding_hash) != (finding_ref, finding_hash):
            raise ValueError("governed frame finding identity mismatch")
        self._bind_finding(finding, result)
        return finding

    def _by_ref(self, ref: str) -> list[ObservedConstraintHypothesisV1]:
        found = []
        for path in self.root.glob("*/runtime-semantic/*/hypothesis.manifest.v1.json"):
            try:
                manifest = _Manifest.model_validate_json(path.read_text(encoding="utf-8"))
                if manifest.artifact_ref == ref: found.append(self.get_hypothesis(hypothesis_ref=ref, hypothesis_hash=manifest.artifact_hash))
            except (ValueError, OSError, json.JSONDecodeError): continue
        return found

    def _resolve_predecessor_chain(self, ref: str, *, visited: set[str]) -> ObservedConstraintHypothesisV1:
        if ref in visited:
            raise ValueError("predecessor hypothesis chain contains a cycle")
        visited.add(ref)
        matches = self._by_ref(ref)
        if len(matches) != 1:
            raise ValueError("predecessor hypothesis must resolve to one committed governed artifact")
        predecessor = matches[0]
        if predecessor.predecessor_hypothesis_ref is not None:
            self._resolve_predecessor_chain(predecessor.predecessor_hypothesis_ref, visited=visited)
        return predecessor

    def _manifest(self, ref: str, digest: str, kind: Literal["retrieval_snapshot", "hypothesis", "bounded_projection"]) -> tuple[Path, _Manifest]:
        _safe(ref, f"{kind}_ref"); self._hash(digest)
        paths = list(self.root.glob(f"*/runtime-semantic/{digest[7:]}/{kind}.manifest.v1.json")); valid = []
        for path in paths:
            manifest = _Manifest.model_validate_json(path.read_text(encoding="utf-8"))
            if kind == "retrieval_snapshot":
                expected_kinds = {"runtime_semantic_retrieval_snapshot.v1"}
            elif kind == "bounded_projection":
                expected_kinds = {"decision_local_projection.v2"}
            else:
                expected_kinds = {
                    "observed_constraint_hypothesis.v1",
                    "observed_constraint_hypothesis.v2",
                    "observed_constraint_hypothesis.v3",
                }
            if manifest.artifact_kind not in expected_kinds: raise ValueError("manifest kind mismatch")
            if path.parent != self._dir(manifest.run_id, manifest.artifact_hash) or (manifest.artifact_ref, manifest.artifact_hash) != (ref, digest): raise ValueError("manifest path/identity mismatch")
            self._own(manifest.organization_id, manifest.workspace_id); valid.append((path.parent, manifest))
        if len(valid) != 1: raise ValueError(f"committed {kind.replace('_', ' ')} not found or ambiguous")
        return valid[0]

    def _dir(self, run_id: str, digest: str) -> Path:
        _safe(run_id, "run_id"); self._hash(digest); target = (self.root / run_id / "runtime-semantic" / digest[7:]).resolve()
        if not target.is_relative_to(self.root): raise ValueError("artifact path escapes governed runs")
        return target

    @staticmethod
    def _bind(h: ObservedConstraintHypothesisV1, s: RuntimeSemanticRetrievalSnapshotV1) -> None:
        names = ("accepted_graph_release_ref", "accepted_graph_release_hash", "governed_rule_projection_ref", "governed_rule_projection_hash", "activation_set_snapshot_ref", "activation_set_snapshot_hash", "effective_context_resolver_hash")
        if tuple(getattr(h, n) for n in names) != tuple(getattr(s, n) for n in names): raise ValueError("hypothesis/retrieval authority pin mismatch")

    @staticmethod
    def _bind_match_result(r: FrameMatchResultDTOV2, h: ObservedConstraintHypothesisV1, s: RuntimeSemanticRetrievalSnapshotV1) -> None:
        if (r.observed_hypothesis_ref, r.observed_hypothesis_hash) != (h.hypothesis_id, h.hypothesis_hash):
            raise ValueError("frame match result hypothesis binding mismatch")
        if (r.source_identity_ref, r.source_hash) != (h.source_binding.source_identity_ref, h.source_binding.source_hash):
            raise ValueError("frame match result source binding mismatch")
        if r.model_dump(mode="json", by_alias=True)["evidenceRefs"] != [
            item.model_dump(mode="json", by_alias=True) for item in h.evidence_refs
        ]:
            raise ValueError("frame match result EvidenceRefs differ from hypothesis")
        names = ("accepted_graph_release_ref", "accepted_graph_release_hash", "activation_set_snapshot_ref", "activation_set_snapshot_hash", "effective_context_resolver_hash")
        if tuple(getattr(r, n) for n in names) != tuple(getattr(h, n) for n in names):
            raise ValueError("frame match result authority pin mismatch")
        if (r.canonical_pattern_ref, r.canonical_pattern_hash) not in set(zip(h.retrieved_pattern_refs, h.retrieved_pattern_hashes)):
            raise ValueError("frame match result pattern is outside hypothesis pins")
        if r.governed_rule_version_ref not in s.ordered_governed_rule_version_refs:
            raise ValueError("frame match result rule is outside retrieval authority")
        if (r.activation_resolver_policy_version, r.resolved_activation_set_hash) != (
            s.activation_resolver_policy_version, s.resolved_activation_set_hash
        ):
            raise ValueError("frame match result activation authority mismatch")

    @staticmethod
    def _bind_finding(f: GovernedFrameFindingV1, r: FrameMatchResultDTOV2) -> None:
        pairs = (
            ("frame_match_result_ref", "result_id"), ("frame_match_result_hash", "result_hash"),
            ("organization_id", "organization_id"), ("workspace_id", "workspace_id"),
            ("semantic_relation", "semantic_relation"), ("frame_match_state", "frame_match_state"),
            ("observed_hypothesis_ref", "observed_hypothesis_ref"),
            ("observed_hypothesis_hash", "observed_hypothesis_hash"),
            ("source_identity_ref", "source_identity_ref"), ("source_hash", "source_hash"),
            ("canonical_pattern_ref", "canonical_pattern_ref"), ("canonical_pattern_hash", "canonical_pattern_hash"),
            ("accepted_graph_release_ref", "accepted_graph_release_ref"), ("accepted_graph_release_hash", "accepted_graph_release_hash"),
            ("governed_rule_version_ref", "governed_rule_version_ref"), ("governed_rule_version_hash", "governed_rule_version_hash"),
            ("activation_set_snapshot_ref", "activation_set_snapshot_ref"), ("activation_set_snapshot_hash", "activation_set_snapshot_hash"),
            ("activation_resolver_policy_version", "activation_resolver_policy_version"),
            ("effective_context_resolver_hash", "effective_context_resolver_hash"),
            ("resolved_activation_set_hash", "resolved_activation_set_hash"),
        )
        if any(getattr(f, left) != getattr(r, right) for left, right in pairs):
            raise ValueError("governed frame finding differs from frame match authority")
        if f.model_dump(mode="json", by_alias=True)["evidenceRefs"] != r.model_dump(mode="json", by_alias=True)["evidenceRefs"]:
            raise ValueError("governed frame finding EvidenceRefs differ from frame match")

    def _own(self, org: str, workspace: str) -> None:
        if org != self.organization_id: raise ValueError("runtime semantic artifact organization ownership mismatch")
        if workspace != self.workspace_id: raise ValueError("runtime semantic artifact workspace ownership mismatch")

    @staticmethod
    def _hash(value: str) -> None:
        if not isinstance(value, str) or _SHA.fullmatch(value) is None: raise ValueError("artifact hash must be an exact lowercase sha256 ref")

__all__ = ["RuntimeSemanticHypothesisArtifactStore"]
