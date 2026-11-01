"""Package-only facade of the existing personal Skill candidate owner.

No host discovery, materialization, admission or activation is exposed here.
The existing candidate family, request custody and assessment owner are reused.
"""
from datetime import datetime, timezone
import json
import re
import yaml

from contracts.learning_trace_evidence import LearningTraceObservationV2
from contracts.learning_workbench import SkillArtifactVersionV1, SkillFileV1
from contracts.learned_skill_publication import LearnedSkillAssessmentV1
from contracts.ontology_building import ArtifactPinV1
from eos.artifact_hashing import content_hash
from services.learned_skill_publication_service import LearnedSkillPublicationService
from services.learning_trace_evidence_service import LearningTraceEvidenceService


class LearnedSkillPackageCustody(LearnedSkillPublicationService):
    """Internal package-input projection of the same owner, never a publisher."""

    def _scope(self, context):
        return LearningTraceEvidenceService.public_scope(context)

    @staticmethod
    def _metadata(skill_id, content):
        if not isinstance(content, str) or not content.strip() or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", skill_id):
            raise ValueError("INVALID_PROCEDURE_CHANGE")
        metadata = {}
        if content.startswith("---\n"):
            parts = content.split("\n---", 1)
            if len(parts) != 2:
                raise ValueError("INVALID_SKILL_FRONTMATTER")
            metadata = yaml.safe_load(parts[0][4:]) or {}
            if not isinstance(metadata, dict):
                raise ValueError("INVALID_SKILL_FRONTMATTER")
        if metadata.get("name", skill_id) != skill_id:
            raise ValueError("SKILL_IDENTITY_MISMATCH")
        return metadata

    def _check_content(self, context, skill_id, content):
        self._scope(context)
        return self._metadata(skill_id, content)

    def _assert_base(self, context, base):
        self._scope(context)
        active = self._active(context, base.skill_id)
        if active:
            payload = json.loads(active["content_yaml"])
            actual = payload.get("member_versions", {}).get(base.skill_id, payload.get("version"))
            if actual != base.pin().model_dump(mode="json"):
                raise ValueError("STALE_PREDECESSOR")
        elif base.provenance != "legacy":
            raise ValueError("STALE_PREDECESSOR")

    def _projection(self, context, candidate):
        candidate, version, base = self._objects(context, candidate.artifact_ref, candidate.artifact_hash)
        return {"candidate_ref": candidate.artifact_ref, "candidate_hash": candidate.artifact_hash,
            "version_ref": version.artifact_ref, "version_hash": version.artifact_hash,
            "base_ref": base.artifact_ref, "base_hash": base.artifact_hash,
            "package_diff": self.package_diff(base, version),
            "host_readiness": "NOT_EVALUATED", "admission": "NOT_ASSERTED",
            "activation": "NOT_ASSERTED", "available_actions": []}

    def capture_package(self, context, source_pin, request_id):
        self._scope(context)
        pin = ArtifactPinV1.model_validate(source_pin)
        source = self._load(context, pin.artifact_ref, LearningTraceObservationV2, pin.artifact_hash)
        if pin.artifact_kind != source.ARTIFACT_KIND or source.source_kind != "core_receipt" or source.event_kind != "package":
            raise ValueError("OWNER_PACKAGE_SOURCE_REQUIRED")
        skill_id = source.payload["skill_id"]
        files = tuple(SkillFileV1.model_validate(item) for item in source.payload["files"])
        content = next((f.content_bytes().decode("utf-8") for f in files if f.relative_path == "SKILL.md"), "")
        self._metadata(skill_id, content)
        request_hash = content_hash({"source": pin.model_dump(mode="json")})
        recovered = self._store.get_operation_result(**self._scope(context), actor_ref=context.user_id,
            operation="capture_learned_skill_package", request_id=request_id,
            request_hash=request_hash, artifact_kind="skill_artifact_version")
        if recovered:
            return self._load(context, recovered.artifact_ref, SkillArtifactVersionV1, recovered.artifact_hash)
        existing = [v for v in self._owned(context, SkillArtifactVersionV1, "skill_artifact_version.v1") if v.skill_id == skill_id]
        if self._active(context, skill_id) or any(v.files != files for v in existing):
            raise ValueError("STALE_PACKAGE_BASELINE")
        version = existing[0] if existing else SkillArtifactVersionV1.create(**self._scope(context),
            owner_ref=context.user_id, skill_id=skill_id, version=1, provenance="legacy",
            files=files, recorded_at=datetime.now(timezone.utc))
        winner = self._store.put_many_with_operation_result(artifacts=(version,), **self._scope(context),
            actor_ref=context.user_id, operation="capture_learned_skill_package", request_id=request_id,
            request_hash=request_hash, result_pin=version.pin())
        return self._load(context, winner.artifact_ref, SkillArtifactVersionV1, winner.artifact_hash)

    def propose_package(self, context, baseline_ref, baseline_hash, new_content, request_id, origin):
        base = self._load(context, baseline_ref, SkillArtifactVersionV1, baseline_hash)
        before = self._content(base)
        old_metadata = self._metadata(base.skill_id, before)
        new_metadata = self._metadata(base.skill_id, new_content)
        old_header = before.split("\n---",1)[0] if before.startswith("---\n") else ""
        new_header = new_content.split("\n---",1)[0] if new_content.startswith("---\n") else ""
        if old_header != new_header or old_metadata != new_metadata:
            raise ValueError("PACKAGE_METADATA_CHANGE_OUTSIDE_SCOPE")
        if not isinstance(origin, dict) or not origin.get("evidence_refs"):
            raise ValueError("OWNER_EVIDENCE_REQUIRED")
        for raw_pin in origin["evidence_refs"]:
            pin = ArtifactPinV1.model_validate(raw_pin)
            raw = self._store.get_raw(**self._scope(context), artifact_ref=pin.artifact_ref)
            if raw is None or raw["artifact_hash"] != pin.artifact_hash:
                raise ValueError("INVALID_CORRECTION_ORIGIN")
            owner = raw["payload"].get("owner_ref") or raw["payload"].get("profile", {}).get("owner_ref")
            if owner != context.user_id:
                raise PermissionError("FORBIDDEN_LEARNING_SCOPE")
        return super().propose(context, baseline_ref, baseline_hash, new_content, request_id, origin)

    def _classify_package(self, candidate, version, base):
        # This facade accepts only the already-validated body-only package delta.
        if tuple(f for f in base.files if f.relative_path != "SKILL.md") != tuple(f for f in version.files if f.relative_path != "SKILL.md"):
            raise ValueError("PACKAGE_RESOURCES_CHANGED")
        if self._metadata(base.skill_id, self._content(base)) != self._metadata(version.skill_id, self._content(version)):
            raise ValueError("PACKAGE_METADATA_CHANGE_OUTSIDE_SCOPE")
        from services.candidate_admission_classifier import CandidateAdmissionClassifier
        classification = CandidateAdmissionClassifier().classify_text(candidate=candidate.pin(),
            candidate_text=self._content(version), evidence_refs=(base.pin(), version.pin()))
        if re.search(r"!\x60|\{\{|\{%|\{#", self._content(version)):
            from contracts.candidate_admission_classification import CandidateAdmissionClassificationV1
            if classification.disposition != "AUTHORITY_VIOLATION":
                classification = CandidateAdmissionClassificationV1(**{**classification.model_dump(),
                    "disposition": "SEMANTIC_CHANGE_REQUIRES_ADMISSION", "reason_code": "PACKAGE_EFFECT_REQUIRES_ENGINEERING"})
        return classification

    def assess_package(self, context, candidate_ref, candidate_hash):
        candidate, version, base = self._objects(context, candidate_ref, candidate_hash)
        self._assert_base(context, base)
        classification = self._classify_package(candidate, version, base)
        row = LearnedSkillAssessmentV1.create(**self._scope(context), owner_ref=context.user_id,
            candidate=candidate.pin(), version=version.pin(), predecessor=base.pin(),
            classification=classification, eligible=classification.disposition == "PROCEDURAL_SAFE_TO_PROJECT",
            counterevidence=("Package-only structural assessment; host readiness, admission, native use and utility are not established.",),
            recorded_at=datetime.now(timezone.utc))
        self._store.put(row)
        return row

    def _host_operation_unavailable(self, *args, **kwargs):
        raise ValueError("PACKAGE_CUSTODY_IS_NOT_RUNTIME_PUBLICATION")

    propose = _host_operation_unavailable
    capture_baseline = _host_operation_unavailable
    inventory = _host_operation_unavailable
    inspect = _host_operation_unavailable
    approve = _host_operation_unavailable
    activate = _host_operation_unavailable
    publish = _host_operation_unavailable
    recover = _host_operation_unavailable
