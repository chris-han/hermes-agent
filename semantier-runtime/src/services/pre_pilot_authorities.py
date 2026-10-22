"""Load and persist the frozen PRE_PILOT authority set.

These artifacts content-address existing contracts and implementation files. They are
freeze records, not replacement evaluator, scorer, runtime, or collection owners.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from datetime import datetime

from contracts.experiment_integrity_snapshot import (
    PRE_PILOT_AUTHORITY_KINDS,
    ExperimentIntegritySnapshotV1,
    PrePilotAuthorityArtifactV1,
)
from semantier.cognitive_context.store import CognitiveContextStore

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AUTHORITY_DIR = _REPO_ROOT / "scripts/pilots/tender_decision_local_deepseek_v1/authority"
_FILES = (
    "evaluator.quality-vv.v1.json",
    "obligation_suite.quality-vv.v1.json",
    "pre_pilot_integrity_policy.quality-vv.v1.json",
    "runtime_contract.quality-vv.v1.json",
    "scorer_contract.v1.json",
    "collector_contract.quality-vv.v1.json",
)


def load_pre_pilot_authorities(
    *, organization_id: str, workspace_id: str,
) -> tuple[PrePilotAuthorityArtifactV1, ...]:
    authorities: list[PrePilotAuthorityArtifactV1] = []
    for name in _FILES:
        path = _AUTHORITY_DIR / name
        raw = json.loads(path.read_text(encoding="utf-8"))
        authority = PrePilotAuthorityArtifactV1.model_validate({
            **raw,
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "artifact_path": path.relative_to(_REPO_ROOT).as_posix(),
        })
        authorities.append(authority)
    result = tuple(authorities)
    observed_kinds = {item.authority_kind for item in result}
    expected_kinds = set(PRE_PILOT_AUTHORITY_KINDS.values())
    if len(observed_kinds) != len(result) or observed_kinds != expected_kinds:
        raise ValueError("PRE_PILOT_AUTHORITY_SET_INCOMPLETE_OR_DUPLICATED")
    if not validate_pre_pilot_authority_sources(result):
        raise ValueError("PRE_PILOT_AUTHORITY_SOURCE_DRIFT")
    return result


def persist_pre_pilot_authorities(
    store: CognitiveContextStore,
    *, organization_id: str,
    workspace_id: str,
) -> tuple[PrePilotAuthorityArtifactV1, ...]:
    authorities = load_pre_pilot_authorities(
        organization_id=organization_id, workspace_id=workspace_id,
    )
    return tuple(store.save_or_get_derived_projection(item) for item in authorities)


def create_pre_pilot_integrity_snapshot(
    authorities: tuple[PrePilotAuthorityArtifactV1, ...],
    *, organization_id: str,
    workspace_id: str,
    created_at: datetime,
) -> ExperimentIntegritySnapshotV1:
    by_kind = {item.authority_kind: item for item in authorities}
    if len(by_kind) != len(authorities) or set(by_kind) != set(PRE_PILOT_AUTHORITY_KINDS.values()):
        raise ValueError("PRE_PILOT_AUTHORITY_SET_INCOMPLETE_OR_DUPLICATED")
    if any((item.organization_id, item.workspace_id) != (organization_id, workspace_id)
           for item in authorities):
        raise ValueError("PRE_PILOT_AUTHORITY_SCOPE_MISMATCH")
    def pin(kind: str):
        return by_kind[kind].pin()
    rows = tuple({
        "artifact_kind": item.authority_kind,
        "artifact_ref": item.authority_ref,
        "artifact_hash": item.authority_hash,
        "payload": item.payload,
    } for item in authorities)
    from services.ontology_quality_profile_service import tender_mvl0_quality_profile
    profile = tender_mvl0_quality_profile(organization_id=organization_id, workspace_id=workspace_id)
    rows += ({**profile.pin().model_dump(), "payload": profile.model_dump(mode="json")},)
    return ExperimentIntegritySnapshotV1.create(
        quality_profile=profile.pin(),
        organization_id=organization_id,
        workspace_id=workspace_id,
        integrity_stage="PRE_PILOT",
        evaluator=pin(PRE_PILOT_AUTHORITY_KINDS["evaluator"]),
        obligation_suite=pin(PRE_PILOT_AUTHORITY_KINDS["obligation_suite"]),
        integrity_policy=pin(PRE_PILOT_AUTHORITY_KINDS["integrity_policy"]),
        runtime_contract=pin(PRE_PILOT_AUTHORITY_KINDS["runtime_contract"]),
        scorer_contract=pin(PRE_PILOT_AUTHORITY_KINDS["scorer_contract"]),
        collector_contract=pin(PRE_PILOT_AUTHORITY_KINDS["collector_contract"]),
        authority_payloads=rows,
        created_at=created_at,
    )


def validate_pre_pilot_authority_sources(
    authorities: tuple[PrePilotAuthorityArtifactV1, ...],
) -> bool:
    for authority in authorities:
        sources = authority.payload.get("implementation_sources")
        if not isinstance(sources, list) or not sources:
            return False
        for source in sources:
            if not isinstance(source, dict):
                return False
            relative_path = source.get("path")
            expected = source.get("sha256")
            if not isinstance(relative_path, str) or not isinstance(expected, str):
                return False
            candidate = (_REPO_ROOT / relative_path).resolve()
            if _REPO_ROOT not in candidate.parents or not candidate.is_file():
                return False
            observed = "sha256:" + sha256(candidate.read_bytes()).hexdigest()
            if observed != expected:
                return False
    return True


__all__ = [
    "load_pre_pilot_authorities",
    "persist_pre_pilot_authorities",
    "create_pre_pilot_integrity_snapshot",
    "validate_pre_pilot_authority_sources",
]
