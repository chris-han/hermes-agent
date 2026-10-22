"""Pure retained verification for advisory source-family artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256
import json
import re
from typing import Any, Callable, Mapping

from contracts.canonical_source_ir import (
    CanonicalSourceDocument,
    content_hash as canonical_ir_content_hash,
    source_block_identity,
)
from contracts.evidence_location import EvidenceRef, SourceIdentity
from contracts.graph_intent_brief import GraphIntentBrief
from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_context_reconstruction import CapturedJsonV1, CapturedSourceV1
from eos.artifact_hashing import canonical_json, content_hash
from semantier.knowledge_feature_packs.canonical_source_ir_adapter import (
    AmbiguousSourceMapping,
    resolve_unique_mapping,
)
from semantier.knowledge_feature_packs.evidence_selector_builder import (
    evidence_ref_identity,
    evidence_ref_for_block,
)

from .common import (
    PinKey,
    _captured_json,
    _expect_dependencies,
    _expect_identity,
    _pin_key,
    _payload,
    _require_relative_ref,
    _require_supplied_source,
    _strict_base64,
    _validate_closure,
)


_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


_GRAPH_INTENT_REF = re.compile(r"^graph-intent-[0-9a-f]{24}$")


_PUBLICATION_FIELDS = {
    "schema_version",
    "publication_kind",
    "organization_id",
    "workspace_id",
    "lineage_id",
    "graph_intent_ref",
    "graph_intent_hash",
    "source_identity_ref",
    "predecessor_graph_intent_ref",
    "artifacts",
}


_BINDING_FIELDS = {
    "schema_version",
    "organization_id",
    "workspace_id",
    "lineage_id",
    "graph_intent_ref",
    "graph_intent_hash",
    "source_identity",
    "source_identity_hash",
    "canonical_source_ir_ref",
    "canonical_source_ir_hash",
}


@dataclass(frozen=True, slots=True)
class _PublicationProof:
    manifest: dict[str, Any]
    brief: GraphIntentBrief
    identity_payload: dict[str, Any]
    link: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class _SourceProof:
    identity: SourceIdentity
    identity_payload: dict[str, Any]
    canonical_payload: dict[str, Any]
    canonical_hash: str
    evidence_payloads: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _SourceMemberView:
    payload: dict[str, Any]
    identity: SourceIdentity
    identity_payload: dict[str, Any]
    canonical: CanonicalSourceDocument


def _canonical_source_ir_ref(native_hash: str) -> str:
    digest = native_hash.removeprefix("sha256:")
    return f"artifacts/canonical-source-ir/{digest}/canonical_source_ir.v1.json"


def _expect_wrapper_identity(
    source: CapturedSourceV1, *, kind: str, prefix: str
) -> None:
    digest = source.payload.bytes_hash.removeprefix("sha256:")
    _expect_identity(
        source, kind=kind, ref=f"{prefix}:{digest}", digest=f"sha256:{digest}"
    )


def _publication_proof(
    source: CapturedSourceV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> _PublicationProof:
    source, payload = _payload(source)
    _expect_wrapper_identity(
        source,
        kind="advisory_graph_intent_publication_capture",
        prefix="advisory-graph-intent-publication-capture",
    )
    _expect_dependencies(source, (), closure)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "manifest",
        "artifacts",
    }:
        raise ValueError("invalid advisory graph intent publication capture shape")
    if payload["schema_version"] != "advisory_graph_intent_publication_capture.v1":
        raise ValueError("unsupported advisory graph intent publication capture")

    _manifest_capture, manifest = _captured_json(
        payload["manifest"], "graph intent publication manifest"
    )
    if not isinstance(manifest, dict) or set(manifest) != _PUBLICATION_FIELDS:
        raise ValueError("invalid graph intent publication manifest shape")
    if (
        manifest["schema_version"] != "graph_intent_publication_manifest.v1"
        or manifest["publication_kind"] != "graph_intent_brief_lifecycle"
        or any(
            not isinstance(manifest[field], str) or not manifest[field]
            for field in (
                "organization_id",
                "workspace_id",
                "lineage_id",
                "source_identity_ref",
            )
        )
        or _GRAPH_INTENT_REF.fullmatch(str(manifest["graph_intent_ref"])) is None
        or _SHA256.fullmatch(str(manifest["graph_intent_hash"])) is None
    ):
        raise ValueError("invalid native graph intent publication manifest")

    raw_artifacts = payload["artifacts"]
    manifest_artifacts = manifest["artifacts"]
    if not isinstance(raw_artifacts, list) or not isinstance(manifest_artifacts, list):
        raise ValueError("invalid graph intent publication artifact membership")
    manifest_by_role: dict[str, dict[str, Any]] = {}
    for item in manifest_artifacts:
        if not isinstance(item, dict) or set(item) != {
            "artifact_role",
            "artifact_path",
            "artifact_hash",
        }:
            raise ValueError("invalid graph intent publication manifest artifact")
        role = item["artifact_role"]
        if not isinstance(role, str) or role in manifest_by_role:
            raise ValueError("duplicate or invalid graph intent publication role")
        if _SHA256.fullmatch(str(item["artifact_hash"])) is None:
            raise ValueError("invalid graph intent publication artifact hash")
        manifest_by_role[role] = item
    if [item["artifact_role"] for item in manifest_artifacts] != sorted(
        manifest_by_role
    ):
        raise ValueError("graph intent publication manifest roles are not canonical")

    artifacts: dict[str, dict[str, Any]] = {}
    for item in raw_artifacts:
        if not isinstance(item, dict) or set(item) != {
            "role",
            "payload",
            "bytes_base64",
        }:
            raise ValueError("invalid retained graph intent publication artifact")
        role = item["role"]
        if not isinstance(role, str) or role in artifacts:
            raise ValueError("duplicate or invalid retained graph intent role")
        manifest_item = manifest_by_role.get(role)
        if manifest_item is None:
            raise ValueError("retained graph intent role is absent from manifest")
        captured, decoded = _captured_json(
            item["payload"], f"retained graph intent {role} payload"
        )
        raw = _strict_base64(
            item["bytes_base64"], f"retained graph intent {role} bytes"
        )
        if f"sha256:{sha256(raw).hexdigest()}" != manifest_item["artifact_hash"]:
            raise ValueError("retained graph intent original bytes hash mismatch")
        try:
            original = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "retained graph intent original bytes are invalid JSON"
            ) from exc
        if canonical_json(original) != captured.canonical_json or original != decoded:
            raise ValueError("retained graph intent original bytes payload mismatch")
        artifacts[role] = {"payload": decoded, "manifest": manifest_item}
    if [item["role"] for item in raw_artifacts] != sorted(artifacts):
        raise ValueError("retained graph intent roles are not canonical")

    predecessor_ref = manifest["predecessor_graph_intent_ref"]
    expected_roles = {"graph_intent_brief", "source_binding_evidence"}
    if predecessor_ref is not None:
        if _GRAPH_INTENT_REF.fullmatch(str(predecessor_ref)) is None:
            raise ValueError("invalid graph intent predecessor ref")
        expected_roles.add("successor_link")
    if set(artifacts) != expected_roles or set(manifest_by_role) != expected_roles:
        raise ValueError("retained W1 publication has missing or extra roles")

    brief_payload = artifacts["graph_intent_brief"]["payload"]
    if not isinstance(brief_payload, dict):
        raise ValueError("retained graph intent brief must be an object")
    brief = GraphIntentBrief.from_dict(brief_payload)
    if brief.as_dict() != brief_payload:
        raise ValueError("retained graph intent brief encoding mismatch")
    if (
        brief.graph_intent_id != manifest["graph_intent_ref"]
        or brief.graph_intent_hash != manifest["graph_intent_hash"]
        or brief.predecessor_graph_intent_ref != predecessor_ref
    ):
        raise ValueError("graph intent publication subject mismatch")

    digest = brief.graph_intent_hash.removeprefix("sha256:")
    expected_paths = {
        "graph_intent_brief": f"graph-intent-briefs/{digest}/graph_intent_brief.v1.json",
        "source_binding_evidence": f"graph-intent-briefs/{digest}/source_binding_evidence.v1.json",
    }
    if predecessor_ref is not None:
        expected_paths["successor_link"] = (
            f"graph-intent-briefs/successor-links/{predecessor_ref}/"
            f"{brief.graph_intent_id}.json"
        )
    if any(
        artifacts[role]["manifest"]["artifact_path"] != path
        for role, path in expected_paths.items()
    ):
        raise ValueError("graph intent publication native relative path mismatch")

    binding = artifacts["source_binding_evidence"]["payload"]
    if not isinstance(binding, dict) or set(binding) != _BINDING_FIELDS:
        raise ValueError("invalid graph intent source binding evidence shape")
    identity = SourceIdentity.model_validate(binding["source_identity"])
    identity_payload = identity.model_dump(
        mode="json", by_alias=True, exclude_none=False
    )
    if identity_payload != binding["source_identity"]:
        raise ValueError("graph intent source identity encoding mismatch")
    if identity.created_at.utcoffset() != timedelta(0):
        raise ValueError("graph intent source identity time must be UTC")
    _require_relative_ref(identity.artifact_ref, "source identity artifact_ref")
    native_binding = brief.source_binding
    binding_checks = {
        "schema_version": "graph_intent_source_binding_evidence.v1",
        "organization_id": native_binding.organization_id,
        "workspace_id": native_binding.workspace_id,
        "lineage_id": native_binding.lineage_id,
        "graph_intent_ref": brief.graph_intent_id,
        "graph_intent_hash": brief.graph_intent_hash,
        "source_identity_hash": "sha256:" + content_hash(identity_payload),
        "canonical_source_ir_ref": native_binding.canonical_source_ir_ref,
        "canonical_source_ir_hash": native_binding.canonical_source_ir_hash,
    }
    if any(binding.get(field) != value for field, value in binding_checks.items()):
        raise ValueError("graph intent source binding evidence mismatch")
    if (
        identity.source_identity_ref != native_binding.source_identity_ref
        or identity.organization_id != native_binding.organization_id
        or identity.workspace_id != native_binding.workspace_id
        or identity.source_hash != native_binding.source_hash
        or identity.source_version != native_binding.source_version
        or manifest["organization_id"] != native_binding.organization_id
        or manifest["workspace_id"] != native_binding.workspace_id
        or manifest["lineage_id"] != native_binding.lineage_id
        or manifest["source_identity_ref"] != native_binding.source_identity_ref
    ):
        raise ValueError("graph intent native source binding mismatch")
    _require_relative_ref(
        native_binding.canonical_source_ir_ref, "canonical_source_ir_ref"
    )

    link: dict[str, Any] | None = None
    if predecessor_ref is None:
        if brief.review.state != "proposed":
            raise ValueError("retained W1 root must be proposed")
    else:
        if brief.review.state != "accepted":
            raise ValueError("retained W1 successor must be accepted")
        link = artifacts["successor_link"]["payload"]
        if not isinstance(link, dict) or set(link) != {
            "schema_version",
            "predecessor_graph_intent_ref",
            "predecessor_graph_intent_hash",
            "successor_graph_intent_ref",
            "successor_graph_intent_hash",
        }:
            raise ValueError("invalid retained graph intent successor link")
        if (
            link["schema_version"] != "graph_intent_successor_link.v1"
            or link["predecessor_graph_intent_ref"] != predecessor_ref
            or _SHA256.fullmatch(str(link["predecessor_graph_intent_hash"])) is None
            or link["successor_graph_intent_ref"] != brief.graph_intent_id
            or link["successor_graph_intent_hash"] != brief.graph_intent_hash
        ):
            raise ValueError("retained graph intent successor link mismatch")
    return _PublicationProof(manifest, brief, identity_payload, link)


def _source_proof(
    source: CapturedSourceV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> _SourceProof:
    view = _source_member_view(source, closure)
    payload = view.payload
    identity = view.identity
    identity_payload = view.identity_payload
    canonical = view.canonical
    canonical_payload = payload["canonical_source_ir"]
    canonical_hash = canonical_ir_content_hash(canonical_payload)
    mapped_ids = [
        block_id
        for mapping in canonical.source_map.mappings
        for block_id in mapping.block_ids
    ]
    block_ids = [block.block_id for block in canonical.blocks]
    if (
        not block_ids
        or len(block_ids) != len(set(block_ids))
        or len(mapped_ids) != len(set(mapped_ids))
        or set(mapped_ids) != set(block_ids)
    ):
        raise ValueError("advisory canonical source IR membership mismatch")

    expected_evidence: list[dict[str, Any]] = []
    for block in canonical.blocks:
        if block.content_hash != canonical_ir_content_hash(block.content):
            raise ValueError("advisory canonical source block content hash mismatch")
        try:
            locator = resolve_unique_mapping(canonical, block.block_id)
        except AmbiguousSourceMapping as exc:
            raise ValueError("advisory canonical source locator mismatch") from exc
        if (
            locator.source_identity_ref != identity.source_identity_ref
            or locator.source_hash != identity.source_hash
            or locator.media_type != identity.media_type
            or block.block_id
            != source_block_identity(identity.source_hash, locator, block.block_type)
        ):
            raise ValueError("advisory canonical source locator or block mismatch")
        evidence = evidence_ref_for_block(
            canonical, block, source_identity=identity, created_at=identity.created_at
        )
        if evidence.created_at.utcoffset() != timedelta(0):
            raise ValueError("advisory evidence time must be UTC")
        expected_evidence.append(
            evidence.model_dump(mode="json", by_alias=True, exclude_none=False)
        )
    if payload["evidence"] != expected_evidence:
        raise ValueError("advisory source evidence membership mismatch")
    return _SourceProof(
        identity,
        identity_payload,
        canonical_payload,
        canonical_hash,
        tuple(expected_evidence),
    )


def _source_member_payload(payload: Any) -> _SourceMemberView:
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "schema_version",
            "identity",
            "canonical_source_ir",
            "source_bytes_base64",
            "evidence",
        }
        or payload["schema_version"] != "advisory_source_capture.v1"
    ):
        raise ValueError("invalid advisory source capture shape")
    identity = SourceIdentity.model_validate(payload["identity"])
    identity_payload = identity.model_dump(
        mode="json", by_alias=True, exclude_none=False
    )
    if identity_payload != payload[
        "identity"
    ] or identity.created_at.utcoffset() != timedelta(0):
        raise ValueError("advisory source identity encoding mismatch")
    _require_relative_ref(identity.artifact_ref, "source identity artifact_ref")
    source_bytes = _strict_base64(
        payload["source_bytes_base64"], "original source bytes"
    )
    if f"sha256:{sha256(source_bytes).hexdigest()}" != identity.source_hash:
        raise ValueError("advisory original source bytes hash mismatch")
    canonical = CanonicalSourceDocument.model_validate(payload["canonical_source_ir"])
    if (
        canonical.model_dump(mode="json", by_alias=True, exclude_none=False)
        != payload["canonical_source_ir"]
        or canonical.source_identity_ref != identity.source_identity_ref
    ):
        raise ValueError("advisory canonical source IR encoding mismatch")
    if not isinstance(payload["evidence"], list):
        raise ValueError("advisory source evidence membership mismatch")
    return _SourceMemberView(payload, identity, identity_payload, canonical)


def _source_member_value(source: CapturedJsonV1) -> _SourceMemberView:
    """Validate one exact live source-capture value without retained wrappers."""

    _captured, payload = _captured_json(source, "advisory source capture")
    return _source_member_payload(payload)


def _source_member_view(
    source: CapturedSourceV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> _SourceMemberView:
    """Validate the common source basis without rewalking unrelated members."""
    source, payload = _payload(source)
    _expect_wrapper_identity(
        source, kind="advisory_source_capture", prefix="advisory-source-capture"
    )
    _expect_dependencies(source, (), closure)
    return _source_member_payload(payload)


def _matching_source_member_capture(
    *,
    closure: Mapping[PinKey, CapturedSourceV1],
    predicate: Callable[[_SourceMemberView], bool],
    label: str,
) -> tuple[CapturedSourceV1, _SourceMemberView]:
    matches: list[tuple[CapturedSourceV1, _SourceMemberView]] = []
    for candidate in closure.values():
        if candidate.source.artifact_kind != "advisory_source_capture":
            continue
        view = _source_member_view(candidate, closure)
        if predicate(view):
            matches.append((candidate, view))
    if len(matches) != 1:
        raise ValueError(f"{label} must resolve to exactly one advisory source capture")
    return matches[0]


def verify_advisory_graph_intent_publication_capture_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify retained W1 committed-manifest and original-role proof."""
    closure = _validate_closure(closure)
    source = _require_supplied_source(source, closure)
    _publication_proof(source, closure)
    return ()


def verify_advisory_source_capture_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify retained original bytes, canonical IR and exact evidence members."""
    closure = _validate_closure(closure)
    source = _require_supplied_source(source, closure)
    _source_proof(source, closure)
    return ()


def verify_source_identity_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify one exact aliased SourceIdentity member of its source capture."""
    closure = _validate_closure(closure)
    source, payload = _payload(_require_supplied_source(source, closure))
    identity = SourceIdentity.model_validate(payload)
    identity_payload = identity.model_dump(
        mode="json", by_alias=True, exclude_none=False
    )
    if identity_payload != payload or identity.created_at.utcoffset() != timedelta(0):
        raise ValueError("retained source identity member encoding mismatch")
    _require_relative_ref(identity.artifact_ref, "source identity artifact_ref")
    _expect_identity(
        source,
        kind="source_identity",
        ref=identity.source_identity_ref,
        digest="sha256:" + content_hash(identity_payload),
    )
    owner, _view = _matching_source_member_capture(
        closure=closure,
        predicate=lambda view: view.identity_payload == identity_payload,
        label="source identity member",
    )
    return _expect_dependencies(source, (owner.source,), closure)


def verify_canonical_source_ir_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify one exact canonical IR member and its governed native ref."""
    closure = _validate_closure(closure)
    source, payload = _payload(_require_supplied_source(source, closure))
    canonical = CanonicalSourceDocument.model_validate(payload)
    canonical_payload = canonical.model_dump(
        mode="json", by_alias=True, exclude_none=False
    )
    if canonical_payload != payload:
        raise ValueError("retained canonical source IR member encoding mismatch")
    native_hash = canonical_ir_content_hash(canonical_payload)
    _expect_identity(
        source,
        kind="canonical_source_ir",
        ref=_canonical_source_ir_ref(native_hash),
        digest=native_hash,
    )
    _require_relative_ref(source.source.artifact_ref, "canonical_source_ir_ref")
    owner, _view = _matching_source_member_capture(
        closure=closure,
        predicate=lambda candidate: (
            candidate.payload["canonical_source_ir"] == canonical_payload
        ),
        label="canonical source IR member",
    )
    return _expect_dependencies(source, (owner.source,), closure)


def verify_evidence_ref_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify one exact native EvidenceRef member of its source capture."""
    closure = _validate_closure(closure)
    source, payload = _payload(_require_supplied_source(source, closure))
    evidence = EvidenceRef.model_validate(payload)
    evidence_payload = evidence.model_dump(
        mode="json", by_alias=True, exclude_none=False
    )
    if evidence_payload != payload or evidence.created_at.utcoffset() != timedelta(0):
        raise ValueError("retained evidence member encoding mismatch")
    _expect_identity(
        source,
        kind="evidence_ref",
        ref=evidence.evidence_ref,
        digest="sha256:" + content_hash(evidence_payload),
    )
    matches: list[CapturedSourceV1] = []
    for candidate in closure.values():
        if candidate.source.artifact_kind != "advisory_source_capture":
            continue
        view = _source_member_view(candidate, closure)
        if view.payload["evidence"].count(evidence_payload) != 1:
            continue
        block_ref = evidence.provenance.get("source_block_id")
        blocks = [
            block for block in view.canonical.blocks if block.block_id == block_ref
        ]
        if len(blocks) != 1:
            raise ValueError("evidence member source block mismatch")
        block = blocks[0]
        if block.content_hash != canonical_ir_content_hash(block.content):
            raise ValueError("evidence member source block content mismatch")
        try:
            locator = resolve_unique_mapping(view.canonical, block.block_id)
        except AmbiguousSourceMapping as exc:
            raise ValueError("evidence member source locator mismatch") from exc
        if (
            locator.source_identity_ref != view.identity.source_identity_ref
            or locator.source_hash != view.identity.source_hash
            or locator.media_type != view.identity.media_type
            or block.block_id
            != source_block_identity(
                view.identity.source_hash, locator, block.block_type
            )
        ):
            raise ValueError("evidence member source locator mismatch")
        expected = evidence_ref_for_block(
            view.canonical,
            block,
            source_identity=view.identity,
            created_at=view.identity.created_at,
        ).model_dump(mode="json", by_alias=True, exclude_none=False)
        if expected != evidence_payload:
            raise ValueError("evidence member native selector mismatch")
        matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(
            "evidence member must resolve to exactly one advisory source capture"
        )
    if (
        evidence.selector_hash != evidence.selector.selector_hash()
        or evidence.evidence_ref
        != evidence_ref_identity(
            organization_id=evidence.organization_id,
            workspace_id=evidence.workspace_id,
            source_identity_ref=evidence.source_identity_ref,
            selector_hash=evidence.selector_hash,
            evidence_role=evidence.evidence_role,
        )
    ):
        raise ValueError("evidence member native identity mismatch")
    owner = matches[0]
    return _expect_dependencies(source, (owner.source,), closure)


def _verify_graph_intent(
    *,
    source: CapturedSourceV1,
    closure: Mapping[PinKey, CapturedSourceV1],
) -> tuple[ArtifactPinV1, ...]:
    source, payload = _payload(source)
    current_key = _pin_key(source.source)
    if not isinstance(payload, dict):
        raise ValueError("retained GraphIntent brief must be an object")
    brief = GraphIntentBrief.from_dict(payload)
    if brief.as_dict() != payload:
        raise ValueError("retained GraphIntent brief encoding mismatch")
    _expect_identity(
        source,
        kind="graph_intent_brief",
        ref=brief.graph_intent_id,
        digest=brief.graph_intent_hash,
    )

    publications: list[tuple[CapturedSourceV1, _PublicationProof]] = []
    for candidate in closure.values():
        if (
            candidate.source.artifact_kind
            != "advisory_graph_intent_publication_capture"
        ):
            continue
        proof = _publication_proof(candidate, closure)
        if proof.brief.as_dict() == brief.as_dict():
            publications.append((candidate, proof))
    if len(publications) != 1:
        raise ValueError("GraphIntent must resolve to exactly one publication capture")
    publication_source, publication = publications[0]
    source_capture, source_view = _matching_source_member_capture(
        closure=closure,
        predicate=lambda view: (
            view.identity_payload == publication.identity_payload
            and canonical_ir_content_hash(view.payload["canonical_source_ir"])
            == brief.source_binding.canonical_source_ir_hash
        ),
        label="GraphIntent native source",
    )
    if (
        source_view.identity.source_identity_ref
        != brief.source_binding.source_identity_ref
        or source_view.identity.source_hash != brief.source_binding.source_hash
        or source_view.identity.source_version != brief.source_binding.source_version
    ):
        raise ValueError("GraphIntent source capture binding mismatch")
    if brief.source_binding.canonical_source_ir_ref != _canonical_source_ir_ref(
        brief.source_binding.canonical_source_ir_hash
    ):
        raise ValueError("GraphIntent canonical source IR ref mismatch")

    dependencies = [publication_source.source, source_capture.source]
    if publication.link is None:
        if brief.predecessor_graph_intent_ref is not None:
            raise ValueError("GraphIntent root predecessor mismatch")
    else:
        parent_key = (
            "graph_intent_brief",
            publication.link["predecessor_graph_intent_ref"],
            publication.link["predecessor_graph_intent_hash"],
        )
        parent = closure.get(parent_key)
        if parent is None:
            raise ValueError("retained GraphIntent parent is missing")
        if parent_key == current_key:
            raise ValueError("retained GraphIntent predecessor cycle")
        parent, parent_payload = _payload(parent)
        if not isinstance(parent_payload, dict):
            raise ValueError("retained GraphIntent parent must be an object")
        parent_brief = GraphIntentBrief.from_dict(parent_payload)
        if parent_brief.as_dict() != parent_payload:
            raise ValueError("retained GraphIntent parent encoding mismatch")
        _expect_identity(
            parent,
            kind="graph_intent_brief",
            ref=parent_brief.graph_intent_id,
            digest=parent_brief.graph_intent_hash,
        )
        if (
            parent_brief.predecessor_graph_intent_ref is not None
            or parent_brief.review.state != "proposed"
            or parent_brief.source_binding != brief.source_binding
        ):
            raise ValueError("retained GraphIntent accepted parent mismatch")
        parent_semantics = parent_brief.as_dict()
        child_semantics = brief.as_dict()
        for candidate in (parent_semantics, child_semantics):
            for field in (
                "graph_intent_id",
                "graph_intent_hash",
                "review",
                "predecessor_graph_intent_ref",
                "successor_graph_intent_refs",
            ):
                candidate.pop(field)
        if parent_semantics != child_semantics:
            raise ValueError("retained GraphIntent accepted review changed semantics")
        dependencies.append(parent.source)
    return _expect_dependencies(source, tuple(dependencies), closure)


def verify_graph_intent_brief_v1(
    *, source: CapturedSourceV1, closure: Mapping[PinKey, CapturedSourceV1]
) -> tuple[ArtifactPinV1, ...]:
    """Verify native GraphIntent identity and its acyclic W1 proof closure."""
    closure = _validate_closure(closure)
    source = _require_supplied_source(source, closure)
    return _verify_graph_intent(source=source, closure=closure)
