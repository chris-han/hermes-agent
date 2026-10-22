"""Offline ontology audit export with no parser, retrieval, reasoner, or model calls."""

from __future__ import annotations

from contracts.ontology_audit import OntologyAuditArtifactV1, OntologyAuditExportV1
from contracts.ontology_building import OntologyBuildManifestV1
from contracts.ontology_build_inputs import OntologyBuildInputEnvelopeV1
from eos.artifact_hashing import content_hash
from storage.ontology_build_store import OntologyBuildStore


class OntologyAuditService:
    def export(
        self, *, store: OntologyBuildStore, organization_id: str,
        workspace_id: str, build_ref: str,
    ) -> OntologyAuditExportV1:
        manifest = store.get(
            organization_id=organization_id, workspace_id=workspace_id,
            artifact_ref=build_ref, contract_type=OntologyBuildManifestV1,
        )
        if manifest is None:
            raise LookupError("ontology build not found")
        archived = {}
        envelope_pins = [pin for pin in manifest.child_artifacts if pin.artifact_kind == "ontology_build_inputs"]
        if len(envelope_pins) > 1:
            raise ValueError("audit requires one exact source-first input envelope")
        for pin in envelope_pins:
            envelope = store.get(organization_id=organization_id, workspace_id=workspace_id,
                                 artifact_ref=pin.artifact_ref, contract_type=OntologyBuildInputEnvelopeV1)
            if envelope is None or (envelope.envelope_ref, envelope.envelope_hash) != (pin.artifact_ref, pin.artifact_hash):
                raise ValueError("pinned audit input envelope is missing or invalid")
            if (envelope.organization_id, envelope.workspace_id, envelope.build_ref) != (
                organization_id, workspace_id, manifest.build_ref
            ):
                raise ValueError("audit input envelope tenant/build mismatch")
            for item in envelope.audit_inputs():
                archived[(item.artifact_kind, item.artifact_ref, item.artifact_hash)] = item
            supplied = {(item.artifact_kind, item.artifact_ref, item.artifact_hash) for item in manifest.child_artifacts}
            if not set(archived) <= supplied:
                raise ValueError("manifest does not preserve exact archived input pins")
        artifacts = []
        for pin in manifest.child_artifacts:
            key = (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)
            if key in archived:
                artifacts.append(archived[key])
                continue
            raw = store.get_raw(
                organization_id=organization_id, workspace_id=workspace_id,
                artifact_ref=pin.artifact_ref,
            )
            if raw is None or raw["artifact_hash"] != pin.artifact_hash:
                raise ValueError(f"pinned audit artifact is missing or invalid: {pin.artifact_kind}")
            artifacts.append(OntologyAuditArtifactV1(
                artifact_kind=pin.artifact_kind, artifact_ref=pin.artifact_ref,
                artifact_hash=pin.artifact_hash, schema_version=raw["schema_version"],
                payload=raw["payload"],
            ))
        ordered = tuple(sorted(artifacts, key=lambda item: item.artifact_kind.encode("utf-8")))
        basis = {
            "organization_id": organization_id, "workspace_id": workspace_id,
            "build_ref": manifest.build_ref, "build_hash": manifest.build_hash,
            "artifacts": [item.model_dump(mode="json") for item in ordered],
            "replay_mode": "OFFLINE_PINNED_ONLY",
            "exported_at": manifest.created_at.isoformat().replace("+00:00", "Z"),
        }
        digest = content_hash(basis)
        return OntologyAuditExportV1(
            audit_ref=f"ontology-audit:{digest}", audit_hash=f"sha256:{digest}",
            **basis,
        )


__all__ = ["OntologyAuditService"]
