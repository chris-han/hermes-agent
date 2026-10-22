"""Append-only governed persistence for material-correction reruns."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from contracts.corpus_graph_schema import CorpusGraphSchemaCandidate
from contracts.governed_extract import ExtractResult
from contracts.material_correction_rerun import (
    MaterialCorrectionRerun,
    build_schema_feedback_retest,
    material_correction_outcome_reason,
    validate_material_correction_material,
)
from contracts.source_semantic_witness import (
    SourceSemanticWitnessSetManifestV1,
    SourceSemanticWitnessV1,
)
from eos.artifact_hashing import content_hash
from storage.atomic_json_artifact import publish_immutable_json


_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_RERUN_REF = re.compile(r"^material-correction-rerun:[0-9a-f]{24}$")


class MaterialCorrectionRerunStore:
    """Publish evidence first and expose authority only when the envelope exists."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        organization_id: str,
        workspace_id: str,
        discovery_service: object,
    ) -> None:
        for label, value in (
            ("organization_id", organization_id),
            ("workspace_id", workspace_id),
        ):
            if (
                not isinstance(value, str)
                or not value.isascii()
                or _SAFE_SEGMENT.fullmatch(value) is None
                or value in {".", ".."}
            ):
                raise ValueError(f"{label} must be a safe single-segment ASCII identifier")
        from services.graph_schema_discovery_service import GraphSchemaDiscoveryService

        if type(discovery_service) is not GraphSchemaDiscoveryService:
            raise ValueError("material rerun requires exact governed discovery authority")
        self.organization_id = organization_id
        self.workspace_id = workspace_id
        self.discovery_service = discovery_service
        resolved_root = Path(workspace_root).resolve()
        self.workspace_path = (
            resolved_root
            if resolved_root.name == workspace_id
            and resolved_root.parent.name == "workspaces"
            else resolved_root / "workspaces" / workspace_id
        ).resolve()
        self.root = (
            self.workspace_path
            / "artifacts"
            / "material-correction-reruns"
        ).resolve()

    def put_pending(
        self,
        rerun: MaterialCorrectionRerun,
        *,
        proposed_positive_witnesses: list[dict[str, Any]],
        proposed_hard_negative_witnesses: list[dict[str, Any]],
        proposed_positive_manifest: dict[str, Any],
        proposed_hard_negative_manifest: dict[str, Any],
        correction_material: dict[str, Any],
    ) -> dict[str, Any]:
        if rerun.outcome != "pending_adjudication":
            raise ValueError("put_pending requires pending_adjudication")
        self._require_ownership(rerun)
        positive = self._witnesses(proposed_positive_witnesses, "positive_exemplar", 32)
        negative = self._witnesses(proposed_hard_negative_witnesses, "hard_negative", 6)
        positive_manifest = SourceSemanticWitnessSetManifestV1.model_validate(
            proposed_positive_manifest
        )
        negative_manifest = SourceSemanticWitnessSetManifestV1.model_validate(
            proposed_hard_negative_manifest
        )
        self._require_manifest(rerun.after or {}, positive_manifest, positive, "positive")
        self._require_manifest(rerun.after or {}, negative_manifest, negative, "hard_negative")
        material = validate_material_correction_material(correction_material)
        self._require_correction_material(rerun, material)
        self._require_reason_matches_material(rerun, material)
        self._validate_cross_binding(
            rerun,
            correction_material=material,
            pending=(positive, negative, positive_manifest, negative_manifest),
        )
        publish_immutable_json(
            self._correction_material_path(rerun.correction_material_ref),
            material,
            conflict_error="MATERIAL_CORRECTION_MATERIAL_CONFLICT",
        )
        artifact_root = self._artifact_root(rerun.rerun_hash)
        for directory, witnesses in (
            ("positive_witnesses", positive),
            ("hard_negative_witnesses", negative),
        ):
            for index, witness in enumerate(witnesses):
                publish_immutable_json(
                    artifact_root / directory / f"{index:02d}.json",
                    witness.model_dump(mode="json", by_alias=True),
                    conflict_error="MATERIAL_CORRECTION_RERUN_ARTIFACT_CONFLICT",
                )
        for name, payload in (
            ("positive_manifest.json", proposed_positive_manifest),
            ("hard_negative_manifest.json", proposed_hard_negative_manifest),
        ):
            publish_immutable_json(
                artifact_root / name,
                payload,
                conflict_error="MATERIAL_CORRECTION_RERUN_ARTIFACT_CONFLICT",
            )
        # The envelope is the commit marker and is deliberately published last.
        return publish_immutable_json(
            artifact_root / "material_correction_rerun.v1.json",
            rerun.as_dict(),
            conflict_error="MATERIAL_CORRECTION_RERUN_ARTIFACT_CONFLICT",
        )

    def put_accepted(
        self,
        rerun: MaterialCorrectionRerun,
        *,
        correction_material: dict[str, Any],
        comparison: dict[str, Any],
    ) -> dict[str, Any]:
        if rerun.outcome != "accepted":
            raise ValueError("put_accepted requires accepted outcome")
        self._require_ownership(rerun)
        material = validate_material_correction_material(correction_material)
        self._require_correction_material(rerun, material)
        self._require_reason_matches_material(rerun, material)
        persisted_comparison = self._load_comparison(rerun)
        if persisted_comparison != comparison:
            raise ValueError("accepted rerun comparison publication mismatch")
        self._validate_accepted_cross_binding(
            rerun, material, persisted_comparison
        )
        publish_immutable_json(
            self._correction_material_path(rerun.correction_material_ref),
            material,
            conflict_error="MATERIAL_CORRECTION_MATERIAL_CONFLICT",
        )
        return publish_immutable_json(
            self._artifact_root(rerun.rerun_hash)
            / "material_correction_rerun.v1.json",
            rerun.as_dict(),
            conflict_error="MATERIAL_CORRECTION_RERUN_ARTIFACT_CONFLICT",
        )

    def put_route_only(
        self,
        rerun: MaterialCorrectionRerun,
        *,
        correction_material: dict[str, Any],
    ) -> dict[str, Any]:
        if rerun.outcome != "route_only":
            raise ValueError("put_route_only requires route_only outcome")
        self._require_ownership(rerun)
        material = validate_material_correction_material(correction_material)
        self._require_correction_material(rerun, material)
        self._require_reason_matches_material(rerun, material)
        self._validate_route_only_cross_binding(rerun, material)
        publish_immutable_json(
            self._correction_material_path(rerun.correction_material_ref),
            material,
            conflict_error="MATERIAL_CORRECTION_MATERIAL_CONFLICT",
        )
        return publish_immutable_json(
            self._artifact_root(rerun.rerun_hash)
            / "material_correction_rerun.v1.json",
            rerun.as_dict(),
            conflict_error="MATERIAL_CORRECTION_RERUN_ARTIFACT_CONFLICT",
        )

    def get(self, rerun_ref: str, *, rerun_hash: str) -> dict[str, Any]:
        if _RERUN_REF.fullmatch(rerun_ref) is None:
            raise ValueError("rerun_ref must be an exact material rerun ref")
        if _SHA256.fullmatch(rerun_hash) is None:
            raise ValueError("rerun_hash must be an exact lowercase sha256 ref")
        artifact_root = self._artifact_root(rerun_hash)
        envelope_path = artifact_root / "material_correction_rerun.v1.json"
        if not envelope_path.is_file():
            raise ValueError("material correction rerun is not committed")
        rerun = MaterialCorrectionRerun.from_dict(
            json.loads(envelope_path.read_text(encoding="utf-8"))
        )
        if rerun.rerun_ref != rerun_ref or rerun.rerun_hash != rerun_hash:
            raise ValueError("material correction rerun identity mismatch")
        self._require_ownership(rerun)
        from storage.schema_learning_feedback_store import SchemaLearningFeedbackStore

        feedback = SchemaLearningFeedbackStore(
            self.root.parents[3], workspace_id=self.workspace_id
        ).get(rerun.feedback_ref)
        if f"sha256:{content_hash(feedback)}" != rerun.feedback_hash:
            raise ValueError("material correction rerun feedback pin mismatch")
        material_path = self._correction_material_path(rerun.correction_material_ref)
        if not material_path.is_file():
            raise ValueError("material correction material artifact is missing")
        material = validate_material_correction_material(
            json.loads(material_path.read_text(encoding="utf-8"))
        )
        self._require_correction_material(rerun, material)
        self._require_reason_matches_material(rerun, material)
        if rerun.outcome == "pending_adjudication":
            pending = self._replay_pending(artifact_root, rerun)
            self._validate_cross_binding(
                rerun,
                correction_material=material,
                pending=pending,
            )
        elif rerun.outcome == "accepted":
            comparison = self._load_comparison(rerun)
            self._validate_accepted_cross_binding(rerun, material, comparison)
        else:
            self._validate_route_only_cross_binding(rerun, material)
        return rerun.as_dict()

    def get_pending_evidence(
        self, rerun_ref: str, *, rerun_hash: str
    ) -> dict[str, Any]:
        """Replay the committed pending envelope and its exact witness payloads."""

        payload = self.get(rerun_ref, rerun_hash=rerun_hash)
        rerun = MaterialCorrectionRerun.from_dict(payload)
        if rerun.outcome != "pending_adjudication":
            raise ValueError("material correction rerun is not pending adjudication")
        positive, negative, positive_manifest, negative_manifest = (
            self._replay_pending(self._artifact_root(rerun.rerun_hash), rerun)
        )
        correction_material = validate_material_correction_material(
            json.loads(
                self._correction_material_path(rerun.correction_material_ref).read_text(
                    encoding="utf-8"
                )
            )
        )
        return {
            "rerun": payload,
            "correction_material": correction_material,
            "proposed_positive_witnesses": tuple(positive),
            "proposed_hard_negative_witnesses": tuple(negative),
            "proposed_positive_manifest": positive_manifest,
            "proposed_hard_negative_manifest": negative_manifest,
        }

    def _validate_route_only_cross_binding(
        self, rerun: MaterialCorrectionRerun, material: dict[str, Any]
    ) -> None:
        _intent, candidate, decision = self._bound_state("before", rerun.before)
        self._require_before_gate(rerun.before, candidate, decision)
        if (
            material.get("feedback_ref") != rerun.feedback_ref
            or material.get("feedback_hash") != rerun.feedback_hash
        ):
            raise ValueError("route-only material feedback binding mismatch")
        from services.relation_usage_grounding_service import (
            _create_relation_usage_grounding_authority,
        )
        from storage.human_grounding_store import HumanGroundingStore
        from semantier.knowledge_feature_packs.falkordb_backend import (
            InMemoryGraphBackend,
        )

        authority = _create_relation_usage_grounding_authority(
            self.root.parents[3],
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
            profile_registry=self.discovery_service.profile_registry,
            profile_relation_type_proposal_store=(
                self.discovery_service.profile_relation_type_proposal_store
            ),
        )
        grounding = HumanGroundingStore(
            self.root.parents[3] / "eos.db",
            graph_backend=InMemoryGraphBackend(),
            _relation_usage_grounding_authority=authority,
        ).get_relation_usage_grounding_for_feedback(
            rerun.feedback_ref,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
        )
        expected = {
            "schema_version": "relation_usage_route_material.v1",
            "feedback_ref": rerun.feedback_ref,
            "feedback_hash": rerun.feedback_hash,
            "grounding_decision_ref": grounding["grounding_decision_ref"],
            "grounding_decision_hash": grounding["grounding_decision_hash"],
            "finding_kind": grounding["finding_kind"],
            "required_route": grounding["required_route"],
            "subject_ref": grounding["subject_binding"]["subject_ref"],
            "subject_hash": grounding["subject_binding"]["subject_hash"],
        }
        if material != expected:
            raise ValueError("route-only material does not match governed Grounding")
        if (
            grounding["schema_decision_binding"]["schema_decision_ref"]
            != decision.schema_decision_id
            or grounding["schema_decision_binding"]["schema_decision_hash"]
            != decision.schema_decision_hash
        ):
            raise ValueError("route-only Grounding Decision binding mismatch")

    def _load_comparison(self, rerun: MaterialCorrectionRerun) -> dict[str, Any]:
        if rerun.comparison_ref is None or rerun.comparison_hash is None:
            raise ValueError("accepted rerun comparison pins are missing")
        path = (
            self.workspace_path
            / "artifacts/schema-discovery/feedback-retests"
            / f"{rerun.comparison_ref}.json"
        ).resolve()
        root = (
            self.workspace_path
            / "artifacts/schema-discovery/feedback-retests"
        ).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("accepted rerun comparison artifact is missing")
        comparison = json.loads(path.read_text(encoding="utf-8"))
        if (
            comparison.get("comparison_id") != rerun.comparison_ref
            or comparison.get("comparison_hash") != rerun.comparison_hash
            or f"sha256:{content_hash({key: value for key, value in comparison.items() if key not in {'comparison_id', 'comparison_hash'}})}"
            != rerun.comparison_hash
        ):
            raise ValueError("accepted rerun comparison identity mismatch")
        return comparison

    def _validate_accepted_cross_binding(
        self,
        rerun: MaterialCorrectionRerun,
        material: dict[str, Any],
        comparison: dict[str, Any],
    ) -> None:
        before_intent, before_candidate, before_decision = self._bound_state(
            "before", rerun.before
        )
        after_intent, after_candidate, after_decision = self._bound_state(
            "after", rerun.after or {}
        )
        graph_intent_correction = (
            material.get("schema_version")
            == "graph_intent_correction_material.v1"
        )
        intent_lineage_valid = (
            after_intent.predecessor_graph_intent_ref
            == before_intent.graph_intent_id
            if graph_intent_correction
            else after_intent == before_intent
        )
        if (
            not intent_lineage_valid
            or after_candidate is None
            or after_decision is None
            or after_candidate.predecessor_candidate_ref
            != before_candidate.candidate_id
            or after_decision.predecessor_schema_decision_ref
            != before_decision.schema_decision_id
        ):
            raise ValueError("accepted rerun successor lineage mismatch")
        self._require_before_gate(rerun.before, before_candidate, before_decision)
        self._require_before_gate(rerun.after or {}, after_candidate, after_decision)
        changes = material.get("changes")
        if not isinstance(changes, dict):
            raise ValueError("accepted rerun candidate correction material is invalid")
        if graph_intent_correction:
            if any(
                getattr(after_intent, field).as_dict() != value
                for field, value in changes.items()
            ):
                raise ValueError(
                    "accepted rerun GraphIntent does not match correction material"
                )
            if (
                after_candidate.graph_intent_binding
                != {
                    "graph_intent_ref": after_intent.graph_intent_id,
                    "graph_intent_hash": after_intent.graph_intent_hash,
                    "review_state": after_intent.review.state,
                    "source_identity_ref": after_intent.source_binding.source_identity_ref,
                }
                or after_candidate.source_binding != before_candidate.source_binding
                or after_candidate.compiler_provenance.get("taxonomy_ref")
                != before_candidate.compiler_provenance.get("taxonomy_ref")
                or after_candidate.compiler_provenance.get("taxonomy_hash")
                != before_candidate.compiler_provenance.get("taxonomy_hash")
            ):
                raise ValueError("accepted GraphIntent rerun authority binding mismatch")
        else:
            expected = before_candidate.as_dict()
            for field in ("schema_version", "candidate_id", "candidate_hash"):
                expected.pop(field)
            expected.update(changes)
            expected["predecessor_candidate_ref"] = before_candidate.candidate_id
            expected["successor_candidate_refs"] = []
            expected["clarification_refs"] = list(
                dict.fromkeys(
                    [*expected.get("clarification_refs", []), rerun.feedback_ref]
                )
            )
            expected_candidate = CorpusGraphSchemaCandidate.create_v4(expected)
            if expected_candidate.as_dict() != after_candidate.as_dict():
                raise ValueError(
                    "accepted rerun candidate does not match correction material"
                )
        if (
            comparison.get("baseline_schema_decision_ref")
            != before_decision.schema_decision_id
            or comparison.get("baseline_schema_decision_hash")
            != before_decision.schema_decision_hash
            or comparison.get("successor_schema_decision_ref")
            != after_decision.schema_decision_id
            or comparison.get("successor_schema_decision_hash")
            != after_decision.schema_decision_hash
            or comparison.get("source_binding") != before_decision.source_binding
        ):
            raise ValueError("accepted rerun comparison binding mismatch")
        baseline_extract = self._load_extract_result(
            str(comparison.get("baseline_extract_ref") or ""),
            str(comparison.get("baseline_extract_hash") or ""),
        )
        successor_extract = self._load_extract_result(
            str(comparison.get("successor_extract_ref") or ""),
            str(comparison.get("successor_extract_hash") or ""),
        )
        self._require_extract_binding(
            baseline_extract, decision=before_decision, candidate=before_candidate
        )
        self._require_extract_binding(
            successor_extract, decision=after_decision, candidate=after_candidate
        )
        if baseline_extract["source_binding"] != successor_extract["source_binding"]:
            raise ValueError("accepted rerun Extract source inputs are not comparable")
        expected_comparison = build_schema_feedback_retest(
            baseline_decision=before_decision.as_dict(),
            successor_decision=after_decision.as_dict(),
            baseline_extract=baseline_extract,
            successor_extract=successor_extract,
        )
        if comparison != expected_comparison:
            raise ValueError(
                "accepted rerun comparison does not match governed Extract results"
            )

    def _load_extract_result(
        self, extract_run_ref: str, extract_result_hash: str
    ) -> dict[str, Any]:
        if (
            not extract_run_ref.startswith("extract-run-")
            or _SHA256.fullmatch(extract_result_hash) is None
        ):
            raise ValueError("accepted rerun Extract identity is invalid")
        runs_root = (self.workspace_path / "runs").resolve()
        path = (
            runs_root
            / extract_run_ref
            / "artifacts"
            / "extract_result.v1.json"
        ).resolve()
        if not path.is_relative_to(runs_root) or not path.is_file():
            raise ValueError("accepted rerun ExtractResult artifact is missing")
        result = ExtractResult.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        if (
            result.extract_run_id != extract_run_ref
            or result.extract_result_hash != extract_result_hash
        ):
            raise ValueError("accepted rerun ExtractResult identity mismatch")
        return result.as_dict()

    @staticmethod
    def _require_extract_binding(
        extract: dict[str, Any], *, decision: Any, candidate: Any
    ) -> None:
        expected_source = {
            "lineage_id": decision.source_binding["lineage_id"],
            "canonical_source_ir_ref": decision.source_binding[
                "canonical_source_ir_ref"
            ],
            "canonical_source_hash": decision.source_binding["source_hash"],
        }
        schema = extract["schema_binding"]
        if (
            extract["source_binding"] != expected_source
            or schema.get("schema_decision_ref") != decision.schema_decision_id
            or schema.get("schema_decision_hash")
            != decision.schema_decision_hash
            or schema.get("candidate_ref") != candidate.candidate_id
            or schema.get("candidate_hash") != candidate.candidate_hash
        ):
            raise ValueError(
                "accepted rerun ExtractResult Decision/source/candidate binding mismatch"
            )

    def _replay_pending(self, artifact_root: Path, rerun: MaterialCorrectionRerun):
        loaded: dict[str, Any] = {}
        for directory, manifest_name, kind, count, prefix in (
            ("positive_witnesses", "positive_manifest.json", "positive_exemplar", 32, "positive"),
            ("hard_negative_witnesses", "hard_negative_manifest.json", "hard_negative", 6, "hard_negative"),
        ):
            paths = sorted((artifact_root / directory).glob("*.json"))
            witnesses = self._witnesses(
                [json.loads(path.read_text(encoding="utf-8")) for path in paths],
                kind,
                count,
            )
            manifest = SourceSemanticWitnessSetManifestV1.model_validate(
                json.loads((artifact_root / manifest_name).read_text(encoding="utf-8"))
            )
            self._require_manifest(rerun.after or {}, manifest, witnesses, prefix)
            loaded[prefix] = (witnesses, manifest)
        positive, positive_manifest = loaded["positive"]
        negative, negative_manifest = loaded["hard_negative"]
        return positive, negative, positive_manifest, negative_manifest

    def _validate_cross_binding(
        self,
        rerun: MaterialCorrectionRerun,
        *,
        correction_material: dict[str, Any],
        pending: tuple[Any, Any, Any, Any],
    ) -> None:
        before_intent, before_candidate, before_decision = self._bound_state(
            "before", rerun.before
        )
        after_intent, _, _ = self._bound_state("after", rerun.after or {})
        if after_intent.predecessor_graph_intent_ref != before_intent.graph_intent_id:
            raise ValueError("material correction GraphIntent predecessor lineage mismatch")
        if after_intent.source_binding != before_intent.source_binding:
            raise ValueError("material correction GraphIntent source binding mismatch")
        before_payload = before_intent.as_dict()
        after_payload = after_intent.as_dict()
        changes = correction_material["changes"]
        material_fields = {
            "document_function",
            "represented_or_governed_subject",
            "primary_relationship",
            "preferred_representation",
            "authority_boundary",
            "material_dimensions",
            "source_role_interpretation",
            "unresolved_questions",
        }
        if any(
            after_payload[field] != changes.get(field, before_payload[field])
            for field in material_fields
        ):
            raise ValueError("correction material does not produce the after GraphIntent")
        if any(before_payload[field] == after_payload[field] for field in changes):
            raise ValueError("correction material contains a non-material no-op")
        self._require_before_gate(rerun.before, before_candidate, before_decision)
        self._require_pending_lineage(rerun, after_intent, pending)
        from services.graph_schema_discovery_service import (
            GraphSchemaDiscoveryService,
            _TRUSTED_PERSISTED_RECONSTRUCTION_DESCRIPTOR,
        )
        if (
            GraphSchemaDiscoveryService.__dict__.get(
                "reconstruct_persisted_material_correction_witnesses"
            )
            is not _TRUSTED_PERSISTED_RECONSTRUCTION_DESCRIPTOR
        ):
            raise ValueError(
                "material correction requires trusted reconstruction authority"
            )
        reconstructed = _TRUSTED_PERSISTED_RECONSTRUCTION_DESCRIPTOR(
            self.discovery_service,
            predecessor_graph_intent=before_intent,
            successor_graph_intent=after_intent,
            positive_manifest_ref=rerun.before["positive_manifest_ref"],
            positive_manifest_hash=rerun.before["positive_manifest_hash"],
            hard_negative_manifest_ref=rerun.before["hard_negative_manifest_ref"],
            hard_negative_manifest_hash=rerun.before[
                "hard_negative_manifest_hash"
            ],
            actor_ref=rerun.created_by,
        )
        self._require_exact_reconstruction(rerun, pending, reconstructed)

    def _bound_state(self, label: str, state: dict[str, Any]):
        intent = self.discovery_service.graph_intent_store.get(
            state["graph_intent_ref"], graph_intent_hash=state["graph_intent_hash"]
        )
        binding = intent.source_binding.as_dict()
        if (
            binding["organization_id"] != self.organization_id
            or binding["workspace_id"] != self.workspace_id
        ):
            raise ValueError(f"material correction rerun {label} ownership mismatch")
        if state["candidate_ref"] is None:
            return intent, None, None
        candidate = CorpusGraphSchemaCandidate.from_dict(
            self.discovery_service.candidate_store.get_by_hash(state["candidate_hash"])
        )
        if candidate.candidate_id != state["candidate_ref"]:
            raise ValueError(f"material correction rerun {label} candidate mismatch")
        decision = self.discovery_service.get_schema_decision_contract(
            state["schema_decision_ref"]
        )
        if decision.schema_decision_hash != state["schema_decision_hash"]:
            raise ValueError(f"material correction rerun {label} decision mismatch")
        expected_intent_binding = {
            "graph_intent_ref": intent.graph_intent_id,
            "graph_intent_hash": intent.graph_intent_hash,
            "review_state": intent.review.state,
            "source_identity_ref": intent.source_binding.source_identity_ref,
        }
        if (
            candidate.source_binding != binding
            or decision.source_binding != binding
            or candidate.graph_intent_binding != expected_intent_binding
            or decision.graph_intent_binding != expected_intent_binding
            or decision.candidate_binding.get("candidate_id") != candidate.candidate_id
            or decision.candidate_binding.get("candidate_hash") != candidate.candidate_hash
            or decision.approved_schema.get("relation_types")
            != [dict(item) for item in candidate.proposed_relation_types]
            or decision.extraction_plan != candidate.extraction_plan
        ):
            raise ValueError(
                f"material correction rerun {label} Decision/Candidate/GraphIntent binding mismatch"
            )
        return intent, candidate, decision

    @staticmethod
    def _require_before_gate(state, candidate, decision) -> None:
        if candidate is None or decision is None:
            raise ValueError("material correction before authority is missing")
        gates = [
            item
            for item in candidate.synthesis_decisions
            if item.get("decision_kind") == "exemplar_reconstruction"
        ]
        if len(gates) != 1 or not isinstance(gates[0].get("selected_value"), dict):
            raise ValueError("material correction accepted exemplar lineage is missing")
        gate = gates[0]["selected_value"]
        expected = {
            "positive_manifest_ref": state["positive_manifest_ref"],
            "positive_manifest_hash": state["positive_manifest_hash"],
            "positive_count": state["positive_count"],
            "hard_negative_manifest_ref": state["hard_negative_manifest_ref"],
            "hard_negative_manifest_hash": state["hard_negative_manifest_hash"],
            "hard_negative_count": state["hard_negative_count"],
            "material_mismatches": state["material_mismatches"],
        }
        if any(gate.get(key) != value for key, value in expected.items()):
            raise ValueError("material correction before witness gate binding mismatch")
        if not set((state["positive_manifest_ref"], state["hard_negative_manifest_ref"])).issubset(
            set(decision.candidate_binding.get("decision_evidence_refs") or [])
        ):
            raise ValueError("material correction Decision witness evidence binding mismatch")

    def _require_pending_lineage(self, rerun, after_intent, pending) -> None:
        positive, negative, positive_manifest, negative_manifest = pending
        after = rerun.after or {}
        after_digest = after_intent.graph_intent_hash.removeprefix("sha256:")
        expected_source = {
            "sourceIdentitySchemaVersion": "semantier.source_identity.v1",
            "sourceIdentityRef": after_intent.source_binding.source_identity_ref,
            "sourceHash": after_intent.source_binding.source_hash,
            "canonicalSourceIrRef": after_intent.source_binding.canonical_source_ir_ref,
            "canonicalSourceIrHash": after_intent.source_binding.canonical_source_ir_hash,
        }
        for kind, witnesses, manifest, predecessor_ref in (
            (
                "positive_exemplar",
                positive,
                positive_manifest,
                rerun.before["positive_manifest_ref"],
            ),
            (
                "hard_negative",
                negative,
                negative_manifest,
                rerun.before["hard_negative_manifest_ref"],
            ),
        ):
            expected_review_set = (
                f"review-set:material-correction:{after_digest[:20]}:{kind}"
            )
            if (
                manifest.organization_id != self.organization_id
                or manifest.workspace_id != self.workspace_id
                or manifest.predecessor_manifest_ref != predecessor_ref
                or manifest.review_set_ref != expected_review_set
            ):
                raise ValueError("material correction witness manifest lineage mismatch")
            if kind == "positive_exemplar" and (
                manifest.source_binding.model_dump(mode="json", by_alias=True)
                != expected_source
            ):
                raise ValueError("material correction positive witness source binding mismatch")
            for witness in witnesses:
                if (
                    witness.review.state != "proposed"
                    or witness.review.predecessor_witness_ref is None
                    or not witness.review.review_ref.startswith(
                        f"review:material-correction:{after_digest[:16]}:"
                    )
                ):
                    raise ValueError("material correction witness predecessor binding mismatch")
        predecessor_refs = {
            witness.review.predecessor_witness_ref
            for witness in (*positive, *negative)
        }
        if any(
            mismatch["witness_ref"] not in predecessor_refs
            for mismatch in after["material_mismatches"]
        ):
            raise ValueError("material correction mismatch witness lineage mismatch")

    @staticmethod
    def _require_exact_reconstruction(rerun, pending, reconstructed) -> None:
        positive, negative, positive_manifest, negative_manifest = pending
        actual_positive = [
            witness.model_dump(mode="json", by_alias=True) for witness in positive
        ]
        actual_negative = [
            witness.model_dump(mode="json", by_alias=True) for witness in negative
        ]
        actual_positive_manifest = positive_manifest.model_dump(
            mode="json", by_alias=True
        )
        actual_negative_manifest = negative_manifest.model_dump(
            mode="json", by_alias=True
        )
        if (
            actual_positive != reconstructed["proposed_positive_witnesses"]
            or actual_negative != reconstructed["proposed_hard_negative_witnesses"]
            or actual_positive_manifest
            != reconstructed["proposed_positive_manifest"]
            or actual_negative_manifest
            != reconstructed["proposed_hard_negative_manifest"]
            or (rerun.after or {})["material_mismatches"]
            != reconstructed["material_mismatches"]
        ):
            raise ValueError(
                "material correction witnesses do not match trusted reconstruction"
            )
        expected_positive_identities = tuple(
            (item["witnessId"], item["witnessHash"], item["review"]["reviewHash"])
            for item in reconstructed["proposed_positive_witnesses"]
        )
        expected_negative_identities = tuple(
            (item["witnessId"], item["witnessHash"], item["review"]["reviewHash"])
            for item in reconstructed["proposed_hard_negative_witnesses"]
        )
        actual_positive_identities = tuple(
            (item.witness_id, item.witness_hash, item.review.review_hash)
            for item in positive
        )
        actual_negative_identities = tuple(
            (item.witness_id, item.witness_hash, item.review.review_hash)
            for item in negative
        )
        if (
            actual_positive_identities != expected_positive_identities
            or actual_negative_identities != expected_negative_identities
        ):
            raise ValueError(
                "material correction ordered witness identities do not match reconstruction"
            )

    def _require_correction_material(
        self, rerun: MaterialCorrectionRerun, payload: dict[str, Any]
    ) -> None:
        if f"sha256:{content_hash(payload)}" != rerun.correction_material_hash:
            raise ValueError("correction material hash mismatch")

    @staticmethod
    def _require_reason_matches_material(
        rerun: MaterialCorrectionRerun, payload: dict[str, Any]
    ) -> None:
        expected = material_correction_outcome_reason(payload)
        if (
            expected
            == (
                "pending_adjudication",
                "graph_intent_successor_requires_witness_adjudication",
            )
            and rerun.outcome == "accepted"
        ):
            expected = (
                "accepted",
                "graph_intent_successor_recompiled_cleanly",
            )
        if (rerun.outcome, rerun.reason) != expected:
            raise ValueError("material correction outcome/reason mismatch")

    def _correction_material_path(self, artifact_ref: str) -> Path:
        resolved = (self.workspace_path / artifact_ref).resolve()
        if not resolved.is_relative_to(self.workspace_path):
            raise ValueError("correction material path escapes governed workspace")
        return resolved

    @staticmethod
    def _witnesses(payloads: list[dict[str, Any]], kind: str, count: int):
        witnesses = [SourceSemanticWitnessV1.model_validate(item) for item in payloads]
        if len(witnesses) != count or any(item.witness_kind != kind for item in witnesses):
            raise ValueError("material correction rerun witness set mismatch")
        return witnesses

    @staticmethod
    def _require_manifest(state, manifest, witnesses, prefix: str) -> None:
        key = "positive" if prefix == "positive" else "hard_negative"
        if (
            manifest.manifest_ref != state[f"{key}_manifest_ref"]
            or manifest.content_hash != state[f"{key}_manifest_hash"]
        ):
            raise ValueError("material correction rerun witness manifest mismatch")
        manifest.verify_witnesses(witnesses)

    def _require_ownership(self, rerun: MaterialCorrectionRerun) -> None:
        if rerun.organization_id != self.organization_id:
            raise ValueError("material correction rerun organization ownership mismatch")
        if rerun.workspace_id != self.workspace_id:
            raise ValueError("material correction rerun workspace ownership mismatch")

    def _artifact_root(self, rerun_hash: str) -> Path:
        resolved = (self.root / rerun_hash.removeprefix("sha256:")).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("material correction rerun path escapes governed workspace")
        return resolved


__all__ = ["MaterialCorrectionRerunStore"]
