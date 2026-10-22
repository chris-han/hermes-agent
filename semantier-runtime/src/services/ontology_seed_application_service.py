"""Governed-store orchestration for explicit selection and pinned seed application."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
import sqlite3
from typing import Callable

from pydantic import TypeAdapter

from contracts.graph_intent_brief import GraphIntentBrief
from contracts.ontology_build_inputs import OntologyBuildInputEnvelopeV1
from contracts.ontology_building import (OntologyBuildManifestV1, ExploratoryGraphSnapshotV1,
    CanonicalEvidenceAdequacyAssessmentV1, CanonicalEvidenceBoundaryV1, ArtifactPinV1 as BuildPin,
    CandidateGraphDiffProjectionV1, CandidateGraphLineageProjectionV1)
from contracts.ontology_hub_snapshot import OntologyHubSnapshotV1
from contracts.ontology_seeds import OntologyCapabilityProfileV1
from contracts.ontology_seed_alignment import (INPUT_PIN_FIELDS, ExactAlignmentInputV1, OntologySeedAlignmentCommandV1,
    AlignmentDecisionV1, SourceExtensionV1, SourceGroundedSeedDeltaV1, OntologySeedComparisonV1)
from contracts.ontology_seed_application import SeedSelectionReceiptV1, SeedApplyReceiptV1
from contracts.ontology_seed_state import OntologySeedStateV1, ArtifactPinV1
from eos.artifact_hashing import canonical_json, content_hash
from services.ontology_audit_service import OntologyAuditService
from services.ontology_build_coordinator import OntologyBuildCoordinator
from services.ontology_seed_alignment_service import OntologySeedAlignmentService
from storage.ontology_build_store import OntologyBuildStore


@dataclass(frozen=True)
class GovernedSeedActor:
    """Internal context from the governed route resolver; never a JSON body model."""
    organization_id: str
    workspace_id: str
    actor_ref: str


@dataclass(frozen=True)
class SeedSelectionResultV1:
    state: OntologySeedStateV1
    receipt: SeedSelectionReceiptV1


@dataclass(frozen=True)
class SeedApplyResultV1:
    state: OntologySeedStateV1
    receipt: SeedApplyReceiptV1
    manifest: OntologyBuildManifestV1 | None


def _pin(value, ref, digest):
    return ArtifactPinV1(artifact_ref=getattr(value, ref), artifact_hash=getattr(value, digest))


def _exact(value, ref, digest):
    return ExactAlignmentInputV1.of(value, ref, digest)


def _child(manifest, kind):
    pins = [pin for pin in manifest.child_artifacts if pin.artifact_kind == kind]
    if len(pins) != 1:
        raise ValueError(f'exact manifest child missing: {kind}')
    return pins[0]


def _load(store, context, pin, cls, ref, digest):
    value = store.get(organization_id=context.organization_id, workspace_id=context.workspace_id,
        artifact_ref=pin.artifact_ref, contract_type=cls)
    if value is None or (getattr(value, ref), getattr(value, digest)) != (pin.artifact_ref, pin.artifact_hash):
        raise ValueError('missing or mismatched pinned seed application artifact')
    if isinstance(pin, ExactAlignmentInputV1) and _exact(value, ref, digest) != pin:
        raise ValueError('seed application artifact full payload digest mismatch')
    return value


def _state_successor(previous, now, **updates):
    values = previous.model_dump(exclude={'state_ref', 'state_hash'})
    values.update(revision=previous.revision + 1, predecessor_state_ref=previous.state_ref,
        predecessor_state_hash=previous.state_hash, created_at=now, **updates)
    # Preserve typed children/defaults rather than hash an unvalidated raw map.
    for name in ('query', 'policy', 'recommendations', 'external_candidates', 'provider_failures'):
        values[name] = getattr(previous, name)
    return OntologySeedStateV1.create(**values)


class OntologySeedApplicationService:
    def __init__(self, *, clock: Callable[[], datetime] | None = None):
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _now(self, earliest):
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() != timezone.utc.utcoffset(now):
            raise ValueError('seed action time must be timezone-aware UTC')
        if now < earliest:
            raise ValueError('seed action cannot predate baseline or selected state')
        return now

    def _request(self, context, operation, request_id, **values):
        if not isinstance(context, GovernedSeedActor) or not all((context.organization_id, context.workspace_id, context.actor_ref)):
            raise PermissionError('governed seed actor context is required')
        if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}', request_id) is None:
            raise ValueError('invalid bounded seed request id')
        material = {name: value.model_dump(mode='json') if hasattr(value, 'model_dump') else value for name, value in values.items()}
        if len(canonical_json(material).encode('utf-8')) > 2_000_000:
            raise ValueError('seed request exceeds 2 MB bound')
        return 'sha256:' + content_hash(dict(organization_id=context.organization_id, workspace_id=context.workspace_id,
            actor_ref=context.actor_ref, operation=operation, request_id=request_id, **material))

    def _prior(self, store, context, operation, request_id, digest):
        found = store.get_seed_request(organization_id=context.organization_id, workspace_id=context.workspace_id,
            actor_ref=context.actor_ref, operation=operation, request_id=request_id)
        if found is not None and found[0].request_hash != digest:
            raise ValueError('seed request id was already used with different input')
        return found

    def _baseline(self, store, context, build_ref, expected_state):
        expected_state = ArtifactPinV1.model_validate(expected_state.model_dump(mode='json'))
        manifest = store.get(organization_id=context.organization_id, workspace_id=context.workspace_id,
            artifact_ref=build_ref, contract_type=OntologyBuildManifestV1)
        if manifest is None:
            raise LookupError('baseline build not found')
        state = store.get_seed_state_head(organization_id=context.organization_id, workspace_id=context.workspace_id, build_ref=build_ref)
        if state is None or (state.state_ref, state.state_hash) != (expected_state.artifact_ref, expected_state.artifact_hash):
            raise ValueError('expected ontology seed state is stale')
        if state.build_hash != manifest.build_hash or state.search_status not in ('matched', 'searched', 'no_match'):
            raise ValueError('seed selection requires exact completed discovery')
        if manifest.assurance_profile != 'A0_EXPLORATORY':
            raise ValueError('seed application currently supports A0_EXPLORATORY only')
        self._validate_source_first_baseline(store, context, manifest)
        return manifest, state

    def _validate_source_first_baseline(self, store, context, manifest):
        """Verify historical production hash bases without replaying compiler code."""
        if manifest.predecessor_build_ref is not None:
            raise ValueError('seed application baseline must be an initial source-first build')
        envelope = _load(store, context, _child(manifest, 'ontology_build_inputs'), OntologyBuildInputEnvelopeV1, 'envelope_ref', 'envelope_hash')
        snapshot = _load(store, context, _child(manifest, 'exploratory_graph_snapshot'), ExploratoryGraphSnapshotV1, 'snapshot_ref', 'snapshot_hash')
        boundary = _load(store, context, _child(manifest, 'canonical_evidence_boundary'), CanonicalEvidenceBoundaryV1, 'boundary_ref', 'boundary_hash')
        assessment = _load(store, context, _child(manifest, 'canonical_evidence_adequacy_assessment'), CanonicalEvidenceAdequacyAssessmentV1, 'assessment_ref', 'assessment_hash')
        brief = GraphIntentBrief.from_dict(envelope.graph_intent)
        compiler_pin = _child(manifest, 'ontology_source_first_compiler')
        if envelope.compiler_version not in ('semantier_source_first_exploratory_v1', 'semantier_source_first_exploratory_v2', 'semantier_source_first_exploratory_v3') or compiler_pin.artifact_ref != envelope.compiler_version:
            raise ValueError('unsupported archived source-first compiler identity')
        import base64
        from hashlib import sha256

        if compiler_pin.artifact_hash != 'sha256:' + sha256(
                base64.b64decode(envelope.compiler_bytes_base64, validate=True)).hexdigest():
            raise ValueError('baseline compiler bytes do not bind retained compiler identity')
        build_basis = dict(source_binding=brief.source_binding.as_dict(), graph_intent_ref=brief.graph_intent_id,
            graph_intent_hash=brief.graph_intent_hash, compiler_version=envelope.compiler_version,
            compiler_hash=compiler_pin.artifact_hash, assurance_profile=manifest.assurance_profile)
        if manifest.build_ref != 'ontology-build:' + content_hash(build_basis) or envelope.build_ref != manifest.build_ref:
            raise ValueError('baseline build identity does not bind archived inputs')
        projected = OntologyBuildCoordinator().project_manifest(build_ref=manifest.build_ref,
            organization_id=manifest.organization_id, workspace_id=manifest.workspace_id,
            assurance_profile=manifest.assurance_profile, assessment=assessment,
            additional_artifacts=tuple(pin for pin in manifest.child_artifacts if pin.artifact_kind != 'canonical_evidence_adequacy_assessment'),
            created_at=manifest.created_at)
        if projected != manifest:
            raise ValueError('baseline manifest does not bind original coordinator hash basis')
        if snapshot.completeness_state != 'incomplete' or snapshot.predecessor_snapshot_ref is not None or snapshot.change_cause != 'initial_exploration':
            raise ValueError('baseline snapshot differs from source-first provisional defaults')
        def verify(value, ref, digest, prefix, excluded):
            payload = value.model_dump(mode='json', exclude={ref, digest, 'schema_version', *excluded})
            if isinstance(value, CanonicalEvidenceBoundaryV1):
                payload['target_time'] = value.target_time.isoformat()
            actual = content_hash(payload)
            if (getattr(value, ref), getattr(value, digest)) != (prefix + ':' + actual, 'sha256:' + actual):
                raise ValueError('baseline artifact does not bind original source-first hash basis')
        if envelope.compiler_version in {'semantier_source_first_exploratory_v2', 'semantier_source_first_exploratory_v3'}:
            try:
                verified, integrity = store.get_verified_exploratory_snapshot(
                    organization_id=context.organization_id, workspace_id=context.workspace_id,
                    pin=_child(manifest, 'exploratory_graph_snapshot'))
            except ValueError as exc:
                raise ValueError(f'baseline snapshot integrity failed: {exc}') from exc
            if (verified != snapshot or integrity.producer_profile != 'source_first_build.v1'
                    or integrity.producer_version != envelope.compiler_version):
                raise ValueError('baseline snapshot integrity differs from retained compiler')
        else:
            verify(snapshot, 'snapshot_ref', 'snapshot_hash', 'exploratory-snapshot',
                {'created_at', 'authority_state', 'completeness_state', 'predecessor_snapshot_ref', 'predecessor_snapshot_hash'})
        verify(boundary, 'boundary_ref', 'boundary_hash', 'canonical-boundary', {'created_at'})
        verify(assessment, 'assessment_ref', 'assessment_hash', 'evidence-adequacy', {'assessed_at'})
        if (snapshot.created_at, boundary.created_at, assessment.assessed_at) != (manifest.created_at,) * 3 or boundary.target_time != manifest.created_at:
            raise ValueError('baseline timestamps differ from hashed canonical boundary')
        if (assessment.boundary_ref, assessment.boundary_hash) != (
                boundary.boundary_ref, boundary.boundary_hash):
            raise ValueError('baseline adequacy bindings are inconsistent')

    def select_registry_seed(self, *, store: OntologyBuildStore, context: GovernedSeedActor, build_ref: str,
                             expected_state: ArtifactPinV1, recommendation: ArtifactPinV1, request_id: str) -> SeedSelectionResultV1:
        digest = self._request(context, 'select_seed', request_id, build_ref=build_ref, expected_state=expected_state,
            mode='registry_recommendation', recommendation=recommendation)
        prior = self._prior(store, context, 'select_seed', request_id, digest)
        if prior:
            return self._selection_result(store, context, *prior)
        manifest, state = self._baseline(store, context, build_ref, expected_state)
        matches = [item for item in state.visible_recommendations if (item.recommendation_ref, item.recommendation_hash) ==
            (recommendation.artifact_ref, recommendation.artifact_hash) and item.status == 'PROPOSED']
        if len(matches) != 1:
            raise ValueError('selection requires an exact visible proposed recommendation')
        candidate = matches[0]
        profile = _load(store, context, ArtifactPinV1(artifact_ref=candidate.profile_ref, artifact_hash=candidate.profile_hash),
            OntologyCapabilityProfileV1, 'profile_ref', 'profile_hash')
        snapshot = _load(store, context, ArtifactPinV1(artifact_ref=profile.identity.provenance['snapshot_ref'],
            artifact_hash=profile.identity.provenance['snapshot_hash']), OntologyHubSnapshotV1, 'snapshot_ref', 'snapshot_hash')
        if (candidate.ontology_ref, candidate.ontology_version, candidate.ontology_source_content_hash, candidate.ontology_graph_content_hash) != (
                snapshot.ontology_ref, snapshot.version, snapshot.source_content_hash, snapshot.graph_content_hash):
            raise ValueError('recommendation does not bind imported ontology version')
        return self._select(store, context, manifest, state, snapshot, profile, request_id, digest,
            mode='registry_recommendation', recommendation=recommendation)

    def prepare_external_selection(self, *, store: OntologyBuildStore, context: GovernedSeedActor,
            build_ref: str, expected_state: ArtifactPinV1, external_ref: str):
        """Validate the local selection boundary before any catalog fetch/import."""
        if not isinstance(context, GovernedSeedActor) or not all((context.organization_id, context.workspace_id, context.actor_ref)):
            raise PermissionError('governed seed actor context is required')
        _, state = self._baseline(store, context, build_ref, expected_state)
        candidates = [item for item in state.external_candidates if item.external_ref == external_ref]
        if len(candidates) != 1:
            raise ValueError('external selection must name one exact discovered candidate')
        return candidates[0]

    def record_imported_seed_selection(self, *, store: OntologyBuildStore, context: GovernedSeedActor, build_ref: str,
            expected_state: ArtifactPinV1, imported_snapshot: ArtifactPinV1, imported_profile: ArtifactPinV1,
            inspected_external_candidate_ref: str, request_id: str) -> SeedSelectionResultV1:
        digest = self._request(context, 'select_seed', request_id, build_ref=build_ref, expected_state=expected_state,
            mode='manual_external', imported_snapshot=imported_snapshot, imported_profile=imported_profile,
            inspected_external_candidate_ref=inspected_external_candidate_ref)
        prior = self._prior(store, context, 'select_seed', request_id, digest)
        if prior:
            return self._selection_result(store, context, *prior)
        manifest, state = self._baseline(store, context, build_ref, expected_state)
        candidates = [item for item in state.external_candidates if item.external_ref == inspected_external_candidate_ref]
        if len(candidates) != 1:
            raise ValueError('manual import selection requires an exact inspected discovery candidate')
        candidate = candidates[0]
        snapshot = _load(store, context, imported_snapshot, OntologyHubSnapshotV1, 'snapshot_ref', 'snapshot_hash')
        profile = _load(store, context, imported_profile, OntologyCapabilityProfileV1, 'profile_ref', 'profile_hash')
        if snapshot.origin.external_ref != candidate.external_ref:
            raise ValueError('imported snapshot differs from inspected external candidate')
        return self._select(store, context, manifest, state, snapshot, profile, request_id, digest,
            mode='manual_external', external_candidate_ref=candidate.external_ref,
            external_candidate_hash='sha256:' + content_hash(candidate.model_dump(mode='json')))

    def _select(self, store, context, manifest, state, snapshot, profile, request_id, digest, **mode):
        now = self._now(max(manifest.created_at, state.created_at))
        receipt = SeedSelectionReceiptV1.create(organization_id=context.organization_id, workspace_id=context.workspace_id,
            build_ref=manifest.build_ref, actor_ref=context.actor_ref, request_id=request_id, request_hash=digest,
            expected_state=_exact(state, 'state_ref', 'state_hash'), baseline_manifest=_exact(manifest, 'build_ref', 'build_hash'),
            imported_snapshot=_exact(snapshot, 'snapshot_ref', 'snapshot_hash'), imported_profile=_exact(profile, 'profile_ref', 'profile_hash'),
            created_at=now, **mode)
        # Validates exact profile/source/provenance and the archived baseline.
        self.load_alignment_inputs(store=store, context=context, build_ref=manifest.build_ref, selection=receipt)
        successor = _state_successor(state, now, selection_receipt=_pin(receipt, 'receipt_ref', 'receipt_hash'),
            selected_recommendation=receipt.recommendation, imported_snapshot=_pin(snapshot, 'snapshot_ref', 'snapshot_hash'),
            alignment_actions=(), source_grounded_delta=None, latest_apply_receipt=None)
        try:
            store.append_seed_state(successor, expected_state_ref=state.state_ref, expected_state_hash=state.state_hash,
                related_artifacts=(receipt,), request_receipt=receipt,
                verify_before_commit=lambda: self._selection_result(store, context, receipt, successor))
        except (ValueError, sqlite3.IntegrityError):
            prior = self._prior(store, context, 'select_seed', request_id, digest)
            if prior:
                return self._selection_result(store, context, *prior)
            raise
        return SeedSelectionResultV1(successor, receipt)

    def _selection_result(self, store, context, receipt, state):
        self.load_alignment_inputs(store=store, context=context, build_ref=receipt.build_ref, selection=receipt)
        return SeedSelectionResultV1(state, receipt)

    def inspect_selected_seed(self, *, store: OntologyBuildStore, context: GovernedSeedActor,
                              build_ref: str, selection_receipt: ArtifactPinV1):
        """Resolve an exact historical selection; no current-head requirement."""
        if not isinstance(context, GovernedSeedActor) or not all((context.organization_id, context.workspace_id, context.actor_ref)):
            raise PermissionError('governed seed actor context is required')
        pin = ArtifactPinV1.model_validate(selection_receipt.model_dump(mode='json'))
        selection = _load(store, context, pin, SeedSelectionReceiptV1, 'receipt_ref', 'receipt_hash')
        return self.load_alignment_inputs(store=store, context=context, build_ref=build_ref, selection=selection)

    def load_alignment_inputs(self, *, store: OntologyBuildStore, context: GovernedSeedActor,
                              build_ref: str, selection: SeedSelectionReceiptV1):
        selection = SeedSelectionReceiptV1.model_validate(selection.model_dump(mode='json'))
        if (selection.organization_id, selection.workspace_id, selection.build_ref) != (context.organization_id, context.workspace_id, build_ref):
            raise PermissionError('selection is outside governed tenant/build')
        manifest = _load(store, context, selection.baseline_manifest, OntologyBuildManifestV1, 'build_ref', 'build_hash')
        self._validate_source_first_baseline(store, context, manifest)
        discovery = _load(store, context, selection.expected_state, OntologySeedStateV1, 'state_ref', 'state_hash')
        if (discovery.build_ref, discovery.build_hash) != (manifest.build_ref, manifest.build_hash):
            raise ValueError('selection predecessor does not bind baseline build')
        inputs = dict(baseline_manifest=manifest,
            baseline_snapshot=_load(store, context, _child(manifest, 'exploratory_graph_snapshot'), ExploratoryGraphSnapshotV1, 'snapshot_ref', 'snapshot_hash'),
            input_envelope=_load(store, context, _child(manifest, 'ontology_build_inputs'), OntologyBuildInputEnvelopeV1, 'envelope_ref', 'envelope_hash'),
            seed_snapshot=_load(store, context, selection.imported_snapshot, OntologyHubSnapshotV1, 'snapshot_ref', 'snapshot_hash'),
            seed_profile=_load(store, context, selection.imported_profile, OntologyCapabilityProfileV1, 'profile_ref', 'profile_hash'))
        profile, snapshot = inputs['seed_profile'], inputs['seed_snapshot']
        if (profile.ontology_ref, profile.version, profile.source_content_hash, profile.graph_content_hash,
            profile.identity.provenance.get('snapshot_ref'), profile.identity.provenance.get('snapshot_hash')) != (
            snapshot.ontology_ref, snapshot.version, snapshot.source_content_hash, snapshot.graph_content_hash, snapshot.snapshot_ref, snapshot.snapshot_hash):
            raise ValueError('selected profile does not bind exact imported snapshot')
        if selection.mode == 'registry_recommendation':
            recommendations = [item for item in discovery.visible_recommendations if
                (item.recommendation_ref, item.recommendation_hash) ==
                (selection.recommendation.artifact_ref, selection.recommendation.artifact_hash) and item.status == 'PROPOSED']
            if len(recommendations) != 1 or (recommendations[0].profile_ref, recommendations[0].profile_hash,
                    recommendations[0].ontology_ref, recommendations[0].ontology_version,
                    recommendations[0].ontology_source_content_hash, recommendations[0].ontology_graph_content_hash) != (
                    profile.profile_ref, profile.profile_hash, snapshot.ontology_ref, snapshot.version,
                    snapshot.source_content_hash, snapshot.graph_content_hash):
                raise ValueError('selection receipt differs from exact recommendation/profile')
        else:
            candidates = [item for item in discovery.external_candidates if item.external_ref == selection.external_candidate_ref]
            if len(candidates) != 1 or 'sha256:' + content_hash(candidates[0].model_dump(mode='json')) != selection.external_candidate_hash or snapshot.origin.external_ref != selection.external_candidate_ref:
                raise ValueError('selection receipt differs from exact inspected external candidate')
        OntologyAuditService().export(store=store, organization_id=context.organization_id,
            workspace_id=context.workspace_id, build_ref=build_ref)
        return inputs

    def apply_selected_seed(self, *, store: OntologyBuildStore, context: GovernedSeedActor, build_ref: str,
            expected_state: ArtifactPinV1, selection_receipt: ArtifactPinV1, request_id: str,
            decisions: tuple[AlignmentDecisionV1, ...], extensions: tuple[SourceExtensionV1, ...] = ()) -> SeedApplyResultV1:
        decisions = TypeAdapter(tuple[AlignmentDecisionV1, ...]).validate_python([item.model_dump(mode='json') for item in decisions])
        extensions = TypeAdapter(tuple[SourceExtensionV1, ...]).validate_python([item.model_dump(mode='json') for item in extensions])
        if len(decisions) > 128 or len(extensions) > 128:
            raise ValueError('seed application supports at most 128 decisions/extensions')
        digest = self._request(context, 'apply_seed', request_id, build_ref=build_ref, expected_state=expected_state,
            selection_receipt=selection_receipt, decisions=[item.model_dump(mode='json') for item in decisions],
            extensions=[item.model_dump(mode='json') for item in extensions])
        prior = self._prior(store, context, 'apply_seed', request_id, digest)
        if prior:
            return self._apply_result(store, context, *prior)
        manifest, state = self._baseline(store, context, build_ref, expected_state)
        if state.selection_receipt != selection_receipt or state.latest_apply_receipt is not None:
            raise ValueError('apply requires the exact selected state without a prior application')
        selection = _load(store, context, selection_receipt, SeedSelectionReceiptV1, 'receipt_ref', 'receipt_hash')
        if state.selected_recommendation != selection.recommendation:
            raise ValueError('selected recommendation differs from exact receipt')
        if selection.recommendation is not None and not any(
                (item.recommendation_ref, item.recommendation_hash) ==
                (selection.recommendation.artifact_ref, selection.recommendation.artifact_hash)
                for item in state.visible_recommendations):
            raise ValueError('selected recommendation was ignored or is no longer visible')
        if state.imported_snapshot != ArtifactPinV1(artifact_ref=selection.imported_snapshot.artifact_ref, artifact_hash=selection.imported_snapshot.artifact_hash):
            raise ValueError('selected state imported snapshot differs from receipt')
        inputs = self.load_alignment_inputs(store=store, context=context, build_ref=build_ref, selection=selection)
        now = self._now(max(manifest.created_at, state.created_at, selection.created_at))
        command = OntologySeedAlignmentCommandV1.for_inputs(**inputs, actor_ref=context.actor_ref, reviewed_at=now,
            decisions=decisions, extensions=extensions)
        result = OntologySeedAlignmentService().apply(command=command, governed_actor_ref=context.actor_ref, **inputs)
        artifacts = [result.alignment, result.delta, result.seed_comparison, result.lineage]
        successor_manifest = None
        if result.successor is not None:
            successor_manifest, envelope = self._successor_manifest(store, context, inputs, state, selection, result)
            artifacts.extend((result.successor, result.diff, envelope, successor_manifest))
        receipt = SeedApplyReceiptV1.create(organization_id=context.organization_id, workspace_id=context.workspace_id,
            build_ref=build_ref, actor_ref=context.actor_ref, request_id=request_id, request_hash=digest,
            expected_state=_exact(state, 'state_ref', 'state_hash'), baseline_manifest=_exact(manifest, 'build_ref', 'build_hash'),
            selection_receipt=_exact(selection, 'receipt_ref', 'receipt_hash'), alignment=_exact(command, 'alignment_ref', 'alignment_hash'),
            delta=_exact(result.delta, 'delta_ref', 'delta_hash'), comparison=_exact(result.seed_comparison, 'comparison_ref', 'comparison_hash'),
            lineage=_exact(result.lineage, 'lineage_ref', 'lineage_hash'), outcome='APPLIED' if successor_manifest else 'NO_DELTA',
            successor_manifest=_exact(successor_manifest, 'build_ref', 'build_hash') if successor_manifest else None,
            successor_snapshot=_exact(result.successor, 'snapshot_ref', 'snapshot_hash') if result.successor else None,
            diff=_exact(result.diff, 'diff_ref', 'diff_hash') if result.diff else None, created_at=now)
        successor_state = _state_successor(state, now, latest_apply_receipt=_pin(receipt, 'receipt_ref', 'receipt_hash'),
            selection_receipt=state.selection_receipt, alignment_actions=(*state.alignment_actions, _pin(command, 'alignment_ref', 'alignment_hash')), source_grounded_delta=_pin(result.delta, 'delta_ref', 'delta_hash'))
        artifacts.append(receipt)
        try:
            store.append_seed_state(successor_state, expected_state_ref=state.state_ref, expected_state_hash=state.state_hash,
                related_artifacts=tuple(artifacts), request_receipt=receipt,
                snapshot_integrities=(result.successor_integrity,) if result.successor_integrity else (),
                verify_before_commit=lambda: self._apply_result(store, context, receipt, successor_state))
        except (ValueError, sqlite3.IntegrityError):
            prior = self._prior(store, context, 'apply_seed', request_id, digest)
            if prior:
                return self._apply_result(store, context, *prior)
            raise
        return SeedApplyResultV1(successor_state, receipt, successor_manifest)

    def _successor_manifest(self, store, context, inputs, state, selection, result):
        baseline = inputs['baseline_manifest']
        source = inputs['input_envelope']
        envelope = OntologyBuildInputEnvelopeV1.create(**(source.model_dump(mode='json', exclude={'envelope_ref', 'envelope_hash'}) |
            {'build_ref': result.successor.build_ref}))
        changed = {'ontology_build_inputs', 'exploratory_graph_snapshot', 'canonical_evidence_adequacy_assessment'}
        pins = [pin for pin in baseline.child_artifacts if pin.artifact_kind not in changed]
        def add(kind, artifact, ref, digest):
            pins.append(BuildPin(artifact_kind=kind, artifact_ref=getattr(artifact, ref), artifact_hash=getattr(artifact, digest)))
        for kind, artifact, ref, digest in (
            ('ontology_build_inputs', envelope, 'envelope_ref', 'envelope_hash'),
            ('exploratory_graph_snapshot', result.successor, 'snapshot_ref', 'snapshot_hash'),
            ('baseline_manifest', baseline, 'build_ref', 'build_hash'),
            ('baseline_snapshot', inputs['baseline_snapshot'], 'snapshot_ref', 'snapshot_hash'),
            ('baseline_ontology_build_inputs', source, 'envelope_ref', 'envelope_hash'),
            ('selected_seed_state', state, 'state_ref', 'state_hash'),
            ('selection_predecessor_state', _load(store, context, selection.expected_state, OntologySeedStateV1, 'state_ref', 'state_hash'), 'state_ref', 'state_hash'),
            ('seed_selection_receipt', selection, 'receipt_ref', 'receipt_hash'),
            ('ontology_hub_snapshot', inputs['seed_snapshot'], 'snapshot_ref', 'snapshot_hash'),
            ('ontology_capability_profile', inputs['seed_profile'], 'profile_ref', 'profile_hash'),
            ('ontology_seed_alignment', result.alignment, 'alignment_ref', 'alignment_hash'),
            ('source_grounded_seed_delta', result.delta, 'delta_ref', 'delta_hash'),
            ('ontology_seed_comparison', result.seed_comparison, 'comparison_ref', 'comparison_hash'),
            ('candidate_graph_lineage_projection', result.lineage, 'lineage_ref', 'lineage_hash'),
            ('candidate_graph_diff_projection', result.diff, 'diff_ref', 'diff_hash'),
        ):
            add(kind, artifact, ref, digest)
        assessment = _load(store, context, _child(baseline, 'canonical_evidence_adequacy_assessment'),
            CanonicalEvidenceAdequacyAssessmentV1, 'assessment_ref', 'assessment_hash')
        successor = OntologyBuildCoordinator().project_manifest(build_ref=result.successor.build_ref,
            organization_id=context.organization_id, workspace_id=context.workspace_id, assurance_profile=baseline.assurance_profile,
            assessment=assessment, additional_artifacts=tuple(sorted(pins, key=lambda pin: pin.artifact_kind)),
            created_at=result.alignment.reviewed_at, predecessor_build_ref=baseline.build_ref, predecessor_build_hash=baseline.build_hash)
        return successor, envelope

    def _apply_result(self, store, context, receipt, state):
        selection = _load(store, context, receipt.selection_receipt, SeedSelectionReceiptV1, 'receipt_ref', 'receipt_hash')
        inputs = self.load_alignment_inputs(store=store, context=context, build_ref=receipt.build_ref, selection=selection)
        command = _load(store, context, receipt.alignment, OntologySeedAlignmentCommandV1, 'alignment_ref', 'alignment_hash')
        if command.actor_ref != receipt.actor_ref or command.reviewed_at != receipt.created_at:
            raise ValueError('apply receipt does not bind command actor/time')
        for name, fields in INPUT_PIN_FIELDS.items():
            if _exact(inputs[name], *fields) != getattr(command.inputs, name):
                raise ValueError('archived command input payload changed')
        _load(store, context, receipt.expected_state, OntologySeedStateV1, 'state_ref', 'state_hash')
        for pin, cls, ref, digest in (
            (receipt.alignment, OntologySeedAlignmentCommandV1, 'alignment_ref', 'alignment_hash'),
            (receipt.delta, SourceGroundedSeedDeltaV1, 'delta_ref', 'delta_hash'),
            (receipt.comparison, OntologySeedComparisonV1, 'comparison_ref', 'comparison_hash'),
            (receipt.lineage, CandidateGraphLineageProjectionV1, 'lineage_ref', 'lineage_hash'),
        ):
            _load(store, context, pin, cls, ref, digest)
        manifest = None
        if receipt.successor_manifest:
            manifest = _load(store, context, receipt.successor_manifest, OntologyBuildManifestV1, 'build_ref', 'build_hash')
            manifest_snapshot = _child(manifest, 'exploratory_graph_snapshot')
            if (manifest_snapshot.artifact_ref, manifest_snapshot.artifact_hash) != (
                    receipt.successor_snapshot.artifact_ref, receipt.successor_snapshot.artifact_hash):
                raise ValueError('successor manifest does not bind apply receipt snapshot')
            snapshot, _ = store.get_verified_exploratory_snapshot(
                organization_id=context.organization_id,
                workspace_id=context.workspace_id,
                pin=BuildPin(
                    artifact_kind='exploratory_graph_snapshot',
                    artifact_ref=receipt.successor_snapshot.artifact_ref,
                    artifact_hash=receipt.successor_snapshot.artifact_hash,
                ),
            )
            if _exact(snapshot, 'snapshot_ref', 'snapshot_hash') != receipt.successor_snapshot:
                raise ValueError('apply receipt successor snapshot payload changed')
            _load(store, context, receipt.diff, CandidateGraphDiffProjectionV1, 'diff_ref', 'diff_hash')
            OntologyAuditService().export(store=store, organization_id=context.organization_id,
                workspace_id=context.workspace_id, build_ref=manifest.build_ref)
        return SeedApplyResultV1(state, receipt, manifest)
