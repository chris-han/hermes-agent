"""Read-only federation of exact persisted owner bindings; no lineage persistence."""
from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlencode
from contracts.evidence_location import SourceIdentity
from contracts.retained_context_items import RetainedContextItemChangeV1

from contracts.ontology_building import ArtifactPinV1, OntologyBuildManifestV1, ExploratoryGraphSnapshotV1
from contracts.project_memory_state import ProjectStateProjectionV1
from contracts.runtime_context_reconstruction import MemoryCandidateV1
from contracts.learning_workbench import SkillArtifactVersionV1
from contracts.semantic_kernel import Evidence, ContextGraph, DecisionRecord, ChangeCandidate, Ontology, WorkingContext
from contracts.change_candidate_v2 import ChangeCandidateV2
from contracts.ontology_admission import OntologyAdmissionDecisionV1, OntologyReleaseBundleV1
from contracts.runtime_working_context import RuntimeTaskContextV1, RuntimeTaskContextV2, RuntimeTaskContextV3
from contracts.runtime_semantic_evaluation import RuntimeExecutionAdmissionV1, RuntimeEvaluationRunV1, RuntimeEvaluationRequestOutcomeV1
from contracts.learned_skill_publication import LearnedSkillProposalV1, LearnedSkillAssessmentV1, LearnedSkillPublicationReceiptV1
from services.project_memory_evidence_binder import ProjectMemoryEvidenceBinder
from services.semantica_project_memory_adapter import SemanticaProjectMemoryAdapter
from runtime_paths import workspace_root_path, workspace_hermes_home_path

# Explicit owner contracts and identity fields; never recursively discover arbitrary pins.
_OWNERS = {
    'retained_context_item_change.v1': (RetainedContextItemChangeV1, 'retained_context_item_change', 'artifact_ref', 'artifact_hash', 'experience'),
    'ontology_build_manifest.v1': (OntologyBuildManifestV1, 'ontology_build_manifest', 'build_ref', 'build_hash', 'semantic'),
    'exploratory_graph_snapshot.v1': (ExploratoryGraphSnapshotV1, 'exploratory_graph_snapshot', 'snapshot_ref', 'snapshot_hash', 'semantic'),
    'ontology_admission_decision.v1': (OntologyAdmissionDecisionV1, 'ontology_admission_decision', 'decision_ref', 'decision_hash', 'governance'),
    'ontology_release_bundle.v1': (OntologyReleaseBundleV1, 'ontology_release_bundle', 'release_bundle_ref', 'release_bundle_hash', 'governance'),
    'runtime_working_context.v1': (RuntimeTaskContextV1, 'runtime_working_context', 'context_ref', 'context_hash', 'runtime'),
    'runtime_working_context.v2': (RuntimeTaskContextV2, 'runtime_working_context', 'context_ref', 'context_hash', 'runtime'),
    'runtime_working_context.v3': (RuntimeTaskContextV3, 'runtime_working_context', 'context_ref', 'context_hash', 'runtime'),
    'runtime_execution_admission.v1': (RuntimeExecutionAdmissionV1, 'runtime_execution_admission', 'admission_ref', 'admission_hash', 'runtime'),
    'runtime_evaluation_run.v1': (RuntimeEvaluationRunV1, 'runtime_evaluation_run', 'run_ref', 'run_hash', 'runtime'),
    'runtime_evaluation_request_outcome.v1': (RuntimeEvaluationRequestOutcomeV1, 'runtime_evaluation_request_outcome', 'outcome_ref', 'outcome_hash', 'runtime'),
    'learned_skill_proposal.v1': (LearnedSkillProposalV1, 'learned_skill_proposal', 'artifact_ref', 'artifact_hash', 'learning'),
    'learned_skill_assessment.v1': (LearnedSkillAssessmentV1, 'learned_skill_assessment', 'artifact_ref', 'artifact_hash', 'learning'),
    'learned_skill_publication_receipt.v1': (LearnedSkillPublicationReceiptV1, 'learned_skill_publication_receipt', 'artifact_ref', 'artifact_hash', 'learning'),
    'project_state_projection.v1': (ProjectStateProjectionV1, 'project_state_projection', 'projection_ref', 'projection_hash', 'project_state'),
    'memory_candidate.v1': (MemoryCandidateV1, 'memory_candidate', 'candidate_ref', 'candidate_hash', 'project_memory'),
    'skill_artifact_version.v1': (SkillArtifactVersionV1, 'skill_artifact_version', 'artifact_ref', 'artifact_hash', 'learning'),
    'semantier.evidence.v1': (Evidence, 'evidence', 'artifact_ref', 'artifact_hash', 'evidence'),
    'semantier.ontology.v1': (Ontology, 'ontology', 'artifact_ref', 'artifact_hash', 'semantic'),
    'semantier.context_graph.v1': (ContextGraph, 'context_graph', 'artifact_ref', 'artifact_hash', 'semantic'),
    'semantier.decision_record.v1': (DecisionRecord, 'decision_record', 'artifact_ref', 'artifact_hash', 'runtime'),
    'semantier.change_candidate.v1': (ChangeCandidate, 'change_candidate', 'artifact_ref', 'artifact_hash', 'learning'),
    'semantier.change_candidate.v2': (ChangeCandidateV2, 'change_candidate', 'artifact_ref', 'artifact_hash', 'learning'),
}


def _pin(value, spec):
    return ArtifactPinV1(artifact_kind=spec[1], artifact_ref=getattr(value, spec[2]), artifact_hash=getattr(value, spec[3]))


def _bindings(value):
    """Native field names are the edge evidence, not inferred lifecycle stages."""
    if isinstance(value, OntologyBuildManifestV1):
        for pin in value.child_artifacts:
            yield 'child_artifacts', pin, False
        if value.predecessor_build_ref:
            yield 'predecessor_build', ArtifactPinV1(artifact_kind='ontology_build_manifest',
                artifact_ref=value.predecessor_build_ref, artifact_hash=value.predecessor_build_hash), False
    elif isinstance(value, ExploratoryGraphSnapshotV1):
        yield 'source_identity', ArtifactPinV1(artifact_kind='source_identity',
            artifact_ref=value.source_identity_ref, artifact_hash=value.source_hash), False
        yield 'canonical_source_ir', ArtifactPinV1(artifact_kind='canonical_source_ir',
            artifact_ref=value.canonical_source_ir_ref, artifact_hash=value.canonical_source_ir_hash), False
        if value.predecessor_snapshot_ref:
            yield 'predecessor_snapshot', ArtifactPinV1(artifact_kind='exploratory_graph_snapshot',
                artifact_ref=value.predecessor_snapshot_ref, artifact_hash=value.predecessor_snapshot_hash), False
    elif isinstance(value, (OntologyAdmissionDecisionV1, OntologyReleaseBundleV1)):
        yield 'build', ArtifactPinV1(artifact_kind='ontology_build_manifest',
            artifact_ref=value.build_ref, artifact_hash=value.build_hash), False
        if isinstance(value, OntologyReleaseBundleV1):
            yield 'admission_decision', ArtifactPinV1(artifact_kind='ontology_admission_decision',
                artifact_ref=value.admission_decision_ref, artifact_hash=value.admission_decision_hash), False
    elif isinstance(value, RuntimeTaskContextV1):
        for field in ('goal', 'parent', 'branch_origin', 'transition'):
            if (pin := getattr(value, field)):
                yield field, pin, False
        yield 'world.source_graph', value.world.source_graph, False
        if (formation := getattr(value, 'formation', None)):
            yield 'formation', formation, False
        for pin in value.semantic_pins:
            yield 'semantic_pins', pin, False
        for element in value.elements:
            yield 'elements.source_artifact', element.source_artifact, False
            if element.admission_receipt:
                yield 'elements.admission_receipt', element.admission_receipt, False
            for evidence in element.evidence:
                yield 'elements.evidence', evidence, False
            if element.derivation:
                yield 'elements.derivation', element.derivation, False
    elif isinstance(value, RuntimeExecutionAdmissionV1):
        for field in ('context', 'world', 'provider_request', 'certified_plan', 'epoch', 'activation_snapshot'):
            if (pin := getattr(value, field)):
                yield field, pin, False
    elif isinstance(value, RuntimeEvaluationRunV1):
        for field in ('admission', 'root_run', 'parent_run'):
            if (pin := getattr(value, field)):
                yield field, pin, False
    elif isinstance(value, RuntimeEvaluationRequestOutcomeV1):
        for field in ('run_pin', 'execution_admission_pin', 'state_pin', 'epoch_pin', 'authority_release_receipt_pin'):
            if (pin := getattr(value, field)):
                yield field, pin, False
        for pin in value.result_pins:
            yield 'result_pins', pin, False
    elif isinstance(value, (LearnedSkillProposalV1, LearnedSkillAssessmentV1, LearnedSkillPublicationReceiptV1)):
        for field in ('candidate', 'version', 'predecessor'):
            if (pin := getattr(value, field, None)):
                yield field, pin, False
    elif isinstance(value, ProjectStateProjectionV1):
        yield 'memory_candidate', value.memory_candidate, False
    elif isinstance(value, MemoryCandidateV1):
        for source in value.source_objects:
            yield 'source_objects.source', source.source, False
        for evidence in value.evidence:
            yield 'evidence', evidence, False
    elif isinstance(value, SkillArtifactVersionV1):
        if value.predecessor:
            yield 'predecessor', value.predecessor, False
    elif isinstance(value, Evidence):
        yield 'source', ArtifactPinV1(artifact_kind='source_identity',
            artifact_ref=value.source_ref, artifact_hash=value.source_hash), False
        for pin in value.provenance:
            yield 'provenance', pin, False
    elif isinstance(value, ContextGraph):
        for field in ('ontology', 'predecessor', 'admission_decision'):
            if (pin := getattr(value, field)):
                yield field, pin, False
        for pin in value.evidence:
            yield 'evidence', pin, False
        for field in ('source_inputs', 'selected_inputs'):
            for pin in getattr(value.formation, field):
                yield 'formation.' + field, pin, False
        for excluded in value.formation.excluded_inputs:
            yield 'formation.excluded_inputs', excluded.input, False
        for field in ('formation_policy', 'predecessor_context', 'applied_change_candidate'):
            if (pin := getattr(value.formation, field)):
                yield 'formation.' + field, pin, False
    elif isinstance(value, WorkingContext):
        for field in ('context_graph', 'ontology', 'selection_policy'):
            yield field, getattr(value, field), False
        for pin in value.evidence:
            yield 'evidence', pin, False
    elif isinstance(value, DecisionRecord):
        if value.reasoning.working_context:
            yield 'reasoning.working_context', value.reasoning.working_context.pin(), False
        for field in ('subject', 'context', 'ontology'):
            yield field, getattr(value, field), False
        for pin in value.evidence:
            yield 'evidence', pin, False
        if value.outcome_observation:
            yield 'outcome_observation', value.outcome_observation, True
    elif isinstance(value, (ChangeCandidate, ChangeCandidateV2)):
        if value.predecessor:
            yield 'predecessor', value.predecessor, False
        profile = value if isinstance(value, ChangeCandidate) else value.profile
        for pin in getattr(profile, 'basis_refs', ()):
            yield 'profile.basis_refs', pin, False
        origin = getattr(profile, 'origin', None)
        for field in ('context', 'decision', 'workflow'):
            if origin and (pin := getattr(origin, field, None)):
                yield 'origin.' + field, pin, False
        for field in ('evidence', 'commentary_refs'):
            for pin in getattr(origin, field, ()):
                yield 'origin.' + field, pin, False
        for observation in getattr(profile, 'observations', ()):
            yield 'observations.case', observation.case, False
        for pin in getattr(profile, 'successor_inputs', ()):
            yield 'successor_inputs', pin, True


class MemoryWorkbenchProjectionService:
    def __init__(self, *, store, learning=None, sources=None, documents=None):
        self.store = store
        self.learning = learning
        self.sources = sources
        self.documents = documents

    @staticmethod
    def _scope(context):
        if not (context.authenticated and context.user_id and context.organization_id
                and context.workspace_id):
            raise PermissionError('active authenticated workspace required')
        if (context.workspace_root.resolve() != workspace_root_path(context.workspace_id).resolve()
                or context.hermes_home.resolve() != workspace_hermes_home_path(context.workspace_id).resolve()):
            raise PermissionError('outside governed workspace')

    def _read(self, context, pin):
        if pin.artifact_kind == 'documentary_representation' and self.documents is not None:
            try:
                row = self.documents.resolve_active_source_representation(artifact_ref=pin.artifact_ref,
                    organization_id=context.organization_id, workspace_id=context.workspace_id)
            except ValueError as exc:
                if str(exc) == 'active governed normalized representation not found':
                    return None, None
                raise
            if 'sha256:' + row['artifact_content_hash'].removeprefix('sha256:') != pin.artifact_hash:
                raise ValueError('exact documentary representation hash mismatch')
            return row, (dict, 'documentary_representation', 'artifact_ref', 'artifact_content_hash', 'documentary')
        if pin.artifact_kind == 'source_identity' and self.sources is not None:
            try:
                _, identity = self.sources.resolve_source_identity_lineage(
                    source_identity_ref=pin.artifact_ref, organization_id=context.organization_id,
                    workspace_id=context.workspace_id)
            except ValueError as exc:
                if str(exc) == 'SOURCE_IDENTITY_LINEAGE_NOT_FOUND_OR_AMBIGUOUS':
                    return None, None
                raise
            if identity.source_hash != pin.artifact_hash:
                raise ValueError('exact source identity hash mismatch')
            return identity, (SourceIdentity, 'source_identity', 'source_identity_ref', 'source_hash', 'documentary_source')
        raw = self.store.get_raw(organization_id=context.organization_id, workspace_id=context.workspace_id,
                                artifact_ref=pin.artifact_ref)
        if raw is None:
            if pin.artifact_kind == 'working_context':
                for decision_pin in self.store.list_artifact_pins(organization_id=context.organization_id,
                        workspace_id=context.workspace_id, schema_version='semantier.decision_record.v1'):
                    decision, _ = self._read(context, decision_pin.model_copy(update={'artifact_kind': 'decision_record'}))
                    value = decision.reasoning.working_context
                    if value and value.value_ref == pin.artifact_ref:
                        if value.pin() != pin:
                            raise ValueError('exact embedded WorkingContext hash mismatch')
                        if (value.organization_id, value.workspace_id) != (context.organization_id, context.workspace_id):
                            raise PermissionError('outside recorded WorkingContext scope')
                        return value, (WorkingContext, 'working_context', 'value_ref', 'value_hash', 'runtime')
            return None, None
        if raw['artifact_hash'] != pin.artifact_hash:
            raise ValueError('exact artifact hash mismatch')
        spec = _OWNERS.get(raw['schema_version'])
        if spec is None:
            return None, None
        value = spec[0].model_validate(raw['payload'])
        if (value.organization_id, value.workspace_id) != (context.organization_id, context.workspace_id):
            raise PermissionError('outside native artifact owner scope')
        if isinstance(value, (SkillArtifactVersionV1, LearnedSkillProposalV1, LearnedSkillAssessmentV1, LearnedSkillPublicationReceiptV1, RetainedContextItemChangeV1)) and value.owner_ref != context.user_id:
            raise PermissionError('outside personal procedure owner scope')
        personal_owner = getattr(getattr(value, 'profile', None), 'owner_ref', None)
        if personal_owner and personal_owner != context.user_id:
            raise PermissionError('outside personal candidate owner scope')
        if _pin(value, spec) != pin:
            raise ValueError('exact artifact kind/ref/hash mismatch')
        return value, spec

    def project_states(self, *, context, as_of):
        self._scope(context)
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError('as_of requires a timezone')
        as_of = as_of.astimezone(timezone.utc)
        items = []
        spec = _OWNERS['project_state_projection.v1']
        for listed in self.store.list_artifact_pins(organization_id=context.organization_id,
                workspace_id=context.workspace_id, schema_version='project_state_projection.v1'):
            pin = listed.model_copy(update={'artifact_kind': spec[1]})
            projection, _ = self._read(context, pin)
            try:
                candidate, _ = self._read(context, projection.memory_candidate)
                candidate_status = 'unavailable'
            except PermissionError:
                candidate, candidate_status = None, 'forbidden'
            except ValueError:
                candidate, candidate_status = None, 'unavailable'
            row = {'projection': projection.model_dump(mode='json'), 'status': candidate_status,
                   'candidate': None, 'effectivity': None}
            if candidate is not None:
                ProjectMemoryEvidenceBinder.bind(projection=projection, candidate=candidate)
                row.update(status='available', candidate=candidate.model_dump(mode='json'),
                    effectivity=SemanticaProjectMemoryAdapter().effectivity_view(
                        projection=projection, candidate=candidate, as_of=as_of).model_dump(mode='json'))
            items.append(row)
        return {'as_of': as_of.isoformat().replace('+00:00','Z'), 'items': items,
                'coverage': 'persisted_project_state', 'authority': 'derived_non_authority'}

    def semantic_records(self, *, context):
        self._scope(context)
        items = []
        for schema in ('ontology_build_manifest.v1', 'exploratory_graph_snapshot.v1',
                       'semantier.ontology.v1', 'semantier.context_graph.v1'):
            spec = _OWNERS[schema]
            for listed in self.store.list_artifact_pins(organization_id=context.organization_id,
                    workspace_id=context.workspace_id, schema_version=schema):
                pin = listed.model_copy(update={'artifact_kind': spec[1]})
                try:
                    value, _ = self._read(context, pin)
                    status = 'available'
                except (ValueError, PermissionError):
                    value, status = None, 'unavailable'
                items.append({'pin': pin.model_dump(mode='json'), 'owner': spec[4], 'status': status,
                    'owner_href': '/ontology-workbench2?' + urlencode({'buildRef': value.build_ref, 'view': 'graph'})
                        if isinstance(value, OntologyBuildManifestV1) else None})
        return {'items': items, 'coverage': 'persisted_native_semantic_artifacts',
                'authority': 'read_projection_non_authority'}

    def lineage(self, *, context, subject):
        self._scope(context)
        # Validate the selected exact identity before traversing even if it is unavailable.
        self._read(context, subject)
        scan_errors = []
        edges = []
        for schema, spec in _OWNERS.items():
            for listed in self.store.list_artifact_pins(organization_id=context.organization_id,
                    workspace_id=context.workspace_id, schema_version=schema):
                pin = listed.model_copy(update={'artifact_kind':spec[1]})
                try:
                    value, _ = self._read(context, pin)
                except PermissionError:
                    # A workspace-scoped index does not grant personal-owner visibility.
                    continue
                except ValueError:
                    scan_errors.append({'pin': pin.model_dump(mode='json'), 'status': 'unavailable',
                                        'reason': 'owner_integrity_or_scope_unavailable'})
                    continue
                bindings = list(_bindings(value))
                if isinstance(value, DecisionRecord) and value.reasoning.working_context:
                    embedded = value.reasoning.working_context
                    embedded_pin = embedded.pin()
                    for field, origin, _ in _bindings(embedded):
                        edges.append({'source': origin.artifact_ref, 'target': embedded_pin.artifact_ref,
                            'source_pin': origin.model_dump(mode='json'), 'target_pin': embedded_pin.model_dump(mode='json'),
                            'binding': 'reasoning.working_context.' + field, 'receipt': pin.model_dump(mode='json')})
                if isinstance(value, RetainedContextItemChangeV1) and value.revision > 1:
                    # Only an exact retained predecessor with the same owner/item and
                    # sequential revision establishes this edge. Initial content digests
                    # have no artifact reference and must not be invented as pins.
                    for previous_pin in self.store.list_artifact_pins(organization_id=context.organization_id,
                            workspace_id=context.workspace_id, schema_version='retained_context_item_change.v1'):
                        previous_pin = previous_pin.model_copy(update={'artifact_kind': 'retained_context_item_change'})
                        if previous_pin.artifact_hash != value.expected_revision:
                            continue
                        previous, _ = self._read(context, previous_pin)
                        if (previous.owner_ref, previous.item_ref, previous.revision, previous.content) == (
                                value.owner_ref, value.item_ref, value.revision - 1, value.prior_content):
                            bindings.append(('expected_revision', previous_pin, False))
                for field, other, forward in bindings:
                    source, target = (pin, other) if forward else (other, pin)
                    edges.append({'source': source.artifact_ref, 'target': target.artifact_ref,
                                  'source_pin': source.model_dump(mode='json'), 'target_pin': target.model_dump(mode='json'),
                                  'binding': field, 'receipt': pin.model_dump(mode='json')})
        if self.sources is not None and self.documents is not None:
            source_pins = {ArtifactPinV1.model_validate(edge[side + '_pin']) for edge in edges
                for side in ('source', 'target') if edge[side + '_pin']['artifact_kind'] == 'source_identity'}
            if subject.artifact_kind == 'source_identity':
                source_pins.add(subject)
            if subject.artifact_kind == 'documentary_representation':
                document, _ = self._read(context, subject)
                if document is not None:
                    for identity in self.sources.list_source_identities(organization_id=context.organization_id,
                            workspace_id=context.workspace_id, lineage_id=document['lineage_id']):
                        if identity.artifact_ref == document['original_path'] and identity.source_hash.removeprefix('sha256:') == document['original_content_hash'].removeprefix('sha256:'):
                            source_pins.add(ArtifactPinV1(artifact_kind='source_identity',artifact_ref=identity.source_identity_ref,artifact_hash=identity.source_hash))
            for source_pin in source_pins:
                try:
                    source, _ = self._read(context, source_pin)
                    if source is None:
                        continue
                    lineage_id, _ = self.sources.resolve_source_identity_lineage(source_identity_ref=source_pin.artifact_ref,
                        organization_id=context.organization_id,workspace_id=context.workspace_id)
                    for listed in self.documents.list_active_source_representations(lineage_id,
                            organization_id=context.organization_id,workspace_id=context.workspace_id):
                        document_pin=ArtifactPinV1(artifact_kind='documentary_representation',artifact_ref=listed['artifact_ref'],
                            artifact_hash='sha256:'+listed['artifact_content_hash'].removeprefix('sha256:'))
                        document, _ = self._read(context,document_pin)
                        if document['original_path'] != source.artifact_ref or document['original_content_hash'].removeprefix('sha256:') != source.source_hash.removeprefix('sha256:'):
                            continue
                        edges.append({'source':source_pin.artifact_ref,'target':document_pin.artifact_ref,
                            'source_pin':source_pin.model_dump(mode='json'),'target_pin':document_pin.model_dump(mode='json'),
                            'binding':'normalized_from','receipt':ArtifactPinV1(artifact_kind='source_representation_edge',
                                artifact_ref=document['representation_id'],artifact_hash='sha256:'+document['content_hash'].removeprefix('sha256:')).model_dump(mode='json')})
                except ValueError:
                    scan_errors.append({'pin':source_pin.model_dump(mode='json'),'status':'unavailable','reason':'documentary_owner_binding_unavailable'})
        # A reference alone is not an identity: source versions can share a ref.
        # Traverse native kind/ref/hash triples and retain both version branches.
        def key(pin):
            if isinstance(pin, ArtifactPinV1):
                return (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)
            return (pin['artifact_kind'], pin['artifact_ref'], pin['artifact_hash'])
        identities = {key(subject)}
        while True:
            expanded = identities | {key(e[side + '_pin']) for e in edges
                if key(e['source_pin']) in identities or key(e['target_pin']) in identities
                for side in ('source', 'target')}
            if expanded == identities:
                break
            identities = expanded
        selected = [e for e in edges if key(e['source_pin']) in identities and key(e['target_pin']) in identities]
        pins = {key(subject): subject}
        for e in selected:
            for side in ('source', 'target'):
                pin = ArtifactPinV1.model_validate(e[side + '_pin'])
                pins[key(pin)] = pin
        nodes=[]
        for identity, pin in sorted(pins.items()):
            try:
                value, spec = self._read(context, pin)
                status = 'available' if value is not None else 'unavailable'
            except PermissionError:
                value, spec, status = None, None, 'forbidden'
            except ValueError:
                value, spec, status = None, None, 'unavailable'
            nodes.append({'pin':pin.model_dump(mode='json'), 'owner':spec[4] if spec else pin.artifact_kind,
                'status':status,
                'native':(value if isinstance(value, dict) else value.model_dump(mode='json')) if value is not None else None})
        for node in nodes:
            node['owner_state'] = None
            if self.learning and node['status'] == 'available' and node['pin']['artifact_kind'] == 'change_candidate' and node['native'].get('profile', {}).get('owner_ref'):
                try:
                    node['owner_state'] = self.learning.inspect(context=context, kind='candidate', ref=node['pin']['artifact_ref'])
                except (ValueError, PermissionError, LookupError):
                    node['owner_state'] = {'status': 'unavailable', 'reason': 'learning_owner_state_unavailable'}
            node['owner_href'] = None
            if node['status'] != 'available':
                continue
            native = node['native']
            kind = node['pin']['artifact_kind']
            if kind == 'source_identity':
                node['owner_href'] = '/memory?' + urlencode({'lens': 'documents', 'knowledge_view': 'source', 'knowledge_path': native['artifact_ref'], 'kind': node['pin']['artifact_kind'], 'ref': node['pin']['artifact_ref'], 'hash': node['pin']['artifact_hash']})
            elif kind == 'documentary_representation':
                node['owner_href'] = '/memory?' + urlencode({'lens': 'documents', 'knowledge_view': 'source', 'knowledge_path': native['artifact_path'], 'kind': kind, 'ref': node['pin']['artifact_ref'], 'hash': node['pin']['artifact_hash']})
            elif kind == 'ontology_build_manifest':
                node['owner_href'] = '/ontology-workbench2?' + urlencode({'buildRef': native['build_ref'], 'view': 'lineage'})
            elif kind == 'exploratory_graph_snapshot':
                builds = [e for e in selected if e['source_pin'] == node['pin']
                          and e['binding'] == 'child_artifacts' and e['target_pin']['artifact_kind'] == 'ontology_build_manifest']
                if len(builds) == 1:
                    node['owner_href'] = '/ontology-workbench2?' + urlencode({'buildRef': builds[0]['target'], 'view': 'graph'})
            elif kind in ('skill_artifact_version', 'change_candidate'):
                # Only the personal procedural profile is handled by Learning.
                if kind == 'skill_artifact_version' or native.get('profile', {}).get('owner_ref'):
                    node['owner_href'] = '/learning?' + urlencode({'kind': 'procedure' if kind == 'skill_artifact_version' else 'candidate', 'ref': node['pin']['artifact_ref']})
            elif kind == 'retained_context_item_change':
                node['owner_href'] = '/learning?' + urlencode({'kind': 'context', 'ref': native['item_ref']})
        return {'subject':subject.model_dump(mode='json'), 'nodes':nodes,
                'edges':sorted(selected,key=lambda e:(e['source'],e['target'],e['binding'])),
                'coverage':list(_OWNERS), 'owner_errors':scan_errors,
                'gaps':{'documentary':'existing_wiki_provenance_owner; no_inferred_synthesis_edges',
                    'recorded_use': 'available_owner_receipts' if any(
                        n['status'] == 'available' and n['pin']['artifact_kind'] == 'runtime_evaluation_request_outcome'
                        for n in nodes) else 'unknown_without_owner_receipt'},
                'authority':'read_projection_non_authority'}
