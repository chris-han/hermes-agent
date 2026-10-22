"""Pure, reviewed seed extension over archived source and imported definition pins."""
from __future__ import annotations

from dataclasses import dataclass

from contracts.ontology_artifact_integrity import (
    ExploratoryGraphSnapshotIntegrityV1,
    create_exploratory_snapshot_integrity,
)
from contracts.canonical_source_ir import CanonicalSourceDocument
from contracts.graph_intent_brief import GraphIntentBrief
from contracts.evidence_location import SourceIdentity
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_for_block
from contracts.ontology_build_inputs import OntologyBuildInputEnvelopeV1
from contracts.ontology_hub_snapshot import OntologyHubSnapshotV1
from contracts.ontology_seeds import OntologyCapabilityProfileV1
from contracts.ontology_building import (
    OntologyBuildManifestV1, ExploratoryGraphNodeV1, ExploratoryGraphEdgeV1, ExploratoryGraphSnapshotV1,
    CandidateGraphDiffProjectionV1, CandidateGraphLineageProjectionV1,
)
from contracts.ontology_seed_alignment import (
    INPUT_PIN_FIELDS, ExactAlignmentInputV1, OntologySeedAlignmentCommandV1,
    SeedDefinitionContributionV1, SeedDerivationV1, AlignmentConflictV1,
    SourceGroundedSeedDeltaV1, OntologySeedComparisonV1,
)
from eos.artifact_hashing import content_hash, canonical_json
from services.ontology_lineage_service import OntologyLineageService


@dataclass(frozen=True)
class OntologySeedAlignmentResult:
    alignment: OntologySeedAlignmentCommandV1
    delta: SourceGroundedSeedDeltaV1
    successor: ExploratoryGraphSnapshotV1 | None
    successor_integrity: ExploratoryGraphSnapshotIntegrityV1 | None
    diff: CandidateGraphDiffProjectionV1 | None
    lineage: CandidateGraphLineageProjectionV1
    seed_comparison: OntologySeedComparisonV1


def _require_child(manifest, kind, ref, digest):
    pins = [pin for pin in manifest.child_artifacts if pin.artifact_kind == kind]
    if len(pins) != 1 or (pins[0].artifact_ref, pins[0].artifact_hash) != (ref, digest):
        raise ValueError(f"alignment input is not the exact manifest {kind} pin")


def _validate_inputs(command, inputs, actor):
    if not actor or actor != command.actor_ref:
        raise PermissionError("alignment actor does not match governed actor context")
    scope = (command.organization_id, command.workspace_id)
    for name, value in inputs.items():
        if (value.organization_id, value.workspace_id) != scope:
            raise PermissionError("alignment inputs cross governed tenant scope")
        if ExactAlignmentInputV1.of(value, *INPUT_PIN_FIELDS[name]) != getattr(command.inputs, name):
            raise ValueError(f"alignment {name} claimed identity or full payload hash mismatch")
    manifest, snapshot, envelope = inputs['baseline_manifest'], inputs['baseline_snapshot'], inputs['input_envelope']
    seed, profile = inputs['seed_snapshot'], inputs['seed_profile']
    if snapshot.build_ref != manifest.build_ref or envelope.build_ref != manifest.build_ref:
        raise ValueError("source-first build input identity mismatch")
    if command.reviewed_at < max(snapshot.created_at, manifest.created_at):
        raise ValueError("alignment review predates the pinned source-first build")
    for kind, artifact, ref, digest in (
        ('exploratory_graph_snapshot', snapshot, 'snapshot_ref', 'snapshot_hash'),
        ('ontology_build_inputs', envelope, 'envelope_ref', 'envelope_hash'),
    ):
        _require_child(manifest, kind, getattr(artifact, ref), getattr(artifact, digest))
    for archived in envelope.audit_inputs():
        _require_child(manifest, archived.artifact_kind, archived.artifact_ref, archived.artifact_hash)
    brief = GraphIntentBrief.from_dict(envelope.graph_intent)
    binding = brief.source_binding
    if (snapshot.source_identity_ref, snapshot.source_hash, snapshot.canonical_source_ir_ref, snapshot.canonical_source_ir_hash) != (
        binding.source_identity_ref, binding.source_hash, binding.canonical_source_ir_ref, binding.canonical_source_ir_hash):
        raise ValueError("baseline snapshot does not bind the archived canonical source")
    if (profile.ontology_ref, profile.version, profile.source_content_hash, profile.graph_content_hash,
        profile.identity.provenance.get('snapshot_ref'), profile.identity.provenance.get('snapshot_hash')) != (
        seed.ontology_ref, seed.version, seed.source_content_hash, seed.graph_content_hash, seed.snapshot_ref, seed.snapshot_hash):
        raise ValueError("seed profile does not bind the exact imported seed snapshot")


def _canonical_support(canonical, envelope):
    blocks = {block.block_id: block for block in canonical.blocks}
    if len(blocks) != len(canonical.blocks):
        raise ValueError("ambiguous canonical block identity")
    locators = {block_id: [] for block_id in blocks}
    for mapping in canonical.source_map.mappings:
        for block_id in mapping.block_ids:
            if block_id in locators:
                locators[block_id].extend(locator.model_dump(mode='json') for locator in mapping.source_locators)
    aliases = {f"evidence:canonical-block:{block_id.removeprefix('sha256:')}":
               f"evidence:canonical-block:{block_id.removeprefix('sha256:')}" for block_id in blocks}
    if envelope.compiler_version in {"semantier_source_first_exploratory_v2", "semantier_source_first_exploratory_v3"}:
        identity = SourceIdentity.model_validate(envelope.source_identity)
        aliases = {f"evidence:canonical-block:{block.block_id.removeprefix('sha256:')}":
                   evidence_ref_for_block(canonical, block, source_identity=identity).evidence_ref
                   for block in blocks.values()}
    return blocks, locators, aliases


def _quotes(quotes, blocks, locators, aliases):
    refs = set()
    for anchor in quotes:
        block = blocks.get(anchor.block_id)
        alias = f"evidence:canonical-block:{anchor.block_id.removeprefix('sha256:')}"
        if block is None or anchor.evidence_ref not in {alias, aliases.get(alias)}:
            raise ValueError("alignment quote has an unknown canonical block/evidence ref")
        if anchor.source_locator.model_dump(mode='json') not in locators[anchor.block_id]:
            raise ValueError("alignment quote source locator does not match archived canonical IR")
        text = block.content if isinstance(block.content, str) else canonical_json(block.content)
        if not anchor.quote.strip() or anchor.quote not in text:
            raise ValueError("alignment quote text does not occur in the exact canonical block")
        refs.add(aliases[alias])
    return tuple(sorted(refs))


def _frame_participants(frame, participants, frame_kinds, predicate_nodes, evidence_refs):
    links = []
    payload = frame.model_dump(mode='json')
    simple = {'actor_ref', 'object_ref', 'qualified_term', 'event_time_ref', 'left_entity_ref', 'right_entity_ref'}
    for field in sorted(simple & payload.keys()):
        ref = payload[field]
        if ref not in participants:
            raise ValueError(f"normative {field} requires an explicit source participant mapping")
        links.append((field, participants[ref]))
    for field in ('actor_scope', 'object_scope'):
        for index, ref in enumerate(payload.get(field, ())):
            if ref not in participants:
                raise ValueError(f"normative {field} contains an unmapped seed participant")
            links.append((f'{field}.{index}', participants[ref]))
    rule_fields = {'applies_to_rule_ref': 'rule', 'higher_priority_ref': 'rule', 'lower_priority_ref': 'rule'}
    frame_lists = {'exception_refs': 'exception', 'scope_refs': 'scope', 'qualification_refs': 'qualification',
                   'temporal_condition_refs': 'temporal_condition', 'precedence_refs': 'precedence'}
    for field, kind in rule_fields.items():
        if field in payload:
            ref = payload[field]
            if frame_kinds.get(ref) != kind:
                raise ValueError(f"normative {field} requires an explicitly contributed {kind} frame")
            links.append((field, participants[ref]))
    for field, kind in frame_lists.items():
        for index, ref in enumerate(payload.get(field, ())):
            if frame_kinds.get(ref) != kind:
                raise ValueError(f"normative {field} contains an unmapped frame")
            links.append((f'{field}.{index}', participants[ref]))
    for condition in (getattr(frame, 'conditions', None) or ()):
        if condition.fact_ref not in participants or not set(condition.evidence_refs) <= set(evidence_refs):
            raise ValueError("normative condition requires mapped canonical participant and evidence")
        links.append((f'condition:{condition.predicate_id}', participants[condition.fact_ref]))
    for index, ref in enumerate(payload.get('condition_refs', ())):
        if ref not in predicate_nodes:
            raise ValueError("precedence condition is not explicitly represented")
        links.append((f'condition_refs.{index}', predicate_nodes[ref]))
    if frame.frame_kind == 'precedence' and frame.higher_priority_ref == frame.lower_priority_ref:
        raise ValueError("precedence cannot prioritize a rule over itself")
    if frame.frame_kind == 'entity_distinction' and frame.left_entity_ref == frame.right_entity_ref:
        raise ValueError("entity distinction requires different mapped participants")
    if frame.frame_kind == 'scope' and not (frame.actor_scope or frame.object_scope):
        raise ValueError("scope requires an explicit actor or object scope")
    return links


def _scalar_slots(value, path=''):
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            yield from _scalar_slots(child, f'{path}.{key}' if path else key)
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _scalar_slots(child, f'{path}.{index}')
    elif value is not None:
        yield path, str(value)


def _frame_shapes(frames, nodes, edges):
    if not frames:
        return 'NOT_APPLICABLE'
    from rdflib import Graph, Namespace, RDF, URIRef, Literal
    from pyshacl import validate
    ns = Namespace('urn:semantier:seed-alignment:')
    data = Graph()
    frame_nodes = [node for node in nodes if node.node_kind.startswith('normative_frame:')]
    for node in frame_nodes:
        subject = URIRef('urn:node:' + node.node_id)
        data.add((subject, RDF.type, ns.Frame))
        for ref in node.evidence_refs:
            data.add((subject, ns.evidence, Literal(ref)))
        for edge in edges:
            if edge.source_node_id == node.node_id and edge.relation_type == 'has_semantic_slot':
                data.add((subject, ns.slot, URIRef('urn:node:' + edge.target_node_id)))
    shapes = Graph().parse(data='''@prefix sh: <http://www.w3.org/ns/shacl#> .
      @prefix a: <urn:semantier:seed-alignment:> .
      a:FrameShape a sh:NodeShape; sh:targetClass a:Frame;
        sh:property [sh:path a:evidence; sh:minCount 1];
        sh:property [sh:path a:slot; sh:minCount 1].''', format='turtle')
    conforms, _, _ = validate(data, shacl_graph=shapes, inference='none', abort_on_first=False)
    if not conforms:
        raise ValueError("source-grounded normative frame structural validation failed")
    return 'PASS_STRUCTURAL_ONLY'


class OntologySeedAlignmentService:
    def apply(self, *, command: OntologySeedAlignmentCommandV1, governed_actor_ref: str,
              baseline_manifest: OntologyBuildManifestV1, baseline_snapshot: ExploratoryGraphSnapshotV1,
              input_envelope: OntologyBuildInputEnvelopeV1,
              seed_snapshot: OntologyHubSnapshotV1, seed_profile: OntologyCapabilityProfileV1) -> OntologySeedAlignmentResult:
        command = OntologySeedAlignmentCommandV1.model_validate(command.model_dump(mode='json'))
        typed_inputs = {
            'baseline_manifest': (baseline_manifest, OntologyBuildManifestV1),
            'baseline_snapshot': (baseline_snapshot, ExploratoryGraphSnapshotV1),
            'input_envelope': (input_envelope, OntologyBuildInputEnvelopeV1),
            'seed_snapshot': (seed_snapshot, OntologyHubSnapshotV1),
            'seed_profile': (seed_profile, OntologyCapabilityProfileV1),
        }
        inputs = {name: contract.model_validate(value.model_dump(mode='json')) for name, (value, contract) in typed_inputs.items()}
        _validate_inputs(command, inputs, governed_actor_ref)
        baseline, seed = inputs['baseline_snapshot'], inputs['seed_snapshot']
        canonical = CanonicalSourceDocument.model_validate(inputs['input_envelope'].canonical_source_ir)
        blocks, locators, aliases = _canonical_support(canonical, inputs['input_envelope'])
        known_evidence = set(aliases.values())
        source_nodes = {node.node_id: node for node in baseline.nodes}
        if len(source_nodes) != len(baseline.nodes) or any(not set(node.evidence_refs) <= known_evidence for node in baseline.nodes):
            raise ValueError("baseline source nodes have ambiguous identity or unbound canonical evidence")
        decisions, contributions, conflicts = {}, [], []
        disposition = {'EXACT': 'reused', 'BROADER': 'specialized', 'NARROWER': 'specialized', 'RELATED': 'extended', 'DISTINCT': 'rejected', 'UNRESOLVED': 'unresolved'}
        for decision in sorted(command.decisions, key=lambda item: item.alignment_id):
            if decision.source_node_id not in source_nodes:
                raise ValueError("alignment source definition is not in the exact source-first snapshot")
            triples = tuple(triple for triple in seed.canonical_inventory if triple.subject == decision.seed_iri)
            if not triples or ':' not in decision.seed_iri:
                raise ValueError("alignment seed IRI is not in the exact imported definition inventory")
            refs = _quotes(decision.canonical_evidence, blocks, locators, aliases)
            if not set(refs) <= set(source_nodes[decision.source_node_id].evidence_refs):
                raise ValueError("alignment evidence does not support its selected source definition")
            state = disposition[decision.relation]
            code = None
            if decision.canonical_consistency == 'CONFLICT':
                state, code = 'rejected', 'canonical_conflict'
            elif decision.canonical_consistency == 'UNRESOLVED':
                state, code = 'unresolved', 'unresolved_consistency'
            elif decision.relation == 'DISTINCT':
                code = 'distinct_definition'
            if code:
                conflicts.append(AlignmentConflictV1(alignment_id=decision.alignment_id, conflict_kind=code,
                    message=decision.rationale, canonical_evidence_refs=refs))
            contributions.append(SeedDefinitionContributionV1(alignment_id=decision.alignment_id, seed_iri=decision.seed_iri,
                source_node_id=decision.source_node_id, relation=decision.relation, disposition=state,
                seed_definition=triples, canonical_evidence_refs=refs, rationale=decision.rationale))
            decisions[decision.alignment_id] = (decision, state)
        accepted = [extension for extension in command.extensions if decisions[extension.alignment_id][1] not in ('rejected', 'unresolved')]
        prefix = command.alignment_hash[7:31]
        ids = {extension.extension_id: f'seed-extension:{prefix}:{extension.extension_id}' for extension in accepted}
        frames = [extension.frame for extension in accepted if extension.extension_kind == 'normative_frame']
        frame_kinds = {frame.frame_id: frame.frame_kind for frame in frames}
        if len(frame_kinds) != len(frames):
            raise ValueError("contributed normative frame ids must be unique")
        if set(frame_kinds) & (set(source_nodes) | set(ids)) or set(ids) & set(source_nodes):
            raise ValueError("source, extension and frame participant identities must be disjoint")
        participants = {node_id: node_id for node_id in source_nodes}
        participants.update(ids)
        participants.update({extension.frame.frame_id: ids[extension.extension_id] for extension in accepted if extension.extension_kind == 'normative_frame'})
        predicate_nodes = {}
        for extension in accepted:
            if extension.extension_kind != 'normative_frame':
                continue
            for index, predicate in enumerate((getattr(extension.frame, 'conditions', None) or ())):
                if predicate.predicate_id in predicate_nodes:
                    raise ValueError("normative predicate ids must be globally unique")
                predicate_nodes[predicate.predicate_id] = f'{ids[extension.extension_id]}:slot:conditions.{index}.predicate_id'
        added_nodes, added_edges, derivations, applied, compiled_frames = [], [], [], [], []
        quote_count = sum(len(decision.canonical_evidence) for decision in command.decisions)
        def edge(source, target, relation, refs):
            digest = content_hash({'source': source, 'target': target, 'relation': relation})
            added_edges.append(ExploratoryGraphEdgeV1(edge_id=f'seed-source-edge:{digest}', source_node_id=source,
                target_node_id=target, relation_type=relation, evidence_refs=refs))
        for extension in sorted(accepted, key=lambda item: item.extension_id):
            decision, _ = decisions[extension.alignment_id]
            refs = _quotes(extension.canonical_evidence, blocks, locators, aliases)
            quote_count += len(extension.canonical_evidence)
            node_id = ids[extension.extension_id]
            if extension.extension_kind == 'definition':
                label, kind = extension.definition, f'source_definition:{extension.definition_kind}'
            else:
                frame = extension.frame
                if inputs['input_envelope'].compiler_version in {"semantier_source_first_exploratory_v2", "semantier_source_first_exploratory_v3"}:
                    def normalize_refs(value):
                        if isinstance(value, dict):
                            return {key: ([aliases.get(ref, ref) for ref in child]
                                          if key == 'evidence_refs' else normalize_refs(child))
                                    for key, child in value.items()}
                        if isinstance(value, list):
                            return [normalize_refs(child) for child in value]
                        return value
                    payload = normalize_refs(frame.model_dump(mode='json', exclude={'frame_hash'}))
                    frame = type(frame).model_validate({**payload, 'frame_hash': 'sha256:' + content_hash(payload)})
                if not set(frame.evidence_refs) <= set(refs) or not set(frame.source_region_refs) <= {anchor.block_id for anchor in extension.canonical_evidence}:
                    raise ValueError("normative frame source regions/evidence do not match canonical quote support")
                links = _frame_participants(frame, participants, frame_kinds, predicate_nodes, refs)
                frame_payload = frame.model_dump(mode='json', exclude={'frame_hash'})
                compiled = type(frame).model_validate({**frame_payload, 'frame_hash': 'sha256:' + content_hash(frame_payload)})
                compiled_frames.append(compiled)
                label, kind = frame.display_name, f'normative_frame:{frame.frame_kind}'
                ignored = {'frame_id', 'display_name', 'display_name_zh', 'frame_kind', 'epistemic_status', 'evidence_refs', 'source_region_refs'}
                for path, value in _scalar_slots({key: value for key, value in frame_payload.items() if key not in ignored}):
                    slot_id = f'{node_id}:slot:{path}'
                    added_nodes.append(ExploratoryGraphNodeV1(node_id=slot_id, node_kind='normative_semantic_slot', label=value or '(empty)', evidence_refs=refs))
                    edge(node_id, slot_id, 'has_semantic_slot', refs)
                for slot, target in links:
                    edge(node_id, target, f'has_participant:{slot}', refs)
            added_nodes.append(ExploratoryGraphNodeV1(node_id=node_id, node_kind=kind, label=label, evidence_refs=refs))
            edge(node_id, decision.source_node_id, 'extends_source_interpretation', refs)
            derivations.append(SeedDerivationV1(seed_snapshot_ref=seed.snapshot_ref, seed_snapshot_hash=seed.snapshot_hash,
                seed_iri=decision.seed_iri, candidate_node_id=node_id, alignment_id=decision.alignment_id))
            applied.append(extension.extension_id)
        all_nodes = [node.node_id for node in (*baseline.nodes, *added_nodes)]
        if len(all_nodes) != len(set(all_nodes)):
            raise ValueError("source-grounded extension generated colliding node identities")
        if any(edge.source_node_id not in set(all_nodes) or edge.target_node_id not in set(all_nodes) for edge in added_edges):
            raise ValueError("source-grounded extension contains an unresolved participant endpoint")
        structural = _frame_shapes(compiled_frames, added_nodes, added_edges)
        delta = SourceGroundedSeedDeltaV1.create(organization_id=command.organization_id, workspace_id=command.workspace_id,
            alignment_ref=command.alignment_ref, alignment_hash=command.alignment_hash, inputs=command.inputs,
            contributions=tuple(contributions), added_nodes=tuple(added_nodes), added_edges=tuple(added_edges),
            normative_frames=tuple(sorted(compiled_frames, key=lambda frame: frame.frame_id)), seed_derivations=tuple(derivations),
            conflicts=tuple(conflicts), applied_extension_ids=tuple(applied), structural_measurements={
                'canonical_quotes_validated': quote_count, 'baseline_evidence_binding_valid': True,
                'successor_evidence_binding_valid': all(set(node.evidence_refs) <= known_evidence for node in added_nodes),
                'normative_frame_count': len(compiled_frames), 'frame_shape_result': structural,
                'semantic_entailment': 'NOT_CERTIFIED', 'paired_competency_evaluation': 'NOT_SUPPLIED',
            }, created_at=command.reviewed_at)
        successor, successor_integrity, diff = None, None, None
        lineage_service = OntologyLineageService()
        if added_nodes or added_edges:
            payload = baseline.model_dump(mode='json', exclude={'snapshot_ref', 'snapshot_hash'})
            payload.update(build_ref='ontology-build:' + content_hash({'baseline': command.inputs.baseline_manifest.model_dump(), 'delta_hash': delta.delta_hash}),
                nodes=[node.model_dump(mode='json') for node in (*baseline.nodes, *added_nodes)],
                edges=[item.model_dump(mode='json') for item in (*baseline.edges, *added_edges)],
                predecessor_snapshot_ref=baseline.snapshot_ref, predecessor_snapshot_hash=baseline.snapshot_hash,
                change_cause='schema_rebuild', triggering_ref=delta.delta_ref, created_at=command.reviewed_at.isoformat(),
                authority_state='provisional', completeness_state='incomplete')
            digest = content_hash(payload)
            successor = ExploratoryGraphSnapshotV1(snapshot_ref=f'exploratory-snapshot:{digest}', snapshot_hash=f'sha256:{digest}', **payload)
            successor_integrity = create_exploratory_snapshot_integrity(
                snapshot=successor,
                producer_profile='source_grounded_seed_successor.v1',
                producer_version='semantier_source_grounded_seed_successor_v1',
                recorded_at=command.reviewed_at,
            )
            diff = lineage_service.compare(baseline=baseline, successor=successor)
        lineage = lineage_service.project(snapshots=(baseline, successor) if successor else (baseline,))
        candidate = successor or baseline
        comparison = OntologySeedComparisonV1.create(organization_id=command.organization_id, workspace_id=command.workspace_id,
            seed_snapshot=command.inputs.seed_snapshot,
            candidate_snapshot=ExactAlignmentInputV1.of(candidate, 'snapshot_ref', 'snapshot_hash'),
            delta_ref=delta.delta_ref, delta_hash=delta.delta_hash, contributions=tuple(contributions), created_at=command.reviewed_at)
        return OntologySeedAlignmentResult(
            command,
            delta,
            successor,
            successor_integrity,
            diff,
            lineage,
            comparison,
        )
