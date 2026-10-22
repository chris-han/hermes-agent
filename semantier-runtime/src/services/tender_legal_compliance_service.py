
from storage.hosted_postgres import sql_query,begin_write,in_transaction
"""Exact retained Tender bindings and scoped compliance publication.

No source acquisition, review/admission creation or provider invocation occurs here.
"""
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import DecisionRecord
from contracts.tender_legal_compliance import TenderLegalComplianceBasisV1
from eos.artifact_hashing import content_hash


def normative_input_pin(inputs):
    """Review binds input content before the review itself exists (acyclic)."""
    digest = content_hash(inputs.model_dump(mode='json', exclude={'binding_review'}))
    return ArtifactPinV1(artifact_kind='tender_normative_binding_inputs',
                        artifact_ref='tender-normative-binding-inputs:'+digest, artifact_hash='sha256:'+digest)


class TenderLegalComplianceService:
    def __init__(self, *, build_store, authority_resolver, workspace_root, db_path):
        from storage.ontology_build_store import OntologyBuildStore
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        if not isinstance(build_store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError('compliance requires existing governed owners')
        self.store, self.authority = build_store, authority_resolver
        self.workspace_root, self.db_path = Path(workspace_root), Path(db_path)

    @contextmanager
    def _adoptions(self, actor_context):
        from storage.knowledge_builder_store import KnowledgeBuilderStore
        from storage.graph_intent_brief_store import GraphIntentBriefStore
        from services.tender_owner_adoption_service import TenderOwnerAdoptionService
        registry = KnowledgeBuilderStore(self.db_path, workspace_root=self.workspace_root)
        try:
            yield TenderOwnerAdoptionService(store=self.store, authority_resolver=self.authority,
                intent_store=GraphIntentBriefStore(self.workspace_root,
                    organization_id=actor_context.organization_id, workspace_id=actor_context.workspace_id,
                    source_registry=registry))
        finally:
            registry.close()

    def _advisory_inputs(self, *, actor_context, finding, projection, frozen_at):
        from contracts.tender_legal_compliance import TenderOwnerAdvisoryInputsV1
        from contracts.normative_schema import NormativeSchemaContractV1
        from contracts.normative_semantics import NormativeSemanticCandidateSetV1
        from contracts.ontology_validation import CompiledRulePlanV1
        from services.tender_observed_facts_projection_service import TenderObservedFactsProjectionService
        scope = dict(organization_id=actor_context.organization_id, workspace_id=actor_context.workspace_id)
        with self._adoptions(actor_context) as owner:
            entries = owner.select(actor_context=actor_context, canonical_pattern_ref=finding.canonical_pattern_ref,
                canonical_pattern_hash=finding.canonical_pattern_hash, ontology_pin=projection.ontology.pin())
        if not entries:
            return None, ('QUALIFIED_NORMATIVE_BINDING_MISSING',)
        if len(entries) != 1:
            return None, ('OWNER_ADOPTED_MAPPING_UNRESOLVED_MULTIPLE',)
        entry = entries[0]
        proposal, adoption = entry['proposal'], entry['adoption']
        def load(pin, model):
            return self.store.get(**scope, artifact_ref=pin.artifact_ref, contract_type=model)
        schema = load(proposal.schema_, NormativeSchemaContractV1)
        candidate = load(proposal.candidate, NormativeSemanticCandidateSetV1)
        plan = load(proposal.plan, CompiledRulePlanV1)
        # No case observation is manufactured from a screening MATCH. Reuse only
        # exact receipts over this located target and this normative definition.
        facts, receipt, gaps = self._project_advisory_facts(
            finding=finding, projection=projection, schema=schema, plan=plan, frozen_at=frozen_at)
        return TenderOwnerAdvisoryInputsV1(proposal=proposal, adoption=adoption,
            source_decision=load(adoption.source_decision, DecisionRecord),
            interpretation_decision=load(adoption.interpretation_decision, DecisionRecord),
            normative_schema=schema, candidate=candidate, plan=plan, facts=facts, projection=receipt), tuple(dict.fromkeys((*proposal.unknowns, *gaps)))

    def _project_advisory_facts(self, *, finding, projection, schema, plan, frozen_at):
        from contracts.tender_observed_facts_projection import (
            TenderFactSubjectBindingV1, TenderFactSensorContractV1,
            TenderFactObservationReceiptV1, TenderCategoricalFactObservationReceiptV1)
        from contracts.tender_document_date_observation import TenderDocumentDateObservationV1
        from services.tender_observed_facts_projection_service import TenderObservedFactsProjectionService
        scope = dict(organization_id=finding.organization_id, workspace_id=finding.workspace_id)
        selectors = {e['selectorHash'] for e in finding.evidence_refs}
        target = tuple(e.pin() for e in projection.evidence if e.evidence_role == 'target'
            and e.source_ref == finding.source_identity_ref and e.source_hash == finding.source_hash and e.selector_hash in selectors)
        projector = TenderObservedFactsProjectionService()
        subjects = []
        nodes = {node.node_id for node in projection.context.nodes}
        for assertion in projection.context.assertions:
            if assertion.origin != 'SOURCE' or assertion.subject_ref not in nodes or assertion.object_ref not in nodes:
                continue
            for evidence in projection.evidence:
                if evidence.pin() not in target or evidence.pin() not in assertion.evidence:
                    continue
                irs = [pin for pin in evidence.provenance if pin.artifact_kind == 'canonical_source_ir']
                if len(irs) != 1:
                    continue
                subject = TenderFactSubjectBindingV1(organization_id=finding.organization_id,
                    workspace_id=finding.workspace_id, context_graph=projection.context.pin(),
                    actor_node_ref=assertion.subject_ref, condition_node_ref=assertion.object_ref,
                    association_assertion_ref=assertion.assertion_id,
                    association_assertion_hash='sha256:'+content_hash(assertion.model_dump(mode='json')),
                    occurrence_evidence=evidence.pin(), source_ref=evidence.source_ref, source_hash=evidence.source_hash,
                    selector_hash=evidence.selector_hash, canonical_source_ir=irs[0])
                projector.resolve_subject(store=self.store, subject_binding=subject)
                subjects.append(subject)
        if len(subjects) != 1:
            return None, None, ('EXACT_CASE_SUBJECT_MISSING_OR_AMBIGUOUS',)
        subject = subjects[0]
        required = projector.required_fact_refs(plan)
        sensors = tuple(TenderFactSensorContractV1.from_schema(schema=schema, fact_ref=d.fact_ref)
            for d in schema.fact_definitions or () if d.fact_ref in required and d.value_type == 'boolean')
        observations, categorical, dates = [], [], []
        for model, output in ((TenderFactObservationReceiptV1,observations),
            (TenderCategoricalFactObservationReceiptV1,categorical), (TenderDocumentDateObservationV1,dates)):
            for pin in self.store.list_artifact_pins(**scope,schema_version=model.model_fields['schema_version'].default):
                value=self.store.get(**scope,artifact_ref=pin.artifact_ref,contract_type=model)
                if value.fact_ref not in required or not set(value.target_evidence).issubset(set(target)):
                    continue
                if getattr(value,'subject_binding',subject) != subject:
                    continue
                output.append(value)
        if any(len({v.fact_ref for v in values}) != len(values) for values in (observations,categorical,dates)):
            return None, None, ('CASE_FACT_OBSERVATIONS_AMBIGUOUS',)
        facts, receipt = projector.project(store=self.store, plan=plan, schema=schema, target_evidence=target,
            sensor_contracts=sensors, observations=tuple(observations), categorical_observations=tuple(categorical),
            document_date_observations=tuple(dates), subject_binding=subject, frozen_at=frozen_at)
        return facts, receipt, tuple('CASE_FACT_UNRESOLVED:'+ref for ref in receipt.unresolved_fact_refs)

    def verify_advisory_owners(self, *, actor_context, basis, projection, verify_facts=True):
        a = basis.advisory
        if a is None:
            return
        with self._adoptions(actor_context) as owner:
            proposal = owner.validate_live(actor_context=actor_context, adoption=a.adoption)
            from contracts.tender_owner_adoption import ontology_contract_hash
            if proposal != a.proposal or proposal.ontology_contract_hash != ontology_contract_hash(projection.ontology):
                raise ValueError('ADVISORY_LIVE_BINDING_MISMATCH')
            entries = owner.select(actor_context=actor_context,
                canonical_pattern_ref=basis.finding.canonical_pattern_ref, canonical_pattern_hash=basis.finding.canonical_pattern_hash, ontology_pin=projection.ontology.pin())
            if len(entries) != 1 or entries[0]['adoption'] != a.adoption:
                raise ValueError('ADVISORY_SELECTION_CHANGED')
        scope = dict(organization_id=basis.organization_id,workspace_id=basis.workspace_id)
        for value, ref in ((a.normative_schema,a.normative_schema.schema_ref),
            (a.candidate,a.candidate.candidate_set_ref),(a.plan,a.plan.plan_ref)):
            self._load_exact(scope,value,ref)
        if a.facts is not None:
            self._load_exact(scope,a.facts,a.facts.facts_ref)
            self._load_exact(scope,a.projection,a.projection.projection_ref)
            if a.projection.subject_binding.context_graph != projection.context.pin():
                raise ValueError('ADVISORY_CASE_CONTEXT_MISMATCH')
        if verify_facts:
            facts, receipt, _ = self._project_advisory_facts(finding=basis.finding, projection=projection,
                schema=a.normative_schema, plan=a.plan, frozen_at=a.projection.frozen_at if a.projection else basis.frozen_at)
            if facts != a.facts or receipt != a.projection:
                raise ValueError('ADVISORY_FACT_REPLAY_MISMATCH')

    def _load_exact(self, scope, value, ref):
        retained = self.store.get(**scope, artifact_ref=ref, contract_type=type(value))
        if retained is None or retained != value:
            raise ValueError('RETAINED_NORMATIVE_INPUT_MISMATCH')
        return retained

    @staticmethod
    def verify_candidate_provisions(*, candidate, rule, provisions):
        expected = (candidate.source_identity_ref, candidate.source_hash,
                    candidate.canonical_source_ir_ref, candidate.canonical_source_ir_hash)
        if (not rule.evidence_refs
            or not set(rule.evidence_refs).issubset({p.evidence.evidence_ref for p in provisions})
            or any((p.identity.source_identity_ref, p.identity.source_hash,
                    p.canonical_source_ir.artifact_ref, p.canonical_source_ir.artifact_hash) != expected
                   for p in provisions)):
            raise ValueError('NORMATIVE_RULE_PROVISION_MISMATCH')

    @staticmethod
    def verify_provision_metadata(*, provision, retained):
        """Labels and section text come from the pinned legal owner, not callers."""
        source, version = retained["source"], retained["version"]
        sections = [row for row in retained["sections"] if row["section_id"] == provision.section_id]
        if (len(sections) != 1 or source["source_id"] != provision.source_id
            or version["version_id"] != provision.version_id or version["source_id"] != provision.source_id
            or source["canonical_title"] != provision.source_title
            or version["version_identity"] != provision.version_label):
            raise ValueError("NORMATIVE_PROVISION_METADATA_MISMATCH")
        section = sections[0]
        if section["version_id"] != provision.version_id or section["stable_locator"] != provision.article_locator:
            raise ValueError("NORMATIVE_PROVISION_METADATA_MISMATCH")
        text = json.loads(section["anchor_json"]).get("retained_text")
        if not isinstance(text, str) or not text or content_hash({"text": text}) != section["text_hash"]:
            raise ValueError("NORMATIVE_RETAINED_SECTION_TEXT_REQUIRED")
        if not provision.evidence.selector.exact_quote or provision.evidence.selector.exact_quote not in text:
            raise ValueError("NORMATIVE_SECTION_QUOTE_MISMATCH")

    def verify_normative_owners(self, *, basis, projection):
        """Live publication preflight; offline replay uses the frozen result only."""
        from contracts.tender_observed_facts_projection import (
            TenderFactSensorContractV1, TenderCategoricalFactSensorContractV1,
            TenderFactObservationReceiptV1, TenderCategoricalFactObservationReceiptV1,
        )
        from contracts.tender_document_date_observation import TenderDocumentDateObservationV1
        from services.tender_observed_facts_projection_service import TenderObservedFactsProjectionService
        from services.ontology_certification_successor_service import OntologyCertificationSuccessorService
        from eos.legal_corpus_store import LegalCorpusStore
        from storage.knowledge_builder_store import KnowledgeBuilderStore
        from storage.graph_intent_brief_store import GraphIntentBriefStore
        from semantier.knowledge_feature_packs.evidence_resolver import resolve_evidence_ref

        n = basis.normative
        if n is None:
            return None
        scope = dict(organization_id=basis.organization_id, workspace_id=basis.workspace_id)
        for value, ref in ((n.normative_schema,n.normative_schema.schema_ref), (n.candidate,n.candidate.candidate_set_ref),
                           (n.plan,n.plan.plan_ref), (n.authority,n.authority.set_ref),
                           (n.facts,n.facts.facts_ref), (n.projection,n.projection.projection_ref)):
            self._load_exact(scope, value, ref)
        from services.normative_semantic_production_service import NormativeSemanticProductionService
        produced = NormativeSemanticProductionService.read_persisted(
            store=self.store, **scope, candidate_pin=ArtifactPinV1(
                artifact_kind='normative_semantic_candidate_set',
                artifact_ref=n.candidate.candidate_set_ref, artifact_hash=n.candidate.candidate_set_hash))
        if produced != n.candidate:
            raise ValueError('NORMATIVE_PRODUCTION_REPLAY_MISMATCH')
        review = self.store.get(**scope, artifact_ref=n.binding_review.artifact_ref, contract_type=DecisionRecord)
        finding_pin = ArtifactPinV1(artifact_kind='governed_frame_finding', artifact_ref=basis.finding.finding_id,
                                   artifact_hash=basis.finding.finding_hash)
        if (review is None or review.pin() != n.binding_review or review.purpose != 'REVIEW'
            or review.result_code != 'NORMATIVE_BINDING_ACCEPTED' or review.subject != finding_pin
            or review.context != projection.context.pin() or review.ontology != projection.ontology.pin()
            or normative_input_pin(n) not in tuple(pin for invocation in review.reasoning.invocations for pin in invocation.inputs)):
            raise ValueError('EXACT_NORMATIVE_BINDING_REVIEW_REQUIRED')
        if n.projection.subject_binding.context_graph != projection.context.pin():
            raise ValueError('CASE_CONTEXT_BINDING_MISMATCH')
        if not set(n.projection.target_evidence).issubset({e.pin() for e in projection.evidence}):
            raise ValueError('CASE_EVIDENCE_BINDING_MISMATCH')
        projector = TenderObservedFactsProjectionService()
        projector.resolve_subject(store=self.store, subject_binding=n.projection.subject_binding)
        models = {
            'tender_fact_sensor_contract': TenderFactSensorContractV1,
            'tender_categorical_fact_sensor_contract': TenderCategoricalFactSensorContractV1,
            'tender_fact_observation_receipt': TenderFactObservationReceiptV1,
            'tender_categorical_fact_observation_receipt': TenderCategoricalFactObservationReceiptV1,
            'tender_document_date_observation': TenderDocumentDateObservationV1,
        }
        loaded = []
        for pin in (*n.projection.sensor_contracts, *n.projection.observation_receipts):
            model = models.get(pin.artifact_kind)
            if model is None:
                raise ValueError('UNSUPPORTED_FACT_PRODUCER')
            value = self.store.get(**scope, artifact_ref=pin.artifact_ref, contract_type=model)
            if value is None or value.pin() != pin:
                raise ValueError('FACT_PRODUCER_RECEIPT_MISSING')
            loaded.append(value)
        facts, receipt = projector.project(store=self.store, plan=n.plan, schema=n.normative_schema,
            target_evidence=n.projection.target_evidence, subject_binding=n.projection.subject_binding,
            sensor_contracts=tuple(v for v in loaded if isinstance(v,TenderFactSensorContractV1)),
            observations=tuple(v for v in loaded if isinstance(v,TenderFactObservationReceiptV1)),
            categorical_observations=tuple(v for v in loaded if isinstance(v,TenderCategoricalFactObservationReceiptV1)),
            document_date_observations=tuple(v for v in loaded if isinstance(v,TenderDocumentDateObservationV1)),
            frozen_at=n.projection.frozen_at)
        if facts != n.facts or receipt != n.projection:
            raise ValueError('FACT_PROJECTION_REPLAY_MISMATCH')
        legal = LegalCorpusStore(self.db_path)
        OntologyCertificationSuccessorService().verify_authority_dependencies(authority=n.authority, legal_store=legal)
        registry = KnowledgeBuilderStore(self.db_path, workspace_root=self.workspace_root)
        try:
            source_owner = GraphIntentBriefStore(self.workspace_root, **scope, source_registry=registry)
            rule = next(frame for frame in n.candidate.frames if frame.frame_kind == 'rule' and frame.frame_id == n.rule_id)
            self.verify_candidate_provisions(candidate=n.candidate, rule=rule, provisions=n.provisions)
            for provision in n.provisions:
                dependency = next(d for d in n.authority.dependencies
                                  if (d.source_id, d.version_id) == (provision.source_id, provision.version_id))
                retained = legal.resolve_certification_source_evidence(**scope,
                    source_id=dependency.source_id, version_id=dependency.version_id,
                    artifact_id=dependency.artifact_id, artifact_hash=dependency.artifact_hash,
                    section_ids=(provision.section_id,))
                self.verify_provision_metadata(provision=provision, retained=retained)
                identity, document = source_owner.resolve_evidence_source(
                    source_identity_ref=provision.identity.source_identity_ref,
                    canonical_source_ir_ref=provision.canonical_source_ir.artifact_ref,
                    canonical_source_ir_hash=provision.canonical_source_ir.artifact_hash)
                if identity != provision.identity:
                    raise ValueError('NORMATIVE_SOURCE_VERSION_MISMATCH')
                relative = Path(identity.artifact_ref)
                root = self.workspace_root.resolve()
                path = (root/relative).resolve()
                if relative.is_absolute() or '..' in relative.parts or root not in path.parents:
                    raise ValueError('NORMATIVE_SOURCE_OUTSIDE_WORKSPACE')
                if 'sha256:'+sha256(path.read_bytes()).hexdigest() != identity.source_hash:
                    raise ValueError('NORMATIVE_ORIGINAL_HASH_MISMATCH')
                location = resolve_evidence_ref(provision.evidence, document,
                    artifact_ref=provision.canonical_source_ir.artifact_ref, selector_hash=provision.evidence.selector_hash)
                if location.status not in ('exact','relocated_exact'):
                    raise ValueError('NORMATIVE_OCCURRENCE_MISMATCH')
        finally:
            registry.close()
        return review

    def publish_episode(self, *, actor_context, episode, inspection_service, basis_by_finding=None):
        """Append decisions after the immutable episode; repeated calls reuse exact pins."""
        from storage.runtime_semantic_hypothesis_artifact_store import RuntimeSemanticHypothesisArtifactStore
        from services.tender_review_service import TenderReviewService
        from services.tender_decision_record_service import TenderDecisionRecordService
        self.authority.require_current_actor(actor_context=actor_context)
        result = episode.inspection_result
        if (result.organization_id,result.workspace_id) != (actor_context.organization_id,actor_context.workspace_id):
            raise ValueError('FORBIDDEN_SCOPE')
        artifacts = RuntimeSemanticHypothesisArtifactStore(inspection_service._runtime_root,
            organization_id=result.organization_id, workspace_id=result.workspace_id)
        decisions = []
        supplied = basis_by_finding or {}
        if not set(supplied).issubset({f.finding_id for f in result.governed_findings}):
            raise ValueError('FINDING_NOT_IN_EPISODE')
        for finding in result.governed_findings:
            hypothesis = artifacts.get_hypothesis(hypothesis_ref=finding.observed_hypothesis_ref,
                                                  hypothesis_hash=finding.observed_hypothesis_hash)
            projection = artifacts.get_bounded_projection(projection_ref=hypothesis.decision_projection_ref,
                                                          projection_hash=hypothesis.decision_projection_hash)
            basis = supplied.get(finding.finding_id)
            if basis is None:
                # Resolve adopted mapping on the ordinary publication path. A
                # retained exact assessment is reused on recovery, not recomputed.
                advisory, gaps = self._advisory_inputs(actor_context=actor_context, finding=finding,
                    projection=projection, frozen_at=datetime.now(timezone.utc))
                previous = [entry for entry in self.read_episode(store=self.store,episode=episode)
                    if entry['basis'].finding == finding]
                def comparable(a):
                    if a is None:
                        return None
                    raw=a.model_dump(mode='json',exclude={'facts','projection'})
                    raw['facts']=a.facts.facts if a.facts else None
                    raw['projection']=a.projection.model_dump(mode='json',exclude={
                        'projection_ref','projection_hash','observed_facts','frozen_at'}) if a.projection else None
                    return raw
                from contracts.tender_legal_compliance import freeze_assessment_selection
                finding_pin=ArtifactPinV1(artifact_kind='governed_frame_finding',artifact_ref=finding.finding_id,artifact_hash=finding.finding_hash)
                selection=freeze_assessment_selection(inventory=(finding_pin,),entries=previous)[0]
                active = [entry for entry in previous if entry['decision'].pin() not in selection.superseded]
                exact = [entry for entry in active if comparable(entry['basis'].advisory)==comparable(advisory)
                    and entry['basis'].gaps==gaps]
                if len(exact)==1:
                    self.verify_advisory_owners(actor_context=actor_context,basis=exact[0]['basis'],projection=projection)
                    decisions.append(exact[0]['decision'])
                    continue
                predecessor = selection.selected
                # Legacy unbound episode timestamp preserves historical identity.
                basis = TenderLegalComplianceBasisV1.create(
                    organization_id=result.organization_id, workspace_id=result.workspace_id,
                    episode=TenderReviewService.episode_pin(episode),
                    inspection_result=ArtifactPinV1(artifact_kind='runtime_semantic_inspection_result.v1',
                        artifact_ref=result.result_ref, artifact_hash=result.result_hash),
                    finding=finding, target_ir=ArtifactPinV1(artifact_kind='canonical_source_ir',
                        artifact_ref=result.canonical_source_ir_ref, artifact_hash=result.canonical_source_ir_hash),
                    normative=None, advisory=advisory, gaps=gaps, supersedes_assessment=predecessor,
                    frozen_at=advisory.projection.frozen_at if advisory is not None and advisory.projection is not None else datetime.now(timezone.utc) if advisory is not None or predecessor is not None else projection.context.recorded_at)
            decision = TenderDecisionRecordService(build_store=self.store, authority_resolver=self.authority).record_evaluation(
                actor_context=actor_context, episode=episode, basis=basis, projection=projection, hypothesis=hypothesis,
                compliance_owner=self)
            decisions.append(decision)
        return tuple(decisions)

    @staticmethod
    def read_episode(*, store, episode):
        """Read retained decisions only; never reinterpret or resolve current law."""
        from contracts.tender_legal_compliance import TenderLegalComplianceVerificationV1, TenderFrozenComplianceDecisionV1
        from services.tender_review_service import TenderReviewService
        episode_pin = TenderReviewService.episode_pin(episode)
        result = episode.inspection_result
        rows = store._connection.execute(
            sql_query("SELECT payload_json FROM ontology_build_artifacts WHERE organization_id=? AND workspace_id=? "
            "AND schema_version='semantier.decision_record.v1' "
            "AND json_extract(payload_json,'$.reasoning.strategy')='tender_scoped_compliance_evaluation' "
            "AND EXISTS (SELECT 1 FROM json_each(payload_json,'$.reasoning.invocations[0].inputs') AS input "
            "WHERE json_extract(input.value,'$.artifact_kind')='tender_journey_inspection_episode' "
            "AND json_extract(input.value,'$.artifact_ref')=?) "
            "ORDER BY artifact_ref","SELECT payload_json FROM ontology_build_artifacts WHERE organization_id=%s AND workspace_id=%s AND schema_version='semantier.decision_record.v1' AND (payload_json::jsonb #>> '{reasoning,strategy}')='tender_scoped_compliance_evaluation' AND EXISTS (SELECT 1 FROM json_each(payload_json,'$.reasoning.invocations[0].inputs') AS input WHERE (input.value::jsonb #>> '{artifact_kind}')='tender_journey_inspection_episode' AND (input.value::jsonb #>> '{artifact_ref}')=%s) ORDER BY artifact_ref"),
            (result.organization_id,result.workspace_id,episode.transition_id),
        ).fetchall()
        retained = []
        for row in rows:
            decision = DecisionRecord.model_validate_json(row[0])
            invocation = decision.reasoning.invocations[0]
            basis_pins = tuple(pin for pin in invocation.inputs if pin.artifact_kind == 'tender_legal_compliance_basis')
            if len(basis_pins) != 1 or episode_pin not in invocation.inputs:
                raise ValueError('RETAINED_COMPLIANCE_DECISION_MISMATCH')
            basis_pin = basis_pins[0]
            basis = store.get(organization_id=result.organization_id, workspace_id=result.workspace_id,
                artifact_ref=basis_pin.artifact_ref, contract_type=TenderLegalComplianceBasisV1)
            verification = TenderLegalComplianceVerificationV1.model_validate_json(invocation.proof_steps[0])
            if (basis is None or basis.pin() != basis_pin or basis.episode != episode_pin
                or basis.finding not in result.governed_findings or verification.basis != basis_pin
                or verification.outcome != decision.result_code or verification.status != 'VERIFIED'
                or decision.subject.artifact_ref != basis.finding.finding_id
                or decision.subject.artifact_hash != basis.finding.finding_hash):
                raise ValueError('RETAINED_COMPLIANCE_DECISION_MISMATCH')
            entry = TenderFrozenComplianceDecisionV1(decision=decision, basis=basis, verification=verification)
            retained.append(dict(decision=entry.decision, basis=entry.basis, verification=entry.verification))
        return tuple(retained)
