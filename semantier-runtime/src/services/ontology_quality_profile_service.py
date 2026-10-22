"""Repository-owned bounded Tender/GDFM MVL0 V&V profile definition."""
from contracts.ontology_quality import QualityProfileV1, QualityRequirementV1


class QualityProfileService:
    @classmethod
    def verify_consumer_basis(cls, *, store, profile, run):
        """Certification/admission may select only canonical or authorized exact profiles."""
        canonical = tender_mvl0_quality_profile(organization_id=profile.organization_id, workspace_id=profile.workspace_id)
        if profile == canonical:
            return
        from storage.ontology_build_store import OntologyBuildStore
        from contracts.ontology_quality import QualityEvaluationRunV1
        if not isinstance(store, OntologyBuildStore):
            raise ValueError('QUALITY_PROFILE_AUTHORIZED_SUCCESSOR_STORE_REQUIRED')
        scope = dict(organization_id=profile.organization_id, workspace_id=profile.workspace_id)
        persisted, review = cls.read_persisted(store=store, **scope, profile_pin=profile.pin())
        if persisted != profile or review is None:
            raise ValueError('QUALITY_PROFILE_AUTHORIZED_SUCCESSOR_REQUIRED')
        retained = store.get(**scope, artifact_ref=run.run_ref, contract_type=QualityEvaluationRunV1)
        if retained != run or run.profile_definition != profile:
            raise ValueError('QUALITY_PROFILE_SUCCESSOR_EXACT_RUN_REQUIRED')
        run.require_basis(quality_profile=profile.pin(), subject=review.candidate,
            competency=review.competency, validation=review.validation)
        if review.reviewed_at > run.executed_at:
            raise ValueError('QUALITY_APPLICABILITY_REVIEW_POSTDATES_EVALUATION')

    @classmethod
    def verify_evaluation_basis(cls, *, profile, review, candidate, validation):
        if profile.applicability_review is None:
            if review is not None:
                raise ValueError('QUALITY_APPLICABILITY_UNEXPECTED_REVIEW')
            return
        if review is None or profile.applicability_review != review.pin():
            raise ValueError('QUALITY_APPLICABILITY_EXACT_REVIEW_REQUIRED')
        predecessor = tender_mvl0_quality_profile(organization_id=profile.organization_id, workspace_id=profile.workspace_id)
        if cls.successor(predecessor=predecessor, review=review) != profile:
            raise ValueError('QUALITY_APPLICABILITY_PROFILE_MISMATCH')
        if ((review.candidate.artifact_ref, review.candidate.artifact_hash) != (candidate.candidate_set_ref, candidate.candidate_set_hash)
            or (review.validation.artifact_ref, review.validation.artifact_hash) != (validation.validation_contract_ref, validation.validation_contract_hash)
            or (review.competency.artifact_ref, review.competency.artifact_hash) != (candidate.competency_contract_ref, candidate.competency_contract_hash)
            or (review.source_identity_ref, review.source_hash) != (candidate.source_identity_ref, candidate.source_hash)):
            raise ValueError('QUALITY_APPLICABILITY_EVALUATION_BASIS_MISMATCH')
        audit = review.conflict_audit
        if audit is None or audit.get('audit_hash') != review.conflict_audit_hash:
            raise ValueError('QUALITY_APPLICABILITY_RETAINED_AUDIT_REQUIRED')
        # Source bytes were independently verified by the persisted review owner.
        # Direct evaluation checks retained inventory identity; it grants no admission.
        spans = {s['evidence_ref']: s['text'] for r in audit.get('inventory', []) for s in r.get('source_spans', [])}
        cls.verify_conflict_audit(candidate=candidate, validation=validation, audit=audit,
            source=dict(source_citation_keys={ref: ref for ref in spans},
                source_reading_index=[dict(citation_key=ref, text=text) for ref, text in spans.items()]))
        cls.verify_precedence_basis(candidate=candidate, validation=validation, expectations=review.precedence_expectations)

    @staticmethod
    def _operation(review):
        from eos.artifact_hashing import content_hash
        if not review.request_id or not review.reviewer_kind:
            raise ValueError('QUALITY_APPLICABILITY_AUTHENTICATED_REVIEW_REQUIRED')
        return dict(organization_id=review.organization_id, workspace_id=review.workspace_id,
            actor_ref=review.actor_ref, operation='REVIEW_QUALITY_REQUIREMENT_APPLICABILITY',
            request_id=review.request_id, request_hash='sha256:' + content_hash(review.model_dump(
                mode='json', exclude={'review_ref', 'review_hash', 'reviewed_at'})))

    @classmethod
    def _verify_review_basis(cls, *, store, review):
        import json
        from contracts.artifact_pin import ArtifactPinV1
        from contracts.normative_semantics import NormativeSemanticProposalEvidenceV1
        from contracts.validation_contract import ValidationContractV1
        from services.normative_semantic_production_service import NormativeSemanticProductionService
        from services.validation_contract_review_service import ValidationContractReviewService
        from services.ontology_quality_evaluation_service import QualityEvaluationService
        scope = dict(organization_id=review.organization_id, workspace_id=review.workspace_id)
        candidate = NormativeSemanticProductionService.read_persisted(store=store, **scope, candidate_pin=review.candidate)
        validation = store.get(**scope, artifact_ref=review.validation.artifact_ref, contract_type=ValidationContractV1)
        if validation is None or validation.validation_contract_hash != review.validation.artifact_hash:
            raise ValueError('QUALITY_APPLICABILITY_VALIDATION_PIN_MISMATCH')
        decision, accepted = ValidationContractReviewService.read_source_decision_persisted(
            store=store, **scope, decision_pin=ArtifactPinV1(artifact_kind='source_validation_review_decision',
                artifact_ref=validation.human_oracle_ref, artifact_hash=validation.human_oracle_hash))
        if decision.decision != 'accepted' or accepted != validation:
            raise ValueError('QUALITY_APPLICABILITY_ACCEPTED_VALIDATION_REQUIRED')
        QualityEvaluationService._same_source(candidate, validation)
        if (review.source_identity_ref, review.source_hash, review.competency.artifact_ref,
            review.competency.artifact_hash) != (candidate.source_identity_ref, candidate.source_hash,
            candidate.competency_contract_ref, candidate.competency_contract_hash):
            raise ValueError('QUALITY_APPLICABILITY_SOURCE_SCOPE_MISMATCH')
        receipt = store.get(**scope, artifact_ref=candidate.proposal_evidence_ref,
            contract_type=NormativeSemanticProposalEvidenceV1)
        if receipt is None or receipt.evidence_hash != candidate.proposal_evidence_hash:
            raise ValueError('QUALITY_APPLICABILITY_SOURCE_RECEIPT_MISMATCH')
        source = json.loads(receipt.input_text)['source_context']
        if review.conflict_audit is None or review.conflict_audit.get('audit_hash') != review.conflict_audit_hash:
            raise ValueError('QUALITY_APPLICABILITY_RETAINED_AUDIT_REQUIRED')
        cls.verify_conflict_audit(candidate=candidate, validation=validation, source=source, audit=review.conflict_audit)
        cls.verify_precedence_basis(candidate=candidate, validation=validation, expectations=review.precedence_expectations)
        permitted = set(source['source_citation_keys'].values()) | {review.conflict_audit_hash}
        if any(not set(entry.evidence_refs).issubset(permitted) for entry in review.entries):
            raise ValueError('QUALITY_APPLICABILITY_ENTRY_EVIDENCE_MISMATCH')
        excluded = [e for e in review.entries if e.state == 'NOT_APPLICABLE']
        if any(review.conflict_audit_hash not in e.evidence_refs for e in excluded):
            raise ValueError('QUALITY_APPLICABILITY_EXCLUSION_AUDIT_REQUIRED')
        return candidate, validation

    @classmethod
    def read_persisted(cls, *, store, organization_id, workspace_id, profile_pin):
        from contracts.ontology_quality import QualityProfileApplicabilityReviewV1
        scope = dict(organization_id=organization_id, workspace_id=workspace_id)
        if profile_pin.artifact_kind != 'quality_profile':
            raise ValueError('QUALITY_PROFILE_PIN_KIND_MISMATCH')
        profile = store.get(**scope, artifact_ref=profile_pin.artifact_ref, contract_type=QualityProfileV1)
        if profile is None or profile.pin() != profile_pin:
            raise ValueError('QUALITY_PROFILE_PIN_MISMATCH')
        canonical = tender_mvl0_quality_profile(**scope)
        if profile == canonical:
            return profile, None
        if profile.applicability_review is None or profile.predecessor_profile != canonical.pin():
            raise ValueError('QUALITY_PROFILE_AUTHORIZED_SUCCESSOR_REQUIRED')
        review = store.get(**scope, artifact_ref=profile.applicability_review.artifact_ref,
            contract_type=QualityProfileApplicabilityReviewV1)
        if review is None or review.pin() != profile.applicability_review:
            raise ValueError('QUALITY_APPLICABILITY_REVIEW_PIN_MISMATCH')
        predecessor = store.get(**scope, artifact_ref=canonical.profile_ref, contract_type=QualityProfileV1)
        if predecessor != canonical or cls.successor(predecessor=predecessor, review=review) != profile:
            raise ValueError('QUALITY_APPLICABILITY_SUCCESSOR_REPLAY_MISMATCH')
        if store.get_operation_result(**cls._operation(review), artifact_kind='quality_profile') != profile_pin:
            raise ValueError('QUALITY_APPLICABILITY_AUTHORIZED_OPERATION_REQUIRED')
        cls._verify_review_basis(store=store, review=review)
        return profile, review

    @classmethod
    def review_persisted(cls, *, store, authority_resolver, ctx, candidate_pin, validation_pin,
        entries, conflict_audit, precedence_expectations, reviewer_kind, request_id, reviewed_at):
        from contracts.artifact_pin import ArtifactPinV1
        from contracts.knowledge_classification import RuntimeContextDTO
        from contracts.ontology_quality import QualityProfileApplicabilityReviewV1, QualityRequirementApplicabilityEntryV1
        from services.normative_semantic_production_service import NormativeSemanticProductionService
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path
        if not isinstance(store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError('quality applicability review requires production owners')
        if not ctx.authenticated or store._path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve():
            raise ValueError('QUALITY_APPLICABILITY_WORKSPACE_OWNER_MISMATCH')
        actor = RuntimeContextDTO(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, actor_id=ctx.user_id)
        authority_resolver.require_current_actor(actor_context=actor, authorization_class='TENANT_ADMIN')
        scope = dict(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        candidate = NormativeSemanticProductionService.read_persisted(store=store, **scope, candidate_pin=candidate_pin)
        predecessor = tender_mvl0_quality_profile(**scope)
        review = QualityProfileApplicabilityReviewV1.create(**scope, actor_ref='user:' + ctx.user_id,
            request_id=request_id, reviewer_kind=reviewer_kind, predecessor_profile=predecessor.pin(),
            source_identity_ref=candidate.source_identity_ref, source_hash=candidate.source_hash,
            competency=ArtifactPinV1(artifact_kind='competency_contract', artifact_ref=candidate.competency_contract_ref,
                artifact_hash=candidate.competency_contract_hash), candidate=candidate_pin, validation=validation_pin,
            entries=tuple(QualityRequirementApplicabilityEntryV1.model_validate(e) for e in entries),
            conflict_audit_hash=conflict_audit['audit_hash'], conflict_audit=conflict_audit,
            precedence_expectations=precedence_expectations, reviewed_at=reviewed_at)
        operation = cls._operation(review)
        existing = store.get_operation_result(**operation, artifact_kind='quality_profile')
        if existing is not None:
            return cls.read_persisted(store=store, **scope, profile_pin=existing)
        successor = cls.successor(predecessor=predecessor, review=review)
        cls._verify_review_basis(store=store, review=review)
        winner = store.put_many_with_operation_result(**operation, artifacts=(predecessor, review, successor),
            result_pin=successor.pin(), verify_before_commit=lambda: authority_resolver.require_current_actor(
                actor_context=actor, authorization_class='TENANT_ADMIN'))
        return cls.read_persisted(store=store, **scope, profile_pin=winner)

    @staticmethod
    def verify_precedence_basis(*, candidate, validation, expectations):
        """Require independently reviewed contrasts for every represented source branch."""
        from contracts.ontology_quality import QualityPrecedenceExpectationV1
        expectations = tuple(QualityPrecedenceExpectationV1.model_validate(e) for e in expectations)
        frames = {f.frame_id: f for f in candidate.frames if f.frame_kind in {'exception', 'precedence'}}
        tests = {t.test_id: t for t in validation.executable_tests}
        keys = [(e.test_id, e.rule_ref, e.relation_ref) for e in expectations]
        if len(keys) != len(set(keys)):
            raise ValueError('QUALITY_PRECEDENCE_DUPLICATE_EXPECTATION')
        if not frames or {e.relation_ref for e in expectations} != set(frames):
            raise ValueError('QUALITY_PRECEDENCE_SOURCE_BRANCH_COVERAGE_MISSING')
        coverage = {ref: set() for ref in frames}
        for expectation in expectations:
            frame = frames[expectation.relation_ref]
            test = tests.get(expectation.test_id)
            target = frame.applies_to_rule_ref if frame.frame_kind == 'exception' else frame.lower_priority_ref
            controller = frame.frame_id if frame.frame_kind == 'exception' else frame.higher_priority_ref
            if (expectation.relation_kind, expectation.rule_ref, expectation.controlling_ref) != (
                frame.frame_kind, target, controller
            ):
                raise ValueError('QUALITY_PRECEDENCE_RELATION_MISMATCH')
            if test is None or not set(expectation.evidence_refs).issubset(frame.evidence_refs):
                raise ValueError('QUALITY_PRECEDENCE_CASE_OR_SOURCE_MISMATCH')
            # The supplement cannot rewrite the accepted answer for its CQ target.
            if test.result_rule_ref == target and expectation.expected_rule_outcome != test.expected_result:
                raise ValueError('QUALITY_PRECEDENCE_ACCEPTED_ANSWER_MISMATCH')
            coverage[frame.frame_id].add(expectation.expected_branch_outcome)
        if any(states != {'MATCH', 'NO_MATCH', 'UNKNOWN'} for states in coverage.values()):
            raise ValueError('QUALITY_PRECEDENCE_BRANCH_CONTRASTS_MISSING')

    @staticmethod
    def verify_conflict_audit(*, candidate, validation, source, audit):
        """Bind reviewed exclusions to source bytes; do not infer legal compatibility."""
        from itertools import combinations
        from eos.artifact_hashing import content_hash
        expected = dict(
            candidate_pin=dict(artifact_ref=candidate.candidate_set_ref, artifact_hash=candidate.candidate_set_hash),
            validation_pin=dict(artifact_ref=validation.validation_contract_ref, artifact_hash=validation.validation_contract_hash),
            source_identity_ref=candidate.source_identity_ref, source_hash=candidate.source_hash,
            source_proposal_hash=candidate.proposal_evidence_hash)
        if any(audit.get(k) != v for k, v in expected.items()):
            raise ValueError('QUALITY_CONFLICT_AUDIT_BASIS_MISMATCH')
        digest = 'sha256:' + content_hash({k: v for k, v in audit.items() if k != 'audit_hash'})
        if audit.get('audit_hash') != digest:
            raise ValueError('QUALITY_CONFLICT_AUDIT_HASH_MISMATCH')
        if audit.get('status') != 'NO_GENUINE_CONFLICT_IN_FROZEN_SCOPE':
            raise ValueError('QUALITY_CONFLICT_EXCLUSION_NOT_ESTABLISHED')
        rules = {f.frame_id: f for f in candidate.frames if f.frame_kind == 'rule'}
        inventory = audit.get('inventory', [])
        if [r.get('rule_ref') for r in inventory] != sorted(rules):
            raise ValueError('QUALITY_CONFLICT_INVENTORY_INCOMPLETE')
        texts = {source['source_citation_keys'][r['citation_key']]: r['text']
                 for r in source['source_reading_index']}
        for row in inventory:
            rule = rules[row['rule_ref']]
            if (row.get('frame_hash'), row.get('modality'), row.get('consequence')) != (
                rule.frame_hash, rule.modality, rule.consequence
            ) or not (row.get('reviewed_consequence_domain') or '').strip() or not rule.consequence:
                raise ValueError('QUALITY_CONFLICT_CONSEQUENCE_MISMATCH')
            spans = row.get('source_spans', [])
            if [s.get('evidence_ref') for s in spans] != list(rule.evidence_refs) or any(
                s.get('evidence_ref') not in texts or s.get('text') != texts[s['evidence_ref']] for s in spans
            ):
                raise ValueError('QUALITY_CONFLICT_SOURCE_SPAN_MISMATCH')
        pairs = audit.get('pairs', [])
        universe = list(combinations(sorted(rules), 2))
        if [tuple(p.get('pair', ())) for p in pairs] != universe:
            raise ValueError('QUALITY_CONFLICT_PAIR_UNIVERSE_INCOMPLETE')
        allowed = {'SOURCE_REVIEWED_NO_INCOMPATIBLE_CONSEQUENCE',
                   'SOURCE_REVIEWED_COMPATIBLE_CONSEQUENCES',
                   'SOURCE_REVIEWED_NON_COAPPLICABLE_ARTICLE35_LIFECYCLE'}
        for pair in pairs:
            left, right = pair['pair']
            if pair.get('classification') not in allowed or not (pair.get('rationale') or '').strip():
                raise ValueError('QUALITY_CONFLICT_PAIR_UNRESOLVED')
            if pair.get('source_evidence_refs') != sorted(set(rules[left].evidence_refs + rules[right].evidence_refs)):
                raise ValueError('QUALITY_CONFLICT_PAIR_SOURCE_MISMATCH')
        if audit.get('pair_counts') != dict(source_reviewed_resolved=len(universe),
            unresolved_applicability=0, genuine_conflict=0, total=len(universe)):
            raise ValueError('QUALITY_CONFLICT_PAIR_COUNTS_MISMATCH')

    @staticmethod
    def successor(*, predecessor, review):
        """Derive policy-preserving bytes; persistence separately verifies review authority."""
        from contracts.ontology_quality import QualityProfileApplicabilityReviewV1
        predecessor = QualityProfileV1.model_validate(predecessor)
        review = QualityProfileApplicabilityReviewV1.model_validate(review)
        scope = dict(organization_id=predecessor.organization_id, workspace_id=predecessor.workspace_id)
        canonical = tender_mvl0_quality_profile(**scope)
        if predecessor != canonical:
            raise ValueError('QUALITY_APPLICABILITY_CANONICAL_PREDECESSOR_REQUIRED')
        if (review.organization_id, review.workspace_id) != (predecessor.organization_id, predecessor.workspace_id):
            raise ValueError('QUALITY_APPLICABILITY_SCOPE_MISMATCH')
        if review.predecessor_profile != predecessor.pin():
            raise ValueError('QUALITY_APPLICABILITY_PREDECESSOR_MISMATCH')
        if tuple(e.requirement_id for e in review.entries) != tuple(r.requirement_id for r in predecessor.requirements):
            raise ValueError('QUALITY_APPLICABILITY_COMPLETE_MATRIX_REQUIRED')
        requirements = []
        for requirement, entry in zip(predecessor.requirements, review.entries, strict=True):
            if entry.state == 'UNRESOLVED_APPLICABILITY':
                raise ValueError('QUALITY_APPLICABILITY_UNRESOLVED')
            if entry.state != 'REQUIRED' and entry.requirement_id != 'conflict_detection':
                raise ValueError('QUALITY_APPLICABILITY_REQUIREMENT_MUST_REMAIN_REQUIRED')
            values = requirement.model_dump()
            if entry.state == 'NOT_APPLICABLE':
                values.update(applicability=entry.state, applicability_rationale=entry.rationale,
                              applicability_evidence_refs=entry.evidence_refs)
            requirements.append(QualityRequirementV1(**values))
        return QualityProfileV1.create(**scope, name=predecessor.name,
            allow_warnings=predecessor.allow_warnings, requirements=tuple(requirements),
            predecessor_profile=predecessor.pin(), applicability_review=review.pin())


def tender_mvl0_quality_profile(*, organization_id: str, workspace_id: str) -> QualityProfileV1:
    requirements = {
        'LOGIC': ('semantic_integrity',),
        'CONSTRAINT': ('shape_conformance',),
        'MODELING': ('precedence_not_applicability', 'evidence_not_authority',
                     'candidate_not_accepted_truth', 'conflict_not_pairwise_fabrication', 'resolution_not_detection'),
        'FITNESS': ('applicability', 'exception', 'temporal_validity', 'conflict_detection',
                    'precedence_resolution', 'verdict', 'unknown_or_insufficient_evidence'),
        'GROUNDING': ('evidence_rule_binding', 'authority_rule_binding', 'world_time_decision_binding',
                      'candidate_semantics_binding', 'projection_graph_binding'),
        'REPLAY': ('deterministic_reconstruction', 'historical_immutability', 'predecessor_successor_integrity'),
    }
    return QualityProfileV1.create(organization_id=organization_id, workspace_id=workspace_id, name='tender_gdfm_mvl0', allow_warnings=True,
        requirements=tuple(sorted((QualityRequirementV1(requirement_id=name, gate=gate)
                                  for gate, names in requirements.items() for name in names),
                                 key=lambda item: item.requirement_id)))
