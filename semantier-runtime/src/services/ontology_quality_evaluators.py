"""Bounded ontology V&V producers. Each check owns canonical requirement evidence."""
from hashlib import sha256
from pathlib import Path

from contracts.ontology_building import ArtifactPinV1
from contracts.ontology_paired_evaluation import ArtifactPayloadPinV1
from contracts.ontology_quality import QualityFindingV1, QualityRequirementResultV1, normative_evidence_pin
from services.deterministic_rule_evaluator import SUPPORTED_OPERATORS
from services.ontology_rule_plan_compiler import OntologyRulePlanCompiler


def candidate_pin(candidate):
    return ArtifactPinV1(artifact_kind='normative_semantic_candidate_set',
        artifact_ref=candidate.candidate_set_ref, artifact_hash=candidate.candidate_set_hash)


def _scope(profile, *artifacts):
    expected = (profile.organization_id, profile.workspace_id)
    if any((item.organization_id, item.workspace_id) != expected for item in artifacts if item is not None):
        raise PermissionError('quality evaluator inputs cross tenant scope')


def _result(*, profile, candidate, requirement_id, failures=(), unavailable=None, evidence=(), extra_findings=(), normative_evidence=(), not_applicable=False, source_requirement_assessments=()):
    requirement = next((item for item in profile.requirements if item.requirement_id == requirement_id), None)
    if requirement is None:
        return None  # This profile does not request the check.
    if failures or extra_findings:
        unavailable = None
    status = 'BLOCK' if failures or extra_findings else requirement.unavailable_status if unavailable else 'PASS'
    if not_applicable:
        if requirement.applicability != 'NOT_APPLICABLE' or not evidence:
            raise ValueError('quality exclusion requires explicit reviewed applicability evidence')
        if status == 'PASS':
            status = 'NOT_APPLICABLE'
    refs = tuple(sorted({ref for frame in candidate.frames for ref in frame.evidence_refs}))
    findings = tuple(QualityFindingV1.create(requirement_id=requirement_id, gate=requirement.gate,
        severity='BLOCK' if status == 'BLOCK' else 'WARN', code='evaluation_unavailable' if unavailable else 'requirement_violation',
        message=message, evidence_refs=refs, subject_refs=(candidate.candidate_set_ref,))
        for message in sorted(set(failures or ((unavailable,) if unavailable else ()))))
    return QualityRequirementResultV1.create(requirement_id=requirement_id, gate=requirement.gate,
        quality_profile=profile.pin(), subject=candidate_pin(candidate),
        evaluator=ArtifactPinV1(artifact_kind='quality_evaluator', artifact_ref='ontology-quality:' + requirement.gate,
            artifact_hash='sha256:' + sha256(Path(__file__).read_bytes()).hexdigest()),
        source_requirement_assessments=source_requirement_assessments if requirement_id == 'verdict' else (),
        normative_evidence=normative_evidence, status=status, findings=tuple(sorted((*findings, *extra_findings), key=lambda item: item.finding_ref)),
        unavailable_reason=unavailable, evidence=tuple(sorted(evidence,
            key=lambda item: (item.artifact_kind, item.artifact_ref, item.artifact_hash))))


def _selected(*results):
    return tuple(result for result in results if result is not None)


def _plan_pin(plan):
    return ArtifactPinV1(artifact_kind='compiled_rule_plan', artifact_ref=plan.plan_ref, artifact_hash=plan.plan_hash)


class LogicalQualityEvaluator:
    def evaluate(self, *, candidate, profile):
        _scope(profile, candidate)
        by_id = {frame.frame_id: frame for frame in candidate.frames}
        failures = []
        if len(by_id) != len(candidate.frames):
            failures.append('logical:duplicate-frame')
        if candidate.completeness_state != 'COMPLETE' or candidate.unresolved_slot_refs:
            failures.append('logical:candidate-incomplete')
        for frame in candidate.frames:
            refs = ()
            if frame.frame_kind == 'rule':
                refs = (*frame.exception_refs, *frame.scope_refs, *frame.qualification_refs,
                        *frame.temporal_condition_refs, *frame.precedence_refs)
            elif frame.frame_kind in {'exception', 'scope'}:
                refs = (frame.applies_to_rule_ref,)
                if refs[0] in by_id and by_id[refs[0]].frame_kind != 'rule':
                    failures.append('logical:invalid-rule-endpoint:' + refs[0])
            elif frame.frame_kind == 'precedence':
                refs = (frame.higher_priority_ref, frame.lower_priority_ref)
            for ref in refs:
                if ref not in by_id:
                    failures.append('logical:missing-frame:' + ref)
            for predicate in (getattr(frame, 'conditions', None) or ()):
                if predicate.operator not in SUPPORTED_OPERATORS:
                    failures.append('logical:unsupported-operator:' + predicate.operator)
        return _result(profile=profile, candidate=candidate, requirement_id='semantic_integrity', failures=failures)


class GroundingQualityEvaluator:
    def evaluate(self, *, candidate, validation, plan, profile, authority=None):
        _scope(profile, candidate, validation, plan, authority)
        common = dict(profile=profile, candidate=candidate)
        by_id = {frame.frame_id: frame for frame in candidate.frames}
        evidence_failures = []
        for rule in plan.rules:
            for item in (rule, *rule.predicates, *rule.exception_predicates, *(rule.input_requirements or ())):
                frame = by_id.get(item.ir_node_ref)
                if frame is None or not set(item.evidence_refs).issubset(frame.evidence_refs):
                    evidence_failures.append('grounding:unbound-rule-evidence:' + item.ir_node_ref)
        candidate_failures = []
        if (plan.candidate_set_ref, plan.candidate_set_hash) != (candidate.candidate_set_ref, candidate.candidate_set_hash):
            candidate_failures.append('grounding:candidate-pin-mismatch')
        if (plan.validation_contract_ref, plan.validation_contract_hash) != (validation.validation_contract_ref, validation.validation_contract_hash):
            candidate_failures.append('grounding:validation-pin-mismatch')
        for name in ('source_identity_ref', 'source_hash', 'competency_contract_ref', 'competency_contract_hash',
                     'normative_schema_ref', 'normative_schema_hash'):
            if getattr(candidate, name) != getattr(validation, name):
                candidate_failures.append('grounding:independent-basis-mismatch:' + name)
        projection_failures = list(candidate_failures)
        if (len({rule.rule_id for rule in plan.rules}) != len(plan.rules)
            or {rule.rule_id for rule in plan.rules} != {frame.frame_id for frame in candidate.frames if frame.frame_kind == 'rule'}):
            projection_failures.append('grounding:projection-rule-coverage-mismatch')
        time_failures = []
        for rule in plan.rules:
            source = by_id.get(rule.ir_node_ref)
            for ref in getattr(source, 'temporal_condition_refs', ()):
                frame = by_id.get(ref)
                if frame is None:
                    time_failures.append('grounding:missing-world-time-frame:' + ref)
                elif frame.frame_kind == 'temporal_condition':
                    expected = {(frame.event_time_ref, 'DATE_ON_OR_AFTER', frame.effective_from),
                                (frame.event_time_ref, 'DATE_ON_OR_BEFORE', frame.effective_until)}
                    actual = {(p.fact_ref, p.operator, p.expected_value) for p in rule.predicates if p.ir_node_ref == ref}
                    if actual != expected:
                        time_failures.append('grounding:world-time-binding-mismatch:' + ref)
        authority_failures = []
        authority_evidence = ()
        if authority is not None:
            # Only an exact governed source/dependency binding can establish this
            # relation; ordinary evidence references never become authority.
            sources = [item for item in authority.dependencies
                       if (item.source_id, item.source_content_hash or item.artifact_hash) ==
                       (candidate.source_identity_ref, candidate.source_hash)]
            for frame in candidate.frames:
                if frame.frame_kind not in {'rule', 'precedence'}:
                    continue
                role = 'precedence_authority' if frame.frame_kind == 'precedence' else 'rule_authority'
                regions = {section.section_id for item in sources for section in item.sections if role in section.dependency_roles}
                # Reviewed canonical-region locators are source-scoped identities,
                # not article labels or inferred text similarity.
                regions.update(section.stable_locator.removeprefix('canonical-region:')
                    for item in sources if item.source_content_hash is not None
                    for section in item.sections if role in section.dependency_roles
                    and section.stable_locator.startswith('canonical-region:'))
                if not set(frame.source_region_refs).issubset(regions):
                    authority_failures.append('grounding:authority-rule-binding-mismatch:' + frame.frame_id)
            authority_evidence = (ArtifactPinV1(artifact_kind='normative_authority_dependency_set',
                artifact_ref=authority.set_ref, artifact_hash=authority.set_hash),)
        return _selected(
            _result(**common, requirement_id='evidence_rule_binding', failures=evidence_failures, evidence=(_plan_pin(plan),)),
            _result(**common, requirement_id='candidate_semantics_binding', failures=candidate_failures, evidence=(_plan_pin(plan),)),
            _result(**common, requirement_id='projection_graph_binding', failures=projection_failures, evidence=(_plan_pin(plan),)),
            _result(**common, requirement_id='world_time_decision_binding', failures=time_failures, evidence=(_plan_pin(plan),)),
            _result(**common, requirement_id='authority_rule_binding', failures=authority_failures,
                unavailable='missing_authority_basis' if authority is None else None, evidence=authority_evidence),
        )


class ReplayQualityEvaluator:
    def evaluate(self, *, candidate, validation, plan, profile, retained_inputs=(), predecessor=None):
        _scope(profile, candidate, validation, plan, predecessor)
        common = dict(profile=profile, candidate=candidate)
        reconstruction_failures = []
        try:
            expected = OntologyRulePlanCompiler(compiler_ref=plan.compiler_ref, compiler_hash=plan.compiler_hash).compile(
                candidate=candidate, validation_contract_ref=validation.validation_contract_ref,
                validation_contract_hash=validation.validation_contract_hash,
                competency_refs=plan.rules[0].competency_refs, compiled_at=plan.compiled_at, domain_basis=plan.domain_basis)
            if expected != plan:
                reconstruction_failures.append('replay:projection-mismatch')
        except (ValueError, IndexError) as exc:
            reconstruction_failures.append('replay:projection-unavailable:' + str(exc))
        expected_inputs = tuple(ArtifactPayloadPinV1.of(artifact, ref, hash_) for artifact, ref, hash_ in (
            (candidate, 'candidate_set_ref', 'candidate_set_hash'),
            (validation, 'validation_contract_ref', 'validation_contract_hash'), (plan, 'plan_ref', 'plan_hash')))
        supplied = {pin.artifact_ref: pin for pin in retained_inputs}
        immutable_failures = ['replay:historical-input-mismatch:' + pin.artifact_ref
                              for pin in expected_inputs if pin.artifact_ref in supplied and supplied[pin.artifact_ref] != pin]
        if len(supplied) != len(retained_inputs):
            immutable_failures.append('replay:duplicate-historical-input')
        missing_inputs = any(pin.artifact_ref not in supplied for pin in expected_inputs)
        lineage_failures = []
        missing_predecessor = candidate.predecessor_candidate_set_ref is not None and predecessor is None
        if predecessor is not None:
            if (candidate.predecessor_candidate_set_ref, candidate.predecessor_candidate_set_hash) != (
                predecessor.candidate_set_ref, predecessor.candidate_set_hash):
                lineage_failures.append('replay:predecessor-pin-mismatch')
            if candidate.compiled_at < predecessor.compiled_at:
                lineage_failures.append('replay:successor-precedes-parent')
        elif candidate.predecessor_candidate_set_ref is None and candidate.change_cause != 'initial_compilation':
            lineage_failures.append('replay:successor-missing-parent-pin')
        return _selected(
            _result(**common, requirement_id='deterministic_reconstruction', failures=reconstruction_failures, evidence=(_plan_pin(plan),)),
            _result(**common, requirement_id='historical_immutability', failures=immutable_failures,
                unavailable='missing_historical_inputs' if missing_inputs and not immutable_failures else None,
                evidence=tuple(ArtifactPinV1(artifact_kind='artifact_payload', artifact_ref=pin.artifact_ref,
                    artifact_hash=pin.payload_hash) for pin in retained_inputs)),
            _result(**common, requirement_id='predecessor_successor_integrity', failures=lineage_failures,
                unavailable='missing_predecessor' if missing_predecessor else None),
        )


class ModelingQualityEvaluator:
    def evaluate(self, *, candidate, validation, plan, profile, authority=None, fitness_results=(), bundles=(), applicability_review=None):
        from contracts.normative_semantics import NormativeSemanticCandidateSetV1
        from contracts.ontology_certification import NormativeAuthorityDependencySetV1
        _scope(profile, candidate, validation, plan)
        common = dict(profile=profile, candidate=candidate)
        precedence_refs = {frame.frame_id for frame in candidate.frames if frame.frame_kind == 'precedence'}
        failures = ['modeling:precedence-used-as-applicability:' + predicate.ir_node_ref
                    for rule in plan.rules for predicate in (*rule.predicates, *(rule.input_requirements or ()))
                    if predicate.ir_node_ref in precedence_refs]
        candidate_failures = []
        if not isinstance(candidate, NormativeSemanticCandidateSetV1):
            candidate_failures.append('modeling:subject-is-not-a-candidate')
        if validation.independence.semantic_candidate_input_ref is not None:
            candidate_failures.append('modeling:candidate-used-as-validation-oracle')
        authority_failures = []
        if authority is not None:
            if not isinstance(authority, NormativeAuthorityDependencySetV1):
                authority_failures.append('modeling:evidence-used-as-authority')
            elif authority.set_ref == candidate.candidate_set_ref:
                authority_failures.append('modeling:candidate-used-as-authority')
            else:
                _scope(profile, authority)
        normative = tuple(a for result in fitness_results for a in result.normative_evidence)
        normative_failures = []
        if normative:
            try:
                FitnessQualityEvaluator.verify_normative_evidence(plan=plan, bundles=bundles, artifacts=normative)
            except ValueError as exc:
                normative_failures.append('modeling:' + str(exc))
        has_conflict = any(a.schema_version == 'conflict_core.v1' for a in normative)
        has_resolution = any(a.schema_version == 'conflict_resolution.v1' for a in normative)
        normative_pins = tuple(normative_evidence_pin(a) for a in normative)
        conflict_unavailable = None if has_conflict else 'missing_conflict_evaluation'
        resolution_unavailable = None if has_resolution else 'missing_resolution_evaluation'
        if applicability_review is not None and any(r.requirement_id == 'conflict_detection' and
            r.applicability == 'NOT_APPLICABLE' for r in profile.requirements):
            from services.ontology_quality_profile_service import QualityProfileService
            try:
                QualityProfileService.verify_evaluation_basis(profile=profile, review=applicability_review,
                    candidate=candidate, validation=validation)
                FitnessQualityEvaluator.verify_precedence_traces(candidate=candidate, validation=validation,
                    plan=plan, bundles=bundles, expectations=applicability_review.precedence_expectations)
                if normative:
                    raise ValueError('conflict-artifacts-introduced-under-reviewed-exclusion')
                if {r.rule_id for r in plan.rules} != {f.frame_id for f in candidate.frames if f.frame_kind == 'rule'}:
                    raise ValueError('fabricated-or-missing-projected-rule')
                if any(r.requirement_id == 'precedence_resolution' and r.status != 'PASS' for r in fitness_results):
                    raise ValueError('precedence-fitness-not-established')
                if not any(r.requirement_id == 'precedence_resolution' for r in fitness_results):
                    raise ValueError('precedence-fitness-missing')
            except ValueError as exc:
                normative_failures.append('modeling:' + str(exc))
            normative_pins = (applicability_review.pin(), _plan_pin(plan))
            conflict_unavailable = resolution_unavailable = None
        return _selected(
            _result(**common, requirement_id='precedence_not_applicability', failures=failures, evidence=(_plan_pin(plan),)),
            _result(**common, requirement_id='evidence_not_authority', failures=authority_failures,
                unavailable='missing_authority_basis' if authority is None else None),
            _result(**common, requirement_id='candidate_not_accepted_truth', failures=candidate_failures),
            _result(**common, requirement_id='conflict_not_pairwise_fabrication', failures=normative_failures, evidence=normative_pins,
                unavailable=conflict_unavailable),
            _result(**common, requirement_id='resolution_not_detection', failures=normative_failures, evidence=normative_pins,
                unavailable=resolution_unavailable),
        )


class FitnessQualityEvaluator:
    """Execute the independently reviewed oracle; no caller-supplied substitute cases."""
    REQUIREMENTS = ('applicability', 'exception', 'temporal_validity', 'conflict_detection',
                    'precedence_resolution', 'verdict', 'unknown_or_insufficient_evidence')

    @staticmethod
    def source_requirement_mapping(*, candidate, validation):
        """Bounded typed meaning alignment against a candidate-independent inventory.

        Complete inspection supports MISSING; absent/unresolved bases support only
        AMBIGUOUS. Alternative adequate frames remain disjunctive support.
        """
        from contracts.validation_contract import SourceRequirementAssessmentV1
        from eos.artifact_hashing import content_hash
        inventory = validation.source_requirement_inventory
        if inventory is None:
            return ()  # Historical artifacts make no instance-coverage claim.
        basis_valid = inventory.eligibility == 'ELIGIBLE' and candidate.completeness_state == 'COMPLETE'
        basis_valid = basis_valid and all(getattr(candidate, key) == getattr(validation, key) for key in (
            'source_identity_ref', 'source_hash', 'competency_contract_ref', 'competency_contract_hash',
            'normative_schema_ref', 'normative_schema_hash'))
        def meaning(predicate):
            return content_hash(dict(operator=predicate.operator, fact_ref=predicate.fact_ref,
                expected_value=predicate.expected_value, comparison_fact_ref=predicate.comparison_fact_ref))
        by_id = {frame.frame_id: frame for frame in candidate.frames}
        supported = {}
        results = []
        items = sorted(inventory.instances, key=lambda i: (i.semantic_dimension != 'rule', i.instance_id))
        for item in items:
            matches = []
            rejected = set()
            uncertain = not basis_valid
            if item.owning_rule_id is not None:
                uncertain = uncertain or any(a.instance_id == item.owning_rule_id and a.mapping_status == 'AMBIGUOUS' for a in results)
            for frame in candidate.frames:
                if frame.frame_kind != item.semantic_dimension:
                    continue
                if (item.occurrence_ref not in frame.source_region_refs
                    or not set(item.evidence_refs).issubset(frame.evidence_refs)):
                    rejected.add('wrong_source_occurrence_or_evidence')
                    continue
                if frame.epistemic_status != 'FOUND':
                    uncertain = True
                    continue
                actual = sorted(meaning(p) for p in frame.conditions)
                if not any(actual == sorted(meaning(p) for p in bundle) for bundle in item.support_alternatives):
                    rejected.add('wrong_meaning_or_grouping')
                    continue
                if item.semantic_dimension == 'rule':
                    if frame.modality != item.modality:
                        rejected.add('wrong_modality')
                        continue
                else:
                    parents = supported.get(item.owning_rule_id, ())
                    if frame.applies_to_rule_ref not in parents:
                        rejected.add('wrong_parent_or_direction')
                        continue
                    if frame.frame_id not in by_id[frame.applies_to_rule_ref].exception_refs:
                        rejected.add('unattached_support')
                        continue
                matches.append(frame.frame_id)
            status = 'AMBIGUOUS' if uncertain else 'MAPPED' if matches else 'MISSING'
            supported[item.instance_id] = tuple(sorted(matches)) if status == 'MAPPED' else ()
            results.append(SourceRequirementAssessmentV1(instance_id=item.instance_id,
                owning_rule_id=item.owning_rule_id, mapping_status=status,
                reason='typed_source_support' if status == 'MAPPED' else
                    'inventory_or_candidate_basis_unresolved' if status == 'AMBIGUOUS' else
                    'no_adequate_source_bound_support:' + ','.join(sorted(rejected or {'absent_kind'})),
                supporting_frame_refs=supported[item.instance_id]))
        return tuple(sorted(results, key=lambda a: a.instance_id))

    @staticmethod
    def source_requirement_witnesses(*, validation, plan, bundles, assessments):
        """Read actual retained per-frame traces; never substitute a final verdict."""
        from contracts.validation_contract import SourceRequirementAssessmentV1
        instances = {i.instance_id: i for i in validation.source_requirement_inventory.instances}
        output, failures = [], {}
        for assessment in assessments:
            observations = []
            case_ids = []
            for test, bundle in zip(validation.executable_tests, bundles, strict=True):
                if assessment.instance_id not in test.required_obligation_ids:
                    continue
                case_ids.append(test.source_case_id or test.test_id)
                expected = test.expected_instance_results[assessment.instance_id]
                states = []
                for ref in assessment.supporting_frame_refs:
                    if instances[assessment.instance_id].semantic_dimension == 'rule':
                        states.extend(r.outcome for r in bundle.candidate_rule_evaluations if r.rule_id == ref)
                    else:
                        for rule in bundle.candidate_rule_evaluations:
                            group = next((g for r in plan.rules if r.rule_id == rule.rule_id
                                for g in r.exception_groups or () if g.ir_node_ref == ref), None)
                            predicates = tuple(p for p in rule.exception_results if p.ir_node_ref == ref)
                            if group is None or tuple(p.predicate_id for p in predicates) != group.predicate_ids:
                                continue
                            values = {p.result for p in predicates}
                            states.append('NO_MATCH' if 'FAILED' in values else
                                'UNKNOWN' if values & {'UNKNOWN', 'UNSUPPORTED'} else 'MATCH')
                actual = 'MATCH' if 'MATCH' in states else 'UNKNOWN' if 'UNKNOWN' in states else 'NO_MATCH' if states else None
                observation = 'NOT_RUN' if actual is None else 'FAIL' if actual != expected else 'UNKNOWN' if actual == 'UNKNOWN' else 'PASS'
                observations.append(observation)
                if observation in {'FAIL', 'NOT_RUN'}:
                    failures.setdefault(test.test_id, []).append('fitness:instance-witness:' + assessment.instance_id + ':' + observation)
            status = 'NOT_RUN' if not observations or 'NOT_RUN' in observations else 'FAIL' if 'FAIL' in observations else 'UNKNOWN' if 'UNKNOWN' in observations else 'PASS'
            output.append(SourceRequirementAssessmentV1(**{**assessment.model_dump(),
                'witness_status': status, 'witness_case_ids': tuple(sorted(case_ids))}))
        return tuple(output), failures

    @classmethod
    def verify_precedence_traces(cls, *, candidate, validation, plan, bundles, expectations):
        """Verify source-reviewed branch traces without rerunning the execution engine."""
        from services.ontology_quality_profile_service import QualityProfileService
        QualityProfileService.verify_precedence_basis(candidate=candidate, validation=validation, expectations=expectations)
        if (plan.candidate_set_ref, plan.candidate_set_hash) != (candidate.candidate_set_ref, candidate.candidate_set_hash):
            raise ValueError('quality-precedence-candidate-mismatch')
        if not bundles:
            raise ValueError('quality-precedence-cases-missing')
        cls.verify_retained_cases(plan=plan, validation=validation, bundles=bundles, evaluated_at=bundles[0].evaluated_at)
        cases = {b.execution_epoch_ref: b for b in bundles}
        rules = {r.rule_id: r for r in plan.rules}
        relations = {r.ir_node_ref: r for r in plan.precedences or ()}
        for expected in expectations:
            bundle = cases['quality-case:' + expected.test_id]
            actual = next((r for r in bundle.candidate_rule_evaluations if r.rule_id == expected.rule_ref), None)
            rule = rules.get(expected.rule_ref)
            if actual is None or rule is None or actual.outcome != expected.expected_rule_outcome:
                raise ValueError('quality-precedence-rule-outcome-mismatch')
            if expected.relation_kind == 'exception':
                group = next((g for g in rule.exception_groups or () if g.ir_node_ref == expected.relation_ref), None)
                predicates = tuple(p for p in actual.exception_results if p.ir_node_ref == expected.relation_ref)
                if group is None or tuple(p.predicate_id for p in predicates) != group.predicate_ids:
                    raise ValueError('quality-precedence-exception-trace-missing')
                states = {p.result for p in predicates}
                outcome = 'NO_MATCH' if 'FAILED' in states else 'UNKNOWN' if states & {'UNKNOWN', 'UNSUPPORTED'} else 'MATCH'
            else:
                relation = relations.get(expected.relation_ref)
                traces = [r for r in actual.precedence_results or () if r.ir_node_ref == expected.relation_ref]
                if relation is None or len(traces) != 1:
                    raise ValueError('quality-precedence-relation-trace-missing')
                trace = traces[0]
                if (trace.higher_priority_ref, trace.lower_priority_ref) != (expected.controlling_ref, expected.rule_ref):
                    raise ValueError('quality-precedence-controller-mismatch')
                predicates = trace.condition_results
                if tuple(p.predicate_id for p in predicates) != tuple(p.predicate_id for p in relation.conditions):
                    raise ValueError('quality-precedence-guard-coverage-mismatch')
                outcome = trace.outcome
            if outcome != expected.expected_branch_outcome:
                raise ValueError('quality-precedence-branch-mismatch')
            if {p.predicate_id: p.result for p in predicates} != expected.expected_predicate_results:
                raise ValueError('quality-precedence-predicate-result-mismatch')

    @staticmethod
    def verify_normative_evidence(*, plan, bundles, artifacts):
        """Verify stored stage identities/relationships, never rerun a current engine."""
        from contracts.normative_conflict import (
            NormativeSetV1, NormativeCompatibilityResultV1, ConflictCoreV1, ConflictResolutionV1,
        )
        sets = {a.normative_set_ref: a for a in artifacts if isinstance(a, NormativeSetV1)}
        results = {a.result_ref: a for a in artifacts if isinstance(a, NormativeCompatibilityResultV1)}
        cores = {a.conflict_core_ref: a for a in artifacts if isinstance(a, ConflictCoreV1)}
        resolutions = [a for a in artifacts if isinstance(a, ConflictResolutionV1)]
        contexts = {b.bundle_ref: b for b in bundles}
        rule_ids = {r.rule_id for r in plan.rules}
        for item in sets.values():
            bundle = contexts.get(item.context_ref)
            if (bundle is None or item.context_hash != bundle.bundle_hash
                or (item.organization_id, item.workspace_id) != (plan.organization_id, plan.workspace_id)
                or (item.compiled_rule_plan_ref, item.compiled_rule_plan_hash) != (plan.plan_ref, plan.plan_hash)
                or item.formed_at != bundle.evaluated_at):
                raise ValueError('normative-set-case-binding-mismatch')
            groups = (set(item.member_rule_refs), set(item.inactive_rule_refs), set(item.unknown_rule_refs))
            if set.union(*groups) != rule_ids or sum(map(len, groups)) != len(rule_ids):
                raise ValueError('normative-set-rule-partition-mismatch')
        if len(results) != len(sets) or {r.normative_set_ref for r in results.values()} != set(sets):
            raise ValueError('normative-compatibility-coverage-mismatch')
        for result in results.values():
            item = sets.get(result.normative_set_ref)
            if item is None or result.normative_set_hash != item.normative_set_hash or result.evaluated_at != item.formed_at:
                raise ValueError('compatibility-set-binding-mismatch')
            if not set(result.conflicting_effect_refs).issubset(e.effect_ref for e in item.effects):
                raise ValueError('compatibility-effect-binding-mismatch')
        conflicts = {r.result_ref for r in results.values() if r.outcome == 'CONFLICT'}
        if len(cores) != len(conflicts) or {c.compatibility_result_ref for c in cores.values()} != conflicts:
            raise ValueError('conflict-core-coverage-mismatch')
        for core in cores.values():
            result = results[core.compatibility_result_ref]
            item = sets[result.normative_set_ref]
            effects = {e.effect_ref: e for e in item.effects}
            members = {effects[ref].rule_ref for ref in result.conflicting_effect_refs}
            if (core.normative_set_ref != item.normative_set_ref or core.context_ref != item.context_ref
                or set(core.member_effect_refs) != set(result.conflicting_effect_refs)
                or set(core.member_rule_refs) != members):
                raise ValueError('conflict-core-membership-mismatch')
        if len(resolutions) != len(cores) or {r.conflict_core_ref for r in resolutions} != set(cores):
            raise ValueError('conflict-resolution-coverage-mismatch')
        for resolution in resolutions:
            core = cores[resolution.conflict_core_ref]
            item = sets[core.normative_set_ref]
            if (not set(resolution.controlling_rule_refs).issubset(core.member_rule_refs)
                or resolution.resolved_at != item.formed_at):
                raise ValueError('conflict-resolution-binding-mismatch')

    @staticmethod
    def verify_retained_cases(*, plan, validation, bundles, evaluated_at):
        """Check the retained case bindings without evaluating current runtime code."""
        from eos.artifact_hashing import content_hash
        if len(bundles) != len(validation.executable_tests):
            raise ValueError('fitness retained case coverage mismatch')
        rule_ids = tuple(rule.rule_id for rule in plan.rules)
        for test, bundle in zip(validation.executable_tests, bundles, strict=True):
            if ((bundle.organization_id, bundle.workspace_id) != (plan.organization_id, plan.workspace_id)
                or (bundle.plan_ref, bundle.plan_hash) != (plan.plan_ref, plan.plan_hash)
                or bundle.evaluated_at != evaluated_at
                or bundle.execution_epoch_ref != 'quality-case:' + test.test_id
                or content_hash(bundle.observed_facts) != content_hash(test.observed_facts)
                or bundle.target_evidence_refs != test.evidence_refs):
                raise ValueError('fitness retained case binding mismatch')
            if (bundle.all_rule_justifications is None
                or tuple(item.rule_id for item in bundle.candidate_rule_evaluations) != rule_ids):
                raise ValueError('fitness retained rule coverage mismatch')

    def evaluate(self, *, candidate, validation, plan, profile, evaluated_at, applicability_review=None):
        from contracts.validation_contract import ValidationContractV1
        from services.deterministic_rule_evaluator import DeterministicRuleEvaluator
        _scope(profile, candidate, validation, plan)
        validation = ValidationContractV1.model_validate(validation.model_dump(mode='json'))
        if plan is not None and (plan.validation_contract_ref, plan.validation_contract_hash) != (
            validation.validation_contract_ref, validation.validation_contract_hash):
            raise ValueError('fitness validation contract pin mismatch')
        assessments = self.source_requirement_mapping(candidate=candidate, validation=validation)
        instance_mapping = {a.instance_id: a for a in assessments}
        common = dict(profile=profile, candidate=candidate, source_requirement_assessments=assessments)
        missing = []
        kinds = {frame.frame_kind for frame in candidate.frames}
        if 'relative_temporal_condition' in kinds:
            kinds.add('temporal_condition')
        for obligation in validation.obligations:
            dimension = obligation.semantic_dimension
            if validation.source_requirement_inventory is not None:
                satisfied = instance_mapping[obligation.obligation_id].mapping_status != 'MISSING'
            elif dimension == 'modality':
                satisfied = any(frame.frame_kind == 'rule' and frame.modality for frame in candidate.frames)
            elif dimension == 'evidence_binding':
                satisfied = all(frame.evidence_refs for frame in candidate.frames)
            elif dimension in {'expected_inference', 'prohibited_inference'}:
                satisfied = bool(validation.executable_tests)
            else:
                satisfied = dimension in kinds
            if not satisfied:
                missing.append(obligation)
        missing_findings = {name: [] for name in self.REQUIREMENTS}
        dimension_gate = {'exception': 'exception', 'temporal_condition': 'temporal_validity',
                          'precedence': 'precedence_resolution'}
        for obligation in missing:
            for name in {'verdict', dimension_gate.get(obligation.semantic_dimension, 'applicability')}:
                missing_findings[name].append(QualityFindingV1.create(requirement_id=name, gate='FITNESS',
                    severity='BLOCK', code='missing_source_obligation', message='Missing declared ' + obligation.semantic_dimension,
                    subject_refs=tuple(sorted((candidate.candidate_set_ref, obligation.obligation_id))),
                    evidence_refs=tuple(sorted(set(obligation.evidence_refs)))))
        unavailable = None
        if validation.source_requirement_inventory is not None and (not assessments or any(a.mapping_status == 'AMBIGUOUS' for a in assessments)):
            unavailable = 'source_requirement_inventory_unavailable'
        if plan is not None and validation.source_requirement_inventory is not None and (
            plan.candidate_set_ref, plan.candidate_set_hash) != (candidate.candidate_set_ref, candidate.candidate_set_hash):
            raise ValueError('fitness candidate plan pin mismatch')
        if validation.review_state != 'accepted':
            unavailable = 'unreviewed_validation_oracle'
        elif any(test.observed_facts is None or test.expected_result not in {'MATCH', 'NO_MATCH', 'UNKNOWN'}
                 for test in validation.executable_tests):
            unavailable = 'missing_executable_case_basis'
        elif plan is None:
            unavailable = 'missing_compiled_plan'
        if unavailable:
            return _selected(*(_result(**common, requirement_id=name, unavailable=unavailable,
                extra_findings=missing_findings[name]) for name in self.REQUIREMENTS)), ()
        evaluator = DeterministicRuleEvaluator()
        bundles = tuple(evaluator.evaluate(plan=plan, execution_epoch_ref='quality-case:' + test.test_id,
            observed_facts=dict(test.observed_facts), target_evidence_refs=test.evidence_refs,
            evaluated_at=evaluated_at, include_all_rules=True) for test in validation.executable_tests)
        instance_failures = {}
        if validation.source_requirement_inventory is not None:
            assessments, instance_failures = self.source_requirement_witnesses(
                validation=validation, plan=plan, bundles=bundles, assessments=assessments)
            common['source_requirement_assessments'] = assessments
        failures = {name: [] for name in self.REQUIREMENTS}
        measured = {name: [] for name in self.REQUIREMENTS}
        retained_normative = []
        obligations = {item.obligation_id: item for item in validation.obligations}
        kinds = {frame.frame_kind for frame in candidate.frames}
        if 'relative_temporal_condition' in kinds:
            kinds.add('temporal_condition')
        for test, bundle in zip(validation.executable_tests, bundles, strict=True):
            intrinsic = evaluator.evaluate_rules_without_precedence(plan, test.observed_facts)
            applicable_failures = []
            expected = test.expected_rule_results
            if expected is not None:
                if set(expected) != {item.rule_id for item in intrinsic}:
                    applicable_failures.append('fitness:rule-expectations-incomplete:' + test.test_id)
                for item in intrinsic:
                    if item.rule_id in expected and item.outcome != expected[item.rule_id]:
                        applicable_failures.append(f'fitness:{test.test_id}:{item.rule_id}:expected-{expected[item.rule_id]}:actual-{item.outcome}')
            elif len(plan.rules) > 1:
                applicable_failures.append('fitness:rule-expectations-incomplete:' + test.test_id)
            else:
                # A single-rule oracle can also express intrinsic applicability
                # when no cross-rule priority suppression is present.
                if intrinsic[0].outcome != test.expected_result:
                    applicable_failures.append(f'fitness:{test.test_id}:intrinsic-outcome-mismatch')
            verdict_failures = []
            actual = bundle.result
            if test.result_rule_ref is not None:
                # Compare the reviewed CQ target after exception/precedence evaluation.
                # The legacy scalar remains authoritative for whole-plan tests.
                actual = next((item.outcome for item in bundle.candidate_rule_evaluations
                    if item.rule_id == test.result_rule_ref), None)
            if actual != test.expected_result:
                verdict_failures.append(f'fitness:{test.test_id}:expected-{test.expected_result}:actual-{actual}')
            if test.expected_compatibility is not None:
                from services.normative_conflict_analysis import (
                    NormativeSetCompiler, NormativeCompatibilityEvaluator, ConflictResolver,
                )
                engine_hash = 'sha256:' + sha256(Path(__file__).with_name('normative_conflict_analysis.py').read_bytes()).hexdigest()
                normative = NormativeSetCompiler().compile(candidate=candidate, plan=plan, evaluation=bundle,
                    context_ref=bundle.bundle_ref, context_hash=bundle.bundle_hash, formed_at=evaluated_at)
                compatibility_evaluator = NormativeCompatibilityEvaluator(
                    evaluator_ref='services/normative_conflict_analysis.py', evaluator_hash=engine_hash)
                compatibility = compatibility_evaluator.evaluate(normative_set=normative, evaluated_at=evaluated_at)
                artifacts = [normative, compatibility]
                core = resolution = None
                if compatibility.outcome == 'CONFLICT':
                    core = compatibility_evaluator.extract_core(normative_set=normative, result=compatibility)
                    resolution = ConflictResolver().resolve(core=core, plan=plan,
                        observed_facts=test.observed_facts, resolved_at=evaluated_at)
                    artifacts.extend((core, resolution))
                retained_normative.extend(artifacts)
                measured['conflict_detection'].extend(normative_evidence_pin(item) for item in artifacts)
                measured['conflict_detection'].append(ArtifactPinV1(artifact_kind='runtime_semantic_evaluation_bundle',
                    artifact_ref=bundle.bundle_ref, artifact_hash=bundle.bundle_hash))
                if compatibility.outcome != test.expected_compatibility:
                    failures['conflict_detection'].append('fitness:compatibility-mismatch:' + test.test_id)
                if (core.member_rule_refs if core else ()) != test.expected_conflict_rule_refs:
                    failures['conflict_detection'].append('fitness:conflict-membership-mismatch:' + test.test_id)
                failures['conflict_detection'].extend(applicable_failures)
                if test.expected_resolution is not None:
                    measured['precedence_resolution'].extend(normative_evidence_pin(item) for item in artifacts)
                    if resolution is None or resolution.outcome != test.expected_resolution:
                        failures['precedence_resolution'].append('fitness:resolution-mismatch:' + test.test_id)
                    if (resolution.controlling_rule_refs if resolution else ()) != test.expected_controlling_rule_refs:
                        failures['precedence_resolution'].append('fitness:controller-mismatch:' + test.test_id)
                    failures['precedence_resolution'].extend(applicable_failures)
            missing_obligations = []
            for ref in test.required_obligation_ids:
                obligation = obligations[ref]
                dimension = obligation.semantic_dimension
                if validation.source_requirement_inventory is not None:
                    satisfied = instance_mapping[ref].mapping_status == 'MAPPED'
                elif dimension == 'modality':
                    satisfied = any(frame.frame_kind == 'rule' and frame.modality for frame in candidate.frames)
                elif dimension == 'evidence_binding':
                    satisfied = all(frame.evidence_refs for frame in candidate.frames)
                elif dimension in {'expected_inference', 'prohibited_inference'}:
                    satisfied = True  # The independently declared case is actually executed above.
                else:
                    satisfied = dimension in kinds
                if not satisfied:
                    missing_obligations.append('fitness:missing-obligation:' + ref)
            missing_obligations.extend(instance_failures.get(test.test_id, ()))
            dimensions = {obligations[ref].semantic_dimension for ref in test.required_obligation_ids}
            names = ['applicability', 'verdict']
            if 'exception' in dimensions:
                names.append('exception')
            if 'temporal_condition' in dimensions:
                names.append('temporal_validity')
            if test.expected_result == 'UNKNOWN' or test.case_kind == 'ambiguity':
                names.append('unknown_or_insufficient_evidence')
            pin = ArtifactPinV1(artifact_kind='runtime_semantic_evaluation_bundle',
                artifact_ref=bundle.bundle_ref, artifact_hash=bundle.bundle_hash)
            for name in names:
                measured[name].append(pin)
                errors = applicable_failures if name == 'applicability' else verdict_failures
                failures[name].extend((*errors, *missing_obligations))
        excluded = False
        if applicability_review is not None:
            from services.ontology_quality_profile_service import QualityProfileService
            QualityProfileService.verify_evaluation_basis(profile=profile, review=applicability_review,
                candidate=candidate, validation=validation)
            excluded = any(r.requirement_id == 'conflict_detection' and r.applicability == 'NOT_APPLICABLE'
                for r in profile.requirements)
            if excluded:
                measured['conflict_detection'].append(applicability_review.pin())
                if retained_normative:
                    failures['conflict_detection'].append('fitness:conflict-evidence-under-reviewed-exclusion')
            try:
                self.verify_precedence_traces(candidate=candidate, validation=validation, plan=plan,
                    bundles=bundles, expectations=applicability_review.precedence_expectations)
            except ValueError as exc:
                failures['precedence_resolution'].append('fitness:' + str(exc))
            measured['precedence_resolution'].append(applicability_review.pin())
        return _selected(*(_result(**common, requirement_id=name, failures=failures[name],
            unavailable='missing_competency_cases' if not measured[name] else None,
            evidence=tuple(measured[name]), extra_findings=missing_findings[name],
            not_applicable=excluded and name == 'conflict_detection',
            normative_evidence=tuple(sorted(retained_normative, key=lambda a: (a.schema_version, normative_evidence_pin(a).artifact_ref)))
                if name == 'conflict_detection' else ()) for name in self.REQUIREMENTS)), bundles


class ConsumerBindingQualityEvaluator:
    """Opt-in verification-only seam; no change to legacy FITNESS registration."""

    def evaluate(self, *, verification, profile):
        from contracts.source_consumer_verification import ConsumerVerificationV1
        v = ConsumerVerificationV1.model_validate(verification.model_dump(mode='json'))
        if (v.organization_id, v.workspace_id) != (profile.organization_id, profile.workspace_id):
            raise ValueError('consumer evaluation tenant scope mismatch')
        requirement = next(r for r in profile.requirements if r.requirement_id == 'source_consumer_binding')
        if requirement.gate != 'FITNESS' or requirement.unavailable_status != 'NOT_EVALUATED':
            raise ValueError('unsupported consumer verification profile')
        unavailable = v.eligibility == 'UNRESOLVED'
        failures = tuple(a for a in v.target_assessments if a.mapping_status != 'MAPPED' or a.witness_status != 'PASS')
        findings = () if unavailable else tuple(sorted((QualityFindingV1.create(
            requirement_id='source_consumer_binding', gate='FITNESS', severity='BLOCK',
            code='consumer_support_unverified', message=a.requirement_id + '/' + a.target_id + ':' + a.mapping_status + '/' + a.witness_status,
            evidence_refs=a.case_ids, subject_refs=tuple(sorted((a.requirement_id, a.target_id))),
        ) for a in failures), key=lambda f:f.finding_ref))
        return QualityRequirementResultV1.create(requirement_id='source_consumer_binding', gate='FITNESS',
            quality_profile=profile.pin(), subject=v.plan,
            evaluator=ArtifactPinV1(artifact_kind='quality_evaluator', artifact_ref='ontology-quality:source-consumer-binding',
                artifact_hash='sha256:' + sha256(Path(__file__).read_bytes()).hexdigest()),
            status='NOT_EVALUATED' if unavailable else 'BLOCK' if failures else 'PASS', findings=findings,
            unavailable_reason='source_target_inventory_unresolved' if unavailable else None,
            consumer_verification=v)
