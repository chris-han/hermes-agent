"""Assemble exact canonical evaluator evidence without qualification by omission."""
from datetime import datetime, timedelta
from importlib.metadata import version
import sys

from eos.artifact_hashing import content_hash
from hashlib import sha256
from pathlib import Path

from contracts.ontology_building import ArtifactPinV1
from contracts.ontology_quality import (
    QualityEvaluationRunV1, QualityFindingV1, QualityProfileV1, QualityRequirementResultV1,
)


class QualityEvaluationService:
    def evaluate(self, *, candidate, validation, plan, profile, executed_at,
                 authority=None, retained_inputs=(), predecessor=None, applicability_review=None):
        from services.ontology_quality_evaluators import (
            FitnessQualityEvaluator, GroundingQualityEvaluator, LogicalQualityEvaluator,
            ModelingQualityEvaluator, ReplayQualityEvaluator, _result, _scope, candidate_pin,
        )
        from services.ontology_structural_validator import ConstraintQualityEvaluator
        _scope(profile, candidate, validation, plan, predecessor, authority)
        from services.ontology_quality_profile_service import QualityProfileService
        QualityProfileService.verify_evaluation_basis(profile=profile, review=applicability_review,
            candidate=candidate, validation=validation)
        common = dict(candidate=candidate, validation=validation, plan=plan, profile=profile)
        gates = {item.gate for item in profile.requirements}
        results = []
        bundles = ()
        for gate in ('LOGIC', 'CONSTRAINT', 'FITNESS', 'MODELING', 'GROUNDING', 'REPLAY'):
            if gate not in gates:
                continue
            try:
                if gate == 'LOGIC':
                    produced = (LogicalQualityEvaluator().evaluate(candidate=candidate, profile=profile),)
                elif gate == 'CONSTRAINT':
                    produced = (ConstraintQualityEvaluator().evaluate(candidate, profile=profile),)
                elif gate == 'MODELING':
                    produced = ModelingQualityEvaluator().evaluate(**common, authority=authority,
                        fitness_results=tuple(r for r in results if r.gate == 'FITNESS'), bundles=bundles,
                        applicability_review=applicability_review)
                elif gate == 'FITNESS':
                    produced, bundles = FitnessQualityEvaluator().evaluate(**common, evaluated_at=executed_at,
                        applicability_review=applicability_review)
                elif gate == 'GROUNDING':
                    produced = GroundingQualityEvaluator().evaluate(**common, authority=authority)
                else:
                    produced = ReplayQualityEvaluator().evaluate(**common, retained_inputs=retained_inputs, predecessor=predecessor)
            except Exception as exc:
                produced = tuple(_result(profile=profile, candidate=candidate,
                    requirement_id=item.requirement_id, unavailable='evaluator_error:' + type(exc).__name__)
                    for item in profile.requirements if item.gate == gate)
            results.extend(item for item in produced if item is not None)
        run = self.assemble(profile=profile, subject=candidate_pin(candidate),
            competency=ArtifactPinV1(artifact_kind='competency_contract',
                artifact_ref=candidate.competency_contract_ref, artifact_hash=candidate.competency_contract_hash),
            validation=ArtifactPinV1(artifact_kind='validation_contract',
                artifact_ref=validation.validation_contract_ref, artifact_hash=validation.validation_contract_hash),
            results=tuple(results), executed_at=executed_at)
        return run, bundles

    def assemble(
        self, *, profile: QualityProfileV1, subject: ArtifactPinV1,
        competency: ArtifactPinV1, validation: ArtifactPinV1,
        results: tuple[QualityRequirementResultV1, ...], executed_at: datetime,
    ) -> QualityEvaluationRunV1:
        profile = QualityProfileV1.model_validate(profile)
        results = tuple(QualityRequirementResultV1.model_validate(result) for result in results)
        by_id = {result.requirement_id: result for result in results}
        if len(by_id) != len(results):
            raise ValueError('duplicate quality requirement result')
        if set(by_id) - {item.requirement_id for item in profile.requirements}:
            raise ValueError('unknown quality requirement result')
        evaluator = ArtifactPinV1(artifact_kind='quality_evaluation_service',
            artifact_ref='services/ontology_quality_evaluation_service.py',
            artifact_hash='sha256:' + sha256(Path(__file__).read_bytes()).hexdigest())
        for requirement in profile.requirements:
            if requirement.requirement_id in by_id:
                continue
            finding = QualityFindingV1.create(requirement_id=requirement.requirement_id,
                gate=requirement.gate, code='missing_evaluator',
                severity='BLOCK' if requirement.unavailable_status == 'BLOCK' else 'WARN',
                message='No evaluator result was supplied for this profile requirement.', evidence_refs=())
            by_id[requirement.requirement_id] = QualityRequirementResultV1.create(
                requirement_id=requirement.requirement_id, gate=requirement.gate,
                quality_profile=profile.pin(), subject=subject, evaluator=evaluator,
                status=requirement.unavailable_status, unavailable_reason='missing_evaluator', findings=(finding,))
        return QualityEvaluationRunV1.create(organization_id=profile.organization_id,
            workspace_id=profile.workspace_id, profile_definition=profile, quality_profile=profile.pin(),
            subject=subject, competency=competency, validation=validation,
            results=tuple(by_id[item.requirement_id] for item in profile.requirements), executed_at=executed_at)

    @staticmethod
    def _code_units():
        from contracts.ontology_paired_evaluation import EvaluatorCodeUnitV1
        root = Path(__file__).resolve().parents[1]
        paths = ("services/ontology_quality_evaluation_service.py", "services/ontology_quality_evaluators.py",
            "services/ontology_quality_profile_service.py", "contracts/ontology_quality.py", "contracts/artifact_pin.py", "contracts/normative_semantics.py", "services/ontology_structural_validator.py",
            "services/ontology_rule_plan_compiler.py",
            "services/deterministic_rule_evaluator.py", "services/normative_conflict_analysis.py",
            "contracts/normative_conflict.py", "contracts/ontology_validation.py", "contracts/ontology_certification.py",
            "contracts/validation_contract.py",
            "contracts/source_validation_execution.py", "contracts/ontology_paired_evaluation.py", "eos/artifact_hashing.py")
        units = []
        for path in paths:
            code = (root / path).read_bytes()
            units.append(EvaluatorCodeUnitV1(code_ref=path, code_bytes=code,
                code_hash="sha256:" + sha256(code).hexdigest()))
        return tuple(units)

    def run_source_persisted(self, *, ctx, authority_resolver, store, candidate_pin, review_pin,
                             request_id, executed_at, authority_pin=None, profile_pin=None):
        from contracts.knowledge_classification import RuntimeContextDTO
        from contracts.ontology_building import ArtifactPinV1
        from services.normative_semantic_production_service import NormativeSemanticProductionService
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from services.validation_contract_review_service import ValidationContractReviewService
        from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path

        if not isinstance(store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("source validation requires production owners")
        actor = RuntimeContextDTO(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, actor_id=ctx.user_id)
        authority_resolver.require_current_actor(actor_context=actor)
        if not ctx.authenticated or store._path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve():
            raise ValueError("SOURCE_VALIDATION_WORKSPACE_OWNER_MISMATCH")
        if executed_at.utcoffset() != timedelta(0):
            raise ValueError("source validation requires UTC time")
        scope = dict(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        from services.ontology_quality_profile_service import QualityProfileService, tender_mvl0_quality_profile
        profile = tender_mvl0_quality_profile(**scope)
        applicability_review = None
        if profile_pin is not None:
            profile, applicability_review = QualityProfileService.read_persisted(store=store, **scope, profile_pin=profile_pin)
            if applicability_review is not None and applicability_review.reviewed_at > executed_at:
                raise ValueError('QUALITY_APPLICABILITY_REVIEW_POSTDATES_EVALUATION')
        code_units = self._code_units()
        environment = dict(python=sys.version, **{name: version(name) for name in ("pydantic", "pyshacl", "rdflib")})
        request_hash = "sha256:" + content_hash(dict(candidate=candidate_pin.model_dump(mode="json"),
            review=review_pin.model_dump(mode="json"), authority=authority_pin.model_dump(mode="json") if authority_pin else None,
            profile=profile.pin().model_dump(mode="json"), code={unit.code_ref: unit.code_hash for unit in code_units},
            environment=environment))
        operation = dict(**scope, actor_ref="user:" + ctx.user_id, operation="VALIDATE_SOURCE_SEMANTICS",
            request_id=request_id, request_hash=request_hash)
        prior = store.get_operation_result(**operation, artifact_kind="source_validation_execution")
        if prior is not None:
            return self.read_source_execution(store=store, **scope, execution_pin=prior)
        candidate = NormativeSemanticProductionService.read_persisted(store=store, **scope, candidate_pin=candidate_pin)
        review, validation = ValidationContractReviewService.read_source_decision_persisted(
            store=store, **scope, decision_pin=review_pin)
        if validation is None or review.decision != "accepted":
            raise ValueError("SOURCE_VALIDATION_ACCEPTED_REVIEW_REQUIRED")
        self._same_source(candidate, validation)
        QualityProfileService.verify_evaluation_basis(profile=profile, review=applicability_review,
            candidate=candidate, validation=validation)
        from contracts.normative_semantics import NormativeSemanticCandidateSetV1
        from contracts.ontology_certification import NormativeAuthorityDependencySetV1
        authority = predecessor = None
        if authority_pin is not None:
            if authority_pin.artifact_kind != 'normative_authority_dependency_set':
                raise ValueError('SOURCE_VALIDATION_AUTHORITY_KIND_MISMATCH')
            authority = store.get(**scope, artifact_ref=authority_pin.artifact_ref, contract_type=NormativeAuthorityDependencySetV1)
            if authority is None or authority.set_hash != authority_pin.artifact_hash:
                raise ValueError('SOURCE_VALIDATION_AUTHORITY_PIN_MISMATCH')
        if candidate.predecessor_candidate_set_ref is not None:
            predecessor = store.get(**scope, artifact_ref=candidate.predecessor_candidate_set_ref,
                contract_type=NormativeSemanticCandidateSetV1)
            if predecessor is not None and predecessor.candidate_set_hash != candidate.predecessor_candidate_set_hash:
                raise ValueError('SOURCE_VALIDATION_PREDECESSOR_PIN_MISMATCH')
        receipt, artifacts = self._compute_source_run(candidate=candidate, review=review, validation=validation,
            profile=profile, code_units=code_units, environment=environment, executed_at=executed_at,
            authority=authority, predecessor=predecessor, applicability_review=applicability_review)
        winner = store.put_many_with_operation_result(**operation, artifacts=(*artifacts, receipt),
            result_pin=ArtifactPinV1(artifact_kind="source_validation_execution",
                artifact_ref=receipt.execution_ref, artifact_hash=receipt.execution_hash),
            verify_before_commit=lambda: authority_resolver.require_current_actor(actor_context=actor))
        return self.read_source_execution(store=store, **scope, execution_pin=winner)

    @staticmethod
    def _same_source(candidate, validation):
        for name in ("organization_id", "workspace_id", "source_identity_ref", "source_hash",
                     "normative_schema_ref", "normative_schema_hash", "competency_contract_ref", "competency_contract_hash"):
            if getattr(candidate, name) != getattr(validation, name):
                raise ValueError("SOURCE_VALIDATION_INPUT_MISMATCH:" + name)

    def _compute_source_run(self, *, candidate, review, validation, profile, code_units, environment, executed_at,
                            authority=None, predecessor=None, applicability_review=None):
        from contracts.ontology_paired_evaluation import ArtifactPayloadPinV1
        from contracts.source_validation_execution import SourceValidationExecutionV1
        from services.ontology_quality_evaluators import FitnessQualityEvaluator, LogicalQualityEvaluator, _result, candidate_pin
        from services.ontology_rule_plan_compiler import OntologyRulePlanCompiler
        from services.ontology_structural_validator import ConstraintQualityEvaluator

        compiler_hash = next(unit.code_hash for unit in code_units
                             if unit.code_ref == "services/ontology_rule_plan_compiler.py")
        plan = None
        compilation_error = None
        bundles = ()
        pin = ArtifactPayloadPinV1.of
        try:
            plan = OntologyRulePlanCompiler(compiler_ref="compiler:ontology-rule-plan:v1", compiler_hash=compiler_hash).compile(
                candidate=candidate, validation_contract_ref=validation.validation_contract_ref,
                validation_contract_hash=validation.validation_contract_hash,
                competency_refs=tuple(sorted({test.competency_id for test in validation.executable_tests})),
                compiled_at=executed_at)
        except ValueError as exc:
            compilation_error = str(exc)
            run = self.assemble(profile=profile, subject=candidate_pin(candidate),
                competency=ArtifactPinV1(artifact_kind="competency_contract",
                    artifact_ref=candidate.competency_contract_ref, artifact_hash=candidate.competency_contract_hash),
                validation=ArtifactPinV1(artifact_kind="validation_contract",
                    artifact_ref=validation.validation_contract_ref, artifact_hash=validation.validation_contract_hash),
                results=(LogicalQualityEvaluator().evaluate(candidate=candidate, profile=profile),
                         ConstraintQualityEvaluator().evaluate(candidate, profile=profile),
                         *FitnessQualityEvaluator().evaluate(candidate=candidate, validation=validation, plan=None,
                             profile=profile, evaluated_at=executed_at)[0],
                         *(_result(profile=profile, candidate=candidate, requirement_id=item.requirement_id,
                             unavailable="plan_compilation_rejected:" + compilation_error)
                           for item in profile.requirements if item.gate not in {"LOGIC", "CONSTRAINT", "FITNESS"})),
                executed_at=executed_at)
        else:
            run, bundles = self.evaluate(candidate=candidate, validation=validation, plan=plan, profile=profile,
                executed_at=executed_at, authority=authority, predecessor=predecessor, applicability_review=applicability_review,
                retained_inputs=(pin(candidate, "candidate_set_ref", "candidate_set_hash"),
                    pin(validation, "validation_contract_ref", "validation_contract_hash"), pin(plan, "plan_ref", "plan_hash")))
        receipt = SourceValidationExecutionV1.create(organization_id=candidate.organization_id, workspace_id=candidate.workspace_id,
            candidate_pin=pin(candidate, "candidate_set_ref", "candidate_set_hash"),
            review_pin=pin(review, "decision_ref", "decision_hash"),
            validation_pin=pin(validation, "validation_contract_ref", "validation_contract_hash"),
            profile_pin=pin(profile, "profile_ref", "profile_hash"),
            authority_pin=pin(authority, "set_ref", "set_hash") if authority else None,
            predecessor_pin=pin(predecessor, "candidate_set_ref", "candidate_set_hash") if predecessor else None,
            run_pin=pin(run, "run_ref", "run_hash"), plan_pin=pin(plan, "plan_ref", "plan_hash") if plan else None,
            evaluation_pins=tuple(pin(bundle, "bundle_ref", "bundle_hash") for bundle in bundles),
            compilation_error=compilation_error,
            code_units=code_units, environment=environment, executed_at=executed_at)
        return receipt, (profile, run, *((plan,) if plan else ()), *bundles)

    @classmethod
    def read_source_execution(cls, *, store, organization_id, workspace_id, execution_pin):
        """Verify retained bytes and links; never run the current evaluator during replay."""
        from contracts.normative_semantics import NormativeSemanticCandidateSetV1
        from contracts.ontology_paired_evaluation import ArtifactPayloadPinV1
        from contracts.ontology_validation import CompiledRulePlanV1, RuntimeSemanticEvaluationBundleV1
        from contracts.source_validation_execution import SourceValidationExecutionV1
        from contracts.validation_contract import SourceValidationReviewDecisionV1, ValidationContractV1
        from services.ontology_quality_evaluators import FitnessQualityEvaluator, candidate_pin

        scope = dict(organization_id=organization_id, workspace_id=workspace_id)
        if execution_pin.artifact_kind != "source_validation_execution":
            raise ValueError("SOURCE_VALIDATION_EXECUTION_KIND_MISMATCH")
        receipt = store.get(**scope, artifact_ref=execution_pin.artifact_ref, contract_type=SourceValidationExecutionV1)
        if receipt is None or receipt.execution_hash != execution_pin.artifact_hash:
            raise ValueError("SOURCE_VALIDATION_EXECUTION_PIN_MISMATCH")

        def load(pin, model, ref_field, hash_field):
            artifact = store.get(**scope, artifact_ref=pin.artifact_ref, contract_type=model)
            if artifact is None or ArtifactPayloadPinV1.of(artifact, ref_field, hash_field) != pin:
                raise ValueError("SOURCE_VALIDATION_OUTPUT_PIN_MISMATCH")
            return artifact

        candidate = load(receipt.candidate_pin, NormativeSemanticCandidateSetV1, "candidate_set_ref", "candidate_set_hash")
        review = load(receipt.review_pin, SourceValidationReviewDecisionV1, "decision_ref", "decision_hash")
        validation = load(receipt.validation_pin, ValidationContractV1, "validation_contract_ref", "validation_contract_hash")
        profile = load(receipt.profile_pin, QualityProfileV1, "profile_ref", "profile_hash")
        applicability_review = None
        if profile.applicability_review is not None:
            from services.ontology_quality_profile_service import QualityProfileService
            _, applicability_review = QualityProfileService.read_persisted(store=store, **scope, profile_pin=profile.pin())
            QualityProfileService.verify_evaluation_basis(profile=profile, review=applicability_review,
                candidate=candidate, validation=validation)
            if applicability_review.reviewed_at > receipt.executed_at:
                raise ValueError('QUALITY_APPLICABILITY_REVIEW_POSTDATES_EVALUATION')
        authority = None
        if receipt.authority_pin is not None:
            from contracts.ontology_certification import NormativeAuthorityDependencySetV1
            authority = load(receipt.authority_pin, NormativeAuthorityDependencySetV1, 'set_ref', 'set_hash')
        if receipt.predecessor_pin is not None:
            predecessor = load(receipt.predecessor_pin, NormativeSemanticCandidateSetV1, 'candidate_set_ref', 'candidate_set_hash')
            if (candidate.predecessor_candidate_set_ref, candidate.predecessor_candidate_set_hash) != (predecessor.candidate_set_ref, predecessor.candidate_set_hash):
                raise ValueError('SOURCE_VALIDATION_PREDECESSOR_PIN_MISMATCH')
        run = load(receipt.run_pin, QualityEvaluationRunV1, "run_ref", "run_hash")
        cls._same_source(candidate, validation)
        run.require_basis(quality_profile=profile.pin(), subject=candidate_pin(candidate),
            competency=ArtifactPinV1(artifact_kind="competency_contract", artifact_ref=candidate.competency_contract_ref,
                artifact_hash=candidate.competency_contract_hash),
            validation=ArtifactPinV1(artifact_kind="validation_contract", artifact_ref=validation.validation_contract_ref,
                artifact_hash=validation.validation_contract_hash))
        if (review.decision != "accepted" or validation.review_state != "accepted"
            or (validation.human_oracle_ref, validation.human_oracle_hash) != (review.decision_ref, review.decision_hash)
            or run.executed_at != receipt.executed_at):
            raise ValueError("SOURCE_VALIDATION_RETAINED_INPUT_MISMATCH")
        if receipt.plan_pin is None:
            if run.disposition not in {"NOT_QUALIFIED", "NOT_EVALUABLE"}:
                raise ValueError("SOURCE_VALIDATION_MISSING_PLAN")
            bundles = ()
        else:
            plan = load(receipt.plan_pin, CompiledRulePlanV1, "plan_ref", "plan_hash")
            if ((plan.candidate_set_ref, plan.candidate_set_hash) != (candidate.candidate_set_ref, candidate.candidate_set_hash)
                or (plan.validation_contract_ref, plan.validation_contract_hash) != (
                    validation.validation_contract_ref, validation.validation_contract_hash)
                or plan.compiled_at != receipt.executed_at):
                raise ValueError("SOURCE_VALIDATION_RETAINED_PLAN_MISMATCH")
            bundles = tuple(load(pin, RuntimeSemanticEvaluationBundleV1, "bundle_ref", "bundle_hash")
                for pin in receipt.evaluation_pins)
            if bundles:
                FitnessQualityEvaluator.verify_retained_cases(plan=plan, validation=validation,
                    bundles=bundles, evaluated_at=receipt.executed_at)
                if applicability_review is not None:
                    precedence_result = next(r for r in run.results if r.requirement_id == 'precedence_resolution')
                    if precedence_result.status == 'PASS':
                        FitnessQualityEvaluator.verify_precedence_traces(candidate=candidate, validation=validation,
                            plan=plan, bundles=bundles, expectations=applicability_review.precedence_expectations)
        normative = tuple(a for result in run.results for a in result.normative_evidence)
        if normative:
            if receipt.plan_pin is None:
                raise ValueError("SOURCE_VALIDATION_NORMATIVE_PLAN_REQUIRED")
            FitnessQualityEvaluator.verify_normative_evidence(plan=plan, bundles=bundles, artifacts=normative)
        retained_bundles = {(bundle.bundle_ref, bundle.bundle_hash) for bundle in bundles}
        cited_bundles = {(pin.artifact_ref, pin.artifact_hash) for result in run.results for pin in result.evidence
                         if pin.artifact_kind == "runtime_semantic_evaluation_bundle"}
        if cited_bundles != retained_bundles:
            raise ValueError("SOURCE_VALIDATION_EVALUATION_EVIDENCE_MISMATCH")
        for result in run.results:
            for evidence in result.evidence:
                if evidence.artifact_kind == 'quality_profile_applicability_review' and (
                    applicability_review is None or evidence != applicability_review.pin()
                ):
                    raise ValueError('QUALITY_APPLICABILITY_RESULT_EVIDENCE_MISMATCH')
                if evidence.artifact_kind == 'normative_authority_dependency_set' and (authority is None or
                    (evidence.artifact_ref, evidence.artifact_hash) != (authority.set_ref, authority.set_hash)):
                    raise ValueError('SOURCE_VALIDATION_AUTHORITY_EVIDENCE_MISMATCH')
                if evidence.artifact_kind == "compiled_rule_plan" and (receipt.plan_pin is None or
                    (evidence.artifact_ref, evidence.artifact_hash) != (receipt.plan_pin.artifact_ref, receipt.plan_pin.artifact_hash)):
                    raise ValueError("SOURCE_VALIDATION_PLAN_EVIDENCE_MISMATCH")
        return receipt, run
