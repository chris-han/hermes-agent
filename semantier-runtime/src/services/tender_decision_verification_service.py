"""Deterministic Tender verification over supplied frozen values only."""

from pydantic import BaseModel, ValidationError

from contracts.decision_local_projection import MaterialDependencyRequirementV1
from contracts.decision_local_projection_v2 import DecisionLocalProjectionV2
from contracts.decision_verification import (
    COMPOSITION_POLICY,
    CompositionNodeTraceV1,
    ComposedDecisionTraceV1,
    DecisionVerificationV1,
    InterpretedPredicateTraceV1,
    ConflictControllerBindingV1,
    ConflictMembershipReceiptV1,
    ConflictResolutionReceiptV1,
    PredicateResultReceiptV1,
    TenderCompositionSpecV1,
    VerifiedDecisionHandoffV1,
)
from contracts.ontology_building import ArtifactPinV1
from contracts.typed_decision_model import (
    DecisionDistributionV1,
    TypedDecisionFrontierV1,
    TypedDecisionModelInputV1,
)
from eos.artifact_hashing import content_hash
from services.decision_local_projection_service import DecisionLocalProjectionService
from services.tender_decision_local_projection_service import (
    TenderDecisionLocalProjectionService,
)


def _payload(value):
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def _pin(kind, ref, digest):
    return ArtifactPinV1(artifact_kind=kind, artifact_ref=ref, artifact_hash=digest)


def _unknown_dominant_all(values):
    values = tuple(values)
    if "UNKNOWN" in values:
        return "UNKNOWN"
    if "FALSE" in values:
        return "FALSE"
    return "TRUE"


def _decisive_false_all(values):
    values = tuple(values)
    if "FALSE" in values:
        return "FALSE"
    if "UNKNOWN" in values:
        return "UNKNOWN"
    return "TRUE"


def _true_dominant_any(values):
    values = tuple(values)
    if "TRUE" in values:
        return "TRUE"
    if "UNKNOWN" in values:
        return "UNKNOWN"
    return "FALSE"


class TenderDecisionVerificationService:
    """No provider, store, retrieval, execution or authority dependencies."""

    def verify(
        self, *, projection, frontier, model_input, distribution=None, epistemic_admissibility=None
    ) -> DecisionVerificationV1:
        raw = {
            name: _payload(value)
            for name, value in (
                ("projection", projection),
                ("frontier", frontier),
                ("model_input", model_input),
                ("distribution", distribution),
                ("epistemic_admissibility", epistemic_admissibility),
            )
        }
        basis_hash = "sha256:" + content_hash(raw)
        pins = {}

        def result(status, diagnostic, **values):
            return DecisionVerificationV1.create(
                supplied_basis_hash=basis_hash,
                status=status,
                diagnostics=(diagnostic,),
                **pins,
                **values,
            )

        try:
            projection = DecisionLocalProjectionV2.model_validate(raw["projection"])
        except (ValidationError, ValueError, KeyError, TypeError):
            return result("INVALID_BASIS", "INVALID_DLP")
        pins["projection"] = _pin(
            "decision_local_projection",
            projection.projection_ref,
            projection.projection_hash,
        )
        if (
            projection.decision_contract.purpose
            != "tender_restriction_semantic_predicate"
        ):
            return result("INVALID_BASIS", "TENDER_DECISION_CONTRACT_REQUIRED")
        if (
            projection.runtime_context.organization_id,
            projection.runtime_context.workspace_id,
        ) != (projection.legacy.organization_id, projection.legacy.workspace_id):
            return result("INVALID_BASIS", "TENDER_RUNTIME_AUTHORITY_MISMATCH")
        if projection.working_context.ontology != projection.context.ontology:
            return result("INVALID_BASIS", "WORKING_CONTEXT_ONTOLOGY_MISMATCH")
        if len(projection.decision_contract.semantic_predicates) != 1:
            return result(
                "COMPOSITION_UNAVAILABLE", "INTERPRETED_SINGLE_PREDICATE_ONLY"
            )
        # Reuse the compiler's authority-binding validation against the retained
        # bytes only. The resolver here cannot retrieve current/live rules.
        try:
            rules = {
                rule["rule_version_id"]: rule for rule in projection.governed_rules
            }
            if len(rules) != len(projection.governed_rules) or set(rules) != set(
                projection.runtime_context.ordered_governed_rule_version_refs
            ):
                return result("INVALID_BASIS", "FROZEN_RULE_CLOSURE_MISMATCH")
            constraints, semantic_type, family = (
                TenderDecisionLocalProjectionService.validate_runtime_binding(
                    runtime_context=projection.runtime_context,
                    retrieval_snapshot=projection.retrieval_snapshot,
                    canonical_pattern_ref=projection.canonical_pattern_ref,
                    governed_rule_resolver=rules.get,
                )
            )
            release = next(
                pin
                for pin in constraints
                if pin.artifact_kind == "accepted_graph_release"
            )
            if projection.context.world.source_graph != release:
                return result("INVALID_BASIS", "DECISION_SOURCE_GRAPH_MISMATCH")
            expected_contract = TenderDecisionLocalProjectionService.decision_contract(
                issue_type=semantic_type,
                family_id=family,
                deterministic_constraints=constraints,
                material_requirements=tuple(
                    MaterialDependencyRequirementV1(
                        dependency_id=d.dependency_id,
                        required_evidence_role=d.required_evidence_role,
                        min_count=d.min_count,
                    )
                    for d in projection.source_region.material_dependencies
                ),
            )
            if expected_contract != projection.decision_contract:
                return result("INVALID_BASIS", "FROZEN_DECISION_CONTRACT_MISMATCH")
        except (ValidationError, ValueError, KeyError, TypeError):
            return result("INVALID_BASIS", "FROZEN_RUNTIME_BINDING_MISMATCH")
        if epistemic_admissibility is not None:
            try:
                from contracts.decision_graph_projection import PredicateEpistemicAdmissibilityV1
                admissibility = PredicateEpistemicAdmissibilityV1.model_validate(epistemic_admissibility)
                predicate_id = projection.decision_contract.semantic_predicates[0].predicate_id
                if admissibility.predicate_id != predicate_id:
                    return result("BINDING_MISMATCH", "EPISTEMIC_ADMISSIBILITY_PREDICATE_MISMATCH")
                if admissibility.projection_sufficiency_proof.artifact_kind != "projection_sufficiency_proof":
                    return result("BINDING_MISMATCH", "EPISTEMIC_ADMISSIBILITY_PROOF_KIND_MISMATCH")
                if admissibility.status == "INADMISSIBLE":
                    if distribution is not None:
                        return result("BINDING_MISMATCH", "INADMISSIBLE_PREDICATE_MUST_NOT_HAVE_MODEL_EVIDENCE")
                    return result("STRUCTURALLY_RESOLVED", "STRUCTURAL_EPISTEMIC_INADMISSIBILITY", semantic_result="UNKNOWN")
            except (ValidationError, ValueError, KeyError, TypeError) as exc:
                return result("BINDING_MISMATCH", "INVALID_EPISTEMIC_ADMISSIBILITY:" + str(exc))
        if frontier is None or model_input is None:
            return result("INSUFFICIENT_BASIS", "MISSING_FRONTIER_OR_MODEL_INPUT")
        try:
            frontier = TypedDecisionFrontierV1.model_validate(raw["frontier"])
            frontier.verify_against_projection(projection)
            model_input = TypedDecisionModelInputV1.model_validate(raw["model_input"])
            if model_input != TypedDecisionModelInputV1.from_projection(
                projection, frontier
            ):
                return result("BINDING_MISMATCH", "MODEL_INPUT_DLP_MISMATCH")
        except (ValidationError, ValueError, KeyError, TypeError):
            return result("BINDING_MISMATCH", "FRONTIER_OR_MODEL_INPUT_MISMATCH")
        pins["frontier"] = _pin(
            "typed_decision_frontier", frontier.frontier_ref, frontier.frontier_hash
        )
        pins["model_input"] = _pin(
            "typed_decision_model_input", model_input.input_ref, model_input.input_hash
        )
        if len(frontier.questions) != 1:
            return result(
                "COMPOSITION_UNAVAILABLE", "INTERPRETED_SINGLE_PREDICATE_ONLY"
            )
        if distribution is None:
            return result("INSUFFICIENT_BASIS", "MISSING_MODEL_EVIDENCE")
        try:
            distribution = DecisionDistributionV1.model_validate(raw["distribution"])
            distribution.verify_against(model_input)
        except (ValidationError, ValueError, KeyError, TypeError):
            return result(
                "INVALID_MODEL_EVIDENCE",
                "MODEL_EVIDENCE_IDENTITY_OR_COVERAGE_MISMATCH",
            )
        pins["distribution"] = _pin(
            "decision_distribution",
            distribution.distribution_ref,
            distribution.distribution_hash,
        )
        question = frontier.questions[0]
        selected = distribution.question_results[0].selected_candidate_id
        proposed = next(
            c.semantic_value for c in question.candidates if c.candidate_id == selected
        )
        _, normalized = DecisionLocalProjectionService.evaluate_semantic_proposal(
            projection.legacy,
            producer_state="PRODUCER_OBSERVATION",
            proposed_semantic_result="SEMANTIC_" + proposed,
        )
        semantic = normalized.removeprefix("SEMANTIC_")
        missing = any(
            d.status == "MISSING"
            for d in projection.legacy.material_dependency_resolutions
        )
        diagnostic = (
            "MATERIAL_DEPENDENCY_MISSING"
            if missing
            else (
                "MODEL_SELECTED_UNKNOWN"
                if semantic == "UNKNOWN"
                else "PREDICATE_VERIFIED"
            )
        )
        trace = InterpretedPredicateTraceV1(
            predicate_id=question.predicate_id,
            selected_candidate_id=selected,
            proposed_semantic_result=proposed,
            material_dependencies=projection.legacy.material_dependency_resolutions,
            semantic_result=semantic,
            normalization="MATERIAL_DEPENDENCY_MISSING"
            if missing
            else "MODEL_SELECTION",
        )
        return result("VERIFIED", diagnostic, semantic_result=semantic, trace=trace)

    @staticmethod
    def receipt_from_interpreted(
        verification: DecisionVerificationV1,
    ) -> PredicateResultReceiptV1:
        verification = DecisionVerificationV1.model_validate(_payload(verification))
        if (
            verification.status != "VERIFIED"
            or verification.scope != "INTERPRETED_PREDICATE"
            or verification.trace is None
            or verification.semantic_result is None
        ):
            raise ValueError("VERIFIED_INTERPRETED_PREDICATE_REQUIRED")
        return PredicateResultReceiptV1.create(
            predicate_id=verification.trace.predicate_id,
            evaluation_mode="INTERPRETED",
            semantic_result=verification.semantic_result,
            basis=(verification.pin(),),
        )

    @staticmethod
    def conflict_membership_receipt(
        *, decision_scope, left_factor, right_factor, projection, membership
    ) -> ConflictMembershipReceiptV1:
        left = PredicateResultReceiptV1.model_validate(_payload(left_factor))
        right = PredicateResultReceiptV1.model_validate(_payload(right_factor))
        projection_pin = projection.pin() if hasattr(projection, "pin") else projection
        return ConflictMembershipReceiptV1.create(
            decision_scope=decision_scope,
            left_factor=left.pin(),
            right_factor=right.pin(),
            projection=projection_pin,
            membership=membership,
            basis=(left.pin(), right.pin(), projection_pin),
        )

    @staticmethod
    def resolve_conflict(
        *, conflict_membership, projection, controller=None
    ) -> ConflictResolutionReceiptV1:
        membership = ConflictMembershipReceiptV1.model_validate(_payload(conflict_membership))
        projection_pin = projection.pin() if hasattr(projection, "pin") else projection
        if membership.projection != projection_pin:
            raise ValueError("CONFLICT_MEMBERSHIP_PROJECTION_MISMATCH")
        binding = (
            None
            if controller is None
            else ConflictControllerBindingV1.model_validate(_payload(controller))
        )
        if membership.membership == "UNKNOWN":
            resolution = "UNKNOWN"
        elif membership.membership == "COMPATIBLE":
            resolution = "NO_RESOLUTION_NEEDED"
        elif binding is None:
            resolution = "UNRESOLVED_CONFLICT"
        else:
            resolution = "RESOLVED_TO_LEFT" if binding.winner == "LEFT" else "RESOLVED_TO_RIGHT"
        return ConflictResolutionReceiptV1.create(
            conflict_membership=membership.pin(),
            projection=projection_pin,
            controller=binding,
            resolution=resolution,
        )

    def verify_composition(
        self, *, composition_spec, predicate_receipts
    ) -> DecisionVerificationV1:
        raw_receipts = sorted(
            (_payload(item) for item in predicate_receipts),
            key=lambda item: item["predicate_id"],
        )
        raw = {
            "composition_policy": COMPOSITION_POLICY.model_dump(mode="json"),
            "composition_spec": _payload(composition_spec),
            "predicate_receipts": raw_receipts,
        }
        basis_hash = "sha256:" + content_hash(raw)

        def failed(status, diagnostic):
            return DecisionVerificationV1.create(
                scope="COMPOSED_PREDICATE_SET",
                supplied_basis_hash=basis_hash,
                status=status,
                diagnostics=(diagnostic,),
            )

        try:
            spec = TenderCompositionSpecV1.model_validate(raw["composition_spec"])
            receipts = tuple(
                PredicateResultReceiptV1.model_validate(item)
                for item in raw["predicate_receipts"]
            )
        except (ValidationError, ValueError, KeyError, TypeError):
            return failed("INVALID_BASIS", "INVALID_COMPOSITION_BASIS")
        by_id = {receipt.predicate_id: receipt for receipt in receipts}
        if len(by_id) != len(receipts):
            return failed("INVALID_BASIS", "DUPLICATE_PREDICATE_RECEIPT")
        if set(by_id) != spec.leaf_ids():
            return failed("BINDING_MISMATCH", "COMPOSITION_LEAF_COVERAGE_MISMATCH")

        input_values = tuple(
            by_id[predicate_id].semantic_result
            for predicate_id in spec.input_predicate_ids
        )
        ordinary_values = tuple(
            by_id[predicate_id].semantic_result
            for predicate_id in spec.ordinary_predicate_ids
        )
        input_result = _decisive_false_all(input_values)
        ordinary_result = _unknown_dominant_all(ordinary_values)
        nodes = [
            CompositionNodeTraceV1(
                node_id="input_availability",
                operator="INPUT_CONJUNCTION",
                inputs=input_values,
                semantic_result=input_result,
            ),
            CompositionNodeTraceV1(
                node_id="ordinary_conditions",
                operator="ALL_OF",
                inputs=ordinary_values,
                semantic_result=ordinary_result,
            ),
        ]
        exception_results = []
        for group in spec.exception_groups:
            values = tuple(
                by_id[predicate_id].semantic_result
                for predicate_id in group.predicate_ids
            )
            group_result = _decisive_false_all(values)
            exception_results.append(group_result)
            nodes.append(
                CompositionNodeTraceV1(
                    node_id=group.node_id,
                    operator="EXCEPTION_CONJUNCTION",
                    inputs=values,
                    semantic_result=group_result,
                )
            )
        exception_union = _true_dominant_any(exception_results)
        nodes.append(
            CompositionNodeTraceV1(
                node_id="exception_union",
                operator="ANY_OF",
                inputs=tuple(exception_results),
                semantic_result=exception_union,
            )
        )
        if input_result != "TRUE":
            final_result = "UNKNOWN"
        elif exception_union == "TRUE":
            final_result = "FALSE"
        elif "UNKNOWN" in (ordinary_result, exception_union):
            final_result = "UNKNOWN"
        elif ordinary_result == "FALSE":
            final_result = "FALSE"
        else:
            final_result = "TRUE"
        nodes.append(
            CompositionNodeTraceV1(
                node_id="final",
                operator="FINAL",
                inputs=(input_result, ordinary_result, exception_union),
                semantic_result=final_result,
            )
        )
        canonical_receipts = tuple(sorted(receipts, key=lambda item: item.predicate_id))
        trace = ComposedDecisionTraceV1(
            receipts=canonical_receipts,
            nodes=tuple(nodes),
            semantic_result=final_result,
        )
        diagnostic = (
            "COMPOSITION_VERIFIED_UNKNOWN"
            if final_result == "UNKNOWN"
            else "COMPOSITION_VERIFIED"
        )
        return DecisionVerificationV1.create(
            scope="COMPOSED_PREDICATE_SET",
            supplied_basis_hash=basis_hash,
            status="VERIFIED",
            semantic_result=final_result,
            composition_policy=COMPOSITION_POLICY,
            composition_spec=spec,
            composed_trace=trace,
            diagnostics=(diagnostic,),
        )

    @staticmethod
    def case_effectivity_gaps(*, facts, authority):
        """An unzoned calendar date cannot establish a narrower legal instant."""
        from datetime import date, datetime, time, timedelta, timezone
        value = facts.get("fact.document_date")
        try:
            if not isinstance(value, str):
                raise ValueError("date string required")
            observed = date.fromisoformat(value)
            if observed.isoformat() != value:
                raise ValueError("canonical date required")
        except ValueError:
            return ("CASE_DOCUMENT_DATE_REQUIRED",)
        if authority.as_of_time.date() != observed:
            return ("CASE_AUTHORITY_DATE_MISMATCH",)
        # The producer retains a calendar date, not an event timezone. Cover
        # adjacent UTC days rather than silently interpreting that date as UTC.
        # A narrower transition requires a separately governed temporal basis.
        try:
            start = datetime.combine(observed, time.min, tzinfo=timezone.utc) - timedelta(days=1)
            end = datetime.combine(observed, time.max, tzinfo=timezone.utc) + timedelta(days=1)
        except OverflowError:
            return ("CASE_DATE_EFFECTIVITY_UNRESOLVED",)
        if any(dependency.effective_from > start or (
            dependency.effective_to is not None and dependency.effective_to < end
        ) for dependency in authority.dependencies):
            return ("CASE_DATE_EFFECTIVITY_UNRESOLVED",)
        return ()

    def verify_legal_compliance(self, *, basis, episode, projection, hypothesis, binding_review=None):
        """Frozen computation; live owner/provenance checks belong to publication."""
        from pydantic import TypeAdapter
        from contracts.tender_legal_compliance import (
            TenderLegalComplianceBasisV1, TenderLegalComplianceVerificationV1,
        )
        from contracts.ontology_building import ArtifactPinV1
        from services.tender_review_service import TenderReviewService
        from services.deterministic_rule_evaluator import DeterministicRuleEvaluator
        from services.tender_legal_compliance_service import normative_input_pin
        policy = ArtifactPinV1(artifact_kind="compliance_disposition_policy",
            artifact_ref="source-rule-effect-disposition:v2",
            artifact_hash="sha256:" + content_hash({
                "version":"source_rule_effect_disposition.v2", "publication":"verified_source_fact_review_binding",
                "screening":"candidate_only", "absence":"UNKNOWN", "scope":"exact_rule_and_case",
                "effectivity":"observed_unzoned_calendar_date_conservative_adjacent_utc_days",
            }))
        basis = TenderLegalComplianceBasisV1.model_validate(basis.model_dump(mode="json"))
        def receipt(outcome, diagnostics, evaluation=None, status="VERIFIED"):
            return TenderLegalComplianceVerificationV1.create(basis=basis.pin(), policy=policy,
                status=status, outcome=outcome, diagnostics=diagnostics,
                projection=ArtifactPinV1(artifact_kind="decision_local_projection", artifact_ref=projection.projection_ref,
                                         artifact_hash=projection.projection_hash),
                hypothesis=ArtifactPinV1(artifact_kind="runtime_semantic_hypothesis", artifact_ref=hypothesis.hypothesis_id,
                                         artifact_hash=hypothesis.hypothesis_hash),
                evaluation=TypeAdapter(dict).dump_python(evaluation, mode="json") if evaluation is not None else None)
        result = episode.inspection_result
        if (basis.episode != TenderReviewService.episode_pin(episode)
            or (basis.inspection_result.artifact_ref,basis.inspection_result.artifact_hash) != (result.result_ref,result.result_hash)
            or basis.finding not in result.governed_findings
            or (basis.target_ir.artifact_ref,basis.target_ir.artifact_hash) != (result.canonical_source_ir_ref,result.canonical_source_ir_hash)):
            return receipt("UNKNOWN", ("EPISODE_BINDING_MISMATCH",), status="INVALID_BINDING")
        finding = basis.finding
        if ((hypothesis.hypothesis_id, hypothesis.hypothesis_hash) != (
                finding.observed_hypothesis_ref, finding.observed_hypothesis_hash)
            or (hypothesis.decision_projection_ref, hypothesis.decision_projection_hash) != (
                projection.projection_ref, projection.projection_hash)):
            return receipt("UNKNOWN", ("HYPOTHESIS_PROJECTION_BINDING_MISMATCH",), status="INVALID_BINDING")
        if (projection.canonical_pattern_ref != finding.canonical_pattern_ref
            or (projection.legacy.organization_id, projection.legacy.workspace_id) != (basis.organization_id,basis.workspace_id)
            or projection.source_region.source_binding.source_hash != finding.source_hash
            or projection.source_region.source_binding.source_identity_ref != finding.source_identity_ref):
            return receipt("UNKNOWN", ("DECISION_CONTEXT_BINDING_MISMATCH",), status="INVALID_BINDING")
        if basis.advisory is not None:
            a = basis.advisory
            policy = ArtifactPinV1(artifact_kind="compliance_disposition_policy",
                artifact_ref="owner-adopted-internal-advisory:v1",
                artifact_hash="sha256:" + content_hash({"version":"owner_adopted_internal_advisory.v1",
                    "adoption_policy":a.proposal.policy_hash, "effectivity":"documentary_unknown_is_unknown",
                    "scope":"exact_adopted_rule_and_located_case", "independent_legal_validation":"NOT_ESTABLISHED"}))
            from contracts.tender_owner_adoption import ontology_contract_hash
            if a.proposal.ontology_contract_hash != ontology_contract_hash(projection.ontology):
                return receipt("UNKNOWN", ("ADVISORY_ONTOLOGY_MISMATCH",), status="INVALID_BINDING")
            if a.projection is None or a.projection.subject_binding is None:
                return receipt("UNKNOWN", tuple(dict.fromkeys((*basis.gaps, "EXACT_CASE_SUBJECT_MISSING_OR_AMBIGUOUS"))))
            if (a.projection.subject_binding.context_graph != projection.context.pin()
                or not set(a.projection.target_evidence).issubset({e.pin() for e in projection.evidence})):
                return receipt("UNKNOWN", ("ADVISORY_CASE_CONTEXT_MISMATCH",), status="INVALID_BINDING")
            evaluated = DeterministicRuleEvaluator().evaluate_scoped_disposition(
                candidate=a.candidate, plan=a.plan, rule_id=a.proposal.rule_ref, observed_facts=a.facts.facts,
                context_pin=projection.context.pin(), execution_epoch_ref=result.result_ref,
                target_evidence_refs=a.facts.evidence_refs, evaluated_at=basis.frozen_at)
            # The conditional trace is useful internally but cannot establish
            # statutory effectivity or case facts that have not been observed.
            gaps = tuple(dict.fromkeys((*a.proposal.unknowns,*basis.gaps)))
            if gaps:
                return receipt("UNKNOWN", gaps, evaluated)
            return receipt(evaluated["proposed_outcome"], (evaluated["reason_code"],), evaluated)
        if basis.normative is None or basis.gaps:
            return receipt("UNKNOWN", basis.gaps or ("QUALIFIED_NORMATIVE_BINDING_MISSING",))
        n = basis.normative
        subject = ArtifactPinV1(artifact_kind="governed_frame_finding", artifact_ref=finding.finding_id, artifact_hash=finding.finding_hash)
        if (binding_review is None or binding_review.pin() != n.binding_review
            or binding_review.purpose != "REVIEW" or binding_review.result_code != "NORMATIVE_BINDING_ACCEPTED"
            or binding_review.subject != subject or binding_review.context != projection.context.pin()
            or binding_review.ontology != projection.ontology.pin()
            or normative_input_pin(n) not in tuple(pin for i in binding_review.reasoning.invocations for pin in i.inputs)):
            return receipt("UNKNOWN", ("EXACT_NORMATIVE_BINDING_REVIEW_REQUIRED",), status="INVALID_BINDING")
        temporal_gaps = self.case_effectivity_gaps(facts=n.facts.facts, authority=n.authority)
        if temporal_gaps:
            return receipt("UNKNOWN", temporal_gaps)
        evaluated = DeterministicRuleEvaluator().evaluate_scoped_disposition(
            candidate=n.candidate, plan=n.plan, rule_id=n.rule_id, observed_facts=n.facts.facts,
            context_pin=projection.context.pin(), execution_epoch_ref=result.result_ref,
            target_evidence_refs=n.facts.evidence_refs, evaluated_at=basis.frozen_at)
        return receipt(evaluated["proposed_outcome"], (evaluated["reason_code"],), evaluated)

    def handoff(self, *, verification, **basis) -> VerifiedDecisionHandoffV1:
        try:
            verification = DecisionVerificationV1.model_validate(_payload(verification))
        except (ValidationError, ValueError, KeyError, TypeError) as exc:
            raise ValueError("VERIFICATION_REPLAY_MISMATCH") from exc
        if verification.scope != "INTERPRETED_PREDICATE":
            raise ValueError("COMPOSED_HANDOFF_NOT_AUTHORIZED")
        replayed = self.verify(**basis)
        if verification != replayed:
            raise ValueError("VERIFICATION_REPLAY_MISMATCH")
        if replayed.status != "VERIFIED":
            raise ValueError("DECISION_NOT_VERIFIED")
        projection = DecisionLocalProjectionV2.model_validate(
            _payload(basis["projection"])
        )
        return VerifiedDecisionHandoffV1(
            organization_id=projection.legacy.organization_id,
            workspace_id=projection.legacy.workspace_id,
            verification=replayed,
            predicate_id=replayed.trace.predicate_id,
            semantic_result=replayed.semantic_result,
            subject=_pin(
                "runtime_semantic_source_region",
                projection.source_region.source_region_ref,
                "sha256:"
                + content_hash(projection.source_region.model_dump(mode="json")),
            ),
            context=projection.context.pin(),
            ontology=projection.ontology.pin(),
            working_context=projection.working_context.pin(),
            evidence=projection.legacy.evidence,
            diagnostics=replayed.diagnostics,
        )
