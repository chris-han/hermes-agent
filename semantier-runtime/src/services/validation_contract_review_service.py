"""Governed append-only review of independent validation obligations."""

from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path

from contracts.ontology_building import ArtifactPinV1
from contracts.validation_contract import SourceValidationCaseReviewV1, SourceValidationReviewDecisionV1, ValidationContractV1
from eos.artifact_hashing import content_hash
from services.external_validation_compiler import ExternalValidationCompiler
from services.source_validation_case_production_service import SourceValidationCaseProductionService
from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path


class ValidationContractReviewService:
    def review_case_outcome_persisted(self, *, ctx, authority_resolver, store, proposal_pin,
                                      case_id, expected_result, rationale, request_id, decided_at):
        """Preserve an explicit single-case decision independently of whole-set review."""
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.runtime_context_transition_service import RuntimeAuthorityResolver

        if not isinstance(store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("source case review requires production owners")
        actor = RuntimeContextDTO(organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id, actor_id=ctx.user_id)
        authority_resolver.require_current_actor(actor_context=actor)
        if not ctx.authenticated or store._path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve():
            raise ValueError("VALIDATION_REVIEW_WORKSPACE_OWNER_MISMATCH")
        if decided_at.utcoffset() != timedelta(0):
            raise ValueError("validation review requires UTC time")
        if expected_result not in {"MATCH", "NO_MATCH", "UNKNOWN"} or not rationale.strip():
            raise ValueError("explicit case outcome and rationale required")
        scope = dict(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        request_hash = "sha256:" + content_hash(dict(proposal=proposal_pin.model_dump(mode="json"),
            case_id=case_id, expected_result=expected_result, rationale=rationale))
        operation = dict(**scope, actor_ref="user:" + ctx.user_id, operation="REVIEW_SOURCE_VALIDATION_CASE_OUTCOME",
            request_id=request_id, request_hash=request_hash)
        prior = store.get_operation_result(**operation, artifact_kind="source_validation_case_review")
        if prior is not None:
            return self.read_case_outcome_persisted(store=store, **scope, review_pin=prior)
        receipt, proposal = SourceValidationCaseProductionService.read_persisted(
            store=store, **scope, proposal_pin=proposal_pin)
        case = next((case for case in proposal.cases if case.case_id == case_id), None)
        if case is None:
            raise ValueError("VALIDATION_REVIEW_CASE_NOT_FOUND")
        basis = dict(schema_version="source_validation_case_review.v1", **scope,
            actor_ref=operation["actor_ref"], proposal_ref=receipt.evidence_ref, proposal_hash=receipt.evidence_hash,
            case_id=case_id, original_case_hash="sha256:" + content_hash(case.model_dump(mode="json")),
            expected_result=expected_result, rationale=rationale,
            decided_at=decided_at.isoformat().replace("+00:00", "Z"))
        digest = content_hash(basis)
        review = SourceValidationCaseReviewV1(review_ref="source-validation-case-review:" + digest,
            review_hash="sha256:" + digest, **basis)
        authority_resolver.require_current_actor(actor_context=actor)
        pin = store.put_many_with_operation_result(**operation, artifacts=(review,),
            result_pin=ArtifactPinV1(artifact_kind="source_validation_case_review",
                artifact_ref=review.review_ref, artifact_hash=review.review_hash))
        return self.read_case_outcome_persisted(store=store, **scope, review_pin=pin)

    @classmethod
    def read_case_outcome_persisted(cls, *, store, organization_id, workspace_id, review_pin):
        scope = dict(organization_id=organization_id, workspace_id=workspace_id)
        if review_pin.artifact_kind != "source_validation_case_review":
            raise ValueError("VALIDATION_REVIEW_INPUT_KIND_MISMATCH")
        review = store.get(**scope, artifact_ref=review_pin.artifact_ref, contract_type=SourceValidationCaseReviewV1)
        if review is None or review.review_hash != review_pin.artifact_hash:
            raise ValueError("VALIDATION_REVIEW_PIN_MISMATCH")
        _, proposal = SourceValidationCaseProductionService.read_persisted(store=store, **scope,
            proposal_pin=ArtifactPinV1(artifact_kind="source_validation_proposal_evidence",
                artifact_ref=review.proposal_ref, artifact_hash=review.proposal_hash))
        case = next((case for case in proposal.cases if case.case_id == review.case_id), None)
        if case is None or "sha256:" + content_hash(case.model_dump(mode="json")) != review.original_case_hash:
            raise ValueError("VALIDATION_REVIEW_CASE_MISMATCH")
        return review, case.model_copy(update={"expected_result": review.expected_result, "rationale": review.rationale})

    def decide_source_cases_persisted(self, *, ctx, authority_resolver, store, proposal_pin,
                                      decision, rationale, request_id, decided_at, case_review_pins=(), case_rule_expectations=None, case_result_rule_refs=None, source_requirement_inventory=None):
        """Publish an explicit review command, never infer review from generation."""
        from contracts.knowledge_classification import RuntimeContextDTO
        from services.runtime_context_transition_service import RuntimeAuthorityResolver

        if not isinstance(store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("source case review requires production owners")
        actor = RuntimeContextDTO(organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id, actor_id=ctx.user_id)
        authority_resolver.require_current_actor(actor_context=actor)
        if not ctx.authenticated or store._path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve():
            raise ValueError("VALIDATION_REVIEW_WORKSPACE_OWNER_MISMATCH")
        if decided_at.utcoffset() != timedelta(0):
            raise ValueError("validation review requires UTC time")
        if decision not in {"accepted", "rejected"} or not rationale.strip():
            raise ValueError("explicit validation review decision and rationale required")
        scope = dict(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        compiler_ref = "compiler:reviewed-source-validation:v1"
        compiler_hash = "sha256:" + content_hash(dict(
            review=sha256(Path(__file__).read_bytes()).hexdigest(),
            compiler=sha256(Path(__file__).with_name("external_validation_compiler.py").read_bytes()).hexdigest()))
        selected = tuple(ArtifactPinV1.model_validate(pin) for pin in case_review_pins)
        case_basis = {"case_review_pins": [pin.model_dump(mode="json") for pin in selected]} if selected else {}
        if case_rule_expectations is not None:
            from pydantic import TypeAdapter
            from contracts.validation_contract import CaseRuleExpectations
            case_basis["case_rule_expectations"] = TypeAdapter(CaseRuleExpectations).validate_python(case_rule_expectations)
        if case_result_rule_refs is not None:
            from pydantic import TypeAdapter
            from contracts.validation_contract import CaseResultRuleRefs
            case_basis["case_result_rule_refs"] = TypeAdapter(CaseResultRuleRefs).validate_python(case_result_rule_refs)
        if source_requirement_inventory is not None:
            from contracts.validation_contract import SourceRequirementInventoryV1
            case_basis["source_requirement_inventory"] = SourceRequirementInventoryV1.model_validate(
                source_requirement_inventory).model_dump(mode="json")
        request_hash = "sha256:" + content_hash(dict(proposal=proposal_pin.model_dump(mode="json"),
            decision=decision, rationale=rationale, compiler_ref=compiler_ref, compiler_hash=compiler_hash, **case_basis))
        operation = dict(**scope, actor_ref="user:" + ctx.user_id, operation="REVIEW_SOURCE_VALIDATION_CASES",
            request_id=request_id, request_hash=request_hash)
        prior = store.get_operation_result(**operation, artifact_kind="source_validation_review_decision")
        if prior is not None:
            return self.read_source_decision_persisted(store=store, **scope, decision_pin=prior)
        receipt, proposal = SourceValidationCaseProductionService.read_persisted(
            store=store, **scope, proposal_pin=proposal_pin)
        basis = dict(schema_version="source_validation_review_decision.v1", **scope,
            actor_ref=operation["actor_ref"], proposal_ref=receipt.evidence_ref, proposal_hash=receipt.evidence_hash,
            decision=decision, rationale=rationale, compiler_ref=compiler_ref, compiler_hash=compiler_hash, **case_basis,
            decided_at=decided_at.isoformat().replace("+00:00", "Z"))
        digest = content_hash(basis)
        review = SourceValidationReviewDecisionV1(decision_ref="source-validation-review:" + digest,
            decision_hash="sha256:" + digest, **basis)
        proposal = self._apply_case_reviews(store=store, review=review, proposal=proposal)
        contracts = self._reviewed_contracts(review, receipt, proposal)
        authority_resolver.require_current_actor(actor_context=actor)
        pin = store.put_many_with_operation_result(**operation, artifacts=(review, *contracts),
            result_pin=ArtifactPinV1(artifact_kind="source_validation_review_decision",
                artifact_ref=review.decision_ref, artifact_hash=review.decision_hash))
        return self.read_source_decision_persisted(store=store, **scope, decision_pin=pin)

    @classmethod
    def _apply_case_reviews(cls, *, store, review, proposal):
        corrected = {}
        for pin in review.case_review_pins:
            individual, case = cls.read_case_outcome_persisted(store=store,
                organization_id=review.organization_id, workspace_id=review.workspace_id, review_pin=pin)
            if (individual.proposal_ref, individual.proposal_hash) != (review.proposal_ref, review.proposal_hash):
                raise ValueError("VALIDATION_REVIEW_CASE_PROPOSAL_MISMATCH")
            if case.case_id in corrected:
                raise ValueError("VALIDATION_REVIEW_DUPLICATE_CASE_DECISION")
            corrected[case.case_id] = case
        # Individual outcomes never resolve competency coverage or approve other cases.
        return proposal.model_copy(update={"cases": tuple(corrected.get(case.case_id, case) for case in proposal.cases)})

    @classmethod
    def _reviewed_contracts(cls, review, receipt, proposal):
        if review.decision == "rejected":
            return ()
        request = json.loads(receipt.input_text)
        source, schema = request["source_context"], request["normative_schema"]
        critical = {question["competency_id"] for question in source["competency"]["questions"]
            if question["criticality"] == "critical"}
        if critical.intersection(proposal.unresolved_competency_ids) or not proposal.cases:
            raise ValueError("VALIDATION_REVIEW_CRITICAL_CASES_UNRESOLVED")
        expectations = review.case_rule_expectations
        if expectations is not None and set(expectations) != {case.case_id for case in proposal.cases}:
            raise ValueError("VALIDATION_REVIEW_CASE_RULE_COVERAGE_MISMATCH")
        targets = review.case_result_rule_refs
        if targets is not None and (expectations is None or set(targets) != set(expectations)
            or any(target not in expectations[case] for case, target in targets.items())):
            raise ValueError("VALIDATION_REVIEW_RESULT_RULE_BINDING_MISMATCH")
        dimensions = sorted({dimension for case in proposal.cases for dimension in case.required_semantic_dimensions})
        compiled_source = dict(semantic_dimensions=[dict(dimension=dimension, critical=True,
            evidence_refs=sorted({ref for case in proposal.cases if dimension in case.required_semantic_dimensions
                for ref in case.evidence_refs})) for dimension in dimensions],
            competency_oracle=[{**case.model_dump(mode="json"), "source_case_id": case.case_id,
                **({"expected_rule_results": expectations[case.case_id]} if expectations is not None else {}),
                **({"result_rule_ref": targets[case.case_id]} if targets is not None else {})}
                for case in proposal.cases])
        if review.source_requirement_inventory is not None:
            from services.normative_schema_production_service import NormativeSchemaProductionService
            inventory = review.source_requirement_inventory
            evidence = {item["evidenceRef"]: item["provenance"]["source_block_id"]
                for item in NormativeSchemaProductionService.source_evidence(source)}
            facts = {item["fact_ref"] for item in schema.get("fact_definitions") or ()}
            questions = {q["competency_id"]: q["question"] for q in source["competency"]["questions"]
                if q["review_state"] == "confirmed"}
            if any(questions.get(ref) != question for ref, question in inventory.competency_questions.items()):
                raise ValueError("VALIDATION_REVIEW_INVENTORY_COMPETENCY_MISMATCH")
            for item in inventory.instances:
                if any(ref not in evidence or evidence[ref] != item.occurrence_ref for ref in item.evidence_refs):
                    raise ValueError("VALIDATION_REVIEW_INVENTORY_SOURCE_BINDING_MISMATCH")
                if any(p.fact_ref not in facts or (p.comparison_fact_ref is not None and p.comparison_fact_ref not in facts)
                    for bundle in item.support_alternatives for p in bundle):
                    raise ValueError("VALIDATION_REVIEW_INVENTORY_FACT_BINDING_MISMATCH")
            compiled_source["source_requirement_inventory"] = inventory.model_dump(mode="json")
        boundary = source["boundary"]
        compiler = ExternalValidationCompiler(compiler_ref=review.compiler_ref, compiler_hash=review.compiler_hash,
            run_ref=receipt.evidence_ref, prompt_hash="sha256:" + sha256(receipt.instructions.encode()).hexdigest(),
            provider_ref=receipt.provider, model_ref=receipt.model,
            config_hash="sha256:" + content_hash(receipt.output_schema), compiled_at=review.decided_at,
            source_region_coverage_ref=receipt.evidence_ref)
        proposed = compiler.compile(source=compiled_source, organization_id=review.organization_id,
            workspace_id=review.workspace_id, source_identity_ref=boundary["source_identity_ref"],
            source_hash=boundary["source_hash"], boundary_ref=boundary["boundary_ref"], boundary_hash=boundary["boundary_hash"],
            competency_contract_ref=schema["competency_contract_ref"], competency_contract_hash=schema["competency_contract_hash"],
            normative_schema_ref=schema["schema_ref"], normative_schema_hash=schema["schema_hash"],
            human_oracle_ref=review.decision_ref, human_oracle_hash=review.decision_hash)
        accepted = cls().decide(predecessor=proposed, decision="accepted", decision_ref=review.decision_ref,
            actor_authorized=True, decided_at=review.decided_at)
        return proposed, accepted

    @classmethod
    def read_source_decision_persisted(cls, *, store, organization_id, workspace_id, decision_pin):
        scope = dict(organization_id=organization_id, workspace_id=workspace_id)
        if decision_pin.artifact_kind != "source_validation_review_decision":
            raise ValueError("VALIDATION_REVIEW_INPUT_KIND_MISMATCH")
        review = store.get(**scope, artifact_ref=decision_pin.artifact_ref, contract_type=SourceValidationReviewDecisionV1)
        if review is None or review.decision_hash != decision_pin.artifact_hash:
            raise ValueError("VALIDATION_REVIEW_PIN_MISMATCH")
        receipt, proposal = SourceValidationCaseProductionService.read_persisted(store=store, **scope,
            proposal_pin=ArtifactPinV1(artifact_kind="source_validation_proposal_evidence",
                artifact_ref=review.proposal_ref, artifact_hash=review.proposal_hash))
        proposal = cls._apply_case_reviews(store=store, review=review, proposal=proposal)
        contracts = cls._reviewed_contracts(review, receipt, proposal)
        for expected in contracts:
            actual = store.get(**scope, artifact_ref=expected.validation_contract_ref, contract_type=ValidationContractV1)
            if actual != expected:
                raise ValueError("VALIDATION_REVIEW_REPLAY_MISMATCH")
        return review, contracts[-1] if contracts else None

    @classmethod
    def verify_source_contract(cls, *, store, validation):
        _, expected = cls.read_source_decision_persisted(store=store,
            organization_id=validation.organization_id, workspace_id=validation.workspace_id,
            decision_pin=ArtifactPinV1(artifact_kind="source_validation_review_decision",
                artifact_ref=validation.human_oracle_ref, artifact_hash=validation.human_oracle_hash))
        if expected is None or validation != expected:
            raise ValueError("VALIDATION_REVIEW_CONTRACT_MISMATCH")

    def decide(
        self,
        *,
        predecessor: ValidationContractV1,
        decision: str,
        decision_ref: str,
        actor_authorized: bool,
        decided_at: datetime,
    ) -> ValidationContractV1:
        if predecessor.review_state != "proposed":
            raise ValueError("review decisions apply only to a proposed validation contract")
        if decision not in {"accepted", "rejected"}:
            raise ValueError("validation review decision must be accepted or rejected")
        payload = predecessor.model_dump(mode="json")
        basis = {
            **payload,
            "validation_contract_ref": None,
            "validation_contract_hash": None,
            "review_state": decision,
            "predecessor_validation_contract_ref": predecessor.validation_contract_ref,
            "predecessor_validation_contract_hash": predecessor.validation_contract_hash,
            "review_decision_ref": decision_ref,
            "compiled_at": decided_at.isoformat(),
        }
        digest = content_hash(basis)
        basis["validation_contract_ref"] = f"validation-contract:{digest}"
        basis["validation_contract_hash"] = f"sha256:{digest}"
        return ValidationContractV1.model_validate(basis)


__all__ = ["ValidationContractReviewService"]
