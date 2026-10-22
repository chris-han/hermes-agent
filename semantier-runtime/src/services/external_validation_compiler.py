"""Independent source/competency validation compiler.

The public compile signature intentionally cannot accept semantic candidates.
This makes correlated semantic/validation discovery visible instead of letting
one generated inventory define both the implementation and its oracle.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from contracts.validation_contract import (
    ExecutableCompetencyTestV1,
    SourceRequirementInventoryV1,
    ValidationContractV1,
    ValidationIndependenceV1,
    ValidationObligationV1,
)
from eos.artifact_hashing import content_hash


class ExternalValidationCompiler:
    def __init__(
        self,
        *,
        compiler_ref: str,
        compiler_hash: str,
        run_ref: str,
        prompt_hash: str,
        provider_ref: str,
        model_ref: str,
        config_hash: str,
        compiled_at: datetime,
        source_region_coverage_ref: str = "source-region-coverage:complete",
    ) -> None:
        self._independence = ValidationIndependenceV1(
            compiler_ref=compiler_ref,
            compiler_hash=compiler_hash,
            run_ref=run_ref,
            prompt_hash=prompt_hash,
            config_hash=config_hash,
            provider_ref=provider_ref,
            model_ref=model_ref,
            source_region_coverage_ref=source_region_coverage_ref,
        )
        self._compiled_at = compiled_at

    def compile(
        self,
        *,
        source: dict[str, Any],
        organization_id: str,
        workspace_id: str,
        source_identity_ref: str,
        source_hash: str,
        boundary_ref: str,
        boundary_hash: str,
        competency_contract_ref: str,
        competency_contract_hash: str,
        normative_schema_ref: str,
        normative_schema_hash: str,
        human_oracle_ref: str,
        human_oracle_hash: str,
    ) -> ValidationContractV1:
        dimensions = [
            item
            for item in source.get("semantic_dimensions", [])
            if isinstance(item, dict) and item.get("critical") is True
        ]
        cases = [
            item for item in source.get("competency_oracle", []) if isinstance(item, dict)
        ]
        inventory = source.get("source_requirement_inventory")
        if inventory is not None:
            inventory = SourceRequirementInventoryV1.model_validate(inventory)
            if (inventory.source_identity_ref, inventory.source_hash, inventory.boundary_ref, inventory.boundary_hash) != (
                source_identity_ref, source_hash, boundary_ref, boundary_hash):
                raise ValueError("source requirement inventory pin mismatch")
            dimensions = [dict(dimension=i.semantic_dimension, instance_id=i.instance_id,
                evidence_refs=i.evidence_refs, source_instance=i.model_dump(mode="json")) for i in inventory.instances]
        if (not dimensions and inventory is None) or not cases:
            raise ValueError("reviewed semantic dimensions and competency oracle are required")

        if inventory is not None and inventory.eligibility == "ELIGIBLE":
            for instance in inventory.instances:
                outcomes = {case.get("expected_instance_results", {}).get(instance.instance_id) for case in cases
                    if instance.instance_id in case.get("required_source_instance_ids", ())}
                if not {"MATCH", "NO_MATCH"}.issubset(outcomes):
                    raise ValueError("source requirement distinguishing cases missing: " + instance.instance_id)

        obligations: list[ValidationObligationV1] = []
        cross_cutting_dimensions: set[str] = set()
        for dimension in dimensions:
            name = str(dimension["dimension"])
            mapped = tuple(dict.fromkeys(
                str(case["competency_id"])
                for case in cases
                if (dimension["instance_id"] in case.get("required_source_instance_ids", []) if inventory is not None
                    else name in case.get("required_semantic_dimensions", []))
            ))
            if not mapped and inventory is not None:
                raise ValueError("source requirement has no declared case witness")
            if not mapped:
                cross_cutting_dimensions.add(name)
                mapped = tuple(dict.fromkeys(str(case["competency_id"]) for case in cases))
            obligation_basis = {
                "semantic_dimension": name,
                "competency_ids": mapped,
                "evidence_refs": tuple(str(ref) for ref in dimension["evidence_refs"]),
                "human_oracle_ref": human_oracle_ref,
            }
            if inventory is not None:
                obligation_basis["source_instance"] = dimension["source_instance"]
            digest = content_hash(obligation_basis)
            obligations.append(
                ValidationObligationV1(
                    obligation_id=dimension["instance_id"] if inventory is not None else f"obligation.{name}",
                    obligation_hash=f"sha256:{digest}",
                    semantic_dimension=name,
                    competency_ids=mapped,
                    acceptance_criterion=f"candidate preserves source-grounded {name}",
                    evidence_refs=obligation_basis["evidence_refs"],
                    provenance_kind="source_derived" if inventory is not None else "authority_reviewed",
                )
            )
        obligations.sort(key=lambda item: item.obligation_id.encode("utf-8"))
        counts = Counter(str(case["competency_id"]) for case in cases)
        declared_dimensions = {obligation.semantic_dimension for obligation in obligations}
        tests: list[ExecutableCompetencyTestV1] = []
        for case in cases:
            competency_id = str(case["competency_id"])
            required_dimensions = set(case.get("required_semantic_dimensions", []))
            if not required_dimensions.issubset(declared_dimensions):
                raise ValueError("oracle case references undeclared validation dimensions")
            test_basis = {
                "competency_id": competency_id,
                "case_kind": case["case_kind"],
                "question": case["question"],
                "expected_result": case["expected_result"],
                "required_obligation_ids": tuple(obligation.obligation_id for obligation in obligations
                    if competency_id in obligation.competency_ids
                    and obligation.semantic_dimension in required_dimensions | cross_cutting_dimensions),
                "evidence_refs": tuple(str(ref) for ref in case["evidence_refs"]),
            }
            if inventory is not None:
                required_ids = tuple(sorted(case.get("required_source_instance_ids", ())))
                if (not required_ids or len(set(required_ids)) != len(required_ids)
                    or not set(required_ids).issubset(i.instance_id for i in inventory.instances)):
                    raise ValueError("case references undeclared or duplicate source requirements")
                test_basis["required_obligation_ids"] = required_ids
                test_basis["expected_instance_results"] = case.get("expected_instance_results")
            if "source_case_id" in case or "observed_facts" in case:
                if not case.get("source_case_id") or not isinstance(case.get("observed_facts"), dict):
                    raise ValueError("reviewed source case identity and facts must be paired")
                test_basis.update(source_case_id=case["source_case_id"], observed_facts=case["observed_facts"])
            if case.get("expected_rule_results") is not None:
                test_basis["expected_rule_results"] = case["expected_rule_results"]
            if case.get("result_rule_ref") is not None:
                test_basis["result_rule_ref"] = case["result_rule_ref"]
            for name in ("expected_compatibility", "expected_conflict_rule_refs", "expected_resolution", "expected_controlling_rule_refs"):
                if case.get(name):
                    test_basis[name] = case[name]
            digest = content_hash(test_basis)
            tests.append(
                ExecutableCompetencyTestV1(
                    test_id=f"competency-test.{competency_id}" + (f".{digest}" if counts[competency_id] > 1 else ""),
                    test_hash=f"sha256:{digest}",
                    provenance_kind="source_derived" if inventory is not None else "authority_reviewed",
                    **test_basis,
                )
            )
        tests.sort(key=lambda item: item.test_id.encode("utf-8"))
        basis = {
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "source_identity_ref": source_identity_ref,
            "source_hash": source_hash,
            "boundary_ref": boundary_ref,
            "boundary_hash": boundary_hash,
            "competency_contract_ref": competency_contract_ref,
            "competency_contract_hash": competency_contract_hash,
            "normative_schema_ref": normative_schema_ref,
            "normative_schema_hash": normative_schema_hash,
            "human_oracle_ref": human_oracle_ref,
            "human_oracle_hash": human_oracle_hash,
            "obligations": [item.model_dump(mode="json") for item in obligations],
            "executable_tests": [item.model_dump(mode="json") for item in tests],
            "independence": self._independence.model_dump(mode="json"),
        }
        if inventory is not None:
            basis["source_requirement_inventory"] = inventory.model_dump(mode="json")
        digest = content_hash(basis)
        return ValidationContractV1(
            validation_contract_ref=f"validation-contract:{digest}",
            validation_contract_hash=f"sha256:{digest}",
            obligations=tuple(obligations),
            executable_tests=tuple(tests),
            independence=self._independence,
            review_state="proposed",
            compiled_at=self._compiled_at,
            **{key: value for key, value in basis.items() if key not in {"obligations", "executable_tests", "independence"}},
        )


__all__ = ["ExternalValidationCompiler"]
