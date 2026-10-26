"""Source-neutral bounded assessment; all writes stay with OntologyBuildStore."""

from datetime import datetime, timezone
from contracts.requirements_reconciliation import (
    RequirementsReviewV1,
    RequirementQuestionV1,
)
from contracts.normative_conflict import NormativeSetV1, NormativeDomainBasisV2
from eos.artifact_hashing import content_hash
from services.normative_conflict_analysis import (
    NormativeSetCompiler,
    NormativeCompatibilityEvaluator,
    _digest,
)


class RequirementsReconciliationService:
    @staticmethod
    def assess(
        *,
        candidate,
        journey_id,
        source_role,
        actor_ref,
        prior=None,
        dispositions=None,
        source_evidence=None,
        journey_revision=None,
        domain_basis=None,
        source_predecessor_ref=None,
    ):
        if domain_basis is not None:
            domain_basis.verify_candidate(candidate)
        now = datetime.now(timezone.utc)
        source_evidence = tuple(
            source_evidence
            if source_evidence is not None
            else prior.source_evidence
            if prior
            else ()
        )
        frames = {f.frame_id: f for f in candidate.frames}
        same_lineage = prior is not None and (
            candidate.source_identity_ref == prior.candidate.source_identity_ref
            or source_predecessor_ref == prior.candidate.source_identity_ref
        )
        source_anchor = (
            (prior.source_lineage_anchor or prior.candidate.source_identity_ref)
            if same_lineage
            else candidate.source_identity_ref
        )
        materials = RequirementsReconciliationService._frame_materials(
            candidate.frames, source_evidence
        )
        unknown, groups = set(candidate.unresolved_slot_refs), {}
        if candidate.completeness_state != "COMPLETE" and not unknown:
            unknown.add(candidate.candidate_set_ref)
        rules = [f for f in candidate.frames if f.frame_kind == "rule"]
        context_dependent = set()
        for frame in candidate.frames:
            if frame.frame_kind == "entity_distinction":
                unknown.add(frame.frame_id)
                if frame.rule_effects:
                    context_dependent.update(e.rule_ref for e in frame.rule_effects)
                else:
                    context_dependent.update(r.frame_id for r in rules)
        for rule in rules:
            # Only unconditional, fully interpreted contexts are sound without
            # observations. Conditional/exception/time semantics are explicit gaps.
            if (
                rule.frame_id in context_dependent
                or not rule.actor_ref
                or not rule.object_ref
                or rule.epistemic_status != "FOUND"
                or rule.interpretation_required
                or rule.conditions
                or rule.exception_refs
                or rule.temporal_condition_refs
                or rule.qualification_refs
                or not rule.rule_effects
                or rule.modality == "permission"
            ):
                unknown.add(rule.frame_id)
                continue
            scopes = [frames.get(ref) for ref in rule.scope_refs]
            if any(
                f is None
                or f.frame_kind != "scope"
                or f.epistemic_status != "FOUND"
                or f.conditions
                for f in scopes
            ):
                unknown.add(rule.frame_id)
                continue
            context = (
                rule.actor_ref,
                rule.object_ref,
                tuple(
                    sorted(
                        (tuple(sorted(f.actor_scope)), tuple(sorted(f.object_scope)))
                        for f in scopes
                    )
                ),
            )
            groups.setdefault(context, []).append(rule)
        # Different scope declarations can overlap. Equality of their labels is
        # not an overlap proof; preserve UNKNOWN across unlike declarations.
        scope_groups = {}
        for context, members in groups.items():
            scope_groups.setdefault(context[:2], []).append(members)
        for variants in scope_groups.values():
            if len(variants) > 1:
                unknown.update(
                    rule.frame_id for members in variants for rule in members
                )
        questions = []
        assessed_sets, assessments = [], []
        for context, members in groups.items():
            all_effects = tuple(
                NormativeSetCompiler._effect(r, e)
                for r in members
                for e in r.rule_effects
            )
            by_fact = {}
            for effect in all_effects:
                by_fact.setdefault(effect.fact_ref, []).append(effect)
            for fact, fact_effects in sorted(by_fact.items()):
                effects = tuple(sorted(fact_effects, key=lambda e: e.effect_ref))
                payload = dict(
                    schema_version="normative_set.v1",
                    organization_id=candidate.organization_id,
                    workspace_id=candidate.workspace_id,
                    context_ref="requirements-context:" + content_hash(context),
                    context_hash="sha256:" + content_hash(context),
                    compiled_rule_plan_ref=candidate.candidate_set_ref,
                    compiled_rule_plan_hash=candidate.candidate_set_hash,
                    member_rule_refs=tuple(sorted(r.frame_id for r in members)),
                    effects=effects,
                    inactive_rule_refs=(),
                    unknown_rule_refs=(),
                    closure_state="COMPLETE",
                    unresolved_refs=(),
                    formed_at=now,
                )
                if domain_basis is not None:
                    payload["domain_basis"] = domain_basis.model_dump(mode="json")
                digest = _digest(
                    {**payload, "effects": [e.model_dump(mode="json") for e in effects]}
                )
                ns = NormativeSetV1(
                    normative_set_ref="normative-set:" + digest,
                    normative_set_hash="sha256:" + digest,
                    **payload,
                )
                ev = NormativeCompatibilityEvaluator(
                    evaluator_ref="requirements-algebra:v2",
                    evaluator_hash="sha256:"
                    + content_hash(
                        {"contract": "typed_bounded_algebra.v2"}
                    ),
                )
                result = ev.evaluate(normative_set=ns, evaluated_at=now)
                assessed_sets.append(ns)
                assessments.append(result)
                unknown.update(
                    e.rule_ref
                    for e in effects
                    if e.effect_ref in result.unresolved_refs
                )
                if result.outcome == "CONFLICT":
                    selected = [
                        e
                        for e in effects
                        if e.effect_ref in result.conflicting_effect_refs
                    ]
                    ids = tuple(sorted({e.rule_ref for e in selected}))
                    questions.append(
                        RequirementsReconciliationService._question(
                            "verified_conflict",
                            ids,
                            frames,
                            candidate,
                            source_role,
                            fact_ref=fact,
                            proof=result,
                            source_anchor=source_anchor,
                            materials=materials,
                        )
                    )
        for ref in sorted(unknown):
            questions.append(
                RequirementsReconciliationService._question(
                    "missing_semantics",
                    (ref,),
                    frames,
                    candidate,
                    source_role,
                    source_anchor=source_anchor,
                    materials=materials,
                )
            )
        keys = {(q.question_id, q.dependency_hash) for q in questions}
        retained = tuple(
            d
            for d in (
                dispositions
                if dispositions is not None
                else prior.dispositions
                if prior
                else ()
            )
            if (d.question_id, d.dependency_hash) in keys
        )
        all_dispositions = (
            dispositions
            if dispositions is not None
            else prior.dispositions
            if prior
            else ()
        )
        superseded = (prior.superseded_dispositions if prior else ()) + tuple(
            d
            for d in all_dispositions
            if (d.question_id, d.dependency_hash) not in keys
        )
        previous = (
            RequirementsReconciliationService._frame_materials(
                prior.candidate.frames, prior.source_evidence
            )
            if prior and same_lineage
            else {}
        )
        current = materials
        if prior is not None and not same_lineage:
            previous = {
                f.frame_id: "different_source_lineage:" + f.frame_hash
                for f in prior.candidate.frames
            }
        affected = tuple(
            sorted(
                k
                for k in previous.keys() | current.keys()
                if previous.get(k) != current.get(k)
            )
        )
        complete = not unknown and candidate.completeness_state == "COMPLETE"
        payload = dict(
            schema_version="requirements_review.v1",
            organization_id=candidate.organization_id,
            workspace_id=candidate.workspace_id,
            journey_id=journey_id,
            revision=prior.revision + 1 if prior else 1,
            journey_revision=journey_revision
            if journey_revision is not None
            else prior.journey_revision + 1
            if prior
            else 0,
            predecessor_pin=prior.pin() if prior else None,
            source_role=source_role,
            candidate=candidate,
            source_evidence=source_evidence,
            source_lineage_anchor=source_anchor,
            outcome="CONFLICT"
            if any(q.kind == "verified_conflict" for q in questions)
            else "COMPATIBLE"
            if complete
            else "UNKNOWN",
            coverage="BOUNDED_COMPLETE" if complete else "INCOMPLETE",
            supported_domains=tuple(sorted({f.value_type for f in domain_basis.facts})) if domain_basis else (),
            supported_predicates=("EQ", "IN", "GE", "GT", "LE", "LT"),
            unknown_rule_refs=tuple(sorted(unknown)),
            questions=tuple(questions),
            dispositions=retained,
            superseded_dispositions=superseded,
            affected_rule_refs=affected,
            actor_ref=actor_ref,
            created_at=now,
        )
        if not domain_basis:
            payload.pop("supported_domains")
        if assessed_sets:
            payload["assessed_sets"] = tuple(assessed_sets)
            payload["assessments"] = tuple(assessments)
        digest = _digest(payload)
        return RequirementsReviewV1(
            review_ref="requirements-review:" + digest,
            review_hash="sha256:" + digest,
            **payload,
        )

    @staticmethod
    def _question(
        kind,
        ids,
        frames,
        candidate,
        role,
        fact_ref=None,
        proof=None,
        source_anchor=None,
        materials=None,
    ):
        dependencies = set(ids)
        for ref in ids:
            f = frames.get(ref)
            if f is not None and f.frame_kind == "rule":
                for name in (
                    "scope_refs",
                    "exception_refs",
                    "qualification_refs",
                    "temporal_condition_refs",
                    "precedence_refs",
                ):
                    dependencies.update(getattr(f, name))
        basis = dict(
            kind=kind,
            role=role,
            source=source_anchor or candidate.source_identity_ref,
            fact_ref=fact_ref,
            frames={
                ref: materials.get(ref)
                if materials is not None
                else frames[ref].frame_hash
                if ref in frames
                else None
                for ref in sorted(dependencies)
            },
        )
        digest = content_hash(basis)
        evidence = tuple(
            sorted(
                {e for ref in ids if ref in frames for e in frames[ref].evidence_refs}
            )
        )
        return RequirementQuestionV1(
            question_id="requirement-question:" + digest,
            kind=kind,
            member_rule_refs=ids,
            evidence_refs=evidence,
            dependency_hash="sha256:" + digest,
            fact_ref=fact_ref,
            proof_result_ref=proof.result_ref if proof else None,
            member_effect_refs=proof.conflicting_effect_refs if proof else (),
        )

    @staticmethod
    def persisted_source_evidence(store, candidate, prior=None):
        import json
        from contracts.evidence_location import EvidenceRef
        from contracts.normative_semantics import NormativeSemanticProposalEvidenceV1

        if candidate.proposal_evidence_ref is None:
            if prior is not None and (
                candidate.source_identity_ref,
                candidate.source_hash,
            ) == (prior.candidate.source_identity_ref, prior.candidate.source_hash):
                return prior.source_evidence
            return ()
        receipt = store.get(
            organization_id=candidate.organization_id,
            workspace_id=candidate.workspace_id,
            artifact_ref=candidate.proposal_evidence_ref,
            contract_type=NormativeSemanticProposalEvidenceV1,
        )
        if receipt is None or receipt.evidence_hash != candidate.proposal_evidence_hash:
            raise ValueError("REQUIREMENTS_SOURCE_EVIDENCE_MISMATCH")
        source = json.loads(receipt.input_text)["source_context"]
        from services.normative_schema_production_service import (
            NormativeSchemaProductionService,
        )

        evidence = tuple(
            EvidenceRef.model_validate(e)
            for e in NormativeSchemaProductionService.source_evidence(source)
        )
        referenced = {e for f in candidate.frames for e in f.evidence_refs}
        return tuple(e for e in evidence if e.evidence_ref in referenced)

    @staticmethod
    def persisted_domain_basis(store, candidate):
        from contracts.normative_schema import NormativeSchemaContractV1

        schema = store.get(
            organization_id=candidate.organization_id,
            workspace_id=candidate.workspace_id,
            artifact_ref=candidate.normative_schema_ref,
            contract_type=NormativeSchemaContractV1,
        )
        if schema is None:
            if candidate.proposal_evidence_ref is not None:
                raise ValueError("REQUIREMENTS_SCHEMA_BASIS_MISSING")
            return None
        if schema.schema_hash != candidate.normative_schema_hash:
            raise ValueError("REQUIREMENTS_SCHEMA_BASIS_MISMATCH")
        basis = NormativeDomainBasisV2.from_schema(schema)
        basis.verify_candidate(candidate)
        return basis

    @staticmethod
    def _frame_materials(frames, evidence):
        by_ref = {e.evidence_ref: e for e in evidence}
        counts = {}
        for e in evidence:
            quote = e.selector.exact_quote
            if quote:
                counts[quote] = counts.get(quote, 0) + 1
        result = {}
        for frame in frames:
            quotes = [
                by_ref[ref].selector.exact_quote if ref in by_ref else None
                for ref in frame.evidence_refs
            ]
            if not quotes or any(not q or counts.get(q) != 1 for q in quotes):
                result[frame.frame_id] = frame.frame_hash
                continue
            meaning = frame.model_dump(
                mode="json",
                exclude={"frame_hash", "evidence_refs", "source_region_refs"},
            )

            # Nested predicates/effects also carry transport-specific evidence
            # identifiers. Bind their exact unique quotes, preserving all meaning.
            def ground(value):
                if isinstance(value, dict):
                    return {
                        key: sorted(by_ref[ref].selector.exact_quote for ref in item)
                        if key == "evidence_refs"
                        and all(
                            ref in by_ref and by_ref[ref].selector.exact_quote
                            for ref in item
                        )
                        else ground(item)
                        for key, item in value.items()
                    }
                if isinstance(value, list):
                    return [ground(item) for item in value]
                return value

            result[frame.frame_id] = content_hash(
                dict(meaning=ground(meaning), source_quotes=sorted(quotes))
            )
        return result

    @staticmethod
    def persisted_source_predecessor(source_registry, candidate, prior):
        if (
            prior is None
            or candidate.source_identity_ref == prior.candidate.source_identity_ref
            or source_registry is None
        ):
            return None
        from storage.knowledge_builder_store import KnowledgeBuilderStore

        if not isinstance(source_registry, KnowledgeBuilderStore):
            raise TypeError("source lineage requires the production source registry")
        _, identity = source_registry.resolve_source_identity_lineage(
            organization_id=candidate.organization_id,
            workspace_id=candidate.workspace_id,
            source_identity_ref=candidate.source_identity_ref,
        )
        if identity.source_hash != candidate.source_hash:
            raise ValueError("REQUIREMENTS_SOURCE_LINEAGE_MISMATCH")
        return identity.supersedes_source_identity_ref
