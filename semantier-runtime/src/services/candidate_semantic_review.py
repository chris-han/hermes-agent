"""Pinned hypothesis review over the existing kernel/workspace publication owners."""

from __future__ import annotations

from storage.hosted_postgres import sql_query,begin_write,in_transaction

import json
from datetime import datetime, timezone
from contracts.artifact_pin import ArtifactPinV1
from contracts.knowledge_classification import RuntimeContextDTO
from contracts.candidate_review import (
    CandidateReviewDimensionV1,
    CandidateReviewRequirementV1,
    CandidateReviewEvidenceV1,
    CandidateReviewMemberV1,
    CandidateReviewMaterialV1,
    CandidateReviewProjectionV1,
    CandidateReviewPredicateV1,
    CandidateReviewResultV1,
    CandidateReviewDisclosureV1,
)
from contracts.decision_graph_projection import PredicateEpistemicRequirementV1
from contracts.semantic_kernel import (
    ChangeCandidate,
    ContextGraphOrigin,
    ContextGraph,
    Evidence,
    DecisionRecord,
    ReasoningTrace,
    ReasonerInvocation,
)
from contracts.ontology_workbench_context import SemanticGraphEditCommandV1
from eos.artifact_hashing import content_hash
from services.change_candidate_service import (
    ChangeCandidateService,
    ChangeCandidateError,
)
from services.epistemic_admissibility_routing_service import (
    EpistemicAdmissibilityRoutingService,
)
from storage.ontology_build_store import OntologyBuildStore


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _pin(kind, definition):
    digest = content_hash(definition)
    return ArtifactPinV1(
        artifact_kind=kind,
        artifact_ref=kind + ":" + digest,
        artifact_hash="sha256:" + digest,
    )


_POLICY = {
    "version": "candidate-review-mechanics.v1",
    "checks": [
        "exact_preimage",
        "operand_membership",
        "assertion_evidence_availability",
    ],
    "interpreted_calls_authorized": 0,
}
_COMPILER = _pin(
    "candidate_review_compiler", {"version": "candidate-review-closure.v1"}
)


class CandidateSemanticReviewService:
    def __init__(self, *, store: OntologyBuildStore):
        if type(store) is not OntologyBuildStore:
            raise TypeError("review requires the real kernel store")
        self._store = store
        self._candidates = ChangeCandidateService(store=store)

    def _load(self, scope, pin, model):
        return self._candidates._load(scope=scope, pin=pin, model=model)

    def _command(self, candidate):
        if candidate.stage != "PROPOSAL" or not isinstance(
            candidate.origin, ContextGraphOrigin
        ):
            raise ValueError("PROPOSAL_REQUIRED")
        if not candidate.delta.claims_semantic_graph_change:
            raise ValueError("NOT_RUN_CONTRACT_MISMATCH")
        delta = json.loads(candidate.delta.description)
        if set(delta) != {"graph", "operation", "evidence", "rationale"}:
            raise ValueError("NOT_RUN_CONTRACT_MISMATCH")
        op = delta["operation"]
        refs = tuple(
            sorted(
                {
                    op[k]
                    for k in (
                        "node_ref",
                        "edge_ref",
                        "source_node_ref",
                        "target_node_ref",
                    )
                    if k in op
                }
            )
        )
        return SemanticGraphEditCommandV1.model_validate(
            {
                "command": "SUBMIT_SEMANTIC_GRAPH_EDIT",
                "graph_pin": delta["graph"],
                "edit": {
                    "schema_version": "semantic_graph_edit.v1",
                    "graph": delta["graph"],
                    "operation": op,
                    "affected_object_refs": refs,
                    "evidence_refs": [p["artifact_ref"] for p in delta["evidence"]],
                    "source": "DIRECT",
                },
                "evidence_pins": delta["evidence"],
                "rationale": delta["rationale"],
            }
        )

    def compile(
        self,
        *,
        organization_id,
        workspace_id,
        proposal_pin,
        material_basis=None,
        development_requirements=(),
    ):
        scope = (organization_id, workspace_id)
        candidate = self._load(scope, proposal_pin, ChangeCandidate)
        if any(r.basis_class != "DEVELOPMENT" for r in development_requirements):
            raise ValueError("DEVELOPMENT_REQUIREMENTS_REQUIRED")
        return self._compile(
            candidate=candidate,
            scope=scope,
            material_basis=material_basis,
            requirements=None
            if not development_requirements
            else development_requirements,
            policy=_POLICY,
            compiler=_COMPILER,
        )

    def _compile(
        self,
        *,
        candidate,
        scope,
        material_basis,
        requirements,
        policy,
        compiler,
        recorded_unavailable=(),
    ):
        command = self._command(candidate)
        context, basis, delta = self._candidates.reconcile_semantic_graph_edit(
            organization_id=scope[0], workspace_id=scope[1], command=command
        )
        if (context.pin(), command.evidence_pins, basis, delta) != (
            candidate.origin.context,
            candidate.origin.evidence,
            candidate.evaluation_basis,
            candidate.delta,
        ):
            raise ValueError("REVIEW_ORIGINAL_BASIS_MISMATCH")
        affected = set(command.edit.affected_object_refs)
        assertions = tuple(
            a
            for a in context.assertions
            if a.subject_ref in affected or a.object_ref in affected
        )
        associated = (
            {p for a in assertions for p in a.evidence}
            if context.assertions
            else set(context.evidence)
        )
        allowed = associated | set(command.evidence_pins)
        if not allowed <= set(context.formation.source_inputs):
            raise ValueError("EVIDENCE_OUT_OF_CLOSURE")
        if requirements is None:

            def mechanical(name, pins, kind):
                return CandidateReviewRequirementV1(
                    requirement=PredicateEpistemicRequirementV1(
                        predicate_id=name, required_semantic_dimension_ids=("evidence",)
                    ),
                    mode="COMPUTABLE",
                    check_kind=kind,
                    basis_class="CORE_MECHANICAL",
                    dimensions=(
                        CandidateReviewDimensionV1(
                            dimension_id="evidence",
                            required_evidence_pins=tuple(
                                sorted(pins, key=lambda p: p.artifact_ref)
                            ),
                        ),
                    ),
                )

            requirements = [
                mechanical("edit.preimage", command.evidence_pins, "PREIMAGE")
            ]
            requirements += [
                mechanical(
                    "assertion." + content_hash(a.model_dump(mode="json")),
                    a.evidence,
                    "EVIDENCE",
                )
                for a in assertions
            ]
            requirements += [
                CandidateReviewRequirementV1(
                    requirement=PredicateEpistemicRequirementV1(
                        predicate_id="semantic.relation"
                    ),
                    mode="INTERPRETED",
                    check_kind="SEMANTIC",
                    basis_class="UNAVAILABLE",
                )
            ]
        else:
            requirements = tuple(
                CandidateReviewRequirementV1.model_validate(r.model_dump(mode="json"))
                for r in requirements
            )
            for r in requirements:
                for d in r.dimensions:
                    if not set(d.required_evidence_pins) <= allowed:
                        raise ValueError("EVIDENCE_OUT_OF_CLOSURE")
        supporting = []
        missing = []
        for pin in sorted(allowed, key=lambda p: p.artifact_ref):
            if pin in recorded_unavailable:
                missing.append(pin)
                continue
            try:
                evidence = self._load(scope, pin, Evidence)
            except ChangeCandidateError as exc:
                if exc.code != "OBJECT_NOT_FOUND" or pin in command.evidence_pins:
                    raise
                # Do not turn a foreign existing witness into mere absence.
                row = self._store._connection.execute(
                    sql_query("SELECT organization_id,workspace_id FROM ontology_build_artifacts WHERE artifact_ref=?",'SELECT organization_id,workspace_id FROM ontology_build_artifacts WHERE artifact_ref=%s'),
                    (pin.artifact_ref,),
                ).fetchone()
                if row is not None:
                    raise ValueError("FORBIDDEN_SCOPE") from exc
                missing.append(pin)
                continue
            supporting.append(
                CandidateReviewEvidenceV1(
                    pin=pin,
                    evidence_role=evidence.evidence_role,
                    source_hash=evidence.source_hash,
                    selector_hash=evidence.selector_hash,
                )
            )
        members = []
        for kind, items, key in (
            ("node", context.nodes, "node_id"),
            ("edge", context.edges, "edge_id"),
            ("assertion", assertions, "assertion_id"),
        ):
            for item in items:
                ref = getattr(item, key)
                if kind == "assertion" or ref in affected:
                    members.append(
                        CandidateReviewMemberV1(
                            kind=kind,
                            object_ref=ref,
                            content_hash="sha256:"
                            + content_hash(item.model_dump(mode="json")),
                            evidence_pins=getattr(item, "evidence", ()),
                        )
                    )
        return CandidateReviewProjectionV1.create(
            organization_id=scope[0],
            workspace_id=scope[1],
            proposal_pin=candidate.pin(),
            proposal_delta_hash="sha256:"
            + content_hash(candidate.delta.model_dump(mode="json")),
            baseline_graph_pin=context.pin(),
            baseline_state=context.state,
            ontology_pin=context.ontology,
            proposed_edit_json=_json(command.edit.model_dump(mode="json")),
            original_material_pins=(
                context.world.source_graph,
                context.formation.formation_policy,
                *context.formation.source_inputs,
            ),
            original_embedded_hash="sha256:"
            + content_hash(
                {
                    "intent": context.intent.model_dump(mode="json"),
                    "world": context.world.model_dump(mode="json"),
                    "formation": context.formation.model_dump(mode="json"),
                }
            ),
            selected_evidence_pins=command.evidence_pins,
            supporting_evidence=tuple(supporting),
            unresolved_evidence_pins=tuple(missing),
            closure_members=tuple(members),
            requirements=tuple(
                sorted(requirements, key=lambda r: r.requirement.predicate_id)
            ),
            policy_definition=_json(policy),
            policy_pin=_pin("candidate_review_policy", policy),
            compiler_pin=compiler,
            evaluated_material_basis=material_basis,
        )

    def evaluate(self, *, projection, development_evaluator=None, _verify_only=False):
        projection = CandidateReviewProjectionV1.model_validate(
            projection.model_dump(mode="json")
        )
        routing = EpistemicAdmissibilityRoutingService().compile_candidate_review(
            projection=projection
        )
        results = []
        for r in projection.requirements:
            available, missing = routing[r.requirement.predicate_id]
            status = (
                "NOT_ESTABLISHED"
                if r.basis_class == "UNAVAILABLE"
                else "INADMISSIBLE"
                if missing
                else "ADMISSIBLE"
            )
            execution = "NOT_RUN"
            verdict = None
            semantic = None
            failure = None
            count = 0
            if status == "NOT_ESTABLISHED":
                reason = "RULE_BASIS_UNAVAILABLE"
            elif missing:
                reason = "INSUFFICIENT_EVIDENCE"
            elif r.mode == "INTERPRETED":
                reason = (
                    "EVALUATOR_UNAVAILABLE"
                    if r.evaluator_pin is None or r.qualification_pin is None
                    else "CALLS_NOT_AUTHORIZED"
                )
                if (
                    reason == "CALLS_NOT_AUTHORIZED"
                    and r.basis_class == "DEVELOPMENT"
                    and development_evaluator is not None
                    and not _verify_only
                ):
                    count = 1
                    try:
                        semantic = development_evaluator(projection, r)
                        if semantic not in {
                            "SEMANTIC_TRUE",
                            "SEMANTIC_FALSE",
                            "SEMANTIC_UNKNOWN",
                        }:
                            raise ValueError("invalid external verdict")
                        execution = "RAN"
                        reason = (
                            "EVALUATOR_ABSTENTION"
                            if semantic == "SEMANTIC_UNKNOWN"
                            else "DEVELOPMENT_EVALUATOR_RESULT"
                        )
                    except (ConnectionError, TimeoutError):
                        semantic = None
                        execution = "FAILED"
                        failure = "PROVIDER"
                        reason = "PROVIDER_FAILURE"
                    except ValueError:
                        semantic = None
                        execution = "FAILED"
                        failure = "UNSUPPORTED_CONTRACT"
                        reason = "EVALUATOR_CONTRACT_FAILURE"
            else:
                # Actual exact checks against real owner state; no model guesses.
                if r.check_kind == "PREIMAGE" and not _verify_only:
                    candidate = self._load(
                        (projection.organization_id, projection.workspace_id),
                        projection.proposal_pin,
                        ChangeCandidate,
                    )
                    command = self._command(candidate)
                    self._candidates.reconcile_semantic_graph_edit(
                        organization_id=projection.organization_id,
                        workspace_id=projection.workspace_id,
                        command=command,
                    )
                if not _verify_only:
                    for witness in projection.supporting_evidence:
                        self._load(
                            (projection.organization_id, projection.workspace_id),
                            witness.pin,
                            Evidence,
                        )
                execution = "RAN"
                verdict = "PASS"
                count = 1
                reason = "EXACT_MECHANICAL_CHECK"
            results.append(
                CandidateReviewPredicateV1(
                    predicate_id=r.requirement.predicate_id,
                    rule_pin=r.rule_pin(),
                    required_dimensions=r.requirement.required_semantic_dimension_ids,
                    available_dimensions=available,
                    missing_dimensions=missing,
                    admissibility=status,
                    execution=execution,
                    reason=reason,
                    deterministic_result=verdict,
                    semantic_verdict=semantic,
                    failure_kind=failure,
                    invocation_count=count,
                    evidence_pins=tuple(w.pin for w in projection.supporting_evidence),
                )
            )
        return tuple(results)

    def _resolve_material(self, *, ctx, cursors, projection):
        from contracts.knowledge_classification import RuntimeContextDTO
        from contracts.ontology_workbench_context import WorkspaceArtifactPreparationV1

        actor = RuntimeContextDTO(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            actor_id=ctx.user_id,
        )
        cursors._scope(
            ctx
        )  # Production authority and governed session, including retries.
        cursors._authority.resolve_pins(
            actor_context=actor,
            pins=(
                projection.proposal_pin,
                projection.baseline_graph_pin,
                projection.ontology_pin,
                *tuple(
                    p
                    for p in projection.original_material_pins
                    if p.artifact_kind != "workspace_artifact_preparation"
                    and p not in projection.unresolved_evidence_pins
                ),
            ),
        )
        graph = self._load(
            (ctx.organization_id, ctx.workspace_id),
            projection.baseline_graph_pin,
            ContextGraph,
        )
        if (
            graph.formation.formation_policy.artifact_kind
            == "workspace_artifact_preparation"
        ):
            pin = graph.formation.formation_policy
            value = self._store.get(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                artifact_ref=pin.artifact_ref,
                contract_type=WorkspaceArtifactPreparationV1,
            )
            if value is None or value.preparation_hash != pin.artifact_hash:
                raise ValueError("REPLAY_BASIS_NOT_CLOSED")
        return graph

    def _currency(self, *, ctx, cursors, projection):
        graph = self._resolve_material(ctx=ctx, cursors=cursors, projection=projection)
        material = projection.evaluated_material_basis
        head = cursors.get(ctx=ctx, cursor_ref=material.cursor_ref)
        reasons = []
        if (
            CandidateReviewMaterialV1.from_head(
                head, preparation_pin=material.preparation_pin
            )
            != material
        ):
            reasons.append("STALE_MATERIAL_CONTEXT")
        for kind, pin in (
            ("semantier.context_graph.v1", graph.pin()),
            ("semantier.ontology.v1", projection.ontology_pin),
        ):
            successor = self._store._connection.execute(
                sql_query("SELECT 1 FROM ontology_build_artifacts WHERE organization_id=? AND workspace_id=? AND schema_version=? AND json_extract(payload_json,'$.predecessor.artifact_ref')=? LIMIT 1","SELECT 1 FROM ontology_build_artifacts WHERE organization_id=%s AND workspace_id=%s AND schema_version=%s AND (payload_json::jsonb #>> '{predecessor,artifact_ref}')=%s LIMIT 1"),
                (ctx.organization_id, ctx.workspace_id, kind, pin.artifact_ref),
            ).fetchone()
            if successor:
                reasons.append(
                    "STALE_"
                    + (
                        "SEMANTIC_BASIS"
                        if kind.endswith("context_graph.v1")
                        else "ONTOLOGY_BASIS"
                    )
                )
        scope = cursors._scope(ctx)
        _, _, runtime = cursors._facts(
            scope,
            head.selection_pin,
            head.design_binding_set_pin,
            head.runtime_binding_set_pin,
        )
        if (
            graph.formation.formation_policy.artifact_kind
            == "workspace_artifact_preparation"
        ):
            active = [entry for entry in runtime.entries if entry.state == "ACTIVE"]
            if (
                len(active) != 1
                or active[0].preparation_pin != graph.formation.formation_policy
            ):
                reasons.append("STALE_PREPARATION_BASIS")
            from contracts.ontology_workbench_context import (
                WorkspaceArtifactPreparationV1,
            )

            original = self._store.get(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                artifact_ref=graph.formation.formation_policy.artifact_ref,
                contract_type=WorkspaceArtifactPreparationV1,
            )
            source = cursors._authority.resolve_pins(
                actor_context=RuntimeContextDTO(
                    organization_id=ctx.organization_id,
                    workspace_id=ctx.workspace_id,
                    actor_id=ctx.user_id,
                ),
                pins=(graph.world.source_graph,),
            )[0]
            if (
                graph.world.source_graph not in original.derived_artifact_pins
                or getattr(source, "ontology", None) != graph.ontology
            ):
                raise ValueError("REPLAY_BASIS_NOT_CLOSED")
        for pin in projection.unresolved_evidence_pins:
            try:
                self._load((ctx.organization_id, ctx.workspace_id), pin, Evidence)
            except ChangeCandidateError as exc:
                if exc.code != "OBJECT_NOT_FOUND":
                    raise
            else:
                reasons.append("STALE_EVIDENCE_BASIS")
                break
        if projection.compiler_pin != _COMPILER or projection.policy_pin != _pin(
            "candidate_review_policy", _POLICY
        ):
            reasons.append("STALE_REVIEW_DEPENDENCY")
        return tuple(reasons)

    def prior(self, *, ctx, request_id, envelope_hash):
        row = self._store._connection.execute(
            sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?",'SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s'),
            (
                ctx.organization_id,
                ctx.workspace_id,
                ctx.user_id,
                "REVIEW_SEMANTIC_GRAPH_EDIT",
                request_id,
            ),
        ).fetchone()
        if row is None:
            return None
        record = self._load(
            (ctx.organization_id, ctx.workspace_id),
            ArtifactPinV1(
                artifact_kind="decision_record",
                artifact_ref=row[1],
                artifact_hash=row[2],
            ),
            DecisionRecord,
        )
        if (
            record.candidate_review is None
            or record.candidate_review.envelope_hash != envelope_hash
            or row[0] != record.candidate_review.request_hash
        ):
            raise ValueError("REQUEST_CONFLICT")
        return record

    def publish(
        self,
        *,
        ctx,
        cursors,
        projection,
        predicates,
        human_outcome,
        human_rationale,
        request_id,
        envelope_hash,
        evaluated_at,
    ):
        projection = CandidateReviewProjectionV1.model_validate(
            projection.model_dump(mode="json")
        )
        return self._store.publish_candidate_review(
            service=self,
            ctx=ctx,
            cursors=cursors,
            projection=projection,
            predicates=predicates,
            human_outcome=human_outcome,
            human_rationale=human_rationale,
            request_id=request_id,
            envelope_hash=envelope_hash,
            evaluated_at=evaluated_at,
        )

    def _publish_on_connection(
        self,
        *,
        ctx,
        cursors,
        projection,
        predicates,
        human_outcome,
        human_rationale,
        request_id,
        envelope_hash,
        evaluated_at,
    ):
        if (
            not in_transaction(self._store._connection)
            or cursors._store is not self._store
        ):
            raise ValueError("REVIEW_WORKSPACE_OWNER_MISMATCH")
        if (
            projection.evaluated_material_basis is None
            or (ctx.organization_id, ctx.workspace_id)
            != (projection.organization_id, projection.workspace_id)
            or (ctx.user_id, projection.evaluated_material_basis.session_id)
            != (
                projection.evaluated_material_basis.actor_ref,
                cursors._scope(ctx)["session_id"],
            )
        ):
            raise ValueError("FORBIDDEN_SCOPE")
        if any(r.basis_class == "DEVELOPMENT" for r in projection.requirements):
            raise ValueError("DEVELOPMENT_REVIEW_CANNOT_PUBLISH")
        prior = self.prior(ctx=ctx, request_id=request_id, envelope_hash=envelope_hash)
        checked_at = datetime.now(timezone.utc)
        if prior is not None:
            if (
                projection.projection_hash
                != prior.candidate_review.projection.projection_hash
                or (human_outcome, human_rationale)
                != (
                    prior.candidate_review.human_outcome,
                    prior.candidate_review.human_rationale,
                )
            ):
                raise ValueError("REQUEST_CONFLICT")
            reasons = self._currency(
                ctx=ctx, cursors=cursors, projection=prior.candidate_review.projection
            )
            reasons = tuple(
                dict.fromkeys((*prior.candidate_review.superseded_reasons, *reasons))
            )
            return CandidateReviewDisclosureV1(
                decision_pin=prior.pin(),
                result=prior.candidate_review,
                delivery_status="SUPERSEDED"
                if reasons
                else "CURRENT_AS_OF_PUBLICATION",
                superseded_reasons=reasons,
                checked_at=checked_at,
            )
        # Reconstruct the supplied projection at the trust boundary; a hash alone is not authority.
        actual = self.compile(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            proposal_pin=projection.proposal_pin,
            material_basis=projection.evaluated_material_basis,
        )
        if (
            actual != projection
            or self.evaluate(projection=actual, _verify_only=True) != predicates
        ):
            raise ValueError("REVIEW_DERIVATION_MISMATCH")
        reasons = self._currency(ctx=ctx, cursors=cursors, projection=projection)
        request_hash = "sha256:" + content_hash(
            {
                "envelope_hash": envelope_hash,
                "projection_hash": projection.projection_hash,
                "human_outcome": human_outcome,
                "human_rationale": human_rationale,
            }
        )
        closure = (
            "INCOMPLETE"
            if any(p.admissibility == "INADMISSIBLE" for p in predicates)
            else "NOT_ESTABLISHED"
            if any(p.admissibility == "NOT_ESTABLISHED" for p in predicates)
            else "COMPLETE"
        )
        result = CandidateReviewResultV1.create(
            projection=projection,
            predicates=predicates,
            closure_status=closure,
            human_outcome=human_outcome,
            human_rationale=human_rationale,
            request_hash=request_hash,
            envelope_hash=envelope_hash,
            evaluated_at=evaluated_at,
            checked_at=checked_at,
            delivery_status="SUPERSEDED" if reasons else "CURRENT_AS_OF_PUBLICATION",
            superseded_reasons=reasons,
        )
        invocations = tuple(
            ReasonerInvocation(
                invocation_id=p.predicate_id,
                strategy="DETERMINISTIC",
                executable_engine=projection.compiler_pin,
                rules=(p.rule_pin, projection.policy_pin),
                inputs=(projection.proposal_pin, projection.baseline_graph_pin),
                outputs=(p.deterministic_result,),
                proof_steps=(p.reason,),
                evidence=p.evidence_pins,
            )
            for p in predicates
            if p.execution == "RAN"
        )
        record = DecisionRecord.create(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            purpose="REVIEW",
            subject=projection.proposal_pin,
            context=projection.baseline_graph_pin,
            ontology=projection.ontology_pin,
            result_code="REVIEW_RECORDED",
            epistemic_status="UNKNOWN",
            evidence=tuple(w.pin for w in projection.supporting_evidence),
            reasoning=ReasoningTrace(
                strategy="CANDIDATE_REVIEW",
                reasoning_version="candidate-review.v1",
                invocations=invocations,
                diagnostics=("SEMANTIC_EFFICACY_UNMEASURED",),
            ),
            actor_id=ctx.user_id,
            reasoner_identity=projection.compiler_pin.artifact_ref,
            candidate_review=result,
            recorded_at=checked_at,
        )
        self._store._insert_prepared((self._store._prepare(record),))
        self._store._connection.execute(
            sql_query("INSERT INTO workspace_operation_request_index (organization_id,workspace_id,actor_ref,operation,request_id,request_hash,receipt_ref,receipt_hash) VALUES (?,?,?,?,?,?,?,?)",'INSERT INTO workspace_operation_request_index (organization_id,workspace_id,actor_ref,operation,request_id,request_hash,receipt_ref,receipt_hash) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'),
            (
                ctx.organization_id,
                ctx.workspace_id,
                ctx.user_id,
                "REVIEW_SEMANTIC_GRAPH_EDIT",
                request_id,
                request_hash,
                record.artifact_ref,
                record.artifact_hash,
            ),
        )
        return CandidateReviewDisclosureV1(
            decision_pin=record.pin(),
            result=result,
            delivery_status=result.delivery_status,
            superseded_reasons=reasons,
            checked_at=checked_at,
        )

    def replay(self, *, organization_id, workspace_id, decision_pin):
        scope = (organization_id, workspace_id)
        record = self._load(scope, decision_pin, DecisionRecord)
        result = record.candidate_review
        if result is None:
            raise ValueError("NOT_RUN_CONTRACT_MISMATCH")
        projection = result.projection
        candidate = self._load(scope, projection.proposal_pin, ChangeCandidate)
        requirements = projection.requirements
        # Retained rule definitions are used, never today's policy, cursor or provider.
        rebuilt = self._compile(
            candidate=candidate,
            scope=scope,
            material_basis=projection.evaluated_material_basis,
            requirements=requirements,
            policy=json.loads(projection.policy_definition),
            compiler=projection.compiler_pin,
            recorded_unavailable=projection.unresolved_evidence_pins,
        )
        if (
            rebuilt != projection
            or self.evaluate(projection=rebuilt, _verify_only=True) != result.predicates
        ):
            raise ValueError("REVIEW_REPLAY_MISMATCH")
        return record
