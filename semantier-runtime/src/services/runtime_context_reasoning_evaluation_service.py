"""Context-consuming actor producer with durable dispatch and raw return capture.

Producer execution emits immutable INFERENCE. The separate bounded grading
operation creates independently referenced EVALUATION without native execution.
"""

from __future__ import annotations

from dataclasses import asdict
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import DecisionRecord, Evidence, ReasonerInvocation, ReasoningTrace
from contracts.runtime_producer_integrity import (
    RuntimeContextReferencePanelV1,
    RuntimeContextReasoningRequestV1,
    RuntimeContextReasoningCaptureV1,
    RuntimeContextExecutionAdmissionV2,
    RuntimeContextRequestIdentityV2,
    RuntimeContextRequestOutcomeV2,
    RuntimeEvaluationStateV2,
)
from contracts.runtime_semantic_evaluation import (
    GovernedTargetPinV1,
    RuntimeEvaluationRunV1,
    RuntimeEvaluationDiagnosticV1,
)
from eos.artifact_hashing import content_hash
from services.runtime_document_evaluation_service import _captured
from services.runtime_errors import RuntimeContextError
from services.runtime_semantica_reasoning_service import (
    RuntimeSemanticaReasoningService,
)
from services.working_context_projection_service import WorkingContextProjectionService
from storage.runtime_semantic_evaluation_store import (
    RuntimeSemanticEvaluationStore,
    temporal_publication_hash,
)


class RuntimeContextReasoningEvaluationService:
    def __init__(self, *, store, reasoning_service, projection_service):
        if (
            not isinstance(store, RuntimeSemanticEvaluationStore)
            or not isinstance(reasoning_service, RuntimeSemanticaReasoningService)
            or not isinstance(projection_service, WorkingContextProjectionService)
        ):
            raise TypeError("shared native services required")
        if (
            reasoning_service._store is not store._build_store
            or projection_service._store is not store._build_store
            or projection_service._authority is not store._authority_resolver
            or reasoning_service._authority_resolver is not store._authority_resolver
        ):
            raise RuntimeContextError("INVALID_STATE")
        self._store = store
        self._reasoner = reasoning_service
        self._projector = projection_service

    @staticmethod
    def _source_pin(cls, kind):
        import inspect
        from hashlib import sha256

        path = Path(inspect.getsourcefile(cls))
        payload = path.read_bytes()
        if kind == "reasoning_adapter":
            payload += (
                b"\\0"
                + Path(
                    inspect.getsourcefile(RuntimeSemanticaReasoningService)
                ).read_bytes()
            )
        digest = sha256(payload).hexdigest()
        return ArtifactPinV1(
            artifact_kind=kind,
            artifact_ref=f"{kind}:{digest}",
            artifact_hash=f"sha256:{digest}",
        )

    def prepare_request(
        self, *, actor_context, case, context, task_ref, goal, rules, max_dispatches=2
    ):
        organization, workspace, _ = self._store._base_access(actor_context)
        graph = self._projector._load_context(actor_context=actor_context, pin=context)
        case_graph = self._projector._load_context(
            actor_context=actor_context, pin=case
        )
        wc = self._projector.project_for_reasoning(
            actor_context=actor_context,
            context=context,
            task_ref=task_ref,
            task_goal=goal,
            selected_assertion_ids=tuple(a.assertion_id for a in graph.assertions),
            delegated_operation="CONTEXT_REASONING",
            commitment_mode="AUTO",
            formed_at=self._store._now(),
        )
        facts, _ = self._reasoner._facts(
            context=graph, assertion_ids=wc.selected_assertion_ids
        )
        target = GovernedTargetPinV1(
            schema_version="governed_target_pin.v1",
            organization_id=organization,
            workspace_id=workspace,
            artifact=case,
            target_type="graph",
            evidence=case_graph.evidence,
            scope_bridge=None,
        )
        request = RuntimeContextReasoningRequestV1.create(
            schema_version="runtime_context_reasoning_request.v1",
            organization_id=organization,
            workspace_id=workspace,
            context=context,
            world=context,
            case=case,
            world_basis=graph.world,
            intent=graph.intent,
            source_inputs=graph.formation.source_inputs,
            target=target,
            working_context=wc,
            facts=facts,
            rules=rules,
            goal=goal,
            executable=self._reasoner._engine_pin(),
            adapter=self._source_pin(type(self), "reasoning_adapter"),
            projector=self._source_pin(type(self._projector), "context_projector"),
            projection_recipe="all_declared_assertions.v1",
            max_depth=10,
            max_dispatches=max_dispatches,
        )
        self._store._build_store.put(request)
        self._request(actor_context, request.artifact_pin(), live=True)
        return request

    def _request(self, actor, pin, *, live=False):
        request = self._store.get_provider_request(actor_context=actor, pin=pin)
        if not isinstance(request, RuntimeContextReasoningRequestV1):
            raise RuntimeContextError("CONTEXT_REASONING_REQUEST_REQUIRED")
        responsibility = request.working_context.responsibility
        if (
            responsibility.responsibility_bearer_id != actor.actor_id
            or responsibility.delegated_actor_id != self._projector._runtime_actor_id
            or responsibility.delegated_operation != "CONTEXT_REASONING"
        ):
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        graph = self._projector._load_context(actor_context=actor, pin=request.context)
        self._reasoner._validate_working_context(
            working_context=request.working_context, context=graph
        )
        if live and (
            request.executable != self._reasoner._engine_pin()
            or request.adapter != self._source_pin(type(self), "reasoning_adapter")
            or request.projector
            != self._source_pin(type(self._projector), "context_projector")
        ):
            raise RuntimeContextError("EXECUTABLE_BASIS_CHANGED")
        return request

    @staticmethod
    def validate_treatment(baseline, successor):
        if baseline.working_context.ontology != successor.working_context.ontology:
            raise ValueError("ONTOLOGY_TREATMENT_UNSUPPORTED")
        protected = (
            "organization_id",
            "workspace_id",
            "case",
            "target",
            "world_basis",
            "intent",
            "source_inputs",
            "rules",
            "goal",
            "executable",
            "adapter",
            "projector",
            "projection_recipe",
            "max_depth",
            "max_dispatches",
        )
        if (
            any(
                getattr(baseline, field) != getattr(successor, field)
                for field in protected
            )
            or baseline.working_context.task_ref != successor.working_context.task_ref
        ):
            raise ValueError("PROTECTED_BASIS_CHANGED")
        return (
            "TREATMENT_OBSERVED"
            if baseline.facts != successor.facts
            else "TREATMENT_NOT_OBSERVED"
        )

    def evaluate(self, *, actor_context, request, request_id, parent_run=None):
        organization, workspace, session = self._store._base_access(actor_context)
        req = self._request(actor_context, request)
        identity = RuntimeContextRequestIdentityV2(
            schema_version="runtime_evaluation_request_identity.v2",
            organization_id=organization,
            workspace_id=workspace,
            actor_ref=actor_context.actor_id,
            operation="EVALUATE_RUNTIME_CONTEXT_REASONING",
            request_id=request_id,
            context=req.context,
            provider_request=request,
            target=req.target,
            business_effective_at=req.working_context.formed_at,
            parent_run=parent_run,
        )
        old = self._store.get_request_outcome(
            actor_context=actor_context, identity=identity
        )
        if old is not None:
            return self._store.get_run(actor_context=actor_context, pin=old.run_pin)
        self._request(actor_context, request, live=True)
        graph = self._projector._load_context(
            actor_context=actor_context, pin=req.context
        )
        root = None
        if parent_run is not None:
            parent = self._store.get_run(actor_context=actor_context, pin=parent_run)
            root = parent.root_run or parent.artifact_pin()
        now = self._store._now()
        admission = RuntimeContextExecutionAdmissionV2.create(
            schema_version="runtime_execution_admission.v2",
            purpose="CONTEXT_REASONING",
            organization_id=organization,
            workspace_id=workspace,
            actor_id=actor_context.actor_id,
            session_ref=session,
            context=req.context,
            world=req.world,
            provider_request=request,
            target=req.target,
            context_stream_ref=graph.artifact_ref,
            context_revision=graph.revision,
            certified_plan=None,
            epoch=None,
            activation_snapshot=None,
            activation_events=(),
            business_effective_at=req.working_context.formed_at,
            admitted_at=now,
        )
        run = RuntimeEvaluationRunV1.create(
            schema_version="runtime_evaluation_run.v1",
            organization_id=organization,
            workspace_id=workspace,
            admission=admission.artifact_pin(),
            root_run=root,
            parent_run=parent_run,
            request_id=request_id,
            created_by=actor_context.actor_id,
            created_at=now,
        )
        state = RuntimeEvaluationStateV2.create(
            schema_version="runtime_evaluation_state.v2",
            organization_id=organization,
            workspace_id=workspace,
            run=run.artifact_pin(),
            revision=0,
            predecessor=None,
            status="LOADING",
            capture=None,
            results=(),
            decisions=(),
            diagnostics=(),
            recorded_at=now,
            producer_mode="CONTEXT_REASONING",
            max_dispatches=req.max_dispatches,
        )
        outcome = RuntimeContextRequestOutcomeV2.create(
            schema_version="runtime_evaluation_request_outcome.v2",
            organization_id=organization,
            workspace_id=workspace,
            actor_ref=actor_context.actor_id,
            operation=identity.operation,
            request_id=request_id,
            normalized_request_hash=identity.normalized_request_hash(),
            run_pin=run.artifact_pin(),
            execution_admission_pin=admission.artifact_pin(),
            state_pin=state.artifact_pin(),
            epoch_pin=None,
            authority_release_receipt_pin=None,
            result_pins=(),
            recorded_at=now,
        )
        winner = self._store.admit(
            actor_context=actor_context,
            identity=identity,
            admission=admission,
            run=run,
            state=state,
            outcome=outcome,
            verify_before_commit=lambda connection: self._request(
                actor_context, request, live=True
            ),
        )
        stable = self._store.get_run(actor_context=actor_context, pin=winner.run_pin)
        if stable == run:
            self.resume(actor_context=actor_context, run=stable.artifact_pin())
        return stable

    def _publish(
        self,
        actor,
        previous,
        outcome,
        *,
        status,
        phase,
        artifacts=(),
        decisions=(),
        diagnostics=(),
    ):
        now = self._store._now()
        values = {
            k: getattr(previous, k)
            for k in type(previous).model_fields
            if k not in {"state_ref", "state_hash"}
        }
        values.update(
            revision=previous.revision + 1,
            predecessor=previous.artifact_pin(),
            status=status,
            decisions=decisions,
            diagnostics=diagnostics,
            recorded_at=now,
        )
        state = RuntimeEvaluationStateV2.create(**values)
        values = {
            k: getattr(outcome, k)
            for k in type(outcome).model_fields
            if k not in {"outcome_ref", "outcome_hash"}
        }
        values.update(state_pin=state.artifact_pin(), recorded_at=now)
        updated = type(outcome).create(**values)
        admission = self._store.get_admission(
            actor_context=actor, pin=outcome.execution_admission_pin
        )
        inputs = (
            (previous.capture,)
            if phase == "complete"
            else (admission.provider_request,)
        )
        published = self._store.publish(
            actor_context=actor,
            previous=previous,
            previous_outcome=outcome,
            state=state,
            outcome=updated,
            artifacts=artifacts,
            phase=phase,
            request_hash=temporal_publication_hash(
                actor_id=actor.actor_id, state=previous, phase=phase, inputs=inputs
            ),
        )
        return self._store.get_state(actor_context=actor, pin=published.state_pin)

    @staticmethod
    def _native_json(value):
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {
                key: RuntimeContextReasoningEvaluationService._native_json(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                RuntimeContextReasoningEvaluationService._native_json(item)
                for item in value
            ]
        return value

    def _decision_from_capture(self, *, request, capture):
        graph = self._reasoner._context(request.context)
        wc = request.working_context
        native = capture.native_output
        _, index = self._reasoner._facts(
            context=graph, assertion_ids=wc.selected_assertion_ids
        )
        if native is not None:
            native = SimpleNamespace(**native)
            if native.rule_used is not None:
                native.rule_used = SimpleNamespace(**native.rule_used)
            return self._reasoner._decision(
                mode="backward",
                context=graph,
                working_context=wc,
                native=native,
                explanation="Retained native backward proof.",
                rules=request.rules,
                fact_index=index,
                recorded_at=capture.captured_at,
                observation=capture.artifact_pin(),
                request=request.artifact_pin(),
                executable=request.executable,
                persist=False,
            )
        rules = tuple(
            self._reasoner._pin("reasoning_rule", "semantica-rule", {"exact_rule": r})
            for r in request.rules
        )
        invocation = ReasonerInvocation(
            invocation_id=f"semantica-invocation:{content_hash(capture.model_dump(mode='json'))}",
            strategy="semantica_backward",
            executable_engine=request.executable,
            rules=rules,
            inputs=(
                request.context,
                wc.pin(),
                request.artifact_pin(),
                capture.artifact_pin(),
                *wc.evidence,
            ),
            outputs=("UNKNOWN",),
            proof_steps=(f"no-proof:{request.goal}",),
            evidence=wc.evidence,
        )
        trace = ReasoningTrace(
            strategy="semantica_backward",
            reasoning_version=request.executable.artifact_hash,
            invocations=(invocation,),
            working_context=wc,
            derived_assertion_ids=(),
            diagnostics=(f"NO_PROOF:{request.goal}",),
        )
        return DecisionRecord.create(
            schema_version="semantier.decision_record.v1",
            organization_id=request.organization_id,
            workspace_id=request.workspace_id,
            purpose="INFERENCE",
            subject=request.context,
            context=request.context,
            ontology=wc.ontology,
            result_code="UNKNOWN",
            epistemic_status="UNKNOWN",
            evidence=wc.evidence,
            admissibility=None,
            reasoning=trace,
            actor_id=wc.responsibility.delegated_actor_id,
            reasoner_identity=request.executable.artifact_ref,
            authorization=None,
            outcome_observation=capture.artifact_pin(),
            recorded_at=capture.captured_at,
        )

    def _fail(self, actor, stable_run, request, code, *, attempt_id=None):
        organization, workspace, _ = self._store._base_access(actor)
        state, outcome = self._store._validate_projection(
            organization, workspace, stable_run
        )
        if state.status in {"COMPLETE", "FAILED"}:
            return state
        if attempt_id is not None:
            registrations = [
                event
                for event in state.attempt_events
                if event.kind == "DISPATCH_REGISTERED"
            ]
            if not registrations or registrations[-1].attempt_id != attempt_id:
                return state
        diagnostic = RuntimeEvaluationDiagnosticV1.create(
            schema_version="runtime_evaluation_diagnostic.v1",
            organization_id=organization,
            workspace_id=workspace,
            run=stable_run.artifact_pin(),
            code=code,
            message="The context reasoning producer did not complete.",
            input_pins=(request.artifact_pin(),),
            evidence_pins=(() if state.capture is None else (state.capture,)),
            recorded_at=self._store._now(),
        )
        return self._publish(
            actor,
            state,
            outcome,
            status="FAILED",
            phase="fail",
            artifacts=(diagnostic,),
            diagnostics=(*state.diagnostics, diagnostic.artifact_pin()),
        )

    def resume(self, *, actor_context, run):
        stable = self._store.get_run(actor_context=actor_context, pin=run)
        admission = self._store.get_admission(
            actor_context=actor_context, pin=stable.admission
        )
        if not isinstance(admission, RuntimeContextExecutionAdmissionV2):
            raise RuntimeContextError("CONTEXT_REASONING_ADMISSION_REQUIRED")
        while True:
            organization, workspace, _ = self._store._base_access(actor_context)
            state, outcome = self._store._validate_projection(
                organization, workspace, stable
            )
            if state.status in {"COMPLETE", "FAILED"}:
                return state
            req = self._request(actor_context, admission.provider_request)
            if state.status == "LOADING":
                self._publish(
                    actor_context, state, outcome, status="RUNNING", phase="start"
                )
                continue
            if state.capture is None:
                self._request(actor_context, req.artifact_pin(), live=True)
                try:
                    state, attempt = self._store.register_attempt(
                        actor_context=actor_context, run=run
                    )
                except RuntimeContextError as exc:
                    if "DISPATCH_BUDGET_EXCEEDED" not in str(exc):
                        raise
                    return self._fail(
                        actor_context, stable, req, "DISPATCH_BUDGET_EXCEEDED"
                    )
                if attempt is None:
                    return state
                self._store._base_access(actor_context)
                try:
                    native = self._reasoner._execute_backward(
                        facts=req.facts,
                        rules=req.rules,
                        goal=req.goal,
                        max_depth=req.max_depth,
                    )
                except (RuntimeError, ValueError, TypeError, KeyError, IndexError):
                    self._store.record_attempt_exception(
                        actor_context=actor_context,
                        run=run,
                        attempt=attempt,
                        code="CONTEXT_REASONING_CALL_FAILED",
                    )
                    return self._fail(
                        actor_context,
                        stable,
                        req,
                        "CONTEXT_REASONING_CALL_FAILED",
                        attempt_id=attempt.attempt_id,
                    )
                try:
                    capture = RuntimeContextReasoningCaptureV1.create(
                        schema_version="runtime_context_reasoning_capture.v1",
                        organization_id=organization,
                        workspace_id=workspace,
                        run=run,
                        provider_request=req.artifact_pin(),
                        attempt_id=attempt.attempt_id,
                        raw_output=_captured(
                            None
                            if native is None
                            else self._native_json(asdict(native))
                        ),
                        captured_at=self._store._now(),
                    )
                except (RuntimeError, ValueError, TypeError, KeyError, IndexError):
                    # The native call may have returned; a serializer failure is
                    # apparatus failure, never proof of an actor call exception.
                    return self._fail(
                        actor_context,
                        stable,
                        req,
                        "CONTEXT_REASONING_CAPTURE_INVALID",
                        attempt_id=attempt.attempt_id,
                    )
                self._store.capture_attempt(
                    actor_context=actor_context,
                    run=run,
                    attempt=attempt,
                    capture=capture,
                )
                continue
            capture = self._store.get_capture(
                actor_context=actor_context, pin=state.capture
            )
            if req.adapter != self._source_pin(type(self), "reasoning_adapter"):
                raise RuntimeContextError("CONVERSION_BASIS_CHANGED")
            try:
                decision = self._decision_from_capture(request=req, capture=capture)
            except (RuntimeError, ValueError, TypeError, KeyError, IndexError):
                return self._fail(
                    actor_context, stable, req, "CONTEXT_REASONING_CONVERSION_FAILED"
                )
            return self._publish(
                actor_context,
                state,
                outcome,
                status="COMPLETE",
                phase="complete",
                artifacts=(decision,),
                decisions=(decision.pin(),),
            )

    def replay(self, *, actor_context, run):
        stable = self._store.get_run(actor_context=actor_context, pin=run)
        state = self._store.get_current_state(actor_context=actor_context, run=run)
        admission = self._store.get_admission(
            actor_context=actor_context, pin=stable.admission
        )
        request = self._store.get_provider_request(
            actor_context=actor_context, pin=admission.provider_request
        )
        capture = (
            None
            if state.capture is None
            else self._store.get_capture(actor_context=actor_context, pin=state.capture)
        )
        decision = (
            None
            if not state.decisions
            else self._store._build_store.get(
                organization_id=state.organization_id,
                workspace_id=state.workspace_id,
                artifact_ref=state.decisions[0].artifact_ref,
                contract_type=DecisionRecord,
            )
        )
        return {
            "request": request,
            "state": state,
            "capture": capture,
            "decision": decision,
        }

    @staticmethod
    def reference_environment():
        import sys
        import pydantic

        return (("python", sys.version), ("pydantic", pydantic.__version__))

    def reference_panel(self, *, actor_context, pin):
        organization, workspace, _ = self._store._base_access(actor_context)
        panel = self._store._read(organization, workspace, pin, RuntimeContextReferencePanelV1)
        if (panel.scorer != self._source_pin(type(self), "context_reference_scorer")
                or panel.scorer_source != Path(__file__).read_text()
                or panel.scorer_environment != self.reference_environment()):
            raise ValueError("SCORER_BASIS_CHANGED")
        import inspect
        from services.change_candidate_comparison_service import ChangeCandidateComparisonService

        if (panel.comparison_policy != self._source_pin(ChangeCandidateComparisonService, "context_comparison_policy")
                or panel.comparison_source != Path(inspect.getsourcefile(ChangeCandidateComparisonService)).read_text()):
            raise ValueError("COMPARISON_CODE_BASIS_CHANGED")
        first = self._request(actor_context, panel.matched.baseline_request)
        second = self._request(actor_context, panel.holdout.baseline_request)
        for slot in ("rules", "goal", "executable", "adapter", "projector", "projection_recipe",
                     "max_depth", "max_dispatches"):
            if getattr(first, slot) != getattr(second, slot):
                raise ValueError("PROTECTED_SCORER_BASIS_CHANGED")
        for case in (panel.matched, panel.holdout):
            baseline = self._request(actor_context, case.baseline_request)
            successor = self._request(actor_context, case.successor_request)
            if baseline.case != case.case or successor.case != case.case:
                raise ValueError("REFERENCE_CASE_MISMATCH")
            self.validate_treatment(baseline, successor)
            reference = self._store._build_store.get(
                organization_id=organization, workspace_id=workspace,
                artifact_ref=case.reference.artifact_ref, contract_type=Evidence)
            if reference is None or reference.pin() != case.reference:
                raise ValueError("REFERENCE_BASIS_MISSING")
            import json
            from hashlib import sha256

            quote = reference.selector.exact_quote
            if (reference.evidence_role != "independent_development_reference"
                    or reference.source_hash != "sha256:" + sha256(quote.encode()).hexdigest()
                    or reference.observed_at > panel.frozen_at
                    or reference.recorded_at > panel.frozen_at
                    or json.loads(quote) != {
                        "case": case.case.model_dump(mode="json"),
                        "goal": baseline.goal, "expected_native": case.expected_native,
                        "baseline_request": case.baseline_request.model_dump(mode="json"),
                        "successor_request": case.successor_request.model_dump(mode="json"),
                    }
                    or case.reference in baseline.working_context.evidence
                    or case.reference in successor.working_context.evidence):
                raise ValueError("INDEPENDENT_REFERENCE_MISMATCH")
        return panel

    def grade(self, *, actor_context, panel_pin, run, persist=True):
        """Grade retained native computation against an independent frozen oracle.

        No producer dispatch, parser, source lookup or executable discovery occurs.
        The caller must retain this panel before the first dispatch in every arm.
        """
        panel = self.reference_panel(actor_context=actor_context, pin=panel_pin)
        retained = self.replay(actor_context=actor_context, run=run)
        request, state = retained["request"], retained["state"]
        capture, inference = retained["capture"], retained["decision"]
        case = next((case for case in (panel.matched, panel.holdout)
                     if request.artifact_pin() in (case.baseline_request, case.successor_request)), None)
        if case is None:
            raise ValueError("UNRELATED_PRODUCER_RUN")
        if any(event.recorded_at < panel.frozen_at for event in state.attempt_events):
            raise ValueError("REFERENCE_NOT_FROZEN_BEFORE_DISPATCH")
        report = self._store.attempt_report(actor_context=actor_context, run=run)
        if (state.status != "COMPLETE" or report["trial_validity"] != "VALID"
                or capture is None or inference is None):
            raise ValueError("INCOMPLETE_PRODUCER_EVIDENCE")
        if (inference.purpose != "INFERENCE" or inference.context != request.context
                or inference.outcome_observation != capture.artifact_pin()
                or capture.run != run or capture.provider_request != request.artifact_pin()):
            raise ValueError("PRODUCER_PROJECTION_MISMATCH")
        # Verify the exact semantic projection using retained bytes and the pinned
        # conversion recipe; never ask the native engine for another result.
        expected_inference = self._decision_from_capture(request=request, capture=capture)
        if inference != expected_inference:
            raise ValueError("PRODUCER_PROJECTION_MISMATCH")
        native = capture.native_output
        if native is not None and native.get("conclusion") != request.goal:
            raise ValueError("NATIVE_GOAL_MISMATCH")
        observed = "UNKNOWN" if native is None else "PROOF"
        result = "MATCH" if observed == case.expected_native else "NO_MATCH"
        evidence = tuple(sorted((*inference.evidence, case.reference),
                                key=lambda pin: (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)))
        inputs = (panel_pin, run, state.artifact_pin(), request.artifact_pin(),
                  capture.artifact_pin(), inference.pin(), case.reference)
        invocation = ReasonerInvocation(
            invocation_id="context-independent-reference-evaluation",
            strategy=panel.policy, executable_engine=panel.scorer,
            rules=(panel.comparison_policy,), inputs=inputs, outputs=(result,),
            proof_steps=(f"native={observed}", f"reference={case.expected_native}"),
            evidence=evidence)
        decision = DecisionRecord.create(
            schema_version="semantier.decision_record.v1",
            organization_id=panel.organization_id, workspace_id=panel.workspace_id,
            purpose="EVALUATION", subject=case.case, context=request.context,
            ontology=request.working_context.ontology, result_code=result,
            epistemic_status="KNOWN", evidence=evidence, admissibility=None,
            reasoning=ReasoningTrace(strategy=panel.policy, reasoning_version=panel.scorer.artifact_hash,
                                    invocations=(invocation,), derived_assertion_ids=(), diagnostics=()),
            actor_id=actor_context.actor_id, reasoner_identity=panel.scorer.artifact_ref,
            authorization=None, outcome_observation=run, recorded_at=capture.captured_at)
        if persist:
            self._store._build_store.put(decision)
        return decision

    def panel_report(self, *, actor_context, proposal, runs):
        """Reduce this finite two-case panel without dispatch or persistence."""
        if len(runs) != 4:
            raise ValueError("FOUR_PLANNED_ARM_SLOTS_REQUIRED")
        organization, workspace, _ = self._store._base_access(actor_context)
        from contracts.semantic_kernel import ChangeCandidate
        from services.change_candidate_comparison_service import ChangeCandidateComparisonService

        candidate = self._store._build_store.get(organization_id=organization, workspace_id=workspace,
            artifact_ref=proposal.artifact_ref, contract_type=ChangeCandidate)
        if candidate is None or candidate.pin() != proposal:
            raise ValueError("PROPOSAL_REQUIRED")
        panel = self._store._read(organization, workspace,
            candidate.evaluation_basis.obligation_suite, RuntimeContextReferencePanelV1)
        requests = (panel.matched.baseline_request, panel.matched.successor_request,
                    panel.holdout.baseline_request, panel.holdout.successor_request)
        rows, grades = [], []
        for index, (run, expected_request) in enumerate(zip(runs, requests)):
            row = {"case": "matched" if index < 2 else "holdout",
                   "arm": "baseline" if index % 2 == 0 else "successor",
                   "request": expected_request.model_dump(mode="json"),
                   "run": None if run is None else run.model_dump(mode="json"),
                   "status": "UNATTEMPTED", "scorable": False,
                   "task_result": "UNMEASURED", "epistemic_status": None,
                   "cause": None, "accounting": None, "evaluation": None}
            grade = None
            if run is not None:
                try:
                    retained = self.replay(actor_context=actor_context, run=run)
                    if retained["request"].artifact_pin() != expected_request:
                        raise ValueError("UNRELATED_PRODUCER_RUN")
                    row["status"] = retained["state"].status
                    row["accounting"] = self._store.attempt_report(actor_context=actor_context, run=run)
                    if retained["decision"] is not None:
                        row["epistemic_status"] = retained["decision"].epistemic_status
                    grade = self.grade(actor_context=actor_context, panel_pin=panel.artifact_pin(), run=run, persist=False)
                    row.update(scorable=True, task_result="CORRECT" if grade.result_code == "MATCH" else "INCORRECT",
                               evaluation=grade.pin().model_dump(mode="json"))
                except ValueError as exc:
                    row["cause"] = str(exc)
            rows.append(row)
            grades.append(grade)
        verdict, comparison, cause = "INDETERMINATE", None, None
        comparable = 0
        if all(grade is not None for grade in grades):
            successor_context = self._reasoner._context(grades[1].context)
            try:
                compared = ChangeCandidateComparisonService(store=self._store._build_store,
                    context_evaluator=self, actor_context=actor_context).compare(
                        proposal_pin=proposal, matched_baseline_decision_pin=grades[0].pin(),
                        matched_successor_decision_pin=grades[1].pin(),holdout_baseline_decision_pin=grades[2].pin(),
                        holdout_successor_decision_pin=grades[3].pin(),
                        successor_input_pin=successor_context.formation.formation_policy,
                        actor_id=actor_context.actor_id,persist=False, retained_evaluations=tuple(grades))
                verdict, comparison, comparable = compared.result_code, compared.pin().model_dump(mode="json"), 2
            except ValueError as exc:
                cause = str(exc)
        return {"planned_cases": 2, "planned_arms": 4,
                "attempted_arms": sum(row["run"] is not None for row in rows),
                "completed_arms": sum(row["status"] == "COMPLETE" for row in rows),
                "scorable_arms": sum(row["scorable"] for row in rows),
                "scorable_cases": sum(all(row["scorable"] for row in rows[i:i+2]) for i in (0,2)),
                "comparable_cases": comparable,
                "correct_arms": sum(row["task_result"] == "CORRECT" for row in rows),
                "incorrect_arms": sum(row["task_result"] == "INCORRECT" for row in rows),
                "unknown_arms": sum(row["epistemic_status"] == "UNKNOWN" for row in rows),
                "unmeasured_arms": sum(not row["scorable"] for row in rows),
                "critical_violations": int(verdict == "WORSE"),
                "verdict": verdict, "comparison": comparison, "cause": cause,
                "physical_execution_count": None, "cost": None, "rows": rows}
