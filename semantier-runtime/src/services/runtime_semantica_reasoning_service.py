"""Kernel-native Semantica reasoning persisted as DecisionRecord.

No reasoning database or derivation artifact family is owned here. Facts come from one
exact admitted ContextGraph; native proof lineage, exact rule text, executable bytes and
Evidence pins are embedded into the immutable DecisionRecord.
"""

from __future__ import annotations

from storage.hosted_postgres import sql_query,begin_write,in_transaction

from datetime import datetime, timezone
from hashlib import sha256
import inspect
from pathlib import Path

from contracts.knowledge_classification import RuntimeContextDTO
from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import (
    ContextAssertion,
    ContextGraph,
    DecisionRecord,
    Evidence,
    ReasonerInvocation,
    ReasoningTrace,
    WorkingContext,
)
from eos.artifact_hashing import content_hash
from semantier.knowledge_feature_packs.semantica_adapter import (
    _ensure_vendored_semantica_importable,
)
from services.runtime_context_transition_service import RuntimeAuthorityResolver
from storage.ontology_build_store import OntologyBuildStore


class RuntimeSemanticaReasoningService:
    """Execute the vendored Semantica reasoner against an admitted ContextGraph."""

    def __init__(
        self,
        *,
        build_store: OntologyBuildStore,
        authority_resolver: RuntimeAuthorityResolver | None = None,
    ) -> None:
        if not isinstance(build_store, OntologyBuildStore):
            raise TypeError("reasoning requires the shared kernel store")
        if authority_resolver is not None and not isinstance(
            authority_resolver, RuntimeAuthorityResolver
        ):
            raise TypeError("replay access verification requires the runtime authority resolver")
        self._store = build_store
        self._authority_resolver = authority_resolver

    @staticmethod
    def _pin(kind: str, prefix: str, payload: object) -> ArtifactPinV1:
        digest = content_hash(payload)
        return ArtifactPinV1(
            artifact_kind=kind,
            artifact_ref=f"{prefix}:{digest}",
            artifact_hash=f"sha256:{digest}",
        )

    @staticmethod
    def _engine_pin() -> ArtifactPinV1:
        """Bind actual vendored reasoning source bytes, never a version label."""

        _ensure_vendored_semantica_importable()
        from semantica.reasoning import ExplanationGenerator, Reasoner

        sources = (inspect.getsourcefile(Reasoner), inspect.getsourcefile(ExplanationGenerator))
        if any(value is None for value in sources):
            raise ValueError("SEMANTICA_IMPLEMENTATION_UNRESOLVABLE")
        roots = {Path(value).resolve().parent for value in sources if value is not None}
        files = sorted({path for root in roots for path in root.rglob("*.py") if path.is_file()})
        if not files:
            raise ValueError("SEMANTICA_IMPLEMENTATION_UNRESOLVABLE")
        digest = sha256()
        for path in files:
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        value = digest.hexdigest()
        return ArtifactPinV1(
            artifact_kind="reasoner_executable",
            artifact_ref=f"semantica-reasoner:{value}",
            artifact_hash=f"sha256:{value}",
        )

    def _context(self, pin: ArtifactPinV1) -> ContextGraph:
        if pin.artifact_kind != "context_graph":
            raise ValueError("CONTEXT_GRAPH_REQUIRED")
        row = self._store._connection.execute(
            sql_query("SELECT organization_id,workspace_id,artifact_hash FROM ontology_build_artifacts WHERE artifact_ref=?",'SELECT organization_id,workspace_id,artifact_hash FROM ontology_build_artifacts WHERE artifact_ref=%s'),
            (pin.artifact_ref,),
        ).fetchone()
        if row is None:
            raise ValueError("CONTEXT_NOT_FOUND")
        if row[2] != pin.artifact_hash:
            raise ValueError("HASH_MISMATCH")
        context = self._store.get(
            organization_id=str(row[0]),
            workspace_id=str(row[1]),
            artifact_ref=pin.artifact_ref,
            contract_type=ContextGraph,
        )
        if context is None or context.pin() != pin:
            raise ValueError("HASH_MISMATCH")
        if context.state != "ADMITTED":
            raise ValueError("ADMITTED_CONTEXT_REQUIRED")
        return context

    def _validate_working_context(
        self,
        *,
        working_context: WorkingContext,
        context: ContextGraph,
    ) -> tuple[ContextAssertion, ...]:
        from services.working_context_projection_service import WorkingContextProjectionService

        if working_context.context_graph != context.pin():
            raise ValueError("WORKING_CONTEXT_SOURCE_MISMATCH")
        if (working_context.organization_id, working_context.workspace_id) != (
            context.organization_id,
            context.workspace_id,
        ):
            raise ValueError("WORKING_CONTEXT_SCOPE_MISMATCH")
        if working_context.ontology != context.ontology:
            raise ValueError("WORKING_CONTEXT_ONTOLOGY_MISMATCH")
        if working_context.selection_policy != WorkingContextProjectionService.selection_policy():
            raise ValueError("UNTRUSTED_WORKING_CONTEXT_SELECTOR")

        nodes = {item.node_id for item in context.nodes}
        edges = {item.edge_id: item for item in context.edges}
        assertions = {item.assertion_id: item for item in context.assertions}
        if any(item not in nodes for item in working_context.selected_node_ids):
            raise ValueError("UNKNOWN_NODE_SELECTION")
        if any(item not in edges for item in working_context.selected_edge_ids):
            raise ValueError("UNKNOWN_EDGE_SELECTION")
        if any(item not in assertions for item in working_context.selected_assertion_ids):
            raise ValueError("UNKNOWN_ASSERTION_SELECTION")
        selected_node_set = set(working_context.selected_node_ids)
        for edge_id in working_context.selected_edge_ids:
            edge = edges[edge_id]
            if edge.source_id not in selected_node_set or edge.target_id not in selected_node_set:
                raise ValueError("SELECTED_EDGE_ENDPOINTS_REQUIRED")

        selected_assertions = tuple(
            assertions[item] for item in working_context.selected_assertion_ids
        )
        expected_evidence = self._verify_evidence(
            context=context,
            assertions=selected_assertions,
        )
        if expected_evidence != working_context.evidence:
            raise ValueError("WORKING_CONTEXT_EVIDENCE_MISMATCH")
        return selected_assertions

    @staticmethod
    def _fact(assertion: ContextAssertion) -> str:
        return f"{assertion.predicate}({assertion.subject_ref}, {assertion.object_ref})"

    def _facts(
        self,
        *,
        context: ContextGraph,
        assertion_ids: tuple[str, ...] | None,
    ) -> tuple[tuple[str, ...], dict[str, ContextAssertion]]:
        by_id = {item.assertion_id: item for item in context.assertions}
        assertions = tuple(context.assertions)
        if assertion_ids is not None:
            if len(assertion_ids) != len(set(assertion_ids)):
                raise ValueError("DUPLICATE_ASSERTION_SELECTION")
            if any(item not in by_id for item in assertion_ids):
                raise ValueError("UNKNOWN_ASSERTION_SELECTION")
            assertions = tuple(by_id[item] for item in assertion_ids)
        if not assertions:
            raise ValueError("REASONING_REQUIRES_CONTEXT_FACTS")
        fact_index = {self._fact(item): item for item in assertions}
        if len(fact_index) != len(assertions):
            raise ValueError("AMBIGUOUS_CONTEXT_FACTS")
        return tuple(fact_index), fact_index

    def _verify_evidence(
        self,
        *,
        context: ContextGraph,
        assertions: tuple[ContextAssertion, ...],
    ) -> tuple[ArtifactPinV1, ...]:
        context_keys = {
            (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash) for pin in context.evidence
        }
        selected = {
            (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash): pin
            for assertion in assertions
            for pin in assertion.evidence
        }
        if not selected or not set(selected) <= context_keys:
            raise ValueError("UNBOUND_REASONING_EVIDENCE")
        for pin in selected.values():
            evidence = self._store.get(
                organization_id=context.organization_id,
                workspace_id=context.workspace_id,
                artifact_ref=pin.artifact_ref,
                contract_type=Evidence,
            )
            if evidence is None or evidence.pin() != pin:
                raise ValueError("UNRESOLVABLE_REASONING_EVIDENCE")
        return tuple(selected[key] for key in sorted(selected))

    @staticmethod
    def _native_rule_text(native) -> str:
        if native.rule_used is None:
            raise ValueError("reasoning result requires a native rule")
        return (
            "IF "
            + " AND ".join(str(item) for item in native.rule_used.conditions)
            + " THEN "
            + str(native.rule_used.conclusion)
        )

    @staticmethod
    def _exact_rule_text(native, rules: tuple[str, ...]) -> str:
        if native.rule_used is None:
            raise ValueError("reasoning result requires a native rule")
        rule_id = str(native.rule_used.rule_id)
        if rule_id.startswith("rule_") and rule_id[5:].isdigit():
            index = int(rule_id[5:]) - 1
            if 0 <= index < len(rules):
                return rules[index]
        native_text = RuntimeSemanticaReasoningService._native_rule_text(native)
        for rule in rules:
            if " ".join(rule.split()) == " ".join(native_text.split()):
                return rule
        raise ValueError("SEMANTICA_RULE_IDENTITY_UNRESOLVABLE")

    def _decision(
        self,
        *,
        mode: str,
        context: ContextGraph,
        working_context: WorkingContext,
        native,
        explanation: str,
        rules: tuple[str, ...],
        fact_index: dict[str, ContextAssertion],
        recorded_at: datetime,
        observation: ArtifactPinV1 | None = None,
        request: ArtifactPinV1 | None = None,
        executable: ArtifactPinV1 | None = None,
        persist: bool = True,
    ) -> DecisionRecord:
        conclusion = str(native.conclusion)
        premises = tuple(str(item) for item in native.premises)
        premise_assertions = tuple(fact_index[item] for item in premises if item in fact_index)
        if len(premise_assertions) != len(premises):
            raise ValueError("UNBOUND_REASONING_PREMISE")
        direct_fact = native.rule_used is None and conclusion in fact_index
        if native.rule_used is None and not direct_fact:
            raise ValueError("UNBOUND_DIRECT_REASONING_RESULT")
        if direct_fact and not premise_assertions:
            premise_assertions = (fact_index[conclusion],)
        evidence = self._verify_evidence(context=context, assertions=premise_assertions)
        rule_text = None if direct_fact else self._exact_rule_text(native, rules)
        rule_pin = None if direct_fact else self._pin(
            "reasoning_rule", "semantica-rule", {"exact_rule": rule_text}
        )
        engine = executable or self._engine_pin()
        invocation_basis = {
            "mode": mode,
            "context": context.pin().model_dump(mode="json"),
            "working_context": working_context.pin().model_dump(mode="json"),
            "engine": engine.model_dump(mode="json"),
            "rule": None if rule_pin is None else rule_pin.model_dump(mode="json"),
            "premises": premises,
            "conclusion": conclusion,
        }
        invocation = ReasonerInvocation(
            invocation_id=f"semantica-invocation:{content_hash(invocation_basis)}",
            strategy=f"semantica_{mode}",
            executable_engine=engine,
            rules=() if rule_pin is None else (rule_pin,),
            inputs=(context.pin(), working_context.pin(), *evidence,
                    *((request,) if request is not None else ()),
                    *((observation,) if observation is not None else ())),
            outputs=(conclusion,),
            proof_steps=tuple(
                [
                    *(f"premise:{item}" for item in premises),
                    f"fact:{conclusion}" if direct_fact else f"rule:{rule_text}",
                    f"explanation:{explanation}",
                ]
            ),
            evidence=evidence,
        )
        trace = ReasoningTrace(
            strategy=f"semantica_{mode}",
            reasoning_version=engine.artifact_hash,
            invocations=(invocation,),
            working_context=working_context,
            derived_assertion_ids=(
                f"derived-assertion:{content_hash(invocation_basis)}",
            ),
            diagnostics=(),
        )
        decision = DecisionRecord.create(
            schema_version="semantier.decision_record.v1",
            organization_id=context.organization_id,
            workspace_id=context.workspace_id,
            purpose="INFERENCE",
            subject=context.pin(),
            context=context.pin(),
            ontology=context.ontology,
            result_code=conclusion,
            epistemic_status="KNOWN",
            evidence=evidence,
            admissibility=None,
            reasoning=trace,
            actor_id=working_context.responsibility.delegated_actor_id,
            reasoner_identity=engine.artifact_ref,
            authorization=None,
            outcome_observation=observation,
            recorded_at=recorded_at.astimezone(timezone.utc),
        )
        if persist:
            self._store.put(decision)
        return decision

    def forward(
        self,
        *,
        working_context: WorkingContext,
        rules: tuple[str, ...],
        recorded_at: datetime,
    ) -> tuple[DecisionRecord, ...]:
        _ensure_vendored_semantica_importable()
        from semantica.reasoning import ExplanationGenerator, Reasoner

        graph = self._context(working_context.context_graph)
        selected_assertions = self._validate_working_context(
            working_context=working_context,
            context=graph,
        )
        facts, fact_index = self._facts(
            context=graph,
            assertion_ids=tuple(item.assertion_id for item in selected_assertions),
        )
        reasoner = Reasoner()
        for rule in rules:
            reasoner.add_rule(rule)
        for fact in facts:
            reasoner.add_fact(fact)
        explainer = ExplanationGenerator()
        decisions = tuple(
            self._decision(
                mode="forward",
                context=graph,
                working_context=working_context,
                native=native,
                explanation=explainer.generate_explanation(native).natural_language,
                rules=rules,
                fact_index=fact_index,
                recorded_at=recorded_at,
            )
            for native in reasoner.forward_chain()
        )
        return tuple(sorted(decisions, key=lambda item: item.result_code))

    @staticmethod
    def _execute_backward(*, facts: tuple[str, ...], rules: tuple[str, ...], goal: str, max_depth: int = 10):
        """The one native backward computation boundary; captures happen outside it."""
        _ensure_vendored_semantica_importable()
        from semantica.reasoning import Reasoner
        reasoner = Reasoner()
        for rule in rules:
            reasoner.add_rule(rule)
        for fact in facts:
            reasoner.add_fact(fact)
        return reasoner.backward_chain(goal, max_depth=max_depth)

    def backward(
        self,
        *,
        working_context: WorkingContext,
        rules: tuple[str, ...],
        recorded_at: datetime,
    ) -> DecisionRecord:
        _ensure_vendored_semantica_importable()
        from semantica.reasoning import ExplanationGenerator

        graph = self._context(working_context.context_graph)
        selected_assertions = self._validate_working_context(
            working_context=working_context,
            context=graph,
        )
        facts, fact_index = self._facts(
            context=graph,
            assertion_ids=tuple(item.assertion_id for item in selected_assertions),
        )
        evidence = self._verify_evidence(
            context=graph,
            assertions=selected_assertions,
        )
        goal = working_context.task_goal
        native = self._execute_backward(facts=facts, rules=rules, goal=goal)
        if native is not None:
            return self._decision(
                mode="backward",
                context=graph,
                working_context=working_context,
                native=native,
                explanation=ExplanationGenerator().generate_explanation(native).natural_language,
                rules=rules,
                fact_index=fact_index,
                recorded_at=recorded_at,
            )

        engine = self._engine_pin()
        rule_pins = tuple(
            self._pin("reasoning_rule", "semantica-rule", {"exact_rule": rule})
            for rule in rules
        )
        invocation_basis = {
            "mode": "backward",
            "context": graph.pin().model_dump(mode="json"),
            "working_context": working_context.pin().model_dump(mode="json"),
            "engine": engine.model_dump(mode="json"),
            "rules": [pin.model_dump(mode="json") for pin in rule_pins],
            "facts": facts,
            "goal": goal,
            "result": "UNKNOWN",
        }
        invocation = ReasonerInvocation(
            invocation_id=f"semantica-invocation:{content_hash(invocation_basis)}",
            strategy="semantica_backward",
            executable_engine=engine,
            rules=rule_pins,
            inputs=(graph.pin(), working_context.pin(), *evidence),
            outputs=(goal,),
            proof_steps=(f"no-proof:{goal}",),
            evidence=evidence,
        )
        trace = ReasoningTrace(
            strategy="semantica_backward",
            reasoning_version=engine.artifact_hash,
            invocations=(invocation,),
            working_context=working_context,
            derived_assertion_ids=(),
            diagnostics=(f"NO_PROOF:{goal}",),
        )
        decision = DecisionRecord.create(
            schema_version="semantier.decision_record.v1",
            organization_id=graph.organization_id,
            workspace_id=graph.workspace_id,
            purpose="INFERENCE",
            subject=graph.pin(),
            context=graph.pin(),
            ontology=graph.ontology,
            result_code="UNKNOWN",
            epistemic_status="UNKNOWN",
            evidence=evidence,
            admissibility=None,
            reasoning=trace,
            actor_id=working_context.responsibility.delegated_actor_id,
            reasoner_identity=engine.artifact_ref,
            authorization=None,
            outcome_observation=None,
            recorded_at=recorded_at.astimezone(timezone.utc),
        )
        self._store.put(decision)
        return decision

    def verify_replay(
        self,
        decision: DecisionRecord,
        *,
        actor_context: RuntimeContextDTO,
    ) -> WorkingContext:
        """Verify the recorded bounded reasoning basis without live reselection."""

        if self._authority_resolver is None:
            raise ValueError("REPLAY_AUTHORITY_RESOLVER_REQUIRED")
        self._authority_resolver.require_current_actor(actor_context=actor_context)
        if (
            decision.organization_id,
            decision.workspace_id,
        ) != (
            actor_context.organization_id,
            actor_context.workspace_id,
        ):
            raise ValueError("REPLAY_SCOPE_MISMATCH")
        if decision.purpose != "INFERENCE":
            raise ValueError("REPLAY_INFERENCE_DECISION_REQUIRED")
        working_context = decision.reasoning.working_context
        if working_context is None:
            raise ValueError("REPLAY_WORKING_CONTEXT_REQUIRED")
        graph = self._context(working_context.context_graph)
        self._validate_working_context(
            working_context=working_context,
            context=graph,
        )
        if decision.context != graph.pin() or decision.ontology != graph.ontology:
            raise ValueError("REPLAY_CONTEXT_MISMATCH")
        if decision.actor_id != working_context.responsibility.delegated_actor_id:
            raise ValueError("REPLAY_ACTOR_MISMATCH")
        for invocation in decision.reasoning.invocations:
            if working_context.pin() not in invocation.inputs:
                raise ValueError("REPLAY_WORKING_CONTEXT_INPUT_MISSING")
            if graph.pin() not in invocation.inputs:
                raise ValueError("REPLAY_CONTEXT_INPUT_MISSING")
        if any(pin not in working_context.evidence for pin in decision.evidence):
            raise ValueError("REPLAY_EVIDENCE_OUTSIDE_WORKING_CONTEXT")
        return working_context


__all__ = ["RuntimeSemanticaReasoningService"]
