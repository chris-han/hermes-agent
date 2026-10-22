"""Typed competency proposal and governed successor decisions."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from contracts.ontology_building import (
    CanonicalEvidenceBoundaryV1,
    CompetencyContractV1,
    CompetencyQuestionV1,
    ConversationDecisionV1,
)
from eos.artifact_hashing import content_hash


class CompetencyContractService:
    @classmethod
    def _applicability_successor(cls, contract, decision, matrix):
        from contracts.ontology_building import ArtifactPinV1
        if set(matrix) != {q.competency_id for q in contract.questions}:
            raise ValueError('COMPETENCY_APPLICABILITY_MATRIX_INCOMPLETE')
        questions=[]
        for question in contract.questions:
            payload=question.model_dump(mode='json')
            rows=matrix[question.competency_id]
            payload.update(dimension_applicability=rows,
                required_semantic_dimensions=[r['dimension'] for r in rows if r['state']=='REQUIRED'])
            questions.append(CompetencyQuestionV1.model_validate(payload).model_dump(mode='json'))
        payload=contract.model_dump(mode='json',exclude={'competency_contract_ref','competency_contract_hash'})
        payload.update(questions=questions,predecessor_ref=contract.competency_contract_ref,
            predecessor_hash=contract.competency_contract_hash,created_at=decision.decided_at.isoformat(),
            applicability_review_pin=ArtifactPinV1(artifact_kind='conversation_decision',
                artifact_ref=decision.decision_ref,artifact_hash=decision.decision_hash).model_dump(mode='json'))
        digest=content_hash(payload)
        return CompetencyContractV1(competency_contract_ref='competency-contract:'+digest,
            competency_contract_hash='sha256:'+digest,**payload)

    @classmethod
    def verify_applicability_persisted(cls, *, store, contract):
        """Replay reviewed source-scope exclusions, never infer them from absence."""
        import json
        from contracts.ontology_building import ArtifactPinV1
        from contracts.normative_schema import NormativeSchemaProposalEvidenceV1
        from services.normative_schema_production_service import NormativeSchemaProductionService
        if contract.applicability_review_pin is None:
            return
        scope=dict(organization_id=contract.organization_id,workspace_id=contract.workspace_id)
        pin=contract.applicability_review_pin
        decision=store.get(**scope,artifact_ref=pin.artifact_ref,contract_type=ConversationDecisionV1)
        if decision is None or decision.decision_hash != pin.artifact_hash:
            raise ValueError('COMPETENCY_APPLICABILITY_REVIEW_PIN_MISMATCH')
        expected_decision=cls().create_decision(**scope,actor_type=decision.actor_type,
            governed_actor_ref=decision.governed_actor_ref,channel=decision.channel,
            decision_kind=decision.decision_kind,target_ref=decision.target_ref,rationale=decision.rationale,
            evidence_refs=decision.evidence_refs,decided_at=decision.decided_at)
        if expected_decision != decision:
            raise ValueError('COMPETENCY_APPLICABILITY_REVIEW_HASH_MISMATCH')
        review=json.loads(decision.rationale)
        if review.get('review_kind') != 'cq_dimension_applicability.v1':
            raise ValueError('COMPETENCY_APPLICABILITY_REVIEW_KIND_MISMATCH')
        prior=store.get(**scope,artifact_ref=contract.predecessor_ref,contract_type=CompetencyContractV1)
        if (prior is None or prior.competency_contract_hash != contract.predecessor_hash
            or review['predecessor_hash'] != prior.competency_contract_hash
            or decision.target_ref != prior.competency_contract_ref):
            raise ValueError('COMPETENCY_APPLICABILITY_PREDECESSOR_MISMATCH')
        schema,_=NormativeSchemaProductionService.read_persisted(store=store,**scope,
            schema_pin=ArtifactPinV1.model_validate(review['source_schema_pin']))
        if (schema.competency_contract_ref,schema.competency_contract_hash) != (prior.competency_contract_ref,prior.competency_contract_hash):
            raise ValueError('COMPETENCY_APPLICABILITY_SOURCE_SCOPE_MISMATCH')
        receipt=store.get(**scope,artifact_ref=schema.proposal_evidence_ref,contract_type=NormativeSchemaProposalEvidenceV1)
        source=json.loads(receipt.input_text)
        evidence={e['evidenceRef'] for e in source['source_evidence']}
        reviewed={ref for rows in review['matrix'].values() for row in rows for ref in row['evidence_refs']}
        if not reviewed.issubset(evidence) or set(decision.evidence_refs) != reviewed:
            raise ValueError('COMPETENCY_APPLICABILITY_SOURCE_EVIDENCE_MISMATCH')
        if cls._applicability_successor(prior,decision,review['matrix']) != contract:
            raise ValueError('COMPETENCY_APPLICABILITY_REPLAY_MISMATCH')

    def review_dimension_applicability_persisted(self, *, store, authority_resolver, ctx,
        competency_pin, source_schema_pin, matrix, request_id, decided_at):
        import json
        from contracts.knowledge_classification import RuntimeContextDTO
        from contracts.ontology_building import ArtifactPinV1
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path
        if not isinstance(store,OntologyBuildStore) or not isinstance(authority_resolver,RuntimeAuthorityResolver):
            raise TypeError('competency applicability requires production owners')
        if not ctx.authenticated or store._path.resolve()!=ontology_build_store_path(ctx.workspace_root).resolve():
            raise ValueError('COMPETENCY_WORKSPACE_OWNER_MISMATCH')
        actor=RuntimeContextDTO(organization_id=ctx.organization_id,workspace_id=ctx.workspace_id,actor_id=ctx.user_id)
        authority_resolver.require_current_actor(actor_context=actor, authorization_class="TENANT_ADMIN")
        scope=dict(organization_id=ctx.organization_id,workspace_id=ctx.workspace_id)
        if competency_pin.artifact_kind!='competency_contract':raise ValueError('COMPETENCY_PIN_REQUIRED')
        prior=store.get(**scope,artifact_ref=competency_pin.artifact_ref,contract_type=CompetencyContractV1)
        if prior is None or prior.competency_contract_hash!=competency_pin.artifact_hash:
            raise ValueError('COMPETENCY_PIN_MISMATCH')
        if any(q.review_state!='confirmed' for q in prior.questions):
            raise ValueError('COMPETENCY_REVIEW_REQUIRED')
        self.verify_applicability_persisted(store=store,contract=prior)
        rationale=json.dumps(dict(review_kind='cq_dimension_applicability.v1',
            predecessor_hash=prior.competency_contract_hash,source_schema_pin=source_schema_pin.model_dump(mode='json'),
            matrix=matrix),sort_keys=True,ensure_ascii=False)
        operation=dict(**scope,actor_ref='user:'+ctx.user_id,operation='REVIEW_CQ_DIMENSION_APPLICABILITY',
            request_id=request_id,request_hash='sha256:'+content_hash(dict(competency=competency_pin.model_dump(mode='json'),rationale=rationale)))
        cached=store.get_operation_result(**operation,artifact_kind='competency_contract')
        if cached is not None:
            successor=store.get(**scope,artifact_ref=cached.artifact_ref,contract_type=CompetencyContractV1)
            self.verify_applicability_persisted(store=store,contract=successor)
            return successor
        decision=self.create_decision(**scope,actor_type='system',governed_actor_ref=operation['actor_ref'],
            channel='chat',decision_kind='define_boundary',target_ref=prior.competency_contract_ref,
            rationale=rationale,evidence_refs=tuple(sorted({ref for rows in matrix.values() for r in rows for ref in r['evidence_refs']})),
            decided_at=decided_at)
        successor=self._applicability_successor(prior,decision,matrix)
        # Validate against the exact source owner before publishing either artifact.
        from contracts.normative_schema import NormativeSchemaProposalEvidenceV1
        from services.normative_schema_production_service import NormativeSchemaProductionService
        schema,_=NormativeSchemaProductionService.read_persisted(store=store,**scope,schema_pin=source_schema_pin)
        if (schema.competency_contract_ref,schema.competency_contract_hash)!=(prior.competency_contract_ref,prior.competency_contract_hash):
            raise ValueError('COMPETENCY_APPLICABILITY_SOURCE_SCOPE_MISMATCH')
        receipt=store.get(**scope,artifact_ref=schema.proposal_evidence_ref,contract_type=NormativeSchemaProposalEvidenceV1)
        evidence={e['evidenceRef'] for e in json.loads(receipt.input_text)['source_evidence']}
        if not set(decision.evidence_refs).issubset(evidence):
            raise ValueError('COMPETENCY_APPLICABILITY_SOURCE_EVIDENCE_MISMATCH')
        store.put_many_with_operation_result(**operation,artifacts=(decision,successor),
            result_pin=ArtifactPinV1(artifact_kind='competency_contract',artifact_ref=successor.competency_contract_ref,
                artifact_hash=successor.competency_contract_hash),
            verify_before_commit=lambda:authority_resolver.require_current_actor(actor_context=actor, authorization_class="TENANT_ADMIN"))
        self.verify_applicability_persisted(store=store,contract=successor)
        return successor

    def review_absent_dimension_scope_persisted(self, *, store, authority_resolver, ctx,
        competency_pin, schema_pin, question_id, dimension, exclusion, rationale, request_id, decided_at):
        """Explicitly narrow one question using reviewed absence, retaining its behavioral obligation."""
        from contracts.knowledge_classification import RuntimeContextDTO
        from contracts.ontology_building import ArtifactPinV1
        from services.normative_schema_production_service import NormativeSchemaProductionService
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path
        if not isinstance(store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("competency scope review requires production owners")
        if not ctx.authenticated or store._path.resolve() != ontology_build_store_path(ctx.workspace_root).resolve():
            raise ValueError("COMPETENCY_WORKSPACE_OWNER_MISMATCH")
        actor = RuntimeContextDTO(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, actor_id=ctx.user_id)
        authority_resolver.require_current_actor(actor_context=actor)
        if competency_pin.artifact_kind != "competency_contract" or not exclusion.strip() or not rationale.strip():
            raise ValueError("competency scope review requires exact inputs and explicit rationale/exclusion")
        scope = dict(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        operation = dict(**scope, actor_ref="user:" + ctx.user_id,
            operation="REVIEW_COMPETENCY_ABSENT_DIMENSION_SCOPE", request_id=request_id,
            request_hash="sha256:" + content_hash(dict(competency=competency_pin.model_dump(mode="json"),
                schema=schema_pin.model_dump(mode="json"), question_id=question_id, dimension=dimension,
                exclusion=exclusion, rationale=rationale)))
        previous = store.get_operation_result(**operation, artifact_kind="competency_contract")
        if previous is not None:
            return store.get(**scope, artifact_ref=previous.artifact_ref, contract_type=CompetencyContractV1)
        schema, _ = NormativeSchemaProductionService.read_persisted(store=store, **scope, schema_pin=schema_pin)
        if (schema.competency_contract_ref, schema.competency_contract_hash) != (competency_pin.artifact_ref, competency_pin.artifact_hash):
            raise ValueError("COMPETENCY_SCOPE_SCHEMA_MISMATCH")
        slot = next((item for item in schema.slots if item.dimension == dimension), None)
        if slot is None or slot.epistemic_status != "NONE_EVIDENCED" or schema.source_review_ref is None:
            raise ValueError("COMPETENCY_SCOPE_REVIEWED_ABSENCE_REQUIRED")
        contract = store.get(**scope, artifact_ref=competency_pin.artifact_ref, contract_type=CompetencyContractV1)
        if contract is None or contract.competency_contract_hash != competency_pin.artifact_hash:
            raise ValueError("COMPETENCY_PIN_MISMATCH")
        targets = [q for q in contract.questions if q.competency_id == question_id]
        if (len(targets) != 1 or targets[0].review_state != "confirmed"
            or dimension not in targets[0].required_semantic_dimensions
            or len(targets[0].required_semantic_dimensions) < 2):
            raise ValueError("COMPETENCY_SCOPE_QUESTION_MISMATCH")
        decision = self.create_decision(**scope, actor_type="human", governed_actor_ref=operation["actor_ref"],
            channel="chat", decision_kind="define_boundary", target_ref=question_id,
            rationale=rationale + "\nExplicit exclusion: " + exclusion,
            evidence_refs=(schema.schema_ref, schema.source_review_ref, slot.search_coverage_ref), decided_at=decided_at)
        questions = tuple(q.model_copy(update=dict(required_semantic_dimensions=tuple(
            d for d in q.required_semantic_dimensions if d != dimension), provenance_kind="user_confirmed"))
            if q.competency_id == question_id else q for q in contract.questions)
        payload = contract.model_dump(mode="json", exclude={"competency_contract_ref", "competency_contract_hash", "created_at"})
        payload.update(predecessor_ref=contract.competency_contract_ref, predecessor_hash=contract.competency_contract_hash,
            questions=[q.model_dump(mode="json") for q in questions], out_of_scope=list(dict.fromkeys((*contract.out_of_scope, exclusion))))
        digest = content_hash(dict(contract=payload, semantic_decision=decision.semantic_payload(),
            schema=schema_pin.model_dump(mode="json")))
        successor = CompetencyContractV1(competency_contract_ref="competency-contract:" + digest,
            competency_contract_hash="sha256:" + digest, created_at=decided_at, **payload)
        pin = store.put_many_with_operation_result(**operation, artifacts=(decision, successor),
            result_pin=ArtifactPinV1(artifact_kind="competency_contract", artifact_ref=successor.competency_contract_ref,
                artifact_hash=successor.competency_contract_hash),
            verify_before_commit=lambda: authority_resolver.require_current_actor(actor_context=actor))
        return store.get(**scope, artifact_ref=pin.artifact_ref, contract_type=CompetencyContractV1)

    def propose_persisted(
        self, *, store, intent_store, organization_id: str, workspace_id: str,
        origin_pin, actor_ref: str, request_id: str, created_at: datetime,
    ) -> CompetencyContractV1:
        """Propose certification questions from the exact reviewed A0 intent.

        Normative dimensions are the existing schema contract's required coverage,
        not inferred legal facts. All resulting questions remain unresolved.
        """
        from contracts.ontology_building import ArtifactPinV1, OntologyBuildManifestV1
        from contracts.normative_schema import NormativeSchemaContractV1
        from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path
        from storage.graph_intent_brief_store import GraphIntentBriefStore
        if not isinstance(store, OntologyBuildStore) or not isinstance(intent_store, GraphIntentBriefStore):
            raise TypeError("competency proposal requires production artifact owners")
        if (intent_store.organization_id, intent_store.workspace_id) != (organization_id, workspace_id) or store._path.resolve() != ontology_build_store_path(intent_store.workspace_path).resolve():
            raise ValueError("COMPETENCY_WORKSPACE_OWNER_MISMATCH")
        if origin_pin.artifact_kind != "ontology_build_manifest":
            raise ValueError("COMPETENCY_ORIGIN_PIN_REQUIRED")
        scope = dict(organization_id=organization_id, workspace_id=workspace_id)
        operation = dict(**scope, actor_ref=actor_ref, operation="PROPOSE_CERTIFICATION_COMPETENCY",
            request_id=request_id, request_hash="sha256:" + content_hash(origin_pin.model_dump(mode="json")))
        prior = store.get_operation_result(**operation, artifact_kind="competency_contract")
        if prior is not None:
            return store.get(**scope, artifact_ref=prior.artifact_ref, contract_type=CompetencyContractV1)
        origin = store.get(**scope, artifact_ref=origin_pin.artifact_ref, contract_type=OntologyBuildManifestV1)
        if origin is None or origin.build_hash != origin_pin.artifact_hash:
            raise ValueError("COMPETENCY_ORIGIN_PIN_MISMATCH")
        def child(kind):
            pins = [item for item in origin.child_artifacts if item.artifact_kind == kind]
            if len(pins) != 1:
                raise ValueError("COMPETENCY_REQUIRED_CHILD_MISSING:" + kind)
            return pins[0]
        boundary_pin = child("canonical_evidence_boundary")
        boundary = store.get(**scope, artifact_ref=boundary_pin.artifact_ref, contract_type=CanonicalEvidenceBoundaryV1)
        if boundary is None or boundary.boundary_hash != boundary_pin.artifact_hash:
            raise ValueError("COMPETENCY_BOUNDARY_PIN_MISMATCH")
        intent_pin = child("graph_intent_brief")
        brief = intent_store.get(intent_pin.artifact_ref, graph_intent_hash=intent_pin.artifact_hash)
        if not brief.compiler_eligible:
            raise ValueError("GRAPH_INTENT_REVIEW_REQUIRED")
        source = brief.source_binding
        if (source.source_identity_ref, source.source_hash, source.canonical_source_ir_ref, source.canonical_source_ir_hash) != (boundary.source_identity_ref, boundary.source_hash, boundary.canonical_source_ir_ref, boundary.canonical_source_ir_hash):
            raise ValueError("COMPETENCY_SOURCE_LINEAGE_MISMATCH")
        snapshot, _ = store.get_verified_exploratory_snapshot(**scope, pin=child("exploratory_graph_snapshot"))
        if snapshot.build_ref != origin.build_ref:
            raise ValueError("COMPETENCY_SNAPSHOT_LINEAGE_MISMATCH")
        evidence = {item.node_id: item.evidence_refs for item in snapshot.nodes if item.node_kind == "reviewed_competency_question"}
        questions = tuple(dict(competency_id=item.question_id, question=item.text,
            required_semantic_dimensions=NormativeSchemaContractV1.REQUIRED_DIMENSIONS,
            acceptance_obligations=(item.success_condition,),
            evidence_refs=evidence.get("intent-competency:" + item.question_id, ())) for item in brief.competency_questions)
        result = self.propose(boundary=boundary, graph_intent_brief_ref=brief.graph_intent_id,
            graph_intent_brief_hash=brief.graph_intent_hash, questions=questions, created_at=created_at)
        existing = store.get(**scope, artifact_ref=result.competency_contract_ref, contract_type=CompetencyContractV1)
        if existing is not None:
            if existing.model_dump(exclude={"created_at"}) != result.model_dump(exclude={"created_at"}):
                raise ValueError("COMPETENCY_CONTENT_CONFLICT")
            result = existing
        pin = store.put_many_with_operation_result(**operation, artifacts=(result,),
            result_pin=ArtifactPinV1(artifact_kind="competency_contract",
                artifact_ref=result.competency_contract_ref, artifact_hash=result.competency_contract_hash))
        return store.get(**scope, artifact_ref=pin.artifact_ref, contract_type=CompetencyContractV1)

    def review_persisted(
        self, *, store, authority_resolver, actor_context, competency_pin,
        question_id: str, decision_kind: str, rationale: str, channel: str,
        request_id: str, decided_at: datetime,
    ) -> CompetencyContractV1:
        from contracts.ontology_building import ArtifactPinV1
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        from storage.ontology_build_store import OntologyBuildStore
        if not isinstance(store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("competency review requires production owners")
        authority_resolver.require_current_actor(actor_context=actor_context)
        if competency_pin.artifact_kind != "competency_contract":
            raise ValueError("COMPETENCY_PIN_REQUIRED")
        scope = dict(organization_id=actor_context.organization_id, workspace_id=actor_context.workspace_id)
        actor_ref = "user:" + actor_context.actor_id
        operation = dict(**scope, actor_ref=actor_ref, operation="REVIEW_CERTIFICATION_COMPETENCY",
            request_id=request_id, request_hash="sha256:" + content_hash(dict(
                competency=competency_pin.model_dump(mode="json"), question_id=question_id,
                decision_kind=decision_kind, rationale=rationale, channel=channel)))
        prior = store.get_operation_result(**operation, artifact_kind="competency_contract")
        if prior is not None:
            return store.get(**scope, artifact_ref=prior.artifact_ref, contract_type=CompetencyContractV1)
        contract = store.get(**scope, artifact_ref=competency_pin.artifact_ref, contract_type=CompetencyContractV1)
        if contract is None or contract.competency_contract_hash != competency_pin.artifact_hash:
            raise ValueError("COMPETENCY_PIN_MISMATCH")
        questions = [item for item in contract.questions if item.competency_id == question_id]
        if len(questions) != 1:
            raise ValueError("COMPETENCY_QUESTION_REQUIRED")
        decision = self.create_decision(**scope, actor_type="human", governed_actor_ref=actor_ref,
            channel=channel, decision_kind=decision_kind, target_ref=question_id, rationale=rationale,
            evidence_refs=questions[0].evidence_refs, decided_at=decided_at)
        successor = self.apply_decision(contract, decision, actor_authorized=True)
        pin = store.put_many_with_operation_result(**operation, artifacts=(decision, successor),
            result_pin=ArtifactPinV1(artifact_kind="competency_contract",
                artifact_ref=successor.competency_contract_ref, artifact_hash=successor.competency_contract_hash),
            verify_before_commit=lambda: authority_resolver.require_current_actor(actor_context=actor_context))
        return store.get(**scope, artifact_ref=pin.artifact_ref, contract_type=CompetencyContractV1)

    def propose(
        self,
        *,
        boundary: CanonicalEvidenceBoundaryV1,
        graph_intent_brief_ref: str,
        graph_intent_brief_hash: str,
        questions: tuple[dict[str, Any], ...],
        created_at: datetime,
    ) -> CompetencyContractV1:
        typed_questions = tuple(
            CompetencyQuestionV1(
                competency_id=str(item["competency_id"]),
                question=str(item["question"]),
                criticality="proposed_critical",
                review_state="unresolved",
                required_semantic_dimensions=tuple(item["required_semantic_dimensions"]),
                acceptance_obligations=tuple(item["acceptance_obligations"]),
                evidence_refs=tuple(item["evidence_refs"]),
                provenance_kind="source_derived",
            )
            for item in questions
        )
        basis = {
            "organization_id": boundary.organization_id,
            "workspace_id": boundary.workspace_id,
            "boundary_ref": boundary.boundary_ref,
            "boundary_hash": boundary.boundary_hash,
            "graph_intent_brief_ref": graph_intent_brief_ref,
            "graph_intent_brief_hash": graph_intent_brief_hash,
            "intended_domain": boundary.intended_domain,
            "questions": [item.model_dump(mode="json") for item in typed_questions],
        }
        digest = content_hash(basis)
        return CompetencyContractV1(
            competency_contract_ref=f"competency-contract:{digest}",
            competency_contract_hash=f"sha256:{digest}",
            organization_id=boundary.organization_id,
            workspace_id=boundary.workspace_id,
            boundary_ref=boundary.boundary_ref,
            boundary_hash=boundary.boundary_hash,
            graph_intent_brief_ref=graph_intent_brief_ref,
            graph_intent_brief_hash=graph_intent_brief_hash,
            intended_domain=boundary.intended_domain,
            in_scope=(boundary.intended_use,),
            out_of_scope=boundary.explicit_exclusions,
            questions=typed_questions,
            created_at=created_at,
        )

    def create_decision(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        actor_type: str,
        governed_actor_ref: str,
        channel: str,
        decision_kind: str,
        target_ref: str,
        rationale: str,
        evidence_refs: tuple[str, ...],
        decided_at: datetime,
    ) -> ConversationDecisionV1:
        basis = {
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "actor_type": actor_type,
            "governed_actor_ref": governed_actor_ref,
            "decision_kind": decision_kind,
            "target_ref": target_ref,
            "rationale": rationale,
            "evidence_refs": evidence_refs,
            "decided_at": decided_at.isoformat(),
        }
        digest = content_hash(basis)
        return ConversationDecisionV1(
            decision_ref=f"conversation-decision:{digest}",
            decision_hash=f"sha256:{digest}",
            channel=channel,
            **basis,
        )

    def apply_decision(
        self,
        contract: CompetencyContractV1,
        decision: ConversationDecisionV1,
        *,
        actor_authorized: bool,
    ) -> CompetencyContractV1:
        if (decision.organization_id, decision.workspace_id) != (
            contract.organization_id,
            contract.workspace_id,
        ):
            raise PermissionError("decision is outside the competency tenant scope")
        found = False
        questions: list[CompetencyQuestionV1] = []
        for question in contract.questions:
            if question.competency_id != decision.target_ref:
                questions.append(question)
                continue
            found = True
            states = {
                "confirm_criticality": ("critical", "confirmed"),
                "downgrade_criticality": ("noncritical", "confirmed"),
                "exclude_competency": ("excluded", "rejected"),
                "define_boundary": (question.criticality, "confirmed"),
            }
            criticality, review_state = states[decision.decision_kind]
            questions.append(
                question.model_copy(
                    update={
                        "criticality": criticality,
                        "review_state": review_state,
                        "provenance_kind": (
                            "user_confirmed"
                            if decision.actor_type == "human"
                            else "system_confirmed"
                        ),
                    }
                )
            )
        if not found:
            raise ValueError("decision target is not in the competency contract")
        basis = {
            "predecessor_ref": contract.competency_contract_ref,
            "predecessor_hash": contract.competency_contract_hash,
            "semantic_decision": decision.semantic_payload(),
            "questions": [item.model_dump(mode="json") for item in questions],
        }
        digest = content_hash(basis)
        return CompetencyContractV1(
            competency_contract_ref=f"competency-contract:{digest}",
            competency_contract_hash=f"sha256:{digest}",
            organization_id=contract.organization_id,
            workspace_id=contract.workspace_id,
            boundary_ref=contract.boundary_ref,
            boundary_hash=contract.boundary_hash,
            graph_intent_brief_ref=contract.graph_intent_brief_ref,
            graph_intent_brief_hash=contract.graph_intent_brief_hash,
            intended_domain=contract.intended_domain,
            in_scope=contract.in_scope,
            out_of_scope=contract.out_of_scope,
            questions=tuple(questions),
            predecessor_ref=contract.competency_contract_ref,
            predecessor_hash=contract.competency_contract_hash,
            created_at=decision.decided_at,
        )


__all__ = ["CompetencyContractService"]
