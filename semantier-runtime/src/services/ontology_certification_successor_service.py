"""Immutable A0-to-A1 certification successor publication."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Iterable

from contracts.normative_schema import NormativeSchemaContractV1, SchemaAdequacyReportV1, SourceSearchCoverageManifestV1
from contracts.normative_semantics import NormativeSemanticCandidateSetV1
from contracts.ontology_admission import CertifiedSemanticExecutionPlanV1
from contracts.ontology_artifact_integrity import create_exploratory_snapshot_integrity
from contracts.ontology_building import (
    ArtifactPinV1,
    CanonicalEvidenceBoundaryV1,
    CandidateGraphDiffProjectionV1,
    CandidateGraphLineageProjectionV1,
    CompetencyContractV1,
    ExploratoryGraphEdgeV1,
    ExploratoryGraphNodeV1,
    ExploratoryGraphSnapshotV1,
    OntologyBuildManifestV1,
)
from contracts.ontology_certification import (
    NormativeAuthorityDependencySetV1,
    NormativeAuthorityDependencyV1,
    AuthoritySectionPinV1,
    OntologyCertificationCommandV1,
    OntologyCertificationReadinessV1,
    OntologyCertificationReceiptV1,
)
from contracts.ontology_validation import (
    CompiledRulePlanV1,
    RuntimeSemanticEvaluationBundleV1,
)
from contracts.ontology_paired_evaluation import OntologyPairedObligationSuiteV1
from contracts.ontology_quality import QualityEvaluationRunV1, QualityProfileV1
from contracts.validation_contract import ValidationContractV1
from eos.artifact_hashing import content_hash
from services.ontology_build_coordinator import OntologyBuildCoordinator
from services.ontology_evidence_adequacy_service import OntologyEvidenceAdequacyService
from services.ontology_lineage_service import OntologyLineageService
from services.ontology_schema_adequacy_service import OntologySchemaAdequacyService
from storage.ontology_build_store import OntologyBuildStore


@dataclass(frozen=True)
class CertificationLegalSelection:
    source_id: str
    version_id: str
    artifact_id: str
    artifact_hash: str
    section_roles: dict[str, tuple[str, ...]]
    section_hashes: dict[str, tuple[str, str]]


class OntologyCertificationSuccessorService:
    PRODUCER_PROFILE = "a1_certification_successor.v1"
    PRODUCER_VERSION = "semantier_a1_certification_successor_v1"

    @staticmethod
    def resolve_authority_dependency(
        *, legal_store, organization_id: str, workspace_id: str,
        source_id: str, version_id: str, artifact_id: str, artifact_hash: str,
        section_roles: dict[str, tuple[str, ...]],
        bind_source_content: bool = False,
    ) -> NormativeAuthorityDependencyV1:
        """Construct dependency facts from the legal owner, retaining selected roles.

        Roles are reviewed semantic input, not inferred from text or source labels.
        This method neither approves sources nor certifies a dependency set.
        """
        from eos.legal_corpus_store import LegalCorpusStore
        if not isinstance(legal_store, LegalCorpusStore):
            raise TypeError("certification requires the production legal corpus owner")
        evidence = legal_store.resolve_certification_source_evidence(
            organization_id=organization_id, workspace_id=workspace_id,
            source_id=source_id, version_id=version_id, artifact_id=artifact_id,
            artifact_hash=artifact_hash, section_ids=tuple(section_roles),
        )
        source, version, artifact = (evidence[key] for key in ("source", "version", "artifact"))
        source_content_hash = None
        if bind_source_content:
            import re
            source_content_hash = version['content_hash']
            if not re.fullmatch(r'sha256:[0-9a-f]{64}', source_content_hash):
                raise ValueError('CERTIFICATION_SOURCE_CONTENT_HASH_REQUIRED')
        return NormativeAuthorityDependencyV1(
            source_id=source["source_id"], version_id=version["version_id"],
            authority_tier=source["authority_tier"], governance_state=source["governance_state"],
            jurisdiction=source["jurisdiction"], artifact_id=artifact["artifact_id"],
            artifact_hash="sha256:" + artifact["content_hash"], storage_ref=artifact["storage_ref"],
            source_content_hash=source_content_hash,
            effective_from=version["effective_from"], effective_to=version["effective_to"],
            sections=tuple(AuthoritySectionPinV1(
                section_id=row["section_id"], stable_locator=row["stable_locator"],
                text_hash=row["text_hash"], local_hash=row["local_hash"],
                dependency_roles=section_roles[row["section_id"]],
            ) for row in evidence["sections"]),
        )

    def verify_authority_dependencies(self, *, authority: NormativeAuthorityDependencySetV1, legal_store) -> None:
        for dependency in authority.dependencies:
            resolved = self.resolve_authority_dependency(
                legal_store=legal_store, organization_id=authority.organization_id,
                workspace_id=authority.workspace_id, source_id=dependency.source_id,
                version_id=dependency.version_id, artifact_id=dependency.artifact_id,
                artifact_hash=dependency.artifact_hash,
                section_roles={section.section_id: section.dependency_roles for section in dependency.sections},
                bind_source_content=dependency.source_content_hash is not None,
            )
            if resolved != dependency:
                raise ValueError("CERTIFICATION_LEGAL_OWNER_MISMATCH")

    @staticmethod
    def _legal_coverage_config_hash(dependencies) -> str:
        return "sha256:" + content_hash(dict(selection_scope="exact_legal_sections.v1",
            dependencies=[item.model_dump(mode="json") for item in dependencies]))

    def review_source_coverage(
        self, *, store: OntologyBuildStore, legal_store, authority_resolver, actor_context,
        selections: tuple[CertificationLegalSelection, ...], request_id: str, reviewed_at: datetime,
        bind_source_content: bool = False,
    ) -> SourceSearchCoverageManifestV1:
        """Record explicit review of a bounded selection, not corpus exhaustiveness."""
        from services.runtime_context_transition_service import RuntimeAuthorityResolver
        if not isinstance(store, OntologyBuildStore) or not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("coverage review requires production owners")
        if not selections:
            raise ValueError("CERTIFICATION_DEPENDENCY_SELECTION_REQUIRED")
        authority_resolver.require_current_actor(actor_context=actor_context)
        scope = dict(organization_id=actor_context.organization_id, workspace_id=actor_context.workspace_id)
        actor_ref = "user:" + actor_context.actor_id
        selection_basis = [asdict(item) for item in selections]
        if bind_source_content:
            selection_basis = dict(selections=selection_basis, bind_source_content=True)
        operation = dict(**scope, actor_ref=actor_ref, operation="REVIEW_CERTIFICATION_SOURCE_COVERAGE",
            request_id=request_id, request_hash="sha256:" + content_hash(selection_basis))
        prior = store.get_operation_result(**operation, artifact_kind="source_search_coverage_manifest")
        if prior is not None:
            return store.get(**scope, artifact_ref=prior.artifact_ref, contract_type=SourceSearchCoverageManifestV1)
        dependencies = tuple(self.resolve_authority_dependency(legal_store=legal_store, **scope,
            source_id=item.source_id, version_id=item.version_id, artifact_id=item.artifact_id,
            artifact_hash=item.artifact_hash,
            section_roles={key:item.section_roles[key] for key in sorted(item.section_roles)},
            bind_source_content=bind_source_content) for item in selections)
        if len({(item.source_id, item.version_id) for item in dependencies}) != len(dependencies):
            raise ValueError("CERTIFICATION_DUPLICATE_DEPENDENCY")
        for selected, dependency in zip(selections, dependencies):
            if selected.section_hashes != {section.section_id:(section.text_hash, section.local_hash) for section in dependency.sections}:
                raise ValueError("CERTIFICATION_LEGAL_SECTION_PIN_MISMATCH")
        refs = tuple("legal-section:" + section.section_id for item in dependencies for section in item.sections)
        basis = dict(**scope, source_region_refs=refs, excluded_region_refs=(),
            search_methods=("explicit_pinned_section_review",),
            pattern_classes=tuple(sorted({role for item in dependencies for section in item.sections for role in section.dependency_roles})),
            parser_ref="persisted_legal_sections.v1", provider_ref="LegalCorpusStore",
            config_hash=self._legal_coverage_config_hash(dependencies),
            dependency_traversal_refs=tuple(item.version_id for item in dependencies), result_refs=refs,
            governed_reviewer_ref=actor_ref, reviewed_at=reviewed_at.isoformat())
        digest = content_hash(basis)
        coverage = SourceSearchCoverageManifestV1(coverage_ref="source-search-coverage:" + digest,
            coverage_hash="sha256:" + digest, **basis)
        pin = store.put_many_with_operation_result(**operation, artifacts=(coverage,),
            result_pin=ArtifactPinV1(artifact_kind="source_search_coverage_manifest",
                artifact_ref=coverage.coverage_ref, artifact_hash=coverage.coverage_hash),
            verify_before_commit=lambda: authority_resolver.require_current_actor(actor_context=actor_context))
        return store.get(**scope, artifact_ref=pin.artifact_ref, contract_type=SourceSearchCoverageManifestV1)

    def publish_authority_dependency_set(
        self, *, store: OntologyBuildStore, legal_store,
        organization_id: str, workspace_id: str, origin_pin: ArtifactPinV1,
        coverage_pin: ArtifactPinV1, selections: tuple[CertificationLegalSelection, ...],
        jurisdiction: str, as_of_time: datetime, created_at: datetime,
        actor_ref: str, request_id: str,
        bind_source_content: bool = False,
    ) -> NormativeAuthorityDependencySetV1:
        """Publish exact legal-owner facts under an existing reviewed coverage pin."""
        if not isinstance(store, OntologyBuildStore):
            raise TypeError("certification requires the production build owner")
        if not selections:
            raise ValueError("CERTIFICATION_DEPENDENCY_SELECTION_REQUIRED")
        if origin_pin.artifact_kind != "ontology_build_manifest" or coverage_pin.artifact_kind != "source_search_coverage_manifest":
            raise ValueError("CERTIFICATION_INPUT_KIND_MISMATCH")
        scope = dict(organization_id=organization_id, workspace_id=workspace_id)
        request_basis = dict(origin=origin_pin.model_dump(mode="json"),
            coverage=coverage_pin.model_dump(mode="json"), selections=[asdict(item) for item in selections],
            jurisdiction=jurisdiction, as_of_time=as_of_time.isoformat())
        if bind_source_content:
            request_basis['bind_source_content'] = True
        request_hash = "sha256:" + content_hash(request_basis)
        operation = dict(**scope, actor_ref=actor_ref, operation="PREPARE_CERTIFICATION_AUTHORITY",
                         request_id=request_id, request_hash=request_hash)
        previous = store.get_operation_result(**operation, artifact_kind="normative_authority_dependency_set")
        if previous is not None:
            return store.get(**scope, artifact_ref=previous.artifact_ref, contract_type=NormativeAuthorityDependencySetV1)
        origin = store.get(**scope, artifact_ref=origin_pin.artifact_ref, contract_type=OntologyBuildManifestV1)
        coverage = store.get(**scope, artifact_ref=coverage_pin.artifact_ref, contract_type=SourceSearchCoverageManifestV1)
        if origin is None or origin.build_hash != origin_pin.artifact_hash:
            raise ValueError("CERTIFICATION_ORIGIN_PIN_MISMATCH")
        if coverage is None or coverage.coverage_hash != coverage_pin.artifact_hash:
            raise ValueError("CERTIFICATION_COVERAGE_PIN_MISMATCH")
        def child(kind):
            matches = [item for item in origin.child_artifacts if item.artifact_kind == kind]
            if len(matches) != 1:
                raise ValueError("CERTIFICATION_REQUIRED_CHILD_MISSING:" + kind)
            return matches[0]
        snapshot_pin = child("exploratory_graph_snapshot")
        snapshot, _ = store.get_verified_exploratory_snapshot(**scope, pin=snapshot_pin)
        if snapshot.build_ref != origin.build_ref:
            raise ValueError("CERTIFICATION_INPUT_LINEAGE_MISMATCH:snapshot_build")
        boundary_pin = child("canonical_evidence_boundary")
        boundary = store.get(**scope, artifact_ref=boundary_pin.artifact_ref, contract_type=CanonicalEvidenceBoundaryV1)
        if boundary is None or boundary.boundary_hash != boundary_pin.artifact_hash:
            raise ValueError("CERTIFICATION_BOUNDARY_PIN_MISMATCH")
        dependencies = tuple(self.resolve_authority_dependency(
            legal_store=legal_store, **scope, source_id=item.source_id, version_id=item.version_id,
            artifact_id=item.artifact_id, artifact_hash=item.artifact_hash,
            section_roles={key:item.section_roles[key] for key in sorted(item.section_roles)},
            bind_source_content=bind_source_content,
        ) for item in selections)
        for selected, dependency in zip(selections, dependencies):
            actual = {section.section_id: (section.text_hash, section.local_hash)
                      for section in dependency.sections}
            if selected.section_hashes != actual:
                raise ValueError("CERTIFICATION_LEGAL_SECTION_PIN_MISMATCH")
        if coverage.config_hash != self._legal_coverage_config_hash(dependencies):
            raise ValueError("CERTIFICATION_REVIEWED_SELECTION_MISMATCH")
        selected_sections = {"legal-section:" + section.section_id
                             for item in dependencies for section in item.sections}
        if not selected_sections.issubset(set(coverage.result_refs)):
            raise ValueError("CERTIFICATION_LEGAL_COVERAGE_MISMATCH")
        basis = dict(**scope, origin_build_ref=origin.build_ref, origin_build_hash=origin.build_hash,
            origin_snapshot_ref=snapshot.snapshot_ref, origin_snapshot_hash=snapshot.snapshot_hash,
            boundary_ref=boundary.boundary_ref, boundary_hash=boundary.boundary_hash,
            jurisdiction=jurisdiction, as_of_time=as_of_time.isoformat(),
            dependencies=[item.model_dump(mode="json") for item in dependencies],
            search_coverage_ref=coverage.coverage_ref, search_coverage_hash=coverage.coverage_hash,
            created_at=created_at.isoformat())
        digest = content_hash(basis)
        result = NormativeAuthorityDependencySetV1(set_ref="normative-authority-dependencies:" + digest,
            set_hash="sha256:" + digest, **basis)
        winner = store.put_many_with_operation_result(**operation, artifacts=(result,),
            result_pin=ArtifactPinV1(artifact_kind="normative_authority_dependency_set",
                                    artifact_ref=result.set_ref, artifact_hash=result.set_hash))
        return store.get(**scope, artifact_ref=winner.artifact_ref, contract_type=NormativeAuthorityDependencySetV1)

    def certify_persisted(
        self, *, store: OntologyBuildStore, organization_id: str, workspace_id: str,
        command: OntologyCertificationCommandV1, execution_plan_pin: ArtifactPinV1,
        schema_adequacy_pin: ArtifactPinV1, representative_evaluation_pin: ArtifactPinV1,
        created_at: datetime, paired_suite_pin: ArtifactPinV1 | None = None,
    ) -> tuple[OntologyBuildManifestV1, OntologyCertificationReceiptV1]:
        """Resolve an exact, tenant-scoped input chain before successor publication.

        This consumes reviewed artifacts; it does not review competency, admit legal
        sources, or activate the resulting build. No latest-artifact lookup is used.
        """
        scope = dict(organization_id=organization_id, workspace_id=workspace_id)

        def load(label, contract, ref, digest, hash_field):
            artifact = store.get(**scope, artifact_ref=ref, contract_type=contract)
            if artifact is None:
                raise ValueError("CERTIFICATION_INPUT_NOT_FOUND:" + label)
            if getattr(artifact, hash_field) != digest:
                raise ValueError("CERTIFICATION_INPUT_PIN_MISMATCH:" + label)
            if (artifact.organization_id, artifact.workspace_id) != (organization_id, workspace_id):
                raise ValueError("CERTIFICATION_INPUT_SCOPE_MISMATCH:" + label)
            return artifact

        def selected(label, pin, kind, contract, hash_field):
            if pin.artifact_kind != kind:
                raise ValueError("CERTIFICATION_INPUT_KIND_MISMATCH:" + label)
            return load(label, contract, pin.artifact_ref, pin.artifact_hash, hash_field)

        def same(label, actual, expected):
            if actual != expected:
                raise ValueError("CERTIFICATION_INPUT_LINEAGE_MISMATCH:" + label)

        origin = load("origin_manifest", OntologyBuildManifestV1, command.origin_build_ref,
                      command.origin_build_hash, "build_hash")

        def child(kind):
            matches = [pin for pin in origin.child_artifacts if pin.artifact_kind == kind]
            if len(matches) != 1:
                raise ValueError("CERTIFICATION_REQUIRED_CHILD_MISSING:" + kind)
            return matches[0]

        snapshot_pin = child("exploratory_graph_snapshot")
        same("origin_snapshot", (snapshot_pin.artifact_ref, snapshot_pin.artifact_hash),
             (command.origin_snapshot_ref, command.origin_snapshot_hash))
        snapshot, _ = store.get_verified_exploratory_snapshot(**scope, pin=snapshot_pin)
        same("snapshot_build", snapshot.build_ref, origin.build_ref)
        boundary = selected("origin_boundary", child("canonical_evidence_boundary"), "canonical_evidence_boundary",
                            CanonicalEvidenceBoundaryV1, "boundary_hash")
        source_pin, ir_pin, intent_pin = child("source_identity"), child("canonical_source_ir"), child("graph_intent_brief")
        # SourceIdentity pins hash the identity payload, not the source bytes.
        same("boundary_source", boundary.source_identity_ref, source_pin.artifact_ref)
        same("boundary_ir", (boundary.canonical_source_ir_ref, boundary.canonical_source_ir_hash),
             (ir_pin.artifact_ref, ir_pin.artifact_hash))
        authority = load("authority", NormativeAuthorityDependencySetV1, command.authority_dependency_set_ref,
                         command.authority_dependency_set_hash, "set_hash")
        same("authority_origin", (authority.origin_build_ref, authority.origin_build_hash,
                                  authority.origin_snapshot_ref, authority.origin_snapshot_hash),
             (origin.build_ref, origin.build_hash, snapshot.snapshot_ref, snapshot.snapshot_hash))
        same("authority_boundary", (authority.boundary_ref, authority.boundary_hash),
             (boundary.boundary_ref, boundary.boundary_hash))
        from eos.legal_corpus_store import LegalCorpusStore
        self.verify_authority_dependencies(authority=authority, legal_store=LegalCorpusStore())
        load("search_coverage", SourceSearchCoverageManifestV1, authority.search_coverage_ref,
             authority.search_coverage_hash, "coverage_hash")
        competency = load("competency", CompetencyContractV1, command.reviewed_competency_contract_ref,
                          command.reviewed_competency_contract_hash, "competency_contract_hash")
        same("competency_intent", (competency.graph_intent_brief_ref, competency.graph_intent_brief_hash),
             (intent_pin.artifact_ref, intent_pin.artifact_hash))
        execution = selected("execution_plan", execution_plan_pin, "certified_semantic_execution_plan",
                             CertifiedSemanticExecutionPlanV1, "execution_plan_hash")
        candidate = load("candidate", NormativeSemanticCandidateSetV1, execution.canonical_normative_ir_ref,
                         execution.canonical_normative_ir_hash, "candidate_set_hash")
        schema = load("schema", NormativeSchemaContractV1, candidate.normative_schema_ref,
                      candidate.normative_schema_hash, "schema_hash")
        if schema.teaching_candidate or candidate.teaching_candidate:
            raise ValueError("TEACHING_CANDIDATE_NOT_CERTIFIABLE")
        adequacy = selected("schema_adequacy", schema_adequacy_pin, "schema_adequacy_report",
                            SchemaAdequacyReportV1, "report_hash")
        same("adequacy_schema", (adequacy.normative_schema_ref, adequacy.normative_schema_hash),
             (schema.schema_ref, schema.schema_hash))
        same("schema_intent", (schema.graph_intent_brief_ref, schema.graph_intent_brief_hash),
             (intent_pin.artifact_ref, intent_pin.artifact_hash))
        validation = load("validation", ValidationContractV1, execution.validation_contract_ref,
                          execution.validation_contract_hash, "validation_contract_hash")
        if validation.human_oracle_ref.startswith("source-validation-review:") or any(
            test.source_case_id is not None for test in validation.executable_tests):
            from services.validation_contract_review_service import ValidationContractReviewService
            ValidationContractReviewService.verify_source_contract(store=store, validation=validation)
        for label, artifact in (("competency", competency), ("schema", schema), ("validation", validation)):
            same(label + "_boundary", (artifact.boundary_ref, artifact.boundary_hash),
                 (boundary.boundary_ref, boundary.boundary_hash))
        for label, artifact in (("schema", schema), ("candidate", candidate), ("validation", validation)):
            same(label + "_competency", (artifact.competency_contract_ref, artifact.competency_contract_hash),
                 (competency.competency_contract_ref, competency.competency_contract_hash))
        for label, artifact in (("candidate", candidate), ("validation", validation)):
            same(label + "_source", (artifact.source_identity_ref, artifact.source_hash),
                 (boundary.source_identity_ref, boundary.source_hash))
        same("candidate_ir", (candidate.canonical_source_ir_ref, candidate.canonical_source_ir_hash),
             (ir_pin.artifact_ref, ir_pin.artifact_hash))
        same("validation_schema", (validation.normative_schema_ref, validation.normative_schema_hash),
             (schema.schema_ref, schema.schema_hash))
        run = load("quality_evaluation", QualityEvaluationRunV1, execution.quality_evaluation.artifact_ref,
                   execution.quality_evaluation.artifact_hash, "run_hash")
        profile = selected("quality_profile", execution.quality_profile, "quality_profile", QualityProfileV1, "profile_hash")
        run.require_basis(quality_profile=profile.pin(),
            subject=self._artifact_pin(candidate, "normative_semantic_candidate_set", "candidate_set_ref", "candidate_set_hash"),
            competency=self._artifact_pin(competency, "competency_contract", "competency_contract_ref", "competency_contract_hash"),
            validation=self._artifact_pin(validation, "validation_contract", "validation_contract_ref", "validation_contract_hash"))
        compiled = load("compiled_plan", CompiledRulePlanV1, execution.compiled_rule_plan_ref,
                        execution.compiled_rule_plan_hash, "plan_hash")
        for label, artifact in (("compiled_plan", compiled),):
            same(label + "_candidate", (artifact.candidate_set_ref, artifact.candidate_set_hash),
                 (candidate.candidate_set_ref, candidate.candidate_set_hash))
            same(label + "_validation", (artifact.validation_contract_ref, artifact.validation_contract_hash),
                 (validation.validation_contract_ref, validation.validation_contract_hash))
        evaluation = selected("representative_evaluation", representative_evaluation_pin, "runtime_semantic_evaluation_bundle",
                              RuntimeSemanticEvaluationBundleV1, "bundle_hash")
        same("evaluation_plan", (evaluation.plan_ref, evaluation.plan_hash), (compiled.plan_ref, compiled.plan_hash))
        paired = None if paired_suite_pin is None else selected(
            "paired_suite", paired_suite_pin, "ontology_paired_obligation_suite", OntologyPairedObligationSuiteV1, "suite_hash")
        if paired is not None:
            same("paired_source", (paired.source_identity_ref, paired.source_hash,
                                   paired.canonical_source_ir_ref, paired.canonical_source_ir_hash),
                 (boundary.source_identity_ref, boundary.source_hash, ir_pin.artifact_ref, ir_pin.artifact_hash))
            same("paired_intent", (paired.graph_intent_ref, paired.graph_intent_hash),
                 (intent_pin.artifact_ref, intent_pin.artifact_hash))
        return self.certify(
            store=store, command=command, origin_manifest=origin, origin_snapshot=snapshot,
            origin_boundary=boundary, authority=authority, competency=competency,
            schema=schema, schema_adequacy=adequacy, candidate=candidate, validation=validation,
            quality_evaluation=run, compiled_plan=compiled, execution_plan=execution,
            representative_evaluation=evaluation, paired_suite=paired, created_at=created_at,
        )

    def resolve_receipt(
        self, *, store: OntologyBuildStore, organization_id: str, workspace_id: str,
        receipt_pin: ArtifactPinV1, origin_pin: ArtifactPinV1,
    ) -> tuple[OntologyCertificationReceiptV1, OntologyBuildManifestV1 | None]:
        """Verify an exact persisted certification and its immutable successor."""
        if receipt_pin.artifact_kind != "ontology_certification_receipt.v1" or origin_pin.artifact_kind != "ontology_build_manifest":
            raise ValueError("CERTIFICATION_PIN_KIND_MISMATCH")
        scope = dict(organization_id=organization_id, workspace_id=workspace_id)
        receipt = store.get(**scope, artifact_ref=receipt_pin.artifact_ref, contract_type=OntologyCertificationReceiptV1)
        if receipt is None:
            raise ValueError("CERTIFICATION_RECEIPT_NOT_FOUND")
        basis = receipt.model_dump(mode="json", exclude={"schema_version", "receipt_ref", "receipt_hash"})
        basis["created_at"] = receipt.created_at.isoformat()
        identity = content_hash({"request_id": receipt.request_id, "request_hash": receipt.request_hash,
                                 "successor_build_hash": receipt.successor_build_hash})
        if receipt.receipt_hash != receipt_pin.artifact_hash or receipt.receipt_hash != "sha256:" + content_hash(basis) or receipt.receipt_ref != "ontology-certification-receipt:" + identity:
            raise ValueError("CERTIFICATION_RECEIPT_PIN_MISMATCH")
        if (receipt.origin_build_ref, receipt.origin_build_hash) != (origin_pin.artifact_ref, origin_pin.artifact_hash):
            raise ValueError("CERTIFICATION_ORIGIN_MISMATCH")
        origin = store.get(**scope, artifact_ref=origin_pin.artifact_ref, contract_type=OntologyBuildManifestV1)
        if origin is None or origin.build_hash != origin_pin.artifact_hash:
            raise ValueError("CERTIFICATION_ORIGIN_NOT_FOUND")
        if receipt.status != "CERTIFIED_BOUNDED":
            return receipt, None
        if receipt.blocker_refs or receipt.introduced_gap_ids:
            raise ValueError("CERTIFICATION_NOT_READY")
        successor = store.get(**scope, artifact_ref=receipt.successor_build_ref, contract_type=OntologyBuildManifestV1)
        if successor is None or successor.build_hash != receipt.successor_build_hash:
            raise ValueError("CERTIFICATION_SUCCESSOR_NOT_FOUND")
        if (successor.predecessor_build_ref, successor.predecessor_build_hash) != (origin.build_ref, origin.build_hash) or successor.assurance_profile != "A1_INTERNAL_POLICY_MVL":
            raise ValueError("CERTIFICATION_SUCCESSOR_ORIGIN_MISMATCH")
        def child(manifest, kind):
            pins = [pin for pin in manifest.child_artifacts if pin.artifact_kind == kind]
            if len(pins) != 1:
                raise ValueError("CERTIFICATION_REQUIRED_CHILD_MISSING:" + kind)
            return pins[0]
        for kind in ("source_identity", "canonical_source_ir", "graph_intent_brief"):
            if child(origin, kind) != child(successor, kind):
                raise ValueError("CERTIFICATION_SOURCE_MISMATCH")
        snapshot_pin = child(successor, "exploratory_graph_snapshot")
        if (snapshot_pin.artifact_ref, snapshot_pin.artifact_hash) != (receipt.successor_snapshot_ref, receipt.successor_snapshot_hash):
            raise ValueError("CERTIFICATION_SNAPSHOT_MISMATCH")
        snapshot, integrity = store.get_verified_exploratory_snapshot(**scope, pin=snapshot_pin)
        if snapshot.build_ref != successor.build_ref or snapshot.completeness_state != "bounded_complete" or snapshot.unresolved_gap_ids or integrity.producer_profile != self.PRODUCER_PROFILE:
            raise ValueError("CERTIFICATION_SNAPSHOT_NOT_READY")
        readiness_pin = child(successor, "ontology_certification_readiness")
        readiness = store.get(**scope, artifact_ref=readiness_pin.artifact_ref, contract_type=OntologyCertificationReadinessV1)
        if readiness is None or readiness.readiness_hash != readiness_pin.artifact_hash or (readiness.origin_build_ref, readiness.origin_build_hash) != (origin.build_ref, origin.build_hash) or readiness.compile_certifiable_blockers or readiness.certify_blockers:
            raise ValueError("CERTIFICATION_NOT_READY")
        validation_pin = child(successor, "quality_evaluation_run")
        validation = store.get(**scope, artifact_ref=validation_pin.artifact_ref, contract_type=QualityEvaluationRunV1)
        execution_pin = child(successor, "certified_semantic_execution_plan")
        execution = store.get(**scope, artifact_ref=execution_pin.artifact_ref, contract_type=CertifiedSemanticExecutionPlanV1)
        if validation is None or validation.run_hash != validation_pin.artifact_hash or validation.disposition not in {"QUALIFIED", "QUALIFIED_WITH_WARNINGS"}:
            raise ValueError("CERTIFICATION_VALIDATION_NOT_READY")
        if execution is None or execution.execution_plan_hash != execution_pin.artifact_hash or execution.assurance_profile != "A1_INTERNAL_POLICY_MVL" or execution.quality_evaluation != validation.pin():
            raise ValueError("CERTIFICATION_EXECUTION_PLAN_MISMATCH")
        certify_gates = [gate for gate in successor.gates if gate.operation == "CERTIFY"]
        if len(certify_gates) != 1 or certify_gates[0].decision != "PROCEED":
            raise ValueError("CERTIFICATION_NOT_READY")
        return receipt, successor

    @staticmethod
    def _pin(kind: str, ref: str, digest: str) -> ArtifactPinV1:
        return ArtifactPinV1(artifact_kind=kind, artifact_ref=ref, artifact_hash=digest)

    @staticmethod
    def _artifact_pin(artifact, kind: str, ref_field: str, hash_field: str) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind=kind,
            artifact_ref=str(getattr(artifact, ref_field)),
            artifact_hash=str(getattr(artifact, hash_field)),
        )

    @staticmethod
    def _evidence_refs(authority: NormativeAuthorityDependencySetV1) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                f"legal-section:{section.section_id}"
                for dependency in authority.dependencies
                for section in dependency.sections
            )
        )

    def _boundary(
        self,
        *,
        origin_boundary: CanonicalEvidenceBoundaryV1,
        created_at: datetime,
    ) -> CanonicalEvidenceBoundaryV1:
        basis = {
            "organization_id": origin_boundary.organization_id,
            "workspace_id": origin_boundary.workspace_id,
            "source_identity_ref": origin_boundary.source_identity_ref,
            "source_hash": origin_boundary.source_hash,
            "canonical_source_ir_ref": origin_boundary.canonical_source_ir_ref,
            "canonical_source_ir_hash": origin_boundary.canonical_source_ir_hash,
            "intended_domain": origin_boundary.intended_domain,
            "intended_use": origin_boundary.intended_use,
            "target_time": created_at.isoformat(),
            "jurisdiction": "CN",
            "explicit_exclusions": (
                "binding_legal_conclusion",
                "A3_REGULATORY_HIGH_ASSURANCE",
                "automatic_admission",
                "automatic_activation",
            ),
            "created_at": created_at.isoformat(),
        }
        digest = content_hash(basis)
        return CanonicalEvidenceBoundaryV1(
            boundary_ref=f"canonical-boundary:{digest}",
            boundary_hash=f"sha256:{digest}",
            created_at=created_at,
            **{key: value for key, value in basis.items() if key not in {"target_time", "created_at"}},
            target_time=created_at,
        )

    @staticmethod
    def _tested_dimensions(validation: ValidationContractV1, competency_id: str) -> set[str]:
        tested_obligation_ids = {
            ref for test in validation.executable_tests if test.competency_id == competency_id
            for ref in test.required_obligation_ids
        }
        return {item.semantic_dimension for item in validation.obligations
            if competency_id in item.competency_ids and item.obligation_id in tested_obligation_ids}

    def _gap_disposition(
        self, *, origin: ExploratoryGraphSnapshotV1, competency: CompetencyContractV1,
        schema_adequacy: SchemaAdequacyReportV1, candidate: NormativeSemanticCandidateSetV1,
        validation: ValidationContractV1, quality_evaluation: QualityEvaluationRunV1,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Close evidenced gaps; a reviewed downgrade limits scope, not history."""
        eligible = set()
        if (candidate.completeness_state == "COMPLETE" and validation.review_state == "accepted"
            and quality_evaluation.disposition in {"QUALIFIED", "QUALIFIED_WITH_WARNINGS"}):
            eligible.update({"gap:certification_not_started", "gap:normative_semantics_unverified"})
            rows = {row.competency_id: row for row in schema_adequacy.rows}
            for question in competency.questions:
                row = rows.get(question.competency_id)
                if (question.review_state == "confirmed" and row is not None
                    and row.representability == "REPRESENTABLE"
                    and set(question.required_semantic_dimensions).issubset(
                        self._tested_dimensions(validation, question.competency_id))):
                    eligible.add("gap:competency:" + question.competency_id)
        nonblocking = {"gap:competency:" + question.competency_id for question in competency.questions
            if (question.criticality, question.review_state) in {("noncritical", "confirmed"), ("excluded", "rejected")}}
        closed = tuple(gap for gap in origin.unresolved_gap_ids if gap in eligible)
        remaining = tuple(gap for gap in origin.unresolved_gap_ids if gap not in eligible)
        return closed, remaining, tuple(gap for gap in remaining if gap not in nonblocking)

    def _readiness(
        self,
        *,
        origin: OntologyBuildManifestV1,
        authority: NormativeAuthorityDependencySetV1,
        competency: CompetencyContractV1,
        schema_adequacy: SchemaAdequacyReportV1,
        validation: ValidationContractV1,
        quality_evaluation: QualityEvaluationRunV1,
        execution_plan: CertifiedSemanticExecutionPlanV1,
        assessed_at: datetime,
    ) -> OntologyCertificationReadinessV1:
        compile_blockers: list[str] = []
        certify_blockers: list[str] = []
        if not authority.dependencies:
            compile_blockers.append("readiness:authority-dependencies-missing")
        critical_questions = tuple(item for item in competency.questions if item.criticality == "critical")
        reviewed_states = {("critical", "confirmed"), ("noncritical", "confirmed"), ("excluded", "rejected")}
        if (not critical_questions
            or len({item.competency_id for item in competency.questions}) != len(competency.questions)
            or any((item.criticality, item.review_state) not in reviewed_states for item in competency.questions)):
            compile_blockers.append("readiness:critical-competency-unreviewed")
        if schema_adequacy.overall_decision != "PASS" or not schema_adequacy.certifiable_s3_allowed:
            compile_blockers.append("readiness:schema-adequacy-failed")
        if validation.review_state != "accepted":
            certify_blockers.append("readiness:validation-contract-unaccepted")
        for question in critical_questions:
            tested_dimensions = self._tested_dimensions(validation, question.competency_id)
            if not set(question.required_semantic_dimensions).issubset(tested_dimensions):
                certify_blockers.append("readiness:critical-competency-validation-coverage-missing")
                break
        if quality_evaluation.disposition not in {"QUALIFIED", "QUALIFIED_WITH_WARNINGS"}:
            certify_blockers.append("readiness:validation-run-failed")
        if (
            execution_plan.quality_evaluation != quality_evaluation.pin()
            or execution_plan.quality_profile != quality_evaluation.quality_profile
        ):
            certify_blockers.append("readiness:execution-plan-validation-pin-mismatch")
        evidence_refs = (
            authority.set_ref,
            competency.competency_contract_ref,
            schema_adequacy.report_ref,
            validation.validation_contract_ref,
            quality_evaluation.run_ref,
            execution_plan.execution_plan_ref,
        )
        basis = {
            "organization_id": origin.organization_id,
            "workspace_id": origin.workspace_id,
            "origin_build_ref": origin.build_ref,
            "origin_build_hash": origin.build_hash,
            "compile_certifiable_blockers": tuple(compile_blockers),
            "certify_blockers": tuple(certify_blockers),
            "admit_blockers": (),
            "activation_blockers": ("readiness:explicit-admission-and-activation-required",),
            "evidence_refs": evidence_refs,
            "assessed_at": assessed_at.isoformat(),
        }
        digest = content_hash(basis)
        return OntologyCertificationReadinessV1(
            readiness_ref=f"ontology-certification-readiness:{digest}",
            readiness_hash=f"sha256:{digest}",
            assessed_at=assessed_at,
            **{key: value for key, value in basis.items() if key != "assessed_at"},
        )

    def _snapshot(
        self,
        *,
        build_ref: str,
        origin: ExploratoryGraphSnapshotV1,
        candidate: NormativeSemanticCandidateSetV1,
        authority: NormativeAuthorityDependencySetV1,
        quality_evaluation: QualityEvaluationRunV1,
        closed_gap_ids: tuple[str, ...],
        created_at: datetime,
    ) -> ExploratoryGraphSnapshotV1:
        if not set(closed_gap_ids).issubset(origin.unresolved_gap_ids):
            raise ValueError("certification cannot close a gap absent from its origin")
        remaining_gaps = tuple(gap for gap in origin.unresolved_gap_ids if gap not in closed_gap_ids)
        nodes = list(origin.nodes)
        edges = list(origin.edges)
        node_ids = {item.node_id for item in nodes}
        edge_ids = {item.edge_id for item in edges}
        source_node_by_evidence: dict[str, str] = {}
        for node in origin.nodes:
            for evidence_ref in node.evidence_refs:
                source_node_by_evidence.setdefault(evidence_ref, node.node_id)
        for dependency in authority.dependencies:
            for section in dependency.sections:
                evidence_ref = f"legal-section:{section.section_id}"
                node_id = f"legal-authority:{section.section_id}"
                if node_id not in node_ids:
                    nodes.append(
                        ExploratoryGraphNodeV1(
                            node_id=node_id,
                            node_kind="authoritative_legal_section",
                            label=f"{dependency.source_id} · {section.stable_locator}",
                            evidence_refs=(evidence_ref,),
                        )
                    )
                    node_ids.add(node_id)
                source_node_by_evidence[evidence_ref] = node_id
        frame_node_ids: dict[str, str] = {}
        for frame in candidate.frames:
            node_id = f"normative-frame:{frame.frame_id}"
            frame_node_ids[frame.frame_id] = node_id
            if node_id not in node_ids:
                nodes.append(
                    ExploratoryGraphNodeV1(
                        node_id=node_id,
                        node_kind=f"certified_{frame.frame_kind}",
                        label=frame.display_name,
                        evidence_refs=frame.evidence_refs,
                    )
                )
                node_ids.add(node_id)
            for evidence_ref in frame.evidence_refs:
                target = source_node_by_evidence.get(evidence_ref)
                if target is None:
                    continue
                edge_basis = {"frame": frame.frame_id, "evidence": evidence_ref, "target": target}
                digest = content_hash(edge_basis)
                edge_id = f"certified-evidence:{digest}"
                if edge_id in edge_ids:
                    continue
                edges.append(
                    ExploratoryGraphEdgeV1(
                        edge_id=edge_id,
                        source_node_id=node_id,
                        target_node_id=target,
                        relation_type="evidenced_by",
                        structural_roles=("evidence",),
                        evidence_refs=(evidence_ref,),
                    )
                )
                edge_ids.add(edge_id)
        for frame in candidate.frames:
            source_id = frame_node_ids[frame.frame_id]
            refs: Iterable[str] = ()
            if frame.frame_kind == "rule":
                refs = (
                    *frame.exception_refs,
                    *frame.scope_refs,
                    *frame.qualification_refs,
                    *frame.temporal_condition_refs,
                    *frame.precedence_refs,
                )
            elif frame.frame_kind in {"exception", "scope"}:
                refs = (frame.applies_to_rule_ref,)
            elif frame.frame_kind == "precedence":
                refs = (frame.higher_priority_ref, frame.lower_priority_ref)
            for target_ref in refs:
                target_id = frame_node_ids.get(target_ref)
                if target_id is None:
                    continue
                basis = {"source": frame.frame_id, "target": target_ref}
                digest = content_hash(basis)
                edge_id = f"certified-semantic:{digest}"
                if edge_id in edge_ids:
                    continue
                edges.append(
                    ExploratoryGraphEdgeV1(
                        edge_id=edge_id,
                        source_node_id=source_id,
                        target_node_id=target_id,
                        relation_type="certified_semantic_link",
                        structural_roles=("dependency",),
                        evidence_refs=frame.evidence_refs,
                    )
                )
                edge_ids.add(edge_id)
        semantic_basis = {
            "schema_version": "exploratory_graph_snapshot.v1",
            "organization_id": origin.organization_id,
            "workspace_id": origin.workspace_id,
            "build_ref": build_ref,
            "source_identity_ref": origin.source_identity_ref,
            "source_hash": origin.source_hash,
            "canonical_source_ir_ref": origin.canonical_source_ir_ref,
            "canonical_source_ir_hash": origin.canonical_source_ir_hash,
            "authority_state": "provisional",
            "completeness_state": "bounded_complete",
            "nodes": [item.model_dump(mode="json") for item in nodes],
            "edges": [item.model_dump(mode="json") for item in edges],
            "unresolved_gap_ids": remaining_gaps,
            "predecessor_snapshot_ref": origin.snapshot_ref,
            "predecessor_snapshot_hash": origin.snapshot_hash,
            "change_cause": "schema_rebuild",
            "triggering_ref": quality_evaluation.run_ref,
            "created_at": created_at.isoformat().replace("+00:00", "Z"),
        }
        digest = content_hash(semantic_basis)
        return ExploratoryGraphSnapshotV1(
            snapshot_ref=f"exploratory-snapshot:{digest}",
            snapshot_hash=f"sha256:{digest}",
            organization_id=origin.organization_id,
            workspace_id=origin.workspace_id,
            build_ref=build_ref,
            source_identity_ref=origin.source_identity_ref,
            source_hash=origin.source_hash,
            canonical_source_ir_ref=origin.canonical_source_ir_ref,
            canonical_source_ir_hash=origin.canonical_source_ir_hash,
            authority_state="provisional",
            completeness_state="bounded_complete",
            nodes=tuple(nodes),
            edges=tuple(edges),
            unresolved_gap_ids=remaining_gaps,
            predecessor_snapshot_ref=origin.snapshot_ref,
            predecessor_snapshot_hash=origin.snapshot_hash,
            change_cause="schema_rebuild",
            triggering_ref=quality_evaluation.run_ref,
            created_at=created_at,
        )

    def _lineage_history(
        self,
        *,
        store: OntologyBuildStore,
        origin_manifest: OntologyBuildManifestV1,
        origin_snapshot: ExploratoryGraphSnapshotV1,
    ) -> tuple[ExploratoryGraphSnapshotV1, ...]:
        history = [origin_snapshot]
        current = origin_manifest
        seen = {current.build_ref}
        while current.predecessor_build_ref is not None:
            predecessor = store.get(
                organization_id=current.organization_id,
                workspace_id=current.workspace_id,
                artifact_ref=current.predecessor_build_ref,
                contract_type=OntologyBuildManifestV1,
            )
            if predecessor is None or predecessor.build_hash != current.predecessor_build_hash:
                raise ValueError("certification lineage predecessor manifest is missing or invalid")
            if predecessor.build_ref in seen:
                raise ValueError("certification lineage contains a cycle")
            seen.add(predecessor.build_ref)
            pins = tuple(
                pin for pin in predecessor.child_artifacts
                if pin.artifact_kind == "exploratory_graph_snapshot"
            )
            if len(pins) != 1:
                raise ValueError("certification lineage predecessor requires one snapshot")
            snapshot, _ = store.get_verified_exploratory_snapshot(
                organization_id=predecessor.organization_id,
                workspace_id=predecessor.workspace_id,
                pin=pins[0],
            )
            history.append(snapshot)
            current = predecessor
        history.reverse()
        return tuple(history)

    def certify(
        self,
        *,
        store: OntologyBuildStore,
        command: OntologyCertificationCommandV1,
        origin_manifest: OntologyBuildManifestV1,
        origin_snapshot: ExploratoryGraphSnapshotV1,
        origin_boundary: CanonicalEvidenceBoundaryV1,
        authority: NormativeAuthorityDependencySetV1,
        competency: CompetencyContractV1,
        schema: NormativeSchemaContractV1,
        schema_adequacy: SchemaAdequacyReportV1,
        candidate: NormativeSemanticCandidateSetV1,
        validation: ValidationContractV1,
        quality_evaluation: QualityEvaluationRunV1,
        compiled_plan: CompiledRulePlanV1,
        execution_plan: CertifiedSemanticExecutionPlanV1,
        representative_evaluation: RuntimeSemanticEvaluationBundleV1,
        paired_suite: OntologyPairedObligationSuiteV1 | None = None,
        created_at: datetime,
    ) -> tuple[OntologyBuildManifestV1, OntologyCertificationReceiptV1]:
        if schema.teaching_candidate or candidate.teaching_candidate or compiled_plan.teaching_candidate:
            raise ValueError("TEACHING_CANDIDATE_NOT_CERTIFIABLE")
        if (command.origin_build_ref, command.origin_build_hash) != (
            origin_manifest.build_ref,
            origin_manifest.build_hash,
        ):
            raise ValueError("certification command origin build pin mismatch")
        if (command.origin_snapshot_ref, command.origin_snapshot_hash) != (
            origin_snapshot.snapshot_ref,
            origin_snapshot.snapshot_hash,
        ):
            raise ValueError("certification command origin snapshot pin mismatch")
        if (command.reviewed_competency_contract_ref, command.reviewed_competency_contract_hash) != (
            competency.competency_contract_ref,
            competency.competency_contract_hash,
        ):
            raise ValueError("certification command competency pin mismatch")
        if (command.authority_dependency_set_ref, command.authority_dependency_set_hash) != (
            authority.set_ref,
            authority.set_hash,
        ):
            raise ValueError("certification command authority pin mismatch")
        if origin_manifest.assurance_profile not in {
            "A0_EXPLORATORY", "A1_INTERNAL_POLICY_MVL"
        }:
            raise ValueError("certification origin must be A0 or an existing A1 successor")
        if origin_manifest.assurance_profile == "A1_INTERNAL_POLICY_MVL" and (
            origin_snapshot.completeness_state != "bounded_complete"
            or not any(
                pin.artifact_kind == "ontology_certification_readiness"
                for pin in origin_manifest.child_artifacts
            )
        ):
            raise ValueError("A1 correction origin must already be certification-ready")
        if quality_evaluation.disposition not in {"QUALIFIED", "QUALIFIED_WITH_WARNINGS"}:
            raise ValueError("certification requires qualified quality evaluation")
        OntologySchemaAdequacyService().verify_report(
            schema=schema, questions=competency.questions, report=schema_adequacy,
        )
        successor_boundary = self._boundary(origin_boundary=origin_boundary, created_at=created_at)
        assessment = OntologyEvidenceAdequacyService().assess(
            boundary=successor_boundary,
            source_authority_role="interpretive_with_authoritative_dependencies",
            available_definition_regions=("scope", "rule", "qualification", "exception", "effectivity"),
            missing_normative_dependencies=(),
            search_evidence_refs=(authority.search_coverage_ref, *self._evidence_refs(authority)),
            authoritative_dependency_refs=tuple(item.version_id for item in authority.dependencies),
            assessed_at=created_at,
        )
        readiness = self._readiness(
            origin=origin_manifest,
            authority=authority,
            competency=competency,
            schema_adequacy=schema_adequacy,
            validation=validation,
            quality_evaluation=quality_evaluation,
            execution_plan=execution_plan,
            assessed_at=created_at,
        )
        if assessment.findings or readiness.compile_certifiable_blockers or readiness.certify_blockers:
            raise ValueError("certification readiness remains blocked")
        closed_gap_ids, _, blocking_gaps = self._gap_disposition(origin=origin_snapshot,
            competency=competency, schema_adequacy=schema_adequacy, candidate=candidate,
            validation=validation, quality_evaluation=quality_evaluation)
        if blocking_gaps:
            raise ValueError("CERTIFICATION_UNRESOLVED_MATERIAL_GAPS:" + ",".join(blocking_gaps))
        for artifact, model, ref, hash_ in (
            (quality_evaluation, QualityEvaluationRunV1, 'run_ref', 'run_hash'),
            (quality_evaluation.profile_definition, QualityProfileV1, 'profile_ref', 'profile_hash')):
            persisted = store.get(organization_id=origin_manifest.organization_id, workspace_id=origin_manifest.workspace_id,
                artifact_ref=getattr(artifact, ref), contract_type=model)
            if persisted is None or persisted != artifact:
                raise ValueError('certification requires pre-existing exact quality evidence')
        from services.ontology_a1_certification_evaluator import OntologyA1CertificationEvaluator
        OntologyA1CertificationEvaluator().verify_execution_inputs(
            quality_evaluation=quality_evaluation, validation=validation, candidate=candidate, plan=compiled_plan, store=store)
        projection_compiler_hash = "sha256:" + sha256(Path(__file__).read_bytes()).hexdigest()
        build_basis = {
            "projection_compiler_ref": self.PRODUCER_VERSION,
            "projection_compiler_hash": projection_compiler_hash,
            "origin_build_ref": origin_manifest.build_ref,
            "origin_build_hash": origin_manifest.build_hash,
            "authority_set_hash": authority.set_hash,
            "competency_hash": competency.competency_contract_hash,
            "schema_hash": schema.schema_hash,
            "schema_adequacy_hash": schema_adequacy.report_hash,
            "candidate_hash": candidate.candidate_set_hash,
            "validation_hash": validation.validation_contract_hash,
            "quality_evaluation_hash": quality_evaluation.run_hash,
            "compiled_plan_hash": compiled_plan.plan_hash,
            "execution_plan_hash": execution_plan.execution_plan_hash,
            "paired_suite_hash": paired_suite.suite_hash if paired_suite is not None else None,
            "assurance_profile": command.target_assurance_profile,
        }
        build_digest = content_hash(build_basis)
        build_ref = f"ontology-build:{build_digest}"
        existing = store.get(
            organization_id=origin_manifest.organization_id,
            workspace_id=origin_manifest.workspace_id,
            artifact_ref=build_ref,
            contract_type=OntologyBuildManifestV1,
        )
        request_hash = "sha256:" + content_hash(command.model_dump(mode="json"))
        receipt_identity = content_hash(
            {
                "request_id": command.request_id,
                "request_hash": request_hash,
                "successor_build_hash": existing.build_hash if existing is not None else None,
            }
        ) if existing is not None else None
        if existing is not None:
            receipt_ref = f"ontology-certification-receipt:{receipt_identity}"
            receipt = store.get(
                organization_id=origin_manifest.organization_id,
                workspace_id=origin_manifest.workspace_id,
                artifact_ref=receipt_ref,
                contract_type=OntologyCertificationReceiptV1,
            )
            if receipt is None:
                raise ValueError("certification successor exists without matching receipt")
            return existing, receipt
        successor = self._snapshot(
            build_ref=build_ref,
            origin=origin_snapshot,
            candidate=candidate,
            authority=authority,
            quality_evaluation=quality_evaluation,
            closed_gap_ids=closed_gap_ids,
            created_at=created_at,
        )
        lineage_service = OntologyLineageService()
        lineage_history = self._lineage_history(
            store=store,
            origin_manifest=origin_manifest,
            origin_snapshot=origin_snapshot,
        )
        lineage: CandidateGraphLineageProjectionV1 = lineage_service.project(
            snapshots=(*lineage_history, successor)
        )
        diff: CandidateGraphDiffProjectionV1 = lineage_service.compare(
            baseline=origin_snapshot,
            successor=successor,
        )
        origin_by_kind = {pin.artifact_kind: pin for pin in origin_manifest.child_artifacts}
        additional_items = [
            self._artifact_pin(successor_boundary, "canonical_evidence_boundary", "boundary_ref", "boundary_hash"),
            self._artifact_pin(successor, "exploratory_graph_snapshot", "snapshot_ref", "snapshot_hash"),
            quality_evaluation.quality_profile,
            origin_by_kind["graph_intent_brief"],
            origin_by_kind["source_identity"],
            origin_by_kind["canonical_source_ir"],
            self._artifact_pin(authority, "normative_authority_dependency_set", "set_ref", "set_hash"),
            self._pin("source_search_coverage_manifest", authority.search_coverage_ref, authority.search_coverage_hash),
            self._artifact_pin(competency, "competency_contract", "competency_contract_ref", "competency_contract_hash"),
            self._artifact_pin(schema, "normative_schema_contract", "schema_ref", "schema_hash"),
            self._artifact_pin(schema_adequacy, "schema_adequacy_report", "report_ref", "report_hash"),
            self._artifact_pin(candidate, "normative_semantic_candidate_set", "candidate_set_ref", "candidate_set_hash"),
            self._artifact_pin(validation, "validation_contract", "validation_contract_ref", "validation_contract_hash"),
            self._artifact_pin(quality_evaluation, "quality_evaluation_run", "run_ref", "run_hash"),
            self._artifact_pin(compiled_plan, "compiled_rule_plan", "plan_ref", "plan_hash"),
            self._artifact_pin(execution_plan, "certified_semantic_execution_plan", "execution_plan_ref", "execution_plan_hash"),
            self._artifact_pin(representative_evaluation, "runtime_semantic_evaluation_bundle", "bundle_ref", "bundle_hash"),
            self._artifact_pin(lineage, "candidate_graph_lineage_projection", "lineage_ref", "lineage_hash"),
            self._artifact_pin(diff, "candidate_graph_diff_projection", "diff_ref", "diff_hash"),
        ]
        if paired_suite is not None:
            additional_items.append(
                self._artifact_pin(
                    paired_suite,
                    "ontology_paired_obligation_suite",
                    "suite_ref",
                    "suite_hash",
                )
            )
        additional = tuple(additional_items)
        manifest = OntologyBuildCoordinator().project_certification_manifest(
            build_ref=build_ref,
            organization_id=origin_manifest.organization_id,
            workspace_id=origin_manifest.workspace_id,
            assurance_profile=command.target_assurance_profile,
            assessment=assessment,
            readiness=readiness,
            additional_artifacts=additional,
            created_at=created_at,
            predecessor_build_ref=origin_manifest.build_ref,
            predecessor_build_hash=origin_manifest.build_hash,
        )
        receipt_basis = {
            "organization_id": origin_manifest.organization_id,
            "workspace_id": origin_manifest.workspace_id,
            "request_id": command.request_id,
            "request_hash": request_hash,
            "origin_build_ref": origin_manifest.build_ref,
            "origin_build_hash": origin_manifest.build_hash,
            "successor_build_ref": manifest.build_ref,
            "successor_build_hash": manifest.build_hash,
            "successor_snapshot_ref": successor.snapshot_ref,
            "successor_snapshot_hash": successor.snapshot_hash,
            "status": "CERTIFIED_BOUNDED",
            "blocker_refs": (),
            "closed_gap_ids": closed_gap_ids,
            "introduced_gap_ids": (),
            "target_assurance_profile": command.target_assurance_profile,
            "certified_scope": "bounded_internal_tender_restriction_screening",
            "prohibited_claims": (
                "binding_legal_conclusion",
                "A3_REGULATORY_HIGH_ASSURANCE",
                "automatic_admission",
                "automatic_activation",
            ),
            "evidence_refs": (
                authority.set_ref,
                competency.competency_contract_ref,
                schema.schema_ref,
                schema_adequacy.report_ref,
                validation.validation_contract_ref,
                quality_evaluation.run_ref,
                execution_plan.execution_plan_ref,
            ),
            "created_at": created_at.isoformat(),
        }
        receipt_digest = content_hash(receipt_basis)
        receipt_identity = content_hash(
            {
                "request_id": command.request_id,
                "request_hash": request_hash,
                "successor_build_hash": manifest.build_hash,
            }
        )
        receipt = OntologyCertificationReceiptV1(
            receipt_ref=f"ontology-certification-receipt:{receipt_identity}",
            receipt_hash=f"sha256:{receipt_digest}",
            created_at=created_at,
            **{key: value for key, value in receipt_basis.items() if key != "created_at"},
        )
        integrity = create_exploratory_snapshot_integrity(
            snapshot=successor,
            producer_profile=self.PRODUCER_PROFILE,
            producer_version=self.PRODUCER_VERSION,
            recorded_at=created_at,
        )
        store.put_many_with_integrity(
            artifacts=(
                successor_boundary,
                assessment,
                readiness,
                successor,
                lineage,
                diff,
                manifest,
                receipt,
            ),
            snapshot_integrities=(integrity,),
        )
        return manifest, receipt


__all__ = ["OntologyCertificationSuccessorService"]
