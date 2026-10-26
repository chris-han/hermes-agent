"""Append-only SQLite persistence for ontology-build companion artifacts."""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from storage.hosted_postgres import enabled,require_tables,sql_query,begin_write,in_transaction,integrity_errors
from threading import RLock

_HR_ATTEMPT_LOCK = RLock()
_HR_ACTIVE_ATTEMPTS: set[str] = set()
from typing import TYPE_CHECKING, Callable, Iterator, TypeVar

from pydantic import BaseModel, ValidationError
from contracts.jd_compliance_review import JdReviewV1, JdBasisEvidenceV1, JdFeedbackV1
from contracts.observed_constraint_hypothesis import EffectiveRestrictionRuntimeContextV1
from contracts.runtime_semantic_inspection_result import RuntimeSemanticInspectionResultV1, RuntimeSemanticInspectionContextV1
from contracts import semantic_kernel as kernel_contracts
from contracts.change_candidate_v2 import ChangeCandidateV2
from contracts.learning_workbench import LEARNING_ARTIFACT_MODELS as SKILL_ARTIFACT_MODELS
from contracts.learning_trace_evidence import LEARNING_TRACE_ARTIFACT_MODELS
from contracts.retained_context_items import RETAINED_CONTEXT_ARTIFACT_MODELS
from contracts.learned_skill_publication import LEARNED_SKILL_ARTIFACT_MODELS
from contracts.project_memory_state import ProjectStateProjectionV1, TypedStateDecisionV1
from contracts.business_answers_receivables import RECEIVABLES_MODELS

from contracts.runtime_context_reconstruction import (
    ARTIFACT_SCHEMA_REGISTRY,
    UtcDateTime,
)

from contracts.ontology_artifact_integrity import (
    ExploratoryGraphSnapshotIntegrityV1,
)
from contracts.normative_semantics import NormativeSemanticCandidateSetV1
from contracts.runtime_restriction_perception import (
    RuntimeRestrictionPerceptionRequestV1, RuntimeRestrictionPerceptionAttemptV1,
    RuntimeRestrictionPerceptionResponseV1, RuntimeRestrictionPerceptionResponseV2,
)
from contracts.runtime_judgment_execution import (
    RuntimeJudgmentAttemptV1,
    RuntimeJudgmentRequestV1,
    RuntimeJudgmentResponseV1,
)
from contracts.normative_schema import SourceSearchCoverageManifestV1
from contracts.ontology_build_inputs import OntologyBuildInputEnvelopeV1
from contracts.ontology_certification import (
    NormativeAuthorityDependencySetV1,
    OntologyCertificationReadinessV1,
    OntologyCertificationReceiptV1,
)
from contracts.ontology_building import (
    ArtifactPinV1,
    CanonicalEvidenceBoundaryV1,
    CompetencyContractV1,
    ExploratoryGraphSnapshotV1,
    OntologyBuildManifestV1,
)
from contracts.ontology_paired_evaluation import (
    ObservedFactsV1,
    OntologyPairedEvaluationRunV1,
    OntologyPairedObligationSuiteV1,
    PairedEvaluationSubjectV1,
    PairedEvaluatorBundleV1,
)
from contracts.ontology_validation import CompiledRulePlanV1
from contracts.tender_document_date_observation import TenderDocumentDateObservationV1
from contracts.tender_observed_facts_projection import (
    TenderFactObservationReceiptV1, TenderCategoricalFactObservationReceiptV1,
    TenderFactObservationReceiptV2, TenderObservedFactsProjectionReceiptV2,
)
from contracts.ontology_repair import (
    AdvisoryNextProposalCommandV1,
    AdvisoryRepairProposalV1,
    AdvisoryNextProposalReceiptV1,
)
from contracts.runtime_learning_paired_evaluation import (
    RuntimeRepairObligationSuiteV1,
    RuntimeRepairPairV1,
    RuntimeRepairPairedEvaluationRunV1,
    RuntimeRepairPairedEvaluationReportV1,
    RuntimeRepairPairedComparisonViewV1,
)
from contracts import runtime_inspection_report as report_contracts
from contracts import runtime_report_export as export_contracts
from contracts.runtime_learning_trial import (
    AdvisoryCompilerInputsV1,
    AdvisoryPairedSuiteReviewV1,
    AdvisoryTrialBuildInputsV1,
    AdvisoryTrialVariantV1,
)
from contracts.runtime_producer_integrity import PRODUCER_RECORD_MODELS
from contracts.runtime_semantic_evaluation import (
    RuntimeEvaluationDiagnosticV1,
    RuntimeEvaluationRequestOutcomeV1,
    RuntimeEvaluationRunV1,
    RuntimeEvaluationStateV1,
    RuntimeExecutionAdmissionV1,
    RuntimeTemporalCaptureV1,
    RuntimeTemporalProviderDescriptorV1,
    RuntimeTemporalProviderProfileV1,
    RuntimeTemporalProviderProfileV2,
    RuntimeTemporalProviderRequestV1,
    RuntimeTemporalSupportedResultV1,
    RuntimeTemporalUnavailableResultV1,
    RuntimeTemporalWorldSnapshotV1,
    TemporalQueryV1,
)
from contracts.runtime_temporal_source import (
    RuntimeTemporalContextSelectionV1,
    RuntimeTemporalNormalizationV1,
    RuntimeTemporalSourceArchiveV1,
    RuntimeTemporalSourceGraphV1,
)
from contracts.runtime_working_context import (
    RuntimeContextTransitionReceiptV3,
    RuntimeTaskContextV3,
)
from contracts.validation_contract import ValidationContractV1
from contracts import ontology_workbench_context as artifact_contracts
from contracts.etl_pipeline import EtlReceipt
from eos.artifact_hashing import canonical_json, content_hash

if TYPE_CHECKING:
    from services.runtime_learning_trial_service import (
        AdvisoryTrialOwnerContext,
        ControlledTrialCaseId,
    )


_ADVISORY_PUBLICATION_INDEX = "advisory_trial_native_publication_index"
_ADVISORY_INPUT_SCHEMA = "advisory_compiler_inputs.v1"
_ADVISORY_SUITE_REVIEW_SCHEMA = "advisory_paired_suite_review.v1"
_ADVISORY_SUITE_REVIEWER = "trial-preparation:controlled_internal_policy_v1"
_ADVISORY_SUITE_DECISION_PREFIX = "advisory-paired-suite-review:"
_ADVISORY_SUITE_ACTIVATION_VERSION = 1
_ADVISORY_INITIAL_OUTPUTS = (
    (
        "initial_candidate",
        "normative_semantic_candidate_set",
        NormativeSemanticCandidateSetV1,
        "candidate_set_ref",
        "candidate_set_hash",
    ),
    (
        "initial_validation",
        "validation_contract",
        ValidationContractV1,
        "validation_contract_ref",
        "validation_contract_hash",
    ),
    ("initial_plan", "compiled_rule_plan", CompiledRulePlanV1, "plan_ref", "plan_hash"),
)
_ADVISORY_TRIAL_OUTPUTS = (
    (
        "trial_candidate",
        "normative_semantic_candidate_set",
        NormativeSemanticCandidateSetV1,
        "candidate_set_ref",
        "candidate_set_hash",
    ),
    ("trial_plan", "compiled_rule_plan", CompiledRulePlanV1, "plan_ref", "plan_hash"),
    (
        "trial_derivation",
        "advisory_trial_variant",
        AdvisoryTrialVariantV1,
        "variant_ref",
        "variant_hash",
    ),
)

_ADVISORY_PUBLICATION_ROLES = (
    "compiler_inputs",
    "initial_candidate",
    "initial_validation",
    "initial_plan",
    "trial_candidate",
    "trial_plan",
    "trial_derivation",
    "observed_facts",
)
_ADVISORY_OLD_PUBLICATION_ROLES = _ADVISORY_PUBLICATION_ROLES[:-1]
_ADVISORY_INDEX_COLUMNS = (
    "organization_id",
    "workspace_id",
    "publication_kind",
    "artifact_ref",
    "artifact_hash",
    "payload_hash",
    "input_ref",
    "input_hash",
    "derivation_ref",
    "derivation_hash",
    "subject_ref",
    "subject_hash",
)


def _advisory_publication_table_sql(*, include_observed_facts: bool) -> str:
    roles = (
        _ADVISORY_PUBLICATION_ROLES
        if include_observed_facts
        else _ADVISORY_OLD_PUBLICATION_ROLES
    )
    role_sql = ", ".join(f"'{role}'" for role in roles)
    initial_roles = (
        "'compiler_inputs', 'initial_candidate', 'initial_validation', 'initial_plan'"
    )
    if include_observed_facts:
        initial_roles += ", 'observed_facts'"
    return f"""
        CREATE TABLE {_ADVISORY_PUBLICATION_INDEX} (
            organization_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            publication_kind TEXT NOT NULL CHECK (publication_kind IN ({role_sql})),
            artifact_ref TEXT NOT NULL,
            artifact_hash TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            input_ref TEXT NOT NULL,
            input_hash TEXT NOT NULL,
            derivation_ref TEXT,
            derivation_hash TEXT,
            subject_ref TEXT NOT NULL,
            subject_hash TEXT NOT NULL,
            PRIMARY KEY (organization_id, workspace_id, publication_kind, artifact_ref),
            CHECK (
                (publication_kind IN ({initial_roles})
                 AND derivation_ref IS NULL AND derivation_hash IS NULL)
                OR (publication_kind IN ('trial_candidate', 'trial_plan', 'trial_derivation')
                    AND derivation_ref IS NOT NULL AND derivation_hash IS NOT NULL)
            )
        )
    """


_ADVISORY_INDEX_SQL = {
    "advisory_trial_derivation_subject_unique": f"""
        CREATE UNIQUE INDEX advisory_trial_derivation_subject_unique
        ON {_ADVISORY_PUBLICATION_INDEX}(organization_id, workspace_id, subject_ref)
        WHERE publication_kind = 'trial_derivation'
    """,
    "advisory_initial_input_unique": f"""
        CREATE UNIQUE INDEX advisory_initial_input_unique
        ON {_ADVISORY_PUBLICATION_INDEX}(
            organization_id, workspace_id, publication_kind, input_ref)
        WHERE publication_kind IN ('initial_candidate', 'initial_validation', 'initial_plan')
    """,
    "advisory_trial_derivation_unique": f"""
        CREATE UNIQUE INDEX advisory_trial_derivation_unique
        ON {_ADVISORY_PUBLICATION_INDEX}(
            organization_id, workspace_id, publication_kind, derivation_ref)
        WHERE publication_kind IN ('trial_candidate', 'trial_plan', 'trial_derivation')
    """,
}


def _normalized_schema_sql(value: str) -> str:
    normalized = " ".join(value.replace("IF NOT EXISTS ", "").split())
    return normalized.replace("( ", "(").replace(" )", ")").replace(", ", ",")


def _advisory_index_rows(connection: sqlite3.Connection) -> list[tuple]:
    return connection.execute(
        f"SELECT {', '.join(_ADVISORY_INDEX_COLUMNS)} "
        f"FROM {_ADVISORY_PUBLICATION_INDEX} "
        f"ORDER BY {', '.join(_ADVISORY_INDEX_COLUMNS)}"
    ).fetchall()


def _validate_advisory_publication_schema(
    connection: sqlite3.Connection, *, include_observed_facts: bool
) -> None:
    table = connection.execute(
        sql_query("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?","SELECT sql FROM sqlite_master WHERE type = 'table' AND name = %s"),
        (_ADVISORY_PUBLICATION_INDEX,),
    ).fetchone()
    if table is None or _normalized_schema_sql(table[0]) != _normalized_schema_sql(
        _advisory_publication_table_sql(include_observed_facts=include_observed_facts)
    ):
        raise RuntimeError("unsupported advisory publication index schema")
    info = connection.execute(
        f"PRAGMA table_info({_ADVISORY_PUBLICATION_INDEX})"
    ).fetchall()
    expected_info = [
        (
            index,
            name,
            "TEXT",
            1 if name not in {"derivation_ref", "derivation_hash"} else 0,
            None,
            index + 1 if index < 4 else 0,
        )
        for index, name in enumerate(_ADVISORY_INDEX_COLUMNS)
    ]
    if info != expected_info:
        raise RuntimeError("unsupported advisory publication index columns")
    indexes = {
        name: sql
        for name, sql in connection.execute(
            sql_query("SELECT name, sql FROM sqlite_master "
            "WHERE type = 'index' AND tbl_name = ? AND sql IS NOT NULL","SELECT name, sql FROM sqlite_master WHERE type = 'index' AND tbl_name = %s AND sql IS NOT NULL"),
            (_ADVISORY_PUBLICATION_INDEX,),
        ).fetchall()
    }
    if set(indexes) != set(_ADVISORY_INDEX_SQL) or any(
        _normalized_schema_sql(indexes[name]) != _normalized_schema_sql(expected_sql)
        for name, expected_sql in _ADVISORY_INDEX_SQL.items()
    ):
        raise RuntimeError("unsupported advisory publication index definitions")


def _create_advisory_publication_schema(connection: sqlite3.Connection) -> None:
    connection.execute(_advisory_publication_table_sql(include_observed_facts=True))
    for sql in _ADVISORY_INDEX_SQL.values():
        connection.execute(sql)


def _ensure_advisory_publication_schema(connection: sqlite3.Connection) -> None:
    present = connection.execute(
        sql_query("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?","SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = %s"),
        (_ADVISORY_PUBLICATION_INDEX,),
    ).fetchone()
    if present is None:
        begin_write(connection)
        try:
            _create_advisory_publication_schema(connection)
            _validate_advisory_publication_schema(
                connection, include_observed_facts=True
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        return
    try:
        _validate_advisory_publication_schema(connection, include_observed_facts=True)
        return
    except RuntimeError:
        _validate_advisory_publication_schema(connection, include_observed_facts=False)

    before = _advisory_index_rows(connection)
    old_name = f"{_ADVISORY_PUBLICATION_INDEX}_seven_role_migration"
    begin_write(connection)
    try:
        for name in _ADVISORY_INDEX_SQL:
            connection.execute(f"DROP INDEX {name}")
        connection.execute(
            f"ALTER TABLE {_ADVISORY_PUBLICATION_INDEX} RENAME TO {old_name}"
        )
        _create_advisory_publication_schema(connection)
        columns = ", ".join(_ADVISORY_INDEX_COLUMNS)
        connection.execute(
            f"INSERT INTO {_ADVISORY_PUBLICATION_INDEX} ({columns}) "
            f"SELECT {columns} FROM {old_name}"
        )
        after = _advisory_index_rows(connection)
        if after != before or len(after) != len(before):
            raise RuntimeError("advisory publication migration changed rows")
        connection.execute(f"DROP TABLE {old_name}")
        _validate_advisory_publication_schema(connection, include_observed_facts=True)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def _validate_existing_advisory_suite_publications(
    connection: sqlite3.Connection,
) -> None:
    """Validate exact one-to-one protected pairs after owner activation."""

    from services._runtime_context_advisory.paired_inputs import _suite_from_subject

    rows = connection.execute(
        sql_query("""SELECT organization_id, workspace_id, artifact_ref, artifact_hash,
                  schema_version, payload_json
           FROM ontology_build_artifacts
           WHERE schema_version IN (?, ?)""",'SELECT organization_id, workspace_id, artifact_ref, artifact_hash,\n                  schema_version, payload_json\n           FROM ontology_build_artifacts\n           WHERE schema_version IN (%s, %s)'),
        (_ADVISORY_SUITE_REVIEW_SCHEMA, "ontology_paired_obligation_suite.v1"),
    ).fetchall()
    by_ref = {row[2]: row for row in rows}
    expected_suites: dict[str, tuple[str, str, str, str, str, str]] = {}
    for row in rows:
        if row[4] != _ADVISORY_SUITE_REVIEW_SCHEMA:
            continue
        try:
            receipt = AdvisoryPairedSuiteReviewV1.model_validate_json(row[5])
            suite = _suite_from_subject(receipt)
        except (ValidationError, ValueError) as exc:
            raise RuntimeError("unsupported preseeded advisory suite review") from exc
        suite_row = by_ref.get(suite.suite_ref)
        if (
            row[:5]
            != (
                receipt.organization_id,
                receipt.workspace_id,
                receipt.decision_ref,
                receipt.decision_hash,
                _ADVISORY_SUITE_REVIEW_SCHEMA,
            )
            or row[5] != canonical_json(receipt.model_dump(mode="json"))
            or suite_row is None
            or suite_row[:5]
            != (
                suite.organization_id,
                suite.workspace_id,
                suite.suite_ref,
                suite.suite_hash,
                "ontology_paired_obligation_suite.v1",
            )
            or suite_row[5] != canonical_json(suite.model_dump(mode="json"))
        ):
            raise RuntimeError("unsupported preseeded advisory suite review")
        expected_suites[receipt.decision_ref] = suite_row

    validated_decisions: set[str] = set()
    for row in rows:
        if row[4] != "ontology_paired_obligation_suite.v1":
            continue
        try:
            payload = json.loads(row[5])
        except json.JSONDecodeError as exc:
            raise RuntimeError("unsupported preseeded advisory suite") from exc
        controlled = payload.get("reviewer_ref") == _ADVISORY_SUITE_REVIEWER or str(
            payload.get("decision_ref", "")
        ).startswith(_ADVISORY_SUITE_DECISION_PREFIX)
        if not controlled:
            continue
        decision_ref = payload.get("decision_ref")
        if (
            decision_ref in validated_decisions
            or expected_suites.get(decision_ref) != row
        ):
            raise RuntimeError("unsupported preseeded advisory suite")
        validated_decisions.add(decision_ref)
    if validated_decisions != set(expected_suites):
        raise RuntimeError("unsupported preseeded advisory suite review")


def _activate_advisory_suite_publication(connection: sqlite3.Connection) -> None:
    """Activate once on empty protected history, then validate exact pairs."""

    if in_transaction(connection):
        raise RuntimeError("advisory suite activation requires an owned transaction")
    if (
        int(connection.execute("PRAGMA user_version").fetchone()[0])
        == _ADVISORY_SUITE_ACTIVATION_VERSION
    ):
        # An already activated store needs a consistent validation snapshot, not
        # a writer reservation that blocks an unrelated reader/writer handshake.
        connection.execute("BEGIN")
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == _ADVISORY_SUITE_ACTIVATION_VERSION:
                _validate_existing_advisory_suite_publications(connection)
                connection.commit()
                return
            # A changed preflight observation must use the locked activation
            # path below; never activate history inside the read-only branch.
            connection.rollback()
        except BaseException:
            connection.rollback()
            raise
    begin_write(connection)
    try:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version == 0:
            rows = connection.execute(
                sql_query("""SELECT schema_version, payload_json
                   FROM ontology_build_artifacts
                   WHERE schema_version IN (?, ?)""",'SELECT schema_version, payload_json\n                   FROM ontology_build_artifacts\n                   WHERE schema_version IN (%s, %s)'),
                (
                    _ADVISORY_SUITE_REVIEW_SCHEMA,
                    "ontology_paired_obligation_suite.v1",
                ),
            ).fetchall()
            for schema, payload_json in rows:
                if schema == _ADVISORY_SUITE_REVIEW_SCHEMA:
                    raise RuntimeError(
                        "unsupported pre-activation advisory suite history"
                    )
                try:
                    payload = json.loads(payload_json)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("unsupported preseeded advisory suite") from exc
                if payload.get("reviewer_ref") == _ADVISORY_SUITE_REVIEWER or str(
                    payload.get("decision_ref", "")
                ).startswith(_ADVISORY_SUITE_DECISION_PREFIX):
                    raise RuntimeError(
                        "unsupported pre-activation advisory suite history"
                    )
            connection.execute(
                f"PRAGMA user_version = {_ADVISORY_SUITE_ACTIVATION_VERSION}"
            )
        elif version == _ADVISORY_SUITE_ACTIVATION_VERSION:
            _validate_existing_advisory_suite_publications(connection)
        else:
            raise RuntimeError(
                f"unsupported ontology build store user_version {version}"
            )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


class _WorkspaceConnection(sqlite3.Connection):
    """Keep the owned callback authorizer installed on this real connection.

    This enforces the in-process callback API, not a hostile-Python sandbox.
    """

    _owned_authorizer = False
    _authorizer_replacement_attempted = False
    _writable_blob_attempted = False

    def set_authorizer(self, authorizer_callback):
        if self._owned_authorizer:
            self._authorizer_replacement_attempted = True
            raise RuntimeError("private advisory publication index guard is owned")
        return super().set_authorizer(authorizer_callback)

    def blobopen(self, table, column, rowid, /, *, readonly=False, name="main"):
        # Incremental writes bypass authorizers and total_changes. Never issue
        # writable handles, including before entry into an owned callback.
        # Store writers use SQL; native read-only blob handles remain supported.
        if not readonly:
            if self._owned_authorizer:
                self._writable_blob_attempted = True
            raise RuntimeError("workspace writable incremental blobs are forbidden")
        return super().blobopen(table, column, rowid, readonly=readonly, name=name)


ContractT = TypeVar("ContractT", bound=BaseModel)

_SEED_CAS_SCHEMAS = {
    "ontology_seed_state.v1",
    "ontology_seed_ignore.v1",
    "ontology_seed_selection.v1",
    "ontology_seed_apply.v1",
}
_RUNTIME_CONTEXT_CAS_SCHEMAS = {
    "runtime_working_context.v1",
    "runtime_context_transition.v1",
    "runtime_context_transition_receipt.v1",
}
_RUNTIME_CONTEXT_AUTHORITY_SCHEMAS = {
    "runtime_context_proposal_provider_descriptor.v1",
    "runtime_context_scope_policy.v1",
    "runtime_context_element_admission.v1",
}
_EXPLORATORY_INTEGRITY_SCHEMA = "exploratory_graph_snapshot_integrity.v1"
_VALIDATION_WRITE_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_ANALYZE,
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
        sqlite3.SQLITE_CREATE_TEMP_VIEW,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_CREATE_VTABLE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_INDEX,
        sqlite3.SQLITE_DROP_TEMP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_TRIGGER,
        sqlite3.SQLITE_DROP_TEMP_VIEW,
        sqlite3.SQLITE_DROP_TRIGGER,
        sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_DROP_VTABLE,
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_REINDEX,
        sqlite3.SQLITE_SAVEPOINT,
        sqlite3.SQLITE_TRANSACTION,
        sqlite3.SQLITE_UPDATE,
    }
)


def ontology_build_store_path(workspace_root: str | Path) -> Path:
    return (
        Path(workspace_root) / "runs" / "ontology-building" / "ontology-builds.sqlite"
    )


LEARNING_ARTIFACT_MODELS = {
    **SKILL_ARTIFACT_MODELS, **LEARNING_TRACE_ARTIFACT_MODELS, **RETAINED_CONTEXT_ARTIFACT_MODELS, **LEARNED_SKILL_ARTIFACT_MODELS,
}


_KERNEL_ARTIFACT_MODELS: dict[str, type[BaseModel]] = {
    "semantier.ontology.v1": kernel_contracts.Ontology,
    "semantier.evidence.v1": kernel_contracts.Evidence,
    "semantier.context_graph.v1": kernel_contracts.ContextGraph,
    "semantier.decision_record.v1": kernel_contracts.DecisionRecord,
    "semantier.change_candidate.v1": kernel_contracts.ChangeCandidate,
    "semantier.change_candidate.v2": ChangeCandidateV2,
}


_IDENTITY_FIELDS: dict[str, tuple[str, str]] = {
    **{schema: ("artifact_ref", "artifact_hash") for schema in LEARNING_ARTIFACT_MODELS},
    "quality_profile.v1": ("profile_ref", "profile_hash"),
    "quality_profile_applicability_review.v1": ("review_ref", "review_hash"),
    "quality_evaluation_run.v1": ("run_ref", "run_hash"),
    "semantier.ontology.v1": ("artifact_ref", "artifact_hash"),
    "semantier.evidence.v1": ("artifact_ref", "artifact_hash"),
    "semantier.context_graph.v1": ("artifact_ref", "artifact_hash"),
    "semantier.decision_record.v1": ("artifact_ref", "artifact_hash"),
    "semantier.change_candidate.v1": ("artifact_ref", "artifact_hash"),
    "semantier.change_candidate.v2": ("artifact_ref", "artifact_hash"),
    "requirement_baseline.v1": ("artifact_ref", "artifact_hash"),
    "requirement_change_qualification.v1": ("artifact_ref", "artifact_hash"),
    "requirement_change_admission.v1": ("artifact_ref", "artifact_hash"),
    "requirement_baseline_activation.v1": ("artifact_ref", "artifact_hash"),
    "requirement_context_projection.v1": ("artifact_ref", "artifact_hash"),
    "release_set_manifest.v1": ("artifact_ref", "artifact_hash"),
    "activation_binding_projection.v1": ("artifact_ref", "artifact_hash"),
    "ccc_decision_receipt.v1": ("artifact_ref", "artifact_hash"),
    _ADVISORY_INPUT_SCHEMA: ("input_ref", "input_hash"),
    _ADVISORY_SUITE_REVIEW_SCHEMA: ("decision_ref", "decision_hash"),
    "advisory_trial_build_inputs.v1": ("input_ref", "input_hash"),
    "advisory_trial_variant.v1": ("variant_ref", "variant_hash"),
    "canonical_evidence_boundary.v1": ("boundary_ref", "boundary_hash"),
    "canonical_evidence_adequacy_assessment.v1": ("assessment_ref", "assessment_hash"),
    "exploratory_graph_snapshot.v1": ("snapshot_ref", "snapshot_hash"),
    "competency_contract.v1": ("competency_contract_ref", "competency_contract_hash"),
    "conversation_decision.v1": ("decision_ref", "decision_hash"),
    "ontology_build_manifest.v1": ("build_ref", "build_hash"),
    "normative_schema_contract.v1": ("schema_ref", "schema_hash"),
    "normative_schema_mapping_review.v1": ("review_ref", "review_hash"),
    "normative_schema_proposal_evidence.v1": ("evidence_ref", "evidence_hash"),
    "normative_semantic_proposal_evidence.v1": ("evidence_ref", "evidence_hash"),
    "normative_semantic_correction_review.v1": ("review_ref", "review_hash"),
    "source_validation_proposal_evidence.v1": ("evidence_ref", "evidence_hash"),
    "source_validation_review_decision.v1": ("decision_ref", "decision_hash"),
    "source_validation_case_review.v1": ("review_ref", "review_hash"),

    "source_validation_execution.v1": ("execution_ref", "execution_hash"),
    "schema_adequacy_report.v1": ("report_ref", "report_hash"),
    "source_search_coverage_manifest.v1": ("coverage_ref", "coverage_hash"),
    "normative_semantic_candidate_set.v1": ("candidate_set_ref", "candidate_set_hash"),
    "validation_contract.v1": ("validation_contract_ref", "validation_contract_hash"),
    "candidate_graph_diff_projection.v1": ("diff_ref", "diff_hash"),
    "candidate_graph_lineage_projection.v1": ("lineage_ref", "lineage_hash"),
    "expense_teaching_policy.v1": ("policy_ref", "policy_hash"),
    "compiled_rule_plan.v1": ("plan_ref", "plan_hash"),
    "runtime_semantic_evaluation_bundle.v1": ("bundle_ref", "bundle_hash"),
    "certified_semantic_execution_plan.v1": (
        "execution_plan_ref",
        "execution_plan_hash",
    ),
    "ontology_admission_decision.v1": ("decision_ref", "decision_hash"),
    "ontology_release_bundle.v1": ("release_bundle_ref", "release_bundle_hash"),
    "runtime_execution_epoch.v1": ("epoch_ref", "epoch_hash"),
    "human_disposition.v1": ("disposition_ref", "disposition_hash"),
    "learning_evidence.v1": ("learning_evidence_ref", "learning_evidence_hash"),
    "failure_attribution.v1": ("attribution_ref", "attribution_hash"),
    "ontology_capability_profile.v1": ("profile_ref", "profile_hash"),
    "ontology_build_inputs.v1": ("envelope_ref", "envelope_hash"),
    "normative_authority_dependency_set.v1": ("set_ref", "set_hash"),
    "ontology_certification_readiness.v1": ("readiness_ref", "readiness_hash"),
    "ontology_certification_receipt.v1": ("receipt_ref", "receipt_hash"),
    "ontology_hub_snapshot.v1": ("snapshot_ref", "snapshot_hash"),
    "ontology_seed_state.v1": ("state_ref", "state_hash"),
    "ontology_seed_ignore.v1": ("ignore_ref", "ignore_hash"),
    "ontology_seed_selection.v1": ("receipt_ref", "receipt_hash"),
    "ontology_seed_apply.v1": ("receipt_ref", "receipt_hash"),
    "ontology_seed_alignment.v1": ("alignment_ref", "alignment_hash"),
    "source_grounded_seed_delta.v1": ("delta_ref", "delta_hash"),
    "ontology_seed_comparison.v1": ("comparison_ref", "comparison_hash"),
    "ontology_paired_obligation_suite.v1": ("suite_ref", "suite_hash"),
    "paired_observed_facts.v1": ("facts_ref", "facts_hash"),
    "tender_observed_facts_projection_receipt.v1": ("projection_ref", "projection_hash"),
    "paired_evaluator_bundle.v1": ("evaluator_ref", "evaluator_hash"),
    "ontology_paired_evaluation_run.v1": ("run_ref", "run_hash"),
    "ontology_paired_evaluation_report.v1": ("report_ref", "report_hash"),
    "runtime_repair_pair.v1": ("pair_ref", "pair_hash"),
    "runtime_repair_obligation_suite.v1": ("suite_ref", "suite_hash"),
    "runtime_repair_paired_evaluation_run.v1": ("run_ref", "run_hash"),
    "runtime_repair_paired_evaluation_report.v1": ("report_ref", "report_hash"),
    "runtime_repair_paired_comparison_view.v1": ("comparison_ref", "comparison_hash"),
    "runtime_review_authority_request.v1": ("authority_request_ref", "authority_request_hash"),
    "runtime_review_authority_receipt.v1": ("authority_ref", "authority_hash"),
    "runtime_review_authority_issue_result.v1": ("result_ref", "result_hash"),
    "empirical_episode_plan.v1": ("plan_ref", "plan_hash"),
    "empirical_episode.v1": ("episode_ref", "episode_hash"),
    "empirical_episode_assessment.v1": ("assessment_ref", "assessment_hash"),
    "runtime_working_context.v1": ("context_ref", "context_hash"),
    "runtime_context_transition_proposal.v1": ("proposal_ref", "proposal_hash"),
    "runtime_context_transition.v1": ("transition_ref", "transition_hash"),
    "runtime_context_transition_receipt.v1": ("receipt_ref", "receipt_hash"),
    "runtime_context_proposal_provider_descriptor.v1": (
        "descriptor_ref",
        "descriptor_hash",
    ),
    "runtime_context_scope_policy.v1": ("policy_ref", "policy_hash"),
    "runtime_context_element_admission.v1": ("admission_ref", "admission_hash"),
    _EXPLORATORY_INTEGRITY_SCHEMA: ("integrity_ref", "integrity_hash"),
}

_WORKSPACE_ARTIFACT_MODELS = {
    "tender_fact_observation_receipt.v2": (TenderFactObservationReceiptV2, "receipt_ref", "receipt_hash"),
    "tender_observed_facts_projection_receipt.v2": (TenderObservedFactsProjectionReceiptV2, "projection_ref", "projection_hash"),
    "etl_run_receipt.v1": (EtlReceipt, "receipt_ref", "receipt_hash"),
    "tender_categorical_fact_observation_receipt.v1": (
        TenderCategoricalFactObservationReceiptV1, "receipt_ref", "receipt_hash",
    ),
    "tender_fact_observation_receipt.v1": (
        TenderFactObservationReceiptV1,
        "receipt_ref",
        "receipt_hash",
    ),
    "tender_document_date_observation.v1": (
        TenderDocumentDateObservationV1,
        "observation_ref",
        "observation_hash",
    ),
    "runtime_judgment_request.v1": (
        RuntimeJudgmentRequestV1,
        "artifact_ref",
        "artifact_hash",
    ),
    "runtime_judgment_attempt.v1": (
        RuntimeJudgmentAttemptV1,
        "artifact_ref",
        "artifact_hash",
    ),
    "runtime_judgment_response.v1": (
        RuntimeJudgmentResponseV1,
        "artifact_ref",
        "artifact_hash",
    ),
    "typed_state_decision.v1": (TypedStateDecisionV1, "decision_ref", "decision_hash"),
    "project_state_projection.v1": (ProjectStateProjectionV1, "projection_ref", "projection_hash"),
    "runtime_restriction_perception_request.v1": (RuntimeRestrictionPerceptionRequestV1, "artifact_ref", "artifact_hash"),
    "runtime_restriction_perception_attempt.v1": (RuntimeRestrictionPerceptionAttemptV1, "artifact_ref", "artifact_hash"),
    "runtime_restriction_perception_response.v1": (RuntimeRestrictionPerceptionResponseV1, "artifact_ref", "artifact_hash"),
    "runtime_restriction_perception_response.v2": (RuntimeRestrictionPerceptionResponseV2, "artifact_ref", "artifact_hash"),
    "runtime_semantic_inspection_context.v1": (RuntimeSemanticInspectionContextV1, "context_ref", "context_hash"),
    "runtime_semantic_inspection_result.v1": (RuntimeSemanticInspectionResultV1, "result_ref", "result_hash"),
    "runtime_inspection_report.v1": (report_contracts.RuntimeInspectionReportV1, "report_ref", "report_hash"),
    "runtime_report_replay_inputs.v1": (report_contracts.RuntimeReportReplayInputsV1, "replay_ref", "replay_hash"),
    "runtime_report_receipt.v1": (report_contracts.RuntimeReportReceiptV1, "receipt_ref", "receipt_hash"),
    "runtime_report_export_receipt.v1": (export_contracts.RuntimeReportExportReceiptV1, "receipt_ref", "receipt_hash"),
    "workspace_artifact.v1": (artifact_contracts.WorkspaceArtifactV1, "artifact_ref", "artifact_hash"),
    "workspace_artifact_collection.v1": (artifact_contracts.WorkspaceArtifactCollectionV1, "collection_ref", "collection_hash"),
    "workspace_artifact_preparation_claim.v1": (artifact_contracts.WorkspaceArtifactPreparationClaimV1, "claim_ref", "claim_hash"),
    "workspace_artifact_preparation_adapter_result.v1": (artifact_contracts.WorkspaceArtifactPreparationAdapterResultV1, "adapter_result_ref", "adapter_result_hash"),
    "workspace_artifact_preparation.v1": (artifact_contracts.WorkspaceArtifactPreparationV1, "preparation_ref", "preparation_hash"),
    "workspace_artifact_preparation_catalog.v1": (artifact_contracts.WorkspaceArtifactPreparationCatalogV1, "catalog_ref", "catalog_hash"),
    "prepare_workspace_artifact_result.v1": (artifact_contracts.PrepareWorkspaceArtifactResultV1, "result_ref", "result_hash"),
    "workspace_artifact_selection.v1": (artifact_contracts.WorkspaceArtifactSelectionV1, "selection_ref", "selection_hash"),
    "lifecycle_artifact_binding_entry.v1": (artifact_contracts.LifecycleArtifactBindingEntryV1, "binding_ref", "binding_hash"),
    "lifecycle_artifact_binding_set.v1": (artifact_contracts.LifecycleArtifactBindingSetV1, "binding_set_ref", "binding_set_hash"),
    "workbench_projection_cursor_head.v3": (artifact_contracts.WorkbenchProjectionCursorHeadV3, "cursor_head_ref", "cursor_head_hash"),
    "workbench_evidence_focus.v1": (artifact_contracts.WorkbenchEvidenceFocusV1, "focus_ref", "focus_hash"),
    "workbench_action_acknowledgement.v6": (artifact_contracts.WorkbenchActionAcknowledgementV6, "receipt_ref", "receipt_hash"),
    "artifact_workspace_projection.v1": (artifact_contracts.ArtifactWorkspaceProjectionV1, "projection_ref", "projection_hash"),
    "workspace_artifact_operation_receipt.v1": (artifact_contracts.WorkspaceArtifactOperationReceiptV1, "receipt_ref", "receipt_hash"),
}
_IDENTITY_FIELDS.update({schema: (ref, hash_field) for schema, (_, ref, hash_field) in _WORKSPACE_ARTIFACT_MODELS.items()})


_TASK909_MODELS: dict[str, tuple[type[BaseModel], ...]] = {
    "runtime_temporal_source_archive.v1": (RuntimeTemporalSourceArchiveV1,),
    "runtime_temporal_normalization.v1": (RuntimeTemporalNormalizationV1,),
    "runtime_temporal_source_graph.v1": (RuntimeTemporalSourceGraphV1,),
    "runtime_temporal_context_selection.v1": (RuntimeTemporalContextSelectionV1,),
    "runtime_working_context.v3": (RuntimeTaskContextV3,),
    "runtime_context_transition_receipt.v3": (RuntimeContextTransitionReceiptV3,),
    "runtime_temporal_query.v1": (TemporalQueryV1,),
    "runtime_provider_descriptor.v1": (RuntimeTemporalProviderDescriptorV1,),
    "runtime_provider_profile.v1": (RuntimeTemporalProviderProfileV1,),
    "runtime_provider_profile.v2": (RuntimeTemporalProviderProfileV2,),
    "runtime_provider_request.v1": (RuntimeTemporalProviderRequestV1,),
    "runtime_execution_admission.v1": (RuntimeExecutionAdmissionV1,),
    "runtime_evaluation_run.v1": (RuntimeEvaluationRunV1,),
    "runtime_evaluation_diagnostic.v1": (RuntimeEvaluationDiagnosticV1,),
    "runtime_temporal_capture.v1": (RuntimeTemporalCaptureV1,),
    "runtime_temporal_world_snapshot.v1": (RuntimeTemporalWorldSnapshotV1,),
    "runtime_provider_result.v1": (
        RuntimeTemporalSupportedResultV1,
        RuntimeTemporalUnavailableResultV1,
    ),
    "runtime_evaluation_state.v1": (RuntimeEvaluationStateV1,),
    "runtime_evaluation_request_outcome.v1": (RuntimeEvaluationRequestOutcomeV1,),
}
_TASK909_IDENTITY_FIELDS = {
    "runtime_temporal_source_archive.v1": ("archive_ref", "archive_hash"),
    "runtime_temporal_normalization.v1": ("normalization_ref", "normalization_hash"),
    "runtime_temporal_source_graph.v1": ("graph_ref", "graph_hash"),
    "runtime_temporal_context_selection.v1": ("selection_ref", "selection_hash"),
    "runtime_working_context.v3": ("context_ref", "context_hash"),
    "runtime_context_transition_receipt.v3": ("receipt_ref", "receipt_hash"),
    "runtime_temporal_query.v1": ("query_ref", "query_hash"),
    "runtime_provider_descriptor.v1": ("descriptor_ref", "descriptor_hash"),
    "runtime_provider_profile.v1": ("profile_ref", "profile_hash"),
    "runtime_provider_profile.v2": ("profile_ref", "profile_hash"),
    "runtime_provider_request.v1": ("request_ref", "request_hash"),
    "runtime_execution_admission.v1": ("admission_ref", "admission_hash"),
    "runtime_evaluation_run.v1": ("run_ref", "run_hash"),
    "runtime_evaluation_diagnostic.v1": ("diagnostic_ref", "diagnostic_hash"),
    "runtime_temporal_capture.v1": ("capture_ref", "capture_hash"),
    "runtime_temporal_world_snapshot.v1": ("snapshot_ref", "snapshot_hash"),
    "runtime_provider_result.v1": ("result_ref", "result_hash"),
    "runtime_evaluation_state.v1": ("state_ref", "state_hash"),
    "runtime_evaluation_request_outcome.v1": ("outcome_ref", "outcome_hash"),
}
for _producer_model in PRODUCER_RECORD_MODELS:
    _producer_schema = _producer_model.model_fields["schema_version"].annotation.__args__[0]
    _TASK909_MODELS[_producer_schema] = (_producer_model,)
    _TASK909_IDENTITY_FIELDS[_producer_schema] = (_producer_model.REF_FIELD, _producer_model.HASH_FIELD)
from contracts.hr_workbench import HrJobModelV1, HrScreeningRunV1, HrSourceBindingV1, HrJobModelV2, HrScreeningRunV2, HrInterpretationV1
for _hr_model in (HrJobModelV1, HrScreeningRunV1, HrSourceBindingV1, HrJobModelV2, HrScreeningRunV2, HrInterpretationV1):
    _hr_schema = _hr_model.model_fields["schema_version"].default
    _TASK909_MODELS[_hr_schema] = (_hr_model,)
    _IDENTITY_FIELDS[_hr_schema] = ("artifact_ref", "artifact_hash")
from contracts.model_gateway import ModelReexecutionRecordV1
_TASK909_MODELS['model_reexecution_record.v1'] = (ModelReexecutionRecordV1,)
_IDENTITY_FIELDS['model_reexecution_record.v1'] = ('artifact_ref', 'artifact_hash')
from contracts.requirements_reconciliation import RequirementsReviewV1
_TASK909_MODELS['requirements_review.v1'] = (RequirementsReviewV1,)
_IDENTITY_FIELDS['requirements_review.v1'] = ('review_ref', 'review_hash')
from contracts.policy_revision import PolicyRevisionRecordV1
_TASK909_MODELS['policy_revision_record.v1'] = (PolicyRevisionRecordV1,)
_IDENTITY_FIELDS['policy_revision_record.v1'] = ('artifact_ref', 'artifact_hash')
_TASK909_MODELS["jd_review.v1"] = (JdReviewV1,)
_TASK909_MODELS["jd_feedback.v1"] = (JdFeedbackV1,)
_IDENTITY_FIELDS["jd_feedback.v1"] = ("feedback_ref", "feedback_hash")
_TASK909_MODELS["jd_basis_evidence.v1"] = (JdBasisEvidenceV1,)
_IDENTITY_FIELDS["jd_basis_evidence.v1"] = ("basis_ref", "basis_hash")
_IDENTITY_FIELDS["jd_review.v1"] = ("review_id", "result_hash")
_IDENTITY_FIELDS.update(_TASK909_IDENTITY_FIELDS)


_FORMATION_EVIDENCE_SCHEMAS = {
    "reconstruction_query.v1",
    "candidate_capture.v1",
    "memory_candidate.v1",
    "memory_candidate_set.v1",
    "reconstruction_governance_capture.v1",
    "reconstructed_context.v1",
    "admitted_reconstructed_context.v1",
    "frozen_context_formation.v1",
}
_LEGACY_CONTEXT_SCHEMAS = _RUNTIME_CONTEXT_CAS_SCHEMAS | {
    "runtime_context_transition_proposal.v1",
}
_REQUEST_ASSOCIATION_SCHEMAS = {
    "runtime_context_formation_command.v1",
    "runtime_context_persistence_manifest.v1",
}
_V2_CONTEXT_SCHEMAS = {
    schema.removesuffix(".v1") + ".v2" for schema in _LEGACY_CONTEXT_SCHEMAS
}
_ADVISORY_CONTEXT_SCHEMAS = {
    "advisory_context_command.v1",
    "advisory_context_formation.v1",
    "advisory_context_receipt.v1",
}
_ADVISORY_PROPOSAL_MODELS = {
    "advisory_next_proposal_command.v1": AdvisoryNextProposalCommandV1,
    "advisory_repair_proposal.v1": AdvisoryRepairProposalV1,
    "advisory_next_proposal_receipt.v1": AdvisoryNextProposalReceiptV1,
}
for _schema, _model in _ADVISORY_PROPOSAL_MODELS.items():
    _IDENTITY_FIELDS[_schema] = (_model.identity_ref_field, _model.identity_hash_field)
_WORKSPACE_REGISTRY_SCHEMAS = (
    _FORMATION_EVIDENCE_SCHEMAS
    | _LEGACY_CONTEXT_SCHEMAS
    | _V2_CONTEXT_SCHEMAS
    | _REQUEST_ASSOCIATION_SCHEMAS
    | _ADVISORY_CONTEXT_SCHEMAS
    | {"tender_review_record.v1", "tender_review_report.v1", "tender_review_report.v2", "tender_legal_compliance_basis.v1", "tender_review_docx_derivative.v1"}
)
_RUNTIME_CONTEXT_CAS_SCHEMAS |= _V2_CONTEXT_SCHEMAS - {
    "runtime_context_transition_proposal.v2"
}
for _schema in _WORKSPACE_REGISTRY_SCHEMAS:
    _registration = ARTIFACT_SCHEMA_REGISTRY[_schema]
    _IDENTITY_FIELDS[_schema] = (_registration.ref_field, _registration.hash_field)
_RUNTIME_CONTEXT_CAS_SCHEMAS |= {
    "runtime_working_context.v3",
    "runtime_context_transition_receipt.v3",
}


def _validate_registered_artifact(artifact: BaseModel) -> BaseModel:
    """Revalidate the literal schema with its closed registered model and identity."""
    schema = getattr(artifact, "schema_version", "")
    registration = ARTIFACT_SCHEMA_REGISTRY.get(schema)
    if schema not in _WORKSPACE_REGISTRY_SCHEMAS or registration is None:
        raise ValueError(f"unsupported ontology artifact {schema!r}")
    if type(artifact) is not registration.model:
        raise ValueError("INVALID_STATE")
    payload = artifact.model_dump(mode="json")
    try:
        validated = registration.model.model_validate(payload)
    except ValidationError as exc:
        # Identity validators run after structural validation. Do not expose
        # Pydantic errors as the store's public version/publication contract.
        identity_failure = any(
            "identity" in error["msg"] or "hash mismatch" in error["msg"]
            for error in exc.errors()
        )
        raise ValueError(
            "HASH_MISMATCH" if identity_failure else "INVALID_STATE"
        ) from exc
    if validated.model_dump(mode="json") != payload:
        raise ValueError("HASH_MISMATCH")
    return validated


_IDENTITY_FIELDS.update(
    {schema: ("artifact_ref", "artifact_hash") for schema in RECEIVABLES_MODELS}
)


def _reject_context_publication(prepared) -> None:
    for item in prepared:
        schema = item[4]
        if schema in _ADVISORY_PROPOSAL_MODELS:
            raise ValueError(
                "advisory proposal artifacts require dedicated repair publication"
            )
        if schema in _LEGACY_CONTEXT_SCHEMAS:
            raise ValueError("LEGACY_WRITE_FORBIDDEN")
        if schema in _REQUEST_ASSOCIATION_SCHEMAS:
            raise ValueError(
                "runtime context request association artifacts require append_context CAS"
            )
        if schema == "runtime_context_transition_proposal.v2":
            raise ValueError("runtime context proposals require put_proposal")
        if schema in _V2_CONTEXT_SCHEMAS:
            raise ValueError(
                "committed runtime context artifacts require append_context CAS"
            )


def _reject_trial_projection_publication(prepared) -> None:
    """Reserve controlled-trial records for their future owning private port."""

    for item in prepared:
        schema = item[4]
        if schema == _ADVISORY_INPUT_SCHEMA:
            raise ValueError(
                "advisory compiler inputs require a dedicated publication port"
            )
        if schema == "advisory_trial_variant.v1":
            raise ValueError(
                "advisory trial derivations require a dedicated publication port"
            )
        payload = json.loads(item[5])
        if schema == "advisory_trial_build_inputs.v1":
            raise ValueError(
                "controlled trial projections require a dedicated trial-pair port"
            )
        if schema == "exploratory_graph_snapshot.v1" and (
            payload.get("change_cause") == "controlled_trial_projection"
        ):
            raise ValueError(
                "controlled trial projections require a dedicated trial-pair port"
            )
        if schema == _EXPLORATORY_INTEGRITY_SCHEMA and (
            payload.get("producer_profile") == "advisory_trial_projection.v1"
        ):
            raise ValueError(
                "controlled trial projections require a dedicated trial-pair port"
            )
        if schema == "ontology_build_manifest.v1" and (
            str(payload.get("build_ref", "")).startswith(
                "ontology-build:advisory-trial:"
            )
            or any(
                child.get("artifact_kind") == "advisory_trial_build_inputs"
                for child in payload.get("child_artifacts", ())
            )
        ):
            raise ValueError(
                "controlled trial projections require a dedicated trial-pair port"
            )


def _reject_advisory_suite_publication(prepared) -> None:
    """Reserve the fixed controlled-suite pair for its private owner port."""

    for item in prepared:
        schema = item[4]
        if schema == _ADVISORY_SUITE_REVIEW_SCHEMA:
            raise ValueError(
                "advisory suite reviews require a dedicated suite-review port"
            )
        if schema != "ontology_paired_obligation_suite.v1":
            continue
        payload = json.loads(item[5])
        if payload.get("reviewer_ref") == _ADVISORY_SUITE_REVIEWER or str(
            payload.get("decision_ref", "")
        ).startswith(_ADVISORY_SUITE_DECISION_PREFIX):
            raise ValueError(
                "controlled advisory suites require a dedicated suite-review port"
            )


from contracts.tender_owner_adoption import TenderOwnerAdoptionProposalV1, TenderOwnerAdoptionV1, TenderOwnerRevocationV1
for _owner_model in (TenderOwnerAdoptionProposalV1, TenderOwnerAdoptionV1, TenderOwnerRevocationV1):
    _owner_schema = _owner_model.model_fields['schema_version'].default
    _TASK909_MODELS[_owner_schema] = (_owner_model,)
    _IDENTITY_FIELDS[_owner_schema] = ('artifact_ref', 'artifact_hash')


def _reject_protected_publication(prepared) -> None:
    """Require protected artifact families to use their governed store ports."""

    if any(item[4] in _ADVISORY_CONTEXT_SCHEMAS for item in prepared):
        raise ValueError(
            "advisory context artifacts require dedicated formation publication"
        )
    for item in prepared:
        payload = json.loads(item[5])
        if item[4] in {'tender_owner_adoption.v1', 'tender_owner_revocation.v1'} or (
            item[4] == 'semantier.decision_record.v1' and payload.get('reasoner_identity') == 'policy:tender-owner-internal-advisory:v1'):
            raise ValueError('OWNER_ADOPTION_DEDICATED_PRODUCER_REQUIRED')
    _reject_context_publication(prepared)
    _reject_trial_projection_publication(prepared)
    _reject_advisory_suite_publication(prepared)
    if any(item[4] in _SEED_CAS_SCHEMAS for item in prepared):
        raise ValueError("ontology seed state/actions require append_seed_state CAS")
    if any(item[4] in _RUNTIME_CONTEXT_CAS_SCHEMAS for item in prepared):
        raise ValueError(
            "committed runtime context artifacts require append_context CAS"
        )
    if any(item[4] in _RUNTIME_CONTEXT_AUTHORITY_SCHEMAS for item in prepared):
        raise ValueError("runtime context authority artifacts require authority facade")
    if any(item[4] == _EXPLORATORY_INTEGRITY_SCHEMA for item in prepared):
        raise ValueError(
            "exploratory snapshot integrity requires prospective integrity transaction"
        )


class OntologyBuildStore:
    def reserve_learning_command(self, *, organization_id, workspace_id, actor_ref, request_id, request_hash):
        if enabled():require_tables(self._connection,'learning_review_command_requests')
        else:
            self._connection.execute("""CREATE TABLE IF NOT EXISTS learning_review_command_requests (
                        organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL, actor_ref TEXT NOT NULL,
                        request_id TEXT NOT NULL, request_hash TEXT NOT NULL,
                        PRIMARY KEY(organization_id,workspace_id,actor_ref,request_id))""")
        with self._advisory_publication_transaction():
            row = self._connection.execute(sql_query('SELECT request_hash FROM learning_review_command_requests '
                'WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND request_id=?','SELECT request_hash FROM learning_review_command_requests WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND request_id=%s'),
                (organization_id,workspace_id,actor_ref,request_id)).fetchone()
            if row and row[0] != request_hash:
                raise ValueError('REQUEST_CONFLICT')
            if not row:
                self._connection.execute(sql_query('INSERT INTO learning_review_command_requests VALUES (?,?,?,?,?)','INSERT INTO learning_review_command_requests VALUES (%s,%s,%s,%s,%s)'),
                    (organization_id,workspace_id,actor_ref,request_id,request_hash))

    def learning_review_state(self, *, organization_id, workspace_id, actor_ref, notice_id):
        if enabled():require_tables(self._connection,'learning_review_presentation')
        else:
            self._connection.execute("""CREATE TABLE IF NOT EXISTS learning_review_presentation (
                        organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL, actor_ref TEXT NOT NULL,
                        notice_id TEXT NOT NULL, is_read INTEGER NOT NULL DEFAULT 0, deferred INTEGER NOT NULL DEFAULT 0,
                        updated_at TEXT NOT NULL, PRIMARY KEY(organization_id,workspace_id,actor_ref,notice_id))""")
        row = self._connection.execute(sql_query('SELECT is_read,deferred FROM learning_review_presentation '
            'WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND notice_id=?','SELECT is_read,deferred FROM learning_review_presentation WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND notice_id=%s'),
            (organization_id or '', workspace_id, actor_ref, notice_id)).fetchone()
        return {'read': bool(row[0]), 'deferred': bool(row[1])} if row else {'read': False, 'deferred': False}

    def set_learning_review_state(self, *, organization_id, workspace_id, actor_ref, notice_id,
                                  operation, request_id, request_hash):
        self.learning_review_state(organization_id=organization_id, workspace_id=workspace_id,
            actor_ref=actor_ref, notice_id=notice_id)
        if enabled():require_tables(self._connection,'learning_review_presentation_requests')
        else:
            self._connection.execute("""CREATE TABLE IF NOT EXISTS learning_review_presentation_requests (
                        workspace_id TEXT NOT NULL, actor_ref TEXT NOT NULL, request_id TEXT NOT NULL,
                        request_hash TEXT NOT NULL, PRIMARY KEY(workspace_id,actor_ref,request_id))""")
        with self._advisory_publication_transaction():
            old = self._connection.execute(sql_query('SELECT request_hash FROM learning_review_presentation_requests '
                'WHERE workspace_id=? AND actor_ref=? AND request_id=?','SELECT request_hash FROM learning_review_presentation_requests WHERE workspace_id=%s AND actor_ref=%s AND request_id=%s'), (workspace_id,actor_ref,request_id)).fetchone()
            if old and old[0] != request_hash:
                raise ValueError('REQUEST_CONFLICT')
            if not old:
                self._connection.execute(sql_query('INSERT INTO learning_review_presentation_requests VALUES (?,?,?,?)','INSERT INTO learning_review_presentation_requests VALUES (%s,%s,%s,%s)'),
                    (workspace_id,actor_ref,request_id,request_hash))
                self._connection.execute(sql_query('INSERT INTO learning_review_presentation VALUES (?,?,?,?,?,?,?) '
                    'ON CONFLICT(organization_id,workspace_id,actor_ref,notice_id) DO UPDATE SET '
                    'is_read=MAX(is_read,excluded.is_read),deferred=MAX(deferred,excluded.deferred),updated_at=excluded.updated_at','INSERT INTO learning_review_presentation VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(organization_id,workspace_id,actor_ref,notice_id) DO UPDATE SET is_read=GREATEST(learning_review_presentation.is_read,excluded.is_read),deferred=GREATEST(learning_review_presentation.deferred,excluded.deferred),updated_at=excluded.updated_at'),
                    (organization_id or '',workspace_id,actor_ref,notice_id,int(operation=='mark_notice_read'),
                     int(operation=='defer_candidate'),datetime.now(timezone.utc).isoformat()))
        return self.learning_review_state(organization_id=organization_id,workspace_id=workspace_id,
            actor_ref=actor_ref,notice_id=notice_id)

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        if enabled():
            from storage.ontology_postgres_connection import connect_ontology,ontology_owned_tables
            self._connection=connect_ontology(self._path)
            try:require_tables(self._connection,*ontology_owned_tables())
            except BaseException:
                self._connection.close()
                raise
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._path, factory=_WorkspaceConnection)
        self._connection.execute("""CREATE TABLE IF NOT EXISTS hr_interpretation_heads (
            organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL, actor_id TEXT NOT NULL, session_id TEXT NOT NULL,
            model_ref TEXT NOT NULL, resume_ref TEXT NOT NULL, requirement_id TEXT NOT NULL,
            artifact_ref TEXT NOT NULL, artifact_hash TEXT NOT NULL,
            PRIMARY KEY (organization_id,workspace_id,actor_id,session_id,model_ref,resume_ref,requirement_id))""")
        self._connection.execute("""CREATE TABLE IF NOT EXISTS hr_screening_operations (
            organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL, actor_id TEXT NOT NULL, session_id TEXT NOT NULL,
            request_id TEXT NOT NULL, request_hash TEXT NOT NULL, command_json TEXT NOT NULL, status TEXT NOT NULL,
            owner_pid INTEGER NOT NULL, attempt_token TEXT NOT NULL, checkpoint_json TEXT NOT NULL, checkpoint_hash TEXT NOT NULL,
            result_ref TEXT, result_hash TEXT, stop_disposition TEXT, started_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            PRIMARY KEY(organization_id,workspace_id,actor_id,session_id,request_id))""")
        user_version = int(
            self._connection.execute("PRAGMA user_version").fetchone()[0]
        )
        if user_version not in (0, _ADVISORY_SUITE_ACTIVATION_VERSION):
            self._connection.close()
            raise RuntimeError(
                f"unsupported ontology build store user_version {user_version}"
            )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ontology_build_artifacts (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                artifact_ref TEXT NOT NULL,
                artifact_hash TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY (organization_id, workspace_id, artifact_ref)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ontology_seed_state_index (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                build_ref TEXT NOT NULL,
                revision INTEGER NOT NULL,
                state_ref TEXT NOT NULL UNIQUE,
                state_hash TEXT NOT NULL,
                predecessor_state_ref TEXT,
                predecessor_state_hash TEXT,
                PRIMARY KEY (organization_id, workspace_id, build_ref, revision)
            )
            """
        )
        self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ontology_build_artifact_ref_unique ON ontology_build_artifacts(artifact_ref)"
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ontology_artifact_integrity_index (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                subject_ref TEXT NOT NULL,
                subject_hash TEXT NOT NULL,
                integrity_ref TEXT NOT NULL,
                integrity_hash TEXT NOT NULL,
                producer_profile TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                PRIMARY KEY (organization_id, workspace_id, subject_ref),
                UNIQUE (integrity_ref)
            )
            """
        )
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS ontology_seed_request_index (
                organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                actor_ref TEXT NOT NULL, operation TEXT NOT NULL, request_id TEXT NOT NULL,
                request_hash TEXT NOT NULL, receipt_ref TEXT NOT NULL, receipt_hash TEXT NOT NULL,
                state_ref TEXT NOT NULL, state_hash TEXT NOT NULL,
                PRIMARY KEY (organization_id, workspace_id, actor_ref, operation, request_id)
            )
        """)
        self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ontology_seed_request_receipt_unique ON ontology_seed_request_index(receipt_ref)"
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_context_heads (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                stream_ref TEXT NOT NULL,
                version INTEGER NOT NULL,
                context_ref TEXT NOT NULL,
                context_hash TEXT NOT NULL,
                PRIMARY KEY (organization_id, workspace_id, stream_ref)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_context_version_index (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                stream_ref TEXT NOT NULL,
                version INTEGER NOT NULL,
                context_ref TEXT NOT NULL UNIQUE,
                context_hash TEXT NOT NULL,
                PRIMARY KEY (organization_id, workspace_id, stream_ref, version)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_context_request_index (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                request_id TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                receipt_ref TEXT NOT NULL,
                receipt_hash TEXT NOT NULL,
                PRIMARY KEY (
                    organization_id, workspace_id, actor_id, operation, request_id
                )
            )
            """
        )
        self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS runtime_context_request_receipt_unique "
            "ON runtime_context_request_index(receipt_ref)"
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workspace_operation_request_index (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                actor_ref TEXT NOT NULL,
                operation TEXT NOT NULL,
                request_id TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                receipt_ref TEXT NOT NULL,
                receipt_hash TEXT NOT NULL,
                PRIMARY KEY (
                    organization_id, workspace_id, actor_ref, operation, request_id
                )
            )
            """
        )
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS workspace_artifacts (
                organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                content_hash TEXT NOT NULL, artifact_ref TEXT NOT NULL UNIQUE,
                artifact_hash TEXT NOT NULL, recorded_at TEXT NOT NULL,
                PRIMARY KEY (organization_id, workspace_id, content_hash)
            )
        """)
        for table in (
            "workspace_artifact_collection_heads",
            "workspace_artifact_preparation_catalog_heads",
            "workspace_artifact_selection_heads",
        ):
            self._connection.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision >= 0),
                    head_ref TEXT NOT NULL, head_hash TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (organization_id, workspace_id)
                )
            """)
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS workspace_artifact_preparation_heads (
                organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                artifact_ref TEXT NOT NULL,
                purpose TEXT NOT NULL CHECK(purpose IN ('DESIGN_SOURCE', 'RUNTIME_TARGET')),
                revision INTEGER NOT NULL CHECK(revision >= 0),
                head_ref TEXT NOT NULL, head_hash TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY (organization_id, workspace_id, artifact_ref, purpose)
            )
        """)
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS workspace_artifact_binding_set_heads (
                organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                lifecycle TEXT NOT NULL, role TEXT NOT NULL,
                revision INTEGER NOT NULL CHECK(revision >= 0),
                head_ref TEXT NOT NULL, head_hash TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY (organization_id, workspace_id, lifecycle, role),
                CHECK((lifecycle='DESIGN' AND role='SOURCE') OR
                      (lifecycle='RUNTIME' AND role='TARGET'))
            )
        """)
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS workbench_projection_cursors (
                organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                session_id TEXT NOT NULL, actor_ref TEXT NOT NULL, cursor_ref TEXT NOT NULL,
                revision INTEGER NOT NULL CHECK(revision >= 0),
                cursor_head_ref TEXT NOT NULL UNIQUE, cursor_head_hash TEXT NOT NULL,
                projection_basis_hash TEXT NOT NULL,
                selection_ref TEXT NOT NULL, selection_hash TEXT NOT NULL,
                design_binding_set_ref TEXT NOT NULL, design_binding_set_hash TEXT NOT NULL,
                runtime_binding_set_ref TEXT NOT NULL, runtime_binding_set_hash TEXT NOT NULL,
                design_snapshot_ref TEXT, design_snapshot_hash TEXT,
                PRIMARY KEY(organization_id,workspace_id,session_id,actor_ref,cursor_ref),
                CHECK((design_snapshot_ref IS NULL) = (design_snapshot_hash IS NULL))
            )
        """)
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS workspace_artifact_preparation_operation_index (
                organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                correlation_ref TEXT NOT NULL, operation_key TEXT NOT NULL,
                request_id TEXT NOT NULL, request_hash TEXT NOT NULL,
                claim_ref TEXT NOT NULL, claim_hash TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('PENDING', 'COMPLETE')),
                adapter_result_ref TEXT, adapter_result_hash TEXT,
                preparation_ref TEXT, preparation_hash TEXT,
                receipt_ref TEXT, receipt_hash TEXT,
                PRIMARY KEY(organization_id,workspace_id,correlation_ref),
                UNIQUE(organization_id,workspace_id,operation_key),
                CHECK((adapter_result_ref IS NULL) = (adapter_result_hash IS NULL)),
                CHECK((preparation_ref IS NULL) = (preparation_hash IS NULL)),
                CHECK((receipt_ref IS NULL) = (receipt_hash IS NULL)),
                CHECK(state != 'COMPLETE' OR
                      (adapter_result_ref IS NOT NULL AND preparation_ref IS NOT NULL
                       AND receipt_ref IS NOT NULL))
            )
        """)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_evaluation_state_index (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                run_ref TEXT NOT NULL,
                revision INTEGER NOT NULL CHECK(revision>=0),
                state_ref TEXT NOT NULL,
                state_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                outcome_ref TEXT NOT NULL,
                outcome_hash TEXT NOT NULL,
                PRIMARY KEY(organization_id, workspace_id, run_ref, revision),
                UNIQUE(organization_id, workspace_id, state_ref)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_evaluation_heads (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                run_ref TEXT NOT NULL,
                revision INTEGER NOT NULL CHECK(revision>=0),
                state_ref TEXT NOT NULL,
                state_hash TEXT NOT NULL,
                outcome_ref TEXT NOT NULL,
                outcome_hash TEXT NOT NULL,
                PRIMARY KEY(organization_id, workspace_id, run_ref)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_context_provider_descriptor_index (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                descriptor_ref TEXT NOT NULL,
                descriptor_hash TEXT NOT NULL,
                implementation_id TEXT NOT NULL,
                implementation_version TEXT NOT NULL,
                implementation_hash TEXT NOT NULL,
                configuration_hash TEXT NOT NULL,
                PRIMARY KEY (organization_id, workspace_id, descriptor_ref),
                UNIQUE (descriptor_ref)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_context_authority_request_index (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                request_id TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                artifact_kind TEXT NOT NULL,
                artifact_ref TEXT NOT NULL,
                artifact_hash TEXT NOT NULL,
                PRIMARY KEY (
                    organization_id, workspace_id, actor_id, operation, request_id
                ),
                UNIQUE (artifact_ref)
            )
            """
        )
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS tender_review_heads (
                organization_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                session_ref TEXT NOT NULL, target_ref TEXT NOT NULL, target_hash TEXT NOT NULL,
                subject_hash TEXT NOT NULL, actor_ref TEXT NOT NULL,
                record_ref TEXT NOT NULL, record_hash TEXT NOT NULL,
                PRIMARY KEY (organization_id,workspace_id,session_ref,target_ref,target_hash,subject_hash,actor_ref)
            )
        """)
        _ensure_advisory_publication_schema(self._connection)
        self._connection.commit()
        try:
            _activate_advisory_suite_publication(self._connection)
        except BaseException:
            self._connection.close()
            raise

    @contextmanager
    def _advisory_publication_transaction(self):
        """Serialize fixed private ports without generic callback authorization."""
        if in_transaction(self._connection):
            raise RuntimeError(
                "advisory publication cannot nest inside an active transaction"
            )
        begin_write(self._connection)
        try:
            yield
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    @staticmethod
    def _require_final_advisory_owner_access(
        context: AdvisoryTrialOwnerContext,
    ) -> None:
        """Recheck actual actor/session after the final cross-owner capture."""
        context.authority_resolver.require_current_actor(
            actor_context=context.actor_context
        )
        if (
            context.session_resolver.resolve(
                actor_context=context.actor_context, session_ref=context.session_ref
            )
            != context.session_ref
        ):
            raise ValueError("ADVISORY_COMPILER_INPUT_SCOPE_MISMATCH")

    def _publish_advisory_compiler_inputs(
        self,
        *,
        context: AdvisoryTrialOwnerContext,
        inputs: AdvisoryCompilerInputsV1,
    ) -> AdvisoryCompilerInputsV1:
        # Fixed behavioral import after store initialization; annotation imports
        # remain TYPE_CHECKING-only and no caller can select the validator.
        from services.runtime_learning_trial_service import (
            _require_advisory_trial_owner_context,
            _resolve_current_compiler_input_values,
        )

        _require_advisory_trial_owner_context(context)
        if getattr(context, "build_store", None) is not self:
            raise ValueError("ADVISORY_TRIAL_STORE_MISMATCH")
        if in_transaction(self._connection):
            raise RuntimeError(
                "advisory publication cannot nest inside an active transaction"
            )
        _resolve_current_compiler_input_values(context=context, inputs=inputs)
        inputs = AdvisoryCompilerInputsV1.model_validate(inputs.model_dump(mode="json"))
        prepared = self._prepare(inputs)
        pin = ArtifactPinV1(
            artifact_kind="advisory_compiler_inputs",
            artifact_ref=inputs.input_ref,
            artifact_hash=inputs.input_hash,
        )
        with self._advisory_publication_transaction():
            _resolve_current_compiler_input_values(context=context, inputs=inputs)
            artifact_exists = self._connection.execute(
                sql_query("SELECT 1 FROM ontology_build_artifacts WHERE artifact_ref = ?",'SELECT 1 FROM ontology_build_artifacts WHERE artifact_ref = %s'),
                (inputs.input_ref,),
            ).fetchone()
            index_exists = self._connection.execute(
                sql_query("SELECT 1 FROM advisory_trial_native_publication_index WHERE artifact_ref = ?",'SELECT 1 FROM advisory_trial_native_publication_index WHERE artifact_ref = %s'),
                (inputs.input_ref,),
            ).fetchone()
            if artifact_exists is not None or index_exists is not None:
                saved = self.get_published_advisory_compiler_inputs(
                    organization_id=inputs.organization_id,
                    workspace_id=inputs.workspace_id,
                    pin=pin,
                )
                if canonical_json(saved.model_dump(mode="json")) != prepared[5]:
                    raise ValueError("ADVISORY_COMPILER_INPUT_PAYLOAD_MISMATCH")
            else:
                self._insert_prepared((prepared,))
                self._connection.execute(
                    sql_query("""INSERT INTO advisory_trial_native_publication_index (
                        organization_id, workspace_id, publication_kind,
                        artifact_ref, artifact_hash, payload_hash,
                        input_ref, input_hash, derivation_ref, derivation_hash,
                        subject_ref, subject_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",'INSERT INTO advisory_trial_native_publication_index (\n                        organization_id, workspace_id, publication_kind,\n                        artifact_ref, artifact_hash, payload_hash,\n                        input_ref, input_hash, derivation_ref, derivation_hash,\n                        subject_ref, subject_hash\n                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'),
                    self._advisory_compiler_input_index_values(inputs),
                )
                saved = self.get_published_advisory_compiler_inputs(
                    organization_id=inputs.organization_id,
                    workspace_id=inputs.workspace_id,
                    pin=pin,
                )
            _resolve_current_compiler_input_values(context=context, inputs=inputs)
            self._require_final_advisory_owner_access(context)
        return saved

    @staticmethod
    def _advisory_compiler_input_index_values(
        inputs: AdvisoryCompilerInputsV1,
    ) -> tuple:
        return (
            inputs.organization_id,
            inputs.workspace_id,
            "compiler_inputs",
            inputs.input_ref,
            inputs.input_hash,
            "sha256:" + content_hash(inputs.model_dump(mode="json")),
            inputs.input_ref,
            inputs.input_hash,
            None,
            None,
            inputs.input_ref,
            inputs.input_hash,
        )

    def get_published_advisory_compiler_inputs(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        pin: ArtifactPinV1,
    ) -> AdvisoryCompilerInputsV1:
        """Read exact durable publication proof; this is not an access grant."""
        if (
            not isinstance(pin, ArtifactPinV1)
            or pin.artifact_kind != "advisory_compiler_inputs"
        ):
            raise ValueError("ADVISORY_COMPILER_INPUT_PIN_MISMATCH")
        row = self._connection.execute(
            sql_query("""SELECT organization_id, workspace_id, artifact_hash, schema_version, payload_json
            FROM ontology_build_artifacts WHERE artifact_ref = ?""",'SELECT organization_id, workspace_id, artifact_hash, schema_version, payload_json\n            FROM ontology_build_artifacts WHERE artifact_ref = %s'),
            (pin.artifact_ref,),
        ).fetchone()
        if row is None or row[:4] != (
            organization_id,
            workspace_id,
            pin.artifact_hash,
            _ADVISORY_INPUT_SCHEMA,
        ):
            raise ValueError("ADVISORY_COMPILER_INPUT_PUBLICATION_MISMATCH")
        try:
            inputs = AdvisoryCompilerInputsV1.model_validate_json(row[4])
        except ValidationError as exc:
            raise ValueError("ADVISORY_COMPILER_INPUT_PAYLOAD_MISMATCH") from exc
        if (
            inputs.organization_id,
            inputs.workspace_id,
            inputs.input_ref,
            inputs.input_hash,
        ) != (
            organization_id,
            workspace_id,
            pin.artifact_ref,
            pin.artifact_hash,
        ) or canonical_json(inputs.model_dump(mode="json")) != row[4]:
            raise ValueError("ADVISORY_COMPILER_INPUT_PAYLOAD_MISMATCH")
        rows = self._connection.execute(
            sql_query("""SELECT organization_id, workspace_id, publication_kind,
                artifact_ref, artifact_hash, payload_hash, input_ref, input_hash,
                derivation_ref, derivation_hash, subject_ref, subject_hash
            FROM advisory_trial_native_publication_index WHERE artifact_ref = ?""",'SELECT organization_id, workspace_id, publication_kind,\n                artifact_ref, artifact_hash, payload_hash, input_ref, input_hash,\n                derivation_ref, derivation_hash, subject_ref, subject_hash\n            FROM advisory_trial_native_publication_index WHERE artifact_ref = %s'),
            (pin.artifact_ref,),
        ).fetchall()
        if rows != [self._advisory_compiler_input_index_values(inputs)]:
            raise ValueError("ADVISORY_COMPILER_INPUT_INDEX_MISMATCH")
        return inputs

    @contextmanager
    def iter_source_first_build_inputs(
        self, *, organization_id: str, workspace_id: str
    ) -> AbstractContextManager[Iterator[OntologyBuildInputEnvelopeV1]]:
        """Stream only exact scoped source-first input envelopes."""

        cursor = self._connection.execute(
            sql_query("""SELECT artifact_ref, artifact_hash, payload_json
            FROM ontology_build_artifacts
            WHERE organization_id = ? AND workspace_id = ? AND schema_version = ?
            ORDER BY artifact_ref""",'SELECT artifact_ref, artifact_hash, payload_json\n            FROM ontology_build_artifacts\n            WHERE organization_id = %s AND workspace_id = %s AND schema_version = %s\n            ORDER BY artifact_ref'),
            (organization_id, workspace_id, "ontology_build_inputs.v1"),
        )

        def values() -> Iterator[OntologyBuildInputEnvelopeV1]:
            for artifact_ref, artifact_hash, payload_json in cursor:
                value = OntologyBuildInputEnvelopeV1.model_validate_json(payload_json)
                if (
                    value.organization_id,
                    value.workspace_id,
                    value.envelope_ref,
                    value.envelope_hash,
                    canonical_json(value.model_dump(mode="json")),
                ) != (
                    organization_id,
                    workspace_id,
                    artifact_ref,
                    artifact_hash,
                    payload_json,
                ):
                    raise ValueError("ONTOLOGY_BUILD_INPUT_CANDIDATE_MISMATCH")
                yield value

        try:
            yield values()
        finally:
            cursor.close()

    @contextmanager
    def iter_advisory_compiler_inputs(
        self, *, organization_id: str, workspace_id: str
    ) -> AbstractContextManager[Iterator[AdvisoryCompilerInputsV1]]:
        """Stream exact scoped protected compiler-input witnesses."""

        cursor = self._connection.execute(
            sql_query("""SELECT artifact_ref, artifact_hash
            FROM ontology_build_artifacts
            WHERE organization_id = ? AND workspace_id = ? AND schema_version = ?
            ORDER BY artifact_ref""",'SELECT artifact_ref, artifact_hash\n            FROM ontology_build_artifacts\n            WHERE organization_id = %s AND workspace_id = %s AND schema_version = %s\n            ORDER BY artifact_ref'),
            (organization_id, workspace_id, _ADVISORY_INPUT_SCHEMA),
        )

        def values() -> Iterator[AdvisoryCompilerInputsV1]:
            for artifact_ref, artifact_hash in cursor:
                yield self.get_published_advisory_compiler_inputs(
                    organization_id=organization_id,
                    workspace_id=workspace_id,
                    pin=ArtifactPinV1(
                        artifact_kind="advisory_compiler_inputs",
                        artifact_ref=artifact_ref,
                        artifact_hash=artifact_hash,
                    ),
                )

        try:
            yield values()
        finally:
            cursor.close()

    @contextmanager
    def iter_paired_evaluator_bundles(
        self, *, organization_id: str, workspace_id: str
    ) -> AbstractContextManager[Iterator[PairedEvaluatorBundleV1]]:
        """Stream exact scoped paired-evaluator archive witnesses."""

        cursor = self._connection.execute(
            sql_query("""SELECT artifact_ref, artifact_hash, payload_json
            FROM ontology_build_artifacts
            WHERE organization_id = ? AND workspace_id = ? AND schema_version = ?
            ORDER BY artifact_ref""",'SELECT artifact_ref, artifact_hash, payload_json\n            FROM ontology_build_artifacts\n            WHERE organization_id = %s AND workspace_id = %s AND schema_version = %s\n            ORDER BY artifact_ref'),
            (organization_id, workspace_id, "paired_evaluator_bundle.v1"),
        )

        def values() -> Iterator[PairedEvaluatorBundleV1]:
            for artifact_ref, artifact_hash, payload_json in cursor:
                value = PairedEvaluatorBundleV1.model_validate_json(payload_json)
                if (
                    value.organization_id,
                    value.workspace_id,
                    value.evaluator_ref,
                    value.evaluator_hash,
                    canonical_json(value.model_dump(mode="json")),
                ) != (
                    organization_id,
                    workspace_id,
                    artifact_ref,
                    artifact_hash,
                    payload_json,
                ):
                    raise ValueError("PAIRED_EVALUATOR_CANDIDATE_MISMATCH")
                yield value

        try:
            yield values()
        finally:
            cursor.close()

    @contextmanager
    def iter_paired_evaluation_runs(
        self, *, organization_id: str, workspace_id: str
    ) -> AbstractContextManager[Iterator[OntologyPairedEvaluationRunV1]]:
        """Stream exact scoped paired-run discovery witnesses."""

        cursor = self._connection.execute(
            sql_query("""SELECT artifact_ref, artifact_hash, payload_json
            FROM ontology_build_artifacts
            WHERE organization_id = ? AND workspace_id = ? AND schema_version = ?
            ORDER BY artifact_ref""",'SELECT artifact_ref, artifact_hash, payload_json\n            FROM ontology_build_artifacts\n            WHERE organization_id = %s AND workspace_id = %s AND schema_version = %s\n            ORDER BY artifact_ref'),
            (organization_id, workspace_id, "ontology_paired_evaluation_run.v1"),
        )

        def values() -> Iterator[OntologyPairedEvaluationRunV1]:
            for artifact_ref, artifact_hash, payload_json in cursor:
                value = OntologyPairedEvaluationRunV1.model_validate_json(payload_json)
                if (
                    value.organization_id,
                    value.workspace_id,
                    value.run_ref,
                    value.run_hash,
                    canonical_json(value.model_dump(mode="json")),
                ) != (
                    organization_id,
                    workspace_id,
                    artifact_ref,
                    artifact_hash,
                    payload_json,
                ):
                    raise ValueError("PAIRED_EVALUATION_RUN_CANDIDATE_MISMATCH")
                yield value

        try:
            yield values()
        finally:
            cursor.close()

    @staticmethod
    def _advisory_observed_facts_index_values(
        facts: ObservedFactsV1,
        *,
        input_graph_intent: ArtifactPinV1,
        source_capture: ArtifactPinV1,
    ) -> tuple:
        return (
            facts.organization_id,
            facts.workspace_id,
            "observed_facts",
            facts.facts_ref,
            facts.facts_hash,
            "sha256:" + content_hash(facts.model_dump(mode="json")),
            source_capture.artifact_ref,
            source_capture.artifact_hash,
            None,
            None,
            input_graph_intent.artifact_ref,
            input_graph_intent.artifact_hash,
        )

    def get_published_advisory_observed_facts(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        pin: ArtifactPinV1,
    ) -> tuple[ObservedFactsV1, ArtifactPinV1, ArtifactPinV1]:
        """Read exact durable facts and witness pairs; this grants no access."""

        if (
            not isinstance(pin, ArtifactPinV1)
            or pin.artifact_kind != "paired_observed_facts"
        ):
            raise ValueError("ADVISORY_OBSERVED_FACTS_PIN_MISMATCH")
        pin = ArtifactPinV1.model_validate(pin.model_dump(mode="json"))
        artifact = self._connection.execute(
            sql_query("""SELECT organization_id, workspace_id, artifact_ref, artifact_hash,
                schema_version, payload_json FROM ontology_build_artifacts
                WHERE artifact_ref = ?""",'SELECT organization_id, workspace_id, artifact_ref, artifact_hash,\n                schema_version, payload_json FROM ontology_build_artifacts\n                WHERE artifact_ref = %s'),
            (pin.artifact_ref,),
        ).fetchone()
        if artifact is None or artifact[:5] != (
            organization_id,
            workspace_id,
            pin.artifact_ref,
            pin.artifact_hash,
            "paired_observed_facts.v1",
        ):
            raise ValueError("ADVISORY_OBSERVED_FACTS_PUBLICATION_MISMATCH")
        try:
            facts = ObservedFactsV1.model_validate_json(artifact[5])
        except ValidationError as exc:
            raise ValueError("ADVISORY_OBSERVED_FACTS_PAYLOAD_MISMATCH") from exc
        if self._prepare(facts) != artifact:
            raise ValueError("ADVISORY_OBSERVED_FACTS_PAYLOAD_MISMATCH")
        rows = self._connection.execute(
            sql_query(f"""SELECT {", ".join(_ADVISORY_INDEX_COLUMNS)}
            FROM {_ADVISORY_PUBLICATION_INDEX} WHERE artifact_ref = ?""",f"SELECT {', '.join(_ADVISORY_INDEX_COLUMNS)}\n            FROM {_ADVISORY_PUBLICATION_INDEX} WHERE artifact_ref = %s"),
            (pin.artifact_ref,),
        ).fetchall()
        if len(rows) != 1:
            raise ValueError("ADVISORY_OBSERVED_FACTS_INDEX_MISMATCH")
        row = rows[0]
        graph_pin = ArtifactPinV1(
            artifact_kind="graph_intent_brief",
            artifact_ref=row[10],
            artifact_hash=row[11],
        )
        source_pin = ArtifactPinV1(
            artifact_kind="advisory_source_capture",
            artifact_ref=row[6],
            artifact_hash=row[7],
        )
        graph_suffix = graph_pin.artifact_ref.removeprefix("graph-intent-")
        if (
            source_pin.artifact_ref
            != "advisory-source-capture:"
            + source_pin.artifact_hash.removeprefix("sha256:")
            or not graph_pin.artifact_ref.startswith("graph-intent-")
            or len(graph_suffix) != 24
            or any(character not in "0123456789abcdef" for character in graph_suffix)
            or row
            != self._advisory_observed_facts_index_values(
                facts,
                input_graph_intent=graph_pin,
                source_capture=source_pin,
            )
        ):
            raise ValueError("ADVISORY_OBSERVED_FACTS_INDEX_MISMATCH")
        return facts, graph_pin, source_pin

    def _publish_advisory_observed_facts(
        self,
        *,
        context: AdvisoryTrialOwnerContext,
        input_graph_intent: ArtifactPinV1,
        case_id: ControlledTrialCaseId,
        frozen_at: UtcDateTime,
    ) -> ObservedFactsV1:
        from services.runtime_learning_trial_service import (
            _require_advisory_trial_owner_context,
            _resolve_current_observed_facts_values,
        )

        _require_advisory_trial_owner_context(context)
        if context.build_store is not self:
            raise ValueError("ADVISORY_TRIAL_STORE_MISMATCH")
        if in_transaction(self._connection):
            raise RuntimeError(
                "advisory publication cannot nest inside an active transaction"
            )

        facts, graph_pin, source_pin = _resolve_current_observed_facts_values(
            context=context,
            input_graph_intent=input_graph_intent,
            case_id=case_id,
            frozen_at=frozen_at,
        )
        prepared = self._prepare(facts)
        facts_pin = ArtifactPinV1(
            artifact_kind="paired_observed_facts",
            artifact_ref=facts.facts_ref,
            artifact_hash=facts.facts_hash,
        )
        scope = {
            "organization_id": facts.organization_id,
            "workspace_id": facts.workspace_id,
        }
        with self._advisory_publication_transaction():
            locked = _resolve_current_observed_facts_values(
                context=context,
                input_graph_intent=graph_pin,
                case_id=case_id,
                frozen_at=frozen_at,
            )
            if locked != (facts, graph_pin, source_pin):
                raise ValueError("ADVISORY_OBSERVED_FACTS_SOURCE_CHANGED")
            artifact_exists = self._connection.execute(
                sql_query("SELECT 1 FROM ontology_build_artifacts WHERE artifact_ref = ?",'SELECT 1 FROM ontology_build_artifacts WHERE artifact_ref = %s'),
                (facts.facts_ref,),
            ).fetchone()
            association_exists = self._connection.execute(
                sql_query(f"SELECT 1 FROM {_ADVISORY_PUBLICATION_INDEX} WHERE artifact_ref = ?",f'SELECT 1 FROM {_ADVISORY_PUBLICATION_INDEX} WHERE artifact_ref = %s'),
                (facts.facts_ref,),
            ).fetchone()
            if artifact_exists is None and association_exists is None:
                self._insert_prepared((prepared,))
                self._connection.execute(
                    f"INSERT INTO {_ADVISORY_PUBLICATION_INDEX} "
                    f"({', '.join(_ADVISORY_INDEX_COLUMNS)}) "
                    f"VALUES ({', '.join(sql_query('?','%s') for _ in _ADVISORY_INDEX_COLUMNS)})",
                    self._advisory_observed_facts_index_values(
                        facts,
                        input_graph_intent=graph_pin,
                        source_capture=source_pin,
                    ),
                )
            elif artifact_exists is None or association_exists is None:
                raise ValueError("ADVISORY_OBSERVED_FACTS_INDEX_INCOMPLETE")
            saved, saved_graph, saved_source = (
                self.get_published_advisory_observed_facts(**scope, pin=facts_pin)
            )
            if (
                saved != facts
                or saved_graph != graph_pin
                or saved_source != source_pin
                or self._prepare(saved) != prepared
            ):
                raise ValueError("ADVISORY_OBSERVED_FACTS_PUBLICATION_CONFLICT")
            final = _resolve_current_observed_facts_values(
                context=context,
                input_graph_intent=saved_graph,
                case_id=case_id,
                frozen_at=saved.frozen_at,
            )
            if final != (saved, saved_graph, saved_source):
                raise ValueError("ADVISORY_OBSERVED_FACTS_SOURCE_CHANGED")
            self._require_final_advisory_owner_access(context)
        return saved

    def get_published_advisory_suite_review(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        decision: ArtifactPinV1,
    ) -> tuple[AdvisoryPairedSuiteReviewV1, OntologyPairedObligationSuiteV1]:
        """Read one exact protected receipt/suite pair; this grants no access."""

        from services._runtime_context_advisory.paired_inputs import (
            _suite_from_subject,
        )

        if (
            not isinstance(decision, ArtifactPinV1)
            or decision.artifact_kind != "advisory_paired_suite_review"
            or not decision.artifact_ref.startswith(_ADVISORY_SUITE_DECISION_PREFIX)
        ):
            raise ValueError("ADVISORY_SUITE_DECISION_PIN_MISMATCH")
        decision = ArtifactPinV1.model_validate(decision.model_dump(mode="json"))
        receipt_row = self._connection.execute(
            sql_query("""SELECT organization_id, workspace_id, artifact_ref, artifact_hash,
                      schema_version, payload_json
               FROM ontology_build_artifacts WHERE artifact_ref = ?""",'SELECT organization_id, workspace_id, artifact_ref, artifact_hash,\n                      schema_version, payload_json\n               FROM ontology_build_artifacts WHERE artifact_ref = %s'),
            (decision.artifact_ref,),
        ).fetchone()
        if receipt_row is None or receipt_row[:5] != (
            organization_id,
            workspace_id,
            decision.artifact_ref,
            decision.artifact_hash,
            _ADVISORY_SUITE_REVIEW_SCHEMA,
        ):
            raise ValueError("ADVISORY_SUITE_REVIEW_PUBLICATION_MISMATCH")
        try:
            receipt = AdvisoryPairedSuiteReviewV1.model_validate_json(receipt_row[5])
        except ValidationError as exc:
            raise ValueError("ADVISORY_SUITE_REVIEW_PAYLOAD_MISMATCH") from exc
        if self._prepare(receipt) != receipt_row:
            raise ValueError("ADVISORY_SUITE_REVIEW_PAYLOAD_MISMATCH")
        suite = _suite_from_subject(receipt)
        suite_row = self._connection.execute(
            sql_query("""SELECT organization_id, workspace_id, artifact_ref, artifact_hash,
                      schema_version, payload_json
               FROM ontology_build_artifacts WHERE artifact_ref = ?""",'SELECT organization_id, workspace_id, artifact_ref, artifact_hash,\n                      schema_version, payload_json\n               FROM ontology_build_artifacts WHERE artifact_ref = %s'),
            (suite.suite_ref,),
        ).fetchone()
        if suite_row is None or suite_row[:5] != (
            organization_id,
            workspace_id,
            suite.suite_ref,
            suite.suite_hash,
            "ontology_paired_obligation_suite.v1",
        ):
            raise ValueError("ADVISORY_SUITE_PUBLICATION_MISMATCH")
        try:
            saved_suite = OntologyPairedObligationSuiteV1.model_validate_json(
                suite_row[5]
            )
        except ValidationError as exc:
            raise ValueError("ADVISORY_SUITE_PAYLOAD_MISMATCH") from exc
        if saved_suite != suite or self._prepare(saved_suite) != suite_row:
            raise ValueError("ADVISORY_SUITE_PAYLOAD_MISMATCH")
        return receipt, saved_suite

    def _publish_advisory_suite_review(
        self,
        *,
        context: AdvisoryTrialOwnerContext,
        compiler_inputs: ArtifactPinV1,
        receipt: AdvisoryPairedSuiteReviewV1,
        suite: OntologyPairedObligationSuiteV1,
    ) -> tuple[AdvisoryPairedSuiteReviewV1, OntologyPairedObligationSuiteV1]:
        from services.runtime_learning_trial_service import (
            _require_advisory_trial_owner_context,
            _resolve_current_suite_review_values,
        )

        _require_advisory_trial_owner_context(context)
        if context.build_store is not self:
            raise ValueError("ADVISORY_TRIAL_STORE_MISMATCH")
        if in_transaction(self._connection):
            raise RuntimeError(
                "advisory publication cannot nest inside an active transaction"
            )
        if not isinstance(receipt, AdvisoryPairedSuiteReviewV1) or not isinstance(
            suite, OntologyPairedObligationSuiteV1
        ):
            raise ValueError("ADVISORY_SUITE_PAYLOAD_MISMATCH")
        receipt = AdvisoryPairedSuiteReviewV1.model_validate(
            receipt.model_dump(mode="json")
        )
        suite = OntologyPairedObligationSuiteV1.model_validate(
            suite.model_dump(mode="json")
        )
        competency = (
            ArtifactPinV1(
                artifact_kind="competency_contract",
                artifact_ref=suite.competency_contract_ref,
                artifact_hash=suite.competency_contract_hash,
            )
            if suite.competency_contract_ref is not None
            else None
        )

        def resolve_expected():
            return _resolve_current_suite_review_values(
                context=context,
                compiler_inputs=compiler_inputs,
                cases=tuple(suite.cases),
                reviewed_at=suite.reviewed_at,
                competency=competency,
            )

        expected = resolve_expected()
        if expected != (receipt, suite):
            raise ValueError("ADVISORY_SUITE_PAYLOAD_MISMATCH")
        prepared = (self._prepare(receipt), self._prepare(suite))
        decision = ArtifactPinV1(
            artifact_kind="advisory_paired_suite_review",
            artifact_ref=receipt.decision_ref,
            artifact_hash=receipt.decision_hash,
        )
        scope = {
            "organization_id": receipt.organization_id,
            "workspace_id": receipt.workspace_id,
        }
        refs = (receipt.decision_ref, suite.suite_ref)
        with self._advisory_publication_transaction():
            if resolve_expected() != expected:
                raise ValueError("ADVISORY_SUITE_SOURCE_CHANGED")
            rows = self._connection.execute(
                sql_query("""SELECT artifact_ref FROM ontology_build_artifacts
                   WHERE artifact_ref IN (?, ?)""",'SELECT artifact_ref FROM ontology_build_artifacts\n                   WHERE artifact_ref IN (%s, %s)'),
                refs,
            ).fetchall()
            if not rows:
                self._insert_prepared(prepared)
            elif len(rows) != 2 or {row[0] for row in rows} != set(refs):
                raise ValueError("ADVISORY_SUITE_PARTIAL_OR_CONFLICTING")
            saved = self.get_published_advisory_suite_review(**scope, decision=decision)
            if saved != expected or tuple(self._prepare(value) for value in saved) != (
                prepared
            ):
                raise ValueError("ADVISORY_SUITE_PUBLICATION_CONFLICT")
            if resolve_expected() != saved:
                raise ValueError("ADVISORY_SUITE_SOURCE_CHANGED")
            self._require_final_advisory_owner_access(context)
        return saved

    def _advisory_initial_index_rows(
        self, inputs: AdvisoryCompilerInputsV1
    ) -> list[tuple]:
        # Query the exact invocation key, including foreign-scope rows so a
        # corrupted association cannot turn an existing invocation into a miss.
        return self._connection.execute(
            sql_query("""SELECT organization_id, workspace_id, publication_kind,
                artifact_ref, artifact_hash, payload_hash, input_ref, input_hash,
                derivation_ref, derivation_hash, subject_ref, subject_hash
            FROM advisory_trial_native_publication_index
            WHERE input_ref = ? AND publication_kind IN
                ('initial_candidate', 'initial_validation', 'initial_plan')""","SELECT organization_id, workspace_id, publication_kind,\n                artifact_ref, artifact_hash, payload_hash, input_ref, input_hash,\n                derivation_ref, derivation_hash, subject_ref, subject_hash\n            FROM advisory_trial_native_publication_index\n            WHERE input_ref = %s AND publication_kind IN\n                ('initial_candidate', 'initial_validation', 'initial_plan')"),
            (inputs.input_ref,),
        ).fetchall()

    @staticmethod
    def _advisory_initial_index_values(
        inputs: AdvisoryCompilerInputsV1,
        output: BaseModel,
        publication_kind: str,
        ref_field: str,
        hash_field: str,
    ) -> tuple:
        ref, digest = getattr(output, ref_field), getattr(output, hash_field)
        return (
            inputs.organization_id,
            inputs.workspace_id,
            publication_kind,
            ref,
            digest,
            "sha256:" + content_hash(output.model_dump(mode="json")),
            inputs.input_ref,
            inputs.input_hash,
            None,
            None,
            ref,
            digest,
        )

    def _read_advisory_native_artifact(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        pin: ArtifactPinV1,
        contract_type: type[ContractT],
    ) -> ContractT:
        row = self._connection.execute(
            sql_query("""SELECT organization_id, workspace_id, artifact_ref, artifact_hash,
                schema_version, payload_json FROM ontology_build_artifacts
                WHERE artifact_ref = ?""",'SELECT organization_id, workspace_id, artifact_ref, artifact_hash,\n                schema_version, payload_json FROM ontology_build_artifacts\n                WHERE artifact_ref = %s'),
            (pin.artifact_ref,),
        ).fetchone()
        if row is None or row[:4] != (
            organization_id,
            workspace_id,
            pin.artifact_ref,
            pin.artifact_hash,
        ):
            raise ValueError("ADVISORY_INITIAL_ARTIFACT_MISMATCH")
        value = contract_type.model_validate_json(row[5])
        if self._prepare(value) != row:
            raise ValueError("ADVISORY_INITIAL_PAYLOAD_MISMATCH")
        return value

    def get_advisory_initial_outputs(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        inputs: ArtifactPinV1,
    ) -> tuple[
        NormativeSemanticCandidateSetV1, ValidationContractV1, CompiledRulePlanV1
    ]:
        """Read exact storage proof, without claiming current access or source bytes."""
        from services._runtime_context_advisory.compiler import (
            _validate_stored_initial_output_values,
        )

        saved_input = self.get_published_advisory_compiler_inputs(
            organization_id=organization_id,
            workspace_id=workspace_id,
            pin=inputs,
        )
        rows = self._advisory_initial_index_rows(saved_input)
        if len(rows) != 3 or {row[2] for row in rows} != {
            item[0] for item in _ADVISORY_INITIAL_OUTPUTS
        }:
            raise ValueError("ADVISORY_INITIAL_INDEX_INCOMPLETE")
        by_kind = {row[2]: row for row in rows}
        outputs = []
        for (
            publication_kind,
            kind,
            contract,
            ref_field,
            hash_field,
        ) in _ADVISORY_INITIAL_OUTPUTS:
            row = by_kind[publication_kind]
            output = self._read_advisory_native_artifact(
                organization_id=organization_id,
                workspace_id=workspace_id,
                pin=ArtifactPinV1(
                    artifact_kind=kind, artifact_ref=row[3], artifact_hash=row[4]
                ),
                contract_type=contract,
            )
            expected = self._advisory_initial_index_values(
                saved_input, output, publication_kind, ref_field, hash_field
            )
            all_rows = self._connection.execute(
                sql_query(f"SELECT * FROM {_ADVISORY_PUBLICATION_INDEX} WHERE artifact_ref = ?",f'SELECT * FROM {_ADVISORY_PUBLICATION_INDEX} WHERE artifact_ref = %s'),
                (row[3],),
            ).fetchall()
            if row != expected or all_rows != [expected]:
                raise ValueError("ADVISORY_INITIAL_INDEX_MISMATCH")
            outputs.append(output)
        boundary = self._read_advisory_native_artifact(
            organization_id=organization_id,
            workspace_id=workspace_id,
            pin=saved_input.boundary,
            contract_type=CanonicalEvidenceBoundaryV1,
        )
        competency = self._read_advisory_native_artifact(
            organization_id=organization_id,
            workspace_id=workspace_id,
            pin=saved_input.competency,
            contract_type=CompetencyContractV1,
        )
        candidate, validation, plan = outputs
        _validate_stored_initial_output_values(
            inputs=saved_input,
            boundary=boundary,
            competency=competency,
            candidate=candidate,
            validation=validation,
            plan=plan,
        )
        return candidate, validation, plan

    def get_advisory_producing_inputs(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        output: ArtifactPinV1,
    ) -> AdvisoryCompilerInputsV1:
        """Resolve an exact initial or trial output through complete storage proof."""
        if not isinstance(output, ArtifactPinV1):
            raise ValueError("ADVISORY_INITIAL_OUTPUT_PIN_MISMATCH")
        specification = next(
            (
                item
                for item in _ADVISORY_INITIAL_OUTPUTS
                if item[1] == output.artifact_kind
            ),
            None,
        )
        if specification is None:
            raise ValueError("ADVISORY_INITIAL_OUTPUT_PIN_MISMATCH")
        rows = self._connection.execute(
            sql_query(f"SELECT * FROM {_ADVISORY_PUBLICATION_INDEX} WHERE artifact_ref = ?",f'SELECT * FROM {_ADVISORY_PUBLICATION_INDEX} WHERE artifact_ref = %s'),
            (output.artifact_ref,),
        ).fetchall()
        if len(rows) == 1 and rows[0][2] in {"trial_candidate", "trial_plan"}:
            row = rows[0]
            if output.artifact_kind not in {
                "normative_semantic_candidate_set",
                "compiled_rule_plan",
            }:
                raise ValueError("ADVISORY_TRIAL_DERIVATION_PIN_MISMATCH")
            expected_kind = (
                "trial_candidate"
                if output.artifact_kind == "normative_semantic_candidate_set"
                else "trial_plan"
            )
            if row[:5] != (
                organization_id,
                workspace_id,
                expected_kind,
                output.artifact_ref,
                output.artifact_hash,
            ):
                raise ValueError("ADVISORY_TRIAL_INDEX_MISMATCH")
            values = self._read_advisory_trial_outputs(
                organization_id=organization_id,
                workspace_id=workspace_id,
                pin=ArtifactPinV1(
                    artifact_kind="advisory_trial_variant",
                    artifact_ref=row[8],
                    artifact_hash=row[9],
                ),
            )
            value = values[0 if expected_kind == "trial_candidate" else 1]
            if self._prepare(value)[2:4] != (output.artifact_ref, output.artifact_hash):
                raise ValueError("ADVISORY_TRIAL_INDEX_MISMATCH")
            return self.get_published_advisory_compiler_inputs(
                organization_id=organization_id,
                workspace_id=workspace_id,
                pin=values[2].inputs,
            )
        if len(rows) != 1 or rows[0][:5] != (
            organization_id,
            workspace_id,
            specification[0],
            output.artifact_ref,
            output.artifact_hash,
        ):
            raise ValueError("ADVISORY_INITIAL_INDEX_MISMATCH")
        row = rows[0]
        input_pin = ArtifactPinV1(
            artifact_kind="advisory_compiler_inputs",
            artifact_ref=row[6],
            artifact_hash=row[7],
        )
        saved = self.get_published_advisory_compiler_inputs(
            organization_id=organization_id,
            workspace_id=workspace_id,
            pin=input_pin,
        )
        outputs = self.get_advisory_initial_outputs(
            organization_id=organization_id,
            workspace_id=workspace_id,
            inputs=input_pin,
        )
        value = outputs[_ADVISORY_INITIAL_OUTPUTS.index(specification)]
        if (getattr(value, specification[3]), getattr(value, specification[4])) != (
            output.artifact_ref,
            output.artifact_hash,
        ):
            raise ValueError("ADVISORY_INITIAL_OUTPUT_PIN_MISMATCH")
        return saved

    def _advisory_trial_index_rows(
        self, receipt: AdvisoryTrialVariantV1
    ) -> list[tuple]:
        # Include conflicting subject/ref associations even when the requested
        # derivation key itself has no rows. Foreign scope is never a cache miss.
        return self._connection.execute(
            sql_query(f"""SELECT * FROM {_ADVISORY_PUBLICATION_INDEX}
            WHERE derivation_ref = ? OR artifact_ref IN (?, ?)
            OR (publication_kind = 'trial_derivation' AND subject_ref = ?)""",f"SELECT * FROM {_ADVISORY_PUBLICATION_INDEX}\n            WHERE derivation_ref = %s OR artifact_ref IN (%s, %s)\n            OR (publication_kind = 'trial_derivation' AND subject_ref = %s)"),
            (
                receipt.variant_ref,
                receipt.variant_ref,
                receipt.candidate.artifact_ref,
                receipt.candidate.artifact_ref,
            ),
        ).fetchall()

    @staticmethod
    def _advisory_trial_index_values(
        inputs: AdvisoryCompilerInputsV1,
        receipt: AdvisoryTrialVariantV1,
        output: BaseModel,
        specification: tuple,
    ) -> tuple:
        ref, digest = (
            getattr(output, specification[3]),
            getattr(output, specification[4]),
        )
        subject = (
            (receipt.candidate.artifact_ref, receipt.candidate.artifact_hash)
            if specification[0] == "trial_derivation"
            else (ref, digest)
        )
        return (
            inputs.organization_id,
            inputs.workspace_id,
            specification[0],
            ref,
            digest,
            "sha256:" + content_hash(output.model_dump(mode="json")),
            inputs.input_ref,
            inputs.input_hash,
            receipt.variant_ref,
            receipt.variant_hash,
            *subject,
        )

    def _advisory_trial_source_values(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        receipt: AdvisoryTrialVariantV1,
        _allow_repair: bool = True,
    ) -> tuple[
        AdvisoryCompilerInputsV1,
        tuple[
            NormativeSemanticCandidateSetV1, ValidationContractV1, CompiledRulePlanV1
        ],
        NormativeSemanticCandidateSetV1,
        AdvisoryTrialVariantV1 | None,
        CompetencyContractV1,
    ]:
        scope = dict(organization_id=organization_id, workspace_id=workspace_id)
        if (receipt.organization_id, receipt.workspace_id) != (
            organization_id,
            workspace_id,
        ):
            raise ValueError("ADVISORY_TRIAL_SCOPE_MISMATCH")
        inputs = self.get_published_advisory_compiler_inputs(
            **scope, pin=receipt.inputs
        )
        initial = self.get_advisory_initial_outputs(**scope, inputs=receipt.inputs)
        original = initial[0]
        original_pin = ArtifactPinV1(
            artifact_kind="normative_semantic_candidate_set",
            artifact_ref=original.candidate_set_ref,
            artifact_hash=original.candidate_set_hash,
        )
        prior = None
        if receipt.variant == "ADD_MISSING_FRAME":
            if not _allow_repair:
                raise ValueError("repair predecessor is not the native omitted variant")
            prior_pin = self._advisory_derivation_pin_for_candidate(
                **scope, candidate=receipt.source
            )
            source, _prior_plan, prior = self._read_advisory_trial_outputs(
                **scope, pin=prior_pin, _allow_repair=False
            )
            if (
                prior.variant != "OMIT_EXCEPTION"
                or prior.inputs != receipt.inputs
                or prior.source != original_pin
            ):
                raise ValueError("repair predecessor is not the native omitted variant")
        else:
            if receipt.source != original_pin:
                raise ValueError(
                    "trial receipt source is not an initial native candidate"
                )
            source = original
        competency = self._read_advisory_native_artifact(
            **scope, pin=inputs.competency, contract_type=CompetencyContractV1
        )
        return inputs, initial, source, prior, competency

    def _read_advisory_trial_outputs(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        pin: ArtifactPinV1,
        _allow_repair: bool = True,
    ) -> tuple[
        NormativeSemanticCandidateSetV1, CompiledRulePlanV1, AdvisoryTrialVariantV1
    ]:
        from services._runtime_context_advisory.compiler import (
            _validate_advisory_trial_output_values,
        )

        if (
            not isinstance(pin, ArtifactPinV1)
            or pin.artifact_kind != "advisory_trial_variant"
        ):
            raise ValueError("ADVISORY_TRIAL_DERIVATION_PIN_MISMATCH")
        scope = dict(organization_id=organization_id, workspace_id=workspace_id)
        receipt = self._read_advisory_native_artifact(
            **scope, pin=pin, contract_type=AdvisoryTrialVariantV1
        )
        rows = self._advisory_trial_index_rows(receipt)
        if len(rows) != 3 or {row[2] for row in rows} != {
            item[0] for item in _ADVISORY_TRIAL_OUTPUTS
        }:
            raise ValueError("ADVISORY_TRIAL_INDEX_INCOMPLETE")
        inputs, initial, source, prior, competency = self._advisory_trial_source_values(
            **scope, receipt=receipt, _allow_repair=_allow_repair
        )
        by_kind = {row[2]: row for row in rows}
        outputs = []
        for specification in _ADVISORY_TRIAL_OUTPUTS:
            row = by_kind[specification[0]]
            value = self._read_advisory_native_artifact(
                **scope,
                pin=ArtifactPinV1(
                    artifact_kind=specification[1],
                    artifact_ref=row[3],
                    artifact_hash=row[4],
                ),
                contract_type=specification[2],
            )
            expected = self._advisory_trial_index_values(
                inputs, receipt, value, specification
            )
            all_rows = self._connection.execute(
                sql_query(f"SELECT * FROM {_ADVISORY_PUBLICATION_INDEX} WHERE artifact_ref = ?",f'SELECT * FROM {_ADVISORY_PUBLICATION_INDEX} WHERE artifact_ref = %s'),
                (row[3],),
            ).fetchall()
            if row != expected or all_rows != [expected]:
                raise ValueError("ADVISORY_TRIAL_INDEX_MISMATCH")
            outputs.append(value)
        candidate, plan, saved = outputs
        if saved != receipt:
            raise ValueError("ADVISORY_TRIAL_DERIVATION_MISMATCH")
        _validate_advisory_trial_output_values(
            inputs=inputs,
            source=source,
            original=initial[0],
            prior_receipt=prior,
            validation=initial[1],
            competency=competency,
            candidate=candidate,
            plan=plan,
            receipt=receipt,
        )
        return candidate, plan, receipt

    def get_published_advisory_trial_derivation(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        pin: ArtifactPinV1,
    ) -> AdvisoryTrialVariantV1:
        return self._read_advisory_trial_outputs(
            organization_id=organization_id, workspace_id=workspace_id, pin=pin
        )[2]

    def _advisory_derivation_pin_for_candidate(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        candidate: ArtifactPinV1,
    ) -> ArtifactPinV1:
        if (
            not isinstance(candidate, ArtifactPinV1)
            or candidate.artifact_kind != "normative_semantic_candidate_set"
        ):
            raise ValueError("ADVISORY_TRIAL_CANDIDATE_PIN_MISMATCH")
        rows = self._connection.execute(
            sql_query(f"SELECT * FROM {_ADVISORY_PUBLICATION_INDEX} WHERE publication_kind = 'trial_derivation' AND subject_ref = ?",f"SELECT * FROM {_ADVISORY_PUBLICATION_INDEX} WHERE publication_kind = 'trial_derivation' AND subject_ref = %s"),
            (candidate.artifact_ref,),
        ).fetchall()
        if (
            len(rows) != 1
            or rows[0][:2] != (organization_id, workspace_id)
            or rows[0][10:] != (candidate.artifact_ref, candidate.artifact_hash)
        ):
            raise ValueError("ADVISORY_TRIAL_INDEX_MISMATCH")
        return ArtifactPinV1(
            artifact_kind="advisory_trial_variant",
            artifact_ref=rows[0][3],
            artifact_hash=rows[0][4],
        )

    def get_advisory_trial_derivation_for_candidate(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        candidate: ArtifactPinV1,
    ) -> AdvisoryTrialVariantV1:
        pin = self._advisory_derivation_pin_for_candidate(
            organization_id=organization_id,
            workspace_id=workspace_id,
            candidate=candidate,
        )
        receipt = self.get_published_advisory_trial_derivation(
            organization_id=organization_id, workspace_id=workspace_id, pin=pin
        )
        if receipt.candidate != candidate:
            raise ValueError("ADVISORY_TRIAL_CANDIDATE_PIN_MISMATCH")
        return receipt

    def get_advisory_trial_plan_for_derivation(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        derivation: ArtifactPinV1,
    ) -> CompiledRulePlanV1:
        return self._read_advisory_trial_outputs(
            organization_id=organization_id, workspace_id=workspace_id, pin=derivation
        )[1]

    def _publish_advisory_trial_derivation(
        self,
        *,
        context: AdvisoryTrialOwnerContext,
        derivation: AdvisoryTrialVariantV1,
    ) -> AdvisoryTrialVariantV1:
        from services.runtime_learning_trial_service import (
            _compile_advisory_trial_values,
            _require_advisory_trial_owner_context,
            _resolve_current_compiler_input_values,
        )
        from services._runtime_context_advisory.compiler import (
            _validated_live_value,
            _validate_advisory_trial_output_values,
            _validate_live_initial_output_values,
        )

        _require_advisory_trial_owner_context(context)
        if context.build_store is not self:
            raise ValueError("ADVISORY_TRIAL_STORE_MISMATCH")
        if in_transaction(self._connection):
            raise RuntimeError(
                "advisory publication cannot nest inside an active transaction"
            )
        derivation = _validated_live_value(
            derivation, AdvisoryTrialVariantV1, "derivation"
        )
        scope = dict(
            organization_id=context.actor_context.organization_id,
            workspace_id=context.actor_context.workspace_id,
        )
        inputs, initial, source, prior, competency = self._advisory_trial_source_values(
            **scope, receipt=derivation
        )
        boundary, live_competency, _brief, capture = (
            _resolve_current_compiler_input_values(context=context, inputs=inputs)
        )
        _validate_live_initial_output_values(
            inputs=inputs,
            boundary=boundary,
            competency=live_competency,
            source_capture=capture,
            candidate=initial[0],
            validation=initial[1],
            plan=initial[2],
        )
        pin = ArtifactPinV1(
            artifact_kind="advisory_trial_variant",
            artifact_ref=derivation.variant_ref,
            artifact_hash=derivation.variant_hash,
        )
        had_publication = (
            bool(self._advisory_trial_index_rows(derivation))
            or self._connection.execute(
                sql_query("SELECT 1 FROM ontology_build_artifacts WHERE artifact_ref = ?",'SELECT 1 FROM ontology_build_artifacts WHERE artifact_ref = %s'),
                (derivation.variant_ref,),
            ).fetchone()
            is not None
        )
        if had_publication:
            proposed = self._read_advisory_trial_outputs(**scope, pin=pin)
        else:
            proposed = _compile_advisory_trial_values(
                inputs=inputs,
                source=source,
                original=initial[0],
                validation=initial[1],
                competency=competency,
                derivation=derivation,
            )
        if canonical_json(proposed[2].model_dump(mode="json")) != canonical_json(
            derivation.model_dump(mode="json")
        ):
            raise ValueError("ADVISORY_TRIAL_DERIVATION_MISMATCH")
        _validate_advisory_trial_output_values(
            inputs=inputs,
            source=source,
            original=initial[0],
            prior_receipt=prior,
            validation=initial[1],
            competency=competency,
            candidate=proposed[0],
            plan=proposed[1],
            receipt=proposed[2],
        )
        prepared = tuple(self._prepare(value) for value in proposed)
        with self._advisory_publication_transaction():
            current = self._advisory_trial_source_values(**scope, receipt=derivation)
            if current != (inputs, initial, source, prior, competency):
                raise ValueError("ADVISORY_TRIAL_SOURCE_MISMATCH")
            boundary, live_competency, _brief, capture = (
                _resolve_current_compiler_input_values(context=context, inputs=inputs)
            )
            _validate_live_initial_output_values(
                inputs=inputs,
                boundary=boundary,
                competency=live_competency,
                source_capture=capture,
                candidate=initial[0],
                validation=initial[1],
                plan=initial[2],
            )
            if (
                self._advisory_trial_index_rows(derivation)
                or self._connection.execute(
                    sql_query("SELECT 1 FROM ontology_build_artifacts WHERE artifact_ref = ?",'SELECT 1 FROM ontology_build_artifacts WHERE artifact_ref = %s'),
                    (derivation.variant_ref,),
                ).fetchone()
                is not None
            ):
                outputs = self._read_advisory_trial_outputs(**scope, pin=pin)
                if tuple(self._prepare(value) for value in outputs) != prepared:
                    raise ValueError("ADVISORY_TRIAL_PAYLOAD_MISMATCH")
            else:
                if had_publication:
                    raise ValueError("ADVISORY_TRIAL_INDEX_INCOMPLETE")
                for value in prepared:
                    if (
                        self._connection.execute(
                            sql_query(f"SELECT 1 FROM {_ADVISORY_PUBLICATION_INDEX} WHERE artifact_ref = ?",f'SELECT 1 FROM {_ADVISORY_PUBLICATION_INDEX} WHERE artifact_ref = %s'),
                            (value[2],),
                        ).fetchone()
                        is not None
                    ):
                        raise ValueError("ADVISORY_TRIAL_OUTPUT_ALREADY_ASSOCIATED")
                _validate_advisory_trial_output_values(
                    inputs=inputs,
                    source=source,
                    original=initial[0],
                    prior_receipt=prior,
                    validation=initial[1],
                    competency=competency,
                    candidate=proposed[0],
                    plan=proposed[1],
                    receipt=proposed[2],
                )
                self._insert_prepared(prepared)
                for specification, output in zip(
                    _ADVISORY_TRIAL_OUTPUTS, proposed, strict=True
                ):
                    self._connection.execute(
                        sql_query(f"INSERT INTO {_ADVISORY_PUBLICATION_INDEX} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",f'INSERT INTO {_ADVISORY_PUBLICATION_INDEX} VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'),
                        self._advisory_trial_index_values(
                            inputs, derivation, output, specification
                        ),
                    )
                outputs = self._read_advisory_trial_outputs(**scope, pin=pin)
            _resolve_current_compiler_input_values(context=context, inputs=inputs)
            self._require_final_advisory_owner_access(context)
        return outputs[2]

    def _publish_advisory_trial_pair(
        self,
        *,
        context: AdvisoryTrialOwnerContext,
        baseline_inputs: AdvisoryTrialBuildInputsV1,
        baseline_snapshot: ExploratoryGraphSnapshotV1,
        baseline_integrity: ExploratoryGraphSnapshotIntegrityV1,
        baseline_manifest: OntologyBuildManifestV1,
        successor_inputs: AdvisoryTrialBuildInputsV1,
        successor_snapshot: ExploratoryGraphSnapshotV1,
        successor_integrity: ExploratoryGraphSnapshotIntegrityV1,
        successor_manifest: OntologyBuildManifestV1,
    ) -> PairedEvaluationSubjectV1:
        from services.runtime_learning_trial_service import (
            RuntimeLearningTrialService,
            _authenticate_advisory_trial_pair_resources,
            _require_advisory_trial_owner_context,
            _resolve_context_only_pair_values,
        )

        _require_advisory_trial_owner_context(context)
        if context.build_store is not self:
            raise ValueError("ADVISORY_TRIAL_STORE_MISMATCH")
        if in_transaction(self._connection):
            raise RuntimeError(
                "advisory publication cannot nest inside an active transaction"
            )
        supplied = (
            baseline_inputs,
            baseline_snapshot,
            baseline_integrity,
            baseline_manifest,
            successor_inputs,
            successor_snapshot,
            successor_integrity,
            successor_manifest,
        )
        supplied = tuple(
            type(value).model_validate(value.model_dump(mode="json"))
            for value in supplied
        )
        (
            baseline_inputs,
            baseline_snapshot,
            baseline_integrity,
            baseline_manifest,
            successor_inputs,
            successor_snapshot,
            successor_integrity,
            successor_manifest,
        ) = supplied

        def pin(kind, value, ref_field, hash_field):
            return ArtifactPinV1(
                artifact_kind=kind,
                artifact_ref=getattr(value, ref_field),
                artifact_hash=getattr(value, hash_field),
            )

        def resolve_expected():
            producer = _authenticate_advisory_trial_pair_resources(
                context.installed_resources
            )
            resolved = _resolve_context_only_pair_values(
                context=context,
                source_manifest=ArtifactPinV1(
                    artifact_kind="ontology_build_manifest",
                    artifact_ref=baseline_inputs.origin_manifest.artifact_ref,
                    artifact_hash=baseline_inputs.origin_manifest.artifact_hash,
                ),
                baseline_candidate=baseline_inputs.candidate,
                baseline_derivation=baseline_inputs.derivation,
                baseline_plan=baseline_inputs.plan,
                successor_candidate=successor_inputs.candidate,
                successor_derivation=successor_inputs.derivation,
                successor_plan=successor_inputs.plan,
                validation=baseline_inputs.validation,
                compiler_inputs=baseline_inputs.compiler_inputs,
                producer_module_bytes=producer,
            )
            if (
                baseline_inputs.recorded_at != successor_inputs.recorded_at
                or baseline_inputs.validation != successor_inputs.validation
                or baseline_inputs.compiler_inputs != successor_inputs.compiler_inputs
            ):
                raise ValueError("ADVISORY_TRIAL_PAIR_INPUT_MISMATCH")
            expected = RuntimeLearningTrialService._project_context_only_pair_values(
                origin_inputs=resolved.origin_inputs,
                origin_manifest=resolved.origin_manifest,
                origin_snapshot=resolved.origin_snapshot,
                origin_integrity=resolved.origin_integrity,
                source_capture=resolved.source_capture,
                boundary=resolved.boundary,
                assessment=resolved.assessment,
                brief=resolved.brief,
                competency=resolved.competency,
                compiler_inputs=resolved.compiler_inputs,
                original_candidate=resolved.original_candidate,
                baseline_candidate=resolved.baseline_candidate,
                baseline_derivation=resolved.baseline_derivation,
                baseline_plan=resolved.baseline_plan,
                successor_candidate=resolved.successor_candidate,
                successor_derivation=resolved.successor_derivation,
                successor_plan=resolved.successor_plan,
                validation=resolved.validation,
                producer_module_bytes=producer,
                prepared_at=baseline_inputs.recorded_at,
            )
            if tuple(
                canonical_json(value.model_dump(mode="json")) for value in expected[:8]
            ) != tuple(
                canonical_json(value.model_dump(mode="json")) for value in supplied
            ):
                raise ValueError("ADVISORY_TRIAL_PAIR_PAYLOAD_MISMATCH")
            return expected

        expected = resolve_expected()
        prepared_artifacts = tuple(
            self._prepare(value)
            for value in (
                expected[0],
                expected[1],
                expected[3],
                expected[4],
                expected[5],
                expected[7],
            )
        )
        prepared_integrities = tuple(
            self._prepare(value) for value in (expected[2], expected[6])
        )
        expected_prepared = (*prepared_artifacts, *prepared_integrities)
        refs = tuple(item[2] for item in expected_prepared)
        if len(set(refs)) != 8:
            raise ValueError("ADVISORY_TRIAL_PAIR_IDENTITY_CONFLICT")

        def require_exact_state() -> bool:
            rows = {
                row[2]: row
                for row in self._connection.execute(
                    sql_query("""SELECT organization_id, workspace_id, artifact_ref,
                              artifact_hash, schema_version, payload_json
                       FROM ontology_build_artifacts
                       WHERE artifact_ref IN (?, ?, ?, ?, ?, ?, ?, ?)""",'SELECT organization_id, workspace_id, artifact_ref,\n                              artifact_hash, schema_version, payload_json\n                       FROM ontology_build_artifacts\n                       WHERE artifact_ref IN (%s, %s, %s, %s, %s, %s, %s, %s)'),
                    refs,
                ).fetchall()
            }
            if not rows:
                return False
            if len(rows) != 8 or any(
                rows.get(item[2]) != item for item in expected_prepared
            ):
                raise ValueError("ADVISORY_TRIAL_PAIR_PARTIAL_OR_CONFLICTING")
            scope = dict(
                organization_id=baseline_inputs.organization_id,
                workspace_id=baseline_inputs.workspace_id,
            )
            for snapshot in (baseline_snapshot, successor_snapshot):
                saved_snapshot, saved_integrity = (
                    self.get_verified_exploratory_snapshot(
                        **scope,
                        pin=pin(
                            "exploratory_graph_snapshot",
                            snapshot,
                            "snapshot_ref",
                            "snapshot_hash",
                        ),
                    )
                )
                expected_integrity = (
                    baseline_integrity
                    if snapshot.snapshot_ref == baseline_snapshot.snapshot_ref
                    else successor_integrity
                )
                if saved_snapshot != snapshot or saved_integrity != expected_integrity:
                    raise ValueError("ADVISORY_TRIAL_PAIR_INTEGRITY_CONFLICT")
            return True

        with self._advisory_publication_transaction():
            current = resolve_expected()
            if tuple(current[:8]) != tuple(expected[:8]):
                raise ValueError("ADVISORY_TRIAL_PAIR_SOURCE_CHANGED")
            if not require_exact_state():
                self._insert_snapshot_integrities(
                    snapshots=(expected[1], expected[5]),
                    integrities=(expected[2], expected[6]),
                    prepared_artifacts=prepared_artifacts,
                    prepared_integrities=prepared_integrities,
                )
                if not require_exact_state():
                    raise ValueError("ADVISORY_TRIAL_PAIR_PUBLICATION_INCOMPLETE")
            final_values = resolve_expected()
            if tuple(final_values[:8]) != tuple(expected[:8]):
                raise ValueError("ADVISORY_TRIAL_PAIR_SOURCE_CHANGED")
            self._require_final_advisory_owner_access(context)
        return expected[8]

    def _compile_and_publish_advisory_initial(
        self,
        *,
        context: AdvisoryTrialOwnerContext,
        inputs: ArtifactPinV1,
    ) -> tuple[
        NormativeSemanticCandidateSetV1, ValidationContractV1, CompiledRulePlanV1
    ]:
        from services.runtime_learning_trial_service import (
            _compile_advisory_initial_values,
            _require_advisory_trial_owner_context,
            _resolve_current_compiler_input_values,
        )
        from services._runtime_context_advisory.compiler import (
            _validate_live_initial_output_values,
        )

        _require_advisory_trial_owner_context(context)
        if context.build_store is not self:
            raise ValueError("ADVISORY_TRIAL_STORE_MISMATCH")
        if in_transaction(self._connection):
            raise RuntimeError(
                "advisory publication cannot nest inside an active transaction"
            )
        scope = {
            "organization_id": context.actor_context.organization_id,
            "workspace_id": context.actor_context.workspace_id,
        }
        saved_input = self.get_published_advisory_compiler_inputs(**scope, pin=inputs)
        boundary, competency, _brief, source_capture = (
            _resolve_current_compiler_input_values(context=context, inputs=saved_input)
        )
        had_publication = bool(self._advisory_initial_index_rows(saved_input))
        if had_publication:
            proposed = self.get_advisory_initial_outputs(**scope, inputs=inputs)
        else:
            proposed = _compile_advisory_initial_values(
                inputs=saved_input,
                boundary=boundary,
                competency=competency,
                source_capture=source_capture,
            )
        _validate_live_initial_output_values(
            inputs=saved_input,
            boundary=boundary,
            competency=competency,
            source_capture=source_capture,
            candidate=proposed[0],
            validation=proposed[1],
            plan=proposed[2],
        )
        prepared = tuple(self._prepare(value) for value in proposed)
        with self._advisory_publication_transaction():
            current_input = self.get_published_advisory_compiler_inputs(
                **scope, pin=inputs
            )
            if current_input != saved_input:
                raise ValueError("ADVISORY_COMPILER_INPUT_PAYLOAD_MISMATCH")
            boundary, competency, _brief, source_capture = (
                _resolve_current_compiler_input_values(
                    context=context, inputs=current_input
                )
            )
            if self._advisory_initial_index_rows(current_input):
                outputs = self.get_advisory_initial_outputs(**scope, inputs=inputs)
                if tuple(self._prepare(value) for value in outputs) != prepared:
                    raise ValueError("ADVISORY_INITIAL_PAYLOAD_MISMATCH")
            else:
                if had_publication:
                    raise ValueError("ADVISORY_INITIAL_INDEX_INCOMPLETE")
                for value in prepared:
                    if (
                        self._connection.execute(
                            sql_query(f"SELECT 1 FROM {_ADVISORY_PUBLICATION_INDEX} WHERE artifact_ref = ?",f'SELECT 1 FROM {_ADVISORY_PUBLICATION_INDEX} WHERE artifact_ref = %s'),
                            (value[2],),
                        ).fetchone()
                        is not None
                    ):
                        raise ValueError("ADVISORY_INITIAL_OUTPUT_ALREADY_ASSOCIATED")
                _validate_live_initial_output_values(
                    inputs=current_input,
                    boundary=boundary,
                    competency=competency,
                    source_capture=source_capture,
                    candidate=proposed[0],
                    validation=proposed[1],
                    plan=proposed[2],
                )
                self._insert_prepared(prepared)
                for specification, value in zip(
                    _ADVISORY_INITIAL_OUTPUTS, proposed, strict=True
                ):
                    self._connection.execute(
                        sql_query(f"INSERT INTO {_ADVISORY_PUBLICATION_INDEX} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",f'INSERT INTO {_ADVISORY_PUBLICATION_INDEX} VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'),
                        self._advisory_initial_index_values(
                            current_input,
                            value,
                            specification[0],
                            specification[3],
                            specification[4],
                        ),
                    )
                outputs = self.get_advisory_initial_outputs(**scope, inputs=inputs)
            _validate_live_initial_output_values(
                inputs=current_input,
                boundary=boundary,
                competency=competency,
                source_capture=source_capture,
                candidate=outputs[0],
                validation=outputs[1],
                plan=outputs[2],
            )
            _resolve_current_compiler_input_values(
                context=context, inputs=current_input
            )
            self._require_final_advisory_owner_access(context)
        return outputs

    @contextmanager
    def _guard_readonly_workspace_callback(self, label: str):
        """Keep validation and advisory supplementary verification read-only."""
        connection = self._connection
        if enabled():
            if not in_transaction(connection):raise RuntimeError(f"{label} requires an owned transaction")
            with connection.callback_guard(readonly=True):yield
            return
        if not in_transaction(connection):
            raise RuntimeError(f"{label} requires an owned transaction")
        changes_before = connection.total_changes
        denied = False

        def authorize(action, _arg1, arg2, _database, _source):
            nonlocal denied
            if action in _VALIDATION_WRITE_ACTIONS or (
                action == sqlite3.SQLITE_PRAGMA and arg2 is not None
            ):
                denied = True
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)
        connection._authorizer_replacement_attempted = False
        connection._writable_blob_attempted = False
        connection._owned_authorizer = True
        try:
            yield
            if (
                denied
                or connection._authorizer_replacement_attempted
                or connection._writable_blob_attempted
                or connection.total_changes != changes_before
            ):
                raise RuntimeError(f"{label} must be read-only")
            if not in_transaction(connection):
                raise RuntimeError(f"{label} ended the owned transaction")
        except sqlite3.DatabaseError as exc:
            if denied:
                raise RuntimeError(f"{label} must be read-only") from exc
            raise
        finally:
            connection._owned_authorizer = False
            connection.set_authorizer(None)

    @contextmanager
    def _guard_private_advisory_index(self):
        connection = self._connection
        if enabled():
            with connection.callback_guard(readonly=False):yield
            return
        denied = False

        def authorize(action, arg1, arg2, _database, _source):
            nonlocal denied
            protected_target = any(
                target in (arg1, arg2)
                for target in (
                    _ADVISORY_PUBLICATION_INDEX,
                    "ontology_build_artifacts",
                )
            )
            if (
                protected_target
                and action in _VALIDATION_WRITE_ACTIONS
                or action
                in (
                    sqlite3.SQLITE_TRANSACTION,
                    sqlite3.SQLITE_SAVEPOINT,
                    sqlite3.SQLITE_CREATE_TRIGGER,
                    sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
                )
                or action == sqlite3.SQLITE_PRAGMA
                and arg2 is not None
            ):
                denied = True
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)
        connection._authorizer_replacement_attempted = False
        connection._writable_blob_attempted = False
        connection._owned_authorizer = True
        try:
            yield
            if (
                denied
                or connection._authorizer_replacement_attempted
                or connection._writable_blob_attempted
            ):
                raise RuntimeError(
                    "generic callbacks cannot write ontology build artifacts or "
                    "private advisory publication index"
                )
        except sqlite3.DatabaseError as exc:
            if denied:
                raise RuntimeError(
                    "generic callbacks cannot write ontology build artifacts or "
                    "private advisory publication index"
                ) from exc
            raise
        finally:
            connection._owned_authorizer = False
            connection.set_authorizer(None)

    @staticmethod
    def _prepare(artifact: BaseModel) -> tuple[str, str, str, str, str, str]:
        from contracts.state_evolution import (
            EvolutionIRV1, EvolutionExecutionPlanV1, EvolutionRunV1,
            evolution_hash, evolution_canonical_json, verify_evolution_plan,
        )
        evolution_types = {EvolutionIRV1: "evolution-ir", EvolutionExecutionPlanV1: "evolution-plan", EvolutionRunV1: "evolution-run"}
        if type(artifact) in evolution_types:
            payload = artifact.model_dump(mode="json", exclude_unset=True)
            type(artifact).model_validate(payload)
            if type(artifact) is EvolutionExecutionPlanV1:
                verify_evolution_plan(payload)
                identity = payload["basisHash"]
            elif type(artifact) is EvolutionRunV1:
                identity = payload["runId"]
                if identity != evolution_hash({key: value for key, value in payload.items() if key != "runId"}):
                    raise ValueError("EVOLUTION_RESULT_HASH_MISMATCH")
                if payload["resultHash"] != evolution_hash({key: payload[key] for key in ("status", "events", "series", "finalState", "diagnostics")}):
                    raise ValueError("EVOLUTION_RESULT_HASH_MISMATCH")
            else:
                identity = evolution_hash(payload)
            return (
                payload["scope"]["organizationId"], payload["scope"]["workspaceId"],
                evolution_types[type(artifact)] + ":" + identity.split(":")[1],
                evolution_hash(payload), payload["schemaVersion"], evolution_canonical_json(payload),
            )
        payload = artifact.model_dump(mode="json")
        if type(artifact) is EffectiveRestrictionRuntimeContextV1:
            context = EffectiveRestrictionRuntimeContextV1.model_validate(payload)
            digest = content_hash(context.model_dump(mode="json", by_alias=True))
            return (
                context.organization_id, context.workspace_id,
                "effective-restriction-runtime-context:" + digest[:24],
                "sha256:" + digest, context.schema_version, canonical_json(payload),
            )
        # Revalidate copied/constructed model instances at the persistence
        # boundary so hash-binding validators cannot be bypassed by model_copy.
        schema_version = str(payload.get("schema_version") or "")
        if schema_version in _KERNEL_ARTIFACT_MODELS:
            model = _KERNEL_ARTIFACT_MODELS[schema_version]
            if type(artifact) is not model:
                raise ValueError("INVALID_STATE")
            if model.model_validate(payload).model_dump(mode="json") != payload:
                raise ValueError("HASH_MISMATCH")
        elif schema_version in LEARNING_ARTIFACT_MODELS:
            model = LEARNING_ARTIFACT_MODELS[schema_version]
            if type(artifact) is not model:
                raise ValueError("INVALID_STATE")
            if model.model_validate(payload).model_dump(mode="json") != payload:
                raise ValueError("HASH_MISMATCH")
        elif schema_version in _WORKSPACE_ARTIFACT_MODELS:
            model = _WORKSPACE_ARTIFACT_MODELS[schema_version][0]
            if type(artifact) is not model:
                raise ValueError("INVALID_STATE")
            if model.model_validate(payload).model_dump(mode="json") != payload:
                raise ValueError("HASH_MISMATCH")
        elif schema_version in _TASK909_MODELS:
            models = _TASK909_MODELS[schema_version]
            if type(artifact) not in models:
                raise ValueError("INVALID_STATE")
            validated = type(artifact).model_validate(payload)
            if validated.model_dump(mode="json") != payload:
                raise ValueError("HASH_MISMATCH")
        elif schema_version in _ADVISORY_PROPOSAL_MODELS:
            model = _ADVISORY_PROPOSAL_MODELS[schema_version]
            if type(artifact) is not model:
                raise ValueError("INVALID_STATE")
            if model.model_validate(payload).model_dump(mode="json") != payload:
                raise ValueError("HASH_MISMATCH")
        elif schema_version in _WORKSPACE_REGISTRY_SCHEMAS:
            _validate_registered_artifact(artifact)
        elif schema_version in RECEIVABLES_MODELS:
            model = RECEIVABLES_MODELS[schema_version]
            if type(artifact) is not model:
                raise ValueError("INVALID_STATE")
            model.model_validate(payload)
        elif schema_version in _IDENTITY_FIELDS:
            type(artifact).model_validate(payload)
        try:
            ref_field, hash_field = _IDENTITY_FIELDS[schema_version]
        except KeyError as exc:
            raise ValueError(
                f"unsupported ontology artifact {schema_version!r}"
            ) from exc
        artifact_ref = str(payload[ref_field])
        artifact_hash = str(payload[hash_field])
        if schema_version == "runtime_report_export_receipt.v1":
            organization_id = str(payload["scope"]["organization_id"])
            workspace_id = str(payload["scope"]["workspace_id"])
        else:
            organization_id = str(payload["organization_id"])
            workspace_id = str(payload["workspace_id"])
        payload_json = canonical_json(payload)
        return (
            organization_id,
            workspace_id,
            artifact_ref,
            artifact_hash,
            schema_version,
            payload_json,
        )

    def _publish_tender_owner_adoption(self, *, service, actor_context, proposal, adoption,
                                      expected_head_pin, request_id, recorded_at):
        """Closed producer port; atomic commitments and append-only revocation CAS."""
        from services.tender_owner_adoption_service import TenderOwnerAdoptionService
        from contracts.tender_owner_adoption import TenderOwnerAdoptionV1, TenderOwnerRevocationV1, validate_frozen
        if type(service) is not TenderOwnerAdoptionService or service.store is not self:
            raise TypeError('owner adoption requires its exact producer')
        scope = service._scope(actor_context, owner=True)
        operation = 'REVOKE_OWNER_ADOPTION' if adoption is not None else 'ADOPT_OWNER_ADVISORY'
        request_hash = 'sha256:' + content_hash(dict(proposal=proposal.pin().model_dump(mode='json'),
            adoption=adoption.pin().model_dump(mode='json') if adoption else None,
            expected_head=expected_head_pin.model_dump(mode='json') if expected_head_pin else None))
        operation_args = dict(**scope, actor_ref=actor_context.actor_id, operation=operation,
            request_id=request_id, request_hash=request_hash)
        kind = 'tender_owner_revocation' if adoption else 'tender_owner_adoption'
        model = TenderOwnerRevocationV1 if adoption else TenderOwnerAdoptionV1
        begin_write(self._connection)
        try:
            prior = self.get_operation_result(**operation_args, artifact_kind=kind)
            if prior is not None:
                result = service._get(scope, prior, model)
            elif adoption is None:
                service._verify_proposal(actor_context, proposal)
                # Same exact bundle remains one commitment, independent of request retries.
                existing = [a for a in service._history(scope, TenderOwnerAdoptionV1) if a.proposal == proposal.pin()]
                if existing:
                    if len(existing) != 1 or service.head(actor_context=actor_context, adoption=existing[0]) != existing[0].pin():
                        raise ValueError('OWNER_ADOPTION_BUNDLE_ALREADY_REVOKED')
                    result = service.read(actor_context=actor_context, adoption_pin=existing[0].pin())
                else:
                    source = service._decision(actor_context, proposal, 'SOURCE_ADOPTION', proposal.canonical_source_ir, recorded_at)
                    interpretation = service._decision(actor_context, proposal, 'INTERPRETATION_ADOPTION', proposal.candidate, recorded_at)
                    result = TenderOwnerAdoptionV1.create(**scope, proposal=proposal.pin(), source_decision=source.pin(),
                        interpretation_decision=interpretation.pin(), actor_id=actor_context.actor_id, adopted_at=recorded_at)
                    validate_frozen(adoption=result,proposal=proposal,source_decision=source,interpretation_decision=interpretation)
                    self._insert_prepared(tuple(self._prepare(v) for v in (source, interpretation, result)))
            else:
                current = service.read(actor_context=actor_context, adoption_pin=adoption.pin())
                if current != adoption or service.head(actor_context=actor_context, adoption=adoption) != expected_head_pin or expected_head_pin != adoption.pin():
                    raise ValueError('STALE_OWNER_ADOPTION_HEAD')
                if recorded_at < adoption.adopted_at:
                    raise ValueError('OWNER_REVOCATION_PRECEDES_ADOPTION')
                decision = service._decision(actor_context, proposal, 'REVOKE_OWNER_ADOPTION', adoption.pin(), recorded_at)
                result = TenderOwnerRevocationV1.create(**scope, adoption=adoption.pin(), predecessor=expected_head_pin,
                    decision=decision.pin(), actor_id=actor_context.actor_id, revoked_at=recorded_at)
                self._insert_prepared(tuple(self._prepare(v) for v in (decision,result)))
            if prior is None:
                self._connection.execute(sql_query('INSERT INTO workspace_operation_request_index VALUES (?,?,?,?,?,?,?,?)','INSERT INTO workspace_operation_request_index VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'),
                    (scope['organization_id'],scope['workspace_id'],actor_context.actor_id,operation,request_id,request_hash,
                     result.artifact_ref,result.artifact_hash))
            service._scope(actor_context, owner=True)
            self._connection.commit()
            return result
        except BaseException:
            self._connection.rollback()
            raise

    def put(self, artifact: BaseModel) -> None:
        self.put_many((artifact,))

    def put_many(self, artifacts: tuple[BaseModel, ...]) -> None:
        """Append a related artifact set in one SQLite transaction."""

        prepared = tuple(self._prepare(artifact) for artifact in artifacts)
        _reject_protected_publication(prepared)
        # Acquire the SQLite write reservation before checking version bindings.
        # A deferred transaction would let two connections both observe no
        # binding and then insert different content for the same ontology version.
        begin_write(self._connection)
        try:
            self._insert_prepared(prepared)
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    def hr_interpretation_head(self, value):
        row = self._connection.execute(
            sql_query("SELECT artifact_ref,artifact_hash FROM hr_interpretation_heads WHERE "
            "organization_id=? AND workspace_id=? AND actor_id=? AND session_id=? AND model_ref=? AND resume_ref=? AND requirement_id=?",'SELECT artifact_ref,artifact_hash FROM hr_interpretation_heads WHERE organization_id=%s AND workspace_id=%s AND actor_id=%s AND session_id=%s AND model_ref=%s AND resume_ref=%s AND requirement_id=%s'),
            (value.organization_id, value.workspace_id, value.actor_id, value.session_id,
             value.model_pin.artifact_ref, value.resume_pin.artifact_ref, value.requirement_id),
        ).fetchone()
        return None if row is None else ArtifactPinV1(artifact_kind="hr_interpretation", artifact_ref=row[0], artifact_hash=row[1])

    def put_hr_interpretation(self, value, verify_before_commit):
        """Append a source-backed review and CAS its task-local active head."""
        key = (value.organization_id, value.workspace_id, value.actor_id, value.session_id,
               value.model_pin.artifact_ref, value.resume_pin.artifact_ref, value.requirement_id)
        def validate(connection):
            if self.hr_interpretation_head(value) != value.predecessor_pin:
                raise ValueError("STALE_INTERPRETATION_HEAD")
            verify_before_commit()
            return True
        def indexes(connection):
            connection.execute(sql_query("INSERT INTO hr_interpretation_heads VALUES (?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(organization_id,workspace_id,actor_id,session_id,model_ref,resume_ref,requirement_id) "
                "DO UPDATE SET artifact_ref=excluded.artifact_ref,artifact_hash=excluded.artifact_hash",'INSERT INTO hr_interpretation_heads VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(organization_id,workspace_id,actor_id,session_id,model_ref,resume_ref,requirement_id) DO UPDATE SET artifact_ref=excluded.artifact_ref,artifact_hash=excluded.artifact_hash'),
                (*key, value.artifact_ref, value.artifact_hash))
        self.atomic_write(artifacts=(value,), validate_before_insert=validate,
                          mutate_indexes=indexes, verify_before_commit=verify_before_commit)

    @staticmethod
    def _hr_owner_alive(row):
        import os
        if row["owner_pid"] == os.getpid():
            with _HR_ATTEMPT_LOCK:
                return row["attempt_token"] in _HR_ACTIVE_ATTEMPTS
        from storage.graph_intent_brief_store import GraphIntentPublicationTransaction
        return GraphIntentPublicationTransaction._pid_is_alive(row["owner_pid"])

    def _hr_row(self, scope, request_id):
        row = self._connection.execute(
            sql_query("SELECT request_hash,command_json,status,owner_pid,attempt_token,checkpoint_json,checkpoint_hash,"
            "result_ref,result_hash,stop_disposition,started_at,updated_at FROM hr_screening_operations "
            "WHERE organization_id=? AND workspace_id=? AND actor_id=? AND session_id=? AND request_id=?",'SELECT request_hash,command_json,status,owner_pid,attempt_token,checkpoint_json,checkpoint_hash,result_ref,result_hash,stop_disposition,started_at,updated_at FROM hr_screening_operations WHERE organization_id=%s AND workspace_id=%s AND actor_id=%s AND session_id=%s AND request_id=%s'),
            (*scope, request_id),
        ).fetchone()
        if row is None:
            raise ValueError("OPERATION_NOT_FOUND")
        fields = ("request_hash", "command_json", "status", "owner_pid", "attempt_token", "checkpoint_json", "checkpoint_hash",
                  "result_ref", "result_hash", "stop_disposition", "started_at", "updated_at")
        result = dict(zip(fields, row))
        result["command"] = json.loads(result.pop("command_json"))
        result["checkpoint"] = json.loads(result.pop("checkpoint_json"))
        if result["request_hash"] != "sha256:" + content_hash(result["command"]):
            raise ValueError("REQUEST_CONFLICT")
        if result["checkpoint_hash"] != "sha256:" + content_hash(result["checkpoint"]):
            raise ValueError("CHECKPOINT_INTEGRITY_UNPROVABLE")
        if [x["review_pin"] for x in result["checkpoint"]] != result["command"]["resume_pins"][:len(result["checkpoint"])]:
            raise ValueError("INVALID_OPERATION_PROGRESS")
        result["request_id"] = request_id
        return result

    @staticmethod
    def _hr_projection(row):
        from contracts.hr_workbench import HrOperationV1
        return HrOperationV1.model_validate(dict(
            request_id=row["request_id"], request_hash=row["request_hash"], command=row["command"], status=row["status"],
            processed_candidates=len(row["checkpoint"]),
            result_pin=None if not row["result_ref"] else dict(artifact_kind="hr_screening_run", artifact_ref=row["result_ref"], artifact_hash=row["result_hash"]),
            stop_disposition=row["stop_disposition"], started_at=row["started_at"], updated_at=row["updated_at"]))

    def hr_operation(self, *, scope, request_id):
        row = self._hr_row(scope, request_id)
        if row["status"] in ("RUNNING", "STOP_REQUESTED") and not self._hr_owner_alive(row):
            begin_write(self._connection)
            try:
                row = self._hr_row(scope, request_id)
                if row["status"] in ("RUNNING", "STOP_REQUESTED") and not self._hr_owner_alive(row):
                    self._connection.execute(sql_query("UPDATE hr_screening_operations SET status='INTERRUPTED',updated_at=? "
                        "WHERE organization_id=? AND workspace_id=? AND actor_id=? AND session_id=? AND request_id=?","UPDATE hr_screening_operations SET status='INTERRUPTED',updated_at=%s WHERE organization_id=%s AND workspace_id=%s AND actor_id=%s AND session_id=%s AND request_id=%s"),
                        (datetime.now(timezone.utc).isoformat(), *scope, request_id))
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
            row = self._hr_row(scope, request_id)
        return self._hr_projection(row)

    def hr_claim(self, *, scope, request_id, request_hash, command):
        import os, uuid
        token = uuid.uuid4().hex
        with _HR_ATTEMPT_LOCK:
            _HR_ACTIVE_ATTEMPTS.add(token)
        claimed = False
        try:
            begin_write(self._connection)
            try:
                try:
                    row = self._hr_row(scope, request_id)
                except ValueError as exc:
                    if str(exc) != "OPERATION_NOT_FOUND": raise
                    now = datetime.now(timezone.utc).isoformat()
                    self._connection.execute(sql_query("INSERT INTO hr_screening_operations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",'INSERT INTO hr_screening_operations VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'),
                        (*scope, request_id, request_hash, canonical_json(command), "RUNNING", os.getpid(), token,
                         "[]", "sha256:" + content_hash([]), None, None, None, now, now))
                    claimed = True
                else:
                    if row["request_hash"] != request_hash:
                        raise ValueError("REQUEST_CONFLICT")
                    lost = row["status"] in ("RUNNING", "STOP_REQUESTED") and not self._hr_owner_alive(row)
                    if row["status"] == "INTERRUPTED" or lost:
                        # A retained stop is not forgotten when the worker disappears.
                        status = "STOP_REQUESTED" if row["stop_disposition"] == "ACKNOWLEDGED" else "RUNNING"
                        self._connection.execute(sql_query("UPDATE hr_screening_operations SET status=?,owner_pid=?,attempt_token=?,updated_at=? "
                            "WHERE organization_id=? AND workspace_id=? AND actor_id=? AND session_id=? AND request_id=?",'UPDATE hr_screening_operations SET status=%s,owner_pid=%s,attempt_token=%s,updated_at=%s WHERE organization_id=%s AND workspace_id=%s AND actor_id=%s AND session_id=%s AND request_id=%s'),
                            (status, os.getpid(), token, datetime.now(timezone.utc).isoformat(), *scope, request_id))
                        claimed = True
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
            row = self._hr_row(scope, request_id)
            return claimed, token, row
        finally:
            if not claimed:
                self.hr_release(token)

    @staticmethod
    def hr_release(token):
        with _HR_ATTEMPT_LOCK:
            _HR_ACTIVE_ATTEMPTS.discard(token)

    def hr_stop(self, *, scope, request_id, request_hash):
        begin_write(self._connection)
        try:
            row = self._hr_row(scope, request_id)
            if row["request_hash"] != request_hash:
                raise ValueError("REQUEST_CONFLICT")
            if row["result_ref"]:
                disposition = "TOO_LATE" if row["status"] == "COMPLETED" else "FINAL"
                self._connection.execute(sql_query("UPDATE hr_screening_operations SET stop_disposition=? "
                    "WHERE organization_id=? AND workspace_id=? AND actor_id=? AND session_id=? AND request_id=?",'UPDATE hr_screening_operations SET stop_disposition=%s WHERE organization_id=%s AND workspace_id=%s AND actor_id=%s AND session_id=%s AND request_id=%s'),
                    (disposition, *scope, request_id))
            else:
                self._connection.execute(sql_query("UPDATE hr_screening_operations SET status='STOP_REQUESTED',stop_disposition='ACKNOWLEDGED',updated_at=? "
                    "WHERE organization_id=? AND workspace_id=? AND actor_id=? AND session_id=? AND request_id=?","UPDATE hr_screening_operations SET status='STOP_REQUESTED',stop_disposition='ACKNOWLEDGED',updated_at=%s WHERE organization_id=%s AND workspace_id=%s AND actor_id=%s AND session_id=%s AND request_id=%s"),
                    (datetime.now(timezone.utc).isoformat(), *scope, request_id))
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise
        return self._hr_projection(self._hr_row(scope, request_id))

    def hr_checkpoint(self, *, scope, request_id, token, candidate=None):
        begin_write(self._connection)
        try:
            row = self._hr_row(scope, request_id)
            if row["attempt_token"] != token or row["status"] not in ("RUNNING", "STOP_REQUESTED"):
                raise ValueError("OPERATION_CLAIM_LOST")
            if candidate is not None:
                checkpoint = [*row["checkpoint"], candidate]
                if [x["review_pin"] for x in checkpoint] != row["command"]["resume_pins"][:len(checkpoint)]:
                    raise ValueError("INVALID_OPERATION_PROGRESS")
                self._connection.execute(sql_query("UPDATE hr_screening_operations SET checkpoint_json=?,checkpoint_hash=?,updated_at=? "
                    "WHERE organization_id=? AND workspace_id=? AND actor_id=? AND session_id=? AND request_id=?",'UPDATE hr_screening_operations SET checkpoint_json=%s,checkpoint_hash=%s,updated_at=%s WHERE organization_id=%s AND workspace_id=%s AND actor_id=%s AND session_id=%s AND request_id=%s'),
                    (canonical_json(checkpoint), "sha256:" + content_hash(checkpoint), datetime.now(timezone.utc).isoformat(), *scope, request_id))
            self._connection.commit()
            return row["status"] != "STOP_REQUESTED"
        except BaseException:
            self._connection.rollback()
            raise

    def hr_finish(self, *, scope, request_id, token, make_run, verify_before_commit):
        """Arbitrate a stop and publish the one immutable HR terminal receipt."""
        from contracts.hr_workbench import HrScreeningRunV2
        begin_write(self._connection)
        try:
            row = self._hr_row(scope, request_id)
            if row["attempt_token"] != token or row["status"] not in ("RUNNING", "STOP_REQUESTED"):
                raise ValueError("OPERATION_CLAIM_LOST")
            value = make_run(row["status"] == "STOP_REQUESTED", row["checkpoint"])
            if type(value) is not HrScreeningRunV2 or value.request_id != request_id or value.request_hash != row["request_hash"]:
                raise ValueError("OPERATION_RESULT_MISMATCH")
            if tuple(getattr(value, k) for k in ("organization_id", "workspace_id", "actor_id", "session_id")) != scope:
                raise ValueError("OPERATION_SCOPE_MISMATCH")
            cmd = row["command"]
            if (value.job_ref != cmd["job_ref"] or value.model_pin.model_dump(mode="json") != cmd["model_pin"]
                or [p.model_dump(mode="json") for p in value.resume_pins] != cmd["resume_pins"]
                or [p.model_dump(mode="json") for p in value.interpretation_pins] != cmd["interpretation_pins"] or value.as_of != cmd["as_of"]):
                raise ValueError("OPERATION_RESULT_MISMATCH")
            if row["status"] == "STOP_REQUESTED" and value.status != "CANCELLED":
                raise ValueError("STOP_PUBLICATION_CONFLICT")
            if value.status == "COMPLETED":
                verify_before_commit()
            prepared = (self._prepare(value),)
            _reject_protected_publication(prepared)
            self._insert_prepared(prepared)
            self._connection.execute(sql_query("INSERT INTO workspace_operation_request_index VALUES (?,?,?,?,?,?,?,?)",'INSERT INTO workspace_operation_request_index VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'),
                (scope[0], scope[1], scope[2], "HR_SCREEN", request_id, row["request_hash"], value.artifact_ref, value.artifact_hash))
            self._connection.execute(sql_query("UPDATE hr_screening_operations SET status=?,result_ref=?,result_hash=?,stop_disposition=?,updated_at=? "
                "WHERE organization_id=? AND workspace_id=? AND actor_id=? AND session_id=? AND request_id=?",'UPDATE hr_screening_operations SET status=%s,result_ref=%s,result_hash=%s,stop_disposition=%s,updated_at=%s WHERE organization_id=%s AND workspace_id=%s AND actor_id=%s AND session_id=%s AND request_id=%s'),
                (value.status, value.artifact_ref, value.artifact_hash, "FINAL" if value.status == "CANCELLED" else row["stop_disposition"],
                 datetime.now(timezone.utc).isoformat(), *scope, request_id))
            self._connection.commit()
            return value.pin()
        except BaseException:
            self._connection.rollback()
            raise

    def hr_operations(self, *, scope):
        requests = self._connection.execute(sql_query("SELECT request_id FROM hr_screening_operations "
            "WHERE organization_id=? AND workspace_id=? AND actor_id=? AND session_id=? ORDER BY started_at,request_id",'SELECT request_id FROM hr_screening_operations WHERE organization_id=%s AND workspace_id=%s AND actor_id=%s AND session_id=%s ORDER BY started_at,request_id'), scope).fetchall()
        return [self.hr_operation(scope=scope, request_id=r[0]) for r in requests]

    def requirements_head(
        self, *, organization_id, workspace_id, journey_id, source_role
    ):
        row = self._connection.execute(
            sql_query(
                "SELECT receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?",
                "SELECT receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s",
            ),
            (
                organization_id,
                workspace_id,
                "system:requirements",
                "REQUIREMENTS_HEAD",
                journey_id + ":" + source_role,
            ),
        ).fetchone()
        if row is None:
            return None
        review = self.get(
            organization_id=organization_id,
            workspace_id=workspace_id,
            artifact_ref=row[0],
            contract_type=RequirementsReviewV1,
        )
        if review is None or review.review_hash != row[1]:
            raise ValueError("REQUIREMENTS_HEAD_INTEGRITY")
        return review

    def publish_requirements_review(
        self, *, review, request_id, request_hash, verify_before_commit
    ):
        """Append result and command receipt with a per-task/source-role CAS.

        Uses the existing operation index on both SQLite and PostgreSQL.
        """
        review = RequirementsReviewV1.model_validate(review.model_dump(mode="json"))
        scope = dict(
            organization_id=review.organization_id, workspace_id=review.workspace_id
        )
        winner = review.pin()
        command_scope = (
            review.organization_id,
            review.workspace_id,
            review.actor_ref,
            "REQUIREMENTS_COMMAND",
            request_id,
        )
        head_scope = (
            review.organization_id,
            review.workspace_id,
            "system:requirements",
            "REQUIREMENTS_HEAD",
            review.journey_id + ":" + review.source_role,
        )

        def validate(connection):
            nonlocal winner
            prior = self.get_operation_result(
                **scope,
                actor_ref=review.actor_ref,
                operation="REQUIREMENTS_COMMAND",
                request_id=request_id,
                request_hash=request_hash,
                artifact_kind="requirements_review",
            )
            if prior is not None:
                winner = prior
                verify_before_commit(replay=True)
                return False
            current = self.requirements_head(
                **scope, journey_id=review.journey_id, source_role=review.source_role
            )
            if (current.pin() if current else None) != review.predecessor_pin:
                raise ValueError("STALE_REQUIREMENTS_REVIEW")
            if (
                current is not None
                and current.journey_revision >= review.journey_revision
            ):
                raise ValueError("REQUIREMENTS_ADOPTION_REQUIRED")
            verify_before_commit(replay=False)
            return True

        def indexes(connection):
            connection.execute(
                sql_query(
                    "INSERT INTO workspace_operation_request_index VALUES (?,?,?,?,?,?,?,?)",
                    "INSERT INTO workspace_operation_request_index VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                ),
                (*command_scope, request_hash, review.review_ref, review.review_hash),
            )
            if review.predecessor_pin is None:
                connection.execute(
                    sql_query(
                        "INSERT INTO workspace_operation_request_index VALUES (?,?,?,?,?,?,?,?)",
                        "INSERT INTO workspace_operation_request_index VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    ),
                    (*head_scope, request_hash, review.review_ref, review.review_hash),
                )
            else:
                updated = connection.execute(
                    sql_query(
                        "UPDATE workspace_operation_request_index SET request_hash=?,receipt_ref=?,receipt_hash=? WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=? AND receipt_ref=? AND receipt_hash=?",
                        "UPDATE workspace_operation_request_index SET request_hash=%s,receipt_ref=%s,receipt_hash=%s WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s AND receipt_ref=%s AND receipt_hash=%s",
                    ),
                    (
                        request_hash,
                        review.review_ref,
                        review.review_hash,
                        *head_scope,
                        review.predecessor_pin.artifact_ref,
                        review.predecessor_pin.artifact_hash,
                    ),
                )
                if updated.rowcount != 1:
                    raise ValueError("STALE_REQUIREMENTS_REVIEW")

        try:
            self.atomic_write(
                artifacts=(review,),
                validate_before_insert=validate,
                mutate_indexes=indexes,
                verify_before_commit=lambda: verify_before_commit(replay=False),
            )
        except integrity_errors() as exc:
            replay = self.get_operation_result(
                **scope,
                actor_ref=review.actor_ref,
                operation="REQUIREMENTS_COMMAND",
                request_id=request_id,
                request_hash=request_hash,
                artifact_kind="requirements_review",
            )
            if replay is None:
                raise ValueError("STALE_REQUIREMENTS_REVIEW") from exc
            verify_before_commit(replay=True)
            return replay
        return winner

    def get_operation_result(
        self, *, organization_id: str, workspace_id: str, actor_ref: str,
        operation: str, request_id: str, request_hash: str, artifact_kind: str,
    ) -> ArtifactPinV1 | None:
        """Recover a published operation before invoking its producer again."""
        row = self._connection.execute(
            sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index "
            "WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?",'SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s'),
            (organization_id, workspace_id, actor_ref, operation, request_id),
        ).fetchone()
        if row is None:
            return None
        if row[0] != request_hash:
            raise ValueError("REQUEST_CONFLICT")
        raw = self.get_raw(
            organization_id=organization_id, workspace_id=workspace_id, artifact_ref=row[1],
        )
        if raw is None or raw["artifact_hash"] != row[2]:
            raise ValueError("INTEGRITY_UNPROVABLE")
        return ArtifactPinV1(artifact_kind=artifact_kind, artifact_ref=row[1], artifact_hash=row[2])

    def publish_tender_review(self, *, artifact, command, verify_before_commit):
        """Closed documentary publication: exact aliases and actor/subject CAS only."""
        from contracts.tender_review import TenderReviewRecordV1, TenderReviewReportV1, TenderReviewReportV2
        if type(artifact) not in (TenderReviewRecordV1, TenderReviewReportV1, TenderReviewReportV2):
            raise TypeError("documentary artifact required")
        # Revalidate copies before any query/index mutation.
        artifact = type(artifact).model_validate(artifact.model_dump(mode="json"))
        scope = (artifact.organization_id, artifact.workspace_id, artifact.recorded_by)
        operation = artifact.KIND.upper()
        request_hash = "sha256:" + content_hash(command.model_dump(mode="json"))
        aliases = ("request:" + command.request_id, "key:" + command.idempotency_key)
        winner = artifact.pin()
        head_key = None
        if type(artifact) is TenderReviewRecordV1:
            head_key = (artifact.organization_id, artifact.workspace_id, artifact.session_ref,
                        artifact.target_artifact_pin.artifact_ref, artifact.target_artifact_pin.artifact_hash,
                        content_hash(artifact.subject.model_dump(mode="json")), artifact.recorded_by)

        def validate(connection):
            nonlocal winner
            matches = []
            for alias in aliases:
                row = connection.execute(
                    sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index "
                    "WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?",'SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s'),
                    (*scope, operation, alias),
                ).fetchone()
                if row is not None:
                    if row[0] != request_hash:
                        raise ValueError("REQUEST_CONFLICT")
                    matches.append((row[1], row[2]))
            if matches:
                if len(set(matches)) != 1:
                    raise ValueError("REQUEST_CONFLICT")
                winner = ArtifactPinV1(artifact_kind=artifact.KIND, artifact_ref=matches[0][0], artifact_hash=matches[0][1])
                # Replay is still an authorized read; missing/tampered artifacts fail clearly.
                saved = self.get(organization_id=scope[0], workspace_id=scope[1], artifact_ref=winner.artifact_ref, contract_type=type(artifact))
                if saved is None or saved.pin() != winner:
                    raise ValueError("INTEGRITY_UNPROVABLE")
                verify_before_commit(replay=True)
                return False
            verify_before_commit(replay=False)
            if head_key is not None:
                row = connection.execute(
                    sql_query("SELECT record_ref,record_hash FROM tender_review_heads WHERE organization_id=? AND workspace_id=? "
                    "AND session_ref=? AND target_ref=? AND target_hash=? AND subject_hash=? AND actor_ref=?",'SELECT record_ref,record_hash FROM tender_review_heads WHERE organization_id=%s AND workspace_id=%s AND session_ref=%s AND target_ref=%s AND target_hash=%s AND subject_hash=%s AND actor_ref=%s'), head_key,
                ).fetchone()
                expected = command.expected_head_pin
                actual = None if row is None else ArtifactPinV1(artifact_kind="tender_review_record", artifact_ref=row[0], artifact_hash=row[1])
                if actual != expected or artifact.predecessor_record_pin != actual:
                    raise ValueError("STALE_REVIEW_HEAD")
            return True

        def indexes(connection):
            for alias in aliases:
                connection.execute(
                    sql_query("INSERT INTO workspace_operation_request_index VALUES (?,?,?,?,?,?,?,?)",'INSERT INTO workspace_operation_request_index VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'),
                    (*scope, operation, alias, request_hash, artifact.pin().artifact_ref, artifact.pin().artifact_hash),
                )
            if head_key is not None:
                connection.execute(
                    sql_query("INSERT INTO tender_review_heads VALUES (?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(organization_id,workspace_id,session_ref,target_ref,target_hash,subject_hash,actor_ref) "
                    "DO UPDATE SET record_ref=excluded.record_ref,record_hash=excluded.record_hash",'INSERT INTO tender_review_heads VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(organization_id,workspace_id,session_ref,target_ref,target_hash,subject_hash,actor_ref) DO UPDATE SET record_ref=excluded.record_ref,record_hash=excluded.record_hash'),
                    (*head_key, artifact.record_ref, artifact.record_hash),
                )

        self.atomic_write(artifacts=(artifact,), validate_before_insert=validate, mutate_indexes=indexes,
                          verify_before_commit=lambda: verify_before_commit(replay=False))
        return winner

    def put_many_with_operation_result(
        self,
        *,
        artifacts: tuple[BaseModel, ...],
        organization_id: str,
        workspace_id: str,
        actor_ref: str,
        operation: str,
        request_id: str,
        request_hash: str,
        result_pin: ArtifactPinV1,
        verify_before_commit=None,
    ) -> ArtifactPinV1:
        """Atomically publish artifacts and bind one idempotent operation result.

        The request index is workflow state only. It does not create another
        semantic artifact family; exact retries recover the originally committed
        result pin before any candidate artifacts are inserted.
        """

        winner = result_pin

        def validate_before_insert(connection):
            nonlocal winner
            row = connection.execute(
                sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index "
                "WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?",'SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s'),
                (organization_id, workspace_id, actor_ref, operation, request_id),
            ).fetchone()
            if row is None:
                return True
            if row[0] != request_hash:
                raise ValueError("REQUEST_CONFLICT")
            stored = connection.execute(
                sql_query("SELECT artifact_hash FROM ontology_build_artifacts WHERE artifact_ref=?",'SELECT artifact_hash FROM ontology_build_artifacts WHERE artifact_ref=%s'),
                (row[1],),
            ).fetchone()
            if stored is None or stored[0] != row[2]:
                raise ValueError("INTEGRITY_UNPROVABLE")
            winner = ArtifactPinV1(
                artifact_kind=result_pin.artifact_kind,
                artifact_ref=row[1],
                artifact_hash=row[2],
            )
            return False

        def mutate_indexes(connection):
            connection.execute(
                sql_query("INSERT INTO workspace_operation_request_index "
                "(organization_id,workspace_id,actor_ref,operation,request_id,request_hash,receipt_ref,receipt_hash) "
                "VALUES (?,?,?,?,?,?,?,?)",'INSERT INTO workspace_operation_request_index (organization_id,workspace_id,actor_ref,operation,request_id,request_hash,receipt_ref,receipt_hash) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'),
                (
                    organization_id,
                    workspace_id,
                    actor_ref,
                    operation,
                    request_id,
                    request_hash,
                    result_pin.artifact_ref,
                    result_pin.artifact_hash,
                ),
            )

        self._atomic_write(
            artifacts=artifacts,
            validate_before_insert=validate_before_insert,
            mutate_indexes=mutate_indexes,
            verify_before_commit=verify_before_commit,
        )
        return winner

    def focus_workbench_evidence(self, *, service, ctx, cursor_store, command, snapshot_pin, artifact_pin, preparation_pin):
        """Commit validated focus, cursor, acknowledgement and request aliases."""
        from services.workbench_context_resolver import WorkbenchContextResolver

        if type(service) is not WorkbenchContextResolver or service._store is not self:
            raise TypeError("focus requires its bound workspace owner")
        if in_transaction(self._connection):
            raise RuntimeError("focus transaction cannot nest")
        begin_write(self._connection)
        try:
            result = service._focus_design_evidence_on_connection(
                self._connection, ctx=ctx, cursor_store=cursor_store, command=command,
                snapshot_pin=snapshot_pin, artifact_pin=artifact_pin, preparation_pin=preparation_pin,
            )
            cursor_store._scope(ctx)
            self._connection.commit()
            return result
        except BaseException:
            self._connection.rollback()
            raise

    def bind_workspace_artifact(self, *, service, ctx, command):
        """Commit binding, optional selection, cursor, and receipt together."""
        from services.lifecycle_artifact_binding_service import LifecycleArtifactBindingService

        if type(service) is not LifecycleArtifactBindingService or service._build is not self:
            raise TypeError("binding requires its bound workspace owner")
        if in_transaction(self._connection):
            raise RuntimeError("binding transaction cannot nest")
        service._artifacts._access(ctx)
        begin_write(self._connection)
        try:
            result = service._bind_on_connection(self._connection, ctx, command)
            service._artifacts.resolve(ctx=ctx, artifact_pin=command.artifact_pin)
            self._connection.commit()
            return result
        except BaseException:
            self._connection.rollback()
            raise

    def workspace_artifact_preparation_phase(self, *, service, ctx, command, phase):
        """Bound preparation phases; adapter execution never enters this port."""
        from services.workspace_artifact_preparation_service import WorkspaceArtifactPreparationService

        if type(service) is not WorkspaceArtifactPreparationService or service._build is not self:
            raise TypeError("preparation requires its bound owner")
        if phase not in ("CLAIM", "FINALIZE") or in_transaction(self._connection):
            raise RuntimeError("invalid preparation transaction boundary")
        service._artifacts._access(ctx)
        begin_write(self._connection)
        try:
            action = service._claim_on_connection if phase == "CLAIM" else service._finalize_on_connection
            result = action(self._connection, ctx, command)
            service._artifacts._access(ctx)
            self._connection.commit()
            return result
        except BaseException:
            self._connection.rollback()
            raise

    def select_workspace_artifact(self, *, service, ctx, command):
        """Commit selection, cursor and replay receipt through their bound owner."""
        from services.workspace_artifact_service import WorkspaceArtifactService

        if type(service) is not WorkspaceArtifactService or service._store is not self:
            raise TypeError("workspace artifact selection requires its bound owner")
        if in_transaction(self._connection):
            raise RuntimeError("workspace artifact selection cannot nest")
        service._access(ctx)
        begin_write(self._connection)
        try:
            receipt = service._select_on_connection(self._connection, ctx, command)
            service._access(ctx)
            self._connection.commit()
            return receipt
        except BaseException:
            self._connection.rollback()
            raise

    def initialize_workspace_artifact_cursor(self, *, service, ctx) -> str:
        """Dedicated owner port; generic mutation callbacks remain index-only."""
        from services.workspace_artifact_service import WorkspaceArtifactService

        if type(service) is not WorkspaceArtifactService or service._store is not self:
            raise TypeError("workspace artifact initialization requires its bound owner")
        if in_transaction(self._connection):
            raise RuntimeError("workspace artifact initialization cannot nest")
        service._access(ctx)
        begin_write(self._connection)
        try:
            cursor_ref = service._initialize_cursor_on_connection(self._connection, ctx)
            service._access(ctx)
            self._connection.commit()
            return cursor_ref
        except BaseException:
            self._connection.rollback()
            raise

    def publish_candidate_review(self, *, service, ctx, cursors, **inputs):
        """Owner-bound review/result currency and request publication, including retries."""
        from services.candidate_semantic_review import CandidateSemanticReviewService
        from storage.workbench_projection_cursor_store import WorkbenchProjectionCursorStore
        if type(service) is not CandidateSemanticReviewService or service._store is not self or type(cursors) is not WorkbenchProjectionCursorStore or cursors._store is not self:
            raise TypeError("review requires its real bound workspace owners")
        if in_transaction(self._connection):
            raise RuntimeError("review publication cannot nest")
        begin_write(self._connection)
        try:
            result = service._publish_on_connection(ctx=ctx, cursors=cursors, **inputs)
            self._connection.commit()
            return result
        except BaseException:
            self._connection.rollback()
            raise

    def atomic_write(
        self,
        *,
        artifacts: tuple[BaseModel, ...],
        validate_before_insert: Callable[[sqlite3.Connection], bool],
        mutate_indexes: Callable[[sqlite3.Connection], None],
        verify_before_commit: Callable[[], None] | None = None,
    ) -> None:
        """Validate and publish one bounded workspace operation atomically."""

        self._atomic_write(
            artifacts=artifacts,
            validate_before_insert=validate_before_insert,
            mutate_indexes=mutate_indexes,
            verify_before_commit=verify_before_commit,
        )

    def _publish_advisory_context(
        self,
        *,
        actor_context,
        command,
        formation,
        receipt,
        verify_before_commit,
    ):
        from contracts.runtime_context_reconstruction import AdvisoryContextCommandV1
        from services.runtime_context_formation_service import (
            _verify_advisory_triple,
            _require_bound_advisory_access,
            _revalidate,
        )
        from storage.runtime_working_context_store import RuntimeTaskContextStore

        command = _revalidate(AdvisoryContextCommandV1, command)
        _require_bound_advisory_access(
            build_store=self,
            actor_context=actor_context,
            session_ref=command.session_ref,
            pins=tuple(
                pin
                for pair in command.request.outcomes
                for pin in (pair.episode, pair.assessment)
            ),
        )
        triple = _verify_advisory_triple(
            actor_context=actor_context,
            command=command,
            formation=formation,
            receipt=receipt,
        )
        command, formation, receipt = triple
        facade = RuntimeTaskContextStore(self)
        winner = receipt
        winner_formation = formation

        def validate_before_insert(connection):
            nonlocal winner, winner_formation
            if connection is not self._connection:
                raise ValueError("INVALID_STATE")
            saved = facade.get_advisory_receipt_for_request(
                actor_context=actor_context, command=command
            )
            if saved is None:
                return True
            winner = saved
            winner_formation = facade.get_advisory_formation(
                actor_context=actor_context, pin=saved.formation
            )
            return False

        def mutate_indexes(connection):
            connection.execute(
                sql_query("INSERT INTO workspace_operation_request_index "
                "(organization_id, workspace_id, actor_ref, operation, request_id, request_hash, receipt_ref, receipt_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",'INSERT INTO workspace_operation_request_index (organization_id, workspace_id, actor_ref, operation, request_id, request_hash, receipt_ref, receipt_hash) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'),
                (
                    command.organization_id,
                    command.workspace_id,
                    command.actor_id,
                    command.operation,
                    command.request.request_id,
                    command.command_hash,
                    receipt.receipt_ref,
                    receipt.receipt_hash,
                ),
            )

        def current_access():
            _require_bound_advisory_access(
                build_store=self,
                actor_context=actor_context,
                session_ref=command.session_ref,
                pins=tuple(source.source for source in winner_formation.sources),
            )
            verify_before_commit()
            _require_bound_advisory_access(
                build_store=self,
                actor_context=actor_context,
                session_ref=command.session_ref,
                pins=tuple(source.source for source in winner_formation.sources),
            )

        # Full retained verification/current-source reads can exceed SQLite's
        # five-second default under two contending authentic first writers.
        # Bound this operation's wait without changing generic UoW behavior.
        previous_timeout = None if enabled() else self._connection.execute("PRAGMA busy_timeout").fetchone()[0]
        if not enabled():self._connection.execute("PRAGMA busy_timeout=60000")
        try:
            self._atomic_write(
                artifacts=triple,
                validate_before_insert=validate_before_insert,
                mutate_indexes=mutate_indexes,
                verify_before_commit=current_access,
                advisory_actor=actor_context,
            )
        finally:
            if not enabled():self._connection.execute(f"PRAGMA busy_timeout={previous_timeout}")
        return winner

    def _validate_advisory_proposal(self, *, actor_context, command, proposal, receipt):
        """Closed typed insertion authority; no callback or registration grants it."""
        from services.ontology_repair_service import (
            _proposal_evidence,
            _verify_advisory_proposal_triple,
        )

        formation = _proposal_evidence(
            actor_context=actor_context, command=command, store=self
        )
        triple = _verify_advisory_proposal_triple(
            actor_context=actor_context,
            command=command,
            proposal=proposal,
            receipt=receipt,
            formation=formation,
        )
        return formation, triple

    def _publish_advisory_proposal(
        self, *, actor_context, command, proposal, receipt, verify_before_commit=None
    ):
        # Keep external reads bounded and outside the writer reservation. The
        # same typed/current-owner checks run again on the reserved connection.
        self._validate_advisory_proposal(
            actor_context=actor_context,
            command=command,
            proposal=proposal,
            receipt=receipt,
        )
        previous_timeout = None if enabled() else self._connection.execute("PRAGMA busy_timeout").fetchone()[0]
        if not enabled():self._connection.execute("PRAGMA busy_timeout=60000")
        try:
            return self._atomic_write(
                artifacts=(command, proposal, receipt),
                validate_before_insert=None,
                mutate_indexes=None,
                verify_before_commit=verify_before_commit,
                advisory_actor=actor_context,
            )
        finally:
            if not enabled():self._connection.execute(f"PRAGMA busy_timeout={previous_timeout}")

    def _atomic_write(
        self,
        *,
        artifacts,
        validate_before_insert,
        mutate_indexes,
        verify_before_commit=None,
        advisory_actor=None,
    ) -> AdvisoryNextProposalReceiptV1 | None:

        if in_transaction(self._connection):
            raise RuntimeError("atomic_write cannot nest inside an active transaction")
        begin_write(self._connection)
        try:
            if advisory_actor is not None and any(
                getattr(value, "schema_version", None) in _ADVISORY_PROPOSAL_MODELS
                for value in artifacts
            ):
                # This closed branch owns validation and index insertion. Caller
                # callbacks never authorize a proposal or its request binding.
                if (
                    not isinstance(artifacts, tuple)
                    or len(artifacts) != 3
                    or validate_before_insert is not None
                    or mutate_indexes is not None
                ):
                    raise ValueError("INVALID_STATE")
                from services.ontology_repair_service import OntologyRepairService

                result = OntologyRepairService()._publish_advisory_proposal(
                    actor_context=advisory_actor,
                    command=artifacts[0],
                    proposal=artifacts[1],
                    receipt=artifacts[2],
                    store=self,
                    verify_before_commit=verify_before_commit,
                )
                self._connection.commit()
                return result
            with self._guard_readonly_workspace_callback("validate_before_insert"):
                should_insert = validate_before_insert(self._connection)
            if not isinstance(should_insert, bool):
                raise TypeError("validate_before_insert must return bool")
            if not should_insert:
                if advisory_actor is not None:
                    with self._guard_readonly_workspace_callback(
                        "advisory verify_before_commit"
                    ):
                        verify_before_commit()
                self._connection.commit()
                return
            if not isinstance(artifacts, tuple) or not all(
                isinstance(artifact, BaseModel) for artifact in artifacts
            ):
                raise TypeError("artifacts must be a tuple of BaseModel instances")
            prepared = tuple(self._prepare(artifact) for artifact in artifacts)
            if advisory_actor is None:
                _reject_protected_publication(prepared)
            else:
                from services.runtime_context_formation_service import (
                    _verify_advisory_triple,
                )

                if len(artifacts) != 3:
                    raise ValueError("INVALID_STATE")
                _verify_advisory_triple(
                    actor_context=advisory_actor,
                    command=artifacts[0],
                    formation=artifacts[1],
                    receipt=artifacts[2],
                )
            self._insert_prepared(prepared)
            with self._guard_private_advisory_index():
                mutate_indexes(self._connection)
                if advisory_actor is None and verify_before_commit is not None:
                    verify_before_commit()
            if advisory_actor is not None:
                with self._guard_readonly_workspace_callback(
                    "advisory verify_before_commit"
                ):
                    verify_before_commit()
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    def _insert_prepared(
        self,
        prepared: tuple[tuple[str, str, str, str, str, str], ...],
    ) -> set[str]:
        """Insert already validated artifacts on the caller-owned transaction."""

        inserted: set[str] = set()
        for (
            organization_id,
            workspace_id,
            artifact_ref,
            artifact_hash,
            schema_version,
            payload_json,
        ) in prepared:
            if schema_version == _EXPLORATORY_INTEGRITY_SCHEMA:
                raise ValueError(
                    "exploratory snapshot integrity requires prospective integrity transaction"
                )
            if schema_version == "ontology_hub_snapshot.v1":
                snapshot_payload = json.loads(payload_json)
                rows = self._connection.execute(
                    sql_query("SELECT payload_json FROM ontology_build_artifacts WHERE organization_id = ? AND workspace_id = ? AND schema_version = ?",'SELECT payload_json FROM ontology_build_artifacts WHERE organization_id = %s AND workspace_id = %s AND schema_version = %s'),
                    (organization_id, workspace_id, schema_version),
                ).fetchall()
                for (encoded,) in rows:
                    previous = json.loads(encoded)
                    if (previous["ontology_ref"], previous["version"]) == (
                        snapshot_payload["ontology_ref"],
                        snapshot_payload["version"],
                    ) and previous != snapshot_payload:
                        raise ValueError(
                            "ontology Hub version is immutable; reuse the existing snapshot"
                        )
            if schema_version == "ontology_capability_profile.v1":
                profile_payload = json.loads(payload_json)
                version_rows = self._connection.execute(
                    sql_query("""
                    SELECT payload_json FROM ontology_build_artifacts
                    WHERE organization_id = ? AND workspace_id = ?
                      AND schema_version = ?
                    """,'\n                    SELECT payload_json FROM ontology_build_artifacts\n                    WHERE organization_id = %s AND workspace_id = %s\n                      AND schema_version = %s\n                    '),
                    (organization_id, workspace_id, schema_version),
                ).fetchall()
                for (existing_payload_json,) in version_rows:
                    existing_profile = json.loads(existing_payload_json)
                    if (
                        existing_profile["ontology_ref"]
                        == profile_payload["ontology_ref"]
                        and existing_profile["version"] == profile_payload["version"]
                        and (
                            existing_profile["source_content_hash"]
                            != profile_payload["source_content_hash"]
                            or existing_profile["graph_content_hash"]
                            != profile_payload["graph_content_hash"]
                        )
                    ):
                        raise ValueError(
                            "ontology version cannot be reused with a different content hash"
                        )
            existing = self._connection.execute(
                sql_query("SELECT organization_id, workspace_id, artifact_hash, payload_json "
                "FROM ontology_build_artifacts WHERE artifact_ref = ?",'SELECT organization_id, workspace_id, artifact_hash, payload_json FROM ontology_build_artifacts WHERE artifact_ref = %s'),
                (artifact_ref,),
            ).fetchone()
            if existing is not None:
                if existing == (
                    organization_id,
                    workspace_id,
                    artifact_hash,
                    payload_json,
                ):
                    continue
                raise ValueError(
                    "ontology artifacts are append-only; create a successor ref"
                )
            self._connection.execute(
                sql_query("""
                INSERT INTO ontology_build_artifacts (
                    organization_id, workspace_id, artifact_ref, artifact_hash,
                    schema_version, payload_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,'\n                INSERT INTO ontology_build_artifacts (\n                    organization_id, workspace_id, artifact_ref, artifact_hash,\n                    schema_version, payload_json, recorded_at\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    organization_id,
                    workspace_id,
                    artifact_ref,
                    artifact_hash,
                    schema_version,
                    payload_json,
                    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                ),
            )
            inserted.add(artifact_ref)
        return inserted

    def put_many_with_integrity(
        self,
        *,
        artifacts: tuple[BaseModel, ...],
        snapshot_integrities: tuple[ExploratoryGraphSnapshotIntegrityV1, ...],
        verify_before_commit: Callable[[], None] | None = None,
    ) -> None:
        """Append new exploratory snapshots and their owner integrity atomically."""

        if not snapshot_integrities:
            raise ValueError("at least one exploratory snapshot integrity is required")
        if any(
            not isinstance(item, ExploratoryGraphSnapshotIntegrityV1)
            for item in snapshot_integrities
        ):
            raise TypeError(
                "snapshot_integrities must contain ExploratoryGraphSnapshotIntegrityV1"
            )
        if any(
            getattr(item, "schema_version", None) == _EXPLORATORY_INTEGRITY_SCHEMA
            for item in artifacts
        ):
            raise ValueError(
                "integrity artifacts belong in snapshot_integrities, not artifacts"
            )
        snapshots = tuple(
            item for item in artifacts if isinstance(item, ExploratoryGraphSnapshotV1)
        )
        subject_refs = tuple(item.subject.artifact_ref for item in snapshot_integrities)
        if len(subject_refs) != len(set(subject_refs)):
            raise ValueError("each exploratory snapshot accepts exactly one integrity")
        if {item.snapshot_ref for item in snapshots} != set(subject_refs):
            raise ValueError(
                "each transaction snapshot must have one exact integrity subject"
            )

        prepared_artifacts = tuple(self._prepare(item) for item in artifacts)
        _reject_context_publication(prepared_artifacts)
        prepared_integrities = tuple(
            self._prepare(item) for item in snapshot_integrities
        )
        _reject_trial_projection_publication(
            (*prepared_artifacts, *prepared_integrities)
        )
        _reject_advisory_suite_publication((*prepared_artifacts, *prepared_integrities))
        if any(item[4] in _SEED_CAS_SCHEMAS for item in prepared_artifacts):
            raise ValueError(
                "ontology seed state/actions require append_seed_state CAS"
            )
        if any(item[4] in _RUNTIME_CONTEXT_CAS_SCHEMAS for item in prepared_artifacts):
            raise ValueError(
                "committed runtime context artifacts require append_context CAS"
            )
        if any(
            item[4] in _RUNTIME_CONTEXT_AUTHORITY_SCHEMAS for item in prepared_artifacts
        ):
            raise ValueError(
                "runtime context authority artifacts require authority facade"
            )

        begin_write(self._connection)
        try:
            self._insert_snapshot_integrities(
                snapshots=snapshots,
                integrities=tuple(snapshot_integrities),
                prepared_artifacts=prepared_artifacts,
                prepared_integrities=prepared_integrities,
            )
            if verify_before_commit is not None:
                with self._guard_private_advisory_index():
                    verify_before_commit()
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    def _insert_snapshot_integrities(
        self,
        *,
        snapshots: tuple[BaseModel, ...],
        integrities: tuple[ExploratoryGraphSnapshotIntegrityV1, ...],
        prepared_artifacts: tuple[tuple[str, str, str, str, str, str], ...],
        prepared_integrities: tuple[tuple[str, str, str, str, str, str], ...],
    ) -> set[str]:
        """Validate and insert exact pairs on an already active transaction."""

        from contracts.ontology_artifact_integrity import (
            verify_exploratory_snapshot_integrity,
        )

        snapshots_by_ref = {item.snapshot_ref: item for item in snapshots}
        new_index_refs: set[str] = set()
        for integrity in integrities:
            snapshot = snapshots_by_ref[integrity.subject.artifact_ref]
            verify_exploratory_snapshot_integrity(
                snapshot=snapshot, integrity=integrity
            )
            subject_row = self._connection.execute(
                sql_query("""
                SELECT organization_id, workspace_id, artifact_hash, schema_version,
                       payload_json
                FROM ontology_build_artifacts WHERE artifact_ref = ?
                """,'\n                SELECT organization_id, workspace_id, artifact_hash, schema_version,\n                       payload_json\n                FROM ontology_build_artifacts WHERE artifact_ref = %s\n                '),
                (snapshot.snapshot_ref,),
            ).fetchone()
            integrity_row = self._connection.execute(
                sql_query("""
                SELECT organization_id, workspace_id, artifact_hash, schema_version,
                       payload_json
                FROM ontology_build_artifacts WHERE artifact_ref = ?
                """,'\n                SELECT organization_id, workspace_id, artifact_hash, schema_version,\n                       payload_json\n                FROM ontology_build_artifacts WHERE artifact_ref = %s\n                '),
                (integrity.integrity_ref,),
            ).fetchone()
            index_row = self._connection.execute(
                sql_query("""
                SELECT organization_id, workspace_id, subject_hash, integrity_ref,
                       integrity_hash, producer_profile, payload_hash
                FROM ontology_artifact_integrity_index
                WHERE organization_id = ? AND workspace_id = ? AND subject_ref = ?
                """,'\n                SELECT organization_id, workspace_id, subject_hash, integrity_ref,\n                       integrity_hash, producer_profile, payload_hash\n                FROM ontology_artifact_integrity_index\n                WHERE organization_id = %s AND workspace_id = %s AND subject_ref = %s\n                '),
                (
                    snapshot.organization_id,
                    snapshot.workspace_id,
                    snapshot.snapshot_ref,
                ),
            ).fetchone()
            integrity_index_row = self._connection.execute(
                sql_query("""
                SELECT organization_id, workspace_id, subject_ref, subject_hash,
                       integrity_hash, producer_profile, payload_hash
                FROM ontology_artifact_integrity_index
                WHERE integrity_ref = ?
                """,'\n                SELECT organization_id, workspace_id, subject_ref, subject_hash,\n                       integrity_hash, producer_profile, payload_hash\n                FROM ontology_artifact_integrity_index\n                WHERE integrity_ref = %s\n                '),
                (integrity.integrity_ref,),
            ).fetchone()
            any_existing = any(
                item is not None
                for item in (
                    subject_row,
                    integrity_row,
                    index_row,
                    integrity_index_row,
                )
            )
            if any_existing:
                if (
                    subject_row is not None
                    and integrity_row is None
                    and index_row is None
                    and integrity_index_row is None
                ):
                    raise ValueError("LEGACY_INTEGRITY_UNPROVABLE")
                expected_subject = (
                    snapshot.organization_id,
                    snapshot.workspace_id,
                    snapshot.snapshot_hash,
                    snapshot.schema_version,
                    canonical_json(snapshot.model_dump(mode="json")),
                )
                expected_integrity = (
                    integrity.organization_id,
                    integrity.workspace_id,
                    integrity.integrity_hash,
                    integrity.schema_version,
                    canonical_json(integrity.model_dump(mode="json")),
                )
                expected_index = (
                    integrity.organization_id,
                    integrity.workspace_id,
                    integrity.subject.artifact_hash,
                    integrity.integrity_ref,
                    integrity.integrity_hash,
                    integrity.producer_profile,
                    integrity.payload_hash,
                )
                expected_integrity_index = (
                    integrity.organization_id,
                    integrity.workspace_id,
                    integrity.subject.artifact_ref,
                    integrity.subject.artifact_hash,
                    integrity.integrity_hash,
                    integrity.producer_profile,
                    integrity.payload_hash,
                )
                if (
                    subject_row != expected_subject
                    or integrity_row != expected_integrity
                    or index_row != expected_index
                    or integrity_index_row != expected_integrity_index
                ):
                    raise ValueError("INTEGRITY_CONFLICT")
            else:
                new_index_refs.add(integrity.integrity_ref)

        inserted = self._insert_prepared(prepared_artifacts)
        for (
            organization_id,
            workspace_id,
            integrity_ref,
            integrity_hash,
            schema_version,
            payload_json,
        ) in prepared_integrities:
            if (
                self._connection.execute(
                    sql_query("SELECT 1 FROM ontology_build_artifacts WHERE artifact_ref = ?",'SELECT 1 FROM ontology_build_artifacts WHERE artifact_ref = %s'),
                    (integrity_ref,),
                ).fetchone()
                is None
            ):
                self._connection.execute(
                    sql_query("""
                    INSERT INTO ontology_build_artifacts (
                        organization_id, workspace_id, artifact_ref, artifact_hash,
                        schema_version, payload_json, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,'\n                    INSERT INTO ontology_build_artifacts (\n                        organization_id, workspace_id, artifact_ref, artifact_hash,\n                        schema_version, payload_json, recorded_at\n                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)\n                    '),
                    (
                        organization_id,
                        workspace_id,
                        integrity_ref,
                        integrity_hash,
                        schema_version,
                        payload_json,
                        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    ),
                )
        for integrity in integrities:
            if integrity.integrity_ref not in new_index_refs:
                continue
            self._connection.execute(
                sql_query("""
                INSERT INTO ontology_artifact_integrity_index (
                    organization_id, workspace_id, subject_ref, subject_hash,
                    integrity_ref, integrity_hash, producer_profile, payload_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,'\n                INSERT INTO ontology_artifact_integrity_index (\n                    organization_id, workspace_id, subject_ref, subject_hash,\n                    integrity_ref, integrity_hash, producer_profile, payload_hash\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    integrity.organization_id,
                    integrity.workspace_id,
                    integrity.subject.artifact_ref,
                    integrity.subject.artifact_hash,
                    integrity.integrity_ref,
                    integrity.integrity_hash,
                    integrity.producer_profile,
                    integrity.payload_hash,
                ),
            )
        return inserted

    def get_verified_exploratory_snapshot(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        pin: ArtifactPinV1,
    ) -> tuple[ExploratoryGraphSnapshotV1, ExploratoryGraphSnapshotIntegrityV1]:
        """Resolve one snapshot only through its exact owner integrity binding."""

        from contracts.ontology_artifact_integrity import (
            verify_exploratory_snapshot_integrity,
        )

        pin = ArtifactPinV1.model_validate(pin.model_dump(mode="json"))
        if pin.artifact_kind != "exploratory_graph_snapshot":
            raise ValueError("INTEGRITY_SUBJECT_MISMATCH")
        subject_row = self._connection.execute(
            sql_query("""
            SELECT organization_id, workspace_id, artifact_hash, schema_version,
                   payload_json
            FROM ontology_build_artifacts WHERE artifact_ref = ?
            """,'\n            SELECT organization_id, workspace_id, artifact_hash, schema_version,\n                   payload_json\n            FROM ontology_build_artifacts WHERE artifact_ref = %s\n            '),
            (pin.artifact_ref,),
        ).fetchone()
        if subject_row is None:
            raise ValueError("INTEGRITY_CONFLICT")
        if (subject_row[0], subject_row[1]) != (organization_id, workspace_id):
            raise PermissionError("artifact ref is outside the governed tenant scope")
        if (subject_row[2], subject_row[3]) != (
            pin.artifact_hash,
            "exploratory_graph_snapshot.v1",
        ):
            raise ValueError("INTEGRITY_SUBJECT_MISMATCH")
        index_row = self._connection.execute(
            sql_query("""
            SELECT subject_hash, integrity_ref, integrity_hash, producer_profile,
                   payload_hash
            FROM ontology_artifact_integrity_index
            WHERE organization_id = ? AND workspace_id = ? AND subject_ref = ?
            """,'\n            SELECT subject_hash, integrity_ref, integrity_hash, producer_profile,\n                   payload_hash\n            FROM ontology_artifact_integrity_index\n            WHERE organization_id = %s AND workspace_id = %s AND subject_ref = %s\n            '),
            (organization_id, workspace_id, pin.artifact_ref),
        ).fetchone()
        if index_row is None:
            unindexed_integrity = self._connection.execute(
                sql_query("""
                SELECT artifact_ref FROM ontology_build_artifacts
                WHERE schema_version = ?
                  AND json_extract(payload_json, '$.subject.artifact_ref') = ?
                LIMIT 1
                ""","\n                SELECT artifact_ref FROM ontology_build_artifacts\n                WHERE schema_version = %s\n                  AND (payload_json::jsonb #>> '{subject,artifact_ref}') = %s\n                LIMIT 1\n                "),
                (
                    _EXPLORATORY_INTEGRITY_SCHEMA,
                    pin.artifact_ref,
                ),
            ).fetchone()
            foreign_index = self._connection.execute(
                sql_query("""
                SELECT integrity_ref FROM ontology_artifact_integrity_index
                WHERE subject_ref = ? LIMIT 1
                """,'\n                SELECT integrity_ref FROM ontology_artifact_integrity_index\n                WHERE subject_ref = %s LIMIT 1\n                '),
                (pin.artifact_ref,),
            ).fetchone()
            if unindexed_integrity is not None or foreign_index is not None:
                raise ValueError("INTEGRITY_CONFLICT")
            raise ValueError("LEGACY_INTEGRITY_UNPROVABLE")
        integrity_row = self._connection.execute(
            sql_query("""
            SELECT organization_id, workspace_id, artifact_hash, schema_version,
                   payload_json
            FROM ontology_build_artifacts WHERE artifact_ref = ?
            """,'\n            SELECT organization_id, workspace_id, artifact_hash, schema_version,\n                   payload_json\n            FROM ontology_build_artifacts WHERE artifact_ref = %s\n            '),
            (index_row[1],),
        ).fetchone()
        if integrity_row is None or (
            integrity_row[0],
            integrity_row[1],
            integrity_row[2],
            integrity_row[3],
        ) != (
            organization_id,
            workspace_id,
            index_row[2],
            _EXPLORATORY_INTEGRITY_SCHEMA,
        ):
            raise ValueError("INTEGRITY_CONFLICT")
        try:
            snapshot = ExploratoryGraphSnapshotV1.model_validate(
                json.loads(subject_row[4])
            )
            integrity = ExploratoryGraphSnapshotIntegrityV1.model_validate(
                json.loads(integrity_row[4])
            )
        except ValueError as exc:
            raise ValueError("INTEGRITY_CONFLICT") from exc
        if (
            snapshot.snapshot_ref,
            snapshot.snapshot_hash,
            integrity.integrity_ref,
            integrity.integrity_hash,
            integrity.subject.artifact_hash,
            integrity.producer_profile,
            integrity.payload_hash,
        ) != (
            pin.artifact_ref,
            pin.artifact_hash,
            index_row[1],
            index_row[2],
            index_row[0],
            index_row[3],
            index_row[4],
        ):
            raise ValueError("INTEGRITY_CONFLICT")
        verify_exploratory_snapshot_integrity(snapshot=snapshot, integrity=integrity)
        return snapshot, integrity

    def get(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        artifact_ref: str,
        contract_type: type[ContractT],
    ) -> ContractT | None:
        row = self._connection.execute(
            sql_query("SELECT organization_id, workspace_id, payload_json FROM ontology_build_artifacts WHERE artifact_ref = ?",'SELECT organization_id, workspace_id, payload_json FROM ontology_build_artifacts WHERE artifact_ref = %s'),
            (artifact_ref,),
        ).fetchone()
        if row is None:
            return None
        if row[0] != organization_id or row[1] != workspace_id:
            raise PermissionError("artifact ref is outside the governed tenant scope")
        return contract_type.model_validate(json.loads(row[2]))

    def has_retained_context_changes(self, *, workspace_id: str) -> bool:
        """Presence guard for runtime reconciliation; it grants no access or authority."""
        return self._connection.execute(
            sql_query("SELECT 1 FROM ontology_build_artifacts WHERE workspace_id=? AND schema_version=? LIMIT 1",'SELECT 1 FROM ontology_build_artifacts WHERE workspace_id=%s AND schema_version=%s LIMIT 1'),
            (workspace_id, "retained_context_item_change.v1"),
        ).fetchone() is not None

    def list_jd_review_lineage_pins(
        self, *, organization_id: str, workspace_id: str, actor_id: str,
        session_id: str, lineage_review_id: str,
    ) -> tuple[ArtifactPinV1, ...]:
        """Select only the caller's immutable review lineage before payload reads."""
        rows = self._connection.execute(
            sql_query("""SELECT artifact_ref, artifact_hash FROM ontology_build_artifacts
            WHERE organization_id=? AND workspace_id=? AND schema_version='jd_review.v1'
            AND json_extract(payload_json, '$.actor_id')=?
            AND json_extract(payload_json, '$.session_id')=?
            AND (artifact_ref=? OR json_extract(payload_json, '$.successor.lineage_review_id')=?
                 OR json_extract(payload_json, '$.successor.supersedes_review_id')=?)
            ORDER BY artifact_ref""","SELECT artifact_ref, artifact_hash FROM ontology_build_artifacts\n            WHERE organization_id=%s AND workspace_id=%s AND schema_version='jd_review.v1'\n            AND (payload_json::jsonb #>> '{actor_id}')=%s\n            AND (payload_json::jsonb #>> '{session_id}')=%s\n            AND (artifact_ref=%s OR (payload_json::jsonb #>> '{successor,lineage_review_id}')=%s\n                 OR (payload_json::jsonb #>> '{successor,supersedes_review_id}')=%s)\n            ORDER BY artifact_ref"),
            (organization_id, workspace_id, actor_id, session_id,
             lineage_review_id, lineage_review_id, lineage_review_id),
        ).fetchall()
        return tuple(ArtifactPinV1(artifact_kind='jd_review',artifact_ref=ref,artifact_hash=digest) for ref,digest in rows)

    def list_artifact_pins(
        self, *, organization_id: str, workspace_id: str, schema_version: str
    ) -> tuple[ArtifactPinV1, ...]:
        """Return exact, scope-bound pins in deterministic artifact order.

        This is intentionally a read-only index seam.  Callers still resolve
        each pin through ``get`` so the stored hash and contract payload are
        revalidated at the semantic boundary.
        """
        rows = self._connection.execute(
            sql_query("""SELECT artifact_ref, artifact_hash
            FROM ontology_build_artifacts
            WHERE organization_id=? AND workspace_id=? AND schema_version=?
            ORDER BY artifact_ref, artifact_hash""",'SELECT artifact_ref, artifact_hash\n            FROM ontology_build_artifacts\n            WHERE organization_id=%s AND workspace_id=%s AND schema_version=%s\n            ORDER BY artifact_ref, artifact_hash'),
            (organization_id, workspace_id, schema_version),
        ).fetchall()
        learning_model = LEARNING_ARTIFACT_MODELS.get(schema_version)
        return tuple(
            ArtifactPinV1(
                artifact_kind=learning_model.ARTIFACT_KIND if learning_model else "change_candidate",
                artifact_ref=artifact_ref,
                artifact_hash=artifact_hash,
            )
            for artifact_ref, artifact_hash in rows
        )

    def list_learning_graph_records(self, *, organization_id, workspace_id, owner_ref, as_of, after_ref='', limit=201):
        """Read one bounded personal learning cohort; filter before decoding."""
        rows = self._connection.execute(sql_query(
            "SELECT artifact_ref, artifact_hash, schema_version FROM ontology_build_artifacts WHERE organization_id=? AND workspace_id=? AND COALESCE(json_extract(payload_json,'$.owner_ref'),json_extract(payload_json,'$.profile.owner_ref'))=? AND (schema_version IN ('skill_artifact_version.v1','learned_skill_proposal.v1','learned_skill_assessment.v1','learned_skill_publication_receipt.v1','retained_context_item_change.v1','learning_trace_observation.v1') OR schema_version='semantier.change_candidate.v2') AND (substr(recorded_at,1,19)||'.'||CASE WHEN instr(recorded_at,'.')=0 THEN '000000' ELSE substr(recorded_at,instr(recorded_at,'.')+1,6) END||'Z')<=? AND artifact_ref>? ORDER BY artifact_ref LIMIT ?",
            "SELECT artifact_ref, artifact_hash, schema_version FROM ontology_build_artifacts WHERE organization_id=%s AND workspace_id=%s AND COALESCE(payload_json::jsonb->>'owner_ref',payload_json::jsonb->'profile'->>'owner_ref')=%s AND (schema_version IN ('skill_artifact_version.v1','learned_skill_proposal.v1','learned_skill_assessment.v1','learned_skill_publication_receipt.v1','retained_context_item_change.v1','learning_trace_observation.v1') OR schema_version='semantier.change_candidate.v2') AND recorded_at::timestamptz<=%s::timestamptz AND artifact_ref>%s ORDER BY artifact_ref LIMIT %s"),
            (organization_id,workspace_id,owner_ref,as_of,after_ref,min(201,max(1,limit)))).fetchall()
        return tuple(dict(artifact_ref=r[0],artifact_hash=r[1],schema_version=r[2]) for r in rows)

    def list_learning_trace_pins(self, *, organization_id, workspace_id, owner_ref, session_id, limit):
        """Bound the selected observation scope before decoding any payloads."""
        rows = self._connection.execute(sql_query(
            "SELECT artifact_ref, artifact_hash FROM ontology_build_artifacts WHERE organization_id=? AND workspace_id=? AND schema_version='learning_trace_observation.v1' AND json_extract(payload_json,'$.owner_ref')=? AND json_extract(payload_json,'$.session_id')=? ORDER BY artifact_ref LIMIT ?",
            "SELECT artifact_ref, artifact_hash FROM ontology_build_artifacts WHERE organization_id=%s AND workspace_id=%s AND schema_version='learning_trace_observation.v1' AND payload_json::jsonb->>'owner_ref'=%s AND payload_json::jsonb->>'session_id'=%s ORDER BY artifact_ref LIMIT %s"),
            (organization_id, workspace_id, owner_ref, session_id, limit)).fetchall()
        return tuple(ArtifactPinV1(artifact_kind='learning_trace_observation',artifact_ref=ref,artifact_hash=digest) for ref,digest in rows)

    def get_raw(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        artifact_ref: str,
    ) -> dict[str, object] | None:
        """Return an exact stored payload for deterministic audit packaging."""

        row = self._connection.execute(
            sql_query("SELECT organization_id, workspace_id, artifact_hash, schema_version, payload_json FROM ontology_build_artifacts WHERE artifact_ref = ?",'SELECT organization_id, workspace_id, artifact_hash, schema_version, payload_json FROM ontology_build_artifacts WHERE artifact_ref = %s'),
            (artifact_ref,),
        ).fetchone()
        if row is None:
            return None
        if row[0] != organization_id or row[1] != workspace_id:
            raise PermissionError("artifact ref is outside the governed tenant scope")
        return {
            "artifact_hash": row[2],
            "schema_version": row[3],
            "payload": json.loads(row[4]),
        }

    def get_seed_state_head(
        self, *, organization_id: str, workspace_id: str, build_ref: str
    ):
        """Return the validated head of one append-only seed-discovery chain."""
        owns_read_transaction = not in_transaction(self._connection)
        if owns_read_transaction:
            self._connection.execute(sql_query("BEGIN","BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY"))
        try:
            rows = self._seed_state_index_rows(
                organization_id=organization_id,
                workspace_id=workspace_id,
                build_ref=build_ref,
            )
            states = self._validate_seed_state_index(
                organization_id=organization_id,
                workspace_id=workspace_id,
                build_ref=build_ref,
                rows=rows,
            )
            if owns_read_transaction:
                self._connection.commit()
            return states[-1] if states else None
        except BaseException:
            if owns_read_transaction:
                self._connection.rollback()
            raise

    def _seed_state_index_rows(
        self, *, organization_id: str, workspace_id: str, build_ref: str
    ):
        return self._connection.execute(
            sql_query("""
            SELECT state_index.revision, state_index.state_ref, state_index.state_hash,
                   state_index.predecessor_state_ref, state_index.predecessor_state_hash,
                   artifact.organization_id, artifact.workspace_id, artifact.schema_version,
                   artifact.artifact_hash, artifact.payload_json
            FROM ontology_seed_state_index AS state_index
            LEFT JOIN ontology_build_artifacts AS artifact
              ON artifact.artifact_ref = state_index.state_ref
            WHERE state_index.organization_id = ? AND state_index.workspace_id = ?
              AND state_index.build_ref = ?
            ORDER BY state_index.revision
            """,'\n            SELECT state_index.revision, state_index.state_ref, state_index.state_hash,\n                   state_index.predecessor_state_ref, state_index.predecessor_state_hash,\n                   artifact.organization_id, artifact.workspace_id, artifact.schema_version,\n                   artifact.artifact_hash, artifact.payload_json\n            FROM ontology_seed_state_index AS state_index\n            LEFT JOIN ontology_build_artifacts AS artifact\n              ON artifact.artifact_ref = state_index.state_ref\n            WHERE state_index.organization_id = %s AND state_index.workspace_id = %s\n              AND state_index.build_ref = %s\n            ORDER BY state_index.revision\n            '),
            (organization_id, workspace_id, build_ref),
        ).fetchall()

    def _validate_seed_state_index(
        self, *, organization_id: str, workspace_id: str, build_ref: str, rows
    ):
        from contracts.ontology_seed_state import OntologySeedStateV1

        indexed_count = self._connection.execute(
            sql_query("SELECT COUNT(*) FROM ontology_seed_state_index WHERE organization_id = ? AND workspace_id = ?",'SELECT COUNT(*) FROM ontology_seed_state_index WHERE organization_id = %s AND workspace_id = %s'),
            (organization_id, workspace_id),
        ).fetchone()[0]
        artifact_count = self._connection.execute(
            sql_query("SELECT COUNT(*) FROM ontology_build_artifacts WHERE organization_id = ? AND workspace_id = ? "
            "AND schema_version = 'ontology_seed_state.v1'","SELECT COUNT(*) FROM ontology_build_artifacts WHERE organization_id = %s AND workspace_id = %s AND schema_version = 'ontology_seed_state.v1'"),
            (organization_id, workspace_id),
        ).fetchone()[0]
        if indexed_count != artifact_count:
            raise ValueError("ontology seed state index/artifact membership is corrupt")
        states = []
        for (
            revision,
            indexed_ref,
            indexed_hash,
            indexed_predecessor_ref,
            indexed_predecessor_hash,
            stored_org,
            stored_workspace,
            stored_schema,
            stored_hash,
            encoded,
        ) in rows:
            if encoded is None:
                raise ValueError("indexed ontology seed state artifact is missing")
            state = OntologySeedStateV1.model_validate(json.loads(encoded))
            if (stored_org, stored_workspace, stored_schema, stored_hash) != (
                organization_id,
                workspace_id,
                "ontology_seed_state.v1",
                indexed_hash,
            ):
                raise ValueError(
                    "ontology seed state artifact/index row binding is corrupt"
                )
            if (
                state.organization_id,
                state.workspace_id,
                state.build_ref,
                state.revision,
                state.state_ref,
                state.state_hash,
                state.predecessor_state_ref,
                state.predecessor_state_hash,
            ) != (
                organization_id,
                workspace_id,
                build_ref,
                revision,
                indexed_ref,
                indexed_hash,
                indexed_predecessor_ref,
                indexed_predecessor_hash,
            ):
                raise ValueError("ontology seed state row binding is corrupt")
            states.append(state)
        predecessor = None
        for index, state in enumerate(states, start=1):
            if state.revision != index:
                raise ValueError("ontology seed state chain has a revision gap or fork")
            expected = (
                (None, None)
                if predecessor is None
                else (predecessor.state_ref, predecessor.state_hash)
            )
            if (state.predecessor_state_ref, state.predecessor_state_hash) != expected:
                raise ValueError("ontology seed state chain has a broken predecessor")
            predecessor = state
        return states

    def append_seed_state(
        self,
        state: BaseModel,
        *,
        expected_state_ref: str | None,
        expected_state_hash: str | None,
        related_artifacts: tuple[BaseModel, ...] = (),
        snapshot_integrities: tuple[ExploratoryGraphSnapshotIntegrityV1, ...] = (),
        request_receipt: BaseModel | None = None,
        verify_before_commit: Callable[[], None] | None = None,
    ) -> None:
        """CAS-append a state and related receipts in one SQLite transaction."""
        from contracts.ontology_seed_state import OntologySeedStateV1

        if not isinstance(state, OntologySeedStateV1):
            raise TypeError("state must be OntologySeedStateV1")
        if (expected_state_ref is None) != (expected_state_hash is None):
            raise ValueError("expected seed state ref/hash must be paired")
        if any(
            getattr(item, "schema_version", None) == _EXPLORATORY_INTEGRITY_SCHEMA
            for item in related_artifacts
        ):
            raise ValueError(
                "integrity artifacts belong in snapshot_integrities, not related_artifacts"
            )
        if any(
            not isinstance(item, ExploratoryGraphSnapshotIntegrityV1)
            for item in snapshot_integrities
        ):
            raise TypeError(
                "snapshot_integrities must contain ExploratoryGraphSnapshotIntegrityV1"
            )
        related_snapshots = tuple(
            item
            for item in related_artifacts
            if isinstance(item, ExploratoryGraphSnapshotV1)
        )
        if {item.snapshot_ref for item in related_snapshots} != {
            item.subject.artifact_ref for item in snapshot_integrities
        }:
            raise ValueError(
                "each related exploratory snapshot must have one exact integrity subject"
            )
        if len(snapshot_integrities) != len(
            {item.subject.artifact_ref for item in snapshot_integrities}
        ):
            raise ValueError("each exploratory snapshot accepts exactly one integrity")
        prepared = tuple(self._prepare(item) for item in (state, *related_artifacts))
        prepared_integrities = tuple(
            self._prepare(item) for item in snapshot_integrities
        )
        _reject_context_publication(prepared[1:])
        _reject_trial_projection_publication((*prepared[1:], *prepared_integrities))
        _reject_advisory_suite_publication((*prepared[1:], *prepared_integrities))
        if any(item[4] in _RUNTIME_CONTEXT_CAS_SCHEMAS for item in prepared[1:]):
            raise ValueError("runtime context artifacts require append_context CAS")
        if any(item[4] in _RUNTIME_CONTEXT_AUTHORITY_SCHEMAS for item in prepared[1:]):
            raise ValueError(
                "runtime context authority artifacts require authority facade"
            )
        for item in related_artifacts:
            if getattr(item, "schema_version", None) == "ontology_seed_ignore.v1" and (
                item.organization_id,
                item.workspace_id,
                item.build_ref,
                item.predecessor_state_ref,
                item.predecessor_state_hash,
                item.successor_state_ref,
                item.successor_state_hash,
            ) != (
                state.organization_id,
                state.workspace_id,
                state.build_ref,
                state.predecessor_state_ref,
                state.predecessor_state_hash,
                state.state_ref,
                state.state_hash,
            ):
                raise ValueError(
                    "ignore receipt does not bind the appended seed-state transition"
                )
        action_receipts = [
            item
            for item in related_artifacts
            if getattr(item, "schema_version", None)
            in ("ontology_seed_selection.v1", "ontology_seed_apply.v1")
        ]
        if action_receipts != ([] if request_receipt is None else [request_receipt]):
            raise ValueError(
                "selection/apply transition requires exactly its indexed request receipt"
            )
        if request_receipt is not None:
            if (
                request_receipt not in related_artifacts
                or request_receipt.schema_version
                not in ("ontology_seed_selection.v1", "ontology_seed_apply.v1")
            ):
                raise ValueError(
                    "request receipt must be an exact related selection/apply artifact"
                )
            if (
                request_receipt.organization_id,
                request_receipt.workspace_id,
                request_receipt.build_ref,
                request_receipt.expected_state.artifact_ref,
                request_receipt.expected_state.artifact_hash,
            ) != (
                state.organization_id,
                state.workspace_id,
                state.build_ref,
                expected_state_ref,
                expected_state_hash,
            ):
                raise ValueError("request receipt does not bind exact state transition")
            pin = (
                state.selection_receipt
                if request_receipt.operation == "select_seed"
                else state.latest_apply_receipt
            )
            if pin is None or (pin.artifact_ref, pin.artifact_hash) != (
                request_receipt.receipt_ref,
                request_receipt.receipt_hash,
            ):
                raise ValueError("state must pin its exact request receipt")
        begin_write(self._connection)
        try:
            rows = self._seed_state_index_rows(
                organization_id=state.organization_id,
                workspace_id=state.workspace_id,
                build_ref=state.build_ref,
            )
            states = self._validate_seed_state_index(
                organization_id=state.organization_id,
                workspace_id=state.workspace_id,
                build_ref=state.build_ref,
                rows=rows,
            )
            previous = states[-1] if states else None
            current_pin = (
                (None, None)
                if previous is None
                else (previous.state_ref, previous.state_hash)
            )
            if current_pin != (expected_state_ref, expected_state_hash):
                raise ValueError("expected ontology seed state is stale")
            if (
                state.revision != len(states) + 1
                or (state.predecessor_state_ref, state.predecessor_state_hash)
                != current_pin
            ):
                raise ValueError(
                    "ontology seed successor does not extend the current head"
                )
            if request_receipt is not None:
                if (
                    previous is None
                    or request_receipt.expected_state.payload_hash
                    != "sha256:" + content_hash(previous.model_dump(mode="json"))
                ):
                    raise ValueError(
                        "request receipt does not bind complete predecessor state"
                    )
                if (
                    request_receipt.baseline_manifest.artifact_ref,
                    request_receipt.baseline_manifest.artifact_hash,
                    request_receipt.created_at,
                ) != (state.build_ref, state.build_hash, state.created_at):
                    raise ValueError("request receipt does not bind state build/time")
                if state.created_at < previous.created_at:
                    raise ValueError("seed action cannot predate predecessor state")
                allowed = {
                    "state_ref",
                    "state_hash",
                    "revision",
                    "predecessor_state_ref",
                    "predecessor_state_hash",
                    "created_at",
                    "selection_receipt",
                    "latest_apply_receipt",
                    "selected_recommendation",
                    "imported_snapshot",
                    "alignment_actions",
                    "source_grounded_delta",
                }
                if state.model_dump(
                    mode="json", exclude=allowed
                ) != previous.model_dump(mode="json", exclude=allowed):
                    raise ValueError("seed action changed unrelated discovery inputs")
                if request_receipt.operation == "select_seed":
                    expected_snapshot = request_receipt.imported_snapshot
                    if state.imported_snapshot is None or (
                        state.imported_snapshot.artifact_ref,
                        state.imported_snapshot.artifact_hash,
                    ) != (
                        expected_snapshot.artifact_ref,
                        expected_snapshot.artifact_hash,
                    ):
                        raise ValueError("selected state snapshot differs from receipt")
                    if (
                        state.selected_recommendation != request_receipt.recommendation
                        or state.alignment_actions
                        or state.source_grounded_delta
                        or state.latest_apply_receipt
                    ):
                        raise ValueError("selection must reset application projections")
                else:
                    if (
                        previous.selection_receipt is None
                        or previous.latest_apply_receipt is not None
                        or (
                            previous.selection_receipt.artifact_ref,
                            previous.selection_receipt.artifact_hash,
                        )
                        != (
                            request_receipt.selection_receipt.artifact_ref,
                            request_receipt.selection_receipt.artifact_hash,
                        )
                    ):
                        raise ValueError(
                            "apply receipt does not bind selected predecessor"
                        )
                    if (
                        state.selection_receipt,
                        state.imported_snapshot,
                        state.selected_recommendation,
                    ) != (
                        previous.selection_receipt,
                        previous.imported_snapshot,
                        previous.selected_recommendation,
                    ):
                        raise ValueError("apply cannot change selected seed")
                    if (
                        tuple(
                            (pin.artifact_ref, pin.artifact_hash)
                            for pin in state.alignment_actions
                        )
                        != (
                            *tuple(
                                (pin.artifact_ref, pin.artifact_hash)
                                for pin in previous.alignment_actions
                            ),
                            (
                                request_receipt.alignment.artifact_ref,
                                request_receipt.alignment.artifact_hash,
                            ),
                        )
                        or state.source_grounded_delta is None
                        or (
                            state.source_grounded_delta.artifact_ref,
                            state.source_grounded_delta.artifact_hash,
                        )
                        != (
                            request_receipt.delta.artifact_ref,
                            request_receipt.delta.artifact_hash,
                        )
                    ):
                        raise ValueError(
                            "apply state must project exact command and delta"
                        )
            elif previous is not None and (
                state.selection_receipt,
                state.latest_apply_receipt,
            ) != (previous.selection_receipt, previous.latest_apply_receipt):
                if (
                    state.selection_receipt is not None
                    or state.latest_apply_receipt is not None
                ):
                    raise ValueError("new selection/apply pins require indexed receipt")
            ignore_receipts = [
                item
                for item in related_artifacts
                if getattr(item, "schema_version", None) == "ontology_seed_ignore.v1"
            ]
            old_ignored = (
                () if previous is None else previous.ignored_recommendation_refs
            )
            added_ignored = tuple(
                ref
                for ref in state.ignored_recommendation_refs
                if ref not in set(old_ignored)
            )
            if added_ignored or ignore_receipts:
                if (
                    previous is None
                    or len(added_ignored) != 1
                    or len(ignore_receipts) != 1
                ):
                    raise ValueError(
                        "ignore transition requires exactly one matching receipt"
                    )
                receipt = ignore_receipts[0]
                recommendation = next(
                    (
                        item
                        for item in previous.visible_recommendations
                        if item.recommendation_ref == added_ignored[0]
                        and item.status == "PROPOSED"
                    ),
                    None,
                )
                if recommendation is None or previous.query is None:
                    raise ValueError(
                        "ignore receipt does not identify a visible proposed recommendation"
                    )
                unchanged_exclusions = {
                    "schema_version",
                    "state_ref",
                    "state_hash",
                    "revision",
                    "predecessor_state_ref",
                    "predecessor_state_hash",
                    "ignored_recommendation_refs",
                    "created_at",
                }
                if state.model_dump(
                    mode="json", exclude=unchanged_exclusions
                ) != previous.model_dump(mode="json", exclude=unchanged_exclusions):
                    raise ValueError(
                        "ignore transition changed unrelated discovery state"
                    )
                if state.ignored_recommendation_refs != (
                    *old_ignored,
                    recommendation.recommendation_ref,
                ):
                    raise ValueError(
                        "ignore transition must append one recommendation ref"
                    )
                if (
                    receipt.query_ref,
                    receipt.query_hash,
                    receipt.recommendation_ref,
                    receipt.recommendation_hash,
                    receipt.ignored_at,
                ) != (
                    previous.query.query_ref,
                    previous.query.query_hash,
                    recommendation.recommendation_ref,
                    recommendation.recommendation_hash,
                    state.created_at,
                ):
                    raise ValueError(
                        "ignore receipt does not bind the exact ignored recommendation/query"
                    )
            for (
                organization_id,
                workspace_id,
                _artifact_ref,
                _artifact_hash,
                _schema_version,
                _payload_json,
            ) in prepared:
                if (organization_id, workspace_id) != (
                    state.organization_id,
                    state.workspace_id,
                ):
                    raise ValueError("related seed artifacts must share tenant scope")
            if snapshot_integrities:
                inserted = self._insert_snapshot_integrities(
                    snapshots=related_snapshots,
                    integrities=tuple(snapshot_integrities),
                    prepared_artifacts=prepared,
                    prepared_integrities=prepared_integrities,
                )
            else:
                inserted = self._insert_prepared(prepared)
            if state.state_ref in inserted:
                self._connection.execute(
                    sql_query("""
                    INSERT INTO ontology_seed_state_index (
                        organization_id, workspace_id, build_ref, revision,
                        state_ref, state_hash, predecessor_state_ref, predecessor_state_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,'\n                    INSERT INTO ontology_seed_state_index (\n                        organization_id, workspace_id, build_ref, revision,\n                        state_ref, state_hash, predecessor_state_ref, predecessor_state_hash\n                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)\n                    '),
                    (
                        state.organization_id,
                        state.workspace_id,
                        state.build_ref,
                        state.revision,
                        state.state_ref,
                        state.state_hash,
                        state.predecessor_state_ref,
                        state.predecessor_state_hash,
                    ),
                )
            if request_receipt is not None:
                self._connection.execute(
                    sql_query("""
                    INSERT INTO ontology_seed_request_index VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,'\n                    INSERT INTO ontology_seed_request_index VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                    (
                        state.organization_id,
                        state.workspace_id,
                        request_receipt.actor_ref,
                        request_receipt.operation,
                        request_receipt.request_id,
                        request_receipt.request_hash,
                        request_receipt.receipt_ref,
                        request_receipt.receipt_hash,
                        state.state_ref,
                        state.state_hash,
                    ),
                )
            if verify_before_commit is not None:
                verify_before_commit()
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    def get_seed_request(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        actor_ref: str,
        operation: str,
        request_id: str,
    ):
        """Resolve one indexed immutable request receipt and its resulting state."""
        from contracts.ontology_seed_application import (
            SeedSelectionReceiptV1,
            SeedApplyReceiptV1,
        )
        from contracts.ontology_seed_state import OntologySeedStateV1

        # A lost/mislabeled authoritative index cannot make a committed action
        # look unseen. Check both directions with bounded diagnostic output.
        orphan = self._connection.execute(
            sql_query("""
            SELECT a.artifact_ref FROM ontology_build_artifacts a
            LEFT JOIN ontology_seed_request_index i ON i.receipt_ref = a.artifact_ref
            WHERE a.organization_id=? AND a.workspace_id=?
              AND a.schema_version IN ('ontology_seed_selection.v1', 'ontology_seed_apply.v1')
              AND (i.receipt_ref IS NULL OR i.organization_id IS NOT a.organization_id
                OR i.workspace_id IS NOT a.workspace_id OR i.receipt_hash IS NOT a.artifact_hash
                OR i.actor_ref IS NOT json_extract(a.payload_json, '$.actor_ref')
                OR i.operation IS NOT json_extract(a.payload_json, '$.operation')
                OR i.request_id IS NOT json_extract(a.payload_json, '$.request_id')
                OR i.request_hash IS NOT json_extract(a.payload_json, '$.request_hash'))
            LIMIT 1
        ""","\n            SELECT a.artifact_ref FROM ontology_build_artifacts a\n            LEFT JOIN ontology_seed_request_index i ON i.receipt_ref = a.artifact_ref\n            WHERE a.organization_id=%s AND a.workspace_id=%s\n              AND a.schema_version IN ('ontology_seed_selection.v1', 'ontology_seed_apply.v1')\n              AND (i.receipt_ref IS NULL OR i.organization_id IS DISTINCT FROM a.organization_id\n                OR i.workspace_id IS DISTINCT FROM a.workspace_id OR i.receipt_hash IS DISTINCT FROM a.artifact_hash\n                OR i.actor_ref IS DISTINCT FROM (a.payload_json::jsonb #>> '{actor_ref}')\n                OR i.operation IS DISTINCT FROM (a.payload_json::jsonb #>> '{operation}')\n                OR i.request_id IS DISTINCT FROM (a.payload_json::jsonb #>> '{request_id}')\n                OR i.request_hash IS DISTINCT FROM (a.payload_json::jsonb #>> '{request_hash}'))\n            LIMIT 1\n        "),
            (organization_id, workspace_id),
        ).fetchone()
        dangling = self._connection.execute(
            sql_query("""
            SELECT i.receipt_ref FROM ontology_seed_request_index i
            LEFT JOIN ontology_build_artifacts a ON a.artifact_ref = i.receipt_ref
            WHERE i.organization_id=? AND i.workspace_id=?
              AND (a.artifact_ref IS NULL OR a.organization_id IS NOT i.organization_id
                OR a.workspace_id IS NOT i.workspace_id
                OR a.schema_version NOT IN ('ontology_seed_selection.v1', 'ontology_seed_apply.v1'))
            LIMIT 1
        ""","\n            SELECT i.receipt_ref FROM ontology_seed_request_index i\n            LEFT JOIN ontology_build_artifacts a ON a.artifact_ref = i.receipt_ref\n            WHERE i.organization_id=%s AND i.workspace_id=%s\n              AND (a.artifact_ref IS NULL OR a.organization_id IS DISTINCT FROM i.organization_id\n                OR a.workspace_id IS DISTINCT FROM i.workspace_id\n                OR a.schema_version NOT IN ('ontology_seed_selection.v1', 'ontology_seed_apply.v1'))\n            LIMIT 1\n        "),
            (organization_id, workspace_id),
        ).fetchone()
        if orphan is not None or dangling is not None:
            raise ValueError(
                "seed request index is missing or inconsistent with committed receipts"
            )
        row = self._connection.execute(
            sql_query("""
            SELECT request_hash, receipt_ref, receipt_hash, state_ref, state_hash
            FROM ontology_seed_request_index
            WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?
        """,'\n            SELECT request_hash, receipt_ref, receipt_hash, state_ref, state_hash\n            FROM ontology_seed_request_index\n            WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s\n        '),
            (organization_id, workspace_id, actor_ref, operation, request_id),
        ).fetchone()
        if row is None:
            return None
        cls = {
            "select_seed": SeedSelectionReceiptV1,
            "apply_seed": SeedApplyReceiptV1,
        }.get(operation)
        if cls is None:
            raise ValueError("unknown seed request operation")
        receipt = self.get(
            organization_id=organization_id,
            workspace_id=workspace_id,
            artifact_ref=row[1],
            contract_type=cls,
        )
        state = self.get(
            organization_id=organization_id,
            workspace_id=workspace_id,
            artifact_ref=row[3],
            contract_type=OntologySeedStateV1,
        )
        if (
            receipt is None
            or state is None
            or (receipt.request_hash, receipt.receipt_hash, state.state_hash)
            != (row[0], row[2], row[4])
        ):
            raise ValueError(
                "seed request index has missing or corrupt immutable artifacts"
            )
        if (
            state.build_ref,
            state.predecessor_state_ref,
            state.predecessor_state_hash,
        ) != (
            receipt.build_ref,
            receipt.expected_state.artifact_ref,
            receipt.expected_state.artifact_hash,
        ):
            raise ValueError("seed request does not bind resulting state predecessor")
        if (receipt.actor_ref, receipt.operation, receipt.request_id) != (
            actor_ref,
            operation,
            request_id,
        ):
            raise ValueError("seed request index does not bind requested action")
        pin = (
            state.selection_receipt
            if operation == "select_seed"
            else state.latest_apply_receipt
        )
        if pin is None or (pin.artifact_ref, pin.artifact_hash) != (
            receipt.receipt_ref,
            receipt.receipt_hash,
        ):
            raise ValueError("seed request state does not bind receipt")
        return receipt, state

    def list_hub_catalog_page(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        limit: int,
        snapshot_ref: str | None = None,
        profile_ref: str | None = None,
    ):
        """Count in SQL and materialize at most limit exact snapshot/profile rows.

        LEFT JOIN deliberately retains snapshots with no profile; selected
        corrupt profile bindings fail validation rather than disappearing.
        """
        from contracts.ontology_hub_snapshot import OntologyHubSnapshotV1
        from contracts.ontology_seeds import OntologyCapabilityProfileV1

        if not 1 <= limit <= 1000:
            raise ValueError("Hub catalogue limit must be between 1 and 1000")
        relation = sql_query("""
            FROM ontology_build_artifacts AS snapshot
            LEFT JOIN ontology_build_artifacts AS profile
              ON profile.organization_id = snapshot.organization_id
             AND profile.workspace_id = snapshot.workspace_id
             AND profile.schema_version = 'ontology_capability_profile.v1'
             AND json_extract(profile.payload_json, '$.identity.provenance.snapshot_ref') = snapshot.artifact_ref
            WHERE snapshot.organization_id = ? AND snapshot.workspace_id = ?
              AND snapshot.schema_version = 'ontology_hub_snapshot.v1'
        ""","\n            FROM ontology_build_artifacts AS snapshot\n            LEFT JOIN ontology_build_artifacts AS profile\n              ON profile.organization_id = snapshot.organization_id\n             AND profile.workspace_id = snapshot.workspace_id\n             AND profile.schema_version = 'ontology_capability_profile.v1'\n             AND (profile.payload_json::jsonb #>> '{identity,provenance,snapshot_ref}') = snapshot.artifact_ref\n            WHERE snapshot.organization_id = %s AND snapshot.workspace_id = %s\n              AND snapshot.schema_version = 'ontology_hub_snapshot.v1'\n        ")
        params: list[object] = [organization_id, workspace_id]
        if snapshot_ref is not None:
            relation += sql_query(" AND snapshot.artifact_ref = ?",' AND snapshot.artifact_ref = %s')
            params.append(snapshot_ref)
        if profile_ref is not None:
            relation += sql_query(" AND profile.artifact_ref = ?",' AND profile.artifact_ref = %s')
            params.append(profile_ref)
        total = self._connection.execute(
            "SELECT COUNT(*) " + relation, params
        ).fetchone()[0]
        rows = self._connection.execute(
            "SELECT snapshot.payload_json, profile.payload_json, snapshot.artifact_ref, snapshot.artifact_hash, profile.artifact_ref, profile.artifact_hash "
            + relation
            + sql_query(" ORDER BY json_extract(snapshot.payload_json, '$.ontology_ref'), json_extract(snapshot.payload_json, '$.version'), profile.artifact_ref LIMIT ?"," ORDER BY (snapshot.payload_json::jsonb #>> '{ontology_ref}'), (snapshot.payload_json::jsonb #>> '{version}'), profile.artifact_ref LIMIT %s"),
            [*params, limit],
        ).fetchall()
        results = []
        for (
            snapshot_json,
            profile_json,
            stored_snapshot_ref,
            stored_snapshot_hash,
            stored_profile_ref,
            stored_profile_hash,
        ) in rows:
            snapshot = OntologyHubSnapshotV1.model_validate(json.loads(snapshot_json))
            if (
                snapshot.organization_id,
                snapshot.workspace_id,
                snapshot.snapshot_ref,
                snapshot.snapshot_hash,
            ) != (
                organization_id,
                workspace_id,
                stored_snapshot_ref,
                stored_snapshot_hash,
            ):
                raise ValueError("Hub snapshot row binding is corrupt")
            profile = (
                OntologyCapabilityProfileV1.model_validate(json.loads(profile_json))
                if profile_json
                else None
            )
            if profile is not None and (
                profile.organization_id,
                profile.workspace_id,
                profile.profile_ref,
                profile.profile_hash,
                profile.ontology_ref,
                profile.version,
                profile.source_content_hash,
                profile.graph_content_hash,
                profile.identity.provenance.get("snapshot_ref"),
                profile.identity.provenance.get("snapshot_hash"),
            ) != (
                organization_id,
                workspace_id,
                stored_profile_ref,
                stored_profile_hash,
                snapshot.ontology_ref,
                snapshot.version,
                snapshot.source_content_hash,
                snapshot.graph_content_hash,
                snapshot.snapshot_ref,
                snapshot.snapshot_hash,
            ):
                raise ValueError(
                    "capability profile does not bind its exact Hub snapshot"
                )
            results.append((snapshot, profile))
        return tuple(results), total

    def list_hub_snapshots(self, *, organization_id: str, workspace_id: str):
        """Enumerate only validated governed snapshots; no live Hub dependency."""
        from contracts.ontology_hub_snapshot import OntologyHubSnapshotV1

        rows = self._connection.execute(
            sql_query("SELECT payload_json FROM ontology_build_artifacts WHERE organization_id = ? AND workspace_id = ? AND schema_version = ? ORDER BY artifact_ref",'SELECT payload_json FROM ontology_build_artifacts WHERE organization_id = %s AND workspace_id = %s AND schema_version = %s ORDER BY artifact_ref'),
            (organization_id, workspace_id, "ontology_hub_snapshot.v1"),
        ).fetchall()
        return tuple(
            OntologyHubSnapshotV1.model_validate(json.loads(row[0])) for row in rows
        )

    def find_hub_snapshot(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        ontology_ref: str,
        version: str,
    ):
        matches = [
            snapshot
            for snapshot in self.list_hub_snapshots(
                organization_id=organization_id, workspace_id=workspace_id
            )
            if snapshot.ontology_ref == ontology_ref and snapshot.version == version
        ]
        if len(matches) > 1:
            raise ValueError("ambiguous exact ontology Hub version")
        return matches[0] if matches else None

    def list_hub_profiles(
        self, *, organization_id: str, workspace_id: str, snapshot_ref: str
    ):
        from contracts.ontology_hub_snapshot import OntologyHubSnapshotV1
        from contracts.ontology_seeds import OntologyCapabilityProfileV1

        snapshot = self.get(
            organization_id=organization_id,
            workspace_id=workspace_id,
            artifact_ref=snapshot_ref,
            contract_type=OntologyHubSnapshotV1,
        )
        if snapshot is None:
            raise LookupError("ontology Hub snapshot not found")
        rows = self._connection.execute(
            sql_query("SELECT payload_json FROM ontology_build_artifacts WHERE organization_id = ? AND workspace_id = ? AND schema_version = ? ORDER BY artifact_ref",'SELECT payload_json FROM ontology_build_artifacts WHERE organization_id = %s AND workspace_id = %s AND schema_version = %s ORDER BY artifact_ref'),
            (organization_id, workspace_id, "ontology_capability_profile.v1"),
        ).fetchall()
        profiles = []
        for (encoded,) in rows:
            profile = OntologyCapabilityProfileV1.model_validate(json.loads(encoded))
            if profile.identity.provenance.get("snapshot_ref") != snapshot_ref:
                continue
            if (
                profile.ontology_ref,
                profile.version,
                profile.source_content_hash,
                profile.graph_content_hash,
                profile.identity.provenance.get("snapshot_hash"),
            ) != (
                snapshot.ontology_ref,
                snapshot.version,
                snapshot.source_content_hash,
                snapshot.graph_content_hash,
                snapshot.snapshot_hash,
            ):
                raise ValueError(
                    "capability profile does not bind its exact Hub snapshot"
                )
            profiles.append(profile)
        return tuple(
            sorted(
                profiles, key=lambda profile: (profile.revision, profile.profile_ref)
            )
        )

    def close(self) -> None:
        self._connection.close()


__all__ = ["OntologyBuildStore", "ontology_build_store_path"]
