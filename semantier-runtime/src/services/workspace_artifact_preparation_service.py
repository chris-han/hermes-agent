
from storage.hosted_postgres import sql_query,begin_write,in_transaction
"""Correlation-bound preparation adapters using existing source owners."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import BinaryIO, Literal, Protocol

from agents.auth_session import RequestContext
from contracts.knowledge_classification import RuntimeContextDTO
from contracts.canonical_source_ir import CanonicalSourceDocument
from contracts.evidence_location import SourceIdentity
from contracts.ontology_building import ArtifactPinV1
from contracts.ontology_workbench_context import (
    WorkspaceArtifactV1,
    WorkspaceArtifactPreparationClaimV1,
    WorkspaceArtifactPreparationAdapterResultV1,
    PrepareWorkspaceArtifactCommandV1,
    PrepareWorkspaceArtifactResultV1,
    WorkspaceArtifactPreparationV1,
    WorkspaceArtifactPreparationCatalogV1,
    WorkspaceArtifactCollectionV1,
    WorkspaceArtifactOperationReceiptV1,
)
from eos.artifact_hashing import content_hash
from semantier.knowledge_feature_packs.canonical_source_ir_adapter import persist_canonical_source_ir
from semantier.knowledge_feature_packs.canonical_source_ir_native_adapter import adapt_docx_bytes, adapt_pdf_bytes
from services.original_html_source import adapt_html_bytes
from services.runtime_temporal_source_service import RuntimeTemporalSourceService
from services.workspace_artifact_service import WorkspaceArtifactService, _address, _pin
from storage.knowledge_builder_store import KnowledgeBuilderStore
from storage.knowledge_extraction_store import KnowledgeExtractionStore
from storage.workspace_artifact_blob_store import WorkspaceArtifactBlobStore


class RuntimeTargetPreparationAdapter:
    """Bridge eligible governed workspace bytes into the existing P3 temporal graph owner."""

    def __init__(self, *, artifact_service: WorkspaceArtifactService):
        if not isinstance(artifact_service, WorkspaceArtifactService):
            raise TypeError("target preparation requires the workspace artifact owner")
        self._artifacts = artifact_service
        self._build = artifact_service._store

    def _temporal_owner(self, ctx):
        cursors = self._artifacts._cursors
        if cursors is None:
            return None
        scope = cursors._scope(ctx)
        actor = RuntimeContextDTO(
            organization_id=scope["organization_id"],
            workspace_id=scope["workspace_id"],
            actor_id=scope["actor_ref"],
            session_id=scope["session_id"],
        )
        return actor, RuntimeTemporalSourceService(
            build_store=self._build,
            authority_resolver=self._artifacts._authority,
            session_resolver=cursors._sessions,
        )

    def resolve_ready_target(self, *, ctx, artifact_pin, preparation_pin):
        self._artifacts.resolve(ctx=ctx, artifact_pin=artifact_pin)
        preparation = self._artifacts._get(
            ctx,
            preparation_pin,
            WorkspaceArtifactPreparationV1,
            "preparation_hash",
        )
        if (
            preparation.artifact_pin != artifact_pin
            or preparation.purpose != "RUNTIME_TARGET"
            or preparation.outcome != "READY"
        ):
            raise ValueError("TARGET_UNSUPPORTED")
        rows = self._build._connection.execute(
            sql_query("SELECT correlation_ref FROM workspace_artifact_preparation_operation_index WHERE organization_id=? AND workspace_id=? AND preparation_ref=? AND preparation_hash=? AND state='COMPLETE'","SELECT correlation_ref FROM workspace_artifact_preparation_operation_index WHERE organization_id=%s AND workspace_id=%s AND preparation_ref=%s AND preparation_hash=%s AND state='COMPLETE'"),
            (
                ctx.organization_id,
                ctx.workspace_id,
                preparation_pin.artifact_ref,
                preparation_pin.artifact_hash,
            ),
        ).fetchall()
        if len(rows) != 1:
            raise ValueError("PREPARATION_RECONCILIATION_FAILED")
        result = self.exact_get_result(
            ctx=ctx,
            correlation_ref=rows[0][0],
            artifact_pin=artifact_pin,
            purpose="RUNTIME_TARGET",
        )
        if result is None:
            raise ValueError("PREPARATION_RECONCILIATION_FAILED")
        graph_pins = [
            pin for pin in result.derived_artifact_pins
            if pin.artifact_kind == "runtime_temporal_source_graph"
        ]
        owner = self._temporal_owner(ctx)
        if len(graph_pins) != 1 or owner is None:
            raise ValueError("PREPARATION_RECONCILIATION_FAILED")
        actor, temporal = owner
        return temporal.get_graph(actor_context=actor, pin=graph_pins[0])

    def exact_get_result(self, *, ctx, correlation_ref, artifact_pin, purpose):
        self._artifacts.resolve(ctx=ctx, artifact_pin=artifact_pin)
        row = self._build._connection.execute(
            sql_query("SELECT claim_ref,claim_hash,adapter_result_ref,adapter_result_hash FROM workspace_artifact_preparation_operation_index WHERE organization_id=? AND workspace_id=? AND correlation_ref=?",'SELECT claim_ref,claim_hash,adapter_result_ref,adapter_result_hash FROM workspace_artifact_preparation_operation_index WHERE organization_id=%s AND workspace_id=%s AND correlation_ref=%s'),
            (ctx.organization_id, ctx.workspace_id, correlation_ref),
        ).fetchone()
        if row is None or purpose != "RUNTIME_TARGET":
            raise ValueError("PREPARATION_RECONCILIATION_FAILED")
        claim = self._artifacts._get(
            ctx,
            _pin("workspace_artifact_preparation_claim", row[0], row[1]),
            WorkspaceArtifactPreparationClaimV1,
            "claim_hash",
        )
        if (
            claim.artifact_pin,
            claim.purpose,
            claim.correlation_ref,
            claim.actor_ref,
        ) != (
            artifact_pin,
            purpose,
            correlation_ref,
            ctx.user_id,
        ):
            raise ValueError("PREPARATION_RECONCILIATION_FAILED")
        if row[2] is None and row[3] is None:
            return None
        if row[2] is None or row[3] is None:
            raise ValueError("PREPARATION_RECONCILIATION_FAILED")
        result = self._artifacts._get(
            ctx,
            _pin("workspace_artifact_preparation_adapter_result", row[2], row[3]),
            WorkspaceArtifactPreparationAdapterResultV1,
            "adapter_result_hash",
        )
        if (result.artifact_pin, result.purpose, result.correlation_ref) != (
            artifact_pin, purpose, correlation_ref
        ):
            raise ValueError("PREPARATION_RECONCILIATION_FAILED")
        if result.outcome == "UNSUPPORTED":
            if result.failure_code != "TARGET_UNSUPPORTED" or result.derived_artifact_pins:
                raise ValueError("PREPARATION_RECONCILIATION_FAILED")
        elif result.outcome == "READY":
            if result.failure_code is not None or result.source_identity_pin is not None or result.canonical_source_ir_pin is not None:
                raise ValueError("PREPARATION_RECONCILIATION_FAILED")
            graph_pins = [pin for pin in result.derived_artifact_pins if pin.artifact_kind == "runtime_temporal_source_graph"]
            owner = self._temporal_owner(ctx)
            if len(graph_pins) != 1 or owner is None:
                raise ValueError("PREPARATION_RECONCILIATION_FAILED")
            actor, temporal = owner
            temporal.get_graph(actor_context=actor, pin=graph_pins[0])
        else:
            raise ValueError("PREPARATION_RECONCILIATION_FAILED")
        return result

    def prepare(self, *, ctx, correlation_ref, artifact, byte_stream):
        pin = _pin("workspace_artifact", artifact.artifact_ref, artifact.artifact_hash)
        saved = self.exact_get_result(
            ctx=ctx,
            correlation_ref=correlation_ref,
            artifact_pin=pin,
            purpose="RUNTIME_TARGET",
        )
        if saved is not None:
            return saved
        if in_transaction(self._build._connection):
            raise RuntimeError(
                "preparation adapter cannot run under the workspace write lock"
            )
        current = self._artifacts.resolve(ctx=ctx, artifact_pin=pin)
        data = byte_stream.read(current.byte_size + 1)
        if (
            current != artifact
            or len(data) != current.byte_size
            or "sha256:" + sha256(data).hexdigest() != current.content_hash
        ):
            raise ValueError("ARTIFACT_PIN_MISMATCH")
        outcome = "UNSUPPORTED"
        failure_code = "TARGET_UNSUPPORTED"
        derived_pins = []
        owner = self._temporal_owner(ctx)
        if current.media_type == "application/x-ipynb+json" and owner is not None:
            actor, temporal = owner
            archive = temporal.capture_notebook(
                actor_context=actor,
                content=data,
                request_id=f"{correlation_ref}:capture",
            )
            graph = temporal.import_notebook_source(
                actor_context=actor,
                source=archive.artifact_pin(),
                request_id=f"{correlation_ref}:import",
            )
            outcome = "READY"
            failure_code = None
            derived_pins = sorted(
                (archive.artifact_pin(), graph.normalization, graph.artifact_pin()),
                key=lambda value: (value.artifact_kind, value.artifact_ref, value.artifact_hash),
            )
        result = _address(
            WorkspaceArtifactPreparationAdapterResultV1,
            "workspace-artifact-preparation-adapter-result",
            "adapter_result_ref",
            "adapter_result_hash",
            dict(
                schema_version="workspace_artifact_preparation_adapter_result.v1",
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                artifact_pin=pin.model_dump(mode="json"),
                purpose="RUNTIME_TARGET",
                correlation_ref=correlation_ref,
                outcome=outcome,
                source_identity_pin=None,
                canonical_source_ir_pin=None,
                derived_artifact_pins=[value.model_dump(mode="json") for value in derived_pins],
                failure_code=failure_code,
                completed_at=datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            ),
        )
        winner = result

        def validate(connection):
            nonlocal winner
            saved = self.exact_get_result(
                ctx=ctx,
                correlation_ref=correlation_ref,
                artifact_pin=pin,
                purpose="RUNTIME_TARGET",
            )
            if saved is not None:
                winner = saved
                return False
            return True

        def mutate(connection):
            changed = connection.execute(
                sql_query("UPDATE workspace_artifact_preparation_operation_index SET adapter_result_ref=?,adapter_result_hash=? WHERE organization_id=? AND workspace_id=? AND correlation_ref=? AND adapter_result_ref IS NULL AND state='PENDING'","UPDATE workspace_artifact_preparation_operation_index SET adapter_result_ref=%s,adapter_result_hash=%s WHERE organization_id=%s AND workspace_id=%s AND correlation_ref=%s AND adapter_result_ref IS NULL AND state='PENDING'"),
                (
                    result.adapter_result_ref,
                    result.adapter_result_hash,
                    ctx.organization_id,
                    ctx.workspace_id,
                    correlation_ref,
                ),
            ).rowcount
            if changed != 1:
                raise ValueError("PREPARATION_RECONCILIATION_FAILED")

        self._build.atomic_write(
            artifacts=(result,),
            validate_before_insert=validate,
            mutate_indexes=mutate,
            verify_before_commit=lambda: self._artifacts.resolve(
                ctx=ctx, artifact_pin=pin
            ),
        )
        return winner


class WorkspaceArtifactPreparationService:
    def __init__(self, *, artifact_service: WorkspaceArtifactService, design_adapter):
        if not isinstance(artifact_service, WorkspaceArtifactService) or not isinstance(
            design_adapter, KnowledgeDesignSourcePreparationAdapter
        ):
            raise TypeError("preparation requires its real artifact and design owners")
        self._artifacts = artifact_service
        self._build = artifact_service._store
        if design_adapter._purpose != "DESIGN_SOURCE":
            raise ValueError("DESIGN_SOURCE_ADAPTER_REQUIRED")
        self._design = design_adapter
        self._document_target = KnowledgeDesignSourcePreparationAdapter(
            artifact_service=artifact_service, knowledge_store=design_adapter._knowledge,
            extraction_store=design_adapter._extraction, purpose="RUNTIME_TARGET",
        )
        self._target = RuntimeTargetPreparationAdapter(
            artifact_service=artifact_service
        )

    def _adapter(self, purpose, *, ctx, artifact_pin):
        if purpose == "DESIGN_SOURCE":
            return self._design
        artifact = self._artifacts.resolve(ctx=ctx, artifact_pin=artifact_pin)
        if artifact.media_type in {"application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/pdf"}:
            return self._document_target
        return self._target

    def resolve(self, *, ctx, artifact_pin, preparation_pin, purpose):
        """Resolve a workspace preparation for binding, without rerunning it."""
        preparation, _ = self._resolve(ctx=ctx, artifact_pin=artifact_pin, preparation_pin=preparation_pin, purpose=purpose)
        return preparation

    def resolve_design_source(self, *, ctx, artifact_pin, preparation_pin, expected_ir_pin=None):
        """Return the exact verified native identity/IR pair for source consumers."""
        preparation, source = self._resolve(ctx=ctx, artifact_pin=artifact_pin, preparation_pin=preparation_pin, purpose="DESIGN_SOURCE")
        if expected_ir_pin is not None and preparation.canonical_source_ir_pin != expected_ir_pin:
            raise ValueError("ARTIFACT_PIN_MISMATCH")
        return source

    def resolve_runtime_target(self, *, ctx, artifact_pin, preparation_pin):
        """Return the prepared DOCX identity/IR pair or notebook temporal graph."""
        _preparation, target = self._resolve(
            ctx=ctx,
            artifact_pin=artifact_pin,
            preparation_pin=preparation_pin,
            purpose="RUNTIME_TARGET",
        )
        return target

    def _resolve(self, *, ctx, artifact_pin, preparation_pin, purpose):
        self._artifacts.resolve(ctx=ctx, artifact_pin=artifact_pin)
        if (
            purpose not in ("DESIGN_SOURCE", "RUNTIME_TARGET")
            or preparation_pin.artifact_kind != "workspace_artifact_preparation"
        ):
            raise ValueError("ARTIFACT_PIN_MISMATCH")
        preparation = self._artifacts._get(
            ctx, preparation_pin, WorkspaceArtifactPreparationV1, "preparation_hash"
        )
        if preparation.artifact_pin != artifact_pin or preparation.purpose != purpose:
            raise ValueError("ARTIFACT_PIN_MISMATCH")
        if preparation.outcome != "READY":
            raise ValueError(
                "SOURCE_PREPARATION_FAILED"
                if purpose == "DESIGN_SOURCE"
                else "TARGET_UNSUPPORTED"
            )
        rows = self._build._connection.execute(
            sql_query("SELECT correlation_ref,adapter_result_ref,adapter_result_hash FROM workspace_artifact_preparation_operation_index WHERE organization_id=? AND workspace_id=? AND preparation_ref=? AND preparation_hash=? AND state='COMPLETE'","SELECT correlation_ref,adapter_result_ref,adapter_result_hash FROM workspace_artifact_preparation_operation_index WHERE organization_id=%s AND workspace_id=%s AND preparation_ref=%s AND preparation_hash=%s AND state='COMPLETE'"),
            (
                ctx.organization_id,
                ctx.workspace_id,
                preparation_pin.artifact_ref,
                preparation_pin.artifact_hash,
            ),
        ).fetchall()
        if len(rows) != 1:
            raise ValueError("PREPARATION_RECONCILIATION_FAILED")
        correlation, ref, digest = rows[0]
        result = self._artifacts._get(
            ctx,
            _pin("workspace_artifact_preparation_adapter_result", ref, digest),
            WorkspaceArtifactPreparationAdapterResultV1,
            "adapter_result_hash",
        )
        if (
            result.correlation_ref != correlation
            or result.purpose != purpose
            or result.artifact_pin != artifact_pin
            or any(
                getattr(result, field) != getattr(preparation, field)
                for field in (
                    "outcome",
                    "source_identity_pin",
                    "canonical_source_ir_pin",
                    "derived_artifact_pins",
                    "failure_code",
                )
            )
        ):
            raise ValueError("PREPARATION_RECONCILIATION_FAILED")
        adapter = self._adapter(purpose, ctx=ctx, artifact_pin=artifact_pin)
        if adapter is self._target:
            target = self._target.resolve_ready_target(
                ctx=ctx,
                artifact_pin=artifact_pin,
                preparation_pin=preparation_pin,
            )
            self._artifacts._access(ctx)
            return preparation, target
        source = adapter._verify_ready_result(ctx, result)
        self._artifacts._access(ctx)
        return preparation, source

    def prepare(self, *, ctx, command):
        command = PrepareWorkspaceArtifactCommandV1.model_validate(
            command.model_dump(mode="json")
        )
        claim = self._build.workspace_artifact_preparation_phase(
            service=self, ctx=ctx, command=command, phase="CLAIM"
        )
        artifact = self._artifacts.resolve(ctx=ctx, artifact_pin=command.artifact_pin)
        adapter = self._adapter(command.purpose, ctx=ctx, artifact_pin=command.artifact_pin)
        saved = adapter.exact_get_result(
            ctx=ctx,
            correlation_ref=claim.correlation_ref,
            artifact_pin=command.artifact_pin,
            purpose=command.purpose,
        )
        if saved is None:
            blobs = WorkspaceArtifactBlobStore(
                ctx=ctx, max_bytes=self._artifacts._max_bytes
            )
            with blobs.exact_get(
                byte_owner_ref=artifact.byte_owner_ref,
                expected_hash=artifact.content_hash,
                expected_size=artifact.byte_size,
            ) as stream:
                adapter.prepare(
                    ctx=ctx,
                    correlation_ref=claim.correlation_ref,
                    artifact=artifact,
                    byte_stream=stream,
                )
        return self._build.workspace_artifact_preparation_phase(
            service=self, ctx=ctx, command=command, phase="FINALIZE"
        )

    def _claim_on_connection(self, connection, ctx, command):
        self._artifacts.resolve(ctx=ctx, artifact_pin=command.artifact_pin)
        revision = (
            0
            if command.expected_preparation_revision is None
            else command.expected_preparation_revision + 1
        )
        correlation = "workspace-artifact-prepare:" + content_hash(
            dict(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                artifact_pin=command.artifact_pin.model_dump(mode="json"),
                purpose=command.purpose,
                successor_revision=revision,
                predecessor_pin=command.expected_preparation_pin.model_dump(mode="json")
                if command.expected_preparation_pin
                else None,
            )
        )
        request_hash = "sha256:" + content_hash(command.model_dump(mode="json"))
        scope = (ctx.organization_id, ctx.workspace_id)
        public = connection.execute(
            sql_query("SELECT request_hash FROM workspace_operation_request_index WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation='PREPARE_ARTIFACT' AND request_id=?","SELECT request_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation='PREPARE_ARTIFACT' AND request_id=%s"),
            (*scope, ctx.user_id, command.request_id),
        ).fetchone()
        if public is not None and public[0] != request_hash:
            raise ValueError("REQUEST_CONFLICT")
        row = connection.execute(
            sql_query("SELECT claim_ref,claim_hash,operation_key,request_id,request_hash FROM workspace_artifact_preparation_operation_index WHERE organization_id=? AND workspace_id=? AND correlation_ref=?",'SELECT claim_ref,claim_hash,operation_key,request_id,request_hash FROM workspace_artifact_preparation_operation_index WHERE organization_id=%s AND workspace_id=%s AND correlation_ref=%s'),
            (*scope, correlation),
        ).fetchone()
        if row is not None:
            claim = self._artifacts._get(
                ctx,
                _pin("workspace_artifact_preparation_claim", row[0], row[1]),
                WorkspaceArtifactPreparationClaimV1,
                "claim_hash",
            )
            if (
                tuple(row[2:])
                != (claim.operation_key, claim.request_id, claim.request_hash)
                or claim.correlation_ref != correlation
                or claim.operation_key != correlation + f":r{revision}"
                or claim.artifact_pin != command.artifact_pin
                or claim.purpose != command.purpose
                or claim.expected_preparation_pin != command.expected_preparation_pin
                or claim.expected_preparation_revision
                != command.expected_preparation_revision
            ):
                raise ValueError("PREPARATION_RECONCILIATION_FAILED")
            if (
                claim.request_hash != request_hash
                or claim.actor_ref != ctx.user_id
                or claim.request_id != command.request_id
            ):
                raise ValueError("REQUEST_CONFLICT")
            return claim
        if public is not None:
            raise ValueError("PREPARATION_RECONCILIATION_FAILED")
        self._check_predecessor(connection, ctx, command)
        claim = _address(
            WorkspaceArtifactPreparationClaimV1,
            "workspace-artifact-preparation-claim",
            "claim_ref",
            "claim_hash",
            dict(
                schema_version="workspace_artifact_preparation_claim.v1",
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                artifact_pin=command.artifact_pin.model_dump(mode="json"),
                purpose=command.purpose,
                expected_preparation_pin=command.expected_preparation_pin.model_dump(
                    mode="json"
                )
                if command.expected_preparation_pin
                else None,
                expected_preparation_revision=command.expected_preparation_revision,
                correlation_ref=correlation,
                operation_key=correlation + f":r{revision}",
                request_id=command.request_id,
                request_hash=request_hash,
                actor_ref=ctx.user_id,
                state="PENDING",
                claimed_at=datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            ),
        )
        self._build._insert_prepared((self._build._prepare(claim),))
        connection.execute(
            sql_query("INSERT INTO workspace_artifact_preparation_operation_index (organization_id,workspace_id,correlation_ref,operation_key,request_id,request_hash,claim_ref,claim_hash,state) VALUES (?,?,?,?,?,?,?,?,?)",'INSERT INTO workspace_artifact_preparation_operation_index (organization_id,workspace_id,correlation_ref,operation_key,request_id,request_hash,claim_ref,claim_hash,state) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)'),
            (
                *scope,
                correlation,
                claim.operation_key,
                command.request_id,
                request_hash,
                claim.claim_ref,
                claim.claim_hash,
                "PENDING",
            ),
        )
        return claim

    def _check_predecessor(self, connection, ctx, command):
        row = connection.execute(
            sql_query("SELECT revision,head_ref,head_hash FROM workspace_artifact_preparation_heads WHERE organization_id=? AND workspace_id=? AND artifact_ref=? AND purpose=?",'SELECT revision,head_ref,head_hash FROM workspace_artifact_preparation_heads WHERE organization_id=%s AND workspace_id=%s AND artifact_ref=%s AND purpose=%s'),
            (
                ctx.organization_id,
                ctx.workspace_id,
                command.artifact_pin.artifact_ref,
                command.purpose,
            ),
        ).fetchone()
        if row is None:
            if command.expected_preparation_pin is not None:
                raise ValueError("STALE_PREPARATION_HEAD")
            return
        if command.expected_preparation_pin is None or (
            row[0],
            _pin("workspace_artifact_preparation", row[1], row[2]),
        ) != (command.expected_preparation_revision, command.expected_preparation_pin):
            raise ValueError("STALE_PREPARATION_HEAD")
        previous = self._artifacts._get(
            ctx,
            command.expected_preparation_pin,
            WorkspaceArtifactPreparationV1,
            "preparation_hash",
        )
        if (
            previous.artifact_pin != command.artifact_pin
            or previous.purpose != command.purpose
            or previous.revision != row[0]
        ):
            raise ValueError("PREPARATION_RECONCILIATION_FAILED")

    def _result(self, ctx, receipt, preparation):
        return _address(
            PrepareWorkspaceArtifactResultV1,
            "workspace-artifact-prepare-result",
            "result_ref",
            "result_hash",
            dict(
                schema_version="prepare_workspace_artifact_result.v1",
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                artifact_pin=receipt.artifact_pin.model_dump(mode="json"),
                preparation_pin=receipt.preparation_pin.model_dump(mode="json"),
                preparation_revision=preparation.revision,
                preparation_catalog_pin=receipt.preparation_catalog_pin.model_dump(
                    mode="json"
                ),
                preparation_catalog_revision=receipt.preparation_catalog_revision,
                predecessor_pin=preparation.predecessor_pin.model_dump(mode="json")
                if preparation.predecessor_pin
                else None,
                receipt_pin=_pin(
                    "workspace_artifact_operation_receipt",
                    receipt.receipt_ref,
                    receipt.receipt_hash,
                ).model_dump(mode="json"),
                outcome=preparation.outcome,
                recorded_at=receipt.recorded_at.isoformat().replace("+00:00", "Z"),
            ),
        )

    def _finalize_on_connection(self, connection, ctx, command):
        claim = self._claim_on_connection(connection, ctx, command)
        scope = (ctx.organization_id, ctx.workspace_id)
        adapter = self._adapter(command.purpose, ctx=ctx, artifact_pin=command.artifact_pin).exact_get_result(
            ctx=ctx,
            correlation_ref=claim.correlation_ref,
            artifact_pin=command.artifact_pin,
            purpose=command.purpose,
        )
        if adapter is None:
            raise ValueError("PREPARATION_RECONCILIATION_FAILED")
        row = connection.execute(
            sql_query("SELECT state,preparation_ref,preparation_hash,receipt_ref,receipt_hash FROM workspace_artifact_preparation_operation_index WHERE organization_id=? AND workspace_id=? AND correlation_ref=?",'SELECT state,preparation_ref,preparation_hash,receipt_ref,receipt_hash FROM workspace_artifact_preparation_operation_index WHERE organization_id=%s AND workspace_id=%s AND correlation_ref=%s'),
            (*scope, claim.correlation_ref),
        ).fetchone()
        if row[0] == "COMPLETE":
            preparation = self._artifacts._get(
                ctx,
                _pin("workspace_artifact_preparation", row[1], row[2]),
                WorkspaceArtifactPreparationV1,
                "preparation_hash",
            )
            receipt = self._artifacts._get(
                ctx,
                _pin("workspace_artifact_operation_receipt", row[3], row[4]),
                WorkspaceArtifactOperationReceiptV1,
                "receipt_hash",
            )
            public = connection.execute(
                sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation='PREPARE_ARTIFACT' AND request_id=?","SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation='PREPARE_ARTIFACT' AND request_id=%s"),
                (*scope, ctx.user_id, command.request_id),
            ).fetchone()
            if (
                public is None
                or tuple(public)
                != (claim.request_hash, receipt.receipt_ref, receipt.receipt_hash)
                or receipt.operation != "PREPARE_ARTIFACT"
                or receipt.actor_ref != ctx.user_id
                or receipt.request_id != command.request_id
                or receipt.request_hash != claim.request_hash
                or receipt.artifact_pin != command.artifact_pin
                or receipt.preparation_pin
                != _pin("workspace_artifact_preparation", row[1], row[2])
                or preparation.artifact_pin != command.artifact_pin
                or preparation.purpose != command.purpose
                or preparation.predecessor_pin != command.expected_preparation_pin
                or preparation.revision
                != (
                    0
                    if command.expected_preparation_revision is None
                    else command.expected_preparation_revision + 1
                )
                or any(
                    getattr(preparation, field) != getattr(adapter, field)
                    for field in (
                        "outcome",
                        "source_identity_pin",
                        "canonical_source_ir_pin",
                        "derived_artifact_pins",
                        "failure_code",
                    )
                )
            ):
                raise ValueError("PREPARATION_RECONCILIATION_FAILED")
            catalog = self._artifacts._get(
                ctx,
                receipt.preparation_catalog_pin,
                WorkspaceArtifactPreparationCatalogV1,
                "catalog_hash",
            )
            collection = self._artifacts._get(
                ctx,
                receipt.collection_pin,
                WorkspaceArtifactCollectionV1,
                "collection_hash",
            )
            if (
                catalog.revision != receipt.preparation_catalog_revision
                or receipt.preparation_pin not in catalog.preparation_pins
                or collection.revision != receipt.collection_revision
                or command.artifact_pin not in collection.artifact_pins
            ):
                raise ValueError("PREPARATION_RECONCILIATION_FAILED")
            result = self._result(ctx, receipt, preparation)
            return self._artifacts._get(
                ctx,
                _pin(
                    "prepare_workspace_artifact_result",
                    result.result_ref,
                    result.result_hash,
                ),
                PrepareWorkspaceArtifactResultV1,
                "result_hash",
            )
        self._check_predecessor(connection, ctx, command)
        collection = self._artifacts._head(
            ctx,
            "workspace_artifact_collection_heads",
            WorkspaceArtifactCollectionV1,
            "collection_ref",
            "collection_hash",
        )
        if command.artifact_pin not in collection.artifact_pins:
            raise ValueError("ARTIFACT_NOT_FOUND")
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        base = dict(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        revision = (
            0
            if command.expected_preparation_revision is None
            else command.expected_preparation_revision + 1
        )
        preparation = _address(
            WorkspaceArtifactPreparationV1,
            "workspace-artifact-preparation",
            "preparation_ref",
            "preparation_hash",
            dict(
                **base,
                schema_version="workspace_artifact_preparation.v1",
                artifact_pin=command.artifact_pin.model_dump(mode="json"),
                purpose=command.purpose,
                outcome=adapter.outcome,
                source_identity_pin=adapter.source_identity_pin.model_dump(mode="json")
                if adapter.source_identity_pin
                else None,
                canonical_source_ir_pin=adapter.canonical_source_ir_pin.model_dump(
                    mode="json"
                )
                if adapter.canonical_source_ir_pin
                else None,
                derived_artifact_pins=[
                    p.model_dump(mode="json") for p in adapter.derived_artifact_pins
                ],
                predecessor_pin=command.expected_preparation_pin.model_dump(mode="json")
                if command.expected_preparation_pin
                else None,
                revision=revision,
                failure_code=adapter.failure_code,
                prepared_by=ctx.user_id,
                prepared_at=now,
            ),
        )
        prep_pin = _pin(
            "workspace_artifact_preparation",
            preparation.preparation_ref,
            preparation.preparation_hash,
        )
        catalog_row = connection.execute(
            sql_query("SELECT 1 FROM workspace_artifact_preparation_catalog_heads WHERE organization_id=? AND workspace_id=?",'SELECT 1 FROM workspace_artifact_preparation_catalog_heads WHERE organization_id=%s AND workspace_id=%s'),
            scope,
        ).fetchone()
        if catalog_row is None:
            catalog = _address(
                WorkspaceArtifactPreparationCatalogV1,
                "workspace-artifact-preparation-catalog",
                "catalog_ref",
                "catalog_hash",
                dict(
                    **base,
                    schema_version="workspace_artifact_preparation_catalog.v1",
                    revision=0,
                    preparation_pins=[],
                    predecessor_pin=None,
                    recorded_at=now,
                ),
            )
            self._build._insert_prepared((self._build._prepare(catalog),))
        else:
            catalog = self._artifacts._head(
                ctx,
                "workspace_artifact_preparation_catalog_heads",
                WorkspaceArtifactPreparationCatalogV1,
                "catalog_ref",
                "catalog_hash",
            )
        pins = sorted(
            [
                p
                for p in catalog.preparation_pins
                if p != command.expected_preparation_pin
            ]
            + [prep_pin],
            key=lambda p: (p.artifact_kind, p.artifact_ref, p.artifact_hash),
        )
        catalog = _address(
            WorkspaceArtifactPreparationCatalogV1,
            "workspace-artifact-preparation-catalog",
            "catalog_ref",
            "catalog_hash",
            dict(
                **base,
                schema_version="workspace_artifact_preparation_catalog.v1",
                revision=catalog.revision + 1,
                preparation_pins=[p.model_dump(mode="json") for p in pins],
                predecessor_pin=_pin(
                    "workspace_artifact_preparation_catalog",
                    catalog.catalog_ref,
                    catalog.catalog_hash,
                ).model_dump(mode="json"),
                recorded_at=now,
            ),
        )
        receipt = _address(
            WorkspaceArtifactOperationReceiptV1,
            "workspace-artifact-receipt",
            "receipt_ref",
            "receipt_hash",
            dict(
                **base,
                schema_version="workspace_artifact_operation_receipt.v1",
                operation="PREPARE_ARTIFACT",
                artifact_pin=command.artifact_pin.model_dump(mode="json"),
                collection_pin=_pin(
                    "workspace_artifact_collection",
                    collection.collection_ref,
                    collection.collection_hash,
                ).model_dump(mode="json"),
                collection_revision=collection.revision,
                preparation_catalog_pin=_pin(
                    "workspace_artifact_preparation_catalog",
                    catalog.catalog_ref,
                    catalog.catalog_hash,
                ).model_dump(mode="json"),
                preparation_catalog_revision=catalog.revision,
                preparation_pin=prep_pin.model_dump(mode="json"),
                selection_pin=None,
                selection_revision=None,
                binding_set_pin=None,
                binding_set_revision=None,
                material_context_revision=None,
                cursor_pin=None,
                ingress=command.ingress,
                request_id=command.request_id,
                request_hash=claim.request_hash,
                idempotency_key=command.idempotency_key,
                actor_ref=ctx.user_id,
                outcome="PREPARED"
                if adapter.outcome == "READY"
                else "PREPARATION_FAILED",
                recorded_at=now,
            ),
        )
        result = self._result(ctx, receipt, preparation)
        self._build._insert_prepared(
            tuple(
                self._build._prepare(value)
                for value in (preparation, catalog, receipt, result)
            )
        )
        connection.execute(
            sql_query("INSERT INTO workspace_artifact_preparation_heads VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(organization_id,workspace_id,artifact_ref,purpose) DO UPDATE SET revision=excluded.revision,head_ref=excluded.head_ref,head_hash=excluded.head_hash,recorded_at=excluded.recorded_at",'INSERT INTO workspace_artifact_preparation_heads VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(organization_id,workspace_id,artifact_ref,purpose) DO UPDATE SET revision=excluded.revision,head_ref=excluded.head_ref,head_hash=excluded.head_hash,recorded_at=excluded.recorded_at'),
            (
                *scope,
                command.artifact_pin.artifact_ref,
                command.purpose,
                revision,
                preparation.preparation_ref,
                preparation.preparation_hash,
                now,
            ),
        )
        connection.execute(
            sql_query("INSERT INTO workspace_artifact_preparation_catalog_heads VALUES (?,?,?,?,?,?) ON CONFLICT(organization_id,workspace_id) DO UPDATE SET revision=excluded.revision,head_ref=excluded.head_ref,head_hash=excluded.head_hash,recorded_at=excluded.recorded_at",'INSERT INTO workspace_artifact_preparation_catalog_heads VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT(organization_id,workspace_id) DO UPDATE SET revision=excluded.revision,head_ref=excluded.head_ref,head_hash=excluded.head_hash,recorded_at=excluded.recorded_at'),
            (*scope, catalog.revision, catalog.catalog_ref, catalog.catalog_hash, now),
        )
        connection.execute(
            sql_query("INSERT INTO workspace_operation_request_index VALUES (?,?,?,?,?,?,?,?)",'INSERT INTO workspace_operation_request_index VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'),
            (
                *scope,
                ctx.user_id,
                "PREPARE_ARTIFACT",
                command.request_id,
                claim.request_hash,
                receipt.receipt_ref,
                receipt.receipt_hash,
            ),
        )
        connection.execute(
            sql_query("UPDATE workspace_artifact_preparation_operation_index SET state='COMPLETE',preparation_ref=?,preparation_hash=?,receipt_ref=?,receipt_hash=? WHERE organization_id=? AND workspace_id=? AND correlation_ref=?","UPDATE workspace_artifact_preparation_operation_index SET state='COMPLETE',preparation_ref=%s,preparation_hash=%s,receipt_ref=%s,receipt_hash=%s WHERE organization_id=%s AND workspace_id=%s AND correlation_ref=%s"),
            (
                preparation.preparation_ref,
                preparation.preparation_hash,
                receipt.receipt_ref,
                receipt.receipt_hash,
                *scope,
                claim.correlation_ref,
            ),
        )
        return result


class WorkspaceArtifactPreparationAdapter(Protocol):
    def exact_get_result(
        self,
        *,
        ctx: RequestContext,
        correlation_ref: str,
        artifact_pin: ArtifactPinV1,
        purpose: Literal["DESIGN_SOURCE", "RUNTIME_TARGET"],
    ) -> WorkspaceArtifactPreparationAdapterResultV1 | None: ...
    def prepare(
        self,
        *,
        ctx: RequestContext,
        correlation_ref: str,
        artifact: WorkspaceArtifactV1,
        byte_stream: BinaryIO,
    ) -> WorkspaceArtifactPreparationAdapterResultV1: ...


class KnowledgeDesignSourcePreparationAdapter:
    def __init__(
        self,
        *,
        artifact_service: WorkspaceArtifactService,
        knowledge_store: KnowledgeBuilderStore,
        extraction_store: KnowledgeExtractionStore,
        purpose: Literal["DESIGN_SOURCE", "RUNTIME_TARGET"] = "DESIGN_SOURCE",
    ):
        if (
            not isinstance(artifact_service, WorkspaceArtifactService)
            or not isinstance(knowledge_store, KnowledgeBuilderStore)
            or not isinstance(extraction_store, KnowledgeExtractionStore)
        ):
            raise TypeError(
                "DOCX preparation requires real workspace and source owners"
            )
        if purpose not in ("DESIGN_SOURCE", "RUNTIME_TARGET"):
            raise ValueError("INVALID_PREPARATION_PURPOSE")
        self._purpose = purpose
        self._artifacts = artifact_service
        self._build = artifact_service._store
        self._knowledge = knowledge_store
        self._extraction = extraction_store

    def _source_id(self, ctx, correlation_ref, artifact_pin):
        return "kbs_" + content_hash(
            dict(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                artifact_pin=artifact_pin.model_dump(mode="json"),
                purpose=self._purpose,
                correlation_ref=correlation_ref,
            )
        )

    def _claim(self, ctx, correlation_ref, artifact_pin):
        self._artifacts._access(ctx)
        row = self._build._connection.execute(
            sql_query("SELECT claim_ref,claim_hash,adapter_result_ref,adapter_result_hash FROM workspace_artifact_preparation_operation_index WHERE organization_id=? AND workspace_id=? AND correlation_ref=?",'SELECT claim_ref,claim_hash,adapter_result_ref,adapter_result_hash FROM workspace_artifact_preparation_operation_index WHERE organization_id=%s AND workspace_id=%s AND correlation_ref=%s'),
            (ctx.organization_id, ctx.workspace_id, correlation_ref),
        ).fetchone()
        if row is None:
            raise ValueError("PREPARATION_RECONCILIATION_FAILED")
        claim = self._artifacts._get(
            ctx,
            _pin("workspace_artifact_preparation_claim", row[0], row[1]),
            WorkspaceArtifactPreparationClaimV1,
            "claim_hash",
        )
        if (
            claim.artifact_pin != artifact_pin
            or claim.purpose != self._purpose
            or claim.correlation_ref != correlation_ref
            or claim.actor_ref != ctx.user_id
        ):
            raise ValueError("PREPARATION_RECONCILIATION_FAILED")
        return row

    def exact_get_result(self, *, ctx, correlation_ref, artifact_pin, purpose):
        if purpose != self._purpose:
            raise ValueError("PREPARATION_RECONCILIATION_FAILED")
        row = self._claim(ctx, correlation_ref, artifact_pin)
        self._artifacts.resolve(ctx=ctx, artifact_pin=artifact_pin)
        if row[2] is None and row[3] is None:
            return None
        if row[2] is None or row[3] is None:
            raise ValueError("PREPARATION_RECONCILIATION_FAILED")
        result = self._artifacts._get(
            ctx,
            _pin("workspace_artifact_preparation_adapter_result", row[2], row[3]),
            WorkspaceArtifactPreparationAdapterResultV1,
            "adapter_result_hash",
        )
        if (
            result.artifact_pin != artifact_pin
            or result.correlation_ref != correlation_ref
            or result.purpose != purpose
        ):
            raise ValueError("PREPARATION_RECONCILIATION_FAILED")
        self._verify_ready_result(ctx, result)
        return result

    def _verify_ready_result(self, ctx, result):
        if result.outcome == "READY":
            if (
                result.source_identity_pin is None
                or result.canonical_source_ir_pin is None
                or result.source_identity_pin.artifact_kind != "source_identity"
                or result.canonical_source_ir_pin.artifact_kind != "canonical_source_ir"
            ):
                raise ValueError("PREPARATION_RECONCILIATION_FAILED")
            identity = self._knowledge.get_source_identity(
                source_identity_ref=result.source_identity_pin.artifact_ref,
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                lineage_id=self._source_id(
                    ctx, result.correlation_ref, result.artifact_pin
                ),
            )
            if (
                identity is None
                or "sha256:"
                + content_hash(
                    identity.model_dump(mode="json", by_alias=True, exclude_none=False)
                )
                != result.source_identity_pin.artifact_hash
            ):
                raise ValueError("PREPARATION_RECONCILIATION_FAILED")
            path = ctx.workspace_root / result.canonical_source_ir_pin.artifact_ref
            resolved = path.resolve()
            if (
                not resolved.is_relative_to(ctx.workspace_root.resolve())
                or path.is_symlink()
            ):
                raise ValueError("PREPARATION_RECONCILIATION_FAILED")
            document = CanonicalSourceDocument.model_validate(
                json.loads(path.read_text())
            )
            if (
                document.source_identity_ref != identity.source_identity_ref
                or "sha256:"
                + content_hash(
                    document.model_dump(mode="json", by_alias=True, exclude_none=False)
                )
                != result.canonical_source_ir_pin.artifact_hash
            ):
                raise ValueError("PREPARATION_RECONCILIATION_FAILED")
            return identity, document

    def prepare(self, *, ctx, correlation_ref, artifact, byte_stream):
        pin = _pin("workspace_artifact", artifact.artifact_ref, artifact.artifact_hash)
        saved = self.exact_get_result(
            ctx=ctx,
            correlation_ref=correlation_ref,
            artifact_pin=pin,
            purpose=self._purpose,
        )
        if saved is not None:
            return saved
        if in_transaction(self._build._connection):
            raise RuntimeError(
                "preparation adapter cannot run under the workspace write lock"
            )
        current = self._artifacts.resolve(ctx=ctx, artifact_pin=pin)
        if (
            current != artifact
            or self._knowledge.workspace_root != ctx.workspace_root.resolve()
        ):
            raise ValueError("PREPARATION_RECONCILIATION_FAILED")
        data = byte_stream.read(artifact.byte_size + 1)
        if (
            len(data) != artifact.byte_size
            or "sha256:" + sha256(data).hexdigest() != artifact.content_hash
        ):
            raise ValueError("ARTIFACT_PIN_MISMATCH")
        identity_pin = ir_pin = None
        outcome, failure = "UNSUPPORTED", "UNSUPPORTED_SOURCE_PREPARATION"
        if artifact.media_type in {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/pdf",
            "text/html",
        }:
            source_id = self._source_id(ctx, correlation_ref, pin)
            identity_ref = "governed-source:" + source_id
            try:
                document = (
                    adapt_html_bytes(data, source_identity_ref=identity_ref)
                    if artifact.media_type == "text/html" else adapt_pdf_bytes(
                        data,
                        source_identity_ref=identity_ref,
                        media_type=artifact.media_type,
                    )
                    if artifact.media_type == "application/pdf"
                    else adapt_docx_bytes(
                        data,
                        source_identity_ref=identity_ref,
                        media_type=artifact.media_type,
                    )
                )
            except ValueError:
                outcome, failure = "FAILED", "SOURCE_PREPARATION_FAILED"
            else:
                source = self._knowledge.create_source(
                    source_id=source_id,
                    organization_id=ctx.organization_id,
                    workspace_id=ctx.workspace_id,
                    source_kind="file",
                    source_ref=correlation_ref,
                    source_text=artifact.byte_owner_ref,
                    source_metadata={
                        "original_path": artifact.byte_owner_ref,
                        "original_content_hash": artifact.content_hash,
                        "original_name": artifact.original_name,
                    },
                )
                ir_ref = persist_canonical_source_ir(
                    document,
                    ctx.workspace_root,
                    organization_id=ctx.organization_id,
                    workspace_id=ctx.workspace_id,
                )
                self._knowledge.bind_canonical_ir_ref(
                    source_id=source_id,
                    organization_id=ctx.organization_id,
                    workspace_id=ctx.workspace_id,
                    canonical_ir_ref=ir_ref,
                    expected_content_hash=source["content_hash"],
                )
                identity = SourceIdentity(
                    schemaVersion="semantier.source_identity.v1",
                    sourceIdentityRef=identity_ref,
                    organizationId=ctx.organization_id,
                    workspaceId=ctx.workspace_id,
                    sourceHash=artifact.content_hash,
                    mediaType=artifact.media_type,
                    originalName=artifact.original_name,
                    artifactRef=artifact.byte_owner_ref,
                    sourceVersion=None,
                    supersedesSourceIdentityRef=None,
                    createdAt=datetime.fromisoformat(
                        source["created_at"].replace("Z", "+00:00")
                    ),
                )
                self._knowledge.register_source_identity(
                    source_identity=identity, lineage_id=source_id
                )
                identity = self._knowledge.get_source_identity(
                    source_identity_ref=identity_ref,
                    organization_id=ctx.organization_id,
                    workspace_id=ctx.workspace_id,
                    lineage_id=source_id,
                )
                identity_pin = _pin(
                    "source_identity",
                    identity_ref,
                    "sha256:"
                    + content_hash(
                        identity.model_dump(
                            mode="json", by_alias=True, exclude_none=False
                        )
                    ),
                )
                ir_pin = _pin(
                    "canonical_source_ir",
                    ir_ref,
                    "sha256:"
                    + content_hash(
                        document.model_dump(
                            mode="json", by_alias=True, exclude_none=False
                        )
                    ),
                )
                self._extraction.record_source_representation(
                    organization_id=ctx.organization_id,
                    workspace_id=ctx.workspace_id,
                    source_id=source_id,
                    lineage_id=source_id,
                    relation_kind="normalized_from",
                    original_path=artifact.byte_owner_ref,
                    original_content_hash=artifact.content_hash[7:],
                    artifact_ref=ir_ref,
                    artifact_path=ir_ref,
                    artifact_content_hash=sha256(
                        (ctx.workspace_root / ir_ref).read_bytes()
                    ).hexdigest(),
                )
                outcome, failure = "READY", None
        result = _address(
            WorkspaceArtifactPreparationAdapterResultV1,
            "workspace-artifact-preparation-adapter-result",
            "adapter_result_ref",
            "adapter_result_hash",
            dict(
                schema_version="workspace_artifact_preparation_adapter_result.v1",
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                artifact_pin=pin.model_dump(mode="json"),
                purpose=self._purpose,
                correlation_ref=correlation_ref,
                outcome=outcome,
                source_identity_pin=identity_pin.model_dump(mode="json")
                if identity_pin
                else None,
                canonical_source_ir_pin=ir_pin.model_dump(mode="json")
                if ir_pin
                else None,
                derived_artifact_pins=[],
                failure_code=failure,
                completed_at=datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            ),
        )
        winner = result

        def validate(connection):
            nonlocal winner
            existing = self.exact_get_result(
                ctx=ctx,
                correlation_ref=correlation_ref,
                artifact_pin=pin,
                purpose=self._purpose,
            )
            if existing is not None:
                winner = existing
                return False
            return True

        def mutate(connection):
            changed = connection.execute(
                sql_query("UPDATE workspace_artifact_preparation_operation_index SET adapter_result_ref=?,adapter_result_hash=? WHERE organization_id=? AND workspace_id=? AND correlation_ref=? AND adapter_result_ref IS NULL AND state='PENDING'","UPDATE workspace_artifact_preparation_operation_index SET adapter_result_ref=%s,adapter_result_hash=%s WHERE organization_id=%s AND workspace_id=%s AND correlation_ref=%s AND adapter_result_ref IS NULL AND state='PENDING'"),
                (
                    result.adapter_result_ref,
                    result.adapter_result_hash,
                    ctx.organization_id,
                    ctx.workspace_id,
                    correlation_ref,
                ),
            ).rowcount
            if changed != 1:
                raise ValueError("PREPARATION_RECONCILIATION_FAILED")

        self._build.atomic_write(
            artifacts=(result,),
            validate_before_insert=validate,
            mutate_indexes=mutate,
            verify_before_commit=lambda: self._artifacts.resolve(
                ctx=ctx, artifact_pin=pin
            ),
        )
        return winner
