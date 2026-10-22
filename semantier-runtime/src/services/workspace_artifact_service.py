
from storage.hosted_postgres import sql_query,begin_write,in_transaction
"""Shared workspace artifacts: byte identity and transactional upload receipts."""

from datetime import datetime, timezone
from typing import BinaryIO
from uuid import uuid4

from agents.auth_session import RequestContext
from contracts.knowledge_classification import RuntimeContextDTO
from contracts.ontology_building import ArtifactPinV1
from contracts.ontology_workbench_context import (
    UploadWorkspaceArtifactCommandV1,
    SelectWorkspaceArtifactCommandV1,
    WorkspaceArtifactV1,
    WorkspaceArtifactCollectionV1,
    WorkspaceArtifactOperationReceiptV1,
    WorkspaceArtifactSelectionV1,
    LifecycleArtifactBindingSetV1,
    WorkspaceArtifactPreparationCatalogV1,
    WorkspaceArtifactPreparationV1,
    WorkspaceArtifactListEntryV1,
    ArtifactWorkspaceProjectionV1,
    serialize_workspace_artifact_descriptor,
)
from eos.artifact_hashing import content_hash
from services.runtime_context_transition_service import RuntimeAuthorityResolver
from storage.ontology_build_store import OntologyBuildStore
from storage.workspace_artifact_blob_store import WorkspaceArtifactBlobStore
from storage.workbench_projection_cursor_store import WorkbenchProjectionCursorStore


def _address(model, prefix, ref_field, hash_field, values):
    digest = content_hash(values)
    return model.model_validate(
        {**values, ref_field: prefix + ":" + digest, hash_field: "sha256:" + digest}
    )


def _pin(kind, ref, digest):
    return ArtifactPinV1(artifact_kind=kind, artifact_ref=ref, artifact_hash=digest)


class WorkspaceArtifactService:
    def __init__(
        self,
        *,
        build_store: OntologyBuildStore,
        authority_resolver: RuntimeAuthorityResolver,
        max_bytes: int,
        cursor_store: WorkbenchProjectionCursorStore | None = None,
        cursor_ref: str | None = None,
    ):
        if not isinstance(build_store, OntologyBuildStore) or not isinstance(
            authority_resolver, RuntimeAuthorityResolver
        ):
            raise TypeError(
                "shared artifacts require production store and authority resolver"
            )
        self._store = build_store
        self._authority = authority_resolver
        self._max_bytes = max_bytes
        import os
        self._quota_bytes = int(os.environ.get("SEMANTIER_WORKSPACE_ARTIFACT_QUOTA_BYTES", "104857600"))
        if self._quota_bytes < 0:
            raise ValueError("INVALID_BYTE_LENGTH")
        self._cursors = cursor_store
        self._cursor_ref = cursor_ref

    def _head(self, ctx, table, model, ref_field, hash_field, extra=()):
        clause = sql_query(" AND lifecycle=? AND role=?",' AND lifecycle=%s AND role=%s') if extra else ""
        row = self._store._connection.execute(
            sql_query(f"SELECT revision,head_ref,head_hash FROM {table} WHERE organization_id=? AND workspace_id=?{clause}",f'SELECT revision,head_ref,head_hash FROM {table} WHERE organization_id=%s AND workspace_id=%s{clause}'),
            (ctx.organization_id, ctx.workspace_id, *extra),
        ).fetchone()
        if row is None:
            raise ValueError("ARTIFACT_HEAD_NOT_FOUND")
        value = self._get(ctx, _pin("head", row[1], row[2]), model, hash_field)
        if value.revision != row[0] or getattr(value, ref_field) != row[1]:
            raise ValueError("ARTIFACT_PIN_MISMATCH")
        return value

    def _initialize_cursor(self, ctx):
        self._cursor_ref = self._store.initialize_workspace_artifact_cursor(
            service=self, ctx=ctx
        )

    def _initialize_cursor_on_connection(self, connection, ctx):
        governed = self._cursors._scope(ctx)
        existing = connection.execute(
            sql_query("SELECT cursor_ref FROM workbench_projection_cursors WHERE organization_id=? AND workspace_id=? AND session_id=? AND actor_ref=? LIMIT 2",'SELECT cursor_ref FROM workbench_projection_cursors WHERE organization_id=%s AND workspace_id=%s AND session_id=%s AND actor_ref=%s LIMIT 2'),
            tuple(governed.values()),
        ).fetchall()
        if len(existing) > 1:
            raise ValueError("CURSOR_REQUIRED")
        if existing:
            return self._cursors.get(ctx=ctx, cursor_ref=existing[0][0]).cursor_ref
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        scope = dict(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        definitions = (
            (
                "workspace_artifact_collection_heads",
                WorkspaceArtifactCollectionV1,
                "workspace-artifacts",
                "collection_ref",
                "collection_hash",
                dict(
                    schema_version="workspace_artifact_collection.v1",
                    artifact_pins=[],
                    recorded_at=now,
                ),
                (),
            ),
            (
                "workspace_artifact_preparation_catalog_heads",
                WorkspaceArtifactPreparationCatalogV1,
                "workspace-artifact-preparation-catalog",
                "catalog_ref",
                "catalog_hash",
                dict(
                    schema_version="workspace_artifact_preparation_catalog.v1",
                    preparation_pins=[],
                    predecessor_pin=None,
                    recorded_at=now,
                ),
                (),
            ),
            (
                "workspace_artifact_selection_heads",
                WorkspaceArtifactSelectionV1,
                "workspace-artifact-selection",
                "selection_ref",
                "selection_hash",
                dict(
                    schema_version="workspace_artifact_selection.v1",
                    artifact_pin=None,
                    predecessor_pin=None,
                    selected_by=ctx.user_id,
                    selected_at=now,
                ),
                (),
            ),
            (
                "workspace_artifact_binding_set_heads",
                LifecycleArtifactBindingSetV1,
                "workspace-artifact-binding-set",
                "binding_set_ref",
                "binding_set_hash",
                dict(
                    schema_version="lifecycle_artifact_binding_set.v1",
                    lifecycle="DESIGN",
                    role="SOURCE",
                    entries=[],
                    predecessor_pin=None,
                    recorded_at=now,
                ),
                ("DESIGN", "SOURCE"),
            ),
            (
                "workspace_artifact_binding_set_heads",
                LifecycleArtifactBindingSetV1,
                "workspace-artifact-binding-set",
                "binding_set_ref",
                "binding_set_hash",
                dict(
                    schema_version="lifecycle_artifact_binding_set.v1",
                    lifecycle="RUNTIME",
                    role="TARGET",
                    entries=[],
                    predecessor_pin=None,
                    recorded_at=now,
                ),
                ("RUNTIME", "TARGET"),
            ),
        )
        facts = []
        for table, model, prefix, ref_field, hash_field, fields, extra in definitions:
            clause = sql_query(" AND lifecycle=? AND role=?",' AND lifecycle=%s AND role=%s') if extra else ""
            exists = connection.execute(
                sql_query(f"SELECT 1 FROM {table} WHERE organization_id=? AND workspace_id=?{clause}",f'SELECT 1 FROM {table} WHERE organization_id=%s AND workspace_id=%s{clause}'),
                (*scope.values(), *extra),
            ).fetchone()
            if exists is None:
                value = _address(
                    model,
                    prefix,
                    ref_field,
                    hash_field,
                    dict(**scope, **fields, revision=0),
                )
                self._store._insert_prepared((self._store._prepare(value),))
                values = (
                    *scope.values(),
                    *extra,
                    0,
                    getattr(value, ref_field),
                    getattr(value, hash_field),
                    now,
                )
                connection.execute(
                    f"INSERT INTO {table} VALUES ({','.join(sql_query('?','%s') for _ in values)})",
                    values,
                )
            else:
                value = self._head(ctx, table, model, ref_field, hash_field, extra)
            # New session cursors pin the existing immutable empty genesis;
            # they do not rewind the workspace's current fact heads.
            if model in (WorkspaceArtifactSelectionV1, LifecycleArtifactBindingSetV1):
                while value.revision > 0:
                    predecessor = value.predecessor_pin
                    if (
                        predecessor is None
                        or predecessor.artifact_kind
                        != value.schema_version.removesuffix(".v1")
                    ):
                        raise ValueError("CURSOR_GENESIS_HISTORY_MISMATCH")
                    previous = self._get(ctx, predecessor, model, hash_field)
                    if previous.revision != value.revision - 1:
                        raise ValueError("CURSOR_GENESIS_HISTORY_MISMATCH")
                    if model is LifecycleArtifactBindingSetV1 and (
                        previous.lifecycle,
                        previous.role,
                    ) != (value.lifecycle, value.role):
                        raise ValueError("CURSOR_GENESIS_HISTORY_MISMATCH")
                    value = previous
            facts.append(
                _pin(
                    value.schema_version.removesuffix(".v1"),
                    getattr(value, ref_field),
                    getattr(value, hash_field),
                )
            )
        cursor_ref = "workbench-cursor:" + content_hash(
            {**self._cursors._scope(ctx), "server_nonce": uuid4().hex}
        )
        head = self._cursors.create_genesis_on_connection(
            connection,
            ctx=ctx,
            cursor_ref=cursor_ref,
            selection_pin=facts[2],
            design_binding_set_pin=facts[3],
            runtime_binding_set_pin=facts[4],
        )
        return head.cursor_ref

    def list(self, *, ctx: RequestContext) -> ArtifactWorkspaceProjectionV1:
        self._access(ctx)
        if not isinstance(self._cursors, WorkbenchProjectionCursorStore):
            raise ValueError("governed session cursor owner required")
        if self._cursor_ref is None:
            self._initialize_cursor(ctx)
        connection = self._store._connection
        if in_transaction(connection):
            raise RuntimeError("list requires its own read snapshot")
        connection.execute(sql_query("BEGIN","BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY"))
        try:
            cursor = self._cursors.get(ctx=ctx, cursor_ref=self._cursor_ref)
            collection = self._head(
                ctx,
                "workspace_artifact_collection_heads",
                WorkspaceArtifactCollectionV1,
                "collection_ref",
                "collection_hash",
            )
            catalog = self._head(
                ctx,
                "workspace_artifact_preparation_catalog_heads",
                WorkspaceArtifactPreparationCatalogV1,
                "catalog_ref",
                "catalog_hash",
            )
            preparations = [
                self._get(ctx, pin, WorkspaceArtifactPreparationV1, "preparation_hash")
                for pin in catalog.preparation_pins
            ]
            preparations.sort(
                key=lambda p: (
                    p.artifact_pin.artifact_ref,
                    p.purpose,
                    p.revision,
                    p.preparation_ref,
                    p.preparation_hash,
                )
            )
            artifacts = []
            blobs = WorkspaceArtifactBlobStore(ctx=ctx, max_bytes=self._max_bytes)
            for pin in collection.artifact_pins:
                artifact = self._get(ctx, pin, WorkspaceArtifactV1, "artifact_hash")
                access = "AVAILABLE"
                try:
                    with blobs.exact_get(
                        byte_owner_ref=artifact.byte_owner_ref,
                        expected_hash=artifact.content_hash,
                        expected_size=artifact.byte_size,
                    ):
                        pass
                except (OSError, ValueError):
                    access = "INACCESSIBLE"
                artifacts.append(
                    WorkspaceArtifactListEntryV1(
                        schema_version="workspace_artifact_list_entry.v1",
                        artifact=serialize_workspace_artifact_descriptor(artifact),
                        access_state=access,
                        active_preparation_pins=tuple(
                            _pin(
                                "workspace_artifact_preparation",
                                p.preparation_ref,
                                p.preparation_hash,
                            )
                            for p in preparations
                            if p.artifact_pin == pin
                        ),
                    )
                )
            selection = self._get(
                ctx,
                cursor.selection_pin,
                WorkspaceArtifactSelectionV1,
                "selection_hash",
            )
            design = self._get(
                ctx,
                cursor.design_binding_set_pin,
                LifecycleArtifactBindingSetV1,
                "binding_set_hash",
            )
            runtime = self._get(
                ctx,
                cursor.runtime_binding_set_pin,
                LifecycleArtifactBindingSetV1,
                "binding_set_hash",
            )
            result = _address(
                ArtifactWorkspaceProjectionV1,
                "workspace-artifact-projection",
                "projection_ref",
                "projection_hash",
                dict(
                    schema_version="artifact_workspace_projection.v1",
                    organization_id=ctx.organization_id,
                    workspace_id=ctx.workspace_id,
                    cursor_ref=cursor.cursor_ref,
                    material_projection_hash=cursor.projection_basis_hash,
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
                    artifacts=[item.model_dump(mode="json") for item in artifacts],
                    preparations=[
                        item.model_dump(mode="json") for item in preparations
                    ],
                    selection=selection.model_dump(mode="json"),
                    design_sources=design.model_dump(mode="json"),
                    runtime_target=runtime.model_dump(mode="json"),
                    workspace_selection_head=self._head(ctx, "workspace_artifact_selection_heads", WorkspaceArtifactSelectionV1, "selection_ref", "selection_hash").model_dump(mode="json"),
                    workspace_design_sources_head=self._head(ctx, "workspace_artifact_binding_set_heads", LifecycleArtifactBindingSetV1, "binding_set_ref", "binding_set_hash", ("DESIGN", "SOURCE")).model_dump(mode="json"),
                    workspace_runtime_target_head=self._head(ctx, "workspace_artifact_binding_set_heads", LifecycleArtifactBindingSetV1, "binding_set_ref", "binding_set_hash", ("RUNTIME", "TARGET")).model_dump(mode="json"),
                    material_context_revision=cursor.revision,
                    projected_at=max(
                        cursor.recorded_at, collection.recorded_at, catalog.recorded_at
                    )
                    .isoformat()
                    .replace("+00:00", "Z"),
                ),
            )
            self._access(ctx)
            connection.commit()
            return result
        except BaseException:
            connection.rollback()
            raise

    def select(
        self, *, ctx: RequestContext, command: SelectWorkspaceArtifactCommandV1
    ) -> WorkspaceArtifactOperationReceiptV1:
        self._access(ctx)
        if not isinstance(self._cursors, WorkbenchProjectionCursorStore):
            raise ValueError("governed session cursor owner required")
        command = SelectWorkspaceArtifactCommandV1.model_validate(
            command.model_dump(mode="json")
        )
        return self._store.select_workspace_artifact(
            service=self, ctx=ctx, command=command
        )

    def _select_on_connection(self, connection, ctx, command):
        scope = (ctx.organization_id, ctx.workspace_id)
        request_hash = "sha256:" + content_hash(command.model_dump(mode="json"))
        row = connection.execute(
            sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation='SELECT_ARTIFACT' AND request_id=?","SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation='SELECT_ARTIFACT' AND request_id=%s"),
            (*scope, ctx.user_id, command.request_id),
        ).fetchone()
        if row is not None:
            if row[0] != request_hash:
                raise ValueError("REQUEST_CONFLICT")
            receipt = self._get(
                ctx,
                _pin("workspace_artifact_operation_receipt", row[1], row[2]),
                WorkspaceArtifactOperationReceiptV1,
                "receipt_hash",
            )
            if (
                receipt.operation,
                receipt.actor_ref,
                receipt.request_id,
                receipt.request_hash,
            ) != ("SELECT_ARTIFACT", ctx.user_id, command.request_id, request_hash):
                raise ValueError("ARTIFACT_PIN_MISMATCH")
            self.resolve(ctx=ctx, artifact_pin=receipt.artifact_pin)
            collection = self._get(
                ctx,
                receipt.collection_pin,
                WorkspaceArtifactCollectionV1,
                "collection_hash",
            )
            selection = self._get(
                ctx,
                receipt.selection_pin,
                WorkspaceArtifactSelectionV1,
                "selection_hash",
            )
            cursor = self._cursors.exact_head(ctx=ctx, pin=receipt.cursor_pin)
            governed = self._cursors._scope(ctx)
            if (
                (cursor.session_id, cursor.actor_ref, cursor.cursor_ref)
                != (governed["session_id"], ctx.user_id, command.cursor_ref)
                or cursor.revision != receipt.material_context_revision
                or cursor.selection_pin != receipt.selection_pin
                or selection.revision != receipt.selection_revision
                or selection.artifact_pin != receipt.artifact_pin
                or collection.revision != receipt.collection_revision
                or receipt.artifact_pin not in collection.artifact_pins
            ):
                raise ValueError("ARTIFACT_PIN_MISMATCH")
            self._cursors._facts(
                governed,
                cursor.selection_pin,
                cursor.design_binding_set_pin,
                cursor.runtime_binding_set_pin,
            )
            return receipt
        artifact = self.resolve(ctx=ctx, artifact_pin=command.artifact_pin)
        collection = self._head(
            ctx,
            "workspace_artifact_collection_heads",
            WorkspaceArtifactCollectionV1,
            "collection_ref",
            "collection_hash",
        )
        if collection.revision != command.expected_collection_revision:
            raise ValueError("STALE_COLLECTION_HEAD")
        if command.artifact_pin not in collection.artifact_pins:
            raise ValueError("ARTIFACT_NOT_FOUND")
        previous = self._head(
            ctx,
            "workspace_artifact_selection_heads",
            WorkspaceArtifactSelectionV1,
            "selection_ref",
            "selection_hash",
        )
        previous_pin = _pin(
            "workspace_artifact_selection",
            previous.selection_ref,
            previous.selection_hash,
        )
        if (
            previous.revision != command.expected_selection_revision
            or previous_pin != command.expected_selection_pin
        ):
            raise ValueError("STALE_SELECTION_HEAD")
        cursor = self._cursors.get(ctx=ctx, cursor_ref=command.cursor_ref)
        if (
            cursor.revision != command.expected_material_context_revision
            or cursor.projection_basis_hash != command.expected_projection_hash
        ):
            raise ValueError("STALE_MATERIAL_CONTEXT")
        now = datetime.now(timezone.utc)
        selected = _address(
            WorkspaceArtifactSelectionV1,
            "workspace-artifact-selection",
            "selection_ref",
            "selection_hash",
            dict(
                schema_version="workspace_artifact_selection.v1",
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                artifact_pin=command.artifact_pin.model_dump(mode="json"),
                revision=previous.revision + 1,
                predecessor_pin=previous_pin.model_dump(mode="json"),
                selected_by=ctx.user_id,
                selected_at=now.isoformat().replace("+00:00", "Z"),
            ),
        )
        selected_pin = _pin(
            "workspace_artifact_selection",
            selected.selection_ref,
            selected.selection_hash,
        )
        self._store._insert_prepared((self._store._prepare(selected),))
        connection.execute(
            sql_query("UPDATE workspace_artifact_selection_heads SET revision=?,head_ref=?,head_hash=?,recorded_at=? WHERE organization_id=? AND workspace_id=?",'UPDATE workspace_artifact_selection_heads SET revision=%s,head_ref=%s,head_hash=%s,recorded_at=%s WHERE organization_id=%s AND workspace_id=%s'),
            (
                selected.revision,
                selected.selection_ref,
                selected.selection_hash,
                now.isoformat(),
                *scope,
            ),
        )
        material = dict(
            **self._cursors._scope(ctx),
            cursor_ref=cursor.cursor_ref,
            revision=cursor.revision + 1,
            selection_pin=selected_pin.model_dump(mode="json"),
            design_binding_set_pin=cursor.design_binding_set_pin.model_dump(
                mode="json"
            ),
            runtime_binding_set_pin=cursor.runtime_binding_set_pin.model_dump(
                mode="json"
            ),
            evidence_focus_pin=cursor.evidence_focus_pin.model_dump(mode="json") if cursor.evidence_focus_pin else None,
            design_snapshot_pin=cursor.design_snapshot_pin.model_dump(mode="json") if cursor.design_snapshot_pin else None,
        )
        successor = self._cursors.advance_on_connection(
            connection,
            ctx=ctx,
            cursor_ref=cursor.cursor_ref,
            expected_head_pin=_pin(
                "workbench_projection_cursor_head",
                cursor.cursor_head_ref,
                cursor.cursor_head_hash,
            ),
            expected_revision=command.expected_material_context_revision,
            expected_projection_basis_hash=command.expected_projection_hash,
            selection_pin=selected_pin,
            design_binding_set_pin=cursor.design_binding_set_pin,
            runtime_binding_set_pin=cursor.runtime_binding_set_pin,
            successor_projection_basis_hash="sha256:" + content_hash(material),
            recorded_at=now,
        )
        receipt = _address(
            WorkspaceArtifactOperationReceiptV1,
            "workspace-artifact-receipt",
            "receipt_ref",
            "receipt_hash",
            dict(
                schema_version="workspace_artifact_operation_receipt.v1",
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                operation="SELECT_ARTIFACT",
                artifact_pin=command.artifact_pin.model_dump(mode="json"),
                collection_pin=_pin(
                    "workspace_artifact_collection",
                    collection.collection_ref,
                    collection.collection_hash,
                ).model_dump(mode="json"),
                collection_revision=collection.revision,
                preparation_catalog_pin=None,
                preparation_catalog_revision=None,
                preparation_pin=None,
                selection_pin=selected_pin.model_dump(mode="json"),
                selection_revision=selected.revision,
                binding_set_pin=None,
                binding_set_revision=None,
                material_context_revision=successor.revision,
                cursor_pin=_pin(
                    "workbench_projection_cursor_head",
                    successor.cursor_head_ref,
                    successor.cursor_head_hash,
                ).model_dump(mode="json"),
                ingress=command.ingress,
                request_id=command.request_id,
                request_hash=request_hash,
                idempotency_key=command.idempotency_key,
                actor_ref=ctx.user_id,
                outcome="SELECTED",
                recorded_at=now.isoformat().replace("+00:00", "Z"),
            ),
        )
        self._store._insert_prepared((self._store._prepare(receipt),))
        connection.execute(
            sql_query("INSERT INTO workspace_operation_request_index VALUES (?,?,?,?,?,?,?,?)",'INSERT INTO workspace_operation_request_index VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'),
            (
                *scope,
                ctx.user_id,
                "SELECT_ARTIFACT",
                command.request_id,
                request_hash,
                receipt.receipt_ref,
                receipt.receipt_hash,
            ),
        )
        self.resolve(
            ctx=ctx,
            artifact_pin=_pin(
                "workspace_artifact", artifact.artifact_ref, artifact.artifact_hash
            ),
        )
        return receipt

    def _access(self, ctx):
        if (
            not isinstance(ctx, RequestContext)
            or not ctx.authenticated
            or not ctx.organization_id
        ):
            raise ValueError("AUTHORITY_CONTEXT_REQUIRED")
        self._authority.require_current_actor(
            authorization_class="WORKSPACE_LOCAL",
            actor_context=RuntimeContextDTO(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                actor_id=ctx.user_id,
            )
        )

    def _get(self, ctx, pin, model, hash_field):
        try:
            value = self._store.get(
                organization_id=ctx.organization_id,
                workspace_id=ctx.workspace_id,
                artifact_ref=pin.artifact_ref,
                contract_type=model,
            )
        except PermissionError as exc:
            raise ValueError("ARTIFACT_NOT_FOUND") from exc
        if value is None:
            raise ValueError("ARTIFACT_NOT_FOUND")
        if getattr(value, hash_field) != pin.artifact_hash:
            raise ValueError("ARTIFACT_PIN_MISMATCH")
        return value

    def resolve(
        self, *, ctx: RequestContext, artifact_pin: ArtifactPinV1
    ) -> WorkspaceArtifactV1:
        self._access(ctx)
        if artifact_pin.artifact_kind != "workspace_artifact":
            raise ValueError("ARTIFACT_PIN_MISMATCH")
        artifact = self._get(ctx, artifact_pin, WorkspaceArtifactV1, "artifact_hash")
        blobs = WorkspaceArtifactBlobStore(ctx=ctx, max_bytes=self._max_bytes)
        with blobs.exact_get(
            byte_owner_ref=artifact.byte_owner_ref,
            expected_hash=artifact.content_hash,
            expected_size=artifact.byte_size,
        ):
            pass
        return artifact

    def _retry(self, ctx, command, request_hash):
        row = self._store._connection.execute(
            sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation='UPLOAD_ARTIFACT' AND request_id=?","SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation='UPLOAD_ARTIFACT' AND request_id=%s"),
            (ctx.organization_id, ctx.workspace_id, ctx.user_id, command.request_id),
        ).fetchone()
        if row is None:
            return None
        if row[0] != request_hash:
            raise ValueError("REQUEST_CONFLICT")
        receipt = self._get(
            ctx,
            _pin("workspace_artifact_operation_receipt", row[1], row[2]),
            WorkspaceArtifactOperationReceiptV1,
            "receipt_hash",
        )
        if (
            receipt.operation != "UPLOAD_ARTIFACT"
            or receipt.actor_ref != ctx.user_id
            or receipt.request_id != command.request_id
            or receipt.request_hash != request_hash
        ):
            raise ValueError("ARTIFACT_PIN_MISMATCH")
        self.resolve(ctx=ctx, artifact_pin=receipt.artifact_pin)
        collection = self._get(
            ctx,
            receipt.collection_pin,
            WorkspaceArtifactCollectionV1,
            "collection_hash",
        )
        if (
            collection.revision != receipt.collection_revision
            or receipt.artifact_pin not in collection.artifact_pins
        ):
            raise ValueError("ARTIFACT_PIN_MISMATCH")
        return receipt

    def upload(
        self,
        *,
        ctx: RequestContext,
        command: UploadWorkspaceArtifactCommandV1,
        byte_stream: BinaryIO,
    ) -> WorkspaceArtifactOperationReceiptV1:
        self._access(ctx)
        command = UploadWorkspaceArtifactCommandV1.model_validate(
            command.model_dump(mode="json")
        )
        blobs = WorkspaceArtifactBlobStore(ctx=ctx, max_bytes=self._max_bytes)
        stage = blobs.stage(
            byte_stream=byte_stream,
            expected_hash=command.byte_hash,
            expected_size=command.byte_length,
            original_name=command.original_name,
            declared_media_type=command.declared_media_type,
        )
        try:
            for attempt in range(2):
                try:
                    return self._upload_staged(ctx, command, blobs, stage)
                except ValueError as exc:
                    if str(exc) != "STALE_COLLECTION_HEAD" or attempt:
                        raise
                    # A concurrent equal-byte winner is the sole reconciliation
                    # case. Different uploads still surface their stale CAS.
                    winner = self._store._connection.execute(
                        sql_query("SELECT artifact_ref FROM workspace_artifacts WHERE organization_id=? AND workspace_id=? AND content_hash=?",'SELECT artifact_ref FROM workspace_artifacts WHERE organization_id=%s AND workspace_id=%s AND content_hash=%s'),
                        (ctx.organization_id, ctx.workspace_id, stage.content_hash),
                    ).fetchone()
                    if winner is None:
                        raise
            raise AssertionError("unreachable upload attempt")
        finally:
            blobs.discard_stage(stage_ref=stage.stage_ref)

    def _upload_staged(self, ctx, command, blobs, stage):
        request_hash = "sha256:" + content_hash(
            {
                "command": command.model_dump(mode="json"),
                "verified_byte_hash": stage.content_hash,
                "verified_byte_size": stage.byte_size,
                "detected_media_type": stage.detected_media_type,
            }
        )
        retry = self._retry(ctx, command, request_hash)
        if retry is not None:
            return retry
        connection = self._store._connection
        scope = (ctx.organization_id, ctx.workspace_id)
        head = connection.execute(
            sql_query("SELECT revision,head_ref,head_hash FROM workspace_artifact_collection_heads WHERE organization_id=? AND workspace_id=?",'SELECT revision,head_ref,head_hash FROM workspace_artifact_collection_heads WHERE organization_id=%s AND workspace_id=%s'),
            scope,
        ).fetchone()
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        base = dict(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        if head is None:
            collection = _address(
                WorkspaceArtifactCollectionV1,
                "workspace-artifacts",
                "collection_ref",
                "collection_hash",
                dict(
                    **base,
                    schema_version="workspace_artifact_collection.v1",
                    revision=0,
                    artifact_pins=[],
                    recorded_at=now,
                ),
            )
        else:
            collection = self._get(
                ctx,
                _pin("workspace_artifact_collection", head[1], head[2]),
                WorkspaceArtifactCollectionV1,
                "collection_hash",
            )
            if collection.revision != head[0]:
                raise ValueError("ARTIFACT_PIN_MISMATCH")
        row = connection.execute(
            sql_query("SELECT artifact_ref,artifact_hash FROM workspace_artifacts WHERE organization_id=? AND workspace_id=? AND content_hash=?",'SELECT artifact_ref,artifact_hash FROM workspace_artifacts WHERE organization_id=%s AND workspace_id=%s AND content_hash=%s'),
            (*scope, stage.content_hash),
        ).fetchone()
        if row is not None:
            artifact = self.resolve(
                ctx=ctx, artifact_pin=_pin("workspace_artifact", *row)
            )
            if artifact.media_type != stage.detected_media_type:
                raise ValueError("ARTIFACT_METADATA_CONFLICT")
            outcome = "REUSED"
        else:
            if collection.revision != command.expected_collection_revision:
                raise ValueError("STALE_COLLECTION_HEAD")
            digest = stage.content_hash[7:]
            artifact = _address(
                WorkspaceArtifactV1,
                "workspace-artifact",
                "artifact_ref",
                "artifact_hash",
                dict(
                    **base,
                    schema_version="workspace_artifact.v1",
                    content_hash=stage.content_hash,
                    original_name=command.original_name,
                    media_type=stage.detected_media_type,
                    byte_size=stage.byte_size,
                    byte_owner_ref=f"artifacts/workspace-originals/sha256/{digest[:2]}/{digest}",
                    created_by=ctx.user_id,
                    created_at=now,
                ),
            )
            outcome = "CREATED"
        artifact_pin = _pin(
            "workspace_artifact", artifact.artifact_ref, artifact.artifact_hash
        )
        previous = collection
        if outcome == "CREATED":
            pins = sorted(
                (*collection.artifact_pins, artifact_pin),
                key=lambda p: (p.artifact_kind, p.artifact_ref, p.artifact_hash),
            )
            collection = _address(
                WorkspaceArtifactCollectionV1,
                "workspace-artifacts",
                "collection_ref",
                "collection_hash",
                dict(
                    **base,
                    schema_version="workspace_artifact_collection.v1",
                    revision=collection.revision + 1,
                    artifact_pins=[pin.model_dump(mode="json") for pin in pins],
                    recorded_at=now,
                ),
            )
        elif artifact_pin not in collection.artifact_pins:
            raise ValueError("ARTIFACT_PIN_MISMATCH")
        receipt = _address(
            WorkspaceArtifactOperationReceiptV1,
            "workspace-artifact-receipt",
            "receipt_ref",
            "receipt_hash",
            dict(
                **base,
                schema_version="workspace_artifact_operation_receipt.v1",
                operation="UPLOAD_ARTIFACT",
                artifact_pin=artifact_pin.model_dump(mode="json"),
                collection_pin=_pin(
                    "workspace_artifact_collection",
                    collection.collection_ref,
                    collection.collection_hash,
                ).model_dump(mode="json"),
                collection_revision=collection.revision,
                preparation_catalog_pin=None,
                preparation_catalog_revision=None,
                preparation_pin=None,
                selection_pin=None,
                selection_revision=None,
                binding_set_pin=None,
                binding_set_revision=None,
                material_context_revision=None,
                cursor_pin=None,
                ingress=command.ingress,
                request_id=command.request_id,
                request_hash=request_hash,
                idempotency_key=command.idempotency_key,
                actor_ref=ctx.user_id,
                outcome=outcome,
                recorded_at=now,
            ),
        )
        winner = receipt

        def validate(connection):
            nonlocal winner
            self._access(ctx)
            saved = self._retry(ctx, command, request_hash)
            if saved is not None:
                winner = saved
                return False
            current = connection.execute(
                sql_query("SELECT revision,head_ref,head_hash FROM workspace_artifact_collection_heads WHERE organization_id=? AND workspace_id=?",'SELECT revision,head_ref,head_hash FROM workspace_artifact_collection_heads WHERE organization_id=%s AND workspace_id=%s'),
                scope,
            ).fetchone()
            if current != head:
                raise ValueError("STALE_COLLECTION_HEAD")
            if outcome == "CREATED":
                used = sum(
                    self._get(ctx, member, WorkspaceArtifactV1, "artifact_hash").byte_size
                    for member in previous.artifact_pins
                )
                if used + artifact.byte_size > self._quota_bytes:
                    raise ValueError("INVALID_BYTE_LENGTH")
            return True

        def mutate(connection):
            published = blobs.publish(stage=stage, artifact_ref=artifact.artifact_ref)
            if published != artifact.byte_owner_ref:
                raise ValueError("ARTIFACT_PIN_MISMATCH")
            if outcome == "CREATED":
                connection.execute(
                    sql_query("INSERT INTO workspace_artifacts VALUES (?,?,?,?,?,?)",'INSERT INTO workspace_artifacts VALUES (%s,%s,%s,%s,%s,%s)'),
                    (
                        *scope,
                        artifact.content_hash,
                        artifact.artifact_ref,
                        artifact.artifact_hash,
                        now,
                    ),
                )
                connection.execute(
                    sql_query("INSERT INTO workspace_artifact_collection_heads VALUES (?,?,?,?,?,?) ON CONFLICT(organization_id,workspace_id) DO UPDATE SET revision=excluded.revision,head_ref=excluded.head_ref,head_hash=excluded.head_hash,recorded_at=excluded.recorded_at",'INSERT INTO workspace_artifact_collection_heads VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT(organization_id,workspace_id) DO UPDATE SET revision=excluded.revision,head_ref=excluded.head_ref,head_hash=excluded.head_hash,recorded_at=excluded.recorded_at'),
                    (
                        *scope,
                        collection.revision,
                        collection.collection_ref,
                        collection.collection_hash,
                        now,
                    ),
                )
            connection.execute(
                sql_query("INSERT INTO workspace_operation_request_index VALUES (?,?,?,?,?,?,?,?)",'INSERT INTO workspace_operation_request_index VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'),
                (
                    *scope,
                    ctx.user_id,
                    "UPLOAD_ARTIFACT",
                    command.request_id,
                    request_hash,
                    receipt.receipt_ref,
                    receipt.receipt_hash,
                ),
            )

        artifacts = (
            (previous, artifact, collection, receipt)
            if outcome == "CREATED"
            else (receipt,)
        )
        self._store.atomic_write(
            artifacts=artifacts,
            validate_before_insert=validate,
            mutate_indexes=mutate,
            verify_before_commit=lambda: self._access(ctx),
        )
        return winner
