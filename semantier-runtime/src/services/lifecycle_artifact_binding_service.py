"""Workspace lifecycle binding through the shared fact/cursor transaction."""

from storage.hosted_postgres import sql_query

from datetime import datetime, timezone

from contracts.ontology_workbench_context import (
    BindLifecycleArtifactCommandV1,
    ClearLifecycleArtifactBindingCommandV1,
    LifecycleArtifactBindingEntryV1,
    LifecycleArtifactBindingSetV1,
    WorkspaceArtifactCollectionV1,
    WorkspaceArtifactSelectionV1,
    WorkspaceArtifactOperationReceiptV1,
)
from eos.artifact_hashing import content_hash
from services.workspace_artifact_service import WorkspaceArtifactService, _address, _pin
from services.workspace_artifact_preparation_service import (
    WorkspaceArtifactPreparationService,
)
from storage.workbench_projection_cursor_store import WorkbenchProjectionCursorStore


class LifecycleArtifactBindingService:
    def __init__(self, *, artifact_service, preparation_service):
        if not isinstance(artifact_service, WorkspaceArtifactService) or not isinstance(
            preparation_service, WorkspaceArtifactPreparationService
        ):
            raise TypeError("binding requires real artifact and preparation owners")
        if preparation_service._build is not artifact_service._store or not isinstance(
            artifact_service._cursors, WorkbenchProjectionCursorStore
        ):
            raise TypeError(
                "binding requires one workspace store and governed cursor owner"
            )
        self._artifacts = artifact_service
        self._preparation = preparation_service
        self._build = artifact_service._store
        self._cursors = artifact_service._cursors

    def project(self, *, ctx):
        return self._artifacts.list(ctx=ctx)

    def bind(self, *, ctx, command):
        command = BindLifecycleArtifactCommandV1.model_validate(
            command.model_dump(mode="json")
        )
        return self._build.bind_workspace_artifact(
            service=self, ctx=ctx, command=command
        )

    def clear(self, *, ctx, command):
        command = ClearLifecycleArtifactBindingCommandV1.model_validate(
            command.model_dump(mode="json")
        )
        return self._build.bind_workspace_artifact(
            service=self, ctx=ctx, command=command
        )

    def _bind_on_connection(self, connection, ctx, command):
        clearing = isinstance(command, ClearLifecycleArtifactBindingCommandV1)
        operation = ("CLEAR_" if clearing else "BIND_") + command.role
        self._artifacts.resolve(ctx=ctx, artifact_pin=command.artifact_pin)
        if not clearing:
            if command.preparation_pin is None:
                raise ValueError(
                    "SOURCE_NOT_PREPARED" if command.role == "SOURCE" else "TARGET_UNSUPPORTED"
                )
            prepared = self._preparation.resolve(
                ctx=ctx,
                artifact_pin=command.artifact_pin,
                preparation_pin=command.preparation_pin,
                purpose="DESIGN_SOURCE" if command.role == "SOURCE" else "RUNTIME_TARGET",
            )
        scope = (ctx.organization_id, ctx.workspace_id)
        request_hash = "sha256:" + content_hash(command.model_dump(mode="json"))
        row = connection.execute(
            sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?", 'SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s'),
            (*scope, ctx.user_id, operation, command.request_id),
        ).fetchone()
        if row is not None:
            if row[0] != request_hash:
                raise ValueError("REQUEST_CONFLICT")
            receipt = self._artifacts._get(
                ctx,
                _pin("workspace_artifact_operation_receipt", row[1], row[2]),
                WorkspaceArtifactOperationReceiptV1,
                "receipt_hash",
            )
            snapshot = self._artifacts._get(
                ctx,
                receipt.binding_set_pin,
                LifecycleArtifactBindingSetV1,
                "binding_set_hash",
            )
            cursor = self._cursors.exact_head(ctx=ctx, pin=receipt.cursor_pin)
            governed = self._cursors._scope(ctx)
            if (
                (
                    receipt.operation,
                    receipt.request_hash,
                    receipt.request_id,
                    receipt.actor_ref,
                    receipt.artifact_pin,
                )
                != (
                    operation,
                    request_hash,
                    command.request_id,
                    ctx.user_id,
                    command.artifact_pin,
                )
                or snapshot.revision != receipt.binding_set_revision
                or (
                    any(
                        e.artifact_pin == command.artifact_pin for e in snapshot.entries
                    )
                    if clearing
                    else not any(
                        e.artifact_pin == command.artifact_pin
                        and e.preparation_pin == command.preparation_pin
                        for e in snapshot.entries
                    )
                )
                or (
                    cursor.design_binding_set_pin
                    if command.role == "SOURCE"
                    else cursor.runtime_binding_set_pin
                )
                != receipt.binding_set_pin
                or (
                    cursor.session_id,
                    cursor.actor_ref,
                    cursor.cursor_ref,
                    cursor.revision,
                )
                != (
                    governed["session_id"],
                    ctx.user_id,
                    command.cursor_ref,
                    receipt.material_context_revision,
                )
            ):
                raise ValueError("ARTIFACT_PIN_MISMATCH")
            self._cursors._facts(
                governed,
                cursor.selection_pin,
                cursor.design_binding_set_pin,
                cursor.runtime_binding_set_pin,
            )
            return receipt
        collection = self._artifacts._head(
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
        previous = self._artifacts._head(
            ctx,
            "workspace_artifact_binding_set_heads",
            LifecycleArtifactBindingSetV1,
            "binding_set_ref",
            "binding_set_hash",
            extra=(command.lifecycle, command.role),
        )
        previous_pin = _pin(
            "lifecycle_artifact_binding_set",
            previous.binding_set_ref,
            previous.binding_set_hash,
        )
        if (previous_pin, previous.revision) != (
            command.expected_binding_set_pin,
            command.expected_binding_set_revision,
        ):
            raise ValueError("STALE_BINDING_SET_HEAD")
        cursor = self._cursors.get(ctx=ctx, cursor_ref=command.cursor_ref)
        if (cursor.revision, cursor.projection_basis_hash) != (
            command.expected_material_context_revision,
            command.expected_projection_hash,
        ):
            raise ValueError("STALE_MATERIAL_CONTEXT")
        now = datetime.now(timezone.utc)
        timestamp = now.isoformat().replace("+00:00", "Z")
        base = dict(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id)
        # Workspace heads serialize publications; entries remain the exact task's
        # cursor view, so another session never supplies hidden source/target state.
        task_pin = cursor.design_binding_set_pin if command.role == "SOURCE" else cursor.runtime_binding_set_pin
        task_bindings = self._artifacts._get(ctx, task_pin, LifecycleArtifactBindingSetV1, "binding_set_hash")
        old_entry = next(
            (e for e in task_bindings.entries if e.artifact_pin == command.artifact_pin),
            None,
        )
        entries = [] if command.role == "TARGET" else [e for e in task_bindings.entries if e is not old_entry]
        entry = None
        if clearing and old_entry is None:
            raise ValueError("BINDING_NOT_FOUND")
        if not clearing:
            index = (
                command.insert_at
                if command.insert_at is not None
                else (task_bindings.entries.index(old_entry) if old_entry else len(entries))
            )
            if index > len(entries):
                raise ValueError("INVALID_LIFECYCLE_ROLE")
            entry = _address(
                LifecycleArtifactBindingEntryV1,
                "workspace-artifact-binding",
                "binding_ref",
                "binding_hash",
                dict(
                    **base,
                    schema_version="lifecycle_artifact_binding_entry.v1",
                    lifecycle=command.lifecycle,
                    role=command.role,
                    artifact_pin=command.artifact_pin.model_dump(mode="json"),
                    preparation_pin=None
                    if clearing
                    else command.preparation_pin.model_dump(mode="json"),
                    source_identity_pin=(
                        prepared.source_identity_pin.model_dump(mode="json")
                        if prepared.source_identity_pin is not None
                        else None
                    ),
                    canonical_source_ir_pin=(
                        prepared.canonical_source_ir_pin.model_dump(mode="json")
                        if prepared.canonical_source_ir_pin is not None
                        else None
                    ),
                    entry_revision=old_entry.entry_revision + 1 if old_entry else 0,
                    predecessor_entry_pin=_pin(
                        "lifecycle_artifact_binding_entry",
                        old_entry.binding_ref,
                        old_entry.binding_hash,
                    ).model_dump(mode="json")
                    if old_entry
                    else None,
                    state="ACTIVE",
                    bound_by=ctx.user_id,
                    bound_at=timestamp,
                ),
            )
            entries.insert(index, entry)
        snapshot = _address(
            LifecycleArtifactBindingSetV1,
            "workspace-artifact-binding-set",
            "binding_set_ref",
            "binding_set_hash",
            dict(
                **base,
                schema_version="lifecycle_artifact_binding_set.v1",
                lifecycle=command.lifecycle,
                role=command.role,
                entries=[e.model_dump(mode="json") for e in entries],
                revision=previous.revision + 1,
                predecessor_pin=previous_pin.model_dump(mode="json"),
                recorded_at=timestamp,
            ),
        )
        binding_pin = _pin(
            "lifecycle_artifact_binding_set",
            snapshot.binding_set_ref,
            snapshot.binding_set_hash,
        )
        selection_pin = cursor.selection_pin
        selected = None
        if not clearing and command.select_after_bind:
            prior_selection = self._artifacts._head(
                ctx,
                "workspace_artifact_selection_heads",
                WorkspaceArtifactSelectionV1,
                "selection_ref",
                "selection_hash",
            )
            prior_pin = _pin(
                "workspace_artifact_selection",
                prior_selection.selection_ref,
                prior_selection.selection_hash,
            )
            if (prior_pin, prior_selection.revision) != (
                command.expected_selection_pin,
                command.expected_selection_revision,
            ):
                raise ValueError("STALE_SELECTION_HEAD")
            selected = _address(
                WorkspaceArtifactSelectionV1,
                "workspace-artifact-selection",
                "selection_ref",
                "selection_hash",
                dict(
                    **base,
                    schema_version="workspace_artifact_selection.v1",
                    artifact_pin=command.artifact_pin.model_dump(mode="json"),
                    revision=prior_selection.revision + 1,
                    predecessor_pin=prior_pin.model_dump(mode="json"),
                    selected_by=ctx.user_id,
                    selected_at=timestamp,
                ),
            )
            selection_pin = _pin(
                "workspace_artifact_selection",
                selected.selection_ref,
                selected.selection_hash,
            )
            self._build._insert_prepared((self._build._prepare(selected),))
            connection.execute(
                sql_query("UPDATE workspace_artifact_selection_heads SET revision=?,head_ref=?,head_hash=?,recorded_at=? WHERE organization_id=? AND workspace_id=?", 'UPDATE workspace_artifact_selection_heads SET revision=%s,head_ref=%s,head_hash=%s,recorded_at=%s WHERE organization_id=%s AND workspace_id=%s'),
                (
                    selected.revision,
                    selected.selection_ref,
                    selected.selection_hash,
                    timestamp,
                    *scope,
                ),
            )
        self._build._insert_prepared(
            tuple(self._build._prepare(v) for v in (entry, snapshot) if v is not None)
        )
        connection.execute(
            sql_query("UPDATE workspace_artifact_binding_set_heads SET revision=?,head_ref=?,head_hash=?,recorded_at=? WHERE organization_id=? AND workspace_id=? AND lifecycle=? AND role=?", 'UPDATE workspace_artifact_binding_set_heads SET revision=%s,head_ref=%s,head_hash=%s,recorded_at=%s WHERE organization_id=%s AND workspace_id=%s AND lifecycle=%s AND role=%s'),
            (
                snapshot.revision,
                snapshot.binding_set_ref,
                snapshot.binding_set_hash,
                timestamp,
                *scope,
                command.lifecycle,
                command.role,
            ),
        )
        design_pin = (
            binding_pin if command.role == "SOURCE" else cursor.design_binding_set_pin
        )
        runtime_pin = (
            binding_pin if command.role == "TARGET" else cursor.runtime_binding_set_pin
        )
        material = dict(
            **self._cursors._scope(ctx),
            cursor_ref=cursor.cursor_ref,
            revision=cursor.revision + 1,
            selection_pin=selection_pin.model_dump(mode="json"),
            design_binding_set_pin=design_pin.model_dump(mode="json"),
            runtime_binding_set_pin=runtime_pin.model_dump(mode="json"),
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
            expected_revision=cursor.revision,
            expected_projection_basis_hash=command.expected_projection_hash,
            selection_pin=selection_pin,
            design_binding_set_pin=design_pin,
            runtime_binding_set_pin=runtime_pin,
            successor_projection_basis_hash="sha256:" + content_hash(material),
            recorded_at=now,
        )
        receipt = _address(
            WorkspaceArtifactOperationReceiptV1,
            "workspace-artifact-receipt",
            "receipt_ref",
            "receipt_hash",
            dict(
                **base,
                schema_version="workspace_artifact_operation_receipt.v1",
                operation=operation,
                artifact_pin=command.artifact_pin.model_dump(mode="json"),
                collection_pin=_pin(
                    "workspace_artifact_collection",
                    collection.collection_ref,
                    collection.collection_hash,
                ).model_dump(mode="json"),
                collection_revision=collection.revision,
                preparation_pin=None if clearing else command.preparation_pin.model_dump(mode="json"),
                preparation_catalog_pin=None,
                preparation_catalog_revision=None,
                selection_pin=selection_pin.model_dump(mode="json")
                if selected
                else None,
                selection_revision=selected.revision if selected else None,
                binding_set_pin=binding_pin.model_dump(mode="json"),
                binding_set_revision=snapshot.revision,
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
                outcome="CLEARED" if clearing else "BOUND",
                recorded_at=timestamp,
            ),
        )
        self._build._insert_prepared((self._build._prepare(receipt),))
        connection.execute(
            sql_query("INSERT INTO workspace_operation_request_index VALUES (?,?,?,?,?,?,?,?)", 'INSERT INTO workspace_operation_request_index VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'),
            (
                *scope,
                ctx.user_id,
                operation,
                command.request_id,
                request_hash,
                receipt.receipt_ref,
                receipt.receipt_hash,
            ),
        )
        return receipt
