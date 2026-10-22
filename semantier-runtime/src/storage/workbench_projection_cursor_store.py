
from storage.hosted_postgres import sql_query,begin_write,in_transaction
"""Single persisted material cursor owner, within the workspace transaction."""

from datetime import datetime, timezone

from agents.auth_session import RequestContext
from contracts.knowledge_classification import RuntimeContextDTO
from contracts.ontology_building import ArtifactPinV1, ExploratoryGraphSnapshotV1
from contracts.ontology_workbench_context import (
    WorkbenchProjectionCursorHeadV3,
    WorkbenchEvidenceFocusV1,
    WorkspaceArtifactSelectionV1,
    LifecycleArtifactBindingSetV1,
)
from eos.artifact_hashing import content_hash
from services.runtime_context_transition_service import RuntimeAuthorityResolver
from services.runtime_session_service import GovernedRuntimeSessionResolver
from storage.ontology_build_store import OntologyBuildStore

_PRESERVE_FOCUS = object()
_PRESERVE_DESIGN_SNAPSHOT = object()


class WorkbenchProjectionCursorStore:
    def __init__(
        self,
        *,
        build_store: OntologyBuildStore,
        authority_resolver: RuntimeAuthorityResolver,
        session_resolver: GovernedRuntimeSessionResolver,
        session_ref: str,
    ):
        if (
            not isinstance(build_store, OntologyBuildStore)
            or not isinstance(authority_resolver, RuntimeAuthorityResolver)
            or not isinstance(session_resolver, GovernedRuntimeSessionResolver)
        ):
            raise TypeError("cursor requires production workspace owners")
        self._store = build_store
        self._authority = authority_resolver
        self._sessions = session_resolver
        self._session_ref = session_ref

    def _scope(self, ctx):
        if (
            not isinstance(ctx, RequestContext)
            or not ctx.authenticated
            or not ctx.organization_id
        ):
            raise ValueError("AUTHORITY_CONTEXT_REQUIRED")
        actor = RuntimeContextDTO(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            actor_id=ctx.user_id,
        )
        self._authority.require_current_actor(actor_context=actor, authorization_class="WORKSPACE_LOCAL")
        session = self._sessions.resolve(
            actor_context=actor, session_ref=self._session_ref, authorization_class="WORKSPACE_LOCAL"
        )
        return dict(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            session_id=session,
            actor_ref=ctx.user_id,
        )

    def _facts(self, scope, selection_pin, design_pin, runtime_pin):
        results = []
        for pin, model, kind, hash_field in (
            (
                selection_pin,
                WorkspaceArtifactSelectionV1,
                "workspace_artifact_selection",
                "selection_hash",
            ),
            (
                design_pin,
                LifecycleArtifactBindingSetV1,
                "lifecycle_artifact_binding_set",
                "binding_set_hash",
            ),
            (
                runtime_pin,
                LifecycleArtifactBindingSetV1,
                "lifecycle_artifact_binding_set",
                "binding_set_hash",
            ),
        ):
            if pin.artifact_kind != kind:
                raise ValueError("CURSOR_FACT_MISMATCH")
            value = self._store.get(
                organization_id=scope["organization_id"],
                workspace_id=scope["workspace_id"],
                artifact_ref=pin.artifact_ref,
                contract_type=model,
            )
            if value is None or getattr(value, hash_field) != pin.artifact_hash:
                raise ValueError("CURSOR_FACT_MISMATCH")
            results.append(value)
        if (
            results[1].lifecycle,
            results[1].role,
            results[2].lifecycle,
            results[2].role,
        ) != ("DESIGN", "SOURCE", "RUNTIME", "TARGET"):
            raise ValueError("CURSOR_FACT_MISMATCH")
        return results

    def exact_head(self, *, ctx: RequestContext, pin: ArtifactPinV1):
        scope = self._scope(ctx)
        if pin.artifact_kind != "workbench_projection_cursor_head":
            raise ValueError("CURSOR_PIN_MISMATCH")
        raw = self._store.get_raw(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, artifact_ref=pin.artifact_ref)
        if raw is None or raw["schema_version"] != "workbench_projection_cursor_head.v3":
            raise ValueError("CURSOR_PIN_MISMATCH")
        head = WorkbenchProjectionCursorHeadV3.model_validate(raw["payload"])
        if (head.cursor_head_ref, head.cursor_head_hash, head.session_id, head.actor_ref) != (pin.artifact_ref, pin.artifact_hash, scope["session_id"], scope["actor_ref"]) or raw["artifact_hash"] != pin.artifact_hash:
            raise ValueError("CURSOR_PIN_MISMATCH")
        self._focus(scope, getattr(head, "evidence_focus_pin", None))
        self._design_snapshot(ctx, getattr(head, "design_snapshot_pin", None))
        return head

    def _design_snapshot(self, ctx, pin):
        if pin is None:
            return None
        if pin.artifact_kind != "exploratory_graph_snapshot":
            raise ValueError("CURSOR_FACT_MISMATCH")
        actor = RuntimeContextDTO(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            actor_id=ctx.user_id,
        )
        values = self._authority.resolve_pins(actor_context=actor, pins=(pin,))
        if len(values) != 1 or not isinstance(values[0], ExploratoryGraphSnapshotV1):
            raise ValueError("CURSOR_FACT_MISMATCH")
        return values[0]

    def _focus(self, scope, pin):
        if pin is None:
            return None
        if pin.artifact_kind != "workbench_evidence_focus":
            raise ValueError("CURSOR_FACT_MISMATCH")
        focus = self._store.get(organization_id=scope["organization_id"], workspace_id=scope["workspace_id"], artifact_ref=pin.artifact_ref, contract_type=WorkbenchEvidenceFocusV1)
        if focus is None or (focus.focus_ref, focus.focus_hash, focus.organization_id, focus.workspace_id, focus.session_id, focus.actor_ref) != (pin.artifact_ref, pin.artifact_hash, scope["organization_id"], scope["workspace_id"], scope["session_id"], scope["actor_ref"]):
            raise ValueError("CURSOR_FACT_MISMATCH")
        return focus

    def get(
        self, *, ctx: RequestContext, cursor_ref: str
    ) -> WorkbenchProjectionCursorHeadV3:
        scope = self._scope(ctx)
        row = self._store._connection.execute(
            sql_query("SELECT revision,cursor_head_ref,cursor_head_hash,projection_basis_hash,selection_ref,selection_hash,design_binding_set_ref,design_binding_set_hash,runtime_binding_set_ref,runtime_binding_set_hash,design_snapshot_ref,design_snapshot_hash FROM workbench_projection_cursors WHERE organization_id=? AND workspace_id=? AND session_id=? AND actor_ref=? AND cursor_ref=?",'SELECT revision,cursor_head_ref,cursor_head_hash,projection_basis_hash,selection_ref,selection_hash,design_binding_set_ref,design_binding_set_hash,runtime_binding_set_ref,runtime_binding_set_hash,design_snapshot_ref,design_snapshot_hash FROM workbench_projection_cursors WHERE organization_id=%s AND workspace_id=%s AND session_id=%s AND actor_ref=%s AND cursor_ref=%s'),
            (*scope.values(), cursor_ref),
        ).fetchone()
        if row is None:
            raise ValueError("CURSOR_NOT_FOUND")
        head = self.exact_head(ctx=ctx, pin=ArtifactPinV1(artifact_kind="workbench_projection_cursor_head", artifact_ref=row[1], artifact_hash=row[2]))
        if head is None or (head.session_id, head.actor_ref, head.cursor_ref) != (
            scope["session_id"],
            ctx.user_id,
            cursor_ref,
        ):
            raise ValueError("CURSOR_PIN_MISMATCH")
        design_snapshot_pin = getattr(head, "design_snapshot_pin", None)
        expected = (
            head.revision,
            head.cursor_head_ref,
            head.cursor_head_hash,
            head.projection_basis_hash,
            head.selection_pin.artifact_ref,
            head.selection_pin.artifact_hash,
            head.design_binding_set_pin.artifact_ref,
            head.design_binding_set_pin.artifact_hash,
            head.runtime_binding_set_pin.artifact_ref,
            head.runtime_binding_set_pin.artifact_hash,
            design_snapshot_pin.artifact_ref if design_snapshot_pin else None,
            design_snapshot_pin.artifact_hash if design_snapshot_pin else None,
        )
        if row != expected:
            raise ValueError("CURSOR_PIN_MISMATCH")
        self._facts(
            scope,
            head.selection_pin,
            head.design_binding_set_pin,
            head.runtime_binding_set_pin,
        )
        return head

    def advance_on_connection(
        self,
        connection,
        *,
        ctx: RequestContext,
        cursor_ref: str,
        expected_head_pin: ArtifactPinV1,
        expected_revision: int,
        expected_projection_basis_hash: str,
        selection_pin: ArtifactPinV1,
        design_binding_set_pin: ArtifactPinV1,
        runtime_binding_set_pin: ArtifactPinV1,
        successor_projection_basis_hash: str,
        recorded_at: datetime,
        evidence_focus_pin=_PRESERVE_FOCUS,
        design_snapshot_pin=_PRESERVE_DESIGN_SNAPSHOT,
    ) -> WorkbenchProjectionCursorHeadV3:
        if connection is not self._store._connection or not in_transaction(connection):
            raise RuntimeError("cursor writes require the active workspace transaction")
        scope = self._scope(ctx)
        previous = self.get(ctx=ctx, cursor_ref=cursor_ref)
        if (
            type(expected_revision) is not int
            or expected_revision != previous.revision
            or expected_projection_basis_hash != previous.projection_basis_hash
            or expected_head_pin.artifact_kind != "workbench_projection_cursor_head"
            or expected_head_pin.artifact_ref != previous.cursor_head_ref
            or expected_head_pin.artifact_hash != previous.cursor_head_hash
        ):
            raise ValueError("STALE_MATERIAL_CONTEXT")
        self._facts(
            scope, selection_pin, design_binding_set_pin, runtime_binding_set_pin
        )
        prior_focus_pin = previous.evidence_focus_pin
        prior_snapshot_pin = previous.design_snapshot_pin
        if evidence_focus_pin is _PRESERVE_FOCUS:
            evidence_focus_pin = prior_focus_pin
        if design_snapshot_pin is _PRESERVE_DESIGN_SNAPSHOT:
            design_snapshot_pin = prior_snapshot_pin
        if design_snapshot_pin != prior_snapshot_pin:
            evidence_focus_pin = None
        self._design_snapshot(ctx, design_snapshot_pin)
        focus = self._focus(scope, evidence_focus_pin)
        if evidence_focus_pin != prior_focus_pin and focus is not None:
            prior_focus = self._focus(scope, prior_focus_pin)
            if focus.predecessor_pin != prior_focus_pin or focus.revision != (prior_focus.revision + 1 if prior_focus else 0):
                raise ValueError("CURSOR_FACT_MISMATCH")
            prior_selection, _, _ = self._facts(scope, previous.selection_pin, previous.design_binding_set_pin, previous.runtime_binding_set_pin)
            if focus.focus.expected_selection_pin != previous.selection_pin or focus.focus.expected_selection_revision != prior_selection.revision:
                raise ValueError("STALE_SELECTION_HEAD")
        material = dict(
            **scope,
            cursor_ref=cursor_ref,
            revision=previous.revision + 1,
            selection_pin=selection_pin.model_dump(mode="json"),
            design_binding_set_pin=design_binding_set_pin.model_dump(mode="json"),
            runtime_binding_set_pin=runtime_binding_set_pin.model_dump(mode="json"),
            evidence_focus_pin=evidence_focus_pin.model_dump(mode="json") if evidence_focus_pin else None,
            design_snapshot_pin=design_snapshot_pin.model_dump(mode="json") if design_snapshot_pin else None,
        )
        if successor_projection_basis_hash != "sha256:" + content_hash(material):
            raise ValueError("CURSOR_PROJECTION_BASIS_MISMATCH")
        basis = dict(
            **material,
            schema_version="workbench_projection_cursor_head.v3",
            predecessor_pin=expected_head_pin.model_dump(mode="json"),
            projection_basis_hash=successor_projection_basis_hash,
            recorded_at=recorded_at.isoformat().replace("+00:00", "Z"),
        )
        digest = content_hash(basis)
        head = WorkbenchProjectionCursorHeadV3.model_validate(
            {
                **basis,
                "cursor_head_ref": "workbench-projection-cursor-head:" + digest,
                "cursor_head_hash": "sha256:" + digest,
            }
        )
        self._store._insert_prepared((self._store._prepare(head),))
        changed = connection.execute(
            sql_query("UPDATE workbench_projection_cursors SET revision=?,cursor_head_ref=?,cursor_head_hash=?,projection_basis_hash=?,selection_ref=?,selection_hash=?,design_binding_set_ref=?,design_binding_set_hash=?,runtime_binding_set_ref=?,runtime_binding_set_hash=?,design_snapshot_ref=?,design_snapshot_hash=? WHERE organization_id=? AND workspace_id=? AND session_id=? AND actor_ref=? AND cursor_ref=? AND revision=? AND cursor_head_ref=? AND cursor_head_hash=? AND projection_basis_hash=?",'UPDATE workbench_projection_cursors SET revision=%s,cursor_head_ref=%s,cursor_head_hash=%s,projection_basis_hash=%s,selection_ref=%s,selection_hash=%s,design_binding_set_ref=%s,design_binding_set_hash=%s,runtime_binding_set_ref=%s,runtime_binding_set_hash=%s,design_snapshot_ref=%s,design_snapshot_hash=%s WHERE organization_id=%s AND workspace_id=%s AND session_id=%s AND actor_ref=%s AND cursor_ref=%s AND revision=%s AND cursor_head_ref=%s AND cursor_head_hash=%s AND projection_basis_hash=%s'),
            (
                head.revision,
                head.cursor_head_ref,
                head.cursor_head_hash,
                head.projection_basis_hash,
                selection_pin.artifact_ref,
                selection_pin.artifact_hash,
                design_binding_set_pin.artifact_ref,
                design_binding_set_pin.artifact_hash,
                runtime_binding_set_pin.artifact_ref,
                runtime_binding_set_pin.artifact_hash,
                design_snapshot_pin.artifact_ref if design_snapshot_pin else None,
                design_snapshot_pin.artifact_hash if design_snapshot_pin else None,
                *scope.values(),
                cursor_ref,
                previous.revision,
                previous.cursor_head_ref,
                previous.cursor_head_hash,
                previous.projection_basis_hash,
            ),
        ).rowcount
        if changed != 1:
            raise ValueError("STALE_MATERIAL_CONTEXT")
        return head

    def select_design_snapshot_on_connection(
        self,
        connection,
        *,
        ctx: RequestContext,
        cursor_ref: str,
        expected_revision: int,
        expected_projection_basis_hash: str,
        design_snapshot_pin: ArtifactPinV1 | None,
        recorded_at: datetime,
    ) -> WorkbenchProjectionCursorHeadV3:
        previous = self.get(ctx=ctx, cursor_ref=cursor_ref)
        material = dict(
            **self._scope(ctx),
            cursor_ref=cursor_ref,
            revision=previous.revision + 1,
            selection_pin=previous.selection_pin.model_dump(mode="json"),
            design_binding_set_pin=previous.design_binding_set_pin.model_dump(mode="json"),
            runtime_binding_set_pin=previous.runtime_binding_set_pin.model_dump(mode="json"),
            evidence_focus_pin=None if design_snapshot_pin != previous.design_snapshot_pin else previous.evidence_focus_pin.model_dump(mode="json") if previous.evidence_focus_pin else None,
            design_snapshot_pin=design_snapshot_pin.model_dump(mode="json") if design_snapshot_pin else None,
        )
        return self.advance_on_connection(
            connection,
            ctx=ctx,
            cursor_ref=cursor_ref,
            expected_head_pin=ArtifactPinV1(
                artifact_kind="workbench_projection_cursor_head",
                artifact_ref=previous.cursor_head_ref,
                artifact_hash=previous.cursor_head_hash,
            ),
            expected_revision=expected_revision,
            expected_projection_basis_hash=expected_projection_basis_hash,
            selection_pin=previous.selection_pin,
            design_binding_set_pin=previous.design_binding_set_pin,
            runtime_binding_set_pin=previous.runtime_binding_set_pin,
            successor_projection_basis_hash="sha256:" + content_hash(material),
            recorded_at=recorded_at,
            design_snapshot_pin=design_snapshot_pin,
        )

    def create_genesis_on_connection(
        self,
        connection,
        *,
        ctx: RequestContext,
        cursor_ref: str,
        selection_pin: ArtifactPinV1,
        design_binding_set_pin: ArtifactPinV1,
        runtime_binding_set_pin: ArtifactPinV1,
    ) -> WorkbenchProjectionCursorHeadV3:
        if connection is not self._store._connection or not in_transaction(connection):
            raise RuntimeError("cursor writes require the active workspace transaction")
        scope = self._scope(ctx)
        selection, design, runtime = self._facts(
            scope, selection_pin, design_binding_set_pin, runtime_binding_set_pin
        )
        if (
            any(value.revision != 0 for value in (selection, design, runtime))
            or selection.artifact_pin is not None
            or design.entries
            or runtime.entries
        ):
            raise ValueError("CURSOR_GENESIS_REQUIRES_EMPTY_FACTS")
        material = dict(
            **scope,
            cursor_ref=cursor_ref,
            revision=0,
            selection_pin=selection_pin.model_dump(mode="json"),
            design_binding_set_pin=design_binding_set_pin.model_dump(mode="json"),
            runtime_binding_set_pin=runtime_binding_set_pin.model_dump(mode="json"),
            evidence_focus_pin=None,
            design_snapshot_pin=None,
        )
        basis = dict(
            **material,
            schema_version="workbench_projection_cursor_head.v3",
            predecessor_pin=None,
            projection_basis_hash="sha256:" + content_hash(material),
            recorded_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        digest = content_hash(basis)
        head = WorkbenchProjectionCursorHeadV3.model_validate(
            {
                **basis,
                "cursor_head_ref": "workbench-projection-cursor-head:" + digest,
                "cursor_head_hash": "sha256:" + digest,
            }
        )
        self._store._insert_prepared((self._store._prepare(head),))
        connection.execute(
            sql_query("INSERT INTO workbench_projection_cursors (organization_id,workspace_id,session_id,actor_ref,cursor_ref,revision,cursor_head_ref,cursor_head_hash,projection_basis_hash,selection_ref,selection_hash,design_binding_set_ref,design_binding_set_hash,runtime_binding_set_ref,runtime_binding_set_hash,design_snapshot_ref,design_snapshot_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",'INSERT INTO workbench_projection_cursors (organization_id,workspace_id,session_id,actor_ref,cursor_ref,revision,cursor_head_ref,cursor_head_hash,projection_basis_hash,selection_ref,selection_hash,design_binding_set_ref,design_binding_set_hash,runtime_binding_set_ref,runtime_binding_set_hash,design_snapshot_ref,design_snapshot_hash) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'),
            (
                *scope.values(),
                cursor_ref,
                0,
                head.cursor_head_ref,
                head.cursor_head_hash,
                head.projection_basis_hash,
                selection_pin.artifact_ref,
                selection_pin.artifact_hash,
                design_binding_set_pin.artifact_ref,
                design_binding_set_pin.artifact_hash,
                runtime_binding_set_pin.artifact_ref,
                runtime_binding_set_pin.artifact_hash,
                None,
                None,
            ),
        )
        return head
