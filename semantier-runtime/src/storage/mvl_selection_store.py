"""Workspace-scoped USE_FOR_MVL selection records.

Semantica owns graph and ontology versions. This store only records which
immutable native versions a Semantier MVL experiment has selected.
"""

from __future__ import annotations

import sqlite3
from storage.hosted_postgres import enabled,connect_authority,mapped_store,require_tables,sql_query,begin_write,integrity_errors
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


MVL_SELECTION_SCHEMA_VERSION = "mvl_selection.v1"
SelectionStatus = Literal["active", "superseded"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MvlSelection:
    schema_version: str
    selection_id: str
    experiment_id: str
    workspace_id: str
    profile_ref: str
    profile_version: str
    graph_version_ref: str
    ontology_version_ref: str | None
    graph_semantica_version_id: str
    ontology_semantica_version_id: str | None
    graph_tag: str
    ontology_tag: str | None
    graph_checksum: str
    ontology_checksum: str | None
    selected_by: str
    selected_at: str
    supersedes_selection_id: str | None
    status: SelectionStatus

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "selection_id": self.selection_id,
            "experiment_id": self.experiment_id,
            "workspace_id": self.workspace_id,
            "profile_ref": self.profile_ref,
            "profile_version": self.profile_version,
            "graph_version_ref": self.graph_version_ref,
            "ontology_version_ref": self.ontology_version_ref,
            "graph_semantica_version_id": self.graph_semantica_version_id,
            "ontology_semantica_version_id": self.ontology_semantica_version_id,
            "graph_tag": self.graph_tag,
            "ontology_tag": self.ontology_tag,
            "graph_checksum": self.graph_checksum,
            "ontology_checksum": self.ontology_checksum,
            "selected_by": self.selected_by,
            "selected_at": self.selected_at,
            "supersedes_selection_id": self.supersedes_selection_id,
            "status": self.status,
        }


class MvlSelectionNotFound(RuntimeError):
    """The requested MVL selection/ref is not in the workspace control store."""


class MvlSelectionConflict(RuntimeError):
    """The selection conflicts with an immutable or active selection invariant."""


class MvlSelectionStore:
    """Append-only active/superseded MVL selection manifest."""

    _process_locks: dict[Path, threading.RLock] = {}
    _process_locks_guard = threading.Lock()

    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        if not workspace_id:
            raise ValueError("workspace_id is required")
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_id = workspace_id
        self.root = self.workspace_root / ".semantier" / "mvl"
        if not enabled():self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "selection.sqlite3"
        self._initialize()

    @property
    def _lock(self) -> threading.RLock:
        with self._process_locks_guard:
            return self._process_locks.setdefault(self.db_path, threading.RLock())

    def _connect(self) -> sqlite3.Connection:
        if enabled():
            binding=mapped_store('standalone',self.db_path)
            if binding.get('workspace_id',self.workspace_id)!=self.workspace_id:
                raise ValueError('foreign workspace auxiliary binding')
            return connect_authority('standalone',self.db_path)
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        if enabled():
            with self._connect() as connection:require_tables(connection,'mvl_selections')
            return
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS mvl_selections (
                    selection_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    profile_ref TEXT NOT NULL,
                    profile_version TEXT NOT NULL,
                    graph_version_ref TEXT NOT NULL,
                    ontology_version_ref TEXT,
                    graph_semantica_version_id TEXT NOT NULL,
                    ontology_semantica_version_id TEXT,
                    graph_tag TEXT NOT NULL,
                    ontology_tag TEXT,
                    graph_checksum TEXT NOT NULL,
                    ontology_checksum TEXT,
                    selected_by TEXT NOT NULL,
                    selected_at TEXT NOT NULL,
                    supersedes_selection_id TEXT,
                    status TEXT NOT NULL CHECK (status IN ('active', 'superseded')),
                    UNIQUE (workspace_id, graph_version_ref),
                    UNIQUE (workspace_id, ontology_version_ref)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_mvl_selection_experiment ON mvl_selections (workspace_id, experiment_id, status)"
            )

    def create_selection(
        self,
        *,
        experiment_id: str,
        profile_ref: str,
        profile_version: str,
        graph_version_ref: str,
        graph_semantica_version_id: str,
        graph_tag: str,
        graph_checksum: str,
        selected_by: str,
        ontology_version_ref: str | None = None,
        ontology_semantica_version_id: str | None = None,
        ontology_tag: str | None = None,
        ontology_checksum: str | None = None,
    ) -> MvlSelection:
        required = {
            "experiment_id": experiment_id,
            "profile_ref": profile_ref,
            "profile_version": profile_version,
            "graph_version_ref": graph_version_ref,
            "graph_semantica_version_id": graph_semantica_version_id,
            "graph_tag": graph_tag,
            "graph_checksum": graph_checksum,
            "selected_by": selected_by,
        }
        missing = sorted(name for name, value in required.items() if not value)
        if missing:
            raise ValueError(f"mvl selection missing fields: {missing}")
        if not graph_version_ref.startswith(f"urn:semantier:mvl:graph:{self.workspace_id}:"):
            raise ValueError("graph_version_ref must be a Semantier MVL graph URN for this workspace")
        if ontology_version_ref and not ontology_version_ref.startswith(
            f"urn:semantier:mvl:ontology:{self.workspace_id}:"
        ):
            raise ValueError("ontology_version_ref must be a Semantier MVL ontology URN for this workspace")
        with self._lock, self._connect() as connection:
            if enabled():begin_write(connection)
            prior = connection.execute(
                sql_query("""
                SELECT selection_id FROM mvl_selections
                WHERE workspace_id = ? AND experiment_id = ? AND status = 'active'
                ORDER BY selected_at DESC, selection_id DESC LIMIT 1
                ""","\n                SELECT selection_id FROM mvl_selections\n                WHERE workspace_id = %s AND experiment_id = %s AND status = 'active'\n                ORDER BY selected_at DESC, selection_id DESC LIMIT 1\n                "),
                (self.workspace_id, experiment_id),
            ).fetchone()
            supersedes = prior["selection_id"] if prior else None
            if supersedes:
                connection.execute(
                    sql_query("UPDATE mvl_selections SET status = 'superseded' WHERE selection_id = ?","UPDATE mvl_selections SET status = 'superseded' WHERE selection_id = %s"),
                    (supersedes,),
                )
            selection = MvlSelection(
                schema_version=MVL_SELECTION_SCHEMA_VERSION,
                selection_id=f"mvl_sel_{uuid.uuid4().hex}",
                experiment_id=experiment_id,
                workspace_id=self.workspace_id,
                profile_ref=profile_ref,
                profile_version=profile_version,
                graph_version_ref=graph_version_ref,
                ontology_version_ref=ontology_version_ref,
                graph_semantica_version_id=graph_semantica_version_id,
                ontology_semantica_version_id=ontology_semantica_version_id,
                graph_tag=graph_tag,
                ontology_tag=ontology_tag,
                graph_checksum=graph_checksum,
                ontology_checksum=ontology_checksum,
                selected_by=selected_by,
                selected_at=_utc_now(),
                supersedes_selection_id=supersedes,
                status="active",
            )
            try:
                connection.execute(
                    sql_query("""
                    INSERT INTO mvl_selections VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,'\n                    INSERT INTO mvl_selections VALUES (\n                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n                    )\n                    '),
                    (
                        selection.selection_id,
                        selection.schema_version,
                        selection.experiment_id,
                        selection.workspace_id,
                        selection.profile_ref,
                        selection.profile_version,
                        selection.graph_version_ref,
                        selection.ontology_version_ref,
                        selection.graph_semantica_version_id,
                        selection.ontology_semantica_version_id,
                        selection.graph_tag,
                        selection.ontology_tag,
                        selection.graph_checksum,
                        selection.ontology_checksum,
                        selection.selected_by,
                        selection.selected_at,
                        selection.supersedes_selection_id,
                        selection.status,
                    ),
                )
            except integrity_errors() as exc:
                raise MvlSelectionConflict("MVL selection ref already exists in this workspace") from exc
            return selection

    def get_selection(self, selection_id: str) -> MvlSelection:
        row = self._fetch_one(sql_query("selection_id = ?",'selection_id = %s'), (selection_id,))
        if row is None:
            raise MvlSelectionNotFound("selection is not available in this workspace")
        return MvlSelection(**dict(row))

    def resolve_graph_ref(self, graph_version_ref: str) -> MvlSelection:
        row = self._fetch_one(sql_query("graph_version_ref = ?",'graph_version_ref = %s'), (graph_version_ref,))
        if row is None:
            raise MvlSelectionNotFound("graph ref is not selected in this workspace")
        return MvlSelection(**dict(row))

    def active_for_experiment(self, experiment_id: str) -> MvlSelection:
        row = self._fetch_one(
            sql_query("experiment_id = ? AND status = 'active'","experiment_id = %s AND status = 'active'"),
            (experiment_id,),
            order_by="selected_at DESC, selection_id DESC",
        )
        if row is None:
            raise MvlSelectionNotFound("experiment has no active MVL selection")
        return MvlSelection(**dict(row))

    def _fetch_one(
        self,
        where: str,
        params: tuple[Any, ...],
        *,
        order_by: str | None = None,
    ) -> sqlite3.Row | None:
        query = sql_query(f"SELECT * FROM mvl_selections WHERE workspace_id = ? AND {where}",f'SELECT * FROM mvl_selections WHERE workspace_id = %s AND {where}')
        if order_by:
            query += f" ORDER BY {order_by}"
        query += " LIMIT 1"
        with self._connect() as connection:
            return connection.execute(query, (self.workspace_id, *params)).fetchone()

