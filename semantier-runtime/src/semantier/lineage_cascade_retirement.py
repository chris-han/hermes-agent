"""Real cross-store execution for lineage-owned cascade retirement."""

from __future__ import annotations

from storage import hosted_postgres as hosted

import uuid
from datetime import datetime, timezone
from typing import Iterable

from semantier.lineage_cascade_inventory import (
    CascadeInventory,
    CascadeInventoryEntry,
    LineageCascadeInventoryService,
)
from semantier.semantica_context_graph.artifact_repository import (
    ContextGraphArtifactRepository,
)
from storage.graph_schema_discovery_store import GraphSchemaDiscoveryStore
from storage.knowledge_extraction_store import KnowledgeExtractionStore


class CascadeRetirementError(RuntimeError):
    pass


class RequiresGovernedRetirement(CascadeRetirementError):
    pass


class GovernedReasonRequired(CascadeRetirementError):
    pass


class ActiveResidualsRemain(CascadeRetirementError):
    def __init__(self, residuals: tuple[CascadeInventoryEntry, ...]) -> None:
        self.residuals = residuals
        super().__init__(f"{len(residuals)} active lineage materializations remain")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LineageCascadeRetirementService:
    def __init__(
        self,
        *,
        inventory_service: LineageCascadeInventoryService,
        extraction_store: KnowledgeExtractionStore,
        artifact_repositories: Iterable[ContextGraphArtifactRepository],
        schema_stores: Iterable[GraphSchemaDiscoveryStore],
        actor_id: str,
    ) -> None:
        self.inventory_service = inventory_service
        self.extraction_store = extraction_store
        self.artifact_repositories = tuple(artifact_repositories)
        self.schema_stores = tuple(schema_stores)
        self.actor_id = actor_id

    def tombstone_metadata(self, inventory: CascadeInventory) -> None:
        self.inventory_service.assert_scope_membership(inventory)
        store = self.extraction_store
        now = _now()
        root_definitions = {
            "source": ("knowledge_builder_sources", "source_id"),
            "discovery": ("knowledge_builder_discovery_runs", "discovery_run_id"),
            "extraction": ("knowledge_builder_extraction_runs", "extraction_run_id"),
        }
        with store._lock:
            try:
                hosted.begin_write(store._conn)
                for entry in inventory.entries:
                    definition = root_definitions.get(entry.category)
                    if definition is not None:
                        table, id_column = definition
                        cursor = store._conn.execute(
                            hosted.sql_query(f"UPDATE {table} SET retirement_state = 'tombstoned' "
                            f"WHERE {id_column} = ? AND retirement_state = 'active'", f"UPDATE {table} SET retirement_state = 'tombstoned' WHERE {id_column} = %s AND retirement_state = 'active'"),
                            (entry.member_id,),
                        )
                        if cursor.rowcount:
                            self._append_eos_event(
                                inventory=inventory,
                                table=table,
                                member_id=entry.member_id,
                                reason="lineage cascade metadata retirement",
                                now=now,
                            )
                    elif entry.category == "representation":
                        latest = store._conn.execute(
                            hosted.sql_query("""SELECT next_state FROM source_representation_lifecycle_events
                            WHERE representation_id = ?
                            ORDER BY created_at DESC, rowid DESC LIMIT 1""", 'SELECT next_state FROM source_representation_lifecycle_events\n                            WHERE representation_id = %s\n                            ORDER BY created_at DESC, event_sequence DESC LIMIT 1'),
                            (entry.member_id,),
                        ).fetchone()
                        if latest is not None and latest["next_state"] == "active":
                            store._append_representation_event(
                                entry.member_id,
                                previous_state="active",
                                next_state="tombstoned",
                                actor_id=self.actor_id,
                                operation_id=inventory.operation_id,
                                reason="lineage cascade metadata retirement",
                                created_at=now,
                            )
                    elif entry.category == "origin":
                        cursor = store._conn.execute(
                            hosted.sql_query("""UPDATE contextgraph_artifact_origins
                            SET origin_kind = 'source_tombstoned'
                            WHERE origin_id = ? AND origin_kind != 'source_tombstoned'""", "UPDATE contextgraph_artifact_origins\n                            SET origin_kind = 'source_tombstoned'\n                            WHERE origin_id = %s AND origin_kind != 'source_tombstoned'"),
                            (entry.member_id,),
                        )
                        if cursor.rowcount:
                            self._append_eos_event(
                                inventory=inventory,
                                table="contextgraph_artifact_origins",
                                member_id=entry.member_id,
                                reason="lineage cascade origin retirement",
                                now=now,
                            )
                    elif entry.category == "graph_release":
                        cursor = store._conn.execute(
                            hosted.sql_query("""UPDATE tender_graph_releases
                            SET retirement_state = 'tombstoned'
                            WHERE graph_version = ? AND retirement_state = 'active'""", "UPDATE tender_graph_releases\n                            SET retirement_state = 'tombstoned'\n                            WHERE graph_version = %s AND retirement_state = 'active'"),
                            (entry.member_id,),
                        )
                        if cursor.rowcount:
                            self._append_eos_event(
                                inventory=inventory,
                                table="tender_graph_releases",
                                member_id=entry.member_id,
                                reason="lineage cascade graph release retirement",
                                now=now,
                            )
                store._conn.commit()
            except Exception:
                store._conn.rollback()
                raise

    def _append_eos_event(
        self,
        *,
        inventory: CascadeInventory,
        table: str,
        member_id: str,
        reason: str,
        now: str,
    ) -> None:
        self.extraction_store._conn.execute(
            hosted.sql_query("""INSERT OR IGNORE INTO lineage_materialization_retirement_events (
            event_id, operation_id, lineage_id, store_name, table_name, member_id,
            previous_state, next_state, actor_id, reason, created_at
            ) VALUES (?, ?, ?, 'eos', ?, ?, 'active', 'tombstoned', ?, ?, ?)""", "INSERT INTO lineage_materialization_retirement_events (\n            event_id, operation_id, lineage_id, store_name, table_name, member_id,\n            previous_state, next_state, actor_id, reason, created_at\n            ) VALUES (%s, %s, %s, 'eos', %s, %s, 'active', 'tombstoned', %s, %s, %s) ON CONFLICT DO NOTHING"),
            (
                f"evt_{uuid.uuid4().hex}",
                inventory.operation_id,
                inventory.lineage_id,
                table,
                member_id,
                self.actor_id,
                reason,
                now,
            ),
        )

    def tombstone_artifacts(
        self,
        inventory: CascadeInventory,
        *,
        governed_reason: str | None,
    ) -> None:
        if inventory.retirement_mode == "ordinary" and any(
            entry.authority_state == "authoritative" for entry in inventory.entries
        ):
            raise RequiresGovernedRetirement(
                "authoritative materializations require governed retirement"
            )
        if inventory.retirement_mode == "governed" and not str(
            governed_reason or ""
        ).strip():
            raise GovernedReasonRequired("governed retirement reason is required")

        reason = str(governed_reason).strip() if governed_reason is not None else None
        for repository in self.artifact_repositories:
            repository.tombstone_artifact_chain(
                source_id=inventory.lineage_id,
                operation_id=inventory.operation_id,
                actor_id=self.actor_id,
                reason=reason,
            )
        for store in self.schema_stores:
            self._tombstone_schema_store(
                store=store,
                inventory=inventory,
                reason=reason or "ordinary lineage retirement",
            )

    def _tombstone_schema_store(
        self,
        *,
        store: GraphSchemaDiscoveryStore,
        inventory: CascadeInventory,
        reason: str,
    ) -> None:
        now = _now()
        with store._lock, store._connect() as connection:
            hosted.begin_write(connection)
            for table in (
                "graph_schema_discovery_runs",
                "semantic_discovery_decisions",
                "graph_schema_discovery_briefs",
                "llm_proposal_evidence",
            ):
                connection.execute(
                    hosted.sql_query(f"UPDATE {table} SET retirement_state = 'tombstoned' "
                    "WHERE lineage_id = ? AND retirement_state = 'active'", f"UPDATE {table} SET retirement_state = 'tombstoned' WHERE lineage_id = %s AND retirement_state = 'active'"),
                    (inventory.lineage_id,),
                )
            connection.execute(
                hosted.sql_query("""INSERT OR IGNORE INTO lineage_retirement_events (
                event_id, operation_id, lineage_id, actor_id, reason,
                previous_state, next_state, created_at
                ) VALUES (?, ?, ?, ?, ?, 'active', 'tombstoned', ?)""", "INSERT INTO lineage_retirement_events (\n                event_id, operation_id, lineage_id, actor_id, reason,\n                previous_state, next_state, created_at\n                ) VALUES (%s, %s, %s, %s, %s, 'active', 'tombstoned', %s) ON CONFLICT DO NOTHING"),
                (
                    f"evt_{uuid.uuid4().hex}",
                    inventory.operation_id,
                    inventory.lineage_id,
                    self.actor_id,
                    reason,
                    now,
                ),
            )
            connection.commit()

    def assert_retired(self, inventory: CascadeInventory) -> None:
        residuals = self.inventory_service.active_residuals(inventory)
        if residuals:
            raise ActiveResidualsRemain(residuals)


__all__ = [
    "ActiveResidualsRemain",
    "CascadeRetirementError",
    "GovernedReasonRequired",
    "LineageCascadeRetirementService",
    "RequiresGovernedRetirement",
]
