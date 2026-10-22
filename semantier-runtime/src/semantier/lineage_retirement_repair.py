"""Deterministic, hash-gated repair of legacy lineage retirement residuals."""

from __future__ import annotations

from storage import hosted_postgres as hosted

import hashlib
import json
import os
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from semantier.lineage_cascade_inventory import LineageCascadeInventoryService
from semantier.lineage_cascade_retirement import LineageCascadeRetirementService
from semantier.workspace_retirement_stager import WorkspaceRetirementStager
from storage.knowledge_extraction_store import KnowledgeExtractionStore


REPORT_SCHEMA_VERSION = "contextgraph_lineage_repair_report.v1"
RESULT_SCHEMA_VERSION = "contextgraph_lineage_repair_result.v1"


class LineageRetirementRepairError(RuntimeError):
    pass


class RepairReportChanged(LineageRetirementRepairError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(payload: dict[str, Any], field: str) -> str:
    material = dict(payload)
    material.pop(field, None)
    return "sha256:" + hashlib.sha256(_canonical(material).encode()).hexdigest()


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(_canonical(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class LineageRetirementRepairService:
    def __init__(
        self,
        *,
        workspace_root: str | Path,
        workspace_id: str,
        organization_id: str,
        actor_id: str,
        extraction_store: KnowledgeExtractionStore,
        inventory_service: LineageCascadeInventoryService,
        cascade_service: LineageCascadeRetirementService,
        stager: WorkspaceRetirementStager,
    ) -> None:
        root = Path(workspace_root)
        if not root.is_absolute():
            root = root.resolve()
        self.workspace_root = root.resolve()
        self.workspace_id = workspace_id
        self.organization_id = organization_id
        self.actor_id = actor_id
        self.store = extraction_store
        self.inventory_service = inventory_service
        self.cascade_service = cascade_service
        self.stager = stager

    @property
    def reports_root(self) -> Path:
        return self.workspace_root / "runs" / "contextgraph-retirement-repair"

    def _eligible_rows(self) -> list[Any]:
        rows = self.store._conn.execute(
            hosted.sql_query("""SELECT operation.*,
                      source.workspace_id AS governed_workspace_id,
                      source.organization_id AS governed_organization_id
            FROM source_retirement_operations operation
            JOIN knowledge_builder_sources source
              ON source.source_id = operation.source_id
            WHERE source.workspace_id = ? AND source.organization_id = ?
              AND (operation.workspace_id IS NULL OR operation.workspace_id = source.workspace_id)
              AND (operation.organization_id IS NULL OR operation.organization_id = source.organization_id)
              AND operation.state IN ('completed', 'failed/retryable')
            ORDER BY operation.created_at, operation.operation_id""", "SELECT operation.*,\n                      source.workspace_id AS governed_workspace_id,\n                      source.organization_id AS governed_organization_id\n            FROM source_retirement_operations operation\n            JOIN knowledge_builder_sources source\n              ON source.source_id = operation.source_id\n            WHERE source.workspace_id = %s AND source.organization_id = %s\n              AND (operation.workspace_id IS NULL OR operation.workspace_id = source.workspace_id)\n              AND (operation.organization_id IS NULL OR operation.organization_id = source.organization_id)\n              AND operation.state IN ('completed', 'failed/retryable')\n            ORDER BY operation.created_at, operation.operation_id"),
            (self.workspace_id, self.organization_id),
        ).fetchall()
        eligible = []
        for row in rows:
            if row["state"] == "completed":
                eligible.append(row)
                continue
            checkpoint = str(row["failure_stage"] or "")
            inventory_path = (
                self.workspace_root / "runs" / "contextgraph-retirement"
                / str(row["operation_id"]) / "cascade-inventory.v1.json"
            )
            if not inventory_path.is_file() or checkpoint not in {
                "requested", "file_staged", "metadata_tombstoned", "artifacts_tombstoned"
            }:
                eligible.append(row)
        return eligible

    @staticmethod
    def _entry_action(operation_id: str, entry) -> dict[str, Any]:
        return {
            "operation_id": operation_id,
            "lineage_id": entry.lineage_id,
            "category": entry.category,
            "store": entry.store,
            "member_id": entry.member_id,
            "relative_path": entry.relative_path,
            "authority_state": entry.authority_state,
            "content_hash": entry.content_hash,
        }

    def _scan(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        operations: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        ambiguous: list[dict[str, Any]] = []
        rows = self._eligible_rows()
        selected_by_lineage: dict[str, Any] = {}
        for row in rows:
            mode = str(row["retirement_mode"] or "ordinary")
            operation = {
                "operation_id": str(row["operation_id"]),
                "lineage_id": str(row["source_id"]),
                "state": str(row["state"]),
                "retirement_mode": mode,
            }
            operations.append(operation)
            lineage_id = str(row["source_id"])
            current = selected_by_lineage.get(lineage_id)
            rank = (
                1 if row["state"] == "completed" else 0,
                str(row["updated_at"]),
                str(row["operation_id"]),
            )
            if current is None or rank > current[0]:
                selected_by_lineage[lineage_id] = (rank, row)

        for lineage_id in sorted(selected_by_lineage):
            row = selected_by_lineage[lineage_id][1]
            mode = str(row["retirement_mode"] or "ordinary")
            inventory = self.inventory_service._build(
                operation_id=str(row["operation_id"]),
                lineage_id=lineage_id,
                retirement_mode=mode,
                created_at=str(row["created_at"]),
            )
            residuals = self.inventory_service.active_residuals(inventory)
            authoritative = [item for item in residuals if item.authority_state == "authoritative"]
            if authoritative and mode != "governed":
                ambiguous.extend(
                    {
                        **self._entry_action(str(row["operation_id"]), item),
                        "reason": "authoritative_retirement_not_governed",
                    }
                    for item in authoritative
                )
                continue
            actions.extend(
                self._entry_action(str(row["operation_id"]), item)
                for item in residuals
            )
        operations.sort(key=lambda item: (item["lineage_id"], item["operation_id"]))
        actions.sort(key=lambda item: tuple(str(item.get(key) or "") for key in (
            "operation_id", "category", "store", "member_id", "relative_path"
        )))
        ambiguous.sort(key=lambda item: tuple(str(item.get(key) or "") for key in (
            "operation_id", "reason", "store", "member_id"
        )))
        return operations, actions, ambiguous

    def dry_run(self, *, repair_id: str | None = None) -> dict[str, Any]:
        repair_id = str(repair_id or f"repair_{uuid.uuid4().hex}").strip()
        if not repair_id or repair_id != Path(repair_id).name or "/" in repair_id or "\\" in repair_id:
            raise ValueError("repair_id must be a safe path segment")
        operations, actions, ambiguous = self._scan()
        payload: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "repair_id": repair_id,
            "workspace_id": self.workspace_id,
            "organization_id": self.organization_id,
            "actor_id": self.actor_id,
            "mode": "dry-run",
            "created_at": _now(),
            "eligible_operations": operations,
            "proposed_actions": actions,
            "ambiguous_entries": ambiguous,
            "counts_by_store": dict(sorted(Counter(item["store"] for item in actions).items())),
            "before_active_total": len(actions),
            "after_active_total": None,
            "report_hash": "",
        }
        payload["report_hash"] = _hash(payload, "report_hash")
        _write_once(self.reports_root / repair_id / "repair-report.v1.json", payload)
        return payload

    def _find_report(self, report_hash: str) -> tuple[Path, dict[str, Any]]:
        matches = []
        for path in sorted(self.reports_root.glob("*/repair-report.v1.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("report_hash") == report_hash:
                matches.append((path, payload))
        if len(matches) != 1:
            raise LineageRetirementRepairError("exactly one immutable repair report must match")
        path, payload = matches[0]
        if payload.get("schema_version") != REPORT_SCHEMA_VERSION or _hash(payload, "report_hash") != report_hash:
            raise LineageRetirementRepairError("repair report hash is invalid")
        return path, payload

    def apply(self, *, report_hash: str) -> dict[str, Any]:
        report_path, report = self._find_report(report_hash)
        result_path = report_path.with_name("repair-result.v1.json")
        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        operations, actions, ambiguous = self._scan()
        if (
            operations != report["eligible_operations"]
            or actions != report["proposed_actions"]
            or ambiguous != report["ambiguous_entries"]
        ):
            raise RepairReportChanged("repair scope changed after dry-run")

        actions_by_operation: dict[str, list[dict[str, Any]]] = {}
        for action in actions:
            actions_by_operation.setdefault(action["operation_id"], []).append(action)
        before_total = len(actions)
        for operation in operations:
            operation_id = operation["operation_id"]
            if operation_id not in actions_by_operation:
                continue
            inventory = self.inventory_service._build(
                operation_id=operation_id,
                lineage_id=operation["lineage_id"],
                retirement_mode=operation["retirement_mode"],
                created_at=report["created_at"],
            )
            manifest_path = self.stager.stage_inventory(inventory)
            self.cascade_service.tombstone_metadata(inventory)
            self.cascade_service.tombstone_artifacts(
                inventory,
                governed_reason="approved deterministic legacy repair"
                if inventory.retirement_mode == "governed" else None,
            )
            self.cascade_service.assert_retired(inventory)
            self.stager.unlink(staged_path=manifest_path)
            after = len(self.inventory_service.active_residuals(inventory))
            self.store._conn.execute(
                hosted.sql_query("""INSERT OR IGNORE INTO lineage_repair_events (
                event_id, repair_id, report_hash, operation_id, lineage_id,
                workspace_id, organization_id, actor_id, action_count,
                before_active_count, after_active_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 'INSERT INTO lineage_repair_events (\n                event_id, repair_id, report_hash, operation_id, lineage_id,\n                workspace_id, organization_id, actor_id, action_count,\n                before_active_count, after_active_count, created_at\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING'),
                (
                    f"repair_evt_{uuid.uuid4().hex}", report["repair_id"], report_hash,
                    operation_id, inventory.lineage_id, self.workspace_id,
                    self.organization_id, self.actor_id,
                    len(actions_by_operation[operation_id]),
                    len(actions_by_operation[operation_id]), after, _now(),
                ),
            )
            self.store._conn.commit()

        _operations_after, actions_after, ambiguous_after = self._scan()
        result: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "repair_id": report["repair_id"],
            "report_hash": report_hash,
            "workspace_id": self.workspace_id,
            "organization_id": self.organization_id,
            "actor_id": self.actor_id,
            "mode": "apply",
            "created_at": _now(),
            "before_active_total": before_total,
            "after_active_total": len(actions_after),
            "ambiguous_entries": ambiguous_after,
            "applied_action_count": before_total - len(actions_after),
            "result_hash": "",
        }
        result["result_hash"] = _hash(result, "result_hash")
        _write_once(result_path, result)
        return result


__all__ = [
    "LineageRetirementRepairError",
    "LineageRetirementRepairService",
    "RepairReportChanged",
    "REPORT_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
]
