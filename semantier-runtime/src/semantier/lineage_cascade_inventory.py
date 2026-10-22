"""Deterministic, workspace-governed lineage cascade inventories."""

from __future__ import annotations

from storage import hosted_postgres as hosted

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Protocol

from storage.knowledge_extraction_store import KnowledgeExtractionStore
from semantier.governed_upload_ref import (
    KNOWLEDGE_UPLOAD_REF_PREFIX,
    decode_governed_upload_ref,
)
from storage.graph_schema_discovery_store import GraphSchemaDiscoveryStore
from semantier.semantica_context_graph.artifact_repository import (
    ContextGraphArtifactRepository,
)


INVENTORY_SCHEMA_VERSION = "contextgraph_lineage_cascade_inventory.v1"
RetirementMode = Literal["ordinary", "governed"]
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._-]{0,255}$")


class InventoryError(RuntimeError):
    pass


class InventoryChanged(InventoryError):
    pass


class InventoryScopeMismatch(InventoryError):
    pass


@dataclass(frozen=True)
class CascadeInventoryEntry:
    category: str
    store: str
    member_id: str
    lineage_id: str
    lifecycle_state: str
    authority_state: str | None = None
    relative_path: str | None = None
    content_hash: str | None = None
    physically_removable: bool = False
    file_present: bool | None = None
    shared_container: bool = False


@dataclass(frozen=True)
class CascadeInventory:
    operation_id: str
    lineage_id: str
    retirement_mode: RetirementMode
    workspace_id: str
    organization_id: str
    actor_id: str
    created_at: str
    entries: tuple[CascadeInventoryEntry, ...]
    category_counts: dict[str, int]
    authority_states: tuple[str, ...]
    inventory_hash: str
    schema_version: str = INVENTORY_SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "operation_id": self.operation_id,
            "lineage_id": self.lineage_id,
            "retirement_mode": self.retirement_mode,
            "workspace_id": self.workspace_id,
            "organization_id": self.organization_id,
            "actor_id": self.actor_id,
            "created_at": self.created_at,
            "entries": [asdict(entry) for entry in self.entries],
            "category_counts": dict(sorted(self.category_counts.items())),
            "authority_states": list(self.authority_states),
            "inventory_hash": self.inventory_hash,
        }


class InventoryCollector(Protocol):
    def collect(self, lineage_id: str) -> tuple[CascadeInventoryEntry, ...]: ...


class EosLineageInventoryCollector:
    """Collect the EOS-owned portion of one source build lineage."""

    def __init__(
        self,
        *,
        store: KnowledgeExtractionStore,
        workspace_root: str | Path,
        organization_id: str,
        workspace_id: str,
    ) -> None:
        self.store = store
        self.workspace_root = Path(workspace_root).resolve()
        self.organization_id = organization_id
        self.workspace_id = workspace_id

    def collect(self, lineage_id: str) -> tuple[CascadeInventoryEntry, ...]:
        connection = self.store._conn
        source = connection.execute(
            hosted.sql_query("""SELECT * FROM knowledge_builder_sources
            WHERE COALESCE(lineage_id, source_id) = ?
              AND organization_id = ? AND workspace_id = ?""", 'SELECT * FROM knowledge_builder_sources\n            WHERE COALESCE(lineage_id, source_id) = %s\n              AND organization_id = %s AND workspace_id = %s'),
            (lineage_id, self.organization_id, self.workspace_id),
        ).fetchone()
        entries: list[CascadeInventoryEntry] = []
        if source is not None:
            lifecycle = connection.execute(
                hosted.sql_query("""SELECT next_state FROM source_lifecycle_events
                WHERE source_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1""", 'SELECT next_state FROM source_lifecycle_events\n                WHERE source_id = %s ORDER BY created_at DESC, event_sequence DESC LIMIT 1'),
                (lineage_id,),
            ).fetchone()
            source_state = (
                "tombstoned"
                if source["retirement_state"] == "tombstoned"
                or lifecycle is not None
                and lifecycle["next_state"]
                in {"metadata_tombstoned", "artifacts_tombstoned", "completed"}
                else "active"
            )
            entries.append(
                CascadeInventoryEntry(
                    category="source",
                    store="eos",
                    member_id=str(source["source_id"]),
                    lineage_id=lineage_id,
                    lifecycle_state=source_state,
                    content_hash=str(source["content_hash"]),
                )
            )
            source_path = str(source["source_ref"] or "").strip()
            if source_path.startswith(KNOWLEDGE_UPLOAD_REF_PREFIX):
                source_path = decode_governed_upload_ref(
                    source_path,
                    expected_workspace_id=self.workspace_id,
                )
            if source_path:
                entries.append(
                    CascadeInventoryEntry(
                        category="source_file",
                        store="workspace",
                        member_id=str(source["source_id"]),
                        lineage_id=lineage_id,
                        lifecycle_state=source_state,
                        relative_path=source_path,
                        content_hash=str(source["source_hash"]),
                        physically_removable=True,
                    )
                )
            metadata = json.loads(str(source["source_metadata_json"] or "{}"))
            canonical_ir_ref = str(metadata.get("canonical_ir_ref") or "").strip()
            canonical_ir_path = Path(canonical_ir_ref)
            if (
                canonical_ir_ref
                and canonical_ir_path.parts[:2]
                == ("artifacts", "canonical-source-ir")
                and canonical_ir_path.name == "canonical_source_ir.v1.json"
            ):
                artifact_hash = canonical_ir_path.parts[2]
                ir_hash = (
                    f"sha256:{artifact_hash}"
                    if re.fullmatch(r"[0-9a-f]{64}", artifact_hash)
                    else None
                )
                entries.append(
                    CascadeInventoryEntry(
                        category="canonical_source_ir",
                        store="workspace",
                        member_id=str(source["source_id"]),
                        lineage_id=lineage_id,
                        lifecycle_state=source_state,
                        relative_path=canonical_ir_ref,
                        content_hash=ir_hash,
                        physically_removable=True,
                    )
                )

        discoveries = connection.execute(
            hosted.sql_query("""SELECT discovery_run_id, content_hash, retirement_state
            FROM knowledge_builder_discovery_runs
            WHERE COALESCE(lineage_id, source_id) = ?
              AND organization_id = ? AND workspace_id = ?""", 'SELECT discovery_run_id, content_hash, retirement_state\n            FROM knowledge_builder_discovery_runs\n            WHERE COALESCE(lineage_id, source_id) = %s\n              AND organization_id = %s AND workspace_id = %s'),
            (lineage_id, self.organization_id, self.workspace_id),
        ).fetchall()
        entries.extend(
            CascadeInventoryEntry(
                category="discovery",
                store="eos",
                member_id=str(row["discovery_run_id"]),
                lineage_id=lineage_id,
                lifecycle_state=str(row["retirement_state"]),
                content_hash=str(row["content_hash"]),
            )
            for row in discoveries
        )

        representations = self.store.list_active_source_representations(
            lineage_id,
            organization_id=self.organization_id,
            workspace_id=self.workspace_id,
        )
        for row in representations:
            representation_id = str(row["representation_id"])
            entries.append(
                CascadeInventoryEntry(
                    category="representation",
                    store="eos",
                    member_id=representation_id,
                    lineage_id=lineage_id,
                    lifecycle_state="active",
                    content_hash=str(row["content_hash"]),
                )
            )
            entries.append(
                CascadeInventoryEntry(
                    category="normalized_file",
                    store="workspace",
                    member_id=representation_id,
                    lineage_id=lineage_id,
                    lifecycle_state="active",
                    relative_path=str(row["artifact_path"]),
                    content_hash=str(row["artifact_content_hash"]),
                    physically_removable=True,
                )
            )

        extractions = connection.execute(
            hosted.sql_query("""SELECT extraction_run_id, content_hash, retirement_state
            FROM knowledge_builder_extraction_runs
            WHERE COALESCE(lineage_id, source_id) = ?
              AND organization_id = ? AND workspace_id = ?""", 'SELECT extraction_run_id, content_hash, retirement_state\n            FROM knowledge_builder_extraction_runs\n            WHERE COALESCE(lineage_id, source_id) = %s\n              AND organization_id = %s AND workspace_id = %s'),
            (lineage_id, self.organization_id, self.workspace_id),
        ).fetchall()
        entries.extend(
            CascadeInventoryEntry(
                category="extraction",
                store="eos",
                member_id=str(row["extraction_run_id"]),
                lineage_id=lineage_id,
                lifecycle_state=str(row["retirement_state"]),
                content_hash=str(row["content_hash"]),
            )
            for row in extractions
        )

        origins = connection.execute(
            hosted.sql_query("""SELECT origin_id, origin_kind, authority_state, content_hash
            FROM contextgraph_artifact_origins
            WHERE COALESCE(lineage_id, source_id) = ?
              AND organization_id = ? AND workspace_id = ?""", 'SELECT origin_id, origin_kind, authority_state, content_hash\n            FROM contextgraph_artifact_origins\n            WHERE COALESCE(lineage_id, source_id) = %s\n              AND organization_id = %s AND workspace_id = %s'),
            (lineage_id, self.organization_id, self.workspace_id),
        ).fetchall()
        entries.extend(
            CascadeInventoryEntry(
                category="origin",
                store="eos",
                member_id=str(row["origin_id"]),
                lineage_id=lineage_id,
                lifecycle_state=(
                    "tombstoned"
                    if row["origin_kind"] == "source_tombstoned"
                    else "active"
                ),
                authority_state=str(row["authority_state"]),
                content_hash=str(row["content_hash"]),
            )
            for row in origins
        )

        release_table = connection.execute(
            hosted.sql_query("SELECT 1 FROM sqlite_master WHERE type='table' AND name='tender_graph_releases'", "SELECT 1 FROM pg_catalog.pg_tables WHERE schemaname=current_schema() AND tablename='tender_graph_releases'")
        ).fetchone()
        if release_table is not None:
            releases = connection.execute(
                hosted.sql_query("""SELECT release.graph_version, release.content_hash,
                          release.retirement_state
                FROM tender_graph_releases AS release
                JOIN sensitive_term_candidate_graphs AS candidate
                  ON candidate.candidate_graph_id = release.candidate_graph_id
                JOIN knowledge_builder_extraction_runs AS extraction
                  ON extraction.extraction_run_id = candidate.extraction_run_id
                 AND extraction.organization_id = candidate.organization_id
                 AND extraction.workspace_id IS candidate.workspace_id
                WHERE COALESCE(extraction.lineage_id, extraction.source_id) = ?
                  AND release.organization_id = ?
                  AND release.workspace_id = ?""", 'SELECT release.graph_version, release.content_hash,\n                          release.retirement_state\n                FROM tender_graph_releases AS release\n                JOIN sensitive_term_candidate_graphs AS candidate\n                  ON candidate.candidate_graph_id = release.candidate_graph_id\n                JOIN knowledge_builder_extraction_runs AS extraction\n                  ON extraction.extraction_run_id = candidate.extraction_run_id\n                 AND extraction.organization_id = candidate.organization_id\n                 AND extraction.workspace_id IS NOT DISTINCT FROM candidate.workspace_id\n                WHERE COALESCE(extraction.lineage_id, extraction.source_id) = %s\n                  AND release.organization_id = %s\n                  AND release.workspace_id = %s'),
                (lineage_id, self.organization_id, self.workspace_id),
            ).fetchall()
            entries.extend(
                CascadeInventoryEntry(
                    category="graph_release",
                    store="eos",
                    member_id=str(row["graph_version"]),
                    lineage_id=lineage_id,
                    lifecycle_state=str(row["retirement_state"]),
                    authority_state="authoritative",
                    content_hash=str(row["content_hash"]),
                )
                for row in releases
            )
        return tuple(entries)


class GraphArtifactInventoryCollector:
    def __init__(self, *, repository: ContextGraphArtifactRepository) -> None:
        self.repository = repository

    def collect(self, lineage_id: str) -> tuple[CascadeInventoryEntry, ...]:
        artifacts = [
            artifact
            for artifact in self.repository.list_lineage()
            if (artifact.lineage_id or artifact.lineage_root_artifact_id) == lineage_id
        ]
        if not artifacts:
            return ()
        entries: list[CascadeInventoryEntry] = []
        for artifact in artifacts:
            entries.append(
                CascadeInventoryEntry(
                    category="graph",
                    store=f"graph_manifest:{self.repository.domain_id}",
                    member_id=artifact.artifact_id,
                    lineage_id=lineage_id,
                    lifecycle_state=artifact.lifecycle_state,
                    authority_state=artifact.authority_state,
                    content_hash=artifact.content_hash,
                )
            )
            entries.append(
                CascadeInventoryEntry(
                    category="graph_file",
                    store="workspace",
                    member_id=artifact.artifact_id,
                    lineage_id=lineage_id,
                    lifecycle_state=artifact.lifecycle_state,
                    authority_state=artifact.authority_state,
                    relative_path=(
                        self.repository.root.relative_to(
                            self.repository.workspace_root
                        )
                        / artifact.relative_artifact_path
                    ).as_posix(),
                    content_hash=artifact.content_hash,
                    physically_removable=True,
                )
            )
        entries.append(
            CascadeInventoryEntry(
                category="graph_manifest",
                store=f"graph_manifest:{self.repository.domain_id}",
                member_id=self.repository.domain_id,
                lineage_id=lineage_id,
                lifecycle_state="active",
                relative_path=self.repository.manifest_path.relative_to(
                    self.repository.workspace_root
                ).as_posix(),
                physically_removable=False,
                shared_container=True,
            )
        )
        return tuple(entries)


class GraphSchemaInventoryCollector:
    def __init__(self, *, store: GraphSchemaDiscoveryStore) -> None:
        self.store = store

    def collect(self, lineage_id: str) -> tuple[CascadeInventoryEntry, ...]:
        definitions = (
            ("graph_schema_discovery_runs", "run_id", "schema_run", "created_at"),
            ("semantic_discovery_decisions", "decision_id", "schema_decision", "decision_hash"),
            ("graph_schema_discovery_briefs", "brief_id", "schema_artifact", "brief_hash"),
            ("llm_proposal_evidence", "evidence_id", "schema_evidence", "proposal_hash"),
        )
        entries: list[CascadeInventoryEntry] = []
        with self.store._connect() as connection:
            for table, id_column, category, hash_column in definitions:
                rows = connection.execute(
                    hosted.sql_query(f"SELECT {id_column}, {hash_column}, retirement_state FROM {table} "
                    "WHERE lineage_id = ? AND workspace_id = ?", f'SELECT {id_column}, {hash_column}, retirement_state FROM {table} WHERE lineage_id = %s AND workspace_id = %s'),
                    (lineage_id, self.store.workspace_id),
                ).fetchall()
                entries.extend(
                    CascadeInventoryEntry(
                        category=category,
                        store="graph_schema_discovery",
                        member_id=str(row[id_column]),
                        lineage_id=lineage_id,
                        lifecycle_state=str(row["retirement_state"]),
                        content_hash=(str(row[hash_column]) if row[hash_column] else None),
                    )
                    for row in rows
                )
        if entries:
            entries.append(
                CascadeInventoryEntry(
                    category="schema_store",
                    store="graph_schema_discovery",
                    member_id=self.store.workspace_id,
                    lineage_id=lineage_id,
                    lifecycle_state="active",
                    relative_path=self.store.db_path.relative_to(
                        self.store.workspace_root
                    ).as_posix(),
                    physically_removable=False,
                    shared_container=True,
                )
            )
        return tuple(entries)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _inventory_hash(payload: dict[str, object]) -> str:
    unhashed = {key: value for key, value in payload.items() if key != "inventory_hash"}
    return f"sha256:{hashlib.sha256(_canonical_json(unhashed).encode('utf-8')).hexdigest()}"


class LineageCascadeInventoryService:
    def __init__(
        self,
        *,
        workspace_root: str | Path,
        workspace_id: str,
        organization_id: str,
        actor_id: str,
        collectors: Iterable[InventoryCollector],
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_id = self._require_id(workspace_id, "workspace_id")
        self.organization_id = self._require_id(organization_id, "organization_id")
        self.actor_id = self._require_id(actor_id, "actor_id")
        self.collectors = tuple(collectors)

    @staticmethod
    def _require_id(value: str, label: str) -> str:
        if not isinstance(value, str) or not _SAFE_ID.fullmatch(value.strip()):
            raise ValueError(f"{label} must be an ASCII-stable identifier")
        return value.strip()

    def _path(self, operation_id: str) -> Path:
        safe_operation_id = self._require_id(operation_id, "operation_id")
        return (
            self.workspace_root
            / "runs"
            / "contextgraph-retirement"
            / safe_operation_id
            / "cascade-inventory.v1.json"
        )

    def _normalize_entry(
        self, entry: CascadeInventoryEntry, *, lineage_id: str
    ) -> CascadeInventoryEntry:
        if entry.lineage_id != lineage_id:
            raise ValueError("unsupported cross-lineage inventory entry")
        for label, value in (
            ("category", entry.category),
            ("store", entry.store),
            ("member_id", entry.member_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"inventory {label} is required")
        if entry.relative_path is None:
            return replace(entry, file_present=None)
        relative = Path(entry.relative_path)
        if (
            relative.is_absolute()
            or not entry.relative_path.strip()
            or ".." in relative.parts
            or "\\" in entry.relative_path
        ):
            raise ValueError("inventory file must use a governed relative path")
        resolved = (self.workspace_root / relative).resolve()
        if resolved != self.workspace_root and self.workspace_root not in resolved.parents:
            raise ValueError("inventory file must use a governed relative path")
        return replace(entry, relative_path=relative.as_posix(), file_present=resolved.is_file())

    def _collect(self, lineage_id: str) -> tuple[CascadeInventoryEntry, ...]:
        entries = [
            self._normalize_entry(entry, lineage_id=lineage_id)
            for collector in self.collectors
            for entry in collector.collect(lineage_id)
        ]
        identities: set[tuple[str, str, str, str | None]] = set()
        for entry in entries:
            identity = (
                entry.category,
                entry.store,
                entry.member_id,
                entry.relative_path,
            )
            if identity in identities:
                raise ValueError(f"duplicate cascade inventory member: {entry.member_id}")
            identities.add(identity)
        return tuple(
            sorted(
                entries,
                key=lambda item: (
                    item.category,
                    item.store,
                    item.member_id,
                    item.relative_path or "",
                ),
            )
        )

    def _build(
        self,
        *,
        operation_id: str,
        lineage_id: str,
        retirement_mode: RetirementMode,
        created_at: str,
        entries: tuple[CascadeInventoryEntry, ...] | None = None,
    ) -> CascadeInventory:
        operation_id = self._require_id(operation_id, "operation_id")
        lineage_id = self._require_id(lineage_id, "lineage_id")
        if retirement_mode not in {"ordinary", "governed"}:
            raise ValueError("retirement_mode must be ordinary or governed")
        entries = entries if entries is not None else self._collect(lineage_id)
        counts = dict(Counter(entry.category for entry in entries))
        authority_states = tuple(
            sorted({entry.authority_state for entry in entries if entry.authority_state})
        )
        provisional = CascadeInventory(
            operation_id=operation_id,
            lineage_id=lineage_id,
            retirement_mode=retirement_mode,
            workspace_id=self.workspace_id,
            organization_id=self.organization_id,
            actor_id=self.actor_id,
            created_at=created_at,
            entries=entries,
            category_counts=counts,
            authority_states=authority_states,
            inventory_hash="",
        )
        return replace(provisional, inventory_hash=_inventory_hash(provisional.as_dict()))

    def preview(
        self,
        *,
        operation_id: str,
        lineage_id: str,
        retirement_mode: RetirementMode,
        escalate_authoritative: bool = False,
    ) -> CascadeInventory:
        path = self._path(operation_id)
        if path.exists():
            existing = self.load(operation_id)
            expected_mode = retirement_mode
            if (
                escalate_authoritative
                and retirement_mode == "ordinary"
                and existing.retirement_mode == "governed"
            ):
                expected_mode = "governed"
            expected_scope = (
                self._require_id(lineage_id, "lineage_id"),
                expected_mode,
                self.workspace_id,
                self.organization_id,
                self.actor_id,
            )
            actual_scope = (
                existing.lineage_id,
                existing.retirement_mode,
                existing.workspace_id,
                existing.organization_id,
                existing.actor_id,
            )
            if actual_scope != expected_scope:
                raise InventoryScopeMismatch("operation is bound to a different inventory scope")
            return existing
        entries = self._collect(lineage_id)
        effective_mode = retirement_mode
        if escalate_authoritative and any(
            entry.authority_state == "authoritative" for entry in entries
        ):
            effective_mode = "governed"
        inventory = self._build(
            operation_id=operation_id,
            lineage_id=lineage_id,
            retirement_mode=effective_mode,
            created_at=datetime.now(timezone.utc).isoformat(),
            entries=entries,
        )
        self._write_once(path, inventory.as_dict())
        return inventory

    def _write_once(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(_canonical_json(payload))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load(self, operation_id: str) -> CascadeInventory:
        payload = json.loads(self._path(operation_id).read_text(encoding="utf-8"))
        if payload.get("schema_version") != INVENTORY_SCHEMA_VERSION:
            raise InventoryError("unsupported cascade inventory schema")
        if payload.get("inventory_hash") != _inventory_hash(payload):
            raise InventoryError("cascade inventory hash mismatch")
        entries = tuple(CascadeInventoryEntry(**entry) for entry in payload["entries"])
        return CascadeInventory(
            operation_id=payload["operation_id"],
            lineage_id=payload["lineage_id"],
            retirement_mode=payload["retirement_mode"],
            workspace_id=payload["workspace_id"],
            organization_id=payload["organization_id"],
            actor_id=payload["actor_id"],
            created_at=payload["created_at"],
            entries=entries,
            category_counts=dict(payload["category_counts"]),
            authority_states=tuple(payload["authority_states"]),
            inventory_hash=payload["inventory_hash"],
            schema_version=payload["schema_version"],
        )

    def assert_current(self, inventory: CascadeInventory) -> None:
        current = self._build(
            operation_id=inventory.operation_id,
            lineage_id=inventory.lineage_id,
            retirement_mode=inventory.retirement_mode,
            created_at=inventory.created_at,
        )
        if current.inventory_hash != inventory.inventory_hash:
            raise InventoryChanged("cascade inventory changed after preview")

    @staticmethod
    def _scope_member(entry: CascadeInventoryEntry) -> tuple[object, ...]:
        return (
            entry.category,
            entry.store,
            entry.member_id,
            entry.lineage_id,
            entry.authority_state,
            entry.relative_path,
            entry.content_hash,
            entry.physically_removable,
            entry.shared_container,
        )

    def assert_scope_membership(self, inventory: CascadeInventory) -> None:
        current = self._collect(inventory.lineage_id)
        if tuple(map(self._scope_member, current)) != tuple(
            map(self._scope_member, inventory.entries)
        ):
            raise InventoryChanged("cascade inventory membership changed after preview")

    def active_residuals(
        self, inventory: CascadeInventory
    ) -> tuple[CascadeInventoryEntry, ...]:
        current = self._collect(inventory.lineage_id)
        return tuple(
            entry
            for entry in current
            if not entry.shared_container
            and entry.lifecycle_state == "active"
            and not (entry.relative_path is not None and entry.file_present is False)
        )


__all__ = [
    "CascadeInventory",
    "CascadeInventoryEntry",
    "EosLineageInventoryCollector",
    "GraphArtifactInventoryCollector",
    "GraphSchemaInventoryCollector",
    "INVENTORY_SCHEMA_VERSION",
    "InventoryChanged",
    "InventoryError",
    "InventoryScopeMismatch",
    "LineageCascadeInventoryService",
]
