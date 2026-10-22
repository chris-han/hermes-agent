"""Lineage projection service (Plan B M1, extended M5+).

Pivots on ``source_id`` as the canonical lineage identifier (Plan B
M5+ lineage-id semantic). One source upload begins one lineage; the
lineage then accumulates downstream phases (extraction run, candidate
graph, evaluation graph, authoritative graph) which all reference the
same ``source_id`` via ``contextgraph_artifact_origins.source_id`` and
``knowledge_builder_extraction_runs.source_id``.

States (M5+ — see 2026-08-26-workspace-lineage-surface plan D7):

- ``in_progress`` — the source has been uploaded but no graph artifact
  exists yet. Normal workflow state; NOT an orphan.
- ``live`` — graph artifacts exist and the upstream chain is intact
  (source → extraction → graph).
- ``orphan`` — a graph artifact exists but one or more upstream phases
  is missing. Subtypes:
    - ``missing_source`` — graph artifact has no source binding, OR
      source row exists but has been tombstoned, OR graph was imported
      without a source binding (``imported_without_source`` is a
      label-only variant of ``missing_source``).
    - ``missing_extraction`` — graph artifact exists, source exists,
      but no ``knowledge_builder_extraction_runs`` row references this
      source. Implicitly also missing_schema because schema lives on
      extraction runs.

State and subtype are DERIVED from actual lineage-edge field presence,
not from the denormalized ``origin_kind`` column. ``origin_kind`` is
kept in the lineage-edge table for backward compatibility but no
longer governs orphan classification (Plan B D2 re-stated for M5+).

Read functions are deterministic and never mutate state. The
retire_lineage function (M3) routes lineage tombstoning through Plan
A's source-retirement orchestrator (Plan A M2 + M3 invariants).
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import sqlite3
import threading
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


from storage.hosted_postgres import (
    enabled,
    connect_authority,
    mapped_store,
    require_tables,
    sql_query,
)
from storage.ontology_build_store import ontology_build_store_path
from semantier.semantica_context_graph.artifact_repository import (
    validate_contextgraph_connection,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class LineageProjectionError(RuntimeError):
    """Base class for lineage projection errors."""


class UnknownLineage(LineageProjectionError):
    """lineage_id (== source_id) is not present in the workspace."""


# Lineage states and orphan subtypes — ASCII machine identifiers (Law 4).
LINEAGE_STATE_IN_PROGRESS = "in_progress"
LINEAGE_STATE_LIVE = "live"
LINEAGE_STATE_ORPHAN = "orphan"

ORPHAN_SUBTYPE_MISSING_SOURCE = "missing_source"
ORPHAN_SUBTYPE_MISSING_EXTRACTION = "missing_extraction"

VALID_FILTERS = {"all", "in_progress", "live", "orphan"}


def _source_display_title(source_row: Any) -> str:
    """Resolve a governed human-readable source title with stable fallback."""
    try:
        metadata = json.loads(source_row["source_metadata_json"] or "{}")
    except (json.JSONDecodeError, TypeError):
        metadata = {}
    if isinstance(metadata, dict):
        for key in (
            "display_name",
            "original_filename",
            "file_name",
            "filename",
            "name",
        ):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    source_ref = str(source_row["source_ref"] or "").strip()
    if source_ref:
        return source_ref.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return str(source_row["source_id"])


def _humanize_machine_kind(value: str) -> str:
    return value.replace("-", " ").replace("_", " ").strip().capitalize()


class LineageProjectionService:
    """Plan B M1/M5+: workspace-relative lineage projection for a single domain.

    The service is per-domain (organization_id, workspace_id, domain_id)
    because ``graph_artifacts`` lives in the manifest.sqlite3 file under
    ``<workspace_root>/.semantier/context-graphs/<domain_id>/``. A
    workspace-level lens (Plan B D5) that needs to aggregate across
    multiple domains should iterate the service per domain.
    """

    _lock_key: tuple[str, str, str] | None = None

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        organization_id: str,
        workspace_id: str,
        domain_id: str,
        eos_path: str | Path | None = None,
        connection=None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.organization_id = organization_id
        self.workspace_id = workspace_id
        self.domain_id = domain_id
        self.manifest_path = (
            self.workspace_root
            / ".semantier"
            / "context-graphs"
            / domain_id
            / "manifest.sqlite3"
        )
        self.eos_path = (
            Path(eos_path).resolve()
            if eos_path is not None
            else self.workspace_root / "eos.db"
        )
        self._conn_lock = threading.RLock()
        self._manifest_store_path = ontology_build_store_path(self.workspace_root)
        self._borrowed_connection = connection

    @property
    def exists(self) -> bool:
        if enabled():
            with self._read_connection():
                return True
        return self.manifest_path.exists()

    def _query(self, sqlite, postgres):
        if not enabled():
            return sqlite
        from psycopg import sql

        manifest = sql.Identifier(
            mapped_store("standalone", self._manifest_store_path)["schema"]
        )
        eos = sql.Identifier(mapped_store("eos", self.eos_path)["schema"])
        prefix = sql.SQL("""WITH graph_artifacts AS (
            SELECT * FROM {manifest}.graph_artifacts
            WHERE organization_id={organization} AND workspace_id={workspace} AND domain_id={domain}),
        active_source_representations AS (
            SELECT representation.* FROM {eos}.source_representation_edges representation
            JOIN {eos}.source_representation_lifecycle_events lifecycle ON lifecycle.event_id=(
                SELECT latest.event_id FROM {eos}.source_representation_lifecycle_events latest
                WHERE latest.representation_id=representation.representation_id
                ORDER BY latest.created_at DESC,latest.event_sequence DESC LIMIT 1)
            WHERE lifecycle.next_state='active') """).format(
            manifest=manifest,
            eos=eos,
            organization=sql.Literal(self.organization_id),
            workspace=sql.Literal(self.workspace_id),
            domain=sql.Literal(self.domain_id),
        )
        return prefix + sql.SQL(postgres).format(
            eos=eos, eos_name=sql.Literal(mapped_store("eos", self.eos_path)["schema"])
        )

    @contextmanager
    def _read_connection(self):
        if self._borrowed_connection is not None:
            validate_contextgraph_connection(
                self._borrowed_connection, self._manifest_store_path, self.eos_path
            )
            yield self._borrowed_connection
        else:
            with self._connect() as conn:
                yield conn

    def _connect(self):
        if enabled():
            conn = connect_authority("standalone", self._manifest_store_path)
            try:
                conn.execute(
                    "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
                )
                require_tables(
                    conn,
                    "graph_artifacts",
                    "runtime_feedback",
                    "artifact_lifecycle_events_local",
                )
                return conn
            except BaseException:
                conn.close()
                raise
        manifest_target = (
            str(self.manifest_path) if self.manifest_path.exists() else ":memory:"
        )
        connection = sqlite3.connect(manifest_target, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("ATTACH DATABASE ? AS eos", (str(self.eos_path),))
        graph_artifacts_exists = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' "
            "AND name = 'graph_artifacts'"
        ).fetchone()[0]
        if not graph_artifacts_exists:
            connection.execute(
                """
                CREATE TEMP TABLE graph_artifacts (
                  artifact_id TEXT, organization_id TEXT, workspace_id TEXT,
                  domain_id TEXT, graph_id TEXT, graph_version TEXT,
                  artifact_kind TEXT, authority_state TEXT, run_mode TEXT,
                  parent_artifact_id TEXT, lineage_root_artifact_id TEXT,
                  relative_artifact_path TEXT, lifecycle_state TEXT,
                  created_at TEXT
                )
                """
            )
        representation_tables = connection.execute(
            "SELECT COUNT(*) FROM eos.sqlite_master WHERE type = 'table' "
            "AND name IN ('source_representation_edges', "
            "'source_representation_lifecycle_events')"
        ).fetchone()[0]
        if representation_tables == 2:
            connection.execute(
                """
                CREATE TEMP VIEW active_source_representations AS
                SELECT representation.*
                FROM eos.source_representation_edges representation
                JOIN eos.source_representation_lifecycle_events lifecycle
                  ON lifecycle.event_id = (
                    SELECT latest.event_id
                    FROM eos.source_representation_lifecycle_events latest
                    WHERE latest.representation_id = representation.representation_id
                    ORDER BY latest.created_at DESC, latest.rowid DESC LIMIT 1
                  )
                WHERE lifecycle.next_state = 'active'
                """
            )
        else:
            # Compatibility for pre-migration/read-only fixtures. Production
            # stores create the durable tables through KnowledgeExtractionStore.
            connection.execute(
                """
                CREATE TEMP TABLE active_source_representations (
                  representation_id TEXT, organization_id TEXT, workspace_id TEXT,
                  source_id TEXT, relation_kind TEXT, artifact_ref TEXT,
                  artifact_path TEXT, version INTEGER, created_at TEXT
                )
                """
            )
        return connection

    # ---- list_lineages --------------------------------------------------

    def list_lineages(
        self,
        *,
        filter: str = "all",  # noqa: A002 - Plan B domain keyword
    ) -> list[dict[str, Any]]:
        """Return every lineage in this domain, optionally filtered.

        Filter values (M5+):

        - ``all`` (default): every lineage in the domain (in_progress,
          live, orphan).
        - ``in_progress``: sources that have been uploaded to this
          workspace but have no graph artifact anywhere. The source has
          no domain binding until its first graph artifact is written;
          the projection surfaces in-progress lineages at the workspace
          level regardless of which domain the graph eventually lands in.
        - ``live``: lineages with a full upstream chain (source →
          extraction → graph artifact).
        - ``orphan``: lineages whose graph artifact is detached from
          one or more upstream phases.

        Each returned row carries:

        - ``lineage_id`` (== ``source_id``): the canonical lineage
          identifier (M5+ change).
        - ``source_id``: the source document id.
        - ``first_artifact_id``: the id of the first graph artifact in
          this domain for this lineage. ``None`` for in_progress.
        - ``lineage_state``: ``in_progress`` | ``live`` | ``orphan``.
        - ``orphan_subtype``: ``missing_source`` |
          ``missing_extraction`` | ``None``.
        - ``has_authoritative_artifact``: any authoritative descendant
          exists in this lineage.
        - ``descendant_count``: graph artifact count in this domain.
        - ``version_count``: distinct graph_version in this domain.
        - ``display_label``: ``source_id`` (or
          ``imported:<lineage_id>`` for imported-without-source).
        - ``source_governance_state``: the source row's governance_state
          if available, else ``None``.
        - ``lineage_root_artifact_id``: back-compat alias for
          ``lineage_id``. Old API consumers that read this field still
          see the canonical lineage identifier (= source_id).

        State derivation (D7):
          - If no graph_artifacts row references the source in this
            domain, lineage_state = in_progress.
          - Else if contextgraph_artifact_origins has a row with this
            source_id AND a knowledge_builder_extraction_runs row
            exists for the source, lineage_state = live.
          - Else lineage_state = orphan.
          - Orphan subtype:
              - missing_source if contextgraph_artifact_origins has no
                row referencing the source_id.
              - missing_extraction if no extraction_run exists.
        """
        if filter not in VALID_FILTERS:
            raise LineageProjectionError(f"unknown filter: {filter}")

        with self._conn_lock:
            with self._read_connection() as conn:
                # Pull sources that have at least one graph artifact in
                # this domain. Joined against the lineage-edge table and
                # the extraction-run table to compute state and
                # aggregate counts in one pass.
                attached_rows = conn.execute(
                    self._query(
                        """
                    SELECT
                      s.source_id AS lineage_id,
                      s.source_id AS source_id,
                      s.source_ref AS source_ref,
                      s.governance_state AS source_governance_state,
                      s.created_at AS source_created_at,
                      s.source_metadata_json AS source_metadata_json,
                      MIN(ga.artifact_id) AS first_artifact_id,
                      MAX(ga.created_at) AS latest_artifact_created_at,
                      COUNT(*) AS descendant_count,
                      (
                        SELECT COUNT(*)
                        FROM active_source_representations representation
                        WHERE representation.source_id = s.source_id
                          AND representation.organization_id = s.organization_id
                          AND representation.workspace_id = s.workspace_id
                      ) AS active_representation_count,
                      COUNT(DISTINCT ga.graph_version) AS version_count,
                      MAX(CASE WHEN ga.authority_state = 'authoritative' THEN 1 ELSE 0 END) AS has_authoritative_artifact,
                      (
                        SELECT COUNT(*) FROM eos.contextgraph_artifact_origins o
                        WHERE o.source_id = s.source_id
                          AND o.organization_id = s.organization_id
                          AND o.workspace_id = s.workspace_id
                      ) AS lineage_edge_count,
                      (
                        SELECT COUNT(*) FROM eos.knowledge_builder_extraction_runs er
                        WHERE er.source_id = s.source_id
                          AND er.organization_id = s.organization_id
                      ) AS extraction_run_count,
                      (
                        SELECT MAX(er.created_at) FROM eos.knowledge_builder_extraction_runs er
                        WHERE er.source_id = s.source_id
                          AND er.organization_id = s.organization_id
                      ) AS latest_extraction_created_at,
                      (
                        SELECT MAX(representation.created_at)
                        FROM active_source_representations representation
                        WHERE representation.source_id = s.source_id
                          AND representation.organization_id = s.organization_id
                          AND representation.workspace_id = s.workspace_id
                      ) AS latest_representation_created_at,
                      (
                        SELECT MAX(o.created_at) FROM eos.contextgraph_artifact_origins o
                        WHERE o.source_id = s.source_id
                          AND o.organization_id = s.organization_id
                          AND o.workspace_id = s.workspace_id
                      ) AS latest_lineage_edge_created_at
                    FROM eos.knowledge_builder_sources s
                    INNER JOIN graph_artifacts ga
                      ON ga.organization_id = s.organization_id
                     AND ga.workspace_id = s.workspace_id
                     AND ga.domain_id = ?
                     AND ga.lifecycle_state != 'deleted'
                     AND (
                       ga.lineage_root_artifact_id = s.source_id
                       OR EXISTS (
                         SELECT 1 FROM eos.contextgraph_artifact_origins o
                         WHERE o.artifact_id = ga.artifact_id
                           AND o.organization_id = ga.organization_id
                           AND o.workspace_id = ga.workspace_id
                           AND o.source_id = s.source_id
                       )
                     )
                    WHERE s.organization_id = ?
                      AND (s.workspace_id = ? OR s.workspace_id IS NULL)
                      AND NOT EXISTS (
                        SELECT 1 FROM eos.source_retirement_operations retirement
                        WHERE retirement.source_id = s.source_id
                          AND retirement.state = 'completed'
                      )
                    GROUP BY s.source_id
                    """,
                        "\n                    SELECT\n                      s.source_id AS lineage_id,\n                      s.source_id AS source_id,\n                      s.source_ref AS source_ref,\n                      s.governance_state AS source_governance_state,\n                      s.created_at AS source_created_at,\n                      s.source_metadata_json AS source_metadata_json,\n                      MIN(ga.artifact_id) AS first_artifact_id,\n                      MAX(ga.created_at) AS latest_artifact_created_at,\n                      COUNT(*) AS descendant_count,\n                      (\n                        SELECT COUNT(*)\n                        FROM active_source_representations representation\n                        WHERE representation.source_id = s.source_id\n                          AND representation.organization_id = s.organization_id\n                          AND representation.workspace_id = s.workspace_id\n                      ) AS active_representation_count,\n                      COUNT(DISTINCT ga.graph_version) AS version_count,\n                      MAX(CASE WHEN ga.authority_state = 'authoritative' THEN 1 ELSE 0 END) AS has_authoritative_artifact,\n                      (\n                        SELECT COUNT(*) FROM {eos}.contextgraph_artifact_origins o\n                        WHERE o.source_id = s.source_id\n                          AND o.organization_id = s.organization_id\n                          AND o.workspace_id = s.workspace_id\n                      ) AS lineage_edge_count,\n                      (\n                        SELECT COUNT(*) FROM {eos}.knowledge_builder_extraction_runs er\n                        WHERE er.source_id = s.source_id\n                          AND er.organization_id = s.organization_id\n                      ) AS extraction_run_count,\n                      (\n                        SELECT MAX(er.created_at) FROM {eos}.knowledge_builder_extraction_runs er\n                        WHERE er.source_id = s.source_id\n                          AND er.organization_id = s.organization_id\n                      ) AS latest_extraction_created_at,\n                      (\n                        SELECT MAX(representation.created_at)\n                        FROM active_source_representations representation\n                        WHERE representation.source_id = s.source_id\n                          AND representation.organization_id = s.organization_id\n                          AND representation.workspace_id = s.workspace_id\n                      ) AS latest_representation_created_at,\n                      (\n                        SELECT MAX(o.created_at) FROM {eos}.contextgraph_artifact_origins o\n                        WHERE o.source_id = s.source_id\n                          AND o.organization_id = s.organization_id\n                          AND o.workspace_id = s.workspace_id\n                      ) AS latest_lineage_edge_created_at\n                    FROM {eos}.knowledge_builder_sources s\n                    INNER JOIN graph_artifacts ga\n                      ON ga.organization_id = s.organization_id\n                     AND ga.workspace_id = s.workspace_id\n                     AND ga.domain_id = %s\n                     AND ga.lifecycle_state != 'deleted'\n                     AND (\n                       ga.lineage_root_artifact_id = s.source_id\n                       OR EXISTS (\n                         SELECT 1 FROM {eos}.contextgraph_artifact_origins o\n                         WHERE o.artifact_id = ga.artifact_id\n                           AND o.organization_id = ga.organization_id\n                           AND o.workspace_id = ga.workspace_id\n                           AND o.source_id = s.source_id\n                       )\n                     )\n                    WHERE s.organization_id = %s\n                      AND (s.workspace_id = %s OR s.workspace_id IS NULL)\n                      AND NOT EXISTS (\n                        SELECT 1 FROM {eos}.source_retirement_operations retirement\n                        WHERE retirement.source_id = s.source_id\n                          AND retirement.state = 'completed'\n                      )\n                    GROUP BY s.source_id\n                    ",
                    ),
                    (self.domain_id, self.organization_id, self.workspace_id),
                ).fetchall()

                # Pull sources that have NO graph artifact in this
                # workspace (any domain). These are in_progress
                # lineages — workspace-level (not strictly per-domain).
                in_progress_rows = conn.execute(
                    self._query(
                        """
                    SELECT
                      s.source_id AS lineage_id,
                      s.source_id AS source_id,
                      s.source_ref AS source_ref,
                      s.governance_state AS source_governance_state,
                      s.created_at AS source_created_at,
                      s.source_metadata_json AS source_metadata_json,
                      NULL AS first_artifact_id,
                      NULL AS latest_artifact_created_at,
                      0 AS descendant_count,
                      (
                        SELECT COUNT(*)
                        FROM active_source_representations representation
                        WHERE representation.source_id = s.source_id
                          AND representation.organization_id = s.organization_id
                          AND representation.workspace_id = s.workspace_id
                      ) AS active_representation_count,
                      0 AS version_count,
                      0 AS has_authoritative_artifact,
                      0 AS lineage_edge_count,
                      (
                        SELECT COUNT(*) FROM eos.knowledge_builder_extraction_runs er
                        WHERE er.source_id = s.source_id
                          AND er.organization_id = s.organization_id
                      ) AS extraction_run_count,
                      (
                        SELECT MAX(er.created_at) FROM eos.knowledge_builder_extraction_runs er
                        WHERE er.source_id = s.source_id
                          AND er.organization_id = s.organization_id
                      ) AS latest_extraction_created_at,
                      (
                        SELECT MAX(representation.created_at)
                        FROM active_source_representations representation
                        WHERE representation.source_id = s.source_id
                          AND representation.organization_id = s.organization_id
                          AND representation.workspace_id = s.workspace_id
                      ) AS latest_representation_created_at,
                      NULL AS latest_lineage_edge_created_at
                    FROM eos.knowledge_builder_sources s
                    WHERE s.organization_id = ?
                      AND (s.workspace_id = ? OR s.workspace_id IS NULL)
                      AND NOT EXISTS (
                        SELECT 1 FROM eos.source_retirement_operations retirement
                        WHERE retirement.source_id = s.source_id
                          AND retirement.state = 'completed'
                      )
                      AND NOT EXISTS (
                        SELECT 1 FROM graph_artifacts ga
                        WHERE (
                          ga.lineage_root_artifact_id = s.source_id
                          OR EXISTS (
                            SELECT 1 FROM eos.contextgraph_artifact_origins o
                            WHERE o.artifact_id = ga.artifact_id
                              AND o.organization_id = ga.organization_id
                              AND o.workspace_id = ga.workspace_id
                              AND o.source_id = s.source_id
                          )
                        )
                      )
                    """,
                        "\n                    SELECT\n                      s.source_id AS lineage_id,\n                      s.source_id AS source_id,\n                      s.source_ref AS source_ref,\n                      s.governance_state AS source_governance_state,\n                      s.created_at AS source_created_at,\n                      s.source_metadata_json AS source_metadata_json,\n                      NULL AS first_artifact_id,\n                      NULL AS latest_artifact_created_at,\n                      0 AS descendant_count,\n                      (\n                        SELECT COUNT(*)\n                        FROM active_source_representations representation\n                        WHERE representation.source_id = s.source_id\n                          AND representation.organization_id = s.organization_id\n                          AND representation.workspace_id = s.workspace_id\n                      ) AS active_representation_count,\n                      0 AS version_count,\n                      0 AS has_authoritative_artifact,\n                      0 AS lineage_edge_count,\n                      (\n                        SELECT COUNT(*) FROM {eos}.knowledge_builder_extraction_runs er\n                        WHERE er.source_id = s.source_id\n                          AND er.organization_id = s.organization_id\n                      ) AS extraction_run_count,\n                      (\n                        SELECT MAX(er.created_at) FROM {eos}.knowledge_builder_extraction_runs er\n                        WHERE er.source_id = s.source_id\n                          AND er.organization_id = s.organization_id\n                      ) AS latest_extraction_created_at,\n                      (\n                        SELECT MAX(representation.created_at)\n                        FROM active_source_representations representation\n                        WHERE representation.source_id = s.source_id\n                          AND representation.organization_id = s.organization_id\n                          AND representation.workspace_id = s.workspace_id\n                      ) AS latest_representation_created_at,\n                      NULL AS latest_lineage_edge_created_at\n                    FROM {eos}.knowledge_builder_sources s\n                    WHERE s.organization_id = %s\n                      AND (s.workspace_id = %s OR s.workspace_id IS NULL)\n                      AND NOT EXISTS (\n                        SELECT 1 FROM {eos}.source_retirement_operations retirement\n                        WHERE retirement.source_id = s.source_id\n                          AND retirement.state = 'completed'\n                      )\n                      AND NOT EXISTS (\n                        SELECT 1 FROM graph_artifacts ga\n                        WHERE (\n                          ga.lineage_root_artifact_id = s.source_id\n                          OR EXISTS (\n                            SELECT 1 FROM {eos}.contextgraph_artifact_origins o\n                            WHERE o.artifact_id = ga.artifact_id\n                              AND o.organization_id = ga.organization_id\n                              AND o.workspace_id = ga.workspace_id\n                              AND o.source_id = s.source_id\n                          )\n                        )\n                      )\n                    ",
                    ),
                    (self.organization_id, self.workspace_id),
                ).fetchall()

        summaries = [
            {
                **self._serialise_lineage_summary(dict(row)),
                "domain_id": self.domain_id,
            }
            for row in (*attached_rows, *in_progress_rows)
        ]

        if filter == "all":
            return summaries
        return [s for s in summaries if s["lineage_state"] == filter]

    @staticmethod
    def _derive_state_and_subtype(
        *,
        descendant_count: int,
        extraction_run_count: int,
        lineage_edge_count: int,
    ) -> tuple[str, str | None]:
        """Plan B D7 (M5+): derive lineage_state and orphan_subtype.

        Pure function — no DB access. The projection SQL computes the
        counts; this function classifies.

        Order matters:
          1. descendant_count == 0 → in_progress (always).
          2. lineage_edge_count > 0 AND extraction_run_count > 0 →
             live (full chain present).
          3. lineage_edge_count == 0 → orphan, missing_source.
          4. extraction_run_count == 0 → orphan, missing_extraction.
          5. fallback → orphan, missing_extraction (defensive).
        """
        if descendant_count == 0:
            return LINEAGE_STATE_IN_PROGRESS, None
        if lineage_edge_count > 0 and extraction_run_count > 0:
            return LINEAGE_STATE_LIVE, None
        if lineage_edge_count == 0:
            return LINEAGE_STATE_ORPHAN, ORPHAN_SUBTYPE_MISSING_SOURCE
        if extraction_run_count == 0:
            return LINEAGE_STATE_ORPHAN, ORPHAN_SUBTYPE_MISSING_EXTRACTION
        return LINEAGE_STATE_ORPHAN, ORPHAN_SUBTYPE_MISSING_EXTRACTION

    @staticmethod
    def _serialise_lineage_summary(row: dict[str, Any]) -> dict[str, Any]:
        semantic_descendant_count = int(row.get("descendant_count") or 0)
        active_representation_count = int(row.get("active_representation_count") or 0)
        extraction_run_count = int(row.get("extraction_run_count") or 0)
        lineage_edge_count = int(row.get("lineage_edge_count") or 0)
        lineage_state, orphan_subtype = (
            LineageProjectionService._derive_state_and_subtype(
                descendant_count=semantic_descendant_count,
                extraction_run_count=extraction_run_count,
                lineage_edge_count=lineage_edge_count,
            )
        )
        first_artifact_id = row.get("first_artifact_id")
        timestamp_candidates = [
            str(timestamp)
            for timestamp in (
                row.get("source_created_at"),
                row.get("latest_artifact_created_at"),
                row.get("latest_extraction_created_at"),
                row.get("latest_representation_created_at"),
                row.get("latest_lineage_edge_created_at"),
            )
            if timestamp
        ]
        source_id = row.get("source_id") or row.get("lineage_id")
        display_label = (
            _source_display_title(row) if source_id else f"orphan:{first_artifact_id}"
        )
        return {
            "lineage_id": row["lineage_id"],
            # Back-compat alias — keep returning this field so old
            # callers still work; it now equals source_id (M5+ change).
            "lineage_root_artifact_id": row["lineage_id"],
            "domain_id": row.get("domain_id") or "",
            "source_id": source_id,
            "first_artifact_id": first_artifact_id,
            "display_label": display_label,
            "lineage_state": lineage_state,
            "orphan_subtype": orphan_subtype,
            # Legacy origin_kind kept for callers that still read it.
            "origin_kind": _legacy_origin_kind(lineage_state, orphan_subtype),
            "source_governance_state": row.get("source_governance_state"),
            "has_authoritative_artifact": bool(
                row.get("has_authoritative_artifact") or 0
            ),
            "descendant_count": semantic_descendant_count + active_representation_count,
            "visible_descendant_count": semantic_descendant_count
            + active_representation_count,
            "classifier_descendant_count": semantic_descendant_count,
            "semantic_descendant_count": semantic_descendant_count,
            "active_representation_count": active_representation_count,
            "version_count": int(row.get("version_count") or 0),
            "extraction_run_count": extraction_run_count,
            "lineage_edge_count": lineage_edge_count,
            "created_at": row.get("source_created_at"),
            "updated_at": max(timestamp_candidates) if timestamp_candidates else None,
        }

    # ---- get_lineage_graph ---------------------------------------------

    def get_lineage_graph(
        self,
        *,
        lineage_id: str,
        graph_version: str | None = None,
    ) -> dict[str, Any]:
        """Return the descendants of a lineage as nodes + edges.

        Pivots on ``lineage_id`` (== source_id). For in_progress
        lineages, returns a single-node DAG with the source row only.

        The graph is rendered on the Sigma canvas through the existing
        ``computeGraphTopology(..., 'hierarchical', ...)`` layout.
        Vertices are graph_artifacts (or a single source node for
        in_progress); edges are lineage edges from
        ``contextgraph_artifact_origins``.
        """
        with self._conn_lock:
            with self._read_connection() as conn:
                source_row = conn.execute(
                    self._query(
                        """
                    SELECT source_id, source_ref, governance_state, source_metadata_json,
                           created_at
                    FROM eos.knowledge_builder_sources
                    WHERE source_id = ?
                      AND organization_id = ?
                      AND (workspace_id = ? OR workspace_id IS NULL)
                    LIMIT 1
                    """,
                        "\n                    SELECT source_id, source_ref, governance_state, source_metadata_json,\n                           created_at\n                    FROM {eos}.knowledge_builder_sources\n                    WHERE source_id = %s\n                      AND organization_id = %s\n                      AND (workspace_id = %s OR workspace_id IS NULL)\n                    LIMIT 1\n                    ",
                    ),
                    (lineage_id, self.organization_id, self.workspace_id),
                ).fetchone()
                if source_row is None:
                    raise UnknownLineage(f"lineage not found: {lineage_id}")

                version_clause = ""
                version_params: tuple[Any, ...] = ()
                if graph_version is not None:
                    version_clause = sql_query(
                        "AND ga.graph_version = ?", "AND ga.graph_version = %s"
                    )
                    version_params = (graph_version,)

                artifact_rows = conn.execute(
                    self._query(
                        f"""
                    SELECT DISTINCT
                      ga.artifact_id, ga.organization_id, ga.workspace_id,
                      ga.domain_id, ga.graph_id, ga.graph_version,
                      ga.artifact_kind, ga.authority_state, ga.run_mode,
                      ga.parent_artifact_id, ga.lineage_root_artifact_id,
                      ga.relative_artifact_path, ga.lifecycle_state,
                      ga.created_at
                    FROM graph_artifacts ga
                    WHERE ga.organization_id = ?
                      AND ga.workspace_id = ?
                      AND ga.domain_id = ?
                      AND ga.lifecycle_state != 'deleted'
                      AND (
                        ga.lineage_root_artifact_id = ?
                        OR EXISTS (
                          SELECT 1 FROM eos.contextgraph_artifact_origins o
                          WHERE o.artifact_id = ga.artifact_id
                            AND o.organization_id = ga.organization_id
                            AND o.workspace_id = ga.workspace_id
                            AND o.source_id = ?
                        )
                      )
                      {version_clause}
                    """,
                        f"\n                    SELECT DISTINCT\n                      ga.artifact_id, ga.organization_id, ga.workspace_id,\n                      ga.domain_id, ga.graph_id, ga.graph_version,\n                      ga.artifact_kind, ga.authority_state, ga.run_mode,\n                      ga.parent_artifact_id, ga.lineage_root_artifact_id,\n                      ga.relative_artifact_path, ga.lifecycle_state,\n                      ga.created_at\n                    FROM graph_artifacts ga\n                    WHERE ga.organization_id = %s\n                      AND ga.workspace_id = %s\n                      AND ga.domain_id = %s\n                      AND ga.lifecycle_state != 'deleted'\n                      AND (\n                        ga.lineage_root_artifact_id = %s\n                        OR EXISTS (\n                          SELECT 1 FROM {{eos}}.contextgraph_artifact_origins o\n                          WHERE o.artifact_id = ga.artifact_id\n                            AND o.organization_id = ga.organization_id\n                            AND o.workspace_id = ga.workspace_id\n                            AND o.source_id = %s\n                        )\n                      )\n                      {version_clause}\n                    ",
                    ),
                    (
                        self.organization_id,
                        self.workspace_id,
                        self.domain_id,
                        lineage_id,
                        lineage_id,
                        *version_params,
                    ),
                ).fetchall()

                lineage_edge_rows = conn.execute(
                    self._query(
                        """
                    SELECT
                      o.origin_id, o.source_id, o.extraction_run_id,
                      o.candidate_graph_id, o.artifact_id,
                      o.source_identity_ref, o.relation_kind, o.origin_kind,
                      o.authority_state
                    FROM eos.contextgraph_artifact_origins o
                    WHERE o.organization_id = ?
                      AND o.workspace_id = ?
                      AND o.source_id = ?
                    """,
                        "\n                    SELECT\n                      o.origin_id, o.source_id, o.extraction_run_id,\n                      o.candidate_graph_id, o.artifact_id,\n                      o.source_identity_ref, o.relation_kind, o.origin_kind,\n                      o.authority_state\n                    FROM {eos}.contextgraph_artifact_origins o\n                    WHERE o.organization_id = %s\n                      AND o.workspace_id = %s\n                      AND o.source_id = %s\n                    ",
                    ),
                    (self.organization_id, self.workspace_id, lineage_id),
                ).fetchall()
                representation_rows = conn.execute(
                    self._query(
                        """
                    SELECT representation.*
                    FROM active_source_representations representation
                    WHERE representation.organization_id = ?
                      AND representation.workspace_id = ?
                      AND representation.source_id = ?
                    ORDER BY representation.version
                    """,
                        "\n                    SELECT representation.*\n                    FROM active_source_representations representation\n                    WHERE representation.organization_id = %s\n                      AND representation.workspace_id = %s\n                      AND representation.source_id = %s\n                    ORDER BY representation.version\n                    ",
                    ),
                    (self.organization_id, self.workspace_id, lineage_id),
                ).fetchall()
                discovery_rows = conn.execute(
                    self._query(
                        """
                    SELECT discovery_run_id, source_id, run_status,
                           governance_state, started_at, completed_at
                    FROM eos.knowledge_builder_discovery_runs
                    WHERE organization_id = ?
                      AND (workspace_id = ? OR workspace_id IS NULL)
                      AND (lineage_id = ? OR source_id = ?)
                      AND retirement_state = 'active'
                    ORDER BY started_at
                    """,
                        "\n                    SELECT discovery_run_id, source_id, run_status,\n                           governance_state, started_at, completed_at\n                    FROM {eos}.knowledge_builder_discovery_runs\n                    WHERE organization_id = %s\n                      AND (workspace_id = %s OR workspace_id IS NULL)\n                      AND (lineage_id = %s OR source_id = %s)\n                      AND retirement_state = 'active'\n                    ORDER BY started_at\n                    ",
                    ),
                    (
                        self.organization_id,
                        self.workspace_id,
                        lineage_id,
                        lineage_id,
                    ),
                ).fetchall()
                extraction_rows = conn.execute(
                    self._query(
                        """
                    SELECT extraction_run_id, discovery_run_id, source_id,
                           run_status, started_at, completed_at
                    FROM eos.knowledge_builder_extraction_runs
                    WHERE organization_id = ?
                      AND (workspace_id = ? OR workspace_id IS NULL)
                      AND (lineage_id = ? OR source_id = ?)
                      AND retirement_state = 'active'
                    ORDER BY started_at
                    """,
                        "\n                    SELECT extraction_run_id, discovery_run_id, source_id,\n                           run_status, started_at, completed_at\n                    FROM {eos}.knowledge_builder_extraction_runs\n                    WHERE organization_id = %s\n                      AND (workspace_id = %s OR workspace_id IS NULL)\n                      AND (lineage_id = %s OR source_id = %s)\n                      AND retirement_state = 'active'\n                    ORDER BY started_at\n                    ",
                    ),
                    (
                        self.organization_id,
                        self.workspace_id,
                        lineage_id,
                        lineage_id,
                    ),
                ).fetchall()
                table_names = {
                    row["name"]
                    for row in conn.execute(
                        self._query(
                            "SELECT name FROM eos.sqlite_master WHERE type = 'table' AND name IN ('sensitive_term_candidate_graphs', 'tender_graph_releases')",
                            "SELECT tablename AS name FROM pg_catalog.pg_tables WHERE schemaname={eos_name} AND tablename IN ('sensitive_term_candidate_graphs','tender_graph_releases')",
                        )
                    ).fetchall()
                }
                candidate_rows = []
                if "sensitive_term_candidate_graphs" in table_names:
                    candidate_rows = conn.execute(
                        self._query(
                            """
                        SELECT candidate_graph_id, extraction_run_id,
                               allowed_base_graph_version
                        FROM eos.sensitive_term_candidate_graphs
                        WHERE organization_id = ?
                          AND (workspace_id = ? OR workspace_id IS NULL)
                          AND extraction_run_id IN (
                            SELECT extraction_run_id
                            FROM eos.knowledge_builder_extraction_runs
                            WHERE organization_id = ?
                              AND (workspace_id = ? OR workspace_id IS NULL)
                              AND (lineage_id = ? OR source_id = ?)
                              AND retirement_state = 'active'
                          )
                        ORDER BY candidate_graph_id
                        """,
                            "\n                        SELECT candidate_graph_id, extraction_run_id,\n                               allowed_base_graph_version\n                        FROM {eos}.sensitive_term_candidate_graphs\n                        WHERE organization_id = %s\n                          AND (workspace_id = %s OR workspace_id IS NULL)\n                          AND extraction_run_id IN (\n                            SELECT extraction_run_id\n                            FROM {eos}.knowledge_builder_extraction_runs\n                            WHERE organization_id = %s\n                              AND (workspace_id = %s OR workspace_id IS NULL)\n                              AND (lineage_id = %s OR source_id = %s)\n                              AND retirement_state = 'active'\n                          )\n                        ORDER BY candidate_graph_id\n                        ",
                        ),
                        (
                            self.organization_id,
                            self.workspace_id,
                            self.organization_id,
                            self.workspace_id,
                            lineage_id,
                            lineage_id,
                        ),
                    ).fetchall()
                release_rows = []
                if candidate_rows and "tender_graph_releases" in table_names:
                    candidate_ids = [
                        row["candidate_graph_id"] for row in candidate_rows
                    ]
                    placeholders = ",".join(
                        "%s" if enabled() else "?" for _ in candidate_ids
                    )
                    release_rows = conn.execute(
                        self._query(
                            f"""
                        SELECT graph_version, candidate_graph_id,
                               materialization_status
                        FROM eos.tender_graph_releases
                        WHERE organization_id = ?
                          AND (workspace_id = ? OR workspace_id IS NULL)
                          AND candidate_graph_id IN ({placeholders})
                        ORDER BY graph_version
                        """,
                            f"\n                        SELECT graph_version, candidate_graph_id,\n                               materialization_status\n                        FROM {{eos}}.tender_graph_releases\n                        WHERE organization_id = %s\n                          AND (workspace_id = %s OR workspace_id IS NULL)\n                          AND candidate_graph_id IN ({placeholders})\n                        ORDER BY graph_version\n                        ",
                        ),
                        (
                            self.organization_id,
                            self.workspace_id,
                            *candidate_ids,
                        ),
                    ).fetchall()

        source_title = _source_display_title(source_row)
        discovery_indexes = {
            row["discovery_run_id"]: index
            for index, row in enumerate(discovery_rows, start=1)
        }
        extraction_indexes = {
            row["extraction_run_id"]: index
            for index, row in enumerate(extraction_rows, start=1)
        }
        nodes = [
            {
                "id": source_row["source_id"],
                "display_title": source_title,
                "kind": "source",
                "authority_state": "candidate",
                "graph_version": None,
                "graph_id": None,
                "parent_artifact_id": None,
                "relative_artifact_path": None,
                "lifecycle_state": source_row["governance_state"],
                "created_at": source_row["created_at"],
                "node_type": "source",
            }
        ]
        nodes.extend(
            {
                "id": r["artifact_id"],
                "display_title": " · ".join(
                    filter(
                        None,
                        (
                            _humanize_machine_kind(r["artifact_kind"]),
                            r["graph_version"],
                        ),
                    )
                ),
                "kind": r["artifact_kind"],
                "authority_state": r["authority_state"],
                "graph_version": r["graph_version"],
                "graph_id": r["graph_id"],
                "parent_artifact_id": r["parent_artifact_id"],
                "relative_artifact_path": r["relative_artifact_path"],
                "lifecycle_state": r["lifecycle_state"],
                "created_at": r["created_at"],
                "node_type": "graph_artifact",
            }
            for r in artifact_rows
        )
        nodes.extend(
            {
                "id": row["discovery_run_id"],
                "display_title": (
                    "Discovery"
                    if len(discovery_rows) == 1
                    else f"Discovery {discovery_indexes[row['discovery_run_id']]}"
                )
                + f" · {source_title}",
                "kind": "discovery_run",
                "authority_state": row["governance_state"].lower(),
                "graph_version": None,
                "graph_id": None,
                "parent_artifact_id": row["source_id"],
                "relative_artifact_path": None,
                "lifecycle_state": row["run_status"],
                "created_at": row["started_at"],
                "node_type": "discovery_run",
            }
            for row in discovery_rows
        )
        nodes.extend(
            {
                "id": row["extraction_run_id"],
                "display_title": f"Extraction {extraction_indexes[row['extraction_run_id']]} · {source_title}",
                "kind": "extraction_run",
                "authority_state": "candidate",
                "graph_version": None,
                "graph_id": None,
                "parent_artifact_id": row["discovery_run_id"],
                "relative_artifact_path": None,
                "lifecycle_state": row["run_status"],
                "created_at": row["started_at"],
                "node_type": "extraction_run",
            }
            for row in extraction_rows
        )
        nodes.extend(
            {
                "id": row["candidate_graph_id"],
                "display_title": " · ".join(
                    filter(
                        None,
                        (
                            "Candidate graph",
                            row["allowed_base_graph_version"],
                        ),
                    )
                ),
                "kind": "candidate_graph",
                "authority_state": "candidate",
                "graph_version": row["allowed_base_graph_version"],
                "graph_id": row["candidate_graph_id"],
                "parent_artifact_id": row["extraction_run_id"],
                "relative_artifact_path": None,
                "lifecycle_state": "proposed",
                "created_at": None,
                "node_type": "candidate_graph",
            }
            for row in candidate_rows
        )
        nodes.extend(
            {
                "id": row["graph_version"],
                "display_title": f"Graph release · {row['graph_version']}",
                "kind": "graph_release",
                "authority_state": "authoritative",
                "graph_version": row["graph_version"],
                "graph_id": row["candidate_graph_id"],
                "parent_artifact_id": row["candidate_graph_id"],
                "relative_artifact_path": None,
                "lifecycle_state": row["materialization_status"],
                "created_at": None,
                "node_type": "graph_release",
            }
            for row in release_rows
        )
        nodes.extend(
            {
                "id": representation["representation_id"],
                "display_title": " · ".join(
                    filter(
                        None,
                        (
                            _humanize_machine_kind(representation["relation_kind"]),
                            str(representation["artifact_path"] or "")
                            .replace("\\", "/")
                            .rsplit("/", 1)[-1],
                        ),
                    )
                ),
                "kind": representation["relation_kind"],
                "authority_state": "derived",
                "graph_version": representation["version"],
                "graph_id": None,
                "parent_artifact_id": source_row["source_id"],
                "relative_artifact_path": representation["artifact_path"],
                "artifact_ref": representation["artifact_ref"],
                "lifecycle_state": "active",
                "created_at": representation["created_at"],
                "node_type": "normalized_representation",
            }
            for representation in representation_rows
        )
        edges = [
            {
                "id": f"edge_{r['origin_id']}",
                "source": r["source_id"] or r["candidate_graph_id"] or "",
                "target": r["artifact_id"] or "",
                "relation_kind": r["relation_kind"],
                "origin_kind": r["origin_kind"],
                "authority_state": r["authority_state"],
                "edge_type": "lineage_edge",
            }
            for r in lineage_edge_rows
        ]
        edges.extend(
            {
                "id": f"edge_{representation['representation_id']}",
                "source": lineage_id,
                "target": representation["representation_id"],
                "relation_kind": "normalized_from",
                "origin_kind": "normalized_from",
                "authority_state": "derived",
                "edge_type": "representation_edge",
            }
            for representation in representation_rows
        )
        edges.extend(
            {
                "id": f"edge_discovery_{row['discovery_run_id']}",
                "source": lineage_id,
                "target": row["discovery_run_id"],
                "relation_kind": "discovered_as",
                "origin_kind": "workflow_phase",
                "authority_state": row["governance_state"].lower(),
                "edge_type": "workflow_phase_edge",
            }
            for row in discovery_rows
        )
        edges.extend(
            {
                "id": f"edge_extraction_{row['extraction_run_id']}",
                "source": row["discovery_run_id"],
                "target": row["extraction_run_id"],
                "relation_kind": "extracted_as",
                "origin_kind": "workflow_phase",
                "authority_state": "candidate",
                "edge_type": "workflow_phase_edge",
            }
            for row in extraction_rows
        )
        edges.extend(
            {
                "id": f"edge_candidate_{row['candidate_graph_id']}",
                "source": row["extraction_run_id"],
                "target": row["candidate_graph_id"],
                "relation_kind": "proposed_as",
                "origin_kind": "workflow_phase",
                "authority_state": "candidate",
                "edge_type": "workflow_phase_edge",
            }
            for row in candidate_rows
        )
        edges.extend(
            {
                "id": f"edge_release_{row['graph_version']}",
                "source": row["candidate_graph_id"],
                "target": row["graph_version"],
                "relation_kind": "released_as",
                "origin_kind": "workflow_phase",
                "authority_state": "authoritative",
                "edge_type": "workflow_phase_edge",
            }
            for row in release_rows
        )
        return {
            "lineage_id": lineage_id,
            # Back-compat alias — equals source_id.
            "lineage_root_artifact_id": lineage_id,
            "domain_id": self.domain_id,
            "graph_version": graph_version,
            "nodes": nodes,
            "edges": edges,
            "edge_count": len(edges),
            "node_count": len(nodes),
        }

    # ---- versions -------------------------------------------------------

    def list_lineage_versions(
        self,
        *,
        lineage_id: str,
    ) -> list[dict[str, Any]]:
        """Return distinct ``graph_version`` rows for a lineage.

        For in_progress lineages, returns an empty list (no graph
        artifacts yet).
        """
        with self._conn_lock:
            with self._read_connection() as conn:
                source_exists = conn.execute(
                    self._query(
                        "SELECT 1 FROM eos.knowledge_builder_sources WHERE source_id = ? AND organization_id = ? AND (workspace_id = ? OR workspace_id IS NULL)",
                        "SELECT 1 FROM {eos}.knowledge_builder_sources WHERE source_id = %s AND organization_id = %s AND (workspace_id = %s OR workspace_id IS NULL)",
                    ),
                    (lineage_id, self.organization_id, self.workspace_id),
                ).fetchone()
                if source_exists is None:
                    raise UnknownLineage(f"lineage not found: {lineage_id}")
                rows = conn.execute(
                    self._query(
                        """
                    SELECT
                      graph_version,
                      COUNT(*) AS artifact_count,
                      MAX(CASE WHEN authority_state = 'authoritative' THEN 1 ELSE 0 END) AS has_authoritative,
                      MIN(created_at) AS created_at
                    FROM graph_artifacts
                    WHERE organization_id = ?
                      AND workspace_id = ?
                      AND domain_id = ?
                      AND (
                        lineage_root_artifact_id = ?
                        OR EXISTS (
                          SELECT 1 FROM eos.contextgraph_artifact_origins o
                          WHERE o.artifact_id = graph_artifacts.artifact_id
                            AND o.organization_id = graph_artifacts.organization_id
                            AND o.workspace_id = graph_artifacts.workspace_id
                            AND o.source_id = ?
                        )
                      )
                    GROUP BY graph_version
                    ORDER BY created_at ASC
                    """,
                        "\n                    SELECT\n                      graph_version,\n                      COUNT(*) AS artifact_count,\n                      MAX(CASE WHEN authority_state = 'authoritative' THEN 1 ELSE 0 END) AS has_authoritative,\n                      MIN(created_at) AS created_at\n                    FROM graph_artifacts\n                    WHERE organization_id = %s\n                      AND workspace_id = %s\n                      AND domain_id = %s\n                      AND (\n                        lineage_root_artifact_id = %s\n                        OR EXISTS (\n                          SELECT 1 FROM {eos}.contextgraph_artifact_origins o\n                          WHERE o.artifact_id = graph_artifacts.artifact_id\n                            AND o.organization_id = graph_artifacts.organization_id\n                            AND o.workspace_id = graph_artifacts.workspace_id\n                            AND o.source_id = %s\n                        )\n                      )\n                    GROUP BY graph_version\n                    ORDER BY created_at ASC\n                    ",
                    ),
                    (
                        self.organization_id,
                        self.workspace_id,
                        self.domain_id,
                        lineage_id,
                        lineage_id,
                    ),
                ).fetchall()
                return [
                    {
                        "graph_version": r["graph_version"],
                        "artifact_count": int(r["artifact_count"] or 0),
                        "has_authoritative": bool(r["has_authoritative"] or 0),
                        "created_at": r["created_at"],
                    }
                    for r in rows
                ]


def _legacy_origin_kind(lineage_state: str, orphan_subtype: str | None) -> str:
    """Translate (lineage_state, orphan_subtype) → legacy origin_kind.

    Kept for backward compatibility with the lineage-edge table's
    denormalized ``origin_kind`` column. New code should use
    lineage_state / orphan_subtype directly.
    """
    if lineage_state == LINEAGE_STATE_LIVE:
        return "attached"
    if lineage_state == LINEAGE_STATE_IN_PROGRESS:
        return "unresolved"
    if orphan_subtype == ORPHAN_SUBTYPE_MISSING_SOURCE:
        return "imported_without_source"
    if orphan_subtype == ORPHAN_SUBTYPE_MISSING_EXTRACTION:
        return "source_tombstoned"
    return "unresolved"


def discover_workspace_domains(workspace_root: str | Path) -> list[str]:
    """Return the list of domain ids present under a workspace.

    Workspace lens aggregators (Plan B D5) read this to iterate the
    lineage projection service per domain.
    """
    if enabled():
        with connect_authority(
            "standalone", ontology_build_store_path(Path(workspace_root).resolve())
        ) as conn:
            require_tables(conn, "graph_artifacts")
            rows = conn.execute(
                "SELECT DISTINCT domain_id FROM graph_artifacts ORDER BY domain_id"
            ).fetchall()
        domain_root = Path(workspace_root) / ".semantier" / "context-graphs"
        file_domains = {path.name for path in domain_root.iterdir() if path.is_dir()} if domain_root.exists() else set()
        return sorted({row["domain_id"] for row in rows} | file_domains)
    root = Path(workspace_root) / ".semantier" / "context-graphs"
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


__all__ = [
    "LineageProjectionService",
    "LineageProjectionError",
    "UnknownLineage",
    "LINEAGE_STATE_IN_PROGRESS",
    "LINEAGE_STATE_LIVE",
    "LINEAGE_STATE_ORPHAN",
    "ORPHAN_SUBTYPE_MISSING_SOURCE",
    "ORPHAN_SUBTYPE_MISSING_EXTRACTION",
    "discover_workspace_domains",
]
