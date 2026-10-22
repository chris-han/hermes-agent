"""Business analytics Hermes plugin."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from eos.embedded_policy_runtime import (
    evaluate_embedded_policy,
    load_embedded_policy,
)

logger = logging.getLogger(__name__)

_db_lock = threading.Lock()
_db: Any = None
_db_eos_path: str | None = None
_db_governed_view_scope: tuple[int, str, str, str] | None = None

_REQUIRED_LAKEHOUSE_FILES = (
    "raw_rea_events.parquet",
    "governed_rea_claims",
    "journal_voucher_projections",
    "general_ledger_views",
    "financial_statement_packages",
    "tax_filing_packages",
    "accounting_archive_packages",
    "cube_cost_revenue_yearly.parquet",
    "cube_event_type_yearly.parquet",
)
_ANALYTICS_REA_VIEW = "analytics_rea_claims"
_ANALYTICS_JVP_VIEW = "analytics_journal_voucher_projections"
_ANALYTICS_GL_VIEW = "analytics_general_ledger_views"
_ANALYTICS_FS_VIEW = "analytics_financial_statement_packages"
_ANALYTICS_TAX_VIEW = "analytics_tax_filing_packages"
_ANALYTICS_ARCHIVE_VIEW = "analytics_accounting_archive_packages"
_ANALYTICS_EXPENSE_AUDIT_MATRIX_VIEW = "analytics_expense_audit_matrix"
_ANALYTICS_COST_REVENUE_CUBE_VIEW = "analytics_cost_revenue_yearly"
_ANALYTICS_EVENT_TYPE_CUBE_VIEW = "analytics_event_type_yearly"
_LAKEHOUSE_MANIFEST_FILENAME = "lakehouse_manifest.json"
_LAKEHOUSE_MANIFEST_VERSION = "1.2"
_GOVERNED_SCHEMA_LABELS_PATH = Path(__file__).with_name("governed_schema_labels.json")
_GOVERNED_SHARD_COLUMNS = ("org_id", "period_id")
_PARTITIONED_GOVERNED_DATASETS = {
    "governed_rea_claims",
    "journal_voucher_projections",
    "general_ledger_views",
    "financial_statement_packages",
    "tax_filing_packages",
    "accounting_archive_packages",
}
_EOS_ANALYTICS_SOURCE_TABLES = {
    "rea_claims": ("rea_id", "claim_hash"),
    "journal_voucher_projections": ("jvp_id", "content_hash"),
    "general_ledger_views": ("view_id", "content_hash"),
    "financial_statement_packages": ("package_id", "content_hash"),
    "tax_filing_packages": ("filing_id", "content_hash"),
    "accounting_archive_packages": ("archive_id", "content_hash"),
}
_GOVERNED_QUERY_FORBIDDEN_SQL_PATTERNS = (
    " attach ",
    " copy ",
    " create ",
    " delete ",
    " detach ",
    " drop ",
    " export ",
    " import ",
    " insert ",
    " install ",
    " load ",
    " pragma ",
    " read_csv",
    " read_json",
    " read_parquet",
    " replace ",
    " update ",
    " vacuum ",
    "duckdb_",
    " eos.",
    "sqlite_master",
    "sqlite_",
    "pragma_",
    "information_schema",
)
_GOVERNED_QUERY_SCOPED_VIEWS = (
    "org_rea_claims",
    "org_journal_voucher_projections",
    "org_general_ledger_views",
    "org_general_ledger_balance_lines",
    "org_financial_statement_packages",
    "org_tax_filing_packages",
    "org_accounting_archive_packages",
)
_FULLWIDTH_SQL_PUNCTUATION = {
    "，": ",",
    "；": ";",
    "（": "(",
    "）": ")",
}


def _split_error_code(raw: str) -> tuple[str, str]:
    text = (raw or "").strip()
    if ":" in text:
        head, tail = text.split(":", 1)
        code = head.strip()
        msg = tail.strip()
        if code and code.replace("_", "").isalnum() and code.upper() == code:
            return code, msg
    return "UNCLASSIFIED_ERROR", text


def _error_hint_for_code(code: str) -> str:
    hints = {
        "SQL_REQUIRED": "Provide a non-empty sql string in args.sql.",
        "EOS_DB_MISSING": "Run: semantier bootstrap --replace",
        "LAKEHOUSE_MISSING": "Run: semantier bootstrap --replace",
        "LAKEHOUSE_MANIFEST_MISSING": "Run: semantier bootstrap --replace",
        "LAKEHOUSE_MANIFEST_INVALID": "Regenerate lakehouse artifacts with: semantier bootstrap --replace",
        "LAKEHOUSE_MANIFEST_VERSION_MISMATCH": "Regenerate lakehouse artifacts with: semantier bootstrap --replace",
        "LAKEHOUSE_SCHEMA_MISMATCH": "Regenerate lakehouse artifacts with: semantier bootstrap --replace",
        "LAKEHOUSE_DATASET_MISSING": "Regenerate lakehouse artifacts with: semantier bootstrap --replace",
        "LAKEHOUSE_STALE_EOS": "Regenerate lakehouse artifacts after EOS updates: semantier bootstrap --replace",
        "LAKEHOUSE_STALE_SIM": "Regenerate lakehouse artifacts after simulator updates: semantier bootstrap --replace",
        "REAL_ACTIVE_DATASET_VERSION_REQUIRED": "Bind the REAL organization to an active dataset version before querying.",
        "LEGACY_REAL_CLAIMS_UNVERSIONED": "Run governed REA admission for the active dataset version before querying REAL data.",
        "QUERY_SQL_IDENTIFIER_VIOLATION": (
            "Use ASCII identifiers such as debit_10k, credit_10k, net_10k; "
            "if quoting is required, quote each identifier separately."
        ),
        "QUERY_SQL_SYNTAX_VIOLATION": (
            "Use ASCII SQL punctuation and ASCII identifiers, for example: debit_10k, credit_10k, net_10k."
        ),
        "QUERY_EXECUTION_ERROR": "Check SQL syntax and referenced views/tables.",
    }
    return hints.get(code, "Inspect details and resolve the reported runtime condition.")


def _json_error_response(raw: str, *, details: dict[str, Any] | None = None) -> str:
    code, message = _split_error_code(raw)
    payload: dict[str, Any] = {
        "error": True,
        "error_code": code,
        "message": message,
        "hint": _error_hint_for_code(code),
    }
    if details:
        payload["details"] = details
    return json.dumps(payload, ensure_ascii=False)


def _load_governed_schema_labels() -> dict[str, Any]:
    with _GOVERNED_SCHEMA_LABELS_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _column_metadata_for_result(columns: list[str]) -> dict[str, dict[str, Any]]:
    catalog = _load_governed_schema_labels()
    metadata: dict[str, dict[str, Any]] = {}
    for view in (catalog.get("views") or {}).values():
        if not isinstance(view, dict):
            continue
        for column, column_metadata in (view.get("columns") or {}).items():
            if column in columns and isinstance(column_metadata, dict):
                metadata[column] = dict(column_metadata)
        for dynamic_column in view.get("dynamic_columns") or []:
            if not isinstance(dynamic_column, dict):
                continue
            name_pattern = str(dynamic_column.get("name_pattern") or "")
            if not name_pattern:
                continue
            for column in columns:
                if column in metadata:
                    continue
                match = re.fullmatch(name_pattern, column)
                if not match:
                    continue
                column_metadata = {
                    key: value
                    for key, value in dynamic_column.items()
                    if key not in {"name_pattern", "display_name_zh_template"}
                }
                template = dynamic_column.get("display_name_zh_template")
                if isinstance(template, str):
                    column_metadata["display_name_zh"] = template.format(*match.groups(), year=match.group(1))
                metadata[column] = column_metadata
    return metadata


def _display_columns_for_result(columns: list[str]) -> list[str]:
    metadata = _column_metadata_for_result(columns)
    return [
        str(metadata.get(column, {}).get("display_name_zh") or column)
        for column in columns
    ]


def _contains_fullwidth_sql_punctuation(sql: str) -> bool:
    in_single_quote = False
    index = 0
    while index < len(sql):
        char = sql[index]
        if char == "'":
            if in_single_quote and index + 1 < len(sql) and sql[index + 1] == "'":
                index += 2
                continue
            in_single_quote = not in_single_quote
        elif not in_single_quote and char in _FULLWIDTH_SQL_PUNCTUATION:
            return True
        index += 1
    return False


def _contains_quoted_fullwidth_comma_identifier(sql: str) -> bool:
    in_single_quote = False
    in_double_quote = False
    quoted_identifier: list[str] = []
    index = 0
    while index < len(sql):
        char = sql[index]
        if in_single_quote:
            if char == "'" and index + 1 < len(sql) and sql[index + 1] == "'":
                index += 2
                continue
            if char == "'":
                in_single_quote = False
            index += 1
            continue
        if in_double_quote:
            if char == '"' and index + 1 < len(sql) and sql[index + 1] == '"':
                quoted_identifier.append('"')
                index += 2
                continue
            if char == '"':
                if "，" in "".join(quoted_identifier):
                    return True
                quoted_identifier = []
                in_double_quote = False
            else:
                quoted_identifier.append(char)
            index += 1
            continue
        if char == "'":
            in_single_quote = True
        elif char == '"':
            in_double_quote = True
            quoted_identifier = []
        index += 1
    return False


def _contains_non_ascii_sql_identifier(sql: str) -> bool:
    in_single_quote = False
    index = 0
    while index < len(sql):
        char = sql[index]
        if char == "'":
            if in_single_quote and index + 1 < len(sql) and sql[index + 1] == "'":
                index += 2
                continue
            in_single_quote = not in_single_quote
        elif not in_single_quote and not char.isascii():
            return True
        index += 1
    return False


def _workspace_id_from_hermes_home() -> str | None:
    raw_home = os.environ.get("HERMES_HOME") or os.environ.get("HERMES_SESSION_HERMES_HOME")
    if not raw_home:
        return None
    try:
        hermes_home = Path(raw_home).expanduser().resolve()
    except OSError:
        return None
    workspace_id = hermes_home.parent.name.strip() if hermes_home.name == ".hermes" else hermes_home.name.strip()
    return workspace_id or None


@contextmanager
def _bound_eos_db_path(eos_path: Path):
    previous = os.environ.get("SEMANTIER_EOS_DB_PATH")
    os.environ["SEMANTIER_EOS_DB_PATH"] = str(eos_path.resolve())
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("SEMANTIER_EOS_DB_PATH", None)
        else:
            os.environ["SEMANTIER_EOS_DB_PATH"] = previous


def _active_workspace_org_context() -> dict[str, Any] | None:
    workspace_id = _workspace_id_from_hermes_home()
    if not workspace_id:
        return None
    try:
        from agents import gateway_identity
    except Exception:
        return {
            "authorization_error": "AUTH_CONTEXT_UNAVAILABLE",
            "workspace_id": workspace_id,
        }

    with _bound_eos_db_path(_resolve_eos_path()):
        user = gateway_identity.get_user_record_by_workspace_id(workspace_id)
        if not isinstance(user, dict):
            return {
                "authorization_error": "AUTH_CONTEXT_REQUIRED",
                "workspace_id": workspace_id,
            }
        user_id = str(user.get("user_id") or "").strip()
        if not user_id:
            return {
                "authorization_error": "AUTH_CONTEXT_REQUIRED",
                "workspace_id": workspace_id,
            }
        org_context = gateway_identity.get_user_organization_context(user_id)
        if not isinstance(org_context, dict):
            return {
                "authorization_error": "AUTH_CONTEXT_REQUIRED",
                "workspace_id": workspace_id,
                "user_id": user_id,
            }
        org_context = dict(org_context)
        org_context["workspace_id"] = workspace_id
        org_context["user_id"] = user_id
        try:
            from agents.governed_context import resolve_governed_activation_context

            activation_context = resolve_governed_activation_context(
                user_id=user_id,
                workspace_id=workspace_id,
                persist_snapshot=False,
            )
        except Exception:
            activation_context = None
    if isinstance(activation_context, dict):
        org_context["activation_set_snapshot_id"] = activation_context.get(
            "activation_set_snapshot_id"
        )
        org_context["activation_resolver_policy_version"] = activation_context.get(
            "activation_resolver_policy_version"
        )
        org_context["resolved_activation_set_hash"] = activation_context.get(
            "resolved_activation_set_hash"
        )
        for source in activation_context.get("sources") or []:
            if not isinstance(source, dict):
                continue
            if (
                str(source.get("source_kind") or "") == "dataset"
                and str(source.get("dataset_usage_role") or "") == "primary_analytics"
                and bool(source.get("query_context_included"))
            ):
                org_context["dataset_type"] = source.get("dataset_type")
                org_context["dataset_key"] = source.get("dataset_key")
                org_context["active_dataset_version_id"] = source.get(
                    "dataset_version_id"
                )
                org_context["dataset_asset_version_id"] = source.get(
                    "dataset_asset_version_id"
                )
                org_context["dataset_usage_role"] = source.get("dataset_usage_role")
                org_context["dataset_authority_level"] = source.get(
                    "dataset_authority_level"
                )
                org_context["dataset_readiness_status"] = source.get(
                    "dataset_readiness_status"
                )
                break
    return org_context


def _authorize_active_governed_dataset_access() -> tuple[dict[str, Any] | None, str]:
    org_context = _active_workspace_org_context()
    if org_context is None:
        return None, _json_error_response(
            "AUTH_CONTEXT_REQUIRED: governed query requires a resolved workspace authorization context",
            details={"decision": "DENY"},
        )

    auth_error = str(org_context.get("authorization_error") or "").strip()
    if auth_error:
        return None, _json_error_response(
            f"{auth_error}: governed workspace authorization context could not be resolved",
            details={
                "decision": "DENY",
                "workspace_id": org_context.get("workspace_id"),
            },
        )

    organization_id = str(org_context.get("organization_id") or "").strip()
    dataset_key = str(org_context.get("dataset_key") or "").strip()
    active_dataset_version_id = str(org_context.get("active_dataset_version_id") or "").strip()
    if organization_id and not (dataset_key or active_dataset_version_id):
        active: dict[str, Any] | None = None
        try:
            from storage.knowledge_source_activation_store import KnowledgeSourceActivationStore

            store = KnowledgeSourceActivationStore(db_path=_resolve_eos_path())
            try:
                active = store.get_active_dataset_binding(
                    organization_id=organization_id,
                    workspace_id=str(org_context.get("workspace_id") or "").strip() or None,
                )
            finally:
                store.close()
        except Exception:
            pass
        if active is None:
            try:
                active = _duckdb_active_dataset_binding(
                    eos_path=_resolve_eos_path(),
                    organization_id=organization_id,
                    workspace_id=str(org_context.get("workspace_id") or "").strip() or None,
                )
            except Exception:
                active = None
        if active is not None:
            org_context["dataset_type"] = active.get("dataset_type")
            org_context["dataset_key"] = active.get("dataset_key")
            org_context["active_dataset_version_id"] = active.get("dataset_version_id")
            org_context["dataset_asset_version_id"] = active.get("dataset_asset_version_id")
            org_context["dataset_usage_role"] = active.get("dataset_usage_role")
            org_context["dataset_authority_level"] = active.get("authority_level")
            org_context["dataset_readiness_status"] = active.get("readiness_status")
            dataset_key = str(org_context.get("dataset_key") or "").strip()
            active_dataset_version_id = str(
                org_context.get("active_dataset_version_id") or ""
            ).strip()
    if not organization_id or not (dataset_key or active_dataset_version_id):
        return None, _json_error_response(
            "DENY_DATASET_SCOPE_MISSING: active organization must have a governed dataset binding",
            details={
                "decision": "DENY",
                "active_organization_name": org_context.get("organization_name"),
                "active_dataset_key": dataset_key or None,
                "active_dataset_version_id": active_dataset_version_id or None,
            },
        )

    return org_context, ""


def _resolve_eos_path() -> Path:
    """Return EOS SQLite path, preferring explicit environment overrides."""
    override = os.environ.get("SEMANTIER_EOS_DB_PATH")
    if override:
        return Path(override).expanduser().resolve()

    shared_runtime = os.environ.get("SEMANTIER_LOCAL_STATE_DIR")
    if shared_runtime:
        return (Path(shared_runtime).expanduser() / "eos.db").resolve()

    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        hermes_path = Path(hermes_home).expanduser().resolve()
        workspace_local = hermes_path / "eos.db"
        if workspace_local.exists():
            return workspace_local.resolve()
        repo_shared = (Path(__file__).resolve().parents[3] / ".semantier-home" / "eos.db").resolve()
        if repo_shared.exists():
            return repo_shared

    return (Path(__file__).resolve().parents[3] / ".semantier-home" / "eos.db").resolve()


def _resolve_lakehouse_dir() -> Path:
    override = os.environ.get("SEMANTIER_LAKEHOUSE_DIR")
    if override:
        return Path(override).expanduser().resolve()

    shared_runtime = os.environ.get("SEMANTIER_LOCAL_STATE_DIR")
    if shared_runtime:
        return (Path(shared_runtime).expanduser() / "lakehouse").resolve()

    eos_path = _resolve_eos_path()
    return (eos_path.parent / "lakehouse").resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _analytics_source_counts_match_manifest(
    *,
    eos_path: Path,
    manifest: dict[str, Any],
) -> bool:
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict):
        return False
    governed_claims = datasets.get("governed_rea_claims")
    if not isinstance(governed_claims, dict) or "rows" not in governed_claims:
        return False
    try:
        expected_rea_claims = int(governed_claims.get("rows") or 0)
    except (TypeError, ValueError):
        return False
    try:
        with sqlite3.connect(str(eos_path)) as conn:
            current_rea_claims = int(
                conn.execute("SELECT COUNT(*) FROM rea_claims").fetchone()[0]
            )
    except sqlite3.Error:
        return False
    return current_rea_claims == expected_rea_claims


def _sqlite_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _table_exists_sqlite(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return bool(row and row[0])


def _eos_analytics_source_fingerprints(eos_path: Path) -> dict[str, dict[str, object]]:
    fingerprints: dict[str, dict[str, object]] = {}
    with sqlite3.connect(str(eos_path)) as conn:
        for table_name, (id_column, hash_column) in _EOS_ANALYTICS_SOURCE_TABLES.items():
            if not _table_exists_sqlite(conn, table_name):
                fingerprints[table_name] = {
                    "rows": 0,
                    "content_hash": hashlib.sha256(b"").hexdigest(),
                    "missing": True,
                }
                continue
            rows = conn.execute(
                f"""
                SELECT {id_column}, {hash_column}
                FROM {_sqlite_identifier(table_name)}
                ORDER BY {id_column}
                """
            ).fetchall()
            digest = hashlib.sha256()
            for row_id, row_hash in rows:
                digest.update(str(row_id).encode("utf-8"))
                digest.update(b"\0")
                digest.update(str(row_hash).encode("utf-8"))
                digest.update(b"\n")
            fingerprints[table_name] = {
                "rows": len(rows),
                "content_hash": digest.hexdigest(),
            }
    return fingerprints


def _duckdb_eos_analytics_source_fingerprints(eos_path: Path) -> dict[str, dict[str, object]]:
    import duckdb

    fingerprints: dict[str, dict[str, object]] = {}
    conn = duckdb.connect(":memory:")
    try:
        escaped = str(eos_path).replace("'", "''")
        conn.execute(f"ATTACH '{escaped}' AS eos (TYPE SQLITE)")
        for table_name, (id_column, hash_column) in _EOS_ANALYTICS_SOURCE_TABLES.items():
            if not _table_exists(conn, table_name):
                fingerprints[table_name] = {
                    "rows": 0,
                    "content_hash": hashlib.sha256(b"").hexdigest(),
                    "missing": True,
                }
                continue
            rows = conn.execute(
                f"""
                SELECT {_sqlite_identifier(id_column)}, {_sqlite_identifier(hash_column)}
                FROM eos.{_sqlite_identifier(table_name)}
                ORDER BY {_sqlite_identifier(id_column)}
                """
            ).fetchall()
            digest = hashlib.sha256()
            for row_id, row_hash in rows:
                digest.update(str(row_id).encode("utf-8"))
                digest.update(b"\0")
                digest.update(str(row_hash).encode("utf-8"))
                digest.update(b"\n")
            fingerprints[table_name] = {
                "rows": len(rows),
                "content_hash": digest.hexdigest(),
            }
    finally:
        conn.close()
    return fingerprints


def _duckdb_active_dataset_binding(
    *,
    eos_path: Path,
    organization_id: str,
    workspace_id: str | None,
    dataset_usage_role: str = "primary_analytics",
) -> dict[str, Any] | None:
    import duckdb

    scope_keys = [workspace_id, "organization"] if workspace_id else ["organization"]
    conn = duckdb.connect(":memory:")
    try:
        escaped = str(eos_path).replace("'", "''")
        conn.execute(f"ATTACH '{escaped}' AS eos (TYPE SQLITE)")
        for scope_key in scope_keys:
            if not scope_key:
                continue
            rows = conn.execute(
                """
                SELECT
                  ksa.activation_id,
                  ksa.knowledge_source_id,
                  ksa.dataset_asset_version_id,
                  ksa.organization_id,
                  ksa.workspace_id,
                  ksa.activation_scope,
                  ksa.scope_key,
                  ksa.activation_status,
                  ksa.workspace_override_policy,
                  ksa.user_context_control_level,
                  ksa.activation_event_head_hash,
                  ksa.authority_source_manifest_json,
                  ksa.created_at,
                  ksa.content_hash,
                  ks.source_kind,
                  ks.semantic_tier,
                  ks.authority_role,
                  dav.dataset_type,
                  dav.dataset_key,
                  dav.dataset_version_id,
                  dav.dataset_usage_role,
                  dav.authority_level,
                  dav.readiness_status,
                  dav.schema_contract_hash,
                  dav.lineage_manifest_hash,
                  dav.quality_evidence_hash
                FROM eos.knowledge_source_activation ksa
                JOIN eos.knowledge_source ks
                  ON ks.knowledge_source_id = ksa.knowledge_source_id
                JOIN eos.dataset_asset_version dav
                  ON dav.dataset_asset_version_id = ksa.dataset_asset_version_id
                WHERE ksa.organization_id = ?
                  AND ksa.scope_key = ?
                  AND ksa.activation_status = 'active'
                  AND dav.dataset_usage_role = ?
                ORDER BY ksa.created_at DESC
                """,
                [organization_id, scope_key, dataset_usage_role],
            ).fetchall()
            if len(rows) > 1:
                raise ValueError("ambiguous active dataset activation")
            if rows:
                columns = [desc[0] for desc in conn.description]
                return dict(zip(columns, rows[0]))
    finally:
        conn.close()
    return None


def _analytics_sources_match_manifest(
    *,
    eos_path: Path,
    manifest: dict[str, Any],
) -> bool:
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        return False
    eos_source = sources.get("eos_db")
    if not isinstance(eos_source, dict):
        return False
    from storage import hosted_postgres as hosted
    if hosted.enabled():
        if eos_source.get('source_kind') != 'postgresql':
            return False
        try:
            from eos.analytics_snapshot import read_analytics_snapshot
            return eos_source == read_analytics_snapshot(eos_path).provenance
        except Exception:
            return False
    if eos_source.get('source_kind') == 'postgresql':
        return False
    expected = eos_source.get("analytics_tables")
    if not isinstance(expected, dict):
        return _analytics_source_counts_match_manifest(
            eos_path=eos_path,
            manifest=manifest,
        )

    try:
        current = _eos_analytics_source_fingerprints(eos_path)
    except sqlite3.Error:
        try:
            current = _duckdb_eos_analytics_source_fingerprints(eos_path)
        except Exception:
            return False

    for table_name, expected_payload in expected.items():
        if not isinstance(expected_payload, dict):
            return False
        current_payload = current.get(str(table_name))
        if current_payload is None:
            return False
        if int(expected_payload.get("rows") or 0) != int(current_payload.get("rows") or 0):
            return False
        if str(expected_payload.get("content_hash") or "") != str(current_payload.get("content_hash") or ""):
            return False
    return True


def _validate_lakehouse_dir(lakehouse_dir: Path) -> tuple[bool, str]:
    if not lakehouse_dir.exists():
        return False, f"LAKEHOUSE_MISSING: directory not found at {lakehouse_dir}. run: semantier bootstrap --replace"

    manifest_path = lakehouse_dir / _LAKEHOUSE_MANIFEST_FILENAME
    if not manifest_path.exists():
        return False, (
            f"LAKEHOUSE_MANIFEST_MISSING: {_LAKEHOUSE_MANIFEST_FILENAME} not found at {lakehouse_dir}. "
            "run: semantier bootstrap --replace"
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"LAKEHOUSE_MANIFEST_INVALID: failed to parse {manifest_path}: {exc}"

    if str(manifest.get("manifest_version") or "") != _LAKEHOUSE_MANIFEST_VERSION:
        return False, (
            f"LAKEHOUSE_MANIFEST_VERSION_MISMATCH: expected {_LAKEHOUSE_MANIFEST_VERSION}, "
            f"got {manifest.get('manifest_version')!r}. run: semantier bootstrap --replace"
        )

    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict):
        return False, "LAKEHOUSE_SCHEMA_MISMATCH: manifest datasets section is missing or invalid"

    missing_in_manifest = [name for name in _REQUIRED_LAKEHOUSE_FILES if name not in datasets]
    if missing_in_manifest:
        return False, (
            "LAKEHOUSE_SCHEMA_MISMATCH: required datasets missing in manifest: "
            + ", ".join(sorted(missing_in_manifest))
        )

    missing = [name for name in _REQUIRED_LAKEHOUSE_FILES if not (lakehouse_dir / name).exists()]
    if missing:
        return False, (
            "LAKEHOUSE_DATASET_MISSING: required Parquet missing at "
            f"{lakehouse_dir}: {', '.join(sorted(missing))}. run: semantier bootstrap --replace"
        )

    sources = manifest.get("sources") if isinstance(manifest, dict) else None
    if isinstance(sources, dict):
        eos_src = sources.get("eos_db")
        if isinstance(eos_src, dict):
            manifest_hash = str(eos_src.get("sha256") or "").strip()
            eos_path = _resolve_eos_path().expanduser()
            from storage import hosted_postgres as hosted
            if hosted.enabled() or eos_src.get('source_kind') == 'postgresql':
                if not _analytics_sources_match_manifest(eos_path=eos_path, manifest=manifest):
                    return False, (
                        "LAKEHOUSE_STALE_EOS: PostgreSQL analytics source snapshot does not match current authority. "
                        "run: semantier bootstrap --replace"
                    )
            elif manifest_hash and eos_path.exists():
                current_hash = _sha256_file(eos_path)
                if manifest_hash != current_hash:
                    if not _analytics_sources_match_manifest(
                        eos_path=eos_path,
                        manifest=manifest,
                    ):
                        return False, (
                            "LAKEHOUSE_STALE_EOS: lakehouse manifest eos_db hash does not match current eos.db. "
                            "run: semantier bootstrap --replace"
                        )

        sim_src = sources.get("sim_json")
        if isinstance(sim_src, dict):
            sim_path_raw = str(sim_src.get("path") or "").strip()
            manifest_hash = str(sim_src.get("sha256") or "").strip()
            if sim_path_raw and manifest_hash:
                sim_path = Path(sim_path_raw).expanduser()
                if sim_path.exists() and _sha256_file(sim_path) != manifest_hash:
                    return False, (
                        "LAKEHOUSE_STALE_SIM: lakehouse manifest sim_json hash does not match current sim.json. "
                        "run: semantier bootstrap --replace"
                    )

    return True, ""


def _resolve_default_policy_path() -> Path:
    """Return default Gate-1 policy path for embedded runtime evaluation."""
    return Path(__file__).resolve().parents[3] / "policies" / "rea.rego"


def _table_exists(conn: Any, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_catalog = 'eos'
          AND table_schema = 'main'
          AND table_name = ?
        """,
        [table_name],
    ).fetchone()
    return bool(row and row[0])


def _main_view_exists(conn: Any, view_name: str) -> bool:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_catalog = 'memory'
          AND table_schema = 'main'
          AND table_name = ?
        """,
        [view_name],
    ).fetchone()
    return bool(row and row[0])


def _table_has_column(conn: Any, table_name: str, column_name: str) -> bool:
    try:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_catalog = 'eos'
              AND table_schema = 'main'
              AND table_name = ?
              AND column_name = ?
            """,
            [table_name, column_name],
        ).fetchone()
    except Exception:
        return False
    return bool(row and row[0])


def _main_view_has_column(conn: Any, view_name: str, column_name: str) -> bool:
    try:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_catalog = 'memory'
              AND table_schema = 'main'
              AND table_name = ?
              AND column_name = ?
            """,
            [view_name, column_name],
        ).fetchone()
    except Exception:
        return False
    return bool(row and row[0])


def _create_empty_view(conn: Any, view_name: str) -> None:
    conn.execute(f"DROP VIEW IF EXISTS {view_name}")
    conn.execute(f"CREATE VIEW {view_name} AS SELECT * FROM (SELECT 1) WHERE FALSE")


def _analytics_claim_years(conn: Any) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT SUBSTRING(quarter, 1, 4) AS year
        FROM analytics_rea_claims
        WHERE quarter IS NOT NULL
        ORDER BY year
        """
    ).fetchall()
    years: list[str] = []
    for row in rows:
        year = str(row[0] or "")
        if len(year) == 4 and year.isdigit():
            years.append(year)
    return years


def _create_analytics_views(conn: Any) -> None:
    conn.execute(f"DROP VIEW IF EXISTS {_ANALYTICS_REA_VIEW}")
    conn.execute(
        """
        CREATE VIEW analytics_rea_claims AS
        SELECT
            COALESCE(
                NULLIF(regexp_extract(json_extract_string(claim_json, '$.evidence_ref'), '(E\\d+)', 1), ''),
                NULLIF(regexp_extract(json_extract_string(claim_json, '$.evidence_ref'), '(S\\d+)', 1), ''),
                rea_id
            ) AS event_id,
            workflow_id,
            json_extract_string(claim_json, '$.dataset_version_id') AS dataset_version_id,
            json_extract_string(claim_json, '$.resource_type') AS event_type,
            TRY_CAST(json_extract(claim_json, '$.amount') AS DOUBLE) / 10000.0 AS amount_10k,
            json_extract_string(claim_json, '$.currency') AS currency,
            json_extract_string(claim_json, '$.counterparty') AS counterparty,
            json_extract_string(claim_json, '$.occurred_at') AS occurred_at,
            regexp_extract(json_extract_string(claim_json, '$.evidence_ref'), '(20\\d{2}Q[1-4])', 1) AS quarter,
            json_extract_string(claim_json, '$.evidence_ref') AS evidence_ref,
            COALESCE(TRY_CAST(json_extract(claim_json, '$.projectable') AS BOOLEAN), FALSE) AS projectable,
            COALESCE(TRY_CAST(json_extract(claim_json, '$.blocked') AS BOOLEAN), FALSE) AS blocked,
            json_extract_string(claim_json, '$.block_reason') AS block_reason,
            json_extract_string(claim_json, '$.coa_code') AS coa_code,
            claim_hash,
            created_at,
            claim_json
        FROM eos.rea_claims
        """
    )

    conn.execute(f"DROP VIEW IF EXISTS {_ANALYTICS_JVP_VIEW}")
    if _table_exists(conn, "journal_voucher_projections"):
        conn.execute(
            """
            CREATE VIEW analytics_journal_voucher_projections AS
            SELECT
                jvp_id,
                org_id,
                period_id,
                source_event_id,
                projection_bundle_version,
                coa_version,
                voucher_type,
                lines_json,
                justification_ref,
                projection_status,
                content_hash,
                created_at
            FROM eos.journal_voucher_projections
            """
        )
    else:
        conn.execute(f"CREATE VIEW {_ANALYTICS_JVP_VIEW} AS SELECT * FROM (SELECT 1) WHERE FALSE")

    conn.execute(f"DROP VIEW IF EXISTS {_ANALYTICS_GL_VIEW}")
    if _table_exists(conn, "general_ledger_views"):
        conn.execute(f"CREATE VIEW {_ANALYTICS_GL_VIEW} AS SELECT * FROM eos.general_ledger_views")
    else:
        conn.execute(f"CREATE VIEW {_ANALYTICS_GL_VIEW} AS SELECT * FROM (SELECT 1) WHERE FALSE")

    conn.execute(f"DROP VIEW IF EXISTS {_ANALYTICS_FS_VIEW}")
    if _table_exists(conn, "financial_statement_packages"):
        conn.execute(f"CREATE VIEW {_ANALYTICS_FS_VIEW} AS SELECT * FROM eos.financial_statement_packages")
    else:
        conn.execute(f"CREATE VIEW {_ANALYTICS_FS_VIEW} AS SELECT * FROM (SELECT 1) WHERE FALSE")

    conn.execute(f"DROP VIEW IF EXISTS {_ANALYTICS_TAX_VIEW}")
    if _table_exists(conn, "tax_filing_packages"):
        conn.execute(f"CREATE VIEW {_ANALYTICS_TAX_VIEW} AS SELECT * FROM eos.tax_filing_packages")
    else:
        conn.execute(f"CREATE VIEW {_ANALYTICS_TAX_VIEW} AS SELECT * FROM (SELECT 1) WHERE FALSE")

    conn.execute(f"DROP VIEW IF EXISTS {_ANALYTICS_ARCHIVE_VIEW}")
    if _table_exists(conn, "accounting_archive_packages"):
        conn.execute(f"CREATE VIEW {_ANALYTICS_ARCHIVE_VIEW} AS SELECT * FROM eos.accounting_archive_packages")
    else:
        conn.execute(f"CREATE VIEW {_ANALYTICS_ARCHIVE_VIEW} AS SELECT * FROM (SELECT 1) WHERE FALSE")

    # Expense completeness audit matrix (event_type x dataset-derived year)
    conn.execute(f"DROP VIEW IF EXISTS {_ANALYTICS_EXPENSE_AUDIT_MATRIX_VIEW}")
    year_columns = []
    for year in _analytics_claim_years(conn):
        escaped_year = year.replace("'", "''")
        year_columns.extend(
            [
                f"SUM(CASE WHEN year = '{escaped_year}' THEN 1 ELSE 0 END) AS cnt_{year}",
                (
                    f"ROUND(SUM(CASE WHEN year = '{escaped_year}' THEN amount_10k ELSE 0 END), 2) "
                    f"AS amount_{year}_10k"
                ),
            ]
        )
    dynamic_columns = "".join(f",\n            {column}" for column in year_columns)
    conn.execute(
        f"""
        CREATE VIEW analytics_expense_audit_matrix AS
        SELECT
            event_type{dynamic_columns},
            COUNT(*) AS total_rows
        FROM (
            SELECT
                event_type,
                amount_10k,
                SUBSTRING(quarter, 1, 4) AS year
            FROM analytics_rea_claims
            WHERE quarter IS NOT NULL
        ) t
        GROUP BY event_type
        ORDER BY event_type
        """
    )


def _create_lakehouse_views(conn: Any) -> None:
    lakehouse_dir = _resolve_lakehouse_dir().resolve()

    def _create_view(view_name: str, parquet_name: str) -> None:
        dataset_path = (lakehouse_dir / parquet_name).resolve()
        parquet_path = str(dataset_path).replace("'", "''")
        conn.execute(f"DROP VIEW IF EXISTS {view_name}")
        if parquet_name in _PARTITIONED_GOVERNED_DATASETS:
            conn.execute(
                f"""
                CREATE VIEW {view_name} AS
                SELECT * FROM read_parquet('{parquet_path}/*/*/*.parquet', hive_partitioning = true)
                """
            )
            return
        conn.execute(
            f"""
            CREATE VIEW {view_name} AS
            SELECT * FROM read_parquet('{parquet_path}')
            """
        )

    _create_view("raw_rea_events", "raw_rea_events.parquet")
    _create_view("governed_rea_claims", "governed_rea_claims")
    _create_view("lakehouse_journal_voucher_projections", "journal_voucher_projections")
    _create_view("lakehouse_general_ledger_views", "general_ledger_views")
    _create_view("lakehouse_financial_statement_packages", "financial_statement_packages")
    _create_view("lakehouse_tax_filing_packages", "tax_filing_packages")
    _create_view("lakehouse_accounting_archive_packages", "accounting_archive_packages")
    _create_view(_ANALYTICS_COST_REVENUE_CUBE_VIEW, "cube_cost_revenue_yearly.parquet")
    _create_view(_ANALYTICS_EVENT_TYPE_CUBE_VIEW, "cube_event_type_yearly.parquet")


def _create_governed_org_views(conn: Any, org_context: dict[str, Any]) -> None:
    """Build org-scoped governed views using the canonical governed shard key.

    The governed analytics surface is scoped first by org_id and commonly sliced
    within that shard by period_id.
    """
    organization_id = str(org_context.get("organization_id") or "").strip()
    escaped_org_id = organization_id.replace("'", "''")
    dataset_type = str(org_context.get("dataset_type") or "").strip().upper()
    active_dataset_version_id = str(org_context.get("active_dataset_version_id") or "").strip()
    escaped_dataset_version_id = active_dataset_version_id.replace("'", "''")
    lakehouse_dataset_filter = ""
    eos_dataset_filter = ""
    if dataset_type == "REAL":
        if active_dataset_version_id:
            lakehouse_dataset_filter = f" AND dataset_version_id = '{escaped_dataset_version_id}'"
            eos_dataset_filter = (
                " AND json_extract_string(claim_json, '$.dataset_version_id') = "
                f"'{escaped_dataset_version_id}'"
            )
        else:
            lakehouse_dataset_filter = " AND 1 = 0"
            eos_dataset_filter = " AND 1 = 0"

    conn.execute("DROP VIEW IF EXISTS org_rea_claims")
    if _main_view_exists(conn, "governed_rea_claims"):
        conn.execute(
            f"""
            CREATE VIEW org_rea_claims AS
            SELECT
                rea_id,
                workflow_id,
                org_id,
                dataset_version_id,
                period_id,
                event_type,
                amount_10k,
                currency,
                counterparty,
                occurred_at,
                quarter,
                evidence_ref,
                projectable,
                blocked,
                block_reason,
                coa_code,
                claim_hash,
                created_at,
                claim_json
            FROM governed_rea_claims
            WHERE org_id = '{escaped_org_id}'{lakehouse_dataset_filter}
            """
        )
    else:
        conn.execute(
            f"""
            CREATE VIEW org_rea_claims AS
            SELECT
                rea_id,
                workflow_id,
                json_extract_string(claim_json, '$.org_id') AS org_id,
                json_extract_string(claim_json, '$.dataset_version_id') AS dataset_version_id,
                SUBSTRING(json_extract_string(claim_json, '$.occurred_at'), 1, 7) AS period_id,
                json_extract_string(claim_json, '$.resource_type') AS event_type,
                TRY_CAST(json_extract(claim_json, '$.amount') AS DOUBLE) / 10000.0 AS amount_10k,
                json_extract_string(claim_json, '$.currency') AS currency,
                json_extract_string(claim_json, '$.counterparty') AS counterparty,
                json_extract_string(claim_json, '$.occurred_at') AS occurred_at,
                regexp_extract(json_extract_string(claim_json, '$.evidence_ref'), '(20\\d{{2}}Q[1-4])', 1) AS quarter,
                json_extract_string(claim_json, '$.evidence_ref') AS evidence_ref,
                COALESCE(TRY_CAST(json_extract(claim_json, '$.projectable') AS BOOLEAN), FALSE) AS projectable,
                COALESCE(TRY_CAST(json_extract(claim_json, '$.blocked') AS BOOLEAN), FALSE) AS blocked,
                json_extract_string(claim_json, '$.block_reason') AS block_reason,
                json_extract_string(claim_json, '$.coa_code') AS coa_code,
                claim_hash,
                created_at,
                claim_json
            FROM eos.rea_claims
            WHERE json_extract_string(claim_json, '$.org_id') = '{escaped_org_id}'{eos_dataset_filter}
            """
        )

    scoped_tables = (
        ("org_journal_voucher_projections", "journal_voucher_projections", "lakehouse_journal_voucher_projections"),
        ("org_general_ledger_views", "general_ledger_views", "lakehouse_general_ledger_views"),
        ("org_financial_statement_packages", "financial_statement_packages", "lakehouse_financial_statement_packages"),
        ("org_tax_filing_packages", "tax_filing_packages", "lakehouse_tax_filing_packages"),
        ("org_accounting_archive_packages", "accounting_archive_packages", "lakehouse_accounting_archive_packages"),
    )
    for view_name, table_name, lakehouse_view_name in scoped_tables:
        if _main_view_exists(conn, lakehouse_view_name) and _main_view_has_column(conn, lakehouse_view_name, "org_id"):
            conn.execute(f"DROP VIEW IF EXISTS {view_name}")
            conn.execute(
                f"""
                CREATE VIEW {view_name} AS
                SELECT *
                FROM {lakehouse_view_name}
                WHERE org_id = '{escaped_org_id}'
                """
            )
        elif _table_exists(conn, table_name) and _table_has_column(conn, table_name, "org_id"):
            conn.execute(f"DROP VIEW IF EXISTS {view_name}")
            conn.execute(
                f"""
                CREATE VIEW {view_name} AS
                SELECT *
                FROM eos.{table_name}
                WHERE org_id = '{escaped_org_id}'
                """
            )
        else:
            _create_empty_view(conn, view_name)

    conn.execute("DROP VIEW IF EXISTS org_general_ledger_balance_lines")
    if _main_view_exists(conn, "org_general_ledger_views"):
        conn.execute(
            """
            CREATE VIEW org_general_ledger_balance_lines AS
            SELECT
                view_id,
                org_id,
                period_id,
                source_jvp_ids_json,
                coa_version,
                projection_bundle_version,
                je.key AS account_code,
                split_part(je.key, '.', 1) AS account_prefix,
                TRY_CAST(json_extract(je.value, '$.debit') AS DOUBLE) AS debit,
                TRY_CAST(json_extract(je.value, '$.credit') AS DOUBLE) AS credit,
                TRY_CAST(json_extract(je.value, '$.net') AS DOUBLE) AS net,
                TRY_CAST(json_extract(je.value, '$.debit') AS DOUBLE) / 10000.0 AS debit_10k,
                TRY_CAST(json_extract(je.value, '$.credit') AS DOUBLE) / 10000.0 AS credit_10k,
                TRY_CAST(json_extract(je.value, '$.net') AS DOUBLE) / 10000.0 AS net_10k,
                content_hash,
                created_at,
                balances_json
            FROM org_general_ledger_views,
                 json_each(balances_json) AS je
            """
        )
    else:
        _create_empty_view(conn, "org_general_ledger_balance_lines")


def _ensure_governed_org_views(conn: Any, org_context: dict[str, Any]) -> None:
    global _db_governed_view_scope

    scope = (
        id(conn),
        str(org_context.get("organization_id") or "").strip(),
        str(org_context.get("dataset_type") or "").strip().upper(),
        str(org_context.get("active_dataset_version_id") or "").strip(),
    )
    if _db_governed_view_scope == scope:
        return
    _create_governed_org_views(conn, org_context)
    _db_governed_view_scope = scope


def _real_governed_query_diagnostics(conn: Any, org_context: dict[str, Any]) -> tuple[dict[str, Any], str]:
    dataset_type = str(org_context.get("dataset_type") or "").strip().upper()
    if dataset_type != "REAL":
        return {}, ""

    organization_id = str(org_context.get("organization_id") or "").strip()
    active_dataset_version_id = str(org_context.get("active_dataset_version_id") or "").strip()
    if not active_dataset_version_id:
        return (
            {
                "status": "blocked",
                "blocking_reasons": ["REAL_ACTIVE_DATASET_VERSION_REQUIRED"],
                "active_dataset_version_id": None,
            },
            "REAL_ACTIVE_DATASET_VERSION_REQUIRED: REAL governed query requires an active dataset version",
        )

    escaped_org_id = organization_id.replace("'", "''")
    escaped_dataset_version_id = active_dataset_version_id.replace("'", "''")
    if _main_view_exists(conn, "governed_rea_claims"):
        legacy_count = _scalar_int(
            conn,
            f"""
            SELECT COUNT(*)
            FROM governed_rea_claims
            WHERE org_id = '{escaped_org_id}'
              AND COALESCE(dataset_version_id, '') = ''
            """,
        )
        active_claim_count = _scalar_int(
            conn,
            f"""
            SELECT COUNT(*)
            FROM governed_rea_claims
            WHERE org_id = '{escaped_org_id}'
              AND dataset_version_id = '{escaped_dataset_version_id}'
            """,
        )
    else:
        legacy_count = _scalar_int(
            conn,
            f"""
            SELECT COUNT(*)
            FROM eos.rea_claims
            WHERE json_extract_string(claim_json, '$.org_id') = '{escaped_org_id}'
              AND COALESCE(json_extract_string(claim_json, '$.dataset_version_id'), '') = ''
            """,
        )
        active_claim_count = _scalar_int(
            conn,
            f"""
            SELECT COUNT(*)
            FROM eos.rea_claims
            WHERE json_extract_string(claim_json, '$.org_id') = '{escaped_org_id}'
              AND json_extract_string(claim_json, '$.dataset_version_id') = '{escaped_dataset_version_id}'
            """,
        )

    diagnostics: dict[str, Any] = {
        "status": "ready" if active_claim_count else "no_active_version_claims",
        "active_dataset_version_id": active_dataset_version_id,
        "active_version_claim_count": active_claim_count,
        "unversioned_legacy_claim_count": legacy_count,
        "blocking_reasons": [],
    }
    if legacy_count:
        diagnostics["status"] = "blocked"
        diagnostics["blocking_reasons"] = ["LEGACY_REAL_CLAIMS_UNVERSIONED"]
        return (
            diagnostics,
            "LEGACY_REAL_CLAIMS_UNVERSIONED: REAL org has unversioned governed claims and must be migrated before query",
        )

    if _table_exists(conn, "coa_projection_bundles"):
        active_bundle_count = _scalar_int(
            conn,
            f"""
            SELECT COUNT(*)
            FROM eos.coa_projection_bundles
            WHERE org_id = '{escaped_org_id}'
              AND status = 'ACTIVE'
            """,
        )
        diagnostics["active_coa_projection_bundle_count"] = active_bundle_count
        if active_claim_count and not active_bundle_count:
            diagnostics["blocking_reasons"].append("REAL_COA_REQUIRED")

    if _table_exists(conn, "real_company_admission_exceptions"):
        diagnostics["projection_exception_count"] = _scalar_int(
            conn,
            f"""
            SELECT COUNT(*)
            FROM eos.real_company_admission_exceptions
            WHERE organization_id = '{escaped_org_id}'
              AND dataset_version_id = '{escaped_dataset_version_id}'
            """,
        )

    return diagnostics, ""


def _scalar_int(conn: Any, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _reset_db() -> None:
    global _db, _db_eos_path, _db_governed_view_scope

    with _db_lock:
        if _db is not None:
            try:
                _db.close()
            except Exception:
                pass
        _db = None
        _db_eos_path = None
        _db_governed_view_scope = None


def _load_db() -> Any:
    global _db, _db_eos_path, _db_governed_view_scope

    eos_path = _resolve_eos_path().resolve()
    with _db_lock:
        from storage import hosted_postgres as hosted
        if _db is not None and _db_eos_path == str(eos_path) and not hosted.enabled():
            return _db

        if not hosted.enabled() and not eos_path.exists():
            raise FileNotFoundError(f"eos.db not found at {eos_path}")

        lakehouse_dir = _resolve_lakehouse_dir().resolve()
        ok, message = _validate_lakehouse_dir(lakehouse_dir)
        if not ok:
            raise RuntimeError(message)

        import duckdb

        if _db is not None:
            try:
                _db.close()
            except Exception:
                pass

        conn = duckdb.connect(":memory:")
        conn.execute("SET TimeZone = 'UTC'")
        escaped = str(eos_path).replace("'", "''")
        if hosted.enabled():
            from eos.analytics_snapshot import read_analytics_snapshot, load_duckdb_snapshot
            snapshot = read_analytics_snapshot(eos_path)
            manifest = json.loads((lakehouse_dir / _LAKEHOUSE_MANIFEST_FILENAME).read_text())
            if manifest['sources']['eos_db'] != snapshot.provenance:
                conn.close()
                raise RuntimeError('LAKEHOUSE_STALE_EOS: PostgreSQL snapshot changed during analytics load')
            load_duckdb_snapshot(conn, snapshot)
        else:
            conn.execute(f"ATTACH '{escaped}' AS eos (TYPE SQLITE)")
        _create_analytics_views(conn)
        _create_lakehouse_views(conn)

        if not _table_exists(conn, "rea_claims"):
            raise RuntimeError("EOS DB missing required table: rea_claims")

        _db = conn
        _db_eos_path = str(eos_path)
        _db_governed_view_scope = None
        logger.info("Business analytics projection loaded for governed EOS binding at %s", eos_path)
        return conn


_GOVERNED_QUERY_SCHEMA = {
    "name": "governed_query",
    "description": (
        "Run read-only analytics against organization-scoped governed views. "
        "SQL identifiers and separators must be ASCII-only; localized labels must be mapped through governed_schema metadata. "
        "Authorization is resolved from the active workspace user and organization in the governed auth store. "
        "Available views: org_rea_claims, org_journal_voucher_projections, org_general_ledger_views, "
        "org_general_ledger_balance_lines, "
        "org_financial_statement_packages, org_tax_filing_packages, org_accounting_archive_packages. "
        "Raw EOS catalogs, files, and unscoped lakehouse views are not exposed. "
        "For table profiling, quality gates, sample-size checks, or data-feasibility summaries, "
        "use the sibling governed_data_engineering toolset instead of improvising diagnostics in SQL."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": (
                    "Read-only DuckDB SQL query against org_* scoped views. "
                    "Use ASCII-only SQL identifiers and ASCII punctuation separators."
                ),
            },
            "reload": {
                "type": "boolean",
                "description": "Force reload of EOS SQLite attachment (default false).",
                "default": False,
            },
        },
        "required": ["sql"],
    },
}


_GOVERNED_SCHEMA_SCHEMA = {
    "name": "governed_schema",
    "description": (
        "Inspect authorization-scoped governed analytics view schemas before writing SQL. "
        "Returns columns, display metadata, and optional sample rows for org_* governed views. "
        "Use this tool to map localized display labels to ASCII machine identifiers before calling governed_query. "
        "Use governed_data_engineering for profiling or quality-gate diagnostics over explicit table artifacts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "view": {
                "type": "string",
                "description": (
                    "Optional ASCII org_* scoped view name. Omit to inspect all governed analytics views."
                ),
            },
            "include_sample_rows": {
                "type": "boolean",
                "description": "Include up to sample_limit rows per inspected view (default true).",
                "default": True,
            },
            "sample_limit": {
                "type": "integer",
                "description": "Maximum sample rows per view, clamped to 0..5 (default 1).",
                "default": 1,
            },
            "reload": {
                "type": "boolean",
                "description": "Force reload of EOS SQLite attachment (default false).",
                "default": False,
            },
        },
        "required": [],
    },
}


_DUCKDB_PYTHON_SCHEMA = {
    "name": "duckdb_python",
    "description": (
        "Run read-only DuckDB SQL through the Python DuckDB API against the same "
        "authorization-enforced org_* governed views as governed_query. Use this "
        "instead of terminal, execute_code, ad hoc Python, or raw DB files when a "
        "DuckDB/Python-backed analytics tool surface is needed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": (
                    "Read-only DuckDB SQL query against org_* scoped views. "
                    "Use ASCII-only SQL identifiers and ASCII punctuation separators."
                ),
            },
            "reload": {
                "type": "boolean",
                "description": "Force reload of EOS SQLite attachment (default false).",
                "default": False,
            },
        },
        "required": ["sql"],
    },
}


_GOVERNED_POLICY_EVAL_SCHEMA = {
    "name": "governed_policy_eval",
    "description": (
        "Evaluate a REA semantic claim with zero-dependency embedded policy runtime "
        "using hash-pinned rego policy text."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "input": {
                "type": "object",
                "description": "Policy input object. Expected shape includes semantic_claim.resources[].delta.",
            },
            "policy_path": {
                "type": "string",
                "description": "Optional absolute or relative path to a .rego policy file.",
            },
        },
        "required": ["input"],
    },
}


_GOVERNED_REPORT_EXPORT_SCHEMA = {
    "name": "governed_report_export",
    "description": (
        "Persist a generated analytics report under the active session artifact root "
        "and attach artifact metadata to the session run payload for Inspector. "
        "Return one user-facing artifact path link target; do not ask agents to "
        "print a separate Inspector raw-link line. "
        "Use this after drafting a governed analytics report when the user asks "
        "to generate, create, prepare, or save a business report, including "
        "Chinese tax-report requests including the phrase 生成报税报告."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Report body to persist as a UTF-8 markdown artifact.",
            },
            "session_id": {
                "type": "string",
                "description": (
                    "Active workspace session id, optionally workspace-prefixed. "
                    "Defaults to the trusted active execution boundary when available."
                ),
            },
            "run_id": {
                "type": "string",
                "description": "Optional current run id. Defaults to report-export.",
            },
            "filename_stem": {
                "type": "string",
                "description": "Optional ASCII filename stem. Unsafe characters are normalized.",
            },
        },
        "required": ["content"],
    },
}


def _validate_governed_query_sql(sql: str) -> tuple[bool, str]:
    normalized = " " + " ".join(sql.lower().split()) + " "
    stripped = sql.strip().lower()
    if not stripped:
        return False, "SQL_REQUIRED: sql is required"
    if _contains_quoted_fullwidth_comma_identifier(sql):
        return False, (
            "QUERY_SQL_IDENTIFIER_VIOLATION: quoted identifiers must name one column; "
            'use "debit_10k", "credit_10k", and "net_10k" as separate ASCII identifiers.'
        )
    if _contains_fullwidth_sql_punctuation(sql):
        return False, (
            "QUERY_SQL_SYNTAX_VIOLATION: SQL must use ASCII punctuation for separators; "
            "use debit_10k, credit_10k, net_10k."
        )
    if _contains_non_ascii_sql_identifier(sql):
        return False, (
            "QUERY_SQL_IDENTIFIER_VIOLATION: governed_query identifiers must be ASCII-stable; "
            "use display_columns or column_metadata for localized labels."
        )
    if not (stripped.startswith("select ") or stripped.startswith("with ")):
        return False, "QUERY_NOT_READ_ONLY: governed_query only allows SELECT or WITH queries"
    if ";" in stripped.rstrip(";"):
        return False, "QUERY_NOT_READ_ONLY: governed_query accepts one read-only statement at a time"
    if any(pattern in normalized for pattern in _GOVERNED_QUERY_FORBIDDEN_SQL_PATTERNS):
        return False, (
            "QUERY_SCOPE_VIOLATION: governed_query cannot access raw catalogs, files, "
            "unscoped EOS tables, or mutation functions"
        )
    if not any(view_name in normalized for view_name in _GOVERNED_QUERY_SCOPED_VIEWS):
        return False, (
            "QUERY_SCOPE_VIOLATION: governed_query requires one of the org_* scoped governed views"
        )
    return True, ""


def _execute_governed_duckdb_sql(args: dict) -> dict[str, Any]:
    if args.get("reload", False):
        _reset_db()

    ok, requirement_error = _check_analytics_requirements()
    if not ok:
        return json.loads(_json_error_response(
            requirement_error,
            details={"stage": "requirements_check"},
        ))

    sql = (args.get("sql") or "").strip()
    ok, validation_error = _validate_governed_query_sql(sql)
    if not ok:
        return json.loads(_json_error_response(validation_error))

    org_context, denial = _authorize_active_governed_dataset_access()
    if org_context is None:
        return json.loads(denial)

    try:
        conn = _load_db()
    except (FileNotFoundError, RuntimeError) as exc:
        return json.loads(_json_error_response(str(exc), details={"stage": "db_init"}))

    try:
        with _db_lock:
            real_diagnostics, real_error = _real_governed_query_diagnostics(conn, org_context)
            if real_error:
                return json.loads(_json_error_response(
                    real_error,
                    details={"stage": "real_dataset_preflight", **real_diagnostics},
                ))
            _ensure_governed_org_views(conn, org_context)
            rel = conn.execute(sql)
            columns = [desc[0] for desc in rel.description]
            rows = rel.fetchall()
        result = [dict(zip(columns, row)) for row in rows]
        return {
            "columns": columns,
            "display_columns": _display_columns_for_result(columns),
            "column_metadata": _column_metadata_for_result(columns),
            "rows": result,
            "count": len(result),
            "authorization": {
                "decision": "ALLOW",
                "organization_name": org_context.get("organization_name"),
                "dataset_key": org_context.get("dataset_key"),
                "active_dataset_version_id": org_context.get("active_dataset_version_id"),
                "activation_set_snapshot_id": org_context.get("activation_set_snapshot_id"),
                "activation_resolver_policy_version": org_context.get("activation_resolver_policy_version"),
                "resolved_activation_set_hash": org_context.get("resolved_activation_set_hash"),
                "real_dataset_diagnostics": real_diagnostics or None,
            },
        }
    except Exception as exc:
        return json.loads(_json_error_response(
            f"QUERY_EXECUTION_ERROR: Query failed: {exc}",
            details={"stage": "query_exec"},
        ))


def _normalize_schema_view(raw_view: Any) -> str | None:
    view = str(raw_view or "").strip()
    if not view:
        return None
    if not view.isascii() or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", view):
        raise ValueError("SCHEMA_VIEW_IDENTIFIER_INVALID: view must be an ASCII org_* identifier")
    if view not in _GOVERNED_QUERY_SCOPED_VIEWS:
        allowed = ", ".join(_GOVERNED_QUERY_SCOPED_VIEWS)
        raise ValueError(f"SCHEMA_VIEW_NOT_GOVERNED: view must be one of: {allowed}")
    return view


def _clamped_sample_limit(raw_limit: Any) -> int:
    try:
        value = int(raw_limit)
    except (TypeError, ValueError):
        value = 1
    return max(0, min(value, 5))


def _describe_governed_view(conn: Any, view: str, *, sample_limit: int) -> dict[str, Any]:
    describe_rel = conn.execute(f"DESCRIBE SELECT * FROM {view}")
    describe_columns = [desc[0] for desc in describe_rel.description]
    describe_rows = [dict(zip(describe_columns, row)) for row in describe_rel.fetchall()]
    columns = [str(row.get("column_name") or "") for row in describe_rows if row.get("column_name")]
    payload: dict[str, Any] = {
        "columns": columns,
        "column_types": {
            str(row.get("column_name")): str(row.get("column_type") or "")
            for row in describe_rows
            if row.get("column_name")
        },
        "display_columns": _display_columns_for_result(columns),
        "column_metadata": _column_metadata_for_result(columns),
    }
    if sample_limit > 0:
        sample_rel = conn.execute(f"SELECT * FROM {view} LIMIT {sample_limit}")
        sample_columns = [desc[0] for desc in sample_rel.description]
        payload["sample_rows"] = [
            dict(zip(sample_columns, row))
            for row in sample_rel.fetchall()
        ]
    return payload


def _execute_governed_schema(args: dict) -> dict[str, Any]:
    if args.get("reload", False):
        _reset_db()

    ok, requirement_error = _check_analytics_requirements()
    if not ok:
        return json.loads(_json_error_response(
            requirement_error,
            details={"stage": "requirements_check"},
        ))

    try:
        requested_view = _normalize_schema_view(args.get("view"))
    except ValueError as exc:
        return json.loads(_json_error_response(str(exc)))

    include_sample_rows = bool(args.get("include_sample_rows", True))
    sample_limit = _clamped_sample_limit(args.get("sample_limit", 1)) if include_sample_rows else 0
    views = (requested_view,) if requested_view else _GOVERNED_QUERY_SCOPED_VIEWS

    org_context, denial = _authorize_active_governed_dataset_access()
    if org_context is None:
        return json.loads(denial)

    try:
        conn = _load_db()
    except (FileNotFoundError, RuntimeError) as exc:
        return json.loads(_json_error_response(str(exc), details={"stage": "db_init"}))

    try:
        with _db_lock:
            real_diagnostics, real_error = _real_governed_query_diagnostics(conn, org_context)
            if real_error:
                return json.loads(_json_error_response(
                    real_error,
                    details={"stage": "real_dataset_preflight", **real_diagnostics},
                ))
            _ensure_governed_org_views(conn, org_context)
            view_payloads = {
                view: _describe_governed_view(conn, view, sample_limit=sample_limit)
                for view in views
            }
        return {
            "views": view_payloads,
            "count": len(view_payloads),
            "sample_limit": sample_limit,
            "authorization": {
                "decision": "ALLOW",
                "organization_name": org_context.get("organization_name"),
                "dataset_key": org_context.get("dataset_key"),
                "active_dataset_version_id": org_context.get("active_dataset_version_id"),
                "activation_set_snapshot_id": org_context.get("activation_set_snapshot_id"),
                "activation_resolver_policy_version": org_context.get("activation_resolver_policy_version"),
                "resolved_activation_set_hash": org_context.get("resolved_activation_set_hash"),
                "real_dataset_diagnostics": real_diagnostics or None,
            },
        }
    except Exception as exc:
        return json.loads(_json_error_response(
            f"SCHEMA_DISCOVERY_ERROR: Schema discovery failed: {exc}",
            details={"stage": "schema_discovery"},
        ))


def _governed_query_handler(args: dict, **_kw: Any) -> str:
    return json.dumps(
        _execute_governed_duckdb_sql(args),
        ensure_ascii=False,
        indent=2,
    )


def _governed_schema_handler(args: dict, **_kw: Any) -> str:
    return json.dumps(
        _execute_governed_schema(args),
        ensure_ascii=False,
        indent=2,
    )


def _duckdb_python_handler(args: dict, **_kw: Any) -> str:
    payload = _execute_governed_duckdb_sql(args)
    if not payload.get("error"):
        payload["execution"] = {
            "tool": "duckdb_python",
            "engine": "duckdb",
            "api": "python",
            "scope": "authorized_org_scoped_views",
        }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _governed_policy_eval_handler(args: dict, **_kw: Any) -> str:
    input_obj = args.get("input")
    if not isinstance(input_obj, dict):
        return json.dumps({"error": "input must be an object"}, ensure_ascii=False)

    raw_path = args.get("policy_path")
    policy_path = Path(raw_path).expanduser() if isinstance(raw_path, str) and raw_path.strip() else _resolve_default_policy_path()

    try:
        bundle = load_embedded_policy(policy_path)
        result = evaluate_embedded_policy(bundle, input_obj)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"Policy evaluation failed: {exc}"}, ensure_ascii=False)


def _safe_filename_stem(value: Any) -> str:
    raw = str(value or "governed_report").strip().lower()
    normalized = re.sub(r"[^a-z0-9._-]+", "-", raw).strip(".-_")
    return normalized or "governed_report"


def _current_execution_boundary_session_id() -> str:
    try:
        from gateway.execution_boundary import current_execution_boundary

        boundary = current_execution_boundary()
    except Exception:
        return ""
    if boundary is None:
        return ""
    return str(getattr(boundary, "session_id", "") or "").strip()


def _session_id_from_artifacts_dir() -> str:
    raw = os.environ.get("SEMANTIER_WORKSPACE_ARTIFACTS_DIR")
    if not raw:
        return ""
    path = Path(raw).expanduser()
    parts = path.parts
    if len(parts) >= 3 and parts[-1] == "artifacts" and parts[-3] == "sessions":
        return parts[-2]
    return ""


def _resolve_report_export_session_id(args: dict) -> str:
    explicit = str(args.get("session_id") or "").strip()
    if explicit:
        return explicit
    boundary_session_id = _current_execution_boundary_session_id()
    if boundary_session_id:
        return boundary_session_id
    return _session_id_from_artifacts_dir()


def _now_epoch_millis() -> int:
    from datetime import datetime, timezone

    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _governed_report_export_handler(args: dict, **_kw: Any) -> str:
    content = args.get("content")
    if not isinstance(content, str) or not content.strip():
        return _json_error_response("REPORT_CONTENT_REQUIRED: content must be a non-empty string")

    session_id = _resolve_report_export_session_id(args)
    if not session_id:
        return _json_error_response("SESSION_ID_REQUIRED: session_id is required")

    hermes_home_raw = os.environ.get("HERMES_HOME")
    if not hermes_home_raw:
        return _json_error_response("HERMES_HOME_REQUIRED: active workspace HERMES_HOME is required")

    from agents.workspace_session_logs import workspace_session_artifacts_dir, workspace_session_dir

    hermes_home = Path(hermes_home_raw).expanduser().resolve()
    artifacts_dir = workspace_session_artifacts_dir(hermes_home, session_id).resolve()
    stem = _safe_filename_stem(args.get("filename_stem"))
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    artifact_path = artifacts_dir / f"{stem}-{digest[:12]}.md"
    artifact_path.write_text(content, encoding="utf-8")
    relative_path = artifact_path.relative_to(artifacts_dir).as_posix()
    encoded_relative_path = quote(relative_path, safe="")
    raw_url = (
        f"/sessions/{quote(session_id, safe='')}/artifacts/"
        f"{'/'.join(quote(part, safe='') for part in relative_path.split('/'))}/raw"
    )
    display_path = f"/sessions/{session_id}/artifacts/{relative_path}"
    inspector_href = f"#artifact={encoded_relative_path}"

    descriptor = {
        "artifact_id": f"report:{digest[:16]}",
        "kind": "governed_report",
        "media_type": "text/markdown",
        "filename": artifact_path.name,
        "path": str(artifact_path),
        "relative_path": relative_path,
        "raw_url": raw_url,
        "sha256": digest,
        "size_bytes": artifact_path.stat().st_size,
    }

    now_ms = _now_epoch_millis()
    run_id = str(args.get("run_id") or "report-export").strip() or "report-export"
    runs_dir = workspace_session_dir(hermes_home, session_id) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_path = runs_dir / f"{_safe_filename_stem(run_id)}.json"
    try:
        run_payload = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        run_payload = {}
    if not isinstance(run_payload, dict):
        run_payload = {}
    run_payload.setdefault("runId", run_id)
    run_payload.setdefault("sessionKey", session_id)
    run_payload.setdefault("status", "complete")
    run_payload.setdefault("createdAt", now_ms)
    run_payload["updatedAt"] = now_ms

    artifacts = run_payload.get("artifacts")
    if not isinstance(artifacts, list):
        artifacts = []
    artifacts = [item for item in artifacts if not (isinstance(item, dict) and item.get("path") == str(artifact_path))]
    artifacts.append(descriptor)
    run_payload["artifacts"] = artifacts

    lifecycle_events = run_payload.get("lifecycleEvents")
    if not isinstance(lifecycle_events, list):
        lifecycle_events = []
    lifecycle_events.append(
        {
            "type": "artifact_created",
            "timestamp": now_ms,
            "artifact": descriptor,
        }
    )
    run_payload["lifecycleEvents"] = lifecycle_events
    run_path.write_text(json.dumps(run_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    return json.dumps(
        {
            "artifact": descriptor,
            "display_path": display_path,
            "inspector_href": inspector_href,
            "user_message_guidance": (
                "Show one saved-location line: markdown-link the visible file path "
                f"{display_path} to {inspector_href}. Do not print a separate "
                "Inspector link or raw URL line."
            ),
            "run_payload": str(run_path),
            "lifecycle_event": lifecycle_events[-1],
        },
        ensure_ascii=False,
        indent=2,
    )


def _check_analytics_requirements() -> tuple[bool, str]:
    try:
        import duckdb as _  # noqa: F401, PLC0415
    except ImportError:
        return False, "duckdb not installed (run: uv add duckdb)"

    eos_path = _resolve_eos_path()
    from storage import hosted_postgres as hosted
    if not hosted.enabled() and not eos_path.exists():
        return False, f"EOS_DB_MISSING: eos.db not found at {eos_path}. run: semantier bootstrap --replace"

    lakehouse_dir = _resolve_lakehouse_dir().expanduser()
    ok, message = _validate_lakehouse_dir(lakehouse_dir)
    if not ok:
        return False, message

    try:
        import duckdb  # noqa: PLC0415

        conn = duckdb.connect(":memory:")
        escaped = str(eos_path).replace("'", "''")
        if hosted.enabled():
            from eos.analytics_snapshot import read_analytics_snapshot, load_duckdb_snapshot
            load_duckdb_snapshot(conn, read_analytics_snapshot(eos_path))
        else:
            conn.execute(f"ATTACH '{escaped}' AS eos (TYPE SQLITE)")
        ok = _table_exists(conn, "rea_claims")
        conn.close()
        if not ok:
            return False, "eos.db missing required table rea_claims"
    except Exception as exc:
        return False, f"unable to attach eos.db via DuckDB: {exc}"

    return True, ""


def _analytics_tool_available() -> bool:
    ok, _message = _check_analytics_requirements()
    return ok


def register(ctx: Any) -> None:
    ctx.register_tool(
        name="governed_query",
        toolset="governed_analytics",
        schema=_GOVERNED_QUERY_SCHEMA,
        handler=_governed_query_handler,
        emoji="🔐",
    )
    ctx.register_tool(
        name="governed_schema",
        toolset="governed_analytics",
        schema=_GOVERNED_SCHEMA_SCHEMA,
        handler=_governed_schema_handler,
        emoji="🧭",
    )
    ctx.register_tool(
        name="duckdb_python",
        toolset="governed_analytics",
        schema=_DUCKDB_PYTHON_SCHEMA,
        handler=_duckdb_python_handler,
        emoji="📊",
    )
    ctx.register_tool(
        name="governed_policy_eval",
        toolset="governed_analytics",
        schema=_GOVERNED_POLICY_EVAL_SCHEMA,
        handler=_governed_policy_eval_handler,
        emoji="⚖️",
    )
    ctx.register_tool(
        name="governed_report_export",
        toolset="governed_analytics",
        schema=_GOVERNED_REPORT_EXPORT_SCHEMA,
        handler=_governed_report_export_handler,
        emoji="📄",
    )
