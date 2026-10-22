from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _error(code: str, message: str, **details: Any) -> str:
    payload: dict[str, Any] = {
        "ok": False,
        "error": True,
        "error_code": code,
        "message": message,
    }
    if details:
        payload["details"] = details
    return _json(payload)


def _text(args: dict[str, Any], name: str, default: str = "") -> str:
    value = args.get(name)
    text = str(value).strip() if value is not None else ""
    return text or default


def _text_value(value: Any) -> str:
    return str(value or "").strip()


def _optional_text(value: Any) -> str | None:
    text = _text_value(value)
    return text or None


def _int(args: dict[str, Any], name: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        value = int(args.get(name, default))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _load_csv(args: dict[str, Any]) -> tuple[list[str], list[dict[str, str]], str | None]:
    csv_path = _text(args, "csv_path")
    if not csv_path:
        return [], [], "CSV_PATH_REQUIRED: csv_path is required"
    path = Path(csv_path).expanduser()
    if not path.exists() or not path.is_file():
        return [], [], f"CSV_NOT_FOUND: CSV file does not exist: {csv_path}"
    encoding = _text(args, "encoding", "utf-8-sig")
    delimiter = _text(args, "delimiter", ",")
    if len(delimiter) != 1:
        return [], [], "CSV_DELIMITER_INVALID: delimiter must be exactly one character"
    max_rows = _int(args, "max_rows", 50000, minimum=1)
    try:
        with path.open(newline="", encoding=encoding) as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            fieldnames = [str(item) for item in (reader.fieldnames or [])]
            rows: list[dict[str, str]] = []
            for index, row in enumerate(reader):
                if index >= max_rows:
                    break
                rows.append({str(key): "" if value is None else str(value) for key, value in row.items() if key is not None})
    except UnicodeDecodeError as exc:
        return [], [], f"CSV_ENCODING_ERROR: unable to decode CSV with {encoding}: {exc}"
    except OSError as exc:
        return [], [], f"CSV_READ_ERROR: unable to read CSV: {exc}"
    return fieldnames, rows, None


def _split_error(raw: str) -> tuple[str, str]:
    if ":" in raw:
        code, message = raw.split(":", 1)
        return code.strip(), message.strip()
    return "UNCLASSIFIED_ERROR", raw.strip()


def _missing_count(rows: list[dict[str, str]], column: str) -> int:
    return sum(1 for row in rows if str(row.get(column, "")).strip() == "")


def _numeric_values(rows: list[dict[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = str(row.get(column, "")).strip().replace(",", "")
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _basic_stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mean = sum(ordered) / n
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": mean,
        "median": ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2,
    }


def _column_profile(rows: list[dict[str, str]], column: str) -> dict[str, Any]:
    row_count = len(rows)
    values = [str(row.get(column, "")).strip() for row in rows]
    non_empty = [value for value in values if value]
    unique_values = set(non_empty)
    numeric_values = _numeric_values(rows, column)
    inferred_type = "numeric" if non_empty and len(numeric_values) / max(1, len(non_empty)) >= 0.9 else "string"
    profile: dict[str, Any] = {
        "missing_count": row_count - len(non_empty),
        "missing_rate": round((row_count - len(non_empty)) / row_count, 6) if row_count else 0,
        "unique_count": len(unique_values),
        "inferred_type": inferred_type,
        "is_constant": len(unique_values) <= 1 and row_count > 0,
        "is_id_like": row_count >= 10 and len(unique_values) / row_count >= 0.95,
    }
    stats = _basic_stats(numeric_values)
    if stats:
        profile["numeric"] = stats
    if unique_values and len(unique_values) <= 20:
        profile["top_values"] = Counter(non_empty).most_common(10)
    return profile


def _class_balance(rows: list[dict[str, str]], target: str) -> dict[str, Any] | None:
    if not target or target not in (rows[0].keys() if rows else ()):
        return None
    values = [str(row.get(target, "")).strip() for row in rows if str(row.get(target, "")).strip()]
    if not values:
        return None
    counts = Counter(values)
    total = sum(counts.values())
    return {
        "target": target,
        "class_count": len(counts),
        "counts": dict(counts.most_common()),
        "minority_rate": round(min(counts.values()) / total, 6) if total else 0,
    }


def governed_profile_table(args: dict[str, Any], **_kw: Any) -> str:
    columns, rows, error = _load_csv(args)
    if error:
        code, message = _split_error(error)
        return _error(code, message)

    sample_limit = _int(args, "sample_limit", 5, minimum=0, maximum=20)
    target = _text(args, "target")
    duplicate_count = len(rows) - len({tuple(row.get(column, "") for column in columns) for row in rows})
    column_profiles = {column: _column_profile(rows, column) for column in columns}
    issues: list[dict[str, Any]] = []
    for column, profile in column_profiles.items():
        if profile["missing_rate"] >= 0.5:
            issues.append({"severity": "HIGH", "code": "HIGH_MISSING_RATE", "column": column})
        if profile["is_constant"]:
            issues.append({"severity": "MED", "code": "CONSTANT_COLUMN", "column": column})
        if profile["is_id_like"]:
            issues.append({"severity": "LOW", "code": "ID_LIKE_COLUMN", "column": column})
    if duplicate_count:
        issues.append({"severity": "MED", "code": "DUPLICATE_ROWS", "count": duplicate_count})

    return _json(
        {
            "ok": True,
            "source": {"kind": "csv", "path": _text(args, "csv_path"), "authority": "explicit_artifact"},
            "row_count": len(rows),
            "column_count": len(columns),
            "columns": columns,
            "duplicate_row_count": duplicate_count,
            "column_profiles": column_profiles,
            "target_profile": _class_balance(rows, target) if target else None,
            "sample_rows": rows[:sample_limit],
            "issues": issues,
            "limitations": [
                "CSV artifact profiling is diagnostic only and does not establish governed identity or semantic authority.",
                "Use governed analytics or dataset import records for organization-scoped authority.",
            ],
        }
    )


def company_refresh_lakehouse(args: dict[str, Any], **_kw: Any) -> str:
    from eos.real_company_lakehouse_refresh import refresh_lakehouse
    from eos.real_company_setup_state import real_company_setup_status_payload

    context = _resolve_company_context(args)
    organization_id = _text_value(context.get("organization_id"))
    dataset_type = _text_value(context.get("dataset_type"))
    dataset_version_id = _text_value(context.get("dataset_version_id"))
    error_code = _text_value(context.get("error_code"))
    if error_code:
        return _json(
            {
                "ok": False,
                "setup_status": error_code,
                "blocking_reasons": [error_code],
                "result": {
                    "error_code": error_code,
                    "details": context,
                    "handoff": "resolve_active_workspace_organization",
                },
                "next_actions": [],
            }
        )
    if dataset_type != "REAL":
        status = real_company_setup_status_payload(
            organization_id=organization_id,
            dataset_type=dataset_type,
            active_dataset_version_id=dataset_version_id,
        )
        return _json(
            {
                "ok": True,
                "setup_status": status.get("setup_status"),
                "blocking_reasons": status.get("blocking_reasons", []),
                "result": {
                    **status,
                    "handoff": "governed_query",
                },
                "next_actions": ["governed_query"],
            }
        )

    result = refresh_lakehouse(
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        requested_by=_text(args, "requested_by"),
        idempotency_key=_text(args, "idempotency_key"),
        db_path=_optional_text(args.get("db_path")),
        sim_json=_optional_text(args.get("sim_json")),
        output_dir=_optional_text(args.get("output_dir")),
        replace=bool(args.get("replace", True)),
    )
    status = real_company_setup_status_payload(
        organization_id=organization_id,
        dataset_type=dataset_type,
        active_dataset_version_id=dataset_version_id,
    )
    return _json(
        {
            "ok": bool(result.get("ok", True)),
            "setup_status": status.get("setup_status"),
            "blocking_reasons": status.get("blocking_reasons", []),
            "result": {**status, "result": result},
            "next_actions": status.get("next_actions", []),
        }
    )


def _resolve_company_context(args: dict[str, Any]) -> dict[str, str | None]:
    organization_id = _text_value(args.get("organization_id") or args.get("company_id"))
    dataset_type = _text_value(args.get("dataset_type") or args.get("company_type")).upper()
    dataset_version_id = _optional_text(
        args.get("active_dataset_version_id") or args.get("dataset_version_id") or args.get("dataset_key")
    )

    active_context = _active_workspace_org_context()
    if active_context and _text_value(active_context.get("organization_id")):
        organization_id = _text_value(active_context.get("organization_id"))
        dataset_type = _text_value(active_context.get("dataset_type")).upper()
        dataset_version_id = (
            _optional_text(active_context.get("active_dataset_version_id"))
            or _optional_text(active_context.get("dataset_key"))
            or dataset_version_id
        )

    if not dataset_type:
        dataset_type = _resolve_dataset_type(organization_id)
    if not organization_id:
        return {
            "organization_id": organization_id,
            "dataset_type": dataset_type,
            "dataset_version_id": dataset_version_id,
            "error_code": "ACTIVE_ORGANIZATION_REQUIRED",
        }
    if not dataset_type:
        return {
            "organization_id": organization_id,
            "dataset_type": dataset_type,
            "dataset_version_id": dataset_version_id,
            "error_code": "DATASET_CONTEXT_REQUIRED",
        }

    return {
        "organization_id": organization_id,
        "dataset_type": dataset_type,
        "dataset_version_id": dataset_version_id,
    }


def _resolve_dataset_type(organization_id: str) -> str:
    if organization_id:
        try:
            from agents.gateway_identity import _load_organizations

            org = _load_organizations().get(organization_id) or {}
            governed_type = _text_value(org.get("dataset_type")).upper()
            if governed_type:
                return governed_type
        except Exception:
            pass
    return ""


def _active_workspace_org_context() -> dict[str, Any] | None:
    workspace_id = _workspace_id_from_session_home()
    if not workspace_id:
        return None
    try:
        from agents import gateway_identity

        user = gateway_identity.get_user_record_by_workspace_id(workspace_id)
        if not isinstance(user, dict):
            return None
        user_id = _text_value(user.get("user_id"))
        if not user_id:
            return None
        org_context = gateway_identity.get_user_organization_context(user_id)
        return dict(org_context) if isinstance(org_context, dict) else None
    except Exception:
        return None


def _workspace_id_from_session_home() -> str | None:
    raw_home = os.environ.get("HERMES_HOME") or os.environ.get("HERMES_SESSION_HERMES_HOME")
    if not raw_home:
        return None
    try:
        hermes_home = Path(raw_home).expanduser().resolve()
    except OSError:
        return None
    workspace_id = hermes_home.parent.name.strip() if hermes_home.name == ".hermes" else hermes_home.name.strip()
    return workspace_id or None


def _rule_failed(failures: list[dict[str, Any]], code: str, **payload: Any) -> None:
    failures.append({"code": code, **payload})


def governed_quality_gate(args: dict[str, Any], **_kw: Any) -> str:
    columns, rows, error = _load_csv(args)
    if error:
        code, message = _split_error(error)
        return _error(code, message)
    rules = args.get("rules")
    if not isinstance(rules, dict):
        return _error("QUALITY_RULES_REQUIRED", "rules must be an object")

    failures: list[dict[str, Any]] = []
    dataset_rules = rules.get("dataset") if isinstance(rules.get("dataset"), dict) else {}
    min_rows = dataset_rules.get("min_rows")
    max_rows = dataset_rules.get("max_rows")
    if isinstance(min_rows, int) and len(rows) < min_rows:
        _rule_failed(failures, "MIN_ROWS", expected=min_rows, actual=len(rows))
    if isinstance(max_rows, int) and len(rows) > max_rows:
        _rule_failed(failures, "MAX_ROWS", expected=max_rows, actual=len(rows))
    if bool(dataset_rules.get("no_duplicate_rows")):
        duplicate_count = len(rows) - len({tuple(row.get(column, "") for column in columns) for row in rows})
        if duplicate_count:
            _rule_failed(failures, "DUPLICATE_ROWS", count=duplicate_count)

    column_rules = rules.get("columns") if isinstance(rules.get("columns"), dict) else {}
    for column, raw_rule in column_rules.items():
        rule = raw_rule if isinstance(raw_rule, dict) else {}
        exists = column in columns
        if bool(rule.get("required")) and not exists:
            _rule_failed(failures, "REQUIRED_COLUMN", column=column)
            continue
        if not exists:
            continue
        if bool(rule.get("non_null")):
            missing = _missing_count(rows, column)
            if missing:
                _rule_failed(failures, "NON_NULL", column=column, missing_count=missing)
        if bool(rule.get("unique")):
            values = [row.get(column, "") for row in rows]
            if len(values) != len(set(values)):
                _rule_failed(failures, "UNIQUE", column=column)
        values = _numeric_values(rows, column)
        if values and isinstance(rule.get("min"), (int, float)) and min(values) < float(rule["min"]):
            _rule_failed(failures, "MIN_VALUE", column=column, expected=rule["min"], actual=min(values))
        if values and isinstance(rule.get("max"), (int, float)) and max(values) > float(rule["max"]):
            _rule_failed(failures, "MAX_VALUE", column=column, expected=rule["max"], actual=max(values))
        if isinstance(rule.get("enum"), list):
            allowed = {str(item) for item in rule["enum"]}
            invalid = sorted({row.get(column, "") for row in rows if row.get(column, "") not in allowed})
            if invalid:
                _rule_failed(failures, "ENUM", column=column, invalid_values=invalid[:10])
        if isinstance(rule.get("regex"), str):
            pattern = re.compile(rule["regex"])
            invalid_count = sum(1 for row in rows if not pattern.fullmatch(str(row.get(column, ""))))
            if invalid_count:
                _rule_failed(failures, "REGEX", column=column, invalid_count=invalid_count)

    return _json(
        {
            "ok": not failures,
            "status": "PASS" if not failures else "FAIL",
            "row_count": len(rows),
            "column_count": len(columns),
            "failures": failures,
            "source": {"kind": "csv", "path": _text(args, "csv_path"), "authority": "explicit_artifact"},
        }
    )


def governed_sample_size_check(args: dict[str, Any], **_kw: Any) -> str:
    task = _text(args, "task")
    n = _int(args, "n", 0, minimum=0)
    features = _int(args, "features", 0, minimum=0)
    per_class = args.get("per_class") if isinstance(args.get("per_class"), list) else []
    per_class_values = [int(item) for item in per_class if isinstance(item, int) and item >= 0]
    findings: list[dict[str, Any]] = []

    status = "ok"
    if task == "classification":
        classes = _int(args, "classes", len(per_class_values), minimum=0)
        min_class = min(per_class_values) if per_class_values else (n // classes if classes else 0)
        if classes <= 1 or min_class < 50:
            status = "insufficient"
        elif min_class < 100:
            status = "warn"
        findings.append({"metric": "min_class_samples", "value": min_class, "warn_below": 100, "insufficient_below": 50})
    elif task == "regression":
        ratio = n / max(1, features)
        if features <= 0 or ratio < 10:
            status = "insufficient"
        elif ratio < 20:
            status = "warn"
        findings.append({"metric": "samples_per_feature", "value": round(ratio, 3), "warn_below": 20, "insufficient_below": 10})
    elif task == "detection":
        min_class = min(per_class_values) if per_class_values else 0
        if min_class < 25:
            status = "insufficient"
        elif min_class < 50:
            status = "warn"
        findings.append({"metric": "min_detection_instances", "value": min_class, "warn_below": 50, "insufficient_below": 25})
    else:
        return _error("TASK_INVALID", "task must be classification, regression, or detection")

    return _json(
        {
            "ok": status != "insufficient",
            "status": status,
            "n": n,
            "task": task,
            "findings": findings,
            "limitations": ["This is an empirical warning gate, not a formal power analysis."],
        }
    )


def governed_data_feasibility(args: dict[str, Any], **_kw: Any) -> str:
    dimensions = {
        "data_support": _text(args, "data_support"),
        "quality": _text(args, "quality"),
        "scale": _text(args, "scale"),
        "feature_value": _text(args, "feature_value"),
    }
    rank = {"ok": 0, "warn": 1, "insufficient": 2}
    if any(value not in rank for value in dimensions.values()):
        return _error("FEASIBILITY_STATUS_INVALID", "dimension statuses must be ok, warn, or insufficient")
    worst = max(dimensions.values(), key=lambda item: rank[item])
    overall = {
        "ok": "usable",
        "warn": "usable_with_caveats",
        "insufficient": "insufficient",
    }[worst]
    evidence = args.get("evidence") if isinstance(args.get("evidence"), dict) else {}
    return _json(
        {
            "ok": overall != "insufficient",
            "project": _text(args, "project"),
            "overall": overall,
            "dimensions": dimensions,
            "evidence": {key: str(evidence.get(key) or "") for key in dimensions},
            "handoff": {
                "idea_review": "continue" if overall != "insufficient" else "block_until_data_gap_resolved",
                "artifact": "data_feasibility.md",
            },
        }
    )
