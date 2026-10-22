"""Bounded deterministic ETL evaluator. This module is also standalone-exportable.

Preview/check/run share computation. Only callers publish retained run receipts;
this evaluator performs no filesystem, network, model or output effects.
"""

from copy import deepcopy
from decimal import Decimal, InvalidOperation, localcontext
import re

EXECUTOR_VERSION = "native-etl.v1"
DISPLAY_LIMIT = 20
MAX_ROWS = 100000
MAX_NODES = 64
ALLOWED = {
    "source",
    "select",
    "rename",
    "normalize",
    "decimal",
    "filter",
    "join",
    "assert",
    "quarantine",
    "group",
    "export",
}


def strict_decimal(value):
    text = str(value).strip()
    if not re.fullmatch(r"[+-]?\d{1,24}(?:\.\d{1,12})?", text):
        raise ValueError("invalid_decimal")
    with localcontext() as ctx:
        ctx.prec = 48
        number = Decimal(text)
        if not number.is_finite():
            raise ValueError("invalid_decimal")
        return number


def wire(value):
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, dict):
        return {k: wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire(v) for v in value]
    return value


def lineage(rows):
    unique = {}
    for row in rows:
        for loc in row.get("__sources", []):
            unique[(loc["source"], loc["sheet"], loc["row"])] = loc
    return [unique[k] for k in sorted(unique)]


def _issue(code, node, rows, column=None):
    return {
        "code": code,
        "stage_id": node,
        "column": column,
        "actual_value": wire(rows[0].get(column)) if rows and column else None,
        "sources": lineage(rows),
    }


# Closed operation contracts: configurations are data, never executable text.
CONFIG_KEYS = {
    "source": {
        "operation",
        "source",
        "artifact_pin",
        "sheet",
        "header_row",
        "encoding",
    },
    "select": {"operation", "columns"},
    "rename": {"operation", "columns"},
    "normalize": {"operation", "columns"},
    "decimal": {"operation", "column", "target"},
    "filter": {"operation", "column", "comparison", "value", "type"},
    "join": {
        "operation",
        "left_stage",
        "right_stage",
        "left_key",
        "right_key",
        "columns",
        "cardinality",
    },
    "assert": {"operation", "column", "rule", "values"},
    "quarantine": {"operation", "reasons"},
    "group": {"operation", "keys", "sum", "count"},
    "export": {"operation"},
}


def validate_config(config):
    if not isinstance(config, dict) or config.get("operation") not in CONFIG_KEYS:
        raise ValueError("UNSUPPORTED_OPERATION")
    op = config["operation"]
    if set(config) - CONFIG_KEYS[op]:
        raise ValueError("UNSUPPORTED_ETL_CONFIGURATION")
    required = CONFIG_KEYS[op] - {"operation"}
    if op == "source":
        required = {"source"}
    if op == "assert":
        required = {"column", "rule"} | (
            {"values"} if config.get("rule") == "allowed" else set()
        )
    if required - set(config):
        raise ValueError("ETL_CONFIGURATION_REQUIRED")
    for key in {
        "source",
        "column",
        "target",
        "left_stage",
        "right_stage",
        "left_key",
        "right_key",
        "sum",
        "count",
    } & set(config):
        if (
            not isinstance(config[key], str)
            or not config[key]
            or config[key].startswith("__")
        ):
            raise ValueError("INVALID_ETL_COLUMNS")
    for key in {"keys", "reasons", "values"} & set(config):
        if (
            not isinstance(config[key], list)
            or not config[key]
            or any(not isinstance(x, str) or not x for x in config[key])
        ):
            raise ValueError("INVALID_ETL_CONFIGURATION")
    if op == "group" and (
        len(config["keys"]) != len(set(config["keys"]))
        or config["sum"] in config["keys"]
        or config["count"] in config["keys"]
        or config["count"] == config["sum"]
    ):
        raise ValueError("ETL_COLUMN_COLLISION")
    if "columns" in config:
        columns = config["columns"]
        if op in {"rename", "join"}:
            if (
                not isinstance(columns, dict)
                or not columns
                or any(
                    not isinstance(k, str)
                    or not isinstance(v, str)
                    or not k
                    or not v
                    or k.startswith("__")
                    or v.startswith("__")
                    for k, v in columns.items()
                )
            ):
                raise ValueError("INVALID_ETL_COLUMNS")
        elif (
            not isinstance(columns, list)
            or not columns
            or any(
                not isinstance(c, str) or not c or c.startswith("__") for c in columns
            )
        ):
            raise ValueError("INVALID_ETL_COLUMNS")


def evaluate(
    definition,
    sources,
    stage_id,
    *,
    mode="preview",
    cancelled=None,
    source_schemas=None,
):
    typed = is_typed_definition(definition)
    typed_definition = definition if typed else None
    if typed:
        validate_typed_definition(definition, source_schemas)
        definition = adapt_etl_definition(definition)
    if mode not in {"preview", "check", "run"}:
        raise ValueError("INVALID_ETL_MODE")
    nodes = {n["id"]: n for n in definition["nodes"]}
    if not 0 < len(nodes) <= MAX_NODES or len(nodes) != len(definition["nodes"]):
        raise ValueError("INVALID_ETL_NODES")
    if stage_id not in nodes:
        raise ValueError("UNKNOWN_STAGE")
    parents = {n: [] for n in nodes}
    for e in definition["edges"]:
        if e.get("dependency", {}).get("dependencyClass") != "data":
            raise ValueError("ETL_REQUIRES_DATA_DEPENDENCY")
        if e["source"] not in nodes or e["target"] not in nodes:
            raise ValueError("INVALID_ETL_EDGE")
        parents[e["target"]].append(e["source"])
    results, issues, exceptions, executed, states = {}, [], [], [], {}
    visiting = set()
    quarantined = set()

    def checked(rows):
        for row in rows:
            if cancelled and cancelled():
                raise InterruptedError("ETL_CANCELLED")
            yield row

    def visit(key):
        if key in results:
            return results[key]
        if key in visiting:
            raise ValueError("ETL_CYCLE")
        if cancelled and cancelled():
            raise InterruptedError("ETL_CANCELLED")
        visiting.add(key)
        config = nodes[key].get("config") or {}
        validate_config(config)
        op = config["operation"]
        bound_parents = parents[key]
        if op == "join":
            left, right = config.get("left_stage"), config.get("right_stage")
            if (
                not left
                or not right
                or left == right
                or set(bound_parents) != {left, right}
            ):
                raise ValueError("JOIN_SIDE_BINDING_REQUIRED")
            bound_parents = [left, right]
        inputs = [visit(p) for p in bound_parents]
        expected_inputs = 0 if op == "source" else 2 if op == "join" else 1
        if len(inputs) != expected_inputs:
            raise ValueError("INVALID_ETL_INPUT_ARITY")
        rows = deepcopy(inputs[0]) if inputs else []
        states[key] = "running"
        if op == "source":
            source = config.get("source")
            if source not in sources:
                raise ValueError("SOURCE_UNAVAILABLE")
            value = sources[source]
            rows = deepcopy(value() if callable(value) else value)
        elif op == "select":
            cols = config["columns"]
            if len(cols) != len(set(cols)) or any(c.startswith("__") for c in cols):
                raise ValueError("INVALID_ETL_COLUMNS")
            rows = [
                {**{c: r[c] for c in cols}, "__sources": r["__sources"]} for r in rows
            ]
        elif op == "rename":
            mapping = config["columns"]
            renamed = []
            for row in checked(rows):
                new = {mapping.get(c, c): v for c, v in row.items()}
                if len(new) != len(row) or any(
                    c.startswith("__") for c in mapping.values()
                ):
                    raise ValueError("ETL_COLUMN_COLLISION")
                renamed.append(new)
            rows = renamed
        elif op == "normalize":
            for row in checked(rows):
                for col in config["columns"]:
                    row[col] = str(row[col]).strip()
        elif op == "decimal":
            col, target = config["column"], config["target"]
            if target.startswith("__"):
                raise ValueError("INVALID_ETL_COLUMNS")
            for row in checked(rows):
                try:
                    row[target] = strict_decimal(row[col])
                except (ValueError, InvalidOperation):
                    row[target] = None
                    issues.append(_issue("invalid_decimal", key, [row], col))
        elif op == "filter":
            col, comparison, value = (
                config["column"],
                config["comparison"],
                config["value"],
            )
            ops = {
                "eq": lambda a, b: a == b,
                "ne": lambda a, b: a != b,
                "lt": lambda a, b: a < b,
                "le": lambda a, b: a <= b,
                "gt": lambda a, b: a > b,
                "ge": lambda a, b: a >= b,
            }
            if comparison not in ops or config.get("type") not in {"string", "decimal"}:
                raise ValueError("INVALID_ETL_FILTER")
            convert = strict_decimal if config["type"] == "decimal" else str
            compare_value = convert(value)
            kept = []
            for row in checked(rows):
                try:
                    if ops[comparison](convert(row[col]), compare_value):
                        kept.append(row)
                except ValueError:
                    # Failed casts must remain available to checks/quarantine, never vanish.
                    kept.append(row)
                    issues.append(_issue("invalid_decimal", key, [row], col))
            rows = kept
        elif op == "join":
            if config.get("cardinality") not in {"many_to_one", "one_to_one"}:
                raise ValueError("JOIN_CARDINALITY_REQUIRED")
            left_key, right_key = config["left_key"], config["right_key"]
            right = inputs[1]
            index = {}
            for row in checked(right):
                value = row[right_key]
                if value in index or value is None or value == "":
                    issues.append(
                        _issue(
                            "join_cardinality",
                            key,
                            [row, index.get(value, row)],
                            right_key,
                        )
                    )
                index[value] = row
            if config["cardinality"] == "one_to_one":
                seen = {}
                for row in checked(rows):
                    value = row[left_key]
                    if value in seen:
                        issues.append(
                            _issue(
                                "join_cardinality", key, [seen[value], row], left_key
                            )
                        )
                    seen[value] = row
            if any(
                i["code"] == "join_cardinality" and i["stage_id"] == key for i in issues
            ):
                states[key] = "failed"
            else:
                for row in checked(rows):
                    other = index.get(row[left_key])
                    if other is None:
                        issues.append(_issue("join_unmatched", key, [row], left_key))
                    else:
                        row["__sources"] = lineage([row, other])
                    for source, target in config["columns"].items():
                        if target in row or target.startswith("__"):
                            raise ValueError("ETL_COLUMN_COLLISION")
                        row[target] = other.get(source) if other else None
        elif op == "assert":
            col, rule = config["column"], config["rule"]
            if rule not in {"required", "unique", "allowed", "decimal", "join_match"}:
                raise ValueError("UNSUPPORTED_ASSERTION")
            seen = {}
            for row in checked(rows):
                value = row[col]
                failed = False
                if rule in {"required", "join_match"}:
                    failed = value is None or value == ""
                elif rule == "allowed":
                    failed = value not in config["values"]
                elif rule == "decimal":
                    try:
                        strict_decimal(value)
                    except ValueError:
                        failed = True
                elif rule == "unique":
                    if value in seen:
                        issues.append(_issue("unique", key, [seen[value]], col))
                        failed = True
                    seen[value] = row
                if failed:
                    issues.append(_issue(rule, key, [row], col))
        elif op == "quarantine":
            reasons = set(config["reasons"])
            if not reasons or "join_cardinality" in reasons:
                raise ValueError("INVALID_QUARANTINE_POLICY")
            by_location = {}
            for idx, issue in enumerate(issues):
                if issue["code"] in reasons and issue["sources"]:
                    loc = issue["sources"][0]
                    by_location.setdefault(
                        (loc["source"], loc["sheet"], loc["row"]), []
                    ).append((idx, issue))
            kept = []
            for row in checked(rows):
                own = {
                    (location["source"], location["sheet"], location["row"])
                    for location in row["__sources"]
                }
                # A joined customer's lineage must not quarantine its other orders.
                matching = [
                    (idx, i)
                    for loc in own
                    for idx, i in by_location.get(loc, [])
                    if all(
                        (location["source"], location["sheet"], location["row"]) in own
                        for location in i["sources"]
                    )
                ]
                if matching:
                    exceptions.append(
                        {**row, "__reasons": sorted({i["code"] for _, i in matching})}
                    )
                    quarantined.update(idx for idx, _ in matching)
                else:
                    kept.append(row)
            rows = kept
        elif op == "group":
            keys, col = config["keys"], config["sum"]
            if "currency" not in keys and any("currency" in r for r in rows):
                raise ValueError("CURRENCY_DIMENSION_REQUIRED")
            groups = {}
            for row in checked(rows):
                if row.get(col) is None:
                    continue  # Diagnostic aggregate is partial; status cannot claim success.
                try:
                    strict_decimal(row[col])
                except ValueError:
                    issues.append(_issue("invalid_decimal", key, [row], col))
                    continue
                k = tuple(row[c] for c in keys)
                groups.setdefault(k, []).append(row)
            output = []
            for k, members in sorted(groups.items()):
                with localcontext() as context:
                    context.prec = 48
                    total = sum(
                        (strict_decimal(m[col]) for m in members), Decimal("0.00")
                    )
                output.append(
                    {
                        **dict(zip(keys, k)),
                        col: total,
                        config["count"]: len(members),
                        "__sources": lineage(members),
                    }
                )
            rows = output
        elif op == "export":
            # Identity transformation only; preview can never perform an output effect.
            pass
        if len(rows) > MAX_ROWS:
            raise ValueError("ETL_ROW_LIMIT_EXCEEDED")
        results[key] = rows
        executed.append(key)
        if states[key] != "failed":
            states[key] = "completed"
        visiting.remove(key)
        return rows

    try:
        full = visit(stage_id)
        unresolved = [i for idx, i in enumerate(issues) if idx not in quarantined]
        failed = bool(unresolved) or any(v == "failed" for v in states.values())
        status = "failed" if failed and mode == "run" else "completed"
    except (KeyError, TypeError, InvalidOperation):
        full = []
        status = "failed"
        for key, value in states.items():
            if value == "running":
                states[key] = "failed"
                issues.append(_issue("stage_configuration_incompatible", key, []))
        unresolved = issues
    except InterruptedError:
        full = []
        unresolved = issues
        status = "cancelled"
        for key, value in states.items():
            if value == "running":
                states[key] = "cancelled"
    for key in nodes:
        states.setdefault(key, "not_run")
    profiles = {}
    for column in dict.fromkeys(
        k for row in full for k in row if not k.startswith("__")
    ):
        values = [row.get(column) for row in full]
        relevant = [i for i in unresolved if i["column"] == column]
        profiles[column] = {
            "types": sorted({type(v).__name__ for v in values if v is not None}),
            "null_count": sum(v is None or v == "" for v in values),
            "invalid_count": len(relevant),
            "row_count": len(full),
            "assertion_outcome": "not_run"
            if mode == "preview" or status == "cancelled"
            else "failed"
            if relevant
            else "passed",
        }
    full = wire(full)
    result = {
        "executor_version": EXECUTOR_VERSION,
        "mode": mode,
        "stage_id": stage_id,
        "status": status,
        "executed_nodes": executed,
        "node_status": states,
        "full_check_status": "not_run"
        if mode == "preview" or status == "cancelled"
        else "failed"
        if unresolved
        else "passed",
        "processed_rows": len(full),
        "returned_rows": min(len(full), DISPLAY_LIMIT),
        "display_truncated": len(full) > DISPLAY_LIMIT,
        "column_profiles": profiles,
        "columns": list(profiles),
        "sample_method": "display_only",
        "rows": full[:DISPLAY_LIMIT],
        "full_rows": full,
        "issues": wire(
            [
                {**issue, "resolved": index in quarantined}
                for index, issue in enumerate(issues)
            ]
        ),
        "unresolved_issue_count": len(unresolved),
        "accepted_rows": len(
            {
                (location["source"], location["sheet"], location["row"])
                for r in full
                for location in r.get("__sources", [])
                if location["source"]
                == next(
                    (
                        nodes[key].get("config", {}).get("source")
                        for key in executed
                        if nodes[key].get("config", {}).get("operation") == "source"
                    ),
                    None,
                )
            }
        ),
        "inputs": [
            {
                "stage_id": parent,
                "total_rows": len(results.get(parent, [])),
                "rows": wire(results.get(parent, [])[:DISPLAY_LIMIT]),
            }
            for parent in (
                [
                    nodes[stage_id]["config"]["left_stage"],
                    nodes[stage_id]["config"]["right_stage"],
                ]
                if nodes[stage_id].get("config", {}).get("operation") == "join"
                else parents[stage_id]
            )
        ],
        "exceptions": wire(exceptions),
        "stages": {key: wire(value) for key, value in results.items()},
    }

    if typed:
        inferred = {}
        for key, node in nodes.items():
            if node["config"]["operation"] == "source" and results.get(key):
                inferred[key] = list(
                    dict.fromkeys(
                        c for row in results[key] for c in row if not c.startswith("__")
                    )
                )
        for key, node in nodes.items():
            c = node["config"]
            if c["operation"] == "source" and source_schemas is not None:
                supplied = source_schemas.get(key, source_schemas.get(c["source"]))
                if supplied is not None:
                    inferred[key] = supplied
        schemas = validate_typed_definition(typed_definition, inferred)
        result["relation_schemas"] = schemas
        result["columns"] = [column["name"] for column in schemas[stage_id]["columns"]]
    return result


def is_typed_definition(definition):
    return any(
        "operationSchemaVersion" in n.get("config", {})
        for n in definition.get("nodes", [])
    ) or any(
        "schemaVersion" in e.get("config", {}) for e in definition.get("edges", [])
    )


def _typed_graph(definition):
    from contracts.etl_pipeline import ETL_OPERATION_ADAPTER, EtlInputBinding

    nodes = {n["id"]: n for n in definition["nodes"]}
    if not 0 < len(nodes) <= MAX_NODES or len(nodes) != len(definition["nodes"]):
        raise ValueError("INVALID_ETL_NODES")
    configs = {}
    for key, node in nodes.items():
        config = node.get("config", {})
        if config.get("operationSchemaVersion") != "etl_operation.v2":
            raise ValueError("ETL_OPERATION_VERSION_REQUIRED")
        configs[key] = ETL_OPERATION_ADAPTER.validate_python(config).model_dump(
            exclude_unset=True, exclude_none=True
        )
    inputs = {key: {} for key in nodes}
    edge_ids = [edge["id"] for edge in definition["edges"] if "id" in edge]
    if len(set(edge_ids)) != len(edge_ids):
        raise ValueError("DUPLICATE_ETL_EDGE")
    for edge in definition["edges"]:
        source, target = edge["source"], edge["target"]
        if source not in nodes or target not in nodes or source == target:
            raise ValueError("INVALID_ETL_EDGE")
        if (
            edge.get("transitionClass", "normal") != "normal"
            or edge.get("dependency", {}).get("dependencyClass") != "data"
        ):
            raise ValueError("ETL_REQUIRES_DATA_DEPENDENCY")
        binding = EtlInputBinding.model_validate(edge.get("config", {}))
        port = binding.targetPort
        allowed = (
            {"left", "right"} if configs[target]["operation"] == "join" else {"input"}
        )
        if configs[target]["operation"] == "source" or port not in allowed:
            raise ValueError("INVALID_ETL_PORT")
        if port in inputs[target]:
            raise ValueError("DUPLICATE_ETL_INPUT")
        inputs[target][port] = source
    for key, config in configs.items():
        required = (
            set()
            if config["operation"] == "source"
            else {"left", "right"}
            if config["operation"] == "join"
            else {"input"}
        )
        if set(inputs[key]) != required:
            raise ValueError("ETL_INPUT_BINDING_REQUIRED")
        if len(set(inputs[key].values())) != len(inputs[key]):
            raise ValueError("DUPLICATE_ETL_INPUT")
    ordered, visiting, done = [], set(), set()

    def visit(key):
        if key in visiting:
            raise ValueError("ETL_CYCLE")
        if key in done:
            return
        visiting.add(key)
        for port in sorted(inputs[key]):
            visit(inputs[key][port])
        visiting.remove(key)
        done.add(key)
        ordered.append(key)

    for key in nodes:
        visit(key)
    return configs, inputs, ordered


def validate_typed_definition(definition, source_schemas=None):
    """Validate the whole submitted DAG, then derive ordered structural schemas.

    Parser-declared schemas are keyed by source stage ID (source name also accepted).
    A missing schema is explicitly unknown; an empty declared schema is known-empty.
    This projection has no persistence or authority effects.
    """
    from contracts.etl_pipeline import EtlRelationSchema

    configs, inputs, ordered = _typed_graph(definition)
    schemas = {}
    source_schemas = source_schemas or {}
    for key in ordered:
        c = configs[key]
        op = c["operation"]
        if op == "source":
            supplied = source_schemas.get(key, source_schemas.get(c["source"]))
            if supplied is None:
                schema = {
                    "schemaVersion": "etl_relation.v2",
                    "columns": [],
                    "known": False,
                }
            elif isinstance(supplied, dict):
                schema = EtlRelationSchema.model_validate(supplied).model_dump()
            else:
                schema = EtlRelationSchema(
                    columns=[
                        {"name": name, "type": "unknown"}
                        if isinstance(name, str)
                        else name
                        for name in supplied
                    ]
                ).model_dump()
            schemas[key] = schema
            continue
        parent = inputs[key].get("input", inputs[key].get("left"))
        incoming = schemas[parent]
        columns = deepcopy(incoming["columns"])
        names = [column["name"] for column in columns]
        known = incoming["known"]

        def require(requested, schema=incoming):
            if schema["known"]:
                present = {column["name"] for column in schema["columns"]}
                if set(requested) - present:
                    raise ValueError("UNKNOWN_ETL_COLUMN:" + key)

        if op in {"select", "normalize", "rename"}:
            require(c["columns"])
        elif op in {"decimal", "filter", "assert"}:
            require([c["column"]])
        elif op == "group":
            require(c["keys"] + [c["sum"]])
        if op == "select":
            by_name = {column["name"]: column for column in columns}
            columns = [
                by_name.get(name, {"name": name, "type": "unknown"})
                for name in c["columns"]
            ]
            known = True
        elif op == "rename":
            if len(set(c["columns"].values())) != len(c["columns"]):
                raise ValueError("ETL_COLUMN_COLLISION")
            columns = [
                {**column, "name": c["columns"].get(column["name"], column["name"])}
                for column in columns
            ]
        elif op == "normalize":
            columns = [
                {
                    **column,
                    "type": "string"
                    if column["name"] in c["columns"]
                    else column["type"],
                }
                for column in columns
            ]
        elif op == "decimal":
            if c["target"] in names and c["target"] != c["column"]:
                raise ValueError("ETL_COLUMN_COLLISION")
            if c["target"] in names:
                columns = [
                    {
                        **column,
                        "type": "decimal"
                        if column["name"] == c["target"]
                        else column["type"],
                    }
                    for column in columns
                ]
            else:
                columns.append({"name": c["target"], "type": "decimal"})
        elif op == "filter" and c["type"] == "decimal":
            strict_decimal(c["value"])
        elif op == "join":
            right = schemas[inputs[key]["right"]]
            require([c["left_key"]])
            require([c["right_key"]] + list(c["columns"]), right)
            targets = list(c["columns"].values())
            if len(set(targets)) != len(targets) or set(targets) & set(names):
                raise ValueError("ETL_COLUMN_COLLISION")
            by_name = {column["name"]: column for column in right["columns"]}
            columns.extend(
                {"name": target, "type": by_name.get(source, {}).get("type", "unknown")}
                for source, target in c["columns"].items()
            )
            known = known and right["known"]
        elif op == "group":
            if known and "currency" in names and "currency" not in c["keys"]:
                raise ValueError("CURRENCY_DIMENSION_REQUIRED")
            by_name = {column["name"]: column for column in columns}
            columns = [
                by_name.get(name, {"name": name, "type": "unknown"})
                for name in c["keys"]
            ] + [
                {"name": c["sum"], "type": "decimal"},
                {"name": c["count"], "type": "integer"},
            ]
            known = True
        schemas[key] = EtlRelationSchema(columns=columns, known=known).model_dump()
    return schemas


def adapt_etl_definition(definition):
    """Return a transient evaluator-compatible copy without changing saved identity."""
    if not is_typed_definition(definition):
        return deepcopy(definition)
    configs, inputs, _ = _typed_graph(definition)
    validate_typed_definition(definition)
    result = deepcopy(definition)
    for node in result["nodes"]:
        c = deepcopy(configs[node["id"]])
        c.pop("operationSchemaVersion")
        if c["operation"] == "join":
            c["left_stage"] = inputs[node["id"]]["left"]
            c["right_stage"] = inputs[node["id"]]["right"]
        node["config"] = c
    return result


def migrate_etl_definition(definition):
    """Explicit v1 successor migration; legacy edge order never determines roles."""
    if is_typed_definition(definition):
        validate_typed_definition(definition)
        return deepcopy(definition)
    result = deepcopy(definition)
    nodes = {n["id"]: n for n in result["nodes"]}
    if len(nodes) != len(result["nodes"]):
        raise ValueError("INVALID_ETL_NODES")
    parents = {key: [] for key in nodes}
    for edge in result["edges"]:
        if edge["source"] not in nodes or edge["target"] not in nodes:
            raise ValueError("INVALID_ETL_EDGE")
        parents[edge["target"]].append(edge["source"])
    for key, node in nodes.items():
        config = node["config"]
        validate_config(config)
        if config["operation"] == "join":
            left, right = config["left_stage"], config["right_stage"]
            if (
                left == right
                or len(parents[key]) != 2
                or set(parents[key]) != {left, right}
            ):
                raise ValueError("JOIN_SIDE_BINDING_REQUIRED")
        for edge in result["edges"]:
            if edge["target"] != key:
                continue
            if edge.get("config"):
                raise ValueError("AMBIGUOUS_LEGACY_ETL_BINDING")
            port = (
                ("left" if edge["source"] == left else "right")
                if config["operation"] == "join"
                else "input"
            )
            edge["config"] = {
                "schemaVersion": "etl_binding.v2",
                "sourcePort": "rows",
                "targetPort": port,
            }
        config.pop("left_stage", None)
        config.pop("right_stage", None)
        config["operationSchemaVersion"] = "etl_operation.v2"
    validate_typed_definition(result)
    return result
