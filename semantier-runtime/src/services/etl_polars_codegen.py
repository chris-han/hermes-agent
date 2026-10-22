"""Deterministic graph-specific eager Polars source generation.

Only emitter-owned Python is executable. Saved operation configs are quoted data;
no recipe interpreter, Semantier imports, live retrieval or model is exported.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from services.etl_transformations import (
    adapt_etl_definition,
    validate_config,
    MAX_NODES,
)

EMITTER_VERSION = "etl-polars-codegen.v1"
POLARS_VERSION = "1.41.2"
OPENPYXL_VERSION = "3.1.5"


def _literal(value):
    return repr(value)


def _prepare(definition, stage_id):
    clean = json.loads(json.dumps(definition, allow_nan=False))
    versions = {
        node.get("config", {}).get("operationSchemaVersion")
        for node in clean.get("nodes", [])
    } - {None}
    if versions - {"etl_operation.v2"}:
        raise ValueError("UNSUPPORTED_OPERATION_SCHEMA_VERSION")
    clean = adapt_etl_definition(clean)
    nodes = {node["id"]: node for node in clean["nodes"]}
    if not 0 < len(nodes) <= MAX_NODES or len(nodes) != len(clean["nodes"]):
        raise ValueError("INVALID_ETL_NODES")
    if stage_id not in nodes:
        raise ValueError("UNKNOWN_STAGE")
    parents = {key: [] for key in nodes}
    for edge in clean["edges"]:
        if edge.get("dependency", {}).get("dependencyClass") != "data":
            raise ValueError("ETL_REQUIRES_DATA_DEPENDENCY")
        if edge["source"] not in nodes or edge["target"] not in nodes:
            raise ValueError("INVALID_ETL_EDGE")
        parents[edge["target"]].append(edge["source"])
    # Validate every config before handing out an executable claim, including unused branches.
    for node in nodes.values():
        config = node.get("config") or {}
        validate_config(config)
        op = config["operation"]
        if op == "filter" and (
            config["comparison"] not in {"eq", "ne", "lt", "le", "gt", "ge"}
            or config["type"] not in {"string", "decimal"}
        ):
            raise ValueError("INVALID_ETL_FILTER")
        if op == "join" and config["cardinality"] not in {"many_to_one", "one_to_one"}:
            raise ValueError("JOIN_CARDINALITY_REQUIRED")
        if op == "assert" and config["rule"] not in {
            "required",
            "unique",
            "allowed",
            "decimal",
            "join_match",
        }:
            raise ValueError("UNSUPPORTED_ASSERTION")
        if op == "quarantine" and "join_cardinality" in config["reasons"]:
            raise ValueError("INVALID_QUARANTINE_POLICY")
        if op == "select" and len(config["columns"]) != len(set(config["columns"])):
            raise ValueError("INVALID_ETL_COLUMNS")
        if op == "source":
            if (
                type(config.get("header_row", 1)) is not int
                or not 1 <= config.get("header_row", 1) <= 100
                or not isinstance(config.get("sheet", "Sheet1"), str)
                or not config.get("sheet", "Sheet1")
            ):
                raise ValueError("PARSER_PROFILE_REQUIRED")
            if config.get("encoding", "utf-8-sig") not in {
                "utf-8",
                "utf-8-sig",
                "gb18030",
            }:
                raise ValueError("UNSUPPORTED_ENCODING")
    source_aliases = [
        node["config"]["source"]
        for node in nodes.values()
        if node["config"]["operation"] == "source"
    ]
    if len(source_aliases) != len(set(source_aliases)):
        raise ValueError("DUPLICATE_SOURCE_ALIAS")
    order, visiting, visited = [], set(), set()

    def visit(key):
        if key in visiting:
            raise ValueError("ETL_CYCLE")
        if key in visited:
            return
        visiting.add(key)
        config = nodes[key]["config"]
        bound = parents[key]
        if config["operation"] == "join":
            left, right = config["left_stage"], config["right_stage"]
            if left == right or set(bound) != {left, right} or len(bound) != 2:
                raise ValueError("JOIN_SIDE_BINDING_REQUIRED")
            bound = parents[key] = [left, right]
        expected = (
            0
            if config["operation"] == "source"
            else 2
            if config["operation"] == "join"
            else 1
        )
        if len(bound) != expected:
            raise ValueError("INVALID_ETL_INPUT_ARITY")
        for parent in bound:
            visit(parent)
        visiting.remove(key)
        visited.add(key)
        order.append(key)

    visit(stage_id)
    return clean, nodes, parents, order


def _statement(config, parent_names, stage):
    op = config["operation"]
    p = parent_names[0] if parent_names else None
    q = _literal
    if op == "source":
        profile = {
            key: value
            for key, value in config.items()
            if key not in {"operation", "artifact_pin"}
        }
        return f"load_source(paths, SOURCE_BASIS, {q(profile)}, context)"
    if op == "select":
        return f"{p}.select({q(config['columns'] + ['__sources'])})"
    if op == "rename":
        return f"{p}.rename({q(config['columns'])}, strict=False)"
    if op == "normalize":
        # Native normalization stringifies null to 'None' rather than preserving null.
        expressions = ", ".join(
            f"pl.col({q(col)}).map_elements(lambda value: str(value).strip(), return_dtype=pl.String, skip_nulls=False)"
            for col in config["columns"]
        )
        return f"{p}.with_columns({expressions})"
    if op == "decimal":
        return f"cast_decimal({p}, {q(config['column'])}, {q(config['target'])}, {q(stage)}, context)"
    if op == "filter":
        return f"filter_rows({p}, {q(config['column'])}, {q(config['comparison'])}, {q(config['value'])}, {q(config['type'])}, {q(stage)}, context)"
    if op == "join":
        return f"checked_join({p}, {parent_names[1]}, {q(config['left_key'])}, {q(config['right_key'])}, {q(config['columns'])}, {q(config['cardinality'])}, {q(stage)}, context)"
    if op == "assert":
        return f"assert_rows({p}, {q(config['column'])}, {q(config['rule'])}, {q(config.get('values', []))}, {q(stage)}, context)"
    if op == "quarantine":
        return f"quarantine_rows({p}, {q(config['reasons'])}, context)"
    if op == "group":
        return f"exact_group({p}, {q(config['keys'])}, {q(config['sum'])}, {q(config['count'])}, {q(stage)}, context)"
    if op == "export":
        return f"{p}.clone()"
    raise ValueError("UNSUPPORTED_OPERATION")


def generate_polars_export(definition, stage_id, workflow_hash=None, source_basis=None):
    """Return downloadable text assets after closed-contract compilation.

    Service callers bind source_basis to the exact saved receipt. Omitted basis is
    supported for explicit local development but cannot claim pinned source replay.
    """
    clean, nodes, parents, order = _prepare(definition, stage_id)
    basis = json.loads(json.dumps(source_basis or [], allow_nan=False))
    if len({item["source"] for item in basis}) != len(basis) or any(
        not isinstance(item.get("content_hash"), str)
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", item["content_hash"])
        for item in basis
    ):
        raise ValueError("INVALID_SOURCE_BASIS")
    if basis and any(
        nodes[key]["config"]["source"] not in {item["source"] for item in basis}
        for key in order
        if nodes[key]["config"]["operation"] == "source"
    ):
        raise ValueError("SOURCE_BASIS_REQUIRED")
    for key in order:
        config = nodes[key]["config"]
        if config["operation"] != "source":
            continue
        expected = next(
            (item for item in basis if item["source"] == config["source"]), None
        )
        if (
            expected
            and "artifact_pin" in expected
            and config.get("artifact_pin") != expected["artifact_pin"]
        ):
            raise ValueError("ETL_SOURCE_PIN_BASIS_MISMATCH")
        if expected and "parser_profile" in expected:
            for field, default in [
                ("header_row", 1),
                ("sheet", "Sheet1"),
                ("encoding", "utf-8-sig"),
            ]:
                expected_value = expected["parser_profile"].get(field)
                if config.get(field, default) != (
                    expected_value if expected_value is not None else default
                ):
                    raise ValueError("ETL_PARSER_BASIS_MISMATCH")
    support = (
        Path(__file__)
        .with_name("etl_codegen_assets")
        .joinpath("polars_support.py")
        .read_text()
    )
    metadata = {
        "compiler_version": "etl-typed-compiler.v1",
        "executor_version": "native-etl.v1",
        "emitter_version": EMITTER_VERSION,
        "support_version": "etl-polars-support.v1",
        "contract_version": "etl_operation.v2"
        if any(
            "operationSchemaVersion" in node.get("config", {})
            for node in definition["nodes"]
        )
        else "native-etl.v1",
        "workflow_hash": workflow_hash,
        "workflow_version": definition.get("workflowVersion"),
        "stage_id": stage_id,
        "definition_hash": "sha256:"
        + hashlib.sha256(
            json.dumps(
                definition, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest(),
        "polars_version": POLARS_VERSION,
        "openpyxl_version": OPENPYXL_VERSION,
        "source_basis": basis,
        "source_hash_checks": bool(basis),
        "semantic_policy": {
            "arithmetic": "Decimal48; strict 24 integral/12 fractional lexical digits; no float",
            "order": "native sorted group keys; stable source and left-join order",
            "null": "native v1 strict casts/diagnostics",
            "parser": "native-etl.v1 CSV/XLSX exact header/encoding/sheet; original source locations",
        },
    }
    script = (
        support
        + "\n\n# Frozen compilation identity; configs below are data literals.\n"
    )
    script += f"METADATA = {_literal(metadata)}\nSOURCE_BASIS = {_literal(basis)}\n\n"
    names = {key: f"stage_{i:02d}" for i, key in enumerate(order)}
    mappings = {}
    for key in order:
        start = len(script.splitlines()) + 1
        op = nodes[key]["config"]["operation"]
        parent_names = [f"input_{i}" for i, _ in enumerate(parents[key])]
        parameters = (
            ["paths", "context"] if op == "source" else parent_names + ["context"]
        )
        comment_key = str(key).replace("\r", r"\r").replace("\n", r"\n")
        script += f"# Stage {comment_key} ({op})\ndef {names[key]}({', '.join(parameters)}):\n"
        script += (
            f"    return {_statement(nodes[key]['config'], parent_names, key)}\n\n"
        )
        mappings[key] = {
            "start_line": start,
            "end_line": len(script.splitlines()),
            "function": names[key],
            "operation": op,
        }
    script += (
        "def run_pipeline(paths):\n    context = {'issues': [], 'exceptions': [], 'quarantined': set(), 'stages': {}, 'executed_nodes': [], 'source_basis': [], 'node_status': "
        + repr({key: "not_run" for key in nodes})
        + "}\n"
    )
    for key in order:
        args = (
            ["paths", "context"]
            if nodes[key]["config"]["operation"] == "source"
            else [names[parent] + "_rows" for parent in parents[key]] + ["context"]
        )
        script += f"    context['node_status'][{key!r}] = 'running'\n    {names[key]}_rows = retain_stage({key!r}, {names[key]}({', '.join(args)}), context)\n"
    primary = next(
        nodes[key]["config"]["source"]
        for key in order
        if nodes[key]["config"]["operation"] == "source"
    )
    script += (
        f"    return finish({names[stage_id]}_rows, context, {primary!r}), context\n\n"
    )
    script += """def main():
    if pl.__version__ != METADATA['polars_version']:
        raise ValueError('POLARS_VERSION_MISMATCH')
    parser = argparse.ArgumentParser(description='Replay this saved eager Polars dataflow with exact local inputs.')
    parser.add_argument('--input', action='append', required=True, metavar='SOURCE=FILE')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    bindings = [value.split('=',1) for value in args.input]
    if any(len(item)!=2 for item in bindings) or len({item[0] for item in bindings}) != len(bindings):
        raise ValueError('INVALID_INPUT_BINDINGS')
    result, context = run_pipeline(dict(bindings))
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    (output/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    if result['status'] != 'completed':
        raise SystemExit('Full checks failed: '+json.dumps(result['issues']))
    for name, rows in [('result.csv',result['full_rows']),('exceptions.csv',result['exceptions'])]:
        empty_columns = None
        if METADATA['contract_version'] == 'etl_operation.v2':
            empty_columns = result['columns'] + ['__sources'] + (['__reasons'] if name == 'exceptions.csv' else [])
        (output/name).write_bytes(canonical_csv(rows, columns=empty_columns).encode('utf-8'))
    files = {name:'sha256:'+hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ('result.csv','exceptions.csv','evaluation.json')}
    manifest = dict(METADATA,source_basis=context['source_basis'],accepted_rows=result['accepted_rows'],exceptional_rows=len(result['exceptions']),files=files,external_replay_authority=False)
    manifest['script_hash'] = 'sha256:'+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,separators=(',',':')),encoding='utf-8')


if __name__ == '__main__':
    main()
"""
    code_map = dict(
        metadata,
        stages=mappings,
        script_hash="sha256:" + hashlib.sha256(script.encode()).hexdigest(),
    )
    return {
        "pipeline_polars.py": script,
        "requirements.txt": f"polars=={POLARS_VERSION}\nopenpyxl=={OPENPYXL_VERSION}\n",
        "code-map.json": json.dumps(
            code_map, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        + "\n",
    }
