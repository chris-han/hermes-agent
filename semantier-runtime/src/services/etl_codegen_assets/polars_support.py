"""etl-polars-support.v1: exact local parsing, row diagnostics and Decimal48.

Frames are eager Polars relations. Decimal values deliberately use Object columns:
Polars Decimal128 precision38 cannot represent every native precision48 reduction.
No numeric input is converted through float. These helpers do not interpret recipes.
"""

import argparse  # noqa: F401 -- consumed by the generated CLI appended to this asset
import csv
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import io
import json
import operator
from pathlib import Path
import re

import polars as pl

MAX_ROWS = 100000
SUPPORT_VERSION = "etl-polars-support.v1"


def wire(value):
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, dict):
        return {key: wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire(item) for item in value]
    return value


def exact_decimal(value):
    text = str(value).strip()
    if not re.fullmatch(r"[+-]?\d{1,24}(?:\.\d{1,12})?", text):
        raise ValueError("invalid_decimal")
    with localcontext() as context:
        context.prec = 48
        return Decimal(text)


def locations(rows):
    unique = {}
    for row in rows:
        for loc in row.get("__sources", []):
            unique[(loc["source"], loc["sheet"], loc["row"])] = loc
    return [unique[key] for key in sorted(unique)]


def frame(rows, columns=None):
    columns = (
        columns
        if columns is not None
        else list(dict.fromkeys(key for row in rows for key in row))
    )
    series = []
    for col in columns:
        values = [row.get(col) for row in rows]
        if col.startswith("__") or any(isinstance(value, Decimal) for value in values):
            series.append(pl.Series(col, values, dtype=pl.Object))
        else:
            series.append(pl.Series(col, values, strict=True))
    return pl.DataFrame(series)


def issue(context, code, stage, rows, column=None):
    context["issues"].append(
        {
            "code": code,
            "stage_id": stage,
            "column": column,
            "actual_value": wire(rows[0].get(column)) if rows and column else None,
            "sources": locations(rows),
        }
    )


def load_source(paths, basis, profile, context):
    source = profile["source"]
    if source not in paths:
        raise ValueError("SOURCE_UNAVAILABLE")
    path = Path(paths[source])
    data = path.read_bytes()
    if len(data) > 20 * 1024 * 1024:
        raise ValueError("SOURCE_SIZE_LIMIT_EXCEEDED")
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    expected = next((item for item in basis if item["source"] == source), None)
    if expected and digest != expected["content_hash"]:
        raise ValueError("ETL_SOURCE_BASIS_MISMATCH")
    header, sheet = profile.get("header_row", 1), profile.get("sheet", "Sheet1")
    if (
        type(header) is not int
        or not 1 <= header <= 100
        or not isinstance(sheet, str)
        or not sheet
    ):
        raise ValueError("PARSER_PROFILE_REQUIRED")
    if path.suffix.lower() == ".csv":
        encoding = profile.get("encoding", "utf-8-sig")
        if encoding not in {"utf-8", "utf-8-sig", "gb18030"}:
            raise ValueError("UNSUPPORTED_ENCODING")
        values = list(csv.reader(io.StringIO(data.decode(encoding))))
    elif path.suffix.lower() == ".xlsx":
        import openpyxl

        if openpyxl.__version__ != METADATA["openpyxl_version"]:  # noqa: F821 -- frozen identity appended by emitter
            raise ValueError("OPENPYXL_VERSION_MISMATCH")
        workbook = openpyxl.load_workbook(
            io.BytesIO(data), read_only=True, data_only=True
        )
        try:
            if sheet not in workbook.sheetnames:
                raise ValueError("SHEET_UNAVAILABLE")
            values = []
            for row in workbook[sheet].iter_rows(values_only=True):
                values.append(["" if value is None else str(value) for value in row])
                if len(values) > MAX_ROWS + header:
                    raise ValueError("ETL_ROW_LIMIT_EXCEEDED")
        finally:
            workbook.close()
    else:
        raise ValueError("UNSUPPORTED_SOURCE")
    if len(values) < header:
        raise ValueError("HEADER_UNAVAILABLE")
    columns = values[header - 1]
    if (
        not columns
        or len(columns) != len(set(columns))
        or any(not col or col.startswith("__") for col in columns)
    ):
        raise ValueError("INVALID_ETL_COLUMNS")
    if len(values) - header > MAX_ROWS:
        raise ValueError("ETL_ROW_LIMIT_EXCEEDED")
    rows = []
    for index, values_row in enumerate(values[header:], header + 1):
        if len(values_row) != len(columns):
            raise ValueError("SOURCE_ROW_WIDTH_MISMATCH")
        rows.append(
            dict(
                zip(columns, values_row),
                __sources=[{"source": source, "sheet": sheet, "row": index}],
            )
        )
    resolved_profile = {
        key: profile.get(key, default)
        for key, default in [
            ("header_row", 1),
            ("sheet", "Sheet1"),
            ("encoding", "utf-8-sig"),
        ]
    }
    if expected and "parser_profile" in expected:
        expected_profile = {
            key: expected["parser_profile"].get(key)
            if expected["parser_profile"].get(key) is not None
            else default
            for key, default in [
                ("header_row", 1),
                ("sheet", "Sheet1"),
                ("encoding", "utf-8-sig"),
            ]
        }
        if resolved_profile != expected_profile:
            raise ValueError("ETL_PARSER_BASIS_MISMATCH")
    context["source_basis"].append(
        dict(
            expected or {},
            source=source,
            content_hash=digest,
            total_rows=len(rows),
            parser_profile=resolved_profile,
        )
    )
    return frame(rows, columns + ["__sources"])


def cast_decimal(data, column, target, stage, context):
    rows = data.to_dicts()
    for row in rows:
        try:
            row[target] = exact_decimal(row[column])
        except (ValueError, InvalidOperation):
            row[target] = None
            issue(context, "invalid_decimal", stage, [row], column)
    return frame(rows, list(dict.fromkeys(data.columns + [target])))


def filter_rows(data, column, comparison, literal, kind, stage, context):
    convert = exact_decimal if kind == "decimal" else str
    expected = convert(literal)
    comparator = {
        "eq": operator.eq,
        "ne": operator.ne,
        "lt": operator.lt,
        "le": operator.le,
        "gt": operator.gt,
        "ge": operator.ge,
    }[comparison]
    mask = []
    for row in data.to_dicts():
        try:
            mask.append(comparator(convert(row[column]), expected))
        except ValueError:
            mask.append(True)
            issue(context, "invalid_decimal", stage, [row], column)
    return data.filter(pl.Series("__keep", mask, dtype=pl.Boolean))


def checked_join(
    left, right, left_key, right_key, mapping, cardinality, stage, context
):
    left_rows, right_rows = left.to_dicts(), right.to_dicts()
    index = {}
    for row in right_rows:
        key = row[right_key]
        if key in index or key is None or key == "":
            issue(
                context,
                "join_cardinality",
                stage,
                [row, index.get(key, row)],
                right_key,
            )
        index[key] = row
    if cardinality == "one_to_one":
        seen = {}
        for row in left_rows:
            key = row[left_key]
            if key in seen:
                issue(context, "join_cardinality", stage, [seen[key], row], left_key)
            seen[key] = row
    if any(
        item["code"] == "join_cardinality" and item["stage_id"] == stage
        for item in context["issues"]
    ):
        context["node_status"][stage] = "failed"
        return left.clone()
    if set(mapping.values()) & set(left.columns):
        raise ValueError("ETL_COLUMN_COLLISION")
    # Polars join over stable row identities supports exact Object/Decimal keys too.
    # Explicit Python equality chooses native key identity; Polars performs the eager
    # relation join, preserving left order and unmatched null projections.
    positions = {row[right_key]: i for i, row in enumerate(right_rows)}
    left_index = pl.DataFrame(
        {
            "__left_row": range(len(left_rows)),
            "__right_row": pl.Series(
                [positions.get(row[left_key]) for row in left_rows], dtype=pl.Int64
            ),
        }
    )
    right_index = pl.DataFrame(
        {
            "__right_row": pl.Series(range(len(right_rows)), dtype=pl.Int64),
            "__matched": pl.Series(range(len(right_rows)), dtype=pl.Int64),
        }
    )
    joined = left_index.join(
        right_index, on="__right_row", how="left", maintain_order="left"
    )
    output = []
    for identity in joined.to_dicts():
        row = dict(left_rows[identity["__left_row"]])
        other = (
            right_rows[identity["__matched"]]
            if identity["__matched"] is not None
            else None
        )
        if other is None:
            issue(context, "join_unmatched", stage, [row], left_key)
        else:
            row["__sources"] = locations([row, other])
        for source, target in mapping.items():
            row[target] = other.get(source) if other else None
        output.append(row)
    return frame(output, left.columns + list(mapping.values()))


def assert_rows(data, column, rule, allowed, stage, context):
    seen = {}
    for row in data.to_dicts():
        value, failed = row[column], False
        if rule in {"required", "join_match"}:
            failed = value is None or value == ""
        elif rule == "allowed":
            failed = value not in allowed
        elif rule == "decimal":
            try:
                exact_decimal(value)
            except ValueError:
                failed = True
        elif rule == "unique":
            if value in seen:
                issue(context, "unique", stage, [seen[value]], column)
                failed = True
            seen[value] = row
        if failed:
            issue(context, rule, stage, [row], column)
    return data.clone()


def quarantine_rows(data, reasons, context):
    by_location = {}
    for index, item in enumerate(context["issues"]):
        if item["code"] in reasons and item["sources"]:
            loc = item["sources"][0]
            by_location.setdefault(
                (loc["source"], loc["sheet"], loc["row"]), []
            ).append((index, item))
    mask = []
    for row in data.to_dicts():
        own = {(loc["source"], loc["sheet"], loc["row"]) for loc in row["__sources"]}
        matching = [
            (index, item)
            for loc in own
            for index, item in by_location.get(loc, [])
            if all(
                (loc["source"], loc["sheet"], loc["row"]) in own
                for loc in item["sources"]
            )
        ]
        mask.append(not matching)
        if matching:
            context["exceptions"].append(
                dict(row, __reasons=sorted({item["code"] for _, item in matching}))
            )
            context["quarantined"].update(index for index, _ in matching)
    return data.filter(pl.Series("__keep", mask, dtype=pl.Boolean))


def exact_group(data, keys, amount, count, stage, context):
    if "currency" not in keys and "currency" in data.columns and data.height:
        raise ValueError("CURRENCY_DIMENSION_REQUIRED")
    mask = []
    for row in data.to_dicts():
        valid = row.get(amount) is not None
        if valid:
            try:
                exact_decimal(row[amount])
            except ValueError:
                valid = False
                issue(context, "invalid_decimal", stage, [row], amount)
        mask.append(valid)
    valid_data = data.filter(pl.Series("__valid", mask, dtype=pl.Boolean))
    # Polars partitions relations; Decimal48 reduction is intentionally visible.
    if any(valid_data.schema[key] == pl.Object for key in keys):
        identities, group_ids = {}, []
        for row in valid_data.to_dicts():
            identity = tuple(row[key] for key in keys)
            group_ids.append(identities.setdefault(identity, len(identities)))
        partitioned = valid_data.with_columns(
            pl.Series("__group_id", group_ids, dtype=pl.Int64)
        )
        partitions = [
            part.drop("__group_id")
            for part in partitioned.partition_by("__group_id", maintain_order=True)
        ]
    else:
        partitions = valid_data.partition_by(keys, maintain_order=True)
    groups = []
    for partition in partitions:
        members = partition.to_dicts()
        identity = tuple(members[0][key] for key in keys)
        with localcontext() as context48:
            context48.prec = 48
            total = sum(
                (exact_decimal(row[amount]) for row in members), Decimal("0.00")
            )
        groups.append(
            (
                identity,
                dict(
                    zip(keys, identity),
                    **{
                        amount: total,
                        count: len(members),
                        "__sources": locations(members),
                    },
                ),
            )
        )
    return frame(
        [row for _, row in sorted(groups, key=lambda pair: pair[0])],
        keys + [amount, count, "__sources"],
    )


def canonical_csv(rows, columns=None):
    stream = io.StringIO(newline="")
    columns = (
        list(dict.fromkeys(key for row in rows for key in row))
        if rows or columns is None
        else columns
    )
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in wire(rows):
        writer.writerow(
            {
                key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
        )
    return stream.getvalue()


def retain_stage(stage, data, context):
    if data.height > MAX_ROWS:
        raise ValueError("ETL_ROW_LIMIT_EXCEEDED")
    context["stages"][stage] = wire(data.to_dicts())
    context["executed_nodes"].append(stage)
    if context["node_status"][stage] != "failed":
        context["node_status"][stage] = "completed"
    return data


def finish(data, context, primary_source):
    rows = wire(data.to_dicts())
    unresolved = [
        item
        for index, item in enumerate(context["issues"])
        if index not in context["quarantined"]
    ]
    failed = bool(unresolved) or "failed" in context["node_status"].values()
    return {
        "status": "failed" if failed else "completed",
        "full_check_status": "failed" if unresolved else "passed",
        "full_rows": rows,
        "columns": [col for col in data.columns if not col.startswith("__")],
        "exceptions": wire(context["exceptions"]),
        "issues": [
            dict(item, resolved=index in context["quarantined"])
            for index, item in enumerate(context["issues"])
        ],
        "accepted_rows": len(
            {
                (loc["source"], loc["sheet"], loc["row"])
                for row in rows
                for loc in row.get("__sources", [])
                if loc["source"] == primary_source
            }
        ),
        "executed_nodes": context["executed_nodes"],
        "node_status": context["node_status"],
        "stages": context["stages"],
    }
