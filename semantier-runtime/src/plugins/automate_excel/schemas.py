from __future__ import annotations

TOOLSET_NAME = "automate_excel"


def _schema(description: str, properties: dict, required: tuple[str, ...] = ()):
    return {
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False,
        },
    }


_STRING_LIST = {"type": "array", "items": {"type": "string"}}
_SHEET = {"type": "string", "default": "0", "description": "Worksheet name or zero-based index."}
_ENCODING = {"type": "string", "default": "utf-8-sig"}
_CSV_SEP = {"type": "string", "default": ","}

MERGE_EXCEL_SHEETS_SCHEMA = _schema(
    "Merge multiple Excel files or selected sheets into one workbook.",
    {
        "inputs": _STRING_LIST,
        "output": {"type": "string"},
        "sheet": _STRING_LIST,
        "header_row": {"type": "integer", "default": 0},
    },
    ("inputs", "output"),
)

EXPORT_EXCEL_TO_CSV_SCHEMA = _schema(
    "Export one Excel sheet to CSV.",
    {
        "input": {"type": "string"},
        "output": {"type": "string"},
        "sheet": _SHEET,
        "encoding": _ENCODING,
        "sep": _CSV_SEP,
    },
    ("input", "output"),
)

CONVERT_CSV_TO_EXCEL_SCHEMA = _schema(
    "Convert one or more CSV files to an Excel workbook.",
    {
        "input": {"type": "string"},
        "inputs": _STRING_LIST,
        "output": {"type": "string"},
        "encoding": _ENCODING,
        "sep": _CSV_SEP,
        "sheet_name": {"type": "string"},
    },
    ("output",),
)

FILTER_EXCEL_ROWS_SCHEMA = _schema(
    "Filter Excel rows with one or more conditions.",
    {
        "input": {"type": "string"},
        "output": {"type": "string"},
        "sheet": _SHEET,
        "where": _STRING_LIST,
        "encoding": _ENCODING,
    },
    ("input", "output", "where"),
)

SPLIT_EXCEL_FILE_SCHEMA = _schema(
    "Split an Excel file by row count or by a column value.",
    {
        "input": {"type": "string"},
        "sheet": _SHEET,
        "by_rows": {"type": "integer", "minimum": 1},
        "by_column": {"type": "string"},
        "output_dir": {"type": "string", "default": "."},
        "prefix": {"type": "string", "default": "part"},
    },
    ("input",),
)

DEDUPLICATE_EXCEL_ROWS_SCHEMA = _schema(
    "Remove duplicate rows from an Excel file.",
    {
        "input": {"type": "string"},
        "output": {"type": "string"},
        "sheet": _SHEET,
        "keys": {"type": "string"},
        "keep": {"type": "string", "enum": ["first", "last"], "default": "first"},
        "encoding": _ENCODING,
    },
    ("input", "output", "keys"),
)

AGGREGATE_EXCEL_ROWS_SCHEMA = _schema(
    "Group and aggregate Excel rows.",
    {
        "input": {"type": "string"},
        "output": {"type": "string"},
        "sheet": _SHEET,
        "group_by": {"type": "string"},
        "agg": _STRING_LIST,
    },
    ("input", "group_by", "agg"),
)

VALIDATE_EXCEL_FILE_SCHEMA = _schema(
    "Validate required columns, duplicate keys, and empty rows.",
    {
        "input": {"type": "string"},
        "sheet": _SHEET,
        "key_cols": {"type": "string"},
        "require_cols": {"type": "string"},
        "encoding": _ENCODING,
        "header_row": {"type": "integer", "minimum": 0},
    },
    ("input",),
)

SELECT_EXCEL_COLUMNS_SCHEMA = _schema(
    "Select, rename, and order columns in an Excel file.",
    {
        "input": {"type": "string"},
        "output": {"type": "string"},
        "sheet": _SHEET,
        "columns": {"type": "string"},
        "rename": {"type": "string"},
        "encoding": _ENCODING,
    },
    ("input", "output", "columns"),
)

MERGE_EXCEL_TABLES_SCHEMA = _schema(
    "Merge two Excel tables by shared key columns.",
    {
        "left": {"type": "string"},
        "right": {"type": "string"},
        "on": {"type": "string"},
        "output": {"type": "string"},
        "sheet_left": _SHEET,
        "sheet_right": _SHEET,
        "how": {"type": "string", "enum": ["left", "inner", "outer"], "default": "left"},
        "suffixes": {"type": "string", "default": "_left,_right"},
    },
    ("left", "right", "on", "output"),
)

VLOOKUP_EXCEL_TABLES_SCHEMA = _schema(
    "Left-join a main table with multiple lookup tables.",
    {
        "main": {"type": "string"},
        "lookups": _STRING_LIST,
        "output": {"type": "string"},
        "sheet_main": _SHEET,
        "suffix": {"type": "string", "default": "_lookup"},
    },
    ("main", "lookups", "output"),
)

TRANSPOSE_EXCEL_TABLE_SCHEMA = _schema(
    "Transpose an Excel table.",
    {
        "input": {"type": "string"},
        "output": {"type": "string"},
        "sheet": _SHEET,
        "first_col_as_header": {"type": "boolean", "default": False},
    },
    ("input", "output"),
)

FILL_EXCEL_TEMPLATE_SCHEMA = _schema(
    "Fill an Excel template row using values from an Excel or CSV data table.",
    {
        "template": {"type": "string"},
        "data": {"type": "string"},
        "output": {"type": "string"},
        "sheet_template": _SHEET,
        "sheet_data": _SHEET,
        "pattern_row": {"type": "integer", "minimum": 1, "default": 2},
        "header_row": {"type": "integer", "minimum": 1, "default": 1},
    },
    ("template", "data", "output"),
)

RENAME_EXCEL_SHEETS_SCHEMA = _schema(
    "Rename Excel worksheets by explicit names, prefix, or suffix.",
    {
        "input": {"type": "string"},
        "output": {"type": "string"},
        "rename": _STRING_LIST,
        "prefix": {"type": "string"},
        "suffix": {"type": "string"},
    },
    ("input",),
)

FORMAT_EXCEL_CONDITIONAL_SCHEMA = _schema(
    "Apply conditional formatting to an Excel range or column.",
    {
        "input": {"type": "string"},
        "output": {"type": "string"},
        "sheet": _SHEET,
        "range": {"type": "string"},
        "column": {"type": "string"},
        "rule": {"type": "string", "enum": ["gt", "gte", "lt", "lte", "eq", "between", "duplicates"]},
        "value": {"type": "string"},
        "min": {"type": "string"},
        "max": {"type": "string"},
        "fill": {"type": "string", "default": "FF6B6B"},
        "color_scale": {"type": "boolean", "default": False},
    },
    ("input",),
)

FORMAT_EXCEL_COLUMNS_AS_TEXT_SCHEMA = _schema(
    "Set selected Excel columns to text format.",
    {
        "input": {"type": "string"},
        "output": {"type": "string"},
        "sheet": _SHEET,
        "columns": {"type": "string"},
        "header_row": {"type": "integer", "minimum": 1, "default": 1},
    },
    ("input", "columns"),
)
