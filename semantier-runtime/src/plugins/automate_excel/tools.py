from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _text(args: dict[str, Any], name: str) -> str:
    return str(args.get(name) or "").strip()


def _items(args: dict[str, Any], name: str) -> list[str]:
    value = args.get(name)
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _append(argv: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        argv.extend([flag, text])


def _append_bool(argv: list[str], flag: str, value: Any) -> None:
    if bool(value):
        argv.append(flag)


def _script_path(script_name: str) -> Path:
    path = Path(__file__).resolve().parent / "scripts" / script_name
    if not path.exists():
        raise RuntimeError(f"bundled script missing: {script_name}")
    return path


def _run(script_name: str, argv: list[str], output_hint: Any = None) -> str:
    try:
        completed = subprocess.run(
            [sys.executable, str(_script_path(script_name)), *argv],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(Path.cwd()),
        )
    except Exception as exc:
        return _json({"ok": False, "error": str(exc), "script": script_name})

    return _json(
        {
            "ok": completed.returncode == 0,
            "script": script_name,
            "return_code": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "output": output_hint,
        }
    )


def merge_excel_sheets(args: dict[str, Any], **_kw: Any) -> str:
    argv = ["--inputs", *_items(args, "inputs")]
    _append(argv, "--output", args.get("output"))
    for sheet in _items(args, "sheet"):
        argv.extend(["--sheet", sheet])
    _append(argv, "--header-row", args.get("header_row"))
    return _run("merge_sheets.py", argv, _text(args, "output"))


def export_excel_to_csv(args: dict[str, Any], **_kw: Any) -> str:
    argv: list[str] = []
    _append(argv, "--input", args.get("input"))
    _append(argv, "--output", args.get("output"))
    _append(argv, "--sheet", args.get("sheet"))
    _append(argv, "--encoding", args.get("encoding"))
    _append(argv, "--sep", args.get("sep"))
    return _run("excel_to_csv.py", argv, _text(args, "output"))


def convert_csv_to_excel(args: dict[str, Any], **_kw: Any) -> str:
    argv: list[str] = []
    if _text(args, "input"):
        _append(argv, "--input", args.get("input"))
    inputs = _items(args, "inputs")
    if inputs:
        argv.extend(["--inputs", *inputs])
    _append(argv, "--output", args.get("output"))
    _append(argv, "--encoding", args.get("encoding"))
    _append(argv, "--sep", args.get("sep"))
    _append(argv, "--sheet-name", args.get("sheet_name"))
    return _run("csv_to_excel.py", argv, _text(args, "output"))


def filter_excel_rows(args: dict[str, Any], **_kw: Any) -> str:
    argv: list[str] = []
    _append(argv, "--input", args.get("input"))
    _append(argv, "--output", args.get("output"))
    _append(argv, "--sheet", args.get("sheet"))
    for where in _items(args, "where"):
        argv.extend(["--where", where])
    _append(argv, "--encoding", args.get("encoding"))
    return _run("filter_excel.py", argv, _text(args, "output"))


def split_excel_file(args: dict[str, Any], **_kw: Any) -> str:
    argv: list[str] = []
    _append(argv, "--input", args.get("input"))
    _append(argv, "--sheet", args.get("sheet"))
    _append(argv, "--by-rows", args.get("by_rows"))
    _append(argv, "--by-column", args.get("by_column"))
    _append(argv, "--output-dir", args.get("output_dir"))
    _append(argv, "--prefix", args.get("prefix"))
    return _run("split_excel.py", argv, _text(args, "output_dir") or ".")


def deduplicate_excel_rows(args: dict[str, Any], **_kw: Any) -> str:
    argv: list[str] = []
    _append(argv, "--input", args.get("input"))
    _append(argv, "--output", args.get("output"))
    _append(argv, "--sheet", args.get("sheet"))
    _append(argv, "--keys", args.get("keys"))
    _append(argv, "--keep", args.get("keep"))
    _append(argv, "--encoding", args.get("encoding"))
    return _run("deduplicate_excel.py", argv, _text(args, "output"))


def aggregate_excel_rows(args: dict[str, Any], **_kw: Any) -> str:
    argv: list[str] = []
    _append(argv, "--input", args.get("input"))
    _append(argv, "--output", args.get("output"))
    _append(argv, "--sheet", args.get("sheet"))
    _append(argv, "--group-by", args.get("group_by"))
    for agg in _items(args, "agg"):
        argv.extend(["--agg", agg])
    return _run("aggregate_excel.py", argv, _text(args, "output"))


def validate_excel_file(args: dict[str, Any], **_kw: Any) -> str:
    argv: list[str] = []
    _append(argv, "--input", args.get("input"))
    _append(argv, "--sheet", args.get("sheet"))
    _append(argv, "--key-cols", args.get("key_cols"))
    _append(argv, "--require-cols", args.get("require_cols"))
    _append(argv, "--encoding", args.get("encoding"))
    _append(argv, "--header-row", args.get("header_row"))
    return _run("validate_excel.py", argv)


def select_excel_columns(args: dict[str, Any], **_kw: Any) -> str:
    argv: list[str] = []
    _append(argv, "--input", args.get("input"))
    _append(argv, "--output", args.get("output"))
    _append(argv, "--sheet", args.get("sheet"))
    _append(argv, "--columns", args.get("columns"))
    _append(argv, "--rename", args.get("rename"))
    _append(argv, "--encoding", args.get("encoding"))
    return _run("select_columns.py", argv, _text(args, "output"))


def merge_excel_tables(args: dict[str, Any], **_kw: Any) -> str:
    argv: list[str] = []
    _append(argv, "--left", args.get("left"))
    _append(argv, "--right", args.get("right"))
    _append(argv, "--on", args.get("on"))
    _append(argv, "--output", args.get("output"))
    _append(argv, "--sheet-left", args.get("sheet_left"))
    _append(argv, "--sheet-right", args.get("sheet_right"))
    _append(argv, "--how", args.get("how"))
    _append(argv, "--suffixes", args.get("suffixes"))
    return _run("merge_tables.py", argv, _text(args, "output"))


def vlookup_excel_tables(args: dict[str, Any], **_kw: Any) -> str:
    argv: list[str] = []
    _append(argv, "--main", args.get("main"))
    lookups = _items(args, "lookups")
    if lookups:
        argv.extend(["--lookups", *lookups])
    _append(argv, "--output", args.get("output"))
    _append(argv, "--sheet-main", args.get("sheet_main"))
    _append(argv, "--suffix", args.get("suffix"))
    return _run("vlookup_multi.py", argv, _text(args, "output"))


def transpose_excel_table(args: dict[str, Any], **_kw: Any) -> str:
    argv: list[str] = []
    _append(argv, "--input", args.get("input"))
    _append(argv, "--output", args.get("output"))
    _append(argv, "--sheet", args.get("sheet"))
    _append_bool(argv, "--first-col-as-header", args.get("first_col_as_header"))
    return _run("transpose_excel.py", argv, _text(args, "output"))


def fill_excel_template(args: dict[str, Any], **_kw: Any) -> str:
    argv: list[str] = []
    _append(argv, "--template", args.get("template"))
    _append(argv, "--data", args.get("data"))
    _append(argv, "--output", args.get("output"))
    _append(argv, "--sheet-template", args.get("sheet_template"))
    _append(argv, "--sheet-data", args.get("sheet_data"))
    _append(argv, "--pattern-row", args.get("pattern_row"))
    _append(argv, "--header-row", args.get("header_row"))
    return _run("template_fill.py", argv, _text(args, "output"))


def rename_excel_sheets(args: dict[str, Any], **_kw: Any) -> str:
    argv: list[str] = []
    _append(argv, "--input", args.get("input"))
    _append(argv, "--output", args.get("output"))
    for rename in _items(args, "rename"):
        argv.extend(["--rename", rename])
    _append(argv, "--prefix", args.get("prefix"))
    _append(argv, "--suffix", args.get("suffix"))
    return _run("rename_sheets.py", argv, _text(args, "output") or _text(args, "input"))


def format_excel_conditional(args: dict[str, Any], **_kw: Any) -> str:
    argv: list[str] = []
    _append(argv, "--input", args.get("input"))
    _append(argv, "--output", args.get("output"))
    _append(argv, "--sheet", args.get("sheet"))
    _append(argv, "--range", args.get("range"))
    _append(argv, "--column", args.get("column"))
    _append(argv, "--rule", args.get("rule"))
    _append(argv, "--value", args.get("value"))
    _append(argv, "--min", args.get("min"))
    _append(argv, "--max", args.get("max"))
    _append(argv, "--fill", args.get("fill"))
    _append_bool(argv, "--color-scale", args.get("color_scale"))
    return _run("format_conditional.py", argv, _text(args, "output") or _text(args, "input"))


def format_excel_columns_as_text(args: dict[str, Any], **_kw: Any) -> str:
    argv: list[str] = []
    _append(argv, "--input", args.get("input"))
    _append(argv, "--output", args.get("output"))
    _append(argv, "--sheet", args.get("sheet"))
    _append(argv, "--columns", args.get("columns"))
    _append(argv, "--header-row", args.get("header_row"))
    return _run("format_columns_as_text.py", argv, _text(args, "output") or _text(args, "input"))
