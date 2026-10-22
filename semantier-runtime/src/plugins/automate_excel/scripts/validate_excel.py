#!/usr/bin/env python3
"""
校验 Excel 表：检查必须列、空行、重复键等。
用法:
  python validate_excel.py --input data.xlsx
  python validate_excel.py --input data.csv --encoding gb18030 --header-row 2
  python validate_excel.py --input data.xlsx --require-cols 名称,金额 --key-cols 编号
"""
import argparse
import csv
import sys
from pathlib import Path

import pandas as pd


def _excel_engine(path):
    return "xlrd" if Path(path).suffix.lower() == ".xls" else "openpyxl"


def _header_arg(value):
    if value == "":
        return 0
    row = int(value)
    if row <= 0:
        return None
    return row - 1


def _read_csv_with_standard_reader(path, *, encoding, header_row):
    rows = list(csv.reader(path.read_text(encoding=encoding).splitlines(), delimiter=","))
    header_index = max(int(header_row) - 1, 0)
    if header_index >= len(rows):
        return pd.DataFrame()
    headers = [str(value).strip() or f"column_{index + 1}" for index, value in enumerate(rows[header_index])]
    records = []
    for row in rows[header_index + 1 :]:
        padded = [str(value).strip() for value in row[: len(headers)]]
        if len(padded) < len(headers):
            padded.extend([""] * (len(headers) - len(padded)))
        records.append(padded)
    return pd.DataFrame(records, columns=headers)


def _read_table(path, args):
    header = _header_arg(args.header_row)
    if path.suffix.lower() == ".csv":
        header_row = int(args.header_row) if str(args.header_row).strip() else 1
        skiprows = range(header_row - 1) if header_row > 1 else None
        csv_kwargs = {
            "encoding": args.encoding,
            "header": None if header_row <= 0 else 0,
            "skiprows": skiprows,
            "sep": None,
            "engine": "python",
        }
        try:
            return pd.read_csv(path, **csv_kwargs)
        except Exception:
            return _read_csv_with_standard_reader(
                path,
                encoding=args.encoding,
                header_row=header_row,
            )
    return pd.read_excel(
        path,
        sheet_name=args.sheet,
        engine=_excel_engine(path),
        header=header,
    )


def main():
    parser = argparse.ArgumentParser(description="校验 Excel 表结构与数据")
    parser.add_argument("--input", "-i", required=True, help="输入 .xlsx/.xls/.csv 文件")
    parser.add_argument("--sheet", default=0, help="工作表名或索引")
    parser.add_argument("--key-cols", default="", help="用于检查重复的列名，逗号分隔")
    parser.add_argument("--require-cols", default="", help="必须存在的列名，逗号分隔")
    parser.add_argument("--encoding", default="utf-8", help="CSV 文件编码")
    parser.add_argument("--header-row", default="", help="表头行号，1 基；0 表示无表头")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"文件不存在: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = _read_table(path, args)
    except Exception as e:
        print(f"读取失败: {e}", file=sys.stderr)
        sys.exit(1)

    errors = []

    if args.require_cols:
        required = [c.strip() for c in args.require_cols.split(",") if c.strip()]
        missing = [c for c in required if c not in df.columns]
        if missing:
            errors.append(f"缺少必须列: {missing}")

    if args.key_cols:
        keys = [c.strip() for c in args.key_cols.split(",") if c.strip()]
        key_missing = [c for c in keys if c not in df.columns]
        if key_missing:
            errors.append(f"键列不存在: {key_missing}")
        else:
            dup = df[df.duplicated(subset=keys, keep=False)]
            if not dup.empty:
                errors.append(f"存在重复键，共 {len(dup)} 行")

    # 全空行
    empty = df.dropna(how="all")
    if len(empty) < len(df):
        errors.append(f"存在 {len(df) - len(empty)} 行全空行")

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)
    print("校验通过")
    sys.exit(0)


if __name__ == "__main__":
    main()
