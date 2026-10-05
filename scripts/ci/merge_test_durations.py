#!/usr/bin/env python3
"""Merge healthy per-slice timings without overwriting another slice's updates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _read_durations(path: Path) -> dict[str, float]:
    durations = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(durations, dict) or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in durations.values()
    ):
        raise ValueError(f"{path}: expected finite, nonnegative durations")
    return durations


def merge_durations(
    baseline_path: Path, shards_root: Path, slice_count: int
) -> dict[str, float]:
    if slice_count < 1:
        raise ValueError("slice count must be positive")
    baseline = _read_durations(baseline_path)
    expected = {f"test-durations-shard-{index}" for index in range(1, slice_count + 1)}
    actual = {path.name for path in shards_root.glob("test-durations-shard-*")}
    if actual != expected:
        raise ValueError(
            f"incomplete shard set: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )

    updates: dict[str, float] = {}
    for name in sorted(expected):
        shard = _read_durations(shards_root / name / "test_durations.json")
        for file, duration in shard.items():
            # Every shard carries the same old entries as well as its updates.
            if file in baseline and duration == baseline[file]:
                continue
            if file in updates:
                raise ValueError(f"{file}: updated by multiple test slices")
            updates[file] = duration
    return {**baseline, **updates}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--slice-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        durations = merge_durations(args.baseline, args.shards, args.slice_count)
        args.output.write_text(
            json.dumps(durations, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Merged {args.slice_count} slices into {len(durations)} cached durations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
