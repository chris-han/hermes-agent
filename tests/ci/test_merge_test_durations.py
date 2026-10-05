"""The duration cache must retain every slice's healthy update or fail closed."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "merge_test_durations.py"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _run(root: Path, count: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline", str(root / "baseline.json"),
            "--shards", str(root / "shards"),
            "--slice-count", str(count),
            "--output", str(root / "merged.json"),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
def test_preserves_all_updates_and_unchanged_baseline(tmp_path, order):
    baseline = {"slow.py": 80.0, "other.py": 2.0, "unchanged.py": 7.0}
    changes = [{"slow.py": 120.0}, {"other.py": 1.0}, {"new.py": 4.0}]
    _write_json(tmp_path / "baseline.json", baseline)
    for index, change_index in enumerate(order, start=1):
        _write_json(
            tmp_path / "shards" / f"test-durations-shard-{index}" / "test_durations.json",
            {**baseline, **changes[change_index]},
        )

    result = _run(tmp_path, len(changes))

    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "merged.json").read_text()) == {
        **baseline, "slow.py": 120.0, "other.py": 1.0, "new.py": 4.0,
    }
    # A selected-job rerun replaces only that slice's artifact. The merge
    # must keep the other successful slices while taking the replacement.
    rerun_slice = order.index(0) + 1
    _write_json(
        tmp_path / "shards" / f"test-durations-shard-{rerun_slice}" / "test_durations.json",
        {**baseline, "slow.py": 160.0},
    )
    rerun = _run(tmp_path, len(changes))
    assert rerun.returncode == 0, rerun.stderr
    assert json.loads((tmp_path / "merged.json").read_text()) == {
        **baseline, "slow.py": 160.0, "other.py": 1.0, "new.py": 4.0,
    }
    assert json.loads((tmp_path / "baseline.json").read_text()) == baseline


@pytest.mark.parametrize("problem", ["missing", "conflict", "unexpected", "invalid"])
def test_rejects_incomplete_or_conflicting_cache_without_replacing_output(tmp_path, problem):
    baseline = {"slow.py": 80.0}
    _write_json(tmp_path / "baseline.json", baseline)
    _write_json(tmp_path / "merged.json", {"previous.py": 5.0})
    first = {**baseline, "slow.py": 120.0}
    second = {**baseline, "new.py": 3.0}
    if problem == "conflict":
        second["slow.py"] = 140.0
    if problem == "invalid":
        second["new.py"] = -1.0
    _write_json(tmp_path / "shards" / "test-durations-shard-1" / "test_durations.json", first)
    if problem != "missing":
        _write_json(
            tmp_path / "shards" / "test-durations-shard-2" / "test_durations.json", second
        )
    if problem == "unexpected":
        _write_json(
            tmp_path / "shards" / "test-durations-shard-3" / "test_durations.json", baseline
        )

    result = _run(tmp_path, 2)

    assert result.returncode != 0
    assert "error:" in result.stderr
    assert json.loads((tmp_path / "merged.json").read_text()) == {"previous.py": 5.0}
