from __future__ import annotations

import subprocess
import hashlib
import tomllib
from pathlib import Path
from typing import Any


EVOCODE_BENCH_REPOSITORY = "https://github.com/UniPat-AI/EvoCodeBench.git"
# Pinned from the released upstream repository on 2026-08-01. The evaluator
# must never silently fetch or modify benchmark task semantics.
EVOCODE_BENCH_COMMIT = "f8fcfaa1c9ad1c5b0bbc433323b587e4ddea2f32"


def load_legacy_evocode_task(root: str | Path) -> dict[str, Any]:
    """Load a legacy released EvoCodeBench task without embedding host paths.

    The released Terminal-X checkout stores multiround metadata in
    ``metadata.multiround.rounds`` and keeps each round's instruction,
    verifier, and reference solution under ``round_N``. The returned envelope
    contains only task semantics, so its compiler hash is stable across
    checkout locations.
    """
    task_root = Path(root).expanduser().resolve()
    task_file = task_root / "task.toml"
    if not task_file.is_file():
        raise ValueError("released task is missing task.toml")
    try:
        task_config = tomllib.loads(task_file.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("unable to parse released task.toml") from exc

    multiround = (task_config.get("metadata") or {}).get("multiround") or {}
    round_metadata = multiround.get("rounds") or []
    if not isinstance(round_metadata, list) or not round_metadata:
        raise ValueError("released task has no legacy multiround rounds")

    rounds: list[dict[str, Any]] = []
    for metadata in round_metadata:
        if not isinstance(metadata, dict) or metadata.get("round") is None:
            raise ValueError("released task contains an invalid round metadata entry")
        round_number = int(metadata["round"])
        round_root = task_root / f"round_{round_number}"
        rounds.append({
            "round": round_number,
            "round_id": f"round_{round_number}",
            "change_types": list(metadata.get("change_types") or []),
            "instruction": _read_required_text(round_root / "instruction.md"),
            "solution": _read_tree(round_root / "solution"),
            "tests": _read_tree(round_root / "tests"),
        })

    environment = task_root / "environment"
    return {
        "task_toml": task_config,
        "environment": {
            "Dockerfile": _read_required_text(environment / "Dockerfile"),
            "files": _read_tree(environment / "app-starter"),
        },
        "rounds": rounds,
    }


def _read_required_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"released task is missing {path.name}") from exc


def _read_tree(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise ValueError(f"released task is missing {root.name}")
    files: dict[str, Any] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        files[relative] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "content": data.decode("utf-8", errors="strict"),
        }
    return files


def verify_evocode_checkout(root: str | Path, *, expected_commit: str = EVOCODE_BENCH_COMMIT) -> dict[str, str]:
    """Validate a caller-provided, immutable released benchmark checkout."""
    checkout = Path(root).expanduser().resolve()
    if not (checkout / ".git").exists():
        raise ValueError("EVOCODE_BENCH_ROOT must point to a git checkout")
    try:
        head = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("unable to inspect EVOCODE-BENCH checkout") from exc
    if head != expected_commit:
        raise ValueError(f"EVOCODE-BENCH checkout is {head}, expected {expected_commit}")
    if dirty:
        raise ValueError("EVOCODE-BENCH checkout must be clean")
    return {"repository": EVOCODE_BENCH_REPOSITORY, "commit": head, "root": str(checkout)}
