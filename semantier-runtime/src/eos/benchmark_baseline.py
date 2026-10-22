from __future__ import annotations

import json
from pathlib import Path


def load_baseline_outputs(path: str | Path) -> dict[str, dict]:
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list):
        raise ValueError("invalid_baseline_artifact_list")

    result: dict[str, dict] = {}
    for row in rows:
        required = [
            "case_id",
            "model_id",
            "model_version",
            "prompt_hash",
            "seed",
            "label",
            "journal_entry",
            "evidence_refs",
        ]
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(f"invalid_baseline_artifact:{','.join(missing)}")
        result[row["case_id"]] = row
    return result


def load_baseline_families(
    primary_path: str | Path,
    secondary_path: str | Path | None = None,
) -> dict[str, dict[str, dict]]:
    families = {"baseline": load_baseline_outputs(primary_path)}
    if secondary_path is not None:
        families["baseline_secondary"] = load_baseline_outputs(secondary_path)
    return families
