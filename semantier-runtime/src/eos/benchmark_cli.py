from __future__ import annotations

import json
import uuid
from pathlib import Path

from eos.benchmark_baseline import load_baseline_families
from eos.benchmark_case_store import load_case_packets, load_v2_case_packets
from eos.benchmark_governed_runner import run_governed_case
from eos.benchmark_metrics import score_case_outcome_multi
from eos.benchmark_report import summarize_scores


def run_results_bundle(
    output_path: str,
    case_path: str | None = None,
    v2_cases_dir: str | None = None,
    baseline_path: str | None = None,
    secondary_baseline_path: str | None = None,
) -> dict:
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    if bool(case_path) == bool(v2_cases_dir):
        raise ValueError("exactly_one_of_case_path_or_v2_cases_dir_required")

    if case_path is not None:
        cases = load_case_packets(case_path)
    else:
        cases = load_v2_case_packets(v2_cases_dir)

    baseline_families = (
        load_baseline_families(baseline_path, secondary_baseline_path)
        if baseline_path is not None
        else {}
    )
    scored_cases = []
    governed_runs = []

    for case in cases:
        governed = run_governed_case(case, run_id=run_id)
        governed_runs.append(governed)
        systems = {
            family_name: family_rows[case["case_id"]]
            for family_name, family_rows in baseline_families.items()
        }
        systems["governed"] = {
            "label": governed["governed_label"],
            "journal_entry": governed["governed_journal_entry"],
        }
        scored = score_case_outcome_multi(
            gold=case["gold"],
            systems=systems,
        )
        scored["case_id"] = case["case_id"]
        scored["scenario_type"] = case["scenario_type"]
        scored_cases.append(scored)

    summary = summarize_scores(scored_cases)
    payload = {
        "run_id": run_id,
        "input_mode": "legacy_json" if case_path is not None else "v2_yaml_dir",
        "summary": summary,
        "governed_runs": governed_runs,
        "scored_cases": scored_cases,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload
