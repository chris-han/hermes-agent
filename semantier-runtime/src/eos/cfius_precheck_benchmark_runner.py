from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eos.artifact_hashing import content_hash


def _count_cases_with_tag(cases: list[dict[str, Any]], tag: str) -> int:
    return sum(tag in case.get("coverage_tags", []) for case in cases)


def _count_cases_with_any_tag(cases: list[dict[str, Any]], tags: tuple[str, ...]) -> int:
    return sum(any(tag in case.get("coverage_tags", []) for tag in tags) for case in cases)


def load_cfius_benchmark_corpus(cases_path: str | Path) -> list[dict[str, Any]]:
    cases = json.loads(Path(cases_path).read_text())
    if not isinstance(cases, list):
        raise ValueError("invalid_cfius_benchmark_case_list")
    normalized: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("invalid_cfius_benchmark_case")
        if "fixture_id" not in case:
            raise ValueError("invalid_cfius_benchmark_case")
        case.setdefault("coverage_tags", [])
        normalized.append(case)
    return normalized


def run_cfius_precheck_benchmark(*, manifest_path: str | Path, cases_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text())
    cases = load_cfius_benchmark_corpus(cases_path)

    manifest_body = dict(manifest)
    manifest_ref = manifest_body.pop("benchmark_ref", None)
    manifest_hash = manifest_body.pop("benchmark_hash", None)
    expected_hash = "sha256:" + content_hash(manifest_body)
    if manifest_hash != expected_hash or manifest_ref != f"benchmark:{expected_hash[7:]}":
        raise ValueError("cfius benchmark manifest hash mismatch")

    fixture_ids = tuple(case["fixture_id"] for case in cases)
    if len(fixture_ids) != len(set(fixture_ids)):
        raise ValueError("cfius benchmark fixture ids must be unique")
    if tuple(manifest.get("primary_fixture_ids", ())) != fixture_ids:
        raise ValueError("cfius benchmark manifest and corpus fixture ids must match exactly")

    report = {
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_ref": manifest_ref,
        "benchmark_hash": manifest_hash,
        "fixture_count": len(cases),
        "primary_fixture_ids": fixture_ids,
        "denominators": {
            "part_800_or_tid": _count_cases_with_any_tag(cases, ("part_800", "tid")),
            "mandatory_filing": _count_cases_with_tag(cases, "mandatory_filing"),
            "part_802": _count_cases_with_tag(cases, "part_802"),
            "evidence_gap_or_conflict": _count_cases_with_any_tag(
                cases, ("evidence_gap", "evidence_conflict", "conflict")
            ),
            "confidentiality": _count_cases_with_tag(cases, "confidentiality"),
            "matter_isolation": _count_cases_with_tag(cases, "matter_isolation"),
            "replay_tamper": _count_cases_with_tag(cases, "replay_tamper"),
            "hybrid": _count_cases_with_tag(cases, "hybrid"),
            "deterministic": _count_cases_with_tag(cases, "deterministic"),
            "interpreted": _count_cases_with_tag(cases, "interpreted"),
            "unknown": _count_cases_with_tag(cases, "unknown"),
        },
        "coverage_gate_status": {
            "part_800_or_tid": _count_cases_with_any_tag(cases, ("part_800", "tid")) >= 6,
            "mandatory_filing": _count_cases_with_tag(cases, "mandatory_filing") >= 4,
            "part_802": _count_cases_with_tag(cases, "part_802") >= 6,
            "evidence_gap_or_conflict": _count_cases_with_any_tag(
                cases, ("evidence_gap", "evidence_conflict", "conflict")
            ) >= 3,
            "confidentiality": _count_cases_with_tag(cases, "confidentiality") >= 1,
            "matter_isolation": _count_cases_with_tag(cases, "matter_isolation") >= 1,
            "replay_tamper": _count_cases_with_tag(cases, "replay_tamper") >= 1,
            "hybrid": _count_cases_with_tag(cases, "hybrid") >= 6,
            "deterministic": _count_cases_with_tag(cases, "deterministic") >= 6,
            "interpreted": _count_cases_with_tag(cases, "interpreted") >= 4,
            "unknown": _count_cases_with_tag(cases, "unknown") >= 4,
        },
    }
    if output_path is not None:
        Path(output_path).write_text(json.dumps(report, indent=2, sort_keys=True))
    return report