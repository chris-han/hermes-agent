from __future__ import annotations


def _system_names(scored_cases: list[dict]) -> list[str]:
    first = scored_cases[0]
    return [key for key in first.keys() if key not in {"case_id", "scenario_type"}]


def summarize_scores(scored_cases: list[dict]) -> dict:
    total = len(scored_cases)
    if total == 0:
        raise ValueError("empty_benchmark_result_set")

    systems = _system_names(scored_cases)

    def summarize(system_name: str) -> dict:
        unsafe = sum(case[system_name]["unsafe_allow"] for case in scored_cases) / total
        correct = sum(case[system_name]["booking_correct"] for case in scored_cases) / total
        return {
            "unsafe_allow_rate": unsafe,
            "booking_correct_rate": correct,
        }

    summary = {system_name: summarize(system_name) for system_name in systems}
    summary["case_count"] = total
    summary["by_scenario"] = summarize_by_scenario(scored_cases)
    return summary


def summarize_by_scenario(scored_cases: list[dict]) -> dict:
    systems = _system_names(scored_cases)
    grouped: dict[str, list[dict]] = {}
    for case in scored_cases:
        grouped.setdefault(case["scenario_type"], []).append(case)

    result: dict[str, dict] = {}
    for scenario, items in grouped.items():
        count = len(items)
        result[scenario] = {"case_count": count}
        for system_name in systems:
            result[scenario][system_name] = {
                "unsafe_allow_rate": sum(item[system_name]["unsafe_allow"] for item in items) / count,
                "booking_correct_rate": sum(item[system_name]["booking_correct"] for item in items) / count,
            }
    return result
