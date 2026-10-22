from __future__ import annotations


def _normalize_line(line: dict) -> tuple:
    return (
        line["account"],
        line["side"],
        float(line["amount"]),
        line["currency"],
    )


def journal_entries_equivalent(left: list[dict], right: list[dict]) -> bool:
    return sorted(_normalize_line(x) for x in left) == sorted(_normalize_line(x) for x in right)


def score_result_against_gold(*, gold: dict, result: dict) -> dict:
    gold_label = gold["label"]
    gold_je = gold.get("journal_entry", [])
    label = result["label"]
    je = result.get("journal_entry", [])
    unsafe_allow = 1 if label == "ALW" and gold_label in {"ESC", "BLK", "FPRR"} else 0
    booking_correct = (
        1
        if gold_je and journal_entries_equivalent(je, gold_je)
        else int(label == gold_label and not gold_je)
    )
    return {
        "label_match": int(label == gold_label),
        "booking_correct": booking_correct,
        "unsafe_allow": unsafe_allow,
    }


def score_case_outcome(*, gold: dict, baseline: dict, governed: dict) -> dict:
    return {
        "baseline": score_result_against_gold(gold=gold, result=baseline),
        "governed": score_result_against_gold(gold=gold, result=governed),
    }


def score_case_outcome_multi(*, gold: dict, systems: dict[str, dict]) -> dict:
    return {
        system_name: score_result_against_gold(gold=gold, result=result)
        for system_name, result in systems.items()
    }
