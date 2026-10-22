from __future__ import annotations

from typing import Literal, TypedDict


OutcomeLabel = Literal["ALW", "AWC", "ESC", "BLK", "FPRR"]
ScenarioType = Literal[
    "positive",
    "correctable_negative",
    "evidence_gap",
    "policy_conflict",
    "projection_rejection",
    "replay",
]


class JournalLine(TypedDict):
    account: str
    side: Literal["DEBIT", "CREDIT"]
    amount: float
    currency: str


class GoldExpectation(TypedDict, total=False):
    label: OutcomeLabel
    journal_entry: list[JournalLine]
    expected_rejected_projection: bool
    expected_policy_ids: list[str]
    expected_evidence_refs: list[str]


class BenchmarkCasePacket(TypedDict):
    case_id: str
    scenario_type: ScenarioType
    org_id: str
    rea_fact_packet: dict
    case_context: dict
    gold: GoldExpectation

