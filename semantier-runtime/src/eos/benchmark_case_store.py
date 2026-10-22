from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml  # PyYAML — already a test dependency

from eos.benchmark_types import BenchmarkCasePacket

# ---------------------------------------------------------------------------
# V2 YAML loader
# ---------------------------------------------------------------------------

_V2_SCENARIO_MAP = {
    "positive": "positive",
    "correctable_negative": "correctable_negative",
    "evidence_gap": "evidence_gap",
    "policy_conflict": "policy_conflict",
    "projection_rejection": "projection_rejection",
    "replay": "replay",
}


def _v2_yaml_to_packet(doc: dict[str, Any]) -> BenchmarkCasePacket:
    """Convert a v2 benchmark YAML document to a BenchmarkCasePacket.

    The v2 YAML schema is richer than BenchmarkCasePacket: it carries source
    provenance, governed_inputs, and verbatim raw_fields.  Only the fields
    required by the runner are projected; the rest are preserved under
    ``case_context["v2_metadata"]`` so they remain accessible without
    polluting the top-level contract.
    """
    case_id: str = doc["case_id"]
    rfp = doc["rea_fact_packet"]
    event_block = rfp["event"]
    gi = doc.get("governed_inputs", {})
    gold_label: str = doc["gold_outcome"]

    # Build a runner-compatible rea_fact_packet: add synthetic event_id and
    # flatten the structured sub-blocks into a flat dict.
    flat_rfp: dict[str, Any] = {
        "event_id": f"v2::{case_id}",
        "event_type": event_block.get("type", "unknown"),
        "date": event_block.get("date"),
        "document_ref": event_block.get("document_ref"),
        "seller_nif": event_block.get("seller_nif"),
        "settlement_status": event_block.get("settlement_status", "unknown"),
        "resource_description": rfp.get("resource", {}).get("description"),
        "amount_gross": rfp.get("resource", {}).get("amount_gross"),
        "currency": rfp.get("resource", {}).get("currency"),
    }

    # Build gold expectation: ALW/AWC cases have a JE; others do not.
    gold_je_block = doc.get("gold_je")
    journal_entry: list[dict] = []
    if gold_je_block and gold_label in ("ALW", "AWC"):
        debit = gold_je_block.get("debit_account", "")
        credit = gold_je_block.get("credit_account", "")
        raw_amount = gold_je_block.get("amount", "0")
        try:
            amount = float(str(raw_amount).replace(",", "."))
        except (ValueError, TypeError):
            amount = 0.0
        currency = rfp.get("resource", {}).get("currency") or "EUR"
        journal_entry = [
            {"account": debit, "side": "DEBIT", "amount": amount, "currency": currency},
            {"account": credit, "side": "CREDIT", "amount": amount, "currency": currency},
        ]

    gold: dict[str, Any] = {"label": gold_label}
    if journal_entry:
        gold["journal_entry"] = journal_entry
    if gi.get("policy_ids"):
        gold["expected_policy_ids"] = gi["policy_ids"]

    evidence_refs: list[str] = []
    source = doc.get("source", {})
    if source.get("artifact_ref"):
        evidence_refs.append(source["artifact_ref"])

    packet: BenchmarkCasePacket = {
        "case_id": case_id,
        "scenario_type": _V2_SCENARIO_MAP[doc["scenario_class"]],
        "org_id": "org_benchmark_v2",
        "rea_fact_packet": flat_rfp,
        "case_context": {
            "authority_domain": "accounting",
            "policy_ids": gi.get("policy_ids", []),
            "evidence_refs": evidence_refs,
            "v2_metadata": {
                "source_id": source.get("source_id"),
                "artifact_ref": source.get("artifact_ref"),
                "label_deterministic": doc.get("label_deterministic"),
                "annotation_date": str(doc.get("annotation_date", "")),
                "governed_inputs": gi,
            },
        },
        "gold": gold,
    }
    return packet


def load_v2_case_packets(cases_dir: str | Path) -> list[BenchmarkCasePacket]:
    """Load all ``*.yaml`` files in *cases_dir* and return them as BenchmarkCasePackets."""
    cases_dir = Path(cases_dir)
    packets: list[BenchmarkCasePacket] = []
    for path in sorted(cases_dir.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        packets.append(_v2_yaml_to_packet(doc))
    return packets


# ---------------------------------------------------------------------------
# Legacy JSON loader (unchanged)
# ---------------------------------------------------------------------------


def load_case_packets(path: str | Path) -> list[BenchmarkCasePacket]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError("invalid_benchmark_case_packet_list")

    cases: list[BenchmarkCasePacket] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("invalid_benchmark_case_packet")
        if "case_id" not in item or "gold" not in item:
            raise ValueError("invalid_benchmark_case_packet")
        item.setdefault("case_context", {})
        cases.append(item)
    return cases

