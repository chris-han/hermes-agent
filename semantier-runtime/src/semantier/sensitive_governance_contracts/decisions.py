from __future__ import annotations

from dataclasses import asdict, dataclass

from semantier.sensitive_governance_contracts.enums import DisplayAction, DisclosureAction


@dataclass(frozen=True)
class AuthorityReferenceSet:
    authority_set_id: str
    tier_refs: list[dict]
    authority_hash: str


@dataclass(frozen=True)
class DisclosureDecision:
    decision_id: str
    action: DisclosureAction
    policy_ref: str
    reason_codes: list[str]
    authority: AuthorityReferenceSet

    def to_dict(self) -> dict:
        result = asdict(self)
        result["action"] = self.action.value
        return result


@dataclass(frozen=True)
class DisplayDecision:
    decision_id: str
    action: DisplayAction
    policy_ref: str
    reason_codes: list[str]
    authority: AuthorityReferenceSet

    def to_dict(self) -> dict:
        result = asdict(self)
        result["action"] = self.action.value
        return result
