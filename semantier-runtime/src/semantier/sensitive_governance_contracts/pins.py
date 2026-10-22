from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SensitiveGovernancePins:
    contract_version: str
    ontology_version: str
    authority_bundle_version: str
    policy_bundle_version: str
    display_profile_version: str
    fixture_manifest_hash: str
    decision_committed_event_id: str | None = None
    execution_prepared_event_id: str | None = None
    execution_terminal_event_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
