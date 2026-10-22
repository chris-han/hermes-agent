from __future__ import annotations

from typing import Protocol

from contracts.tender_sensitive_term_detection_modeling_core import GovernedRuleVersionDTO


class GovernedRuleVersionService(Protocol):
    def get_rule_version(
        self,
        rule_version_id: str,
        organization_id: str,
    ) -> GovernedRuleVersionDTO: ...

    def list_rule_versions_by_ids(
        self,
        rule_version_ids: list[str],
        organization_id: str,
    ) -> list[GovernedRuleVersionDTO]: ...

    def verify_rule_version_content_hash(
        self,
        rule_version_id: str,
        expected_hash: str,
    ) -> bool: ...

    def assert_rule_versions_owned_by_organization(
        self,
        rule_version_ids: list[str],
        organization_id: str,
    ) -> None: ...
