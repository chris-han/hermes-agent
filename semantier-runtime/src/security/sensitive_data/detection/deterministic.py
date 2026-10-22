from __future__ import annotations

import re
from dataclasses import dataclass

from semantier.sensitive_governance_contracts.enums import FindingClass
from semantier.sensitive_governance_contracts.hashing import canonical_hash


@dataclass(frozen=True)
class DetectionFinding:
    finding_id: str
    finding_class: FindingClass
    start: int
    end: int
    detector_id: str
    evidence_ref: str


@dataclass(frozen=True)
class _PatternRule:
    detector_id: str
    finding_class: FindingClass
    pattern: re.Pattern[str]


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


_RULES: tuple[_PatternRule, ...] = (
    _PatternRule(
        "cn_mobile_v1",
        FindingClass.PERSONAL_INFORMATION,
        _compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    ),
    _PatternRule(
        "cn_mobile_hyphenated_v1",
        FindingClass.PERSONAL_INFORMATION,
        _compile(r"(?<![\d-])1[3-9]\d-\d{4}-\d{4}(?![\d-])"),
    ),
    _PatternRule(
        "health_lexical_candidate_v1",
        FindingClass.SENSITIVE_PERSONAL_INFORMATION,
        _compile(r"哮喘"),
    ),
    _PatternRule(
        "cn_id_number_v1",
        FindingClass.SENSITIVE_PERSONAL_INFORMATION,
        _compile(r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"),
    ),
    _PatternRule(
        "cvv_cvc_fixture_v1",
        FindingClass.CREDENTIAL_OR_SECRET,
        _compile(r"\b(?:CVV|CVC)\s*[:：=]\s*\d{3,4}\b"),
    ),
    _PatternRule(
        "api_key_fixture_v1",
        FindingClass.CREDENTIAL_OR_SECRET,
        _compile(r"\b(?:API[_ -]?KEY|TOKEN|SECRET|PASSWORD|密码)\s*[:：=]\s*[A-Za-z0-9_\-]{6,}\b|sk-[A-Za-z0-9_\-]{8,}"),
    ),
    _PatternRule(
        "private_key_fixture_v1",
        FindingClass.CREDENTIAL_OR_SECRET,
        _compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ),
    _PatternRule(
        "expert_identity_context_v1",
        FindingClass.TENDER_EXPERT_INFORMATION,
        _compile(r"(?:评标专家|专家)\s*[:：]?\s*[\u4e00-\u9fff]{2,4}(?:\s*手机\s*1[3-9]\d{9})?"),
    ),
    _PatternRule(
        "unpublished_budget_v1",
        FindingClass.TENDER_SENSITIVE_INFORMATION,
        _compile(r"(?:未公开)?(?:预算|控制价|最高限价|标底)\s*[:：为是]?\s*[\d,.]+(?:万|万元|元|人民币)?"),
    ),
    _PatternRule(
        "supplier_quote_v1",
        FindingClass.BIDDER_CONFIDENTIAL_INFORMATION,
        _compile(r"(?:供应商|投标人|报价)\s*[\u4e00-\u9fffA-Za-z0-9（）()]*\s*(?:报价|投标价)?\s*[:：为是]?\s*[\d,.]+(?:万|万元|元|人民币)?"),
    ),
    _PatternRule(
        "internal_score_v1",
        FindingClass.TENDER_SENSITIVE_INFORMATION,
        _compile(r"(?:内部评分|评审分|技术分|商务分)\s*[:：为是]?\s*\d+(?:\.\d+)?"),
    ),
    _PatternRule(
        "ordinary_commercial_secret_marking_v1",
        FindingClass.ORDINARY_COMMERCIAL_SECRET,
        _compile(r"普通商密"),
    ),
    _PatternRule(
        "core_commercial_secret_marking_v1",
        FindingClass.CORE_COMMERCIAL_SECRET,
        _compile(r"核心商密"),
    ),
    _PatternRule(
        "existing_state_secret_marking_v1",
        FindingClass.EXISTING_STATE_SECRET_MARKING,
        _compile(r"(?:秘密★|机密★|绝密★|(?<!疑似)国家秘密|密级\s*[:：]\s*(?:秘密|机密|绝密))"),
    ),
    _PatternRule(
        "suspected_state_secret_v1",
        FindingClass.SUSPECTED_STATE_SECRET,
        _compile(r"(?:疑似国家秘密|涉密待定|疑似涉密)"),
    ),
)


class SensitiveDataDetector:
    detector_version = "repo_deterministic_sensitive_detector_v3"

    def detect(self, text: str) -> list[DetectionFinding]:
        findings: list[DetectionFinding] = []
        seen: set[tuple[int, int, FindingClass]] = set()
        for rule in _RULES:
            for match in rule.pattern.finditer(text):
                key = (match.start(), match.end(), rule.finding_class)
                if key in seen:
                    continue
                seen.add(key)
                finding_id = "sg_finding_" + canonical_hash(
                    {
                        "detector_id": rule.detector_id,
                        "class": rule.finding_class.value,
                        "start": match.start(),
                        "end": match.end(),
                    }
                )[:24]
                findings.append(
                    DetectionFinding(
                        finding_id=finding_id,
                        finding_class=rule.finding_class,
                        start=match.start(),
                        end=match.end(),
                        detector_id=rule.detector_id,
                        evidence_ref=f"text_range:{match.start()}:{match.end()}",
                    )
                )
        # Source-local Markdown procurement tables: exact supplier cell only,
        # not a company-name inference across arbitrary narrative or tables.
        for table in re.finditer(r"(?m)^\|([^\n]*)\|\n\|[-: |]+\|\n(?P<rows>(?:\|[^\n]*\|(?:\n|$))+)", text):
            headers = [cell.strip() for cell in table.group(1).split('|')]
            if '供应商' not in headers or not any(cell.startswith('报价') for cell in headers):
                continue
            supplier_column = headers.index('供应商')
            for row in re.finditer(r"(?m)^\|[^\n]*\|", table.group('rows')):
                cells = list(re.finditer(r"(?<=\|)[^|]*(?=\|)", row.group()))
                if supplier_column >= len(cells):
                    continue
                cell = cells[supplier_column]
                value = cell.group().strip()
                if not value or value in {'—', '-'}:
                    continue
                start = table.start('rows') + row.start() + cell.start() + len(cell.group()) - len(cell.group().lstrip())
                end = start + len(value)
                detector_id = 'supplier_markdown_quote_cell_v1'
                finding_class = FindingClass.BIDDER_CONFIDENTIAL_INFORMATION
                finding_id = 'sg_finding_' + canonical_hash({'detector_id': detector_id,
                    'class': finding_class.value, 'start': start, 'end': end})[:24]
                findings.append(DetectionFinding(finding_id=finding_id, finding_class=finding_class,
                    start=start, end=end, detector_id=detector_id, evidence_ref=f'text_range:{start}:{end}'))
        return sorted(findings, key=lambda finding: (finding.start, finding.end, finding.finding_class.value))


class PresidioSensitiveDataDetectorAdapter:
    """Optional provider adapter boundary; provider types do not cross this API."""

    provider_id = "presidio_optional_adapter_v1"

    def __init__(self, provider: object | None = None) -> None:
        self._provider = provider

    def detect(self, text: str) -> list[DetectionFinding]:
        if self._provider is None:
            return SensitiveDataDetector().detect(text)
        raise NotImplementedError("Presidio adapter provider invocation is not enabled for the MVP")
