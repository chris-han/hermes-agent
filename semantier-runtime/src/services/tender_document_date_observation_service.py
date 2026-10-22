"""Extract Tender document dates only from a pinned issuer/agency date field."""
from __future__ import annotations

from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
import re

from contracts.artifact_pin import ArtifactPinV1
from contracts.canonical_source_ir import (
    CanonicalSourceDocument,
    SourceLocator,
    content_hash as source_content_hash,
    source_block_identity,
)
from contracts.evidence_location import SourceIdentity
from contracts.normative_schema import NormativeSchemaContractV1
from contracts.tender_document_date_observation import TenderDocumentDateObservationV1
from eos.artifact_hashing import content_hash


_DATE_LABEL = re.compile(r"^\s*日期\s*[:：]\s*(.*?)\s*$")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_CHINESE_DATE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_SELECTOR_CONTRACT = {
    "selector": "tender_issuer_agency_attestation_date.v1",
    "source_format": "pdf",
    "native_provider": "pymupdf",
    "required_ordered_fields": ["招标人", "招标代理机构", "日期"],
    "field_rule": "one explicit 日期 field after both issuer and agency fields on the same source page",
    "ambiguous_matches": "UNKNOWN",
    "partial_or_missing_date": "UNKNOWN",
}
_NORMALIZER_CONTRACT = {
    "normalizer": "tender_document_date_normalizer.v1",
    "accepted_full_date_grammars": ["YYYY-MM-DD", "YYYY年M月D日"],
    "invalid_or_partial_values": "UNKNOWN",
}


def _sha256(data: bytes) -> str:
    return "sha256:" + sha256(data).hexdigest()


def _content_pin_hash(value: object) -> str:
    return "sha256:" + content_hash(value)


def _artifact_pin(kind: str, ref: str, digest: str) -> ArtifactPinV1:
    return ArtifactPinV1(
        artifact_kind=kind,
        artifact_ref=ref,
        artifact_hash=digest,
    )


class TenderDocumentDateObservationService:
    """Observe a date only from an exact, source-authored role field.

    The service ignores ``SourceIdentity.created_at``. It supports the current
    PDF native-source representation and leaves all unrecognized cases UNKNOWN.
    """

    @staticmethod
    def _normalize(raw: str) -> str | None:
        value = raw.strip()
        if _ISO_DATE.fullmatch(value):
            try:
                return date.fromisoformat(value).isoformat()
            except ValueError:
                return None
        match = _CHINESE_DATE.fullmatch(value)
        if not match:
            return None
        try:
            return date(
                int(match.group(1)), int(match.group(2)), int(match.group(3))
            ).isoformat()
        except ValueError:
            return None

    @staticmethod
    def _locator_by_block(
        document: CanonicalSourceDocument,
    ) -> dict[str, SourceLocator]:
        result: dict[str, SourceLocator] = {}
        for mapping in document.source_map.mappings:
            if len(mapping.block_ids) != 1 or len(mapping.source_locators) != 1:
                continue
            block_id = mapping.block_ids[0]
            if block_id in result:
                continue
            result[block_id] = mapping.source_locators[0]
        return result

    def observe(
        self,
        *,
        source_identity: SourceIdentity,
        canonical_source_ir: CanonicalSourceDocument,
        schema: NormativeSchemaContractV1,
        observed_at: datetime,
    ) -> TenderDocumentDateObservationV1:
        if (source_identity.organization_id, source_identity.workspace_id) != (
            schema.organization_id,
            schema.workspace_id,
        ):
            raise ValueError("FOREIGN_SCOPE")
        if canonical_source_ir.source_identity_ref != source_identity.source_identity_ref:
            raise ValueError("SOURCE_IDENTITY_MISMATCH")
        definition = next(
            (
                item
                for item in (schema.fact_definitions or ())
                if item.fact_ref == "fact.document_date"
            ),
            None,
        )
        if definition is None or definition.value_type != "date":
            raise ValueError("DOCUMENT_DATE_DEFINITION_REQUIRED")

        locators = self._locator_by_block(canonical_source_ir)
        if not locators or any(
            locator.source_hash != source_identity.source_hash
            for locator in locators.values()
        ):
            raise ValueError("CANONICAL_SOURCE_HASH_MISMATCH")
        candidates: list[tuple[str, str]] = []
        for block in canonical_source_ir.blocks:
            locator = locators.get(block.block_id)
            if (
                locator is None
                or locator.media_type != "application/pdf"
                or locator.format != "pdf"
                or locator.native_provider != "pymupdf"
                or not isinstance(block.content, str)
            ):
                continue
            if (
                source_content_hash(block.content) != block.content_hash
                or source_block_identity(
                    locator.source_hash, locator, "paragraph"
                )
                != block.block_id
            ):
                raise ValueError("CANONICAL_SOURCE_BLOCK_BINDING_MISMATCH")
            lines = block.content.splitlines()
            issuer_lines = [
                index
                for index, line in enumerate(lines)
                if re.match(r"^\s*招标人\s*[:：]", line)
            ]
            agency_lines = [
                index
                for index, line in enumerate(lines)
                if re.match(r"^\s*招标代理机构\s*[:：]", line)
            ]
            if not issuer_lines or not agency_lines:
                continue
            role_end = max(issuer_lines[0], agency_lines[0])
            next_line = role_end + 1
            while next_line < len(lines) and not lines[next_line].strip():
                next_line += 1
            if next_line >= len(lines):
                continue
            match = _DATE_LABEL.fullmatch(lines[next_line])
            if match:
                candidates.append((block.block_id, match.group(1).strip()))

        target_evidence = tuple(
            _artifact_pin(
                "evidence",
                f"evidence:canonical-block:{block_id.removeprefix('sha256:')}",
                block_id,
            )
            for block_id in sorted({block_id for block_id, _ in candidates})
        )
        raw_values = tuple(raw for _, raw in candidates)
        normalized: str | None = None
        reason: str | None = None
        if len(candidates) > 1:
            reason = "AMBIGUOUS_SOURCE_DATE"
        elif not candidates or not candidates[0][1]:
            reason = "MISSING_OR_INCOMPLETE"
        else:
            normalized = self._normalize(candidates[0][1])
            if normalized is None:
                reason = "INVALID_SOURCE_DATE"

        source_identity_hash = _content_pin_hash(
            source_identity.model_dump(mode="json", by_alias=True, exclude_none=False)
        )
        canonical_payload = canonical_source_ir.model_dump(
            mode="json", by_alias=True, exclude_none=False
        )
        canonical_hash = _content_pin_hash(canonical_payload)
        definition_hash = _content_pin_hash(
            definition.model_dump(mode="json", by_alias=True, exclude_none=False)
        )
        return TenderDocumentDateObservationV1.create(
            organization_id=source_identity.organization_id,
            workspace_id=source_identity.workspace_id,
            status="OBSERVED" if normalized else "UNKNOWN",
            reason=reason,
            normalized_value=normalized,
            raw_values=raw_values,
            source_identity=_artifact_pin(
                "source_identity",
                source_identity.source_identity_ref,
                source_identity_hash,
            ),
            canonical_source_ir=_artifact_pin(
                "canonical_source_ir",
                f"canonical-source-ir:{source_identity.source_identity_ref}",
                canonical_hash,
            ),
            normative_schema=_artifact_pin(
                "normative_schema_contract",
                schema.schema_ref,
                schema.schema_hash,
            ),
            target_evidence=target_evidence,
            fact_definition=_artifact_pin(
                "normative_fact_definition",
                f"normative-fact-definition:{definition.fact_ref}",
                definition_hash,
            ),
            fact_definition_hash=definition_hash,
            selector=_artifact_pin(
                "source_document_date_selector",
                "source-document-date-selector:tender-issuer-agency-attestation.v1",
                _content_pin_hash(_SELECTOR_CONTRACT),
            ),
            normalizer=_artifact_pin(
                "source_document_date_normalizer",
                "source-document-date-normalizer:tender-document-date.v1",
                _content_pin_hash(_NORMALIZER_CONTRACT),
            ),
            producer=_artifact_pin(
                "source_document_date_producer",
                "source-document-date-producer:tender-document-date-observation.v1",
                _sha256(Path(__file__).read_bytes()),
            ),
            observed_at=observed_at,
        )


__all__ = ["TenderDocumentDateObservationService"]
