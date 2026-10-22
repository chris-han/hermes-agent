"""Immutable server-authored selection/application receipts, separate from source authority."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.ontology_seed_alignment import ExactAlignmentInputV1
from contracts.ontology_seed_state import ArtifactPinV1
from eos.artifact_hashing import content_hash

HASH = r'^sha256:[0-9a-f]{64}$'


class _Receipt(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    PREFIX: ClassVar[str]
    receipt_ref: str
    receipt_hash: str = Field(pattern=HASH)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    build_ref: str = Field(min_length=1)
    actor_ref: str = Field(min_length=1)
    request_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$')
    request_hash: str = Field(pattern=HASH)
    expected_state: ExactAlignmentInputV1
    baseline_manifest: ExactAlignmentInputV1
    created_at: datetime
    authority_state: Literal['provisional'] = 'provisional'

    @model_validator(mode='after')
    def validate_identity(self):
        if self.created_at.tzinfo is None or self.created_at.utcoffset() != timedelta(0):
            raise ValueError('receipt time must be timezone-aware UTC')
        digest = content_hash(self.model_dump(mode='json', exclude={'receipt_ref', 'receipt_hash'}))
        if (self.receipt_ref, self.receipt_hash) != (f'{self.PREFIX}:{digest}', f'sha256:{digest}'):
            raise ValueError('receipt identity does not bind full content')
        return self

    @classmethod
    def create(cls, **values):
        provisional = cls.model_construct(receipt_ref='pending', receipt_hash='sha256:' + '0' * 64, **values)
        digest = content_hash(provisional.model_dump(mode='json', exclude={'receipt_ref', 'receipt_hash'}))
        return cls(receipt_ref=f'{cls.PREFIX}:{digest}', receipt_hash=f'sha256:{digest}', **values)


class SeedSelectionReceiptV1(_Receipt):
    PREFIX = 'ontology-seed-selection'
    schema_version: Literal['ontology_seed_selection.v1'] = 'ontology_seed_selection.v1'
    operation: Literal['select_seed'] = 'select_seed'
    mode: Literal['registry_recommendation', 'manual_external']
    recommendation: ArtifactPinV1 | None = None
    external_candidate_ref: str | None = None
    external_candidate_hash: str | None = Field(default=None, pattern=HASH)
    imported_snapshot: ExactAlignmentInputV1
    imported_profile: ExactAlignmentInputV1

    @model_validator(mode='after')
    def selection_mode(self):
        if self.mode == 'registry_recommendation':
            if self.recommendation is None or self.external_candidate_ref is not None or self.external_candidate_hash is not None:
                raise ValueError('registry selection requires only a recommendation pin')
        elif self.recommendation is not None or self.external_candidate_ref is None or self.external_candidate_hash is None:
            raise ValueError('manual selection requires an exact inspected external candidate')
        return self


class SeedApplyReceiptV1(_Receipt):
    PREFIX = 'ontology-seed-apply'
    schema_version: Literal['ontology_seed_apply.v1'] = 'ontology_seed_apply.v1'
    operation: Literal['apply_seed'] = 'apply_seed'
    selection_receipt: ExactAlignmentInputV1
    alignment: ExactAlignmentInputV1
    delta: ExactAlignmentInputV1
    comparison: ExactAlignmentInputV1
    lineage: ExactAlignmentInputV1
    outcome: Literal['APPLIED', 'NO_DELTA']
    successor_manifest: ExactAlignmentInputV1 | None = None
    successor_snapshot: ExactAlignmentInputV1 | None = None
    diff: ExactAlignmentInputV1 | None = None

    @model_validator(mode='after')
    def outcome_pins(self):
        pins = (self.successor_manifest, self.successor_snapshot, self.diff)
        if self.outcome == 'APPLIED' and any(pin is None for pin in pins):
            raise ValueError('applied receipt requires all successor pins')
        if self.outcome == 'NO_DELTA' and any(pin is not None for pin in pins):
            raise ValueError('empty delta must not publish a successor')
        return self
