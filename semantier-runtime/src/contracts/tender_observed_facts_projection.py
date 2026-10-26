"""Derived, non-authoritative contracts for governed Tender fact projection."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, TypeAdapter, StrictInt

from contracts.artifact_pin import ArtifactPinV1
from contracts.normative_schema import NormativeSchemaContractV1
from eos.artifact_hashing import content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def utc_timestamps(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and (
                value.tzinfo is None or value.utcoffset() != timedelta(0)
            ):
                raise ValueError(f"{name} must be timezone-aware UTC")
        return self


def _pin_key(pin: ArtifactPinV1) -> tuple[str, str, str]:
    return (pin.artifact_kind, pin.artifact_ref, pin.artifact_hash)


class TenderFactSensorContractV1(_Contract):
    """Model-visible meaning for exactly one source-derived Boolean fact.

    This is a derived representation contract, not semantic authority.  Its
    source meaning is copied exactly from the pinned normative schema fact
    definition so a scorer never has to reconstruct meaning from model memory.
    """

    schema_version: Literal["tender_fact_sensor_contract.v1"] = (
        "tender_fact_sensor_contract.v1"
    )
    sensor_ref: str = Field(min_length=1)
    sensor_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    normative_schema_ref: str = Field(min_length=1)
    normative_schema_hash: str = Field(pattern=SHA256_PATTERN)
    fact_ref: str = Field(min_length=1)
    value_type: Literal["boolean"] = "boolean"
    source_meaning: str = Field(min_length=1)
    definition_evidence_refs: tuple[str, ...] = Field(min_length=1)
    frontier: tuple[
        Literal["TRUE"], Literal["FALSE"], Literal["UNKNOWN"]
    ] = ("TRUE", "FALSE", "UNKNOWN")
    composition_role: Literal["INPUT_ONLY"] = "INPUT_ONLY"

    @model_validator(mode="after")
    def exact_identity(self):
        if self.frontier != ("TRUE", "FALSE", "UNKNOWN"):
            raise ValueError("EXACT_TERNARY_FRONTIER_REQUIRED")
        if tuple(sorted(set(self.definition_evidence_refs))) != self.definition_evidence_refs:
            raise ValueError("DEFINITION_EVIDENCE_MUST_BE_CANONICAL")
        payload = self.model_dump(
            mode="json", exclude={"sensor_ref", "sensor_hash"}
        )
        digest = content_hash(payload)
        if (self.sensor_ref, self.sensor_hash) != (
            f"tender-fact-sensor:{digest}",
            f"sha256:{digest}",
        ):
            raise ValueError("SENSOR_HASH_MISMATCH")
        return self

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="tender_fact_sensor_contract",
            artifact_ref=self.sensor_ref,
            artifact_hash=self.sensor_hash,
        )

    @classmethod
    def from_schema(
        cls, *, schema: NormativeSchemaContractV1, fact_ref: str
    ) -> "TenderFactSensorContractV1":
        definitions = {
            item.fact_ref: item for item in (schema.fact_definitions or ())
        }
        try:
            definition = definitions[fact_ref]
        except KeyError as exc:
            raise ValueError("FACT_DEFINITION_MISSING") from exc
        if definition.value_type != "boolean":
            raise ValueError("BOOLEAN_FACT_REQUIRED")
        values = {
            "schema_version": "tender_fact_sensor_contract.v1",
            "organization_id": schema.organization_id,
            "workspace_id": schema.workspace_id,
            "normative_schema_ref": schema.schema_ref,
            "normative_schema_hash": schema.schema_hash,
            "fact_ref": definition.fact_ref,
            "value_type": "boolean",
            "source_meaning": definition.definition,
            "definition_evidence_refs": tuple(
                sorted(set(definition.evidence_refs))
            ),
            "frontier": ("TRUE", "FALSE", "UNKNOWN"),
            "composition_role": "INPUT_ONLY",
        }
        digest = content_hash(values)
        return cls(
            **values,
            sensor_ref=f"tender-fact-sensor:{digest}",
            sensor_hash=f"sha256:{digest}",
        )


class TenderFactObservationReceiptV1(_Contract):
    """Replayable result of an external governed sensor over target evidence.

    The receipt is evidence, never authority.  A projector may consume it only
    when its sensor contract exactly matches the current normative definition.
    """

    schema_version: Literal["tender_fact_observation_receipt.v1"] = (
        "tender_fact_observation_receipt.v1"
    )
    receipt_ref: str = Field(min_length=1)
    receipt_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    sensor: ArtifactPinV1
    fact_ref: str = Field(min_length=1)
    semantic_result: Literal["TRUE", "FALSE", "UNKNOWN"]
    target_evidence: tuple[ArtifactPinV1, ...] = Field(min_length=1)
    producer: ArtifactPinV1
    producer_config: ArtifactPinV1
    basis: tuple[ArtifactPinV1, ...] = Field(min_length=1)
    subject_binding: TenderFactSubjectBindingV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    cited_evidence: tuple[ArtifactPinV1, ...] | None = Field(default=None, exclude_if=lambda value: value is None)
    observed_at: datetime

    @model_validator(mode="after")
    def exact_identity(self):
        if self.sensor.artifact_kind != "tender_fact_sensor_contract":
            raise ValueError("FACT_SENSOR_CONTRACT_REQUIRED")
        if any(pin.artifact_kind != "evidence" for pin in self.target_evidence):
            raise ValueError("TARGET_EVIDENCE_REQUIRED")
        if tuple(sorted(self.target_evidence, key=_pin_key)) != self.target_evidence:
            raise ValueError("TARGET_EVIDENCE_MUST_BE_CANONICAL")
        if len({_pin_key(pin) for pin in self.target_evidence}) != len(
            self.target_evidence
        ):
            raise ValueError("TARGET_EVIDENCE_MUST_BE_UNIQUE")
        if tuple(sorted(self.basis, key=_pin_key)) != self.basis:
            raise ValueError("OBSERVATION_BASIS_MUST_BE_CANONICAL")
        if (self.subject_binding is None) != (self.cited_evidence is None):
            raise ValueError("SCOPED_BOOLEAN_FIELDS_REQUIRED_TOGETHER")
        if self.subject_binding is not None:
            if (self.organization_id, self.workspace_id) != (self.subject_binding.organization_id, self.subject_binding.workspace_id):
                raise ValueError("SUBJECT_SCOPE_MISMATCH")
            if tuple(sorted(set(self.cited_evidence), key=_pin_key)) != self.cited_evidence:
                raise ValueError("CITED_EVIDENCE_MUST_BE_CANONICAL_UNIQUE")
            if not set(self.cited_evidence).issubset(self.target_evidence):
                raise ValueError("CITED_EVIDENCE_OUTSIDE_TARGET")
            if self.semantic_result != "UNKNOWN" and self.subject_binding.occurrence_evidence not in self.cited_evidence:
                raise ValueError("DETERMINATE_BOOLEAN_REQUIRES_OCCURRENCE_CITATION")
        payload = self.model_dump(
            mode="json", exclude={"receipt_ref", "receipt_hash"}
        )
        digest = content_hash(payload)
        if (self.receipt_ref, self.receipt_hash) != (
            f"tender-fact-observation:{digest}",
            f"sha256:{digest}",
        ):
            raise ValueError("OBSERVATION_HASH_MISMATCH")
        return self

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="tender_fact_observation_receipt",
            artifact_ref=self.receipt_ref,
            artifact_hash=self.receipt_hash,
        )

    @classmethod
    def create(cls, **values) -> "TenderFactObservationReceiptV1":
        values = dict(values)
        values["target_evidence"] = tuple(
            sorted(tuple(values["target_evidence"]), key=_pin_key)
        )
        values["basis"] = tuple(sorted(tuple(values["basis"]), key=_pin_key))
        if values.get("cited_evidence") is not None:
            values["cited_evidence"] = tuple(sorted(values["cited_evidence"], key=_pin_key))
        payload = {
            "schema_version": "tender_fact_observation_receipt.v1",
            **values,
        }
        probe = cls.model_construct(
            **payload,
            receipt_ref="pending",
            receipt_hash="sha256:" + "0" * 64,
        )
        digest = content_hash(
            probe.model_dump(
                mode="json", exclude={"receipt_ref", "receipt_hash"}
            )
        )
        return cls(
            **payload,
            receipt_ref=f"tender-fact-observation:{digest}",
            receipt_hash=f"sha256:{digest}",
        )



class TenderFactSubjectBindingV1(_Contract):
    """Exact case actor/condition occurrence; an embedded value, not authority."""
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    context_graph: ArtifactPinV1
    actor_node_ref: str = Field(min_length=1)
    condition_node_ref: str = Field(min_length=1)
    association_assertion_ref: str = Field(min_length=1)
    association_assertion_hash: str = Field(pattern=SHA256_PATTERN)
    occurrence_evidence: ArtifactPinV1
    source_ref: str = Field(min_length=1)
    source_hash: str = Field(pattern=SHA256_PATTERN)
    selector_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_source_ir: ArtifactPinV1

    @model_validator(mode="after")
    def kinds_and_machine_ids(self):
        for field, kind in (("context_graph", "context_graph"), ("occurrence_evidence", "evidence"),
                            ("canonical_source_ir", "canonical_source_ir")):
            if getattr(self, field).artifact_kind != kind:
                raise ValueError("SUBJECT_BINDING_PIN_KIND_MISMATCH")
        for field in ("actor_node_ref", "condition_node_ref", "association_assertion_ref"):
            if not getattr(self, field).isascii() or getattr(self, field).strip() != getattr(self, field):
                raise ValueError("SUBJECT_BINDING_ASCII_ID_REQUIRED")
        return self


class TenderCategoricalFactSensorContractV1(_Contract):
    schema_version: Literal["tender_categorical_fact_sensor_contract.v1"] = "tender_categorical_fact_sensor_contract.v1"
    sensor_ref: str = Field(min_length=1)
    sensor_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    normative_schema_ref: str = Field(min_length=1)
    normative_schema_hash: str = Field(pattern=SHA256_PATTERN)
    fact_ref: str = Field(min_length=1)
    value_type: Literal["string"] = "string"
    source_meaning: str = Field(min_length=1)
    definition_evidence_refs: tuple[str, ...] = Field(min_length=1)
    allowed_values: tuple[str, ...] = Field(min_length=1)
    composition_role: Literal["INPUT_ONLY"] = "INPUT_ONLY"

    @model_validator(mode="after")
    def exact_identity(self):
        if (len(set(self.allowed_values)) != len(self.allowed_values)
            or any(not v or v != v.strip() for v in self.allowed_values)):
            raise ValueError("EXACT_CATEGORICAL_DOMAIN_REQUIRED")
        digest = content_hash(self.model_dump(mode="json", exclude={"sensor_ref", "sensor_hash"}))
        if (self.sensor_ref, self.sensor_hash) != (f"tender-categorical-fact-sensor:{digest}", f"sha256:{digest}"):
            raise ValueError("CATEGORICAL_SENSOR_HASH_MISMATCH")
        return self

    def pin(self):
        return ArtifactPinV1(artifact_kind="tender_categorical_fact_sensor_contract",
                             artifact_ref=self.sensor_ref, artifact_hash=self.sensor_hash)

    @classmethod
    def from_schema(cls, *, schema, fact_ref):
        definition = next((d for d in schema.fact_definitions or () if d.fact_ref == fact_ref), None)
        if definition is None:
            raise ValueError("FACT_DEFINITION_MISSING")
        if definition.value_type != "string" or not definition.allowed_values:
            raise ValueError("CATEGORICAL_FACT_REQUIRED")
        body = dict(schema_version="tender_categorical_fact_sensor_contract.v1",
                    organization_id=schema.organization_id, workspace_id=schema.workspace_id,
                    normative_schema_ref=schema.schema_ref, normative_schema_hash=schema.schema_hash,
                    fact_ref=fact_ref, value_type="string", source_meaning=definition.definition,
                    definition_evidence_refs=tuple(sorted(set(definition.evidence_refs))),
                    allowed_values=definition.allowed_values, composition_role="INPUT_ONLY")
        digest = content_hash(body)
        return cls(**body, sensor_ref=f"tender-categorical-fact-sensor:{digest}", sensor_hash=f"sha256:{digest}")


class TenderCategoricalFactObservationReceiptV1(_Contract):
    schema_version: Literal["tender_categorical_fact_observation_receipt.v1"] = "tender_categorical_fact_observation_receipt.v1"
    receipt_ref: str = Field(min_length=1)
    receipt_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    sensor: ArtifactPinV1
    fact_ref: str = Field(min_length=1)
    subject_binding: TenderFactSubjectBindingV1
    semantic_result: Literal["KNOWN", "UNKNOWN"]
    value: str | None
    reason: Literal["MISSING", "AMBIGUOUS", "CONFLICTING", "INSUFFICIENT_EVIDENCE"] | None
    target_evidence: tuple[ArtifactPinV1, ...] = Field(min_length=1)
    cited_evidence: tuple[ArtifactPinV1, ...]
    producer: ArtifactPinV1
    producer_config: ArtifactPinV1
    basis: tuple[ArtifactPinV1, ...] = Field(min_length=1)
    observed_at: datetime

    @model_validator(mode="after")
    def exact_identity(self):
        if self.sensor.artifact_kind != "tender_categorical_fact_sensor_contract":
            raise ValueError("CATEGORICAL_SENSOR_REQUIRED")
        if (self.organization_id, self.workspace_id) != (self.subject_binding.organization_id, self.subject_binding.workspace_id):
            raise ValueError("SUBJECT_SCOPE_MISMATCH")
        if self.semantic_result == "KNOWN":
            if self.value is None or self.reason is not None or not self.cited_evidence:
                raise ValueError("KNOWN_CATEGORY_REQUIRES_VALUE_AND_EVIDENCE")
            if self.subject_binding.occurrence_evidence not in self.cited_evidence:
                raise ValueError("KNOWN_CATEGORY_REQUIRES_OCCURRENCE_CITATION")
        elif self.value is not None or self.reason is None:
            raise ValueError("UNKNOWN_CATEGORY_MUST_REMAIN_ABSENT")
        for name in ("target_evidence", "cited_evidence", "basis"):
            pins = getattr(self, name)
            if tuple(sorted(pins, key=_pin_key)) != pins or len(set(_pin_key(p) for p in pins)) != len(pins):
                raise ValueError("CATEGORICAL_PINS_MUST_BE_CANONICAL_UNIQUE")
        if any(p.artifact_kind != "evidence" for p in (*self.target_evidence, *self.cited_evidence)):
            raise ValueError("TARGET_EVIDENCE_REQUIRED")
        if not set(self.cited_evidence).issubset(self.target_evidence):
            raise ValueError("CITED_EVIDENCE_OUTSIDE_TARGET")
        digest = content_hash(self.model_dump(mode="json", exclude={"receipt_ref", "receipt_hash"}))
        if (self.receipt_ref, self.receipt_hash) != (f"tender-categorical-fact-observation:{digest}", f"sha256:{digest}"):
            raise ValueError("CATEGORICAL_OBSERVATION_HASH_MISMATCH")
        return self

    def pin(self):
        return ArtifactPinV1(artifact_kind="tender_categorical_fact_observation_receipt",
                             artifact_ref=self.receipt_ref, artifact_hash=self.receipt_hash)

    @classmethod
    def create(cls, **values):
        for name in ("target_evidence", "cited_evidence", "basis"):
            values[name] = tuple(sorted(values[name], key=_pin_key))
        draft = cls.model_construct(schema_version="tender_categorical_fact_observation_receipt.v1",
                                    receipt_ref="pending", receipt_hash="sha256:" + "0" * 64, **values)
        body = draft.model_dump(mode="json", exclude={"receipt_ref", "receipt_hash"})
        digest = content_hash(body)
        return cls(**body, receipt_ref=f"tender-categorical-fact-observation:{digest}", receipt_hash=f"sha256:{digest}")

class TenderObservedFactsProjectionReceiptV1(_Contract):
    """Exact provenance for one target-evidence -> ObservedFacts projection."""

    schema_version: Literal["tender_observed_facts_projection_receipt.v1"] = (
        "tender_observed_facts_projection_receipt.v1"
    )
    projection_ref: str = Field(min_length=1)
    projection_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    subject_binding: TenderFactSubjectBindingV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    compiled_plan: ArtifactPinV1
    normative_schema: ArtifactPinV1
    target_evidence: tuple[ArtifactPinV1, ...] = Field(min_length=1)
    sensor_contracts: tuple[ArtifactPinV1, ...]
    observation_receipts: tuple[ArtifactPinV1, ...]
    required_fact_refs: tuple[str, ...] = Field(min_length=1)
    unresolved_fact_refs: tuple[str, ...]
    observed_facts: ArtifactPinV1
    projector: ArtifactPinV1
    frozen_at: datetime

    @model_validator(mode="after")
    def exact_identity(self):
        for pins, label in (
            (self.target_evidence, "TARGET_EVIDENCE"),
            (self.sensor_contracts, "SENSOR_CONTRACTS"),
            (self.observation_receipts, "OBSERVATION_RECEIPTS"),
        ):
            if tuple(sorted(pins, key=_pin_key)) != pins:
                raise ValueError(f"{label}_MUST_BE_CANONICAL")
            if len({_pin_key(pin) for pin in pins}) != len(pins):
                raise ValueError(f"{label}_MUST_BE_UNIQUE")
        if tuple(sorted(set(self.required_fact_refs))) != self.required_fact_refs:
            raise ValueError("REQUIRED_FACT_REFS_MUST_BE_CANONICAL")
        if tuple(sorted(set(self.unresolved_fact_refs))) != self.unresolved_fact_refs:
            raise ValueError("UNRESOLVED_FACT_REFS_MUST_BE_CANONICAL")
        if not set(self.unresolved_fact_refs).issubset(self.required_fact_refs):
            raise ValueError("UNRESOLVED_FACT_NOT_REQUIRED")
        if self.observed_facts.artifact_kind != "paired_observed_facts":
            raise ValueError("OBSERVED_FACTS_PIN_REQUIRED")
        payload = self.model_dump(
            mode="json", exclude={"projection_ref", "projection_hash"}
        )
        digest = content_hash(payload)
        if (self.projection_ref, self.projection_hash) != (
            f"tender-observed-facts-projection:{digest}",
            f"sha256:{digest}",
        ):
            raise ValueError("PROJECTION_HASH_MISMATCH")
        return self

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="tender_observed_facts_projection_receipt",
            artifact_ref=self.projection_ref,
            artifact_hash=self.projection_hash,
        )

    @classmethod
    def create(cls, **values) -> "TenderObservedFactsProjectionReceiptV1":
        values = dict(values)
        for name in (
            "target_evidence",
            "sensor_contracts",
            "observation_receipts",
        ):
            values[name] = tuple(sorted(tuple(values[name]), key=_pin_key))
        for name in ("required_fact_refs", "unresolved_fact_refs"):
            values[name] = tuple(sorted(set(values[name])))
        payload = {
            "schema_version": "tender_observed_facts_projection_receipt.v1",
            **values,
        }
        probe = cls.model_construct(
            **payload,
            projection_ref="pending",
            projection_hash="sha256:" + "0" * 64,
        )
        digest = content_hash(
            probe.model_dump(
                mode="json", exclude={"projection_ref", "projection_hash"}
            )
        )
        return cls(
            **payload,
            projection_ref=f"tender-observed-facts-projection:{digest}",
            projection_hash=f"sha256:{digest}",
        )


__all__ = [
    "TenderFactSubjectBindingV1",
    "TenderCategoricalFactSensorContractV1",
    "TenderCategoricalFactObservationReceiptV1",
    "TenderFactObservationReceiptV1",
    "TenderFactSensorContractV1",
    "TenderObservedFactsProjectionReceiptV1",
]

TenderFactObservationReceiptV1.model_rebuild()


class SupportBundleV1(_Contract):
    operator: Literal['ALL_AVAILABLE'] = 'ALL_AVAILABLE'
    premises: tuple[StrictInt, ...] = Field(min_length=1, max_length=4)
    complete: Literal[True] = True

    @model_validator(mode='after')
    def unique(self):
        if tuple(sorted(set(self.premises))) != self.premises:
            raise ValueError('SUPPORT_DUPLICATE_PREMISES')
        return self


class SupportOriginGroupV1(_Contract):
    group_id: str = Field(pattern=r'^[A-Za-z0-9_.:-]+$')
    origin: ArtifactPinV1
    members: tuple[StrictInt, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode='after')
    def unique(self):
        if tuple(sorted(set(self.members))) != self.members:
            raise ValueError('SUPPORT_DUPLICATE_GROUP_MEMBERS')
        return self


class BooleanSupportProfileV1(_Contract):
    """Finite license payload interpreted only after existing rule-governance checks."""
    kind: Literal['fact_support_boolean.v1'] = 'fact_support_boolean.v1'
    organization_id: str
    workspace_id: str
    sensor: ArtifactPinV1
    subject_binding: TenderFactSubjectBindingV1 | None = None
    value: bool = Field(strict=True)
    operator: Literal['ANY_SUFFICIENT_BUNDLE'] = 'ANY_SUFFICIENT_BUNDLE'
    premises: tuple[ArtifactPinV1, ...] = Field(min_length=1, max_length=4)
    bundles: tuple[SupportBundleV1, ...] = Field(min_length=1, max_length=4)
    alternatives_coverage: Literal['COMPLETE','PARTIAL','UNKNOWN']
    groups: tuple[SupportOriginGroupV1, ...] = Field(max_length=4)
    group_coverage: Literal['COMPLETE','PARTIAL','UNKNOWN']

    @property
    def profile_hash(self):
        return 'sha256:' + content_hash(self.model_dump(mode='json'))

    @model_validator(mode='after')
    def finite(self):
        if len(set(self.premises)) != len(self.premises):
            raise ValueError('SUPPORT_DUPLICATE_PREMISES')
        if any(p.artifact_kind != 'evidence' for p in self.premises):
            raise ValueError('UNSUPPORTED_DERIVATION')
        indices=set(range(len(self.premises)))
        if any(not set(b.premises)<=indices for b in self.bundles):
            raise ValueError('SUPPORT_DANGLING_PREMISE')
        if len({b.premises for b in self.bundles})!=len(self.bundles):
            raise ValueError('SUPPORT_DUPLICATE_BUNDLE')
        if set(i for b in self.bundles for i in b.premises)!=indices:
            raise ValueError('SUPPORT_UNUSED_PREMISE')
        if len({g.origin for g in self.groups})!=len(self.groups):
            raise ValueError('SUPPORT_DUPLICATE_ORIGIN')
        if len({g.group_id for g in self.groups})!=len(self.groups):
            raise ValueError('SUPPORT_DUPLICATE_GROUP')
        members=[i for g in self.groups for i in g.members]
        if not set(members)<=indices or len(set(members))!=len(members):
            raise ValueError('SUPPORT_AMBIGUOUS_GROUP_MAPPING')
        if self.group_coverage=='COMPLETE' and set(members)!=indices:
            raise ValueError('SUPPORT_INCOMPLETE_GROUP_MAPPING')
        return self


class FactSupportDeclarationV1(_Contract):
    profile: BooleanSupportProfileV1
    rule_version: ArtifactPinV1
    activation_id: str
    original_observation: ArtifactPinV1
    licensed_at: datetime


def _create_v2(cls, prefix, ref_field, hash_field, values):
    values=dict(values)
    values.pop('schema_version',None)
    values={k:TypeAdapter(cls.model_fields[k].annotation).validate_python(v) for k,v in values.items()}
    draft=cls.model_construct(**values, **{ref_field:'pending',hash_field:'sha256:'+'0'*64})
    body=draft.model_dump(mode='json',exclude={ref_field,hash_field})
    digest=content_hash(body)
    return cls.model_validate({**body,ref_field:prefix+digest,hash_field:'sha256:'+digest})


class TenderFactObservationReceiptV2(TenderFactObservationReceiptV1):
    schema_version: Literal['tender_fact_observation_receipt.v2']='tender_fact_observation_receipt.v2'
    support_declaration: FactSupportDeclarationV1

    @model_validator(mode='after')
    def exact_identity(self):
        body=self.model_dump(mode='json',exclude={'receipt_ref','receipt_hash'})
        digest=content_hash(body)
        if (self.receipt_ref,self.receipt_hash)!=('tender-fact-observation:v2:'+digest,'sha256:'+digest):
            raise ValueError('OBSERVATION_HASH_MISMATCH')
        legacy={k:v for k,v in body.items() if k not in {'schema_version','support_declaration'}}
        legacy['schema_version']='tender_fact_observation_receipt.v1'
        old_digest=content_hash(legacy)
        TenderFactObservationReceiptV1.model_validate({**legacy,'receipt_ref':'tender-fact-observation:'+old_digest,'receipt_hash':'sha256:'+old_digest})
        p=self.support_declaration.profile
        if ((p.organization_id,p.workspace_id,p.sensor,p.subject_binding) !=
            (self.organization_id,self.workspace_id,self.sensor,self.subject_binding)
            or self.semantic_result not in ('TRUE','FALSE')
            or p.value != (self.semantic_result=='TRUE')
            or not set(p.premises)<=set(self.target_evidence)):
            raise ValueError('SUPPORT_OBSERVATION_BINDING_MISMATCH')
        if self.support_declaration.original_observation.artifact_kind!='tender_fact_observation_receipt':
            raise ValueError('SUPPORT_ORIGINAL_OBSERVATION_REQUIRED')
        return self

    @classmethod
    def create(cls, **values):
        return _create_v2(cls,'tender-fact-observation:v2:','receipt_ref','receipt_hash',values)


class TenderObservedFactsProjectionReceiptV2(TenderObservedFactsProjectionReceiptV1):
    schema_version: Literal['tender_observed_facts_projection_receipt.v2']='tender_observed_facts_projection_receipt.v2'
    support_observation_refs: tuple[str,...] = Field(min_length=1)

    @model_validator(mode='after')
    def exact_identity(self):
        body=self.model_dump(mode='json',exclude={'projection_ref','projection_hash'})
        digest=content_hash(body)
        if (self.projection_ref,self.projection_hash)!=('tender-observed-facts-projection:v2:'+digest,'sha256:'+digest):
            raise ValueError('PROJECTION_HASH_MISMATCH')
        legacy={k:v for k,v in body.items() if k not in {'schema_version','support_observation_refs'}}
        legacy['schema_version']='tender_observed_facts_projection_receipt.v1'
        old_digest=content_hash(legacy)
        TenderObservedFactsProjectionReceiptV1.model_validate({**legacy,'projection_ref':'tender-observed-facts-projection:'+old_digest,'projection_hash':'sha256:'+old_digest})
        if (tuple(sorted(set(self.support_observation_refs)))!=self.support_observation_refs
            or not set(self.support_observation_refs)<={p.artifact_ref for p in self.observation_receipts}):
            raise ValueError('SUPPORT_PROJECTION_BINDING_MISMATCH')
        return self

    @classmethod
    def create(cls, **values):
        return _create_v2(cls,'tender-observed-facts-projection:v2:','projection_ref','projection_hash',values)
