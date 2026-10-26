"""Documentary review only: immutable opinions never confer semantic authority."""
from datetime import datetime, timedelta
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_serializer

from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_working_context import V2UtcDateTime
from eos.artifact_hashing import content_hash

Id = Annotated[str, Field(strict=True, min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")]
Hash = Annotated[str, Field(strict=True, pattern=r"^sha256:[0-9a-f]{64}$")]
Note = Annotated[str, Field(strict=True, max_length=20000)]
Material = Annotated[str, Field(strict=True, max_length=2000)]
Action = Literal["RECORD_CONCERN", "PROPOSE_CHANGE", "REQUEST_MATERIAL", "LEAVE_UNRESOLVED", "NO_CONCERN_RAISED_IN_SCOPE"]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def utc_only(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and (value.tzinfo is None or value.utcoffset() != timedelta(0)):
                raise ValueError("persisted times must be timezone-aware UTC")
        return self


class SourceOccurrenceSubjectV1(_Contract):
    kind: Literal["SOURCE_OCCURRENCE"]
    source_block_ref: Id
    evidence_ref: Id
    selector_hash: Hash


class GovernedFindingSubjectV1(_Contract):
    kind: Literal["GOVERNED_FINDING"]
    source_block_ref: Id
    evidence_ref: Id
    selector_hash: Hash
    journey_ref: Id
    episode_pin: ArtifactPinV1
    finding_pin: ArtifactPinV1


Subject = Annotated[SourceOccurrenceSubjectV1 | GovernedFindingSubjectV1, Field(discriminator="kind")]


class TenderSupportingCitationV1(_Contract):
    """Exact reviewer-selected source occurrence; the enclosing record owns attribution."""
    schema_version: Literal["tender_supporting_citation.v1"]
    artifact_pin: ArtifactPinV1
    preparation_pin: ArtifactPinV1
    preparation_purpose: Literal["DESIGN_SOURCE", "RUNTIME_TARGET"]
    source_identity_ref: Id
    source_hash: Hash
    source_identity: dict
    canonical_source_ir_pin: ArtifactPinV1
    subject: SourceOccurrenceSubjectV1
    evidence: dict
    exact_quote: str = Field(strict=True, min_length=1)
    resolved_location: dict
    provision_label: str
    selection_origin: Literal["REVIEWER_SELECTED"]
    applicability: Literal["UNKNOWN"]


class _Opinion(_Contract):
    status: Literal["DRAFT", "RECORDED"]
    action: Action | None
    user_note: Note
    proposed_replacement: Note
    requested_material: Material
    supporting_citations: tuple[TenderSupportingCitationV1, ...] | None = Field(default=None, max_length=100, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def actionable_opinion(self):
        if self.status == "RECORDED" and self.action is None:
            raise ValueError("recorded review requires an action")
        if self.action == "REQUEST_MATERIAL" and not self.requested_material.strip():
            raise ValueError("material request must name the material")
        return self


class TenderReviewRecordCommandV1(_Opinion):
    request_id: Id
    idempotency_key: Id
    expected_material_context_revision: Annotated[int, Field(strict=True, ge=0)]
    cursor_ref: Id
    target_artifact_pin: ArtifactPinV1
    preparation_pin: ArtifactPinV1
    subject: Subject
    expected_head_pin: ArtifactPinV1 | None

    @model_validator(mode="after")
    def correct_kinds(self):
        for name, kind in (("target_artifact_pin", "workspace_artifact"), ("preparation_pin", "workspace_artifact_preparation"), ("expected_head_pin", "tender_review_record")):
            value = getattr(self, name)
            if value is not None and value.artifact_kind != kind:
                raise ValueError(f"{name} must pin {kind}")
        return self


class TenderReviewReportCommandV1(_Contract):
    request_id: Id
    idempotency_key: Id
    target_artifact_pin: ArtifactPinV1
    preparation_pin: ArtifactPinV1
    review_record_pins: tuple[ArtifactPinV1, ...]
    episode_pin: ArtifactPinV1 | None
    expected_material_context_revision: Annotated[int, Field(strict=True, ge=0)]
    predecessor_report_pin: ArtifactPinV1 | None

    @model_validator(mode="after")
    def correct_kinds(self):
        for name, kind in (("target_artifact_pin", "workspace_artifact"), ("preparation_pin", "workspace_artifact_preparation"), ("predecessor_report_pin", "tender_review_report")):
            pin = getattr(self, name)
            if pin is not None and pin.artifact_kind != kind:
                raise ValueError(f"{name} must pin {kind}")
        return self

    @model_validator(mode="after")
    def unique_manifest(self):
        refs = [pin.artifact_ref for pin in self.review_record_pins]
        if len(refs) != len(set(refs)) or any(pin.artifact_kind != "tender_review_record" for pin in self.review_record_pins):
            raise ValueError("report requires unique exact review record pins")
        return self


class _Artifact(_Contract):
    organization_id: Id
    workspace_id: Id
    session_ref: Id
    task_key: Hash
    target_artifact_pin: ArtifactPinV1
    preparation_pin: ArtifactPinV1
    source_identity_ref: Id
    source_hash: Hash
    canonical_source_ir_pin: ArtifactPinV1
    recorded_by: Id
    recorded_at: V2UtcDateTime
    request_id: Id
    idempotency_key: Id
    documentary_status: Literal["PROPOSED"] = "PROPOSED"
    REF_FIELD: ClassVar[str]
    HASH_FIELD: ClassVar[str]
    KIND: ClassVar[str]

    @model_validator(mode="after")
    def bound_identity(self):
        basis = self.model_dump(mode="json", exclude={self.REF_FIELD, self.HASH_FIELD})
        digest = content_hash(basis)
        if (getattr(self, self.REF_FIELD), getattr(self, self.HASH_FIELD)) != (self.KIND.replace("_", "-") + ":" + digest, "sha256:" + digest):
            raise ValueError("documentary artifact identity mismatch")
        expected_task = "sha256:" + content_hash({"session_ref": self.session_ref, "target_artifact_pin": self.target_artifact_pin.model_dump(mode="json")})
        if self.task_key != expected_task:
            raise ValueError("task identity mismatch")
        return self

    def pin(self):
        return ArtifactPinV1(artifact_kind=self.KIND, artifact_ref=getattr(self, self.REF_FIELD), artifact_hash=getattr(self, self.HASH_FIELD))

    @classmethod
    def create(cls, **values):
        for name, field in cls.model_fields.items():
            if name not in values and not field.is_required():
                values[name] = field.get_default(call_default_factory=True)
        # Absent optional citations preserve every legacy hash and wire shape.
        if values.get("supporting_citations") is None:
            values.pop("supporting_citations", None)
        digest = content_hash(values)
        return cls.model_validate({**values, cls.REF_FIELD: cls.KIND.replace("_", "-") + ":" + digest, cls.HASH_FIELD: "sha256:" + digest})


class TenderReviewRecordV1(_Artifact, _Opinion):
    schema_version: Literal["tender_review_record.v1"]
    record_ref: Id
    record_hash: Hash
    subject: Subject
    predecessor_record_pin: ArtifactPinV1 | None
    expected_head_pin: ArtifactPinV1 | None
    exact_quote: str = Field(strict=True)
    resolved_location: dict
    input_qualification: Literal["SOURCE_DOCUMENTARY_ONLY", "NATIVE_FINDING_LIVE_UNQUALIFIED"]
    machine_status: str | None
    REF_FIELD = "record_ref"
    HASH_FIELD = "record_hash"
    KIND = "tender_review_record"


class TenderReviewReportV1(_Artifact):
    schema_version: Literal["tender_review_report.v1"]
    report_ref: Id
    report_hash: Hash
    review_record_pins: tuple[ArtifactPinV1, ...]
    records: tuple[TenderReviewRecordV1, ...]
    predecessor_report_pin: ArtifactPinV1 | None
    episode_pin: ArtifactPinV1 | None
    native_context_pin: ArtifactPinV1 | None = None
    native_result_pin: ArtifactPinV1 | None = None
    native_basis_snapshot: dict | None = None
    native_finding_inventory: tuple[ArtifactPinV1, ...]
    unreviewed_finding_pins: tuple[ArtifactPinV1, ...]
    coverage: Literal["MANUAL_RECORDED_SCOPE_AUTOMATION_NOT_RUN", "NATIVE_CANDIDATE_SCOPE_LIVE_UNQUALIFIED"]
    limitations: tuple[str, ...]
    projection_version: Literal["tender_review_report_projection.v1"]
    REF_FIELD = "report_ref"
    HASH_FIELD = "report_hash"
    KIND = "tender_review_report"

    @model_validator(mode="after")
    def frozen_manifest(self):
        refs = [pin.artifact_ref for pin in self.review_record_pins]
        if len(refs) != len(set(refs)):
            raise ValueError("duplicate report records")
        if self.episode_pin is None and (self.native_finding_inventory or self.unreviewed_finding_pins or self.coverage != "MANUAL_RECORDED_SCOPE_AUTOMATION_NOT_RUN"):
            raise ValueError("manual report cannot claim an automated inventory")
        inventory = {(pin.artifact_ref, pin.artifact_hash) for pin in self.native_finding_inventory}
        if len(inventory) != len(self.native_finding_inventory):
            raise ValueError("duplicate native candidates")
        reviewed = {(record.subject.finding_pin.artifact_ref, record.subject.finding_pin.artifact_hash)
                    for record in self.records if isinstance(record.subject, GovernedFindingSubjectV1) and record.status == "RECORDED"}
        if not reviewed.issubset(inventory):
            raise ValueError("reviewed finding outside report inventory")
        expected = tuple(pin for pin in self.native_finding_inventory if (pin.artifact_ref, pin.artifact_hash) not in reviewed)
        if self.unreviewed_finding_pins != expected:
            raise ValueError("unreviewed candidate conservation mismatch")
        if self.native_basis_snapshot is not None:
            from contracts.tender_legal_compliance import TenderFrozenComplianceDecisionV1
            from contracts.evidence_location import SourceIdentity, EvidenceRef
            expected_originals = {}
            for raw in self.native_basis_snapshot.get("compliance_decisions", ()):
                attached = TenderFrozenComplianceDecisionV1.model_validate(raw)
                basis = attached.basis
                if (basis.episode != self.episode_pin or basis.inspection_result != self.native_result_pin
                    or (basis.organization_id, basis.workspace_id) != (self.organization_id, self.workspace_id)
                    or (basis.finding.finding_id, basis.finding.finding_hash) not in inventory):
                    raise ValueError("REPORT_COMPLIANCE_BINDING_MISMATCH")
                for evidence in basis.finding.model_dump(mode="json", by_alias=True)["evidenceRefs"]:
                    ref = EvidenceRef.model_validate(evidence)
                    expected_originals[(attached.decision.artifact_ref, "TENDER", ref.evidence_ref)] = (
                        attached.decision.pin(), self.source_identity_ref, self.source_hash,
                        basis.target_ir, ref, None)
                for identity, ir, evidence, citation in basis.original_provisions():
                    expected_originals[(attached.decision.artifact_ref, "NORMATIVE", evidence.evidence_ref)] = (
                        attached.decision.pin(), identity.source_identity_ref,
                        identity.source_hash, ir, evidence, citation)
            if "compliance_selection" in self.native_basis_snapshot:
                from contracts.tender_legal_compliance import freeze_assessment_selection, TenderAssessmentSelectionV1
                actual_selection = tuple(TenderAssessmentSelectionV1.model_validate(value) for value in self.native_basis_snapshot["compliance_selection"])
                expected_selection = freeze_assessment_selection(inventory=self.native_finding_inventory, entries=self.native_basis_snapshot.get("compliance_decisions", ()))
                if actual_selection != expected_selection:
                    raise ValueError("REPORT_ASSESSMENT_SELECTION_MISMATCH")
            if "compliance_originals" in self.native_basis_snapshot:
                seen = set()
                for original in self.native_basis_snapshot["compliance_originals"]:
                    decision = ArtifactPinV1.model_validate(original["decision"])
                    identity = SourceIdentity.model_validate(original["identity"])
                    evidence = EvidenceRef.model_validate(original["evidence"])
                    key = (decision.artifact_ref, original["role"], evidence.evidence_ref)
                    actual = (decision, identity.source_identity_ref, identity.source_hash,
                              ArtifactPinV1.model_validate(original["canonical_source_ir"]), evidence, original.get("citation"))
                    if (key in seen or expected_originals.get(key) != actual
                        or (identity.organization_id, identity.workspace_id) != (self.organization_id, self.workspace_id)
                        or bool(original["location"]) != (original["status"] in ("exact", "relocated_exact"))):
                        raise ValueError("REPORT_COMPLIANCE_ORIGINAL_MISMATCH")
                    seen.add(key)
                if seen != set(expected_originals):
                    raise ValueError("REPORT_COMPLIANCE_ORIGINAL_INVENTORY_MISMATCH")
        if tuple(record.pin() for record in self.records) != self.review_record_pins:
            raise ValueError("frozen report record manifest mismatch")
        for record in self.records:
            if (record.organization_id, record.workspace_id, record.session_ref, record.task_key, record.recorded_by) != (self.organization_id, self.workspace_id, self.session_ref, self.task_key, self.recorded_by):
                raise ValueError("foreign report record")
        return self


class TenderReviewDocxDerivativeV1(_Artifact):
    """Native report custody receipt; independent of legacy detection runs."""
    schema_version: Literal["tender_review_docx_derivative.v1"]
    derivative_ref: Id
    derivative_hash: Hash
    report_pin: ArtifactPinV1
    content_hash: Hash
    byte_size: Annotated[int, Field(strict=True, ge=0)]
    byte_owner_ref: str
    projection_manifest: dict
    REF_FIELD = "derivative_ref"
    HASH_FIELD = "derivative_hash"
    KIND = "tender_review_docx_derivative"

    @model_validator(mode="after")
    def report_custody(self):
        digest = self.content_hash[7:]
        if (self.report_pin.artifact_kind != "tender_review_report"
            or self.byte_owner_ref != f"artifacts/workspace-originals/sha256/{digest[:2]}/{digest}"
            or self.projection_manifest.get("report_pin") != self.report_pin.model_dump(mode="json")
            or self.projection_manifest.get("native_projection_policy") not in {"tender_native_report_docx.v1", "tender_native_report_docx.v2", "tender_documentary_opinion_docx.v1"}):
            raise ValueError("NATIVE_DOCX_REPORT_BINDING_MISMATCH")
        return self


class DocumentaryOriginalV1(_Contract):
    """A selected original is documentary custody, never a normative provision."""
    logical_source_id: str
    artifact_pin: ArtifactPinV1
    preparation_pin: ArtifactPinV1
    source_identity_ref: Id
    source_hash: Hash
    canonical_source_ir_pin: ArtifactPinV1


class TenderAuthoredQueryV1(_Contract):
    schema_version: Literal['tender_authored_query.v1']
    authorship: Literal['EXPLICIT_AI_NOMINATION', 'EXPLICIT_USER_NOMINATION']
    nomination: str = Field(min_length=1, max_length=2000)
    authored_spec_sha256: Hash
    request: 'SourceTextQueryRequest'
    originals: tuple[DocumentaryOriginalV1, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode='after')
    def source_manifest(self):
        if self.authored_spec_sha256 != 'sha256:' + content_hash(self.request.spec):
            raise ValueError('authored query hash mismatch')
        if len({p.logical_source_id for p in self.originals}) != len(self.originals):
            raise ValueError('duplicate documentary original')
        if {p.logical_source_id for p in self.originals} != {p.logical_source_id for p in self.request.sources}:
            raise ValueError('documentary original manifest mismatch')
        return self


class DocumentaryOccurrenceV1(_Contract):
    identity: 'SourceIdentity'
    canonical_source_ir_pin: ArtifactPinV1
    evidence: 'EvidenceRef | None'
    location: dict
    exact_quote: str = Field(min_length=1)
    quote_hash: Hash

    @field_serializer('identity', 'evidence')
    def source_wire(self, value):
        return None if value is None else value.model_dump(mode='json', by_alias=True)

    @model_validator(mode='after')
    def exact_binding(self):
        from hashlib import sha256
        from pydantic import TypeAdapter
        from contracts.evidence_location import PdfPageRect, HtmlTextRange, DocxStructuralRange
        TypeAdapter(PdfPageRect | HtmlTextRange | DocxStructuralRange).validate_python(self.location)
        if self.quote_hash != 'sha256:' + sha256(self.exact_quote.encode()).hexdigest():
            raise ValueError('documentary quote mismatch')
        if self.location.get('exactText') != self.exact_quote:
            raise ValueError('documentary location quote mismatch')
        if self.evidence is None:
            if self.location.get('kind') != 'html_text_range' or self.location.get('sourceHash') != self.identity.source_hash or self.location.get('quoteHash') != self.quote_hash:
                raise ValueError('documentary HTML occurrence mismatch')
        elif (self.evidence.source_identity_ref != self.identity.source_identity_ref
            or self.evidence.organization_id != self.identity.organization_id
            or self.evidence.workspace_id != self.identity.workspace_id
            or self.evidence.selector.exact_quote != self.exact_quote
            or self.evidence.selector_hash != self.evidence.selector.selector_hash()):
            raise ValueError('documentary occurrence mismatch')
        return self


class DocumentaryLawV1(_Contract):
    article: str
    logical_source_id: str
    edition: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    original: DocumentaryOriginalV1
    occurrence: DocumentaryOccurrenceV1


class TenderDocumentaryCandidateV1(_Contract):
    schema_version: Literal['tender_documentary_candidate.v1']
    query_artifact_pin: ArtifactPinV1
    authored_query: TenderAuthoredQueryV1
    tender: DocumentaryOccurrenceV1
    retrieval: dict
    laws: tuple[DocumentaryLawV1, ...]
    applicability: Literal['UNKNOWN']
    review_state: Literal['UNREVIEWED']
    association: Literal['EXPLICIT_NOMINATION_NOT_APPLICABILITY']

    @model_validator(mode='after')
    def frozen_retrieval(self):
        from services.source_text_query_service import verify_source_text_export
        from services.source_text_query_engine import digest
        verify_source_text_export(self.retrieval)
        if self.retrieval['start']['request'] != self.authored_query.request.model_dump(mode='json'):
            raise ValueError('documentary query mismatch')
        output = self.retrieval['output']
        if output['state'] not in ('COMPLETE_EMPTY', 'COMPLETE_NONEMPTY'):
            raise ValueError('incomplete retrieval cannot attach as complete')
        actual = []
        originals = {x.logical_source_id: x for x in self.authored_query.originals}
        pins = {x.logical_source_id: x for x in self.authored_query.request.sources}
        for law in self.laws:
            pin = pins[law.logical_source_id]
            identity = law.occurrence.identity
            if law.original != originals[law.logical_source_id] or (identity.organization_id, identity.workspace_id, identity.source_hash) != (
                    self.authored_query.request.organization_id, self.authored_query.request.workspace_id, 'sha256:' + pin.raw_sha256):
                raise ValueError('documentary law source mismatch')
            if (identity.source_identity_ref, identity.source_hash, law.occurrence.canonical_source_ir_pin) != (law.original.source_identity_ref, law.original.source_hash, law.original.canonical_source_ir_pin):
                raise ValueError('documentary exact original binding mismatch')
            if law.edition != pin.edition:
                raise ValueError('documentary edition mismatch')
            actual.append(dict(article=law.article, source_path=law.logical_source_id, source_hash=pin.text_sha256,
                raw_hash=pin.raw_sha256, edition=law.edition, text=law.occurrence.exact_quote,
                text_hash=law.occurrence.quote_hash.removeprefix('sha256:'), start=law.start, end=law.end))
        if digest(sorted(actual, key=lambda a:a['article'])) != digest(sorted(output['articles'], key=lambda a:a['article'])):
            raise ValueError('documentary article manifest mismatch')
        return self


class TenderReviewReportV2(TenderReviewReportV1):
    schema_version: Literal['tender_review_report.v2']
    projection_version: Literal['tender_review_report_projection.v2']
    documentary_candidates: tuple[TenderDocumentaryCandidateV1, ...] = Field(min_length=1, max_length=1)

    @model_validator(mode='after')
    def candidate_scope(self):
        for candidate in self.documentary_candidates:
            if (candidate.authored_query.request.organization_id, candidate.authored_query.request.workspace_id) != (self.organization_id, self.workspace_id):
                raise ValueError('documentary query scope mismatch')
            source = candidate.tender
            if (source.identity.organization_id, source.identity.workspace_id, source.identity.source_identity_ref,
                source.identity.source_hash, source.canonical_source_ir_pin) != (
                self.organization_id, self.workspace_id, self.source_identity_ref, self.source_hash, self.canonical_source_ir_pin):
                raise ValueError('documentary report-to-clause mismatch')
            if (candidate.retrieval['start']['actor_id'], candidate.retrieval['start']['session_id']) != (self.recorded_by, self.session_ref):
                raise ValueError('documentary retrieval owner mismatch')
        return self


class TenderCandidateReportCommandV1(TenderReviewReportCommandV1):
    schema_version: Literal['tender_candidate_report_command.v1']
    cursor_ref: Id
    query_artifact_pin: ArtifactPinV1
    subject: SourceOccurrenceSubjectV1
    start_offset: int = Field(strict=True, ge=0)
    end_offset: int = Field(strict=True, gt=0)
    exact_quote: str = Field(min_length=1, max_length=20000)


from contracts.source_text_query import SourceTextQueryRequest
from contracts.evidence_location import SourceIdentity, EvidenceRef
TenderAuthoredQueryV1.model_rebuild()
DocumentaryOccurrenceV1.model_rebuild()
TenderDocumentaryCandidateV1.model_rebuild()
TenderReviewReportV2.model_rebuild()
