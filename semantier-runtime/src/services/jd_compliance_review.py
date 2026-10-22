"""Bounded JD candidate review over governed parser, classifier and artifact owners."""

from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import io
import re
import zipfile
from xml.etree import ElementTree as ET

from contracts.jd_compliance_review import JdReviewV1, JdFindingV1, JdBasisEvidenceV1
from contracts.knowledge_classification import canonical_content_hash
from contracts.evidence_location import EvidenceSelector
from plugins.document_extraction.storage import sha256_bytes
from plugins.document_extraction.service import extract_live
from plugins.knowledge_parser.service import parse_knowledge_source
from services.knowledge_parsing_evidence_service import (
    InMemoryKnowledgeParsingEvidenceService,
)
from plugins.knowledge_sensitive_labeling.classifier import classify_canonical_input
from semantier.semantica_context_graph.artifact_repository import (
    ContextGraphArtifactRepository,
)

DOCX_MEDIA = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def text_projection(text: str) -> bytes:
    from xml.sax.saxutils import escape

    paragraphs = "".join(
        '<w:p><w:r><w:t xml:space="preserve">' + escape(line) + "</w:t></w:r></w:p>"
        for line in (text.splitlines() or [""])
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as package:

        def write_part(name, content):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            package.writestr(info, content)

        write_part(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>',
        )
        write_part(
            "_rels/.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>',
        )
        write_part(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
            + paragraphs
            + "<w:sectPr/></w:body></w:document>",
        )
    return buffer.getvalue()


def _location(segment, document_kind, start, end, quote):
    layout = segment.layout_metadata
    if document_kind == "pdf" and layout.get("bbox") and layout.get("page"):
        chars = layout.get("chars") or []
        if "".join(c["text"] for c in chars) == segment.text and chars[start:end]:
            boxes = [c["bbox"] for c in chars[start:end]]
            x0 = min(b[0] for b in boxes)
            y0 = min(b[1] for b in boxes)
            x1 = max(b[2] for b in boxes)
            y1 = max(b[3] for b in boxes)
            return dict(
                kind="pdf_page_rect",
                page=layout["page"],
                x=x0,
                y=y0,
                width=x1 - x0,
                height=y1 - y0,
                unit="pdf-point",
                origin="top-left",
                precision="span",
                exactText=quote,
            )
        x0, y0, x1, y1 = layout["bbox"]
        return dict(
            kind="pdf_page_rect",
            page=layout["page"],
            x=x0,
            y=y0,
            width=x1 - x0,
            height=y1 - y0,
            unit="pdf-point",
            origin="top-left",
            precision="block",
            exactText=segment.text,
        )
    anchor = layout.get("anchor") or {}
    if document_kind == "docx" and anchor.get("element_kind") in {
        "paragraph",
        "table_cell",
    }:
        if anchor["element_kind"] == "table_cell":
            path = [
                "table:" + str(anchor["table_index"] - 1),
                "row:" + str(anchor["row_index"] - 1),
                "cell:" + str(anchor["cell_index"] - 1),
                "paragraph:" + str(anchor.get("cell_paragraph_index", 1) - 1),
            ]
        else:
            path = [
                "paragraph:"
                + str(
                    anchor.get("render_paragraph_index", anchor["paragraph_index"] - 1)
                )
            ]
        return dict(
            kind="docx_structural_range",
            structuralPath=path,
            startOffset=start,
            endOffset=end,
            exactText=quote,
        )
    return dict(
        kind="unresolved", reason="Table or non-native location has not been qualified"
    )


def analyze_segments(canonical, *, source_identity_ref, document_kind, basis_documents):
    findings = []
    patterns = [
        (
            "RECRUITMENT_CONDITION",
            r"女性[\s\u3000]*已婚[\s\u3000]*已育[\s\u3000]*优先|(?:仅限|限)[\s\u3000]*男性|男性[\s\u3000]*优先",
        ),
        (
            "RECRUITMENT_BEHAVIOR",
            r"(?:要求|必须|需|进行|提供|安排)?[\s\u3000]*妊娠(?:测试|检测)|孕检",
        ),
        (
            "PERSONAL_INFORMATION",
            r"详细住址|身份证(?:号|号码)?|家属信息|家庭成员|已婚[，,、\s]*育有[^；。\n]*|(?:丈夫|妻子)(?:在|为|任|是)[^，,；。\n]*|家庭已定居[^；。\n]*",
        ),
        (
            "ORGANIZATION_INFORMATION",
            r"未公开(?:薪酬|编制)[^；。\n]*|内部薪酬[^；。\n]*|薪酬职级[^；。\n]*|内部未定薪信息|未公开经营信息",
        ),
        ("QUALIFICATION_REVIEW", r"年龄[^；。\n]*|照片|民族|户籍|健康状况"),
    ]
    full_text = "\n".join(s.text for s in canonical.segments)
    special = bool(
        re.search(r"特殊岗位|特殊行业|例外待|国家规定.*待|公务员|事业单位", full_text)
    )
    for segment in canonical.segments:
        text = segment.text
        for category, pattern in patterns:
            for match in re.finditer(pattern, text):
                quote = match.group()
                start, end = match.span()
                left = (
                    max(text.rfind(mark, 0, start) for mark in ("；", ";", "。", "\n"))
                    + 1
                )
                rights = [
                    text.find(mark, end)
                    for mark in ("；", ";", "。", "\n")
                    if text.find(mark, end) >= 0
                ]
                clause = text[left : min(rights) if rights else len(text)]
                before = text[left:start]
                quoted = bool(re.search(r"[“「\"]([^”」\"]*)$", before))
                meta_quote = quoted and bool(
                    re.search(r"培训|风险示例|法条|引用|批评|经历|第三人", clause)
                )
                local_negation = bool(
                    re.search(
                        r"(?:不得|不应|禁止|严禁|并非|无需|无须|不将|不以|不)[^，,；;。]{0,12}$",
                        before,
                    )
                )
                if (
                    meta_quote
                    or local_negation
                    or re.search(
                        r"不以.*婚育|不得.*婚育|不要求.*(?:婚育|孕检)|法条引用|引用批评",
                        clause,
                    )
                ):
                    continue
                role = (
                    "RECRUITER_REQUIREMENT"
                    if category.startswith("RECRUITMENT")
                    else "INFORMATION_REQUEST"
                )
                if category == "ORGANIZATION_INFORMATION":
                    role = "ORGANIZATION_INFORMATION_STATEMENT"
                if category == "PERSONAL_INFORMATION" and not re.search(
                    r"索取|提交|提供|必填|填写|要求", text
                ):
                    role = "PERSONAL_SELF_DESCRIPTION"
                selector = EvidenceSelector(
                    schemaVersion="semantier.evidence_selector.v1",
                    selectorKind="structure",
                    exactQuote=quote,
                    prefix=None,
                    suffix=None,
                    normalizedQuote=None,
                    structuralPath=[segment.source_anchor_ref],
                    table=None,
                    sourceElementRef=segment.source_anchor_ref,
                )
                fid = (
                    "jdf_"
                    + canonical_content_hash(
                        [
                            canonical.source_hash,
                            segment.segment_id,
                            start,
                            end,
                            category,
                        ]
                    ).split(":")[1][:24]
                )
                basis = [
                    b["basis_ref"]
                    for b in basis_documents
                    if (
                        b.get("source_id") == "women-rights-2022"
                        and category.startswith("RECRUITMENT")
                    )
                    or (
                        b.get("source_id") == "pipl-2021"
                        and category == "PERSONAL_INFORMATION"
                    )
                ]
                edit = None
                if category.startswith("RECRUITMENT") and not special:
                    edit = dict(
                        segment_id=segment.segment_id,
                        start_offset=start,
                        end_offset=end,
                        replacement="",
                        change_type="PROPOSED_REMOVAL",
                        proposal_only=True,
                    )
                findings.append(
                    JdFindingV1(
                        finding_id=fid,
                        requirement_id="jdr_" + segment.segment_id,
                        summary={
                            "RECRUITMENT_CONDITION": "招聘条件需复核",
                            "RECRUITMENT_BEHAVIOR": "招聘体检要求需复核",
                            "PERSONAL_INFORMATION": "个人信息目的与必要性需复核",
                            "ORGANIZATION_INFORMATION": "组织内部信息的外发范围待核实",
                            "QUALIFICATION_REVIEW": "岗位资格与适用依据需核实",
                        }[category],
                        finding_class="ORGANIZATION_INFORMATION"
                        if category == "ORGANIZATION_INFORMATION"
                        else "PERSONAL_INFORMATION"
                        if category == "PERSONAL_INFORMATION"
                        else "RECRUITMENT_RISK_CANDIDATE",
                        content_category=category,
                        sensitivity_assessment="CANDIDATE"
                        if category in {"PERSONAL_INFORMATION", "RECRUITMENT_BEHAVIOR"}
                        else "UNKNOWN",
                        task_relevance="IRRELEVANT"
                        if role == "PERSONAL_SELF_DESCRIPTION"
                        else "UNKNOWN",
                        collection_necessity="UNNECESSARY"
                        if role == "PERSONAL_SELF_DESCRIPTION"
                        else "UNKNOWN",
                        legal_applicability="UNKNOWN",
                        compliance_status="RISK_CANDIDATE"
                        if category.startswith("RECRUITMENT")
                        else "NEEDS_REVIEW",
                        remediation_status="AUTO_DRAFT_ELIGIBLE"
                        if edit
                        else "REVIEW_REQUIRED",
                        evidence_ref=dict(
                            schemaVersion="semantier.evidence_ref.v1",
                            evidenceRef="jd-evidence:" + fid,
                            organizationId=canonical.organization_id,
                            workspaceId=canonical.workspace_id,
                            selectorSchemaVersion="semantier.evidence_selector.v1",
                            sourceIdentityRef=source_identity_ref,
                            selectorHash=selector.selector_hash(),
                            selector=selector.model_dump(mode="json", by_alias=True),
                            evidenceRole="JD_REVIEW_CANDIDATE",
                            provenance={
                                "source_hash": canonical.source_hash,
                                "parser_profile_version": canonical.parser_profile_version,
                                "occurrence_ref": fid,
                            },
                            createdAt=datetime.now(timezone.utc).isoformat(),
                        ),
                        anchor_ref=segment.source_anchor_ref,
                        occurrence_ref=fid,
                        presentation_location=_location(
                            segment, document_kind, start, end, quote
                        ),
                        basis_refs=basis,
                        information_categories=["MEDICAL_HEALTH_COLLECTION_REQUEST"]
                        if category == "RECRUITMENT_BEHAVIOR"
                        else ["PERSONAL_INFORMATION_CANDIDATE"]
                        if category == "PERSONAL_INFORMATION"
                        else [],
                        basis_evidence_refs=[
                            dict(
                                basis_ref=b["basis_ref"],
                                basis_hash=b["basis_hash"],
                                article=c["article"],
                                evidence_ref=c["evidence_ref"],
                                presentation_location=c["presentation_location"],
                            )
                            for b in basis_documents
                            if b["basis_ref"] in basis
                            for c in b.get("clauses", [])
                            if c["article"] in {"第四十三条", "第六条", "第二十八条"}
                        ],
                        missing_facts=[
                            "statutory_exception",
                            "institutional_rule_activation",
                        ]
                        if category.startswith("RECRUITMENT")
                        else [
                            "target_audience",
                            "disclosure_scope",
                            "institutional_rule_activation",
                        ]
                        if category == "ORGANIZATION_INFORMATION"
                        else [
                            "processing_purpose",
                            "recruitment_stage",
                            "institutional_rule_activation",
                        ],
                        role=role,
                        proposed_edit=edit,
                    )
                )
    return findings


class JdComplianceReviewService:
    def __init__(
        self,
        *,
        store,
        workspace_root,
        actor_context,
        trusted_session,
        snapshot_resolver=None,
    ):
        self.store = store
        self.workspace_root = Path(workspace_root)
        self.context = actor_context
        self.bound = trusted_session
        self.snapshot_resolver = snapshot_resolver
        if self.context.session_id != self.bound.session_id:
            raise PermissionError("FORBIDDEN_SESSION")

    def _access(self, review):
        if (
            review.organization_id,
            review.workspace_id,
            review.actor_id,
            review.session_id,
        ) != (
            self.context.organization_id,
            self.context.workspace_id,
            self.context.actor_id,
            self.context.session_id,
        ):
            raise PermissionError("FORBIDDEN_SCOPE")

    def get(self, review_id):
        if not re.fullmatch(r"jdrv_[a-zA-Z0-9_-]+", review_id):
            raise ValueError("REVIEW_NOT_FOUND")
        review = self.store.get(
            organization_id=self.context.organization_id,
            workspace_id=self.context.workspace_id,
            artifact_ref=review_id,
            contract_type=JdReviewV1,
        )
        if review is None:
            raise ValueError("REVIEW_NOT_FOUND")
        self._access(review)
        # Saved evidence is replayed without parser, model, resolver or network.
        return review

    def _rules(self):
        unavailable = (
            [],
            "UNAVAILABLE",
            "unavailable",
            {
                "status": "UNAVAILABLE",
                "reason": "Governed principal or active snapshot unavailable",
            },
        )
        if self.snapshot_resolver is None:
            return unavailable
        governed = self.snapshot_resolver.resolve_governed_context(
            user_id=self.context.actor_id, workspace_id=self.context.workspace_id
        )
        if not governed or not governed.get("principal_entitlement_ref"):
            return unavailable
        if (
            governed.get("organization_id"),
            governed.get("workspace_id"),
            governed.get("user_id"),
        ) != (
            self.context.organization_id,
            self.context.workspace_id,
            self.context.actor_id,
        ):
            raise PermissionError("FORBIDDEN_RULE_SCOPE")
        try:
            snapshot = self.snapshot_resolver.materialize_snapshot(
                requested_rule_families=[],
                context=self.context,
                principal_entitlement_ref=governed["principal_entitlement_ref"],
            )
            rules, rule_set_hash = (
                self.snapshot_resolver.list_rules_for_activation_snapshot(
                    activation_set_snapshot_id=snapshot["activation_set_snapshot_id"],
                    resolved_activation_set_hash=snapshot[
                        "resolved_activation_set_hash"
                    ],
                    context=self.context,
                    requested_rule_families=[],
                )
            )
            pins = dict(
                status="AVAILABLE",
                activation_set_snapshot_id=snapshot["activation_set_snapshot_id"],
                resolved_activation_set_hash=snapshot["resolved_activation_set_hash"],
                rule_version_set_hash=rule_set_hash,
                principal_entitlement_ref=governed["principal_entitlement_ref"],
                activation_resolver_policy_version=snapshot.get(
                    "activation_resolver_policy_version"
                ),
                effective_at=snapshot.get("effective_at"),
                rule_versions=[
                    dict(
                        rule_version_id=r["rule_version_id"],
                        content_hash=r["content_hash"],
                    )
                    for r in rules
                ],
            )
            return rules, "AVAILABLE", snapshot["activation_set_snapshot_id"], pins
        except ValueError as exc:
            if str(exc) != "ACTIVATION_SNAPSHOT_NOT_MATERIALIZED":
                raise
            return unavailable

    def create(
        self,
        *,
        document_ref,
        filename,
        review_id=None,
        version_id="v1",
        basis_documents=None,
        successor=None,
    ):
        if (
            Path(document_ref).is_absolute()
            or not document_ref.startswith(("uploads/", "artifacts/governed/"))
            or ".." in Path(document_ref).parts
        ):
            raise PermissionError("FORBIDDEN_SOURCE")
        path = (self.bound.session_root / document_ref).resolve()
        path.relative_to(self.bound.session_root.resolve())
        if basis_documents is None:
            basis_documents = self.basis_documents()
        unavailable_basis = []
        available_basis = []
        for document in basis_documents:
            try:
                registered = self.store.get(
                    organization_id=self.context.organization_id,
                    workspace_id=self.context.workspace_id,
                    artifact_ref=document["basis_ref"],
                    contract_type=JdBasisEvidenceV1,
                )
                if (
                    registered is None
                    or registered.basis_hash != document["basis_hash"]
                ):
                    raise ValueError("STALE_BASIS_HASH")
                self._basis_source_path(registered)
                available_basis.append(document)
            except (OSError, ValueError) as exc:
                unavailable_basis.append(
                    "BASIS_SOURCE_UNAVAILABLE:"
                    + document["basis_ref"]
                    + ":"
                    + type(exc).__name__
                )
        basis_documents = available_basis
        raw = path.read_bytes()
        original_hash = sha256_bytes(raw)
        kind = Path(filename).suffix.lower().lstrip(".")
        if kind not in {"docx", "pdf", "txt", "md"}:
            raise ValueError("UNSUPPORTED_DOCUMENT_KIND")
        display_ref = document_ref
        projection = None
        if kind in {"txt", "md"}:
            text = raw.decode("utf-8", errors="strict")
            projected = text_projection(text)
            display_ref = (
                "artifacts/governed/display-" + original_hash.split(":")[1] + ".docx"
            )
            target = self.bound.session_root / display_ref
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with target.open("xb") as out:
                    out.write(projected)
            except FileExistsError:
                if target.read_bytes() != projected:
                    raise ValueError("PROJECTION_CONFLICT")
            projection = {
                "kind": "DOCX_DISPLAY_PROJECTION",
                "original_source_hash": original_hash,
                "original_document_ref": document_ref,
                "paragraph_map": [
                    {"original_line": i, "display_paragraph": i}
                    for i, _ in enumerate(text.splitlines())
                ],
            }
        extracted = extract_live(
            {"document_ref": display_ref, "allow_ocr": False},
            trusted_session=self.bound,
        )
        document = extracted["document"]
        artifacts = extracted["artifacts"]
        if not document.get("elements"):
            raise ValueError("NO_POSITIONED_TEXT_OR_SCANNED_DOCUMENT")
        display_kind = "pdf" if kind == "pdf" else "docx"
        document = self._segment_document(document)
        from types import SimpleNamespace

        parsing_evidence = InMemoryKnowledgeParsingEvidenceService()
        parsed = parse_knowledge_source(
            args=dict(
                source_kind=display_kind,
                canonical_document=document,
                source_ref=display_ref,
                source_hash=artifacts["source_hash"],
                parser_profile_version=display_kind + ".v1",
                normalized_artifact_ref=artifacts["normalized_document_artifact_ref"],
                normalized_artifact_hash=artifacts["normalized_document_artifact_hash"],
            ),
            ctx=SimpleNamespace(
                organization_id=self.context.organization_id,
                workspace_id=self.context.workspace_id,
                user_id=self.context.actor_id,
                member_role=self.context.actor_role,
            ),
            parsing_evidence_service=parsing_evidence,
        )
        canonical = parsing_evidence.get_canonical_classification_input(
            parsed["canonical_input"]["input_id"], self.context
        )
        review_id = review_id or "jdrv_" + __import__("uuid").uuid4().hex
        identity_ref = "jd-source:" + review_id
        findings = analyze_segments(
            canonical,
            source_identity_ref=identity_ref,
            document_kind=display_kind,
            basis_documents=basis_documents or [],
        )
        rules, basis_status, resolver_ref, classification_basis = self._rules()
        classified = classify_canonical_input(
            canonical_input=canonical,
            active_rules=rules,
            resolver_evidence_ref=resolver_ref,
        )
        nodes = []
        edges = []
        for segment in canonical.segments:
            requirement_context = bool(
                re.match(r"^\s*(?:\d+[.、]\s*)?(?:岗位)?要求[:：]", segment.text)
            )
            for index, match in enumerate(
                re.finditer(r"[^，,；;。\n]+", segment.text), 1
            ):
                clause = match.group()
                if not clause.strip():
                    continue
                node = dict(
                    id="jdr_" + segment.segment_id + ":clause:" + str(index),
                    kind="requirement",
                    label=clause,
                    text=clause,
                    anchor_ref=segment.source_anchor_ref,
                    segment_id=segment.segment_id,
                    start_offset=match.start(),
                    end_offset=match.end(),
                    requirement_level="PREFERRED"
                    if "优先" in clause
                    else "NECESSARY"
                    if requirement_context or re.search(r"必须|必要|要求", clause)
                    else "UNSPECIFIED",
                    authority_state="candidate",
                )
                nodes.append(node)
        for finding in findings:
            start = (
                finding.proposed_edit["start_offset"]
                if finding.proposed_edit
                else next(
                    (
                        s.text.find(finding.evidence_ref["selector"]["exactQuote"])
                        for s in canonical.segments
                        if s.source_anchor_ref == finding.anchor_ref
                    ),
                    -1,
                )
            )
            owner = next(
                (
                    n
                    for n in nodes
                    if n["anchor_ref"] == finding.anchor_ref
                    and n["start_offset"] <= start < n["end_offset"]
                ),
                None,
            )
            if owner:
                finding.requirement_id = owner["id"]
                edges.append(
                    dict(
                        id="jde_" + finding.finding_id,
                        source=owner["id"],
                        target=finding.finding_id,
                        type="has_candidate_finding",
                        authority_role="evidence_only",
                    )
                )
            nodes.append(
                dict(
                    id=finding.finding_id,
                    kind="finding",
                    label=finding.summary,
                    text=finding.summary,
                    anchor_ref=finding.anchor_ref,
                    authority_state="candidate",
                )
            )
        graph_payload = {
            "nodes": nodes,
            "edges": edges,
            "source_hash": canonical.source_hash,
            "canonical_input_id": canonical.input_id,
        }
        graph_meta = self._checkpoint(review_id, graph_payload, canonical.source_hash)
        now = datetime.now(timezone.utc)
        identity = dict(
            schemaVersion="semantier.source_identity.v1",
            sourceIdentityRef=identity_ref,
            organizationId=self.context.organization_id,
            workspaceId=self.context.workspace_id,
            sourceHash=canonical.source_hash,
            mediaType="application/pdf" if kind == "pdf" else DOCX_MEDIA,
            originalName=filename
            if kind in {"docx", "pdf"}
            else filename + ".display.docx",
            artifactRef=display_ref,
            sourceVersion=version_id,
            supersedesSourceIdentityRef=successor.get("supersedes_source_identity_ref")
            if successor
            else None,
            createdAt=now.isoformat(),
        )
        payload = dict(
            schema_version="jd_review.v1",
            organization_id=self.context.organization_id,
            workspace_id=self.context.workspace_id,
            actor_id=self.context.actor_id,
            session_id=self.context.session_id,
            review_id=review_id,
            version_id=version_id,
            status="CHECKED",
            recorded_at=now.isoformat(),
            source=dict(
                identity=identity,
                source_identity_ref=identity_ref,
                source_hash=canonical.source_hash,
                original_source_hash=original_hash,
                filename=filename,
                document_kind=display_kind,
                document_version=version_id,
                artifact_ref=display_ref,
                original_artifact_ref=document_ref,
                projection=projection,
            ),
            canonical_input=asdict(canonical),
            requirements_graph={
                **graph_payload,
                "artifact_id": graph_meta.artifact_id,
                "artifact_hash": graph_meta.content_hash,
            },
            findings=[f.model_dump(mode="json") for f in findings],
            basis_documents=basis_documents or [],
            active_basis_status=basis_status,
            classification_basis=classification_basis,
            classifier_findings=[asdict(f) for f in classified.occurrences]
            + [asdict(f) for f in classified.fields],
            successor=successor,
            coverage=[
                "Born-digital document candidate checks only",
                "Legal applicability and institutional activation remain unknown",
                "No complete legal compliance claim",
                *unavailable_basis,
            ],
        )
        if successor:
            for edit in successor.get("edit_map", []):
                native = edit.pop("_native_anchor", None)
                ordinal = edit.pop("_source_order", None)
                if successor.get("format_conversion") == "PDF_TO_DOCX_DISPLAY_DRAFT":
                    candidates = [
                        s
                        for s in canonical.segments
                        if (s.layout_metadata.get("anchor") or {}).get(
                            "render_paragraph_index"
                        )
                        == ordinal
                    ]
                else:
                    fields = (
                        "body_index",
                        "table_index",
                        "row_index",
                        "cell_index",
                        "cell_paragraph_index",
                    )
                    candidates = [
                        s
                        for s in canonical.segments
                        if native is not None
                        and all(
                            (s.layout_metadata.get("anchor") or {}).get(k)
                            == native.get(k)
                            for k in fields
                        )
                    ]
                edit["new_anchor_ref"] = (
                    candidates[0].source_anchor_ref if len(candidates) == 1 else None
                )
                edit["new_source_identity_ref"] = identity_ref
                edit["new_source_hash"] = canonical.source_hash
        if successor and successor.get("requirement_preservation"):
            new_texts = [s.text for s in canonical.segments]
            preserved = []
            lost = []
            for item in successor["requirement_preservation"]:
                matched = any(
                    canonical_content_hash(text.strip()) == item["expected_text_hash"]
                    for text in new_texts
                )
                item["status"] = "RETAINED" if matched else "LOST"
                (preserved if matched else lost).extend(item["requirement_ids"])
            changed = set(successor.get("changed_requirements", []))
            successor["retained_requirements"] = [
                ref for ref in preserved if ref not in changed
            ]
            successor["lost_requirements"] = lost
            successor["unresolved_findings"] = [f.finding_id for f in findings]
            if lost:
                payload["status"] = "PARTIAL"
                successor["recheck"]["status"] = "FACT_PRESERVATION_FAILED"
            payload["successor"] = successor
        payload["result_hash"] = "pending"
        normalized = JdReviewV1.model_construct(
            **{**payload, "recorded_at": now, "findings": findings}
        ).model_dump(mode="json", exclude={"result_hash"})
        payload["result_hash"] = canonical_content_hash(normalized)
        result = JdReviewV1.model_validate(payload)
        self.store.put(result)
        return result

    @staticmethod
    def _segment_document(document):
        """Expose parser-owned table-cell paragraphs; never locate by text search."""
        elements = []
        for element in document.get("elements", []):
            if element.get("kind") != "table":
                elements.append(element)
                continue
            for row in element.get("rows", []):
                for cell in row.get("cells", []):
                    for pi, text in enumerate(
                        cell.get("paragraphs") or cell["text"].split("\n"), 1
                    ):
                        if not text:
                            continue
                        anchor = {
                            **element.get("anchor", {}),
                            "element_kind": "table_cell",
                            "row_index": cell["row"],
                            "cell_index": cell["column"],
                            "cell_paragraph_index": pi,
                        }
                        elements.append(
                            dict(
                                id=cell["element_id"] + ".p" + str(pi),
                                kind="table_cell_paragraph",
                                text=text,
                                anchor=anchor,
                                local_text_hash=canonical_content_hash(text),
                                structure=cell.get("structure", {}),
                            )
                        )
        return {**document, "elements": elements}

    def _repository(self):
        return ContextGraphArtifactRepository(
            self.workspace_root,
            organization_id=self.context.organization_id,
            workspace_id=self.context.workspace_id,
            domain_id="jd-review",
        )

    def _checkpoint(self, review_id, payload, source_hash):
        from semantier.knowledge_feature_packs.semantica_adapter import (
            _ensure_vendored_semantica_importable,
            SEMANTICA_COMMIT,
        )

        _ensure_vendored_semantica_importable()
        from semantica.context import ContextGraph

        graph = ContextGraph(advanced_analytics=False)
        graph.graph_id = review_id
        graph.add_nodes(
            [
                dict(
                    id=n["id"],
                    type=n["kind"],
                    content=n["text"],
                    properties={k: v for k, v in n.items() if k not in {"id", "text"}},
                )
                for n in payload["nodes"]
            ]
        )
        graph.add_edges(
            [
                dict(
                    source=e["source"],
                    target=e["target"],
                    type=e["type"],
                    properties={"authority_role": "evidence_only"},
                )
                for e in payload["edges"]
            ]
        )
        return self._repository().create_checkpoint(
            graph,
            lineage_id=review_id,
            graph_id=review_id,
            graph_version=review_id + "-candidate-v1",
            artifact_kind="candidate",
            semantica_commit=SEMANTICA_COMMIT,
            source_manifest_hash=source_hash,
            created_by_user_id=self.context.actor_id,
        )

    def get_graph(self, review_id):
        review = self.get(review_id)
        self._access(review)
        repository = self._repository()
        metadata = repository.get(review.requirements_graph["artifact_id"])
        if (
            metadata.content_hash != review.requirements_graph["artifact_hash"]
            or metadata.created_by_user_id != self.context.actor_id
        ):
            raise ValueError("STALE_GRAPH_PIN")
        from semantier.knowledge_feature_packs.semantica_adapter import (
            _ensure_vendored_semantica_importable,
        )

        _ensure_vendored_semantica_importable()
        return repository.load_graph(metadata.artifact_id)

    def source_path(self, review_id):
        review = self.get(review_id)
        path = (self.bound.session_root / review.source["artifact_ref"]).resolve()
        path.relative_to(self.bound.session_root.resolve())
        if sha256_bytes(path.read_bytes()) != review.source["source_hash"]:
            raise ValueError("STALE_SOURCE_HASH")
        original = (
            self.bound.session_root / review.source["original_artifact_ref"]
        ).resolve()
        original.relative_to(self.bound.session_root.resolve())
        if sha256_bytes(original.read_bytes()) != review.source["original_source_hash"]:
            raise ValueError("STALE_ORIGINAL_SOURCE_HASH")
        return path

    def basis_documents(self):
        documents = []
        for pin in self.store.list_artifact_pins(
            organization_id=self.context.organization_id,
            workspace_id=self.context.workspace_id,
            schema_version="jd_basis_evidence.v1",
        ):
            basis = self.store.get(
                organization_id=self.context.organization_id,
                workspace_id=self.context.workspace_id,
                artifact_ref=pin.artifact_ref,
                contract_type=JdBasisEvidenceV1,
            )
            if (basis.actor_id, basis.session_id) != (
                self.context.actor_id,
                self.context.session_id,
            ):
                continue
            documents.append(
                {
                    **basis.document,
                    "basis_ref": basis.basis_ref,
                    "basis_hash": basis.basis_hash,
                    "source_id": basis.source_id,
                    "activation_status": basis.activation_status,
                }
            )
        superseded = {document.get("supersedes_basis_ref") for document in documents}
        return [
            document
            for document in documents
            if document["basis_ref"] not in superseded
        ]

    def register_basis(self, *, source_id, document):
        """Explicit offline provisioning only; this does not admit or activate rules."""
        from contracts.evidence_location import SourceIdentity, EvidenceRef

        predecessor_ref = document.get("supersedes_basis_ref")
        if predecessor_ref:
            predecessor = self.store.get(
                organization_id=self.context.organization_id,
                workspace_id=self.context.workspace_id,
                artifact_ref=predecessor_ref,
                contract_type=JdBasisEvidenceV1,
            )
            if predecessor is None:
                raise ValueError("BASIS_PREDECESSOR_NOT_FOUND")
            if (
                predecessor.actor_id,
                predecessor.session_id,
                predecessor.source_id,
            ) != (self.context.actor_id, self.context.session_id, source_id):
                raise PermissionError("FORBIDDEN_BASIS_SCOPE")
        source = document["source"]
        identity = SourceIdentity.model_validate(source["identity"])
        if (identity.organization_id, identity.workspace_id) != (
            self.context.organization_id,
            self.context.workspace_id,
        ):
            raise PermissionError("FORBIDDEN_BASIS_SCOPE")
        if (
            identity.source_identity_ref != source["source_identity_ref"]
            or identity.source_hash != source["source_hash"]
        ):
            raise ValueError("BASIS_IDENTITY_MISMATCH")
        path = (self.bound.session_root / source["artifact_ref"]).resolve()
        path.relative_to(self.bound.session_root.resolve())
        if sha256_bytes(path.read_bytes()) != source["source_hash"]:
            raise ValueError("STALE_BASIS_HASH")
        for clause in document.get("clauses", []):
            evidence = EvidenceRef.model_validate(clause["evidence_ref"])
            if (
                evidence.source_identity_ref != identity.source_identity_ref
                or evidence.provenance.get("source_hash") != identity.source_hash
            ):
                raise ValueError("BASIS_EVIDENCE_MISMATCH")
        payload = dict(
            schema_version="jd_basis_evidence.v1",
            organization_id=self.context.organization_id,
            workspace_id=self.context.workspace_id,
            actor_id=self.context.actor_id,
            session_id=self.context.session_id,
            basis_ref="jd-basis:"
            + canonical_content_hash(
                [self.context.actor_id, self.context.session_id, source_id, document]
            ).split(":")[1],
            source_id=source_id,
            document=document,
            activation_status="NOT_ACTIVATED",
        )
        payload["basis_hash"] = canonical_content_hash(payload)
        basis = JdBasisEvidenceV1.model_validate(payload)
        self.store.put(basis)
        return basis

    def _basis_source_path(self, basis):
        if (
            basis.organization_id,
            basis.workspace_id,
            basis.actor_id,
            basis.session_id,
        ) != (
            self.context.organization_id,
            self.context.workspace_id,
            self.context.actor_id,
            self.context.session_id,
        ):
            raise PermissionError("FORBIDDEN_BASIS_SCOPE")
        source = basis.document["source"]
        path = (self.bound.session_root / source["artifact_ref"]).resolve()
        path.relative_to(self.bound.session_root.resolve())
        if sha256_bytes(path.read_bytes()) != source["source_hash"]:
            raise ValueError("STALE_BASIS_HASH")
        return path

    def successor(self, review_id, command):
        from services.governed_artifact_service import (
            StoreBackedGovernedArtifactService,
        )

        parent = self.get(review_id)
        if (command.result_hash, command.source_hash, command.version_id) != (
            parent.result_hash,
            parent.source["source_hash"],
            parent.version_id,
        ):
            raise ValueError("STALE_REVIEW_PIN")
        # Retry only the stored, owner-scoped predecessor; the client cannot supply edits.
        retry_of_review_id = None
        if parent.status not in {"CHECKED", "PARTIAL"} and not command.cancel:
            self.source_path(parent.review_id)
            retry_of_review_id = parent.review_id
            predecessor = (parent.successor or {}).get("supersedes_review_id")
            if not predecessor:
                raise ValueError("INCOMPLETE_REVIEW")
            parent = self.get(predecessor)
            if parent.status not in {"CHECKED", "PARTIAL"}:
                raise ValueError("INCOMPLETE_REVIEW")
        # Validate pinned bytes before an idempotent replay as well.
        source = self.source_path(parent.review_id)
        current_basis_refs = {basis["basis_ref"] for basis in self.basis_documents()}
        for basis in parent.basis_documents:
            if basis["basis_ref"] not in current_basis_refs:
                raise ValueError("STALE_BASIS_VERSION")
            registered = self.store.get(
                organization_id=self.context.organization_id,
                workspace_id=self.context.workspace_id,
                artifact_ref=basis["basis_ref"],
                contract_type=JdBasisEvidenceV1,
            )
            if registered is None or registered.basis_hash != basis["basis_hash"]:
                raise ValueError("STALE_BASIS_HASH")
            self._basis_source_path(registered)
        request_hash = canonical_content_hash(command.model_dump(mode="json"))
        child_id = (
            "jdrv_"
            + canonical_content_hash([review_id, command.request_id]).split(":")[1][:32]
        )
        existing = self.store.get(
            organization_id=self.context.organization_id,
            workspace_id=self.context.workspace_id,
            artifact_ref=child_id,
            contract_type=JdReviewV1,
        )
        if existing is not None:
            self._access(existing)
            opposite_cancel_hash = canonical_content_hash(
                command.model_copy(update={"cancel": not command.cancel}).model_dump(
                    mode="json"
                )
            )
            if existing.successor.get("request_hash") not in {
                request_hash,
                opposite_cancel_hash,
            }:
                raise ValueError("REQUEST_CONFLICT")
            # The immutable terminal receipt wins either ordering of cancel/generate.
            return existing
        if command.cancel:
            return self._record_outcome(
                parent,
                child_id,
                "CANCELLED",
                command.request_id,
                request_hash=request_hash,
            )
        edits = [
            f
            for f in parent.findings
            if f.proposed_edit
            and f.presentation_location
            and f.presentation_location.get("kind")
            in {"docx_structural_range", "pdf_page_rect"}
        ]
        raw = source.read_bytes()
        ref = None
        generated_artifact = None
        try:
            draft_bytes = self._draft_bytes(parent, edits, raw)
            ref = "artifacts/governed/" + child_id + ".docx"
            generated_artifact = StoreBackedGovernedArtifactService(
                trusted_session=self.bound
            ).create_derivative_artifact(
                parent_ref=parent.source["artifact_ref"],
                media_type=DOCX_MEDIA,
                bytes_or_stream=draft_bytes,
                metadata={
                    "artifact_id": child_id,
                    "artifact_ref": ref,
                    "request_id": command.request_id,
                    "parent_result_hash": parent.result_hash,
                    "proposal_only": True,
                },
                context=self.context,
            )
            retained = [n["id"] for n in parent.requirements_graph["nodes"]]
            successor = dict(
                lineage_review_id=(parent.successor or {}).get(
                    "lineage_review_id", parent.review_id
                ),
                supersedes_review_id=parent.review_id,
                supersedes_version_id=parent.version_id,
                supersedes_source_identity_ref=parent.source["source_identity_ref"],
                request_id=command.request_id,
                request_hash=request_hash,
                retry_of_review_id=retry_of_review_id,
                edit_map=[
                    dict(
                        finding_id=f.finding_id,
                        old_anchor_ref=f.anchor_ref,
                        old_source_identity_ref=parent.source["source_identity_ref"],
                        old_source_hash=parent.source["source_hash"],
                        _native_anchor=next(
                            s["layout_metadata"].get("anchor")
                            for s in parent.canonical_input["segments"]
                            if s["segment_id"] == f.proposed_edit["segment_id"]
                        ),
                        _source_order=next(
                            s["order_index"] - 1
                            for s in parent.canonical_input["segments"]
                            if s["segment_id"] == f.proposed_edit["segment_id"]
                        ),
                        old_occurrence_ref=f.occurrence_ref,
                        new_anchor_ref=f.anchor_ref,
                        new_occurrence_ref=None,
                        change_type="PROPOSED_REMOVAL",
                        proposal_only=True,
                    )
                    for f in edits
                ],
                recheck={"status": "CHECKED", "version_id": child_id},
                retained_requirements=retained,
                changed_requirements=sorted({f.requirement_id for f in edits}),
                lost_requirements=[],
                unresolved_findings=[
                    f.finding_id for f in parent.findings if f not in edits
                ],
            )
            preservation = []
            for segment in parent.canonical_input["segments"]:
                text = segment["text"]
                for finding in sorted(
                    [
                        f
                        for f in edits
                        if f.proposed_edit["segment_id"] == segment["segment_id"]
                    ],
                    key=lambda f: f.proposed_edit["start_offset"],
                    reverse=True,
                ):
                    edit = finding.proposed_edit
                    text = (
                        text[: edit["start_offset"]]
                        + edit["replacement"]
                        + text[edit["end_offset"] :]
                    )
                preservation.append(
                    dict(
                        requirement_ids=[
                            n["id"]
                            for n in parent.requirements_graph["nodes"]
                            if n.get("segment_id") == segment["segment_id"]
                        ],
                        expected_text_hash=canonical_content_hash(text.strip()),
                    )
                )
            successor["requirement_preservation"] = preservation
            successor["format_conversion"] = (
                "PDF_TO_DOCX_DISPLAY_DRAFT"
                if parent.source["document_kind"] == "pdf"
                else None
            )
            return self.create(
                document_ref=ref,
                filename=Path(parent.source["filename"]).stem + ".draft.docx",
                review_id=child_id,
                version_id=child_id,
                basis_documents=parent.basis_documents,
                successor=successor,
            )
        except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
            return self._record_outcome(
                parent,
                child_id,
                "RECHECK_FAILED",
                command.request_id,
                type(exc).__name__,
                generated_ref=generated_artifact.artifact_ref
                if generated_artifact is not None
                else None,
                generated_hash=generated_artifact.content_hash
                if generated_artifact is not None
                else None,
                request_hash=request_hash,
            )

    def _draft_bytes(self, parent, edits, raw):
        if parent.source["document_kind"] == "pdf":
            lines = []
            for segment in parent.canonical_input["segments"]:
                text = segment["text"]
                for finding in sorted(
                    [
                        f
                        for f in edits
                        if f.proposed_edit["segment_id"] == segment["segment_id"]
                    ],
                    key=lambda f: f.proposed_edit["start_offset"],
                    reverse=True,
                ):
                    edit = finding.proposed_edit
                    text = (
                        text[: edit["start_offset"]]
                        + edit["replacement"]
                        + text[edit["end_offset"] :]
                    )
                lines.append(text)
            return text_projection("\n".join(lines))
        with zipfile.ZipFile(io.BytesIO(raw)) as package:
            xml = ET.fromstring(package.read("word/document.xml"))
            body = xml.find(".//" + W + "body")
            grouped = {}
            for finding in edits:
                edit = finding.proposed_edit
                segment = next(
                    s
                    for s in parent.canonical_input["segments"]
                    if s["segment_id"] == edit["segment_id"]
                )
                anchor = segment["layout_metadata"]["anchor"]
                key = (
                    anchor["body_index"] - 1,
                    anchor.get("row_index"),
                    anchor.get("cell_index"),
                    anchor.get("cell_paragraph_index"),
                )
                grouped.setdefault(key, []).append(finding)
            for (body_index, row_index, cell_index, pi), items in grouped.items():
                node = list(body)[body_index]
                if row_index is not None:
                    node = (
                        node.findall(W + "tr")[row_index - 1]
                        .findall(W + "tc")[cell_index - 1]
                        .findall(W + "p")[pi - 1]
                    )
                from plugins.document_extraction.parsers.ooxml import paragraph_text

                text = paragraph_text(node)
                for finding in sorted(
                    items, key=lambda f: f.proposed_edit["start_offset"], reverse=True
                ):
                    edit = finding.proposed_edit
                    a, b = edit["start_offset"], edit["end_offset"]
                    if text[a:b] != finding.evidence_ref["selector"]["exactQuote"]:
                        raise ValueError("EDIT_ANCHOR_MISMATCH")
                    text = text[:a] + edit["replacement"] + text[b:]
                for child in list(node):
                    if child.tag != W + "pPr":
                        node.remove(child)
                run = ET.SubElement(node, W + "r")
                for index, line in enumerate(text.split("\n")):
                    if index:
                        ET.SubElement(run, W + "br")
                    item = ET.SubElement(run, W + "t")
                    item.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                    item.text = line
            output = io.BytesIO()
            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as target:
                for item in package.infolist():
                    target.writestr(
                        item,
                        ET.tostring(xml, encoding="utf-8", xml_declaration=True)
                        if item.filename == "word/document.xml"
                        else package.read(item.filename),
                    )
        return output.getvalue()

    def _record_outcome(
        self,
        parent,
        review_id,
        status,
        request_id,
        reason=None,
        generated_ref=None,
        generated_hash=None,
        request_hash=None,
    ):
        payload = parent.model_dump(mode="json", exclude={"result_hash"})
        payload.update(
            review_id=review_id,
            version_id=review_id,
            status=status,
            recorded_at=datetime.now(timezone.utc).isoformat(),
        )
        payload["successor"] = dict(
            lineage_review_id=(parent.successor or {}).get(
                "lineage_review_id", parent.review_id
            ),
            supersedes_review_id=parent.review_id,
            supersedes_version_id=parent.version_id,
            request_id=request_id,
            request_hash=request_hash,
            edit_map=[],
            recheck={
                "status": "FAILED" if status != "CANCELLED" else "CANCELLED",
                "reason": reason,
            },
            retained_requirements=[],
            lost_requirements=[],
            unresolved_findings=[f.finding_id for f in parent.findings],
        )
        if (
            generated_ref
            and generated_hash
            and (self.bound.session_root / generated_ref).is_file()
            and sha256_bytes((self.bound.session_root / generated_ref).read_bytes())
            == generated_hash
        ):
            digest = sha256_bytes(
                (self.bound.session_root / generated_ref).read_bytes()
            )
            source = dict(parent.source)
            identity = {
                **source["identity"],
                "sourceIdentityRef": "jd-source:" + review_id,
                "sourceHash": digest,
                "artifactRef": generated_ref,
                "sourceVersion": review_id,
                "mediaType": DOCX_MEDIA,
                "originalName": Path(source["filename"]).stem + ".draft.docx",
                "supersedesSourceIdentityRef": parent.source["source_identity_ref"],
            }
            source.update(
                identity=identity,
                source_identity_ref=identity["sourceIdentityRef"],
                source_hash=digest,
                original_source_hash=digest,
                artifact_ref=generated_ref,
                original_artifact_ref=generated_ref,
                document_kind="docx",
                document_version=review_id,
                filename=identity["originalName"],
                projection=None,
            )
            payload.update(
                source=source,
                version_id=review_id,
                canonical_input={},
                requirements_graph={"nodes": [], "edges": []},
                findings=[],
                classifier_findings=[],
            )
            payload["successor"]["generated_artifact_ref"] = generated_ref
            payload["successor"]["generation_status"] = "COMPLETE"
        payload["recorded_at"] = (
            datetime.fromisoformat(payload["recorded_at"])
            .isoformat()
            .replace("+00:00", "Z")
        )
        payload["result_hash"] = canonical_content_hash(payload)
        outcome = JdReviewV1.model_validate(payload)
        self.store.put(outcome)
        return outcome

    def feedback(self, review_id, command):
        from contracts.jd_compliance_review import JdFeedbackV1

        review = self.get(review_id)
        if (review.version_id, review.result_hash) != (
            command.version_id,
            command.result_hash,
        ):
            raise ValueError("STALE_REVIEW_PIN")
        if command.finding_id not in {f.finding_id for f in review.findings}:
            raise ValueError("FINDING_NOT_IN_REVIEW")
        ref = (
            "jd-feedback:"
            + canonical_content_hash(
                [review_id, self.context.actor_id, command.request_id]
            ).split(":")[1]
        )
        existing = self.store.get(
            organization_id=self.context.organization_id,
            workspace_id=self.context.workspace_id,
            artifact_ref=ref,
            contract_type=JdFeedbackV1,
        )
        if existing:
            if any(
                getattr(existing, key) != getattr(command, key)
                for key in type(command).model_fields
            ):
                raise ValueError("REQUEST_CONFLICT")
            return existing
        payload = dict(
            **command.model_dump(mode="json"),
            schema_version="jd_feedback.v1",
            organization_id=self.context.organization_id,
            workspace_id=self.context.workspace_id,
            actor_id=self.context.actor_id,
            session_id=self.context.session_id,
            feedback_ref=ref,
            review_id=review_id,
            recorded_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            authority_role="NON_AUTHORITATIVE_USER_FEEDBACK",
        )
        payload["feedback_hash"] = canonical_content_hash(payload)
        feedback = JdFeedbackV1.model_validate(payload)
        self.store.put(feedback)
        return feedback
