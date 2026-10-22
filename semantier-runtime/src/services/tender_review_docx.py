"""Deterministic native-report projection onto retained original DOCX bytes."""
from hashlib import sha256
from io import BytesIO
import zipfile
from xml.etree import ElementTree

from contracts.canonical_source_ir import CanonicalSourceDocument
from contracts.evidence_location import EvidenceRef
from contracts.tender_legal_compliance import TenderFrozenComplianceDecisionV1
from contracts.tender_review import TenderReviewReportV1
from eos.artifact_hashing import content_hash
from plugins.tender_sensitive_term_detection_modeling.docx_output import write_labeled_docx_bytes
from semantier.knowledge_feature_packs.evidence_resolver import resolve_evidence_ref, lineage_for_evidence_ref

NATIVE_DOCX_POLICY = "tender_native_report_docx.v1"
CONCISE_NATIVE_DOCX_POLICY = "tender_native_report_docx.v2"
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def render_native_report_docx(*, report, document, source_bytes):
    """Use frozen decisions only, without legacy detection-run membership."""
    report = TenderReviewReportV1.model_validate(report.model_dump(mode="json"))
    document = CanonicalSourceDocument.model_validate(document.model_dump(mode="json", by_alias=True))
    if report.episode_pin is None or report.native_basis_snapshot is None:
        raise ValueError("NATIVE_REPORT_REQUIRED")
    if (document.source_identity_ref != report.source_identity_ref
        or "sha256:" + sha256(source_bytes).hexdigest() != report.source_hash
        or "sha256:" + content_hash(document.model_dump(mode="json", by_alias=True, exclude_none=False))
            != report.canonical_source_ir_pin.artifact_hash):
        raise ValueError("REPORT_SOURCE_PIN_MISMATCH")
    entries = tuple(TenderFrozenComplianceDecisionV1.model_validate(raw)
                    for raw in report.native_basis_snapshot.get("compliance_decisions", ()))
    if not entries:
        raise ValueError("FROZEN_COMPLIANCE_DECISIONS_REQUIRED")
    policy = report.native_basis_snapshot.get("docx_projection_policy", NATIVE_DOCX_POLICY)
    if policy not in {NATIVE_DOCX_POLICY, CONCISE_NATIVE_DOCX_POLICY}:
        raise ValueError("NATIVE_DOCX_POLICY_UNSUPPORTED")
    selections = report.native_basis_snapshot.get("compliance_selection", ())
    # Only explicit successor edges change legacy annotation selection. Old
    # snapshots without that field retain their original byte-level renderer.
    superseded = {pin['artifact_ref'] for item in selections for pin in item.get('superseded', ())}
    entries = tuple(entry for entry in entries if entry.decision.artifact_ref not in superseded)
    blocks = {block.block_id: block for block in document.blocks}
    elements, findings, unsupported = [], [], []
    with zipfile.ZipFile(BytesIO(source_bytes)) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
        existing_comments = "word/comments.xml" in archive.namelist() or root.find(".//" + _W + "commentReference") is not None
        body = root.find(_W + "body")
        body_nodes = [] if body is None else [node for node in body if node.tag in {_W + "p", _W + "tbl"}]

    for entry in entries:
        basis, decision = entry.basis, entry.decision
        if (basis.target_ir != report.canonical_source_ir_pin
            or basis.finding.source_hash != report.source_hash
            or basis.finding.source_identity_ref != report.source_identity_ref):
            raise ValueError("REPORT_COMPLIANCE_SOURCE_MISMATCH")
        source_text = "Normative source/article/version: unavailable; qualified binding missing."
        if basis.normative is not None:
            source_text = "\n".join(
                f"Source: {p.source_title}; article: {p.article_locator}; version: {p.version_label} "
                f"({p.version_id}); original: {p.identity.source_identity_ref} {p.identity.source_hash}; "
                f"quote: {p.evidence.selector.exact_quote}"
                for p in basis.normative.provisions)
        if basis.advisory is not None:
            a = basis.advisory
            p = a.proposal
            source_text = "\n".join((
                "OWNER_ADOPTED_INTERNAL_ADVISORY; nonbinding internal assistance; independent_legal_validation=NOT_ESTABLISHED",
                f"Owner: {a.adoption.actor_id}; adopted: {a.adoption.adopted_at.isoformat()}; policy: {p.policy_ref} {p.policy_hash}",
                f"Source decision: {a.source_decision.artifact_ref}; interpretation decision: {a.interpretation_decision.artifact_ref}",
                f"Rule: {p.rule_ref}; plan: {p.plan.artifact_ref} {p.plan.artifact_hash}",
                "Conditions / exceptions / precedence: " + a.plan.model_dump_json(),
                "Unresolved: " + ", ".join((*p.unknowns, *basis.gaps)),
                *(f"Source: {identity.original_name}; version: {citation['version']}; original: {identity.source_identity_ref} {identity.source_hash}; quote: {evidence.selector.exact_quote}"
                  for identity, ir, evidence, citation in basis.original_provisions())))
        if policy == CONCISE_NATIVE_DOCX_POLICY and basis.advisory is not None:
            # Full conditions, source quotes and compiled plan remain frozen in
            # JSON. Every occurrence still carries a readable source and reason.
            source_groups = {}
            for identity, ir, evidence, citation in basis.original_provisions():
                key = (identity.source_identity_ref, identity.source_hash, citation['version'])
                if key not in source_groups:
                    source_groups[key] = [identity, citation, 0]
                source_groups[key][2] += 1
            source_text = "\n".join((
                "OWNER_ADOPTED_INTERNAL_ADVISORY; nonbinding internal assistance; independent_legal_validation=NOT_ESTABLISHED",
                f"Outcome: {entry.verification.outcome}; rule: {p.rule_ref}; plan: {p.plan.artifact_ref} {p.plan.artifact_hash}",
                "Unresolved: " + ", ".join((*p.unknowns, *basis.gaps)),
                *(f"Source: {identity.original_name}; article/location: {citation['article']}; version: {citation['version']}; original: {identity.source_identity_ref} {identity.source_hash}; source spans: {count} (complete inventory and quotations in frozen JSON)"
                  for identity, citation, count in source_groups.values())))
        selections = report.native_basis_snapshot.get("compliance_selection", ())
        selection = next((item for item in selections
                          if item["finding"] == decision.subject.model_dump(mode="json")), None)
        selection_lines = ("Assessment selection: " + selection["status"],) if selection else ()
        reason = "\n".join((f"Decision: {decision.artifact_ref} {decision.artifact_hash}",
            f"Report: {report.report_ref} {report.report_hash}", *selection_lines,
            "Reason: " + ", ".join(entry.verification.diagnostics), source_text,
            "Scoped evaluation; unresolved inputs do not establish a legal violation."))
        for raw in basis.finding.model_dump(mode="json", by_alias=True)["evidenceRefs"]:
            evidence = EvidenceRef.model_validate(raw)
            occurrence_id = decision.artifact_ref + ":" + evidence.evidence_ref
            def reject(code):
                unsupported.append(dict(finding_id=occurrence_id, decision_ref=decision.artifact_ref,
                                        evidence_ref=evidence.evidence_ref, reason=code))
            if existing_comments:
                reject("EXISTING_DOCX_COMMENTS_UNSUPPORTED")
                continue
            resolved = resolve_evidence_ref(evidence, document,
                artifact_ref=report.canonical_source_ir_pin.artifact_ref, selector_hash=evidence.selector_hash)
            if resolved.status not in {"exact", "relocated_exact"} or resolved.location.get("kind") != "docx_structural_range":
                reject("EXACT_DOCX_OCCURRENCE_REQUIRED:" + resolved.status + ":" + str(resolved.location.get("reason", "unsupported location")))
                continue
            lineage = lineage_for_evidence_ref(evidence, document)
            block = blocks[lineage.source_block_id]
            loc = lineage.source_locator
            path = resolved.location["structuralPath"]
            if loc.get("packagePart") != "word/document.xml" or not path or type(path[0]) is not int:
                reject("DOCX_LOCATION_UNSUPPORTED")
                continue
            body_index = path[0]
            node = body_nodes[body_index - 1] if 1 <= body_index <= len(body_nodes) else None
            element = dict(anchor_ref=occurrence_id, text=block.content, body_index=body_index)
            paragraph = None
            if block.block_type == "paragraph" and len(path) == 1 and node is not None and node.tag == _W + "p":
                element["kind"] = "paragraph"
                paragraph = node
            elif (block.block_type == "cell" and len(path) == 7 and path[1::2] == ["table", "row", "cell"]
                  and all(type(path[i]) is int and path[i] > 0 for i in (2, 4, 6))
                  and node is not None and node.tag == _W + "tbl"):
                tables = [item for item in body_nodes if item.tag == _W + "tbl"]
                rows = node.findall(_W + "tr")
                cells = rows[path[4]-1].findall(_W + "tc") if path[4] <= len(rows) else []
                cell = cells[path[6]-1] if path[6] <= len(cells) else None
                if path[2] <= len(tables) and tables[path[2]-1] is node and cell is not None:
                    paragraphs = cell.findall(_W + "p")
                    if len(paragraphs) == 1 and not cell.findall(_W + "tbl"):
                        paragraph = paragraphs[0]
                element.update(kind="table_cell", table_index=path[2], row_index=path[4], cell_index=path[6])
            # The shared writer splits plain runs. Do not erase fields, hyperlinks,
            # tracked changes, tabs, breaks or multi-paragraph cell content.
            if (paragraph is None or any(child.tag not in {_W+"pPr", _W+"r"} for child in paragraph)
                or any(part.tag not in {_W+"rPr", _W+"t"} for run in paragraph.findall(_W+"r") for part in run)):
                reject("DOCX_LOCATION_OR_MARKUP_UNSUPPORTED")
                continue
            elements.append(element)
            findings.append(dict(finding_id=occurrence_id, target_anchor_ref=occurrence_id,
                matched_text=resolved.location["exactText"], start_offset=resolved.location["startOffset"],
                end_offset=resolved.location["endOffset"], issue_type=decision.result_code,
                judgment_basis=reason))
    if findings:
        rendered, manifest = write_labeled_docx_bytes(source_docx_bytes=source_bytes,
            canonical_document={"elements": elements}, findings=findings, created_by=report.recorded_by,
            created_at=report.recorded_at.isoformat())
    else:
        rendered, manifest = source_bytes, dict(comments_written=0, unsupported_findings=[], comment_ranges=[])
    manifest.update(native_projection_policy=policy, report_pin=report.pin().model_dump(mode="json"),
                    decision_pins=[entry.decision.pin().model_dump(mode="json") for entry in entries])
    manifest["unsupported_findings"] = unsupported + manifest["unsupported_findings"]
    return rendered, manifest
