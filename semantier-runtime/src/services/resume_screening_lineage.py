"""Pinned fictional teaching profile over the existing deterministic evaluator.

Associations are reviewed proposals; textual support is never verified competence.
This adapter accepts no inferred general resume interpretation or hiring verdict.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from contracts.ontology_validation import CompiledPredicateV1
from services.deterministic_rule_evaluator import DeterministicRuleEvaluator


def digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_teaching_basis() -> dict:
    return json.loads(
        (
            Path(__file__).parents[1] / "assets/workshop/resume-teaching-basis-v1.json"
        ).read_text()
    )


def exact_location(text: str, quote: str, occurrence: int | None = None) -> dict | None:
    if not quote:
        return None
    starts = []
    cursor = 0
    while (found := text.find(quote, cursor)) >= 0:
        starts.append(found)
        cursor = found + len(quote)
    if not starts or (occurrence is None and len(starts) != 1):
        return None
    selected = 0 if occurrence is None else occurrence
    if selected < 0 or selected >= len(starts):
        return None
    start = starts[selected]
    paragraph = text[:start].count("\n")
    offset = start - (text.rfind("\n", 0, start) + 1)
    return dict(
        presentation_location=dict(
            kind="docx_structural_range",
            structuralPath=[paragraph],
            startOffset=offset,
            endOffset=offset + len(quote),
            exactText=quote,
        ),
        quote=quote,
        start=start,
        end=start + len(quote),
        occurrence=selected,
        coordinate="unicode_codepoint",
        text_hash=digest(text),
        quote_hash=digest(quote),
    )


def union_months(intervals: list, as_of: str) -> int:
    today = date.fromisoformat(as_of)

    def month(value):
        if value == "至今":
            return today.year * 12 + today.month - 1
        year, part = value.split(".")
        if not 1 <= int(part) <= 12:
            raise ValueError("invalid month")
        return int(year) * 12 + int(part) - 1

    ranges = sorted((month(a), month(b)) for a, b in intervals)
    merged = []
    for start, end in ranges:
        if end < start or end > month("至今"):
            raise ValueError("invalid/future interval")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(b - a for a, b in merged)


def evaluate_lineage(texts: dict, basis: dict, *, source_refs: dict) -> dict:
    hashes = {role: digest(texts[role]) for role in ("jd", "resume")}
    intact = hashes == basis["source_text_hashes"]
    nodes = []
    edges = []
    requirements = []

    def node(id, kind, label, **values):
        nodes.append(dict(id=id, kind=kind, label=label, **values))

    for requirement in basis["requirements"]:
        rid = requirement["id"]
        node(rid, "requirement", requirement["label"])
        terms = []
        for authored in requirement["terms"]:
            tid = authored["id"]
            evidence = []
            locations = [
                ("jd", authored["jd_quote"]),
                *(("resume", quote) for quote in authored["quotes"]),
            ]
            for index, (role, quote) in enumerate(locations):
                locator = (
                    exact_location(
                        texts[role],
                        quote,
                        occurrence=authored.get("jd_occurrence")
                        if role == "jd"
                        else None,
                    )
                    if intact
                    else None
                )
                eid = f"{tid}:evidence:{index}"
                item = dict(
                    id=eid,
                    role=role,
                    source_ref=source_refs[role],
                    source_text_hash=hashes[role],
                    location=locator,
                    resolution="exact" if locator else "unresolved",
                )
                evidence.append(item)
                node(
                    eid,
                    "evidence",
                    quote,
                    **{k: v for k, v in item.items() if k != "id"},
                )
                edges.append(dict(source=tid, target=eid, kind="grounded_in"))
            value = authored["value"]
            operator = "EQ"
            expected = True
            if authored["basis_kind"] == "computation":
                operator = basis["experience_comparator"]
                expected = basis["experience_bound_months"]
                value = union_months(basis["intervals"], basis["as_of"])
            qualified = (
                intact
                and bool(authored["quotes"])
                and all(e["location"] for e in evidence)
            )
            predicate = CompiledPredicateV1(
                predicate_id=tid,
                operator=operator,
                fact_ref=tid,
                expected_value=expected,
                ir_node_ref=tid,
                evidence_refs=tuple(e["id"] for e in evidence),
                strict_operands=True,
            )
            node(
                tid,
                "predicate",
                authored["label"],
                predicate=predicate.model_dump(mode="json"),
                observed_fact=value if qualified else None,
                condition=authored["condition"],
            )
            # The evaluator reads the typed predicate and its evidence bindings
            # from the same graph that is persisted and projected by Inspector.
            graph_predicate = nodes[-1]
            bound_ids = {
                e["target"]
                for e in edges
                if e["source"] == tid and e["kind"] == "grounded_in"
            }
            bound_evidence = [
                n for n in nodes if n["kind"] == "evidence" and n["id"] in bound_ids
            ]
            graph_value = graph_predicate["observed_fact"]
            facts = (
                {tid: graph_value}
                if graph_value is not None
                and bound_evidence
                and all(n["resolution"] == "exact" for n in bound_evidence)
                else {}
            )
            evaluation = DeterministicRuleEvaluator._evaluate_predicate(
                CompiledPredicateV1.model_validate(graph_predicate["predicate"]),
                facts,
                optional_absence_is_failure=False,
                strict_operands=True,
            )
            decision = {"SATISFIED": "SUPPORTED_BY_RESUME", "FAILED": "NOT_MET"}.get(
                evaluation.result, "UNKNOWN"
            )
            did = tid + ":decision"
            node(did, "decision", decision)
            edges.extend(
                dict(source=e["id"], target=did, kind="decision_basis")
                for e in evidence
            )
            edges.extend(
                [
                    dict(
                        source=rid,
                        target=tid,
                        kind="requires"
                        if authored["condition"] == "required"
                        else "prefers",
                    ),
                    dict(source=tid, target=did, kind="evaluated_as"),
                ]
            )
            terms.append(
                dict(
                    id=tid,
                    label=authored["label"],
                    condition=authored["condition"],
                    basis_kind=authored["basis_kind"],
                    predicate=predicate.model_dump(mode="json"),
                    evaluation=evaluation.model_dump(mode="json"),
                    decision=decision,
                    evidence=evidence,
                    verified_competence=False,
                )
            )
        required_results = [
            t["evaluation"]["result"] for t in terms if t["condition"] == "required"
        ]
        aggregate = (
            "NOT_MET"
            if "FAILED" in required_results
            else "UNKNOWN"
            if "UNKNOWN" in required_results
            else "SUPPORTED_BY_RESUME"
            if required_results
            else "PREFERENCE_ONLY"
        )
        requirements.append(
            dict(id=rid, label=requirement["label"], terms=terms, decision=aggregate)
        )
    result = dict(
        schema_version="resume_screening_lineage.v1",
        basis_id=basis["basis_id"],
        basis_hash=digest(
            json.dumps(basis, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ),
        as_of=basis["as_of"],
        source_integrity="PINNED" if intact else "CHANGED",
        source_text_hashes=hashes,
        source_refs=source_refs,
        requirements=requirements,
        graph=dict(nodes=nodes, edges=edges),
        suitability_verdict="UNAVAILABLE",
        interpretation="reviewed_teaching_basis",
        experience_interval_convention=basis["interval_convention"],
    )
    result["result_hash"] = digest(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return result


def document_projection(text: str) -> bytes:
    """Deterministic, lossless paragraph projection for the existing DOCX viewer."""
    import io
    import zipfile
    from xml.sax.saxutils import escape

    body = "".join(
        '<w:p><w:r><w:t xml:space="preserve">' + escape(line) + "</w:t></w:r></w:p>"
        for line in text.splitlines()
    )
    files = {
        "[Content_Types].xml": '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>',
        "_rels/.rels": '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>',
        "word/document.xml": '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
        + body
        + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720"/></w:sectPr></w:body></w:document>',
    }
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, '<?xml version="1.0" encoding="UTF-8"?>' + content)
    return out.getvalue()
