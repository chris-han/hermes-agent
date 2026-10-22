
from storage.hosted_postgres import sql_query,begin_write,in_transaction
"""Bounded factual screening over pinned native JD graphs and parser evidence.

No filename heuristics, teaching profiles, legal activation or hiring scores.
"""

from datetime import date, datetime, timezone
import re
from types import SimpleNamespace
from contracts.hr_workbench import HrJobModelV1, HrScreeningRunV1, HrSourceBindingV1, HrJobModelV2, HrScreeningRunV2, HrInterpretationV1
from contracts.ontology_validation import CompiledPredicateV1
from contracts.artifact_pin import ArtifactPinV1
from eos.artifact_hashing import content_hash
from services.deterministic_rule_evaluator import DeterministicRuleEvaluator
from services.jd_compliance_review import _location
from services.resume_screening_lineage import union_months

PROFILE = "hr_explicit_facts.v1"
SENSITIVE = re.compile(
    r"婚|育有|怀孕|妊娠|健康|病史|性别|男性|女性|民族|户籍|年龄|身份证|照片"
)
NEGATED = re.compile(r"不得|不要求|无需|无须|禁止|不以|培训|引用|示例|教学")
DEGREES = {
    "高中": 0,
    "中专": 0,
    "大专": 1,
    "专科": 1,
    "本科": 2,
    "学士": 2,
    "硕士": 3,
    "博士": 4,
}


def evidence(review, segment, start=0, end=None):
    end = len(segment["text"]) if end is None else end
    quote = segment["text"][start:end]
    anchor = segment["source_anchor_ref"]
    ref = "hr-evidence:" + content_hash([review.result_hash, anchor, start, end])
    return dict(
        evidence_ref=ref,
        anchor_ref=anchor,
        quote=quote,
        source_hash=review.source["source_hash"],
        source_identity_ref=review.source["source_identity_ref"],
        presentation_location=_location(
            SimpleNamespace(**segment),
            review.source["document_kind"],
            start,
            end,
            quote,
        ),
    )


def compile_requirements(review):
    segments = {s["segment_id"]: s for s in review.canonical_input["segments"]}
    result = []
    for node in review.requirements_graph["nodes"]:
        if node["kind"] != "requirement":
            continue
        text = node["text"]
        segment = segments[node["segment_id"]]
        item = dict(
            requirement_id=node["id"],
            text=text,
            modality="PREFERRED"
            if re.search(r"优先|preferred", text, re.I)
            else "REQUIRED",
            evidence=evidence(
                review, segment, node["start_offset"], node["end_offset"]
            ),
            predicate=None,
            unit=None,
            reason="INTERPRETATION_UNAVAILABLE",
        )
        if SENSITIVE.search(text) or NEGATED.search(text):
            item["reason"] = "EXCLUDED_NON_FIT_OR_NON_OPERATIVE"
        elif re.search(r"或|且|不超过|至多|低于|少于|以内|以下", text):
            pass
        else:
            # Deliberately bounded grammar: qualified/specialist years need interpretation.
            numeric = re.fullmatch(
                r"(?:岗位要求|任职要求|要求)?[:：\s]*(?:具备|具有|拥有)?(?:(?:工作经验|工作年限)\s*(?:至少|不少于|[:：])?\s*(\d+(?:\.\d+)?)\s*年|(\d+(?:\.\d+)?)\s*年以上(?:的)?工作经验)[。.!！\s]*",
                text.lstrip("◆•*- "),
            )
            degree = re.fullmatch(
                r"(?:岗位要求|任职要求|要求)?[:：\s]*(?:具备|具有)?("
                + "|".join(DEGREES)
                + r")(?:及?以上)?(?:学历|学位)?(?:优先)?[。.!！\s]*",
                text.lstrip("◆•*- "),
            )
            if numeric:
                value = float(numeric.group(1) or numeric.group(2)) * 12
                item.update(
                    predicate=dict(
                        fact_ref="experience_months",
                        operator="GE",
                        expected_value=value,
                    ),
                    unit="month",
                    reason="EXPLICIT_NUMERIC_REQUIREMENT",
                )
            elif degree:
                item.update(
                    predicate=dict(
                        fact_ref="education_level",
                        operator="GE",
                        expected_value=DEGREES[degree.group(1)],
                    ),
                    unit="education_level",
                    reason="EXPLICIT_EDUCATION_REQUIREMENT",
                )
        result.append(item)
        # A degree lower bound is independently checkable, but never proves a
        # specialist major. Retain the original combined condition as UNKNOWN.
        if (
            item["reason"] == "INTERPRETATION_UNAVAILABLE"
            and not SENSITIVE.search(text)
            and not NEGATED.search(text)
            and not re.search(r"或|不|未|无|非|以下|以内", text)
        ):
            match = re.search(r"(本科|大专|专科|硕士|博士)(?:及?以上)?学历", text)
            if match:
                component = dict(
                    item,
                    requirement_id=item["requirement_id"] + ":education",
                    text=match.group(0),
                    reason="EXPLICIT_EDUCATION_COMPONENT",
                    unit="education_level",
                    predicate=dict(
                        fact_ref="education_level",
                        operator="GE",
                        expected_value=DEGREES[match.group(1)],
                    ),
                    evidence=evidence(
                        review,
                        segment,
                        node["start_offset"] + match.start(),
                        node["start_offset"] + match.end(),
                    ),
                )
                result.append(component)
    return result


def compile_clauses(review):
    """Compile auxiliary v2 meaning from exact canonical spans and section pins.

    A section establishes source role, not evidence of a candidate's ability.
    Unknown syntax is retained, never upgraded to a mandatory predicate.
    """
    from contracts.hr_workbench import HrClauseV2

    rows = []
    section_role = None
    structure = None
    headings = re.compile(r"^(?:\d+[.、]\s*)?(岗位职责|工作职责|工作内容|职责|任职要求|岗位要求|招聘条件|资格要求|任职资格|职位要求|要求|其他说明|岗位说明|福利待遇|公司简介|公司概况|关于我们|招聘流程|Responsibilities|Qualifications|Requirements|Information)\s*[:：]?", re.I)
    for segment in review.canonical_input["segments"]:
        for index, match in enumerate(re.finditer(r"[^；;。\n]+", segment["text"]), 1):
            original = match.group()
            text = original.strip().lstrip("◆•*- ")
            ref = "jdr_" + segment["segment_id"] + ":clause:" + str(index)
            proof = evidence(review, segment, match.start(), match.end())
            heading = headings.match(text)
            role = section_role or "UNKNOWN"
            heading_proof = structure
            body = text
            if heading:
                label = heading.group(1).lower()
                section_role = ("DUTY" if label in ("岗位职责", "工作职责", "工作内容", "职责", "responsibilities")
                                else "INFORMATIONAL" if label in ("其他说明", "岗位说明", "福利待遇", "公司简介", "公司概况", "关于我们", "招聘流程", "information")
                                else "QUALIFICATION")
                body = text[heading.end():].strip()
                heading_start = match.start() + original.find(text)
                structure = evidence(review, segment, heading_start, heading_start + heading.end())
                heading_proof = structure
                role = section_role if body else "HEADING"
            elif re.match(r"(?:职位名称|岗位名称|招聘岗位|岗位|职位|Job title)\s*[:：]", text, re.I):
                role = "TITLE"
                section_role = None
                structure = None
                heading_proof = None
            elif re.fullmatch(r"教育背景[:：]?|经\s*验[:：]?|技能技巧[:：]?|态\s*度[:：]?", text) and section_role == "QUALIFICATION":
                role = "HEADING"
                heading_proof = structure
            elif re.fullmatch(r"[^:：，,。；;]+[:：]", text):
                # An unclassified explicit heading terminates inherited meaning.
                role, section_role, structure = "HEADING", None, proof
                heading_proof = proof
            elif not rows and text.endswith(("职位说明书", "岗位说明书")):
                role = "TITLE"
            elif section_role is None:
                # Explicit qualification syntax may establish its own role.
                if re.search(r"至少|不少于|以上.*(?:学历|经验)|学历.*以上|优先|必须|必要|熟悉|掌握|preferred|required", text, re.I):
                    role = "QUALIFICATION"
                elif re.match(r"负责|承担|维护|执行|监控", text):
                    role = "DUTY"
            modality = ("PREFERRED" if re.search(r"优先|preferred", body, re.I) else
                        "REQUIRED" if role == "QUALIFICATION" and (
                            section_role == "QUALIFICATION" or re.search(r"至少|不少于|以上|必须|required", body, re.I)) else
                        "UNSPECIFIED")
            reason = "INTERPRETATION_UNAVAILABLE"
            if SENSITIVE.search(body) or NEGATED.search(body):
                role, modality, reason = "EXCLUDED", "UNSPECIFIED", "EXCLUDED_NON_FIT_OR_NON_OPERATIVE"
            elif role != "QUALIFICATION":
                modality, reason = "UNSPECIFIED", "SOURCE_ROLE_NOT_FIT"
            base = dict(requirement_id=ref, text=original, role=role, modality=modality,
                        evidence=proof, structure_evidence=heading_proof, predicate=None,
                        unit=None, reason=reason)
            children = []
            if role == "QUALIFICATION":
                # Upper bounds, negations and ambiguous alternatives remain unresolved.
                bounded = not re.search(r"不超过|至多|低于|少于|以内|以下|可能|假设|不|未|无|非", body)
                if bounded:
                    degree = re.search(r"(本科|学士|大专|专科|硕士|博士)(?:及?以上)?(?:学历|学位)?", body)
                    major = re.search(r"([^，,；;：:\s]+?)专业", body)
                    numeric = re.fullmatch(r"(?:具备|具有|拥有)?(?:(?:工作经验|工作年限)\s*(?:至少|不少于|[:：])?\s*(\d+(?:\.\d+)?)\s*年|(\d+(?:\.\d+)?)\s*年以上(?:的)?工作经验)[.!！\s]*", body)
                    specialist = re.search(r"(\d+(?:\.\d+)?)\s*年以上(?:的)?(.+?)(?:工作)?经验", body) if not numeric else None
                    skill = re.search(r"(?:熟悉|掌握|具备)\s*([^，,；;]+?)(?:技能|能力)?(?:优先)?$", body)
                    covered = []
                    def child(subject, found, *, predicate=None, domain=None, unit=None, threshold=None):
                        covered.append(found.span())
                        offset = match.start() + original.find(body) + found.start()
                        return dict(base, requirement_id=ref + ":" + subject, text=found.group(),
                                    evidence=evidence(review, segment, offset, offset + len(found.group())),
                                    subject=subject, domain=domain, predicate=predicate, unit=unit,
                                    comparator=predicate["operator"] if predicate else "GE" if subject == "specialist_experience" else "EQ",
                                    threshold=threshold, reason="EXPLICIT_EDUCATION_COMPONENT" if subject == "degree" else "INTERPRETATION_UNAVAILABLE")
                    if degree:
                        children.append(child("degree", degree, predicate=dict(fact_ref="education_level", operator="GE", expected_value=DEGREES[degree.group(1)]), unit="education_level", threshold=DEGREES[degree.group(1)]))
                    if major:
                        children.append(child("major", major, domain=major.group(1), threshold=major.group(1)))
                    if specialist:
                        children.append(child("specialist_experience", specialist, domain=specialist.group(2).strip(), unit="month", threshold=float(specialist.group(1))*12))
                    if skill and not numeric and not specialist and not degree:
                        children.append(child("skill", skill, domain=skill.group(1).strip(), threshold=skill.group(1).strip()))
                    if numeric:
                        threshold = float(numeric.group(1) or numeric.group(2))*12
                        base.update(subject="experience", comparator="GE", threshold=threshold, unit="month",
                                    predicate=dict(fact_ref="experience_months", operator="GE", expected_value=threshold), reason="EXPLICIT_NUMERIC_REQUIREMENT")
                    elif len(children) == 1 and (not re.search(r"或|且|以及|并|和|，|,", body)):
                        only = children[0]
                        # Recognized leaf cannot erase any unrecognized qualifier.
                        standalone = (only["subject"] != "degree" or re.fullmatch(r"(?:具备|具有)?(?:本科|学士|大专|专科|硕士|博士)(?:及?以上)?(?:学历|学位)?(?:优先)?[.!！\s]*", body))
                        if standalone:
                            base.update({k:v for k,v in only.items() if k not in ("requirement_id", "text", "evidence", "structure_evidence")})
                            children = []
                    connectors = list(re.finditer(r"并且|以及|或|且|并", body))
                    if connectors and not major:
                        # Uniform explicit operators over bounded leaf syntax. Keep each
                        # unrecognized part as its own UNKNOWN operand, never drop it.
                        parts = []
                        start = 0
                        for separator in connectors:
                            parts.append((start, separator.start()))
                            start = separator.end()
                        parts.append((start, len(body)))
                        children = []
                        for part_index, (begin, end) in enumerate(parts):
                            part = body[begin:end].strip()
                            offset = match.start() + original.find(body) + begin + len(body[begin:end]) - len(body[begin:end].lstrip())
                            component = dict(base, requirement_id=ref + ":operand:" + str(part_index), text=part,
                                             evidence=evidence(review, segment, offset, offset+len(part)),
                                             subject="unresolved", predicate=None, reason="INTERPRETATION_UNAVAILABLE")
                            deg = re.fullmatch(r"(本科|学士|大专|专科|硕士|博士)(?:及?以上)?(?:学历|学位)?(?:优先)?", part)
                            num = re.fullmatch(r"(?:工作经验|工作年限)\s*(?:至少|不少于|[:：])?\s*(\d+(?:\.\d+)?)\s*年", part)
                            claim = re.fullmatch(r"(?:熟悉|掌握)\s*(.+?)(?:技能)?", part)
                            if deg:
                                component.update(subject="degree", unit="education_level", comparator="GE", threshold=DEGREES[deg.group(1)],
                                    predicate=dict(fact_ref="education_level", operator="GE", expected_value=DEGREES[deg.group(1)]), reason="EXPLICIT_EDUCATION_COMPONENT")
                            elif num:
                                threshold = float(num.group(1))*12
                                component.update(subject="experience", unit="month", comparator="GE", threshold=threshold,
                                    predicate=dict(fact_ref="experience_months", operator="GE", expected_value=threshold), reason="EXPLICIT_NUMERIC_REQUIREMENT")
                            elif claim:
                                component.update(subject="skill", domain=claim.group(1), comparator="EQ", threshold=claim.group(1))
                            children.append(component)
                    if children:
                        if not (connectors and not major):
                            # Every uncovered source operand survives, regardless of
                            # how many other components the compiler recognized.
                            cursor = 0
                            for begin, end in sorted(covered) + [(len(body), len(body))]:
                                residual = body[cursor:begin]
                                for gap in re.finditer(r"[^，,；;。.!！\s]+", residual):
                                    if re.fullmatch(r"(?:并且|以及|或|且|并|和|具备|具有|优先)+", gap.group()):
                                        continue
                                    offset = match.start() + original.find(body) + cursor + gap.start()
                                    children.append(dict(base, requirement_id=ref+":unresolved:"+str(offset),
                                        text=gap.group(), evidence=evidence(review, segment, offset, offset+len(gap.group())),
                                        subject="unresolved", predicate=None, reason="INTERPRETATION_UNAVAILABLE"))
                                cursor = max(cursor, end)
                        connective = "OR" if re.search(r"或", body) and not re.search(r"且|并|以及", body) and not (major and "或" in major.group()) else "AND"
                        # Mixed connectives cannot silently collapse into one operator.
                        if re.search(r"或", body) and re.search(r"且|并|以及", body):
                            children.append(dict(base, requirement_id=ref+":mixed", subject="unresolved", reason="AMBIGUOUS_CONNECTIVE"))
                            connective = "AND"
                            base["reason"] = "AMBIGUOUS_CONNECTIVE"
                        base.update(connective=connective, child_ids=[c["requirement_id"] for c in children])
                        for c in children:
                            c["parent_requirement_id"] = ref
            rows.append(HrClauseV2.model_validate(base))
            rows.extend(HrClauseV2.model_validate(c) for c in children)
    return rows


def condition_outcome(connective, outcomes):
    """Partial-information connective; missing operands never become booleans."""
    if not outcomes:
        return "UNKNOWN"
    dominant = "SUPPORTED_GAP" if connective == "AND" else "SUPPORTED_MATCH"
    identity = "SUPPORTED_MATCH" if connective == "AND" else "SUPPORTED_GAP"
    if dominant in outcomes:
        return dominant
    if "CONFLICT" in outcomes:
        return "CONFLICT"
    if any(x not in ("SUPPORTED_MATCH", "SUPPORTED_GAP") for x in outcomes):
        return "UNKNOWN"
    return identity


def extract_facts(review, as_of):
    observations = {}
    unfinished = re.compile(
        r"在读|就读|肄业|未|不|无|预计|预期|计划|假设|以下|以内|至多|不到|少于|低于"
    )
    intervals = []
    interval_evidence = []
    invalid_dates = False
    candidate = None
    section = None
    headers = re.compile(
        r"教育背景|教育经历|工作经历|工作经验(?!\s*[:：]\s*\d)|项目经历|个人评价|职业技能"
    )
    dates = re.compile(r"(\d{4}[./-]\d{1,2})\s*[–—~至-]\s*(\d{4}[./-]\d{1,2}|至今)")
    for segment in review.canonical_input["segments"]:
        text = segment["text"]
        name = re.search(r"(?:姓名|Name)\s*[:：]\s*([^\s，,；;]+)", text, re.I)
        if name and candidate is None:
            candidate = name.group(1)
        for m in re.finditer(
            r"(?:工作经验|工作年限)\s*[:：]\s*(\d+(?:\.\d+)?)\s*年(?=$|[\s，,；;。.!！])",
            text,
        ):
            if NEGATED.search(
                text[max(0, m.start() - 16) : m.end()]
            ) or unfinished.search(text[max(0, m.start() - 12) : m.end() + 12]):
                continue
            observations.setdefault("experience_months", []).append(
                (float(m.group(1)) * 12, evidence(review, segment, m.start(), m.end()))
            )
        for m in re.finditer(
            r"(?:学历|学位)\s*[:：]\s*("
            + "|".join(DEGREES)
            + r")(?=$|[\s，,；;。.!！])",
            text,
        ):
            if NEGATED.search(
                text[max(0, m.start() - 16) : m.end()]
            ) or unfinished.search(text[max(0, m.start() - 12) : m.end() + 12]):
                continue
            observations.setdefault("education_level", []).append(
                (DEGREES[m.group(1)], evidence(review, segment, m.start(), m.end()))
            )
        # Native DOCX textboxes may share a canonical paragraph. Keep the native
        # anchor and offsets while recognizing explicit section headings inline.
        marks = list(headers.finditer(text))
        spans = [(0, marks[0].start() if marks else len(text), section)]
        for i, mark in enumerate(marks):
            section = (
                "education"
                if mark.group().startswith("教育")
                else "work"
                if mark.group().startswith("工作")
                else None
            )
            spans.append(
                (
                    mark.end(),
                    marks[i + 1].start() if i + 1 < len(marks) else len(text),
                    section,
                )
            )
        for start, end, kind in spans:
            chunk = text[start:end]
            if kind == "education":
                if (
                    unfinished.search(chunk)
                    or NEGATED.search(chunk)
                    or re.search(r"至今|进修|研修|结业|培训", chunk)
                ):
                    continue
                education_dates = [
                    tuple(re.sub(r"[/\-]", ".", v) for v in m.groups())
                    for m in dates.finditer(chunk)
                ]
                try:
                    if education_dates:
                        union_months(education_dates, as_of)
                except ValueError:
                    continue
                for m in re.finditer("本科|学士|硕士|博士|大专|专科", chunk):
                    observations.setdefault("education_level", []).append(
                        (
                            DEGREES[m.group()],
                            evidence(
                                review, segment, start + m.start(), start + m.end()
                            ),
                        )
                    )
            elif kind == "work":
                if unfinished.search(chunk) or NEGATED.search(chunk):
                    continue
                found = list(dates.finditer(chunk))
                for m in found:
                    intervals.append(
                        tuple(re.sub(r"[/\-]", ".", v) for v in m.groups())
                    )
                    interval_evidence.append(
                        evidence(review, segment, start + m.start(), start + m.end())
                    )
                if not found and re.search(r"\d{4}|至今", chunk):
                    invalid_dates = True
    if intervals:
        try:
            months = union_months(intervals, as_of)
            observations.setdefault("experience_months", []).extend(
                (months, e) for e in interval_evidence
            )
        except ValueError:
            invalid_dates = True
    facts = {}
    proofs = {}
    conflicts = set()
    for key, values in observations.items():
        distinct = {v for v, _ in values}
        proofs[key] = [e for _, e in values]
        if len(distinct) > 1:
            conflicts.add(key)
        elif not (key == "experience_months" and invalid_dates):
            facts[key] = values[0][0]
    return candidate, facts, proofs, conflicts


class HrWorkbenchService:
    def __init__(self, *, review_service, artifact_service=None, request_context=None):
        self.review = review_service
        self.store = review_service.store
        self.artifacts = artifact_service
        self.request_context = request_context

    def _scope(self):
        c = self.review.context
        return dict(
            organization_id=c.organization_id,
            workspace_id=c.workspace_id,
            actor_id=c.actor_id,
            session_id=c.session_id,
        )

    @staticmethod
    def review_pin(review):
        return ArtifactPinV1(
            artifact_kind="jd_review",
            artifact_ref=review.review_id,
            artifact_hash=review.result_hash,
        )

    def _get(self, pin, model):
        expected = model.model_fields["schema_version"].default.split(".")[0]
        if pin.artifact_kind != expected:
            raise ValueError("ARTIFACT_KIND_MISMATCH")
        value = self.store.get(
            organization_id=self.review.context.organization_id,
            workspace_id=self.review.context.workspace_id,
            artifact_ref=pin.artifact_ref,
            contract_type=model,
        )
        if value is None:
            raise ValueError("HR_NOT_FOUND")
        self.review._access(value)
        if value.artifact_hash != pin.artifact_hash:
            raise ValueError("HR_PIN_MISMATCH")
        return value

    def get_model(self, pin):
        raw = self.store.get_raw(organization_id=self.review.context.organization_id, workspace_id=self.review.context.workspace_id, artifact_ref=pin.artifact_ref)
        cls = HrJobModelV2 if raw and raw["schema_version"] == "hr_job_model.v2" else HrJobModelV1
        return self._get(pin, cls)

    def get_run(self, pin):
        raw = self.store.get_raw(organization_id=self.review.context.organization_id, workspace_id=self.review.context.workspace_id, artifact_ref=pin.artifact_ref)
        cls = HrScreeningRunV2 if raw and raw["schema_version"] == "hr_screening_run.v2" else HrScreeningRunV1
        return self._get(pin, cls)

    def _review(self, pin, live=True, purpose=None):
        if pin.artifact_kind != "jd_review":
            raise ValueError("ARTIFACT_KIND_MISMATCH")
        review = self.review.get(pin.artifact_ref)
        if review.result_hash != pin.artifact_hash:
            raise ValueError("STALE_REVIEW_HASH")
        if live:
            if self.artifacts:
                bindings = [
                    x
                    for x in self.list_artifacts(HrSourceBindingV1)
                    if x.review_pin == pin
                ]
                if (
                    bindings
                    and purpose
                    and not any(x.purpose == purpose for x in bindings)
                ):
                    raise ValueError("SOURCE_PURPOSE_MISMATCH")
                # Legacy exact reviews remain readable, but new runs require custody binding.
                if not bindings:
                    predecessor = (review.successor or {}).get("supersedes_review_id")
                    if not predecessor:
                        raise ValueError("SHARED_SOURCE_BINDING_REQUIRED")
                    self._review(
                        self.review_pin(self.review.get(predecessor)), purpose=purpose
                    )
                for binding in bindings:
                    self.artifacts.resolve(
                        ctx=self.request_context,
                        artifact_pin=binding.workspace_artifact_pin,
                    )
            try:
                self.review.source_path(review.review_id)
            except OSError as exc:
                raise ValueError("STALE_SOURCE_UNAVAILABLE") from exc
            self.review.get_graph(review.review_id)
        return review

    def model(self, review_id, *, job_ref=None, predecessor_pin=None):
        review = self.review.get(review_id)
        self._review(self.review_pin(review), purpose="JD")
        if predecessor_pin:
            parent = self.get_model(predecessor_pin)
            if parent.job_ref != job_ref:
                raise ValueError("JOB_MODEL_MISMATCH")
        elif job_ref is not None and job_ref != review_id:
            raise ValueError("PREDECESSOR_REQUIRED")
        if review.status not in ("CHECKED", "PARTIAL"):
            raise ValueError("MODEL_NOT_ELIGIBLE")
        graph = review.requirements_graph
        clauses = compile_clauses(review)
        value = HrJobModelV2.create(
            **self._scope(),
            recorded_at=review.recorded_at,
            job_ref=job_ref or review_id,
            review_pin=self.review_pin(review),
            predecessor_pin=predecessor_pin,
            source=review.source,
            graph_pin={
                "artifact_id": graph["artifact_id"],
                "artifact_hash": graph["artifact_hash"],
            },
            requirements=clauses,
            coverage=dict(source_clauses=sum(c.parent_requirement_id is None for c in clauses),
                          qualifications=sum(c.role == "QUALIFICATION" and c.parent_requirement_id is None for c in clauses),
                          automatic_conditions=sum(c.predicate is not None for c in clauses),
                          unresolved_conditions=sum(c.role == "QUALIFICATION" and not c.predicate and not c.child_ids for c in clauses)),
        )
        self.store.put(value)
        return value

    def get_interpretation(self, pin):
        return self._get(pin, HrInterpretationV1)

    def interpretation_evidence(self, *, model_pin, resume_pin, requirement_id):
        model = self.get_model(model_pin)
        if not isinstance(model, HrJobModelV2):
            raise ValueError("VERSIONED_MEANING_REQUIRED")
        self._review(model.review_pin, purpose="JD")
        resume = self._review(resume_pin, purpose="RESUME")
        condition = next((c for c in model.requirements if c.requirement_id == requirement_id), None)
        if condition is None or condition.child_ids or condition.subject not in ("major", "skill", "specialist_experience"):
            raise ValueError("INTERPRETATION_CONDITION_MISMATCH")
        candidates = [dict(selection=dict(anchor_ref=s["source_anchor_ref"], source_hash=resume.source["source_hash"],
                                         quote=s["text"], start_offset=0, end_offset=len(s["text"])),
                           evidence=evidence(resume, s))
                      for s in resume.canonical_input["segments"] if s["text"].strip()]
        return dict(model_pin=model_pin.model_dump(mode="json"), resume_pin=resume_pin.model_dump(mode="json"),
                    requirement_id=requirement_id, source=resume.source, candidates=candidates,
                    interpretation_capability="AUTOMATION_UNAVAILABLE")

    def interpretation(self, *, model_pin, resume_pin, requirement_id, selections,
                       interpretation_value, source_value, disposition, predecessor_pin=None, supports=True):
        from contracts.hr_workbench import HrInterpretationCommand
        command = HrInterpretationCommand.model_validate(dict(
            model_pin=model_pin, resume_pin=resume_pin, requirement_id=requirement_id,
            selections=selections, interpretation_value=interpretation_value, source_value=source_value,
            disposition=disposition, predecessor_pin=predecessor_pin, supports=supports))
        model = self.get_model(model_pin)
        if not isinstance(model, HrJobModelV2):
            raise ValueError("VERSIONED_MEANING_REQUIRED")
        self._review(model.review_pin, purpose="JD")
        resume = self._review(resume_pin, purpose="RESUME")
        condition = next((c for c in model.requirements if c.requirement_id == requirement_id), None)
        if condition is None or condition.role != "QUALIFICATION" or condition.child_ids or condition.subject not in ("major", "skill", "specialist_experience"):
            raise ValueError("INTERPRETATION_CONDITION_MISMATCH")
        if interpretation_value != condition.domain:
            raise ValueError("INTERPRETATION_DOMAIN_MISMATCH")
        if predecessor_pin:
            predecessor = self.get_interpretation(predecessor_pin)
            if (predecessor.model_pin, predecessor.resume_pin, predecessor.requirement_id) != (model_pin, resume_pin, requirement_id):
                raise ValueError("INTERPRETATION_SCOPE_MISMATCH")
        proofs = []
        intervals = []
        _, _, automatic_proofs, _ = extract_facts(resume, datetime.now(timezone.utc).date().isoformat())
        for selection in command.selections:
            segments = [s for s in resume.canonical_input["segments"] if s["source_anchor_ref"] == selection.anchor_ref]
            if len(segments) != 1 or selection.source_hash != resume.source["source_hash"]:
                raise ValueError("INTERPRETATION_SOURCE_MISMATCH")
            segment = segments[0]
            if (selection.end_offset > len(segment["text"]) or selection.start_offset >= selection.end_offset
                or segment["text"][selection.start_offset:selection.end_offset] != selection.quote):
                raise ValueError("INTERPRETATION_QUOTE_MISMATCH")
            if supports and (NEGATED.search(selection.quote) or re.search(r"不会|不熟悉|未掌握|没有|不具备|未取得|无相关", selection.quote)):
                raise ValueError("NON_OPERATIVE_INTERPRETATION")
            proofs.append(evidence(resume, segment, selection.start_offset, selection.end_offset))
            if condition.subject == "specialist_experience":
                matches = list(re.finditer(r"(\d{4}[./-]\d{1,2})\s*[–—~至-]\s*(\d{4}[./-]\d{1,2}|至今)", selection.quote))
                if not matches:
                    raise ValueError("WORK_INTERVAL_EVIDENCE_REQUIRED")
                for match in matches:
                    # Date syntax alone is insufficient: it must be a native work observation.
                    if not any(p["anchor_ref"] == selection.anchor_ref and p["quote"] == match.group()
                               for p in automatic_proofs.get("experience_months", [])):
                        raise ValueError("WORK_INTERVAL_EVIDENCE_REQUIRED")
                    intervals.append(tuple(re.sub(r"[/\-]", ".", x) for x in match.groups()))
        if not any(source_value in p["quote"] for p in proofs):
            raise ValueError("INTERPRETATION_VALUE_NOT_IN_SOURCE")
        if condition.subject == "major" and not any(re.search(r"专业\s*[:：]\s*" + re.escape(source_value) + r"(?=$|[\s，,；;。])", p["quote"]) for p in proofs):
            raise ValueError("EXPLICIT_MAJOR_CLAIM_REQUIRED")
        if condition.subject == "skill" and not any(re.search(r"技能|熟悉|掌握|开发|维护|操作|使用", p["quote"]) for p in proofs):
            raise ValueError("EXPLICIT_SKILL_CLAIM_REQUIRED")
        if condition.subject != "specialist_experience" and len(proofs) != 1:
            raise ValueError("SINGLE_CLAIM_EVIDENCE_REQUIRED")
        value = HrInterpretationV1.create(**self._scope(), recorded_at=datetime.now(timezone.utc),
            job_ref=model.job_ref, model_pin=model_pin, resume_pin=resume_pin,
            requirement_id=requirement_id, subject=condition.subject,
            interpretation_value=interpretation_value, source_value=source_value,
            disposition=disposition, predecessor_pin=predecessor_pin, supports=supports,
            source_evidence=proofs, intervals=intervals)
        def verify():
            self._review(model.review_pin, purpose="JD")
            self._review(resume_pin, purpose="RESUME")
        self.store.put_hr_interpretation(value, verify)
        return value

    def _interpretations(self, pins, model, resume_pins):
        values = []
        keys = set()
        for pin in pins:
            value = self.get_interpretation(pin)
            key = (value.resume_pin.artifact_ref, value.requirement_id)
            if key in keys:
                raise ValueError("DUPLICATE_INTERPRETATION")
            keys.add(key)
            if value.model_pin != model.pin() or value.resume_pin not in resume_pins or value.job_ref != model.job_ref:
                raise ValueError("INTERPRETATION_SCOPE_MISMATCH")
            if self.store.hr_interpretation_head(value) != pin:
                raise ValueError("STALE_INTERPRETATION_HEAD")
            self._review(value.resume_pin, purpose="RESUME")
            values.append(value)
        return values

    def evaluate(self, model, resume, as_of, interpretations=()):
        name, facts, proofs, conflicts = extract_facts(resume, as_of)
        results = []
        for clause in model.requirements:
            req = clause.model_dump(mode="json") if hasattr(clause, "role") else clause
            predicate = req["predicate"]
            outcome = "UNKNOWN"
            observed = None
            matching = []
            reviewed = next((v for v in interpretations if v.resume_pin == self.review_pin(resume)
                             and v.requirement_id == req["requirement_id"]), None)
            support_basis = "AUTOMATIC" if predicate else "UNRESOLVED"
            if reviewed and reviewed.disposition == "CONFIRMED":
                key = "reviewed:" + req["requirement_id"]
                if reviewed.subject == "specialist_experience":
                    facts[key] = union_months(reviewed.intervals, as_of) if reviewed.supports else None
                    predicate = dict(fact_ref=key, operator="GE", expected_value=req["threshold"])
                else:
                    facts[key] = reviewed.supports
                    predicate = dict(fact_ref=key, operator="EQ", expected_value=True)
                proofs[key] = list(reviewed.source_evidence)
                if reviewed.subject == "skill":
                    denial = re.compile(r"(?:不会|未掌握|不熟悉|不具备)\s*" + re.escape(reviewed.source_value))
                    for segment in resume.canonical_input["segments"]:
                        if denial.search(segment["text"]) and not NEGATED.search(segment["text"]):
                            conflicts.add(key)
                            proofs[key].append(evidence(resume, segment))
                support_basis = "REVIEWED"
            if req["reason"] in ("EXCLUDED_NON_FIT_OR_NON_OPERATIVE", "SOURCE_ROLE_NOT_FIT"):
                outcome = "EXCLUDED"
            elif predicate:
                key = predicate["fact_ref"]
                observed = facts.get(key)
                matching = proofs.get(key, [])
                if key in conflicts:
                    outcome = "CONFLICT"
                else:
                    compiled = CompiledPredicateV1(
                        predicate_id=req["requirement_id"],
                        ir_node_ref=req["requirement_id"],
                        evidence_refs=(req["evidence"]["evidence_ref"],),
                        strict_operands=True,
                        **predicate,
                    )
                    evaluation = DeterministicRuleEvaluator._evaluate_predicate(
                        compiled,
                        facts,
                        optional_absence_is_failure=False,
                        strict_operands=True,
                    )
                    outcome = {
                        "SATISFIED": "SUPPORTED_MATCH",
                        "FAILED": "SUPPORTED_GAP",
                    }.get(evaluation.result, "UNKNOWN")
            results.append(
                dict(
                    requirement_id=req["requirement_id"],
                    **({"role": req["role"], "parent_requirement_id": req["parent_requirement_id"], "child_ids": req["child_ids"], "support_basis": support_basis, "interpretation_pins": [reviewed.pin().model_dump(mode="json")] if reviewed else []} if "role" in req else {}),
                    modality=req["modality"],
                    outcome=outcome,
                    observed_value=observed,
                    expected_value=predicate["expected_value"] if predicate else None,
                    unit=req["unit"],
                    reason=req["reason"],
                    jd_evidence=req["evidence"],
                    resume_evidence=matching,
                )
            )
        by_id = {r["requirement_id"]: r for r in results}
        for clause in model.requirements:
            if hasattr(clause, "child_ids") and clause.child_ids:
                parent = by_id[clause.requirement_id]
                child_results = [by_id[c] for c in clause.child_ids]
                parent["outcome"] = "UNKNOWN" if clause.reason == "AMBIGUOUS_CONNECTIVE" else condition_outcome(clause.connective, [c["outcome"] for c in child_results])
                parent["resume_evidence"] = [e for c in child_results for e in c["resume_evidence"]]
                parent["interpretation_pins"] = [p for c in child_results for p in c.get("interpretation_pins", [])]
                parent["support_basis"] = "REVIEWED" if any(c["support_basis"] == "REVIEWED" for c in child_results) else "UNRESOLVED"
        return dict(
            candidate_name=name,
            review_pin=self.review_pin(resume).model_dump(mode="json"),
            source=resume.source,
            results=results,
            **({"coverage": dict(
                automatic_support=sum(r["support_basis"] == "AUTOMATIC" and r["outcome"] in ("SUPPORTED_MATCH", "SUPPORTED_GAP") for r in results if not r.get("child_ids")),
                reviewed_support=sum(r["support_basis"] == "REVIEWED" and r["outcome"] in ("SUPPORTED_MATCH", "SUPPORTED_GAP") for r in results if not r.get("child_ids")),
                unresolved=sum(r["outcome"] in ("UNKNOWN", "CONFLICT") for r in results if r.get("role") == "QUALIFICATION" and not r.get("child_ids")))} if isinstance(model, HrJobModelV2) else {}),
            sensitivity_findings=[f.model_dump(mode="json") for f in resume.findings],
            fit_score=None,
        )

    def _operation_scope(self):
        scope = self._scope()
        return tuple(scope[k] for k in ("organization_id", "workspace_id", "actor_id", "session_id"))

    def operation_status(self, request_id):
        value = self.store.hr_operation(scope=self._operation_scope(), request_id=request_id)
        if value.result_pin:
            run = self.get_run(value.result_pin)
            if run.request_id != value.request_id or run.request_hash != value.request_hash:
                raise ValueError("OPERATION_RESULT_MISMATCH")
        return value

    def stop(self, *, job_ref, model_pin, resume_pins, request_id, as_of, interpretation_pins=(), cancel=False):
        model = self.get_model(model_pin)
        if model.job_ref != job_ref:
            raise ValueError("JOB_MODEL_MISMATCH")
        if not isinstance(model, HrJobModelV2):
            raise ValueError("VERSIONED_OPERATION_REQUIRED")
        command = dict(job_ref=job_ref, model_pin=model_pin.model_dump(mode="json"),
                       resume_pins=[p.model_dump(mode="json") for p in resume_pins], as_of=as_of,
                       session_id=self.review.context.session_id,
                       interpretation_pins=[p.model_dump(mode="json") for p in interpretation_pins])
        digest = "sha256:" + content_hash(command)
        self.store.hr_stop(scope=self._operation_scope(), request_id=request_id, request_hash=digest)
        state = self.store.hr_operation(scope=self._operation_scope(), request_id=request_id)
        if state.status == "INTERRUPTED":
            # A lost owner can acknowledge cancellation from its frozen inputs
            # without opening mutable sources or resuming candidate work.
            interpretations = [self.get_interpretation(pin) for pin in interpretation_pins]
            self._start_operation(model=model, sources=[], resume_pins=resume_pins,
                interpretations=interpretations, request_id=request_id, as_of=as_of, command=command, digest=digest)
            return self.operation_status(request_id)
        return state

    def _start_operation(self, *, model, sources, resume_pins, interpretations, request_id, as_of, command, digest):
        scope = self._operation_scope()
        claimed, token, state = self.store.hr_claim(scope=scope, request_id=request_id, request_hash=digest, command=command)
        if not claimed:
            if state["result_ref"]:
                return self.get_run(ArtifactPinV1(artifact_kind="hr_screening_run", artifact_ref=state["result_ref"], artifact_hash=state["result_hash"]))
            return self.store._hr_projection(state)
        failure = None
        try:
            try:
                for resume in sources[len(state["checkpoint"]):]:
                    if not self.store.hr_checkpoint(scope=scope, request_id=request_id, token=token):
                        break
                    candidate = self.evaluate(model, resume, as_of, interpretations) if interpretations else self.evaluate(model, resume, as_of)
                    if not self.store.hr_checkpoint(scope=scope, request_id=request_id, token=token, candidate=candidate):
                        break
            except Exception:
                failure = "EVALUATION_FAILED"
            def make_run(stopped, checkpoint):
                return HrScreeningRunV2.create(**self._scope(), recorded_at=datetime.now(timezone.utc),
                    job_ref=model.job_ref, model_pin=model.pin(), resume_pins=resume_pins,
                    interpretation_pins=[v.pin() for v in interpretations],
                    interpretation_snapshots=[v.model_dump(mode="json") for v in interpretations],
                    request_id=request_id, request_hash=digest, as_of=as_of,
                    status="CANCELLED" if stopped else "FAILED" if failure else "COMPLETED",
                    failure_code=failure, candidates=[] if stopped or failure else checkpoint,
                    model_snapshot=model.model_dump(mode="json"))
            def verify():
                self._review(model.review_pin, purpose="JD")
                for pin in resume_pins:
                    self._review(pin, purpose="RESUME")
                self._interpretations([v.pin() for v in interpretations], model, resume_pins)
            winner = self.store.hr_finish(scope=scope, request_id=request_id, token=token,
                                          make_run=make_run, verify_before_commit=verify)
            return self.get_run(winner)
        finally:
            self.store.hr_release(token)

    def start(
        self, *, job_ref, model_pin, resume_pins, request_id, as_of, cancel=False, interpretation_pins=()
    ):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
            raise ValueError("INVALID_REQUEST_ID")
        if date.fromisoformat(as_of).isoformat() != as_of:
            raise ValueError("INVALID_AS_OF")
        if not 1 <= len(resume_pins) <= 30 or len(
            {p.artifact_ref for p in resume_pins}
        ) != len(resume_pins):
            raise ValueError("INVALID_RESUME_MEMBERSHIP")
        model = self.get_model(model_pin)
        if model.job_ref != job_ref:
            raise ValueError("JOB_MODEL_MISMATCH")
        command = dict(
            job_ref=job_ref,
            model_pin=model_pin.model_dump(mode="json"),
            resume_pins=[p.model_dump(mode="json") for p in resume_pins],
            as_of=as_of,
            session_id=self.review.context.session_id,
            **({"interpretation_pins": [p.model_dump(mode="json") for p in interpretation_pins]} if isinstance(model, HrJobModelV2) else {}),
        )
        digest = "sha256:" + content_hash(command)
        scope = self._scope()
        operation = dict(
            organization_id=scope["organization_id"],
            workspace_id=scope["workspace_id"],
            actor_ref=scope["actor_id"],
            operation="HR_SCREEN",
            request_id=request_id,
            request_hash=digest,
        )
        previous = self.store.get_operation_result(
            **operation, artifact_kind="hr_screening_run"
        )
        if previous:
            return self.get_run(previous)
        if not isinstance(model, HrJobModelV2):
            raise ValueError("LEGACY_MODEL_READ_ONLY_UPGRADE_REQUIRED")
        if cancel:
            return self.stop(job_ref=job_ref, model_pin=model_pin, resume_pins=resume_pins,
                             request_id=request_id, as_of=as_of, interpretation_pins=interpretation_pins)
        self._review(model.review_pin, purpose="JD")
        sources = [self._review(p, purpose="RESUME") for p in resume_pins]
        interpretations = self._interpretations(interpretation_pins, model, resume_pins)
        return self._start_operation(model=model, sources=sources, resume_pins=resume_pins,
            interpretations=interpretations, request_id=request_id, as_of=as_of, command=command, digest=digest)

    def list_artifacts(self, model):
        import json

        scope = self._scope()
        rows = self.store._connection.execute(
            sql_query("SELECT payload_json FROM ontology_build_artifacts WHERE organization_id=? AND workspace_id=? AND schema_version=? ORDER BY recorded_at, artifact_ref",'SELECT payload_json FROM ontology_build_artifacts WHERE organization_id=%s AND workspace_id=%s AND schema_version=%s ORDER BY recorded_at, artifact_ref'),
            (
                scope["organization_id"],
                scope["workspace_id"],
                model.model_fields["schema_version"].default,
            ),
        ).fetchall()
        results = []
        for row in rows:
            data = json.loads(row[0])
            if (
                data.get("actor_id") == scope["actor_id"]
                and data.get("session_id") == scope["session_id"]
            ):
                results.append(model.model_validate(data))
        return results

    def import_artifact(self, *, artifact_pin, purpose):
        from pathlib import Path
        from storage.workspace_artifact_blob_store import WorkspaceArtifactBlobStore

        if self.artifacts is None:
            raise ValueError("SHARED_SOURCE_REQUIRED")
        artifact = self.artifacts.resolve(
            ctx=self.request_context, artifact_pin=artifact_pin
        )
        matches = [
            x
            for x in self.list_artifacts(HrSourceBindingV1)
            if x.workspace_artifact_pin == artifact_pin and x.purpose == purpose
        ]
        if matches:
            self._review(matches[0].review_pin)
            return matches[0]
        suffix = Path(artifact.original_name).suffix.lower()
        if suffix not in (".docx", ".pdf", ".txt", ".md"):
            raise ValueError("UNSUPPORTED_DOCUMENT_KIND")
        blobs = WorkspaceArtifactBlobStore(
            ctx=self.request_context, max_bytes=self.artifacts._max_bytes
        )
        with blobs.exact_get(
            byte_owner_ref=artifact.byte_owner_ref,
            expected_hash=artifact.content_hash,
            expected_size=artifact.byte_size,
        ) as stream:
            raw = stream.read()
        # Explicit governed import, bound to the existing workspace byte owner.
        relative = (
            "artifacts/governed/hr-" + artifact.content_hash.split(":")[1] + suffix
        )
        from services.governed_artifact_service import _publish_immutable_bytes

        target = (self.review.bound.session_root / relative).resolve()
        target.relative_to(self.review.bound.session_root.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        _publish_immutable_bytes(target, raw)
        review_id = "jdrv_" + content_hash(
            [self._scope(), artifact_pin.model_dump(mode="json"), purpose]
        )
        try:
            review = self.review.get(review_id)
        except ValueError as exc:
            if str(exc) != "REVIEW_NOT_FOUND":
                raise
            review = self.review.create(
                document_ref=relative,
                filename=artifact.original_name,
                review_id=review_id,
            )
        binding = HrSourceBindingV1.create(
            **self._scope(),
            recorded_at=review.recorded_at,
            workspace_artifact_pin=artifact_pin,
            review_pin=self.review_pin(review),
            purpose=purpose,
            content_hash=artifact.content_hash,
        )
        self.store.put(binding)
        return binding

    def context(self):
        return dict(
            models=[
                m.model_dump(mode="json") for m in [*self.list_artifacts(HrJobModelV1), *self.list_artifacts(HrJobModelV2)]
            ],
            runs=[
                r.model_dump(mode="json") for r in [*self.list_artifacts(HrScreeningRunV1), *self.list_artifacts(HrScreeningRunV2)]
            ],
            sources=[
                s.model_dump(mode="json")
                for s in self.list_artifacts(HrSourceBindingV1)
            ],
            operations=[v.model_dump(mode="json") for v in self.store.hr_operations(scope=self._operation_scope())],
            interpretations=[v.model_dump(mode="json") for v in self.list_artifacts(HrInterpretationV1)],
            limitations=[
                "AUXILIARY_SOURCE_REVIEW",
                "NOT_ACTIVATED",
                "INTERPRETATION_UNAVAILABLE",
                "HUMAN_VALUE_UNMEASURED",
            ],
        )

    def remediate(self, *, model_pin, request_id):
        from contracts.jd_compliance_review import JdSuccessorCommandV1

        model = self.get_model(model_pin)
        parent = self._review(model.review_pin, purpose="JD")
        child = self.review.successor(
            parent.review_id,
            JdSuccessorCommandV1(
                request_id=request_id,
                result_hash=parent.result_hash,
                source_hash=parent.source["source_hash"],
                version_id=parent.version_id,
            ),
        )
        if child.status not in ("CHECKED", "PARTIAL"):
            raise ValueError("REMEDIATION_FAILED")
        return self.model(
            child.review_id, job_ref=model.job_ref, predecessor_pin=model_pin
        )
