
from storage.hosted_postgres import sql_query,begin_write,in_transaction
"""Exact source documentary review over the existing workspace owners.

Read/replay/export never invokes parsing, retrieval, inference or activation.
"""
import base64
from datetime import datetime, timezone
import json

from contracts.ontology_building import ArtifactPinV1
from contracts.ontology_workbench_context import LifecycleArtifactBindingSetV1
from contracts.tender_review import (
    TenderReviewRecordCommandV1, TenderReviewReportCommandV1,
    TenderReviewRecordV1, TenderReviewReportV1, TenderReviewReportV2, TenderCandidateReportCommandV1, GovernedFindingSubjectV1,
)
from eos.artifact_hashing import content_hash, canonical_json
from services.workspace_artifact_service import WorkspaceArtifactService
from services.workspace_artifact_preparation_service import WorkspaceArtifactPreparationService
from semantier.knowledge_feature_packs.evidence_selector_builder import evidence_ref_for_block, build_evidence_selectors
from contracts.evidence_location import EvidenceRef
from semantier.knowledge_feature_packs.evidence_resolver import resolve_evidence_ref


class TenderReviewService:
    def __init__(self, *, artifact_service, preparation_service, session_ref, journey_service=None, inspection_service=None):
        if not isinstance(artifact_service, WorkspaceArtifactService) or not isinstance(preparation_service, WorkspaceArtifactPreparationService):
            raise TypeError("review requires real artifact and preparation owners")
        if preparation_service._artifacts is not artifact_service or artifact_service._cursors is None:
            raise TypeError("review requires one workspace/cursor owner")
        self._artifacts = artifact_service
        self._preparation = preparation_service
        self._store = artifact_service._store
        self._session = session_ref
        self._journeys = journey_service
        self._inspection = inspection_service

    def _scope(self, ctx):
        scope = self._artifacts._cursors._scope(ctx)
        if scope["session_id"] != self._session:
            raise ValueError("FORBIDDEN_SESSION")
        return scope

    def _compliance_originals(self, *, ctx, target, entries):
        """Freeze source presentation over exact retained IR; never parse or infer."""
        from contracts.tender_legal_compliance import TenderFrozenComplianceDecisionV1
        from storage.graph_intent_brief_store import GraphIntentBriefStore
        _, preparation, identity, document, _ = target
        sources = GraphIntentBriefStore(ctx.workspace_root, organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id, source_registry=self._preparation._design._knowledge)
        originals = []
        for raw in entries:
            entry = TenderFrozenComplianceDecisionV1.model_validate(raw)
            basis = entry.basis
            targets = [("TENDER", identity, preparation.canonical_source_ir_pin,
                        EvidenceRef.model_validate(evidence), document, None)
                       for evidence in basis.finding.model_dump(mode="json", by_alias=True)["evidenceRefs"]]
            targets.extend(("NORMATIVE", identity, ir, evidence, None, citation)
                for identity, ir, evidence, citation in basis.original_provisions())
            for role, source, ir, evidence, retained, citation in targets:
                status, location = "unavailable", None
                try:
                    if retained is None:
                        resolved_identity, retained = sources.resolve_evidence_source(
                            source_identity_ref=source.source_identity_ref,
                            canonical_source_ir_ref=ir.artifact_ref,
                            canonical_source_ir_hash=ir.artifact_hash)
                        if resolved_identity != source:
                            raise ValueError("COMPLIANCE_SOURCE_IDENTITY_MISMATCH")
                    resolved = resolve_evidence_ref(evidence, retained,
                        artifact_ref=ir.artifact_ref, selector_hash=evidence.selector_hash)
                    status = resolved.status
                    if status in ("exact", "relocated_exact"):
                        location = resolved.location
                except (FileNotFoundError, KeyError, ValueError):
                    # Preserve the exact original identity while disclosing that
                    # the current retained location cannot be resolved.
                    pass
                originals.append(dict(decision=entry.decision.pin().model_dump(mode="json"),
                    role=role, identity=source.model_dump(mode="json", by_alias=True),
                    canonical_source_ir=ir.model_dump(mode="json"),
                    evidence=evidence.model_dump(mode="json", by_alias=True),
                    status=status, location=location, citation=citation))
        return originals

    def _cursor(self, ctx, cursor_ref=None):
        self._scope(ctx)
        if cursor_ref is None:
            rows = self._store._connection.execute(
                sql_query("SELECT cursor_ref FROM workbench_projection_cursors WHERE organization_id=? AND workspace_id=? AND session_id=? AND actor_ref=? LIMIT 2",'SELECT cursor_ref FROM workbench_projection_cursors WHERE organization_id=%s AND workspace_id=%s AND session_id=%s AND actor_ref=%s LIMIT 2'),
                (ctx.organization_id, ctx.workspace_id, self._session, ctx.user_id),
            ).fetchall()
            if len(rows) != 1:
                raise ValueError("CURSOR_REQUIRED")
            cursor_ref = rows[0][0]
        return self._artifacts._cursors.get(ctx=ctx, cursor_ref=cursor_ref)

    def _target(self, ctx, cursor, target_pin=None, preparation_pin=None):
        bindings = self._store.get(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
                                   artifact_ref=cursor.runtime_binding_set_pin.artifact_ref, contract_type=LifecycleArtifactBindingSetV1)
        if bindings is None or bindings.binding_set_hash != cursor.runtime_binding_set_pin.artifact_hash:
            raise ValueError("CURSOR_FACT_MISMATCH")
        if not bindings.entries:
            if target_pin is not None:
                raise ValueError("TARGET_NOT_BOUND")
            return None
        entry = bindings.entries[0]
        if entry.state != "ACTIVE" or entry.preparation_pin is None:
            raise ValueError("TARGET_NOT_READY")
        if target_pin is not None and (entry.artifact_pin != target_pin or entry.preparation_pin != preparation_pin):
            raise ValueError("TARGET_PREPARATION_MISMATCH")
        preparation = self._preparation.resolve(ctx=ctx, artifact_pin=entry.artifact_pin, preparation_pin=entry.preparation_pin, purpose="RUNTIME_TARGET")
        artifact = self._artifacts.resolve(ctx=ctx, artifact_pin=entry.artifact_pin)
        if artifact.media_type not in {"application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/pdf"} or preparation.canonical_source_ir_pin is None:
            raise ValueError("DOCUMENT_REVIEW_TARGET_REQUIRED")
        identity, document = self._preparation.resolve_runtime_target(ctx=ctx, artifact_pin=entry.artifact_pin, preparation_pin=entry.preparation_pin)
        return entry, preparation, identity, document, artifact

    @staticmethod
    def _items(identity, document, preparation):
        result = []
        blocks = [block for block in document.blocks if block.block_type in ("paragraph", "cell") and isinstance(block.content, str) and block.content.strip()]
        selectors = build_evidence_selectors(document, blocks)
        for block in blocks:
            evidence = evidence_ref_for_block(document, block, source_identity=identity, created_at=identity.created_at, selector=selectors[block.block_id])
            location = resolve_evidence_ref(evidence, document, artifact_ref=preparation.canonical_source_ir_pin.artifact_ref,
                                            selector_hash=evidence.selector_hash)
            if location.status not in ("exact", "relocated_exact", "block", "text_fallback"):
                continue
            result.append(dict(subject=dict(kind="SOURCE_OCCURRENCE", source_block_ref=block.block_id,
                                           evidence_ref=evidence.evidence_ref, selector_hash=evidence.selector_hash),
                               exact_quote=location.matched_text, resolved_location=location.location,
                               block_type=block.block_type, ordinal=block.ordinal,
                               evidence=evidence.model_dump(mode="json", by_alias=True)))
        return result

    def _reference_materials(self, ctx, target_pin):
        """Available files are not clause links. No parser, evaluation or adoption runs here."""
        projection = self._artifacts.list(ctx=ctx)
        references = []
        for entry in projection.artifacts:
            artifact = entry.artifact
            pin = ArtifactPinV1(artifact_kind="workspace_artifact", artifact_ref=artifact.artifact_ref, artifact_hash=artifact.artifact_hash)
            if pin == target_pin:
                continue
            prepared = [p for p in projection.preparations if p.artifact_pin == pin
                        and any(h.artifact_ref == p.preparation_ref and h.artifact_hash == p.preparation_hash for h in entry.active_preparation_pins)]
            row = dict(artifact_pin=pin.model_dump(mode="json"), original_name=artifact.original_name,
                       access_state=entry.access_state, preparation_status="NOT_PREPARED",
                       relationship="AVAILABLE_UNLINKED", provisions=[])
            for prep in prepared:
                row["preparation_status"] = prep.outcome
                if entry.access_state != "AVAILABLE" or prep.outcome != "READY" or prep.canonical_source_ir_pin is None:
                    continue
                prep_pin = ArtifactPinV1(artifact_kind="workspace_artifact_preparation", artifact_ref=prep.preparation_ref, artifact_hash=prep.preparation_hash)
                try:
                    identity, document = (self._preparation.resolve_design_source(ctx=ctx, artifact_pin=pin, preparation_pin=prep_pin)
                        if prep.purpose == "DESIGN_SOURCE" else self._preparation.resolve_runtime_target(ctx=ctx, artifact_pin=pin, preparation_pin=prep_pin))
                    if not hasattr(document, "blocks"):
                        continue
                    for item in self._items(identity, document, prep):
                        row["provisions"].append(dict(schema_version="tender_supporting_citation.v1",
                            artifact_pin=pin.model_dump(mode="json"), preparation_pin=prep_pin.model_dump(mode="json"), preparation_purpose=prep.purpose,
                            source_identity_ref=identity.source_identity_ref, source_hash=identity.source_hash,
                            source_identity=identity.model_dump(mode="json", by_alias=True),
                            canonical_source_ir_pin=prep.canonical_source_ir_pin.model_dump(mode="json"),
                            subject=item["subject"], evidence=item["evidence"], exact_quote=item["exact_quote"],
                            resolved_location=item["resolved_location"], provision_label="",
                            selection_origin="REVIEWER_SELECTED", applicability="UNKNOWN"))
                except (FileNotFoundError, KeyError, ValueError, PermissionError):
                    row["preparation_status"] = "UNAVAILABLE"
            references.append(row)
        return references

    def _validate_supporting_citations(self, ctx, citations):
        # Resolve exact historical preparations, never substitute the current head.
        seen = set()
        for citation in citations or ():
            artifact = self._artifacts.resolve(ctx=ctx, artifact_pin=citation.artifact_pin)
            prep = self._preparation.resolve(ctx=ctx, artifact_pin=citation.artifact_pin,
                preparation_pin=citation.preparation_pin, purpose=citation.preparation_purpose)
            identity, document = (self._preparation.resolve_design_source(ctx=ctx, artifact_pin=citation.artifact_pin, preparation_pin=citation.preparation_pin)
                if citation.preparation_purpose == "DESIGN_SOURCE" else self._preparation.resolve_runtime_target(ctx=ctx, artifact_pin=citation.artifact_pin, preparation_pin=citation.preparation_pin))
            item = next((i for i in self._items(identity, document, prep) if i["subject"] == citation.subject.model_dump(mode="json")), None)
            if (item is None or citation.source_identity_ref != identity.source_identity_ref or citation.source_hash != identity.source_hash
                or citation.source_identity != identity.model_dump(mode="json", by_alias=True)
                or citation.canonical_source_ir_pin != prep.canonical_source_ir_pin
                or citation.exact_quote != item["exact_quote"] or citation.evidence != item["evidence"]
                or citation.resolved_location != item["resolved_location"] or citation.provision_label != ""):
                raise ValueError("SUPPORTING_CITATION_MISMATCH")
            key = (citation.preparation_pin.artifact_hash, citation.subject.evidence_ref)
            if key in seen:
                raise ValueError("DUPLICATE_SUPPORTING_CITATION")
            seen.add(key)

    def context(self, *, ctx, cursor_ref=None, journey_ref=None, episode_ref=None):
        cursor = self._cursor(ctx, cursor_ref)
        result = dict(context_projection_version="tender_review_context_projection.v2", session_ref=self._session, cursor_ref=cursor.cursor_ref,
                      material_context_revision=cursor.revision, target_artifact_pin=None, preparation_pin=None,
                      source_identity_ref=None, source_hash=None, canonical_source_ir_pin=None, document_name=None,
                      source_items=[], records=[], reports=[], automatic_status="NOT_SELECTED",
                      limitations=["AUTOMATION_LIVE_UNQUALIFIED", "LEGAL_CORRECTNESS_UNQUALIFIED", "HUMAN_VALUE_UNMEASURED"])
        target = self._target(ctx, cursor)
        if target is None:
            result["status"] = "EMPTY"
            return result
        entry, preparation, identity, document, artifact = target
        result.update(status="READY", target_artifact_pin=entry.artifact_pin.model_dump(mode="json"),
                      preparation_pin=entry.preparation_pin.model_dump(mode="json"), source_identity_ref=identity.source_identity_ref,
                      source_hash=identity.source_hash, canonical_source_ir_pin=preparation.canonical_source_ir_pin.model_dump(mode="json"),
                      document_name=artifact.original_name, media_type=artifact.media_type, source_items=self._items(identity, document, preparation))
        result["reference_materials"] = self._reference_materials(ctx, entry.artifact_pin)
        from services.tender_documentary_candidate import read_authored_query
        result['authored_queries'] = []
        for item in self._artifacts.list(ctx=ctx).artifacts:
            if item.artifact.media_type != 'text/plain' or item.access_state != 'AVAILABLE':
                continue
            pin = ArtifactPinV1(artifact_kind='workspace_artifact', artifact_ref=item.artifact.artifact_ref, artifact_hash=item.artifact.artifact_hash)
            try:
                query = read_authored_query(self, ctx, pin)
            except (ValueError, OSError):
                continue
            result['authored_queries'].append(dict(pin=pin.model_dump(mode='json'), nomination=query.nomination, authorship=query.authorship))
        record_page = self.list_records(ctx=ctx, target_pin=entry.artifact_pin, limit=100)
        result["records"] = record_page["records"]
        result["records_continuation"] = record_page["continuation"]
        result["reports"] = self._list_reports(ctx, entry.artifact_pin)
        episode_pin = None
        if (journey_ref is None) != (episode_ref is None):
            raise ValueError("PAIRED_EPISODE_SELECTION_REQUIRED")
        if journey_ref is not None:
            episode = self._episode(ctx, journey_ref=journey_ref, episode_ref=episode_ref, target=target)
            episode_pin = self.episode_pin(episode)
            result["automatic_status"] = "SAVED_NATIVE_EPISODE_LIVE_UNQUALIFIED"
            result["episode_pin"] = episode_pin.model_dump(mode="json")
            result["journey_ref"] = journey_ref
            result["native_candidates"] = []
            result["native_findings"] = []
            for finding in episode.inspection_result.governed_findings:
                frozen_finding = finding.model_dump(mode="json", by_alias=True)
                result["native_findings"].append(frozen_finding)
                for frozen in frozen_finding["evidenceRefs"]:
                    evidence = EvidenceRef.model_validate(dict(frozen))
                    selected = self._native_item(target, evidence)
                    selected["subject"] = dict(kind="GOVERNED_FINDING", **selected["subject"], journey_ref=journey_ref,
                                               episode_pin=episode_pin.model_dump(mode="json"),
                                               finding_pin=dict(artifact_kind="governed_frame_finding", artifact_ref=finding.finding_id, artifact_hash=finding.finding_hash))
                    selected["machine_status"] = finding.decision_status
                    result["native_candidates"].append(selected)
        result["record_head_pins"] = [pin.model_dump(mode="json") for pin in self._task_heads(ctx, entry.artifact_pin, episode_pin=episode_pin)]
        return result

    def _get(self, ctx, pin, model):
        self._scope(ctx)
        if pin.artifact_kind != model.KIND:
            raise ValueError("DOCUMENTARY_PIN_MISMATCH")
        value = self._store.get(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
                                artifact_ref=pin.artifact_ref, contract_type=model)
        if value is None:
            raise KeyError("DOCUMENTARY_ARTIFACT_NOT_FOUND")
        if value.pin() != pin:
            raise ValueError("DOCUMENTARY_PIN_MISMATCH")
        if value.session_ref != self._session or value.recorded_by != ctx.user_id:
            raise PermissionError("FORBIDDEN_DOCUMENTARY_OWNER")
        return value

    def get_record(self, *, ctx, pin):
        return self._get(ctx, pin, TenderReviewRecordV1)

    def get_report(self, *, ctx, pin):
        self._scope(ctx)
        row = self._store._connection.execute(sql_query("SELECT schema_version FROM ontology_build_artifacts WHERE organization_id=? AND workspace_id=? AND artifact_ref=?",'SELECT schema_version FROM ontology_build_artifacts WHERE organization_id=%s AND workspace_id=%s AND artifact_ref=%s'), (ctx.organization_id, ctx.workspace_id, pin.artifact_ref)).fetchone()
        model = TenderReviewReportV2 if row and row[0] == 'tender_review_report.v2' else TenderReviewReportV1
        return self._get(ctx, pin, model)

    def _rows(self, ctx, target_pin, schema, limit, after=""):
        self._scope(ctx)
        return self._store._connection.execute(
            sql_query("SELECT payload_json FROM ontology_build_artifacts WHERE organization_id=? AND workspace_id=? AND schema_version=? "
            "AND json_extract(payload_json,'$.session_ref')=? AND json_extract(payload_json,'$.recorded_by')=? "
            "AND json_extract(payload_json,'$.target_artifact_pin.artifact_ref')=? AND json_extract(payload_json,'$.target_artifact_pin.artifact_hash')=? "
            "AND artifact_ref>? ORDER BY artifact_ref LIMIT ?","SELECT payload_json FROM ontology_build_artifacts WHERE organization_id=%s AND workspace_id=%s AND schema_version=%s AND (payload_json::jsonb #>> '{session_ref}')=%s AND (payload_json::jsonb #>> '{recorded_by}')=%s AND (payload_json::jsonb #>> '{target_artifact_pin,artifact_ref}')=%s AND (payload_json::jsonb #>> '{target_artifact_pin,artifact_hash}')=%s AND artifact_ref>%s ORDER BY artifact_ref LIMIT %s"),
            (ctx.organization_id, ctx.workspace_id, schema, self._session, ctx.user_id, target_pin.artifact_ref, target_pin.artifact_hash, after, limit),
        ).fetchall()

    def list_records(self, *, ctx, target_pin, limit=100, continuation=None):
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100 or target_pin.artifact_kind != "workspace_artifact":
            raise ValueError("INVALID_LIST_FILTER")
        binding = content_hash(dict(scope=self._scope(ctx), target=target_pin.model_dump(mode="json")))
        after = ""
        if continuation:
            try:
                token = json.loads(base64.urlsafe_b64decode(continuation.encode()))
                if set(token) != {"scope", "after"} or token["scope"] != binding or not isinstance(token["after"], str):
                    raise ValueError()
                after = token["after"]
            except (ValueError, TypeError, UnicodeError) as exc:
                raise ValueError("INVALID_CONTINUATION") from exc
        rows = self._rows(ctx, target_pin, "tender_review_record.v1", limit + 1, after)
        records = [TenderReviewRecordV1.model_validate_json(row[0]) for row in rows[:limit]]
        for value in records:
            self.get_record(ctx=ctx, pin=value.pin())
        next_cursor = None
        if len(rows) > limit:
            next_cursor = base64.urlsafe_b64encode(canonical_json(dict(scope=binding, after=records[-1].record_ref)).encode()).decode()
        return dict(records=[record.model_dump(mode="json") for record in records], continuation=next_cursor)

    def _list_reports(self, ctx, target_pin):
        result = []
        for model, schema in ((TenderReviewReportV1, 'tender_review_report.v1'), (TenderReviewReportV2, 'tender_review_report.v2')):
            for row in self._rows(ctx, target_pin, schema, 100):
                result.append(self.get_report(ctx=ctx, pin=model.model_validate_json(row[0]).pin()).model_dump(mode='json'))
        return result

    def _basis(self, ctx, target):
        entry, preparation, identity, _document, _artifact = target
        return dict(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id, session_ref=self._session,
                    task_key="sha256:" + content_hash(dict(session_ref=self._session, target_artifact_pin=entry.artifact_pin.model_dump(mode="json"))),
                    target_artifact_pin=entry.artifact_pin.model_dump(mode="json"), preparation_pin=entry.preparation_pin.model_dump(mode="json"),
                    source_identity_ref=identity.source_identity_ref, source_hash=identity.source_hash,
                    canonical_source_ir_pin=preparation.canonical_source_ir_pin.model_dump(mode="json"), recorded_by=ctx.user_id,
                    recorded_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

    def save_record(self, *, ctx, command):
        command = TenderReviewRecordCommandV1.model_validate(command.model_dump(mode="json"))
        replay = self._replay(ctx, command, TenderReviewRecordV1)
        if replay is not None:
            return replay
        cursor = self._cursor(ctx, command.cursor_ref)
        target = self._target(ctx, cursor, command.target_artifact_pin, command.preparation_pin)
        entry, preparation, identity, document, _artifact = target
        machine_status = None
        qualification = "SOURCE_DOCUMENTARY_ONLY"
        if isinstance(command.subject, GovernedFindingSubjectV1):
            episode = self._episode(ctx, journey_ref=command.subject.journey_ref,
                                    episode_ref=command.subject.episode_pin.artifact_ref, target=target)
            if self.episode_pin(episode) != command.subject.episode_pin:
                raise ValueError("EPISODE_PIN_MISMATCH")
            findings = [finding for finding in episode.inspection_result.governed_findings
                        if finding.finding_id == command.subject.finding_pin.artifact_ref
                        and finding.finding_hash == command.subject.finding_pin.artifact_hash]
            if len(findings) != 1 or command.subject.finding_pin.artifact_kind != "governed_frame_finding":
                raise ValueError("FINDING_NOT_IN_EPISODE")
            finding = findings[0]
            evidence = [EvidenceRef.model_validate(dict(frozen)) for frozen in finding.model_dump(mode="json", by_alias=True)["evidenceRefs"]
                        if frozen.get("evidenceRef") == command.subject.evidence_ref]
            if len(evidence) != 1:
                raise ValueError("FINDING_EVIDENCE_NOT_IN_EPISODE")
            item = self._native_item(target, evidence[0])
            machine_status = finding.decision_status
            qualification = "NATIVE_FINDING_LIVE_UNQUALIFIED"
        else:
            items = self._items(identity, document, preparation)
            item = next((item for item in items if item["subject"]["source_block_ref"] == command.subject.source_block_ref), None)
        if item is None or (item["subject"]["source_block_ref"], item["subject"]["evidence_ref"], item["subject"]["selector_hash"]) != (command.subject.source_block_ref, command.subject.evidence_ref, command.subject.selector_hash):
            raise ValueError("SOURCE_SUBJECT_MISMATCH")
        self._validate_supporting_citations(ctx, command.supporting_citations)
        fields = command.model_dump(mode="json", exclude={"cursor_ref", "expected_material_context_revision", "target_artifact_pin", "preparation_pin"})
        value = TenderReviewRecordV1.create(**self._basis(ctx, target), **fields, schema_version="tender_review_record.v1",
                                            predecessor_record_pin=fields["expected_head_pin"], exact_quote=item["exact_quote"],
                                            resolved_location=item["resolved_location"], input_qualification=qualification, machine_status=machine_status)

        def verify(*, replay):
            current = self._cursor(ctx, command.cursor_ref)
            if not replay:
                if current.revision != command.expected_material_context_revision:
                    raise ValueError("STALE_MATERIAL_CONTEXT")
                self._target(ctx, current, command.target_artifact_pin, command.preparation_pin)
                self._validate_supporting_citations(ctx, command.supporting_citations)
        pin = self._store.publish_tender_review(artifact=value, command=command, verify_before_commit=verify)
        return self.get_record(ctx=ctx, pin=pin)

    def _replay(self, ctx, command, model):
        self._scope(ctx)
        request_hash = "sha256:" + content_hash(command.model_dump(mode="json"))
        pins = []
        for alias in ("request:" + command.request_id, "key:" + command.idempotency_key):
            row = self._store._connection.execute(
                sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index "
                "WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?",'SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s'),
                (ctx.organization_id, ctx.workspace_id, ctx.user_id, model.KIND.upper(), alias),
            ).fetchone()
            if row is not None:
                if row[0] != request_hash:
                    raise ValueError("REQUEST_CONFLICT")
                pins.append(ArtifactPinV1(artifact_kind=model.KIND, artifact_ref=row[1], artifact_hash=row[2]))
        if pins:
            if any(pin != pins[0] for pin in pins):
                raise ValueError("REQUEST_CONFLICT")
            return self._get(ctx, pins[0], model)
        return None

    def _record_head(self, record):
        row = self._store._connection.execute(
            sql_query("SELECT record_ref,record_hash FROM tender_review_heads WHERE organization_id=? AND workspace_id=? "
            "AND session_ref=? AND target_ref=? AND target_hash=? AND subject_hash=? AND actor_ref=?",'SELECT record_ref,record_hash FROM tender_review_heads WHERE organization_id=%s AND workspace_id=%s AND session_ref=%s AND target_ref=%s AND target_hash=%s AND subject_hash=%s AND actor_ref=%s'),
            (record.organization_id, record.workspace_id, record.session_ref, record.target_artifact_pin.artifact_ref,
             record.target_artifact_pin.artifact_hash, content_hash(record.subject.model_dump(mode="json")), record.recorded_by),
        ).fetchone()
        return None if row is None else ArtifactPinV1(artifact_kind="tender_review_record", artifact_ref=row[0], artifact_hash=row[1])

    def save_report(self, *, ctx, command):
        candidate_command = isinstance(command, TenderCandidateReportCommandV1)
        command_model = TenderCandidateReportCommandV1 if candidate_command else TenderReviewReportCommandV1
        report_model = TenderReviewReportV2 if candidate_command else TenderReviewReportV1
        command = command_model.model_validate(command.model_dump(mode="json"))
        replay = self._replay(ctx, command, report_model)
        if replay is not None:
            return replay
        cursor = self._cursor(ctx)
        target = self._target(ctx, cursor, command.target_artifact_pin, command.preparation_pin)
        records = tuple(self.get_record(ctx=ctx, pin=pin) for pin in command.review_record_pins)
        for record in records:
            if record.target_artifact_pin != command.target_artifact_pin or record.preparation_pin != command.preparation_pin:
                raise ValueError("REPORT_INPUT_TASK_MISMATCH")
        subjects = [content_hash(record.subject.model_dump(mode="json")) for record in records]
        if len(subjects) != len(set(subjects)):
            raise ValueError("REPORT_CONFLICTING_SUBJECT_HEADS")
        episode = None
        if command.episode_pin is not None:
            episode = self._episode(ctx, episode_ref=command.episode_pin.artifact_ref, target=target)
            if self.episode_pin(episode) != command.episode_pin:
                raise ValueError("EPISODE_PIN_MISMATCH")
        for record in records:
            if isinstance(record.subject, GovernedFindingSubjectV1) and record.subject.episode_pin != command.episode_pin:
                raise ValueError("REPORT_EPISODE_MISMATCH")
        inventory = [] if episode is None else [dict(artifact_kind="governed_frame_finding", artifact_ref=finding.finding_id, artifact_hash=finding.finding_hash)
                                               for finding in episode.inspection_result.governed_findings]
        reviewed = {(record.subject.finding_pin.artifact_ref, record.subject.finding_pin.artifact_hash)
                    for record in records if isinstance(record.subject, GovernedFindingSubjectV1) and record.status == "RECORDED"}
        unreviewed = [pin for pin in inventory if (pin["artifact_ref"], pin["artifact_hash"]) not in reviewed]
        if command.predecessor_report_pin is not None:
            predecessor = self.get_report(ctx=ctx, pin=command.predecessor_report_pin)
            if predecessor.target_artifact_pin != command.target_artifact_pin:
                raise ValueError("REPORT_INPUT_TASK_MISMATCH")
        compliance = []
        if episode is not None:
            from pydantic import TypeAdapter
            from services.tender_legal_compliance_service import TenderLegalComplianceService
            compliance = TypeAdapter(list).dump_python(list(
                TenderLegalComplianceService.read_episode(store=self._store, episode=episode)), mode="json")
        from contracts.tender_legal_compliance import freeze_assessment_selection
        selections = [item.model_dump(mode="json") for item in freeze_assessment_selection(inventory=inventory, entries=compliance)]
        extra = {}
        if candidate_command:
            if cursor.revision != command.expected_material_context_revision or cursor.cursor_ref != command.cursor_ref:
                raise ValueError('STALE_MATERIAL_CONTEXT')
            from services.tender_documentary_candidate import prepare_candidate, attach_candidate
            prepared_candidate = prepare_candidate(self, ctx, command, target)
            packet = self._invoke_candidate_query(ctx, prepared_candidate[0].request)
            extra['documentary_candidates'] = [attach_candidate(command, prepared_candidate, packet).model_dump(mode='json')]
        value = report_model.create(**self._basis(ctx, target), **extra, schema_version="tender_review_report.v2" if candidate_command else "tender_review_report.v1",
                                            request_id=command.request_id, idempotency_key=command.idempotency_key,
                                            review_record_pins=[pin.model_dump(mode="json") for pin in command.review_record_pins],
                                            records=[record.model_dump(mode="json") for record in records],
                                            predecessor_report_pin=None if command.predecessor_report_pin is None else command.predecessor_report_pin.model_dump(mode="json"),
                                            episode_pin=None if episode is None else command.episode_pin.model_dump(mode="json"),
                                            native_context_pin=None if episode is None else self._inspection_pin(episode.inspection_context, "runtime_semantic_inspection_context.v1"),
                                            native_result_pin=None if episode is None else self._inspection_pin(episode.inspection_result, "runtime_semantic_inspection_result.v1"),
                                            native_basis_snapshot=None if episode is None else dict(docx_projection_policy="tender_native_report_docx.v2", findings=[finding.model_dump(mode="json", by_alias=True) for finding in episode.inspection_result.governed_findings],
                                                                                                  source_regions=[region.model_dump(mode="json") for region in episode.inspection_context.source_regions],
                                                                                                  compliance_decisions=compliance,
                                                                                                  compliance_selection=selections,
                                                                                                  compliance_originals=self._compliance_originals(ctx=ctx, target=target, entries=compliance)),
                                            native_finding_inventory=inventory, unreviewed_finding_pins=unreviewed,
                                            coverage="MANUAL_RECORDED_SCOPE_AUTOMATION_NOT_RUN" if episode is None else "NATIVE_CANDIDATE_SCOPE_LIVE_UNQUALIFIED",
                                            limitations=["AUTOMATION_NOT_RUN" if episode is None else "AUTOMATION_LIVE_UNQUALIFIED", "LEGAL_CORRECTNESS_UNQUALIFIED", "HUMAN_VALUE_UNMEASURED", "INCOMPLETE_SOURCE_REVIEW_INVENTORY"],
                                            projection_version="tender_review_report_projection.v2" if candidate_command else "tender_review_report_projection.v1")

        def verify(*, replay):
            current = self._cursor(ctx, cursor.cursor_ref)
            if not replay:
                if current.revision != command.expected_material_context_revision:
                    raise ValueError("STALE_MATERIAL_CONTEXT")
                self._target(ctx, current, command.target_artifact_pin, command.preparation_pin)
                for record in records:
                    if self._record_head(record) != record.pin():
                        raise ValueError("STALE_REVIEW_HEAD")
                if {pin.artifact_ref for pin in self._task_heads(ctx, command.target_artifact_pin, episode_pin=command.episode_pin)} != {pin.artifact_ref for pin in command.review_record_pins}:
                    raise ValueError("REPORT_INCOMPLETE_MANIFEST")
        pin = self._store.publish_tender_review(artifact=value, command=command, verify_before_commit=verify)
        return self.get_report(ctx=ctx, pin=pin)

    def _invoke_candidate_query(self, ctx, request):
        import threading
        if threading.current_thread() is not threading.main_thread():
            raise ValueError('NATIVE_MAIN_THREAD_REQUIRED')
        from services.source_text_query_service import SourceTextQueryService
        from contracts.knowledge_classification import RuntimeContextDTO
        from agents.hermes_boundary_provider import SemantierExecutionBoundaryProvider
        from gateway.execution_boundary import ExecutionBoundaryRequest
        actor = RuntimeContextDTO(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
            actor_id=ctx.user_id, session_id=self._session)
        boundary = SemantierExecutionBoundaryProvider().resolve(ExecutionBoundaryRequest(
            source='tender-documentary', session_id=self._session, user_id=ctx.user_id, workspace_id=ctx.workspace_id))
        root = ctx.workspace_root.parent.parent
        db = self._preparation._design._extraction._db_path
        service = SourceTextQueryService(runtime_root=root, db_path=db, authority_resolver=self._artifacts._authority)
        return service.invoke(request, actor_context=actor, boundary=boundary)

    def export_report(self, *, ctx, pin):
        # Wire digest is independent of the canonical self-bearing artifact hash.
        return canonical_json(self.get_report(ctx=ctx, pin=pin).model_dump(mode="json")).encode("utf-8")


    def export_native_docx(self, *, ctx, report_pin):
        """Publish only from the exact frozen native report, through workspace custody."""
        from hashlib import sha256
        from io import BytesIO
        from contracts.tender_review import TenderReviewDocxDerivativeV1
        from storage.workspace_artifact_blob_store import WorkspaceArtifactBlobStore
        from services.tender_review_docx import render_native_report_docx
        report = self.get_report(ctx=ctx, pin=report_pin)
        artifact = self._artifacts.resolve(ctx=ctx, artifact_pin=report.target_artifact_pin)
        if artifact.media_type != "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            raise ValueError("DOCX_ORIGINAL_REQUIRED")
        identity, document = self._preparation.resolve_runtime_target(ctx=ctx,
            artifact_pin=report.target_artifact_pin, preparation_pin=report.preparation_pin)
        if (identity.source_identity_ref, identity.source_hash) != (report.source_identity_ref, report.source_hash):
            raise ValueError("REPORT_SOURCE_PIN_MISMATCH")
        blobs = WorkspaceArtifactBlobStore(ctx=ctx, max_bytes=self._artifacts._max_bytes)
        with blobs.exact_get(byte_owner_ref=artifact.byte_owner_ref, expected_hash=artifact.content_hash,
                             expected_size=artifact.byte_size) as source:
            rendered, manifest = render_native_report_docx(report=report, document=document, source_bytes=source.read())
        digest = sha256(rendered).hexdigest()
        byte_ref = f"artifacts/workspace-originals/sha256/{digest[:2]}/{digest}"
        fields = {name: getattr(report, name) for name in (
            "organization_id", "workspace_id", "session_ref", "task_key", "target_artifact_pin",
            "preparation_pin", "source_identity_ref", "source_hash", "canonical_source_ir_pin",
            "recorded_by", "recorded_at", "documentary_status")}
        # Retain the report timestamp and actor: retries do not create successor opinions.
        from pydantic import TypeAdapter
        fields = TypeAdapter(dict).dump_python(fields, mode="json")
        derivative = TenderReviewDocxDerivativeV1.create(**fields,
            schema_version="tender_review_docx_derivative.v1", report_pin=report_pin.model_dump(mode="json"),
            request_id="native-docx:" + report.report_hash[7:], idempotency_key="native-docx:" + report.report_hash[7:],
            content_hash="sha256:" + digest, byte_size=len(rendered), byte_owner_ref=byte_ref,
            projection_manifest=manifest)
        self._scope(ctx)
        stage = blobs.stage(byte_stream=BytesIO(rendered), expected_hash=derivative.content_hash,
            expected_size=derivative.byte_size, original_name="annotated-tender.docx",
            declared_media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        try:
            self._scope(ctx)
            if blobs.publish(stage=stage, artifact_ref=derivative.derivative_ref) != byte_ref:
                raise ValueError("NATIVE_DOCX_BYTE_OWNER_MISMATCH")
            self._store.put(derivative)
        finally:
            blobs.discard_stage(stage_ref=stage.stage_ref)
        return derivative

    def download_native_docx(self, *, ctx, pin):
        """Read the retained derivative; never rerun its projection on download."""
        from contracts.tender_review import TenderReviewDocxDerivativeV1
        from storage.workspace_artifact_blob_store import WorkspaceArtifactBlobStore
        derivative = self._get(ctx, pin, TenderReviewDocxDerivativeV1)
        report = self.get_report(ctx=ctx, pin=derivative.report_pin)
        expected_decisions = [
            raw["decision"] for raw in (report.native_basis_snapshot or {}).get("compliance_decisions", ())]
        from contracts.semantic_kernel import DecisionRecord
        expected_pins = [DecisionRecord.model_validate(raw).pin().model_dump(mode="json") for raw in expected_decisions]
        if ((derivative.target_artifact_pin, derivative.preparation_pin, derivative.source_hash,
             derivative.source_identity_ref, derivative.canonical_source_ir_pin) != (
            report.target_artifact_pin, report.preparation_pin, report.source_hash,
            report.source_identity_ref, report.canonical_source_ir_pin)
            or derivative.projection_manifest.get("decision_pins") != expected_pins
            or derivative.projection_manifest.get("native_projection_policy") != (report.native_basis_snapshot or {}).get("docx_projection_policy", "tender_documentary_opinion_docx.v1" if report.episode_pin is None and report.records else "tender_native_report_docx.v1")):
            raise ValueError("NATIVE_DOCX_REPORT_BINDING_MISMATCH")
        blobs = WorkspaceArtifactBlobStore(ctx=ctx, max_bytes=self._artifacts._max_bytes)
        with blobs.exact_get(byte_owner_ref=derivative.byte_owner_ref, expected_hash=derivative.content_hash,
                             expected_size=derivative.byte_size) as stream:
            return derivative, stream.read()

    @staticmethod
    def episode_pin(episode):
        return ArtifactPinV1(artifact_kind="tender_journey_inspection_episode", artifact_ref=episode.transition_id,
                             artifact_hash="sha256:" + content_hash(episode.model_dump(mode="json")))

    @staticmethod
    def _inspection_pin(artifact, kind):
        ref_field, hash_field = ("context_ref", "context_hash") if kind.split(".")[0].endswith("context") else ("result_ref", "result_hash")
        return dict(artifact_kind=kind, artifact_ref=getattr(artifact, ref_field), artifact_hash=getattr(artifact, hash_field))

    def _episode(self, ctx, *, episode_ref, target, journey_ref=None):
        self._scope(ctx)
        if self._journeys is None or self._inspection is None:
            raise ValueError("NATIVE_EPISODE_NOT_SELECTED")
        head = self._journeys._store.get_session_journey(organization_id=ctx.organization_id, workspace_id=ctx.workspace_id,
                                                        actor_ref="user:" + ctx.user_id, session_ref=self._session)
        if head is None or journey_ref is not None and head.journey_id != journey_ref:
            raise ValueError("EPISODE_SESSION_MISMATCH")
        episode = self._journeys.resolve_inspection_episode(ctx=ctx, journey_id=head.journey_id, transition_id=episode_ref,
                                                           inspection_service=self._inspection, authority_resolver=self._artifacts._authority)
        _entry, preparation, identity, _document, _artifact = target
        result = episode.inspection_result
        if (result.source_identity_ref, result.source_hash, result.canonical_source_ir_ref, result.canonical_source_ir_hash) != (
            identity.source_identity_ref, identity.source_hash, preparation.canonical_source_ir_pin.artifact_ref, preparation.canonical_source_ir_pin.artifact_hash):
            raise ValueError("EPISODE_SOURCE_MISMATCH")
        return episode

    @staticmethod
    def _native_item(target, evidence):
        _entry, preparation, identity, document, _artifact = target
        if evidence.source_identity_ref != identity.source_identity_ref or (evidence.organization_id, evidence.workspace_id) != (identity.organization_id, identity.workspace_id):
            raise ValueError("FINDING_SOURCE_MISMATCH")
        location = resolve_evidence_ref(evidence, document, artifact_ref=preparation.canonical_source_ir_pin.artifact_ref,
                                        selector_hash=evidence.selector_hash)
        if location.status not in ("exact", "relocated_exact"):
            raise ValueError("SOURCE_SUBJECT_MISMATCH")
        return dict(subject=dict(source_block_ref=evidence.provenance.get("source_block_id"),
                                 evidence_ref=evidence.evidence_ref, selector_hash=evidence.selector_hash),
                    exact_quote=location.matched_text, resolved_location=location.location,
                    evidence=evidence.model_dump(mode="json", by_alias=True))


    def _task_heads(self, ctx, target_pin, *, episode_pin=None):
        rows = self._store._connection.execute(
            sql_query("SELECT record_ref,record_hash FROM tender_review_heads WHERE organization_id=? AND workspace_id=? AND session_ref=? "
            "AND target_ref=? AND target_hash=? AND actor_ref=? ORDER BY record_ref",'SELECT record_ref,record_hash FROM tender_review_heads WHERE organization_id=%s AND workspace_id=%s AND session_ref=%s AND target_ref=%s AND target_hash=%s AND actor_ref=%s ORDER BY record_ref'),
            (ctx.organization_id, ctx.workspace_id, self._session, target_pin.artifact_ref, target_pin.artifact_hash, ctx.user_id),
        ).fetchall()
        pins = tuple(ArtifactPinV1(artifact_kind="tender_review_record", artifact_ref=row[0], artifact_hash=row[1]) for row in rows)
        return tuple(pin for pin in pins
                     if not isinstance((record := self.get_record(ctx=ctx, pin=pin)).subject, GovernedFindingSubjectV1)
                     or record.subject.episode_pin == episode_pin)
