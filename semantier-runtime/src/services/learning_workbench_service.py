"""Read-only learning projections over existing exact artifact owners."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from urllib.parse import quote

from agents.auth_session import RequestContext
from contracts.learning_workbench import ProcedureInspectionV1, SkillArtifactVersionV1
from contracts.ontology_building import ArtifactPinV1
from storage.ontology_build_store import OntologyBuildStore


class LearningWorkbenchService:
    def __init__(self, *, store: OntologyBuildStore, governance=None):
        self._store = store
        self._governance = governance

    def _context_owner(self):
        from services.retained_context_item_service import RetainedContextItemService
        return RetainedContextItemService(store=self._store)

    def _procedure_owner(self):
        from services.learned_skill_publication_service import LearnedSkillPublicationService
        if self._governance is None:
            raise RuntimeError("Canonical procedure governance owner required")
        return LearnedSkillPublicationService(store=self._store, governance=self._governance)

    def _intake_owner(self):
        from services.hermes_skill_review_service import HermesSkillReviewService
        return HermesSkillReviewService(procedures=self._procedure_owner())

    def _contribution_owner(self):
        from services.organization_learning_contribution_service import OrganizationLearningContributionService
        return OrganizationLearningContributionService(procedures=self._procedure_owner(), governance=self._governance)

    def _review_state_owner(self):
        from services.learning_review_state_service import LearningReviewStateService
        return LearningReviewStateService(store=self._store)

    @staticmethod
    def _present(raw: dict, kind: str) -> dict:
        """Normalize owner projections without advancing any lifecycle dimension."""
        dimensions = ("proposal", "qualification", "admission", "activation", "runtime_availability",
                      "applicability", "recorded_use", "causal_attribution")
        states = {name: "unknown" for name in dimensions}
        states.update(raw.get("states") or {})
        for name in dimensions:
            if name in raw:
                value = raw[name]
                states[name] = "this workspace" if isinstance(value, dict) else str(value)
        result = {**raw, "kind": kind,
                  "ref": raw.get("ref") or raw.get("candidate_ref") or raw.get("artifact_ref") or raw.get("item_ref"),
                  "title": raw.get("title") or raw.get("skill_id") or ("Retained context" if kind == "context" else "Proposed change"),
                  "leading_answer": raw.get("leading_answer") or "Exact retained version. Future availability is not confirmed.",
                  "decisive_uncertainty": raw.get("decisive_uncertainty"), "states": states,
                  "evidence": raw.get("evidence") or [],
                  "counter_evidence": raw.get("counter_evidence") or raw.get("counterevidence") or [],
                  "history": raw.get("history") or [], "available_actions": raw.get("available_actions") or []}
        result["hash"] = raw.get("hash") or raw.get("candidate_hash") or raw.get("artifact_hash")
        if kind in ('pending', 'procedure', 'candidate', 'contribution'):
            from eos.evaluation_progress import selected_profile_coverage, source_readiness
            result['component_ids'] = raw.get('component_ids', ['C6'])
            result['native_target_kind'] = 'skill_package'
            result['native_target_ref'] = raw.get('skill_id') or result['ref']
            result['baseline_coverage'] = selected_profile_coverage()
            result['source_readiness'] = source_readiness()
            effect_scope = raw.get('effect_scope') or raw.get('applicability')
            result['effect_scope'] = 'this workspace' if isinstance(effect_scope, dict) else effect_scope
            result['declaration_state'] = 'declared'
            result['consumed_state'] = 'unknown'
            result['enforced_state'] = 'unknown'
            result['adapter_coverage'] = 'native_skill_package' if kind != 'contribution' else 'review_only'
        if result.get('origin') and not (isinstance(result['origin'], dict) and result['origin'].get('session_key') and result['origin'].get('message_id')):
            result['native_origin'] = result.pop('origin')
        if kind == 'candidate':
            if raw.get('disposition') != 'REJECTED':
                result['available_actions'] = list(dict.fromkeys([*result['available_actions'], *(['preview_organization_contribution'] if raw.get('candidate_kind') != 'CHANGE_SET' else []), 'defer_candidate']))
            result['destination_organization_id'] = raw.get('applicability', {}).get('organization_id')
        if kind == "result":
            result["learning_observations"] = raw.get("learning_observations", [])
            if raw.get("learning_observation_error"):
                result["learning_observation_error"] = raw["learning_observation_error"]
        if kind == "context":
            result['component_ids'] = ['C3']
            result["title"] = (raw.get("content") or "Suppressed context item").splitlines()[0][:100]
            result["available_actions"] = [{"correct": "correct_context", "suppress": "suppress_context"}.get(action, action)
                                           for action in result["available_actions"]]
            result["decisive_uncertainty"] = "Use in this result is not recorded. Changes affect future snapshots only."
        return result

    def _inventory_snapshot(self, *, context: RequestContext) -> dict:
        self._require_personal_scope(context)
        pending = [self._present(item, 'pending') for item in self._intake_owner().pending(context)]
        if not context.organization_id:
            return {"objects": pending, "notices": self._review_state_owner().notices(context, pending),
                    "unavailable_reason": "ORGANIZATION_LEARNING_UNAVAILABLE"}
        retained = self._context_owner()
        objects = [self._present({**item, "history": retained.history(context=context, item_ref=item["item_ref"])}, "context")
                   for item in retained.list_items(context=context)]
        owner = self._procedure_owner()
        for item in owner.inventory(context):
            objects.append(self._present(item, item.get("object_kind", item.get("kind", "procedure"))))
        objects.extend(pending)
        objects.extend(self._present(item, 'contribution') for item in self._contribution_owner().inventory(context))
        unique = {(item['kind'], item['ref'], item.get('hash')): item for item in objects}
        return {"objects": list(unique.values()), "notices": self._review_state_owner().notices(context, unique.values())}

    def inventory(self, *, context: RequestContext, limit: int = 100, cursor: str | None = None) -> dict:
        """Bounded exact-revision pages; changed snapshots require an explicit refresh."""
        from eos.artifact_hashing import content_hash
        self._require_personal_scope(context)
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError('INVALID_INVENTORY_LIMIT')
        scope = content_hash({'organization_id':context.organization_id,
            'workspace_id':context.workspace_id,'user_id':context.user_id})
        decoded = None
        if cursor is not None:
            try:
                decoded = json.loads(bytes.fromhex(cursor).decode('utf-8'))
                if (not isinstance(decoded, dict) or set(decoded) != {'v','scope','inventory_hash','offset','limit'}
                    or type(decoded['v']) is not int or decoded['v'] != 1 or decoded['scope'] != scope
                    or type(decoded['limit']) is not int or decoded['limit'] != limit
                    or type(decoded['offset']) is not int or decoded['offset'] <= 0
                    or decoded['offset'] % limit != 0):
                    raise ValueError('invalid cursor')
            except (ValueError, UnicodeError, TypeError) as exc:
                raise ValueError('INVALID_INVENTORY_CURSOR') from exc
        snapshot = self._inventory_snapshot(context=context)
        unique = {(item['kind'], item['ref'], item.get('hash') or ''):item for item in snapshot['objects']}
        objects = [unique[key] for key in sorted(unique)]
        # Pin values as well as identities: changed lifecycle/read state must not be
        # silently combined with another page's prior snapshot.
        all_notices = sorted(snapshot['notices'], key=lambda notice:notice['id'])
        digest = 'sha256:'+content_hash({'scope':scope,'objects':objects,'notices':all_notices})
        if decoded is not None and decoded['inventory_hash'] != digest:
            raise ValueError('INVENTORY_CURSOR_STALE')
        offset = decoded['offset'] if decoded else 0
        if offset > len(objects):
            raise ValueError('INVALID_INVENTORY_CURSOR')
        page = objects[offset:offset+limit]
        refs = {(item['kind'],item['ref'],item.get('hash')) for item in page}
        notices = [notice for notice in all_notices if (notice['kind'],notice['ref'],notice['hash']) in refs]
        next_cursor = None
        if offset+limit < len(objects):
            next_cursor = json.dumps({'v':1,'scope':scope,'inventory_hash':digest,
                'offset':offset+limit,'limit':limit},sort_keys=True,separators=(',',':')).encode().hex()
        return {**snapshot,'objects':page,'notices':notices,'inventory_hash':digest,
                'next_cursor':next_cursor,'total_count':len(objects),
                'component_facets':self._component_facets(objects)}

    @staticmethod
    def _component_facets(objects):
        """Declared reference roles; missing writers are not fabricated proposals."""
        catalog = [('C1','system_instructions'),('C2','instructional_constraints'),
            ('C3','retained_context'),('C4','tool_use_guidance'),('C5','delegation'),
            ('C6','skill_package'),('C7','helper_scripts'),('C8','runtime_interception')]
        return [{'component_id':component,'reference_role':role,
            'source_ref':'docs/derived/prd-learning-governance-workbench-v1.md#313-eight-reference-roles-mapped-to-the-actual-hermes-architecture',
            'object_count':sum(component in item.get('component_ids',[]) for item in objects),
            'writer_state':'scoped_owner_adapter' if component in ('C3','C6') else 'engineering_required',
            'reason':'Existing scoped retained-context/skill owner; availability remains object-specific.' if component in ('C3','C6') else 'No general review writer for this reference role; use its actual engineering owner.',
            'declaration_state':'reference_role','loaded_state':'unknown','consumed_state':'unknown','enforced_state':'unknown'}
            for component,role in catalog]

    def context_snapshot(self, *, context: RequestContext, session_key: str) -> dict:
        """Return a session-bound safe receipt over retained-context refs/revisions only."""
        self._require_scope(context)
        from agents.webapi.session_routes import (
            _get_workspace_session_detail_primary,
            _resolve_workspace_runtime_session_id,
            _session_belongs_to_workspace,
        )
        runtime_id = _resolve_workspace_runtime_session_id(context, session_key)
        if not _session_belongs_to_workspace(runtime_id, context):
            raise LookupError("LEARNING_RESULT_NOT_FOUND")
        detail = _get_workspace_session_detail_primary(context, runtime_id)
        if detail is None:
            raise LookupError("LEARNING_RESULT_NOT_FOUND")
        rows = self._context_owner().list_items(context=context)
        items = sorted(
            [
                {
                    "item_ref": str(item["item_ref"]),
                    "revision": str(item["revision"]),
                    "runtime_availability": str(item.get("runtime_availability") or "unknown"),
                }
                for item in rows
            ],
            key=lambda item: (item["item_ref"], item["revision"]),
        )
        basis = {
            "session_key": session_key,
            "runtime_session_id": runtime_id,
            "items": items,
        }
        encoded = json.dumps(basis, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        started_at = detail.get("started_at")
        try:
            created_at = datetime.fromtimestamp(float(started_at), tz=timezone.utc).isoformat().replace("+00:00", "Z")
        except (TypeError, ValueError, OSError):
            created_at = None
        return {
            "snapshot_ref": "learning-context-snapshot:" + digest[:24],
            "snapshot_hash": "sha256:" + digest,
            "session_key": session_key,
            "runtime_session_id": runtime_id,
            "session_created_at": created_at,
            "items": items,
            "recorded_use": "unknown",
            "causal_attribution": "unknown",
        }

    def inspect(self, *, context: RequestContext, kind: str, ref: str = "",
                session_key: str | None = None, message_id: str | None = None,
                payload_hash: str | None = None) -> dict:
        if kind not in ("result", "pending"):
            self._require_scope(context)
        if kind == "result":
            if not session_key or not message_id:
                raise ValueError("EXACT_RESULT_LOCATOR_REQUIRED")
            result = self._present(self.inspect_result(context=context, session_key=session_key, message_id=message_id), kind)
        elif kind == 'pending':
            if not payload_hash:
                raise ValueError('EXACT_PENDING_HASH_REQUIRED')
            result = self._present(self._intake_owner().inspect_pending(context, ref, payload_hash), kind)
        elif kind == 'contribution':
            result = self._present(self._contribution_owner().inspect(context, ref), kind)
        elif kind == "context":
            item = next((item for item in self._context_owner().list_items(context=context) if item["item_ref"] == ref), None)
            if item is None:
                raise LookupError("LEARNING_OBJECT_NOT_FOUND")
            result = self._present(item, kind)
            result["history"] = self._context_owner().history(context=context, item_ref=ref)
        elif kind in ("procedure", "candidate"):
            raw = self._procedure_owner().inspect(context, ref)
            if raw.get("object_kind", raw.get("kind")) != kind:
                raise ValueError("LEARNING_OBJECT_KIND_MISMATCH")
            result = self._present(raw, kind)
        else:
            raise ValueError("INVALID_LEARNING_OBJECT_KIND")
        if kind != "result" and (session_key or message_id):
            result["origin"] = self.inspect_result(context=context, session_key=session_key or "",
                message_id=message_id or "")["origin"]
        return result

    def act(self, *, context: RequestContext, operation: str, values: dict) -> dict:
        if operation in ('mark_notice_read', 'defer_candidate'):
            self._require_personal_scope(context)
            objects = self._inventory_snapshot(context=context)['objects']
            changed = self._review_state_owner().set_state(context, ref=values['candidate_ref'],
                expected_hash=values['candidate_hash'], operation=operation,
                request_id=values['request_id'], objects=objects)
            return {'object': changed}
        self._require_scope(context)
        # Refresh current governed membership at command ingress, not from cached context fields.
        from agents.webapi.organization_service import get_user_membership_for_organization
        membership = get_user_membership_for_organization(user_id=context.user_id, organization_id=context.organization_id)
        if membership is None or membership.get('membership_status') != 'active':
            raise PermissionError('FORBIDDEN_LEARNING_SCOPE')
        if operation != 'preview_organization_contribution':
            from eos.artifact_hashing import content_hash
            if not values.get('request_id'):
                raise ValueError('REQUEST_ID_REQUIRED')
            self._store.reserve_learning_command(organization_id=context.organization_id,
                workspace_id=context.workspace_id,actor_ref=context.user_id,request_id=values['request_id'],
                request_hash=content_hash({'operation':operation,'values':values}))
        if operation == 'import_pending':
            result = self._intake_owner().import_pending(context, values['native_id'], values['payload_hash'], values['request_id'])
            return {'object': self._present(result, 'candidate')}
        if operation in ('preview_organization_contribution', 'submit_organization_contribution'):
            owner = self._contribution_owner()
            args = (context, values['candidate_ref'], values['candidate_hash'], values['destination_organization_id'])
            if operation == 'preview_organization_contribution':
                preview = owner.preview(*args)
                result = self.inspect(context=context, kind='candidate', ref=values['candidate_ref'])
                result['contribution_preview'] = preview
                result['available_actions'] = list(dict.fromkeys([
                    *result['available_actions'], 'submit_organization_contribution']))
                return {'object': result}
            result = owner.submit(*args, values['disclosure_hash'], values.get('consent'), values['request_id'])
            return {'object': self._present(result, 'contribution')}
        if operation in ('approve_organization_contribution', 'reject_organization_contribution'):
            result = self._contribution_owner().decide(context, values['candidate_ref'], values['candidate_hash'],
                operation == 'approve_organization_contribution')
            return {'object': self._present(result, 'contribution')}
        origin = values.get("origin")
        if origin:
            origin = self.inspect_result(context=context, session_key=origin["session_key"],
                                         message_id=origin["message_id"])["origin"]
        if operation in ("correct_context", "suppress_context"):
            owner = self._context_owner()
            fields = {name: values[name] for name in ("item_ref", "expected_revision", "request_id")}
            if operation == "correct_context":
                fields["new_content"] = values["new_content"]
            changed = (owner.correct if operation == "correct_context" else owner.suppress)(
                context=context, origin=origin, **fields)
            result = self._present(changed, "context")
            result["history"] = owner.history(context=context, item_ref=values["item_ref"])
        else:
            owner = self._procedure_owner()
            if operation == "propose_procedure":
                result = self._present(owner.propose(context, values["baseline_ref"], values["baseline_hash"],
                    values["new_content"], values["request_id"], origin=origin), "candidate")
            elif operation == "recover_procedure":
                result = self._present(owner.recover(context, values["version_ref"], values["version_hash"],
                    values["request_id"]), "candidate")
            elif operation in ("assess_candidate", "approve_candidate", "activate_candidate", "publish_procedure", "reject_candidate"):
                args = (context, values["candidate_ref"], values["candidate_hash"])
                if operation == "assess_candidate":
                    assessment = owner.assess(*args)
                    result = self._present(owner.inspect(context, values["candidate_ref"]), "candidate")
                    result.update(assessment_ref=assessment["artifact_ref"], assessment_hash=assessment["artifact_hash"])
                elif operation == "approve_candidate":
                    result = self._present(owner.approve(*args, values["assessment_ref"], values["assessment_hash"]), "candidate")
                elif operation == "activate_candidate":
                    result = self._present(owner.activate(*args), "candidate")
                elif operation == 'reject_candidate':
                    result = self._present(owner.reject(*args), 'candidate')
                else:
                    result = self._present(owner.publish(*args), "candidate")
            else:
                raise ValueError("INVALID_LEARNING_OPERATION")
        if origin:
            result["origin"] = origin
        return {"object": result}

    @staticmethod
    def _require_personal_scope(context: RequestContext) -> None:
        if not context.authenticated or not context.user_id or not context.workspace_id:
            raise PermissionError("FORBIDDEN_LEARNING_SCOPE")

    @staticmethod
    def _require_scope(context: RequestContext) -> None:
        if (not context.authenticated or not context.user_id or not context.organization_id
                or not context.workspace_id):
            raise PermissionError("FORBIDDEN_LEARNING_SCOPE")

    @staticmethod
    def _result_source_evidence(context: RequestContext, runtime_id: str, session_key: str, messages: list[dict]) -> dict:
        """Resolve a recorded expense tool result through its existing offline owner."""
        from plugins.document_extraction.storage import BoundSession
        from runtime_paths import workspace_session_runs_root_from_hermes_home
        from services.expense_reimbursement_service import ExpenseReimbursementService
        calls = {call.get("id"): call for message in messages if message.get("role") == "assistant"
                 for call in message.get("tool_calls", []) if isinstance(call, dict)}
        for message in reversed(messages):
            if message.get("role") != "tool" or message.get("tool_name") != "evaluate_expense_reimbursement":
                continue
            try:
                raw = json.loads(message.get("content") or "")
            except (ValueError, TypeError):
                continue
            if not isinstance(raw, dict) or raw.get("schema_version") != "expense_reimbursement_result.v1":
                continue
            try:
                function = calls.get(message.get("tool_call_id"), {}).get("function", {})
                name = function.get("name")
                if name == "tool_call":
                    from tools.tool_search_validation import normalize_tool_call_entries
                    entries, error = normalize_tool_call_entries(json.loads(function.get("arguments", "{}")))
                    if error or len(entries) != 1:
                        raise ValueError("INVALID_RECORDED_TOOL_BINDING")
                    name = entries[0]["name"]
                if name != "evaluate_expense_reimbursement":
                    raise ValueError("INVALID_RECORDED_TOOL_BINDING")
                bound = BoundSession(context.hermes_home, runtime_id,
                    workspace_session_runs_root_from_hermes_home(context.hermes_home, runtime_id).parent)
                result = ExpenseReimbursementService(bound).replay(raw["result_ref"], raw["result_hash"])
                if result != raw:
                    raise ValueError("RECORDED_RESULT_MISMATCH")
                pin = {key: result[key] for key in ("result_ref", "result_hash")}
                return {
                    "source_review": {"kind": "expense_reimbursement", "session_key": session_key, **pin},
                    "evidence": [{"result_pin": pin, "produced_at": result["produced_at"],
                        "tool_message_id": str(message["id"]), "pins": result["pins"],
                        "rule_trace": result.get("rule_trace"), "policy_lineage": result.get("policy_lineage"),
                        "sources": [{key: source[key] for key in ("name", "source_identity_ref", "source_hash")}
                                    for source in (result.get("invoice_source"), result.get("policy_source")) if source]}],
                }
            except (ValueError, KeyError, TypeError, OSError):
                return {"source_evidence_error": "RECORDED_SOURCE_EVIDENCE_INTEGRITY_FAILED"}
        return {}

    def inspect_result(self, *, context: RequestContext, session_key: str, message_id: str) -> dict:
        """Read the recorded result; absent use/causality evidence stays unknown."""
        self._require_personal_scope(context)
        from runtime_paths import workspace_hermes_home_path, workspace_root_path
        from agents.webapi.session_routes import (
            _get_workspace_session_messages_primary, _resolve_workspace_runtime_session_id,
            _session_belongs_to_workspace,
        )
        if (context.hermes_home.resolve() != workspace_hermes_home_path(context.workspace_id).resolve()
                or context.workspace_root.resolve() != workspace_root_path(context.workspace_id).resolve()):
            raise PermissionError("FORBIDDEN_LEARNING_SCOPE")
        runtime_id = _resolve_workspace_runtime_session_id(context, session_key)
        if not _session_belongs_to_workspace(runtime_id, context):
            raise LookupError("LEARNING_RESULT_NOT_FOUND")
        payload = _get_workspace_session_messages_primary(context, runtime_id)
        if payload is None:
            raise LookupError("LEARNING_RESULT_NOT_FOUND")
        selected = next((message for message in payload[1]
                         if str(message.get("id")) == str(message_id) and message.get("role") == "assistant"), None)
        if selected is None:
            raise LookupError("LEARNING_RESULT_NOT_FOUND")
        observations = []
        # Scope this optional evidence to native tool occurrences before this answer,
        # following the last user request. It never attributes the answer to a skill.
        selected_index = payload[1].index(selected)
        prior = payload[1][:selected_index + 1]
        start = max((i for i, m in enumerate(prior) if m.get("role") == "user"), default=0)
        turn_messages = prior[start:]
        observation_error = None
        from agents.workspace_session_logs import list_workspace_session_trajectory
        try:
            if any(row.get("record_type") == "procedure_execution_receipt"
                   for row in list_workspace_session_trajectory(context.hermes_home, runtime_id)):
                from services.personal_procedure_capture_service import PersonalProcedureCaptureService
                capture = PersonalProcedureCaptureService(store=self._store, procedure_owner=self._procedure_owner())
                call_ids = {m.get("tool_call_id") for m in turn_messages if m.get("role") == "tool"}
                observations = [value.model_dump(mode="json") for value in capture.project_session(
                    context=context, session_id=runtime_id, messages=payload[1], reviewed_targets={},
                    active_conflicts=(), tool_call_ids=call_ids)]
        except (ValueError, KeyError, TypeError, OSError, PermissionError):
            # Optional learning metadata cannot revoke an already-recorded result.
            observation_error = "PRODUCER_RECEIPT_INTEGRITY_FAILED"
        return {
            "kind": "result", "ref": f"result:{session_key}:{message_id}", "title": "Recorded result",
            "content": selected.get("content") or "",
            "leading_answer": "This result is unchanged. Procedure use and causal attribution are not recorded.",
            "decisive_uncertainty": "The recorded result has no exact procedure-use or causal-attribution receipt.",
            "states": {"recorded_use": "unknown", "causal_attribution": "unknown",
                       "applicability": "unknown", "runtime_availability": "unknown"},
            "evidence": [], "counter_evidence": [], "history": [],
            **self._result_source_evidence(context, runtime_id, session_key, turn_messages),
            "learning_observations": observations,
            "learning_observation_error": observation_error,
            "available_actions": ["fix_result", "choose_future_context", "choose_future_procedure"],
            "origin": {"session_key": session_key, "message_id": str(message_id),
                       "return_path": "/chat/" + quote(session_key, safe="")},
        }

    def inspect_procedure(self, *, context: RequestContext, pin: ArtifactPinV1) -> ProcedureInspectionV1:
        if (not context.authenticated or not context.user_id or not context.organization_id
                or not context.workspace_id):
            raise PermissionError("FORBIDDEN_LEARNING_SCOPE")
        if pin.artifact_kind != SkillArtifactVersionV1.ARTIFACT_KIND:
            raise ValueError("EXACT_PIN_MISMATCH")
        version = self._store.get(
            organization_id=context.organization_id, workspace_id=context.workspace_id,
            artifact_ref=pin.artifact_ref, contract_type=SkillArtifactVersionV1,
        )
        if version is None:
            raise LookupError("LEARNING_OBJECT_NOT_FOUND")
        if version.owner_ref != context.user_id:
            raise PermissionError("FORBIDDEN_LEARNING_SCOPE")
        if version.pin() != pin:
            raise ValueError("EXACT_PIN_MISMATCH")
        return ProcedureInspectionV1(
            object_pin=version.pin(), skill_id=version.skill_id, version=version.version, files=version.files,
            organization_id=version.organization_id, workspace_id=version.workspace_id,
            leading_answer="Exact procedure version retained. Availability for future sessions is not confirmed.",
            decisive_uncertainty=(
                "Qualification, activation, current runtime availability, applicability and recorded use are unknown."
            ),
        )
