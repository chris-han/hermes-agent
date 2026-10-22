"""Expense task runner over workspace uploads and existing extraction owners."""
import json
from pathlib import Path
from services.expense_reimbursement_service import ExpenseReimbursementService
from services.governed_artifact_service import _publish_immutable_bytes
from storage.workspace_artifact_blob_store import WorkspaceArtifactBlobStore


class ExpenseWorkbenchService:
    def __init__(self, *, ctx, bound, artifacts, policy_owner):
        self.ctx, self.bound, self.artifacts = ctx, bound, artifacts
        self.business = ExpenseReimbursementService(bound, policy_owner=policy_owner, input_actor_ref='user:' + ctx.user_id)

    def context(self):
        return dict(session_ref=self.bound.session_id, result=self.business.current())

    def evaluate(self, *, artifact_pin, submitted_on, applicant_fields, predecessor_result=None):
        artifact = self.artifacts.resolve(ctx=self.ctx, artifact_pin=artifact_pin)
        if Path(artifact.original_name).suffix.lower() != '.pdf':
            raise ValueError('EXPENSE_PDF_REQUIRED')
        blobs = WorkspaceArtifactBlobStore(ctx=self.ctx, max_bytes=self.artifacts._max_bytes)
        with blobs.exact_get(byte_owner_ref=artifact.byte_owner_ref,
                            expected_hash=artifact.content_hash, expected_size=artifact.byte_size) as stream:
            raw = stream.read()
        relative = 'uploads/expense-' + artifact.content_hash.split(':')[1] + '.pdf'
        target = self.business._path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        _publish_immutable_bytes(target, raw)
        from plugins.document_extraction.service import extract_live
        from plugins.financial_documents.service import extract_live as extract_financial
        extracted = extract_live(dict(document_ref=relative), trusted_session=self.bound)
        financial = extract_financial(dict(canonical_document_artifact_ref=extracted['artifacts']['normalized_document_artifact_ref'],
                                          jurisdiction_hint='CN'), trusted_session=self.bound)
        return self.business.evaluate(financial_ref=financial['artifacts']['financial_extraction_artifact_ref'],
                                      submitted_on=submitted_on, applicant_fields=applicant_fields,
                                      predecessor_result=predecessor_result,
                                      source_material_pin=artifact_pin.model_dump(mode='json'))

    def retain_inference(self, *, result_ref, result_hash):
        """Freeze an actual completed Chat turn that read this exact result."""
        from agents.webapi.session_routes import _workspace_session_db, _db_message_to_workspace_message
        from plugins.document_extraction.storage import sha256_payload, utc_now_iso
        result = self.business.replay(result_ref, result_hash)
        db = _workspace_session_db(self.ctx)
        try:
            session = db.get_session(self.bound.session_id)
            messages = []
            for item in db.get_messages(self.bound.session_id):
                message = _db_message_to_workspace_message(item)
                # Retain observable input/output, never hidden reasoning.
                message.pop('reasoning', None)
                message.pop('reasoning_content', None)
                if item.get('api_content') is not None:
                    message['api_content'] = item['api_content']
                messages.append(message)
        finally:
            db.close()
        if session is None:
            raise ValueError('EXPENSE_CHAT_SESSION_MISSING')
        final_index = next((index for index in range(len(messages)-1, -1, -1)
            if messages[index].get('role') == 'assistant' and messages[index].get('finish_reason') == 'stop'), None)
        if final_index is None:
            raise ValueError('EXPENSE_COMPLETED_CHAT_REQUIRED')
        start = next((index for index in range(final_index, -1, -1)
                      if messages[index].get('role') == 'user'), None)
        if start is None:
            raise ValueError('EXPENSE_CHAT_INPUT_MISSING')
        turn = messages[start:final_index+1]
        calls = {call['id']: call for message in turn if message.get('role') == 'assistant'
                 for call in message.get('tool_calls', []) if isinstance(call, dict) and call.get('id')}
        matched = False
        for message in turn:
            if message.get('role') != 'tool' or message.get('tool_name') != 'evaluate_expense_reimbursement':
                continue
            try:
                read = json.loads(message['content'])
                call = calls.get(message.get('tool_call_id'), {})
                function = call.get('function', {})
                arguments = json.loads(function.get('arguments', '{}'))
                if not isinstance(arguments, dict) or not isinstance(read, dict):
                    continue
                function_name = function.get('name')
                if function_name == 'tool_call':
                    # Native deferred dispatch retains the wrapper call and the
                    # underlying tool name on its response. Normalize only the
                    # recorded single local call; never consult current config.
                    from tools.tool_search_validation import normalize_tool_call_entries
                    entries, error = normalize_tool_call_entries(arguments)
                    if error or len(entries) != 1:
                        continue
                    function_name, arguments = entries[0]['name'], entries[0]['arguments']
            except (ValueError, TypeError):
                continue
            if (function_name == 'evaluate_expense_reimbursement'
                and (arguments.get('action') == 'read_current' or (arguments.get('action') == 'replay'
                    and arguments.get('result_ref') == result_ref and arguments.get('result_hash') == result_hash))
                and read.get('result_ref') == result_ref and read.get('result_hash') == result_hash and read == result):
                matched = True
        if not matched:
            raise ValueError('EXPENSE_CHAT_EXACT_RESULT_BASIS_REQUIRED')
        final = messages[final_index]
        inference = dict(origin='ModelInference', label='模型推测', claim=final['content'],
            confirmation_status='unverified', confidence=None, policy_clause_ref=None,
            business_disposition_authority=False, evidence_sufficiency='actual_chat_inputs_retained',
            input_result_pin=dict(result_ref=result_ref, result_hash=result_hash),
            actual_input_messages=turn[:-1], generation_message=final,
            generation=dict(session_ref=self.bound.session_id, message_id=final['id'],
                session_model=session.get('model'), model_identity_basis='native_session_record',
                provider=None, token_usage=None, missing_provenance=['per_call_provider', 'per_call_token_usage']),
            missing_basis=['ACTUAL_EXPENSE_PURPOSE_NOT_ESTABLISHED_BY_INVOICE'],
            captured_at=utc_now_iso())
        inference['inference_hash'] = sha256_payload(inference)
        payload = {key: value for key, value in result.items() if key not in {'result_ref', 'result_hash'}}
        payload['model_inferences'] = [*result['model_inferences'], inference]
        payload['produced_at'] = utc_now_iso()
        payload['predecessor_result'] = dict(result_ref=result_ref, result_hash=result_hash)
        payload['pins'] = [*payload['pins'], self.business._pin(result_ref)]
        return self.business._publish('result', payload)
