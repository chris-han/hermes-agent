"""Session-scoped expense snapshots; policy execution remains in core rule owners.

The policy-unavailable path preserves useful invoice facts and drafts without
inventing requirements, approval, or an expense purpose from invoice text.
"""
from datetime import date, datetime, timezone
import json
from pathlib import Path

from plugins.document_extraction.storage import (
    BoundSession, sha256_bytes, sha256_payload, utc_now_iso,
)
from storage.atomic_json_artifact import publish_immutable_json


class ExpenseReimbursementService:
    def __init__(self, session: BoundSession, *, policy_owner=None, input_actor_ref=None):
        if policy_owner is not None:
            from services.expense_teaching_policy_service import ExpenseTeachingPolicyService
            if not isinstance(policy_owner, ExpenseTeachingPolicyService):
                raise TypeError('expense policy requires its native owner')
        self.session = session
        self.policy_owner = policy_owner
        self.input_actor_ref = input_actor_ref

    def _path(self, ref):
        if not isinstance(ref, str) or not ref or Path(ref).is_absolute():
            raise ValueError('EXPENSE_SESSION_REF_REQUIRED')
        path = (self.session.session_root / ref).resolve()
        if not path.is_relative_to(self.session.session_root.resolve()):
            raise ValueError('EXPENSE_CROSS_SESSION_REF')
        return path

    def _artifact(self, ref):
        value = json.loads(self._path(ref).read_text(encoding='utf-8'))
        supplied = value.get('artifact_hash')
        if supplied != sha256_payload({k: v for k, v in value.items() if k != 'artifact_hash'}):
            raise ValueError('EXPENSE_ARTIFACT_PIN_MISMATCH')
        return value

    def _pin(self, ref):
        return {'ref': ref, 'hash': sha256_bytes(self._path(ref).read_bytes())}

    def _publish(self, kind, payload):
        digest = sha256_payload(payload)
        ref = f'artifacts/expense_reimbursement/{kind}-{digest[7:]}.json'
        result = {**payload, f'{kind}_ref': ref, f'{kind}_hash': digest}
        publish_immutable_json(self._path(ref), result, conflict_error='EXPENSE_IMMUTABLE_CONFLICT')
        return result

    def evaluate(self, *, financial_ref, submitted_on, applicant_fields, predecessor_result=None, source_material_pin=None):
        if not isinstance(submitted_on, str) or date.fromisoformat(submitted_on).isoformat() != submitted_on:
            raise ValueError('EXPENSE_SUBMISSION_DATE_INVALID')
        if not isinstance(applicant_fields, dict) or set(applicant_fields) - {'name', 'department', 'purpose'}:
            raise ValueError('EXPENSE_APPLICANT_FIELDS_INVALID')
        if any(not isinstance(value, str) for value in applicant_fields.values()):
            raise ValueError('EXPENSE_APPLICANT_FIELDS_INVALID')
        financial = self._artifact(financial_ref)
        canonical_ref = financial['canonical_document_artifact_ref']
        canonical = self._artifact(canonical_ref)
        source_ref = canonical['source_artifact_ref']
        if sha256_bytes(self._path(source_ref).read_bytes()) != canonical['source_hash']:
            raise ValueError('EXPENSE_SOURCE_PIN_MISMATCH')
        if financial['document'].get('schema_version') != 'invoice_contract.v1':
            raise ValueError('EXPENSE_INVOICE_SCHEMA_INVALID')
        payload = {
            'schema_version': 'expense_reimbursement_result.v1',
            'session_id': self.session.session_id,
            'workspace_binding': sha256_payload(str(self.session.workspace_home.resolve())),
            'evaluation_status': 'insufficient_basis', 'disposition': None,
            'limitations': ['POLICY_NOT_PREPARED'], 'missing_fields': [],
            'invoice': financial['document'],
            'extraction_findings': financial.get('findings', []),
            'extraction_unresolved_fields': financial.get('unresolved_fields', []),
            'applicant_fields': {key: {'value': value, 'origin': 'user_provided' if self.input_actor_ref else 'provided_input_unconfirmed',
                                      'actor_ref': self.input_actor_ref, 'recorded_at': utc_now_iso()}
                                 for key, value in applicant_fields.items()},
            'submitted_on': submitted_on, 'business_timezone': 'Asia/Shanghai',
            'policy': None, 'model_inferences': [],
            'pins': [self._pin(ref) for ref in (financial_ref, canonical_ref, source_ref)],
            'produced_at': utc_now_iso(),
            'source_material_pin': source_material_pin,
        }
        payload['invoice_source'] = dict(
            source_identity_ref=f'session-upload:{self.session.session_id}:{canonical["source_hash"]}',
            source_hash=canonical['source_hash'], name='invoice.pdf',
            spans=[self._source_span(block['id'], block['text'], block['page'], block['bbox'])
                   for block in canonical['document'].get('blocks', []) if block.get('bbox') and block.get('text')])
        if self.policy_owner is not None:
            prepared = self.policy_owner.current()
            if prepared is not None:
                self._apply_policy(payload, prepared)
        if predecessor_result is not None:
            previous = self.replay(predecessor_result['result_ref'], predecessor_result['result_hash'])
            if self._pin(financial_ref) not in previous['pins']:
                raise ValueError('EXPENSE_PREDECESSOR_INVOICE_MISMATCH')
            payload['model_inferences'] = previous['model_inferences']
            payload['predecessor_result'] = predecessor_result
            payload['pins'].append(self._pin(predecessor_result['result_ref']))
            for field in payload['applicant_fields'].values():
                field['related_inference_hashes'] = [item['inference_hash'] for item in previous['model_inferences']]
        return self._publish('result', payload)

    @staticmethod
    def evaluate_policy(*, policy, plan, candidate, invoice_date, submitted_on, applicant_fields, target_evidence_refs):
        """Adapt observations only; all policy predicates/effects come from the plan."""
        from contracts.ontology_building import ArtifactPinV1
        from services.deterministic_rule_evaluator import DeterministicRuleEvaluator
        values = dict(invoice_date=invoice_date, submitted_on=submitted_on,
                      is_draft=False, **{key: applicant_fields.get(key, '').strip() for key in ('name', 'department', 'purpose')})
        facts = {ref: values[field] for ref, field in policy.field_bindings.items() if values[field] is not None}
        facts_hash = sha256_payload(facts)
        if plan.precedences:
            raise ValueError('EXPENSE_TEACHING_PRECEDENCE_NOT_QUALIFIED')
        evaluations = DeterministicRuleEvaluator().evaluate_rules_without_precedence(plan, facts)
        trace = dict(schema_version='expense_teaching_rule_trace.v1', teaching_candidate=True,
            plan_ref=plan.plan_ref, plan_hash=plan.plan_hash, observed_facts_hash=facts_hash,
            target_evidence_refs=sorted(set(target_evidence_refs)),
            rule_evaluations=[item.model_dump(mode='json') for item in evaluations],
            evaluated_at=utc_now_iso())
        missing = sorted({policy.field_bindings[p.fact_ref] for rule in plan.rules for p in rule.input_requirements or ()
            if p.operator == 'EXISTS' and p.fact_ref in policy.field_bindings and values[policy.field_bindings[p.fact_ref]] in ('', None)})
        outcomes = {item.rule_id: item.outcome for item in evaluations}
        effects = [(rule, effect) for rule in plan.rules for effect in rule.disposition_effects or ()
                   if effect.fact_ref in policy.disposition_effects]
        limitations = []
        invalid_date = False
        try:
            invalid_date = invoice_date is not None and date.fromisoformat(invoice_date) > date.fromisoformat(submitted_on)
        except (TypeError, ValueError):
            invalid_date = True
        if invalid_date:
            limitations.append('INVOICE_DATE_INVALID_OR_FUTURE')
        scope_frames = {frame.frame_id for frame in candidate.frames if frame.frame_kind == 'scope'}
        scope_unmatched = any(check.ir_node_ref in scope_frames and check.result != 'SATISFIED'
            for item in evaluations for check in item.predicate_results)
        if scope_unmatched:
            limitations.append('OUTSIDE_OR_UNKNOWN_TEACHING_SCOPE')
        unknown = scope_unmatched or not effects or any(outcomes[rule.rule_id] == 'UNKNOWN' for rule, _ in effects)
        matched = [policy.disposition_effects[effect.fact_ref] for rule, effect in effects
                   if outcomes[rule.rule_id] == 'MATCH' and effect.operator == 'EQ' and effect.expected_value is True]
        # NO_MATCH means this policy did not trigger its review obligation; it is
        # not authority to approve, submit or pay anything.
        disposition = None if unknown or invalid_date else (matched[0] if matched else 'allowed')
        if unknown:
            limitations.append('POLICY_INPUT_BASIS_INCOMPLETE')
        return dict(evaluation_status='insufficient_basis' if disposition is None else 'evaluated',
            disposition=disposition, missing_fields=missing, limitations=limitations,
            observed_facts=facts, rule_trace=trace)

    def _apply_policy(self, payload, prepared):
        from services.governed_artifact_service import _publish_immutable_bytes
        policy, plan = prepared['policy'], prepared['plan']
        assessed = self.evaluate_policy(policy=policy, plan=plan, candidate=prepared['candidate'], invoice_date=payload['invoice'].get('invoice_date'),
            submitted_on=payload['submitted_on'], applicant_fields={key: value['value'] for key, value in payload['applicant_fields'].items()},
            target_evidence_refs=[ref for item in payload['invoice']['evidence'] for ref in item['evidence_ids']])
        pdf_ref = 'artifacts/expense_reimbursement/policy-' + policy.source_hash[7:] + '.pdf'
        path = self._path(pdf_ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        _publish_immutable_bytes(path, prepared['source_bytes'])
        frozen = {key: value.model_dump(mode='json', by_alias=True) if hasattr(value, 'model_dump') else value for key, value in prepared.items()
                  if key not in {'source_bytes'}}
        snapshot = self._publish('policy_snapshot', frozen)
        payload.update(assessed, policy={**policy.model_dump(mode='json'), 'source_ref': pdf_ref,
            'snapshot_ref': snapshot['policy_snapshot_ref'], 'snapshot_hash': snapshot['policy_snapshot_hash']})
        payload['pins'].extend([self._pin(pdf_ref), self._pin(snapshot['policy_snapshot_ref'])])
        ir = prepared['canonical_source']
        blocks = {block['blockId']: block for block in ir['blocks']}
        spans = []
        for mapping in ir['sourceMap']['mappings']:
            for locator in mapping['sourceLocators']:
                metadata = locator.get('nativeMetadata') or {}
                for block_id in mapping['blockIds']:
                    block = blocks[block_id]
                    for span in metadata.get('text_spans', []):
                        start, end = span['start_offset'], span['end_offset']
                        spans.append(self._source_span(f'{block_id}:{start}:{end}', block['content'][start:end],
                            block['structure']['page'], span['bbox']))
        payload['policy_source'] = dict(source_identity_ref=policy.source_identity_ref,
            source_hash=policy.source_hash, name=prepared['source_identity'].original_name, spans=spans)
        candidate = prepared['candidate']
        frames = {frame.frame_id: frame for frame in candidate.frames}
        edges = []
        for frame in frames.values():
            for field, targets in frame.model_dump(mode='json').items():
                if field.endswith('_refs') and isinstance(targets, list):
                    for target in targets:
                        if target in frames:
                            edges.append(dict(edge_id=f'{frame.frame_id}:{field}:{target}',
                                source_node_id=frame.frame_id, target_node_id=target,
                                relation_type=field, evidence_refs=list(frame.evidence_refs)))
        payload['policy_graph'] = dict(snapshot_ref=candidate.candidate_set_ref,
            snapshot_hash=candidate.candidate_set_hash,
            nodes=[dict(node_id=frame.frame_id, node_kind=frame.frame_kind,
                        label=frame.display_name_zh or frame.display_name,
                        evidence_refs=list(frame.evidence_refs)) for frame in frames.values()], edges=edges)
        schema = prepared['schema']
        lineage_nodes = [
            (policy.source_identity_ref, 'source', prepared['source_identity'].original_name),
            (schema.schema_ref, 'schema', '政策概念结构 · 教学候选'),
            (candidate.candidate_set_ref, 'semantics', '来源语义 · 已复核教学候选'),
            (plan.plan_ref, 'compiled_rules', '编译规则 · 教学评估'),
        ]
        links = [(policy.source_identity_ref, candidate.candidate_set_ref, 'source_identity_ref'),
                 (candidate.normative_schema_ref, candidate.candidate_set_ref, 'normative_schema_ref'),
                 (plan.candidate_set_ref, plan.plan_ref, 'candidate_set_ref')]
        payload['policy_lineage'] = dict(snapshot_ref=policy.policy_ref, snapshot_hash=policy.policy_hash,
            nodes=[dict(node_id=ref, node_kind=kind, label=label, evidence_refs=[policy.source_identity_ref])
                   for ref, kind, label in lineage_nodes],
            edges=[dict(edge_id=f'{source}:{kind}:{target}', source_node_id=source, target_node_id=target,
                        relation_type=kind, evidence_refs=[policy.source_identity_ref]) for source, target, kind in links])

    @staticmethod
    def _source_span(ref, text, page, bbox):
        x, y, right, bottom = bbox
        return dict(evidence_ref=ref, quote=text, presentation_location=dict(kind='pdf_page_rect',
            page=page, x=x, y=y, width=right-x, height=bottom-y, unit='pdf-point',
            origin='top-left', precision='span', exactText=text))

    def replay(self, result_ref, result_hash):
        result = json.loads(self._path(result_ref).read_text(encoding='utf-8'))
        basis = {k: v for k, v in result.items() if k not in {'result_ref', 'result_hash'}}
        if (result.get('result_hash') != result_hash or sha256_payload(basis) != result_hash
            or result.get('result_ref') != result_ref):
            raise ValueError('EXPENSE_RESULT_PIN_MISMATCH')
        if (result.get('session_id') != self.session.session_id
            or result.get('workspace_binding') != sha256_payload(str(self.session.workspace_home.resolve()))):
            raise ValueError('EXPENSE_SESSION_MISMATCH')
        for pin in result['pins']:
            if self._pin(pin['ref']) != pin:
                raise ValueError('EXPENSE_DEPENDENCY_PIN_MISMATCH')
        return result

    def current(self):
        results = []
        for path in (self.session.session_root / 'artifacts/expense_reimbursement').glob('result-*.json'):
            raw = json.loads(path.read_text())
            results.append(self.replay(raw['result_ref'], raw['result_hash']))
        return max(results, key=lambda row: (row['produced_at'], row['result_hash'])) if results else None

    def save_draft(self, result_ref, result_hash):
        result = self.replay(result_ref, result_hash)
        return self._publish('draft', {
            'schema_version': 'expense_reimbursement_draft.v1',
            'status': 'draft_requires_review' if result['disposition'] == 'exception_review_required' else ('draft_incomplete' if result['missing_fields'] else 'draft_insufficient_basis' if result['evaluation_status'] != 'evaluated' else 'draft_ready_for_review'),
            'session_id': self.session.session_id,
            'result_ref': result_ref, 'result_hash': result_hash,
            'invoice': result['invoice'], 'applicant_fields': result['applicant_fields'],
            'model_inferences': result['model_inferences'],
            'disposition': result['disposition'], 'produced_at': utc_now_iso(),
        })
