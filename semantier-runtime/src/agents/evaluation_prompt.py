"""Internal, explicit EOS-bound DEVELOPMENT prompt-section loading.

No transport exposes this binding. EOS records are evaluation context, never
identity/organization authority or publication/admission credentials. Scope is
checked against the already governed execution boundary, not resolved here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eos.evaluation_control_plane import EvaluationSubjectManifestV2
from prompts.loader import load_repo_prompt
from storage.evaluation_control_plane_store import EvaluationControlPlaneStore


@dataclass(frozen=True)
class EvaluationPromptBinding:
    store: EvaluationControlPlaneStore
    run_ref: str

    def resolve(self, boundary: Any, *, execution_profile: dict[str, Any]) -> dict[str, Any]:
        run = self.store.get_run(self.run_ref)
        if run['completed_at'] is not None or run['validity'] != 'VALID':
            raise PermissionError('evaluation run is closed or invalid')
        environment = self.store.get_environment_manifest(run['environment_manifest_ref'])
        subject = self.store.get_subject_manifest(run['subject_manifest_ref'])
        EvaluationSubjectManifestV2.from_record(subject)
        if (run['subject_manifest_ref'] != subject['content_hash'] or
                run['environment_manifest_ref'] != environment['environment_manifest_ref'] or
                subject['subject_type'] != 'harness' or
                subject['environment_manifest_ref'] != environment['environment_manifest_ref'] or
                subject['environment_fingerprint'] != environment['environment_fingerprint'] or
                subject['runtime_commit'] != environment['runtime_commit']):
            raise ValueError('evaluation subject/environment mismatch')
        attrs = environment['attributes']
        if attrs.get('execution_purpose') != 'DEVELOPMENT_EVALUATION':
            raise PermissionError('evaluation-only scope required')
        if any(not attrs.get(key) or attrs[key] != getattr(boundary, field, None)
               for key, field in [('workspace_id', 'workspace_id'), ('session_id', 'session_id'),
                                  ('actor_user_id', 'user_id')]):
            raise PermissionError('evaluation prompt scope mismatch')
        if (execution_profile.get('model') != 'qwen3.5-plus' or
                attrs.get('execution_profile') != execution_profile):
            raise ValueError('evaluation execution profile mismatch')
        treatment = attrs.get('prompt_treatment', {})
        if (set(treatment) != {'slot', 'asset', 'sha256'} or
                treatment['slot'] != 'legal_classification' or treatment['asset'] not in {
                    'evaluation/ace_baseline.md', 'evaluation/ace_candidate.md'}):
            raise ValueError('unsupported evaluation prompt section')
        asset = load_repo_prompt(treatment['asset'], allowed_namespace='evaluation')
        if asset.sha256 != treatment['sha256']:
            raise ValueError('evaluation prompt asset hash mismatch')
        return {'text': asset.text, 'receipt': {
            'run_ref': self.run_ref, 'environment_manifest_ref': environment['environment_manifest_ref'],
            'subject_manifest_ref': subject['content_hash'], 'slot': treatment['slot'],
            'asset': asset.relative_name, 'asset_sha256': asset.sha256,
            'delivery_boundary': 'SemantierAgent.AIAgent.run_conversation.system_message',
            'provider_consumption': 'UNVERIFIED',
        }}
