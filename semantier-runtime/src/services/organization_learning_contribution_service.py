"""Sanitized organization snapshots and reviewer decisions; no distribution engine."""
from __future__ import annotations

import json
import re
from pathlib import PurePosixPath

from contracts.learning_workbench import SkillArtifactVersionV1
from eos.artifact_hashing import canonical_json, content_hash


def disclosure_files(files):
    """Conservative declared document-only disclosure. Exclusions cannot be consented away."""
    from security.sensitive_data.detection.deterministic import SensitiveDataDetector
    allowed, omitted = [], []
    for file in files:
        path = PurePosixPath(file.relative_path)
        reason = None
        if (path.suffix.lower() not in ('.md', '.txt', '.rst') or
            re.search(r'(?i)(private|secret|credential|oracle|gold|expected|reference[_-]?answer|protected|\.env)', str(path))):
            reason = 'EXCLUDED_DISCLOSURE_CLASS'
        try:
            text = file.content_bytes().decode('utf-8')
        except UnicodeDecodeError:
            text = ''
            reason = 'NON_TEXT_DISCLOSURE'
        if not reason and (SensitiveDataDetector().detect(text) or
            re.search(r'(?i)(authorization\s*:\s*\S+|bearer\s+[\w.-]+|https?://[^\s/]+:[^\s@]+@|[A-Za-z0-9_]*(?:password|passwd|api[_-]?key|access[_-]?key|token|secret|credential)[A-Za-z0-9_]*\s*[\"\']?\s*[:=]|evaluator[_-]?only|protected[_ -]reference)', text)):
            reason = 'SENSITIVE_CONTENT_EXCLUDED'
        if reason:
            # Do not disclose excluded filenames, hashes, or snippets to recipients.
            omitted.append(reason)
        else:
            allowed.append(file.model_dump(mode='json'))
    return allowed, omitted


class OrganizationLearningContributionService:
    def __init__(self, *, procedures, governance):
        self.procedures = procedures
        self.governance = governance

    @staticmethod
    def _membership(context, destination, *, reviewer=False):
        from agents.webapi.organization_service import get_user_membership_for_organization
        if not context.authenticated or not context.user_id or not destination:
            raise PermissionError('FORBIDDEN_ORGANIZATION_CONTRIBUTION')
        membership = get_user_membership_for_organization(user_id=context.user_id, organization_id=destination)
        if (not membership or membership.get('membership_status') != 'active' or
            (reviewer and membership.get('member_role') != 'owner')):
            raise PermissionError('FORBIDDEN_ORGANIZATION_CONTRIBUTION')
        return membership

    def _baseline(self, organization_id, skill_id):
        rows = [row for row in self.governance.list_candidates(category='organization_learning_contribution')
                if row['scoped_owner'] == organization_id and row['status'] == 'APPROVED'
                and json.loads(row['content_yaml'])['skill_id'] == skill_id]
        if not rows:
            return None
        row = max(rows, key=lambda row: (row['approved_at'] or '', row['candidate_id']))
        return {'ref': row['candidate_id'], 'hash': row['content_hash']}

    def preview(self, context, candidate_ref, candidate_hash, destination):
        self._membership(context, destination)
        candidate_projection = self.procedures.inspect(context, candidate_ref)
        if candidate_projection.get('candidate_kind') == 'CHANGE_SET':
            raise ValueError('PACKAGE_SET_CONTRIBUTION_REQUIRES_ENGINEERING')
        candidate, version, base = self.procedures._objects(context, candidate_ref, candidate_hash)
        files, omitted = disclosure_files(version.files)
        if not any(file['relative_path'] == 'SKILL.md' for file in files):
            raise ValueError('DISCLOSURE_SKILL_EXCLUDED')
        manifest = {'schema_version': 'organization_learning_disclosure.v1',
            'destination_organization_id': destination, 'skill_id': version.skill_id,
            'source_version': version.pin().model_dump(mode='json'),
            'source_candidate': candidate.pin().model_dump(mode='json'),
            'organization_baseline': self._baseline(destination, version.skill_id),
            'files': files, 'evidence_coverage': 'unavailable',
            'comparability': 'unknown_after_sanitization',
            'limitations': ['Sanitized snapshot; empirical efficacy and recipient runtime use are unknown.',
                'Admission does not activate or distribute this package.']}
        return {**manifest, 'disclosure_hash': content_hash(manifest),
                'omitted_paths': omitted}

    def submit(self, context, candidate_ref, candidate_hash, destination, disclosure_hash, consent, request_id):
        if consent is not True or not request_id:
            raise ValueError('EXPLICIT_DISCLOSURE_CONSENT_REQUIRED')
        self._membership(context, destination)
        command_hash = content_hash({'candidate_ref':candidate_ref,'candidate_hash':candidate_hash,
            'destination':destination,'disclosure_hash':disclosure_hash,'consent':consent})
        prior = self.governance.organization_contribution_request(organization_id=destination,
            actor=context.user_id, request_id=request_id)
        if prior:
            if json.loads(prior['content_yaml'])['command_hash'] != command_hash:
                raise ValueError('REQUEST_CONFLICT')
            return self._project(context, prior)
        preview = self.preview(context, candidate_ref, candidate_hash, destination)
        if preview['disclosure_hash'] != disclosure_hash:
            raise ValueError('STALE_DISCLOSURE')
        # Immutable new-scope body; personal admission and active pins are untouched.
        body = {key: value for key, value in preview.items() if key not in ('omitted_paths', 'disclosure_hash')}
        body.update(submitter=context.user_id, request_id=request_id, disclosure_hash=disclosure_hash,
                    command_hash=command_hash)
        row = self.governance.publish_organization_contribution(body=body,
            verify_authority=lambda: self._membership(context, destination))
        return self._project(context, row)

    def _row(self, context, ref, expected_hash=None):
        row = self.governance.get_by_id(ref)
        if not row or row['category'] != 'organization_learning_contribution':
            raise LookupError('ORGANIZATION_CONTRIBUTION_NOT_FOUND')
        self._membership(context, row['scoped_owner'])
        if context.organization_id != row['scoped_owner']:
            raise PermissionError('FORBIDDEN_ORGANIZATION_CONTRIBUTION')
        if content_hash(json.loads(row['content_yaml'])) != row['content_hash']:
            raise ValueError('CONTRIBUTION_INTEGRITY_FAILURE')
        if expected_hash is not None and expected_hash != row['content_hash']:
            raise ValueError('EXACT_HASH_MISMATCH')
        return row

    def inspect(self, context, ref):
        return self._project(context, self._row(context, ref))

    def inventory(self, context):
        self._membership(context, context.organization_id)
        return [self._project(context, row) for row in self.governance.list_candidates(category='organization_learning_contribution')
                if row['scoped_owner'] == context.organization_id]

    def decide(self, context, ref, expected_hash, approve):
        row = self._row(context, ref, expected_hash)
        self._membership(context, row['scoped_owner'], reviewer=True)
        body = json.loads(row['content_yaml'])
        result = self.governance.decide_organization_contribution(candidate_id=ref,
            expected_hash=expected_hash, approved=approve, actor=context.user_id,
            expected_baseline=body['organization_baseline'],
            verify_authority=lambda: self._membership(context, row['scoped_owner'], reviewer=True))
        return self._project(context, result)

    def _project(self, context, row):
        body = json.loads(row['content_yaml'])
        if content_hash(body) != row['content_hash']:
            raise ValueError('CONTRIBUTION_INTEGRITY_FAILURE')
        membership = self._membership(context, row['scoped_owner'])
        actions = ['approve_organization_contribution', 'reject_organization_contribution'] if row['status'] == 'PENDING' and membership.get('member_role') == 'owner' else []
        return {'kind': 'contribution', 'object_kind': 'contribution', 'ref': row['candidate_id'],
            'hash': row['content_hash'], 'title': body['skill_id'], 'component_ids': ['C6'],
            'leading_answer': 'Organization proposal '+row['status'].lower()+'. Runtime distribution is unavailable.',
            'decisive_uncertainty': 'Sanitization changes the package; empirical comparability remains unknown.',
            'destination_organization_id': row['scoped_owner'], 'effect_scope': 'organization proposal only',
            'files': body['files'], 'content': next((__import__('base64').b64decode(f['content_base64']).decode('utf-8') for f in body['files'] if f['relative_path'] == 'SKILL.md'), ''),
            'source_version': body['source_version'], 'organization_baseline': body['organization_baseline'],
            'states': {'admission': row['status'].lower(), 'activation': 'not_activated',
                'runtime_availability': 'unsupported_distribution', 'qualification': 'unknown',
                'recorded_use': 'unknown', 'causal_attribution': 'unknown'},
            'available_actions': actions, 'evidence': [], 'counter_evidence': body['limitations'],
            'history': self.governance.list_candidate_events(row['candidate_id'])}
