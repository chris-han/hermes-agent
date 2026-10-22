"""Derived exact-revision notices with per-user read/defer presentation metadata."""
from __future__ import annotations

from eos.artifact_hashing import content_hash
from runtime_paths import workspace_hermes_home_path


class LearningReviewStateService:
    def __init__(self, *, store):
        self.store = store

    @staticmethod
    def _scope(context):
        expected = workspace_hermes_home_path(context.workspace_id).resolve()
        if (not context.authenticated or not context.user_id or not context.workspace_id
            or context.hermes_home.resolve() != expected or context.workspace_root.resolve() != expected):
            raise PermissionError('FORBIDDEN_LEARNING_SCOPE')
        return dict(organization_id=context.organization_id, workspace_id=context.workspace_id,
                    actor_ref=context.user_id)

    @staticmethod
    def identity(item):
        return 'learning-notice:' + content_hash({'kind': item.get('kind') or item.get('object_kind'),
            'ref': item.get('ref') or item.get('candidate_ref'),
            'hash': item.get('hash') or item.get('candidate_hash'),
            'admission': item.get('admission') or item.get('states', {}).get('admission'),
            'activation': item.get('activation') or item.get('states', {}).get('activation'),
            'publication': item.get('publication'), 'readiness': item.get('readiness')})

    def notices(self, context, objects):
        scope = self._scope(context)
        result = {}
        for item in objects:
            kind = item.get('kind') or item.get('object_kind')
            if kind not in ('pending', 'candidate', 'contribution'):
                continue
            ref = item.get('ref') or item.get('candidate_ref')
            digest = item.get('hash') or item.get('candidate_hash')
            if not ref or not digest:
                continue
            identity = self.identity(item)
            result[identity] = {'id': identity, 'kind': kind, 'ref': ref, 'hash': digest,
                'title': item.get('title') or item.get('skill_id') or 'Review a proposed change',
                'created_at': item.get('recorded_at'),
                **self.store.learning_review_state(**scope, notice_id=identity)}
        return list(result.values())

    def set_state(self, context, *, ref, expected_hash, operation, request_id, objects):
        scope = self._scope(context)
        if operation not in ('mark_notice_read', 'defer_candidate') or not request_id:
            raise ValueError('INVALID_REVIEW_PRESENTATION_COMMAND')
        item = next((item for item in objects if (item.get('ref') or item.get('candidate_ref')) == ref), None)
        if item is None:
            raise LookupError('LEARNING_OBJECT_NOT_FOUND')
        if (item.get('hash') or item.get('candidate_hash')) != expected_hash:
            raise ValueError('EXACT_HASH_MISMATCH')
        identity = self.identity(item)
        self.store.set_learning_review_state(**scope, notice_id=identity, operation=operation,
            request_id=request_id, request_hash=content_hash({'operation':operation,'notice_id':identity}))
        return {**item, 'review_state': self.store.learning_review_state(**scope, notice_id=identity)}
