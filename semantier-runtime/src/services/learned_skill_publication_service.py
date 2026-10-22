"""Scoped learned-procedure successors and exact runtime materialization.

OntologyBuildStore retains immutable bytes and assessment. GovernanceStore owns
admission/activation. Hermes resolves the runtime projection, never authority.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

from agents.auth_session import RequestContext
from contracts.change_candidate_v2 import AtomicChangeProfileV1, ChangeCandidateV2, ChangeSetProfileV1
from contracts.governed_delta import ArtifactVersionPinV1, DeltaLens, DeltaLensResultV1, DeltaLensStatus, DeltaTargetType
from contracts.learning_workbench import SkillArtifactVersionV1, SkillFileV1
from contracts.learned_skill_publication import LearnedSkillAssessmentV1, LearnedSkillPublicationReceiptV1, LearnedSkillProposalV1
from contracts.ontology_building import ArtifactPinV1
from eos.artifact_hashing import canonical_json, content_hash
from runtime_paths import bind_workspace_env, workspace_hermes_home_path
from services.candidate_admission_classifier import CandidateAdmissionClassifier
from services.governed_delta_service import GovernedDeltaService


def _now():
    return datetime.now(timezone.utc)


def _file(path, data):
    return SkillFileV1(relative_path=path, content_base64=base64.b64encode(data).decode('ascii'), content_hash='sha256:' + hashlib.sha256(data).hexdigest())


class LearnedSkillPublicationService:
    def __init__(self, *, store, governance):
        self._store = store
        self._governance = governance

    def _set_owner(self):
        from services.learned_skill_package_set import LearnedSkillPackageSet
        return LearnedSkillPackageSet(self)

    def _is_set(self, context, ref, hash=None):
        return isinstance(self._load(context, ref, ChangeCandidateV2, hash).profile, ChangeSetProfileV1)

    def _scope(self, context):
        if (not isinstance(context, RequestContext) or not context.authenticated or not context.user_id
                or not context.organization_id or not context.workspace_id):
            raise PermissionError('FORBIDDEN_LEARNING_SCOPE')
        expected = workspace_hermes_home_path(context.workspace_id).resolve()
        if context.hermes_home.resolve() != expected or context.workspace_root.resolve() != expected:
            raise PermissionError('FORBIDDEN_RUNTIME_HOME')
        return dict(organization_id=context.organization_id, workspace_id=context.workspace_id)

    @contextmanager
    def _home(self, context):
        self._scope(context)
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        from tools.skill_review_capture import managed_workspace, stage_actor, bind_generation_verifier
        token = set_hermes_home_override(context.hermes_home)
        managed_token = managed_workspace.set(True)
        actor_token = stage_actor.set({"user_id":context.user_id,"workspace_id":context.workspace_id})
        try:
            with bind_generation_verifier(lambda packages: self._verify_runtime_generation_fresh(context, packages), (context.organization_id, context.workspace_id, context.user_id)):
                yield
        finally:
            stage_actor.reset(actor_token)
            managed_workspace.reset(managed_token)
            reset_hermes_home_override(token)

    def _verify_runtime_generation_fresh(self, context, packages):
        # Tool executors propagate ContextVars across threads. Each invocation
        # opens its own canonical read handles rather than sharing SQLite owners.
        from storage.ontology_build_store import OntologyBuildStore
        from storage.governance_store import GovernanceStore
        artifact_path = self._store._path
        governance_path = self._governance._db_path
        from storage.hosted_postgres import enabled
        if not enabled() and (not Path(artifact_path).is_file() or not Path(governance_path).is_file()):
            raise ValueError('MANAGED_CANONICAL_VERIFIER_UNAVAILABLE')
        artifacts = OntologyBuildStore(artifact_path)
        governance = GovernanceStore(governance_path)
        try:
            return LearnedSkillPublicationService(store=artifacts, governance=governance)._verify_runtime_generation(context, packages)
        finally:
            artifacts.close()
            governance.close()

    def _verify_runtime_generation(self, context, packages):
        """Disk is a projection. Only canonical published versions authorize load."""
        from contracts.learned_skill_publication import LearnedSkillSetReceiptV1
        self._scope(context)
        versions = {v.artifact_ref:v for v in self._owned(context, SkillArtifactVersionV1, 'skill_artifact_version.v1')}
        receipts = self._owned(context, LearnedSkillPublicationReceiptV1, 'learned_skill_publication_receipt.v1') + self._owned(context, LearnedSkillSetReceiptV1, 'learned_skill_set_receipt.v1')
        latest = {}
        for receipt in receipts:
            candidate = self._load(context, receipt.candidate.artifact_ref, ChangeCandidateV2, receipt.candidate.artifact_hash)
            row = self._row(context, candidate)
            if row is None:
                continue
            events = self._governance.list_candidate_events(row['candidate_id'])
            publication = next((e for e in reversed(events) if e['event_type'] == 'PROCEDURE_PUBLISHED' and e['event']['payload'].get('receipt') == receipt.pin().model_dump(mode='json')), None)
            if publication is None:
                continue  # retained fact alone is never publication authority
            if not any(e['candidate_event_id'] == receipt.activation_ref and e['event_type'] == 'ACTIVATED' for e in events):
                raise ValueError('MANAGED_CANONICAL_PUBLICATION_INVALID')
            pins = receipt.versions if isinstance(receipt, LearnedSkillSetReceiptV1) else (receipt.version,)
            payload = json.loads(row['content_yaml'])
            approved = list(payload['member_versions'].values()) if payload.get('member_versions') else [payload['version']]
            key = lambda pin: (pin['artifact_ref'], pin['artifact_hash'])
            exact_pins = [p.model_dump(mode='json') for p in pins]
            if sorted(exact_pins, key=key) != sorted(approved, key=key) or sorted(exact_pins, key=key) != sorted(publication['event']['payload']['versions'], key=key):
                raise ValueError('MANAGED_CANONICAL_PUBLICATION_INVALID')
            deltas = [m.delta for m in candidate.profile.members] if isinstance(candidate.profile, ChangeSetProfileV1) else [candidate.profile.delta]
            if sorted((d.successor_candidate.version_id, d.successor_candidate.content_hash) for d in deltas) != sorted((p.artifact_ref, p.artifact_hash) for p in pins):
                raise ValueError('MANAGED_CANONICAL_PUBLICATION_INVALID')
            for pin in pins:
                version = versions.get(pin.artifact_ref)
                if version is None or version.pin() != pin:
                    raise ValueError('MANAGED_CANONICAL_PUBLICATION_INVALID')
                order = (publication['created_at'], publication['candidate_event_id'])
                if version.skill_id not in latest or order > latest[version.skill_id][0]:
                    latest[version.skill_id] = (order, version)
        # Omitting a managed/revoked name must not restore an unmanaged duplicate.
        if not set(latest) <= set(packages):
            raise ValueError('MANAGED_CANONICAL_GENERATION_MISMATCH')
        for name in latest:
            entry = packages[name]
            version = latest[name][1]
            expected = {'version_ref':version.artifact_ref, 'version_hash':version.artifact_hash, 'state':version.package_state, 'relative_path':name+'/'+version.artifact_hash[7:], 'files':{f.relative_path:f.content_hash for f in version.files}}
            if entry != expected:
                raise ValueError('MANAGED_CANONICAL_GENERATION_MISMATCH')
        return frozenset(latest)

    def _load(self, context, ref, model, expected_hash=None):
        obj = self._store.get(**self._scope(context), artifact_ref=ref, contract_type=model)
        if obj is None:
            raise LookupError('LEARNING_OBJECT_NOT_FOUND')
        owner = getattr(obj, 'owner_ref', None) or getattr(obj.profile, 'owner_ref', None)
        if owner != context.user_id:
            raise PermissionError('FORBIDDEN_LEARNING_SCOPE')
        if expected_hash is not None and obj.artifact_hash != expected_hash:
            raise ValueError('EXACT_HASH_MISMATCH')
        return obj

    def _name(self, context, skill_id):
        return f'personal:{context.organization_id}:{context.workspace_id}:{context.user_id}:{skill_id}'

    def _files(self, directory):
        directory = Path(directory)
        result = []
        for path in sorted(directory.rglob('*')):
            if path.is_symlink():
                raise ValueError('UNMANAGED_SKILL_SYMLINK')
            if path.is_file() and path.relative_to(directory).as_posix() != '.semantier-publication.json':
                result.append(_file(path.relative_to(directory).as_posix(), path.read_bytes()))
        return tuple(result)

    def _resolve(self, context, skill_id, intended=None):
        # Raw discovery is deliberately separate from skill_view: even its
        # preprocess=False path can install declared pm dependencies.
        from tools.skills_tool import _skill_search_dirs, _locate_skill
        from tools.skill_review_capture import name_check
        name_check(skill_id)
        with self._home(context):
            from hermes_cli.config import load_config
            trusted = load_config().get('skills', {}).get('trusted_project_dirs', [])
            if str(context.hermes_home) in trusted and (context.hermes_home / '.agents' / 'skills' / skill_id / 'SKILL.md').exists():
                raise ValueError('RUNTIME_SKILL_SHADOWED')
            roots, _ = _skill_search_dirs()
            error, directory, skill_md = _locate_skill(skill_id, None, roots)
            if error is not None or directory is None:
                raise ValueError('RUNTIME_SKILL_UNAVAILABLE: ' + str(error))
            if intended is not None and skill_md.resolve() != (Path(intended) / 'SKILL.md').resolve():
                from tools.skill_review_capture import resolve_managed
                managed = resolve_managed(skill_id)
                if managed is None or managed.resolve() != directory.resolve():
                    raise ValueError('RUNTIME_SKILL_SHADOWED')
            self._check_content(context, skill_id, skill_md.read_text(encoding='utf-8-sig'))
            return directory, {'skill_dir': str(directory), 'readiness_status': 'available'}

    def _owned(self, context, model, schema):
        result = []
        for pin in self._store.list_artifact_pins(**self._scope(context), schema_version=schema):
            try:
                result.append(self._load(context, pin.artifact_ref, model))
            except PermissionError:
                continue
        return result

    def list_candidates(self, context):
        return [self._projection(context, candidate) for candidate in self._owned(context, ChangeCandidateV2, 'semantier.change_candidate.v2') if isinstance(candidate.profile, (AtomicChangeProfileV1, ChangeSetProfileV1)) and candidate.profile.authority_domain_ref == 'personal-procedure']

    def inventory(self, context):
        self._scope(context)
        from tools.skills_tool import _find_all_skills
        with self._home(context):
            skills = _find_all_skills()
        for skill in skills:
            try:
                self.capture_baseline(context, skill['name'])
            except (ValueError, PermissionError):
                continue
        return [self.inspect(context, version.artifact_ref) for version in self._owned(context, SkillArtifactVersionV1, 'skill_artifact_version.v1')] + self.list_candidates(context)

    def capture_baseline(self, context, skill_id):
        self._scope(context)
        intended = context.hermes_home / 'skills' / skill_id
        directory, _ = self._resolve(context, skill_id, intended)
        if context.hermes_home.resolve() not in directory.resolve().parents:
            raise PermissionError('FORBIDDEN_RUNTIME_SKILL_PATH')
        files = self._files(directory)
        versions = self._owned(context, SkillArtifactVersionV1, 'skill_artifact_version.v1')
        active = self._active(context, skill_id)
        if active:
            payload = json.loads(active['content_yaml'])
            expected = payload.get('member_versions', {}).get(skill_id, payload.get('version'))
            exact = next((v for v in versions if v.pin().model_dump(mode='json') == expected and v.files == files), None)
            if exact:
                return exact
        for version in versions:
            if version.skill_id == skill_id and version.files == files:
                return version
        if any(v.skill_id == skill_id for v in versions):
            raise ValueError('STALE_RUNTIME_BASELINE')
        version = SkillArtifactVersionV1.create(**self._scope(context), owner_ref=context.user_id, skill_id=skill_id, version=1, provenance='legacy', files=files, recorded_at=_now())
        self._store.put(version)
        return version

    def _check_content(self, context, skill_id, content):
        from tools.skills_tool import _safe_frontmatter, skill_matches_platform, _is_skill_disabled, _get_required_environment_variables, _is_env_var_persisted, load_env
        with self._home(context):
            metadata = _safe_frontmatter(content=content)
            if metadata.get('name', skill_id) != skill_id:
                raise ValueError('SKILL_IDENTITY_MISMATCH')
            if not skill_matches_platform(metadata) or _is_skill_disabled(skill_id):
                raise ValueError('RUNTIME_SKILL_UNAVAILABLE')
            requirements = _get_required_environment_variables(metadata)
            if requirements:
                if (context.hermes_home / '.env').is_symlink():
                    raise ValueError('RUNTIME_SKILL_DEPENDENCIES_UNAVAILABLE')
                snapshot = load_env()
                if any(not item.get('optional') and not _is_env_var_persisted(item['name'], snapshot) for item in requirements):
                    raise ValueError('RUNTIME_SKILL_DEPENDENCIES_UNAVAILABLE')
            # Readiness here only inspects confined existing state. It never
            # captures secrets or registers execution credentials/passthrough.
            for credential in metadata.get('required_credential_files', []):
                path = Path(credential) if isinstance(credential, str) else None
                if path is None or path.is_absolute() or '..' in path.parts:
                    raise ValueError('RUNTIME_SKILL_DEPENDENCIES_UNAVAILABLE')
                candidate = context.hermes_home / path
                if candidate.is_symlink() or context.hermes_home.resolve() not in candidate.resolve().parents or not candidate.is_file():
                    raise ValueError('RUNTIME_SKILL_DEPENDENCIES_UNAVAILABLE')

    def _version_pin(self, version):
        return ArtifactVersionPinV1(artifact_type='skill_artifact_version', artifact_id=version.skill_id, version_id=version.artifact_ref, content_hash=version.artifact_hash, owner_ref=version.owner_ref, organization_id=version.organization_id, workspace_id=version.workspace_id, parent_version_ref=version.predecessor.artifact_ref if version.predecessor else None)

    def propose(self, context, baseline_ref, baseline_hash, new_content, request_id, origin=None, *, package_files=None):
        base = self._load(context, baseline_ref, SkillArtifactVersionV1, baseline_hash)
        if not isinstance(new_content, str) or (not new_content.strip() and package_files is None) or not request_id:
            raise ValueError('INVALID_PROCEDURE_CHANGE')
        if origin is not None:
            if not isinstance(origin, dict):
                raise ValueError('INVALID_CORRECTION_ORIGIN')
            # Source pins are validated against retained artifacts in this
            # authenticated scope; return locators remain navigation metadata.
            for source in origin.get('evidence_refs', ()):
                pin = ArtifactPinV1.model_validate(source)
                raw = self._store.get_raw(**self._scope(context), artifact_ref=pin.artifact_ref)
                if raw is None or raw['artifact_hash'] != pin.artifact_hash:
                    raise ValueError('INVALID_CORRECTION_ORIGIN')
        request_hash = content_hash({'base': base.pin().model_dump(mode='json'), 'content': new_content, 'origin': origin, 'package_files': [f.model_dump(mode='json') for f in package_files] if package_files is not None else None})
        recovered = self._store.get_operation_result(**self._scope(context), actor_ref=context.user_id, operation='propose_learned_skill', request_id=request_id, request_hash=request_hash, artifact_kind='change_candidate')
        if recovered:
            return self._projection(context, self._load(context, recovered.artifact_ref, ChangeCandidateV2, recovered.artifact_hash))
        self._assert_base(context, base)
        if new_content:
            self._check_content(context, base.skill_id, new_content)
        files = tuple(_file('SKILL.md', new_content.encode('utf-8')) if f.relative_path == 'SKILL.md' else f for f in base.files) if package_files is None else tuple(package_files)
        version = SkillArtifactVersionV1.create(**self._scope(context), owner_ref=context.user_id, skill_id=base.skill_id, version=base.version + 1, provenance='managed', predecessor=base.pin(), files=files, recorded_at=_now())
        envelope = GovernedDeltaService.build_envelope(delta_id='skill-successor:' + version.artifact_hash[7:], target_type=DeltaTargetType.WORKFLOW, predecessor=self._version_pin(base), successor_candidate=self._version_pin(version), lens_results={lens: DeltaLensResultV1(lens=lens, status=DeltaLensStatus.UNAVAILABLE, basis_refs=(base.artifact_ref, version.artifact_ref) if base.artifact_ref < version.artifact_ref else (version.artifact_ref, base.artifact_ref)) for lens in DeltaLens})
        candidate = ChangeCandidateV2.create(**self._scope(context), revision=0, candidate_kind='ATOMIC_CHANGE', profile=AtomicChangeProfileV1(owner_ref=context.user_id, authority_domain_ref='personal-procedure', rationale=new_content or 'Remove this personal procedure.', basis_refs=(base.pin(),), delta=envelope), recorded_at=_now())
        binding = LearnedSkillProposalV1.create(**self._scope(context), owner_ref=context.user_id, candidate=candidate.pin(), version=version.pin(), predecessor=base.pin(), request_id=request_id, request_hash=request_hash, origin=origin, recorded_at=_now())
        winner = self._store.put_many_with_operation_result(artifacts=(version, candidate, binding), **self._scope(context), actor_ref=context.user_id, operation='propose_learned_skill', request_id=request_id, request_hash=request_hash, result_pin=candidate.pin())
        return self._projection(context, self._load(context, winner.artifact_ref, ChangeCandidateV2, winner.artifact_hash))

    def _objects(self, context, candidate_ref, candidate_hash):
        candidate = self._load(context, candidate_ref, ChangeCandidateV2, candidate_hash)
        if not isinstance(candidate.profile, AtomicChangeProfileV1) or candidate.profile.authority_domain_ref != 'personal-procedure':
            raise ValueError('INVALID_PROCEDURE_CANDIDATE')
        pin = candidate.profile.delta.successor_candidate
        version = self._load(context, pin.version_id, SkillArtifactVersionV1, pin.content_hash)
        base = self._load(context, version.predecessor.artifact_ref, SkillArtifactVersionV1, version.predecessor.artifact_hash)
        if candidate.profile.delta.predecessor != self._version_pin(base) or version.skill_id != base.skill_id or candidate.profile.rationale != (self._content(version) or 'Remove this personal procedure.'):
            raise ValueError('EXACT_CANDIDATE_MISMATCH')
        return candidate, version, base

    def _active(self, context, skill_id):
        candidates = []
        for row in self._governance.list_candidates(category='learned_procedure', status='ACTIVE'):
            if row['scoped_owner'] != context.user_id:
                continue
            payload = json.loads(row['content_yaml'])
            same_scope = row['name'].startswith(self._name(context, ''))
            if row['name'] == self._name(context, skill_id) or (same_scope and skill_id in payload.get('member_versions', {})):
                events = self._governance.list_candidate_events(row['candidate_id'])
                activation = next((e for e in reversed(events) if e['event_type'] == 'ACTIVATED'), {})
                candidates.append((activation.get('created_at', ''), row))
        return max(candidates, key=lambda pair: pair[0])[1] if candidates else None

    def _assert_base(self, context, base):
        active = self._active(context, base.skill_id)
        if active:
            payload = json.loads(active['content_yaml'])
            actual = payload.get('member_versions', {}).get(base.skill_id, payload['version'])
            if actual != base.pin().model_dump(mode='json'):
                raise ValueError('STALE_PREDECESSOR')
            return
        if base.provenance != 'legacy':
            raise ValueError('STALE_PREDECESSOR')
        if base.package_state == 'absent':
            if (context.hermes_home / 'skills' / base.skill_id).exists():
                raise ValueError('STALE_PREDECESSOR')
            return
        directory, _ = self._resolve(context, base.skill_id, context.hermes_home / 'skills' / base.skill_id)
        if self._files(directory) != base.files:
            raise ValueError('STALE_PREDECESSOR')

    def assess(self, context, candidate_ref, candidate_hash):
        if self._is_set(context, candidate_ref, candidate_hash):
            return self._set_owner().assess(context, candidate_ref, candidate_hash)
        candidate, version, base = self._objects(context, candidate_ref, candidate_hash)
        self._assert_base(context, base)
        if version.files:
            self._check_content(context, version.skill_id, self._content(version))
        classification = self._classify_package(candidate, version, base)
        # Routing evidence establishes no admission. Exact manual scoped correction
        # may proceed if procedural-safe; unsupported semantics fail closed.
        eligible = classification.disposition == 'PROCEDURAL_SAFE_TO_PROJECT'
        assessment = LearnedSkillAssessmentV1.create(**self._scope(context), owner_ref=context.user_id, candidate=candidate.pin(), version=version.pin(), predecessor=base.pin(), classification=classification, eligible=eligible, counterevidence=('User-directed scoped correction; empirical generalization and recorded use are unknown.', 'Classification routes the proposal and grants no authority.') + (() if eligible else (classification.reason_code,)), recorded_at=_now())
        self._store.put(assessment)
        return assessment.model_dump(mode='json')

    def _row(self, context, candidate):
        return next((row for row in self._governance.list_candidates(category='learned_procedure') if json.loads(row['content_yaml']).get('candidate', {}).get('artifact_ref') == candidate.artifact_ref and row['scoped_owner'] == context.user_id), None)

    def approve(self, context, candidate_ref, candidate_hash, assessment_ref, assessment_hash):
        if self._is_set(context, candidate_ref, candidate_hash):
            return self._set_owner().approve(context, candidate_ref, candidate_hash, assessment_ref, assessment_hash)
        candidate, version, base = self._objects(context, candidate_ref, candidate_hash)
        assessment = self._load(context, assessment_ref, LearnedSkillAssessmentV1, assessment_hash)
        if (assessment.candidate, assessment.version, assessment.predecessor) != (candidate.pin(), version.pin(), base.pin()):
            raise ValueError('STALE_ASSESSMENT')
        if not assessment.eligible:
            raise ValueError('ADMISSION_ROUTED_OUT')
        self._assert_base(context, base)
        if version.files:
            self._check_content(context, version.skill_id, self._content(version))
        if self._classify_package(candidate, version, base) != assessment.classification:
            raise ValueError('STALE_ASSESSMENT')
        payload = {'candidate': candidate.pin().model_dump(mode='json'), 'version': version.pin().model_dump(mode='json'), 'assessment': assessment.pin().model_dump(mode='json')}
        row = self._governance.propose(name=self._name(context, version.skill_id), category='learned_procedure', content_yaml=canonical_json(payload), content_hash=content_hash(payload), proposed_by=context.user_id, scoped_owner=context.user_id, rollback_window_hours=24, rollback_strategy='Publish an exact retained predecessor as a new governed transition')
        if row['status'] == 'PENDING':
            # Real deterministic checks, explicitly limited to exact manual
            # correction integrity; no empirical outcome or replay is invented.
            verified = self._classify_package(candidate, version, base)
            if verified != assessment.classification or not assessment.eligible:
                raise ValueError('STALE_ASSESSMENT')
            self._governance.record_candidate_event(candidate_id=row['candidate_id'], event_type='PROCEDURE_INTEGRITY_CHECKED', actor=context.user_id, event_payload={'assessment': assessment.pin().model_dump(mode='json'), 'checks': ['exact_candidate_version_and_dependency_bytes', 'exact_predecessor_and_scope', 'deterministic_admission_routing'], 'counterevidence': list(assessment.counterevidence), 'empirical_generalization': 'unknown'})
            self._governance.record_replay_result(row['candidate_id'], regressions=[], tested_claims=3)
            self._governance.record_approval(row['candidate_id'], context.user_id, actor_roles={'scoped_owner'})
        return self._projection(context, candidate)

    def reject(self, context, candidate_ref, candidate_hash):
        if self._is_set(context, candidate_ref, candidate_hash):
            return self._set_owner().reject(context, candidate_ref, candidate_hash)
        candidate, version, base = self._objects(context, candidate_ref, candidate_hash)
        row = self._row(context, candidate)
        if row is None:
            payload = {'candidate': candidate.pin().model_dump(mode='json'), 'version': version.pin().model_dump(mode='json')}
            row = self._governance.propose(name=self._name(context, version.skill_id), category='learned_procedure', content_yaml=canonical_json(payload), content_hash=content_hash(payload), proposed_by=context.user_id, scoped_owner=context.user_id, rollback_window_hours=24, rollback_strategy='Retain rejected history')
        if row['status'] == 'PENDING':
            self._governance.record_rejection(row['candidate_id'], rejected_by=context.user_id, reason='Owner rejected exact proposal')
        elif row['status'] != 'REJECTED':
            raise ValueError('CANNOT_REJECT_ADMITTED_PROCEDURE')
        return self._projection(context, candidate)

    def _classify_package(self, candidate, version, base):
        classifier = CandidateAdmissionClassifier()
        classification = classifier.classify_text(candidate=candidate.pin(), candidate_text=self._content(version) or self._content(base), evidence_refs=(base.pin(), version.pin()))
        before = {f.relative_path:f for f in base.files}
        after = {f.relative_path:f for f in version.files}
        changed = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
        # Only documentation-only reference changes share the personal procedure
        # owner. Scripts, templates and unknown/binary formats require engineering.
        for path in changed:
            if path != 'SKILL.md' and path in after and path.endswith(('.md', '.txt')):
                try:
                    text = after[path].content_bytes().decode('utf-8')
                except UnicodeDecodeError:
                    text = ''
                routed = classifier.classify_text(candidate=candidate.pin(), candidate_text=text, evidence_refs=(base.pin(),version.pin()))
                if routed.disposition == 'AUTHORITY_VIOLATION' or (routed.disposition == 'SEMANTIC_CHANGE_REQUIRES_ADMISSION' and classification.disposition != 'AUTHORITY_VIOLATION'):
                    classification = routed
        unsupported = any(p != 'SKILL.md' and not (p.startswith('references/') and p.endswith(('.md','.txt'))) for p in changed)
        from tools.skills_tool import _safe_frontmatter
        old_meta = _safe_frontmatter(content=self._content(base)) if base.files else {}
        new_meta = _safe_frontmatter(content=self._content(version)) if version.files else {}
        capabilities = ('deps','required_environment_variables','required_credential_files','setup','prerequisites','required_binaries','metadata','platforms','allowed_tools','allowed-tools','permissions','requires')
        capability_change = bool(version.files and any(old_meta.get(k) != new_meta.get(k) for k in capabilities))
        import re
        executable_markers = re.compile(r'!`|\{\{|\{%|\{#')
        changed_executable_text = any(p in after and executable_markers.search(after[p].content_bytes().decode('utf-8', errors='replace')) for p in changed)
        unsupported = unsupported or changed_executable_text
        if (unsupported or capability_change) and classification.disposition != 'AUTHORITY_VIOLATION':
            from contracts.candidate_admission_classification import CandidateAdmissionClassificationV1
            classification = CandidateAdmissionClassificationV1(**{**classification.model_dump(), 'disposition':'SEMANTIC_CHANGE_REQUIRES_ADMISSION', 'reason_code':'PACKAGE_EFFECT_REQUIRES_ENGINEERING', 'evidence_refs':(base.pin(),version.pin())})
        policy_hash = content_hash({'classifier':classification.classifier_hash, 'version':'personal-package-effects.v1', 'documentation_prefix':'references/', 'documentation_suffixes':['.md','.txt'], 'capability_fields':capabilities, 'authority_priority':True, 'preprocess_markers':executable_markers.pattern, 'deletion_basis':'exact_retained_predecessor'})
        return classification.model_copy(update={'classifier_version':classification.classifier_version+'+personal-package-effects.v1','classifier_hash':'sha256:'+policy_hash})

    def activate(self, context, candidate_ref, candidate_hash):
        if self._is_set(context, candidate_ref, candidate_hash):
            return self._set_owner().activate(context, candidate_ref, candidate_hash)
        candidate, version, base = self._objects(context, candidate_ref, candidate_hash)
        row = self._row(context, candidate)
        if row is None or row['status'] not in ('APPROVED', 'ACTIVE'):
            raise ValueError('PROCEDURE_NOT_APPROVED')
        with self._governance.learned_procedure_transition(name=self._name(context, version.skill_id), expected_version=base.pin().model_dump(mode='json') if row['status'] != 'ACTIVE' else version.pin().model_dump(mode='json')):
            if row['status'] == 'ACTIVE' and (self._active(context, version.skill_id) or {}).get('candidate_id') != row['candidate_id']:
                raise ValueError('STALE_ACTIVATION')
            if row['status'] != 'ACTIVE':
                self._assert_base(context, base)
                self._governance.activate(row['candidate_id'], actor=context.user_id, actor_roles={'scoped_owner'})
        return self._projection(context, candidate)

    def _atomic_swap(self, link, target):
        if target.is_symlink() or not target.exists():
            os.replace(link, target)
        else:
            # Linux/WSL atomic directory-to-symlink exchange retains the legacy
            # directory at link. No gap or partial support-file replacement.
            libc = ctypes.CDLL(None, use_errno=True)
            renameat2 = libc.renameat2
            renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            if renameat2(-100, os.fsencode(link), -100, os.fsencode(target), 2):
                raise OSError(ctypes.get_errno(), 'Atomic skill exchange failed')

    def publish(self, context, candidate_ref, candidate_hash):
        if self._is_set(context, candidate_ref, candidate_hash):
            return self._set_owner().publish(context, candidate_ref, candidate_hash)
        from tools.skill_manager_tool import _skill_mutation_locks
        candidate, version, base = self._objects(context, candidate_ref, candidate_hash)
        with self._home(context), _skill_mutation_locks([version.skill_id]):
            return self._publish(context, candidate_ref, candidate_hash)

    def _publish(self, context, candidate_ref, candidate_hash):
        candidate, version, base = self._objects(context, candidate_ref, candidate_hash)
        row = self._row(context, candidate)
        if row is None or row['status'] != 'ACTIVE':
            raise ValueError('PROCEDURE_NOT_ACTIVE')
        if version.files:
            self._check_content(context, version.skill_id, self._content(version))
        target = context.hermes_home / 'skills' / version.skill_id
        if (context.hermes_home / 'skills').is_symlink():
            raise PermissionError('FORBIDDEN_PUBLICATION_PATH')
        # Resolve before touching bytes so project precedence cannot silently
        # publish an unused materialization. The predecessor remains available.
        if target.exists() and (target / 'SKILL.md').exists():
            retained_root = context.hermes_home / '.learned-procedures'
            if target.is_symlink() and retained_root.resolve() not in target.resolve().parents:
                raise ValueError('FORBIDDEN_PUBLICATION_PATH')
            if retained_root.is_dir() and not retained_root.is_symlink():
                actual_files = self._files(target)
            else:
                directory, _ = self._resolve(context, version.skill_id, target)
                actual_files = self._files(directory)
        else:
            actual_files = ()
        known = self._owned(context, SkillArtifactVersionV1, 'skill_artifact_version.v1')
        if not any(v.skill_id == version.skill_id and v.files == actual_files for v in known):
            raise ValueError('STALE_RUNTIME_MATERIALIZATION')
        # Serialize canonical head changes until readback and receipt retention.
        with self._governance.learned_procedure_transition(name=self._name(context, version.skill_id), expected_version=version.pin().model_dump(mode='json')) as active:
            if active is None or active['candidate_id'] != row['candidate_id'] or (self._active(context, version.skill_id) or {}).get('candidate_id') != row['candidate_id']:
                raise ValueError('STALE_PUBLICATION')
            activation = next(e['candidate_event_id'] for e in reversed(self._governance.list_candidate_events(row['candidate_id'])) if e['event_type'] == 'ACTIVATED')
            root = context.hermes_home / '.learned-procedures' / version.skill_id
            if root.is_symlink() or root.parent.is_symlink() or (context.hermes_home.resolve() not in root.resolve().parents):
                raise PermissionError('FORBIDDEN_PUBLICATION_PATH')
            root.mkdir(parents=True, exist_ok=True)
            materialized = root / version.artifact_hash[7:]
            if not materialized.exists():
                staging = root / ('.stage-' + uuid.uuid4().hex)
                staging.mkdir()
                for file in version.files:
                    path = staging / file.relative_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('wb') as stream:
                        stream.write(file.content_bytes())
                        stream.flush()
                        os.fsync(stream.fileno())
                if self._files(staging) != version.files:
                    raise ValueError('PUBLICATION_HASH_MISMATCH')
                (staging / '.semantier-publication.json').write_text(canonical_json({'version': version.pin().model_dump(mode='json'), 'activation_ref': activation}))
                os.replace(staging, materialized)
            if materialized.is_symlink() or self._files(materialized) != version.files:
                raise ValueError('PUBLICATION_HASH_MISMATCH')
            sidecar = materialized / '.semantier-publication.json'
            metadata = root / ('.metadata-' + uuid.uuid4().hex)
            with metadata.open('w') as stream:
                stream.write(canonical_json({'version': version.pin().model_dump(mode='json'), 'activation_ref': activation}))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(metadata, sidecar)
            link = root / ('.swap-' + uuid.uuid4().hex)
            link.symlink_to(materialized, target_is_directory=True)
            self._atomic_swap(link, target)
            # Preserve exchanged legacy bytes. Symlink-to-symlink replacement
            # has removed link; directory exchange leaves history here.
            if link.exists() and not link.is_symlink():
                os.replace(link, root / ('.legacy-' + uuid.uuid4().hex))
            self._write_generation(context, version, materialized)
            directory = materialized
            if self._files(directory) != version.files:
                raise ValueError('PUBLICATION_READBACK_MISMATCH')
            receipts = self._owned(context, LearnedSkillPublicationReceiptV1, 'learned_skill_publication_receipt.v1')
            receipt = next((r for r in receipts if r.version == version.pin() and r.activation_ref == activation), None)
            if receipt is None:
                receipt = LearnedSkillPublicationReceiptV1.create(**self._scope(context), owner_ref=context.user_id, candidate=candidate.pin(), version=version.pin(), activation_ref=activation, runtime_path=str(target), recorded_at=_now())
                self._store.put(receipt)
            self._governance.record_learned_procedure_publication(candidate_id=row['candidate_id'], actor=context.user_id, receipt=receipt.pin().model_dump(mode='json'), versions=[version.pin().model_dump(mode='json')])
        result = self._projection(context, candidate)
        result['receipt_ref'] = receipt.artifact_ref
        result['receipt_hash'] = receipt.artifact_hash
        return result

    def _write_generation(self, context, version, materialized):
        from tools.skill_review_capture import generation, digest
        from tools.skill_usage import skill_file_lock
        from utils import atomic_json_write
        with self._home(context), skill_file_lock(context.hermes_home / '.learned-procedures' / '.generation.lock'):
            packages = generation(live=True)
            packages[version.skill_id] = {'version_ref':version.artifact_ref,'version_hash':version.artifact_hash,'state':version.package_state, 'relative_path': materialized.relative_to(context.hermes_home / '.learned-procedures').as_posix(), 'files':{f.relative_path:f.content_hash for f in version.files}}
            atomic_json_write(context.hermes_home / '.learned-procedures' / 'active-generation.json', {'schema_version':'hermes.managed_generation.v1','generation_hash':digest(packages),'packages':packages})

    @staticmethod
    def package_diff(base, version):
        before = {f.relative_path:f.content_bytes() for f in base.files}
        after = {f.relative_path:f.content_bytes() for f in version.files}
        result = []
        for path in sorted(before.keys() | after.keys()):
            if before.get(path) == after.get(path):
                continue
            binary = False
            values = []
            for raw in (before.get(path), after.get(path)):
                try:
                    values.append(raw.decode('utf-8') if raw is not None else None)
                except UnicodeDecodeError:
                    values.append(None)
                    binary = True
            result.append({'skill_id':version.skill_id, 'path':path, 'operation':'create' if path not in before else ('delete' if path not in after else 'modify'), 'before':values[0], 'after':values[1], 'binary':binary})
        return result

    @staticmethod
    def _content(version):
        return next((file.content_bytes().decode('utf-8') for file in version.files if file.relative_path == 'SKILL.md'), '')

    @staticmethod
    def _sidecar_matches(directory, version, receipt):
        if receipt is None:
            return False
        if getattr(receipt, 'schema_version', '') == 'learned_skill_set_receipt.v1':
            from tools.skill_review_capture import generation
            return version.pin() in receipt.versions
        try:
            payload = json.loads((directory / '.semantier-publication.json').read_text())
            return payload == {'version': version.pin().model_dump(mode='json'), 'activation_ref': receipt.activation_ref}
        except (OSError, ValueError):
            return False

    def _publication_evidence(self, context, version, row):
        if not row or row['status'] != 'ACTIVE':
            return None
        events = self._governance.list_candidate_events(row['candidate_id'])
        activation = next((e['candidate_event_id'] for e in reversed(events) if e['event_type'] == 'ACTIVATED'), None)
        receipt = next((r for r in self._owned(context, LearnedSkillPublicationReceiptV1, 'learned_skill_publication_receipt.v1') if r.version == version.pin() and r.activation_ref == activation), None)
        if receipt is None:
            from contracts.learned_skill_publication import LearnedSkillSetReceiptV1
            receipt = next((r for r in self._owned(context, LearnedSkillSetReceiptV1, 'learned_skill_set_receipt.v1') if version.pin() in r.versions and r.activation_ref == activation), None)
        if receipt is not None and not any(e['event_type'] == 'PROCEDURE_PUBLISHED' and e['event']['payload'].get('receipt') == receipt.pin().model_dump(mode='json') for e in events):
            return None
        return receipt

    def _projection(self, context, candidate):
        if isinstance(candidate.profile, ChangeSetProfileV1):
            return self._set_owner().project(context, candidate.artifact_ref, candidate.artifact_hash)
        _, version, base = self._objects(context, candidate.artifact_ref, candidate.artifact_hash)
        row = self._row(context, candidate)
        current = not row or row['status'] != 'ACTIVE' or (self._active(context, version.skill_id) or {}).get('candidate_id') == row['candidate_id']
        receipt = self._publication_evidence(context, version, row) if current else None
        publication = 'unpublished'
        try:
            directory, _ = self._resolve(context, version.skill_id, context.hermes_home / 'skills' / version.skill_id)
            if self._files(directory) == version.files:
                bound = self._sidecar_matches(directory, version, receipt)
                publication = 'published' if bound else 'runtime_matches_retained_version'
        except ValueError:
            # Raw exact bytes remain inspectable after a crash. They are not
            # load authority until a canonical publication receipt is retained.
            target = context.hermes_home / 'skills' / version.skill_id
            retained_root = context.hermes_home / '.learned-procedures'
            if target.exists() and not retained_root.is_symlink() and retained_root.resolve() in target.resolve().parents and self._files(target) == version.files:
                publication = 'runtime_matches_retained_version'
        if version.package_state == 'absent' and row and row['status'] == 'ACTIVE' and receipt:
            from tools.skill_review_capture import generation
            with self._home(context):
                entry = generation().get(version.skill_id, {})
            if entry.get('version_ref') == version.artifact_ref and entry.get('state') == 'absent':
                publication = 'published'
        assessments = [a for a in self._owned(context, LearnedSkillAssessmentV1, 'learned_skill_assessment.v1') if a.candidate == candidate.pin()]
        assessment = max(assessments, key=lambda a: a.recorded_at) if assessments else None
        actions = []
        stale = False
        try:
            self._assert_base(context, base)
        except ValueError:
            stale = True
        if row is None and not stale:
            actions.extend(('assess_candidate', 'reject_candidate'))
            if assessment and assessment.eligible:
                actions.append('approve_candidate')
        elif row and row['status'] == 'APPROVED' and not stale:
            actions.append('activate_candidate')
        elif row and row['status'] == 'ACTIVE' and current:
            actions.append('publish_procedure')
            if version.predecessor:
                actions.append('recover_procedure')
        answer = 'Proposal saved. This change has not been applied.'
        if row and row['status'] == 'APPROVED':
            answer = 'Approved. Activation and runtime publication are still required.'
        elif row and row['status'] == 'REJECTED':
            answer = 'Rejected. This branch is retained; inspect its owner history.'
        elif row and row['status'] == 'ACTIVE':
            answer = 'Available for future sessions. Use is not recorded.' if publication == 'published' else 'Activated. Exact runtime publication is not confirmed.'
        binding = next((p for p in self._owned(context, LearnedSkillProposalV1, 'learned_skill_proposal.v1') if p.candidate == candidate.pin()), None)
        counter = list(assessment.counterevidence) if assessment else ['Assessment has not been retained.']
        if stale and (not row or row['status'] != 'ACTIVE'):
            counter.append('The predecessor changed; this proposal requires a refreshed comparison.')
        return {'object_kind': 'candidate', 'kind': 'candidate', 'disposition': row['status'] if row else 'PROPOSED', 'ref': candidate.artifact_ref, 'hash': candidate.artifact_hash, 'title': version.skill_id, 'content': self._content(version), 'before': self._content(base), 'after': self._content(version), 'revision': candidate.revision, 'origin': binding.origin if binding else None, 'proposal_ref': binding.artifact_ref if binding else None, 'proposal_hash': binding.artifact_hash if binding else None, 'proposal': 'saved', 'candidate_ref': candidate.artifact_ref, 'candidate_hash': candidate.artifact_hash, 'version_ref': version.artifact_ref, 'version_hash': version.artifact_hash, 'base_ref': base.artifact_ref, 'base_hash': base.artifact_hash, 'skill_id': version.skill_id, 'assessment_ref': assessment.artifact_ref if assessment else None, 'assessment_hash': assessment.artifact_hash if assessment else None, 'receipt_ref': receipt.artifact_ref if receipt else None, 'receipt_hash': receipt.artifact_hash if receipt else None, 'admission': 'approved' if row and row['status'] in ('APPROVED', 'ACTIVE') else 'proposed', 'activation': 'active' if row and row['status'] == 'ACTIVE' and current else ('superseded' if not current else 'inactive'), 'publication': publication, 'runtime_availability': 'available' if publication == 'published' else 'unconfirmed', 'qualification': 'unknown', 'recorded_use': 'unknown', 'causal_attribution': 'unknown', 'applicability': {'organization_id': context.organization_id, 'workspace_id': context.workspace_id, 'owner_ref': context.user_id}, 'leading_answer': answer, 'counter_evidence': counter, 'package_diff': self.package_diff(base, version), 'material_unknowns': ['Empirical generalization, observed use and causal attribution are unknown.'], 'available_actions': actions, 'history': self._governance.list_candidate_events(row['candidate_id']) if row else []}

    def inspect(self, context, ref):
        raw = self._store.get_raw(**self._scope(context), artifact_ref=ref)
        if not raw:
            raise LookupError('LEARNING_OBJECT_NOT_FOUND')
        if raw['schema_version'] == 'semantier.change_candidate.v2':
            return self._projection(context, self._load(context, ref, ChangeCandidateV2))
        version = self._load(context, ref, SkillArtifactVersionV1)
        available = 'unavailable'
        active = self._active(context, version.skill_id)
        receipt = self._publication_evidence(context, version, active)
        try:
            directory, _ = self._resolve(context, version.skill_id, context.hermes_home / 'skills' / version.skill_id)
            if self._files(directory) == version.files:
                available = 'legacy_runtime_available' if version.provenance == 'legacy' else ('available' if self._sidecar_matches(directory, version, receipt) else 'unconfirmed')
        except ValueError:
            pass
        active_payload = json.loads(active['content_yaml']) if active else {}
        is_active = bool(active and active_payload.get('member_versions', {}).get(version.skill_id, active_payload.get('version')) == version.pin().model_dump(mode='json'))
        return {**version.model_dump(mode='json'), 'object_kind': 'procedure', 'kind': 'procedure', 'ref': version.artifact_ref, 'hash': version.artifact_hash, 'title': version.skill_id, 'content': self._content(version), 'revision': version.version, 'activation': 'active' if is_active else 'unknown', 'admission': 'approved' if is_active else 'unknown', 'runtime_availability': available, 'qualification': 'unknown', 'recorded_use': 'unknown', 'causal_attribution': 'unknown', 'leading_answer': 'Legacy procedure is available; historical qualification is unknown.' if available == 'legacy_runtime_available' else ('Available for future sessions. Use is not recorded.' if available == 'available' else 'Exact version retained; availability is not confirmed.'), 'counter_evidence': ['Observed use, empirical qualification and causal attribution are unknown.'], 'available_actions': (['propose_procedure'] + (['recover_procedure'] if version.predecessor and is_active else [])) if available in ('available', 'legacy_runtime_available') else [], 'applicability': {'organization_id': context.organization_id, 'workspace_id': context.workspace_id, 'owner_ref': context.user_id}, 'history': self._governance.list_candidate_events(active['candidate_id']) if is_active else []}

    def recover(self, context, version_ref, version_hash, request_id):
        if version_ref.startswith('change_candidate:') and self._is_set(context, version_ref, version_hash):
            return self._set_owner().recover(context, version_ref, version_hash, request_id)
        version = self._load(context, version_ref, SkillArtifactVersionV1, version_hash)
        if version.predecessor is None:
            raise ValueError('RECOVERY_PREDECESSOR_UNAVAILABLE')
        predecessor = self._load(context, version.predecessor.artifact_ref, SkillArtifactVersionV1, version.predecessor.artifact_hash)
        content = self._content(predecessor)
        proposal = self.propose(context, version.artifact_ref, version.artifact_hash, content, 'recover:' + request_id, origin={'operation': 'restore_predecessor', 'recovery_target': predecessor.pin().model_dump(mode='json'), 'evidence_refs': [predecessor.pin().model_dump(mode='json')]}, package_files=predecessor.files)
        candidate = self._load(context, proposal['candidate_ref'], ChangeCandidateV2, proposal['candidate_hash'])
        row = self._row(context, candidate)
        if row is None:
            assessment = self.assess(context, candidate.artifact_ref, candidate.artifact_hash)
            self.approve(context, candidate.artifact_ref, candidate.artifact_hash, assessment['artifact_ref'], assessment['artifact_hash'])
        self.activate(context, candidate.artifact_ref, candidate.artifact_hash)
        return self.publish(context, candidate.artifact_ref, candidate.artifact_hash)
