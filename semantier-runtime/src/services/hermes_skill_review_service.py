"""Authenticated exact Hermes intake. Native custody never grants authority."""
from __future__ import annotations
from contextlib import contextmanager
import base64
import json
from pathlib import Path
from agents.auth_session import RequestContext
from runtime_paths import workspace_hermes_home_path, bind_workspace_env
from services.learned_skill_publication_service import _file, _now
from contracts.learning_workbench import SkillArtifactVersionV1


class HermesSkillReviewService:
    def __init__(self, *, procedures):
        self.procedures = procedures

    def _scope(self, context):
        if not isinstance(context, RequestContext) or not context.authenticated or not context.user_id or not context.workspace_id:
            raise PermissionError('FORBIDDEN_LEARNING_SCOPE')
        expected = workspace_hermes_home_path(context.workspace_id).resolve()
        if context.hermes_home.resolve() != expected or context.workspace_root.resolve() != expected:
            raise PermissionError('FORBIDDEN_RUNTIME_HOME')

    @contextmanager
    def _home(self, context):
        self._scope(context)
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        token = set_hermes_home_override(context.hermes_home)
        try:
            yield
        finally:
            reset_hermes_home_override(token)

    def _project(self, context, record):
        from tools.skill_review_capture import digest
        capture = record.get('exact_capture')
        capture = capture if isinstance(capture, dict) else None
        native_payload_hash = digest(record.get('payload', {}))
        payload_hash = digest({'payload':record.get('payload', {}),'exact_capture':capture,'custody':record.get('custody')})
        ready = bool(capture and capture.get('schema_version') == 'hermes.skill_stage.v1' and capture.get('payload_hash') == native_payload_hash and capture.get('capture_hash') == digest({k:v for k,v in capture.items() if k != 'capture_hash'}))
        packages = capture.get('packages', []) if ready else []
        try:
            from tools.skill_review_capture import name_check
            if not isinstance(packages, list) or not packages:
                raise ValueError('EMPTY_CAPTURE')
            names = [name_check(p['skill_id']) for p in packages]
            if len(names) != len(set(names)):
                raise ValueError('DUPLICATE_CAPTURE_MEMBER')
            for package in packages:
                for field in ('base', 'candidate'):
                    if not isinstance(package[field], dict) or (package[field] and 'SKILL.md' not in package[field]):
                        raise ValueError('INVALID_CAPTURE_PACKAGE')
                    for path, raw in package[field].items():
                        _file(path, base64.b64decode(raw, validate=True))
        except (ValueError,TypeError,KeyError):
            ready = False
            packages = []
        diffs = []
        for package in packages:
            before = package['base']; after = package['candidate']
            for path in sorted(before.keys() | after.keys()):
                if before.get(path) == after.get(path):
                    continue
                values = []; binary = False
                for raw in (before.get(path), after.get(path)):
                    try:
                        values.append(base64.b64decode(raw,validate=True).decode('utf-8') if raw is not None else None)
                    except (ValueError,UnicodeDecodeError):
                        values.append(None); binary = True
                diffs.append({'skill_id':package['skill_id'],'path':path,'before':values[0],'after':values[1],'binary':binary,'operation':'create' if path not in before else ('delete' if path not in after else 'modify')})
        return {'package_diff':diffs, 'object_kind':'pending', 'kind':'pending', 'ref':record['id'], 'hash':payload_hash, 'native_id':record['id'], 'payload_hash':payload_hash, 'revision':payload_hash, 'capture_hash':capture.get('capture_hash') if capture else None, 'title':record.get('summary') or 'Pending skill change', 'origin':record.get('origin','foreground'), 'recorded_at':record.get('created_at'), 'readiness':'ready' if ready else 'original_base_unknown', 'packages':packages, 'available_actions':['import_pending'] if ready and context.organization_id else [], 'counter_evidence':[] if ready else ['Original base is unknown; recreate an exact proposal before import.'], 'leading_answer':'Pending exact review; no runtime change.', 'applicability':{'organization_id':context.organization_id,'workspace_id':context.workspace_id,'owner_ref':context.user_id}}

    def pending(self, context):
        from tools.write_approval import list_pending, SKILLS
        with self._home(context):
            records = list_pending(SKILLS)
        return [self._project(context,r) for r in records if r.get('custody') == {'user_id':context.user_id,'workspace_id':context.workspace_id}]

    def inspect_pending(self, context, native_id, payload_hash):
        from tools.write_approval import get_pending, SKILLS
        with self._home(context):
            record = get_pending(SKILLS,native_id)
        if record is None or record.get('custody') != {'user_id':context.user_id,'workspace_id':context.workspace_id}:
            raise LookupError('PENDING_NOT_FOUND')
        result = self._project(context,record)
        if result['payload_hash'] != payload_hash:
            raise ValueError('PENDING_PAYLOAD_HASH_CONFLICT')
        return result

    def import_pending(self, context, native_id, payload_hash, request_id):
        scope = self.procedures._scope(context)
        from tools.skill_manager_tool import _skill_mutation_locks
        from tools.skill_review_capture import name_check, read_package
        detail = self.inspect_pending(context,native_id,payload_hash)
        if detail['readiness'] != 'ready':
            raise ValueError('PENDING_CAPTURE_TAMPERED_OR_BASE_UNKNOWN')
        packages = detail['packages']
        if len(packages) > 1:
            with self._home(context), _skill_mutation_locks([name_check(p['skill_id']) for p in packages]):
                self.inspect_pending(context,native_id,payload_hash)
                return self.procedures._set_owner().propose(context,packages,{'native_id':native_id,'payload_hash':payload_hash},'hermes-set:'+native_id+':'+payload_hash)
        if not packages:
            raise ValueError('EMPTY_SKILL_PACKAGE')
        package = packages[0]
        name = name_check(package['skill_id'])
        with self._home(context), _skill_mutation_locks([name]):
            # Re-read source after lock: native ID replacement cannot retarget approval.
            self.inspect_pending(context,native_id,payload_hash)
            target = context.hermes_home/'skills'/name
            files = tuple(_file(p,base64.b64decode(b,validate=True)) for p,b in sorted(package['base'].items()))
            versions = self.procedures._owned(context,SkillArtifactVersionV1,'skill_artifact_version.v1')
            pinned = package.get('base_version')
            base = next((v for v in versions if v.skill_id == name and v.files == files and (not pinned or (v.artifact_ref == pinned['artifact_ref'] and v.artifact_hash == pinned['artifact_hash']))),None)
            if base is None:
                if read_package(target) != package['base']:
                    raise ValueError('STALE_PREDECESSOR')
                if versions and any(v.skill_id == name for v in versions):
                    raise ValueError('STALE_PREDECESSOR')
                base = SkillArtifactVersionV1.create(**scope, owner_ref=context.user_id, skill_id=name, version=1, provenance='legacy', files=files, recorded_at=_now())
                self.procedures._store.put(base)
            candidate_files = tuple(_file(p,base64.b64decode(b,validate=True)) for p,b in sorted(package['candidate'].items()))
            content = next((f.content_bytes().decode('utf-8') for f in candidate_files if f.relative_path == 'SKILL.md'),'')
            return self.procedures.propose(context,base.artifact_ref,base.artifact_hash,content,'hermes:'+native_id+':'+payload_hash,origin={'native_id':native_id,'payload_hash':payload_hash,'capture_hash':detail['capture_hash'],'operation':'hermes_pending','origin':detail['origin']},package_files=candidate_files)

    def acknowledge(self, context, native_id, payload_hash):
        # Source remains retained; acknowledgment is a receipt alongside it.
        from tools.write_approval import _pending_path, SKILLS
        from tools.skill_manager_tool import _skill_mutation_locks
        from utils import atomic_json_write
        with self._home(context), _skill_mutation_locks(['pending-ack']):
            self.inspect_pending(context,native_id,payload_hash)
            path = _pending_path(SKILLS,native_id)
            record = json.loads(path.read_text())
            record['acknowledged_payload_hash'] = payload_hash
            atomic_json_write(path,record)
        return True


@contextmanager
def bind_runtime_package_resolver(boundary):
    """Bind the read owner using governed identity, never a home-derived actor."""
    from tools.skill_review_capture import bind_generation_verifier
    from agents.auth_session import _gateway_identity_module
    from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path
    from storage.governance_store import GovernanceStore, _db_path
    from storage.hosted_postgres import enabled as hosted_storage_enabled
    from services.learned_skill_publication_service import LearnedSkillPublicationService
    home = Path(boundary.paths.hermes_home) if boundary.paths.hermes_home else None
    if home is None:
        raise PermissionError('MANAGED_PACKAGE_AUTHORITY_UNAVAILABLE')
    identity = _gateway_identity_module()
    record = identity.get_user_record(boundary.user_id) if boundary.user_id else None
    organization = identity.get_user_organization_context(boundary.user_id) if boundary.user_id else None
    if not record or record.get('workspace_id') != boundary.workspace_id:
        raise PermissionError('MANAGED_PACKAGE_AUTHORITY_UNAVAILABLE')
    expected = workspace_hermes_home_path(boundary.workspace_id)
    hosted = hosted_storage_enabled()
    if home.resolve() != expected.resolve() or (not hosted and not _db_path().is_file()):
        raise PermissionError('FORBIDDEN_RUNTIME_HOME')
    governance = GovernanceStore()
    path = ontology_build_store_path(home)
    organization_id = organization.get('organization_id') if organization and organization.get('membership_status') == 'active' else None
    # No publication history means legacy skill loading needs no invented org
    # or artifact store. Once published, missing authority must be unavailable.
    if not organization_id or (not hosted and not path.is_file()):
        published = any(row['scoped_owner'] == boundary.user_id and (':'+boundary.workspace_id+':'+boundary.user_id+':') in row['name'] and any(e['event_type']=='PROCEDURE_PUBLISHED' for e in governance.list_candidate_events(row['candidate_id'])) for row in governance.list_candidates(category='learned_procedure'))
        def empty_generation(packages):
            if published or packages:
                raise PermissionError('MANAGED_PACKAGE_AUTHORITY_UNAVAILABLE')
            return frozenset()
        try:
            with bind_generation_verifier(empty_generation,(organization_id,boundary.workspace_id,boundary.user_id)):
                yield
        finally:
            governance.close()
        return
    context = RequestContext(authenticated=True,user_id=boundary.user_id,workspace_id=boundary.workspace_id,workspace_slug=record.get('workspace_slug') or boundary.workspace_id,organization_id=organization_id,membership_status='active',workspace_root=home,hermes_home=home)
    store = OntologyBuildStore(path)
    owner = LearnedSkillPublicationService(store=store,governance=governance)
    try:
        with bind_generation_verifier(lambda packages:owner._verify_runtime_generation_fresh(context,packages),(context.organization_id,context.workspace_id,context.user_id)):
            yield
    finally:
        store.close()
        governance.close()
