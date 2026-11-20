"""Explicit one-Skill/one-context capture through the active native profile.

Electron owns durable pending snapshots in its existing local settings. This
module reads selected source bytes only; it neither admits nor publishes them.
"""
from datetime import UTC, datetime
import base64
import hashlib
from pathlib import PurePosixPath

from hermes_constants import get_hermes_home
from hermes_cli.semantier_personal_contract import (
    ArtifactPinV1, PersonalCapabilityTransferV1, RetainedContextItemChangeV1,
    SkillArtifactVersionV1, SkillFileV1, content_hash,
)


def _permitted(path):
    parts=[part.casefold() for part in PurePosixPath(path).parts]
    return not (any(p.startswith('.env') or p in {'.aws','.ssh','credentials','credentials.json','sessions','state.db','auth.db'} for p in parts)
        or PurePosixPath(path).suffix.casefold() in {'.db','.sqlite','.sqlite3','.pem','.key','.pfx'}
        or PurePosixPath(path).name.casefold().endswith(('-wal','-shm','.wal','.lock')))


def snapshot(skill_id):
    from tools.skill_review_capture import name_check
    from tools.skill_manager_tool import _skill_mutation_locks
    from tools.memory_tool import load_on_disk_store
    from hermes_cli.semantier_personal_package import _windows_package_paths
    from tools.skills_hub import HubLockFile
    root=get_hermes_home()
    directory=root/'skills'/name_check(skill_id)
    if HubLockFile().get_installed(skill_id):
        raise ValueError('distributed Skill retains its existing source; select a local personal learned Skill')
    if any(path.is_symlink() or getattr(path,'is_junction',lambda:False)() for path in (root/'skills',directory)) or not directory.is_dir():
        raise ValueError('selected personal Skill must be in the active profile')
    files=[]
    remaining=1048576
    with _skill_mutation_locks([skill_id]):
        for path in sorted(directory.rglob('*')):
            if path.is_symlink() or getattr(path,'is_junction',lambda:False)():
                raise ValueError('selected package contains a redirected path')
            if not path.is_file():
                continue
            relative=path.relative_to(directory).as_posix()
            if not _permitted(relative) or len(files)>=128:
                raise ValueError('selected package contains disallowed or excessive files')
            with path.open('rb') as stream:
                data=stream.read(remaining+1)
            remaining-=len(data)
            if remaining<0:
                raise ValueError('selected package exceeds bounded payload')
            files.append(SkillFileV1(relative_path=relative,content_base64=base64.b64encode(data).decode('ascii'),
                content_hash='sha256:'+hashlib.sha256(data).hexdigest()).model_dump(mode='json'))
    _windows_package_paths([skill_id,*(file['relative_path'] for file in files)])
    if not any(file['relative_path']=='SKILL.md' for file in files):
        raise ValueError('selected Skill requires SKILL.md')
    memory=root/'memories'
    if any(path.is_symlink() or getattr(path,'is_junction',lambda:False)() for path in (memory,memory/'MEMORY.md',memory/'USER.md')):
        raise ValueError('selected context contains a redirected path')
    store=load_on_disk_store()
    contexts=[]
    for name,entries in [('MEMORY.md',store.memory_entries),('USER.md',store.user_entries)]:
        for entry in entries:
            if entry and len(entry.encode('utf-8'))<=16384 and len(contexts)<128:
                contexts.append({'source_file':name,'content':entry,
                    'content_hash':'sha256:'+hashlib.sha256(entry.encode('utf-8')).hexdigest()})
    value={'skill_id':skill_id,'files':files,'contexts':contexts}
    return value|{'snapshot_hash':'sha256:'+content_hash(value)}


def capture(selected,context_record,*,scope,incarnation,previous=None,acknowledgement=None):
    # Reject renderer edits and local source changes since the user's preview.
    if not isinstance(selected,dict) or selected!=snapshot(selected.get('skill_id')):
        raise ValueError('selected source changed; preview again')
    if context_record not in selected['contexts']:
        raise ValueError('exact selected context required')
    if not isinstance(scope,dict) or set(scope)!={'organization_id','workspace_id','owner_ref'} or not all(isinstance(v,str) and v for v in scope.values()):
        raise ValueError('exact authenticated source scope required')
    old=PersonalCapabilityTransferV1.model_validate(previous) if previous is not None else None
    if old and ((old.organization_id,old.workspace_id,old.owner_ref)!=(scope['organization_id'],scope['workspace_id'],scope['owner_ref'])
            or old.source_store_incarnation!=incarnation or old.skill.skill_id!=selected['skill_id']
            or old.decision_context.source_file!=context_record['source_file']):
        raise ValueError('personal source scope or identity changed')
    files=tuple(SkillFileV1.model_validate(file) for file in selected['files'])
    same_skill=old is not None and files==old.skill.files
    same_note=old is not None and context_record['content']==old.decision_context.content
    if same_skill and same_note:
        return old
    if old and acknowledgement is None:
        raise ValueError('durable predecessor ACK required before recapturing changed source')
    now=datetime.now(UTC)
    skill=old.skill if same_skill else SkillArtifactVersionV1.create(**scope,skill_id=selected['skill_id'],
        version=old.skill.version+1 if old else 1,provenance='managed' if old else 'legacy',
        predecessor=old.skill.pin() if old else None,files=files,recorded_at=now)
    note=old.decision_context if same_note else RetainedContextItemChangeV1.create(**scope,
        item_ref=old.decision_context.item_ref if old else 'personal-selected-context:'+content_hash([selected['skill_id'],context_record['source_file']]),
        source_file=context_record['source_file'],revision=old.decision_context.revision+1 if old else 1,
        expected_revision=old.decision_context.artifact_hash if old else 'source-initial',
        prior_content=old.decision_context.content if old else '',content=context_record['content'],action='correct',
        request_id='personal-context:'+content_hash([incarnation,selected['skill_id'],context_record]),recorded_at=now,
        origin={'source_ref':skill.artifact_ref,'source_hash':skill.artifact_hash,
            'operation':'personal_selected_context_capture','source_file':context_record['source_file'],
            'source_content_hash':context_record['content_hash']})
    return PersonalCapabilityTransferV1.create(**scope,source_store_incarnation=incarnation,
        source_object_id=content_hash([skill.skill_id,note.item_ref]),source_revision=f'{skill.version}:{note.revision}',
        allowed_purpose='personal_capability_reuse',skill=skill,decision_context=note,
        context_basis=old.context_basis if same_note else skill,
        predecessor=ArtifactPinV1.model_validate(acknowledgement) if old else None,recorded_at=max(skill.recorded_at,note.recorded_at))


# Native authenticated dashboard input. These values are source metadata, never
# cloud authority; Runtime rechecks the actual authenticated owner on transfer.
from pydantic import BaseModel, ConfigDict, Field


class PersonalSnapshotCommand(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    skill_id: str=Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$')


class PersonalCaptureCommand(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    selected: dict
    context_record: dict
    scope: dict[str,str]
    incarnation: str=Field(pattern=r'^[a-f0-9]{32}$')
    previous: dict | None=None
    acknowledgement: dict | None=None


class PersonalInstallCommand(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    package: dict
    confirmed: bool=False
    replace_existing: bool=False
