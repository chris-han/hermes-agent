"""Native consumer of an authenticated managed package response.

The caller obtains the response through current device proof. These integrity and
installer checks create no cloud entitlement or local substitute for core authority.
"""
from hermes_cli.semantier_personal_contract import (
    ArtifactPinV1, PersonalCapabilityTransferV1, SkillArtifactVersionV1,
)
from tools.skills_hub_models import SkillSource,SkillBundle,SkillMeta


class PersonalPackageSource(SkillSource):
    SOURCE_ID='semantier-personal'

    def __init__(self,package):
        if not isinstance(package,dict) or set(package)!={'source_import','source_transfer','candidate','publication','approved_version'}:
            raise ValueError('exact personal package response required')
        self.source=PersonalCapabilityTransferV1.model_validate(package['source_transfer'])
        self.version=SkillArtifactVersionV1.model_validate(package['approved_version'])
        pins={key:ArtifactPinV1.model_validate(package[key]) for key in ('source_import','candidate','publication')}
        if any(pins[key].artifact_kind!=kind for key,kind in (
                ('source_import','personal_capability_import'),('candidate','change_candidate'),('publication','learned_skill_publication_receipt'))):
            raise ValueError('personal publication pins required')
        if ((self.source.organization_id,self.source.workspace_id,self.source.owner_ref,self.source.skill.skill_id,self.source.skill.files)
                !=(self.version.organization_id,self.version.workspace_id,self.version.owner_ref,self.version.skill_id,self.version.files)):
            raise ValueError('approved package must preserve exact source author and bytes')
        self.identifier=self.SOURCE_ID+'/'+self.version.artifact_hash[7:]+'/'+self.version.skill_id
        self.metadata={**{key:pin.model_dump(mode='json') for key,pin in pins.items()},
            'source_transfer':self.source.pin().model_dump(mode='json'),'approved_version':self.version.pin().model_dump(mode='json'),
            'source_store_incarnation':self.source.source_store_incarnation,'source_object_id':self.source.source_object_id,
            'source_revision':self.source.source_revision,'author_ref':self.source.owner_ref,'allowed_purpose':self.source.allowed_purpose}

    def search(self,query,limit=10):
        return []  # Private pinned delivery never enters public discovery.

    def inspect(self,identifier):
        if identifier!=self.identifier:
            return None
        return SkillMeta(name=self.version.skill_id,description='Personal source-linked Skill',source=self.SOURCE_ID,
            identifier=identifier,trust_level='community',extra=self.metadata)

    def fetch(self,identifier):
        if identifier!=self.identifier:
            return None
        return SkillBundle(name=self.version.skill_id,files={file.relative_path:file.content_bytes() for file in self.version.files},
            source=self.SOURCE_ID,identifier=identifier,trust_level='community',metadata=self.metadata)


def _windows_package_paths(paths):
    folded=set()
    reserved={'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}
    for path in paths:
        if path.casefold() in folded or any(part.rstrip(' .')!=part or any(char in part for char in '<>:\"|?*')
                or part.split('.',1)[0].upper() in reserved for part in path.split('/')):
            raise ValueError('personal package paths are not unambiguous Windows files')
        folded.add(path.casefold())


def _compatible(source):
    from tools.skills_tool import _safe_frontmatter,skill_matches_platform,_is_skill_disabled
    _windows_package_paths([source.version.skill_id,*(file.relative_path for file in source.version.files)])
    content=next(file.content_bytes().decode('utf-8') for file in source.version.files if file.relative_path=='SKILL.md')
    metadata=_safe_frontmatter(content=content)
    if metadata.get('name',source.version.skill_id)!=source.version.skill_id or not skill_matches_platform(metadata) or _is_skill_disabled(source.version.skill_id):
        raise ValueError('personal Skill is unavailable on this native profile')
    if any(metadata.get(key) for key in ('deps','setup','required_environment_variables','required_credential_files','required_binaries','prerequisites','requires')):
        raise ValueError('personal MVL requires a dependency-small Skill; native setup remains separate')


def _existing(source,replace_existing):
    from hermes_constants import get_hermes_home
    from tools.skills_hub import HubLockFile
    from tools.skills_hub_install import _resolve_lock_install_path,bundle_content_hash
    from tools.skills_guard import content_hash
    entry=HubLockFile().get_installed(source.version.skill_id)
    target=get_hermes_home()/'skills'/source.version.skill_id
    if target.parent.is_symlink() or getattr(target.parent,'is_junction',lambda:False)():
        raise ValueError('personal install requires the active native profile skills root')
    if not entry:
        if target.exists() or target.is_symlink():
            raise ValueError('personal install would replace an unmanaged local Skill')
        return None
    old=entry.get('metadata') or {}
    if (entry['source'],old.get('source_store_incarnation'),old.get('source_object_id'),old.get('author_ref'))!=(source.SOURCE_ID,source.source.source_store_incarnation,source.source.source_object_id,source.source.owner_ref):
        raise ValueError('personal installation source conflict')
    path=_resolve_lock_install_path(entry['install_path'],source.version.skill_id)
    if content_hash(path)!=entry['content_hash']:
        raise ValueError('personal installation has local edits')
    if old.get('approved_version')==source.metadata['approved_version']:
        if content_hash(path)!=bundle_content_hash(source.fetch(source.identifier)):
            raise ValueError('personal installation bytes do not match approved version')
        return 'unchanged'
    if not replace_existing or source.source.predecessor is None or source.source.predecessor.model_dump(mode='json')!=old.get('source_import'):
        raise ValueError('explicit exact-predecessor personal update required')
    return 'replace'


def install_personal_package(package,*,confirmed=False,replace_existing=False,console=None):
    if confirmed is not True:
        raise PermissionError('explicit personal package installation required')
    source=PersonalPackageSource(package)
    _compatible(source)
    existing=_existing(source,replace_existing)
    if existing=='unchanged':
        return {'status':'unchanged','metadata':source.metadata}
    from hermes_cli.skills_hub import do_install
    installed=do_install(source.identifier,source_id=source.SOURCE_ID,source_adapter=source,
        skip_confirm=True,force=existing=='replace',invalidate_cache=False,console=console)
    return {'status':'installed' if installed is True else 'not_installed','metadata':source.metadata}
