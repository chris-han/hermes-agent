"""Bounded Windows x64 acceptance identity. No upstream publication authority."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from hermes_cli.release_channels import public_base,validate_request

IDENTITY={'token':'b0fdc2a7e4b08d16','displayName':'Semantier Agent Acceptance',
'appId':'com.semantier.agent.acceptance','appNamePascal':'SemantierAgentAcceptance',
'artifactNamePascal':'SemantierAgent','cliName':'semantier-agent-acceptance',
'windowsExecutableName':'Semantier Agent','msixAppIdWithOrg':'Semantier.Agent.Acceptance'}


def is_acceptance_request(value):
    return (value.get('semantierScope')=='semantier-windows-x64/v1' and value.get('identity')==IDENTITY
            and value.get('channel')=='semantier-acceptance' and value.get('repository')=='chris-han/hermes-agent'
            and re.fullmatch(r'0\.0\.[1-9][0-9]*',value.get('version','')) is not None)


def acceptance_request(*,commit,controller,version,source_version,base):
    if not re.fullmatch(r'0\.0\.[1-9][0-9]*',version):raise ValueError('acceptance uses the existing numeric patch allocation')
    sequence=int(version.split('.')[2])
    if sequence>65535:raise ValueError('acceptance sequence exceeds Windows version range')
    base=public_base(base)
    value=dict(schema=1,buildId=uuid4().hex,channel='semantier-acceptance',sequence=sequence,
               repository='chris-han/hermes-agent',commit=commit,controllerCommit=controller,
               sourceVersion=source_version,version=version,windowsVersion=version+'.0',
               releaseTag='v'+version,identity=IDENTITY,bundleEnv={'HERMES_HOME':None,'HERMES_DATA_DIR_SUFFIX':'-semantier-acceptance'},publicBase=base,
               semantierScope='semantier-windows-x64/v1')
    validate_request(value)
    return value



def _native_object_identity(row,kind,identity,publisher,version,arch,identities):
    key=row['key']
    extension='.msixbundle' if kind.endswith('bundle') else '.msix'
    if not key.endswith(extension):raise ValueError('Native object extension mismatch')
    name=row.get('identity')
    if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9.-]{3,50}',name):raise ValueError('Invalid native identity')
    resource=row.get('resourceId','')
    if kind=='asset-package':
        if name!=identity or not isinstance(resource,str) or not re.fullmatch('runtime[0-9]{4}',resource) or row.get('allowExecution') is not True:
            raise ValueError('Invalid executable asset package binding')
    elif resource:raise ValueError('Unexpected native resource identity')
    if (name,kind,resource) in identities:raise ValueError('Ambiguous native package identity')
    identities.add((name,kind,resource))
    if kind.startswith('main-') and name!=identity:raise ValueError('Main native identity mismatch')
    if kind.startswith('optional-') and (name==identity or not name.startswith(identity+'.')):
        raise ValueError('Related native identity mismatch')
    if any(row.get(k)!=v for k,v in dict(publisher=publisher,version=version,arch='neutral' if kind.endswith('bundle') or kind=='asset-package' else arch).items()):
        raise ValueError('Native identity/version/architecture mismatch')


def validate_native_graph(value, *, build_id, identity, publisher, version, arch):
    """Producer-side structural gate; never proof of signing or native execution.

    Remote URIs are derived from qualified base + immutable keys, never supplied
    independently. Every referenced outer/inner object has its own bytes/digest
    binding. Keep this gate separate from admission until real layout proof.
    """
    import base64
    from hermes_cli.release_channels import artifact_key, build_prefix, require_sha256
    if not isinstance(value, dict) or type(value.get('schema')) is not int or value['schema'] != 1 or value.get('transport') != 'windows-package-uri':
        raise ValueError('Unsupported native graph')
    require_sha256(value.get('payloadInventorySha256'))
    prefix=build_prefix(build_id)
    objects=value.get('objects')
    if not isinstance(objects,list) or not objects or len(objects)>64:
        raise ValueError('Invalid native object set')
    rows={}
    identities=set()
    for row in objects:
        if not isinstance(row,dict):raise ValueError('Invalid native object')
        key=artifact_key(row.get('key'))
        if not key.startswith(prefix) or key in rows:raise ValueError('Native object build/uniqueness mismatch')
        kind=row.get('kind')
        if kind not in ('main-package','optional-package','asset-package','main-bundle','optional-bundle'):
            raise ValueError('Invalid native object kind')
        _native_object_identity(row,kind,identity,publisher,version,arch,identities)
        if type(row.get('size')) is not int or not 0<row['size']<=480_000_000:
            raise ValueError('Native object exceeds cacheable cap')
        require_sha256(row.get('sha256'))
        digest=row.get('nativeDigest')
        try:
            decoded=base64.b64decode(digest,validate=True)
            if len(decoded)!=32 or base64.b64encode(decoded).decode()!=digest:raise ValueError('Invalid native digest')
        except (ValueError,TypeError) as exc:raise ValueError('Invalid native digest') from exc
        refs=row.get('references')
        if not isinstance(refs,list) or any(not isinstance(k,str) for k in refs) or len(set(refs))!=len(refs):
            raise ValueError('Invalid native references')
        if kind.endswith('package') and refs:raise ValueError('Package cannot introduce unbound dependencies')
        rows[key]=row
    root=value.get('root')
    if not isinstance(root,str) or root not in rows or rows[root]['kind'] not in ('main-package','main-bundle'):
        raise ValueError('Invalid native graph root')
    if sum(r['kind']=='main-package' for r in objects)!=1 or sum(r['kind']=='main-bundle' for r in objects)>1:
        raise ValueError('Ambiguous native main package')
    seen=set(); active=set()
    def visit(key):
        if key not in rows:raise ValueError('Missing native object')
        if key in active:raise ValueError('Native graph cycle')
        if key in seen:return
        active.add(key)
        for reference in rows[key]['references']:visit(reference)
        active.remove(key);seen.add(key)
    visit(root)
    if seen!=set(rows):raise ValueError('Unreachable native graph object')
    return value



def _native_payload_files(inventory,max_bytes,reserve):
    from pathlib import PurePosixPath
    from urllib.parse import unquote
    capacity=max_bytes-reserve
    footprints={'AppxManifest.xml','AppxBlockMap.xml','AppxSignature.p7x','[Content_Types].xml'}
    files={};folded=set()
    for record in inventory:
        name=unquote(record['path'],errors='strict')
        path=PurePosixPath(name)
        if (not name or path.is_absolute() or path.as_posix()!=name or any(c in name for c in '\\:\x00"\r\n*?<>|')
                or any(part in ('.','..') or part.endswith((' ','.')) or re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?',part,re.IGNORECASE) for part in path.parts)):
            raise ValueError('Unsafe native payload path')
        if name in footprints or name.startswith('AppxMetadata/'):continue
        if name.casefold() in folded:raise ValueError('Native payload path collision')
        folded.add(name.casefold())
        size=record.get('bytes');compressed=record.get('compressed_bytes')
        if type(size) is not int or type(compressed) is not int or size<0 or compressed<0:
            raise ValueError('Invalid native payload size')
        weight=compressed+300+4*len(name.encode('utf-8'))+((size+65535)//65536)*80
        if weight>capacity:raise ValueError(f'Indivisible native payload exceeds package cap: {name}')
        files[name]={'path':name,'weight':weight}
    if not files:raise ValueError('Native payload is empty')
    return files


def partition_native_assets(inventory, *, max_bytes=480_000_000, previous=None, main_required=()):
    """Plan whole-file asset packages; SDK/post-sign sizes remain decisive.

    Native package folding preserves original relative paths. This plan is not
    qualification and never authorizes deployment/publication. Existing members
    keep their assignments across A/B; an overflow requires explicit relayout.
    """
    if type(max_bytes) is not int or not 8_000_000 <= max_bytes <= 480_000_000:
        raise ValueError('Native package cap outside supported planning bounds')
    reserve=4_000_000
    files=_native_payload_files(inventory,max_bytes,reserve)
    required=set(main_required)|{p for p in files if not p.startswith('app/resources/agent-payload/')}
    if not required<=set(files):raise ValueError('Required main payload absent')
    assignment={};groups={'main':{'id':'main','files':[],'estimatedBytes':reserve}}
    def put(name,group):
        row=files[name]
        if group not in groups:groups[group]={'id':group,'files':[],'estimatedBytes':reserve}
        if groups[group]['estimatedBytes']+row['weight']>max_bytes:
            raise ValueError(f'Native assignment overflow requires qualified relayout: {group}')
        groups[group]['files'].append(name);groups[group]['estimatedBytes']+=row['weight'];assignment[name]=group
    if previous is not None:
        if not isinstance(previous,dict):raise ValueError('Invalid previous native assignment')
        for name,group in previous.items():
            if name not in files:continue
            if not isinstance(group,str) or not (group=='main' or re.fullmatch('runtime[0-9]{4}',group)):
                raise ValueError('Invalid native asset ID')
            if name in required and group!='main':raise ValueError('Required main payload moved to asset package')
            put(name,group)
    for name in sorted(required-set(assignment)):put(name,'main')
    for name in sorted(set(files)-set(assignment),key=lambda p:(-files[p]['weight'],p)):
        candidates=[g for g,v in groups.items() if v['estimatedBytes']+files[name]['weight']<=max_bytes]
        if candidates:
            group=min(candidates,key=lambda g:(max_bytes-groups[g]['estimatedBytes']-files[name]['weight'],g))
        else:
            index=1
            while f'runtime{index:04d}' in groups:index+=1
            group=f'runtime{index:04d}'
        put(name,group)
    for group in groups.values():group['files'].sort()
    return {'schema':1,'kind':'windows-folded-assets','maxObjectBytes':max_bytes,
            'assignment':dict(sorted(assignment.items())),'groups':list(groups.values()),
            'qualification':'NOT_RUN'}


def write_native_asset_layout(output, plan, *, manifest, source):
    """Emit the existing SDK packaging-layout contract; never rewrite payloads."""
    import ntpath
    import xml.etree.ElementTree as ET
    namespace='http://schemas.microsoft.com/appx/makeappx/2017'
    ET.register_namespace('',namespace)
    def node(parent,name,**attributes):return ET.SubElement(parent,'{'+namespace+'}'+name,attributes)
    root=ET.Element('{'+namespace+'}PackagingLayout')
    family=node(root,'PackageFamily',ID='SemantierAgent',ManifestPath=manifest,
                FlatBundle='true',ResourceManager='false',FileName='SemantierAgent.msixbundle')
    for group in plan['groups']:
        if group['id']=='main':
            package=node(family,'Package',ID='main',ProcessorArchitecture='x64',FileName='SemantierAgent-main.msix')
        else:
            package=node(family,'AssetPackage',ID=group['id'],AllowExecution='true',FileName='SemantierAgent-'+group['id']+'.msix')
        container=node(package,'Files')
        for name in group['files']:
            node(container,'File',SourcePath=ntpath.join(source,*name.split('/')),DestinationPath=name.replace('/',chr(92)))
    with Path(output).open('xb') as stream:ET.ElementTree(root).write(stream,encoding='utf-8',xml_declaration=True)



def prepare_native_asset_layout(package, directory, output, *, expected_sha256, previous=None):
    """Verify unpacked bytes against the pinned MSIX block map before SDK build."""
    import base64
    import math
    import zipfile
    import xml.etree.ElementTree as ET
    from urllib.parse import unquote
    from hermes_cli.release_channels import require_sha256
    require_sha256(expected_sha256)
    package,directory,output=Path(package),Path(directory).resolve(),Path(output)
    with package.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=expected_sha256:
            raise ValueError('Native input package digest mismatch')
    with zipfile.ZipFile(package) as archive:
        inventory=[dict(path=f.filename,bytes=f.file_size,compressed_bytes=f.compress_size)
                   for f in archive.infolist() if not f.is_dir()]
        manifest=ET.fromstring(archive.read('AppxManifest.xml'))
        identity=manifest.find('{*}Identity')
        if identity is None or identity.get('Name')!=IDENTITY['msixAppIdWithOrg'] or identity.get('Publisher')!='CN=Semantier Technology, O=Semantier Technology' or identity.get('ProcessorArchitecture')!='x64':
            raise ValueError('Native input is not the bound Semantier x64 package')
        required={unquote(n.get('Executable').replace(chr(92),'/')) for n in manifest.iter() if n.get('Executable')}
        old=None if previous is None else json.loads(Path(previous).read_text(encoding='utf-8-sig'))['assignment']
        plan=partition_native_assets(inventory,previous=old,main_required=required)
        blockmap=ET.fromstring(archive.read('AppxBlockMap.xml'))
        if blockmap.get('HashMethod')!='http://www.w3.org/2001/04/xmlenc#sha256':
            raise ValueError('Unsupported native input block-map digest')
        blocks={}
        for node in blockmap:
            name=node.get('Name','').replace(chr(92),'/')
            if name in blocks:raise ValueError('Ambiguous native input block map')
            blocks[name]=node
        members={unquote(f.filename):f for f in archive.infolist() if not f.is_dir()}
        files=[]
        # Include the manifest so the layout cannot point at a changed identity.
        for name in sorted([*plan['assignment'],'AppxManifest.xml']):
            file=directory.joinpath(*name.split('/'))
            if file.is_symlink() or not file.resolve().is_relative_to(directory) or not file.is_file():
                raise ValueError(f'Native payload location mismatch: {name}')
            entry=members.get(name);node=blocks.get(name)
            if entry is None or node is None or file.stat().st_size!=entry.file_size or int(node.get('Size','-1'))!=entry.file_size or len(node)!=math.ceil(entry.file_size/65536):
                raise ValueError(f'Native payload inventory mismatch: {name}')
            digest=hashlib.sha256()
            with file.open('rb') as stream:
                for block in node:
                    body=stream.read(65536)
                    if hashlib.sha256(body).digest()!=base64.b64decode(block.get('Hash',''),validate=True):
                        raise ValueError(f'Native input block mismatch: {name}')
                    digest.update(body)
                if stream.read(1):raise ValueError(f'Native payload size changed: {name}')
            files.append(dict(path=name,size=entry.file_size,sha256=digest.hexdigest()))
    plan.update(inputSha256=expected_sha256,identity=dict(identity.attrib),
                payloadInventorySha256=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                files=files)
    output.mkdir(parents=True,exist_ok=False)
    (output/'layout-plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    write_native_asset_layout(output/'layout.xml',plan,manifest=str(directory/'AppxManifest.xml'),source=str(directory))
    return plan


def _inspect_native_bundle(file,identity,members):
    import zipfile
    import xml.etree.ElementTree as ET
    name=identity['Name'];publisher=identity['Publisher'];version=identity['Version']
    with zipfile.ZipFile(file) as z:
        if any(n.lower().endswith(('.msix','.appx')) for n in z.namelist()):raise ValueError('Native bundle must use external package references')
        if 'AppxSignature.p7x' not in z.namelist():raise ValueError('Native bundle is unsigned')
        manifest=ET.fromstring(z.read('AppxMetadata/AppxBundleManifest.xml'));ident=manifest.find('{*}Identity')
        if ident is None or any(ident.get(k)!=v for k,v in {'Name':name,'Publisher':publisher,'Version':version}.items()):raise ValueError('Native bundle identity discontinuity')
        if any(node.tag.rsplit('}',1)[-1] not in ('Identity','Packages') for node in manifest):
            raise ValueError('Native bundle introduces unbound related objects')
        refs=manifest.findall('{*}Packages/{*}Package')
        if any(node.tag.rsplit('}',1)[-1]!='TargetDeviceFamily' for ref in refs for node in ref.findall('{*}Dependencies/*')):
            raise ValueError('Native bundle introduces an unbound package dependency')
        if len(refs)!=len(members) or {n.get('FileName') for n in refs}!=set(members):raise ValueError('Native bundle graph incomplete')
        for n in refs:
            group=members[n.get('FileName')]
            if n.get('Version')!=version or (group=='main' and (n.get('Type')!='application' or n.get('Architecture')!='x64')) or (group!='main' and (n.get('Type')!='resource' or n.get('ResourceId')!=group or any(r.attrib for r in n.findall('{*}Resources/{*}Resource')))):raise ValueError('Native bundle member mismatch')


def inspect_native_asset_graph(directory, plan, digests, *, source_commit):
    """Bind final native objects and full payload; caller verifies Windows signatures."""
    import zipfile
    import xml.etree.ElementTree as ET
    from urllib.parse import unquote
    directory=Path(directory)
    identity=plan['identity'];version=identity['Version'];publisher=identity['Publisher'];name=identity['Name']
    expected_files={r['path']:r for r in plan['files'] if r['path']!='AppxManifest.xml'}
    seen=set();objects=[];build_id=None
    members={f"SemantierAgent-{g['id']}.msix":g['id'] for g in plan['groups']}
    if set(digests)!=set(members)|{'SemantierAgent.msixbundle'}:raise ValueError('Native digest set differs from complete graph')
    def object_row(filename,kind,arch,resource=None):
        file=directory/filename
        if file.is_symlink() or not file.is_file():raise ValueError('Native object missing or linked')
        with file.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        row=dict(key=filename,kind=kind,identity=name,publisher=publisher,version=version,arch=arch,
                 size=file.stat().st_size,sha256=sha,nativeDigest=digests[filename],references=[])
        if resource:row.update(resourceId=resource,allowExecution=True)
        return row
    for filename,group in members.items():
        row=object_row(filename,'main-package' if group=='main' else 'asset-package','x64' if group=='main' else 'neutral',None if group=='main' else group)
        with zipfile.ZipFile(directory/filename) as z:
            names=[unquote(n) for n in z.namelist() if not n.endswith('/')]
            if len(names)!=len(set(names)) or 'AppxSignature.p7x' not in names:raise ValueError('Native package unsigned or ambiguous')
            manifest=ET.fromstring(z.read('AppxManifest.xml'));ident=manifest.find('{*}Identity')
            if any(node.tag.rsplit('}',1)[-1]!='TargetDeviceFamily' for node in manifest.findall('{*}Dependencies/*')):
                raise ValueError('Native graph introduces an unbound package dependency')
            if ident is None or any(ident.get(k)!=v for k,v in {'Name':name,'Publisher':publisher,'Version':version}.items()):raise ValueError('Native member identity discontinuity')
            if group=='main':
                if ident.get('ProcessorArchitecture')!='x64' or ident.get('ResourceId'):raise ValueError('Native main architecture mismatch')
            else:
                properties=manifest.find('{*}Properties')
                if ident.get('ResourceId')!=group or ident.get('ProcessorArchitecture','neutral')!='neutral' or properties is None or properties.findtext('{*}ResourcePackage')!='true' or properties.findtext('{*}AllowExecution','true')!='true':raise ValueError('Native asset execution/resource mismatch')
            for entry in z.infolist():
                path=unquote(entry.filename)
                if entry.is_dir() or path in {'AppxManifest.xml','AppxBlockMap.xml','AppxSignature.p7x','[Content_Types].xml'} or path.startswith('AppxMetadata/'):continue
                expected=expected_files.get(path)
                if expected is None or path in seen or plan['assignment'].get(path)!=group or entry.file_size!=expected['size']:raise ValueError('Native payload graph mismatch')
                with z.open(entry) as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
                if sha!=expected['sha256']:raise ValueError('Native payload content mismatch')
                seen.add(path)
                if path=='app/resources/install-stamp.json':
                    stamp=json.loads(z.read(entry));build_id=stamp['channelBuild']['buildId']
                    if stamp['commit']!=source_commit:raise ValueError('Native payload source mismatch')
        objects.append(row)
    if seen!=set(expected_files) or build_id is None:raise ValueError('Native payload incomplete or unstamped')
    filename='SemantierAgent.msixbundle';root=object_row(filename,'main-bundle','neutral')
    _inspect_native_bundle(directory/filename,identity,members)
    prefix='releases/channel-builds/'+build_id+'/'
    root['references']=[prefix+filename for filename in members]
    objects.insert(0,root)
    for row in objects:row['key']=prefix+row['key']
    graph=dict(schema=1,transport='windows-package-uri',root=root['key'],payloadInventorySha256=plan['payloadInventorySha256'],objects=objects)
    return validate_native_graph(graph,build_id=build_id,identity=name,publisher=publisher,version=version,arch='x64')


def native_graph_main():
    p=argparse.ArgumentParser(description='Inspect complete final-signed native graph; Windows signer owns signature verification')
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--digests',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(sys.argv[2:])
    graph=inspect_native_asset_graph(a.directory,json.loads(a.plan.read_text(encoding='utf-8-sig')),json.loads(a.digests.read_text(encoding='utf-8-sig')),source_commit=a.commit)
    with a.output.open('x',encoding='utf-8') as stream:json.dump(graph,stream,indent=2);stream.write('\n')


def native_layout_main():
    p=argparse.ArgumentParser(description='Prepare a byte-bound native asset layout; no signing, deployment or publication')
    p.add_argument('--package',type=Path,required=True);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--previous',type=Path)
    a=p.parse_args(sys.argv[2:])
    plan=prepare_native_asset_layout(a.package,a.directory,a.output,expected_sha256=a.sha256,previous=a.previous)
    print(json.dumps(dict(output=str(a.output),groups=len(plan['groups']),payloadInventorySha256=plan['payloadInventorySha256'],qualification='NOT_RUN')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--controller',required=True);p.add_argument('--version',required=True);p.add_argument('--base',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import tomllib
    source_version=tomllib.loads((a.source/'pyproject.toml').read_text(encoding='utf-8-sig'))['project']['version']
    value=acceptance_request(commit=a.commit,controller=a.controller,version=a.version,source_version=source_version,base=a.base)
    if a.output.exists():raise ValueError('preserve existing channel request')
    a.output.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    if sys.argv[1:2]==['native-layout']:native_layout_main()
    elif sys.argv[1:2]==['native-graph']:native_graph_main()
    else:main()
