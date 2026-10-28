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
        if kind not in ('main-package','optional-package','main-bundle','optional-bundle'):
            raise ValueError('Invalid native object kind')
        extension='.msixbundle' if kind.endswith('bundle') else '.msix'
        if not key.endswith(extension):raise ValueError('Native object extension mismatch')
        name=row.get('identity')
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9.-]{3,50}',name):raise ValueError('Invalid native identity')
        if (name,kind) in identities:raise ValueError('Ambiguous native package identity')
        identities.add((name,kind))
        if kind.startswith('main-') and name!=identity:raise ValueError('Main native identity mismatch')
        if kind.startswith('optional-') and (name==identity or not name.startswith(identity+'.')):
            raise ValueError('Related native identity mismatch')
        if any(row.get(k)!=v for k,v in dict(publisher=publisher,version=version,arch=arch).items()):
            raise ValueError('Native identity/version/architecture mismatch')
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


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--controller',required=True);p.add_argument('--version',required=True);p.add_argument('--base',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import tomllib
    source_version=tomllib.loads((a.source/'pyproject.toml').read_text(encoding='utf-8-sig'))['project']['version']
    value=acceptance_request(commit=a.commit,controller=a.controller,version=a.version,source_version=source_version,base=a.base)
    if a.output.exists():raise ValueError('preserve existing channel request')
    a.output.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
