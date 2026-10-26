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
               releaseTag='v'+version,identity=IDENTITY,bundleEnv={},publicBase=base,
               semantierScope='semantier-windows-x64/v1')
    validate_request(value)
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--controller',required=True);p.add_argument('--version',required=True);p.add_argument('--base',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import tomllib
    source_version=tomllib.loads((a.source/'pyproject.toml').read_text())['project']['version']
    value=acceptance_request(commit=a.commit,controller=a.controller,version=a.version,source_version=source_version,base=a.base)
    if a.output.exists():raise ValueError('preserve existing channel request')
    a.output.write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':main()
