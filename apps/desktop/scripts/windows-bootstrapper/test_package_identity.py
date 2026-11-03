"""Windows test of the production sparse-bundle identity/closure validator."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from zipfile import ZipFile
import pytest

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/d/repo/deliverables/work')
DOTNET=Path('/mnt/c/Program Files/dotnet/dotnet.exe')

def win(path): return subprocess.check_output(['/usr/bin/wslpath','-w',str(path)],text=True).strip()

@pytest.mark.skipif(not DOTNET.is_file() or not ROOT.is_dir() or not Path('/usr/bin/wslpath').is_file(),reason='requires Windows dotnet')
def test_bundle_closure_rejects_missing_or_wrong_version():
    with tempfile.TemporaryDirectory(prefix='package-identity-0080-',dir=ROOT) as tmp:
        root=Path(tmp); project=root/'project';data=root/'data';project.mkdir();data.mkdir()
        for source in (HERE/'ActiveRelease.cs',HERE/'PackageIdentity.cs',HERE/'resolver-tests/PackageProbe.cs',HERE/'resolver-tests/PackageProbe.csproj'):
            shutil.copy2(source,project/source.name)
        version='0.0.80.0';identity='Semantier.Agent.Acceptance';publisher='CN=Semantier Technology, O=Semantier Technology'
        package=f'<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"><Identity Name="{identity}" Publisher="{publisher}" Version="{version}"/></Package>'
        bundle=f'<Bundle xmlns="http://schemas.microsoft.com/appx/2017/bundle"><Identity Name="{identity}" Publisher="{publisher}" Version="{version}"/><Packages><Package FileName="main.msix"/></Packages></Bundle>'
        for name,entry,body in [('bundle.msixbundle','AppxMetadata/AppxBundleManifest.xml',bundle),('main.msix','AppxManifest.xml',package)]:
            with ZipFile(data/name,'w') as z:z.writestr(entry,body)
        prefix='releases/channel-builds/'+'a'*32+'/'
        objects=[]
        for name in ('bundle.msixbundle','main.msix'):
            raw=(data/name).read_bytes();objects.append({'key':prefix+name,'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'version':version})
        (data/'native-graph.json').write_text(json.dumps({'root':objects[0]['key'],'objects':objects}))
        result=subprocess.run([str(DOTNET),'run','--project',win(project/'PackageProbe.csproj'),'--',win(data)],capture_output=True,text=True,timeout=60,env={**os.environ,'DOTNET_GENERATE_ASPNET_CERTIFICATE':'false','DOTNET_CLI_TELEMETRY_OPTOUT':'1'})
        assert result.returncode==0,result.stdout+result.stderr
        assert 'PACKAGE_IDENTITY_AND_CLOSURE_PASS' in result.stdout
