"""Real retained publisher-signed bundle; no installation or trust changes."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import pytest

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/d/repo/deliverables/work')
DOTNET=Path('/mnt/c/Program Files/dotnet/dotnet.exe')
RETAINED=Path('/mnt/d/repo/deliverables/SemantierAgent-terminal-0078-v1/signed/distribution')

def win(path):return subprocess.check_output(['wslpath','-w',str(path)],text=True).strip()

@pytest.mark.skipif(not DOTNET.is_file() or not (RETAINED/'SemantierAgent.msixbundle').is_file() or not ROOT.is_dir() or not shutil.which('wslpath'),reason='requires Windows and retained signed 0.0.78 bundle')
def test_pinned_signer_passes_and_corrupted_or_wrong_publisher_fails():
    with tempfile.TemporaryDirectory(prefix='package-signature-0080-',dir=ROOT) as tmp:
        project=Path(tmp)
        for source in (HERE/'ActiveRelease.cs',HERE/'PackageSignature.cs',HERE/'resolver-tests/SignatureProbe.cs',HERE/'resolver-tests/SignatureProbe.csproj'):
            shutil.copy2(source,project/source.name)
        result=subprocess.run([str(DOTNET),'run','--project',win(project/'SignatureProbe.csproj'),'--',win(RETAINED),win(project)],
            capture_output=True,text=True,timeout=120,env={**os.environ,'DOTNET_GENERATE_ASPNET_CERTIFICATE':'false','DOTNET_CLI_TELEMETRY_OPTOUT':'1'})
        assert result.returncode==0,result.stdout+result.stderr
        assert 'SIGNATURE_CORRUPTION_REFUSED' in result.stdout
