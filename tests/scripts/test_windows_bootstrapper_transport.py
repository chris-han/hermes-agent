"""Run the bootstrapper transport's controlled tests on the real Windows runtime."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest

SOURCE = Path(__file__).resolve().parents[2] / 'apps/desktop/scripts/windows-bootstrapper'
WORK = Path('/mnt/d/repo/deliverables/work')
DOTNET = Path('/mnt/c/Program Files/dotnet/dotnet.exe')


@pytest.mark.skipif(not WORK.is_dir() or not DOTNET.is_file() or not shutil.which('wslpath'),
                    reason='Windows build host and D: work root required')
def test_windows_http3_fallback_and_range_resume():
    with tempfile.TemporaryDirectory(prefix='bootstrapper-transport-test-', dir=WORK) as tmp:
        root = Path(tmp)
        shutil.copy2(SOURCE / 'BootstrapperTransport.cs', root / 'BootstrapperTransport.cs')
        shutil.copy2(SOURCE / 'transport-tests/Probe.cs', root / 'Probe.cs')
        (root / 'Probe.csproj').write_text(
            '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>net8.0</TargetFramework>'
            '<OutputType>Exe</OutputType></PropertyGroup></Project>', encoding='utf-8')
        project = subprocess.check_output(['wslpath', '-w', str(root / 'Probe.csproj')], text=True).strip()
        env = os.environ.copy()
        values = {'DOTNET_CLI_HOME': r'D:\repo\deliverables\cache\dotnet-cli',
                  'NUGET_PACKAGES': r'D:\repo\deliverables\cache\nuget-packages',
                  'TEMP': r'D:\repo\deliverables\work', 'TMP': r'D:\repo\deliverables\work',
                  'DOTNET_CLI_TELEMETRY_OPTOUT': '1', 'DOTNET_GENERATE_ASPNET_CERTIFICATE': 'false'}
        env.update(values)
        env['WSLENV'] = ':'.join(filter(None, [env.get('WSLENV', ''), *values]))
        result = subprocess.run([str(DOTNET), 'run', '--project', project], env=env,
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'TRANSPORT_POLICY_TEST_PASS' in result.stdout
