"""Exercise one compiled executable per channel against changing metadata snapshots.

Fixtures contain no installable packages; this does not qualify signature or installation.
"""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
PS = Path('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')
ROOT = Path('/mnt/d/repo/deliverables/work')
AVAILABLE = PS.is_file() and ROOT.is_dir() and shutil.which('wslpath')

def win(path):
    return subprocess.check_output(['wslpath', '-w', str(path)], text=True).strip()

def fixture(build, version, *, publisher='CN=Semantier Technology, O=Semantier Technology', object_build=None):
    root=f'releases/channel-builds/{build}/SemantierAgent.msixbundle'
    obj=f'releases/channel-builds/{object_build or build}/SemantierAgent.msixbundle'
    artifact={'key':root,'size':10,'sha256':'b'*64}
    manifest={'schema':2,'request':{'buildId':build,'channel':'semantier-acceptance','sequence':1,'windowsVersion':version,
               'identity':{'msixAppIdWithOrg':'Semantier.Agent.Acceptance'}},
              'packages':[{'platform':'win32','arch':'x64','variant':'native-msix-assets','identity':'Semantier.Agent.Acceptance',
                           'publisher':publisher,'version':version,'artifact':artifact}],
              'installation':{'schema':1,'root':root,'objects':[{'key':obj,'size':10,'sha256':'b'*64}]}}
    raw=json.dumps(manifest,separators=(',',':')).encode()
    record={'schema':1,'name':'semantier-acceptance','state':'active','revision':1,'nativeHead':{'buildId':build,
            'sequence':1,'manifestKey':f'releases/channel-builds/{build}/build.json','sha256':hashlib.sha256(raw).hexdigest()}}
    return json.dumps(record,separators=(',',':')).encode(),raw

@unittest.skipUnless(AVAILABLE, 'Requires Windows build host')
class DynamicBootstrapperTest(unittest.TestCase):
    def test_build_refuses_output_outside_owned_work_root(self):
        result=subprocess.run([str(PS),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',win(HERE/'Build.ps1'),
            '-Channel','channel1','-OutputDirectory',r'C:\semantier-0080-invalid-output'],capture_output=True,text=True,timeout=30)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('fresh task-owned directory',result.stderr)

    def test_existing_distribution_lock_is_not_removed_or_bypassed(self):
        with tempfile.TemporaryDirectory(prefix='setup-lock-0080-',dir=ROOT) as tmp:
            script=win(HERE/'Build.ps1').replace("'","''");out=win(Path(tmp)/'output').replace("'","''")
            command=r"""
$ErrorActionPreference='Stop'
$path='D:\repo\hermes-build.distribution.lock'
$hold=[IO.FileStream]::new($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None,4096,[IO.FileOptions]::DeleteOnClose)
try {
 $failed=$false
 try { & 'SCRIPT' -Channel channel1 -OutputDirectory 'OUTPUT' } catch { $failed=$true }
 if(!$failed -or !(Test-Path -LiteralPath $path) -or (Test-Path -LiteralPath 'OUTPUT')) { throw 'lock was bypassed or removed' }
 'LOCK_PRESERVED'
} finally { $hold.Dispose() }
""".replace('SCRIPT',script).replace('OUTPUT',out)
            result=subprocess.run([str(PS),'-NoProfile','-NonInteractive','-EncodedCommand',base64.b64encode(command.encode('utf-16le')).decode()],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('LOCK_PRESERVED',result.stdout)

    def test_same_compiled_channel_binaries_follow_active_and_reject_invalid_metadata(self):
        with tempfile.TemporaryDirectory(prefix='dynamic-setup-0080-',dir=ROOT) as tmp:
            base=Path(tmp)
            for channel in ('channel1','channel2'):
                output=base/channel
                build=subprocess.run([str(PS),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',win(HERE/'Build.ps1'),
                    '-Channel',channel,'-OutputDirectory',win(output)],capture_output=True,text=True,timeout=300)
                self.assertEqual(build.returncode,0,build.stdout+build.stderr)
                exe=output/('semantier-setup-ch1.exe' if channel=='channel1' else 'semantier-setup-ch2.exe')
                self.assertTrue(exe.is_file())
                def inspect(before,manifest,after):
                    files=[base/'before.json',base/'manifest.json',base/'after.json']
                    for path,data in zip(files,(before,manifest,after)): path.write_bytes(data)
                    return subprocess.run([str(exe),'--inspect-metadata',*(win(path) for path in files)],capture_output=True,text=True,timeout=30)
                a,ma=fixture('a'*32,'0.0.79.0'); b,mb=fixture('b'*32,'0.0.80.0')
                for record,manifest,expected in ((a,ma,'0.0.79.0'),(b,mb,'0.0.80.0'),(a,ma,'0.0.79.0')):
                    selected=inspect(record,manifest,record)
                    self.assertEqual(selected.returncode,0,selected.stderr)
                    self.assertIn(expected,selected.stdout)
                self.assertNotEqual(inspect(a,ma,b).returncode,0)
                self.assertNotEqual(inspect(a,mb,a).returncode,0)
                bad,wrong=fixture('a'*32,'0.0.79.0',publisher='CN=Other')
                self.assertNotEqual(inspect(bad,wrong,bad).returncode,0)
                mixed,wrong=fixture('a'*32,'0.0.79.0',object_build='b'*32)
                self.assertNotEqual(inspect(mixed,wrong,mixed).returncode,0)
