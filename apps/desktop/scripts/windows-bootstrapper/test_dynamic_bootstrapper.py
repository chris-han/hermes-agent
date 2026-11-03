"""Exercise one compiled executable per channel against changing metadata snapshots.

Fixtures contain no installable packages; this does not qualify signature or installation.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
PS = Path('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')
ROOT = Path('/mnt/d/repo/deliverables/work')
CHANNEL1_ROOT = Path('/mnt/d/repo/hermes-build/.cache/channel1-release/work')
AVAILABLE = PS.is_file() and ROOT.is_dir() and Path('/usr/bin/wslpath').is_file()

def win(path):
    return subprocess.check_output(['/usr/bin/wslpath', '-w', str(path)], text=True).strip()

def fixture(build, version, *, key, origin, publisher='CN=Semantier Technology, O=Semantier Technology', object_build=None):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from scripts.releases.simple_native_slots import add_installation_proof
    root=f'releases/channel-builds/{build}/SemantierAgent.msixbundle'
    obj=f'releases/channel-builds/{object_build or build}/SemantierAgent.msixbundle'
    artifact={'key':root,'size':10,'sha256':'b'*64}
    manifest={'schema':2,'request':{'buildId':build,'channel':'semantier-acceptance','sequence':1,'windowsVersion':version,'publicBase':origin,
               'identity':{'msixAppIdWithOrg':'Semantier.Agent.Acceptance'}},
              'packages':[{'platform':'win32','arch':'x64','variant':'native-msix-assets','identity':'Semantier.Agent.Acceptance',
                           'publisher':publisher,'version':version,'artifact':artifact}],
              'installation':{'schema':1,'root':root,'objects':[{'key':obj,'size':10,'sha256':'b'*64}]}}
    add_installation_proof(manifest,lambda payload: key.sign(payload,padding.PKCS1v15(),hashes.SHA256()))
    raw=json.dumps(manifest,separators=(',',':')).encode()
    record={'schema':1,'name':'semantier-acceptance','state':'active','revision':1,'nativeHead':{'buildId':build,
            'sequence':1,'manifestKey':f'releases/channel-builds/{build}/build.json','sha256':hashlib.sha256(raw).hexdigest()}}
    return json.dumps(record,separators=(',',':')).encode(),raw

def legacy_fixture(build,version,*,key,origin):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from scripts.releases.simple_native_slots import legacy_installation_projection
    prefix=f'releases/channel-builds/{build}/';root=prefix+'SemantierAgent.msixbundle';main=prefix+'SemantierAgent.msix'
    publisher='CN=Semantier Technology, O=Semantier Technology';identity='Semantier.Agent.Acceptance'
    common={'publisher':publisher,'version':version,'size':10,'sha256':'b'*64,'nativeDigest':base64.b64encode(b'd'*32).decode()}
    graph={'schema':1,'transport':'windows-package-uri','payloadInventorySha256':'c'*64,'root':root,'objects':[
        {**common,'key':root,'kind':'main-bundle','identity':identity,'arch':'neutral','references':[main]},
        {**common,'key':main,'kind':'main-package','identity':identity,'arch':'x64','references':[]}]}
    graph_bytes=json.dumps(graph,separators=(',',':')).encode()
    descriptor={'key':prefix+'native-graph.json','size':len(graph_bytes),'sha256':hashlib.sha256(graph_bytes).hexdigest()}
    manifest={'schema':2,'request':{'buildId':build,'channel':'semantier-acceptance','sequence':1,'windowsVersion':version,
        'publicBase':'https://s3.semantier.com','identity':{'msixAppIdWithOrg':identity}},
        'packages':[{'platform':'win32','arch':'x64','variant':'native-msix-assets','identity':identity,'publisher':publisher,
            'version':version,'artifact':{'key':root,'size':10,'sha256':'b'*64},'native':{'graph':descriptor}}]}
    raw=json.dumps(manifest,separators=(',',':')).encode()
    head={'buildId':build,'sequence':1,'manifestKey':prefix+'build.json','sha256':hashlib.sha256(raw).hexdigest()}
    record={'schema':1,'name':'semantier-acceptance','state':'active','revision':1,'nativeHead':head}
    projection=legacy_installation_projection(raw,graph_bytes,head,origin,lambda _:None,
        lambda payload:key.sign(payload,padding.PKCS1v15(),hashes.SHA256()))
    return json.dumps(record,separators=(',',':')).encode(),raw,json.dumps(projection,separators=(',',':')).encode()

@unittest.skipUnless(AVAILABLE, 'Requires Windows build host')
class DynamicBootstrapperTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('SEMANTIER_SIGNED_SETUP_DIR'), 'Requires explicit signed setup candidate directory')
    def test_signed_candidates_follow_simulated_active_and_rollback(self):
        from scripts.releases.simple_native_slots import sign_windows_installation_proof

        class ExistingSigner:
            def sign(self, payload, *_args):
                return sign_windows_installation_proof(payload, '57BF15739B4AA447842EAF388856A01B4E5029FF')

        signed = Path(os.environ['SEMANTIER_SIGNED_SETUP_DIR'])
        receipt = {'kind': 'synthetic-active-selection-on-signed-candidates-v2', 'cases': []}
        with tempfile.TemporaryDirectory(prefix='signed-active-fixture-', dir=ROOT) as tmp:
            work = Path(tmp)
            for channel, origin in (('channel1', 'https://download.semantier.cn'),
                                    ('channel2', 'https://download.semantier.com')):
                exe = signed / channel / ('semantier-setup-ch1.exe' if channel == 'channel1' else 'semantier-setup-ch2.exe')
                self.assertTrue(exe.is_file())
                exe_hash = hashlib.sha256(exe.read_bytes()).hexdigest()
                a, ma = fixture('a'*32, '0.0.79.0', key=ExistingSigner(), origin=origin)
                b, mb = fixture('b'*32, '0.0.80.0', key=ExistingSigner(), origin=origin)
                files = [work/'before.json', work/'manifest.json', work/'after.json']
                def inspect(before, manifest, after):
                    for path, content in zip(files, (before, manifest, after)):
                        path.write_bytes(content)
                    return subprocess.run([str(exe), '--inspect-metadata', *(win(path) for path in files)],
                                          capture_output=True, text=True, timeout=30)
                for label, record, manifest, version in (('active-79', a, ma, '0.0.79.0'),
                                                          ('active-80', b, mb, '0.0.80.0'),
                                                          ('rollback-79', a, ma, '0.0.79.0')):
                    result = inspect(record, manifest, record)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(version, result.stdout)
                    receipt['cases'].append({'channel': channel, 'case': label, 'version': version,
                                             'exeSha256': exe_hash, 'headSha256': hashlib.sha256(record).hexdigest(),
                                             'manifestSha256': hashlib.sha256(manifest).hexdigest()})
                bad_digest = json.loads(a)
                bad_digest['nativeHead']['sha256'] = '0'*64
                bad_digest = json.dumps(bad_digest, separators=(',', ':')).encode()
                altered = json.loads(ma)
                altered['installationProof']['signature'] = base64.b64encode(bytes(256)).decode()
                altered = json.dumps(altered, separators=(',', ':')).encode()
                altered_head = json.loads(a)
                altered_head['nativeHead']['sha256'] = hashlib.sha256(altered).hexdigest()
                altered_head = json.dumps(altered_head, separators=(',', ':')).encode()
                foreign = 'https://download.semantier.com' if channel == 'channel1' else 'https://download.semantier.cn'
                foreign_head, foreign_manifest = fixture('a'*32, '0.0.79.0', key=ExistingSigner(), origin=foreign)
                wrong_head, wrong_manifest = fixture('a'*32, '0.0.79.0', key=ExistingSigner(), origin=origin, publisher='CN=Other')
                mixed_head, mixed_manifest = fixture('a'*32, '0.0.79.0', key=ExistingSigner(), origin=origin, object_build='b'*32)
                absent = json.loads(a)
                del absent['nativeHead']
                absent = json.dumps(absent, separators=(',', ':')).encode()
                for label, before, manifest, after in (
                    ('invalid-digest', bad_digest, ma, bad_digest),
                    ('invalid-signature', altered_head, altered, altered_head),
                    ('wrong-origin', foreign_head, foreign_manifest, foreign_head),
                    ('wrong-publisher', wrong_head, wrong_manifest, wrong_head),
                    ('mixed-release', mixed_head, mixed_manifest, mixed_head),
                    ('conflicting-head', a, ma, b),
                    ('missing-active', absent, ma, absent),
                ):
                    result = inspect(before, manifest, after)
                    self.assertNotEqual(result.returncode, 0, label)
                    receipt['cases'].append({'channel': channel, 'case': label, 'result': 'REFUSED',
                                             'exeSha256': exe_hash, 'headSha256': hashlib.sha256(before).hexdigest(),
                                             'manifestSha256': hashlib.sha256(manifest).hexdigest(),
                                             'afterSha256': hashlib.sha256(after).hexdigest()})
                self.assertEqual(hashlib.sha256(exe.read_bytes()).hexdigest(), exe_hash)
        print('SIGNED_ACTIVE_FIXTURE_RECEIPT=' + json.dumps(receipt, sort_keys=True))

    def test_build_refuses_output_outside_owned_work_root(self):
        for channel, output in (('channel1', r'C:\semantier-0080-invalid-output'),
                                ('channel1', r'D:\repo\deliverables\work\wrong-channel'),
                                ('channel2', r'D:\repo\hermes-build\.cache\channel1-release\work\wrong-channel')):
            result=subprocess.run([str(PS),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',win(HERE/'Build.ps1'),
                '-Channel',channel,'-OutputDirectory',output],capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('fresh task-owned directory',result.stderr)

    def test_existing_distribution_lock_is_not_removed_or_bypassed(self):
        CHANNEL1_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='setup-lock-0080-',dir=CHANNEL1_ROOT) as tmp:
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
        from datetime import datetime, timedelta, timezone
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
        CHANNEL1_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='dynamic-setup-0080-',dir=CHANNEL1_ROOT) as tmp, tempfile.TemporaryDirectory(prefix='dynamic-setup-0080-',dir=ROOT) as tmp2:
            base=Path(tmp)
            key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
            name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'Disposable installer fixture')])
            now=datetime.now(timezone.utc)
            cert=(x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
                .serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(minutes=1))
                .not_valid_after(now+timedelta(days=1)).sign(key,hashes.SHA256()))
            der=cert.public_bytes(serialization.Encoding.DER)
            source=base/'source/apps/desktop/scripts/windows-bootstrapper'
            shutil.copytree(HERE,source,ignore=shutil.ignore_patterns('__pycache__'))
            assets=base/'source/apps/desktop/assets';assets.mkdir(parents=True)
            for name in ('icon.png','icon.ico'): shutil.copy2(HERE/'../../assets'/name,assets/name)
            (source/'SemantierAgent-signing.cer').write_bytes(der)
            old='10A6B55E9A03BBC0DCEB8D899C1776BF1FBAD963C82050C16534DA4983C9661D'
            for name in ('Build.ps1','ActiveRelease.cs'):
                path=source/name; content=path.read_text();self.assertIn(old,content)
                path.write_text(content.replace(old,hashlib.sha256(der).hexdigest().upper()))
            path=source/'Installer.cs';content=path.read_text()
            old_thumb='57BF15739B4AA447842EAF388856A01B4E5029FF'
            self.assertIn(old.lower(),content);self.assertIn(old_thumb,content)
            path.write_text(content.replace(old.lower(),hashlib.sha256(der).hexdigest()).replace(old_thumb,hashlib.sha1(der).hexdigest().upper()))
            for channel in ('channel1','channel2'):
                origin='https://download.semantier.cn' if channel=='channel1' else 'https://download.semantier.com'
                output=(base if channel == 'channel1' else Path(tmp2))/channel
                build=subprocess.run([str(PS),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',win(source/'Build.ps1'),
                    '-Channel',channel,'-OutputDirectory',win(output)],capture_output=True,text=True,timeout=300)
                self.assertEqual(build.returncode,0,build.stdout+build.stderr)
                exe=output/('semantier-setup-ch1.exe' if channel=='channel1' else 'semantier-setup-ch2.exe')
                self.assertTrue(exe.is_file())
                def inspect(before,manifest,after,projection=None):
                    contents=[before,manifest]+([projection] if projection is not None else [])+[after]
                    files=[base/f'input-{i}.json' for i in range(len(contents))]
                    for path,data in zip(files,contents): path.write_bytes(data)
                    return subprocess.run([str(exe),'--inspect-metadata',*(win(path) for path in files)],capture_output=True,text=True,timeout=30)
                a,ma=fixture('a'*32,'0.0.79.0',key=key,origin=origin); b,mb=fixture('b'*32,'0.0.80.0',key=key,origin=origin)
                for record,manifest,expected in ((a,ma,'0.0.79.0'),(b,mb,'0.0.80.0'),(a,ma,'0.0.79.0')):
                    selected=inspect(record,manifest,record)
                    self.assertEqual(selected.returncode,0,selected.stderr)
                    self.assertIn(expected,selected.stdout)
                old,old_manifest,old_projection=legacy_fixture('c'*32,'0.0.79.0',key=key,origin=origin)
                for record,manifest,extra,expected in ((old,old_manifest,old_projection,'0.0.79.0'),(b,mb,None,'0.0.80.0'),(old,old_manifest,old_projection,'0.0.79.0')):
                    selected=inspect(record,manifest,record,extra)
                    self.assertEqual(selected.returncode,0,selected.stderr)
                    self.assertIn(expected,selected.stdout)
                self.assertNotEqual(inspect(old,old_manifest,old).returncode,0)
                self.assertNotEqual(inspect(old,old_manifest,b,old_projection).returncode,0)
                foreign_origin='https://download.semantier.cn' if channel=='channel2' else 'https://download.semantier.com'
                _,_,foreign_projection=legacy_fixture('c'*32,'0.0.79.0',key=key,origin=foreign_origin)
                self.assertNotEqual(inspect(old,old_manifest,old,foreign_projection).returncode,0)
                broken=json.loads(old_projection);broken['installation']['objects'][0]['sha256']='0'*64
                self.assertNotEqual(inspect(old,old_manifest,old,json.dumps(broken).encode()).returncode,0)
                broken=json.loads(old_projection);broken['installationProof']['signature']=base64.b64encode(bytes(256)).decode()
                self.assertNotEqual(inspect(old,old_manifest,old,json.dumps(broken).encode()).returncode,0)
                altered=json.loads(old_manifest);altered['request']['publicBase']='https://other.example'
                altered=json.dumps(altered,separators=(',',':')).encode();changed=json.loads(old);changed['nativeHead']['sha256']=hashlib.sha256(altered).hexdigest()
                changed=json.dumps(changed,separators=(',',':')).encode()
                self.assertNotEqual(inspect(changed,altered,changed,old_projection).returncode,0)
                self.assertNotEqual(inspect(a,ma,b).returncode,0)
                self.assertNotEqual(inspect(a,mb,a).returncode,0)
                bad,wrong=fixture('a'*32,'0.0.79.0',key=key,origin=origin,publisher='CN=Other')
                self.assertNotEqual(inspect(bad,wrong,bad).returncode,0)
                mixed,wrong=fixture('a'*32,'0.0.79.0',key=key,origin=origin,object_build='b'*32)
                self.assertNotEqual(inspect(mixed,wrong,mixed).returncode,0)
                wrong_origin,wrong=fixture('a'*32,'0.0.79.0',key=key,origin='https://download.semantier.cn' if channel=='channel2' else 'https://download.semantier.com')
                self.assertNotEqual(inspect(wrong_origin,wrong,wrong_origin).returncode,0)
                tampered=json.loads(ma);tampered['installationProof']['signature']=base64.b64encode(bytes(256)).decode()
                raw=json.dumps(tampered,separators=(',',':')).encode(); record=json.loads(a);record['nativeHead']['sha256']=hashlib.sha256(raw).hexdigest()
                record=json.dumps(record,separators=(',',':')).encode()
                self.assertNotEqual(inspect(record,raw,record).returncode,0)
                tampered=json.loads(ma);tampered['installation']['objects'][0]['size']=11
                raw=json.dumps(tampered,separators=(',',':')).encode();record=json.loads(a);record['nativeHead']['sha256']=hashlib.sha256(raw).hexdigest()
                record=json.dumps(record,separators=(',',':')).encode()
                self.assertNotEqual(inspect(record,raw,record).returncode,0)
                unsigned=json.loads(ma);del unsigned['installationProof']
                raw=json.dumps(unsigned,separators=(',',':')).encode();record=json.loads(a);record['nativeHead']['sha256']=hashlib.sha256(raw).hexdigest()
                record=json.dumps(record,separators=(',',':')).encode()
                self.assertNotEqual(inspect(record,raw,record).returncode,0)
