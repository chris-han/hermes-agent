"""Windows-only behavioural checks of the Channel1 build admission boundary.

The sample ZIP contains synthetic bytes, never an installable package. It does
not prove a release signature, installation, or publisher qualification.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
POWERSHELL = Path('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')
WINDOWS_TMP_ROOT = Path('/mnt/d/repo/deliverables')
AVAILABLE = POWERSHELL.is_file() and WINDOWS_TMP_ROOT.is_dir() and shutil.which('wslpath')


def win(path):
    return subprocess.check_output(['wslpath', '-w', str(path)], text=True).strip()


def digest(value):
    return hashlib.sha256(value).hexdigest()


@unittest.skipUnless(AVAILABLE, 'Requires the retained Windows build host')
class Channel1BindingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='channel1-pins-test-', dir=WINDOWS_TMP_ROOT)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, *, graph_version='0.0.65.0', kind='main-bundle'):
        cert = (HERE / 'SemantierAgent-signing.cer').read_bytes()
        prefix = 'releases/channel-builds/' + 'a' * 32 + '/'
        members = {'SemantierAgent-signing.cer': cert,
                   'SemantierAgent.msixbundle': b'fixture bundle',
                   'SemantierAgent-main.msix': b'fixture main'}
        graph = {'schema': 1, 'transport': 'windows-package-uri',
                 'root': prefix + 'SemantierAgent.msixbundle',
                 'objects': [
                     {'key': prefix + name, 'kind': item_kind,
                      'identity': 'Semantier.Agent.Acceptance', 'publisher': 'CN=Semantier Technology, O=Semantier Technology',
                      'version': graph_version,
                      'sha256': digest(members[name])}
                     for name, item_kind in [('SemantierAgent.msixbundle', kind),
                                              ('SemantierAgent-main.msix', 'main-package')]]}
        members['native-graph.json'] = json.dumps(graph).encode()
        archive = self.root / 'fixture.zip'
        with ZipFile(archive, 'w', ZIP_DEFLATED) as output:
            for name, content in members.items():
                output.writestr(name, content)
        raw = archive.read_bytes()
        descriptor = {'origin': 'https://update.semantier.cn',
                      'zip_path': '/' + prefix + 'fixture.zip',
                      'zip_sha256': digest(raw), 'zip_size': len(raw),
                      'cert_sha256': digest(cert), 'graph_sha256': digest(members['native-graph.json']),
                      'thumbprint': '57BF15739B4AA447842EAF388856A01B4E5029FF',
                      'identity': 'Semantier.Agent.Acceptance', 'app_version': '0.0.65.0',
                      'members': [{'name': name, 'sha256': digest(data)} for name, data in members.items()]}
        path = self.root / 'descriptor.json'
        path.write_text(json.dumps(descriptor))
        return path, archive

    def build(self, descriptor, archive):
        return subprocess.run([str(POWERSHELL), '-NoProfile', '-NonInteractive',
                               '-ExecutionPolicy', 'Bypass', '-File', win(HERE / 'Build.ps1'),
                               '-OutputDirectory', win(self.root / 'output'),
                               '-Channel1Descriptor', win(descriptor),
                               '-Channel1Zip', win(archive)],
                              text=True, capture_output=True, timeout=120, check=False)

    def test_matching_fixture_compiles_and_runs_existing_self_tests(self):
        result = self.build(*self.fixture())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('BUILD_AND_UI_TESTS_PASS', result.stdout)
        self.assertTrue((self.root / 'output' / 'semantier-setup-channel1.exe').is_file())

    def test_same_size_tampering_is_refused_before_compile(self):
        descriptor, archive = self.fixture()
        raw = bytearray(archive.read_bytes())
        raw[len(raw) // 2] ^= 1
        archive.write_bytes(raw)
        result = self.build(descriptor, archive)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Channel1 ZIP hash mismatch', result.stderr)
        self.assertFalse((self.root / 'output' / 'semantier-setup-channel1.exe').exists())

    def test_graph_version_and_package_kind_must_match(self):
        for kwargs, message in [({'graph_version': '0.0.66.0'}, 'Channel1 graph object invalid'),
                                ({'kind': 'unrelated'}, 'Channel1 graph package kind mismatch')]:
            with self.subTest(kwargs=kwargs):
                descriptor, archive = self.fixture(**kwargs)
                result = self.build(descriptor, archive)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                shutil.rmtree(self.root / 'output')

    def test_certificate_thumbprint_must_match_embedded_certificate(self):
        descriptor, archive = self.fixture()
        obj = json.loads(descriptor.read_text())
        obj['thumbprint'] = 'A' * 40
        descriptor.write_text(json.dumps(obj))
        result = self.build(descriptor, archive)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Descriptor thumbprint mismatch', result.stderr)


if __name__ == '__main__':
    unittest.main()
