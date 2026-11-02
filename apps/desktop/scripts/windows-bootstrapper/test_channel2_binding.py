"""Windows checks for exact Channel 2 payload admission."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
POWERSHELL = Path('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')
ROOT = Path('/mnt/d/repo/deliverables')
AVAILABLE = POWERSHELL.is_file() and ROOT.is_dir() and shutil.which('wslpath')

def win(path):
    return subprocess.check_output(['wslpath', '-w', str(path)], text=True).strip()

@unittest.skipUnless(AVAILABLE, 'Requires the retained Windows build host')
class Channel2BindingTest(unittest.TestCase):
    def test_refuses_unreviewed_zip_before_creating_a_setup(self):
        with tempfile.TemporaryDirectory(prefix='channel2-negative-', dir=ROOT) as tmp:
            root = Path(tmp)
            bad_zip = root / 'bad.zip'
            bad_zip.write_bytes(b'not the reviewed release')
            result = subprocess.run([str(POWERSHELL), '-NoProfile', '-NonInteractive',
                '-ExecutionPolicy', 'Bypass', '-File', win(HERE / 'Build.ps1'),
                '-OutputDirectory', win(root / 'output'), '-Channel2Zip', win(bad_zip)],
                capture_output=True, text=True, timeout=120, check=False)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn('Channel2 ZIP mismatch', result.stderr)
            self.assertFalse((root / 'output' / 'semantier-setup-channel2.exe').exists())

if __name__ == '__main__':
    unittest.main()
