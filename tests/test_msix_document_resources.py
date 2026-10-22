"""Exercise the production resource copy, not an independent glob approximation."""
import os
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MsixDocumentResourcesTests(unittest.TestCase):
    def test_only_windows_inert_template_source_is_omitted(self):
        node = shutil.which("node")
        matcher = Path(os.environ.get("HERMES_TEST_APP_BUILDER_LIB", ROOT / "node_modules/app-builder-lib/dist/fileMatcher.js"))
        if not node or not matcher.is_file():
            self.skipTest("Node and the locked electron-builder test dependency are required")
        result = subprocess.run([node, str(ROOT / "tests/fixtures/msix-document-resources.mjs"),
                                 str(ROOT), str(matcher)], capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Real electron-builder resource copy:", result.stdout)
