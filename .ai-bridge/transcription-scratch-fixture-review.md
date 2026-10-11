# Review-only transcription fixture repair

Status: NOT APPLIED. CodexPro rejected the source edit with its secret-content filter. The script below has not been executed. No post-repair test result is claimed.

Target: /home/chris/repo/semantier-runtime/.worktrees/ci-contract-followthrough/hermes-agent/tests/tools/test_transcription_tools.py

Verified current SHA-256: 94fdb2907e86b3aaf896c2803f8747d6d4738d305faba9c979bc0aa40a1d29fc

Before evidence: hosted Python slice 3 and a local canonical run both report 3 failed, 65 passed. Each failure is a fake TemporaryDirectory accepting only prefix, while production correctly passes dir=execution_scratch_dir().

Proposed scope: add one fixture binding the actual ExecutionBoundary to a temporary artifact root; place the three mocked output directories under that root's tmp directory; require all three temporary-directory fakes to receive exactly that scratch path. Retain all existing sanitized-environment, transcription-output, provider, and literal-argv safety assertions. No production code changes.

## Guarded manual patch

Run only if you choose to apply the reviewed source repair. It checks the exact original hash and replacement counts, compiles the candidate, makes an exclusive backup, checks for concurrent modification, then writes the target. It does not commit, push, or run tests.

```python
from pathlib import Path
import hashlib
import json

p = Path("/home/chris/repo/semantier-runtime/.worktrees/ci-contract-followthrough/hermes-agent/tests/tools/test_transcription_tools.py")
b = p.read_bytes()
assert hashlib.sha256(b).hexdigest() == "94fdb2907e86b3aaf896c2803f8747d6d4738d305faba9c979bc0aa40a1d29fc", "Target changed; stop and re-review"
s = b.decode("utf-8")
replacements = [
    ('''# Fixtures
# ============================================================================

''', '''# Fixtures
# ============================================================================

@pytest.fixture
def local_stt_scratch(tmp_path):
    from gateway import execution_boundary as boundary

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    scope = boundary.ExecutionBoundary(
        source="test", paths=boundary.BoundaryPaths(artifacts_root=artifacts),
        policy=boundary.BoundaryPolicy(allowed_write_roots=(artifacts,)),
    )
    with boundary.bind_execution_boundary(scope):
        scratch = artifacts / "tmp"
        scratch.mkdir()
        yield scratch


''', 1),
    ('''self, monkeypatch, sample_wav, tmp_path
''', '''self, monkeypatch, sample_wav, local_stt_scratch
''', 2),
    ('''def test_command_fallback_with_template(self, monkeypatch, sample_ogg, tmp_path):''', '''def test_command_fallback_with_template(self, monkeypatch, sample_ogg, local_stt_scratch):''', 1),
    ('''out_dir = tmp_path / "local-out"''', '''out_dir = local_stt_scratch / "local-out"''', 2),
    ('''output_dir = tmp_path / "transcript-output"''', '''output_dir = local_stt_scratch / "transcript-output"''', 1),
    ('''def fake_tempdir(prefix=None):''', '''def fake_tempdir(prefix=None, *, dir=None):
            assert dir == str(local_stt_scratch)''', 3),
]
for old, new, expected in replacements:
    assert s.count(old) == expected, f"Expected {expected} matches, found {s.count(old)}"
    s = s.replace(old, new)
compile(s, str(p), "exec")
backup = p.with_name(p.name + ".before-scratch-fixture-repair")
with backup.open("xb") as f:
    f.write(b)
assert p.read_bytes() == b, "Concurrent modification; stop"
p.write_bytes(s.encode("utf-8"))
print("Scratch-fixture patch applied; backup preserved. No commit or push performed.")
```

After applying, the assistant must inspect the exact diff and run:

```text
HERMES_PYTHON=/home/chris/repo/semantier-runtime/.venv/bin/python HERMES_TEST_WORKERS=4 scripts/run_tests.sh tests/tools/test_transcription_tools.py tests/gateway/test_execution_scratch_boundary.py
```
