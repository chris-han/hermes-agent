"""Merged path admission preserves byte and link verification."""
import pytest

from pm.store import tree_digest
from scripts.bundles.native_prepared import _verified_digest


def test_verified_tree_matches_existing_digest_and_reads_current_bytes(tmp_path):
    root = tmp_path / "payload"
    root.mkdir()
    (root / "module.py").write_text("original")
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"ignored cache")
    original = tree_digest(root)
    assert _verified_digest(root, root) == original
    (root / "module.py").write_text("modified")
    assert _verified_digest(root, root) != original
    assert _verified_digest(root, root) == tree_digest(root)


def test_escaped_links_in_excluded_cache_are_still_rejected(tmp_path):
    root = tmp_path / "payload"
    cache = root / "__pycache__"
    cache.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_text("outside")
    try:
        (cache / "escaped").symlink_to(outside)
    except OSError:
        pytest.skip("Host account cannot create symlinks")
    with pytest.raises(ValueError, match="escaped"):
        _verified_digest(root, root)
