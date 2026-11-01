"""Exercise Python bundle Git calls with the native Windows interpreter."""
import os
from pathlib import Path
import subprocess
import tempfile

from scripts.bundles.git_command import git_command
from scripts.bundles.desktop_prepare import git
from scripts.releases.commit_build import version_at


def main():
    if os.name != "nt" or not os.environ.get("SEMANTIER_WSL_GIT_DISTRO"):
        raise RuntimeError("explicit Windows/WSL Git test required")
    source = Path(r"D:\repo\hermes-build")
    revision = git(source, "rev-parse", "HEAD")
    if len(revision) != 40 or not version_at(source, revision):
        raise RuntimeError("source provenance check failed")
    tree = subprocess.check_output(git_command(source, "ls-tree", "-r", "-z", revision), cwd=source)
    if b"pyproject.toml\0" not in tree:
        raise RuntimeError("binary Git output changed")
    with tempfile.TemporaryDirectory(prefix="semantier-wsl-git-") as temp:
        archive = Path(temp) / "source.tar"
        subprocess.run(git_command(source, "archive", "--format=tar", "--output", str(archive),
                                   revision, "--", "pyproject.toml"), cwd=source, check=True)
        if archive.stat().st_size == 0:
            raise RuntimeError("archive is empty")
    print("WINDOWS_PYTHON_WSL_GIT_PASS")


if __name__ == "__main__":
    main()
