"""Unsupported packages are refused before bootstrapping a mutation worker."""
import pytest

from pm import client
from pm.package import InstallError


@pytest.mark.platforms("posix")
def test_posix_git_refusal_does_not_bootstrap_the_worker(monkeypatch):
    def unexpected_worker(*args, **kwargs):
        pytest.fail("An unavailable package must not bootstrap PM's worker runtime")

    monkeypatch.setattr(client, "_request", unexpected_worker)
    with pytest.raises(InstallError, match="POSIX uses system git by choice"):
        client.ensure("git", explicit=True)
