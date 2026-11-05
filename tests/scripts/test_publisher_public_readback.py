"""The anonymous public body must agree before a channel can be selected."""
import hashlib
import io

import pytest

from hermes_cli.release_channels import ChannelReader
from scripts.releases.public_readback import verify_public_object


@pytest.mark.parametrize('body', [b'public artifact', b'broken artifact', b'', b'public artifact extra'])
def test_public_body_must_match_exact_signed_object(monkeypatch, body):
    expected = b'public artifact'
    url = 'https://download.semantier.com/releases/test.msix'
    class Response(io.BytesIO):
        status = 200
        def geturl(self):
            return url
    calls = []
    def open_public(request, timeout):
        assert request.full_url == url
        assert not request.has_header('Authorization')
        assert timeout == 30
        calls.append('read')
        return Response(body)
    def initialize(self, origin):
        self.base_url = origin
        self.opener = open_public
    monkeypatch.setattr(ChannelReader, '__init__', initialize)
    descriptor = {'key': 'releases/test.msix', 'size': len(expected),
                  'sha256': hashlib.sha256(expected).hexdigest()}
    guard = lambda: calls.append('guard')
    if body == expected:
        verify_public_object('https://download.semantier.com', descriptor, guard)
        assert calls == ['guard', 'read', 'guard']
    else:
        with pytest.raises(ValueError, match='Public object'):
            verify_public_object('https://download.semantier.com', descriptor, guard)


def test_public_failure_prevents_conditional_head_write(monkeypatch):
    from scripts.releases.simple_native_slots import R2Driver
    from hermes_cli.release_channels import canonical_json
    body = canonical_json({'installation': {'objects': [{'key': 'releases/test.msix',
        'size': 3, 'sha256': 'a' * 64}]}})
    class Store:
        def get(self, key):
            return body, 'etag'
        def put(self, *args):
            pytest.fail('Public readback failure must preserve the active release')
    driver = R2Driver(Store(), 'https://download.semantier.com')
    monkeypatch.setattr(driver, 'verify_lock', lambda: None)
    def fail(*args):
        raise ValueError('Public object SHA-256 mismatch')
    monkeypatch.setattr(driver, 'verify_public_object', fail)
    with pytest.raises(ValueError, match='Public object'):
        driver.commit('semantier-acceptance', 'etag', {'manifestKey': 'releases/build.json',
            'sha256': hashlib.sha256(body).hexdigest()})
