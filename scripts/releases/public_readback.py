"""Anonymous, bounded hash verification before a publisher selects a release."""
import hashlib
import time
from urllib.request import Request

from hermes_cli.release_channels import ChannelReader, artifact_key, require_sha256


def verify_public_object(origin, descriptor, guard):
    key = artifact_key(descriptor['key'])
    expected_hash = require_sha256(descriptor['sha256'])
    expected_size = descriptor['size']
    if type(expected_size) is not int or expected_size <= 0:
        raise ValueError('Invalid public object size')
    reader = ChannelReader(origin)
    url = reader.base_url + '/' + key
    guard()
    digest = hashlib.sha256()
    size = 0
    deadline = time.monotonic() + 1800
    request = Request(url, headers={'Cache-Control': 'no-cache', 'User-Agent': 'SemantierPublisher/1.0'})
    with reader.opener(request, timeout=30) as response:
        if response.geturl() != url or response.status != 200:
            raise ValueError('Public object response or origin mismatch')
        while chunk := response.read(min(1024 * 1024, expected_size + 1 - size)):
            size += len(chunk)
            if size > expected_size or time.monotonic() > deadline:
                raise ValueError('Public object size or read deadline exceeded')
            digest.update(chunk)
    if size != expected_size or digest.hexdigest() != expected_hash:
        raise ValueError('Public object SHA-256 or size mismatch')
    guard()
    print('PUBLIC_READBACK_PASS ' + key + ' ' + digest.hexdigest(), flush=True)
