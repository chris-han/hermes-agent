"""Bounded, version-fenced streaming verification of an immutable R2 object."""
import hashlib
import http.client
import re
from urllib.parse import urlsplit


def verify_resumable(transport, url, creds, expected_size, expected_sha256, *, attempts=8):
    """Verify every byte, resuming interrupted reads only at a validated offset."""
    if type(expected_size) is not int or expected_size <= 0 or not re.fullmatch('[a-f0-9]{64}', expected_sha256):
        raise ValueError('Invalid readback identity')
    parsed=urlsplit(url)
    if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError('Invalid readback URL')
    digest=hashlib.sha256(); offset=0; etag=None
    for _ in range(attempts):
        extra={'Range':f'bytes={offset}-','If-Match':etag} if offset else {}
        headers=transport.r2_headers('GET',parsed.netloc,parsed.path,parsed.query,
            transport.EMPTY_SHA,transport.amz_timestamp(),creds,extra_headers=extra)
        connection=None
        try:
            connection=transport._connection(url,timeout=45.0)
            connection.request('GET',parsed.path+('?' + parsed.query if parsed.query else ''),headers=headers)
            response=connection.getresponse()
            current=response.getheader('etag')
            if not current or current.startswith('W/') or (etag is not None and current!=etag):
                raise ValueError('Readback object version changed')
            if offset:
                if response.status!=206 or response.getheader('content-range')!=f'bytes {offset}-{expected_size-1}/{expected_size}':
                    raise ValueError('Readback range mismatch')
            elif response.status!=200:
                raise ValueError('Readback response mismatch')
            if response.getheader('content-length')!=str(expected_size-offset):
                raise ValueError('Readback length mismatch')
            etag=current
            while offset<expected_size:
                try:
                    chunk=response.read(min(1024*1024,expected_size-offset))
                except http.client.IncompleteRead as error:
                    chunk=error.partial
                    if len(chunk)>expected_size-offset:
                        raise ValueError('Readback exceeded expected size')
                    if chunk:
                        digest.update(chunk);offset+=len(chunk)
                    raise
                if not chunk:
                    raise http.client.IncompleteRead(b'')
                if len(chunk)>expected_size-offset:
                    raise ValueError('Readback exceeded expected size')
                digest.update(chunk);offset+=len(chunk)
            if digest.hexdigest()!=expected_sha256:
                raise ValueError('Readback SHA-256 mismatch')
            return
        except (OSError,http.client.HTTPException):
            continue
        finally:
            if connection is not None:
                connection.close()
    raise OSError('Readback retries exhausted')
