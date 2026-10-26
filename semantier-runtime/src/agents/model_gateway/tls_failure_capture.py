"""Bounded passive TLS evidence from the native connection's existing trace hook.

The Proxy log owner retains these observations. They are not financial receipts,
and this transport never opens a diagnostic connection or changes TLS policy.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import ipaddress
import json
import logging
import os
import re
import ssl
import sys

import httpx

logger = logging.getLogger(__name__)
MAX_EVENT_BYTES = 48 * 1024
MAX_CERT_BYTES = 8 * 1024
MAX_CHAIN_BYTES = 24 * 1024


def _identifier(value, limit=128):
    return value if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.:\-]{1,' + str(limit) + '}', value) else None


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=1)
def _versions():
    versions = {}
    for name in ('httpx', 'httpcore', 'anyio', 'openai', 'litellm', 'certifi', 'cryptography'):
        try:
            value = importlib.metadata.version(name)
            versions[name] = value if re.fullmatch(r'[A-Za-z0-9.+_-]{1,64}', value) else None
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': '.'.join(map(str, sys.version_info[:3])), 'openssl': ssl.OPENSSL_VERSION,
            'clients': versions, 'missing_version_reason': 'distribution_metadata_unavailable'}


def _runtime():
    digest = os.environ.get('SEMANTIER_RUNTIME_IMAGE_DIGEST', '')
    declared = digest if re.fullmatch(r'sha256:[a-f0-9]{64}', digest) else None
    return {**_versions(), 'image': {'digest': declared,
            'basis': 'operator_declared' if declared else 'unavailable',
            'unavailable_reason': None if declared else 'image_identity_not_exposed_to_process'}}


def _policy(context):
    certs = context.get_ca_certs(binary_form=True)
    trust_complete = len(certs) <= 512
    trust = sorted(hashlib.sha256(cert).hexdigest() for cert in certs[:512])
    ciphers = context.get_ciphers()
    ciphers_complete = len(ciphers) <= 128
    policy = {'verify_mode': int(context.verify_mode), 'check_hostname': context.check_hostname,
              'verify_flags': int(context.verify_flags), 'security_level': context.security_level,
              'options': int(context.options),
              'minimum_version': int(context.minimum_version), 'maximum_version': int(context.maximum_version),
              'trust_sha256': _digest(trust) if trust_complete else None,
              'trust_complete': trust_complete,
              'trust_unavailable_reason': None if trust_complete else 'trust_inventory_exceeds_bound',
              'ciphers_sha256': _digest([c['name'] for c in ciphers]) if ciphers_complete else None,
              'ciphers_complete': ciphers_complete,
              'ciphers_unavailable_reason': None if ciphers_complete else 'cipher_inventory_exceeds_bound',
              'trust_env': False, 'retries': 0}
    return {**policy, 'configuration_sha256': _digest(policy)}


def _exceptions(exc):
    chain, seen, code = [], set(), None
    while exc is not None and id(exc) not in seen and len(chain) < 8:
        seen.add(id(exc))
        # Exception strings/args/traceback locals can contain credentials or bodies.
        module = type(exc).__module__
        name = type(exc).__name__
        known = module in {'ssl', 'httpx', 'httpcore', 'builtins'} and name in {
            'ConnectError', 'ConnectTimeout', 'SSLError', 'SSLCertVerificationError',
            'TimeoutError', 'OSError', 'ValueError', 'RuntimeError'}
        item = {'type': module + '.' + name if known else 'other_exception',
                'context_display_suppressed': exc.__suppress_context__}
        if isinstance(exc, ssl.SSLCertVerificationError):
            value = getattr(exc, 'verify_code', None)
            if isinstance(value, int) and not isinstance(value, bool):
                code = value
                item['verification_error_code'] = value
        chain.append(item)
        # HTTPcore suppresses display of the retained SSL context. Inspect only
        # its type/numeric code, never its text, args or traceback locals.
        exc = exc.__cause__ or exc.__context__
    return chain, code, exc is not None


def _certificates(stream):
    result = {'available': False, 'certificates': [], 'truncated': False,
              'unavailable_reason': 'ssl_object_not_exposed_after_failed_handshake'}
    if stream is None:
        return result
    obj = stream.get_extra_info('ssl_object')
    if obj is None:
        return result
    getter = getattr(obj, 'get_unverified_chain', None)
    chain = getter() if callable(getter) else [obj.getpeercert(binary_form=True)]
    if not chain:
        result['unavailable_reason'] = 'peer_certificate_not_exposed'
        return result
    total = 0
    result['truncated'] = len(chain) > 4
    for cert in chain[:4]:
        if hasattr(cert, 'public_bytes'):
            cert = cert.public_bytes()
        if isinstance(cert, str) and len(cert) <= MAX_CERT_BYTES * 2:
            cert = ssl.PEM_cert_to_DER_cert(cert)
        if not isinstance(cert, bytes) or len(cert) > MAX_CERT_BYTES or total + len(cert) > MAX_CHAIN_BYTES:
            result['truncated'] = True
            continue
        from cryptography.x509 import load_der_x509_certificate
        try:
            load_der_x509_certificate(cert)
        except ValueError:
            result['truncated'] = True
            continue
        total += len(cert)
        result['certificates'].append({'der_base64': base64.b64encode(cert).decode('ascii'),
                                       'sha256': hashlib.sha256(cert).hexdigest()})
    result['available'] = bool(result['certificates'])
    result['unavailable_reason'] = None if result['available'] else 'no_bounded_public_certificate_exposed'
    return result


class TlsFailureTransport(httpx.AsyncHTTPTransport):
    """Same native HTTPX transport/policy, observing only this request's trace."""
    def __init__(self, correlation):
        super().__init__(retries=0, trust_env=False)
        self._correlation = correlation

    async def handle_async_request(self, request):
        previous = request.extensions.get('trace')
        stream, context, sni, peer = None, None, None, None
        failed = False

        async def trace(name, info):
            nonlocal stream, context, sni, peer, failed
            try:
                if name == 'connection.connect_tcp.complete':
                    stream = info.get('return_value')
                    address = stream.get_extra_info('server_addr') if stream else None
                    if isinstance(address, tuple) and len(address) >= 2:
                        host = str(ipaddress.ip_address(address[0]))
                        if isinstance(address[1], int) and 0 < address[1] <= 65535:
                            peer = {'address': host, 'port': address[1]}
                elif name == 'connection.start_tls.started':
                    context = info.get('ssl_context')
                    value = info.get('server_hostname')
                    sni = _identifier(value.decode('ascii') if isinstance(value, bytes) else value, 253)
                elif name == 'connection.start_tls.failed':
                    failed = True
            except Exception:
                # Observation failure must not affect connection/verification behavior.
                pass
            if previous is not None:
                await previous(name, info)

        request.extensions['trace'] = trace
        try:
            return await super().handle_async_request(request)
        except Exception as exc:
            if failed:
                try:
                    attempt = self._correlation()
                    invocation_id = _identifier(getattr(attempt, 'invocation_id', None))
                    attempt_id = _identifier(getattr(attempt, 'attempt_id', None))
                    chain, code, truncated = _exceptions(exc)
                    try:
                        certificates = _certificates(stream)
                    except Exception:
                        certificates = {'available': False, 'certificates': [], 'truncated': False,
                                        'unavailable_reason': 'peer_certificate_access_failed'}
                    try:
                        policy = _policy(context) if context else None
                    except Exception:
                        policy = None
                    event = {'schema_version': 'tls_failure_observation.v1',
                             'observed_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                             'observation_basis': 'same_connection',
                             'invocation_id': invocation_id, 'attempt_id': attempt_id,
                             'correlation_unavailable_reason': None if invocation_id and attempt_id else 'missing_or_invalid_attempt_identity',
                             'endpoint': {'hostname': _identifier(request.url.host, 253),
                                          'port': request.url.port or 443, 'sni': sni,
                                          'sni_unavailable_reason': None if sni else 'sni_not_exposed'},
                             'peer': peer, 'peer_unavailable_reason': None if peer else 'tcp_peer_not_exposed',
                             'proxy_path': {'kind': 'direct', 'basis': 'native_transport_no_proxy_trust_env_false',
                                            'transparent_intermediaries': 'unobservable'},
                             'verification_error_code': code,
                             'verification_code_unavailable_reason': None if code is not None else 'not_exposed_by_exception',
                             'exception_chain': chain, 'exception_chain_truncated': truncated,
                             'tls_policy': policy, 'tls_policy_unavailable_reason': None if policy else 'context_snapshot_unavailable',
                             'runtime': _runtime(), 'peer_certificates': certificates}
                    wire = json.dumps(event, sort_keys=True, separators=(',', ':'))
                    if len(wire.encode()) <= MAX_EVENT_BYTES:
                        logger.warning('%s', wire)
                except Exception:
                    # No fallback writer, retry, financial mutation or replacement exception.
                    pass
            raise
        finally:
            if previous is None:
                request.extensions.pop('trace', None)
            else:
                request.extensions['trace'] = previous
