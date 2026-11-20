"""Portable codecs are not Windows key-storage or TPM qualification."""
import struct
import sys
import pytest
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature


def test_cng_public_and_signature_formats_preserve_p256_proof():
    from hermes_cli.semantier_device_key import public_key_pem,der_signature
    key=ec.generate_private_key(ec.SECP256R1())
    point=key.public_key().public_numbers()
    blob=struct.pack('<II',0x31534345,32)+point.x.to_bytes(32,'big')+point.y.to_bytes(32,'big')
    pem=public_key_pem(blob)
    assert serialization.load_pem_public_key(pem.encode()).public_numbers()==point
    message=b'bounded signature codec test'
    r,s=decode_dss_signature(key.sign(message,ec.ECDSA(hashes.SHA256())))
    signature=der_signature(r.to_bytes(32,'big')+s.to_bytes(32,'big'))
    key.public_key().verify(signature,message,ec.ECDSA(hashes.SHA256()))
    for bad in (blob[:-1],struct.pack('<II',0x314b4345,32)+blob[8:],struct.pack('<II',0x31534345,31)+blob[8:]):
        with pytest.raises(ValueError):
            public_key_pem(bad)
    with pytest.raises(ValueError):
        der_signature(b'bad')


def test_no_software_key_fallback_on_non_windows():
    from hermes_cli.semantier_device_key import CngDeviceKey
    if sys.platform=='win32':
        pytest.skip('non-Windows refusal only')
    with pytest.raises(OSError,match='Windows'):
        CngDeviceKey('a'*32)
