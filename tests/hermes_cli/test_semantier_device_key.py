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


def test_native_helper_rejects_cross_account_signing_before_key_access(monkeypatch):
    from hermes_cli import semantier_device_key as module
    calls=[]
    monkeypatch.setattr(module,'CngDeviceKey',lambda *a,**kw:calls.append((a,kw)))
    with pytest.raises(ValueError,match='scope'):
        module.native_key_request(dict(operation='sign_enrollment',key_id='a'*32,
            expected_scope=dict(user_id='owner',organization_id='org',workspace_id='ws'),
            context=dict(user_id='foreign',organization_id='org',workspace_id='ws')))
    assert calls==[]


def test_native_helper_carries_public_proof_and_never_asserts_attestation(monkeypatch,capsys):
    from hermes_cli import semantier_device_key as module
    class Key:
        public_key='PUBLIC KEY'
        def __init__(self,key_id,*,create=False):
            assert key_id=='a'*32 and create is True
        def __enter__(self):return self
        def __exit__(self,*_):pass
    monkeypatch.setattr(module,'CngDeviceKey',Key)
    assert module.native_key_request(dict(operation='create',key_id='a'*32))==dict(
        key_id='a'*32,public_key='PUBLIC KEY',assurance='NOT_VERIFIED')
    import io
    import json
    from types import SimpleNamespace
    monkeypatch.setattr(sys,'stdin',SimpleNamespace(buffer=io.BytesIO(json.dumps(dict(operation='create',key_id='a'*32)).encode())))
    assert module._main()==0
    assert json.loads(capsys.readouterr().out)=={'ok':True,'result':dict(key_id='a'*32,public_key='PUBLIC KEY',assurance='NOT_VERIFIED')}
    with pytest.raises(ValueError):
        module.native_key_request(dict(operation='export_private',key_id='a'*32))


@pytest.mark.parametrize('wire',[
    '{"operation":"create","operation":"delete","key_id":"'+ 'a'*32+'"}',
    ' '*32769,
    '{"operation":"export_private","key_id":"'+ 'a'*32+'"}',
])
def test_native_helper_stdin_refuses_ambiguous_or_unbounded_requests(wire):
    import json
    import subprocess
    result=subprocess.run([sys.executable,'-m','hermes_cli.semantier_device_key'],input=wire,
                          text=True,capture_output=True,timeout=15,check=False)
    assert result.returncode==2
    assert json.loads(result.stdout)=={'ok':False,'error':'INVALID_NATIVE_DEVICE_KEY_REQUEST'}
    assert result.stderr==''
