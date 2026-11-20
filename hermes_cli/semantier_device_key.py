"""Windows-only nonexportable CNG signing adapter; no attestation authority.

Only explicit native enrollment may create a key. The caller retains its opaque
key ID in the existing native connection owner; no private material is persisted
by Python. Provider selection never establishes server-verified TPM assurance.
"""
import ctypes
import hashlib
import re
import struct
import sys

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

PROVIDER='Microsoft Platform Crypto Provider'
_PREFIX='semantier.managed-device.v1.'


def public_key_pem(blob):
    if len(blob)!=72 or struct.unpack('<II',blob[:8])!=(0x31534345,32):
        raise ValueError('CNG P256 public key required')
    public=ec.EllipticCurvePublicNumbers(int.from_bytes(blob[8:40],'big'),int.from_bytes(blob[40:],'big'),ec.SECP256R1()).public_key()
    return public.public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo).decode('ascii')


def der_signature(raw):
    if len(raw)!=64:
        raise ValueError('CNG P256 signature required')
    return encode_dss_signature(int.from_bytes(raw[:32],'big'),int.from_bytes(raw[32:],'big'))


class CngDeviceKey:
    """One current-Windows-user key. No software fallback or implicit key reset."""
    def __init__(self,key_id,*,create=False):
        if sys.platform!='win32':
            raise OSError('Windows CNG device key unavailable on this platform')
        if not isinstance(key_id,str) or not re.fullmatch('[a-f0-9]{32}',key_id) or type(create) is not bool:
            raise ValueError('opaque native device key ID required')
        self.key_id=key_id
        self._provider=ctypes.c_void_p()
        self._key=ctypes.c_void_p()
        self._api=ctypes.WinDLL('ncrypt.dll')
        handle=ctypes.c_void_p
        dword=ctypes.c_uint32
        word=ctypes.c_wchar_p
        ptr=ctypes.c_void_p
        definitions={
            'NCryptOpenStorageProvider':[ctypes.POINTER(handle),word,dword],
            'NCryptOpenKey':[handle,ctypes.POINTER(handle),word,dword,dword],
            'NCryptCreatePersistedKey':[handle,ctypes.POINTER(handle),word,word,dword,dword],
            'NCryptSetProperty':[handle,word,ptr,dword,dword],
            'NCryptGetProperty':[handle,word,ptr,dword,ctypes.POINTER(dword),dword],
            'NCryptFinalizeKey':[handle,dword],
            'NCryptExportKey':[handle,handle,word,ptr,ptr,dword,ctypes.POINTER(dword),dword],
            'NCryptSignHash':[handle,ptr,ptr,dword,ptr,dword,ctypes.POINTER(dword),dword],
            'NCryptDeleteKey':[handle,dword],
            'NCryptFreeObject':[handle],
        }
        for name,args in definitions.items():
            fn=getattr(self._api,name)
            fn.argtypes=args
            fn.restype=ctypes.c_int32
        try:
            self._check(self._api.NCryptOpenStorageProvider(ctypes.byref(self._provider),PROVIDER,0))
            if create:
                self._check(self._api.NCryptCreatePersistedKey(self._provider,ctypes.byref(self._key),'ECDSA_P256',_PREFIX+key_id,0,0))
                policy=dword(0)
                self._check(self._api.NCryptSetProperty(self._key,'Export Policy',ctypes.byref(policy),4,0))
                usage=dword(2)  # NCRYPT_ALLOW_SIGNING_FLAG
                self._check(self._api.NCryptSetProperty(self._key,'Key Usage',ctypes.byref(usage),4,0))
                self._check(self._api.NCryptFinalizeKey(self._key,0))
            else:
                self._check(self._api.NCryptOpenKey(self._provider,ctypes.byref(self._key),_PREFIX+key_id,0,0))
            policy=dword()
            size=dword()
            self._check(self._api.NCryptGetProperty(self._key,'Export Policy',ctypes.byref(policy),4,ctypes.byref(size),0))
            if size.value!=4 or policy.value!=0:
                raise OSError('CNG device key is not nonexportable')
            self.public_key=public_key_pem(self._export_public())
        except BaseException as exc:
            if create and self._key.value:
                status=self._api.NCryptDeleteKey(self._key,0)
                if not status:
                    self._key=ctypes.c_void_p()
                else:
                    exc.add_note('CNG failed-create key cleanup unavailable (0x%08x)'%(status & 0xffffffff))
            self.close()
            raise

    @staticmethod
    def _check(status):
        if status:
            raise OSError('Windows CNG device key operation failed (0x%08x)'%(status & 0xffffffff))

    def _export_public(self):
        size=ctypes.c_uint32()
        self._check(self._api.NCryptExportKey(self._key,None,'ECCPUBLICBLOB',None,None,0,ctypes.byref(size),0))
        if size.value!=72:
            raise ValueError('CNG P256 public blob required')
        output=ctypes.create_string_buffer(size.value)
        self._check(self._api.NCryptExportKey(self._key,None,'ECCPUBLICBLOB',None,output,len(output),ctypes.byref(size),0))
        return output.raw[:size.value]

    def _sign(self,message):
        if not self._key.value:
            raise OSError('CNG device key is closed')
        digest=ctypes.create_string_buffer(hashlib.sha256(message).digest())
        output=ctypes.create_string_buffer(64)
        size=ctypes.c_uint32()
        self._check(self._api.NCryptSignHash(self._key,None,digest,32,output,64,ctypes.byref(size),0))
        return der_signature(output.raw[:size.value])

    def sign_enrollment(self,context):
        from hermes_cli.semantier_device_proof import enrollment_message
        return self._sign(enrollment_message(context))

    def sign_operation(self,context):
        from hermes_cli.semantier_device_proof import operation_message
        return self._sign(operation_message(context))

    def delete(self):
        """Explicit retirement only; callers revoke managed authorization separately."""
        if self._key.value:
            self._check(self._api.NCryptDeleteKey(self._key,0))
            self._key=ctypes.c_void_p()  # NCryptDeleteKey releases the handle.

    def close(self):
        for name in ('_key','_provider'):
            handle=getattr(self,name,None)
            if handle is not None and handle.value:
                self._api.NCryptFreeObject(handle)
                setattr(self,name,ctypes.c_void_p())

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()
