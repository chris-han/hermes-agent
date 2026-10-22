import os
import hmac
import hashlib
from typing import Optional


_KEY = None


def load_signing_key_from_env(env_var: str = "SEMANTIER_SIGNING_KEY") -> Optional[bytes]:
    global _KEY
    v = os.getenv(env_var)
    if not v:
        return None
    _KEY = v.encode("utf-8")
    return _KEY


def set_signing_key(key: bytes) -> None:
    global _KEY
    _KEY = key


def sign(data: str) -> Optional[str]:
    """Return hex HMAC-SHA256 signature of `data` if key configured, else None."""
    if _KEY is None:
        load_signing_key_from_env()
    if _KEY is None:
        return None
    mac = hmac.new(_KEY, data.encode("utf-8"), hashlib.sha256)
    return mac.hexdigest()


def verify(signature: str, data: str) -> bool:
    if _KEY is None:
        load_signing_key_from_env()
    if _KEY is None:
        return False
    expected = hmac.new(_KEY, data.encode("utf-8"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
