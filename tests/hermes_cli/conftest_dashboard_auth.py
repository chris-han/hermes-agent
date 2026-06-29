"""Stub auth provider + shared fixtures for dashboard-auth tests.

NOT a pytest conftest.py -- this is an importable helper module.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time

from hermes_cli.dashboard_auth.base import (
    DashboardAuthProvider,
    InvalidCodeError,
    LoginStart,
    RefreshExpiredError,
    Session,
)

_STUB_SECRET = b"stub-test-secret-not-for-prod"
_SIG_LEN = hashlib.sha256().digest_size


def _sign(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode()
    sig = hmac.new(_STUB_SECRET, raw, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(raw + sig).decode()


def _unsign(token: str) -> dict | None:
    try:
        blob = base64.urlsafe_b64decode(token.encode())
        if len(blob) <= _SIG_LEN:
            return None
        raw, sig = blob[:-_SIG_LEN], blob[-_SIG_LEN:]
        expected = hmac.new(_STUB_SECRET, raw, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            return None
        return json.loads(raw)
    except Exception:
        return None


class StubAuthProvider(DashboardAuthProvider):
    """Local fake IDP for dashboard-auth E2E tests."""

    name = "stub"
    display_name = "Stub IdP (test only)"

    def __init__(self, default_ttl: int = 3600):
        self._default_ttl = default_ttl
        self._state_to_verifier: dict[str, str] = {}

    def start_login(self, *, redirect_uri: str) -> LoginStart:
        state = secrets.token_urlsafe(16)
        verifier = secrets.token_urlsafe(32)
        self._state_to_verifier[state] = verifier
        return LoginStart(
            redirect_url=f"{redirect_uri}?code=stub_code&state={state}",
            cookie_payload={
                "hermes_session_pkce": f"state={state};verifier={verifier}",
            },
        )

    def complete_login(
        self, *, code: str, state: str, code_verifier: str, redirect_uri: str,
    ) -> Session:
        if code != "stub_code":
            raise InvalidCodeError(f"stub expects code='stub_code', got {code!r}")
        expected_verifier = self._state_to_verifier.get(state)
        if expected_verifier is None or expected_verifier != code_verifier:
            raise InvalidCodeError("stub state/verifier mismatch")
        del self._state_to_verifier[state]

        now = int(time.time())
        exp = now + self._default_ttl
        return Session(
            user_id="stub-user-1",
            email="stub@example.test",
            display_name="Stub User",
            org_id="stub-org-1",
            provider=self.name,
            expires_at=exp,
            access_token=_sign({
                "sub": "stub-user-1",
                "email": "stub@example.test",
                "name": "Stub User",
                "org_id": "stub-org-1",
                "exp": exp,
            }),
            refresh_token=_sign({
                "sub": "stub-user-1",
                "kind": "refresh",
                "exp": now + 30 * 86400,
            }),
        )

    def verify_session(self, *, access_token: str):
        payload = _unsign(access_token)
        if payload is None or payload.get("exp", 0) <= int(time.time()):
            return None
        return Session(
            user_id=payload["sub"],
            email=payload["email"],
            display_name=payload["name"],
            org_id=payload["org_id"],
            provider=self.name,
            expires_at=payload["exp"],
            access_token=access_token,
            refresh_token="",
        )

    def refresh_session(self, *, refresh_token: str) -> Session:
        payload = _unsign(refresh_token)
        if payload is None or payload.get("exp", 0) <= int(time.time()):
            raise RefreshExpiredError("stub refresh token expired/invalid")
        now = int(time.time())
        exp = now + self._default_ttl
        return Session(
            user_id=payload["sub"],
            email="stub@example.test",
            display_name="Stub User",
            org_id="stub-org-1",
            provider=self.name,
            expires_at=exp,
            access_token=_sign({
                "sub": payload["sub"],
                "email": "stub@example.test",
                "name": "Stub User",
                "org_id": "stub-org-1",
                "exp": exp,
            }),
            refresh_token=_sign({
                "sub": payload["sub"],
                "kind": "refresh",
                "exp": now + 30 * 86400,
            }),
        )

    def revoke_session(self, *, refresh_token: str) -> None:
        return None
