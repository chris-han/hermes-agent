from gateway.config import Platform
from gateway.session import SessionSource, build_session_key


def test_workspace_owner_scopes_gateway_session_key():
    source = SessionSource(
        platform=Platform.WEIXIN,
        chat_id="wx-a",
        chat_type="dm",
        user_id="wx-a",
        workspace_owner_id="ws-a",
    )

    assert build_session_key(source) == "agent:main:workspace:ws-a:weixin:dm:wx-a"

