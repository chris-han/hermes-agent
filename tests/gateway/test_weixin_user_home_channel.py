from gateway.session import SessionSource
from gateway.config import Platform


def test_ensure_auth_db_creates_auth_db_file(monkeypatch, tmp_path):
    monkeypatch.setenv("SEMANTIER_LOCAL_STATE_DIR", str(tmp_path / ".semantier-home"))

    from agents.auth_db import ensure_auth_db

    path = ensure_auth_db()
    assert path.name == "auth.db"
    assert path.exists()


def test_ensure_auth_db_delegates_to_upstream(monkeypatch, tmp_path):
    import agents.auth_db as shim

    captured: dict[str, object] = {}

    def fake_ensure(*, json_loader=None):
        captured["json_loader"] = json_loader

    target = tmp_path / ".semantier-home" / "auth.db"
    monkeypatch.setattr(shim._UPSTREAM, "ensure_auth_db", fake_ensure)
    monkeypatch.setattr(shim, "auth_db_path", lambda: target)

    loader = lambda name: None
    path = shim.ensure_auth_db(json_loader=loader)

    assert captured == {"json_loader": loader}
    assert path == target


def test_weixin_user_scoped_home_channel_self_heals_missing_home(monkeypatch, tmp_path):
    monkeypatch.setenv("SEMANTIER_LOCAL_STATE_DIR", str(tmp_path / ".semantier-home"))

    from agents.auth_db import ensure_auth_db, get_weixin_runtime_account, save_weixin_runtime_account
    from gateway.run import _weixin_source_has_user_scoped_home_channel

    ensure_auth_db()
    save_weixin_runtime_account(
        {
            "account_id": "acct-1@im.bot",
            "owner_user_id": "user-1",
            "owner_workspace_id": "ws-123",
            "external_user_id": "wx-user-1",
            "user_id": "wx-user-1",
            "token": "token-123",
            "saved_at": "2026-05-17T00:00:00Z",
        }
    )

    source = SessionSource(
        platform=Platform.WEIXIN,
        chat_id="wx-user-1",
        chat_type="dm",
        user_id="wx-user-1",
        workspace_owner_id="ws-123",
    )

    assert _weixin_source_has_user_scoped_home_channel(source) is True
    saved = get_weixin_runtime_account("acct-1@im.bot")
    assert saved is not None
    assert saved["home_channel"] == "wx-user-1"


def test_weixin_user_scoped_home_channel_self_heals_without_source_workspace(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SEMANTIER_LOCAL_STATE_DIR", str(tmp_path / ".semantier-home"))

    from agents.auth_db import ensure_auth_db, get_weixin_runtime_account, save_weixin_runtime_account
    from gateway.run import _weixin_source_has_user_scoped_home_channel

    ensure_auth_db()
    save_weixin_runtime_account(
        {
            "account_id": "acct-1@im.bot",
            "owner_user_id": "user-1",
            "owner_workspace_id": "ws-123",
            "external_user_id": "wx-user-1",
            "user_id": "wx-user-1",
            "token": "token-123",
            "saved_at": "2026-05-17T00:00:00Z",
        }
    )

    source = SessionSource(
        platform=Platform.WEIXIN,
        chat_id="wx-user-1",
        chat_type="dm",
        user_id="wx-user-1",
    )

    assert _weixin_source_has_user_scoped_home_channel(source) is True
    saved = get_weixin_runtime_account("acct-1@im.bot")
    assert saved is not None
    assert saved["home_channel"] == "wx-user-1"
