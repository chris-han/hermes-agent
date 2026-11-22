"""Cloud Billing account default is the only automatic desktop Cloud recommendation."""

from contextlib import nullcontext

import pytest


@pytest.mark.parametrize(
    ("preferred", "expected"),
    [
        ("deepseek-v4", "deepseek-v4"),
        ("not-available", ""),
        (None, ""),
        ("image-only", ""),
    ],
)
def test_cloud_default_reads_verified_account_preference(monkeypatch, preferred, expected):
    from agent import semantier_gateway
    from hermes_cli.web_routers import models

    seen = []
    monkeypatch.setattr(models, "_config_profile_scope", lambda profile: (seen.append(profile), nullcontext())[1])
    monkeypatch.setattr(semantier_gateway, "gateway_settings", lambda *, allow_unselected: {"authenticated": allow_unselected})

    def catalog(*, refresh, settings):
        assert refresh is True
        assert settings == {"authenticated": True}
        return {
            "object": "list",
            "recommended_default_model": preferred,
            "data": [
                {"id": "qwen3.5-plus", "modality": "chat"},
                {"id": "deepseek-v4", "modality": "chat"},
                {"id": "image-only", "modality": "image"},
            ],
        }

    monkeypatch.setattr(semantier_gateway, "cloud_catalog", catalog)
    assert models.get_recommended_default_model("semantier-cloud", profile="account-profile") == {
        "provider": "semantier-cloud", "model": expected, "free_tier": None
    }
    assert seen == ["account-profile"]


@pytest.mark.parametrize(
    ("assignment_account", "expected"),
    [
        ("account-a", "qwen3.5-plus"),
        ("account-b", "deepseek-v4"),
        (None, "deepseek-v4"),
    ],
)
def test_cloud_default_respects_workspace_assignment_only_for_current_account(
    monkeypatch, assignment_account, expected,
):
    from agent import semantier_gateway
    from hermes_cli.web_routers import models

    monkeypatch.setattr(models, "_config_profile_scope", lambda profile: nullcontext())
    monkeypatch.setattr(semantier_gateway, "gateway_settings", lambda *, allow_unselected: "scope")
    monkeypatch.setattr(semantier_gateway, "cloud_catalog", lambda **kwargs: {
        "object": "list",
        "recommended_default_model": "deepseek-v4",
        "assignments": {"main": {"model": "qwen3.5-plus", "account_id": assignment_account}},
        "data": [
            {"id": "deepseek-v4", "account_id": "account-a", "modality": "chat"},
            {"id": "qwen3.5-plus", "account_id": "account-a", "modality": "chat"},
        ],
    })
    assert models.get_recommended_default_model("semantier-cloud")["model"] == expected


def test_cloud_default_unavailable_never_guesses_from_model_list(monkeypatch):
    from agent import semantier_gateway
    from hermes_cli.web_routers import models

    monkeypatch.setattr(models, "_config_profile_scope", lambda profile: nullcontext())
    monkeypatch.setattr(semantier_gateway, "gateway_settings", lambda *, allow_unselected: "scope")

    def unavailable(**kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(semantier_gateway, "cloud_catalog", unavailable)
    assert models.get_recommended_default_model("semantier-cloud")["model"] == ""
