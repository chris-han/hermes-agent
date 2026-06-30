"""Tests for the Alibaba DashScope provider profile."""


def test_alibaba_profile_exposes_dashscope_metadata():
    from providers import get_provider_profile

    profile = get_provider_profile("alibaba")

    assert profile is not None
    assert profile.display_name == "Qwen Cloud / DashScope"
    assert profile.env_vars == ("DASHSCOPE_API_KEY",)
    assert profile.base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert profile.default_aux_model == "qwen3.5-plus"


def test_dashscope_alias_resolves_to_alibaba_profile():
    from providers import get_provider_profile

    profile = get_provider_profile("dashscope")

    assert profile is not None
    assert profile.name == "alibaba"
