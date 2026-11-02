"""Semantier desktop CLI branding must not leak upstream Hermes labels."""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console

from hermes_cli import banner
from hermes_cli.cli_render import _build_compact_banner


def test_branded_full_banner_uses_windows_product_version(monkeypatch):
    monkeypatch.setenv("SEMANTIER_CLI_BRAND", "1")
    monkeypatch.setenv("SEMANTIER_CLI_VERSION", "0.0.77")
    output = StringIO()
    console = Console(file=output, width=140, color_system="truecolor", force_terminal=True, no_color=False)

    with (
        patch.object(banner.shutil, "get_terminal_size", return_value=SimpleNamespace(columns=140)),
        patch.object(banner, "_mcp_configured", return_value=False),
        patch.object(banner, "get_update_result", return_value=0),
    ):
        banner.build_welcome_banner(
            console, model="qwen3.5-plus", cwd="C:\\Users\\Test",
            tools=[], enabled_toolsets=[], provider="dashscope",
            get_toolset_for_tool=lambda _: "other",
            availability={"unavailable_toolsets": []},
            skills_by_category={},
        )

    rendered = output.getvalue()
    assert "Semantier Agent v0.0.77" in rendered
    assert "司序科技" in rendered
    assert "Nous Research" not in rendered
    assert "Hermes Agent" not in rendered
    assert "██" in rendered
    # Semantier's canonical Asimov Action Yellow, not Hermes gold/bronze.
    assert "38;2;232;201;45" in rendered
    assert "38;2;255;215;0" not in rendered
    assert "38;2;205;127;50" not in rendered


def test_branded_compact_banner_even_in_fast_startup(monkeypatch):
    monkeypatch.setenv("SEMANTIER_CLI_BRAND", "1")
    monkeypatch.setenv("SEMANTIER_CLI_VERSION", "0.0.77")
    monkeypatch.setenv("HERMES_FAST_STARTUP_BANNER", "1")
    with patch("hermes_cli.cli_render.shutil.get_terminal_size", return_value=SimpleNamespace(columns=88)):
        rendered = _build_compact_banner()
    assert "SEMANTIER" in rendered
    assert "Semantier Agent v0.0.77" in rendered
    assert "NOUS HERMES" not in rendered
    assert "#e8c92d" in rendered
    assert "Nous Research" not in rendered


def test_narrow_branded_banner_has_no_upstream_signature(monkeypatch):
    monkeypatch.setenv("SEMANTIER_CLI_BRAND", "1")
    with patch("hermes_cli.cli_render.shutil.get_terminal_size", return_value=SimpleNamespace(columns=29)):
        rendered = _build_compact_banner()
    assert "SEMANTIER" in rendered
    assert "司序科技" in rendered
    assert "Nous Research" not in rendered


def test_default_cli_preserves_upstream_model_attribution(monkeypatch):
    monkeypatch.delenv("SEMANTIER_CLI_BRAND", raising=False)
    monkeypatch.delenv("SEMANTIER_CLI_VERSION", raising=False)
    lines = banner._banner_left_lines(
        "qwen3.5-plus", "C:\\Users\\Test", None, None, "dashscope",
        accent="#FFD700", dim="#777777",
    )
    assert "Nous Research" in lines[0]
