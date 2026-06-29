from __future__ import annotations

import gateway.run as gateway_run


def test_effective_disabled_toolsets_for_message_adds_terminal_and_code_execution_for_analytics():
    disabled = gateway_run._effective_disabled_toolsets_for_message(
        ["web"],
        "请做成本分析并给我利润率趋势",
    )

    assert disabled is not None
    assert "web" in disabled
    assert "terminal" in disabled
    assert "code_execution" in disabled


def test_effective_disabled_toolsets_for_message_keeps_non_analytics_unchanged():
    disabled = gateway_run._effective_disabled_toolsets_for_message(
        ["web"],
        "写一封周末团建邀请邮件",
    )

    assert disabled == ["web"]
