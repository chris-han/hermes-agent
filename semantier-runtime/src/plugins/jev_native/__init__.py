"""Explicit experiment candidate. No hooks, paid tool, or automatic activation."""
import json


def register(ctx):
    ctx.register_auxiliary_task(
        "jev_native", display_name="Jev native experiment",
        description="Pinned native request execution-contract qualification",
    )
    def setup(parser):
        parser.add_argument("action", choices=["status"])
    def status(args):
        from dataclasses import asdict
        from .host import resolve_route
        try:
            result = {"route": asdict(resolve_route(ctx)), "live_execution": "BLOCKED_NO_PAID_EXPERIMENT_AUTHORITY"}
        except Exception:
            result = {"route": None, "live_execution": "BLOCKED_HOST_POLICY"}
        print(json.dumps(result, sort_keys=True))
        return result
    ctx.register_cli_command(name="jev-native", help="Inspect disabled experimental Jev route",
                             setup_fn=setup, handler_fn=status)
