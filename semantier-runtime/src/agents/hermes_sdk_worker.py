"""Fixed-operation local SDK worker, owned by one immutable Hermes profile.

Executed only with the pinned upstream interpreter. Never import this module in
Core. The private stdio pipe is not an HTTP service or a second lifecycle engine.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import sys


class SdkBootstrapRequired(RuntimeError):
    pass


def main():
    wire = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    # Native installer consoles belong on this worker's stderr, never the wire.
    # No Runtime/global/other-tenant stdout or environment is rebound.
    sys.stdout = sys.stderr
    from hermes_cli import plugins_cmd
    from hermes_cli.config import load_config, save_config
    from hermes_constants import get_hermes_home

    home = str(get_hermes_home().resolve())

    def dispatch(operation, args):
        if operation in {
            "plugins.install",
            "plugins.enable",
            "plugins.update",
            "plugins.remove",
        }:
            from pm._uv import _toolchain

            if (
                not os.environ.get("HERMES_RUNTIME_DIR")
                or _toolchain(realize=False) is None
            ):
                raise SdkBootstrapRequired()
        if operation == "identity":
            import inspect

            return {
                "home": home,
                "module": plugins_cmd.__file__,
                "signatures": {
                    name: str(inspect.signature(getattr(plugins_cmd, name)))
                    for name in (
                        "dashboard_install_plugin",
                        "dashboard_set_agent_plugin_enabled",
                        "dashboard_update_user_plugin",
                        "dashboard_remove_user_plugin",
                    )
                },
            }
        if operation == "storage.seams":
            import importlib.util
            import inspect
            from cron import jobs
            from hermes_cli import kanban_db_connect as kanban_db
            from agent.memory_provider import MemoryProvider
            from agent.context_engine import ContextEngine

            return {
                "cron_provider_module": importlib.util.find_spec(
                    "cron.storage_provider"
                )
                is not None,
                "cron_create": str(inspect.signature(jobs.create_job)),
                "kanban_connect": str(inspect.signature(kanban_db.connect)),
                "kanban_module": kanban_db.__file__,
                "memory_initialize": str(inspect.signature(MemoryProvider.initialize)),
                "context_module": inspect.getfile(ContextEngine),
            }
        if operation == "plugins.inventory":
            enabled = plugins_cmd._get_enabled_set()
            disabled = plugins_cmd._get_disabled_set()
            active = plugins_cmd._category_active_names()
            return [
                dict(
                    id=key,
                    name=name,
                    version=version,
                    description=description,
                    source=source,
                    status=plugins_cmd._plugin_status(
                        name,
                        enabled,
                        disabled,
                        key,
                        source=source,
                        dir_path=path,
                        active=active,
                    ),
                )
                for name, version, description, source, path, key in plugins_cmd._discover_all_plugins()
            ]
        if operation == "plugins.dependency_consent":
            consented, reason = plugins_cmd._consent_python_deps(
                args["name"],
                tuple(args["dependencies"]),
                plugins_cmd._console(),
                assume_yes=False,
            )
            return {"consented": consented, "reason": str(reason)}
        if operation == "plugins.install":
            return plugins_cmd.dashboard_install_plugin(**args)
        if operation == "plugins.enable":
            expected = args.pop("expected_ref", None)
            if (
                expected is not None
                and plugins_cmd.pinned_revision(args["name"]) != expected
            ):
                raise ValueError("PLUGIN_APPROVAL_PIN_MISMATCH")
            return plugins_cmd.dashboard_set_agent_plugin_enabled(**args)
        if operation == "plugins.update":
            if not plugins_cmd.pinned_revision(args["name"]):
                raise ValueError("UNPINNED_PLUGIN_UPDATE_REFUSED")
            return plugins_cmd.dashboard_update_user_plugin(**args)
        if operation == "plugins.remove":
            return plugins_cmd.dashboard_remove_user_plugin(**args)
        if operation == "plugins.list":
            # Pin-bound facade listing; lifecycle/scanning remains upstream-owned.
            return [list(row[:6]) for row in plugins_cmd._discover_all_plugins()]
        if operation == "profile.config":
            return load_config()
        if operation == "profile.configure":
            save_config(args["config"], strip_defaults=False)
            return {"ok": True}
        raise ValueError("UNSUPPORTED_SDK_OPERATION")

    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            result = dispatch(request["operation"], request.get("arguments", {}))
            response = {"id": request["id"], "ok": True, "result": result}
        except Exception as exc:
            # Never send exception strings that may contain credentials/URLs.
            response = {
                "id": request.get("id"),
                "ok": False,
                "error": type(exc).__name__,
            }
        wire.write(json.dumps(response, default=str) + "\n")
        wire.flush()


if __name__ == "__main__":
    main()
