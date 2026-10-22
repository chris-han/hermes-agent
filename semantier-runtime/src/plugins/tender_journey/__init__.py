from pathlib import Path
import json

from agents.plugin_capability_contract import validate_registered_tools
from . import tools


def register(ctx):
    schemas = json.loads(Path(__file__).with_name("schemas.json").read_text(encoding="utf-8"))
    validate_registered_tools(Path(__file__).with_name("plugin.yaml"), schemas, capability_to_tools={
        "resolve_workflow": {"tender_journey_resolve"},
        "execute_workflow": {"tender_journey_action"},
        "approve": set(), "admit": set(), "change_authority_binding": set(), "release_mutation": set(),
    })
    ctx.register_tool(name="tender_journey_resolve", toolset="tender_journey",
                      schema=schemas["tender_journey_resolve"], handler=tools.tender_journey_resolve, is_async=True)
    ctx.register_tool(name="tender_journey_action", toolset="tender_journey",
                      schema=schemas["tender_journey_action"], handler=tools.tender_journey_action, is_async=True)
