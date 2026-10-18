"""Provision Hindsight with the PM plugin dependency union."""
from pathlib import Path


def sync_dependencies(*, explicit: bool = False) -> None:
    import pm
    from pm.plugin_inputs import Candidates
    from pm.environments_adopt import adopt_selected
    from pm.paths import repo_root, runtime_facts_path

    pm.sync_venv(explicit=explicit, plugins=Candidates([Path(__file__).resolve().parent]))
    if runtime_facts_path().is_file():
        adopt_selected(repo_root())
