"""Optional Semantier core integration seams."""

from pathlib import Path


# When Hermes is checked out as Semantier's submodule, keep parent-owned
# runtime modules importable beneath the shared ``agents`` namespace without
# vendoring them into Hermes or eagerly importing runtime state.
_semantier_agents_path = Path(__file__).resolve().parents[2] / "src" / "agents"
if _semantier_agents_path.is_dir():
    __path__.append(str(_semantier_agents_path))
