from __future__ import annotations

from pathlib import Path
from typing import Any

from .store import GraphKernelStore


class GraphTraversal:
    def __init__(self, db_path: str | Path | None = None, *, _allow_offline_path: bool = False) -> None:
        if db_path is None:
            self.store = GraphKernelStore()
        elif _allow_offline_path:
            self.store = GraphKernelStore.for_offline_path(db_path)
        else:
            self.store = GraphKernelStore(db_path)

    @classmethod
    def for_offline_path(cls, db_path: str | Path) -> GraphTraversal:
        return cls(db_path, _allow_offline_path=True)

    def get_neighbors(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.store.get_neighbors(**kwargs)

    def trace_lineage(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.store.trace_lineage(**kwargs)

    def find_dependencies(self, **kwargs: Any) -> list[dict[str, Any]]:
        kwargs.setdefault("predicates", ["DEPENDS_ON", "CONSUMES", "READS", "CALLS"])
        return self.store.trace_lineage(**kwargs)

    def find_impact(self, **kwargs: Any) -> list[dict[str, Any]]:
        kwargs.setdefault("predicates", ["DEPENDS_ON", "PRODUCES", "CONSUMES", "GOVERNED_BY"])
        return self.store.trace_lineage(**kwargs)

    def find_authority_path(self, **kwargs: Any) -> list[dict[str, Any]]:
        kwargs.setdefault("predicates", ["GOVERNED_BY", "CONSTRAINS", "ACTIVATED_AS", "INCLUDED_IN"])
        kwargs.setdefault("governance_states", ["active"])
        return self.store.trace_lineage(**kwargs)

    def explain_execution(self, **kwargs: Any) -> list[dict[str, Any]]:
        kwargs.setdefault("predicates", ["JUSTIFIED_BY", "VALIDATED_AGAINST", "USED_CONTEXT", "BOUND_TO"])
        kwargs.setdefault("governance_states", ["active"])
        return self.store.trace_lineage(**kwargs)

    def compare_snapshots(self, **kwargs: Any) -> dict[str, Any]:
        return self.store.compare_snapshots(**kwargs)
