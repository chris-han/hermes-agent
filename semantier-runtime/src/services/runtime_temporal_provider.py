"""Private in-memory execution kernel for the closed temporal provider."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import re

from contracts.runtime_semantic_evaluation import (
    RuntimeTemporalProviderConfigV1,
    TemporalQueryV1,
)
from semantica.kg import TemporalGraphQuery, TemporalPatternDetector
from semantica.kg.temporal_reasoning import (
    TemporalInterval,
    TemporalReasoningEngine,
)


_ERROR_PREFIX = "TEMPORAL_KERNEL_INVALID_INPUT:"
_UTC_SECOND_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|\+00:00)\Z")


@dataclass(frozen=True)
class _TemporalKernelCapture:
    raw_output: dict[str, object] | list[dict[str, object]]
    selected_relationships: tuple[dict[str, object], ...] | None
    world_snapshot: dict[str, object] | None
    limitations: tuple[str, ...]


def _invalid(reason: str) -> ValueError:
    return ValueError(f"{_ERROR_PREFIX} {reason}")


def _copy_finite_json(
    value: object, path: str, active_container_ids: set[int] | None = None
) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _invalid(f"{path} must contain finite JSON values")
        return value
    if isinstance(value, (Mapping, list, tuple)):
        active = active_container_ids if active_container_ids is not None else set()
        container_id = id(value)
        if container_id in active:
            raise _invalid(f"{path} contains a cyclic container")
        active.add(container_id)
        try:
            if isinstance(value, Mapping):
                copied: dict[str, object] = {}
                for key, item in value.items():
                    if type(key) is not str:
                        raise _invalid(f"{path} mapping key must be a string")
                    copied[key] = _copy_finite_json(item, f"{path}.{key}", active)
                return copied
            if isinstance(value, list):
                return [
                    _copy_finite_json(item, f"{path}[{index}]", active)
                    for index, item in enumerate(value)
                ]
            return tuple(
                _copy_finite_json(item, f"{path}[{index}]", active)
                for index, item in enumerate(value)
            )
        finally:
            active.remove(container_id)
    raise _invalid(f"{path} must contain only finite JSON data")


def _strict_nonempty_string(value: object, path: str) -> str:
    if type(value) is not str or not value:
        raise _invalid(f"{path} must be a nonempty string")
    return value


def _parse_utc_second(value: object, path: str) -> datetime:
    if type(value) is not str or not _UTC_SECOND_PATTERN.fullmatch(value):
        raise _invalid(
            f"{path} must be a full UTC ISO-8601 timestamp at whole-second precision"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _invalid(f"{path} must be a valid UTC ISO-8601 timestamp") from exc
    if parsed.utcoffset() != timedelta(0) or parsed.microsecond:
        raise _invalid(f"{path} must use UTC whole-second precision")
    return parsed.astimezone(timezone.utc)


def _validate_and_copy_graph(graph: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(graph, Mapping):
        raise _invalid("graph must be a mapping")
    for required in ("entities", "relationships"):
        if required not in graph:
            raise _invalid(f"{required} is required")
        if not isinstance(graph[required], (list, tuple)):
            raise _invalid(f"{required} must be a list or tuple")

    copied = _copy_finite_json(graph, "graph")
    assert isinstance(copied, dict)
    entities = copied["entities"]
    relationships = copied["relationships"]
    assert isinstance(entities, (list, tuple))
    assert isinstance(relationships, (list, tuple))

    entity_ids: set[str] = set()
    for index, entity in enumerate(entities):
        path = f"entities[{index}]"
        if not isinstance(entity, Mapping):
            raise _invalid(f"{path} must be a mapping row")
        entity_id = _strict_nonempty_string(entity.get("id"), f"{path}.id")
        if entity_id in entity_ids:
            raise _invalid(f"duplicate entity id {entity_id!r}")
        entity_ids.add(entity_id)

    relationship_ids: set[str] = set()
    for index, relationship in enumerate(relationships):
        path = f"relationships[{index}]"
        if not isinstance(relationship, Mapping):
            raise _invalid(f"{path} must be a mapping row")
        relationship_id = _strict_nonempty_string(relationship.get("id"), f"{path}.id")
        if relationship_id in relationship_ids:
            raise _invalid(f"duplicate relationship id {relationship_id!r}")
        relationship_ids.add(relationship_id)

        source = _strict_nonempty_string(relationship.get("source"), f"{path}.source")
        target = _strict_nonempty_string(relationship.get("target"), f"{path}.target")
        _strict_nonempty_string(relationship.get("type"), f"{path}.type")
        if source not in entity_ids:
            raise _invalid(f"{path}.source does not identify an entity")
        if target not in entity_ids:
            raise _invalid(f"{path}.target does not identify an entity")
        if "valid_to" in relationship:
            raise _invalid(f"{path}.valid_to is not a normalized temporal field")

        valid_from = _parse_utc_second(
            relationship.get("valid_from"), f"{path}.valid_from"
        )
        valid_until = _parse_utc_second(
            relationship.get("valid_until"), f"{path}.valid_until"
        )
        if valid_from > valid_until:
            raise _invalid(f"{path}.bounds must satisfy valid_from <= valid_until")
        if "timestamp" in relationship:
            timestamp = _parse_utc_second(
                relationship["timestamp"], f"{path}.timestamp"
            )
            if timestamp != valid_from or timestamp != valid_until:
                raise _invalid(f"{path}.timestamp must equal both temporal bounds")

    return copied


def _validate_query_precision(query: TemporalQueryV1) -> None:
    for field in ("at_time", "start_time", "end_time"):
        value = getattr(query, field)
        if value is not None and value.microsecond != 0:
            raise _invalid(f"query.{field} must use whole-second precision")


def _detached(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _detached(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detached(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detached(item) for item in value)
    return copy.deepcopy(value)


def _relationship_selection(
    raw_output: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    relationships = raw_output.get("relationships", [])
    assert isinstance(relationships, (list, tuple))
    selected = _detached(tuple(relationships))
    assert isinstance(selected, tuple)
    return selected


def _strict_range_selection(
    relationships: list[object] | tuple[object, ...],
    start: datetime,
    end: datetime,
) -> tuple[dict[str, object], ...]:
    if start == end:
        return ()
    query_interval = TemporalInterval(start=start, end=end)
    reasoning = TemporalReasoningEngine()
    selected = []
    for relationship in relationships:
        assert isinstance(relationship, Mapping)
        candidate = TemporalInterval(
            start=_parse_utc_second(
                relationship["valid_from"], "relationship.valid_from"
            ),
            end=_parse_utc_second(
                relationship["valid_until"], "relationship.valid_until"
            ),
        )
        if reasoning.overlaps(candidate, query_interval):
            selected.append(_detached(relationship))
    return tuple(selected)


def _execute_temporal_kernel(
    *,
    query: TemporalQueryV1,
    configuration: RuntimeTemporalProviderConfigV1,
    graph: Mapping[str, object],
) -> _TemporalKernelCapture:
    """Execute a validated normalized graph through the real temporal library."""

    graph_copy = _validate_and_copy_graph(graph)
    _validate_query_precision(query)

    native_output: dict[str, object] | list[dict[str, object]]
    selected: tuple[dict[str, object], ...] | None = None
    snapshot: dict[str, object] | None = None

    if query.operation in ("point", "world_snapshot"):
        engine = TemporalGraphQuery(
            temporal_granularity=configuration.temporal_granularity
        )
        if query.include_boundaries:
            native_output = engine.query_time_range(
                graph_copy,
                query.query_text,
                query.at_time,
                query.at_time,
            )
        else:
            native_output = engine.query_at_time(
                graph_copy,
                query.query_text,
                query.at_time,
            )
        selected = _relationship_selection(native_output)
        if query.operation == "world_snapshot":
            detached_graph = _detached(graph_copy)
            assert isinstance(detached_graph, dict)
            detached_entities = _detached(list(graph_copy["entities"]))
            detached_relationships = _detached(list(selected))
            assert isinstance(detached_entities, list)
            assert isinstance(detached_relationships, list)
            detached_graph["entities"] = detached_entities
            detached_graph["relationships"] = detached_relationships
            snapshot = detached_graph
    elif query.operation == "range":
        engine = TemporalGraphQuery(
            temporal_granularity=configuration.temporal_granularity
        )
        native_output = engine.query_time_range(
            graph_copy,
            query.query_text,
            query.start_time,
            query.end_time,
        )
        if query.include_boundaries:
            selected = _relationship_selection(native_output)
        else:
            assert query.start_time is not None and query.end_time is not None
            relationships = graph_copy["relationships"]
            assert isinstance(relationships, (list, tuple))
            selected = _strict_range_selection(
                relationships, query.start_time, query.end_time
            )
    elif query.operation == "path":
        engine = TemporalGraphQuery(
            temporal_granularity=configuration.temporal_granularity
        )
        assert query.source_id is not None and query.target_id is not None
        native_output = engine.find_temporal_paths(
            graph_copy,
            query.source_id,
            query.target_id,
            query.start_time,
            query.end_time,
        )
    elif query.operation == "pattern":
        detector = TemporalPatternDetector()
        relationships = graph_copy["relationships"]
        assert isinstance(relationships, (list, tuple))
        canonical_graph = _detached(graph_copy)
        assert isinstance(canonical_graph, dict)
        canonical_graph["relationships"] = sorted(
            relationships,
            key=lambda row: (row["valid_from"], row["id"]),
        )
        assert query.pattern_type is not None and query.min_frequency is not None
        native_output = detector.detect_temporal_patterns(
            canonical_graph,
            pattern_type=query.pattern_type,
            min_frequency=query.min_frequency,
            gap_tolerance=timedelta(
                seconds=configuration.pattern_gap_tolerance_seconds
            ),
        )
    else:
        engine = TemporalGraphQuery(
            temporal_granularity=configuration.temporal_granularity
        )
        native_output = engine.analyze_evolution(
            graph_copy,
            start_time=query.start_time,
            end_time=query.end_time,
        )

    detached_raw = _detached(native_output)
    assert isinstance(detached_raw, (dict, list))
    limitations = []
    if query.query_text:
        limitations.append("NATURAL_LANGUAGE_QUERY_NOT_APPLIED")
    if query.operation == "world_snapshot":
        limitations.append("ENTITY_LIFETIME_FILTERING_NOT_APPLIED")
    return _TemporalKernelCapture(
        raw_output=detached_raw,
        selected_relationships=selected,
        world_snapshot=snapshot,
        limitations=tuple(limitations),
    )
