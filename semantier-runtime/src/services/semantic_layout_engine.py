"""Deterministic renderer-neutral semantic graph layout."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eos.artifact_hashing import content_hash


_DEFAULT_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "catalogs"
    / "semantic_layout_profile.v1.json"
)


class GraphLayoutLimitExceeded(ValueError):
    """The configured deterministic layout bound was exceeded."""


class SemanticLayoutProfileV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^semantic_layout_profile\.v1$")
    profile_ref: str
    profile_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    engine_version: str
    maximum_node_count: int = Field(gt=0)
    maximum_edge_count: int = Field(ge=0)
    iterations: int = Field(ge=0)
    normalization_min: float
    normalization_max: float
    cluster_anchor_strength: float = Field(ge=0.0)
    repulsion_strength: float = Field(ge=0.0)
    collision_distance: float = Field(gt=0.0)
    default_attraction_weight: float = Field(ge=0.0)
    relation_family_attraction_weights: dict[str, float]
    relation_type_attraction_weights: dict[str, float]

    @model_validator(mode="after")
    def validate_profile(self) -> "SemanticLayoutProfileV1":
        if not self.profile_ref.isascii() or not self.engine_version.isascii():
            raise ValueError("layout profile identities must be ASCII-stable")
        if self.normalization_min >= self.normalization_max:
            raise ValueError("normalization range must be increasing")
        payload = self.model_dump(mode="json")
        payload.pop("profile_hash")
        if self.profile_hash != "sha256:" + content_hash(payload):
            raise ValueError("semantic layout profile hash mismatch")
        return self


@dataclass(frozen=True)
class SemanticCoordinate:
    x: float
    y: float
    z_semantic: float | None
    cluster_id: str


@dataclass(frozen=True)
class SemanticLayoutResult:
    coordinates: dict[str, SemanticCoordinate]
    coordinate_hash: str
    diagnostics: dict[str, Any]


class ClusterResolver:
    """Resolve macro clusters only from governed metadata or stable fallback."""

    _FIELDS = ("bounded_context_id", "domain_id", "schema_id")

    def resolve(self, node: Mapping[str, Any]) -> tuple[str, bool]:
        properties = node.get("properties") or {}
        for field in self._FIELDS:
            value = properties.get(field)
            if isinstance(value, str) and value and value.isascii():
                return value, False
        object_type_id = str(node.get("object_type_id") or "unclassified")
        prefix = object_type_id.split(".", 1)[0]
        return f"fallback:{prefix}", True


class AnchorPlanner:
    def plan(self, cluster_ids: tuple[str, ...]) -> dict[str, tuple[float, float]]:
        if not cluster_ids:
            return {}
        return {
            cluster_id: (
                math.cos(2.0 * math.pi * index / len(cluster_ids)),
                math.sin(2.0 * math.pi * index / len(cluster_ids)),
            )
            for index, cluster_id in enumerate(cluster_ids)
        }


class LayerResolver:
    def resolve(self, node: Mapping[str, Any]) -> float | None:
        value = (node.get("properties") or {}).get("semantic_layer")
        if value is None:
            return None
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("semantic_layer must be finite")
        return number


class ConstrainedForceRefiner:
    def __init__(self, profile: SemanticLayoutProfileV1) -> None:
        self._profile = profile

    def refine(
        self,
        positions: dict[str, list[float]],
        clusters: Mapping[str, str],
        anchors: Mapping[str, tuple[float, float]],
        edges: tuple[dict[str, Any], ...],
    ) -> None:
        ids = tuple(positions)
        for _ in range(self._profile.iterations):
            delta = {node_id: [0.0, 0.0] for node_id in ids}
            for index, left in enumerate(ids):
                for right in ids[index + 1 :]:
                    dx = positions[left][0] - positions[right][0]
                    dy = positions[left][1] - positions[right][1]
                    distance_sq = max(dx * dx + dy * dy, 1e-9)
                    distance = math.sqrt(distance_sq)
                    force = self._profile.repulsion_strength / distance_sq
                    if distance < self._profile.collision_distance:
                        force *= 2.0
                    ux, uy = dx / distance, dy / distance
                    delta[left][0] += ux * force
                    delta[left][1] += uy * force
                    delta[right][0] -= ux * force
                    delta[right][1] -= uy * force
            for edge in edges:
                source, target = edge["source"], edge["target"]
                if source not in positions or target not in positions:
                    continue
                weight = self._profile.relation_type_attraction_weights.get(
                    edge["relation_type_id"],
                    self._profile.relation_family_attraction_weights.get(
                        edge["relation_family_id"],
                        self._profile.default_attraction_weight,
                    ),
                )
                dx = positions[target][0] - positions[source][0]
                dy = positions[target][1] - positions[source][1]
                delta[source][0] += dx * weight
                delta[source][1] += dy * weight
                delta[target][0] -= dx * weight
                delta[target][1] -= dy * weight
            for node_id in ids:
                anchor = anchors[clusters[node_id]]
                delta[node_id][0] += (
                    anchor[0] - positions[node_id][0]
                ) * self._profile.cluster_anchor_strength
                delta[node_id][1] += (
                    anchor[1] - positions[node_id][1]
                ) * self._profile.cluster_anchor_strength
                positions[node_id][0] += max(-0.1, min(0.1, delta[node_id][0]))
                positions[node_id][1] += max(-0.1, min(0.1, delta[node_id][1]))


class LayoutNormalizer:
    def __init__(self, minimum: float, maximum: float) -> None:
        self._minimum = minimum
        self._maximum = maximum

    def normalize(self, positions: Mapping[str, list[float]]) -> dict[str, tuple[float, float]]:
        if not positions:
            return {}
        xs = [position[0] for position in positions.values()]
        ys = [position[1] for position in positions.values()]
        return {
            node_id: (
                self._axis(position[0], min(xs), max(xs)),
                self._axis(position[1], min(ys), max(ys)),
            )
            for node_id, position in positions.items()
        }

    def _axis(self, value: float, low: float, high: float) -> float:
        if high == low:
            return 0.0
        ratio = (value - low) / (high - low)
        return self._minimum + ratio * (self._maximum - self._minimum)


class SemanticLayoutEngine:
    def __init__(self, profile: SemanticLayoutProfileV1) -> None:
        self.profile = profile
        self.cluster_resolver = ClusterResolver()
        self.anchor_planner = AnchorPlanner()
        self.layer_resolver = LayerResolver()
        self.force_refiner = ConstrainedForceRefiner(profile)
        self.normalizer = LayoutNormalizer(
            profile.normalization_min, profile.normalization_max
        )

    def layout(
        self,
        *,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        coordinate_seed: str,
    ) -> SemanticLayoutResult:
        if len(nodes) > self.profile.maximum_node_count:
            raise GraphLayoutLimitExceeded(
                f"node count {len(nodes)} exceeds {self.profile.maximum_node_count}"
            )
        if len(edges) > self.profile.maximum_edge_count:
            raise GraphLayoutLimitExceeded(
                f"edge count {len(edges)} exceeds {self.profile.maximum_edge_count}"
            )
        ordered_nodes = tuple(sorted(nodes, key=lambda node: str(node["id"]).encode()))
        ordered_edges = tuple(
            sorted(edges, key=lambda edge: str(edge["relation_instance_ref"]).encode())
        )
        node_ids = [str(node["id"]) for node in ordered_nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("layout nodes contain duplicate identities")
        clusters: dict[str, str] = {}
        fallback: list[str] = []
        layers: dict[str, float | None] = {}
        for node in ordered_nodes:
            node_id = str(node["id"])
            clusters[node_id], used_fallback = self.cluster_resolver.resolve(node)
            layers[node_id] = self.layer_resolver.resolve(node)
            if used_fallback:
                fallback.append(node_id)
        anchors = self.anchor_planner.plan(
            tuple(sorted(set(clusters.values()), key=lambda item: item.encode()))
        )
        positions = {
            node_id: self._initial_position(node_id, coordinate_seed, anchors[clusters[node_id]])
            for node_id in node_ids
        }
        self.force_refiner.refine(positions, clusters, anchors, ordered_edges)
        normalized = self.normalizer.normalize(positions)
        coordinates = {
            node_id: SemanticCoordinate(
                x=normalized[node_id][0],
                y=normalized[node_id][1],
                z_semantic=layers[node_id],
                cluster_id=clusters[node_id],
            )
            for node_id in node_ids
        }
        coordinate_payload = [
            {
                "id": node_id,
                "x": coordinate.x,
                "y": coordinate.y,
                "z_semantic": coordinate.z_semantic,
                "cluster_id": coordinate.cluster_id,
            }
            for node_id, coordinate in coordinates.items()
        ]
        return SemanticLayoutResult(
            coordinates=coordinates,
            coordinate_hash="sha256:" + content_hash(coordinate_payload),
            diagnostics={"fallback_cluster_node_ids": fallback},
        )

    @staticmethod
    def _initial_position(
        node_id: str, seed: str, anchor: tuple[float, float]
    ) -> list[float]:
        digest = sha256(f"{seed}\0{node_id}".encode()).digest()
        jitter_x = (int.from_bytes(digest[:8], "big") / (2**64 - 1) - 0.5) * 0.2
        jitter_y = (int.from_bytes(digest[8:16], "big") / (2**64 - 1) - 0.5) * 0.2
        return [anchor[0] + jitter_x, anchor[1] + jitter_y]


def load_default_semantic_layout_profile() -> SemanticLayoutProfileV1:
    raw = json.loads(_DEFAULT_PROFILE.read_text(encoding="utf-8"))
    raw["profile_hash"] = "sha256:" + content_hash(raw)
    return SemanticLayoutProfileV1.model_validate(raw)
