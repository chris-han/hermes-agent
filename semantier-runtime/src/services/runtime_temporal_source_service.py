"""Governed byte intake and deterministic normalization for temporal notebooks."""

from __future__ import annotations

from storage.hosted_postgres import sql_query,begin_write,in_transaction

import ast
import base64
from datetime import datetime, timezone
import hashlib
from importlib import resources
import json
import math
from typing import Mapping, TypeVar

from pydantic import BaseModel, ValidationError

from contracts.canonical_source_ir import SourceLocator
from contracts.evidence_location import EvidenceSelector
from contracts.knowledge_classification import RuntimeContextDTO
from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import (
    Evidence,
    Ontology,
    OntologyEntityType,
    OntologyRelationType,
)
from contracts.runtime_context_reconstruction import CapturedJsonV1
from contracts.runtime_temporal_source import (
    RuntimeTemporalEntityV1,
    RuntimeTemporalNormalizationV1,
    RuntimeTemporalRelationshipV1,
    RuntimeTemporalRowProvenanceV1,
    RuntimeTemporalSourceArchiveV1,
    RuntimeTemporalSourceGraphV1,
)
from eos.artifact_hashing import canonical_json, content_hash
from services.runtime_context_transition_service import RuntimeAuthorityResolver
from services.runtime_session_service import GovernedRuntimeSessionResolver
from storage.ontology_build_store import OntologyBuildStore
from services.runtime_errors import RuntimeContextError


_MAX_SOURCE_BYTES = 1_048_576
_MAX_CELLS = 256
_MAX_CELL_BYTES = 65_536
_MAX_AST_NODES = 10_000
_MAX_AST_DEPTH = 32
_MAX_ROWS = 2_000
_NORMALIZER_ID = "semantier.notebook_temporal.v1"
_MEDIA_TYPE = "application/x-ipynb+json"
_TEMPORAL_KERNEL_ONTOLOGY_RECORDED_AT = datetime(2026, 9, 11, tzinfo=timezone.utc)
T = TypeVar("T", bound=BaseModel)


def _temporal_kernel_ontology(*, organization_id: str, workspace_id: str) -> Ontology:
    """Stable system ontology for the temporal source adapter.

    Row-level relationship labels remain source data in RuntimeTemporalSourceGraph;
    the canonical kernel topology uses one governed TEMPORAL_RELATION type rather
    than promoting arbitrary runtime row values into ontology authority.
    """

    return Ontology.create(
        schema_version="semantier.ontology.v1",
        organization_id=organization_id,
        workspace_id=workspace_id,
        version="semantier-runtime-temporal-v1",
        predecessor=None,
        entity_types=(
            OntologyEntityType(type_id="TemporalEntity", label="Temporal Entity"),
        ),
        relation_types=(
            OntologyRelationType(
                type_id="TEMPORAL_RELATION",
                label="Temporal Relation",
                directional=True,
            ),
        ),
        rules=(),
        semantic_frames=(),
        provenance=(),
        recorded_at=_TEMPORAL_KERNEL_ONTOLOGY_RECORDED_AT,
    )


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _captured(value: object) -> CapturedJsonV1:
    encoded = canonical_json(value)
    return CapturedJsonV1(
        media_type="application/json",
        canonical_json=encoded,
        bytes_hash=_sha256_bytes(encoded.encode("utf-8")),
    )


def _source_error(code: str, exc: BaseException | None = None) -> RuntimeContextError:
    error = RuntimeContextError(code)
    if exc is not None:
        error.__cause__ = exc
    return error


def _json_depth_scan(raw: str) -> None:
    depth = 0
    quoted = False
    escaped = False
    for char in raw:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > 32:
                raise _source_error("TEMPORAL_SOURCE_LIMIT_EXCEEDED")
        elif char in "]}":
            depth -= 1
            if depth < 0:
                raise _source_error("TEMPORAL_SOURCE_INVALID")


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _source_error("TEMPORAL_SOURCE_INVALID")
        result[key] = value
    return result


def _parse_notebook(content: bytes) -> dict[str, object]:
    if len(content) > _MAX_SOURCE_BYTES:
        raise _source_error("TEMPORAL_SOURCE_LIMIT_EXCEEDED")
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise _source_error("TEMPORAL_SOURCE_INVALID", exc)
    _json_depth_scan(text)

    def nonfinite(_token: str) -> object:
        raise _source_error("TEMPORAL_SOURCE_INVALID")

    try:
        notebook = json.loads(
            text,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=nonfinite,
        )
    except RuntimeContextError:
        raise
    except (ValueError, RecursionError, MemoryError) as exc:
        raise _source_error("TEMPORAL_SOURCE_INVALID", exc)
    if not isinstance(notebook, dict):
        raise _source_error("TEMPORAL_SOURCE_INVALID")
    nbformat = notebook.get("nbformat")
    minor = notebook.get("nbformat_minor")
    cells = notebook.get("cells")
    if (
        isinstance(nbformat, bool)
        or not isinstance(nbformat, int)
        or nbformat != 4
        or isinstance(minor, bool)
        or not isinstance(minor, int)
        or minor < 0
        or not isinstance(cells, list)
    ):
        raise _source_error("TEMPORAL_SOURCE_INVALID")
    if len(cells) > _MAX_CELLS:
        raise _source_error("TEMPORAL_SOURCE_LIMIT_EXCEEDED")
    return notebook


def _cell_source(notebook: Mapping[str, object], index: int) -> str:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise _source_error("TEMPORAL_SOURCE_INVALID")
    cells = notebook["cells"]
    if not isinstance(cells, list) or index >= len(cells):
        raise _source_error("TEMPORAL_SOURCE_INVALID")
    cell = cells[index]
    if not isinstance(cell, dict) or cell.get("cell_type") != "code":
        raise _source_error("TEMPORAL_SOURCE_INVALID")
    source = cell.get("source")
    if isinstance(source, str):
        joined = source
    elif isinstance(source, list) and all(isinstance(part, str) for part in source):
        joined = "".join(source)
    else:
        raise _source_error("TEMPORAL_SOURCE_INVALID")
    if len(joined.encode("utf-8")) > _MAX_CELL_BYTES:
        raise _source_error("TEMPORAL_SOURCE_LIMIT_EXCEEDED")
    return joined


def _ast_depth(node: ast.AST, depth: int = 1) -> int:
    children = list(ast.iter_child_nodes(node))
    return (
        depth
        if not children
        else max(_ast_depth(child, depth + 1) for child in children)
    )


def _literal(node: ast.AST, *, depth: int = 0) -> object:
    if depth > 16:
        raise _source_error("TEMPORAL_SOURCE_LIMIT_EXCEEDED")
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None or isinstance(value, (bool, str, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise _source_error("TEMPORAL_SOURCE_INVALID")
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(item, depth=depth + 1) for item in node.elts]
    if isinstance(node, ast.Dict):
        result: dict[str, object] = {}
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            if key_node is None:
                raise _source_error("TEMPORAL_SOURCE_INVALID")
            key = _literal(key_node, depth=depth + 1)
            if not isinstance(key, str) or key in result:
                raise _source_error("TEMPORAL_SOURCE_INVALID")
            result[key] = _literal(value_node, depth=depth + 1)
        return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _literal(node.operand, depth=depth + 1)
        if isinstance(operand, bool) or not isinstance(operand, (int, float)):
            raise _source_error("TEMPORAL_SOURCE_INVALID")
        value = -operand if isinstance(node.op, ast.USub) else operand
        if isinstance(value, float) and not math.isfinite(value):
            raise _source_error("TEMPORAL_SOURCE_INVALID")
        return value
    raise _source_error("TEMPORAL_SOURCE_INVALID")


def _collections(
    source: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    try:
        tree = ast.parse(source)
        nodes = list(ast.walk(tree))
        depth = _ast_depth(tree)
    except (SyntaxError, RecursionError, MemoryError) as exc:
        raise _source_error("TEMPORAL_SOURCE_INVALID", exc)
    if len(nodes) > _MAX_AST_NODES or depth > _MAX_AST_DEPTH:
        raise _source_error("TEMPORAL_SOURCE_LIMIT_EXCEEDED")
    assignments: dict[str, list[ast.AST]] = {"entities": [], "relationships": []}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            names = [
                target.id
                for target in statement.targets
                if isinstance(target, ast.Name)
            ]
            for name in set(names) & assignments.keys():
                if len(statement.targets) != 1 or names != [name]:
                    raise _source_error("TEMPORAL_SOURCE_INVALID")
                assignments[name].append(statement.value)
        elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
            target = statement.target
            if isinstance(target, ast.Name) and target.id in assignments:
                raise _source_error("TEMPORAL_SOURCE_INVALID")
    if any(len(values) != 1 for values in assignments.values()):
        raise _source_error("TEMPORAL_SOURCE_INVALID")
    decoded: list[list[dict[str, object]]] = []
    for name in ("entities", "relationships"):
        value = _literal(assignments[name][0])
        if not isinstance(value, list) or not all(
            isinstance(row, dict) for row in value
        ):
            raise _source_error("TEMPORAL_SOURCE_INVALID")
        if len(value) > _MAX_ROWS:
            raise _source_error("TEMPORAL_SOURCE_LIMIT_EXCEEDED")
        decoded.append(value)
    return decoded[0], decoded[1]


def _utc(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise _source_error("TEMPORAL_SOURCE_INVALID")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _source_error("TEMPORAL_SOURCE_INVALID", exc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    if parsed.microsecond:
        raise _source_error("TEMPORAL_SOURCE_INVALID")
    return parsed


class RuntimeTemporalSourceService:
    def __init__(
        self,
        *,
        build_store: OntologyBuildStore,
        authority_resolver: RuntimeAuthorityResolver,
        session_resolver: GovernedRuntimeSessionResolver,
    ) -> None:
        if not isinstance(build_store, OntologyBuildStore):
            raise TypeError("build_store must be OntologyBuildStore")
        if not isinstance(authority_resolver, RuntimeAuthorityResolver):
            raise TypeError("authority_resolver must be RuntimeAuthorityResolver")
        if not isinstance(session_resolver, GovernedRuntimeSessionResolver):
            raise TypeError("session_resolver must be GovernedRuntimeSessionResolver")
        self._build_store = build_store
        self._authority_resolver = authority_resolver
        self._session_resolver = session_resolver
        module_bytes = (
            resources.files("services")
            .joinpath("runtime_temporal_source_service.py")
            .read_bytes()
        )
        self._implementation_hash = _sha256_bytes(module_bytes)

    def _access(self, actor_context: RuntimeContextDTO) -> tuple[str, str, str]:
        self._authority_resolver.require_current_actor(actor_context=actor_context, authorization_class="WORKSPACE_LOCAL")
        workspace = actor_context.workspace_id
        session = actor_context.session_id
        if not workspace:
            raise RuntimeContextError("FORBIDDEN_SCOPE")
        if not session:
            raise RuntimeContextError("FORBIDDEN_SESSION")
        canonical = self._session_resolver.resolve(
            actor_context=actor_context, session_ref=session, authorization_class="WORKSPACE_LOCAL"
        )
        if canonical != session:
            raise RuntimeContextError("FORBIDDEN_SESSION")
        return actor_context.organization_id, workspace, canonical

    def _get(
        self, *, actor_context: RuntimeContextDTO, pin: ArtifactPinV1, model: type[T]
    ) -> T:
        organization, workspace, _session = self._access(actor_context)
        expected_kind = model.ARTIFACT_KIND
        if pin.artifact_kind != expected_kind:
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        raw = self._build_store.get_raw(
            organization_id=organization,
            workspace_id=workspace,
            artifact_ref=pin.artifact_ref,
        )
        if raw is None:
            raise RuntimeContextError("OBJECT_NOT_FOUND")
        if raw["artifact_hash"] != pin.artifact_hash:
            raise RuntimeContextError("HASH_MISMATCH")
        try:
            value = model.model_validate(raw["payload"])
        except (ValidationError, ValueError, TypeError) as exc:
            raise RuntimeContextError("HASH_MISMATCH") from exc
        if (
            raw["schema_version"] != value.schema_version
            or value.artifact_pin() != pin
            or canonical_json(value.model_dump(mode="json"))
            != canonical_json(raw["payload"])
        ):
            raise RuntimeContextError("HASH_MISMATCH")
        recorded_session = getattr(value, "session_ref", None)
        if (
            recorded_session is not None
            and recorded_session != actor_context.session_id
        ):
            raise RuntimeContextError("FORBIDDEN_SESSION")
        return value

    def get_archive(
        self, *, actor_context: RuntimeContextDTO, pin: ArtifactPinV1
    ) -> RuntimeTemporalSourceArchiveV1:
        return self._get(
            actor_context=actor_context, pin=pin, model=RuntimeTemporalSourceArchiveV1
        )

    def get_normalization(
        self, *, actor_context: RuntimeContextDTO, pin: ArtifactPinV1
    ) -> RuntimeTemporalNormalizationV1:
        value = self._get(
            actor_context=actor_context, pin=pin, model=RuntimeTemporalNormalizationV1
        )
        self.get_archive(actor_context=actor_context, pin=value.source)
        return value

    def get_graph(
        self, *, actor_context: RuntimeContextDTO, pin: ArtifactPinV1
    ) -> RuntimeTemporalSourceGraphV1:
        value = self._get(
            actor_context=actor_context, pin=pin, model=RuntimeTemporalSourceGraphV1
        )
        archive = self.get_archive(actor_context=actor_context, pin=value.source)
        normalization = self.get_normalization(
            actor_context=actor_context, pin=value.normalization
        )
        if normalization.source != archive.artifact_pin():
            raise RuntimeContextError("INTEGRITY_UNPROVABLE")
        return value

    def _operation(
        self,
        *,
        actor_context: RuntimeContextDTO,
        operation: str,
        request_id: str,
        request_hash: str,
        artifacts: tuple[BaseModel, ...],
        result_pin: ArtifactPinV1,
        result_model: type[T],
    ) -> T:
        organization, workspace, _session = self._access(actor_context)
        winner: T | None = None
        key = (organization, workspace, actor_context.actor_id, operation, request_id)

        def validate(connection) -> bool:
            nonlocal winner
            rows = connection.execute(
                sql_query("SELECT request_hash, receipt_ref, receipt_hash FROM workspace_operation_request_index "
                "WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?",'SELECT request_hash, receipt_ref, receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s'),
                key,
            ).fetchmany(2)
            if len(rows) > 1:
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            if rows:
                if rows[0][0] != request_hash:
                    raise RuntimeContextError("REQUEST_CONFLICT")
                saved_pin = ArtifactPinV1(
                    artifact_kind=result_pin.artifact_kind,
                    artifact_ref=rows[0][1],
                    artifact_hash=rows[0][2],
                )
                winner = self._get(
                    actor_context=actor_context, pin=saved_pin, model=result_model
                )
                if result_model is RuntimeTemporalSourceGraphV1:
                    winner = self.get_graph(actor_context=actor_context, pin=saved_pin)
                if (
                    winner.actor_id != actor_context.actor_id
                    or winner.request_id != request_id
                    or winner.request_hash != request_hash
                ):
                    raise RuntimeContextError("INTEGRITY_UNPROVABLE")
                return False
            orphan = connection.execute(
                sql_query("SELECT 1 FROM ontology_build_artifacts WHERE organization_id=? AND workspace_id=? "
                "AND schema_version=? AND json_extract(payload_json, '$.actor_id')=? "
                "AND json_extract(payload_json, '$.request_id')=? LIMIT 1","SELECT 1 FROM ontology_build_artifacts WHERE organization_id=%s AND workspace_id=%s AND schema_version=%s AND (payload_json::jsonb #>> '{actor_id}')=%s AND (payload_json::jsonb #>> '{request_id}')=%s LIMIT 1"),
                (
                    organization,
                    workspace,
                    result_model.model_fields["schema_version"].annotation.__args__[0],
                    actor_context.actor_id,
                    request_id,
                ),
            ).fetchone()
            if orphan is not None:
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            return True

        def mutate(connection) -> None:
            connection.execute(
                sql_query("INSERT INTO workspace_operation_request_index "
                "(organization_id,workspace_id,actor_ref,operation,request_id,request_hash,receipt_ref,receipt_hash) "
                "VALUES (?,?,?,?,?,?,?,?)",'INSERT INTO workspace_operation_request_index (organization_id,workspace_id,actor_ref,operation,request_id,request_hash,receipt_ref,receipt_hash) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)'),
                (*key, request_hash, result_pin.artifact_ref, result_pin.artifact_hash),
            )

        self._build_store.atomic_write(
            artifacts=artifacts,
            validate_before_insert=validate,
            mutate_indexes=mutate,
            verify_before_commit=lambda: self._access(actor_context),
        )
        if winner is not None:
            return winner
        return self._get(
            actor_context=actor_context, pin=result_pin, model=result_model
        )

    def capture_notebook(
        self, *, actor_context: RuntimeContextDTO, content: bytes, request_id: str
    ) -> RuntimeTemporalSourceArchiveV1:
        organization, workspace, session = self._access(actor_context)
        if not isinstance(content, bytes):
            raise _source_error("TEMPORAL_SOURCE_INVALID")
        if not isinstance(request_id, str) or not request_id:
            raise _source_error("TEMPORAL_SOURCE_INVALID")
        _parse_notebook(content)
        content_digest = _sha256_bytes(content)
        command = {
            "organization_id": organization,
            "workspace_id": workspace,
            "actor_id": actor_context.actor_id,
            "session_ref": session,
            "request_id": request_id,
            "media_type": _MEDIA_TYPE,
            "content_hash": content_digest,
            "size_bytes": len(content),
        }
        request_hash = "sha256:" + content_hash(command)
        archive = RuntimeTemporalSourceArchiveV1.create(
            schema_version="runtime_temporal_source_archive.v1",
            organization_id=organization,
            workspace_id=workspace,
            session_ref=session,
            media_type=_MEDIA_TYPE,
            content_base64=base64.b64encode(content).decode("ascii"),
            content_hash=content_digest,
            size_bytes=len(content),
            actor_id=actor_context.actor_id,
            request_id=request_id,
            request_hash=request_hash,
            recorded_at=datetime.now(timezone.utc),
        )
        return self._operation(
            actor_context=actor_context,
            operation="CAPTURE_RUNTIME_TEMPORAL_SOURCE",
            request_id=request_id,
            request_hash=request_hash,
            artifacts=(archive,),
            result_pin=archive.artifact_pin(),
            result_model=RuntimeTemporalSourceArchiveV1,
        )

    def import_notebook_source(
        self,
        *,
        actor_context: RuntimeContextDTO,
        source: ArtifactPinV1,
        request_id: str,
    ) -> RuntimeTemporalSourceGraphV1:
        """Import the single temporal-data cell without exposing notebook cell mechanics upstream."""
        archive = self.get_archive(actor_context=actor_context, pin=source)
        content = base64.b64decode(archive.content_base64, validate=True)
        notebook = _parse_notebook(content)
        candidates: list[int] = []
        cells = notebook.get("cells")
        if not isinstance(cells, list):
            raise _source_error("TEMPORAL_SOURCE_INVALID")
        for index, cell in enumerate(cells):
            if not isinstance(cell, dict) or cell.get("cell_type") != "code":
                continue
            try:
                _collections(_cell_source(notebook, index))
            except RuntimeContextError:
                continue
            candidates.append(index)
        if len(candidates) != 1:
            raise _source_error("TEMPORAL_SOURCE_INVALID")
        return self.import_source(
            actor_context=actor_context,
            source=source,
            source_cell_index=candidates[0],
            request_id=request_id,
        )

    def import_source(
        self,
        *,
        actor_context: RuntimeContextDTO,
        source: ArtifactPinV1,
        source_cell_index: int,
        request_id: str,
    ) -> RuntimeTemporalSourceGraphV1:
        organization, workspace, session = self._access(actor_context)
        archive = self.get_archive(actor_context=actor_context, pin=source)
        command = {
            "organization_id": organization,
            "workspace_id": workspace,
            "actor_id": actor_context.actor_id,
            "session_ref": session,
            "request_id": request_id,
            "source": source.model_dump(mode="json"),
            "source_cell_index": source_cell_index,
            "normalizer_id": _NORMALIZER_ID,
            "implementation_hash": self._implementation_hash,
        }
        request_hash = "sha256:" + content_hash(command)
        existing = self._build_store._connection.execute(
            sql_query("SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index "
            "WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND operation=? AND request_id=?",'SELECT request_hash,receipt_ref,receipt_hash FROM workspace_operation_request_index WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND operation=%s AND request_id=%s'),
            (
                organization,
                workspace,
                actor_context.actor_id,
                "IMPORT_RUNTIME_TEMPORAL_SOURCE",
                request_id,
            ),
        ).fetchone()
        if existing is not None:
            if existing[0] != request_hash:
                raise RuntimeContextError("REQUEST_CONFLICT")
            graph = self.get_graph(
                actor_context=actor_context,
                pin=ArtifactPinV1(
                    artifact_kind="runtime_temporal_source_graph",
                    artifact_ref=existing[1],
                    artifact_hash=existing[2],
                ),
            )
            if (
                graph.actor_id != actor_context.actor_id
                or graph.request_id != request_id
                or graph.request_hash != request_hash
            ):
                raise RuntimeContextError("INTEGRITY_UNPROVABLE")
            return graph
        content = base64.b64decode(archive.content_base64, validate=True)
        notebook = _parse_notebook(content)
        cell_source = _cell_source(notebook, source_cell_index)
        entity_rows, relationship_rows = _collections(cell_source)
        cell_hash = _sha256_bytes(cell_source.encode("utf-8"))
        entities: list[RuntimeTemporalEntityV1] = []
        relationships: list[RuntimeTemporalRelationshipV1] = []
        provenance: list[RuntimeTemporalRowProvenanceV1] = []
        entity_ids: set[str] = set()

        def locator(collection: str, row_index: int) -> SourceLocator:
            return SourceLocator(
                source_identity_ref=archive.archive_ref,
                source_hash=archive.content_hash,
                media_type=_MEDIA_TYPE,
                native_provider=None,
                format="ipynb",
                package_part=None,
                native_id=None,
                ordinal_path=["cells", source_cell_index, collection, row_index],
                native_metadata={},
                section_id=None,
                paragraph_id=None,
                table_id=None,
                row_id=None,
                column_id=None,
            )

        for index, row in enumerate(entity_rows):
            object_id = row.get("id")
            if (
                not isinstance(object_id, str)
                or not object_id
                or object_id in entity_ids
            ):
                raise _source_error("TEMPORAL_SOURCE_INVALID")
            entity_ids.add(object_id)
            temporal = {
                key: row[key]
                for key in ("valid_from", "valid_to", "valid_until", "timestamp")
                if key in row
            }
            for value in temporal.values():
                _utc(value)
            properties = {key: value for key, value in row.items() if key != "id"}
            entities.append(
                RuntimeTemporalEntityV1(id=object_id, properties=_captured(properties))
            )
            provenance.append(
                RuntimeTemporalRowProvenanceV1(
                    collection="entities",
                    row_index=index,
                    object_id=object_id,
                    locator=locator("entities", index),
                    original_row=_captured(row),
                    original_temporal_fields=_captured(temporal),
                )
            )

        relationship_ids: set[str] = set()
        typed_keys = {
            "id",
            "source",
            "target",
            "type",
            "valid_from",
            "valid_until",
            "timestamp",
            "valid_to",
        }
        for index, row in enumerate(relationship_rows):
            if "notebook10_provenance" in row:
                raise _source_error("TEMPORAL_SOURCE_INVALID")
            relationship_id = row.get("id", f"n10-r{index:02d}")
            source_id, target_id, relationship_type = (
                row.get("source"),
                row.get("target"),
                row.get("type"),
            )
            if (
                not isinstance(relationship_id, str)
                or not relationship_id
                or relationship_id in relationship_ids
                or not isinstance(source_id, str)
                or not source_id
                or not isinstance(target_id, str)
                or not target_id
                or not isinstance(relationship_type, str)
                or not relationship_type
                or source_id not in entity_ids
                or target_id not in entity_ids
            ):
                raise _source_error("TEMPORAL_SOURCE_INVALID")
            relationship_ids.add(relationship_id)
            temporal = {
                key: row[key]
                for key in ("valid_from", "valid_to", "valid_until", "timestamp")
                if key in row
            }
            if any(not isinstance(value, str) for value in temporal.values()):
                raise _source_error("TEMPORAL_SOURCE_INVALID")
            timestamp = _utc(row["timestamp"]) if "timestamp" in row else None
            if timestamp is not None:
                valid_from = _utc(row.get("valid_from", row["timestamp"]))
                end_candidates = [
                    _utc(row[key]) for key in ("valid_to", "valid_until") if key in row
                ]
                valid_until = end_candidates[0] if end_candidates else timestamp
                if any(
                    value != timestamp
                    for value in (valid_from, valid_until, *end_candidates)
                ):
                    raise _source_error("TEMPORAL_SOURCE_INVALID")
            else:
                if "valid_from" not in row or not (
                    {"valid_to", "valid_until"} & row.keys()
                ):
                    raise _source_error("TEMPORAL_SOURCE_INVALID")
                valid_from = _utc(row["valid_from"])
                end_candidates = [
                    _utc(row[key]) for key in ("valid_to", "valid_until") if key in row
                ]
                if len(set(end_candidates)) != 1:
                    raise _source_error("TEMPORAL_SOURCE_INVALID")
                valid_until = end_candidates[0]
            if valid_from > valid_until:
                raise _source_error("TEMPORAL_SOURCE_INVALID")
            properties = {
                key: value for key, value in row.items() if key not in typed_keys
            }
            properties["notebook10_provenance"] = {
                "source_cell_index": source_cell_index,
                "source_row_index": index,
                "original_row_sha256": hashlib.sha256(
                    canonical_json(row).encode("utf-8")
                ).hexdigest(),
                "original_temporal_fields": temporal,
            }
            relationships.append(
                RuntimeTemporalRelationshipV1(
                    id=relationship_id,
                    source=source_id,
                    target=target_id,
                    type=relationship_type,
                    valid_from=valid_from,
                    valid_until=valid_until,
                    timestamp=timestamp,
                    properties=_captured(properties),
                )
            )
            provenance.append(
                RuntimeTemporalRowProvenanceV1(
                    collection="relationships",
                    row_index=index,
                    object_id=relationship_id,
                    locator=locator("relationships", index),
                    original_row=_captured(row),
                    original_temporal_fields=_captured(temporal),
                )
            )

        normalization = RuntimeTemporalNormalizationV1.create(
            schema_version="runtime_temporal_normalization.v1",
            organization_id=organization,
            workspace_id=workspace,
            source=archive.artifact_pin(),
            source_cell_index=source_cell_index,
            source_cell_hash=cell_hash,
            normalizer_id=_NORMALIZER_ID,
            implementation_hash=self._implementation_hash,
            rows=tuple(provenance),
        )
        selector = EvidenceSelector(
            schema_version="semantier.evidence_selector.v1",
            selector_kind="structure",
            exact_quote=None,
            prefix=None,
            suffix=None,
            normalized_quote=None,
            structural_path=["cells", str(source_cell_index)],
            table=None,
            source_element_ref=f"notebook-cell:{source_cell_index}:{cell_hash}",
        )
        kernel_evidence = Evidence.create(
            schema_version="semantier.evidence.v1",
            organization_id=organization,
            workspace_id=workspace,
            source_ref=archive.archive_ref,
            source_hash=archive.content_hash,
            selector=selector,
            selector_hash=selector.selector_hash(),
            evidence_role="temporal_source_cell",
            provenance=(archive.artifact_pin(), normalization.artifact_pin()),
            observed_at=archive.recorded_at,
            source_time=None,
            recorded_at=archive.recorded_at,
        )
        kernel_ontology = _temporal_kernel_ontology(
            organization_id=organization,
            workspace_id=workspace,
        )
        graph = RuntimeTemporalSourceGraphV1.create(
            schema_version="runtime_temporal_source_graph.v1",
            organization_id=organization,
            workspace_id=workspace,
            session_ref=session,
            source=archive.artifact_pin(),
            normalization=normalization.artifact_pin(),
            ontology=kernel_ontology.pin(),
            evidence=(kernel_evidence.pin(),),
            entities=tuple(entities),
            relationships=tuple(relationships),
            actor_id=actor_context.actor_id,
            request_id=request_id,
            request_hash=request_hash,
            recorded_at=datetime.now(timezone.utc),
        )
        return self._operation(
            actor_context=actor_context,
            operation="IMPORT_RUNTIME_TEMPORAL_SOURCE",
            request_id=request_id,
            request_hash=request_hash,
            artifacts=(normalization, kernel_evidence, kernel_ontology, graph),
            result_pin=graph.artifact_pin(),
            result_model=RuntimeTemporalSourceGraphV1,
        )

    @staticmethod
    def kernel_graph(graph: RuntimeTemporalSourceGraphV1) -> dict[str, object]:
        entities = []
        for row in graph.entities:
            properties = json.loads(row.properties.canonical_json)
            entities.append({"id": row.id, **properties})
        relationships = []
        for row in graph.relationships:
            properties = json.loads(row.properties.canonical_json)
            item = {
                "id": row.id,
                "source": row.source,
                "target": row.target,
                "type": row.type,
                "valid_from": row.valid_from.isoformat(),
                "valid_until": row.valid_until.isoformat(),
                **properties,
            }
            if row.timestamp is not None:
                item["timestamp"] = row.timestamp.isoformat()
            relationships.append(item)
        return {"entities": entities, "relationships": relationships}


__all__ = ["RuntimeTemporalSourceService"]
