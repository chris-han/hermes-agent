"""Shared persistence primitive for the existing accepted-release origin store."""

from __future__ import annotations

import json
from sqlite3 import Connection
from storage.hosted_postgres import sql_query

from contracts.accepted_graph_release_lineage import (
    AcceptedGraphReleaseLineage,
    AcceptedGraphReleaseLineageV2,
)
from contracts.knowledge_feature_pack import content_hash


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def record_accepted_release_origin(
    connection: Connection,
    *,
    graph_version: str,
    candidate_graph_id: str,
    source_id: str,
    lineage_id: str,
    extraction_run_id: str,
    organization_id: str,
    workspace_id: str,
    source_identity_ref: str | None,
    release_hash: str,
    release_lineage: (
        AcceptedGraphReleaseLineage | AcceptedGraphReleaseLineageV2
        | dict[str, object] | None
    ),
    created_at: str,
    authority_state: str = "authoritative",
) -> None:
    """Idempotently bind one release to exact lineage on the caller's transaction."""
    if authority_state not in {"candidate", "authoritative"}:
        raise ValueError("accepted release origin authority state is invalid")
    if not source_id or lineage_id != source_id:
        raise ValueError("release artifact lineage_id must equal source_id")
    if isinstance(
        release_lineage,
        (AcceptedGraphReleaseLineage, AcceptedGraphReleaseLineageV2),
    ):
        exact_lineage = release_lineage
    elif isinstance(release_lineage, dict):
        exact_lineage = (
            AcceptedGraphReleaseLineageV2.parse_unverified(release_lineage)
            if release_lineage.get("schema_version")
            == "accepted_graph_release_lineage.v2"
            else AcceptedGraphReleaseLineage.from_dict(release_lineage)
        )
    else:
        exact_lineage = None
    if exact_lineage is not None:
        exact_pins = {
            "accepted release ref": (
                exact_lineage.accepted_graph_release_ref,
                graph_version,
            ),
            "accepted release hash": (
                exact_lineage.accepted_graph_release_hash,
                release_hash,
            ),
            "candidate graph": (
                exact_lineage.candidate_graph_id,
                candidate_graph_id,
            ),
            "organization": (exact_lineage.organization_id, organization_id),
            "workspace": (exact_lineage.workspace_id, workspace_id),
            "source identity": (
                exact_lineage.source_identity_ref,
                source_identity_ref,
            ),
            "candidate graph materialization extraction run": (
                exact_lineage.candidate_graph_materialization_extract_run_ref,
                extraction_run_id,
            ),
        }
        for label, (actual, expected) in exact_pins.items():
            if actual != expected:
                raise ValueError(f"{label} mismatch in release artifact origin")
        release_columns = {
            str(row["name"])
            for row in connection.execute(sql_query(
                                              'PRAGMA table_info(tender_graph_releases)',
                                              "SELECT column_name AS name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name='tender_graph_releases'",
                                          ))
        }
        if "lineage_contract" in release_columns:
            release_row = connection.execute(
                sql_query(
                    """SELECT lineage_contract FROM tender_graph_releases
                WHERE graph_version=?""",
                    """SELECT lineage_contract FROM tender_graph_releases
                WHERE graph_version=%s""",
                ),
                (graph_version,),
            ).fetchone()
            expected_contract = (
                "exact_v4_v3"
                if isinstance(exact_lineage, AcceptedGraphReleaseLineageV2)
                else "exact_v3"
            )
            if release_row is not None and release_row["lineage_contract"] != expected_contract:
                raise ValueError(
                    "historical legacy release cannot be enriched with exact lineage"
                )
    payload: dict[str, object] = {
        "artifact_id": graph_version,
        "candidate_graph_id": candidate_graph_id,
        "extraction_run_id": extraction_run_id,
        "lineage_id": lineage_id,
        "organization_id": organization_id,
        "release_hash": release_hash,
        "source_id": source_id,
        "workspace_id": workspace_id,
    }
    release_lineage_json = None
    release_lineage_hash = None
    if exact_lineage is not None:
        release_lineage_json = _json(exact_lineage.as_dict())
        release_lineage_hash = exact_lineage.lineage_hash
        payload["source_identity_ref"] = source_identity_ref
        payload["release_lineage_hash"] = release_lineage_hash
    origin_id = f"origin_{content_hash(payload)[7:31]}"
    existing = connection.execute(
        sql_query(
            """SELECT * FROM contextgraph_artifact_origins
        WHERE artifact_id=? AND organization_id=? AND workspace_id=?
          AND relation_kind='accepted_release'
        ORDER BY created_at, origin_id LIMIT 1""",
            """SELECT * FROM contextgraph_artifact_origins
        WHERE artifact_id=%s AND organization_id=%s AND workspace_id=%s
          AND relation_kind='accepted_release'
        ORDER BY created_at, origin_id LIMIT 1""",
        ),
        (graph_version, organization_id, workspace_id),
    ).fetchone()
    if existing is not None:
        if (
            existing["content_hash"] == content_hash(payload)
            and existing["release_lineage_json"] == release_lineage_json
            and existing["release_lineage_hash"] == release_lineage_hash
        ):
            existing_state = str(existing["authority_state"])
            if existing_state == authority_state:
                return
            if existing_state == "candidate" and authority_state == "authoritative":
                connection.execute(
                    sql_query(
                        """UPDATE contextgraph_artifact_origins
                    SET authority_state='authoritative'
                    WHERE origin_id=? AND authority_state='candidate'""",
                        """UPDATE contextgraph_artifact_origins
                    SET authority_state='authoritative'
                    WHERE origin_id=%s AND authority_state='candidate'""",
                    ),
                    (existing["origin_id"],),
                )
                return
            raise ValueError("accepted release origin authority cannot be downgraded")
        raise ValueError("accepted release origin already exists with different lineage")
    connection.execute(
        sql_query(
            """INSERT INTO contextgraph_artifact_origins
        (origin_id,workspace_id,organization_id,source_id,lineage_id,
         extraction_run_id,candidate_graph_id,artifact_id,source_identity_ref,
         relation_kind,origin_kind,authority_state,release_lineage_json,
         release_lineage_hash,content_hash,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,'accepted_release','attached',
                ?,?,?,?,?)""",
            """INSERT INTO contextgraph_artifact_origins
        (origin_id,workspace_id,organization_id,source_id,lineage_id,
         extraction_run_id,candidate_graph_id,artifact_id,source_identity_ref,
         relation_kind,origin_kind,authority_state,release_lineage_json,
         release_lineage_hash,content_hash,created_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'accepted_release','attached',
                %s,%s,%s,%s,%s)""",
        ),
        (
            origin_id,
            workspace_id,
            organization_id,
            source_id,
            lineage_id,
            extraction_run_id,
            candidate_graph_id,
            graph_version,
            source_identity_ref,
            authority_state,
            release_lineage_json,
            release_lineage_hash,
            content_hash(payload),
            created_at,
        ),
    )


def verify_exact_release_origin_reservation(
    connection: Connection,
    *,
    graph_version: str,
    authority_state: str = "candidate",
) -> str:
    """Verify the exact candidate origin before authority promotion."""
    if authority_state not in {"candidate", "authoritative"}:
        raise ValueError("exact release origin authority state is invalid")
    release = connection.execute(
        sql_query(
            """SELECT candidate_graph_id,content_hash,organization_id,workspace_id,
        lineage_contract FROM tender_graph_releases WHERE graph_version=?""",
            """SELECT candidate_graph_id,content_hash,organization_id,workspace_id,
        lineage_contract FROM tender_graph_releases WHERE graph_version=%s""",
        ),
        (graph_version,),
    ).fetchone()
    if release is None or release["lineage_contract"] not in {
        "exact_v3", "exact_v4_v3"
    }:
        raise ValueError("exact release reservation not found")
    origins = connection.execute(
        sql_query(
            """SELECT * FROM contextgraph_artifact_origins
        WHERE artifact_id=? AND relation_kind='accepted_release'""",
            """SELECT * FROM contextgraph_artifact_origins
        WHERE artifact_id=%s AND relation_kind='accepted_release'""",
        ),
        (graph_version,),
    ).fetchall()
    if len(origins) != 1:
        raise ValueError("exact release candidate origin reservation is missing")
    origin = origins[0]
    if origin["authority_state"] != authority_state:
        raise ValueError(
            f"exact release origin is not {authority_state}"
        )
    if not origin["release_lineage_json"] or not origin["release_lineage_hash"]:
        raise ValueError("exact release candidate origin lineage is missing")
    try:
        lineage_payload = json.loads(origin["release_lineage_json"])
        lineage = (
            AcceptedGraphReleaseLineageV2.parse_unverified(lineage_payload)
            if release["lineage_contract"] == "exact_v4_v3"
            else AcceptedGraphReleaseLineage.from_dict(lineage_payload)
        )
    except (TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("exact release candidate origin lineage is invalid") from exc
    payload = {
        "artifact_id": origin["artifact_id"],
        "candidate_graph_id": origin["candidate_graph_id"],
        "extraction_run_id": origin["extraction_run_id"],
        "lineage_id": origin["lineage_id"],
        "organization_id": origin["organization_id"],
        "release_hash": lineage.accepted_graph_release_hash,
        "source_id": origin["source_id"],
        "workspace_id": origin["workspace_id"],
        "source_identity_ref": origin["source_identity_ref"],
        "release_lineage_hash": origin["release_lineage_hash"],
    }
    if (
        lineage.lineage_hash != origin["release_lineage_hash"]
        or origin["content_hash"] != content_hash(payload)
        or lineage.accepted_graph_release_ref != graph_version
        or lineage.accepted_graph_release_hash != release["content_hash"]
        or lineage.candidate_graph_id != release["candidate_graph_id"]
        or origin["candidate_graph_id"] != release["candidate_graph_id"]
        or lineage.organization_id != release["organization_id"]
        or origin["organization_id"] != release["organization_id"]
        or lineage.workspace_id != release["workspace_id"]
        or origin["workspace_id"] != release["workspace_id"]
        or origin["source_id"] != origin["lineage_id"]
        or lineage.source_identity_ref != origin["source_identity_ref"]
        or lineage.candidate_graph_materialization_extract_run_ref
        != origin["extraction_run_id"]
    ):
        raise ValueError("exact release candidate origin reservation is invalid")
    return str(origin["origin_id"])
