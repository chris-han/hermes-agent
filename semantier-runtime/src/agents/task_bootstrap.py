"""Explicit zero-state task composition migration. Never invoked by app startup.

Existing identity, financial, job, artifact and governance owners retain their
schemas. Workspace allocation records identity bindings, not native transcripts.
"""

from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

SCHEMAS = {
    "execution": "semantier_task_execution",
    "standalone": "semantier_task_artifacts",
    "eos": "semantier_task_eos",
}


def configure():
    if os.environ.get("SEMANTIER_STORAGE_BACKEND") != "postgresql":
        raise RuntimeError("task runtime requires PostgreSQL; no local fallback")
    from storage.hosted_postgres import database_url
    import psycopg

    with psycopg.connect(database_url()) as c:
        row = c.execute(
            "SELECT workspace_root,platform_root FROM semantier_identity.task_runtime_contract WHERE singleton=1"
        ).fetchone()
        if row is None:
            raise RuntimeError("task runtime migration required")
    for key, value in (
        ("SEMANTIER_WORKSPACES_ROOT", row[0]),
        ("SEMANTIER_LOCAL_STATE_DIR", row[1]),
    ):
        if str(Path(os.environ.get(key, value)).resolve()) != value:
            raise RuntimeError(
                "task runtime storage root differs from installed contract"
            )
        os.environ[key] = value
    root = Path(row[1])
    os.environ["SEMANTIER_AUTHORITY_STORE_MAP"] = json.dumps(
        {
            str(root / name): {"owner": owner, "schema": SCHEMAS[owner]}
            for name, owner in (
                ("state.db", "execution"),
                ("artifacts.db", "standalone"),
                ("eos.db", "eos"),
            )
        }
    )
    from agents.auth_db_postgres import ensure_auth_db

    ensure_auth_db()
    from storage.hosted_postgres import connect_authority, require_tables

    for name, owner, table in (
        ("state.db", "execution", "semantier_jobs"),
        ("artifacts.db", "standalone", "ontology_build_artifacts"),
        ("eos.db", "eos", "policy_candidates"),
    ):
        with connect_authority(owner, root / name) as c:
            if owner == "execution":
                from storage.hosted_postgres import execution_schema_sql

                require_tables(
                    c,
                    *re.findall(
                        r"CREATE TABLE IF NOT EXISTS (\w+)", execution_schema_sql()
                    ),
                )
            elif owner == "standalone":
                from storage.ontology_postgres_connection import ontology_owned_tables

                require_tables(c, *ontology_owned_tables())
            else:
                from eos.postgres import require_ready

                require_ready(c)
    return root


def allocate_signup_storage(c, user):
    c.execute(
        "SELECT semantier_identity.provision_workspace_storage(%s,%s)",
        (user["user_id"], user["workspace_id"]),
    )


def migrate(*, dsn, workspace_root, platform_root, app_role="semantier_app"):
    import psycopg
    from psycopg import sql
    from agents.auth_db_postgres import migrate as migrate_identity
    from agents.model_gateway.runtime import (
        migrate as migrate_gateway,
        GatewayConfiguration,
    )
    from storage.hosted_postgres import migrate_execution
    from eos.postgres import migrate_eos

    # Migration credentials are passed explicitly, never written into runtime config.
    prior = os.environ.get("SEMANTIER_DATABASE_URL")
    os.environ["SEMANTIER_DATABASE_URL"] = dsn
    try:
        migrate_identity()
        asyncio.run(
            migrate_gateway(GatewayConfiguration(dsn, "", ""), app_role=app_role)
        )
        migrate_execution(dsn, SCHEMAS["execution"], app_role=app_role)
        migrate_eos(dsn, SCHEMAS["eos"], app_role=app_role)
        root, platform = (
            str(Path(workspace_root).resolve()),
            str(Path(platform_root).resolve()),
        )
        with psycopg.connect(dsn) as c:
            # Refuse reuse of any pre-existing native workspace/history allocation.
            if c.execute(
                "SELECT to_regclass('semantier_identity.workspace_runtime_bindings')"
            ).fetchone()[0]:
                raise ValueError(
                    "fresh task database required; existing allocations preserved"
                )
            ns, role = sql.Identifier(SCHEMAS["standalone"]), sql.Identifier(app_role)
            c.execute(sql.SQL("CREATE SCHEMA {} AUTHORIZATION CURRENT_USER").format(ns))
            c.execute(sql.SQL("SET LOCAL search_path TO {},pg_catalog").format(ns))
            c.execute(
                (
                    Path(__file__).parents[1] / "storage/ontology_build_postgres.sql"
                ).read_text()
            )
            c.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(ns, role))
            c.execute(
                sql.SQL(
                    "GRANT SELECT,INSERT,UPDATE,DELETE ON ALL TABLES IN SCHEMA {} TO {}"
                ).format(ns, role)
            )
            c.execute(
                sql.SQL(
                    "GRANT USAGE,SELECT ON ALL SEQUENCES IN SCHEMA {} TO {}"
                ).format(ns, role)
            )
            c.execute("""CREATE TABLE semantier_identity.task_runtime_contract(
                singleton integer PRIMARY KEY CHECK(singleton=1), workspace_root text NOT NULL, platform_root text NOT NULL);
                CREATE TABLE semantier_identity.workspace_runtime_bindings(
                user_id text PRIMARY KEY REFERENCES semantier_identity.users(user_id), workspace_id text UNIQUE NOT NULL,
                workspace_root text UNIQUE NOT NULL, created_at timestamptz NOT NULL DEFAULT now());""")
            c.execute(
                "INSERT INTO semantier_identity.task_runtime_contract VALUES(1,%s,%s)",
                (root, platform),
            )
            c.execute("""CREATE FUNCTION semantier_identity.provision_workspace_storage(requested_user text, requested_workspace text)
                RETURNS text LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
                DECLARE owned jsonb; root text;
                BEGIN
                IF requested_workspace !~ '^[A-Za-z0-9_-]{1,128}$' THEN RAISE EXCEPTION 'canonical workspace required'; END IF;
                SELECT payload_json::jsonb INTO owned FROM semantier_identity.users WHERE user_id=requested_user;
                IF owned IS NULL OR owned->>'workspace_id' IS DISTINCT FROM requested_workspace
                    OR owned->>'membership_status' IS DISTINCT FROM 'active'
                    OR owned->>'organization_id' IS DISTINCT FROM 'personal-'||requested_user
                    OR owned->>'member_role' IS DISTINCT FROM 'owner' THEN RAISE EXCEPTION 'canonical workspace owner required'; END IF;
                SELECT workspace_root INTO root FROM semantier_identity.task_runtime_contract WHERE singleton=1;
                INSERT INTO semantier_identity.workspace_runtime_bindings(user_id,workspace_id,workspace_root)
                    VALUES(requested_user,requested_workspace,root||'/'||requested_workspace) ON CONFLICT DO NOTHING;
                IF NOT EXISTS(SELECT 1 FROM semantier_identity.workspace_runtime_bindings WHERE user_id=requested_user AND workspace_id=requested_workspace AND workspace_root=root||'/'||requested_workspace)
                    THEN RAISE EXCEPTION 'workspace binding conflict'; END IF;
                RETURN root||'/'||requested_workspace;
                END $$; REVOKE ALL ON FUNCTION semantier_identity.provision_workspace_storage(text,text) FROM PUBLIC;""")
            c.execute(
                sql.SQL(
                    "GRANT EXECUTE ON FUNCTION semantier_identity.provision_workspace_storage(text,text) TO {}"
                ).format(role)
            )
            c.execute(
                sql.SQL("GRANT USAGE ON SCHEMA semantier_identity TO {}").format(role)
            )
            c.execute(
                sql.SQL(
                    "GRANT SELECT,INSERT,UPDATE,DELETE ON ALL TABLES IN SCHEMA semantier_identity TO {}"
                ).format(role)
            )
            c.execute(
                sql.SQL(
                    "REVOKE INSERT,UPDATE,DELETE ON semantier_identity.task_runtime_contract,semantier_identity.workspace_runtime_bindings,semantier_identity.schema_version FROM {}"
                ).format(role)
            )
        return {
            "composition": "independent-tasks.v1",
            "workspace_root": root,
            "platform_root": platform,
            "schemas": SCHEMAS,
        }
    finally:
        if prior is None:
            os.environ.pop("SEMANTIER_DATABASE_URL", None)
        else:
            os.environ["SEMANTIER_DATABASE_URL"] = prior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--platform-root", required=True)
    parser.add_argument("--app-role", default="semantier_app")
    args = parser.parse_args()
    # Explicit operator-owned env; never accepted by HTTP.
    print(
        json.dumps(
            migrate(
                dsn=os.environ["SEMANTIER_MIGRATION_DATABASE_URL"],
                workspace_root=args.workspace_root,
                platform_root=args.platform_root,
                app_role=args.app_role,
            )
        )
    )


if __name__ == "__main__":
    main()
