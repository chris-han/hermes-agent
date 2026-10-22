"""Migration-installed, bounded allocation for a canonical new workspace.

Runtime supplies only the canonical user/workspace pair. The SECURITY DEFINER
function owns a frozen native/execution/auxiliary template and the trusted workspace root;
no runtime credential can select a schema, path, role, or SQL statement.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys

import psycopg
from psycopg import sql


def _template_and_contract(*, app_role, workspace_root):
    native_root = Path(__file__).resolve().parents[2] / "hermes-agent"
    if str(native_root) not in sys.path:
        sys.path.insert(0, str(native_root))
    from hermes_state_postgresql import postgres_scope_schema_sql

    if workspace_root is None:
        from agents.gateway_identity import resolve_workspace_paths

        workspace_root = resolve_workspace_paths("__provisioning__")[1].parent
    root = Path(workspace_root).expanduser().resolve()
    native = postgres_scope_schema_sql()
    from storage.hosted_postgres import execution_schema_sql
    from storage.hosted_auxiliary_paths import auxiliary_schema_sql

    execution = execution_schema_sql()
    auxiliary = auxiliary_schema_sql()
    owned_tables = set()
    for asset in (native, execution, auxiliary):
        tables = set(
            re.findall(r"CREATE TABLE IF NOT EXISTS\s+(\w+)", asset, re.IGNORECASE)
        )
        if owned_tables & tables:
            raise ValueError("workspace schema table family collision")
        owned_tables.update(tables)
    template = native + "\n" + execution + "\n" + auxiliary
    ddl_hash = hashlib.sha256(template.encode()).hexdigest()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # The operator gate pins the allocation policy as well as its DDL assets.
    digest = hashlib.sha256((ddl_hash + ":" + source_hash).encode()).hexdigest()
    return template, {
        "version": 1,
        "template_hash": digest,
        "ddl_template_hash": ddl_hash,
        "allocator_source_hash": source_hash,
        "workspace_root": str(root),
        "app_role": app_role,
        "registry": "semantier_identity.workspace_runtime_bindings",
    }


def expected_workspace_provisioning_contract(
    *, app_role="semantier_app", workspace_root=None
):
    """Read-only operator expectation; the exact builder used by installation."""
    return _template_and_contract(app_role=app_role, workspace_root=workspace_root)[1]


def install_workspace_provisioning(
    dsn: str,
    *,
    app_role: str = "semantier_app",
    workspace_root: str | Path | None = None,
):
    """Explicit migration credential operation; never called by runtime startup."""
    template, expected = _template_and_contract(
        app_role=app_role, workspace_root=workspace_root
    )
    with psycopg.connect(dsn) as c:
        _install_on_connection(c, template, expected)
    return expected


def _install_on_connection(c, template, expected):
    digest = expected["template_hash"]
    root = expected["workspace_root"]
    app_role = expected["app_role"]
    c.execute(
        "SELECT pg_advisory_xact_lock(hashtext('workspace-provisioning-install'))"
    )
    c.execute("""CREATE TABLE IF NOT EXISTS semantier_identity.workspace_provisioning_contract (
        singleton integer PRIMARY KEY CHECK(singleton=1), template_hash text NOT NULL,
        workspace_root text NOT NULL, app_role text NOT NULL);
        CREATE TABLE IF NOT EXISTS semantier_identity.workspace_runtime_bindings (
        user_id text PRIMARY KEY REFERENCES semantier_identity.users(user_id),
        workspace_id text UNIQUE NOT NULL, tenant_id text NOT NULL,
        source_store_id text UNIQUE NOT NULL, schema_name text UNIQUE NOT NULL,
        hermes_home text UNIQUE NOT NULL, created_at timestamptz NOT NULL DEFAULT now());""")
    prior = c.execute(
        "SELECT template_hash,workspace_root,app_role FROM semantier_identity.workspace_provisioning_contract WHERE singleton=1"
    ).fetchone()
    contract = (digest, str(root), app_role)
    if prior is not None and tuple(prior) != contract:
        raise ValueError(
            "workspace provisioning v1 frozen contract changed; explicit migration required"
        )
    c.execute(
        "INSERT INTO semantier_identity.workspace_provisioning_contract VALUES(1,%s,%s,%s) ON CONFLICT DO NOTHING",
        contract,
    )
    _install_history(c, app_role)
    body = sql.SQL(r"""
DECLARE
owned jsonb; intent_org text; tenant text; source_id text; target_schema text;
prior_schema text; prior_path text;
BEGIN
PERFORM pg_catalog.pg_advisory_xact_lock_shared(pg_catalog.hashtext('workspace-provisioning-install'));
IF NOT EXISTS (SELECT 1 FROM semantier_identity.workspace_provisioning_contract WHERE singleton=1 AND template_hash={digest}) THEN
    RAISE EXCEPTION 'workspace allocation contract changed; explicit migration required';
END IF;
IF requested_user IS NULL OR requested_workspace IS NULL
   OR length(requested_user)>128 OR length(requested_workspace)>128
   OR requested_user !~ '^[A-Za-z0-9_-]+$' OR requested_workspace !~ '^[A-Za-z0-9_-]+$' THEN
    RAISE EXCEPTION 'canonical workspace identity required';
END IF;
SELECT u.payload_json::jsonb INTO owned FROM semantier_identity.users u
  WHERE u.user_id=requested_user;
IF owned IS NULL OR owned->>'user_id' IS DISTINCT FROM requested_user
   OR owned->>'workspace_id' IS DISTINCT FROM requested_workspace THEN
    RAISE EXCEPTION 'canonical workspace identity required';
END IF;
PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext('workspace-allocation'),pg_catalog.hashtext(requested_workspace));
SELECT schema_name,hermes_home INTO prior_schema,prior_path
  FROM semantier_identity.workspace_runtime_bindings
  WHERE user_id=requested_user AND workspace_id=requested_workspace;
IF FOUND THEN RETURN prior_schema; END IF;
tenant := 'personal-' || requested_user;
SELECT organization_id INTO intent_org FROM semantier_identity.gateway_projection_intents
  WHERE user_id=requested_user;
IF intent_org IS DISTINCT FROM tenant OR owned->>'organization_id' IS DISTINCT FROM tenant
   OR owned->>'member_role' IS DISTINCT FROM 'owner'
   OR owned->>'membership_status' IS DISTINCT FROM 'active'
   OR NOT EXISTS (SELECT 1 FROM semantier_identity.organizations
       WHERE organization_id=tenant AND payload_json::jsonb->>'created_by'=requested_user)
   OR NOT EXISTS (SELECT 1 FROM jsonb_array_elements(owned->'organization_memberships') m
       WHERE m->>'organization_id'=tenant AND m->>'member_role'='owner'
         AND m->>'membership_status'='active') THEN
    RAISE EXCEPTION 'canonical personal workspace owner required';
END IF;
source_id := 'workspace-native-' || requested_workspace;
target_schema := 'hermes_' || substr(encode(sha256(convert_to(
    '[' || to_json(tenant)::text || ',' || to_json(requested_workspace)::text || ',' || to_json(source_id)::text || ']', 'UTF8')), 'hex'),1,40);
EXECUTE format('CREATE SCHEMA %I',target_schema);
PERFORM set_config('search_path',format('%I,pg_catalog',target_schema),true);
EXECUTE {template};
EXECUTE format('INSERT INTO %I.scope_binding(singleton,tenant_id,workspace_id,source_store_id) VALUES(1,$1,$2,$3)',target_schema)
  USING tenant,requested_workspace,source_id;
EXECUTE format('GRANT USAGE ON SCHEMA %I TO %I',target_schema,{role});
EXECUTE format('GRANT SELECT,INSERT,UPDATE,DELETE ON ALL TABLES IN SCHEMA %I TO %I',target_schema,{role});
EXECUTE format('GRANT USAGE,SELECT ON ALL SEQUENCES IN SCHEMA %I TO %I',target_schema,{role});
EXECUTE format('REVOKE INSERT,UPDATE,DELETE ON %I.scope_binding,%I.migration_receipts,%I.schema_version,%I.hosted_room_legacy_imports,%I.semantier_execution_imports,%I.standalone_imports,%I.benchmark_read_model_schema_metadata FROM %I',target_schema,target_schema,target_schema,target_schema,target_schema,target_schema,target_schema,{role});
INSERT INTO semantier_identity.workspace_runtime_bindings(user_id,workspace_id,tenant_id,source_store_id,schema_name,hermes_home)
  VALUES(requested_user,requested_workspace,tenant,source_id,target_schema,{root} || '/' || requested_workspace);
RETURN target_schema;
END
""").format(
        template=sql.Literal(template),
        role=sql.Literal(app_role),
        root=sql.Literal(str(root)),
        digest=sql.Literal(digest),
    )
    c.execute(
        sql.SQL("""CREATE OR REPLACE FUNCTION semantier_identity.provision_workspace_storage(requested_user text,requested_workspace text)
        RETURNS text LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,semantier_identity AS {}""").format(
            sql.Literal(body.as_string(c))
        )
    )
    c.execute(
        sql.SQL("GRANT USAGE ON SCHEMA semantier_identity TO {}").format(
            sql.Identifier(app_role)
        )
    )
    c.execute(
        "REVOKE ALL ON FUNCTION semantier_identity.provision_workspace_storage(text,text) FROM PUBLIC"
    )
    c.execute(
        sql.SQL(
            "GRANT EXECUTE ON FUNCTION semantier_identity.provision_workspace_storage(text,text) TO {}"
        ).format(sql.Identifier(app_role))
    )
    c.execute(
        sql.SQL(
            "REVOKE ALL ON semantier_identity.workspace_runtime_bindings,semantier_identity.workspace_provisioning_contract FROM {}"
        ).format(sql.Identifier(app_role))
    )
    c.execute(
        sql.SQL(
            "GRANT SELECT ON semantier_identity.workspace_runtime_bindings,semantier_identity.workspace_provisioning_contract TO {}"
        ).format(sql.Identifier(app_role))
    )


def _install_history(c, app_role):
    c.execute("""CREATE TABLE IF NOT EXISTS semantier_identity.workspace_provisioning_contract_history (
        transition_id text PRIMARY KEY, prior_template_hash text NOT NULL,
        template_hash text NOT NULL, workspace_root text NOT NULL, app_role text NOT NULL,
        upgraded_at timestamptz NOT NULL DEFAULT now(),
        CHECK(prior_template_hash<>template_hash));
        CREATE OR REPLACE FUNCTION semantier_identity.workspace_contract_history_append_only()
        RETURNS trigger LANGUAGE plpgsql SET search_path=pg_catalog AS $$
        BEGIN RAISE EXCEPTION 'workspace contract history is append-only'; END $$;
        DROP TRIGGER IF EXISTS workspace_contract_history_immutable ON semantier_identity.workspace_provisioning_contract_history;
        CREATE TRIGGER workspace_contract_history_immutable BEFORE UPDATE OR DELETE OR TRUNCATE
        ON semantier_identity.workspace_provisioning_contract_history FOR EACH STATEMENT
        EXECUTE FUNCTION semantier_identity.workspace_contract_history_append_only();""")
    c.execute(
        "REVOKE ALL ON FUNCTION semantier_identity.workspace_contract_history_append_only() FROM PUBLIC"
    )
    c.execute(
        sql.SQL(
            "REVOKE ALL ON semantier_identity.workspace_provisioning_contract_history FROM PUBLIC,{}"
        ).format(sql.Identifier(app_role))
    )
    c.execute(
        sql.SQL(
            "GRANT SELECT ON semantier_identity.workspace_provisioning_contract_history TO {}"
        ).format(sql.Identifier(app_role))
    )


def allocate_signup_storage(connection, user: dict):
    """Participate in the existing identity/billing transaction, with no DDL here."""
    retained = _retained_static_schema(connection, user)
    if retained is not None:
        return (retained,)
    return connection.execute(
        "SELECT semantier_identity.provision_workspace_storage(%s,%s)",
        (user["user_id"], user["workspace_id"]),
    ).fetchone()


LEGACY_WORKSPACE_CONTRACT_HASH = (
    "2c43ad2287d6f834a8aaa0a5279607b3630703364c6ccb2622d5dfb7264db758"
)
PRE_ACCOUNT_SYNC_WORKSPACE_CONTRACT_HASH = (
    "193866ab138c77118ec87285ca871927ba83f36cbe3c666464ccc6a7046ab679"
)
ACCOUNT_SYNC_WORKSPACE_CONTRACT_HASH = (
    "47c9dc1558bac262c53cb84859b7437b3faaee107a6b0cf408f351e6f3bdff0d"
)


def _static_scope(home, *, workspace=None):
    raw = os.environ.get("HERMES_STATE_SCOPE_MAP")
    if not raw:
        return None
    try:
        mapping = json.loads(raw)
        if not isinstance(mapping, dict):
            raise ValueError
        if workspace is not None and any(
            key != home and isinstance(value, dict) and value.get("workspace_id") == workspace
            for key, value in mapping.items()
        ):
            raise ValueError
        value = mapping.get(home)
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != {
            "tenant_id", "workspace_id", "source_store_id"
        }:
            raise ValueError
        from hermes_state_postgresql import SessionScope

        return SessionScope(**value)
    except (TypeError, ValueError):
        raise ValueError("Invalid explicit retained native scope mapping") from None


def _retained_static_schema(connection, user):
    contract = connection.execute(
        "SELECT workspace_root FROM semantier_identity.workspace_provisioning_contract WHERE singleton=1"
    ).fetchone()
    if contract is None:
        raise ValueError("Workspace storage provisioning migration required")
    root = tuple(contract.values())[0] if hasattr(contract, "values") else contract[0]
    uid, workspace = user.get("user_id"), user.get("workspace_id")
    if any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value)
           for value in (uid, workspace)):
        raise ValueError("Canonical retained workspace identity required")
    home = str(Path(root) / workspace)
    scope = _static_scope(home, workspace=workspace)
    if scope is None:
        return None
    canonical = connection.execute(
        "SELECT payload_json FROM semantier_identity.users WHERE user_id=%s", (uid,)
    ).fetchone()
    payload = tuple(canonical.values())[0] if hasattr(canonical, "values") else canonical[0] if canonical else None
    owned = json.loads(payload) if payload else {}
    if (not isinstance(owned, dict) or owned.get("user_id") != uid or owned.get("workspace_id") != workspace
            or scope.workspace_id != workspace):
        raise ValueError("Retained native scope differs from canonical workspace owner")
    # A runtime caller cannot use a static map to bypass a conflicting protected
    # allocation. Imported homes normally have no allocation-registry entry.
    allocated = connection.execute(
        "SELECT user_id,workspace_id,tenant_id,source_store_id,schema_name,hermes_home FROM semantier_identity.workspace_runtime_bindings "
        "WHERE user_id=%s OR workspace_id=%s OR hermes_home=%s", (uid, workspace, home)
    ).fetchall()
    if allocated:
        bindings = [tuple(row.values()) if hasattr(row, "values") else tuple(row) for row in allocated]
        if bindings != [(uid, workspace, scope.tenant_id, scope.source_store_id, scope.schema, home)]:
            raise ValueError("Retained native scope conflicts with an allocated workspace")
        registered_workspace_scope((*bindings[0], payload), root)
    elif scope.tenant_id != "user:" + uid:
        raise ValueError("Retained native scope differs from canonical workspace owner")
    from hermes_state_postgresql import require_scope_ready

    require_scope_ready(connection, scope)
    return scope.schema


def registered_workspace_scope(row, workspace_root, scope_type=None):
    """Validate only the protected allocator's canonical personal-workspace binding."""
    if scope_type is None:
        from hermes_state_postgresql import SessionScope

        scope_type = SessionScope
    user, workspace, tenant, source, schema, home, payload = row
    try:
        owned = json.loads(payload)
        scope = scope_type(tenant, workspace, source)
        valid = (
            isinstance(owned, dict)
            and owned.get("user_id") == user
            and owned.get("workspace_id") == workspace
            and ((tenant == "personal-" + user and source == "workspace-native-" + workspace)
                 or (tenant == "user:" + user and
                     (source == "workspace-native-" + workspace or _static_scope(home, workspace=workspace) == scope)))
            and schema == scope.schema
            and home == str(Path(workspace_root) / workspace)
            and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", user)
            and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", workspace)
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Invalid allocated workspace registry binding")
    return scope


def _validate_registered_namespaces(c, expected):
    from hermes_state_postgresql import require_scope_ready
    from storage.hosted_postgres import (
        execution_schema_sql,
        require_tables,
        require_meeting_ready,
    )
    from storage.hosted_auxiliary_paths import auxiliary_schema_sql

    rows = c.execute("""SELECT b.user_id,b.workspace_id,b.tenant_id,b.source_store_id,b.schema_name,b.hermes_home,u.payload_json
        FROM semantier_identity.workspace_runtime_bindings b
        LEFT JOIN semantier_identity.users u ON u.user_id=b.user_id""").fetchall()
    scopes = [
        registered_workspace_scope(row, expected["workspace_root"]) for row in rows
    ]
    execution = execution_schema_sql()
    auxiliary = auxiliary_schema_sql()
    from hermes_state_postgresql import postgres_scope_schema_sql

    owned = set(
        re.findall(
            r"CREATE TABLE IF NOT EXISTS\s+(\w+)",
            postgres_scope_schema_sql() + "\n" + execution + "\n" + auxiliary,
            re.I,
        )
    )
    protected = {
        "scope_binding",
        "migration_receipts",
        "schema_version",
        "hosted_room_legacy_imports",
        "semantier_execution_imports",
        "standalone_imports",
        "benchmark_read_model_schema_metadata",
    }
    for scope in scopes:
        c.execute(
            sql.SQL("SET LOCAL search_path TO {},pg_catalog").format(
                sql.Identifier(scope.schema)
            )
        )
        require_scope_ready(c, scope)
        require_tables(c, *owned)
        require_meeting_ready(c)
        if not c.execute(
            "SELECT has_schema_privilege(%s,%s,'USAGE') AND NOT has_schema_privilege(%s,%s,'CREATE')",
            (expected["app_role"], scope.schema, expected["app_role"], scope.schema),
        ).fetchone()[0]:
            raise ValueError(
                "Allocated namespace application schema privileges are not ready"
            )
        for table in owned:
            target = scope.schema + "." + table
            needed = "SELECT" if table in protected else "SELECT,INSERT,UPDATE,DELETE"
            for privilege in needed.split(","):
                if not c.execute(
                    "SELECT has_table_privilege(%s,%s,%s)",
                    (expected["app_role"], target, privilege),
                ).fetchone()[0]:
                    raise ValueError(
                        "Allocated namespace application table privileges are not ready"
                    )
            if (
                table in protected
                and c.execute(
                    "SELECT has_table_privilege(%s,%s,'INSERT,UPDATE,DELETE,TRUNCATE')",
                    (expected["app_role"], target),
                ).fetchone()[0]
            ):
                raise ValueError(
                    "Allocated namespace protected metadata is application-writable"
                )
    return len(scopes)


def upgrade_workspace_provisioning(
    dsn,
    *,
    expected_prior_template_hash,
    expected_current_template_hash,
    app_role="semantier_app",
    workspace_root=None,
):
    """Explicit, quiesced known-predecessor migration; never runtime drift fallback.

    CONNECT fencing spans the atomic history/pointer/function transaction. It is
    restored on ordinary failure; a lost operator connection may deliberately
    leave the candidate inaccessible until explicit operator recovery.
    """
    template, expected = _template_and_contract(
        app_role=app_role, workspace_root=workspace_root
    )
    if expected_prior_template_hash not in {
        LEGACY_WORKSPACE_CONTRACT_HASH, PRE_ACCOUNT_SYNC_WORKSPACE_CONTRACT_HASH,
        ACCOUNT_SYNC_WORKSPACE_CONTRACT_HASH,
    }:
        raise ValueError("Unsupported prior workspace provisioning contract")
    if expected_current_template_hash != expected["template_hash"]:
        raise ValueError(
            "Expected current workspace provisioning contract differs from pinned source"
        )
    desired = (expected["template_hash"], expected["workspace_root"], app_role)
    prior = (expected_prior_template_hash, expected["workspace_root"], app_role)
    with psycopg.connect(dsn, autocommit=True) as c:
        caller, owner = c.execute(
            "SELECT current_user,pg_get_userbyid(nspowner) FROM pg_namespace WHERE nspname='semantier_identity'"
        ).fetchone()
        if caller != owner or caller == app_role:
            raise ValueError(
                "Workspace contract upgrade requires the migration schema owner"
            )
        role = c.execute(
            "SELECT rolsuper,rolcreatedb,rolcreaterole FROM pg_roles WHERE rolname=%s",
            (app_role,),
        ).fetchone()
        if (
            role is None
            or any(role)
            or c.execute(
                "SELECT has_database_privilege(%s,current_database(),'CREATE')",
                (app_role,),
            ).fetchone()[0]
        ):
            raise ValueError(
                "Workspace app role must retain its least-privilege boundary"
            )
        public_connect = c.execute("""SELECT EXISTS(SELECT 1 FROM pg_database d,
            LATERAL aclexplode(COALESCE(d.datacl,acldefault('d',d.datdba))) a
            WHERE d.datname=current_database() AND a.grantee=0 AND a.privilege_type='CONNECT')""").fetchone()[
            0
        ]
        if public_connect:
            raise ValueError(
                "Private database PUBLIC CONNECT must be revoked before upgrade"
            )
        c.execute("SELECT pg_advisory_lock(hashtext('workspace-provisioning-install'))")
        try:
            actual = c.execute(
                "SELECT template_hash,workspace_root,app_role FROM semantier_identity.workspace_provisioning_contract WHERE singleton=1"
            ).fetchone()
            if actual is None or tuple(actual) not in {prior, desired}:
                raise ValueError(
                    "Prior workspace provisioning contract/root/role mismatch"
                )
            had_connect = c.execute(
                """SELECT EXISTS(SELECT 1 FROM pg_database d,
                LATERAL aclexplode(COALESCE(d.datacl,acldefault('d',d.datdba))) a
                WHERE d.datname=current_database() AND a.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s)
                AND a.privilege_type='CONNECT')""",
                (app_role,),
            ).fetchone()[0]
            database = c.execute("SELECT current_database()").fetchone()[0]
            c.execute(
                sql.SQL("REVOKE CONNECT ON DATABASE {} FROM {}").format(
                    sql.Identifier(database), sql.Identifier(app_role)
                )
            )
            try:
                if c.execute(
                    "SELECT has_database_privilege(%s,current_database(),'CONNECT')",
                    (app_role,),
                ).fetchone()[0]:
                    raise ValueError(
                        "Application inherits CONNECT; establish the private migration boundary first"
                    )
                if c.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() AND usename=%s",
                    (app_role,),
                ).fetchone()[0]:
                    raise ValueError(
                        "Quiesce every workspace application connection before contract upgrade"
                    )
                with c.transaction():
                    c.execute(
                        "LOCK TABLE semantier_identity.workspace_runtime_bindings IN ACCESS EXCLUSIVE MODE"
                    )
                    current = c.execute(
                        "SELECT template_hash,workspace_root,app_role FROM semantier_identity.workspace_provisioning_contract WHERE singleton=1 FOR UPDATE"
                    ).fetchone()
                    if tuple(current) != tuple(actual):
                        raise ValueError(
                            "Prior workspace provisioning contract changed concurrently"
                        )
                    count = _validate_registered_namespaces(c, expected)
                    if (
                        _template_and_contract(
                            app_role=app_role, workspace_root=workspace_root
                        )[1]
                        != expected
                    ):
                        raise ValueError(
                            "Workspace provisioning source changed during upgrade"
                        )
                    transition = hashlib.sha256(
                        json.dumps([*prior, *desired], separators=(",", ":")).encode()
                    ).hexdigest()
                    if tuple(actual) == desired:
                        row = c.execute(
                            "SELECT prior_template_hash,template_hash,workspace_root,app_role FROM semantier_identity.workspace_provisioning_contract_history WHERE transition_id=%s",
                            (transition,),
                        ).fetchone()
                        if row != (expected_prior_template_hash, *desired):
                            raise ValueError(
                                "Current contract lacks the exact protected upgrade history"
                            )
                        status = "already_upgraded"
                    else:
                        _install_history(c, app_role)
                        c.execute(
                            "INSERT INTO semantier_identity.workspace_provisioning_contract_history(transition_id,prior_template_hash,template_hash,workspace_root,app_role) VALUES(%s,%s,%s,%s,%s)",
                            (transition, expected_prior_template_hash, *desired),
                        )
                        c.execute(
                            "UPDATE semantier_identity.workspace_provisioning_contract SET template_hash=%s WHERE singleton=1",
                            (expected["template_hash"],),
                        )
                        _install_on_connection(c, template, expected)
                        status = "upgraded"
            finally:
                if had_connect:
                    c.execute(
                        sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                            sql.Identifier(database), sql.Identifier(app_role)
                        )
                    )
        finally:
            c.execute(
                "SELECT pg_advisory_unlock(hashtext('workspace-provisioning-install'))"
            )
    return {
        **expected,
        "status": status,
        "prior_template_hash": expected_prior_template_hash,
        "validated_namespaces": count,
        "history_transition_id": transition,
    }
