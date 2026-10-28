"""PostgreSQL client-operations owner; explicit DDL, no runtime bootstrap."""
from contextlib import contextmanager
from pathlib import Path
import json
import re
import secrets
from uuid import uuid4
import hashlib
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from contracts.client_operations import GRANTS


class Store:
    def __init__(self,dsn,schema='semantier_client_operations'):
        if not re.fullmatch(r'semantier_client_operations(?:_[a-z0-9_]+)?',schema):raise ValueError('invalid client operations schema')
        self.dsn,self.schema=dsn,schema

    @contextmanager
    def connection(self):
        with psycopg.connect(self.dsn,row_factory=dict_row,connect_timeout=5) as c:
            c.execute(sql.SQL('SET LOCAL search_path TO {},pg_catalog').format(sql.Identifier(self.schema)))
            c.execute("SET LOCAL TIME ZONE 'UTC'")
            yield c

    def migrate(self,app_role=None,*,legacy_target_id=None):
        with self.connection() as c:
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(self.schema+'-migration',))
            c.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(self.schema)))
            c.execute(Path(__file__).with_name('schema.sql').read_text())
            legacy = c.execute('SELECT 1 FROM channels ch WHERE NOT EXISTS(SELECT 1 FROM delivery_heads h WHERE h.channel=ch.channel) LIMIT 1').fetchone()
            if legacy:
                from contracts.client_operations import DELIVERY_TARGET_IDS
                if legacy_target_id not in DELIVERY_TARGET_IDS:
                    raise ValueError('verified legacy delivery target is required for migration')
                c.execute('INSERT INTO delivery_heads(target_id,channel,revision,release_id,record) '
                          'SELECT %s,ch.channel,ch.revision,ch.release_id,ch.record FROM channels ch '
                          'WHERE NOT EXISTS(SELECT 1 FROM delivery_heads h WHERE h.channel=ch.channel)', (legacy_target_id,))
            if app_role:
                role=sql.Identifier(app_role);schema=sql.Identifier(self.schema)
                c.execute(sql.SQL('GRANT USAGE ON SCHEMA {} TO {}').format(schema,role))
                c.execute(sql.SQL('GRANT SELECT,INSERT,UPDATE,DELETE ON ALL TABLES IN SCHEMA {} TO {}').format(schema,role))
                for table in ['schema_version','grants','delivery_grants']:
                    c.execute(sql.SQL('REVOKE INSERT,UPDATE,DELETE,TRUNCATE ON {}.{} FROM {}').format(schema,sql.Identifier(table),role))
                c.execute(sql.SQL('REVOKE UPDATE,DELETE,TRUNCATE ON {}.audit FROM {}').format(schema,role))
                c.execute(sql.SQL('REVOKE UPDATE,DELETE,TRUNCATE ON {}.release_delivery_variants FROM {}').format(schema,role))
        return {'schema':self.schema,'version':3}

    def grant(self,user_id,capabilities,channel):
        if not set(capabilities)<=GRANTS:raise ValueError('unsupported capability')
        with self.connection() as c:
            c.execute('DELETE FROM grants WHERE user_id=%s AND channel=%s',(user_id,channel))
            for capability in capabilities:c.execute('INSERT INTO grants VALUES(%s,%s,%s)',(user_id,capability,channel))
            self.audit(c,'migration-owner','grant',user_id,{'capabilities':sorted(capabilities),'channel':channel})

    def permissions(self,user_id,channel):
        with self.connection() as c:
            return {r['capability'] for r in c.execute('SELECT capability FROM grants WHERE user_id=%s AND channel=%s',(user_id,channel))}

    def grant_targets(self, principal_id, target_ids, actions, channel):
        from contracts.client_operations import DELIVERY_ACTIONS, DELIVERY_TARGET_IDS
        if not set(target_ids) <= set(DELIVERY_TARGET_IDS) or not set(actions) <= DELIVERY_ACTIONS:
            raise ValueError('unsupported delivery target grant')
        with self.connection() as c:
            c.execute('DELETE FROM delivery_grants WHERE principal_id=%s AND channel=%s', (principal_id, channel))
            for target_id in sorted(set(target_ids)):
                for action in sorted(set(actions)):
                    c.execute('INSERT INTO delivery_grants VALUES(%s,%s,%s,%s)', (principal_id, channel, target_id, action))
            self.audit(c, 'migration-owner', 'delivery_grant', principal_id,
                       {'channel': channel, 'target_ids': sorted(set(target_ids)), 'actions': sorted(set(actions))})

    def target_permissions(self, principal_id, channel):
        with self.connection() as c:
            rows = c.execute('SELECT target_id,action FROM delivery_grants WHERE principal_id=%s AND channel=%s',
                             (principal_id, channel)).fetchall()
        result = {}
        for row in rows:
            result.setdefault(row['target_id'], set()).add(row['action'])
        return result

    def issue(self,*,kind,channel,expires_at,user_id=None,installation_ref=None):
        with self.connection() as c:
            return self.issue_in(c,kind=kind,channel=channel,expires_at=expires_at,user_id=user_id,installation_ref=installation_ref)

    @staticmethod
    def issue_in(c,*,kind,channel,expires_at,user_id=None,installation_ref=None):
        secret='sco_'+secrets.token_urlsafe(32)
        c.execute('INSERT INTO principals VALUES(%s,%s,%s,%s,%s,%s,%s,NULL)',(hashlib.sha256(secret.encode()).hexdigest(),uuid4().hex,kind,user_id,installation_ref,channel,expires_at))
        return secret

    def principal(self,secret,kind):
        with self.connection() as c:
            return c.execute('SELECT * FROM principals WHERE token_hash=%s AND kind=%s AND revoked_at IS NULL AND expires_at>now()',(hashlib.sha256(secret.encode()).hexdigest(),kind)).fetchone()

    @staticmethod
    def audit(c,actor,action,object_id,body):
        c.execute('INSERT INTO audit(event_id,actor,action,object_id,body) VALUES(%s,%s,%s,%s,%s::jsonb)',(uuid4().hex,actor,action,object_id,json.dumps(body)))

    def retain(self):
        with self.connection() as c:
            c.execute("UPDATE diagnostics SET payload=NULL,deleted_at=now() WHERE payload IS NOT NULL AND expires_at<=now()")
            c.execute("UPDATE authorizations SET ciphertext=NULL,nonce=NULL WHERE expires_at<=now()")
            c.execute("DELETE FROM authorizations WHERE expires_at<now()-interval '30 days'")
            c.execute("DELETE FROM events WHERE received_at<now()-interval '90 days'")
            c.execute("DELETE FROM access_events WHERE received_at<now()-interval '90 days'")
            c.execute("DELETE FROM principals WHERE expires_at<now()-interval '90 days'")
