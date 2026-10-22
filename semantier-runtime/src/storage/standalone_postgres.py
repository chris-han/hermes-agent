"""Explicit administration for the standalone event/draft/approval owners."""
from pathlib import Path
import re

import psycopg
from psycopg import sql


def migrate_standalone(dsn,schema,*,app_role=None):
    if not re.fullmatch('[a-z][a-z0-9_]{0,62}',schema):raise ValueError('invalid standalone schema')
    with psycopg.connect(dsn) as c:
        c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('standalone-migrate:'+schema,))
        c.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(schema)))
        c.execute(sql.SQL('SET LOCAL search_path TO {},pg_catalog').format(sql.Identifier(schema)))
        from storage.hosted_auxiliary_paths import auxiliary_schema_sql
        asset=auxiliary_schema_sql()
        c.execute(asset)
        if app_role:
            c.execute(sql.SQL('GRANT USAGE ON SCHEMA {} TO {}').format(sql.Identifier(schema),sql.Identifier(app_role)))
            tables=re.findall(r'CREATE TABLE IF NOT EXISTS (\w+)',asset)
            c.execute(sql.SQL('GRANT SELECT,INSERT,UPDATE,DELETE ON {} TO {}').format(
                sql.SQL(',').join(sql.Identifier(schema,name) for name in tables),sql.Identifier(app_role)))
            sequences=c.execute("SELECT DISTINCT ns.nspname,seq.relname FROM pg_catalog.pg_class seq "
                "JOIN pg_catalog.pg_namespace ns ON ns.oid=seq.relnamespace "
                "JOIN pg_catalog.pg_depend d ON d.objid=seq.oid AND d.classid='pg_class'::regclass "
                "JOIN pg_catalog.pg_class owned ON owned.oid=d.refobjid "
                "WHERE seq.relkind='S' AND d.deptype IN ('a','i') "
                "AND owned.relnamespace=to_regnamespace(%s) AND owned.relname=ANY(%s)", (schema,tables)).fetchall()
            for sequence_schema,sequence_name in sequences:
                c.execute(sql.SQL('GRANT USAGE,SELECT ON SEQUENCE {} TO {}').format(sql.Identifier(sequence_schema,sequence_name),sql.Identifier(app_role)))
            c.execute(sql.SQL('REVOKE INSERT,UPDATE,DELETE ON standalone_imports,benchmark_read_model_schema_metadata FROM {}').format(sql.Identifier(app_role)))
    return {'schema':schema,'owner':'standalone','version':1}


def import_standalone(path,dsn,schema,source_store_id):
    """Import one frozen source with exact history and checked target parity."""
    from datetime import datetime,timezone
    import hashlib,json,sqlite3
    if not re.fullmatch('[a-z][a-z0-9_]{0,62}',schema):raise ValueError('invalid standalone schema')
    if not isinstance(source_store_id,str) or not source_store_id:raise ValueError('explicit stable source_store_id required')
    path=Path(path).resolve()
    if not path.is_file():raise ValueError('standalone source missing')
    from storage.hosted_auxiliary_paths import auxiliary_schema_sql
    # Only these finite checked-in owner families can enter this importer.
    allowed=tuple(name for name in re.findall(r'CREATE TABLE IF NOT EXISTS\s+(\w+)',auxiliary_schema_sql(),re.I)
                  if name not in {'standalone_imports','benchmark_read_model_schema_metadata'})
    source=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    try:
        source.row_factory=sqlite3.Row;source.execute('BEGIN')
        source_user_version=source.execute('PRAGMA user_version').fetchone()[0]
        tables={r[0] for r in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}-{'sqlite_sequence'}
        if tables-set(allowed):raise ValueError('unsupported standalone source tables')
        columns={table:[r[1] for r in source.execute('PRAGMA table_info("'+table+'")')] for table in allowed if table in tables}
        rows={table:[dict(r) for r in source.execute('SELECT * FROM "'+table+'"')] for table in allowed if table in tables}
        if 'semantic_feedback_events' in tables:
            columns['semantic_feedback_events'].append('event_sequence')
            rows['semantic_feedback_events']=[dict(r) for r in source.execute('SELECT *,rowid AS event_sequence FROM semantic_feedback_events')]
    finally:source.close()
    def canonical(value):
        if isinstance(value,datetime):return value.astimezone(timezone.utc).isoformat()
        if isinstance(value,(bytes,bytearray,memoryview)):return {'bytes_hex':bytes(value).hex()}
        return value
    def fingerprint(names,values):
        items=sorted(json.dumps([canonical(row[name]) for name in names],ensure_ascii=False,separators=(',',':'),allow_nan=False) for row in values)
        return hashlib.sha256(json.dumps({'columns':names,'rows':items},ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    source_fingerprints={name:fingerprint(columns[name],values) for name,values in rows.items()}
    source_hash=hashlib.sha256(json.dumps(source_fingerprints,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    receipt={'source_user_version':source_user_version,'source_store_id':source_store_id,'source_fingerprint':source_hash,'table_counts':{k:len(v) for k,v in rows.items()},'source_table_fingerprints':source_fingerprints,'normalizations':[],'target_table_fingerprints':{}}
    if 'semantic_feedback_events' in tables:
        receipt['ordering_bindings']={'semantic_feedback_events':'retained SQLite rowid mapped to PostgreSQL event_sequence'}
    with psycopg.connect(dsn) as c:
        c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('standalone-import:'+schema,))
        c.execute(sql.SQL('SET LOCAL search_path TO {},pg_catalog').format(sql.Identifier(schema)))
        prior=c.execute('SELECT source_store_id,source_fingerprint,receipt_json FROM standalone_imports').fetchall()
        if prior:
            if len(prior)!=1 or prior[0][:2]!=(source_store_id,source_hash) or prior[0][2].get('source_user_version',0)!=source_user_version:raise ValueError('standalone namespace bound to a different source or changed snapshot')
            return {**prior[0][2],'status':'already_imported'}
        for table,values in rows.items():
            if c.execute(sql.SQL('SELECT 1 FROM {} LIMIT 1').format(sql.Identifier(table))).fetchone():raise ValueError('standalone target collision')
            target_columns={column.name for column in c.execute(sql.SQL('SELECT * FROM {} LIMIT 0').format(sql.Identifier(table))).description}
            if set(columns[table])-target_columns:raise ValueError('standalone source schema drift')
            normalized=[]
            for original in values:
                row=dict(original)
                for key,value in row.items():
                    if value and (key.endswith('_at') or key=='timestamp'):
                        stamp=datetime.fromisoformat(str(value).replace('Z','+00:00'))
                        if stamp.tzinfo is None:
                            if table in {'semantic_feedback_events','skill_patch_proposals','effective_skill_versions','graph_schema_discovery_runs','semantic_discovery_decisions','graph_schema_discovery_briefs','llm_proposal_evidence','corpus_graph_schema_sample_results','corpus_graph_schema_feedback_retests'}:
                                raise ValueError('hash-bound auxiliary timestamp requires explicit source normalization policy')
                            stamp=stamp.replace(tzinfo=timezone.utc)
                            row[key]=stamp.isoformat()
                            receipt['normalizations'].append({'table':table,'column':key,'source_row_sha256':fingerprint(columns[table],[original]),'basis':'legacy_naive_utc'})
                        elif stamp.utcoffset()!=timezone.utc.utcoffset(stamp):
                            raise ValueError('non-UTC standalone timestamp requires an explicit source normalization policy')
                        # Typed domain metadata has no artifact/history hash. Text
                        # fields keep already-aware UTC spelling byte-for-byte.
                        if table=='domain_events' and key=='created_at':row[key]=stamp.astimezone(timezone.utc).isoformat()
                c.execute(sql.SQL('INSERT INTO {} ({}) VALUES ({})').format(sql.Identifier(table),sql.SQL(',').join(map(sql.Identifier,row)),sql.SQL(',').join(sql.Placeholder() for _ in row)),tuple(row.values()))
                normalized.append(row)
            names=columns[table]
            actual=[dict(zip(names,row)) for row in c.execute(sql.SQL('SELECT {} FROM {}').format(sql.SQL(',').join(map(sql.Identifier,names)),sql.Identifier(table))).fetchall()]
            expected_hash=fingerprint(names,normalized);actual_hash=fingerprint(names,actual)
            if expected_hash!=actual_hash:raise ValueError('standalone target canonical fingerprint mismatch: '+table)
            receipt['target_table_fingerprints'][table]=actual_hash
        if 'semantic_feedback_events' in tables:
            c.execute("SELECT setval(pg_get_serial_sequence('semantic_feedback_events','event_sequence'),greatest(coalesce((SELECT max(event_sequence) FROM semantic_feedback_events),0),1),(SELECT count(*)>0 FROM semantic_feedback_events))")
        if 'domain_events' in tables:
            c.execute("SELECT setval(pg_get_serial_sequence('domain_events','id'),greatest(coalesce((SELECT max(id) FROM domain_events),0),1),(SELECT count(*)>0 FROM domain_events))")
        c.execute('INSERT INTO standalone_imports(source_store_id,source_path,source_fingerprint,receipt_json) VALUES(%s,%s,%s,%s::jsonb)',(source_store_id,str(path),source_hash,json.dumps(receipt,sort_keys=True)))
    return {**receipt,'status':'imported'}
