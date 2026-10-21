"""Real hosted retention follows native archive, lineage, guard and cadence semantics."""
import json
from pathlib import Path
import time
import uuid
import psycopg
from psycopg import sql
import pytest

class PrivateDSN(str):
    def __repr__(self):return '<private test DSN>'

@pytest.fixture
def hosted(tmp_path):
    settings=Path(__file__).resolve().parents[3]/'.artifacts/shared-pg-v1/candidate-pg-private.json'
    if not settings.is_file():pytest.skip('disposable PostgreSQL fixture unavailable')
    dsns={key:PrivateDSN(value) for key,value in json.loads(settings.read_text())['dsns'].items()}
    from hermes_state_postgresql import PostgreSQLSessionDB,SessionScope,provision_scope
    scope=SessionScope('maintenance-tenant','maintenance-workspace',str(uuid.uuid4()))
    provision_scope(dsns['semantier_migrate'],scope,app_role='semantier_app')
    db=PostgreSQLSessionDB(dsns['semantier_app'],scope,home=tmp_path)
    yield db,dsns
    db.close()
    with psycopg.connect(dsns['semantier_migrate']) as conn:conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(scope.schema)))

def stamp(db,sid,at,*,ended=True,reason='done',**values):
    values={'started_at':at,'last_activity_at':at,'ended_at':at if ended else None,'end_reason':reason if ended else None,**values}
    if getattr(db,'backend',None)=='postgresql':
        with db._transaction() as conn:db._update(conn,'sessions',values,'id',sid)
    else:
        db._conn.execute('UPDATE sessions SET '+','.join(key+'=?' for key in values)+' WHERE id=?',(*values.values(),sid));db._conn.commit()

def seed_archive(db,now):
    old=now-10*86400
    for sid in ('old','recent-message','pinned','bot','parent','tip'):
        db.create_session(sid,'api_server');stamp(db,sid,old,ended=False)
    stamp(db,'pinned',old,ended=False,pinned=1)
    stamp(db,'bot',old,ended=False,hidden=1,title=db.CANONICAL_BOT_CHAT_TITLE)
    inherited=json.dumps({'_branched_from':'older-branch-ancestor'})
    stamp(db,'parent',old,reason='compression',model_config=inherited)
    stamp(db,'tip',old,ended=False,parent_session_id='parent',model_config=inherited)
    db.append_message('recent-message','user','fresh message',timestamp=now)

def test_auto_archive_matches_sqlite_and_preserves_auto_archive_provenance(hosted,tmp_path,monkeypatch):
    from hermes_state import SessionDB
    pg,_=hosted;now=time.time()
    with SessionDB(tmp_path/'baseline.sqlite3') as baseline:
        seed_archive(baseline,now);seed_archive(pg,now)
        expected=baseline.maybe_auto_archive(idle_days=3)
        expected_rows={sid:(baseline.get_session(sid)['archived'],baseline.get_session(sid)['auto_archived']) for sid in ('old','recent-message','pinned','bot','parent','tip')}
    import sqlite3
    monkeypatch.setattr(sqlite3,'connect',lambda *a,**kw:pytest.fail('hosted maintenance opened SQLite'))
    assert pg.maybe_auto_archive(idle_days=3)==expected=={'skipped':False,'archived':2}
    assert {sid:(pg.get_session(sid)['archived'],pg.get_session(sid)['auto_archived']) for sid in expected_rows}==expected_rows
    assert pg.maybe_auto_archive(idle_days=3)=={'skipped':True,'archived':0}
    assert not (tmp_path/'state.db').exists()

def test_auto_prune_respects_guards_retention_sources_and_never_rewrites_pg(hosted,tmp_path,monkeypatch):
    pg,_=hosted;now=time.time();old=now-120*86400
    for sid,source,ended in [('remove','cron',True),('guarded','cron',False),('orphan','cron',False),('messaging','telegram',False),('pinned','cron',False),('recent','cron',True)]:
        pg.create_session(sid,source);stamp(pg,sid,old,ended=ended)
    assert pg.try_acquire_session_turn_lease('guarded','owner')
    stamp(pg,'guarded',old)
    stamp(pg,'pinned',old,ended=False,pinned=1)
    pg.create_session('kept-parent','cron');stamp(pg,'kept-parent',old,reason='compression')
    pg.create_session('kept-tip','cron');stamp(pg,'kept-tip',now,ended=False,parent_session_id='kept-parent')
    with pg._transaction() as conn:pg._insert(conn,'messages',{'session_id':'recent','role':'user','content':'new activity','timestamp':now,'message_uid':uuid.uuid4().hex})
    sessions=tmp_path/'sessions';sessions.mkdir();(sessions/'remove.json').write_text('private transcript');(sessions/'guarded.json').write_text('keep')
    import sqlite3,hermes_state_repair,hermes_state_holders
    def forbidden(*a,**kw):pytest.fail('PostgreSQL maintenance invoked a SQLite file operation')
    monkeypatch.setattr(sqlite3,'connect',forbidden)
    monkeypatch.setattr(hermes_state_repair,'_try_acquire_auto_maintenance_lock',forbidden)
    monkeypatch.setattr(hermes_state_holders,'foreign_state_db_holders',forbidden)
    assert pg.maybe_auto_prune_and_vacuum(retention_days=90,sessions_dir=sessions,vacuum=True)=={'skipped':False,'pruned':1,'closed':1,'vacuumed':False}
    assert pg.get_session('remove') is None and not (sessions/'remove.json').exists()
    assert pg.get_session('guarded') is not None and (sessions/'guarded.json').exists()
    assert pg.get_session('recent') is not None
    assert pg.get_session('kept-parent') is not None
    assert pg.get_session('orphan')['end_reason']=='startup_orphan_reap'
    assert pg.get_session('messaging')['ended_at'] is None
    assert pg.get_session('pinned')['ended_at'] is None
    assert pg.maybe_auto_prune_and_vacuum(retention_days=90)['skipped'] is True
    with pg._transaction() as conn:pg._insert(conn,'system_prompts',{'hash':'unreferenced-history','prompt':'retained until an actual prune'})
    assert pg.prune_sessions(older_than_days=90,exclude_active_write_guards=True)==0
    assert pg._one(sql.SQL('SELECT prompt FROM {} WHERE hash=%s').format(pg.table('system_prompts')),('unreferenced-history',)) is not None
    assert pg.maybe_auto_prune_and_vacuum(retention_days=-1)['skipped'] is True


def test_maintenance_admission_and_corrupt_meta_are_scope_local(hosted):
    pg,dsns=hosted
    pg.create_session('old','cli');stamp(pg,'old',time.time()-120*86400)
    pg.set_meta('last_auto_prune','legacy-invalid')
    with psycopg.connect(dsns['semantier_app']) as holder:
        key=pg.scope.schema+'\x1fnative-auto-maintenance'
        holder.execute('SELECT pg_advisory_lock(hashtextextended(%s,0))',(key,))
        assert pg.maybe_auto_prune_and_vacuum()['skipped'] is True
        assert pg.get_session('old') is not None
        holder.execute('SELECT pg_advisory_unlock(hashtextextended(%s,0))',(key,))
    assert pg.maybe_auto_prune_and_vacuum()=={'skipped':False,'pruned':1,'closed':0,'vacuumed':False}
    assert float(pg.get_meta('last_auto_prune'))>0
    now=time.time();old=now-120*86400
    pg.create_session('owned-open','api_server');stamp(pg,'owned-open',old,ended=False)
    with pg._transaction() as conn:pg._insert(conn,'gateway_heartbeats',{'backend_id':'live-owner','pid':1,'started_at':old,'last_heartbeat':now})
    assert pg.sweep_orphaned_sessions(max_idle_seconds=90*86400,sources=('api_server',))==[]
    assert pg.sweep_orphaned_sessions(max_idle_seconds=90*86400,sources=('api_server',),respect_gateway_heartbeats=False)==['owned-open']


def test_archive_flags_match_native_unfiltered_fork_component(hosted,tmp_path):
    from hermes_state import SessionDB
    pg,_=hosted;old=time.time()-10*86400
    def seed(db,ended=False):
        for marker in ('_branched_from','_delegate_from','_reset_from','tool'):
            parent=marker+'-parent';child=marker+'-child'
            db.create_session(parent,'cli');stamp(db,parent,old,reason='compression',archived=0,auto_archived=0)
            db.create_session(child,'tool' if marker=='tool' else 'cli')
            config={} if marker=='tool' else {marker:parent}
            stamp(db,child,old,ended=ended,parent_session_id=parent,model_config=json.dumps(config),archived=0,auto_archived=0)
        db.create_session('recent-sibling','cli');stamp(db,'recent-sibling',time.time(),ended=False,parent_session_id='_branched_from-parent',model_config=json.dumps({'_branched_from':'_branched_from-parent'}),archived=0,auto_archived=0)
    def flags(db):
        ids=[marker+suffix for marker in ('_branched_from','_delegate_from','_reset_from','tool') for suffix in ('-parent','-child')]+['recent-sibling']
        return {sid:(db.get_session(sid)['archived'],db.get_session(sid)['auto_archived']) for sid in ids}
    with SessionDB(tmp_path/'fork-oracle.sqlite3') as baseline:
        seed(baseline);seed(pg)
        assert baseline.archive_stale_sessions(3)==pg.archive_stale_sessions(3)==4
        assert flags(pg)==flags(baseline)
        seed(baseline,ended=True);seed(pg,ended=True)
        assert baseline.archive_sessions(older_than_days=3)==pg.archive_sessions(older_than_days=3)==4
        assert flags(pg)==flags(baseline)
