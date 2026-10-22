"""Native ontology callback fences for the existing in-process callback API.

These are real psycopg connections/cursors. SQL is never translated, and this
API fence is not a sandbox against hostile Python or direct libpq access.
"""
from contextlib import contextmanager
from pathlib import Path
import re

import psycopg

from storage.hosted_postgres import database_url,mapped_store,HostedStorageUnavailable


def ontology_owned_tables():
    return frozenset(re.findall(r'CREATE TABLE IF NOT EXISTS\s+(\w+)',
        Path(__file__).with_name('ontology_build_postgres.sql').read_text(),re.I))


class OntologyCursor(psycopg.Cursor):
    def execute(self,query,params=None,*,prepare=None,binary=None):
        self.connection.check_callback_sql(query)
        return super().execute(query,params,prepare=prepare,binary=binary)

    def executemany(self,query,params_seq,*,returning=False):
        self.connection.check_callback_sql(query)
        return super().executemany(query,params_seq,returning=returning)

    def stream(self,query,params=None,*,binary=None,size=1):
        self.connection.check_callback_sql(query)
        return super().stream(query,params,binary=binary,size=size)

    def copy(self,statement,params=None,*,writer=None):
        self.connection.deny_callback_operation('COPY')
        return super().copy(statement,params,writer=writer)


class OntologyConnection(psycopg.Connection):
    _callback_guard=None
    _callback_denied=False

    def cursor(self,name='',*,binary=False,row_factory=None,scrollable=None,withhold=False):
        if name or scrollable is not None or withhold:
            self.deny_callback_operation('server cursor')
            raise ValueError('ontology store uses ordinary native cursors')
        # Always retain the owned cursor fence, including cursors held before a
        # callback. No SQL dialect or SQLite attributes are emulated here.
        cursor=OntologyCursor(self,row_factory=row_factory or self.row_factory)
        if binary:cursor.format=psycopg.pq.Format.BINARY
        return cursor

    def execute(self,query,params=None,*,prepare=None,binary=False):
        self.check_callback_sql(query)
        return super().execute(query,params,prepare=prepare,binary=binary)

    def _deny(self,operation):
        self._callback_denied=True
        raise RuntimeError(f'ontology callback {operation} violates '+self._callback_guard)

    def deny_callback_operation(self,operation):
        if self._callback_guard is not None:self._deny(operation)

    def commit(self):
        self.deny_callback_operation('commit')
        return super().commit()

    def rollback(self):
        self.deny_callback_operation('rollback')
        return super().rollback()

    def close(self):
        self.deny_callback_operation('close')
        return super().close()

    def transaction(self,savepoint_name=None,force_rollback=False):
        self.deny_callback_operation('transaction')
        return super().transaction(savepoint_name,force_rollback)

    @property
    def autocommit(self):
        return psycopg.Connection.autocommit.fget(self)

    @autocommit.setter
    def autocommit(self,value):
        self.deny_callback_operation('autocommit')
        psycopg.Connection.autocommit.fset(self,value)

    def tpc_begin(self,xid):
        self.deny_callback_operation('two-phase transaction')
        return super().tpc_begin(xid)

    def tpc_prepare(self):
        self.deny_callback_operation('two-phase prepare')
        return super().tpc_prepare()

    def tpc_commit(self,xid=None):
        self.deny_callback_operation('two-phase commit')
        return super().tpc_commit(xid)

    def tpc_rollback(self,xid=None):
        self.deny_callback_operation('two-phase rollback')
        return super().tpc_rollback(xid)

    def set_autocommit(self,value):
        self.deny_callback_operation('autocommit')
        return super().set_autocommit(value)

    def check_callback_sql(self,query):
        if self._callback_guard is None:return
        if isinstance(query,psycopg.sql.Composable):query=query.as_string(self)
        if isinstance(query,bytes):query=query.decode('utf-8')
        if not isinstance(query,str):self._deny('unsupported statement')
        # This deliberately accepts a finite SQL surface from owned callbacks.
        # Quoted values cannot introduce commands; comments, dollar quoting,
        # quoted identifiers and statement batches are outside that surface.
        text=re.sub(r"'(?:''|[^'])*'","''",query)
        if any(token in text for token in (';','--','/*','*/','$','"')):
            self._deny('unsupported statement shape')
        # Owned indexes use PostgreSQL's finite INSERT upsert form. Remove
        # only its structural keywords for the command/relation checks below;
        # its assignments, predicates and subqueries remain fully inspected.
        # Likewise accept only the leading SET of the owned UPDATE grammar.
        structure=re.sub(r'\bON\s+CONFLICT\s*(?:\([^()]+\))?\s+DO\s+(?:UPDATE\s+SET|NOTHING)\b',
                         ' ',text,flags=re.I)
        structure=re.sub(r'^(\s*UPDATE\s+[A-Za-z_][A-Za-z_0-9.]*\s+)SET\b',
                         r'\1',structure,flags=re.I)
        words=re.findall(r'[A-Za-z_][A-Za-z_0-9]*',structure)
        if not words:self._deny('empty statement')
        command=words[0].upper()
        if command not in {'SELECT','INSERT','UPDATE','DELETE'}:self._deny('transaction or DDL')
        if self._callback_guard=='read-only' and command!='SELECT':self._deny('write')
        forbidden={'WITH','CREATE','ALTER','DROP','TRUNCATE','COMMIT','ROLLBACK','SAVEPOINT','RELEASE','BEGIN','END','SET','RESET','COPY','CALL','DO','GRANT','REVOKE','MERGE','PREPARE','EXECUTE','VACUUM','ANALYZE','INTO','TABLE','USING'}
        # INTO is needed only by the explicit INSERT grammar.
        if command=='INSERT':forbidden.remove('INTO')
        if forbidden.intersection(w.upper() for w in words):self._deny('unsupported statement shape')
        tables=ontology_owned_tables()
        refs=re.findall(r'\b(?:FROM|JOIN|UPDATE|INTO)\s+([A-Za-z_][A-Za-z_0-9.]*)',structure,re.I)
        if any(name not in tables for name in refs):self._deny('unowned table')
        self._check_relation_grammar(text)
        if command!='SELECT':
            pattern={'INSERT':r'^\s*INSERT\s+INTO\s+(\w+)', 'UPDATE':r'^\s*UPDATE\s+(\w+)', 'DELETE':r'^\s*DELETE\s+FROM\s+(\w+)'}[command]
            target=re.match(pattern,text,re.I)
            if not target or target[1] not in tables:self._deny('unowned mutation')
            if target[1] in {'ontology_build_artifacts','advisory_trial_native_publication_index'}:
                self._deny('private advisory publication index write')
        if re.search(r'\b\w+\.\w+\s*\(',text):self._deny('qualified SQL function')
        functions=set(re.findall(r'\b([A-Za-z_][A-Za-z_0-9]*)\s*\(',text))
        allowed={'count','max','min','coalesce','lower','upper','length','greatest','least','cast','exists','in','not','and','or','where','values','filter','over','conflict'}|set(tables)
        if any(fn.lower() not in allowed for fn in functions):self._deny('unowned SQL function')

    def _check_relation_grammar(self,text):
        # Comma joins hide additional relations from the finite FROM/JOIN
        # grammar above. Require explicit JOIN for every relation. Track clause
        # depth so ordinary SELECT lists and function arguments retain commas.
        depth=0
        relation_depths=set()
        for token in re.findall(r'[A-Za-z_][A-Za-z_0-9]*|[(),]',text):
            upper=token.upper()
            if token=='(':
                depth+=1
            elif token==')':
                relation_depths.discard(depth)
                depth-=1
            elif upper=='FROM':
                relation_depths.add(depth)
            elif upper in {'WHERE','GROUP','HAVING','ORDER','LIMIT','OFFSET','UNION','EXCEPT','INTERSECT','RETURNING'}:
                relation_depths.discard(depth)
            elif token==',' and depth in relation_depths:
                self._deny('comma-separated relations')

    @contextmanager
    def callback_guard(self,*,readonly):
        if self._callback_guard is not None:raise RuntimeError('ontology callback guards cannot nest')
        if self.info.transaction_status!=psycopg.pq.TransactionStatus.INTRANS:
            raise RuntimeError('ontology callback requires an owned transaction')
        self._callback_guard='read-only' if readonly else 'private advisory publication index'
        self._callback_denied=False
        try:
            yield
            if self._callback_denied:raise RuntimeError('ontology callback caught a denied operation')
            if self.info.transaction_status!=psycopg.pq.TransactionStatus.INTRANS:
                raise RuntimeError('ontology callback invalidated the owned transaction')
        finally:
            self._callback_guard=None
            self._callback_denied=False


def connect_ontology(path):
    schema=mapped_store('standalone',path)['schema']
    try:
        return OntologyConnection.connect(database_url(),autocommit=True,
            options=f'-c search_path={schema},pg_catalog -c timezone=UTC')
    except psycopg.Error:
        raise HostedStorageUnavailable('hosted ontology PostgreSQL authority unavailable') from None
