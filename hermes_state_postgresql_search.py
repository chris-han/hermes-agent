"""PostgreSQL search over the native canonical transcript."""
import re
from psycopg import sql
from hermes_state_search import SessionSearchMixin


class PostgreSQLSearchMixin:
    _SEARCH_MESSAGE_RESULT_FIELDS=SessionSearchMixin._SEARCH_MESSAGE_RESULT_FIELDS

    def search_messages(self,query,source_filter=None,exclude_sources=None,role_filter=None,limit=20,offset=0,
            sort=None,include_inactive=False,fields=None,after_ts=None,before_ts=None):
        selected=SessionSearchMixin._search_message_fields.__func__(self.__class__,fields)
        if not query or not query.strip(): return []
        tokens=re.findall(r'"[^"]+"|\S+',query.strip())
        terms=[t.strip('"').rstrip('*') for t in tokens if t.upper() not in ('AND','OR','NOT')]
        if not terms or any(not term for term in terms): return []
        cjk=any(ord(ch)>0x2fff for term in terms for ch in term)
        text=sql.SQL("COALESCE(m.content,'') || ' ' || COALESCE(m.tool_name,'') || ' ' || COALESCE(m.tool_calls,'')")
        where,params=[],[]
        if not include_inactive: where.append(sql.SQL('(m.active=1 OR m.compacted=1)'))
        where.append(sql.SQL("COALESCE(m.display_kind,'')<>'hidden'"))
        for column,values,negative in [('source',source_filter,False),('source',exclude_sources,True),('role',role_filter,False)]:
            if values is not None:
                where.append(sql.SQL('{}'+('<>ALL(%s)' if negative else '=ANY(%s)')).format(sql.Identifier('m' if column=='role' else 's',column)))
                params.append(list(values))
        if after_ts is not None: where.append(sql.SQL('s.started_at>=%s')); params.append(after_ts)
        if before_ts is not None: where.append(sql.SQL('s.started_at<%s')); params.append(before_ts)
        base_where=sql.SQL(' AND ').join(where)
        if cjk or (role_filter and 'tool' in role_filter):
            matches=[]; pending='AND'; negate=False
            for token in tokens:
                if token.upper() in ('AND','OR'): pending=token.upper(); continue
                if token.upper()=='NOT': negate=True; continue
                pred=sql.SQL('strpos(lower({}),lower(%s))>0').format(text)
                if negate: pred=sql.SQL('NOT ({})').format(pred); negate=False
                if matches: matches.append(sql.SQL(pending))
                matches.append(pred); params.append(token.strip('"').rstrip('*')); pending='AND'
            match=sql.SQL('(')+sql.SQL(' ').join(matches)+sql.SQL(')')
        else:
            ts=[]
            for token in tokens:
                upper=token.upper()
                if upper in ('AND','OR','NOT'):
                    if ts and ts[-1] not in ('&','|','& !'): ts.append({'AND':'&','OR':'|','NOT':'& !'}[upper])
                    continue
                words=re.findall(r'\w+',token,flags=re.UNICODE)
                if not words: continue
                part=' <-> '.join("'"+word.replace("'","''")+"'" for word in words)
                if token.endswith('*'): part+=':*'
                if ts and ts[-1] not in ('&','|','& !'): ts.append('&')
                ts.append('('+part+')')
            if not ts: return []
            match=sql.SQL("to_tsvector('simple',{}) @@ to_tsquery('simple',%s)").format(text)
            params.append(' '.join(ts))
        direction='ASC' if sort=='oldest' else 'DESC'
        query_sql=sql.SQL('SELECT m.id,m.session_id,m.role,m.content,m.timestamp,m.tool_name,s.source,s.model,s.started_at AS session_started FROM {} m JOIN {} s ON s.id=m.session_id WHERE {} AND {} ORDER BY m.timestamp {},m.id {} LIMIT %s OFFSET %s').format(self.table('messages'),self.table('sessions'),base_where,match,sql.SQL(direction),sql.SQL(direction))
        rows=self._all(query_sql,[*params,limit,offset])
        if not rows and not cjk and not role_filter:
            match=sql.SQL(' AND ').join(sql.SQL('strpos(lower({}),lower(%s))>0').format(text) for _ in terms)
            rows=self._all(sql.SQL('SELECT m.id,m.session_id,m.role,m.content,m.timestamp,m.tool_name,s.source,s.model,s.started_at AS session_started FROM {} m JOIN {} s ON s.id=m.session_id WHERE {} AND {} ORDER BY m.timestamp {},m.id {} LIMIT %s OFFSET %s').format(self.table('messages'),self.table('sessions'),base_where,match,sql.SQL(direction),sql.SQL(direction)),[*params[:-1],*terms,limit,offset])
        output=[]
        for row in rows:
            content=row.pop('content') or ''; anchor=content.casefold().find(terms[0].casefold())
            row['snippet']=content[max(0,anchor-40):max(0,anchor-40)+120]
            context=self.get_messages(row['session_id'],include_inactive=True)
            position=next((i for i,m in enumerate(context) if m['id']==row['id']),0)
            row['context']=[{'role':m['role'],'content':str(m.get('content') or '')[:200]} for m in context[max(0,position-1):position+2]]
            output.append({k:v for k,v in row.items() if selected is None or k in selected})
        return output
