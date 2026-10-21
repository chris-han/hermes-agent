"""PostgreSQL search over canonical rows, with native BM25 and fallback ordering."""
import json
import re
import unicodedata
from psycopg import sql
from hermes_state_search import SessionSearchMixin, _strip_cjk_wildcards, _non_operator_tokens, _flatten_text
from hermes_state_common import FTS_TOOL_CONTENT_PREFIX_CHARS, FTS_TRIGRAM_EXCLUDED_SOURCES


_FIELDS=('content','tool_name','tool_calls')


def _words(text):
    normalized=''.join(c for c in unicodedata.normalize('NFD',text.lower()) if not '\u0300'<=c<='\u036f')
    return re.findall(r'[^\W_]+',normalized)


def _units(query, *, like=False):
    groups=[[]];negative=False
    for token in re.findall(r'"[^"]*"|\S+',query):
        operator=token.upper() if like else token
        if operator=='OR':
            if groups[-1]:groups.append([])
            negative=False
        elif operator in ('AND','NEAR'):
            continue
        elif operator=='NOT':negative=True
        else:
            term=token.strip('"').rstrip('*').strip()
            if term:
                groups[-1].append((term,negative,token.endswith('*')))
                negative=False
    return [g for g in groups if g]


class PostgreSQLSearchMixin:
    _SEARCH_MESSAGE_RESULT_FIELDS=SessionSearchMixin._SEARCH_MESSAGE_RESULT_FIELDS

    def _search_match(self,groups,*,substring=False):
        groups_sql=[];params=[]
        for group in groups:
            if not any(not negated for _,negated,_ in group):return sql.SQL('FALSE'),[]
            clauses=[]
            for term,negated,prefix in group:
                columns=[]
                for column in _FIELDS:
                    if substring:
                        columns.append(sql.SQL('strpos(lower(COALESCE(m.{},\'\')),lower(%s))>0').format(sql.Identifier(column)))
                        params.append(term)
                    else:
                        words=_words(term)
                        if not words:return sql.SQL('FALSE'),[]
                        query=' <-> '.join("'"+word.replace("'","''")+"'" for word in words)
                        if prefix:query+=':*'
                        text=sql.SQL('COALESCE(m.{},\'\')').format(sql.Identifier(column))
                        if column=='content':
                            text=sql.SQL("CASE WHEN m.role='tool' THEN left({}, {}) ELSE {} END").format(text,sql.Literal(FTS_TOOL_CONTENT_PREFIX_CHARS),text)
                        columns.append(sql.SQL("to_tsvector('simple',{}) @@ to_tsquery('simple',%s)").format(text))
                        params.append(query)
                clause=sql.SQL('(')+sql.SQL(' OR ').join(columns)+sql.SQL(')')
                clauses.append(sql.SQL('NOT ')+clause if negated else clause)
            groups_sql.append(sql.SQL('(')+sql.SQL(' AND ').join(clauses)+sql.SQL(')'))
        return sql.SQL('(')+sql.SQL(' OR ').join(groups_sql)+sql.SQL(')'),params

    def _search_ranks(self,groups,*,trigram=False):
        """The native FTS5 BM25 formula (k1=1.2, b=.75), computed by PostgreSQL.

        Each indexed column keeps its own positions, so phrases cannot cross
        content/tool-name/tool-call boundaries. Statistics include hidden and
        inactive rows, as the native derived index does; filters apply afterwards.
        """
        units=[];matched_groups=[]
        for group in groups:
            matched_terms=[]
            if not any(not negative for _,negative,_ in group):
                continue
            for term,negative,prefix in group:
                words=[term.lower()[i:i+3] for i in range(len(term)-2)] if trigram else _words(term)
                if words:
                    units.append({'words':words,'prefix':False if trigram else prefix,'negative':negative})
                    clause=sql.SQL('EXISTS(SELECT 1 FROM frequencies check_frequency WHERE check_frequency.id=f.id AND check_frequency.term_id={} AND check_frequency.frequency>0)').format(sql.Literal(len(units)))
                    matched_terms.append(sql.SQL('NOT ')+clause if negative else clause)
            if matched_terms:matched_groups.append(sql.SQL('(')+sql.SQL(' AND ').join(matched_terms)+sql.SQL(')'))
        if not units:return sql.SQL('SELECT NULL::bigint AS id,0::double precision AS rank,FALSE AS matched WHERE FALSE'),[]
        if trigram:
            normalized=sql.SQL('lower(field.value)')
            tokens=sql.SQL('ARRAY(SELECT substr({},p,3) FROM generate_series(1,char_length({})-2) p)').format(normalized,normalized)
        else:
            tokens=sql.SQL(r"array_remove(regexp_split_to_array(regexp_replace(normalize(lower(field.value),NFD),U&'[\0300-\036f]','','g'),'[^[:alnum:]]+'),'')")
        corpus_where=sql.SQL("WHERE m.role<>'tool' AND s.source<>ALL(%s)") if trigram else sql.SQL('')
        params=[json.dumps(units)]
        if trigram:params.append(list(FTS_TRIGRAM_EXCLUDED_SOURCES))
        query=sql.SQL('''WITH units AS (
            SELECT u.term_id,ARRAY(SELECT jsonb_array_elements_text(u.value->'words')) AS words,
                (u.value->>'prefix')::boolean AS prefix,(u.value->>'negative')::boolean AS negative
            FROM jsonb_array_elements(%s::jsonb) WITH ORDINALITY AS u(value,term_id)),
        columns AS (
            SELECT m.id,field.ordinality,{} AS tokens
            FROM {} m JOIN {} s ON s.id=m.session_id
            CROSS JOIN LATERAL unnest(ARRAY[
                CASE WHEN m.role='tool' THEN left(COALESCE(m.content,''),{}) ELSE COALESCE(m.content,'') END,
                COALESCE(m.tool_name,''),COALESCE(m.tool_calls,'')]) WITH ORDINALITY field(value,ordinality) {}),
        documents AS (SELECT id,SUM(cardinality(tokens))::double precision AS length FROM columns GROUP BY id),
        frequencies AS (
            SELECT c.id,u.term_id,SUM((SELECT count(*) FROM generate_subscripts(c.tokens,1) p(i)
                WHERE CASE WHEN u.prefix THEN
                    (cardinality(u.words)=1 OR c.tokens[p.i:p.i+cardinality(u.words)-2]=u.words[1:cardinality(u.words)-1])
                    AND left(c.tokens[p.i+cardinality(u.words)-1],length(u.words[cardinality(u.words)]))=u.words[cardinality(u.words)]
                ELSE c.tokens[p.i:p.i+cardinality(u.words)-1]=u.words END))::double precision AS frequency
            FROM columns c CROSS JOIN units u GROUP BY c.id,u.term_id),
        term_stats AS (SELECT term_id,count(*) FILTER(WHERE frequency>0)::double precision AS documents FROM frequencies GROUP BY term_id),
        corpus_stats AS (SELECT count(*)::double precision AS documents,avg(length) AS average_length FROM documents)
        SELECT f.id,-SUM(CASE WHEN f.frequency>0 AND NOT u.negative THEN
            GREATEST(ln((n.documents-t.documents+0.5)/(t.documents+0.5)),0.000001)
            *(f.frequency*2.2)/(f.frequency+1.2*(0.25+0.75*d.length/NULLIF(n.average_length,0))) ELSE 0 END) AS rank,
            {} AS matched
        FROM frequencies f JOIN documents d ON d.id=f.id JOIN term_stats t ON t.term_id=f.term_id
        JOIN units u ON u.term_id=f.term_id CROSS JOIN corpus_stats n GROUP BY f.id''').format(tokens,self.table('messages'),self.table('sessions'),sql.Literal(FTS_TOOL_CONTENT_PREFIX_CHARS),corpus_where,sql.SQL(' OR ').join(matched_groups))
        return query,params

    def search_messages(self,query,source_filter=None,exclude_sources=None,role_filter=None,limit=20,offset=0,
            sort=None,include_inactive=False,fields=None,after_ts=None,before_ts=None):
        selected=SessionSearchMixin._search_message_fields.__func__(self.__class__,fields)
        if not query or not query.strip():return []
        query=SessionSearchMixin._sanitize_fts5_query(query)
        if not query:return []
        where=[];params=[]
        if not include_inactive:where.append(sql.SQL('(m.active=1 OR m.compacted=1)'))
        where.append(sql.SQL("COALESCE(m.display_kind,'')<>'hidden'"))
        for column,values,negative in [('source',source_filter,False),('source',exclude_sources,True),('role',role_filter,False)]:
            if values is not None and (column!='role' or values):
                where.append(sql.SQL('{}'+('<>ALL(%s)' if negative else '=ANY(%s)')).format(sql.Identifier('m' if column=='role' else 's',column)))
                params.append(list(values))
        if after_ts is not None:where.append(sql.SQL('s.started_at>=%s'));params.append(after_ts)
        if before_ts is not None:where.append(sql.SQL('s.started_at<%s'));params.append(before_ts)
        base_where=sql.SQL(' AND ').join(where)
        order=sort.strip().lower() if isinstance(sort,str) else None
        cjk=SessionSearchMixin._contains_cjk(query)
        tool=bool(role_filter and 'tool' in role_filter)
        groups=_units(query,like=tool)
        if not groups:return []
        if cjk and not tool:
            raw=_strip_cjk_wildcards(query).strip('"').strip()
            groups=[[(term,False,False)] for term in (_non_operator_tokens(raw) or [raw])]
        select=sql.SQL('SELECT m.id,m.session_id,m.role,m.content,m.content_bytes,m.timestamp,m.tool_name,s.source,s.model,s.started_at AS session_started FROM {} m JOIN {} s ON s.id=m.session_id').format(self.table('messages'),self.table('sessions'))
        def find(current,*,substring=False,trigram=False,like_order=False):
            match,match_params=self._search_match(current,substring=substring)
            extra=sql.SQL('')
            extra_params=[]
            if trigram:
                extra=sql.SQL(" AND m.role<>'tool' AND s.source<>ALL(%s)")
                extra_params=[list(FTS_TRIGRAM_EXCLUDED_SOURCES)]
            if like_order:
                direction='ASC' if tool and order=='oldest' else 'DESC'
                statement=select+sql.SQL(' WHERE {} AND {}{} ORDER BY m.timestamp {},m.id {} LIMIT %s OFFSET %s').format(base_where,match,extra,sql.SQL(direction),sql.SQL(direction))
                return self._all(statement,[*params,*match_params,*extra_params,limit,offset])
            ranks,rank_params=self._search_ranks(current,trigram=trigram)
            ordering=sql.SQL('m.timestamp ASC,ranked.rank,m.id') if order=='oldest' else sql.SQL('m.timestamp DESC,ranked.rank,m.id') if order=='newest' else sql.SQL('ranked.rank,m.id')
            match=match if trigram else sql.SQL('ranked.matched')
            statement=sql.SQL('WITH ranked AS (')+ranks+sql.SQL(') ')+select+sql.SQL(' JOIN ranked ON ranked.id=m.id WHERE {} AND {}{} ORDER BY {} LIMIT %s OFFSET %s').format(base_where,match,extra,ordering)
            return self._all(statement,[*rank_params,*params,*(match_params if trigram else []),*extra_params,limit,offset])
        rows=find(groups,substring=cjk or tool,like_order=cjk or tool)
        if not rows and not cjk and not tool:
            if SessionSearchMixin._trigram_eligible_tokens(query):rows=find(groups,substring=True,trigram=True)
            if not rows:
                relaxed=SessionSearchMixin._or_relaxed_query(query)
                if relaxed:rows=find(_units(relaxed))
        terms=[term for group in groups for term,negative,_ in group if not negative]
        output=[]
        for row in rows:
            raw=row.pop('content_bytes',None)
            content=bytes(raw).decode('utf-8') if raw is not None else row.pop('content') or ''
            row.pop('content',None)
            anchor=content.lower().find(terms[0].lower()) if terms else 0
            row['snippet']=content[max(0,anchor-40):max(0,anchor-40)+120]
            if selected is None or 'context' in selected:
                context=self._all(sql.SQL('''SELECT * FROM {} WHERE id IN (
                    %s,(SELECT id FROM {} WHERE session_id=%s AND (timestamp,id)<(%s,%s) ORDER BY timestamp DESC,id DESC LIMIT 1),
                    (SELECT id FROM {} WHERE session_id=%s AND (timestamp,id)>(%s,%s) ORDER BY timestamp,id LIMIT 1)) ORDER BY timestamp,id''').format(self.table('messages'),self.table('messages'),self.table('messages')),
                    (row['id'],row['session_id'],row['timestamp'],row['id'],row['session_id'],row['timestamp'],row['id']))
                row['context']=[{'role':m['role'],'content':_flatten_text(self._decode_message(m)['content'])[:200]} for m in context]
            output.append({k:v for k,v in row.items() if selected is None or k in selected})
        return output
