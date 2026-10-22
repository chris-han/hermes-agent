"""Runtime-owned QuerySpec 1.0.0 literal snapshot engine and offline verifier.

Ported from qualified source-text semantics; no research imports or file inputs.
Source loading, authority and durable attempts belong to SourceTextQueryService.
"""
import hashlib
import json
import re
import signal
import time
from pathlib import Path
NS='urn:semantier:official-snapshot:'
BUDGET={'records':10000,'calls':32,'seconds':30,'source_codepoints':50000}
def canonical(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()

def sha(b):return hashlib.sha256(b).hexdigest()

def fsha(p):return sha(Path(p).read_bytes())

def P(s):return '<'+NS+s+'>'

def graph_builder(documents):
    """Only full public sources/metadata; no query, reference or expected IDs."""
    triples=[];articles=[];docs=[];codepoints=0
    def add(s,p,o):triples.append([s,NS+p,str(o)])
    pattern=re.compile(r'^第([一二三四五六七八九十百]+)条[\u3000 \t]+',re.M)
    for source in documents.values():
        text=source['text'];assert sha(text.encode())==source['source_text_sha256']
        codepoints+=len(text);did=NS+'doc-'+source['source_text_sha256']
        title=text.splitlines()[0].split('_')[0]
        doc={**source,'id':did,'title':title,'text':text};docs.append(doc)
        for field,value in [('source_path',source['source_path']),('source_hash',source['source_text_sha256']),('raw_hash',source['raw_source_sha256']),('edition',source['declared_snapshot_edition']),('title',title)]:add(did,field,value)
        headings=list(pattern.finditer(text))
        for i,m in enumerate(headings):
            start=m.start();end=headings[i+1].start() if i+1<len(headings) else len(text)
            end=start+len(text[start:end].rstrip());quote=text[start:end]
            aid=NS+'article-'+sha((source['source_text_sha256']+':'+str(start)+':'+str(end)).encode())
            article={'id':aid,'doc':did,'number':m.group(1),'start':start,'end':end,'text':quote};articles.append(article)
            for field,value in [('doc',did),('number',m.group(1)),('start',start),('end',end),('text',quote),('text_hash',sha(quote.encode()))]:add(aid,field,value)
            # Every sentence is retained; referral detection is a fixed public
            # textual recipe, not gold role/ID/path annotation. No target
            # catalog is bound in these research inputs; this is UNKNOWN,
            # never a claim that the legal rules do not exist.
            cursor=start
            for sentence in re.finditer(r'[^。]*。',quote):
                raw=sentence.group();left=len(raw)-len(raw.lstrip());right=len(raw.rstrip())
                ss=start+sentence.start()+left;ee=start+sentence.start()+right
                body=text[ss:ee];sid=NS+'sentence-'+sha((source['source_text_sha256']+':'+str(ss)+':'+str(ee)).encode())
                for field,value in [('parent',aid),('start',ss),('end',ee),('text',body)]:add(sid,field,value)
                if re.search(r'由.+制订.*批准',body,re.S) or re.search(r'法律或者国务院.+有规定的.+依照其规定',body,re.S):
                    add(sid,'target_state','UNKNOWN');add(sid,'reason','No version-pinned target catalog is bound in this research snapshot; referral wording does not resolve its downstream rules.')
    aliases={d['title']:d['id'] for d in docs}
    for doc in docs:
        for title,alias in re.findall(r'《([^》]+)》（以下简称([^）]+)）',doc['text']):
            if title in aliases:aliases[alias]=aliases[title]
    by_number={(a['doc'],a['number']):a['id'] for a in articles}
    for a in articles:
        for alias,did in aliases.items():
            for number in re.findall(re.escape(alias)+r'第([一二三四五六七八九十百]+)条',a['text']):
                target=by_number.get((did,number))
                if target:add(a['id'],'explicit_ref',target)
    triples=sorted({tuple(t) for t in triples})
    return {'triples':[list(t) for t in triples],'source_codepoints':codepoints,'article_count':len(articles),'source_document_count':len(docs),'target_catalogs_bound':False,'reference_role':'source-only literal/structural research graph; no legal authority'}

def unknowns_query():
    return 'SELECT DISTINCT ?sentence ?source_path ?start ?end ?text ?state ?reason WHERE { VALUES ?article { RESULT_URIS } ?sentence ' + P('parent') + ' ?article ; ' + P('start') + ' ?start ; ' + P('end') + ' ?end ; ' + P('text') + ' ?text ; ' + P('target_state') + ' ?state ; ' + P('reason') + ' ?reason . ?article ' + P('doc') + ' ?doc . ?doc ' + P('source_path') + ' ?source_path . } ORDER BY ?source_path ?start'

def execute_native(graph,query,budget):
    """Native graph query + metered observational projection, no reference."""
    from semantier.knowledge_feature_packs.semantica_adapter import _ensure_vendored_semantica_importable
    _ensure_vendored_semantica_importable()
    from importlib.metadata import version
    if version("pyoxigraph") != "0.5.11":
        raise RuntimeError("pinned PyOxigraph 0.5.11 unavailable")
    from semantica.triplet_store import TripletStore
    from semantica.semantic_extract.types import Triplet
    began=time.monotonic();calls=0;records=0;ingested=0;trace=[]
    o={'articles':[],'unknowns':[],'snapshot_text_completion':'INCOMPLETE','downstream_dependency_completion':'UNKNOWN','legal_applicability':'NOT_ASSESSED','trace':trace,'reason':None}
    def timeout(signum,frame):raise TimeoutError('wall_budget')
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,budget['seconds'])
    try:
        if graph['source_codepoints']>budget['source_codepoints']:raise TimeoutError('source_budget')
        n=len(graph['triples'])
        if 2*n>=budget['records']:raise TimeoutError('graph_and_parity_record_budget')
        store=TripletStore(backend='oxigraph',enable_caching=False,enable_optimization=False)
        store.add_triplets([Triplet(subject=s,predicate=p,object=v,metadata={'datatype':'http://www.w3.org/2001/XMLSchema#integer'} if p.rsplit(':',1)[-1] in ('start','end') else {}) for s,p,v in graph['triples']]);records+=n;ingested=n
        if calls>=budget['calls']:raise TimeoutError('call_budget')
        calls+=1;native=store.get_triplets();records+=len(native)
        if sorted([t.subject,t.predicate,t.object] for t in native)!=sorted(graph['triples']):raise ValueError('lossless graph parity failed')
        if any(t.metadata.get('datatype')!='http://www.w3.org/2001/XMLSchema#integer' for t in native if t.predicate.rsplit(':',1)[-1] in ('start','end')):raise ValueError('numeric datatype parity failed')
        o['lossless_graph_parity']=True
        def run(sql,fields):
            nonlocal calls,records
            if calls>=budget['calls']:raise TimeoutError('call_budget')
            limit=(budget['records']-records)//fields
            if limit<=0:raise TimeoutError('record_budget')
            calls+=1;r=store.execute_query(sql+' LIMIT '+str(limit));rows=r.bindings
            records+=sum(len(b) for b in rows);trace.append({'query':sql+' LIMIT '+str(limit),'bindings':rows,'metadata':r.metadata})
            if len(rows)>=limit:raise TimeoutError('page_not_exhausted_at_budget')
            return [{k:v['value'] for k,v in b.items()} for b in rows]
        rows=run(query['articles'],9)
        for row in rows:
            o['articles'].append({k:(int(v) if k in ('start','end') else v) for k,v in row.items()})
        if rows:
            uris=' '.join('<'+r['article']+'>' for r in rows)
            unknowns=run(query['unknowns_template'].replace('RESULT_URIS',uris),7)
            o['unknowns']=[{k:(int(v) if k in ('start','end') else v) for k,v in row.items()} for row in unknowns]
        o['snapshot_text_completion']='COMPLETE_SNAPSHOT_TEXT' if rows else 'UNKNOWN';o['reason']='finite_native_queries_exhausted' if rows else 'no_matching_source_text'
    except TimeoutError as exc:o.update(snapshot_text_completion='INCOMPLETE',reason=str(exc))
    except Exception as exc:o.update(snapshot_text_completion='ACTOR_FAILURE',reason=type(exc).__name__+': '+str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    o['cost']={'backend_calls':calls,'charged_records':records,'graph_ingestion_records':ingested,'returned_query_bytes':sum(len(canonical(t['bindings'])) for t in trace),'elapsed_seconds':time.monotonic()-began}
    return o

VERSION='1.0.0';COMPILER='snapshot-sparql/1';VERIFIER='snapshot-set/1'
FIELDS=('article','source_path','source_hash','raw_hash','edition','text','text_hash','start','end')
def digest(x):return sha(canonical(x))
def scope_sources(documents):
 return {p:{'text_sha256':d['source_text_sha256'],'raw_sha256':d['raw_source_sha256'],'edition':d['declared_snapshot_edition']} for p,d in documents.items()}
def make_spec(graph,documents,root,*,direction='incoming',hops=0,targets=None,include_roots=True,budget=None):
 return {'version':VERSION,'compiler_version':COMPILER,'verifier_version':VERIFIER,'scope':{'graph_sha256':digest(graph),'sources':scope_sources(documents)},'root':root,'expansion':{'relation':'explicit_ref','direction':direction,'max_hops':hops,'predicates':targets or []},'results':{'include_roots':include_roots,'semantics':'distinct_full_articles'},'dependencies':'all_selected_parent_target_state_sentences','budget':budget or dict(BUDGET),'empty':'VERIFIED_SCOPE_ONLY'}
def keys(obj,expected):
 if not isinstance(obj,dict) or set(obj)!=set(expected.split()):raise ValueError('invalid_contract_fields')
def validate(spec,expected_hash,graph,documents):
 keys(spec,'version compiler_version verifier_version scope root expansion results dependencies budget empty')
 if (spec['version'],spec['compiler_version'],spec['verifier_version'])!=(VERSION,COMPILER,VERIFIER):raise ValueError('consumer_version_mismatch')
 if digest(spec)!=expected_hash:raise ValueError('spec_hash_mismatch')
 keys(spec['scope'],'graph_sha256 sources');keys(spec['expansion'],'relation direction max_hops predicates');keys(spec['results'],'include_roots semantics');keys(spec['budget'],'records calls seconds source_codepoints')
 e=spec['expansion']
 if e['relation']!='explicit_ref' or e['direction'] not in ('incoming','outgoing') or type(e['max_hops'])!=int or e['max_hops'] not in (0,1):raise ValueError('unsupported_expansion')
 for predicates,target in [(spec['root'],False),(e['predicates'],True)]:
  if not isinstance(predicates,list) or len(predicates)>8:raise ValueError('predicate_bound')
  for p in predicates:
   keys(p,'field op value')
   if p['field'] not in (('text',) if target else ('title','number','text')) or p['op'] not in ('equals','contains') or (p['field']!='text' and p['op']!='equals') or not isinstance(p['value'],str) or len(p['value'])>256:raise ValueError('unsupported_predicate')
 if spec['results']['semantics']!='distinct_full_articles' or type(spec['results']['include_roots'])!=bool or spec['dependencies']!='all_selected_parent_target_state_sentences' or spec['empty']!='VERIFIED_SCOPE_ONLY':raise ValueError('unsupported_semantics')
 for k,v in spec['budget'].items():
  if type(v) not in (int,float) or not 0<v<=BUDGET[k] or (k!='seconds' and type(v)!=int):raise ValueError('invalid_budget')
 if digest(graph)!=spec['scope']['graph_sha256']:raise ValueError('graph_pin_mismatch')
 if scope_sources(documents)!=spec['scope']['sources']:raise ValueError('source_pin_mismatch')
 if graph.get('target_catalogs_bound') is not False:raise ValueError('unsupported_target_catalog')
 for d in documents.values():
  if sha(d['text'].encode())!=d['source_text_sha256']:raise ValueError('source_text_mismatch')
 if sum(len(d['text']) for d in documents.values())!=graph['source_codepoints']:raise ValueError('source_count_mismatch')
 return expected_hash

def compile_query(spec):
 """Compile selectors to native SPARQL; independent verifier does not call this."""
 def filters(predicates,prefix):
  terms=[]
  for p in predicates:
   v='?'+prefix+p['field'];lit=json.dumps(p['value'],ensure_ascii=False)
   terms.append(v+' = '+lit if p['op']=='equals' else 'CONTAINS('+v+','+lit+')')
  return 'FILTER('+' && '.join(terms)+') ' if terms else ''
 root='?root '+P('doc')+' ?rd ; '+P('number')+' ?rnumber ; '+P('text')+' ?rtext . ?rd '+P('title')+' ?rtitle . '+filters(spec['root'],'r')
 branches=[]
 if spec['results']['include_roots']:branches.append('{ '+root+'BIND(?root AS ?article) }')
 if spec['expansion']['max_hops']:
  relation=('?article '+P('explicit_ref')+' ?root . ') if spec['expansion']['direction']=='incoming' else ('?root '+P('explicit_ref')+' ?article . ')
  branches.append('{ '+root+relation+'?article '+P('text')+' ?ttext . '+filters(spec['expansion']['predicates'],'t')+'}')
 selection=' UNION '.join(branches) if branches else 'FILTER(false) '
 binding='?article '+P('doc')+' ?doc ; '+P('text')+' ?text ; '+P('start')+' ?start ; '+P('end')+' ?end ; '+P('text_hash')+' ?text_hash . ?doc '+P('source_path')+' ?source_path ; '+P('source_hash')+' ?source_hash ; '+P('raw_hash')+' ?raw_hash ; '+P('edition')+' ?edition . '
 return {'articles':'SELECT DISTINCT '+' '.join('?'+x for x in FIELDS)+' WHERE { '+selection+' '+binding+'} ORDER BY ?source_path ?start','unknowns_template':unknowns_query()}

class BudgetStop(Exception):pass

def independent_basis(spec,graph,documents,tick=lambda n:None):
 """Independent literal-index enumeration: no compiler, engine or actor output."""
 index={}
 for s,p,v in graph['triples']:
  tick(1);index.setdefault(s,{}).setdefault(p.removeprefix(NS),[]).append(v)
 def one(x,k):
  values=x.get(k,[])
  if len(values)!=1:raise ValueError('non_singleton_source_field '+k)
  return values[0]
 all_articles={};attrs={}
 for aid,a in index.items():
  if 'number' not in a:continue
  d=index[one(a,'doc')];path=one(d,'source_path');doc=documents[path];start=int(one(a,'start'));end=int(one(a,'end'));text=one(a,'text')
  row={'article':aid,'source_path':path,'source_hash':one(d,'source_hash'),'raw_hash':one(d,'raw_hash'),'edition':one(d,'edition'),'text':text,'text_hash':one(a,'text_hash'),'start':start,'end':end}
  if not (0<=start<end<=len(doc['text']) and doc['text'][start:end]==text and sha(text.encode())==row['text_hash'] and row['source_hash']==doc['source_text_sha256'] and row['raw_hash']==doc['raw_source_sha256'] and row['edition']==doc['declared_snapshot_edition']):raise ValueError('source_span_binding')
  attrs[aid]={'title':one(d,'title'),'number':one(a,'number'),'text':text};all_articles[aid]=row
 def matches(a,ps):return all((a[p['field']]==p['value'] if p['op']=='equals' else p['value'] in a[p['field']]) for p in ps)
 roots={aid for aid in all_articles if matches(attrs[aid],spec['root'])};selected=set(roots) if spec['results']['include_roots'] else set();edges=[]
 if spec['expansion']['max_hops']:
  for aid in all_articles:
   for dest in index[aid].get('explicit_ref',[]):
    root,target=(dest,aid) if spec['expansion']['direction']=='incoming' else (aid,dest)
    if root in roots and target in all_articles and matches(attrs[target],spec['expansion']['predicates']):selected.add(target);edges.append([root,target])
 unknowns=[]
 for sid,s in index.items():
  if 'target_state' not in s:continue
  parent=one(s,'parent')
  if parent not in selected:continue
  a=all_articles[parent];start=int(one(s,'start'));end=int(one(s,'end'));text=one(s,'text')
  if not (a['start']<=start<end<=a['end'] and documents[a['source_path']]['text'][start:end]==text and one(s,'target_state')=='UNKNOWN' and one(s,'reason')):raise ValueError('dependency_source_binding')
  unknowns.append({'sentence':sid,'source_path':a['source_path'],'start':start,'end':end,'text':text,'state':one(s,'target_state'),'reason':one(s,'reason')})
 return {'articles':sorted([all_articles[x] for x in selected],key=lambda a:a['article']),'unknowns':sorted(unknowns,key=lambda a:a['sentence']),'roots':sorted(roots),'edges':sorted(edges)}

def binding(spec):
 return {'spec_sha256':digest(spec),'semantic_version':VERSION,'compiler_version':COMPILER,'verifier_version':VERIFIER,'graph_sha256':spec['scope']['graph_sha256'],'implementation_sha256':fsha(__file__)}

def reconstruct(spec,expected_hash,graph,documents,output,basis=None):
 validate(spec,expected_hash,graph,documents)
 basis=independent_basis(spec,graph,documents) if basis is None else basis
 q=compile_query(spec);trace=output['trace'];articles=output['articles'];unknowns=output['unknowns']
 expected_queries=[q['articles']]
 if articles:expected_queries.append(q['unknowns_template'].replace('RESULT_URIS',' '.join('<'+a['article']+'>' for a in articles)))
 exhausted=len(trace)==len(expected_queries);projected=[]
 for t,query in zip(trace,expected_queries):
  sql,limit=t['query'].rsplit(' LIMIT ',1);exhausted &= sql==query and len(t['bindings'])<int(limit)
  projected.append([{k:int(v['value']) if k in ('start','end') else v['value'] for k,v in row.items()} for row in t['bindings']])
 trace_match=projected==([articles,unknowns] if articles else [articles]) and (bool(articles) or not unknowns)
 exact=(sorted(articles,key=lambda x:x['article'])==basis['articles'] and sorted(unknowns,key=lambda x:x['sentence'])==basis['unknowns'])
 cost=output['cost'];within=cost['charged_records']<=spec['budget']['records'] and cost['backend_calls']<=spec['budget']['calls'] and cost['elapsed_seconds']<=spec['budget']['seconds']
 fields=sum(len(row) for t in trace for row in t['bindings']);n=len(graph['triples'])
 accounting=(cost['charged_records']==3*n+2*fields and cost.get('verification_records')==n+fields and cost['backend_calls']==1+len(trace) and cost.get('graph_ingestion_records')==n and cost['returned_query_bytes']==sum(len(canonical(t['bindings'])) for t in trace))
 within &= graph['source_codepoints']<=spec['budget']['source_codepoints']
 checks={'cost_accounting':accounting,'exact_source_sets':exact,'trace_exhausted':bool(exhausted),'trace_matches_output':trace_match,'graph_parity':output.get('lossless_graph_parity',False),'within_budget':within,'consumer_binding':output.get('binding')==binding(spec),'verification_finished':output.get('verification_finished',False)}
 return {'binding':binding(spec),'compiled_query_sha256':digest(q),'basis_sha256':digest(basis),'checks':checks,'valid':all(checks.values()),'state':'COMPLETE_NONEMPTY' if articles else 'COMPLETE_EMPTY','scope_statement':'Only the pinned graph/query scope; no inference about existence of legal rules.'}

def verify_certificate(spec,expected_hash,graph,documents,output):
 try:
  c=reconstruct(spec,expected_hash,graph,documents,output)
  return c['valid'] and c==output.get('certificate') and output.get('state')==c['state'] and output.get('execution_state')=='COMPLETE' and output.get('legal_applicability')=='NOT_ASSESSED' and output.get('evidence_availability')==('AVAILABLE' if output['articles'] else 'EMPTY_WITHIN_SCOPE') and output.get('dependency_status')==('UNKNOWN' if output['unknowns'] else 'NONE_OBSERVED_WITHIN_SCOPE')
 except (ValueError,KeyError,TypeError,IndexError):return False

def execute(spec,expected_hash,graph,documents):
 began=time.monotonic();o={'articles':[],'unknowns':[],'trace':[],'certificate':None,'state':'INVALID_QUERY','execution_state':'NOT_RUN','evidence_availability':'UNKNOWN','dependency_status':'UNKNOWN','legal_applicability':'NOT_ASSESSED','verification_finished':False,'cost':{'backend_calls':0,'charged_records':0,'returned_query_bytes':0,'elapsed_seconds':0}}
 def finish():
  o['cost']['elapsed_seconds']=time.monotonic()-began
  if o['state'].startswith('COMPLETE') and o['cost']['elapsed_seconds']>spec['budget']['seconds']:o.update(state='INCOMPLETE',certificate=None,reason='verification_wall_budget')
  return o
 try:validate(spec,expected_hash,graph,documents)
 except (ValueError,KeyError,TypeError) as e:o['reason']=str(e);return finish()
 o['binding']=binding(spec)
 if graph['source_codepoints']>spec['budget']['source_codepoints']:o.update(state='INCOMPLETE',reason='source_budget');return finish()
 try:
  native=execute_native(graph,compile_query(spec),spec['budget']);o.update({k:v for k,v in native.items() if k!='snapshot_text_completion'})
 except Exception as e:o.update(state='EXECUTION_FAILURE',execution_state='FAILED',reason=repr(e));return finish()
 if native['snapshot_text_completion']=='ACTOR_FAILURE':o.update(state='EXECUTION_FAILURE',execution_state='FAILED');return finish()
 if native['snapshot_text_completion']=='INCOMPLETE':o.update(state='INCOMPLETE',execution_state='INCOMPLETE');return finish()
 o['execution_state']='COMPLETE';o['cost']['verification_records']=0
 def tick(n):
  if o['cost']['charged_records']+n>spec['budget']['records']:raise BudgetStop('verification_record_budget')
  if time.monotonic()-began>spec['budget']['seconds']:raise BudgetStop('verification_wall_budget')
  o['cost']['charged_records']+=n;o['cost']['verification_records']+=n
 try:
  basis=independent_basis(spec,graph,documents,tick);tick(sum(len(a) for a in o['articles'])+sum(len(u) for u in o['unknowns']))
  o['verification_finished']=True;o['cost']['elapsed_seconds']=time.monotonic()-began;c=reconstruct(spec,expected_hash,graph,documents,o,basis)
  if not c['valid']:o.update(state='INCOMPLETE',reason='source_verification_failed');return finish()
  o.update(state=c['state'],certificate=c,evidence_availability='AVAILABLE' if o['articles'] else 'EMPTY_WITHIN_SCOPE',dependency_status='UNKNOWN' if o['unknowns'] else 'NONE_OBSERVED_WITHIN_SCOPE',reason='scoped_query_and_verification_complete')
 except BudgetStop as e:o.update(state='INCOMPLETE',reason=str(e))
 except (ValueError,KeyError,TypeError) as e:o.update(state='INCOMPLETE',reason='verification_unfinished: '+str(e))
 o['cost']['elapsed_seconds']=time.monotonic()-began
 return finish()
