"""Finite source acquisition and scoped durable collection orchestration."""
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import re
import httpx
from billing.supplier_catalog import SupplierCatalog, _safe
from urllib.parse import urlsplit


def catalog_endpoint(api_base):
    try:
        p=urlsplit(api_base)
        if (p.scheme!='https' or p.username or p.password or p.query or p.fragment
                or p.port not in (None,443) or p.path.rstrip('/')!='/compatible-mode/v1'
                or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,62}\.cn-beijing\.maas\.aliyuncs\.com',p.hostname or '')):
            raise ValueError
    except (ValueError,TypeError):
        raise ValueError('UNSUPPORTED_CATALOG_ENDPOINT') from None
    return 'https://'+p.hostname+'/api/v1/models'

from billing.supplier_pricing_projection import digest
from billing.supplier_pricing_document import normalize_api, parse_document, DOCUMENT_URL, PARSER, NORMALIZER

MODELS = ('qwen3.5-plus','qwen3.8-max','deepseek-v4.1-flash-1017')
async def _get(client, url, *, params=None, remaining, limit, api_key=None):
    async with client.stream('GET',url,params=params,follow_redirects=False,
            headers={'Accept-Encoding':'identity', **({'Authorization':'Bearer '+api_key} if api_key else {})}, timeout=httpx.Timeout(30,connect=5)) as response:
        data=bytearray()
        async for chunk in response.aiter_bytes(chunk_size=65536):
            if len(data)+len(chunk)>min(remaining,limit):
                raise ValueError('SOURCE_SIZE_LIMIT')
            data.extend(chunk)
        return response.status_code,bytes(data),response.headers.get('content-type','')


async def capture_sources(api_base, models, *, client=None, api_key=None):
    if (not isinstance(models,list) or not models or len(models)!=len(set(models))
            or not set(models)<=set(MODELS)):
        raise ValueError('INVALID_MODEL_SELECTION')
    if client is None:
        async with httpx.AsyncClient(trust_env=False,follow_redirects=False,
                transport=httpx.AsyncHTTPTransport(retries=0,trust_env=False)) as owned:
            return await capture_sources(api_base,models,client=owned,api_key=api_key)
    at=datetime.now(timezone.utc).isoformat()
    api={'status':'FAILED','raw':{'collected_at':at,'endpoint_template':
        'https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/models','responses':[]},'reason':None}
    document={'status':'FAILED','url':DOCUMENT_URL,'reason':None}
    used,pages=0,0
    try:
        async with asyncio.timeout(90):
            endpoint=catalog_endpoint(api_base)
            for model in sorted(models):
                total=None;seen=set();page=1
                while total is None or len(seen)<total:
                    if pages>=20:raise ValueError('PAGE_LIMIT_REACHED')
                    query=dict(model=model,service_site='asia-pacific-china',language='zh-CN',page_no=page,page_size=20)
                    status,data,_=await _get(client,endpoint,params=query,remaining=8*1024*1024-used,limit=8*1024*1024,api_key=api_key)
                    pages+=1;used+=len(data)
                    body=json.loads(data);_safe(body)
                    api['raw']['responses'].append({'query':query,'http_status':status,'body':body,
                        'sha256':hashlib.sha256(data).hexdigest(),'size':len(data),'raw_text':data.decode('utf-8')})
                    output=body.get('output') or {}
                    count,size,number=(output.get(k) for k in ('total','page_size','page_no'))
                    items=output.get('models')
                    if (status!=200 or body.get('success') is not True or any(type(n) is not int for n in (count,size,number))
                            or not 0<=count<=400 or size!=20 or number!=page or not isinstance(items,list)
                            or (total is not None and total!=count)):
                        raise ValueError('BUSINESS_OR_PAGINATION_FAILURE')
                    total=count
                    identities=[(m.get('model'),m.get('inference_provider')) for m in items]
                    if (len(identities)!=len(set(identities)) or seen.intersection(identities)
                            or len(items)!=min(20,max(0,total-(page-1)*20))):
                        raise ValueError('DUPLICATE_OR_MISSING_PAGE')
                    seen.update(identities);page+=1
                    if total==0:break
            catalog=SupplierCatalog.capture(api['raw'])
            if catalog.completeness!='COMPLETE_FOR_QUERY':raise ValueError('CATALOG_PARTIAL')
            available={r['model'] for r in normalize_api(api['raw'])}
            if not set(models)<=available:raise ValueError('SELECTED_MODEL_MISSING')
            api['status']='COMPLETE'
    except (httpx.HTTPError,TimeoutError,ValueError,TypeError,AttributeError,KeyError) as exc:
        api['status']='PARTIAL' if api['raw']['responses'] else 'FAILED'
        api['reason']=str(exc) if isinstance(exc,ValueError) and re.fullmatch('[A-Z_]+',str(exc)) else 'API_CAPTURE_FAILED'
    try:
        async with asyncio.timeout(30):
            status,data,ctype=await _get(client,DOCUMENT_URL,remaining=4*1024*1024,limit=4*1024*1024)
            document.update(http_status=status,sha256=hashlib.sha256(data).hexdigest(),size=len(data))
            if status!=200:raise ValueError('DOCUMENT_HTTP_FAILURE')
            html=data.decode('utf-8')
            document['raw_html']=html
            document.update(parse_document(html,models,api['raw']))
            document['status']='COMPLETE' if document['rows'] else 'UNAVAILABLE'
            if not document['rows']:document['reason']='DECLARED_TABLE_NOT_FOUND'
    except (httpx.HTTPError,TimeoutError,UnicodeError,ValueError,TypeError,KeyError) as exc:
        document['reason']=str(exc) if isinstance(exc,ValueError) and re.fullmatch('[A-Z_]+',str(exc)) else 'DOCUMENT_CAPTURE_FAILED'
    normalized=[]
    if api['raw']['responses']:
        try:normalized=normalize_api(api['raw'])
        except (ValueError,TypeError,AttributeError,KeyError):
            api.update(status='PARTIAL',reason='UNSUPPORTED_SOURCE_SHAPE')
    snapshot={'schema_version':'supplier-pricing-snapshot.v1','collected_at':at,'effective_from':None,
        'models':sorted(models),'region':'cn-beijing','api':api,'document':document,
        'normalized':normalized,'normalizer_version':NORMALIZER}
    snapshot['version_id']='sha256:'+digest(snapshot)
    return snapshot
