"""Validate an admitted nonsecret physical identity; never adapt provider requests."""
from __future__ import annotations
import json
import re
from urllib.parse import parse_qsl,unquote,urlsplit
import httpx

_FIELDS=('api_base','api_version','aws_region_name','aws_bedrock_runtime_endpoint')
_PREFIXES={'azure':'azure/','bedrock':'bedrock/','ark':'volcengine/','dashscope':'dashscope/'}


def extract_physical_binding(params,provider):
    model=params.get('model')
    prefix=_PREFIXES.get(provider)
    if not prefix or not isinstance(model,str) or not model.startswith(prefix) or not model[len(prefix):]:
        raise ValueError('unsupported native physical model binding')
    binding={'provider_model_id':model,**{key:params.get(key) for key in _FIELDS}}
    if any(value is not None and (not isinstance(value,str) or not value or len(value)>2048) for value in binding.values()):
        raise ValueError('unresolved native physical binding')
    _target(binding,provider)
    return binding


def _url(value):
    if not isinstance(value,str):
        raise ValueError('explicit native endpoint required')
    parsed=urlsplit(value)
    if parsed.scheme not in {'http','https'} or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError('unsupported native physical endpoint')
    return parsed


def _origin(parsed):
    return parsed.scheme,parsed.hostname,parsed.port or (443 if parsed.scheme=='https' else 80)


def _path(value):
    decoded=unquote(value)
    if any(segment in {'.','..'} for segment in decoded.split('/')) or '\\' in decoded or '%' in decoded:
        raise ValueError('ambiguous native request path')
    return decoded.rstrip('/')


def _model(binding,provider):
    model=binding.get('provider_model_id')
    prefix=_PREFIXES.get(provider)
    if not prefix or not isinstance(model,str) or not model.startswith(prefix):
        raise ValueError('unsupported native physical model binding')
    model=model[len(prefix):]
    if provider=='bedrock' and model.startswith('converse/'):
        model=model[len('converse/'):]
    if not model or '%' in model or '\\' in model or any(ord(c)<32 for c in model):
        raise ValueError('unsupported native physical model identity')
    return model


def _target(binding,provider):
    _model(binding,provider)
    base=binding.get('api_base')
    if provider=='azure':
        if not base or not binding.get('api_version'):
            raise ValueError('Azure endpoint and api_version must be pinned')
    elif provider=='bedrock':
        region=binding.get('aws_region_name')
        if not isinstance(region,str) or not re.fullmatch(r'[a-z]{2}(?:-[a-z]+)+-\d+',region):
            raise ValueError('Bedrock native region must be pinned')
        suffix='amazonaws.com.cn' if region.startswith('cn-') else 'amazonaws.com'
        base=base or binding.get('aws_bedrock_runtime_endpoint') or f'https://bedrock-runtime.{region}.{suffix}'
    elif provider=='ark':
        base=base or 'https://ark.cn-beijing.volces.com/api/v3'
    elif provider=='dashscope':
        base=base or 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    else:
        raise ValueError('unsupported native physical provider')
    endpoint=_url(base)
    if endpoint.query:
        raise ValueError('native endpoint query must be separately pinned')
    _path(endpoint.path)
    return endpoint


def verify_native_request(binding,provider,request:httpx.Request):
    """Call before the durable send claim and transport, after the origin allowlist.

    Missing endpoint defaults are restricted to these source-pinned provider defaults;
    the capture caller must independently allow the exact resulting origin. No env
    discovery, retry, rewrite, SDK call or credential resolution occurs here.
    """
    if not isinstance(binding,dict) or request.method!='POST':
        raise ValueError('admitted native physical binding and POST required')
    target=_target(binding,provider)
    actual=_url(str(request.url))
    if _origin(actual)!=_origin(target):
        raise ValueError('native endpoint changed after admission')
    host=request.headers.get('host')
    if host and _origin(_url(actual.scheme+'://'+host))!=_origin(actual):
        raise ValueError('native authority header differs from admitted endpoint')
    base=_path(target.path)
    path=_path(actual.path)
    model=_model(binding,provider)
    if provider=='azure':
        expected=base+'/openai/deployments/'+model+'/chat/completions'
        if path!=expected or parse_qsl(actual.query,keep_blank_values=True)!=[('api-version',binding['api_version'])]:
            raise ValueError('Azure deployment or api-version changed after admission')
    elif provider=='bedrock':
        if path not in {base+'/model/'+model+'/converse',base+'/model/'+model+'/converse-stream'} or actual.query:
            raise ValueError('Bedrock model or protocol changed after admission')
        auth=request.headers.get('authorization','')
        if auth.startswith('AWS'):
            scope=re.search(r'Credential=[^,\s/]+/\d{8}/([^/,]+)/([^/,]+)/aws4_request(?:[,\s]|$)',auth)
            if not scope or scope.group(1)!=binding['aws_region_name'] or scope.group(2)!='bedrock':
                raise ValueError('Bedrock signed region changed after admission')
    else:
        if path!=base+'/chat/completions' or actual.query:
            raise ValueError('native chat endpoint changed after admission')
        try:
            body=json.loads(request.content)
        except (ValueError,httpx.RequestNotRead) as exc:
            raise ValueError('native model body unavailable') from exc
        if not isinstance(body,dict) or body.get('model')!=model:
            raise ValueError('native model changed after admission')
