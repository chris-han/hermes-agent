"""Controlled DEVELOPMENT channel edges using pinned real Python SDKs.

Controlled loopback and explicitly isolated official sandbox endpoints only.
No merchant sales or refunds are activated. A verified observation is still checked against the persisted attempt
and immutable order by CommerceService before it can enqueue fulfillment.
"""
from decimal import Decimal, InvalidOperation
import json
import time
from urllib.parse import urlsplit, parse_qsl


def controlled_endpoint(value):
    p=urlsplit(value)
    if p.scheme!='http' or p.hostname not in {'127.0.0.1','localhost','::1'} or p.username or p.password or p.fragment or p.query:
        raise ValueError('controlled loopback payment endpoint required')
    return value.rstrip('/')


def amount(value):
    try: n=Decimal(str(value))
    except (InvalidOperation,ValueError): raise ValueError('invalid channel amount') from None
    if not n.is_finite() or n<=0 or n!=n.quantize(Decimal('.01')):
        raise ValueError('invalid channel amount')
    return format(n,'.2f')


class WeChatChannel:
    method='wechat'
    environment='DEVELOPMENT'

    def __init__(self,*,merchant_id,app_id,private_key,certificate_serial,api_key,platform_public_key,platform_key_id,endpoint,notify_url):
        from wechatpayv3 import WeChatPay, WeChatPayType
        from wechatpayv3.core import Core
        from wechatpayv3.type import RequestType
        from wechatpayv3.utils import build_authorization
        class ControlledCore(Core):
            # Retain SDK API builders, signing, signature and AES-GCM owners;
            # replace only its redirect-following network edge for TEST scope.
            def request(core,path,method=RequestType.GET,data=None,skip_verify=False,sign_data=None,files=None,cipher_data=False,headers=None):
                import httpx
                if skip_verify or files or cipher_data or sign_data is not None:
                    raise ValueError('unsupported controlled SDK request')
                body=data if isinstance(data,str) else json.dumps(data) if data else ''
                h={'Content-Type':'application/json','Accept':'application/json','Wechatpay-Serial':core._public_key_id,
                   'Authorization':build_authorization(path,method.value,core._mchid,core._cert_serial_no,core._private_key,data=body)}
                with httpx.Client(trust_env=False,follow_redirects=False,timeout=5) as client:
                    response=client.request(method.value,controlled_endpoint(core._gate_way)+path,headers=h,content=body.encode())
                if 300<=response.status_code<400:raise ValueError('payment redirect forbidden')
                if 200<=response.status_code<300 and not core._verify_signature(response.headers,response.text):
                    raise ValueError('unverified payment response')
                return response.status_code,response.text

        endpoint=controlled_endpoint(endpoint); self.notify_url=controlled_endpoint(notify_url)
        self.merchant_id,self.app_id=merchant_id,app_id
        self.sdk=WeChatPay(WeChatPayType.NATIVE,merchant_id,private_key,certificate_serial,app_id,api_key,public_key=platform_public_key,public_key_id=platform_key_id,timeout=5)
        # SDK2.0.4 has no public base-url option; pinned private seam is qualified
        # by real HTTP contract tests. Never replace signing or verification.
        self.sdk._core=ControlledCore(mchid=merchant_id,cert_serial_no=certificate_serial,private_key=private_key,apiv3_key=api_key,public_key=platform_public_key,public_key_id=platform_key_id,timeout=5)
        self.sdk._core._gate_way=endpoint
        self.platform_key_id=platform_key_id

    def _response(self,result):
        code,body=result
        if not 200<=code<300: raise ValueError('payment channel response uncertain; query existing attempt')
        return json.loads(body) if body else {}

    def create(self,attempt_id,product):
        p=self._response(self.sdk.pay(description='SEMANTIER TEST '+product['product_id'],out_trade_no=attempt_id,amount={'total':int(Decimal(amount(product['amount']))*100),'currency':'CNY'},notify_url=self.notify_url))
        if not isinstance(p.get('code_url'),str):raise ValueError('checkout unavailable')
        return {'qr_url':p['code_url'],'environment':self.environment}

    def observation(self,p):
        if p.get('mchid')!=self.merchant_id or p.get('appid')!=self.app_id: raise ValueError('payment merchant/application mismatch')
        if p.get('trade_state')!='SUCCESS':return None
        a=p.get('amount',{})
        if type(a.get('total')) is not int or a.get('currency')!='CNY':raise ValueError('payment amount/currency mismatch')
        return {'attempt_id':p['out_trade_no'],'reference':p['transaction_id'],'amount':amount(Decimal(a['total'])/100),'currency':'CNY','merchant_id':self.merchant_id,'app_id':self.app_id,'environment':self.environment,'method':self.method,'status':'SUCCESS'}

    def notification(self,headers,body):
        h={k.lower():v for k,v in headers.items()}
        try:
            if h.get('wechatpay-serial')!=self.platform_key_id or abs(time.time()-int(h['wechatpay-timestamp']))>300: raise ValueError('stale or unknown payment signer')
            p=self.sdk.callback(h,body)
            if not p or p.get('event_type')!='TRANSACTION.SUCCESS':raise ValueError('unverified payment')
            return self.observation(p['resource'])
        except Exception as exc:raise ValueError('unverified payment notification') from exc

    def query(self,attempt_id):
        return self.observation(self._response(self.sdk.query(out_trade_no=attempt_id)))

    def close(self,attempt_id):
        self._response(self.sdk.close(out_trade_no=attempt_id))
        return {'status':'CLOSED','attempt_id':attempt_id}

    def refund(self,*args,**kwargs):
        raise ValueError('approved refund policy unavailable')


class AlipayChannel:
    method='alipay'
    environment='DEVELOPMENT'

    def __init__(self,*,merchant_id,app_id,private_key,platform_public_key,endpoint,notify_url):
        from alipay import AliPay
        class ControlledAliPay(AliPay):
            def verified_sync_response(client,data,response_type):
                import httpx
                url=controlled_endpoint(client._gateway)+'?'+client.sign_data(data)
                with httpx.Client(trust_env=False,follow_redirects=False,timeout=5) as transport:
                    response=transport.get(url)
                if response.status_code!=200:raise ValueError('payment response/redirect unavailable')
                return client._verify_and_return_sync_response(response.text,response_type)
        from alipay.utils import AliPayConfig
        endpoint=controlled_endpoint(endpoint); self.notify_url=controlled_endpoint(notify_url)
        self.merchant_id,self.app_id=merchant_id,app_id
        self.sdk=ControlledAliPay(appid=app_id,app_private_key_string=private_key,alipay_public_key_string=platform_public_key,sign_type='RSA2',config=AliPayConfig(timeout=5))
        self.sdk._gateway=endpoint

    def _request(self,kind,biz):
        names={'Precreate':'precreate','Query':'query','Close':'close'}
        method=getattr(self.sdk,'api_alipay_trade_'+names[kind])
        if kind=='Precreate':biz=biz|{'notify_url':self.notify_url}
        result=method(**biz)
        if result.get('code')!='10000':raise ValueError('payment channel response uncertain; query existing attempt')
        return result

    def create(self,attempt_id,product):
        p=self._request('Precreate',{'out_trade_no':attempt_id,'total_amount':amount(product['amount']),'subject':'SEMANTIER TEST '+product['product_id']})
        if p.get('out_trade_no')!=attempt_id or not isinstance(p.get('qr_code'),str):raise ValueError('checkout binding mismatch')
        return {'qr_url':p['qr_code'],'environment':self.environment}

    def observation(self,p,*,notification=False):
        if notification and (p.get('app_id')!=self.app_id or p.get('seller_id')!=self.merchant_id):raise ValueError('payment merchant/application mismatch')
        if p.get('trade_status') not in {'TRADE_SUCCESS','TRADE_FINISHED'}:return None
        return {'attempt_id':p['out_trade_no'],'reference':p['trade_no'],'amount':amount(p['total_amount']),'currency':'CNY','merchant_id':self.merchant_id,'app_id':self.app_id,'environment':self.environment,'method':self.method,'status':'SUCCESS'}

    def notification(self,headers,body):
        try:
            pairs=parse_qsl(body.decode() if isinstance(body,bytes) else body,keep_blank_values=True,strict_parsing=True)
            p=dict(pairs)
            if len(p)!=len(pairs) or p.get('sign_type')!='RSA2':raise ValueError('invalid signature fields')
            signature=p.pop('sign')
            if not self.sdk.verify(p.copy(),signature):raise ValueError('unverified payment')
            p.pop('sign_type')
            return self.observation(p,notification=True)
        except Exception as exc:raise ValueError('unverified payment notification') from exc

    def query(self,attempt_id):return self.observation(self._request('Query',{'out_trade_no':attempt_id}))

    def close(self,attempt_id):
        p=self._request('Close',{'out_trade_no':attempt_id})
        if p.get('out_trade_no')!=attempt_id:raise ValueError('close binding mismatch')
        return {'status':'CLOSED','attempt_id':attempt_id}

    def refund(self,*args,**kwargs):raise ValueError('approved refund policy unavailable')


def configured_channels():
    """Explicit server-owned DEVELOPMENT configuration; never browser input."""
    import os
    if os.getenv('SEMANTIER_ENVIRONMENT','').lower()!='development' or os.getenv('SEMANTIER_COMMERCE_ENVIRONMENT')!='DEVELOPMENT':return {}
    sandbox=os.getenv('SEMANTIER_ALIPAY_SANDBOX_CONFIG')
    if sandbox:
        if os.getenv('SEMANTIER_COMMERCE_CHANNELS_JSON'):raise ValueError('channel environments cannot be mixed')
        return {'alipay':configured_sandbox(sandbox)}
    raw=os.getenv('SEMANTIER_COMMERCE_CHANNELS_JSON')
    if not raw:return {}
    try:
        configs=json.loads(raw)
        if not isinstance(configs,dict) or set(configs)-{'wechat','alipay'}:raise ValueError()
        types={'wechat':WeChatChannel,'alipay':AlipayChannel}
        return {name:types[name](**config) for name,config in configs.items()}
    except Exception:raise ValueError('invalid controlled channel configuration') from None


ALIPAY_SANDBOX_ENDPOINT='https://openapi-sandbox.dl.alipaydev.com/gateway.do'

def sandbox_endpoint(value):
    # Compare the canonical URL, including absent credentials/query/fragment/port.
    if value!=ALIPAY_SANDBOX_ENDPOINT:raise ValueError('exact Alipay sandbox endpoint required')
    return value


def sandbox_notify_url(value):
    if value is None:return None
    p=urlsplit(value)
    if (p.scheme!='https' or not p.hostname or p.username or p.password or p.port not in {None,443}
            or p.path!='/api/billing/commerce/notifications/alipay' or p.query or p.fragment):
        raise ValueError('HTTPS signed notification path required')
    return value


class AlipaySandboxChannel(AlipayChannel):
    """Independently selected virtual-funds website channel; no endpoint override."""
    environment='ALIPAY_SANDBOX'

    def __init__(self,*,merchant_id,app_id,private_key,platform_public_key,endpoint,notify_url,return_origin):
        from alipay import AliPay
        from alipay.utils import AliPayConfig
        sandbox_endpoint(endpoint)
        self.notify_url=sandbox_notify_url(notify_url)
        p=urlsplit(return_origin)
        if (p.scheme not in {'http','https'} or not p.hostname or p.username or p.password or p.path not in {'','/'} or p.query or p.fragment
                or (p.scheme=='http' and p.hostname not in {'localhost','127.0.0.1'})):
            raise ValueError('server-owned Billing return origin required')
        self.return_origin=return_origin.rstrip('/')
        self.merchant_id,self.app_id=merchant_id,app_id
        class SandboxAliPay(AliPay):
            def verified_sync_response(client,data,response_type):
                import httpx
                try:
                    with httpx.Client(trust_env=False,follow_redirects=False,timeout=10) as transport:
                        # POST avoids signed request parameters in access/log URLs.
                        response=transport.post(sandbox_endpoint(client._gateway),content=client.sign_data(data),headers={'Content-Type':'application/x-www-form-urlencoded'})
                    if response.status_code!=200 or len(response.content)>65536:raise ValueError()
                    return client._verify_and_return_sync_response(response.text,response_type)
                except Exception:
                    # SDK/network exception strings can contain signed payloads.
                    raise ValueError('sandbox verification unavailable; query existing attempt') from None
        self.sdk=SandboxAliPay(appid=app_id,app_private_key_string=private_key,alipay_public_key_string=platform_public_key,sign_type='RSA2',debug=True,config=AliPayConfig(timeout=10))
        self.sdk._gateway=endpoint

    def create(self,attempt_id,product):
        from urllib.parse import quote,urlencode
        continuation=self.return_origin+'/billing/orders/'+quote(product['order_id'],safe='')+'?'+urlencode({'account_id':product['account_id'],'lang':product.get('language','en') if product.get('language','en') in {'en','zh'} else 'en'})
        query=self.sdk.api_alipay_trade_page_pay(subject='SEMANTIER SANDBOX '+product['product_id'],out_trade_no=attempt_id,total_amount=amount(product['amount']),return_url=continuation,notify_url=self.notify_url)
        return {'kind':'page_redirect','url':sandbox_endpoint(self.sdk._gateway)+'?'+query,'environment':self.environment}

    def observation(self,p,*,notification=False):
        for name,expected in [('app_id',self.app_id),('seller_id',self.merchant_id),('currency','CNY')]:
            if name in p and p[name]!=expected:raise ValueError('sandbox payment binding mismatch')
        status=p.get('trade_status')
        if status in {'WAIT_BUYER_PAY','TRADE_CLOSED'}:return None
        if status not in {'TRADE_SUCCESS','TRADE_FINISHED'}:raise ValueError('unsupported sandbox payment status')
        return super().observation(p,notification=notification)


def configured_sandbox(config_path):
    """Managed operator assets only, and an independently isolated financial DB."""
    import os
    from pathlib import Path
    from psycopg.conninfo import conninfo_to_dict
    try:
        db=conninfo_to_dict(os.environ['SEMANTIER_MODEL_GATEWAY_DATABASE_URL'])['dbname']
        if not db.startswith('alipay_sandbox_'):raise ValueError()
        root=(Path(os.environ['SEMANTIER_LOCAL_STATE_DIR'])/'payments/alipay-sandbox').resolve()
        path=Path(config_path)
        if path.is_symlink() or path.resolve()!=root/'config.json':raise ValueError()
        config=json.loads(path.read_text())
        if config['environment']!='ALIPAY_SANDBOX' or config['production_enabled'] is not False or config['sign_type']!='RSA2':raise ValueError()
        def key(name):
            p=Path(config[name]);p=p if p.is_absolute() else root/p
            if p.is_symlink() or p.resolve().parent!=root:raise ValueError()
            return p.read_text()
        return AlipaySandboxChannel(merchant_id=config['merchant_id'],app_id=config['app_id'],private_key=key('private_key_file'),platform_public_key=key('platform_public_key_file'),endpoint=config['endpoint'],notify_url=os.getenv('SEMANTIER_ALIPAY_SANDBOX_NOTIFY_URL'),return_origin=os.environ['SEMANTIER_ALIPAY_SANDBOX_RETURN_ORIGIN'])
    except Exception:
        raise ValueError('invalid isolated Alipay sandbox configuration') from None
