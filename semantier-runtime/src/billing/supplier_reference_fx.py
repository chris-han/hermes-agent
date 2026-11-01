"""ECB reference conversion custody; never supplier settlement or a live replay dependency."""
from datetime import datetime, timezone, timedelta, date
from decimal import Decimal, localcontext, InvalidOperation, ROUND_HALF_EVEN
import asyncio
import hashlib
import xml.etree.ElementTree as ET
import httpx
from psycopg.types.json import Jsonb
from billing.rating import FxQuote
from billing.supplier_pricing_projection import digest

URL='https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml'
SOURCE='European Central Bank; calculated USD/CNY cross-rate; reference only, not settlement; '+URL
PREFIX='supplier-reference-fx:'

def parse_ecb(raw, collected_at):
    if len(raw)>262144 or b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise ValueError('FX_SOURCE_INVALID')
    if collected_at.utcoffset() is None:raise ValueError('FX_TIMEZONE_REQUIRED')
    collected_at=collected_at.astimezone(timezone.utc)
    try:
        root=ET.fromstring(raw)
        days=[n for n in root.iter() if 'time' in n.attrib]
        if len(days)!=1:raise ValueError()
        day=date.fromisoformat(days[0].attrib['time'])
        if not timedelta(0)<=collected_at.date()-day<=timedelta(days=7):raise ValueError()
        rates={}
        for n in days[0]:
            currency=n.attrib['currency']
            if currency in rates:raise ValueError()
            rates[currency]=Decimal(n.attrib['rate'])
            if (not rates[currency].is_finite() or not Decimal('1e-9')<=rates[currency]<=Decimal('1e12')
                    or rates[currency].as_tuple().exponent < -12):raise ValueError()
        with localcontext() as ctx:
            ctx.prec=50
            exact=format(rates['USD']/rates['CNY'],'f')
            rate=format(Decimal(exact).quantize(Decimal('0.000000000001'),rounding=ROUND_HALF_EVEN),'f')
    except (ET.ParseError,ValueError,KeyError,InvalidOperation,TypeError):
        raise ValueError('FX_SOURCE_INVALID_OR_STALE') from None
    payload={'schema_version':'ecb-reference-fx.v1','url':URL,'source':SOURCE,
        'raw_xml':raw.decode('utf-8'),'sha256':hashlib.sha256(raw).hexdigest(),
        'quote_date':day.isoformat(),'collected_at':collected_at.isoformat(),
        'publisher_base':'EUR','publisher_unit':'currency units per 1 EUR',
        'publisher_rates':{k:format(rates[k],'f') for k in ('USD','CNY')},
        'base':'CNY','quote':'USD','unit':'USD per 1 CNY','rate':rate,
        'unrounded_cross_rate':exact,'rounding':'12 decimal places, ROUND_HALF_EVEN',
        'purpose':'reference_conversion_not_settlement'}
    quote=FxQuote('ecb-reference:'+digest(payload),'CNY','USD',rate,SOURCE,collected_at.isoformat(),purpose=payload['purpose'])
    return payload,quote

def validate_reference(quote,payload,now=None):
    now=now or datetime.now(timezone.utc)
    try:
        parsed,expected=parse_ecb(payload['raw_xml'].encode(),datetime.fromisoformat(payload['collected_at']))
        if parsed!=payload or expected!=quote:raise ValueError()
        if not timedelta(0)<=now.date()-date.fromisoformat(payload['quote_date'])<=timedelta(days=7):raise ValueError()
        if datetime.fromisoformat(payload['collected_at'])>now:raise ValueError()
    except (KeyError,ValueError,TypeError):raise ValueError('QUALIFIED_CURRENT_REFERENCE_FX_REQUIRED') from None

async def retain_reference(billing, payload,quote):
    validate_reference(quote,payload)
    async with billing.connection() as c:
        await c.execute("INSERT INTO prices(version_id,kind,payload) VALUES(%s,'supplier',%s) ON CONFLICT DO NOTHING",(PREFIX+quote.quote_id,Jsonb(payload)))
        old=await billing._one(c,"SELECT payload FROM prices WHERE version_id=%s AND kind='supplier'",(PREFIX+quote.quote_id,))
        if old['payload']!=payload:raise ValueError('IMMUTABLE_FX_SOURCE')
        await c.execute('INSERT INTO fx_quotes(quote_id,payload) VALUES(%s,%s) ON CONFLICT DO NOTHING',(quote.quote_id,Jsonb(quote.__dict__)))
        old=await billing._one(c,'SELECT payload FROM fx_quotes WHERE quote_id=%s',(quote.quote_id,))
        if old['payload']!=quote.__dict__:raise ValueError('IMMUTABLE_FX_QUOTE')

async def require_reference(billing, quote):
    async with billing.connection() as c:
        row=await billing._one(c,"SELECT payload FROM prices WHERE version_id=%s AND kind='supplier'",(PREFIX+quote.quote_id,),required=False)
    validate_reference(quote,row['payload'] if row else {})
    return row['payload']

async def _refresh_reference(billing):
    async with httpx.AsyncClient(trust_env=False,follow_redirects=False,transport=httpx.AsyncHTTPTransport(retries=0)) as client:
        async with client.stream('GET',URL,timeout=30,headers={'Accept-Encoding':'identity'}) as response:
            if response.status_code!=200:raise ValueError('FX_SOURCE_HTTP_UNAVAILABLE')
            raw=bytearray()
            async for chunk in response.aiter_bytes(chunk_size=65536):
                if len(raw)+len(chunk)>262144:raise ValueError('FX_SOURCE_SIZE_LIMIT')
                raw.extend(chunk)
    payload,quote=parse_ecb(bytes(raw),datetime.now(timezone.utc))
    await retain_reference(billing,payload,quote)
    return quote


async def refresh_reference(billing):
    async with asyncio.timeout(30):
        return await _refresh_reference(billing)
