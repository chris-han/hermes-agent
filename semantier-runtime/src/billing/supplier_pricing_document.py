"""Pure versioned normalization of retained Alibaba source bytes."""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
import re
from billing.supplier_catalog import SupplierCatalog, _range
DOCUMENT_URL = 'https://help.aliyun.com/zh/model-studio/model-pricing'
PARSER = 'alibaba-official-tables.v2'
NORMALIZER = 'alibaba-model-studio-api.v2'

def _normalize_prices(prices):
    """Annotate only source-explicit batch modes; preserve unknown conditions."""
    result = deepcopy(prices)
    for price in result:
        if not isinstance(price, dict) or 'mode' in price:
            continue
        kind, name = price.get('type'), price.get('price_name')
        if not isinstance(name, str):
            continue
        file_label = bool(re.search(r'\bBatch File\b', name, re.I))
        chat_label = bool(re.search(r'\bBatch Chat\b', name, re.I))
        if kind in ('input_token_batch', 'output_token_batch') and file_label and not chat_label:
            price['mode'] = 'batch_file'
        elif kind in ('input_token_batch_chat', 'output_token_batch_chat') and chat_label and not file_label:
            price['mode'] = 'batch_chat'
    return result

def normalize_api(raw):
    catalog = SupplierCatalog.capture(raw)
    rows = []
    for page in catalog.raw['responses']:
        for model in page.get('body', {}).get('output', {}).get('models', []):
            for tier in model.get('prices', []):
                rows.append({'model':model['model'], 'provider':'dashscope', 'region':'cn-beijing',
                    'service_site':'asia-pacific-china', 'currency':'CNY',
                    'currency_basis':'official-cn-beijing-list-price', 'effective_from':None,
                    'range':tier.get('range_name'), 'prices':_normalize_prices(tier.get('prices', []))})
    return rows


class PriceTables(HTMLParser):
    """Retain table cells with explicit heading context and HTML row spans."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.headings = {}
        self.heading = None
        self.heading_text = []
        self.tables = []
        self.table = None
        self.row = None
        self.cell = None
        self.span = 1
        self.carry = {}
        self.links = []
        self.deepseek_text = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('h2','h3','h4','h5'):
            self.heading, self.heading_text = int(tag[1]), []
        if tag == 'a' and attrs.get('href','').startswith('https://help.aliyun.com/'):
            self.links.append(attrs['href'])
        if tag == 'table' and self.table is None:
            self.table = {'context':list(self.headings.values()), 'rows':[]}
            self.carry = {}
        if tag == 'tr' and self.table is not None:
            self.row = []
        if tag in ('th','td') and self.row is not None:
            self.cell = []
            try:self.span = min(100, max(1, int(attrs.get('rowspan','1'))))
            except ValueError:self.span = 1

    def handle_data(self, data):
        if self.heading:self.heading_text.append(data)
        if self.cell is not None:self.cell.append(data)
        if self.headings.get(3)=='DeepSeek' and not self.heading:self.deepseek_text.append(data)

    def handle_endtag(self, tag):
        if tag in ('h2','h3','h4','h5') and self.heading:
            n=self.heading
            self.headings={k:v for k,v in self.headings.items() if k<n}
            self.headings[n]=' '.join(''.join(self.heading_text).split())
            self.heading=None
        if tag in ('td','th') and self.cell is not None:
            self.row.append((' '.join(' '.join(self.cell).split()), self.span))
            self.cell=None
        if tag == 'tr' and self.row is not None:
            values, col = [], 0
            for text, span in self.row:
                while col in self.carry:
                    value, remain=self.carry[col];values.append(value)
                    if remain<=1:del self.carry[col]
                    else:self.carry[col]=(value,remain-1)
                    col+=1
                values.append(text)
                if span>1:self.carry[col]=(text,span-1)
                col+=1
            while col in self.carry:
                value,remain=self.carry[col];values.append(value)
                if remain<=1:del self.carry[col]
                else:self.carry[col]=(value,remain-1)
                col+=1
            self.table['rows'].append(values)
            self.row=None
        if tag=='table' and self.table is not None:
            self.tables.append(self.table);self.table=None


def parse_document(html, models, api_raw=None):
    parser=PriceTables();parser.feed(html)
    rows=[]
    for table in parser.tables:
        context=' '.join(table['context'])
        if '北京' not in context:continue
        for cells in table['rows']:
            for model in models:
                if any(re.search(r'(?<![\w.-])'+re.escape(model)+r'(?![\w.-])',cell) for cell in cells):
                    rows.append({'model':model,'region':'cn-beijing','context':table['context'],
                                 'column_headings':table['rows'][0] if table['rows'] else [],'cells':cells})
    matches={m:{'status':'UNAVAILABLE','reason':'DECLARED_TABLE_NOT_FOUND'} for m in models}
    try:api_rows=normalize_api(api_raw) if api_raw else []
    except (ValueError,TypeError,AttributeError,KeyError):api_rows=[]
    for model in models:
        found=[r for r in rows if r['model']==model]
        # A declared input/output million-token header and explicit yuan values
        # are needed for corroboration. Other rows stay visible, never guessed.
        valid=[r for r in found if '输入' in ' '.join(r['column_headings']) and '输出' in ' '.join(r['column_headings'])
               and '百万' in ' '.join(r['column_headings'])]
        from decimal import Decimal
        compared=[]
        expected_rows=[r for r in api_rows if r['model']==model]
        for expected_row in expected_rows:
            band_rows=valid
            if len(expected_rows)>1:
                band_rows=[]
                for row in valid:
                    indexes=[i for i,h in enumerate(row['column_headings']) if ('单次请求' in h or '输入范围' in h)]
                    if len(indexes)!=1 or indexes[0]>=len(row['cells']):continue
                    text=re.sub(r'\s+','',row['cells'][indexes[0]]).lower().replace('tokens','输入').replace('token','输入').replace('≤','<=')
                    text=re.sub(r'^0<','',text)
                    try:same=_range(text)==_range(expected_row['range'])
                    except ValueError:same=False
                    if same:band_rows.append(row)
            for dimension,label in [('input_token','输入'),('output_token','输出')]:
                values=set()
                for row in band_rows:
                    indexes=[i for i,h in enumerate(row['column_headings']) if label in h and '单次请求' not in h and '范围' not in h]
                    if len(indexes)!=1 or indexes[0]>=len(row['cells']):continue
                    values.update(Decimal(v) for v in re.findall(r'(\d+(?:\.\d+)?)\s*元',row['cells'][indexes[0]]))
                try:expected={Decimal(p['price']) for p in expected_row['prices'] if p.get('type')==dimension}
                except (InvalidOperation,ValueError,TypeError):expected=set()
                if not values or not expected:
                    compared.append(None)
                else:compared.append(expected==values)
        if found and not api_rows:
            matches[model]={'status':'UNAVAILABLE','reason':'API_REFERENCE_UNAVAILABLE'}
        if compared and None not in compared:
            match=all(compared)
            matches[model]={'status':'MATCH' if match else 'CONFLICT','reason':None if match else 'API_DOCUMENT_PRICE_DISAGREEMENT'}
    window_text=' '.join(' '.join(parser.deepseek_text).split())
    windows=re.findall(r'忙时为北京时间\s*(\d{2}:\d{2})\s*至\s*(\d{2}:\d{2})[，,]\s*闲时为北京时间\s*(\d{2}:\d{2})\s*至次日\s*(\d{2}:\d{2})',window_text)
    time_windows={}
    if len(set(windows))==1:
        peak_start,peak_end,off_start,off_end=windows[0]
        if peak_start==off_end and peak_end==off_start:
            for model in models:
                if model.startswith('deepseek-') and any(r['model']==model for r in rows):
                    time_windows[model]={'timezone':'Asia/Shanghai','offpeak_start':off_start,'offpeak_end':off_end,
                        'source_ref':DOCUMENT_URL+'#deepseek','parser_version':PARSER}
    return {'parser_version':PARSER,'rows':rows,'matches':matches,'time_windows':time_windows,
            'rule_source_refs':sorted(set(parser.links))}


