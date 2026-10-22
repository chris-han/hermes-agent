from __future__ import annotations

import hashlib
import re
from decimal import Decimal
from typing import Any

from .base import extract_generic


def _rows(canonical):
    rows = []
    blocks = [b for b in canonical.get('blocks', []) if len(b.get('bbox', [])) == 4]
    for block in sorted(blocks, key=lambda b: (b.get('page', 1), b['bbox'][1], b['bbox'][0])):
        center = (block['bbox'][1] + block['bbox'][3]) / 2
        row = next((r for r in reversed(rows) if r['page'] == block.get('page', 1) and abs(r['y'] - center) < 3), None)
        if row is None:
            row = {'page': block.get('page', 1), 'y': center, 'blocks': []}
            rows.append(row)
        row['blocks'].append(block)
    for row in rows:
        row['blocks'].sort(key=lambda b: b['bbox'][0])
    return rows


def extract(canonical: dict[str, Any], *, preferred_currency: str | None = None) -> dict[str, Any]:
    result = extract_generic(canonical, preferred_currency=preferred_currency or 'CNY', adapter_name='china_digital_invoice_v1')
    rows = _rows(canonical)
    if not rows:
        return result
    invoice = result['document']
    # Native PDF observation order is not reading order. Resolve labels and
    # table columns geometrically without altering their original evidence IDs.
    invoice['evidence'] = []
    def evidence(field, blocks):
        ids = list(dict.fromkeys(i for b in blocks for i in b.get('evidence_ids', [])))
        if ids:
            invoice['evidence'].append({'field': field, 'evidence_ids': ids, 'confidence': None})
    for row in rows:
        bs = row['blocks']
        for index, b in enumerate(bs):
            label = re.sub(r'\s+', '', b['text'])
            following = bs[index + 1:]
            for prefix, field in [('发票号码', 'invoice_number'), ('开票日期', 'invoice_date')]:
                if label.startswith(prefix):
                    value = label[len(prefix):].lstrip('：:')
                    source = b
                    if not value and following:
                        source = following[0]; value = source['text'].strip()
                    if field == 'invoice_date':
                        value = value.replace('年', '-').replace('月', '-').replace('日', '')
                    invoice[field] = value
                    evidence(field, [source])
            if label in ('（小写）', '(小写)') and following:
                amount = following[0]
                value = amount['text'].strip().lstrip('¥￥')
                if re.fullmatch(r'\d+\.\d{2}', value):
                    invoice['total'] = value; evidence('total', [amount])
        names = [i for i,b in enumerate(bs) if b['text'].strip() in ('名称：', '名称:')]
        if len(names) == 2:
            for index, role in zip(names, ['customer', 'supplier']):
                if index + 1 < len(bs):
                    source = bs[index+1]
                    invoice[role] = {'name': source['text'], 'taxpayer_id': None}
                    evidence(role + '.name', [source])
    header = next((r for r in rows if any(b['text'] == '项目名称' for b in r['blocks'])), None)
    items = []
    if header:
        hb = header['blocks']
        def column(label):
            # Headers may be split into separate authored glyph blocks.
            for i,b in enumerate(hb):
                for end in range(i + 1, min(i + 4, len(hb)) + 1):
                    if ''.join(x['text'].strip() for x in hb[i:end]) == label:
                        return (b['bbox'][0] + hb[end-1]['bbox'][2]) / 2
            return None
        labels = ['项目名称','规格型号','单位','数量','单价','金额','税率/征收率','税额']
        centers = [column(label) for label in labels]
        if all(c is not None for c in centers):
            bounds = [(a+b)/2 for a,b in zip(centers, centers[1:])]
            def cells(row):
                groups = [[] for _ in centers]
                for b in row['blocks']:
                    x = (b['bbox'][0]+b['bbox'][2])/2
                    groups[sum(x > edge for edge in bounds)].append(b)
                return groups
            table_closed = False
            for row in rows:
                if row['page'] != header['page'] or row['y'] <= header['y']:
                    continue
                cs = cells(row)
                texts = [''.join(b['text'] for b in cell).strip() for cell in cs]
                joined = ''.join(texts)
                if joined.startswith('合计'):
                    for field, col in [('subtotal',5),('tax',7)]:
                        value = texts[col].lstrip('¥￥')
                        if re.fullmatch(r'-?\d+\.\d{2}', value):
                            invoice[field] = value; evidence(field,cs[col])
                if joined.startswith(('小计', '合计')):
                    table_closed = True
                if table_closed:
                    continue
                if not texts[0].startswith('*') or not re.fullmatch(r'-?\d+\.\d{2}',texts[5]):
                    # Preserve wrapped item descriptions, not summary rows.
                    if items and texts[0] and not any(texts[2:]) and not joined.startswith(('合计','小计','价税合计','备注','开票人')):
                        items[-1]['description'] += texts[0]
                        evidence(f"line_items.{len(items)-1}.description", cs[0])
                    continue
                item = {'line_number':len(items)+1,'description':texts[0], 'quantity':texts[3] or None,
                    'unit_price':texts[4] or None,'amount':texts[5], 'tax_rate':texts[6], 'tax':texts[7] or None,
                    'related_line_number':None}
                if Decimal(item['amount']) < 0 and items and items[-1]['description'].startswith(texts[0]):
                    item['related_line_number'] = items[-1]['line_number']
                items.append(item)
                for field,col in [('description',0),('quantity',3),('unit_price',4),('amount',5),('tax_rate',6),('tax',7)]:
                    evidence(f'line_items.{len(items)-1}.{field}', cs[col])
    invoice['line_items'] = items
    findings = []
    if all(invoice.get(f) is not None for f in ['subtotal','tax','total']):
        if Decimal(invoice['subtotal']) + Decimal(invoice['tax']) != Decimal(invoice['total']):
            findings.append({'code':'ARITHMETIC_RECONCILIATION_FAILED','severity':'review'})
    if items:
        for field,target in [('amount','subtotal'),('tax','tax')]:
            if invoice.get(target) is None or any(r[field] is None for r in items) or sum(Decimal(r[field]) for r in items) != Decimal(invoice[target]):
                findings.append({'code':'LINE_ITEMS_RECONCILIATION_FAILED','field':field,'severity':'review'})
    unresolved = [field for field in ['invoice_number','invoice_date','subtotal','tax','total'] if not invoice.get(field)]
    if not invoice.get('supplier',{}).get('name'): unresolved.append('supplier.name')
    if not invoice.get('customer'): unresolved.append('customer.name')
    invoice['validation']['findings'] = findings
    invoice['validation']['duplicate_fingerprint'] = 'sha256:' + hashlib.sha256('|'.join(str(invoice.get(f,'')) for f in ['supplier','invoice_number','invoice_date','total']).encode()).hexdigest()
    return {'document':invoice,'findings':findings,'unresolved_fields':unresolved,'status':'review_required' if findings or unresolved else 'admissible'}
