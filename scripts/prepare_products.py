"""Normalize supplied Poizon snapshots without modifying raw files."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from uuid import UUID, uuid5

from canonical import canonicalize
from product_data import ROOT, attributes_text, embedding_text, dump, jsonl

NAMESPACE = UUID('99e355af-50fe-4ba6-8888-4827009c4107')
INDEX = 'poizon_products_v1'


def load_snapshot(path):
    raw = path.read_text(encoding='utf-8')
    repairs = []
    try:
        return json.loads(raw), repairs
    except json.JSONDecodeError:
        # These exports delimit base-property values with a propertyValueId field.
        # Escape only otherwise unescaped quotes inside that exact value boundary.
        pattern = r'("value"\s*:\s*")(.*?)("\s*,\s*"propertyValueId"\s*:)'
        def fix(match):
            value = match[2]
            corrected = re.sub(r'(?<!\\)"', r'\\"', value)
            if value != corrected:
                repairs.append({'kind': 'escape_property_value_quotes', 'offset': match.start(2), 'original_value': value})
            return match[1] + corrected + match[3]
        fixed = re.sub(pattern, fix, raw, flags=re.S)
        try:
            obj = json.loads(fixed)
        except json.JSONDecodeError:
            obj = json.loads(fixed, strict=False)
            repairs.append({'kind': 'accept_literal_control_characters'})
        return obj, repairs


def normalize(envelope, path):
    if envelope.get('success') is not True:
        raise ValueError('Source response is not successful')
    # 合成快照自带 _synthetic 标记；真实快照没有这个键，走原路径。
    synthetic = envelope.get('_synthetic')
    provider = 'synthetic' if synthetic else 'poizon'
    data = envelope['data']
    detail = data['detail']
    spu_id = str(detail['spuId'])
    if spu_id != path.stem or not detail.get('title'):
        raise ValueError('SPU/file identity mismatch or missing title')
    attrs = {}
    def add(key, value):
        if key and value is not None and str(value).strip():
            values = attrs.setdefault(str(key), [])
            if str(value).strip() not in values:
                values.append(str(value).strip())
    base = data.get('baseProperties', {})
    for item in base.get('list', []):
        add(item.get('key'), item.get('value'))
    brands = base.get('brandList', [])
    for brand in brands:
        add('品牌', brand.get('brandName'))
    sale = data.get('saleProperties', {}).get('list', [])
    for item in sale:
        add(item.get('name'), item.get('value'))
    attributes = {k: v[0] if len(v) == 1 else v for k, v in attrs.items()}
    lookup = {str(p['propertyValueId']): p for p in sale}
    skus = []
    for sku in data.get('skus', []):
        props = []
        for prop in sku.get('properties', []):
            source = lookup.get(str(prop['propertyValueId']), {})
            props.append({'property_value_id': str(prop['propertyValueId']),
                          'name': source.get('name'), 'value': source.get('value'),
                          'level': prop.get('level')})
        skus.append({'sku_id': str(sku['skuId']), 'status_raw': sku.get('status'),
                     'auth_price_raw': sku.get('authPrice'), 'properties': props})
    return {
        'product_id': str(uuid5(NAMESPACE, f'{provider}:spu:' + spu_id)),
        'spu_id': spu_id, 'title': detail['title'], 'subtitle': detail.get('subTitle') or '',
        'attributes': attributes, 'attributes_text': attributes_text(attributes),
        'description': detail.get('desc') or '',
        'category': detail.get('categoryName') or None,
        'category_id': str(detail['categoryId']),
        'category_path_ids': [str(detail[k]) for k in ('level1CategoryId', 'level2CategoryId', 'categoryId') if detail.get(k) is not None],
        'brand_names': [b['brandName'] for b in brands if b.get('brandName')],
        'brand_id': str(detail['brandId']) if detail.get('brandId') is not None else None,
        'article_number': detail.get('articleNumber') or None,
        'image_url': detail.get('logoUrl') or None,
        'release_date_raw': detail.get('sellDate') or None,
        'auth_price_raw': detail.get('authPrice'),
        'skus': skus, 'size_info': data.get('sizeInfo', {}),
        'source': {'provider': provider, 'file': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'collected_at': None, 'live_verified': False,
                   **({'synthetic': synthetic} if synthetic else {})},
    }


def apply_overlay(product, overlay):
    """把合成层写进商品，并在 source.synthetic 里逐字段留痕。原值非空的字段不覆盖。"""
    patch = (overlay or {}).get('products', {}).get(product['product_id'])
    if not patch:
        return False
    applied = []
    for key in ('description', 'subtitle'):
        if key in patch and not (product.get(key) or '').strip():
            product[key] = patch[key]
            applied.append(key)
    prices = patch.get('sku_prices') or {}
    filled = [sku['sku_id'] for sku in product['skus']
              if not sku.get('auth_price_raw') and sku['sku_id'] in prices]
    for sku in product['skus']:
        if sku['sku_id'] in filled:
            sku['auth_price_raw'] = prices[sku['sku_id']]
    if not applied and not filled:
        return False
    product['source']['synthetic'] = {
        'generator': overlay.get('generator'), 'version': overlay.get('version'),
        'generated_at': overlay.get('generated_at'), 'whole_record': False,
        'fields': applied,
        **({'sku_auth_price_raw': filled, 'sku_price_method': patch.get('sku_price_method')} if filled else {}),
    }
    return True


def synthetic_report(products):
    marks = [p['source'].get('synthetic') for p in products]
    if not any(marks):
        return {}
    whole = [m for m in marks if m and m.get('whole_record')]
    return {'synthetic_products': len(whole),
            'synthetic_field_products': sum(1 for m in marks if m and not m.get('whole_record')),
            'synthetic_sku_prices': sum(len(m.get('sku_auth_price_raw', [])) for m in marks if m)}


def mapping():
    keywords = ['product_id', 'spu_id', 'category', 'category_id', 'category_path_ids',
                'brand_names', 'brand_id', 'brand_key', 'brand_display', 'color_keys', 'material_keys',
                'article_number', 'release_date_raw']
    props = {k: {'type': 'keyword'} for k in keywords}
    props.update({k: {'type': 'text', 'analyzer': 'cjk'} for k in ['title', 'attributes_text']})
    props.update({k: {'type': 'text', 'index': False} for k in ['description', 'subtitle', 'image_url']})
    props.update({k: {'type': 'object', 'enabled': False} for k in ['attributes', 'skus', 'size_info', 'source']})
    props['auth_price_raw'] = {'type': 'long', 'index': False, 'doc_values': False}
    return {'settings': {'number_of_shards': 1, 'number_of_replicas': 0,
                         'index.query.default_field': ['title', 'attributes_text']},
            'mappings': {'dynamic': 'strict', 'properties': props}}


def generate(source, output, overlay=None):
    products, repairs, rejected = [], [], []
    directories = [source] if isinstance(source, Path) else list(source)
    files = sorted((path for directory in directories for path in directory.glob('*.json')),
                   key=lambda path: (directories.index(path.parent), path.name))
    if not files:
        raise ValueError('No raw JSON files found')
    for path in files:
        try:
            raw, fixes = load_snapshot(path)
            product = normalize(raw, path)
            canonicalize(product)
            apply_overlay(product, overlay)
            products.append(product)
            if fixes:
                repairs.append({'file': path.name, 'repairs': fixes})
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({'file': path.name, 'error': str(exc)})
    if len({p['product_id'] for p in products}) != len(products):
        raise ValueError('Duplicate SPU identity')
    output.mkdir(parents=True, exist_ok=True)
    jsonl(output / 'products.jsonl', products)
    bulk = []
    for product in products:
        bulk.extend([{'index': {'_index': INDEX, '_id': product['product_id']}}, product])
    jsonl(output / 'es.bulk.jsonl', bulk)
    jsonl(output / 'qdrant.inputs.jsonl', [{'id': p['product_id'], 'text': embedding_text(p), 'payload': p} for p in products])
    dump(output / 'es.index.json', mapping())
    dump(output / 'es.search.json', {'size': 10, 'query': {'multi_match': {
        'query': '银色 项链', 'fields': ['title^2', 'attributes_text']}}})
    dump(output / 'quality-report.json', {
        'files': len(files), 'accepted': len(products), 'repaired_files': repairs, 'rejected': rejected,
        'categories': dict(Counter(p['category'] for p in products)),
        'empty_descriptions': sum(not p['description'] for p in products),
        'sku_count': sum(len(p['skus']) for p in products),
        'unresolved_sku_properties': sum(not prop['name'] for p in products for s in p['skus'] for prop in s['properties']),
        'max_embedding_characters': max((len(embedding_text(p)) for p in products), default=0),
        **synthetic_report(products),
    })
    print(f'{len(products)}/{len(files)} products; {len(repairs)} repaired files; {len(rejected)} rejected')
    if rejected:
        raise ValueError('Unresolved source errors: see quality-report.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, nargs='+', default=[ROOT / 'fixtures/raw/poizon'])
    parser.add_argument('--output', type=Path, default=ROOT / 'fixtures/products')
    parser.add_argument('--overlay', type=Path, default=None, help='合成层 overlay.json；省略则只导出真实数据')
    args = parser.parse_args()
    overlay = json.loads(args.overlay.read_text(encoding='utf-8')) if args.overlay else None
    generate(args.source, args.output, overlay)
