import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_products import generate
from product_data import attributes_text, embedding_text
from synthesize_products import build_digital, build_overlay, describe

REAL = ROOT / 'fixtures/products'
CATALOG = ROOT / 'fixtures/catalog'
RAW = ROOT / 'fixtures/raw/poizon'
SYNTH_RAW = ROOT / 'fixtures/raw/synthetic'
OVERLAY = json.loads((ROOT / 'fixtures/synthetic/overlay.json').read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def test_generation_is_reproducible(tmp_path):
    real = rows(REAL / 'products.jsonl')
    assert build_overlay(real, OVERLAY['generated_at']) == OVERLAY
    written = build_digital(tmp_path, OVERLAY['generated_at'])
    assert len(written) == 18
    for path in tmp_path.iterdir():
        assert path.read_bytes() == (SYNTH_RAW / path.name).read_bytes()


def test_catalog_matches_sources(tmp_path):
    generate([RAW, SYNTH_RAW], tmp_path, OVERLAY)
    for path in tmp_path.iterdir():
        assert path.read_bytes() == (CATALOG / path.name).read_bytes()


def test_every_synthetic_value_is_marked():
    real = {r['product_id']: r for r in rows(REAL / 'products.jsonl')}
    for product in rows(CATALOG / 'products.jsonl'):
        mark = product['source'].get('synthetic')
        if product['source']['provider'] == 'synthetic':
            assert mark['whole_record'] is True and product['product_id'] not in real
            continue
        # 真实商品：凡是与原目录不同的字段，都必须出现在标记里。
        before = real[product['product_id']]
        for field in ('description', 'subtitle'):
            assert (product[field] != before[field]) == (field in mark['fields'])
        filled = [s['sku_id'] for s, b in zip(product['skus'], before['skus'])
                  if s['auth_price_raw'] != b['auth_price_raw']]
        assert filled == mark.get('sku_auth_price_raw', [])


def test_overlay_never_overwrites_existing_values():
    real = {r['product_id']: r for r in rows(REAL / 'products.jsonl')}
    for product in rows(CATALOG / 'products.jsonl'):
        before = real.get(product['product_id'])
        if before is None:
            continue
        for field in ('title', 'category', 'attributes', 'attributes_text', 'brand_names'):
            assert product[field] == before[field]
        for sku, original in zip(product['skus'], before['skus']):
            if original['auth_price_raw']:
                assert sku['auth_price_raw'] == original['auth_price_raw']


def test_filled_prices_follow_the_recorded_method():
    for product in rows(CATALOG / 'products.jsonl'):
        mark = product['source'].get('synthetic') or {}
        if not mark.get('sku_auth_price_raw'):
            continue
        filled = {s['sku_id']: s['auth_price_raw'] for s in product['skus']
                  if s['sku_id'] in mark['sku_auth_price_raw']}
        assert all(v > 0 for v in filled.values())
        if mark['sku_price_method'] == 'spu_auth_price_cny_x100':
            # SKU 单位是分，SPU 单位是元。
            assert set(filled.values()) == {product['auth_price_raw'] * 100}
        else:
            assert mark['sku_price_method'].endswith('median_of_real_skus')
            assert not product['auth_price_raw']


def test_descriptions_only_restate_existing_attributes():
    for product in rows(CATALOG / 'products.jsonl'):
        if 'description' not in (product['source'].get('synthetic') or {}).get('fields', []):
            continue
        assert product['description'] == describe(product)
        values = {str(v) for value in product['attributes'].values()
                  for v in (value if isinstance(value, list) else [value])}
        body = product['description'].split('。', 1)[1].rstrip('。')
        for clause in filter(None, body.split('，')):
            assert any(value and value in clause for value in values), clause


def test_synthetic_text_never_reaches_the_vectors():
    # 向量只吃 title + attributes，所以补描述不会让已导出的 384 维向量失效。
    # 不能断言描述字面不出现：它本就由属性拼成，字面必然与属性文本重叠。
    # 真正的不变量是 embedding_text 的构成只有这两个字段。
    for product in rows(CATALOG / 'products.jsonl'):
        assert embedding_text(product) == (
            f"标题：{product['title']}\n属性：{attributes_text(product['attributes'])}")
    inputs = rows(CATALOG / 'qdrant.inputs.jsonl')
    assert all(row['text'] == embedding_text(row['payload']) for row in inputs)


def test_real_catalog_is_untouched():
    real = rows(REAL / 'products.jsonl')
    assert len(real) == 103
    assert all(p['source']['provider'] == 'poizon' for p in real)
    assert all('synthetic' not in p['source'] for p in real)
    assert sum(not p['description'] for p in real) == 103


def test_catalog_shape():
    catalog = rows(CATALOG / 'products.jsonl')
    report = json.loads((CATALOG / 'quality-report.json').read_text(encoding='utf-8'))
    assert len(catalog) == 121 and len({p['product_id'] for p in catalog}) == 121
    assert report['sku_count'] == 673 and report['empty_descriptions'] == 0
    assert report['synthetic_products'] == 18
    assert report['synthetic_field_products'] == 103
    assert report['synthetic_sku_prices'] == 76
    assert not any(s['auth_price_raw'] in (None, 0) for p in catalog for s in p['skus'])
