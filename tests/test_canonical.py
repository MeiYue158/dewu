import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from canonical import BRAND_DISPLAY, ambiguous_brands, canonicalize

EXPORTS = [ROOT / 'fixtures/products/products.jsonl', ROOT / 'fixtures/catalog/products.jsonl']


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def test_ambiguous_brands_have_a_display_name():
    # 一个 brand_id 出现多种写法时必须在表里指定展示名，
    # 否则展示名会随商品顺序漂移。新数据引入新歧义时这里会失败。
    for path in EXPORTS:
        assert set(ambiguous_brands(rows(path))) <= set(BRAND_DISPLAY)


def test_source_spellings_are_never_edited():
    for path in EXPORTS:
        for product in rows(path):
            assert product['brand_display'] in (product['brand_names'] or [''])  \
                or product['brand_id'] in BRAND_DISPLAY
            # 原值必须原样保留，归一只发生在派生字段上。
            assert canonicalize(dict(product))['brand_names'] == product['brand_names']


def test_brand_key_collapses_spellings_but_not_brands():
    for path in EXPORTS:
        products = rows(path)
        by_key = {}
        for product in products:
            by_key.setdefault(product['brand_key'], set()).add(product['brand_display'])
        # 每个 key 恰好一个展示名；不同品牌不会被并到一起。
        assert all(len(names) == 1 for names in by_key.values())
        assert len(by_key) == len({p['brand_id'] for p in products})


def test_tiffany_spellings_share_one_key():
    products = rows(EXPORTS[1])
    tiffany = [p for p in products if p['brand_id'] == '10261']
    assert len({tuple(p['brand_names']) for p in tiffany}) == 2
    assert {p['brand_key'] for p in tiffany} == {'10261'}
    assert {p['brand_display'] for p in tiffany} == {'TIFFANY & CO./蒂芙尼'}


def test_mapping_indexes_the_derived_fields():
    for directory in ('fixtures/products', 'fixtures/catalog'):
        mapping = json.loads((ROOT / directory / 'es.index.json').read_text(encoding='utf-8'))
        properties = mapping['mappings']['properties']
        for field in ('brand_key', 'brand_display'):
            assert properties[field] == {'type': 'keyword'}
