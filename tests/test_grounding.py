import json
from pathlib import Path

from backend.grounding import check, evidence_of
from backend.knowledge import load_default
from backend.need import UserNeed
from backend.retrieval import Catalog, plan

ROOT = Path(__file__).resolve().parents[1]
KB = load_default(ROOT)
CATALOG = Catalog.load(ROOT / 'fixtures/catalog/products.jsonl')
BY_ID = {p['product_id']: p for p in CATALOG.products}


def necklaces(limit=3):
    need = UserNeed(intent='discover', constraints=(
        KB.schema.constrain('category', '==', '项链'),
        KB.schema.constrain('price_cny', '<=', 20000)))
    return CATALOG.search(plan(need, KB.schema), limit=limit).hits


HITS = necklaces()
CHEAPEST = min(hit.price_cny for hit in HITS)


def test_a_price_taken_from_the_results_is_supported():
    report = check(f'这条项链 {CHEAPEST:g} 元。', HITS, BY_ID)
    assert report.ok and report.claims[0].supported
    assert report.claims[0].source in ('price_cny', 'skus.price')


def test_an_invented_price_is_unsupported_and_the_sentence_is_dropped():
    report = check(f'这条项链 {CHEAPEST:g} 元。另一条只要 9 元。', HITS, BY_ID)
    assert not report.ok
    assert [c.text for c in report.unsupported] == ['9']
    # 剔除的是整句，不是只抹掉数字——留半句比删掉更容易误导。
    assert '9 元' not in report.stripped and f'{CHEAPEST:g} 元' in report.stripped


def test_counts_are_not_treated_as_product_facts():
    report = check('共 11 件符合条件。', HITS, BY_ID)
    assert report.ok and report.stripped == '共 11 件符合条件。'


def test_a_product_from_outside_this_turn_has_no_support():
    other = CATALOG.search(plan(UserNeed(intent='discover', constraints=(
        KB.schema.constrain('category', '==', '耳钉'),)), KB.schema), limit=1).hits
    price = other[0].price_cny
    # 这一轮检索的是项链，引用一件没被检索到的耳钉同样属于没有依据。
    report = check(f'还有一款 {price:g} 元的。', HITS, BY_ID)
    assert not report.ok and report.stripped == ''


def test_thousands_separators_do_not_break_the_match():
    report = check(f'价格 {CHEAPEST:,.0f} 元。', HITS, BY_ID)
    assert report.ok


def test_attribute_numbers_are_citable():
    """属性里的数字（尺寸、克重）也要能被引用，不只是价格。"""
    from backend.grounding import NUMBER
    for hit in HITS:
        for key, raw in BY_ID[hit.product_id]['attributes'].items():
            if key in ('主货号', '发售价格', '发售日期'):
                continue                    # 货号和发售价不是可引用的商品事实
            found = NUMBER.findall(str(raw))
            if found:
                report = check(f'{key}是 {found[0]}。', (hit,), BY_ID)
                assert report.ok, (key, raw, report.claims)
                assert report.claims[0].source == f'attributes.{key}'
                return
    raise AssertionError('命中结果里没有带数字的属性，用例失效')


def test_numbers_inside_a_product_name_are_supported():
    """「TIFFANY 1837 系列」里的 1837 有依据——漏登记会误删整句合法文案。"""
    from backend.grounding import NUMBER
    hit = next((h for h in HITS if NUMBER.findall(h.title)), None)
    assert hit is not None, '命中结果里没有带数字的商品名，用例失效'
    number = NUMBER.findall(hit.title)[0]
    report = check(f'推荐 {hit.title} 这款。', (hit,), BY_ID)
    assert report.ok, [c for c in report.claims if not c.supported]
    assert any(c.text == number and c.source == 'title' for c in report.claims)


def test_text_without_numbers_passes_through():
    report = check('这几款都挺适合送礼。', HITS, BY_ID)
    assert report.ok and report.claims == () and report.stripped


def test_evidence_table_is_limited_to_this_turn():
    table = evidence_of(HITS, BY_ID)
    assert all(hit.title in table for hit in HITS)
    # 没被检索到的商品不进证据表。
    missing = next(p for p in CATALOG.products
                   if p['product_id'] not in {h.product_id for h in HITS})
    assert missing['title'] not in table


def test_empty_results_support_nothing():
    report = check('有一款 1920 元的。', (), {})
    assert not report.ok and report.stripped == ''
