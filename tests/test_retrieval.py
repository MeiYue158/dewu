import json
from pathlib import Path

from backend.knowledge import load_default
from backend.need import Constraint, UserNeed
from backend.retrieval import Catalog, Query, plan

ROOT = Path(__file__).resolve().parents[1]
KB = load_default(ROOT)
SCHEMA = KB.schema
CATALOG = Catalog.load(ROOT / 'fixtures/catalog/products.jsonl')


def need_of(*constraints, intent='discover'):
    return UserNeed(intent=intent, constraints=tuple(constraints))


def run(*constraints, limit=10):
    return CATALOG.search(plan(need_of(*constraints), SCHEMA), limit=limit)


# ---- 翻译 ----

def test_state_fields_never_reach_the_query():
    query = plan(need_of(Constraint(field_name='occasion', op='==', value='gift', kind='hard'),
                         SCHEMA.constrain('category', '==', '项链')), SCHEMA)
    # occasion 是会话状态，目录没有这个字段，不该下发到检索。
    assert [f.field_name for f in query.filters] == ['category']


def test_hard_filters_and_soft_boosts_are_separated():
    query = plan(need_of(SCHEMA.constrain('category', '==', '项链'),
                         SCHEMA.constrain('镶嵌材质', '==', '未镶嵌')), SCHEMA)
    assert [f.field_name for f in query.filters] == ['category']
    assert [f.field_name for f in query.boosts] == ['镶嵌材质']


def test_low_coverage_filters_keep_undeclared_products():
    query = plan(need_of(SCHEMA.constrain('材质', 'CONTAINS', '银'),
                         SCHEMA.constrain('镶嵌材质', '==', '未镶嵌')), SCHEMA)
    policies = {f.field_name: f.missing_policy for f in query.filters + query.boosts}
    # 材质覆盖 65% 可以严格排除；镶嵌材质只有 18%，排除等于丢掉大半目录。
    assert policies['材质'] == 'exclude' and policies['镶嵌材质'] == 'keep'


# ---- 硬过滤 ----

def test_category_filter_matches_the_catalog_count():
    result = run(SCHEMA.constrain('category', '==', '项链'), limit=100)
    assert result.total == 25
    assert {hit.category for hit in result.hits} == {'项链'}


def test_in_operator_covers_every_listed_value():
    jewellery = ['项链', '手链', '戒指', '耳钉', '项坠/吊坠', '耳环', '手镯']
    result = run(SCHEMA.constrain('category', 'IN', jewellery), limit=200)
    assert result.total == 79
    assert set(SCHEMA.field('category').domain) >= {hit.category for hit in result.hits}


def test_contains_matches_substrings_of_free_text():
    result = run(SCHEMA.constrain('材质', 'CONTAINS', '钻'), limit=100)
    assert result.total == 29


def test_material_taxonomy_filters_by_level():
    broad = run(SCHEMA.constrain('material_keys', '==', '银'), limit=100)
    leaf = run(SCHEMA.constrain('material_keys', '==', '银/925银'), limit=100)
    assert broad.total == 15 and leaf.total == 11
    # 朴素子串会把「金属树脂」算成金、「水钻」算成钻石；层级标签不会。
    assert run(SCHEMA.constrain('material_keys', '==', '金'), limit=100).total == 49
    assert run(SCHEMA.constrain('material_keys', '==', '宝石/钻石'), limit=100).total == 28


def test_all_operator_requires_every_value():
    both = run(SCHEMA.constrain('material_keys', 'ALL', ['银', '宝石/钻石']), limit=100)
    silver = run(SCHEMA.constrain('material_keys', '==', '银'), limit=100)
    assert 0 < both.total < silver.total
    raw = {json.loads(l)['product_id']: json.loads(l)
           for l in (ROOT / 'fixtures/catalog/products.jsonl').read_text().splitlines()}
    for hit in both.hits:
        keys = set(raw[hit.product_id]['material_keys'])
        assert {'银', '宝石/钻石'} <= keys


def test_color_filter_uses_the_derived_keys():
    result = run(SCHEMA.constrain('color_keys', '==', '玫瑰金'), limit=100)
    assert result.total == 23


def test_filters_combine_as_conjunction():
    both = run(SCHEMA.constrain('category', '==', '项链'),
               SCHEMA.constrain('color_keys', '==', '银'), limit=100)
    assert 0 < both.total < 25


# ---- SKU 级语义 ----

def test_price_filter_runs_on_skus_and_reports_the_matching_ones():
    result = run(SCHEMA.constrain('category', '==', '项链'),
                 SCHEMA.constrain('price_cny', '<=', 5000), limit=100)
    assert result.total > 0
    for hit in result.hits:
        assert hit.price_cny <= 5000
        assert hit.sku_ids                      # 命中的是哪些 SKU 必须说清楚
    raw = {json.loads(l)['product_id']: json.loads(l)
           for l in (ROOT / 'fixtures/catalog/products.jsonl').read_text().splitlines()}
    for hit in result.hits:
        prices = {s['sku_id']: s['auth_price_raw'] / 100 for s in raw[hit.product_id]['skus']}
        assert all(prices[sku] <= 5000 for sku in hit.sku_ids)


def test_sku_conditions_must_hold_on_the_same_sku():
    """价格来自 A、尺码来自 B 的跨 SKU 假命中，是这类系统最隐蔽的错误。"""
    product = {
        'product_id': 'p1', 'title': 't', 'category': '戒指', 'brand_display': 'b',
        'attributes': {}, 'color_keys': [],
        'skus': [
            {'sku_id': 'cheap_small', 'auth_price_raw': 100000,
             'properties': [{'name': '尺码', 'value': '小号'}]},
            {'sku_id': 'dear_large', 'auth_price_raw': 900000,
             'properties': [{'name': '尺码', 'value': '大号'}]},
        ]}
    catalog = Catalog([product])
    query = plan(need_of(SCHEMA.constrain('price_cny', '<=', 2000),
                         SCHEMA.constrain('尺码', '==', '大号')), SCHEMA)
    # 便宜的是小号，大号很贵 —— 没有任何一个 SKU 同时满足。
    assert catalog.search(query).total == 0
    ok = plan(need_of(SCHEMA.constrain('price_cny', '<=', 2000),
                      SCHEMA.constrain('尺码', '==', '小号')), SCHEMA)
    hit = catalog.search(ok).hits[0]
    assert hit.sku_ids == ('cheap_small',)


# ---- 缺字段的代价要报出来 ----

def test_products_dropped_for_a_missing_field_are_counted():
    result = run(SCHEMA.constrain('材质', 'CONTAINS', '银'), limit=100)
    # 材质只有 79/121 条声明，另外 42 条是被字段缺失淘汰的，不是不满足条件。
    assert result.dropped_for_missing_field['材质'] == 42
    assert result.total + 42 <= 121


def test_keep_policy_does_not_drop_undeclared_products():
    strict = run(SCHEMA.constrain('镶嵌材质', '==', '未镶嵌', kind='hard'), limit=200)
    lenient = plan(need_of(SCHEMA.constrain('镶嵌材质', '==', '未镶嵌')), SCHEMA)
    # 低覆盖字段进了 boosts，不参与过滤，所以结果是全目录。
    assert lenient.filters == () and CATALOG.search(lenient, limit=200).total == 121
    assert strict.total < 121


# ---- soft 影响排序 ----

def test_soft_matches_rank_first_and_are_explained():
    query = plan(need_of(SCHEMA.constrain('category', '==', '手链'),
                         SCHEMA.constrain('镶嵌材质', '==', '未镶嵌')), SCHEMA)
    hits = CATALOG.search(query, limit=14).hits
    assert hits[0].score == 1.0 and '镶嵌材质' in hits[0].matched
    assert hits[-1].score == 0.0 and hits[-1].matched == ()
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:]))


def test_without_soft_constraints_ranking_falls_back_to_price():
    hits = run(SCHEMA.constrain('category', '==', '手链'), limit=14).hits
    prices = [h.price_cny for h in hits if h.price_cny is not None]
    assert prices == sorted(prices)


# ---- 0 命中归因 ----

def test_probe_identifies_the_blocking_constraint():
    query = plan(need_of(SCHEMA.constrain('category', '==', '耳钉'),
                         SCHEMA.constrain('price_cny', '<=', 2000)), SCHEMA)
    assert CATALOG.search(query).total == 0
    # probe 只报事实：放宽任一条各能放出多少。
    counts = {p.field_name: p.count_if_relaxed for p in CATALOG.probe(query)}
    assert counts['price_cny'] == 10            # 耳钉共 10 件
    assert counts['category'] > counts['price_cny']   # 放弃类目结果更多，但那不是用户想要的
    # 建议放宽时保护用户锚定的类目，于是瓶颈落在预算上。
    blocking = CATALOG.blocking(query)
    assert blocking.field_name == 'price_cny' and blocking.count_if_relaxed == 10
    assert 'category' not in {p.field_name for p in CATALOG.relax_options(query)}


def test_probe_returns_nothing_useful_when_no_single_relaxation_helps():
    query = plan(need_of(SCHEMA.constrain('category', '==', '耳钉'),
                         SCHEMA.constrain('color_keys', '==', '绿')), SCHEMA)
    if CATALOG.search(query).total == 0:
        probes = CATALOG.probe(query)
        assert all(p.count_if_relaxed >= 0 for p in probes)


# ---- 与理解层串起来 ----

def test_a_ready_interpretation_produces_results():
    need = need_of(SCHEMA.constrain('category', '==', '项链'),
                   SCHEMA.constrain('price_cny', '<=', 20000),
                   SCHEMA.constrain('材质', 'CONTAINS', '银'))
    result = CATALOG.search(plan(need, SCHEMA), limit=5)
    assert result.total > 0
    for hit in result.hits:
        assert hit.category == '项链' and hit.price_cny <= 20000
