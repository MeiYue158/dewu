import json
from pathlib import Path

from backend.knowledge import load_default
from backend.need import UserNeed
from backend.retrieval import Catalog, plan

ROOT = Path(__file__).resolve().parents[1]
KB = load_default(ROOT)
SCHEMA = KB.schema
CATALOG = Catalog.load(ROOT / 'fixtures/catalog/products.jsonl')


def need_of(*constraints):
    return UserNeed(intent='discover', constraints=tuple(constraints))


def empty(*constraints):
    need = need_of(*constraints)
    query = plan(need, SCHEMA)
    assert CATALOG.search(query, limit=0).total == 0
    return need, query


EARRINGS = (SCHEMA.constrain('category', '==', '耳钉'),
            SCHEMA.constrain('price_cny', '<=', 2000))
EARRINGS_SILVER = EARRINGS + (SCHEMA.constrain('material_keys', '==', '银/925银'),)


def test_a_dead_end_always_offers_a_way_out():
    for constraints in (EARRINGS, EARRINGS_SILVER,
                        (SCHEMA.constrain('category', '==', '项链'),
                         SCHEMA.constrain('material_keys', '==', '铂金'),
                         SCHEMA.constrain('price_cny', '<=', 3000))):
        _, query = empty(*constraints)
        options = CATALOG.relaxations(query)
        assert options, '0 命中必须给出可操作的出口'
        assert all(option.result_count > 0 for option in options)


def test_widening_a_boundary_is_preferred_over_dropping_it():
    _, query = empty(*EARRINGS)
    options = CATALOG.relaxations(query)
    # 「预算调到 6750」比「取消预算」保留更多意图，所以排在前面。
    assert options[0].kind == 'widen'
    assert options[0].patch == {'price_cny': ('<=', 6750.0)}
    assert options[0].result_count == 1
    assert any(option.kind == 'drop' for option in options)


def test_the_widened_boundary_is_the_catalog_floor_not_a_guess():
    _, query = empty(*EARRINGS)
    widened = CATALOG.relaxations(query)[0]
    products = [json.loads(line) for line in
                (ROOT / 'fixtures/catalog/products.jsonl').read_text().splitlines()]
    cheapest = min(sku['auth_price_raw'] / 100
                   for p in products if p['category'] == '耳钉'
                   for sku in p['skus'] if sku.get('auth_price_raw'))
    assert widened.patch['price_cny'][1] == cheapest == 6750.0


def test_pairs_are_tried_when_no_single_relaxation_helps():
    _, query = empty(*EARRINGS_SILVER)
    singles = [p for p in CATALOG.probe(query)
               if p.field_name != 'category' and p.count_if_relaxed > 0]
    assert singles == [], '这个用例的前提是单条放宽都解不开'
    options = CATALOG.relaxations(query)
    assert len(options) >= 2                       # T-010 要求至少两个选项
    assert all(len(option.fields) >= 2 for option in options)


def test_a_combination_keeps_more_intent_than_dropping_both():
    _, query = empty(*EARRINGS_SILVER)
    best = CATALOG.relaxations(query)[0]
    # 丢材质 + 把预算挪到目录下界，好过把两条都取消。
    assert best.kind == 'widen'
    assert best.patch == {'material_keys': ('==', None), 'price_cny': ('<=', 6750.0)}


def test_every_patch_is_executable_and_actually_yields_results():
    """T-010 的 patch 可执行率：卡片上的每个按钮都要真能改写状态并重查。"""
    for constraints in (EARRINGS, EARRINGS_SILVER):
        need, query = empty(*constraints)
        for option in CATALOG.relaxations(query):
            revised = option.apply(need, SCHEMA)
            found = CATALOG.search(plan(revised, SCHEMA), limit=None)
            assert found.total == option.result_count, option.label


def test_the_anchor_category_is_never_proposed_for_relaxation():
    _, query = empty(*EARRINGS_SILVER)
    for option in CATALOG.relaxations(query):
        assert 'category' not in option.fields
    # 显式放开保护后才会出现——策略是参数，不是写死的。
    unprotected = CATALOG.relaxations(query, protect=())
    assert any('category' in option.fields for option in unprotected)


def test_relaxation_preserves_the_monotonicity_rule():
    """放宽后硬约束数不得增加（T-012 的自动回归项）。"""
    need, query = empty(*EARRINGS_SILVER)
    for option in CATALOG.relaxations(query):
        revised = option.apply(need, SCHEMA)       # apply 内部声明 relaxation=True
        assert len(revised.hard) <= len(need.hard)


def test_a_query_with_results_offers_nothing():
    need = need_of(SCHEMA.constrain('category', '==', '项链'))
    assert CATALOG.relaxations(plan(need, SCHEMA)) == ()


def test_options_are_ordered_by_how_much_intent_they_keep():
    _, query = empty(*(SCHEMA.constrain('category', '==', '项链'),
                       SCHEMA.constrain('material_keys', '==', '铂金'),
                       SCHEMA.constrain('price_cny', '<=', 3000)))
    options = CATALOG.relaxations(query)
    kinds = [option.kind for option in options]
    assert kinds == sorted(kinds, key=lambda k: {'widen': 0, 'drop': 1}[k])
