import json
from pathlib import Path

import pytest

from backend.catalog_schema import CatalogSchema, SlotError

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = CatalogSchema.load(ROOT / 'fixtures/catalog/field-schema.json')


def test_schema_matches_the_catalog_it_was_derived_from():
    products = [json.loads(line) for line in
                (ROOT / 'fixtures/catalog/products.jsonl').read_text(encoding='utf-8').splitlines()
                if line.strip()]
    assert SCHEMA.products == len(products) == 121
    assert SCHEMA.skus == sum(len(p['skus']) for p in products) == 673
    categories = {p['category'] for p in products}
    assert set(SCHEMA.field('category').domain) == categories


def test_only_well_covered_fields_may_be_hard():
    for name in SCHEMA.hard_safe():
        assert SCHEMA.field(name).coverage_ratio >= 0.5
    # 覆盖率不足的字段仍可用，但只能当排序项。
    sparse = SCHEMA.field('镶嵌材质')
    assert sparse.filterable and not sparse.hard_filter_safe
    assert sparse.constraint_kind == 'soft'
    assert SCHEMA.constrain('镶嵌材质', '==', '未镶嵌').kind == 'soft'


def test_hard_filter_cost_is_reported():
    # 材质 只有 79/121 条声明，硬过滤会静默淘汰另外 42 条。
    assert SCHEMA.excluded_by('材质') == 121 - SCHEMA.field('材质').coverage == 42
    assert SCHEMA.excluded_by('category') == 0
    with pytest.raises(SlotError, match='unknown field'):
        SCHEMA.excluded_by('cushion_level')


def test_unknown_field_is_refused_not_guessed():
    # 旧轨迹里的鞋类字段在这份目录里不存在，必须报错而不是静默通过。
    for name in ('cushion_level', 'waterproof_mm', 'carbon_plate', 'required_power_w'):
        with pytest.raises(SlotError, match='unknown field'):
            SCHEMA.validate(name, '>=', 7)


def test_value_outside_the_domain_is_refused():
    assert SCHEMA.validate('category', '==', '项链') == '项链'
    with pytest.raises(SlotError, match='is not a value of'):
        SCHEMA.validate('category', '==', '智能马桶')
    with pytest.raises(SlotError, match='is not a value of'):
        SCHEMA.validate('color_keys', '==', '藏青')


def test_operator_must_suit_the_field_kind():
    with pytest.raises(SlotError, match='not valid for a enum field'):
        SCHEMA.validate('category', '>=', '项链')
    with pytest.raises(SlotError, match='not valid for a range field'):
        SCHEMA.validate('price_cny', 'CONTAINS', 3000)
    assert SCHEMA.validate('price_cny', '<=', 3000) == 3000.0


def test_between_takes_two_numbers_in_order():
    assert SCHEMA.validate('price_cny', 'BETWEEN', [1000, 5000]) == (1000.0, 5000.0)
    for bad in (3000, [3000], [1, 2, 3], ['a', 'b']):
        with pytest.raises(SlotError):
            SCHEMA.validate('price_cny', 'BETWEEN', bad)
    with pytest.raises(SlotError, match='exceeds upper'):
        SCHEMA.validate('price_cny', 'BETWEEN', [5000, 1000])


def test_price_rejects_nonsense_values():
    for bad in ('3000', None, -1, 0, True):
        with pytest.raises(SlotError):
            SCHEMA.validate('price_cny', '<=', bad)


def test_lists_are_only_allowed_with_set_operators():
    assert SCHEMA.validate('color_keys', 'IN', ['银', '金']) == ('银', '金')
    with pytest.raises(SlotError, match='takes a single value'):
        SCHEMA.validate('color_keys', '==', ['银', '金'])
    with pytest.raises(SlotError, match='empty'):
        SCHEMA.validate('color_keys', 'IN', [])


def test_non_filterable_fields_are_refused_with_their_reason():
    with pytest.raises(SlotError, match='identifier'):
        SCHEMA.validate('主货号', '==', 'GRP11153')
    with pytest.raises(SlotError, match='constant'):
        SCHEMA.validate('温馨提示', 'CONTAINS', '礼盒')


def test_constrain_produces_a_usable_constraint():
    constraint = SCHEMA.constrain('price_cny', '<=', 3000)
    assert (constraint.field_name, constraint.op, constraint.value) == ('price_cny', '<=', 3000.0)
    assert constraint.kind == 'hard'
    derived = SCHEMA.constrain('color_keys', '==', '银', kind='derived',
                               derived_from=('occasion',), evidence='送礼偏好银饰')
    assert derived.kind == 'derived' and derived.evidence == '送礼偏好银饰'


def test_values_are_trimmed_not_reinterpreted():
    assert SCHEMA.validate('category', '==', '  项链  ') == '项链'
    with pytest.raises(SlotError, match='must not be empty'):
        SCHEMA.validate('category', '==', '   ')
