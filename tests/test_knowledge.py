import json
from pathlib import Path

import pytest

from backend.catalog_schema import SlotError
from backend.knowledge import KnowledgeBase, load_default
from backend.need import UserNeed

ROOT = Path(__file__).resolve().parents[1]
KB = load_default(ROOT)
SCHEMA_PATH = ROOT / 'fixtures/kb/scene-rules.json'


def need_with(*constraints):
    return UserNeed(intent='discover', constraints=tuple(constraints))


def test_every_rule_targets_a_field_the_catalog_can_answer():
    # 这条测试就是上一版知识库失效的那个缺口：鞋服字段 + 珠宝目录。
    for rule in KB.rules:
        for target in rule.derive:
            KB.schema.validate(target['field'], target['op'], target['value'])
            assert target['evidence']


def test_a_rule_on_a_missing_field_fails_to_load(tmp_path):
    raw = json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))
    raw['rules'] = [{'id': 'SR-X', 'scene': 'x', 'triggers': ['跑步机'], 'state': {},
                     'derive': [{'field': 'cushion_level', 'op': '>=', 'value': 7,
                                 'evidence': '低跑量人群缓震优先'}]}]
    path = tmp_path / 'kb.json'
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
    with pytest.raises(SlotError, match='unknown field'):
        KnowledgeBase.load(path, KB.schema, KB.bounds)


def test_a_derived_rule_without_evidence_fails_to_load(tmp_path):
    raw = json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))
    raw['rules'] = [{'id': 'SR-Y', 'scene': 'y', 'triggers': ['送礼'], 'state': {},
                     'derive': [{'field': 'category', 'op': '==', 'value': '项链'}]}]
    path = tmp_path / 'kb.json'
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
    with pytest.raises(ValueError, match='needs evidence'):
        KnowledgeBase.load(path, KB.schema, KB.bounds)


def test_state_fields_must_not_shadow_catalog_fields(tmp_path):
    raw = json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))
    raw['state_fields']['category'] = {'domain': ['x'], 'note': 'oops'}
    path = tmp_path / 'kb.json'
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
    with pytest.raises(ValueError, match='collides with a catalog field'):
        KnowledgeBase.load(path, KB.schema, KB.bounds)


def test_gift_phrase_derives_the_jewelry_categories():
    constraints, trace = KB.derive('想买条项链送女朋友')
    by_field = {c.field_name: c for c in constraints}
    assert by_field['occasion'].value == 'gift'
    category = by_field['category']
    assert category.kind == 'derived'
    assert category.derived_from == ('occasion',)
    assert '项链' in category.value and '手镯' in category.value
    assert category.evidence and trace[0]['rule'] == 'SR-G01'


def test_derived_constraints_die_when_the_scene_changes():
    constraints, _ = KB.derive('送女朋友的生日礼物')
    need = need_with(*constraints)
    after, split = need.rewrite([
        type(constraints[0])(field_name='occasion', op='==', value='self', kind='hard')])
    # 场合从送礼改成自用，由它推出的类目范围自动作废，而不是继续累加。
    assert [c.field_name for c, _ in split.discarded] == ['category']
    assert after.of('category') is None


def test_material_phrases_have_no_state_source_so_stay_plain():
    constraints, _ = KB.derive('要银的')
    material = next(c for c in constraints if c.field_name == 'material_keys')
    assert material.kind == 'hard'           # material_keys 覆盖 65%，够做硬过滤
    assert material.derived_from == () and material.evidence


def material_of(utterance):
    return next(c for c in KB.derive(utterance)[0] if c.field_name == 'material_keys')


def test_material_taxonomy_is_addressable_by_level():
    """「银的」取父级，「925银」取叶子——不必在宽窄之间二选一。"""
    assert material_of('要银的').value == '银'
    assert material_of('要925银的').value == '银/925银'
    assert material_of('要纯银的').value == '银/纯银'


def test_nested_triggers_resolve_by_specificity_not_file_order():
    # 「要925银的」里天然含有「银的」，两条规则都会命中。
    assert {r.id for r, _ in KB.match('要925银的')} == {'SR-M01', 'SR-M01b'}
    # 归并时层级更深的胜出，与规则在文件里的先后无关。
    assert material_of('要925银的').value == '银/925银'


def test_parallel_materials_combine_instead_of_competing():
    """银和钻石是两种材质，不是同一槽位的竞争值——漏掉任何一个都是错的。"""
    both = material_of('要银的带钻的')
    assert both.op == 'ALL' and set(both.value) == {'银', '宝石/钻石'}
    assert set(material_of('要金的带钻的').value) == {'金', '宝石/钻石'}


def test_low_coverage_fields_are_degraded_to_soft():
    constraints, _ = KB.derive('要降噪的')
    noise = next(c for c in constraints if c.field_name == '主动降噪')
    # 只有 6/121 件声明该字段，做硬门槛会静默淘汰其余，只能参与排序。
    assert noise.kind == 'derived' and KB.schema.field('主动降噪').constraint_kind == 'soft'
    boxed = next(c for c in KB.derive('要礼盒')[0] if c.field_name == '礼盒')
    assert KB.schema.field('礼盒').constraint_kind == 'soft'
    assert boxed.evidence


def test_unmatched_utterance_derives_nothing():
    assert KB.derive('今天天气怎么样') == ((), ())


def test_relative_phrase_needs_an_anchor():
    constraint, trace = KB.resolve_vague('有没有便宜点的', anchor=None)
    assert constraint is None and trace['needs_anchor'] is True
    constraint, trace = KB.resolve_vague('有没有便宜点的', anchor=3000)
    assert (constraint.field_name, constraint.op, constraint.value) == ('price_cny', '<=', 2100.0)
    # 锚点必须可审计，否则用户无从判断「便宜点」被理解成了多少。
    assert constraint.anchor_source == '便宜点 @ 3000'


def test_range_phrase_resolves_to_both_ends():
    constraint, _ = KB.resolve_vague('有没有同价位的', anchor=1000)
    assert constraint.op == 'BETWEEN' and constraint.value == (800.0, 1200.0)


def test_budget_conflict_is_detected_from_catalog_bounds():
    need = need_with(KB.schema.constrain('price_cny', '<=', 3000),
                     KB.schema.constrain('材质', 'CONTAINS', '钻'))
    conflict = KB.conflicts(need)[0]
    # 目录里含钻的最低 6500，这个预算买不到——由数据判定，不是模型感觉。
    assert conflict.field == '材质' and conflict.floor == 6500 and conflict.requested == 3000
    assert '6500' in conflict.reason


def test_affordable_budget_reports_no_conflict():
    need = need_with(KB.schema.constrain('price_cny', '<=', 20000),
                     KB.schema.constrain('材质', 'CONTAINS', '钻'))
    assert KB.conflicts(need) == ()
    assert KB.conflicts(need_with(KB.schema.constrain('材质', 'CONTAINS', '钻'))) == ()


def test_category_floor_conflict():
    need = need_with(KB.schema.constrain('price_cny', '<=', 2000),
                     KB.schema.constrain('category', '==', '耳钉'))
    conflict = KB.conflicts(need)[0]
    assert conflict.field == 'category' and conflict.floor == 6750
