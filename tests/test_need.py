import pytest

from backend.need import ACTIONS, Constraint, Entity, UserNeed


def hard(name, value, op='=='):
    return Constraint(field_name=name, op=op, value=value, kind='hard')


def soft(name, value, op='<='):
    return Constraint(field_name=name, op=op, value=value, kind='soft')


def derived(name, value, sources, evidence, op='=='):
    return Constraint(field_name=name, op=op, value=value, kind='derived',
                      derived_from=tuple(sources), evidence=evidence)


# T-012 turn_2 的状态：锚定一双竞速鞋，已推导出新手约束。
def t012_state():
    return UserNeed(
        intent='discover',
        constraints=(
            hard('category', 'shoes/running'),
            hard('user_level', 'beginner'),
            hard('use_scenario', 'running'),
            # 「新手不穿碳板」这条规则同时消费了 user_level 和 use_scenario，
            # 两个来源都要声明，否则场景变化时不会触发失效。
            derived('carbon_plate', False, ['user_level', 'use_scenario'],
                    '碳板推进感强，需足踝力量支撑；新手易崴脚'),
            derived('cushion_level', 7, ['user_level', 'use_scenario'],
                    '低跑量人群缓震优先于回弹', op='>='),
            soft('price_cny', 800),
        ),
        entities=(Entity(sku='SKU_NIKE_ZOOMX_VF3', role='reference',
                         resolved_from='这个'),),
    )


def test_t012_turn3_three_way_split():
    """用户推翻需求：跑步 → 日常穿为主。"""
    before = t012_state()
    after, split = before.rewrite([
        hard('category', 'shoes/sneaker|shoes/running'),
        hard('use_scenario', 'daily_wear_primary'),
    ], relaxation=True)

    log = split.as_log()
    assert log['rewritten'] == [
        {'field': 'category', 'from': 'shoes/running', 'to': 'shoes/sneaker|shoes/running'},
        {'field': 'use_scenario', 'from': 'running', 'to': 'daily_wear_primary'},
    ]
    # 由旧场景推导而来的作废；与场景无关的保留。
    assert [c.field_name for c, _ in log['discarded'] and split.discarded] == \
           ['carbon_plate', 'cushion_level']
    assert sorted(log['retained']) == ['price_cny', 'user_level']
    assert after.of('carbon_plate') is None and after.of('cushion_level') is None
    assert after.of('user_level').value == 'beginner'
    assert after.of('price_cny').value == 800


def test_discarded_constraints_say_why():
    _, split = t012_state().rewrite([hard('use_scenario', 'daily_wear_primary')])
    reasons = {c.field_name: why for c, why in split.discarded}
    assert reasons['carbon_plate'] == 'derived from changed field(s): use_scenario'


def test_failure_propagates_through_a_chain():
    need = UserNeed(intent='discover', constraints=(
        hard('use_scenario', 'running'),
        derived('subtype', 'road', ['use_scenario'], '跑步机场地平整'),
        derived('outsole', 'flat', ['subtype'], '公路型大底对应平底纹'),
    ))
    assert need.invalidated_by({'use_scenario'}) == {'use_scenario', 'subtype', 'outsole'}
    after, split = need.rewrite([hard('use_scenario', 'trail')])
    # subtype 直接依赖，outsole 间接依赖，两级都要作废。
    assert [c.field_name for c, _ in split.discarded] == ['subtype', 'outsole']
    assert after.constraints == (hard('use_scenario', 'trail'),)


def test_partition_accounts_for_every_constraint():
    before = t012_state()
    _, split = before.rewrite([hard('use_scenario', 'daily_wear_primary')],
                              drop=('price_cny',))
    counted = len(split.retained) + len(split.discarded) + len(split.rewritten)
    assert counted == len(before.constraints)


def test_withdrawn_field_is_discarded_not_rewritten():
    after, split = t012_state().rewrite([], drop=('price_cny',), relaxation=True)
    assert ('price_cny', 'user withdrew this requirement') in \
           [(c.field_name, why) for c, why in split.discarded]
    assert after.of('price_cny') is None


def test_relaxation_must_not_add_hard_constraints():
    """T-012 的自动回归项：放宽之后硬约束数不得增加。"""
    need = UserNeed(intent='discover', constraints=(hard('category', 'x'),))
    with pytest.raises(ValueError, match='relaxation'):
        need.rewrite([hard('budget', 500), hard('color', 'red')], relaxation=True)
    # 不声明为放宽时，同样的改写是允许的。
    widened, _ = need.rewrite([hard('budget', 500), hard('color', 'red')])
    assert len(widened.hard) == 3


def test_new_fields_are_reported_as_added():
    _, split = t012_state().rewrite([hard('color', 'black')])
    assert [c.field_name for c in split.added] == ['color']


def test_entities_change_role_and_are_never_dropped():
    need = t012_state().remember(Entity(sku='SKU_B', role='candidate', resolved_from='这双'))
    after, _ = need.rewrite([hard('use_scenario', 'daily_wear_primary')])
    assert {e.sku for e in after.entities} == {'SKU_NIKE_ZOOMX_VF3', 'SKU_B'}
    recast = after.recast('SKU_B', 'target')
    assert recast.entity('SKU_B').role == 'target'
    assert recast.entity('SKU_NIKE_ZOOMX_VF3').role == 'reference'
    with pytest.raises(KeyError):
        recast.recast('SKU_MISSING', 'target')


def test_low_confidence_entity_is_flagged_for_disambiguation():
    assert Entity(sku='S', role='candidate', resolved_from='那个', confidence='low') \
        .needs_disambiguation
    assert not Entity(sku='S', role='target', resolved_from='这个').needs_disambiguation


def test_clarify_budget_switches_strategy_instead_of_asking_forever():
    need = UserNeed(intent='discover')
    need = need.clarified(limit=2)
    assert need.strategy == 'narrow_down' and need.clarify_rounds_used == 1
    need = need.clarified(limit=2)
    assert need.strategy == 'explore_breadth' and need.clarify_rounds_used == 2


def test_derived_constraints_must_carry_their_derivation():
    with pytest.raises(ValueError, match='derived_from'):
        Constraint(field_name='x', op='==', value=1, kind='derived')
    with pytest.raises(ValueError, match='evidence'):
        Constraint(field_name='x', op='==', value=1, kind='derived', derived_from=('y',))
    with pytest.raises(ValueError, match='derive from itself'):
        Constraint(field_name='x', op='==', value=1, kind='derived',
                   derived_from=('x',), evidence='e')


def test_one_constraint_per_field():
    with pytest.raises(ValueError, match='one constraint per field'):
        UserNeed(intent='discover', constraints=(hard('budget', 1), hard('budget', 2)))
    with pytest.raises(ValueError, match='repeat a field'):
        UserNeed(intent='discover').rewrite([hard('budget', 1), hard('budget', 2)])


def test_unknown_operator_is_rejected():
    with pytest.raises(ValueError, match='unknown op'):
        Constraint(field_name='x', op='~=', value=1, kind='hard')


def test_derived_constraints_still_filter():
    need = t012_state()
    assert {c.field_name for c in need.hard} == {
        'category', 'user_level', 'use_scenario', 'carbon_plate', 'cushion_level'}
    assert {c.field_name for c in need.soft} == {'price_cny'}


def test_action_vocabulary_is_closed():
    assert 'rewrite_state' in ACTIONS and len(ACTIONS) == 12
