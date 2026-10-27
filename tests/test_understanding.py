import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import intents
from backend.knowledge import load_default
from backend.need import Constraint, Entity, UserNeed
from backend.understanding import Understanding, parse

ROOT = Path(__file__).resolve().parents[1]
KB = load_default(ROOT)


class FakeClient:
    """按脚本返回模型回复；不发网络请求。"""

    def __init__(self, reply):
        self.reply = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=self.reply))])


def reader(reply):
    client = FakeClient(reply)
    return Understanding(KB, model='test', api_key='k', client_factory=lambda: client), client


# ---- 解析 ----

def test_parse_tolerates_fences_and_prose():
    assert parse('```json\n{"intent": "discover"}\n```') == ({'intent': 'discover'}, None)
    assert parse('好的，结果是 {"intent": "policy"} 。')[0] == {'intent': 'policy'}
    assert parse('没有 JSON') == ({}, 'no JSON object in reply')
    assert parse('{"intent": }')[1].startswith('malformed JSON')
    assert parse('[1, 2]')[1] == 'no JSON object in reply'


# ---- unknown 出口 ----

def test_out_of_scope_utterance_never_enters_the_business_flow():
    reader_, _ = reader({'intent': 'unknown', 'slots': []})
    result = reader_.interpret('今天天气怎么样')
    assert result.unknown and not result.ready
    assert result.need.constraints == ()


def test_unreadable_reply_falls_back_to_unknown_not_a_guess():
    for bad in ('抱歉我不知道', {'intent': 'shopping'}, '{"intent": '):
        reader_, _ = reader(bad)
        result = reader_.interpret('想买点什么')
        assert result.unknown and result.error


def test_slots_are_ignored_when_intent_is_unknown():
    reader_, _ = reader({'intent': 'unknown',
                         'slots': [{'field': 'category', 'op': '==', 'value': '项链'}]})
    assert reader_.interpret('随便聊聊').need.constraints == ()


def test_an_out_of_scope_turn_keeps_the_session_intent():
    before = UserNeed(intent='compare', constraints=(
        KB.schema.constrain('category', '==', '项链'),))
    reader_, _ = reader({'intent': 'unknown', 'slots': []})
    result = reader_.interpret('今天几号', before)
    # 调用方看到的是 unknown（用于分支），但存下来的会话意图不变。
    assert result.unknown and result.need.intent == 'compare'
    assert result.need.of('category').value == '项链'


# ---- 槽位校验 ----

def test_invalid_slots_are_dropped_with_a_reason():
    reader_, _ = reader({'intent': 'discover', 'slots': [
        {'field': 'category', 'op': '==', 'value': '项链'},
        {'field': 'cushion_level', 'op': '>=', 'value': 7},      # 目录没有这个字段
        {'field': 'category', 'op': '==', 'value': '智能马桶'},   # 不在取值域
        {'field': 'price_cny', 'op': '<=', 'value': '三千'},      # 类型不对
    ]})
    result = reader_.interpret('想买条项链')
    assert result.need.of('category').value == '项链'
    assert {r['field'] for r in result.rejected} == {'cushion_level', 'category', 'price_cny'}
    assert any('unknown field' in r['reason'] for r in result.rejected)
    # 校验不过的字段算没填，所以预算仍在缺失清单里。
    assert 'price_cny' in result.missing_core


def test_malformed_slot_entries_do_not_crash():
    reader_, _ = reader({'intent': 'discover', 'slots': ['项链', None, 42]})
    result = reader_.interpret('想买条项链')
    assert len(result.rejected) == 3 and result.need.of('category') is None


def test_slots_not_a_list_is_reported():
    reader_, _ = reader({'intent': 'discover', 'slots': {'category': '项链'}})
    assert reader_.interpret('项链').rejected[0]['reason'].startswith('slots must be a list')


# ---- 缺失由代码推导 ----

def test_missing_core_is_derived_not_reported_by_the_model():
    reader_, _ = reader({'intent': 'discover', 'slots': [
        {'field': 'category', 'op': '==', 'value': '项链'}]})
    result = reader_.interpret('想买条项链')
    assert result.missing_core == ('price_cny',) and not result.ready
    full, _ = reader({'intent': 'discover', 'slots': [
        {'field': 'category', 'op': '==', 'value': '项链'},
        {'field': 'price_cny', 'op': '<=', 'value': 3000}]})
    done = full.interpret('想买条项链，预算3000')
    assert done.missing_core == () and done.ready


def test_entity_shortfall_is_counted_as_missing():
    reader_, _ = reader({'intent': 'compare', 'slots': []})
    result = reader_.interpret('这两个哪个好')
    assert result.missing_core == ('entity#1',)
    with_one = UserNeed(intent='compare', entities=(
        Entity(sku='S1', role='candidate', resolved_from='这个'),))
    assert reader_.interpret('这两个哪个好', with_one).missing_core == ('entity#2',)


# ---- 知识库与模型协同 ----

def test_knowledge_base_fires_even_when_the_model_proposes_nothing():
    reader_, _ = reader({'intent': 'discover', 'slots': []})
    result = reader_.interpret('想买个东西送女朋友')
    category = result.need.of('category')
    assert category.kind == 'derived' and category.derived_from == ('occasion',)
    assert result.need.of('occasion').value == 'gift'
    assert result.trace[0]['rule'] == 'SR-G01'


def test_model_slots_override_knowledge_base_defaults():
    reader_, _ = reader({'intent': 'discover', 'slots': [
        {'field': 'category', 'op': '==', 'value': '手链'}]})
    result = reader_.interpret('送女朋友，要手链')
    # 用户说死了类目，就不该再用知识库推出的宽泛范围。
    assert result.need.of('category').value == '手链'


def test_relative_phrase_uses_the_existing_budget_as_anchor():
    reader_, _ = reader({'intent': 'discover', 'slots': []})
    before = UserNeed(intent='discover', constraints=(
        KB.schema.constrain('price_cny', '<=', 3000),))
    result = reader_.interpret('有没有便宜点的', before)
    assert result.need.of('price_cny').value == 2100.0
    assert result.need.of('price_cny').anchor_source == '便宜点 @ 3000'


def test_relative_phrase_without_an_anchor_stays_unfilled():
    reader_, _ = reader({'intent': 'discover', 'slots': []})
    result = reader_.interpret('有没有便宜点的')
    assert result.need.of('price_cny') is None
    assert result.trace[-1]['needs_anchor'] is True
    assert 'price_cny' in result.missing_core


def test_conflict_blocks_readiness():
    reader_, _ = reader({'intent': 'discover', 'slots': [
        {'field': 'category', 'op': '==', 'value': '耳钉'},
        {'field': 'price_cny', 'op': '<=', 'value': 2000}]})
    result = reader_.interpret('预算2000的耳钉')
    assert result.missing_core == () and not result.ready
    assert result.conflicts[0].field == 'category' and result.conflicts[0].floor == 6750


# ---- 意图切换 ----

def test_switching_scene_invalidates_the_old_derived_constraints():
    gift, _ = reader({'intent': 'discover', 'slots': []})
    first = gift.interpret('送女朋友的生日礼物')
    assert first.need.of('category').kind == 'derived'
    silver, _ = reader({'intent': 'discover', 'slots': [
        {'field': '材质', 'op': 'CONTAINS', 'value': '银'}]})
    second = silver.interpret('算了我自己戴，要银的', first.need)
    # occasion 没被新话语改写，所以类目范围仍在；材质是新增的。
    assert second.need.of('材质').value == '银'
    assert second.need.of('occasion').value == 'gift'


# ---- 提示词只暴露目录答得了的字段 ----

def test_prompt_lists_only_well_covered_fields():
    reader_, client = reader({'intent': 'unknown', 'slots': []})
    reader_.interpret('你好')
    system = client.calls[0]['messages'][0]['content']
    assert 'category' in system and 'price_cny' in system
    for absent in ('cushion_level', '镶嵌材质', '主动降噪'):
        assert absent not in system
    assert all(i in system for i in intents.INTENTS)
