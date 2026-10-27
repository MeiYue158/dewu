import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.knowledge import load_default
from backend.main import create_app
from backend.need import UserNeed
from backend.orchestrator import Orchestrator
from backend.retrieval import Catalog, Retriever
from backend.store import HistoryStore
from backend.understanding import Understanding

ROOT = Path(__file__).resolve().parents[1]
KB = load_default(ROOT)
CATALOG = Catalog.load(ROOT / 'fixtures/catalog/products.jsonl')


class ScriptedClient:
    """按轮次返回预设的抽取结果；不发网络请求。"""

    def __init__(self, replies):
        self.replies = replies
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(reply, ensure_ascii=False)))])


def app_with(replies, db, clarify_limit=3):
    client = ScriptedClient(replies)
    reader = Understanding(KB, model='t', api_key='k', client_factory=lambda: client)
    orchestrator = Orchestrator(reader, CATALOG, Retriever(CATALOG),
                                clarify_limit=clarify_limit)
    return create_app(db, orchestrator=orchestrator)


def slots(*pairs, intent='discover'):
    return {'intent': intent,
            'slots': [{'field': f, 'op': o, 'value': v} for f, o, v in pairs]}


def body(content, user='u1', session='s1'):
    return {'user_id': user, 'session_id': session, 'content': content}


def test_a_turn_returns_a_card_not_free_text(tmp_path):
    app = app_with([slots(('category', '==', '项链'), ('price_cny', '<=', 20000))],
                   tmp_path / 'c.db')
    with TestClient(app) as http:
        payload = http.post('/api/guide/turn', json=body('项链两万以内')).json()
    assert payload['action'] == 'search' and payload['card'] == 'item_list'
    assert payload['total'] > 0 and payload['items']
    first = payload['items'][0]
    assert first['sku_ids'] and first['category'] == '项链'
    assert first['price_cny'] <= 20000


def test_state_survives_across_requests(tmp_path):
    db = tmp_path / 'c.db'
    app = app_with([slots(('category', '==', '项链')),
                    slots(('price_cny', '<=', 20000))], db)
    with TestClient(app) as http:
        first = http.post('/api/guide/turn', json=body('想买条项链')).json()
        assert first['action'] == 'ask_clarify' and first['asked_for'] == 'price_cny'
        second = http.post('/api/guide/turn', json=body('两万以内')).json()
    # 第二轮没有再说类目，但它必须还在——状态是从库里取回来的。
    fields = {c['field']: c['value'] for c in second['constraints']}
    assert fields['category'] == '项链' and fields['price_cny'] == 20000
    assert second['action'] == 'search'


def test_state_is_restored_after_a_restart(tmp_path):
    db = tmp_path / 'c.db'
    with TestClient(app_with([slots(('category', '==', '项链'))], db)) as http:
        http.post('/api/guide/turn', json=body('想买条项链'))
    with TestClient(app_with([slots(('price_cny', '<=', 20000))], db)) as http:
        after = http.post('/api/guide/turn', json=body('两万以内')).json()
    assert {c['field'] for c in after['constraints']} >= {'category', 'price_cny'}


def test_sessions_do_not_share_state(tmp_path):
    db = tmp_path / 'c.db'
    app = app_with([slots(('category', '==', '项链'))], db)
    with TestClient(app) as http:
        http.post('/api/guide/turn', json=body('想买条项链', session='s1'))
        other = http.post('/api/guide/turn', json=body('想买条项链', session='s2')).json()
    assert other['action'] == 'ask_clarify'
    with TestClient(app) as http:
        state = HistoryStore(db).load_state('u1', 's2')
    assert state['intent'] == 'discover'


def test_dead_end_exposes_previews_that_match_reality(tmp_path):
    app = app_with([slots(('category', '==', '耳钉'), ('price_cny', '<=', 2000))],
                   tmp_path / 'c.db')
    with TestClient(app) as http:
        payload = http.post('/api/guide/turn', json=body('预算2000的耳钉')).json()
    assert payload['card'] == 'no_match' and payload['action'] == 'flag_conflict'
    assert '6750' in payload['text']
    assert payload['relax_options']
    first = payload['relax_options'][0]
    assert first['kind'] == 'widen' and first['result_preview'] > 0


def test_derivations_are_visible_in_the_response(tmp_path):
    app = app_with([slots(('price_cny', '<=', 20000))], tmp_path / 'c.db')
    with TestClient(app) as http:
        payload = http.post('/api/guide/turn', json=body('送女朋友的礼物，两万以内')).json()
    derived = [c for c in payload['constraints'] if c['kind'] == 'derived']
    # 推导出来的约束必须带依据，否则用户无从判断系统凭什么收窄了范围。
    assert derived and all(c['evidence'] for c in derived)
    assert any(entry.get('rule') == 'SR-G01' for entry in payload['trace'])


def test_out_of_scope_keeps_the_session(tmp_path):
    db = tmp_path / 'c.db'
    app = app_with([slots(('category', '==', '项链'), ('price_cny', '<=', 20000)),
                    {'intent': 'unknown', 'slots': []}], db)
    with TestClient(app) as http:
        http.post('/api/guide/turn', json=body('项链两万以内'))
        payload = http.post('/api/guide/turn', json=body('今天几号')).json()
    assert payload['action'] == 'handoff' and payload['items'] == []
    assert {c['field'] for c in payload['constraints']} >= {'category', 'price_cny'}
    stored = HistoryStore(db).load_state('u1', 's1')
    assert stored['intent'] == 'discover'       # 越界不把会话意图改成 unknown


def test_invalid_input_is_still_rejected_before_the_pipeline(tmp_path):
    app = app_with([slots(('category', '==', '项链'))], tmp_path / 'c.db')
    with TestClient(app) as http:
        for bad in ({}, body(' '), {**body('x'), 'extra': 1}):
            assert http.post('/api/guide/turn', json=bad).status_code == 422


def test_the_plain_chat_endpoints_are_untouched(tmp_path):
    app = app_with([slots(('category', '==', '项链'))], tmp_path / 'c.db')
    with TestClient(app) as http:
        payload = http.post('/api/chat/messages', json=body('你好')).json()
    # 导购端点是新增的，原有的问答接口行为不变。
    assert payload['mode'] == 'local_stub'
    assert payload['message']['content'].startswith('[本地占位回复]')


def test_serialisation_round_trips_every_constraint_shape():
    need = UserNeed(intent='discover', constraints=(
        KB.schema.constrain('category', 'IN', ['项链', '手链']),
        KB.schema.constrain('price_cny', 'BETWEEN', [1000, 5000]),
        KB.schema.constrain('material_keys', 'ALL', ['银', '宝石/钻石']),
    ))
    restored = UserNeed.from_dict(json.loads(json.dumps(need.as_dict(), ensure_ascii=False)))
    # 元组必须还原成元组，否则 frozen dataclass 的相等性会悄悄失效。
    assert restored == need
    assert isinstance(restored.of('category').value, tuple)
