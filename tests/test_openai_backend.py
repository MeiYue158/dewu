import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.models import ChatContext, HistoryMessage
from backend.service import (SYSTEM_PROMPT, ChatService, LocalBackend, OpenAIBackend,
                             select_backend)
from backend.store import HistoryStore


class FakeClient:
    """替代 openai.OpenAI：记录请求，按脚本返回，不发网络请求。"""

    def __init__(self, text=None, pieces=None, error=None):
        # 给了分片就让整段回复等于分片拼接，两条路径的脚本保持一致。
        self.pieces, self.error = pieces, error
        self.text = text if text is not None else ''.join(p for p in (pieces or ['真实回复']) if p)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        if kwargs.get('stream'):
            return iter([self._chunk(p) for p in (self.pieces or [self.text])])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))])

    @staticmethod
    def _chunk(piece):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])


def backend_with(client, **kwargs):
    return OpenAIBackend(model='test-model', api_key='k', client_factory=lambda: client, **kwargs)


def context(content='你好', history=()):
    return ChatContext(user_id='u1', session_id='s1',
                       message=HistoryMessage(role='user', content=content),
                       history=[HistoryMessage(**h) for h in history])


def test_reply_sends_system_prompt_then_history():
    client = FakeClient('这是回复')
    backend = backend_with(client)
    text = backend.reply(context('继续', [{'role': 'user', 'content': '你好'},
                                         {'role': 'assistant', 'content': '在'}]))
    assert text == '这是回复'
    assert client.calls[0]['model'] == 'test-model'
    assert client.calls[0]['messages'] == [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': '你好'},
        {'role': 'assistant', 'content': '在'},
        {'role': 'user', 'content': '继续'},
    ]
    assert client.closed


def test_stream_yields_only_nonempty_deltas():
    client = FakeClient(pieces=['真实', '', None, '回复'])
    assert list(backend_with(client).stream(context())) == ['真实', '回复']
    assert client.calls[0]['stream'] is True


def test_empty_reply_is_an_error_not_an_empty_message():
    with pytest.raises(ValueError):
        backend_with(FakeClient('   ')).reply(context())


def test_mode_names_the_model():
    assert backend_with(FakeClient()).mode == 'openai:test-model'


def test_missing_key_fails_loudly(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(ValueError, match='OPENAI_API_KEY'):
        OpenAIBackend()


def test_switch_defaults_to_local(monkeypatch):
    monkeypatch.delenv('CHAT_BACKEND', raising=False)
    assert isinstance(select_backend(), LocalBackend)
    monkeypatch.setenv('CHAT_BACKEND', 'local')
    assert isinstance(select_backend(), LocalBackend)


def test_switch_opts_in_to_openai(monkeypatch):
    monkeypatch.setenv('CHAT_BACKEND', 'OpenAI')
    monkeypatch.setenv('OPENAI_API_KEY', 'k')
    monkeypatch.setenv('OPENAI_MODEL', 'gpt-test')
    assert select_backend().mode == 'openai:gpt-test'


def test_unknown_switch_value_is_rejected():
    with pytest.raises(ValueError, match='Unknown CHAT_BACKEND'):
        select_backend('anthropic')


def test_history_is_truncated_to_max_turns(tmp_path):
    store = HistoryStore(tmp_path / 'chat.db')
    store.initialize()
    for i in range(5):
        store.append_turn('u1', 's1', HistoryMessage(role='user', content=f'问{i}'),
                          HistoryMessage(role='assistant', content=f'答{i}'))
    client = FakeClient('ok')
    backend = backend_with(client)
    ChatService(store, backend, max_history_turns=2).chat(
        type('R', (), {'user_id': 'u1', 'session_id': 's1', 'content': '新问题'})())
    sent = [m['content'] for m in client.calls[0]['messages'] if m['role'] != 'system']
    # 只送最近 2 轮（4 条）加当前消息；库里仍是 10 条。
    assert sent == ['问3', '答3', '问4', '答4', '新问题']
    assert len(store.load('u1', 's1')) == 12


def test_endpoints_return_real_model_output(tmp_path):
    client = FakeClient(pieces=['真实', '模型', '回复'])
    app = create_app(tmp_path / 'chat.db', backend_with(client))
    with TestClient(app) as http:
        body = {'user_id': 'u1', 'session_id': 's1', 'content': '你好'}
        payload = http.post('/api/chat/messages', json=body).json()
        assert payload['message']['content'] == '真实模型回复'
        assert payload['mode'] == 'openai:test-model'
        with http.stream('POST', '/api/chat/stream', json=body) as response:
            text = ''.join(response.iter_text())
    deltas = [json.loads(line[6:])['text'] for line in text.splitlines()
              if line.startswith('data: ') and '"text"' in line]
    assert deltas == ['真实', '模型', '回复']


def test_model_failure_does_not_persist_a_partial_turn(tmp_path):
    client = FakeClient(error=RuntimeError('upstream 503'))
    app = create_app(tmp_path / 'chat.db', backend_with(client))
    with TestClient(app, raise_server_exceptions=False) as http:
        body = {'user_id': 'u1', 'session_id': 's1', 'content': '你好'}
        assert http.post('/api/chat/messages', json=body).status_code == 500
        with http.stream('POST', '/api/chat/stream', json=body) as response:
            assert response.status_code == 200          # 响应头已发出，错误只能走事件
            assert 'MODEL_ERROR' in ''.join(response.iter_text())
    client.error = None
    with TestClient(app) as http:
        http.post('/api/chat/messages', json={'user_id': 'u1', 'session_id': 's1', 'content': '再来'})
        # 前两次失败没有落库，所以这一轮看到的历史仍是空的。
        assert client.calls[-1]['messages'][1:] == [{'role': 'user', 'content': '再来'}]
