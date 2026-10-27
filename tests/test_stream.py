import json

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app


class StreamingBackend:
    mode = "test"

    def __init__(self, pieces=("测试", "回复")):
        self.pieces = pieces
        self.contexts = []
        self.fail = False

    def reply(self, context):
        return "".join(self.pieces)

    def stream(self, context):
        self.contexts.append(context)
        for piece in self.pieces:
            if self.fail:
                raise RuntimeError("backend unavailable")
            yield piece


class PlainBackend:
    """没有 stream 方法，用来验证退化路径。"""

    mode = "plain"

    def reply(self, context):
        return "整段回复"


def payload(user="u1", session="s1", content="你好"):
    return {"user_id": user, "session_id": session, "content": content}


def read_events(client, body=None):
    """把 SSE 响应解析成 [(event, data), ...]，注释帧丢弃。"""
    with client.stream("POST", "/api/chat/stream", json=body or payload()) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        text = "".join(response.iter_text())
    events = []
    for block in text.split("\n\n"):
        lines = [line for line in block.splitlines() if line and not line.startswith(":")]
        if not lines:
            continue
        fields = dict(line.split(": ", 1) for line in lines)
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_deltas_rebuild_the_final_message(tmp_path):
    backend = StreamingBackend(("测", "试回", "复"))
    with TestClient(create_app(tmp_path / "chat.db", backend)) as client:
        events = read_events(client)
    names = [name for name, _ in events]
    assert names == ["conversation.started", "assistant.delta", "assistant.delta",
                     "assistant.delta", "assistant.text", "conversation.completed"]
    deltas = "".join(data["text"] for name, data in events if name == "assistant.delta")
    final = dict(events)["assistant.text"]
    assert deltas == final["message"]["content"] == "测试回复"
    assert final["message"]["role"] == "assistant" and final["mode"] == "test"


def test_stream_persists_exactly_one_turn(tmp_path):
    backend = StreamingBackend()
    with TestClient(create_app(tmp_path / "chat.db", backend)) as client:
        read_events(client)
        assert backend.contexts[-1].history == []
        read_events(client, payload(content="继续"))
        assert backend.contexts[-1].to_messages() == [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "测试回复"},
            {"role": "user", "content": "继续"},
        ]
        # 流式与非流式共用同一份历史。
        client.post("/api/chat/messages", json=payload(content="换非流式"))
        read_events(client, payload(content="再回流式"))
        assert len(backend.contexts[-1].history) == 6


def test_mid_stream_failure_reports_error_and_persists_nothing(tmp_path):
    backend = StreamingBackend()
    with TestClient(create_app(tmp_path / "chat.db", backend)) as client:
        backend.fail = True
        events = read_events(client)
        assert [name for name, _ in events] == ["conversation.started", "error"]
        assert dict(events)["error"] == {
            "code": "MODEL_ERROR", "message": "backend unavailable", "retryable": True,
        }
        backend.fail = False
        read_events(client)
        assert backend.contexts[-1].history == []


def test_backend_without_stream_falls_back_to_one_delta(tmp_path):
    with TestClient(create_app(tmp_path / "chat.db", PlainBackend())) as client:
        events = read_events(client)
    deltas = [data["text"] for name, data in events if name == "assistant.delta"]
    assert deltas == ["整段回复"]


def test_empty_generation_is_an_error_not_a_crash(tmp_path):
    with TestClient(create_app(tmp_path / "chat.db", StreamingBackend(("", " ")))) as client:
        events = read_events(client)
    assert [name for name, _ in events] == ["conversation.started", "assistant.delta", "error"]


@pytest.mark.parametrize("body", [
    {}, payload(user=" "), payload(content="\n "), {**payload(), "history": []},
])
def test_invalid_input_still_rejected_before_streaming(tmp_path, body):
    with TestClient(create_app(tmp_path / "chat.db")) as client:
        assert client.post("/api/chat/stream", json=body).status_code == 422
