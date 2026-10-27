import pytest
from fastapi.testclient import TestClient

from backend.main import create_app


class RecordingBackend:
    mode = "test"

    def __init__(self):
        self.contexts = []
        self.fail = False

    def reply(self, context):
        self.contexts.append(context)
        if self.fail:
            raise RuntimeError("backend unavailable")
        return "测试回复"


def payload(user="u1", session="s1", content="你好"):
    return {"user_id": user, "session_id": session, "content": content}


def test_history_is_ordered_isolated_and_persistent(tmp_path):
    db = tmp_path / "chat.db"
    backend = RecordingBackend()
    with TestClient(create_app(db, backend)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.post("/api/chat/messages", json=payload()).status_code == 200
        assert backend.contexts[-1].history == []
        client.post("/api/chat/messages", json=payload(content="继续"))
        assert backend.contexts[-1].to_messages() == [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "测试回复"},
            {"role": "user", "content": "继续"},
        ]
        client.post("/api/chat/messages", json=payload(user="u2"))
        assert backend.contexts[-1].history == []
        client.post("/api/chat/messages", json=payload(session="s2"))
        assert backend.contexts[-1].history == []
    with TestClient(create_app(db, backend)) as client:
        client.post("/api/chat/messages", json=payload(content="重启后"))
        assert len(backend.contexts[-1].history) == 4


@pytest.mark.parametrize("body", [
    {}, payload(user=" "), payload(session=""), payload(content="\n "),
    {**payload(), "history": []}, {**payload(), "user_id": 123},
])
def test_invalid_input(tmp_path, body):
    with TestClient(create_app(tmp_path / "chat.db")) as client:
        assert client.post("/api/chat/messages", json=body).status_code == 422


def test_failure_does_not_persist_partial_turn(tmp_path):
    backend = RecordingBackend()
    with TestClient(create_app(tmp_path / "chat.db", backend),
                    raise_server_exceptions=False) as client:
        backend.fail = True
        assert client.post("/api/chat/messages", json=payload()).status_code == 500
        backend.fail = False
        assert client.post("/api/chat/messages", json=payload()).status_code == 200
        assert backend.contexts[-1].history == []


def test_stub_is_explicit(tmp_path):
    with TestClient(create_app(tmp_path / "chat.db")) as client:
        response = client.post("/api/chat/messages", json=payload()).json()
        assert response["mode"] == "local_stub"
        assert response["message"]["role"] == "assistant"
