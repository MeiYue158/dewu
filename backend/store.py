import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from backend.models import HistoryMessage


class HistoryStore:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_session
                ON messages(user_id, session_id, id);
                CREATE TABLE IF NOT EXISTS session_state (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    need TEXT NOT NULL,
                    PRIMARY KEY (user_id, session_id)
                );
            """)

    def load(self, user_id: str, session_id: str) -> list[HistoryMessage]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT role, content FROM messages WHERE user_id=? AND session_id=? ORDER BY id",
                (user_id, session_id),
            ).fetchall()
        return [HistoryMessage(role=role, content=content) for role, content in rows]

    def load_state(self, user_id: str, session_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                'SELECT need FROM session_state WHERE user_id=? AND session_id=?',
                (user_id, session_id)).fetchone()
        return json.loads(row[0]) if row else None

    def save_state(self, user_id: str, session_id: str, need: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                'INSERT INTO session_state(user_id, session_id, need) VALUES (?, ?, ?) '
                'ON CONFLICT(user_id, session_id) DO UPDATE SET need=excluded.need',
                (user_id, session_id, json.dumps(need, ensure_ascii=False)))

    def append_turn(self, user_id: str, session_id: str,
                    message: HistoryMessage, reply: HistoryMessage) -> None:
        # both rows in one transaction so we never end up with half a turn
        with self.connect() as connection:
            connection.executemany(
                "INSERT INTO messages(user_id, session_id, role, content) VALUES (?, ?, ?, ?)",
                [(user_id, session_id, item.role, item.content) for item in (message, reply)],
            )
