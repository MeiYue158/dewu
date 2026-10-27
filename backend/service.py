import os
from collections.abc import Iterator
from threading import Lock
from typing import Protocol

from backend.models import ChatContext, ChatMessage, ChatResponse, HistoryMessage
from backend.store import HistoryStore

# the catalog is not wired into the chat path yet, so the model must not
# invent product facts. matches what the PRD says - no guessing stock,
# material, discounts, after-sales or compatibility.
SYSTEM_PROMPT = (
    '你是一个商品导购助手，用简洁的中文回答。'
    '当前版本尚未接入商品目录，你没有任何商品、价格、库存、规格或兼容性数据。'
    '遇到需要具体商品事实的问题，直接说明暂时查不到，并请用户描述预算、用途和偏好，'
    '不要凭记忆给出商品名、型号或价格。'
)


class ChatBackend(Protocol):
    mode: str

    def reply(self, context: ChatContext) -> str: ...


class StreamingChatBackend(ChatBackend, Protocol):
    """optional capability.

    if a backend implements stream() we send deltas, otherwise it falls back
    to one chunk. we do not fake the chunking - see _deltas below.
    """

    def stream(self, context: ChatContext) -> Iterator[str]: ...


class LocalBackend:
    """stub backend. only there to excercise the api + history plumbing."""

    mode = "local_stub"

    def reply(self, context: ChatContext) -> str:
        return f"[本地占位回复] 已加载 {len(context.history)} 条历史消息；当前消息：{context.message.content}"

    def stream(self, context: ChatContext) -> Iterator[str]:
        # fixed-size slices, just so the frontend has something to glue together
        text = self.reply(context)
        for start in range(0, len(text), 12):
            yield text[start:start + 12]


class OpenAIBackend:
    """real chat model over an OpenAI-compatible endpoint."""

    def __init__(self, model=None, base_url=None, api_key=None, client_factory=None):
        self.model = model or os.getenv('OPENAI_MODEL') or 'gpt-4.1-mini'
        self.base_url = base_url or os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1'
        self._api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self._api_key:
            raise ValueError('OPENAI_API_KEY is required for the openai chat backend')
        self._client_factory = client_factory
        self.mode = f'openai:{self.model}'

    def _client(self):
        if self._client_factory is not None:
            return self._client_factory()
        from openai import OpenAI
        # key only ever comes from env. never log it, never echo it back
        return OpenAI(base_url=self.base_url, api_key=self._api_key, timeout=60, max_retries=1)

    @staticmethod
    def _messages(context: ChatContext) -> list[dict[str, str]]:
        return [{'role': 'system', 'content': SYSTEM_PROMPT}, *context.to_messages()]

    def reply(self, context: ChatContext) -> str:
        with self._client() as client:
            completion = client.chat.completions.create(
                model=self.model, messages=self._messages(context))
        text = (completion.choices[0].message.content or '').strip()
        if not text:
            raise ValueError('Model returned an empty reply')
        return text

    def stream(self, context: ChatContext) -> Iterator[str]:
        with self._client() as client:
            for chunk in client.chat.completions.create(
                    model=self.model, messages=self._messages(context), stream=True):
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content
                if piece:
                    yield piece


def select_backend(name=None) -> ChatBackend:
    """stub by default.

    you have to set CHAT_BACKEND=openai on purpose to spend money. an unknown
    value raises instead of quietly falling back - a silent fallback here
    would be really annoying to debug.
    """
    choice = (name or os.getenv('CHAT_BACKEND') or 'local').strip().lower()
    if choice == 'local':
        return LocalBackend()
    if choice == 'openai':
        return OpenAIBackend()
    raise ValueError(f'Unknown CHAT_BACKEND: {choice!r} (expected "local" or "openai")')


class GuideService:
    """one guide turn: load session state, run the orchestrator, save it back."""

    def __init__(self, store: HistoryStore, orchestrator):
        self.store = store
        self.orchestrator = orchestrator
        self._lock = Lock()

    def turn(self, request: ChatMessage):
        from backend.need import UserNeed
        with self._lock:
            saved = self.store.load_state(request.user_id, request.session_id)
            before = UserNeed.from_dict(saved) if saved else None
            result = self.orchestrator.respond(request.content, before)
            self.store.save_state(request.user_id, request.session_id, result.need.as_dict())
            # also log the text so a page refresh still shows the conversation
            self.store.append_turn(
                request.user_id, request.session_id,
                HistoryMessage(role='user', content=request.content),
                HistoryMessage(role='assistant', content=result.text or '（无文本）'))
            return result


class ChatService:
    def __init__(self, store: HistoryStore, backend: ChatBackend, max_history_turns=None):
        self.store = store
        self.backend = backend
        # one turn = user message + assistant reply.
        # only the context we send to the model gets trimmed, db keeps all of it
        self.max_history_turns = int(os.getenv('MAX_HISTORY_TURNS', '6')) \
            if max_history_turns is None else int(max_history_turns)
        if self.max_history_turns < 0:
            raise ValueError('max_history_turns must not be negative')
        # single process for now, serialise so nobody reads stale history
        self._lock = Lock()

    def chat(self, request: ChatMessage) -> ChatResponse:
        with self._lock:
            context = self._context(request)
            reply = HistoryMessage(role="assistant", content=self.backend.reply(context))
            self.store.append_turn(request.user_id, request.session_id, context.message, reply)
            return self._response(request, reply)

    def chat_stream(self, request: ChatMessage) -> Iterator[tuple[str, dict]]:
        """yields (event name, payload) and the route turns it into SSE.

        same write-to-db timing as chat(): the turn only lands once the whole
        reply generated. if it blows up half way we emit an error event and
        write nothing - half a turn in the history is worse than no turn.
        """
        with self._lock:
            context = self._context(request)
            yield "conversation.started", {
                "user_id": request.user_id, "session_id": request.session_id,
                "mode": self.backend.mode,
            }
            pieces: list[str] = []
            try:
                for delta in self._deltas(context):
                    if not delta:
                        continue
                    pieces.append(delta)
                    yield "assistant.delta", {"text": delta}
            except Exception as error:
                # headers already went out, so we cannot change the status code
                # any more. the error has to travel inside the event body.
                yield "error", self._error(str(error))
                return
            if not "".join(pieces).strip():
                yield "error", self._error("backend produced no content")
                return
            reply = HistoryMessage(role="assistant", content="".join(pieces))
            self.store.append_turn(request.user_id, request.session_id, context.message, reply)
            yield "assistant.text", self._response(request, reply).model_dump()
            yield "conversation.completed", {
                "user_id": request.user_id, "session_id": request.session_id,
            }

    def _context(self, request: ChatMessage) -> ChatContext:
        history = self.store.load(request.user_id, request.session_id)
        keep = 2 * self.max_history_turns
        return ChatContext(
            user_id=request.user_id,
            session_id=request.session_id,
            message=HistoryMessage(role="user", content=request.content),
            history=history[-keep:] if keep else [],
        )

    def _response(self, request: ChatMessage, reply: HistoryMessage) -> ChatResponse:
        return ChatResponse(user_id=request.user_id, session_id=request.session_id,
                            message=reply, mode=self.backend.mode)

    def _deltas(self, context: ChatContext) -> Iterator[str]:
        stream = getattr(self.backend, "stream", None)
        if stream is None:
            # backend has no stream() - send it as one delta, do not fake chunks
            yield self.backend.reply(context)
            return
        yield from stream(context)

    @staticmethod
    def _error(message: str) -> dict:
        return {"code": "MODEL_ERROR", "message": message, "retryable": True}
