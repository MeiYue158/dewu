import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from backend import sse
from backend.models import ChatMessage, ChatResponse, ProductCard, RelaxOption, TurnResponse
from backend.service import ChatBackend, ChatService, GuideService, select_backend
from backend.store import HistoryStore


def build_guide(store: HistoryStore, orchestrator=None) -> GuideService | None:
    """the guide path needs both the catalog and the knowledge base.

    if either is missing we just do not mount the route. half-working is worse
    than absent - at least a missing route is obvious.
    TODO: this currently means a 404, which looks like a typo in the url.
    probably should be a 503 so the caller can tell the difference.
    """
    if orchestrator is not None:
        return GuideService(store, orchestrator)
    try:
        from backend.knowledge import load_default
        from backend.orchestrator import Orchestrator
        from backend.retrieval import Catalog, Retriever
        from backend.understanding import Understanding
        root = Path(__file__).resolve().parent.parent
        catalog = Catalog.load(root / 'fixtures/catalog/products.jsonl')
        knowledge = load_default(root)
    except (FileNotFoundError, KeyError, ValueError):
        return None
    return GuideService(store, Orchestrator(Understanding(knowledge), catalog,
                                            Retriever(catalog)))


def as_response(request: ChatMessage, turn) -> TurnResponse:
    return TurnResponse(
        user_id=request.user_id, session_id=request.session_id,
        action=turn.action, card=turn.card, text=turn.text,
        options=list(turn.options), asked_for=turn.asked_for, total=turn.total,
        items=[ProductCard(product_id=hit.product_id, title=hit.title, category=hit.category,
                           brand=hit.brand, price_cny=hit.price_cny,
                           sku_ids=list(hit.sku_ids), why=list(hit.matched))
               for hit in turn.hits],
        relax_options=[RelaxOption(label=option.label, kind=option.kind,
                                   fields=list(option.fields),
                                   result_preview=option.result_count)
                       for option in turn.relaxations],
        constraints=[{'field': c.field_name, 'op': c.op,
                      'value': list(c.value) if isinstance(c.value, tuple) else c.value,
                      'kind': c.kind, 'evidence': c.evidence,
                      'anchor_source': c.anchor_source}
                     for c in turn.need.constraints],
        trace=list(turn.trace), rejected=list(turn.rejected))


def create_app(db_path: Path | None = None, backend: ChatBackend | None = None,
               orchestrator=None) -> FastAPI:
    store = HistoryStore(db_path if db_path is not None else Path(os.getenv(
        "CHAT_DB_PATH", str(Path(__file__).resolve().parent.parent / "data" / "chat.sqlite3")
    )))
    service = ChatService(store, backend if backend is not None else select_backend())
    guide = build_guide(store, orchestrator)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.initialize()
        yield

    app = FastAPI(title="Chat API", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/chat/messages", response_model=ChatResponse)
    def chat(request: ChatMessage) -> ChatResponse:
        return service.chat(request)

    @app.post("/api/chat/stream")
    def chat_stream(request: ChatMessage) -> StreamingResponse:
        # pydantic has already validated the body before the generator starts,
        # so bad input still gets a normal 422 and never reaches the stream
        def events():
            yield sse.comment("open")
            for event, payload in service.chat_stream(request):
                yield sse.frame(event, payload)

        return StreamingResponse(events(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx buffers otherwise and the deltas
                                       # all arrive at once, which defeats the point
        })

    if guide is not None:
        @app.post("/api/guide/turn", response_model=TurnResponse)
        def guide_turn(request: ChatMessage) -> TurnResponse:
            return as_response(request, guide.turn(request))

    return app


app = create_app()
