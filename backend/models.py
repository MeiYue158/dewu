from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Content = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32000)]
# ids stay strings - the sqlite column is TEXT and ChatResponse says str too.
# after stripping we need at least 1 char, so an all-whitespace user_id gets
# rejected with 422 instead of landing in the db
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]

class ChatMessage(BaseModel):
    """client only sends the current message. history is loaded server side."""

    model_config = ConfigDict(extra="forbid")
    user_id: Identifier
    session_id: Identifier
    content: Content


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Content


class ChatContext(BaseModel):
    user_id: Identifier
    session_id: Identifier
    message: HistoryMessage
    history: list[HistoryMessage] = Field(default_factory=list)

    def to_messages(self) -> list[dict[str, str]]:
        """build model input in order. current message appended exactly once."""
        return [item.model_dump() for item in [*self.history, self.message]]


class ProductCard(BaseModel):
    product_id: str
    title: str
    category: str
    brand: str
    price_cny: float | None
    sku_ids: list[str]
    why: list[str]              # which soft conditions matched = why we show it


class RelaxOption(BaseModel):
    label: str
    kind: Literal["widen", "drop"]
    fields: list[str]
    result_preview: int         # !!! must equal what you actually get after
                                # clicking. a preview that lies is worse than
                                # no preview at all


class TurnResponse(BaseModel):
    """one guide turn.

    action and card values come from the registry. this layer is not allowed
    to invent new ones - there is a test that checks it.
    """

    user_id: str
    session_id: str
    action: str
    card: str
    text: str
    options: list[str] = Field(default_factory=list)
    asked_for: str | None = None
    items: list[ProductCard] = Field(default_factory=list)
    total: int = 0
    relax_options: list[RelaxOption] = Field(default_factory=list)
    constraints: list[dict] = Field(default_factory=list)
    trace: list[dict] = Field(default_factory=list)
    rejected: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    user_id: str
    session_id: str
    message: HistoryMessage
    mode: str
