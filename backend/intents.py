"""Intent registry - the only place intents are defined."""
from dataclasses import dataclass

from backend.need import ACTIONS


@dataclass(frozen=True)
class Intent:
    id: str
    label: str
    # required slots - if any is missing we ask.
    # optional ones only move the ranking around, never block
    required_slots: tuple[str, ...]
    optional_slots: tuple[str, ...]
    required_entities: int
    first_action: str
    card: str

    def __post_init__(self):
        if self.first_action not in ACTIONS:
            raise ValueError(f'{self.id}: unknown action {self.first_action!r}')


INTENTS = {
    intent.id: intent for intent in (
        Intent(id='discover', label='模糊选品',
               required_slots=('category', 'price_cny'),
               optional_slots=('材质', 'color_keys', 'brand_key', '尺码', '礼盒'),
               required_entities=0, first_action='search', card='item_list'),
        Intent(id='spec_query', label='规格查询',
               required_slots=(), optional_slots=(),
               required_entities=1, first_action='fetch_spec', card='spec_explain'),
        Intent(id='compatibility', label='兼容判断',
               required_slots=(), optional_slots=(),
               required_entities=2, first_action='verify', card='spec_verify'),
        Intent(id='compare', label='多商品比较',
               required_slots=(), optional_slots=(),
               required_entities=2, first_action='compare', card='comparison_table'),
        Intent(id='policy', label='服务政策',
               required_slots=(), optional_slots=(),
               required_entities=0, first_action='fetch_policy', card='policy'),
        Intent(id='unknown', label='越界或无关',
               required_slots=(), optional_slots=(),
               required_entities=0, first_action='handoff', card='partial_answer'),
    )
}

BUSINESS_INTENTS = tuple(i for i in INTENTS if i != 'unknown')


def get(intent_id: str) -> Intent:
    try:
        return INTENTS[intent_id]
    except KeyError:
        raise ValueError(f'unknown intent: {intent_id!r}') from None


def missing_core(intent_id: str, filled: set[str], entity_count: int) -> tuple[str, ...]:
    """work out what is missing in code, do NOT ask the model for it.

    if the model reports both "known slots" and "missing slots" it can
    contradict itself (says budget is 3000 and also says budget is missing).
    missing = required - filled, that is all it ever is.
    """
    intent = get(intent_id)
    missing = [slot for slot in intent.required_slots if slot not in filled]
    if entity_count < intent.required_entities:
        missing.append(f'entity#{entity_count + 1}')
    return tuple(missing)
