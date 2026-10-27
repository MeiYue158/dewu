"""One turn: understand -> check completeness -> search -> card."""
from dataclasses import dataclass, field
from typing import Any

from backend.catalog_schema import CatalogSchema
from backend.grounding import Grounding, check
from backend.intents import get as get_intent
from backend.knowledge import Conflict, KnowledgeBase
from backend.need import UserNeed
from backend.retrieval import Catalog, Hit, Relaxation, Retriever, plan
from backend.understanding import Interpretation, Understanding

# how many times we are allowed to ask before giving up and just showing
# something. the exact number is still an open product question so it stays a
# parameter - i did not want to bury a guess in the code.
CLARIFY_LIMIT = 3

QUESTIONS = {
    'category': '想看哪一类？',
    'price_cny': '预算大概多少？',
    'material_keys': '材质上有偏好吗？',
    'color_keys': '想要什么颜色？',
}


def _values(product: dict, field_name: str, level: str) -> list[str]:
    """all values of a field on one product.

    !!! string fields have to be wrapped in a list. iterating a string gives
    you characters - that bug shipped quick options like ('chain','neck')
    instead of ('necklace','ring') and it looked almost plausible.
    """
    if level == 'product':
        raw = product.get(field_name)
    elif level == 'attribute':
        raw = product['attributes'].get(field_name)
    elif level == 'sku_property':
        return [p['value'] for sku in product['skus'] for p in sku['properties']
                if p.get('name') == field_name and p.get('value')]
    else:
        return []
    if not raw:
        return []
    return [str(v) for v in (raw if isinstance(raw, list) else [raw]) if str(v).strip()]


@dataclass(frozen=True)
class Turn:
    action: str
    card: str
    need: UserNeed
    text: str = ''
    options: tuple[str, ...] = ()
    hits: tuple[Hit, ...] = ()
    total: int = 0
    relaxations: tuple[Relaxation, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    asked_for: str | None = None
    trace: tuple[dict, ...] = ()
    rejected: tuple[dict, ...] = ()
    grounding: Grounding | None = None      # 出口前的接地校验结果

    @property
    def needs_user(self) -> bool:
        return self.action in ('ask_clarify', 'fallback', 'flag_conflict')


class Orchestrator:
    def __init__(self, understanding: Understanding, catalog: Catalog,
                 retriever: Retriever | None = None, clarify_limit: int = CLARIFY_LIMIT,
                 writer=None):
        self.understanding = understanding
        self.catalog = catalog
        self.retriever = retriever or Retriever(catalog)
        self.clarify_limit = clarify_limit
        # optional copy writer. without one we use fixed text.
        # either way the result goes through the grounding check
        self.writer = writer
        self._by_id = {p['product_id']: p for p in catalog.products}

    @property
    def kb(self) -> KnowledgeBase:
        return self.understanding.kb

    @property
    def schema(self) -> CatalogSchema:
        return self.kb.schema

    # ---- clarification ----

    def options_for(self, field_name: str, need: UserNeed) -> tuple[str, ...]:
        """quick options always come from the catalog, so clicking one works."""
        spec = self.schema.field(field_name)
        if spec is None:
            return ()
        if spec.kind == 'range':
            return self._price_bands(need)
        counts: dict[str, int] = {}
        for product in self.catalog.products:
            for value in _values(product, field_name, spec.level):
                counts[value] = counts.get(value, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return tuple(value for value, _ in ranked[:4])

    def price_cuts(self, need: UserNeed) -> tuple[float, ...]:
        """cut points are real prices from the candidate set.

        quartiles collapse when prices are clustered - earrings are 10 items
        with only 4 distinct prices, and all three quartile cuts landed on
        11400, which gave me a band literally labelled "11400-11400".
        using deduped real prices means every band has something in it.
        """
        query = plan(need, self.schema)
        prices = sorted({hit.price_cny for hit in self.catalog.candidates(query).hits
                         if hit.price_cny is not None})
        if len(prices) < 2:
            return ()
        if len(prices) <= 4:
            cuts = prices[:-1]              # 价位太少就按真实价位分，不强凑四档
        else:
            quartiles = {prices[len(prices) * n // 4] for n in (1, 2, 3)}
            cuts = sorted(quartiles - {prices[-1]})
        return tuple(cuts)

    def _price_bands(self, need: UserNeed) -> tuple[str, ...]:
        cuts = self.price_cuts(need)
        if not cuts:
            return ()
        bands = [f'{cuts[0]:g} 以内']
        bands += [f'{low:g}–{high:g}' for low, high in zip(cuts, cuts[1:])]
        bands.append(f'{cuts[-1]:g} 以上')
        return tuple(bands)

    def _clarify(self, result: Interpretation) -> Turn:
        asked = result.missing_core[0]
        need = result.need.clarified(self.clarify_limit)
        return Turn(action='ask_clarify', card='clarify', need=need,
                    text=QUESTIONS.get(asked, f'还需要知道：{asked}'),
                    options=self.options_for(asked, need), asked_for=asked,
                    trace=result.trace, rejected=result.rejected)

    # ---- main flow ----

    def respond(self, utterance: str, need: UserNeed | None = None) -> Turn:
        result = self.understanding.interpret(utterance, need)
        common = {'trace': result.trace, 'rejected': result.rejected}

        if result.unknown:
            # out of scope - do not enter the business flow, do not touch state
            return Turn(action='handoff', card='partial_answer', need=result.need,
                        text='这个问题我这边查不到，我们可以继续聊商品。', **common)

        intent = get_intent(result.intent)
        if result.missing_core and result.need.strategy == 'narrow_down' \
                and result.need.clarify_rounds_used < self.clarify_limit:
            return self._clarify(result)

        found = self.retriever.run(result.need, self.schema, text=utterance)
        if found.total:
            text, grounding = self._narrate(utterance, found.hits, result)
            return Turn(action=intent.first_action, card=intent.card, need=result.need,
                        text=text, hits=found.hits, total=found.total,
                        conflicts=result.conflicts, grounding=grounding, **common)

        query = plan(result.need, self.schema)
        options = self.catalog.relaxations(query)
        action = 'flag_conflict' if result.conflicts else 'fallback'
        reason = result.conflicts[0].reason if result.conflicts else '这些条件组合起来没有商品'
        return Turn(action=action, card='no_match', need=result.need, text=reason,
                    relaxations=options, conflicts=result.conflicts,
                    options=tuple(option.label for option in options), **common)

    def _narrate(self, utterance: str, hits, result) -> tuple[str, Grounding | None]:
        """run the copy through grounding. unsupported sentences get cut.

        the writer can be a model or nothing at all. either way no unchecked
        number leaves this function. writing "do not make things up" in the
        prompt relies on the model cooperating; this relies on checking.
        """
        default = ('已按你说的条件找到' if not result.need.clarify_rounds_used
                   else '先按已知条件给你看一批')
        if self.writer is None:
            return default, None
        draft = self.writer(utterance, hits, result.need)
        grounding = check(draft, hits, self._by_id)
        # everything got cut -> fall back to the fixed text, not an empty string
        return (grounding.stripped or default), grounding

    def apply(self, turn: Turn, index: int) -> UserNeed:
        """user clicked option N. the patch has to be directly executable."""
        return turn.relaxations[index].apply(turn.need, self.schema)
