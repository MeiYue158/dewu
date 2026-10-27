"""Constraints -> executable search. hard/derived filter, soft ranks."""
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from backend.bm25 import BM25Index, rrf
from backend.catalog_schema import CatalogSchema
from backend.need import Constraint, UserNeed

# what to do with products that never declared the field:
# exclude (user asked for it explicitly) or keep (we only inferred it)
MissingPolicy = str


@dataclass(frozen=True)
class Filter:
    field_name: str
    op: str
    value: Any
    level: str
    missing_policy: MissingPolicy = 'exclude'
    evidence: str | None = None

    @property
    def sku_level(self) -> bool:
        return self.level in ('sku', 'sku_property')


@dataclass(frozen=True)
class Query:
    filters: tuple[Filter, ...] = ()
    boosts: tuple[Filter, ...] = ()
    text: str = ''                      # used by the hybrid side, not by filtering

    def without(self, field_name: str) -> 'Query':
        return replace(self, filters=tuple(f for f in self.filters if f.field_name != field_name))


@dataclass(frozen=True)
class Hit:
    product_id: str
    title: str
    category: str
    brand: str
    price_cny: float | None
    sku_ids: tuple[str, ...]
    matched: tuple[str, ...]            # soft conditions that hit = why we show it
    score: float


@dataclass(frozen=True)
class Result:
    hits: tuple[Hit, ...]
    total: int
    dropped_for_missing_field: dict[str, int]


@dataclass(frozen=True)
class Probe:
    field_name: str
    count_if_relaxed: int


@dataclass(frozen=True)
class Relaxation:
    """a relaxation suggestion that can actually be executed.

    `patch` goes straight into UserNeed.rewrite, so "click to search again"
    is a real thing and not just a sentence on a card with no way out.
    there is a test that applies every patch and checks the count matches
    the preview we showed.
    """
    kind: str                       # 'widen' 放宽边界 / 'drop' 丢弃条件
    fields: tuple[str, ...]
    patch: dict                     # 字段 -> (op, value)；值为 None 表示丢弃
    result_count: int
    label: str

    def apply(self, need: UserNeed, schema: CatalogSchema) -> UserNeed:
        changes = [schema.constrain(name, op, value)
                   for name, (op, value) in self.patch.items() if value is not None]
        drop = tuple(name for name, (_, value) in self.patch.items() if value is None)
        updated, _ = need.rewrite(changes, drop=drop, relaxation=True)
        return updated


def plan(need: UserNeed, schema: CatalogSchema) -> Query:
    """only fields the catalog knows about go into the query.

    session state like `occasion` stays in the need - it drives invalidation
    but there is no such column to filter on.
    """
    filters, boosts = [], []
    for constraint in need.constraints:
        spec = schema.field(constraint.field_name)
        if spec is None:
            continue
        item = Filter(field_name=constraint.field_name, op=constraint.op,
                      value=constraint.value, level=spec.level,
                      # strict exclusion only for well covered fields, otherwise
                      # one weak condition wipes out most of the catalog
                      missing_policy='exclude' if spec.hard_filter_safe else 'keep',
                      evidence=constraint.evidence)
        (filters if constraint.filters else boosts).append(item)
    return Query(tuple(filters), tuple(boosts))


class Catalog:
    def __init__(self, products: list[dict]):
        self.products = products

    @classmethod
    def load(cls, path: Path | str) -> 'Catalog':
        rows = [json.loads(line) for line in
                Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
        return cls(rows)

    # ---- reading values ----

    @staticmethod
    def _product_values(product: dict, item: Filter) -> list | None:
        """values of this field on the product. None means not declared at all
        (which is different from declared-and-did-not-match)."""
        if item.level == 'product':
            raw = product.get(item.field_name)
        elif item.level == 'attribute':
            raw = product['attributes'].get(item.field_name)
        else:
            return None
        if raw is None or raw == '':
            return None
        return list(raw) if isinstance(raw, list) else [raw]

    @staticmethod
    def _sku_values(sku: dict, item: Filter) -> list | None:
        if item.field_name == 'price_cny':
            price = sku.get('auth_price_raw')
            return [price / 100] if price else None
        found = [p['value'] for p in sku['properties']
                 if p.get('name') == item.field_name and p.get('value')]
        return found or None

    # ---- matching ----

    @staticmethod
    def _satisfies(values: list, item: Filter) -> bool:
        wanted = item.value
        if item.op == 'BETWEEN':
            low, high = wanted
            return any(low <= float(v) <= high for v in values)
        if item.op in ('<=', '>=', '<', '>'):
            target = float(wanted)
            checks = {'<=': lambda v: v <= target, '>=': lambda v: v >= target,
                      '<': lambda v: v < target, '>': lambda v: v > target}
            return any(checks[item.op](float(v)) for v in values)
        options = set(wanted) if isinstance(wanted, (list, tuple)) else {wanted}
        if item.op == 'ALL':
            present = {str(v) for v in values}
            return all(str(o) in present for o in options)
        if item.op == 'IN':
            return any(str(v) in {str(o) for o in options} for v in values)
        if item.op == 'CONTAINS':
            return any(str(o) in str(v) for v in values for o in options)
        matched = any(str(v) in {str(o) for o in options} for v in values)
        return matched if item.op == '==' else not matched

    def _match(self, product: dict, item: Filter) -> bool | None:
        """True/False 为命中与否；None 表示该商品未声明这个字段。"""
        values = self._product_values(product, item)
        return None if values is None else self._satisfies(values, item)

    def _matching_skus(self, product: dict, items: tuple[Filter, ...]) -> tuple[list[str], dict[str, bool]]:
        """all sku level conditions have to hold on one single sku.

        !!! if you check them independently you get fake matches - cheap price
        from the small size, the size the user wanted from an expensive one.
        """
        undeclared = {}
        matched = []
        for sku in product['skus']:
            ok = True
            for item in items:
                values = self._sku_values(sku, item)
                if values is None:
                    undeclared[item.field_name] = True
                    if item.missing_policy == 'exclude':
                        ok = False
                        break
                    continue
                if not self._satisfies(values, item):
                    ok = False
                    break
            if ok:
                matched.append(sku['sku_id'])
        return matched, undeclared

    # ---- search ----

    def candidates(self, query: Query) -> Result:
        """every candidate after filtering, untruncated.
        hybrid ranking needs the whole set, not the first page."""
        return self.search(query, limit=None)

    def search(self, query: Query, limit: int | None = 10) -> Result:
        product_filters = tuple(f for f in query.filters if not f.sku_level)
        sku_filters = tuple(f for f in query.filters if f.sku_level)
        dropped: dict[str, int] = {}
        hits = []
        for product in self.products:
            keep = True
            for item in product_filters:
                verdict = self._match(product, item)
                if verdict is None:
                    dropped[item.field_name] = dropped.get(item.field_name, 0) + 1
                    if item.missing_policy == 'exclude':
                        keep = False
                        break
                    continue
                if not verdict:
                    keep = False
                    break
            if not keep:
                continue
            sku_ids = [s['sku_id'] for s in product['skus']]
            if sku_filters:
                sku_ids, undeclared = self._matching_skus(product, sku_filters)
                for name in undeclared:
                    dropped[name] = dropped.get(name, 0) + 1
                if not sku_ids:
                    continue
            hits.append(self._to_hit(product, sku_ids, query.boosts))
        hits.sort(key=lambda hit: (-hit.score, hit.price_cny or float('inf'), hit.product_id))
        return Result(tuple(hits if limit is None else hits[:limit]), len(hits), dropped)

    def _to_hit(self, product: dict, sku_ids: list[str], boosts: tuple[Filter, ...]) -> Hit:
        matched = []
        for item in boosts:
            values = self._product_values(product, item)
            if values is None and item.sku_level:
                values = next((v for s in product['skus']
                               if (v := self._sku_values(s, item)) is not None), None)
            if values is not None and self._satisfies(values, item):
                matched.append(item.field_name)
        prices = [s['auth_price_raw'] / 100 for s in product['skus']
                  if s['sku_id'] in set(sku_ids) and s.get('auth_price_raw')]
        return Hit(product_id=product['product_id'], title=product['title'],
                   category=product['category'], brand=product['brand_display'],
                   price_cny=min(prices) if prices else None,
                   sku_ids=tuple(sku_ids), matched=tuple(matched),
                   score=len(matched) / len(boosts) if boosts else 0.0)

    def probe(self, query: Query) -> tuple[Probe, ...]:
        """relax one condition at a time and see how many results come back.

        facts only, no policy. which one we should actually suggest relaxing
        is somebody elses decision - see relax_options.
        """
        return tuple(sorted(
            (Probe(item.field_name, self.search(query.without(item.field_name), limit=0).total)
             for item in query.filters),
            key=lambda probe: -probe.count_if_relaxed))

    def relax_options(self, query: Query, protect: tuple[str, ...] = ('category',)) -> tuple[Probe, ...]:
        """which conditions we are willing to suggest relaxing.

        fields in `protect` never get suggested. this matters: if the user
        asked for earrings, dropping the budget still leaves 10 options, while
        dropping the *category* returns way more results but basically tells
        them to stop wanting what they wanted. sorting by result count is just
        display order, it is not how we choose.
        """
        return tuple(p for p in self.probe(query)
                     if p.count_if_relaxed > 0 and p.field_name not in protect)

    def _numbers(self, product: dict, item: Filter) -> list[float]:
        values = self._product_values(product, item)
        if values is None and item.sku_level:
            values = [v for sku in product['skus']
                      for v in (self._sku_values(sku, item) or [])]
        out = []
        for value in values or []:
            try:
                out.append(float(value))
            except (TypeError, ValueError):
                continue
        return out

    def _widen(self, query: Query, item: Filter) -> Relaxation | None:
        """move a numeric bound to a value that actually exists in the catalog.

        "raise your budget to 6750" keeps far more of the original intent than
        "forget about budget".
        """
        if item.op not in ('<=', '<', '>=', '>'):
            return None
        without = query.without(item.field_name)
        survivors = [p for p in self.products if self._passes(p, without)]
        numbers = [n for product in survivors for n in self._numbers(product, item)]
        if not numbers:
            return None
        edge = min(numbers) if item.op in ('<=', '<') else max(numbers)
        if (item.op in ('<=', '<') and edge <= float(item.value)) or \
           (item.op in ('>=', '>') and edge >= float(item.value)):
            return None                     # 边界没变松，说明瓶颈不在这条
        widened = replace(item, op='<=' if item.op in ('<=', '<') else '>=', value=edge)
        count = self.search(replace(without, filters=without.filters + (widened,)), limit=None).total
        if not count:
            return None
        shown = f'{edge:g}'
        return Relaxation('widen', (item.field_name,), {item.field_name: (widened.op, edge)},
                          count, f'{item.field_name} 放宽到 {shown}')

    def _passes(self, product: dict, query: Query) -> bool:
        for item in query.filters:
            if item.sku_level:
                continue
            verdict = self._match(product, item)
            if verdict is None:
                if item.missing_policy == 'exclude':
                    return False
                continue
            if not verdict:
                return False
        sku_filters = tuple(f for f in query.filters if f.sku_level)
        if sku_filters:
            matched, _ = self._matching_skus(product, sku_filters)
            return bool(matched)
        return True

    def relaxations(self, query: Query, protect: tuple[str, ...] = ('category',),
                    limit: int = 4, max_fields: int = 2) -> tuple[Relaxation, ...]:
        """suggestions ordered by how much of the original intent survives:
        move a bound first, then drop one condition, then two.

        !!! has to keep going when no single relaxation helps, otherwise a
        dead end has no exit at all and the user is just stuck.
        """
        if self.search(query, limit=0).total:
            return ()
        candidates = tuple(f for f in query.filters if f.field_name not in protect)
        found: list[Relaxation] = []

        for item in candidates:                              # 1. move a bound - keeps most intent
            widened = self._widen(query, item)
            if widened:
                found.append(widened)
        for item in candidates:                              # 2. drop one
            count = self.search(query.without(item.field_name), limit=0).total
            if count:
                found.append(Relaxation('drop', (item.field_name,),
                                        {item.field_name: (item.op, None)}, count,
                                        f'不限 {item.field_name}'))
        if not found and max_fields >= 2:                    # 3. two at a time
            for dropped in candidates:
                trimmed = query.without(dropped.field_name)
                # drop one, then see if moving a bound fixes the rest.
                # still better than throwing away both conditions
                for other in (f for f in trimmed.filters if f.field_name not in protect):
                    widened = self._widen(trimmed, other)
                    if widened:
                        found.append(Relaxation(
                            'widen', (dropped.field_name, other.field_name),
                            {dropped.field_name: (dropped.op, None), **widened.patch},
                            widened.result_count,
                            f'不限 {dropped.field_name}，{widened.label}'))
                count = self.search(trimmed, limit=0).total
                if count:
                    found.append(Relaxation('drop', (dropped.field_name,),
                                            {dropped.field_name: (dropped.op, None)}, count,
                                            f'不限 {dropped.field_name}'))
            for first in range(len(candidates)):
                for second in range(first + 1, len(candidates)):
                    names = (candidates[first].field_name, candidates[second].field_name)
                    trimmed = query.without(names[0]).without(names[1])
                    count = self.search(trimmed, limit=0).total
                    if count:
                        found.append(Relaxation(
                            'drop', names,
                            {n: (f.op, None) for n, f in
                             zip(names, (candidates[first], candidates[second]))},
                            count, f'同时不限 {names[0]} 和 {names[1]}'))
        order = {'widen': 0, 'drop': 1}
        found.sort(key=lambda r: (order[r.kind], len(r.fields), -r.result_count))
        return tuple(found[:limit])

    def blocking(self, query: Query, protect: tuple[str, ...] = ('category',)) -> Probe | None:
        """the bottleneck: which condition unblocks things without giving up
        whatever the user anchored on."""
        options = self.relax_options(query, protect)
        return options[0] if options else None


class Retriever:
    """filter first, then rank inside whatever survived.

    keeping these two apart is not academic. "digital products under 1000"
    never got above rank 4 no matter how I tuned retrieval, but as a filter
    it is precision 1.00 / recall 1.00 - it was a structured query pretending
    to be a text query the whole time.

    so: retrieval decides the order inside the candidate set. it does not get
    to decide who is in the set.
    """

    def __init__(self, catalog: 'Catalog', index: BM25Index | None = None,
                 vectors: dict[str, list[float]] | None = None, embedder=None,
                 dense_weight: float | None = None, lexical_weight: float | None = None,
                 rrf_k: int | None = None):
        self.catalog = catalog
        self.index = index if index is not None else BM25Index.over_products(catalog.products)
        self.vectors = vectors or {}
        self.embedder = embedder
        # dense is the stronger arm on this corpus (Hit@5 1.000 vs 0.850) so it
        # gets more weight by default. at 1:1 a query dense had at rank 1 fell
        # to rank 6 because bm25 missed it completely and diluted the fusion.
        self.dense_weight = float(os.getenv('SOURCE_WEIGHT_VECTOR', '2.0')
                                  if dense_weight is None else dense_weight)
        self.lexical_weight = float(os.getenv('SOURCE_WEIGHT_ES', '1.0')
                                    if lexical_weight is None else lexical_weight)
        self.rrf_k = int(os.getenv('RRF_K', '60') if rrf_k is None else rrf_k)

    def _dense(self, text: str, allowed: set[str]) -> list[str]:
        if not (self.embedder and self.vectors):
            return []
        query = self.embedder.embed_query(text)
        scored = ((sum(a * b for a, b in zip(query, vector)), doc_id)
                  for doc_id, vector in self.vectors.items() if doc_id in allowed)
        return [doc_id for _, doc_id in sorted(scored, reverse=True)]

    def run(self, need: UserNeed, schema: CatalogSchema, text: str = '',
            limit: int = 10) -> Result:
        result = self.catalog.candidates(plan(need, schema))
        if not text or not result.hits:
            return replace(result, hits=result.hits[:limit])
        allowed = {hit.product_id for hit in result.hits}
        rankings = [r for r in (self._dense(text, allowed),
                                [s.doc_id for s in self.index.search(text, limit=len(allowed),
                                                                     allowed=allowed)]) if r]
        if not rankings:
            return replace(result, hits=result.hits[:limit])
        weights = ((self.dense_weight, self.lexical_weight) if len(rankings) == 2
                   else (1.0,))
        order = rrf(*rankings, k=self.rrf_k, weights=weights, limit=len(allowed))
        position = {doc_id: index for index, doc_id in enumerate(order)}
        ranked = sorted(result.hits,
                        key=lambda hit: (position.get(hit.product_id, len(order)), hit.product_id))
        return replace(result, hits=tuple(ranked[:limit]))
