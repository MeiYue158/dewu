"""Grounding check: every fact in the reply must point at a retrieved field."""
import re
from dataclasses import dataclass

from backend.retrieval import Hit

# prices, carats, lengths... and bare numbers. strip thousands separators first
NUMBER = re.compile(r'\d[\d,]*(?:\.\d+)?')
# numbers after these words are counts, not product facts. skip them
COUNTING = re.compile(r'(?:共|找到|为你|一共|有)\s*$')


@dataclass(frozen=True)
class Claim:
    text: str
    kind: str                       # 'number' / 'name'
    supported: bool
    source: str | None = None       # which field backs it, so we can show the source


@dataclass(frozen=True)
class Grounding:
    claims: tuple[Claim, ...]
    stripped: str                   # the text with unsupported sentences removed

    @property
    def unsupported(self) -> tuple[Claim, ...]:
        return tuple(c for c in self.claims if not c.supported)

    @property
    def ok(self) -> bool:
        return not self.unsupported


def evidence_of(hits: tuple[Hit, ...], products: dict[str, dict] | None = None) -> dict[str, str]:
    """flatten everything quotable this turn into value -> source field.

    only the search results go in here, not the whole catalog. if the model
    talks about a product we never retrieved, that is unsupported too even
    though the product exists.
    """
    table: dict[str, str] = {}
    for hit in hits:
        for value, source in ((hit.title, 'title'), (hit.brand, 'brand'),
                              (hit.category, 'category')):
            if not value:
                continue
            table.setdefault(str(value), source)
            # numbers inside a product name ("TIFFANY 1837", "18K") are backed
            # by the name itself. !!! forgetting this deleted a perfectly good
            # sentence that just quoted the real title
            for number in NUMBER.findall(str(value)):
                table.setdefault(_canonical(number), source)
        if hit.price_cny is not None:
            table.setdefault(_canonical(hit.price_cny), 'price_cny')
        for sku in hit.sku_ids:
            table.setdefault(str(sku), 'sku_id')
        product = (products or {}).get(hit.product_id) or {}
        for key, raw in (product.get('attributes') or {}).items():
            for item in (raw if isinstance(raw, list) else [raw]):
                if str(item).strip():
                    table.setdefault(str(item).strip(), f'attributes.{key}')
                    for number in NUMBER.findall(str(item)):
                        table.setdefault(_canonical(number), f'attributes.{key}')
        for sku in product.get('skus') or []:
            if sku.get('auth_price_raw'):
                table.setdefault(_canonical(sku['auth_price_raw'] / 100), 'skus.price')
    return table


def _canonical(value) -> str:
    text = str(value).replace(',', '')
    try:
        number = float(text)
    except ValueError:
        return text
    return f'{number:g}'


def check(text: str, hits: tuple[Hit, ...],
          products: dict[str, dict] | None = None) -> Grounding:
    """check each number. cut the whole sentence, not just the number.

    "the other one is only ~~9~~ yuan" is more misleading than deleting it.
    """
    table = evidence_of(hits, products)
    claims: list[Claim] = []
    keep: list[str] = []
    for sentence in _sentences(text):
        unsupported = False
        for match in NUMBER.finditer(sentence):
            if COUNTING.search(sentence[:match.start()]):
                continue                # "11 results" is a count, not a product fact
            value = _canonical(match.group())
            source = table.get(value)
            claims.append(Claim(match.group(), 'number', source is not None, source))
            if source is None:
                unsupported = True
        if not unsupported:
            keep.append(sentence)
    return Grounding(tuple(claims), ''.join(keep).strip())


def _sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[。！？\n])', text or '')
    return [part for part in parts if part.strip()]
