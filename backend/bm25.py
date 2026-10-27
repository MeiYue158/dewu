"""BM25 over the catalog, tokenized the way the ES mapping says it will be."""
import math
import re
from collections import Counter
from dataclasses import dataclass

K1 = 1.5
B = 0.75
# latin/digit runs stay whole, CJK gets split into bigrams - same as ES cjk
LATIN = re.compile(r'[0-9a-z]+')
CJK = re.compile(r'[㐀-䶿一-鿿぀-ヿ가-힯]+')
SPLIT = re.compile(r'[^0-9a-z㐀-䶿一-鿿぀-ヿ가-힯]+')


def tokenize(text: str) -> list[str]:
    tokens = []
    for chunk in SPLIT.split((text or '').lower()):
        if not chunk:
            continue
        for part in re.findall(r'[0-9a-z]+|[^0-9a-z]+', chunk):
            if LATIN.fullmatch(part):
                tokens.append(part)
            elif len(part) == 1:
                tokens.append(part)
            else:
                # bigrams: 蓝牙音箱 -> 蓝牙, 牙音, 音箱
                tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
    return tokens


def indexed_text(product: dict) -> str:
    """same whitelist as the vector side: title + attributes only.

    description and category stay out. scripts/product_data.py:embedding_text
    is where the whitelist really lives - test_bm25 asserts the two are equal
    so changing one without the other fails loudly.
    """
    attributes = '；'.join(
        f'{key}：' + ('、'.join(value) if isinstance(value, list) else str(value))
        for key, value in sorted(product['attributes'].items()))
    return f"标题：{product['title']}\n属性：{attributes}"


@dataclass(frozen=True)
class Scored:
    doc_id: str
    score: float


class BM25Index:
    def __init__(self, documents: dict[str, str]):
        self.ids = list(documents)
        self.terms = {doc_id: Counter(tokenize(text)) for doc_id, text in documents.items()}
        self.lengths = {doc_id: sum(counts.values()) for doc_id, counts in self.terms.items()}
        total = sum(self.lengths.values())
        self.average_length = total / len(self.ids) if self.ids else 0.0
        frequency: Counter = Counter()
        for counts in self.terms.values():
            frequency.update(counts.keys())
        count = len(self.ids)
        self.idf = {term: math.log(1 + (count - df + 0.5) / (df + 0.5))
                    for term, df in frequency.items()}
        self.postings: dict[str, list[str]] = {}
        for doc_id, counts in self.terms.items():
            for term in counts:
                self.postings.setdefault(term, []).append(doc_id)

    @classmethod
    def over_products(cls, products: list[dict]) -> 'BM25Index':
        return cls({p['product_id']: indexed_text(p) for p in products})

    def search(self, query: str, limit: int = 10,
               allowed: set[str] | None = None) -> tuple[Scored, ...]:
        """`allowed` is the post-filter candidate set. we only rank inside it."""
        wanted = Counter(tokenize(query))
        if not wanted:
            return ()
        scores: dict[str, float] = {}
        for term, query_count in wanted.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id in self.postings[term]:
                if allowed is not None and doc_id not in allowed:
                    continue
                tf = self.terms[doc_id][term]
                norm = 1 - B + B * self.lengths[doc_id] / (self.average_length or 1)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (K1 + 1) / (tf + K1 * norm) * query_count
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return tuple(Scored(doc_id, score) for doc_id, score in ranked[:limit])


def rrf(*rankings: list[str], k: int = 60, weights: tuple[float, ...] | None = None,
        limit: int = 10) -> tuple[str, ...]:
    """reciprocal rank fusion.

    merges positions rather than scores, which is why it needs no per-corpus
    tuning. mildly counterintuitive: because 1/x is convex, a doc at ranks
    (1,3) scores HIGHER than one at (2,2). so RRF rewards showing up in more
    lists, not having a steady rank. i got that backwards in a test first.
    """
    if weights is None:
        weights = (1.0,) * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError('one weight per ranking')
    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for position, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + position)
    best = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return tuple(doc_id for doc_id, _ in best[:limit])
