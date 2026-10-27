"""Compare dense, BM25 and their fusion on the same labelled queries."""
import argparse
import json
import os
import statistics
import sys
from datetime import date
from pathlib import Path

from product_data import ROOT, dump

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'fixtures/eval'))
from backend.bm25 import BM25Index, rrf           # noqa: E402
from backend.embeddings import OllamaEmbeddings   # noqa: E402
from queries import gold                          # noqa: E402

TOP_K = 10


def dense_ranking(service, vectors, text, limit):
    query = service.embed_query(text)
    scored = sorted(((sum(a * b for a, b in zip(query, point['vector'])), point['id'])
                     for point in vectors), reverse=True)
    return [doc_id for _, doc_id in scored[:limit]]


def score(ranked, relevant):
    hits = [i for i, doc_id in enumerate(ranked[:TOP_K], 1) if doc_id in relevant]
    return {'recall@10': len(hits) / len(relevant),
            'p@5': sum(1 for h in hits if h <= 5) / 5,
            'hit@5': float(any(h <= 5 for h in hits)),
            'mrr@10': 1 / hits[0] if hits else 0.0,
            'first_rank': hits[0] if hits else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=ROOT / 'fixtures/catalog/products.jsonl')
    parser.add_argument('--vectors', type=Path,
                        default=ROOT / 'fixtures/catalog/vectors/qdrant.points.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT / 'fixtures/eval/retrieval-report.json')
    parser.add_argument('--rrf-k', type=int, default=int(os.getenv('RRF_K', '60')))
    parser.add_argument('--dense-weight', type=float,
                        default=float(os.getenv('SOURCE_WEIGHT_VECTOR', '2.0')))
    parser.add_argument('--lexical-weight', type=float,
                        default=float(os.getenv('SOURCE_WEIGHT_ES', '1.0')))
    parser.add_argument('--today', default=date.today().isoformat())
    args = parser.parse_args()

    products = [json.loads(line) for line in
                args.catalog.read_text(encoding='utf-8').splitlines() if line.strip()]
    vectors = [json.loads(line) for line in
               args.vectors.read_text(encoding='utf-8').splitlines() if line.strip()]
    labelled = gold(products)
    index = BM25Index.over_products(products)
    service = OllamaEmbeddings()
    pool = len(products)

    rows = []
    for text, group, relevant in labelled:
        dense = dense_ranking(service, vectors, text, pool)
        lexical = [s.doc_id for s in index.search(text, limit=pool)]
        fused = list(rrf(dense, lexical, k=args.rrf_k, limit=TOP_K,
                         weights=(args.dense_weight, args.lexical_weight)))
        rows.append({'query': text, 'group': group, 'gold': len(relevant),
                     'dense': score(dense, relevant), 'bm25': score(lexical, relevant),
                     'hybrid': score(fused, relevant)})

    summary = {arm: {metric: round(statistics.mean(row[arm][metric] for row in rows), 3)
                     for metric in ('mrr@10', 'recall@10', 'p@5', 'hit@5')}
               for arm in ('dense', 'bm25', 'hybrid')}
    dump(args.output, {'generated_at': args.today, 'products': pool, 'queries': len(rows),
                       'top_k': TOP_K, 'rrf_k': args.rrf_k,
                       'weights': {'dense': args.dense_weight, 'lexical': args.lexical_weight},
                       'summary': summary, 'rows': rows})

    print(f'{pool} 件商品 · {len(rows)} 条查询 · Top-{TOP_K} · RRF k={args.rrf_k} · '
          f'权重 dense:bm25 = {args.dense_weight:g}:{args.lexical_weight:g}\n')
    print(f"{'':<10}{'MRR@10':>9}{'R@10':>8}{'P@5':>7}{'Hit@5':>8}")
    for arm in ('dense', 'bm25', 'hybrid'):
        value = summary[arm]
        print(f"{arm:<10}{value['mrr@10']:>9.3f}{value['recall@10']:>8.3f}"
              f"{value['p@5']:>7.3f}{value['hit@5']:>8.3f}")
    print(f"\n{'查询':<22}{'dense':>7}{'bm25':>7}{'hybrid':>8}   （第一个相关结果的排名）")
    for row in rows:
        marks = [str(row[arm]['first_rank'] or '—') for arm in ('dense', 'bm25', 'hybrid')]
        flag = '  ←' if row['dense']['first_rank'] != 1 else ''
        print(f"{row['query']:<22}{marks[0]:>7}{marks[1]:>7}{marks[2]:>8}{flag}")
    print(f'\n报告：{args.output.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
