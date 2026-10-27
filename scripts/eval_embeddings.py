"""Compare embedding models on the same catalog, queries and Top-K."""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from openai import OpenAI

from product_data import ROOT, embedding_text

sys.path.insert(0, str(ROOT / 'fixtures/eval'))
from queries import gold  # noqa: E402

# name -> (model id, base_url 环境变量, api_key 环境变量, 默认 base_url)
CANDIDATES = {
    'all-minilm': ('all-minilm', 'EMBEDDING_BASE_URL', 'EMBEDDING_API_KEY', 'http://localhost:11434/v1'),
    'bge-m3': ('bge-m3', 'EMBEDDING_BASE_URL', 'EMBEDDING_API_KEY', 'http://localhost:11434/v1'),
    'qwen3-0.6b': ('qwen3-embedding:0.6b', 'EMBEDDING_BASE_URL', 'EMBEDDING_API_KEY', 'http://localhost:11434/v1'),
    'bge-small-zh': ('quentinz/bge-small-zh-v1.5', 'EMBEDDING_BASE_URL', 'EMBEDDING_API_KEY', 'http://localhost:11434/v1'),
    'dmeta-zh': ('shaw/dmeta-embedding-zh', 'EMBEDDING_BASE_URL', 'EMBEDDING_API_KEY', 'http://localhost:11434/v1'),
    'openai-3-small': ('text-embedding-3-small', 'OPENAI_BASE_URL', 'OPENAI_API_KEY', 'https://api.openai.com/v1'),
    'openai-3-large': ('text-embedding-3-large', 'OPENAI_BASE_URL', 'OPENAI_API_KEY', 'https://api.openai.com/v1'),
}
TOP_K = 10


def embed(client, model, texts, batch_size=16):
    """整段送入，不分块：候选模型的上下文都放得下本目录最长的 353 字。"""
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        result = client.embeddings.create(model=model, input=batch, encoding_format='float')
        items = sorted(result.data, key=lambda item: item.index)
        if [item.index for item in items] != list(range(len(batch))):
            raise ValueError('Embedding index mismatch')
        vectors.extend(item.embedding for item in items)
    return [normalize(v) for v in vectors]


def normalize(vector):
    norm = sum(x * x for x in vector) ** 0.5
    if not norm:
        raise ValueError('Zero vector')
    return [x / norm for x in vector]


def rank(query_vector, doc_vectors, ids):
    scored = sorted(zip(ids, (sum(a * b for a, b in zip(query_vector, d)) for d in doc_vectors)),
                    key=lambda item: -item[1])
    return [doc_id for doc_id, _ in scored]


def score(ranked, relevant):
    top = ranked[:TOP_K]
    hits = [i for i, doc_id in enumerate(top, 1) if doc_id in relevant]
    return {
        'recall@10': len(hits) / len(relevant),
        'p@5': sum(1 for h in hits if h <= 5) / 5,
        'hit@5': float(any(h <= 5 for h in hits)),
        'mrr@10': 1 / hits[0] if hits else 0.0,
        'first_rank': hits[0] if hits else None,
    }


def run(name, products, labelled):
    model, url_var, key_var, default_url = CANDIDATES[name]
    client = OpenAI(base_url=os.getenv(url_var, default_url),
                    api_key=os.getenv(key_var) or 'ollama', timeout=180, max_retries=1)
    documents = [embedding_text(p) for p in products]
    ids = [p['product_id'] for p in products]
    with client:
        started = time.perf_counter()
        doc_vectors = embed(client, model, documents)
        index_seconds = time.perf_counter() - started
        latencies, rows = [], []
        for text, group, relevant in labelled:
            tick = time.perf_counter()
            query_vector = embed(client, model, [text])[0]
            latencies.append((time.perf_counter() - tick) * 1000)
            rows.append({'query': text, 'group': group, 'gold': len(relevant),
                         **score(rank(query_vector, doc_vectors, ids), relevant)})
    metrics = {key: statistics.mean(r[key] for r in rows)
               for key in ('recall@10', 'p@5', 'hit@5', 'mrr@10')}
    return {'name': name, 'model': model, 'dimensions': len(doc_vectors[0]),
            'index_seconds': round(index_seconds, 2),
            'query_ms_p50': round(statistics.median(latencies), 1),
            'query_ms_max': round(max(latencies), 1),
            **{k: round(v, 3) for k, v in metrics.items()},
            'by_group': {g: round(statistics.mean(r['mrr@10'] for r in rows if r['group'] == g), 3)
                         for g in sorted({r['group'] for r in rows})},
            'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=['all-minilm'], choices=sorted(CANDIDATES))
    parser.add_argument('--catalog', type=Path, default=ROOT / 'fixtures/catalog/products.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT / 'fixtures/eval/embedding-report.json')
    args = parser.parse_args()

    products = [json.loads(line) for line in args.catalog.read_text(encoding='utf-8').splitlines() if line.strip()]
    labelled = gold(products)
    print(f'{len(products)} 件商品 · {len(labelled)} 条查询 · Top-{TOP_K}\n')

    previous = json.loads(args.output.read_text(encoding='utf-8')) if args.output.exists() else {'runs': []}
    results = {run['name']: run for run in previous.get('runs', [])}
    for name in args.models:
        try:
            results[name] = run(name, products, labelled)
            row = results[name]
            print(f"  {name:<16} dim={row['dimensions']:<5} MRR@10={row['mrr@10']:.3f} "
                  f"R@10={row['recall@10']:.3f} P@5={row['p@5']:.3f} "
                  f"index={row['index_seconds']}s query_p50={row['query_ms_p50']}ms")
        except Exception as error:  # 某个模型拿不到不应中断整轮比较
            print(f'  {name:<16} 失败：{type(error).__name__}: {str(error)[:120]}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {'catalog': str(args.catalog.relative_to(ROOT)), 'products': len(products),
         'queries': len(labelled), 'top_k': TOP_K,
         'runs': [results[k] for k in sorted(results)]},
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'\n报告：{args.output.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
