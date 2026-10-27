"""Create real Qdrant vectors from title + attributes only; does not upload data."""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

from product_data import ROOT, dump, embedding_text, jsonl


def validate_vectors(vectors, count):
    if len(vectors) != count or not vectors:
        raise ValueError('Embedding response count does not match input count')
    size = len(vectors[0])
    if not size:
        raise ValueError('Empty vector')
    for vector in vectors:
        if len(vector) != size or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                          and math.isfinite(x) for x in vector):
            raise ValueError('Vectors must have consistent dimensions and finite numeric values')
        if not any(x != 0 for x in vector):
            raise ValueError('Zero vector is invalid for cosine search')
    return size


def export(rows, vectors, output, model, provider, input_sha256):
    size = validate_vectors(vectors, len(rows))
    points = [{'id': row['id'], 'vector': vector, 'payload': row['payload']}
              for row, vector in zip(rows, vectors)]
    output.mkdir(parents=True, exist_ok=True)
    jsonl(output / 'qdrant.points.jsonl', points)
    dump(output / 'qdrant.upsert.json', {'points': points})
    dump(output / 'qdrant.collection.json', {'vectors': {'size': size, 'distance': 'Cosine'}})
    dump(output / 'embedding.manifest.json', {
        'model': model, 'provider': provider, 'dimensions': size, 'count': len(points),
        'distance': 'Cosine', 'embedding_fields': ['title', 'attributes'],
        'input_sha256': input_sha256,
        'note': 'Use this same embedding model and preprocessing for query vectors.',
    })
    return size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=['ollama', 'openai', 'local'], default='ollama')
    parser.add_argument('--model', default=os.getenv('EMBEDDING_MODEL', 'all-minilm'), help='Embedding model ID or local model path')
    parser.add_argument('--input', type=Path, default=ROOT / 'fixtures/products/qdrant.inputs.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT / 'fixtures/products/vectors')
    parser.add_argument('--batch-size', type=int, default=20)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    raw = args.input.read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Input must contain unique, nonempty product rows')
    for row in rows:
        if row['text'] != embedding_text(row['payload']) or row['id'] != row['payload']['product_id']:
            raise ValueError('Embedding text or product identity does not match source product')
    texts = [row['text'] for row in rows]
    vectors = []
    if args.provider == 'ollama':
        sys.path.insert(0, str(ROOT))
        from backend.embeddings import OllamaEmbeddings
        vectors = OllamaEmbeddings(model=args.model).embed(texts, batch_size=args.batch_size)
    elif args.provider == 'local':
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.model)
        vectors = model.encode(texts, batch_size=args.batch_size,
                               normalize_embeddings=True, show_progress_bar=True).tolist()
    else:
        from openai import OpenAI
        # Do not read or print repository secrets. Credentials are passed by environment.
        options = {}
        if os.getenv('EMBEDDING_BASE_URL'):
            options['base_url'] = os.environ['EMBEDDING_BASE_URL']
        if os.getenv('EMBEDDING_API_KEY'):
            options['api_key'] = os.environ['EMBEDDING_API_KEY']
        with OpenAI(**options) as client:
            for start in range(0, len(texts), args.batch_size):
                batch = texts[start:start + args.batch_size]
                response = client.embeddings.create(model=args.model, input=batch, encoding_format='float')
                items = sorted(response.data, key=lambda item: item.index)
                if [item.index for item in items] != list(range(len(batch))):
                    raise ValueError('Embedding API returned missing or duplicate indexes')
                vectors.extend(item.embedding for item in items)
    size = export(rows, vectors, args.output, args.model, args.provider, hashlib.sha256(raw).hexdigest())
    if args.provider == 'ollama':
        manifest_path = args.output / 'embedding.manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.update(preprocessing=OllamaEmbeddings.preprocessing,
                        chunk_chars=OllamaEmbeddings.chunk_chars,
                        input_chunks=sum(len(OllamaEmbeddings.chunks(t)) for t in texts))
        dump(manifest_path, manifest)
    print(f'Exported {len(rows)} real vectors, dimensions={size}, directory={args.output}')


if __name__ == '__main__':
    main()
