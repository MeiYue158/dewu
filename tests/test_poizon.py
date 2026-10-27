import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_products import generate, load_snapshot
from product_data import embedding_text
from backend.embeddings import OllamaEmbeddings

DATA = ROOT / 'fixtures/products'
RAW = ROOT / 'fixtures/raw/poizon'


def test_source_to_exports(tmp_path):
    generate(RAW, tmp_path)
    for path in tmp_path.iterdir():
        assert path.read_bytes() == (DATA / path.name).read_bytes()
    rows = [json.loads(s) for s in (DATA / 'products.jsonl').read_text().splitlines()]
    bulk = [json.loads(s) for s in (DATA / 'es.bulk.jsonl').read_text().splitlines()]
    inputs = [json.loads(s) for s in (DATA / 'qdrant.inputs.jsonl').read_text().splitlines()]
    assert len(rows) == 103
    assert len(bulk) == 206
    assert len({r['product_id'] for r in rows}) == 103
    for i, (p, inp) in enumerate(zip(rows, inputs)):
        source, _ = load_snapshot(RAW / f"{p['spu_id']}.json")
        assert p['title'] == source['data']['detail']['title']
        assert p['description'] == source['data']['detail']['desc']
        assert inp['text'] == embedding_text(p)
        assert inp['payload'] == p == bulk[2*i+1]
        assert inp['id'] == p['product_id'] == bulk[2*i]['index']['_id']
        assert p['source']['collected_at'] is None
        assert all(prop['name'] for sku in p['skus'] for prop in sku['properties'])
    report = json.loads((DATA / 'quality-report.json').read_text())
    assert report['rejected'] == []
    assert len(report['repaired_files']) == 20
    assert report['sku_count'] == 598


def test_repairs_preserve_content(tmp_path):
    path = tmp_path / 'broken.json'
    path.write_text('{"value":""quoted"","propertyValueId":1,"other":"a\tb"}')
    doc, fixes = load_snapshot(path)
    assert doc['value'] == '"quoted"'
    assert doc['other'] == 'a\tb'
    assert len(fixes) == 2


def test_chunking_preserves_every_character():
    size = OllamaEmbeddings.chunk_chars
    text = '标题：商品\n属性：' + '宽楦透气缓震；' * 100
    chunks = OllamaEmbeddings.chunks(text)
    assert ''.join(chunks) == text
    assert len(chunks) > 1 and all(len(c) <= size for c in chunks)


def test_exported_real_vectors():
    rows = [json.loads(s) for s in (DATA / 'vectors/qdrant.points.jsonl').read_text().splitlines()]
    source_ids = {json.loads(s)['product_id'] for s in (DATA / 'products.jsonl').read_text().splitlines()}
    assert len(rows) == 103
    assert {r['id'] for r in rows} == source_ids
    for row in rows:
        assert len(row['vector']) == 512
        assert all(math.isfinite(x) for x in row['vector'])
        assert abs(sum(x*x for x in row['vector']) - 1) < 1e-5


def test_embedding_aggregation_uses_tail_and_restores_api_order(monkeypatch):
    from types import SimpleNamespace
    import backend.embeddings as module

    class Client:
        def __init__(self, **kwargs):
            self.embeddings = self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def create(self, **kwargs):
            return SimpleNamespace(data=[
                SimpleNamespace(index=i, embedding=[1.0, 0.0] if s.startswith('a') else [0.0, 1.0])
                for i, s in reversed(list(enumerate(kwargs['input'])))])

    monkeypatch.setattr(module, 'OpenAI', Client)
    size = OllamaEmbeddings.chunk_chars
    service = OllamaEmbeddings()
    vectors = service.embed(['a'*size + 'b'*(size//2), 'b'*20])
    assert abs(vectors[0][0] - 2/math.sqrt(5)) < 1e-9
    assert abs(vectors[0][1] - 1/math.sqrt(5)) < 1e-9
    assert vectors[1] == [0.0, 1.0]
    assert service.embed_query('a'*size + 'b'*(size//2)) == vectors[0]
