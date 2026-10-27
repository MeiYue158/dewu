import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from embed_products import export, validate_vectors

FIXTURES = ROOT / 'fixtures/products'


def read_lines(name):
    return [json.loads(line) for line in (FIXTURES / name).read_text().splitlines()]


@pytest.mark.parametrize('vectors,count', [([], 1), ([[0, 0]], 1), ([[1, 0], [1]], 2),
                                          ([[float('nan'), 1]], 1), ([[1, 0]], 2)])
def test_invalid_vectors_rejected(vectors, count):
    with pytest.raises(ValueError):
        validate_vectors(vectors, count)


def test_vector_export_contract(tmp_path):
    # Deliberately synthetic unit-test vectors, never shipped as semantic embeddings.
    rows = read_lines('qdrant.inputs.jsonl')[:2]
    assert export(rows, [[1.0, 0.0], [0.0, 1.0]], tmp_path, 'test-only', 'test', 'test-hash') == 2
    result = json.loads((tmp_path / 'qdrant.upsert.json').read_text())
    assert [p['id'] for p in result['points']] == [r['id'] for r in rows]
    assert json.loads((tmp_path / 'qdrant.collection.json').read_text())['vectors'] == {
        'size': 2, 'distance': 'Cosine'}
