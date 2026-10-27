"""Document and query embeddings over an OpenAI-compatible endpoint."""
import math
import os

from openai import OpenAI


class OllamaEmbeddings:
    # 512 covers the longest text in this catalog (353 chars) so everything
    # gets encoded in one go. that matters: the model comparison was run
    # without chunking, and if production chunks differently the scores we
    # measured stop meaning anything.
    # this value is written into the export manifest - change it and every
    # document vector has to be rebuilt.
    chunk_chars = 512
    preprocessing = 'char512_length_weighted_mean_l2_v1'

    def __init__(self, model=None, base_url=None):
        self.model = model or os.getenv('EMBEDDING_MODEL', 'all-minilm')
        self.base_url = base_url or os.getenv('EMBEDDING_BASE_URL', 'http://localhost:11434/v1')

    @classmethod
    def chunks(cls, text):
        if not text or not text.strip():
            raise ValueError('Embedding input must not be empty')
        size = cls.chunk_chars
        return [text[i:i + size] for i in range(0, len(text), size)]

    def embed(self, texts: list[str], batch_size: int = 20) -> list[list[float]]:
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        chunks = [self.chunks(text) for text in texts]
        flat = [part for group in chunks for part in group]
        vectors = []
        with OpenAI(base_url=self.base_url, api_key=os.getenv('EMBEDDING_API_KEY', 'ollama'),
                    timeout=120, max_retries=1) as client:
            for start in range(0, len(flat), batch_size):
                batch = flat[start:start + batch_size]
                result = client.embeddings.create(model=self.model, input=batch, encoding_format='float')
                items = sorted(result.data, key=lambda item: item.index)
                if [item.index for item in items] != list(range(len(batch))):
                    raise ValueError('Embedding count or index mismatch')
                vectors.extend(item.embedding for item in items)
        if not vectors:
            return []
        dim = len(vectors[0])
        if not dim or any(len(v) != dim or not all(math.isfinite(x) for x in v) for v in vectors):
            raise ValueError('Invalid embedding dimensions or values')
        output, offset = [], 0
        for group in chunks:
            total = [0.0] * dim
            for part, vector in zip(group, vectors[offset:offset + len(group)]):
                norm = math.sqrt(sum(x * x for x in vector))
                if not norm:
                    raise ValueError('Zero embedding')
                for index, value in enumerate(vector):
                    total[index] += value / norm * len(part)
            norm = math.sqrt(sum(x * x for x in total))
            if not norm:
                raise ValueError('Zero aggregated embedding')
            output.append([x / norm for x in total])
            offset += len(group)
        return output

    def embed_query(self, question: str) -> list[float]:
        return self.embed([question])[0]
