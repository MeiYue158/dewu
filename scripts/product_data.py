"""Shared product text and serialization helpers."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def attributes_text(attributes):
    return '；'.join(f'{key}：' + ('、'.join(value) if isinstance(value, list) else str(value))
                    for key, value in sorted(attributes.items()))


def embedding_text(product):
    # 唯一白名单入口：禁止描述、类目或其他元数据进入向量输入。
    return f"标题：{product['title']}\n属性：{attributes_text(product['attributes'])}"


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
