import json
import sys
from pathlib import Path

import pytest

from backend.bm25 import BM25Index, indexed_text, rrf, tokenize
from backend.knowledge import load_default
from backend.need import UserNeed
from backend.retrieval import Catalog, Retriever

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from product_data import embedding_text  # noqa: E402

KB = load_default(ROOT)
SCHEMA = KB.schema
PRODUCTS = [json.loads(line) for line in
            (ROOT / 'fixtures/catalog/products.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()]
CATALOG = Catalog(PRODUCTS)
INDEX = BM25Index.over_products(PRODUCTS)


# ---- 分词 ----

def test_cjk_is_split_into_bigrams_like_the_es_analyzer():
    assert tokenize('蓝牙音箱') == ['蓝牙', '牙音', '音箱']
    assert tokenize('银') == ['银']


def test_latin_and_digits_stay_whole_and_lowercase():
    assert tokenize('925银 18K金') == ['925', '银', '18k', '金']
    assert tokenize('IPX7防水') == ['ipx7', '防水']
    assert tokenize('USB-C') == ['usb', 'c']


def test_empty_input_yields_no_tokens():
    assert tokenize('') == [] and tokenize('   ') == [] and tokenize(None) == []


# ---- 索引内容 ----

def test_indexed_text_matches_the_embedding_whitelist():
    # 两条臂必须看同一份文本，否则它们的名次不可比，融合也就没有意义。
    assert all(indexed_text(product) == embedding_text(product) for product in PRODUCTS)


def test_description_never_reaches_the_index():
    product = next(p for p in PRODUCTS if p['description'])
    assert product['description'] not in indexed_text(product)


# ---- 检索 ----

def test_literal_category_query_ranks_its_own_products_first():
    for name in ('蓝牙音箱', '机械键盘', '羽绒服'):
        top = INDEX.search(name, limit=3)
        found = {next(p for p in PRODUCTS if p['product_id'] == s.doc_id)['category']
                 for s in top}
        assert name in found


def test_literal_attribute_values_are_findable():
    # BM25 的长处正在这里：「不锈钢」「礼盒」是字面属性值，稠密检索容易漏。
    hits = INDEX.search('不锈钢表带', limit=5)
    assert hits and any('不锈钢' in indexed_text(
        next(p for p in PRODUCTS if p['product_id'] == s.doc_id)) for s in hits)


def test_unknown_terms_score_nothing_rather_than_erroring():
    assert INDEX.search('量子纠缠护身符') == ()
    assert INDEX.search('') == ()


def test_allowed_restricts_scoring_to_the_candidate_set():
    everything = INDEX.search('项链', limit=200)
    subset = {s.doc_id for s in everything[:3]}
    limited = INDEX.search('项链', limit=200, allowed=subset)
    assert {s.doc_id for s in limited} == subset


# ---- RRF ----

def test_rrf_rewards_appearing_in_both_rankings():
    """RRF 的要点：两边都出现的文档，胜过只在一边排第一的。"""
    dense, lexical = ['x', 'y'], ['y', 'z']
    assert rrf(dense, lexical, limit=1) == ('y',)      # y 在两边都有
    assert set(rrf(dense, lexical, limit=3)) == {'x', 'y', 'z'}


def test_rrf_does_not_reward_middling_consistency():
    """反直觉但正确：1/x 是凸函数，名次「一头一尾」的和大于「都在中间」。

    k=60 时 1/61 + 1/63 = 0.032266 > 2/62 = 0.032258。
    所以 RRF 奖励的是「出现在更多榜单里」，不是「名次更稳」。
    """
    a, b = ['x', 'y', 'z'], ['z', 'y', 'x']
    assert rrf(a, b, limit=1) == ('x',)


def test_rrf_weights_favour_the_stronger_arm():
    dense, lexical = ['good'], ['noise']
    assert rrf(dense, lexical, weights=(3.0, 1.0), limit=1) == ('good',)
    assert rrf(dense, lexical, weights=(1.0, 3.0), limit=1) == ('noise',)
    with pytest.raises(ValueError, match='one weight per ranking'):
        rrf(dense, lexical, weights=(1.0,))


# ---- 过滤 + 排序的组合 ----

def test_retriever_ranks_inside_the_filtered_set_only():
    retriever = Retriever(CATALOG, index=INDEX)        # 不给向量，退化为纯 BM25
    need = UserNeed(intent='discover', constraints=(SCHEMA.constrain('category', '==', '项链'),))
    result = retriever.run(need, SCHEMA, text='银色 爱心', limit=5)
    assert result.total == 25
    assert all(hit.category == '项链' for hit in result.hits)


def test_retriever_without_text_keeps_the_structured_order():
    retriever = Retriever(CATALOG, index=INDEX)
    need = UserNeed(intent='discover', constraints=(SCHEMA.constrain('category', '==', '手链'),))
    plain = retriever.run(need, SCHEMA, text='', limit=14)
    prices = [h.price_cny for h in plain.hits if h.price_cny is not None]
    assert prices == sorted(prices)


def test_filters_beat_retrieval_on_structured_questions():
    """「预算一千以内的数码产品」检索怎么调都排第 4，过滤是 1.00/1.00。"""
    digital = ['真无线耳机', '头戴式耳机', '智能手表', '机械键盘', '移动电源', '蓝牙音箱']
    need = UserNeed(intent='discover', constraints=(
        SCHEMA.constrain('category', 'IN', digital),
        SCHEMA.constrain('price_cny', '<=', 1000)))
    result = Retriever(CATALOG, index=INDEX).run(need, SCHEMA, limit=20)
    assert result.total == 15
    assert all(hit.price_cny <= 1000 and hit.category in digital for hit in result.hits)
