import json
from pathlib import Path
from types import SimpleNamespace

from backend.knowledge import load_default
from backend.need import ACTIONS, UserNeed
from backend.orchestrator import Orchestrator
from backend.retrieval import Catalog, Retriever, plan
from backend.understanding import Understanding

ROOT = Path(__file__).resolve().parents[1]
KB = load_default(ROOT)
CATALOG = Catalog.load(ROOT / 'fixtures/catalog/products.jsonl')


class FakeClient:
    def __init__(self, replies):
        self.replies = [json.dumps(r, ensure_ascii=False) if isinstance(r, dict) else r
                        for r in (replies if isinstance(replies, list) else [replies])]
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def build(replies, clarify_limit=3):
    client = FakeClient(replies)
    reader = Understanding(KB, model='t', api_key='k', client_factory=lambda: client)
    return Orchestrator(reader, CATALOG, Retriever(CATALOG), clarify_limit=clarify_limit)


def slots(*pairs):
    return {'intent': 'discover',
            'slots': [{'field': f, 'op': o, 'value': v} for f, o, v in pairs]}


# ---- 越界 ----

def test_out_of_scope_hands_off_without_touching_state():
    before = UserNeed(intent='discover', constraints=(KB.schema.constrain('category', '==', '项链'),))
    turn = build({'intent': 'unknown', 'slots': []}).respond('今天天气怎么样', before)
    assert turn.action == 'handoff' and turn.card == 'partial_answer'
    assert turn.need.of('category').value == '项链'      # 已有状态不被清掉
    assert turn.hits == ()


# ---- 澄清 ----

def test_missing_core_slot_asks_with_catalog_backed_options():
    turn = build(slots(('category', '==', '项链'))).respond('想买条项链')
    assert turn.action == 'ask_clarify' and turn.card == 'clarify'
    assert turn.asked_for == 'price_cny' and turn.needs_user
    # 价位档来自当前候选集的真实分布，所以点哪个都有结果。
    assert len(turn.options) == 4 and all('–' in o or '以' in o for o in turn.options)
    assert turn.need.clarify_rounds_used == 1


def test_price_bands_are_never_degenerate():
    """价位集中时四分位切点会撞在一起，档位就成了「11400–11400」这种空档。"""
    orchestrator = build(slots())
    for name in ('耳钉', '项链', '移动电源', '手链'):
        need = UserNeed(intent='discover',
                        constraints=(KB.schema.constrain('category', '==', name),))
        cuts = orchestrator.price_cuts(need)
        assert len(cuts) == len(set(cuts)), name          # 切点不重复
        assert list(cuts) == sorted(cuts), name
        bands = orchestrator.options_for('price_cny', need)
        assert len(bands) == len(cuts) + 1
        assert len(set(bands)) == len(bands), name        # 档位标签不重复


def test_every_price_band_actually_contains_products():
    """点哪一档都要有结果——快捷选项不能是死路。"""
    orchestrator = build(slots())
    for name in ('耳钉', '项链', '移动电源'):
        base = (KB.schema.constrain('category', '==', name),)
        cuts = orchestrator.price_cuts(UserNeed(intent='discover', constraints=base))
        bounds = [(None, cuts[0])] + list(zip(cuts, cuts[1:])) + [(cuts[-1], None)]
        for low, high in bounds:
            # 一个字段只能有一条约束，所以中间档走 BETWEEN 而不是两条边界。
            if low is None:
                band = KB.schema.constrain('price_cny', '<=', high)
            elif high is None:
                band = KB.schema.constrain('price_cny', '>=', low)
            else:
                band = KB.schema.constrain('price_cny', 'BETWEEN', [low, high])
            need = UserNeed(intent='discover', constraints=base + (band,))
            found = CATALOG.search(plan(need, KB.schema), limit=0)
            assert found.total > 0, f'{name} {low}–{high} 是空档'


def test_a_single_price_point_offers_no_bands():
    orchestrator = build(slots())
    need = UserNeed(intent='discover', constraints=(
        KB.schema.constrain('category', '==', '羽绒服'),))     # 只有 1 件
    assert orchestrator.price_cuts(need) == ()
    assert orchestrator.options_for('price_cny', need) == ()


def test_quick_options_only_offer_values_the_catalog_has():
    turn = build(slots(('price_cny', '<=', 3000))).respond('预算3000')
    assert turn.asked_for == 'category'
    domain = set(KB.schema.field('category').domain)
    assert turn.options and set(turn.options) <= domain


def test_clarify_budget_stops_the_questioning():
    orchestrator = build(slots(('category', '==', '项链')), clarify_limit=2)
    need = None
    for _ in range(2):
        turn = orchestrator.respond('想买条项链', need)
        assert turn.action == 'ask_clarify'
        need = turn.need
    # 第三轮不再追问，改为按已知条件先给结果（T-011 的 explore_breadth）。
    final = orchestrator.respond('想买条项链', need)
    assert final.action != 'ask_clarify' and final.total > 0
    assert final.need.strategy == 'explore_breadth'


def test_a_complete_need_skips_clarification():
    turn = build(slots(('category', '==', '项链'), ('price_cny', '<=', 20000))).respond('项链两万以内')
    assert turn.action == 'search' and turn.card == 'item_list'
    assert turn.total > 0 and turn.need.clarify_rounds_used == 0


# ---- 检索结果 ----

def test_results_carry_the_matched_skus_and_count():
    turn = build(slots(('category', '==', '项链'), ('price_cny', '<=', 20000))).respond('项链')
    assert all(hit.sku_ids and hit.category == '项链' for hit in turn.hits)
    assert turn.total >= len(turn.hits)


def test_actions_and_cards_come_from_the_registry():
    turns = [build(slots(('category', '==', '项链'), ('price_cny', '<=', 20000))).respond('项链'),
             build(slots(('category', '==', '耳钉'), ('price_cny', '<=', 2000))).respond('耳钉2000'),
             build({'intent': 'unknown', 'slots': []}).respond('你好')]
    assert all(turn.action in ACTIONS for turn in turns)
    assert {t.card for t in turns} <= {'item_list', 'clarify', 'no_match', 'partial_answer'}


# ---- 死路 ----

def test_dead_end_reports_the_reason_and_offers_executable_patches():
    turn = build(slots(('category', '==', '耳钉'), ('price_cny', '<=', 2000))).respond('预算2000的耳钉')
    assert turn.action == 'flag_conflict' and turn.card == 'no_match'
    assert '6750' in turn.text                      # 归因用的是目录算出的下界
    assert turn.relaxations and turn.options
    assert turn.relaxations[0].kind == 'widen'


def test_the_offered_patch_actually_recovers_results():
    orchestrator = build(slots(('category', '==', '耳钉'), ('price_cny', '<=', 2000)))
    turn = orchestrator.respond('预算2000的耳钉')
    revised = orchestrator.apply(turn, 0)
    found = orchestrator.retriever.run(revised, KB.schema, text='耳钉')
    assert found.total == turn.relaxations[0].result_count > 0


def test_conflict_is_distinguished_from_a_plain_empty_result():
    conflicting = build(slots(('category', '==', '耳钉'), ('price_cny', '<=', 2000))) \
        .respond('预算2000的耳钉')
    assert conflicting.action == 'flag_conflict' and conflicting.conflicts
    # 没有价格下界冲突、只是组合太窄的，走普通兜底。
    plain = build(slots(('category', '==', '项链'), ('price_cny', '<=', 20000),
                        ('color_keys', '==', '橙'))).respond('橙色项链')
    assert plain.total == 0 and plain.action == 'fallback' and plain.conflicts == ()


# ---- 接地校验 ----

def test_a_writer_that_invents_a_price_gets_the_sentence_cut():
    invented = lambda utterance, hits, need: '这条很衬你。另一条只要 7 元。'
    orchestrator = build(slots(('category', '==', '项链'), ('price_cny', '<=', 20000)))
    orchestrator.writer = invented
    turn = orchestrator.respond('项链两万以内')
    assert turn.grounding is not None and not turn.grounding.ok
    assert [c.text for c in turn.grounding.unsupported] == ['7']
    assert '7 元' not in turn.text and '这条很衬你' in turn.text


def test_a_writer_quoting_a_real_price_passes_through():
    orchestrator = build(slots(('category', '==', '项链'), ('price_cny', '<=', 20000)))
    probe = build(slots(('category', '==', '项链'), ('price_cny', '<=', 20000))).respond('项链')
    price = probe.hits[0].price_cny
    orchestrator.writer = lambda u, h, n: f'最便宜的一条 {price:g} 元。'
    turn = orchestrator.respond('项链两万以内')
    assert turn.grounding.ok and f'{price:g}' in turn.text


def test_a_fully_unsupported_draft_falls_back_instead_of_going_blank():
    orchestrator = build(slots(('category', '==', '项链'), ('price_cny', '<=', 20000)))
    orchestrator.writer = lambda u, h, n: '有一款 3 元的，还有一款 5 元的。'
    turn = orchestrator.respond('项链两万以内')
    # 整段被剔光时退回确定性文案，而不是把空串发给用户。
    assert turn.text and not turn.grounding.ok
    assert '3 元' not in turn.text and '5 元' not in turn.text


def test_without_a_writer_the_text_is_deterministic_and_unchecked():
    turn = build(slots(('category', '==', '项链'), ('price_cny', '<=', 20000))).respond('项链')
    assert turn.grounding is None and turn.text


# ---- 审计 ----

def test_rules_and_rejected_slots_are_reported_on_every_turn():
    turn = build({'intent': 'discover', 'slots': [
        {'field': 'cushion_level', 'op': '>=', 'value': 7}]}).respond('送女朋友的生日礼物')
    assert any(entry.get('rule') == 'SR-G01' for entry in turn.trace)
    assert turn.rejected and turn.rejected[0]['field'] == 'cushion_level'
