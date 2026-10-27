import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'fixtures/eval'))
sys.path.insert(0, str(ROOT / 'scripts'))

from backend.catalog_schema import SlotError  # noqa: E402
from backend.intents import INTENTS           # noqa: E402
from backend.knowledge import load_default    # noqa: E402
from understanding import CASES, Case, validate  # noqa: E402

KB = load_default(ROOT)


def test_labels_pass_the_catalog_schema():
    """标注里的字段和取值必须是目录答得了的——标注漂移在这里就被拦住。"""
    validate(KB.schema)


def test_a_drifted_label_is_caught(monkeypatch):
    import understanding as labels
    bad = Case('X-01', 'x', 'discover', 'X', {'cushion_level': 7})
    monkeypatch.setattr(labels, 'CASES', (bad,))
    with pytest.raises(SlotError, match='unknown field'):
        labels.validate(KB.schema)
    worse = Case('X-02', 'x', 'discover', 'X', {'category': '智能马桶'})
    monkeypatch.setattr(labels, 'CASES', (worse,))
    with pytest.raises(SlotError, match='is not a value of'):
        labels.validate(KB.schema)


def test_every_case_has_an_id_and_a_known_intent():
    ids = [case.id for case in CASES]
    assert len(ids) == len(set(ids))
    assert all(case.intent in INTENTS for case in CASES)
    assert all(case.text.strip() for case in CASES)


def test_expected_rules_exist_in_the_knowledge_base():
    known = {rule.id for rule in KB.rules}
    referenced = {rule for case in CASES for rule in case.rules}
    assert referenced <= known, referenced - known


def test_the_set_covers_every_labelled_group_and_both_outcomes():
    groups = Counter(case.group for case in CASES)
    assert len(groups) >= 8 and min(groups.values()) >= 3
    assert sum(case.clarify for case in CASES) >= 10          # 该反问的
    assert sum(not case.clarify for case in CASES) >= 10      # 不该反问的
    assert sum(case.intent == 'unknown' for case in CASES) >= 5


def test_known_gaps_are_kept_in_the_set_not_removed():
    """已知会失败的说法必须留着。评测集的用处是暴露问题，不是掩盖。"""
    gaps = [case for case in CASES if case.rules and case.group.startswith('C')]
    assert len(gaps) >= 5
    missed = [case for case in gaps if not KB.match(case.text)]
    # 这些口语送礼说法当前触发不了任何规则；补了触发词后这条断言会失败，
    # 那时把数字改小即可——它是一个提醒，不是一个目标。
    assert len(missed) >= 4, '触发词覆盖变好了，记得更新这条断言'


def test_cases_with_prior_context_declare_valid_constraints():
    for case in CASES:
        for name, op, value in case.before:
            KB.schema.validate(name, op, value)
