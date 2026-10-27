"""Score intent classification, slot extraction and knowledge-base coverage."""
import argparse
import json
import os
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from product_data import ROOT, dump

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'fixtures/eval'))
from backend.knowledge import load_default            # noqa: E402
from backend.need import UserNeed                     # noqa: E402
from backend.orchestrator import CLARIFY_LIMIT, Orchestrator  # noqa: E402
from backend.retrieval import Catalog                 # noqa: E402
from backend.understanding import Understanding       # noqa: E402
from understanding import CASES, validate             # noqa: E402


def prior(case, schema):
    if not case.before:
        return None
    return UserNeed(intent='discover', constraints=tuple(
        schema.constrain(name, op, value) for name, op, value in case.before))


def slot_scores(case, need):
    """字段层面算 P/R，取值层面只在字段抽对时才算，两者分开看。"""
    got = {c.field_name: c.value for c in need.constraints}
    expected = set(case.slots)
    extracted = {name for name in got if name in expected or name in case.forbidden}
    true_positive = expected & set(got)
    values_checked = values_right = 0
    for name, want in case.slots.items():
        if want is None or name not in got:
            continue
        values_checked += 1
        actual = got[name]
        if isinstance(actual, tuple):
            values_right += want in actual or want == actual
        else:
            values_right += str(actual) == str(want) or actual == want
    leaked = [name for name in case.forbidden if name in got]
    return {
        'expected': len(expected), 'found': len(true_positive),
        'spurious': len(leaked), 'leaked': leaked,
        'values_checked': values_checked, 'values_right': values_right,
        'missing': sorted(expected - set(got)),
    }


def run(orchestrator, cases):
    schema = orchestrator.schema
    rows = []
    for case in cases:
        turn = orchestrator.respond(case.text, prior(case, schema))
        fired = {entry['rule'] for entry in turn.trace if 'rule' in entry}
        unknown = turn.action == 'handoff'
        # 越界时 need.intent 按设计保留前一轮的业务意图，所以要单独判。
        actual = 'unknown' if unknown else turn.need.intent
        rows.append({
            'id': case.id, 'group': case.group, 'text': case.text,
            'expected_intent': case.intent, 'actual_intent': actual,
            'unknown': unknown, 'intent_ok': actual == case.intent,
            'clarify_expected': case.clarify, 'clarify_actual': turn.action == 'ask_clarify',
            'rules_expected': sorted(case.rules), 'rules_fired': sorted(fired),
            'rules_ok': set(case.rules) <= fired,
            'slots': slot_scores(case, turn.need),
            'action': turn.action, 'note': case.note,
        })
    return rows


def summarise(rows):
    business = [r for r in rows if r['expected_intent'] != 'unknown']
    out_of_scope = [r for r in rows if r['expected_intent'] == 'unknown']
    with_rules = [r for r in rows if r['rules_expected']]
    expected = sum(r['slots']['expected'] for r in rows)
    found = sum(r['slots']['found'] for r in rows)
    spurious = sum(r['slots']['spurious'] for r in rows)
    checked = sum(r['slots']['values_checked'] for r in rows)
    right = sum(r['slots']['values_right'] for r in rows)
    clarify_ok = [r for r in rows if r['clarify_expected'] == r['clarify_actual']]
    return {
        'cases': len(rows),
        'intent_accuracy': round(statistics.mean(r['intent_ok'] for r in rows), 3),
        'unknown_recall': round(statistics.mean(r['unknown'] for r in out_of_scope), 3)
        if out_of_scope else None,
        'unknown_false_positive': round(statistics.mean(r['unknown'] for r in business), 3)
        if business else None,
        'slot_recall': round(found / expected, 3) if expected else None,
        'slot_precision': round(found / (found + spurious), 3) if found + spurious else None,
        'slot_value_accuracy': round(right / checked, 3) if checked else None,
        'kb_rule_recall': round(statistics.mean(r['rules_ok'] for r in with_rules), 3)
        if with_rules else None,
        'clarify_accuracy': round(len(clarify_ok) / len(rows), 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'fixtures/eval/understanding-report.json')
    parser.add_argument('--only', help='只跑某一组，例如 "C 送礼场景"')
    parser.add_argument('--today', default=date.today().isoformat())
    args = parser.parse_args()

    knowledge = load_default(ROOT)
    validate(knowledge.schema)          # 标注漂移在这里就被拦住
    catalog = Catalog.load(ROOT / 'fixtures/catalog/products.jsonl')
    orchestrator = Orchestrator(Understanding(knowledge), catalog)

    cases = [c for c in CASES if not args.only or c.group == args.only]
    rows = run(orchestrator, cases)
    summary = summarise(rows)

    dump(args.output, {'generated_at': args.today, 'model': orchestrator.understanding.model,
                       'clarify_limit': CLARIFY_LIMIT, 'summary': summary, 'rows': rows})

    print(f"{len(rows)} 条标注 · 模型 {orchestrator.understanding.model}\n")
    labels = [('intent_accuracy', '意图正确率'), ('unknown_recall', 'unknown 召回'),
              ('unknown_false_positive', 'unknown 误判率'), ('slot_recall', '槽位召回'),
              ('slot_precision', '槽位精度'), ('slot_value_accuracy', '槽位取值正确率'),
              ('kb_rule_recall', '知识库触发召回'), ('clarify_accuracy', '反问决策正确率')]
    for key, label in labels:
        value = summary[key]
        print(f'  {label:<18}{"—" if value is None else f"{value:.3f}"}')

    print(f"\n{'分组':<14}{'条数':>5}{'意图':>7}{'槽位R':>8}{'规则':>7}{'反问':>7}")
    for group in sorted({r['group'] for r in rows}):
        part = [r for r in rows if r['group'] == group]
        got = sum(r['slots']['found'] for r in part)
        want = sum(r['slots']['expected'] for r in part)
        ruled = [r for r in part if r['rules_expected']]
        print(f"{group:<14}{len(part):>5}"
              f"{statistics.mean(r['intent_ok'] for r in part):>7.2f}"
              f"{(got / want if want else 1):>8.2f}"
              f"{(statistics.mean(r['rules_ok'] for r in ruled) if ruled else 1):>7.2f}"
              f"{statistics.mean(r['clarify_expected'] == r['clarify_actual'] for r in part):>7.2f}")

    failures = [r for r in rows if not (r['intent_ok'] and r['rules_ok']
                                        and not r['slots']['missing']
                                        and not r['slots']['leaked']
                                        and r['clarify_expected'] == r['clarify_actual'])]
    print(f"\n未完全通过 {len(failures)}/{len(rows)}：")
    for row in failures:
        problems = []
        if not row['intent_ok']:
            problems.append(f"意图 {row['expected_intent']}→{row['actual_intent']}")
        if not row['rules_ok']:
            problems.append('规则未触发 ' + ','.join(
                sorted(set(row['rules_expected']) - set(row['rules_fired']))))
        if row['slots']['missing']:
            problems.append('漏槽位 ' + ','.join(row['slots']['missing']))
        if row['slots']['leaked']:
            problems.append('多槽位 ' + ','.join(row['slots']['leaked']))
        if row['clarify_expected'] != row['clarify_actual']:
            problems.append('该反问' if row['clarify_expected'] else '不该反问')
        print(f"  {row['id']}  {row['text'][:22]:<24}{' · '.join(problems)}")

    print(f"\n报告：{args.output.relative_to(ROOT)}")


if __name__ == '__main__':
    main()
