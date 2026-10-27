"""Natural language -> structured constraints."""
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from backend import intents
from backend.catalog_schema import SlotError
from backend.knowledge import Conflict, KnowledgeBase
from backend.need import Constraint, Entity, UserNeed

FENCE = re.compile(r'```(?:json)?\s*(.*?)\s*```', re.S)
# enums with fewer values than this get listed in full in the prompt,
# so the model can actually see what it is allowed to pick
ENUM_PROMPT_LIMIT = 40

# in practice the model writes '=' or chinese words like 包含 instead of the
# operators we asked for. normalising spelling is fine. normalising *meaning*
# is not - anything we cannot map still gets rejected.
OP_ALIASES = {'=': '==', '===': '==', '≤': '<=', '≥': '>=', '=<': '<=', '=>': '>=',
              '等于': '==', '小于等于': '<=', '大于等于': '>=', '不等于': '!=',
              '包含': 'CONTAINS', '含': 'CONTAINS', 'contains': 'CONTAINS',
              'in': 'IN', '属于': 'IN', 'between': 'BETWEEN', '介于': 'BETWEEN'}

PROMPT = """你把用户的购物话语转成结构化约束。只输出一个 JSON 对象，不要解释。

格式：
{{"intent": "<意图>", "slots": [{{"field": "<字段>", "op": "<操作符>", "value": <值>}}]}}

意图取值：{intents}
  discover=模糊选品  spec_query=规格查询  compatibility=兼容判断
  compare=多商品比较  policy=服务政策  unknown=与购物无关或超出目录范围

可用字段（只能用这些，其它一律不要输出）：
{fields}

规则：
- 没有明确说出的条件不要填，宁可留空让系统追问。
- 「便宜点」「贵一些」这类相对表述不要自行折算，交给系统处理。
- 与购物无关的话语一律 intent=unknown，slots 为空。"""


@dataclass(frozen=True)
class Interpretation:
    intent: str
    need: UserNeed
    missing_core: tuple[str, ...]
    conflicts: tuple[Conflict, ...] = ()
    rejected: tuple[dict, ...] = ()
    trace: tuple[dict, ...] = ()
    error: str | None = None

    @property
    def unknown(self) -> bool:
        return self.intent == 'unknown'

    @property
    def ready(self) -> bool:
        """核心槽位齐备且无冲突才可以进检索。"""
        return not self.unknown and not self.missing_core and not self.conflicts


class Understanding:
    def __init__(self, kb: KnowledgeBase, model=None, base_url=None, api_key=None,
                 client_factory=None):
        self.kb = kb
        self.model = model or os.getenv('OPENAI_MODEL') or 'gpt-4.1-mini'
        self.base_url = base_url or os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1'
        self._api_key = api_key or os.getenv('OPENAI_API_KEY')
        self._client_factory = client_factory

    # ---- prompt building ----

    def field_catalog(self) -> str:
        lines = []
        for name in self.kb.schema.filterable():
            spec = self.kb.schema.fields[name]
            if spec.tier != 'primary' or spec.superseded_by:
                # keep low-coverage fields out of the prompt, otherwise the model
                # starts filling them in. same for fields a derived one replaced -
                # showing both makes it flip between two vocabularies.
                continue
            if spec.kind == 'range':
                lines.append(f'- {name}（数值，{spec.minimum:.0f}~{spec.maximum:.0f}）'
                             f' 操作符 <= >= BETWEEN')
            elif spec.kind == 'enum' and len(spec.domain) <= ENUM_PROMPT_LIMIT:
                # small enum -> list every value. !!! truncating is a trap:
                # with 27 categories and only 10 shown, the model literally could
                # not produce "bluetooth speaker" because it never saw it exists
                lines.append(f'- {name}（枚举，只能用这些）：' + '、'.join(spec.domain))
            else:
                sample = '、'.join(spec.known_values[:12])
                lines.append(f'- {name}（{spec.kind}，取值不止这些）举例：{sample}')
        return '\n'.join(lines)

    def prompt(self) -> str:
        return PROMPT.format(intents=' / '.join(intents.INTENTS), fields=self.field_catalog())

    # ---- calling the model ----

    def _client(self):
        if self._client_factory is not None:
            return self._client_factory()
        from openai import OpenAI
        if not self._api_key:
            raise ValueError('OPENAI_API_KEY is required for understanding')
        return OpenAI(base_url=self.base_url, api_key=self._api_key, timeout=30, max_retries=1)

    def _propose(self, utterance: str, need: UserNeed) -> tuple[dict, str | None]:
        messages = [{'role': 'system', 'content': self.prompt()}]
        known = summarize(need)
        if known:
            # short follow-ups like "actually make it diamond" get classified
            # as a brand new intent if the model cannot see what came before
            messages.append({'role': 'system', 'content':
                             f'当前会话已确认：{known}\n当前意图：{need.intent}。'
                             f'跟进句若只是补充或修改条件，意图应保持不变。'})
        messages.append({'role': 'user', 'content': utterance})
        with self._client() as client:
            completion = client.chat.completions.create(model=self.model, messages=messages)
        return parse(completion.choices[0].message.content or '')

    # ---- main flow ----

    def interpret(self, utterance: str, need: UserNeed | None = None) -> Interpretation:
        need = need or UserNeed(intent='discover')
        proposal, error = self._propose(utterance, need)
        intent = proposal.get('intent') if proposal else None
        if intent not in intents.INTENTS:
            # unreadable -> unknown. do not guess a business intent and carry on
            intent = 'unknown'
            error = error or f'unusable intent: {proposal.get("intent")!r}'

        derived, trace = self.kb.derive(utterance)
        proposed, rejected = self._validate(proposal.get('slots') or [] if proposal else [])
        anchor = _anchor(need)
        relative, vague_trace = self.kb.resolve_vague(utterance, anchor)
        if vague_trace:
            trace = trace + (vague_trace,)

        changes = _merge(derived, proposed, relative)
        # an out-of-scope line changes neither the constraints nor the session
        # intent. asking "what day is it" does not mean you stopped shopping
        carried = need.intent if intent == 'unknown' else intent
        if intent == 'unknown':
            changes = ()
        updated = need
        if changes or need.intent != carried:
            updated, _ = need.rewrite(changes)
            updated = type(need)(intent=carried, constraints=updated.constraints,
                                 entities=updated.entities,
                                 clarify_rounds_used=updated.clarify_rounds_used,
                                 strategy=updated.strategy)

        filled = {c.field_name for c in updated.constraints}
        missing = intents.missing_core(intent, filled, len(updated.entities))
        return Interpretation(intent=intent, need=updated, missing_core=missing,
                              conflicts=self.kb.conflicts(updated), rejected=tuple(rejected),
                              trace=trace, error=error)

    def _validate(self, slots: Any) -> tuple[list[Constraint], list[dict]]:
        accepted, rejected = [], []
        if not isinstance(slots, list):
            return accepted, [{'reason': f'slots must be a list, got {type(slots).__name__}'}]
        for raw in slots:
            if not isinstance(raw, dict):
                rejected.append({'slot': raw, 'reason': 'not an object'})
                continue
            name, op, value = raw.get('field'), raw.get('op', '=='), raw.get('value')
            op = OP_ALIASES.get(str(op).strip(), str(op).strip())
            try:
                accepted.append(self.kb.schema.constrain(str(name), str(op), value))
            except (SlotError, ValueError) as problem:
                # failed validation = slot not filled. let the caller ask.
                rejected.append({'field': name, 'op': op, 'value': value, 'reason': str(problem)})
        return accepted, rejected


def parse(text: str) -> tuple[dict, str | None]:
    """parse json out of a model reply.

    tolerates ```json fences and chatter around the object. returns an error
    string instead of raising - a bad reply is a normal thing that happens,
    not an exception.
    """
    body = text.strip()
    fenced = FENCE.search(body)
    if fenced:
        body = fenced.group(1)
    start, end = body.find('{'), body.rfind('}')
    if start < 0 or end <= start:
        return {}, 'no JSON object in reply'
    try:
        # we slice from first { to last }, so if it parses it is an object
        return json.loads(body[start:end + 1]), None
    except json.JSONDecodeError as problem:
        return {}, f'malformed JSON: {problem.msg}'


def summarize(need: UserNeed) -> str:
    """squash what we already know into one line, so the model can tell
    whether this turn is a follow-up or a fresh request."""
    return '；'.join(f'{c.field_name} {c.op} {c.value}' for c in need.constraints)


def _anchor(need: UserNeed) -> float | None:
    """anchor for relative phrases = whatever budget we already have."""
    budget = need.of('price_cny')
    if budget is None:
        return None
    value = budget.value
    return float(value[1] if isinstance(value, tuple) else value)


def _merge(*groups) -> tuple[Constraint, ...]:
    """later wins on the same field.

    order is kb -> model -> relative phrase, roughly least specific to most.
    """
    merged: dict[str, Constraint] = {}
    for group in groups:
        for constraint in (group if isinstance(group, (list, tuple)) else [group]):
            if constraint is not None:
                merged[constraint.field_name] = constraint
    return tuple(merged.values())
