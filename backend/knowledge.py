"""Category knowledge base. Scene -> constraints, the only source of derived ones."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.catalog_schema import CatalogSchema, SlotError
from backend.need import Constraint, UserNeed


@dataclass(frozen=True)
class Rule:
    id: str
    scene: str
    triggers: tuple[str, ...]
    state: tuple[tuple[str, str], ...]
    derive: tuple[dict, ...]

    def matches(self, utterance: str) -> str | None:
        """return which trigger phrase matched.

        plain substring, no model involved. less clever but you can always
        explain afterwards why a rule fired.
        """
        return next((t for t in self.triggers if t in utterance), None)


@dataclass(frozen=True)
class VagueTerm:
    id: str
    phrases: tuple[str, ...]
    field: str
    op: str
    anchor_ratio: Any

    def matches(self, utterance: str) -> str | None:
        return next((p for p in self.phrases if p in utterance), None)

    def resolve(self, anchor: float) -> Any:
        if isinstance(self.anchor_ratio, list):
            low, high = self.anchor_ratio
            return (round(anchor * low, 2), round(anchor * high, 2))
        return round(anchor * self.anchor_ratio, 2)


def _combine(new: Constraint, old: Constraint) -> Constraint:
    """merge two constraints that landed on the same field.

    hierarchy wins by depth - 'silver/925' beats plain 'silver'.
    if neither is a prefix of the other they are not competing at all
    (silver and diamond are both materials), so take the union with ALL.
    """
    fresh = tuple(new.value) if isinstance(new.value, tuple) else (new.value,)
    kept = tuple(old.value) if isinstance(old.value, tuple) else (old.value,)
    if len(fresh) == 1 and len(kept) == 1:
        a, b = str(fresh[0]), str(kept[0])
        if a.startswith(b + '/'):
            return new
        if b.startswith(a + '/'):
            return old
    merged = kept + tuple(v for v in fresh if v not in kept)
    if len(merged) == 1:
        return old
    return Constraint(field_name=new.field_name, op='ALL', value=merged,
                      kind=new.kind, derived_from=new.derived_from,
                      evidence=f'{old.evidence or ""}；{new.evidence or ""}'.strip('；'))


@dataclass(frozen=True)
class Conflict:
    field: str
    reason: str
    floor: float
    requested: float


class KnowledgeBase:
    def __init__(self, rules, vague, state_fields, schema: CatalogSchema, bounds: dict):
        self.rules = rules
        self.vague = vague
        self.state_fields = state_fields
        self.schema = schema
        self.bounds = bounds

    @classmethod
    def load(cls, path: Path | str, schema: CatalogSchema, bounds: dict) -> 'KnowledgeBase':
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
        state_fields = raw['state_fields']
        for name in state_fields:
            if schema.field(name) is not None:
                raise ValueError(f'state field {name!r} collides with a catalog field')
        rules = []
        for entry in raw['rules']:
            for declared in entry['state']:
                if declared not in state_fields:
                    raise ValueError(f'{entry["id"]}: undeclared state field {declared!r}')
            for target in entry['derive']:
                # catch bad fields here, at load time.
                # !!! waiting until production returns an empty set is too late
                schema.validate(target['field'], target['op'], target['value'])
                if not target.get('evidence'):
                    raise ValueError(f'{entry["id"]}: derived constraint needs evidence')
            rules.append(Rule(id=entry['id'], scene=entry['scene'],
                              triggers=tuple(entry['triggers']),
                              state=tuple(entry['state'].items()),
                              derive=tuple(entry['derive'])))
        vague = tuple(VagueTerm(id=t['id'], phrases=tuple(t['phrases']), field=t['field'],
                                op=t['op'], anchor_ratio=t['anchor_ratio'])
                      for t in raw['vague_terms']['terms'])
        return cls(tuple(rules), vague, state_fields, schema, bounds)

    # ---- scene rules ----

    def match(self, utterance: str) -> tuple[tuple[Rule, str], ...]:
        hits = ((rule, rule.matches(utterance)) for rule in self.rules)
        return tuple((rule, phrase) for rule, phrase in hits if phrase)

    def derive(self, utterance: str) -> tuple[tuple[Constraint, ...], tuple[dict, ...]]:
        """returns (constraints, trace).

        state fields land first, then the derived constraints hang off them
        via derived_from. that is what makes a scene change kill them later.
        """
        constraints, trace = {}, []
        for rule, phrase in self.match(utterance):
            sources = []
            for name, value in rule.state:
                constraints[name] = Constraint(field_name=name, op='==', value=value, kind='hard')
                sources.append(name)
            for target in rule.derive:
                checked = self.schema.validate(target['field'], target['op'], target['value'])
                spec = self.schema.fields[target['field']]
                # if the rule set a state field, mark the result derived so it
                # dies when the scene changes. things the user said directly
                # (material, colour) have no state source, so coverage decides.
                if sources:
                    constraint = Constraint(
                        field_name=target['field'], op=target['op'], value=checked,
                        kind='derived', derived_from=tuple(sources), evidence=target['evidence'])
                else:
                    constraint = Constraint(
                        field_name=target['field'], op=target['op'], value=checked,
                        kind=spec.constraint_kind, evidence=target['evidence'])
                existing = constraints.get(target['field'])
                if existing is not None:
                    # triggers nest by nature - "want 925 silver" contains
                    # "silver", so both rules fire. merging must not depend on
                    # which rule happens to be written first in the json.
                    constraint = _combine(constraint, existing)
                constraints[target['field']] = constraint
                trace.append({'rule': rule.id, 'scene': rule.scene, 'trigger': phrase,
                              'field': target['field'], 'evidence': target['evidence'],
                              'kind': constraint.kind})
        return tuple(constraints.values()), tuple(trace)

    # ---- relative phrases ----

    def resolve_vague(self, utterance: str, anchor: float | None) -> tuple[Constraint | None, dict | None]:
        """"cheaper" and friends need an anchor to mean anything.

        no anchor -> return None and let the caller ask. we do not invent a
        number out of thin air.
        """
        hit = next(((term, term.matches(utterance)) for term in self.vague
                    if term.matches(utterance)), None)
        if hit is None:
            return None, None
        term, phrase = hit
        if anchor is None:
            return None, {'term': term.id, 'trigger': phrase, 'needs_anchor': True}
        value = term.resolve(anchor)
        shown = int(anchor) if float(anchor).is_integer() else anchor
        constraint = Constraint(field_name=term.field, op=term.op, value=value, kind='hard',
                                anchor_source=f'{phrase} @ {shown}')
        return constraint, {'term': term.id, 'trigger': phrase, 'anchor': anchor, 'value': value}

    # ---- conflict detection ----

    def conflicts(self, need: UserNeed) -> tuple[Conflict, ...]:
        """is this budget even buyable, judged from catalog price floors.

        the numbers come from scanning the catalog, not from the model having
        an opinion about what jewellery costs.
        """
        budget = need.of('price_cny')
        if budget is None or budget.op not in ('<=', '<'):
            return ()
        ceiling = float(budget.value)
        found = []
        material = need.of('材质')
        if material is not None:
            span = self.bounds['price_cny_by_material'].get(str(material.value))
            if span and span['min'] > ceiling:
                found.append(Conflict('材质', f'目录中含「{material.value}」的商品最低 {span["min"]:.0f} 元',
                                      span['min'], ceiling))
        category = need.of('category')
        if category is not None and not isinstance(category.value, tuple):
            span = self.bounds['price_cny_by_category'].get(str(category.value))
            if span and span['min'] > ceiling:
                found.append(Conflict('category', f'「{category.value}」类目最低 {span["min"]:.0f} 元',
                                      span['min'], ceiling))
        return tuple(found)


def load_default(root: Path) -> KnowledgeBase:
    schema_path = root / 'fixtures/catalog/field-schema.json'
    schema = CatalogSchema.load(schema_path)
    bounds = json.loads(schema_path.read_text(encoding='utf-8'))['bounds']
    return KnowledgeBase.load(root / 'fixtures/kb/scene-rules.json', schema, bounds)
