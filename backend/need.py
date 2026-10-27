"""Session need: a constraint graph with dependency edges."""
from dataclasses import dataclass, replace
from typing import Any, Literal

ConstraintKind = Literal['hard', 'soft', 'derived']
EntityRole = Literal['target', 'reference', 'candidate']
Confidence = Literal['high', 'medium', 'low']
Strategy = Literal['narrow_down', 'explore_breadth']

# ALL is for multi-valued fields. material_keys ALL ('silver','gem/diamond')
# means the product needs BOTH of them, not either one.
# I added this later beacuse "silver one with diamonds" was silently losing
# the diamond part - the two values were fighting over the same slot, but
# they are not competing values, they just both happen to be materials.
OPS = frozenset({'==', '!=', '<=', '>=', '<', '>', 'IN', 'ALL', 'BETWEEN', 'CONTAINS'})

# every action the dialogue state machine can take (from my contract notes)
ACTIONS = frozenset({
    'ask_clarify', 'search', 'fetch_spec', 'verify', 'compare', 'fetch_policy',
    'fallback', 'rewrite_state', 'flag_conflict', 'correct_misconception',
    'explain', 'handoff',
})


@dataclass(frozen=True)
class Constraint:
    field_name: str
    op: str
    value: Any
    kind: ConstraintKind
    # which fields this constraint was derived from.
    # when one of those source fields changes, this one dies automatically (T-012)
    derived_from: tuple[str, ...] = ()
    # why we derived it. !!! this must be showable to the user.
    # if the derivation is wrong and we never show it, the user has no way
    # of knowing why the results suddenly got narrower
    evidence: str | None = None
    # where a relative phrase anchored, e.g. "cheaper" -> anchor 349
    anchor_source: str | None = None
    relaxed_by_user: bool = False

    def __post_init__(self):
        if not self.field_name:
            raise ValueError('field_name must not be empty')
        if self.op not in OPS:
            raise ValueError(f'unknown op: {self.op!r}')
        if self.kind == 'derived' and not self.derived_from:
            raise ValueError(f'derived constraint {self.field_name!r} must declare derived_from')
        if self.kind == 'derived' and not self.evidence:
            raise ValueError(f'derived constraint {self.field_name!r} must carry evidence')
        if self.field_name in self.derived_from:
            raise ValueError(f'{self.field_name!r} must not derive from itself')

    @property
    def filters(self) -> bool:
        """does this one actually filter.

        derived ones filter too - they are still hard, just produced by a
        rule instead of typed by the user.
        """
        return self.kind in ('hard', 'derived')


@dataclass(frozen=True)
class Entity:
    sku: str
    role: EntityRole
    resolved_from: str          # the raw words the user used, for "that one earlier"
    confidence: Confidence = 'high'

    @property
    def needs_disambiguation(self) -> bool:
        return self.confidence == 'low'


@dataclass(frozen=True)
class Partition:
    """audit record of one rewrite.

    the three lists added together must equal every constraint we had before.
    there is a test for that, otherwise constraints can silently vanish.
    """
    retained: tuple[Constraint, ...]
    discarded: tuple[tuple[Constraint, str], ...]
    rewritten: tuple[tuple[Constraint, Constraint], ...]
    added: tuple[Constraint, ...]

    def as_log(self) -> dict:
        return {
            'retained': [c.field_name for c in self.retained],
            'discarded': [{'field': c.field_name, 'reason': why} for c, why in self.discarded],
            'rewritten': [{'field': old.field_name, 'from': old.value, 'to': new.value}
                          for old, new in self.rewritten],
            'added': [c.field_name for c in self.added],
        }


@dataclass(frozen=True)
class UserNeed:
    intent: str
    constraints: tuple[Constraint, ...] = ()
    entities: tuple[Entity, ...] = ()
    clarify_rounds_used: int = 0
    strategy: Strategy = 'narrow_down'

    def __post_init__(self):
        seen = [c.field_name for c in self.constraints]
        if len(seen) != len(set(seen)):
            raise ValueError('one constraint per field; rewrite instead of appending')

    # ---- reading ----

    def of(self, field_name: str) -> Constraint | None:
        return next((c for c in self.constraints if c.field_name == field_name), None)

    @property
    def category(self) -> Any | None:
        found = self.of('category')
        return found.value if found else None

    @property
    def hard(self) -> tuple[Constraint, ...]:
        return tuple(c for c in self.constraints if c.filters)

    @property
    def soft(self) -> tuple[Constraint, ...]:
        return tuple(c for c in self.constraints if c.kind == 'soft')

    def entity(self, sku: str) -> Entity | None:
        return next((e for e in self.entities if e.sku == sku), None)

    # ---- invalidation ----

    def invalidated_by(self, changed: set[str]) -> set[str]:
        """transitive closure of the changed fields.

        if A depends on B and C depends on A, then touching B has to kill
        both A and C. one level is not enough, took me a while to notice.
        """
        dead = set(changed)
        while True:
            more = {c.field_name for c in self.constraints
                    if c.field_name not in dead and dead.intersection(c.derived_from)}
            if not more:
                return dead
            dead |= more

    def rewrite(self, changes: tuple[Constraint, ...] | list[Constraint], *,
                drop: tuple[str, ...] | list[str] = (),
                relaxation: bool = False) -> tuple['UserNeed', Partition]:
        """the three-way split. every existing constraint ends up in exactly
        one of retained / discarded / rewritten, nothing is left implicit.

        `drop` is for fields the user took back without giving a new value.
        `relaxation=True` says "this turn is a relaxation" - and then the
        number of hard constraints is not allowed to grow. that is the
        regression check from T-012 and it is enforced here, not in a test.
        """
        incoming = {c.field_name: c for c in changes}
        if len(incoming) != len(list(changes)):
            raise ValueError('changes must not repeat a field')
        dead = self.invalidated_by(set(incoming) | set(drop))

        retained, discarded, rewritten = [], [], []
        for current in self.constraints:
            if current.field_name in incoming:
                rewritten.append((current, incoming[current.field_name]))
            elif current.field_name in drop:
                discarded.append((current, 'user withdrew this requirement'))
            elif dead.intersection(current.derived_from):
                source = sorted(dead.intersection(current.derived_from))
                discarded.append((current, f'derived from changed field(s): {", ".join(source)}'))
            else:
                retained.append(current)

        added = [c for name, c in incoming.items() if self.of(name) is None]
        kept = [new for _, new in rewritten] + retained + added
        result = replace(self, constraints=tuple(sorted(kept, key=lambda c: c.field_name)))
        if relaxation and len(result.hard) > len(self.hard):
            raise ValueError('a relaxation must not increase the number of hard constraints')
        partition = Partition(tuple(retained), tuple(discarded), tuple(rewritten), tuple(added))
        return result, partition

    # ---- entities ----

    def recast(self, sku: str, role: EntityRole) -> 'UserNeed':
        """change the role, never delete the entity.

        if we delete it, "the one I looked at first" becomes unresolvable
        later on. entity resolution on that kind of reference almost never
        works, so just keep it around with a different role.
        """
        if self.entity(sku) is None:
            raise KeyError(sku)
        return replace(self, entities=tuple(
            replace(e, role=role) if e.sku == sku else e for e in self.entities))

    def remember(self, entity: Entity) -> 'UserNeed':
        if self.entity(entity.sku) is not None:
            return self.recast(entity.sku, entity.role)
        return replace(self, entities=self.entities + (entity,))

    # ---- (de)serialisation ----

    def as_dict(self) -> dict:
        return {
            'intent': self.intent,
            'clarify_rounds_used': self.clarify_rounds_used,
            'strategy': self.strategy,
            'constraints': [
                {'field_name': c.field_name, 'op': c.op, 'value': list(c.value)
                 if isinstance(c.value, tuple) else c.value, 'kind': c.kind,
                 'derived_from': list(c.derived_from), 'evidence': c.evidence,
                 'anchor_source': c.anchor_source, 'relaxed_by_user': c.relaxed_by_user}
                for c in self.constraints],
            'entities': [{'sku': e.sku, 'role': e.role, 'resolved_from': e.resolved_from,
                          'confidence': e.confidence} for e in self.entities],
        }

    @classmethod
    def from_dict(cls, raw: dict) -> 'UserNeed':
        # !!! json has no tuples. IN / ALL / BETWEEN come back as lists and if
        # we leave them as lists the frozen dataclass stops comparing equal
        # and the hash breaks. spent way too long on this one, so: cast back.
        constraints = tuple(
            Constraint(field_name=c['field_name'], op=c['op'],
                       value=tuple(c['value']) if isinstance(c['value'], list) else c['value'],
                       kind=c['kind'], derived_from=tuple(c.get('derived_from') or ()),
                       evidence=c.get('evidence'), anchor_source=c.get('anchor_source'),
                       relaxed_by_user=bool(c.get('relaxed_by_user')))
            for c in raw.get('constraints') or ())
        entities = tuple(Entity(sku=e['sku'], role=e['role'], resolved_from=e['resolved_from'],
                                confidence=e.get('confidence', 'high'))
                         for e in raw.get('entities') or ())
        return cls(intent=raw['intent'], constraints=constraints, entities=entities,
                   clarify_rounds_used=int(raw.get('clarify_rounds_used', 0)),
                   strategy=raw.get('strategy', 'narrow_down'))

    # ---- clarify budget ----

    def clarified(self, limit: int) -> 'UserNeed':
        """count one clarification round.

        once we hit the limit we switch strategy instead of keeping on
        asking. a system that asks forever is worse than one that guesses
        and shows something.
        """
        if limit < 1:
            raise ValueError('clarify limit must be at least 1')
        used = self.clarify_rounds_used + 1
        return replace(self, clarify_rounds_used=used,
                       strategy='explore_breadth' if used >= limit else self.strategy)
