"""Gate between what the model proposes and what the catalog can answer."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.need import Constraint

OPS_FOR_KIND = {
    'enum': frozenset({'==', '!=', 'IN', 'ALL', 'CONTAINS'}),
    'text': frozenset({'==', 'ALL', 'CONTAINS'}),
    'range': frozenset({'<=', '>=', '<', '>', '==', 'BETWEEN'}),
}


class SlotError(ValueError):
    """the proposed constraint does not match the catalog.

    callers should treat this as "slot not filled" and ask, NOT as something
    to work around.
    """


@dataclass(frozen=True)
class FieldSpec:
    name: str
    level: str
    kind: str
    coverage: int
    coverage_ratio: float
    tier: str
    filterable: bool
    hard_filter_safe: bool
    domain: tuple[str, ...] = ()
    values: tuple[str, ...] = ()        # all distinct values, used to check CONTAINS
    minimum: float | None = None
    maximum: float | None = None
    superseded_by: str | None = None    # replaced by a derived field.
                                        # still filterable, but kept out of the
                                        # prompt so the model does not flip
                                        # between two vocabularies

    @property
    def known_values(self) -> tuple[str, ...]:
        return self.domain or self.values

    @property
    def constraint_kind(self) -> str:
        """high enough coverage -> hard filter. otherwise demote to ranking only."""
        return 'hard' if self.hard_filter_safe else 'soft'


class CatalogSchema:
    def __init__(self, fields: dict[str, FieldSpec], products: int, skus: int):
        self.fields = fields
        self.products = products
        self.skus = skus

    @classmethod
    def load(cls, path: Path | str) -> 'CatalogSchema':
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
        fields = {}
        for name, entry in raw['fields'].items():
            fields[name] = FieldSpec(
                name=name, level=entry['level'], kind=entry['kind'],
                coverage=entry['coverage'], coverage_ratio=entry['coverage_ratio'],
                tier=entry.get('tier', 'rare'), filterable=entry['filterable'],
                hard_filter_safe=entry.get('hard_filter_safe', False),
                domain=tuple(entry.get('domain', ())),
                values=tuple(entry.get('values', ())),
                minimum=entry.get('min'), maximum=entry.get('max'),
                superseded_by=entry.get('superseded_by'))
        return cls(fields, raw['products'], raw['skus'])

    # ---- lookups ----

    def field(self, name: str) -> FieldSpec | None:
        return self.fields.get(name)

    def filterable(self) -> tuple[str, ...]:
        return tuple(n for n, f in sorted(self.fields.items()) if f.filterable)

    def hard_safe(self) -> tuple[str, ...]:
        return tuple(n for n, f in sorted(self.fields.items()) if f.hard_filter_safe)

    def excluded_by(self, name: str) -> int:
        """how many products a hard filter on this field would silently drop.

        those are the ones that simply do not declare the field. not the same
        thing as "did not match", and the caller needs to be able to tell the
        difference.
        """
        spec = self.fields.get(name)
        if spec is None:
            raise SlotError(f'unknown field: {name!r}')
        total = self.skus if spec.level in ('sku', 'sku_property') else self.products
        return total - spec.coverage

    # ---- validation ----

    def validate(self, name: str, op: str, value: Any) -> Any:
        """return the normalised value, or raise. never guess."""
        spec = self.fields.get(name)
        if spec is None:
            raise SlotError(f'unknown field: {name!r}')
        if not spec.filterable:
            raise SlotError(f'{name!r} is {spec.kind}; it cannot be filtered on')
        if op not in OPS_FOR_KIND[spec.kind]:
            raise SlotError(f'{op!r} is not valid for a {spec.kind} field; '
                            f'use one of {sorted(OPS_FOR_KIND[spec.kind])}')
        if spec.kind == 'range':
            return self._range_value(spec, value, op)
        if isinstance(value, (list, tuple)):
            if op not in ('IN', 'ALL', 'CONTAINS'):
                raise SlotError(f'{op!r} takes a single value, not a list')
            return tuple(self._single(spec, item, op) for item in value) or _empty(name)
        return self._single(spec, value, op)

    @classmethod
    def _range_value(cls, spec: FieldSpec, value: Any, op: str = '<=') -> Any:
        if op == 'BETWEEN':
            # BETWEEN naturally takes two numbers. a single one means nothing here
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise SlotError(f'{spec.name} BETWEEN expects two numbers, got {value!r}')
            low, high = (cls._number(spec, item) for item in value)
            if low > high:
                raise SlotError(f'{spec.name} BETWEEN lower bound {low} exceeds upper {high}')
            return (low, high)
        return cls._number(spec, value)

    @staticmethod
    def _number(spec: FieldSpec, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SlotError(f'{spec.name} expects a number, got {value!r}')
        if value <= 0:
            raise SlotError(f'{spec.name} must be positive, got {value!r}')
        return float(value)

    @staticmethod
    def _single(spec: FieldSpec, value: Any, op: str = '==') -> str:
        text = str(value).strip()
        if not text:
            raise SlotError(f'{spec.name} must not be empty')
        known = spec.known_values
        if op == 'ALL' and spec.kind == 'enum' and text not in spec.domain:
            raise SlotError(f'{text!r} is not a value of {spec.name}; '
                            f'known values: {", ".join(spec.domain[:8])}')
        if op == 'CONTAINS':
            # the substring has to actually hit something in the catalog.
            # otherwise this constraint is guaranteed to return an empty set
            # and the user just sees "no results" with no explanation
            if known and not any(text in value for value in known):
                raise SlotError(f'no value of {spec.name} contains {text!r}; '
                                f'known values: {", ".join(known[:8])}')
            return text
        if spec.kind == 'enum' and text not in spec.domain:
            raise SlotError(f'{text!r} is not a value of {spec.name}; '
                            f'known values: {", ".join(spec.domain[:8])}')
        return text

    def constrain(self, name: str, op: str, value: Any, **extra) -> Constraint:
        """build a constraint after validation.

        note the caller does not get to pick `kind` - coverage decides it.
        (you can still override explicitly, the KB does that for derived ones)
        """
        checked = self.validate(name, op, value)
        spec = self.fields[name]
        kind = extra.pop('kind', None) or spec.constraint_kind
        return Constraint(field_name=name, op=op, value=checked, kind=kind, **extra)


def _empty(name):
    raise SlotError(f'{name} must not be an empty list')
