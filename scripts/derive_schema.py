"""Derive the filterable field schema from the catalog itself."""
import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

from product_data import ROOT, dump

# 枚举判定看两条：取值总数不能太多，且取值数相对覆盖数要低——
# 27 个类目覆盖 121 条是枚举，46 种材质覆盖 79 条就更像自由文本。
ENUM_MAX_VALUES = 40
ENUM_MAX_RATIO = 0.5
IDENTIFIER_RATIO = 0.9      # 取值数/覆盖数 高于此值视为标识符，不可过滤
MIN_COVERAGE = 5            # 覆盖太少的字段做硬过滤没有意义
# 覆盖率分档：硬过滤会淘汰掉没有该字段的商品，覆盖率低的字段用作硬条件
# 等于静默丢掉大半目录，只能降级为排序项。
TIERS = (('primary', 0.5), ('secondary', 0.15), ('rare', 0.0))

# 原始字段 → 取代它的派生字段。被取代的仍可过滤（展示和高级用法要用），
# 但不进模型提示词，否则模型会在两套词表之间摇摆。
SUPERSEDED = {
    '材质': 'material_keys',
    '配色': 'color_keys', '颜色': 'color_keys', '主色': 'color_keys',
    '辅色': 'color_keys', '款式': 'color_keys',
    '品牌': 'brand_key',
    '发售价格': 'price_cny',
}


def values_of(raw):
    return [str(v).strip() for v in (raw if isinstance(raw, list) else [raw]) if str(v).strip()]


def classify(name, counter, coverage, total):
    distinct = len(counter)
    if distinct <= 1:
        return 'constant', f'every product shares one value'
    if coverage >= 20 and distinct / coverage >= IDENTIFIER_RATIO:
        return 'identifier', f'{distinct} values over {coverage} products'
    if coverage < MIN_COVERAGE:
        return 'sparse', f'only {coverage}/{total} products declare it'
    if distinct <= ENUM_MAX_VALUES and distinct / coverage <= ENUM_MAX_RATIO:
        return 'enum', None
    return 'text', None


def spec(name, level, counter, coverage, total):
    kind, reason = classify(name, counter, coverage, total)
    entry = {'level': level, 'kind': kind, 'coverage': coverage,
             'coverage_ratio': round(coverage / total, 3), 'distinct': len(counter)}
    if kind == 'enum':
        entry['domain'] = [value for value, _ in counter.most_common()]
    elif kind == 'text':
        # 高基数字段只能做包含匹配。高频值用于提示，完整取值集用于校验
        # 子串是否真能命中——匹配不到任何值的条件等于静默返回空集。
        entry['frequent'] = [value for value, _ in counter.most_common(12)]
        entry['values'] = [value for value, _ in counter.most_common()]
    else:
        entry['excluded_because'] = reason
    entry['filterable'] = kind in ('enum', 'text')
    ratio = entry['coverage_ratio']
    entry['tier'] = next(name for name, floor in TIERS if ratio >= floor)
    # 只有 primary 档适合做 hard 约束；其余降级为 soft，否则静默淘汰无此字段的商品。
    entry['hard_filter_safe'] = entry['filterable'] and entry['tier'] == 'primary'
    return entry


def derive(products):
    total = len(products)
    fields = {}

    def add(name, level, counter, coverage):
        fields[name] = spec(name, level, counter, coverage, total)

    for name, getter in (('category', lambda p: p['category']),
                         ('brand_key', lambda p: p['brand_key']),
                         ('brand_display', lambda p: p['brand_display']),
                         ('color_keys', lambda p: p['color_keys']),
                         ('material_keys', lambda p: p['material_keys'])):
        counter = Counter(v for p in products for v in values_of(getter(p)))
        add(name, 'product', counter, sum(1 for p in products if values_of(getter(p))))

    attributes, coverage = {}, Counter()
    for product in products:
        for name, raw in product['attributes'].items():
            got = values_of(raw)
            if got:
                attributes.setdefault(name, Counter()).update(got)
                coverage[name] += 1
    for name, counter in attributes.items():
        add(name, 'attribute', counter, coverage[name])

    sku_props, sku_coverage = {}, Counter()
    skus = [s for p in products for s in p['skus']]
    for sku in skus:
        for prop in sku['properties']:
            if prop.get('name') and prop.get('value'):
                sku_props.setdefault(prop['name'], Counter())[prop['value']] += 1
                sku_coverage[prop['name']] += 1
    for name, counter in sku_props.items():
        entry = spec(name, 'sku_property', counter, sku_coverage[name], len(skus))
        entry['coverage_ratio'] = round(sku_coverage[name] / len(skus), 3)
        fields[name] = entry

    prices = [s['auth_price_raw'] for s in skus if s.get('auth_price_raw')]
    fields['price_cny'] = {
        'level': 'sku', 'kind': 'range', 'coverage': len(prices),
        'coverage_ratio': round(len(prices) / len(skus), 3), 'distinct': len(set(prices)),
        # 单位陷阱写进数据，而不是留在某个人的脑子里：SKU 存的是分。
        'source': 'skus[].auth_price_raw', 'source_unit': 'cents', 'divisor': 100,
        'min': min(prices) / 100, 'max': max(prices) / 100, 'filterable': True,
        'tier': 'primary', 'hard_filter_safe': True,
    }
    for name, replacement in SUPERSEDED.items():
        if name in fields and replacement in fields:
            fields[name]['superseded_by'] = replacement
    return {'products': total, 'skus': len(skus), 'fields': dict(sorted(fields.items())),
            'bounds': bounds(products)}


# 伪需求纠正与冲突检测的依据：目录里某个条件实际能买到的最低价。
# 从目录算，不手写，换数据自动跟着变。
MATERIAL_KEYWORDS = ('925银', '纯银', '银', 'K金', '18k金', '黄金', '白金', '铂金',
                     '玫瑰金', '钻', '珍珠', '翡翠', '玉髓')


def price_span(products):
    prices = [s['auth_price_raw'] / 100 for p in products for s in p['skus'] if s.get('auth_price_raw')]
    return {'min': min(prices), 'max': max(prices), 'products': len(products)} if prices else None


def bounds(products):
    by_category, by_material = {}, {}
    for name in sorted({p['category'] for p in products}):
        span = price_span([p for p in products if p['category'] == name])
        if span:
            by_category[name] = span
    for keyword in MATERIAL_KEYWORDS:
        matched = [p for p in products if keyword in str(p['attributes'].get('材质', ''))]
        span = price_span(matched)
        if span:
            by_material[keyword] = span
    return {'price_cny_by_category': by_category, 'price_cny_by_material': by_material}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=ROOT / 'fixtures/catalog/products.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT / 'fixtures/catalog/field-schema.json')
    parser.add_argument('--today', default=date.today().isoformat())
    args = parser.parse_args()

    products = [json.loads(line) for line in
                args.catalog.read_text(encoding='utf-8').splitlines() if line.strip()]
    schema = derive(products)
    dump(args.output, {'generated_at': args.today,
                       'catalog': str(args.catalog.relative_to(ROOT)), **schema})
    usable = [n for n, f in schema['fields'].items() if f['filterable']]
    safe = [n for n, f in schema['fields'].items() if f.get('hard_filter_safe')]
    print(f"{schema['products']} products · {schema['skus']} skus · "
          f"{len(usable)}/{len(schema['fields'])} filterable · {len(safe)} safe as hard filters")
    print('  hard-filter safe: ' + ', '.join(safe))


if __name__ == '__main__':
    main()
