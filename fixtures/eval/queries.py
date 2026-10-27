"""Retrieval ground truth, derived from catalog fields rather than judgement."""


def attr(product, *keys):
    """把指定属性拼成一段可匹配的文本；属性值可能是字符串或列表。"""
    out = []
    for key in keys:
        value = product['attributes'].get(key)
        if value:
            out.extend(value if isinstance(value, list) else [value])
    return '、'.join(str(v) for v in out)


def sku_values(product, name):
    return {p['value'] for s in product['skus'] for p in s['properties']
            if p['name'] == name and p['value']}


def min_price_cny(product):
    prices = [s['auth_price_raw'] for s in product['skus'] if s.get('auth_price_raw')]
    return min(prices) / 100 if prices else None


# (查询, 判定函数, 分组)。分组只用于分段看结果，不参与计分。
QUERIES = [
    # A. 类目直查：查询词就是商品的叶子类目，最低难度。
    ('羽绒服', lambda p: p['category'] == '羽绒服', 'A 类目直查'),
    ('衬衫', lambda p: p['category'] == '衬衫', 'A 类目直查'),
    ('跑步鞋', lambda p: p['category'] == '跑步鞋', 'A 类目直查'),
    ('项链', lambda p: p['category'] == '项链', 'A 类目直查'),
    ('戒指', lambda p: p['category'] == '戒指', 'A 类目直查'),
    ('机械键盘', lambda p: p['category'] == '机械键盘', 'A 类目直查'),
    ('蓝牙音箱', lambda p: p['category'] == '蓝牙音箱', 'A 类目直查'),
    ('移动电源', lambda p: p['category'] == '移动电源', 'A 类目直查'),

    # B. 类目 + 属性：需要同时命中类目和一个结构化字段。
    ('银色的项链', lambda p: p['category'] == '项链' and '银' in attr(p, '配色', '颜色'), 'B 类目+属性'),
    ('925银手链', lambda p: p['category'] == '手链' and '925银' in attr(p, '材质'), 'B 类目+属性'),
    ('玫瑰金戒指', lambda p: p['category'] == '戒指' and '玫瑰金' in attr(p, '配色', '颜色', '材质'), 'B 类目+属性'),
    ('儿童鞋', lambda p: p['category'].startswith('儿童'), 'B 类目+属性'),
    ('不锈钢表带的手表', lambda p: '表' in p['category'] and '不锈钢' in attr(p, '表带材质', '材质'), 'B 类目+属性'),
    ('有礼盒包装的首饰', lambda p: bool(sku_values(p, '礼盒')), 'B 类目+属性'),

    # C. 自然语言导购：用户口吻，没有字面类目词。
    ('送女朋友的生日礼物', lambda p: p['category'] in {'项链', '手链', '戒指', '耳钉', '项坠/吊坠', '耳环', '手镯'}, 'C 自然语言'),
    ('降噪效果好的耳机', lambda p: '耳机' in p['category'] and attr(p, '主动降噪') == '支持', 'C 自然语言'),
    ('续航长一点的充电宝', lambda p: p['category'] == '移动电源', 'C 自然语言'),
    ('冬天穿的厚外套', lambda p: p['category'] == '羽绒服', 'C 自然语言'),
    ('跑马拉松穿什么鞋', lambda p: p['category'] == '跑步鞋', 'C 自然语言'),
    ('预算一千以内的数码产品', lambda p: p['source']['provider'] == 'synthetic'
     and (min_price_cny(p) or 1e9) <= 1000, 'C 自然语言'),
]


def gold(products):
    """返回 [(查询, 分组, {命中的 product_id})]；空集视为查询写错了。"""
    labelled = []
    for text, predicate, group in QUERIES:
        ids = {p['product_id'] for p in products if predicate(p)}
        if not ids:
            raise ValueError(f'Query selects nothing: {text}')
        labelled.append((text, group, ids))
    return labelled
