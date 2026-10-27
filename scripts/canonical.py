"""Derived fields for filtering: brand key, colour keys, material taxonomy."""

# 同一 brand_id 出现多种写法时的首选展示名。
# 目前只有蒂芙尼一条（'TIFFANY & CO./蒂芙尼' 与 'TIFFANYCO./蒂芙尼'）；
# 新数据引入第二条歧义时 test_ambiguous_brands_have_a_display_name 会失败。
BRAND_DISPLAY = {'10261': 'TIFFANY & CO./蒂芙尼'}

# 颜色在来源里散落在五个字段，而且「款式」实际存的几乎全是颜色
# （attribute 层 44 个值里 39 个、SKU 层 213 个里 211 个是颜色词）。
# 不并成一个字段，按「银色」过滤只能命中 58/121，其余散在别处。
COLOR_ATTRIBUTES = ('配色', '颜色', '主色', '辅色', '款式')
COLOR_SKU_PROPERTIES = ('颜色', '款式')
# 先长后短，避免「玫瑰金」被「金」截断；一个写法可命中多个基色
# （钛金灰 → 金、灰），偏向召回：过滤宁可多给，原始写法仍在字段里可供展示。
COLOR_VOCABULARY = ('玫瑰金', '银', '金', '黑', '白', '粉', '红', '绿',
                    '蓝', '紫', '灰', '米', '橙', '黄', '棕', '褐')


def _color_sources(product):
    for name in COLOR_ATTRIBUTES:
        raw = product['attributes'].get(name)
        if raw:
            yield from (raw if isinstance(raw, list) else [raw])
    for sku in product.get('skus') or []:
        for prop in sku['properties']:
            if prop.get('name') in COLOR_SKU_PROPERTIES and prop.get('value'):
                yield prop['value']


def color_keys(product):
    """把散落的颜色写法归到基色；非颜色词（小衫、修身裤）自然不命中。"""
    found = []
    for value in _color_sources(product):
        for base in COLOR_VOCABULARY:
            if base in str(value) and base not in found:
                found.append(base)
    return sorted(found)


# 材质是自由文本，46 种复合写法（K金925银18k金、18K白金钻石绿松石…）。
# 朴素子串有真实误命中：「金属树脂」含「金」却不是金，「水钻」含「钻」却不是钻石。
# 所以用最长优先 + 消费式扫描：命中的词从串里消耗掉，短词不再误触。
# 值为 None 的词只用于消费，不产生标签。
MATERIAL_TERMS = {
    '18K玫瑰金': '金/玫瑰金', '18k玫瑰金': '金/玫瑰金', '玫瑰金': '金/玫瑰金',
    '18K白金': '金/白金', '白金': '金/白金',          # 本目录中铂金一律写作「铂金」
    '18K黄金': '金/黄金', '黄金': '金/黄金',
    '18K金': '金/K金', '18k金': '金/K金', 'K金': '金/K金', 'k金': '金/K金',
    '铂金': '铂金', 'PT950': '铂金', '950': None,      # 「铂金950」里的 950 是成色
    '925银': '银/925银', '纯银': '银/纯银', '银': '银',
    '水钻': '宝石/水钻',                                # 必须排在「钻」之前
    '钻石': '宝石/钻石', '钻': '宝石/钻石',
    '珍珠': '宝石/珍珠', '母贝': '宝石/母贝',
    '红宝石': '宝石/红宝石', '绿松石': '玉石/绿松石',
    '翡翠': '玉石/翡翠', '玉髓': '玉石/玉髓',
    '金属': None,                                       # 金属 ≠ 金
    '树脂': '其他/树脂', '聚酯纤维': '其他/聚酯纤维', '镀铑': '其他/镀铑',
}
# 扫描顺序按词长降序，保证「18K玫瑰金」先于「玫瑰金」先于「金」。
_MATERIAL_SCAN = tuple(sorted(MATERIAL_TERMS, key=len, reverse=True))


def material_keys(product):
    """产出层级标签，如 ('银', '银/925银', '金', '金/K金')。

    父级与叶子都产出，所以「要银的」查 `银`、「要925银」查 `银/925银`，
    不必在宽窄之间二选一。
    """
    text = str(product['attributes'].get('材质') or '')
    found, cursor = [], 0
    while cursor < len(text):
        for term in _MATERIAL_SCAN:
            if text.startswith(term, cursor):
                path = MATERIAL_TERMS[term]
                if path:
                    parts = path.split('/')
                    for depth in range(1, len(parts) + 1):
                        tag = '/'.join(parts[:depth])
                        if tag not in found:
                            found.append(tag)
                cursor += len(term)
                break
        else:
            cursor += 1
    return sorted(found)


def canonicalize(product):
    """只追加派生字段；brand_id 和 brand_names 保持来源原样。"""
    names = product.get('brand_names') or []
    brand_id = product.get('brand_id')
    product['brand_key'] = brand_id or (names[0] if names else None)
    product['brand_display'] = BRAND_DISPLAY.get(brand_id) or (names[0] if names else '')
    product['color_keys'] = color_keys(product)
    product['material_keys'] = material_keys(product)
    return product


def ambiguous_brands(products):
    """返回 brand_id -> 多种写法；展示名表必须覆盖其中每一个。"""
    spellings = {}
    for product in products:
        for name in product.get('brand_names') or []:
            spellings.setdefault(product.get('brand_id'), set()).add(name)
    return {key: value for key, value in spellings.items() if len(value) > 1}
