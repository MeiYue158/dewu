"""Generate the marked synthetic layer: copy, prices, and a digital category."""
import argparse
import json
import statistics
from datetime import date
from pathlib import Path

from product_data import ROOT, attributes_text, dump, jsonl

VERSION = 1
# 真实 SPU 用 6~8 位 id；合成数码单独占 9 开头的区间，肉眼即可分辨。
DIGITAL_SPU_BASE = 900000001
DIGITAL_CATEGORY_BASE = 990000

# 合成品牌全部虚构，避免把真实厂商和编造的规格绑在一起。
BRANDS = [('9900001', '岚核 LANKO'), ('9900002', '砚声 YANVOX'),
          ('9900003', '序比特 XUBIT'), ('9900004', '诺姆 NEROM')]

# (叶子类目, 型号前缀, 规格表, SKU 维度)
DIGITAL = [
    ('真无线耳机', 'TWS', [('蓝牙版本', '5.3'), ('单次续航', '6小时'), ('总续航', '30小时'),
                 ('充电接口', 'USB-C'), ('防水等级', 'IPX4'), ('主动降噪', '支持'),
                 ('适配系统', 'Android、iOS、Windows'), ('单耳重量', '4.5g')],
     [('颜色', ['曜石黑', '云母白', '雾松绿']), ('版本', ['标准版', '降噪版'])]),
    ('头戴式耳机', 'HPH', [('蓝牙版本', '5.2'), ('续航', '40小时'), ('充电接口', 'USB-C'),
                 ('有线模式', '支持3.5mm'), ('主动降噪', '支持'), ('重量', '255g'),
                 ('适配系统', 'Android、iOS、Windows、macOS')],
     [('颜色', ['星夜蓝', '砂岩灰']), ('版本', ['标准版', '旗舰版'])]),
    ('智能手表', 'SWT', [('屏幕尺寸', '1.43英寸'), ('续航', '14天'), ('充电接口', '磁吸'),
                ('防水等级', '5ATM'), ('适配系统', 'Android 8.0及以上、iOS 14及以上'),
                ('表带接口', '22mm通用'), ('重量', '38g')],
     [('颜色', ['钛金灰', '月光银']), ('版本', ['标准版', 'NFC版'])]),
    ('机械键盘', 'KBD', [('轴体', '线性轴'), ('连接方式', '蓝牙5.1、2.4G、有线'),
                ('充电接口', 'USB-C'), ('键位数', '87键'), ('适配系统', 'Windows、macOS'),
                ('键帽材质', 'PBT'), ('重量', '860g')],
     [('颜色', ['炭黑', '奶白']), ('版本', ['有线版', '三模版'])]),
    ('移动电源', 'PWB', [('容量', '20000mAh'), ('最大输出功率', '65W'), ('充电接口', 'USB-C'),
                ('输出接口', 'USB-C×2、USB-A×1'), ('支持协议', 'PD3.0、QC4+'),
                ('适配机型', '支持PD快充的手机与笔记本'), ('重量', '390g')],
     [('颜色', ['深空灰', '浅砂']), ('套装', ['单主机', '含100W线材'])]),
    ('蓝牙音箱', 'SPK', [('蓝牙版本', '5.3'), ('续航', '18小时'), ('充电接口', 'USB-C'),
                ('防水等级', 'IP67'), ('输出功率', '30W'), ('串联播放', '支持双机'),
                ('重量', '680g')],
     [('颜色', ['墨绿', '砖红', '石墨']), ('版本', ['标准版'])]),
]
# 每个叶子类目下生成入门/标准/旗舰三档，规格和价格真实有别，
# 否则同类目商品无法比较，FR-05 的对比卡就演示不出差异。
VARIANTS = {
    '真无线耳机': [('入门', 29900, {'单次续航': '5小时', '总续航': '20小时', '主动降噪': '不支持'}),
                ('标准', 59900, {}),
                ('旗舰', 99900, {'单次续航': '8小时', '总续航': '40小时', '防水等级': 'IPX5', '蓝牙版本': '5.4'})],
    '头戴式耳机': [('入门', 69900, {'续航': '30小时', '主动降噪': '不支持', '重量': '230g'}),
                ('标准', 129900, {}),
                ('旗舰', 199900, {'续航': '60小时', '蓝牙版本': '5.3', '重量': '280g'})],
    '智能手表': [('入门', 49900, {'续航': '7天', '屏幕尺寸': '1.32英寸', '防水等级': '3ATM'}),
               ('标准', 89900, {}),
               ('旗舰', 149900, {'续航': '21天', '屏幕尺寸': '1.5英寸', '防水等级': '10ATM'})],
    '机械键盘': [('入门', 26900, {'键位数': '61键', '连接方式': '有线', '键帽材质': 'ABS', '重量': '620g'}),
               ('标准', 46900, {}),
               ('旗舰', 79900, {'键位数': '104键', '轴体': '段落轴', '重量': '1100g'})],
    '移动电源': [('入门', 15900, {'容量': '10000mAh', '最大输出功率': '30W', '重量': '210g',
                              '输出接口': 'USB-C×1、USB-A×1'}),
               ('标准', 25900, {}),
               ('旗舰', 42900, {'容量': '30000mAh', '最大输出功率': '140W', '重量': '580g'})],
    '蓝牙音箱': [('入门', 19900, {'续航': '12小时', '输出功率': '10W', '重量': '340g'}),
               ('标准', 39900, {}),
               ('旗舰', 69900, {'续航': '24小时', '输出功率': '60W', '重量': '1200g'})],
}

# 描述只复述已有属性，顺序固定；没有该属性就跳过整句。
CLAUSES = [('材质', '材质{}'), ('配色', '配色{}'), ('颜色', '颜色{}'),
           ('尺寸', '尺寸{}'), ('长度', '长度{}'), ('款式', '{}款式'),
           ('风格', '{}风格'), ('适用季节', '适用季节{}')]


def flat(value):
    return '、'.join(value) if isinstance(value, list) else str(value)


def brand_of(product):
    names = product.get('brand_names') or []
    return names[0] if names else (flat(product['attributes']['品牌']) if '品牌' in product['attributes'] else '')


def describe(product):
    """用目录里已有的字段拼成一句话；不引入属性之外的任何断言。"""
    attrs = product['attributes']
    head = f"{brand_of(product)}{product['category'] or ''}".strip()
    parts = [flat(attrs[key]) and template.format(flat(attrs[key]))
             for key, template in CLAUSES if attrs.get(key)]
    parts = [p for p in parts if p]
    body = '，'.join(parts)
    return f'{head}。{body}。' if body else f'{head}。'


def subtitle(product):
    attrs = product['attributes']
    bits = [brand_of(product)]
    for key in ('材质', '款式', '配色', '颜色'):
        if attrs.get(key) and len(bits) < 3:
            bits.append(flat(attrs[key]).split('、')[0])
    return ' · '.join(b for b in bits if b)


def price_plan(products):
    """给整 SPU 缺价的商品补 SKU 价（单位：分）。

    SPU 层 auth_price_raw 的单位是元，SKU 层是分，两者比值中位数为 100，
    因此有 SPU 价时按 ×100 换算；SPU 价也为 0 时退回同类目真实 SKU 价中位数。
    """
    by_category = {}
    for product in products:
        for sku in product['skus']:
            if sku.get('auth_price_raw'):
                by_category.setdefault(product['category'], []).append(sku['auth_price_raw'])
    overall = statistics.median([v for values in by_category.values() for v in values])
    plans = {}
    for product in products:
        missing = [s for s in product['skus'] if not s.get('auth_price_raw')]
        if not missing:
            continue
        if product.get('auth_price_raw'):
            value, method = product['auth_price_raw'] * 100, 'spu_auth_price_cny_x100'
        else:
            values = by_category.get(product['category'])
            value = int(statistics.median(values)) if values else int(overall)
            method = 'category_median_of_real_skus' if values else 'catalog_median_of_real_skus'
        plans[product['product_id']] = ({s['sku_id']: int(value) for s in missing}, method)
    return plans


def build_overlay(products, today):
    plans = price_plan(products)
    entries = {}
    for product in products:
        fields, patch = [], {}
        if not (product.get('description') or '').strip():
            patch['description'] = describe(product)
            fields.append('description')
        if not (product.get('subtitle') or '').strip():
            patch['subtitle'] = subtitle(product)
            fields.append('subtitle')
        prices, method = plans.get(product['product_id'], ({}, None))
        if prices:
            patch['sku_prices'] = prices
            patch['sku_price_method'] = method
            fields.append('sku_auth_price_raw')
        if fields:
            patch['fields'] = fields
            entries[product['product_id']] = patch
    return {'generator': 'synthesize_products.py', 'version': VERSION, 'generated_at': today,
            'basis': 'fixtures/products/products.jsonl',
            'note': 'description/subtitle 仅复述该商品已有属性，不引入新事实；价格为换算或同类目中位数。',
            'products': entries}


def digital_snapshot(index, leaf, code, specs, dimensions, variant, today):
    spu_id = str(DIGITAL_SPU_BASE + index)
    category_id = str(DIGITAL_CATEGORY_BASE + next(i for i, d in enumerate(DIGITAL) if d[0] == leaf))
    brand_id, brand_name = BRANDS[index % len(BRANDS)]
    tier, price, overrides = variant
    specs = [(key, overrides.get(key, value)) for key, value in specs] + [('定位', f'{tier}款')]
    model = f'{code}-{index + 1:02d}'
    sale, value_id = [], int(spu_id) * 100
    for name, values in dimensions:
        for value in values:
            value_id += 1
            sale.append({'propertyValueId': value_id, 'name': name, 'value': value})
    skus, sku_id = [], int(spu_id) * 1000
    groups = [[p for p in sale if p['name'] == name] for name, _ in dimensions]
    def combine(rest, chosen):
        nonlocal sku_id
        if not rest:
            sku_id += 1
            skus.append({'skuId': sku_id, 'status': 1, 'authPrice': price,
                         'properties': [{'propertyValueId': p['propertyValueId'], 'level': i + 1}
                                        for i, p in enumerate(chosen)]})
            return
        for option in rest[0]:
            combine(rest[1:], chosen + [option])
    combine(groups, [])
    base = [{'key': '品牌', 'value': brand_name}, {'key': '主货号', 'value': model}] + \
           [{'key': key, 'value': value} for key, value in specs]
    return spu_id, {
        '_synthetic': {'generator': 'synthesize_products.py', 'version': VERSION,
                       'generated_at': today, 'whole_record': True,
                       'note': '虚构品牌与规格，仅用于补齐数码品类；不对应任何真实商品。'},
        'success': True, 'code': 200, 'message': None,
        'data': {
            'image': {'spuImage': {'images': []}},
            'skus': skus,
            'saleProperties': {'list': sale},
            'sizeInfo': {},
            'baseProperties': {'list': base, 'brandList': [{'brandId': int(brand_id), 'brandName': brand_name}]},
            'detail': {
                'spuId': int(spu_id), 'categoryId': int(category_id),
                'level1CategoryId': DIGITAL_CATEGORY_BASE, 'level2CategoryId': int(category_id),
                'brandId': int(brand_id), 'logoUrl': '', 'title': f'{brand_name} {leaf} {model}',  # 品牌 + 类目 + 型号，不编造系列名
                # 与 describe() 同一条规则：只复述上面列出的规格，不新增断言。
                'subTitle': f'{brand_name} · {leaf}',
                'desc': f'{brand_name}{leaf}。' + '，'.join(f'{k}{v}' for k, v in specs) + '。',
                'categoryName': leaf,
                'articleNumber': model, 'sellDate': f'{today} 00:00:00', 'authPrice': price // 100,
            },
        },
    }


def build_digital(output, today):
    output.mkdir(parents=True, exist_ok=True)
    for existing in output.glob('*.json'):
        existing.unlink()
    written, index = [], 0
    for leaf, code, specs, dimensions in DIGITAL:
        for variant in VARIANTS[leaf]:
            spu_id, envelope = digital_snapshot(index, leaf, code, specs, dimensions, variant, today)
            dump(output / f'{spu_id}.json', envelope)
            written.append(spu_id)
            index += 1
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--products', type=Path, default=ROOT / 'fixtures/products/products.jsonl')
    parser.add_argument('--overlay-output', type=Path, default=ROOT / 'fixtures/synthetic')
    parser.add_argument('--digital-output', type=Path, default=ROOT / 'fixtures/raw/synthetic')
    parser.add_argument('--today', default=date.today().isoformat())
    args = parser.parse_args()

    products = [json.loads(line) for line in args.products.read_text(encoding='utf-8').splitlines() if line.strip()]
    overlay = build_overlay(products, args.today)
    args.overlay_output.mkdir(parents=True, exist_ok=True)
    dump(args.overlay_output / 'overlay.json', overlay)
    digital = build_digital(args.digital_output, args.today)
    filled = sum(len(entry.get('sku_prices', {})) for entry in overlay['products'].values())
    dump(args.overlay_output / 'manifest.json', {
        'generator': 'synthesize_products.py', 'version': VERSION, 'generated_at': args.today,
        'deterministic': True,
        'overlay_products': len(overlay['products']),
        'filled_descriptions': sum('description' in e['fields'] for e in overlay['products'].values()),
        'filled_subtitles': sum('subtitle' in e['fields'] for e in overlay['products'].values()),
        'filled_sku_prices': filled,
        'digital_spus': len(digital), 'digital_categories': sorted({d[0] for d in DIGITAL}),
        'digital_brands_are_fictional': True,
        'excluded_from_embeddings': ['description', 'subtitle'],
    })
    print(f'overlay: {len(overlay["products"])} products, {filled} sku prices; digital: {len(digital)} spus')


if __name__ == '__main__':
    main()
