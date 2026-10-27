"""Labelled cases for intent and slot extraction."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Case:
    id: str
    text: str
    intent: str
    group: str
    # 期望抽到的字段 -> 期望取值；取值为 None 表示只看字段抽没抽到，不看值。
    slots: dict = field(default_factory=dict)
    # 明确不该出现的字段（目录答不了，或用户根本没说）。
    forbidden: tuple = ()
    # 期望命中的知识库规则 id。
    rules: tuple = ()
    # 这一轮是否应该反问（核心槽位不全）。
    clarify: bool = False
    # 多轮场景的前序约束：(字段, 操作符, 值)。
    before: tuple = ()
    note: str = ''


CASES = (
    # ---- A 完整需求：类目和预算都给了，不该再问 ----
    Case('A-01', '想买条项链，预算3000以内', 'discover', 'A 完整需求',
         {'category': '项链', 'price_cny': 3000}, clarify=False),
    Case('A-02', '两万块以内的戒指', 'discover', 'A 完整需求',
         {'category': '戒指', 'price_cny': 20000}, clarify=False),
    Case('A-03', '手链 预算五千', 'discover', 'A 完整需求',
         {'category': '手链', 'price_cny': 5000}, clarify=False),
    Case('A-04', '一千以内的充电宝', 'discover', 'A 完整需求',
         {'category': '移动电源', 'price_cny': 1000}, clarify=False,
         note='充电宝不是目录类目名，正确类目是移动电源'),
    Case('A-05', '预算500买个蓝牙音箱', 'discover', 'A 完整需求',
         {'category': '蓝牙音箱', 'price_cny': 500}, clarify=False),
    Case('A-06', '三万以内的银项链', 'discover', 'A 完整需求',
         {'category': '项链', 'price_cny': 30000, 'material_keys': None},
         rules=('SR-M01',), clarify=False),
    Case('A-07', '一万左右的玫瑰金手链', 'discover', 'A 完整需求',
         {'category': '手链', 'price_cny': None}, clarify=False,
         note='「一万左右」可能被读成区间，所以只看字段不看值'),
    Case('A-08', '黑色的耳机 八百以内', 'discover', 'A 完整需求',
         {'price_cny': 800}, clarify=False,
         note='耳机有两个叶子类目，抽成哪个都算对，所以不标 category'),

    # ---- B 缺核心槽位：该反问 ----
    Case('B-01', '想买条项链', 'discover', 'B 缺槽位',
         {'category': '项链'}, forbidden=('price_cny',), clarify=True),
    Case('B-02', '有什么戒指推荐', 'discover', 'B 缺槽位',
         {'category': '戒指'}, forbidden=('price_cny',), clarify=True),
    Case('B-03', '预算3000', 'discover', 'B 缺槽位',
         {'price_cny': 3000}, forbidden=('category',), clarify=True),
    Case('B-04', '帮我推荐点东西', 'discover', 'B 缺槽位',
         {}, forbidden=('category', 'price_cny'), clarify=True),
    Case('B-05', '想买个数码产品', 'discover', 'B 缺槽位',
         {}, forbidden=('price_cny',), clarify=True,
         note='数码是大类不是叶子类目，抽不出具体类目是对的'),
    Case('B-06', '看看首饰', 'discover', 'B 缺槽位',
         {}, forbidden=('price_cny',), clarify=True,
         note='首饰同样不是叶子类目'),

    # ---- C 送礼场景：知识库该触发 ----
    Case('C-01', '想送女朋友一条项链', 'discover', 'C 送礼场景',
         {'category': '项链'}, rules=('SR-G01',), clarify=True),
    Case('C-02', '送女友的生日礼物，预算两千', 'discover', 'C 送礼场景',
         {'price_cny': 2000}, rules=('SR-G01',), clarify=False,
         note='知识库把类目收敛到首饰，所以核心槽位已齐'),
    Case('C-03', '情人节送什么好，预算五千', 'discover', 'C 送礼场景',
         {'price_cny': 5000}, rules=('SR-G01',), clarify=False),
    Case('C-04', '我老婆生日快到了 想送个首饰', 'discover', 'C 送礼场景',
         {}, rules=('SR-G01',), clarify=True,
         note='口语送礼。触发词表现在没有这种说法，这条预期会失败'),
    Case('C-05', '给我妈买个礼物 预算三千', 'discover', 'C 送礼场景',
         {'price_cny': 3000}, rules=('SR-G01',), clarify=False,
         note='「给X买个礼物」句式，触发词表未覆盖'),
    Case('C-06', '闺蜜结婚 送什么好', 'discover', 'C 送礼场景',
         {}, rules=('SR-G01',), clarify=True,
         note='结婚场景，触发词表未覆盖'),
    Case('C-07', '想给女朋友一个惊喜 预算一万', 'discover', 'C 送礼场景',
         {'price_cny': 10000}, rules=('SR-G01',), clarify=False,
         note='不含「送」字的送礼表达'),
    Case('C-08', '买个礼物送人 五千左右', 'discover', 'C 送礼场景',
         {}, rules=('SR-G01',), clarify=False,
         note='「送人」而非「送女朋友」'),

    # ---- D 材质与颜色 ----
    Case('D-01', '要925银的项链 两万以内', 'discover', 'D 材质颜色',
         {'category': '项链', 'material_keys': '银/925银', 'price_cny': 20000},
         rules=('SR-M01b',), clarify=False),
    Case('D-02', '纯银手链 一万以内', 'discover', 'D 材质颜色',
         {'category': '手链', 'material_keys': '银/纯银', 'price_cny': 10000},
         rules=('SR-M01c',), clarify=False),
    Case('D-03', '带钻的戒指 五万以内', 'discover', 'D 材质颜色',
         {'category': '戒指', 'material_keys': '宝石/钻石', 'price_cny': 50000},
         rules=('SR-M03',), clarify=False),
    Case('D-04', '铂金的 预算三万', 'discover', 'D 材质颜色',
         {'material_keys': '铂金', 'price_cny': 30000}, rules=('SR-M04',), clarify=True),
    Case('D-05', '银色的项链 两万以内', 'discover', 'D 材质颜色',
         {'category': '项链', 'price_cny': 20000}, clarify=False,
         note='「银色」是颜色不是材质，抽成 color_keys 或 material_keys 都讲得通'),
    Case('D-06', '这个是银子的吗', 'spec_query', 'D 材质颜色',
         {}, clarify=True, note='「银子」是口语，触发词表未覆盖；且指代无实体'),
    Case('D-07', '想要闪一点的', 'discover', 'D 材质颜色',
         {}, clarify=True, note='「闪」指向钻石，但触发词表只有「钻」字面'),

    # ---- E 越界：必须 unknown ----
    Case('E-01', '今天天气怎么样', 'unknown', 'E 越界'),
    Case('E-02', '你是谁', 'unknown', 'E 越界'),
    Case('E-03', '帮我写首诗', 'unknown', 'E 越界'),
    Case('E-04', '1+1等于几', 'unknown', 'E 越界'),
    Case('E-05', '你们家有手机壳吗', 'unknown', 'E 越界',
         note='是购物意图但目录里没有这个类目，按越界处理'),
    Case('E-06', '帮我订张机票', 'unknown', 'E 越界'),
    Case('E-07', '谢谢', 'unknown', 'E 越界', note='寒暄，不该进业务流程'),

    # ---- F 相对表述：需要锚点 ----
    Case('F-01', '有没有便宜点的', 'discover', 'F 相对表述',
         {'price_cny': 2100}, before=(('category', '==', '项链'), ('price_cny', '<=', 3000)),
         clarify=False, note='锚到 3000×0.7'),
    Case('F-02', '再便宜一点', 'discover', 'F 相对表述',
         {'price_cny': 1400}, before=(('category', '==', '戒指'), ('price_cny', '<=', 2000)),
         clarify=False),
    Case('F-03', '有没有贵一点的', 'discover', 'F 相对表述',
         {'price_cny': 6500}, before=(('category', '==', '手链'), ('price_cny', '<=', 5000)),
         clarify=False),
    Case('F-04', '有没有便宜点的', 'discover', 'F 相对表述',
         {}, forbidden=('price_cny',), clarify=True,
         note='没有锚点就不该折算，应该反问'),

    # ---- G 多轮改写 ----
    Case('G-01', '算了 我想看看戒指', 'discover', 'G 多轮改写',
         {'category': '戒指'}, before=(('category', '==', '项链'), ('price_cny', '<=', 5000)),
         clarify=False, note='改类目，预算应当保留'),
    Case('G-02', '那预算提到两万', 'discover', 'G 多轮改写',
         {'price_cny': 20000}, before=(('category', '==', '耳钉'), ('price_cny', '<=', 2000)),
         clarify=False),
    Case('G-03', '换成银的吧', 'discover', 'G 多轮改写',
         {'material_keys': None}, before=(('category', '==', '项链'), ('price_cny', '<=', 20000)),
         rules=('SR-M01',), clarify=False),
    Case('G-04', '不要钻石的了', 'discover', 'G 多轮改写',
         {}, before=(('category', '==', '戒指'), ('price_cny', '<=', 50000)),
         clarify=False, note='否定表达，当前不支持取消已有约束，预期会把钻石当成正向条件'),

    # ---- H 口语与省略 ----
    Case('H-01', '项链 3000', 'discover', 'H 口语省略',
         {'category': '项链', 'price_cny': 3000}, clarify=False),
    Case('H-02', '有没有一千块左右的耳钉', 'discover', 'H 口语省略',
         {'category': '耳钉', 'price_cny': None}, clarify=False),
    Case('H-03', '预算大概三四千吧 想看项链', 'discover', 'H 口语省略',
         {'category': '项链', 'price_cny': None}, clarify=False),
    Case('H-04', '想买个送人的 不要太贵', 'discover', 'H 口语省略',
         {}, forbidden=('price_cny',), clarify=True,
         note='「不要太贵」没有锚点，不该折算成具体数字'),
    Case('H-05', '戒子 两万以内', 'discover', 'H 口语省略',
         {'category': '戒指', 'price_cny': 20000}, clarify=False,
         note='错别字：戒子 → 戒指'),
    Case('H-06', '耳钉 六千七百五十', 'discover', 'H 口语省略',
         {'category': '耳钉', 'price_cny': 6750}, clarify=False,
         note='中文数字'),

    # ---- I 冲突：条件本身买不到 ----
    Case('I-01', '预算2000的耳钉', 'discover', 'I 冲突',
         {'category': '耳钉', 'price_cny': 2000}, clarify=False,
         note='耳钉最低 6750，应当识别为冲突'),
    Case('I-02', '预算3000 要带钻的', 'discover', 'I 冲突',
         {'price_cny': 3000, 'material_keys': '宝石/钻石'}, rules=('SR-M03',), clarify=True,
         note='含钻最低 6500'),
    Case('I-03', '五百块的铂金戒指', 'discover', 'I 冲突',
         {'category': '戒指', 'material_keys': '铂金', 'price_cny': 500},
         rules=('SR-M04',), clarify=False),
)


def validate(schema) -> None:
    """标注里的字段和取值必须通过目录 schema；否则是标注漂移，不是模型的错。"""
    for case in CASES:
        for name, value in case.slots.items():
            if value is None:
                if schema.field(name) is None:
                    raise ValueError(f'{case.id}: unknown field {name!r}')
                continue
            op = '<=' if schema.field(name) and schema.field(name).kind == 'range' else '=='
            schema.validate(name, op, value)
        for name in case.forbidden:
            if schema.field(name) is None:
                raise ValueError(f'{case.id}: unknown forbidden field {name!r}')
