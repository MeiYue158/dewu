# Poizon 原始商品导入包

基于 `fixtures/raw/poizon/*.json` 整理，不补造商品、描述或类目。共103个SPU、598个SKU、21个原始叶子类目；商品以首饰为主，也有鞋服、手表和背包。原始文件保持不变。

## Schema 与转换

检索粒度为 **SPU（一商品一点）**，不是SKU。ES `_id`、`product_id`、Qdrant point id 使用由 `poizon:spu:<spuId>` 派生的固定UUIDv5，跨库去重与重复导入保持一致。

| 字段 | 原始来源 | ES mapping / 用途 |
|---|---|---|
| product_id / spu_id | detail.spuId | keyword；跨库身份 / 源SPU |
| title | detail.title | text，cjk；全文检索 |
| attributes | baseProperties.list + brandList + saleProperties.list | object，enabled=false；保留属性原值，相同键多值去重为数组 |
| attributes_text | 按属性键排序拼接 | text，cjk；全文检索 |
| description / subtitle | detail.desc / subTitle | text，index=false；展示 |
| category / category_id | detail.categoryName / categoryId | keyword；保留原始叶子类目，支持过滤 |
| category_path_ids | level1CategoryId / level2CategoryId / categoryId | keyword数组；保留原分类层级ID，不猜测上级名称 |
| brand_id / brand_names | detail.brandId / baseProperties.brandList | keyword；品牌过滤 |
| article_number | detail.articleNumber | keyword；原货号 |
| image_url | detail.logoUrl | text，index=false；展示 |
| release_date_raw | detail.sellDate | keyword；原始发售日期字符串，允许“2025秋冬”等格式 |
| auth_price_raw | detail.authPrice | long，不建索引；单位、语义未经确认 |
| skus | data.skus，关联 saleProperties | object，enabled=false；保留sku_id、原始价格/status、规格对应关系 |
| size_info | data.sizeInfo | object，enabled=false；原尺码表 |
| source | 原文件路径、SHA256、provider | object，enabled=false；来源追溯 |

注意：

- 103条 `desc` 均为空，保持空字符串，不用模型编造描述。
- 20个文件有未转义引号或字面制表符。转换器只对已识别的属性值边界转义引号，并允许读取原始控制字符；输出由标准JSON编码器重新转义。详见 `quality-report.json` 的逐文件修复记录。未知错误会记录并以非零退出码结束。
- 同一属性的多值表示SPU下的选项集合，不能据此推断所有组合都有SKU；具体组合以 `skus` 为准。598个SKU的属性引用全部成功解析。
- `detail.authPrice` 和 `sku.authPrice` 存在量级差异，不擅自换算成人民币售价。`发售价格` 保留原始属性文本，不等于实时成交价；零值不解释为免费。
- 原数据没有可靠采集时间，`source.collected_at=null`，`live_verified=false`。原始商品名称与品牌来自文件，不表示本次已验证商品仍在售。

## 检索边界

ES 默认字段及示例查询均只有 `title` 和 `attributes_text`；类目仅可作过滤条件，描述不参与搜索。

Qdrant向量输入固定为：

```text
标题：<title>
属性：<按键名排序的属性名和值>
```

不拼接 description、category、source、图片或完整SKU表。销售属性已作为属性选项纳入文本。

## 已生成文件

- `products.jsonl`：103行规范化商品。
- `es.index.json`：创建索引配置（本地教学单分片、零副本）。
- `es.bulk.jsonl`：206行Bulk NDJSON，103个index操作，末尾含换行。
- `es.search.json`：仅标题、属性的查询请求体，最多10条。
- `qdrant.inputs.jsonl`：103行嵌入输入和payload。
- `quality-report.json`：计数、原始类目分布、修复记录与缺失字段情况。
- `vectors/qdrant.collection.json`：384维、Cosine。
- `vectors/qdrant.points.jsonl`：103条真实向量及payload。
- `vectors/qdrant.upsert.json`：可直接提交的Qdrant批量请求体。
- `vectors/qdrant.query.json`：“银色项链”的真实查询向量请求，limit=10、score_threshold=0.8。
- `vectors/embedding.manifest.json`：模型、维度、预处理版本、输入文件哈希。
- `vectors/smoke.json`：导出向量的本地余弦查询结果；不是在线Qdrant结果或检索准确率评测。

## Embedding：本地 Ollama

共享实现为 `backend/embeddings.py`，文档和查询均通过 OpenAI SDK 请求本地 `/v1/embeddings`：

```text
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=all-minilm
EMBEDDING_API_KEY=ollama
```

以上为默认值，`ollama` 是本地接口所用占位值，不读取项目的聊天服务密钥。无需修改现有 `.env`。

实际验证：模型输出384维；最长原文353字符，整段请求会被截断。使用96字符分段，每段向量归一化，再按段长度加权平均并归一化，得到每个SPU一个向量。103条商品共294段，已通过原生 `/api/embed` 的 `truncate=false` 诊断确认每段可完整编码；正式数据向量生成全部走 OpenAI 兼容接口。

`char96_length_weighted_mean_l2_v1` 是教学阶段的简单确定性聚合，并非模型原生长文本语义编码。查询必须复用同一实现；换模型、分段策略后需重新生成向量。

```python
from backend.embeddings import OllamaEmbeddings
query_vector = OllamaEmbeddings().embed_query("银色项链")
```

保留业务阈值0.8。实际“银色项链”查询在本批向量上高于等于0.8的结果为0，因此不能把“生成向量成功”等同于“中文语义召回效果达标”。不为得到结果而静默降低阈值。

## 重建

在仓库根目录执行：

```bash
uv run python scripts/prepare_products.py
uv run python scripts/embed_products.py --provider ollama --model all-minilm \
  --input fixtures/products/qdrant.inputs.jsonl --output fixtures/products/vectors
uv run pytest -q tests/test_product_fixtures.py tests/test_poizon.py
```

## 导入 ES

假设本地无认证。创建新索引，不会自动删除旧索引；如果已存在且mapping一致，只重跑Bulk即可。

```bash
curl --fail-with-body -X PUT http://localhost:9200/poizon_products_v1 \
  -H 'Content-Type: application/json' --data-binary @fixtures/products/es.index.json
curl --fail-with-body -X POST 'http://localhost:9200/_bulk?refresh=wait_for' \
  -H 'Content-Type: application/x-ndjson' --data-binary @fixtures/products/es.bulk.jsonl
curl --fail-with-body http://localhost:9200/poizon_products_v1/_count
curl --fail-with-body -X POST http://localhost:9200/poizon_products_v1/_search \
  -H 'Content-Type: application/json' --data-binary @fixtures/products/es.search.json
```

检查Bulk的 `errors=false` 和count=103；HTTP 200不代表所有Bulk条目都成功。

## 导入 Qdrant

```bash
curl --fail-with-body -X PUT http://localhost:6333/collections/poizon_products_v1 \
  -H 'Content-Type: application/json' --data-binary @fixtures/products/vectors/qdrant.collection.json
curl --fail-with-body -X PUT 'http://localhost:6333/collections/poizon_products_v1/points?wait=true' \
  -H 'Content-Type: application/json' --data-binary @fixtures/products/vectors/qdrant.upsert.json
curl --fail-with-body -X POST http://localhost:6333/collections/poizon_products_v1/points/query \
  -H 'Content-Type: application/json' --data-binary @fixtures/products/vectors/qdrant.query.json
```

已完成真实本地Embedding调用和文件校验；未向实际ES/Qdrant实例写入。双接口、意图识别、双路融合与rerank属于后续应用实现，不在本数据导入包中宣称完成。

参考：[Ollama OpenAI compatibility](https://github.com/ollama/ollama/blob/main/docs/api/openai-compatibility.mdx)、[all-minilm 模型](https://ollama.com/library/all-minilm)。

## 使用 .env 导入远程服务

```bash
uv run --env-file .env python scripts/import_products.py
# 分别执行
uv run --env-file .env python scripts/import_products.py --service es
uv run --env-file .env python scripts/import_products.py --service qdrant
```

脚本使用配置中的索引/集合名，ES支持API Key或Basic认证及自定义CA，Qdrant支持API Key。已有资源先核对schema，不会删除索引或集合；相同商品ID会更新为当前导入文件的内容。创建ES索引时不指定分片/副本拓扑，以兼容Serverless。

导入后逐条回读源文档/向量并执行查询，脱敏报告保存在 `docs/evidence/import-<service>.json`。`import-both.json` 是首次尝试，后续单服务报告代表各自最近一次验证。

2026-10-22实测：ES成功写入103条、逐条回读一致、标题/属性查询返回10条。Qdrant首次返回403，未写入；需要配置可访问地址和对应鉴权后重试。此前本地向量验证不等于远程Qdrant导入成功。
