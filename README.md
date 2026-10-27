# Chat API（渐进式实现）

第一步提供 FastAPI 消息接口、内部历史上下文和 SQLite 存储。
当前使用明确标记的本地占位回复，不调用真实模型。原有 `app.py` 示例保持独立。

## 运行

```bash
uv sync
uv run uvicorn backend.main:app --reload
```

接口文档：http://127.0.0.1:8000/docs

```bash
curl http://127.0.0.1:8000/api/chat/messages \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"u001","session_id":"s001","content":"你好"}'
```

返回示例：

```json
{
  "user_id": "u001",
  "session_id": "s001",
  "message": {"role": "assistant", "content": "[本地占位回复] 已加载 0 条历史消息；当前消息：你好"},
  "mode": "local_stub"
}
```

使用相同 `user_id` 和 `session_id` 再次发送消息时，服务端自动带入历史。

## 流式接口（SSE）

`POST /api/chat/stream` 与 `/api/chat/messages` 请求体相同，响应为 `text/event-stream`：

```bash
curl -N http://127.0.0.1:8000/api/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"u001","session_id":"s001","content":"你好"}'
```

```
event: conversation.started
data: {"user_id": "u001", "session_id": "s001", "mode": "local_stub"}

event: assistant.delta
data: {"text": "[本地占位回复] 已加载"}

event: assistant.text
data: {"user_id": "u001", "session_id": "s001", "message": {...}, "mode": "local_stub"}

event: conversation.completed
data: {"user_id": "u001", "session_id": "s001"}
```

- `assistant.delta` 按顺序拼接即为完整回复；`assistant.text` 的载荷与 `/api/chat/messages` 的 `ChatResponse` 一致，可直接用于最终渲染。
- 响应头发出后 HTTP 状态码无法再改，因此生成中途失败走 `event: error`，载荷为 `{"code":"MODEL_ERROR","message":...,"retryable":true}`；此时不写入任何历史。请求体校验仍在流开始前完成，非法输入返回 422。
- 落库时机与非流式一致：整段生成成功后在同一事务写入完整一轮，两个接口共用同一份历史。
- 该端点为 POST，浏览器原生 `EventSource` 只支持 GET，前端需用 `fetch` + `ReadableStream`（或 `@microsoft/fetch-event-source`）消费。

## 消息约定

- `ChatMessage`：外部请求，仅含 `user_id`、`session_id`、`content`。均为非空字符串；ID 最长 128 字符，正文最长 32000 字符；去除首尾空白，禁止额外字段。
- `HistoryMessage`：`role`（user/assistant）、`content`。
- `ChatContext`：内部封装，包含 `user_id`、`session_id`、`message`、`history`。历史不包含当前消息；`to_messages()` 按历史顺序追加当前消息一次。
- `ChatResponse`：返回用户、会话、助手消息和运行模式。

历史按 `(user_id, session_id)` 隔离，成功生成回复后事务保存整轮消息。
默认数据库为项目 `data/chat.sqlite3`，可用环境变量 `CHAT_DB_PATH` 指定路径。
启动时创建表，不在导入模块时创建数据库。

## 代码结构

- `backend/main.py`：应用工厂、生命周期、路由。
- `backend/models.py`：外部与内部消息模型。
- `backend/service.py`：历史组装、回复适配接口（`reply` 必选、`stream` 可选）和本地占位实现。
- `backend/sse.py`：SSE 帧编码。
- `backend/store.py`：SQLite 历史读写。
- `tests/test_chat.py`：输入校验、历史顺序、隔离、持久化、失败不落库。
- `tests/test_stream.py`：SSE 事件顺序、增量拼接、失败不落半轮、无 `stream` 后端的退化路径。

```bash
uv run pytest -q
```

测试采用 FastAPI 官方的 [TestClient 生命周期方式](https://fastapi.tiangolo.com/advanced/testing-events/)。

## 当前边界和下一步

本阶段仅供本地单进程运行；请求串行处理，不支持多 worker 的同会话并发协调。
`user_id` 目前由调用方提供，属于历史分区键，不代表已经完成身份认证。
暂未提供历史截断、请求幂等与真实模型调用；流式回复已就绪，但占位后端的分片是定长切片而非真实模型增量。
下一步实现 `ChatBackend.reply(context)` 的真实模型适配，复用当前历史封装。

## 商品检索数据

[商品 schema、导入文件和 Ollama Embedding 说明](fixtures/products/README.md)：以 `fixtures/raw/poizon` 为来源，包含103个SPU、598个SKU，以及ES Bulk和all-minilm的384维Qdrant真实向量。旧合成数据已由原始数据替换，可在Git历史中查看。

```bash
uv run python scripts/prepare_products.py
uv run python scripts/embed_products.py
```

## 环境配置

配置模板见 [.env.example](.env.example)。将所需字段填入 `.env`；已有文件请补充字段，不要覆盖现有密钥。

代码当前读取进程环境变量，运行时用 `--env-file` 显式加载：

```bash
uv run --env-file .env uvicorn backend.main:app --reload
uv run --env-file .env python scripts/embed_products.py
```

Embedding和SQLite配置已接入；OpenAI聊天、ES、Qdrant配置预留给后续适配器。导入文档中的curl示例使用固定本地地址，不会自动读取这些配置。`.env` 不提交Git，`.env.example` 只保存示例值。
