# AGENTS.md — API 项目专属指令与问题记录

> **本文件必须在每次 Session 开始时优先阅读。**

---

## 一、项目定位

本项目是基于 [gcli2api](https://github.com/su-kaka/gcli2api) 的重构与扩展版本，提供：

- **GeminiCLI 路由**（`/v1/...`）：通过 Google 内部 `cloudcode-pa.googleapis.com` API 调用 Gemini 模型
- **Vertex AI 路由**（`/vertex/v1/...`）：通过 `aiplatform.googleapis.com` 调用 Gemini 模型
- **OpenAI / Gemini / Claude 格式兼容**：三种请求格式均支持
- **模型降级检测**：每次响应自动对比请求模型与实际响应模型并写入日志

**关于 Antigravity（反重力）**：本项目不部署反重力路由（`src/router/antigravity/` 不存在），
但 `config.py` 中**保留** `get_antigravity_api_url()` 和 `get_antigravity_stream2nostream()` 配置函数，
供控制面板 `/config/get` 使用，避免 `AttributeError`。**不要删除这两个函数。**

---

## 二、目录结构

```
├── web.py                          主服务入口（FastAPI + Hypercorn）
├── config.py                       配置管理（ENV > DB > default）
├── log.py                          异步日志系统
├── requirements.txt
├── src/
│   ├── api/
│   │   ├── geminicli.py            GeminiCLI API 客户端
│   │   └── vertex.py               Vertex AI API 客户端
│   ├── router/
│   │   ├── geminicli/              /v1/* 路由（OpenAI / Gemini / Claude 格式）
│   │   └── vertex/                 /vertex/v1/* 路由（同格式）
│   ├── converter/
│   │   ├── gemini_fix.py           请求规范化（thinkingConfig、安全设置等）
│   │   ├── openai2gemini.py        OpenAI ↔ Gemini 格式转换
│   │   └── fake_stream.py          假流式 / 心跳处理
│   ├── credential_manager.py       凭证轮换、刷新、封禁管理
│   ├── storage_adapter.py          存储抽象层（SQLite / 文件系统）
│   ├── panel/                      Web 控制面板（多模块）
│   └── utils.py                    认证、模型列表、常量
└── AGENTS.md
```

---

## 三、关键设计约束

### GeminiCLI 内部 API 响应结构

GeminiCLI 端点 (`v1internal:streamGenerateContent`) 将实际 Gemini 响应**包装**在一个额外层中：

```json
{
  "response": {
    "candidates": [...],
    "modelVersion": "gemini-3-pro-preview-06-05",
    "usageMetadata": {...}
  }
}
```

**不变量**：提取 `modelVersion` 必须先取 `response` 字段，再取 `modelVersion`。
函数 `_extract_gcli_model_version()` 处理此问题。

### Vertex AI 响应结构

Vertex AI 端点使用标准 Gemini 格式，**无 `response` 包装层**：

```json
{
  "candidates": [...],
  "modelVersion": "...",
  "usageMetadata": {...}
}
```

函数 `_extract_vertex_model_version()` 处理此格式。

---

## 四、模型降级检测

### 日志标记

| 日志标记 | 含义 |
|---|---|
| `[MODEL_VERSION]` | 每次请求均记录：请求模型 vs 实际响应模型 |
| `[MODEL_DOWNGRADE_DETECTED]` | 检测到模型系列不匹配（不同代际或不同类型） |
| `[GEMINI_FIX]` | 记录每次请求应用的 thinkingConfig |

### 降级检测逻辑

使用**模型系列**（model series）比较，而非精确字符串匹配：

```
gemini-3-pro-preview        → 系列: gemini-3-pro
gemini-3-pro-preview-06-05  → 系列: gemini-3-pro   （正常，版本后缀）
gemini-2.0-flash-001        → 系列: gemini-2.0-flash  （降级！）
```

实现：`_extract_model_series()` in `src/api/geminicli.py`

---

## 五、错误记录

### [2026-03-03] gemini-3-pro-preview 出字极快但质量极差

**现象**：调用 `gemini-3-pro-preview` 或 `gemini-3.1-pro-preview` 时，响应速度异常快，但质量极低。
调用 `gemini-2.5-pro` 则正常（速度慢、质量高、会 429）。

**根因（三叠加）**：

**根因 1 — `_extract_gcli_model_version` 未实现（P0，模型版本永远不记录）**

原始实现直接对 GeminiCLI 响应外层做 `resp_json.get("modelVersion")` ，而实际 `modelVersion`
嵌套在 `{"response": {"modelVersion": ...}}` 中，导致始终返回 `None`，模型降级完全无法被日志捕获。

**修复**：新增 `_extract_gcli_model_version()` 函数，优先读取 `response.modelVersion`，兼容已展开格式。

**根因 2 — `is_thinking_model` 将所有 pro 模型视为思考模型（P0，质量问题）**

原始代码：`return "think" in model_name or "pro" in model_name.lower()`

`gemini-3-pro-preview`（无后缀）被误判为思考模型，`normalize_gemini_request` 注入了
`{"thinkingConfig": {"includeThoughts": true}}`（无 budget 或 level）。

**此空 thinkingConfig 可能静默覆盖模型的默认思考预算，导致模型在接近零 budget 的情况下运行，
输出极快但质量极低。**

**修复**：`is_thinking_model` 改为仅检查 `"think" in model_name.lower()`，不再兜底所有 pro 模型。
`gemini-3-pro-preview`（无后缀）不再强制注入 thinkingConfig，使用 API 默认值。

**根因 3 — 降级比较使用精确字符串匹配（P1，大量误报/漏报）**

原始代码：`if actual_model_version != model_name`

`"gemini-3-pro-preview" != "gemini-3-pro-preview-06-05"` 始终为 True，产生大量误报（实际无降级）。

**修复**：引入 `_extract_model_series()` 函数，提取 `gemini-3-pro`/`gemini-2.0-flash` 等系列标识进行比较。

**诊断方法**：查看日志中的 `[MODEL_VERSION]` 行，若出现 `[MODEL_DOWNGRADE_DETECTED]` 则确认降级；
同时观察 `[GEMINI_FIX]` 行中的 thinkingConfig 是否符合预期。

---

### thinkingConfig 设置规则（已修复版本）

| 模型名 | thinkingConfig | 说明 |
|---|---|---|
| `gemini-3-pro-preview` | 不注入 | 使用 API 默认值（模型自身决策） |
| `gemini-3-pro-preview-high` | `{thinkingLevel: "high", includeThoughts: true}` | 显式高思考 |
| `gemini-3-pro-preview-low` | `{thinkingLevel: "low", includeThoughts: true}` | 显式低思考 |
| `gemini-2.5-pro` | 不注入 | 使用 API 默认值 |
| `gemini-2.5-pro-high` | `{thinkingBudget: 16000, includeThoughts: true}` | 2.5 系列用 budget |
| `gemini-2.5-pro-maxthinking` | `{thinkingBudget: 32768, includeThoughts: true}` | 旧兼容模式 |

---

## 六、认证流程

所有路由的认证均通过 `src/utils.py:authenticate_flexible()` 统一处理，支持：
- `Authorization: Bearer <token>`
- `x-api-key`, `x-goog-api-key`, `access_token` headers
- `?key=<token>` URL 参数

凭证文件（OAuth refresh_token）存储于 `./creds/` 目录（可通过 `CREDENTIALS_DIR` 环境变量覆盖），
由 `CredentialManager` 负责轮换、刷新和封禁。

---

## 七、环境变量速查

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PORT` | 7861 | 监听端口 |
| `HOST` | 0.0.0.0 | 监听地址 |
| `API_PASSWORD` | pwd | API 鉴权密码 |
| `PANEL_PASSWORD` | pwd | 控制面板密码 |
| `CREDENTIALS_DIR` | ./creds | OAuth 凭证目录 |
| `CODE_ASSIST_ENDPOINT` | https://cloudcode-pa.googleapis.com | GeminiCLI 端点 |
| `VERTEX_AI_LOCATION` | us-central1 | Vertex AI 区域（如 us-central1、asia-east1） |
| `VERTEX_AI_PROJECT_ID` | — | Vertex AI 项目 ID（通常从凭证自动获取，可覆盖） |
| `ANTIGRAVITY_API_URL` | https://daily-cloudcode-pa.sandbox.googleapis.com | Antigravity 端点（保留配置项，路由未部署） |
| `ANTIGRAVITY_STREAM2NOSTREAM` | true | Antigravity 非流式是否使用流式 API（保留配置项） |
| `PROXY` | — | HTTP 代理 |
| `AUTO_BAN` | false | 403/429 自动封禁 |
| `RETURN_THOUGHTS_TO_FRONTEND` | true | 是否返回思维链 |
| `LOG_LEVEL` | info | 日志级别（debug/info/warning/error） |
| `ENABLE_LOG` | 1 | 设为 0 彻底关闭日志 |

---

## 八、行为准则

1. **保留 antigravity 配置函数**：`get_antigravity_api_url()` 和 `get_antigravity_stream2nostream()` 必须存在于 `config.py`，否则面板 `/config/get` 报 `AttributeError`。不要删除它们，也不要引入反重力路由。
2. **每次改动 thinkingConfig 逻辑时**：必须验证 `gemini-3-pro-preview`（无后缀）不再被注入空 thinkingConfig。
3. **每次改动模型版本日志时**：必须验证 GeminiCLI 的 `response` 包装层已正确展开。
4. **降级检测以模型系列为单位**：不使用精确字符串匹配，避免误报。
5. **增加新路由时**：在 `web.py` 中注册，并在本文件更新路由列表。
6. **Vertex AI 渠道**：`src/api/vertex.py` + `src/router/vertex/` 已实现，使用 `geminicli` 模式凭证，区域由 `VERTEX_AI_LOCATION`（默认 `us-central1`）控制，项目 ID 从凭证自动读取。
