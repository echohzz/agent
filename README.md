
# 从零实现一个最小可用 Agent

这是一个基于 Python 标准库和阿里云百炼 `kimi-k3` 的命令行 Agent。项目没有使用 LangGraph、OpenHands、OpenClaw 等 Agent 框架，Agent Runtime、工具注册和执行循环均自行实现。

代码仓库：<https://github.com/echohzz/agent>。AI 辅助开发提示与问题解决记录见 [DEVELOPMENT.md](DEVELOPMENT.md)。

## 1. 题目要求与实现

| 目标 | 实现位置 | 说明 |
| --- | --- | --- |
| 基本 Agent 循环 | `agent/runtime.py` | 用户输入 → LLM 决策 → 工具调用 → 工具结果回传 → 继续循环或返回答案 |
| 工具注册机制 | `agent/tools.py` | 每个工具包含名称、描述、参数 Schema 和执行函数 |
| LLM 输出解析 | `agent/model.py`、`agent/runtime.py` | 解析 `content`、`tool_calls`、`reasoning_content`、`finish_reason` |
| 工具 | `agent/tools.py`、`agent/storage.py` | `calculator`、`search`、`todo` |
| Session 管理 | `agent/storage.py` | `(user_id, session_id)` 作为会话边界，窗口之间相互隔离 |
| Context 管理 | `agent/runtime.py` | 保留近期消息；超过阈值后压缩较早轮次 |
| 异常处理 | `agent/model.py`、`agent/runtime.py` | API 错误、无效响应、无效工具名和参数错误均有明确处理；截断响应不会当作完成 |
| 工具 Trace | `agent/storage.py` | 记录模型决策、思考过程、工具参数和工具结果，可用 `/trace` 查看 |
| 测试 | `test_main.py`、`test_model.py`、`test_cli.py`、`test_smoke.py` | 覆盖 Runtime、百炼适配层、CLI 和验收报告生成 |

## 2. 架构

```text
main.py                 CLI 入口与交互
  └── agent/runtime.py  Agent 循环、输出分流、Context 管理
        ├── agent/model.py    百炼 API 适配
        ├── agent/tools.py    工具定义、Schema、参数校验
        └── agent/storage.py  SQLite Session、Todo、Trace
```

1. 读取 `(user_id, session_id)` 对应的历史消息和摘要。
2. 将系统提示、历史 Context 和工具 Schema 发送给百炼 API。
3. 如果返回 `tool_calls`，校验工具名和参数后执行工具。
4. 将工具结果作为 `tool` 消息追加，再次调用 LLM。
5. 如果返回非空 `content` 且响应正常结束，将其作为最终答案；最多执行 8 轮。`length`、`content_filter` 或未知结束原因会报错；截断答案只写入 Trace 的 `partial_content`，不作为最终答案或有效摘要。

模型返回的 `reasoning_content` 原文只写入 Trace，不会进入下一轮 Context，也不会直接展示为用户答案。

### 工具写入与恢复

- 每次用户请求生成一个回合 ID；SQLite 以 `(user_id, session_id, turn_id, call_id)` 保存参数签名与完整结果。同一回合中相同调用 ID、相同参数返回原结果；参数变化则拒绝该批调用。新的用户回合允许复用调用 ID、添加相同文本的待办。
- 参数签名包含工具名称，并规范化 JSON 键顺序和空格。它只防止相同调用重放；模型为同一意图生成不同 ID 时，不能据此判断是不是重复任务。
- 待办写入、执行记录、工具 Trace 和对应 `tool` 历史消息在同一个 SQLite 事务内提交。嵌套保存点不会提前提交；任何一步失败都回滚当前调用，前面已经提交的调用保留。
- 下一次请求先补齐上次未完成的工具结果，再追加新的用户输入。正常结束或失败的调用不重复执行；批量调用只恢复缺失部分。回合 ID 仅供本地恢复使用，不传给模型 API。
- 原数据库无需删除；启动时自动创建 `tool_results` 表。旧版本历史缺少回合 ID，无法判断未完成调用是否已经写入，因此只补充“结果未知、请先查询状态”的工具错误，不自动重放。
- `shortcut:` 原子性仅覆盖当前内置 SQLite 待办和只读工具；未来接入外部写服务时，必须由对方支持持久化幂等键。本项目不支持多个进程并发修改同一 session。

## 3. 工具

### calculator

使用受限 AST 计算数字表达式，支持括号、`+`、`-`、`*`、`/`、`**`。拒绝函数调用、变量访问、过深表达式和过大结果。

### search

查询三条内置的 Agent 示例资料：Agent 循环、工具调用、会话记忆。它是题目允许的 mock，不访问互联网；返回结果会标记为本地示例资料。

### todo

支持 `add`、`list`、`complete` 三种操作。待办按用户和 session 隔离，存储在 SQLite 中，支持进程重启后继续使用。

对外返回的 `id` 是当前用户、当前 session 内按创建顺序从 1 开始的编号；其他用户或 session 的待办不占用编号。完成后编号不变，新增待办继续递增，分页不重新编号。数据库 `todos.id` 仍为内部全局主键，已有数据无需迁移；旧对话、旧 Trace 中的全局 ID 不改写，完成操作前需查询当前清单核对编号和内容。当前不支持删除待办；若增加删除功能，需将会话编号持久化以避免重排。

`list` 每页最多返回 10 条，返回 `has_more` 和 `next_after_id`；有更多结果时会明确提示未展示完整。下一页传入 `after_id=next_after_id`，默认从 0 开始。分页不会删除或修改数据库中的待办。

## 4. Session 与 Context

`user_id + session_id` 唯一确定一个窗口：同一用户使用不同 session 时，对话和待办互不可见；使用原 session 可以继续之前的对话。

### Memory 召回时机与放置方式

每次用户请求进入 `Agent.run` 时，先按 `(user_id, session_id)` 从 SQLite 读取摘要和消息历史，恢复未完成的工具结果，再追加本次用户输入。每次调用 LLM 前，`_context` 构造消息：第一条 `system` 包含系统指令和较早对话摘要，后面按原顺序放入近期 `user` / `assistant` 消息。工具调用放在 `assistant.tool_calls`，执行结果放在对应的 `tool` 消息；工具 Schema 通过请求的 `tools` 字段单独传递。

这是一种按 session 全量加载近期历史和摘要的基础召回，没有向量检索或跨 session 召回。摘要保留用户约束、关键事实、已完成操作、未完成任务和追问对象；模型思考原文只写 Trace，不进入 memory。待办当前状态通过 `todo` 查询数据库，不能把旧摘要当成实时状态。

每次调用模型前：

- 加载当前 session 的历史消息；
- 保留最近的对话原文；
- 当用户轮次超过 4 轮或 Context 接近 16,000 字符时，移出较早轮次；
- 将已有摘要和移出的完整轮次交给同一个模型，生成最多 3,000 字符的摘要，保留用户限制、关键事实、已完成操作、未完成任务和追问所需的对象 ID；不传入内部思考过程；
- 摘要请求不启用工具；成功后才持久化摘要并移除旧轮次。请求失败或摘要无效时记录 `summary_error` 并保留待压缩消息，提示用户重试；
- 待办清单和当前完成状态以数据库为准，系统提示要求模型调用 `todo` 查询，不能依据旧摘要推断。

每条工具结果进入 Context 前最多保留 4,000 字符；超长结果以有效 JSON 返回 `truncated`、原始长度、部分内容和明确提示，完整结果仍写入 Trace。当前轮累计消息超过 16,000 字符且无法继续移出旧轮次时，停止请求模型并保留历史，提示缩小任务范围，不拆散工具调用和结果。

模型摘要仍可能遗漏或误记，不能保证永久完整记忆。压缩会额外调用模型，增加延迟和费用；16,000 字符是消息预算，不等同于模型 token 上限，也不包含工具 Schema。

## 5. 运行

环境要求：Python 3.10+，无需安装第三方依赖。

先克隆并进入仓库：

```powershell
git clone https://github.com/echohzz/agent.git
cd agent
```

### PowerShell

```powershell
$env:DASHSCOPE_API_KEY = "你的百炼 API Key"
py main.py --user alice --session window-1
```

### PyCharm

在 **Run → Edit Configurations → Environment variables** 中添加：

```text
DASHSCOPE_API_KEY=你的百炼 API Key
```

然后运行 `main.py`。API Key 只从环境变量读取，不写入源码。

### 参数

```powershell
# 单次执行
py main.py --user alice --session window-1 --message "计算 (18+6)/3，并把结果记为待办"

# 指定数据库文件
py main.py --db agent-test.sqlite3 --user alice --session window-1
```

交互模式中：

- `/trace`：查看当前 session 最近 20 条事件，按一次 Agent 回合聚合展示；同一回合的用户输入、上下文摘要、模型决策、工具结果和最终回答放在同一个编号下。模型事件的 `tool_calls` 列出实际调用的工具名称、调用 ID 和参数，参数和结果分层缩进，文本中的换行正常显示；
- `/quit`：退出程序。

## 6. 测试

运行全部自动化测试：

```powershell
py -m unittest -v
```

测试覆盖：

- 直接回答和工具调用循环；
- 工具结果驱动的最终回答；
- 对话追问；
- 用户和 session 隔离；
- SQLite 重启恢复；
- 非法工具参数和非法模型响应；
- 模型摘要合并、近期轮次保留、摘要失败保护及重启恢复；
- 待办分页、下一页追问、压缩后从数据库查询当前状态；
- 超长工具结果截断与完整 Trace、单轮 Context 预算保护；
- 最大执行轮次；
- API 失败时保留用户输入；
- 工具结果、Trace 或历史保存失败时的整体回滚、中断后重启恢复、部分批次恢复和旧版本历史保护；
- 相同调用 ID 跨轮去重、跨用户回合正常复用、参数冲突拒绝和缓存结果截断；
- `finish_reason` 透传，截断回答、截断工具批次和截断摘要保护；
- calculator 安全边界；
- search 的自然问法和无结果场景。
- 百炼 API 成功响应、HTTP 401/429/500、网络异常、非法 JSON 和非法响应结构；
- CLI 缺少 API Key、单次执行、`/trace` 和退出时资源关闭。
- 真实验收脚本的单例选择、成功/失败报告、缺少 Key 时不请求 API（通过模拟适配层离线验证）。

真实 API 冒烟测试：

```powershell
py main.py --db smoke.sqlite3 --user demo --session tool-chain --message "请计算 (18+6)/3，再把结果记为待办"
```
可在 `/trace` 中看到 `calculator → todo → 最终回答` 的执行链路。

面试前可运行真实模型验收脚本：

```powershell
py smoke_real.py
```

脚本独立调用真实百炼模型，运行 10 个场景：直接回答、计算、计算后添加待办、缺少内容时追问、工具临时失败后重试、追问完成待办、用户/session 隔离、重启恢复、实际摘要后追问、本地搜索和实时信息能力边界。第 5 项只给计算工具注入一次故障，模型决策仍来自真实 API。

验收主要检查实际工具调用、计算结果、待办 ID/状态和摘要事实；少数回答检查关键词或标记，不比较完整固定文案。每例使用临时数据库，不接触 `agent.sqlite3`；全部连接关闭后清理。普通 `py -m unittest -v` 不会调用真实 API。

运行需有效额度，会产生模型调用；失败不会被当作通过，进程返回非零退出码。默认生成 `smoke-report.json`，记录模型、北京时间、每例耗时、模型请求次数、对话、工具结果及失败原因，不包含 API Key 或内部思考。

仅补测未完成项并另存报告：

```powershell
py smoke_real.py --case test_09_summary_followup test_10_search_scope --report smoke-recheck.json
```

2026-10-25 历史验证：40 项离线测试全部通过，含使用子进程强制退出后的恢复测试。当时使用 `qwen3.8-max` 的真实验收前 8 项通过；第 9 项在摘要请求时、第 10 项在首个请求时，均遇到百炼 HTTP 403（免费额度耗尽）。本地报告保留原始失败原因；这不是完整的 10/10 通过记录，也不能作为当前 `kimi-k3` 的验收结果。当前模型需要用上述脚本重新验收。生成报告包含测试对话，默认不提交 Git，可在检查后作为验收附件单独提供。

## 7. 设计取舍
- 使用百炼 OpenAI-compatible Chat Completions 接口，当前模型为 `agent/model.py` 中的 `kimi-k3`；账户需有该模型的访问权限和可用额度。
- 使用 Python 标准库实现 HTTP、SQLite、参数校验和 CLI，减少依赖，突出 Agent Runtime 本身。
- `search` 使用 mock，因为题目明确允许；它不伪装成实时搜索。
- 记录 `reasoning_content` 是为了满足题目对模型输出解析和 Trace 的要求；该字段不作为后续模型上下文，避免把内部推理当作事实记忆。

## 8. 提交材料

| 提交要求 | 对应内容 |
| --- | --- |
| 代码链接 | <https://github.com/echohzz/agent> |
| 真实 LLM API | `agent/model.py` 使用 `urllib.request` 请求百炼 Chat Completions，Key 从环境变量读取 |
| 运行方式、系统设计、memory | 本 README 第 2、4、5 节 |
| AI Prompt 与问题解决记录 | [DEVELOPMENT.md](DEVELOPMENT.md)；运行时 Prompt 见 `agent/runtime.py` |
| 测试用例 | `test_main.py`、`test_model.py`、`test_cli.py`、`test_smoke.py`；真实验收见 `smoke_real.py` |

2026-10-25 提交准备检查：`py -m unittest -v` 共 41 项离线测试全部通过。本次未重新请求真实 API；当前 `kimi-k3` 的真实验收仍需运行 `py smoke_real.py`，成功后可附检查过的 JSON 报告。
