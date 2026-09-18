# 两轮改动独立验收

日期：2026-09-19
结论：**REQUEST CHANGES / 不通过；未 commit、未 push。**

## 1. 范围与判定

- 仓库：`MindFlow/mindflow-app`，分支 `main`，本地基线 `03f5d9267e22524095dd623514d5a3f9820a5dcd`。
- 输入：`20260919-delivery.md`、`20260919-review-report.md`、`20260919-phase2-progress.md`，以及当前未提交源码和测试。
- 两轮变更尚未分别提交，因此本次验收的是它们叠加后的工作区，不声称逐提交独立回归通过。
- 第一轮：存在供应商接线、任务恢复和通知契约缺口，不能按“代码缺陷全部修复”验收。
- 第二轮：分组、权重和质量记录已有实现，但仍有正确性问题；进展文档还明确列出未交付功能，不能按完整方案验收。
- 单元测试通过不等于训练协议、真实调用链和异常恢复契约通过。下述独立探针覆盖了现有测试遗漏的情况。
- 本次未修改业务源码、未迁移生产数据库、未训练真实数据、未更换 active 模型、未调用付费模型 API、未读取或输出私有密钥。

## 2. 本次实测

后端使用已有隔离环境 `C:\Users\lenovo\AppData\Local\mindflow-audit\venv312`，并设置：

```powershell
$env:UV_PROJECT_ENVIRONMENT='C:\Users\lenovo\AppData\Local\mindflow-audit\venv312'
$env:MINDFLOW_DATA_DIR='C:\Users\lenovo\AppData\Local\mindflow-audit\acceptance-20260919'
uv run --no-sync python -m pytest tests/ -q
uv run --no-sync python -m ruff check src tests
uv run --no-sync python -m mypy --strict src/mindflow
uv lock --check --offline
```

| 检查 | 本次结果 | 边界 |
|---|---|---|
| 后端全量 pytest | **2321 passed, 4 skipped, 7 warnings，287.71s** | 4 项跳过依赖运行中的 8765 服务；未启动生产服务 |
| Ruff | All checks passed | `src tests` |
| mypy strict | 169 source files，无错误 | 不证明运行时协议正确 |
| uv lock check | 通过，115 packages | 本轮 pyproject 变化是 lint 配置，没有发现锁文件漂移 |
| 前端 lint | 0 errors，10 warnings | e2e 文件未使用变量警告 |
| 前端 build | 通过 | Vite 生产构建 |
| API drift | 3 tests passed | 生成类型一致性 |
| Playwright | **10 passed，27.5s** | `e2e/test-model-center.spec.ts`，包含 375px 无横向溢出 |
| diff check | 通过 | 生成 TS 文件存在 LF/CRLF 提示 |
| 候选文件密钥模式扫描 | 50 个文件，未命中高置信度模式 | 启发式扫描，不等于完整安全审计；未扫描忽略的私有数据 |
| GitHub 远端查询 | 失败：Connection was reset | `git ls-remote origin HEAD refs/heads/main`；没有确认远端最新 SHA |

后端 7 条警告来自健康检查测试中的未 await mock coroutine。它们没有使本次套件失败，也不应隐去。
Playwright 的 HTTP API 为 mock；控制台有 8765 WebSocket 连接拒绝，因此它不构成真实 WebSocket、通知、聊天或完整前后端联调通过的证据。
阶段二文档中的“4 failed”不是本次复验结果；本次为“4 skipped”，不直接背书其“失败与基线一致”的历史断言。

## 3. P1 阻断项

### A01. 校准隔离没有贯穿评估和部署

位置：`backend-next/src/mindflow/train/v2.py:494`、`train/pipeline.py:262`、`train/models/ensemble.py:174`。

- 评估 `clf.fit()` 没有传 `groups`。
- 部署传原始 `dates`，没有传跨午夜合并后的 `group_ids`。
- 即使传了组，随机选出的校准组不满足双类别条件时仍退回逐行拆分。
- 独立训练审查的合成探针：四次评估拟合全部收到 `groups=None`；另一个 8 组样例中，逐行回退使全部 8 组同时出现在基础训练和校准两侧。
- 这是内部校准隔离和评估/部署协议不一致问题，不应误写成已经证明外层测试标签全部泄漏。

验收修复：评估和部署使用同一合并组契约；有组信息时禁止静默逐行回退；无合法校准拆分要明确报告并遵守门控。补公共训练入口测试。

### A02. 最终校准丢弃会话均衡权重

位置：`backend-next/src/mindflow/train/models/ensemble.py:152`、`:203`。

RF/XGBoost 接收权重，sigmoid/isotonic 校准器没有权重参数。独立合成探针中，原始概率均为 0.5，正负会话各 10 行时校准输出 0.5；只把负会话复制至 100 行后输出约 0.0908965。会话均衡没有贯穿最终概率。

验收修复：把校准子集权重传到两种校准器，加入保持会话总权重不变的重复窗口测试。

### A03. 辅助标签预算在切折后失效

位置：`backend-next/src/mindflow/train/v2.py:479`、`train/grouping.py:173`。

预算依据整库显式总权重计算，评估只切片，没有按训练折重新计算。留出集的会话数量因此影响训练折的权重设置。
独立探针：A 日 2 个显式会话，B 日 18 个显式会话，C 日 10 个辅助窗口；留出 B 后训练折显式权重为 2，辅助权重为 20，超过约定的 1 倍上限。

验收修复：每折仅用训练行的来源和会话计算权重；最终训练复用相同算法，不复用包含留出统计的权重数组。

### A04. 重叠同标签会话仍可拆开跨午夜关系

位置：`backend-next/src/mindflow/train/v2.py:194`、`:223`。

日期连接只记录覆盖率胜出的会话。先输入短会话、再输入同标签跨午夜会话，覆盖率相同时短会话胜出，跨午夜会话丢失第一天的连接。
独立探针：窗口为 23:55–00:00、次日 00:05–00:10，归属为 `short/overnight`，日期块仍为两个日期。

验收修复：用于分组的所有有效会话关系与用于权重的单一归属分离，补重叠及会话输入顺序不变性测试。

### A05. 共享并发门只创建，没有接入生产调用

位置：`backend-next/src/mindflow/infrastructure/provider_registry.py:91`。

源码引用核查：`self.concurrency` 除创建外只用于描述配置，没有请求路径 acquire。面板、聊天、归因和干预并未共享限流，配置为 1 不代表真实峰值为 1。

验收修复：在实际请求边界使用同一个 gate；同时启动多个入口，用阻塞 mock 测量供应商请求峰值，而非只测 gate 类。

### A06. 归因与干预没有走新增 ECNU 协议

位置：`backend-next/src/mindflow/infrastructure/provider_registry.py:97`、`infrastructure/llm/client.py:172`、`services/intervention_service.py:519`。

结构化归因仍走 `DeepSeekClient`；干预直接使用其 HTTP pool，仍为 `max_tokens=200` 和 10 秒超时。新增 `get_attribution_model()` 没有调用方，思考配置和输出预算没有覆盖这些入口。

验收修复：统一实际 payload 构造和预算，使用 MockTransport 经生产服务入口核对参数。无需真实付费请求即可测试接线。

### A07. 训练任务持久化未接回查询接口

位置：`backend-next/src/mindflow/services/training_job_service.py:142`、`:155`、`:161`，`api/schemas.py:236`。

真实内存 SQLite 探针：已有 training 任务被更新为 interrupted，但新服务 `current_job=None`，`get_job(id)=None`。路由只查询此方法，重启后仍返回找不到任务。响应 schema 也不接受 interrupted。

验收修复：按用户恢复最新状态，并允许按 ID 查询持久记录；补重启前后 HTTP 查询测试及 interrupted schema/UI 状态。

### A08. 状态持久化乱序可以让终态回退

位置：`backend-next/src/mindflow/services/training_job_service.py:512`、`infrastructure/repositories/training_jobs.py:114`。

状态写入是未串行、未等待完成的 create_task，仓储更新也没有终态保护。
独立延迟仓储探针输出：

```text
PERSIST writes=['succeeded', 'training'] stored=training memory=succeeded
```

验收修复：串行化或使用带序号/条件的单调状态更新，终态提交及关闭需可等待；补乱序与关闭测试。

## 4. P2 功能与数据正确性问题

### A09. ECNU 公共 chat 入口丢失 JSON 模式

位置：`backend-next/src/mindflow/agents/llm_gateway.py:326`、`:251`。

`complete(model="chat")` 先把逻辑档位映射成 `ecnu-max`，而 `_get_model()` 仅用 `deepseek-chat` 判断 JSON 档。
本次 mock 公共入口探针输出 `actual_json_modes=[False]`。已有直接调用 `_get_model("deepseek-chat")` 的测试绕过了错误。
修复需保留逻辑档位和供应商模型 ID 两个概念。

### A10. 质量覆盖时长未裁剪和去重

位置：`backend-next/src/mindflow/services/window_quality.py:170`。

直接累加完整 duration 再截到窗口长度，没有计算事件与窗口交集，也没有合并重叠区间。
本次探针：事件从窗口前 290 秒开始、持续 300 秒，只覆盖当前窗 10 秒，却返回 observed=300、coverage=1.0。
修复需按时间交集及区间并集计算活动、输入和浏览器覆盖；增加跨窗、重叠、重复事件测试。

### A11. 持续运行的采集器会被判为关闭

位置：`backend-next/src/mindflow/services/telemetry_service.py:372`；被调用方法为 `infrastructure/repositories/collector_intervals.py:177`。

调用方需要“与范围重叠”的采集区间，而现有查询只返回“在范围内启动”的区间。
本次真实内存 SQLite 探针：采集器 09:00 开启且未结束，查询 10:00–10:05，活动 enabled/available 都为 False。
此外区间查询失败时默认 activity=True；browser 用有无记录代替开关，input 则直接设为 True，整段状态又复用于所有窗口。这些信息尚不能可靠区分关闭、故障、真零和恢复。

修复需独立的重叠查询与每窗状态，使用真实开关和心跳/采集证据；不能把业务事件时长直接当成全部采集在线时长。

### A12. 通知返回值不等于实际显示

位置：`backend-next/src/mindflow/services/intervention_service.py:1053`、`frontend/src/realtime.ts:132`。

`LogOnlyNotifier.send()` 返回 True，但没有显示通知；浏览器因此被抑制。Windows 交互弹窗失败、普通 toast 成功时返回 False，浏览器又可能重复显示。
这是静态调用链确认，未在真实桌面触发通知。
修复需区分“实际显示”与“提供响应按钮”，并覆盖三类 notifier 分支。

### A13. 新超时预算不覆盖完整工作流

位置：`frontend/src/api.ts:679`、`:709`；`backend-next/src/mindflow/graph/analysis_graph.py:229`、`:1434`。

- 聊天包含模型、工具、再次生成的循环，不是注释中的单次生成；单次模型上限 180 秒不能推出整轮 200 秒足够。
- 手动干预外层 660 秒，内部 `requestOptions()` 仍按默认 30 秒 abort。
- 分析 600 秒仅包住 panel 子图；晚发生的 PanelUnavailableError 还可能执行额外回退。

这些是代码路径和预算不匹配，不是本次实测了长达数分钟的在线请求。
修复需端到端 deadline、剩余预算及取消语义，使用假时钟覆盖工具循环和晚失败回退。

### A14. 响应去重后的 API 返回值不是数据库权威值

位置：`backend-next/src/mindflow/api/routes/intervention.py:141`。

仓储保留首个人工响应后，路由仍返回本次 `body.response`。先 accepted、后 ignored 时，第二次返回值与历史记录不一致。
修复需从服务返回的权威记录构造响应，补 HTTP 重复提交测试。

### A15. 版本报告先写盘，来源字段后赋值

位置：`backend-next/src/mindflow/train/pipeline.py:345`、`:351`。

独立临时目录探针：12 条显式样本，版本报告来源为 `{}`，共享报告为 `{"explicit":12}`。
冲突和歧义计数有同样的赋值时序问题。修复需在任何序列化之前完成字段赋值，并比较版本报告、共享报告和返回对象。

## 5. 其余边界与改进

- 发布异常回滚不完整：`training_job_service.py:477`、`:567`。故障注入令第二个服务 attach 失败后，磁盘回滚到 previous，但 prediction=candidate、telemetry=previous。没有旧版本时也不撤销新指针。注意当前两个 attach 方法仅赋值，该探针验证异常处理契约，不证明正常发布必然失败。
- 显式 `provider="generic"` 仍被 Registry/Gateway 的 ECNU URL/model 启发式覆盖，应让显式配置优先。
- `chat_graph.py:542` 用非空工具文本判断证据存在，错误 JSON 或“暂无数据”也会标成有证据，应使用结构化成功及有效数据状态。
- 跨轮聊天持久化只保留最终答案，不保留完整工具协议历史。新增适配器的单轮 reasoning 往返不能据此宣称跨轮协议已经接通；是否必须保存 reasoning 应按供应商实际契约决定，不能直接暴露到展示接口。
- 阶段二明确未交付：上下文 API/UI、候选片段选择器、每日限额及随机抽样、V4 候选、前向时间留出、扩展就绪度。这些不必全部实现才允许未来的“明确缩小范围的阶段交付”，但本次不能把完整原方案标为完成。
- E0/E1 结果取自交付文档，本次未重跑真实数据实验。E1b BA=0.2598 且质量门未过，因此不能声称模型准确率已有提升；也不能由这些有限实验断言行为信号普遍不可迁移或特征改进必然无效。
- 一次 low/max reasoning token 差异不是完整的服务端参数生效证明；应区分发送参数、服务端明确确认、统计性行为证据。
- 文档中的破坏式 checkout/rm 回退示例不适用于当前两轮共享的脏工作区，本次未执行。

## 6. 下一轮验收门槛

1. 修复 A01–A08，保留合成复现为自动化回归；不能通过降低质量门掩盖问题。
2. 修复 A09–A15，并补真实调用链级 mock、HTTP 路由级和重启级测试。
3. 明确阶段二继续完成还是缩小交付范围；未做项必须保留为未完成。
4. 重跑本报告检查；补聊天、干预、通知及重启恢复的前后端联调，不用模型中心 mock E2E 代替。
5. 校准与权重协议修正后重新生成离线证据，保持旧 active 不变；候选过门之前仍为 shadow。
6. 验收通过后重新查询远端 SHA，检查差异、密钥与文件清单，再进行提交和 push。远端连接故障是额外障碍，不是本次拒绝发布的主要原因。

本次只新增此验收文档。两轮业务改动完整保留，未替用户撤销、覆盖或发布。
