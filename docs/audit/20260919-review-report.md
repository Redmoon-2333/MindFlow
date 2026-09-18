# MindFlow 审核报告（2026-09-19）

> 历史阶段记录。后续独立验收发现校准、权重、供应商接线和任务恢复等遗漏，
> 因此下文当时的通过声明不能代替当前发布门槛。见
> [独立验收](20260919-independent-acceptance.md)及[修复复验记录](20260919-remediation.md)。

冻结基线：`03f5d9267e22524095dd623514d5a3f9820a5dcd`（2026-08-28 02:16:51 +0800，工作区干净）
仓库：`D:\大学相关\01_学业与课程\07_双创\MindFlow\mindflow-app`（外层目录未做任何操作）
远端：检查时 `git ls-remote` 因网络重置不可达；本地 HEAD 与交接记录一致，未执行 pull。

> 本报告只记录**已在本轮实际复现或修复**的问题。凡未验证的事项在文中显式标注。

---

## 0. 结论摘要

| 类别 | 结论 |
|---|---|
| 代码修复验证通过 | 标签来源分离、冲突排除、覆盖率门限、校准隔离、发布原子性与试加载、响应原子性、ECNU 适配与思考档位、前端 204/契约/竞态 |
| 离线效果提升 | **未达成**，见 §3——修正评估口径后模型显著劣于随机，质量门不通过 |
| 真实长期体验 | **未验证**（本轮无长期真实使用数据） |

最重要的一条结论：**正式报告中的 BA=0.664 不可作为部署依据**。它是在把窗口标签混入"显式反馈"评估掩码后得到的；按显式反馈单独评估时，同一模型 BA=0.359、AUC=0.322（劣于随机）。

---

## 1. 环境与基线（Phase A）

### 1.1 测试环境

前序 pytest 无法启动的根因是旧 `.venv`（Python 3.11.15，`pyvenv.cfg` 指向旧 conda 环境，editable `.pth` 含中文路径）。本轮**未改动旧环境**，改用独立审核环境：

- 解释器：uv 管理的 CPython 3.12.12（`C:\Users\lenovo\AppData\Local\mindflow-audit\venv312`）
- 依赖：`uv sync --extra dev --extra ml`（`uv.lock` 未修改）
- 隔离：`UV_PROJECT_ENVIRONMENT` 指向仓库外目录；测试数据目录指向 `C:\Users\lenovo\AppData\Local\mindflow-audit\baseline-data`

### 1.2 基线门禁

| 门禁 | 基线结果 | 修复后结果 |
|---|---|---|
| Ruff | `All checks passed!` | `All checks passed!` |
| Mypy strict | 未在基线跑（环境未就绪） | `Success: no issues found in 166 source files` |
| Pytest | **2246 passed, 4 skipped, 7 warnings**（252.63s） | **2306 passed, 4 skipped, 7 warnings**（347.96s） |

>`2250` 这一数字在本轮**未被当作本轮通过证据**，它是新环境重新跑出的真实计数。

### 1.3 数据快照

真实库用 SQLite backup API 取一致性快照（非文件复制，规避 WAL 写入中状态）：

- 源：`C:\Users\lenovo\AppData\Local\mindflow\mindflow\mindflow.db`
- 快照：`mindflow-audit\snapshot\mindflow_20260919.db`（历史记录为 62,566,400 bytes；原文所记哈希不足 64 位，不能作为完整 SHA-256 校验值，待重新核验）
- 规模：activity_events 45,275；feature_windows 9,924（v3: 8,198 / v2: 1,726）；focus_sessions 507；focus_session_feedback 37

实验产物写入 `backend-next/data/experiments/20260919_audit_repro/`（`data/` 已被 gitignore，不进版本库）。

---

## 2. 标签与训练完整性（Phase D）——核心缺陷

### 2.1 【严重】窗口标签污染显式评估

**位置**：修复前 `src/mindflow/train/v2.py:156-162`

窗口标签样本被标记为 `explicit_mask=True`，与真实用户反馈进入同一个评估掩码。`evaluate_v2_candidates()` 与部署共用该掩码（`pipeline.py:236`），因此质量门考核的是"混合标签上的表现"。

**复现证据**（`data/experiments/20260919_audit_repro/repro_explicit_mask.json`）：

```
explicit_mask_true_count            3291
label_sources_inside_explicit_mask  {window_label: 2688, explicit: 603}
contamination_ratio                 0.8168   ← 81.7% 的评估样本不是用户反馈
```

正式报告自述 `explicit_feedback_count=32`，但其候选指标由 **3,395** 个样本算出（混淆矩阵 955+2440），与 32 个会话自相矛盾——该数字本身就是污染的证据。

**消融实验**（`label_ablation.json`，同一快照、同一日期折）：

| 变体 | 训练样本 | 评估样本 | BA | minority F1 | Brier | AUC | 折稳定 |
|---|---|---|---|---|---|---|---|
| A 显式训练/显式评估 | 567 | 567 | 0.4067 | 0.3399 | 0.4325 | 0.2965 | ✗ |
| B 全量训练/显式评估（当前修复） | 3255 | 567 | 0.3594 | 0.4566 | 0.4919 | 0.3216 | ✗ |
| C 仅显式（无辅助标签） | 567 | 567 | 0.4067 | 0.3399 | 0.4325 | 0.2965 | ✗ |
| D 仅辅助标签训练/显式评估 | 2688 | 567 | 0.5498 | 0.6658 | 0.3678 | 0.6101 | ✗ |

**结论**：在诚实口径下，现有模型（A/B/C/D 全部）质量门均不通过。历史 BA=0.664 来自把辅助标签当作评估真值，不能作为部署依据。**已按规范保持 shadow，未撤下既有生产模型用于新候选失败**。

### 2.2 【中】反馈重叠匹配无覆盖率要求

**位置**：修复前 `v2.py:134-139` — 任一秒级重叠即建立标签，且 `break` 取第一个匹配。

已改为：覆盖率 ≥ 50%（`MIN_WINDOW_COVERAGE`，`v2.py:89`）才建立标签，否则不参与（`v2.py:161`）。回归测试 `test_partial_overlap_below_threshold_does_not_label` / `test_sufficient_overlap_labels_window`。

### 2.3 【中】mixed 反馈回退到弱标签

**位置**：修复前 `v2.py:156-178` — `mixed`（label=None）不满足 `matched_label is not None`，因而落入 `window_labels` 或 `_weak_label` 分支。

已改为：mixed 命中即显式判定为"无标签"并**阻断回退**（`v2.py:227-230`，`source="feedback_mixed"`），同时记录 `ambiguous_window_count`。回归测试 `test_mixed_feedback_does_not_fall_back_to_weaker_sources`。

### 2.4 【中】同窗相反标签被静默裁决

**位置**：修复前 `v2.py:134-139` 的 `break` — 两个会话覆盖同一窗口时取先到者。

已改为：收集全部达标会话，标签冲突则**排除该窗口**（`v2.py:200-220`，`source="feedback_conflict"`，`conflict_window_count`），既不裁决也不回退。回归测试 `test_contradictory_feedback_is_excluded_not_arbitrated`。

### 2.5 【中】session_ids 存的是窗口 ID

`V2TrainingData.session_ids` 实际是特征窗口 ID（`v2.py:241` 原 `sid_list.append(wid)`）。本轮保留该字段语义（调用方用它对齐行），但**显式区分**：`explicit_feedback_count` 只统计真实反馈会话（`v2.py:198-204` 的 `explicit_session_ids`），并在文档与测试中写明两者不同（`test_explicit_sample_dates_come_from_session_start`）。

### 2.6 【严重】校准拆分随机且 scaler 泄漏

**位置**：修复前 `train/models/ensemble.py:106`（`fit_transform` 用全量）与 `:118-123`（`train_test_split` 按行随机）。

- scaler 在拆分**之前**用全量拟合 → 校准留出集的均值/方差泄漏进基准模型；
- 按行随机拆分 → 同一天的窗口可能同时出现在两侧，日期分组失效。

已改为（`ensemble.py:120-133`、`:158-200`）：
- scaler **只用基准训练侧**拟合；
- 新增 `groups` 参数，≥5 个组时按**整组**拆校准留出（日期互斥），并校验两侧均含两类；不足时回退按行分层。
- `ModelManager.train_all` 透传 `groups`（`manager.py:119-144`），pipeline 用真实日期构造（`pipeline.py:245-247`）。

回归测试：`test_scaler_is_fit_on_base_training_split_only`、`test_calibration_split_is_group_disjoint`、`test_grouped_split_keeps_both_classes_on_each_side`、`test_calibration_disabled_uses_all_rows_for_scaler`。

### 2.7 评估改为"显式反馈单独计分"

`evaluate_v2_candidates` 现在只对显式反馈样本计分，但在每折内用**全部监督样本**训练（排除被留出日期），辅助标签结果由 `evaluate_auxiliary_signal`（`v2.py:485`）单独报告，不参与质量门。`train_mask` / `window_label_mask` 为新增字段（`v2.py:70-76`）。

---

## 3. 模型发布与任务状态（Phase C）

### 3.1 【严重】manifest / training_report 共用一个文件

**位置**：修复前 `train/models/manager.py:283`、`pipeline.py:293`

每次训练覆盖同一个 `manifest.json` / `training_report.json`，导致"版本与指标"的对应关系在第二次训练后即丢失。

已改为：每次写 `manifest-<tag>.json` 与 `training_report-<tag>.json`（`manager.py:315`、`pipeline.py:337-345`），共享文件仅作为最新写入的便利副本保留。回归测试 `test_per_version_manifest_is_written`。

### 3.2 【严重】latest 指针非原子写

**位置**：修复前 `manager.py:306`（直接 `write_text`）

已改为 `_atomic_write_text`（`manager.py:337-346`，临时文件 + `os.replace`），`latest.json` 与 manifest 均走该路径。回归测试 `test_latest_pointer_is_valid_json_and_matches_version`、`test_atomic_write_leaves_no_temp_file`。

### 3.3 【严重】激活前不验证可加载性

**位置**：修复前 `manager.py:287-295` — 直接写指针，从不试加载。

已改为：`activate=True` 时先 `_verify_loadable`（`manager.py:348-368`，用独立的探针实例走同一套签名校验与 `joblib.load` 路径），失败抛 `ModelPublicationError`（`manager.py:52`）且**不移动指针**。回归测试 `test_activation_refuses_when_artifacts_fail_trial_load`、`test_shadow_save_does_not_move_pointer`、`test_saved_models_are_reloadable`。

### 3.4 【严重】磁盘已激活但运行时发布失败

**位置**：`services/training_job_service.py:402-411`（修复前）

训练线程内 `save_all(activate=True)` 已改磁盘指针；随后 `_refresh_ready_manager` 若抛错，进程内存仍持旧模型，而磁盘广告新版本。

已改为：
- 发布失败时 `_rollback_activation`（`training_job_service.py:428-465`）把指针回滚到原激活版本，使磁盘与内存一致，候选制品保留在磁盘仅供检查；
- pipeline 捕获 `ModelPublicationError` 后降级为 shadow 并记录 `activation_error`（`pipeline.py:322-336`），不再留下"失败但已激活"的中间态。

### 3.5 【中】训练任务状态仅内存保存

**位置**：修复前 `training_job_service.py:135` — `self._current: _JobState | None`

已改为持久化：
- 迁移 `0024_create_training_jobs.py` + `schema.py` 的 `training_jobs` 表；
- 新仓库 `infrastructure/repositories/training_jobs.py`；
- `TrainingJobService` 在每次状态迁移时写入（`training_job_service.py:457-470`、`:472-505`），启动时 `recover_after_restart()`（`:139-148`）把非终止态标记为 `interrupted`，由 `app.py:785-794` 在启动阶段调用。

回归测试：`test_training_job_lifecycle_round_trips`、`test_restart_marks_running_jobs_interrupted`、`test_restart_recovery_is_idempotent`、`test_latest_job_returns_most_recent`。

---

## 4. 干预与交互（Phase C）

### 4.1 【严重】响应无条件覆盖

**位置**：修复前 `infrastructure/repositories/intervention.py:149-154` — 无条件 UPDATE。

修复后语义（`intervention.py:133-208`）：

| 场景 | 行为 |
|---|---|
| 首次人工响应 | 写入 |
| 重复人工响应 | **保留首次**，回读返回既有状态 |
| auto 超时默认值 | **只能填空槽**，绝不覆盖人工答案 |
| 人工响应在 auto 之后到达 | 允许覆盖 auto 默认值 |

新增 `response_source` 列（迁移 `0025_add_response_source.py`，`schema.py`），桌面弹窗在超时路径标记 `source="auto"`（`intervention_popup.py:153-176`），API 请求新增可选 `source` 字段（`api/schemas.py:77-84`，默认 `human`，向后兼容）。

回归测试（12 项，`tests/test_intervention_response_atomicity.py`）覆盖上表全部四种情形。

### 4.2 【中】"无效"实际提交 annoying

**位置**：修复前 `frontend/src/pages/Intervention.tsx:350` — 选项文案"无效"对应值 `annoying`，而 `annoying` 参与疲劳节流（`config.py:237`），语义错位会误伤节流。

已改为文案"打扰（令人烦扰）"，与后端枚举语义一致（`Intervention.tsx:350`）。

### 4.3 【中】评价草稿跨记录串用

**位置**：`Intervention.tsx:142-167`

已改为：提交前捕获 `submittedFor`；响应只在"用户仍停留在同一记录"时才清空草稿；打开另一条记录时先重置草稿（`Intervention.tsx:397-406`）。

### 4.4 【中】skipped 与原因未展示

**位置**：修复前 `Intervention.tsx:113-124` 丢弃 `result.skipped` / `skip_reason`。

已改为新增 `skipNotice` 状态与提示卡片（`Intervention.tsx:80`、`:208-216`），跳过原因可见。

### 4.5 【中】Focus 反馈草稿未回填

**位置**：修复前 `Focus.tsx:119` 默认 `{label:"mixed", score:3}`

已改为 `draftFor(session)`（`Focus.tsx:113-135`）从 `feedback_label` / `feedback_score` / `feedback_task_type` 回填。同时在 `api.ts` 的 `FocusSession` 接口补上这三个后端已返回的字段（`api.ts:196-202`）——此前它们不在类型里，页面用 `"feedback_label" in session` 绕开类型，属于契约缺失。

### 4.6 【中】204 响应被当 JSON 解析

**位置**：修复前 `api.ts:567` — 无条件 `res.json()`

已改为 `parseJsonBody`（`api.ts:578-590`）：204/205 与空体直接返回 `undefined`，非空但非法 JSON 才报错。删除成功后不再误报失败。

### 4.7 【中】请求超时叠加

**位置**：修复前 `api.ts:536` — AI 超时 90s，但内层 `requestOptions()` 用 30s 默认值，实际 30s 中止。

已改为：`AI_REQUEST_TIMEOUT_MS = 660_000`（对应交互工作流 600s 预算 + 余量）、`CHAT_REQUEST_TIMEOUT_MS = 200_000`（单次生成 180s + 余量），聊天内外层统一使用同一预算（`api.ts:544-564`、`:678-691`）。后端 `_PANEL_WORKFLOW_TIMEOUT_S` 由 120s 提升到 600s（`graph/analysis_graph.py:230`），与前端预算对齐。

### 4.8 【中】聊天新对话不隔离旧回包

**位置**：`Chat.tsx:88-116`

原守卫 `targetSessionId !== activeSessionIdRef.current` 在"新对话 → 新对话"时两侧同为 `null`，迟到的回包会落进新会话。已新增会话世代计数器 `conversationGenRef`（`:88-97`、`:109-114`），切换会话或新建对话时递增，回包需同时通过世代与 ID 校验。

### 4.9 【中】Socket 关闭回调破坏新连接

**位置**：`realtime.ts:81-96`

旧 socket 的 `onclose` 会无条件 `clearTimers()`，即使新连接已建立——这会杀掉新连接的心跳并触发重复重连。已改为：非当前 socket 的关闭事件直接返回（`realtime.ts:87`）。

### 4.10 【中】浏览器与原生重复通知

**位置**：`intervention_service.py`（发送顺序）、`realtime.ts:126-128`

已改为：先发原生通知，再把 `native_delivered` 结果放进 WebSocket 帧（`intervention_service.py:1046-1067`、`:1200-1216`）；前端在 `native_delivered === true` 时跳过浏览器通知（`realtime.ts:129-132`）。原生失败时才由浏览器兜底。

### 4.11 【中】未知应用契约不符

**位置**：修复前 `Settings.tsx:76` 按 `string[]` 渲染，后端返回对象数组（`app_classification.py:115-165`）。

已改为 `UnknownApp` 类型与 `normaliseUnknownApps`（`Settings.tsx:57-88`），展示 `process_name` 与出现次数，填入表单时使用真实进程名；同时容忍纯字符串的历史形态。

### 4.12 【中】工作态分类忽略用户规则

**位置**：修复前 `intervention_service.py:187-206` 直接用内置 `AppClassifier`。

已改为 `_classify_with_user_rules`（`intervention_service.py:219-241`）优先匹配用户规则，再由 `_is_work_category_context` 使用（`intervention_service.py:187-217`），`_load_user_rules`（`intervention_service.py:709-728`）从仓库读取；`app.py:577` 注入 `classification_repo=classification_rules_repository`。用户显式标注为工作的应用在抑制门中生效，与产品承诺一致。

### 4.13 【中】evidence_cited 在工具"被提出"时就置真

**位置**：修复前 `graph/chat_graph.py:403-404`

已改为：仅当工具**实际返回非空结果**且属于证据类工具时才置 `evidence_cited`（`chat_graph.py:540-548`），并把该值显式回写 state（此前 tool_execution_node 不返回该字段，值会丢失）。

---

## 5. ECNU 供应商迁移（Phase B）

### 5.1 官方文档复核（2026-09-19 抓取正文）

- `ecnu-max` 支持 `low/high/max`；思考模式**默认关闭**，需 `thinking:{"type":"enabled"}` 显式开启；
- 工具调用后 `reasoning_content` **必须在后续所有轮次完整回传**，否则可能 400；
- 单用户单模型并发上限 **3**（文档 2026-09-18 更新）；
- 底层模型为 `DeepSeek-V4-Flash-0731`（文档口径，本报告不据此假定能力等价）。

### 5.2 兼容性实测（真实请求，持久化预算）

预算文件：`mindflow-audit\ecnu_budget.json`，累计 **60/200**（兼容探测 8、适配器端到端 3、配对提示 48、配置核验 1），**首轮 200 次上限未超**，恢复执行共享同一计数。

| 探测 | 结果 |
|---|---|
| 认证 + `thinking` + `reasoning_effort=max` | HTTP 200，`has_reasoning=true`，usage 含 `reasoning_tokens: 26` |
| `response_format: json_object` | 200，返回合法 JSON |
| 工具调用首轮 | 200，`finish_reason=tool_calls`，返回 `call_...` |
| 工具次轮回传 `reasoning_content` | 200，正常给出总结 |
| 工具次轮**不**回传 `reasoning_content` | **200**（未复现文档所述 400） |

> 说明：文档称部分模型不回传会 400。本次 `ecnu-max` 未触发。适配器仍按协议回传（这是文档的正式要求，且在服务端模型/路由变化时是安全侧）。

### 5.3 `max` 是否真的生效（不只 HTTP 200）

同一推理密集型提示分别以 `low` 与 `max` 请求，比较服务端计费口径的推理 token：

| 档位 | HTTP | 延迟 | reasoning_tokens | reasoning 字符数 |
|---|---|---|---|---|
| low | 200 | 41.3s | 9,375 | 33,476 |
| max | 200 | 67.1s | **15,133** | 56,139 |

`max` 的推理 token 比 `low` 高 **61%**，延迟高 63%。这是服务端确实按档位调整算力的可观测证据，**不是**仅凭 200 的推断。所有实际生成请求均携带 `reasoning_effort=max`，`reasoning_effort_downgraded=false`。

### 5.4 实现

新增 `infrastructure/llm/ecnu.py`：
- `ECNUChatModel` 继承 `ChatOpenAI`，仅覆写 `_get_request_payload` 与 `_create_chat_result`；
- `thinking` 走 `extra_body`（OpenAI SDK 不接受未知顶层参数，实测必须先入 `extra_body` 才能发出）；
- `reasoning_content` 存于 `AIMessage.additional_kwargs` 并在后续轮次回注到 wire message（langchain-openai 默认丢弃该字段）；
- `resolve_effort` / `supported_efforts` 按模型校验档位，降级**必须显式上报**（`last_downgraded`）；
- 输出上限 `max_completion_tokens=16384`（langchain-openai 的 `max_tokens` 别名）。

新增 `infrastructure/llm/concurrency.py`：全入口共享 `LLMConcurrencyGate`，默认 1（`:44-100`）；实测串行序为 `a-start, a-end, b-start, b-end`。

`ProviderRegistry` 统一面板/聊天/归因/干预文案的模型配置（`provider_registry.py:72-232`），并新增 `describe()` 输出**不含密钥**的供应商证据。

重试策略：401/403/400/422 等**不盲重试**（`llm_gateway.py:118-166`），仅 429/5xx/网络错误重试一次；日志经 `_scrub`（`:169-183`）屏蔽疑似密钥。

### 5.4.1 迁移过程中自查发现并修复的两个问题

这两项由本轮**真实调用**（而非单元测试）暴露，属于自己引入后自己修掉的问题，如实记录：

1. **ECNU 路径丢失 JSON 输出模式**（严重，会破坏面板）。第一版适配器把两个模型档位都指向同一个 `ECNUChatModel` 实例且不带 `response_format`，而原 DeepSeek 的 `chat` 档位一直带 `json_object`。面板每个专家都期望 JSON，实测该配置下模型可能返回散文，编排器解析失败。修复：`chat` 档位恢复 `response_format={"type":"json_object"}`（兼容探测已证明该平台在开启思考时支持 JSON 模式），`reasoner` 档位保持不带；两档使用**各自独立**的缓存实例（`llm_gateway.py:220-260`），否则先调用的一档会决定后续所有请求的模式。回归测试 `test_gateway_json_mode_routes_per_tier`。修复后真实面板路径调用返回合法 JSON（字段齐全）。

2. **显式 base_url 被环境里的 ECNU 模型名劫持**（中）。`_is_ecnu` 同时看模型名，导致显式指定其它主机的网关只要环境里 `MINDFLOW_LLM__MODEL=ecnu-max` 就仍按 ECNU 协议发送 `thinking`。修复：显式传入 `base_url` 时只凭该 URL 判定供应商（`llm_gateway.py:207-224`）。回归测试 `test_explicit_base_url_overrides_ambient_ecnu_settings`。

> 第 1 项说明：**单元测试全绿不等于集成可用**。该问题在 2303 项测试全过的情况下依然存在，只有真实调用才暴露出来。

### 5.5 实际配置位置

应用从 **platformdirs 用户数据目录**加载：`C:\Users\lenovo\AppData\Local\mindflow\mindflow\.env`（由 `config.py:331-348` 的 `get_settings()` 读取）。该文件已写入 ECNU 配置；旧配置备份为 `mindflow-audit\env-backup-20260919.bak`（按秘密管理）。密钥**未**进入源码、测试、命令回显、报告或 Git。

配置解析与真实生成均已验证（`mindflow-audit\ecnu_probe\config_verification.json`）：

```json
{"provider":"ecnu","model":"ecnu-max","base_url_host":"chat.ecnu.edu.cn",
 "thinking_enabled":true,"reasoning_effort_requested":"max",
 "reasoning_effort_effective":"max","reasoning_effort_downgraded":false,
 "max_output_tokens":16384,"concurrency_limit":1}
```

模型类为 `ECNUChatModel`，`has_reasoning=true`。Ollama/规则降级链保留；未自动回调旧 DeepSeek 服务。

### 5.6 提示对照（同模型同档位）

12 个场景（分屏学习、长时阅读、会议、协作、休息、分心、缺数据、工具失败、短时切回、夜间工作、碎片化、稳定深度），两臂均用 `ecnu-max` + `thinking` + `max`，**因此差异只能归因于提示本身**，不包含换模型收益。

| 指标 | 基线提示 | 修订提示 |
|---|---|---|
| HTTP 200 | 12/12 | 12/12 |
| 合法 JSON | 12/12 | 12/12 |
| 结构完整 | 12/12 | 12/12 |
| 引用真实数字 | 11/12 | 11/12 |
| 医疗违禁词 | 0 | 0 |
| 平均延迟 | 3.45s | 9.78s |
| 平均完成 token | 732 | 2,530 |
| 缺数据时明确弃权 | — | 1/1 |
| 工具失败时标注不可用 | — | 1/1 |

干预判定对比（基线无该字段，按是否给出 `cbt_technique` 推断）：

| 场景 | 基线 | 修订 |
|---|---|---|
| S05 主动休息 | 建议干预 | 弃权 |
| S06 明显分心（22 次切换） | 建议干预 | **建议干预** |
| S07 缺数据 | 建议干预 | 弃权 |
| S11 碎片化 | 建议干预 | **建议干预** |
| S12 稳定深度 | 建议干预 | 弃权 |

基线在 **12/12** 场景都给出干预建议（含"主动休息"和"缺数据"），修订提示只对确有行为证据的 S06/S11 建议干预，对其余弃权。这正是"保守提醒"目标所需的方向性证据。

> **首次修订版本更差，已如实保留**：初版修订提示在 12/12 场景全部弃权，包括 22 次切换的明显分心场景——它把"缺少用户标注"误当作证据不足。该结果保留在 `prompt_eval\paired_prompt_results.json` 的上一轮记录与本节说明中；修正后才得到上表结果。
>
> 未做、也不能做：把该对照写成"线上误提醒率下降 20%"。12 个合成场景不构成误提醒率的度量，真实长期效果未验证。

### 5.7 预算与并发纪律

- 首轮上限 200 次 HTTP 请求，已用 60 次，**未把"一个面板"计为一次**（按实际子调用计数）；
- 并发 1（配置项 `max_concurrent_requests`，未在无证据情况下提高）；
- 每次生成显式输出上限 16384，并记录实际 usage、耗时与截断情况；
- 未伪造金额：ECNU 有 credits 计价口径，但本轮无账号额度依据，故不报金额。

---

## 6. 数据库与迁移

| 迁移 | 内容 | 验证 |
|---|---|---|
| `0024_create_training_jobs` | 训练任务持久化表 | 迁移链实测通过（`upgrade head` 输出含该步） |
| `0025_add_response_source` | `intervention_logs.response_source` | 同上；downgrade 用 `batch_alter_table` 兼容 SQLite |

迁移只在临时数据库上验证（`tmp_path`），**未对正式库执行**。正式库升级仍需按 AGENTS.md 要求先 `sqlite3 mindflow.db ".backup ..."`。

前端类型已重新生成并校验无漂移：`npm run generate:api` → `npm run check:api-drift` → `3 tests passed`。

---

## 7. 测试与门禁结果

### 7.1 后端

```
uv run python -m ruff check src tests          → All checks passed!
uv run python -m mypy --strict src/mindflow    → Success: no issues found in 166 source files
uv run python -m pytest tests/ -q              → 2306 passed, 4 skipped, 7 warnings in 347.96s
```

新增回归测试 **54 项**：
- `tests/test_ecnu_adapter.py`（26）：思考字段、输出上限、温度抑制、`reasoning_content` 往返、档位校验与降级上报、供应商识别、JSON 模式分层、显式 base_url 优先、并发门。
- `tests/test_label_provenance_and_publication.py`（16）：辅助标签不入评估掩码、覆盖率门限、mixed 不回退、冲突排除、校准隔离与分组拆分、逐版本 manifest、原子写、试加载拒激活、shadow 不动指针。
- `tests/test_intervention_response_atomicity.py`（12）：响应原子性四情形 + 任务持久化与重启恢复。

另有 6 个既有测试被**修正为断言正确行为**（它们原先断言的是上述缺陷）：
`test_train_v2.py`（2）、`test_collector_intervals.py`（迁移头不再硬编码）、`test_provider_registry.py`（2）、`test_notification*.py`、`test_intervention_service.py`。

### 7.2 前端

```
npm run build           → ✓ built in 207ms
npm run lint            → 0 errors, 10 warnings（全部位于既有 e2e/*.spec.ts）
npm run check:api-drift → 3 tests passed
```

### 7.3 未完成项（如实记录）

- **Playwright 未运行**。桌面与 375px 的视觉验收本轮**没有执行**，因此不存在"无溢出遮挡"的截图证据。前序也未完成。
- **前端新增竞态的自动化测试未补**：`Chat.tsx` 世代守卫、`realtime.ts` socket 身份守卫、`notifications` 去重、`Intervention`/`Focus` 草稿逻辑均已按源码修复并构建通过，但没有对应测试文件，只有人工代码审查。
- **mypy 基线未采集**：环境就绪前基线无法运行，故无法给出"与基线一致"的对比，只能给出当前为 0 错误。
- **未做云端训练、未微调大模型、未自动发布、未 commit / push / PR**。

---

## 8. 遗留问题与风险

1. **模型不可上线**（最高优先级）。诚实口径下四个变体质量门全不通过，折稳定性全部为 false。在拿到更多真实反馈日之前，ML 只能 shadow。生产模型保持原状。
2. **真实反馈量偏少**：37 条反馈、32 个会话、10 个反馈日、567 个显式窗口。按 `v2.py:383-391` 的门槛（≥7 日、≥20 会话、每类 ≥5）刚好达标，但样本量下 fold 稳定性不足以通过。**不应通过降低门限来"达标"**。
3. **辅助标签分布与真实反馈差异大**：辅助窗口标签 focus 率 76.5%，显式反馈 focus 率 47.6%。这是 D 变体（仅辅助标签训练）在显式评估上反而最高的原因，也说明辅助标签与用户真实判断并不同分布——进一步支持"不并入评估"的做法。
4. **多用户假设**：本轮所有路径固定 `user_id=1`，未做多用户验证。
5. **ECNU 单点**：并发上限 1 是为稳妥起见；文档允许 3。提高并发需要先测限流行为，本轮未做（预算与风险控制）。
6. **`reasoning_content` 的隐私面**：已存于后端 `additional_kwargs` 且不写日志、不出 API；但若未来引入会话持久化，需确认它不被落盘到聊天记录表。

---

## 9. 回滚步骤

两轮变更共用工作区，不能直接丢弃全部源码或删除新增文件作为单轮回滚。
先核对目标提交、备份及未提交改动，再对明确的提交执行可审查的 revert，
保留其他人的工作。配置恢复需确认当前配置，数据库降级需在副本验证并备份。
模型回滚需要验证旧工件且同步运行时引用，不能只修改磁盘指针。

---

## 10. 变更文件清单

**新增（8）**
```
backend-next/alembic/versions/0024_create_training_jobs.py
backend-next/alembic/versions/0025_add_response_source.py
backend-next/src/mindflow/infrastructure/llm/concurrency.py
backend-next/src/mindflow/infrastructure/llm/ecnu.py
backend-next/src/mindflow/infrastructure/repositories/training_jobs.py
backend-next/tests/test_ecnu_adapter.py
backend-next/tests/test_intervention_response_atomicity.py
backend-next/tests/test_label_provenance_and_publication.py
```

**修改（24）**
```
backend-next/pyproject.toml
backend-next/src/mindflow/agents/llm_gateway.py
backend-next/src/mindflow/api/routes/intervention.py
backend-next/src/mindflow/api/schemas.py
backend-next/src/mindflow/app.py
backend-next/src/mindflow/config.py
backend-next/src/mindflow/graph/analysis_graph.py
backend-next/src/mindflow/graph/chat_graph.py
backend-next/src/mindflow/infrastructure/intervention_popup.py
backend-next/src/mindflow/infrastructure/provider_registry.py
backend-next/src/mindflow/infrastructure/repositories/intervention.py
backend-next/src/mindflow/infrastructure/schema.py
backend-next/src/mindflow/ports.py
backend-next/src/mindflow/services/intervention_service.py
backend-next/src/mindflow/services/training_job_service.py
backend-next/src/mindflow/train/models/ensemble.py
backend-next/src/mindflow/train/models/manager.py
backend-next/src/mindflow/train/pipeline.py
backend-next/src/mindflow/train/v2.py
backend-next/tests/test_collector_intervals.py
backend-next/tests/test_intervention_service.py
backend-next/tests/test_notification.py
backend-next/tests/test_notification_popup.py
backend-next/tests/test_provider_registry.py
backend-next/tests/test_train_v2.py
frontend/src/api.ts
frontend/src/generated/api-schema.ts
frontend/src/pages/Chat.tsx
frontend/src/pages/Focus.tsx
frontend/src/pages/Intervention.tsx
frontend/src/pages/Settings.tsx
frontend/src/realtime.ts
```

---

## 11. 实验产物索引

| 路径 | 内容 |
|---|---|
| `backend-next/data/experiments/20260919_audit_repro/repro_explicit_mask.py` | 污染复现脚本（可复跑） |
| 同上 `repro_explicit_mask.json` | 污染量化结果 |
| 同上 `label_separation.py` / `.json` | 修复后显式评估 vs 旧口径对比 |
| 同上 `label_ablation.py` / `.json` | 四变体消融（§2.1 表） |
| `mindflow-audit/ecnu_probe/compat_summary.json` | ECNU 兼容探测（认证/JSON/工具往返） |
| `mindflow-audit/ecnu_probe/effort_discrimination.json` | max vs low 推理 token 对照 |
| `mindflow-audit/ecnu_probe/adapter_e2e.json` | 生产代码路径端到端（含并发门） |
| `mindflow-audit/ecnu_probe/config_verification.json` | 真实 .env 解析 + 真实生成验证 |
| `mindflow-audit/prompt_eval/paired_prompt_results.json` | 12 场景 × 2 提示逐条输出与评分 |
| `mindflow-audit/ecnu_budget.json` | 持久化请求计数（60/200）与逐次明细 |
| `mindflow-audit/pytest_baseline.txt` / `pytest_final.txt` | 基线 / 修复后完整测试输出 |
