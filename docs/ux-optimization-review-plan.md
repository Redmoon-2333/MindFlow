# MindFlow 后端 UX 优化审查与实施计划

> 生成：2026-08-03 · 范围：后端 `backend-next/` + 相关前端展示面
> 审查方式：全量只读代码走查（干预引擎 / 通知弹窗 / 调度 / AI 文案 / 前端展示契约）
> 目标：找出"细微处的体验"问题，重点修复平台错配文案、AI 化标题、反馈闭环断裂，并一并提交 GitHub

---

## 一、问题清单（按严重度排序）

### P0 · 功能断裂（必改，Bug）

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| B1 | **干预反馈评分枚举前后端不匹配**：前端 `Intervention.tsx` 提交 `effective/neutral/ineffective`，后端 `InterventionFeedbackRating = Literal["helpful","neutral","annoying"]`。选择「有用/无效」→ 提交 `effective`/`ineffective` → 后端 **422 校验失败**，反馈功能实际不可用（`neutral` 恰好碰巧通过） | `frontend/src/pages/Intervention.tsx:346-349` ↔ `backend-next/src/mindflow/api/schemas.py:64` | 用户无法评价干预，反馈闭环断裂，节流器拿不到 `annoying` 信号 |
| B2 | **`_MIN_NON_IDLE_MINUTES` 是死常量**：scheduler.py 定义了「非空闲活动 ≥10 分钟才可干预」的意图（docstring），但代码从未引用，实际守卫只检查 `all idle`。用户在短时开机 + 频繁分心下会被立即打扰 | `backend-next/src/mindflow/services/scheduler.py:75` | 假阳性打扰，与设计意图不符 |
| B3 | **弹窗超时 120s 硬编码**：`_TkinterInteractivePopup` 写死 `timeout_s: 120`，不随 urgency 变化、不可配置；且不接收/不使用 urgency，`critical` 与 `low` 弹窗外观完全一致 | `backend-next/src/mindflow/infrastructure/notification.py:166` | 弹窗可能长时间霸屏；紧急程度无视觉区分 |

### P1 · 平台错配 / 内容体验（用户点名）

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| U1 | **桌面软件却让用户"把手机调至勿扰模式"**：`environment_optimization` 模板建议「关闭无关标签页，将手机调至勿扰模式」。MindFlow 是**电脑端**专注工具，手机建议文不对板，用户一眼出戏 | `backend-next/src/mindflow/services/intervention_service.py:75` | 用户点名问题；降低可信度 |
| U2 | **LLM 系统提示未声明"桌面端"平台约束**：模型生成提醒时不知道用户在电脑前，可能给出"放下手机/离开书桌"类文案；且**没有禁用手机语境**的约束 | `backend-next/src/mindflow/services/intervention_service.py:104-116` | AI 标题/正文跑偏 |
| U3 | **弹窗标题模板化、过于单调**：无 LLM 时标题永远是「小提示：{type_label} / 来自 MindFlow 的提醒 / 专注提醒」；有 LLM 时标题限 **≤6 字**，逼得模型只能输出"整理环境"式电报标题。用户希望**由 AI 设计有变化、有温度的标题** | `backend-next/src/mindflow/domain/intervention.py:58-64` + `intervention_service.py:112` | 用户点名问题；提醒千篇一律、缺乏温度 |
| U4 | **LLM 标题/正文长度契约自相矛盾**：prompt 说 title≤6字/message≤100字，代码却按 15/200 截断，两者不一致 | `intervention_service.py:112-114, 276-279` | 契约混乱，标题易被硬截断 |

### P2 · 降级一致性 / 死参数 / 文案

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| C1 | **`enhance_with_llm` 参数是死代码**：docstring 写「currently ignored — always False」，实际逻辑是「`llm_client is not None` 就用 LLM」——参数完全不生效且误导调用方 | `intervention_service.py:388-391, 443` | API 契约混乱 |
| C2 | **干预 LLM 未接入三级降级**：干预消息用一个独立的 raw httpx 直连 DeepSeek，不走 ProviderRegistry 的 L1/L2/L3（DeepSeek→Ollama→RuleEngine）降级链。有 key 就只用 DeepSeek，没 key 就永远模板，即使配了 Ollama 也不用 | `app.py:496-511` + `intervention_service.py:236-248` | 与全系统降级设计不一致；本地用户拿不到 AI 文案 |
| C3 | **Chat 降级文案泄漏内部 API 路径**：`_LLM_DOWN_REPLY`/`_SAFE_REPLY` 让用户"查看今日报告 **/api/v1/focus**"——把内部端点暴露给终端用户 | `chat_service.py:70-78` | 文案不专业 |
| C4 | **手动触发 `bypass_deep_work_guard` 语义过宽**：手动触发连深度专注都绕过，用户忙时点触发也会被打断 | `intervention.py:84-92` | 可商榷；保留但注明 |

### P3 · 打磨（小优化，可并入）

| # | 问题 | 位置 |
|---|------|------|
| P1 | 自动干预时间窗 08:00–23:00 硬编码，非可配置项 | `scheduler.py:628-631` |
| P2 | 弹窗按钮文案与前端不一致（弹窗：接受/拒绝/暂时忽略；前端：接受/忽略/关闭） | `intervention_popup.py:28-32` ↔ `Intervention.tsx:236-262` |
| P3 | 前端 `INTERVENTION_TYPE_LABELS` 在 Dashboard.tsx 与 Intervention.tsx 重复定义，易漂移 | 两文件 |
| P4 | 前端 `latencyS` 恒传 0，弹窗记录了真实延迟但网页不记录 → 疗效数据质量差 | `frontend-page-functionality-doc.md:976` |
| P5 | 报告 `pattern_summary` 纯规则模板，无 AI 润色（可作为后续可选增强，非本次必需） | `report_service.py:573-629` |

---

## 二、优化设计（针对上述问题的方案）

### 设计 1：平台感知的干预文案（U1/U2）

- **模板侧**：重写 `_TYPE_TEMPLATES`，删除所有手机语境建议，改为桌面行为建议：
  - `environment_optimization` → 「关闭无关浏览器标签页 / 退出娱乐应用 / 打开系统专注模式」
  - `nudge` → 「把当前窗口最小化，先专注做 5 分钟」等桌面语境动作
- **LLM 侧**：`_LLM_SYSTEM_PROMPT` 增加显式平台约束：
  > "用户在**桌面电脑**前工作。所有建议必须是桌面操作（关闭标签页、退出应用、整理桌面、使用系统勿扰等），**不得提及手机**、躺下、离开书桌等非桌面语境。"
- 可在 user_content 中注入当前最可疑的分心应用名（`summary` 已有 top app），让建议更具体。

### 设计 2：AI 设计多样化标题（U3/U4）

- **放宽 LLM 标题约束**：标题 ≤6 字改为 **≤14 字**（代码侧硬截断 15 保持一致），prompt 改为：
  > `"title": 提醒标题(14字以内，有温度、有变化，不要重复类型标签本身，例如不要总是"整理环境")`
- **模板侧多样化**：无 LLM 时不再使用 3 个固定标题，而是按干预类型 + 强度组合一个**标题变体池**（每类型 ≥3 个变体），随机/轮换取用，避免千篇一律。
- **代码契约统一**：`title` 硬上限统一为 15（中文按字符），`message` 统一为 200，prompt 与代码一致。

### 设计 3：反馈闭环修复（B1）

- **方案 A（推荐）**：后端 `InterventionFeedbackRating` 扩展为同时接受旧值，但语义统一——将前端改为提交 `helpful / neutral / annoying`（与后端、与节流器信号一致）。
- **方案 B**：后端 Literal 改为 `Literal["helpful","neutral","annoying","effective","ineffective"]` 并做归一化映射。
- 选择 **方案 A**：改前端 3 个 option 的 value + 类型，前端 `InterventionRating` 类型与后端自动同步（openapi-fetch 生成），改动最小且语义最清晰。

### 设计 4：干预 LLM 接入三级降级（C2）

- 将 `InterventionService` 的 `llm_client: httpx.AsyncClient | None` 改为接收**一个 `generate_message(...)` 回调**（由 `ProviderRegistry` 提供），走统一降级链：
  - L1 DeepSeek（有 key）→ L2 Ollama（本地，无 key 时）→ L3 模板（始终可用）。
- 最小侵入实现：`app.py` 组装时注入一个 `intervention_message_generator` 函数，`intervention_service.py` 只依赖抽象回调，不直接依赖 httpx/DeepSeek。
- 风险：改动 `InterventionService.__init__` 签名 → 需同步 `tests/test_intervention_service.py` 与 `app.py`。可保留 `llm_client` 参数作兼容回退。

### 设计 5：弹窗体验（B3）

- `timeout_s` 按 urgency 映射：`low=60 / normal=90 / critical=120`（可配置，默认常量表）。
- 弹窗居中改为**右下角 20px 边距**（贴近系统通知习惯，不遮挡当前工作区中央）。
- `strict` 干预映射为 `urgency="critical"`，`gentle` 映射为 `low`，让强度可见。

### 设计 6：调度小修（B2/P1）

- 落实 `_MIN_NON_IDLE_MINUTES` 守卫：`window_min` 内非空闲累计时长 ≥10 分钟才可干预。
- 工作时间窗改为 `Settings` 可配置（`intervention_start_hour=8 / intervention_end_hour=23`）。

### 设计 7：文案清理（C3/C4/P3/P4）

- Chat 降级文案去掉 `/api/v1/focus`，改为「你可以查看**专注分析**页面了解今天的专注情况」。
- 前端干预标签常量抽到 `frontend/src/lib/intervention-labels.ts` 单一来源。
- 前端响应 `latencyS` 改为记录「干预展示 → 用户点击」真实时间差（`performance.now()`）。

---

## 三、实施计划（Team 分工）

### 阶段 0 · 基线确认
- 重跑 `uv run python -m pytest tests/ -q` 记录当前红绿基线（上次基线 1956 passed / 11 failed 真因在 `test_llm_client.py`+`test_langchain_gateway.py` 的 env-robustness，见记忆）。

### 阶段 1 · 后端干预引擎（worker-1，executor）
- `intervention_service.py`：重写 `_TYPE_TEMPLATES`、平台约束 prompt、标题 14 字、标题变体池、统一长度契约。
- `domain/intervention.py`：新增标题变体池常量。
- 新增/更新 `tests/test_intervention_service.py`。

### 阶段 2 · 干预降级与调度（worker-2，executor）
- `app.py` + `intervention_service.py`：`generate_message` 回调接入 ProviderRegistry 三级降级。
- `scheduler.py`：落实 `_MIN_NON_IDLE_MINUTES`、时间窗可配置。
- 更新 `config.py` + 相关测试。

### 阶段 3 · 通知弹窗（worker-3，executor）
- `notification.py` + `intervention_popup.py`：urgency→timeout 映射、右下角定位、强度→urgency。
- 更新 `tests/test_notification.py`、`test_intervention_popup.py`。

### 阶段 4 · 前端反馈与文案（worker-4，executor）
- `Intervention.tsx`：修复反馈枚举（→ helpful/neutral/annoying）、真实 latency。
- 标签常量抽公共文件 + Dashboard 引用。
- `chat_service.py` 降级文案清理（后端 worker-2 或本阶段）。

### 阶段 5 · 验证（verifier）
- 全量 `pytest`、`ruff`、`mypy --strict`（对照基线，不新增错误）。
- 前端 `npm run build` + 关键路径 Playwright 冒烟（干预触发/反馈/历史）。
- 修复 → 复审循环（`team-fix`）。

### 阶段 6 · 提交（git-master）
- **注意**：工作区当前存在**大量非本次改动**（`codex/cleanup` 相关 + 新迁移 0018 + start_mindflow_bg.ps1/vbs + 新测试文件等，见 `git status`）。提交前必须与用户确认：
  - 只提交本次 UX 改动（推荐，分 commit），还是把当前工作区所有改动一并提交？
- 分 commit 提交，conventional commits，push 到 `origin/main`。

---

## 四、需要用户确认的决策

1. **范围**：P0/P1/P2 全部实施 + P3 中的 P1(时间窗)、P3(标签抽离)、P4(latency)？P5(报告 AI 化) 是否纳入？
2. **反馈枚举方案**：确认采用**方案 A**（前端改为 helpful/neutral/annoying）？
3. **提交范围**：本次提交是否**只含 UX 改动**，把工作区现有非相关改动留给用户自行处理？
