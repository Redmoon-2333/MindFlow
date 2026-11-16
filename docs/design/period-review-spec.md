# 阶段总结（Period Review）实现规格 v1 — 绑定决策

> 由主代理定稿。实现者只按本规格写代码；遇到与代码现实冲突、本规格未覆盖的情况，**停下汇报**，不要自行设计。

## 0. 目标

用户想知道"最近几天因为什么分心、具体做了哪些工作、分心了多少次"。本功能：

1. **事实层（本地、确定性）**：用当前已加载的行为分类模型（个人模型或内置 base-v4）对时间段内每个 5 分钟特征窗口重新打分，聚合成 `PeriodDigest`。
2. **脱敏**：`PeriodDigest` 只由数值特征窗口 + 提醒记录的枚举字段构成，结构为白名单（Pydantic `extra="forbid"`，字符串字段全部是枚举或日期），**天然不含**窗口标题、网址、文件路径、应用名、提醒正文、聊天内容。发给大模型的就是这份 JSON 原文，前端可查看。
3. **叙述层（可选 AI）**：把 digest + 证据目录 + 论文检索结果交给大模型写复盘；输出逐条校验证据 ID / 论文 ID / 禁用词，不通过就回退到本地模板。
4. 聊天智能体新增工具 `query_period_review`，用户在对话里问"这几天为什么分心"时可直接拿到同一份 digest。

## 1. 后端文件清单

| 文件 | 动作 |
| --- | --- |
| `backend-next/src/mindflow/domain/period_review.py` | 新建，纯函数 + 数据结构（无 I/O、无 numpy） |
| `backend-next/src/mindflow/services/prediction_service.py` | 新增公开方法 `score_windows`；**不得修改** `_predict_from_windows`、`predict_latest`、`predict_range` 的行为 |
| `backend-next/src/mindflow/services/period_review_service.py` | 新建 |
| `backend-next/src/mindflow/agents/policies.py` | 新增 `PERIOD_REVIEW_POLICY` |
| `backend-next/src/mindflow/services/literature_service.py` | `_ROLE_QUERIES` 新增 `"period_review"` 一项 |
| `backend-next/src/mindflow/api/schemas*.py`（跟随现有 schemas 组织方式） | 新增请求/响应模型 |
| `backend-next/src/mindflow/api/routes/reports.py` | 新增 `POST /reports/period-review` |
| `backend-next/src/mindflow/api/deps.py` | 新增 `get_period_review_service` |
| `backend-next/src/mindflow/app.py` | 构造并挂到 `app.state.period_review_service` |
| `backend-next/src/mindflow/agents/langchain_tools.py`、`graph/tools.py`、`services/chat_service.py` | 新增聊天工具 `query_period_review` |
| `backend-next/tests/test_period_review.py` 等 | 见 §6 |

## 2. `domain/period_review.py`

### 2.1 常量

```python
PERIOD_DAYS_ALLOWED: tuple[int, ...] = (3, 7, 14)
DISTRACTION_PROBA_THRESHOLD = 0.5      # 与 prediction_service 中 focus_probas < 0.5 一致
EPISODE_MERGE_GAP_S = 60.0             # 后一窗口 start − 前一窗口 end ≤ 60s 视为连续
MIN_ACTIVE_WINDOWS = 12                # 少于 12 个有效窗口（≈1 小时）→ data_state="insufficient"
MAX_MODEL_SIGNALS = 3
CONTEXT_LABELS_ZH = {
    "coding": "编程", "writing": "写作与文档", "reading": "查资料与阅读",
    "meeting": "沟通与会议", "entertainment": "娱乐", "social": "社交", "unknown": "未识别",
}
DAYPARTS = (  # (code, 中文, 起始小时含, 结束小时不含)，固定顺序
    ("morning", "上午", 6, 12), ("afternoon", "下午", 12, 18),
    ("evening", "晚上", 18, 24), ("night", "深夜", 0, 6),
)
RESPONSE_KEYS = ("accepted", "ignored", "dismissed", "pending")
NON_BEHAVIORAL_FEATURES = frozenset({"hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "task_type_code"})
```

拖延类型中文标签：`from mindflow.services.intervention_policy import PROCRASTINATION_LABELS_ZH` 不允许（domain 不能依赖 services）。在本模块内定义同值副本 `REASON_LABELS_ZH`（任务厌恶/冲动分心/决策拖延/完美主义/情绪调节困难），并加一个单元测试断言它与 `intervention_policy.PROCRASTINATION_LABELS_ZH` 完全相等。

### 2.2 输入数据结构（frozen dataclass）

```python
@dataclass(frozen=True)
class ScoredWindow:
    start_utc: datetime; end_utc: datetime   # tz-aware UTC
    task_context: str                        # TASK_CONTEXT_CATEGORIES 之一
    p_distraction: float                     # = 模型输出第 0 列

@dataclass(frozen=True)
class InterventionFact:
    triggered_at_utc: datetime
    reasons: tuple[str, ...]                 # context_json.procrastination_types，仅保留 REASON_LABELS_ZH 的键
    response: str                            # RESPONSE_KEYS 之一；None/未知值 → "pending"

@dataclass(frozen=True)
class DistractionDriver:
    feature: str; label: str                 # label 用 domain.feature_labels.feature_label
    direction: Literal["higher", "lower", "similar"]
    distracted_mean: float; focused_mean: float
```

### 2.3 输出：Pydantic 模型（全部 `model_config = ConfigDict(extra="forbid", frozen=True)`）

```jsonc
{
  "period": {"start_date": "2026-11-10", "end_date": "2026-11-16", "days": 7},
  "data_state": "ready" | "insufficient" | "no_model",
  "model": {"version": "base_v4_20261111_r2" | null, "is_base_model": true},
  "totals": {"active_minutes": 1260, "focused_minutes": 1040, "distracted_minutes": 220,
             "distracted_share": 0.175, "days_with_data": 6},
  "episodes": {"count": 14, "avg_minutes": 15.7, "longest_minutes": 45},
  "work_mix": [{"context": "coding", "label": "编程", "minutes": 610, "share": 0.484}],
  "distraction_by_context": [{"context": "entertainment", "label": "娱乐", "minutes": 120, "share": 0.545}],
  "distraction_by_daypart": [{"daypart": "morning", "label": "上午", "episodes": 2, "distracted_minutes": 25}],
  "daily": [{"date": "2026-11-10", "active_minutes": 180, "distracted_minutes": 30, "episodes": 2,
             "top_context": "coding", "top_context_label": "编程"}],
  "model_signals": [{"feature": "app_switch_count", "label": "应用切换次数", "direction": "higher",
                     "distracted_mean": 9.2, "focused_mean": 3.1}],
  "interventions": {"total": 5, "accepted": 2, "ignored": 2, "dismissed": 1, "pending": 0,
                    "by_reason": [{"type": "impulsivity", "label": "冲动分心", "count": 3}]}
}
```

计算规则：
- 窗口分钟数 = `(end_utc − start_utc)` 秒数 / 60，裁剪到 `[0, 5]`。`active_minutes` = 全部窗口分钟和；`distracted` = `p_distraction ≥ 0.5` 的窗口；`focused` = 其余。所有 minutes 字段四舍五入为 int；share 保留 3 位小数；`avg_minutes` 保留 1 位。`distracted_share` 分母为 0 时为 0.0。
- 片段（episode）：按 start 排序的分心窗口，相邻两窗满足 `next.start − prev.end ≤ 60s` 即合并。片段分钟 = 其窗口分钟和。片段的日期和时段按**片段起点**的本地时间（`tz` 参数，`zoneinfo`/`resolve_timezone` 结果）。
- `work_mix`：按全部窗口的 `task_context` 汇总分钟，只列 minutes>0，按 minutes 降序，同分按 `TASK_CONTEXT_CATEGORIES` 顺序；share 分母为 active_minutes。
- `distraction_by_context`：只用分心窗口，share 分母为 distracted_minutes。
- `distraction_by_daypart`：四项全部列出（固定 DAYPARTS 顺序），episodes 按片段起点计，distracted_minutes 按每个分心窗口起点的时段计。
- `daily`：范围内**每一天**都列出（无数据为 0，`top_context` 为 `"unknown"` 时 label `"未识别"`；无窗口的一天 top_context 也填 `"unknown"`）。窗口按其起点本地日期归日；episodes 按片段起点归日。`days_with_data` = active_minutes>0 的天数。
- `model_signals`：直接用传入的 drivers（服务层已排好、≤3 条），转成模型。
- `interventions.by_reason`：每条提醒的每个 reason 计 1，按 count 降序、同分按 REASON_LABELS_ZH 键顺序。
- `data_state`：调用方传 `model_available=False` → `"no_model"`（此时 windows 为空，totals/episodes/work_mix 等全 0/空，daily 仍列出每天 0）；否则窗口数 < 12 → `"insufficient"`；否则 `"ready"`。

函数签名：

```python
def build_period_digest(
    windows: Sequence[ScoredWindow], interventions: Sequence[InterventionFact],
    drivers: Sequence[DistractionDriver], *, start_date: date, end_date: date,
    tz: tzinfo, model_version: str | None, is_base_model: bool, model_available: bool,
) -> PeriodDigest
```

### 2.4 证据目录

`def evidence_catalog(digest: PeriodDigest) -> dict[str, str]`，返回 `ID → "中文标签：值"`。ID 只用 `[A-Za-z0-9_.]`（与 `agents/orchestrator.validate_citations` 的正则一致）：

| ID | 示例值 |
| --- | --- |
| `totals.active_minutes` / `totals.focused_minutes` / `totals.distracted_minutes` | `有效记录时长：1260 分钟` |
| `totals.distracted_share` | `分心时长占比：17.5%` |
| `episodes.count` / `episodes.avg_minutes` / `episodes.longest_minutes` | `分心次数：14 次` / `平均每次分心：15.7 分钟` / `最长一次分心：45 分钟` |
| `work_mix.<context>` | `编程：610 分钟（48.4%）` |
| `distraction_by_context.<context>` | `在娱乐上分心：120 分钟` |
| `daypart.<daypart>`（仅 episodes>0 或 distracted_minutes>0 的） | `上午：分心 2 次，共 25 分钟` |
| `daily.<YYYYMMDD>`（仅 active_minutes>0 的天） | `11月14日：记录 180 分钟，分心 30 分钟，2 次` |
| `signal.<feature>` | `应用切换次数：分心时偏高（9.2 对 3.1）`；direction 文案：higher→偏高、lower→偏低、similar→差别不大 |
| `interventions.total/accepted/ignored/dismissed` | `收到提醒：5 次` / `点了“好，试试看”：2 次` / `点了“先忽略”或未处理：2 次` / `点了“不适合我”：1 次` |
| `reason.<type>` | `提醒原因·冲动分心：3 次` |

### 2.5 本地模板叙述

`def template_narrative(digest: PeriodDigest) -> Narrative`，`Narrative` 结构见 §4.3（source="template"）。文案逐字如下（`{}` 为插值；小时数 = 分钟/60 保留 1 位；百分比保留整数）：

- `data_state == "no_model"`：headline `还没有可用的行为分类模型`；findings `["模型加载后才能按 5 分钟窗口判断专注和分心。"]`（evidence 空）；suggestions `["到“模型中心”确认基础模型已启用。"]`。
- `data_state == "insufficient"`：headline `记录还不够，暂时总结不出规律`；findings `["这段时间只有 {active_minutes} 分钟有效记录，少于 1 小时。"]`（evidence `totals.active_minutes`）；suggestions `["让 MindFlow 在后台多运行几天，再来看看。"]`。
- `ready`：
  - headline：`episodes.count > 0` → `最近{days}天分心 {count} 次，多发生在{top_label}上`（top_label = distraction_by_context[0].label）；否则 `最近{days}天没有识别到明显的分心`。
  - findings（按顺序，条件不满足则跳过）：
    1. `有效记录 {active_h} 小时，模型判断专注 {focused_h} 小时、分心 {distracted_h} 小时，分心约占 {share}%。` → `totals.active_minutes, totals.focused_minutes, totals.distracted_minutes, totals.distracted_share`
    2. work_mix 非空：`时间主要花在{l1}（{s1}%）` + （若有第二项）`和{l2}（{s2}%）` + `。` → `work_mix.<c1>[, work_mix.<c2>]`
    3. episodes.count>0：`分心最常出现在{dp_label}，共 {dp_n} 次；最长一次约 {longest} 分钟。`（dp = episodes 最多的时段，同数取 DAYPARTS 顺序靠前者）→ `daypart.<dp>, episodes.longest_minutes`
    4. model_signals 非空：`模型判断分心时，最明显的信号是` + 前两条用`、`连接的 `{label}{偏高/偏低/差别不大}` + `。` → `signal.<f>...`
    5. `is_base_model` 为真：`当前用的是基础模型，判断可能有偏差，可以在提醒里点“不适合我”纠正。`（无 evidence）
  - suggestions（最多 2 条）：
    - 有 episodes 且 distraction_by_context[0].context ∈ {entertainment, social}：`{dp_label}开始前先写下要做的事，把{ctx_label}类应用留到任务结束后再打开。`
    - 否则有 episodes：`{dp_label}的任务可以拆小一点，每段先定一个能完成的小目标。`
    - `interventions.dismissed > 0`：`有 {dismissed} 次提醒你选了“不适合我”，系统会根据这些反馈调整之后的提醒。`
    - 若以上都不满足：`保持现在的节奏，有需要时再来回顾。`

## 3. `FocusPredictionService.score_windows`

```python
@dataclass(frozen=True)
class WindowScoring:
    status: FocusPredictionStatus            # "ready" | "no_model" | "no_data" | "schema_mismatch" | "inference_error"
    model_version: str | None
    windows: tuple[ScoredWindow, ...]
    drivers: tuple[DistractionDriver, ...]
    reason: str = ""

async def score_windows(self, user_id: int, start: datetime, end: datetime) -> WindowScoring
```

- 无模型/未 fitted → `status="no_model"`。查询用 `list_feature_windows_in_range(..., feature_schema_version=FEATURE_SCHEMA_VERSION)`，再用 `_window_overlaps_range` 过滤；跳过 `_has_activity_data` 为假或 features_json 解析失败的行；无可用行 → `"no_data"`。
- 每行特征按 `V2_FEATURE_NAMES` 顺序用 `_finite_float` 取值；`task_context = task_context_from_feature_window(features)`（来自 `services/telemetry_features.py`）；起止时间解析 `window_start_utc` / `window_end_utc`（用模块已有的解析方式，`_parse_window_end` 同风格）。
- 特征名校验、NaN 校验同 `_predict_from_windows`，失败返回对应 status。
- **不截断**：按 `_MAX_PREDICT_WINDOWS`（500）分块调用 `predict_proba`，拼接结果；概率合法性校验同现有逻辑。`p_distraction = proba[:, 0]`。
- drivers：分心窗口（p≥0.5）与专注窗口各 ≥3 时才计算，否则空。
  1. 先尝试 SHAP：`self._explain(model_manager, distracted_matrix[-500:], focus_probas_of_those, None)`；取 `contributions` 中 `contribution < 0` 且 feature 不在 `NON_BEHAVIORAL_FEATURES` 的，按 contribution 升序取前 3。
  2. `_explain` 返回 None 或筛后为空 → 回退：对每个行为特征算标准化均值差 `|m_d − m_f| / (sqrt((var_d+var_f)/2) + 1e-6)`，降序取前 3（同样排除 NON_BEHAVIORAL_FEATURES）。
  3. 每条 driver 的 `distracted_mean`/`focused_mean` 都用两组矩阵的列均值（保留 3 位）；direction：`m_d > m_f*1.1 + 1e-9` → higher；`m_d < m_f*0.9 − 1e-9` → lower；否则 similar。
- CPU 部分放 `asyncio.to_thread`；整个方法不抛异常（DB 异常 → `"inference_error"`）。

## 4. `services/period_review_service.py`

### 4.1 构造

```python
class PeriodReviewService:
    def __init__(self, *, prediction_service: FocusPredictionService,
                 intervention_repo: <app.py 里现有的 intervention 仓储实例类型>,
                 gateway_provider: Callable[[], PanelLLMGateway | None],
                 timezone: TimezoneLike, ai_timeout_s: float = 90.0) -> None
```

### 4.2 方法

- `async def build_digest(self, user_id: int, days: int, today: date | None = None) -> PeriodDigest`
  - `days` 不在 `PERIOD_DAYS_ALLOWED` → `ValueError`。`today` 默认 `business_today(timezone)`；`start_date = today − (days−1)`。UTC 范围 = `business_day_bounds_utc(start_date)[0]` 到 `business_day_bounds_utc(today)[1]`。
  - `score_windows` → status 为 `"no_model"` 时 `model_available=False`；`"ready"`/`"no_data"` 正常；其他 status 记 warning 日志并按无窗口处理（`model_available=True`，自然落到 insufficient）。
  - `is_base_model`：`model_version` 以 `"base_"` 开头为真（base bundle 版本号形如 `base_v4_20261111_r2`）。
  - 提醒：`intervention_repo.query_range(user_id, start_utc, end_utc)`；`context_json` 是 dict 时取 `procrastination_types`；**只读** `triggered_at`、`context_json.procrastination_types`、`user_response` 三个字段。
- `async def review(self, user_id: int, days: int, use_ai: bool) -> PeriodReviewResult`
  - 返回 `digest`、`llm_payload = digest.model_dump_json()`（紧凑 JSON，即发给模型的原文）、`ai_requested = use_ai`、`narrative`、`generated_at`（UTC ISO）。
  - `use_ai=False` 或 `data_state != "ready"` → 模板叙述（fallback_reason：不请求 AI 时为 None；数据不足时为 None）。
  - AI 路径失败 → 模板叙述，`fallback_reason` 取以下之一：`"AI 未配置"`（gateway_provider 返回 None 或抛 `GatewayNotConfiguredError`）、`"AI 响应超时"`（`asyncio.wait_for` 超时 `ai_timeout_s`）、`"AI 调用失败"`（其他异常，记 warning，不吞掉堆栈：`logger.opt(exception=True)`）、`"AI 输出未通过校验"`（解析或校验失败）。只调用一次，不重试。

### 4.3 AI 调用

- 论文：`literature = retrieve_literature("self-monitoring reflection distraction interruption review", role="period_review")`；`_ROLE_QUERIES["period_review"] = "self-monitoring reflection interruption resumption goal intention procrastination"`。
- system = `PERIOD_REVIEW_SYSTEM_PROMPT + "\n\n" + LITERATURE_RULES`；user = `literature.augment("## period_digest\n" + llm_payload + "\n\n## evidence_catalog\n" + json.dumps(catalog, ensure_ascii=False))`。
- `with llm_call_context(graph="period_review", node="narrate", role="period_review"):` 内调用 `await asyncio.wait_for(gateway.complete(system=..., user=..., model="chat", policy=PERIOD_REVIEW_POLICY), timeout=ai_timeout_s)`。
- `PERIOD_REVIEW_POLICY = CompletionPolicy(role="period_review", reasoning_effort="low", max_output_tokens=4096, intent="只根据阶段统计写复盘，不推测具体内容", node="period_review")`。若 `CompletionPolicy` 的 role 字段有枚举约束导致不能用新值，停下汇报。
- `PERIOD_REVIEW_SYSTEM_PROMPT`（逐字）：

```
你是 MindFlow 的“阶段复盘”助手。用户想回顾最近几天的专注情况。你只能使用 period_digest 里的统计数字：它们来自本地行为分类模型对每 5 分钟窗口的判断，以及系统提醒的记录。
规则：
1. 只写数字里有的事实。数据里没有网页、文件、聊天内容，不要猜测用户具体在看什么、写什么。
2. findings 写 2–4 条发现，每条一句话，必须带至少一个 [证据: ID]，ID 只能取自 evidence_catalog。
3. suggestions 写 1–3 条下一步建议，具体、能马上做、语气平和；引用 research_context 里的论文时用 [论文: chunk_id] 标注，不引用也可以。
4. 如果 model.is_base_model 为 true，在一条 finding 里说明“基础模型的判断可能有偏差，可以在提醒里纠正”。
5. 不使用“诊断”“治疗”“患者”“处方”等医疗用语，不给用户贴性格标签。
只输出 JSON，不要代码块：{"headline": "不超过30字的一句话总结", "findings": [{"text": "..."}], "suggestions": [{"text": "..."}]}
```

- 解析：去掉首尾 ```json / ``` 包裹后 `json.loads`；Pydantic 校验 `headline` 1–40 字，`findings` 1–4 条，`suggestions` 1–3 条，每条 text 1–160 字，`extra="forbid"`。
- 校验（任一失败 → "AI 输出未通过校验"）：
  1. 每条 finding 至少含一个 `[证据: ID]`（正则同 `validate_citations`：`\[证据[:：]\s*([A-Za-z0-9_.]+)\s*\]`），所有出现的证据 ID 都在 catalog 内；
  2. `invalid_literature_citations(全部文本, literature.citation_ids)` 为空；
  3. `agents.types._contains_forbidden_words(全部文本)` 为 None。
- 通过后：把 `[证据: …]`、`[论文: …]` 标记从 text 中删去（并清理多余空格），ID 分别放进该条的 `evidence` / `paper_ids`。

### 4.4 Narrative 结构（Pydantic，extra=forbid）

```jsonc
{
  "source": "ai" | "template",
  "headline": "…",
  "findings": [{"text": "…", "evidence": [{"id": "episodes.count", "label": "分心次数：14 次"}]}],
  "suggestions": [{"text": "…", "paper_ids": ["steel2007_p2"]}],
  "papers": [{"chunk_id": "…", "title": "…", "pages": [2]}],   // 只列被引用的，取自 LiteratureHit
  "fallback_reason": null | "AI 未配置" | "AI 响应超时" | "AI 调用失败" | "AI 输出未通过校验"
}
```

## 5. API 与聊天工具

- `POST /api/v1/reports/period-review`，body `{"days": 3|7|14 = 7, "use_ai": bool = true}`（`days` 用 `Literal[3, 7, 14]`，非法值 422）。响应模型 `PeriodReviewResponse {digest, llm_payload, ai_requested, narrative, generated_at}`，digest 与 narrative 用 §2.3/§4.4 的同一 Pydantic 模型。user_id 固定 1（与现有 reports 路由一致）。
- `app.py`：在 provider_registry 创建之后构造 `PeriodReviewService(prediction_service=prediction_service, intervention_repo=<现有 intervention 仓储实例>, gateway_provider=lambda: provider_registry.get_gateway() if provider_registry is not None else None, timezone=settings.timezone)`，挂 `app.state.period_review_service`；`deps.get_period_review_service` 仿 `get_report_service`。
- 聊天工具 `query_period_review(days_back: int = 7)`：description = `"查看最近几天的阶段总结统计（分心次数、分心发生在哪类事情和哪个时段、主要在做哪类工作、提醒与回应）。用户问“这几天为什么分心”“这周做了什么”“分心了多少次”时调用。只返回统计数字，不含窗口标题和网址。"`。days_back ≤3→3，≤7→7，其他→14；返回 `build_digest(...)` 的 JSON 字符串（不调用 AI）。按现有 `query_interventions` 的写法接入 langchain_tools / graph/tools 适配器 / chat_service 注册。**不修改** `CHAT_SYSTEM_PROMPT`。

## 6. 后端测试（必须新增并通过）

- `tests/test_period_review.py`（纯函数）：片段合并（相邻合并、间隔 >60s 不合并）、时段归属（跨午夜窗口按起点）、work_mix 排序与 share、daily 补零、data_state 三态、evidence_catalog 的 ID 正则合法且值文本正确、模板叙述三种 data_state 的 headline 文案、`REASON_LABELS_ZH == PROCRASTINATION_LABELS_ZH`。
- **隐私测试**：构造提醒记录 `title="秘密项目.docx 修改"`、`message="https://example.com/private"`、`feedback_comment="私密"`，经 service.build_digest 后，`llm_payload` 中不出现这三个字符串；并断言 `PeriodDigest` 所有嵌套模型 `extra == "forbid"`。
- `tests/test_period_review_service.py`：假 gateway 返回合法 JSON → source="ai" 且标记被剥离到 evidence/paper_ids；含未知证据 ID → template + "AI 输出未通过校验"；含“诊断” → 同上；gateway 抛异常 → "AI 调用失败"；超时（ai_timeout_s=0.01 + sleep）→ "AI 响应超时"；gateway_provider 返回 None → "AI 未配置"；use_ai=False → 不调用 gateway。
- `score_windows`：>500 个窗口全部被打分（不截断）；无模型 → no_model；drivers 排除 NON_BEHAVIORAL_FEATURES。用现有测试里的假 classifier/模型管理器写法。
- 路由：200 正常、`days=5` → 422。聊天工具：days_back 映射与返回 JSON 可解析。

门禁（在 `backend-next` 下）：`uv run python -m pytest tests/ -q` 全绿（与改动前相比不得新增失败；若有改动前就失败的用例，列出来汇报）、`uv run python -m ruff check src tests` 0、`uv run python -m mypy --strict src/mindflow` 0。

## 7. 前端

### 7.1 文件

| 文件 | 动作 |
| --- | --- |
| `frontend/src/period-review.ts` | 新建：类型（与 §2.3/§4.4 对应）+ 纯函数 `toPeriodReviewView` |
| `frontend/src/api.ts` | 新增 `getPeriodReview(days: 3 \| 7 \| 14, useAi: boolean): Promise<PeriodReview>`，POST JSON，超时 60000 ms（用 `request` 的第三个参数） |
| `frontend/src/components/PeriodReview.tsx` + `period-review.css` | 新建 |
| `frontend/src/pages/Reports.tsx` | Tab 增加 `"period"` |
| `frontend/tests/period-review.test.ts` | 新建，测纯函数 |
| `frontend/src/generated/api-schema.ts` | `npm run generate:api` 重新生成 |

### 7.2 Reports.tsx 改动（只动这些）

- `type Tab = "daily" | "weekly" | "period"`；Segmented 增加 `{ value: "period", label: "阶段总结" }`。
- `page-lede` 改为 `按日、按周汇总你的专注表现，也可以回顾最近几天的整体情况。`
- `tab === "period"` 时：不渲染日期选择器、错误框、MetricStrip 及下方三个 Surface，改为渲染 `<PeriodReview />`；daily/weekly 现有加载逻辑在 period 下不触发请求。`placeholderStats` 的类型相应收窄，不改其文案。

### 7.3 PeriodReview 组件（自上而下）

1. **工具栏**（`page-toolbar` 同款一行）：
   - `Segmented` ariaLabel `回顾范围`，选项 `最近3天`/`最近7天`/`最近14天`，默认 7。
   - 复选开关：`<label className="period-ai-toggle"><input type="checkbox" checked={useAi} …/> 让 AI 写文字总结</label>` + `HelpTip`，what = `开启后，会把下方“发给 AI 的内容”里的统计数字发给大模型，让它写成文字总结。`，effect = `只发送时长、次数这类数字，不发送窗口标题、网址和文件名；关闭后用本地规则写总结。`（HelpTip 的实际 props 以组件定义为准，语义不变）。默认开启。
   - 按钮 `生成总结`（`btn btn-primary`），加载中禁用并显示 `生成中…`。
2. **初始空态**（未生成过）：`Surface` 内一行 `选好回顾范围，点“生成总结”。`
3. **加载态**：`spinner` + `正在汇总，AI 总结可能需要十几秒…`。
4. **错误**：与 Reports 同款 `error-box` + `重试` 按钮（`getErrorMessage`）。
5. **结果**：
   - **总结卡** `Surface as="section" className="period-narrative"`：`SectionHeader` title = headline，actions 放 `StatusChip`：source=ai → `AI 总结`，template → `本地规则总结`。若 `fallback_reason` 非空，标题下一行小字 `{fallback_reason}，已改用本地规则总结。`。findings 用 `<ul className="period-findings">`，每条 text 后跟证据小标签（`<span className="period-chip">{label}</span>`，最多显示 3 个）；suggestions 用带小标题 `下一步可以试试` 的 `<ol>`；papers 非空时底部一行 `参考文献：` + 每篇 `{title}（第{pages.join("、")}页）`。
   - **MetricStrip** label = `最近{days}天`，四项：`有效记录`（`formatMinutes(active_minutes)`）、`分心次数`（`{count}` unit `次`，help what = `模型把每 5 分钟判断一次，连在一起的分心片段算一次。`）、`分心占比`（`{Math.round(share*100)}` unit `%`）、`最长一次分心`（`{longest_minutes}` unit `分钟`）。
   - **两列网格** `div.period-grid`（≥900px 两列、`gap: 16px`；<900px 单列）：
     - `主要在做什么`：BarChart，data = work_mix，label=label，value=minutes，unit ` 分钟`，valueName `时长`，height 220，`colorFor={() => "var(--signal)"}`，emptyMessage `暂无数据`。
     - `在哪类事情上分心`：同上，data = distraction_by_context，valueName `分心时长`，`colorFor={() => "var(--warn, var(--signal))"}`（若主题 token 中无 `--warn`，查 `signal.css` 里已有的警示色变量名替换；没有任何警示色变量则用 `var(--signal)`）。
   - **每天的情况** Surface：BarChart，data = daily，label = `M/D`（如 `11/14`），value = distracted_minutes，valueName `分心时长`，unit ` 分钟`；下方 `kv-row` 列出四个时段：`上午` → `分心 {episodes} 次 · {distracted_minutes} 分钟`。
   - **模型看到的分心信号** Surface：SectionHeader help what = `由行为分类模型对分心片段做解释得到，说明模型凭什么判断，不代表原因一定如此。`；列表每行 `分心时「{label}」{偏高/偏低/差别不大}`，右侧灰字 `{distracted_mean} 对 {focused_mean}`；空时 `分心片段太少，暂时看不出稳定信号。`
   - **提醒与你的回应** Surface：一行 `共提醒 {total} 次：试试看 {accepted} · 先忽略 {ignored} · 不适合我 {dismissed}`（pending>0 时追加 ` · 未回应 {pending}`）；by_reason 渲染为 `period-chip` 列表 `{label} {count}`；total=0 时 `这段时间没有触发提醒。`
   - **Disclosure**：标题 ai_requested ? `查看发给 AI 的内容` : `查看可发给 AI 的内容（本次没有发送）`；内容 `<pre className="period-payload">` = `JSON.stringify(JSON.parse(llm_payload), null, 2)`；上方一行说明 `这是本次发送的全部数据：只有统计数字和类别，没有窗口标题、网址、文件名和提醒正文。`（未发送时把“本次发送的”改为“可发送的”）。Disclosure 的实际 props 以组件定义为准。
6. 用 `useRef` 请求序号防止旧响应覆盖新结果（与 Reports 同款）。

### 7.4 样式（period-review.css）

- `.period-grid { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }`，`@media (max-width: 899px) { grid-template-columns: 1fr; }`
- `.period-chip { display:inline-block; padding:2px 8px; margin:2px 4px 2px 0; border-radius:999px; font-size:12px; line-height:18px; border:1px solid var(--line, currentColor); opacity:.85; }`（`--line` 若不存在，用 signal.css 中已有的边框色变量名）
- `.period-findings li, .period-narrative ol li { margin: 6px 0; line-height: 1.7; }`
- `.period-payload { max-height: 320px; overflow:auto; font-size:12px; line-height:1.5; white-space:pre; }`
- `.period-ai-toggle { display:inline-flex; align-items:center; gap:6px; }`
- 只用已有主题变量，不引入新色值；深浅主题都要可读。

### 7.5 前端门禁

`npm run build`、`npm run lint`、`npm run check:api-drift`、`npx tsx --test tests/period-review.test.ts`（或项目现有的单测运行方式）全部通过。

## 8. 不做的事

- 不新建数据库表、不写迁移；不缓存结果。
- 不读取 `activity_events`、`browser_segments`、`chat_messages`；不把应用名/域名放进 digest。
- 不改动现有日报、周报、面板、干预的行为。

---

# 附录 A（v1.1）：未识别项目由用户归类，永久记忆

## A.1 后端

- `TelemetryService.list_unclassified(user_id: int, start: datetime, end: datetime, limit: int = 10) -> list[UnclassifiedItem]`（新增；`UnclassifiedItem` 为 frozen dataclass：`kind: Literal["app","site"]`、`key: str`、`display: str`、`minutes: int`）。
  - 读取范围内的 activity events 与 browser segments，读法与 rollup 构建窗口时相同（复用 `_build_feature_window_rows` 读原始数据的同一组仓储调用；跳过 `is_idle` 事件）。
  - 调用现有 `_resolve_task_contexts(user_id, events, browser)` 得到 `process → context`、`domain → context`。
  - app：按进程累计非空闲事件时长（秒），context 为 `"unknown"` 的进入候选。**排除浏览器进程**：范围内 browser segments 出现过的 `browser_name`，统一转成小写、去掉 `.exe` 后，与“进程名转小写、去掉 `.exe`”相等的进程不列入（浏览器应按网站归类）。
  - site：按域名累计 segment 时长，context 为 `"unknown"` 的进入候选。
  - minutes = round(秒/60)，只保留 ≥1；按 minutes 降序，截取前 `limit` 条。`display`：app 取进程名去掉 `.exe`；site 取域名本身。
- 路由新文件 `api/routes/task_context_assign.py`（在 `api/routes/__init__.py` 中按现有方式挂载到 `/api/v1`）：
  - `GET /task-context/unclassified?days=7`：`days` 为 `Literal[3,7,14]`，时间范围与 `PeriodReviewService.build_digest` 相同（本地业务日）；返回 `{"items": [...]}`。**仅供本地页面使用，不进入 llm_payload。**
  - `POST /task-context/assign`：body `{"kind": "app"|"site", "key": str(1..255), "context": "coding"|"writing"|"reading"|"meeting"|"entertainment"|"social"}`（不接受 unknown）。
    - app：映射到活动类别 `CATEGORY_FOR_CONTEXT = {"coding":"code","writing":"document","reading":"browser_work","meeting":"communication","entertainment":"entertainment","social":"social"}`。先删除同一 `process_name`（不区分大小写）且 `window_title_pattern` 为空的已有规则，再新增 `{process_name: key, window_title_pattern: None, category, priority: 50}`。
    - site：`normalize_domain_rule(key)` 为空 → 422。先删除同一域名的 domain 规则，再新增 `{match_type: "domain", match_value: 域名, task_context: context, priority: 50}`。
    - 保存后，用 `asyncio.create_task` 在后台调用 `telemetry_service.rebuild_feature_windows(now-14天, now, user_id=1, apply=True)`，按自然日切块逐日执行（每块一次调用，单块失败只记 warning，继续下一块）。task 引用存在 app.state 的集合里防止被回收，完成后移除。重算不阻塞响应。
    - 响应 `{"saved": true, "recompute": "started"}`，201。
  - 规则存在已有的 `app_classification_rules` / `task_context_rules` 两张表里，是永久的；之后的 rollup 自动生效。
- 测试：`tests/test_task_context_assign.py`，覆盖以下几点：
  - 未识别列表能排除浏览器、按分钟排序、遵守 limit；
  - 已有规则的进程不出现在列表中；
  - app/site 两种 assign 能写入规则并替换旧规则；
  - context=unknown 返回 422；
  - 非法域名返回 422；
  - assign 之后会启动重算（用假的 telemetry service 断言被调用）。

## A.2 前端（PeriodReview 组件内，放在“每天的情况”之后、“模型看到的分心信号”之前）

- 只有生成过总结、且 `GET /task-context/unclassified?days=当前生成的天数` 返回非空时才渲染。生成总结成功后并行请求一次。
- Surface，SectionHeader 标题 `还没认出来的应用和网站`，help what = `告诉 MindFlow 这个应用或网站平时用来做什么。`，effect = `设置会一直保存，以后再遇到它就会自动归类；最近 14 天的记录会在后台重新计算，稍后重新生成总结就能看到变化。`
- 每行一个条目：左边是名称（site 前加小字 `网站`，app 前加小字 `应用`），中间灰字 `{minutes} 分钟`，右边是 `<select>`（首项 `这是……`，值为空）。select 选项依次为：编程 / 写作与文档 / 查资料与阅读 / 沟通与会议 / 娱乐 / 社交，再加一个按钮 `记住`（未选择时禁用）。
  - 保存成功后，该行替换为 `已记住：{名称} → {类别}`（绿色勾），并用现有 toast 提示 `已记住，之后会自动识别`。
  - 保存失败时，行内显示错误文字，不弹窗。
- api.ts 增加 `getUnclassified(days)`、`assignTaskContext(kind, key, context)`；`npm run generate:api` 重新生成 schema。
- 样式沿用 period-review.css，只新增 `.period-unclassified-row`：flex 布局，`gap: 12px`，`align-items: center`，行间分隔线复用已有边框色变量。

# 附录 B（v1.1）：“忽略”与“不适合我”进入建模回路（主动学习追问）

现状：ignored/dismissed 已用于提醒策略（Thompson 采样记为失败，自动超时记 0.5 次失败）以及疲劳节流，但**不进入分类模型的训练标签**。本附录补上这一环：忽略或否定提醒本身无法说明“判断错没错”，所以不直接当标签，而是在之后**追问一次**，用户的回答写成窗口标签（权重 0.8），参与下次训练。

- `services/feedback_request_service.py`：
  - `FeedbackRequestStore` 新增可选字段 `intervention_repository`（类型用现有仓储 Protocol 或具体类，二者择与 app.py 中现成实例匹配者），默认 None；新增字段 `_followed_up: set[str]`。
  - 新常量 `FOLLOWUP_PROMPT = "刚才 MindFlow 提醒你的时候，你其实在专注做事吗？"`。`REASON_LABELS` 增加 `"reminder_ignored": "你忽略了一次提醒"` 和 `"reminder_dismissed": "你选了“不适合我”"`。
  - 新增 `async def _followup_candidate(self, user_id, now)`：
    - 用 `intervention_repository.query_range(user_id, now - REQUEST_TTL, now)` 取数；
    - 取最新一条满足以下条件的记录：`user_response in {"ignored","dismissed"}`，且 id 不在 `_followed_up` 中；
    - 返回 `(id, triggered_at_utc, response)`，没有则返回 None；仓储为 None 或抛异常时返回 None，并记 warning。
  - `maybe_create`：
    - 在“pending/caps”检查通过后、`_decide` 之前，先取 candidate。
    - 有 candidate 时：`window_end = triggered_at`，`window_start = window_end - REQUEST_WINDOW`，`probability = None`，`reasons = ["reminder_ignored" 或 "reminder_dismissed"]`，`prompt = FOLLOWUP_PROMPT`，跳过 `_decide`。
    - 深度工作检查照常执行。只有成功创建请求后，才把 id 加入 `_followed_up`。
    - 冷却时间、每日上限和单个待回答请求的限制完全沿用现有规则。
    - 没有 candidate 时，走原有路径，行为不变。
  - `answer()` 的写标签逻辑不改（focus/distracted 写入 window label，skip 不写）。
- `app.py`：构造 `FeedbackRequestStore` 时传入现有 intervention 仓储实例。
- 测试 `tests/test_feedback_followup.py`：
  - ignored 记录 → 下一次 tick 生成追问，窗口为提醒前 15 分钟，prompt 与 reasons 正确；
  - 同一条记录不会追问第二次；
  - accepted 不触发追问；
  - 冷却时间内不追问；
  - 无 candidate 时回到原有不确定性路径；
  - 回答 focus 后对应窗口被写入标签。
- 原有 `tests/test_feedback_request*.py` 必须全部继续通过。
