# MindFlow 架构全面优化方案

> **日期**: 2026-07-27
> **审查来源**: Explore Agent 全代码库扫描（22 项发现）+ Codex 架构审查（10 项评级）
> **覆盖范围**: 4 层架构（domain → infrastructure → services → agents → api）

---

## 目录

1. [优先级总览](#1-优先级总览)
2. [🔴 P0：LangGraph 状态序列化 + 结构化输出](#2-p0-langgraph)
3. [🟠 P1：层级违规修复](#3-p1-层级违规)
4. [🟡 P2：Schema 集中化 + JSON 规范化](#4-p2-schema)
5. [🟢 P3：代码去重与清理](#5-p3-去重)
6. [实施分期](#6-实施分期)

---

## 1. 优先级总览

| 优先级 | 问题 | 严重度 | 来源 | 影响 |
|--------|------|--------|------|------|
| **P0** | `_PanelRunContext` 嵌入 `PanelState`，阻止序列化/检查点 | 🔴 High | Codex | 无法持久化、重放、中断恢复 |
| **P0** | 手动 JSON 解析有 bool 强制 bug（`bool("false") == True`） | 🔴 High | Codex | 批评家输出"false"会被误判为通过 |
| **P0** | 无 LangGraph Checkpointer | 🟠 High | Explore+Codex | 进程崩溃丢失进度，可能重复付费 LLM 调用 |
| **P1** | `domain/app_classification.py` 导入 `train/` | 🔴 High | Explore+Codex | 层级反转，~3.9s 导入开销 |
| **P1** | `agents/langchain_tools.py` 绕过 services 层直接导入 repositories | 🟠 Medium | Explore | 耦合基础设施，重构困难 |
| **P1** | `sa.Table` 定义在 7 个仓库文件中重复 | 🟠 Medium | Explore+Codex | schema 漂移，迁移与代码不一致 |
| **P2** | JSON blob 列（`*_json`）无法索引查询 | 🟡 Low→Med | Explore | SQLite JSON1 可查询，但无类型安全 |
| **P2** | 仓库无 Protocol 抽象（除 ActivityRepository） | 🟡 Medium | Explore | 服务与具体实现紧耦合 |
| **P2** | `event_type` 列缺少数据库 CHECK 约束 | 🟡 Low | Explore | 无效值只在 Python 层被拒绝 |
| **P3** | `_verdict_dict_to_panel_verdict` 与 `_analysis_to_verdict` 逻辑重复 | 🟡 Low | Codex | 两处维护相同逻辑 |
| **P3** | `_contains_forbidden_words` 在 3 处定义 | 🟢 Low | Explore | 修改需同步三处 |
| **P3** | 大量构造函数参数（PanelService 8 个，ChatService 10+） | 🟢 Low | Explore | 可读性，非功能问题 |

---

## 2. 🔴 P0：LangGraph 状态序列化 + 结构化输出

### 2.1 当前问题

**`orchestrator.py:562-592`** — `PanelState` TypedDict 包含：
```python
class PanelState(TypedDict):
    runtime: _PanelRunContext  # ← 包含 asyncio.Lock + 可变 list

@dataclass
class _PanelRunContext:
    call_count: int = 0
    transcript: list[TranscriptEntry] = field(default_factory=list)
    budget_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
```

问题：
- `asyncio.Lock` 不可序列化 → 无法启用 LangGraph Checkpointer
- `call_count` 和 `transcript` 同时在 `state` 和 `runtime` 中维护 → 双数据源
- `graph.compile()` 无 `checkpointer=` → 无持久化、无重放、无中断恢复

**`orchestrator.py:303`** — 手动 JSON 解析有 bug：
```python
def _parse_critic(raw: str) -> CriticResult:
    data = _safe_parse_json(raw, "critic")
    approved = bool(data.get("approved", False))  # ← bool("false") == True !!!
```
LLM 返回 `{"approved": "false"}` 时，`bool("false")` 为 `True`，批评家形同虚设。

### 2.2 优化方案

**Step 1：用 Pydantic 结构化输出替代手动 JSON 解析（~200 行代码消除）**

```python
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable

class AnalystOutput(BaseModel):
    patterns: list[dict] = Field(default_factory=list)
    anomalies: list[dict] = Field(default_factory=list)
    evidence_citations: list[str] = Field(default_factory=list)

class AttributionOutput(BaseModel):
    attribution_types: list[str] = Field(default_factory=list)
    confidence: dict[str, float] = Field(default_factory=dict)
    argument: str = ""
    evidence_citations: list[str] = Field(default_factory=list)

class CriticOutput(BaseModel):
    approved: bool = False       # Pydantic 正确解析 bool
    issues: list[str] = Field(default_factory=list)

# 使用：
structured_model = self._gateway.model.with_structured_output(AttributionOutput)
result: AttributionOutput = await structured_model.ainvoke(prompt)
```

**Step 2：移除 `_PanelRunContext`，将运行时数据展平到 state 中**

```python
class PanelState(TypedDict):
    bundle_json: str
    valid_metrics: frozenset[str]
    analyst_opinion: ExpertOpinion | None
    attribution_opinions: list[ExpertOpinion]
    conflict_report: ConflictReport | None
    escalated: bool
    moderator_verdict: dict[str, Any] | None
    critic_result: CriticResult | None
    critic_retries: int
    call_count: int                    # ← 从 runtime 移出
    transcript: list[TranscriptEntry]  # ← 从 runtime 移出
    disagreement_summary: DisagreementSummary | None
    rebuttal_delta: object | None
    # runtime: _PanelRunContext — 删除
```

`asyncio.Lock` 改为在 orchestrator 实例中管理（不放入 state）。

**Step 3：添加 LangGraph SqliteSaver Checkpointer**

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def _build_compiled_graph(self, db_path: str):
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = StateGraph(PanelState)
        # ... 添加节点和边 ...
        return graph.compile(checkpointer=checkpointer)

# 调用时使用稳定 thread_id
config = {"configurable": {"thread_id": f"panel_{user_id}_{date}"}}
final = await compiled.ainvoke(initial, config=config)
```

**Step 4：正确使用两种限制**

```python
# recursion_limit: 图超步保护（防死循环）
graph.compile(checkpointer=checkpointer)  # 使用 config={"recursion_limit": 10}

# LLM 调用预算：单独计数器（持久化在 state 中）
# 并行归因/辩论前原子预留 3 次调用额度
```

### 2.3 预期收益

| 指标 | 当前 | 优化后 |
|------|------|--------|
| 手动 JSON 解析代码 | ~200 行 | ~20 行（Pydantic schema 定义） |
| bool 强制 bug | 存在 | 消除 |
| 崩溃恢复 | 不支持 | 支持（checkpoint 重放） |
| 状态可序列化 | ❌ | ✅ |
| 中断/人机交互 | 不支持 | 支持（`interrupt()` + `Command(resume=...)`） |

---

## 3. 🟠 P1：层级违规修复

### 3.1 `domain/app_classification.py` → `train/` 反向依赖

**文件**: `src/mindflow/domain/app_classification.py:29`
```python
from mindflow.train.features import AppClassifier  # ← 领域层导入训练层！
```

**问题**：
- `domain` 是最底层，不应依赖 `train`（离线训练 CLI）
- 导入 `AppClassifier` 时加载了整个 ML 栈，实测 ~3.9s 导入开销
- 违反分层架构的核心约束

**方案**：将 `AppClassifier`（纯同步分类器，无 ML 依赖）移动到 `domain/` 或新建 `infrastructure/classification/`。

```python
# domain/app_classification.py
from mindflow.domain.classifier import AppClassifier  # ← 同层导入

# train/features.py  
from mindflow.domain.classifier import AppClassifier  # ← 训练层导入领域层（正确方向）
```

### 3.2 `agents/langchain_tools.py` 绕过 services 层

**文件**: `src/mindflow/agents/langchain_tools.py:29-37`
```python
from mindflow.infrastructure.repositories.analysis import ...  # ← 跳过 services
from mindflow.infrastructure.repositories.intervention import ...
from mindflow.services.evidence_service import EvidenceBundleBuilder
from mindflow.services.panel_service import PanelService
```

**方案**：工具工厂只依赖 services（通过 Protocol），不直接接触 repositories。

```python
# agents/langchain_tools.py
from mindflow.ports import PanelServicePort, AnalysisServicePort  # Protocol

def make_get_panel_verdict(panel_service: PanelServicePort) -> BaseTool:
    @tool
    async def get_panel_verdict(detail: str = "summary") -> str:
        verdict = await panel_service.get_or_read_daily_panel(...)
        ...
```

### 3.3 仓库 Protocol 抽象

为所有仓库定义 Protocol（遵循 `ports.py` 中 `ActivityRepository` 的模式）：

```python
# mindflow/ports.py
class ProcrastinationAnalysisRepositoryPort(Protocol):
    async def get_by_date(self, user_id: int, target_date: date, *,
                          analysis_kind: str | None = None) -> dict | None: ...
    async def upsert(self, user_id: int, target_date: date, *,
                     analysis_kind: str = ..., ...) -> None: ...
```

---

## 4. 🟡 P2：Schema 集中化 + JSON 规范化

### 4.1 集中化 `sa.Table` 定义

**问题**：7 个仓库文件各自定义 `sa.Table`，与 Alembic 迁移重复。

**方案**：创建 `infrastructure/schema.py` 作为唯一 schema 来源：

```python
# infrastructure/schema.py
from sqlalchemy import MetaData

metadata = MetaData()

activity_events = sa.Table("activity_events", metadata, ...)
procrastination_analyses = sa.Table("procrastination_analyses", metadata, ...)
intervention_logs = sa.Table("intervention_logs", metadata, ...)
# ... 所有表定义
```

仓库文件改为 `from mindflow.infrastructure.schema import procrastination_analyses`。

同时更新 `alembic/env.py`：
```python
from mindflow.infrastructure.schema import metadata
target_metadata = metadata  # 启用 Alembic autogenerate
```

### 4.2 JSON 列评估

| 列 | 是否需要查询/索引 | 建议 |
|----|-------------------|------|
| `procrastination_types_json` | 是（按类型筛选历史） | 拆为 `panel_types` 关联表 |
| `type_confidence_json` | 可能（按置信度排序） | 拆为 `panel_type_confidences` 关联表 |
| `cognitive_distortions_json` | 否（诊断元数据） | 保留 JSON |
| `panel_transcript_json` | 否（展示用） | 保留 JSON |
| `features_json` | 否（ML 训练批量加载） | 保留 JSON |

### 4.3 `event_type` CHECK 约束

```python
# 迁移中
sa.CheckConstraint("event_type IN ('window_snapshot', 'idle_change', 'manual_tag')")
```

---

## 5. 🟢 P3：代码去重与清理

### 5.1 合并裁决转换逻辑

`_verdict_dict_to_panel_verdict`（orchestrator.py:320）与 `_analysis_to_verdict`（panel_service.py:220）提取为共享函数。

### 5.2 统一禁词检查

`_contains_forbidden_words` 在 orchestrator.py、chat_service.py、types.py 三处定义 → 统一到 `agents/types.py`。

### 5.3 服务构造函数简化

PanelService（8 参数）和 ChatService（10+ 参数）考虑引入配置对象：

```python
@dataclass
class PanelServiceConfig:
    activity_repo: ActivityRepositoryPort
    intervention_repo: InterventionLogRepositoryPort
    orchestrator: PanelOrchestrator
    llm_service: LLMServicePort
    analysis_repository: ProcrastinationAnalysisRepositoryPort
    ...
```

---

## 6. 实施分期

### Wave 1：P0 修复（3-4 天）

| 任务 | 内容 | 文件 |
|------|------|------|
| W1.1 | Pydantic 结构化输出 schema（Analyst/Attribution/Moderator/Critic） | 新建 `agents/schemas.py` |
| W1.2 | 替换 `_parse_expert_opinion` / `_parse_verdict` / `_parse_critic` 为 `with_structured_output()` | `agents/orchestrator.py` |
| W1.3 | 移除 `_PanelRunContext`，展平 state，移除 `asyncio.Lock` | `agents/orchestrator.py` |
| W1.4 | 添加 `AsyncSqliteSaver` checkpointer | `agents/orchestrator.py` |
| W1.5 | 正确配置 `recursion_limit` + LLM 调用预算 | `agents/orchestrator.py` |
| W1.6 | 单元测试：结构化输出、bool 解析、checkpoint 重放 | `tests/` |

### Wave 2：P1 层级修复（2-3 天）

| 任务 | 内容 | 文件 |
|------|------|------|
| W2.1 | 移动 `AppClassifier` 到 `domain/classifier.py` | `domain/` + `train/` |
| W2.2 | 为所有仓库定义 Protocol | `ports.py` |
| W2.3 | `agents/langchain_tools.py` 改为依赖 services Protocol | `agents/langchain_tools.py` |
| W2.4 | 服务构造函数改为依赖 Protocol 而非具体类 | `services/*.py` |

### Wave 3：P2 Schema 集中化（1-2 天）

| 任务 | 内容 | 文件 |
|------|------|------|
| W3.1 | 创建 `infrastructure/schema.py`，集中所有 `sa.Table` | 新建文件 |
| W3.2 | 更新所有仓库导入 | `infrastructure/repositories/*.py` |
| W3.3 | 更新 `alembic/env.py` `target_metadata` | `alembic/env.py` |
| W3.4 | 添加 `event_type` CHECK 约束的迁移 | 新建 migration |
| W3.5 | 拆分 `procrastination_types` 和 `type_confidence` 为关联表 | 新建 migration |

### Wave 4：P3 去重（1 天）

| 任务 | 内容 | 文件 |
|------|------|------|
| W4.1 | 合并 verdict 转换逻辑 | `agents/types.py` 或 `panel_service.py` |
| W4.2 | 统一 `_contains_forbidden_words` | `agents/types.py` |
| W4.3 | 服务 Config dataclass | `services/*.py` |

**总估算**：7-10 个工作日

---

## 附录：Codex 完整评级

| 发现 | 严重度 | 代码位置 |
|------|--------|----------|
| `_PanelRunContext` 嵌入 state，asyncio.Lock 阻止序列化 | 🔴 High | `orchestrator.py:582` |
| `_parse_critic` bool 强制 bug（`bool("false") == True`） | 🔴 High | `orchestrator.py:303` |
| 无 LangGraph Checkpointer | 🟠 High | `orchestrator.py:933` |
| domain 导入 train（层级反转，~3.9s 导入开销） | 🟠 High | `app_classification.py:29` |
| Agent 工具绕过 services 导入 repositories | 🟠 Medium | `langchain_tools.py:29` |
| 手动 12-call 预算（recursion_limit ≠ LLM 调用数） | 🟡 Low | `orchestrator.py:987` |
| 无 ToolNode（chat_service 已用 create_agent） | 🟢 Low | `chat_service.py:220` |
| 仓库 sa.Table 重复定义 | 🟡 Medium | 7 个 repository 文件 |
| JSON 文本列（SQLite JSON1 可查询） | 🟢 Low→Med | `analysis.py` 等 |
| Verdict 转换逻辑重复 | 🟡 Low | `orchestrator.py:320` + `panel_service.py:220` |
