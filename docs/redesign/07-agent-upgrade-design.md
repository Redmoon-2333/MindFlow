# MindFlow 多专家智能体升级设计

> **文档编号**: 07-agent-upgrade-design.md
> **日期**: 2026-07-18
> **状态**: 已获用户批准（brainstorming 对谈 6 轮决策后定稿）
> **上游**: 04-architecture-design.md（现有后端架构）· research/llm-cbt.md
> **决策记录**: ①三级递进形态 ②ML=事实层/LLM=推理层 ③ML本地+LLM云端 ④分层交流协议 ⑤混合专家阵容 ⑥每日会诊+按需 ⑦自研轻量编排内核（vs LangGraph/多进程）

---

## §1 定位与三级路线图

```
L1 内部多专家会诊（必做）─→ L2 对话式助手（尽量）─→ L3 自主行动体（理想）
     专家团+编排内核            装上"声带"                装上"手脚"
```

三级共享同一个专家团内核：L1 建专家团，L2 加"用户提问→调度专家→回答"入口，L3 把"人触发"换成"自我触发+行动闭环"。一个内核的三次装配，不是三个系统。

## §2 总体架构：感知-认知-行动

```
┌─ L0 感知层（本地 ML，全部已有，零改动）───────────────────┐
│ 采集 → 专注分数 → Welford基线 → Z-score偏差+严重度         │
│ → HMM状态推断 → 聚类(模式发现) → 分类器 → 节律画像         │
└──────────────────┬────────────────────────────────────┘
                   ▼ EvidenceBundle（证据合同，新增）
┌─ L1 认知层（LLM 专家团，新增 agents/ 模块）───────────────┐
│  ①数据分析师 → ②归因组(CBT/TMT/情绪 三视角并行)            │
│  → [冲突检测器:纯代码] ─无冲突→ ④综合主持人 → 会诊报告      │
│         └─有冲突→ ③辩论轮(1轮反驳) → ④综合(记录分歧)       │
│  → ⑤批评家: 证据引用校验+逻辑审查, 可打回一次               │
└──────────────────┬────────────────────────────────────┘
                   ▼ PanelVerdict（会诊结论）
┌─ L2/L3 行动层（复用+升级现有干预引擎）─────────────────────┐
│ 干预策略 → 节流器守门 → 执行 → 效果评估 → 反馈进下次会诊     │
└───────────────────────────────────────────────────────┘
```

**ML 的三重角色**：①证据供给者（EvidenceBundle 全由 ML 产出）②辩论裁判依据（无 ML 证据支撑的观点被批评家打回）③廉价前哨（规则引擎零成本站岗，异常才召集专家团）。

## §3 EvidenceBundle — 证据合同（最重要接口）

```python
@dataclass(frozen=True)
class EvidenceItem:
    metric: str            # "focus_deviation" / "switch_rate" / "hmm_state_seq"...
    value: float | str
    baseline: float | None # 该用户该时段的基线值
    severity: str          # mild/moderate/severe（ML 低级判断）
    confidence: float
    source: str            # "welford_baseline" / "hmm" / "clustering"...
    human_readable: str    # "周四下午专注度比你的基线低2.3个标准差"

@dataclass(frozen=True)
class EvidenceBundle:
    user_id: int
    window: tuple[datetime, datetime]
    items: tuple[EvidenceItem, ...]
    behavior_summary: BehaviorSummary       # 复用现有脱敏摘要
    intervention_history: tuple[...]        # 近期干预+用户反应（效果反馈闭环）
    novelty_flags: tuple[str, ...]          # 聚类发现的新行为模式
```

**强制规则**：专家结论必须标注 `[证据: metric名]`；批评家校验引用真实性——引用不存在的 metric = 幻觉 = 打回。

## §4 专家团与两档交流协议

| 角色 | 职责 | 模型 |
|------|------|------|
| ①数据分析师 | 挖模式、排显著性、标反常点 | deepseek-chat |
| ②归因组×3 | CBT（认知扭曲）/TMT（E·V·I·D）/情绪调节 独立归因 | deepseek-chat 并行 |
| ③综合主持人 | 去重、裁决、输出 PanelVerdict | deepseek-reasoner |
| ④批评家 | 证据引用校验+逻辑漏洞+过度诊断检查，可打回一次 | deepseek-chat |

```
快速通道（默认 ~6 次调用）: 分析师 → 归因×3并行 → [冲突检测:纯代码] → 主持人 → 批评家
冲突升级（+3 次）: 判据 = top1类型不一致 ∨ 同类型置信度差>0.3 ∨ 批评家打回
                  动作 = 每位归因专家看其他两位完整论证，写一轮反驳/修正 → 重裁
封顶: 辩论≤1轮, 打回≤1次 → 最坏 12 次调用/会诊
成本: 每日1深度+日均1按需 ≈ 年 ¥110-250（预算 ¥2000 十倍余量）
```

## §5 与现有系统集成（零破坏）

| 现有资产 | 集成 |
|---------|------|
| DeepSeekClient/结构化输出/禁词校验 | 直接复用，专家=不同 system prompt 的调用 |
| 三层降级链 | 升级四层：专家团→单专家（今日模式）→Ollama→规则引擎 |
| 危机检测 | 不动，仍在一切 LLM 之前 |
| 节流器/干预引擎 | 不动，PanelVerdict 对齐 ProcrastinationAssessment 形状 |
| procrastination_analyses 表 | 加列 panel_transcript_json（专家发言记录，前端可展示——产品亮点） |
| auto_intervention job | 升级为 L3 前哨：显著异常→触发会诊（而非直接干预） |
| 调度器 | 新增每日 23:30 会诊 job |

## §6 L2 对话式助手（Phase B）

极简 agent loop（~200 行）：`POST /api/v1/chat` → 危机检测 → 意图解析（带工具清单的 LLM 调用）→ 工具循环（query_evidence / get_panel_verdict / run_panel / query_interventions——全是现有服务内部化）→ 生成带证据引用的回答。会话存 chat_sessions 表（10 轮上限，超出摘要压缩）。

## §7 L3 自主行动体（Phase C）

```
感知: 30min 规则引擎巡逻（零成本）→ 异常显著? → 召集会诊
决策: PanelVerdict + 干预历史效果 → 策略师选{时机,强度,措辞}（节流器一票否决）
行动: 干预推送 或 静默观察（"不打扰"也是决策）
学习: EffectivenessService 结果 → 注入下次 EvidenceBundle → 专家团自动偏向有效策略
```

安全边界（不可逾越）：节流上限不受专家投票影响；危机检测最高优先；preferences 一键暂停全部自主行为。

## §8 测试策略

| 层 | 方法 |
|----|------|
| 编排逻辑 | Mock LLM client（录制/回放），测并行调度/冲突判据/升级/打回/降级 |
| 专家输出契约 | Pydantic + 禁词 + 证据引用存在性校验 |
| 冲突检测器 | 纯函数边界矩阵单测 |
| 成本护栏 | 断言一次会诊 ≤12 次调用（计数器） |
| E2E | 真实服务器 + 本地假 LLM server 跑完整会诊 |
| 评估集 | 30 合成场景人工标注，单专家 vs 专家团命中率对比——结题实验数据 |

## §9 分期交付

| Phase | 内容 | 对应 ledger story |
|-------|------|-------------------|
| A（L1） | EvidenceBundle + 编排内核 + 集成 | G001-G003 |
| B（L2） | chat 端点 + 工具循环 | G004 |
| C（L3） | 前哨+决策器+反馈闭环 | G005 |
| 终局 | 评估集 + E2E + 三门禁 | G006 |

## §10 风险与诚实边界

1. 多专家≠必然更准（辩论对推理纠错有效、对事实获取无效——研究共识）→ 对策：证据合同+引用校验+评估集实测对比
2. 会诊延迟 20-60s → 定位为"每日报告+按需"，实时路径仍走规则引擎
3. DeepSeek 单点 → 四层降级链，最坏回退到现有体验

## §11 实施状态（2026-07-18 全部完成）

| Story | 内容 | Commits | 新增测试 |
|-------|------|---------|---------|
| G001 | EvidenceBundle 证据合同 + builder | bb4a8fa, a5ee68e | 75 |
| G002 | 编排内核（5 专家/冲突检测/两档协议/12 封顶/代码级引用校验） | 20d4ae3, 2e3785a | 43 |
| G003 | 集成（panel service/每日 23:30 job/端点限流/迁移 0002） | 608cb1a | 29 |
| G004 | 对话助手（4 工具 agent loop/危机前置/迁移 0003） | 26c52c
