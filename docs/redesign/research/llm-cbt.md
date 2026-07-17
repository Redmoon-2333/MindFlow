# LLM+CBT 拖延归因分析与个性化干预 — 工程实现方案调研报告

> **文档编号**: research/llm-cbt
> **日期**: 2026-07-17
> **作者**: research-llm-cbt agent（document-specialist, MED tier）
> **状态**: 6 大主题全部完成，事实附来源 URL；定价与法规数字为调研时点数据，实施时需复核
> **用途**: Gate 1 验收材料之一，LLM 模块设计（Phase 4）的直接输入

---

## 1. 行为数据→LLM 的上下文工程

### 业界做法
- **Rize AI 模式**：不在原始数据上做 LLM 推理——先统计聚合摘要，再输入 AI 分析层（rize.io/features/productivity）
- **时序推理研究（2025-2026）**：Time Series Reasoning 范式；Apple ML 轻量时序编码器+CoT 微调；时序转可视化输入多模态 LLM 比纯文本准确率高 15-20%（arxiv.org/html/2502.01477v2）
- **TimE 基准（NeurIPS 2025 Spotlight）**：LLM 在密集时间信息、快速事件动态上表现较差 → 必须做特征提取先行
- 7 种时序提示策略：上下文化结构/特征提取先行/混合 LLM+统计/Schema 化/预测/异常检测/领域注入（machinelearningmastery.com）

### MindFlow 设计建议
```
原始事件流 → 滑动窗口聚合(5min/30min/2h) → 特征提取
  ├ 注意力持续时间 P10/P50/P90
  ├ 上下文切换频率（次/小时）
  ├ 社交媒体占比%
  ├ 工作/休息节律图
  └ 与基线偏差评分（z-score）
→ 行为摘要 JSON → LLM 推理
```

**行为摘要 JSON Schema：**
```json
{
  "session": {"intended_task":"论文写作","duration_min":120,"actual_focus_min":32},
  "activity_timeline": [{"t":0,"app":"overleaf","dur":180},{"t":180,"app":"wechat","dur":300}],
  "metrics": {"context_switches_per_hour":18,"longest_focus_block_sec":420,"social_media_ratio":0.58,"productivity_score":0.27},
  "pattern_summary":"打开论文→3min→切微信→5min→循环",
  "baseline_deviation":{"focus_vs_typical":-0.62}
}
```

---

## 2. CBT 结构化提示工程与拖延类型标签体系

### 业界做法
- **Woebot**：结构化顺序模块，2025.6 关闭（监管+商业模式）
- **Wysa**：非线性对话，FDA 突破设备认定，30+ 同行评审研究（pmc.ncbi.nlm.nih.gov/articles/PMC12669916）
- **CRBot 研究（arXiv 2501.15599）**：LLM CBT 机器人的已知问题——权力不平衡、过度建议、误解用户状态、"过度积极"

### 拖延类型学（Steel 2007 TMT + Rozental & Carlbring 2014）
TMT 核心公式：**Motivation = (E × V) / (I × D)**

| 类型 | TMT 变量异常 | 行为表现 | CBT 策略 |
|-----|------------|---------|---------|
| 任务畏惧型 | V↓+E↓ | "太无聊/太难" | 暴露/价值澄清 |
| 冲动/分心型 | I↑ | 切换频繁 | 刺激控制 |
| 决策困难型 | D↑ | 不敢启动 | 目标设定 |
| 完美主义型 | E↓ | "做不好不如不做" | 认知重构 |
| 情绪调节型 | V 波动 | 逃避情绪 | 正念/即时奖赏 |

- Rozental et al. (2015) RCT：iCBT 对拖延有效（N=150），引导式 > 无引导 > 等待（pubmed.ncbi.nlm.nih.gov/25939016）

### 可计算标签体系（规则引擎起点）
```
IF 最长专注块<5min AND 切换>12次/h → "冲动/分心型"
IF 首次启动延迟>30min AND 启动后正常 → "决策困难型"
IF 含"不够好/失败"关键词 → "完美主义型"
IF 社交媒体>55% AND 趋近-回避循环 → "情绪调节型"
IF 任务评分"无聊" AND 自我效能够 → "任务畏惧型"
```

### CBT 干预 Prompt 骨架
```
## 角色
基于 CBT 的拖延干预教练。温和、鼓励但不纵容。
## 交互协议
1. 镜像确认："我注意到你过去2小时专注X分钟，比计划少Y分钟"
2. 归因探索：苏格拉底提问，不给结论
3. 认知重构：识别自动化思维，挑战证据
4. 行动约定：最小下一步
## 安全边界
- 不冒充治疗师；检测自杀/自伤→转介危机热线；每日最多3次主动推送
输出 JSON 格式
```

---

## 3. 结构化输出与可靠性

### 业界三级控制（2026）

| 级别 | 方法 | 可靠性 | 用例 |
|-----|------|-------|------|
| L1 | Prompt+后处理 | 80-95% | 原型 |
| L2 | Function Calling | 95-99% | Agent |
| L3 | 原生结构化/约束解码 | 语法 100% | 生产 |

- 关键实证：**格式约束不保证语义正确**（语法 0% 错误后仍有 5-15% 语义错误）→ 需业务规则验证+重试（arxiv 2606.09395; towardsai）
- 生态：Pydantic + Instructor/Outlines

### MindFlow 三层降级链（核心设计）
```python
async def analyze(segment):
    if cached := cache.get(key): return cached          # 0. 缓存
    try:
        return client.chat.completions.create(          # 1. 主路径：API + 结构化输出
            model="deepseek-chat",
            response_model=ProcrastinationAnalysis, ...)
    except (APIError, TimeoutError, ValidationError): pass
    try: return await local_model_fallback(segment)     # 2. 本地模型（Ollama，可选）
    except: pass
    return rule_engine.analyze(segment)                 # 3. 规则引擎兜底（¥0，永不失败）
```

**Pydantic 输出契约：**
```python
class ProcrastinationAnalysis(BaseModel):
    procrastination_types: list[Literal["task_aversion","impulsivity","decisional","perfectionism","emotional_regulation"]] = Field(min_length=1, max_length=3)
    type_confidence: dict[str, float]
    cognitive_distortions: list[str]
    cbt_technique: Literal["behavioral_experiment","cognitive_restructuring","stimulus_control","goal_setting","graded_exposure"]
    response_text: str = Field(max_length=500)
    next_action: str
```

---

## 4. 成本工程（¥2000/年预算）

### 模型成本对比（2026 调研时点，实施时复核）

| 模型 | 输入/MTok | 输出/MTok | 本地部署 |
|-----|:---------:|:---------:|:--------:|
| DeepSeek V3.2 | $0.27 | $1.10 | MIT |
| DeepSeek R1 | $0.12 | $0.20 | MIT |
| Gemini 2.0 Flash 免费层 | $0 | $0 | No |
| Qwen 3 8B (Ollama) | $0 | $0 | Apache |

### 年成本估算（分层路由）

| 调用类型 | 模型 | 单次成本 | 日频次 | 年成本 |
|---------|------|:-------:|:-----:|:------:|
| 行为归因 | DeepSeek V3.2 | ¥0.007 | 20 | ¥51 |
| 干预生成 | DeepSeek R1 | ¥0.003 | 15 | ¥16 |
| 分类/路由 | 本地/规则 | ¥0 | 50 | ¥0 |
| 周报摘要 | DeepSeek V3.2 | ¥0.02 | 4 | ¥10 |
| 危机检测 | 关键词+本地 | ¥0 | 全部 | ¥0 |
| **合计（单用户）** | | | | **~¥77** |

**结论：¥2000/年预算非常充裕**（单用户 <¥100/年；扩展 10 倍用户 ~¥770/年仍在预算内）。缓存命中（DeepSeek $0.014/MTok）+ 混合路由可再省 37-46%。BYOK（用户自带 key）作为商业化后的成本转嫁选项。

---

## 5. 安全与伦理边界

### 监管格局（2025-2026，美国为主，中国参考认知数字疗法专家共识）

| 法规 | 核心要求 | 生效 |
|------|---------|------|
| California SB 243 | 检测自杀意念+提供危机资源；禁止冒充；用户可起诉 | 2026-01 |
| Illinois HB 1806 | 禁止 AI 提供心理治疗 | 2025-08 |
| Nevada AB 406 / Tennessee SB 1580 | 禁止暗示提供心理保健/冒充专业人员 | 2025-2026 |

2026 Q1 已有 36 个州 70+ 项 AI 聊天机器人法案。Brown University 研究：AI 聊天机器人系统性违反心理健康伦理标准（15 项风险）。

### 必须内建的安全机制

```python
# 1. 危机检测 —— 独立于 LLM 流程，在 LLM 调用之前运行，零成本
class CrisisDetector:
    KEYWORDS = {"zh_crisis": ["自杀","不想活","结束生命","伤害自己","撑不下去"]}
    def on_detected(self):
        return CrisisResponse(message="全国24h心理援助热线", stop_cbt_dialog=True, log_incident=True)

# 2. 干预疲劳节流（JITAI 理论；DIAMANTE RCT：自适应时机 +19% vs 随机 +3.9%）
class InterventionThrottle:
    MAX_DAILY = 3          # 每天最多 3 次主动推送
    MIN_INTERVAL = 7200    # 最小间隔 2 小时
    MAX_SAME_TYPE = 2      # 同类每天最多 2 次
    # 深度工作状态不打扰；7 日忽略率 >60% 视为疲劳，自动降频
```

### 合规清单
1. 文案只用「效率分析/行为洞察/专注建议」，禁用「治疗/诊断/心理干预/患者」
2. 免责声明：首次使用 + 每月一次
3. 危机检测在 LLM 之前、不依赖 LLM
4. 行为数据不上云（本地优先架构天然满足）
5. 年龄声明；日志 30 天超期匿名化

---

## 6. 效果评估（离线 Eval 设计）

### 业界做法
- LLM-as-Judge 与人类一致性 80-90%，成本 1/5000-1/500（deepeval.com）；需明确评分标准+Judge 输出推理+多评委纠偏（JudgeBench ICLR 2025：注意位置偏见和冗长偏好）
- RAGAS 对齐循环：基线 75.6% → prompt 优化 → 86.9%
- 生产级 eval = 离线基准 + 生产监控 + 10% 人工抽检

### MindFlow Eval 设计
- **Golden set 200-400 条**：50% 真实行为摘要（匿名化）+ 30% TMT 理论合成 + 20% 边缘案例（混合/模糊/危机）
- **指标与目标**：

| 指标 | 最低 | 理想 |
|-----|:---:|:----:|
| 类型 Top-1 准确率 | >70% | >80% |
| Jaccard 相似度 | >0.5 | >0.65 |
| 干预技术匹配率 | >60% | >75% |
| LLM Judge 质量分(1-5) | >3.0 | >3.8 |
| 安全违规率 | **0%** | 0% |
| 危机召回率 | >90% | >95% |

- **持续评估**：每 2 周从真实日志抽 20 条模糊案例标注入集；prompt/模型变更自动跑 eval；**安全违规率 >0% 阻塞部署**

---

## 总结架构

```
行为追踪 → 特征聚合 → 类型分类(规则/本地/API) → 干预生成(CBT)
     ↓                                ↓
 危机检测(独立于LLM)               疲劳节流(JITAI)
     ↓                                ↓
 结构化输出 JSON → 降级链(API→本地→规则) → 用户

模型栈（年成本 <¥100，预算 ¥2000 充裕）:
  DeepSeek V3.2(主力归因) + R1(复杂推理) + 本地 Qwen 3 8B(可选) + 规则引擎(兜底)
安全: 独立危机检测 + prompt 安全规则 + 免责声明 + 节流
评估: golden set + LLM Judge + 10% 人工抽检 + 安全违规零容忍
```

## 主要来源
rize.io; machinelearning.apple.com/research/towards-time; arxiv 2502.01477/2501.15599/2606.09395; neurips.cc TimE; en.wikipedia.org Temporal_motivation_theory; pubmed 25939016; pmc PMC12669916/PMC5891720; techsy.io; deepeval.com; docs.ragas.io; ailawsbystate.com; manatt.com; brown.edu; frontiersin.org JITAI; jmir.org/2024/1/e60834; modelmomentum.com; teamai.com; getmaxim.ai; aisuperior.com
