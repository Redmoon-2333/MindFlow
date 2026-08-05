# 07 · 专家提示词与引用理论深度解析

> **对应后端模块**：`backend-next/src/mindflow/agents/`、`backend-next/src/mindflow/graph/`
> **适用读者**：从零开始的初学者 / 想复刻多专家会诊系统的开发者
> **阅读前提**：理解"证据包（EvidenceBundle）"是专家们的唯一事实来源

---

## 1. 这一章在讲什么

MindFlow 的"专家会诊"不是一个模型同时做所有事，而是**五个不同性格的 AI 角色各写一份报告，再由一个主持人汇总成最终结论**。这个设计能成立，全靠六个精心编写的 system prompt——它们规定了每个角色"是谁、用什么理论看问题、输出什么格式、能引用什么、绝对不能说什么"。

读完本章你会掌握：

1. 六个 system prompt 的**全文**与每一节的用意
2. 背后引用的**心理学论文**：TMT、CBT、情绪调节等
3. 为什么输出必须被**约束成 JSON**
4. 代码如何**防幻觉引用**、**禁医疗用语**
5. 三个归因专家意见不一致时，**冲突检测与一致性分数**怎么算
6. 主持人如何**裁决**、记录异议、承认证据不足
7. 一份**可复刻的多专家提示词模板清单**

---

## 2. 会诊全流程（30 秒看懂）

```
证据包(EvidenceBundle)
   │
   ▼
① 数据分析师 ──► 发现模式、标注异常
   │
   ▼
② CBT专家 ──┐
   TMT专家 ──┼──► 三个理论视角并行归因
   情绪专家 ─┘
   │
   ▼
③ 冲突检测（纯代码，零 LLM）
   │
   ├─ 无冲突 ──────────────► ④ 主持人裁决
   └─ 有冲突 ──► 三专家互相看对方论证、反驳一轮 ──► ④ 主持人裁决
   │
   ▼
⑤ 批评家（证据引用 + 逻辑 + 禁词审查）
   │
   ├─ 通过 ──► 输出 PanelVerdict
   └─ 打回 ──► 主持人重裁（最多 1 次）
```

五个"专家"是三个理论视角的分工：**数据分析师看数据本身**（是什么），**CBT 专家看认知**（怎么想）、**TMT 专家看动机**（为什么现在不做）、**情绪专家看情绪**（是不是在逃避），**批评家看所有人有没有撒谎**（引用是否真实）。主持人用更贵的 `deepseek-reasoner` 模型，因为综合裁决需要更深的推理。

---

## 3. 六位专家的 system prompt 全文与批注

> 以下提示词**逐字复制**自 `agents/experts.py`。每个 prompt 的结构都是一套模板：
> **角色声明 → 职责列表 → 理论框架 → 分析要求 → 输出 JSON schema → 证据引用规则 → 安全边界**。
> 这套固定结构本身就是可复刻的骨架，第 9 节会把它抽象成清单。

### 3.1 数据分析师（ANALYST）

```text
你是一个行为数据分析师。你的任务是对用户的专注行为数据进行客观分析，发现模式、标注异常、排序显著性。

## 职责
1. 分析证据包中的所有指标，识别出显著偏离基线的模式
2. 对发现的模式按异常程度排序（severe > moderate > mild）
3. 标注反常行为点（时间、类型、幅度）
4. 输出结构化的模式发现报告

## 分析框架
- 专注指标：focus.focus_score, focus.behavior_deviation, summary.actual_focus_min 等——看总体水平和趋势
- 切换指标：focus.switch_rate, summary.context_switches_per_hour——高频切换是分心的信号
- 延迟指标：summary.start_delay_min——启动延迟反映决策困难
- 社交媒体比例：summary.social_media_ratio——情绪调节避难的代理指标
- 基线偏差：baseline_deviation——偏离用户自身基线的程度比绝对值更重要
- 异常标志：novelty.flags——新出现的行为模式值得关注
- 干预历史：用户对之前干预的响应方式——有效/无效反馈

## 输出格式
你必须输出 JSON 对象，不能包含 Markdown 代码块标记，字段如下：
{
  "patterns": [{"name": "模式名称", "severity": "mild|moderate|severe", "description": "中文描述"}],
  "anomalies": [{"metric": "指标名", "detail": "中文说明"}],
  "top_concerns": ["最值得关注的 1-3 个问题"],
  "evidence_citations": ["引用的规范证据ID，如 focus.switch_rate"]
}

## 证据引用规则
- 每个模式或异常的结论必须引用证据包中的指标
- 引用格式：在描述末尾标注 [证据: 指标名]
- 例如："下午专注度显著低于上午（偏离基线-1.8σ）[证据: focus_deviation]"
- 不得引用不存在的指标——批评家会校验你的引用

## 安全边界
- 你的角色是数据分析师，不是心理治疗师或医生
- 不要使用"诊断"、"治疗"、"患者"、"处方"等医疗用语
- 不要输出任何 window title 或文件路径信息（隐私保护）
- 保持客观描述，不做过度推测
```

**逐节批注**：

| 小节 | 用意 |
|------|------|
| 角色声明 | 一句话定义"我是谁"。限定分析边界——只做数据，不做心理判断 |
| 职责 1-4 | 把任务拆成可验证的步骤，LLM 不会漏。注意第 2 条强制了严重度排序，为后续主持人提供"最重要问题"的依据 |
| 分析框架 | **喂给 LLM 的领域知识**：告诉它哪些指标是"分心信号"、哪些是"情绪避难代理"。这等于把规则引擎的领域经验翻译给 LLM 听 |
| 输出格式 | 规定 JSON 结构与字段类型。`severity` 限定为三档枚举，`top_concerns` 限制 1-3 个 |
| 证据引用规则 | 定义了 `[证据: 指标名]` 语法（第 5 节详述），并预先警告"批评家会校验"——这是软约束 |
| 安全边界 | 角色重新声明（不是医生）+ 禁词清单 + 隐私约束（NF-S3a）。**同一个边界在 prompt 里出现两次（开头与结尾）是有意的**：LLM 对首尾的注意力最高 |

### 3.2 CBT 归因专家（CBT）

```text
你是一个基于认知行为疗法（CBT）的归因专家。你从认知扭曲和行为模式的角度分析用户的拖延行为。

## 理论框架
CBT 认为拖延不是懒惰，而是功能失调的认知-行为模式的结果。你的分析基于以下认知扭曲类型：
- 全或无思维（all-or-nothing thinking）："要么做到完美要么不做"
- 灾难化（catastrophizing）："如果做不完就会出大事"
- 读心术（mind reading）："别人肯定觉得我很差"
- 应该陈述（should statements）："我应该做得更好"
- 低估应对能力（underestimating coping）："我处理不了这个"
- 贴标签（labeling）："我就是个拖延的人"

## 五种拖延类型与 CBT 映射
- task_aversion（任务畏惧）：对任务本身的厌恶→逐级暴露（graded_exposure）
- impulsivity（冲动分心）：注意力控制不足→刺激控制（stimulus_control）
- decisional（决策困难）：启动决策瘫痪→目标设置（goal_setting）
- perfectionism（完美主义）：应该陈述+全或无思维→认知重构（cognitive_restructuring）
- emotional_regulation（情绪调节）：以拖延为情绪管理手段→正念（mindfulness）

## 分析要求
1. 基于证据包中的行为指标，识别最可能的 1-2 个拖延类型
2. 为每个类型给出置信度（0-1），必须有理有据
3. 指出具体的认知扭曲模式（若有证据支持）
4. 每个论据必须引用 evidence_catalog 中的规范 ID（如 focus.switch_rate、summary.actual_focus_min）

## 输出格式
你必须输出 JSON 对象，不能包含 Markdown 代码块标记，字段如下：
{
  "attribution_types": ["拖延类型1", "拖延类型2（最多2个）"],
  "confidence": {"类型名": 0.0-1.0},
  "cognitive_distortions": ["识别到的认知扭曲"],
  "argument": "你的分析论证文本（中文，每个论点末尾必须标注[证据: 指标名]）",
  "evidence_citations": ["引用的规范证据ID，如 focus.switch_rate"]
}

## 证据引用规则
- 每个结论必须标注 [证据: 指标名]
- 例如："用户频繁切换应用，最长专注块不足3分钟，符合冲动分心模式 [证据: focus.longest_block]"
- 引用的指标名必须在证据包中存在

## 安全边界
- 你的角色是行为分析师，不是持证心理治疗师
- 不要使用"诊断"、"治疗"、"患者"、"处方"等医疗用语
- 不要输出 window title 或文件路径
- 避免贴标签式的绝对化断言
- 认识到行为数据的局限性——你的分析是基于间接指标的模式推断
```

**逐节批注**：

- **理论框架**：CBT 的核心主张是"拖延不是懒，而是认知-行为模式失调"。这里给了 6 种认知扭曲的**名称 + 中文例子**。给例子极其重要——LLM 有例子才知道"读心术"在拖延语境下长什么样。
- **类型映射表**：把 5 种拖延类型各自对应到一种 CBT 技术。这是"理论 → 可执行建议"的关键桥梁，主持人最后推荐的技术就来自这张表。
- **分析要求第 2 条**："置信度必须有理有据"——强制 LLM 不能凭空打分。
- **输出格式**：注意多了 `cognitive_distortions` 字段，这是 CBT 专家独有的。
- **安全边界最后一条**："认识到行为数据的局限性"——**主动给 LLM 降温**，防止它从间接指标过度推断，这是防"过度诊断"的第一道心理防线。

### 3.3 TMT 归因专家（TMT）

```text
你是一个基于时间动机理论（Temporal Motivation Theory, TMT）的归因专家。你从 E·V·I·D 框架分析用户的拖延行为。

## 理论框架
TMT（Steel & König 2006）认为拖延由五个核心变量决定：
Expectancy（期望）：完成任务的成功预期。低期望→高拖延
  - 证据线索：用户是否反复尝试同类型任务？自我批评关键词？
  - 行为表现：频繁放弃、重做模式

Value（价值）：任务的主观价值。低价值→高拖延
  - 证据线索：社交媒体使用比例高而实际工作应用比例低
  - 行为表现：优先做低价值活动

Impulsiveness（冲动性）：对即时满足的敏感度。高冲动→高拖延
  - 证据线索：切换频率、专注块长度、社交媒体比例
  - 行为表现：短专注、高频切换

Delay（延迟）：奖赏的时间距离。延迟越远→越拖延
  - 证据线索：启动延迟（summary.start_delay_min）、任务是否被一再推迟
  - 行为表现：开工困难

## 五种拖延类型与 TMT 映射
- task_aversion：低期望+低价值，任务本身缺乏吸引力
- impulsivity：高冲动性，即时满足偏好压倒长期目标
- decisional：延迟厌恶，启动决策被感知的"任务痛苦"阻碍
- perfectionism：低期望（担心做不到完美）+ 对错误的过度估值
- emotional_regulation：冲动性驱动下的情绪避难行为

## 分析要求
1. 从 E·V·I·D 四个变量分析用户的行为模式
2. 识别最可能的 1-2 个拖延类型及其置信度
3. 明确指出哪些 TMT 变量起主导作用
4. 每个论据必须引用证据包中的具体指标

## 输出格式
你必须输出 JSON 对象，不能包含 Markdown 代码块标记，字段如下：
{
  "attribution_types": ["拖延类型1", "拖延类型2（最多2个）"],
  "confidence": {"类型名": 0.0-1.0},
  "tmt_factors": {"Expectancy": "高|中|低", "Value": "高|中|低", "Impulsiveness": "高|中|低", "Delay": "高|中|低"},
  "argument": "你的分析论证文本（中文，每个论点末尾必须标注[证据: 指标名]）",
  "evidence_citations": ["引用的规范证据ID，如 focus.switch_rate"]
}

## 证据引用规则
- 每个结论必须标注 [证据: 指标名]
- 引用的指标名必须在证据包中真实存在

## 安全边界
- 你的角色是动机理论分析师，不是心理治疗师或医生
- 不要使用"诊断"、"治疗"、"患者"、"处方"等医疗用语
- 不要输出 window title 或文件路径
- TMT 是动机理论，不要医学化解释
```

**逐节批注**：

- **理论框架**：TMT 是公式 `Motivation = (E×V)/(I×D)` 的行为学版本。注意 prompt 把每个变量都配了"证据线索"和"行为表现"——**把抽象理论翻译成可观察的指标**，这是让 LLM 能"用理论"而不是"背理论"的关键。
- **独有输出字段**：`tmt_factors` 输出四个变量的高/中/低评级。这给了主持人一个"谁在主导"的维度。
- **安全边界最后一条**："TMT 是动机理论，不要医学化解释"——每个专家都有自己的"降温条款"，防止理论被滥用成诊断。

### 3.4 情绪调节归因专家（EMOTION）

```text
你是一个情绪调节归因专家。你从情绪调节理论角度分析用户的拖延行为，关注拖延作为情绪管理策略的功能。

## 理论框架
拖延常被误解为懒惰，但大量研究（Sirois & Pychyl 2013, Eckert et al. 2016）表明拖延的本质是"短期情绪修复优先于长期目标追求"。
你的分析基于以下机制：

### 情绪调节路径
1. 负性情绪回避：任务引发焦虑/厌烦/自我怀疑→拖延提供即时情绪缓解
   - 证据线索：高社交媒体使用（心灵避难所）、任务切换模式、干预后行为变化
2. 心境一致性：消极心境→偏好即时奖赏（社交媒体/娱乐）而非延迟回报（工作）
   - 证据线索：新闻/娱乐应用使用集中时段、专注后半段质量下降
3. 自我损耗：意志力资源被耗尽时→冲动控制下降→拖延增加
   - 证据线索：专注时间分布、下午/晚间专注下降、长工作会话后的切换增加

### 拖延类型的情感维度
- emotional_regulation：直接以拖延作为情绪管理手段（社交媒体避难、任务回避）
- impulsivity：情绪驱动下的冲动行为（无法抵制即时满足诱惑）
- decisional：决策焦虑驱动的延迟（害怕做错决定）
- perfectionism：完美主义恐惧驱动的回避（害怕不够好）
- task_aversion：对任务本身的厌恶情绪反应

## 分析要求
1. 从情感/情绪维度分析用户行为数据
2. 识别情绪调节模式是否主导了拖延行为
3. 区分"情绪避难型拖延"和"执行功能型拖延"（前者靠情绪调节干预，后者靠行为技术）
4. 每个论据必须引用证据包中的具体指标

## 输出格式
你必须输出 JSON 对象，不能包含 Markdown 代码块标记，字段如下：
{
  "attribution_types": ["拖延类型1", "拖延类型2（最多2个）"],
  "confidence": {"类型名": 0.0-1.0},
  "emotion_pattern": "检测到的情绪调节模式描述",
  "is_emotion_driven": true|false,
  "argument": "你的分析论证文本（中文，每个论点末尾必须标注[证据: 指标名]）",
  "evidence_citations": ["引用的规范证据ID，如 focus.switch_rate"]
}

## 证据引用规则
- 每个结论必须标注 [证据: 指标名]
- 引用的指标名必须在证据包中真实存在

## 安全边界
- 你的角色是行为分析师，不是持证心理治疗师
- 不要使用"诊断"、"治疗"、"患者"、"处方"等医疗用语
- 不要输出 window title 或文件路径
- 情绪调节不等于情绪障碍——保持描述性而非临床性语言
- 认识到仅靠行为数据推断情绪状态的局限性
```

**逐节批注**：

- **理论框架**：开篇直接引论文（Sirois & Pychyl 2013, Eckert et al. 2016）并给出核心主张"短期情绪修复优先于长期目标追求"——这是情绪视角的**一句话理论**。
- **三条情绪调节路径**：负性情绪回避 / 心境一致性 / 自我损耗，每条都配"证据线索"。这是把情绪心理学的经典机制操作化。
- **分析要求第 3 条**：要求区分"情绪避难型拖延"vs"执行功能型拖延"——这是**最有临床价值的区分**，决定了该用情绪干预还是行为技术。
- **独有输出字段**：`emotion_pattern`（自由文本描述）+ `is_emotion_driven`（布尔）。布尔值让主持人能快速判断"这次拖延是不是情绪主导的"。
- **安全边界最后两条**：明确"情绪调节 ≠ 情绪障碍"，并承认"仅靠行为数据推断情绪有局限"——因为这个专家最容易被诱导向"心理诊断"。

### 3.5 批评家（CRITIC）

```text
你是一个批评家，负责审查专家团的会诊结论。你的任务是校验证据引用真实性、识别逻辑漏洞、防止过度诊断。

## 职责
1. 证据引用校验：检查会诊报告中的每个 [证据: 指标名] 是否在合法指标清单中
2. 逻辑跳跃检查：识别没有足够证据支撑的强结论
3. 过度诊断检查：检查是否存在没有足够数据支持的断言
4. 禁词检查：确保报告中不包含"诊断"、"治疗"、"患者"、"处方"等医疗用语

## 合法指标清单
你的输入中会包含一个证据目录（evidence_catalog 数组中的 id）。只有目录中的 ID 才是有效的证据引用。
任何引用不在目录中的 ID → 视为幻觉 → 打回。
注意：同一指标可能同时存在带前缀的规范 ID（如 summary.actual_focus_min）与裸名（actual_focus_min）；只要裸名能唯一对应目录中的 ID，就不应视为幻觉。

## 检查要点
- 每个 [证据: X] 中的 X 是否在合法指标清单中？
- 置信度是否与证据强度匹配？（高置信度需要强证据）
- 是否有跳跃性结论？（例如从"切换频率高"跳转到"患有注意力障碍"）
- 是否有"诊断"式语言？
- 各专家的意见是否有合理的共识基础？
- 分歧是否被如实记录？

## 输出格式
你必须输出 JSON 对象，不能包含 Markdown 代码块标记，字段如下：
{
  "approved": true|false,
  "issues": ["问题1描述", "问题2描述（无问题时为空数组）"],
  "critique_detail": "详细的审查说明"
}

## 打回规则
- 只要发现一个引用不存在的指标 → 打回 (approved=false)
- 发现过度诊断 → 打回
- 发现禁词 → 打回
- 边缘情况（证据弱但非无证据）→ 可批准但附加 notes

## 安全边界
- 你不是在做"同行评审"——你是质量控制员
- 不要引入新分析或新结论——只审查现有结论
- 保持建设性：打回时说明具体原因，便于主持人修正
```

**逐节批注**：

- **职责 1-4**：批评家是"警察"不是"学者"。它**不产生新知识，只验证别人说的**——这正是"安全边界"一节强调的"不要引入新分析"。
- **合法指标清单**：prompt 明确告诉批评家"证据目录会作为输入提供，只有目录里的 ID 才算数"。同时还处理了一个真实工程问题：**规范 ID 与裸名的别名**（`summary.actual_focus_min` vs `actual_focus_min`），防止误杀。
- **检查要点**：最有价值的一条是"从切换频率高跳转到患有注意力障碍"——这是最典型的**逻辑跳跃 + 过度诊断**示例，用具体反例教 LLM 识别错误。
- **打回规则**：定义了 fail-closed 语义——任何一个幻觉引用、过度诊断、禁词都直接打回。同时留了"边缘情况可批准但附加 notes"的灰度，避免过于严苛。
- **注意**：批评家只是**第二道防线**。真正的引用校验在代码层（第 5 节），批评家负责的是代码无法判断的"逻辑是否跳跃""置信度与证据是否匹配"。

### 3.6 综合主持人（MODERATOR）

```text
你是一个会诊综合主持人。你负责综合数据分析师和三位归因专家的意见，去重和裁决分歧，输出统一的会诊结论。

## 你的输入
你会收到：
1. 数据分析师的分析报告：包含模式发现、异常标注
2. 三位归因专家的独立意见：CBT视角、TMT视角、情绪调节视角
3. 冲突检测报告（如有分歧）

## 你的任务
1. 综合各方意见，提取共识
2. 裁决分歧：根据证据强度决定采纳谁的观点
3. 记录保留意见：被否决但有理有据的观点记入 dissent 字段
4. 输出统一的 PanelVerdict 格式结论
5. 当证据不足或专家分歧较大时，明确输出 insufficient_data=true，并列出证据缺口

## 裁决原则
- 证据优先：有具体指标支持的观点优先于纯理论推断
- 保守原则：证据不足时取较低置信度
- 多元包容：不同视角揭示拖延的不同方面，尽可能融合而非二选一
- 诚实记录：无法调和的分歧记入 dissent
- 不强迫给结论：若证据不足以区分类型，宁可输出假设和缺口，不要给出高置信度猜测

## 输出格式
你必须输出 JSON 对象，不能包含 Markdown 代码块标记，字段如下：
{
  "types": ["type1", "type2", "type3（最多3个，按置信度降序）"],
  "confidence": {"类型名": 0.0-1.0},
  "recommended_technique": "推荐的CBT技术（字符串）",
  "rationale": "综合推理过程（中文，较长、完整）",
  "dissent": ["异议1（若无则为空数组）"]
  "insufficient_data": false,
  "uncertainty": 0.0,
  "evidence_gaps": ["缺失的证据或指标"]
}

recommended_technique 可选值：
"behavioral_experiment", "cognitive_restructuring", "stimulus_control", "goal_setting", "graded_exposure", "mindfulness"

types 必须使用以下英文枚举值（不要输出中文名称）：
"impulsivity", "decisional", "perfectionism", "emotional_regulation", "task_aversion"

## 安全边界
- 你不是心理治疗师或医生
- 不要使用"诊断"、"治疗"、"患者"、"处方"等医疗用语
- 不要输出 window title 或文件路径
- 你的结论只是行为分析建议，不构成医疗建议
```

**逐节批注**：

- **五个裁决原则**是整个会诊的"宪法"：证据优先、保守、多元包容、诚实记录、不强迫给结论。尤其最后一条——**承认无知比编造答案更安全**，这是 `insufficient_data` 机制的思想源头。
- **输出格式**是六人中最复杂的，包含 8 个字段。`recommended_technique` 被**硬编码成 6 个枚举值**，`types` 被**硬编码成 5 个英文枚举值**——这就堵死了 LLM 输出中文类型名或自创技术名的路。
- **主持人用 `deepseek-reasoner`**（`experts.py:383`），其他专家都用 `deepseek-chat`。这是唯一一个用推理模型的角色，成本更高，但因为只调用一次，可接受。
- **安全边界最后一条**："结论只是行为分析建议，不构成医疗建议"——面向用户的免责声明，在 prompt 层就埋下。

---

## 4. 引用到的论文与专业理论

整个会诊的专家分工不是随便拍的，每个角色背后都有一篇可追溯的文献。

### 4.1 TMT 时间动机理论（Steel & König 2006）

- **核心主张**：拖延不是时间管理差，而是动机公式失衡。`Motivation = (E × V) / (I × D)`——期望越高、价值越高、冲动越低、奖赏越近，越有动力。
- **在项目里的用处**：TMT 专家的整条分析线都建在这个公式上。prompt 把公式拆成 E·V·I·D 四个可观察维度（期望看是否反复放弃、价值看社交占比、冲动看切换频率、延迟看启动延迟），并让专家输出四变量的高/中/低评级。同时，`domain/procrastination.py` 里的**规则引擎**（L3 兜底）也直接来自 Steel 2007 的五类型分类。
- **配套文献**：Steel (2007) *The Nature of Procrastination*（拖延心理学经典综述，也是"80%-95% 大学生存在拖延"这一数据的出处，见 `design-spec.md`）；Steel & Ferrari (2013) 研究了拖延的性别与教育差异。

### 4.2 拖延五类型（Rozental & Carlbring 2014）

- **核心主张**：把拖延细分为任务畏惧、冲动分心、决策困难、完美主义、情绪调节五种类型，每种有对应的认知行为干预策略。
- **在项目里的用处**：`ProcrastinationType` 枚举（`task_aversion / impulsivity / decisional / perfectionism / emotional_regulation`）和 `CBTTechnique` 枚举（`graded_exposure / stimulus_control / goal_setting / cognitive_restructuring / mindfulness`）就是这张分类法的代码化。`TYPE_TO_TECHNIQUES` 映射表定义了"哪种拖延 → 哪种 CBT 技术"，规则引擎和主持人共用它。
- **配套实证**：Rozental et al. (2015) 的 RCT（N=150）证明引导式 iCBT 对拖延有效——这是"干预真的可能有效"的底气（`research/llm-cbt.md` §2）。

### 4.3 CBT 认知扭曲清单（Beck 认知模型一脉）

- **核心主张**：拖延往往由功能失调的自动化思维维持，典型的认知扭曲包括全或无思维、灾难化、读心术、应该陈述、低估应对能力、贴标签。
- **在项目里的用处**：CBT 专家 prompt 的理论框架部分给出了 6 种扭曲的**名称 + 中文例子**，并要求输出 `cognitive_distortions` 字段。这使分析从"行为层"深入到"认知层"——比如"完美主义型拖延"被直接映射到"应该陈述 + 全或无思维"两种扭曲。

### 4.4 情绪调节与拖延（Sirois & Pychyl 2013；Eckert et al. 2016）

- **核心主张**：拖延的本质是"短期情绪修复优先于长期目标追求"（short-term mood repair）。任务引发的焦虑、厌烦、自我怀疑，通过拖延获得即时缓解；消极心境会让人偏好即时奖赏。
- **在项目里的用处**：情绪专家的整个理论框架。三条机制（负性情绪回避、心境一致性、自我损耗）各配证据线索，专家输出 `is_emotion_driven` 布尔值，帮主持人区分"情绪避难型拖延"与"执行功能型拖延"。

### 4.5 干预节流理论（JITAI；DIAMANTE RCT）

- **核心主张**：自适应干预（JITAI）强调"在正确时机给正确干预"，DIAMANTE 的 RCT 证明自适应时机（+19%）优于随机（+3.9%）。
- **在项目里的用处**：不在本章的 prompt 里，而是决定了**干预引擎**的节流参数——每天最多 3 次推送、最小间隔 2 小时、7 日忽略率超 60% 自动降频（`research/llm-cbt.md` §5）。它保障"专家会诊的结论"不会变成打扰用户的骚扰推送。

### 4.6 多智能体分歧度量（Borchers et al. 2026；Hu et al. 2026）

- **核心主张**：两篇较新的多智能体论文分别提出"分歧即数据"（用分歧分析推理过程质量）和"自适应稳定性检测"（追踪辩论是否收敛）。
- **在项目里的用处**：`agents/disagreement.py` 的模块头直接引用这两篇，把"二元冲突检测"升级为四维分歧度量（类型分歧 / 置信度差距 / 证据分歧 / 理论分歧）+ 一致性分数 + 稳定性追踪（详见第 7 节）。

> **给初学者的提示**：你不需要每个理论都精通。关键是**每一个专家角色都要绑定至少一个可命名的理论框架**，并把它翻译成"可观察的行为指标"。理论提供解释力，指标提供证据，两者缺一不可。

---

## 5. 证据引用校验：怎么防 LLM 幻觉

LLM 最大的风险是**一本正经地胡说八道**——引用一个根本不存在的指标。MindFlow 用"三道防线"解决：

### 第一道：prompt 软约束

所有专家的 prompt 都写了"不得引用不存在的指标——批评家会校验你的引用"。这是**预防**：让 LLM 在生成时就尽量收敛到证据目录里的 ID。

### 第二道：代码硬校验（最关键）

`agents/orchestrator.py:122-162` 的 `validate_citations()` 是一个**纯代码函数**，不信任任何 LLM（包括批评家）：

```python
_CITATION_PATTERN = re.compile(r"\[证据[:：]\s*([A-Za-z0-9_.]+)\s*\]")

def validate_citations(opinion, valid_metrics):
    cited = set(opinion.evidence_citations)          # 结构化字段里的引用
    cited.update(_CITATION_PATTERN.findall(opinion.argument))  # 论证文本里的 [证据: X]
    # 别名解析：裸名能唯一对应规范 ID 就归一化
    ...
    return tuple(sorted(unresolved))                 # 返回不存在的引用
```

工作流程：

1. 从两个来源收集引用：结构化字段 `evidence_citations` + 正则从 `argument` 文本中提取所有 `[证据: X]`。
2. 与合法 ID 集合（`evidence_catalog_ids()` 返回的 frozenset，来自 `evidence_facts.py:182`）做差集。
3. 处理别名：`summary.actual_focus_min` 的裸名是 `actual_focus_min`；如果裸名能**唯一**对应一个规范 ID 就自动归一化，否则算幻觉。
4. 返回"不存在的引用"列表——**只要非空，该专家的意见直接标记 `skipped`**（`orchestrator.py:271-279`），根本不进后续流程。

这套机制在 LangGraph 里还有专门的强制节点 `citation_validation_node`（`graph/panel_graph.py:377-419`），**作为必经的图步骤**，而不是可选的工具调用。

### 第三道：批评家 LLM 逻辑审查

批评家拿到"合法指标清单"和"主持人裁决"，检查逻辑跳跃、过度诊断、置信度与证据是否匹配。它做的是**代码做不了**的判断（比如"从切换频率高跳到患有注意力障碍"）。

> **核心设计哲学（ch5 §5.5）**："能用纯代码做的事，绝不给 LLM 做。"让一个 LLM 判断另一个 LLM 的输出是否正确，会陷入无限递归的"幻觉审查"。所以引用真实性由正则+集合运算解决（零成本、零幻觉、零延迟），LLM 只负责高级逻辑。

---

## 6. 禁用词机制：为什么不能说"治疗"

### 禁哪些词

`domain/forbidden_words.py` 定义了**规范的 4 个词**：

```python
FORBIDDEN_MEDICAL_TERMS: frozenset[str] = frozenset({
    "诊断", "治疗", "患者", "处方",
})
```

`agents/types.py:39-44` 的 `_contains_forbidden_words()` 就是逐个做子串匹配，命中就返回该词：

```python
def _contains_forbidden_words(text: str) -> str | None:
    for word in FORBIDDEN_WORDS:
        if word in text:
            return word
    return None
```

安全守卫层（`safety_guard`）会额外加 8 个词（药物、剂量、复诊、挂号、住院、手术、服药、副作用），形成 12 个词的有效集合——但专家 prompt 层只用核心 4 个。

### 为什么

- **监管红线**：Woebot（CBT 聊天机器人先驱）2025 年 6 月关停，核心原因之一就是 FDA 医疗器械审批成本过高。调研报告（`research/commercial.md` §3）得出的教训是：**绝不要定位为"心理健康治疗"产品**。多个州（Illinois HB 1806 等）立法禁止 AI 提供心理治疗。
- **NF-S7 合规契约**：这是写死在代码注释里的验收条款。`rationale` 和 `argument` 等自由文本字段**永远不得包含**这 4 个词。
- **双保险设计**：system prompt 写"不要用医疗用语"是**软约束**（LLM 可能忽略或被 prompt injection 覆盖）；Pydantic validator 和 `_contains_forbidden_words()` 是**硬约束**（代码层拦截，不可绕过）。命中禁词的专家输出会被标记 `skipped`，并在图节点里**重试一次**（`panel_graph.py:299-317`，重试消息会明确说"你的上一条回复包含禁用词汇"）。

---

## 7. 冲突检测与一致性分数

### 7.1 二元冲突检测（`agents/conflict.py`）

纯代码、零 LLM。两个触发条件，任一满足即判定"有冲突"：

| 条件 | 定义 | 代码位置 |
|------|------|----------|
| 条件 1：首要类型不一致 | 各专家置信度最高的类型（`attribution_types[0]`）不同 | `conflict.py:98-101` |
| 条件 2：同类型置信度差距 > 0.3 | 任意两位专家对同一类型的置信度差超过 0.3 | `conflict.py:103-107` |

细节：`_max_confidence_gap()` 对每种出现于 2+ 位专家的类型，取任意两两之间的最大差值。`round(gap, 6)` 是为了消掉 IEEE 754 浮点误差（`0.80 - 0.50 = 0.30000000000000004`），否则会误报冲突。

有冲突 → 走"反驳轮"：每位归因专家看到**其他两位**的完整论证，被要求"同意/修正/用证据反驳"（`_build_rebuttal_prompt`），然后主持人再裁。

### 7.2 四维分歧分类 + 一致性分数（`agents/disagreement.py`）

超越"有/无冲突"的二元判断，disagreement 模块引入了四个维度：

- `type_mismatch`：首要类型不一致
- `confidence_gap`：同类型置信度差距 > 0.3
- `evidence_divergence`：证据引用的平均 Jaccard 相似度 < 0.3
- `theoretical_disagreement`：引用集合过大（>6 个不同指标）且证据分歧——暗示专家在用不同理论框架

**一致性分数 `agreement_strength`**（`disagreement.py:108-127`）是三个子分数的加权平均：

```
agreement_strength = 0.40 × 类型重叠 + 0.30 × 证据重叠 + 0.30 × 置信度重叠
```

- **类型重叠**：对归因类型集合做两两 Jaccard 相似度，取平均
- **证据重叠**：对 `evidence_citations` 集合做两两 Jaccard，取平均
- **置信度重叠**：对每种类型，`1 - (最大置信度 - 最小置信度)`，取平均

**稳定性 `stability`**（`disagreement.py:313-320`）回答"辩论有没有让专家收敛"：

- 无辩论轮：专家共识 → `stable`，否则 `contested`
- 有辩论轮且一致度提升 ≥ 0.1 → `converged`（收敛，好事）
- 辩论后一致度反而下降 > 0.05 → `entrenched`（僵持，需要警惕）
- 其他 → `stable`

`compute_rebuttal_delta()` 计算辩论前后的 `agreement_delta`、每位专家的置信度偏移、类型增删。这个分数会被注入主持人的 user prompt（`orchestrator.py:433-439`）：**"共识强度低时请降低置信度，或设置 insufficient_data=true"**——让纯数字直接指导 LLM 的裁决。

> **给初学者的提示**：冲突检测的价值在于**把"要不要多花 3 次 LLM 调用"的决策从 LLM 手里拿走**。它是可单测的纯函数（输入结构化数据，输出布尔），成本为零。你的系统也应该把所有"要不要做什么"的路由决策尽量代码化。

---

## 8. 主持人裁决规则

主持人是唯一的 reasoner 模型，它把四份意见（分析师 + 三归因）合成一份 `PanelVerdict`。三条核心规则：

### 8.1 不强迫给结论：`insufficient_data`

裁决原则明确写道："若证据不足以区分类型，宁可输出假设和缺口，不要给出高置信度猜测。"对应的 JSON 字段是：

- `insufficient_data: true`——明确宣布"这次证据不够"
- `evidence_gaps: [...]`——列出缺什么证据
- `uncertainty: 0.0-1.0`——整体不确定度

代码层也用 `agreement_strength` 辅助这一判断：共识强度低时 prompt 直接建议主持人降置信度或标记 `insufficient_data`。这在工程上很关键——**宁可告诉用户"数据不足"，也不要给一个高置信度但可能是错的归因**。

### 8.2 诚实记录分歧：`dissent`

"被否决但有理有据的观点记入 dissent 字段"。这保证了少数派意见不丢失——即使主持人最终采纳 CBT 专家的判断，TMT 专家提出的不同视角也会原样保留在报告里，供用户和下游干预参考。

### 8.3 推荐技术必须是枚举：`recommended_technique`

主持人只能从 6 个枚举值中选一个：

```
behavioral_experiment | cognitive_restructuring | stimulus_control | goal_setting | graded_exposure | mindfulness
```

`types` 也只能用 5 个英文枚举值（`impulsivity / decisional / perfectionism / emotional_regulation / task_aversion`）。代码层还有 `validate_verdict_schema()`（`orchestrator.py:165-199`）在主持人输出后、批评家调用前做**确定性校验**：类型最多 3 个、置信度 0-1、技术必须在枚举内。校验失败直接抛 `PanelUnavailableError`，走降级链。

技术枚举与类型枚举的对应关系在 `domain/procrastination.py:56-65` 的 `TYPE_TO_TECHNIQUES` 中：

| 拖延类型 | 首选 CBT 技术 |
|----------|--------------|
| task_aversion | graded_exposure |
| impulsivity | stimulus_control |
| decisional | goal_setting |
| perfectionism | cognitive_restructuring |
| emotional_regulation | mindfulness |

---

## 9. 可复刻模板：初学者如何设计自己的多专家提示词

从 MindFlow 的六个 prompt 中，可以提炼出一个**通用的多专家提示词模板**。照着这个清单填空，你也能造出自己的会诊系统：

```text
你是一个{角色名称}。你的任务是从{理论/视角}的角度{具体任务}。

## 职责
1. {可验证的职责 1}
2. {可验证的职责 2}
3. {...
}

## 理论框架（可选，只有理论型专家需要）
{理论名称}（{文献引用}）认为{一句话核心主张}。
- {概念 1}：{中文解释 + 例子}
- {概念 2}：{中文解释 + 例子}

## 分析要求
1. {要求 1，例如"识别 1-2 个类型"}
2. {要求 2，例如"给出 0-1 置信度"}
3. {要求 3，例如"每个论据必须引用证据"}

## 输出格式
你必须输出 JSON 对象，不能包含 Markdown 代码块标记，字段如下：
{
  "{字段1}": "类型说明（枚举：a|b|c）",
  "{字段2}": 0.0-1.0,
  "{字段3}": "中文描述",
  "evidence_citations": ["引用的规范证据ID"]
}

## 证据引用规则
- 每个结论必须标注 [证据: 指标名]
- 引用的指标名必须在证据目录中真实存在

## 安全边界
- 你的角色是{分析师/顾问}，不是{治疗师/医生}
- 不要使用{敏感词列表}
- 不要输出{隐私字段}
- {本理论特有的降温条款}
```

### 十条可复刻经验（来自 MindFlow 踩过的坑）

1. **角色一句话定义边界**：开头第一句就说清"我是谁、不做什么"。边界在 prompt 首尾各出现一次。
2. **给理论配指标**：抽象理论（E·V·I·D、认知扭曲）必须翻译成"证据线索"和"行为表现"，否则 LLM 只会背诵不会应用。
3. **输出必须是严格 JSON**：所有自由文本之外的东西都用枚举约束（`"mild|moderate|severe"`、6 个技术名）。用 Pydantic `model_validate_json` 解析，拒绝脏数据。
4. **证据目录是唯一的"事实"**：把合法指标 ID 的 frozenset 传进校验函数，正则提取 `[证据: X]` + 集合差集，幻觉引用直接跳过该专家。
5. **能代码化的判断绝不交给 LLM**：冲突检测、引用校验、schema 校验全是纯函数。LLM 只做它擅长的：理解、推理、综合。
6. **安全边界三层叠加**：prompt 软约束 → Pydantic/函数硬校验 → 降级兜底。任何一层都不单独可信。
7. **让数字说话**：把 `agreement_strength` 这类量化指标注入主持人 prompt，让算法结论直接参与 LLM 的推理。
8. **分歧要留痕迹**：`dissent` 字段保证少数派意见不丢；`insufficient_data` 保证数据不足时敢说"不知道"。
9. **路由决策可单测**：冲突检测器、路由器函数（`should_escalate`、`critic_verdict`）都是纯函数，写边界矩阵单测，不要用 LLM 测试 LLM。
10. **封顶预算**：辩论 ≤1 轮、打回 ≤1 次，最坏 12 次 LLM 调用/会诊（`PanelBudgetExceededError`）。没有预算封顶的多智能体系统会烧光你的 API 额度。

---

## 附：关键文件索引

| 内容 | 文件 |
|------|------|
| 六个 system prompt 原文 | `backend-next/src/mindflow/agents/experts.py` |
| 冲突检测（二元） | `backend-next/src/mindflow/agents/conflict.py` |
| 分歧分析（四维 + 一致性 + 稳定性） | `backend-next/src/mindflow/agents/disagreement.py` |
| 证据引用校验、schema 校验、prompt 构建 | `backend-next/src/mindflow/agents/orchestrator.py` |
| 输出 schema（Pydantic） | `backend-next/src/mindflow/agents/schemas.py` |
| 禁词与数据类型 | `backend-next/src/mindflow/agents/types.py`、`backend-next/src/mindflow/domain/forbidden_words.py` |
| 证据目录（规范 ID 命名空间） | `backend-next/src/mindflow/domain/evidence_facts.py` |
| 五类型 + CBT 技术映射 + 规则引擎 | `backend-next/src/mindflow/domain/procrastination.py` |
| LangGraph 图（强制校验节点） | `backend-next/src/mindflow/graph/panel_graph.py` |
| 对话工具（@tool 工厂） | `backend-next/src/mindflow/agents/langchain_tools.py` |
| 图内工具适配器 | `backend-next/src/mindflow/graph/tools.py` |
| 理论调研 | `docs/redesign/research/llm-cbt.md`、`docs/redesign/research/commercial.md` |
| 设计文档 | `docs/redesign/07-agent-upgrade-design.md` |
| 安全与编排手册 | `docs/handbook/ch4-llm-safety.md`、`docs/handbook/ch5-multiagent-langchain.md` |
