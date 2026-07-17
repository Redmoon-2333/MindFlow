# MindFlow 对标研究综合报告

> **文档编号**: 02-benchmark-research
> **日期**: 2026-07-17
> **作者**: 编排者综合 4 份并行调研（详见 `research/` 目录）
> **用途**: Gate 1 验收材料；需求分析（03）与架构设计（04）的决策依据

## 0. 报告索引

| 报告 | 文件 | 覆盖范围 |
|------|------|---------|
| 现状架构分析 | [01-project-analysis.md](01-project-analysis.md) | 现有后端逐模块判定、16 项技术债、商业化差距 Top 10 |
| 开源架构调研 | [research/oss.md](research/oss.md) | ActivityWatch / WakaTime / arbtt / Tockler / Selfspy |
| 商业竞品调研 | [research/commercial.md](research/commercial.md) | 国际 7 款 + 国内 4 款 + AI 心理干预 3 款 |
| LLM+CBT 工程 | [research/llm-cbt.md](research/llm-cbt.md) | 上下文工程 / CBT 提示 / 成本 / 安全 / eval |
| 生产工程实践 | [research/engineering.md](research/engineering.md) | SQLite / 进程 / 安全 / 跨平台 / 打包 / 可观测性 / API 规范 |

---

## 1. 十大关键结论（跨报告综合）

### ① 市场空白已验证（中等置信度）
无任何产品同时具备「自动行为追踪 + AI 根因分析（CBT）+ 自适应干预」三要素。最接近的 Rize AI Coach 停留在描述性分析；Youper 懂情绪但不关心工作场景。**立项书的差异化定位成立**，但建议调整叙事：把「聪明的自适应干预」做核心卖点，「根因分析」做支撑技术——用户为效果付费，不为理论付费。

### ② 监管红线明确（Woebot 之死 + 美国 36 州立法潮）
Woebot 2025.6 关闭的直接原因是「LLM 心理治疗」无监管路径。**MindFlow 必须定位为「效率教练/行为洞察工具」**，全线禁用「治疗/诊断/心理干预/患者」词汇；危机检测（自杀/自伤关键词）必须独立于 LLM、在 LLM 之前运行。这是 P0 架构约束，不是事后合规补丁。

### ③ 数据模型采用 Bucket + Event + Heartbeat（ActivityWatch 验证）
`timestamp + duration + data(JSON)` 统一事件三元组 + heartbeat 合并机制（pulsetime 窗口内相邻相同事件合并）。解决现有后端最严重的数据失真问题（duration 用配置值估算），且减少 90%+ 磁盘写。

### ④ 采集器架构：薄采集层 + 共享处理逻辑，同进程起步
- WakaTime 模式：平台特定代码压到最薄（50-200 行），共享逻辑（队列/合并/上报）进共享层
- 进程边界裁决：ActivityWatch 用多进程（因其开放 watcher 生态），MindFlow 当前复杂度**同进程线程模型足够**，但必须以 `CollectorService` 接口隔离，未来可无痛拆分
- 跨平台：PyWinCtl 统一 API（Win/macOS/Linux-X11），Wayland 降级为 pid 级采集

### ⑤ LLM 成本完全可控：¥2000 预算充裕一个量级
单用户全功能 ~¥77/年（DeepSeek 主力 + 规则引擎兜底）。**成本不是约束，可靠性才是**：必须实现三层降级链（API → 本地模型(可选) → 规则引擎），LLM 不可用时产品核心功能不消失。

### ⑥ 拖延类型标签体系可计算化
Steel TMT 公式（Motivation = E×V / I×D）+ Rozental CBT 框架 → 5 类可计算标签（任务畏惧/冲动分心/决策困难/完美主义/情绪调节），每类有行为特征规则 + 对应 CBT 策略。规则引擎先行、LLM 增强归因深度——这同时就是降级链的兜底层。

### ⑦ 干预疲劳节流是产品成败关键（立项书关键问题 3 的答案）
竞品失败模式一致：拦截太强 → 反感卸载；太弱 → 无效。JITAI 理论 + DIAMANTE RCT（自适应时机 +19% vs 随机 +3.9%）给出工程方案：每日上限 3 次、最小间隔 2h、深度工作不打扰、7 日忽略率 >60% 自动降频。**节流器是干预引擎的一等公民组件**。

### ⑧ localhost API 安全三层防护
随机 token 文件（600 权限）+ Bearer 认证 → Host header 校验（防 DNS rebinding）→ 前端同源托管。现状（无认证 + CORS 全开）在商业软件中不可接受。

### ⑨ SQLite 生产化 + 桌面迁移策略
busy_timeout/journal_size_limit/integrity_check 补齐；备份用 `VACUUM INTO`（WAL 下直接 cp 不可靠）；Alembic 启动时自动迁移 + 失败降级运行 + `render_as_batch=True`（SQLite 无 ALTER COLUMN）。

### ⑩ 交付链路：PyInstaller + tufup；可观测性 loguru + opt-in Sentry
打包 sklearn+hmmlearn 用 PyInstaller（官方支持，注意 joblib loky）；自动更新用 tufup（TUF 签名协议）；日志 loguru 滚动 JSON；崩溃上报必须 opt-in + 敏感信息过滤。

---

## 2. 已识别的决策冲突（需 Gate 1 裁决）

| # | 冲突 | 选项 | 综合建议 |
|---|------|------|---------|
| C1 | **立项书承诺 GitHub 开源** vs **商业化付费墙** | ①全开源+服务收费 ②open-core（核心开源+高级功能闭源）③开源延迟（结题后开源基础版） | open-core：采集/分析/API 开源（满足结题承诺+简历价值），LLM 编排与自适应干预策略闭源（付费核心） |
| C2 | **单机单用户** vs **多用户/云同步** | ①纯本地单用户 ②本地优先+账号系统预留 ③完整云同步 | ②：schema 保留 user_id 与同步游标字段，但 2027.5 前不实现云端——工期与预算约束 |
| C3 | **LLM 接入模式** | ①内置项目 key（预算付费）②BYOK ③混合（内置免费额度+BYOK 解锁无限） | ③混合，规则引擎保底免费可用 |
| C4 | **技术栈** | ①继续 Python/FastAPI ②换栈（Go/Rust） | ①：ML 生态 + 团队技能 + 可复用资产（数据集/算法）都在 Python 侧；「从零开始」指架构从零，不必换语言 |

## 3. 对需求阶段（03）的输入清单

1. 竞品功能矩阵 → MoSCoW 分级依据（见 commercial.md §4.3 付费墙优先级）
2. 五类拖延标签 + CBT 策略映射 → F3 功能需求的验收标准
3. 节流参数（3次/日、2h 间隔、疲劳降频）→ F4 非功能需求
4. 性能预算：采集 CPU <2%、内存 <50MB（沿用原设计目标，但需实测基准）
5. 安全需求：token 认证、Host 校验、危机检测、数据不上云、脱敏
6. 合规需求：文案词汇表（禁用词）、免责声明、opt-in 遥测
7. 可靠性需求：崩溃自动重启、离线队列、自动迁移+降级、每日备份
8. Eval 需求：golden set ≥200 条、安全违规零容忍门禁
