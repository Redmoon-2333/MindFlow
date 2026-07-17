# MindFlow backend-next 测试与质量验收报告

> **文档编号**: 05-testing-report.md
> **日期**: 2026-07-17
> **状态**: Gate 5 终局验收材料
> **范围**: backend-next/ 完整重写（Wave 1-9）

---

## 1. 自动化测试

| 指标 | 结果 | 要求 (03-requirements) |
|------|------|------------------------|
| 测试总数 | **727 passed + 1 skipped**（skip 为合法条件跳过：分类器数据不足） | — |
| mypy --strict | **76 个源文件零错误** | NF-Q3 ✓ |
| ruff check | **clean** | NF-Q4 ✓ |
| domain 层覆盖率 | 98%（Wave 2 实测） | NF-Q1 ≥80% ✓ |
| 每端点 3 路径 | 全部 19 端点 success+error+edge | NF-Q2 ✓ |
| 属性测试 | hypothesis：focus_score∈[0,100]、置信度∈[0,1]、全定义域不抛异常等 7 组不变量 | — |

## 2. E2E 实机验收（真实服务器，非 mock）

10 步全通过（`python -m mindflow.main` 生产入口 + watchdog）：

| # | 场景 | 结果 |
|---|------|------|
| 1 | GET /health 无认证 | 200，含 collector/db/migration 状态 |
| 2 | 无 token 访问受保护端点 | 401 problem+json |
| 3 | 恶意 Host `[::1].evil.com` | 403（E2E 发现 500 bug 后修复） |
| 4 | 采集器启动 | 200 中文响应 |
| 5-6 | 真实 Win32 采集 12s → 当前活动 | 抓到真实窗口（WindowsTerminal），duration 5.01s 实测 |
| 7 | 今日专注报告 | 200 |
| 8 | LLM 归因（无 API key → L3 规则引擎） | 200 `degraded:true`，中文合规文案（E2E 发现 upsert 方言 bug 后修复） |
| 9 | 数据导出 JSON | 200 attachment 下载头 |
| 10 | 采集器优雅停止 + WS ping/pong | 200 / pong 帧 |

**E2E 独有发现（单元测试全绿但实机崩溃的 2 个 bug，均修复+回归测试）**：
1. `request.url.path` 惰性解析 Host → 畸形 IPv6 Host 使 403 变 500 且兜底处理器二次崩溃 → 8 处换 `scope["path"]`
2. analysis upsert 用 PostgreSQL 专属 `constraint=` 参数（被 type:ignore 掩盖）→ SQLite 运行时 TypeError → 改 `index_elements`

## 3. 性能实测（NF-P 验收）

| 指标 | 实测 | 要求 | 判定 |
|------|------|------|------|
| API p50 | 1.7ms | — | ✓ |
| API p95 | 16.8ms | <50ms (NF-P2) | ✓ 3x 余量 |
| API p99 | 22.8ms | <100ms (NF-P2) | ✓ 4x 余量 |
| 进程 RSS | 109MB（未加载 ML 模型态） | ≤400MB (NF-P6) | ✓ 3.7x 余量 |
| 采集 tick | 日志实测 <5ms（含 DB 写） | ≤50ms (NF-P3) | ✓ |

## 4. 安全审计（security-reviewer 全库审计）

**结论：CONDITIONALLY PASS → 修复后 PASS**
- CRITICAL 0 / HIGH 0 / MEDIUM 4 / LOW 5
- OWASP Top 10 逐项过审（SQL 全参数化、恒定时间 token 比较、host 校验、危机检测独立门、禁词校验、脱敏摘要）
- Gate 阻塞项已修复：M1 通知 XML 转义、M4 偏好负载大小/深度限制；顺手修复 M3 豁免前缀统一、L1 CSP 头
- 留档接受项：M2 导出非真流式（90 天上限已兜底，真流式列入 backlog）、L2 迁移 URL 明文渲染（SQLite 无密码，PG 接入时处理）、L3 WS query token（access_log 已关）

## 5. AI 残留扫描（reviewer-only 全库）

**结论：通过——干净**。0 裸 TODO / 0 假实现 / 0 test.skip 作弊 / 0 死代码 / 0 僵尸依赖 / 17 个配置字段全部有消费方 / 抽查 docstring 与实现一致。3 项 MEDIUM 治理问题（重复 `_non_idle`、2 个掩盖性 type:ignore）已全部修复。

## 6. 评审循环汇总（全程 9 轮）

| 轮次 | 对象 | 结论 | Findings → 修复 |
|------|------|------|-----------------|
| 1 | 需求 03 | REVISE→ACCEPT | 1C+7M+5m 全修 |
| 2 | 架构 04 | REVISE→ACCEPT | 2C(必崩)+7M+5m 全修 |
| 3 | Wave1 基础设施 | APPROVE | 1P1+3P2 修复 |
| 4 | Wave2 domain | APPROVE（公式逐行审计零偏差） | 4P2 修复+1 误报举证 |
| 5 | Wave3 采集器 | 无阻塞 | 4P1+4P2 修复 |
| 6 | Wave4 API 安全 | REQUEST CHANGES | 2P1(Host绕过/token日志)+3P2 修复 |
| 7 | Wave5 报告 | REQUEST CHANGES | 1P1 竞态修复（纠正评审错误药方）+全部 P2 |
| 8 | Wave6 LLM | REQUEST CHANGES | 1P1 修复+1P1 误报举证+P2/P3 |
| 9 | Wave7 干预 | REQUEST CHANGES | 1P0 时钟+2P1+4P2 修复（非今日时钟证明） |

## 7. 交付物清单

- **代码**: backend-next/ — 76 源文件、~40 测试文件、19 REST 端点 + WS、5 个定时任务、三层 LLM 降级链、ML 训练 CLI（版本化模型）
- **文档**: README（商业级中文）、docs/redesign/01-05 全流程文档、7 ADR
- **打包**: mindflow.spec（PyInstaller，编译校验通过；实际构建列入发布清单）
- **git**: 23+ commits，conventional commits，全程可追溯

## 8. 已知边界与 backlog

1. 导出真流式改造（M2，90 天上限兜底中）
2. macOS/X11 采集器结构完整但未实机验证（无对应硬件；CollectorUnavailableError 降级路径已测）
3. PyInstaller 实际构建 + tufup 发布管线（spec 就绪）
4. hmmlearn 未装时走 Markov 降级（pyproject [ml] extra 可选安装）
5. LLM 在线 eval golden set（离线规则引擎路径已全测）
