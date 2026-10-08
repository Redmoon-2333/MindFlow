# MindFlow 页面使用说明

更新日期：2026-10-01。本说明对应当前仓库；测试使用隔离数据与模拟 API，没有调用付费 LLM 或操作生产数据库。

## 第一次使用

1. 启动后端和前端，完成本地登录。
2. 在「系统设置 → 采集与隐私」启动采集，选择是否启用输入统计和浏览器活动。浏览器功能需要安装 `browser_extension/` 扩展并用配对码配对。
3. 正常工作一段时间，再到「活动日志」确认记录是否产生，到「专注分析」查看会话。新安装没有历史数据时，分析和基线可能暂不可用。
4. 有需要时给会话保存真实反馈。之后在「行为洞察」或「专家面板」查看原因，在「干预中心」获取建议，在「高级功能 → 干预执行」管理具体任务。
5. 定期看「报告中心」。个人模型训练属于进阶功能，先在模型中心查看缺什么数据，再决定是否训练。

## 每个页面如何使用

| 页面 | 解决什么问题 | 普通用户如何使用 |
|---|---|---|
| 本地登录 | 允许网页访问本机服务 | 启动后端后使用本地认证入口；也可运行后端 bootstrap 命令打开带一次性凭证的入口。无需注册云端账户。 |
| 仪表盘 `/` | 快速了解当前状态 | 看今天的专注、活动趋势、最近干预和采集状态；可启动/停止采集、暂停/恢复自主干预，进入详情页。没有模型时预测显示不可用，不能当成 0 分。 |
| 专注分析 `/focus` | 了解什么时候专注或分心 | 选择日期，看会话、时长、评分和趋势。若判断不准，在对应会话选择「专注/分心/混合」、自评分数和任务类型，再保存反馈；反馈是训练信号，并非使用前提。 |
| 活动日志 `/activities` | 确认电脑时间花在哪里 | 选择日期范围，按页查看应用、窗口、时间与活动记录。记录缺失时先检查采集是否开启。 |
| 行为洞察 `/analytics` | 找出反复出现的行为模式 | 选择统计范围，看频繁切换、容易分心的应用、个人画像和基线；点击归因分析获取可能的拖延类型、依据与建议。 |
| 报告中心 `/reports` | 回顾一天或一周 | 切换日报/周报并选择日期，看专注、活动分布和每日明细。完整数据下载在系统设置中。 |
| 干预中心 `/intervention` | 分心或拖延时获得一次提醒 | 选择提醒强度，阅读建议；在最新提醒或历史记录中接受、忽略或推迟，并在响应后评价帮助程度。自主策略可能因冷却等条件跳过一次提醒，页面会提示原因。 |
| 专家面板 `/panel` | 更详细地理解今天为什么难以开始 | 点击「运行专家面板」，先看类型、推荐方法和分析依据；按需展开专家讨论和不同意见。「查看上次结果」读取今天已保存的分析。数据不足时先继续采集；降级时可重试，但不保证一定使用云端模型。 |
| AI 对话 `/chat` | 用自然语言讨论任务和专注问题 | 新建对话，输入问题并发送；也可以点击示例问题填入输入框。左侧会话列表用于回看历史，响应到达前显示等待状态。 |
| 系统设置 `/settings` | 控制收集内容、隐私与运行方式 | 「采集与隐私」管理采集、保留期限、扩展配对、清理与自主模式；「应用分类」把应用归入工作/娱乐等类别，可从未知应用填写规则；「数据与偏好」导出 CSV/JSON，高级 JSON 编辑按需展开；「系统状态」看服务与 AI 用量、估算费用。 |
| 模型中心 `/model-center`，高级功能 | 判断是否能训练个人模型 | 依次看「数据准备」缺口、「个人基线」、「模型训练」和「模型状态」。数据满足条件后可开始训练、查看进度，准备阶段可取消。训练完成不等于模型已激活，应看实际模式和质量门；影子/合成演示模型不能当作已就绪的个人模型。 |
| 干预执行 `/execution`，高级功能 | 把建议落实为任务或减少网站干扰 | 「任务管理」填写标题、描述、优先级、截止时间、预计分钟，添加后可编辑、开始、完成、重新打开或删除；编辑时可清空截止时间和预计分钟。「网站拦截」添加域名和原因，可暂停、恢复或移除；需扩展已配对，扩展每分钟同步名单。 |
| AI 诊断 `/diagnostics`，高级功能 | 分析失败时找到原因 | 先看服务健康、就绪和预测状态，再按页查看运行记录；点击运行编号看节点事件与错误。「分析图拓扑」供排查时按需展开，普通用户日常无需操作。 |
| 页面未找到 | 地址有误时返回应用 | 点击返回首页。 |

## 本次修改与接口检查

前 5 页提供了统一的深色背景、蓝色描边、圆角卡片、留白与字体参考。本次将干预中心、专家面板、AI 对话、系统设置、模型中心、干预执行、AI 诊断这 7 页统一到该视觉体系；设置和执行页面按任务分组，长讨论与技术信息折叠，桌面与手机分别布局。样式作用域限定在这些页面。

| 页面/能力 | 已核对的主要接口（统一前缀 `/api/v1`） | 检查结果 |
|---|---|---|
| 登录与实时消息 | `/auth/*`、`/ws` | 现有会话认证、失效回登录和 WebSocket 路径保留。 |
| 概览、专注、日志、报告 | `/activities/current`、`/focus`、`/focus/trend`、`/focus/{id}/feedback`、`/activities`、`/reports/daily`、`/reports/weekly` | 仪表盘组合多个接口；保留真实数据、空数据与失败状态区分；已有日期、反馈和报告契约回归通过。 |
| 行为洞察 | `/analytics/patterns`、`/analytics/profile`、`/analytics/baseline`、`/analytics/attribution` | 归因响应类型修正：允许实际分析来源、空别名及未提供的可选字段。 |
| 干预 | `/intervention/trigger`、`/intervention/history`、`/intervention/{id}/response`、`/intervention/{id}/feedback` | 保留触发、跳过、响应和反馈流程；修复最新时间展示，并防止旧范围响应覆盖新筛选。 |
| 专家面板 | `/panel/today`、`/panel` | 显示缓存/来源/降级、数据不足与不同意见；保留降级重试参数；类型与推荐方法提供中文名称。 |
| 对话 | `/chat`、`/chat/sessions`、`/chat/{id}/messages` | 保留发送、会话和历史读取，加入示例问题与中文失败提示。 |
| 设置 | `/collector*`、`/telemetry/status`、`/telemetry/preferences`、`/telemetry/browser/pairing-code`、`/telemetry/data`、`/autonomy*`、`/app-classifications*`、`/preferences`、`/export`、`/analytics/usage` | 修复设置卡片错误嵌套、漏显示 AI 费用、分类标签不匹配与未知应用对象类型。 |
| 模型 | `/analytics/training-readiness`、`/analytics/training-jobs*`、`/analytics/baseline`、`/analytics/model-status` | 现有数据就绪、训练、轮询、取消、模型状态流程保留；影子模型和演示状态仍明确展示。 |
| 执行 | `/tasks*`、`/interventions/blocklist*` | 补齐已有 PATCH 接口对应的任务编辑；后端修复显式 null 清空可选字段，并验证不影响省略字段。网站名单仍由扩展执行。 |
| 诊断 | `/health/live`、`/health/ready`、`/telemetry/focus-prediction`、`/ai/runs`、`/ai/runs/{id}`、`/ai/graph` | 补齐分页 limit/offset；详情与列表防止旧请求覆盖；窄屏表格在卡片内部滚动。 |

**现有能力边界**：模型版本展示已接入，但模型回滚没有网页 HTTP 接口，仍用 `uv run python -m mindflow.train --rollback <version-tag>`；高级偏好仍是 JSON 编辑器；浏览器扩展配对、真实网站阻断、系统采集和真实 LLM 效果需要在实际环境验收。本次未进行生产环境或付费 LLM 联调。不能将模拟浏览器测试视为这些实际能力已验收。

## 启动方式

后端：Python 3.11+、uv、FastAPI、SQLAlchemy/SQLite、Alembic、LangGraph；LLM 为 DeepSeek → Ollama → 规则引擎，缺云端凭证会按既有逻辑降级。ML 使用 scikit-learn/XGBoost。

```powershell
cd D:\Mindflow\backend-next
uv sync --extra dev --extra ml
uv run python -m mindflow.main
# 可另开终端生成本地登录入口
uv run python -m mindflow.bootstrap
```

前端：React 19、TypeScript、Vite、React Router、MUI/日期组件、原生 CSS；接口请求在 `src/api.ts`，类型漂移检查对照后端 OpenAPI。

```powershell
cd D:\Mindflow\frontend
npm ci
npm run dev
```

默认网页地址 `http://127.0.0.1:4173`，后端 `http://127.0.0.1:8765`；Vite 代理 `/api` 和 WebSocket。自定义地址用 `MINDFLOW_FRONTEND_PORT`、`MINDFLOW_BACKEND_URL`，自定义数据目录用 `MINDFLOW_DATA_DIR`。

## 本轮验证

- 前端生产构建成功；lint 无错误（17 条已有警告，主要来自图标库和旧测试）。
- 前端 11 组契约/状态测试通过；含 API schema 漂移检查。
- Playwright：90 项通过（57 秒），使用模拟 API 与 WebSocket、隔离 Vite 端口，覆盖 1440/1024/390/320 像素布局、任务编辑、设置分组、分页、面板及已有核心回归。测试配置：`frontend/playwright.ui-review.config.ts`。有数据的模型质量门也完成桌面与手机截图复查。
- 后端完整测试：3153 passed、4 skipped；Ruff 无问题、Mypy strict 185 个文件通过。修复测试夹具中维护服务的用户目录回退，测试不会触及真实用户数据目录。
- 截图使用模拟数据，位于 `frontend/screenshots/ui-review/`；代码未提交或推送。
