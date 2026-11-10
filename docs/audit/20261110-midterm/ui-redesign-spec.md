# MindFlow UI 重设计规格 · "Signal" v2（2026-11-10，主代理定稿）

> **分工**：所有设计决策（颜色、字号、尺寸、布局、文案、交互、状态映射）都由主代理在本文件中定死。实现者**只实施，不做任何设计判断**。遇到本文件未覆盖的情况：**停下来，在报告中列出问题，不要自行决定**。
> v2 相对 v1：消除所有"实现者给出值/实现者推理"条款，补全 4 套主题全部色值（已用 WCAG 公式计算对比度）、组件 API 与 CSS 数值、仪表盘逐像素布局、状态阈值、文案、字段映射。

## 0. 为什么重做（用户反馈 + 自查）

用户原话：卡片没对齐、忽大忽小；卡片/组件"AI 味太重，一堆框堆在一起"；行为洞察三个小标题僵硬；仪表盘"一大堆莫名其妙的卡片堆积，人机感重又找不到重点"。要求：抓眼球 + 逻辑清晰，一眼抓住重点，不要信息平铺爆炸。

| # | 页面 | 问题 |
|---|---|---|
| P1 | 仪表盘 | 14 个同权重盒子平铺，没有视觉主角；"我此刻状态/下一步"排在第 5 行 |
| P2 | 仪表盘 | 4 张系统状态卡 + "服务状态"卡占据首屏首行——正常状态不该占注意力 |
| P3 | 仪表盘 | "自主控制"卡下方空洞、"最近干预"卡右侧留白、同行卡片底部参差 |
| P4 | 全局 | 卡片标题一律粗体（微软雅黑 700 合成粗体发糊），标题比内容抢眼 |
| P5 | 全局 | KPI 卡右上角彩色图标方块是模板化装饰 |
| P6 | 全局 | 盒中盒 |
| P7 | 干预中心/专家面板 | 页内 H1 与顶栏重复；左右卡片高度不一；历史标题与筛选不在同一基线 |
| P8 | 干预中心 | 历史条目"为什么是这条提醒"按钮时有时无，列不对齐 |
| P9 | 行为洞察 | "—— 模式分析 ——"居中装饰标题僵硬；空数据渲染固定高度空盒；列表内部滚动条；英文键名；个人画像数字对齐混杂、折行 |
| P10 | 行为洞察 | "个人模型与云端 AI"是系统状态，混在用户洞察里 |
| P11 | 专注分析/报告中心 | 同 P4/P5；超大胶囊按钮；粗体长段落；图表多色相无语义 |
| P12 | 全局 (WIG) | `transition: all`；`outline: none` 无 focus-visible 替代；白色主题 `color-scheme` 不对；`theme-color` 不随主题 |
| P13 | 移动端 375 | 筛选胶囊换行；堆叠后无重点 |

## 1. 设计主张

**MindFlow 是一台读"注意力信号"的仪器。** 用户打开它只想按顺序回答：① 我此刻状态怎么样？② 下一步该做什么？③ 今天/这几天整体如何？
系统健康、模型来源、采集器、数据库都是**背景**：正常时收进顶栏一个胶囊，异常时才升级。

| 来源 | 取 | 舍 |
|---|---|---|
| Astryx Gothic | 深蓝灰表面、靠明度差分层、无投影、静谧 | 做旧/哥特展示字体 |
| Android Holo | 唯一发光青色强调（只给"实时/此刻/主操作"）、细窄标签字 | 2px 直角（改 8px） |
| Real-Time Monitoring | 状态优先于数字、正常隐形→异常升级、动画只作信号 | 红橙绿三色告警堆叠 |
| SaaS Enterprise Analytics | 渐进披露、字重/字号建层级、2–3 色相 + 中性色 | 玻璃拟态 |

**签名元素（全站唯一的大胆之处）**：仪表盘"此刻"读数 —— Bahnschrift Light 64px 专注概率 + 0–100 信号轨上带发光的游标，轨道中段标出"不确定区间"。全站只有这里和实时点用发光。

## 2. Token（逐值定死）

### 2.1 色值（4 套主题；对比度为对 surface 的 WCAG 比值，主代理已计算）

| Token | navy | black | white | burgundy | 用途 |
|---|---|---|---|---|---|
| `--bg` | `#0A0E14` | `#000000` | `#F5F7FA` | `#140A0F` | 页面底、侧栏、顶栏 |
| `--surface` | `#121A24` | `#0C0F10` | `#FFFFFF` | `#1E1218` | 唯一一级容器 |
| `--surface-2` | `#18222E` | `#171B1C` | `#EEF2F6` | `#27171F` | 悬停/选中/输入框 |
| `--line` | `#1F2A36` | `#22282A` | `#DDE3EA` | `#33212A` | 发丝线、容器边 |
| `--line-strong` | `#2B3846` | `#30383A` | `#C5CFDA` | `#452E39` | 控件边框、信号轨底 |
| `--text` | `#D7DEE6` (12.9) | `#E8F7FA` (17.5) | `#0F1A26` (17.6) | `#EEDFE5` (14.1) | 正文、数字 |
| `--text-dim` | `#8E9BAA` (6.2) | `#8A9A9D` (6.6) | `#56657A` (5.9) | `#A9919B` (6.2) | 标签、说明 |
| `--signal` | `#33B5E5` (7.4) | `#33B5E5` (8.1) | `#066A94` (6.0) | `#33B5E5` (7.7) | **专注/实时/主操作** |
| `--signal-hover` | `#5CC6EE` | `#5CC6EE` | `#055A7E` | `#5CC6EE` | 主按钮悬停 |
| `--on-signal` | `#0A0E14` | `#0A0E14` | `#FFFFFF` | `#0A0E14` | 主按钮文字（对比 8.2 / 6.0） |
| `--drift` | `#E0A458` (8.0) | `#E8A951` (9.4) | `#94580A` (5.7) | `#E0A458` (8.3) | **分心/需注意** |
| `--alarm` | `#FF5D5D` (5.8) | `#FF5D5D` (6.4) | `#C2312B` (5.6) | `#FF6B6B` (6.6) | 仅严重异常/破坏性操作 |
| `--ok` | `#5FBF8F` | `#5FBF8F` | `#1A7A50` | `#6CC28F` | 只用于状态胶囊"运行正常"的点 |
| `--steel` | `#5A7A9A` | `#5E7C86` | `#6A7F95` | `#8A6A78` | 次要数据系列（往日柱、对照） |

派生（在 `applyTheme` 中用 `color-mix` 生成，不另给色值）：
- `--signal-soft` = `color-mix(in srgb, var(--signal) 14%, transparent)`；`--drift-soft`、`--alarm-soft` 同理 14%。
- `--signal-glow`：深色三套 = `0 0 12px color-mix(in srgb, var(--signal) 35%, transparent)`；white = `0 0 0 3px color-mix(in srgb, var(--signal) 16%, transparent)`。
- `--focus-ring` = `0 0 0 2px var(--signal)`（用 outline 实现，见 §3.11）。

拖延五类型色（`--type-*`，固定槽位顺序）：

| 类型 | navy/black/burgundy | white |
|---|---|---|
| task_aversion | `#C9A15A` | `#8A6414` |
| impulsivity | `#D9776F` | `#A8463E` |
| decisional | `#6F9BD1` | `#2F64A3` |
| perfectionism | `#A68BD1` | `#6B4FA0` |
| emotional_regulation | `#5FB3A3` | `#2A7A6C` |

（对 surface 对比度 navy 5.7–7.3，white 5.1–6.5，均 ≥4.5。）

### 2.2 旧变量 → 新值映射（`applyTheme` 照此写，旧页面自动换色，不必逐页改）

| 旧变量 | 新值 |
|---|---|
| `--color-bg` `--mf-sidebar-bg` `--mf-topbar-bg` `--mf-bg-radial` `--mf-login-radial` | `bg` |
| `--mf-panel` `--color-bg-elevated` `--chart-tooltip-bg` | `surface` |
| `--color-bg-inset` `--color-primary-light` | `surface-2` |
| `--mf-panel-border` `--mf-hairline` `--color-border` `--mf-accent-line` `--chart-grid` `--chart-track` | `line` |
| `--chart-tooltip-border` | `line-strong` |
| `--mf-text` `--color-text-primary` | `text` |
| `--mf-text-dim` `--mf-text-icon` `--color-text-secondary` `--color-text-tertiary` `--chart-axis` | `text-dim` |
| `--mf-brand` `--color-primary` `--color-info` `--info` `--mf-cyan` `--chart-pos` `--chart-1` | `signal` |
| `--color-primary-hover` | `signal-hover` |
| `--color-on-primary` | `on-signal` |
| `--warn` `--color-warning` `--mf-yellow` `--mf-orange` `--chart-neg` `--chart-2` | `drift` |
| `--bad` `--color-danger` `--mf-red` | `alarm` |
| `--good` `--color-success` `--mf-green` | `ok` |
| `--chart-3` | `steel` |
| `--chart-4` / `--chart-5` / `--chart-6` | `type-perfectionism` / `type-impulsivity` / `text-dim` |
| `--*-soft`（good/warn/bad/info/neutral/chart-pos/chart-neg） | 对应新色 14%（chart-*-soft 24%）color-mix |
| `--shadow-1` `--shadow-2` `--card-highlight` | `none` |
| `--shadow-focus` | `0 0 0 2px var(--signal)` |

e2e 对比度探针读取的 `--mf-text` `--mf-text-dim` `--color-bg` `--mf-panel` `--color-bg-inset` 仍为纯 hex（按上表直接写 hex，不写 var()）。

### 2.3 主题外设

- `document.documentElement.style.colorScheme`：white = `light`，其余 `dark`。
- `applyTheme` 同步 `<meta name="theme-color">` 的 `content` = 该主题 `bg`（不存在则创建并插入 `<head>`）。

### 2.4 字体

- `--font-ui: "Inter", "Microsoft YaHei UI", "PingFang SC", system-ui, sans-serif`
- `--font-num: "Bahnschrift", "DIN Alternate", "Roboto Condensed", "Inter", sans-serif`；凡用 `--font-num` 的元素同时 `font-variant-numeric: tabular-nums`。
- 字重只允许 300（仅 Hero 读数）、400、500、600。**全局搜索 `font-weight: 700|800|900|bold|bolder` 一律改为 600**（标题类）或 500（按钮/标签类）。
- 字号 token 重定义：`--fs-xs 12` `--fs-sm 13` `--fs-base 14` `--fs-md 16` `--fs-lg 20` `--fs-xl 28` `--fs-2xl 44` `--fs-3xl 64`。正文 14/1.5。
- `.label-caps`：12px / 500 / `letter-spacing: .04em` / `text-dim` / line-height 16px。
- 所有 h1–h3 加 `text-wrap: balance`。

### 2.5 形状、间距、动效

- 圆角：`--radius-sm 6px`（控件）`--radius-md 8px` `--radius-lg 8px`（容器，旧 16 一律收敛到 8）`--radius-pill 999px`。
- 间距 token 不变（4/8/12/16/20/24/32/48）。
- 无投影；层级 = 明度差 + 1px `line`。
- 过渡只允许：`color, background-color, border-color` 150ms ease-out；`transform, opacity` 200ms ease-out；信号轨游标 `left` 250ms ease-out。
- `@media (prefers-reduced-motion: reduce)`：所有 `transition-duration` / `animation-duration` 设为 `0.01ms`，实时点脉冲 `animation: none`。

## 3. 组件（API 与 CSS 定死）

新组件放 `src/components/signal/`，样式集中在 `src/components/signal/signal.css`。

### 3.1 `Surface`
`<Surface as?="section"|"div" pad?="md"|"lg" className? aria-labelledby?>`
CSS：`background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 24px`（`pad="lg"` → `28px 32px`）；无阴影；`min-width: 0`。**Surface 内禁止再放 Surface 或任何带边框背景的盒子**；内部分组只用 `border-top: 1px solid var(--line)` + 间距。

### 3.2 `SectionHeader`
`<SectionHeader id? title summary? actions?>`
```
.sh            display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:16px; min-height:32px
.sh-title      16px / 600 / line-height 24px / color text; h2 元素
.sh-title::after  content:""; display:block; width:32px; height:2px; margin-top:6px; border-radius:1px;
                  background:linear-gradient(90deg, var(--signal), transparent)
.sh-summary    13px / 400 / text-dim / margin-top:6px
.sh-actions    display:flex; gap:8px; align-items:center; flex-shrink:0
```
375px：`.sh` 改 `flex-wrap: wrap`，actions 占满一行。

### 3.3 `MetricStrip`
`<MetricStrip label? items={[{key,label,value,unit?,delta?:{text,tone:"up"|"down"|"flat"|"none"}}]}>`
- 自身就是一个 Surface（pad 20px 24px），内部 `display:grid`：有 `label` 时 `grid-template-columns: 64px repeat(n, minmax(0,1fr))`，否则 `repeat(n, minmax(0,1fr))`。
- 首列 label：`.label-caps`，垂直居中。
- 每格：`padding: 0 24px; border-left: 1px solid var(--line)`（第一格数据格也有左线，与 label 列分隔；无 label 时第一格无左线、左 padding 0）。
  - 标签 13px/400 text-dim，margin-bottom 8px
  - 数值 `--font-num` 28px/400/line-height 32px/text；单位 14px text-dim，margin-left 4px
  - 变化 12px，margin-top 6px：`up` = signal，`down` = drift，`flat` = text-dim，`none` = text-dim 显示"较昨日 —"。文案格式 `较昨日 ↑12%` / `较昨日 ↓8%` / `较昨日 持平`。
- ≤600px：label 列变成首行（`grid-column: 1 / -1; margin-bottom: 12px`），数据格 2×2：第 2、4 格左线，第 3、4 格 `border-top: 1px solid var(--line); padding-top:16px; margin-top:16px`；第 1、3 格无左线、左 padding 0。

### 3.4 `Segmented`（阶段 1 只建组件，阶段 2 使用）
`<Segmented ariaLabel options={[{value,label}]} value onChange size?="sm">`
- 标记：`<div role="group" aria-label>` + 每项 `<button type="button" aria-pressed>`。
- 容器：`display:inline-flex; gap:2px; padding:2px; background: var(--bg); border:1px solid var(--line); border-radius:6px; overflow-x:auto; white-space:nowrap; max-width:100%`。
- 项：`height:28px; padding:0 12px; border-radius:4px; 13px/500; color:text-dim; background:transparent`；hover `color:text`；选中 `background:surface-2; color:signal`。

### 3.5 `StatusChip`（顶栏右侧，全站）
**数据**：自己调用 `getHealth`、`getCollectorStatus`、`getAiProviderStatus`、`getModelStatus`（`Promise.allSettled`），挂载时 + 每 60 秒 + 采集器切换后刷新。

**状态优先级**（取第一个命中）：

| 条件 | 点颜色 | 胶囊文字 |
|---|---|---|
| 首次请求未完成 | text-dim | `检查中…` |
| health 请求失败 | alarm | `服务无响应` |
| `health.database.status !== "ok"` | alarm | `数据库异常` |
| `health.status` 不是 `ok`/`healthy` | alarm | `服务异常` |
| collector 已停止 | drift | `采集已停止` |
| 其余 | ok | `运行正常` |

云端 AI 未配置**不升级**（规则引擎是合法降级），只在浮层里显示。

**胶囊**：`<button aria-haspopup="dialog" aria-expanded aria-controls="status-popover">`；height 28px；padding 0 10px；radius pill；`border:1px solid var(--line-strong)`；背景透明；13px/500/text；点 8px 圆，margin-right 8px；末尾 `▾` 10px text-dim。hover 背景 surface-2。

**浮层** `#status-popover`：`role="dialog" aria-label="系统状态"`；绝对定位于胶囊下方右对齐，`top: calc(100% + 8px)`；width 320px（≤600px 为 `calc(100vw - 32px)`）；`background:surface; border:1px solid var(--line-strong); border-radius:8px; padding:16px`；z-index 高于内容。Esc、点外部、再次点胶囊关闭；关闭后焦点回胶囊。
内容为两列定义列表（标签 13px text-dim 宽 72px，值 13px text），每行高 28px，顺序：

| 标签 | 值 |
|---|---|
| 服务 | `正常` / health.status 原文 / `无响应` |
| 数据库 | `已连接` / `异常` |
| 采集器 | `运行中` / `已停止`，同行右侧次按钮（小号 height 28）`停止采集` / `启动采集`，切换中显示 `切换中…` 且 disabled |
| 云端 AI | `已配置 · {model}` / `未配置（使用本地规则）` |
| 预测模型 | `demo_only` → `合成演示模型 · 影子模式`；`loaded` → `个人模型已加载`；否则 `规则引擎模式` |
| 版本 | `health.version` 或 `—` |

采集器切换调用沿用 Dashboard 现有 `startCollector/stopCollector` 逻辑（移过来，不改 API）；失败用现有 toast 报错。

### 3.6 `Disclosure`
`<Disclosure summary defaultOpen?={false}>children</Disclosure>`
- 触发器 `<button type="button" aria-expanded aria-controls>`：13px/500/signal，前置 `▸` 字符 10px，展开时 `transform: rotate(90deg)`（200ms）。无边框无背景，padding 0，height 24px。
- 内容区 `id` 对应 aria-controls：展开时渲染，`margin-top:16px; padding-top:16px; border-top:1px solid var(--line)`。

### 3.7 `LiveDot`
`<LiveDot status={RealtimeStatus}>`：8px 圆 + 后接 12px 文字（gap 6px）。
- `connected`：signal 填充 + `box-shadow: var(--signal-glow)` + 脉冲动画（伪元素 8px 圆环 signal 边框，`scale 1→2.4, opacity .6→0`，1.6s ease-out infinite）；文字 `实时`（text-dim）。
- `connecting`：text-dim 点，文字 `连接中`。
- 其余：text-dim 点，文字 `已断开`。

### 3.8 按钮与表单（全局 class，替换现有按钮样式）
```
.btn            height:32px; padding:0 14px; border-radius:6px; 13px/500; display:inline-flex; align-items:center; gap:6px; border:1px solid transparent; cursor:pointer
.btn-primary    background:var(--signal); color:var(--on-signal)   hover → var(--signal-hover)
.btn-secondary  background:transparent; color:var(--text); border-color:var(--line-strong)   hover → background:var(--surface-2)
.btn-ghost      background:transparent; color:var(--signal); padding:0 8px   hover → background:var(--signal-soft)
.btn-danger     background:transparent; color:var(--alarm); border-color:var(--line-strong)
.btn-sm         height:28px; padding:0 10px; 12px
:disabled       opacity:.45; cursor:not-allowed
input, select, textarea  height:32px(textarea 除外); background-color:var(--surface-2); color:var(--text); border:1px solid var(--line-strong); border-radius:6px; padding:0 10px; 14px
select option   background-color:var(--surface); color:var(--text)
```
每屏最多一个 `.btn-primary`。
- 只有 `.btn`、没有任何变体类的按钮（阶段 1 时其他页面有 23 处）：外观等同 `.btn-secondary`，作为兜底；阶段 2 改各页面时逐个换成显式变体。
- `.btn-danger:hover` 背景色为 `var(--alarm-soft)`。
- `Segmented` 不接受 `size` prop（全站只有一种尺寸）。

### 3.9 列表行（阶段 2 使用，阶段 1 先建 class）
`.list-row`：`display:grid; align-items:center; min-height:48px; border-top:1px solid var(--line); column-gap:16px`，首行无上线。列由各页面在阶段 2 定义。

### 3.10 提示条（替换仪表盘 error-box）
`.notice`：`display:flex; align-items:center; gap:12px; padding:10px 16px; background:var(--surface); border:1px solid var(--line); border-left:2px solid var(--drift); border-radius:6px; 13px/text`；右侧关闭按钮 `aria-label="关闭提示"`，内容 `×`。

### 3.11 WIG 全局修复（全仓 `src/**/*.css` 与 `*.tsx` 内联样式）
1. `transition: all` / `transition: all ...` → 按 §2.5 改成具体属性。
2. 每个 `outline: none` / `outline: 0`：同选择器补 `:focus-visible { outline: 2px solid var(--signal); outline-offset: 2px; }`。全局兜底：`:where(a, button, input, select, textarea, [tabindex]):focus-visible { outline: 2px solid var(--signal); outline-offset: 2px; }`。
3. 只有图标的按钮必须有 `aria-label`（实现者 grep 出清单在报告里列出改了哪些）。
4. `Toaster` 容器加 `aria-live="polite"`（已有则保持）。
5. 省略号统一 `…`（替换 `...` 文案，如 `处理中...` → `处理中…`、`认证中...` → `认证中…`）。

## 4. 页面

### 4.0 外壳（App.tsx + shell.css）

**侧栏** `.mf-navbar`：
- width 232px；`background: var(--bg)`；`border-right: 1px solid var(--line)`；padding 20px 12px。
- Logo 行：logo 图 28×28 + `MindFlow` 16px/600/text，gap 10px，height 40px，padding 0 8px，margin-bottom 24px。**删除** `.mf-logo-line` 元素。
- 导航项 `.mf-nav-item`：height 40px；padding 0 12px；gap 12px；radius 6px；14px/400/text-dim；图标 18px text-dim。hover：color text，背景 surface-2。active：color text、500、背景 surface-2、`box-shadow: inset 2px 0 0 var(--signal)`、图标 signal。
- "高级功能"分组按钮：`.label-caps` 样式，height 32px，padding 0 12px，margin-top 16px。
- 底部"本地会话"：13px text-dim，height 40px，padding 0 12px，hover 背景 surface-2。
- 折叠态/移动抽屉行为不变。

**顶栏** `.mf-header`：height 56px；`background: var(--bg)`；`border-bottom: 1px solid var(--line)`；padding 0 32px（≤600px：0 16px）；`display:flex; align-items:center; gap:12px`。
- 左：折叠按钮 32×32（radius 6，hover surface-2，`aria-label` 保持现有）→ 页面标题 16px/600/text → 日期 13px text-dim，margin-left 4px，格式 `Intl.DateTimeFormat("zh-CN", { month: "long", day: "numeric", weekday: "short" })`（例：`11月10日周二`），每分钟刷新；≤600px 隐藏日期。
- 右（`margin-left:auto`）：`StatusChip`。

**主区** `.mf-main`：padding 32px 32px 48px（≤600px：16px 16px 32px）；内部内容 `max-width: 1280px; margin: 0 auto`。

### 4.1 仪表盘（阶段 1 唯一改信息架构的页面）

**线框（1440 宽）**
```
┌ 仪表盘 11月10日周二                                    [● 运行正常 ▾] ┐
│                                                                        │
│ ┌──────────────────────────── 2fr ─────────┐ ┌──────── 1fr ─────────┐ │
│ │ 此刻  ● 实时                 更新于 2分钟前│ │ 下一步                │ │
│ │                                           │ │                       │ │
│ │ 75%   专注中                               │ │ 先整理一个图表，再     │ │
│ │ (64)  正在使用 Code.exe — mindflow-app…   │ │ 继续下一步。(18/500)   │ │
│ │                                           │ │ 任务拆解 · 12:00 · 未回应│ │
│ │ ━━━━━━━━━░░░░░░░░░━━━━━━━━●━━━━━━━━━━━━  │ │ [去干预中心]           │ │
│ │ 分心        不确定区间            专注     │ │ 最近 7 天共 3 条提醒   │ │
│ │                                           │ │ ───────────────────── │ │
│ │ ● 支持专注  窗口切换次数 · 生产类应用占比  │ │ 自动提醒        开启中 │ │
│ │ ● 拉向分心  空闲比例                       │ │ [暂停30分钟][暂停1小时]│ │
│ │ ▸ 查看完整依据                             │ │ [今天不再提醒] 自定义  │ │
│ │ ───────────────────────────────────────── │ │                       │ │
│ │ 合成演示模型 · 影子模式 · 未通过个人模型质量门│ │                       │ │
│ └───────────────────────────────────────────┘ └───────────────────────┘ │
│ ┌ 今天 │ 专注时长        │ 专注次数   │ 平均专注分    │ 分心比率      ┐ │
│ │      │ 183 min         │ 2 次       │ 83.0 /100     │ 0.0 %         │ │
│ │      │ 较昨日 ↓10%     │ 较昨日 持平│ 较昨日 ↑4%    │ 较昨日 —      │ │
│ └──────────────────────────────────────────────────────────────────────┘ │
│ ┌ 最近 7 天  ▔▔                     ■ 今天  ■ 往日  ┄ 7 日均值 166 分钟 ┐       │
│ │ 每日专注时长（分钟）                                          │       │
│ │ ▇  ▇  ▇  ▇  ▇  ▇  █                                           │       │
│ └──────────────────────────────────────────────────────────────┘       │
└────────────────────────────────────────────────────────────────────────┘
```

**网格**：`.dash { display:grid; grid-template-columns: minmax(0,2fr) minmax(0,1fr); gap:16px; align-items:stretch }`；今天条与 7 天图 `grid-column: 1 / -1`。≤1024px 单列，顺序：此刻 → 下一步 → 今天 → 最近 7 天。
**删除**：页面浮动日期头 `formatFloatHeader`（日期已在顶栏）、`.d-state` 四张状态卡、顶部采集器按钮、`.d-extra` 两张卡、`.d-activity` 卡、`.d-control` 卡、`.d-inference` 卡、所有装饰 iconfont。

#### 4.1.1 此刻（Hero，`<Surface pad="lg" aria-labelledby="now-title">`，`display:flex; flex-direction:column; min-height:340px`）

**状态阈值**（主代理决定，依据：后端不确定性采样带 |p−0.5| ≤ 0.15，见 `services/feedback_sampling` 与 FeedbackRequestStore）：在 `prediction-state.ts` 新增并导出

```ts
export type FocusBand = "focus" | "uncertain" | "drift";
export const FOCUS_BAND_UPPER = 0.65;
export const FOCUS_BAND_LOWER = 0.35;
export function focusBand(p: number): FocusBand  // p ≥ 0.65 → focus；p ≤ 0.35 → drift；其余 uncertain
export const FOCUS_BAND_LABELS = { focus: "专注中", uncertain: "状态摇摆", drift: "分心中" } as const;
```
颜色：focus = signal；uncertain = text；drift = drift。

**第 1 行（标签行）** `display:flex; align-items:center; gap:12px`：`<h2 id="now-title" class="label-caps">此刻</h2>` + `<LiveDot status={realtimeStatus}>`；右侧（margin-left:auto）12px text-dim `更新于 {相对时间}`，用 `Intl.RelativeTimeFormat("zh-CN", { numeric: "auto" })`，以 currentActivity.timestamp 计算（<60s 显示 `刚刚更新`），每 30 秒刷新；无活动则不显示。

**第 2 行（读数行）** margin-top 24px，`display:flex; align-items:flex-end; gap:24px`：
- 读数：`<div class="hero-num" aria-label="专注概率 75.1%">`，内容 `{Math.round(p*100)}` + `<span class="hero-unit">%</span>`。`.hero-num`：`--font-num` 64px / 300 / line-height 1 / 状态色；`.hero-unit`：28px / 300 / text-dim / margin-left 2px。
- 右侧块（`min-width:0; padding-bottom:6px`）：
  - 状态词 `FOCUS_BAND_LABELS[band]`：28px/600/line-height 36px/状态色。
  - 活动行 14px text-dim margin-top 4px，单行 `overflow:hidden; text-overflow:ellipsis; white-space:nowrap`，`title` = 完整文本：
    - 有活动且非空闲：`正在使用 {app_name} — {window_title}`（window_title 空则只到 app_name）
    - 空闲：`当前空闲 · 最后使用 {app_name}`
    - 无活动：`当前没有活动记录`

**第 3 行（信号轨）** margin-top 28px；`role="img"` + `aria-label="专注概率 75.1%，处于专注区间"`：
- 轨道：height 4px；radius 2px；background line-strong；position relative。
- 不确定区间：轨道上 left 35% → 65% 的覆盖段，`background: color-mix(in srgb, var(--text) 18%, transparent)`。
- 游标：12px 圆，`left: calc(p*100% - 6px)`，top -4px，背景状态色；focus 时加 `box-shadow: var(--signal-glow)`，其他状态无发光；`transition: left 250ms ease-out`。
- 刻度文字行 margin-top 8px，12px text-dim：`分心`（左对齐）、`不确定区间`（absolute，left 50%，translateX(-50%)）、`专注`（右对齐）。

**第 4 块（主要因为）** margin-top 24px：取 `explanation.contributions` 按 |contribution| 降序前 3：
- 正值组（contribution > 0）一行：6px signal 圆点 + `支持专注`（13px text-dim，宽 64px）+ 各项 label 以 ` · ` 连接（14px text）。
- 负值组一行（margin-top 8px）：6px drift 圆点 + `拉向分心` + 各项 label。
- 某组为空则不渲染该行。contributions 为空：只渲染一行 13px text-dim `当前模型不提供逐项依据`，且不渲染 Disclosure。
- 其下 margin-top 12px：`<Disclosure summary="查看完整依据">`，展开内容按序：
  1. 12px text-dim 一行：`方法：{METHOD_LABELS[method]} · 按影响大小排序，最多 8 项`
  2. `ContributionBar`（现有组件，props 不变；**去掉其外层边框/背景**，仅改 CSS）
  3. `explanation.note`（13px text-dim，若有）
  4. margin-top 16px，`.label-caps` `当前窗口`，下面定义列表（标签 13px text-dim 宽 72px / 值 13px text，行高 24px）：`应用` app_name｜`窗口` window_title｜`进程` process_name｜`状态` `活跃`/`空闲`｜`最后更新` `toLocaleString("zh-CN",{hour12:false})` + 空格 + 现有 `utcOffsetLabel()`。无活动则整组不渲染。

**第 5 块（来源行）** `margin-top:auto; padding-top:16px; border-top:1px solid var(--line)`（与上方内容最小间距 24px），12px text-dim：
- `modelStatus.demo_only` → `合成数据演示模型 · 影子模式 · 未通过个人模型质量门`
- `modelStatus.loaded` → `个人模型 · {userTerm(predictionView.mode)}`
- 否则 → `规则引擎模式`

**未就绪（`predictionView.ready === false` 或无 prediction）**：读数显示 `—`（text-dim，64px/300）；状态词 `暂无判断`（text）；活动行照常；信号轨与"主要因为"块不渲染，改为一行 14px text-dim：`{statusLabel}：{reason}`（无 prediction 时为 `需积累数据后由模型中心生成`）+ 同行 `.btn-ghost .btn-sm` 链接 `前往模型中心`（to `/model-center`）。

#### 4.1.2 下一步（`<Surface aria-labelledby="next-title">`，`display:flex; flex-direction:column`，与 Hero 同行等高）

- `<h2 id="next-title" class="label-caps">下一步</h2>`，margin-bottom 16px。
- **最新一条干预** = `interventions` 中 `triggered_at || created_at` 最大者：
  - 消息：`item.message || item.title || getInterventionTypeLabel(type)`，18px/500/text/line-height 28px，最多 4 行（`-webkit-line-clamp: 4`），`title` 给全文。
  - 元信息行 margin-top 8px，12px text-dim：`{类型中文} · {formatClock(time)} · {响应}`；响应用现有 `RESPONSE_LABELS`，未回应显示 `未回应`；`已接受` 这一段用 signal 色。
  - 操作 margin-top 16px：`.btn .btn-secondary` `去干预中心`（navigate `/intervention`）。
  - margin-top 12px，12px text-dim：`最近 7 天共 {interventions.length} 条提醒`。
- 无干预：16px text-dim `目前不需要提醒，保持现在的节奏。`，下一行 margin-top 12px `.btn-ghost .btn-sm` `去干预中心`。
- **自动提醒**（`margin-top:auto; padding-top:20px; border-top:1px solid var(--line)`，与上方最小间距 24px）：
  - 标题行：`自动提醒` 14px/500/text；右侧状态 13px：运行中 `开启中`（signal）/ 暂停中 `已暂停至 {HH:mm}`（drift；跨日则 `{M月D日 HH:mm}`）。
  - 运行中：margin-top 12px，`display:flex; flex-wrap:wrap; gap:8px` 的 `.btn-secondary .btn-sm`：`暂停 30 分钟`、`暂停 1 小时`、`今天不再提醒`（时长 = 到本地今天 23:59 的分钟数，<30 分钟时 disabled 并 `title="今天剩余不足 30 分钟"`）、`.btn-ghost .btn-sm` `自定义`。
    - 点 `自定义` 原地展开（替换该按钮行下方）：现有小时/分钟两个输入（保留 aria-label `暂停小时`/`暂停分钟`）+ `.btn-secondary .btn-sm` `暂停` + `.btn-ghost .btn-sm` `取消`。校验沿用现有"至少 30 分钟"。
    - 所有暂停都调用现有 `pauseAutonomy(hours)`。
  - 暂停中：margin-top 12px，`.btn-secondary .btn-sm` `恢复提醒`（调用现有恢复函数）+ 12px text-dim `暂停期间不会推送干预`。
  - 处理中按钮文字 `处理中…`，全部 disabled。

#### 4.1.3 今天（`<MetricStrip label="今天">`，全宽）

| key | 标签 | 值 | 单位 | 变化 |
|---|---|---|---|---|
| focus | `专注时长` | 现有逻辑 | `min` | `dayOverDay(focus_min)` |
| sessions | `专注次数` | 现有逻辑 | `次` | `dayOverDay(session_count)` |
| score | `平均专注分` | 现有逻辑（1 位小数） | `/100` | `dayOverDay(avg_score)` |
| distraction | `分心比率` | 现有逻辑（1 位小数） | `%` | 恒为 `none` |

变化映射：`null` → tone `none`；`|d| < 0.5` → `flat`（`较昨日 持平`）；`d > 0` → `up` `较昨日 ↑{round(d)}%`；`d < 0` → `down` `较昨日 ↓{round(|d|)}%`。**分心比率不显示涨跌色**（none）。
值语义沿用现有 trendStatus：loading → 值显示 `…`（text-dim）；error/未知 → `—`；ready 且无记录 → `0`。trendStatus 为 error 时，在 strip 容器**内部**最后一行（跨所有列，margin-top 12px）显示 12px text-dim `趋势数据加载失败，稍后自动重试`（仅当现有代码已有重试；没有则文案为 `趋势数据加载失败`）。

#### 4.1.4 最近 7 天（`<Surface>`，全宽）

- `<SectionHeader title="最近 7 天" summary="每日专注时长（分钟）" actions={图例}>`；图例 12px text-dim，三项 gap 16px：`■ 今天`（10×10 方块 signal，radius 2）、`■ 往日`（steel）、`┄ 7 日均值 {round} 分钟`（16px 宽 1px 虚线 text-dim；F10：均值数字只在图例出现，不在绘图区内标注）。无今天槽位时不显示"今天"图例（沿用 hasTodaySlot）。
- `BarChart` height 220；`colorFor`：today → `var(--signal)`，其余 → `var(--steel)`；tooltip detail 保留 `分心`、`专注次数`，新增 `平均分`（`score` 非空时，1 位小数）。
- **BarChart 扩展**（唯一允许改的图表 API）：新增可选 prop `referenceLine?: { value: number; label?: string }`，绘制横跨绘图区的 1px 虚线（`stroke: var(--text-dim)`，`stroke-dasharray: 4 4`），`label` 非空时右端上方 12px text-dim 文字。仪表盘传 `{ value: 7 天 focus_min 平均（含 0 的天） }`（**不传 label**，F10：避免与今天柱数值标签碰撞）；数据全为 0 或 trend 非 ready 时不传。仪表盘不传 `yAxisLabel`（副标题已含单位，375px 下旋转轴标题会压住刻度）。`.mf-chart-value-label` 加 `paint-order: stroke; stroke: var(--surface); stroke-width: 3px; stroke-linejoin: round;`，使均值虚线不切穿柱顶数字。
- 空/错误沿用 BarChart 现有 emptyMessage 语义。

#### 4.1.5 数据获取

- Dashboard 的 `fetchData` **移除** `getHealth`、`getCollectorStatus`、`getAiProviderStatus`（移入 StatusChip）；**保留** `getModelStatus`、`getFocusTrend(7)`、`getCurrentActivity`、`getInterventionHistory(7)`、`getAutonomy`、`getFocusPrediction` 及现有实时订阅。
- `health.version` 只在 StatusChip 浮层显示；`realtimeStatus` 只在 Hero 的 LiveDot 显示。
- 现有部分失败提示 `部分数据加载失败（N 项），其余内容已显示` 改用 `.notice` 渲染在网格上方。

#### 4.1.6 字段映射（实现者照此核对，报告中逐行打勾）

| 旧位置 | 字段 | 新位置 |
|---|---|---|
| 状态卡·系统健康 | health.status | StatusChip 状态 + 浮层"服务" |
| 状态卡·采集器 + 顶部采集按钮 | collector.running + 切换 | StatusChip 状态 + 浮层"采集器"+按钮 |
| 状态卡·数据库 | health.database.status | StatusChip 状态 + 浮层"数据库" |
| 状态卡·LLM 层 | provider configured / model | 浮层"云端 AI" |
| 浮动日期头 | 日期星期 | 顶栏日期 |
| 当前活动 | app_name / window_title | Hero 活动行 + 依据展开"当前窗口" |
| 当前活动 | process_name / is_idle | 依据展开"当前窗口"（is_idle 同时影响活动行文案） |
| 当前活动 | timestamp + UTC 偏移 | Hero "更新于…" + 依据展开"最后更新" |
| 当前活动空态 + 实时连接中 | — | 活动行 `当前没有活动记录` + LiveDot |
| 7 日趋势 | 全部 | 最近 7 天卡（+均值线、平均分 tooltip） |
| 4 KPI | 全部（含 delta） | 今天 MetricStrip |
| 自主控制 | 状态/暂停至/小时分钟暂停/恢复/提示 | 下一步·自动提醒 |
| 最近干预（最多 10 条） | 最新一条全部字段 | 下一步主体 |
| 最近干预 | 其余条目 | 条数 + "去干预中心"（完整列表在干预中心） |
| ML 专注预测 | 概率 / 原因 / 模式 / demo 标注 | Hero 读数 / 未就绪行 / 来源行 |
| ML 专注预测 | explanation（方法、贡献、note） | Hero "主要因为" + 依据展开 |
| 服务状态 | 模型已加载/演示/规则引擎 | Hero 来源行 + 浮层"预测模型" |
| 服务状态 | 版本 / 实时连接 / 质量门标注 | 浮层"版本" / LiveDot / 来源行 |

## 5. 工程约束

- 不改后端；不改 API 调用语义；不新增 npm 依赖；不加载网络字体。
- e2e：结构变更导致的断言失败改为**等价断言**（验证同一数据），不得删除数据正确性断言、不得放宽阈值；每处修改在报告中列出原因。`src/types/explainability.ts` 中 10 个 `_wire*` 常量是编译期漂移守卫，**不得删除**。
- 质量门：`npm run build` 0 错误 0 警告；`npm run check:api-drift`；`npx playwright test -c playwright.ui-review.config.ts` 全绿（当前基线 240 passed / 13 skipped，报告通过/跳过数）；`MF_REVIEW_SHOTS=1` 重新生成评审截图（1440 navy、1440 white、375 navy；另加一张 1440 navy 仪表盘 Hero 区域裁剪图 `dashboard-hero-navy.png`），目录 `e2e/screenshots/midterm-review/`。
- 375px 无横向溢出（`document.documentElement.scrollWidth <= 375`）。
- 不提交 git。
- 本机 Bash/Edit 偶发重复执行：每次编辑后 grep 核对无重复块/重复声明。

## 6. 阶段与关卡

- **阶段 1（现在）**：§2 全部 + §3 全部组件 + §4.0 外壳 + §4.1 仪表盘 + §3.11 WIG。其他页面只通过 §2.2 变量映射自动换色、§2.4 字重清扫、§3.8 按钮/表单全局样式、§3.11 修复受影响，**不改其信息架构**。完成后**停止**并汇报：截图路径、§4.1.6 映射逐行打勾、e2e 改动清单、质量门结果、所有"规格未覆盖、我没有自行决定"的问题。
- **阶段 2**：§4.2–4.5（行为洞察、干预中心、专家面板、专注分析/报告中心）。主代理在用户认可阶段 1 后，按本文件同等粒度补写阶段 2 细则再下发；在此之前不得开工。

### 阶段 2 方向（仅供理解全局，细则待补）
- 行为洞察：顶部"要点"一行发现；三个区块改问题式标题"什么时候、被什么打断" / "你的专注习惯" / "为什么会拖延"（TMT 为主角）；~~去掉"个人模型与云端 AI"卡~~（已被 §8.4 第 9 条取代：保留并改版该卡）；只显示中文指标名；空数据折叠成一行。
- 干预中心：去重复 H1；强度 Segmented + "现在提醒我"；最新提醒为主角（理由同容器展开）；历史 SectionHeader + 7/14/30/90 Segmented；固定列修复 P8。
- 专家面板：去重复 H1；裁决为主角；专家意见发丝线列表。
- 专注分析/报告中心：MetricStrip、Segmented、说明收为一行、图表语义色。

---

## 7. 阶段 2 · 全站统一规则（2026-11-10 主代理定稿；用户反馈"好很多了，但还是有些风格/大小不统一"）

**主代理自查（12 页实拍，1440 navy）**：只有仪表盘是新体系；专注分析/报告中心/行为洞察/活动日志仍是旧风格（粗标题、彩色图标方块、彩色胶囊、居中装饰标题"—— 模式分析 ——"、发光大按钮"日报/周报"、三套柱色）；干预中心/专家面板/执行/对话/设置/诊断/模型中心是"工作区"风格（28px H1 与顶栏标题重复 + 分隔线、边框盒式标签栏、卡中套卡）；全站 CSS 共 17 种字号。仪表盘内部也有 4 处不统一（卡片标题三种样式、`min` 与中文单位混用、同卡按钮两种高度、18px 字号不在字阶内）。

以下规则**覆盖全站**，优先级高于 §3/§4 中与之冲突的旧条目（冲突处已在本节注明"替换"）。

### 7.1 字阶与角色（唯一允许的组合）

| 角色 | 字号/字重/行高 | 颜色 | 用在哪 |
|---|---|---|---|
| 卡片标题 `.sh-title` | 16 / 600 / 24 | text | 每个 Surface 的第一行（仪表盘 Hero 除外，见 7.4） |
| 卡内小标题 `.sub-title` | 14 / 600 / 20 | text | 卡内分组（如"自动提醒""当前窗口"） |
| 正文 | 14 / 400 / 1.5 | text | 列表内容、说明正文 |
| 强调正文 | 16 / 500 / 1.5 | text | 卡内唯一主角句（如"下一步"的提醒文案） |
| 次要/元信息 | 13 / 400 / 1.5 | text-dim | 卡片副标题、页面导语、表单标签、列表次要列 |
| 注释/刻度 | 12 / 400 或 500 / 16 | text-dim | 时间戳、图表刻度、图例、徽标 |
| 指标值 | `--font-num` 28 / 400 / 32 | text | MetricStrip 数值（单位 14 text-dim） |
| 状态词 | 28 / 600 / 32 | 语义色 | 仅仪表盘 Hero"专注中/状态摇摆/分心中" |
| Hero 读数 | `--font-num` 64 / 300 | signal/drift/text | 仅仪表盘 |

- **全站 CSS 与内联样式只允许 12/13/14/16/20/28/44/64**（写 `var(--fs-*)`）。旧字号映射：10、11→12；15→14；17、18→16；20、22、24、25、26→20（仅 7.3 的页面分组标题可用 20，其余位置按上表角色改）；30、35→28；48→44。
- 字重只允许 300/400/500/600（§2.4 不变）。`letter-spacing` 只允许 `.label-caps` 的 .04em，其余删除（含"刷 新""日 报"这种字间空格文案：去掉空格）。
- **`.label-caps` 只用于 MetricStrip 之外的极少数"眉标"**：本阶段起全站不再新增使用；仪表盘现有 3 处按 7.4 改掉。

### 7.2 单位与数字格式

- 时长统一以**分钟**为单位、中文单位：`183 分钟`（数值与单位间一个空格；MetricStrip 中单位走 unit 槽，显示为 `183` + `分钟`）。**禁止** `min`、`m`、`4h 30m`、`3h 0m` 写法。≥600 分钟时仍用分钟（不换算小时），保证跨页可比。
- 次数 `次`、分数 `/100`、比例 `%`，与仪表盘一致。
- 未知值统一 `—`（U+2014），不带单位；**全站**替换 `--`、`N/A`、`暂无`（作为数值占位时）。`暂无`作为整句空态文案保留。
- 图表 tooltip 单位同上（`" min"` → `" 分钟"`）。

### 7.3 页面骨架（所有非仪表盘页面）

```
顶栏（已有：标题 + 日期 + StatusChip）
.mf-main
  .page-lede        13/400 text-dim，max-width 720px，margin 0 0 24px；一句话说明本页做什么（文案见 7.6）
  .page-toolbar     （可选）display:flex; flex-wrap:wrap; align-items:center; gap:12px; margin-bottom:24px
                    放本页级控件：日期、范围 Segmented、刷新、页内 Tab（Segmented）。控件高度一律 32px。
  内容区            卡片间距 16px（grid/flex gap 16）；页面内大分组之间 32px
    .page-section-title（可选，仅长页面分组）20/600/28 text，左对齐，margin 32px 0 16px，首个分组 margin-top 0；无装饰线
```
- **删除页面内 H1 与分隔线**（`WorkspacePage` 的 heading 区、各页手写的居中大标题）：标题已在顶栏，重复即删。`WorkspacePage` 的 `description` 改渲染为 `.page-lede`。
- **页内 Tab 一律用 `Segmented`**（设置、模型中心、执行、报告中心"日报/周报"、干预历史"7/14/30/90 天"、行为洞察时间范围），放在 `.page-toolbar` 或所属卡片的 SectionHeader actions 中。原边框盒式按钮组、发光大按钮全部删除。
- **一层容器**：所有卡片用 `Surface`；Surface 内禁止再放带边框/背景的盒子（§3.1 重申）。原"卡中卡"改为 `.list-row` 行或 `border-top` 分组。
- **统计卡一律改为 `MetricStrip`**（无 `label`、无 delta，除非该页已有昨日对比数据）；删除所有彩色图标方块（StatTile 的 icon、focus/reports 的 iconfont 图标）。
- **胶囊/徽标**：只保留 `.badge`（12/500，height 20，padding 0 8px，radius pill，背景 `*-soft`，文字语义色），用于**状态**（通过/未通过/运行中/已停止）；数值、时间段、应用名不得用胶囊，改为普通文字（数值走 `--font-num`）。
- **按钮尺寸**：页面工具栏按钮 `.btn`（32px）；卡片内操作一律 `.btn .btn-sm`（28px）；同一行按钮同一尺寸。裸 `.btn` 全部补显式变体（主操作 `btn-primary`，每屏最多 1 个；其余 `btn-secondary`；文字链式 `btn-ghost`；破坏性 `btn-danger`）。操作行 `display:flex; flex-wrap:wrap; gap:8px; justify-content:flex-start`，**禁止**按钮拉伸等分整行。
- **表单**：label 13/400 text-dim 在上、间距 6px；控件 32px 高（textarea 最小 72px）、radius 6、14px；复选框 `accent-color: var(--signal)`，16×16，与文字 13px 间距 8px；搜索框不再用胶囊圆角。表单字段网格 gap 16px。
- **空态**：Surface 内居中 13px text-dim 一到两行 + 可选一个 `.btn-sm .btn-secondary`；**删除虚线边框框**（dashed border 盒子）。
- **图表配色**（全站）：专注 = `var(--signal)`，分心 = `var(--drift)`，中性/往日 = `var(--steel)`，今天/当前高亮 = `var(--signal)`；不允许其他柱色（绿/棕/青蓝自定义色全部删除）。凡手写 div 柱状图改用 `BarChart` / `StackedBar` 组件。

### 7.4 仪表盘内部统一（替换 §4.1 中的对应条目）

1. 四张卡标题统一为卡片标题样式（16/600 text），文案不变：`此刻`、`下一步`、`今天`、`最近 7 天`。
   - Hero：标签行左侧 `此刻`（`.sh-title`，不再 `.label-caps`）+ `LiveDot`（margin-left 12px）；右侧"更新于…"12px text-dim 不变。
   - 下一步：`下一步` 用 `.sh-title`，margin-bottom 16px。
   - MetricStrip 的 `label` 列文字改为 16/600 text（替换 §3.3"首列 label：.label-caps"），列宽 64px 不变。
   - `SectionHeader` 的 `.sh-title::after` 渐变短线**删除**（替换 §3.2），全站无装饰线。
2. "下一步"提醒文案 18/500 → **16/500**（强调正文角色）。
3. "下一步"卡内按钮统一 `.btn-sm`：`去干预中心`（btn-secondary btn-sm）；暂停组四个按钮 `暂停 30 分钟` / `暂停 1 小时` / `今天不再提醒` / `自定义` **全部** `btn-secondary btn-sm`，同一行 flex-wrap gap 8（`自定义`不再是 ghost，消除缩进错位）。`自动提醒`用 `.sub-title`（14/600）。
4. 单位：MetricStrip `专注时长` 单位 `min` → `分钟`；7 天图 tooltip `" min"` → `" 分钟"`；tooltip 明细 `分心 {n} min` → `分心 {n} 分钟`。
5. 依据展开内 `当前窗口` 用 `.sub-title`（14/600），不再 `.label-caps`。

### 7.5 侧栏图标（替换 iconfont，统一线性风格）

旧 iconfont 图标风格混杂（有填充有线框、视觉大小不一）。新建 `src/components/signal/NavIcon.tsx`：`<NavIcon name />`，渲染 `<svg width=18 height=18 viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">`，内部路径**逐字照抄**下表：

| name | 用于 | SVG 子元素 |
|---|---|---|
| dashboard | 仪表盘 | `<path d="M4 15a8 8 0 1 1 16 0"/><path d="M12 15l4-5"/><circle cx="12" cy="15" r="1.5"/>` |
| focus | 专注分析 | `<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1"/>` |
| activities | 活动日志 | `<path d="M9 6h11M9 12h11M9 18h11"/><circle cx="4.5" cy="6" r="1"/><circle cx="4.5" cy="12" r="1"/><circle cx="4.5" cy="18" r="1"/>` |
| analytics | 行为洞察 | `<path d="M4 19h16"/><path d="M5 15l4-4 3 3 6-7"/><path d="M15 7h3v3"/>` |
| reports | 报告中心 | `<path d="M7 3h7l4 4v14H7z"/><path d="M14 3v4h4"/><path d="M10 12h5M10 16h5"/>` |
| intervention | 干预中心 | `<path d="M6 16v-5a6 6 0 0 1 12 0v5l1.5 2h-15z"/><path d="M10 20.5a2 2 0 0 0 4 0"/>` |
| panel | 专家面板 | `<circle cx="9" cy="9" r="3"/><path d="M3.5 19a5.5 5.5 0 0 1 11 0"/><circle cx="17" cy="8" r="2.5"/><path d="M16 13.2a4.5 4.5 0 0 1 5 4.8"/>` |
| chat | AI 对话 | `<path d="M4 5h16v11H9l-5 4z"/>` |
| settings | 系统设置 | `<path d="M4 7h10M18 7h2M4 17h4M12 17h8"/><circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/>` |
| model | 模型中心 | `<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5M12 12v9"/>` |
| execution | 干预执行 | `<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M8.5 12l2.5 2.5 4.5-5"/>` |
| diagnostics | AI 诊断 | `<path d="M3 12h4l2-6 4 12 2-6h6"/>` |
| logout | 退出（本地会话行） | `<path d="M14 4h5v16h-5"/><path d="M10 8l-4 4 4 4M6 12h10"/>` |
| sidebar | 顶栏折叠按钮 | `<rect x="4" y="5" width="16" height="14" rx="2"/><path d="M9 5v14"/>` |
| chevrons | 高级功能折叠 | `<path d="M12 6l-6 6 6 6M18 6l-6 6 6 6"/>`（展开态保持现有旋转类） |
| close | 移动抽屉关闭（原 icon-chacha） | `<path d="M6 6l12 12M18 6L6 18"/>` |

- App.tsx 中 `NAV`/高级导航数组的 `icon` 字段改为上表 `name`，所有 `<span className="iconfont …">` 换成 `<NavIcon name=… />`，保留原有包裹元素与 className（如 `mf-user-icon`、`mf-nav-group-chevron…`）挂到外层 span 上，保证旋转/颜色样式继续生效。
- shell.css 中针对 `.iconfont` 的 font-size 规则改为对 svg：`.mf-nav-item svg { flex: 0 0 18px; }`，颜色继承（active 已是 signal）。
- `iconfont.css` 仍被 EmptyState/StatTile/Toaster 使用时保留 import；页面统计卡图标按 7.3 删除后若 iconfont 无任何引用，则从 App.tsx 删除 import 与资源**不删**（文件保留，避免误删）。

### 7.6 页面导语文案（`.page-lede`，逐字）

| 页面 | 文案 |
|---|---|
| 专注分析 | 按天查看专注与分心的时长，并校准系统对每段时间的判断。 |
| 活动日志 | 按时间查看记录到的应用和窗口，可按日期筛选或搜索。 |
| 行为洞察 | 从最近一段时间的切换和应用使用中，找出你容易分心的时段与原因。 |
| 报告中心 | 按日或按周汇总你的专注表现。 |
| 其余工作区页面 | 沿用各自现有 `description` 原文，不改字 |

（各页面的逐页细则见 §8。）

---

## 8. 阶段 2 · 逐页细则（主代理定稿 2026-11-10；与 §7 一起是必须照做的实现规范）

### 8.0 通用机械映射（本节没有逐条点名的旧样式，一律按下表改，不留判断空间）

**字号**（CSS 字面量和 TSX 内联 fontSize 都适用，统一写成 `var(--fs-*)`）：

| 旧值 | 新值 |
|---|---|
| 10、11、12 | `--fs-xs`（12） |
| 13 | `--fs-sm`（13） |
| 14、15 | `--fs-base`（14） |
| 16、17、18、20、22、24、25、26 | `--fs-md`（16）。例外：`.page-section-title` 用 20，`.login-title` 用 28 |
| 28、30 | `--fs-xl`（28） |
| 35 | 整条规则删除（它只给 iconfont 统计图标用，图标已删） |
| 48 | `--fs-2xl`（44） |

**字重**
- 700 / bold → 600。
- theme.css 全局加 `b, strong { font-weight: 600; }`。

**颜色**
- `rgb(218,51,51)` → `var(--drift)`。
- 焦点阴影 `rgba(24,144,255,0.2)` 删除，焦点统一为 `outline: 2px solid var(--signal); outline-offset: 2px`。
- 含以下颜色的规则整条删除（对应胶囊已按下文改掉）：`rgba(126,248,165,*)`、`rgba(173,186,87,*)`、`rgba(247,96,93,*)`、`rgba(103,215,144,*)`、`rgba(191,47,33,*)`。
- 旧变量映射：

  | 旧变量 | 新变量 |
  |---|---|
  | `var(--color-text-secondary)` / `var(--color-text-tertiary)` | `var(--text-dim)` |
  | `var(--color-bg-secondary)` | `var(--surface-2)` |
  | `var(--color-success)` / `var(--good)` | `var(--ok)` |
  | `var(--color-warning)` / `var(--warn)` | `var(--drift)` |
  | `var(--bad)` | `var(--alarm)` |

- 内联 `var(--x, #hex)` 去掉 hex 回退值。

**间距**（内联 margin / padding / gap）
- 取 {4, 8, 12, 16, 24, 32} 中最接近的值，距离相同时取大的：6→8，10→12，14→16，20→24。

**box-shadow**
- 页面 CSS 中全部删除（§2.5 规定无阴影）。login.css 例外。

**letter-spacing**
- 全部删除。只保留 `.label-caps` 和 8.6 的配对码。

**内联样式**
- 只保留数据驱动的值：柱高 %、宽度 %、`--opinion-color` 这类 CSS 变量。
- 其余内联排版和颜色按上面各表改写。能归类的移进页面 CSS，类名格式为 `<页面前缀>-<用途>`。

**`.card`（工作区）**
- workspace.css 的 `.workspace-page .card` 统一为 `background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius-lg); padding: 24px;`。
- `.card h3`：16/600，颜色 text，`margin: 0 0 16px`。
- h3 后紧跟 `p.card-desc` 时，`.card-desc` 为 13/400 text-dim，`margin: -12px 0 16px`。

**`.workspace-details`（顶层 `<details>`）**
- 视为一级容器，样式同 `.card`，但 `padding: 16px 24px`。
- summary：14/600。
- 内部禁止再出现带边框或背景的盒子。

**`.workspace-record`**
- 去掉边框、背景、圆角，改为 `border-top: 1px solid var(--line); padding: 12px 0;`。
- `:first-child` 不要上边框。
- h4：14/600。

**`.tabs` / `.tab`**
- 所有使用处改为 `Segmented`（各页的 ariaLabel 和 options 见 8.x）。
- 改完后若已无引用，从 theme.css 删除这两条规则（它们是本次改动造成的孤儿）。

**`.stat-card` / `.kpi-row` / `.mc-kpi-row`**
- 使用处改为 `MetricStrip`。
- 改完后 CSS 若已无引用，删除。

**`EmptyState` 组件**
- 删除图标渲染和 `icon` prop，并同步修改所有调用处。
- 样式：居中，13/400 text-dim，`padding: 32px 0`，无边框。

**`.error-box` / `.success-box` / `.info-box`**
- 类名保留（e2e 依赖）。
- 外观统一为 §3.10 `.notice` 的几何：`padding: 10px 16px; border: 1px solid var(--line); border-radius: var(--radius-sm); font-size: 13px; background: var(--surface)`。
- 左边框 2px，颜色依次为 `--alarm`、`--ok`、`--signal`。

**原生表单控件**（theme.css 全局）
- `input:not([type=checkbox]):not([type=radio]), select`：`height: 32px; padding: 0 10px; border: 1px solid var(--line-strong); border-radius: 6px; background: var(--surface-2); color: var(--text); font-size: 14px`。
- `textarea`：配色同上，`min-height: 72px; padding: 8px 10px`。
- `input[type=checkbox]`：`accent-color: var(--signal); width: 16px; height: 16px`。
- select 保留现有的 svg chevron 背景（density-review 断言它存在）。

**`.native-date-field`**（layout-refinements.css:37）
- 标签 13/400 text-dim，`gap: 6px`。
- 输入框同上，高 32。

### 8.1 共享新增类（实现者 A 写到 signal.css 末尾；B 直接使用）

```css
.page-lede { font-size: var(--fs-sm); color: var(--text-dim); max-width: 720px; margin: 0 0 24px; line-height: 1.5; }
.page-toolbar { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 12px; margin-bottom: 24px; }
.page-section-title { font-size: var(--fs-lg); font-weight: 600; line-height: 28px; color: var(--text); margin: 32px 0 16px; text-align: left; }
.page-section-title:first-of-type { margin-top: 0; }
.sub-title { font-size: var(--fs-base); font-weight: 600; line-height: 20px; color: var(--text); margin: 0 0 8px; }
.page-grid-2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
@media (max-width: 900px) { .page-grid-2 { grid-template-columns: minmax(0, 1fr); } }
.kv-row { display: grid; grid-template-columns: 1fr auto; align-items: center; min-height: 44px; column-gap: 16px; border-top: 1px solid var(--line); font-size: var(--fs-base); }
.kv-row:first-child { border-top: none; }
.kv-row > :first-child { color: var(--text-dim); font-size: var(--fs-sm); }
.kv-row > :last-child { font-family: var(--font-num); color: var(--text); text-align: right; font-variant-numeric: tabular-nums; }
.kv-row--text > :last-child { font-family: inherit; }
.kv-row--strong > :first-child { color: var(--text); font-size: var(--fs-base); }
```

**`.kv-row` 用法**
- 页面里所有"标签—值"列表都用 `.kv-row`。
- 值是中文或长文本时，加 `.kv-row--text`。
- 左列是 14 号正文（时段、应用名）时，加 `.kv-row--strong`。

**图表组件补充（A 负责）**：只加属性、不改外观，供 e2e 精确断言。
- `BarChart`：柱 `<rect class="mf-chart-bar">` 加 `data-value={datum.value}`。
- `StackedBar`：每一段 rect 加 `data-series={series.key}` 和 `data-value={value}`。

**`MetricStrip`（A 负责）**：`.sig-metrics-label` 改为 16/600 text，替换原来的 label-caps（见 §7.4）。

### 8.2 专注分析 `Focus.tsx` + `focus.css`（实现者 B）

根元素仍为 `div.f-page`，从上到下依次渲染：

1. `p.page-lede`，文案见 §7.6。

2. `div.page-toolbar.f-first-row`，内含：
   - `div.f-datepicker.mf-picker > NativeDateField label="选择日期"`，不改。
   - `button.btn.btn-secondary.f-reset`，文案为"刷新"（中间无空格），disabled 逻辑不变。

3. `sessionsError` 的 `.error-box`，不改。

4. `div.f-state > MetricStrip label="当日"`，items 如下：

   | key | label | value | unit |
   |---|---|---|---|
   | focus | 总专注时长 | `String(Math.round(totalFocus))` | `分钟` |
   | sessions | 专注次数 | `String(sessionCount)` | `次` |
   | score | 平均评分 | `avgScore.toFixed(1)` | — |
   | longest | 最长专注 | `String(Math.round(longestBlock))` | `分钟` |

   - 原来显示"—"的条件不变；显示"—"时不传 unit。
   - note 逐字为：`专注评分由系统根据主要应用占比和切换频率自动算出。下方的反馈只用于改进个人模型，可以不填。`
   - 删除 `div.f-description` 和统计图标。

5. `section.sig-surface.f-statics`（`margin-top: 16px`）：
   - `SectionHeader title="近 7 天专注与分心" as="h2"`。
   - 加载中保留 `.spinner`；加载完成后渲染 `StackedBar`，参数如下：
     - `data = trendDays.map(d => ({ key: d.date, label: chartWeekLabel(d.date, todayKey), title: `${d.date.slice(5)} ${chartWeekLabel(d.date, todayKey)}`, values: { focus: Math.round(d.focus_min ?? 0), distraction: Math.round(d.distraction_min ?? 0) }, highlight: d.date === todayKey }))`
     - `series = [{ key: "focus", label: "专注", color: "var(--signal)" }, { key: "distraction", label: "分心", color: "var(--drift)" }]`
     - `unit=" 分钟"`、`height={220}`、`showLegend`
     - `ariaLabel={`近 7 天每日专注与分心分钟数，共 ${trendDays.length} 天`}`
     - `emptyMessage = trendError ? "趋势加载失败，请重试" : "暂无趋势数据"`；trendError 时 data 传空数组。
   - 删除手写柱图（`.statics-chart*`、`.chartbox*`、`.bottom-index*`），以及随之成为孤儿的 chartMax、maxFocus、maxDistraction 等变量。

6. `section.sig-surface.f-conversation`（`margin-top: 16px`）：
   - `SectionHeader title="专注会话" as="h2"`。
   - 空态和加载逻辑不变。空态 `.f-empty` 改为 13 text-dim、居中、`padding: 32px 0`。

7. 每条 `.conversation-item`：
   - 去掉边框和背景，改为 `border-top: 1px solid var(--line); padding: 16px 0`，第一条不要上边框。
   - `.statement-table`：`display: grid; grid-template-columns: 1fr auto; gap: 16px; align-items: center`。
   - 左列 `.left-table`：`display: flex; gap: 32px`，分三组。每组上一行是 12 号 text-dim 小标签，下一行是 14 号 font-num 的值：
     - `.time-table`：标签 `{dayLabel} {MM-DD}`，值为开始时刻。
     - `.during-table`：标签"会话时长"，值 `${Math.round(duration)} 分钟`。
     - `.change-table`：标签"切换次数"，值 `${switches} 次`。
   - 右列 `.right-table`：右对齐，纵向两行：
     - `.process`：应用名，14/500 text。
     - `.score`：`${Math.round(score)} 分 · 专注|分心`，14 号 font-num；focusLike 时颜色为 `var(--signal)`，否则为 `var(--drift)`。
   - 删除 `.description`（内容已并入 score 行）、`.green` 和 `rgb(218,51,51)`。
   - `.self-accession-table`：`display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; margin-top: 12px`，无背景、无边框。
     - select 上方的标签用 13 text-dim；select 按 8.0 处理，另加 `min-width: 120px`。
     - `button.save-buttom` 的 className 改为 `btn btn-secondary btn-sm save-buttom`。
   - `.accession-tip`：删除 iconfont 图标；文字 12 text-dim，`margin-top: 8px`。
   - `.f-saved`：12 号，颜色 `var(--ok)`。行内错误保持 `role="alert"`。

8. 删除 `div.f-bottom`。

9. focus.css：删除因以上改动失去引用的规则，其余规则按 8.0 映射。

10. 本文件的 `formatMinutes` 改为：`m == null || isNaN(m) ? "—" : `${Math.round(m)} 分钟``。

### 8.3 报告中心 `Reports.tsx` + `reports.css` + `report-view.ts`（实现者 B）

1. `p.page-lede`，文案见 §7.6。

2. `div.page-toolbar.top-buttom`，内含：
   - `Segmented ariaLabel="报告类型" options=[{value:"daily",label:"日报"},{value:"weekly",label:"周报"}] value={tab} onChange={v => setTab(v as typeof tab)}`。
   - `div.report-datepicker.mf-picker > NativeDateField`，逻辑不变。
   - 删除 `button.daily-report` / `.weekly-report` 及对应 CSS。

3. `.error-box.report-error`：
   - 重试按钮用 `btn btn-secondary btn-sm`。
   - 内联 marginLeft 改为类 `.report-retry { margin-left: 12px; }`。

4. `div.t-state > MetricStrip`，label 日报为 `当日`、周报为 `本周`，items 如下：

   | 日报 label | 周报 label | value | unit |
   |---|---|---|---|
   | 总专注时长 | 总专注时长 | `String(Math.round(<daily或weekly>.total_focus_minutes))` | `分钟` |
   | 专注次数 | 总专注次数 | `kpis.totalSessions` | `次` |
   | 分心次数 | 总分心次数 | `kpis.totalDistractions` | `次` |
   | 专注评分 | 日均评分 | `kpis.focusScore` / `kpis.avgFocusScore` | — |

   - 无数据或加载中：4 项 value 都显示 `—`，不带 unit。
   - 删除 `DAILY_STATE_ICONS` 和 icon 字段；`placeholderStats` 改为返回 MetricItem 结构。

5. `section.sig-surface.zhuanzhudongcha`（`margin-top: 16px`）：
   - `SectionHeader` 标题：日报为 `今日洞察`，周报为 `本周洞察`。
   - 正文 `.zhuanzhudongcha-text`：14/400 text，`line-height: 1.6`。空态文案不变。

6. `section.sig-surface.chart-box`（`margin-top: 16px`）：
   - `SectionHeader title={chartTitle}`。
   - 渲染 `BarChart`，参数如下：
     - `data = shownBuckets.map((b, i) => ({ key: `${b.hour}-${i}`, label: b.hour, value: b.minutes, title: b.hour }))`
     - `unit=" 分钟"`、`height={220}`、`ariaLabel={chartTitle}`、`emptyMessage="暂无数据"`
     - `colorFor={() => "var(--signal)"}`：这张图只有"专注"一个量，全部用 signal，不做今日高亮。
   - 删除手写柱，以及随之成为孤儿的 shownMax 等变量。

7. `section.sig-surface.report-detail-list`（`margin-top: 16px`）：
   - `SectionHeader title={detailTitle}`。
   - 表格保留（`simple-table` / `weekly-detail-table`），样式统一为：
     - th：12/500 text-dim。
     - td：14 号。
     - 行高 44；行与行之间 `border-top: 1px solid var(--line)`；无斑马纹。
   - 时长列改用新的 formatMinutes。

8. 周报趋势 `.trend-row`：
   - 改为 `div.trend-row` 里的 `.kv-row` 列表，每行为 `<span>{m.label}</span><span>{m.display} · {m.sub}</span>`。
   - 值的颜色：`m.good` 为真时用 `var(--signal)`，否则用 `var(--drift)`。
   - 删除 `.trend-pill` 及其 CSS。

9. `report-view.ts`：
   - `formatMinutes` 同 8.2，输出 `${Math.round(m)} 分钟`。
   - 第 100 行的柱 label 改为 `${Math.round(minutes)} 分钟`。
   - 第 219 行沿用 formatMinutes。
   - 同步更新 `tests/report-state-mapping.test.ts`：`"2h 0m"` → `"120 分钟"`，`"30m"` → `"30 分钟"`，`"60m"` → `"60 分钟"`。

### 8.4 行为洞察 `Analytics.tsx` + `analytics.css`（实现者 B）

1. `p.page-lede`，文案见 §7.6。

2. `div.page-toolbar.time-box`：
   - `Segmented ariaLabel="时间范围" options=DAYS_OPTIONS.map(d => ({ value: String(d), label: `近 ${d} 天` })) value={String(days)} onChange={v => setDays(Number(v))}`。
   - 删除 `.time-text` 和 `select.time`。

3. `.error-box.error-slot`：关闭按钮用 `btn btn-secondary btn-sm`；marginLeft 改为类。

4. `renderTitle` 改为输出 `<h2 className="page-section-title" ref=…>`（保留 titleRefs）；删除旧的 `.title` CSS。

5. 「模式分析」：`div.page-grid-2.top-box`，内含两张卡 `section.sig-surface.high-change` 和 `section.sig-surface.toggle-app`。
   - 卡头 `SectionHeader` 分别为 `高切换时段` 和 `触发应用 Top`。
   - 卡内是 `.kv-row.kv-row--strong` 列表，左列 `row.left`，右列 `row.right`。
   - 右列取值：
     - `toHighSwitchRows` 改为 `${p.switch_count} 次切换`。
     - `toTriggerAppRows` 改为 `${a.count} 次`（数字和"次"之间加空格）。
   - 空态 `.f-empty`：13 text-dim、居中、`padding: 32px 0`。加载中显示 `.spinner`。
   - 删除这两处的 `table.simple-table`，以及 `.full-box1` / `.full-box2` 的 CSS。
   - 卡内仍保留 `div.f-box` 包裹（e2e 会找 `.f-box .spinner`），但这个类不带任何视觉样式。

6. `div.second-box > MetricStrip label="基线对比"`（`margin-top: 16px`）：
   - 平均切换次数：value 为 `baseline?.mean_app_switch_count`，unit `次`。
   - 活跃时间占比：value 为 `(x*100).toFixed(1)`，unit `%`。
   - 空闲时间占比：同上。
   - 缺值时 value 为 `—`，不带 unit。

7. 「个人画像」：`div.third-box > MetricStrip label="画像概览"`，items 如下（delta 的 tone 一律为 `"none"`）：

   | label | value | unit | delta.text |
   |---|---|---|---|
   | 专注高峰 | `peak_focus` 中 `" ("` 或 `"("` 之前的部分 | — | 有 `avg N` 时为 `平均分 N`，否则为 `平均评分最高的时段` |
   | 效率应用 | `productivity_apps.length` | `个` | 前 3 项用 `、` 连接；没有则为 `—` |
   | 平均专注块 | `avg_focus_block_min` | `分钟` | `每次连续专注的平均时长` |
   | 触发应用 | `trigger_apps.length` | `个` | 前 3 项用 `、` 连接；没有则为 `—` |

   profile 为空时，value 为 `—`，不显示 delta。

8. `section.sig-surface.fourth-box`（`margin-top: 16px`）：
   - `SectionHeader title="详细画像"`。
   - 表格保留，样式同 8.3 第 7 条。
   - 趋势列 badge 保留：上升用 `badge-success`，下降用 `badge-warning`，其余用 `badge-neutral`。

9. 「拖延归因」：`div.page-grid-2.fifth-box`（保留 ref），分左右两张卡。

   **左卡** `section.sig-surface.left-box`：
   - `SectionHeader title="归因结果"`，actions 放 `button.btn.btn-primary.btn-sm.attribution-action`（这是本页唯一的 primary 按钮）。
   - 4 条 `.kv-row.kv-row--text`：拖延类型、置信度（值用数字文本，不用 badge）、认知行为技术（CBT）、证据。
   - 证据行：`grid-template-columns: 120px 1fr; align-items: start; padding: 12px 0`；值 `white-space: normal; text-align: left`。
   - 删除 `.attribution-summary`、`.attribution-details`、`.left-box-title*` 的结构和 CSS。

   **右卡** `section.sig-surface.right-box.analysis-capabilities`：
   - `SectionHeader title="个人模型与云端 AI"`。
   - 两组之间用上边框分隔：第二组加 `border-top: 1px solid var(--line); padding-top: 16px; margin-top: 16px`。
   - 组 1：`h3.sub-title`「个人专注模型（ML）」，右侧 badge：

     | 状态 | badge | 文字 |
     |---|---|---|
     | 已就绪 | `badge-success` | 已就绪 |
     | 已加载但未就绪 | `badge-warning` | 已加载，未就绪 |
     | 未加载 | `badge-neutral` | 未加载 |
     | 未知 | `badge-neutral` | 状态未知 |

     下方 `p` 用 13 text-dim，文案沿用原文；`.model-meta` 用 12 text-dim。
   - 组 2：`h3.sub-title`「云端 AI（LLM）」，右侧 badge：

     | 状态 | badge | 文字 |
     |---|---|---|
     | 已配置 | `badge-success` | 已配置 |
     | 未配置 | `badge-neutral` | 未配置 |
     | 未知 | `badge-neutral` | 状态未知 |

     已配置时，在 badge 后面用 13 text-dim 显示模型名。下方 `p` 沿用原文。
   - 删除 `.model-pill` 及其 CSS。

10. `AttributionEvidence`（`.attribution-evidence`）：
    - 根元素加 `sig-surface` 类，`margin-top: 16px`。
    - `.chip-row` 改为一行 14 号 text：`认知扭曲：` 后接各项，用 `、` 连接；不再使用 `.chip`。primitives.css 里的 `.chip` 若已无引用，由 A 删除。
    - `.callout`：由 A 在 primitives.css 改为无背景，`background: transparent; padding: 0 0 0 12px; border-left: 2px solid var(--ok); border-radius: 0`。
    - 其余部分（type bars、TmtGauge、deviation-table）不变。

### 8.5 活动日志 `Activities.tsx` + `activities.css`（实现者 B）

1. `p.page-lede`，文案见 §7.6。

2. `section.sig-surface.present-acti`：
   - 第一行：`h2.sh-title`「当前活动」，`margin-bottom: 8px`。
   - 第二行：14 号，应用名用 `<b>`，后接 ` — ` 和窗口标题（窗口标题用 text-dim）。
   - 单行省略：`white-space: nowrap; overflow: hidden; text-overflow: ellipsis`。

3. `div.page-toolbar.choose-time`（`margin-top: 16px`）：
   - 开始日期、结束日期两个 NativeDateField，保留外层 `.act-datepicker.mf-picker`。
   - `div.search-box`：上方是 13 text-dim 的标签「搜索」；input 宽 280px，按 8.0 处理（圆角 6，不再是胶囊形）。

4. `div.column-options`：
   - 样式：`display: flex; flex-wrap: wrap; gap: 16px; align-items: center; margin: -8px 0 16px; font-size: 13px; color: var(--text-dim)`。
   - 里面放原来的 raw 开关（`#act-raw-toggle`，label 文字不变）和 5 个 `label.column-toggle`。
   - 删除 `.check-primary-text` 包裹，id 和逻辑不变。

5. 删除 `.top-intro` 和所有 `.act-box` 嵌套，以及对应 CSS。

6. `section.sig-surface.detail-list`：
   - 表格样式同 8.3 第 7 条；`.cell-raw` 用 12 号。
   - `.changePage`：`display: flex; justify-content: space-between; align-items: center; margin-top: 16px`。
     - `.page` 用 13 text-dim。
     - 翻页按钮的 className 为 `btn btn-secondary btn-sm last-page` 和 `btn btn-secondary btn-sm next-page`，间距 8。
   - `.table-empty`：13 text-dim、居中、`padding: 32px 0`。

7. 文本替换：
   - 单元格缺值：`--` → `—`。
   - 状态：`idle` → `空闲`，`active` → `活跃`。
   - `formatDuration`：缺值返回 `—`；`s < 60` 返回 `${s} 秒`；否则返回 `${Math.round(s / 60)} 分钟`。

8. 重试按钮用 `btn btn-secondary btn-sm`；marginLeft 改为类。

### 8.6 系统设置 `Settings.tsx`（实现者 A）

**1. 分区切换**
- 改为 `div.page-toolbar > Segmented ariaLabel="设置分区"`。
- options：采集与隐私 / 应用分类 / 数据与偏好 / 外观与 AI / 系统状态（value 与 label 相同），绑定 `section`。
- 删除 `div.tabs.settings-nav`。

**2. 采集与隐私卡 `.privacy-settings`**
- h3「隐私行为采集」后接 `p.card-desc`（文案沿用原文）；右上角的状态 badge 不变。
- 两个 `label.settings-switch` 放进 `div.settings-switches`（`margin-top: 8px`，单列）：
  - switch 去掉边框、背景、圆角，改为 `display: flex; justify-content: space-between; align-items: center; gap: 16px; padding: 12px 0; border-top: 1px solid var(--line)`；第一个不要上边框。
  - `strong` 改为 `span.settings-switch-title`（14/500 text）。
  - 描述改为 `span.settings-switch-desc`（13 text-dim，`display: block; margin-top: 2px`）。
  - 删除原来的 inline grid。
- `.settings-retention`：`display: flex; flex-wrap: wrap; gap: 16px; margin-top: 16px`；其中 `.form-group` 的 label 为 13 text-dim，`margin-bottom: 6px`。
- 配对码 `div.pairing-code`：`margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--line)`，无背景。
  - 提示文字：13 text-dim。
  - 配对码本身：`font-family: var(--font-num); font-size: 28px; font-weight: 400; letter-spacing: 0.12em; margin: 4px 0`。
  - 到期时间：12 text-dim。
- `.telemetry-stats`：
  - 改为 5 条 `div.kv-row`（左标签、右值）；容器 `margin-top: 16px; max-width: 560px`。
  - 删除 strong 上的内联样式，以及 layout-refinements.css 第 110 行那条带边框的规则。
- `.telemetry-actions`：`display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; justify-content: flex-start`。按钮全部用 `btn-sm`：

  | 按钮 | 样式 |
  |---|---|
  | 生成浏览器配对码 | `btn btn-primary btn-sm` |
  | 清除输入数据 / 清除浏览器数据 / 清除反馈数据 | `btn btn-secondary btn-sm` |
  | 清除全部行为数据 | `btn btn-danger btn-sm` |

**3. 数据采集卡、自主控制卡**
- 采集按钮：运行中用 `btn btn-danger btn-sm`，否则用 `btn btn-secondary btn-sm`。
- 恢复按钮：`btn btn-secondary btn-sm`。
- 小时数 input 宽 80，用类 `.settings-hours`。
- 描述用的 div 改为 `p.card-desc`。

**4. 应用分类卡**
- 未识别应用的内嵌盒子（约第 587 行）：去掉背景和圆角，改为上边框加 `padding-top: 16px`。
- 可点击的应用名 `button.badge.badge-warning` 改为 `button.btn.btn-secondary.btn-sm`。
- `.workspace-table` 样式同 8.3 第 7 条。

**5. 系统状态区**（系统信息、AI 用量两张卡）
- label/value 网格改为 `.kv-row`；值是中文的行加 `.kv-row--text`。
- 状态 badge 保留在值的位置。

**6. 数据导出卡与 `details.workspace-details`**
- 按 8.0 处理。
- textarea 用 `font-family: var(--font-mono, monospace); font-size: 13px`。

### 8.7 AI 诊断 `Diagnostics.tsx` + `api.ts`（实现者 A；含 3 个功能 bug）

**1. 类型与后端对齐**（主代理已用 curl 核实真实返回）

```ts
export interface HealthLiveResponse { status: "alive"; version: string; timestamp: string; }
export interface HealthReadyResponse {
  status: "ready" | "not_ready"; version: string; timestamp: string;
  database: { status: "ok" | "error"; connected: boolean; integrity_ok: boolean };
  migration: { applied: boolean };
  checkpoint_store: "available" | "unavailable";
  run_store: "available" | "unavailable";
}
export type AIRunsResponse = components["schemas"]["DiagnosticsListResponse"]; // { items, count, has_more, next_offset }
```

- 所有读 `.total` 的地方改为读 `count`。Diagnostics 里改成 `setTotalRuns(res.count)`。
- 全仓 grep `AIRunsResponse` 和 `getAIRuns(`，其他使用者一并修改。
- 如果 `components["schemas"]["DiagnosticsListResponse"]` 的 items 元素类型与现有 `AIRunItem` 不兼容，停下来汇报，不要自行改造。

**2. health 请求不再静默吞错**
- 新增 state `liveFailed` / `readyFailed`（boolean），分别用 `.catch(() => setLiveFailed(true))` 和 `.catch(() => setReadyFailed(true))`。
- ready 接口在 not_ready 时返回 503，request 会抛错，因此会走 readyFailed 分支。
- `getFocusPrediction` 的 `.catch(() => {})` 同样改为 `setPredictionFailed(true)`。

**3. KPI 改为 `MetricStrip`（不带 label），共 4 项**

| label | value | delta |
|---|---|---|
| 服务存活 | live 有值：`正常`；liveFailed：`无响应`；其余：`—` | live 有值时为 `{ text: `版本 ${live.version}`, tone: "none" }` |
| 就绪检查 | ready 有值：`${passed}`，unit 为 `/4 项`；readyFailed：`未就绪`；其余：`—` | 有失败项：`{ text: "未通过：" + 失败项名称（用、连接）, tone: "down" }`；全部通过：`{ text: "数据库、迁移、检查点、运行记录", tone: "none" }` |
| 专注预测分数 | `predictionView?.display ?? (predictionFailed ? "不可用" : "—")` | 沿用原 sub 文案，tone 为 none |
| 预测模式 | 沿用原来的 mode 中文映射；缺值为 `—` | `{ text: `运行记录 ${totalRuns} 条`, tone: "none" }` |

- `passed` 是以下 4 项中通过的个数：
  - 数据库：`database.status === "ok" && database.integrity_ok`
  - 迁移：`migration.applied`
  - 检查点：`checkpoint_store === "available"`
  - 运行记录：`run_store === "available"`
- 删除 `.kpi-row` / `.stat-card` 的使用，以及 `fontSize: 22` 之类的内联样式。

**4. 其余**
- 运行记录卡头的「刷新记录」：`btn btn-secondary btn-sm`。
- 分页按钮：`btn btn-secondary btn-sm`。
- 行内 run id 按钮：保持 `btn-ghost btn-sm`。
- 内联 fontSize 按 8.0 映射；全页的 `--` 改为 `—`。

### 8.8 干预中心 `Intervention.tsx`（实现者 A）

**跳过提示卡**（约第 277 行，用的是内联 borderLeft）
- 改为 `div.notice.mb16`，删除全部内联样式。
- 标题用 `<strong>`，正文 13 号。

**触发卡 `.intervention-trigger`**
- 三个按钮：温和提醒 `btn btn-secondary btn-sm`；标准干预 `btn btn-primary btn-sm`；严格干预 `btn btn-danger btn-sm`。
- 说明文字 `p` 用 13 text-dim。

**最新干预卡**
- 按钮：接受 `btn btn-secondary btn-sm`；忽略 `btn btn-ghost btn-sm`；关闭 `btn btn-danger btn-sm`。
- `div.proof.rationale-latest` 保持上边框分组。

**干预历史卡**
- 卡头改为 `SectionHeader title="干预历史" actions={<Segmented ariaLabel="历史范围" options=[7,14,30,90].map(d => ({ value: String(d), label: `${d} 天` })) value={String(days)} onChange={v => setDays(Number(v))} />}`；删除 `div.tabs` 和原来的 h3。
- 每条记录：内联的 `padding: "14px 0"` 和 borderBottom 改为类 `.intervention-history-row { padding: 12px 0; border-top: 1px solid var(--line) }`，第一条不要上边框。
- 状态圆点：从 10×10 改为 8×8，颜色为 `var(--ok)` / `var(--drift)`，分别用类 `.intervention-dot--ok` / `.intervention-dot--warn`。
- 反馈编辑区 `.intervention-feedback-editor`：去掉背景、padding、圆角，改为 `margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--line)`。
- select 的 `marginBottom: 8` 改为类；textarea 按 8.0 处理。
- 按钮：提交 `btn btn-primary btn-sm`；取消 `btn btn-ghost btn-sm`；评价保持 `btn btn-ghost btn-sm intervention-history-evaluate`。

badge 映射不变。

### 8.9 干预执行 `Execution.tsx`（实现者 A）

**分区与筛选**
- 顶部 `div.tabs` 改为 `div.page-toolbar > Segmented ariaLabel="执行功能"`，options 为 任务管理 / 网站拦截（value 沿用 `tab` 现有取值，label 不变）。
- 状态筛选的 `div.tabs` 改为 `Segmented ariaLabel="任务状态"`，options 为 全部 / 待办 / 进行中 / 已完成（value 沿用 `statusFilter` 现值），`margin-bottom: 16px`。

**按钮**

| 按钮 | 样式 |
|---|---|
| 添加任务 / 保存任务 | `btn btn-primary btn-sm` |
| 取消编辑 | `btn btn-secondary btn-sm` |
| 编辑 | `btn btn-ghost btn-sm` |
| 开始 / 完成 / 重开 | `btn btn-secondary btn-sm` |
| 删除 | `btn btn-danger btn-sm` |
| 添加拦截 | `btn btn-primary btn-sm` |
| 暂停 / 恢复 | `btn btn-secondary btn-sm` |
| 移除 | `btn btn-danger btn-sm` |

**表单、列表、标题**
- `.workspace-form`：字段网格 `gap: 16px`；`.workspace-field` 的 label 为 13 text-dim，`gap: 6px`。
- 记录列表按 8.0 的 `.workspace-record` 处理。
- 任务标题：内联的 14/600 和 textDecoration 移到类 `.task-title`；已完成状态用 `.task-title--done`（加删除线，颜色 text-dim）。

### 8.10 专家面板 `Panel.tsx`（实现者 A）

**顶部操作**
- 改为 `div.page-toolbar`：运行专家面板用 `btn btn-primary`，查看上次结果用 `btn btn-secondary`。这两个是页面级按钮，高 32px。

**`.meta-strip`**
- 改为 `MetricStrip`（不带 label），`margin-bottom: 24px`。items：
  - 不确定度：显示原值文本。
  - 是否升级复核：`是` / `否`。
  - 模型调用：value 为数字，unit `次`。
  - 讨论轮次：value 为数字，unit `条`。
- 缺值显示 `—`。

**降级警告卡**（约第 222 行）
- 改为 `div.notice.mb16`。
- 「重试 AI 分析」用 `btn btn-secondary btn-sm`，放在 `.notice-action` 里。

**置信度 badge**
- 删除内联 background/color，按现有 confidenceColor 的三档分别映射为 `badge-success` / `badge-warning` / `badge-danger`。

**`.opinion-card`**（primitives.css）
- 改为 `border: 0; background: transparent; border-radius: 0; border-left: 3px solid var(--opinion-color); padding: 4px 0 4px 12px`。
- `.opinion-list` 的 `gap: 16px`。

其余内联样式按 8.0 处理。

### 8.11 AI 对话 `Chat.tsx` + `chat.css`（实现者 A）

- 新对话按钮：`btn btn-secondary btn-sm`，保留 100% 宽。
- 3 个开场建议按钮：`btn btn-secondary btn-sm`。
- 发送按钮：`btn btn-primary`，删除 `height: 42`。输入区为 `display: flex; gap: 8px; align-items: flex-end`。
- 删除对话框：取消用 `btn btn-secondary`，确认删除用 `btn btn-danger`；`.chat-session-dialog h2` 为 16/600。
- 内联样式按 8.0 处理。

### 8.12 模型中心 `ModelCenter.tsx` + `model-center.css`（实现者 A）

**演示提示**
- `info-box.model-demo-notice` 改为 `div.notice.model-demo-notice.mb24`，文案保留。

**分区切换**
- `div.tabs[role=tablist]` 改为 `div.page-toolbar > Segmented ariaLabel="模型中心分区"`。
- options：数据准备 / 个人基线 / 模型训练 / 模型状态（value 沿用 `activeTab` 现值）。

**第一行 4 个 stat-card**（第 369 行）
- 改为 `MetricStrip`（不带 label）。delta 用原来的 sub 文本，tone 为 none。
  - 原始事件：value 用 `toLocaleString()`。
  - 行为特征窗口：unit `个`。
  - 匹配合格窗口：unit `个`。
  - 人工标注：unit `条`。

**第二行 3 个 stat-card**（第 401 行：可训练性 / 可评估性 / 基线就绪）
- 改为 `div.card.mc-section`，h3 为「就绪情况」，内含 3 条 `.kv-row.kv-row--text`。
- 左列：名称；原来的 sub 说明放在下一行，12 text-dim，放进左列的 span 里并设 `display: block`。
- 右列：badge，文字为原 value 文本；原来是 `sub.good` 的用 `badge-success`，`sub.bad` 的用 `badge-warning`。

**`.mc-gate` 与 `.mc-blocker`**
- `.mc-gate`：去掉边框和背景，改为 `display: grid; grid-template-columns: 1fr auto; padding: 12px 0; border-top: 1px solid var(--line)`；第一条不要上边框。
  - `.gate-label`：14/500。
  - `.gate-detail`：12 text-dim。
  - badge 放在右列。
- `.mc-blocker`：同上处理。
- `.mc-baseline-empty`：13 text-dim。

**按钮与字号**
- 开始训练：`btn btn-primary btn-sm`；取消任务：`btn btn-danger btn-sm`。
- `.nf-title`：48 → `--fs-2xl`。
- `.nf-desc`：18 → `--fs-md`。

### 8.13 外壳与工作区组件（实现者 A）

**`WorkspacePage`**
- 改为渲染 `<div className={`workspace-page ${className}`} aria-label={title}><p className="page-lede">{description}</p>{children}</div>`，不再渲染 PageHeader。
- 改完后若 PageHeader 已没有其他 import，删除 `PageHeader.tsx` 以及 primitives.css 中的 `.page-header*` 规则（孤儿清理）。

**A 另外负责**
- §7.5 侧栏图标。
- §7.4 仪表盘内部统一。
- 8.0 全局控件。
- 8.1 共享类。

**login.css**
- 只做字号映射：`.login-btn` 15 → 14，`.login-title` 26 → 28。不改外观。

### 8.14 e2e 迁移表（B 完成后由 A 统一修改；必须保留原断言的意图）

| 旧选择器 / 期望 | 新写法 |
|---|---|
| `.f-state .state-item`（按标题 filter）`.state-item-number` | `.f-state .sig-metric`（按标题 filter hasText）`.sig-metric-value` |
| 专注统计 `toHaveText("10m")` / `"0m"` | `toHaveText("10分钟")` / `"0分钟"`（value 与 unit 的 textContent 之间没有空格） |
| 会话行 `toContainText("2h 0m")` | `toContainText("120 分钟")` |
| density 中 `.state-item` 高度 ≥145 | `expect(page.locator(".f-state .sig-metric")).toHaveCount(4)` |
| `.zhuanzhu-number` 文本为 N | `page.locator('.f-statics .mf-chart-bar[data-series="focus"]').first()` 的 `data-value` 等于 `"N"` |
| `.statics-chart-item` 计数 | `.f-statics .mf-chart-tick--x` 计数 |
| `.chartbox-fenxin, .chartbox-zhuanzhu` 高度为 `[0,0]` | `.f-statics .mf-chart-bar` 的高度全部为 0（包括一个元素都没有的情况） |
| `chartSlots()` 的 date / week | week 改为从 `.mf-chart-tick--x` 的文本读取；date 若已没有可见载体，删除该子断言并在汇报中列出 |
| `.f-statics-title` / `.f-conversation-title` | `page.getByRole("heading", { name: "近 7 天专注与分心" })` / `{ name: "专注会话" }` |
| `.f-reset` | 保留（类名仍在） |
| `.daily-report` / `.weekly-report` | `page.getByRole("button", { name: "日报", exact: true })` / `"周报"` |
| `.t-state .state-item-number` | `.t-state .sig-metric-value` |
| `.t-state` 不含 `"2h"` | 改为不含旧日报总专注分钟数的文本，具体值按该测试 fixture 改写（例如 `"120"`） |
| `.chart-box-item` 计数 | `.chart-box .mf-chart-tick--x` 计数 |
| 零分钟桶高度（filter `"0m"`） | `.chart-box .mf-chart-bar[data-value="0"]` 数量为 7，高度全部为 0 |
| `.chart-box-item-number` / `-column` | 同上，改用 data-value |
| `select.time` 宽度 ≥140；`.time-box select` 的操作 | 断言 `page.getByRole("group", { name: "时间范围" })` 可见；切换天数用 `getByRole("button", { name: "近 30 天" })` 等 |
| `.third-box-item` / `.third-item-content` | `.third-box .sig-metric` / `.third-box .sig-metric-value` |
| `button.tab`（按文本 filter）、`[role="tab"]`、`getByRole("tab", …)` | `page.getByRole("button", { name, exact: true })`；`[role="tab"]` 的 `.nth(i)` 改为 `.sig-seg-item` 的 `.nth(i)` |
| `.stat-card` | `.sig-metric` |
| density 中 `.telemetry-stats > div` 与按钮逐列对齐 | 改为：`.telemetry-stats > .kv-row` 数量为 5；`.telemetry-actions > button` 数量为 5；1440 宽时按钮高度全部相等、top 之差小于 1 |
| workspace-review 的 `/ai/runs` mock `{…, total: 45, limit, offset}` | 改为后端的真实形状 `{ items, count: 45, has_more, next_offset }` |
| `.workspace-heading` 计数为 0 | 保留，另加 `.page-header` 计数为 0 |

`playwright.config.ts` 下的 test-all-* 和 full-e2e 也按上表修改选择器。本轮不运行它们（它们需要真实后端），但 `npm run lint` 必须通过。

### 8.15 分工与门禁

**实现者 B（新子代理）**
- 只能改这些文件：Focus.tsx、focus.css、Reports.tsx、reports.css、report-view.ts、tests/report-state-mapping.test.ts、Analytics.tsx、analytics.css、Activities.tsx、activities.css。
- layout-refinements.css 中，只能改**仅选择** `.f-page`、`.activities`、`.analytics`、`.reports` 子元素的规则；共享的 padding 规则不要动。
- 不跑 Playwright。
- 门禁：
  - `npm run build`：0 错误、0 警告。
  - `npx tsx tests/report-state-mapping.test.ts`：通过。
  - `npm run lint`：没有新增告警。

**实现者 A（fe-signal）**
- 负责 B 范围以外的全部改动，以及 8.14 的 e2e 修改。
- 等 B 汇报完成后再改 e2e，然后跑全量门禁：
  - build：0 错误、0 警告。
  - `check:api-drift`。
  - `npm run lint`。
  - ui-review 全部通过。基线为 240 passed / 15 skipped，数量有变化要逐条说明。
  - 用 `MF_REVIEW_SHOTS=1` 截图。
  - 375 宽度下没有横向溢出。

**A 和 B 都要遵守**
- 不提交。
- 遇到本 spec 没覆盖的情况，或与代码现状冲突时，停下来汇报，不要即兴设计。

### 8.16 面向用户的简化（用户 2026-11-10 追加要求："给用户用的时候，不要有太复杂的页面，例如给他写json文件什么的"）

验收原则：普通用户可见的页面里，不出现手写 JSON、正则、数字优先级、原始枚举值或内部术语。实现者为 B（`Settings.tsx` 与 `settings` 相关 CSS），e2e 由 A 修改。

**1. 删除"高级偏好设置 · JSON 编辑"**
- 依据：主代理已核实，`/preferences` 是自由格式存储，后端唯一读取的键是 `telemetry.activity_retention_days`，而它已由隐私卡里的保留天数下拉框编辑。
- 删除整个 `<details className="workspace-details">`（summary 为"高级偏好设置 · JSON 编辑"）。
- 同时删除 Settings.tsx 中因此成为孤儿的代码：
  - state：`preferences`、`prefLoading`；
  - 函数：`isPreferences`、`parsePreferencesText`、`handlePutPrefs`、`handlePatchPrefs`；
  - 初始加载中的 `getPreferences()` 调用及其结果处理（注意 `Promise.allSettled` 结果按下标取值，要同步调整下标）；
  - 对 `getPreferences` / `putPreferences` / `patchPreferences` 的 import；
  - `.settings-prefs-textarea` CSS。
- `api.ts` 里的 3 个函数保留：API 客户端与后端契约一一对应，不属于本次的孤儿。

**2. 分区改名**
- `数据与偏好` → `数据导出`。`SETTINGS_SECTIONS` 与渲染条件同步修改。

**3. 应用分类表单**
- 可见字段只保留三项：
  - 应用：input，`aria-label="进程名"` 不变；placeholder 改为 `应用进程名，例如 chrome.exe`。
  - 分类：select，只列 `USER_SOFT_CLASSIFICATION_OPTIONS`；删除"高级分类(按需)"这个 optgroup。
  - 按钮：`添加`。
- "窗口标题"条件折叠为表单下方的 `<details className="settings-rule-more">`：
  - summary 为 `更多条件（可选）`，13 号 text-dim，无边框盒子。
  - 内含 input，`aria-label="窗口标题包含"`，placeholder 为 `窗口标题包含的文字，例如 GitHub`。
  - 下方加 12 号 text-dim 说明：`只在窗口标题包含这段文字时才按此分类，不区分大小写。`
  - 依据：后端用 `re.escape` 后做不区分大小写的 search，普通文字即为"包含"匹配，用户无需输入通配符。
- 删除优先级数字输入。提交时自动设置 priority：窗口标题条件非空（trim 后）为 `10`，否则为 `0`。
  - 依据：后端按 priority DESC 匹配；这样带标题条件的具体规则自动优先于同一应用的通用规则。
  - 提交成功后的重置值保持 `priority: 0`。
- 未分类应用的提示文案改为：`点击应用名可直接填入下方表单，再选择分类即可。`
- 已有规则列表的展示不变（历史规则可能是高级分类，原样显示）。

**4. 保留天数文案**
- `输入桶保留天数` → `键盘鼠标活动统计保留天数`（label 与 aria-label 同步改）。
- `活动与浏览器片段保留天数` → `应用与网页记录保留天数`（label 与 aria-label 同步改）。

**5. e2e（A）**
- workspace-review.spec.ts:77-79：改为点击 `数据导出`，并断言 `page.getByText("高级偏好设置")` 数量为 0、`page.getByLabel("偏好设置 JSON")` 数量为 0（锁定"无 JSON 编辑"这一要求）。
- full-e2e.spec.ts:536：`input[placeholder='窗口标题模式']` 改为 `page.getByLabel("窗口标题包含")`，操作前先展开 `更多条件（可选）`。
- full-e2e.spec.ts:561、579-590：同 workspace-review 的处理，删除完整保存 / 局部更新两段。
- test-all-buttons.spec.ts:595（分区表中的 `数据与偏好`）、727-747：同上。

### 8.17 名词提示与文案纠错（用户 2026-11-10 追加要求："如果某些按钮有专有名词，可以弄一个小的提示栏，告诉别人这个东西是干嘛的，启用后会怎么样。总之，还是把用户当成啥也不会的孩子"）

- 实现者：B（可修改下列所有 src 文件，包括原属 A 的文件）。e2e 由 A 在 B 完成后修改。
- 所有文案均由主代理依据代码真实行为撰写（来源：2026-11-10 名词清点），一律逐字使用。

#### 8.17.1 `HelpTip` 组件（新建 `src/components/signal/HelpTip.tsx`；样式追加到 signal.css）

```tsx
interface HelpTipProps { term: string; what: string; effect?: string }
```

**结构**
```html
<span class="help-tip">
  <button type="button" class="help-tip-trigger" aria-label="{term}是什么？" aria-expanded={open} aria-describedby={open ? popId : undefined}>?</button>
</span>
```
- 打开时用 `createPortal` 挂到 `document.body`：`<div role="tooltip" id={popId} class="help-tip-pop" style={{ top, left }}>`。
- 内容为 `<p class="help-tip-what">{what}</p>`；有 effect 时再加 `<p class="help-tip-effect">{effect}</p>`。

**交互**
- 打开：
  - `mouseenter` 打开；`mouseleave` 时若未钉住则关闭。
  - `focus` 打开；`blur` 时若未钉住则关闭。
  - `click` 切换"钉住"状态（触屏靠点击）。
- 关闭：
  - 按 `Escape` 关闭并取消钉住。
  - 钉住时，在 `document` 上 `mousedown`，目标不在 trigger 或 pop 内即关闭。
  - `window` 发生 `scroll`（capture）或 `resize` 时关闭。
- trigger 的 click 先调用 `e.preventDefault(); e.stopPropagation();`：它可能位于 `<label>` 内，避免误切换复选框。

**定位**（打开后在 `useLayoutEffect` 中计算，position 为 fixed）
- `w = pop.offsetWidth`，`h = pop.offsetHeight`，`r = trigger.getBoundingClientRect()`。
- `left = clamp(r.left + r.width/2 - w/2, 8, innerWidth - w - 8)`。
- `top = r.bottom + 6`；若 `top + h > innerHeight - 8`，则改为 `top = r.top - 6 - h`。

**CSS**
```css
.help-tip { display: inline-flex; vertical-align: middle; margin-left: 4px; }
.help-tip-trigger { position: relative; width: 16px; height: 16px; padding: 0; border: 1px solid var(--line-strong); border-radius: 50%; background: transparent; color: var(--text-dim); font-size: var(--fs-xs); font-weight: 600; line-height: 1; cursor: help; display: inline-flex; align-items: center; justify-content: center; }
.help-tip-trigger::before { content: ""; position: absolute; inset: -4px; }  /* 命中区 24px */
.help-tip-trigger:hover, .help-tip-trigger[aria-expanded="true"] { color: var(--signal); border-color: var(--signal); }
.help-tip-trigger:focus-visible { outline: 2px solid var(--signal); outline-offset: 2px; }
.help-tip-pop { position: fixed; z-index: 1000; width: min(280px, calc(100vw - 16px)); padding: 10px 12px; background: var(--chart-tooltip-bg); border: 1px solid var(--chart-tooltip-border); border-radius: var(--radius-sm); color: var(--text); font-size: var(--fs-sm); font-weight: 400; line-height: 1.5; text-align: left; white-space: normal; }
.help-tip-what, .help-tip-effect { margin: 0; }
.help-tip-effect { margin-top: 6px; color: var(--text-dim); }
```

**接入方式**
- `MetricStrip` 的 item 增加可选字段 `help?: { what: string; effect?: string }`，渲染在 label 文字之后，term 取 label。
- `SectionHeader` 增加可选 prop `help?: { what: string; effect?: string }`，渲染在标题文字之后（标题元素内部），term 取 title。
- 其他位置直接写 `<HelpTip term what effect />`。
- 放置规则：
  - 跟在名词文字之后。
  - 按钮的提示放在按钮**后面**作为兄弟元素，绝不放进 button 内部。
  - 一组按钮（例如三档强度）只放一个提示，位于整组之后。

#### 8.17.2 提示文案（term ｜ what ｜ effect；"—"表示不传 effect）

**全局**

| 位置 | term | what | effect |
|---|---|---|---|
| StatusChip 采集器按钮之后 | 采集 | MindFlow 每隔几秒记下你正在用哪个软件、窗口叫什么。数据只存在这台电脑上。 | 停止后不再记录，报告、预测和提醒都会缺这段时间的数据。重启 MindFlow 后会自动重新开始。 |
| StatusChip "云端 AI" 行 | 云端 AI | 联网的大模型（例如 DeepSeek），用来写分析和回答问题。 | 已配置时，你的行为汇总（不含窗口标题）会发给它，可能产生少量费用。未配置时只用本机规则，不联网。 |
| StatusChip "预测模型" 行 | 预测模型 | 根据你的使用习惯，猜你此刻是专注还是分心。 | 演示模型和影子模式只用来展示，判断不算数；个人模型就绪后才会用来提醒你。 |
| FeedbackPrompt 标题"模型校准"之后 | 模型校准 | 系统不确定你刚才在不在专注，所以来问问你。 | 选"专注"或"分心"会教它更了解你；选"跳过"什么都不会记录。 |

**仪表盘**

| 位置 | term | what | effect |
|---|---|---|---|
| 状态词（专注中 / 状态摇摆…）之后 | 专注概率 | 大数字是系统猜你此刻在专注的可能性，越高越可能在专注。 | 它只是估计，可能猜错。猜错时，在弹出的"模型校准"里告诉它就好。 |
| "查看完整依据"之后 | 依据 | 展开后能看到哪些习惯让系统觉得你专注、哪些让它觉得你分心。 | — |
| "自动提醒"小标题之后 | 自动提醒 | 系统发现你可能分心时，主动弹出的小提示。 | 暂停后，这段时间不弹提醒，每天自动生成的 AI 分析也会暂停；记录照常进行，时间到了自动恢复。 |
| MetricStrip 专注时长 | 专注时长 | 连续使用 5 分钟以上、而且很少来回切换软件的时间，算作专注。 | 系统只看切换次数，不看内容，所以长时间看视频也可能被算成专注。 |
| MetricStrip **会话次数**（原"专注次数"，改名） | 会话次数 | 一段连续使用电脑 5 分钟以上的时间叫一次会话，不管专注还是分心都算。 | — |
| MetricStrip 平均专注分 | 平均专注分 | 每段会话按"主要软件占的比例"和"切换多不多"打分，满分 100，这里是平均值。 | — |
| MetricStrip 分心比率 | 分心比率 | 分心时间 ÷（专注时间 + 分心时间）。既不专注也不分心的时间不算在内。 | — |

7 日图 tooltip 中的"专注次数"同步改为"会话次数"。

**专注分析**

| 位置 | term | what | effect |
|---|---|---|---|
| MetricStrip 平均评分 | 平均评分 | 当天所有会话得分的平均值，满分 100。 | — |
| SectionHeader 专注会话 | 专注会话 | 会话是一段连续 5 分钟以上的电脑使用。系统会给每段打分，并判断是专注还是分心。 | 你可以在每段下面告诉系统真实情况，帮它学得更准。 |
| "保存反馈"按钮之后 | 反馈 | 把你对这段时间的真实感受告诉系统。 | 保存后会用来训练你的个人模型；再次保存会覆盖之前的选择。 |

纠错：`.score` 的 专注/分心 改为取后端 `session.session_type`：

| session_type | 显示 | 颜色 |
|---|---|---|
| `focus` | 专注 | `var(--signal)` |
| `distraction` | 分心 | `var(--drift)` |
| 其他 | 一般 | `var(--text-dim)` |

删除 `focusLike = score >= 60` 的前端判定。

**活动日志**

- "显示保留期内原始字段"改名为 `显示事件编号`，其后加提示：
  - term：事件编号
  - what：多显示一列事件类型和编号，方便排查问题。
  - effect：只影响显示，不会改动数据。

**行为洞察**

| 位置 | term | what | effect |
|---|---|---|---|
| SectionHeader 高切换时段 | 高切换时段 | 一天中你最常在不同软件之间来回切换的时段。切换多通常说明注意力容易被打断。 | — |
| SectionHeader 触发应用 Top | 触发应用 | 在被判为分心的时间里，你主要在用的软件。它们不一定是分心的原因，只是经常一起出现。 | — |
| MetricStrip label **我的日常习惯**（原"基线对比"，改名） | 我的日常习惯 | 你过去所有记录的平均水平，可以用来对比某一天是否反常。 | — |
| MetricStrip **常用应用**（原"效率应用"，改名） | 常用应用 | 你使用时间最长的 5 个软件。 | — |
| MetricStrip 平均专注块 | 平均专注块 | 每次连续专注平均能坚持多久。 | — |
| "运行归因分析"按钮之后 | 归因分析 | 让 AI 根据今天的使用记录，分析你可能为什么拖延，并给出一个具体的小方法。 | 会把今天的行为汇总（不含窗口标题）发给云端 AI，可能需要几分钟、产生少量费用；同一天再点会直接显示已有结果。 |
| kv 拖延类型 | 拖延类型 | AI 判断你今天的拖延最可能属于哪一种，比如"任务厌恶"就是因为不喜欢这件事而一直拖着。 | — |
| kv 置信度 | 置信度 | AI 对这个判断有多大把握，0 到 1，越接近 1 越有把握。 | — |
| kv 认知行为技术（CBT） | 认知行为技术 | 心理学里常用的改变习惯的方法。这里推荐一个最适合你当前情况的小技巧。 | — |
| SectionHeader 个人模型与云端 AI | 个人模型与云端 AI | 个人模型在你的电脑上学习你的习惯；云端 AI 负责写分析文字。这里显示它们现在能不能用。 | — |
| 归因详情"类型置信度"标题之后 | 类型置信度 | 每种拖延类型的可能性，条越长越可能。 | — |
| "认知扭曲："之后 | 认知扭曲 | 一些让人更想拖延的想法，比如"必须做到完美才开始"。 | — |
| "动机分解（TMT）"标题之后 | 动机分解 | 用一个心理学公式估计你现在有多想做这件事：觉得能做成（期望）、觉得值得（价值）会提高动力；容易被别的东西吸引（冲动）、截止日期还远（延迟）会降低动力。 | 这是根据使用行为估算的，不是心理测试。 |
| "行为偏离"标题之后 | 行为偏离 | 把今天这个时段和你平时同一时段相比，看看有没有明显不一样。 | — |
| "综合偏离分"之后 | 综合偏离分 | 数字越大，说明和平时越不一样。数据少的时候，系统会更多参考一般人的水平，避免误判。 | — |
| 表头"稳健 z"之后 | 偏离程度 | 这一项和你平时差了多少：正数是比平时高，负数是比平时低，超过 2 就算明显不同。 | — |
| 详细画像中"分心占比"行 | 分心占比 | 分心触发的次数 ÷ 全部活动次数，是次数比例，不是时间比例。 | — |

**报告中心**

- 纠错改名（与后端真实口径一致）：

  | 原名 | 新名 | 备注 |
  |---|---|---|
  | 日报"专注次数" | 会话次数 | 提示同仪表盘 |
  | 周报"总专注时长" | 日均专注时长 | — |
  | 周报"总专注次数" | 有记录天数 | unit 改为 `天` |

- 提示：

  | 位置 | term | what | effect |
  |---|---|---|---|
  | 专注评分 / 日均评分 | 专注评分 | 根据这一天所有使用记录打的分，满分 100：主要软件占比越高、来回切换越少，分越高。 | — |
  | SectionHeader 今日洞察 / 本周洞察 | 洞察 | 根据固定规则自动写出的小结，不调用 AI。 | — |

**干预中心**

| 位置 | term | what | effect |
|---|---|---|---|
| SectionHeader / h3 手动触发干预 | 手动触发 | 不用等系统发现，现在就让它给你一条提醒。 | — |
| 三档强度按钮整组之后 | 提醒强度 | 温和、标准、严格只改变提醒的语气和弹窗的醒目程度。 | 会读取最近 45 分钟的记录，并占用今天自动提醒的次数；如果提醒类型是"环境优化"，系统可能会自动把最让你分心的网站加入拦截名单。 |
| 接受 / 忽略 / 关闭整组之后 | 怎么回应 | 接受：这条提醒有用，你会照做。忽略：先不理它。关闭：不想看到这类提醒。 | 你的选择会让系统学习，以后多给有用的提醒、少给没用的。 |
| "为什么是这条提醒"之后 | 为什么是这条 | 系统会记录每种提醒对你有没有用，再挑一条最可能帮到你的。 | — |
| "候选提醒的实测表现"之后 | 实测表现 | 每种提醒过去对你有效的比例。横线越长表示越不确定，系统会偶尔试一试这些提醒。 | — |
| "评价"按钮之后 | 评价 | 告诉系统这条提醒是有用、一般还是打扰。 | 评价比"接受/忽略"更重要，会直接决定以后多不多用这类提醒；每条只能评价一次。 |

- 纠错：
  - 严格干预按钮改为 `btn btn-secondary btn-sm`（红色只用于删除类操作），覆盖 8.8 的规定。
  - 方法名改为中文：`Thompson 采样` → `按效果挑选`，`规则选择` → `按规则挑选`。

**专家面板**

| 位置 | term | what | effect |
|---|---|---|---|
| "运行专家面板"按钮之后 | 专家面板 | 几位 AI"专家"分别分析你今天的情况，互相讨论后给出结论。 | 需要联网调用云端 AI 多次，可能要几分钟并产生费用；今天已经有结果时会直接显示，不会重新运行。 |
| MetricStrip 不确定度 | 不确定度 | 专家们对结论有多不确定，0 到 1，越接近 0 越有把握。 | — |
| MetricStrip 是否升级复核 | 升级复核 | 专家意见分歧大时，会多加一轮复查。"是"表示这次复查过。 | — |
| MetricStrip 模型调用 | 模型调用 | 这次分析一共请求了几次 AI，次数越多花费越多。 | — |
| MetricStrip **发言条数**（原"讨论轮次"，改名，unit `条`） | 发言条数 | 专家们在讨论中一共说了多少段话。 | — |
| 降级提示中"降级模式"之后 | 降级模式 | 完整的专家讨论没能完成，系统改用更简单的方法给出结果，参考价值会低一些。 | — |
| "重试 AI 分析"按钮之后 | 重试 | 重新运行一次专家讨论。 | 只有上次是降级结果时才会重新运行，会再次产生 AI 费用；如果又失败，会保留原来的结果。 |

**AI 对话**

- "降级模式"之后加提示：
  - term：降级模式
  - what：云端 AI 暂时不可用，这条回复由本机的简单规则生成。
  - effect：—
- "工具:"改为 `AI 做了这些事：`。工具名映射：

  | 原工具名 | 显示 |
  |---|---|
  | query_evidence | 查看行为记录 |
  | get_latest_analysis | 读取最近的分析 |
  | run_panel | 运行专家面板 |
  | query_interventions | 查看提醒记录 |
  | 其他 | 其他工具 |

- 输入框下方加一行 12 号 text-dim：`消息会发送给云端 AI 处理。`

**系统设置**

| 位置 | term | what | effect |
|---|---|---|---|
| 鼠标与键盘聚合统计 开关标题之后 | 键鼠统计 | 每 30 秒记一次你点了几下鼠标、按了几次键盘、滚动了多少，不记录按了哪些键。 | 开启后马上开始统计，帮助更准确地判断专注；关闭后停止统计，已经记下的数据不会自动删除。 |
| 浏览器域名统计 开关标题之后 | 浏览器域名统计 | 记录你在浏览器里正在看哪个网站（只记域名，比如 bilibili.com）。需要先安装并配对 MindFlow 浏览器扩展。 | 开启后能看出网页上的分心情况；无痕窗口永远不会被记录。 |
| 键盘鼠标活动统计保留天数 | 保留天数 | 键盘鼠标统计数据保存多少天。 | 改短后，超过天数的旧数据会在下次清理时被永久删除。 |
| 应用与网页记录保留天数 | 保留天数 | 软件、窗口和网页记录保存多少天。 | 改短后，更早的记录会在每天凌晨 3 点被永久删除，报告和模型训练也会少掉这些数据。 |
| "生成浏览器配对码"按钮之后 | 配对码 | 把 MindFlow 浏览器扩展和这台电脑上的 MindFlow 连起来的临时密码，5 分钟内有效。 | 生成后会自动开启"浏览器域名统计"。配对成功后，扩展会上报你正在看的网站，并帮你拦截名单里的网站。 |
| "清除输入数据"之后 | 清除输入数据 | 删除所有键盘鼠标统计。 | 同时会删除由它算出的行为特征，以及你在"模型校准"里的回答。删除后无法恢复。 |
| "清除浏览器数据"之后 | 清除浏览器数据 | 删除所有网页记录。 | 同时会解除浏览器扩展的配对，并删除行为特征和"模型校准"里的回答。删除后无法恢复。 |
| "清除反馈数据"之后 | 清除反馈数据 | 删除你在"专注分析"页给每段会话的反馈。 | "模型校准"里的回答不会被删除。删除后无法恢复。 |
| "清除全部行为数据"之后 | 清除全部行为数据 | 删除所有记录、报告、分析、提醒记录和训练好的个人模型。 | 聊天记录、设置和 AI Key 会保留；每天的自动备份里可能仍有旧数据。删除后无法恢复。 |
| 自主控制卡 h3 之后 | 自主模式 | 开启时，系统会自己判断什么时候提醒你，并每天自动生成一次 AI 分析。 | 暂停后这两件事都会停下，时间到了自动恢复；记录不受影响。 |
| 应用分类卡 h3 之后 | 应用分类 | 告诉系统某个软件是工作用还是娱乐用。 | 设成"工作软件"后，你在用它时系统会尽量不打扰你；分类也会用于训练个人模型。 |
| "获取未知应用"按钮之后 | 未知应用 | 列出最近用过、但还没分类的软件。 | 只是查看，不会改动任何东西。 |
| 导出按钮之后 | 导出 | 把记录下载成文件。CSV 文件可以用 Excel 打开。 | 文件里包含窗口标题等隐私信息，请妥善保管。 |
| AISettings 的 API Key 标签之后 | API Key | AI 服务给你的"钥匙"，有了它 MindFlow 才能使用云端 AI。 | 保存后需要重启 MindFlow 才生效，使用时按服务商价格计费。Key 只保存在这台电脑上。 |
| AI 用量 "估算费用"之后 | 估算费用 | 近 30 天调用云端 AI 的估算花费，实际以服务商账单为准。 | — |

系统设置纠错：

1. **清除确认框**：`handleClearTelemetry` 的确认框改为 `确定${按钮名}吗？\n${对应 effect 文案}`，例如 `确定清除输入数据吗？\n同时会删除……删除后无法恢复。`。
2. **遥测统计改名**：
   - `今日输入桶` → `今日键鼠统计条数`
   - `今日浏览器片段` → `今日网页记录条数`
3. **分类规则表**：
   - 表头 `窗口标题模式` → `窗口标题包含`。
   - 删除 `优先级` 列。
   - 分类列一律显示中文名，包括"文档 / 浏览器工作 / 沟通 / 社交"等已有映射；没有映射的显示原值。
   - 删除按钮先 `window.confirm("确定删除这条分类规则吗？")`。
4. **导出格式**：option 文案改为 `CSV（可用 Excel 打开）` / `JSON`。
5. **系统状态卡**：
   - 删除"本地规则引擎模式，零 API 成本"这一行（它由模型状态决定，与 AI 费用无关，会误导用户）。
   - 状态值 `ok` / `healthy` 显示为 `正常`，其他值显示为 `异常`。
   - `迁移` 改为 `数据库版本`，值 applied → `已是最新`，pending → `待更新`。
6. **AISettings 文案**：
   - "重启后端"统一改为"重启 MindFlow"。
   - 环境变量提示整句改为：`当前 AI 服务由启动配置决定，网页里的修改暂时不会生效。`（不出现变量名）。

**高级功能页（模型中心 / 干预执行 / AI 诊断）**

| 位置 | term | what | effect |
|---|---|---|---|
| "开始训练"按钮之后 | 训练 | 用你给过的反馈，在这台电脑上训练一个更懂你的个人模型。 | 训练只在本机进行，会占用一些电脑性能；如果新模型通过全部检查，会自动替换当前模型，之后的判断和提醒都会用它。 |
| 质量门禁标题之后 | 质量检查 | 新模型必须通过的 7 项检查，确保它比简单规则更准，才会被启用。 | — |
| 模型状态"运行模式"之后 | 运行模式 | 就绪：模型已正式启用。影子模式：模型在后台试运行，判断只用于观察，不影响提醒。仅规则引擎：还没有可用模型，用固定规则判断。 | — |
| 个人基线卡标题之后 | 个人基线 | 你平时习惯的平均水平，用来判断某一天是否反常，会随着使用自动更新。 | — |
| "添加拦截"按钮之后 | 网站拦截 | 把网站加入拦截名单，打开时浏览器扩展会挡住它。 | 需要先在系统设置里配对浏览器扩展，否则不会生效；大约 1 分钟内同步。 |

高级页纠错：
- 模型中心模型状态中，原样显示的后端英文消息（`ModelCenter.tsx` 约 :733）整行删除，原因改由 `userTerm` 显示。
- 干预执行：删除任务前先 `window.confirm("确定删除这个任务吗？")`；移除网站前先 `window.confirm("确定移除这个网站吗？")`。
- AI 诊断：状态值经 `userTerm` 翻译；耗时单位 `ms` / `s` / `m` → `毫秒` / `秒` / `分钟`。

#### 8.17.3 全局术语与登录

- **`userTerm()`**（`src/lib/user-terms.ts`）：
  - 改为只返回中文名，不再附 `（英文原值）`；无映射时返回原值。
  - 新增映射：`unavailable: "不可用"`、`completed: "已完成"`、`failed: "失败"`、`pending: "等待中"`、`ok: "正常"`、`healthy: "正常"`。
- **登录页**：`开发模式` 分隔区（含"Dev 登录（本地调试）"按钮）只在 `import.meta.env.DEV` 为真时渲染。它与"进入 MindFlow"是同一个处理函数，生产环境中只会造成困惑。

#### 8.17.4 验收

- **门禁**：build 0/0、lint 无新增、check:api-drift、`npx tsx tests/report-state-mapping.test.ts`。
- **报告**：B 列出所有改名与文案变化的旧串 → 新串，供 A 迁移 e2e。
- **A 的追加工作**：
  - 迁移 e2e 中断言旧文案的地方（如"影子模式（shadow）"、"专注次数"、"总专注时长"、"讨论轮次"、"效率应用"、"基线对比"、"显示保留期内原始字段"、"Dev 登录"、"今日输入桶"、"窗口标题模式"、"优先级"列等）。
  - 新增一条 ui-review 用例：在仪表盘聚焦第一个 `.help-tip-trigger`，断言 `role=tooltip` 可见且文本非空；按 Escape 后断言其消失；在 375 宽度下断言 tooltip 的 `getBoundingClientRect()` 在视口内。

### 8.18 登录页粒子动画（用户 2026-11-10："速度调慢，增加张力，再增加与鼠标的互动"）

实现者：B。只改 `src/components/ParticleText.tsx`（`particle.css` 不动，canvas 仍为 `pointer-events: none`）。

保留不变的部分：
- 文字采样（gap 4、字体、字号）；
- mulberry32 种子 `0x4d494e44`；
- 色相 200–240、亮度 55–80、size 1.2–3.0；
- 双层圆（光晕 + 实心）的绘制方式；
- `prefers-reduced-motion` 静态路径（收敛后停止，无任何交互）；
- resize 时重新初始化。

**1. 弹簧物理（张力来源）**

每个粒子新增 `vx`、`vy`（初值 0）和 `k`；删除 `ease`。各粒子取值：
- `k = 0.010 + random() * 0.008`：弹簧刚度。
- `floatSpeed = 0.15 + random() * 0.35`：漂浮变慢约一半。
- `floatAmp = 1.5 + random() * 2.5`。

每帧计算：
- `dtScale = clamp((now - last) / 16.667, 0, 2)`，`now` 取 rAF 时间戳。
- `time += 0.008 * dtScale`：原为 0.016，整体节奏减半。
- 目标点：`targetX = tx + sin(time*floatSpeed + phase)*floatAmp`，`targetY` 同理（沿用原 cos 公式）。
- 先 `ax = (targetX - x) * k + Fx`，`ay` 同理（`Fx`/`Fy` 为下文的鼠标力与冲击波之和）；
- 再 `vx = (vx + ax * dtScale) * FRICTION^dtScale`，`FRICTION = 0.92`；
- 再将速度限幅：`|v| ≤ 24`，即按比例缩放；
- 最后 `x += vx * dtScale`，`y` 同理。

参数为欠阻尼：被推开后会越过原位再回弹 1～2 次，形成"橡皮筋"张力感。开场时粒子从随机位置缓慢收拢，约 3～4 秒成字并带轻微回弹。

**2. 鼠标交互**

指针事件：
- 改用 `pointermove`（覆盖鼠标与触屏），记录 `mouse.x`、`mouse.y`。
- 鼠标速度 `mouse.speed` 每帧更新：`mouse.speed = mouse.speed * 0.85 + hypot(本帧位移) * 0.15`。
- 以下任一情况发生时，把 mouse 置为 `(-9999, -9999)`、`speed = 0`：
  - `window` 上 `pointerout` 且 `relatedTarget === null`；
  - 触屏 `pointerup` / `pointercancel`（`pointerType === "touch"`）。

推力与漩涡（`R = 150`）：
- 对 `d = 粒子到指针距离`，`0 < d < R` 时取 `s = (1 - d/R)^2`。
- 径向推力：`push = 1.4 * s * (1 + min(mouse.speed, 40) / 20)`，方向沿 `(dx, dy) / d`，即远离指针。移动越快，推得越远。
- 切向漩涡：`swirl = 0.8 * s`，方向 `(-dy/d, dx/d)`。这一项让粒子绕指针旋转，呼应 "flow"。

光晕：
- 每帧先在指针处画径向渐变，半径 R，中心 `rgba(110, 200, 255, 0.10)` → 边缘 0。
- 指针不在画面内时不画。

靠近高亮：
- `d < R` 的粒子亮度 `+ 20 * s`（上限 92）。
- alpha 沿用原公式。

点击冲击波：
- 触发：`window` 上 `pointerdown`，且 `e.target` 不在 `.login-card` 内（`closest(".login-card") === null`）。此时 push 一个 `{ x, y, t0: now }`，最多同时存在 3 个，多出的丢弃最早的。
- 时间与半径：`age = (now - t0) / 1000`；`age > 1.2` 时移除；波前半径 `r = age * 650`。
- 冲击力：粒子满足 `band = |d - r| < 36` 时，施加 `impulse = 5 * (1 - band/36) * (1 - age/1.2)`，方向沿 `(dx, dy) / d`，与鼠标力一起计入 `Fx`/`Fy`。
- 视觉：同时画一道波前圆环，`strokeStyle: rgba(120, 210, 255, 0.25 * (1 - age/1.2))`，`lineWidth: 1`。

**3. 张力线（被拉开的橡皮筋）**

- 位移 `disp = hypot(x - targetX, y - targetY)`。
- 每帧把满足 `8 < disp < 140` 的粒子各加一段线（`moveTo(x, y)` → `lineTo(targetX, targetY)`）到**同一个 path**，一次 `stroke`：`strokeStyle: rgba(120, 200, 255, 0.16)`，`lineWidth: 0.6`。
- 绘制顺序：张力线在粒子之前绘制。

**4. 生命周期**

- 所有监听（pointermove / pointerout / pointerup / pointercancel / pointerdown / resize）在 cleanup 中移除；rAF 照常取消。
- reduced-motion 路径不注册任何指针监听。

**5. 门禁**

- build 0/0；lint 无新增。
- 用 Playwright 打开登录页做一次手动烟测（不进入 ui-review），检查以下三点后截 1 张图供主代理查看：
  - 1440 与 375 宽度下 console 无报错；
  - 动画在运行（相隔 500ms 的两帧 canvas 像素不同）；
  - 在 `.login-card` 内点击不产生冲击波，即点击后 300ms 内与未点击时一样，没有画出圆环像素。
- 截图存放：`.playwright-mcp/audit/login-particles.png`。

### 8.19 图表尺寸与横轴标签（用户 2026-11-10 截图反馈："这个部分很拥挤"——报告中心·专注时段分布）

根因（主代理已核实）：`useElementWidth` 的 effect 依赖为 `[]`，只在首次挂载时测量。BarChart / StackedBar 在 `data.length === 0` 时先渲染空态 div（未挂 ref），数据到达后 `<figure ref>` 才出现，但 effect 不会再执行，宽度永远停在 fallback（560）。SVG 按 560 布局后被 CSS 拉伸到卡片宽度，导致字体放大约 2.5 倍、8 个横轴标签互相重叠。所有"先空后有数据"的图表都受影响。

实现者：B。

1. `src/components/charts/useElementWidth.ts` 改为 callback ref：
   - `const [node, setNode] = useState<T | null>(null)`；`const ref = useCallback((el: T | null) => setNode(el), [])`。
   - effect 依赖改为 `[node]`，其余测量逻辑不变（含 ResizeObserver 及其回退）。
   - 返回类型改为 `[(el: T | null) => void, number]`。三个调用方（BarChart、StackedBar、ContributionBar）只做 `ref={ref}`，无需其他改动；若某处读取了 `ref.current`，停下汇报。
2. 横轴标签防重叠（BarChart 与 StackedBar，SVG 标签与 htmlXLabelClass 两种路径都适用）：
   - 估算标签宽度：`textWidth(s) = Σ(字符码 ≥ 0x2E80 ? 12 : 6.6)`，即 12px 字号的近似值。
   - `labelStride = Math.max(1, Math.ceil((maxLabelWidth + 8) / slot))`，其中 `maxLabelWidth` 取所有 label 的最大估算宽度。替换原来的 `slot < 26 ? … : 1`。
3. 报告中心 `专注时段分布`：
   - 柱 label 改为短格式：`b.hour.replace(/:00/g, "").replace("~", "–") + "点"`，例如 `0–3点`、`21–24点`。
   - title（悬停标题）保留原 `b.hour`，例如 `18:00~21:00`。
   - 新增 BarChart 可选 prop `valueName?: string`（悬停框中数值行的名称，默认 `yAxisLabel || "数值"`）；报告图传 `valueName="专注时长"`。
4. 门禁：build 0/0、lint 无新增、report-state-mapping。另用 Playwright 对报告中心做截图烟测（mock 或 ui-review 的 fixture 数据，不连真后端、不用任何真实票据），宽度分别为 1440 和 375，日报下有专注数据。断言并截图：
   - 横轴标签两两 bounding box 不相交；
   - 横轴标签字号的计算值为 12px（`getComputedStyle` 或文字高度 ≤ 16px）；
   - SVG 的 `width` 属性等于容器的 `clientWidth`。
   截图存 `frontend/.playwright-mcp/audit/reports-chart-1440.png`、`reports-chart-375.png`。
5. A 随后跑 ui-review 全量回归。

### 8.20 指标条标签列与行为洞察收尾（用户 2026-11-10 截图："字体和边上的分割线太近"）

实现者：B。

1. `MetricStrip` 标签列（`signal.css` 中的 `.sig-metrics`）：
   - 标签列宽度由固定 64px 改为随文字宽度：`grid-template-columns: minmax(64px, max-content) repeat(var(--sig-metric-cols, 4), minmax(0, 1fr))`。
   - `.sig-metrics-label` 加 `padding-right: 24px; white-space: nowrap;`，使标签单行显示，且与第一条竖分割线至少相隔 24px。
   - ≤600px 的媒体查询不变，标签照旧独占一行。
2. 行为洞察 · 归因详情（`Analytics.tsx` 中的 `AttributionEvidence`）：
   - `p.distortion-line` 去掉"认知扭曲："前缀，只保留各项（用、连接）。原因：标题行已有"认知扭曲"，再写一遍是重复。
   - 卡片顶部多余的分割线：`analytics.css` 增加 `.attribution-evidence > .proof:nth-child(-n + 2) { border-top: 0; padding-top: 0; }`，在 `@media (max-width: 1023px)` 中改为只对 `:first-child` 生效，第二个 `.proof` 恢复 `border-top` 与 `padding-top: var(--space-4)`。
3. `TmtGauge.tsx`：删除英文公式行 `.mf-tmt-formula`（"Motivation = E · V / (1 + I · D)"）及其 CSS，因为卡头右侧已有中文公式。顶部 JSDoc 注释保留。
4. 门禁：
   - build 0/0；lint 无新增。
   - Playwright 截图烟测（mock 数据），宽度为 1440、1024、375，覆盖行为洞察页与仪表盘：
     - 断言每个 `.sig-metrics-label` 只占一行（高度 ≤ 26px，≤600 宽度除外）；
     - 断言标签的右边缘与下一个 `.sig-metric` 的左边框之间距离 ≥ 24px；
     - 截图存 `frontend/.playwright-mcp/audit/metrics-label-{1440,1024,375}.png`。
   - A 随后跑 ui-review 全量回归（`--workers=1`）。

### 8.21 主代理全页截图审核发现（2026-11-10，用户授权"自己所有页面截图审核，没问题就上传并合并"）

实现者：B（含一处后端改动）。A 随后做回归。

1. **全局限流过紧（后端）**：
   - 现状：`RateLimitMiddleware` 全局桶为 100 次/分钟。正常快速切换页面（仪表盘轮询 + 每页 5～15 个请求）即会触发 429，行为洞察页因此显示四遍"请求过于频繁"。答辩现场快速演示时同样有风险。
   - 修改：默认值改为 `global_capacity=300.0`、`global_refill_rate=300.0/60.0`，即 300 次/分钟，并同步修改模块与类 docstring 中的"100 req/min"。
   - 不变：各端点桶（chat、attribution、train、panel 等）与每日上限保持不变，高成本接口仍受原保护。
   - 先写测试：在 `tests/test_middleware_ratelimit.py` 新增 `test_rate_limit_default_global_capacity_is_300`，断言不传参数构造的中间件全局桶容量为 300，并且连续 150 个请求不出现 429（使用不受端点桶限制的路由）；然后改代码使其通过。
   - 门禁：跑该测试文件以及 `uv run python -m ruff check` 和 `uv run python -m mypy --strict src/mindflow/api/middleware/ratelimit.py`。

2. **错误合并去重**（`Analytics.tsx:296`）：
   - `Object.values(errors).filter(Boolean)` 先用 `new Set` 去重，再用 `；` 连接。

3. **渲染错误页面向用户**（`ErrorBoundary.tsx`）：
   - 主文案保持"页面渲染出现问题，请刷新重试"。
   - 原始 `error.message` 移入 `<details>`：summary 为 `技术详情`，12 号 text-dim，默认折叠。
   - 内联样式按 §8.0 改为类。
   - 另在 `main.tsx` 注册 `window.addEventListener("vite:preloadError", …)`。用途：版本更新后，旧页面去加载已不存在的分块时，自动刷新一次。逻辑如下：
     - 若 `sessionStorage["mf-preload-reload"]` 不存在，或距今超过 10 秒：写入当前时间戳，调用 `event.preventDefault()`，再 `location.reload()`；
     - 否则不处理，交给 ErrorBoundary 显示，以防无限刷新。

4. **仪表盘预测不可用文案**（`Dashboard.tsx` 约 572–578 行与 389–393 行）：
   - 不再显示后端原始 reason（其中含"v2 特征窗口"等内部术语），改为按 status 显示：
     - `no_data`：`最近 2 小时的使用记录还不够，暂时无法判断。多用一会儿电脑后会自动更新。`
     - `stale`：`最近一段时间没有新的使用记录，判断已暂停。`
     - `no_model`：`还没有可用的预测模型，攒够反馈后可以在模型中心训练。`
     - `schema_mismatch` / `inference_error` / 未知：`预测模型暂时无法使用，系统会先用简单规则判断。`
     - `predictionView` 为空：`还没有可用的预测模型，攒够反馈后可以在模型中心训练。`
   - "前往模型中心"按钮只在 `no_model`、`schema_mismatch`、`inference_error` 时显示，文案改为 `查看模型状态`，样式为 `btn btn-secondary btn-sm`。
   - `sourceLine` 改为：
     - `demo_only` 时为 `演示模型（模拟数据，仅供展示，不影响提醒）`；
     - `loaded` 时为 `个人模型 · ${userTerm(predictionView?.mode)}`（不变）；
     - 否则为 `规则判断`。

5. **周标签统一**（`Focus.tsx` 中的 `chartWeekLabel`）：
   - 返回 `周X`（例如 `周三`），与仪表盘一致；今天仍为 `今天`。
   - title 为 `${MM-DD} 周X`。

6. **活动日志提示位置**（`Activities.tsx`）：
   - "显示事件编号"的 HelpTip 移入该 label 文字之后（在 `<label>` 内部），使其紧贴文字，不再被 flex gap 隔开 16px。

7. **门禁**：
   - 前端 build 0/0、lint 无新增、report-state-mapping。
   - 后端见第 1 条。
   - 不做截图烟测；截图由主代理统一进行。
   - **在 B 与 A 都完成之前，任何人不得再执行 `npm run build`**：主代理截图依赖 `dist`，构建中途替换 `dist` 会导致分块加载失败。B 只跑 `npx tsc -b` 与 `vite build --outDir` 指向临时目录（`npx vite build --outDir .tmp-dist && rm -rf .tmp-dist`）来验证构建；最终 `npm run build` 由主代理执行。

### 8.22 主代理第二轮全页截图审核（2026-11-10，1440×2400 + 375×2400，`.playwright-mcp/audit/p3`）

实测：专注分析 / 活动日志 / 行为洞察 / 报告中心的 `.page-lede` 位于 x=288、y=112，其余页面位于 x=264、y=88。这几页的卡片比仪表盘、干预中心多缩进 24px，属于用户说的"大小不统一"。以下 8 条全部照做，没有待定项。

1. **页面边距（根因）**：`layout-refinements.css` 第 5 行和第 33 行给 `.f-page / .activities / .analytics / .reports` 又加了一层 padding，而 `.mf-main` 已经负责页面边距（§4.0），第 6–8 行注释里也已经记录过 `.workspace-page` 的同类错误。
   - 第 5 行改为：`:root .f-page, :root .activities, :root .analytics, :root .reports { width: 100%; max-width: none; padding: 0; margin: 0; }`
   - 删除第 30–34 行整个 `@media (max-width: 700px)` 块（含注释）。
   - 把第 1–4 行注释改为一句：`/* Page gutters belong to .mf-main (spec §4.0); page wrappers add none. */`
   - 同时检查 `focus.css / activities.css / analytics.css / reports.css` 的同名选择器有没有 padding，有就删掉。
   - 验收标准：12 个页面的 `.page-lede` 左边界都等于 `.mf-main` 的内容区左边界（1440 宽时 x=264）。
2. **空状态字号**：`primitives.css` 的 `.empty-state` 没有设字号，会继承 16px，干预中心的"最新干预"和专家面板的空状态因此比其他空状态大一号。在 `.empty-state` 中加 `font-size: var(--fs-sm); line-height: var(--leading-normal);`。
3. **工具栏里的帮助图标对齐**：`signal.css` 的 `.page-toolbar` 使用 `align-items: flex-end`。专家面板"运行专家面板"后面的 `?` 因此贴底，比按钮中线低约 5px。新增规则 `.page-toolbar > .help-tip { align-self: center; }`。
   - 如果工具栏里有带标签的字段（例如报告中心的日期框），按钮中线和字段中线本来就一致，这条规则不影响它们。
   - 在页面内容区所有 `.page-toolbar`、`.workspace-actions` 中检查 `?` 是否与相邻按钮垂直居中；`.workspace-actions` 如果是 flex 布局且 `align-items` 不是 center，就补上 `align-items: center`。
4. **模型中心**（`ModelCenter.tsx`）：
   - a) 第 376 行的 `description` 在演示模型下和第 379 行的提示条文字完全一样，重复了。`description` 统一改为非演示文案：`"先在专注分析中校准记录，再查看数据是否足够训练属于自己的行为模型。"`；提示条保持不变。
   - b) 质量门禁说明（第 461 行）改为：`未评估：训练之后才会有结果；未通过：数据或效果还不够，模型不会被启用。`。不再显示英文代码。
   - c) `lib/user-terms.ts` 增加 `calibration_better_than_rule: "校准不如规则引擎"`，这样阻塞项标题不再显示英文 key。
   - d) 后端 `services/training_readiness_service.py:262` 的提示文案 `训练报告缺少校准可用性门控结果（calibration_available）` 改为 `训练报告里没有校准结果，需要重新训练一次`。改之前先 grep `tests/` 确认没有测试断言原文案（主代理已 grep `门控结果`，0 处命中）。
5. **系统设置 · 数据采集 / 自主控制两张卡**（`Settings.tsx` 第 549–615 行）：
   - a) 两张卡结构一致：左侧依次是标题（自主控制带 HelpTip）、一行 `.card-desc` 说明、状态徽章；右侧是操作区，并与整张卡垂直居中。
   - b) 数据采集补充说明：`记录你正在使用的应用和窗口标题，用于专注分析；停止后不再记录。`（原来没有说明）。
   - c) 自主控制的小时数输入（`<input type=number>`）改为与页面其他下拉框同款的 `<select className="settings-hours">`，选项是 `1 小时 / 2 小时 / 4 小时 / 8 小时 / 24 小时`，value 为 1/2/4/8/24，默认 1，`aria-label` 不变。删掉单独的 `<span className="settings-unit">小时</span>`；`.settings-unit` 如果因此再无引用，就删掉对应 CSS。按钮文字保持"暂停"。理由：不让用户自己输入数字（§8.16）。
   - d) 操作区：`.settings-control-card .workspace-actions { display: flex; align-items: center; gap: 8px; flex-wrap: nowrap; }`。`.settings-hours` 宽度改为 `auto`，`min-width: 96px`。
   - e) 375 宽时卡片纵向排列：信息在上，操作区在下且左对齐。
6. **AI 诊断**（`Diagnostics.tsx:190`）："专注预测分数"的值是 `"--"` 时显示为 `"—"`，与全站统一（`prediction-state.ts` 不改，以免影响 focus-prediction-state 的 19 项断言）。
7. **行为洞察 · 我的日常习惯**（`Analytics.tsx:614`）：基线不存在（`baseline === null` 且没有 `baseline` 分区错误、不在加载中）时，给 `MetricStrip` 传 `note="还在积累记录：使用几天后，这里会显示你平时的平均水平。"`；有数据或出错时不传。
8. **门禁**：
   - `npx tsc -b`
   - `npx vite build --outDir .tmp-dist && rm -rf .tmp-dist`（**禁止 `npm run build`**）
   - lint 无新增
   - 后端 ruff，以及 `tests/test_training_readiness.py` 全过
   - A 回归 ui-review `--workers=1`（端口 4300）

   截图由主代理统一进行。

### 8.23 离线演示实机核验发现（2026-11-10，全新隔离目录 `mf_demo_1110b`，未配置 LLM）

实测结论：
- 仪表盘的预测与 TreeSHAP 贡献展示正常。
- 以下三处在离线（L3 规则引擎）路径下不可用或自相矛盾。答辩现场就是离线演示，所以都必须修。

1. **手动干预被拒（后端，`services/intervention_service.py`）**
   - 现象：用户点击"标准干预"后提示"本次提醒已跳过：未检测到显著的拖延模式，无需干预"。但按钮说明写的是"不用等系统发现，现在就让它给你一条提醒"。
   - 原因：`maybe_intervene` 第 1 步调用 `_select_intervention_type(assessment) is None`（最高置信度 < 0.2 且没有推荐技术）后直接跳过，手动和自动两条路径共用这一判断。
   - 改法：仅当 `bypass_throttle=True`（手动）且 `_select_intervention_type` 返回 None 时，**不跳过**。
     - `assessment.types` 非空：用 `rule_selection(assessment.types, bucket=...)` 选出主选方案，再用 `dataclasses.replace` 把 `note` 改为 `未检测到明显的拖延模式（最高可能性 {p:.0%}），按你的请求给出一条通用专注提醒`。
     - `assessment.types` 为空：构造 `Selection(intervention_type="nudge", selection="rule", context_bucket=bucket, candidates=(Candidate(nudge，先验 α/β，chosen=True),), note="当前记录里看不出拖延迹象，按你的请求给出一条通用专注提醒")`，`rationale_payload` 的 `procrastination_type` 为 None。
     - 之后照常走工作态、深度专注、安全检查和持久化等后续步骤；手动路径本来就会绕过的环节，现在依旧绕过。
   - 自动路径（`bypass_throttle=False`）的行为**不变**，仍然跳过，原 skip 文案保留。
   - 测试：
     - 新增"手动 + 低置信度 → 生成干预，note 含'按你的请求'"；
     - 新增"手动 + types 为空 → nudge"；
     - 新增"自动 + 低置信度 → 仍然跳过"；
     - 已有测试不得放宽。
2. **动机分解（TMT）在离线归因中为空（后端，先定位再修）**
   - 现象：在全新目录下 `POST` 归因（前端"运行归因分析"）返回 `tmt=null`，界面显示"暂无动机分解数据"。
   - 已知：`graph/analysis_graph.py:535-541` 会把 `bundle.tmt` 写入状态，`:983` 写入 verdict，`:402-403` 缓存命中时从 transcript 读取。
   - 要做的：用 `scripts/demo.py --data-dir <新的临时目录> --prepare-only`，或等价的 `create_demo_app` 配合 httpx ASGI 客户端复现，查清 `tmt` 在哪一步丢失。候选位置：L3 回退链节点、规则引擎 verdict 构造、transcript 持久化、`evidence_service.compute_tmt` 输入缺失返回 None、路由的非 workflow 分支。
   - 修复要求：`tmt` 和 `deviation` 都是确定性计算，**与是否调用 LLM 无关**。L3 规则引擎路径、单专家降级路径和缓存命中路径都必须返回它们（`deviation` 在没有基线时可以是 null，`tmt` 只要证据包构建成功就不能是 null）。
   - 若根因是 `compute_tmt` 在缺少任务/截止时间时返回 None：改为返回带缺失标记的分解，与 `domain/tmt.py` 中已有的缺失语义保持一致；**不得编造数值**。
   - 回归测试：离线（无 LLM）跑一次 `daily_attribution`，断言响应中 `tmt` 非空，四个分量和 `motivation` 都存在；再跑一次（缓存命中），断言 `tmt` 仍非空。
3. **归因结果自相矛盾（前端，`pages/Analytics.tsx` 归因结果卡）**
   - 现象：拖延类型显示"冲动分心"、置信度显示 0.15，而证据栏写着"未检测到显著的拖延模式"。
   - 改法：当 `cbt_technique == null` 且最高置信度 < 0.2 时（阈值与后端 `intervention_service._select_intervention_type` 一致，在常量旁注释引用该函数）：
     - "拖延类型"一行显示 `没有发现明显的拖延`；
     - "置信度"一行显示 `—`；
     - 下方"类型置信度"条形图照常显示，标题旁加一句弱提示 `都很低，仅供参考`。
     - 其他情况不变。
4. **空状态文案（前端）**
   - `Analytics.tsx:227` 的 `暂无偏离数据` 改为 `还在积累记录：用几天后，才能和你平时的水平比较。`；
   - `TmtGauge.tsx:44` 的 `暂无动机分解数据` 改为 `这次分析没有算出动机分解。`。
5. **门禁**：
   - 后端：全量 pytest（串行），通过数 ≥ 3336 + 新增测试数；ruff 和 mypy --strict 均为 0。
   - 前端：`npx tsc -b`，临时目录构建（禁止 `npm run build`），lint 不新增问题，fe-a 回归 ui-review。
   - 实机：主代理在新的隔离目录下重新截图核验。

### 8.24 桌面干预弹窗与整体风格统一（用户 2026-11-10 截图："这个需要做好看一点，和整体风格不符了"）

现状：`backend-next/src/mindflow/infrastructure/intervention_popup.py` 是浅色系统窗口，带 Tk 羽毛图标和灰色系统标题栏，三个按钮外观完全一样，标题在标题栏和正文里各出现一次。正文中的方法提示写成"（可尝试 刺激控制 方法）"，多了两处空格。

不新增依赖，只用 tkinter。颜色取自 `frontend/src/theme.css`：bg `#0a0e14`、surface `#121a24`、surface-2 `#18222e`、line `#1f2a36`、text `#d7dee6`、text-dim `#8e9baa`、signal `#33b5e5`；hover 用 signal 提亮后的 `#5cc6ec`。

1. **窗口**
   - `overrideredirect(True)` 无边框，去掉系统标题栏和羽毛图标，保持 `-topmost`。
   - 宽 400px（按 DPI 缩放，见第 7 条）。
   - 位于屏幕右下角，右边距 24px，下边距 72px，避开任务栏。
   - 外框 1px，颜色为 line；内部底色为 surface；左侧贴边一条 3px 的 signal 色竖条，与整窗等高。
   - 出现时用 `-alpha` 在约 160ms 内从 0 渐变到 0.97。
2. **顶栏**（内边距 16/20/0/20）
   - 左侧：signal 色 `●` 加文字 `MindFlow · 专注提示`，字体 Microsoft YaHei UI 9 号，颜色 text-dim。
   - 右侧：`×`，平时 text-dim，hover 时 text，光标为 hand2，点击走 `_on_close`（语义不变：ignored，human）。
   - 按住顶栏可以拖动窗口。
3. **标题与正文**
   - 标题：payload title，Microsoft YaHei UI 13 号粗体，颜色 text，上间距 10。
   - 正文：Microsoft YaHei UI 10 号，颜色 text-dim，上间距 6。
   - 两者 `wraplength` 均为 352px，左对齐。
4. **按钮**
   - 用 `tk.Label` 绑定事件实现，以便完全控制配色。整行右对齐，上间距 16，按钮间距 8，从左到右依次为：
     - `不适合我` → dismissed：无底色，文字 text-dim，hover 时文字变为 text；
     - `先忽略` → ignored：底色 surface-2，1px line 色边框，文字 text，hover 时底色变为 line；
     - `好，试试看` → accepted：底色 signal，文字 bg 色并加粗，hover 时底色变为 `#5cc6ec`。
   - 三个按钮内边距均为 14×6，光标为 hand2。
   - `_BUTTON_RESPONSES` 改为以上三个新文案的映射；`ACTION_RESPONSES` **不变**。
   - 键盘：Enter 等于"好，试试看"，Esc 走 `_on_close`。
5. **倒计时条**：窗口底部放一个高 2px、横跨全宽的 Canvas，signal 色，在 `timeout_s` 内按线性收缩到 0，每 200ms 刷新一次；超时逻辑不变。
6. **行为不变**：`post_response`、ready 文件写入时机（窗口显示后写入）、超时以 auto 身份上报、`_finish` 的幂等性都保持原样。
7. **DPI**
   - win32 下在 `tk.Tk()` 之前调用 `ctypes.windll.shcore.SetProcessDpiAwareness(1)`，只用 `contextlib.suppress(AttributeError, OSError)` 包住，不吞其他异常。
   - 所有像素量乘以 `scale = root.winfo_fpixels("1i") / 96`；字体用磅值，不再缩放。
8. **正文文案**（`services/intervention_service.py:495`）：`（可尝试 {cbt_label} 方法）` 改为 `（方法：{cbt_label}）`。此项由 be-fix 做，避免和它正在改的文件冲突。
9. **测试**
   - 已有测试全部保持通过；
   - 新增 `_BUTTON_RESPONSES == {"好，试试看": "accepted", "先忽略": "ignored", "不适合我": "dismissed"}`；
   - 新增一个纯函数（例如 `_countdown_fraction(elapsed, timeout)`），并为它的边界写单测。
10. **预览**
    - 在仓库外写一个预览脚本，用示例 payload 启动弹窗：标题 `该回到正题了`，正文用截图里那段，`api_url` 指向一个不可达的本地端口，`timeout_s=30`。
    - 窗口就绪后用 Pillow 的 `ImageGrab` 截下窗口区域，保存到 `D:/大学相关/01_学业与课程/07_双创/MindFlow/.playwright-mcp/audit/popup/popup-preview.png`，截完立即关闭窗口。
    - 预览脚本本身不进仓库。
11. **点击必须立即关闭**（用户反馈 2026-11-10："弹窗点击无响应，应该能立刻关掉"）
    - 根因：`_finish` 先同步执行 `post_response`（超时 5 秒），之后才 `destroy()`。服务端慢或已退出时，窗口会在原地"卡住"最多 5 秒。
    - 改法：`_finish` 依次做以下几步：
      1. 置 `_responded`；
      2. 取消全部 after 任务；
      3. 立即 `root.withdraw()` 并调用 `root.update_idletasks()`，使窗口在一帧内消失；
      4. 再上报回答，`post_response` 的超时由 5.0 改为 2.0（仅弹窗调用处传参，函数默认值不改）；
      5. 最后 `destroy()`。
    - 窗口显示后调用 `root.focus_force()`，确保 Enter / Esc 能直接生效。
    - 测试：用假 opener 记录调用顺序，断言 `withdraw` 早于 `post_response`；断言 opener 收到的 timeout 为 2.0。
    - 实测：在探针中对四个可点击元素（三个按钮和 ×）分别用 `ctypes` 的 `SendInput` 发送**真实鼠标点击**，确认每次点击后 ≤100ms 窗口即不可见，且上报的回答映射正确。

### 8.25 稳健个人基线：收缩强度改为由数据估计（后端算法，2026-11-10 E1 实验暴露）

证据：`scripts/experiment_midterm_algorithms.py`。E1 在 24 个合成用户上的结果：固定 `k=10`、先验为全局中位数的收缩估计，整体 MSE 比原始桶均值**差 12.6%**（95% CI [−22.3%, −2.9%]）。其中 idle_ratio 与 active_seconds_ratio 分别差 59% 和 51%；只有 app_switch_count 好 12.1%。

原因：
- (1) Efron–Morris 经验贝叶斯的收缩强度应由组间方差与组内方差的比值从数据中估计，而当前实现把它写成了常数；
- (2) 先验忽略了一天之内的时段结构。

两处都是实现与理论不符，不是调参问题。

改法（`backend-next/src/mindflow/domain/baseline.py`）：
1. **层次先验**：桶 (h, d) 的先验均值改为同一小时 h 跨所有星期的合并均值 μ_h，合并权重为各桶的 n。
   - 若该小时合并样本数 < 3，退回现有的全局稳健中位数。
   - 先验方差同理，使用小时合并方差；不足时退回全局 σ²。
2. **数据驱动的 k（矩估计）**：对每个特征，在所有 n_b ≥ 1 的桶上计算：
   - 组内方差 σ²_w：各桶 M2 之和 ÷ (Σn_b − 桶数)，分母 ≤ 0 时取全局方差；
   - 组间方差 τ² = max(0, 加权方差(mean_b − μ_h(b)) − 平均(σ²_w / n_b))；
   - k̂ = σ²_w / τ²，截断到 [0.5, 50]；τ² ≈ 0（< 1e-12）时 k̂ = 50。
   
   结果按特征缓存，`update()` 后失效。
3. `shrunk_expected(..., k: int | None = None)`：
   - `None` 时使用 k̂ 与层次先验；
   - **显式传入 k 时，行为与现在逐字节一致**（现有测试不改）。
   - 返回字典新增 `"k"` 和 `"prior"`（取值 `"hour_pooled"` 或 `"global_robust"`）。
   - `robust_z` 和所有生产调用方都走 `None` 这条路径。
   - `DEFAULT_SHRINKAGE_K` 保留，仅作为显式参数的含义说明；注释写明默认路径已改为数据估计，并引用 Efron & Morris (1975) 与 Morris (1983)。
4. **序列化**：k̂ 和 μ_h 都可以从已序列化的桶统计重新算出，`to_dict` / `from_dict` 格式**不变**。
5. **测试**：
   - (a) 显式 k 时与旧实现一致（已有测试保持通过）；
   - (b) 构造组间差异很大、组内很小的数据，断言 k̂ ≤ 1 且估计贴近桶均值；
   - (c) 构造各桶同分布的数据，断言 k̂ 取到上限 50，估计贴近小时合并均值；
   - (d) 小时合并样本不足时退回全局先验；
   - (e) 冷启动时 n_b = 0 返回先验。
6. **确认性实验（防止对开发数据过拟合）**：
   - 实验脚本的 E1 同时报告三组：`raw`、`eb_fixed_k10`（旧实现，显式 k=10 加全局先验）、`eb_data_k`（新的默认路径）。
   - 开发种子与旧 E1 相同；另加**一组与之完全不相交的确认种子**（用户数相同），两组分别报告。
   - 汇报材料只引用确认种子的结果；结果不利时如实报告，**不再修改算法**。
7. **门禁**：
   - 全量 pytest 串行运行，ruff 和 mypy --strict 均为 0；
   - 重跑 `scripts/experiment_midterm_algorithms.py`，跑全量而不是 `--quick`。

### 8.26 可解释性诚实度与措辞统一（主代理离线实机截图 2026-11-10，`mf_demo_1110c`）

1. **TMT 不得把默认值说成发现**（`frontend/src/components/charts/TmtGauge.tsx`）
   - 现象：四个分量都是 0.50，而且都是缺少数据时的先验。期望分量的 `raw=null`、`source="先验（无近14天自评）"`，界面却写着"主要瓶颈：E 期望 · 成功预期是主要瓶颈"，并给期望行描了边。
   - 判定规则：某分量 `raw == null` 时视为"默认值"。若 TS 类型里没有 `raw`，先在 `types/` 里按后端契约补上；后端 `domain/tmt.py` 的 `TmtComponent` 本来就有 `raw` 字段。
   - 默认值分量：标签后加一个弱提示小标签 `默认值`；条形透明度降到 0.45。
   - 主导分量为默认值时：瓶颈说明行改成 `数据还不够，暂不判断主要瓶颈`，去掉"瓶颈"标签和描边。
   - 四个分量全为默认值时：动机等级标签显示 `数据不足`，代替"动机中等"，数值照常显示。
   - 图例补一句：`默认值 = 还没有你的相关记录，先按中间值 0.5 计算`。
2. **"为什么是这条提醒"的证据行**（后端 `services/intervention_policy.py::selection_evidence`）
   - 拖延类型改用中文标签。在本模块新增 `PROCRASTINATION_LABELS_ZH = {task_aversion: 任务厌恶, impulsivity: 冲动分心, decisional: 决策拖延, perfectionism: 完美主义, emotional_regulation: 情绪调节困难}`，与 `frontend/src/lib/user-terms.ts` 保持一致。services 层不能 import agents 层，所以这里单独定义一份。
   - `上下文：` 改为 `情境：`。
   - 所选候选行：
     - α、β 都等于先验 `PRIOR_ALPHA` / `PRIOR_BETA` 时：`选择{label}：还没有你对这类提醒的反馈，先按默认估计 50%`；
     - 其他情况：`选择{label}：根据你过去的反馈，估计有用的概率 {mean:.0%}`。
   - 备选行：`备选{label}：估计有用的概率 {mean:.0%}`。
   - 文案中不再出现 α、β 和"后验"。图形 `PosteriorBars` 保留不动。
   - 已有测试断言旧字符串的，按新文案更新，并在报告里逐条列出。
3. **时段名称统一**：`frontend/src/pages/Intervention.tsx:47` 中 `evening: "傍晚"` 改为 `"晚间"`，`night: "夜间"` 改为 `"深夜"`，与后端 `bucket_label` 一致。
4. **网页响应按钮与桌面弹窗一致**（`Intervention.tsx` 约 364–389 行与第 534 行）
   - 按钮文字改为 `好，试试看` / `先忽略` / `不适合我`，映射分别为 accepted / ignored / dismissed，保持不变。
   - HelpTip 改为：`好，试试看：这条提醒有用，你会照做。先忽略：现在先不理它。不适合我：这类提醒对你没用，以后会少用。`
   - 状态徽章：accepted `已采纳`，ignored `已忽略`，dismissed `不适合`。
   - 第 534 行的 "接受/忽略" 改为 "好，试试看 / 先忽略"。
5. **门禁**
   - 前端：tsc、临时目录构建（禁止 `npm run build`）、lint 不新增问题。
   - 后端：`tests/test_intervention_policy.py`、`tests/test_intervention_service.py`、`tests/test_routes_intervention.py` 全部通过；ruff 和 mypy --strict 均为 0。
   - 完成后由 fe-a 做 e2e 回归。
