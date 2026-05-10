# MindFlow 后端技术扫盲文档

> 面向全体组员，零基础可读。看完应该能理解后端每个模块在做什么、数据怎么流转、以及你自己该怎么用。

---

## 一、采集器：它是怎么"看到"你在做什么的

### 1.1 原理

采集器通过 Windows API 获取当前活动窗口的信息，每 5 秒采样一次。核心代码在 `collector/tracker.py`。

```
你的操作                          Windows系统                    MindFlow采集器
────────                         ──────────                   ─────────────
点击 VS Code 窗口  ──────→  系统标记该窗口为"前台窗口"  ──→  win32gui.GetForegroundWindow()
                                                               ↓
                                                         获取窗口句柄(hwnd)
                                                               ↓
打字 "main.py"    ──────→  窗口标题更新为"main.py - VSCode" ──→ win32gui.GetWindowText(hwnd)
                                                               ↓
                                                         获取进程名: "Code.exe"
                                                               ↓
                                                         写入 SQLite 数据库
```

**技术细节：**

- `win32gui.GetForegroundWindow()` — Windows API，返回当前用户正在交互的窗口的唯一标识（句柄）
- `win32gui.GetWindowText(hwnd)` — 获取窗口标题栏文字。这就是为什么能在数据里看到 "main.py - Visual Studio Code"
- `win32process.GetWindowThreadProcessId(hwnd)` — 获取该窗口所属的进程 ID
- `psutil.Process(pid).name()` — 将进程 ID 转换为人类可读的进程名，如 "Code.exe"、"chrome.exe"

### 1.2 空闲检测

只监控窗口切换还不够——用户可能离开电脑但窗口还开着。采集器通过检测"最后一次键盘/鼠标操作"来判断用户是否空闲。

```
ctypes.windll.user32.GetLastInputInfo()  →  获取距离上次输入过了多少毫秒
                                           ↓
                               如果 > 60秒(idle_threshold) → 标记为"空闲"
```

这个机制保证了：即使 VS Code 一直开着，如果你去吃饭了，采集器会知道你在"空闲"而非"专注工作"。

### 1.3 采集调度

`collector/scheduler.py` 中的 `CollectorScheduler` 使用 APScheduler（Python 定时任务库）每 5 秒触发一次采集：

```
时间轴:  ──[采]──[采]──[采]──[采]──[采]──[采]──
         0s   5s   10s  15s  20s  25s  30s  ...
```

每次采集称为一个 "tick"。每个 tick 执行：
1. 检测用户是否空闲
2. 获取当前活动窗口信息
3. 写入一条 ActivityLog 记录到数据库

### 1.4 它需要一直开着吗

**是的。** 采集器是一个后台进程，需要在你使用电脑期间持续运行。

建议的启动方式：
```bash
# 开发阶段：开一个终端窗口，让它在后台跑
uvicorn mindflow.main:app --host 127.0.0.1 --port 8765
# 然后通过 API 启动采集：
curl -X POST http://localhost:8765/api/v1/collector/start

# 或者 Python 直接启动：
python -c "
from mindflow.collector.scheduler import collector
import time
collector.start()
print('采集器已启动，按 Ctrl+C 停止')
while True:
    time.sleep(60)
    print('.', end='', flush=True)
"
```

**未来（Phase 1）会改为系统托盘图标**，开机自启，右键菜单控制。Demo 阶段手动启动即可。

### 1.5 隐私

所有数据都存在你本地的 SQLite 文件里，**不上传任何服务器**。窗口标题会原样记录（包括浏览器 URL），但这些数据只在你自己的电脑上。

---

## 二、数据结构：一条记录长什么样

### 2.1 原始采集记录（ActivityLog）

采集器每 5 秒写入一条这样的记录：

```json
{
  "id": 1523,
  "user_id": 1,
  "timestamp": "2026-05-11T14:30:05",
  "process_name": "Code.exe",
  "window_title": "main.py - MindFlow - Visual Studio Code",
  "window_class": "VSCodium",
  "duration_seconds": 5.0,
  "is_idle": 0
}
```

| 字段 | 含义 | 示例 |
|------|------|------|
| `process_name` | 进程名（可执行文件名） | "Code.exe", "chrome.exe", "WeChat.exe" |
| `window_title` | 窗口标题栏文字 | "main.py - VSCode", "bilibili.com - Chrome" |
| `window_class` | Windows 窗口类名（较少使用） | "VSCodium", "Chrome_WidgetWin_1" |
| `duration_seconds` | 距上次采样的间隔 | 5.0（每5秒一条，所以几乎总是5） |
| `is_idle` | 用户是否空闲 | 0=活跃，1=离开 |

### 2.2 数据库 ER 关系

```
User(用户)
  │
  ├──1:N──→ ActivityLog(活动记录)      ← 原始数据，每5秒一条
  │
  ├──1:N──→ FocusSession(专注会话)     ← 分析产出，连续使用同一应用≥30分钟
  │
  └──1:N──→ DailyReport(日报)         ← 每日聚合统计
```

### 2.3 日报（DailyReport）结构

```json
{
  "date": "2026-05-11",
  "total_focus_minutes": 180.0,
  "total_distraction_minutes": 45.0,
  "focus_score": 72.5,
  "top_apps": [
    {"app": "Code.exe", "minutes": 120},
    {"app": "chrome.exe", "minutes": 60},
    {"app": "WeChat.exe", "minutes": 15}
  ],
  "switch_frequency": 8.5
}
```

`focus_score` 计算逻辑：同一应用连续使用越久、切换越少，分数越高（0-100）。

---

## 三、后端架构全景

### 3.1 模块依赖关系

```
config.py  ←  全局配置（数据库路径、采集频率等）
    ↓
models/    ←  SQLAlchemy ORM 模型（定义表结构）
    ↓
collector/ ←  数据采集（依赖 config + models 存数据）
    ↓
analyzer/  ←  行为分析（读 ActivityLog，产出 FocusSession/DailyReport）
    ↓
api/       ←  FastAPI 端点（对外提供数据）
```

这个顺序很重要：**每个模块只依赖它上面的模块，不反向依赖。** 采集器不知道分析器的存在，分析器不知道 API 的存在。

### 3.2 启动流程

当 `uvicorn mindflow.main:app` 启动时：

```
1. FastAPI 创建 app 实例
2. 执行 lifespan 启动回调:
   ├── init_db() → 自动创建 SQLite 表（如果不存在）
   └── 注册 CORS 中间件（允许前端跨域请求）
3. 注册所有 API 路由
4. 注册 WebSocket 路由
5. 开始监听 http://127.0.0.1:8765
```

采集器**不会**随后端自动启动——需要通过 API 手动触发或代码调用 `collector.start()`。

### 3.3 目录结构

```
backend/
├── mindflow/
│   ├── config.py              # 全局配置（从 .env 读取，有默认值）
│   ├── main.py                # FastAPI 入口
│   ├── logging_config.py      # 统一日志格式
│   │
│   ├── collector/             # 数据采集层
│   │   ├── tracker.py         #   Windows 窗口监控 + 空闲检测
│   │   └── scheduler.py       #   定时采集调度器
│   │
│   ├── models/                # 数据模型层
│   │   ├── database.py        #   SQLAlchemy 引擎 + 会话管理
│   │   └── schemas.py         #   ORM 模型定义（User/ActivityLog/FocusSession/DailyReport）
│   │
│   ├── analyzer/              # 行为分析层
│   │   ├── features.py        #   专注分数计算、应用排名、切换频率
│   │   ├── patterns.py        #   专注会话识别、日报生成
│   │   ├── baseline.py        #   个人行为基线（每时段均值/方差）
│   │   ├── deviation.py       #   偏离检测（多维度 z-score）
│   │   ├── context_packer.py  #   LLM 上下文打包器（基线+异常→JSON）
│   │   ├── title_analyzer.py  #   窗口标题分析（URL/文件后缀/关键词）
│   │   ├── data_pipeline.py   #   特征工程管道 + 应用分类器 + 合成数据
│   │   ├── ml_models.py       #   聚类(DBSCAN)/HMM/随机森林
│   │   ├── labeling.py        #   多信号共识标注（辅助工具）
│   │   └── train.py           #   训练入口脚本
│   │
│   ├── api/                   # API 层
│   │   ├── routes.py          #   RESTful 端点（13个）
│   │   └── websocket.py       #   WebSocket 实时推送
│   │
│   ├── llm/                   # LLM 集成（Phase 3 预留）
│   └── intervention/          # 干预策略（Phase 4 预留）
│
├── tests/                     # 测试（镜像 src 结构）
│   ├── test_tracker.py        # 采集器测试
│   ├── test_features.py       # 特征计算测试
│   ├── test_api.py            # API 端点测试（12个）
│   ├── test_baseline.py       # 基线模型测试（7个）
│   ├── test_data_pipeline.py  # 特征管道测试（8个）
│   ├── test_labeling.py       # 标注器测试（12个）
│   ├── test_scheduler.py      # 调度器测试（4个）
│   ├── test_title_analyzer.py # 标题分析测试（15个）
│   └── conftest.py            # 共享 fixtures
│
├── data/                      # 运行时数据（gitignored → 不上传）
│   ├── mindflow.db            #   SQLite 数据库
│   └── models/                #   训练好的 ML 模型
│
└── requirements.txt           # Python 依赖清单
```

---

## 四、ML 管线详解

### 4.1 数据流

```
ActivityLog (原始, 每5秒)
       │
       ▼  BehaviorFeatureExtractor (30分钟窗口聚合)
       │
特征向量 (14维, 每30分钟一个)
       │
       ├──→ BaselineModel.update()  → 更新个人基线（在线学习）
       │
       ├──→ DeviationDetector.score_window() → 计算偏离度
       │
       ├──→ BehaviorClustering.fit() → 发现行为模式簇
       │
       └──→ BehaviorHMM.fit() → 学习状态转移概率
```

### 4.2 特征向量的 14 个维度

对每 30 分钟时间窗口，计算以下特征：

**行为特征（6维）**——不依赖应用名，只看操作模式：

| 特征 | 含义 | 举例 |
|------|------|------|
| `unique_app_count` | 用了几个不同应用 | 3 |
| `switch_frequency` | 每小时切换次数 | 8.5 |
| `max_app_duration` | 最长连续使用时长（秒） | 1200 |
| `idle_ratio` | 空闲时间占比 | 0.05 |
| `hour_of_day` | 几点 | 14 |
| `day_of_week` | 周几 | 2 |

**应用分类特征（3维）**——依赖 AppClassifier 规则，Demo 用，真实数据效果有限：

| 特征 | 含义 |
|------|------|
| `productivity_ratio` | 生产力应用时间占比 |
| `entertainment_ratio` | 娱乐应用时间占比 |
| `social_ratio` | 社交应用时间占比 |

**标题特征（5维）**——从窗口标题提取，零规则维护：

| 特征 | 提取方式 | 举例 |
|------|----------|------|
| `title_code_ratio` | 检测 .py/.js/.java 等文件后缀 | "main.py - VSCode" → 命中 |
| `title_doc_ratio` | 检测 .pdf/.docx/.md 等文件后缀 | "论文.docx - Word" → 命中 |
| `title_url_ratio` | 检测 URL 域名 | "github.com/..." → 命中 |
| `title_meeting_ratio` | 检测 "zoom/teams/会议" 等关键词 | "Zoom Meeting" → 命中 |
| `title_entertainment_ratio` | 检测 "番剧/直播/steam" 等模式 | "B站 - 番剧" → 命中 |

### 4.3 个人基线模型（BaselineModel）

**核心思想：** 不定义"什么是专注"，而是学习"你平时是什么样的"，然后检测"今天跟平时有什么不同"。

使用 **Welford 在线算法** 计算均值和方差，支持增量更新——每来一个新窗口，不重读历史数据，只更新统计量：

```
新均值 = 旧均值 + (新值 - 旧均值) / n
新方差 = 旧M2 + (新值 - 旧均值) × (新值 - 新均值)
```

基线按 **24小时 × 7天** 分成 168 个桶。例如：
- (周二, 10:00-10:30): 平均值 switch_frequency=8.5, std=2.1
- (周六, 22:00-22:30): 平均值 switch_frequency=35.2, std=12.4

同一个用户，工作日白天和周末深夜的行为模式被分开建模。

### 4.4 偏离检测（DeviationDetector）

对每个新窗口，计算每个特征的 z-score（偏离均值多少个标准差）：

```
z = (当前值 - 基线均值) / 基线标准差
```

然后加权融合为"总体偏离度"：

```
总体偏离 = weighted_sum(weight_i × |z_i|) / sum(weights)
```

严重程度分级：
- `|z| < 1.5` → normal（正常波动）
- `1.5 ≤ |z| < 2.5` → mild（有点异常）
- `2.5 ≤ |z| < 4.0` → moderate（明显异常）
- `|z| ≥ 4.0` → severe（极度异常）

### 4.5 偏离检测 ≠ 分心判断

这是最关键的设计理念：

```
偏离检测说: "你现在的切换频率比你平时高了3倍"
            ↑ 客观事实，ML 可以可靠地判断

分心判断说: "高切换频率意味着你在拖延"
            ↑ 主观归因，ML 不应该判断，留给 LLM
```

### 4.6 LLM 上下文打包（ContextPacker）

当用户点击"今日报告"时，系统将 ML 分析结果打包成约 500 token 的 JSON：

```json
{
  "report_date": "2026-05-11",
  "user_profile": {
    "status": "ready",
    "days_collected": 7,
    "typical_features": {
      "switch_frequency": {"typical": 8.5, "range": "6.4-10.6"}
    }
  },
  "anomalies": [
    {
      "time": "14:00-14:30",
      "severity": "moderate",
      "overall_deviation": 2.94,
      "key_deviations": [
        {"feature": "switch_frequency", "z_score": 3.1, "direction": "up"}
      ],
      "sample_titles": ["bilibili.com - 番剧播放 - Google Chrome"]
    }
  ],
  "today_summary": {
    "focus_score": 72.5,
    "anomaly_count": 2
  }
}
```

LLM 拿到这些后，结合 CBT 理论框架，生成类似这样的归因：

> "你今天下午 2 点出现了一次注意力偏离：切换频率从平时的 8 次/时上升到 25 次/时，窗口内容从代码编辑器变成了视频网站。这可能是因为午饭后精力下降导致的被动分心。建议下次这个时段安排一些低认知负荷的任务，或者设置 25 分钟的番茄钟来重建节奏。"

---

## 五、API 接口速查

所有端点前缀：`/api/v1/`

| 方法 | 路径 | 功能 | 需要用户？ |
|------|------|------|-----------|
| GET | `/status` | 系统状态（采集器是否运行、总记录数） | 否 |
| POST | `/collector/start` | 启动采集 | 否 |
| POST | `/collector/stop` | 停止采集 | 否 |
| GET | `/activities/current` | 当前活动窗口（实时） | 否 |
| GET | `/activities/today` | 今日活动摘要 + 应用排名 + 专注分数 | 是 |
| GET | `/focus/today` | 今日专注报告（含日报） | 是 |
| GET | `/focus/trend?days=7` | N 日专注趋势 | 是 |
| GET | `/analytics/patterns` | 今日行为模式（分心时段识别） | 是 |
| GET | `/reports/weekly` | 周报 | 是 |
| GET | `/preferences` | 获取偏好设置 | 是 |
| PUT | `/preferences` | 更新偏好设置 | 是 |

WebSocket: `ws://localhost:8765/ws/activities`（每 2 秒推送当前窗口）

响应格式统一为：
```json
{
  "code": 0,
  "message": "success",
  "data": { ... },
  "timestamp": 1715335200
}
```

---

## 六、实际操作指南

### 6.1 日常开发流程

```bash
# 1. 激活环境
conda activate mindflow

# 2. 启动后端
cd mindflow-app/backend
uvicorn mindflow.main:app --reload --host 127.0.0.1 --port 8765

# 3. 另一个终端，启动采集
curl -X POST http://localhost:8765/api/v1/collector/start

# 4. 前端（如果有的话）
cd mindflow-app/frontend
npm run dev
```

### 6.2 积累个人数据

建议首先连续采集 **3-5 天**，每天在你的正常工作和学习环境中让采集器跑着。这会建立你的个人基线。

然后花 **1-2 天**故意制造一些"分心事件"——比如在某个下午刻意刷 30 分钟 B 站、深夜玩一局游戏——然后看偏离检测能否正确标记这些异常时段。

### 6.3 跑测试

```bash
cd mindflow-app/backend

# 全部测试
python -m pytest tests/ -v

# 单个文件
python -m pytest tests/test_baseline.py -v

# 单个用例
python -m pytest tests/test_features.py::test_calculate_focus_score -v
```

### 6.4 跑 ML 训练（合成数据）

```bash
cd mindflow-app/backend
python -m mindflow.analyzer.train
```

输出包括：基线统计、偏离检测结果、聚类模式、HMM 转移矩阵、LLM 上下文 JSON 样本。

---

## 七、常见问题

**Q: 采集器会影响电脑性能吗？**

A: 设计目标 CPU < 2%，内存 < 50MB。只是每 5 秒调一次 Win32 API + 写一条 SQLite 记录，相当于每 5 秒做一次极轻量的文件操作。

**Q: 我的数据安全吗？**

A: 所有数据只在你的 `data/mindflow.db` 里。这个文件在 `.gitignore` 里，不会被提交到 GitHub。代码里没有任何上传逻辑。

**Q: 窗口标题会记录浏览器 URL 吗？**

A: 会。如果你在 Chrome 里打开 `github.com/MindFlow`，标题栏会显示 `GitHub - MindFlow - Google Chrome`。这也是 TitleAnalyzer 能提取 URL 域名的原因。这些数据只在本地。

**Q: 为什么不用浏览器插件直接获取 URL？**

A: Phase 1 计划做（见 `implementation-plan.md` Task 1.1），需要为 Chrome/Firefox 分别开发扩展。目前窗口标题是零侵入的通用方案。

**Q: Mac 能用吗？**

A: 当前只支持 Windows（采集器用了 `win32gui`/`win32process`）。Phase 2 计划支持 macOS。
