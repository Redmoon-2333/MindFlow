# MindFlow Phase 0 团队操作手册

> 最后更新: 2026-05-11

---

## 一、项目当前状态速览

```
✅ 后端: 全部完成 (Config → Models → Collector → Analyzer → API)
✅ ML:  基线建模 + 偏离检测 + 聚类 + HMM + TitleAnalyzer
✅ 测试: 66 个用例全部通过
✅ 采集器: 可运行，记录窗口标题、进程名、空闲状态
❌ 前端: 未搭建（张皓、杨智杰负责）
❌ 端到端验证: 未做
```

---

## 二、环境准备（所有人）

### 2.1 Conda 环境

```bash
conda create -n mindflow python=3.11 -y
conda activate mindflow
cd mindflow-app/backend
pip install -r requirements.txt
```

### 2.2 验证安装

```bash
# 确认依赖正确
python -c "from mindflow.main import app; print('OK')"

# 跑一遍测试
python -m pytest tests/ -v
# 应该看到 66 passed
```

### 2.3 前端环境（张皓、杨智杰）

```bash
cd mindflow-app/frontend
npm install       # 等胡淙煜初始化前端脚手架后执行
npm run dev       # 启动开发服务器 → http://localhost:5173
```

---

## 三、胡淙煜 — 后端 & ML 操作指南

### 3.1 启动后端开发服务器

```bash
conda activate mindflow
cd mindflow-app/backend
uvicorn mindflow.main:app --reload --host 127.0.0.1 --port 8765
```

访问:
- API 文档: http://localhost:8765/docs
- 健康检查: http://localhost:8765/api/v1/status

### 3.2 启动数据采集器（开始积累个人数据）

```bash
# 方式1: 通过 API 启动
curl -X POST http://localhost:8765/api/v1/collector/start

# 方式2: Python 脚本
python -c "
from mindflow.collector.scheduler import collector
collector.start()
print('Collector running:', collector.is_running)
# 让它后台运行，数据写入 data/mindflow.db
"
```

**建议从今天开始就让采集器跑着**，积累 3-5 天后就有足够的个人数据建立基线。

### 3.3 运行 ML 训练管线（合成数据验证）

```bash
conda activate mindflow
cd mindflow-app/backend
python -m mindflow.analyzer.train
```

输出包括:
- 14 天合成数据的特征提取
- 个人基线建模（7 天基线 → 7 天测试）
- 偏离检测（标注异常时段 + 偏离度排序）
- 聚类结果（5 个行为模式：deep_focus / shallow_work / browsing / procrastination / idle）
- HMM 转移矩阵
- LLM 上下文 JSON 样本
- 模型保存到 `data/models/`

### 3.4 用自己采集的真实数据训练（3-5 天后）

```python
# 脚本: backend/scripts/build_baseline.py (待创建)
# 从 SQLite 读取原始 ActivityLog → 特征提取 → 更新基线 → 保存
```

### 3.5 当前 ML 模块架构速查

```
analyzer/
├── title_analyzer.py    # 窗口标题 → URL域名/文件后缀/会议关键词（零规则维护）
├── baseline.py          # 个人基线模型：每时段均值+方差，Welford在线更新
├── deviation.py         # 偏离检测：多维度z-score，异常排序
├── context_packer.py    # LLM上下文打包：基线对比+异常证据→JSON (~500 tokens)
├── data_pipeline.py     # 特征提取：14维特征/30min窗口 + TitleAnalyzer集成
├── features.py          # 专注分数计算、应用排名、切换频率
├── patterns.py          # 专注会话识别、日报生成
├── ml_models.py         # 聚类(DBSCAN)、分类器(RandomForest)、HMM
├── labeling.py          # 多信号共识标注（已降级为辅助工具）
└── train.py             # 训练入口脚本
```

### 3.6 待办事项（按优先级）

| 优先级 | 任务 | 说明 |
|--------|------|------|
| **P0** | 初始化前端脚手架 | `npm create vite@latest frontend -- --template react-ts`，配置 Ant Design + recharts + proxy |
| **P0** | 持续运行采集器 | 积累 3-5 天个人数据 |
| **P1** | 编写 `scripts/build_baseline.py` | 从 SQLite 读取真实数据 → 构建基线 |
| **P1** | API 新增端点 `GET /api/v1/report/daily` | 调用 DeviationDetector + LLMContextPacker，返回结构化报告 |
| **P2** | 完善测试 | 补充 `test_context_packer.py`，提升覆盖率到 85%+ |
| **P2** | 集成 LLM 调用 | 实现 `llm/attribution.py`：接收 LLMContextPacker 输出，调 API，返回归因 |

---

## 四、张皓 — 前端操作指南

### 4.1 等待脚手架初始化

等胡淙煜用 Vite 初始化前端项目后，你会得到:

```
frontend/
├── src/
│   ├── App.tsx
│   ├── main.tsx
│   ├── components/    ← 你来写
│   ├── hooks/         ← 你来写
│   └── types/         ← 你来写
├── package.json
├── vite.config.ts     ← 已配置 proxy → localhost:8765
└── tsconfig.json
```

### 4.2 需要实现的组件（参考 design-spec.md §5）

| 组件 | 功能 | API 数据来源 |
|------|------|-------------|
| `Dashboard.tsx` | 主页面布局，组合各组件 | — |
| `StatusCard.tsx` | 实时专注状态 + 当前窗口 | `GET /api/v1/activities/current` + WebSocket |
| `AppUsageChart.tsx` | 今日应用使用饼图 | `GET /api/v1/activities/today` → `top_apps` |
| `FocusTrendChart.tsx` | 专注趋势折线图（7天） | `GET /api/v1/focus/trend?days=7` |
| `SettingsPanel.tsx` | 采集开关 + 偏好设置 | `POST /api/v1/collector/start\|stop` + `PUT /api/v1/preferences` |

### 4.3 需要实现的 Hooks

| Hook | 功能 |
|------|------|
| `useApi.ts` | 封装 fetch 请求，统一处理 `{code, data, message}` 响应格式 |
| `useWebSocket.ts` | 连接 `ws://localhost:8765/ws/activities`，接收实时推送 |

### 4.4 技术要点

- **UI 库**: Ant Design 5.x（`npm install antd @ant-design/icons`）
- **图表**: recharts（`npm install recharts`）
- **API 代理**: Vite proxy 已配置，前端请求 `/api/*` 自动转发到 `http://localhost:8765`
- **WebSocket**: 后端每 2 秒推送一次当前窗口信息，格式如下:
  ```json
  {"type": "activity_update", "data": {"window": {...}, "collector_running": true, "timestamp": 1715335200}}
  ```

### 4.5 开发流程

```bash
cd mindflow-app/frontend
npm run dev
# 浏览器打开 http://localhost:5173
# 确保后端也在运行（http://localhost:8765）
```

---

## 五、杨智杰 — 前端 + 数据辅助 操作指南

### 5.1 前端任务

和张皓协作完成前端组件开发。建议分工:
- 杨智杰: `StatusCard.tsx` + `SettingsPanel.tsx` + `useWebSocket.ts`
- 张皓: `Dashboard.tsx` + `AppUsageChart.tsx` + `FocusTrendChart.tsx` + `useApi.ts`

### 5.2 数据清洗任务

ManicTime 和 AWT 数据集需要预处理才能被模型使用:

**ManicTime 格式转换**（`data/datasets/manictime/44个CSV`）:
```
输入: Focus-in \t Focus-out \t Duration \t AppName
输出: timestamp, process_name, window_title(空), duration_seconds, is_idle(0)
```

**AWT 标注验证**（`data/datasets/awt-labelled/`）:
- 用 TitleAnalyzer 分析 AWT 的窗口标题
- 对比 TitleAnalyzer 输出（is_code_editor, is_document, url_domain）与人工标签（Activity 列）的一致性
- 统计准确率，输出混淆矩阵

### 5.3 数据采集辅助

- 自己也运行采集器，增加数据多样性
- 手动记录 2-3 天的"分心事件"（何时开始刷手机/看视频），作为偏离检测的地面真值

---

## 六、端到端验证清单（全员）

以下各项全部打勾才算 Phase 0 Demo 就绪:

- [ ] 后端启动: `uvicorn mindflow.main:app` 无报错，`/docs` 可访问
- [ ] 采集器运行: `POST /api/v1/collector/start` → 数据库 `activity_logs` 表有新记录
- [ ] API 返回正确: `GET /api/v1/activities/current` 返回当前窗口信息
- [ ] WebSocket 推送: 前端连接后收到实时数据
- [ ] 前端 Dashboard: 显示专注得分 + 应用饼图 + 趋势图
- [ ] 设置页: 采集开关可用，偏好可保存
- [ ] ML 基线建模: `train.py` 跑通，`baseline_user1.json` 生成
- [ ] 偏离检测: 测试数据上检测到异常时段
- [ ] 前端构建: `npm run build` 无报错
- [ ] 全员在自己的电脑上跑通一遍

---

## 七、项目关键链接

| 资源 | 路径 |
|------|------|
| 设计规格 | `docs/design-spec.md` |
| 实施计划 | `docs/implementation-plan.md` |
| 开发指南 | `docs/development-guide.md` |
| 后端代码 | `backend/mindflow/` |
| ML 模块 | `backend/mindflow/analyzer/` |
| 测试 | `backend/tests/` |
| 数据集 | `data/datasets/` |
