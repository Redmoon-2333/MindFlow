# MindFlow 真实数据采集指南

训练好模型后，你需要用自己的真实行为数据来训练个性化模型。

## 1. 启动采集器

采集器随 MindFlow 后端自动启动：

```bash
cd mindflow-app\backend-next
python -m mindflow.main
```

启动后，服务器在 `http://localhost:8765` 运行，采集器自动开始每 5 秒轮询当前活动窗口。

## 2. 数据库位置

所有数据存储在本地 SQLite 数据库中：

```
data\mindflow.db
```

（SQLite WAL 模式，数据完全本地，不上传云端）

## 3. 检查采集状态

```bash
# 查看已采集的记录数
sqlite3 data\mindflow.db "SELECT COUNT(*) FROM activity_events"

# 查看最近的采集时间
sqlite3 data\mindflow.db "SELECT MAX(timestamp_utc) FROM activity_events"
```

## 4. 最小数据要求

| 指标 | 最低 | 推荐 |
|------|------|------|
| 采集天数 | 7 天 | 14+ 天 |
| 日均活跃小时 | 4 小时 | 8+ 小时 |
| 需要覆盖的日期类型 | 工作日 + 周末 | 两周完整周期 |

## 5. 用真实数据训练

```bash
# 基础训练
python -m mindflow.train --source db

# 带参数
python -m mindflow.train --source db --days 30 --min-confidence 0.2
```

## 6. 验证训练结果

训练完成后查看报告：

```bash
cat data\models\training_report.json
```

关注这些指标：
- `classifier.accuracy` > 0.65 — 分类器可用
- `classifier.f1` > 0.6 — 专注于分心分类平衡
- `clustering.n_clusters` ≥ 3 — 找到了行为模式聚类
- `hmm.transition_matrix` 非均匀 — 状态转移有意义

## 7. 启动前端查看

```bash
cd frontend
npm install && npm run dev
```

浏览器打开后，你会看到基于你的行为数据生成的专注度趋势和干预建议。

## 8. 隐私说明

- ✓ 所有数据存储在本地 `data\mindflow.db`
- ✓ 采集器只记录应用名称和窗口标题分类
- ✓ 不记录键盘输入、鼠标位置、屏幕截图
- ✓ 不联网上传，无遥测

## 9. 重新训练

模型会随着数据增长而改善。建议在以下时机重新训练：
- 积累 7 天新数据后
- 行为模式发生显著变化时（如假期开始/结束）
- 感觉干预不够准确时

```bash
python -m mindflow.train --source db
```

## 10. 模型版本管理

```bash
# 查看所有已保存的模型版本
python -m mindflow.train --list-versions

# 回滚到指定版本
python -m mindflow.train --rollback 20260717
```
