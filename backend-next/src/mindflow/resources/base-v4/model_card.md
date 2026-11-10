# MindFlow 基础行为模型：可复现实验报告

训练和评测已实际执行；数据与标签均为合成，真实参与者和反馈均为 0。
指标描述合成场景下的分类能力；商业软件效果、真实人群收益另行评估。

- 版本：base_v4_20261111_r2；schema v4（28 维）。
- 总窗口：145,152；训练：59,684；独立校准：14,662。
- 未纳入监督：26,794（休息、离开、信息不足等）。
- 配置：{"seed": 20261110, "personas_per_role": 6, "days": 28, "windows_per_day": 48}。

## 固定测试集的算法对照

BA 为 Balanced accuracy。误提醒代理率指专注窗口被判为分心的比例。
实际通知还受置信度、工作态、节流等条件控制。

| 测试划分 | 方法 | 窗口数 | BA | Macro F1 | Brier↓ | 误提醒代理率↓ | 分心召回↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test_persona | rule_engine | 19507 | 0.5000 | 0.3624 | 0.2966 | 0.00% | 0.00% |
| test_persona | task_context_rule | 19507 | 0.6974 | 0.6988 | 0.1805 | 11.09% | 50.57% |
| test_persona | logistic_regression | 19507 | 0.8062 | 0.8079 | 0.1437 | 14.60% | 75.84% |
| test_persona | random_forest | 19507 | 0.8628 | 0.8692 | 0.1000 | 5.09% | 77.66% |
| test_persona | xgboost | 19507 | 0.8590 | 0.8646 | 0.0976 | 6.17% | 77.96% |
| test_persona | base_rf_xgb_calibrated | 19507 | 0.8618 | 0.8678 | 0.0971 | 5.55% | 77.90% |
| test_future | rule_engine | 19737 | 0.5000 | 0.3642 | 0.2992 | 0.00% | 0.00% |
| test_future | task_context_rule | 19737 | 0.6993 | 0.7008 | 0.1848 | 9.21% | 49.06% |
| test_future | logistic_regression | 19737 | 0.8013 | 0.8047 | 0.1438 | 12.86% | 73.13% |
| test_future | random_forest | 19737 | 0.8480 | 0.8556 | 0.1047 | 4.82% | 74.42% |
| test_future | xgboost | 19737 | 0.8469 | 0.8538 | 0.1022 | 5.77% | 75.15% |
| test_future | base_rf_xgb_calibrated | 19737 | 0.8486 | 0.8561 | 0.1028 | 5.06% | 74.79% |

## 已知困难场景（完整保留测试结果）

| 划分 | 场景 | 窗口数 | 指标 | 结果 |
| --- | --- | ---: | --- | ---: |
| test_persona | game_testing | 124 | normal_work_false_alert_rate | 61.29% |
| test_persona | customer_support | 105 | normal_work_false_alert_rate | 32.38% |
| test_persona | goal_drift | 940 | distraction_recall | 0.43% |
| test_persona | off_topic_reading | 845 | distraction_recall | 1.18% |
| test_future | game_testing | 72 | normal_work_false_alert_rate | 59.72% |
| test_future | customer_support | 73 | normal_work_false_alert_rate | 67.12% |
| test_future | goal_drift | 1084 | distraction_recall | 0.55% |
| test_future | off_topic_reading | 979 | distraction_recall | 2.55% |

相同软件及输入轨迹可能服务于不同目标；这类任务需要本人目标或纠正反馈。
合成身份留出仍共享场景配方，未见模板、真实用户与实际通知效果尚待独立验证。
类别规则只是透明启发式基线，并不代表 Rize、RescueTime 或其他产品。

## 实验解释与复算

训练、校准、未知用户测试按模拟身份隔离；未来日期测试严格晚于训练和校准的截止时刻。
测试集没有参与超参或模型选择，不能将最佳对照算法事后换成已发布模型。
各组结果、场景与角色切片、模拟身份重采样区间见[指标快照](../../../../../docs/models/base-v4-results.json)。完整窗口和逐窗口预测可按[复现说明](../../../../../docs/models/base-v4.md)重新生成。
模型只接收生产协议的 28 维特征，身份、情境名和软件名仅用于覆盖审计。
软件名称仅用于构造不同输入活动与覆盖审计；类别映射会丢失任务意图，因此相同观测仍可能对应不同真实状态。
计划休息与信息不足数据未伪装成分心或真实用户反馈。

## 使用与后续个性化

软件优先加载通过个人质量门的本地模型；否则使用标明来源的合成基础模型。
现有 RF/XGBoost 采用个人反馈重训及独立校准进行个性化，不使用语言模型式的权重微调。
个人训练沿用日期前向评估、唯一会话计数、发布一致性检查和 shadow-only 自动训练策略。
正式对照试用将进一步测量正常任务误打扰、真实偏离召回、弃权覆盖、延迟及用户负担。
