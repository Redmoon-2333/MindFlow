# 基线质量门（改造前，commit 26d5f24，分支 feat/midterm-upgrade，2026-11-09）

环境修复：backend-next/.venv/pyvenv.cfg 的 home 指向已不存在的 C:\Users\lenovo\anaconda3\envs\mindflow，改为 D:\Users\lenovo\anaconda3\envs\mindflow（同为 3.11.15，原文件备份 pyvenv.cfg.bak）；frontend node_modules 缺 react-markdown/remark-gfm，按 lockfile `npm install` 补齐（lockfile 无实质变化，已还原）。
pytest-xdist 在中文路径下 execnet worker 启动失败，改串行运行。

| 门禁 | 结果 |
|---|---|
| pytest（串行） | 3166 passed, 3 failed, 4 skipped, 20 warnings（339s） |
| ruff | All checks passed |
| mypy --strict | Success: no issues found in 186 source files |
| 前端 build (tsc -b && vite build) | 通过 |
| api-schema-drift | 3 tests passed |

3 个失败归类：
- test_baseline_repository::test_create_then_update_keeps_one_row_and_creation_time、test_concurrency_gate::test_queue_latency_is_measured_separately_from_http_latency：单独重跑通过，时间敏感型偶发。
- test_ecnu_service::test_selected_ecnu_without_key_does_not_borrow_direct_or_legacy_key：本机设置了 ECNU_API_KEY/CHATECNU_* 环境变量，测试未隔离环境变量；去掉这些变量后 3/3 通过。属测试非密封问题，非产品缺陷（修复：测试内 monkeypatch.delenv）。
