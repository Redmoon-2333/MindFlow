# MindFlow 本地联调与模型发布验收

日期：2026-09-30。本文为本轮最终报告入口；此前 review、repair、completion
目录为历史记录，不用旧测试计数代替本报告。

## 发布范围

用户先要求全功能验收后推送，随后要求核实真实前后端联调、旧模型可用性，
并至少发布一版本地模型供其他电脑体验。本次发布范围是已验收的本地功能、
界面修复和明确标记的合成演示模型，不是在线 LLM 或个人模型质量的认证。

未修改正式数据库、正式模型、签名密钥、在线凭据或个人配置。
所有有写入的训练、采集、删除、配对测试都在隔离目录执行。
正式模型、数据库、原始行为明细、签名 key/HMAC、密钥暴露日志均不提交。
历史截图和大体积参考基线留在本地。

## 前后端是否真实对接

结论：本地业务调用链已对接。不是仅靠前端 mock 得到的结论：

- 使用实际 Vite 代理、一次性票据交换和 HttpOnly 会话 Cookie。
- 最新 API 矩阵 70/70 通过，6 项跳过；其中训练/采集/删除由独立真实生命周期
  验收补充，500/403 展示分支由隔离回归补充，不能改写原矩阵的跳过数。
- 9 组生命周期通过：真实采集写入、分类恢复、干预响应与反馈写后读回、
  对话历史、规则归因/面板/诊断、真实训练、配对/heartbeat/屏蔽列表、
  非空导出、四类遥测删除与配对令牌撤销。
- 训练终态为 succeeded/shadow，训练开始后取消返回 409，未将它记录为取消成功。
- WebSocket connected 由实际页面验证；离线演示还会核对 API 概率与页面百分比。

完整本地证据：`security-final-api/api-matrix.json`、`live-functional.json`、
`publication-guard.txt`、`demo-live/demo-live.json`、
`cold-python312-live/demo-live.json`。公开摘要见同目录 `release-evidence.json`；
原始日志与截图留在本机，不作为公开数据库或行为记录发布。

## 原来的个人模型

磁盘 active 指向版本 `20260827_182610_841df8`。将三份制品、指针及签名材料
复制到临时目录后，使用当前正常加载器复核：

- 当前环境 scikit-learn 1.9.0。
- 制品保存环境 scikit-learn 1.6.1。
- `load_latest()` 返回 False，原因是 InconsistentVersionWarning。
- 原制品没有被替换、重签或发布。未绕过 HMAC 或版本一致性检查。

因此旧文件仍保留，但当前环境不能直接使用。恢复路径是保留旧环境，或在
当前环境另行重训并重新经过质量门；不能把旧制品“强行兼容”当作修复完成。
证据：`old-model-compatibility.txt`。仓库旧 data 目录的指针不代表正式运行目录。

## 可公开的离线模型

`backend-next/demo-models/v4/` 含分类器、聚类模型、研究用途 HMM 和发布 manifest。
仅使用项目合成生成器：6 个 archetypes × 14 天，seed 42，v4 的 28 维特征。
三份制品合计 5,159,685 bytes，manifest SHA256：
`ae66ae8b19669e98b342cb35de8e1519d8382bc65450db8531e371a9de904199`。

`scripts/demo.py` 检查发布 manifest 的固定 digest、制品 digest、特征顺序和依赖版本；
然后在目标电脑生成自己的 key，正常 HMAC 签名后加载。篡改和不兼容依赖被拒绝。
默认独立使用 `backend-next/data/demo/`，拒绝已有非 demo 目录，重复启动不覆盖模型。
首次全新依赖安装发现 Python 3.12 会被旧锁文件分配到 SciPy 1.18/XGBoost 3.3；
已把 ML 序列化依赖显式固定为 manifest 所列版本，并重锁定到 SciPy 1.17.1/
XGBoost 3.2.0。新增依赖对齐回归，不能依靠本机已有环境隐藏新电脑的版本不匹配。
`.gitattributes` 禁止转换发布清单和模型制品的字节；关闭 `core.autocrlf` 的检出副本
仍具有相同 SHA256。正常本地登录也不再继承代理，避免回环票据请求被转发到代理。

首次启动生成合成统计和当前推理窗口。运行模式 `shadow/demo_only`，
`loaded=true`、`ready=false`，不宣称通过个人数据质量门。页面明确标记演示用途。
采集器、调度器、在线凭据默认关闭；正常的 15 分钟数据新鲜度检查保留，
过期时显示原因，重启演示刷新窗口，不继续展示旧概率。

这是行为 ML，不是本地生成式大语言模型。离线聊天、面板、归因走规则降级。
新电脑需先安装 Node/uv 和锁定依赖、构建前端；不需要采集数据或重新训练模型，
但不是零依赖的免安装可执行文件。完整步骤见根 README 和 demo-models/README。

## 质量门禁

| 检查 | 当前结果 |
| --- | --- |
| 后端全量 | 3156 passed，0 skipped，20 sklearn fixture warnings，553.94s |
| 演示包/认证定向复验 | 23 passed，23.02s；演示包自身含 9 项 |
| Ruff | src/tests/新增 helpers 全部通过 |
| 严格 Mypy | src + 新增 helpers，共 190 files，通过 |
| 前端 build | TypeScript/Vite 通过，1133 modules |
| 前端 lint | 0 errors，17 历史 vendor/E2E warnings |
| 前端契约 | 11 组通过，包括 API 漂移 3/3 |
| Playwright | 最终 219/219 通过，0 retries，9.2m |
| 响应式 | 五宽度 × 12 路由/登录画布/404/键盘，共 75/75 |
| 接口矩阵 | 70 passed，0 failed，6 skipped |
| 生命周期/发布保护 | 9/9 真实生命周期、A/B/C 训练发布保护通过 |
| 依赖安全 | npm 官方审计与 Python 审计均 0 vulnerabilities |
| 全新代码/数据联调 | 1440/390 两种视口，通过 Cookie 登录、真实 API/UI 概率均为 90.0%、WebSocket connected |
| 独立 Python 3.12 环境 | Python 3.12.12，全新依赖安装完成；23/23 演示与认证回归通过；清空 PYTHONPATH 后按 README 直接启动成功，1440/390 两种视口 API/UI 均为 89.6%，WebSocket connected |

后端 warnings 为合成小样本标签集合提示；前端 warnings 为参考字体 vendor JS
和既有 E2E unused。Git whitespace 检查有两处 EOF 空白：既有诊断测试尾行、
原样导入的字体 CSS 尾行，无功能影响；没有顺手改写历史/第三方文件。
发布 manifest 的 CRLF 是固定 digest 覆盖的原始字节，通过路径级
`whitespace=cr-at-eol` 正确识别，不为清理空白而破坏发布校验值。
全新代码/数据联调首先使用已验收的 Python 3.11 依赖环境，并将 PYTHONPATH
指向独立代码快照。随后单独完成 Python 3.12.12 冷安装与启动验证：
sklearn 1.9.0、NumPy 2.3.5、SciPy 1.17.1、joblib 1.5.3、XGBoost 3.2.0，
发布清单校验通过；不使用 PYTHONPATH workaround，直接执行 README 命令启动。
两种视口实测 API 的概率 0.896373，对应页面 89.6%，会话 Cookie 和实时连接正常。
概率会随滚动窗口变化，不能把这一次数值当成固定准确率或个人行为判断。

## 未认证的范围

- 真实付费 DeepSeek 推理：本轮真实调用数为 0；须确认旧凭据轮换和调用授权。
  先前未隔离测试发生的凭据暴露风险仍需用户处理，本轮不记录其值。
- 严格逐像素一比一：尚未完成同数据/时钟/动画条件下的几何与像素签字。
- 操作系统通知实发视觉：自动化使用 LogOnlyNotifier，通知契约测试不能替代实发。
- 个人模型预测质量及其他物理电脑/操作系统：没有用合成模型或本机临时目录替代认证。

## 交付

根 README 提供新电脑安装、构建、演示启动和一次性登录命令。
公开证据为本文和 `release-evidence.json`；完整原始日志与截图留在本机本目录。
内置浏览器在独立 Python 3.12 环境中显示实际预测、影子模型标记和 connected，
截图为 `in-app-python312-prediction.jpg`，截图中无认证票据或 root token。

发布目标为当前仓库的 `origin/main`，不是参考前端仓库。
提交采用精确文件白名单，不用 `git add .`，不强推；最终交付答复记录 GitHub
提交哈希和远端一致性，不能用本地提交成功替代远端推送验证。
