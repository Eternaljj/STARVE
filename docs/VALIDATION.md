# 发布版验证记录

版本：`github-source-v1`。日期：2026-10-26（Asia/Shanghai）。

环境：Linux、CPython 3.14.7、`requirements-dev.txt` 所列依赖。验证从独立发布目录执行，没有请求真实模型。

| 检查 | 结果 |
| --- | --- |
| `python -m pytest -q` | 89 passed、1 skipped；另有 24 个 subtest 通过 |
| H1 `--preflight-only` | passed=true，Provider 请求数为 0 |
| 当前 H2 五策略离线演示 | passed=true；3 个任务族 × 5 个策略，共 15 条轨迹 |
| H2 演示状态覆盖 | 每条轨迹均观测到三个正式状态 |
| 发布目录凭证检查 | 未发现常见格式的真实密钥 |

跳过项为 GPT 分析代码对原 DeepSeek 历史配对账本的形状检查，原因是历史结果账本单独分发。该检查不属于源码离线运行的前置条件。

本记录验证发布版的本地运行与核心接线。真实 Provider 路由、模型响应、计费和正式实验结果需在相应环境中另行验证。
