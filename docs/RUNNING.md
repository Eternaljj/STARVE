# 运行与复现

所有命令在仓库根目录执行。输出目录必须使用新名称。安装步骤见根目录 README。

## 离线入口

```bash
python -m h1_cross_system_20261023.runner --preflight-only --output outputs/h1-preflight
python -m examples.offline_h2 --output outputs/h2-demo
python -m pytest -q
```

H1 准入验证九个 source 的本地环境、状态与四条件分支。H2 演示调用当前五策略运行时，在 WEB/API/SERVICE 各一个本地任务上运行脚本响应；网络连接在演示过程中被禁止。其 Token 和模型决策属于 Mock，不作为正式研究结果。

跨模型离线准入入口：

```bash
python -m h1_gpt_cross_model_6source_20261024.runner --preflight-only
python -m h1_claude_cross_model_6source_20261024.runner --preflight-only
python -m h1_glm_5_3_flash_cross_model_6source_20261025.runner --preflight-only
```

这些模型入口还报告当前凭证准备情况。没有凭证时，必须区分本地检查通过与真实模型准入。

## 真实模型实验

对应配置保留原实验的模型标识、路由、预算和停止条件。配置不包含密钥，按所选入口从环境变量读取：

| 入口 | 凭证环境变量 |
| --- | --- |
| H1 DeepSeek / Anthropic Messages 路由 | `P0A_AUTH_TOKEN` |
| H1 GPT | `H1_P_GPT_API_KEY` |
| H1 Claude | `H1_P_CLAUDE_API_KEY` |
| H1 GLM | `H1_P_GLM_API_KEY` |
| H2 当前 DeepSeek 五策略 | `DS_V4_H1_API_KEY` |

`.env.example` 只列变量名，脚本不会自动读取它。设置变量后，H1 各模型入口通过显式 `--start` 开始新的真实实验。真实请求会消耗所选 Provider 的额度。代码保存请求日志、usage、环境效果和停止原因，并执行预算约束。

H2 正式五策略启动器位于 `starve_h2_final_dsv4_9source_20261023/runner.py`，支持 `--preflight-only`、`--start`、`--config` 和 `--output`。其中 `--preflight-only` 包含真实路由检查，不等同于发布版离线演示。该启动器保留原实验的路由证据和冻结准入要求，需提供配置引用的历史准入记录；未提供这些记录时应使用离线演示检查机制，不能直接声称正式实验可复现。

## 可选原生 Agent 会话

`p0a_pentestgpt/checkpoint.py` 和 `claude_boundary.py` 保存原生会话捕获/恢复实现。这部分依赖原有 PentestGPT 环境中的 `unified_agent`、`claude_agent_sdk` 及其匹配接口，核心 requirements 不包含它们。当前 H1 HTTP Provider、H2 本地环境和离线测试均可独立运行；只有调用原生会话功能时才需要该外部环境。

## 历史分析数据

各分析器接收本轮实验的 JSON/JSONL/CSV 记录。部分跨模型分析函数额外读取配置中的历史 `pair_ledger.json`；这些数据另行分发。随源码保留的历史比较测试在缺少该账本时明确跳过，其余主线测试继续执行。

`formal_H1_multitask_v3_2_freeze_manifest.json` 保留原协议及 source schedule，其中历史文件元数据属于原工作区记录。发布版路径整理有修改记录，不能将原元数据直接当作发布版冻结记录；新实验应保存自己的版本、时间戳、只读权限和修改记录。
