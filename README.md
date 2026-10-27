# H1/H2 Agent Resource Defense

研究本地受控 Agent 任务中的状态、资源消耗与干预机制。当前为精简源码版 `github-core-v3`（2026-10-27）：H1、H2 各保留一个共用入口，模型与接口地址通过统一配置切换。

主要代码：

| 入口 | 用途 |
| --- | --- |
| `h1/runner.py` | 自然轨迹采集、三个状态 checkpoint、BASELINE 与三个 Trap 的配对执行，局部预算 K |
| `h1/analyze.py` | 本轮配对记录、资源指标和描述性分析 |
| `h2/runner.py`、`h2/runtime.py` | 完整轨迹的五策略比较、部署资格、状态优先级与预算 B |
| `adapters/llm_provider.py` | H1/H2 共用的模型配置与 Messages API 接入 |
| `tasks/`、`traps/`、`adapters/` | 本地任务环境、三种 Trap、状态观察与运行适配 |
| `p0a_pentestgpt/gateway.py` | 工具参数、执行权限和效果记录 |


| Trap | 实现文件 | 运行适配 |
| --- | --- | --- |
| BRANCH_NEW_V3 | `traps/task_relevant_branch_v3.py` | `adapters/lifecycle_branch_new_v3_runtime.py` |
| DEPENDENCY_V6 | `traps/dependency_v6.py` | `adapters/lifecycle_dependency_v6_runtime.py` |
| EVIDENCE_NEW_V3 | `traps/evidence_v3.py` | `adapters/evidence_new_v3_runtime.py` |

H2 五策略为 NO_TRAP、FIXED_BRANCH_NEW_V3、FIXED_DEPENDENCY_V6、FIXED_EVIDENCE_NEW_V3、STATE_AWARE_FEASIBLE。旧 Trap 的实现、启动入口及选择器已删除。

## 安装与离线运行

使用 Linux、Python 3.11 及以上版本，在本目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m h1.runner --preflight-only --output outputs/h1-preflight
python -m h2.runner --preflight-only --output outputs/h2-preflight
```

每次使用新的输出目录。离线检查不请求真实模型；H2 检查覆盖三个任务族、五种策略，共 15 条脚本响应轨迹，结果明确标记为 Mock。独立示例入口为 `python -m examples.offline_h2 --output outputs/h2-demo`。

## 模型配置

修改 `configs/provider.json`，或通过环境变量设置 `LLM_MODEL`、`LLM_BASE_URL`、`LLM_API_KEY`。H1/H2 使用同一份配置；当前内置传输支持 Anthropic Messages 兼容接口，地址以 `/v1/messages` 拼接。配置中的示例模型与域名需要替换为实际路由。

真实模型执行需要显式启动：

```bash
python -m h1.runner --one-source-preflight 1 --output outputs/h1-live-check
python -m h1.runner --start --output outputs/h1-live
python -m h2.runner --start --source-index 1 --budget 20 --output outputs/h2-live
```

H2 省略 `--source-index` 时运行全部九个选定任务；每个任务分别运行五种策略。真实执行会消耗所配置模型的额度。其他接口协议需提供对应适配器。

运行与分析见 [运行说明](docs/RUNNING.md)，代码阅读顺序见 [代码导览](docs/CODE_GUIDE.md)。
