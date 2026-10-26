# H1/H2 Agent Resource Defense

用于研究 Agent 在受控本地任务中的资源消耗、状态与干预机制关系。包含任务环境、状态观察、工具执行边界、三种 Trap、H1 配对实验、H2 完整轨迹实验和分析代码。

本目录是从研究工作区整理的独立源码发布版 `github-source-v1`，整理日期为 2026-10-26。运行产物与正式研究数据另行保存；离线示例使用脚本响应，输出明确标记为 Mock。

## 主要实现

| 部分 | 代码入口 | 用途 |
| --- | --- | --- |
| H1 主线 | `h1_cross_system_20261023/runner.py` | 同一自然轨迹采集 RECON、EXPLOITATION、VERIFICATION checkpoint；同一 checkpoint 执行基线和三个 Trap，局部预算 K=12 |
| GPT / Claude / GLM | `h1_gpt_cross_model_6source_20261024/`、`h1_claude_cross_model_6source_20261024/`、`h1_glm_5_3_flash_cross_model_6source_20261025/` | 复用 H1 路径，分别处理模型协议、请求记账和分析 |
| H2 当前五策略 | `starve_h2_final_dsv4_9source_20261023/runner.py` | NO_TRAP、三个固定 Trap、STATE_AWARE_FEASIBLE；按完整轨迹计算预算 B |
| H2 受控接入 | `h2/pentestgpt_dynamic_selection_20261023.py` | Gateway 约束、状态观察、可部署性检查与按固定优先级回退 |
| 三种 Trap | `traps/task_relevant_branch_v3.py`、`traps/dependency_v6.py`、`adapters/evidence_new_v3_runtime.py` | BRANCH_NEW_V3、DEPENDENCY_V6、EVIDENCE_NEW_V3 |
| 任务与边界 | `tasks/`、`p0a_pentestgpt/gateway.py` | WEB/API/SERVICE 本地任务、工具白名单、独立任务判定 |

日期后缀保留了原始实验模块的身份。其他被导入的历史模块属于兼容依赖，阅读顺序见 [代码导览](docs/CODE_GUIDE.md)。

## 安装

使用 Linux 和 Python 3.11 及以上版本；本次在 Python 3.14.7 上验证。在本目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

核心运行与离线验证不需要模型凭证。原生 Claude 会话导入/恢复是可选集成，依赖原有 PentestGPT 的 `unified_agent` 和相应 Agent SDK；其依赖边界见 [运行说明](docs/RUNNING.md)。

## 快速开始

运行 H1 离线准入检查：

```bash
python -m h1_cross_system_20261023.runner --preflight-only --output outputs/h1-preflight
```

运行当前 H2 三任务族、五策略离线演示：

```bash
python -m examples.offline_h2 --output outputs/h2-demo
```

运行测试：

```bash
python -m pytest -q
```

每次运行请使用新的输出目录。离线命令不请求真实模型。H2 演示中的三种 Trap 版本与当前五策略主线一致；旧版接线演示 `python run_h2.py --mode mock --budget 12 --output outputs/h2-legacy-demo` 保留用于理解最初的完整轨迹接口，其策略版本不同。

## 数据与复现

源码包含运行所需的本地任务 fixture、选样记录和实验配置。既有配置中的历史结果路径属于原实验的证据引用，相关结果不随源码分发。

重新运行 H1 可产生新的实验记录；跨模型历史结果比较需要额外提供对应 `pair_ledger.json`。正式 H2 启动器还会核对原批次的路由准入记录及冻结元数据，源码发布本身不能替代这些准入材料。离线演示可独立执行，用于验证核心机制接线。详见 [运行说明](docs/RUNNING.md) 与 [验证记录](docs/VALIDATION.md)。

## 放到 GitHub

新建 GitHub 仓库后，将**本目录中的内容**上传到仓库根目录，使 `README.md` 位于首页。也可解压配套源码包后上传。`.gitignore` 已配置本地凭证、虚拟环境、缓存和实验输出的忽略规则。

仓库链接格式为 `https://github.com/<你的账号>/H1H2-Agent-Resource-Defense`；创建并上传后即可在论文、报告或项目说明中引用。发布前可按作者实际意愿补充作者信息及许可证，当前整理未代替作者选择授权条款。
