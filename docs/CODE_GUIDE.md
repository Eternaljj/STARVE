# 代码导览

建议阅读顺序：

1. `h1/runner.py`：checkpoint、三个状态、K=12 与四条件配对。
2. `h2/runtime.py`：五策略、可部署性检查、状态优先级回退与完整轨迹计量。
3. `tasks/extended_environment.py`、`tasks/multi_task_pool_v1.py`：共用任务环境、实际本地动作与独立判定。
4. `p0a_pentestgpt/gateway.py`：工具白名单、参数校验与执行效果记录。
5. `adapters/lifecycle_state_collector.py`、`adapters/lifecycle_web_behavior.py`：行为证据与状态判断。
6. `adapters/lifecycle_branch_new_v3_runtime.py`、`adapters/lifecycle_dependency_v6_runtime.py`、`adapters/evidence_new_v3_runtime.py`：三种主要 Trap。
7. `adapters/llm_provider.py`：共用模型接入；`h1/analyze.py`、`h2/metrics.py`：分析接口。

| 目录 | 职责 |
| --- | --- |
| `h1/` | H1 唯一运行入口及分析 |
| `h2/` | H2 唯一运行入口、五策略运行时、预算与共用组合逻辑 |
| `core/` | 共用会话、checkpoint、K=12 包装、锁与配对校验 |
| `tasks/` | 六类模板的基础回归 fixture、九个选定实验 fixture 与共用环境 |
| `traps/` | Trap 规则、材料与数据契约 |
| `adapters/` | 状态观察、机制运行、模型传输 |
| `p0a_pentestgpt/` | 工具 Gateway、协议转换、请求日志 |
| `configs/` | 单一模型配置、H2 参数、任务选样与必要契约 |
| `examples/`、`tests/` | 离线示例与回归检查 |

`traps/` 只保留 `task_relevant_branch_v3.py`、`dependency_v6.py`、`evidence_v3.py` 及包入口。三种最新机制均不继承旧版 Trap 实现。图结构、动作观察、预算、计划提交等共用逻辑已提取到 `core/` 和 `adapters/runtime_support.py`。

`tasks/dependency_records.py` 提供任务完整来源记录，V6 根据这些真实记录生成范围明确的配置对象。数据中的已有版本标记属于任务记录契约，不提供旧 Trap 实现或启动入口。

H1 在同一 checkpoint 内比较 Trap 与 BASELINE；H2 为每个任务和策略重新建立完整轨迹。H1 局部预算 K 与 H2 完整轨迹预算 B 分别记账。正式状态为 RECON、EXPLOITATION、VERIFICATION；证据不足时保留 UNKNOWN。任务完成以环境独立判定为准，缺失 Token usage 保持为空。
