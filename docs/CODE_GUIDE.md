# 代码导览

## 从哪里读起

1. `h1_cross_system_20261023/runner.py`：查看状态、实验条件、K=12、自然轨迹采集、checkpoint 准入和配对执行。
2. `tasks/multi_task_pool_v1.py` 与 `tasks/multi_task_pool_v3_dependency_v5_adapter.py`：查看任务模型、实际本地动作和独立任务判定。
3. `p0a_pentestgpt/gateway.py`：查看工具参数校验、允许动作和 effect 记录。
4. `adapters/lifecycle_state_collector.py`、`adapters/lifecycle_web_behavior.py`：查看行为证据如何产生状态标签。
5. `adapters/lifecycle_branch_new_v3_runtime.py`、`adapters/lifecycle_dependency_v6_runtime.py`、`adapters/evidence_new_v3_runtime.py`：查看三种机制如何在执行前曝光并持续处理后续动作。
6. `starve_h2_final_dsv4_9source_20261023/runner.py`：查看完整轨迹的五策略比较、部署资格、优先级回退与预算。
7. 各模型目录的 `analyze.py`、H2 的 `analysis.py`：查看配对记录、资源指标、缺失值和分析输出。

## 目录职责

| 目录/文件 | 职责 |
| --- | --- |
| `core/` | checkpoint 类型、实验身份、目录锁与调度锁 |
| `tasks/` | WEB/API/SERVICE fixtures、环境与 Session 适配 |
| `traps/` | 候选、规则、材料和机制数据契约 |
| `adapters/` | 状态观察、pre-effect 运行时、请求/资源记账 |
| `p0a_pentestgpt/` | 工具 Gateway、协议兼容、请求日志及可选原生会话集成 |
| `h1_*cross*` | H1 主线及跨模型运行与分析 |
| `h2/` | 预算、策略、轨迹循环、受控接入与早期接口验证 |
| `starve_h2_final_dsv4_9source_20261023/` | 当前 H2 五策略正式运行与分析 |
| `configs/` | 既有实验参数和契约 |
| `examples/` | 发布版独立离线演示 |
| `tests/` 及模块内 `tests/` | 随发布版保留的主线回归测试 |
| `h1_evidence_v3_compat.py` | 从历史结果目录移出的 Evidence 兼容源码 |
| `h1_runtime_compat.py` | 从历史脚本提取的共享 K=12 包装类 |
| `SOURCE_INVENTORY.json` | 发布文件和原工作区源码位置对应关系 |

主线仍导入部分有旧版本名称的基础模块，因此保留了必要依赖。`traps/v1.py` 在发布版仅保留被主线导入的数据定义。兼容版本元数据通过版本号和文件修改时间记录。

## 关键口径

H1 在同一个 checkpoint 内比较 Trap 与 BASELINE；H2 为每个 source、预算和策略重新建立完整任务轨迹。H1 的局部 K 与 H2 的完整轨迹 B 分别记账。

正式状态为 RECON、EXPLOITATION、VERIFICATION；没有足够证据时保留 UNKNOWN。任务完成以环境独立判定为准。机制完成、模型自行宣称完成和预算耗尽分别记录。Token 缺失值保留为空，跨模型比较应保留各自 usage 口径。
