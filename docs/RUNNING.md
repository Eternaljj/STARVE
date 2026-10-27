# 运行与分析

在源码根目录运行，输出目录必须使用新名称。依赖安装见根目录 README。

## 离线验证

```bash
python -m pytest -q
python -m h1.runner --preflight-only --output outputs/h1-preflight
python -m h2.runner --preflight-only --output outputs/h2-preflight
python -m examples.offline_h2 --output outputs/h2-demo
```

H1 检查九个选定任务、27 个 checkpoint 和 108 个条件。H2 检查 WEB/API/SERVICE 各一个代表任务的五种策略；每条脚本响应轨迹覆盖三个正式状态。Mock usage 和脚本选择仅用于验证接线。

## 共用模型配置

`configs/provider.json` 同时供 H1/H2 读取。`LLM_MODEL` 和 `LLM_BASE_URL` 环境变量可以覆盖文件中的模型与地址；密钥从 `LLM_API_KEY` 读取。`.env.example` 仅列出变量，程序不会自动加载该文件。

当前内置接入为 Anthropic Messages 兼容协议，实际请求地址为 `<LLM_BASE_URL>/v1/messages`。模型名称可切换，但服务端必须兼容该接口及工具调用协议。其他协议需要实现对应适配器。采样温度保持 0，输出上限 8192，工具调用串行，无自动重试。

H1 先执行单任务真实检查，再运行完整实验：

```bash
python -m h1.runner --one-source-preflight 1 --output outputs/h1-live-check
python -m h1.runner --start --output outputs/h1-live
python -m h1.analyze outputs/h1-live/live
```

H1 完整执行结束后也会自动分析本轮数据。报告根据当前记录生成。

H2 按指定预算启动五策略完整轨迹：

```bash
python -m h2.runner --start --source-index 1 --budget 20 --output outputs/h2-live
```

可选预算为 20、30。省略 `--source-index` 时运行 `configs/sources.json` 中全部九个任务。每条轨迹保存独立请求日志与 `trajectory.json`，总览写入 `summary.json`。技术失败或协议错误会停止当前批次，已产生的记录保留。



