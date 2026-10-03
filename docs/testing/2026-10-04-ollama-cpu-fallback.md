# 本地 Ollama 自动加速与 CPU 回退验证

现场日志显示 Windows / Ollama 0.32.13 / RX 7700 XT / Vulkan 在 bge-m3
warmup 阶段以 `0xc0000409` 退出。服务端口仍响应，重新拉取模型不能恢复。
本变更针对这一执行失败增加有界 CPU 重试，不将错误码本身当作驱动根因证明。

## 自动化回归

- `pytest -q tests/test_ollama_embedding_fallback.py tests/test_ollama_diagnostics.py tests/test_llm_providers.py`：159 passed。
- 健康 API 原有 loopback 冷加载测试新增 CPU 回退分支：未回退的普通冷加载保留兼容策略；已回退后必须实际生成向量才能报告可用。
- 新增回归先验证旧实现失败：现场 runner 崩溃返回空结果、CPU 回退中的超时仍错误报告可用、HTTP 200 无效向量未尝试 CPU；随后修复并验证通过。
- 覆盖原生崩溃 / CUDA OOM / Vulkan device lost、诊断与正式调用共享模式、并发、取消、CPU 再次失败及恢复、endpoint/model 隔离、无效向量、远端及无关错误不切换。
- `pytest -q` 全量：9876 passed、65 skipped、1 failed（2020.30 秒）。失败为
  `tests/test_saved_sync_identity_pipeline.py::test_recommendation_api_preserves_canonical_identity`：
  推荐 worker 代理连接失败（`All connection attempts failed`），HTTP 502 而非期望的 200。
  该测试在本分支单独复跑通过，整个文件 14 项复跑通过；原始 main 代码的该单测也通过。
  尚未证明全量触发的顺序 / 环境原因，不将此结果记为全量通过；本次不改推荐代理链路。
- `ruff check src/ tests/`：通过。
- `mypy src/`：通过，305 source files。

## 真实推理对照

在本机 macOS 使用已安装的 Ollama 0.18.2 和 bge-m3，启动独立临时端口 daemon，
复用现有模型文件、禁用 prune；测试后停止该测试专属进程组，不重启用户的 Ollama。

1. 自动模式真实请求返回 1024 维向量。
2. 仅注入首次 HTTP 500（现场 `llama-server ... 0xc0000409` 签名），CPU 请求转发真实 Ollama。
3. `diagnose_ollama_embedding()` 经 CPU 重试返回 `ok`。
4. 新建 provider 继续使用 CPU，真实请求再次返回 1024 维向量。

实际终端输出（独立测试进程）：

```text
Real default inference: 1024 dimensions
Ollama embedding automatic runner failed; switching model=bge-m3 to CPU for this application session (num_gpu=0)
Injected runner crash -> real CPU inference -> diagnostic ok -> fresh provider CPU inference: PASS
```

这验证了真实 CPU 推理与应用切换链路，但没有复现 Windows Vulkan 原生崩溃，也不代表
Windows 0.32.13 / RX 7700 XT 已实机验收。用户机器应确认：首次失败后日志出现
`switching ... to CPU`、runner 使用 `-ngl 0`、后续向量请求成功且不反复重试 GPU。
若 CPU 同样失败，界面必须保留不可用与具体原因，不能以 `/api/version` 成功验收。

## 交付边界

- 本地 loopback 向量接口共用策略；桌面 / 移动 Web / 插件 / CLI 不各自维护模式。
- 按后端进程的 endpoint/model 记忆 CPU 到退出；重启应用重新尝试自动加速。
- 不改聊天请求、系统 GPU 环境或远端 Ollama；Compose 默认 `http://ollama:11434` 不在此自动回退范围内。
- Windows 包保留 Vulkan / CPU，继续裁剪 CUDA / ROCm；本次未生成或发布安装包。
