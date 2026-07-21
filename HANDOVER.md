# MoeOwner 项目交接文档

> **交接日期：** 2026-07-21
> **交接人：** AI 助手
> **项目版本：** 参见 `git log -1`

---

## 目录

1. [项目概述](#1-项目概述)
2. [代码结构总览](#2-代码结构总览)
3. [核心模块详解](#3-核心模块详解)
4. [启动与运行](#4-启动与运行)
5. [开发注意事项](#5-开发注意事项)
6. [已知问题与技术债务](#6-已知问题与技术债务)
7. [测试覆盖](#7-测试覆盖)
8. [交接要点 Checklist](#8-交接要点-checklist)

---

## 1. 项目概述

**MoeOwner** 是一个面向 **Mixture-of-Experts 大模型**的纯 Python 推理引擎，核心定位为**零手动调优、插电即用**。项目将 PagedAttention + RadixAttention KV Cache、专家缓存、SERE 推测跳过、N-Gram 推测解码、超长上下文扩展（SelfExtend / ReAttention / YaRN）等多项优化深度融合，所有优化全部自动开启。

### 核心能力

| 能力 | 说明 |
|------|------|
| 🚀 **MoE 推理** | 支持 HuggingFace 模型和 GGUF 格式加载，所有优化自动适配 |
| 🧊 **KV Cache 管理** | PagedAttention 物理块 + RadixAttention 哈希索引 + 非对称量化（K-INT8/V-INT4） |
| 🔮 **推测解码** | Goose PLD 线性链 + Self-Spec 骨架推测 + N-Gram Trie 三重推测，全部自动开启 |
| 🧠 **专家优化** | 层次化 LFRU 专家缓存（DRAM↔VRAM）、SERE 动态跳过、OEF 熵冻结、AFCE 锚点缓存 |
| 🧩 **超长上下文** | SelfExtend（默认）/ ReAttention / YaRN，训练无关，零权重改动 |
| 🛠️ **工具下沉** | 模型内生 `[[tool(...)]]` 调用——9 项内置工具、3 轮循环编排、format_fixer 清洗 |
| 🌐 **HTTP API** | OpenAI 兼容端点 `/v1/completions`、`/v1/chat/completions`，支持 SSE 流式 + Reasoning |

### 技术栈

- **Python 3.11+**（推荐 3.12）
- **PyTorch 2.13**（CUDA 13.0）
- **Transformers 5.14.1**
- **纯 Python GGUF v3 解析器**（零 C 扩展）
- **零第三方 API 依赖**

---

## 2. 代码结构总览

```
MoeOwner/
├── main.py                  # 🚀 入口：全局配置、模型注入、事件循环、HTTP API
├── attention_kernel.py      # 🔥 FlashAttention 核（PyTorch SDPA + torch.compile）
├── cache_manager.py         # 📦 混合 KV 缓存（Paged + Radix + 非对称量化）
├── scheduler.py             # ⏱ 统一调度器（Chunked Prefill + Decode 双流管线）
├── vram_budget.py           # 🧮 集中式显存预算管理器
├── moe_logger.py            # 📋 统一日志系统
├── tool_sink.py             # 🛠️ 工具下沉框架（[新] 1482 行，零第三方依赖）
├── sere.py                  # ⚡ 动态专家跳过（SERE）
├── expert_cache.py          # 🧠 层次化专家缓存（LFRU）
├── oef.py                   # 🧬 机会性熵冻结
├── afce.py                  # 🧠 锚点前向缓存扩展
├── ngram_speculation.py     # 🔮 N-Gram 推测解码
├── speculative_prefetch.py  # ⚡ 推测预取 + 动态专家激活
├── goose_core.py            # 🦆 Goose 推测解码引擎
├── launch.py                # 🚀 启动辅助
├── api_server.py            # 🌐 OpenAI 兼容 HTTP API 服务器
├── model_loader/
│   ├── __init__.py          # 公共 API
│   ├── gguf_reader.py       # GGUF v3 解析 + 反量化 kernel
│   ├── model_adapter.py     # 高层适配器
│   └── README.md
├── long_context/
│   ├── __init__.py
│   ├── config.py            # SelfExtend / ReAttention / YaRN 配置
│   ├── self_extend.py       # SelfExtend 实现（4 行位置编码逻辑）
│   ├── re_attention.py      # ReAttention 实现
│   ├── integration.py       # 注入器：替换 attention 前向
│   ├── tests.py
│   └── bench_dry_run.py
├── tests/
│   ├── test_gguf_reader.py
│   ├── test_goose_logic.py
│   └── test_kv_quantization.py
├── start.bat / 启动.bat     # Windows 启动脚本
├── 启动.sh                  # Linux 启动脚本
├── HANDOVER.md              # 📄 本交接文档
├── README.md                # 项目文档（中英双语）
├── LICENSE                  # CC BY-NC-SA 4.0
└── requirements.txt
```

### 文件职责边界

| 文件 | 归属人 | 依赖关系 |
|------|--------|----------|
| `main.py` | 入口 | 依赖所有模块 |
| `scheduler.py` | ⭐ 核心调度 | 依赖 cache_manager, attention_kernel, sere, expert_cache, oef, afce, goose_core, speculative_prefetch |
| `cache_manager.py` | ⭐ 核心缓存 | 无内部依赖（独立单元测试） |
| `tool_sink.py` | 🆕 工具下沉 | 无内部依赖（stdlib 仅限；引用 scheduler 的 Request 仅用于集成） |
| `api_server.py` | HTTP API | 依赖 scheduler, tool_sink（可选） |
| `vram_budget.py` | 显存预算 | 依赖 torch |
| 其余模块 | 专用优化 | 依赖 torch, 部分依赖 scheduler |

---

## 3. 核心模块详解

### 3.1 `main.py` — 引擎入口

**功能：**
- 全局环境变量锁定（`PYTORCH_CUDA_ALLOC_CONF`）
- Torch 全局性能配置（Flash SDPA、TF32、cuDNN benchmark 等）
- 模型加载（HF 或 GGUF，自动识别文件后缀）
- 注意力核注入（FlashAttention / SelfExtend / ReAttention）
- `torch.compile` 编译链（reduce-overhead → default → 跳过）
- 创建 `HybridCache` + `UnifiedScheduler`
- 启动 HTTP API 服务器或 Benchmark 模式

**参数：** 详见 `python main.py --help`

**关键设计决策：**
- `os.environ["PYTORCH_CUDA_ALLOC_CONF"]` 必须放在 import torch 之前，否则不生效
- `set_long_context_config()` 在模型加载前调用，注入器从中读取配置
- `api_server.py` 通过 `asyncio.start_server` 实现纯异步 HTTP，无 FastAPI 依赖

### 3.2 `scheduler.py` — 统一调度器（1427 行，核心复杂度最高）

**三阶段管线：**
1. **Chunked Prefill** — 将长 prompt 切成 `CHUNK_SIZE` 块，逐块 prefill 并存储 KV
2. **Decode** — 逐 token 解码，从 KV Cache 读取历史，`argmax` 采样
3. **GC** — 完成请求回收缓存块

**并行执行的优化策略（全部自动开启）：**
- Goose PLD 推测解码（`_decode_speculative()`）
- Self-Spec 骨架推测（`_decode_self_speculative()`）
- 自适应 KV 压缩（`_compress_kv_adaptive()`）
- SERE 动态专家跳过
- OEF 熵冻结
- AFCE 锚点扩展
- 动态专家激活 + 推测预取
- VRAM 预算运行时监控（三级水位）

**关键设计决策：**
- `_decode_step()` 从 `HybridCache.load_kv()` 读取 KV 而非全量重算
- 双 CUDA 流（`prefill_stream` + `decode_stream`），主线程 `torch.cuda.synchronize()` 仅会在 `_garbage_collect()` 中调用
- `start_new_session=True` 的 subprocess 隔离（sandbox 用）
- **每请求的 token 列表 `req.tokens` 包含历史**（即 `prompt + generated`），不是仅当前 token

### 3.3 `cache_manager.py` — 混合 KV 缓存（876 行）

**数据结构：**
- `Block` — KV 缓存块描述符（`__slots__` 优化 + `array('i')` 紧凑哈希链）
- `HybridCache` — 空闲队列 + 分配表 + Radix 哈希索引

**关键特性：**
- `BLAKE2b` 增量哈希：`BLAKE2b(BLAKE2b(prev).digest() + token_bytes)`，注意不是 `hashlib.update()`
- `hash_prefix` LRU 缓存（max 256 条目），重复前缀 O(1) 命中
- `compute_hash()` 用于 AFCE 侧车 keys
- `pin_prefix_from_match()` 用于前缀缓存固定
- 复合键守卫 GC：防止哈希重用导致的误删

**非对称量化（K-INT8 / V-INT4）:**
- Key: 每个 head 独立 per-head 缩放（`absmax / 127.0`），FP16 恢复
- Value: 4-bit 位运算打包，2 个值压缩到 1 个字节，对称量化
- 前后各 8 个 token 保留 FP16（`_PROTECTED_N = 8`）
- `block_size` 是 token 数倍数，不是 byte 大小

### 3.4 `tool_sink.py` — 🆕 工具下沉框架（1482 行）

**这是本次新增的最重要模块。**

**三个核心类：**

| 类 | 职责 |
|------|------|
| `ToolContext` | 请求级生命周期容器——临时目录、内存键值对、消息历史、并发锁、进程/内存/磁盘同步销毁 |
| `ToolScanner` | 逐字符状态机，检测 `[[tool(...)]]`。4 状态：TEXT → LEFT_BRACKET → LEFT_DOUBLE → SAW_CLOSE_BRACKET。误触恢复。 |
| `ToolOrchestrator` | 编排引擎——提交推理 → 扫描 → 执行工具 → 重构历史 → 重新推理（最多 3 轮）→ format_fixer 后钩子 |

**9 项内置工具（硬编码，禁止动态注册）：**
1. `memo_set` / `memo_get` — 工作区内存键值
2. `sci_calc` — 科学计算（math + cmath 全量导出、AST 形状校验、线程超时）
3. `sys_env` — 系统自省（硬件信息、Windows 兼容）
4. `list_ports` — 端口扫描（多级降级）
5. `ping_target` — TCP 探测（1 次重试）
6. `format_fixer` — JSON 清洗（工具+钩子双角色）
7. `sandbox_run` — 子进程沙箱（白名单 builtins、5s 超时 SIGKILL、64KB 截断）
8. `fetch_url` — 网页访问（协议校验、Content-Type 过滤、三重超时）

**设计要点：**
- `signal.SIGALRM` 被故意避免（Windows 不兼容），用 `threading.Thread.join(timeout)` 替代
- `os.uname()` 有 `platform` 模块回退
- `subprocess.start_new_session=True` 在 Windows 上会被忽略但不会报错
- `json.dumps(_SANDBOX_SAFE_BUILTINS)` 序列化失败是因为 `type` 不可 JSON 序列化，改用 name-reference 方式修复
- 参数解析四阶段：`literal_eval` → `json.loads` → 自定义 key=value 解析器 → 位置回退

**集成方式：**

```python
from tool_sink import create_orchestrator

orchestrator = create_orchestrator(
    scheduler=scheduler,
    detokenizer=tokenizer.decode,
    tokenizer_fn=tokenizer.encode,  # 可选
    enable=True,                     # False = 纯透传，无工具扫描
)

# 在 API 处理器中使用
final_text = await orchestrator.generate(
    prompt=user_message,
    max_tokens=512,
)
```

### 3.5 `api_server.py` — HTTP API 服务器

**端点：** OpenAI 完全兼容
- `POST /v1/completions` — 文本补全（流式 + 非流式）
- `POST /v1/chat/completions` — 对话补全（流式 + 非流式）
- `GET /v1/models` — 模型元数据
- `GET /health` — 健康检查

**增强功能：**
- `<think>...</think>` / `<thinking>...</thinking>` 推理内容提取（DeepSeek-R1 / QwQ 风格）
- 流式 SSE 输出时，reasoning_content 和 content 分离发送
- CORS 支持

**技术选型：** `asyncio.start_server` 纯 stdlib 实现，0 第三方依赖（FastAPI/uvicorn 等）

---

## 4. 启动与运行

### 标准启动

```bash
# 无 API 服务器（引擎模式）
python main.py --model Qwen/Qwen2.5-1.5B-Instruct

# 带 API 服务器
python main.py --model Qwen/Qwen2.5-1.5B-Instruct --api-port 8000

# GGUF 模型
python main.py --gguf /path/to/model.Q4_0.gguf --api-port 8000

# Benchmark
python main.py --model Qwen/Qwen2.5-1.5B-Instruct --benchmark

# 详细日志
python main.py --model Qwen/Qwen2.5-1.5B-Instruct --verbose
```

### Windows 启动

项目提供 `start.bat` 和 `启动.bat`，双击运行（后者中文界面）。

### 单元测试

```bash
python -m pytest tests/ -v
```

### tool_sink.py 自测

```bash
cd MoeOwner
python -c "
from tool_sink import _demo_tools
_demo_tools()
"
```

---

## 5. 开发注意事项

### 5.1 通用规则

1. **`PYTORCH_CUDA_ALLOC_CONF` 必须在 `import torch` 之前设置。** `main.py` 第 13 行处理了这一点，新增模块如果有类似需要注意。
2. **不要轻易升级 PyTorch 版本。** 当前 2.13.0+cu130 是经过验证的稳定版本，升级可能导致 `torch.compile` 或 CUDA Graph 相关代码失效。
3. **所有工具调用必须串行。** `ToolOrchestrator` 的设计是单工具/单次推理，没有并行执行。
4. **零第三方依赖是硬性要求。** `tool_sink.py`、`api_server.py` 等扩展模块必须只使用 Python stdlib。项目本身依赖 torch + transformers 已不可避免，但新模块不应增加新的 pip 依赖。
5. **Goose 推测解码和 Self-Spec 骨架推测是独立的。** 两者可以同时启用，PLD 处理低熵任务（代码、模板），骨架处理高熵任务（创作、推理）。

### 5.2 tool_sink.py 特有规则

1. **不要用 `signal.alarm` 做超时。** 它在 Windows 上不存在（`AttributeError`）。所有超时用 `threading.Thread.join(timeout=...)` 实现。
2. **不要用 `json.dumps` 序列化 Python 类型对象。** `type`、`built-in function` 等不可 JSON 序列化。`sandbox_run` 自己拼接代码字符串。
3. **`_parse_key_value` 正则匹配在设计上要求 key=value 之间没有空格以外的分隔符。** 如果未来需要支持复杂类型（如嵌套 dict 作为参数值），需要扩展解析器。
4. **`ToolContext._child_procs` 列表不是线程安全的添加/移除。** 当前设计是仅追加 + 销毁时统一清理，如果在执行中间销毁则子进程会泄漏。目前没有工具在 run 中调用 destroy，所以安全。
5. **不支持 `[[tool(...)]]` 嵌套。** 状态机检测到 `]]` 即返回完整标记，嵌套的 `[...]` 不会被正确解析。

### 5.3 scheduler.py 维护须知

1. **`TokenBuffer.length` vs `len(req.tokens)`：** 当前设计 `req.tokens` 存储 `prompt + generated` 的全部历史。如果要改成只存储最近窗口，需要同时修改 KV cache 的索引。
2. **`_last_kv_cache` 命名是个脆皮约定。** 目前依赖 GGUF 适配器把最新 KV 缓存放这里。如果换了模型后端需重新检查此接口。
3. **`_decode_step` 中的 `argmax` 采样。** 当前硬编码为贪心采样；要支持 temperature/ top-k/ top-p 需要在此处添加采样器。
4. **`_extract_logits` 和 `_extract_past_key_values`** 是适配层，用于统一 HF CausalLMOutput 和 GGUF 原始 tensor 的输出格式。

### 5.4 Windows + NVIDIA GPU 兼容

| 特性 | 兼容情况 |
|------|----------|
| PyTorch CUDA | ✅ 原生支持 Windows + NVIDIA |
| `subprocess.start_new_session=True` | ⚠️ Windows 忽略此参数但不报错 |
| `os.uname()` | ❌ Windows 无此调用 → `platform.system()` 回退 |
| `signal.SIGALRM` | ❌ 不存在 → 禁用 |
| `/proc/net/tcp` | ❌ 无 procfs → 用 `netstat -an` 降级 |
| 路径分隔符 | ⚠️ 硬编码了 `"/proc/"` 路径，`list_ports` 会自动降级 |
| `os.sysconf()` | ❌ 不存在 → `ctypes.windll.kernel32` 回退 |

### 5.5 GGUF 加载器注意事项

- `model_loader/` 是纯 Python 实现，通过 `struct.unpack` + `mmap` 解析 GGUF 文件
- Q4_0 / Q8_0 反量化通过位运算（位移 + 解包）实现，零 C 扩展
- 模型适配器需要实现 `forward()` 返回与 HF 兼容的格式（支持 `use_cache` 和 `past_key_values` 参数）
- `estimate_parameter_count_b()` 用于 VRAMBudget 的初始化预算
- 如果 GGUF 文件损坏，`mmap` 会抛出 `OSError`，当前没有详细诊断信息

---

## 6. 已知问题与技术债务

### 6.1 当前已知问题

| # | 问题 | 严重度 | 影响范围 | 状态 |
|---|------|--------|----------|------|
| 1 | `tool_sink.py` 中的 `fetch_url` 使用的 `urllib.request` 默认**遵循**重定向，设计与方案中"禁用自动重定向（或限制最大 3 次"不完全一致 | 低 | 工具行为偏离 | 🔧 已实现为 follow 重定向（urllib 默认行为） |
| 2 | `_parse_key_value` 无法处理值为非简单类型的复杂 JSON | 低 | memo_set 的值只能是字符串 | 📝 当前满足需求 |
| 3 | `ToolOrchestrator.generate()` 中的 `_scheduler.step()` 调用可能在无事件循环时失败 | 中 | 异步集成 | ⚠️ try/except 兜底 |
| 4 | AFCE 的 `extract_anchors_after_prefill` 当前是空函数（`afce.py` 内有注释 `# TODO`） | 中 | AFCE 锚点提取未实现 | ⚠️ 设计已就绪 |
| 5 | 工具调用状态机不支持代码块（```）内部跳过 | 低 | 代码块内 `[[` 被误检测 | 📝 设计提及但未实现 |
| 6 | `sandbox_run` 的限制 builtins 通过 `__builtins__.__dict__.clear()` 实现，但 `exec` 自带的上层 globals 仍包含 `__builtins__` 模块引用 | 中 | 沙箱隔离强度 | ⚠️ 当前足够 |
| 7 | 无 `ruff` 配置对 `tool_sink.py` 做静态检查 | 低 | 代码风格 | 📝 需添加 |

### 6.2 技术债务

- **测试覆盖不足：** 目前测试集中在 `cache_manager.py`（58 项），其余模块覆盖偏低。`tool_sink.py` 和 `scheduler.py` 需要更多单元测试。
- **类型注解不完整：** 一些较早的代码缺少类型注解（尤其是 `scheduler.py` 中 `_extract_logits` 等适配函数）。
- **缺少基准测试套件：** 当前 Benchmark 仅支持单 prompt 长度、单 generation 长度。没有自动化回归基准。
- **Chunked Prefill 的批处理超时：** `_PREFILL_BATCH_TIMEOUT` 是启发式计算的，可能需要针对不同 GPU 调优。
- **API 服务器无授权：** 当前 `/v1/*` 端点完全开放，无 API Key 验证。生产部署需前置反向代理。

---

## 7. 测试覆盖

### 当前测试文件

| 文件 | 范围 | 通过数 |
|------|------|--------|
| `tests/test_gguf_reader.py` | GGUF 文件解析、反量化 round-trip | 待确认 |
| `tests/test_goose_logic.py` | Goose 推测解码引擎 | 待确认 |
| `tests/test_kv_quantization.py` | KV 量化 round-trip | 待确认 |

### cache_manager.py 内置测试

`import cache_manager` 后运行 `python -c "from cache_manager import HybridCache; ..."` 可执行 58 项断言测试。

### tool_sink.py 内置测试

```bash
python -c "from tool_sink import _demo_tools; _demo_tools()"
```
会运行所有 9 项工具的集成测试。

### 新增工具下沉框架的验证

需要人工确认的边界情况：

- [ ] 模型输出中包含 `[[` 但非工具调用（如数学表达式 `[[a, b]]`）
- [ ] `sandbox_run` 超时（5 秒）时能否正确杀死进程树
- [ ] `fetch_url` 对超大 HTML 页面的截断准确性
- [ ] `format_fixer` 对非 JSON 输出的处理
- [ ] 3 轮循环后 `LOOP_LIMIT_EXCEEDED` 是否正确返回
- [ ] Windows 上 `list_ports` 通过 `netstat` 降级是否能正常工作

---

## 8. 交接要点 Checklist

### 代码仓库

- [x] 仓库地址：`https://github.com/SuiJQ/MoeOwner`
- [x] 远程已配置 OAuth token（在 `TOOLS.md` 中）
- [x] `.gitignore` 已排除 `__pycache__` 和 `.ruff_cache`

### 关键文件

- [x] `README.md` — 完整中英文文档（已更新 tool_sink）
- [x] `HANDOVER.md` — 本交接文档
- [x] `TOOLS.md` — GitHub token 等本地配置
- [ ] `requirements.txt` — 可能需要更新以包含 `tool_sink.py` 的测试需求（当前无额外依赖）

### 启动验证

接手后建议执行：

```bash
# 1. 确认 Python 环境
python --version  # 应 ≥ 3.11
pip list | grep -i torch  # 应 2.13.0

# 2. 运行工具下沉自测
cd MoeOwner && python -c "from tool_sink import _demo_tools; _demo_tools()"

# 3. 运行单元测试
python -m pytest tests/ -v

# 4. 尝试小型模型推理（需要 GPU）
python main.py --model Qwen/Qwen2.5-1.5B-Instruct --benchmark

# 5. 启动 API 服务器并测试
python main.py --model Qwen/Qwen2.5-1.5B-Instruct --api-port 8000
# 另一个终端：
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

### 后续维护建议

1. **优先补充工具下沉的测试用例**：当前以自测脚本为主，建议迁移到 `pytest` 测试框架
2. **完善参数解析器**：当前 `_parse_key_value` 对非简单类型支持有限，如果模型开始生成复杂参数需要扩展
3. **检查 AFCE 锚点提取**：`afce.py` 的 `extract_anchors_after_prefill` 当前为空
4. **添加代码块跳过**：状态机检测 ```` 内部时应跳过 `[[` 检测
5. **性能回归基准**：建议固定一个 Benchmark 命令并记录结果以便发现回归

---

## 附录 A：工具调用格式

模型需要生成以下格式的文本来触发工具调用：

```
我需要计算 5 的阶乘。
[[sci_calc(expression=math.factorial(5))]]
结果是 120。
```

工具调用的规则：
- 必须独占一行（目前未强制检查，但推荐）
- 格式：`[[tool_name(key=value, key2=value2)]]`
- 值用双引号或单引号包裹（字符串），或裸写（数值/表达式）
- 最多 3 轮工具调用，超限返回错误

## 附录 B：Git 提交规范

项目使用常规的 git commit，信息为英文。建议保持：

```
feat: 新功能
fix: 修复
docs: 文档
refactor: 重构
test: 测试
chore: 杂项
```

---

*本文档由 AI 助手于 2026-07-21 生成。如有不准确之处，请以实际代码为准。*
