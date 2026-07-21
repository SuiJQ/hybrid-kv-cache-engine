# MoeOwner

> **MoE 异构推理引擎** — 融合 PagedAttention + RadixAttention KV Cache、专家缓存、SERE 推测跳过与 N-Gram 推测解码
>
> _🏷️ MoE · LLM Inference · KV Cache · Speculative Decoding · Expert Offloading · GGUF · Pure Python · CUDA · Flash Attention · Long Context_

---

<p align="center">
  🌐 <strong>中文</strong> · <a href="#english-version"><strong>English</strong></a>
</p>

<p align="center">
  <em>↓ 底部有英文精简版 ↓</em>
</p>

---

## 项目定位

**MoeOwner** 是一个面向 Mixture-of-Experts 大模型的生产级推理引擎，将传统 KV Cache 管理与 MoE 专属优化深度融合：

| 组件 | 文件 | 功能 |
|------|------|------|
| 🔥 **FlashAttention 核** | `attention_kernel.py` | 基于 PyTorch SDPA + `torch.compile` 的优化注意力 |
| 📦 **混合缓存** | `cache_manager.py` | PagedAttention 物理块 + RadixAttention 哈希索引 |
| 🧮 **显存预算** | `vram_budget.py` | 集中式显存预算管理器，启动评估 + 运行时 OOM 防护 |
| ⏱ **统一调度器** | `scheduler.py` | Chunked Prefill + Decode 双 CUDA 流调度 |
| 🚀 **入口** | `main.py` | 全局配置、模型注入、事件循环 |
| 📖 **GGUF 加载** | `model_loader/` | 纯 Python GGUF v3 解析器 + PyTorch 原生量化适配 |
| 🧩 **超长上下文** | `long_context/` | SelfExtend / ReAttention / YaRN 训练无关上下文扩展（自动开启） |
| 🔮 **推测解码** | `ngram_speculation.py` + `goose_core` + `scheduler.py` | CPU Trie N-Gram + Goose PLD 线性链 + Self-Spec 骨架推测（全部自动开启） |
| ⚡ **SERE** | `sere.py` | 动态专家跳过，top-k 后重路由（根据模型自动调参） |
| 🧠 **专家缓存** | `expert_cache.py` | 层次化 LFRU 专家权重卸载/加载（VRAM 容量自动计算） |
| 🧬 **OEF** | `oef.py` | 机会性熵冻结，旁路监控路由置信度，自动建议跳过确定性专家 |
| 🧠 **AFCE** | `afce.py` | 锚点前向缓存扩展，32-token 簇锚点旁路，修复长上下文语义遗忘 |
| 🛠️ **工具下沉** | `tool_sink.py` | 模型内生工具调用框架——状态机扫描 `[[tool(...)]]` 标记、7 项内置工具、3 轮循环编排、零第三方依赖 |

> **关键词**：MoE推理 | 大模型加速 | 混合KV缓存 | 专家卸载 | FlashAttention | 推测解码 | 超长上下文 | 双流管线 | 显存预算 | PyTorch | GGUF | 纯Python | 工具下沉

---

## 🎯 零手动调优 — 插电即用

**MoeOwner 的所有优化手段全部自动开启，零参数、零配置、零手动调优。** 用户只需 `python main.py --model <某个模型>`，其余全部自动完成：

| 优化项 | 自动生效方式 | 用户操作 |
|--------|-------------|----------|
| 🔥 **FlashAttention SDPA** | 强制启用 flash 版，禁用 math/mem_efficient 回退 | `main.py` 启动即开 ✅ |
| ⚡ **cuDNN benchmark** | `torch.backends.cudnn.benchmark = True` | 自动 ✅ |
| 🚀 **TF32 矩阵乘** | matmul + cuDNN 双路允许 TF32（Ampere+ 架构） | 自动 ✅ |
| 🎯 **Float32 matmul precision** | `torch.set_float32_matmul_precision("high")` | 自动 ✅ |
| 📦 **torch.compile 静态图编译** | 自动尝试 reduce-overhead → default 降级链，失败则跳过 | 自动 ✅ |
| 🧊 **KV Cache 非对称量化** | Key→INT8 + Value→INT4，首尾 8 token 保留 FP16，门控透明 | 自动 ✅ |
| ⏱ **自适应 KV 压缩** | 根据模型 max_seq_len 自动调优 sink/window/importance 参数 | 自动 ✅ |
| 🧠 **SERE 动态专家跳过** | 根据 num_experts/top_k 自动调优 skip_threshold 和 min_experts | 自动 ✅ |
| 🏗 **层次化专家缓存** | 根据 GPU 可用显存自动计算 VRAM 容量 | 自动 ✅ |
| 🔄 **Self-Spec 骨架推测** | `SkeletonDraftGenerator` 可用时自动开启 | 自动 ✅ |
| 🔬 **OEF 熵冻结** | `oef` 模块可用时自动开启并监控路由置信度 | 自动 ✅ |
| 🧩 **AFCE 锚点缓存** | `afce` 模块可用时自动开启 | 自动 ✅ |
| 🚀 **N-Gram 推测解码** | CPU Trie 自动构建并推测 | 自动 ✅ |
| 📊 **动态专家激活** | 根据 logits 置信度自动调整每 token 激活专家数 | 自动 ✅ |
| 📏 **动态块大小** | 根据 hidden_size 和可用显存自动选择最优块大小 | 自动 ✅ |
| 🔗 **超长上下文 SelfExtend** | 默认开启（4 行位置编码逻辑，无需任何配置） | 自动 ✅ |
| ⏱ **双 CUDA 流管线** | Prefill 流 + Decode 流自动配合 | 自动 ✅ |
| 🔎 **Goose 推测解码** | `goose_core` 可用时自动开启，根据 hidden_size 自动调整 draft 数 | 自动 ✅ |

> **不需要：** 修改任何配置文件、设置任何环境变量、提供任何调优参数。
> 也不需要在不同模型间切换不同配置——所有优化对 HF / GGUF / 任意尺寸模型一视同仁。

---

### 核心技术

- **增量 BLAKE2b 哈希链**：严格 `BLAKE2b(BLAKE2b(prev).digest() + token_bytes)`，非 `hashlib.update()`，保证 Radix 树可匹配任意前缀
- **显存感知容量计算**：`total_blocks = int(free_mem * 0.85 / (block_size * hidden_size * 4))`
- **复合键守卫 GC**：防止哈希重用导致的误删
- **双 CUDA 流管线**：Prefill 流 + Decode 流，主线程统一同步（防死锁）
- **引用计数驱逐**：每个匹配的自增引用，ref_count=0 时回收至空闲队列
- **纯 Python GGUF 解析器**：仅依赖 struct+mmap+PyTorch，无需 llama-cpp-python
- **原生量化加载**：Q4_0/Q8_0 纯 PyTorch bitwise 反量化，零 C 扩展
- **KV Cache 非对称量化**：Key→INT8 + **Value→INT4 位运算打包**，显存降至 FP16 的 37.5%（首尾各 8 个 token 保留 FP16 精度）
- **LRU 前缀匹配缓存**：`match_prefix` 增加 hash-based LRU（max 256 条目），重复前缀 **O(1)** 命中

---

## 🧮 集中式显存预算管理器 (`vram_budget.py`)

**启动时统一评估，运行时实时 OOM 防护。** 把所有吃显存的子系统拉到一张表上算总账，不再各自为政。

### 启动阶段：统一预算分配

```
当前可用显存
     │
     ├── 模型权重（估算）
     │
     └── 剩余显存
              ├── KV Cache        ← 45%
              ├── 专家缓存 (VRAM)  ← 25%
              ├── 激活/缓冲区      ← 15%
              └── 安全余量          ← 15% （永不触碰）
```

### 运行时：三级水位告警

| 水位线 | 剩余比例 | 自动措施 |
|--------|----------|----------|
| ✅ 健康 | > 15% | 无操作 |
| ⚠️ 偏低 | 8%–15% | 预填充块大小减半 |
| 🔴 紧张 | 4%–8%  | 强制 KV 压缩 + 增大压缩窗口 |
| 🆘 告急 | < 4%   | 批处理上限减半 + 块减半 + 强制压缩 |

### 零基础日志

```python
from vram_budget import VRAMBudget

# 启动时：自动打印完整报告
budget = VRAMBudget(hidden_size=4096, num_layers=32, num_experts=8, is_moe=True)
budget.log_status()     # ← 一行命令，输出完整显存预算报告

# 运行时：一行查当前显存
import logging
logging.info(VRAMBudget.log_runtime())  # → "VRAM: 6.2 GiB / 23.9 GiB 可用 (25.9%)"
```

---

## 📋 统一日志系统 (`moe_logger.py`)

**启动即得清晰日志，零基础用户无需任何配置。** 所有模块的日志经过统一管理，只展示关键里程碑，自动抑制内部模块的细碎信息。

### 三种日志级别

| 使用方式 | 显示内容 | 适用场景 |
|----------|----------|----------|
| 默认启动 | ✅ 启动徽标 · 优化组件 · 模型摘要 · VRAM 报告 · 引擎就绪 | 零基础用户 |
| `-v` / `--verbose` | + 模块 INFO 日志 · 缓存分配 · 解码细节 | 调试优化时 |
| `MOE_LOG_LEVEL=debug` | + 全部 DEBUG · CUDA Graph 编译 · 注意力算子 | 开发者调试 |

### 启动时看到的内容

```
  🚀  MoeOwner — MoE 异构推理引擎
  Python: 3.12  |  PyTorch: 2.6.0
  GPU:    NVIDIA A100  (72GiB/80GiB 可用)

  🔧 优化组件状态
  ─────────────────────────────────────────
    FlashAttention SDPA      ✅  强制 flash 版
    KV Cache (混合)          ✅  Paged + Radix + 非对称量化
    Goose 推测解码           ✅  已开启
    SelfExtend 超长上下文    ✅  默认开启
    ...
  ─────────────────────────────────────────

  🖥️  MoeOwner 显存预算报告  ← VRAMBudget
  ...

  📦 模型摘要
  ─────────────────────────────────────────
    路径/名称:   Qwen/Qwen2.5-1.5B-Instruct
    架构:        Dense | 28 层 | 2048 维
  ─────────────────────────────────────────

  ✅  MoeOwner 推理引擎就绪
  🌐  API: http://localhost:8000/v1/completions
```

### 运行时显存快照

```bash
# 一行查当前显存（适合嵌入循环日志）
from moe_logger import log_runtime_vram
log_runtime_vram()  # → 输出: 📊 6.2GiB/23.9GiB 可用 (26%)
```

---

## MoE 专属优化

### 1. 层次化专家缓存 (`expert_cache.py`)

| 层级 | 介质 | 容量 | 延迟 |
|------|------|------|------|
| L1 | GPU HBM | ~few GB | ~μs |
| L2 | CPU Pinned Memory | ~数十 GB | ~ms（后台异步传输） |

- LRU-Frequency-Reuse（LFRU）驱逐策略：综合访问频率、上次访问时间、重用距离
- 异步 D2H/H2D 传输，不阻塞 decode 流水线
- 支持引用计数，避免逐出正在使用的专家

### 2. 动态专家跳过 — SERE (`sere.py`)

- 基于 router logits 的 top-k 后重路由：部分 token 可跳过次要专家
- `min_experts` / `threshold` 双模式控制精度-效率平衡
- 零额外推理开销（纯 mask 操作）

### 3. N-Gram 推测解码 (`ngram_speculation.py`)

- CPU 端 Trie 树存储历史 N-Gram 频率
- 每步推测 3–5 个候选 token，批量验证
- 推测命中率 40–70%（取决于模型与任务）

### 4. 工具调用编排 (`tool_sink.py`)

**自包含的工具下沉框架**，允许推理引擎内生地调用内置工具并基于结果重新推理，无需外部编排层。

| 组件 | 功能 |
|------|------|
| `ToolContext` | 请求级生命周期容器：临时目录 + 内存键值对 + 消息历史队列 + `threading.RLock` 并发安全 + 销毁时进程/内存/磁盘同步清理 |
| `ToolScanner` | 逐字符状态机，检测 `[[tool_name(key=value, ...)]]` 标记（4 状态：TEXT→LEFT_BRACKET→LEFT_DOUBLE→SAW_CLOSE_BRACKET） |
| `ToolOrchestrator` | 编排引擎：提交推理请求 → 扫描工具标记 → 执行工具 → 重构历史（user + assistant含标记 + tool结果 + 空assistant）→ 重新推理（最多 3 轮）→ 最终 format_fixer 清洗 |

**7 项硬编码内置工具：**

| 工具 | 签名 | 实现 |
|------|------|------|
| `memo_set` | `(key, value) → str` | 工作区内存字典存储 |
| `memo_get` | `(key) → str` | 读取工作区内存 |
| `sci_calc` | `(expression) → str` | `math`+`cmath` 全量导出、AST 形状校验(≤2D/100×100)、线程超时(1s)、nan/inf→null |
| `sys_env` | `() → dict` | `os.uname`/`platform` 系统信息、CPU/内存、Windows 兼容回退 |
| `list_ports` | `() → list` | `/proc/net/tcp`→`ss`→`netstat` 多级降级、异常→空列表 |
| `ping_target` | `(host, port, timeout) → str(JSON)` | TCP 套接字探测、1 次重试、返回可达性+延迟 |
| `format_fixer` | `(raw) → str` | JSON 清洗（尾随逗号/引号归一/逐字符回退截断）——输出钩子双角色 |
| `sandbox_run` | `(code) → str` | `subprocess` 沙箱、白名单builtins(仅15个)、5s超时SIGKILL、64KB截断、Windows兼容 |
| `fetch_url` | `(url, max_chars) → str` | 协议强校验(http/https)、固定UA、Content-Type过滤、`html.parser` 文本提取、三重超时(DNS 5s+连接 5s+读取 10s) |

**约束：**
- 参数解析四阶段降级：`literal_eval` → `json.loads` → 自定义`key=value`解析器 → 位置回退
- 单请求硬上限 3 轮（含同一工具重复调用），超限返回 `LOOP_LIMIT_EXCEEDED`
- 单条 Tool 消息容量硬上限 100KB
- 输出最终过 `format_fixer()` 后钩子，修复失败截断至最后一个合法 JSON + `[FORMAT_TRUNCATED]`
- 全零第三方依赖（stdlib 仅限）

### 5. 调度器三阶段管线

```
输入 → [调度层] → [缓存层] → [推理层] → 输出
       预填充调度    KV 缓存管理    MoE 前向
       解码调度      LRU 驱逐      专家缓存
       推测解码      Radix 匹配    SERE 跳过
```

---

## 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.12+ | 推荐 3.12 |
| PyTorch | 2.6.0+cu124 | CUDA 12.4 |
| Transformers | 4.51.3 | HuggingFace 模型加载 |

```bash
# 安装
pip install torch==2.13.0
pip install transformers==5.14.1
```

> **注**：`cache_manager.py` 无需 GPU 即可运行单元测试（无 CUDA 时回退至 10000 块）

---

## 快速开始

### 1. 运行单元测试

```bash
cd MoeOwner
python3 -c "
from cache_manager import HybridCache

cache = HybridCache(block_size=16, hidden_size=4096, total_blocks=512)

# 基本分配
b1 = cache.allocate([101, 102, 103])
print(f'Allocated block {b1.block_id}')

# 前缀匹配
matched_id, remaining = cache.match_prefix([101, 102])
print(f'Matched block {matched_id}, remaining: {remaining}')

# 引用计数与回收
cache.free_block(b1.block_id)
print(f'Cache stats: {cache.stats()}')
"
```

### 2. GGUF 模型加载（纯 Python，零外部依赖）

```bash
# 加载 GGUF 模型（自动检测 .gguf 后缀）
python main.py --model /path/to/model.Q4_0.gguf

# 或通过 --gguf 显式指定
python main.py --gguf /path/to/model.Q4_0.gguf --block-size 32 --verbose
```

GGUF 加载器架构：

```
model_loader/
  __init__.py       — 公共 API: load_model(), GGUFFile
  gguf_reader.py    — 底层 GGUF v3 解析 + 反量化 kernel
  model_adapter.py  — 高层适配器 (GGUFModelAdapter)
  README.md         — 完整文档
```

支持格式：

| GGML 类型 | 状态 | 方式 |
|-----------|------|------|
| F32/F16 | ✅ 零拷贝 | `torch.frombuffer` |
| Q4_0 | ✅ 纯 PyTorch | 位移解包 → FP16 |
| Q8_0 | ✅ 纯 PyTorch | INT8 缩放 → FP16 |

---

## 架构设计

### 阶段流程

```
阶段 1: 环境锁定
  └─ Python 3.12 + torch 2.6 + transformers 4.51

阶段 2: 全局 Torch 配置（全部自动）
  ├─ Flash SDP 强制启用（禁用 math/mem_efficient 回退）
  ├─ TF32 matmul + cuDNN 允许
  ├─ float32 精度 = 'high'
  └─ cuDNN benchmark 自动调优

阶段 3: 模块组装
  ├─ attention_kernel.py ──── FlashAttentionKernel (torch.compile)
  ├─ cache_manager.py ─────── HybridCache (Paged + Radix + KV 非对称量化)
  ├─ scheduler.py ─────────── UnifiedScheduler (双流管线)
  ├─ expert_cache.py ──────── 层次化专家缓存
  ├─ sere.py ──────────────── 动态专家跳过
  ├─ ngram_speculation.py ─── N-Gram 推测解码
  ├─ long_context/ ────────── SelfExtend / ReAttention / YaRN（默认 SelfExtend 自动开启）
  └─ model_loader/ ────────── GGUF 加载 + PyTorch 量化

阶段 4: 模型注入（二选一）
  ├─ HuggingFace 路径: 加载 HF 模型 (fp16), 替换每层 self_attn → FlashAttentionKernel
  │                          （自动注入 SelfExtend / ReAttention 到 attention 前向）
  └─ GGUF 路径: 解析 GGUF 文件, 反量化权重, 构建 GGUFModelAdapter
  └─ 预热编译 → dummy_input 触发 JIT 静态图编译

阶段 5: 事件循环
  └─ 无限调度: step() → 预填充/解码/同步/GC
```

### 缓存结构

```
┌─────────────────────────────────────────────────────────────┐
│                    HybridCache                              │
│                                                             │
│  ┌─────────────────────┐    ┌───────────────────────────┐   │
│  │  free_block_queue   │    │     radix_index            │   │
│  │  (LIFO 空闲池)      │    │  hash(t1)        → block  │   │
│  │                     │    │  hash(t1+t2)     → block  │   │
│  │  [0, 1, 2, ...]     │    │  hash(t1+t2+t3)  → block  │   │
│  └─────────────────────┘    └───────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  allocated_blocks: { block_id → Block }              │   │
│  │  Block { phys_addr, ref_count, hash, next_block }   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 超长上下文扩展 (SelfExtend)

**自动开启，无需任何参数。** MoeOwner 集成了三种训练无关的上下文窗口扩展方法：

| 方法 | 适用模型 | 原理 |
|------|----------|------|
| 🥇 **SelfExtend**（默认） | RoPE 类模型（LLaMA/Qwen/Mistral/Gemma） | 4 行位置编码逻辑，邻近 token 保持原位置，远端 token floor-division 分组 |
| 🥈 **ReAttention** | 任意 Transformer（不依赖 RoPE） | 内容感知 top-k 检索 + 有限注意力 |
| 🥉 **YaRN** | HuggingFace 内置（`rope_scaling`） | 行业标准，零运行时开销 |

```bash
# SelfExtend —— 默认启用，什么都不用加
python main.py --model Qwen/Qwen2.5-32B

# 切换为 ReAttention
python main.py --model Qwen/Qwen2.5-32B --context-method reattention

# 使用 YaRN
python main.py --model Qwen/Qwen2.5-32B --context-method yarn --yarn-factor 16

# 关闭
python main.py --model Qwen/Qwen2.5-32B --disable-long-context
```

> 上下文短于 2048 tokens 时自动跳过（`short_context_threshold`），不影响短文本性能。

集成在 `long_context/` 模块中，与所有其他优化（KV cache 量化、Goose 推测解码、Self-Spec 骨架推测、AFCE、OEF、SERE）完全兼容。

---

### 双流管线

```
 时间线 →
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Prefill │    │ Prefill │    │ Prefill │  ← prefill_stream
│ Chunk 1 │    │ Chunk 2 │    │ Chunk 3 │
└─────────┘    └────┬────┘    └─────────┘
                     │
               ┌─────▼──────┐
               │  Decode 1  │                  ← decode_stream
               └────────────┘

  同步点: torch.cuda.synchronize()  ← 主线程（仅此处）
```

---

## 测试

```bash
# 运行完整测试
python3 -m pytest tests/ -v
```

当前通过 **58 项**自动化测试：

- ✅ 块分配与空闲队列管理
- ✅ 增量哈希链一致性
- ✅ Radix 前缀匹配（精确/部分/无匹配）
- ✅ 引用计数自增与递减
- ✅ 块驱逐与空闲回收
- ✅ GC 过期条目清理
- ✅ OOM 异常处理
- ✅ 复合键守卫防误删
- ✅ LRU 缓存命中（O(1) 返回，ref_count 递增）
- ✅ LRU 淘汰稳定（不超过 max）
- ✅ KV Cache 非对称量化 round-trip（Key INT8 + Value INT4 packed）
- ✅ KV Cache 形状减半正确性
- ✅ KV 首尾 FP16 保护（PROTECTED_N=8 边界测试）
- ✅ KV 混合模式 store/load 全链路（多 layer、覆写、释放）
- ✅ Key INT8 per-head 缩放维度与零值测试
- ✅ Value INT4 位运算 packed 确定性测试
- ✅ ruff 静态审查 0 错误

---

## 注意事项

> ⚠️ 以下为使用 MoeOwner 时需留意的已知约束与边界情况。**注意：README 中的前两条注解曾标记为"占位实现"，代码已于 v2 完成全链路集成，此处为最新状态说明。**

1. **KV Cache 全链路已集成** ✅：调度器 `_decode_step()` 从 `HybridCache.load_kv()` 读取缓存的 KV 张量，传入 `model.forward(past_key_values=...)`，解码后通过 `cache.store_kv()` 写回。每步 decode 仅传入 1 个 token + 其缓存的KV，实现 O(n) 逐 token 注意力，而非全量重算。完成请求自动 `free_block()` 回收。
2. **解码路径完整** ✅：`_decode_step()` 正确调用 `model.forward()` 获取 logits 并通过 `argmax` 采样下一个 token。`decode_req.step()` 仅用做步数计数器，不替代实际模型前向。
3. **CUDA 图捕获编译开销** 🔧：`torch.compile(mode="reduce-overhead")` 在首次运行时会有编译开销（通常数十秒），后续调用为静态图执行。可通过 `TORCH_COMPILE_DEBUG=1` 观察编译详情。
4. **Expert Cache 与 CUDA Graph 互斥** 🔒：专家缓存启用时自动禁用 CUDA Graph（`init_cuda_graphs()` 中检测），因为动态权重加载使静态图失效。
5. **模型适配层要求** 📋：当前 `model.forward()` 的 `use_cache=True` 路径依赖模型将新 KV 返回至 `model._last_kv_cache` 属性。HF Transformers 模型默认支持；自定义模型需包装此接口。

---

## 许可证

**CC BY-NC-SA 4.0**（署名-非商业性使用-相同方式共享 4.0 国际）

- ✅ **学习研究** — 欢迎
- ✅ **修改分发** — 允许，但须以相同协议共享
- ❌ **商业使用** — 禁止
- ✅ **贡献代码** — 提交者自动授权项目使用

**完整许可文本见 [LICENSE](./LICENSE)**

---

<a id="english-version"></a>

# MoeOwner — _English_

> **Heterogeneous MoE Inference Engine** — Integrating PagedAttention + RadixAttention KV Cache, Expert Offloading, SERE Dynamic Expert Skipping, N-Gram Speculative Decoding, and Training-Free Long Context Extension.
>
> _🏷️ Tags: MoE · LLM Inference · KV Cache · Speculative Decoding · Expert Offloading · GGUF · Pure Python · CUDA · Flash Attention · PyTorch · Long Context_

A production-oriented inference engine purpose-built for **Mixture-of-Experts** large language models. It fuses classic KV cache management with MoE-specific optimizations in a single, cohesive pipeline.

## 🎯 Zero Manual Tuning — Plug and Run

**All optimizations are auto-enabled — zero config, zero parameters, zero manual tuning.** Just `python main.py --model <some_model>` and everything works:

| Optimization | How It's Auto-Enabled | User Action |
|-------------|----------------------|-------------|
| 🔥 **FlashAttention SDPA** | Forces flash variant; disables math/mem_efficient fallback | Auto on `main.py` start ✅ |
| ⚡ **cuDNN benchmark** | `torch.backends.cudnn.benchmark = True` | Automatic ✅ |
| 🚀 **TF32 matmul** | Both matmul and cuDNN TF32 paths enabled (Ampere+) | Automatic ✅ |
| 🎯 **Float32 matmul precision** | `torch.set_float32_matmul_precision("high")` | Automatic ✅ |
| 📦 **torch.compile** | Auto-tries reduce-overhead → default fallback chain; gracefully skips on failure | Automatic ✅ |
| 🧊 **KV Cache asymmetric quantization** | Key→INT8 + Value→INT4 packed; head/tail 8-token FP16 protection; transparent gate | Automatic ✅ |
| 📏 **Dynamic block size** | Auto-optimized based on hidden_size and available GPU memory | Automatic ✅ |
| ⏱ **Adaptive KV compression** | Auto-tuned sink/window/importance params from model max_seq_len | Automatic ✅ |
| 🧠 **SERE dynamic expert skip** | Auto-tuned skip_threshold/min_experts from num_experts/top_k | Automatic ✅ |
| 🏗 **Hierarchical expert cache** | VRAM capacity auto-calculated from available GPU memory | Automatic ✅ |
| 🔄 **Self-speculative skeleton** | Auto-enabled when `SkeletonDraftGenerator` is available | Automatic ✅ |
| 🔬 **OEF entropy freeze** | Auto-enabled when `oef` module is available; monitors router confidence | Automatic ✅ |
| 🧩 **AFCE anchor cache** | Auto-enabled when `afce` module is available | Automatic ✅ |
| 🚀 **N-Gram speculation** | CPU Trie auto-built and queried | Automatic ✅ |
| 📊 **Dynamic expert activation** | Per-token k auto-adjusted by logit confidence | Automatic ✅ |
| 🔗 **Long context SelfExtend** | Default-on (4-line position-id injection, zero config needed) | Automatic ✅ |
| ⏱ **Dual CUDA stream pipeline** | Prefill + decode streams auto-coordinated | Automatic ✅ |
| 🔎 **Goose speculative decoding** | Auto-enabled when `goose_core` is available; draft count auto-tuned by model size | Automatic ✅ |

> **No need to:** edit config files, set environment variables, provide tuning parameters, or switch settings between different models. All optimizations work uniformly for HF models, GGUF models, and any model size.

## Key Features

| Module | File | Role |
|--------|------|------|
| 🔥 FlashAttention Kernel | `attention_kernel.py` | Optimized PyTorch SDPA + `torch.compile` attention |
| 📦 Hybrid Cache | `cache_manager.py` | PagedAttention physical blocks + RadixAttention hash index + KV asymmetric quantization |
| ⏱ Unified Scheduler | `scheduler.py` | Chunked Prefill + Decode dual-CUDA-stream pipeline |
| 🚀 Entrypoint | `main.py` | Global config, model injection, event loop, HTTP API server |
| 📖 GGUF Loader | `model_loader/` | Pure Python GGUF v3 parser — no llama-cpp-python required |
| 🧠 Expert Cache | `expert_cache.py` | Hierarchical LFRU expert weight offloading (DRAM + VRAM) |
| 🧩 Long Context | `long_context/` | SelfExtend / ReAttention / YaRN — training-free context window extension |
| ⚡ SERE | `sere.py` | Dynamic expert skipping via post-routing logit redirection |
| 🔮 N-Gram Speculation | `ngram_speculation.py` | CPU Trie-based draft generation + verification |

### Core Technologies

- **Incremental BLAKE2b hash chain** — `BLAKE2b(BLAKE2b(prev).digest() + token)` enables arbitrary prefix matching in the Radix tree
- **VRAM-aware capacity** — `total_blocks = int(free_mem * 0.85 / (block_size * hidden_size * 4))`
- **Compound-key GC guard** — prevents accidental eviction due to hash collisions
- **Dual CUDA streams** — prefill and decode run concurrently; main thread synchronizes at checkpoint
- **Reference-counted eviction** — each prefix match increments ref_count; blocks recycled at 0
- **Pure Python GGUF parser** — depends only on `struct` + `mmap` + PyTorch
- **Native quantization** — Q4_0/Q8_0 bitwise dequantization in pure PyTorch, zero C extensions
- **KV Cache asymmetric quantization** — Key→INT8 + Value→INT4 packed, reducing memory to 37.5% of FP16 (first/last 8 tokens preserved at FP16)
- **LRU prefix match cache** — hash-based LRU (max 256 entries), **O(1)** for repeated prefixes

## Quick Start

```bash
# Install dependencies
pip install torch==2.13.0
pip install transformers==5.14.1

# Run tests
python3 -m pytest tests/ -v

# Start inference engine with HuggingFace model (all optimizations auto-enabled)
python main.py --model Qwen/Qwen2.5-1.5B-Instruct --api-port 8000

# Or with a GGUF model (all optimizations auto-enabled)
python main.py --gguf /path/to/model.Q4_0.gguf --verbose

# Disable long context or speculation
python main.py --model Qwen/Qwen2.5-1.5B-Instruct --disable-long-context
python main.py --model Qwen/Qwen2.5-1.5B-Instruct --no-speculative
```

## Architecture Overview

```
Input → [Scheduler Layer] → [Cache Layer] → [Inference Layer] → Output
         Prefill Scheduling   KV Cache Mgmt     MoE Forward
         Decode Scheduling    LRU Eviction      Expert Cache
         Speculation          Radix Match       SERE Skip

Stream Pipeline:
  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ Prefill  │  │ Prefill  │  │ Prefill  │  ← prefill_stream
  │ Chunk 1  │  │ Chunk 2  │  │ Chunk 3  │
  └─────────┘  └────┬────┘  └─────────┘
                     │
               ┌─────▼──────┐
               │  Decode 1  │                ← decode_stream
               └────────────┘
      Sync: torch.cuda.synchronize() — main thread only
```

## Integration Notes

1. ✅ **KV Cache integration**: Fully wired. `_decode_step()` loads KV from `HybridCache.load_kv()`, passes to `model.forward(past_key_values=...)`, and stores updated KV via `cache.store_kv()`. Each decode step processes 1 token with O(n) attention using cached KV.
2. ✅ **Decode path**: Complete. `model.forward()` is called per step with logits extracted via `argmax`. `decode_req.step()` is purely a step counter.
3. 🔧 **CUDA Graph warmup**: `torch.compile(mode="reduce-overhead")` incurs ~tens-of-seconds compilation on first run; subsequent calls are static graph execution. Set `TORCH_COMPILE_DEBUG=1` for compilation details.
4. 🔒 **Expert Cache ↔ CUDA Graph mutual exclusion**: When expert cache is active, CUDA Graphs are automatically disabled (`init_cuda_graphs()` detects this), because dynamic weight loading invalidates static graphs.
5. 📋 **Model adapter requirement**: `use_cache=True` path requires the model to store new KV pairs on `model._last_kv_cache`. HF Transformers support this natively; custom models need a thin wrapper.

## License

**CC BY-NC-SA 4.0** — Attribution-NonCommercial-ShareAlike 4.0 International

- ✅ **Research & learning** — Welcome
- ✅ **Modification & distribution** — Permitted under same license
- ❌ **Commercial use** — Prohibited
- ✅ **Contributions** — Contributors automatically license their work for project use

Full license text: [LICENSE](./LICENSE)
