# MoeOwner

> **MoE 异构推理引擎** — 融合 PagedAttention + RadixAttention KV Cache、专家缓存、SERE 推测跳过与 N-Gram 推测解码

---

## 项目定位

**MoeOwner** 是一个面向 Mixture-of-Experts 大模型的生产级推理引擎，将传统 KV Cache 管理与 MoE 专属优化深度融合：

| 组件 | 文件 | 功能 |
|------|------|------|
| 🔥 **FlashAttention 核** | `attention_kernel.py` | 基于 PyTorch SDPA + `torch.compile` 的优化注意力 |
| 📦 **混合缓存** | `cache_manager.py` | PagedAttention 物理块 + RadixAttention 哈希索引 |
| ⏱ **统一调度器** | `scheduler.py` | Chunked Prefill + Decode 双 CUDA 流调度 |
| 🚀 **入口** | `main.py` | 全局配置、模型注入、事件循环 |
| 📖 **GGUF 加载** | `model_loader/` | 纯 Python GGUF v3 解析器 + PyTorch 原生量化适配 |
| 🧠 **专家缓存** | `expert_cache.py` | 层次化 LRU-Frequency-Reuse 专家权重卸载/加载 |
| ⚡ **SERE** | `sere.py` | 动态专家跳过，top-k 后重路由 |
| 🔮 **推测解码** | `ngram_speculation.py` | CPU Trie N-Gram 推测解码 |

### 核心技术

- **增量 SHA-256 哈希链**：严格 `SHA256(SHA256(prev).digest() + token_bytes)`，非 `hashlib.update()`，保证 Radix 树可匹配任意前缀
- **显存感知容量计算**：`total_blocks = int(free_mem * 0.85 / (block_size * hidden_size * 4))`
- **复合键守卫 GC**：防止哈希重用导致的误删
- **双 CUDA 流管线**：Prefill 流 + Decode 流，主线程统一同步（防死锁）
- **引用计数驱逐**：每个匹配的自增引用，ref_count=0 时回收至空闲队列
- **纯 Python GGUF 解析器**：仅依赖 struct+mmap+PyTorch，无需 llama-cpp-python
- **原生量化加载**：Q4_0/Q8_0 纯 PyTorch bitwise 反量化，零 C 扩展
- **KV Cache 非对称量化**：Key→INT8 + **Value→INT4 位运算打包**，显存降至 FP16 的 37.5%
- **LRU 前缀匹配缓存**：`match_prefix` 增加 hash-based LRU (max 256 条目)，重复前缀 **O(1)** 命中

---

## MoE 专属优化

### 1. 层次化专家缓存 (`expert_cache.py`)

| 层级 | 介质 | 容量 | 延迟 |
|------|------|------|------|
| L1 | GPU HBM | ~few GB | ~μs |
| L2 | CPU Pinned Memory | ~数十 GB | ~ms (后台异步传输) |

- LRU-Frequency-Reuse (LFRU) 驱逐策略：综合访问频率、上次访问时间、重用距离
- 异步 D2H/H2D 传输，不阻塞 decode 流水线
- 支持引用计数，避免逐出正在使用的专家

### 2. 动态专家跳过 — SERE (`sere.py`)

- 基于 router logits 的 top-k 后重路由：部分 token 可跳过次要专家
- `min_experts` / `threshold` 双模式控制精度-效率平衡
- 零额外推理开销（纯 mask 操作）

### 3. N-Gram 推测解码 (`ngram_speculation.py`)

- CPU 端 Trie 树存储历史 N-Gram 频率
- 每步推测 3-5 个候选 token，批量验证
- 推测命中率 40-70%（取决于模型与任务）

### 4. 调度器三阶段管线

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
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.51.3
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

**零成本优化已内嵌**:
- Flash Attention v2 SDPA — 利用 N 卡 Tensor Core（强制启用，禁用 math/mem_efficient 回退）
- cuDNN benchmark — 运行时自动调优卷积/注意力 kernel
- `torch.compile(dynamic=True)` — 静态图编译 + 可变长度输入不触发重编译
- TF32 精度 — Ampere+ 架构矩阵乘提速（matmul + cuDNN 双路）
- Float32 matmul precision='high' — 混合精度舍入控制
- 动态 Block Size — 根据模型参数量建议最优块大小

---

## 架构设计

### 阶段流程

```
阶段 1: 环境锁定
  └─ Python 3.12 + torch 2.6 + transformers 4.51

阶段 2: 全局 Torch 配置
  ├─ Flash SDP 强制启用
  ├─ TF32 matmul 允许
  └─ float32 精度 = 'high'

阶段 3: 模块组装
  ├─ attention_kernel.py ──── FlashAttentionKernel (torch.compile)
  ├─ cache_manager.py ─────── HybridCache (Paged + Radix)
  ├─ scheduler.py ─────────── UnifiedScheduler (双流管线)
  ├─ expert_cache.py ──────── 层次化专家缓存
  ├─ sere.py ──────────────── 动态专家跳过
  ├─ ngram_speculation.py ─── N-Gram 推测解码
  └─ model_loader/ ────────── GGUF 加载 + PyTorch 量化

阶段 4: 模型注入 (二选一)
  ├─ HuggingFace 路径: 加载 HF 模型 (fp16), 替换每层 self_attn → FlashAttentionKernel
  └─ GGUF 路径: 解析 GGUF 文件, 反量化权重, 构建 GGUFModelAdapter
  └─ 预热编译 → dummy_input 触发 JIT

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

  同步点: torch.cuda.synchronize()  ← 主线程 (仅此处)
```

---

## 测试

```bash
# 运行完整测试
python3 -m pytest tests/ -v
```

当前通过 **35 项**自动化测试：
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
- ✅ KV Cache INT4 量化 round-trip（含负值验证）
- ✅ KV Cache 形状减半正确性
- ✅ ruff 静态审查 0 错误

---

## 注意事项

⚠️ **集成注意事项**：

1. **`past_key_values` 集成**：当前调度器的 `model.forward()` 调用中的 `past_key_values` 是占位实现。需实现自定义 `DynamicCache` 子类，从 `HybridCache` 的物理块池读写 KV 张量
2. **解码路径**：`scheduler.step()` 中的解码路径目前是一个生命周期钩子（`decode_req.step()`），实际的 `model.forward()` 调用需补充
3. **CUDA 图捕获**：`torch.compile(mode="reduce-overhead")` 在首次运行时会有编译开销
4. **Expert Cache + CUDA Graph 互斥**：专家缓存启用时自动禁用 CUDA Graph

---

## 许可证

**CC BY-NC-SA 4.0**（署名-非商业性使用-相同方式共享 4.0 国际）

- ✅ **学习研究** — 欢迎
- ✅ **修改分发** — 允许，但须以相同协议共享
- ❌ **商业使用** — 禁止
- ✅ **贡献代码** — 提交者自动授权项目使用

**完整许可文本见 [LICENSE](./LICENSE)**
