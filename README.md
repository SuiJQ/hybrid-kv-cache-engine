# PyDense

> **稠密模型推理引擎** — 纯 Transformers 融合 FlashAttention + PagedAttention/RadixAttention KV Cache + Goose 推测解码
>
> _🏷️ LLM Inference · KV Cache · Flash Attention · Speculative Decoding · Self-Spec (ACL'24) · Long Context · Pure PyTorch_

---

<p align="center">
  🌐 <strong>中文</strong> · <a href="#english-version"><strong>English</strong></a>
</p>

---

## 项目定位

**PyDense** 是一个面向**稠密（Dense）Transformer 模型**的生产级推理引擎，基于 HuggingFace Transformers + BitsAndBytes 加载模型，融合以下核心技术：

| 组件 | 文件 | 功能 |
|------|------|------|
| 🔥 **FlashAttention 核** | `attention_kernel.py` | 基于 PyTorch SDPA + `torch.compile` 优化注意力 |
| 📦 **混合 KV 缓存** | `cache_manager.py` | PagedAttention 物理块 + RadixAttention 哈希索引 + 前缀缓存 |
| 🧮 **显存预算** | `vram_budget.py` | 集中式显存预算管理器，启动评估 + 运行时 OOM 防护 |
| ⏱ **统一调度器** | `scheduler.py` | Chunked Prefill + Decode 双 CUDA 流调度 |
| 🚀 **入口** | `main.py` | 全局配置、多镜像源下载、模型加载、事件循环 |
| 🔮 **推测解码** | `goose_core.py` + `scheduler.py` | Goose PLD 线性链 + 树注意力 + Self-Spec 骨架推测（全部自动开启） |
| 🛠️ **工具下沉** | `tool_sink.py` | 模型内生工具调用框架 — 状态机扫描 `[[tool(...)]]` 标记、7 项内置工具、3 轮循环编排 |
| 🔗 **HTTP API** | `api_server.py` | OpenAI 兼容异步 HTTP 服务，支持流式/非流式、推理内容输出 |

> **关键词**：稠密模型推理 | FlashAttention | 混合 KV 缓存 | 推测解码 | 超长上下文 | 双流管线 | 显存预算 | PyTorch | Transformers | BitsAndBytes | 工具下沉

---

## 🎯 零手动调优 — 插电即用

**PyDense 的所有优化全部自动开启，零参数、零配置、零手动调优。** 用户只需 `python main.py --model Qwen/Qwen2.5-7B-Instruct`，其余全部自动完成：

| 优化项 | 自动生效方式 |
|--------|-------------|
| 🔥 **FlashAttention SDPA** | 强制启用 flash 版，禁用 math/mem_efficient 回退 |
| ⚡ **cuDNN benchmark** | `torch.backends.cudnn.benchmark = True` |
| 🚀 **TF32 矩阵乘** | matmul + cuDNN 双路允许 TF32（Ampere+ 架构）|
| 📦 **torch.compile 静态图编译** | 自动尝试 reduce-overhead → default 降级链 |
| 🧊 **KV Cache** | PagedAttention + RadixAttention 混合索引 |
| 📐 **分块预填充** | Sarathi-style chunked prefill，自动调优块大小 |
| 🧠 **自适应 KV 压缩** | H2O + StreamingLLM 混合，自动调优 sink/window 参数 |
| 🚀 **Goose 推测解码** | PLD 模式匹配 + 树注意力，根据 hidden_size 自动调参 |
| 🔄 **Self-Spec 骨架推测** | 跳层草稿生成 + 全模型验证（ACL'24） |
| 📊 **双 CUDA 流管线** | Prefill 流 + Decode 流自动配合 |
| 🧮 **显存预算管理** | 启动时精确评估、运行时实时水位监控、自动降级 |
| 🌐 **多镜像源下载** | 自动检测国内环境，fallback hf-mirror |
| 📦 **BitsAndBytes 量化** | 可选 `--quantize 4bit` 或 `--quantize 8bit` |

> 不需要修改任何配置文件、设置任何环境变量（镜像源自动选择）、提供任何调优参数。

---

## 🚀 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 基础用法

```bash
# 默认模型 Qwen2.5-7B-Instruct（自动下载）
python main.py

# 指定其它模型
python main.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# 国内用户自动使用 hf-mirror 镜像
python main.py --model Qwen/Qwen2.5-7B-Instruct --mirror hf-mirror
```

### 量化模式

```bash
# 4-bit 量化（适合 24G 显存跑 70B 模型）
python main.py --model Qwen/Qwen2.5-72B-Instruct --quantize 4bit

# 8-bit 量化
python main.py --model Qwen/Qwen2.5-14B-Instruct --quantize 8bit
```

### 交互式聊天

```bash
python main.py --model Qwen/Qwen2.5-7B-Instruct chat
```

启动后进入类似 `ollama run` 的终端聊天界面：
- `>>>` 输入提示
- `Ctrl+D` 或 `/exit` 退出
- `/clear` 清空对话历史

### 启动器（推荐新手）

```bash
# 交互式菜单
python launch.py

# 一键启动（带基准测试）
python launch.py --auto

# 跳过测试，直接启动
python launch.py --quick
```

### HTTP API 服务

```bash
# 默认监听 0.0.0.0:8080
python main.py --model Qwen/Qwen2.5-7B-Instruct

# 自定义端口
python main.py --model Qwen/Qwen2.5-7B-Instruct --port 8000
```

API 端点：

```bash
# 文本补全
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好", "max_tokens": 100, "stream": false}'

# 聊天补全
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "你好"}], "stream": true}'
```

### 基准测试

```bash
# 默认 128 prompt + 128 gen
python main.py --model Qwen/Qwen2.5-1.5B-Instruct --benchmark

# 长文本测试
python main.py --model Qwen/Qwen2.5-1.5B-Instruct --benchmark --prompt-len 2048 --gen-len 256
```

---

## 📦 支持的模型

任何 HuggingFace Transformers 支持的**稠密因果语言模型**：

- **Qwen 系列**: Qwen2.5-{0.5B/1.5B/7B/14B/32B/72B}-Instruct
- **DeepSeek**: DeepSeek-R1-Distill-Qwen-{7B/14B/32B}
- **Llama 3**: Meta-Llama-3.1-{8B/70B}-Instruct
- **Mistral**: Mistral-7B-Instruct-v0.3, Mixtral (MoE 模型不支持)
- **Gemma**: Gemma-2-{2B/9B/27B}-it
- **Phi**: Phi-3/Phi-3.5 系列
- **其他**: 任何 `AutoModelForCausalLM` 支持的模型

> **注意**: PyDense 专为**稠密 Transformer 模型**设计，不兼容 MoE 模型（如 Mixtral、DeepSeek-V2/V3、Qwen2-MoE）。

---

## 🏗 项目架构

```
main.py                  ← 入口：多镜像下载 + 模型加载 + CLI
launch.py                ← 启动器（新手友好，交互式菜单）
engine_logger.py         ← 统一日志系统
attention_kernel.py      ← FlashAttention 编译核
cache_manager.py         ← 混合 KV 缓存（PagedAttention + RadixAttention）
scheduler.py             ← 统一调度器（预填充/解码/推测）
vram_budget.py           ← 显存预算管理
goose_core.py            ← 推测解码引擎（PLD + 树注意力）
tool_sink.py             ← 工具调用框架
api_server.py            ← OpenAI 兼容 HTTP API

long_context/            ← 超长上下文扩展（SelfExtend / YaRN）
```

---

## 🔧 高级配置

### 镜像源

镜像优先级：1) `--mirror` 参数 2) `HF_ENDPOINT` 环境变量 3) 自动探测（国内 → hf-mirror）

```bash
# 内置镜像
python main.py --model Qwen/Qwen2.5-7B-Instruct --mirror hf-mirror
python main.py --model Qwen/Qwen2.5-7B-Instruct --mirror huggingface

# 自定义 endpoint
export HF_ENDPOINT=https://your-mirror.example.com
python main.py --model Qwen/Qwen2.5-7B-Instruct
```

### 日志级别

```bash
# 更多细节
python main.py --model Qwen/Qwen2.5-7B-Instruct -v

# 调试模式
export MOE_LOG_LEVEL=debug
python main.py --model Qwen/Qwen2.5-7B-Instruct
```

---

## 📊 性能特性

| 特性 | 效果 |
|------|------|
| FlashAttention SDPA | 2-4x 注意力加速 |
| torch.compile | 10-30% 端到端加速 |
| KV Cache | 解码 O(n) → O(1) 注意力计算 |
| Chunked Prefill | 消除首 token 长等待 |
| Goose 推测解码 | 10-40% 解码加速 |
| 自适应 KV 压缩 | 长上下文显存减半 |
| 双 CUDA 流 | 预填充与解码并行 |

---

## English Version

### Quick Start

```bash
pip install -r requirements.txt

# Default: Qwen2.5-7B-Instruct
python main.py

# Interactive chat
python main.py --model Qwen/Qwen2.5-7B-Instruct chat

# Quantization for larger models
python main.py --model Qwen/Qwen2.5-72B-Instruct --quantize 4bit
```

### Core Architecture

PyDense is a dense Transformer inference engine built on HuggingFace Transformers:

- **KV Cache**: PagedAttention + RadixAttention hybrid
- **Scheduler**: Chunked prefill + decode with dual CUDA streams
- **Speculation**: Goose PLD + self-speculative (ACL'24 skip-layer)
- **Memory**: VRAM budget manager with auto-scaling

### Supported Models

All dense causal LMs from HuggingFace: Qwen2.5, Llama 3.1, DeepSeek-R1-Distill, Mistral, Gemma 2, Phi-3, etc.

**Note**: Does NOT support MoE models (Mixtral, DeepSeek-V2/V3, Qwen2-MoE).

---

## License

MIT
