"""
engine_logger.py — PyDense 统一日志系统

集中管理全系统日志输出，自动抑制内部模块的细碎日志，仅展示里程碑事件。
零基础用户无需任何配置即得清晰日志。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 使用方式
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 1. 零基础：默认启动即得清晰日志
    python main.py --model Qwen/Qwen2.5-1.5B-Instruct

    # 2. 想看更多细节
    python main.py --model Qwen/Qwen2.5-1.5B-Instruct -v          # 显示模块 INFO 日志
    python main.py --model Qwen/Qwen2.5-1.5B-Instruct --verbose   # 同上

    # 3. 调试模式（所有 DEBUG 日志）
    export MOE_LOG_LEVEL=debug
    python main.py --model Qwen/Qwen2.5-1.5B-Instruct

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔄 模块内调用
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    from engine_logger import log

    log.info("常规事件")
    log.warning("需要注意")
    log.error("出错")
    log.debug("调试细节（仅 --verbose 或 MOE_LOG_LEVEL=debug 时显示）")
"""

from __future__ import annotations

import logging
import os
import sys
import time

# ── 日志格式 ──────────────────────────────────────────────────────────
_FMT = "%(asctime)s | %(levelname)-5s | %(message)s"
_DATEFMT = "%H:%M:%S"


# ── 模块级别映射：默认 INFO，可调 ────────────────────────────────────
_DEFAULT_LEVELS: dict[str, int] = {
    "root": logging.INFO,
    "engine": logging.INFO,
    "scheduler": logging.INFO,
    "cache_manager": logging.WARNING,
    "vram_budget": logging.INFO,
    "goose_core": logging.WARNING,
    "tool_sink": logging.WARNING,
    "api_server": logging.INFO,
    "attention_kernel": logging.WARNING,
    "urllib3": logging.WARNING,
    "huggingface_hub": logging.WARNING,
    "filelock": logging.WARNING,
    "transformers": logging.WARNING,
}

# ── 环境变量覆盖 ─────────────────────────────────────────────────────
def apply_module_levels() -> None:
    """应用模块日志级别映射。"""
    level_name = os.environ.get("MOE_LOG_LEVEL", "").upper()
    root_level = logging.DEBUG if level_name == "DEBUG" else logging.INFO

    logging.basicConfig(level=root_level, format=_FMT, datefmt=_DATEFMT)

    for mod_name, level in _DEFAULT_LEVELS.items():
        if level_name == "DEBUG":
            level = logging.DEBUG
        logger_mod = logging.getLogger(mod_name)
        logger_mod.setLevel(level)
        logger_mod.propagate = True

    if level_name == "DEBUG":
        logging.getLogger("engine").info("MOE_LOG_LEVEL=debug → 全部 DEBUG 级别")


def get_logger(name: str) -> logging.Logger:
    """获取统一日志器。"""
    return logging.getLogger(name)


# ═══════════════════════════════════════════════════════════════════════════
# 日志速记函数（来自旧版 moe_logger.log）
# ── 用法： from engine_logger import log; log.info("...")
# ═══════════════════════════════════════════════════════════════════════════

# 默认日志器，模块直接 from engine_logger import log
log = logging.getLogger("engine")


# ═══════════════════════════════════════════════════════════════════════════
# 打印函数
# ═══════════════════════════════════════════════════════════════════════════

def print_banner() -> None:
    """打印启动横幅。"""
    banner = f"""
{'=' * 60}
  PyDense — 稠密模型推理引擎
  Dense Transformer Inference with FlashAttention + KV Cache + Speculative Decoding
{'=' * 60}
"""
    print(banner, file=sys.stderr)


def print_optimizations(
    *,
    hf_ok: bool = True,
) -> None:
    """打印已启用的优化列表。"""
    leader = "  •"
    log.info("━" * 60)
    log.info("🛠️  运行优化")
    log.info("%s FlashAttention (SDPA flash)", leader)
    log.info("%s torch.compile (reduce-overhead)", leader)
    log.info("%s TF32 cuDNN + matmul", leader)
    log.info("%s Dual CUDA Stream pipeline (prefill | decode)", leader)
    log.info("%s HybridCache (PagedAttention + RadixAttention)", leader)
    log.info("%s Chunked Prefill (Sarathi-style)", leader)
    log.info("%s Adaptive KV Compression (H2O + StreamingLLM)", leader)
    log.info("%s Goose speculative decoding", leader)
    log.info("%s CUDA graphs + static memory", leader)
    log.info("━" * 60)


def print_model_summary(
    *,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    vocab_size: int,
    max_seq_len: int,
    model_name: str = "",
) -> None:
    """打印模型摘要。"""
    leader = "  •"
    log.info("━" * 60)
    log.info("📦 模型信息")
    if model_name:
        log.info("%s 模型: %s", leader, model_name)
    log.info(
        "%s %d 层 · %d 维 · %d 头 · %d KV 头 · %d vocab · %d max_seq",
        leader,
        num_layers, hidden_size, num_heads, num_kv_heads,
        vocab_size, max_seq_len,
    )
    log.info("━" * 60)


def print_engine_ready(
    *,
    model_name: str = "",
    port: int = 8080,
    mirror: object = None,
) -> None:
    """打印引擎就绪消息。"""
    mirror_str = f" ({mirror})" if mirror else ""
    log.info("━" * 60)
    log.info("🚀 引擎就绪%s", mirror_str)
    log.info("   模型: %s", model_name)
    log.info("   HTTP: http://localhost:%d", port)
    log.info("   聊天: python main.py --model %s chat", model_name)
    log.info("━" * 60)


def print_request_summary(
    *,
    prompt_len: int,
    gen_len: int,
    prefill_tokens: int,
    gen_tokens: int,
    elapsed: float,
) -> None:
    """打印单次请求性能摘要。"""
    prefill_tps = prefill_tokens / max(elapsed, 0.001)
    decode_tps = gen_tokens / max(elapsed, 0.001)
    total_tokens = prefill_tokens + gen_tokens

    log.info("━" * 60)
    log.info("📊 请求摘要")
    log.info("   Prompt: %d tokens  |  生成: %d tokens  |  合计: %d tokens",
             prompt_len, gen_len, total_tokens)
    log.info("   耗时: %.2f s", elapsed)
    log.info("   Prefill 吞吐: %.1f tok/s", prefill_tps)
    log.info("   Decode 吞吐:  %.1f tok/s", decode_tps)
    log.info("   总吞吐:      %.1f tok/s", total_tokens / max(elapsed, 0.001))
    log.info("━" * 60)


# ── 其他辅助 ──────────────────────────────────────────────────────────
def log_runtime_vram(logger_obj: logging.Logger | None = None) -> None:
    """打印当前显存使用情况。"""
    if not torch.cuda.is_available():
        return
    _LOGGER = logger_obj or log
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    _LOGGER.info("VRAM: allocated=%.2f GiB, reserved=%.2f GiB", allocated, reserved)
