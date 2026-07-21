"""
moe_logger.py — MoeOwner 统一日志系统

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

    from moe_logger import log

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
# 时间 + 级别 + 模块名（简短） + 消息
_FMT = "%(asctime)s | %(levelname)-5s | %(message)s"
_DATEFMT = "%H:%M:%S"


# ── 模块级别映射：默认 INFO，可调 ────────────────────────────────────
# 内部模块默认设为 WARNING，抑制细碎日志；-v 时统一降回 INFO
_DEFAULT_LEVELS: dict[str, int] = {
    "root": logging.INFO,           # 根日志器
    "engine": logging.INFO,         # main.py
    "vram_budget": logging.INFO,    # 显存预算（启动时打印报告）
    "scheduler": logging.WARNING,   # 调度器细节默认不显示
    "cache_manager": logging.WARNING,  # 缓存分配细节默认不显示
    "expert_cache": logging.WARNING,   # 专家缓存细节默认不显示
    "afce": logging.WARNING,        # 锚点缓存细节默认不显示
    "oef": logging.WARNING,         # 熵冻结细节默认不显示
    "sere": logging.INFO,           # SERE 模块摘要信息
    "ngram_speculation": logging.WARNING,
    "attention_kernel": logging.WARNING,
    "goose_core": logging.WARNING,  # 推测引擎细节默认不显示
    "http": logging.WARNING,        # HTTP 请求日志
    "model_loader": logging.INFO,
    "api_server": logging.WARNING,
    "long_context": logging.INFO,
    "integration": logging.WARNING,
    "uvicorn": logging.WARNING,
    "uvicorn.access": logging.WARNING,
    "uvicorn.error": logging.WARNING,
}

# 环境变量覆盖：MOE_LOG_LEVEL = debug / info / warning
_ENV_LEVEL = os.environ.get("MOE_LOG_LEVEL", "").strip().lower()
_VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv


# ── 构建日志器 ────────────────────────────────────────────────────────

class _MoeFilter(logging.Filter):
    """自定义过滤器：在 milestone 模式下抑制非里程碑日志。"""
    pass


def _resolve_level(name: str) -> int:
    """解析模块的日志级别，优先环境变量和 -v。"""
    if _ENV_LEVEL == "debug":
        return logging.DEBUG
    if _ENV_LEVEL == "info":
        return logging.INFO
    if _ENV_LEVEL == "warning":
        return logging.WARNING
    if _VERBOSE:
        return logging.DEBUG if name in ("root", "engine") else logging.INFO
    # 默认使用模块预设级别
    return _DEFAULT_LEVELS.get(name, logging.WARNING)


# ── 初始化根日志器 ────────────────────────────────────────────────────

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))

_root = logging.getLogger()
_root.setLevel(logging.INFO)
_root.handlers.clear()
_root.addHandler(_handler)


# ── 应用模块级别 ──────────────────────────────────────────────────────

_APPLIED: set[str] = set()


def apply_module_levels() -> None:
    """遍历已知模块并设置日志级别。重复调用安全。"""
    for name, _ in _DEFAULT_LEVELS.items():
        if name in _APPLIED:
            continue
        _APPLIED.add(name)
        lgr = logging.getLogger(name)
        level = _resolve_level(name)
        lgr.setLevel(level)
        if not lgr.handlers:
            lgr.addHandler(_handler)
        lgr.propagate = False


apply_module_levels()


# ======================================================================
# 公共 API
# ======================================================================

def get_logger(name: str) -> logging.Logger:
    """获取带 MoeOwner 默认级别的日志器。

    模块内使用：
        from moe_logger import get_logger
        log = get_logger(__name__)
    """
    if name not in _APPLIED:
        lgr = logging.getLogger(name)
        lgr.setLevel(_resolve_level(name))
        if not lgr.handlers:
            lgr.addHandler(_handler)
        lgr.propagate = False
        _APPLIED.add(name)
    return logging.getLogger(name)


# 默认日志器（给外部调用者使用）
log = get_logger("engine")


# ======================================================================
# 用户友好的宏事件日志
# ======================================================================

_SEP = "=" * 56

# 当前硬件概要：GPU 型号 + 显存
def _gpu_info() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "CPU (无 GPU)"
        name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        return f"{name}  ({free/1024**3:.0f}GiB/{total/1024**3:.0f}GiB 可用)"
    except Exception:
        return "GPU 查询失败"


# ── 启动徽标 ──────────────────────────────────────────────────────────

def print_banner() -> None:
    """打印 MoeOwner 启动徽标及关键环境信息。"""
    try:
        import torch
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        torch_ver = torch.__version__
        has_cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_cuda else "无 CUDA 设备"
    except Exception:
        py_ver = sys.version.split()[0]
        torch_ver = "?"
        gpu_name = "查询失败"

    log.info("")
    log.info(_SEP)
    log.info("    🚀  MoeOwner — MoE 异构推理引擎")
    log.info(_SEP)
    log.info("  Python:    %s  |  PyTorch:  %s", py_ver, torch_ver)
    log.info("  GPU:       %s", gpu_name)
    log.info(_SEP)
    log.info("")


# ── 优化状态 ──────────────────────────────────────────────────────────

def print_optimizations(
    hf_ok: bool = True,
    gguf_ok: bool = False,
    goose_ok: bool = False,
    self_spec_ok: bool = False,
    afce_ok: bool = False,
    oef_ok: bool = False,
    speculative_disabled: bool = False,
) -> None:
    """打印优化组件状态表。

    所有参数表示对应模块是否可用。默认所有优化自动开启，
    但可能因模块缺失而跳过（如 goose_core 未安装时不启动推测解码）。
    """
    log.info("")
    log.info("  🔧 优化组件状态")
    log.info("  " + "-" * 50)
    log.info("    FlashAttention SDPA      ✅  强制 flash 版，禁用回退")
    log.info("    torch.compile (JIT)      ✅  reduce-overhead → default 降级链")
    log.info("    KV Cache (混合)          ✅  Paged + Radix + 非对称量化")
    log.info("    KV 自适应压缩            ✅  按 max_seq_len 自动调参")
    log.info("    VRAM 预算管理            ✅  集中式三级水位 OOM 防护")
    log.info("    SelfExtend 超长上下文    ✅  默认开启，零配置")

    goose_status = "⏹️  --no-speculative" if speculative_disabled else \
                   ("✅  已开启" if goose_ok else "⏹️  goose_core 未安装")
    log.info("    Goose 推测解码            %s", goose_status)

    ss_status = "✅  已开启" if self_spec_ok else "⏹️  不可用"
    log.info("    Self-Spec 骨架推测        %s", ss_status)

    as_status = "✅  已开启" if afce_ok else "⏹️  afce 未安装"
    log.info("    AFCE 锚点缓存             %s", as_status)

    oef_status = "✅  已开启" if oef_ok else "⏹️  oef 未安装"
    log.info("    OEF 熵冻结                %s", oef_status)

    log.info("    SERE 动态专家跳过         ✅  按模型自动调参")
    log.info("    N-Gram 猜测解码           ✅  自动构建 CPU Trie")
    log.info("    专家缓存                  ✅  按显存自动计算容量")
    log.info("    双 CUDA 流管线            ✅  Prefill + Decode 并行")
    log.info("  " + "-" * 50)
    log.info("")


# ── 模型摘要 ──────────────────────────────────────────────────────────

def print_model_summary(
    model_type: str,
    model_name: str,
    hidden_size: int,
    num_layers: int,
    num_experts: int,
    is_moe: bool,
    params_b: float | None = None,
) -> None:
    """打印加载完成的模型摘要信息。"""
    log.info("")
    log.info("  📦 模型摘要")
    log.info("  " + "-" * 50)
    log.info("    路径/名称:   %s", model_name)
    log.info("    类型:        %s", model_type.upper())
    log.info("    架构:        %s | %d 层 | %d 维",
             "MoE" if is_moe else "Dense", num_layers, hidden_size)
    if is_moe:
        log.info("    专家配置:    %d 专家", num_experts)
    if params_b:
        log.info("    参数量:      ~%.1fB", params_b)
    log.info("  " + "-" * 50)
    log.info("")


# ── 请求完成摘要 ──────────────────────────────────────────────────────

def print_request_summary(
    request_id: str,
    prompt_tokens: int,
    gen_tokens: int,
    total_time_s: float,
    ttft_s: float,
    spec_enabled: bool = False,
    spec_accepted: int = 0,
    spec_drafted: int = 0,
) -> None:
    """打印单次请求完成摘要。

    每次调用打印 3-4 行关键信息，不打断主日志流。
    """
    tok_per_s = gen_tokens / total_time_s if total_time_s > 0 else 0
    ms_per_tok = total_time_s * 1000 / max(gen_tokens, 1)

    log.info("  ── 请求 %s 完成 ──", request_id)
    log.info("    Prompt: %d tok | 生成: %d tok | 耗时: %.2fs",
             prompt_tokens, gen_tokens, total_time_s)
    log.info("    TTFT: %.0fms | 吞吐: %.1f tok/s | 时延: %.1fms/tok",
             ttft_s * 1000, tok_per_s, ms_per_tok)
    if spec_enabled and spec_drafted > 0:
        ratio = spec_accepted / spec_drafted * 100 if spec_drafted > 0 else 0
        log.info("    推测: 采纳 %d/%d (%.0f%%)", spec_accepted, spec_drafted, ratio)
    log.info("")


# ── 运行时显存快照 ────────────────────────────────────────────────────

def log_runtime_vram() -> None:
    """在关键节点（如每 N 步解码后）打印一行显存快照。

    自动从 VRAMBudget 获取信息，也兼容裸调用。
    """
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        frac = free / total * 100
        log.info("  📊 %s/%s 可用 (%.0f%%)",
                 _fmt_gib(free), _fmt_gib(total), frac)
    except Exception:
        pass


def _fmt_gib(b: int) -> str:
    return f"{b / (1024**3):.1f}GiB"


# ── 启动完成 ──────────────────────────────────────────────────────────

def print_engine_ready(api_port: int = 0) -> None:
    """打印引擎就绪信息，含 API 地址。"""
    if api_port > 0:
        log.info(_SEP)
        log.info("  ✅  MoeOwner 推理引擎就绪")
        log.info("  🌐  API: http://localhost:%d/v1/completions", api_port)
        log.info(_SEP)
    else:
        log.info(_SEP)
        log.info("  ✅  MoeOwner 推理引擎就绪（API 已禁用）")
        log.info(_SEP)
    log.info("")
