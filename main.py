#!/usr/bin/env python3
"""
Pure Python Inference Engine — Main Entry Point (MoE-first)

Usage:
    python main.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    python main.py --gguf path/to/model.gguf
"""

# ═══════════════════════════════════════════════════════════════════════════
# [Step 0] Env var MUST be set before any import touches torch.
# ═══════════════════════════════════════════════════════════════════════════
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import asyncio
import logging
import signal
import sys

import torch
from transformers import AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("engine")


# ---------------------------------------------------------------------------
# Stage 2: Global torch performance configuration
# ---------------------------------------------------------------------------


def configure_global_torch() -> None:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    os.environ.setdefault("TORCH_SDPA_OPTIMIZED", "1")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_stream(torch.cuda.Stream())
    logger.info("Tier 1 zero-cost optimisations applied.")


configure_global_torch()


# ---------------------------------------------------------------------------
# Stage 3: Import engine modules
# ---------------------------------------------------------------------------

try:
    from cache_manager import Block, HybridCache  # noqa: F401
    from scheduler import DecodeRequest, Request, UnifiedScheduler  # noqa: F401

    logger.info("Imported cache_manager & scheduler modules")
except ImportError as exc:
    logger.error("Failed to import engine modules: %s", exc)
    sys.exit(1)

_HAS_GGUF = False
try:
    from model_loader import load_model as load_gguf_model
    from model_loader.gguf_reader import GGUFFile as _GGUFFile  # noqa: F401

    _HAS_GGUF = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Stage 4: Model loading & kernel injection
# ---------------------------------------------------------------------------

# [Bug 5] USE_COMPILE is informational only — the scheduler's CUDAGraphManager
# auto-detects whether to use pre-recorded graphs vs direct forward.
USE_COMPILE: bool = True


def _inject_attention_kernel(layer: torch.nn.Module) -> None:
    """Replace ``self_attn.forward`` with compiled flash SDPA kernel.

    Works for both dense and MoE (Mixtral/Qwen2-MoE/etc.) models since
    attention structure is identical.
    """
    from attention_kernel import FlashAttentionKernel  # noqa: PLC0415

    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return

    def _patched_forward(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value=None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple:
        batch_size, seq_len, _ = hidden_states.shape
        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        v = attn.v_proj(hidden_states)

        num_heads = attn.num_heads
        num_kv_heads = getattr(attn, "num_key_value_heads", num_heads)
        head_dim = attn.head_dim

        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        softmax_scale = head_dim**-0.5
        attn_output = FlashAttentionKernel.forward(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = attn_output.to(hidden_states.dtype)
        attn_output = attn.o_proj(attn_output)
        return (attn_output, None)

    attn.forward = _patched_forward
    attn.is_causal = True
    logger.debug("Injected FlashAttentionKernel into layer %s", type(layer).__name__)


def _try_compile_model(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
) -> torch.nn.Module:
    """Attempt torch.compile with fallback chain.

    Returns the compiled model if successful, else the original model.
    Sets global USE_COMPILE accordingly.
    """
    global USE_COMPILE

    for mode, fullgraph in [("reduce-overhead", True), ("default", False)]:
        try:
            compiled = torch.compile(model, mode=mode, fullgraph=fullgraph, dynamic=False)
            with torch.no_grad():
                compiled(dummy_input)
            logger.info("torch.compile success (mode=%s, fullgraph=%s)", mode, fullgraph)
            USE_COMPILE = True
            return compiled
        except Exception as exc:
            logger.warning("torch.compile (mode=%s, fullgraph=%s) failed: %s", mode, fullgraph, exc)

    logger.info("torch.compile permanently disabled — using raw model + CUDA Graphs")
    USE_COMPILE = False
    return model


def load_and_inject_model(model_name: str) -> torch.nn.Module:
    """Load HF model, inject attention kernels, attempt compile."""
    logger.info("Loading HuggingFace model: %s ...", model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.to(device="cuda")
    model.eval()

    layers = getattr(model.model, "layers", None)
    if layers is not None:
        for layer in layers:
            _inject_attention_kernel(layer)
    else:
        logger.warning("Could not locate model.model.layers; attention injection skipped.")

    # [Step 2] Warmup + torch.compile fallback chain
    dummy_input = torch.randint(0, 1000, (1, 64), device="cuda")
    with torch.no_grad():
        model(dummy_input)

    model = _try_compile_model(model, dummy_input)

    logger.info("HuggingFace model loaded (USE_COMPILE=%s).", USE_COMPILE)
    return model


def load_and_inject_gguf_model(
    gguf_path: str,
    device: str = "cuda",
    block_size: int | None = None,
) -> torch.nn.Module:
    """Load model from GGUF file via mmap zero-copy path."""

    if not _HAS_GGUF:
        raise ImportError("model_loader not available.")

    if block_size is None:
        block_size = 32

    logger.info("Loading GGUF model: %s (block_size=%d)", gguf_path, block_size)
    model = load_gguf_model(
        path=gguf_path,
        dtype="fp16",
        device=device,
        block_size=block_size,
    )

    est_params = model.estimated_parameter_count_b
    suggested_bs = model.suggest_block_size(est_params)
    if suggested_bs != block_size:
        logger.info(
            "Suggested block_size=%d for ~%.1fB model (current=%d)",
            suggested_bs,
            est_params,
            block_size,
        )

    # [Step 2] Warmup + torch.compile fallback chain
    dummy_input = torch.randint(0, 1000, (1, 64), device="cuda")
    with torch.no_grad():
        model(dummy_input)

    model = _try_compile_model(model, dummy_input)

    logger.info(
        "GGUF model loaded: %d layers, %.1fB params, is_moe=%s, USE_COMPILE=%s",
        model.num_layers,
        est_params,
        getattr(model, "is_moe", False),
        USE_COMPILE,
    )
    return model


# ---------------------------------------------------------------------------
# Stage 5: Main async loop
# ---------------------------------------------------------------------------


def _setup_signal_handlers(scheduler: UnifiedScheduler) -> None:
    def _signal_handler(signum, frame):
        logger.info("Received signal %d, shutting down gracefully...", signum)
        scheduler.shutdown()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure Python Inference Engine — Hybrid Paged+Radix KV Cache"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL_NAME", ""),
        help="HuggingFace model name or local path to GGUF file",
    )
    parser.add_argument(
        "--gguf",
        type=str,
        default=None,
        help="Path to .gguf file (overrides --model if set)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Number of tokens per KV cache block",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=4096,
        help="Hidden dimension (default: 4096)",
    )
    parser.add_argument(
        "--total-blocks",
        type=int,
        default=None,
        help="Override automatic GPU-memory-based block count",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.gguf or (args.model and args.model.endswith(".gguf")):
        gguf_path = args.gguf or args.model
        model = load_and_inject_gguf_model(
            gguf_path=gguf_path,
            device="cuda",
            block_size=args.block_size,
        )
        model_hidden_size = model.hidden_size
        block_size = args.block_size or model.block_size
    else:
        model = load_and_inject_model(args.model)
        model_hidden_size = args.hidden_size
        block_size = args.block_size or 16

    cache = HybridCache(
        block_size=block_size,
        hidden_size=model_hidden_size,
        total_blocks=args.total_blocks,
    )
    scheduler = UnifiedScheduler(model, cache)
    _setup_signal_handlers(scheduler)

    logger.info("Engine running. Press Ctrl+C to stop.")
    try:
        while scheduler._running:
            await scheduler.step()
            await asyncio.sleep(0.001)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Engine stopped.")


if __name__ == "__main__":
    asyncio.run(main())
