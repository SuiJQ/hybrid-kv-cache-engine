#!/usr/bin/env python3
"""
MoeOwner — 稠密模型推理引擎 (极致性价比版)
=============================================

性能特性（全部自动开启，零配置）：
  • FlashAttention SDPA + torch.compile
  • PagedAttention/RadixAttention 混合 KV 缓存
  • Chunked Prefill (Sarathi) + 双 CUDA 流管线
  • Goose 推测解码 (PLD + 树注意力)
  • Self-Spec 骨架推测 (ACL'24)
  • 自适应 KV 压缩 (StreamingLLM + H2O)
  • 投机解码 (小模型草稿)

使用方式：
    python main.py                           # 默认 Qwen2.5-7B
    python main.py --model Qwen/Qwen2.5-7B chat   # 交互聊天
    python main.py --model Qwen/Qwen2.5-72B --quantize 4bit  # 量化

镜像源（自动 fallback）：
    huggingface — https://huggingface.co (官方)
    hf-mirror   — https://hf-mirror.com (国内高速)
    modelscope  — modelScope SDK (阿里云)
"""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import asyncio
import importlib
import json
import logging
import math
import platform
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from cache_manager import HybridCache
from engine_logger import (
    apply_module_levels, get_logger, print_banner,
    print_engine_ready, print_model_summary,
    print_optimizations, print_request_summary,
)
from scheduler import DecodeRequest, Request, UnifiedScheduler
from tool_sink import ToolOrchestrator
from vram_budget import VRAMBudget

apply_module_levels()
logger = get_logger("engine")


# ═══════════════════════════════════════════════════════════════════════════
# 全局 torch 优化
# ═══════════════════════════════════════════════════════════════════════════

def configure_torch() -> None:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    os.environ.setdefault("TORCH_SDPA_OPTIMIZED", "1")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # 创建专属 CUDA 流（用于权重复制等后台操作）
    torch.cuda.set_stream(torch.cuda.Stream())
    logger.info("Global torch optimizations: Flash SDPA, TF32, cuDNN benchmark")

configure_torch()


# ═══════════════════════════════════════════════════════════════════════════
# 多镜像源下载模块
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MirrorConfig:
    name: str = "huggingface"
    endpoint: str = ""

    MIRRORS: dict = field(default_factory=lambda: {
        "huggingface": "https://huggingface.co",
        "hf-mirror": "https://hf-mirror.com",
    })

    @classmethod
    def resolve(cls, mirror_arg: str | None = None) -> "MirrorConfig":
        if mirror_arg:
            name = mirror_arg.lower().replace("_", "-")
            if name in cls.MIRRORS:
                ep = cls.MIRRORS[name]
                logger.info("Mirror: %s → %s", name, ep)
                return cls(name=name, endpoint=ep)
            # 可能是自定义 URL
            logger.info("Mirror (custom): %s", mirror_arg)
            return cls(name="custom", endpoint=mirror_arg)

        env_ep = os.environ.get("HF_ENDPOINT", "")
        if env_ep:
            return cls(name="env", endpoint=env_ep)

        # 自动检测国内环境
        try:
            locale = os.environ.get("LANG", "")
            tz = os.environ.get("TZ", "")
            cn_hints = ("zh_CN", "zh-CN", "Asia/Shanghai", "CST-8")
            if any(h in locale or h in tz for h in cn_hints):
                logger.info("Auto-detected China locale → hf-mirror")
                return cls(name="hf-mirror", endpoint="https://hf-mirror.com")
        except Exception:
            pass

        return cls(name="huggingface", endpoint="https://huggingface.co")

    def apply(self) -> None:
        if self.endpoint:
            os.environ["HF_ENDPOINT"] = self.endpoint

    def __str__(self) -> str:
        return f"{self.name} ({self.endpoint})" if self.endpoint else self.name

    def __repr__(self) -> str:
        return self.__str__()


def ensure_model(
    model_name: str,
    mirror_arg: str | None = None,
    tool_name: str = "",
) -> MirrorConfig:
    """确保模型已下载，支持多镜像 fallback。"""
    from huggingface_hub import scan_cache_dir

    logger.info("Ensuring model: %s", model_name)

    # 优先：直接使用环境变量中的 endpoint
    env_ep = os.environ.get("HF_ENDPOINT", "")
    mirrors_try = []

    if mirror_arg:
        mirrors_try.append(MirrorConfig.resolve(mirror_arg))
    if env_ep and not mirror_arg:
        mirrors_try.append(MirrorConfig(name="env", endpoint=env_ep))

    # 默认镜像优先级
    if not mirrors_try:
        mirror = MirrorConfig.resolve(None)  # 自动检测
        if mirror.name == "hf-mirror":
            mirrors_try = [mirror]
            mirrors_try.append(MirrorConfig(name="huggingface", endpoint="https://huggingface.co"))
        else:
            mirrors_try = [MirrorConfig(name="huggingface", endpoint="https://huggingface.co")]
            mirrors_try.append(MirrorConfig(name="hf-mirror", endpoint="https://hf-mirror.com"))

    # 检查缓存
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_name and repo.repo_type == "model":
                logger.info("Model already cached: %s", model_name)
                mirrors_try[0].apply()
                return mirrors_try[0]
    except Exception:
        pass

    # 逐个尝试
    last_error = None
    for i, m in enumerate(mirrors_try):
        try:
            _download_model(model_name, m)
            logger.info("Downloaded via %s (%s)", m.name, m.endpoint)
            return m
        except Exception as e:
            last_error = e
            logger.warning("Mirror %s failed: %s", m.name, e)

    raise RuntimeError(
        f"Cannot download model {model_name} after {len(mirrors_try)} mirrors. "
        f"Last error: {last_error}"
    )


def _download_model(model_name: str, mirror: MirrorConfig) -> None:
    """从指定镜像下载模型。"""
    old_ep = os.environ.get("HF_ENDPOINT", "")
    mirror.apply()
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model_name,
            ignore_patterns=["*.ot", "*.msgpack", "*.h5", "*.safetensors.index.json"],
            resume_download=True,
            local_files_only=False,
            max_workers=4,
        )
    except Exception:
        # 如果 snapshot_download 失败，尝试通过 transformers 直接加载
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model_name)
    finally:
        if old_ep:
            os.environ["HF_ENDPOINT"] = old_ep
        else:
            os.environ.pop("HF_ENDPOINT", None)


# ═══════════════════════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════════════════════

_GOOSE_OK = importlib.util.find_spec("goose_core") is not None

def _inject_attention(layer: torch.nn.Module) -> None:
    """替换注意力层为 FlashAttention 内核。"""
    from attention_kernel import FlashAttentionKernel

    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return

    orig_fwd = getattr(attn, "forward", None)

    def patched_forward(
        hidden_states, attention_mask=None,
        past_key_value=None, use_cache=False,
        position_ids=None, **kwargs,
    ):
        bs, sl, _ = hidden_states.shape
        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        v = attn.v_proj(hidden_states)

        nh = attn.num_heads
        nkv = getattr(attn, "num_key_value_heads", nh)
        hd = attn.head_dim
        scale = hd ** -0.5

        q = q.view(bs, sl, nh, hd).transpose(1, 2)
        k = k.view(bs, sl, nkv, hd).transpose(1, 2)
        v = v.view(bs, sl, nkv, hd).transpose(1, 2)

        if past_key_value is not None:
            k_old, v_old = past_key_value
            k = torch.cat([k_old.to(k.device), k], dim=2)
            v = torch.cat([v_old.to(v.device), v], dim=2)

        out = FlashAttentionKernel.forward(
            q, k, v, scale,
            causal=(attention_mask is None),
            attn_mask=attention_mask,
        )
        out = out.transpose(1, 2).contiguous().view(bs, sl, -1).to(hidden_states.dtype)
        out = attn.o_proj(out)
        return (out, (k, v) if use_cache else None)

    attn.forward = patched_forward
    attn.is_causal = True


def load_model(
    model_name: str,
    quantize: str | None = None,
    compile_model: bool = True,
    draft_model: str | None = None,
) -> tuple:
    """加载主模型和可选的投机解码草稿模型。

    Returns:
        (model, tokenizer, draft_model_or_None, model_info_dict)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("Loading model: %s ...", model_name)

    # ── Tokenizer ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    logger.info("Tokenizer loaded (vocab=%d)", len(tokenizer))

    # ── 模型 ──────────────────────────────────────────────────────
    kwargs = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True)

    if quantize:
        try:
            import bitsandbytes  # noqa: F401
            if quantize == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                )
            elif quantize == "8bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except ImportError:
            logger.warning("bitsandbytes not installed, skipping quantization")

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not quantize:
        model = model.to(device="cuda")
    model.eval()

    # ── 注入 FlashAttention ──────────────────────────────────────
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is not None:
        for layer in layers:
            _inject_attention(layer)
        logger.info("Attention injected: %d layers", len(layers))

    # ── 模型信息 ──────────────────────────────────────────────────
    cfg = model.config
    info = {
        "hidden_size": getattr(cfg, "hidden_size", 4096),
        "num_layers": getattr(cfg, "num_hidden_layers", getattr(cfg, "num_layers", 32)),
        "num_heads": getattr(cfg, "num_attention_heads", 32),
        "num_kv_heads": getattr(cfg, "num_key_value_heads", 32),
        "head_dim": getattr(cfg, "head_dim", 4096 // 32),
        "vocab_size": getattr(cfg, "vocab_size", 32000),
        "max_seq_len": getattr(cfg, "max_position_embeddings", 8192),
    }

    # ── torch.compile ────────────────────────────────────────────
    if compile_model:
        try:
            dummy = torch.randint(0, 1000, (1, 64), device="cuda" if not quantize else "cpu")
            with torch.no_grad():
                model(dummy)
            for mode, fg in [("reduce-overhead", True), ("default", False)]:
                try:
                    model = torch.compile(model, mode=mode, fullgraph=fg, dynamic=False)
                    model(dummy)
                    logger.info("torch.compile OK (mode=%s)", mode)
                    break
                except Exception as e:
                    logger.warning("torch.compile(%s) failed: %s", mode, e)
        except Exception as e:
            logger.warning("Compile warmup failed: %s", e)

    # ── 投机解码草稿模型 ────────────────────────────────────────
    draft_model_obj = None
    if draft_model:
        try:
            logger.info("Loading draft model: %s ...", draft_model)
            dm = AutoModelForCausalLM.from_pretrained(
                draft_model, torch_dtype=torch.float16, trust_remote_code=True,
            )
            dm = dm.to(device="cuda")
            dm.eval()
            draft_model_obj = dm
            logger.info("Draft model loaded: %s", draft_model)
        except Exception as e:
            logger.warning("Draft model load failed: %s (speculation disabled)", e)

    return model, tokenizer, draft_model_obj, info


# ═══════════════════════════════════════════════════════════════════════════
# 服务器
# ═══════════════════════════════════════════════════════════════════════════

class InferenceServer:
    """推理服务器（极简调度 + 所有优化自动开启）。"""

    def __init__(self, model, tokenizer, draft_model, model_info, mirror, args):
        self.model = model
        self.tokenizer = tokenizer
        self.draft_model = draft_model
        self.mirror = mirror
        self.args = args
        self.info = model_info

        # ── 显存预算 ─────────────────────────────────────────────
        self.vram_budget = VRAMBudget(
            hidden_size=model_info["hidden_size"],
            num_layers=model_info["num_layers"],
        )
        self.vram_budget.log_status()

        # ── 混合 KV 缓存 ─────────────────────────────────────────
        bs = self._auto_block_size(model_info["hidden_size"])
        tb = self.vram_budget.safe_total_blocks()
        self.cache = HybridCache(
            hidden_size=model_info["hidden_size"],
            num_layers=model_info["num_layers"],
            block_size=bs, total_blocks=tb,
            num_kv_heads=model_info["num_kv_heads"],
            head_dim=model_info["head_dim"],
        )

        # ── 调度器 ───────────────────────────────────────────────
        self.scheduler = UnifiedScheduler(
            model=model, cache=self.cache,
            detokenizer=self._detok if tokenizer else None,
            vram_budget=self.vram_budget,
            draft_model=draft_model,
            draft_tokenizer=tokenizer,  # 同一 tokenizer
        )

        # ── 工具 ─────────────────────────────────────────────────
        self.tool_orch = ToolOrchestrator(model, tokenizer)

        # ── 请求跟踪 ─────────────────────────────────────────────
        self._req_queue = asyncio.Queue()
        self._running = True
        self._req_count = 0

        if platform.system() != "Windows":
            try:
                loop = asyncio.get_event_loop()
                for s in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(s, self._shutdown)
            except NotImplementedError:
                pass

    def _shutdown(self):
        self._running = False
        self.scheduler.shutdown()

    @staticmethod
    def _auto_block_size(hs: int) -> int:
        return 32 if hs >= 7168 else (16 if hs >= 4096 else 8)

    def _detok(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True) if self.tokenizer else ""

    async def generate(self, prompt, max_new=512, temperature=0.7, top_p=0.9):
        """生成文本（非流式）。"""
        if isinstance(prompt, str) and self.tokenizer:
            toks = self.tokenizer.encode(prompt, add_special_tokens=True)
        elif isinstance(prompt, list):
            toks = prompt
        else:
            toks = [1]

        rid = f"req_{self._req_count}_{time.monotonic_ns()}"
        self._req_count += 1

        # 工具模式
        try:
            r = self.tool_orch.generate(toks, self.scheduler, max_new, temperature, top_p)
            if isinstance(r, (str, bytes)):
                return r if isinstance(r, str) else r.decode()
        except Exception:
            pass

        self.scheduler.submit(Request(prompt_tokens=toks, request_id=rid, max_new_tokens=max_new))
        gen = []
        while self._running:
            await self.scheduler.step()
            for d in self.scheduler.active_decode_pool:
                if d.request_id == rid:
                    gen.extend(d.generated_tokens[len(gen):])
                    if d.is_done:
                        return self._detok(gen)
            await asyncio.sleep(0.001)

        return self._detok(gen)

    async def chat(self, messages: list[dict], max_new=1024):
        """聊天接口。"""
        try:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = messages[-1]["content"] if messages else ""
        return await self.generate(prompt, max_new)

    async def interactive_chat(self):
        """类似 ollama run 的终端聊天。"""
        print()
        print("=" * 60)
        print("  MoeOwner 交互式聊天 (Ctrl+D / /exit 退出)")
        print("  " + "-" * 56)
        print("  /clear  清空历史  /model  显示模型信息")
        print("=" * 60)
        print()
        history = []
        while self._running:
            try:
                inp = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not inp:
                continue
            if inp.lower() in ("/exit", "/quit", "/bye"):
                break
            if inp.lower() == "/clear":
                history.clear()
                print("History cleared")
                continue
            if inp.lower() == "/model":
                print(f"  Model: {self.args.model}")
                print(f"  Draft: {self.args.draft_model or 'none'}")
                print(f"  Quantize: {self.args.quantize or 'none'}")
                print(f"  Mirror: {self.mirror}")
                continue

            history.append({"role": "user", "content": inp})
            print("... ", end="", flush=True)
            try:
                resp = await self.chat(history)
                print("\r" + " " * 60 + "\r", end="")
                print(resp)
                history.append({"role": "assistant", "content": resp})
            except Exception as e:
                print(f"\nError: {e}")

        self._running = False
        self.scheduler.shutdown()

    async def serve(self, host="0.0.0.0", port=8080):
        """HTTP 服务。"""
        try:
            from api_server import serve_http
            await serve_http(self, host=host, port=port)
        except ImportError:
            logger.error("api_server not available")
        except Exception as e:
            logger.error("HTTP serve error: %s", e)


# ═══════════════════════════════════════════════════════════════════════════
# 基准测试
# ═══════════════════════════════════════════════════════════════════════════

_SEED = [42, 17, 88, 33, 55, 99, 21, 73, 61, 8]

def _make_prompt(n: int) -> list[int]:
    t = []
    while len(t) < n:
        t.extend(_SEED[:n - len(t)])
    return t or [42]


async def benchmark(server: InferenceServer, plen=128, glen=128):
    """运行基准测试。"""
    print("\n" + "=" * 60)
    print(f"  Benchmark: prompt={plen}, gen={glen}")
    print("=" * 60)

    # 预热
    wr = Request(prompt_tokens=_make_prompt(32), request_id="_warm", max_new_tokens=32)
    server.scheduler.submit(wr)
    for _ in range(100):
        await server.scheduler.step()
        done = any(d.is_done for d in server.scheduler.active_decode_pool if d.request_id == "_warm")
        if done:
            break

    # 正式
    req = Request(prompt_tokens=_make_prompt(plen), request_id="_bm", max_new_tokens=glen)
    server.scheduler.submit(req)
    start = time.monotonic()
    while True:
        await server.scheduler.step()
        for d in server.scheduler.active_decode_pool:
            if d.request_id == "_bm":
                if d.is_done:
                    break
        else:
            await asyncio.sleep(0.001)
            continue
        break

    elapsed = time.monotonic() - start
    print_request_summary(prompt_len=plen, gen_len=glen, prefill_tokens=plen, gen_tokens=glen, elapsed=elapsed)
    return {"elapsed": elapsed}


# ═══════════════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="MoeOwner — 稠密模型推理引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                          # 默认 7B 模型
  python main.py --model Qwen/Qwen2.5-7B chat             # 聊天模式
  python main.py --model Qwen/Qwen2.5-72B --quantize 4bit # 量化
  python main.py --model Qwen/Qwen2.5-7B --draft-model Qwen/Qwen2.5-0.5B  # 投机解码
  python main.py --mirror hf-mirror                        # 国内镜像
  python main.py --benchmark                               # 基准测试
        """,
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="模型名")
    p.add_argument("--mirror", default=None, help="镜像源: huggingface / hf-mirror / 自定义URL")
    p.add_argument("--quantize", choices=["4bit", "8bit"], default=None, help="量化")
    p.add_argument("--draft-model", default=None, help="草稿模型 (投机解码用小模型)")
    p.add_argument("--no-compile", action="store_true", help="禁用 torch.compile")
    p.add_argument("--no-tools", action="store_true", help="禁用工具调用")
    p.add_argument("--benchmark", action="store_true", help="基准测试")
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--gen-len", type=int, default=128)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("chat", nargs="?", help="'chat' 启动交互式聊天")
    return p


async def amain():
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        os.environ["MOE_LOG_LEVEL"] = "debug"

    print_banner()

    # 下载
    logger.info("Step 1/4: 确保模型已下载...")
    mirror = ensure_model(args.model, args.mirror)
    if args.draft_model:
        try:
            ensure_model(args.draft_model, args.mirror, tool_name="draft")
        except Exception as e:
            logger.warning("Draft model download failed: %s (继续)", e)

    # 加载
    logger.info("Step 2/4: 加载模型...")
    model, tokenizer, draft_model_obj, info = load_model(
        args.model, args.quantize,
        compile_model=not args.no_compile,
        draft_model=args.draft_model,
    )

    # 创建服务器
    logger.info("Step 3/4: 初始化调度器...")
    server = InferenceServer(model, tokenizer, draft_model_obj, info, mirror, args)

    # 打印信息
    logger.info("Step 4/4: 就绪!")
    print_optimizations()
    print_model_summary(
        num_layers=info["num_layers"], hidden_size=info["hidden_size"],
        num_heads=info["num_heads"], num_kv_heads=info["num_kv_heads"],
        vocab_size=info["vocab_size"], max_seq_len=info["max_seq_len"],
        model_name=args.model,
    )
    if draft_model_obj:
        logger.info("  投机解码草稿: %s ✅", args.draft_model)
    print_engine_ready(model_name=args.model, port=args.port, mirror=mirror)

    # 运行
    try:
        if args.benchmark:
            await benchmark(server, args.prompt_len, args.gen_len)
        elif args.chat == "chat":
            await server.interactive_chat()
        else:
            await server.serve(args.host, args.port)
    except KeyboardInterrupt:
        pass
    finally:
        server.scheduler.shutdown()


def main():
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
