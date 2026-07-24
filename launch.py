#!/usr/bin/env python3
"""
PyDense 引导式启动器 — 零基础用户友好入口
===============================================

自动完成：环境检测 → 依赖安装 → 模型下载 → 启动引擎
全程对话式引导，不需要记忆任何命令行参数。

Usage:
    python launch.py              ← 交互式（推荐）
    python launch.py --auto       ← 全自动（默认模型 + 默认设置）
    python launch.py --quick      ← 快速启动（跳过依赖检查）
"""

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import NoReturn

DEVNULL = subprocess.DEVNULL

# ═══════════════════════════════════════════════════════════════════════════
# 控制台工具
# ═══════════════════════════════════════════════════════════════════════════

class Style:
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    DIM = "\033[2m"
    BLUE = "\033[94m"


def c(text: str, *styles: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{''.join(styles)}{text}{Style.RESET}"


def header(text: str):
    print(f"\n{c('━' * 60, Style.DIM)}")
    print(f"  {c(text, Style.BOLD, Style.CYAN)}")
    print(f"{c('━' * 60, Style.DIM)}")


def success(text: str):
    print(f"  {c('✅', Style.GREEN)} {text}")


def warn(text: str):
    print(f"  {c('⚠️ ', Style.YELLOW)} {text}")


def error(text: str):
    print(f"  {c('❌', Style.RED)} {text}")


def info(text: str):
    print(f"  {c('ℹ️ ', Style.DIM)} {text}")


def ask(question: str, default: str = "y") -> bool:
    prompt = f"  {c('?', Style.MAGENTA)} {question} "
    if default == "y":
        prompt += c("[Y/n]: ", Style.DIM)
    else:
        prompt += c("[y/N]: ", Style.DIM)

    while True:
        try:
            ans = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if not ans:
            return default == "y"
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print(f"  {c('请输入 y 或 n', Style.YELLOW)}")


def select(options: list[str], prompt: str = "请选择", default: int = 0) -> int:
    """交互式选择。返回选中项的 index。"""
    print(f"  {c(prompt, Style.CYAN)}")
    for i, opt in enumerate(options):
        marker = c("➤", Style.GREEN) if i == default else " "
        print(f"    {marker} [{i}] {opt}")
    print(f"  {c(f'选择 [0-{len(options)-1}] (默认 {default}): ', Style.DIM)}", end="")
    try:
        ans = input().strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not ans:
        return default
    try:
        idx = int(ans)
        if 0 <= idx < len(options):
            return idx
    except ValueError:
        pass
    return default


# ═══════════════════════════════════════════════════════════════════════════
# 依赖检测
# ═══════════════════════════════════════════════════════════════════════════

MIN_PYTHON = (3, 10)
CORE_DEPS = {
    "torch": {"import": "torch", "pip": "torch", "min": "2.0.0"},
    "transformers": {"import": "transformers", "pip": "transformers", "min": "4.38.0"},
    "numpy": {"import": "numpy", "pip": "numpy", "min": "1.24.0"},
    "huggingface_hub": {"import": "huggingface_hub", "pip": "huggingface_hub", "min": "0.20.0"},
}

OPTIONAL_DEPS = {
    "bitsandbytes": {"import": "bitsandbytes", "pip": "bitsandbytes", "desc": "4bit/8bit 量化"},
    "sentencepiece": {"import": "sentencepiece", "pip": "sentencepiece", "desc": "部分模型的 tokenizer"},
    "accelerate": {"import": "accelerate", "pip": "accelerate", "desc": "大模型加载加速"},
}


def check_python() -> bool:
    """检查 Python 版本。"""
    v = sys.version_info
    if (v.major, v.minor) >= MIN_PYTHON:
        success(f"Python {v.major}.{v.minor}.{v.micro}")
        return True
    error(f"Python {v.major}.{v.minor}.{v.micro} < 需要 3.10+")
    return False


def check_cuda() -> dict:
    """检查 CUDA 和 GPU 状态。"""
    result = {"available": False, "name": "N/A", "vram_gb": 0, "cuda_version": "N/A"}

    # 先看能否 import torch
    if not importlib.util.find_spec("torch"):
        return result

    import torch

    if not torch.cuda.is_available():
        warn("CUDA 不可用（推理将使用 CPU，速度极慢）")
        return result

    try:
        result["available"] = True
        result["name"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        result["vram_gb"] = total / (1024**3)
        result["free_gb"] = free / (1024**3)
        result["cuda_version"] = torch.version.cuda or "N/A"
        success(f"GPU: {result['name']}")
        info(f"  显存: {result['free_gb']:.0f}/{result['vram_gb']:.0f} GiB 空闲")
        info(f"  CUDA: {result['cuda_version']}")
    except Exception:
        pass
    return result


def check_deps(auto_install: bool = False) -> bool:
    """检查核心依赖。auto_install=True 时自动 pip install。"""
    header("依赖检查")
    ok = True
    missing = []

    for name, dep in CORE_DEPS.items():
        spec = importlib.util.find_spec(dep["import"])
        if spec is not None:
            try:
                mod = importlib.import_module(dep["import"])
                ver = getattr(mod, "__version__", "?")
                success(f"{name} {ver}")
            except Exception:
                success(f"{name} (loaded)")
        else:
            error(f"{name} 未安装")
            ok = False
            missing.append(dep["pip"])

    for name, dep in OPTIONAL_DEPS.items():
        spec = importlib.util.find_spec(dep["import"])
        if spec is not None:
            success(f"{name} ({dep['desc']})")
        else:
            info(f"{name} 未安装 ({dep['desc']} — 可选)")

    if missing and auto_install:
        if ask(f"缺少 {len(missing)} 个依赖，是否自动安装?"):
            return _pip_install(missing)
        return False

    return ok


def _pip_install(packages: list[str]) -> bool:
    """执行 pip install。"""
    cmd = [sys.executable, "-m", "pip", "install"] + packages
    print(f"\n  运行: {' '.join(cmd)}\n")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            # 重新加载
            for pkg in packages:
                for name in list(sys.modules.keys()):
                    if pkg.replace("-", "_") in name:
                        sys.modules.pop(name, None)
            success(f"安装完成: {', '.join(packages)}")
            return True
        error(f"安装失败: {r.stderr[:200]}")
        return False
    except Exception as e:
        error(f"安装出错: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# 模型选择
# ═══════════════════════════════════════════════════════════════════════════

RECOMMENDED_MODELS: list[dict] = [
    {"name": "Qwen/Qwen2.5-7B-Instruct", "desc": "通用引擎，7B 最强", "vram": 16},
    {"name": "Qwen/Qwen2.5-1.5B-Instruct", "desc": "轻量级，笔记本也能跑", "vram": 4},
    {"name": "Qwen/Qwen2.5-14B-Instruct", "desc": "更强推理，需要 24G 显存", "vram": 28},
    {"name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "desc": "深度思考模型", "vram": 16},
    {"name": "Qwen/Qwen2.5-32B-Instruct", "desc": "旗舰级，需要 48G+ 显存", "vram": 64},
    {"name": "Qwen/Qwen2.5-72B-Instruct", "desc": "超大杯，推荐 4bit 量化", "vram": 80},
]

DRAFT_MODELS: list[dict] = [
    {"name": "none", "desc": "不使用投机解码"},
    {"name": "Qwen/Qwen2.5-0.5B-Instruct", "desc": "轻量草稿（推荐）"},
    {"name": "Qwen/Qwen2.5-1.5B-Instruct", "desc": "更强草稿，需要更多显存"},
]


def select_model(gpu_vram_gb: float = 0) -> tuple[str, str | None, str | None]:
    """交互式模型选择。返回 (model_name, draft_model_name_or_None, mirror_or_None)。"""
    header("模型选择")

    # ── 推荐 ──────────────────────────────────────────────────────
    print(f"  {c('推荐模型（根据你的 GPU 显存过滤）:', Style.BOLD)}")
    filtered = []
    for m in RECOMMENDED_MODELS:
        if gpu_vram_gb > 0 and gpu_vram_gb < m["vram"] * 0.8:
            continue
        filtered.append(m)
    if not filtered:
        filtered = RECOMMENDED_MODELS

    for i, m in enumerate(filtered):
        vram_str = f"({m['vram']}G 推荐)" if gpu_vram_gb > 0 else ""
        print(f"    [{i}] {m['name']} — {m['desc']} {vram_str}")

    print(f"    [{len(filtered)}] 手动输入模型名")
    print(f"    [{len(filtered)+1}] 使用默认: {filtered[0]['name']}")

    try:
        choice = input(f"\n  {c('选择模型', Style.CYAN)} [{len(filtered)+1}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return filtered[0]["name"], None, None

    if not choice or choice == str(len(filtered) + 1):
        model = filtered[0]["name"]
    elif choice == str(len(filtered)):
        model = input(f"  {c('输入模型名', Style.CYAN)}: ").strip()
    else:
        try:
            idx = int(choice)
            model = filtered[idx]["name"]
        except (ValueError, IndexError):
            model = filtered[0]["name"]

    # ── 投机解码草稿 ─────────────────────────────────────────────
    print(f"\n  {c('投机解码（可选）:', Style.BOLD)}")
    print(f"    用小模型生成草稿，大模型一次验证多个 token，显著加速。")
    for i, dm in enumerate(DRAFT_MODELS):
        print(f"    [{i}] {dm['name']} — {dm['desc']}")

    try:
        dc = input(f"\n  {c('选择草稿模型', Style.CYAN)} [0]: ").strip()
    except (EOFError, KeyboardInterrupt):
        dc = "0"
    if not dc:
        dc = "0"
    draft = None if dc == "0" else DRAFT_MODELS[int(dc)]["name"]

    # ── 镜像源 ───────────────────────────────────────────────────
    header("镜像源")
    print(f"    [0] 自动选择（国内自动用 hf-mirror）")
    print(f"    [1] huggingface (官方)")
    print(f"    [2] hf-mirror (国内镜像)")
    print(f"    [3] 自定义 URL")
    try:
        mc = input(f"\n  {c('选择镜像', Style.CYAN)} [0]: ").strip()
    except (EOFError, KeyboardInterrupt):
        mc = "0"
    if not mc:
        mc = "0"

    mirror_map = {"0": None, "1": "huggingface", "2": "hf-mirror", "3": "custom"}
    mirror = mirror_map.get(mc, None)
    if mirror == "custom":
        mirror = input(f"  {c('输入镜像 URL', Style.CYAN)}: ").strip()

    return model, draft, mirror


# ═══════════════════════════════════════════════════════════════════════════
# 启动主流程
# ═══════════════════════════════════════════════════════════════════════════

BANNER = f"""
{c('╔════════════════════════════════════════════╗', Style.CYAN)}
{c('║', Style.CYAN)}    {c('PyDense — 稠密模型推理引擎', Style.BOLD, Style.CYAN)}    {c('║', Style.CYAN)}
{c('║', Style.CYAN)}    {c('FlashAttention + KV Cache + 推测解码', Style.DIM)}    {c('║', Style.CYAN)}
{c('╚════════════════════════════════════════════╝', Style.CYAN)}
"""


def show_welcome():
    print(BANNER)
    info(f"Python {sys.version.split()[0]}  |  {platform.system()}  |  {platform.machine()}")
    print()


def run_interactive():
    """交互式引导流程。"""
    show_welcome()

    # ── Step 1: 环境检测 ──────────────────────────────────────────
    header("🔍 环境检测")
    py_ok = check_python()
    if not py_ok:
        error("Python 版本不符合要求，请安装 Python 3.10+")
        sys.exit(1)

    gpu = check_cuda()
    print()

    # ── Step 2: 依赖安装 ──────────────────────────────────────────
    deps_ok = check_deps(auto_install=False)
    if not deps_ok:
        if ask("是否自动安装缺失的核心依赖?"):
            check_deps(auto_install=True)

    # ── Step 3: 模型选择 ──────────────────────────────────────────
    vram = gpu.get("vram_gb", 0)
    model_name, draft_model, mirror = select_model(vram)

    # ── Step 4: 量化选择 ──────────────────────────────────────────
    quantize = None
    if gpu["available"] and vram > 0:
        # 估算模型是否需要量化
        has_bnb = importlib.util.find_spec("bitsandbytes") is not None
        header("量化")
        if vram < 24:
            if ask("显存有限，是否启用 4-bit 量化推荐?"):
                quantize = "4bit"
        elif vram < 48:
            qopts = ["无 (FP16)", "8-bit 量化", "4-bit 量化"]
            qi = select(qopts, "选择量化方式")
            if qi == 1:
                quantize = "8bit"
            elif qi == 2:
                quantize = "4bit"
        else:
            info("显存充足，使用 FP16 精度")

        if quantize and not has_bnb:
            warn("BitsAndBytes 未安装，量化不可用")
            if ask("是否自动安装 bitsandbytes?"):
                _pip_install(["bitsandbytes"])
                quantize = None  # 重试时需要重新选择

    # ── Step 5: 确认 ──────────────────────────────────────────────
    header("📋 配置摘要")
    info(f"模型:       {model_name}")
    info(f"投机解码:   {draft_model or '不使用'}")
    info(f"量化:       {quantize or 'FP16'}")
    info(f"镜像:       {mirror or '自动'}")
    print()

    if not ask("确认启动?"):
        info("已取消")
        return

    # ── Step 6: 构建命令并执行 ───────────────────────────────────
    cmd = [sys.executable, "main.py", "--model", model_name]
    if draft_model:
        cmd += ["--draft-model", draft_model]
    if quantize:
        cmd += ["--quantize", quantize]
    if mirror and mirror not in ("huggingface", "auto"):
        cmd += ["--mirror", mirror]

    print(f"\n  {c('启动中...', Style.GREEN)}")
    print(f"  {c(' '.join(cmd), Style.DIM)}")
    print()
    os.execvp(cmd[0], cmd)


# ═══════════════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PyDense 引导式启动器",
    )
    parser.add_argument("--auto", action="store_true", help="全自动模式（默认配置）")
    parser.add_argument("--quick", action="store_true", help="跳过依赖检查")
    parser.add_argument("--model", type=str, default=None, help="模型名")
    parser.add_argument("--draft-model", type=str, default=None, help="草稿模型")
    parser.add_argument("--mirror", type=str, default=None, help="镜像源")
    parser.add_argument("--quantize", type=str, default=None, help="量化: 4bit/8bit")
    args = parser.parse_args()

    if args.auto or (args.model and not args.quick):
        cmd = [sys.executable, "main.py", "--model", args.model or "Qwen/Qwen2.5-7B-Instruct"]
        if args.draft_model:
            cmd += ["--draft-model", args.draft_model]
        if args.quantize:
            cmd += ["--quantize", args.quantize]
        if args.mirror:
            cmd += ["--mirror", args.mirror]
        # 先检查依赖
        if not args.quick:
            show_welcome()
            check_deps(auto_install=True)
        os.execvp(cmd[0], cmd)

    elif args.quick:
        cmd = [sys.executable, "main.py", "--model", args.model or "Qwen/Qwen2.5-7B-Instruct"]
        os.execvp(cmd[0], cmd)

    else:
        run_interactive()


if __name__ == "__main__":
    import argparse
    main()
