#!/usr/bin/env python3
"""
MoeOwner 启动器 —— 零基础用户友好入口
========================================

使用方法：
    python launch.py                    ← 交互式菜单（推荐）
    python launch.py --auto             ← 全自动：测试 → 启动
    python launch.py --quick            ← 跳过测试，直接启动

不需要记忆任何命令行参数，按提示操作即可。
"""

import os
import sys
import time
import shutil

# ── 控制台配色 ──────────────────────────────────────────────────────

class Style:
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    DIM = "\033[2m"

def c(text: str, *styles: str) -> str:
    """Apply ANSI styles to text; fallback if stdout isn't a TTY."""
    if not sys.stdout.isatty():
        return text
    prefix = "".join(styles)
    return f"{prefix}{text}{Style.RESET}"


# ── Banner ──────────────────────────────────────────────────────────

BANNER = f"""
{Style.CYAN}{Style.BOLD}
    ╔═══════════════════════════════════════════╗
    ║                                           ║
    ║     MoeOwner 推理引擎                      ║
    ║     Hybrid KV Cache + Goose 推测解码       ║
    ║     + 超长上下文 (SelfExtend 默认开启)     ║
    ║                                           ║
    ╚═══════════════════════════════════════════╝
{Style.RESET}
"""

MENU = f"""
{Style.BOLD}请选择操作：{Style.RESET}
  {Style.GREEN}[1]{Style.RESET}  完整流程：自动测试 → 启动服务
  {Style.GREEN}[2]{Style.RESET}  仅运行基准测试（检测性能）
  {Style.GREEN}[3]{Style.RESET}  直接启动服务（跳过测试）
  {Style.GREEN}[4]{Style.RESET}  查看系统状态
  {Style.GREEN}[q]{Style.RESET}  退出

{Style.DIM}  💡 超长上下文 (SelfExtend) 默认自动开启，无需任何参数{Style.RESET}
"""


# ── 模型发现 ────────────────────────────────────────────────────────

def find_gguf_models() -> list[str]:
    """自动搜索当前目录下的 .gguf 文件。"""
    models = []
    search_dirs = [".", "models", "../models", "hf", "./model_loader"]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".gguf"):
                models.append(os.path.join(d, f))
    return sorted(models)


def find_hf_models() -> list[str]:
    """检测环境中的 HF 模型名。"""
    env_model = os.environ.get("MODEL_NAME", "")
    return [env_model] if env_model else []


def ask_model(args: list[str]) -> tuple[str, str]:
    """让用户选择模型。返回 (model_path, model_type)  model_type = 'gguf' | 'hf'.

    流程：
      1) 自动搜索 .gguf
      2) 检查环境变量 MODEL_NAME
      3) 让用户选择或手动输入
    """
    gguf_models = find_gguf_models()
    hf_models = find_hf_models()
    all_options = []

    if gguf_models:
        all_options.append(("gguf", "[GGUF]", gguf_models))
    if hf_models:
        all_options.append(("hf", "[HF]", hf_models))
    all_options.append(("manual", "[自定义]", []))

    if not sys.stdout.isatty() or "--auto" in args or "--quick" in args:
        # 非交互模式 —— 自动选择
        if gguf_models:
            return gguf_models[0], "gguf"
        if hf_models:
            return hf_models[0], "hf"
        print(f"{Style.YELLOW}未找到模型文件，请使用 --gguf 或 --model 参数指定{Style.RESET}")
        sys.exit(1)

    print(f"\n{Style.BOLD}模型选择{Style.RESET}")
    print(f"{Style.DIM}─" * 40 + Style.RESET)

    idx = 1
    options_map = {}
    for mtype, label, models in all_options:
        if mtype == "manual":
            print(f"  {Style.GREEN}[{idx}]{Style.RESET}  {label} 手动指定模型路径/名称")
            options_map[idx] = ("manual", "")
            idx += 1
        else:
            for m in models:
                display = os.path.basename(m) if mtype == "gguf" else m
                print(f"  {Style.GREEN}[{idx}]{Style.RESET}  {label} {display}")
                options_map[idx] = (mtype, m)
                idx += 1

    choice = input(f"\n{Style.CYAN}请选择 [1-{idx-1}]:{Style.RESET} ").strip()
    try:
        c = int(choice)
        if c in options_map:
            mtype, path = options_map[c]
            if mtype == "manual":
                path = input(f"{Style.CYAN}输入 GGUF 文件路径或 HF 模型名: {Style.RESET}").strip()
                if path.endswith(".gguf"):
                    return path, "gguf"
                else:
                    return path, "hf"
            return path, mtype
    except (ValueError, TypeError):
        pass

    # Fallback: 第一个模型
    for mtype, label, models in all_options:
        if models:
            return models[0], mtype

    path = input(f"{Style.YELLOW}未找到模型，请输入 GGUF 路径或 HF 模型名: {Style.RESET}").strip()
    return path, "gguf" if path.endswith(".gguf") else "hf"


# ── 超长上下文方法解析 ────────────────────────────────────────────

LONG_CONTEXT_METHODS = ["none", "selfextend", "reattention", "yarn"]

def parse_context_method(raw_args: list[str]) -> str:
    """Extract --context-method value from raw CLI args if present."""
    for i, a in enumerate(raw_args):
        if a == "--context-method" and i + 1 < len(raw_args):
            val = raw_args[i + 1]
            if val in LONG_CONTEXT_METHODS:
                return val
        if a == "--disable-long-context":
            return "none"
    env_val = os.environ.get("CONTEXT_METHOD", "")
    return env_val if env_val in LONG_CONTEXT_METHODS else "selfextend"


def ask_context_method() -> str:
    """交互式选择超长上下文方法。"""
    print(f"\n{Style.BOLD}超长上下文扩展方法{Style.RESET}")
    print(f"{Style.DIM}  默认: selfextend (4行代码，通吃RoPE模型，已开启)")
    print(f"  • selfextend  — 4行代码，通吃RoPE模型 (LLaMA/Qwen/Mistral)")
    print(f"  • reattention — 任意Transformer均可用，两步注意力")
    print(f"  • yarn        — 行业标准，HF Transformers内置")
    print(f"  • none        — 关闭{Style.RESET}")
    choices_text = "/".join(f"{c}" for c in LONG_CONTEXT_METHODS)
    method = input(f"{Style.CYAN}选择 [{choices_text}，直接回车默认 selfextend]: {Style.RESET}").strip().lower()
    return method if method in LONG_CONTEXT_METHODS else "selfextend"


# ── 功能函数 ────────────────────────────────────────────────────────

def run_command(cmd: list[str], desc: str = "") -> int:
    """运行命令并显示进度。"""
    import subprocess
    print(f"\n{Style.DIM}  $ {' '.join(cmd)}{Style.RESET}")
    if desc:
        print(f"  {Style.CYAN}→ {desc}...{Style.RESET}")
    sys.stdout.flush()
    result = subprocess.run(cmd)
    return result.returncode


def run_benchmark(model_path: str, mtype: str, prompt_len: int = 128,
                  gen_len: int = 128, use_spec: bool = False,
                  context_method: str = "none") -> int:
    """运行基准测试。"""
    if not os.path.exists(os.path.join(os.path.dirname(__file__) or ".", "main.py")):
        print(f"{Style.RED}错误: 找不到 main.py{Style.RESET}")
        return 1

    cmd = [sys.executable, "-u", "main.py", "--benchmark",
           "--benchmark-prompt-len", str(prompt_len),
           "--benchmark-gen-len", str(gen_len)]
    if mtype == "gguf":
        cmd += ["--gguf", model_path]
    else:
        cmd += ["--model", model_path]
    if use_spec:
        cmd += ["--speculative"]
    if context_method and context_method != "none":
        cmd += ["--context-method", context_method]

    print(f"\n{Style.BOLD}{Style.MAGENTA}基准测试：{Style.RESET}")
    print(f"{Style.DIM}  • 预热: 32 → 32 tokens")
    print(f"  • 测试: {prompt_len} → {gen_len} tokens")
    print(f"  • 推测: {'已启用 (Goose)' if use_spec else '未启用'}")
    print(f"  • 超长上下文: {context_method if context_method != 'none' else '未启用'}")
    print(f"  • 模型: {os.path.basename(model_path) if mtype == 'gguf' else model_path}{Style.RESET}")
    print(f"{Style.DIM}  (测试中，请耐心等待...){Style.RESET}\n")
    sys.stdout.flush()
    return run_command(cmd, "运行基准测试")


def run_interactive(model_path: str, mtype: str, api_port: int = 8000,
                    use_spec: bool = False,
                    context_method: str = "none"):
    """启动 API 服务器（交互模式）。"""
    cmd = [sys.executable, "-u", "main.py",
           "--api-port", str(api_port)]
    if mtype == "gguf":
        cmd += ["--gguf", model_path]
    else:
        cmd += ["--model", model_path]
    if use_spec:
        cmd += ["--speculative"]
    if context_method and context_method != "none":
        cmd += ["--context-method", context_method]
        if context_method == "selfextend":
            cmd += ["--neighbor-window", "1024", "--group-size", "8"]
        elif context_method == "reattention":
            cmd += ["--reattn-top-k", "2048"]
        elif context_method == "yarn":
            cmd += ["--yarn-factor", "8"]

    print(f"\n{Style.GREEN}{Style.BOLD}启动服务...{Style.RESET}")
    print(f"{Style.DIM}  • API: http://localhost:{api_port}/v1/completions")
    print(f"  • 健康检查: http://localhost:{api_port}/health")
    print(f"  • 模型: {os.path.basename(model_path) if mtype == 'gguf' else model_path}")
    print(f"  • 推测: {'已启用' if use_spec else '未启用'}")
    print(f"  • 超长上下文: {context_method if context_method != 'none' else '未启用'}")
    print(f"{Style.DIM}  • 按 Ctrl+C 停止服务{Style.RESET}\n")
    sys.stdout.flush()

    try:
        run_command(cmd, "服务运行中")
    except KeyboardInterrupt:
        print(f"\n{Style.YELLOW}已停止{Style.RESET}")


def show_status():
    """显示系统状态。"""
    print(f"\n{Style.BOLD}系统状态{Style.RESET}")
    print(f"{Style.DIM}─" * 40 + Style.RESET)

    # Python 版本
    print(f"  Python:     {sys.version.split()[0]}")

    # GPU 检测
    try:
        import torch
        print(f"  PyTorch:    {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  GPU:        {torch.cuda.get_device_name(0)}")
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"  显存:       {free_mem/1024**3:.1f}GB / {total_mem/1024**3:.1f}GB 可用")
        else:
            print(f"{Style.YELLOW}  GPU:        未检测到 CUDA 设备{Style.RESET}")
    except ImportError:
        print(f"{Style.RED}  PyTorch:    未安装{Style.RESET}")

    # GGUF 支持
    try:
        from model_loader.gguf_reader import GGUFFile
        print(f"  GGUF:       支持")
    except ImportError:
        print(f"{Style.YELLOW}  GGUF:       不可用{Style.RESET}")

    # Goose 推测
    try:
        import goose_core
        print(f"  Goose:      可用 (Phase 0/1)")
    except ImportError:
        print(f"{Style.YELLOW}  Goose:      不可用{Style.RESET}")

    # 发现的模型
    ggufs = find_gguf_models()
    if ggufs:
        print(f"  模型文件:   {len(ggufs)} 个 .gguf 文件")
        for m in ggufs[:3]:
            size = os.path.getsize(m) / 1024**3
            print(f"              • {os.path.basename(m)} ({size:.1f}GB)")
        if len(ggufs) > 3:
            print(f"              • ... 还有 {len(ggufs)-3} 个")
    else:
        print(f"{Style.YELLOW}  模型文件:   未找到 .gguf 文件{Style.RESET}")

    print()


# ── 启动入口 ────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    # 无参数 → 交互式菜单
    # --auto    → 自动：测试 → 启动
    # --quick   → 跳过测试，直接启动

    print(BANNER)

    auto_mode = "--auto" in args
    quick_mode = "--quick" in args

    # 非交互模式下：自动选择模型，自动运行
    if auto_mode or quick_mode or not sys.stdout.isatty():
        model_path, mtype = ask_model(args)
        use_spec = "--speculative" in args or "-s" in args
        context_method = parse_context_method(args)

        if not quick_mode:
            rc = run_benchmark(model_path, mtype,
                               prompt_len=int(os.environ.get("BENCH_PROMPT_LEN", "128")),
                               gen_len=int(os.environ.get("BENCH_GEN_LEN", "128")),
                               use_spec=use_spec,
                               context_method=context_method)
            if rc != 0:
                print(f"{Style.RED}测试失败 (exit={rc})。是否继续启动？{Style.RESET}")
                try:
                    resp = input("[y/N]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    resp = "n"
                if resp != "y":
                    sys.exit(rc)

        port = int(os.environ.get("API_PORT", "8000"))
        run_interactive(model_path, mtype, api_port=port, use_spec=use_spec,
                        context_method=context_method)
        return

    # ── 交互式菜单 ────────────────────────────────────────────────

    model_path = None
    mtype = "gguf"

    while True:
        print(MENU)
        choice = input(f"{Style.CYAN}请输入选项: {Style.RESET}").strip().lower()

        if choice == "q":
            print(f"{Style.YELLOW}再见！{Style.RESET}")
            break

        if choice == "4":
            show_status()
            input(f"{Style.DIM}按 Enter 继续...{Style.RESET}")
            continue

        if choice in ("1", "2", "3"):
            # 选择模型（首次或记忆）
            if model_path is None:
                model_path, mtype = ask_model(args)

            use_spec_input = input(f"{Style.CYAN}启用 Goose 推测解码？(y/N): {Style.RESET}").strip().lower()
            use_spec = use_spec_input == "y"
            print(f"\n  {Style.DIM}💡 超长上下文扩展默认已开启 (SelfExtend)")
            print(f"  如需切换，启动后用 --context-method 指定{Style.RESET}")
            context_method = "selfextend"  # 默认开启，不再询问

            if choice == "2":
                # 仅基准测试
                run_benchmark(model_path, mtype, use_spec=use_spec,
                              context_method=context_method)
                input(f"\n{Style.DIM}按 Enter 继续...{Style.RESET}")

            elif choice == "1":
                # 完整流程：先测试，再启动
                rc = run_benchmark(model_path, mtype, use_spec=use_spec,
                                    context_method=context_method)
                if rc != 0:
                    print(f"{Style.RED}测试失败。仍要启动服务吗？{Style.RESET}")
                    cont = input("[y/N]: ").strip().lower()
                    if cont != "y":
                        continue
                run_interactive(model_path, mtype, use_spec=use_spec,
                                context_method=context_method)
                break  # 服务结束后退出

            elif choice == "3":
                # 直接启动
                run_interactive(model_path, mtype, use_spec=use_spec,
                                context_method=context_method)
                break
        else:
            print(f"{Style.RED}无效选项，请重新输入{Style.RESET}")


def windows_entry():
    """Windows 双击兼容入口。"""
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Style.YELLOW}已退出。{Style.RESET}")
    except Exception as e:
        print(f"\n{Style.RED}发生错误: {e}{Style.RESET}")
        import traceback
        traceback.print_exc()
    finally:
        if sys.stdout.isatty() and os.name == "nt":
            input(f"\n{Style.DIM}按 Enter 退出...{Style.RESET}")


if __name__ == "__main__":
    # Windows: 双击运行时 stdout 可能没有行缓冲
    if os.name == "nt":
        sys.stdout.reconfigure(line_buffering=True, errors="replace")
    windows_entry()
