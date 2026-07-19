#!/usr/bin/env bash
# MoeOwner 一键启动脚本 (macOS / Linux)
# 用法: 在终端中 bash 启动.sh，或 chmod +x 后 ./启动.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 查找 Python ──
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "❌ 未找到 Python，请先安装 Python 3.10+"
    exit 1
fi

echo "  ✓ Python: $($PYTHON --version 2>&1)"

# ── 启动交互菜单 ──
exec "$PYTHON" launch.py
