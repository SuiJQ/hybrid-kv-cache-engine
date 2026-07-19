@echo off
chcp 65001 >nul
title MoeOwner 推理引擎

echo.
echo  ╔══════════════════════════════════════════╗
echo  ║                                          ║
echo  ║   MoeOwner 推理引擎                       ║
echo  ║   双击运行 | 零配置启动                    ║
echo  ║                                          ║
echo  ╚══════════════════════════════════════════╝
echo.

:: ── 查找 Python ──
set PYTHON=
for %%p in (python python3 py) do (
    where %%p >nul 2>&1 && set PYTHON=%%p && goto FOUND
)
echo [错误] 未找到 Python，请先安装 Python 3.10+
pause
exit /b

:FOUND
echo  使用 Python: %PYTHON%
echo.

:: ── 进入脚本所在目录 ──
cd /d "%~dp0"

:: ── 启动交互菜单 ──
%PYTHON% launch.py

pause
