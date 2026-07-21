#!/usr/bin/env python3
"""
tool_sink.py — 推理框架工具下沉 (Tool-Sinking Inference Framework)

自包含·硬化版·重构

Provides a self-contained tool-calling framework that intercepts streaming
model output, detects ``[[tool(...)]]`` markers, executes built-in tools,
reconstructs conversation history, and re-invokes the model for multi-turn
reasoning.

Integration
-----------
Use ``ToolOrchestrator.generate()`` as a drop-in wrapper around the
scheduler's submit+poll flow.  Designed for MoeOwner inference engine,
requires zero third-party dependencies.

Conventions
-----------
- All tools: ``def tool(**kwargs) -> str | int | float | dict | list``
- Tool call syntax: ``[[tool_name(key=value, ...)]]`` (one per line)
- Max 3 turns per request (LOOP_LIMIT_EXCEEDED after 3)
- Tools execute in-process (serial, never parallel)
- Output is always cleaned through ``format_fixer`` before returning

Windows & NVIDIA GPU Compatibility
----------------------------------
- ``subprocess`` sandbox uses ``start_new_session=True`` (ignored on Windows).
- ``signal.alarm`` is NOT used (Windows-incompatible); threading Timer for timeout.
- ``os.uname()`` fallback for Windows via ``platform`` module.
- All file operations use ``shutil.rmtree`` (cross-platform).
"""

from __future__ import annotations

import ast
import gc
import html.parser
import json
import logging
import math
import cmath
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

_MAX_TURNS: int = 3
_TOOL_MSG_CAP: int = 100 * 1024  # 100 KB per tool message
_SANDBOX_TIMEOUT: float = 5.0
_SANDBOX_OUTPUT_CAP: int = 64 * 1024  # 64 KB
_FETCH_TIMEOUT_DNS: float = 5.0
_FETCH_TIMEOUT_CONN: float = 5.0
_FETCH_TIMEOUT_READ: float = 10.0
_RECURSION_LIMIT: int = 1000
_SCALAR_EVAL_TIMEOUT: float = 1.0
_MEMORY_MELT_THRESHOLD_MB: int = 512

_TOOL_PATTERN: re.Pattern = re.compile(r'^\[\[(\w+)\((.+)\)\]\]$')

# Builtins white-list for sandbox execution
_SANDBOX_SAFE_BUILTINS: dict[str, Any] = {
    'True': True, 'False': False, 'None': None,
    'int': int, 'float': float, 'str': str,
    'list': list, 'tuple': tuple, 'dict': dict,
    'print': print, 'len': len, 'range': range,
    'sum': sum, 'min': min, 'max': max,
    'abs': abs, 'round': round, 'pow': pow,
}

# ═══════════════════════════════════════════════════════════════════════
# HTML Text Extractor (stdlib only)
# ═══════════════════════════════════════════════════════════════════════


class _HTMLTextExtractor(html.parser.HTMLParser):
    """Extracts visible text from HTML, stripping script/style tags."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_tag: str | None = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ('script', 'style'):
            self._skip_tag = tag

    def handle_endtag(self, tag: str) -> None:
        if self._skip_tag == tag:
            self._skip_tag = None

    def handle_data(self, data: str) -> None:
        if self._skip_tag is None:
            text = data.strip()
            if text:
                self._text_parts.append(text)

    def get_text(self) -> str:
        return ' '.join(self._text_parts)


# ═══════════════════════════════════════════════════════════════════════
# format_fixer — JSON output cleanup (tool + hook dual role)
# ═══════════════════════════════════════════════════════════════════════


def format_fixer(raw: str) -> str:
    """Clean and validate JSON-like text from model output.

    - Removes trailing commas
    - Fixes mixed single/double quotes
    - Validates via ``json.loads``
    - On failure: truncates to last valid JSON structure + ``[FORMAT_TRUNCATED]``

    This is both a user-callable tool and the mandatory final output hook.
    """
    if not raw or not raw.strip():
        return raw

    # Stage 1: basic character cleanup
    cleaned = raw.strip()

    # Remove trailing commas before closing brackets/braces
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)

    # Replace single quotes used as string delimiters with double quotes
    # Only do this for top-level strings (heuristic: outside brackets)
    # Use a simple approach: replace single-quote-wrapped strings
    cleaned = _fix_single_quotes(cleaned)

    # Stage 2: try parsing
    try:
        json.loads(cleaned)
        return cleaned  # Valid JSON as-is
    except (json.JSONDecodeError, ValueError):
        pass

    # Stage 3: try stripping trailing garbage
    # Scan backwards through the string to find the last valid JSON
    for end in range(len(cleaned), 0, -1):
        candidate = cleaned[:end]
        try:
            json.loads(candidate)
            return candidate + "[FORMAT_TRUNCATED]"
        except (json.JSONDecodeError, ValueError):
            continue

    # Nothing worked — return original with truncation marker
    # (but cap it so we don't return unbounded garbage)
    capped = cleaned[:_TOOL_MSG_CAP]
    return capped + "[FORMAT_TRUNCATED]"


def _fix_single_quotes(text: str) -> str:
    """Replace single-quoted strings with double-quoted ones, heuristically."""
    result = []
    i = 0
    in_string = False
    string_char = None
    escape = False

    while i < len(text):
        ch = text[i]

        if escape:
            result.append(ch)
            escape = False
            i += 1
            continue

        if ch == '\\' and in_string:
            result.append(ch)
            escape = True
            i += 1
            continue

        if ch in ("'", '"') and not in_string:
            in_string = True
            string_char = ch
            if ch == "'":
                result.append('"')  # Normalize to double quote
            else:
                result.append(ch)
            i += 1
            continue

        if ch == string_char and in_string:
            in_string = False
            string_char = None
            if ch == "'":
                result.append('"')  # Normalize to double quote
            else:
                result.append(ch)
            i += 1
            continue

        result.append(ch)
        i += 1

    return ''.join(result)


# ═══════════════════════════════════════════════════════════════════════
# ToolScanner — character-by-character state machine
# ═══════════════════════════════════════════════════════════════════════

# States
_TEXT = 0
_LEFT_BRACKET = 1       # saw one [
_LEFT_DOUBLE = 2        # saw [[, accumulating tool call
_SAW_CLOSE_BRACKET = 3  # saw one ] inside [[...]] — waiting for second ]


class ToolScanner:
    """Character-level state machine that detects ``[[tool(...)]]`` markers.

    Detects the pattern ``[[tool_name(key=value, ...)]]`` by processing
    one character at a time.  When a complete tool call is found, its
    full text (including the ``[[ ]]`` brackets) is returned.

    Text emitted via ``._output_parts`` contains everything *before* the
    first tool call — once a tool call is returned, the scanner resets
    and the caller is responsible for acting on the result.
    """

    def __init__(self) -> None:
        self._state: int = _TEXT
        self._buffer: list[str] = []          # Accumulated text before tool call
        self._tag_buf: list[str] = []         # Characters inside [[...]]
        self._output_parts: list[str] = []    # All text output so far (pre-tool)
        self._pending_tool: str | None = None

    @property
    def accumulated_text(self) -> str:
        """Return all accumulated text that has been emitted so far."""
        return ''.join(self._output_parts)

    def feed(self, char: str) -> str | None:
        """Feed one character. Returns the full tool call string when detected."""
        if self._state == _TEXT:
            if char == '[':
                self._state = _LEFT_BRACKET
                self._tag_buf = ['[']
            else:
                self._output_parts.append(char)
                self._buffer.append(char)

        elif self._state == _LEFT_BRACKET:
            if char == '[':
                self._state = _LEFT_DOUBLE
                self._tag_buf.append('[')
            else:
                # False alarm — emit '[' and current char as text
                self._output_parts.append('[')
                self._output_parts.append(char)
                self._buffer.append('[')
                self._buffer.append(char)
                self._state = _TEXT

        elif self._state == _LEFT_DOUBLE:
            if char == ']':
                self._state = _SAW_CLOSE_BRACKET
                self._tag_buf.append(']')
            else:
                self._tag_buf.append(char)

        elif self._state == _SAW_CLOSE_BRACKET:
            # Previous char was a single ] — now check what follows
            if char == ']':
                # Second ] — we have [[...]] complete
                self._tag_buf.append(']')
                raw = ''.join(self._tag_buf)
                m = _TOOL_PATTERN.match(raw)
                if m:
                    self._pending_tool = raw
                    self._state = _TEXT
                    return raw
                else:
                    # Matches [[...]] pattern but not a valid tool call
                    self._output_parts.append(raw)
                    self._buffer.append(raw)
                    self._tag_buf.clear()
                    self._state = _TEXT
            else:
                # Was just a single ] in content — not a tool call close
                content = ''.join(self._tag_buf)
                self._output_parts.append(content)
                self._buffer.append(content)
                self._tag_buf.clear()
                # Current char starts new content
                self._output_parts.append(char)
                self._buffer.append(char)
                self._state = _TEXT

        return None

    def flush(self) -> str:
        """Return any remaining buffered text. Call after stream ends."""
        if self._tag_buf:
            remaining = ''.join(self._tag_buf)
            self._output_parts.append(remaining)
            self._buffer.append(remaining)
            self._tag_buf.clear()
        self._state = _TEXT
        return ''.join(self._buffer)


# ═══════════════════════════════════════════════════════════════════════
# ToolContext — lifecycle container for one user request
# ═══════════════════════════════════════════════════════════════════════


class ToolContext:
    """Request-scoped context with temp directory, memo store, and thread safety.

    Destroy semantics: synchronous cleanup of memory + temp dir + child procs.
    Once destroyed (``.invalid == True``), all method calls are rejected.
    """

    def __init__(self, user_query: str) -> None:
        self._tmpdir: str = tempfile.mkdtemp(
            prefix="ctx_",
            suffix=os.urandom(8).hex(),
        )
        self._memo: dict[str, str] = {}
        self._msg_history: list[dict[str, str]] = [
            {"role": "user", "content": user_query}
        ]
        self._user_query: str = user_query
        self._turn: int = 0
        self._lock: threading.RLock = threading.RLock()
        self._invalid: bool = False
        self._child_procs: list[subprocess.Popen] = []

    # ── Lifecycle ───────────────────────────────────────────────────

    @property
    def invalid(self) -> bool:
        return self._invalid

    @property
    def turn(self) -> int:
        return self._turn

    def advance_turn(self) -> int:
        with self._lock:
            self._turn += 1
            return self._turn

    def destroy(self) -> None:
        """Synchronous blocking destroy. Blocks up to 5s for temp dir removal."""
        with self._lock:
            if self._invalid:
                return
            self._invalid = True

            # Terminate child processes
            for proc in self._child_procs:
                try:
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait(timeout=2)
                        if proc.poll() is None:
                            proc.kill()
                except Exception:
                    pass
            self._child_procs.clear()

            # Clear memory
            self._memo.clear()
            gc.collect()

            # Remove temp directory (synchronous)
            def _rmtree() -> None:
                try:
                    shutil.rmtree(self._tmpdir, ignore_errors=True)
                except Exception:
                    pass

            t = threading.Thread(target=_rmtree, daemon=True)
            t.start()
            t.join(timeout=5.0)

    def _check_valid(self) -> None:
        if self._invalid:
            raise RuntimeError("ToolContext has been destroyed")

    # ── Message history management ─────────────────────────────────

    def push_message(self, role: str, content: str) -> None:
        with self._lock:
            self._check_valid()
            self._msg_history.append({"role": role, "content": content})

    def reset_history(self, messages: list[dict[str, str]]) -> None:
        """Replace the entire history (for turn reconstruction)."""
        with self._lock:
            self._check_valid()
            self._msg_history = list(messages)

    def get_history(self) -> list[dict[str, str]]:
        with self._lock:
            self._check_valid()
            return list(self._msg_history)

    def build_prompt(self) -> str:
        """Build a text prompt from the current message history."""
        with self._lock:
            self._check_valid()
            parts: list[str] = []
            for msg in self._msg_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    parts.append(f"System: {content}")
                elif role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
                elif role == "tool":
                    parts.append(f"[Tool result]: {content}")
                else:
                    parts.append(content)
            return "\n".join(parts)

    # ── Tool: memo_set ─────────────────────────────────────────────

    def tool_memo_set(self, **kwargs: Any) -> str:
        """memo_set(key: str, value: str) -> str"""
        key = str(kwargs.get("key", ""))
        value = str(kwargs.get("value", ""))
        if not key:
            return "ERROR: memo_set requires non-empty 'key'"
        with self._lock:
            self._check_valid()
            self._memo[key] = value
        return "OK"

    # ── Tool: memo_get ─────────────────────────────────────────────

    def tool_memo_get(self, **kwargs: Any) -> str:
        """memo_get(key: str) -> str"""
        key = str(kwargs.get("key", ""))
        with self._lock:
            self._check_valid()
            return self._memo.get(key, "")

    # ── Tool: sci_calc ─────────────────────────────────────────────

    def tool_sci_calc(self, **kwargs: Any) -> str:
        """sci_calc(expression: str) -> str"""
        expression = str(kwargs.get("expression", ""))
        if not expression:
            return "ERROR: sci_calc requires 'expression' string"

        # Validate data shape for list/tuple inputs in expression
        # We do a simple scan: if the expression contains "[[...]]" or nested lists
        # that exceed 2D or 100 elements, reject.
        try:
            _validate_tensor_shape_in_expr(expression)
        except ValueError as exc:
            return f"ERROR: {exc}"

        # Restrict recursion depth
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(_RECURSION_LIMIT)

        result_container: list[str] = []

        def _eval_safe() -> None:
            try:
                eval_globals: dict[str, Any] = {
                    'math': math,
                    'cmath': cmath,
                    '__builtins__': {},
                }
                val = eval(expression, eval_globals)
                if isinstance(val, (int, float)):
                    if math.isnan(val) or math.isinf(val):
                        result_container.append("null")
                    else:
                        result_container.append(str(val))
                elif isinstance(val, complex):
                    result_container.append(str(val))
                elif isinstance(val, (list, tuple)):
                    result_container.append(str(val))
                else:
                    result_container.append(str(val))
            except Exception as exc:
                result_container.append(f"ERROR: {exc}")

        # Run eval in a daemon thread with timeout
        eval_thread = threading.Thread(target=_eval_safe, daemon=True)
        eval_thread.start()
        eval_thread.join(timeout=_SCALAR_EVAL_TIMEOUT)
        sys.setrecursionlimit(old_limit)

        if eval_thread.is_alive():
            return "ERROR: sci_calc timed out (>1s)"

        return result_container[0] if result_container else "ERROR: no result"

    # ── Tool: sys_env ──────────────────────────────────────────────

    def tool_sys_env(self, **kwargs: Any) -> dict:
        """sys_env() -> dict"""
        _ = kwargs  # no args expected
        info: dict[str, Any] = {}
        try:
            uname = os.uname()
            info["sysname"] = uname.sysname
            info["nodename"] = uname.nodename
            info["release"] = uname.release
            info["machine"] = uname.machine
        except AttributeError:
            # Windows fallback
            info["sysname"] = platform.system()
            info["release"] = platform.release()
            info["machine"] = platform.machine()

        info["cpu_count"] = os.cpu_count() or 0

        try:
            page_size = os.sysconf('SC_PAGE_SIZE')
            phys_pages = os.sysconf('SC_PHYS_PAGES')
            info["memory_total_bytes"] = page_size * phys_pages
        except (AttributeError, ValueError, OSError):
            # Windows fallback
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                info["memory_total_bytes"] = kernel32.GlobalMemoryStatusEx().ullTotalPhys
            except Exception:
                info["memory_total_bytes"] = "unavailable"

        info["python_version"] = sys.version
        info["platform"] = sys.platform
        return info

    # ── Tool: list_ports ───────────────────────────────────────────

    def tool_list_ports(self, **kwargs: Any) -> list:
        """list_ports() -> list of listening port numbers."""
        _ = kwargs
        ports: list[int] = []
        errors: list[str] = []

        # Try /proc/net/tcp first (Linux)
        for proc_path in ("/proc/net/tcp", "/proc/net/tcp6"):
            ports.extend(self._parse_proc_net(proc_path, errors))

        if ports:
            return sorted(set(ports))

        # Fallback: shell commands
        try:
            if sys.platform.startswith("linux"):
                result = subprocess.run(
                    ["ss", "-tuln"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    ports = self._parse_ss_output(result.stdout)
                else:
                    errors.append(result.stderr[:256].strip())
            else:
                result = subprocess.run(
                    ["netstat", "-an"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    ports = self._parse_netstat_output(result.stdout)
                else:
                    errors.append(result.stderr[:256].strip())
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            errors.append(str(exc)[:128])

        if errors and not ports:
            logger.debug("list_ports errors: %s", "; ".join(errors))

        return sorted(set(ports))

    @staticmethod
    def _parse_proc_net(path: str, errors: list[str]) -> list[int]:
        ports: list[int] = []
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        local_addr = parts[1]
                        state = parts[3]
                        if state == "0A":  # TCP_LISTEN
                            if ":" in local_addr:
                                hex_port = local_addr.split(":")[1]
                                ports.append(int(hex_port, 16))
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            errors.append(str(exc)[:128])
        return ports

    @staticmethod
    def _parse_ss_output(text: str) -> list[int]:
        ports: list[int] = []
        for line in text.splitlines():
            parts = line.split()
            for part in parts:
                if ":" in part:
                    try:
                        port_candidate = part.split(":")[-1]
                        ports.append(int(port_candidate))
                    except (ValueError, IndexError):
                        continue
        return ports

    @staticmethod
    def _parse_netstat_output(text: str) -> list[int]:
        ports: list[int] = []
        for line in text.splitlines():
            parts = line.split()
            for part in parts:
                if ":" in part and part.rfind(":") != part.find(":"):
                    # IPv4:port
                    try:
                        port_candidate = part.split(":")[-1]
                        ports.append(int(port_candidate))
                    except (ValueError, IndexError):
                        continue
                elif ":" in part and part.count(":") == 1:
                    # Simple host:port
                    try:
                        port_candidate = part.split(":")[-1]
                        ports.append(int(port_candidate))
                    except (ValueError, IndexError):
                        continue
        return ports

    # ── Tool: ping_target ──────────────────────────────────────────

    def tool_ping_target(self, **kwargs: Any) -> str:
        """ping_target(host: str, port: int = 80, timeout: float = 3.0) -> str (JSON)."""
        host = str(kwargs.get("host", ""))
        port = int(kwargs.get("port", 80))
        timeout = float(kwargs.get("timeout", 3.0))

        if not host:
            return json.dumps({"reachable": False, "error": "No host provided", "latency_ms": None})

        total_timeout = timeout * 2  # Retry budget
        start = time.monotonic()

        for attempt in range(2):  # Max 2 attempts (1 initial + 1 retry)
            attempt_start = time.monotonic()
            remaining = total_timeout - (attempt_start - start)
            if remaining <= 0:
                break

            try:
                sock = socket.create_connection((host, port), timeout=min(timeout, remaining))
                elapsed = (time.monotonic() - attempt_start) * 1000  # ms
                sock.close()
                return json.dumps({
                    "reachable": True,
                    "latency_ms": round(elapsed, 2),
                    "host": host,
                    "port": port,
                })
            except (socket.timeout, ConnectionRefusedError, OSError) as exc:
                if attempt == 0:
                    continue
                return json.dumps({
                    "reachable": False,
                    "error": str(exc)[:256],
                    "latency_ms": None,
                })

        return json.dumps({
            "reachable": False,
            "error": "All retries exhausted",
            "latency_ms": None,
        })

    # ── Tool: format_fixer (static, also available as tool) ────────

    @staticmethod
    def tool_format_fixer(**kwargs: Any) -> str:
        """format_fixer(raw: str) -> str"""
        raw = str(kwargs.get("raw", ""))
        return format_fixer(raw)

    # ── Tool: sandbox_run ──────────────────────────────────────────

    def tool_sandbox_run(self, **kwargs: Any) -> str:
        """sandbox_run(code: str) -> str — subprocess sandbox with restricted builtins."""
        code = str(kwargs.get("code", ""))
        if not code:
            return "ERROR: sandbox_run requires 'code' string"

        # Wrap code with restricted builtins
        sandbox_code = self._build_sandbox_code(code)

        try:
            proc = subprocess.Popen(
                [sys.executable, "-c", sandbox_code],
                cwd=self._tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            self._child_procs.append(proc)

            try:
                stdout, stderr = proc.communicate(timeout=_SANDBOX_TIMEOUT)
            except subprocess.TimeoutExpired:
                # SIGKILL process group
                try:
                    os.killpg(os.getpgid(proc.pid), 9)
                except (ProcessLookupError, OSError):
                    try:
                        proc.kill()
                    except OSError:
                        pass
                return "Timeout"

            output = ""
            if stdout:
                output += stdout.decode("utf-8", errors="replace")
            if stderr:
                err_text = stderr.decode("utf-8", errors="replace")
                if err_text.strip():
                    if output:
                        output += "\n[STDERR]\n" + err_text
                    else:
                        output = err_text

            # Cap output size
            if len(output) > _SANDBOX_OUTPUT_CAP:
                output = output[:_SANDBOX_OUTPUT_CAP] + "[TRUNCATED]"

            return output

        except FileNotFoundError:
            return "ERROR: Python interpreter not found"
        except OSError as exc:
            return f"ERROR: sandbox execution failed: {exc}"

    @staticmethod
    def _build_sandbox_code(user_code: str) -> str:
        """Wrap user code with restricted builtins and import math.

        Builds the builtins dict from name-to-name mappings.  Uses
        a stored reference to ``exec`` before ``__builtins__`` is
        cleared, so the sandbox wrapper itself does not break.
        """
        # Map each key to what it should resolve to at runtime
        items: list[str] = []
        for k, v in _SANDBOX_SAFE_BUILTINS.items():
            if isinstance(v, bool):
                items.append(f"'{k}': {repr(v)}")
            elif v is None:
                items.append(f"'{k}': None")
            elif isinstance(v, type):
                items.append(f"'{k}': _SB_TYPE_{k}")
            else:
                # Functions: store reference before clearing
                items.append(f"'{k}': _SB_FN_{k}")

        builtins_src = "{" + ", ".join(items) + "}"

        # Build type definitions (names like int, float, str are accessible)
        type_defs = []
        fn_defs = []
        for k, v in _SANDBOX_SAFE_BUILTINS.items():
            if isinstance(v, type):
                type_defs.append(f"_SB_TYPE_{k} = {k}")
            elif not isinstance(v, bool) and v is not None:
                fn_defs.append(f"_SB_FN_{k} = {k}")

        type_str = "; ".join(type_defs)
        fn_str = "; ".join(fn_defs)

        # Escape the user code for safe embedding
        escaped_code = user_code.replace('\\', '\\\\').replace("'", "\\'")
        return (
            "import math\n"
            "import builtins as _sb_builtins_mod\n"
            f"{type_str}\n"
            f"{fn_str}\n"
            "_sb_exec = _sb_builtins_mod.exec\n"
            "__builtins__.__dict__.clear()\n"
            f"__builtins__.__dict__.update({builtins_src})\n"
            f"_sb_exec('''{escaped_code}''')\n"
        )

    # ── Tool: fetch_url ────────────────────────────────────────────

    def tool_fetch_url(self, **kwargs: Any) -> str:
        """fetch_url(url: str, max_chars: int = 50000) -> str"""
        url = str(kwargs.get("url", ""))
        max_chars = int(kwargs.get("max_chars", 50000))

        if not url:
            return "ERROR: fetch_url requires 'url'"

        # Protocol validation
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"ERROR: Unsupported protocol '{parsed.scheme}'. Only http:// and https:// allowed."

        # Build request with fixed User-Agent
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; SelfHostedBot/1.0)",
            },
        )

        # Set global socket timeout for DNS+connect
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(_FETCH_TIMEOUT_DNS)

        try:
            response = urllib.request.urlopen(
                req,
                timeout=_FETCH_TIMEOUT_CONN + _FETCH_TIMEOUT_READ,
            )

            # Check Content-Type
            content_type = response.headers.get("Content-Type", "")
            if any(skip in content_type for skip in ("application/octet-stream", "image/", "video/", "audio/")):
                return f"ERROR: Skipped binary content (Content-Type: {content_type})"

            # Read response with timeout
            body = response.read(_FETCH_TIMEOUT_READ * 1024 * 100)  # up to ~1MB in timeout budget

            # Try to decode as text
            charset = response.headers.get_content_charset()
            try:
                text = body.decode(charset or "utf-8", errors="replace")
            except (LookupError, UnicodeDecodeError):
                text = body.decode("utf-8", errors="replace")

            # Extract visible text from HTML
            extractor = _HTMLTextExtractor()
            extractor.feed(text)
            visible_text = extractor.get_text()

            # Truncate at sentence boundary
            if len(visible_text) > max_chars:
                cut = visible_text[:max_chars]
                # Try to find last sentence boundary
                last_period = cut.rfind('.')
                last_newline = cut.rfind('\n')
                split_at = max(last_period, last_newline)
                if split_at > max_chars // 2:
                    visible_text = visible_text[:split_at + 1]
                else:
                    visible_text = cut

            return visible_text

        except urllib.error.HTTPError as exc:
            return (
                f"HTTP {exc.code}: {exc.reason}\n"
                f"Headers: {dict(exc.headers)}\n"
            )
        except urllib.error.URLError as exc:
            reason = exc.reason
            return f"URL Error: {reason}"[:1024]
        except (socket.timeout, OSError) as exc:
            return f"TIMEOUT: {exc}"[:512]
        finally:
            socket.setdefaulttimeout(old_timeout)

    # ── Tool dispatch ─────────────────────────────────────────────

    _TOOL_REGISTRY: dict[str, Callable[..., Any]] = {}

    @classmethod
    def _init_registry(cls) -> None:
        if cls._TOOL_REGISTRY:
            return
        cls._TOOL_REGISTRY = {
            "memo_set": cls.tool_memo_set,
            "memo_get": cls.tool_memo_get,
            "sci_calc": cls.tool_sci_calc,
            "sys_env": cls.tool_sys_env,
            "list_ports": cls.tool_list_ports,
            "ping_target": cls.tool_ping_target,
            "format_fixer": cls.tool_format_fixer,
            "sandbox_run": cls.tool_sandbox_run,
            "fetch_url": cls.tool_fetch_url,
        }

    def execute(self, tool_name: str, args_str: str) -> str:
        """Parse args_str, look up tool, execute it.

        Args are parsed keyword-style: ``key=value, key2=value2``.
        Falls back to positional ``query`` param on parse failure.
        """
        self._init_registry()

        if tool_name not in self._TOOL_REGISTRY:
            return f"ERROR: Unknown tool '{tool_name}'. Available: {', '.join(sorted(self._TOOL_REGISTRY))}"

        # Parse arguments
        kwargs = self._parse_args(args_str)
        if kwargs is None:
            return "ERROR: Tool signature mismatch. Expected key=value."

        # Execute
        try:
            method = self._TOOL_REGISTRY[tool_name]
            result = method(self, **kwargs) if tool_name not in ("format_fixer",) else method(**kwargs)
            # Ensure serializable
            return self._serialize_result(result)
        except Exception as exc:
            return f"ERROR: Tool '{tool_name}' execution failed: {exc}"

    def _parse_args(self, args_str: str) -> dict[str, Any] | None:
        """Parse ``key=value, key2=value2`` into a dict.

        Attempts in order:
        1. Wraps in braces and tries ``ast.literal_eval`` (expects
           ``'key': value`` style with quoted keys).
        2. Tries ``json.loads`` on the braced string.
        3. Custom ``key=value`` parser (handles unquoted keys,
           quoted and unquoted string values, numeric values).
        4. Return ``{"query": args_str}`` as positional fallback.
        """
        args_str = args_str.strip()

        if not args_str:
            return {}

        # Attempt 1: ast.literal_eval (quoted keys: 'key': value)
        try:
            return ast.literal_eval("{" + args_str + "}")
        except (SyntaxError, ValueError, TypeError):
            pass

        # Attempt 2: json.loads (double-quoted keys: "key": value)
        try:
            json_str = _fix_single_quotes("{" + args_str + "}")
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

        # Attempt 3: Custom key=value parser
        result = self._parse_key_value(args_str)
        if result is not None:
            return result

        # Attempt 4: positional fallback
        return {"query": args_str}

    @staticmethod
    def _parse_key_value(text: str) -> dict[str, Any] | None:
        """Parse ``key="value", key2=123`` style keyword arguments.

        Returns a dict or None on parse failure.
        """
        result: dict[str, Any] = {}
        # Match key=value pairs, handling quoted and unquoted values
        # Pattern: key = "quoted value" OR key = unquoted_value OR key = 123
        # The trailing ",\s*" consumes the comma AND any whitespace after
        # it, so the next match starts on the next key name.
        pattern = re.compile(
            r'(\w+)\s*=\s*'
            r'(?:"([^"]*)"|'         # double-quoted
            r"'([^']*)'|"             # single-quoted
            r'([^,\s][^,]*?))'        # unquoted (up to comma or end)
            r'\s*(?:,\s*|$)'
        )
        pos = 0
        while pos < len(text):
            m = pattern.match(text, pos)
            if not m:
                return None
            key = m.group(1)
            # Value: one of the alternations should match
            value = (m.group(2) or m.group(3) or m.group(4) or '').strip()
            # Try to coerce numeric values
            try:
                if '.' in value or 'e' in value.lower():
                    value = float(value)
                else:
                    value = int(value)
            except (ValueError, TypeError):
                pass  # Keep as string
            result[key] = value
            pos = m.end()
        return result if result else None

    @staticmethod
    def _serialize_result(result: Any) -> str:
        """Ensure result is a JSON-serializable string."""
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(result)


# ═══════════════════════════════════════════════════════════════════════
# Validation helpers
# ═══════════════════════════════════════════════════════════════════════


def _validate_tensor_shape_in_expr(expression: str) -> None:
    """Check expression for unsupported data shapes.

    Scans the expression string for list/tuple literals that exceed
    2D or 100 elements per dimension.
    """
    # Walk the AST to find list/tuple literals
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError:
        return  # Let eval handle syntax errors

    _check_list_depth(tree)


def _check_list_depth(node: ast.AST, depth: int = 0) -> None:
    """Recursively check list/tuple depth in AST."""
    if depth > 2:
        raise ValueError(
            "Unsupported data shape. Only scalars, flat lists, "
            "and 2D lists (max 100x100) allowed."
        )

    if isinstance(node, (ast.List, ast.Tuple)):
        if len(node.elts) > 100:
            raise ValueError(
                "Unsupported data shape. Only scalars, flat lists, "
                "and 2D lists (max 100x100) allowed."
            )
        for elt in node.elts:
            _check_list_depth(elt, depth + 1)

    # Recurse into AST elements that might contain lists
    for child in ast.iter_child_nodes(node):
        _check_list_depth(child, depth)


# ═══════════════════════════════════════════════════════════════════════
# ToolOrchestrator — manages the turn loop and scheduler interaction
# ═══════════════════════════════════════════════════════════════════════


class ToolOrchestrator:
    """Orchestrates multi-turn tool-calling inference.

    Wraps the scheduler submit+poll flow with:
    1. Character-level scanning for ``[[tool(...)]]`` markers.
    2. Tool execution with result collection.
    3. History reconstruction and re-submission (max 3 turns).
    4. Output accumulation and ``format_fixer`` final hook.

    This is a synchronous helper designed to be called from async
    API handlers.  It calls the scheduler's ``step()`` loop internally
    through a provided polling mechanism.

    Usage
    -----
    .. code-block:: python

        orchestrator = ToolOrchestrator(scheduler, tokenizer_decoder)
        final_text = await orchestrator.generate(
            user_prompt="Solve 2+2 and tell me",
            max_tokens=512,
        )
    """

    def __init__(
        self,
        scheduler: Any,
        detokenizer: Callable[[list[int]], str] | None = None,
        tokenizer_fn: Callable[[str], list[int]] | None = None,
        enable: bool = True,
    ) -> None:
        self._scheduler = scheduler
        self._detokenizer = detokenizer
        self._tokenizer_fn = tokenizer_fn
        self._enable = enable

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        request_id_prefix: str = "toolsink",
    ) -> str:
        """Run the full tool-sinking inference loop.

        Parameters
        ----------
        prompt:
            The user's question / system prompt.
        max_tokens:
            Max new tokens per inference turn.
        request_id_prefix:
            Prefix for scheduler request IDs.

        Returns
        -------
        Final accumulated text (no tool call markers), passed through
        ``format_fixer`` hook.
        """
        if not self._enable:
            return await self._passthrough_generate(
                prompt, max_tokens, f"{request_id_prefix}-pt",
            )

        ctx = ToolContext(prompt)

        try:
            all_text_parts: list[str] = []
            tool_call_count = 0

            while tool_call_count < _MAX_TURNS:
                # Build prompt from current history
                current_prompt = ctx.build_prompt()
                request_id = f"{request_id_prefix}-{tool_call_count}-{os.urandom(4).hex()}"

                # Submit request and collect full output
                result = await self._submit_and_collect(
                    prompt=current_prompt,
                    max_tokens=max_tokens,
                    request_id=request_id,
                )

                if result is None:
                    all_text_parts.append("[ERROR: Generation timed out]")
                    break

                # ── Scan result for the first [[tool(...)]] marker ──
                # Using state machine scanner for compatibility
                scanner = ToolScanner()
                first_tool_pos = -1
                first_tool_len = 0
                first_tool_name = ""
                first_tool_args = ""

                for pos, ch in enumerate(result):
                    tm = scanner.feed(ch)
                    if tm is not None:
                        m = _TOOL_PATTERN.match(tm)
                        if m:
                            first_tool_pos = pos - len(tm) + 1  # start of [[ in result
                            first_tool_len = len(tm)
                            first_tool_name = m.group(1)
                            first_tool_args = m.group(2)
                            break

                if first_tool_pos == -1:
                    # No tool call — final answer
                    all_text_parts.append(result)
                    break

                # ── Split result into pre-tool and tool parts ──
                pre_tool_text = result[:first_tool_pos]
                tool_marker = result[first_tool_pos:first_tool_pos + first_tool_len]

                # If there's text after the tool call, save it for later
                post_tool_text = result[first_tool_pos + first_tool_len:]

                # Emit the text before the tool call
                if pre_tool_text:
                    all_text_parts.append(pre_tool_text)

                # ── Execute the tool ──
                tool_call_count += 1
                ctx.advance_turn()
                tool_result = ctx.execute(first_tool_name, first_tool_args)

                # Cap tool result
                if len(tool_result) > _TOOL_MSG_CAP:
                    tool_result = tool_result[:_TOOL_MSG_CAP] + "[TRUNCATED]"

                # ── Reconstruct history for next turn ──
                # The assistant's full output (including the tool marker)
                # goes into history so the model can see what it generated.
                assistant_output = result[:first_tool_pos + first_tool_len]
                if post_tool_text:
                    # If there's trailing text from the same assistant turn,
                    # append it to the assistant output in history
                    assistant_output = result

                new_history: list[dict[str, str]] = [
                    {"role": "user", "content": ctx._user_query},
                    {"role": "assistant", "content": assistant_output},
                    {"role": "tool", "content": tool_result},
                    {"role": "assistant", "content": ""},
                ]
                ctx.reset_history(new_history)

                if tool_call_count >= _MAX_TURNS:
                    all_text_parts.append("[LOOP_LIMIT_EXCEEDED]")
                    break

            # ── Join all text parts ──
            final_output = "".join(all_text_parts)

            # ── Apply format_fixer final hook ──
            final_output = format_fixer(final_output)

            return final_output

        except Exception as exc:
            logger.error("ToolOrchestrator error: %s", exc, exc_info=True)
            return f"[ToolSink Error: {exc}]"
        finally:
            ctx.destroy()

    async def _submit_and_collect(
        self,
        prompt: str,
        max_tokens: int,
        request_id: str,
    ) -> str | None:
        """Submit a prompt to the scheduler and collect the full output text.

        Polls ``active_decode_pool`` for the request, accumulates tokens,
        and returns the full decoded text.
        """
        from scheduler import Request  # noqa: PLC0415

        # Tokenize
        prompt_tokens = self._tokenize(prompt)

        # Submit
        req = Request(
            prompt_tokens=prompt_tokens,
            request_id=request_id,
            max_new_tokens=max_tokens,
        )
        self._scheduler.submit(req)

        # Poll
        timeout = max(30.0, max_tokens * 0.5)
        deadline = time.monotonic() + timeout
        completed = None
        last_count = 0
        accumulated_ids: list[int] = []

        while time.monotonic() < deadline:
            await asyncio_sleep(0.05)

            # If the scheduler's main loop is running, we just poll.
            # If not, we need to advance the scheduler ourselves.
            # The API server may need to call step() explicitly.
            # We call step() here to ensure forward progress.
            try:
                await self._scheduler.step()
            except RuntimeError:
                # If no event loop, we can't await step()
                pass

            for dr in self._scheduler.active_decode_pool:
                if dr.request_id == request_id:
                    if len(dr.generated_tokens) > last_count:
                        accumulated_ids.extend(dr.generated_tokens[last_count:])
                        last_count = len(dr.generated_tokens)
                    if dr.is_done:
                        completed = dr
                    break

            if completed is not None:
                break

        if completed is None:
            return None

        # Decode
        return self._detokenize(accumulated_ids)

    async def _passthrough_generate(
        self,
        prompt: str,
        max_tokens: int,
        request_id: str,
    ) -> str:
        """Simple passthrough: submit, poll, return. No tool scanning."""
        result = await self._submit_and_collect(prompt, max_tokens, request_id)
        if result is None:
            return "[Generation timed out]"
        return result

    def _tokenize(self, text: str) -> list[int]:
        """Tokenize text to token IDs."""
        if self._tokenizer_fn is not None:
            try:
                return self._tokenizer_fn(text)
            except Exception:
                pass
        # Fallback: byte-level tokenization
        return list(text.encode("utf-8"))

    def _detokenize(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        if self._detokenizer is not None:
            try:
                return self._detokenizer(token_ids)
            except Exception:
                pass
        # Fallback
        try:
            return bytes(token_ids).decode("utf-8", errors="replace")
        except (ValueError, OverflowError):
            return str(token_ids)


# ═══════════════════════════════════════════════════════════════════════
# Async sleep helper (works with or without running event loop)
# ═══════════════════════════════════════════════════════════════════════


async def asyncio_sleep(seconds: float) -> None:
    """Async sleep that also works in contexts without a running loop."""
    try:
        loop = asyncio_get_running_loop()
        if loop is None or not loop.is_running():
            time.sleep(seconds)
            return
    except RuntimeError:
        time.sleep(seconds)
        return
    import asyncio
    await asyncio.sleep(seconds)


def asyncio_get_running_loop() -> Any | None:
    """Safely get the running event loop, or None."""
    import asyncio
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


# ═══════════════════════════════════════════════════════════════════════
# API Server Integration
# ═══════════════════════════════════════════════════════════════════════


def create_orchestrator(
    scheduler: Any,
    detokenizer: Callable[[list[int]], str] | None = None,
    tokenizer_fn: Callable[[str], list[int]] | None = None,
    enable: bool = True,
) -> ToolOrchestrator:
    """Factory function to create a ToolOrchestrator.

    Call this from the API server to integrate tool sinking.

    Parameters
    ----------
    scheduler:
        The MoeOwner ``UnifiedScheduler`` instance.
    detokenizer:
        Callable ``list[int] → str`` for decoding token IDs.
        Pass ``tokenizer.decode`` from the loaded HF tokenizer.
    tokenizer_fn:
        Callable ``str → list[int]`` for encoding text.
        Pass ``tokenizer.encode`` from the loaded HF tokenizer.
    enable:
        Set to ``False`` to disable tool sinking (passthrough mode).

    Returns
    -------
    A ``ToolOrchestrator`` ready for ``.generate()``.
    """
    return ToolOrchestrator(
        scheduler=scheduler,
        detokenizer=detokenizer,
        tokenizer_fn=tokenizer_fn,
        enable=enable,
    )


# ═══════════════════════════════════════════════════════════════════════
# Standalone test / demo
# ═══════════════════════════════════════════════════════════════════════


def _demo_tools() -> None:
    """Run a quick self-test of all built-in tools."""
    print("=" * 60)
    print("ToolSink — Built-in Tools Self-Test")
    print("=" * 60)

    ctx = ToolContext("test query")

    # 1. memo_set + memo_get
    print("\n[1] memo_set/memo_get")
    r1 = ctx.tool_memo_set(key="test_key", value="hello world")
    print(f"  memo_set: {r1}")
    r2 = ctx.tool_memo_get(key="test_key")
    print(f"  memo_get: {r2}")
    r3 = ctx.tool_memo_get(key="nonexistent")
    print(f"  memo_get (missing): {r3!r}")

    # 2. sci_calc
    print("\n[2] sci_calc")
    r4 = ctx.tool_sci_calc(expression="math.sin(math.pi/2)")
    print(f"  sin(pi/2): {r4}")
    r5 = ctx.tool_sci_calc(expression="math.factorial(5)")
    print(f"  5!: {r5}")
    r6 = ctx.tool_sci_calc(expression="cmath.sqrt(-1+0j)")
    print(f"  sqrt(-1): {r6}")

    # 3. sys_env
    print("\n[3] sys_env")
    r7 = ctx.tool_sys_env()
    print(f"  machine: {r7.get('machine', 'N/A')}")
    print(f"  cpu_count: {r7.get('cpu_count', 'N/A')}")
    print(f"  python: {r7.get('python_version', 'N/A')[:50]}")

    # 4. list_ports
    print("\n[4] list_ports")
    r8 = ctx.tool_list_ports()
    print(f"  ports ({len(r8)}): {r8[:10]}{'...' if len(r8) > 10 else ''}")

    # 5. ping_target
    print("\n[5] ping_target")
    r9 = ctx.tool_ping_target(host="127.0.0.1", port=80, timeout=1.0)
    print(f"  ping localhost:80: {r9}")

    # 6. format_fixer
    print("\n[6] format_fixer")
    r10 = ctx.tool_format_fixer(raw='{"a": 1, "b": 2,}')
    print(f"  fixed JSON: {r10}")

    # 7. sandbox_run
    print("\n[7] sandbox_run")
    r11 = ctx.tool_sandbox_run(code="print(sum([1,2,3,4,5]))")
    print(f"  sandbox: {r11.strip()}")

    # 8. fetch_url (not run by default - network needed)
    print("\n[8] fetch_url (skipped in self-test)")

    # Scanner test
    print("\n" + "=" * 60)
    print("Scanner Self-Test")
    print("=" * 60)
    scanner = ToolScanner()
    test_text = "Hello [[memo_set(key=abc, value=123)]] world"
    calls = []
    for ch in test_text:
        hit = scanner.feed(ch)
        if hit:
            calls.append(hit)
    print(f"  Scanner found {len(calls)} tool call(s)")
    for c in calls:
        print(f"    {c}")
    print(f"  Accumulated text: {scanner.accumulated_text!r}")

    # Cleanup
    ctx.destroy()
    print("\nDone.")


if __name__ == "__main__":
    _demo_tools()
