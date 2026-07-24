"""
api_server.py — Fully OpenAI-compatible async HTTP API server for PyDense.

Endpoints
---------
- ``POST /v1/completions``       — Text completions (streaming + non-streaming)
- ``POST /v1/chat/completions``   — Chat completions (streaming + non-streaming)
- ``GET  /v1/models``             — Model metadata
- ``GET  /health``                — Health check

Deep Thinking (Reasoning) Support
---------------------------------
- Detects ``<think>...</think>`` and ``<thinking>...</thinking>`` blocks in model output.
- Non-streaming: returns ``reasoning_content`` alongside ``content``.
- Streaming: reasoning tokens sent first (``delta.reasoning_content``),
  then answer tokens (``delta.content``).

Built on ``asyncio`` + stdlib — no third-party dependencies.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scheduler import DecodeRequest, UnifiedScheduler

logger = logging.getLogger(__name__)

# ─── Reasoning / Deep Thinking ────────────────────────────────────────────

_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_THINKING_PATTERN = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)

# ─── HTTP primitives ───────────────────────────────────────────────────────

_STATUS_LINES: dict[int, bytes] = {
    200: b"200 OK",
    400: b"400 Bad Request",
    404: b"404 Not Found",
    500: b"500 Internal Server Error",
    503: b"503 Service Unavailable",
}

_CT_JSON = b"application/json"
_CT_PLAIN = b"text/plain"
_CT_SSE = b"text/event-stream"

# System prompt for chat models that don't have one built-in
_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# Common: tuple size for (status_code, body) return values
_TUPLE_RESPONSE_LEN = 2


# ═══════════════════════════════════════════════════════════════════════
# Reasoning / Deep Thinking helpers
# ═══════════════════════════════════════════════════════════════════════


def extract_reasoning(text: str) -> tuple[str, str]:
    """Extract reasoning/thinking content from model output.

    Supports both ``<think>...</think>`` (DeepSeek-R1 style)
    and ``<thinking>...</thinking>`` patterns.

    Returns
    -------
    (reasoning_content, final_content)
        ``reasoning_content`` is ``""`` when no think block is detected.
    """
    # DeepSeek-R1 style: <think>...</think>
    m = _THINK_PATTERN.search(text)
    if m:
        return m.group(1).strip(), _THINK_PATTERN.sub("", text).strip()

    # QwQ / general: <thinking>...</thinking>
    m = _THINKING_PATTERN.search(text)
    if m:
        return m.group(1).strip(), _THINKING_PATTERN.sub("", text).strip()

    return "", text


# ═══════════════════════════════════════════════════════════════════════
# Chat message → prompt conversion
# ═══════════════════════════════════════════════════════════════════════


def _format_chat_prompt(messages: list[dict]) -> str:
    """Convert an OpenAI-format messages list into a text prompt.

    Handles system, user, and assistant roles.  Falls back to simple
    concatenation for unknown roles.  This is model-agnostic; models
    with a native chat template should prefer that instead.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not isinstance(content, str):
            # Handle content arrays (multi-modal) — take text parts only
            if isinstance(content, list):
                text_parts = [
                    c["text"] for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                ]
                content = " ".join(text_parts)
            else:
                content = str(content)

        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(content)

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# HTTP helpers
# ═══════════════════════════════════════════════════════════════════════


async def _recv_headers(reader: asyncio.StreamReader) -> tuple[str, str, dict[str, str]]:
    """Read HTTP/1.1 request line + headers. Returns (method, path, headers)."""
    line = await reader.readline()
    if not line:
        raise ConnectionError("Client disconnected")
    parts = line.decode("utf-8", errors="replace").strip().split(maxsplit=2)
    if len(parts) < _TUPLE_RESPONSE_LEN:
        raise ValueError(f"Malformed request line: {line!r}")
    method, path = parts[0], parts[1]

    headers: dict[str, str] = {}
    while True:
        hdr = await reader.readline()
        if hdr in (b"\r\n", b"\n", b""):
            break
        decoded = hdr.decode("utf-8", errors="replace").strip()
        if ":" in decoded:
            k, v = decoded.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    return method, path, headers


def _parse_content_length(headers: dict[str, str]) -> int:
    raw = headers.get("content-length", "0")
    try:
        return max(0, int(raw))
    except (ValueError, TypeError):
        return 0


async def _read_body(reader: asyncio.StreamReader, length: int) -> bytes:
    if length == 0:
        return b"{}"
    body = b""
    while len(body) < length:
        chunk = await reader.read(length - len(body))
        if not chunk:
            break
        body += chunk
    return body


def _build_response(
    status: int,
    body: dict | str,
    content_type: bytes = _CT_JSON,
    cors: bool = True,
) -> bytes:
    """Build a complete HTTP response (non-streaming)."""
    if isinstance(body, str):
        payload = body.encode("utf-8")
        content_type = _CT_PLAIN if content_type is _CT_JSON else content_type
    else:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")

    header_lines = [
        f"HTTP/1.1 {_STATUS_LINES.get(status, b'200 OK').decode()}",
        f"Content-Type: {content_type.decode()}",
        f"Content-Length: {len(payload)}",
        "Connection: keep-alive",
    ]
    if cors:
        header_lines.extend([
            "Access-Control-Allow-Origin: *",
            "Access-Control-Allow-Methods: POST, GET, OPTIONS",
            "Access-Control-Allow-Headers: Content-Type, Authorization",
        ])
    raw = "\r\n".join(header_lines).encode() + b"\r\n\r\n" + payload
    return raw


async def _send_sse(
    writer: asyncio.StreamWriter,
    event_data: dict,
) -> None:
    """Send one SSE ``data:`` line followed by a blank line."""
    line = f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
    writer.write(line.encode("utf-8"))
    await writer.drain()


async def _send_sse_done(writer: asyncio.StreamWriter) -> None:
    """Send the SSE termination signal."""
    writer.write(b"data: [DONE]\n\n")
    await writer.drain()


# ═══════════════════════════════════════════════════════════════════════
# Tokenizer helpers
# ═══════════════════════════════════════════════════════════════════════


def _fallback_tokenize(text: str) -> list[int]:
    """Heuristic fallback tokenizer (byte-level)."""
    return list(text.encode("utf-8"))


def _fallback_detokenize(token_ids: list[int]) -> str:
    """Heuristic fallback detokenizer."""
    try:
        return bytes(token_ids).decode("utf-8", errors="replace")
    except (ValueError, OverflowError):
        return str(token_ids)


# ═══════════════════════════════════════════════════════════════════════
# Request submission + polling helpers
# ═══════════════════════════════════════════════════════════════════════


class _StreamingTracker:
    """Tracks incremental token generation for SSE streaming."""

    def __init__(self) -> None:
        self._last_count = 0
        self._reasoning_done = False
        self._content_sent = ""

    @property
    def last_count(self) -> int:
        return self._last_count

    @last_count.setter
    def last_count(self, value: int) -> None:
        self._last_count = value

    @property
    def reasoning_done(self) -> bool:
        return self._reasoning_done

    @reasoning_done.setter
    def reasoning_done(self, value: bool) -> None:
        self._reasoning_done = value

    @property
    def content_sent(self) -> str:
        return self._content_sent

    @content_sent.setter
    def content_sent(self, value: str) -> None:
        self._content_sent = value


def _submit_and_poll(
    scheduler: UnifiedScheduler,
    prompt_tokens: list[int],
    request_id: str,
    max_new_tokens: int,
    timeout_s: float,
    tokenizer_decoder: object | None,
    stream: bool = False,
) -> tuple[int | None, list[int] | None, int | None]:
    """Synchronous helper: submit a request and poll until done or timeout.

    Used when we can't run async properly.  Returns
    ``(completed_request, output_tokens, output_text_or_None)``.
    """
    from scheduler import Request  # noqa: PLC0415

    req = Request(
        prompt_tokens=prompt_tokens,
        request_id=request_id,
        max_new_tokens=max_new_tokens,
    )
    scheduler.submit(req)

    deadline = time.monotonic() + timeout_s
    completed_request = None

    while time.monotonic() < deadline:
        time.sleep(0.05)
        for dr in scheduler.active_decode_pool:
            if dr.request_id == request_id and dr.is_done:
                completed_request = dr
                break
        if completed_request is not None:
            break

    return completed_request


async def _poll_until_done(
    scheduler: UnifiedScheduler,
    request_id: str,
    timeout_s: float,
) -> DecodeRequest | None:
    """Async-poll for a request's completion.

    Returns the ``DecodeRequest`` when done, or ``None`` on timeout.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        for dr in scheduler.active_decode_pool:
            if dr.request_id == request_id and dr.is_done:
                return dr
    return None


async def _decode_tokens(
    token_ids: list[int],
    tokenizer_decoder: object | None,
) -> str:
    """Decode a list of token IDs to text."""
    if tokenizer_decoder is not None:
        try:
            return tokenizer_decoder(token_ids)
        except Exception:
            pass
    return _fallback_detokenize(token_ids)


# ═══════════════════════════════════════════════════════════════════════
# OpenAI-compatible error response
# ═══════════════════════════════════════════════════════════════════════


def _openai_error(status: int, message: str, etype: str = "invalid_request_error") -> dict:
    return {
        "error": {
            "message": message,
            "type": etype,
            "param": None,
            "code": None,
        }
    }


# ═══════════════════════════════════════════════════════════════════════
# Endpoint handlers
# ═══════════════════════════════════════════════════════════════════════

# ── /v1/models ────────────────────────────────────────────────────────


def _handle_models(model_name: str | None = None) -> dict:
    """Return OpenAI-compatible /v1/models response.

    ``model_name`` should be set from the loaded model (e.g. ``"Qwen/Qwen2.5-1.5B-Instruct"``).
    Falls back to a placeholder when unknown.
    """
    name = model_name or "PyDense"
    data = [
        {
            "id": name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "PyDense",
            "permission": [],
        }
    ]
    return {"object": "list", "data": data}


# ── /v1/completions ──────────────────────────────────────────────────


async def _handle_completions(
    req_data: dict,
    scheduler: UnifiedScheduler,
    tokenizer_decoder: object | None,
    model_name: str | None,
) -> tuple[int, Any]:
    """Handle ``POST /v1/completions`` (text in → text out)."""
    # ── Extract prompt ──
    prompt = req_data.get("prompt")
    if prompt is None:
        return 400, _openai_error(400, "Missing required field: 'prompt'")
    if isinstance(prompt, list):
        prompt = "\n".join(str(p) for p in prompt)
    if not isinstance(prompt, str) or not prompt.strip():
        return 400, _openai_error(400, "'prompt' must be a non-empty string or list of strings")

    # ── Parameters ──
    max_tokens = int(req_data.get("max_tokens", 256))
    stream = bool(req_data.get("stream", False))
    stop_strs = req_data.get("stop")
    if isinstance(stop_strs, str):
        stop_strs = [stop_strs]
    echo = bool(req_data.get("echo", False))
    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"

    # ── Tokenize ──
    prompt_tokens = _tokenize(prompt, tokenizer_decoder)

    # ── Submit inference request ──
    from scheduler import Request  # noqa: PLC0415

    req = Request(
        prompt_tokens=prompt_tokens,
        request_id=request_id,
        max_new_tokens=max_tokens,
    )
    scheduler.submit(req)

    # ── Stream ──
    if stream:
        # Return immediately; the connection stays open for SSE.
        # The caller will send SSE headers and then stream tokens.
        return await _stream_completions(
            scheduler, request_id, max_tokens, tokenizer_decoder,
            prompt_text=prompt if echo else None,
            stop_strs=stop_strs,
        )

    # ── Non-streaming: poll until done ──
    timeout_s = max(30.0, max_tokens * 0.5)
    completed = await _poll_until_done(scheduler, request_id, timeout_s)
    if completed is None:
        return 503, _openai_error(503, "Request timed out", etype="server_error")

    output_text = await _decode_tokens(completed.generated_tokens, tokenizer_decoder)

    # ── Stop strings ──
    if stop_strs:
        for s in stop_strs:
            idx = output_text.find(s)
            if idx != -1:
                output_text = output_text[:idx]

    # ── Echo prompt ──
    full_text = (prompt if echo else "") + output_text

    response = _build_completions_response(
        request_id, full_text, len(prompt_tokens),
        len(completed.generated_tokens), reason="stop",
    )
    return 200, response


async def _stream_completions(
    scheduler: UnifiedScheduler,
    request_id: str,
    max_tokens: int,
    tokenizer_decoder: object | None,
    prompt_text: str | None = None,
    stop_strs: list[str] | None = None,
) -> tuple[int, dict]:
    """Handle streaming for ``/v1/completions``.

    Returns a special marker tuple; the caller must check for it.
    """
    # This is handled inline in _handle_client for SSE.
    # Return a marker so the caller knows it should use SSE.
    return ("__sse_completions__", {
        "request_id": request_id,
        "max_tokens": max_tokens,
        "tokenizer_decoder": tokenizer_decoder,
        "prompt_text": prompt_text,
        "stop_strs": stop_strs,
    })


def _build_completions_response(
    request_id: str,
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    reason: str = "stop",
) -> dict:
    return {
        "id": request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": "PyDense",
        "choices": [
            {
                "text": text,
                "index": 0,
                "finish_reason": reason,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ── /v1/chat/completions ─────────────────────────────────────────────


async def _handle_chat_completions(
    req_data: dict,
    scheduler: UnifiedScheduler,
    tokenizer_decoder: object | None,
    model_name: str | None,
) -> tuple[int, Any]:
    """Handle ``POST /v1/chat/completions`` (messages in → structured out)."""
    # ── Extract messages ──
    messages = req_data.get("messages")
    if not messages or not isinstance(messages, list):
        return 400, _openai_error(400, "Missing required field: 'messages' (must be an array)")

    # ── Parameters ──
    max_tokens = int(req_data.get("max_tokens", 256))
    stream = bool(req_data.get("stream", False))
    stop_strs = req_data.get("stop")
    if isinstance(stop_strs, str):
        stop_strs = [stop_strs]
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model_name_resp = model_name or req_data.get("model", "PyDense")

    # ── Convert messages to prompt ──
    prompt = _format_chat_prompt(messages)
    prompt_tokens = _tokenize(prompt, tokenizer_decoder)

    # ── Submit inference request ──
    from scheduler import Request  # noqa: PLC0415

    req = Request(
        prompt_tokens=prompt_tokens,
        request_id=request_id,
        max_new_tokens=max_tokens,
    )
    scheduler.submit(req)

    # ── Stream ──
    if stream:
        return ("__sse_chat__", {
            "request_id": request_id,
            "max_tokens": max_tokens,
            "model": model_name_resp,
            "tokenizer_decoder": tokenizer_decoder,
            "stop_strs": stop_strs,
        })

    # ── Non-streaming: poll until done ──
    timeout_s = max(30.0, max_tokens * 0.5)
    completed = await _poll_until_done(scheduler, request_id, timeout_s)
    if completed is None:
        return 503, _openai_error(503, "Request timed out", etype="server_error")

    output_text = await _decode_tokens(completed.generated_tokens, tokenizer_decoder)

    # ── Stop strings ──
    if stop_strs:
        for s in stop_strs:
            idx = output_text.find(s)
            if idx != -1:
                output_text = output_text[:idx]

    # ── Extract reasoning / deep thinking ──
    reasoning_content, final_content = extract_reasoning(output_text)

    # ── Build response ──
    choice: dict[str, Any] = {
        "index": 0,
        "message": {
            "role": "assistant",
            "content": final_content or output_text,
        },
        "finish_reason": "stop",
        "logprobs": None,
    }
    if reasoning_content:
        choice["message"]["reasoning_content"] = reasoning_content

    response = {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name_resp,
        "choices": [choice],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(completed.generated_tokens),
            "total_tokens": len(prompt_tokens) + len(completed.generated_tokens),
        },
    }
    return 200, response


# ═══════════════════════════════════════════════════════════════════════
# Streaming SSE loop
# ═══════════════════════════════════════════════════════════════════════


async def _sse_stream_completions(
    scheduler: UnifiedScheduler,
    writer: asyncio.StreamWriter,
    params: dict,
) -> None:
    """Stream tokens for ``/v1/completions`` over SSE."""
    request_id = params["request_id"]
    max_tokens = params["max_tokens"]
    tokenizer_decoder = params["tokenizer_decoder"]
    stop_strs = params.get("stop_strs")

    tracker = _StreamingTracker()
    timeout_s = max(30.0, max_tokens * 0.5)
    deadline = time.monotonic() + timeout_s
    finished = False

    while time.monotonic() < deadline and not finished:
        await asyncio.sleep(0.05)
        for dr in scheduler.active_decode_pool:
            if dr.request_id != request_id:
                continue

            # ── New tokens since last send? ──
            current_count = len(dr.generated_tokens)
            if current_count > tracker.last_count:
                new_tokens = dr.generated_tokens[tracker.last_count:]
                new_text = await _decode_tokens(new_tokens, tokenizer_decoder)
                tracker.last_count = current_count

                # Check stop strings
                if stop_strs:
                    for s in stop_strs:
                        if s in new_text:
                            new_text = new_text[: new_text.find(s)]
                            finished = True
                            break

                sse_data = {
                    "id": request_id,
                    "object": "text_completion",
                    "choices": [
                        {
                            "text": new_text,
                            "index": 0,
                            "finish_reason": "stop" if finished or dr.is_done else None,
                            "logprobs": None,
                        }
                    ],
                }
                await _send_sse(writer, sse_data)

            if dr.is_done or finished:
                finished = True
                break

    if not finished:
        # Send timeout finish
        sse_data = {
            "id": request_id,
            "object": "text_completion",
            "choices": [{"text": "", "index": 0, "finish_reason": "length", "logprobs": None}],
        }
        await _send_sse(writer, sse_data)

    await _send_sse_done(writer)


async def _sse_stream_chat(
    scheduler: UnifiedScheduler,
    writer: asyncio.StreamWriter,
    params: dict,
) -> None:
    """Stream tokens for ``/v1/chat/completions`` over SSE,
    with deep thinking / reasoning support.

    Reasoning tokens are sent first as ``delta.reasoning_content``.
    When the think block closes, ``delta.content`` takes over.
    """
    request_id = params["request_id"]
    max_tokens = params["max_tokens"]
    model = params.get("model", "PyDense")
    tokenizer_decoder = params["tokenizer_decoder"]
    stop_strs = params.get("stop_strs")

    tracker = _StreamingTracker()
    timeout_s = max(30.0, max_tokens * 0.5)
    deadline = time.monotonic() + timeout_s
    finished = False

    # Initial role message
    init_sse = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    await _send_sse(writer, init_sse)

    accumulated_text = ""

    while time.monotonic() < deadline and not finished:
        await asyncio.sleep(0.05)
        for dr in scheduler.active_decode_pool:
            if dr.request_id != request_id:
                continue

            current_count = len(dr.generated_tokens)
            if current_count <= tracker.last_count:
                if dr.is_done:
                    finished = True
                break

            new_tokens = dr.generated_tokens[tracker.last_count:]
            tracker.last_count = current_count

            # Decode incremental text
            new_text = await _decode_tokens(new_tokens, tokenizer_decoder)
            accumulated_text += new_text

            # Stop strings
            if stop_strs:
                for s in stop_strs:
                    if s in new_text:
                        new_text = new_text[: new_text.find(s)]
                        finished = True
                        break

            # ── Deep thinking / reasoning detection ──
            # We track the accumulated text to know when a think block opens/closes.
            # This handles streaming into <think>...</think> incrementally.

            # Scan accumulated text for think boundaries
            # Strategy: look at the full accumulated text to find current state

            delta_content = None
            delta_reasoning = None

            if not tracker.reasoning_done:
                # Still potentially inside a think block or before it
                # Check if we've reached a closing tag
                closing_think = accumulated_text.find("</think>")
                closing_thinking = accumulated_text.find("</thinking>")

                if closing_think != -1:
                    # Everything before </think> is reasoning (minus <think> tag itself)
                    # The opening <think> might already have been passed
                    open_think = accumulated_text.find("<think>")
                    if open_think == -1:
                        open_think = 0
                    else:
                        open_think += len("<think>")

                    reasoning_full = accumulated_text[open_think:closing_think]
                    delta_reasoning = reasoning_full
                    tracker.reasoning_done = True

                    # The remaining text after </think> is content
                    after_think = accumulated_text[closing_think + len("</think>"):]
                    if after_think.strip():
                        delta_content = after_think

                elif closing_thinking != -1:
                    open_thinking = accumulated_text.find("<thinking>")
                    if open_thinking == -1:
                        open_thinking = 0
                    else:
                        open_thinking += len("<thinking>")

                    reasoning_full = accumulated_text[open_thinking:closing_thinking]
                    delta_reasoning = reasoning_full
                    tracker.reasoning_done = True

                    after_thinking = accumulated_text[closing_thinking + len("</thinking>"):]
                    if after_thinking.strip():
                        delta_content = after_thinking

                elif "<think>" in accumulated_text or "<thinking>" in accumulated_text:
                    # We're inside an open think block but haven't seen the close yet
                    # Extract reasoning text after the tag
                    if "<think>" in accumulated_text:
                        open_pos = accumulated_text.rfind("<think>") + len("<think>")
                        reasoning_part = accumulated_text[open_pos:]
                    elif "<thinking>" in accumulated_text:
                        open_pos = accumulated_text.rfind("<thinking>") + len("<thinking>")
                        reasoning_part = accumulated_text[open_pos:]
                    else:
                        reasoning_part = new_text

                    if reasoning_part.strip():
                        delta_reasoning = reasoning_part
                else:
                    # Not in a think block — this is pure content
                    tracker.reasoning_done = True
                    delta_content = new_text
            else:
                # Reasoning done — everything is content
                delta_content = new_text

            # ── Send SSE delta ──
            if delta_reasoning is not None and delta_reasoning.strip():
                sse_data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"reasoning_content": delta_reasoning, "content": None},
                            "finish_reason": None,
                        }
                    ],
                }
                await _send_sse(writer, sse_data)

            if delta_content is not None and delta_content.strip():
                sse_data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta_content},
                            "finish_reason": None,
                        }
                    ],
                }
                await _send_sse(writer, sse_data)

            if dr.is_done or finished:
                finished = True
                break

    # ── Final chunk ──
    finish_reason = "stop" if finished else "length"
    final_sse = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    await _send_sse(writer, final_sse)
    await _send_sse_done(writer)


# ═══════════════════════════════════════════════════════════════════════
# Tokenization
# ═══════════════════════════════════════════════════════════════════════


def _tokenize(text: str, tokenizer_decoder: object | None) -> list[int]:
    """Tokenize text using the real tokenizer, or fallback."""
    if tokenizer_decoder is not None:
        try:
            # Some tokenizers expose encode
            if hasattr(tokenizer_decoder, "__self__") and hasattr(tokenizer_decoder.__self__, "encode"):
                return tokenizer_decoder.__self__.encode(text)
        except Exception:
            pass
    return _fallback_tokenize(text)


# ═══════════════════════════════════════════════════════════════════════
# Connection handler / router
# ═══════════════════════════════════════════════════════════════════════


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    scheduler: UnifiedScheduler,
    tokenizer_decoder: object | None,
    model_name: str | None,
) -> None:
    """Handle one HTTP connection."""
    try:
        method, path, headers = await _recv_headers(reader)
    except (ConnectionError, ValueError, asyncio.IncompleteReadError) as exc:
        logger.debug("HTTP read error: %s", exc)
        with contextlib.suppress(Exception):
            writer.close()
        return

    # ── CORS preflight ──
    if method == "OPTIONS":
        writer.write(_build_response(200, "OK"))
        await writer.drain()
        writer.close()
        return

    # ── Health check ──
    if method == "GET" and path in {"/health", "/"}:
        writer.write(_build_response(200, {"status": "ok", "engine": "PyDense"}))
        await writer.drain()
        writer.close()
        return

    # ── /v1/models ──
    if method == "GET" and path == "/v1/models":
        writer.write(_build_response(200, _handle_models(model_name)))
        await writer.drain()
        writer.close()
        return

    # ── Read body for POST ──
    if method != "POST":
        writer.write(_build_response(404, {"error": "Not found"}))
        await writer.drain()
        writer.close()
        return

    cl = _parse_content_length(headers)
    body_bytes = await _read_body(reader, cl)
    try:
        req_data = json.loads(body_bytes)
    except json.JSONDecodeError as exc:
        writer.write(_build_response(400, _openai_error(400, f"Invalid JSON: {exc}")))
        await writer.drain()
        writer.close()
        return

    # ── Route ──
    if path == "/v1/completions":
        result = await _handle_completions(req_data, scheduler, tokenizer_decoder, model_name)
    elif path == "/v1/chat/completions":
        result = await _handle_chat_completions(req_data, scheduler, tokenizer_decoder, model_name)
    else:
        writer.write(_build_response(404, {"error": "Not found"}))
        await writer.drain()
        writer.close()
        return

    # ── Handle streaming SSE responses ──
    sse_completions = "__sse_completions__"
    sse_chat = "__sse_chat__"
    if isinstance(result, tuple) and len(result) == _TUPLE_RESPONSE_LEN and result[0] in (sse_completions, sse_chat):
        sse_type = result[0]
        params = result[1]

        # Send SSE headers first
        sse_headers = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Expose-Headers: X-Accel-Buffering\r\n"
            "X-Accel-Buffering: no\r\n"
            "\r\n"
        )
        writer.write(sse_headers.encode("utf-8"))
        await writer.drain()

        if sse_type == sse_completions:
            await _sse_stream_completions(scheduler, writer, params)
        else:
            await _sse_stream_chat(scheduler, writer, params)

        writer.close()
        return

    # ── Normal (non-streaming) response ──
    if isinstance(result, tuple) and len(result) == _TUPLE_RESPONSE_LEN:
        status_code, resp_body = result
    else:
        status_code, resp_body = 200, result

    writer.write(_build_response(status_code, resp_body))
    await writer.drain()
    writer.close()


# ═══════════════════════════════════════════════════════════════════════
# Server entry point
# ═══════════════════════════════════════════════════════════════════════


async def run_api_server(
    scheduler: UnifiedScheduler,
    host: str = "0.0.0.0",
    port: int = 8000,
    tokenizer_decoder: object | None = None,
    model_name: str | None = None,
) -> None:
    """Start the async HTTP API server on the given host/port.

    Parameters
    ----------
    scheduler:
        The engine scheduler to submit inference requests to.
    host, port:
        Bind address.
    tokenizer_decoder:
        The model tokenizer's ``.decode`` method, or a callable
        ``list[int] → str``.  ``None`` falls back to byte-level decoding.
    model_name:
        The model identifier (e.g. ``"Qwen/Qwen2.5-1.5B-Instruct"``).
        Used in ``/v1/models`` and ``/v1/chat/completions`` responses.
    """

    async def _on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _handle_client(reader, writer, scheduler, tokenizer_decoder, model_name)

    server = await asyncio.start_server(_on_connect, host, port)
    addr = server.sockets[0].getsockname()
    logger.info("API server listening on http://%s:%d", addr[0], addr[1])
    print(f"\n━━━ PyDense API running at http://{addr[0]}:{addr[1]} ━━━")
    print("     POST /v1/completions        — Text completion (streaming supported)")
    print("     POST /v1/chat/completions    — Chat completion (streaming, reasoning supported)")
    print("     GET  /v1/models              — Model info")
    print("     GET  /health                 — Health check\n")

    async with server:
        await server.serve_forever()
