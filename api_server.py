"""
api_server.py — Zero-dependency async HTTP API server for MoeOwner.

Exposes a JSON-based inference API at ``/v1/completions``
(OpenAI-compatible request/response schema).

Built on ``asyncio`` + stdlib — no third-party dependencies.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scheduler import UnifiedScheduler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tiny async HTTP request/response parser/writer (stdlib only)
# ---------------------------------------------------------------------------

_STATUS_LINES = {
    200: b"200 OK",
    400: b"400 Bad Request",
    404: b"404 Not Found",
    500: b"500 Internal Server Error",
    503: b"503 Service Unavailable",
}

_CONTENT_JSON = b"application/json"
_CONTENT_PLAIN = b"text/plain"


async def _recv_headers(reader: asyncio.StreamReader) -> tuple[str, str, dict[str, str]]:
    """Read HTTP/1.1 request line + headers. Returns (method, path, headers)."""
    line = await reader.readline()
    if not line:
        raise ConnectionError("Client disconnected")
    parts = line.decode("utf-8", errors="replace").strip().split(maxsplit=2)
    _min_parts = 2
    if len(parts) < _min_parts:
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
    content_type: bytes = _CONTENT_JSON,
    cors: bool = True,
) -> bytes:
    if isinstance(body, str):
        payload = body.encode("utf-8")
        content_type = _CONTENT_PLAIN
    else:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")

    header_lines = [
        f"HTTP/1.1 {_STATUS_LINES.get(status, b'200 OK').decode()}",
        f"Content-Type: {content_type.decode()}",
        f"Content-Length: {len(payload)}",
        "Connection: keep-alive",
    ]
    if cors:
        header_lines.extend(
            [
                "Access-Control-Allow-Origin: *",
                "Access-Control-Allow-Methods: POST, GET, OPTIONS",
                "Access-Control-Allow-Headers: Content-Type",
            ]
        )
    raw = "\r\n".join(header_lines).encode() + b"\r\n\r\n" + payload
    return raw


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    scheduler: UnifiedScheduler,
    tokenizer_decoder: object | None,
) -> None:
    """Handle one HTTP connection."""
    try:
        method, path, headers = await _recv_headers(reader)
    except (ConnectionError, ValueError, asyncio.IncompleteReadError) as exc:
        logger.debug("HTTP read error: %s", exc)
        with contextlib.suppress(Exception):
            writer.close()
        return

    # CORS preflight
    if method == "OPTIONS":
        writer.write(_build_response(200, "OK"))
        await writer.drain()
        writer.close()
        return

    # Health check
    if method == "GET" and (path in {"/health", "/"}):
        writer.write(_build_response(200, {"status": "ok", "engine": "MoeOwner"}))
        await writer.drain()
        writer.close()
        return

    if method == "POST" and path == "/v1/completions":
        cl = _parse_content_length(headers)
        body_bytes = await _read_body(reader, cl)
        try:
            req_data = json.loads(body_bytes)
        except json.JSONDecodeError as exc:
            writer.write(_build_response(400, {"error": f"Invalid JSON: {exc}"}))
            await writer.drain()
            writer.close()
            return

        result = await _handle_completion(req_data, scheduler, tokenizer_decoder)
        _result_tuple_len = 2
        if isinstance(result, tuple) and len(result) == _result_tuple_len:
            status_code, resp_body = result
        else:
            status_code, resp_body = 200, result

        writer.write(_build_response(status_code, resp_body))
        await writer.drain()
        writer.close()
        return

    # 404
    writer.write(_build_response(404, {"error": "Not found"}))
    await writer.drain()
    writer.close()


def _extract_prompt(req_data: dict) -> str | None:
    """Extract prompt string from OpenAI-compatible request."""
    prompt = req_data.get("prompt")
    if prompt is None:
        prompt = req_data.get("messages")
        if isinstance(prompt, list):
            # Simple message extraction: take last user message
            for msg in reversed(prompt):
                if isinstance(msg, dict) and msg.get("role") in ("user", "system"):
                    content = msg.get("content", "")
                    if content:
                        return content
            return None
    if isinstance(prompt, str):
        return prompt
    return None


async def _handle_completion(
    req_data: dict,
    scheduler: UnifiedScheduler,
    tokenizer_decoder: object | None,
) -> tuple[int, dict]:
    """Handle a single /v1/completions request."""
    prompt_text = _extract_prompt(req_data)
    if not prompt_text:
        return 400, {"error": "Missing 'prompt' (string) or 'messages' array"}

    max_tokens = int(req_data.get("max_tokens", 256))
    float(req_data.get("temperature", 0.0))
    request_id = req_data.get("id", str(uuid.uuid4()))

    # Tokenize (simple heuristic) — for production use a real tokenizer
    # This is a placeholder; real tokenization requires the model's tokenizer.
    if tokenizer_decoder is not None:
        try:
            token_ids = tokenizer_decoder.encode(prompt_text)
        except Exception:
            token_ids = _fallback_tokenize(prompt_text)
    else:
        token_ids = _fallback_tokenize(prompt_text)

    from scheduler import Request  # noqa: PLC0415

    req = Request(
        prompt_tokens=token_ids,
        request_id=request_id,
        max_new_tokens=max_tokens,
    )

    scheduler.submit(req)

    # Wait for completion
    timeout = max(30.0, max_tokens * 0.5)
    deadline = time.monotonic() + timeout
    completed_request = None

    while time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        for dr in scheduler.active_decode_pool:
            if dr.request_id == request_id and dr.is_done:
                completed_request = dr
                break
        if completed_request is not None:
            break
        # Also check if the request has moved from pending to decode
        if req not in scheduler.pending_requests:
            pass  # It's either in decode or done

    if completed_request is None:
        return 503, {"error": "Request timed out"}

    # Decode tokens back to text
    if tokenizer_decoder is not None:
        try:
            output_text = tokenizer_decoder.decode(completed_request.generated_tokens)
        except Exception:
            output_text = _fallback_detokenize(completed_request.generated_tokens)
    else:
        output_text = _fallback_detokenize(completed_request.generated_tokens)

    response: dict = {
        "id": request_id,
        "object": "text_completion",
        "choices": [
            {
                "text": output_text,
                "index": 0,
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": len(token_ids),
            "completion_tokens": len(completed_request.generated_tokens),
            "total_tokens": len(token_ids) + len(completed_request.generated_tokens),
        },
    }
    return 200, response


def _fallback_tokenize(text: str) -> list[int]:
    """Heuristic fallback tokenizer (byte-level)."""
    encoded = text.encode("utf-8")
    tokens = [b for b in encoded]
    return tokens  # returns byte values as token IDs


def _fallback_detokenize(token_ids: list[int]) -> str:
    """Heuristic fallback detokenizer."""
    try:
        return bytes(token_ids).decode("utf-8", errors="replace")
    except (ValueError, OverflowError):
        return str(token_ids)


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------


async def run_api_server(
    scheduler: UnifiedScheduler,
    host: str = "0.0.0.0",
    port: int = 8000,
    tokenizer_decoder: object | None = None,
) -> None:
    """Start the async HTTP API server on the given host/port."""

    async def _on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _handle_client(reader, writer, scheduler, tokenizer_decoder)

    server = await asyncio.start_server(_on_connect, host, port)
    addr = server.sockets[0].getsockname()
    logger.info("API server listening on http://%s:%d", addr[0], addr[1])
    print(f"\n━━━ MoeOwner API running at http://{addr[0]}:{addr[1]} ━━━")
    print("     POST /v1/completions — OpenAI-compatible inference")
    print("     GET  /health         — health check\n")

    async with server:
        await server.serve_forever()
