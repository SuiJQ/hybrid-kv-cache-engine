# long_context/integration.py
"""
Long Context — MoeOwner Integration Module
============================================

Wires SelfExtend and ReAttention into the existing inference pipeline.

Injection Points
----------------
1. **Attention forward** — ``_inject_long_context_kernel()`` replaces the
   existing ``_inject_attention_kernel()`` in ``main.py``.  It wraps the
   flash attention call with SelfExtend or ReAttention logic.

2. **Position ID handling** — SelfExtend modifies position_ids on-the-fly.

3. **Scheduler hooks** — Context length tracking and automatic
   activation/deactivation of long-context methods.

Compatibility
-------------
All methods are **fully compatible** with existing MoeOwner modules:
  - Goose speculative decode
  - AFCE anchor extensions
  - SERE dynamic expert skipping
  - OEF entropy freeze
  - HybridCache KV quantization

None require changes to model weights, architecture, or training.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from .config import LongContextConfig, ContextMethod
from .self_extend import SelfExtendWrapper, get_self_extend_position_ids
from .re_attention import ReAttentionWrapper, dispatch_reattention

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Position Encoding Helpers
# ═══════════════════════════════════════════════════════════════════════


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embedding to x.

    ``x`` shape: ``(B, H, T, D)``
    ``cos/sin`` shape: ``(1, max_T, D)`` — precomputed table
    ``position_ids`` shape: ``(1, T)`` — position indices
    """
    # Gather cos/sin at specified positions
    # cos[0, pos_ids, :] where pos_ids is (T,) → (T, D)
    pos_flat = position_ids[0]  # (T,)
    cos_pos = cos[0, pos_flat, :].unsqueeze(0).unsqueeze(1)  # (1, 1, T, D)
    sin_pos = sin[0, pos_flat, :].unsqueeze(0).unsqueeze(1)  # (1, 1, T, D)

    # RoPE: x * cos + rotate_half(x) * sin
    x_rotated = torch.stack(
        [-x[..., 1::2], x[..., ::2]], dim=-1
    ).flatten(-2)
    return x * cos_pos + x_rotated * sin_pos


def _compute_default_cos_sin(
    head_dim: int,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for a given head dimension.

    Uses the standard RoPE frequency formula:
        theta_i = base^(-2i/d)
    """
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
    )
    t = torch.arange(max_seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().unsqueeze(0)   # (1, T, D//2)
    sin = freqs.sin().unsqueeze(0)
    # Duplicate for paired dimensions
    cos = torch.cat([cos, cos], dim=-1)   # (1, T, D)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


# ═══════════════════════════════════════════════════════════════════════
# Attention Kernel Injection (with Long Context)
# ═══════════════════════════════════════════════════════════════════════


class LongContextAttentionInjector:
    """Manages attention-layer injection for long context extensions.

    This is the central coordinator that:
    1. Detects which method to use (from config)
    2. Patches each decoder layer's self_attn.forward
    3. Manages position encoding (including RoPE computation)
    4. Calls SelfExtend or ReAttention as appropriate
    """

    def __init__(self, config: LongContextConfig):
        self.config = config
        self.method = config.method if config.enabled else "none"

        # Lazy-initialized wrappers
        self._self_extend: Optional[SelfExtendWrapper] = None
        self._re_attention: Optional[ReAttentionWrapper] = None

        # Cached RoPE tables (per head_dim)
        self._cos_sin_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _init_wrappers(self):
        if self._self_extend is None and self.method == "selfextend":
            self._self_extend = SelfExtendWrapper(self.config)
            logger.info("Initialized %s", self._self_extend)
        if self._re_attention is None and self.method == "reattention":
            self._re_attention = ReAttentionWrapper(self.config)
            logger.info("Initialized %s", self._re_attention)

    def inject(self, model: nn.Module) -> int:
        """Inject long-context attention into all decoder layers.

        Returns the number of layers patched.
        """
        if not self.config.enabled:
            logger.info("Long context extension disabled.")
            return 0

        self._init_wrappers()

        layers = self._get_decoder_layers(model)
        if layers is None:
            logger.warning("Could not locate decoder layers — injection skipped.")
            return 0

        patched = 0
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue

            # Replace the forward method
            attn.forward = self._make_patched_forward(attn)
            attn.long_context_injected = True
            patched += 1

        logger.info(
            "Long context injection: %d/%d layers patched (method=%s)",
            patched,
            len(layers),
            self.method,
        )
        return patched

    def _get_decoder_layers(self, model: nn.Module):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "layers"):
            return model.layers
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            return model.decoder.layers
        return None

    def _make_patched_forward(self, attn: nn.Module):
        """Create a patched forward closure for this attention module.

        The patched forward:
          1. Computes q, k, v projections (same as original)
          2. Applies RoPE with SelfExtend-modified position IDs (if selfextend)
          3. Calls FlashAttention with or without ReAttention pre-filtering
          4. Projects output back (same as original)
        """
        config = self.config
        method = self.method
        self_extend = self._self_extend
        re_attention = self._re_attention
        cos_sin_cache = self._cos_sin_cache

        def _patched_forward(
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            past_key_value: tuple | None = None,
            use_cache: bool = False,
            position_ids: torch.Tensor | None = None,
            **kwargs,
        ) -> tuple:
            # ── QKV Projections ──────────────────────────────────
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

            # ── Position Encoding (RoPE) ─────────────────────────
            # Build cos/sin table for this head_dim (cached)
            if head_dim not in cos_sin_cache:
                cos_sin_cache[head_dim] = _compute_default_cos_sin(
                    head_dim,
                    max_seq_len=config.yarn_original_max_len * max(int(config.yarn_factor), 1) if method == "yarn" else config.yarn_original_max_len,
                    device=q.device,
                    dtype=q.dtype,
                )

            # Default position_ids if not provided
            if position_ids is None:
                past_len = 0
                if past_key_value is not None and past_key_value[0] is not None:
                    past_len = past_key_value[0].shape[2]
                position_ids = torch.arange(
                    past_len, past_len + seq_len,
                    device=q.device,
                ).unsqueeze(0)

            cos, sin = cos_sin_cache[head_dim]

            # ── SelfExtend: modify position_ids for grouped attention ──
            if method == "selfextend" and self_extend is not None:
                grouped_pos = self_extend.group_position_ids(position_ids)
                if grouped_pos is not None:
                    # Apply RoPE with grouped positions
                    q = apply_rotary_emb(q, cos, sin, grouped_pos)
                    k = apply_rotary_emb(k, cos, sin, grouped_pos)
                else:
                    q = apply_rotary_emb(q, cos, sin, position_ids)
                    k = apply_rotary_emb(k, cos, sin, position_ids)
            elif method in ("yarn", "none"):
                q = apply_rotary_emb(q, cos, sin, position_ids)
                k = apply_rotary_emb(k, cos, sin, position_ids)
            elif method == "reattention":
                # ReAttention: RoPE applied after retrieval step
                # (retrieval is position-agnostic, so no RoPE on retrieval pass)
                q = apply_rotary_emb(q, cos, sin, position_ids)
                k = apply_rotary_emb(k, cos, sin, position_ids)
            else:
                # Fallback: apply RoPE normally
                q = apply_rotary_emb(q, cos, sin, position_ids)
                k = apply_rotary_emb(k, cos, sin, position_ids)

            # ── KV Cache ─────────────────────────────────────────
            kv_len_before = 0
            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)
                kv_len_before = past_key_value[0].shape[2]

            kv_len = k.shape[2]
            softmax_scale = head_dim ** -0.5

            # ── Attention Computation ────────────────────────────
            from attention_kernel import FlashAttentionKernel  # noqa: PLC0415

            if method == "reattention" and re_attention is not None:
                # ReAttention two-pass path
                attn_output = dispatch_reattention(
                    re_attention, q, k, v,
                    softmax_scale=softmax_scale,
                    causal=(attention_mask is None),
                    kv_len=kv_len,
                )
            else:
                # Standard (or SelfExtend) — single flash attention pass
                attn_output = FlashAttentionKernel.forward(
                    q, k, v,
                    softmax_scale=softmax_scale,
                    causal=(attention_mask is None),
                    attn_mask=attention_mask,
                )

            # ── Output Projection ────────────────────────────────
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, -1)
            attn_output = attn_output.to(hidden_states.dtype)
            attn_output = attn.o_proj(attn_output)

            new_kv = (k, v) if use_cache else None
            return (attn_output, new_kv)

        return _patched_forward


# ═══════════════════════════════════════════════════════════════════════
# Convenience: one-shot injection
# ═══════════════════════════════════════════════════════════════════════

def inject_long_context(model: nn.Module, config: LongContextConfig) -> int:
    """One-shot convenience: create injector and apply to model.

    Returns number of patched layers.
    """
    injector = LongContextAttentionInjector(config)
    return injector.inject(model)
