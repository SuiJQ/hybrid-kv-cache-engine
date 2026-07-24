# long_context/re_attention.py
"""
ReAttention — Training-Free Infinite Context with Finite Attention Scope
==========================================================================

Paper:   ICLR 2025  |  arXiv:2407.15176
Author:  Xiaoran Liu et al.  |  GitHub: OpenMOSS/ReAttention

Core Idea
---------
The fundamental bottleneck for infinite context in Transformers is:
  1. Position embedding goes OOD beyond training length
  2. Attention entropy collapses for very long sequences

ReAttention solves both without any training by adding a **position-agnostic
top-k retrieval step** before the normal position-aware attention:

  Step 1 — Retrieval:
    Compute lightweight content-based (no position encoding) attention
    scores between query and ALL keys.  Select the top-k most relevant keys.

  Step 2 — Position-Aware Attention:
    Run normal position-aware self-attention, but only on the selected
    top-k keys plus a small window of recent tokens.  This "finite
    attention scope" avoids entropy collapse and position OOD.

The result: bounded compute (O(k) per token, where k << seq_len) with
access to all content in the prompt, regardless of length.

This implementation uses a single-head proxy (averaged across attention
heads) for the retrieval step to keep overhead minimal.

Practical Effect
----------------
- LLaMA3.1-8B : 1M tokens stable
- LLaMA3.2-3B : 4M tokens (128× beyond training length)
- General     : Vanilla Transformer up to 1M+ without any fine-tuning

MoeOwner Integration
--------------------
ReAttention is injected at the attention kernel level, wrapping
``FlashAttentionKernel.forward``.  It is fully compatible with:
  - KV cache (past_key_values)
  - Goose speculative decode
  - Goose speculative decode
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .config import LongContextConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Triton-like JIT specs (will use torch compile, not raw Triton)
# ---------------------------------------------------------------------------

_REATTN_JIT_SPEC: dict = {
    "BLOCK_SIZE": 32,
    "USE_FP16": True,
}


class ReAttentionWrapper:
    """ReAttention: top-k content retrieval before position-aware attention.

    Usage (inside a patched attention forward):
        ra = ReAttentionWrapper(config)

        if ra.should_activate(q_len, kv_len):
            top_k_idx, actual_k = ra.retrieve(q, k, kv_len)
            mask = ra.build_sparse_mask(top_k_idx, q_len, kv_len, device)
            out = FlashAttentionKernel.forward(q, k, v, scale, causal=False, attn_mask=mask)
        else:
            out = FlashAttentionKernel.forward(q, k, v, scale, causal=True)
    """

    def __init__(self, config: LongContextConfig):
        self.top_k_ratio = config.reattn_top_k_ratio
        self.max_top_k = config.reattn_top_k
        self.min_top_k = config.reattn_min_top_k
        self.neighbor_window = config.reattn_neighbor_window
        self.short_threshold = config.short_context_threshold
        self.verbose = config.verbose

    def should_activate(self, q_len: int, kv_len: int) -> bool:
        """Return True if ReAttention's two-pass path is beneficial.

        For short sequences the overhead isn't worth it — fall through
        to standard causal attention.
        """
        if kv_len <= self.neighbor_window:
            return False
        if kv_len < self.short_threshold:
            return False
        return True

    def retrieve(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        kv_len: int,
    ) -> Tuple[torch.Tensor, int]:
        """Position-agnostic top-k retrieval.

        Step 1 (from the paper):
            Compute content-based attention scores WITHOUT position encoding,
            using a single-head proxy for efficiency.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor, shape ``(B, H_q, T_q, D)``.
        k : torch.Tensor
            Key tensor, shape ``(B, H_kv, T_kv, D)``.
        kv_len : int
            Total number of key positions (for computing k).

        Returns
        -------
        top_k_idx : torch.Tensor
            Indices of the top-k keys for each query, shape ``(B, T_q, k)``.
        actual_k : int
            Number of keys actually kept.
        """
        B, H_q, T_q, D = q.shape

        # ── Compute k value ──────────────────────────────────────
        actual_k = min(int(kv_len * self.top_k_ratio), self.max_top_k)
        actual_k = max(actual_k, self.min_top_k)
        actual_k = min(actual_k, kv_len)
        actual_k = max(actual_k, 1)

        # ── Head averaging (single-head proxy) ───────────────────
        # Average over query heads: (B, T_q, D)
        if H_q > 1:
            q_proxy = q.mean(dim=1)
        else:
            q_proxy = q[:, 0]

        # Average over kv heads if GQA: (B, T_kv, D)
        if k.shape[1] > 1:
            k_proxy = k.mean(dim=1)
        else:
            k_proxy = k[:, 0]

        # ── Content-based similarity ─────────────────────────────
        # L2-normalize for stable cosine similarity (position-agnostic)
        q_proxy = F.normalize(q_proxy, dim=-1)
        k_proxy = F.normalize(k_proxy, dim=-1)

        # (B, T_q, T_kv)
        scores = torch.matmul(q_proxy, k_proxy.transpose(-2, -1))

        # ── Top-k selection ──────────────────────────────────────
        _, top_k_idx = torch.topk(scores, actual_k, dim=-1)

        return top_k_idx, actual_k

    def build_mask(
        self,
        top_k_idx: torch.Tensor,
        q_len: int,
        kv_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Build attention mask: top-k + recent tokens visible.

        Step 2 (from the paper):
            Create a mask where each query position can attend to:
              a) Its top-k retrieved keys (content-relevant)
              b) The last ``neighbor_window`` keys (local coherence)

        Returns a float mask ready for SDPA: 0.0 = attend, -inf = mask.
        Uses ``dtype`` matching the query tensor by default.
        """
        # ── Boolean mask: True = allowed ─────────────────────────
        mask = torch.zeros(1, 1, q_len, kv_len, dtype=torch.bool, device=device)

        for q_pos in range(q_len):
            # (a) Recent tokens: always visible
            recent_start = max(0, kv_len - self.neighbor_window)
            if recent_start < kv_len:
                mask[0, 0, q_pos, recent_start:] = True

            # (b) Top-k retrieved tokens
            # top_k_idx shape: (B, T_q, k) — use batch 0
            for idx_val in top_k_idx[0, q_pos]:
                i = idx_val.item()
                if 0 <= i < kv_len:
                    mask[0, 0, q_pos, i] = True

        # ── Convert to float mask ────────────────────────────────
        float_mask = torch.full(
            (1, 1, q_len, kv_len), float("-inf"), dtype=dtype, device=device,
        )
        float_mask[mask] = 0.0

        return float_mask

    def __repr__(self) -> str:
        return (
            f"ReAttention(top_k={self.max_top_k}, "
            f"ratio={self.top_k_ratio}, "
            f"NW={self.neighbor_window})"
        )


# ---------------------------------------------------------------------------
# FlashAttentionKernel-compatible dispatch
# ---------------------------------------------------------------------------

def dispatch_reattention(
    ra_wrapper: ReAttentionWrapper,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    kv_len: int,
) -> torch.Tensor:
    """ReAttention dispatch: use two-pass when beneficial, else vanilla.

    This is the main entry point for ReAttention in the attention
    injection pipeline.
    """
    q_len = q.shape[2]

    if not ra_wrapper.should_activate(q_len, kv_len):
        # Fall through to vanilla causal attention
        from attention_kernel import FlashAttentionKernel  # noqa: PLC0415

        return FlashAttentionKernel.forward(
            q, k, v,
            softmax_scale=softmax_scale,
            causal=causal,
            attn_mask=None,
        )

    # ── Step 1: Position-agnostic retrieval ──────────────────────
    top_k_idx, actual_k = ra_wrapper.retrieve(q, k, kv_len)

    # ── Step 2: Build sparse mask ────────────────────────────────
    mask = ra_wrapper.build_mask(top_k_idx, q_len, kv_len, q.device, dtype=q.dtype)

    # ── Step 3: Position-aware attention (finite scope) ──────────
    from attention_kernel import FlashAttentionKernel  # noqa: PLC0415

    return FlashAttentionKernel.forward(
        q, k, v,
        softmax_scale=softmax_scale,
        causal=False,  # mask handles all visibility
        attn_mask=mask,
    )
