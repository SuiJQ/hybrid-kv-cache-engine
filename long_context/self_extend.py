# long_context/self_extend.py
"""
SelfExtend — Bi-Level Attention for Training-Free Context Extension
=====================================================================

Paper: "LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning"
       ICML 2024 Spotlight  |  arXiv:2401.01325

Core Idea
---------
LLMs have an inherent ability to handle long contexts — the bottleneck is
position encoding OOD (Out-Of-Distribution).  SelfExtend solves this by
constructing **bi-level attention**:

  • **Neighbor Attention** — tokens within ``neighbor_window`` use original
    position IDs.  Captures local, fine-grained dependencies.
  • **Grouped Attention** — tokens beyond ``neighbor_window`` get FLOOR-divided
    position IDs: ``floor((pos - NW) / group_size) + NW``.  This maps faraway
    positions into the trained range.  Captures long-range semantic structure.

The two are combined: local tokens use the stronger neighbor signal;
distant tokens fall back to grouped attention.

Implementation (4-line core)
----------------------------
Only the position_ids fed into RoPE need changing.  The rest of the
attention pipeline is unchanged.

    # Original:
    cos, sin = rotary_emb(q, seq_len)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

    # SelfExtend:
    grouped_ids = position_ids.clone()
    mask = grouped_ids >= neighbor_window
    grouped_ids[mask] = ((grouped_ids[mask] - NW) // GS + NW)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from .config import LongContextConfig

logger = logging.getLogger(__name__)


def get_self_extend_position_ids(
    position_ids: torch.Tensor,
    neighbor_window: int,
    group_size: int,
    short_threshold: int = 2048,
) -> Optional[torch.Tensor]:
    """Create grouped position IDs for SelfExtend's far-range attention.

    Parameters
    ----------
    position_ids : torch.Tensor
        Original position IDs, shape ``(1, seq_len)``.
    neighbor_window : int
        Number of initial + recent tokens that keep original positions.
    group_size : int
        Divide distant positions by this factor.
    short_threshold : int
        Sequences shorter than this skip SelfExtend.

    Returns
    -------
    torch.Tensor or None
        Grouped position IDs, or ``None`` if the sequence is too short
        to benefit from SelfExtend.  When ``None``, caller should use
        ``position_ids`` as-is.

    Notes
    -----
    This is the **core 4-line SelfExtend logic**.  Everything else is
    bookkeeping and integration.

    Example (neighbor_window=4, group_size=2):
        Original:    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        Grouped:     [0, 1, 2, 3, 4, 4, 5, 5, 6, 6]
    """
    seq_len = position_ids.shape[-1]

    if seq_len <= neighbor_window:
        return None
    if seq_len < short_threshold:
        return None

    grouped = position_ids.clone()
    mask = grouped >= neighbor_window
    # FLOOR division: (pos - NW) // GS + NW
    grouped[mask] = ((grouped[mask] - neighbor_window) // group_size + neighbor_window)

    return grouped


class SelfExtendWrapper:
    """Wraps a flash-attention kernel with SelfExtend bi-level attention.

    This wrapper is designed to be composed with the existing
    ``FlashAttentionKernel.forward`` — it modifies the position_ids
    before they reach the RoPE computation, then blends the outputs.

    Usage (inside an attention forward):
        se_wrapper = SelfExtendWrapper(neighbor_window=1024, group_size=8)

        # For grouped attention (far tokens):
        grouped_pos = se_wrapper.group_position_ids(position_ids)
        if grouped_pos is not None:
            # Compute RoPE with grouped positions
            # Then flash attention with grouped q, k, v
            grouped_out = flash_attn(grouped_q, grouped_k, v, ...)

        # For neighbor attention (recent tokens):
        remainder = max(0, seq_len - neighbor_window)
        neighbor_q = q[:, :, remainder:, :]
        neighbor_k = k
        neighbor_out = flash_attn(neighbor_q, neighbor_k, v, ...)

        # Blend:
        out = se_wrapper.blend(grouped_out, neighbor_out, seq_len)
    """

    def __init__(self, config: LongContextConfig):
        self.neighbor_window = config.neighbor_window
        self.group_size = config.group_size
        self.short_threshold = config.short_context_threshold
        self.verbose = config.verbose

    def group_position_ids(
        self, position_ids: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Get grouped position IDs for the far-range attention pass."""
        return get_self_extend_position_ids(
            position_ids,
            self.neighbor_window,
            self.group_size,
            self.short_threshold,
        )

    def get_neighbor_slice(self, seq_len: int) -> int:
        """Return the start index for neighbor-only tokens."""
        if seq_len <= self.neighbor_window:
            return 0
        return seq_len - self.neighbor_window

    def blend(
        self,
        grouped_out: torch.Tensor,
        neighbor_out: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Blend grouped and neighbor attention outputs.

        For positions within neighbor_window, prefer neighbor attention.
        For earlier positions, use grouped attention.
        """
        if seq_len <= self.neighbor_window:
            return neighbor_out

        split = seq_len - self.neighbor_window
        # Earlier tokens: use grouped attention
        output = grouped_out.clone()
        # Recent tokens: use neighbor attention (which has finer position resolution)
        output[:, :, split:, :] = neighbor_out[:, :, split:, :]

        return output

    def __repr__(self) -> str:
        return (
            f"SelfExtend(NW={self.neighbor_window}, "
            f"GS={self.group_size})"
        )
