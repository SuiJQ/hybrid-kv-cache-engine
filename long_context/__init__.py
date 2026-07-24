# long_context/__init__.py
"""
MoeOwner Long Context Extension Suite
=======================================

Two training-free, framework-level context window extension techniques
that require ZERO model weight modification:

  1. SelfExtend  (ICML 2024 Spotlight)  —  floor-divided RoPE grouping
  2. ReAttention (ICLR 2025)            —  position-agnostic top-k + finite attention

Both integrate into the existing attention injection pipeline without
changing any model weights, architecture, or training procedure.

Design Principles
-----------------
- Zero weight modification
- Zero training / fine-tuning
- Zero architecture changes (pure inference-time logic)
- Full compatibility with existing hooks (Goose speculative decoding, KV cache)
- Configurable per-deployment via CLI flags or config dict
"""

from .config import LongContextConfig, ContextMethod
from .self_extend import SelfExtendWrapper, get_self_extend_position_ids
from .re_attention import ReAttentionWrapper

__all__ = [
    "LongContextConfig",
    "ContextMethod",
    "SelfExtendWrapper",
    "get_self_extend_position_ids",
    "ReAttentionWrapper",
]
