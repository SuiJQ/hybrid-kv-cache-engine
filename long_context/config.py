# long_context/config.py
"""
LongContextConfig — Single source of truth for all long-context extension knobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ContextMethod = Literal["selfextend", "reattention", "yarn", "none"]


@dataclass
class LongContextConfig:
    """Configuration for training-free context window extension.

    All fields have sensible defaults.  The only thing you typically
    need to set is ``method`` (or pass ``--context-method`` on CLI).

    Strategy Selection
    ------------------
    - ``selfextend`` — Best for RoPE-based models (LLaMA, Mistral,
      Qwen, Gemma).  ~4 lines of position-id logic.  Extends 8-32×.
    - ``reattention`` — Best for any Transformer, doesn't depend on
      RoPE.  Adds a light top-k retrieval step before normal attention.
    - ``yarn`` — Industry baseline; already built into HF Transformers.
      Pass ``rope_scaling`` in model config.  Zero runtime overhead.
    - ``none`` — Disabled (fallback to original model context window).

    Compatibility
    -------------
    All methods are compatible with:
    - Goose speculative decoding
    - AFCE anchor extensions
    - KV cache quantization (already in HybridCache)
    - SERE dynamic expert skipping
    - OEF entropy freeze
    """

    # ── Method selection ──────────────────────────────────────────
    # SelfExtend is the default — 4 lines of position-id logic, works with
    # all RoPE-based models (LLaMA, Qwen, Mistral, Gemma, DeepSeek, etc.).
    enabled: bool = True
    method: ContextMethod = "selfextend"

    # ── SelfExtend ────────────────────────────────────────────────
    # Tokens within ``neighbor_window`` keep original position IDs.
    # Tokens beyond get position = (pos - NW) // group_size + NW.
    neighbor_window: int = 1024
    group_size: int = 8

    # ── ReAttention ───────────────────────────────────────────────
    # How many keys to keep after content-based top-k filtering.
    reattn_top_k: int = 2048
    reattn_top_k_ratio: float = 0.1  # fraction of total keys
    reattn_min_top_k: int = 128
    reattn_neighbor_window: int = 64  # always-visible recent tokens

    # ── YaRN / NTK-aware ─────────────────────────────────────────
    yarn_factor: float = 8.0
    yarn_original_max_len: int = 4096

    # ── Short-circuit ─────────────────────────────────────────────
    # Skip long-context logic when sequence length is below this
    # threshold.  Avoids degrading short-context performance.
    short_context_threshold: int = 2048

    # ── Debug ─────────────────────────────────────────────────────
    verbose: bool = False

    # ── Factory helpers ───────────────────────────────────────────

    @classmethod
    def from_cli(cls, args: object) -> "LongContextConfig":
        """Build config from parsed CLI args."""
        method_str = getattr(args, "context_method", "selfextend")
        if method_str not in ("selfextend", "reattention", "yarn", "none"):
            method_str = "selfextend"

        # --disable-long-context overrides method selection
        if getattr(args, "disable_long_context", False):
            return cls(enabled=False, method="none")

        return cls(
            enabled=method_str != "none",
            method=method_str,
            neighbor_window=getattr(args, "neighbor_window", cls.neighbor_window),
            group_size=getattr(args, "group_size", cls.group_size),
            reattn_top_k=getattr(args, "reattn_top_k", cls.reattn_top_k),
            reattn_top_k_ratio=getattr(args, "reattn_top_k_ratio", cls.reattn_top_k_ratio),
            reattn_min_top_k=getattr(args, "reattn_min_top_k", cls.reattn_min_top_k),
            reattn_neighbor_window=getattr(args, "reattn_neighbor_window", cls.reattn_neighbor_window),
            yarn_factor=getattr(args, "yarn_factor", cls.yarn_factor),
            yarn_original_max_len=getattr(args, "yarn_original_max_len", cls.yarn_original_max_len),
            short_context_threshold=getattr(args, "short_context_threshold", cls.short_context_threshold),
            verbose=getattr(args, "verbose", False),
        )

    def add_cli_args(self, parser: object) -> None:
        """Register CLI arguments on an argparse parser."""
        group = parser.add_argument_group("Long Context Extension")
        group.add_argument(
            "--context-method",
            choices=["selfextend", "reattention", "yarn", "none"],
            default="selfextend",
            help=(
                "Training-free context extension method "
                "(default: selfextend — 4 lines, no weight change, "
                "works with all RoPE models: LLaMA/Qwen/Mistral)"
            ),
        )
        group.add_argument(
            "--disable-long-context",
            action="store_true",
            default=False,
            help="Disable long context extension completely",
        )
        group.add_argument(
            "--neighbor-window", type=int, default=1024,
            help="SelfExtend: tokens within this window keep original positions",
        )
        group.add_argument(
            "--group-size", type=int, default=8,
            help="SelfExtend: floor-division group size for distant tokens",
        )
        group.add_argument(
            "--reattn-top-k", type=int, default=2048,
            help="ReAttention: max top-k keys to keep per query",
        )
        group.add_argument(
            "--reattn-top-k-ratio", type=float, default=0.1,
            help="ReAttention: fraction of keys to keep",
        )
        group.add_argument(
            "--reattn-min-top-k", type=int, default=128,
            help="ReAttention: minimum top-k keys (even for short inputs)",
        )
        group.add_argument(
            "--reattn-neighbor-window", type=int, default=64,
            help="ReAttention: always-visible recent tokens",
        )
        group.add_argument(
            "--yarn-factor", type=float, default=8.0,
            help="YaRN: context expansion factor",
        )
        group.add_argument(
            "--yarn-original-max-len", type=int, default=4096,
            help="YaRN: model's original max position",
        )
        group.add_argument(
            "--short-context-threshold", type=int, default=2048,
            help="Skip long-context logic below this sequence length",
        )

    def __repr__(self) -> str:
        active = self.method if self.enabled else "disabled"
        parts = [f"LongContextConfig({active}"]
        if self.enabled:
            if self.method == "selfextend":
                parts.append(f"NW={self.neighbor_window}, GS={self.group_size}")
            elif self.method == "reattention":
                parts.append(f"top_k={self.reattn_top_k}, NW={self.reattn_neighbor_window}")
            elif self.method == "yarn":
                parts.append(f"factor={self.yarn_factor}x")
        return ", ".join(parts) + ")"
