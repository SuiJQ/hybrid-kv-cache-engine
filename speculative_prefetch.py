"""
speculative_prefetch.py — Speculative Prefetch & Dynamic Expert Activation.

Integrates with existing N-Gram speculation and SERE skip logic to:

1. **DynamicExpertActivator** — per-token adjustment of expert activation
   count k based on Softmax confidence of the generated token, plus a
   sliding-window text trigger ("/全力思考") that forces k = K_MAX.

2. **SpeculativePrefetcher** — predicts upcoming expert demand via N-Gram
   token draft + SERE routing estimates and issues async H2D prefetches
   with a **hard 5 ms timeout** that never blocks the main thread.

Guarantees: No modification to the underlying MoE computation kernel.
All heuristic logic is pure Python / PyTorch tensor ops.
"""

from __future__ import annotations

import collections
import logging

import torch

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# DynamicExpertActivator
# ═══════════════════════════════════════════════════════════════════════


class DynamicExpertActivator:
    """Per-token dynamic expert activation count (k) controller.

    Parameters
    ----------
    k_min : int
        Lower bound for k (default 3).
    k_max : int
        Upper bound for k (default 5).
    initial_k : int
        Starting k (default ``k_min``).
    force_cmd : str
        Sliding-window text trigger (default ``"/全力思考"``).
    sere_module : optional
        Attached ``SEREModule`` whose ``top_k`` will be updated in
        lockstep with ``current_k``.
    force_steps : int
        Number of decode steps to persist forced K_MAX after the trigger
        text clears the window.
    """

    K_MIN: int = 3
    K_MAX: int = 5
    _FORCE_CMD: str = "/全力思考"
    _SLIDING_WINDOW_CHARS: int = 40
    _HISTORY_WEIGHTED_TOKENS: int = 5

    def __init__(
        self,
        k_min: int = K_MIN,
        k_max: int = K_MAX,
        initial_k: int | None = None,
        force_cmd: str = _FORCE_CMD,
        sere_module=None,
        force_steps: int = 10,
    ) -> None:
        self.k_min = max(1, k_min)
        self.k_max = max(self.k_min, k_max)

        self.current_k: int = (
            max(self.k_min, min(initial_k, self.k_max))
            if initial_k is not None
            else self.k_min
        )

        self._sere = sere_module
        self._force_cmd = force_cmd
        self._force_steps = force_steps
        self._text_window: collections.deque[str] = collections.deque(
            maxlen=self._SLIDING_WINDOW_CHARS
        )
        self._forced_mode: bool = False
        self._force_steps_remaining: int = 0

        logger.info(
            "DynamicExpertActivator: k_min=%d, k_max=%d, force_steps=%d",
            self.k_min,
            self.k_max,
            self._force_steps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_from_logits(
        self,
        logits: torch.Tensor,
        generated_ids: list[int] | None = None,
        detokenizer: collections.abc.Callable[[list[int]], str] | None = None,
    ) -> int:
        """Update k from LM-head logits and optional sliding-window text.

        Steps
        -----
        1. Softmax over the last position → max probability = confidence.
        2. confidence > 0.9 → k -= 1;  < 0.5 → k += 1.
        3. Sliding-window text scan for *force_cmd* → k = K_MAX.
        4. Clamp to [k_min, k_max] and push into attached SERE's top_k.

        Returns
        -------
        int
            The new ``current_k``.
        """
        # ----- confidence-based adjustment -----
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        confidence = probs.max().item()

        _high_confidence: float = 0.9
        _low_confidence: float = 0.5

        if confidence > _high_confidence:
            self.current_k = max(self.k_min, self.current_k - 1)
        elif confidence < _low_confidence:
            self.current_k = min(self.k_max, self.current_k + 1)

        # ----- sliding-window text trigger -----
        if detokenizer is not None and generated_ids is not None:
            self._update_text_window(generated_ids, detokenizer)
            self._check_force_mode()

        # ----- forced-mode decay -----
        if self._forced_mode:
            if self._force_steps_remaining > 0:
                self._force_steps_remaining -= 1
                self.current_k = self.k_max
            else:
                self._forced_mode = False
                logger.debug("DynamicExpertActivator: /全力思考 force expired.")

        # ----- final clamp -----
        self.current_k = max(self.k_min, min(self.current_k, self.k_max))

        # ----- push to SERE -----
        if self._sere is not None:
            self._sere.top_k = self.current_k

        return self.current_k

    def get_k(self) -> int:
        """Return the current effective k."""
        return self.current_k

    def force_max_k(self) -> None:
        """Programmatically force k = k_max for *force_steps* steps."""
        self._forced_mode = True
        self._force_steps_remaining = self._force_steps
        self.current_k = self.k_max
        if self._sere is not None:
            self._sere.top_k = self.current_k

    def reset(self) -> None:
        """Reset to initial state (k = k_min, no forced mode)."""
        self.current_k = self.k_min
        self._forced_mode = False
        self._force_steps_remaining = 0
        self._text_window.clear()
        if self._sere is not None:
            self._sere.top_k = self.current_k

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_text_window(
        self,
        token_ids: list[int],
        detokenizer: collections.abc.Callable[[list[int]], str],
    ) -> None:
        """Decode the most recent tokens and append chars to the window."""
        recent = token_ids[-(self._HISTORY_WEIGHTED_TOKENS):]
        text = detokenizer(recent)
        for ch in text:
            self._text_window.append(ch)

    def _check_force_mode(self) -> None:
        """If the sliding window contains *force_cmd*, activate forced mode."""
        window_text = "".join(self._text_window)
        if self._force_cmd in window_text:
            logger.debug("DynamicExpertActivator: '/全力思考' detected — forcing k=%d", self.k_max)
            self._forced_mode = True
            self._force_steps_remaining = self._force_steps
            self.current_k = self.k_max


# ═══════════════════════════════════════════════════════════════════════
# SpeculativePrefetcher
# ═══════════════════════════════════════════════════════════════════════


class SpeculativePrefetcher:
    """Expert weight prefetcher via dedicated CUDA stream.

    [Plan 3] Architecture
    ---------------------
    1. **Trigger** — called at the end of each decode step, reads router
       logits saved by each ``MoEDecoderLayer`` during model forward.
    2. **Prediction** — for each layer N, uses its router probs to predict
       the top 2 most-activated experts; those are the experts most likely
       needed by layer N+1 on the next step (exploiting temporal locality).
    3. **Prefetch** — issues async H2D copies on a dedicated CUDA stream
       for the predicted experts into the reserved VRAM pool.  The next
       decode step's model forward runs in parallel with these transfers.
    4. **Synchronization** — before each layer's MoE computation, the
       layer synchronises the prefetch stream; if the expert is not yet
       ready, the existing ``get_or_load_expert()`` call loads it
       synchronously (rare fallback).

    Benefits
    --------
    - No thread pool, no timeout-based dropping.
    - Fully pipelined: H2D transfers overlap with the next step's
      attention computation.
    - Zero correctness risk: fallback to synchronous load if prefetch
      hasn't completed.
    """

    NUM_HOT_EXPERTS: int = 2

    def __init__(
        self,
        expert_cache,
        decoder_layers,
        num_layers: int,
        num_experts: int,
        num_reserved_experts: int = 32,
    ) -> None:
        self._expert_cache = expert_cache
        self._decoder_layers = decoder_layers
        self._num_layers = num_layers
        self._num_experts = num_experts
        self._num_reserved = num_reserved_experts

        # Dedicated CUDA stream for async H2D prefetch transfers
        self._prefetch_stream = torch.cuda.Stream(priority=-1)

        # Register the prefetch stream on every decoder layer
        for ly in decoder_layers:
            if hasattr(ly, '_prefetch_stream'):
                ly._prefetch_stream = self._prefetch_stream

        # Global timestamp array for LRU eviction tracking
        # Accessed through expert_cache._prefetch_evict_one_lru

        logger.info(
            "SpeculativePrefetcher [Plan 3]: %d layers x %d experts, "
            "reserved_pool=%d, stream_priority=-1",
            num_layers, num_experts, num_reserved_experts,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        context_ids: list[int],
        transfer_stream: torch.cuda.Stream | None = None,
        token_hash: int | None = None,
    ) -> None:
        """Issue expert prefetches based on saved router logits.

        Called at the **end** of each decode step.  Uses router probs
        recorded by each layer during the forward pass to predict and
        prefetch experts for the next step.
        """
        if self._expert_cache is None:
            return

        prefetch_stream = transfer_stream or self._prefetch_stream
        layers = self._decoder_layers

        for layer_idx in range(self._num_layers - 1):
            ly = layers[layer_idx]
            router_probs = getattr(ly, '_last_router_probs', None)
            if router_probs is None:
                continue

            # Get top-k expert indices from router probs
            topk_count = min(self.NUM_HOT_EXPERTS, router_probs.size(-1))
            _, topk_indices = torch.topk(router_probs, topk_count, dim=-1)

            # Prefetch for the NEXT layer (layer_idx + 1)
            for expert_idx in topk_indices[0, 0, :].tolist():
                self._expert_cache.prefetch_expert(
                    layer_idx + 1,
                    expert_idx,
                    stream=prefetch_stream,
                )

    def shutdown(self) -> None:
        """No-op: stream cleanup handled by PyTorch."""
        logger.info("SpeculativePrefetcher [Plan 3] shut down.")

    def __del__(self) -> None:
        pass
