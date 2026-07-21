"""
sere.py — Dynamic Expert Skipping (SERE).

[Step 3] After router computation, redirects tokens from secondary experts
to the semantically most-similar primary expert, reducing activated experts.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class SEREModule:
    """Dynamic Expert Skip (SERE) — post-routing token redirection.

    All parameters are auto-tuned by default via ``auto_tune()``.
    Manual override is always possible by passing values explicitly.
    """

    @staticmethod
    def auto_tune(
        num_experts: int,
        top_k: int = 2,
    ) -> tuple[float, int]:
        """Auto-tune skip_threshold and min_experts from model config.

        Rules:
        - With more experts → more aggressive skipping (lower threshold).
          MoE-64 can skip more aggressively than MoE-8.
        - min_experts = clamp(1 to top_k-1): never skip below 1 active
          expert, but never skip all non-primary experts either.
        - For top_k=1 there is nothing to skip.

        Returns (skip_threshold, min_experts).
        """
        if top_k <= 1:
            return (0.0, 1)

        # Base threshold decreases with more experts (more opportunity to skip)
        base = 0.15
        if num_experts >= 64:
            threshold = base * 0.6   # 0.09
        elif num_experts >= 32:
            threshold = base * 0.8   # 0.12
        elif num_experts >= 16:
            threshold = base * 1.0   # 0.15
        else:
            threshold = base * 1.2   # 0.18  — fewer experts, be conservative

        # Higher top_k → can be more aggressive
        if top_k >= 4:
            threshold *= 0.85
        elif top_k >= 3:
            threshold *= 0.95

        # min_experts: always keep at least 1, but up to half of top_k
        min_exp = max(1, min(top_k // 2, top_k - 1))

        logger.info(
            "SERE auto_tune: num_experts=%d, top_k=%d → threshold=%.4f, min_experts=%d",
            num_experts, top_k, threshold, min_exp,
        )
        return (threshold, min_exp)

    def __init__(
        self,
        num_experts: int,
        top_k: int = 2,
        skip_threshold: float | None = None,
        min_experts: int | None = None,
    ):
        # Auto-tune if not explicitly provided
        if skip_threshold is None or min_experts is None:
            _st, _me = self.auto_tune(num_experts, top_k)
            skip_threshold = skip_threshold if skip_threshold is not None else _st
            min_experts = min_experts if min_experts is not None else _me
        self.num_experts = num_experts
        self.top_k = top_k
        self.skip_threshold = skip_threshold
        self.min_experts = min(min_experts, top_k)

        logger.info(
            "SEREModule: top_k=%d, skip_threshold=%.3f, min_experts=%d",
            top_k,
            skip_threshold,
            self.min_experts,
        )

    def __call__(
        self,
        router_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Apply SERE to router probabilities.

        [OEF] If ``self._oef_skip_suggestions`` is set (a ``set[int]`` of
        expert IDs), SERE additionally skips experts in that set, treating
        the OEF suggestion as an extra signal.  SERE retains absolute veto
        power — call ``clear_suggestions()`` on the OEF controller to reset.

        Returns (selected_probs, selected_indices, aux_loss).
        """
        b_size, t_len, _num_e = router_probs.shape

        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        gap_1_2 = top_k_probs[..., 0] - top_k_probs[..., 1]

        keep_mask = torch.ones_like(top_k_probs, dtype=torch.bool)

        # OEF: additional skip signal
        oef_suggestions: set[int] | None = getattr(self, '_oef_skip_suggestions', None)
        if oef_suggestions and len(oef_suggestions) > 0:
            oef_device = top_k_indices.device
            for k in range(self.min_experts, self.top_k):
                expert_ids = top_k_indices[..., k]  # (B, T)
                oef_mask = torch.zeros_like(gap_1_2, dtype=torch.bool)
                for e in oef_suggestions:
                    oef_mask = oef_mask | (expert_ids == e)
                # Merge with gap-based skip: skip if gap large OR OEF suggests
                gap_skip = gap_1_2 > self.skip_threshold
                skip_combined = gap_skip | oef_mask
                keep_mask[..., k] = ~skip_combined
        else:
            for k in range(self.min_experts, self.top_k):
                skip = gap_1_2 > self.skip_threshold
                keep_mask[..., k] = ~skip

        masked_probs = top_k_probs * keep_mask.to(top_k_probs.dtype)
        norm = masked_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        selected_probs = masked_probs / norm

        skipped_count = (~keep_mask).sum().item()
        if skipped_count > 0:
            total_slots = b_size * t_len * self.top_k
            logger.debug(
                "SERE: skipped %d/%d slots (%.1f%%)",
                skipped_count,
                total_slots,
                skipped_count / total_slots * 100,
            )

        # OEF suggestions are one-shot — consume and clear immediately
        # to prevent stale suggestions from leaking across steps.
        self._oef_skip_suggestions = None

        return selected_probs, top_k_indices, None
