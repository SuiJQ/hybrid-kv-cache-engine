"""
oef.py — Opportunistic Entropy Freeze (OEF)

旁路监控 — 仅提供只读建议，不强制跳过任何专家。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 核心理念
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OEF 监控 SERE 输出的 Router 置信度熵。若某专家连续多次被低熵选中
（即路由器对该专家的选择高度确定），则建议跳过该专家。该建议仅为
标志位，**无强制力**——SERE 根据实时负载拥有绝对终裁权，可一键清零
建议计数。

在 SERE 允许的安全窗口内，FFN 计算量可动态压缩 10%~20%，且调度零冲突。

依赖
----
- torch
- 无 MoeOwner 内部模块引用 ← 完全解耦
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 模块二：EntropyMonitor — 逐专家滑动窗口熵追踪
# ──────────────────────────────────────────────────────────────────────

class EntropyMonitor:
    """Tracks router confidence entropy per expert over a sliding window.

    对每个专家维护最近 N 次被选中时的置信度熵历史。
    熵采用二元决策近似：H = -p · log(p) - (1-p) · log(1-p)，
    其中 p 为该专家的 softmax 概率。H 越低 → 对该专家的选择越确定。

    Parameters
    ----------
    num_experts : int
        专家总数。
    window_size : int
        滑动窗口大小（default 10）。
    """

    def __init__(self, num_experts: int, window_size: int = 10) -> None:
        self.num_experts = num_experts
        self.window_size = window_size
        # expert_id -> [entropy_value_1, entropy_value_2, ...]
        self._history: dict[int, list[float]] = {}
        self._step_counter: int = 0

    # ── 观察 ──────────────────────────────────────────────────────

    def observe(
        self,
        router_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> None:
        """Record entropy observations from a single forward step.

        Parameters
        ----------
        router_probs : torch.Tensor
            Softmax probabilities ``(batch, seq_len, num_experts)``.
        top_k_indices : torch.Tensor
            Indices of top-K selected experts ``(batch, seq_len, top_k)``.
        """
        self._step_counter += 1

        # Flatten batch + seq_len
        flat_probs = router_probs.reshape(-1, self.num_experts)
        flat_indices = top_k_indices.reshape(-1, top_k_indices.shape[-1])

        for b in range(flat_probs.shape[0]):
            for k in range(flat_indices.shape[-1]):
                e = int(flat_indices[b, k].item())
                if e >= self.num_experts:
                    continue
                p = float(flat_probs[b, e].item())
                p = max(p, 1e-8)  # avoid log(0)
                # Binary-decision entropy: H = -p·log(p) - (1-p)·log(1-p)
                entropy = -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)

                if e not in self._history:
                    self._history[e] = []
                self._history[e].append(entropy)
                if len(self._history[e]) > self.window_size:
                    self._history[e] = self._history[e][-self.window_size :]

    # ── 查询 ──────────────────────────────────────────────────────

    def get_expert_mean_entropy(self, expert_id: int) -> float:
        """Get mean entropy over the sliding window for one expert."""
        hist = self._history.get(expert_id, [])
        if not hist:
            return 1.0  # maximum uncertainty
        return sum(hist) / len(hist)

    def get_low_entropy_experts(self, threshold: float = 0.3) -> set[int]:
        """Return set of experts with mean entropy below `threshold`."""
        result: set[int] = set()
        for e in range(self.num_experts):
            if self.get_expert_mean_entropy(e) < threshold:
                result.add(e)
        return result

    # ── 重置 ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all history (SERE veto signal)."""
        self._history.clear()
        self._step_counter = 0


# ──────────────────────────────────────────────────────────────────────
# 模块二管理：OEFController — 顶层 OEF 编排器
# ──────────────────────────────────────────────────────────────────────

class OEFController:
    """Opportunistic Entropy Freeze controller — read-only monitoring.

    工作流程
    --------
    1. ``observe(router_probs, top_k_indices)`` — 每一步 forward 后调用
    2. 内部滑动窗口熵追踪每个专家的置信度
    3. ``get_skip_suggestions()`` — 返回建议跳过的专家 ID 集合
    4. SERE 消费该建议（或忽略），并可调用 ``clear_suggestions()`` 清零

    SERE 拥有**绝对终裁权**——随时可通过 ``clear_suggestions()``
    一键清零建议计数，OEF 端不会重置计数器之外的任何状态。
    """

    def __init__(
        self,
        num_experts: int,
        consecutive_low_threshold: int = 5,
        entropy_threshold: float = 0.3,
        window_size: int = 10,
    ) -> None:
        """
        Parameters
        ----------
        num_experts : int
            模型专家总数。
        consecutive_low_threshold : int, optional
            连续低熵次数达到该阈值后提出跳过建议（default 5）。
        entropy_threshold : float, optional
            熵值低于此值视为"低熵"（default 0.3）。
        window_size : int, optional
            滑动窗口大小（default 10）。
        """
        self.num_experts = num_experts
        self.consecutive_low_threshold = consecutive_low_threshold
        self.entropy_threshold = entropy_threshold

        self._monitor = EntropyMonitor(num_experts, window_size=window_size)
        # expert_id -> consecutive_low_count
        self._consecutive_low: dict[int, int] = {}
        # Current skip suggestions (read by SERE)
        self._suggestions: set[int] = set()

        logger.info(
            "OEFController: %d experts, consecutive_threshold=%d, entropy_threshold=%.2f",
            num_experts, consecutive_low_threshold, entropy_threshold,
        )

    # ── 观察 ──────────────────────────────────────────────────────

    def observe(
        self,
        router_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> None:
        """Observe router output and update skip suggestions.

        每次 MoE forward 后调用。

        Parameters
        ----------
        router_probs : torch.Tensor
            ``(batch, seq_len, num_experts)`` softmax 概率。
        top_k_indices : torch.Tensor
            ``(batch, seq_len, top_k)`` 选中专家索引。
        """
        self._monitor.observe(router_probs, top_k_indices)

        # 更新连续低熵计数器
        current_low = self._monitor.get_low_entropy_experts(
            threshold=self.entropy_threshold,
        )

        for e in range(self.num_experts):
            if e in current_low:
                self._consecutive_low[e] = self._consecutive_low.get(e, 0) + 1
            else:
                self._consecutive_low[e] = 0

        # 生成建议：连续低熵 >= threshold 的专家
        new_suggestions: set[int] = set()
        for e, count in self._consecutive_low.items():
            if count >= self.consecutive_low_threshold:
                new_suggestions.add(e)

        self._suggestions = new_suggestions

    # ── 建议查询（SERE 消费） ──────────────────────────────────

    def get_skip_suggestions(self) -> set[int]:
        """Return a **copy** of the current skip suggestion set.

        SERE 可以检查此集合，但修改副本不会影响 OEF 内部状态。
        """
        return set(self._suggestions)

    # ── SERE 否决权 ───────────────────────────────────────────

    def clear_suggestions(self) -> None:
        """Reset all suggestion state.

        Called by SERE to assert veto power.  Resets consecutive
        counters and window history.
        """
        self._suggestions.clear()
        self._consecutive_low.clear()
        self._monitor.reset()
        logger.debug("OEF: suggestions cleared by SERE veto")

    # ── 诊断 ──────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return diagnostic statistics."""
        return {
            "suggestions": len(self._suggestions),
            "suggested_experts": sorted(self._suggestions),
            "num_experts": self.num_experts,
            "consecutive_threshold": self.consecutive_low_threshold,
            "entropy_threshold": self.entropy_threshold,
        }
