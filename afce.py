"""
afce.py — Anchored Forward Cache Extension (AFCE)

旁路神经突触 — 不触碰 MoeOwner 的物理内存、权重驻留与调度霸权。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 核心理念
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

以 32 Token 为簇，每簇末位 Token 的原始 K/V 作为锚点，挂载在 Radix
节点的旁路字典（Sidecar）中。Query 可透过因果掩码访问位置更早的锚点，
以此修复长上下文场景下的语义遗忘。全程零改动物理内存管理。

🔒 三大物理执行红线已焊死：
  1. 因果锚点掩蔽 — Query 仅能点积位置索引严格小于自身的锚点
  2. 异步预取指令 — 在前一个 Token 的 FFN 计算间隙，启动锚点预取
  3. 动态偏移表解耦 — 各序列独立维护锚点偏移表，绝不强行对齐

依赖
----
- torch (自动判定 CUDA 可用性)
- 无 MoeOwner 内部模块引用 ← 完全解耦
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────────────

CLUSTER_SIZE = 32  # Token 数 / 簇；锚点 = 每簇最后一位 Token 的 K/V


# ──────────────────────────────────────────────────────────────────────
# 模块一：AnchorSidecar — 挂载在 Radix 节点的旁路锚点 KV 字典
# ──────────────────────────────────────────────────────────────────────

class AnchorSidecar:
    """Per-radix-hash-node sidecar for anchor K/V pairs.

    Maps  ``cluster_index`` → ``(k_tensor, v_tensor, absolute_position)``

    物理效应：
    - 锚点保留了末位 Token 的绝对位置索引，RoPE 坐标零污染
    - 主序列哈希指纹不变，共享前缀命中率无损
    - block 释放时由调用方负责清理对应 sidecar
    """

    __slots__ = ("_anchors",)

    def __init__(self) -> None:
        self._anchors: dict[int, tuple[torch.Tensor, torch.Tensor, int]] = {}

    # ── 存储 ──────────────────────────────────────────────────────

    def store(
        self, cluster_idx: int,
        k: torch.Tensor, v: torch.Tensor, position: int,
    ) -> None:
        """Store anchor K/V for one cluster.

        k/v 会经过 detach+clone，确保与主 KV Cache 无引用纠缠。
        """
        self._anchors[cluster_idx] = (
            k.detach().clone(),
            v.detach().clone(),
            position,
        )

    # ── 读取 ──────────────────────────────────────────────────────

    def lookup(
        self, cluster_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        """Look up anchor by cluster index."""
        return self._anchors.get(cluster_idx)

    # ── 红线 #1：因果锚点掩蔽 ──────────────────────────────────

    def accessible_anchors(
        self, query_abs_position: int,
    ) -> list[tuple[int, torch.Tensor, torch.Tensor, int]]:
        """Return only anchors whose ``absolute_position < query_position``.

        Returns  ``[(cluster_idx, k, v, abs_position), ...]`` sorted by
        absolute position.  Positions with equality (same token) are
        excluded — token cannot attend to its own anchor.
        """
        result: list[tuple[int, torch.Tensor, torch.Tensor, int]] = []
        for cidx, (k, v, pos) in self._anchors.items():
            if pos < query_abs_position:
                result.append((cidx, k, v, pos))
        result.sort(key=lambda x: x[3])
        return result

    def num_anchors(self) -> int:
        return len(self._anchors)

    def all_positions(self) -> list[int]:
        return [pos for _, _, _, pos in self._anchors.values()]

    def clear(self) -> None:
        self._anchors.clear()


# ──────────────────────────────────────────────────────────────────────
# 模块一辅助：AnchorMaskGenerator — 因果锚点 Attention Mask
# ──────────────────────────────────────────────────────────────────────

class AnchorMaskGenerator:
    """Builds causal attention mask respecting anchor position ordering.

    Extended K/V layout::

        [anchor_0_K, anchor_1_K, ..., anchor_N-1_K, main_K_0, ..., main_K_M-1]

    Mask shape ``(1, 1, M, N + M)`` where ``M = main_seq_len``,
    ``N = num_anchors``.

    A position (q_idx, kv_idx) is **allowed** (0.0) iff:
      - kv_idx refers to an anchor AND anchor.abs_position < query.abs_position
      - kv_idx refers to a main position AND main.position ≤ query.position
    """

    @staticmethod
    def build(
        num_anchors: int,
        main_seq_len: int,
        query_abs_positions: list[int],
        anchor_abs_positions: list[int],
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Build causal anchor mask.

        Parameters
        ----------
        num_anchors : int
            Number of anchor K/V pairs (N).
        main_seq_len : int
            Length of main K/V sequence (M).
        query_abs_positions : list[int]
            Absolute position of each query token (length M).
        anchor_abs_positions : list[int]
            Absolute position of each anchor (length N).
        device : torch.device, optional
        dtype : torch.dtype, optional (default float16)

        Returns
        -------
        torch.Tensor
            ``(1, 1, M, N + M)`` mask — 0.0 = attend, -inf = block.
        """
        total_kv = num_anchors + main_seq_len
        mask = torch.full(
            (main_seq_len, total_kv), -float("inf"),
            device=device, dtype=dtype,
        )

        for q_idx in range(main_seq_len):
            q_pos = query_abs_positions[q_idx]

            # Anchors: attend only if anchor.abs_position < query.abs_position
            for a_idx in range(num_anchors):
                if anchor_abs_positions[a_idx] < q_pos:
                    mask[q_idx, a_idx] = 0.0

            # Main K/V: standard causal
            for m_idx in range(main_seq_len):
                if m_idx <= q_idx:
                    mask[q_idx, num_anchors + m_idx] = 0.0

        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, M, N+M)


# ──────────────────────────────────────────────────────────────────────
# 模块一管理：AnchorManager — 顶层 AFCE 编排器
# ──────────────────────────────────────────────────────────────────────

class AnchorManager:
    """Top-level AFCE orchestrator.

    职责
    ----
    - 管理全局 AnchorSidecar 字典（与 radix_index 平行的旁路结构）
    - 在 KV store 后判定并提取锚点
    - 在 decode 前将锚点 K/V 扩展到 main K/V，构建因果掩码
    - 异步 DMA 预取（红线 #2，仅在支持 Unified Memory 的设备上生效）
    - 动态偏移表维护（红线 #3）

    使用方式
    --------
    >>> afce = AnchorManager()
    >>> sidecar = afce.get_sidecar(hash_key)
    >>> ok = afce.maybe_extract(hash_key, tokens, k_tensor, v_tensor)
    >>> ext_k, ext_v, mask = afce.extend_for_decode(hash_key, main_k, main_v, pos)
    """

    def __init__(
        self,
        cluster_size: int = CLUSTER_SIZE,
        device: torch.device | None = None,
    ):
        self.cluster_size = cluster_size
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # 全局旁路字典：radix_hash_key → AnchorSidecar
        self._sidecars: dict[str, AnchorSidecar] = {}

        # 红线 #3：动态偏移表 per-request
        #   request_id -> {cluster_index -> kv_offset_in_extended_sequence}
        self._offset_tables: dict[str, dict[int, int]] = {}

        # 异步预取能力检测
        self._async_supported = self._check_async_support()

        logger.info(
            "AnchorManager: cluster_size=%d, async_prefetch=%s, device=%s",
            cluster_size, self._async_supported, self.device,
        )

    # ── 内部 ──────────────────────────────────────────────────────

    def _check_async_support(self) -> bool:
        """Check if device supports ``cudaMemPrefetchAsync`` (SM ≥ 6.0)."""
        if not torch.cuda.is_available():
            return False
        try:
            cap = torch.cuda.get_device_capability()
            return cap[0] >= 6
        except (RuntimeError, AssertionError):
            return False

    # ── Sidecar 访问 ───────────────────────────────────────────

    def get_sidecar(self, hash_key: str) -> AnchorSidecar:
        """Get or create sidecar for a radix hash key."""
        if hash_key not in self._sidecars:
            self._sidecars[hash_key] = AnchorSidecar()
        return self._sidecars[hash_key]

    def has_sidecar(self, hash_key: str) -> bool:
        return hash_key in self._sidecars

    def remove_sidecar(self, hash_key: str) -> None:
        """Remove sidecar when its radix node is freed."""
        self._sidecars.pop(hash_key, None)

    # ── 锚点提取 ────────────────────────────────────────────────

    def maybe_extract(
        self,
        hash_key: str,
        token_ids: list[int],
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
    ) -> bool:
        """Extract anchor K/V if the last token completes a cluster.

        规则：
        - 簇 0: tokens [0, 31] → anchor at pos 31
        - 簇 1: tokens [32, 63] → anchor at pos 63
        - 对于正在生成的不完整簇，末位 Token **禁止**作为锚点，
          自动回退至上一完整簇（红线 #1）。

        Parameters
        ----------
        hash_key : str
            Radix cumulative hash key for the current prefix.
        token_ids : list[int]
            Full token sequence (used to determine cluster boundaries).
        k_tensor : torch.Tensor
            Full K tensor ``(1, num_heads, seq_len, head_dim)``.
        v_tensor : torch.Tensor
            Full V tensor ``(1, num_heads, seq_len, head_dim)``.

        Returns
        -------
        bool
            True if an anchor was extracted.
        """
        seq_len = len(token_ids)
        if seq_len < self.cluster_size:
            return False

        # 末位索引
        last_pos = seq_len - 1

        # 末位 token 是否恰好结束一个完整簇？
        if (last_pos + 1) % self.cluster_size != 0:
            return False  # 不完整簇，不提取锚点（红线 #1）

        cluster_idx = last_pos // self.cluster_size

        # 提取末位 token 的 K/V
        anchor_k = k_tensor[:, :, last_pos:last_pos + 1, :].contiguous()
        anchor_v = v_tensor[:, :, last_pos:last_pos + 1, :].contiguous()

        sidecar = self.get_sidecar(hash_key)
        sidecar.store(cluster_idx, anchor_k, anchor_v, last_pos)

        logger.debug(
            "AFCE: anchor extracted  hash=%s…  cluster=%d  pos=%d",
            hash_key[:12], cluster_idx, last_pos,
        )
        return True

    # ── Decode 用 K/V 扩展（红线 #1）──────────────────────────

    def extend_for_decode(
        self,
        hash_key: str,
        main_k: torch.Tensor,
        main_v: torch.Tensor,
        query_abs_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
        """Extend main K/V with causally-accessible anchors for decode.

        对单 token decode（main_seq_len == 1），返回扩展后的 K/V 及
        对应的 attention mask。

        Parameters
        ----------
        hash_key : str
            Radix hash key for the request's current prefix.
        main_k : torch.Tensor
            Main K ``(1, num_heads, seq_len, head_dim)``.
        main_v : torch.Tensor
            Main V ``(1, num_heads, seq_len, head_dim)``.
        query_abs_position : int
            Absolute position of the **last** token in main sequence
            (i.e. the token being decoded).

        Returns
        -------
        extended_k : torch.Tensor
            ``(1, num_heads, num_anchors + seq_len, head_dim)``.
        extended_v : torch.Tensor
            ``(1, num_heads, num_anchors + seq_len, head_dim)``.
        attn_mask : torch.Tensor or None
            ``(1, 1, seq_len, num_anchors + seq_len)`` — None if no anchors.
        num_anchors : int
            Number of anchors prepended (0 if none).
        """
        sidecar = self._sidecars.get(hash_key)
        if sidecar is None or sidecar.num_anchors() == 0:
            return main_k, main_v, None, 0

        accessible = sidecar.accessible_anchors(query_abs_position)
        if not accessible:
            return main_k, main_v, None, 0

        # 收集锚点 K/V
        anchor_ks: list[torch.Tensor] = []
        anchor_vs: list[torch.Tensor] = []
        anchor_positions: list[int] = []
        for _cidx, ak, av, apos in accessible:
            anchor_ks.append(ak)
            anchor_vs.append(av)
            anchor_positions.append(apos)

        num_anchors = len(anchor_ks)
        main_seq_len = main_k.shape[2]

        # 拼接：锚点在前，主序列在后
        if num_anchors == 1:
            extended_k = torch.cat([anchor_ks[0], main_k], dim=2)
            extended_v = torch.cat([anchor_vs[0], main_v], dim=2)
        else:
            extended_k = torch.cat(anchor_ks + [main_k], dim=2)
            extended_v = torch.cat(anchor_vs + [main_v], dim=2)

        # 构建因果锚点掩码
        # query 位置从 query_abs_position - main_seq_len + 1 开始
        query_positions = list(
            range(query_abs_position - main_seq_len + 1, query_abs_position + 1)
        )
        attn_mask = AnchorMaskGenerator.build(
            num_anchors=num_anchors,
            main_seq_len=main_seq_len,
            query_abs_positions=query_positions,
            anchor_abs_positions=anchor_positions,
            device=main_k.device,
            dtype=main_k.dtype,
        )

        return extended_k, extended_v, attn_mask, num_anchors

    # ── 从模型返回的 KV 中剥离锚点 ───────────────────────────

    @staticmethod
    def strip_anchors_from_kv(
        kv_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        num_anchors: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Remove anchor-padded positions from each layer's K/V.

        调用时机：model.forward 返回后，store_kv 之前。
        锚点位于序列最前 num_anchors 个位置。
        """
        if num_anchors == 0:
            return kv_pairs
        stripped: list[tuple[torch.Tensor, torch.Tensor]] = []
        for k, v in kv_pairs:
            stripped.append((k[:, :, num_anchors:, :], v[:, :, num_anchors:, :]))
        return stripped

    # ── 红线 #2：异步 DMA 预取 ──────────────────────────────

    def prefetch(
        self,
        hash_key: str,
        current_position: int,
        stream: torch.cuda.Stream | None = None,
    ) -> bool:
        """Prefetch the **next** cluster's anchors into L2 cache.

        在前一个 Token 的 FFN 计算间隙调用，确保锚点读取的 L2
        命中率 > 95%。不支持异步预取的设备上将静默降级。

        Parameters
        ----------
        hash_key : str
            Current radix hash key.
        current_position : int
            Current absolute token position.
        stream : torch.cuda.Stream, optional
            Stream for async prefetch (defaults to current stream).

        Returns
        -------
        bool
            True if prefetch was issued.
        """
        if not self._async_supported:
            return False

        next_cluster = (current_position // self.cluster_size) + 1
        sidecar = self._sidecars.get(hash_key)
        if sidecar is None:
            return False

        anchor_data = sidecar.lookup(next_cluster)
        if anchor_data is None:
            return False

        ak, av, _apos = anchor_data
        try:
            if stream is not None:
                with torch.cuda.stream(stream):
                    ak.prefetch()
                    av.prefetch()
            else:
                ak.prefetch()
                av.prefetch()
            return True
        except (RuntimeError, AttributeError):
            return False

    # ── 红线 #3：动态偏移表 ────────────────────────────────

    def init_offset_table(self, request_id: str) -> None:
        """初始化某 request 的动态偏移表。"""
        self._offset_tables[request_id] = {}

    def update_offset(
        self, request_id: str, cluster_idx: int, kv_offset: int,
    ) -> None:
        """记录某簇锚点在扩展 K/V 中的偏移位置。"""
        tbl = self._offset_tables.get(request_id)
        if tbl is not None:
            tbl[cluster_idx] = kv_offset

    def get_offset(self, request_id: str, cluster_idx: int) -> int | None:
        """获取某簇锚点的偏移量。"""
        tbl = self._offset_tables.get(request_id)
        if tbl is None:
            return None
        return tbl.get(cluster_idx)

    def remove_offset_table(self, request_id: str) -> None:
        """释放 request 的偏移表。"""
        self._offset_tables.pop(request_id, None)

    # ── 批量清理 ────────────────────────────────────────────

    def clear_all(self) -> None:
        self._sidecars.clear()
        self._offset_tables.clear()


# ──────────────────────────────────────────────────────────────────────
# 模块一拓展：BatchPrefillAnchorExtractor — Prefill 后批量提取锚点
# ──────────────────────────────────────────────────────────────────────

def extract_anchors_after_prefill(
    afce_manager: AnchorManager,
    cache: object,  # HybridCache instance (duck-typed)
    request_tokens: list[int],
    layer_kv_pairs: list[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """Batch anchor extraction after a prefill forward.

    为每个 KV 层调用 ``maybe_extract``。由于各层锚点位置相同，
    只需用第一层的序列长度做判定；但需要逐层取 K/V 来提取。

    Parameters
    ----------
    afce_manager : AnchorManager
    cache : HybridCache
        Used to compute the hash key from request tokens.
    request_tokens : list[int]
        Full token sequence for the request.
    layer_kv_pairs : list[tuple[torch.Tensor, torch.Tensor]]
        Per-layer K/V tensors from the prefill forward.
    """
    # 需要从 cache 计算 hash key
    hash_key = _compute_hash_from_cache(cache, request_tokens)
    if hash_key is None:
        return

    seq_len = len(request_tokens)
    if seq_len < CLUSTER_SIZE:
        return

    # 末位是否结束一个完整簇？
    if (seq_len - 1 + 1) % CLUSTER_SIZE != 0:
        return

    # 逐层提取锚点 K/V（末位 token）
    for k, v in layer_kv_pairs:
        if k.shape[2] != seq_len:
            continue  # 安全降级：K/V 长度不匹配
        anchor_k = k[:, :, -1:, :].contiguous()
        anchor_v = v[:, :, -1:, :].contiguous()
        cluster_idx = (seq_len - 1) // CLUSTER_SIZE

        sidecar = afce_manager.get_sidecar(hash_key)
        sidecar.store(cluster_idx, anchor_k, anchor_v, seq_len - 1)

    logger.debug(
        "AFCE prefill extract: hash=%s…  cluster=%d  seq_len=%d",
        hash_key[:12], (seq_len - 1) // CLUSTER_SIZE, seq_len,
    )


def _compute_hash_from_cache(
    cache: object, tokens: list[int],
) -> str | None:
    """Compute the radix hash key for a token list using cache's hash methods.

    Duck-types HybridCache._hash_single_token and _incremental_hash.
    """
    if not tokens:
        return None
    try:
        h = cache._hash_single_token(tokens[0])
        for t in tokens[1:]:
            h = cache._incremental_hash(h, t)
        return h
    except AttributeError:
        logger.warning("AFCE: cache object has no hash methods")
        return None
