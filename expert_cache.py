"""
expert_cache.py — Generalized Weight Cache Infrastructure (MoE-first).

[Step 1] Abstracts the token-sequence hybrid cache into a generic weight cache.
[Step 2] Hierarchical expert offloading: DRAM storage + VRAM cache with async H2D.
[Step 8] LFRU (Frequency-weighted Least Recently Used) eviction policy.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


# ===================================================================
# Layer 1 — Physical Block Cache (generic)
# ===================================================================


@dataclass
class WeightBlock:
    """A physical block descriptor for arbitrary weight data."""

    block_id: int
    payload: object
    payload_bytes: int
    ref_count: int = 1
    access_count: int = 1
    last_access_time: float = field(default_factory=time.monotonic)
    pinned_source: object = None

    def __repr__(self) -> str:
        return (
            f"WeightBlock(id={self.block_id}, bytes={self.payload_bytes}, "
            f"refs={self.ref_count}, accesses={self.access_count})"
        )


class PhysicalBlockCache:
    """Fixed-capacity physical block pool with free-list recycling."""

    def __init__(self, capacity: int, block_bytes: int):
        self.capacity = capacity
        self.block_bytes = block_bytes
        self.free_list: list[int] = list(range(capacity))
        self.allocated: dict[int, WeightBlock] = {}
        logger.info("PhysicalBlockCache: capacity=%d, block_bytes=%d", capacity, block_bytes)

    def alloc(self, payload: object, payload_bytes: int) -> WeightBlock:
        if not self.free_list:
            raise RuntimeError(f"PhysicalBlockCache exhausted (capacity={self.capacity}).")
        block_id = self.free_list.pop()
        block = WeightBlock(block_id=block_id, payload=payload, payload_bytes=payload_bytes)
        self.allocated[block_id] = block
        return block

    def free(self, block_id: int) -> None:
        block = self.allocated.pop(block_id, None)
        if block is None:
            logger.warning("PhysicalBlockCache.free: block %d not found", block_id)
            return
        self.free_list.append(block_id)

    def get(self, block_id: int) -> WeightBlock | None:
        return self.allocated.get(block_id)

    @property
    def used(self) -> int:
        return len(self.allocated)

    @property
    def free_count(self) -> int:
        return len(self.free_list)

    def stats(self) -> dict:
        return {
            "capacity": self.capacity,
            "used": self.used,
            "free": self.free_count,
            "block_bytes": self.block_bytes,
        }


# ===================================================================
# Layer 2 — Weight Hash Index (generic key → block_id)
# ===================================================================


class WeightHashIndex:
    """Generalised radix-index / hash-tree for arbitrary hashable keys."""

    def __init__(self):
        self._index: dict[str, int] = {}
        self._hash_tree: dict[str, list[str]] = {}

    def insert(self, key_hash: str, block_id: int, parent_hash: str | None = None) -> None:
        self._index[key_hash] = block_id
        if parent_hash is not None:
            self._hash_tree.setdefault(parent_hash, []).append(key_hash)

    def lookup(self, key_hash: str) -> int | None:
        return self._index.get(key_hash)

    def get_children(self, key_hash: str) -> list[str]:
        return self._hash_tree.get(key_hash, [])

    def remove_by_block(self, block_id: int) -> list[str]:
        stale = [h for h, bid in self._index.items() if bid == block_id]
        for h in stale:
            if self._index.get(h) == block_id:
                del self._index[h]
        return stale

    def size(self) -> int:
        return len(self._index)


# ===================================================================
# LFRU helper
# ===================================================================


def _lfru_score(block: WeightBlock, now: float, eps: float = 1e-6) -> float:
    age = now - block.last_access_time
    return block.access_count / (age + eps)


# ===================================================================
# Layer 3 — Expert Weight Cache (MoE-specific)
# ===================================================================


class ExpertWeightCache:
    """Hierarchical MoE expert weight cache.

    [Step 2] All expert parameters in pinned host memory; limited VRAM cache.
    [Step 8] LFRU eviction.
    """

    def __init__(
        self,
        vram_capacity: int,
        block_bytes: int,
        num_layers: int,
        num_experts: int,
        intermediate_size: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.block_bytes = block_bytes

        self.vram_cache = PhysicalBlockCache(capacity=vram_capacity, block_bytes=block_bytes)
        self.hash_index = WeightHashIndex()

        expert_weight_bytes = (
            intermediate_size * hidden_size * 2  # gate
            + intermediate_size * hidden_size * 2  # up
            + hidden_size * intermediate_size * 2  # down
        )

        logger.info(
            "ExpertWeightCache: %d layers x %d experts, %.1f GiB DRAM required",
            num_layers,
            num_experts,
            (num_layers * num_experts * expert_weight_bytes) / (1024**3),
        )

        self.dram_gate: list[list[torch.Tensor]] = []
        self.dram_up: list[list[torch.Tensor]] = []
        self.dram_down: list[list[torch.Tensor]] = []

        self._transfer_stream: torch.cuda.Stream | None = None
        self._now: float = time.monotonic()

    # ------------------------------------------------------------------
    # Expert registration
    # ------------------------------------------------------------------

    def register_expert(
        self,
        layer_idx: int,
        expert_idx: int,
        gate_w: torch.Tensor,
        up_w: torch.Tensor,
        down_w: torch.Tensor,
    ) -> None:
        """Register an expert's weights in pinned host memory."""
        ly = layer_idx
        ex = expert_idx

        while len(self.dram_gate) <= ly:
            self.dram_gate.append([])
            self.dram_up.append([])
            self.dram_down.append([])

        while len(self.dram_gate[ly]) <= ex:
            self.dram_gate[ly].append(None)
            self.dram_up[ly].append(None)
            self.dram_down[ly].append(None)

        self.dram_gate[ly][ex] = gate_w.contiguous().pin_memory()
        self.dram_up[ly][ex] = up_w.contiguous().pin_memory()
        self.dram_down[ly][ex] = down_w.contiguous().pin_memory()

        logger.debug(
            "Registered expert L%d.E%d: gate=%s, up=%s, down=%s",
            ly,
            ex,
            list(self.dram_gate[ly][ex].shape),
            list(self.dram_up[ly][ex].shape),
            list(self.dram_down[ly][ex].shape),
        )

    # ------------------------------------------------------------------
    # Expert loading / H2D transfer
    # ------------------------------------------------------------------

    def get_or_load_expert(
        self,
        layer_idx: int,
        expert_idx: int,
        stream: torch.cuda.Stream | None = None,
    ) -> tuple[WeightBlock, dict[str, torch.Tensor]] | None:
        """Return the VRAM block for an expert, loading from DRAM if needed."""
        self._now = time.monotonic()
        ly = layer_idx
        ex = expert_idx
        key = self._expert_key(ly, ex)

        bid = self.hash_index.lookup(key)
        if bid is not None:
            block = self.vram_cache.get(bid)
            if block is not None:
                block.ref_count += 1
                block.access_count += 1
                block.last_access_time = self._now
                return block, block.payload

        if ly >= len(self.dram_gate) or ex >= len(self.dram_gate[ly]):
            logger.warning("Expert L%d.E%d not registered in DRAM", ly, ex)
            return None

        gate_cpu = self.dram_gate[ly][ex]
        up_cpu = self.dram_up[ly][ex]
        down_cpu = self.dram_down[ly][ex]
        if gate_cpu is None:
            return None

        payload_bytes = (
            gate_cpu.numel() * gate_cpu.element_size()
            + up_cpu.numel() * up_cpu.element_size()
            + down_cpu.numel() * down_cpu.element_size()
        )

        if self.vram_cache.free_count == 0:
            self._evict_one()

        gpu_stream = stream or torch.cuda.current_stream()
        with torch.cuda.stream(gpu_stream):
            gate_gpu = gate_cpu.to(device="cuda", non_blocking=True)
            up_gpu = up_cpu.to(device="cuda", non_blocking=True)
            down_gpu = down_cpu.to(device="cuda", non_blocking=True)
            weight_dict = {"gate": gate_gpu, "up": up_gpu, "down": down_gpu}

            block = self.vram_cache.alloc(payload=weight_dict, payload_bytes=payload_bytes)
            block.pinned_source = (gate_cpu, up_cpu, down_cpu)

        self.hash_index.insert(key, block.block_id)
        block.ref_count += 1
        block.access_count = 1
        block.last_access_time = self._now

        logger.debug(
            "Expert L%d.E%d loaded to VRAM block %d (%.1f MB)",
            ly,
            ex,
            block.block_id,
            payload_bytes / (1024 * 1024),
        )
        return block, weight_dict

    # ------------------------------------------------------------------
    # [Step 8] LFRU eviction
    # ------------------------------------------------------------------

    def _evict_one(self) -> int | None:
        if self.vram_cache.used == 0:
            return None
        now = self._now
        worst_id: int | None = None
        worst_score = float("inf")

        for bid, block in self.vram_cache.allocated.items():
            score = _lfru_score(block, now)
            if score < worst_score:
                worst_score = score
                worst_id = bid

        if worst_id is not None:
            block = self.vram_cache.allocated[worst_id]
            logger.debug(
                "LFRU evict: block %d (accesses=%d, score=%.4f)",
                worst_id,
                block.access_count,
                worst_score,
            )
            self.vram_cache.free(worst_id)
            self.hash_index.remove_by_block(worst_id)

        return worst_id

    def evict_up_to(self, needed_slots: int) -> int:
        evicted = 0
        while self.vram_cache.free_count < needed_slots:
            if self._evict_one() is None:
                break
            evicted += 1
        return evicted

    def release_expert(self, layer_idx: int, expert_idx: int) -> None:
        key = self._expert_key(layer_idx, expert_idx)
        bid = self.hash_index.lookup(key)
        if bid is None:
            return
        block = self.vram_cache.get(bid)
        if block is None:
            return
        block.ref_count -= 1
        if block.ref_count <= 0:
            self.vram_cache.free(bid)
            self.hash_index.remove_by_block(bid)

    # ------------------------------------------------------------------
    # Async H2D prefetch
    # ------------------------------------------------------------------

    def _prefetch_evict_one_lru(self) -> int | None:
        """Evict the least recently used (oldest access_time) expert block."""
        if self.vram_cache.used == 0:
            return None
        oldest_id: int | None = None
        oldest_time = float("inf")
        for bid, block in self.vram_cache.allocated.items():
            if block.last_access_time < oldest_time:
                oldest_time = block.last_access_time
                oldest_id = bid
        if oldest_id is not None:
            self.vram_cache.free(oldest_id)
            self.hash_index.remove_by_block(oldest_id)
        return oldest_id

    def prefetch_expert(
        self,
        layer_idx: int,
        expert_idx: int,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Prefetch an expert's weights from DRAM to VRAM (non-blocking).

        [Plan 3] LRU eviction when reserved pool is full; update timestamp
        for already-loaded experts.
        """
        key = self._expert_key(layer_idx, expert_idx)
        existing_bid = self.hash_index.lookup(key)
        if existing_bid is not None:
            block = self.vram_cache.get(existing_bid)
            if block is not None:
                block.last_access_time = time.monotonic()
            return

        if self.vram_cache.free_count == 0:
            self._prefetch_evict_one_lru()

        ly = layer_idx
        ex = expert_idx
        if ly >= len(self.dram_gate) or ex >= len(self.dram_gate[ly]):
            return

        gate_cpu = self.dram_gate[ly][ex]
        if gate_cpu is None:
            return

        payload_bytes = gate_cpu.numel() * gate_cpu.element_size() * 3
        gpu_stream = stream or torch.cuda.current_stream()
        with torch.cuda.stream(gpu_stream):
            gate_gpu = gate_cpu.to(device="cuda", non_blocking=True)
            up_gpu = self.dram_up[ly][ex].to(device="cuda", non_blocking=True)
            down_gpu = self.dram_down[ly][ex].to(device="cuda", non_blocking=True)
            weight_dict = {"gate": gate_gpu, "up": up_gpu, "down": down_gpu}

            block = self.vram_cache.alloc(payload=weight_dict, payload_bytes=payload_bytes)
            block.pinned_source = (gate_cpu, self.dram_up[ly][ex], self.dram_down[ly][ex])

        self.hash_index.insert(key, block.block_id)
        block.access_count = 0
        block.last_access_time = time.monotonic()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expert_key(layer_idx: int, expert_idx: int) -> str:
        return f"expert_{layer_idx}_{expert_idx}"

    def stats(self) -> dict:
        return {
            "vram_used": self.vram_cache.used,
            "vram_capacity": self.vram_cache.capacity,
            "dram_experts": self.num_layers * self.num_experts,
            "hash_index_size": self.hash_index.size(),
        }
