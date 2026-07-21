"""
cache_manager.py — Hybrid Radix-Tree / Block-Based KV Cache Manager

Provides:
  - Block: a single KV cache block descriptor.
  - HybridCache: manages block allocation, prefix matching via radix index,
    reference counting, garbage collection, and GPU-memory-aware sizing.
  - Prefix cache: pinned block support for automatic prefix caching.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
from array import array

import torch

logger = logging.getLogger(__name__)


class Block:
    """A single KV-cache block descriptor.

    Uses ``__slots__`` with ``hash_chain`` stored as ``array('i')``
    for compact memory layout and fast integer iteration.
    """

    __slots__ = ("block_id", "hash_chain", "kv_tensor", "ref_count", "pinned")

    def __init__(
        self,
        block_id: int,
        token_ids: list[int],
        kv_tensor: object = None,
        ref_count: int = 1,
        pinned: bool = False,
    ) -> None:
        self.block_id = block_id
        self.hash_chain: array = array("i", token_ids)
        self.kv_tensor = kv_tensor
        self.ref_count = ref_count
        self.pinned = pinned

    def __repr__(self) -> str:
        return (
            f"Block(id={self.block_id}, chain_len={len(self.hash_chain)}, "
            f"refs={self.ref_count}, pinned={self.pinned})"
        )


class HybridCache:
    """
    A hybrid radix-tree / block-based KV cache.

    Features:
      - Incremental BLAKE2b token hashing.
      - GPU-memory-aware total_blocks calculation.
      - Free-block queue for O(1) allocation.
      - Radix index for longest-prefix matching with LRU cache.
      - Reference-count-based GC.
      - Hash-tree prefetch on leaf cache hit.
      - Pinned block support for prefix caching (automatic LRU eviction).

    [Bug 2] Radix index uses ``dict[str, set[int]]`` so multiple blocks
    sharing the same prefix hash are all tracked — no 1-to-1 overwrite.
    """

    _PREFETCH_FANOUT_MAX: int = 3
    _MEM_FRACTION: float = 0.50
    _PROTECTED_N: int = 8      # number of head/tail tokens kept FP16
    _PINNED_MAX: int = 256     # max pinned prefix-cache blocks
    _PINNED_EVICT_FRACTION: float = 0.20  # evict oldest 20% when full

    def __init__(
        self,
        block_size: int = 16,
        hidden_size: int = 4096,
        total_blocks: int | None = None,
        mem_fraction: float | None = None,
    ) -> None:
        self.block_size = block_size
        self.hidden_size = hidden_size

        if mem_fraction is not None:
            self._MEM_FRACTION = mem_fraction

        if total_blocks is not None:
            self.total_blocks = total_blocks
        else:
            self.total_blocks = self._compute_total_blocks(
                block_size, hidden_size, self._MEM_FRACTION
            )

        self._slot_bytes = self.block_size * self.hidden_size * 2 * 2

        self.free_block_queue: list[int] = list(range(self.total_blocks))
        self.allocated_blocks: dict[int, Block] = {}
        # [Bug 2] set[int] instead of int: multiple blocks can share a prefix.
        self.radix_index: dict[str, set[int]] = {}

        self._hash_tree: dict[str, list[str]] = {}

        # [Bug 9] LRU cache keyed on tuple of first 8 tokens, not hash().
        self._match_cache: dict[tuple, tuple[int | None, int]] = {}
        self._match_cache_lru: list[tuple] = []
        self._match_cache_max = 256

        # Prefix caching: ordered list of pinned block IDs (oldest first)
        self._pinned_lru: list[int] = []

        self._prefetch_stream: object | None = None

        # BlockID → KV payload cache
        # FP16 mode:   {"mode": "fp16", "kv": [(k, v), ...]}
        # Mixed mode:  {"mode": "mixed", "seq_len": int,
        #                "layers": [{k_head,v_head, k_body_q,k_body_scale,
        #                            v_body_packed,v_body_scale,v_body_bias,
        #                            k_tail,v_tail}, ...]}
        self._block_kv_cache: dict[int, dict] = {}

        logger.info(
            "HybridCache initialized: block_size=%d, total_blocks=%d, "
            "mem_frac=%.2f, pinned_max=%d",
            self.block_size,
            self.total_blocks,
            self._MEM_FRACTION,
            self._PINNED_MAX,
        )

    # ------------------------------------------------------------------
    # Memory-aware sizing
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_total_blocks(
        block_size: int = 16, hidden_size: int = 4096, mem_fraction: float = 0.50
    ) -> int:
        try:
            import torch  # noqa: PLC0415

            free_mem, _ = torch.cuda.mem_get_info()
            slot_bytes = block_size * hidden_size * 2 * 2
            total = int((free_mem * mem_fraction) / slot_bytes)
            logger.info("GPU free mem: %d bytes -> total_blocks=%d", free_mem, total)
            return max(total, 1)
        except (ImportError, RuntimeError, AttributeError):
            logger.warning("torch.cuda unavailable; using default total_blocks=10000")
            return 10_000

    def compute_hash(self, token_ids: list[int]) -> str:
        """Compute the cumulative radix hash for a token sequence.

        Used by AFCE to look up the corresponding AnchorSidecar.
        Also used by prefix caching logic.
        """
        if not token_ids:
            return ""
        h = self._hash_single_token(token_ids[0])
        for t in token_ids[1:]:
            h = self._incremental_hash(h, t)
        return h

    def get_gpu_memory_slot(self, block_id: int) -> int:
        return block_id * self._slot_bytes

    # ------------------------------------------------------------------
    # Incremental hash
    # ------------------------------------------------------------------

    @staticmethod
    def _incremental_hash(previous_hex: str, token: int) -> str:
        """Fast incremental radix hash using BLAKE2b.

        Replaced SHA-256 (cryptographic overkill for a radix tree key)
        with BLAKE2b digest_size=8, giving ~5x throughput on the hot path
        while retaining negligible collision probability (2^-64).
        """
        prev_bytes = bytes.fromhex(previous_hex)
        token_bytes = token.to_bytes(4, "little", signed=True)
        return hashlib.blake2b(prev_bytes + token_bytes, digest_size=8).hexdigest()

    @staticmethod
    def _hash_single_token(token: int) -> str:
        """Fast single-token BLAKE2b hash (replaces SHA-256)."""
        token_bytes = token.to_bytes(4, "little", signed=True)
        return hashlib.blake2b(token_bytes, digest_size=8).hexdigest()

    # ------------------------------------------------------------------
    # Prefix caching: pin / unpin / evict
    # ------------------------------------------------------------------

    def pin_block(self, block_id: int) -> None:
        """Pin a block so it is preserved for prefix cache reuse.

        Pinned blocks are skipped during normal free_block and FIFO
        eviction.  They are tracked in a separate LRU list and only
        evicted when the pinned count exceeds ``_PINNED_MAX``.
        """
        block = self.allocated_blocks.get(block_id)
        if block is None:
            logger.warning("pin_block: block %d not found", block_id)
            return
        if not block.pinned:
            block.pinned = True
            self._pinned_lru.append(block_id)
            logger.debug("Block %d pinned (total pinned: %d)", block_id, len(self._pinned_lru))

    def unpin_block(self, block_id: int) -> None:
        """Release a pin on a block.  Does NOT free it, just marks freeable."""
        block = self.allocated_blocks.get(block_id)
        if block is None:
            return
        block.pinned = False
        with contextlib.suppress(ValueError):
            self._pinned_lru.remove(block_id)
        logger.debug("Block %d unpinned", block_id)

    def pin_prefix_from_match(self, block_id: int) -> bool:
        """After a successful prefix match, pin the matched block.

        Returns True if the block was newly pinned (or already pinned).
        Automatically evicts the oldest pinned block if pinned count
        exceeds ``_PINNED_MAX``.

        This is the high-level API the scheduler calls after a prefix
        match succeeds.
        """
        block = self.allocated_blocks.get(block_id)
        if block is None:
            return False

        # Already pinned — just touch LRU order
        if block.pinned:
            with contextlib.suppress(ValueError):
                self._pinned_lru.remove(block_id)
            self._pinned_lru.append(block_id)
            return True

        # Evict oldest pinned block if at capacity
        if len(self._pinned_lru) >= self._PINNED_MAX:
            self._evict_oldest_pinned()

        self.pin_block(block_id)
        return True

    def _evict_oldest_pinned(self) -> int:
        """Evict the 20% oldest pinned blocks, LRU-fashion.

        Returns number of blocks evicted.
        """
        evict_count = max(1, int(len(self._pinned_lru) * self._PINNED_EVICT_FRACTION))
        evicted = 0
        for _ in range(evict_count):
            if not self._pinned_lru:
                break
            oldest = self._pinned_lru.pop(0)
            block = self.allocated_blocks.get(oldest)
            if block and block.pinned:
                block.pinned = False
                self._free_block_forced(oldest)
                evicted += 1
        if evicted:
            logger.debug("Evicted %d oldest pinned blocks", evicted)
        return evicted

    # ------------------------------------------------------------------
    # Full prefix match: is this exact token sequence cached?
    # ------------------------------------------------------------------

    def has_prefix(self, token_ids: list[int]) -> int | None:
        """Check if an exact token sequence is cached as a pinned block.

        Returns block_id of the matching pinned block, or None.
        This is the fast path for ``submit()``: exact match → skip prefill.
        """
        if not token_ids:
            return None
        cumulative_hash = self._hash_single_token(token_ids[0])
        for t in token_ids[1:]:
            cumulative_hash = self._incremental_hash(cumulative_hash, t)

        candidates = self.radix_index.get(cumulative_hash, set())
        for bid in candidates:
            block = self.allocated_blocks.get(bid)
            if block is not None and block.pinned:
                return bid
        return None

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def _free_block_forced(self, block_id: int) -> None:
        """Force-free a block (FIFO eviction path).

        Skips pinned blocks — they are managed separately by
        ``_evict_oldest_pinned``.
        """
        block = self.allocated_blocks.get(block_id)
        if block is not None and block.pinned:
            return

        radix = self.radix_index
        block = self.allocated_blocks.pop(block_id, None)
        if block is None:
            return
        self.free_block_queue.append(block_id)
        self._block_kv_cache.pop(block_id, None)
        stale = [h for h, bids in radix.items() if block_id in bids]
        for h in stale:
            radix[h].discard(block_id)
            if not radix[h]:
                del radix[h]

    def allocate(self, prompt_tokens: list[int]) -> Block:
        """
        Allocate a new block for the given prompt tokens.

        [FIFO Eviction] When free pool is exhausted, batch-evict the oldest
        10%% of allocated blocks (by allocation order) without ref_count check.
        Skips pinned blocks.
        """
        if not prompt_tokens:
            raise ValueError("allocate() requires at least one token")

        free = self.free_block_queue
        blocks = self.allocated_blocks
        radix = self.radix_index
        tree = self._hash_tree

        if not free:
            n_evict = max(1, int(self.total_blocks * 0.10))
            evicted = 0
            # Evict oldest unpinned blocks (iterate copy to avoid mutation issues)
            candidates = list(blocks.keys())
            for bid in candidates[:n_evict * 2]:  # extra headroom for pinned skip
                if not free:
                    break
                if bid in blocks and not blocks[bid].pinned:
                    self._free_block_forced(bid)
                    evicted += 1
                if evicted >= n_evict:
                    break
            logger.debug("FIFO eviction: evicted %d oldest unpinned blocks", evicted)
            if not free:
                # Try evicting one pinned block as last resort
                self._evict_oldest_pinned()
                if not free:
                    raise RuntimeError("No free blocks available after FIFO eviction")

        block_id = free.pop()

        new_block = Block(
            block_id=block_id,
            token_ids=prompt_tokens,
            kv_tensor=None,
            ref_count=1,
            pinned=False,
        )
        blocks[block_id] = new_block

        cumulative = self._hash_single_token(prompt_tokens[0])
        prev_hash: str = cumulative
        radix.setdefault(cumulative, set()).add(block_id)
        tree.setdefault(cumulative, [])

        for token in prompt_tokens[1:]:
            cumulative = self._incremental_hash(cumulative, token)
            radix.setdefault(cumulative, set()).add(block_id)
            tree.setdefault(prev_hash, []).append(cumulative)
            prev_hash = cumulative

        logger.debug(
            "Allocated block %d (tokens=%s...)",
            block_id,
            str(prompt_tokens[:4])[:-1],
        )
        return new_block

    # ------------------------------------------------------------------
    # Prefix matching
    # ------------------------------------------------------------------

    def match_prefix(self, prompt_tokens: list[int]) -> tuple[int | None, list[int]]:
        """
        Find the longest prefix of ``prompt_tokens`` that exists in the cache.

        [Bug 2] Tracks candidate block sets through each step via intersection,
        so only consistent prefix paths are returned.  When no set is consistent
        with the prompt, returns the last consistent prefix.
        """
        blocks = self.allocated_blocks
        radix = self.radix_index
        match_cache = self._match_cache
        match_lru = self._match_cache_lru

        # [Bug 9] Use tuple of first 8 tokens directly as cache key.
        # Use tuple of (first 8 tokens, total length) as cache key so that
        # different prompts with the same first 8 tokens don't collide.
        cache_key = (tuple(prompt_tokens[:8]), len(prompt_tokens))
        if cache_key in match_cache:
            block_id, matched_len = match_cache[cache_key]
            with contextlib.suppress(ValueError):
                match_lru.remove(cache_key)
            match_lru.append(cache_key)
            if block_id is not None:
                block = blocks.get(block_id)
                if block is not None:
                    block.ref_count += 1
                if block is None:
                    # Stale cache entry — block was freed; invalidate and fall
                    # through to the fresh lookup path below.
                    match_cache.pop(cache_key, None)
                    with contextlib.suppress(ValueError):
                        match_lru.remove(cache_key)
                else:
                    if matched_len == len(prompt_tokens):
                        self._try_prefetch_children(block)
                    return block_id, prompt_tokens[matched_len:]
            return None, prompt_tokens

        cumulative_hash: str = ""
        last_matched_block_id: int | None = None
        split_index: int = 0
        prev_candidates: set[int] | None = None

        for i, token in enumerate(prompt_tokens):
            if i == 0:
                cumulative_hash = self._hash_single_token(token)
            else:
                cumulative_hash = self._incremental_hash(cumulative_hash, token)

            step_candidates = radix.get(cumulative_hash, set())
            if not step_candidates:
                break

            if prev_candidates is not None:
                consistent = step_candidates & prev_candidates
            else:
                consistent = step_candidates

            if not consistent:
                break

            last_matched_block_id = next(iter(consistent))
            split_index = i + 1
            prev_candidates = consistent

        result = (last_matched_block_id, split_index)
        match_cache[cache_key] = result
        match_lru.append(cache_key)
        if len(match_lru) > self._match_cache_max:
            old_key = match_lru.pop(0)
            match_cache.pop(old_key, None)

        if last_matched_block_id is not None:
            block = blocks.get(last_matched_block_id)
            if block is not None:
                block.ref_count += 1
            if split_index == len(prompt_tokens) and block is not None:
                self._try_prefetch_children(block)
            return last_matched_block_id, prompt_tokens[split_index:]
        else:
            return None, prompt_tokens

    # ------------------------------------------------------------------
    # [Step 7] Hash-tree prefetch
    # ------------------------------------------------------------------

    def _try_prefetch_children(self, block: Block) -> None:
        try:
            import torch  # noqa: PLC0415

            chain = block.hash_chain
            if len(chain) == 0:
                return
            leaf_hash = self._hash_single_token(chain[0])
            for tok in chain[1:]:
                leaf_hash = self._incremental_hash(leaf_hash, tok)

            children = self._hash_tree.get(leaf_hash, [])
            if not children:
                return

            children = children[: self._PREFETCH_FANOUT_MAX]
            radix = self.radix_index
            allocated = self.allocated_blocks

            # Static check: enough free blocks instead of dynamic cudaMemGetInfo
            if len(self.free_block_queue) < len(children):
                return

            if self._prefetch_stream is None:
                self._prefetch_stream = torch.cuda.Stream(priority=-1)

            with torch.cuda.stream(self._prefetch_stream):
                for child_hash in children:
                    cids = radix.get(child_hash, set())
                    for cid in cids:
                        child_block = allocated.get(cid)
                        if child_block is None:
                            continue
                        kv = child_block.kv_tensor
                        if kv is None:
                            continue
                        # Dict mode (quantized/fp16 list) — skip prefetch
                        if isinstance(kv, dict):
                            continue
                        if kv.device.type != "cuda":
                            _ = kv.to(device="cuda", non_blocking=True)
        except RuntimeError as _rexc:
            logger.warning("_try_prefetch_children RuntimeError: %s", _rexc)
        except Exception as _eexc:
            logger.warning("_try_prefetch_children unexpected error: %s", _eexc)

    # ==================================================================
    # KV asymmetric quantization kernels
    #   Key  → INT8  symmetric (per-head scale)
    #   Value→ INT4  packed asymmetric (symmetric→INT4 +8 bias, per-head)
    # ==================================================================

    @staticmethod
    def _quantize_k_int8(
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize Key from FP16 to INT8 (per-head symmetric).

        Args:
            k: FP16 tensor, shape ``(1, H, T, D)``.
        Returns:
            (k_int8, scale) — int8 ``(1, H, T, D)``, fp16 ``(1, H, 1, 1)``.
        """
        amax = k.abs().amax(dim=(0, 2, 3), keepdim=True).clamp(min=1e-8)
        scale = (amax / 127.0).to(torch.float16)
        k_int8 = (k / scale).round().clamp(-128, 127).to(torch.int8)
        return k_int8, scale

    @staticmethod
    def _dequantize_k_int8(
        k_int8: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize Key from INT8 back to FP16."""
        return k_int8.to(torch.float16) * scale

    @staticmethod
    def _quantize_v_int4(
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize Value from FP16 to INT4 packed (per-head symmetric + bias).

        Uses symmetric INT4 range [-8, 7] with +8 bias to get [0, 15],
        then packs two uint4 values per byte.

        Args:
            v: FP16 tensor, shape ``(1, H, T, D)`` where D is even.
        Returns:
            (packed, scale, bias) —
            - packed uint8 ``(1, H, T, D//2)``
            - scale  fp16   ``(1, H, 1, 1)``
            - bias   fp16   ``(1, H, 1, 1)``  (the +8 offset in real scale)
        """
        head_dim = v.shape[-1]
        assert head_dim % 2 == 0, "head_dim must be even for INT4 packing"

        amax = v.abs().amax(dim=(0, 2, 3), keepdim=True).clamp(min=1e-8)
        # Scale maps ±7*scale to INT4 range [-8, 7]
        scale = (amax / 7.0).to(torch.float16)
        q = (v / scale).round().clamp(-8, 7).to(torch.int8)
        q_biased = (q + 8).to(torch.uint8)  # now in [0, 15]

        # Pack two uint4 per byte: [..., 2i] = low nibble, [..., 2i+1] = high nibble
        *rest, d = q_biased.shape
        d2 = d // 2
        q_paired = q_biased.view(*rest, d2, 2)
        packed = q_paired[..., 0] | (q_paired[..., 1] << 4)

        # Bias is the +8 offset expressed in original value space
        bias = (8.0 * scale).to(torch.float16)
        return packed, scale, bias

    @staticmethod
    def _dequantize_v_int4(
        packed: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize Value from INT4 packed back to FP16.

        Args:
            packed: uint8 ``(1, H, T, D//2)``.
            scale:  fp16  ``(1, H, 1, 1)``.
            bias:   fp16  ``(1, H, 1, 1)``.
        Returns:
            fp16 ``(1, H, T, D)``.
        """
        low = (packed & 0x0F).to(torch.float16)
        high = ((packed >> 4) & 0x0F).to(torch.float16)
        v0 = low * scale - bias
        v1 = high * scale - bias
        # Interleave: [v0[0], v1[0], v0[1], v1[1], ...]
        return torch.stack([v0, v1], dim=-1).flatten(-2)

    # ==================================================================
    # Head/tail protection helpers
    # ==================================================================

    @property
    def _protected_head_tail(self) -> int:
        return self._PROTECTED_N

    @staticmethod
    def _needs_protected_storage(seq_len: int, protect_n: int) -> bool:
        """True if seq_len is short enough that all positions are protected."""
        return seq_len <= 2 * protect_n

    # ==================================================================
    # KV cache storage / retrieval (with asymmetric quantization)
    # ==================================================================

    def store_kv(
        self,
        block_id: int,
        layer_kv_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Store per-layer KV tensors for a block.

        *layer_kv_pairs* is ``[(k_0, v_0), (k_1, v_1), ...]`` — one per
        decoder layer.  Tensors should already be on CUDA.

        If seq_len > 2 * PROTECTED_N, the middle body is quantized
        (Key->INT8, Value->INT4 packed) while the first and last
        PROTECTED_N positions stay FP16.
        """
        if not layer_kv_pairs:
            self._block_kv_cache.pop(block_id, None)
            return

        protect_n = self._PROTECTED_N
        k0 = layer_kv_pairs[0][0]
        seq_len = k0.shape[2]

        if self._needs_protected_storage(seq_len, protect_n):
            # ── FP16 mode (too short to split) ──
            stored = {"mode": "fp16", "kv": layer_kv_pairs}
        else:
            # ── Mixed mode: head FP16 + body quantized + tail FP16 ──
            layers_stored = []
            for k, v in layer_kv_pairs:
                k_head = k[:, :, :protect_n, :].contiguous()
                v_head = v[:, :, :protect_n, :].contiguous()
                k_tail = k[:, :, -protect_n:, :].contiguous()
                v_tail = v[:, :, -protect_n:, :].contiguous()

                k_body = k[:, :, protect_n:-protect_n, :].contiguous()
                v_body = v[:, :, protect_n:-protect_n, :].contiguous()

                k_body_q, k_scale = self._quantize_k_int8(k_body)
                v_packed, v_scale, v_bias = self._quantize_v_int4(v_body)

                layers_stored.append({
                    "k_head": k_head,
                    "v_head": v_head,
                    "k_body_q": k_body_q,
                    "k_body_scale": k_scale,
                    "v_body_packed": v_packed,
                    "v_body_scale": v_scale,
                    "v_body_bias": v_bias,
                    "k_tail": k_tail,
                    "v_tail": v_tail,
                })

            stored = {"mode": "mixed", "seq_len": seq_len, "layers": layers_stored}

        self._block_kv_cache[block_id] = stored
        block = self.allocated_blocks.get(block_id)
        if block is not None:
            block.kv_tensor = stored

    def load_kv(
        self, block_id: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        """Retrieve stored per-layer KV tensors for a block, or None.

        Quantized body is transparently dequantized back to FP16 on load.
        """
        stored = self._block_kv_cache.get(block_id)
        if stored is None:
            return None

        if stored["mode"] == "fp16":
            return stored["kv"]

        # Mixed mode: dequantize body, concat head + body + tail
        result = []
        for layer in stored["layers"]:
            k_body = self._dequantize_k_int8(layer["k_body_q"], layer["k_body_scale"])
            v_body = self._dequantize_v_int4(
                layer["v_body_packed"], layer["v_body_scale"], layer["v_body_bias"]
            )
            k_out = torch.cat([layer["k_head"], k_body, layer["k_tail"]], dim=2)
            v_out = torch.cat([layer["v_head"], v_body, layer["v_tail"]], dim=2)
            result.append((k_out, v_out))

        return result

    def load_kv_stats(
        self, block_id: int
    ) -> dict:
        """Diagnostic: return storage mode, seq_len, and byte savings for a block.

        Returns
        -------
        dict with keys:
          mode, seq_len, fp16_bytes, stored_bytes, saving_ratio
        or empty dict if block not found.
        """
        stored = self._block_kv_cache.get(block_id)
        if stored is None:
            return {}

        if stored["mode"] == "fp16":
            kv = stored["kv"]
            num_positions = 0
            element_count = 0
            if kv:
                num_positions = kv[0][0].shape[2]
                num_heads = kv[0][0].shape[1]
                head_dim = kv[0][0].shape[3]
                num_layers = len(kv)
                element_count = num_layers * 2 * num_positions * num_heads * head_dim
            return {
                "mode": "fp16",
                "seq_len": num_positions,
                "fp16_bytes": element_count * 2,
                "stored_bytes": element_count * 2,
                "saving_ratio": 1.0,
            }

        # Mixed mode
        protect_n = self._PROTECTED_N
        seq_len = stored["seq_len"]
        num_layers = len(stored["layers"])
        if not stored["layers"]:
            return {"mode": "mixed", "seq_len": seq_len, "fp16_bytes": 0, "stored_bytes": 0}

        head_dim = stored["layers"][0]["k_head"].shape[3]
        num_heads = stored["layers"][0]["k_head"].shape[1]
        body_len = seq_len - 2 * protect_n

        # FP16 baseline: 2 bytes per element, K+V = 2 tensors
        fp16_elements = num_layers * 2 * seq_len * num_heads * head_dim
        fp16_bytes = fp16_elements * 2

        # Stored: head+tail FP16 + body quantized
        # Head+tail: 2*protect_n positions FP16 per K and V
        # Body K: body_len * num_heads * head_dim * 1 byte (INT8) + scale (num_heads * 2 bytes)
        # Body V: body_len * num_heads * head_dim/2 * 1 byte (packed) + scale+bias (num_heads * 2 bytes * 2)
        head_tail_elements = 2 * protect_n * num_heads * head_dim
        head_tail_bytes = head_tail_elements * 2 * num_layers * 2  # K+V
        body_k_bytes = body_len * num_heads * head_dim * num_layers  # INT8
        body_k_scale_bytes = num_heads * 2 * num_layers  # fp16 scale
        body_v_bytes = body_len * num_heads * (head_dim // 2) * num_layers  # uint4 packed
        body_v_meta_bytes = num_heads * 2 * 2 * num_layers  # fp16 scale + bias
        stored_bytes = head_tail_bytes + body_k_bytes + body_k_scale_bytes + body_v_bytes + body_v_meta_bytes

        return {
            "mode": "mixed",
            "seq_len": seq_len,
            "fp16_bytes": fp16_bytes,
            "stored_bytes": stored_bytes,
            "saving_ratio": stored_bytes / max(fp16_bytes, 1),
        }

    # ------------------------------------------------------------------
    # Free / reference-count management
    # ------------------------------------------------------------------

    def free_block(self, block_id: int) -> None:
        """
        Decrease the reference count of a block.

        Pinned blocks are NOT freed (call ``unpin_block`` first).

        [Bug 2] Removes block_id from radix_index sets rather than deleting
        straight dict entries.
        """
        block = self.allocated_blocks.get(block_id)
        if block is None:
            logger.warning("free_block: block %d not found", block_id)
            return

        # Pinned blocks: just decrement ref count but don't free
        if block.pinned:
            block.ref_count -= 1
            logger.debug("free_block %d (pinned): ref_count now %d", block_id, block.ref_count)
            return

        free = self.free_block_queue
        blocks = self.allocated_blocks
        radix = self.radix_index

        block.ref_count -= 1
        logger.debug("free_block %d: ref_count now %d", block_id, block.ref_count)

        if block.ref_count <= 0:
            free.append(block_id)
            blocks.pop(block_id, None)
            self._block_kv_cache.pop(block_id, None)
            with contextlib.suppress(ValueError):
                self._pinned_lru.remove(block_id)
            stale = [h for h, bids in radix.items() if block_id in bids]
            for h in stale:
                radix[h].discard(block_id)
                if not radix[h]:
                    del radix[h]
            logger.debug(
                "Block %d recycled to free queue, removed %d radix entries",
                block_id,
                len(stale),
            )

    # ------------------------------------------------------------------
    # Garbage collection
    # ------------------------------------------------------------------

    def gc(self) -> int:
        """[Bug 2] Iterates set values and prunes stale block_ids per hash."""
        removed = 0
        stale_keys: list[str] = []
        blocks = self.allocated_blocks
        radix = self.radix_index

        for h, bids in radix.items():
            stale = [bid for bid in bids if bid not in blocks]
            if stale:
                for bid in stale:
                    bids.discard(bid)
                    removed += 1
                if not bids:
                    stale_keys.append(h)

        for key in stale_keys:
            del radix[key]

        if removed:
            logger.info("GC removed %d stale radix entries", removed)
        return removed

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def used_blocks(self) -> int:
        return len(self.allocated_blocks)

    @property
    def free_blocks(self) -> int:
        return len(self.free_block_queue)

    @property
    def pinned_count(self) -> int:
        return len(self._pinned_lru)

    def stats(self) -> dict[str, int]:
        return {
            "total_blocks": self.total_blocks,
            "used_blocks": self.used_blocks,
            "free_blocks": self.free_blocks,
            "radix_entries": len(self.radix_index),
            "pinned_blocks": self.pinned_count,
        }
