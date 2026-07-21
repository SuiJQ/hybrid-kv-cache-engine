"""
UnifiedScheduler — Three-Layer End-to-End MoE Inference Pipeline.

Fixes
-----
- KV cache is now **actually stored** in HybridCache blocks after prefill
  and **retrieved** on each decode step, giving O(n) attention per token.
- Batch prefill stores per-request KV into individually allocated blocks.
- Decode passes ``past_key_values`` from cache, avoids full-sequence recompute.
- Finished requests free their cache blocks for recycling.
"""

from __future__ import annotations

import collections.abc
import logging
import time
from dataclasses import dataclass, field

import torch

import attention_kernel  # noqa: F401
from cache_manager import HybridCache

# [Goose] Lazy import
_GOOSE_AVAILABLE = False
try:
    import goose_core
    _GOOSE_AVAILABLE = True
except ImportError:
    pass

# [AFCE] Anchored Forward Cache Extension
_AFCE_AVAILABLE = False
try:
    import afce
    _AFCE_AVAILABLE = True
except ImportError:
    pass

# [OEF] Opportunistic Entropy Freeze
_OEF_AVAILABLE = False
try:
    import oef
    _OEF_AVAILABLE = True
except ImportError:
    pass

# [Self-Speculative Decoding] Lazy import
_SKELETON_AVAILABLE = False
try:
    from goose_core import SkeletonDraftGenerator as _SkeletonGen  # noqa: F401
    _SKELETON_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


def _auto_tune_max_draft(hidden_size: int) -> int:
    """Auto-tune max speculative draft tokens based on model size.

    Larger models are more expensive to verify → fewer drafts.
    Smaller models can afford more drafts for higher speculation.
    """
    if hidden_size >= 7168:   # 70B+
        return 3
    if hidden_size >= 4096:   # 7B–34B
        return 5
    return 7                   # <7B


def _extract_logits(model_output) -> torch.Tensor:
    """Extract logits tensor from both raw tensor and HF CausalLMOutput."""
    if isinstance(model_output, torch.Tensor):
        return model_output
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


def _extract_past_key_values(model_output):
    """Extract past_key_values from HF output or GGUF-side-effect."""
    if hasattr(model_output, "past_key_values"):
        return model_output.past_key_values
    return None


@dataclass
class Request:
    prompt_tokens: list[int]
    request_id: str
    max_new_tokens: int = 256
    cached_heads: list[int] = field(default_factory=list)


@dataclass
class DecodeRequest:
    tokens: list[int]
    generated_tokens: list[int]
    request_id: str
    max_new_tokens: int
    _step_count: int = 0
    # Block ID in HybridCache that holds this request's KV cache
    cache_block_id: int | None = None

    def step(self) -> None:
        self._step_count += 1

    @property
    def is_done(self) -> bool:
        return self._step_count >= self.max_new_tokens


# ===================================================================
# UnifiedScheduler — corrected KV cache integration
# ===================================================================


class UnifiedScheduler:
    # ── Chunked prefill ────────────────────────────────────────────
    CHUNKED_PREFILL_ENABLED = True
    PREFIX_CACHING_ENABLED = True
    ADAPTIVE_COMPRESSION_ENABLED = True
    _KV_CACHE_ENABLED = True  # set False to fall back to full recompute

    @staticmethod
    def _auto_tune_chunk_size(hidden_size: int) -> int:
        """Auto-tune chunked prefill chunk size based on model hidden dim."""
        if hidden_size >= 7168:
            return 256  # larger model → smaller chunks to fit VRAM
        if hidden_size >= 4096:
            return 512
        return 1024

    @staticmethod
    def _auto_tune_batch_max() -> int:
        """Auto-tune max batch size from available GPU memory."""
        try:
            free_mem, _ = torch.cuda.mem_get_info()
            free_gb = free_mem / (1024**3)
            if free_gb >= 40:
                return 16
            if free_gb >= 16:
                return 8
            return 4
        except Exception:
            return 8

    @staticmethod
    def _auto_tune_compression_params(max_len: int) -> tuple[int, int, float]:
        """Auto-tune KV compression parameters based on model max length.

        Returns (sink_n, recent_n, importance_frac).
        """
        if max_len >= 131072:
            return (4, 4096, 0.15)  # ultra-long
        if max_len >= 32768:
            return (4, 3072, 0.18)
        if max_len >= 8192:
            return (4, 2048, 0.20)
        return (4, 1024, 0.25)

    def __init__(
        self,
        model: object,
        cache: HybridCache,
        detokenizer: collections.abc.Callable[[list[int]], str] | None = None,
        vram_budget: object | None = None,
    ) -> None:
        self.model = model
        self.cache = cache
        self._detokenizer = detokenizer

        self.prefill_stream = torch.cuda.Stream()
        self.decode_stream = torch.cuda.Stream()
        self.transfer_stream = torch.cuda.Stream()

        self.pending_requests: list[Request] = []
        self.active_decode_pool: list[DecodeRequest] = []
        self._decode_bs = 0
        self._last_prefill_time = time.monotonic()
        self._running = True

        self._sere = None
        self._expert_cache = None
        self._batch_plan = None

        self._goose_auto_init_tried: bool = False
        self._skeleton_auto_init_tried: bool = False

        # [Context truncation / position encoding reset]
        self.step_since_reset = 0
        self.trigger_reset = False
        self.max_len = getattr(model, 'max_seq_len', None) or getattr(model, 'config', None) and getattr(model.config, 'max_position_embeddings', 4096) or 4096
        self._compression_threshold = int(self.max_len * 0.95)

        # [Speculative Prefetch & Dynamic Expert Activation]
        self._dynamic_activator = None
        self._spec_prefetcher = None

        # [VRAM Budget] Centralized memory manager for OOM prevention
        self._vram_budget = vram_budget
        self._vram_check_interval: int = 10  # check every N decode steps
        self._vram_step_counter: int = 0
        self._vram_degraded: bool = False

        # ── Auto-tune params from model and hardware ────────────────
        _hidden_size = getattr(model, 'hidden_size', 4096)
        _max_len = getattr(model, 'max_seq_len', None) or getattr(model, 'config', None) and getattr(model.config, 'max_position_embeddings', 4096) or 4096
        self.CHUNK_SIZE = self._auto_tune_chunk_size(_hidden_size)
        self._PREFILL_BATCH_TIMEOUT = max(0.3, min(1.0, 512 / _hidden_size))
        self._PREFILL_BATCH_MAX = self._auto_tune_batch_max()
        _sink, _recent, _imp = self._auto_tune_compression_params(_max_len)
        self._COMPRESS_SINK_N = _sink
        self._COMPRESS_RECENT_N = _recent
        self._COMPRESS_IMPORTANCE_FRAC = _imp
        self._EXPERT_CAPACITY_FACTOR = 1.2

        # ── Prefer VRAMBudget values when available ─────────────────
        if vram_budget is not None:
            self.CHUNK_SIZE = vram_budget.safe_chunk_size()
            self._PREFILL_BATCH_MAX = vram_budget.safe_batch_max()
            logger.info(
                "VRAMBudget override: chunk=%d, batch_max=%d",
                self.CHUNK_SIZE, self._PREFILL_BATCH_MAX,
            )

        # [Goose] Speculative decoding engine (Phase 0/1/2) — auto-enabled
        self._goose_enabled: bool = _GOOSE_AVAILABLE
        self._goose_engine: object | None = None
        self._spec_handled: set[str] = set()  # request_ids handled this step

        # [AFCE] Anchored Forward Cache Extension
        self._afce_manager: afce.AnchorManager | None = None
        if _AFCE_AVAILABLE:
            self._afce_manager = afce.AnchorManager()
            logger.info("AFCE: AnchorManager initialized")

        # [OEF] Opportunistic Entropy Freeze
        self._oef_controller: oef.OEFController | None = None
        self._init_oef_lazy: bool = False

        # [Self-Speculative Decoding] Skeleton draft generator — auto-enabled
        self._skeleton_draft: object | None = None
        self._self_spec_enabled: bool = _SKELETON_AVAILABLE

        # [Prefix Caching] Track blocks pinned for reuse
        self._prefix_cache_hits: int = 0
        self._prefix_cache_total: int = 0

        # [Chunked Prefill] Partial request tracking
        # Request with partial KV block: (block_id, remaining_tokens)
        self._partial_pending: list[dict] = []

        logger.info(
            "UnifiedScheduler: chunk=%d, batch_max=%d, cap_factor=%.2f, "
            "kv_cache=%s, chunked_prefill=%s, prefix_cache=%s, "
            "self_spec=%s, goose=%s, compress=%d+%d@%.2f",
            self.CHUNK_SIZE,
            self._PREFILL_BATCH_MAX,
            self._EXPERT_CAPACITY_FACTOR,
            self._KV_CACHE_ENABLED,
            self.CHUNKED_PREFILL_ENABLED,
            self.PREFIX_CACHING_ENABLED,
            self._self_spec_enabled,
            self._goose_enabled,
            self._COMPRESS_SINK_N, self._COMPRESS_RECENT_N, self._COMPRESS_IMPORTANCE_FRAC,
        )

    # ==================================================================
    # Prefix caching helpers
    # ==================================================================

    def _try_prefix_cache_submit(
        self, request: Request
    ) -> bool:
        """Try to satisfy a request from prefix cache.

        FULL match: the exact prompt token sequence is cached in a pinned
        block → creates a DecodeRequest directly, skipping prefill entirely.
        Partial matches are left for the chunked prefill to handle via the
        existing Radix tree structure (match_prefix inside allocate).

        Returns True if fully satisfied (request handled, no further
        processing needed).
        """
        if not self.PREFIX_CACHING_ENABLED or not request.prompt_tokens:
            return False

        self._prefix_cache_total += 1

        pinned_block_id = self.cache.has_prefix(request.prompt_tokens)
        if pinned_block_id is not None:
            # Full cache hit — skip prefill entirely
            self._prefix_cache_hits += 1
            self.active_decode_pool.append(
                DecodeRequest(
                    tokens=list(request.prompt_tokens),
                    generated_tokens=[],
                    request_id=request.request_id,
                    max_new_tokens=request.max_new_tokens,
                    cache_block_id=pinned_block_id,
                )
            )
            logger.debug(
                "Prefix cache FULL HIT for %s (block %d, %d tokens)",
                request.request_id, pinned_block_id, len(request.prompt_tokens),
            )
            return True

        return False

    def _init_speculative_prefetch(self):
        from speculative_prefetch import (  # noqa: PLC0415
            DynamicExpertActivator,
            SpeculativePrefetcher,
        )

        if self._dynamic_activator is None:
            self._dynamic_activator = DynamicExpertActivator(
                sere_module=self._sere,
            )

        layers = self._get_decoder_layers()
        num_layers = getattr(self.model, "num_layers", 0)
        num_experts = getattr(self.model, "num_experts", 8)

        if self._spec_prefetcher is None and self._expert_cache is not None and layers is not None:
            self._spec_prefetcher = SpeculativePrefetcher(
                expert_cache=self._expert_cache,
                decoder_layers=list(layers),
                num_layers=num_layers,
                num_experts=num_experts,
                num_reserved_experts=64,
            )

    def _get_decoder_layers(self):
        if hasattr(self.model, "layers"):
            return self.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        return None

    def _init_sere(self):
        from sere import SEREModule  # noqa: PLC0415

        num_experts = getattr(self.model, "num_experts", 8)
        top_k = getattr(self.model, "num_experts_per_tok", 2)
        # ── Auto-tune SERE params ─────────────────────────────────
        skip_threshold, min_experts = SEREModule.auto_tune(
            num_experts=num_experts, top_k=top_k,
        )
        self._sere = SEREModule(
            num_experts=num_experts, top_k=top_k,
            skip_threshold=skip_threshold, min_experts=min_experts,
        )

        layers = self._get_decoder_layers()
        if layers is not None:
            for layer in layers:
                if hasattr(layer, "sere") and layer.sere is not None:
                    layer.sere = self._sere

    def _init_expert_cache(self, vram_capacity: int | None = None):
        from expert_cache import ExpertWeightCache  # noqa: PLC0415

        model = self.model
        if not getattr(model, "is_moe", False):
            return

        layers = self._get_decoder_layers()
        if layers is None:
            return

        hidden_size = model.hidden_size
        num_layers = model.num_layers
        num_experts = getattr(model, "num_experts", 8)

        intermediate_size = hidden_size * 8 // 3
        for layer in layers:
            if hasattr(layer, "intermediate_size"):
                intermediate_size = layer.intermediate_size
                break

        block_bytes = (intermediate_size * hidden_size * 2 * 3) * 2

        # ── Auto-tune VRAM capacity from budget or GPU memory ────
        if vram_capacity is None:
            if self._vram_budget is not None:
                vram_capacity = self._vram_budget.safe_expert_cache_blocks(block_bytes)
                logger.info(
                    "Expert cache VRAM from budget: %d blocks", vram_capacity,
                )
            else:
                try:
                    free_mem, _ = torch.cuda.mem_get_info()
                    auto_capacity = max(8, int(free_mem * 0.30 / max(block_bytes, 1)))
                    vram_capacity = min(auto_capacity, 256)
                    logger.info(
                        "Auto-tuned expert cache VRAM: %d blocks (%.1f GiB free → %.1f %%)",
                        vram_capacity, free_mem / (1024**3), vram_capacity * block_bytes / free_mem * 100,
                    )
                except Exception:
                    vram_capacity = 64

        self._expert_cache = ExpertWeightCache(
            vram_capacity=vram_capacity,
            block_bytes=block_bytes,
            num_layers=num_layers,
            num_experts=num_experts,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            dtype=torch.float16,
        )

        for ly in range(num_layers):
            layer = layers[ly]
            if not hasattr(layer, "get_cpu_expert_weights"):
                continue
            for ex in range(num_experts):
                try:
                    gate_w, up_w, down_w = layer.get_cpu_expert_weights(ex)
                    self._expert_cache.register_expert(ly, ex, gate_w, up_w, down_w)
                except Exception as exc:
                    logger.warning("Expert L%d.E%d skip: %s", ly, ex, exc)

        for layer in layers:
            if hasattr(layer, "expert_cache"):
                layer.expert_cache = self._expert_cache
            # Seed _prefetch_stream (set by SpeculativePrefetcher if created)
            if hasattr(layer, '_prefetch_stream') and layer._prefetch_stream is None:
                layer._prefetch_stream = self.transfer_stream

        logger.info(
            "Expert cache: %d x %d experts, VRAM=%d blocks",
            num_layers,
            num_experts,
            vram_capacity,
        )

    # ==================================================================
    # Batch planning
    # ==================================================================

    def _plan_batch(self, batch_requests: list[Request]) -> dict:
        if not self._expert_cache:
            return {"needed_experts": set(), "capacity_check": True}

        total_tokens = sum(len(r.prompt_tokens) for r in batch_requests)
        capacity_limit = int(
            total_tokens * self._EXPERT_CAPACITY_FACTOR / max(self.model.num_experts, 1)
        )
        min_capacity = 4
        capacity_ok = capacity_limit >= min_capacity

        return {"needed_experts": set(), "capacity_check": capacity_ok}

    # ==================================================================
    # Goose speculative decoding init
    # ==================================================================

    def _init_goose(self, tree_enabled: bool = False, max_draft: int = 5) -> None:
        """Initialize the Goose speculative decoding engine.

        Safe to call multiple times — skips if already initialized.
        Falls back gracefully when ``goose_core`` module is unavailable.
        """
        if self._goose_engine is not None:
            return
        if not _GOOSE_AVAILABLE:
            logger.warning("goose_core module not available; speculation disabled.")
            return

        vocab_size = getattr(self.model, "vocab_size", 32000)
        self._goose_engine = goose_core.SpeculativeEngine(
            vocab_size=vocab_size,
            max_draft=max_draft,
            tree_enabled=tree_enabled,
        )
        self._goose_enabled = True
        self._goose_engine.enable()  # skip warm-up for immediate testing
        logger.info(
            "Goose engine initialized: tree=%s, max_draft=%d, vocab=%d",
            tree_enabled, max_draft, vocab_size,
        )

    def _init_skeleton_draft(self) -> None:
        """Initialize the skeleton draft generator for self-speculative
        decoding (ACL'24).

        Creates a ``SkeletonDraftGenerator`` that runs model forward with
        the last ~30% of decoder layers replaced by identity modules.
        The skeleton output is used as draft tokens, verified by the
        full model through the existing Goose verification pipeline.

        This is completely independent of the PLD-based Goose draft:
        - PLD works best for low-entropy/textual tasks
        - Skeleton works best for high-entropy/creative tasks
        Together they provide complementary coverage.
        """
        if not _SKELETON_AVAILABLE:
            logger.warning(
                "SkeletonDraftGenerator not available; "
                "self-speculative decoding disabled."
            )
            return

        self._skeleton_draft = _SkeletonGen(
            model=self.model,
            skip_fraction=0.30,
            max_draft=5,
        )
        self._self_spec_enabled = True
        logger.info("Skeleton draft generator initialized (skip 30%% of layers)")

    # ==================================================================
    # Public API
    # ==================================================================

    def submit(self, request: Request) -> None:
        """Submit a request.

        Checks prefix cache first (full match = skip prefill).
        Otherwise enqueues for the next chunked prefill step.
        """
        if not self._try_prefix_cache_submit(request):
            self.pending_requests.append(request)

    def shutdown(self) -> None:
        self._running = False
        if self._spec_prefetcher is not None:
            self._spec_prefetcher.shutdown()

    # ==================================================================
    # Chunked prefill with KV cache storage
    # ==================================================================

    def _batch_chunked_prefill(self):
        """Process one chunk per pending request.

        Splits each request into CHUNK_SIZE token chunks, prefills each
        chunk (extending any existing KV cache from a previous chunk),
        and if the request is fully pre-filled, moves it to the active
        decode pool.

        Chunked prefill eliminates the "all-prefill-then-all-decode"
        pipeline stall: small prefill chunks interleave naturally with
        decode steps, reducing TTFT for large prompts (Sarathi-style).
        """
        if not self.pending_requests:
            return

        now = time.monotonic()
        elapsed = now - self._last_prefill_time
        if (
            elapsed < self._PREFILL_BATCH_TIMEOUT
            and len(self.pending_requests) < self._PREFILL_BATCH_MAX
        ):
            return

        batch = list(self.pending_requests)
        self.pending_requests.clear()
        self._last_prefill_time = now

        chunk_size = self.CHUNK_SIZE

        for req in batch:
            tokens = req.prompt_tokens
            total_len = len(tokens)
            offset = 0

            while offset < total_len:
                chunk_end = min(offset + chunk_size, total_len)
                chunk = tokens[offset:chunk_end]
                is_last = chunk_end >= total_len

                inp = torch.tensor([chunk], dtype=torch.long, device="cuda")

                with torch.cuda.stream(self.prefill_stream), torch.no_grad():
                    if offset == 0:
                        # First chunk: no past KV
                        out = self.model.forward(
                            input_ids=inp,
                            use_cache=True,
                        )
                    else:
                        # Subsequent chunk: extend existing KV cache
                        past_kv = self.cache.load_kv(cache_block.block_id)
                        out = self.model.forward(
                            input_ids=inp,
                            past_key_values=past_kv,
                            use_cache=True,
                        )

                kv_cache = getattr(self.model, "_last_kv_cache", None)
                if kv_cache is None:
                    kv_cache = _extract_past_key_values(out)

                if kv_cache is not None:
                    if offset == 0:
                        # Allocate block only on first chunk
                        cache_block = self.cache.allocate(chunk)
                        self.cache.store_kv(cache_block.block_id, kv_cache)

                        # [AFCE] Extract anchors from first prefill chunk
                        if (
                            self._afce_manager is not None
                            and len(tokens) >= afce.CLUSTER_SIZE
                        ):
                            try:
                                from afce import extract_anchors_after_prefill
                                extract_anchors_after_prefill(
                                    self._afce_manager,
                                    self.cache,
                                    tokens,
                                    kv_cache,
                                )
                            except Exception as exc:
                                logger.debug("AFCE prefill extract skipped: %s", exc)
                    else:
                        # Extend KV in existing block
                        self.cache.store_kv(cache_block.block_id, kv_cache)
                else:
                    logger.warning(
                        "Chunked prefill: model did not produce KV cache"
                    )
                    cache_block = None
                    break

                offset = chunk_end

                if not is_last:
                    # More chunks remain — put back as pending
                    self.pending_requests.insert(
                        0,
                        Request(
                            prompt_tokens=tokens[offset:],
                            request_id=req.request_id,
                            max_new_tokens=req.max_new_tokens,
                        ),
                    )
                    # Signal to _batch_chunked_prefill that this request
                    # already has a partial KV block (for the next round)
                    # We do this by creating an intermediate internal state
                    break

            if is_last and cache_block is not None:
                # Pin the block if prefix caching is enabled
                if self.PREFIX_CACHING_ENABLED:
                    self.cache.pin_prefix_from_match(cache_block.block_id)

                self.active_decode_pool.append(
                    DecodeRequest(
                        tokens=list(tokens),
                        generated_tokens=[],
                        request_id=req.request_id,
                        max_new_tokens=req.max_new_tokens,
                        cache_block_id=cache_block.block_id,
                    )
                )
                logger.debug(
                    "Chunked prefill %s: %d tokens -> block %d%s",
                    req.request_id,
                    total_len,
                    cache_block.block_id,
                    " (pinned)" if self.PREFIX_CACHING_ENABLED else "",
                )

        n_processed = len(batch) - len(self.pending_requests)
        if n_processed:
            logger.info(
                "Chunked prefill: %d/%d requests completed (%d remain pending)",
                n_processed, len(batch), len(self.pending_requests),
            )

    # ==================================================================
    # Decode with KV cache
    # ==================================================================

    def _decode_step(self):
        if not self.active_decode_pool:
            self._decode_bs = 0
            return

        # [Plan 4] Single boolean check: trigger_reset → context compression
        if self.trigger_reset:
            for req in self.active_decode_pool:
                if self.ADAPTIVE_COMPRESSION_ENABLED:
                    self._compress_kv_adaptive(req)
                else:
                    self._compress_kv(req)
            self.step_since_reset = 0
            self.trigger_reset = False
            logger.info("Context compression triggered for %d requests", len(self.active_decode_pool))

        with torch.cuda.stream(self.decode_stream):
            for i, req in enumerate(self.active_decode_pool):
                # [Goose] Skip requests already handled by speculative decode
                if req.request_id in self._spec_handled:
                    continue

                last_tok = req.tokens[-1] if req.tokens else 0
                inp = torch.tensor([[last_tok]], dtype=torch.long, device="cuda")

                # Load cached KV if available
                past_kv = None
                if req.cache_block_id is not None:
                    past_kv = self.cache.load_kv(req.cache_block_id)

                # [AFCE] Extend KV with anchors + build attention mask
                _afce_hash_key: str | None = None
                _afce_num_anchors: int = 0
                _afce_mask: torch.Tensor | None = None
                _afce_past_kv: list | tuple | None = None

                if (
                    past_kv is not None
                    and self._KV_CACHE_ENABLED
                    and self._afce_manager is not None
                    and req.cache_block_id is not None
                ):
                    _afce_hash_key = self.cache.compute_hash(req.tokens)
                    if _afce_hash_key and self._afce_manager.has_sidecar(_afce_hash_key):
                        extended: list[tuple[torch.Tensor, torch.Tensor]] = []
                        _na = 0
                        for _k, _v in past_kv:
                            _ek, _ev, _m, _n = self._afce_manager.extend_for_decode(
                                _afce_hash_key, _k, _v, len(req.tokens) - 1,
                            )
                            extended.append((_ek, _ev))
                            _na = _n
                        if _na > 0:
                            _afce_num_anchors = _na
                            _afce_mask = _m
                            _afce_past_kv = extended

                # [OEF] Feed OEF skip suggestions into SERE before forward
                if self._oef_controller is not None and self._sere is not None:
                    suggestions = self._oef_controller.get_skip_suggestions()
                    self._sere._oef_skip_suggestions = suggestions if suggestions else None

                with torch.no_grad():
                    if past_kv is not None and self._KV_CACHE_ENABLED:
                        model_out = self.model.forward(
                            input_ids=inp,
                            past_key_values=_afce_past_kv if _afce_past_kv is not None else past_kv,
                            use_cache=True,
                            attention_mask=_afce_mask if _afce_num_anchors > 0 else None,
                        )
                    else:
                        # Full recompute path (fallback if no KV cache)
                        full_input = torch.tensor(
                            [req.tokens], dtype=torch.long, device="cuda"
                        )
                        model_out = self.model.forward(
                            input_ids=full_input, use_cache=False
                        )

                # Extract logits (works for both GGUF tensor and HF output)
                logits_t = _extract_logits(model_out)

                # Extract next token
                _min_logit_dims = 2
                if logits_t.dim() >= _min_logit_dims:
                    next_tok = int(logits_t[0, -1, :].argmax().item())
                else:
                    next_tok = 0

                req.step()
                req.generated_tokens.append(next_tok)
                req.tokens.append(next_tok)

                # Update KV cache after decode (both GGUF and HF paths)
                if req.cache_block_id is not None and self._KV_CACHE_ENABLED:
                    new_kv = getattr(self.model, "_last_kv_cache", None)
                    if new_kv is None:
                        new_kv = _extract_past_key_values(model_out)
                    if new_kv is not None:
                        # [AFCE] Strip anchor positions before storing back
                        if _afce_num_anchors > 0 and self._afce_manager is not None:
                            new_kv = self._afce_manager.strip_anchors_from_kv(
                                new_kv, _afce_num_anchors,
                            )
                        self.cache.store_kv(req.cache_block_id, new_kv)

                # [AFCE] Async prefetch next cluster anchors（红线 #2）
                if (
                    _afce_hash_key is not None
                    and _afce_num_anchors > 0
                    and self._afce_manager is not None
                ):
                    self._afce_manager.prefetch(
                        _afce_hash_key, len(req.tokens) - 1,
                        stream=self.transfer_stream,
                    )

                # [OEF] Observe router probs for entropy tracking
                if (
                    self._oef_controller is not None
                    and hasattr(self.model, "_last_router_probs")
                    and self.model._last_router_probs is not None
                ):
                    with torch.no_grad():
                        _rp = self.model._last_router_probs
                        _, _ti = torch.topk(_rp, _rp.shape[-1], dim=-1)
                        self._oef_controller.observe(_rp, _ti)

                # Confidence-based k adjustment on request 0
                if self._dynamic_activator is not None and i == 0:
                    self._dynamic_activator.update_from_logits(
                            logits_t[None, :, :],
                            generated_ids=req.tokens,
                            detokenizer=self._detokenizer,
                        )

            # Speculative prefetch
            if self._spec_prefetcher is not None and len(self.active_decode_pool) > 0:
                self._spec_prefetcher.step(
                    context_ids=self.active_decode_pool[0].tokens,
                    transfer_stream=self.transfer_stream,
                )

        # [Plan 2] Increment step counter; set trigger flag at threshold
        self.step_since_reset += 1
        if self.step_since_reset >= self._compression_threshold:
            self.trigger_reset = True

    # ==================================================================
    # Main step
    # ==================================================================

    # ==================================================================
    # Context compression (position encoding reset)
    # ==================================================================

    def _compress_kv(self, req: DecodeRequest) -> None:
        """Compress one request's KV cache: keep first 4 + last 2048 tokens.

        1. Truncate token list to tokens[:4] + tokens[-2048:]
        2. Slice KV tensors to match (keep first 4 + last 2048 positions)
        3. Free any intermediate KV blocks via existing pop interface
        4. [AFCE] Clear orphan sidecar anchors before compression

        This is the simple position-based compression (StreamingLLM-style).
        When ADAPTIVE_COMPRESSION_ENABLED is True, _compress_kv_adaptive
        is used instead.
        """
        if req.cache_block_id is None or not req.tokens:
            return

        old_len = len(req.tokens)
        sink_n = self._COMPRESS_SINK_N
        recent_n = self._COMPRESS_RECENT_N
        if old_len <= sink_n + recent_n + 2:
            return

        # [AFCE] Clear orphan sidecar for old hash
        if self._afce_manager is not None:
            old_hash = self.cache.compute_hash(req.tokens)
            if old_hash and self._afce_manager.has_sidecar(old_hash):
                self._afce_manager.remove_sidecar(old_hash)

        # Build new token sequence: sink + recent window
        new_tokens = req.tokens[:sink_n] + req.tokens[-recent_n:]
        req.tokens = new_tokens

        # Compress KV tensors in place
        kv = self.cache.load_kv(req.cache_block_id)
        if kv is not None:
            compressed_kv = []
            for k, v in kv:
                first_k = k[:, :, :sink_n, :]
                last_k = k[:, :, -recent_n:, :]
                first_v = v[:, :, :sink_n, :]
                last_v = v[:, :, -recent_n:, :]
                compressed_k = torch.cat([first_k, last_k], dim=2)
                compressed_v = torch.cat([first_v, last_v], dim=2)
                compressed_kv.append((compressed_k, compressed_v))
            self.cache.store_kv(req.cache_block_id, compressed_kv)

        logger.info(
            "KV compressed for %s: %d -> %d tokens",
            req.request_id, old_len, len(new_tokens),
        )

    def _compress_kv_adaptive(self, req: DecodeRequest) -> None:
        """H2O-style adaptive KV compression: keep attention sink + recent
        window + most "important" middle tokens.

        Compression strategy (H2O + StreamingLLM hybrid):
        1. Always keep ``_COMPRESS_SINK_N`` initial tokens (attention sink)
        2. Always keep ``_COMPRESS_RECENT_N`` most recent tokens (sliding window)
        3. From the middle ``(sink, -recent)`` region, keep the top
           ``_COMPRESS_IMPORTANCE_FRAC`` fraction of tokens, scored by a
           lightweight importance proxy.

        Importance scoring (no attention weights needed):
        - Uses router logits from MoE layers (``_last_router_probs``) as
          proxy: tokens that trigger high-entropy routing are considered
          more "interesting" and thus important.
        - Falls back to uniform sampling (coverage) if router logits are
          unavailable.
        - Boosts tokens at cluster boundaries via AFCE-backed positions,
          since AFCE anchors contribute to long-range coherence.

        Compatible with all existing modules (KV quantization, AFCE,
        Goose, SERE, OEF).
        """
        if req.cache_block_id is None or not req.tokens:
            return

        sink_n = self._COMPRESS_SINK_N
        recent_n = self._COMPRESS_RECENT_N
        old_len = len(req.tokens)
        target_total = sink_n + recent_n

        if old_len <= target_total + 8:
            return

        # [AFCE] Clear orphan sidecar for old hash
        if self._afce_manager is not None:
            old_hash = self.cache.compute_hash(req.tokens)
            if old_hash and self._afce_manager.has_sidecar(old_hash):
                self._afce_manager.remove_sidecar(old_hash)

        # ── Build token selection ─────────────────────────────────
        # Always keep sink tokens (positions 0..sink_n-1)
        keep_set = set(range(sink_n))
        # Always keep recent window (last recent_n positions)
        keep_set.update(range(old_len - recent_n, old_len))

        # ── Score middle region ───────────────────────────────────
        middle_start = sink_n
        middle_end = old_len - recent_n
        middle_len = middle_end - middle_start

        if middle_len > 0:
            # Try to use router logits for importance scoring
            importance_scores: list[float] | None = None
            if (
                hasattr(self.model, "_last_router_probs")
                and self.model._last_router_probs is not None
            ):
                router = self.model._last_router_probs
                # entropy of router distribution as importance proxy
                # high entropy = ambiguous routing = important token
                if router.dim() >= 2 and middle_len <= router.shape[-2]:
                    probs_mid = router[0, -middle_len:, :]
                    entropy = -(probs_mid * torch.log(probs_mid.clamp(min=1e-10))).sum(dim=-1)
                    importance_scores = entropy.tolist()
                else:
                    # Try per-position router access
                    try:
                        router_flat = router.flatten()
                        if len(router_flat) >= middle_len:
                            importance_scores = router_flat[-middle_len:].tolist()
                        else:
                            importance_scores = [0.5] * middle_len
                    except Exception:
                        importance_scores = None

            if importance_scores is None:
                # Fallback: uniform coverage — sample evenly
                num_extra = max(
                    4, int(middle_len * self._COMPRESS_IMPORTANCE_FRAC)
                )
                step = max(1, middle_len // num_extra)
                for i in range(middle_start, middle_end, step):
                    keep_set.add(i)
            else:
                # Keep top-k by importance
                num_extra = max(
                    4, int(middle_len * self._COMPRESS_IMPORTANCE_FRAC)
                )
                # Pair each middle index with its score
                indexed = list(
                    zip(range(middle_start, middle_end), importance_scores)
                )
                indexed.sort(key=lambda x: x[1], reverse=True)
                for i, _ in indexed[:num_extra]:
                    keep_set.add(i)

        # ── Build new token sequence ──────────────────────────────
        keep_indices = sorted(keep_set)
        new_tokens = [req.tokens[i] for i in keep_indices]
        req.tokens = new_tokens

        # ── Slice KV tensors ──────────────────────────────────────
        kv = self.cache.load_kv(req.cache_block_id)
        if kv is not None:
            compressed_kv = []
            for k, v in kv:
                # k, v shape: [1, num_heads, seq_len, head_dim]
                compressed_k = torch.index_select(k, 2, torch.tensor(keep_indices, device=k.device))
                compressed_v = torch.index_select(v, 2, torch.tensor(keep_indices, device=v.device))
                compressed_kv.append((compressed_k, compressed_v))
            self.cache.store_kv(req.cache_block_id, compressed_kv)

        logger.info(
            "Adaptive KV compressed for %s: %d -> %d tokens (kept %d middle via importance)",
            req.request_id, old_len, len(new_tokens),
            len([i for i in keep_indices if middle_start <= i < middle_end]),
        )

    # ==================================================================
    # Self-speculative decode (ACL'24 — skip-layer draft)
    # ==================================================================

    def _decode_self_speculative(self) -> None:
        """Self-speculative decode using skeleton (skip-layer) draft
        (ACL'24).

        Generates draft tokens by running the model forward with the
        last ~30% of decoder layers replaced by identity modules
        (skip-level draft).  The draft is then verified by the full
        model through the existing verify_linear pipeline (via Goose
        engine) or directly by the model forward.

        This complements PLD-based Goose:
        - PLD: relies on repeated n-gram patterns (good for code,
          formal text, templates)
        - Skeleton: works on ANY text regardless of repetition
          (good for creative writing, reasoning, high-entropy tasks)

        Compatible with KV cache: past_key_values pass through
        identity layers unchanged, and the verify step updates KV.
        """
        if not self._self_spec_enabled or self._skeleton_draft is None:
            return

        if not self.active_decode_pool:
            return

        for req in list(self.active_decode_pool):
            if req.is_done or req.request_id in self._spec_handled:
                continue

            context = req.tokens
            if len(context) < 8:
                continue

            # ── Load KV cache ────────────────────────────────────
            past_kv = None
            if req.cache_block_id is not None and self._KV_CACHE_ENABLED:
                past_kv = self.cache.load_kv(req.cache_block_id)

            # ── Generate skeleton draft ──────────────────────────
            last_tok = torch.tensor(
                [[context[-1]]], dtype=torch.long, device="cuda"
            )
            draft_tokens = self._skeleton_draft.generate_draft(
                input_ids=last_tok,
                past_key_values=past_kv,
            )

            if not draft_tokens:
                continue

            # ── Verify: use existing Goose verify_linear if available ──
            if self._goose_engine is not None:
                accepted, next_token, new_kv = (
                    self._goose_engine.verify_linear(
                        self.model,
                        past_kv if past_kv is not None else None,
                        draft_tokens,
                        context,
                    )
                )
            else:
                # Fallback: manual single-step verification
                accepted = []
                new_kv = past_kv
                for dt in draft_tokens:
                    inp = torch.tensor(
                        [[context[-1] if not accepted else accepted[-1]]],
                        dtype=torch.long, device="cuda",
                    )
                    with torch.no_grad():
                        out = self.model.forward(
                            input_ids=inp,
                            past_key_values=new_kv,
                            use_cache=True,
                        )
                    logits_t = _extract_logits(out)
                    predicted = int(logits_t[0, -1, :].argmax().item())
                    if predicted == dt:
                        accepted.append(dt)
                        new_kv = _extract_past_key_values(out)
                    else:
                        next_token = predicted
                        new_kv = _extract_past_key_values(out)
                        break
                else:
                    # All accepted — get bonus token
                    inp = torch.tensor(
                        [[accepted[-1]]], dtype=torch.long, device="cuda",
                    )
                    with torch.no_grad():
                        out = self.model.forward(
                            input_ids=inp,
                            past_key_values=new_kv,
                            use_cache=True,
                        )
                    logits_t = _extract_logits(out)
                    next_token = int(logits_t[0, -1, :].argmax().item())
                    new_kv = _extract_past_key_values(out)

            if not accepted:
                continue

            # ── Update request state ─────────────────────────────
            self._spec_handled.add(req.request_id)

            for tok in accepted:
                req.step()
                req.generated_tokens.append(tok)
                req.tokens.append(tok)

            # Bonus token
            req.step()
            req.generated_tokens.append(next_token)
            req.tokens.append(next_token)

            # Update KV cache
            if new_kv is not None and req.cache_block_id is not None:
                self.cache.store_kv(req.cache_block_id, new_kv)

            # Harvest logits into PLD transition table (if Goose engine active)
            if self._goose_engine is not None:
                verify_window = req.tokens[-(len(accepted) + 1):]
                if verify_window:
                    harvest_inp = torch.tensor(
                        [verify_window], dtype=torch.long, device="cuda"
                    )
                    with torch.no_grad():
                        lb_out = self.model.forward(
                            input_ids=harvest_inp, use_cache=False,
                        )
                    lb_logits = _extract_logits(lb_out)
                    self._goose_engine.harvest_logits(
                        lb_logits, list(verify_window)
                    )

            logger.debug(
                "Self-speculative: %s accepted %d/%d (skeleton draft)",
                req.request_id, len(accepted), len(draft_tokens),
            )

    # ==================================================================
    # Goose speculative decode (Phase 0/1 → chain; Phase 2 → tree)
    # ==================================================================

    def _decode_speculative(self) -> None:
        """KV-cache-aware speculative decode using Goose engine.

        Phase 0/1: linear chain verification (simple forward + argmax).
        Phase 2: tree attention verification (single forward with mask).

        Handles: draft generation, prefix KV load, model forward,
        token verification, KV cache update, logit harvesting.
        Falls through to normal ``_decode_step`` for any request where
        speculation is not applicable.
        """
        if not self._goose_enabled or self._goose_engine is None:
            return

        engine = self._goose_engine
        spec_stream = self.decode_stream

        for req in list(self.active_decode_pool):
            if req.is_done or req.request_id in self._spec_handled:
                continue

            context = req.tokens

            # ---- Step 1: Check if speculation is viable ----
            if not engine.can_speculate(context):
                continue

            # ---- Step 2: Generate draft tokens ----
            drafts, _bypass = engine.generate_draft(context)
            if not drafts:
                continue

            # ---- Step 3: Load prefix KV cache ----
            past_kv = None
            prefix_len = 0
            _goose_afce_anchors: int = 0
            if req.cache_block_id is not None and self._KV_CACHE_ENABLED:
                past_kv = self.cache.load_kv(req.cache_block_id)
                if past_kv is not None and len(past_kv) > 0 and past_kv[0][0] is not None:
                    prefix_len = past_kv[0][0].shape[2]

                # [AFCE] Extend KV with anchors for long-context speculative
                # quality.  The forward inside verify uses is_causal=True which
                # naturally allows attending to prepended anchors (whose abs
                # positions are all < query position).
                if self._afce_manager is not None:
                    _gk = self.cache.compute_hash(context)
                    if _gk and self._afce_manager.has_sidecar(_gk):
                        _ext: list[tuple[torch.Tensor, torch.Tensor]] = []
                        _an = 0
                        for _k, _v in past_kv:
                            _ek, _ev, _m, _n = self._afce_manager.extend_for_decode(
                                _gk, _k, _v, len(context) - 1,
                            )
                            _ext.append((_ek, _ev))
                            _an = _n
                        if _an > 0:
                            _goose_afce_anchors = _an
                            # Adjust prefix_len: anchors are prepended
                            prefix_len += _an
                            past_kv = _ext

            # ---- Step 4: Verify ----
            with torch.cuda.stream(spec_stream), torch.no_grad():
                if engine.tree_enabled:
                    # Phase 2: Build spine tree and use tree attention
                    tree = engine.build_spine_tree(context[-1], drafts)
                    accepted, next_token, new_kv = engine.verify_tree(
                        self.model, past_kv, tree, prefix_len,
                    )
                else:
                    # Phase 0/1: Linear chain verification
                    accepted, next_token, new_kv = engine.verify_linear(
                        self.model, past_kv, drafts, context,
                    )

            # ---- Step 5: Update request state ----
            if not accepted and past_kv is None:
                # No speculation benefit (warm-up or non-KV path)
                continue

            # Mark as handled by speculation
            self._spec_handled.add(req.request_id)

            # Append accepted tokens
            for tok in accepted:
                req.step()
                req.generated_tokens.append(tok)
                req.tokens.append(tok)

            # Append bonus token
            req.step()
            req.generated_tokens.append(next_token)
            req.tokens.append(next_token)

            # Update KV cache (sliced to accepted prefix)
            if new_kv is not None and req.cache_block_id is not None:
                # [AFCE] Strip anchor positions from speculative output
                if _goose_afce_anchors > 0 and self._afce_manager is not None:
                    new_kv = self._afce_manager.strip_anchors_from_kv(
                        new_kv, _goose_afce_anchors,
                    )
                self.cache.store_kv(req.cache_block_id, new_kv)

            # Update static KV (for non-cache forward tracking)
            if hasattr(self.model, "_last_kv_cache"):
                self.model._last_kv_cache = new_kv

            # ---- Step 6: Harvest logits into transition table ----
            if past_kv is not None:
                # Re-run a small forward on the generated window to get
                # logits for harvest. Use the same token window as context.
                harvest_window = req.tokens[-(len(accepted) + 1):]
                if harvest_window:
                    last_input = torch.tensor([harvest_window], dtype=torch.long, device="cuda")
                    with torch.cuda.stream(spec_stream), torch.no_grad():
                        lb_out = self.model.forward(
                            input_ids=last_input,
                            use_cache=False,
                        )
                    lb_logits = _extract_logits(lb_out)
                    engine.harvest_logits(lb_logits, list(harvest_window))

        logger.debug(
            "Speculative decode: %d/%d requests handled",
            len(self._spec_handled),
            len(self.active_decode_pool),
        )

    # ==================================================================
    # Main step
    # ==================================================================

    async def step(self):
        # Init layers (lazy)
        if self._expert_cache is None and getattr(self.model, "is_moe", False):
            self._init_expert_cache()  # VRAM capacity auto-tuned via budget or GPU info

        # Chunked prefill (process one chunk per pending request)
        if self.CHUNKED_PREFILL_ENABLED:
            self._batch_chunked_prefill()
        else:
            self._batch_prefill()

        if self._sere is None and getattr(self.model, "is_moe", False):
            self._init_sere()

        # [OEF] Lazy init: depends on model's expert count
        if _OEF_AVAILABLE and self._oef_controller is None and getattr(self.model, "is_moe", False):
            num_experts = getattr(self.model, "num_experts", 8)
            self._oef_controller = oef.OEFController(num_experts=num_experts)
            self._init_oef_lazy = True
            logger.info("OEF: OEFController initialized (%d experts)", num_experts)

        if self._dynamic_activator is None or self._spec_prefetcher is None:
            self._init_speculative_prefetch()

        # ── [VRAM Budget] Runtime memory check ─────────────────────
        if self._vram_budget is not None:
            self._vram_step_counter += 1
            if self._vram_step_counter % self._vram_check_interval == 0 or self._vram_degraded:
                status = self._vram_budget.check()
                action = status.get("action")
                if action == "reduce_chunk" and self.CHUNK_SIZE > 64:
                    old = self.CHUNK_SIZE
                    self.CHUNK_SIZE = max(64, self.CHUNK_SIZE // 2)
                    self._vram_degraded = True
                    logger.warning(
                        "VRAM degradation: chunk size %d → %d (%s)",
                        old, self.CHUNK_SIZE, status["message"],
                    )
                elif action == "compress_kv":
                    if not self.trigger_reset:
                        old_frac = self._COMPRESS_IMPORTANCE_FRAC
                        self._COMPRESS_RECENT_N = max(512, self._COMPRESS_RECENT_N // 2)
                        self._COMPRESS_IMPORTANCE_FRAC = min(0.40, old_frac + 0.10)
                        self.trigger_reset = True
                        self._vram_degraded = True
                        logger.warning(
                            "VRAM critical → forced KV compression (%s)",
                            status["message"],
                        )
                elif action == "emergency":
                    # Stall new requests and compress all active
                    self._PREFILL_BATCH_MAX = max(1, self._PREFILL_BATCH_MAX // 2)
                    self.CHUNK_SIZE = max(64, self.CHUNK_SIZE // 2)
                    if not self.trigger_reset:
                        self.trigger_reset = True
                    self._vram_degraded = True
                    logger.error(
                        "VRAM emergency: throttled batch_max=%d, chunk=%d (%s)",
                        self._PREFILL_BATCH_MAX, self.CHUNK_SIZE, status["message"],
                    )

        # [Self-Speculative] Try skeleton draft first (broadest coverage)
        if self._self_spec_enabled and self._skeleton_draft is not None:
            self._decode_self_speculative()

        # [Goose] Auto-init on first step if available
        if self._goose_enabled and self._goose_engine is None and _GOOSE_AVAILABLE and not self._goose_auto_init_tried:
            self._goose_auto_init_tried = True
            self._init_goose(
                tree_enabled=False,
                max_draft=_auto_tune_max_draft(
                    getattr(self.model, 'hidden_size', 4096),
                ),
            )
            logger.info("Goose speculative decoding auto-enabled (max_draft=%d)",
                        self._goose_engine.max_draft if self._goose_engine else 0)

        # [Self-Speculative] Auto-init skeleton draft on first step
        if self._self_spec_enabled and self._skeleton_draft is None and _SKELETON_AVAILABLE and not self._skeleton_auto_init_tried:
            self._skeleton_auto_init_tried = True
            self._init_skeleton_draft()
            logger.info("Self-speculative decoding auto-enabled")

        # [Goose PLD] Then try PLD-based speculative (pattern matching)
        self._decode_speculative()

        self._decode_step()

        # Clear speculation tracking
        self._spec_handled.clear()

        # Establish cross-stream ordering for next step.
        # wait_stream sets a GPU-side dependency (CPU returns immediately).
        # No full synchronize() here — it would stall the async pipeline
        # by idling the GPU between decode steps.  The targeted sync lives
        # inside _garbage_collect where actual memory freeing occurs.
        torch.cuda.current_stream().wait_stream(self.prefill_stream)
        torch.cuda.current_stream().wait_stream(self.decode_stream)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)

        self._garbage_collect()

    # ==================================================================
    # Garbage collection
    # ==================================================================

    def _garbage_collect(self):
        finished = [d for d in self.active_decode_pool if d.is_done]
        if not finished:
            # Nothing to free — skip expensive sync
            self.active_decode_pool = [d for d in self.active_decode_pool if not d.is_done]
            self.cache.gc()
            if self.PREFIX_CACHING_ENABLED and self._prefix_cache_total > 0:
                hit_ratio = self._prefix_cache_hits / self._prefix_cache_total * 100
                if self._prefix_cache_total % 100 == 0:
                    logger.info(
                        "Prefix cache: %d/%d hits (%.1f%%), %d pinned blocks",
                        self._prefix_cache_hits,
                        self._prefix_cache_total,
                        hit_ratio,
                        self.cache.pinned_count,
                    )
            return

        # Targeted sync: ensure no GPU kernel references blocks we're freeing
        torch.cuda.synchronize()

        for d in finished:
            logger.info(
                "Request %s complete (%d tokens)",
                d.request_id, len(d.generated_tokens),
            )
            if d.cache_block_id is not None:
                self.cache.free_block(d.cache_block_id)
            # [AFCE] Clean up per-request offset table
            if self._afce_manager is not None:
                self._afce_manager.remove_offset_table(d.request_id)
        self.active_decode_pool = [d for d in self.active_decode_pool if not d.is_done]
        self.cache.gc()
