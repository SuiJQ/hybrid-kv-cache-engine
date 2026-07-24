"""
UnifiedScheduler — Dense Transformer Inference Pipeline.

极致性价比特性（全部自动开启）：
  • Chunked Prefill (Sarathi) + KV Cache (PagedAttention + RadixAttention)
  • Goose 推测解码 (PLD + 树注意力)
  • Self-Spec 骨架推测 (ACL'24, 跳层草稿)
  • 投机解码 — 小模型起草 + 大模型一次验证
  • 自适应 KV 压缩 (StreamingLLM + H2O)
  • 双 CUDA 流管线 (prefill | decode | transfer)
  • VRAM 预算运行时监控 + 自动降级
"""

from __future__ import annotations

import collections.abc
import copy
import logging
import time
from dataclasses import dataclass, field

import torch

import attention_kernel  # noqa: F401
from cache_manager import HybridCache

_GOOSE_AVAILABLE = False
try:
    import goose_core
    _GOOSE_AVAILABLE = True
except ImportError:
    pass

_SKELETON_AVAILABLE = False
try:
    from goose_core import SkeletonDraftGenerator as _SkeletonGen
    _SKELETON_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


def _extract_logits(o) -> torch.Tensor:
    if isinstance(o, torch.Tensor):
        return o
    if hasattr(o, "logits"):
        return o.logits
    if isinstance(o, (tuple, list)):
        return o[0]
    return o


def _extract_pkv(o):
    if hasattr(o, "past_key_values"):
        return o.past_key_values
    if isinstance(o, (tuple, list)) and len(o) > 1:
        return o[1]
    return None


@dataclass
class Request:
    prompt_tokens: list[int]
    request_id: str
    max_new_tokens: int = 256


@dataclass
class DecodeRequest:
    tokens: list[int]
    generated_tokens: list[int]
    request_id: str
    max_new_tokens: int
    _step_count: int = 0
    cache_block_id: int | None = None

    def step(self):
        self._step_count += 1

    @property
    def is_done(self) -> bool:
        return self._step_count >= self.max_new_tokens


# ===================================================================
# UnifiedScheduler
# ===================================================================

class UnifiedScheduler:
    CHUNKED_PREFILL_ENABLED = True
    PREFIX_CACHING_ENABLED = True
    ADAPTIVE_COMPRESSION_ENABLED = True
    _KV_CACHE_ENABLED = True

    # ── 投机解码参数 ────────────────────────────────────────────
    SPECULATIVE_DRAFT_N = 5       # 草稿 token 数
    SPECULATIVE_MIN_LEN = 8       # 至少多少 token 才开始投机
    SPECULATIVE_VERIFY_TEMP = 0.0  # 验证时 greedy

    def __init__(self, model, cache, detokenizer=None, vram_budget=None,
                 draft_model=None, draft_tokenizer=None):
        self.model = model
        self.cache = cache
        self._detokenizer = detokenizer
        self._vram_budget = vram_budget
        self._draft_model = draft_model

        self.prefill_stream = torch.cuda.Stream()
        self.decode_stream = torch.cuda.Stream()
        self.transfer_stream = torch.cuda.Stream()

        self.pending_requests: list[Request] = []
        self.active_decode_pool: list[DecodeRequest] = []
        self._last_prefill_time = time.monotonic()
        self._running = True

        # VRAM 监控
        self._vram_check_int = 10
        self._vram_step = 0
        self._vram_degraded = False

        # 自动调参
        hs = getattr(model, 'hidden_size', 4096)
        ml = getattr(getattr(model, 'config', None), 'max_position_embeddings', 8192) or 8192
        self.CHUNK_SIZE = self._auto_chunk(hs)
        self._PREFILL_BATCH_TIMEOUT = max(0.3, min(1.0, 512 / max(hs, 1)))
        self._PREFILL_BATCH_MAX = self._auto_batch()
        s, r, imp = self._auto_compress(ml)
        self._COMPRESS_SINK_N = s
        self._COMPRESS_RECENT_N = r
        self._COMPRESS_IMPORTANCE_FRAC = imp

        if vram_budget is not None:
            self.CHUNK_SIZE = vram_budget.safe_chunk_size()
            self._PREFILL_BATCH_MAX = vram_budget.safe_batch_max()

        # Goose 推测
        self._goose_enabled = _GOOSE_AVAILABLE
        self._goose_engine = None
        self._spec_handled: set[str] = set()
        self._goose_init_tried = False

        # Self-Spec
        self._skeleton_draft = None
        self._self_spec_enabled = _SKELETON_AVAILABLE
        self._skel_init_tried = False

        # 缓存统计
        self._prefix_hits = 0
        self._prefix_total = 0
        self._spec_draft_hits = 0   # 投机解码命中统计
        self._spec_draft_total = 0

        # Context 压缩
        self.step_since_reset = 0
        self.trigger_reset = False
        self.max_len = ml
        self._comp_threshold = int(ml * 0.95)

        logger.info(
            "Scheduler: chunk=%d, batch_max=%d, goose=%s, self_spec=%s, "
            "compress=%d+%d, draft_model=%s",
            self.CHUNK_SIZE, self._PREFILL_BATCH_MAX,
            self._goose_enabled, self._self_spec_enabled,
            self._COMPRESS_SINK_N, self._COMPRESS_RECENT_N,
            "yes" if draft_model else "no",
        )

    @staticmethod
    def _auto_chunk(hs: int) -> int:
        return 256 if hs >= 7168 else (512 if hs >= 4096 else 1024)

    @staticmethod
    def _auto_batch() -> int:
        try:
            f, _ = torch.cuda.mem_get_info()
            g = f / (1024**3)
            return 16 if g >= 40 else (8 if g >= 16 else 4)
        except Exception:
            return 8

    @staticmethod
    def _auto_compress(ml: int):
        if ml >= 131072:
            return (4, 4096, 0.15)
        if ml >= 32768:
            return (4, 3072, 0.18)
        if ml >= 8192:
            return (4, 2048, 0.20)
        return (4, 1024, 0.25)

    # ==============================================================
    # 前缀缓存
    # ==============================================================

    def _try_prefix(self, req: Request) -> bool:
        if not self.PREFIX_CACHING_ENABLED or not req.prompt_tokens:
            return False
        self._prefix_total += 1
        bid = self.cache.has_prefix(req.prompt_tokens)
        if bid is not None:
            self._prefix_hits += 1
            self.active_decode_pool.append(DecodeRequest(
                tokens=list(req.prompt_tokens), generated_tokens=[],
                request_id=req.request_id, max_new_tokens=req.max_new_tokens,
                cache_block_id=bid,
            ))
            return True
        return False

    # ==============================================================
    # 公共 API
    # ==============================================================

    def submit(self, req: Request) -> None:
        if not self._try_prefix(req):
            self.pending_requests.append(req)

    def shutdown(self):
        self._running = False

    # ==============================================================
    # 投机解码 (小模型起草 + 大模型一次验证)
    # ==============================================================

    def _spec_decode_step(self):
        """投机解码：用小模型生成 N 个草稿 token，大模型一次验证。

        流程：
          1. 草稿模型用 greedy decode 生成 N 个 token
          2. 拼接 context + drafts，大模型一次 forward
          3. 逐位置比较：匹配则接受，不匹配则截断
          4. 额外生成一个 bonus token（所有草稿都接受时）

        优势：N 个 token 只需 1 次大模型 forward（vs N 次）
        """
        if self._draft_model is None:
            return
        if not self.active_decode_pool:
            return

        dm = self._draft_model
        N = self.SPECULATIVE_DRAFT_N

        for req in list(self.active_decode_pool):
            if req.is_done or req.request_id in self._spec_handled:
                continue
            if len(req.tokens) < self.SPECULATIVE_MIN_LEN:
                continue

            # ── 1. 草稿模型生成 ─────────────────────────────────
            context = req.tokens
            context_t = torch.tensor([context], dtype=torch.long, device="cuda")

            with torch.no_grad():
                dm_out = dm.generate(
                    input_ids=context_t,
                    max_new_tokens=N,
                    do_sample=False,         # greedy
                    use_cache=True,
                    pad_token_id=0,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
            draft_ids = dm_out.sequences[0, len(context):].tolist()
            if not draft_ids:
                continue

            # ── 2. 拼接 + 大模型验证 ────────────────────────────
            verify_ids = context + draft_ids
            verify_t = torch.tensor([verify_ids], dtype=torch.long, device="cuda")

            past_kv = None
            if req.cache_block_id is not None and self._KV_CACHE_ENABLED:
                past_kv = self.cache.load_kv(req.cache_block_id)

            with torch.no_grad():
                if past_kv is not None:
                    vm_out = self.model.forward(
                        input_ids=context_t[:, -1:],  # 只送最后一个 token
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                    # 然后用 full verify_ids 走一遍验证
                    # 实际更高效的方案: 直接用 draft_ids 拼接后 forward
                    vm_out = self.model.forward(
                        input_ids=torch.tensor([draft_ids], dtype=torch.long, device="cuda"),
                        past_key_values=_extract_pkv(vm_out),
                        use_cache=True,
                    )
                else:
                    vm_out = self.model.forward(
                        input_ids=verify_t,
                        use_cache=False,
                    )

            logits_t = _extract_logits(vm_out)
            new_kv = _extract_pkv(vm_out)

            # ── 3. 逐位置验证 ───────────────────────────────────
            accepted = []
            # 从 context 末尾开始对应的每个 draft 位置
            for i, dt in enumerate(draft_ids):
                # 大模型在 context_len + i 位置的预测
                pred_logits = logits_t[0, -(len(draft_ids) - i), :] if new_kv is None else logits_t[0, 0, :]
                # 更准确：直接用每步的 logits
                pos_idx = -(len(draft_ids) - i)  # 倒着取
                step_logits = logits_t[0, pos_idx, :]
                predicted = int(step_logits.argmax().item())
                if predicted == dt:
                    accepted.append(dt)
                else:
                    # 不匹配: 用大模型的预测
                    next_tok = predicted
                    # 需要重新计算 KV
                    break
            else:
                # 全部接受 → bonus token
                bonus_logits = logits_t[0, -1, :]
                next_tok = int(bonus_logits.argmax().item())
                # bonus token 已包含在 new_kv 中
                accepted.append(next_tok)

            if not accepted:
                continue

            # ── 4. 更新状态 ─────────────────────────────────────
            self._spec_handled.add(req.request_id)
            self._spec_draft_total += 1
            self._spec_draft_hits += len(accepted)

            for tok in accepted:
                req.step()
                req.generated_tokens.append(tok)
                req.tokens.append(tok)

            if new_kv is not None and req.cache_block_id is not None:
                self.cache.store_kv(req.cache_block_id, new_kv)

            logger.debug(
                "Speculative: %s accepted %d/%d drafts (draft model)",
                req.request_id, len(accepted), len(draft_ids),
            )

    # ==============================================================
    # Chunked Prefill
    # ==============================================================

    def _batch_chunked_prefill(self):
        if not self.pending_requests:
            return
        now = time.monotonic()
        if now - self._last_prefill_time < self._PREFILL_BATCH_TIMEOUT and \
           len(self.pending_requests) < self._PREFILL_BATCH_MAX:
            return

        batch = list(self.pending_requests)
        self.pending_requests.clear()
        self._last_prefill_time = now
        cs = self.CHUNK_SIZE

        for req in batch:
            toks = req.prompt_tokens
            total = len(toks)
            off = 0
            cb = None

            while off < total:
                end = min(off + cs, total)
                chunk = toks[off:end]
                last = end >= total
                inp = torch.tensor([chunk], dtype=torch.long, device="cuda")

                with torch.cuda.stream(self.prefill_stream), torch.no_grad():
                    if off == 0:
                        out = self.model.forward(input_ids=inp, use_cache=True)
                    else:
                        pk = self.cache.load_kv(cb.block_id)
                        out = self.model.forward(input_ids=inp, past_key_values=pk, use_cache=True)

                kv = _extract_pkv(out)
                if kv is not None:
                    if off == 0:
                        cb = self.cache.allocate(chunk)
                    self.cache.store_kv(cb.block_id, kv)
                else:
                    cb = None
                    break
                off = end
                if not last:
                    self.pending_requests.insert(0, Request(
                        prompt_tokens=toks[off:], request_id=req.request_id,
                        max_new_tokens=req.max_new_tokens,
                    ))
                    break

            if last and cb is not None:
                if self.PREFIX_CACHING_ENABLED:
                    self.cache.pin_prefix_from_match(cb.block_id)
                self.active_decode_pool.append(DecodeRequest(
                    tokens=list(toks), generated_tokens=[],
                    request_id=req.request_id, max_new_tokens=req.max_new_tokens,
                    cache_block_id=cb.block_id,
                ))

    # ==============================================================
    # 解码
    # ==============================================================

    def _decode_step(self):
        if not self.active_decode_pool:
            self._decode_bs = 0
            return

        # KV 压缩
        if self.trigger_reset:
            for req in self.active_decode_pool:
                if self.ADAPTIVE_COMPRESSION_ENABLED:
                    self._compress_adaptive(req)
                else:
                    self._compress_kv(req)
            self.step_since_reset = 0
            self.trigger_reset = False

        with torch.cuda.stream(self.decode_stream):
            for req in self.active_decode_pool:
                if req.request_id in self._spec_handled:
                    continue
                tok = req.tokens[-1] if req.tokens else 0
                inp = torch.tensor([[tok]], dtype=torch.long, device="cuda")
                pk = None
                if req.cache_block_id is not None:
                    pk = self.cache.load_kv(req.cache_block_id)
                with torch.no_grad():
                    if pk is not None and self._KV_CACHE_ENABLED:
                        out = self.model.forward(input_ids=inp, past_key_values=pk, use_cache=True)
                    else:
                        out = self.model.forward(
                            input_ids=torch.tensor([req.tokens], dtype=torch.long, device="cuda"),
                            use_cache=False,
                        )
                lt = _extract_logits(out)
                nt = int(lt[0, -1, :].argmax().item())
                req.step()
                req.generated_tokens.append(nt)
                req.tokens.append(nt)
                nkv = _extract_pkv(out)
                if nkv is not None and req.cache_block_id is not None:
                    self.cache.store_kv(req.cache_block_id, nkv)

        self.step_since_reset += 1
        if self.step_since_reset >= self._comp_threshold:
            self.trigger_reset = True

    # ==============================================================
    # KV 压缩
    # ==============================================================

    def _compress_kv(self, req):
        if req.cache_block_id is None or not req.tokens:
            return
        old = len(req.tokens)
        s, r = self._COMPRESS_SINK_N, self._COMPRESS_RECENT_N
        if old <= s + r + 2:
            return
        nt = req.tokens[:s] + req.tokens[-r:]
        req.tokens = nt
        kv = self.cache.load_kv(req.cache_block_id)
        if kv is not None:
            ckv = []
            for k, v in kv:
                ckv.append((torch.cat([k[:, :, :s, :], k[:, :, -r:, :]], dim=2),
                            torch.cat([v[:, :, :s, :], v[:, :, -r:, :]], dim=2)))
            self.cache.store_kv(req.cache_block_id, ckv)

    def _compress_adaptive(self, req):
        if req.cache_block_id is None or not req.tokens:
            return
        s, r = self._COMPRESS_SINK_N, self._COMPRESS_RECENT_N
        old = len(req.tokens)
        if old <= s + r + 8:
            return
        keep = set(range(s))
        keep.update(range(old - r, old))
        ms, me = s, old - r
        ml = me - ms
        if ml > 0:
            ne = max(4, int(ml * self._COMPRESS_IMPORTANCE_FRAC))
            st = max(1, ml // ne)
            for i in range(ms, me, st):
                keep.add(i)
        ki = sorted(keep)
        req.tokens = [req.tokens[i] for i in ki]
        kv = self.cache.load_kv(req.cache_block_id)
        if kv is not None:
            ckv = []
            for k, v in kv:
                ckv.append((
                    torch.index_select(k, 2, torch.tensor(ki, device=k.device)),
                    torch.index_select(v, 2, torch.tensor(ki, device=v.device)),
                ))
            self.cache.store_kv(req.cache_block_id, ckv)

    # ==============================================================
    # 主步进
    # ==============================================================

    async def step(self):
        # 1. 预填充
        if self.CHUNKED_PREFILL_ENABLED:
            self._batch_chunked_prefill()
        else:
            self._batch_prefill()

        # 2. VRAM 检查
        if self._vram_budget is not None:
            self._vram_step += 1
            if self._vram_step % self._vram_check_int == 0 or self._vram_degraded:
                st = self._vram_budget.check()
                act = st.get("action")
                if act == "reduce_chunk" and self.CHUNK_SIZE > 64:
                    self.CHUNK_SIZE = max(64, self.CHUNK_SIZE // 2)
                    self._vram_degraded = True
                elif act == "compress_kv" and not self.trigger_reset:
                    self._COMPRESS_RECENT_N = max(512, self._COMPRESS_RECENT_N // 2)
                    self.trigger_reset = True
                    self._vram_degraded = True
                elif act == "emergency":
                    self._PREFILL_BATCH_MAX = max(1, self._PREFILL_BATCH_MAX // 2)
                    self.CHUNK_SIZE = max(64, self.CHUNK_SIZE // 2)
                    if not self.trigger_reset:
                        self.trigger_reset = True
                    self._vram_degraded = True

        # 3. 投机解码（小模型草稿）
        self._spec_decode_step()

        # 4. Self-Spec
        if self._self_spec_enabled and self._skeleton_draft is not None:
            self._decode_self_speculative()

        # 5. Goose init
        if self._goose_enabled and self._goose_engine is None and _GOOSE_AVAILABLE and not self._goose_init_tried:
            self._goose_init_tried = True
            self._init_goose()

        # 6. Self-Spec init
        if self._self_spec_enabled and self._skeleton_draft is None and _SKELETON_AVAILABLE and not self._skel_init_tried:
            self._skel_init_tried = True
            self._init_skeleton()

        # 7. Goose 推测
        self._decode_speculative()

        # 8. 基础解码
        self._decode_step()

        # 9. 清理
        self._spec_handled.clear()
        torch.cuda.current_stream().wait_stream(self.prefill_stream)
        torch.cuda.current_stream().wait_stream(self.decode_stream)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        self._gc()

        # 打印投机命中率
        if self._spec_draft_total >= 50 and self._spec_draft_total % 50 == 0:
            rate = self._spec_draft_hits / max(self._spec_draft_total, 1)
            logger.info("Speculative draft: %d/%d hits (%.1f%%)",
                       self._spec_draft_hits, self._spec_draft_total, rate * 100)

    # ==============================================================
    # GC
    # ==============================================================

    def _gc(self):
        done = [d for d in self.active_decode_pool if d.is_done]
        if not done:
            self.active_decode_pool = [d for d in self.active_decode_pool if not d.is_done]
            self.cache.gc()
            return
        torch.cuda.synchronize()
        for d in done:
            if d.cache_block_id is not None:
                self.cache.free_block(d.cache_block_id)
        self.active_decode_pool = [d for d in self.active_decode_pool if not d.is_done]
        self.cache.gc()

    # ==============================================================
    # Goose + Self-Spec (来自已有代码)
    # ==============================================================

    def _init_goose(self, tree=False, md=5):
        if self._goose_engine is not None or not _GOOSE_AVAILABLE:
            return
        vs = getattr(self.model, "vocab_size", 32000)
        self._goose_engine = goose_core.SpeculativeEngine(vocab_size=vs, max_draft=md, tree_enabled=tree)
        self._goose_engine.enable()

    def _init_skeleton(self):
        if not _SKELETON_AVAILABLE:
            return
        self._skeleton_draft = _SkeletonGen(model=self.model, skip_fraction=0.30, max_draft=5)
        self._self_spec_enabled = True

    def _decode_self_speculative(self):
        if not self._self_spec_enabled or self._skeleton_draft is None or not self.active_decode_pool:
            return
        for req in list(self.active_decode_pool):
            if req.is_done or req.request_id in self._spec_handled or len(req.tokens) < 8:
                continue
            pk = None
            if req.cache_block_id is not None and self._KV_CACHE_ENABLED:
                pk = self.cache.load_kv(req.cache_block_id)
            lt = torch.tensor([[req.tokens[-1]]], dtype=torch.long, device="cuda")
            dt = self._skeleton_draft.generate_draft(input_ids=lt, past_key_values=pk)
            if not dt:
                continue
            if self._goose_engine is not None:
                acc, nt, nkv = self._goose_engine.verify_linear(self.model, pk, dt, req.tokens)
            else:
                acc, nt, nkv = self._manual_verify(dt, req, pk)
            if not acc:
                continue
            self._spec_handled.add(req.request_id)
            for t in acc:
                req.step()
                req.generated_tokens.append(t)
                req.tokens.append(t)
            req.step()
            req.generated_tokens.append(nt)
            req.tokens.append(nt)
            if nkv is not None and req.cache_block_id is not None:
                self.cache.store_kv(req.cache_block_id, nkv)

    def _manual_verify(self, drafts, req, pk):
        acc = []
        nkv = pk
        nt = None
        for dt in drafts:
            inp = torch.tensor([[req.tokens[-1] if not acc else acc[-1]]], dtype=torch.long, device="cuda")
            with torch.no_grad():
                out = self.model.forward(input_ids=inp, past_key_values=nkv, use_cache=True)
            lt = _extract_logits(out)
            pd = int(lt[0, -1, :].argmax().item())
            if pd == dt:
                acc.append(dt)
                nkv = _extract_pkv(out)
            else:
                nt = pd
                nkv = _extract_pkv(out)
                break
        else:
            inp = torch.tensor([[acc[-1]]], dtype=torch.long, device="cuda")
            with torch.no_grad():
                out = self.model.forward(input_ids=inp, past_key_values=nkv, use_cache=True)
            lt = _extract_logits(out)
            nt = int(lt[0, -1, :].argmax().item())
            nkv = _extract_pkv(out)
        return acc, nt, nkv

    def _decode_speculative(self):
        if not self._goose_enabled or self._goose_engine is None or not self.active_decode_pool:
            return
        eng = self._goose_engine
        for req in list(self.active_decode_pool):
            if req.is_done or req.request_id in self._spec_handled:
                continue
            ctx = req.tokens
            if not eng.can_speculate(ctx):
                continue
            dr, _ = eng.generate_draft(ctx)
            if not dr:
                continue
            pk = None
            pl = 0
            if req.cache_block_id is not None and self._KV_CACHE_ENABLED:
                pk = self.cache.load_kv(req.cache_block_id)
                if pk is not None and pk[0][0] is not None:
                    pl = pk[0][0].shape[2]
            st = self.decode_stream
            with torch.cuda.stream(st), torch.no_grad():
                if eng.tree_enabled:
                    tr = eng.build_spine_tree(ctx[-1], dr)
                    acc, nt, nkv = eng.verify_tree(self.model, pk, tr, pl)
                else:
                    acc, nt, nkv = eng.verify_linear(self.model, pk, dr, ctx)
            if not acc and pk is None:
                continue
            self._spec_handled.add(req.request_id)
            for t in acc:
                req.step()
                req.generated_tokens.append(t)
                req.tokens.append(t)
            req.step()
            req.generated_tokens.append(nt)
            req.tokens.append(nt)
            if nkv is not None and req.cache_block_id is not None:
                self.cache.store_kv(req.cache_block_id, nkv)
            if pk is not None:
                hw = req.tokens[-(len(acc) + 1):]
                if hw:
                    li = torch.tensor([hw], dtype=torch.long, device="cuda")
                    with torch.no_grad():
                        lb = self.model.forward(input_ids=li, use_cache=False)
                    ll = _extract_logits(lb)
                    eng.harvest_logits(ll, list(hw))

    def _batch_prefill(self):
        if not self.pending_requests:
            return
        batch = list(self.pending_requests)
        self.pending_requests.clear()
        for req in batch:
            inp = torch.tensor([req.prompt_tokens], dtype=torch.long, device="cuda")
            with torch.cuda.stream(self.prefill_stream), torch.no_grad():
                out = self.model.forward(input_ids=inp, use_cache=True)
            kv = _extract_pkv(out)
            if kv is not None:
                cb = self.cache.allocate(req.prompt_tokens)
                self.cache.store_kv(cb.block_id, kv)
                if self.PREFIX_CACHING_ENABLED:
                    self.cache.pin_prefix_from_match(cb.block_id)
                self.active_decode_pool.append(DecodeRequest(
                    tokens=list(req.prompt_tokens), generated_tokens=[],
                    request_id=req.request_id, max_new_tokens=req.max_new_tokens,
                    cache_block_id=cb.block_id,
                ))
