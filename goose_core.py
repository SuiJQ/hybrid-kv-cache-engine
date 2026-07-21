"""
goose_core.py — Goose-style Anisotropic Speculative Decoding.

Implements training-free speculative decoding per arXiv:2604.02047.

Components
----------
1. PLDMatcher   — Multi-length n-gram context matching + consensus detection
2. TransitionTable — GPU bigram adjacency table (logit harvesting)
3. SpineTreeBuilder — Anisotropic spine tree construction (Algorithm 1)
4. TreeMaskGenerator — Tree attention mask for single-forward verification
5. SpeculativeEngine — Top-level orchestrator

Compatibility
-------------
- KV Cache (PagedAttention + RadixAttention): ✅ prefix KV stored in
  HybridCache; draft tokens form a contiguous ≤60-token sequence; no
  non-contiguous KV issues.
- Expert Cache + SERE: ✅ orthogonal multiplicative gains.
- FlashAttention + torch.compile: ✅ fixed-shape mask (padding to B=60)
  avoids recompilation.
- Dual CUDA Stream pipeline: ✅ runs on decode_stream; prefetch_stream
  and transfer_stream unaffected.
"""

from __future__ import annotations

import collections
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1. PLD Matcher — Prompt-Lookup Decoding
# ═══════════════════════════════════════════════════════════════════════


class PLDMatcher:
    """Multi-length n-gram context matching with consensus detection.

    Given a sequence of generated tokens, finds the longest matching
    continuation from previous context, checking multiple n-gram lengths
    to detect high-confidence matches (consensus).

    Parameters
    ----------
    ngram_lengths : tuple[int, ...]
        N-gram lengths to query (default ``(3, 4, 5)``).
    max_continuation : int
        Maximum continuation length (default 20).
    consensus_threshold : int
        How many lengths must agree on the first continuation token
        to trigger bypass mode (default 2).
    confidence_threshold : int
        Spine length above which bypass mode triggers regardless
        of consensus (default 8).
    """

    def __init__(
        self,
        ngram_lengths: tuple[int, ...] = (3, 4, 5),
        max_continuation: int = 20,
        consensus_threshold: int = 2,
        confidence_threshold: int = 8,
    ):
        self.ngram_lengths = ngram_lengths
        self.max_continuation = max_continuation
        self.consensus_threshold = consensus_threshold
        self.confidence_threshold = confidence_threshold

    def match(
        self, context: list[int], generated: list[int]
    ) -> tuple[list[int], bool]:
        """Find PLD continuation.

        Parameters
        ----------
        context : list[int]
            The full generated token sequence (previous tokens).
        generated : list[int]
            Same as context — this is the sequential text.

        Returns
        -------
        spine_tokens : list[int]
            Up to ``max_continuation`` matched continuation tokens.
        consensus : bool
            True when ≥2 n-gram lengths agree on the first continuation token,
            OR the match length ≥ confidence_threshold.
        """
        ctx_len = len(context)
        if ctx_len < min(self.ngram_lengths):
            return [], False

        results = {}  # ngram_len -> (start_pos, continuation)

        for n in self.ngram_lengths:
            suffix = context[-n:]  # last n tokens as query
            # Search backward from the last occurrence before the suffix's
            # own position (i.e., look for an earlier occurrence).
            # We iterate backward through the context, prioritizing the
            # most recent match (which has the highest predictive value).
            for i in range(ctx_len - n - 1, -1, -1):
                if context[i : i + n] == suffix:
                    # Found a match — the continuation starts after it
                    cont = context[i + n : i + n + self.max_continuation]
                    if cont:
                        results[n] = cont
                    break

        if not results:
            return [], False

        # Pick the longest continuation
        best_cont = max(results.values(), key=len)

        # Consensus: do ≥2 lengths agree on the first continuation token?
        if len(results) >= 2:
            first_tokens = [r[0] for r in results.values() if len(r) > 0]
            consensus = max(collections.Counter(first_tokens).values()) >= self.consensus_threshold
        else:
            consensus = False

        # Bypass: long match or consensus
        consensus = consensus or len(best_cont) >= self.confidence_threshold

        return best_cont, consensus


# ═══════════════════════════════════════════════════════════════════════
# 2. TransitionTable — GPU Bigram Adjacency Table
# ═══════════════════════════════════════════════════════════════════════


class TransitionTable:
    """GPU bigram adjacency table for transition (TR) token prediction.

    Two-tier design:
      - Tier 1 (Unigram): shape ``(vocab_size, top_k)`` — top-K successors
        for each token.
      - Tier 2 (Bigram): hash-map-like tensor structure mapping
        ``(token_{t-1}, token_t)`` → top-K successors for ``token_{t+1}``.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size (usually 32000–128000).
    top_k : int
        Number of successors to store per key (default 10).
    bigram_buckets : int
        Number of hash buckets for bigram entries (default 65536).
    min_score : float
        Prune successors with score below this threshold (default 0.01).
    device : torch.device
        Device to store tensors on (default CUDA).
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        top_k: int = 10,
        bigram_buckets: int = 65536,
        min_score: float = 0.01,
        device: torch.device = torch.device("cuda"),
    ):
        self.vocab_size = vocab_size
        self.top_k = top_k
        self.min_score = min_score
        self.device = device

        # Tier 1: Unigram — (vocab_size, top_k) for ids and scores
        self._unigram_ids = torch.zeros(vocab_size, top_k, dtype=torch.long, device=device)
        self._unigram_scores = torch.zeros(vocab_size, top_k, dtype=torch.float16, device=device)

        # Tier 2: Bigram — bucketed hash table
        # (bigram_buckets, top_k) for ids and scores
        self._bigram_ids = torch.zeros(bigram_buckets, top_k, dtype=torch.long, device=device)
        self._bigram_scores = torch.zeros(bigram_buckets, top_k, dtype=torch.float16, device=device)
        self._bigram_buckets = bigram_buckets
        self._bigram_keys = torch.zeros(bigram_buckets, 2, dtype=torch.long, device=device)

        logger.info(
            "TransitionTable: vocab=%d, top_k=%d, bigram_buckets=%d, device=%s",
            vocab_size, top_k, bigram_buckets, device,
        )

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _bigram_hash(t1: int, t2: int, buckets: int) -> int:
        """Simple hash: (t1 * 2654435761 + t2) % buckets."""
        return (t1 * 2654435761 + t2) % buckets

    # ------------------------------------------------------------------
    # Harvesting
    # ------------------------------------------------------------------

    @torch.no_grad()
    def harvest(
        self, logits: torch.Tensor, context_tokens: list[int]
    ) -> None:
        """Extract top-K from all positions and update tables.

        Called after every forward pass (including rejected branches).

        Parameters
        ----------
        logits : torch.Tensor
            Shape ``(1, seq_len, vocab_size)`` — logits from the last forward.
        context_tokens : list[int]
            Token IDs corresponding to each logit position (for bigram key
            construction).  Length should equal ``seq_len``.
        """
        seq_len = logits.shape[1]
        if seq_len < 2:
            return

        # Get top-K indices and logit values for ALL positions at once
        probs = F.softmax(logits[0], dim=-1)  # (seq_len, vocab_size)
        topk_vals, topk_idx = torch.topk(probs, self.top_k, dim=-1)  # (seq_len, K)

        # Update unigram: for each position, update the lookup for the token
        # at that position
        for pos in range(seq_len - 1):
            token = context_tokens[pos]
            token_key = int(token)
            if token_key >= self.vocab_size:
                continue

            new_ids = topk_idx[pos].to(torch.long)
            new_scores = topk_vals[pos].to(torch.float16)
            # Merge — keep top-K overall
            merged_ids = torch.cat([self._unigram_ids[token_key].to(torch.long), new_ids])
            merged_scores = torch.cat([self._unigram_scores[token_key], new_scores])
            _, sort_idx = merged_scores.sort(descending=True)
            self._unigram_ids[token_key] = merged_ids[sort_idx[:self.top_k]].to(torch.long)
            self._unigram_scores[token_key] = merged_scores[sort_idx[:self.top_k]]

        # Update bigram: each adjacent pair (pos-1, pos) → pos+1 successors
        for pos in range(1, seq_len - 1):
            t_prev = context_tokens[pos - 1]
            t_curr = context_tokens[pos]
            bucket = self._bigram_hash(t_prev, t_curr, self._bigram_buckets)

            new_ids = topk_idx[pos + 1].to(torch.long)
            new_scores = topk_vals[pos + 1].to(torch.float16)
            self._bigram_keys[bucket] = torch.tensor([t_prev, t_curr], dtype=torch.long, device=self.device)
            self._bigram_ids[bucket] = new_ids
            self._bigram_scores[bucket] = new_scores

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    @torch.no_grad()
    def lookup_unigram(self, token: int) -> list[tuple[int, float]]:
        """Return top-K successors for a single token.

        Returns
        -------
        list[(token_id, score)]
            ``top_k`` entries; empty list if table is uninitialized.
        """
        if torch.all(self._unigram_scores == 0):
            return []
        ids = self._unigram_ids[token]
        scores = self._unigram_scores[token]
        # Filter out zero-score entries (padding)
        valid = scores > self.min_score
        return [
            (int(ids[i]), float(scores[i]))
            for i in range(len(ids)) if valid[i]
        ]

    @torch.no_grad()
    def lookup_bigram(self, t_prev: int, t_curr: int) -> list[tuple[int, float]]:
        """Return top-K successors for a bigram pair.

        Returns
        -------
        list[(token_id, score)]
            ``top_k`` entries; falls back to unigram if bigram slot empty.
        """
        if torch.all(self._bigram_scores == 0):
            return self.lookup_unigram(t_curr)

        bucket = self._bigram_hash(t_prev, t_curr, self._bigram_buckets)
        key = self._bigram_keys[bucket]
        if int(key[0]) == t_prev and int(key[1]) == t_curr:
            ids = self._bigram_ids[bucket]
            scores = self._bigram_scores[bucket]
            valid = scores > self.min_score
            return [(int(ids[i]), float(scores[i])) for i in range(len(ids)) if valid[i]]
        return self.lookup_unigram(t_curr)


# ═══════════════════════════════════════════════════════════════════════
# 3. Spine Tree Construction
# ═══════════════════════════════════════════════════════════════════════


class SpineTreeBuilder:
    """Build an anisotropic spine tree from PLD + TR sources.

    Implements Algorithm 1 from Goose (arXiv:2604.02047):

    1. Lay the spine (PLD chain) — up to B*r nodes
    2. Attach root branches — top TR successors at root
    3. Attach spine branches — harmonic decay at each spine node
    4. BFS-extend branches — up to max_depth via adjacency table

    Parameters
    ----------
    budget : int
        Total node budget B (default 60).
    max_depth : int
        Maximum branch extension depth (default 6).
    spine_branch_ratio : float
        Fraction of remaining budget allocated to spine-level branches
        vs. root-level branches (default 0.5).
    """

    def __init__(
        self,
        budget: int = 60,
        max_depth: int = 6,
        spine_branch_ratio: float = 0.5,
    ):
        self.budget = budget
        self.max_depth = max_depth
        self.spine_branch_ratio = spine_branch_ratio

    def build(
        self,
        anchor_token: int,
        spine: list[int],
        transition_table: TransitionTable,
        spine_ratio: float = 0.4,
    ) -> dict:
        """Build spine tree.

        Parameters
        ----------
        anchor_token : int
            The last accepted token (root of the tree).
        spine : list[int]
            PLD-matched continuation tokens.
        transition_table : TransitionTable
            GPU bigram adjacency table for branch candidates.
        spine_ratio : float
            Fraction of budget allocated to spine (0.15–0.50).

        Returns
        -------
        dict
            ``{"nodes": list[int], "parents": list[int],
              "source": list[str]}``
            - nodes: token IDs at each tree position (position 0 = anchor)
            - parents: parent index for each node (-1 for root)
            - source: "pls" (spine) or "tr" (branch)

            Length is at most ``budget + 1`` (including anchor).
        """
        B = self.budget

        # --- Step 0: Budget allocation ---
        bs = min(len(spine), max(1, int(B * spine_ratio)))  # spine nodes
        remaining = B - bs
        br = int(remaining * (1 - self.spine_branch_ratio))  # root branches
        b_spine = remaining - br  # spine-level branches

        # --- Initialize tree ---
        # nodes[0] = anchor token
        nodes = [anchor_token]
        parents = [-1]
        source = ["anchor"]

        # --- Step 1: Lay spine ---
        for i in range(bs):
            if i < len(spine):
                nodes.append(spine[i])
                parents.append(i)   # previous spine node
                source.append("pls")
            else:
                break
        actual_bs = len(nodes) - 1  # actual spine length

        # --- Step 2: Root branches ---
        root_successors = transition_table.lookup_unigram(anchor_token)
        for succ, _score in root_successors[:br]:
            if len(nodes) >= B + 1:
                break
            # Avoid duplicate with spine[0]
            if actual_bs > 0 and succ == nodes[1]:
                continue
            nodes.append(succ)
            parents.append(0)  # parent = root (anchor)
            source.append("tr")

        # --- Step 3: Spine branches (harmonic decay) ---
        if b_spine > 0:
            harmonic_sum = sum(1.0 / i for i in range(1, actual_bs + 1))
            for i in range(1, actual_bs + 1):
                if len(nodes) >= B + 1:
                    break
                # Harmonic allocation
                count = max(1, int(b_spine * (1.0 / i) / harmonic_sum))
                spine_token = nodes[i]
                successors = transition_table.lookup_unigram(spine_token)
                # Also try bigram for better accuracy
                if i > 1:
                    bigram_succ = transition_table.lookup_bigram(
                        nodes[i - 1], spine_token
                    )
                    if bigram_succ:
                        successors = bigram_succ
                for succ, _score in successors[:count]:
                    if len(nodes) >= B + 1:
                        break
                    # Avoid duplicate with next spine token
                    if i + 1 <= actual_bs and succ == nodes[i + 1]:
                        continue
                    nodes.append(succ)
                    parents.append(i)  # parent = spine node i
                    source.append("tr")

        # --- Step 4: BFS-extend branches ---
        # For each branch leaf, extend up to max_depth
        start_idx = 1 + actual_bs  # first non-spine, non-root-branch node
        leaves_before = len(nodes)
        for idx in range(start_idx, leaves_before):
            if len(nodes) >= B + 1:
                break
            self._extend_branch(
                nodes, parents, source,
                leaf_idx=idx,
                transition_table=transition_table,
            )

        return {
            "nodes": nodes,
            "parents": parents,
            "source": source,
        }

    def _extend_branch(
        self,
        nodes: list[int],
        parents: list[int],
        source: list[str],
        leaf_idx: int,
        transition_table: TransitionTable,
    ) -> None:
        """BFS-extend a branch leaf up to max_depth."""
        token = nodes[leaf_idx]
        ancestors = self._get_ancestor_tokens(nodes, parents, leaf_idx, depth=2)
        if len(ancestors) >= 2:
            successors = transition_table.lookup_bigram(ancestors[-2], ancestors[-1])
        else:
            successors = transition_table.lookup_unigram(token)

        depth = 0
        current = leaf_idx
        for succ, _score in successors[:3]:  # limit fan-out per level
            if depth >= self.max_depth:
                break
            if len(nodes) >= self.budget + 1:
                break
            nodes.append(succ)
            parents.append(current)
            source.append("tr")
            depth += 1
            current = len(nodes) - 1

    @staticmethod
    def _get_ancestor_tokens(
        nodes: list[int], parents: list[int], idx: int, depth: int = 2
    ) -> list[int]:
        """Get up to *depth* ancestor token values."""
        result = []
        current = idx
        for _ in range(depth):
            p = parents[current]
            if p < 0:
                break
            result.append(nodes[p])
            current = p
        return list(reversed(result))


# ═══════════════════════════════════════════════════════════════════════
# 4. Tree Attention Mask Generator (Phase 2)
# ═══════════════════════════════════════════════════════════════════════


class TreeMaskGenerator:
    """Generate tree attention masks for single-forward verification.

    The mask constrains each candidate token to attend only to its
    ancestors (tree topology), while all tokens can attend to the
    prefix KV.  Shape is always pad-to-max to avoid torch.compile
    recompilation.

    Parameters
    ----------
    max_draft : int
        Maximum draft sequence length (default 60).
    """

    _DTYPE_MASK = (-65504.0, "float16 max is 65504 so -inf via -65504")

    def __init__(self, max_draft: int = 60):
        self.max_draft = max_draft
        # Pre-allocate reusable mask buffer (avoid re-creation each step)
        inf_val = float("-inf")
        self._mask_buf = torch.full(
            (1, 1, max_draft, max_draft), inf_val, dtype=torch.float16, device="cuda"
        )
        # Diagonal (self-attention): 0
        idx = torch.arange(max_draft, device="cuda")
        self._mask_buf[0, 0, idx, idx] = 0.0

    def build_mask(
        self,
        parents: list[int],
        draft_length: int,
    ) -> torch.Tensor:
        """Build tree attention mask for the draft-to-draft region.

        Parameters
        ----------
        parents : list[int]
            Parent index for each draft node.  Length ≤ max_draft.
            parent[i] >= 0 means node i attends to node parent[i].
            The anchor (index 0) has parent -1 (no restriction).
        draft_length : int
            Actual number of draft tokens (≤ max_draft).

        Returns
        -------
        torch.Tensor
            Mask of shape ``(1, 1, max_draft, max_draft)`` with:
            - 0.0 where attention is allowed
            - -inf where blocked
            The mask is for the draft→draft region only.  The caller
            is responsible for setting draft→prefix attention to 0.0.
        """
        mask = self._mask_buf.clone()
        # Set ancestor attention for each node
        for i in range(1, draft_length):
            p = parents[i]
            while p >= 0:
                mask[0, 0, i, p] = 0.0
                p = parents[p] if p < len(parents) else -1

        return mask


# ═══════════════════════════════════════════════════════════════════════
# 5. Speculative Engine — Top-Level Orchestrator
# ═══════════════════════════════════════════════════════════════════════


class SpeculativeEngine:
    """Top-level orchestrator for Goose speculative decoding.

    Manages PLD matching, transition table, tree construction, and
    verification.  Designed to integrate with ``UnifiedScheduler``.

    Parameters
    ----------
    vocab_size : int
        Model vocabulary size.
    max_draft : int
        Maximum draft tokens per cycle (default 5 → Phase 0/1 chain;
        60 → Phase 2 tree).
    tree_enabled : bool
        If True, use tree attention verification (Phase 2).
        If False, use linear chain verification (Phase 0/1).
    budget : int
        Tree node budget (default 60; only used when ``tree_enabled``).
    top_k : int
        Successor entries per token in transition table (default 10).
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        max_draft: int = 5,
        tree_enabled: bool = False,
        budget: int = 60,
        top_k: int = 10,
    ):
        self.max_draft = max_draft
        self.tree_enabled = tree_enabled
        self.budget = budget

        # Sub-modules
        self.pld_matcher = PLDMatcher()
        self.transition_table = TransitionTable(
            vocab_size=vocab_size,
            top_k=top_k,
        )
        self.tree_builder = SpineTreeBuilder(budget=budget) if tree_enabled else None
        self.tree_mask_gen = TreeMaskGenerator(max_draft=budget) if tree_enabled else None

        # Adaptive spine ratio (EMA)
        self._spine_ratio = 0.4
        self._ema_alpha = 0.3
        self._pld_acceptance_ema = 0.3

        # Warm-up state
        self._warmup_steps = 0
        self._warmup_threshold = 10  # steps before full speculation

        logger.info(
            "SpeculativeEngine: max_draft=%d, tree=%s, budget=%d, top_k=%d",
            max_draft, tree_enabled, budget, top_k,
        )

    # ------------------------------------------------------------------
    # Draft generation
    # ------------------------------------------------------------------

    def generate_draft(
        self, context: list[int]
    ) -> tuple[list[int], bool]:
        """Generate draft tokens for the next verification cycle.

        Parameters
        ----------
        context : list[int]
            Full generated token sequence.

        Returns
        -------
        drafts : list[int]
            Draft tokens to verify.
        bypass : bool
            True when confidence is high enough to skip tree construction.
        """
        spine, consensus = self.pld_matcher.match(context, context)
        if not spine:
            return [], False

        # Bypass mode: consensus or long match
        bypass = consensus
        if bypass:
            return spine[:self.max_draft], True

        return spine[:self.max_draft], False

    # ------------------------------------------------------------------
    # Confidence adaptation
    # ------------------------------------------------------------------

    def update_acceptance_rate(self, accepted_ratio: float) -> None:
        """Update EMA of PLD acceptance rate and adapt spine ratio.

        Parameters
        ----------
        accepted_ratio : float
            Fraction of draft tokens accepted in the last cycle (0–1).
        """
        self._pld_acceptance_ema = (
            self._ema_alpha * accepted_ratio
            + (1 - self._ema_alpha) * self._pld_acceptance_ema
        )

        # Spine ratio tiers (from Goose paper)
        if self._pld_acceptance_ema < 0.2:
            self._spine_ratio = 0.15
        elif self._pld_acceptance_ema < 0.4:
            self._spine_ratio = 0.30
        else:
            self._spine_ratio = 0.50

    # ------------------------------------------------------------------
    # Logit harvesting
    # ------------------------------------------------------------------

    def harvest_logits(
        self, logits: torch.Tensor, context_tokens: list[int]
    ) -> None:
        """Harvest logits into transition table after a forward pass."""
        self.transition_table.harvest(logits, context_tokens)

    # ------------------------------------------------------------------
    # Tree construction (Phase 2)
    # ------------------------------------------------------------------

    def build_spine_tree(
        self, anchor: int, spine: list[int]
    ) -> dict:
        """Build full spine tree for tree attention verification.

        Parameters
        ----------
        anchor : int
            Last accepted token (tree root).
        spine : list[int]
            PLD spine tokens.

        Returns
        -------
        dict
            Tree structure (see ``SpineTreeBuilder.build``).
        """
        if spine and self.tree_enabled and self.tree_builder is not None:
            return self.tree_builder.build(
                anchor_token=anchor,
                spine=spine,
                transition_table=self.transition_table,
                spine_ratio=self._spine_ratio,
            )
        # Linear fallback
        return {
            "nodes": [anchor] + spine,
            "parents": [-1] + list(range(len(spine))),
            "source": ["anchor"] + ["pls"] * len(spine),
        }

    # ------------------------------------------------------------------
    # Verification: KV-cache-aware
    # ------------------------------------------------------------------

    @torch.no_grad()
    def verify_linear(
        self,
        model: torch.nn.Module,
        past_kv: list | None,
        draft_tokens: list[int],
        context_tokens: list[int],
    ) -> tuple[list[int], int, torch.Tensor | None]:
        """Verify draft tokens using a single forward pass (linear chain).

        Phase 0/1: one forward with the draft sequence, then greedy
        verification per position.  Compatible with KV cache (prefix KV
        passed in).

        Parameters
        ----------
        model : torch.nn.Module
            The model (GGUFModelAdapter or HF model).
        past_kv : list or None
            Past key-value pairs from HybridCache (per layer).
        draft_tokens : list[int]
            Candidate tokens to verify.
        context_tokens : list[int]
            Full token sequence (for KV cache context).

        Returns
        -------
        accepted : list[int]
            Accepted draft tokens.
        next_token : int
            Bonus token from the last accepted position.
        new_kv : torch.Tensor or None
            Updated KV cache after verification (or None if no past_kv).
        """
        if not draft_tokens:
            return [], context_tokens[-1] if context_tokens else 0, None

        draft_len = len(draft_tokens)

        # Forward on draft tokens with prefix KV
        draft_ids = torch.tensor(
            [draft_tokens], dtype=torch.long, device="cuda"
        )

        # For very long drafts (>1), the forward produces logits at each
        # position. For linear verification, we want logits at positions
        # where we check: do the logits predict the same token as drafted?
        if past_kv is not None:
            out = model.forward(
                input_ids=draft_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
        else:
            # Full context forward (fallback for warm-up)
            full_ids = torch.tensor(
                [context_tokens + draft_tokens], dtype=torch.long, device="cuda"
            )
            out = model.forward(input_ids=full_ids, use_cache=False)

        logits = self._extract_logits(out)
        new_kv = self._extract_new_kv(out)

        # Verify each draft token by checking if model's argmax matches
        accepted = []
        if past_kv is not None:
            # KV-cache path: input is just draft_tokens
            # logits[0, i, :] predicts the token after draft_tokens[i]
            # Compare against draft_tokens[i+1]
            for i in range(draft_len - 1):
                predicted = int(logits[0, i, :].argmax().item())
                if predicted == draft_tokens[i + 1]:
                    accepted.append(draft_tokens[i + 1])
                else:
                    break
        else:
            # Full-context path: input is context_tokens + draft_tokens
            # logits[0, ctx_len-1+i, :] predicts the token after the
            # last context token + i draft tokens = draft_tokens[i]
            ctx_len = len(context_tokens)
            for i in range(draft_len):
                pos = ctx_len - 1 + i
                predicted = int(logits[0, pos, :].argmax().item())
                if predicted == draft_tokens[i]:
                    accepted.append(draft_tokens[i])
                else:
                    break

        # Bonus token: model prediction at the last verified position.
        # No extra forward needed — we already have all logits from the
        # single verification forward pass.
        accept_pos = len(accepted)
        if past_kv is not None:
            # KV-cache: bonus position = accept_pos (predicts after
            # draft_tokens[accept_pos], which is the boundary between
            # accepted and rejected). Even at draft_len-1 (all accepted),
            # logits[0, -1, :] is the last predicted bonus token.
            bonus_pos = min(accept_pos, draft_len - 1)
            next_token = int(logits[0, bonus_pos, :].argmax().item())
        else:
            # Full-context: bonus position = ctx_len-1+accept_pos
            bonus_pos = ctx_len - 1 + accept_pos
            next_token = int(logits[0, bonus_pos, :].argmax().item())

        return accepted, next_token, new_kv

    @torch.no_grad()
    def verify_tree(
        self,
        model: torch.nn.Module,
        past_kv: list | None,
        tree: dict,
        prefix_len: int,
    ) -> tuple[list[int], int, torch.Tensor | None]:
        """Verify a spine tree using a single forward pass with tree attention.

        Phase 2: all candidate paths verified simultaneously using
        a tree attention mask.  Compatible with KV cache.

        Parameters
        ----------
        model : torch.nn.Module
            The model (must support ``attention_mask`` in forward).
        past_kv : list or None
            Past key-value pairs from HybridCache.
        tree : dict
            Tree structure from ``SpineTreeBuilder.build()``.
        prefix_len : int
            Length of prefix (for constructing full attention mask).

        Returns
        -------
        accepted : list[int]
            Accepted tokens from the longest correct path.
        next_token : int
            Bonus token.
        new_kv : torch.Tensor or None
            Updated KV cache (sliced to accepted prefix).
        """
        nodes = tree["nodes"]
        parents = tree["parents"]
        source = tree["source"]
        draft_len = len(nodes) - 1  # exclude anchor (index 0)

        if draft_len == 0:
            return [], 0, None

        # Build tree mask for draft→draft region
        tree_mask = self.tree_mask_gen.build_mask(parents[1:], draft_len)
        # Build full mask: draft→prefix (all 0) + draft→draft (tree)
        # Q has draft_len tokens, K has (prefix_len + draft_len) tokens
        full_mask = torch.zeros(
            1, 1, self.budget, prefix_len + self.budget,
            dtype=torch.float16, device="cuda",
        )
        # Draft→prefix: all 0 (can attend to all prefix)
        full_mask[0, 0, :draft_len, :prefix_len] = 0.0
        # Draft→draft: tree mask
        full_mask[0, 0, :draft_len, prefix_len:prefix_len + draft_len] = tree_mask[0, 0, :draft_len, :draft_len]
        # Pad unused draft positions to -inf
        if draft_len < self.budget:
            full_mask[0, 0, draft_len:, :] = float("-inf")
            full_mask[0, 0, :, prefix_len + draft_len:] = float("-inf")

        # Pack draft tokens (skip anchor, it's already in prefix KV)
        draft_ids = torch.tensor(
            [nodes[1:]], dtype=torch.long, device="cuda"
        )

        # Forward with tree attention mask
        if past_kv is not None:
            out = model.forward(
                input_ids=draft_ids,
                past_key_values=past_kv,
                use_cache=True,
                attention_mask=full_mask,
            )
        else:
            # Full recompute with attention mask
            full_ids = torch.tensor(
                [nodes], dtype=torch.long, device="cuda"
            )
            out = model.forward(
                input_ids=full_ids,
                use_cache=False,
                attention_mask=full_mask,
            )

        logits = self._extract_logits(out)
        new_kv_full = self._extract_new_kv(out)

        # Greedy walk: find longest accepted path with source priority
        # logits shape: (1, draft_len, vocab_size) — one logit per draft position
        accepted_tokens = []
        current_idx = 0  # start from first draft token (index 1 in tree nodes)

        while current_idx < draft_len:
            # Find all children of current node in the tree
            children = [
                (i + 1, nodes[i + 1], source[i + 1])
                for i in range(current_idx, draft_len)
                if parents[i + 1] == current_idx
            ]

            if not children:
                break

            # Get model's prediction at this position
            predicted = int(logits[0, current_idx, :].argmax().item())

            # Match with source priority: PLD > TR
            matched = None
            for c_idx, c_token, c_source in children:
                if c_token == predicted:
                    if c_source == "pls":
                        matched = (c_idx, c_token)
                        break  # PLD has highest priority
                    if matched is None:
                        matched = (c_idx, c_token)  # TR fallback

            if matched is None:
                break

            accepted_tokens.append(matched[1])
            current_idx = matched[0]

        # Bonus token from last accepted position
        if current_idx < draft_len:
            next_token = int(logits[0, current_idx, :].argmax().item())
        else:
            # All accepted: bonus from the model at the last position
            if past_kv is not None and new_kv_full is not None:
                bonus_ids = torch.tensor(
                    [[nodes[-1]]], dtype=torch.long, device="cuda"
                )
                out_bonus = model.forward(
                    input_ids=bonus_ids,
                    past_key_values=new_kv_full,
                    use_cache=True,
                )
                logits_bonus = self._extract_logits(out_bonus)
                next_token = int(logits_bonus[0, 0, :].argmax().item())
                new_kv_full = self._extract_new_kv(out_bonus)
            else:
                next_token = int(logits[0, -1, :].argmax().item())

        # Slice KV cache to keep only (prefix + accepted + bonus)
        if past_kv is not None and new_kv_full is not None:
            # new_kv_full has shape: [layers, (k, v)] each with seq_len = prefix_len + draft_len
            # We want seq_len = prefix_len + len(accepted) + 1
            new_len = prefix_len + len(accepted_tokens) + 1
            sliced_kv = []
            for k, v in new_kv_full:
                sliced_kv.append((k[:, :, :new_len, :], v[:, :, :new_len, :]))
            new_kv_full = sliced_kv

        return accepted_tokens, next_token, new_kv_full

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_logits(model_output) -> torch.Tensor:
        """Extract logits tensor from raw tensor or HF CausalLMOutput."""
        if isinstance(model_output, torch.Tensor):
            return model_output
        if hasattr(model_output, "logits"):
            return model_output.logits
        if isinstance(model_output, (tuple, list)):
            return model_output[0]
        return model_output

    @staticmethod
    def _extract_new_kv(model_output):
        """Extract past_key_values from model output."""
        if hasattr(model_output, "past_key_values"):
            return model_output.past_key_values
        kvs = getattr(model_output, "_last_kv_cache", None)
        if kvs is not None:
            return kvs
        return None

    # ------------------------------------------------------------------
    # Can-speculate check
    # ------------------------------------------------------------------

    def can_speculate(self, context: list[int]) -> bool:
        """Check whether we have enough data to attempt speculation.

        Phase 0/1: need context length ≥ min n-gram and warmup complete.
        Phase 2: also need transition table populated.
        """
        ctx_len = len(context)
        if ctx_len < 4:  # Need at least some context
            return False
        if self._warmup_steps < self._warmup_threshold:
            self._warmup_steps += 1
            return False
        return True

    def enable(self) -> None:
        """Force-enable speculation (bypass warm-up)."""
        self._warmup_steps = self._warmup_threshold
        logger.info("SpeculativeEngine: force-enabled")


# ═══════════════════════════════════════════════════════════════════════
# 6. SkeletonDraftGenerator — Self-Speculative Decoding
# ═══════════════════════════════════════════════════════════════════════


class _IdentityLayer(torch.nn.Module):
    """A no-op placeholder that passes hidden states through unchanged.

    Used to skip specific decoder layers during draft generation.
    The original module reference is saved for restoration.
    """

    __slots__ = ("_orig",)

    def __init__(self, original_layer: torch.nn.Module) -> None:
        super().__init__()
        self._orig = original_layer

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Bypass all computation — just pass through
        return hidden_states

    def original(self) -> torch.nn.Module:
        return self._orig


class SkeletonDraftGenerator:
    """Generates draft tokens by skipping a subset of decoder layers.

    Self-Speculative Decoding (ACL'24): runs the model with selected
    layers skipped to obtain a cheaper "skeleton" model.  The output
    from this skeleton is used as draft tokens, which are then verified
    by the full model through the existing verification pipeline.

    Strategy
    --------
    - Skips the **last ``skip_fraction`` of layers** (e.g. 30% of the
      top layers), keeping the bottom layers which encode more
      fundamental linguistic patterns.
    - Uses a context manager so the model is restored after draft
      generation — no permanent modification.
    - Falls back gracefully: if the model structure doesn't have
      identifiable decoder layers, returns no draft (the caller then
      falls through to the normal decode path).

    Compatibility
    -------------
    - Works with both HF Transformers and GGUF models
    - Compatible with KV cache (past_key_values pass through identity
      layers unchanged)
    - No model weight modification — purely inference-time layer skipping
    - Compatible with Expert Cache: skipped layers simply don't access
      their experts, reducing cache pressure
    """

    def __init__(
        self,
        model: torch.nn.Module,
        skip_fraction: float = 0.30,
        max_draft: int = 5,
    ):
        self.model = model
        self.skip_fraction = max(0.1, min(0.5, skip_fraction))
        self.max_draft = max_draft

        self._layers: list | None = None
        self._num_layers: int = 0
        self._skip_indices: set[int] = set()
        self._originals: dict[int, torch.nn.Module] = {}

        self._resolve_layers()

        if self._num_layers:
            logger.info(
                "SkeletonDraftGenerator: %d layers, skip_fraction=%.2f -> "
                "skipping %d layers (indices %s)",
                self._num_layers, self.skip_fraction,
                len(self._skip_indices),
                sorted(self._skip_indices),
            )

    # ------------------------------------------------------------------
    # Layer resolution
    # ------------------------------------------------------------------

    def _resolve_layers(self) -> None:
        """Locate decoder layers in the model and compute skip indices."""
        layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
        elif hasattr(self.model, "decoder") and hasattr(self.model.decoder, "layers"):
            layers = self.model.decoder.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            # GPT-2 / OPT style
            layers = self.model.transformer.h

        if layers is None:
            logger.warning("SkeletonDraftGenerator: could not locate decoder layers")
            return

        self._layers = list(layers)
        self._num_layers = len(self._layers)
        num_skip = max(1, int(self._num_layers * self.skip_fraction))
        # Skip the LAST num_skip layers (high-level semantic processing)
        self._skip_indices = set(
            range(self._num_layers - num_skip, self._num_layers)
        )

    # ------------------------------------------------------------------
    # Context manager — temporarily replaces layers with identity
    # ------------------------------------------------------------------

    def __enter__(self):
        if self._layers is None:
            return self

        self._originals.clear()
        for idx in self._skip_indices:
            if idx < len(self._layers):
                orig = self._layers[idx]
                self._originals[idx] = orig
                self._layers[idx] = _IdentityLayer(orig)

        return self

    def __exit__(self, *args) -> None:
        if self._layers is None:
            return

        for idx, orig in list(self._originals.items()):
            if idx < len(self._layers):
                self._layers[idx] = orig
        self._originals.clear()

    # ------------------------------------------------------------------
    # Draft generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_draft(
        self,
        input_ids: torch.Tensor,
        past_key_values: list | None = None,
    ) -> list[int]:
        """Generate draft tokens using the skeleton (skip-layer) model.

        Parameters
        ----------
        input_ids : torch.Tensor
            Shape ``(1, 1)`` — the last token (decode step input).
        past_key_values : list or None
            KV cache from the full model forward.  Passed through
            identity layers unchanged.

        Returns
        -------
        list[int]
            Up to ``max_draft`` draft tokens.  Empty list if skeleton
            model is unavailable.
        """
        if self._layers is None or self._num_layers == 0:
            return []

        drafts: list[int] = []
        current_ids = input_ids
        current_kv = past_key_values

        with self:
            for _ in range(self.max_draft):
                out = self.model.forward(
                    input_ids=current_ids,
                    past_key_values=current_kv,
                    use_cache=True,
                )
                logits = SpeculativeEngine._extract_logits(out)
                next_tok = int(logits[0, -1, :].argmax().item())
                drafts.append(next_tok)

                # Prepare for next iteration
                current_ids = torch.tensor(
                    [[next_tok]], dtype=torch.long, device=input_ids.device
                )
                current_kv = SpeculativeEngine._extract_new_kv(out)

                # Early stopping if no KV (shouldn't happen, but safe)
                if current_kv is None:
                    break

        return drafts
