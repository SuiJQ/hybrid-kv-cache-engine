# long_context/tests.py
"""
Tests for the long context extension module.

Run with:
    python -m pytest long_context/tests.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from long_context.config import LongContextConfig, ContextMethod
from long_context.self_extend import SelfExtendWrapper, get_self_extend_position_ids
from long_context.re_attention import ReAttentionWrapper, dispatch_reattention
from long_context.integration import (
    LongContextAttentionInjector,
    apply_rotary_emb,
    _compute_default_cos_sin,
)

# ═══════════════════════════════════════════════════════════════════════
# Config tests
# ═══════════════════════════════════════════════════════════════════════


class TestLongContextConfig:
    def test_default_config(self):
        cfg = LongContextConfig()
        assert cfg.enabled is True  # auto-enabled by default
        assert cfg.method == "selfextend"
        assert cfg.neighbor_window == 1024
        assert cfg.group_size == 8

    def test_selfextend_activation(self):
        cfg = LongContextConfig(method="selfextend", enabled=True)
        assert cfg.enabled
        assert cfg.method == "selfextend"

    def test_reattention_activation(self):
        cfg = LongContextConfig(method="reattention", enabled=True)
        assert cfg.enabled
        assert cfg.method == "reattention"

    def test_repr(self):
        cfg = LongContextConfig(method="selfextend", enabled=True)
        r = repr(cfg)
        assert "selfextend" in r
        assert "NW=1024" in r
        assert "GS=8" in r


# ═══════════════════════════════════════════════════════════════════════
# SelfExtend tests
# ═══════════════════════════════════════════════════════════════════════


class TestSelfExtend:
    def test_get_grouped_position_ids_basic(self):
        """Verify the core 4-line logic produces correct grouped positions."""
        pos = torch.arange(12).unsqueeze(0)  # (1, 12)
        grouped = get_self_extend_position_ids(pos, neighbor_window=4, group_size=2, short_threshold=1)
        expected = torch.tensor([[0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7]])
        assert grouped is not None
        assert torch.equal(grouped, expected), f"{grouped} != {expected}"

    def test_get_grouped_position_ids_large_group(self):
        pos = torch.arange(20).unsqueeze(0)
        grouped = get_self_extend_position_ids(pos, neighbor_window=8, group_size=4, short_threshold=1)
        assert grouped is not None
        # Positions 0-7 keep originals, 8-19 get (pos-8)//4 + 8
        expected = torch.arange(20).unsqueeze(0)
        expected[0, 8:] = (torch.arange(12) // 4) + 8
        assert torch.equal(grouped, expected), f"{grouped} != {expected}"

    def test_no_grouping_below_neighbor_window(self):
        pos = torch.arange(5).unsqueeze(0)
        grouped = get_self_extend_position_ids(pos, neighbor_window=1024, group_size=8)
        assert grouped is None

    def test_no_grouping_below_threshold(self):
        pos = torch.arange(500).unsqueeze(0)
        grouped = get_self_extend_position_ids(
            pos, neighbor_window=128, group_size=8, short_threshold=2048
        )
        assert grouped is None

    def test_wrapper_group_position_ids(self):
        cfg = LongContextConfig(
            method="selfextend", enabled=True,
            neighbor_window=4, group_size=2,
            short_context_threshold=1,  # disable short-circuit
        )
        wrapper = SelfExtendWrapper(cfg)
        pos = torch.arange(12).unsqueeze(0)
        grouped = wrapper.group_position_ids(pos)
        assert grouped is not None
        assert grouped.shape == (1, 12)

    def test_wrapper_blend(self):
        cfg = LongContextConfig(method="selfextend", enabled=True)
        wrapper = SelfExtendWrapper(cfg)
        seq_len = 100
        grouped_out = torch.randn(1, 8, seq_len, 64)
        neighbor_out = torch.randn(1, 8, seq_len, 64)
        blended = wrapper.blend(grouped_out, neighbor_out, seq_len)
        assert blended.shape == (1, 8, seq_len, 64)
        # Last neighbor_window tokens should be from neighbor_out
        split = seq_len - wrapper.neighbor_window
        assert torch.equal(blended[:, :, split:, :], neighbor_out[:, :, split:, :])


# ═══════════════════════════════════════════════════════════════════════
# ReAttention tests
# ═══════════════════════════════════════════════════════════════════════


class TestReAttention:
    def test_should_activate(self):
        cfg = LongContextConfig(method="reattention", enabled=True)
        ra = ReAttentionWrapper(cfg)
        assert ra.should_activate(1, 10000)
        assert not ra.should_activate(1, 32)  # below neighbor_window
        assert not ra.should_activate(1, 1000)  # below short_threshold

    def test_retrieve(self):
        cfg = LongContextConfig(
            method="reattention", enabled=True,
            reattn_top_k=64, reattn_top_k_ratio=0.1,
            reattn_min_top_k=8,
        )
        ra = ReAttentionWrapper(cfg)
        B, H, T_q, T_kv, D = 1, 8, 4, 16, 64
        q = torch.randn(B, H, T_q, D)
        k = torch.randn(B, 2, T_kv, D)
        top_k_idx, actual_k = ra.retrieve(q, k, T_kv)
        # Should pick min(ceil(16*0.1)=2, 64) = 2, clamped to min 8, min 16
        assert actual_k >= 8
        assert actual_k <= 16
        assert top_k_idx.shape == (B, T_q, actual_k)

    def test_build_mask(self):
        cfg = LongContextConfig(method="reattention", enabled=True)
        ra = ReAttentionWrapper(cfg)
        B, T_q, T_kv = 1, 2, 20
        top_k_idx = torch.randint(0, T_kv, (B, T_q, 5))
        mask = ra.build_mask(top_k_idx, T_q, T_kv, "cpu")
        assert mask.shape == (1, 1, T_q, T_kv)
        assert mask.dtype == torch.float16
        # Should have non-inf entries
        assert (mask > float("-inf")).any()


# ═══════════════════════════════════════════════════════════════════════
# Integration tests
# ═══════════════════════════════════════════════════════════════════════


class TestRoPE:
    def test_apply_rotary_emb(self):
        B, H, T, D = 1, 8, 10, 64
        x = torch.randn(B, H, T, D)
        cos, sin = _compute_default_cos_sin(D, 100, "cpu", torch.float32)
        pos = torch.arange(T).unsqueeze(0)
        result = apply_rotary_emb(x, cos, sin, pos)
        assert result.shape == (B, H, T, D)

    def test_cos_sin_cache(self):
        D = 64
        cos, sin = _compute_default_cos_sin(D, 100, "cpu", torch.float32)
        assert cos.shape[1] == 100  # max_seq_len
        assert cos.shape[-1] == D

    def test_cos_sin_yar_scale(self):
        """Verify YaRN-style base frequency scaling works."""
        D = 64
        factor = 8.0
        base = 10000.0 * (factor ** (D / (D - 2)))
        cos, sin = _compute_default_cos_sin(D, 4 * 4096, "cpu", torch.float32, base)
        assert cos.shape[0] == 1


class TestAttentionInjector:
    def test_injector_creation(self):
        cfg = LongContextConfig(method="selfextend", enabled=True)
        injector = LongContextAttentionInjector(cfg)
        injector._init_wrappers()
        assert injector.method == "selfextend"
        assert injector._self_extend is not None

    def test_injector_innit_wrappers(self):
        cfg = LongContextConfig(method="reattention", enabled=True)
        injector = LongContextAttentionInjector(cfg)
        injector._init_wrappers()
        assert injector._re_attention is not None

    def test_disabled_injector(self):
        cfg = LongContextConfig(method="none", enabled=False)
        injector = LongContextAttentionInjector(cfg)
        assert injector.method == "none"


# ═══════════════════════════════════════════════════════════════════════
# Benchmark: integration test for SelfExtend + ReAttention compat
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("method", ["selfextend", "reattention"])
def test_method_creates_valid_config(method):
    """Ensure both methods produce valid, runnable configs."""
    cfg = LongContextConfig(method=method, enabled=True)
    assert cfg.enabled
    assert cfg.method == method

    if method == "selfextend":
        wrapper = SelfExtendWrapper(cfg)
        pos = torch.arange(4096).unsqueeze(0)
        grouped = wrapper.group_position_ids(pos)
        assert grouped is not None
        # Maximum grouped position should be within reason
        max_pos = grouped.max().item()
        assert max_pos < 4096 * 2  # not exploding
    elif method == "reattention":
        ra = ReAttentionWrapper(cfg)
        B, H_q, H_kv, T_q, T_kv, D = 1, 8, 2, 4, 16384, 64
        q = torch.randn(B, H_q, T_q, D)
        k = torch.randn(B, H_kv, T_kv, D)
        v = torch.randn(B, H_kv, T_kv, D)
        scale = D ** -0.5
        out = dispatch_reattention(ra, q, k, v, scale, causal=True, kv_len=T_kv)
        assert out.shape == (B, H_q, T_q, D)
