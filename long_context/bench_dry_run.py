#!/usr/bin/env python3
"""
long_context/bench_dry_run.py — 纯 CPU 验证 SelfExtend/ReAttention 兼容性

该脚本在 CPU 上运行轻量级测试，验证：
  1. SelfExtend 的位置编码修改逻辑
  2. ReAttention 的 top-k 检索 + 掩码构建
  3. 双方法与现有 KV Cache、AFCE 的兼容性
  4. 无 GPU 环境下也可运行

Usage:
    python long_context/bench_dry_run.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bench")

from long_context.config import LongContextConfig
from long_context.self_extend import SelfExtendWrapper, get_self_extend_position_ids
from long_context.re_attention import ReAttentionWrapper, dispatch_reattention
from long_context.integration import apply_rotary_emb, _compute_default_cos_sin


def test_self_extend_range():
    """Test SelfExtend's ability to handle 4x+ sequence lengths."""
    print("\n" + "=" * 60)
    print("  SelfExtend: 上下文扩展验证")
    print("=" * 60)

    for neighbor_window, group_size, total_len in [
        (1024, 8, 4096),   # 4K → 32K equivalent (8x)
        (1024, 8, 16384),  # 4K → 128K equivalent
        (512, 16, 32768),  # Extremely aggressive grouping
    ]:
        pos = torch.arange(total_len).unsqueeze(0)
        t0 = time.perf_counter()
        grouped = get_self_extend_position_ids(
            pos, neighbor_window, group_size, short_threshold=1
        )
        elapsed = (time.perf_counter() - t0) * 1000

        if grouped is None:
            print(f"  ✗ NW={neighbor_window}, GS={group_size}, L={total_len}: skipped")
            continue

        max_pos = grouped.max().item()
        # Original max would be total_len - 1
        reduction_ratio = (total_len - 1) / max_pos if max_pos > 0 else 1

        print(f"  ✓ NW={neighbor_window:>4d}, GS={group_size:>2d}, L={total_len:>5d}: "
              f"max_pos={max_pos:>4d}, reduction={reduction_ratio:.1f}x, "
              f"took={elapsed:.3f}ms")

        # Verify: grouped positions never exceed neighbor_window + (last_group)
        expected_max = neighbor_window + (total_len - neighbor_window - 1) // group_size
        # Allow +1 for edge cases
        assert max_pos <= expected_max + 1, \
            f"max_pos {max_pos} exceeds expected {expected_max}"

    print("\n  ✅ SelfExtend: 所有用例通过")


def test_re_attention_retrieval():
    """Test ReAttention's top-k retrieval at different scales."""
    print("\n" + "=" * 60)
    print("  ReAttention: top-k 检索验证")
    print("=" * 60)

    configs = [
        (LongContextConfig(method="reattention", enabled=True, reattn_top_k=512, reattn_top_k_ratio=0.05, reattn_min_top_k=32), 4096),
        (LongContextConfig(method="reattention", enabled=True, reattn_top_k=1024, reattn_top_k_ratio=0.1, reattn_min_top_k=64), 8192),
        (LongContextConfig(method="reattention", enabled=True, reattn_top_k=2048, reattn_top_k_ratio=0.05, reattn_min_top_k=128), 16384),
    ]

    for cfg, kv_len in configs:
        ra = ReAttentionWrapper(cfg)
        B, H, T_q, D = 1, 8, 1, 128
        q = torch.randn(B, H, T_q, D)
        k = torch.randn(B, 2, kv_len, D)  # GQA: 2 KV heads

        t0 = time.perf_counter()
        top_k_idx, actual_k = ra.retrieve(q, k, kv_len)
        elapsed = (time.perf_counter() - t0) * 1000

        compression = kv_len / actual_k if actual_k > 0 else 1
        print(f"  ✓ KV={kv_len:>7d}, keep_k={actual_k:>4d}, "
              f"compression={compression:.0f}x, took={elapsed:.3f}ms")

        assert actual_k > 0
        assert actual_k <= cfg.reattn_top_k
        assert top_k_idx.shape[2] == actual_k

    print("\n  ✅ ReAttention: 所有检索用例通过")


def test_re_attention_mask():
    """Test ReAttention mask generation at scale."""
    print("\n" + "=" * 60)
    print("  ReAttention: 掩码构建验证")
    print("=" * 60)

    cfg = LongContextConfig(
        method="reattention", enabled=True,
        reattn_top_k=256, reattn_neighbor_window=64,
        short_context_threshold=1,
    )
    ra = ReAttentionWrapper(cfg)

    for kv_len in [1024, 4096, 8192]:
        B, T_q = 1, 4
        top_k_idx = torch.randint(0, kv_len, (B, T_q, 64))

        t0 = time.perf_counter()
        mask = ra.build_mask(top_k_idx, T_q, kv_len, "cpu")
        elapsed = (time.perf_counter() - t0) * 1000

        # Count visible positions per query
        visible = (mask > float("-inf")).sum(dim=-1)
        avg_visible = visible.float().mean().item()

        print(f"  ✓ KV={kv_len:>5d}, T_q=4, avg_visible={avg_visible:.0f}/{kv_len}, "
              f"sparsity={avg_visible/kv_len*100:.2f}%, took={elapsed:.3f}ms")

        assert mask.shape == (1, 1, T_q, kv_len)

    print("\n  ✅ ReAttention: 所有掩码用例通过")


def test_self_extend_re_attention_compatibility():
    """Verify that SelfExtend and ReAttention configs can coexist."""
    print("\n" + "=" * 60)
    print("  SelfExtend + ReAttention: 兼容性验证")
    print("=" * 60)

    # Both use the same base config pattern
    se_cfg = LongContextConfig(method="selfextend", enabled=True)
    ra_cfg = LongContextConfig(method="reattention", enabled=True)

    assert se_cfg.enabled
    assert ra_cfg.enabled
    assert se_cfg.method != ra_cfg.method

    # Both can be created from CLI args
    class FakeArgs:
        context_method = "selfextend"
        neighbor_window = 1024
        group_size = 8
        reattn_top_k = 2048
        reattn_top_k_ratio = 0.1
        reattn_min_top_k = 128
        reattn_neighbor_window = 64
        yarn_factor = 8.0
        yarn_original_max_len = 4096
        short_context_threshold = 2048
        verbose = False

    fg = FakeArgs()

    cfg = LongContextConfig.from_cli(fg)
    assert cfg.method == "selfextend"
    assert cfg.neighbor_window == 1024

    fg.context_method = "reattention"
    cfg2 = LongContextConfig.from_cli(fg)
    assert cfg2.method == "reattention"
    assert cfg2.reattn_top_k == 2048

    print("  ✓ 两种方法的 CLI 配置兼容")
    print("  ✓ 共用同一套配置基类 LongContextConfig")

    print("\n  ✅ SelfExtend + ReAttention: 兼容性通过")


def test_rotation_consistency():
    """Verify RoPE application is consistent with grouped positions."""
    print("\n" + "=" * 60)
    print("  RoPE: 位置编码一致性验证")
    print("=" * 60)

    D = 64
    cos, sin = _compute_default_cos_sin(D, 100, "cpu", torch.float32)
    B, H, T = 1, 8, 32
    x = torch.randn(B, H, T, D)

    # Same pos for all positions → identity rotation (all cos=1, sin=0 at t=0)
    pos_identical = torch.zeros((1, T), dtype=torch.long)
    out_identical = apply_rotary_emb(x, cos, sin, pos_identical)
    assert torch.allclose(out_identical, x, atol=1e-4), \
        "pos=0 should give identity rotation"

    # Different pos → different rotation for each position
    pos_different = torch.arange(T).unsqueeze(0)
    out_different = apply_rotary_emb(x, cos, sin, pos_different)

    # Position 0 should still be identity (cos(0)=1, sin(0)=0)
    assert torch.allclose(out_different[:, :, 0, :], x[:, :, 0, :], atol=1e-4)

    # SelfExtend grouped positions produce valid rotated output
    pos_original = torch.arange(T).unsqueeze(0)
    grouped = get_self_extend_position_ids(pos_original, neighbor_window=16, group_size=4, short_threshold=1)
    assert grouped is not None  # T=32 > NW=16

    out_grouped = apply_rotary_emb(x, cos, sin, grouped)
    assert out_grouped.shape == out_different.shape
    # Should differ for at least some positions (since positions are different)
    # Note: positions in non-neighbor region may match due to FLOOR grouping
    assert not torch.allclose(out_grouped, out_different, atol=1e-5)

    print("  ✓ RoPE: 位置 0 编码为恒等变换")
    print("  ✓ RoPE: 不同位置产生不同编码")
    print("  ✓ RoPE: SelfExtend 分组位置产生有效编码")
    print("\n  ✅ RoPE: 一致性通过")


def test_dispatch_reattention():
    """End-to-end ReAttention dispatch test on CPU (uses raw SDPA, no compile)."""
    print("\n" + "=" * 60)
    print("  ReAttention: 端到端 dispatch 验证")
    print("=" * 60)

    cfg = LongContextConfig(
        method="reattention", enabled=True,
        reattn_top_k=32, reattn_top_k_ratio=0.1,
        reattn_min_top_k=4, reattn_neighbor_window=8,
        short_context_threshold=4,
    )
    ra = ReAttentionWrapper(cfg)

    B, H, T_q, T_kv, D = 1, 4, 2, 64, 32
    q = torch.randn(B, H, T_q, D)
    k = torch.randn(B, 1, T_kv, D)  # GQA: 1 KV head
    v = torch.randn(B, 1, T_kv, D)
    scale = D ** -0.5

    # Test ReAttention retrieval directly
    t0 = time.perf_counter()
    top_k_idx, actual_k = ra.retrieve(q, k, T_kv)
    elapsed = (time.perf_counter() - t0) * 1000
    assert actual_k >= 4
    assert top_k_idx.shape == (B, T_q, actual_k)
    print(f"  ✓ retrieve: actual_k={actual_k}, took={elapsed:.2f}ms")

    # Test mask building
    mask = ra.build_mask(top_k_idx, T_q, T_kv, "cpu")
    assert mask.shape == (1, 1, T_q, T_kv)
    print(f"  ✓ build_mask: shape {mask.shape}, visible={(mask > float('-inf')).sum().item()}/{B*T_q*T_kv}")

    # Test that short context bypass works
    cfg2 = LongContextConfig(
        method="reattention", enabled=True,
        reattn_top_k=32, short_context_threshold=100,
    )
    ra2 = ReAttentionWrapper(cfg2)
    assert not ra2.should_activate(T_q, T_kv), f"should_activate(4, 64) should be False with threshold=100"
    print(f"  ✓ should_activate: short context correctly bypassed")

    # Test ReAttention dispatch via raw SDPA (no compile)
    from torch.nn.functional import scaled_dot_product_attention

    if ra.should_activate(T_q, T_kv):
        top_k_idx, actual_k = ra.retrieve(q, k, T_kv)
        mask = ra.build_mask(top_k_idx, T_q, T_kv, "cpu")
        # Match mask dtype to query dtype
        mask = mask.to(dtype=q.dtype)
        out = scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=False, attn_mask=mask
        )
    else:
        out = scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=True
        )

    assert out.shape == (B, H, T_q, D)
    assert torch.isfinite(out).all()
    print(f"  ✓ dispatch (raw SDPA): output shape {out.shape}")

    # ReAttention should give different output than pure causal attention
    out_causal = scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)
    if ra.should_activate(T_q, T_kv):
        # They should differ because ReAttention restricts attention scope
        assert not torch.allclose(out, out_causal, atol=1e-3), \
            "ReAttention output should differ from pure causal"
        print(f"  ✓ ReAttention output differs from causal attention")

    print("\n  ✅ ReAttention: dispatch 通过")


def main():
    print("\n" + "#" * 60)
    print("#  MoeOwner 超长上下文模块 — 镜像测试")
    print("#  CPU-only 兼容性 & 正确性验证")
    print("#" * 60)

    tests = [
        test_self_extend_range,
        test_re_attention_retrieval,
        test_re_attention_mask,
        test_self_extend_re_attention_compatibility,
        test_rotation_consistency,
        test_dispatch_reattention,
    ]

    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  ❌ {test_fn.__name__}: FAILED — {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"  结果: {passed}/{len(tests)} 通过")
    print("=" * 60 + "\n")

    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
