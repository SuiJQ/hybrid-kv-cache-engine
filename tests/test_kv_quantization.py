"""
test_kv_quantization.py — Round-trip correctness tests for KV asymmetric
quantization (Key→INT8, Value→INT4 packed) and head/tail FP16 protection.

Runs entirely on CPU (no CUDA required).
"""

import torch

from cache_manager import HybridCache

# ===================================================================
# Helper
# ===================================================================

def _make_kv(batch=1, heads=4, seq_len=32, dim=64, seed=42):
    torch.manual_seed(seed)
    k = torch.randn(batch, heads, seq_len, dim, dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, dim, dtype=torch.float16)
    return k, v


# ===================================================================
# Quantization kernel tests
# ===================================================================

class TestQuantizeKeyINT8:
    """Key INT8 symmetric quantisation — per-head scale."""

    def test_round_trip_accuracy(self):
        h = HybridCache(block_size=16, hidden_size=64, total_blocks=10)
        k, _ = _make_kv()
        k_q, scale = h._quantize_k_int8(k)
        k_deq = h._dequantize_k_int8(k_q, scale)
        err = (k - k_deq).abs().max().item()
        assert err < 0.05, f"Key INT8 max err={err} too high"

    def test_int8_dtype(self):
        h = HybridCache(block_size=16, hidden_size=64, total_blocks=10)
        k, _ = _make_kv()
        k_q, _ = h._quantize_k_int8(k)
        assert k_q.dtype == torch.int8, f"Expected int8, got {k_q.dtype}"

    def test_scale_shape(self):
        h = HybridCache(block_size=16, hidden_size=64, total_blocks=10)
        k, _ = _make_kv(seq_len=48)
        _, scale = h._quantize_k_int8(k)
        assert scale.shape == (1, 4, 1, 1), f"Unexpected scale shape: {scale.shape}"
        assert scale.dtype == torch.float16

    def test_zero_input(self):
        h = HybridCache(block_size=16, hidden_size=64, total_blocks=10)
        k = torch.zeros(1, 4, 32, 64, dtype=torch.float16)
        k_q, scale = h._quantize_k_int8(k)
        # All zeros → scale clamped to 1e-8, all INT8 values should be 0
        assert (k_q == 0).all(), "Zero input should quantize to all zeros"
        k_deq = h._dequantize_k_int8(k_q, scale)
        assert (k_deq == 0).all(), "Zero dequant should be zeros"


class TestQuantizeValueINT4:
    """Value INT4 packed quantisation — per-head symmetric + bias."""

    def test_round_trip_accuracy(self):
        h = HybridCache(block_size=16, hidden_size=64, total_blocks=10)
        _, v = _make_kv()
        packed, scale, bias = h._quantize_v_int4(v)
        v_deq = h._dequantize_v_int4(packed, scale, bias)
        err = (v - v_deq).abs().max().item()
        assert err < 0.5, f"Value INT4 max err={err} too high"

    def test_packed_shape(self):
        h = HybridCache(block_size=16, hidden_size=64, total_blocks=10)
        _, v = _make_kv(dim=64)
        packed, _, _ = h._quantize_v_int4(v)
        assert packed.dtype == torch.uint8
        assert packed.shape == (1, 4, 32, 32), f"Unexpected packed shape: {packed.shape}"

    def test_negative_values(self):
        """INT4 symmetric quant handles negative values correctly."""
        h = HybridCache(block_size=16, hidden_size=64, total_blocks=10)
        # Deliberately negative-heavy values
        v = -torch.rand(1, 4, 32, 64, dtype=torch.float16) * 2.0
        packed, scale, bias = h._quantize_v_int4(v)
        v_deq = h._dequantize_v_int4(packed, scale, bias)
        err = (v - v_deq).abs().max().item()
        assert err < 0.5, f"Negative Value INT4 max err={err}"

    def test_pack_unpack_bit_exact(self):
        """Verify nibble packing/unpacking is bit-exact."""
        h = HybridCache(block_size=16, hidden_size=64, total_blocks=10)
        _, v = _make_kv()
        packed, scale, bias = h._quantize_v_int4(v)
        v_deq = h._dequantize_v_int4(packed, scale, bias)
        # Re-quantize: should get identical packed bits (deterministic)
        packed2, _, _ = h._quantize_v_int4(v_deq)
        assert (packed == packed2).all(), "Packing not deterministic"


# ===================================================================
# Store / Load integration tests
# ===================================================================

class TestStoreLoad:
    """HybridCache.store_kv / load_kv with quantisation."""

    def test_fp16_mode_short(self):
        """seq_len <= 16 → store as exact FP16."""
        h = HybridCache(block_size=16, hidden_size=64, total_blocks=10)
        k, v = _make_kv(seq_len=12)
        h.store_kv(0, [(k, v)])
        loaded = h.load_kv(0)
        lk, lv = loaded[0]
        assert (k - lk).abs().max().item() < 1e-6
        assert (v - lv).abs().max().item() < 1e-6

    def test_fp16_mode_exact_16(self):
        """seq_len == 2*PROTECTED_N → exact FP16 boundary case."""
        h = HybridCache(block_size=16, hidden_size=64, total_blocks=10)
        k, v = _make_kv(seq_len=16)
        h.store_kv(0, [(k, v)])
        loaded = h.load_kv(0)
        assert (k - loaded[0][0]).abs().max().item() < 1e-6

    def test_mixed_mode(self):
        """seq_len > 16 → head+tail FP16, body quantized."""
        h = HybridCache(block_size=16, hidden_size=64, total_blocks=10)
        k, v = _make_kv(seq_len=32)
        h.store_kv(0, [(k, v)])
        loaded = h.load_kv(0)
        lk, lv = loaded[0]
        assert lk.shape == (1, 4, 32, 64)
        assert lv.shape == (1, 4, 32, 64)
        err_k = (k - lk).abs().max().item()
        err_v = (v - lv).abs().max().item()
        assert err_k < 0.05, f"Mixed K error={err_k}"
        assert err_v < 0.5, f"Mixed V error={err_v}"

    def test_mixed_body_len_1(self):
        """seq_len = 17 (body has just 1 token)."""
        h = HybridCache(block_size=16, hidden_size=64, total_blocks=10)
        k, v = _make_kv(seq_len=17)
        h.store_kv(0, [(k, v)])
        loaded = h.load_kv(0)
        assert loaded[0][0].shape == (1, 4, 17, 64)

    def test_multi_layer(self):
        """All decoder layers are quantized independently."""
        h = HybridCache(block_size=16, hidden_size=64, total_blocks=10)
        layers = []
        for i in range(3):
            k, v = _make_kv(seq_len=24, seed=42 + i)
            layers.append((k, v))
        h.store_kv(0, layers)
        loaded = h.load_kv(0)
        assert len(loaded) == 3
        for i in range(3):
            e_k = (layers[i][0] - loaded[i][0]).abs().max().item()
            e_v = (layers[i][1] - loaded[i][1]).abs().max().item()
            assert e_k < 0.05, f"Layer {i} K error={e_k}"
            assert e_v < 0.5, f"Layer {i} V error={e_v}"

    def test_overwrite(self):
        """Store to existing block replaces content."""
        h = HybridCache(block_size=16, hidden_size=64, total_blocks=10)
        k1, v1 = _make_kv(seq_len=20)
        k2, v2 = _make_kv(seq_len=30, seed=99)
        h.store_kv(0, [(k1, v1)])
        h.store_kv(0, [(k2, v2)])
        loaded = h.load_kv(0)
        assert loaded[0][0].shape == (1, 4, 30, 64)

    def test_empty_store_deletes(self):
        """Storing empty list removes block from cache."""
        h = HybridCache(block_size=16, hidden_size=64, total_blocks=10)
        k, v = _make_kv(seq_len=20)
        h.store_kv(0, [(k, v)])
        assert h.load_kv(0) is not None
        h.store_kv(0, [])
        assert h.load_kv(0) is None

    def test_free_clears_kv(self):
        """free_block clears quantized storage."""
        h = HybridCache(block_size=16, hidden_size=64, total_blocks=10)
        b = h.allocate([101, 102, 103])
        k, v = _make_kv(seq_len=18)
        h.store_kv(b.block_id, [(k, v)])
        h.free_block(b.block_id)
        assert h.load_kv(b.block_id) is None

    def test_none_block_id(self):
        """load_kv returns None for unknown block."""
        h = HybridCache(block_size=16, hidden_size=64, total_blocks=10)
        assert h.load_kv(999) is None

    def test_large_head_dim(self):
        """head_dim=128 (common in larger models)."""
        h = HybridCache(block_size=16, hidden_size=128, total_blocks=10)
        k, v = _make_kv(seq_len=32, dim=128)
        h.store_kv(0, [(k, v)])
        loaded = h.load_kv(0)
        err_k = (k - loaded[0][0]).abs().max().item()
        err_v = (v - loaded[0][1]).abs().max().item()
        assert err_k < 0.05
        assert err_v < 0.5

    def test_many_heads(self):
        """num_heads=32 (common in 7B-class models)."""
        h = HybridCache(block_size=16, hidden_size=128, total_blocks=10)
        k, v = _make_kv(heads=32, seq_len=24, dim=128)
        h.store_kv(0, [(k, v)])
        loaded = h.load_kv(0)
        assert loaded[0][0].shape == (1, 32, 24, 128)


# ===================================================================
# Diagnostic stats tests
# ===================================================================

class TestLoadKvStats:
    """load_kv_stats returns correct diagnostic info."""

    def test_fp16_stats(self):
        h = HybridCache(block_size=16, hidden_size=64, total_blocks=10)
        k, v = _make_kv(seq_len=12)
        h.store_kv(0, [(k, v)])
        s = h.load_kv_stats(0)
        assert s["mode"] == "fp16"
        assert s["seq_len"] == 12
        assert s["fp16_bytes"] == s["stored_bytes"]

    def test_mixed_stats(self):
        h = HybridCache(block_size=16, hidden_size=64, total_blocks=10)
        k, v = _make_kv(seq_len=32)
        h.store_kv(0, [(k, v)])
        s = h.load_kv_stats(0)
        assert s["mode"] == "mixed"
        assert s["seq_len"] == 32
        assert s["stored_bytes"] < s["fp16_bytes"], "Quantized should be smaller"
        assert 0 < s["saving_ratio"] < 1.0

    def test_stats_missing_block(self):
        h = HybridCache(block_size=16, hidden_size=64, total_blocks=10)
        assert h.load_kv_stats(999) == {}
