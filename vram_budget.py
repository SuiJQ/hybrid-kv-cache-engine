"""
vram_budget.py — 集中式显存预算管理器（稠密模型精简版）

统一管理模型权重、KV Cache、激活值的显存占用预算，
在启动阶段精确评估所有子系统的可分配显存，运行时实时水位监控。

使用示例
────────
    from vram_budget import VRAMBudget

    budget = VRAMBudget(
        hidden_size=4096, num_layers=32, is_moe=False,
    )
    budget.log_status()

    kv_blocks = budget.safe_total_blocks()
    batch_max = budget.safe_batch_max()
    chunk_sz  = budget.safe_chunk_size()

    status = budget.check()
    if status["action"] == "compress_kv":
        scheduler.trigger_compression()
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("vram_budget")


class VRAMBudget:
    """集中式显存预算管理器。

    Parameters
    ----------
    model_params_b : float | None
        已知的模型参数量（十亿），如 7.0、32.0、70.0。
        不提供时根据 hidden_size / num_layers 自动估算。
    hidden_size : int
        模型隐藏维度（default 4096）。
    num_layers : int
        模型 Transformer 层数（default 32）。
    intermediate_size : int, optional
        FFN 中间维度。不提供时按 8/3 × hidden_size 估算。
    block_size : int
        KV Cache 块大小（default 16）。
    """

    # ── 预算分配比例（占总剩余显存的比例）─────────────────────────
    _KV_CACHE_FRAC = 0.55           # 55% → KV Cache（稠密模型分配更多）
    _ACTIVATION_FRAC = 0.25         # 25% → 激活值与计算缓冲
    _HEADROOM_FRAC = 0.20           # 20% → 安全余量

    # ── 运行时水位线 ────────────────────────────────────────────────
    _LOW_WATER_MARK = 0.15
    _CRITICAL_MARK = 0.08
    _PANIC_MARK = 0.04

    def __init__(
        self,
        model_params_b: float | None = None,
        hidden_size: int = 4096,
        num_layers: int = 32,
        intermediate_size: int | None = None,
        block_size: int = 16,
    ):
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._intermediate_size = intermediate_size
        self._block_size = block_size

        # ── 查询 GPU 显存 ─────────────────────────────────────────
        self._cuda_ok = torch.cuda.is_available()
        if not self._cuda_ok:
            logger.warning("CUDA not available — VRAM budget disabled")
            self._total_vram = 0
            self._free_vram = 0
            self._model_weight_gb = 0.0
            self._remaining_gb = 0.0
            self._kv_cache_budget_gb = 0.0
            self._activation_budget_gb = 0.0
            self._headroom_gb = 0.0
            self._initial_free = 0
            self._model_gb_used = 0.0
            return

        total_vram, free_bytes = torch.cuda.mem_get_info()
        self._total_vram = total_vram
        self._free_vram = free_bytes
        self._initial_free = free_bytes

        # ── 估算模型权重 ─────────────────────────────────────────
        if model_params_b is not None:
            weight_gb = model_params_b * 2.0  # fp16 每参数 ~2 bytes
        else:
            weight_gb = self._estimate_weight_gb(
                hidden_size, num_layers, intermediate_size,
            )
        self._model_weight_gb = weight_gb

        # ── 扣权重后的剩余显存 ────────────────────────────────────
        remaining_bytes = free_bytes - int(weight_gb * (1024**3))
        if remaining_bytes < 0:
            shortage = -remaining_bytes / (1024**3)
            logger.error(
                "⚠️  估算权重 %.1f GiB，可用显存 %.1f GiB，短缺 %.1f GiB",
                weight_gb, free_bytes / (1024**3), shortage,
            )
            remaining_bytes = max(remaining_bytes, int(free_bytes * 0.10))

        self._remaining_gb = remaining_bytes / (1024**3)

        # ── 按比例分配预算 ────────────────────────────────────────
        total_frac = (
            self._KV_CACHE_FRAC
            + self._ACTIVATION_FRAC
            + self._HEADROOM_FRAC
        )
        self._kv_cache_budget_gb = self._remaining_gb * (self._KV_CACHE_FRAC / total_frac)
        self._activation_budget_gb = self._remaining_gb * (self._ACTIVATION_FRAC / total_frac)
        self._headroom_gb = self._remaining_gb * (self._HEADROOM_FRAC / total_frac)

        self._model_gb_used = 0.0
        self._last_known_free = free_bytes
        self._last_check_time = 0.0

    # ==================================================================
    # 估算函数
    # ==================================================================

    @staticmethod
    def _estimate_weight_gb(
        hidden_size: int,
        num_layers: int,
        intermediate_size: int | None = None,
    ) -> float:
        """估算稠密 Transformer 模型的权重显存（fp16）。

        公式 (dense):
          embedding: vocab * hidden * 2
          layer: 4 * hidden^2 + 3 * hidden * intermediate
          lm_head: hidden * vocab
        """
        inter = intermediate_size or hidden_size * 8 // 3

        embedding_bytes = 32000 * hidden_size * 2  # 近似 vocab_size 32000
        layer_bytes = (4 * hidden_size * hidden_size) + (3 * hidden_size * inter)
        total_layer_bytes = num_layers * layer_bytes
        head_bytes = hidden_size * 32000 * 2

        total_bytes = embedding_bytes + total_layer_bytes + head_bytes
        return total_bytes / (1024**3)

    # ==================================================================
    # 公共接口
    # ==================================================================

    def safe_total_blocks(self) -> int:
        """返回安全的 KV Cache 块总数。"""
        if not self._cuda_ok:
            return 1024

        block_bytes = self._block_size * self._hidden_size * 2 * 2 * self._num_layers
        if block_bytes <= 0:
            return 1024

        kv_budget_bytes = int(self._kv_cache_budget_gb * (1024**3))
        max_blocks = max(64, kv_budget_bytes // max(block_bytes, 1))
        return min(max_blocks, 65536)

    def safe_batch_max(self) -> int:
        """推荐的最大批处理大小。"""
        if not self._cuda_ok:
            return 8
        free_gb = self._free_vram / (1024**3)
        if free_gb > 40:
            return 16
        if free_gb > 16:
            return 8
        return 4

    def safe_chunk_size(self) -> int:
        """推荐的预填充块大小。"""
        if self._hidden_size >= 7168:
            return 256
        if self._hidden_size >= 4096:
            return 512
        return 1024

    def check(self) -> dict:
        """运行时显存水位检查。

        Returns
        -------
        dict with keys: level, free_gb, free_frac, action, message
        """
        if not self._cuda_ok:
            return {"level": "ok", "free_gb": 0, "free_frac": 0, "action": None,
                    "message": "CUDA unavailable"}

        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            self._last_known_free = free_bytes
        except Exception:
            free_bytes = self._last_known_free

        free_gb = free_bytes / (1024**3)
        free_frac = free_bytes / max(self._total_vram, 1)

        if free_frac < self._PANIC_MARK:
            return {
                "level": "panic", "free_gb": free_gb,
                "free_frac": free_frac,
                "action": "emergency",
                "message": f"显存不足! {free_gb:.1f} GiB ({free_frac:.1%})",
            }
        if free_frac < self._CRITICAL_MARK:
            return {
                "level": "critical", "free_gb": free_gb,
                "free_frac": free_frac,
                "action": "compress_kv",
                "message": f"显存紧张: {free_gb:.1f} GiB ({free_frac:.1%})",
            }
        if free_frac < self._LOW_WATER_MARK:
            return {
                "level": "low", "free_gb": free_gb,
                "free_frac": free_frac,
                "action": "reduce_chunk",
                "message": f"显存偏低: {free_gb:.1f} GiB ({free_frac:.1%})",
            }

        return {
            "level": "ok", "free_gb": free_gb,
            "free_frac": free_frac,
            "action": None,
            "message": f"显存充足: {free_gb:.1f} GiB ({free_frac:.1%})",
        }

    def log_status(self) -> None:
        """打印完整预算报告。"""
        if not self._cuda_ok:
            logger.warning("CUDA unavailable — budget skipped")
            return

        total_gb = self._total_vram / (1024**3)

        logger.info("━" * 60)
        logger.info("🧮 显存预算 (总 %.1f GiB)", total_gb)
        logger.info("  模型权重:  ~%.1f GiB (fp16)", self._model_weight_gb)
        logger.info("  剩余显存:  %.1f GiB", self._remaining_gb)
        logger.info("  KV Cache:  %.1f GiB (%d%%)",
                     self._kv_cache_budget_gb,
                     int(self._KV_CACHE_FRAC / (self._KV_CACHE_FRAC + self._ACTIVATION_FRAC + self._HEADROOM_FRAC) * 100))
        logger.info("  激活值:    %.1f GiB", self._activation_budget_gb)
        logger.info("  安全余量:  %.1f GiB", self._headroom_gb)
        kv_blocks = self.safe_total_blocks()
        block_mb = (kv_blocks * self._block_size * self._hidden_size * 2 * 2 * self._num_layers) / (1024**2)
        logger.info("  KV 块:     %d 块 (~%.0f MiB)", kv_blocks, block_mb)
        logger.info("━" * 60)

    @property
    def vram_ok(self) -> bool:
        return self._cuda_ok and self._free_vram > self._model_weight_gb * (1024**3) * 1.2
