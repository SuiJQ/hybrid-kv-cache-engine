"""
vram_budget.py — 集中式显存预算管理器

统一管理模型权重、KV Cache、专家缓存、推测解码、激活值的显存占用预算，
在启动阶段精确评估所有子系统的可分配显存，运行时实时水位监控，防止 OOM。

🎯 设计目标
───────────
- 高内聚低耦合：独立模块，零 MoeOwner 内部引用依赖
- 完全兼容现有架构：可选传入，不存在时各子系统回退到自身的 auto-tuning
- 零基础友好：一行 ``budget.log_status()`` 打印完整预算报告

使用示例
────────
    from vram_budget import VRAMBudget

    budget = VRAMBudget(
        hidden_size=4096, num_layers=32, num_experts=8, is_moe=True,
    )
    budget.log_status()                    # ← 零基础用户：就这样，打印完整报告

    # 获取各子系统安全值
    kv_blocks = budget.safe_total_blocks()      # → 安全的 KV 块数
    ec_blocks = budget.safe_expert_cache_blocks(block_bytes)  # → 安全专家缓存数
    batch_max = budget.safe_batch_max()         # → 推荐批处理上限
    chunk_sz  = budget.safe_chunk_size()        # → 推荐预填充块大小

    # 运行时水位检查
    status = budget.check()                     # → {"level": "ok", "free_gb": 12.3, "action": None}
    if status["action"] == "compress_kv":
        scheduler.trigger_compression()
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("vram_budget")


class VRAMBudget:
    """集中式显存预算管理器。

    初始化时自动查询 GPU 显存、估算模型权重、按比例分配预算给各个子系统。
    运行时提供水位检查，在显存不足时给出降解建议。

    Parameters
    ----------
    model_params_b : float, optional
        已知的模型参数量（十亿），如 7.0、32.0、70.0。
        不提供时根据 hidden_size / num_layers / num_experts 自动估算。
    hidden_size : int
        模型隐藏维度（default 4096）。
    num_layers : int
        模型 Transformer 层数（default 32）。
    intermediate_size : int, optional
        FFN 中间维度。不提供时按 8/3 × hidden_size 估算。
    num_experts : int
        MoE 专家数（default 8）。
    is_moe : bool
        是否为 MoE 模型（default False）。
    block_size : int
        KV Cache 块大小（default 16）。
    """

    # ── 预算分配比例（占总剩余显存的比例）─────────────────────────
    # 权重先扣掉，以下比例作用于"扣掉权重后的剩余显存"
    _KV_CACHE_FRAC = 0.45           # 45% → KV Cache
    _EXPERT_CACHE_FRAC = 0.25       # 25% → 专家缓存
    _ACTIVATION_FRAC = 0.15         # 15% → 激活值与计算缓冲
    _HEADROOM_FRAC = 0.15           # 15% → 安全余量（永不触碰）

    # ── 运行时水位线（相对初始可用显存的比例）───────────────────
    _LOW_WATER_MARK = 0.15          # 15% → "偏低"警告
    _CRITICAL_MARK = 0.08           # 8%  → 触发强制压缩
    _PANIC_MARK = 0.04              # 4%  → 紧急停车信号

    def __init__(
        self,
        model_params_b: float | None = None,
        hidden_size: int = 4096,
        num_layers: int = 32,
        intermediate_size: int | None = None,
        num_experts: int = 8,
        is_moe: bool = False,
        block_size: int = 16,
    ):
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._intermediate_size = intermediate_size
        self._num_experts = num_experts
        self._is_moe = is_moe
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
            self._expert_cache_budget_gb = 0.0
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
                num_experts, is_moe,
            )
        self._model_weight_gb = weight_gb

        # ── 扣权重后的剩余显存 ────────────────────────────────────
        remaining_bytes = free_bytes - int(weight_gb * (1024**3))
        if remaining_bytes < 0:
            shortage = -remaining_bytes / (1024**3)
            logger.error(
                "⚠️  模型权重估算 %.1f GiB，可用显存 %.1f GiB，短缺 %.1f GiB",
                weight_gb, free_bytes / (1024**3), shortage,
            )
            # 保留至少 10% 可用显存做缓存
            remaining_bytes = max(remaining_bytes, int(free_bytes * 0.10))

        self._remaining_gb = remaining_bytes / (1024**3)

        # ── 按比例分配预算 ────────────────────────────────────────
        total_frac = (
            self._KV_CACHE_FRAC
            + self._EXPERT_CACHE_FRAC
            + self._ACTIVATION_FRAC
            + self._HEADROOM_FRAC
        )
        self._kv_cache_budget_gb = self._remaining_gb * (self._KV_CACHE_FRAC / total_frac)
        self._expert_cache_budget_gb = self._remaining_gb * (self._EXPERT_CACHE_FRAC / total_frac)
        self._activation_budget_gb = self._remaining_gb * (self._ACTIVATION_FRAC / total_frac)
        self._headroom_gb = self._remaining_gb * self._HEADROOM_FRAC / total_frac

        # ── 运行时状态 ────────────────────────────────────────────
        self._pressed: bool = False  # 发生降解后标记为 True

    # ==================================================================
    # 公共 API — 各子系统从中获取安全分配值
    # ==================================================================

    def safe_total_blocks(self) -> int | None:
        """返回 KV Cache 在预算内的最大块数。

        如果 CUDA 不可用返回 None，由调用方自行处理。
        """
        if not self._cuda_ok:
            return None
        slot_bytes = self._block_size * self._hidden_size * 4  # fp16 K+V
        if slot_bytes <= 0:
            return None
        max_blocks = int(self._kv_cache_budget_gb * (1024**3) / slot_bytes)
        result = max(64, max_blocks)
        logger.debug("VRAM budget: KV blocks = %d (budget=%.2f GiB, slot=%d bytes)",
                      result, self._kv_cache_budget_gb, slot_bytes)
        return result

    def safe_expert_cache_blocks(self, block_bytes: int = 0) -> int:
        """返回专家缓存（VRAM 部分）在预算内的最大块数。

        Parameters
        ----------
        block_bytes : int
            每个专家权重块的大小（字节）。不提供时用估算值。
        """
        if not self._cuda_ok:
            return 64
        bb = block_bytes or self._rough_block_bytes()
        if bb <= 0:
            return 64
        max_blocks = int(self._expert_cache_budget_gb * (1024**3) / bb)
        return max(4, max_blocks)

    def safe_batch_max(self) -> int:
        """根据可用显存返回建议的批处理上限。"""
        if not self._cuda_ok:
            return 8
        free_gb = self._free_vram / (1024**3)
        if free_gb >= 40:
            return 16
        if free_gb >= 16:
            return 8
        if free_gb >= 8:
            return 4
        return 2

    def safe_chunk_size(self) -> int:
        """根据剩余显存返回预填充分块大小。"""
        if not self._cuda_ok:
            return 512
        rg = self._remaining_gb
        if rg >= 40:
            return 1024
        if rg >= 16:
            return 512
        if rg >= 8:
            return 256
        if rg >= 4:
            return 128
        return 64

    @property
    def total_vram_gb(self) -> float:
        return self._total_vram / (1024**3) if self._cuda_ok else 0.0

    @property
    def free_vram_gb(self) -> float:
        return self._free_vram / (1024**3) if self._cuda_ok else 0.0

    @property
    def model_weight_gb(self) -> float:
        return self._model_weight_gb

    @property
    def remaining_gb(self) -> float:
        return self._remaining_gb

    @property
    def is_pressed(self) -> bool:
        return self._pressed

    # ==================================================================
    # 运行时显存水位监控
    # ==================================================================

    def check(self) -> dict:
        """检查当前显存水位，返回降解建议。

        Returns
        -------
        dict
            level : 'ok' | 'low' | 'critical' | 'panic'
            free_gb : 当前可用显存 (GiB)
            free_frac : 当前可用比例
            action : None | 'reduce_chunk' | 'compress_kv' | 'emergency'
            message : 人性化描述
        """
        if not self._cuda_ok:
            return {"level": "ok", "free_gb": 0.0, "free_frac": 0.0,
                    "action": None, "message": "CUDA unavailable"}

        free_bytes, _ = self._get_current_free()
        free_gb = free_bytes / (1024**3)
        free_frac = free_bytes / max(self._initial_free, 1)

        if free_frac > self._LOW_WATER_MARK:
            return {"level": "ok", "free_gb": free_gb, "free_frac": free_frac,
                    "action": None, "message": "显存健康"}

        if free_frac > self._CRITICAL_MARK:
            logger.warning("显存偏低: %.1f GiB 可用 (%.1f%%)", free_gb, free_frac * 100)
            return {"level": "low", "free_gb": free_gb, "free_frac": free_frac,
                    "action": "reduce_chunk",
                    "message": f"显存偏低: {free_gb:.1f} GiB 可用"}

        if free_frac > self._PANIC_MARK:
            logger.warning("显存紧张: %.1f GiB 可用 (%.1f%%)", free_gb, free_frac * 100)
            self._pressed = True
            return {"level": "critical", "free_gb": free_gb, "free_frac": free_frac,
                    "action": "compress_kv",
                    "message": f"显存紧张: {free_gb:.1f} GiB 可用，触发 KV 压缩"}

        logger.error("显存告急: %.1f GiB 可用 (%.1f%%)", free_gb, free_frac * 100)
        self._pressed = True
        return {"level": "panic", "free_gb": free_gb, "free_frac": free_frac,
                "action": "emergency",
                "message": f"显存告急: {free_gb:.1f} GiB 可用，紧急降级"}

    # ==================================================================
    # 🎯 零基础日志 — 一条命令打印完整报告
    # ==================================================================

    def log_status(self) -> None:
        """打印完整的显存预算报告。

        零基础用户只需调用这一条命令：
            >>> budget.log_status()

        就能看到完整的 GPU 显存分配情况。
        """
        if not self._cuda_ok:
            logger.info("VRAM 状态: CUDA 不可用 — 显存预算功能已禁用。")
            return

        sep = "=" * 56
        logger.info("")
        logger.info(sep)
        logger.info("  🖥️  MoeOwner 显存预算报告")
        logger.info(sep)
        logger.info("  总 GPU 显存:             %8.1f GiB", self.total_vram_gb)
        logger.info("  启动时可用显存:          %8.1f GiB", self.free_vram_gb)

        if self._is_moe:
            logger.info("  ──────────────────────────────────────────")
            logger.info("  模型类型:                MoE (%d 专家 × %d 层)",
                         self._num_experts, self._num_layers)
        else:
            logger.info("  ──────────────────────────────────────────")
            logger.info("  模型类型:                Dense (%d 层)", self._num_layers)

        logger.info("  估算模型权重:            %8.1f GiB", self._model_weight_gb)
        logger.info("  扣权重后剩余:            %8.1f GiB", self._remaining_gb)

        n_kv = self.safe_total_blocks() or 0
        n_ec = self.safe_expert_cache_blocks()
        logger.info("  ──────────────────────────────────────────")
        logger.info("  预算分配：")
        logger.info("    KV Cache:              %8.1f GiB  (最多 %d 块)",
                     self._kv_cache_budget_gb, n_kv)
        logger.info("    专家缓存 (VRAM):       %8.1f GiB  (最多 %d 块)",
                     self._expert_cache_budget_gb, n_ec)
        logger.info("    激活值与缓冲:          %8.1f GiB",
                     self._activation_budget_gb)
        logger.info("    安全余量 (永不触碰):   %8.1f GiB",
                     self._headroom_gb)
        logger.info("  ──────────────────────────────────────────")
        logger.info("  推荐批处理上限:          %d", self.safe_batch_max())
        logger.info("  推荐预填充块大小:        %d", self.safe_chunk_size())

        current = self.check()
        if current["level"] != "ok":
            logger.info("  ⚠️  运行时告警:            %s", current["message"])
        else:
            logger.info("  运行时状态:              健康 ✅")
        logger.info(sep)
        logger.info("")

    @staticmethod
    def log_runtime() -> str:
        """一行显存状态字符串，适合插入日志。

        用法:
            >>> logger.info(VRAMBudget.log_runtime())
            # → "VRAM: 6.2 GiB / 23.9 GiB 可用 (25.9%)"
        """
        if not torch.cuda.is_available():
            return "VRAM: CUDA 不可用"
        try:
            free, total = torch.cuda.mem_get_info()
            frac = free / total * 100 if total > 0 else 0
            return f"VRAM: {free/(1024**3):.1f} GiB / {total/(1024**3):.1f} GiB 可用 ({frac:.1f}%)"
        except RuntimeError:
            return "VRAM: 查询失败"

    # ==================================================================
    # 估算辅助函数
    # ==================================================================

    def _estimate_weight_gb(
        self,
        hidden_size: int,
        num_layers: int,
        intermediate_size: int | None,
        num_experts: int,
        is_moe: bool,
    ) -> float:
        """估算 fp16 模型权重所需的显存（GiB）。

        公式：
            embed:        vocab_size × hidden_size × 2 (byte per fp16)
                            约 32000 × hidden_size × 2
            per layer:   4 × hidden_size² (attn: Q,K,V,O)
                       + 3 × hidden_size × intermediate_size (FFN: gate,up,down)
            MoE × expert:   FFN 部分乘以专家数
                       + 归一化层 + lm_head + 余量
        """
        if intermediate_size is None:
            intermediate_size = hidden_size * 8 // 3

        vocab_size = 32000  # 常见默认值

        # Embedding
        embed_bytes = 2 * vocab_size * hidden_size * 2
        # 每层
        attn_bytes = 4 * hidden_size * hidden_size * 2
        if is_moe:
            ffn_bytes = 3 * hidden_size * intermediate_size * num_experts * 2
        else:
            ffn_bytes = 3 * hidden_size * intermediate_size * 2

        layer_bytes = attn_bytes + ffn_bytes
        total_bytes = embed_bytes + num_layers * layer_bytes
        total_gib = total_bytes / (1024**3)
        # 加 10% 余量（norm、lm_head 等）
        total_gib *= 1.10
        return total_gib

    def _rough_block_bytes(self) -> int:
        """估算单个专家权重块大小（字节），用于预算计算。"""
        hs = self._hidden_size
        int_sz = self._intermediate_size or (hs * 8 // 3)
        return 3 * hs * int_sz * 2  # gate + up + down, fp16

    @staticmethod
    def _get_current_free() -> tuple[int, int]:
        try:
            return torch.cuda.mem_get_info()
        except RuntimeError:
            logger.warning("torch.cuda.mem_get_info() 查询失败")
            return (0, 0)
