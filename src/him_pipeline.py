"""
src/him_pipeline.py
═══════════════════════════════════════════════════════════════════════
HIM Pipeline — الـ "main" بتاع HIM.

مش بيشتغل standalone — بيتشغّل من جوا الـ GUI loop (showcase_gui.py
أو dashboard.py) كل ما kp_read_batch() يرجّع batch جديد.

الاستخدام (من جوا الـ GUI):
────────────────────────────
    from src.him_pipeline import HIMPipeline

    pipeline = HIMPipeline(model_path="path/to/model.pkl")

    # في الـ poll loop (كل 50ms):
    batch = kp.kp_read_batch(kp._shared_memory_view)
    results = pipeline.process_batch(batch)
    for rule in results.block_rules:
        kp.kp_add_block_rule(rule.to_kp_rule())

الترتيب الداخلي:
────────────────
    kp_read_batch()  →  FlowTable  →  FeatureBuilder  →  XGBModel
         ↓                  ↓               ↓                ↓
    numpy records     Flow objects     X (n,60)        predictions
                                                      + BlockRuleV1
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .flow_builder.flow_table import FlowTable
from .flow_builder.flow       import Flow
from .flow_builder.features   import build_feature_batch, FEATURE_COLUMNS
from .models.xgboost.xgb_model import (
    XGBModel,
    BlockRuleV1,
    PredictionResult,
)

logger = logging.getLogger(__name__)


# ── نتيجة كل دورة process_batch ─────────────────────────────────────────────
@dataclass
class PipelineResult:
    """ما ترجعه process_batch() لكل batch."""

    # الـ flows اللي اتكملت في الـ batch ده
    finished_flows:  List[Flow]            = field(default_factory=list)

    # predictions من المودل (واحدة لكل flow)
    predictions:     List[PredictionResult] = field(default_factory=list)

    # block rules جاهزة ترسلها للكيرنل
    block_rules:     List[BlockRuleV1]      = field(default_factory=list)

    # إحصائيات للـ UI
    packets_in_batch: int = 0
    attacks_detected: int = 0
    flows_active:     int = 0


# ── HIM Pipeline ─────────────────────────────────────────────────────────────
class HIMPipeline:
    """
    Pipeline كامل بتاع HIM.

    Parameters
    ----------
    model_path        : مسار ملف الـ XGBoost .pkl
    inactive_timeout  : ثوان قبل ما الـ flow يتحذف لو مفيش باكيتس (default 10s)
    active_timeout    : ثوان max عمر أي flow (default 60s)
    ai_interval_s     : كل كام ثانية نشغّل الـ AI على الـ flows المكتملة (default 2s)
    """

    def __init__(
        self,
        model_path:       str,
        inactive_timeout: float = 10.0,
        active_timeout:   float = 60.0,
        ai_interval_s:    float = 2.0,
    ):
        # 1. Flow table
        self.flow_table = FlowTable(
            inactive_timeout=inactive_timeout,
            active_timeout=active_timeout,
        )

        # 2. XGBoost model
        self.model = XGBModel(model_path)

        # 3. AI batch interval
        self._ai_interval  = ai_interval_s
        self._last_ai_time = time.monotonic()

        # 4. counters للـ UI
        self.total_packets   = 0
        self.total_flows     = 0
        self.total_attacks   = 0
        self.total_rules_sent = 0

        logger.info(
            "HIMPipeline ready — model=%s inactive=%ss active=%ss ai_interval=%ss",
            model_path, inactive_timeout, active_timeout, ai_interval_s,
        )

    # ── Entry point الوحيد اللي الـ GUI بيستدعيه ────────────────────────────
    def process_batch(self, batch: np.ndarray) -> PipelineResult:
        """
        استقبل batch من kp_read_batch() وشغّل الـ pipeline كامل.

        Parameters
        ----------
        batch : numpy structured array (packet_dtype) من kp_read_batch()
                ممكن يكون فاضي (len == 0) — الدالة بتتعامل معاه صح.

        Returns
        -------
        PipelineResult  ← الـ GUI بياخد منه block_rules ويبعتها للكيرنل
        """
        result = PipelineResult()

        if batch is None or len(batch) == 0:
            result.flows_active = len(self.flow_table._flows)
            return result

        # ── الخطوة 1: أدخل كل باكيت في الـ FlowTable ────────────────────
        result.packets_in_batch = len(batch)
        self.total_packets += len(batch)

        for i in range(len(batch)):
            self.flow_table.add_packet(batch[i])

        result.flows_active = len(self.flow_table._flows)

        # ── الخطوة 2: كل ai_interval ثانية — شغّل الـ AI ─────────────────
        now = time.monotonic()
        if (now - self._last_ai_time) >= self._ai_interval:
            self._run_ai(result)
            self._last_ai_time = now

        return result

    def flush(self) -> PipelineResult:
        """
        Force-expire كل الـ flows النشطة وشغّل الـ AI عليهم.
        استدعيها عند الإغلاق أو عند stop streaming.
        """
        self.flow_table.flush_all()
        result = PipelineResult()
        self._run_ai(result)
        return result

    # ── stats للـ UI ──────────────────────────────────────────────────────
    @property
    def stats(self) -> dict:
        return {
            "total_packets":    self.total_packets,
            "total_flows":      self.total_flows,
            "total_attacks":    self.total_attacks,
            "total_rules_sent": self.total_rules_sent,
            "active_flows":     len(self.flow_table._flows),
        }

    # ── private ───────────────────────────────────────────────────────────
    def _run_ai(self, result: PipelineResult) -> None:
        """
        اجمع الـ flows المكتملة، حسب الـ features، شغّل المودل،
        واملأ result بالـ predictions والـ block_rules.
        """
        finished = self.flow_table.get_finalized_flows()
        if not finished:
            return

        result.finished_flows = finished
        self.total_flows += len(finished)

        # ── الخطوة 3: حساب الـ 60 Feature ──────────────────────────────
        X, meta_list = build_feature_batch(finished)

        # ── الخطوة 4: تشغيل المودل ──────────────────────────────────────
        predictions, block_rules = self.model.run_inference(finished)

        result.predictions  = predictions
        result.block_rules  = block_rules
        result.attacks_detected = len(block_rules)

        self.total_attacks   += len(block_rules)
        self.total_rules_sent += len(block_rules)

        # log summary
        # build a quick lookup: flow_key → confidence
        conf_map = {
            p.meta["flow_key"]: p.confidence
            for p in predictions
        }

        if block_rules:
            for rule in block_rules:
                conf_pct = conf_map.get(rule._flow_key, 0) * 100
                logger.warning(
                    "ATTACK %-25s  %s:%d → %s:%d  conf=%.0f%%  TTL=%dms",
                    rule.reason,
                    rule.src_ip, rule.src_port,
                    rule.dst_ip, rule.dst_port,
                    conf_pct,
                    rule.ttl_ms,
                )
        else:
            logger.debug(
                "batch processed: %d flows, 0 attacks", len(finished)
            )