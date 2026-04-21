"""
File: src/models/xgboost/xgb_model.py

XGBoost multi-class inference engine.
Loads .pkl with joblib, runs predict_proba(), returns BlockRuleV1 suggestions.
BlockRuleV1 here uses str IPs — converted to bytes before calling kp_add_block_rule().
"""

from __future__ import annotations
import socket
import joblib
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from ...flow_builder.features import build_feature_batch, N_FEATURES
from ...flow_builder.flow import Flow, FlowKeyV1

# ── Attack label map ─────────────────────────────────────────────────────────
LABEL_MAP: Dict[int, str] = {
    0:  "BENIGN",
    1:  "DoS_Hulk",
    2:  "DoS_GoldenEye",
    3:  "DoS_Slowloris",
    4:  "DoS_SlowHTTPTest",
    5:  "DDoS",
    6:  "PortScan",
    7:  "FTP-Patator",
    8:  "SSH-Patator",
    9:  "Bot",
    10: "Web_Attack_Brute_Force",
    11: "Web_Attack_XSS",
    12: "Web_Attack_SQL_Injection",
    13: "Infiltration",
    14: "Heartbleed",
}

BLOCK_CLASSES = set(LABEL_MAP.keys()) - {0}

TTL_MAP: Dict[int, int] = {
    1: 120_000, 2: 120_000, 3: 300_000, 4: 300_000,
    5:  60_000, 6:  30_000, 7: 600_000, 8: 600_000,
    9: 600_000, 10: 180_000, 11: 180_000, 12: 180_000,
    13: 600_000, 14: 600_000,
}
DEFAULT_TTL_MS       = 60_000
CONFIDENCE_THRESHOLD = 0.80


# ── BlockRuleV1 (HIM side — str IPs) ────────────────────────────────────────
@dataclass
class BlockRuleV1:
    ip_version:       int
    proto:            int
    src_ip:           str    # dotted string e.g. "10.0.0.1"
    dst_ip:           str
    src_port:         int
    dst_port:         int
    direction_policy: str    # ANY | INBOUND_ONLY | OUTBOUND_ONLY
    action:           str    # BLOCK
    ttl_ms:           int
    reason:           str
    reason_code:      int

    def to_kp_rule(self):
        """
        Convert to kernel_panel.BlockRuleV1 (bytes IPs, no direction_policy).
        Call this before passing to kp_add_block_rule().

        Usage:
            from src.core import kernel_panel as kp
            kp.kp_add_block_rule(rule.to_kp_rule())
        """
        from src.core.kernel_panel import BlockRuleV1 as KpBlockRuleV1

        def _ip_to_bytes(ip_str: str, ver: int) -> bytes:
            af = socket.AF_INET if ver == 4 else socket.AF_INET6
            packed = socket.inet_pton(af, ip_str)
            return (packed + b'\x00' * 12)[:16]

        return KpBlockRuleV1(
            ip_version=self.ip_version,
            proto=self.proto,
            src_ip=_ip_to_bytes(self.src_ip, self.ip_version),
            dst_ip=_ip_to_bytes(self.dst_ip, self.ip_version),
            src_port=self.src_port,
            dst_port=self.dst_port,
            ttl_ms=self.ttl_ms,
        )

    @staticmethod
    def from_key(key: FlowKeyV1, cid: int, ttl: int, label: str) -> "BlockRuleV1":
        r = BlockRuleV1(
            ip_version=key.ip_version, proto=key.proto,
            src_ip=key.src_ip, src_port=key.src_port,
            dst_ip=key.dst_ip, dst_port=key.dst_port,
            direction_policy="ANY", action="BLOCK",
            ttl_ms=ttl, reason=label, reason_code=cid,
        )
        r._flow_key = key   # للـ logging في him_pipeline
        return r


# ── Prediction result ────────────────────────────────────────────────────────
@dataclass
class PredictionResult:
    meta:       Dict[str, Any]
    class_id:   int
    label:      str
    confidence: float
    is_attack:  bool


# ── XGBoost model wrapper ────────────────────────────────────────────────────
class XGBModel:
    """
    Loads a joblib .pkl XGBoost multi-class classifier.
    Input to model: X shape (n, 60) float32 — FEATURE_COLUMNS order.
    """

    def __init__(self, model_path: str):
        self.model = self._load(model_path)
        self._verify()

    @staticmethod
    def _load(path: str):
        try:
            return joblib.load(path)
        except Exception as e:
            raise RuntimeError(
                f"Cannot load model from '{path}'.\n"
                f"Save with: joblib.dump(model, '{path}')\n"
                f"Error: {e}"
            )

    def _verify(self) -> None:
        if not hasattr(self.model, "predict_proba"):
            raise TypeError(f"{type(self.model)} has no predict_proba().")
        n = getattr(self.model, "n_classes_", None)
        if n and n != len(LABEL_MAP):
            print(f"[XGBModel] WARNING: model has {n} classes "
                  f"but LABEL_MAP has {len(LABEL_MAP)}. "
                  "Update LABEL_MAP to match model.classes_.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X.astype(np.float32)).astype(np.float32)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        proba = self.predict_proba(X)
        cids  = np.argmax(proba, axis=1).astype(int)
        confs = proba[np.arange(len(proba)), cids].astype(float)
        return cids, confs

    def get_label(self, cid: int) -> str:
        if hasattr(self.model, "classes_"):
            cls = self.model.classes_
            if cid < len(cls):
                return str(cls[cid])
        return LABEL_MAP.get(cid, f"UNKNOWN_{cid}")

    # ── main inference ────────────────────────────────────────────────────
    def run_inference(
        self,
        flows: List[Flow],
    ) -> Tuple[List[PredictionResult], List[BlockRuleV1]]:
        """
        flows → X(n,60) → XGBoost → predictions + BlockRuleV1 list

        Returns
        -------
        predictions : List[PredictionResult]
        block_rules : List[BlockRuleV1]   ← call .to_kp_rule() before kp_add_block_rule()
        """
        if not flows:
            return [], []

        X, meta_list = build_feature_batch(flows)
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        cids, confs = self.predict(X)

        predictions: List[PredictionResult] = []
        block_rules: List[BlockRuleV1]      = []

        for cid, conf, meta in zip(cids, confs, meta_list):
            cid   = int(cid)
            conf  = float(conf)
            label = self.get_label(cid)
            is_attack = cid in BLOCK_CLASSES and conf >= CONFIDENCE_THRESHOLD

            predictions.append(PredictionResult(
                meta=meta, class_id=cid, label=label,
                confidence=conf, is_attack=is_attack,
            ))

            if is_attack:
                block_rules.append(BlockRuleV1.from_key(
                    key=meta["flow_key"],
                    cid=cid,
                    ttl=TTL_MAP.get(cid, DEFAULT_TTL_MS),
                    label=label,
                ))

        return predictions, block_rules