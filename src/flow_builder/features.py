"""
File: src/flow_builder/features.py

Computes ALL 60 CIC-IDS features in EXACT training order from Flow objects.
XGBoost (.pkl) model was trained on these columns in this order.
Units: durations & IAT in microseconds (µs), rates in packets/second.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Any

from .flow import Flow, FlowKeyV1

# ── EXACT 60-column order — DO NOT REORDER ──────────────────────────────────
FEATURE_COLUMNS: List[str] = [
    "Destination_Port",             #  0
    "Flow_Duration",                #  1   µs
    "Total_Fwd_Packets",            #  2
    "Total_Backward_Packets",       #  3
    "Total_Length_of_Fwd_Packets",  #  4
    "Total_Length_of_Bwd_Packets",  #  5
    "Fwd_Packet_Length_Max",        #  6
    "Fwd_Packet_Length_Min",        #  7
    "Fwd_Packet_Length_Mean",       #  8
    "Fwd_Packet_Length_Std",        #  9
    "Bwd_Packet_Length_Max",        # 10
    "Bwd_Packet_Length_Min",        # 11
    "Bwd_Packet_Length_Mean",       # 12
    "Bwd_Packet_Length_Std",        # 13
    "Flow_IAT_Mean",                # 14  µs
    "Flow_IAT_Std",                 # 15  µs
    "Flow_IAT_Max",                 # 16  µs
    "Flow_IAT_Min",                 # 17  µs
    "Fwd_IAT_Total",                # 18  µs
    "Fwd_IAT_Mean",                 # 19  µs
    "Fwd_IAT_Std",                  # 20  µs
    "Fwd_IAT_Max",                  # 21  µs
    "Fwd_IAT_Min",                  # 22  µs
    "Bwd_IAT_Total",                # 23  µs
    "Bwd_IAT_Mean",                 # 24  µs
    "Bwd_IAT_Std",                  # 25  µs
    "Bwd_IAT_Max",                  # 26  µs
    "Bwd_IAT_Min",                  # 27  µs
    "Fwd_PSH_Flags",                # 28
    "Fwd_URG_Flags",                # 29
    "Fwd_Header_Length",            # 30
    "Bwd_Header_Length",            # 31
    "Fwd_Packets/s",                # 32
    "Bwd_Packets/s",                # 33
    "Min_Packet_Length",            # 34
    "Max_Packet_Length",            # 35
    "Packet_Length_Mean",           # 36
    "Packet_Length_Std",            # 37
    "FIN_Flag_Count",               # 38
    "SYN_Flag_Count",               # 39
    "RST_Flag_Count",               # 40
    "PSH_Flag_Count",               # 41
    "ACK_Flag_Count",               # 42
    "URG_Flag_Count",               # 43
    "CWE_Flag_Count",               # 44
    "ECE_Flag_Count",               # 45
    "Down/Up_Ratio",                # 46
    "Average_Packet_Size",          # 47
    "Init_Win_bytes_forward",       # 48
    "Init_Win_bytes_backward",      # 49
    "act_data_pkt_fwd",             # 50
    "min_seg_size_forward",         # 51
    "Active_Mean",                  # 52  µs
    "Active_Std",                   # 53  µs
    "Active_Max",                   # 54  µs
    "Active_Min",                   # 55  µs
    "Idle_Mean",                    # 56  µs
    "Idle_Std",                     # 57  µs
    "Idle_Max",                     # 58  µs
    "Idle_Min",                     # 59  µs
]

N_FEATURES = len(FEATURE_COLUMNS)
assert N_FEATURES == 60


# ── Statistical helpers ──────────────────────────────────────────────────────
def _mean(v: List[float]) -> float:
    return float(np.mean(v)) if v else 0.0

def _std(v: List[float]) -> float:
    return float(np.std(v, ddof=0)) if len(v) > 1 else 0.0

def _max(v: List[float]) -> float:
    return float(max(v)) if v else 0.0

def _min(v: List[float]) -> float:
    return float(min(v)) if v else 0.0

def _sum(v: List[float]) -> float:
    return float(sum(v)) if v else 0.0

def _iats_us(timestamps_ns: List[int]) -> List[float]:
    """Inter-arrival times in µs from sorted ns timestamps."""
    if len(timestamps_ns) < 2:
        return []
    ts = sorted(timestamps_ns)
    return [(ts[i+1] - ts[i]) / 1_000.0 for i in range(len(ts) - 1)]

def _period_durations_us(periods: List[Tuple[int, int]]) -> List[float]:
    return [(e - s) / 1_000.0 for s, e in periods if e > s]


# ── FeatureBuilder ───────────────────────────────────────────────────────────
class FeatureBuilder:

    @staticmethod
    def build_features(flow: Flow) -> np.ndarray:
        """Returns float64 ndarray shape (60,). NaN/Inf → 0."""

        fwd  = flow.fwd_lens
        bwd  = flow.bwd_lens
        all_ = fwd + bwd

        n_fwd = len(fwd)
        n_bwd = len(bwd)

        dur_us = flow.get_duration_ns() / 1_000.0
        dur_s  = dur_us / 1_000_000.0

        fwd_bytes = sum(fwd)
        bwd_bytes = sum(bwd)

        # packet length stats
        fwd_max  = _max(fwd);  fwd_min  = _min(fwd)
        fwd_mean = _mean(fwd); fwd_std  = _std(fwd)
        bwd_max  = _max(bwd);  bwd_min  = _min(bwd)
        bwd_mean = _mean(bwd); bwd_std  = _std(bwd)
        all_min  = _min(all_); all_max  = _max(all_)
        all_mean = _mean(all_);all_std  = _std(all_)

        # IAT (µs)
        all_ts    = sorted(flow.fwd_ts + flow.bwd_ts)
        flow_iats = _iats_us(all_ts)
        fwd_iats  = _iats_us(flow.fwd_ts)
        bwd_iats  = _iats_us(flow.bwd_ts)

        flow_iat_mean = _mean(flow_iats); flow_iat_std = _std(flow_iats)
        flow_iat_max  = _max(flow_iats);  flow_iat_min = _min(flow_iats)

        fwd_iat_total = _sum(fwd_iats)
        fwd_iat_mean  = _mean(fwd_iats); fwd_iat_std = _std(fwd_iats)
        fwd_iat_max   = _max(fwd_iats);  fwd_iat_min = _min(fwd_iats)

        bwd_iat_total = _sum(bwd_iats)
        bwd_iat_mean  = _mean(bwd_iats); bwd_iat_std = _std(bwd_iats)
        bwd_iat_max   = _max(bwd_iats);  bwd_iat_min = _min(bwd_iats)

        # rates
        fwd_pps = n_fwd / dur_s if dur_s > 0 else 0.0
        bwd_pps = n_bwd / dur_s if dur_s > 0 else 0.0

        # ratios
        down_up      = bwd_bytes / fwd_bytes if fwd_bytes > 0 else 0.0
        total_pkts   = n_fwd + n_bwd
        avg_pkt_size = (fwd_bytes + bwd_bytes) / total_pkts if total_pkts > 0 else 0.0

        # active / idle (µs)
        act_durs  = _period_durations_us(flow.active_periods)
        idle_durs = _period_durations_us(flow.idle_periods)

        active_mean = _mean(act_durs); active_std = _std(act_durs)
        active_max  = _max(act_durs);  active_min = _min(act_durs)
        idle_mean   = _mean(idle_durs); idle_std  = _std(idle_durs)
        idle_max    = _max(idle_durs);  idle_min  = _min(idle_durs)

        min_seg_fwd = flow._min_seg_fwd if flow._min_seg_fwd_set else 0

        row = [
            float(flow.dst_port),           #  0
            dur_us,                          #  1
            float(n_fwd),                    #  2
            float(n_bwd),                    #  3
            float(fwd_bytes),                #  4
            float(bwd_bytes),                #  5
            fwd_max,                         #  6
            fwd_min,                         #  7
            fwd_mean,                        #  8
            fwd_std,                         #  9
            bwd_max,                         # 10
            bwd_min,                         # 11
            bwd_mean,                        # 12
            bwd_std,                         # 13
            flow_iat_mean,                   # 14
            flow_iat_std,                    # 15
            flow_iat_max,                    # 16
            flow_iat_min,                    # 17
            fwd_iat_total,                   # 18
            fwd_iat_mean,                    # 19
            fwd_iat_std,                     # 20
            fwd_iat_max,                     # 21
            fwd_iat_min,                     # 22
            bwd_iat_total,                   # 23
            bwd_iat_mean,                    # 24
            bwd_iat_std,                     # 25
            bwd_iat_max,                     # 26
            bwd_iat_min,                     # 27
            float(flow.fwd_psh),             # 28
            float(flow.fwd_urg),             # 29
            float(flow.fwd_hdr_len),         # 30
            float(flow.bwd_hdr_len),         # 31
            fwd_pps,                         # 32
            bwd_pps,                         # 33
            all_min,                         # 34
            all_max,                         # 35
            all_mean,                        # 36
            all_std,                         # 37
            float(flow.fin_cnt),             # 38
            float(flow.syn_cnt),             # 39
            float(flow.rst_cnt),             # 40
            float(flow.psh_cnt),             # 41
            float(flow.ack_cnt),             # 42
            float(flow.urg_cnt),             # 43
            float(flow.cwe_cnt),             # 44
            float(flow.ece_cnt),             # 45
            down_up,                         # 46
            avg_pkt_size,                    # 47
            float(flow.init_win_fwd),        # 48
            float(flow.init_win_bwd),        # 49
            float(flow.act_data_pkt_fwd),    # 50
            float(min_seg_fwd),              # 51
            active_mean,                     # 52
            active_std,                      # 53
            active_max,                      # 54
            active_min,                      # 55
            idle_mean,                       # 56
            idle_std,                        # 57
            idle_max,                        # 58
            idle_min,                        # 59
        ]

        assert len(row) == N_FEATURES
        arr = np.array(row, dtype=np.float64)
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    @staticmethod
    def build_features_dict(flow: Flow) -> Dict[str, float]:
        arr = FeatureBuilder.build_features(flow)
        return dict(zip(FEATURE_COLUMNS, arr.tolist()))


# ── Batch builder ─────────────────────────────────────────────────────────────
def build_feature_batch(
    flows: List[Flow],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Input : list of finalized Flow objects
    Output: X (n, 60) float64  +  meta list[dict]
    """
    if not flows:
        return np.empty((0, N_FEATURES), dtype=np.float64), []
    X    = np.stack([FeatureBuilder.build_features(f) for f in flows], axis=0)
    meta = [_make_meta(f) for f in flows]
    return X, meta


def _make_meta(flow: Flow) -> Dict[str, Any]:
    return {
        "flow_key":   flow.key,
        "src_ip":     flow.key.src_ip,
        "src_port":   flow.key.src_port,
        "dst_ip":     flow.key.dst_ip,
        "dst_port":   flow.key.dst_port,
        "proto":      flow.key.proto,
        "ip_version": flow.key.ip_version,
        "start_ns":   flow.start_ts_ns,
        "end_ns":     flow.last_ts_ns,
        "n_pkts":     len(flow),
    }