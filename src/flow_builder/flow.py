"""
File: src/flow_builder/flow.py

Bidirectional flow — works directly with numpy records from kp_read_batch().
NO dependency on PacketRecordV1 or FlowKeyV1 dataclasses.
packet_dtype fields: mono_ts_ns, schema_version, if_index, captured_len,
                     wire_len, src_port, dst_port, direction, ip_version,
                     proto, tcp_flags, src_ip[16], dst_ip[16]
"""

from __future__ import annotations
import socket
import time
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

# ── TCP flag masks ──────────────────────────────────────────────────────────
TCP_FIN = 0x01
TCP_SYN = 0x02
TCP_RST = 0x04
TCP_PSH = 0x08
TCP_ACK = 0x10
TCP_URG = 0x20
TCP_ECE = 0x40
TCP_CWR = 0x80


# ── FlowKeyV1 — canonical bidirectional key (defined HERE, not in kernel_panel) ─
@dataclass(frozen=True)
class FlowKeyV1:
    ip_version: int
    proto:      int
    src_ip:     str    # smaller endpoint (canonical A)
    src_port:   int
    dst_ip:     str    # larger  endpoint (canonical B)
    dst_port:   int


# ── IP bytes → string ────────────────────────────────────────────────────────
def _ip_to_str(raw: np.ndarray, ip_version: int) -> str:
    b = bytes(raw)
    if ip_version == 4:
        return socket.inet_ntop(socket.AF_INET,  b[:4])
    return socket.inet_ntop(socket.AF_INET6, b[:16])


# ── Flow ─────────────────────────────────────────────────────────────────────
class Flow:
    """
    One bidirectional flow.
    Accepts numpy records (np.void) directly from kp_read_batch().
    Accumulates every raw stat needed for all 60 CIC-IDS features.
    """

    IDLE_NS = 1_000_000_000   # 1 s gap → new idle period

    def __init__(self, key: FlowKeyV1, first_pkt: np.void):
        self.key = key

        ts = int(first_pkt['mono_ts_ns'])
        self.start_ts_ns   = ts
        self.last_ts_ns    = ts
        self.last_pkt_time = time.monotonic()

        # per-direction timestamps (ns) and payload lengths (wire_len)
        self.fwd_ts:   List[int] = []
        self.bwd_ts:   List[int] = []
        self.fwd_lens: List[int] = []
        self.bwd_lens: List[int] = []

        # header length totals
        self.fwd_hdr_len: int = 0
        self.bwd_hdr_len: int = 0

        # TCP flow-level flag counters
        self.fin_cnt = self.syn_cnt = self.rst_cnt = 0
        self.psh_cnt = self.ack_cnt = self.urg_cnt = 0
        self.cwe_cnt = self.ece_cnt = 0

        # TCP forward-only flag counters
        self.fwd_psh = 0
        self.fwd_urg = 0

        # init window sizes (first packet per direction)
        self.init_win_fwd: int = 0
        self.init_win_bwd: int = 0
        self._init_win_fwd_set = False
        self._init_win_bwd_set = False

        # act_data_pkt_fwd and min_seg_size_forward
        self.act_data_pkt_fwd: int = 0
        self._min_seg_fwd:     int = 0
        self._min_seg_fwd_set: bool = False

        # active / idle period tracking
        self._last_ts:       int = 0
        self._active_start:  int = 0
        self.active_periods: List[Tuple[int, int]] = []
        self.idle_periods:   List[Tuple[int, int]] = []

        # server-side port (canonical dst = larger endpoint)
        self.dst_port: int = key.dst_port

        self.add_packet(first_pkt)

    # ─────────────────────────────────────────────────────────────────────
    def add_packet(self, pkt: np.void) -> None:
        ts    = int(pkt['mono_ts_ns'])
        proto = int(pkt['proto'])
        plen  = int(pkt['wire_len'])
        fwd   = self._is_forward(pkt)

        if fwd:
            self.fwd_ts.append(ts)
            self.fwd_lens.append(plen)
            self.fwd_hdr_len += _hdr_len(proto)
        else:
            self.bwd_ts.append(ts)
            self.bwd_lens.append(plen)
            self.bwd_hdr_len += _hdr_len(proto)

        if proto == 6:
            f = int(pkt['tcp_flags'])
            if f & TCP_FIN: self.fin_cnt += 1
            if f & TCP_SYN: self.syn_cnt += 1
            if f & TCP_RST: self.rst_cnt += 1
            if f & TCP_PSH: self.psh_cnt += 1
            if f & TCP_ACK: self.ack_cnt += 1
            if f & TCP_URG: self.urg_cnt += 1
            if f & TCP_CWR: self.cwe_cnt += 1
            if f & TCP_ECE: self.ece_cnt += 1
            if fwd:
                if f & TCP_PSH: self.fwd_psh += 1
                if f & TCP_URG: self.fwd_urg += 1

            # tcp_window is NOT in packet_dtype — default to 0
            tcp_win = int(pkt['tcp_window']) if 'tcp_window' in pkt.dtype.names else 0
            if fwd and not self._init_win_fwd_set:
                self.init_win_fwd = tcp_win
                self._init_win_fwd_set = True
            elif not fwd and not self._init_win_bwd_set:
                self.init_win_bwd = tcp_win
                self._init_win_bwd_set = True

        if fwd and plen > 0:
            self.act_data_pkt_fwd += 1
            if not self._min_seg_fwd_set or plen < self._min_seg_fwd:
                self._min_seg_fwd     = plen
                self._min_seg_fwd_set = True

        self._update_activity(ts)
        self.last_ts_ns    = ts
        self.last_pkt_time = time.monotonic()

    def finalize(self) -> None:
        """Close last active period. Must call before feature extraction."""
        if self._active_start and self._last_ts:
            self.active_periods.append((self._active_start, self._last_ts))

    # ─────────────────────────────────────────────────────────────────────
    def get_duration_ns(self) -> int:
        return self.last_ts_ns - self.start_ts_ns

    def get_duration_seconds(self) -> float:
        return self.get_duration_ns() / 1e9

    @property
    def is_tcp_closed(self) -> bool:
        return self.key.proto == 6 and (self.fin_cnt > 0 or self.rst_cnt > 0)

    def __len__(self):
        return len(self.fwd_ts) + len(self.bwd_ts)

    def __repr__(self):
        return (f"Flow({self.key.src_ip}:{self.key.src_port}"
                f" ↔ {self.key.dst_ip}:{self.key.dst_port}"
                f" proto={self.key.proto} pkts={len(self)}"
                f" dur={self.get_duration_seconds():.2f}s)")

    # ─────────────────────────────────────────────────────────────────────
    def _is_forward(self, pkt: np.void) -> bool:
        iv  = int(pkt['ip_version'])
        src = _ip_to_str(pkt['src_ip'], iv)
        sp  = int(pkt['src_port'])
        return src == self.key.src_ip and sp == self.key.src_port

    def _update_activity(self, ts: int) -> None:
        if self._last_ts == 0:
            self._active_start = ts
            self._last_ts      = ts
            return
        gap = ts - self._last_ts
        if gap > self.IDLE_NS:
            self.active_periods.append((self._active_start, self._last_ts))
            self.idle_periods.append((self._last_ts, ts))
            self._active_start = ts
        self._last_ts = ts


# ── helpers ──────────────────────────────────────────────────────────────────
def _hdr_len(proto: int) -> int:
    if proto == 6:  return 20
    if proto == 17: return 8
    return 0