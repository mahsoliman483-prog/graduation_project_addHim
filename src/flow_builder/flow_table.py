"""
File: src/flow_builder/flow_table.py

Flow table — accepts numpy records from kp_read_batch() directly.
Uses FlowKeyV1 defined in flow.py (NOT from kernel_panel).
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional
import numpy as np

from .flow import Flow, FlowKeyV1, _ip_to_str


class FlowTable:
    """
    Groups numpy packet records into bidirectional flows.
    Finalizes flows on: TCP FIN/RST, inactive timeout, active timeout.
    """

    def __init__(self,
                 inactive_timeout: float = 10.0,
                 active_timeout:   float = 60.0):
        self._inactive_ns = int(inactive_timeout * 1e9)
        self._active_ns   = int(active_timeout   * 1e9)

        self._flows:     Dict[FlowKeyV1, Flow] = {}
        self._finished:  List[Flow]            = []
        self._last_tick: int                   = 0

    # ── main entry point ──────────────────────────────────────────────────
    def add_packet(self, pkt: np.void) -> Optional[Flow]:
        """
        Add one numpy record from kp_read_batch().
        Returns the finalized Flow if this packet caused a close, else None.
        """
        key = self._make_key(pkt)

        if key not in self._flows:
            self._flows[key] = Flow(key, pkt)
        else:
            self._flows[key].add_packet(pkt)

        flow = self._flows[key]
        ts   = int(pkt['mono_ts_ns'])

        # TCP early close (FIN or RST)
        if flow.is_tcp_closed:
            return self._finalize(key)

        # active timeout
        if (ts - flow.start_ts_ns) >= self._active_ns:
            return self._finalize(key)

        # inactive timeout check — once per second of packet time
        if (ts - self._last_tick) >= 1_000_000_000:
            self._check_timeouts(ts)
            self._last_tick = ts

        return None

    def get_finalized_flows(self) -> List[Flow]:
        """Drain and return all finished flows."""
        out = self._finished[:]
        self._finished.clear()
        return out

    def flush_all(self) -> List[Flow]:
        """Force-expire every active flow (call on shutdown)."""
        for key in list(self._flows):
            self._finalize(key)
        return self.get_finalized_flows()

    def get_stats(self) -> dict:
        return {
            "active_flows":    len(self._flows),
            "pending_finished": len(self._finished),
        }

    # ── private ───────────────────────────────────────────────────────────
    def _make_key(self, pkt: np.void) -> FlowKeyV1:
        """
        Build canonical bidirectional FlowKeyV1 from a numpy record.
        Canonical = smaller (ip, port) tuple becomes src.
        """
        iv = int(pkt['ip_version'])

        src_ip   = _ip_to_str(pkt['src_ip'], iv)
        dst_ip   = _ip_to_str(pkt['dst_ip'], iv)
        src_port = int(pkt['src_port'])
        dst_port = int(pkt['dst_port'])
        proto    = int(pkt['proto'])

        # canonical ordering — smaller endpoint = src
        ep_a = (src_ip, src_port)
        ep_b = (dst_ip, dst_port)
        if ep_a > ep_b:
            ep_a, ep_b = ep_b, ep_a

        return FlowKeyV1(
            ip_version=iv,
            proto=proto,
            src_ip=ep_a[0], src_port=ep_a[1],
            dst_ip=ep_b[0], dst_port=ep_b[1],
        )

    def _check_timeouts(self, now_ns: int) -> None:
        expired = [
            k for k, f in self._flows.items()
            if (now_ns - f.last_ts_ns) >= self._inactive_ns
        ]
        for k in expired:
            self._finalize(k)

    def _finalize(self, key: FlowKeyV1) -> Optional[Flow]:
        flow = self._flows.pop(key, None)
        if flow:
            flow.finalize()
            self._finished.append(flow)
        return flow