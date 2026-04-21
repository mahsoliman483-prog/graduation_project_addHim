"""
showcase_gui.py  -  AMTE SecAI Kernel Pipeline Live Dashboard
Run from project root:  python main.py
                   or:  python showcase_gui.py
Requires: Python 3.8+, numpy, tkinter (stdlib)

[HIM Integration]
  - HIMPipeline  : flow_table → features → XGBoost → BlockRuleV1
  - AlertManager : threat popup (Block / Ignore)
  - AI Stats card: live attacks / flows counter in sidebar
"""

import sys
import os
import time
import socket
import logging

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

# ── path setup ─────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(_ROOT, "src", "core")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _CORE)

import kernel_panel as kp

# ── HIM Pipeline + Alert Manager ──────────────────────────────────────────────
try:
    from src.him_pipeline import HIMPipeline
    from src.alert_system.alert_manager import AlertManager
    _HIM_AVAILABLE = True
except ImportError as _him_err:
    _HIM_AVAILABLE = False
    _him_err_msg   = str(_him_err)

# ── Model path ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(_ROOT, "src", "models", "xgboost", "xgboost_model.pkl")

logger = logging.getLogger(__name__)

# ── Palette ────────────────────────────────────────────────────────────────────
C = {
    "bg":      "#0d1117", "panel":   "#161b22", "card":    "#21262d",
    "border":  "#30363d", "accent":  "#58a6ff", "green":   "#3fb950",
    "red":     "#f85149", "yellow":  "#e3b341", "teal":    "#39d353",
    "purple":  "#bc8cff", "text":    "#e6edf3", "subtext": "#8b949e",
}
MONO  = ("Consolas", 10)
MONO9 = ("Consolas", 9)
UI    = ("Segoe UI", 10)
UI_B  = ("Segoe UI", 10, "bold")
UI_S  = ("Segoe UI", 9)
H1    = ("Segoe UI", 13, "bold")
H2    = ("Segoe UI", 11, "bold")

# ── Helpers ────────────────────────────────────────────────────────────────────
_PROTO = {1: "ICMP", 6: "TCP", 17: "UDP", 58: "ICMPv6", 132: "SCTP"}
_PCOL  = {1: C["yellow"], 6: C["accent"], 17: C["green"], 58: C["yellow"]}

def proto_name(n): return _PROTO.get(int(n), str(int(n)))
def proto_col(n):  return _PCOL.get(int(n), C["subtext"])
def dir_name(d):   return "← In" if int(d) == 1 else "→ Out"
def dir_col(d):    return C["teal"] if int(d) == 1 else C["yellow"]

def fmt_ip(raw, ver):
    try:
        if int(ver) == 6:
            return socket.inet_ntop(socket.AF_INET6, bytes(raw[:16]))
        return socket.inet_ntop(socket.AF_INET, bytes(raw[:4]))
    except Exception:
        return "?.?.?.?"

def fmt_flags(f):
    b = int(f)
    names = [(0x02,"SYN"),(0x10,"ACK"),(0x01,"FIN"),(0x04,"RST"),(0x08,"PSH"),(0x20,"URG")]
    out = [n for mask, n in names if b & mask]
    return "|".join(out) if out else "—"

def parse_ip(text, ver):
    text = text.strip()
    if not text or text in ("0.0.0.0", "::", "any", ""):
        return b'\x00' * 16
    af = socket.AF_INET if ver == 4 else socket.AF_INET6
    try:
        packed = socket.inet_pton(af, text)
    except Exception:
        raise ValueError(f"Invalid {'IPv4' if ver==4 else 'IPv6'} address: '{text}'")
    return (packed + b'\x00' * 12)[:16]

# ── Help content ───────────────────────────────────────────────────────────────
HELP = {
"connection": ("Driver Connection",
"""Establishes a 3-step channel from Python (Ring 3) into the kernel (Ring 0).

Step 1: CreateFileW  → opens a HANDLE to \\\\.\\SecAIDriver
Step 2: OpenEventW   → maps the named kernel event 'SecAIPacketEvent'
Step 3: IOCTL_START_CAPTURE
         → driver calls MmMapLockedPagesSpecifyCache(), returning a
           user-mode virtual address for the 16 MB Shared Memory
           Ring Buffer that bridges Ring 0 and Ring 3.

On Disconnect: CloseHandle triggers EvtFileCleanup in Driver.cpp,
which safely nullifies the user-space mapping pointer."""),

"metrics": ("Ring Buffer Metrics",
"""Direct read of SharedMemoryHeader at offset 0 of shared memory.

  Head     — next WRITE slot (updated atomically by kernel ClassifyFn)
  Tail     — next READ  slot (updated by Python after consuming a batch)
  Capacity — max packet slots ≈ 262,142  = (16 MB − 192 B) / 64 B
  Dropped  — packets dropped because the buffer was full

Head & Tail each occupy their own 64-byte cache line (padded with 56
zeros) to eliminate false-sharing between kernel producer and Python
consumer. FlushProcessWriteBuffers() is called before every read for
a full hardware memory fence on all CPU architectures (Vista+)."""),

"throughput": ("Throughput Monitor",
"""Measures Python-side packet ingestion rate in real time.

  Rate    — total_received ÷ elapsed session seconds
  Total   — running count of all PacketRecordV1 structs consumed
  Session — wall-clock seconds since 'Start Streaming' was pressed

The kernel fires SecAIPacketEvent after every 1,024 packets OR every
5 ms (50,000 × 100-ns ticks), whichever comes first.
Python polls every 50 ms via Tkinter after(), giving ≤ 50 ms latency."""),

"stream": ("Live Kernel Packet Stream",
"""Reads PacketRecordV1 structs from the SPSC ring buffer (lock-free).

Read path per poll cycle:
  1. FlushProcessWriteBuffers() — full hardware memory barrier (mfence)
  2. Read head index from shared memory (kernel write pointer)
  3. np.frombuffer()  — zero-copy cast of shared memory as numpy array
                        (64 bytes per PacketRecordV1 record)
  4. np.copy()        — snapshot to prevent races during iteration
  5. Write new tail   — ctypes.c_uint64.from_buffer() at byte offset 72

Filters below are CLIENT-SIDE only — all packets are still captured
and counted in Throughput regardless of the active filter."""),

"filter_proto": ("Filter: Protocol",
"""Filters the TABLE VIEW by IP protocol (client-side only).

  All    — show every captured protocol
  TCP    — Transmission Control Protocol  (proto = 6)
  UDP    — User Datagram Protocol         (proto = 17)
  ICMP   — Internet Control Message       (proto = 1)
  ICMPv6 — IPv6 ICMP                      (proto = 58)

Tip: select TCP then load a website to isolate browser traffic."""),

"filter_dir": ("Filter: Direction",
"""Filters the TABLE VIEW by traffic direction (client-side only).

  All      — inbound + outbound
  Inbound  — packets arriving from network (FWPS_LAYER_INBOUND_*)
             direction field = 1 in PacketRecordV1
  Outbound — packets leaving the system   (FWPS_LAYER_OUTBOUND_*)
             direction field = 0 in PacketRecordV1

Direction is set by which WFP layer triggered the callout, not IPs."""),

"filter_ipver": ("Filter: IP Version",
"""Filters the TABLE VIEW by IP version (client-side only).

  All  — IPv4 and IPv6
  IPv4 — 32-bit addressed traffic only
  IPv6 — 128-bit addressed traffic only

The driver registers 4 WFP callouts, capturing both IPv4 and IPv6
on both inbound and outbound transport layers simultaneously."""),

"block": ("Block Rule Injection  (Ring 3 → Ring 0)",
"""Sends a BlockRuleV1 struct to the kernel via IOCTL_ADD_BLOCK_RULE.

Code path:
  Python DeviceIoControl(IOCTL_ADD_BLOCK_RULE)
    → EvtIoDeviceControl  in Driver.cpp
    → BlockEngine_AddRule in BlockEngine.cpp
       → InterlockedCompareExchange atomically claims a free slot
         in the g_BlockRules[1024] kernel table

On every packet, the WFP ClassifyFn calls ShouldBlockPacket().
If a rule matches, classifyOut→actionType = FWP_ACTION_BLOCK is
set and the packet is dropped at Ring 0 — no app ever sees it.
Rules auto-expire after ttl_ms milliseconds."""),

"ai_stats": ("AI Detection Engine",
"""Live statistics from the HIM (AI) pipeline.

  Flows     — total bidirectional flows finalized this session
  Attacks   — flows classified as malicious (confidence ≥ 80%)
  Rules     — block rules successfully sent to kernel
  Active    — flows currently being tracked in FlowTable

The AI runs every 2 seconds on finalized flows.
Each flow produces one XGBoost prediction (15 classes).
Confidence threshold: 0.80 — below this the flow is ignored."""),

"ipver_rule": ("Rule: IP Version",
"""Whether the rule targets IPv4 or IPv6 traffic.

  IPv4 — 32-bit address stored in bytes 0-3 of src_ip[16]/dst_ip[16]
  IPv6 — 128-bit address fills all 16 bytes

Separate WFP callouts are registered per IP version:
  FWPS_LAYER_INBOUND_TRANSPORT_V4  /  _V6
  FWPS_LAYER_OUTBOUND_TRANSPORT_V4 /  _V6"""),

"proto_rule": ("Rule: Protocol",
"""IP protocol number the block rule will match.

  TCP  (6)  — HTTP, HTTPS, SSH, RDP, ...
  UDP  (17) — DNS, QUIC, video streaming, gaming, ...
  ICMP (1)  — ping, traceroute, ...
  Any  (0)  — matches ALL protocols

Value read from FWPS_FIELD_INBOUND_TRANSPORT_V4_IP_PROTOCOL."""),

"src_ip": ("Rule: Source IP",
"""Source IP to match. Leave blank / 0.0.0.0 / :: to match ANY.

IPv4 → dotted decimal:  192.168.1.100
IPv6 → colon notation:  fe80::1

Stored in src_ip[16] of BlockRuleV1. Compared byte-by-byte in
ShouldBlockPacket(). All-zero = wildcard (any address)."""),

"dst_ip": ("Rule: Destination IP",
"""Destination IP to match. Leave blank for any.

Examples:
  8.8.8.8              — block traffic to Google DNS (IPv4)
  2001:4860:4860::8888 — block Google DNS over IPv6

Combine with Dst Port to surgically block a specific server."""),

"src_port": ("Rule: Source Port",
"""TCP/UDP source port to match. Enter 0 to match ANY source port.

Source ports are typically ephemeral (1024–65535) on clients.
Use case: isolate traffic from a specific local port/socket."""),

"dst_port": ("Rule: Destination Port",
"""TCP/UDP destination port to match. Enter 0 to match ANY port.

Demo targets:
  80  — HTTP    443 — HTTPS   53 — DNS
  22  — SSH    3389 — RDP   8080 — HTTP-Alt

Demo tip: set Proto=TCP, Dst Port=443, TTL=10000 ms and press Fire.
Your browser will fail all HTTPS pages for 10 seconds, proving the
kernel drops packets before TLS can even handshake."""),

"ttl": ("Rule: TTL (Time-To-Live in ms)",
"""How long the block rule stays active in the kernel.

BlockEngine stores timestamp_added (Windows FILETIME, 100-ns units).
ShouldBlockPacket() checks per packet:
  currentTime > timestamp_added + (ttl_ms × 10,000)
Expired slots are atomically cleared via InterlockedCompareExchange.

Demo values:
  5000  ms = 5-second block  (quick and impressive)
  15000 ms = 15-second block (time to switch windows)
  0        = never expires   (until driver unloads)"""),

"activity": ("Activity Log",
"""Timestamped session log of all pipeline actions.

  ✓  Successful operation
  ✗  Failed operation (with error detail)
  ⚠  Warning / informational note

Every block rule injection is logged here so you can track exactly
which IOCTL commands were sent to the kernel BlockEngine.
Log is session-scoped and not persisted to disk."""),
}

def show_help(root, key):
    title, body = HELP[key]
    win = tk.Toplevel(root)
    win.title(f"  ℹ  {title}")
    win.configure(bg=C["panel"])
    win.resizable(False, False)
    win.grab_set()
    f = tk.Frame(win, bg=C["panel"], padx=24, pady=20)
    f.pack(fill="both", expand=True)
    tk.Label(f, text=title, font=H2, fg=C["accent"], bg=C["panel"]).pack(anchor="w")
    tk.Frame(f, bg=C["border"], height=1).pack(fill="x", pady=8)
    tk.Label(f, text=body, font=MONO9, fg=C["text"], bg=C["panel"],
             justify="left", wraplength=560).pack(anchor="w")
    tk.Button(f, text="  OK  ", font=UI_B, fg=C["bg"], bg=C["accent"],
              activebackground=C["accent"], relief="flat", cursor="hand2",
              command=win.destroy).pack(pady=(16, 0))

def hbtn(parent, root, key, **kw):
    return tk.Button(parent, text=" ? ", font=("Segoe UI", 8, "bold"),
                     fg=C["subtext"], bg=C["card"], activeforeground=C["accent"],
                     activebackground=C["card"], relief="flat", bd=0,
                     cursor="question_arrow",
                     command=lambda: show_help(root, key), **kw)


# ══════════════════════════════════════════════════════════════════════════════
# Main Application
# ══════════════════════════════════════════════════════════════════════════════
class SecAIDashboard(tk.Tk):

    MAX_ROWS = 300

    def __init__(self):
        super().__init__()
        self.title("AMTE SecAI  ·  Kernel Pipeline Showcase  (Ring 0 ↔ Ring 3)")
        self.geometry("1280x840")
        self.minsize(1100, 720)
        self.configure(bg=C["bg"])

        self._capturing  = False
        self._connected  = False
        self._total_pkts = 0
        self._start_time = 0.0

        self._flt_proto = tk.StringVar(value="All")
        self._flt_dir   = tk.StringVar(value="All")
        self._flt_ipver = tk.StringVar(value="All")

        # ── [HIM] إنشاء الـ pipeline والـ alert manager ──────────────────────
        self._pipeline       = None
        self._alert_manager  = None
        self._him_available  = False

        if _HIM_AVAILABLE:
            try:
                self._pipeline      = HIMPipeline(model_path=MODEL_PATH)
                self._alert_manager = AlertManager()
                self._him_available = True
                logger.info("HIMPipeline ready — model: %s", MODEL_PATH)
            except Exception as e:
                logger.warning("HIMPipeline init failed: %s", e)
                self._him_available = False

        self._apply_styles()
        self._build_ui()
        self._try_connect()

        # ── [HIM] warning لو الـ pipeline مش شغّال ──────────────────────────
        if not self._him_available:
            reason = _him_err_msg if not _HIM_AVAILABLE else f"model not found: {MODEL_PATH}"
            self.log(f"AI pipeline disabled — {reason}", "warn")

    # ── TTK dark styles ───────────────────────────────────────────────────────
    def _apply_styles(self):
        s = ttk.Style(self)
        s.theme_use("default")
        s.configure("Treeview", background=C["card"], foreground=C["text"],
                    fieldbackground=C["card"], rowheight=22, font=MONO9,
                    bordercolor=C["border"], relief="flat")
        s.configure("Treeview.Heading", background=C["panel"], foreground=C["subtext"],
                    font=("Segoe UI", 9, "bold"), relief="flat")
        s.map("Treeview",
              background=[("selected", C["accent"])],
              foreground=[("selected", C["bg"])])
        s.configure("TCombobox", fieldbackground=C["card"], background=C["card"],
                    foreground=C["text"], selectbackground=C["accent"],
                    selectforeground=C["bg"], font=UI)
        s.map("TCombobox", fieldbackground=[("readonly", C["card"])])
        s.configure("Vertical.TScrollbar", background=C["border"],
                    troughcolor=C["card"], arrowcolor=C["subtext"], relief="flat")

    # ── Top-level layout ──────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_header()
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=C["bg"], width=272)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_propagate(False)
        self._build_left(left)

        right = tk.Frame(body, bg=C["bg"])
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self._build_right(right)

    # ── Header ────────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg=C["panel"], height=50)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="  ⬡ AMTE SecAI", font=H1,
                 fg=C["accent"], bg=C["panel"]).pack(side="left", padx=10)
        tk.Label(hdr, text="Kernel Pipeline Showcase  ·  Ring 0 ↔ Ring 3",
                 font=UI, fg=C["subtext"], bg=C["panel"]).pack(side="left")

        self._clock_var = tk.StringVar()
        tk.Label(hdr, textvariable=self._clock_var, font=MONO,
                 fg=C["subtext"], bg=C["panel"]).pack(side="right", padx=14)
        self._tick_clock()

        self._led = tk.Label(hdr, text="●", font=("Segoe UI", 18),
                             fg=C["red"], bg=C["panel"])
        self._led.pack(side="right", padx=2)

        self._status_var = tk.StringVar(value="Disconnected")
        tk.Label(hdr, textvariable=self._status_var, font=UI_B,
                 fg=C["text"], bg=C["panel"]).pack(side="right", padx=4)

        hbtn(hdr, self, "connection").pack(side="right", padx=4)

        self._btn_disc = tk.Button(hdr, text="Disconnect", font=UI_B, fg=C["bg"],
                                   bg=C["red"], activebackground=C["red"],
                                   relief="flat", padx=10, pady=3, cursor="hand2",
                                   command=self.disconnect, state="disabled")
        self._btn_disc.pack(side="right", padx=2, pady=8)

        self._btn_conn = tk.Button(hdr, text="Connect", font=UI_B, fg=C["bg"],
                                   bg=C["green"], activebackground=C["green"],
                                   relief="flat", padx=10, pady=3, cursor="hand2",
                                   command=self.connect)
        self._btn_conn.pack(side="right", padx=6, pady=8)

    def _tick_clock(self):
        self._clock_var.set(datetime.now().strftime("  %H:%M:%S  "))
        self.after(1000, self._tick_clock)

    # ── Left sidebar ──────────────────────────────────────────────────────────
    def _build_left(self, parent):
        self._build_metrics_card(parent)
        self._build_throughput_card(parent)
        self._build_ai_stats_card(parent)   # [HIM] جديد
        self._build_activity_card(parent)

    def _card(self, parent, title, help_key=None):
        outer = tk.Frame(parent, bg=C["border"])
        outer.pack(fill="x", pady=3)
        inner = tk.Frame(outer, bg=C["card"])
        inner.pack(fill="both", padx=1, pady=1)
        bar = tk.Frame(inner, bg=C["panel"])
        bar.pack(fill="x")
        tk.Label(bar, text=title, font=UI_B, fg=C["text"],
                 bg=C["panel"], padx=8, pady=5).pack(side="left")
        if help_key:
            hbtn(bar, self, help_key).pack(side="right", padx=6, pady=3)
        content = tk.Frame(inner, bg=C["card"], padx=10, pady=8)
        content.pack(fill="both", expand=True)
        return content

    def _mrow(self, parent, label, var, fg=None):
        row = tk.Frame(parent, bg=C["card"])
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, font=UI_S, fg=C["subtext"],
                 bg=C["card"], width=10, anchor="w").pack(side="left")
        tk.Label(row, textvariable=var, font=MONO, fg=fg or C["text"],
                 bg=C["card"], anchor="w").pack(side="left")

    def _build_metrics_card(self, parent):
        c = self._card(parent, "Ring Buffer Metrics", "metrics")
        self._m_head    = tk.StringVar(value="—")
        self._m_tail    = tk.StringVar(value="—")
        self._m_cap     = tk.StringVar(value="—")
        self._m_dropped = tk.StringVar(value="—")
        self._mrow(c, "Head",     self._m_head,    C["accent"])
        self._mrow(c, "Tail",     self._m_tail,    C["teal"])
        self._mrow(c, "Capacity", self._m_cap,     C["text"])
        self._mrow(c, "Dropped",  self._m_dropped, C["red"])

    def _build_throughput_card(self, parent):
        c = self._card(parent, "Throughput Monitor", "throughput")
        self._t_rate    = tk.StringVar(value="0 pkts/s")
        self._t_total   = tk.StringVar(value="0")
        self._t_session = tk.StringVar(value="0 s")
        self._mrow(c, "Rate",    self._t_rate,    C["yellow"])
        self._mrow(c, "Total",   self._t_total,   C["text"])
        self._mrow(c, "Session", self._t_session, C["subtext"])

    # ── [HIM] AI Stats Card ───────────────────────────────────────────────────
    def _build_ai_stats_card(self, parent):
        c = self._card(parent, "AI Detection Engine", "ai_stats")
        self._ai_flows   = tk.StringVar(value="0")
        self._ai_attacks = tk.StringVar(value="0")
        self._ai_rules   = tk.StringVar(value="0")
        self._ai_active  = tk.StringVar(value="0")
        self._ai_status  = tk.StringVar(
            value="Ready" if self._him_available else "Disabled"
        )
        status_color = C["teal"] if self._him_available else C["subtext"]

        self._mrow(c, "Flows",   self._ai_flows,   C["text"])
        self._mrow(c, "Attacks", self._ai_attacks, C["red"])
        self._mrow(c, "Rules",   self._ai_rules,   C["yellow"])
        self._mrow(c, "Active",  self._ai_active,  C["accent"])

        row = tk.Frame(c, bg=C["card"])
        row.pack(fill="x", pady=2)
        tk.Label(row, text="Status", font=UI_S, fg=C["subtext"],
                 bg=C["card"], width=10, anchor="w").pack(side="left")
        tk.Label(row, textvariable=self._ai_status, font=MONO,
                 fg=status_color, bg=C["card"], anchor="w").pack(side="left")

    def _update_ai_stats(self):
        """بيتستدعى كل poll cycle لو الـ pipeline شغّال."""
        if not self._him_available or self._pipeline is None:
            return
        stats = self._pipeline.stats
        self._ai_flows.set(str(stats["total_flows"]))
        self._ai_attacks.set(str(stats["total_attacks"]))
        self._ai_rules.set(str(stats["total_rules_sent"]))
        self._ai_active.set(str(stats["active_flows"]))

    def _build_activity_card(self, parent):
        outer = tk.Frame(parent, bg=C["border"])
        outer.pack(fill="both", expand=True, pady=3)
        inner = tk.Frame(outer, bg=C["card"])
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        bar = tk.Frame(inner, bg=C["panel"])
        bar.pack(fill="x")
        tk.Label(bar, text="Activity Log", font=UI_B, fg=C["text"],
                 bg=C["panel"], padx=8, pady=5).pack(side="left")
        hbtn(bar, self, "activity").pack(side="right", padx=6, pady=3)
        tk.Button(bar, text="Clear", font=UI_S, fg=C["subtext"], bg=C["panel"],
                  relief="flat", cursor="hand2",
                  command=self._clear_log).pack(side="right", padx=4, pady=3)

        self._log = tk.Text(inner, font=MONO9, bg=C["card"], fg=C["text"],
                            relief="flat", state="disabled", wrap="word",
                            selectbackground=C["accent"])
        vsb = ttk.Scrollbar(inner, orient="vertical", command=self._log.yview)
        self._log.configure(yscrollcommand=vsb.set)
        self._log.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        vsb.pack(side="right", fill="y", pady=6)

        self._log.tag_configure("ok",   foreground=C["teal"])
        self._log.tag_configure("err",  foreground=C["red"])
        self._log.tag_configure("warn", foreground=C["yellow"])
        self._log.tag_configure("ai",   foreground=C["purple"])
        self._log.tag_configure("ts",   foreground=C["subtext"])

    # ── Right main area ───────────────────────────────────────────────────────
    def _build_right(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.rowconfigure(1, weight=0)
        self._build_stream_panel(parent)
        self._build_block_panel(parent)

    def _build_stream_panel(self, parent):
        outer = tk.Frame(parent, bg=C["border"])
        outer.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        inner = tk.Frame(outer, bg=C["card"])
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        inner.rowconfigure(1, weight=1)
        inner.columnconfigure(0, weight=1)

        bar = tk.Frame(inner, bg=C["panel"])
        bar.grid(row=0, column=0, sticky="ew")
        tk.Label(bar, text="Live Kernel Packet Stream", font=UI_B,
                 fg=C["text"], bg=C["panel"], padx=8, pady=5).pack(side="left")
        hbtn(bar, self, "stream").pack(side="left", pady=3)

        ctrl = tk.Frame(bar, bg=C["panel"])
        ctrl.pack(side="right", padx=6, pady=4)

        tk.Label(ctrl, text="Proto:", font=UI_S, fg=C["subtext"],
                 bg=C["panel"]).pack(side="left", padx=(0, 2))
        ttk.Combobox(ctrl, textvariable=self._flt_proto, width=8, state="readonly",
                     values=["All","TCP","UDP","ICMP","ICMPv6"]).pack(side="left", padx=2)
        hbtn(ctrl, self, "filter_proto").pack(side="left", padx=2)

        tk.Label(ctrl, text="Dir:", font=UI_S, fg=C["subtext"],
                 bg=C["panel"]).pack(side="left", padx=(8, 2))
        ttk.Combobox(ctrl, textvariable=self._flt_dir, width=9, state="readonly",
                     values=["All","Inbound","Outbound"]).pack(side="left", padx=2)
        hbtn(ctrl, self, "filter_dir").pack(side="left", padx=2)

        tk.Label(ctrl, text="IP:", font=UI_S, fg=C["subtext"],
                 bg=C["panel"]).pack(side="left", padx=(8, 2))
        ttk.Combobox(ctrl, textvariable=self._flt_ipver, width=6, state="readonly",
                     values=["All","IPv4","IPv6"]).pack(side="left", padx=2)
        hbtn(ctrl, self, "filter_ipver").pack(side="left", padx=2)

        tk.Frame(ctrl, bg=C["border"], width=1).pack(side="left", fill="y", padx=8)
        self._btn_stream = tk.Button(ctrl, text="▶  Start Streaming", font=UI_B,
                                     fg=C["bg"], bg=C["accent"],
                                     activebackground=C["accent"], relief="flat",
                                     padx=10, pady=3, cursor="hand2",
                                     command=self.toggle_capture)
        self._btn_stream.pack(side="left", padx=4)
        tk.Button(ctrl, text="Clear", font=UI_S, fg=C["subtext"], bg=C["panel"],
                  activebackground=C["card"], relief="flat", cursor="hand2",
                  command=self._clear_tree).pack(side="left", padx=4)

        cols = ("Time","IPv","Proto","Src IP","S.Port","Dst IP","D.Port",
                "Wire Len","Dir","TCP Flags")
        self._tree = ttk.Treeview(inner, columns=cols, show="headings", height=12)
        widths = [72, 40, 60, 130, 58, 130, 58, 70, 65, 100]
        for col, w in zip(cols, widths):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, minwidth=w, anchor="center")

        vsb = ttk.Scrollbar(inner, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.grid(row=1, column=0, sticky="nsew", padx=(6, 0), pady=6)
        vsb.grid(row=1, column=1, sticky="ns", pady=6, padx=(0, 4))

        self._tree.tag_configure("tcp",    foreground=C["accent"])
        self._tree.tag_configure("udp",    foreground=C["green"])
        self._tree.tag_configure("icmp",   foreground=C["yellow"])
        self._tree.tag_configure("other",  foreground=C["subtext"])
        self._tree.tag_configure("in",     background="#1a2634")
        self._tree.tag_configure("out",    background="#1a2620")
        self._tree.tag_configure("attack", background="#2d1a1a")  # [HIM] صف هجوم

    def _build_block_panel(self, parent):
        outer = tk.Frame(parent, bg=C["border"])
        outer.grid(row=1, column=0, sticky="ew")
        inner = tk.Frame(outer, bg=C["card"])
        inner.pack(fill="both", padx=1, pady=1)

        bar = tk.Frame(inner, bg=C["panel"])
        bar.pack(fill="x")
        tk.Label(bar, text="⛔  Inject Block Rule  (Ring 3 → Ring 0)", font=UI_B,
                 fg=C["red"], bg=C["panel"], padx=8, pady=5).pack(side="left")
        hbtn(bar, self, "block").pack(side="left", pady=3)

        self._last_rule_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self._last_rule_var, font=MONO9,
                 fg=C["teal"], bg=C["panel"]).pack(side="right", padx=14)

        fields = tk.Frame(inner, bg=C["card"], padx=10, pady=10)
        fields.pack(fill="x")

        def lbl(text, col, row=0):
            f = tk.Frame(fields, bg=C["card"])
            f.grid(row=row, column=col, sticky="w", padx=4)
            tk.Label(f, text=text, font=UI_S, fg=C["subtext"],
                     bg=C["card"]).pack(side="left")
            hbtn(f, self, {
                "IP Version:": "ipver_rule", "Protocol:": "proto_rule",
                "Src IP:": "src_ip", "Dst IP:": "dst_ip",
                "Src Port:": "src_port", "Dst Port:": "dst_port",
                "TTL (ms):": "ttl"
            }.get(text, "block")).pack(side="left")

        def entry(col, default, width=12, row=1):
            e = tk.Entry(fields, font=MONO, bg=C["panel"], fg=C["text"],
                         insertbackground=C["text"], relief="flat",
                         width=width, bd=4)
            e.insert(0, default)
            e.grid(row=row, column=col, padx=4, sticky="w")
            return e

        lbl("IP Version:", 0); lbl("Protocol:", 1); lbl("Src IP:", 2)
        lbl("Dst IP:", 3);     lbl("Src Port:", 4); lbl("Dst Port:", 5)
        lbl("TTL (ms):", 6)

        self._ipver_cb = ttk.Combobox(fields, values=["IPv4","IPv6"],
                                      state="readonly", width=6, font=UI)
        self._ipver_cb.current(0)
        self._ipver_cb.grid(row=1, column=0, padx=4, sticky="w")

        self._proto_cb = ttk.Combobox(
            fields, values=["TCP (6)","UDP (17)","ICMP (1)","Any (0)"],
            state="readonly", width=9, font=UI)
        self._proto_cb.current(0)
        self._proto_cb.grid(row=1, column=1, padx=4, sticky="w")

        self._src_ip   = entry(2, "0.0.0.0", width=16)
        self._dst_ip   = entry(3, "0.0.0.0", width=16)
        self._src_port = entry(4, "0",       width=6)
        self._dst_port = entry(5, "443",     width=6)
        self._ttl      = entry(6, "10000",   width=8)

        btn_row = tk.Frame(inner, bg=C["card"], padx=10, pady=6)
        btn_row.pack(fill="x", pady=(0, 4))
        self._btn_fire = tk.Button(btn_row, text="⛔  Fire IOCTL Rule", font=UI_B,
                                   fg=C["bg"], bg=C["red"], activebackground=C["red"],
                                   relief="flat", padx=14, pady=5, cursor="hand2",
                                   command=self.fire_block_rule)
        self._btn_fire.pack(side="left", padx=(0, 8))
        tk.Button(btn_row, text="Clear Fields", font=UI_S, fg=C["subtext"],
                  bg=C["card"], relief="flat", cursor="hand2",
                  command=self._clear_block_fields).pack(side="left")

    # ── Logic: Connection ─────────────────────────────────────────────────────
    def _try_connect(self):
        try:
            kp.kp_init_driver()
            self._set_connected(True)
            self.log("Driver connected — shared memory mapped (16 MB SPSC ring buffer)", "ok")
        except Exception as e:
            self.log(f"Auto-connect skipped: {e}", "warn")

    def connect(self):
        try:
            kp.kp_init_driver()
            self._set_connected(True)
            self.log("Driver connected — shared memory mapped", "ok")
        except Exception as e:
            self.log(f"Connect failed: {e}", "err")
            messagebox.showerror("Connection Failed", str(e))

    def disconnect(self):
        self._capturing = False
        self._btn_stream.config(text="▶  Start Streaming", bg=C["accent"])

        # ── [HIM] flush الـ pipeline قبل الإغلاق ─────────────────────────────
        if self._him_available and self._pipeline is not None:
            try:
                result = self._pipeline.flush()
                for rule in result.block_rules:
                    kp.kp_add_block_rule(rule.to_kp_rule())
                stats = self._pipeline.stats
                self.log(
                    f"Session — packets={stats['total_packets']:,}  "
                    f"flows={stats['total_flows']}  "
                    f"attacks={stats['total_attacks']}  "
                    f"rules={stats['total_rules_sent']}",
                    "warn"
                )
            except Exception as e:
                self.log(f"Pipeline flush error: {e}", "err")

        try:
            kp.kp_close_driver()
        except Exception:
            pass

        self._set_connected(False)
        self.log("Driver disconnected", "warn")

    def _set_connected(self, on: bool):
        self._connected = on
        self._led.config(fg=C["green"] if on else C["red"])
        self._status_var.set("Connected  ●  Shared Memory Mapped" if on else "Disconnected")
        self._btn_conn.config(state="disabled" if on else "normal")
        self._btn_disc.config(state="normal"   if on else "disabled")
        self._btn_fire.config(state="normal"   if on else "disabled")
        self._btn_stream.config(state="normal" if on else "disabled")

    # ── Logic: Streaming ─────────────────────────────────────────────────────
    def toggle_capture(self):
        if not self._connected:
            messagebox.showwarning("Not Connected", "Connect to the driver first.")
            return
        if self._capturing:
            self._capturing = False
            self._btn_stream.config(text="▶  Start Streaming", bg=C["accent"])
            self.log("Streaming stopped", "warn")
        else:
            self._capturing  = True
            self._total_pkts = 0
            self._start_time = time.time()
            self._btn_stream.config(text="⏸  Stop Streaming", bg=C["yellow"])
            self.log("Streaming started — polling ring buffer every 50 ms", "ok")
            if self._him_available:
                self.log("AI pipeline active — XGBoost inference every 2s", "ai")
            self._poll()

    def _poll(self):
        if not self._capturing:
            return

        # 1. تحديث الـ ring buffer metrics
        try:
            h, t, c, d = kp.kp_get_metrics()
            self._m_head.set(str(h))
            self._m_tail.set(str(t))
            self._m_cap.set(str(c))
            self._m_dropped.set(str(d))
        except Exception:
            pass

        # 2. قراءة batch من الكيرنل
        try:
            batch = kp.kp_read_batch(kp._shared_memory_view)
        except Exception as e:
            self.log(f"Poll error: {e}", "err")
            self._capturing = False
            self._btn_stream.config(text="▶  Start Streaming", bg=C["accent"])
            return

        # 3. ── [HIM] تشغيل الـ pipeline ──────────────────────────────────────
        if self._him_available and self._pipeline is not None:
            try:
                result = self._pipeline.process_batch(batch)

                # إرسال block rules للكيرنل + عرض alert للمستخدم
                for rule in result.block_rules:
                    self._handle_attack(rule)

                # تحديث الـ AI stats card
                self._update_ai_stats()

            except Exception as e:
                self.log(f"AI pipeline error: {e}", "err")
                logger.exception("HIM pipeline error")

        # 4. تحديث الـ throughput + عرض الباكيتس في الجدول
        if batch is not None and len(batch) > 0:
            self._total_pkts += len(batch)
            elapsed = max(time.time() - self._start_time, 0.001)
            pps = int(self._total_pkts / elapsed)
            self._t_rate.set(f"{pps:,} pkts/s")
            self._t_total.set(f"{self._total_pkts:,}")
            self._t_session.set(f"{elapsed:.1f} s")

            ui_sample = batch[-15:] if len(batch) > 15 else batch
            PROTO_MAP  = {"TCP": 6, "UDP": 17, "ICMP": 1, "ICMPv6": 58}
            fp = self._flt_proto.get()
            fd = self._flt_dir.get()
            fv = self._flt_ipver.get()

            shown = 0
            for pkt in ui_sample:
                proto = int(pkt["proto"])
                dirn  = int(pkt["direction"])
                ipver = int(pkt["ip_version"])

                if fp != "All" and proto != PROTO_MAP.get(fp, -1): continue
                if fd == "Inbound"  and dirn != 1: continue
                if fd == "Outbound" and dirn != 0: continue
                if fv == "IPv4"     and ipver != 4: continue
                if fv == "IPv6"     and ipver != 6: continue

                ptag = {6:"tcp", 17:"udp", 1:"icmp"}.get(proto, "other")
                dtag = "in" if dirn == 1 else "out"
                src_ip = fmt_ip(pkt["src_ip"], ipver)
                dst_ip = fmt_ip(pkt["dst_ip"], ipver)

                self._tree.insert("", "end", tags=(ptag, dtag), values=(
                    datetime.now().strftime("%H:%M:%S"),
                    f"v{ipver}",
                    proto_name(proto),
                    src_ip, int(pkt["src_port"]),
                    dst_ip, int(pkt["dst_port"]),
                    int(pkt["wire_len"]),
                    dir_name(dirn),
                    fmt_flags(pkt["tcp_flags"]),
                ))
                shown += 1

            children = self._tree.get_children()
            if len(children) > self.MAX_ROWS:
                for iid in children[:len(children) - self.MAX_ROWS]:
                    self._tree.delete(iid)
            if shown:
                self._tree.yview_moveto(1.0)

        # 5. جدوّل الـ poll الجاي
        self.after(50, self._poll)

    # ── [HIM] معالجة هجوم مكتشف ──────────────────────────────────────────────
    def _handle_attack(self, rule):
        """
        بتتستدعى لكل BlockRuleV1 من الـ AI pipeline.
        بتعرض alert للمستخدم وبتبعت الـ rule للكيرنل لو قرر Block.
        """
        self.log(
            f"🤖 AI detected: {rule.reason}  "
            f"{rule.src_ip}:{rule.src_port} → "
            f"{rule.dst_ip}:{rule.dst_port}",
            "ai"
        )

        # عرض threat alert popup
        if self._alert_manager is not None:
            try:
                decision = self._alert_manager.trigger_alert(
                    threat_type    = rule.reason,
                    severity       = "CRITICAL",
                    description    = (
                        f"AI detected {rule.reason} attack.\n"
                        f"Source: {rule.src_ip}:{rule.src_port}"
                    ),
                    technical_data = {
                        "proc_id":  f"{rule.src_ip}:{rule.src_port}",
                        "path":     f"→ {rule.dst_ip}:{rule.dst_port}",
                        "behavior": rule.reason,
                        "engine":   "XGBoost (CIC-IDS)",
                        "status":   "ACTION REQUIRED",
                        "gpu":      "CPU Inference",
                    }
                )
            except Exception as e:
                self.log(f"Alert error: {e}", "err")
                decision = "BLOCK"   # fallback: block بدون popup
        else:
            decision = "BLOCK"       # لو AlertManager مش موجود — block تلقائي

        if decision == "BLOCK":
            try:
                ok = kp.kp_add_block_rule(rule.to_kp_rule())
                if ok:
                    summary = (
                        f"{rule.reason}  "
                        f"{rule.src_ip}:{rule.src_port} → "
                        f"{rule.dst_ip}:{rule.dst_port}  "
                        f"TTL={rule.ttl_ms}ms"
                    )
                    self._last_rule_var.set(f"🤖 AI Block: {rule.reason}")
                    self.log(f"AI Block → {summary}", "ok")
                    logger.info("AI block rule sent: %s", summary)
                else:
                    self.log(f"AI Block FAILED (kernel rejected): {rule.reason}", "err")
            except Exception as e:
                self.log(f"AI Block error: {e}", "err")
        else:
            self.log(f"User ignored AI alert: {rule.reason}", "warn")

    # ── Logic: Manual Block Rule ──────────────────────────────────────────────
    def fire_block_rule(self):
        if not self._connected:
            messagebox.showwarning("Not Connected", "Connect to the driver first.")
            return
        try:
            ver_str   = self._ipver_cb.get()
            ver       = 4 if ver_str == "IPv4" else 6
            proto_str = self._proto_cb.get()
            proto     = {"TCP (6)":6,"UDP (17)":17,"ICMP (1)":1,"Any (0)":0}.get(proto_str, 6)
            src_ip    = parse_ip(self._src_ip.get(), ver)
            dst_ip    = parse_ip(self._dst_ip.get(), ver)
            src_port  = int(self._src_port.get())
            dst_port  = int(self._dst_port.get())
            ttl_ms    = int(self._ttl.get())

            rule = kp.BlockRuleV1(
                ip_version=ver, proto=proto,
                src_ip=src_ip, dst_ip=dst_ip,
                src_port=src_port, dst_port=dst_port,
                ttl_ms=ttl_ms,
            )
            ok = kp.kp_add_block_rule(rule)
            if ok:
                summary = (
                    f"Block  {proto_str}  "
                    f"src={self._src_ip.get() or 'any'}:{src_port or 'any'}  "
                    f"→  dst={self._dst_ip.get() or 'any'}:{dst_port or 'any'}  "
                    f"TTL={ttl_ms} ms"
                )
                self._last_rule_var.set(f"✓ Last: {summary}")
                self.log(f"IOCTL sent → {summary}", "ok")
                messagebox.showinfo(
                    "IOCTL Fired",
                    f"Rule injected into kernel BlockEngine!\n\n{summary}\n\n"
                    f"ShouldBlockPacket() will now drop matching traffic at Ring 0."
                )
            else:
                self.log("IOCTL rejected by kernel (DeviceIoControl returned False)", "err")
                messagebox.showerror("Kernel Rejected",
                                     "DeviceIoControl returned False.\nCheck driver status.")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            self.log(f"Fire error: {e}", "err")
            messagebox.showerror("Error", str(e))

    # ── Utility ───────────────────────────────────────────────────────────────
    def log(self, msg, level="ok"):
        tag_map = {"ok":"ok", "err":"err", "warn":"warn", "ai":"ai"}
        icon    = {"ok":"✓", "err":"✗", "warn":"⚠", "ai":"🤖"}.get(level, "·")
        ts      = datetime.now().strftime("%H:%M:%S")
        self._log.config(state="normal")
        self._log.insert("end", f"[{ts}] ", "ts")
        self._log.insert("end", f"{icon}  {msg}\n", tag_map.get(level, "ok"))
        self._log.config(state="disabled")
        self._log.yview_moveto(1.0)

    def _clear_log(self):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

    def _clear_tree(self):
        for iid in self._tree.get_children():
            self._tree.delete(iid)
        self._total_pkts = 0
        self._start_time = time.time()
        self._t_rate.set("0 pkts/s")
        self._t_total.set("0")

    def _clear_block_fields(self):
        self._ipver_cb.current(0)
        self._proto_cb.current(0)
        for e, v in [
            (self._src_ip,   "0.0.0.0"),
            (self._dst_ip,   "0.0.0.0"),
            (self._src_port, "0"),
            (self._dst_port, "443"),
            (self._ttl,      "10000"),
        ]:
            e.delete(0, "end")
            e.insert(0, v)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")
    app = SecAIDashboard()
    app.mainloop()