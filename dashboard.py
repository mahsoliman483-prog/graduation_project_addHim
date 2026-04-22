"""
src/gui/dashboard.py
AMTE SecAI — Desktop Application
Full redesign: PyQt6, dark cyber-industrial aesthetic, HIM pipeline integrated.
"""

import sys
import os
import time
import threading
import logging
from collections import deque
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QGraphicsOpacityEffect,
    QSizePolicy, QStackedWidget, QProgressBar, QSpacerItem,
)
from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QObject, QPropertyAnimation,
    QEasingCurve, pyqtProperty, QSize, QThread, QRect,
)
from PyQt6.QtGui import (
    QColor, QPainter, QFont, QFontDatabase, QPen,
    QLinearGradient, QBrush, QPainterPath, QIcon,
)

# ── path setup ─────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src", "core"))

try:
    import kernel_panel as kp
    _KP_AVAILABLE = True
except ImportError:
    _KP_AVAILABLE = False

try:
    from src.him_pipeline import HIMPipeline
    from src.alert_system.alert_manager import AlertManager
    _HIM_AVAILABLE = True
except ImportError:
    _HIM_AVAILABLE = False

MODEL_PATH = os.path.join(_ROOT, "src", "models", "xgboost", "xgboost_model.pkl")

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════
BG       = "#080C10"      # الخلفية الرئيسية — أسود مزرق
SURFACE  = "#0D1117"      # سطح الـ cards
SURFACE2 = "#161B22"      # سطح أفتح للـ hover
BORDER   = "#21262D"      # حدود خفيفة
ACCENT   = "#00D4FF"      # سيان مضيء — اللون الأساسي
ACCENT2  = "#0099BB"      # سيان داكن للـ hover
SUCCESS  = "#00E676"      # أخضر للـ OK
WARNING  = "#FFB300"      # أصفر للـ warnings
DANGER   = "#FF3D71"      # أحمر للـ attacks
PURPLE   = "#B44FFF"      # بنفسجي للـ AI events
TEXT     = "#E6EDF3"      # نص أبيض
TEXT2    = "#8B949E"      # نص رمادي ثانوي
TEXT3    = "#484F58"      # نص خافت

SIDEBAR_W = 240

# ── Fonts ──────────────────────────────────────────────────────────────────────
FONT_MONO  = "JetBrains Mono, Consolas, monospace"
FONT_UI    = "Segoe UI, SF Pro Display, sans-serif"


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
class BackendSignals(QObject):
    log_received     = pyqtSignal(str, str, str)   # time, event, level
    attack_detected  = pyqtSignal(str, str, str)   # reason, src, dst
    stats_updated    = pyqtSignal(dict)
    connection_changed = pyqtSignal(bool)


class AppBackend(QObject):
    """
    بيدير الـ kernel connection والـ HIM pipeline.
    كل الـ I/O بيحصل هنا — الـ GUI بس بيستقبل signals.
    """

    def __init__(self, signals: BackendSignals):
        super().__init__()
        self.signals   = signals
        self.logs      = deque(maxlen=200)
        self.connected = False
        self.capturing = False

        self._pipeline      = None
        self._alert_manager = None
        self._him_ok        = False

        self._total_pkts    = 0
        self._total_flows   = 0
        self._total_attacks = 0
        self._total_rules   = 0
        self._session_start = 0.0

        self._init_him()

    def _init_him(self):
        if not _HIM_AVAILABLE:
            return
        try:
            self._pipeline      = HIMPipeline(model_path=MODEL_PATH)
            self._alert_manager = AlertManager()
            self._him_ok        = True
        except Exception as e:
            logger.warning("HIM init failed: %s", e)

    # ── kernel connection ──────────────────────────────────────────────────
    def connect_driver(self) -> tuple[bool, str]:
        if not _KP_AVAILABLE:
            return False, "kernel_panel not available"
        try:
            kp.kp_init_driver()
            self.connected = True
            self.signals.connection_changed.emit(True)
            return True, "Driver connected — 16 MB shared memory mapped"
        except Exception as e:
            return False, str(e)

    def disconnect_driver(self):
        self.capturing = False
        if self._him_ok and self._pipeline:
            try:
                result = self._pipeline.flush()
                for rule in result.block_rules:
                    if _KP_AVAILABLE:
                        kp.kp_add_block_rule(rule.to_kp_rule())
            except Exception:
                pass
        if _KP_AVAILABLE:
            try:
                kp.kp_close_driver()
            except Exception:
                pass
        self.connected = False
        self.signals.connection_changed.emit(False)

    def get_metrics(self) -> dict:
        if not _KP_AVAILABLE or not self.connected:
            return {"head": 0, "tail": 0, "capacity": 0, "dropped": 0}
        try:
            h, t, c, d = kp.kp_get_metrics()
            return {"head": h, "tail": t, "capacity": c, "dropped": d}
        except Exception:
            return {"head": 0, "tail": 0, "capacity": 0, "dropped": 0}

    def poll_once(self) -> list:
        """يُستدعى كل 50ms من الـ GUI QTimer — يرجع list من الـ packet dicts للعرض."""
        if not _KP_AVAILABLE or not self.connected:
            return []
        try:
            batch = kp.kp_read_batch(kp._shared_memory_view)
        except Exception:
            return []

        if batch is None or len(batch) == 0:
            return []

        self._total_pkts += len(batch)

        # ── HIM pipeline ───────────────────────────────────────────────────
        if self._him_ok and self._pipeline:
            try:
                result = self._pipeline.process_batch(batch)
                if result.block_rules:
                    for rule in result.block_rules:
                        self._handle_rule(rule)
                stats = self._pipeline.stats
                self._total_flows   = stats["total_flows"]
                self._total_attacks = stats["total_attacks"]
                self._total_rules   = stats["total_rules_sent"]
            except Exception as e:
                logger.error("HIM error: %s", e)

        # ── emit stats ─────────────────────────────────────────────────────
        elapsed = max(time.time() - self._session_start, 0.001)
        self.signals.stats_updated.emit({
            "total_pkts":  self._total_pkts,
            "pps":         int(self._total_pkts / elapsed),
            "elapsed":     elapsed,
            "total_flows": self._total_flows,
            "attacks":     self._total_attacks,
            "rules":       self._total_rules,
            "active_flows": self._pipeline.stats["active_flows"] if self._him_ok and self._pipeline else 0,
        })

        # ── convert last 15 pkts to display dicts ─────────────────────────
        import socket as _sock
        sample = batch[-15:] if len(batch) > 15 else batch
        out = []
        for p in sample:
            ver = int(p["ip_version"])
            try:
                src = _sock.inet_ntop(
                    _sock.AF_INET6 if ver == 6 else _sock.AF_INET,
                    bytes(p["src_ip"][:16 if ver==6 else 4])
                )
                dst = _sock.inet_ntop(
                    _sock.AF_INET6 if ver == 6 else _sock.AF_INET,
                    bytes(p["dst_ip"][:16 if ver==6 else 4])
                )
            except Exception:
                src = dst = "?.?.?.?"
            out.append({
                "ts":    datetime.now().strftime("%H:%M:%S"),
                "ver":   ver,
                "proto": int(p["proto"]),
                "src":   src,
                "sp":    int(p["src_port"]),
                "dst":   dst,
                "dp":    int(p["dst_port"]),
                "len":   int(p["wire_len"]),
                "dir":   int(p["direction"]),
                "flags": int(p["tcp_flags"]),
            })
        return out

    def _handle_rule(self, rule):
        src = f"{rule.src_ip}:{rule.src_port}"
        dst = f"{rule.dst_ip}:{rule.dst_port}"

        decision = "BLOCK"
        if self._alert_manager:
            try:
                decision = self._alert_manager.trigger_alert(
                    threat_type    = rule.reason,
                    severity       = "CRITICAL",
                    description    = f"AI detected {rule.reason}.\nSource: {src}",
                    technical_data = {
                        "proc_id":  src,
                        "path":     f"→ {dst}",
                        "behavior": rule.reason,
                        "engine":   "XGBoost (CIC-IDS)",
                        "status":   "ACTION REQUIRED",
                        "gpu":      "CPU Inference",
                    }
                )
            except Exception:
                decision = "BLOCK"

        if decision == "BLOCK" and _KP_AVAILABLE:
            try:
                kp.kp_add_block_rule(rule.to_kp_rule())
            except Exception:
                pass

        ts = datetime.now().strftime("%H:%M:%S")
        self.logs.appendleft((ts, f"[AI] {rule.reason} — {src} → {dst}", "Attack"))
        self.signals.attack_detected.emit(rule.reason, src, dst)
        self.signals.log_received.emit(ts, f"[AI] {rule.reason} blocked: {src} → {dst}", "Attack")

    def start_capture(self):
        self._total_pkts = 0
        self._session_start = time.time()
        self.capturing = True

    def stop_capture(self):
        self.capturing = False

    def send_manual_rule(self, rule) -> bool:
        if not _KP_AVAILABLE:
            return False
        return bool(kp.kp_add_block_rule(rule))


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

class ModernToggle(QWidget):
    toggled = pyqtSignal(bool)

    def __init__(self, parent=None, checked=False):
        super().__init__(parent)
        self.setFixedSize(48, 26)
        self._checked = checked
        self._pos = 3.0 if not checked else 23.0
        self._anim = QPropertyAnimation(self, b"_circle_pos", self)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._anim.setDuration(200)

    @pyqtProperty(float)
    def _circle_pos(self): return self._pos

    @_circle_pos.setter
    def _circle_pos(self, v):
        self._pos = v
        self.update()

    def mouseReleaseEvent(self, e):
        self._checked = not self._checked
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(23.0 if self._checked else 3.0)
        self._anim.start()
        self.toggled.emit(self._checked)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = QColor(ACCENT) if self._checked else QColor(BORDER)
        p.setBrush(QBrush(track))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, 0, 48, 26, 13, 13)
        p.setBrush(QBrush(QColor("white")))
        p.drawEllipse(int(self._pos), 3, 20, 20)
        p.end()


class GlowLabel(QLabel):
    """Label بـ glow effect للـ accent headings."""
    def __init__(self, text, glow_color=ACCENT, *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        self._glow = QColor(glow_color)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        # glow layer
        pen = QPen(self._glow)
        pen.setWidth(1)
        p.setPen(pen)
        p.setOpacity(0.3)
        p.drawText(self.rect().adjusted(1, 1, 1, 1), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text())
        p.setOpacity(1.0)
        p.setPen(QPen(self._glow))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text())
        p.end()


class StatCard(QFrame):
    def __init__(self, title: str, value: str, accent: str = ACCENT, icon: str = ""):
        super().__init__()
        self._accent = accent
        self.setMinimumHeight(110)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {SURFACE};
                border: 1px solid {BORDER};
                border-radius: 12px;
            }}
            QFrame:hover {{
                border: 1px solid {accent};
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(6)

        top = QHBoxLayout()
        lbl_title = QLabel(title.upper())
        lbl_title.setStyleSheet(f"color:{TEXT2}; font-size:10px; letter-spacing:1.5px; font-weight:600; background:transparent; border:none;")
        top.addWidget(lbl_title)
        top.addStretch()
        if icon:
            ico = QLabel(icon)
            ico.setStyleSheet(f"color:{accent}; font-size:18px; background:transparent; border:none;")
            top.addWidget(ico)
        layout.addLayout(top)

        self.v_label = QLabel(value)
        self.v_label.setStyleSheet(f"color:{TEXT}; font-size:30px; font-weight:700; font-family:JetBrains Mono,Consolas; background:transparent; border:none;")
        layout.addWidget(self.v_label)

    def set_value(self, v: str):
        try: self.v_label.setText(v)
        except RuntimeError: pass

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        # bottom accent line
        pen = QPen(QColor(self._accent))
        pen.setWidth(2)
        p.setPen(pen)
        p.setOpacity(0.6)
        r = self.rect()
        p.drawLine(r.left() + 20, r.bottom(), r.left() + 60, r.bottom())
        p.end()


class SidebarButton(QPushButton):
    def __init__(self, icon: str, label: str, parent=None):
        super().__init__(parent)
        self._icon  = icon
        self._label = label
        self.setCheckable(True)
        self.setFixedHeight(46)
        self.setText(f"  {icon}   {label}")
        self.setStyleSheet(f"""
            QPushButton {{
                text-align: left;
                padding-left: 16px;
                color: {TEXT2};
                background: transparent;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 500;
                font-family: {FONT_UI};
            }}
            QPushButton:checked {{
                background: {SURFACE2};
                color: {ACCENT};
                border-left: 2px solid {ACCENT};
                padding-left: 14px;
            }}
            QPushButton:hover:!checked {{
                background: {SURFACE};
                color: {TEXT};
            }}
        """)


class StatusDot(QWidget):
    def __init__(self, color=DANGER, size=10, parent=None):
        super().__init__(parent)
        self._color = color
        self._pulse = 0.0
        self.setFixedSize(size + 8, size + 8)
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)

    def set_color(self, c):
        self._color = c
        self.update()

    def _tick(self):
        self._pulse = (self._pulse + 0.08) % (2 * 3.14159)
        self.update()

    def paintEvent(self, e):
        import math
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self.width() // 2, self.height() // 2
        r = (self.width() - 8) // 2
        # pulse ring
        pulse_alpha = int(80 * abs(math.sin(self._pulse)))
        ring_color = QColor(self._color)
        ring_color.setAlpha(pulse_alpha)
        p.setBrush(QBrush(ring_color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(cx - r - 3, cy - r - 3, (r + 3) * 2, (r + 3) * 2)
        # dot
        p.setBrush(QBrush(QColor(self._color)))
        p.drawEllipse(cx - r, cy - r, r * 2, r * 2)
        p.end()


class PacketRow(QFrame):
    _PROTO = {1:"ICMP", 6:"TCP", 17:"UDP", 58:"ICMPv6"}
    _FLAGS = [(0x02,"SYN"),(0x10,"ACK"),(0x01,"FIN"),(0x04,"RST"),(0x08,"PSH"),(0x20,"URG")]

    def __init__(self, pkt: dict):
        super().__init__()
        proto = pkt["proto"]
        dirn  = pkt["dir"]

        border = {"TCP": ACCENT, "UDP": SUCCESS, "ICMP": WARNING}.get(
            self._PROTO.get(proto, ""), TEXT3)

        self.setStyleSheet(f"""
            QFrame {{
                background: {SURFACE};
                border-left: 2px solid {border};
                border-radius: 4px;
                margin: 1px 0;
            }}
        """)
        self.setFixedHeight(32)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 10, 0)
        lay.setSpacing(0)

        def cell(txt, w, color=TEXT2, mono=False, align=Qt.AlignmentFlag.AlignLeft):
            l = QLabel(txt)
            l.setFixedWidth(w)
            ff = FONT_MONO if mono else FONT_UI
            l.setStyleSheet(f"color:{color}; font-size:11px; font-family:{ff}; background:transparent; border:none;")
            l.setAlignment(align | Qt.AlignmentFlag.AlignVCenter)
            return l

        flags_str = "|".join(n for m,n in self._FLAGS if pkt["flags"] & m) or "—"
        dir_str   = "↓ IN" if dirn == 1 else "↑ OUT"
        dir_col   = SUCCESS if dirn == 1 else WARNING
        pname     = self._PROTO.get(proto, str(proto))

        lay.addWidget(cell(pkt["ts"],    72, TEXT3, mono=True))
        lay.addWidget(cell(f"v{pkt['ver']}", 30, TEXT3))
        lay.addWidget(cell(pname,         46, border))
        lay.addWidget(cell(pkt["src"],   130, TEXT2, mono=True))
        lay.addWidget(cell(f":{pkt['sp']}", 52, TEXT3, mono=True))
        lay.addWidget(cell(pkt["dst"],   130, TEXT2, mono=True))
        lay.addWidget(cell(f":{pkt['dp']}", 52, TEXT3, mono=True))
        lay.addWidget(cell(f"{pkt['len']}B", 56, TEXT2, mono=True, align=Qt.AlignmentFlag.AlignRight))
        lay.addWidget(cell(dir_str,       56, dir_col, align=Qt.AlignmentFlag.AlignCenter))
        lay.addWidget(cell(flags_str,    100, TEXT3))
        lay.addStretch()


class LogEntryWidget(QFrame):
    _COLORS = {
        "Info":    TEXT2,
        "Warning": WARNING,
        "Critical": DANGER,
        "Attack":   PURPLE,
    }

    def __init__(self, ts: str, event: str, level: str):
        super().__init__()
        color = self._COLORS.get(level, TEXT2)
        bg    = f"{SURFACE}CC" if level != "Attack" else f"#1A0A1A"
        self.setStyleSheet(f"background:{bg}; border-left:2px solid {color}; border-radius:4px; margin:1px 0;")
        self.setFixedHeight(38)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 12, 0)
        lay.setSpacing(12)

        ts_lbl = QLabel(ts)
        ts_lbl.setFixedWidth(80)
        ts_lbl.setStyleSheet(f"color:{TEXT3}; font-size:10px; font-family:{FONT_MONO}; background:transparent; border:none;")

        ev_lbl = QLabel(event)
        ev_lbl.setStyleSheet(f"color:{TEXT}; font-size:11px; font-family:{FONT_UI}; background:transparent; border:none;")

        badge = QLabel(level.upper())
        badge.setFixedWidth(72)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(f"""
            color:{color}; background:transparent;
            border: 1px solid {color};
            border-radius: 3px;
            font-size:9px; font-weight:700;
            letter-spacing:1px;
            padding: 2px 4px;
        """)

        lay.addWidget(ts_lbl)
        lay.addWidget(ev_lbl, 1)
        lay.addWidget(badge)


class SettingRow(QFrame):
    def __init__(self, title: str, desc: str, checked: bool = True):
        super().__init__()
        self.setStyleSheet(f"""
            QFrame {{
                background: {SURFACE};
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
            QFrame:hover {{
                border: 1px solid {ACCENT}44;
                background: {SURFACE2};
            }}
        """)
        self.setFixedHeight(72)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 12, 20, 12)

        txt = QVBoxLayout()
        txt.setSpacing(3)
        t = QLabel(title)
        t.setStyleSheet(f"color:{TEXT}; font-size:13px; font-weight:600; background:transparent; border:none;")
        d = QLabel(desc)
        d.setStyleSheet(f"color:{TEXT2}; font-size:11px; background:transparent; border:none;")
        txt.addWidget(t)
        txt.addWidget(d)

        self.toggle = ModernToggle(checked=checked)

        lay.addLayout(txt, 1)
        lay.addWidget(self.toggle)


class SectionHeader(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(f"""
            color: {TEXT2};
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 2px;
            font-family: {FONT_UI};
            background: transparent;
            border: none;
            padding: 4px 0;
        """)


class Divider(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(1)
        self.setStyleSheet(f"background: {BORDER}; border: none;")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE VIEWS
# ══════════════════════════════════════════════════════════════════════════════

class OverviewPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(20)

        # ── hero banner ────────────────────────────────────────────────────
        self.hero = QFrame()
        self.hero.setFixedHeight(120)
        self.hero.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #002A30, stop:1 #001A20);
                border: 1px solid {ACCENT}55;
                border-radius: 14px;
            }}
        """)
        h_lay = QHBoxLayout(self.hero)
        h_lay.setContentsMargins(30, 0, 30, 0)

        self.hero_icon = QLabel("🛡")
        self.hero_icon.setStyleSheet("font-size:44px; background:transparent; border:none;")

        hero_txt = QVBoxLayout()
        hero_txt.setSpacing(4)
        self.hero_title = QLabel("SYSTEM SECURE")
        self.hero_title.setStyleSheet(f"color:{ACCENT}; font-size:22px; font-weight:800; letter-spacing:3px; background:transparent; border:none;")
        self.hero_sub = QLabel("Real-time AI protection active  ·  All systems nominal")
        self.hero_sub.setStyleSheet(f"color:{TEXT2}; font-size:12px; background:transparent; border:none;")
        hero_txt.addWidget(self.hero_title)
        hero_txt.addWidget(self.hero_sub)

        self.hero_dot = StatusDot(SUCCESS, 12)

        h_lay.addWidget(self.hero_icon)
        h_lay.addSpacing(16)
        h_lay.addLayout(hero_txt, 1)
        h_lay.addWidget(self.hero_dot)
        lay.addWidget(self.hero)

        # ── stat cards ─────────────────────────────────────────────────────
        cards_lay = QHBoxLayout()
        cards_lay.setSpacing(12)
        self.cards = {
            "pkts":    StatCard("Packets",         "0",    ACCENT,   "⬡"),
            "flows":   StatCard("Flows",            "0",    SUCCESS,  "⬡"),
            "attacks": StatCard("Attacks Blocked",  "0",    DANGER,   "⛔"),
            "pps":     StatCard("Packets / sec",    "0",    WARNING,  "⚡"),
        }
        for c in self.cards.values():
            cards_lay.addWidget(c)
        lay.addLayout(cards_lay)

        # ── second row: active modules + AI status ─────────────────────────
        row2 = QHBoxLayout()
        row2.setSpacing(12)

        # active modules
        mod_frame = QFrame()
        mod_frame.setStyleSheet(f"background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px;")
        mod_lay = QVBoxLayout(mod_frame)
        mod_lay.setContentsMargins(20, 16, 20, 16)
        mod_lay.setSpacing(12)
        mod_lay.addWidget(SectionHeader("ACTIVE MODULES"))
        mod_lay.addWidget(Divider())

        self.toggles = {}
        for name, on in [("Real-Time AI Engine", True), ("Flow Analysis (HIM)", True),
                         ("Kernel WFP Capture", True), ("Alert System", True)]:
            r = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color:{TEXT}; font-size:12px; background:transparent; border:none;")
            t = ModernToggle(checked=on)
            self.toggles[name] = t
            r.addWidget(lbl)
            r.addStretch()
            r.addWidget(t)
            mod_lay.addLayout(r)

        row2.addWidget(mod_frame, 3)

        # kernel stats
        kern_frame = QFrame()
        kern_frame.setStyleSheet(f"background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px;")
        kern_lay = QVBoxLayout(kern_frame)
        kern_lay.setContentsMargins(20, 16, 20, 16)
        kern_lay.setSpacing(10)
        kern_lay.addWidget(SectionHeader("RING BUFFER"))
        kern_lay.addWidget(Divider())

        self.kern_vars = {}
        for label, key, color in [
            ("Head",     "head",     ACCENT),
            ("Tail",     "tail",     SUCCESS),
            ("Capacity", "capacity", TEXT2),
            ("Dropped",  "dropped",  DANGER),
        ]:
            r = QHBoxLayout()
            l = QLabel(label)
            l.setStyleSheet(f"color:{TEXT2}; font-size:11px; background:transparent; border:none;")
            v = QLabel("—")
            v.setStyleSheet(f"color:{color}; font-size:12px; font-family:{FONT_MONO}; background:transparent; border:none;")
            self.kern_vars[key] = v
            r.addWidget(l)
            r.addStretch()
            r.addWidget(v)
            kern_lay.addLayout(r)

        row2.addWidget(kern_frame, 2)
        lay.addLayout(row2)
        lay.addStretch()

    def update_stats(self, s: dict, metrics: dict):
        self.cards["pkts"].set_value(f"{s.get('total_pkts', 0):,}")
        self.cards["flows"].set_value(str(s.get("total_flows", 0)))
        self.cards["attacks"].set_value(str(s.get("attacks", 0)))
        self.cards["pps"].set_value(f"{s.get('pps', 0):,}")
        for k, v in metrics.items():
            if k in self.kern_vars:
                self.kern_vars[k].setText(f"{v:,}" if isinstance(v, int) else str(v))

    def set_status(self, ok: bool):
        if ok:
            self.hero.setStyleSheet("""
                QFrame {
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                        stop:0 #002A30, stop:1 #001A20);
                    border: 1px solid """ + ACCENT + """55;
                    border-radius: 14px;
                }
            """)
            self.hero_title.setText("SYSTEM SECURE")
            self.hero_title.setStyleSheet(f"color:{ACCENT}; font-size:22px; font-weight:800; letter-spacing:3px; background:transparent; border:none;")
            self.hero_dot.set_color(SUCCESS)
        else:
            self.hero.setStyleSheet("""
                QFrame {
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                        stop:0 #2A0008, stop:1 #1A0005);
                    border: 1px solid """ + DANGER + """55;
                    border-radius: 14px;
                }
            """)
            self.hero_title.setText("DRIVER DISCONNECTED")
            self.hero_title.setStyleSheet(f"color:{DANGER}; font-size:22px; font-weight:800; letter-spacing:3px; background:transparent; border:none;")
            self.hero_dot.set_color(DANGER)


class StreamPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        # header row
        hdr = QHBoxLayout()
        title = QLabel("LIVE PACKET STREAM")
        title.setStyleSheet(f"color:{TEXT}; font-size:16px; font-weight:700; letter-spacing:1px; background:transparent; border:none;")
        self.counter = QLabel("0 packets")
        self.counter.setStyleSheet(f"color:{TEXT2}; font-size:11px; font-family:{FONT_MONO}; background:transparent; border:none;")
        hdr.addWidget(title)
        hdr.addStretch()
        hdr.addWidget(self.counter)
        lay.addLayout(hdr)

        # column headers
        col_hdr = QFrame()
        col_hdr.setStyleSheet(f"background:{SURFACE2}; border-radius:4px;")
        col_hdr.setFixedHeight(28)
        ch_lay = QHBoxLayout(col_hdr)
        ch_lay.setContentsMargins(10, 0, 10, 0)
        ch_lay.setSpacing(0)

        def ch(t, w):
            l = QLabel(t)
            l.setFixedWidth(w)
            l.setStyleSheet(f"color:{TEXT3}; font-size:9px; font-weight:700; letter-spacing:1px; background:transparent; border:none;")
            return l

        for t, w in [("TIME",72),("VER",30),("PROTO",46),("SRC IP",130),("SPORT",52),
                     ("DST IP",130),("DPORT",52),("LEN",56),("DIR",56),("FLAGS",100)]:
            ch_lay.addWidget(ch(t, w))
        ch_lay.addStretch()
        lay.addWidget(col_hdr)

        # scroll area for packets
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{ border:none; background:transparent; }}
            QScrollBar:vertical {{ background:{SURFACE}; width:6px; border:none; }}
            QScrollBar::handle:vertical {{ background:{BORDER}; border-radius:3px; }}
            QScrollBar::handle:vertical:hover {{ background:{TEXT3}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
        """)
        self.pkt_container = QWidget()
        self.pkt_container.setStyleSheet("background:transparent;")
        self.pkt_layout = QVBoxLayout(self.pkt_container)
        self.pkt_layout.setContentsMargins(0, 0, 0, 0)
        self.pkt_layout.setSpacing(1)
        self.pkt_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self.pkt_container)
        lay.addWidget(scroll, 1)

        self._count = 0

    def add_packets(self, pkts: list):
        for p in pkts:
            row = PacketRow(p)
            self.pkt_layout.insertWidget(0, row)
            self._count += 1

        while self.pkt_layout.count() > 300:
            item = self.pkt_layout.takeAt(self.pkt_layout.count() - 1)
            if item and item.widget():
                item.widget().deleteLater()

        self.counter.setText(f"{self._count:,} packets")

    def clear(self):
        while self.pkt_layout.count():
            item = self.pkt_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._count = 0
        self.counter.setText("0 packets")


class LogsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        hdr = QHBoxLayout()
        title = QLabel("SECURITY EVENT LOG")
        title.setStyleSheet(f"color:{TEXT}; font-size:16px; font-weight:700; letter-spacing:1px; background:transparent; border:none;")
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedSize(70, 28)
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background: {SURFACE2}; color: {TEXT2};
                border: 1px solid {BORDER}; border-radius: 6px;
                font-size: 11px;
            }}
            QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
        """)
        clear_btn.clicked.connect(self.clear)
        hdr.addWidget(title)
        hdr.addStretch()
        hdr.addWidget(clear_btn)
        lay.addLayout(hdr)

        # column headers
        col_hdr = QFrame()
        col_hdr.setStyleSheet(f"background:{SURFACE2}; border-radius:4px;")
        col_hdr.setFixedHeight(28)
        ch_lay = QHBoxLayout(col_hdr)
        ch_lay.setContentsMargins(12, 0, 12, 0)
        ch_lay.setSpacing(12)
        for t in ["TIME", "EVENT", "LEVEL"]:
            l = QLabel(t)
            l.setStyleSheet(f"color:{TEXT3}; font-size:9px; font-weight:700; letter-spacing:1px; background:transparent; border:none;")
            ch_lay.addWidget(l)
            if t == "EVENT": ch_lay.addStretch()
        lay.addWidget(col_hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{ border:none; background:transparent; }}
            QScrollBar:vertical {{ background:{SURFACE}; width:6px; border:none; }}
            QScrollBar::handle:vertical {{ background:{BORDER}; border-radius:3px; }}
            QScrollBar::handle:vertical:hover {{ background:{TEXT3}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
        """)
        self.log_container = QWidget()
        self.log_container.setStyleSheet("background:transparent;")
        self.log_layout = QVBoxLayout(self.log_container)
        self.log_layout.setContentsMargins(0, 0, 0, 0)
        self.log_layout.setSpacing(1)
        self.log_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self.log_container)
        lay.addWidget(scroll, 1)

    def add_entry(self, ts: str, event: str, level: str, animate: bool = True):
        row = LogEntryWidget(ts, event, level)
        self.log_layout.insertWidget(0, row)
        if animate:
            row.setMaximumHeight(0)
            anim = QPropertyAnimation(row, b"maximumHeight")
            anim.setDuration(200)
            anim.setStartValue(0)
            anim.setEndValue(38)
            anim.setEasingCurve(QEasingCurve.Type.OutQuad)
            row.anim = anim
            anim.start()
        while self.log_layout.count() > 200:
            item = self.log_layout.takeAt(self.log_layout.count() - 1)
            if item and item.widget():
                item.widget().deleteLater()

    def clear(self):
        while self.log_layout.count():
            item = self.log_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()


class ProtectionPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)

        title = QLabel("PROTECTION SETTINGS")
        title.setStyleSheet(f"color:{TEXT}; font-size:16px; font-weight:700; letter-spacing:1px; background:transparent; border:none;")
        lay.addWidget(title)

        lay.addWidget(SectionHeader("DETECTION ENGINES"))
        for t, d, c in [
            ("Deep Packet Inspection",  "Analyzes every network flow for anomalies using CIC-IDS features", True),
            ("XGBoost AI Engine",       "15-class multi-label attack classification (CIC-IDS2017/2018)", True),
            ("Kernel-Level Hooks (WFP)","Windows Filtering Platform callouts at Ring 0 — pre-stack blocking", True),
            ("Flow Table Tracking",     "Bidirectional flow reconstruction with 10s inactive / 60s active timeout", True),
        ]:
            lay.addWidget(SettingRow(t, d, c))

        lay.addWidget(SectionHeader("RESPONSE ACTIONS"))
        for t, d, c in [
            ("Auto-Block on Detection", "Automatically send BlockRuleV1 to kernel without user confirmation", False),
            ("Alert Popups",            "Show threat notification requiring manual Block / Ignore decision", True),
            ("Session Logging",         "Log all events to logs/secai_DATE.log", True),
        ]:
            lay.addWidget(SettingRow(t, d, c))

        lay.addWidget(SectionHeader("THRESHOLDS"))
        thresh_frame = QFrame()
        thresh_frame.setStyleSheet(f"background:{SURFACE}; border:1px solid {BORDER}; border-radius:10px;")
        t_lay = QVBoxLayout(thresh_frame)
        t_lay.setContentsMargins(20, 16, 20, 16)
        t_lay.setSpacing(10)

        for label, value, color in [
            ("AI Confidence Threshold", "80%",  ACCENT),
            ("Inactive Flow Timeout",   "10 s",  TEXT2),
            ("Active Flow Timeout",     "60 s",  TEXT2),
            ("AI Batch Interval",       "2 s",   TEXT2),
        ]:
            r = QHBoxLayout()
            l = QLabel(label)
            l.setStyleSheet(f"color:{TEXT2}; font-size:12px; background:transparent; border:none;")
            v = QLabel(value)
            v.setStyleSheet(f"color:{color}; font-size:12px; font-weight:700; font-family:{FONT_MONO}; background:transparent; border:none;")
            r.addWidget(l)
            r.addStretch()
            r.addWidget(v)
            t_lay.addLayout(r)

        lay.addWidget(thresh_frame)
        lay.addStretch()


class BlockRulePage(QWidget):
    rule_submitted = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)

        title = QLabel("INJECT BLOCK RULE")
        title.setStyleSheet(f"color:{DANGER}; font-size:16px; font-weight:700; letter-spacing:1px; background:transparent; border:none;")
        lay.addWidget(title)

        sub = QLabel("Send a manual BlockRuleV1 directly to the kernel via IOCTL_ADD_BLOCK_RULE (0x802)")
        sub.setStyleSheet(f"color:{TEXT2}; font-size:11px; background:transparent; border:none;")
        lay.addWidget(sub)

        form = QFrame()
        form.setStyleSheet(f"background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px;")
        f_lay = QVBoxLayout(form)
        f_lay.setContentsMargins(24, 20, 24, 20)
        f_lay.setSpacing(14)

        def field_row(label, placeholder, default=""):
            r = QHBoxLayout()
            l = QLabel(label)
            l.setFixedWidth(140)
            l.setStyleSheet(f"color:{TEXT2}; font-size:11px; font-weight:600; letter-spacing:0.5px; background:transparent; border:none;")
            from PyQt6.QtWidgets import QLineEdit
            e = QLineEdit()
            e.setPlaceholderText(placeholder)
            e.setText(default)
            e.setStyleSheet(f"""
                QLineEdit {{
                    background:{SURFACE2}; color:{TEXT};
                    border:1px solid {BORDER}; border-radius:6px;
                    padding:6px 10px; font-size:12px;
                    font-family:{FONT_MONO};
                }}
                QLineEdit:focus {{ border-color:{ACCENT}; }}
            """)
            r.addWidget(l)
            r.addWidget(e, 1)
            return r, e

        from PyQt6.QtWidgets import QComboBox, QLineEdit

        def combo_row(label, options, default=0):
            r = QHBoxLayout()
            l = QLabel(label)
            l.setFixedWidth(140)
            l.setStyleSheet(f"color:{TEXT2}; font-size:11px; font-weight:600; letter-spacing:0.5px; background:transparent; border:none;")
            c = QComboBox()
            c.addItems(options)
            c.setCurrentIndex(default)
            c.setStyleSheet(f"""
                QComboBox {{
                    background:{SURFACE2}; color:{TEXT};
                    border:1px solid {BORDER}; border-radius:6px;
                    padding:6px 10px; font-size:12px;
                }}
                QComboBox:focus {{ border-color:{ACCENT}; }}
                QComboBox QAbstractItemView {{
                    background:{SURFACE2}; color:{TEXT};
                    border:1px solid {BORDER};
                    selection-background-color:{ACCENT}33;
                }}
            """)
            r.addWidget(l)
            r.addWidget(c, 1)
            return r, c

        r1, self.ipver_cb   = combo_row("IP Version",  ["IPv4", "IPv6"])
        r2, self.proto_cb   = combo_row("Protocol",    ["TCP (6)", "UDP (17)", "ICMP (1)", "Any (0)"])
        r3, self.src_ip_e   = field_row("Source IP",   "0.0.0.0 = any", "0.0.0.0")
        r4, self.dst_ip_e   = field_row("Destination IP", "0.0.0.0 = any", "0.0.0.0")
        r5, self.src_port_e = field_row("Source Port", "0 = any", "0")
        r6, self.dst_port_e = field_row("Dest Port",   "e.g. 443", "443")
        r7, self.ttl_e      = field_row("TTL (ms)",    "e.g. 10000", "10000")

        for r in [r1, r2, r3, r4, r5, r6, r7]:
            f_lay.addLayout(r)

        lay.addWidget(form)

        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet(f"color:{SUCCESS}; font-size:11px; font-family:{FONT_MONO}; background:transparent; border:none;")
        lay.addWidget(self.status_lbl)

        fire_btn = QPushButton("⛔  FIRE IOCTL RULE  →  KERNEL")
        fire_btn.setFixedHeight(46)
        fire_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        fire_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #3D0010, stop:1 #1A0008);
                color: {DANGER};
                border: 1px solid {DANGER}88;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 700;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background: {DANGER}22;
                border-color: {DANGER};
            }}
            QPushButton:pressed {{ background: {DANGER}33; }}
        """)
        fire_btn.clicked.connect(self._fire)
        lay.addWidget(fire_btn)
        lay.addStretch()

    def _fire(self):
        import socket as _sock
        ver_s  = self.ipver_cb.currentText()
        ver    = 4 if ver_s == "IPv4" else 6
        proto  = {"TCP (6)":6,"UDP (17)":17,"ICMP (1)":1,"Any (0)":0}.get(self.proto_cb.currentText(), 6)

        def parse_ip(t):
            t = t.strip()
            if not t or t in ("0.0.0.0","::","any",""):
                return b'\x00'*16
            af = _sock.AF_INET if ver==4 else _sock.AF_INET6
            try:
                return (_sock.inet_pton(af, t) + b'\x00'*12)[:16]
            except Exception:
                raise ValueError(f"Invalid IP: {t}")

        try:
            src_ip   = parse_ip(self.src_ip_e.text())
            dst_ip   = parse_ip(self.dst_ip_e.text())
            src_port = int(self.src_port_e.text() or "0")
            dst_port = int(self.dst_port_e.text() or "0")
            ttl_ms   = int(self.ttl_e.text() or "10000")
        except ValueError as e:
            self.status_lbl.setStyleSheet(f"color:{DANGER}; font-size:11px; font-family:{FONT_MONO}; background:transparent; border:none;")
            self.status_lbl.setText(f"✗ Input error: {e}")
            return

        self.rule_submitted.emit({
            "ip_version": ver, "proto": proto,
            "src_ip": src_ip, "dst_ip": dst_ip,
            "src_port": src_port, "dst_port": dst_port,
            "ttl_ms": ttl_ms,
        })

    def show_result(self, ok: bool, msg: str):
        color = SUCCESS if ok else DANGER
        icon  = "✓" if ok else "✗"
        self.status_lbl.setStyleSheet(f"color:{color}; font-size:11px; font-family:{FONT_MONO}; background:transparent; border:none;")
        self.status_lbl.setText(f"{icon} {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AMTE SecAI  ·  AI-Powered Intrusion Prevention")
        self.resize(1320, 860)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(f"background-color: {BG};")

        self._signals  = BackendSignals()
        self._backend  = AppBackend(self._signals)
        self._capturing = False

        self._signals.log_received.connect(self._on_log)
        self._signals.attack_detected.connect(self._on_attack)
        self._signals.stats_updated.connect(self._on_stats)
        self._signals.connection_changed.connect(self._on_connection)

        self._build_ui()

        # poll timer — 50ms
        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.setInterval(50)

        # stats timer — 1s
        self._stats_timer = QTimer()
        self._stats_timer.timeout.connect(self._update_metrics)
        self._stats_timer.start(1000)

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._build_sidebar(root)
        self._build_content(root)

    def _build_sidebar(self, parent_layout):
        sidebar = QFrame()
        sidebar.setFixedWidth(SIDEBAR_W)
        sidebar.setStyleSheet(f"""
            QFrame {{
                background: {SURFACE};
                border-right: 1px solid {BORDER};
            }}
        """)
        lay = QVBoxLayout(sidebar)
        lay.setContentsMargins(16, 32, 16, 24)
        lay.setSpacing(4)

        # logo
        logo_row = QHBoxLayout()
        dot = StatusDot(DANGER, 8)
        self._logo_dot = dot
        logo_lbl = QLabel("AMTE SecAI")
        logo_lbl.setStyleSheet(f"color:{ACCENT}; font-size:18px; font-weight:800; letter-spacing:1px; background:transparent; border:none;")
        logo_row.addWidget(dot)
        logo_row.addSpacing(8)
        logo_row.addWidget(logo_lbl)
        logo_row.addStretch()
        lay.addLayout(logo_row)

        sub_lbl = QLabel("Intrusion Prevention System")
        sub_lbl.setStyleSheet(f"color:{TEXT3}; font-size:10px; letter-spacing:0.5px; background:transparent; border:none;")
        lay.addWidget(sub_lbl)
        lay.addSpacing(24)

        # connection control
        conn_frame = QFrame()
        conn_frame.setStyleSheet(f"background:{BG}; border:1px solid {BORDER}; border-radius:8px;")
        c_lay = QVBoxLayout(conn_frame)
        c_lay.setContentsMargins(12, 10, 12, 10)
        c_lay.setSpacing(8)

        self._conn_lbl = QLabel("● DISCONNECTED")
        self._conn_lbl.setStyleSheet(f"color:{DANGER}; font-size:10px; font-weight:700; letter-spacing:1px; background:transparent; border:none;")
        c_lay.addWidget(self._conn_lbl)

        self._conn_btn = QPushButton("Connect to Driver")
        self._conn_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._conn_btn.setFixedHeight(32)
        self._conn_btn.setStyleSheet(f"""
            QPushButton {{
                background:{ACCENT}22; color:{ACCENT};
                border:1px solid {ACCENT}55; border-radius:6px;
                font-size:11px; font-weight:600;
            }}
            QPushButton:hover {{ background:{ACCENT}44; border-color:{ACCENT}; }}
        """)
        self._conn_btn.clicked.connect(self._toggle_connection)
        c_lay.addWidget(self._conn_btn)
        lay.addWidget(conn_frame)
        lay.addSpacing(20)

        # nav
        lay.addWidget(SectionHeader("NAVIGATION"))
        lay.addSpacing(4)

        self._nav_btns: dict[str, SidebarButton] = {}
        pages = [
            ("Overview",    "◈", "overview"),
            ("Live Stream", "◉", "stream"),
            ("Event Logs",  "≡", "logs"),
            ("Protection",  "⬡", "protection"),
            ("Block Rule",  "⛔", "block"),
        ]
        for label, icon, key in pages:
            btn = SidebarButton(icon, label)
            btn.clicked.connect(lambda _, k=key: self._switch_page(k))
            self._nav_btns[key] = btn
            lay.addWidget(btn)

        lay.addStretch()

        # stream control
        self._stream_btn = QPushButton("▶  Start Capture")
        self._stream_btn.setFixedHeight(38)
        self._stream_btn.setEnabled(False)
        self._stream_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stream_btn.setStyleSheet(f"""
            QPushButton {{
                background:{ACCENT}; color:{BG};
                border:none; border-radius:8px;
                font-size:12px; font-weight:700;
            }}
            QPushButton:hover {{ background:{ACCENT2}; }}
            QPushButton:disabled {{ background:{BORDER}; color:{TEXT3}; }}
        """)
        self._stream_btn.clicked.connect(self._toggle_capture)
        lay.addWidget(self._stream_btn)

        # ai status
        ai_row = QHBoxLayout()
        ai_lbl = QLabel("AI Engine:")
        ai_lbl.setStyleSheet(f"color:{TEXT3}; font-size:10px; background:transparent; border:none;")
        self._ai_lbl = QLabel("Ready" if _HIM_AVAILABLE else "Unavailable")
        self._ai_lbl.setStyleSheet(f"color:{SUCCESS if _HIM_AVAILABLE else TEXT3}; font-size:10px; font-family:{FONT_MONO}; background:transparent; border:none;")
        ai_row.addWidget(ai_lbl)
        ai_row.addStretch()
        ai_row.addWidget(self._ai_lbl)
        lay.addLayout(ai_row)

        parent_layout.addWidget(sidebar)

    def _build_content(self, parent_layout):
        self._stack = QStackedWidget()
        self._stack.setStyleSheet("background:transparent;")

        # wrap each page in a scroll + padding
        self._pages = {}
        page_constructors = {
            "overview":   OverviewPage,
            "stream":     StreamPage,
            "logs":       LogsPage,
            "protection": ProtectionPage,
            "block":      BlockRulePage,
        }
        for key, cls in page_constructors.items():
            wrapper = QWidget()
            wrapper.setStyleSheet("background:transparent;")
            w_lay = QVBoxLayout(wrapper)
            w_lay.setContentsMargins(32, 28, 32, 28)
            page = cls()
            if key == "block":
                page.rule_submitted.connect(self._on_manual_rule)
            self._pages[key] = page
            w_lay.addWidget(page)
            self._stack.addWidget(wrapper)

        parent_layout.addWidget(self._stack, 1)
        self._switch_page("overview")

    # ── page switching ────────────────────────────────────────────────────────
    def _switch_page(self, key: str):
        idx = list(self._pages.keys()).index(key)
        self._stack.setCurrentIndex(idx)
        for k, btn in self._nav_btns.items():
            btn.setChecked(k == key)
        # fade in
        w = self._stack.currentWidget()
        effect = QGraphicsOpacityEffect(w)
        w.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(180)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._page_anim = anim
        anim.start()

    # ── connection ────────────────────────────────────────────────────────────
    def _toggle_connection(self):
        if self._backend.connected:
            if self._capturing:
                self._toggle_capture()
            self._backend.disconnect_driver()
        else:
            ok, msg = self._backend.connect_driver()
            self._on_log(datetime.now().strftime("%H:%M:%S"), msg, "Info" if ok else "Critical")

    def _on_connection(self, connected: bool):
        if connected:
            self._conn_lbl.setText("● CONNECTED")
            self._conn_lbl.setStyleSheet(f"color:{SUCCESS}; font-size:10px; font-weight:700; letter-spacing:1px; background:transparent; border:none;")
            self._conn_btn.setText("Disconnect")
            self._conn_btn.setStyleSheet(f"""
                QPushButton {{
                    background:{DANGER}22; color:{DANGER};
                    border:1px solid {DANGER}55; border-radius:6px;
                    font-size:11px; font-weight:600;
                }}
                QPushButton:hover {{ background:{DANGER}44; border-color:{DANGER}; }}
            """)
            self._stream_btn.setEnabled(True)
            self._logo_dot.set_color(SUCCESS)
            self._pages["overview"].set_status(True)
        else:
            self._conn_lbl.setText("● DISCONNECTED")
            self._conn_lbl.setStyleSheet(f"color:{DANGER}; font-size:10px; font-weight:700; letter-spacing:1px; background:transparent; border:none;")
            self._conn_btn.setText("Connect to Driver")
            self._conn_btn.setStyleSheet(f"""
                QPushButton {{
                    background:{ACCENT}22; color:{ACCENT};
                    border:1px solid {ACCENT}55; border-radius:6px;
                    font-size:11px; font-weight:600;
                }}
                QPushButton:hover {{ background:{ACCENT}44; border-color:{ACCENT}; }}
            """)
            self._stream_btn.setEnabled(False)
            self._logo_dot.set_color(DANGER)
            self._pages["overview"].set_status(False)

    # ── capture ───────────────────────────────────────────────────────────────
    def _toggle_capture(self):
        if self._capturing:
            self._capturing = False
            self._poll_timer.stop()
            self._backend.stop_capture()
            self._stream_btn.setText("▶  Start Capture")
            self._stream_btn.setStyleSheet(f"""
                QPushButton {{
                    background:{ACCENT}; color:{BG};
                    border:none; border-radius:8px;
                    font-size:12px; font-weight:700;
                }}
                QPushButton:hover {{ background:{ACCENT2}; }}
            """)
            self._on_log(datetime.now().strftime("%H:%M:%S"), "Capture stopped", "Warning")
        else:
            self._capturing = True
            self._backend.start_capture()
            self._poll_timer.start()
            self._stream_btn.setText("⏸  Stop Capture")
            self._stream_btn.setStyleSheet(f"""
                QPushButton {{
                    background:{WARNING}33; color:{WARNING};
                    border:1px solid {WARNING}66; border-radius:8px;
                    font-size:12px; font-weight:700;
                }}
                QPushButton:hover {{ background:{WARNING}55; }}
            """)
            self._on_log(datetime.now().strftime("%H:%M:%S"), "Capture started — polling every 50ms", "Info")
            if _HIM_AVAILABLE and self._backend._him_ok:
                self._on_log(datetime.now().strftime("%H:%M:%S"), "AI pipeline active — XGBoost inference every 2s", "Info")

    # ── poll ──────────────────────────────────────────────────────────────────
    def _poll(self):
        if not self._capturing:
            return
        pkts = self._backend.poll_once()
        if pkts:
            self._pages["stream"].add_packets(pkts)

    def _update_metrics(self):
        metrics = self._backend.get_metrics()
        overview = self._pages["overview"]
        if hasattr(overview, "kern_vars"):
            for k, v in metrics.items():
                if k in overview.kern_vars:
                    overview.kern_vars[k].setText(f"{v:,}")

    # ── signals ───────────────────────────────────────────────────────────────
    def _on_log(self, ts: str, event: str, level: str):
        self._pages["logs"].add_entry(ts, event, level)

    def _on_attack(self, reason: str, src: str, dst: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._pages["logs"].add_entry(ts, f"[AI BLOCK] {reason}  {src} → {dst}", "Attack")
        self._ai_lbl.setText(f"⚠ {reason}")
        self._ai_lbl.setStyleSheet(f"color:{DANGER}; font-size:10px; font-family:{FONT_MONO}; background:transparent; border:none;")
        # reset after 5s
        QTimer.singleShot(5000, lambda: (
            self._ai_lbl.setText("Active"),
            self._ai_lbl.setStyleSheet(f"color:{SUCCESS}; font-size:10px; font-family:{FONT_MONO}; background:transparent; border:none;"),
        ))

    def _on_stats(self, s: dict):
        metrics = self._backend.get_metrics()
        self._pages["overview"].update_stats(s, metrics)

    def _on_manual_rule(self, rule_dict: dict):
        if not _KP_AVAILABLE:
            self._pages["block"].show_result(False, "kernel_panel not available")
            return
        try:
            rule = kp.BlockRuleV1(**{k: rule_dict[k] for k in
                ["ip_version","proto","src_ip","dst_ip","src_port","dst_port","ttl_ms"]})
            ok = self._backend.send_manual_rule(rule)
            if ok:
                self._pages["block"].show_result(True, f"Rule sent to kernel — TTL={rule_dict['ttl_ms']}ms")
                ts = datetime.now().strftime("%H:%M:%S")
                self._on_log(ts, f"Manual block rule fired — dst port {rule_dict['dst_port']} TTL={rule_dict['ttl_ms']}ms", "Warning")
            else:
                self._pages["block"].show_result(False, "Kernel rejected the rule (DeviceIoControl returned False)")
        except Exception as e:
            self._pages["block"].show_result(False, str(e))

    def closeEvent(self, e):
        self._poll_timer.stop()
        self._stats_timer.stop()
        if self._backend.connected:
            self._backend.disconnect_driver()
        e.accept()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())