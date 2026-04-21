import sys
import random
import time
import threading
from collections import deque
from datetime import datetime

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame, 
                             QScrollArea, QGraphicsOpacityEffect)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QPropertyAnimation, QEasingCurve, pyqtProperty, QSize
from PyQt6.QtGui import QColor, QPainter, QFont

# --- 1. BACKEND LOGIC ---
class BackendSignals(QObject):
    log_received = pyqtSignal(str, str, str) 

class DashboardBackend(QObject):
    def __init__(self):
        super().__init__()
        self.signals = BackendSignals()
        self.logs = deque(maxlen=50)
        self.running = True
        
        self.thread = threading.Thread(target=self.simulate_incoming_logs, daemon=True)
        self.thread.start()

    def get_live_stats(self):
        return {
            "files_scanned": f"{14200 + random.randint(1, 500):,}",
            "threats_blocked": "3",
            "active_processes": f"{90 + random.randint(1, 5)}",
            "cpu_usage": f"{random.randint(2, 8)}%"
        }
    
    def get_logs(self):
        return list(self.logs)[::-1]

    def simulate_incoming_logs(self):
        events = [
            ("Outbound Connection Blocked (192.168.1.55)", "Warning"),
            ("Database Definition Sync (v2025.4.1)", "Info"),
            ("Process Whitelisted (code.exe)", "Info"),
            ("Heuristic Analysis: Clean", "Info"),
            ("Unauthorized Access Attempt (Admin)", "Critical"),
            ("Memory Integrity Check Passed", "Info")
        ]
        while self.running:
            time.sleep(random.uniform(2.0, 5.0))
            event, level = random.choice(events)
            now = datetime.now().strftime("%I:%M:%S %p")
            
            self.logs.append((now, event, level))
            self.signals.log_received.emit(now, event, level)

# --- 2. CUSTOM WIDGETS ---

class ModernToggle(QWidget):
    def __init__(self, parent=None, checked=False):
        super().__init__(parent)
        self.setFixedSize(50, 28)
        self._checked = checked
        self._circle_position = 3.0 if not checked else 25.0
        
        self.animation = QPropertyAnimation(self, b"circle_position", self)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.animation.setDuration(300)

    @pyqtProperty(float)
    def circle_position(self):
        return self._circle_position

    @circle_position.setter
    def circle_position(self, pos):
        self._circle_position = pos
        self.update() 

    def mouseReleaseEvent(self, e):
        self._checked = not self._checked
        self.animation.stop()
        if self._checked:
            self.animation.setStartValue(self._circle_position)
            self.animation.setEndValue(25.0)
        else:
            self.animation.setStartValue(self._circle_position)
            self.animation.setEndValue(3.0)
        self.animation.start()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        track_color = QColor("#2ecc71") if self._checked else QColor("#404040")
        p.setBrush(track_color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, 0, 50, 28, 14, 14)

        p.setBrush(QColor("white"))
        p.drawEllipse(int(self._circle_position), 3, 22, 22)
        p.end()

class StatCard(QFrame):
    def __init__(self, title, value):
        super().__init__()
        self.setStyleSheet("background-color: #2b2b2b; border-radius: 15px;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        t_label = QLabel(title)
        t_label.setStyleSheet("color: gray; font-size: 14px; font-weight: bold; background: transparent;")
        
        self.v_label = QLabel(value)
        self.v_label.setStyleSheet("color: white; font-size: 28px; font-weight: bold; background: transparent;")
        
        layout.addWidget(t_label)
        layout.addWidget(self.v_label)

    def update_value(self, new_val):
        try:
            self.v_label.setText(new_val)
        except RuntimeError:
            pass

class SettingRow(QFrame):
    def __init__(self, title, description, checked=True):
        super().__init__()
        self.setStyleSheet("background-color: #2b2b2b; border-radius: 10px;")
        self.setFixedHeight(80)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        
        text_container = QWidget()
        text_container.setStyleSheet("background: transparent;")
        t_layout = QVBoxLayout(text_container)
        t_layout.setContentsMargins(0,0,0,0)
        t_layout.setSpacing(5)
        
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: white; font-weight: bold; font-size: 14px; background: transparent;")
        lbl_desc = QLabel(description)
        lbl_desc.setStyleSheet("color: gray; font-size: 12px; background: transparent;")
        
        t_layout.addWidget(lbl_title)
        t_layout.addWidget(lbl_desc)
        
        toggle = ModernToggle(checked=checked)
        
        layout.addWidget(text_container)
        layout.addStretch()
        layout.addWidget(toggle)

class LogRow(QFrame):
    def __init__(self, time, event, level):
        super().__init__()
        self.setStyleSheet("background-color: transparent;")
        
        # FIXED: Removed invalid method call 'setClipsChildren'
        # QFrame automatically handles geometry
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        time_lbl = QLabel(time)
        time_lbl.setFixedWidth(90)
        time_lbl.setStyleSheet("color: gray; font-family: Consolas;")
        
        event_lbl = QLabel(event)
        event_lbl.setStyleSheet("color: #e0e0e0; font-family: Segoe UI;")
        
        level_lbl = QLabel(level)
        level_lbl.setFixedWidth(80)
        level_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        bg = "#3498db"
        txt = "white"
        if level == "Warning": bg, txt = "#f1c40f", "black"
        if level == "Critical": bg, txt = "#e74c3c", "white"
            
        level_lbl.setStyleSheet(f"background-color: {bg}; color: {txt}; border-radius: 10px; padding: 4px; font-weight: bold; font-size: 11px;")
        
        layout.addWidget(time_lbl)
        layout.addWidget(event_lbl)
        layout.addWidget(level_lbl)

# --- 3. MAIN APPLICATION ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AMTE Security Center")
        self.resize(1200, 800)
        self.setMinimumSize(1000, 700)
        self.setStyleSheet("background-color: #1a1a1a;")

        self.backend = DashboardBackend()
        self.backend.signals.log_received.connect(self.add_log_entry)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.setup_sidebar()
        self.setup_content_area()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(2000)

        self.current_view_name = None
        self.current_view = None
        self.stat_widgets = {}
        
        self.switch_view("Overview")

    def setup_sidebar(self):
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(260)
        self.sidebar.setStyleSheet("background-color: #1a1a1a; border-right: 1px solid #333;")
        layout = QVBoxLayout(self.sidebar)
        layout.setContentsMargins(20, 40, 20, 20)
        layout.setSpacing(10)

        title = QLabel("AMTE SecAI")
        title.setStyleSheet("color: white; font-size: 26px; font-weight: bold; font-family: Segoe UI; border: none;")
        subtitle = QLabel("Enterprise Dashboard")
        subtitle.setStyleSheet("color: gray; font-size: 12px; margin-bottom: 40px; border: none;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.nav_buttons = {}
        for name in ["Overview", "Protection", "Logs"]:
            btn = QPushButton(f"  {name}")
            btn.setCheckable(True)
            btn.setFixedHeight(45)
            btn.setStyleSheet("""
                QPushButton { text-align: left; padding-left: 20px; color: #b0b0b0; background-color: transparent; border-radius: 8px; font-size: 14px; font-weight: 500; border: none; }
                QPushButton:checked { background-color: #2b2b2b; color: white; font-weight: bold; }
                QPushButton:hover { background-color: #222; color: white; }
            """)
            btn.clicked.connect(lambda checked, n=name: self.switch_view(n))
            self.nav_buttons[name] = btn
            layout.addWidget(btn)

        layout.addStretch()
        
        status_label = QLabel("Engine Status:")
        status_label.setStyleSheet("color: gray; font-weight: bold; border: none;")
        self.online_lbl = QLabel("● ONLINE")
        self.online_lbl.setStyleSheet("color: #2ecc71; font-weight: bold; border: none;")
        layout.addWidget(status_label)
        layout.addWidget(self.online_lbl)
        
        self.main_layout.addWidget(self.sidebar)

    def setup_content_area(self):
        self.content_container = QWidget()
        self.content_container.setStyleSheet("background-color: #1a1a1a;")
        self.content_layout = QVBoxLayout(self.content_container)
        self.content_layout.setContentsMargins(30, 30, 30, 30)
        self.main_layout.addWidget(self.content_container)

    def get_overview_view(self):
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        card = QFrame()
        card.setFixedHeight(140)
        card.setStyleSheet("background-color: #2ecc71; border-radius: 15px;")
        card_layout = QHBoxLayout(card)
        card_layout.setContentsMargins(30, 0, 30, 0)
        
        icon = QLabel("🛡️")
        icon.setStyleSheet("font-size: 50px; background: transparent; border: none;")
        text_layout = QVBoxLayout()
        text_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        t1 = QLabel("SYSTEM SECURE")
        t1.setStyleSheet("color: white; font-size: 28px; font-weight: bold; background: transparent; border: none;")
        t2 = QLabel("Real-time protection active")
        t2.setStyleSheet("color: white; font-size: 14px; background: transparent; border: none;")
        text_layout.addWidget(t1)
        text_layout.addWidget(t2)
        card_layout.addWidget(icon)
        card_layout.addLayout(text_layout)
        card_layout.addStretch()
        layout.addWidget(card)

        stats_row = QHBoxLayout()
        self.stat_widgets = {}
        for k, title in [("files_scanned", "Files Scanned"), ("threats_blocked", "Threats Blocked"), ("cpu_usage", "CPU Load")]:
            w = StatCard(title, "...")
            self.stat_widgets[k] = w
            stats_row.addWidget(w)
        layout.addLayout(stats_row)

        toggle_frame = QFrame()
        toggle_frame.setStyleSheet("background-color: #2b2b2b; border-radius: 15px;")
        tf_layout = QVBoxLayout(toggle_frame)
        tf_layout.setContentsMargins(20, 20, 20, 20)
        lbl = QLabel("Active Modules")
        lbl.setStyleSheet("color: gray; font-weight: bold; font-size: 14px; background: transparent;")
        tf_layout.addWidget(lbl)
        
        row = QHBoxLayout()
        for text in ["Real-Time AI", "Cloud Lookup", "Heuristic Engine"]:
            container = QWidget()
            container.setStyleSheet("background: transparent;")
            cl = QHBoxLayout(container)
            t = ModernToggle(checked=True)
            l = QLabel(text)
            l.setStyleSheet("color: white; font-weight: bold; font-size: 13px; background: transparent;")
            cl.addWidget(t)
            cl.addWidget(l)
            cl.addStretch()
            row.addWidget(container)
        tf_layout.addLayout(row)
        layout.addWidget(toggle_frame)
        return view

    def get_logs_view(self):
        view = QWidget()
        layout = QVBoxLayout(view)
        lbl = QLabel("Security Event Logs")
        lbl.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        layout.addWidget(lbl)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; } QWidget { background: #1a1a1a; }")
        
        self.log_container = QWidget()
        self.log_layout = QVBoxLayout(self.log_container)
        self.log_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.log_layout.setSpacing(5)
        
        scroll.setWidget(self.log_container)
        layout.addWidget(scroll)
        return view

    def get_protection_view(self):
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        lbl = QLabel("Advanced Protection Settings")
        lbl.setStyleSheet("color: white; font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(lbl)
        layout.addWidget(SettingRow("Deep Packet Inspection", "Analyzes network traffic for anomalies.", checked=True))
        layout.addWidget(SettingRow("Kernel-Level Hooks", "Prevents unauthorized driver tampering.", checked=True))
        layout.addWidget(SettingRow("Cloud Sandbox", "Automatically uploads suspicious binaries.", checked=False))
        layout.addWidget(SettingRow("USB Shield", "Scans removable media immediately.", checked=True))
        layout.addStretch()
        return view

    def switch_view(self, view_name):
        self.current_view_name = view_name
        for name, btn in self.nav_buttons.items():
            btn.setChecked(name == view_name)

        if self.current_view:
            self.current_view.hide()
            self.content_layout.removeWidget(self.current_view)
            self.current_view.deleteLater()
            self.current_view = None
            self.stat_widgets = {} 

        if view_name == "Overview":
            self.current_view = self.get_overview_view()
        elif view_name == "Logs":
            self.current_view = self.get_logs_view()
            # Restore logs - NO animation for restore to keep it snappy
            for log in self.backend.get_logs():
                self.add_log_entry(*log, animate=False)
        else:
            self.current_view = self.get_protection_view()

        # Simple Page Fade In
        self.current_view.setVisible(False)
        self.content_layout.addWidget(self.current_view)
        
        self.effect = QGraphicsOpacityEffect(self.current_view)
        self.current_view.setGraphicsEffect(self.effect)
        self.anim = QPropertyAnimation(self.effect, b"opacity")
        self.anim.setDuration(250)
        self.anim.setStartValue(0)
        self.anim.setEndValue(1)
        self.anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.current_view.setVisible(True)
        self.anim.start()

    def update_stats(self):
        if self.current_view_name == "Overview" and self.stat_widgets:
            stats = self.backend.get_live_stats()
            try:
                if 'files_scanned' in self.stat_widgets:
                    self.stat_widgets['files_scanned'].update_value(stats['files_scanned'])
                    self.stat_widgets['threats_blocked'].update_value(stats['threats_blocked'])
                    self.stat_widgets['cpu_usage'].update_value(stats['cpu_usage'])
            except RuntimeError:
                pass

    def add_log_entry(self, time, event, level, animate=True):
        if self.current_view_name == "Logs" and hasattr(self, 'log_layout'):
            try:
                row = LogRow(time, event, level)
                self.log_layout.insertWidget(0, row)
                
                # FIXED: Use Geometry Animation (Slide/Expand) instead of Opacity
                if animate:
                    row.setFixedHeight(0) # Start closed
                    anim = QPropertyAnimation(row, b"maximumHeight")
                    anim.setDuration(300)
                    anim.setStartValue(0)
                    anim.setEndValue(40) # Target height
                    anim.setEasingCurve(QEasingCurve.Type.OutQuad)
                    row.anim = anim # Keep reference
                    anim.start()
                else:
                    # Static load
                    row.setFixedHeight(40)

                if self.log_layout.count() > 20:
                    item = self.log_layout.takeAt(20)
                    if item and item.widget():
                        item.widget().deleteLater()
            except RuntimeError:
                pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())