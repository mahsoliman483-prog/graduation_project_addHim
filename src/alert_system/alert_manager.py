import customtkinter as ctk
import sys
import winreg

def get_system_theme():
    try:
        registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
        key = winreg.OpenKey(registry, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
        value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
        return "Light" if value == 1 else "Dark"
    except:
        return "Dark"

ctk.set_appearance_mode(get_system_theme())
ctk.set_default_color_theme("blue") 

class AlertManager:
    def __init__(self):
        self.user_decision = None
        self.app = None
        
        self.c = {
            "success_green": "#2ecc71",   
            "success_hover": "#27ae60",
            "error_red": "#e74c3c",       
            "error_hover": "#c0392b",
            "card_bg": ("gray85", "gray18"),
            "term_bg": ("gray90", "#1e1e1e")
        }

    def trigger_alert(self, threat_type, severity, description, technical_data=None):
        """
        Displays the Alert Window with dynamic data.
        
        technical_data (dict): Optional dictionary for advanced details.
        Expected keys: 'proc_id', 'path', 'behavior', 'gpu', 'engine', 'status'
        """
        self.app = ctk.CTk()
        self.app.title("AMTE Security Action")
        self.app.resizable(False, False)
        self.app.protocol("WM_DELETE_WINDOW", lambda: None)
        self.app.attributes("-topmost", True)
        self.app.attributes("-alpha", 1.0)

        w, h = 800, 600
        x = (self.app.winfo_screenwidth() // 2) - (w // 2)
        y = (self.app.winfo_screenheight() // 2) - (h // 2)
        self.app.geometry(f"{w}x{h}+{x}+{y}")

        # Set defaults if data is missing
        default_tech = {
            "proc_id": "N/A",
            "path": "Unknown",
            "behavior": "Anomaly Detected",
            "gpu": "Integrated Graphics", 
            "engine": "Standard Heuristic",
            "status": "WARNING"
        }
        
        # Merge defaults with provided data
        if technical_data:
            self.tech_data = {**default_tech, **technical_data}
        else:
            self.tech_data = default_tech

        self.threat_data = (threat_type, severity, description)
        self.build_main_ui()
        self.app.mainloop()
        
        return self.user_decision

    def build_main_ui(self):
        for widget in self.app.winfo_children(): widget.destroy()
        
        threat, severity, desc = self.threat_data
        tech = self.tech_data # Shorthand
        
        # --- HEADER ---
        header = ctk.CTkFrame(self.app, fg_color="transparent")
        header.pack(fill="x", padx=30, pady=(30, 10))
        
        ctk.CTkLabel(header, text="🛡️", font=("Segoe UI Emoji", 60)).pack(side="left", padx=(0, 20))
        
        text_frame = ctk.CTkFrame(header, fg_color="transparent")
        text_frame.pack(side="left")
        ctk.CTkLabel(text_frame, text="Security Threat Detected", font=("Segoe UI", 30, "bold")).pack(anchor="w")
        ctk.CTkLabel(text_frame, text="AMTE SecAI Real-Time Protection", font=("Segoe UI", 16), text_color="gray").pack(anchor="w")

        # --- INFO CARD ---
        card = ctk.CTkFrame(self.app, corner_radius=20, fg_color=self.c["card_bg"]) 
        card.pack(fill="x", padx=30, pady=10)
        
        info_row = ctk.CTkFrame(card, fg_color="transparent")
        info_row.pack(fill="x", padx=25, pady=(25, 10))
        
        # Threat Type
        col1 = ctk.CTkFrame(info_row, fg_color="transparent")
        col1.pack(side="left")
        ctk.CTkLabel(col1, text="THREAT TYPE", font=("Segoe UI", 11, "bold"), text_color="gray").pack(anchor="w")
        ctk.CTkLabel(col1, text=threat, font=("Consolas", 18, "bold")).pack(anchor="w")

        # Severity Badge
        col2 = ctk.CTkFrame(info_row, fg_color="transparent")
        col2.pack(side="left", padx=60)
        ctk.CTkLabel(col2, text="SEVERITY", font=("Segoe UI", 11, "bold"), text_color="gray").pack(anchor="w")
        badge = ctk.CTkButton(col2, text=severity, font=("Segoe UI", 12, "bold"), 
                              fg_color=self.c["error_red"], hover=False, 
                              height=24, corner_radius=12, width=80)
        badge.pack(anchor="w", pady=2)

        # Scrollable "Terminal" Log - NOW DYNAMIC
        ctk.CTkLabel(card, text="TECHNICAL DETAILS", font=("Segoe UI", 11, "bold"), text_color="gray").pack(anchor="w", padx=25)
        
        log_box = ctk.CTkTextbox(card, height=100, corner_radius=10, 
                                 fg_color=self.c["term_bg"], text_color=("gray10", "gray80"),
                                 font=("Consolas", 12))
        log_box.pack(fill="x", padx=25, pady=(5, 25))
        
        # Dynamic Text Formatting
        log_text = (
            f"{desc}\n\n"
            f"> Process ID: {tech['proc_id']}\n"
            f"> Path:       {tech['path']}\n"
            f"> Behavior:   {tech['behavior']}"
        )
        
        log_box.insert("0.0", log_text)
        log_box.configure(state="disabled") 

        # --- QUESTION ---
        ctk.CTkLabel(self.app, text="Action Required: Neutralize this threat?", font=("Segoe UI", 18, "bold")).pack(pady=(15, 0), anchor="w", padx=30)

        # --- BUTTONS ---
        btn_frame = ctk.CTkFrame(self.app, fg_color="transparent")
        btn_frame.pack(side="bottom", fill="x", padx=30, pady=20)

        ctk.CTkButton(btn_frame, text="Ignore Risk", font=("Segoe UI", 15, "bold"), height=50, width=180, corner_radius=25,
                      fg_color="transparent", border_width=2, border_color=self.c["error_red"], text_color=("gray20", "gray80"),
                      hover_color=self.c["error_red"], command=self.show_confirmation).pack(side="left")

        ctk.CTkButton(btn_frame, text="Neutralize Threat", font=("Segoe UI", 15, "bold"), height=50, corner_radius=25,
                      fg_color=self.c["success_green"], hover_color=self.c["success_hover"],
                      command=self.on_neutralize).pack(side="right", fill="x", expand=True, padx=(15, 0))

        # --- DYNAMIC STATUS FOOTER ---
        footer = ctk.CTkFrame(self.app, height=30, corner_radius=0, fg_color=("gray90", "#111111"))
        footer.pack(side="bottom", fill="x")
        
        # Dynamic Footer Data
        status_text = f"  ● System Status: {tech['status']}"
        engine_text = f"Engine: {tech['engine']} | GPU: {tech['gpu']}  "
        
        ctk.CTkLabel(footer, text=status_text, font=("Segoe UI", 10, "bold"), text_color=self.c["error_red"]).pack(side="left")
        ctk.CTkLabel(footer, text=engine_text, font=("Consolas", 10), text_color="gray").pack(side="right")

    def show_confirmation(self):
        for widget in self.app.winfo_children(): widget.destroy()
        ctk.CTkLabel(self.app, text="⚠️", font=("Segoe UI Emoji", 72)).pack(pady=(60, 10))
        ctk.CTkLabel(self.app, text="Are you sure?", font=("Segoe UI", 32, "bold")).pack()
        ctk.CTkLabel(self.app, text="Ignoring this threat puts your system at risk.", font=("Segoe UI", 16), text_color="gray").pack(pady=10)
        
        btn_frame = ctk.CTkFrame(self.app, fg_color="transparent")
        btn_frame.pack(pady=40)
        
        ctk.CTkButton(btn_frame, text="Go Back", font=("Segoe UI", 15, "bold"), height=50, width=200, corner_radius=25,
                      fg_color=self.c["success_green"], hover_color=self.c["success_hover"], command=self.build_main_ui).pack(side="left", padx=15)
        
        ctk.CTkButton(btn_frame, text="Yes, Ignore", font=("Segoe UI", 15, "bold"), height=50, width=200, corner_radius=25,
                      fg_color=self.c["error_red"], hover_color=self.c["error_hover"], command=self.on_ignore).pack(side="left", padx=15)

    def on_neutralize(self):
        self.user_decision = "BLOCK"
        self.app.destroy()

    def on_ignore(self):
        self.user_decision = "ALLOW"
        self.app.destroy()

# --- HOW TO USE THIS IN YOUR MAIN APP ---
if __name__ == "__main__":
    alert = AlertManager()
    
    # EXAMPLE: Real Data coming from your Backend Logic
    data = {
        "proc_id": 9821,
        "path": "C:\\Users\\Admin\\AppData\\Local\\Temp\\malware.exe",
        "behavior": "High-Frequency File Modification (Ransomware Pattern)",
        "gpu": "NVIDIA RTX 3050 Ti (CUDA 12.1)",
        "engine": "AMTE Hybrid DNN",
        "status": "ACTION REQUIRED"
    }

    alert.trigger_alert(
        threat_type="Ransomware.Locky", 
        severity="CRITICAL", 
        description="A process is attempting to modify multiple system files rapidly.",
        technical_data=data  # Pass the dictionary here!
    )