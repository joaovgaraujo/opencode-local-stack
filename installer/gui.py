"""Tkinter installer wizard. Tkinter ships with the standard CPython installer
on Windows; on Linux it's sometimes a separate package (python3-tk) - if the
import fails, install.py falls back to the CLI wizard automatically.

This module only handles presentation: hardware summary, model/quant picker
filtered+sorted by fit, and a live log during install. The actual install
pipeline (download, server start, tests, OpenCode setup) lives in install.py
and is passed in as `run_pipeline`, so CLI and GUI never diverge in behavior.

Branches on hwdetect.pick_engine(hw): 'llamacpp' (Windows/Linux, GGUF, has a
primary/conservative profile) vs 'rapidmlx' (macOS/Apple Silicon, MLX repos,
no profile split - see catalog.estimate_mlx_requirements). The picked
model/quant/profile is handed to run_pipeline unchanged either way; profile
is None for the rapidmlx path and install.py's wrapper dispatches on that.
"""
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from . import catalog, hwdetect

VERDICT_COLOR = {"fits": "#1a7f37", "tight": "#9a6700", "no": "#8b8b8b"}
VERDICT_LABEL = {"fits": "Fits", "tight": "Tight fit", "no": "Won't fit"}


def _build_rows(hw, show_experimental):
    rows = []
    for model, quant, profile in catalog.all_variants():
        experimental = quant.get("experimental")
        if experimental and not show_experimental:
            continue
        if experimental and profile != "primary":
            continue  # only show one row per experimental quant, not one per profile
        if not experimental and not quant.get("default") and profile != "primary":
            continue  # keep the list short: conservative profile only alongside the default quant
        verdict = catalog.fit_verdict(model, quant, profile, hw.vram_free_gb or hw.vram_total_gb,
                                       hw.ram_free_gb, hw.disk_free_gb)
        need_vram, need_ram = catalog.estimate_requirements(model, quant, profile)
        rows.append((model, quant, profile, verdict, need_vram, need_ram))
    order = {"fits": 0, "tight": 1, "no": 2}
    rows.sort(key=lambda r: (order[r[3]], r[0]["total_params_b"]))
    return rows


def _build_mlx_rows(hw):
    rows = []
    for model, quant in catalog.all_mlx_variants():
        verdict = catalog.mlx_fit_verdict(model, quant, hw.ram_total_gb, hw.ram_free_gb,
                                           hw.disk_free_gb)
        need_ram = catalog.estimate_mlx_requirements(model, quant)
        rows.append((model, quant, verdict, need_ram))
    order = {"fits": 0, "tight": 1, "no": 2}
    rows.sort(key=lambda r: (order[r[2]], r[0]["total_params_b"]))
    return rows


class InstallerWizard:
    def __init__(self, hw, run_pipeline):
        self.hw = hw
        self.engine = hwdetect.pick_engine(hw)
        self.run_pipeline = run_pipeline
        self.root = tk.Tk()
        self.root.title("opencode-local installer")
        self.root.geometry("880x560")
        self.log_queue = queue.Queue()
        self.selection = None
        self.show_experimental = tk.BooleanVar(value=False)
        self.skip_tests = tk.BooleanVar(value=False)
        self._build_picker_frame()

    # ---- Frame 1: hardware + model picker ----------------------------------
    def _build_picker_frame(self):
        self.picker = ttk.Frame(self.root, padding=12)
        self.picker.pack(fill="both", expand=True)

        hw_box = ttk.LabelFrame(self.picker, text="Detected hardware", padding=8)
        hw_box.pack(fill="x", pady=(0, 10))
        for line in self.hw.summary_lines():
            ttk.Label(hw_box, text=line).pack(anchor="w")

        engine_label = "rapid-mlx (Apple Silicon)" if self.engine == "rapidmlx" else "llama.cpp"
        ttk.Label(self.picker, text=f"Pick a model + quantization for {engine_label} "
                                     f"(sorted by fit on this machine):",
                  font=("", 10, "bold")).pack(anchor="w")

        if self.engine == "rapidmlx":
            columns = ("model", "quant", "mem", "fit")
            headings = {"model": "Model", "quant": "Quantization", "mem": "Est. Unified Memory",
                        "fit": "Fit"}
            widths = {"model": 220, "quant": 300, "mem": 150, "fit": 110}
        else:
            columns = ("model", "quant", "vram", "ram", "fit")
            headings = {"model": "Model", "quant": "Quantization", "vram": "Est. VRAM",
                        "ram": "Est. RAM", "fit": "Fit"}
            widths = {"model": 200, "quant": 300, "vram": 90, "ram": 90, "fit": 110}

        self.tree = ttk.Treeview(self.picker, columns=columns, show="headings", height=14)
        for c in columns:
            self.tree.heading(c, text=headings[c])
            self.tree.column(c, width=widths[c], anchor="w")
        self.tree.pack(fill="both", expand=True, pady=(4, 8))
        self.tree.tag_configure("fits", foreground=VERDICT_COLOR["fits"])
        self.tree.tag_configure("tight", foreground=VERDICT_COLOR["tight"])
        self.tree.tag_configure("no", foreground=VERDICT_COLOR["no"])

        self._populate_tree()

        opts = ttk.Frame(self.picker)
        opts.pack(fill="x")
        if self.engine != "rapidmlx":
            ttk.Checkbutton(opts, text="Show experimental TurboQuant variants (needs a custom "
                                        "llama.cpp fork - see docs/TURBOQUANT.md)",
                            variable=self.show_experimental,
                            command=self._populate_tree).pack(anchor="w")
        ttk.Checkbutton(opts, text="Skip validation tests after install",
                        variable=self.skip_tests).pack(anchor="w")

        btns = ttk.Frame(self.picker)
        btns.pack(fill="x", pady=(10, 0))
        ttk.Button(btns, text="Install", command=self._on_install).pack(side="right")

    def _populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        if self.engine == "rapidmlx":
            self._rows = _build_mlx_rows(self.hw)
            for i, (model, quant, verdict, need_ram) in enumerate(self._rows):
                self.tree.insert("", "end", iid=str(i), tags=(verdict,), values=(
                    model["display_name"], quant["label"], f"{need_ram:.1f} GB",
                    VERDICT_LABEL[verdict],
                ))
        else:
            self._rows = _build_rows(self.hw, self.show_experimental.get())
            for i, (model, quant, profile, verdict, need_vram, need_ram) in enumerate(self._rows):
                label = quant["label"] + (f"  [{profile}]" if profile != "primary" else "")
                self.tree.insert("", "end", iid=str(i), tags=(verdict,), values=(
                    model["display_name"], label, f"{need_vram:.1f} GB", f"{need_ram:.1f} GB",
                    VERDICT_LABEL[verdict],
                ))

    def _on_install(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("No selection", "Pick a model/quantization first.")
            return
        row = self._rows[int(sel[0])]
        if self.engine == "rapidmlx":
            model, quant, verdict, _ = row
            profile = None
        else:
            model, quant, profile, verdict, _, _ = row
        if verdict == "no":
            if not messagebox.askyesno("Doesn't fit",
                                        "This installer estimates this option will NOT fit on "
                                        "your hardware. Continue anyway?"):
                return
        self.selection = (model, quant, profile)
        self.picker.destroy()
        self._build_progress_frame()
        threading.Thread(target=self._run_install_thread, daemon=True).start()
        self.root.after(150, self._poll_log)

    # ---- Frame 2: progress/log ---------------------------------------------
    def _build_progress_frame(self):
        self.progress_frame = ttk.Frame(self.root, padding=12)
        self.progress_frame.pack(fill="both", expand=True)
        ttk.Label(self.progress_frame, text="Installing...", font=("", 11, "bold")).pack(anchor="w")
        self.pbar = ttk.Progressbar(self.progress_frame, mode="determinate", maximum=100)
        self.pbar.pack(fill="x", pady=8)
        self.log_text = tk.Text(self.progress_frame, height=24, wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True)
        self.close_btn = ttk.Button(self.progress_frame, text="Close", command=self.root.destroy,
                                     state="disabled")
        self.close_btn.pack(pady=(8, 0), anchor="e")

    def _log(self, msg):
        self.log_queue.put(("log", msg))

    def _progress(self, done, total, label=""):
        pct = (done / total * 100) if total else 0
        self.log_queue.put(("progress", pct, label))

    def _run_install_thread(self):
        model, quant, profile = self.selection
        try:
            self.run_pipeline(model, quant, profile, self.hw,
                               skip_tests=self.skip_tests.get(),
                               log=self._log, progress=self._progress)
            self.log_queue.put(("done", True))
        except Exception as e:  # surface any failure into the log instead of a silent crash
            self.log_queue.put(("log", f"\n[FAILED] {e}"))
            self.log_queue.put(("done", False))

    def _poll_log(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item[0] == "log":
                    self.log_text.configure(state="normal")
                    self.log_text.insert("end", item[1] + "\n")
                    self.log_text.see("end")
                    self.log_text.configure(state="disabled")
                elif item[0] == "progress":
                    self.pbar["value"] = item[1]
                elif item[0] == "done":
                    self.close_btn.configure(state="normal")
                    return
        except queue.Empty:
            pass
        self.root.after(150, self._poll_log)

    def run(self):
        self.root.mainloop()


def run_gui(hw, run_pipeline):
    wizard = InstallerWizard(hw, run_pipeline)
    wizard.run()
