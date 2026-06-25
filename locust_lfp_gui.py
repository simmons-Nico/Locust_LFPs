#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tkinter GUI for the Locust LFP processing pipeline."""

from __future__ import annotations

import contextlib
import glob
import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk


SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_SCRIPT = SCRIPT_DIR / "plot_baseline_post_firing_rates.py"
CSV_SPIKE_PLOT_SCRIPT = SCRIPT_DIR / "CSV_to_spike_counts.py"
SPIKE_SCRIPT = SCRIPT_DIR / "Spike Count Multiple CSVs ordered.py"
RHS_SCRIPT = SCRIPT_DIR / "RHS_to_CSV(simple).py"
RHD_SCRIPT = SCRIPT_DIR / "RHD_to_CSV (simple).py"
BROWSE_START_DIR = Path(r"C:\Users\simmons\Desktop\Exploring PSDs")
DEFAULT_DATA_DIR = BROWSE_START_DIR

STEP_COLORS = {
    "raw": "#0f766e",
    "spike": "#b45309",
    "plot": "#2563eb",
    "success": "#15803d",
    "danger": "#b91c1c",
    "surface": "#f8fafc",
    "border": "#cbd5e1",
    "text": "#0f172a",
    "muted": "#475569",
}

EPOCH_ROW_COLORS = {
    "baseline": "#dbeafe",
    "stimulation": "#fef3c7",
    "post": "#dcfce7",
    "unlabeled": "#f1f5f9",
}


if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


class QueueWriter:
    def __init__(self, log_queue: queue.Queue[str]):
        self.log_queue = log_queue

    def write(self, text: str) -> None:
        if text:
            self.log_queue.put(text)

    def flush(self) -> None:
        pass


def load_script_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_csv_list(text: str) -> list[str] | None:
    items = [item.strip() for item in text.split(",") if item.strip()]
    return items or None


def parse_diff_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError("Differential pairs must look like A-008,A-010; A-011,A-012")
        pairs.append((parts[0], parts[1]))
    return pairs


def plot_bin_seconds(label: str) -> float:
    lookup = {
        "30 sec": 30.0,
        "1 min": 60.0,
        "2 min": 120.0,
        "3 min": 180.0,
        "4 min": 240.0,
        "5 min": 300.0,
        "10 min": 600.0,
    }
    if label in lookup:
        return lookup[label]
    raw = label.strip().lower()
    if raw.endswith("sec"):
        return float(raw.replace("sec", "").strip())
    if raw.endswith("min"):
        return float(raw.replace("min", "").strip()) * 60.0
    return float(raw)


class LocustPipelineApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Locust LFP Pipeline")
        self.root.geometry("1180x820")
        self.root.minsize(960, 560)
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker_thread: threading.Thread | None = None

        self.recordings: list[dict[str, str]] = []

        self.status_var = tk.StringVar(value="Ready")
        self.stage_var = tk.StringVar(value="Configure a step or run the full pipeline")
        self._configure_styles()
        self._build_ui()
        self.root.after(100, self._drain_log_queue)

    def _configure_styles(self) -> None:
        self.style = ttk.Style(self.root)
        with contextlib.suppress(tk.TclError):
            self.style.theme_use("clam")
        self.root.configure(bg=STEP_COLORS["surface"])
        self.style.configure("TFrame", background=STEP_COLORS["surface"])
        self.style.configure("TLabelframe", background=STEP_COLORS["surface"], bordercolor=STEP_COLORS["border"])
        self.style.configure("TLabelframe.Label", background=STEP_COLORS["surface"], foreground=STEP_COLORS["text"])
        self.style.configure("TLabel", background=STEP_COLORS["surface"], foreground=STEP_COLORS["text"])
        self.style.configure("Hint.TLabel", foreground=STEP_COLORS["muted"])
        self.style.configure("Accent.TButton", background=STEP_COLORS["plot"], foreground="white", padding=(12, 7))
        self.style.map("Accent.TButton", background=[("active", "#1d4ed8")])
        self.style.configure("Raw.TButton", background=STEP_COLORS["raw"], foreground="white", padding=(10, 6))
        self.style.map("Raw.TButton", background=[("active", "#115e59")])
        self.style.configure("Spike.TButton", background=STEP_COLORS["spike"], foreground="white", padding=(10, 6))
        self.style.map("Spike.TButton", background=[("active", "#92400e")])
        self.style.configure("Plot.TButton", background=STEP_COLORS["plot"], foreground="white", padding=(10, 6))
        self.style.map("Plot.TButton", background=[("active", "#1d4ed8")])
        self.style.configure("Secondary.TButton", padding=(9, 5))
        self.style.configure("TNotebook", background=STEP_COLORS["surface"], borderwidth=0)
        self.style.configure("TNotebook.Tab", padding=(16, 8), font=("Segoe UI", 10, "bold"))

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)

        self._build_top_bar()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=12, pady=(8, 6))
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self.raw_tab = ttk.Frame(self.notebook, padding=12)
        self.spike_tab = ttk.Frame(self.notebook, padding=12)
        self.plot_tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.raw_tab, text="1 Raw to CSV")
        self.notebook.add(self.spike_tab, text="2 Spike counts")
        self.notebook.add(self.plot_tab, text="3 Plots")

        self._build_raw_tab()
        self._build_spike_tab()
        self._build_plot_tab()
        self._build_log()

    def _build_top_bar(self) -> None:
        bar = ttk.Frame(self.root, padding=(12, 12, 12, 4))
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(1, weight=1)

        title = ttk.Label(bar, text="Locust LFP Pipeline", font=("Segoe UI", 16, "bold"))
        title.grid(row=0, column=0, sticky="w")
        ttk.Label(bar, textvariable=self.stage_var, style="Hint.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 0))

        chip_frame = ttk.Frame(bar)
        chip_frame.grid(row=0, column=1, rowspan=2, sticky="w", padx=24)
        self._step_chip(chip_frame, "1 Raw", STEP_COLORS["raw"], 0)
        self._step_chip(chip_frame, "2 Spikes", STEP_COLORS["spike"], 1)
        self._step_chip(chip_frame, "3 Plots", STEP_COLORS["plot"], 2)

        ttk.Button(bar, text="Run Full Pipeline", style="Accent.TButton", command=self.run_full_pipeline).grid(
            row=0,
            column=2,
            rowspan=2,
            sticky="e",
            padx=(10, 0),
        )

    def _step_chip(self, parent, text: str, color: str, column: int) -> None:
        chip = tk.Label(
            parent,
            text=text,
            bg=color,
            fg="white",
            padx=12,
            pady=5,
            font=("Segoe UI", 9, "bold"),
        )
        chip.grid(row=0, column=column, padx=(0, 8))

    def _tab_header(self, parent, row: int, number: str, title: str, color: str) -> None:
        header = tk.Frame(parent, bg=color, height=42)
        header.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        header.columnconfigure(1, weight=1)
        tk.Label(
            header,
            text=number,
            bg=color,
            fg="white",
            width=4,
            font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, sticky="nsw", padx=(0, 8), pady=8)
        tk.Label(
            header,
            text=title,
            bg=color,
            fg="white",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=1, sticky="w", pady=8)

    def _on_tab_changed(self, _event=None) -> None:
        current = self.notebook.index(self.notebook.select())
        messages = [
            "Step 1: convert Intan raw recordings into CSV",
            "Step 2: order recordings, label epochs, and count spikes",
            "Step 3: generate selected plot outputs",
        ]
        if 0 <= current < len(messages):
            self.stage_var.set(messages[current])

    def _build_log(self) -> None:
        frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        frame.grid(row=2, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        self.log_text = tk.Text(frame, height=5, wrap="word", state="disabled")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.grid(row=1, column=0, sticky="nsew")
        scroll.grid(row=1, column=1, sticky="ns")

    def _build_raw_tab(self) -> None:
        frame = self.raw_tab
        frame.columnconfigure(1, weight=1)
        self._tab_header(frame, 0, "1", "Raw to CSV", STEP_COLORS["raw"])

        self.raw_format_var = tk.StringVar(value="RHS")
        self.raw_input_dir_var = tk.StringVar(value=str(DEFAULT_DATA_DIR))
        self.raw_output_csv_var = tk.StringVar(
            value=str(DEFAULT_DATA_DIR / "all_channels_amplifier_raw.csv")
        )
        self.raw_channels_var = tk.StringVar(value="")
        self.raw_diff_pairs_var = tk.StringVar(value="")
        self.raw_continuous_time_var = tk.BooleanVar(value=True)
        self.raw_time_col_var = tk.StringVar(value="time_s")

        row = 1
        ttk.Label(frame, text="Intan format").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(frame, textvariable=self.raw_format_var, values=("RHS", "RHD"), width=12, state="readonly").grid(
            row=row, column=1, sticky="w", pady=4
        )

        row += 1
        self._path_row(frame, row, "Raw input folder", self.raw_input_dir_var, self._browse_raw_input_dir)
        row += 1
        self._path_row(frame, row, "Output CSV", self.raw_output_csv_var, self._browse_raw_output_csv)

        row += 1
        ttk.Label(frame, text="Target channels").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.raw_channels_var).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Label(frame, text="Comma-separated; blank means all channels", style="Hint.TLabel").grid(row=row, column=2, sticky="w", padx=8)

        row += 1
        ttk.Label(frame, text="Differential pairs").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.raw_diff_pairs_var).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Label(frame, text="Example: A-008,A-010; A-011,A-012", style="Hint.TLabel").grid(row=row, column=2, sticky="w", padx=8)

        row += 1
        ttk.Label(frame, text="Time column").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.raw_time_col_var, width=18).grid(row=row, column=1, sticky="w", pady=4)

        row += 1
        ttk.Checkbutton(frame, text="Continuous time across raw files", variable=self.raw_continuous_time_var).grid(
            row=row, column=1, sticky="w", pady=4
        )

        row += 1
        ttk.Button(frame, text="Run Raw Conversion", style="Raw.TButton", command=self.run_raw_conversion).grid(
            row=row, column=1, sticky="w", pady=12
        )

    def _build_spike_tab(self) -> None:
        frame = self.spike_tab
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(5, weight=1)
        self._tab_header(frame, 0, "2", "Spike counts", STEP_COLORS["spike"])

        self.spike_csv_dir_var = tk.StringVar(value=str(DEFAULT_DATA_DIR))
        self.spike_glob_var = tk.StringVar(value="*.csv")
        self.spike_window_var = tk.StringVar(value="1 min")
        self.spike_out_dir_var = tk.StringVar(
            value=str(DEFAULT_DATA_DIR / "spike_counts_per_min")
        )
        self.selected_epoch_var = tk.StringVar(value="Baseline")

        row = 1
        self._path_row(frame, row, "CSV folder", self.spike_csv_dir_var, self._browse_spike_csv_dir)
        row += 1
        ttk.Label(frame, text="CSV pattern").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.spike_glob_var, width=18).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Button(frame, text="Refresh Files", style="Secondary.TButton", command=self.refresh_spike_files).grid(row=row, column=2, sticky="w", padx=8)

        row += 1
        ttk.Label(frame, text="Spike-count window").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(
            frame,
            textvariable=self.spike_window_var,
            values=("30 sec", "1 min", "2 min", "3 min", "4 min", "5 min", "10 min"),
            width=12,
            state="readonly",
        ).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Label(frame, text="Use 30 sec here if you want true 30 sec plots later.", style="Hint.TLabel").grid(
            row=row,
            column=2,
            sticky="w",
            padx=8,
        )

        row += 1
        self._path_row(frame, row, "Spike output folder", self.spike_out_dir_var, self._browse_spike_out_dir)

        row += 1
        controls = ttk.Frame(frame)
        controls.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        ttk.Button(controls, text="Move Up", style="Secondary.TButton", command=lambda: self.move_recording(-1)).pack(side="left")
        ttk.Button(controls, text="Move Down", style="Secondary.TButton", command=lambda: self.move_recording(1)).pack(side="left", padx=6)
        ttk.Label(controls, text="Selected label").pack(side="left", padx=(16, 4))
        ttk.Combobox(
            controls,
            textvariable=self.selected_epoch_var,
            values=("Baseline", "Stimulation", "Post 1", "Post 2", "Post 3", "Post"),
            width=20,
        ).pack(side="left")
        ttk.Button(controls, text="Apply Label", style="Secondary.TButton", command=self.apply_epoch_label).pack(side="left", padx=6)

        row += 1
        list_frame = ttk.Frame(frame)
        list_frame.grid(row=row, column=0, columnspan=3, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.recording_listbox = tk.Listbox(list_frame, height=15, exportselection=False)
        self.recording_listbox.bind("<<ListboxSelect>>", self.on_recording_select)
        rec_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.recording_listbox.yview)
        self.recording_listbox.configure(yscrollcommand=rec_scroll.set)
        self.recording_listbox.grid(row=0, column=0, sticky="nsew")
        rec_scroll.grid(row=0, column=1, sticky="ns")

        row += 1
        ttk.Button(frame, text="Run Spike Counting", style="Spike.TButton", command=self.run_spike_counting).grid(
            row=row, column=1, sticky="w", pady=12
        )

    def _build_plot_tab(self) -> None:
        outer = self.plot_tab
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        self.plot_tab_canvas = tk.Canvas(
            outer,
            bg=STEP_COLORS["surface"],
            highlightthickness=0,
            borderwidth=0,
        )
        self.plot_tab_scrollbar = ttk.Scrollbar(
            outer,
            orient="vertical",
            command=self.plot_tab_canvas.yview,
        )
        self.plot_tab_canvas.configure(yscrollcommand=self.plot_tab_scrollbar.set)
        self.plot_tab_canvas.grid(row=0, column=0, sticky="nsew")
        self.plot_tab_scrollbar.grid(row=0, column=1, sticky="ns")

        frame = ttk.Frame(self.plot_tab_canvas)
        self.plot_tab_window = self.plot_tab_canvas.create_window(
            (0, 0),
            window=frame,
            anchor="nw",
        )
        frame.bind("<Configure>", self._on_plot_tab_frame_configure)
        self.plot_tab_canvas.bind("<Configure>", self._on_plot_tab_canvas_configure)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(4, weight=1, minsize=320)
        self._tab_header(frame, 0, "3", "Plots", STEP_COLORS["plot"])

        self.plot_input_var = tk.StringVar(value="")
        self.plot_out_dir_var = tk.StringVar(value="")
        self.plot_id_cols_var = tk.StringVar(value="channel")
        self.plot_group_col_var = tk.StringVar(value="")
        self.plot_time_course_var = tk.BooleanVar(value=True)
        self.continuous_plot_title_var = tk.StringVar(value="Single Electrode - Galvanostatic (16/06/26)")
        self.continuous_boundary_var = tk.StringVar(value="20 min (-200nA)")
        self.continuous_y_top_var = tk.StringVar(value="")
        self.continuous_trace_color_var = tk.StringVar(value="tab:blue")
        self.continuous_marker_var = tk.StringVar(value="o")
        self.continuous_linewidth_var = tk.StringVar(value="2")
        self.continuous_fig_width_var = tk.StringVar(value="16")
        self.continuous_fig_height_var = tk.StringVar(value="5.5")
        self.continuous_dpi_var = tk.StringVar(value="200")
        self.continuous_max_x_ticks_var = tk.StringVar(value="24")
        self.continuous_plot_bin_var = tk.StringVar(value="1 min")

        self.phase_baseline_var = tk.BooleanVar(value=True)
        self.phase_stim_var = tk.BooleanVar(value=True)
        self.phase_post_var = tk.BooleanVar(value=True)
        self.comp_baseline_var = tk.BooleanVar(value=True)
        self.comp_stim_var = tk.BooleanVar(value=True)
        self.comp_post_var = tk.BooleanVar(value=True)
        self.pool_percent_change_var = tk.BooleanVar(value=False)
        self.pooled_percent_title_var = tk.StringVar(value="Firing change from Baseline")
        self.percent_part_titles: dict[str, str] = {}
        self.percent_part_titles_summary_var = tk.StringVar(value="Default non-pooled titles")

        row = 1
        self._path_row(frame, row, "Spike-count file", self.plot_input_var, self._browse_plot_input)
        row += 1
        self._path_row(frame, row, "Plot output folder", self.plot_out_dir_var, self._browse_plot_out_dir)

        row += 1
        selector = ttk.LabelFrame(frame, text="Plot and analysis tools", padding=10)
        selector.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(12, 8))
        selector.columnconfigure(0, weight=1)
        selector.columnconfigure(1, weight=1)
        ttk.Label(
            selector,
            text="Choose a tool below. Its parameters open in the panel underneath.",
            style="Hint.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        self.plot_cards: dict[str, tk.Frame] = {}
        self._plot_tool_card(
            selector,
            key="baseline_post",
            title="Baseline/Post firing-rate plots",
            description="Percent change, spike frequency, time courses, and summary tables.",
            color=STEP_COLORS["plot"],
            row=1,
            column=0,
        )
        self._plot_tool_card(
            selector,
            key="continuous_spike",
            title="Continuous spike-count plots",
            description="Per-channel continuous spike-count curves from spike-count CSVs.",
            color=STEP_COLORS["success"],
            row=1,
            column=1,
        )

        row += 1
        self.plot_detail_frame = ttk.Frame(frame)
        self.plot_detail_frame.grid(row=row, column=0, columnspan=3, sticky="nsew")
        self.plot_detail_frame.columnconfigure(0, weight=1)
        self.plot_detail_frame.rowconfigure(0, weight=1)

        self.plot_detail_panels = {
            "baseline_post": self._build_baseline_post_plot_panel(self.plot_detail_frame),
            "continuous_spike": self._build_continuous_spike_plot_panel(self.plot_detail_frame),
        }
        self.show_plot_panel("baseline_post")
        self._bind_plot_tab_mousewheel_widgets(frame)
        self._refresh_plot_tab_scrollregion()

    def _plot_tool_card(
        self,
        parent,
        key: str,
        title: str,
        description: str,
        color: str,
        row: int,
        column: int,
    ) -> None:
        card = tk.Frame(
            parent,
            bg="white",
            highlightbackground=STEP_COLORS["border"],
            highlightthickness=1,
            padx=12,
            pady=10,
            cursor="hand2",
        )
        card.grid(row=row, column=column, sticky="ew", padx=(0, 8) if column == 0 else (8, 0), pady=2)
        card.columnconfigure(1, weight=1)

        accent = tk.Frame(card, bg=color, width=6)
        accent.grid(row=0, column=0, rowspan=2, sticky="nsw", padx=(0, 10))
        title_label = tk.Label(card, text=title, bg="white", fg=STEP_COLORS["text"], font=("Segoe UI", 10, "bold"))
        title_label.grid(row=0, column=1, sticky="w")
        desc_label = tk.Label(
            card,
            text=description,
            bg="white",
            fg=STEP_COLORS["muted"],
            justify="left",
            wraplength=430,
        )
        desc_label.grid(row=1, column=1, sticky="w", pady=(3, 0))

        for widget in (card, accent, title_label, desc_label):
            widget.bind("<Button-1>", lambda _event, selected=key: self.show_plot_panel(selected))

        self.plot_cards[key] = card

    def show_plot_panel(self, key: str) -> None:
        for panel in getattr(self, "plot_detail_panels", {}).values():
            with contextlib.suppress(tk.TclError):
                panel.grid_remove()

        panel = self.plot_detail_panels[key]
        panel.grid(row=0, column=0, sticky="nsew")
        self._refresh_plot_tab_scrollregion()

        for card_key, card in self.plot_cards.items():
            selected = card_key == key
            card.configure(
                highlightbackground=STEP_COLORS["plot"] if selected else STEP_COLORS["border"],
                highlightthickness=2 if selected else 1,
            )

        names = {
            "baseline_post": "Baseline/Post firing-rate plots selected",
            "continuous_spike": "Continuous spike-count plots selected",
        }
        self.stage_var.set(names.get(key, "Plot tool selected"))

    def _on_plot_tab_frame_configure(self, _event=None) -> None:
        self._refresh_plot_tab_scrollregion()

    def _on_plot_tab_canvas_configure(self, event) -> None:
        self.plot_tab_canvas.itemconfigure(self.plot_tab_window, width=event.width)
        self._refresh_plot_tab_scrollregion()

    def _refresh_plot_tab_scrollregion(self) -> None:
        with contextlib.suppress(tk.TclError):
            self.plot_tab_canvas.configure(scrollregion=self.plot_tab_canvas.bbox("all"))

    def _bind_plot_tab_mousewheel_widgets(self, widget) -> None:
        widget.bind("<MouseWheel>", self._on_plot_tab_mousewheel, add="+")
        widget.bind("<Button-4>", self._on_plot_tab_mousewheel, add="+")
        widget.bind("<Button-5>", self._on_plot_tab_mousewheel, add="+")
        for child in widget.winfo_children():
            self._bind_plot_tab_mousewheel_widgets(child)

    def _on_plot_tab_mousewheel(self, event):
        bbox = self.plot_tab_canvas.bbox("all")
        if not bbox or bbox[3] <= self.plot_tab_canvas.winfo_height():
            return

        if getattr(event, "num", None) == 4:
            units = -3
        elif getattr(event, "num", None) == 5:
            units = 3
        else:
            delta = getattr(event, "delta", 0)
            units = int(-delta / 120) if delta else 0
            if units == 0:
                units = -1 if delta > 0 else 1
        self.plot_tab_canvas.yview_scroll(units, "units")
        return "break"

    def _build_baseline_post_plot_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="Baseline/Post firing-rate plot parameters", padding=10)
        panel.columnconfigure(0, weight=1)

        row = 0
        phase_box = ttk.LabelFrame(panel, text="Recorded phases", padding=10)
        phase_box.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        ttk.Checkbutton(phase_box, text="Baseline", variable=self.phase_baseline_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(phase_box, text="Stimulation", variable=self.phase_stim_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(phase_box, text="Post", variable=self.phase_post_var).pack(side="left", padx=(0, 16))

        row += 1
        comp_box = ttk.LabelFrame(panel, text="Comparisons to output", padding=10)
        comp_box.grid(row=row, column=0, sticky="ew", pady=6)
        ttk.Checkbutton(comp_box, text="Baseline halves", variable=self.comp_baseline_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(comp_box, text="Baseline vs stimulation", variable=self.comp_stim_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(comp_box, text="Baseline vs Post", variable=self.comp_post_var).pack(side="left", padx=(0, 16))

        row += 1
        ttk.Checkbutton(
            panel,
            text="Pool stimulation/Post percent-change into one plot",
            variable=self.pool_percent_change_var,
        ).grid(
            row=row,
            column=0,
            sticky="w",
            pady=6,
        )

        row += 1
        title_box = ttk.LabelFrame(panel, text="Percent-change plot titles", padding=10)
        title_box.grid(row=row, column=0, sticky="ew", pady=6)
        title_box.columnconfigure(1, weight=1)
        ttk.Label(title_box, text="Pooled plot title").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(title_box, textvariable=self.pooled_percent_title_var).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=4,
        )
        ttk.Button(
            title_box,
            text="Set non-pooled titles...",
            style="Secondary.TButton",
            command=self.configure_percent_part_titles,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(
            title_box,
            textvariable=self.percent_part_titles_summary_var,
            style="Hint.TLabel",
        ).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        row += 1
        ttk.Checkbutton(panel, text="Make per-channel time-course plots", variable=self.plot_time_course_var).grid(
            row=row,
            column=0,
            sticky="w",
            pady=6,
        )

        row += 1
        ttk.Button(panel, text="Run Plotting", style="Plot.TButton", command=self.run_plotting).grid(
            row=row,
            column=0,
            sticky="w",
            pady=(6, 0),
        )
        return panel

    def _build_continuous_spike_plot_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="Continuous spike-count plot parameters", padding=10)
        panel.columnconfigure(1, weight=1)
        panel.columnconfigure(3, weight=1)

        row = 0
        ttk.Label(
            panel,
            text="Uses the shared Plot output folder above. Leave it blank to save next to the input CSV.",
            style="Hint.TLabel",
        ).grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 6))
        row += 1
        ttk.Label(panel, text="Plot title").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(panel, textvariable=self.continuous_plot_title_var).grid(row=row, column=1, columnspan=3, sticky="ew", pady=4)
        row += 1
        ttk.Label(panel, text="Spike-count bin").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(
            panel,
            textvariable=self.continuous_plot_bin_var,
            values=("30 sec", "1 min", "2 min", "3 min", "4 min", "5 min", "10 min"),
            width=12,
            state="readonly",
        ).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Label(
            panel,
            text="30 sec requires a CSV generated with 30 sec windows.",
            style="Hint.TLabel",
        ).grid(row=row, column=2, columnspan=2, sticky="w", padx=(14, 0), pady=4)
        row += 1
        ttk.Label(panel, text="Boundary label").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(panel, textvariable=self.continuous_boundary_var).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Label(panel, text="Y-axis max").grid(row=row, column=2, sticky="w", padx=(14, 4), pady=4)
        ttk.Entry(panel, textvariable=self.continuous_y_top_var, width=12).grid(row=row, column=3, sticky="w", pady=4)
        row += 1
        ttk.Label(panel, text="Trace color").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(panel, textvariable=self.continuous_trace_color_var, width=16).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Label(panel, text="Marker").grid(row=row, column=2, sticky="w", padx=(14, 4), pady=4)
        ttk.Combobox(
            panel,
            textvariable=self.continuous_marker_var,
            values=("o", ".", "s", "^", "None"),
            width=10,
        ).grid(row=row, column=3, sticky="w", pady=4)
        row += 1
        ttk.Label(panel, text="Line width").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(panel, textvariable=self.continuous_linewidth_var, width=12).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Label(panel, text="Max x labels").grid(row=row, column=2, sticky="w", padx=(14, 4), pady=4)
        ttk.Entry(panel, textvariable=self.continuous_max_x_ticks_var, width=12).grid(row=row, column=3, sticky="w", pady=4)
        row += 1
        ttk.Label(panel, text="Figure width").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(panel, textvariable=self.continuous_fig_width_var, width=12).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Label(panel, text="Figure height").grid(row=row, column=2, sticky="w", padx=(14, 4), pady=4)
        ttk.Entry(panel, textvariable=self.continuous_fig_height_var, width=12).grid(row=row, column=3, sticky="w", pady=4)
        row += 1
        ttk.Label(panel, text="DPI").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(panel, textvariable=self.continuous_dpi_var, width=12).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Label(panel, text="Blank boundary or y-axis fields are allowed", style="Hint.TLabel").grid(
            row=row,
            column=2,
            columnspan=2,
            sticky="w",
            padx=(14, 0),
            pady=4,
        )
        row += 1
        ttk.Button(panel, text="Run Plotting", style="Plot.TButton", command=self.run_continuous_spike_plot).grid(
            row=row,
            column=1,
            sticky="w",
            pady=(8, 0),
        )
        return panel

    def _path_row(self, parent, row: int, label: str, var: tk.StringVar, command) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="Browse", style="Secondary.TButton", command=command).grid(row=row, column=2, sticky="w", padx=8, pady=4)

    def log(self, text: str) -> None:
        self.log_queue.put(text)

    def _drain_log_queue(self) -> None:
        try:
            while True:
                text = self.log_queue.get_nowait()
                self.log_text.configure(state="normal")
                self.log_text.insert("end", text)
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log_queue)

    def run_background(self, title: str, func, on_success=None) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Pipeline is running", "Wait for the current step to finish before starting another.")
            return

        def worker() -> None:
            writer = QueueWriter(self.log_queue)
            result = None
            failed = False
            self.log_queue.put(f"\n--- {title} ---\n")
            try:
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    result = func()
            except Exception:
                failed = True
                self.log_queue.put(traceback.format_exc())

            def finish() -> None:
                if failed:
                    self.status_var.set(f"{title} failed")
                else:
                    self.status_var.set(f"{title} finished")
                    self.log_queue.put(f"--- {title} finished ---\n")
                    if on_success:
                        on_success(result)

            self.root.after(0, finish)

        self.status_var.set(f"{title} running...")
        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _browse_raw_input_dir(self) -> None:
        self._browse_directory_into(self.raw_input_dir_var)

    def _browse_raw_output_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Choose output CSV",
            initialdir=self._initial_dir(self.raw_output_csv_var),
            initialfile=Path(self.raw_output_csv_var.get()).name if self.raw_output_csv_var.get().strip() else "all_channels_amplifier_raw.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.raw_output_csv_var.set(path)

    def _browse_spike_csv_dir(self) -> None:
        self._browse_directory_into(self.spike_csv_dir_var)
        self.refresh_spike_files()

    def _browse_spike_out_dir(self) -> None:
        self._browse_directory_into(self.spike_out_dir_var)

    def _browse_plot_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose spike-count file",
            initialdir=self._initial_dir(self.plot_input_var),
            filetypes=[("Data files", "*.csv *.xlsx *.xls"), ("All files", "*.*")],
        )
        if path:
            self.plot_input_var.set(path)

    def _browse_plot_out_dir(self) -> None:
        self._browse_directory_into(self.plot_out_dir_var)

    def _browse_directory_into(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory(title="Choose folder", initialdir=self._initial_dir(var))
        if path:
            var.set(path)

    def _initial_dir(self, var: tk.StringVar | None = None) -> str:
        if var is not None:
            raw = var.get().strip()
            if raw:
                candidate = Path(raw)
                if candidate.is_file():
                    candidate = candidate.parent
                if candidate.exists() and candidate.is_dir():
                    return str(candidate)
        return str(BROWSE_START_DIR)

    def refresh_spike_files(self) -> None:
        csv_dir = self.spike_csv_dir_var.get().strip()
        pattern = self.spike_glob_var.get().strip() or "*.csv"
        paths = sorted(glob.glob(os.path.join(csv_dir, pattern)))
        self.recordings = [{"path": path, "label": ""} for path in paths]
        if self.recordings:
            self.recordings[0]["label"] = "Baseline"
        self._refresh_recording_listbox()
        self.log(f"Found {len(self.recordings)} CSV file(s) for spike counting.\n")

    def _refresh_recording_listbox(self) -> None:
        self.recording_listbox.delete(0, "end")
        for i, item in enumerate(self.recordings, start=1):
            label = item["label"] or "unlabeled"
            self.recording_listbox.insert("end", f"{i:>2}. {Path(item['path']).name}    [{label}]")
            self.recording_listbox.itemconfig(
                i - 1,
                background=self._epoch_row_color(label),
                foreground=STEP_COLORS["text"],
                selectbackground="#1d4ed8",
                selectforeground="white",
            )

    def _epoch_row_color(self, label: str) -> str:
        key = label.strip().lower()
        if "baseline" in key:
            return EPOCH_ROW_COLORS["baseline"]
        if "stim" in key or "current" in key:
            return EPOCH_ROW_COLORS["stimulation"]
        if "post" in key or "after" in key or "recovery" in key:
            return EPOCH_ROW_COLORS["post"]
        return EPOCH_ROW_COLORS["unlabeled"]

    def _selected_recording_index(self) -> int | None:
        selection = self.recording_listbox.curselection()
        if not selection:
            return None
        return int(selection[0])

    def on_recording_select(self, _event=None) -> None:
        index = self._selected_recording_index()
        if index is None:
            return
        self.selected_epoch_var.set(self.recordings[index]["label"] or "")

    def apply_epoch_label(self) -> None:
        index = self._selected_recording_index()
        if index is None:
            return
        self.recordings[index]["label"] = self.selected_epoch_var.get().strip()
        self._refresh_recording_listbox()
        self.recording_listbox.selection_set(index)

    def move_recording(self, delta: int) -> None:
        index = self._selected_recording_index()
        if index is None:
            return
        new_index = index + delta
        if new_index < 0 or new_index >= len(self.recordings):
            return
        self.recordings[index], self.recordings[new_index] = self.recordings[new_index], self.recordings[index]
        self._refresh_recording_listbox()
        self.recording_listbox.selection_set(new_index)

    def _format_command(self, cmd: list[str]) -> str:
        return " ".join(f'"{part}"' if " " in part else part for part in cmd)

    def _run_logged_subprocess(self, cmd: list[str], failure_label: str) -> None:
        self.log_queue.put("Running command:\n  " + self._format_command(cmd) + "\n")
        process = subprocess.Popen(
            cmd,
            cwd=str(SCRIPT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            self.log_queue.put(line)
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"{failure_label} failed with exit code {return_code}.")

    def _perform_raw_conversion(self) -> str:
        fmt = self.raw_format_var.get().strip().upper()
        script = RHS_SCRIPT if fmt == "RHS" else RHD_SCRIPT
        module = load_script_module(script, f"{fmt.lower()}_to_csv_gui")
        out_csv = self.raw_output_csv_var.get().strip()
        module.combine_all_amplifier_to_one_csv(
            input_dir=self.raw_input_dir_var.get().strip(),
            out_csv=out_csv,
            target_channels=parse_csv_list(self.raw_channels_var.get()),
            diff_pairs=parse_diff_pairs(self.raw_diff_pairs_var.get()),
            continuous_time=bool(self.raw_continuous_time_var.get()),
            time_col_name=self.raw_time_col_var.get().strip() or "time_s",
        )
        return out_csv

    def _perform_spike_counting(
        self,
        paths: list[str] | None = None,
        labels: dict[str, str] | None = None,
        out_dir: str | None = None,
    ) -> str:
        import matplotlib

        matplotlib.use("Agg", force=True)
        module = load_script_module(SPIKE_SCRIPT, "spike_count_multiple_csvs_gui")
        if paths is None:
            if not self.recordings:
                self.refresh_spike_files()
            paths = [item["path"] for item in self.recordings]
            labels = {item["path"]: item["label"] for item in self.recordings if item["label"].strip()}
        if not paths:
            raise FileNotFoundError("No CSV files were found for spike counting.")
        return module.process_csvs(
            paths,
            out_dir=out_dir if out_dir is not None else self.spike_out_dir_var.get().strip(),
            epoch_labels_by_path=labels or {},
            window_sec=plot_bin_seconds(self.spike_window_var.get()),
        )

    def _detect_percent_part_plot_specs(
        self,
        input_path: str,
        phases: list[str],
        comparisons: list[str],
    ) -> list[dict[str, str]]:
        module = load_script_module(PLOT_SCRIPT, "baseline_post_title_detection")
        recorded_phases = module.parse_recorded_phases(",".join(phases))
        comparison_names = module.parse_comparisons(",".join(comparisons))
        comparison_names = module.filter_comparisons_for_phases(comparison_names, recorded_phases)

        raw = module.load_table(Path(input_path), sheet=module.EXCEL_SHEET, csv_sep=module.CSV_SEPARATOR)
        df = module.validate_and_prepare_data(raw)
        df = df[df["phase"].isin(recorded_phases)].copy()
        if df.empty:
            return []

        id_columns = tuple(col.strip() for col in self.plot_id_cols_var.get().strip().split(",") if col.strip()) or ("channel",)
        group_column = module.choose_group_column(df, self.plot_group_col_var.get().strip())
        allow_post_without_stimulation = (
            module.STIMULATION_LABEL not in recorded_phases
            or not (df["phase"] == module.STIMULATION_LABEL).any()
        )
        df = module.assign_experiment_parts(
            df,
            id_columns,
            group_column,
            allow_post_without_stimulation=allow_post_without_stimulation,
        )
        df = module.add_baseline_half_labels(df, id_columns, group_column)
        baseline_change = module.build_baseline_half_table(df, id_columns, group_column)
        part_table = module.build_experiment_part_table(df, baseline_change, id_columns, group_column)
        if part_table.empty:
            return []

        comparison_set = set(comparison_names)
        specs: list[dict[str, str]] = []
        for (part_index, stimulation_label, part_label), part in part_table.groupby(
            ["part_index", "stimulation_label", "part_label"],
            dropna=False,
            sort=True,
        ):
            has_plot_value = False
            if module.COMPARISON_BASELINE in comparison_set and part["baseline_percent_change"].notna().any():
                has_plot_value = True
            if module.COMPARISON_STIMULATION in comparison_set and part["stimulation_percent_change"].notna().any():
                has_plot_value = True
            if module.COMPARISON_POST in comparison_set and part["post_percent_change"].notna().any():
                has_plot_value = True
            if not has_plot_value:
                continue
            index = int(part_index)
            default_title = str(stimulation_label)
            specs.append(
                {
                    "key": f"part:{index}",
                    "label": f"Part {index}: {default_title}",
                    "default_title": default_title,
                    "part_label": str(part_label),
                }
            )
        return specs

    def _title_dialog(self, specs: list[dict[str, str]]) -> dict[str, str] | None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Non-pooled percent-change plot titles")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.columnconfigure(0, weight=1)

        ttk.Label(
            dialog,
            text="Set the title for each non-pooled percent-change plot.",
            padding=(12, 12, 12, 4),
        ).grid(row=0, column=0, sticky="w")

        body = ttk.Frame(dialog, padding=(12, 4, 12, 8))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)

        entries: dict[str, tk.StringVar] = {}
        for row, spec in enumerate(specs):
            key = spec["key"]
            ttk.Label(body, text=spec["label"]).grid(row=row, column=0, sticky="w", pady=4)
            var = tk.StringVar(value=self.percent_part_titles.get(key, spec["default_title"]))
            entries[key] = var
            ttk.Entry(body, textvariable=var, width=48).grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=4)

        result: dict[str, str] | None = None

        def accept() -> None:
            nonlocal result
            result = {key: var.get().strip() for key, var in entries.items() if var.get().strip()}
            dialog.destroy()

        def cancel() -> None:
            dialog.destroy()

        buttons = ttk.Frame(dialog, padding=(12, 0, 12, 12))
        buttons.grid(row=2, column=0, sticky="e")
        ttk.Button(buttons, text="Cancel", style="Secondary.TButton", command=cancel).pack(side="right")
        ttk.Button(buttons, text="OK", style="Plot.TButton", command=accept).pack(side="right", padx=(0, 8))
        dialog.bind("<Return>", lambda _event: accept())
        dialog.bind("<Escape>", lambda _event: cancel())

        dialog.update_idletasks()
        dialog.geometry(f"+{self.root.winfo_rootx() + 80}+{self.root.winfo_rooty() + 80}")
        self.root.wait_window(dialog)
        return result

    def configure_percent_part_titles(
        self,
        input_path: str | None = None,
        phases: list[str] | None = None,
        comparisons: list[str] | None = None,
    ) -> bool:
        input_path = input_path or self.plot_input_var.get().strip()
        if not input_path:
            messagebox.showerror("Missing input", "Choose a spike-count CSV/XLSX file first.")
            return False
        phases = phases or self.selected_recorded_phases()
        comparisons = comparisons or self.selected_comparisons()
        try:
            specs = self._detect_percent_part_plot_specs(input_path, phases, comparisons)
        except Exception as exc:
            messagebox.showerror("Plot title detection", str(exc))
            return False

        if not specs:
            self.percent_part_titles = {}
            self.percent_part_titles_summary_var.set("No non-pooled percent-change plots detected")
            messagebox.showinfo("Plot titles", "No non-pooled percent-change plots were detected for these settings.")
            return True

        titles = self._title_dialog(specs)
        if titles is None:
            return False
        self.percent_part_titles = titles
        self.percent_part_titles_summary_var.set(f"{len(titles)} non-pooled title(s) set")
        return True

    def _baseline_post_plot_command(
        self,
        input_path: str,
        phases: list[str] | None = None,
        comparisons: list[str] | None = None,
    ) -> tuple[list[str], str]:
        phases = phases or self.selected_recorded_phases()
        comparisons = comparisons or self.selected_comparisons()
        if "baseline" not in phases:
            raise ValueError("Baseline must be selected as a recorded phase.")
        if not comparisons:
            raise ValueError("Select at least one comparison to output.")

        cmd = [
            sys.executable,
            str(PLOT_SCRIPT),
            input_path,
            "--id-cols",
            self.plot_id_cols_var.get().strip() or "channel",
            "--recorded-phases",
            ",".join(phases),
            "--comparisons",
            ",".join(comparisons),
        ]
        out_dir = self.plot_out_dir_var.get().strip()
        if out_dir:
            cmd.extend(["--out-dir", out_dir])
        group_col = self.plot_group_col_var.get().strip()
        if group_col:
            cmd.extend(["--group-col", group_col])
        if self.pool_percent_change_var.get():
            cmd.append("--pool-percent-change-comparisons")
            pooled_title = self.pooled_percent_title_var.get().strip()
            if pooled_title:
                cmd.extend(["--pooled-percent-title", pooled_title])
        elif self.percent_part_titles:
            cmd.extend(["--percent-part-titles", json.dumps(self.percent_part_titles)])
        if not self.plot_time_course_var.get():
            cmd.append("--no-time-course")
        return cmd, out_dir or str(Path(input_path).with_name(f"{Path(input_path).stem}_baseline_post_plots"))

    def _perform_baseline_post_plot(
        self,
        input_path: str,
        phases: list[str] | None = None,
        comparisons: list[str] | None = None,
    ) -> str:
        cmd, out_dir = self._baseline_post_plot_command(input_path, phases=phases, comparisons=comparisons)
        self._run_logged_subprocess(cmd, "Baseline/Post plotting")
        return out_dir

    def _continuous_plot_command(self, input_path: str) -> tuple[list[str], str]:
        if not input_path.lower().endswith(".csv"):
            raise ValueError("The continuous spike-count plotter expects a CSV file.")
        numeric_fields = {
            "Y-axis max": (self.continuous_y_top_var.get().strip(), float, False),
            "Line width": (self.continuous_linewidth_var.get().strip(), float, True),
            "Figure width": (self.continuous_fig_width_var.get().strip(), float, True),
            "Figure height": (self.continuous_fig_height_var.get().strip(), float, True),
            "DPI": (self.continuous_dpi_var.get().strip(), int, True),
            "Max x labels": (self.continuous_max_x_ticks_var.get().strip(), int, True),
        }
        parsed_values: dict[str, float | int | None] = {}
        for label, (raw, caster, required) in numeric_fields.items():
            if not raw and not required:
                parsed_values[label] = None
                continue
            if not raw:
                raise ValueError(f"{label} must be filled in.")
            try:
                value = caster(raw)
            except ValueError as exc:
                raise ValueError(f"{label} must be a number.") from exc
            if value <= 0:
                raise ValueError(f"{label} must be greater than zero.")
            parsed_values[label] = value

        cmd = [
            sys.executable,
            str(CSV_SPIKE_PLOT_SCRIPT),
            input_path,
            "--title",
            self.continuous_plot_title_var.get().strip() or "Single Electrode - Galvanostatic (16/06/26)",
            "--boundary-description",
            self.continuous_boundary_var.get().strip(),
            "--no-prompt-titles",
            "--trace-color",
            self.continuous_trace_color_var.get().strip() or "tab:blue",
            "--trace-marker",
            self.continuous_marker_var.get().strip() or "o",
            "--trace-linewidth",
            str(parsed_values["Line width"]),
            "--fig-width",
            str(parsed_values["Figure width"]),
            "--fig-height",
            str(parsed_values["Figure height"]),
            "--dpi",
            str(parsed_values["DPI"]),
            "--max-x-ticks",
            str(parsed_values["Max x labels"]),
            "--plot-bin-sec",
            str(plot_bin_seconds(self.continuous_plot_bin_var.get())),
        ]
        out_dir = self.plot_out_dir_var.get().strip()
        if out_dir:
            cmd.extend(["--out-dir", out_dir])
        if parsed_values["Y-axis max"] is not None:
            cmd.extend(["--y-lim-top", str(parsed_values["Y-axis max"])])
        return cmd, out_dir or str(Path(input_path).with_name("spike_counts_summary"))

    def _perform_continuous_spike_plot(self, input_path: str) -> str:
        cmd, out_dir = self._continuous_plot_command(input_path)
        self._run_logged_subprocess(cmd, "Continuous spike plotting")
        return out_dir

    def run_raw_conversion(self) -> None:
        def task():
            return self._perform_raw_conversion()

        def success(path):
            out_path = Path(path)
            self.spike_csv_dir_var.set(str(out_path.parent))
            self.refresh_spike_files()

        self.run_background("Raw conversion", task, on_success=success)

    def run_spike_counting(self) -> None:
        if not self.recordings:
            self.refresh_spike_files()
        if not self.recordings:
            messagebox.showerror("No CSV files", "No CSV files were found for spike counting.")
            return

        def task():
            return self._perform_spike_counting()

        def success(path):
            self.plot_input_var.set(str(path))
            if not self.plot_out_dir_var.get().strip():
                spike_path = Path(path)
                self.plot_out_dir_var.set(str(spike_path.with_name(f"{spike_path.stem}_baseline_post_plots")))

        self.run_background("Spike counting", task, on_success=success)

    def selected_recorded_phases(self) -> list[str]:
        phases = []
        if self.phase_baseline_var.get():
            phases.append("baseline")
        if self.phase_stim_var.get():
            phases.append("stimulation")
        if self.phase_post_var.get():
            phases.append("post")
        return phases

    def selected_comparisons(self) -> list[str]:
        comparisons = []
        if self.comp_baseline_var.get():
            comparisons.append("baseline")
        if self.comp_stim_var.get():
            comparisons.append("stimulation")
        if self.comp_post_var.get():
            comparisons.append("post")
        return comparisons

    def run_plotting(self) -> None:
        input_path = self.plot_input_var.get().strip()
        if not input_path:
            messagebox.showerror("Missing input", "Choose a spike-count CSV/XLSX file first.")
            return
        phases = self.selected_recorded_phases()
        comparisons = self.selected_comparisons()
        try:
            self._baseline_post_plot_command(input_path, phases=phases, comparisons=comparisons)
        except ValueError as exc:
            messagebox.showerror("Plot settings", str(exc))
            return
        if not self.pool_percent_change_var.get():
            if not self.configure_percent_part_titles(input_path, phases, comparisons):
                return

        def task():
            return self._perform_baseline_post_plot(input_path, phases=phases, comparisons=comparisons)

        self.run_background("Baseline/Post plotting", task)

    def run_continuous_spike_plot(self) -> None:
        input_path = self.plot_input_var.get().strip()
        if not input_path:
            messagebox.showerror("Missing input", "Choose a spike-count CSV file first.")
            return
        try:
            self._continuous_plot_command(input_path)
        except ValueError as exc:
            messagebox.showerror("Continuous plot settings", str(exc))
            return

        def task():
            return self._perform_continuous_spike_plot(input_path)

        self.run_background("Continuous spike plotting", task)

    def _default_full_pipeline_epoch_label(self) -> str:
        for item in self.recordings:
            if item["label"].strip():
                return item["label"].strip()
        return self.selected_epoch_var.get().strip() or "Baseline"

    def run_full_pipeline(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Pipeline is running", "Wait for the current step to finish before starting another.")
            return

        def task():
            self.log_queue.put("Full pipeline sequence: raw conversion -> spike counting -> Baseline/Post plots -> continuous spike plots\n")

            raw_csv = self._perform_raw_conversion()
            self.log_queue.put(f"\n[full pipeline] Raw CSV ready:\n  {raw_csv}\n")

            spike_label = self._default_full_pipeline_epoch_label()
            spike_csv = self._perform_spike_counting(
                paths=[raw_csv],
                labels={raw_csv: spike_label},
                out_dir=self.spike_out_dir_var.get().strip(),
            )
            self.log_queue.put(f"\n[full pipeline] Spike-count CSV ready:\n  {spike_csv}\n")

            plot_phases = self.selected_recorded_phases()
            plot_comparisons = self.selected_comparisons()
            if spike_label.lower().startswith("baseline") and (
                "stimulation" in plot_phases or "post" in plot_phases
            ):
                plot_phases = ["baseline"]
                plot_comparisons = ["baseline"]
                self.log_queue.put(
                    "[full pipeline] Converted raw output is one labeled CSV, so Baseline/Post plots are running as Baseline-only.\n"
                )

            baseline_plot_dir = self._perform_baseline_post_plot(
                spike_csv,
                phases=plot_phases,
                comparisons=plot_comparisons,
            )
            continuous_plot_dir = self._perform_continuous_spike_plot(spike_csv)

            return {
                "raw_csv": raw_csv,
                "spike_csv": spike_csv,
                "baseline_plot_dir": baseline_plot_dir,
                "continuous_plot_dir": continuous_plot_dir,
            }

        def success(result):
            if not result:
                return
            raw_csv = Path(result["raw_csv"])
            spike_csv = Path(result["spike_csv"])
            self.raw_output_csv_var.set(str(raw_csv))
            self.spike_csv_dir_var.set(str(raw_csv.parent))
            self.plot_input_var.set(str(spike_csv))
            self.plot_out_dir_var.set(str(result["baseline_plot_dir"]))

        self.run_background("Full pipeline", task, on_success=success)


def main() -> None:
    root = tk.Tk()
    app = LocustPipelineApp(root)
    app.refresh_spike_files()
    root.mainloop()


if __name__ == "__main__":
    main()
