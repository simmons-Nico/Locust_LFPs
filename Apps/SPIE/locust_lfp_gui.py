#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tkinter GUI for Signal Processing for Insect Electrophysiology."""

from __future__ import annotations

import contextlib
import glob
import hashlib
import importlib.util
import json
import os
import queue
import re
import subprocess
import sys
import threading
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk
from typing import Mapping

APP_SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (
    APP_SOURCE_DIR.parents[1]
    if APP_SOURCE_DIR.name == "SPIE" and APP_SOURCE_DIR.parent.name == "Apps"
    else APP_SOURCE_DIR
)
FROZEN_ROOT = Path(getattr(sys, "_MEIPASS", APP_SOURCE_DIR))
RUNTIME_ROOT = FROZEN_ROOT if getattr(sys, "frozen", False) else PROJECT_ROOT
APP_ASSET_DIR = FROZEN_ROOT if getattr(sys, "frozen", False) else APP_SOURCE_DIR

for module_dir in (
    APP_SOURCE_DIR,
    PROJECT_ROOT / "Raw Conversion",
    PROJECT_ROOT / "LFP Processing",
    PROJECT_ROOT / "Spike Processing",
    PROJECT_ROOT / "Plotting and Utilities",
    FROZEN_ROOT / "Raw Conversion",
    FROZEN_ROOT / "LFP Processing",
    FROZEN_ROOT / "Spike Processing",
    FROZEN_ROOT / "Plotting and Utilities",
):
    if module_dir.exists():
        sys.path.insert(0, str(module_dir))

from PIL import Image, ImageTk

from combine_raw_chunks import combine_raw_chunks, default_output_path
from lfp_feature_extraction import LFPConfig, parse_band_spec, parse_channel_list, process_lfp_recordings
from lfp_plotting import plot_lfp_features
from spike_lfp_coupling import SpikeLFPCouplingConfig, calculate_spike_lfp_coupling


APP_NAME = "Signal Processing for Insect Electrophysiology (SPIE)"
APP_SHORT_NAME = "SPIE"
SCRIPT_DIR = RUNTIME_ROOT
APP_ICON = APP_ASSET_DIR / "spie_icon.ico"
PLOT_SCRIPT = RUNTIME_ROOT / "Plotting and Utilities" / "plot_baseline_post_firing_rates.py"
CSV_SPIKE_PLOT_SCRIPT = RUNTIME_ROOT / "Spike Processing" / "CSV_to_spike_counts.py"
SPIKE_SCRIPT = RUNTIME_ROOT / "Spike Processing" / "Spike Count Multiple CSVs ordered.py"
RHS_SCRIPT = RUNTIME_ROOT / "Raw Conversion" / "RHS_to_CSV(simple).py"
RHD_SCRIPT = RUNTIME_ROOT / "Raw Conversion" / "RHD_to_CSV (simple).py"
BROWSE_START_DIR = Path(r"C:\Users\simmons\Desktop\Exploring PSDs")
DEFAULT_DATA_DIR = BROWSE_START_DIR

STEP_COLORS = {
    "raw": "#2dd4bf",
    "raw_active": "#5eead4",
    "spike": "#facc15",
    "spike_active": "#fde047",
    "lfp": "#a78bfa",
    "lfp_active": "#c4b5fd",
    "plot": "#60a5fa",
    "plot_active": "#93c5fd",
    "success": "#22c55e",
    "danger": "#fb7185",
    "surface": "#07111f",
    "panel": "#0f172a",
    "panel_alt": "#111c2e",
    "input": "#020617",
    "border": "#334155",
    "border_strong": "#475569",
    "text": "#f8fafc",
    "muted": "#94a3b8",
    "terminal": "#020617",
    "terminal_text": "#d9f99d",
}

EPOCH_ROW_COLORS = {
    "baseline": "#172554",
    "stimulation": "#422006",
    "post": "#052e16",
    "unlabeled": "#1e293b",
}

PERFUSION_CONCENTRATIONS = ("", "10mM", "1mM", "100uM")

EPOCH_TITLE_FIELDS = (
    ("baseline", "Baseline", "Baseline"),
    ("during", "During", "During"),
    ("stimulation", "Stimulation", "Stimulation"),
    ("perfusion", "Perfusion", "Perfusion"),
    ("post", "Post", "Post"),
    ("post 1", "Post 1", "Post 1"),
    ("post 2", "Post 2", "Post 2"),
    ("post 3", "Post 3", "Post 3"),
)
EPOCH_TITLE_DEFAULTS = {
    key: default
    for key, _label, default in EPOCH_TITLE_FIELDS
}

COMBINE_COLUMN_ALIASES = {
    "recording_index": ("recording_index", "recording", "recording_number", "recording_id"),
    "recording_name": ("recording_name", "recording", "recording_id", "file_name", "filename"),
    "epoch_label": ("epoch_label", "epoch", "condition", "condition_label", "phase"),
    "channel": ("channel", "chan", "electrode", "electrode_id"),
    "window_index": ("window_index", "window", "bin", "bin_index"),
    "window_start_s": ("window_start_s", "start_s", "window_start", "start_time_s"),
    "window_end_s": ("window_end_s", "end_s", "window_end", "end_time_s"),
    "window_duration_s": ("window_duration_s", "duration_s", "duration", "window_length_s"),
    "window_label": ("window_label", "window_name", "bin_label"),
    "spike_count": ("spike_count", "spikes", "spike_counts", "count", "n_spikes", "SpikeCount"),
}


if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

if getattr(sys, "frozen", False):
    import matplotlib.backends.backend_svg  # noqa: F401


class QueueWriter:
    def __init__(self, log_queue: queue.Queue[str]):
        self.log_queue = log_queue

    def write(self, text: str) -> None:
        if text:
            self.log_queue.put(text)

    def flush(self) -> None:
        pass


@dataclass(frozen=True)
class SpikePipelineRequest:
    paths: list[str]
    labels_by_path: dict[str, str]
    out_dir: str
    window_sec: float
    polarity: str
    spike_events_path: str | None
    recording_metadata: list[dict]


def _file_fingerprint(path: str | Path) -> dict:
    p = Path(path)
    try:
        stat = p.stat()
        return {
            "path": str(p.resolve()),
            "mtime_ns": int(stat.st_mtime_ns),
            "size_bytes": int(stat.st_size),
        }
    except OSError:
        return {"path": str(p), "missing": True}


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _spike_script_settings(module) -> dict:
    keys = (
        "HP_SPIKE_BAND",
        "SPIKE_Z_THR",
        "POLARITY",
        "REFRACTORY_MS",
        "AMP_MIN_UV",
        "AMP_MAX_UV",
        "W_MIN_MS",
        "W_MAX_MS",
        "SPIKE_CLASSIFICATION_WINDOW_MS",
        "SPIKE_AMPLITUDE_BIN_S",
        "SPIKE_AMPLITUDE_BASELINE_MINUTES",
        "SPIKE_AMPLITUDE_BASELINE_MIN_VALID_BINS",
        "SPIKE_AMPLITUDE_BASELINE_MIN_SPIKES",
        "SPIKE_AMPLITUDE_BASELINE_MAX_ABS_SLOPE_PERCENT_PER_MIN",
        "SPIKE_AMPLITUDE_BASELINE_MAX_CV_PERCENT",
        "DEFAULT_UNRECORDED_TREATMENT_LABEL",
        "TIME_COL",
        "MARKER_COLUMN_TOKENS",
    )
    return {key: getattr(module, key, None) for key in keys}


def run_shared_spike_pipeline(request: SpikePipelineRequest, script_path: Path = SPIKE_SCRIPT) -> dict:
    """Run the authoritative tab-2 spike counter and write optional provenance."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    module = load_script_module(script_path, f"spike_count_multiple_csvs_shared_{threading.get_ident()}")
    polarity = module.normalize_polarity(request.polarity)
    out_dir = request.out_dir or str(Path(request.paths[0]).parent / "spike_counts_per_min")
    spike_outputs = module.process_csvs(
        request.paths,
        out_dir=out_dir,
        epoch_labels_by_path=request.labels_by_path,
        window_sec=request.window_sec,
        polarity=polarity,
        spike_events_path=request.spike_events_path,
        recording_metadata_by_path=request.recording_metadata,
        return_outputs=True,
    )
    if isinstance(spike_outputs, dict):
        spike_count_csv = spike_outputs.get("spike_count_csv", "")
    else:
        spike_count_csv = str(spike_outputs)
        spike_outputs = {}
    spike_events_csv = request.spike_events_path or str(Path(out_dir) / "spike_events.csv")
    provenance = {
        "pipeline": "Spike Count Multiple CSVs ordered.py::process_csvs",
        "script_path": str(Path(script_path).resolve()),
        "script_sha256": _file_sha256(script_path),
        "script_settings": _spike_script_settings(module),
        "ordered_recordings": [_file_fingerprint(path) for path in request.paths],
        "ordered_recording_list": list(request.paths),
        "labels_by_path": dict(request.labels_by_path),
        "recording_metadata": list(request.recording_metadata),
        "window_sec": float(request.window_sec),
        "polarity": polarity,
        "spike_count_csv": str(spike_count_csv),
        "spike_events_csv": str(spike_events_csv),
        "spike_diagnostics_dir": str(Path(out_dir) / "spike_diagnostics"),
        "spike_waveform_summary_csv": str(Path(out_dir) / "spike_waveform_summary.csv"),
        "spike_diagnostic_outputs_csv": str(Path(out_dir) / "spike_diagnostic_outputs.csv"),
        "spike_amplitude_timecourse_csv": str(spike_outputs.get("spike_amplitude_timecourse_csv", "")),
        "spike_amplitude_waveforms_csv": str(spike_outputs.get("spike_amplitude_waveforms_csv", "")),
        "spike_amplitude_plot_paths": list(spike_outputs.get("spike_amplitude_plot_paths", [])),
        "spike_amplitude_qc_warnings": list(spike_outputs.get("spike_amplitude_qc_warnings", [])),
    }
    provenance_json = str(Path(spike_events_csv).with_name("spike_processing_provenance.json"))
    Path(provenance_json).write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    print(f"[events] wrote spike-processing provenance:\n  {provenance_json}")
    return {
        "spike_count_csv": str(spike_count_csv),
        "spike_events_csv": str(spike_events_csv),
        "spike_diagnostics_dir": str(Path(out_dir) / "spike_diagnostics"),
        "spike_waveform_summary_csv": str(Path(out_dir) / "spike_waveform_summary.csv"),
        "spike_diagnostic_outputs_csv": str(Path(out_dir) / "spike_diagnostic_outputs.csv"),
        "spike_amplitude_timecourse_csv": str(spike_outputs.get("spike_amplitude_timecourse_csv", "")),
        "spike_amplitude_waveforms_csv": str(spike_outputs.get("spike_amplitude_waveforms_csv", "")),
        "spike_amplitude_plot_paths": list(spike_outputs.get("spike_amplitude_plot_paths", [])),
        "spike_amplitude_qc_warnings": list(spike_outputs.get("spike_amplitude_qc_warnings", [])),
        "spike_processing_provenance_json": provenance_json or "",
        "spike_processing_provenance": provenance,
    }


class Tooltip:
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")

    def show(self, _event=None) -> None:
        if self.tip is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tip,
            text=self.text,
            justify="left",
            background=STEP_COLORS["panel_alt"],
            foreground=STEP_COLORS["text"],
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=5,
            wraplength=320,
        )
        label.pack()

    def hide(self, _event=None) -> None:
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


def load_script_module(path: Path, module_name: str):
    script_dir = str(path.resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
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


def normalise_table_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def normalise_epoch_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def epoch_phase_order(value: object) -> int:
    text = normalise_epoch_text(value)
    if not text:
        return 3
    if any(term in text for term in ("baseline", "basal", "before", "pre", "control")):
        return 0
    if any(term in text for term in ("post", "after", "recovery", "washout")):
        return 2
    if any(term in text for term in ("during", "stim", "stimulation", "perfusion")):
        return 1
    return 1


def standardise_combine_columns(df):
    norm_to_original = {normalise_table_name(col): col for col in df.columns}
    rename_map = {}
    for canonical, aliases in COMBINE_COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            original = norm_to_original.get(normalise_table_name(alias))
            if original is not None:
                rename_map[original] = canonical
                break
    return df.rename(columns=rename_map)


def contains_perfusion_label(value: object) -> bool:
    return "perfusion" in normalise_epoch_text(value)


class LocustPipelineApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_NAME)
        if APP_ICON.exists():
            with contextlib.suppress(tk.TclError):
                self.root.iconbitmap(str(APP_ICON))
        self.root.geometry("1180x820")
        self.root.minsize(960, 560)
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.worker_cancel_event: threading.Event | None = None
        self.logo_image: ImageTk.PhotoImage | None = None
        self.processing_window: tk.Toplevel | None = None
        self.processing_text: tk.Text | None = None
        self._tooltips: list[Tooltip] = []
        self._collapsible_sections: list[dict[str, object]] = []

        self.recordings: list[dict[str, str]] = []

        self.status_var = tk.StringVar(value="Ready")
        self.stage_var = tk.StringVar(value="Configure a step or run the full pipeline")
        self._configure_styles()
        self._build_ui()
        self.root.after(100, self._drain_log_queue)

    def _add_tooltip(self, widget, text: str):
        self._tooltips.append(Tooltip(widget, text))
        return widget

    def _configure_styles(self) -> None:
        self.style = ttk.Style(self.root)
        with contextlib.suppress(tk.TclError):
            self.style.theme_use("clam")
        base_font = ("Segoe UI", 10)
        field_font = ("Segoe UI", 9)
        subsection_font = ("Segoe UI", 10, "bold")
        section_font = ("Segoe UI", 11, "bold")
        self.root.configure(bg=STEP_COLORS["surface"])
        self.root.option_add("*Font", base_font)
        self.root.option_add("*selectBackground", STEP_COLORS["plot"])
        self.root.option_add("*selectForeground", STEP_COLORS["input"])
        self.root.option_add("*TCombobox*Listbox.background", STEP_COLORS["input"])
        self.root.option_add("*TCombobox*Listbox.foreground", STEP_COLORS["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", STEP_COLORS["plot"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", STEP_COLORS["input"])
        self.style.configure("TFrame", background=STEP_COLORS["surface"])
        self.style.configure("Panel.TFrame", background=STEP_COLORS["panel"])
        self.style.configure("Card.TFrame", background=STEP_COLORS["panel_alt"])
        self.style.configure(
            "TLabelframe",
            background=STEP_COLORS["panel"],
            bordercolor=STEP_COLORS["border"],
            relief="solid",
            borderwidth=1,
        )
        self.style.configure(
            "TLabelframe.Label",
            background=STEP_COLORS["surface"],
            foreground=STEP_COLORS["text"],
            font=section_font,
        )
        self.style.configure(
            "Subsection.TLabelframe",
            background=STEP_COLORS["panel"],
            bordercolor=STEP_COLORS["border"],
            relief="solid",
            borderwidth=1,
        )
        self.style.configure(
            "Subsection.TLabelframe.Label",
            background=STEP_COLORS["surface"],
            foreground=STEP_COLORS["muted"],
            font=subsection_font,
        )
        self.style.configure("TLabel", background=STEP_COLORS["surface"], foreground=STEP_COLORS["text"], font=base_font)
        self.style.configure("Field.TLabel", background=STEP_COLORS["surface"], foreground=STEP_COLORS["text"], font=field_font)
        self.style.configure("Hint.TLabel", background=STEP_COLORS["surface"], foreground=STEP_COLORS["muted"], font=("Segoe UI", 9))
        self.style.configure("Status.TLabel", background=STEP_COLORS["surface"], foreground=STEP_COLORS["muted"], font=("Segoe UI", 9))
        self.style.configure("Title.TLabel", background=STEP_COLORS["surface"], foreground=STEP_COLORS["text"], font=("Segoe UI", 19, "bold"))
        self.style.configure("Stage.TLabel", background=STEP_COLORS["surface"], foreground=STEP_COLORS["muted"], font=("Segoe UI", 10))
        self.style.configure("TEntry", fieldbackground=STEP_COLORS["input"], foreground=STEP_COLORS["text"], insertcolor=STEP_COLORS["text"], padding=(6, 4))
        self.style.map("TEntry", fieldbackground=[("disabled", STEP_COLORS["panel"])], foreground=[("disabled", STEP_COLORS["muted"])])
        self.style.configure("TCombobox", fieldbackground=STEP_COLORS["input"], background=STEP_COLORS["panel_alt"], foreground=STEP_COLORS["text"], arrowcolor=STEP_COLORS["muted"], padding=(6, 4))
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", STEP_COLORS["input"])],
            selectbackground=[("readonly", STEP_COLORS["input"])],
            selectforeground=[("readonly", STEP_COLORS["text"])],
            foreground=[("readonly", STEP_COLORS["text"])],
        )
        self.style.configure("TCheckbutton", background=STEP_COLORS["surface"], foreground=STEP_COLORS["text"], font=base_font)
        self.style.configure("TRadiobutton", background=STEP_COLORS["surface"], foreground=STEP_COLORS["text"], font=base_font)
        self.style.map("TCheckbutton", background=[("active", STEP_COLORS["surface"])], foreground=[("active", STEP_COLORS["text"])])
        self.style.map("TRadiobutton", background=[("active", STEP_COLORS["surface"])], foreground=[("active", STEP_COLORS["text"])])
        self.style.configure("Accent.TButton", background=STEP_COLORS["plot"], foreground=STEP_COLORS["input"], padding=(14, 8), font=("Segoe UI", 10, "bold"))
        self.style.map("Accent.TButton", background=[("active", STEP_COLORS["plot_active"])])
        self.style.configure("Raw.TButton", background=STEP_COLORS["raw"], foreground=STEP_COLORS["input"], padding=(11, 7), font=("Segoe UI", 10, "bold"))
        self.style.map("Raw.TButton", background=[("active", STEP_COLORS["raw_active"])])
        self.style.configure("Spike.TButton", background=STEP_COLORS["spike"], foreground=STEP_COLORS["input"], padding=(11, 7), font=("Segoe UI", 10, "bold"))
        self.style.map("Spike.TButton", background=[("active", STEP_COLORS["spike_active"])])
        self.style.configure("Lfp.TButton", background=STEP_COLORS["lfp"], foreground=STEP_COLORS["input"], padding=(11, 7), font=("Segoe UI", 10, "bold"))
        self.style.map("Lfp.TButton", background=[("active", STEP_COLORS["lfp_active"])])
        self.style.configure("Plot.TButton", background=STEP_COLORS["plot"], foreground=STEP_COLORS["input"], padding=(11, 7), font=("Segoe UI", 10, "bold"))
        self.style.map("Plot.TButton", background=[("active", STEP_COLORS["plot_active"])])
        self.style.configure("Secondary.TButton", background=STEP_COLORS["panel_alt"], foreground=STEP_COLORS["text"], padding=(10, 6), bordercolor=STEP_COLORS["border"])
        self.style.map("Secondary.TButton", background=[("active", STEP_COLORS["border"])], foreground=[("active", STEP_COLORS["text"])])
        self.style.configure("Vertical.TScrollbar", background=STEP_COLORS["panel_alt"], troughcolor=STEP_COLORS["surface"], bordercolor=STEP_COLORS["border"], arrowcolor=STEP_COLORS["muted"])
        self.style.configure("TNotebook", background=STEP_COLORS["surface"], borderwidth=0)
        self.style.configure("TNotebook.Tab", background=STEP_COLORS["panel"], foreground=STEP_COLORS["muted"], padding=(18, 9), font=("Segoe UI", 10, "bold"))
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", STEP_COLORS["panel_alt"]), ("active", STEP_COLORS["border"])],
            foreground=[("selected", STEP_COLORS["text"]), ("active", STEP_COLORS["text"])],
        )

    def _add_scrollable_tab(self, text: str) -> tuple[ttk.Frame, tk.Canvas]:
        outer = ttk.Frame(self.notebook)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        canvas = tk.Canvas(outer, bg=STEP_COLORS["surface"], highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        frame = ttk.Frame(canvas, padding=12)
        window_id = canvas.create_window((0, 0), window=frame, anchor="nw")

        def refresh(_event=None) -> None:
            with contextlib.suppress(tk.TclError):
                canvas.configure(scrollregion=canvas.bbox("all"))

        def resize(event) -> None:
            canvas.itemconfigure(window_id, width=event.width)
            refresh()

        frame.bind("<Configure>", refresh)
        canvas.bind("<Configure>", resize)
        self.notebook.add(outer, text=text)
        return frame, canvas

    def _bind_scroll_canvas_widgets(self, widget, canvas: tk.Canvas) -> None:
        widget.bind("<MouseWheel>", lambda event, c=canvas: self._on_scroll_canvas_mousewheel(event, c), add="+")
        widget.bind("<Button-4>", lambda event, c=canvas: self._on_scroll_canvas_mousewheel(event, c), add="+")
        widget.bind("<Button-5>", lambda event, c=canvas: self._on_scroll_canvas_mousewheel(event, c), add="+")
        for child in widget.winfo_children():
            self._bind_scroll_canvas_widgets(child, canvas)

    def _on_scroll_canvas_mousewheel(self, event, canvas: tk.Canvas):
        bbox = canvas.bbox("all")
        if not bbox or bbox[3] <= canvas.winfo_height():
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
        canvas.yview_scroll(units, "units")
        return "break"

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)

        self._build_top_bar()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=12, pady=(8, 6))
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self.raw_tab, self.raw_tab_canvas = self._add_scrollable_tab("1 Raw conversion")
        self.spike_tab, self.spike_tab_canvas = self._add_scrollable_tab("2 Spike analysis")
        self.lfp_tab, self.lfp_tab_canvas = self._add_scrollable_tab("3 LFP analysis")
        self.plot_tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.plot_tab, text="4 Comparison and plotting")

        self._build_raw_tab()
        self._build_spike_tab()
        self._build_lfp_tab()
        self._build_plot_tab()
        self._bind_scroll_canvas_widgets(self.raw_tab, self.raw_tab_canvas)
        self._bind_scroll_canvas_widgets(self.spike_tab, self.spike_tab_canvas)
        self._bind_scroll_canvas_widgets(self.lfp_tab, self.lfp_tab_canvas)
        self._build_log()

    def _build_top_bar(self) -> None:
        bar = ttk.Frame(self.root, padding=(16, 14, 16, 6))
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(2, weight=1)

        self.logo_image = self._load_logo_image(54)
        if self.logo_image is not None:
            tk.Label(
                bar,
                image=self.logo_image,
                bg=STEP_COLORS["surface"],
                bd=0,
                highlightthickness=0,
            ).grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 12))

        title_block = ttk.Frame(bar)
        title_block.grid(row=0, column=1, rowspan=2, sticky="w")
        ttk.Label(title_block, text=APP_SHORT_NAME, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(title_block, text=APP_NAME, style="Stage.TLabel").grid(row=1, column=0, sticky="w", pady=(1, 0))

        chip_frame = ttk.Frame(bar)
        chip_frame.grid(row=0, column=2, rowspan=2, sticky="w", padx=28)
        self._step_chip(chip_frame, "1 Raw", STEP_COLORS["raw"], 0)
        self._step_chip(chip_frame, "2 Spikes", STEP_COLORS["spike"], 1)
        self._step_chip(chip_frame, "3 LFP", STEP_COLORS["lfp"], 2)
        self._step_chip(chip_frame, "4 Plots", STEP_COLORS["plot"], 3)

        ttk.Button(bar, text="Run Full Pipeline", style="Accent.TButton", command=self.run_full_pipeline).grid(
            row=0,
            column=3,
            rowspan=2,
            sticky="e",
            padx=(10, 0),
        )

        ttk.Label(bar, textvariable=self.stage_var, style="Stage.TLabel").grid(
            row=2,
            column=1,
            columnspan=3,
            sticky="w",
            pady=(8, 0),
        )

    def _load_logo_image(self, size: int) -> ImageTk.PhotoImage | None:
        if not APP_ICON.exists():
            return None
        try:
            image = Image.open(APP_ICON).convert("RGBA")
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(image)
        except Exception:
            return None

    def _step_chip(self, parent, text: str, color: str, column: int) -> None:
        chip = tk.Label(
            parent,
            text=text,
            bg=STEP_COLORS["panel"],
            fg=color,
            padx=12,
            pady=6,
            font=("Segoe UI", 9, "bold"),
            highlightbackground=color,
            highlightcolor=color,
            highlightthickness=1,
        )
        chip.grid(row=0, column=column, padx=(0, 8))

    def _tab_header(self, parent, row: int, number: str, title: str, color: str) -> None:
        header = tk.Frame(
            parent,
            bg=STEP_COLORS["panel"],
            height=52,
            highlightbackground=STEP_COLORS["border"],
            highlightthickness=1,
        )
        header.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 16))
        header.columnconfigure(2, weight=1)
        tk.Frame(header, bg=color, width=5).grid(row=0, column=0, sticky="nsw")
        tk.Label(
            header,
            text=number,
            bg=STEP_COLORS["panel"],
            fg=color,
            width=3,
            font=("Segoe UI", 15, "bold"),
        ).grid(row=0, column=1, sticky="nsw", padx=(14, 8), pady=10)
        tk.Label(
            header,
            text=title,
            bg=STEP_COLORS["panel"],
            fg=STEP_COLORS["text"],
            font=("Segoe UI", 14, "bold"),
        ).grid(row=0, column=2, sticky="w", pady=10)

    def _refresh_scrollregions(self) -> None:
        for name in ("raw_tab_canvas", "spike_tab_canvas", "lfp_tab_canvas", "plot_tab_canvas"):
            canvas = getattr(self, name, None)
            if canvas is not None:
                with contextlib.suppress(tk.TclError):
                    canvas.configure(scrollregion=canvas.bbox("all"))

    def _collapsible_section(
        self,
        parent,
        row: int,
        title: str,
        *,
        start_open: bool = False,
        columnspan: int = 4,
    ) -> ttk.Frame:
        outer = tk.Frame(
            parent,
            bg=STEP_COLORS["panel"],
            highlightbackground=STEP_COLORS["border"],
            highlightthickness=1,
        )
        outer.grid(row=row, column=0, columnspan=columnspan, sticky="ew", pady=6)
        outer.columnconfigure(0, weight=1)

        state = tk.BooleanVar(value=start_open)
        button_text = tk.StringVar(value=("- " if start_open else "+ ") + title)
        body = ttk.Frame(outer, padding=(12, 2, 12, 10))
        body.columnconfigure(1, weight=1)
        body.columnconfigure(3, weight=1)

        def toggle() -> None:
            state.set(not state.get())
            button_text.set(("- " if state.get() else "+ ") + title)
            if state.get():
                body.grid(row=1, column=0, sticky="ew")
            else:
                body.grid_remove()
            self.root.after_idle(self._refresh_scrollregions)

        ttk.Button(
            outer,
            textvariable=button_text,
            style="Secondary.TButton",
            command=toggle,
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=8)

        body.grid(row=1, column=0, sticky="ew")
        if not start_open:
            body.grid_remove()

        self._collapsible_sections.append({"title": title, "open": state, "body": body})
        return body

    def _on_tab_changed(self, _event=None) -> None:
        current = self.notebook.index(self.notebook.select())
        messages = [
            "Step 1: convert Intan raw recordings into CSV",
            "Step 2: order recordings, label epochs, and count spikes",
            "Step 3: extract LFP features and spike-LFP coupling from raw CSVs",
            "Step 4: generate selected comparison and plot outputs",
        ]
        if 0 <= current < len(messages):
            self.stage_var.set(messages[current])

    def _build_log(self) -> None:
        frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        frame.grid(row=2, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=0, sticky="w")
        self.log_text = tk.Text(
            frame,
            height=5,
            wrap="word",
            state="disabled",
            bg=STEP_COLORS["terminal"],
            fg=STEP_COLORS["terminal_text"],
            insertbackground=STEP_COLORS["terminal_text"],
            relief="flat",
            bd=0,
            padx=10,
            pady=8,
            font=("Consolas", 9),
        )
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.grid(row=1, column=0, sticky="nsew")
        scroll.grid(row=1, column=1, sticky="ns")

    def _build_raw_tab(self) -> None:
        frame = self.raw_tab
        frame.columnconfigure(0, weight=1)
        self._tab_header(frame, 0, "1", "Raw conversion", STEP_COLORS["raw"])

        settings = ttk.LabelFrame(frame, text="Raw conversion settings", padding=12)
        settings.grid(row=1, column=0, columnspan=3, sticky="ew")
        settings.columnconfigure(1, weight=1)

        self.raw_format_var = tk.StringVar(value="RHS")
        self.raw_input_dir_var = tk.StringVar(value=str(DEFAULT_DATA_DIR))
        self.raw_output_csv_var = tk.StringVar(
            value=str(DEFAULT_DATA_DIR / "all_channels_amplifier_raw.csv")
        )
        self.raw_channels_var = tk.StringVar(value="")
        self.raw_diff_pairs_var = tk.StringVar(value="")
        self.raw_continuous_time_var = tk.BooleanVar(value=True)
        self.raw_time_col_var = tk.StringVar(value="time_s")
        self.raw_chunk_input_dir_var = tk.StringVar(value=str(DEFAULT_DATA_DIR / "raw_chunks"))
        self.raw_chunk_output_csv_var = tk.StringVar(value=str(DEFAULT_DATA_DIR / "combined_raw_chunks.csv"))
        self.raw_chunk_pattern_var = tk.StringVar(value="chunk_*.csv")
        self.raw_chunk_strict_time_var = tk.BooleanVar(value=True)

        row = 0
        ttk.Label(settings, text="Intan format").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(settings, textvariable=self.raw_format_var, values=("RHS", "RHD"), width=12, state="readonly").grid(
            row=row, column=1, sticky="w", pady=4
        )

        row += 1
        self._path_row(settings, row, "Raw input folder", self.raw_input_dir_var, self._browse_raw_input_dir)
        row += 1
        self._path_row(settings, row, "Output CSV", self.raw_output_csv_var, self._browse_raw_output_csv)

        row += 1
        ttk.Label(settings, text="Target channels").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(settings, textvariable=self.raw_channels_var).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Label(settings, text="Comma-separated; blank means all channels", style="Hint.TLabel").grid(row=row, column=2, sticky="w", padx=8)

        row += 1
        ttk.Label(settings, text="Differential pairs").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(settings, textvariable=self.raw_diff_pairs_var).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Label(settings, text="Example: A-008,A-010; A-011,A-012", style="Hint.TLabel").grid(row=row, column=2, sticky="w", padx=8)

        row += 1
        ttk.Label(settings, text="Time column").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(settings, textvariable=self.raw_time_col_var, width=18).grid(row=row, column=1, sticky="w", pady=4)

        row += 1
        ttk.Checkbutton(settings, text="Continuous time across raw files", variable=self.raw_continuous_time_var).grid(
            row=row, column=1, sticky="w", pady=4
        )

        row += 1
        ttk.Button(settings, text="Run Raw Conversion", style="Raw.TButton", command=self.run_raw_conversion).grid(
            row=row, column=1, sticky="w", pady=12
        )

        combiner = ttk.LabelFrame(frame, text="Combine raw chunks", padding=12)
        combiner.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        combiner.columnconfigure(1, weight=1)

        row = 0
        self._path_row(combiner, row, "Raw chunks folder", self.raw_chunk_input_dir_var, self._browse_raw_chunk_input_dir)
        row += 1
        self._path_row(combiner, row, "Combined CSV", self.raw_chunk_output_csv_var, self._browse_raw_chunk_output_csv)
        row += 1
        ttk.Label(combiner, text="Chunk pattern").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(combiner, textvariable=self.raw_chunk_pattern_var, width=18).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Label(combiner, text="Example: chunk_*.csv", style="Hint.TLabel").grid(row=row, column=2, sticky="w", padx=8)
        row += 1
        ttk.Checkbutton(combiner, text="Check chunk time order", variable=self.raw_chunk_strict_time_var).grid(
            row=row, column=1, sticky="w", pady=4
        )
        row += 1
        ttk.Button(combiner, text="Combine Raw Chunks", style="Raw.TButton", command=self.run_raw_chunk_combine).grid(
            row=row, column=1, sticky="w", pady=12
        )

    def _build_spike_tab(self) -> None:
        frame = self.spike_tab
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)
        self._tab_header(frame, 0, "2", "Spike analysis", STEP_COLORS["spike"])

        settings = ttk.LabelFrame(frame, text="Spike-count settings", padding=12)
        settings.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        settings.columnconfigure(1, weight=1)

        order_box = ttk.LabelFrame(frame, text="Recording order and epoch labels", padding=12)
        order_box.grid(row=2, column=0, columnspan=3, sticky="nsew")
        order_box.columnconfigure(0, weight=1)
        order_box.rowconfigure(1, weight=1)

        self.spike_csv_dir_var = tk.StringVar(value=str(DEFAULT_DATA_DIR))
        self.spike_glob_var = tk.StringVar(value="*.csv")
        self.spike_window_var = tk.StringVar(value="1 min")
        self.spike_polarity_var = tk.StringVar(value="both")
        self.spike_events_csv_var = tk.StringVar(value="")
        self.spike_out_dir_var = tk.StringVar(
            value=str(DEFAULT_DATA_DIR / "spike_counts_per_min")
        )
        self.selected_epoch_var = tk.StringVar(value="Baseline")
        self.selection_count_var = tk.StringVar(value="0 of 0 selected")

        row = 0
        self._path_row(settings, row, "CSV folder", self.spike_csv_dir_var, self._browse_spike_csv_dir)
        row += 1
        ttk.Label(settings, text="CSV pattern").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(settings, textvariable=self.spike_glob_var, width=18).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Button(settings, text="Refresh Files", style="Secondary.TButton", command=self.refresh_spike_files).grid(row=row, column=2, sticky="w", padx=8)

        row += 1
        ttk.Label(settings, text="Spike-count window").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(
            settings,
            textvariable=self.spike_window_var,
            values=("30 sec", "1 min", "2 min", "3 min", "4 min", "5 min", "10 min"),
            width=12,
            state="readonly",
        ).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Label(settings, text="Use 30 sec here if you want true 30 sec plots later.", style="Hint.TLabel").grid(
            row=row,
            column=2,
            sticky="w",
            padx=8,
        )

        row += 1
        ttk.Label(settings, text="Spike polarity").grid(row=row, column=0, sticky="w", pady=4)
        polarity_controls = ttk.Frame(settings)
        polarity_controls.grid(row=row, column=1, columnspan=2, sticky="w", pady=4)
        ttk.Radiobutton(
            polarity_controls,
            text="All spikes",
            variable=self.spike_polarity_var,
            value="both",
        ).pack(side="left", padx=(0, 14))
        ttk.Radiobutton(
            polarity_controls,
            text="Negative only",
            variable=self.spike_polarity_var,
            value="neg",
        ).pack(side="left", padx=(0, 14))
        ttk.Radiobutton(
            polarity_controls,
            text="Positive only",
            variable=self.spike_polarity_var,
            value="pos",
        ).pack(side="left")

        row += 1
        self._path_row(settings, row, "Spike output folder", self.spike_out_dir_var, self._browse_spike_out_dir)
        row += 1

        controls = ttk.Frame(order_box)
        controls.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        ttk.Button(controls, text="Move Up", style="Secondary.TButton", command=lambda: self.move_recording(-1)).pack(side="left")
        ttk.Button(controls, text="Move Down", style="Secondary.TButton", command=lambda: self.move_recording(1)).pack(side="left", padx=6)
        ttk.Label(controls, textvariable=self.selection_count_var, style="Hint.TLabel").pack(side="left", padx=(12, 4))
        ttk.Button(controls, text="Select All", style="Secondary.TButton", command=self.select_all_recordings).pack(side="left", padx=4)
        ttk.Button(controls, text="Deselect All", style="Secondary.TButton", command=self.deselect_all_recordings).pack(side="left", padx=4)
        ttk.Label(controls, text="Selected label").pack(side="left", padx=(16, 4))
        ttk.Combobox(
            controls,
            textvariable=self.selected_epoch_var,
            values=("Baseline", "During", "Stimulation", "Perfusion", "Post 1", "Post 2", "Post 3", "Post"),
            width=20,
        ).pack(side="left")
        ttk.Button(controls, text="Apply Label to Selection", style="Secondary.TButton", command=self.apply_epoch_label).pack(side="left", padx=6)

        list_frame = ttk.Frame(order_box)
        list_frame.grid(row=1, column=0, columnspan=3, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.recording_listbox = tk.Listbox(list_frame, height=15, exportselection=False, selectmode="extended")
        self.recording_listbox.bind("<<ListboxSelect>>", self.on_recording_select)
        rec_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.recording_listbox.yview)
        self.recording_listbox.configure(yscrollcommand=rec_scroll.set)
        self.recording_listbox.grid(row=0, column=0, sticky="nsew")
        rec_scroll.grid(row=0, column=1, sticky="ns")

        run_controls = ttk.Frame(order_box)
        run_controls.grid(row=2, column=0, sticky="w", pady=(12, 0))
        ttk.Button(run_controls, text="Run Spike Counting", style="Spike.TButton", command=self.run_spike_counting).pack(side="left")
        ttk.Button(
            run_controls,
            text="Combine Files",
            style="Secondary.TButton",
            command=self.run_combine_spike_count_files,
        ).pack(side="left", padx=(8, 0))

    def _build_lfp_tab(self) -> None:
        frame = self.lfp_tab
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)
        self._tab_header(frame, 0, "3", "LFP analysis", STEP_COLORS["lfp"])

        self.lfp_feature_csv_var = tk.StringVar(value="")
        self.lfp_output_dir_var = tk.StringVar(value=str(DEFAULT_DATA_DIR / "lfp_analysis"))
        self.lfp_channels_var = tk.StringVar(value="")
        self.lfp_reference_scheme_var = tk.StringVar(value="none")
        self.lfp_window_var = tk.StringVar(value="60")
        self.lfp_exploratory_band_var = tk.StringVar(value="0.5-300")
        self.lfp_primary_band_var = tk.StringVar(value="1-40")
        self.lfp_welch_segment_var = tk.StringVar(value="4")
        self.lfp_welch_overlap_var = tk.StringVar(value="0.5")
        self.lfp_marker_bands_var = tk.StringVar(
            value="very_low=1-4; low=4-8; mid=8-13; high=13-30; high_40_100=40-100; high_100_150=100-150; high_150_300=150-300; total_0p5_300=0.5-300; primary=1-40"
        )
        self.lfp_simple_plot_bands_var = tk.StringVar(
            value="very_slow=0.5-2; slow_motor=2-5; intermediate_motor=5-9; fictive_flight_range=9-15; intact_flight_range=15-25; motor_harmonic_fast_synaptic=25-40; fast_population_activity=40-80"
        )
        self.lfp_notch_var = tk.StringVar(value="50")
        self.lfp_downsample_var = tk.StringVar(value="2000")
        self.lfp_min_valid_var = tk.StringVar(value="0.90")
        self.lfp_clipped_abs_var = tk.StringVar(value="")
        self.lfp_derivative_z_var = tk.StringVar(value="8")
        self.lfp_derivative_warning_fraction_var = tk.StringVar(value="0.005")
        self.lfp_derivative_fraction_var = tk.StringVar(value="0.02")
        self.lfp_derivative_spike_match_ms_var = tk.StringVar(value="2")
        self.lfp_derivative_coincidence_ms_var = tk.StringVar(value="2")
        self.lfp_flatline_min_duration_var = tk.StringVar(value="0.05")
        self.lfp_flatline_fraction_var = tk.StringVar(value="0.05")
        self.lfp_line_noise_ratio_var = tk.StringVar(value="0.25")
        self.lfp_manual_artifacts_var = tk.StringVar(value="")
        self.lfp_peak_prominence_var = tk.StringVar(value="3")
        self.lfp_phase_band_var = tk.StringVar(value="delta=1-4; theta=4-8; alpha=8-13; beta=13-30; gamma_30_60=30-60; gamma_60_100=60-100")
        self.lfp_min_spikes_var = tk.StringVar(value="20")
        self.lfp_surrogates_var = tk.StringVar(value="200")
        self.lfp_pairing_mode_var = tk.StringVar(value="same_channel")
        self.lfp_coupling_processing_mode_var = tk.StringVar(value="complex")
        self.lfp_simple_plots_var = tk.BooleanVar(value=True)
        self.lfp_explicit_pairs_var = tk.StringVar(value="")
        self.lfp_coupling_downsample_var = tk.StringVar(value="2000")
        self.lfp_workload_limit_var = tk.StringVar(value="250000")
        self.lfp_all_to_all_override_var = tk.BooleanVar(value=False)
        self.lfp_preparation_id_var = tk.StringVar(value="Preparation 1")
        self.lfp_signal_unit_var = tk.StringVar(value="uV")

        source_box = ttk.LabelFrame(frame, text="Recording order and epoch labels", padding=12)
        source_box.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(0, 10))
        source_box.columnconfigure(0, weight=1)
        source_box.rowconfigure(1, weight=1)

        source_controls = ttk.Frame(source_box)
        source_controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(source_controls, text="Refresh Ordered CSVs", style="Secondary.TButton", command=self.refresh_spike_files).pack(side="left")
        ttk.Button(source_controls, text="Move Up", style="Secondary.TButton", command=lambda: self.move_recording(-1)).pack(side="left", padx=(8, 0))
        ttk.Button(source_controls, text="Move Down", style="Secondary.TButton", command=lambda: self.move_recording(1)).pack(side="left", padx=(6, 0))
        ttk.Label(source_controls, text="Epoch label").pack(side="left", padx=(16, 4))
        ttk.Combobox(
            source_controls,
            textvariable=self.selected_epoch_var,
            values=("Baseline", "Baseline 1", "Pre", "During", "Treatment", "Stimulation", "Perfusion", "Post", "Post 1", "Recovery", "Washout"),
            width=18,
        ).pack(side="left")
        ttk.Button(source_controls, text="Apply Label to Selection", style="Lfp.TButton", command=self.apply_epoch_label).pack(side="left", padx=(6, 0))

        prep_controls = ttk.Frame(source_box)
        prep_controls.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(prep_controls, text="Preparation ID").pack(side="left")
        ttk.Entry(prep_controls, textvariable=self.lfp_preparation_id_var, width=24).pack(side="left", padx=(6, 8))
        ttk.Button(prep_controls, text="Apply Preparation to Selection", style="Secondary.TButton", command=self.apply_lfp_preparation_to_selection).pack(side="left")
        ttk.Button(prep_controls, text="Apply Preparation to All", style="Secondary.TButton", command=self.apply_lfp_preparation_to_all).pack(side="left", padx=(6, 0))

        list_frame = ttk.Frame(source_box)
        list_frame.grid(row=1, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.lfp_recording_listbox = tk.Listbox(list_frame, height=12, exportselection=False, selectmode="extended")
        self.lfp_recording_listbox.configure(
            bg=STEP_COLORS["input"],
            fg=STEP_COLORS["text"],
            selectbackground=STEP_COLORS["lfp"],
            selectforeground=STEP_COLORS["input"],
            relief="flat",
            highlightbackground=STEP_COLORS["border"],
            highlightcolor=STEP_COLORS["lfp"],
            highlightthickness=1,
            activestyle="none",
            font=("Consolas", 9),
        )
        self.lfp_recording_listbox.bind("<<ListboxSelect>>", self.on_lfp_recording_select)
        lfp_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.lfp_recording_listbox.yview)
        self.lfp_recording_listbox.configure(yscrollcommand=lfp_scroll.set)
        self.lfp_recording_listbox.grid(row=0, column=0, sticky="nsew")
        lfp_scroll.grid(row=0, column=1, sticky="ns")

        settings = ttk.LabelFrame(frame, text="LFP processing settings", padding=12)
        settings.grid(row=2, column=0, columnspan=3, sticky="ew")
        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(3, weight=1)
        row = 0
        ttk.Label(
            settings,
            text="Basic LFP settings stay visible. Open the sections below for detailed extraction, QC, and coupling parameters.",
            style="Hint.TLabel",
        ).grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 8))
        row += 1
        self._path_row(settings, row, "LFP output folder", self.lfp_output_dir_var, self._browse_lfp_output_dir)
        row += 1
        self._path_row(settings, row, "LFP feature CSV", self.lfp_feature_csv_var, self._browse_lfp_feature_csv)
        row += 1
        ttk.Checkbutton(
            settings,
            text="Output simple LFP plots after extraction",
            variable=self.lfp_simple_plots_var,
        ).grid(row=row, column=1, columnspan=3, sticky="w", pady=4)
        row += 1
        ttk.Label(settings, text="Channels").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(settings, textvariable=self.lfp_channels_var).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Label(settings, text="Reference").grid(row=row, column=2, sticky="w", padx=(14, 4), pady=4)
        ttk.Entry(settings, textvariable=self.lfp_reference_scheme_var, width=18).grid(row=row, column=3, sticky="ew", pady=4)
        row += 1
        ttk.Label(settings, text="Signal unit").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(settings, textvariable=self.lfp_signal_unit_var, values=("uV", "mV", "V"), width=10, state="readonly").grid(row=row, column=1, sticky="w", pady=4)
        ttk.Label(settings, text="Window s").grid(row=row, column=2, sticky="w", padx=(14, 4), pady=4)
        ttk.Combobox(settings, textvariable=self.lfp_window_var, values=("30", "60", "120", "300", "600"), width=10, state="readonly").grid(row=row, column=3, sticky="w", pady=4)
        row += 1
        ttk.Label(settings, text="Coupling mode").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(
            settings,
            textvariable=self.lfp_coupling_processing_mode_var,
            values=("simple", "complex"),
            width=16,
            state="readonly",
        ).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Label(settings, text="Pairing mode").grid(row=row, column=2, sticky="w", padx=(14, 4), pady=4)
        ttk.Combobox(
            settings,
            textvariable=self.lfp_pairing_mode_var,
            values=("same_channel",),
            width=16,
            state="readonly",
        ).grid(row=row, column=3, sticky="w", pady=4)
        row += 1

        feature_box = self._collapsible_section(settings, row, "Feature bands and sampling")
        feature_box.columnconfigure(1, weight=1)
        feature_box.columnconfigure(3, weight=1)
        feature_row = 0
        ttk.Label(feature_box, text="Welch segment s").grid(row=feature_row, column=0, sticky="w", pady=4)
        ttk.Entry(feature_box, textvariable=self.lfp_welch_segment_var, width=10).grid(row=feature_row, column=1, sticky="w", pady=4)
        ttk.Label(feature_box, text="Welch overlap").grid(row=feature_row, column=2, sticky="w", padx=(14, 4), pady=4)
        ttk.Entry(feature_box, textvariable=self.lfp_welch_overlap_var, width=10).grid(row=feature_row, column=3, sticky="w", pady=4)
        feature_row += 1
        ttk.Label(feature_box, text="Exploratory band Hz").grid(row=feature_row, column=0, sticky="w", pady=4)
        ttk.Entry(feature_box, textvariable=self.lfp_exploratory_band_var, width=16).grid(row=feature_row, column=1, sticky="w", pady=4)
        ttk.Label(feature_box, text="Primary band Hz").grid(row=feature_row, column=2, sticky="w", padx=(14, 4), pady=4)
        ttk.Entry(feature_box, textvariable=self.lfp_primary_band_var, width=16).grid(row=feature_row, column=3, sticky="w", pady=4)
        feature_row += 1
        ttk.Label(feature_box, text="Marker bands").grid(row=feature_row, column=0, sticky="w", pady=4)
        ttk.Entry(feature_box, textvariable=self.lfp_marker_bands_var).grid(row=feature_row, column=1, columnspan=3, sticky="ew", pady=4)
        feature_row += 1
        ttk.Label(feature_box, text="Simple plot bands").grid(row=feature_row, column=0, sticky="w", pady=4)
        ttk.Entry(feature_box, textvariable=self.lfp_simple_plot_bands_var).grid(row=feature_row, column=1, columnspan=3, sticky="ew", pady=4)
        feature_row += 1
        ttk.Label(feature_box, text="Notch Hz").grid(row=feature_row, column=0, sticky="w", pady=4)
        ttk.Entry(feature_box, textvariable=self.lfp_notch_var, width=10).grid(row=feature_row, column=1, sticky="w", pady=4)
        ttk.Label(feature_box, text="Downsample Hz").grid(row=feature_row, column=2, sticky="w", padx=(14, 4), pady=4)
        ttk.Combobox(feature_box, textvariable=self.lfp_downsample_var, values=("none", "250", "500", "1000", "2000"), width=10).grid(row=feature_row, column=3, sticky="w", pady=4)
        feature_row += 1
        ttk.Label(feature_box, text="Peak prominence dB").grid(row=feature_row, column=0, sticky="w", pady=4)
        ttk.Entry(feature_box, textvariable=self.lfp_peak_prominence_var, width=10).grid(row=feature_row, column=1, sticky="w", pady=4)
        row += 1

        qc_box = self._collapsible_section(settings, row, "Quality control thresholds")
        qc_box.columnconfigure(1, weight=1)
        qc_box.columnconfigure(3, weight=1)
        qc_row = 0
        ttk.Label(qc_box, text="Min valid fraction").grid(row=qc_row, column=0, sticky="w", pady=4)
        ttk.Entry(qc_box, textvariable=self.lfp_min_valid_var, width=10).grid(row=qc_row, column=1, sticky="w", pady=4)
        ttk.Label(qc_box, text="Clipped abs uV").grid(row=qc_row, column=2, sticky="w", padx=(14, 4), pady=4)
        ttk.Entry(qc_box, textvariable=self.lfp_clipped_abs_var, width=10).grid(row=qc_row, column=3, sticky="w", pady=4)
        qc_row += 1
        ttk.Label(qc_box, text="Derivative z").grid(row=qc_row, column=0, sticky="w", pady=4)
        ttk.Entry(qc_box, textvariable=self.lfp_derivative_z_var, width=10).grid(row=qc_row, column=1, sticky="w", pady=4)
        warning_label = ttk.Label(qc_box, text="Derivative warning fraction")
        warning_label.grid(row=qc_row, column=2, sticky="w", padx=(14, 4), pady=4)
        warning_entry = ttk.Entry(qc_box, textvariable=self.lfp_derivative_warning_fraction_var, width=10)
        warning_entry.grid(row=qc_row, column=3, sticky="w", pady=4)
        self._add_tooltip(warning_label, "Candidate derivative warning does not remove samples and does not fail the window.")
        self._add_tooltip(warning_entry, "Default 0.005. Windows at or above this fraction are flagged for review only.")
        qc_row += 1
        hard_label = ttk.Label(qc_box, text="Derivative hard fraction")
        hard_label.grid(row=qc_row, column=0, sticky="w", pady=4)
        hard_entry = ttk.Entry(qc_box, textvariable=self.lfp_derivative_fraction_var, width=10)
        hard_entry.grid(row=qc_row, column=1, sticky="w", pady=4)
        self._add_tooltip(hard_label, "Derivative hard failure rejects the complete window only when candidate burden exceeds this fraction.")
        self._add_tooltip(hard_entry, "Default 0.02. Derivative count alone is not used to reject windows.")
        ttk.Label(qc_box, text="Derivative spike-match ms").grid(row=qc_row, column=2, sticky="w", padx=(14, 4), pady=4)
        ttk.Entry(qc_box, textvariable=self.lfp_derivative_spike_match_ms_var, width=10).grid(row=qc_row, column=3, sticky="w", pady=4)
        qc_row += 1
        flat_dur_label = ttk.Label(qc_box, text="Flatline min duration s")
        flat_dur_label.grid(row=qc_row, column=0, sticky="w", pady=4)
        flat_dur_entry = ttk.Entry(qc_box, textvariable=self.lfp_flatline_min_duration_var, width=10)
        flat_dur_entry.grid(row=qc_row, column=1, sticky="w", pady=4)
        ttk.Label(qc_box, text="Derivative coincidence ms").grid(row=qc_row, column=2, sticky="w", padx=(14, 4), pady=4)
        ttk.Entry(qc_box, textvariable=self.lfp_derivative_coincidence_ms_var, width=10).grid(row=qc_row, column=3, sticky="w", pady=4)
        self._add_tooltip(flat_dur_label, "Short repeated samples are ignored; only sustained flatlines become hard masks.")
        self._add_tooltip(flat_dur_entry, "Default 0.05 seconds.")
        qc_row += 1
        ttk.Label(qc_box, text="Flatline fraction").grid(row=qc_row, column=0, sticky="w", pady=4)
        ttk.Entry(qc_box, textvariable=self.lfp_flatline_fraction_var, width=10).grid(row=qc_row, column=1, sticky="w", pady=4)
        ttk.Label(qc_box, text="Line-noise ratio").grid(row=qc_row, column=2, sticky="w", padx=(14, 4), pady=4)
        ttk.Entry(qc_box, textvariable=self.lfp_line_noise_ratio_var, width=10).grid(row=qc_row, column=3, sticky="w", pady=4)
        qc_row += 1
        ttk.Label(qc_box, text="Manual artifact intervals").grid(row=qc_row, column=0, sticky="w", pady=4)
        ttk.Entry(qc_box, textvariable=self.lfp_manual_artifacts_var).grid(row=qc_row, column=1, columnspan=3, sticky="ew", pady=4)
        row += 1

        coupling_box = self._collapsible_section(settings, row, "Spike-LFP coupling details")
        coupling_box.columnconfigure(1, weight=1)
        coupling_box.columnconfigure(3, weight=1)
        coupling_row = 0
        ttk.Label(coupling_box, text="PPC phase bands Hz").grid(row=coupling_row, column=0, sticky="w", pady=4)
        ttk.Entry(coupling_box, textvariable=self.lfp_phase_band_var).grid(row=coupling_row, column=1, columnspan=3, sticky="ew", pady=4)
        coupling_row += 1
        ttk.Label(coupling_box, text="PPC min spikes").grid(row=coupling_row, column=0, sticky="w", pady=4)
        ttk.Entry(coupling_box, textvariable=self.lfp_min_spikes_var, width=10).grid(row=coupling_row, column=1, sticky="w", pady=4)
        ttk.Label(coupling_box, text="PPC surrogates").grid(row=coupling_row, column=2, sticky="w", padx=(14, 4), pady=4)
        ttk.Entry(coupling_box, textvariable=self.lfp_surrogates_var, width=10).grid(row=coupling_row, column=3, sticky="w", pady=4)
        coupling_row += 1
        ttk.Label(coupling_box, text="Explicit pairs").grid(row=coupling_row, column=0, sticky="w", pady=4)
        ttk.Entry(coupling_box, textvariable=self.lfp_explicit_pairs_var, width=28, state="disabled").grid(row=coupling_row, column=1, columnspan=3, sticky="ew", pady=4)
        coupling_row += 1
        ttk.Label(coupling_box, text="Coupling fs Hz").grid(row=coupling_row, column=0, sticky="w", pady=4)
        ttk.Entry(coupling_box, textvariable=self.lfp_coupling_downsample_var, width=10).grid(row=coupling_row, column=1, sticky="w", pady=4)
        ttk.Label(coupling_box, text="Workload limit").grid(row=coupling_row, column=2, sticky="w", padx=(14, 4), pady=4)
        ttk.Entry(coupling_box, textvariable=self.lfp_workload_limit_var, width=12).grid(row=coupling_row, column=3, sticky="w", pady=4)
        coupling_row += 1
        ttk.Checkbutton(
            coupling_box,
            text="Allow large all-to-all coupling",
            variable=self.lfp_all_to_all_override_var,
            state="disabled",
        ).grid(row=coupling_row, column=1, columnspan=3, sticky="w", pady=4)
        row += 1

        buttons = ttk.Frame(settings)
        buttons.grid(row=row, column=0, columnspan=4, sticky="w", pady=(10, 0))
        ttk.Button(buttons, text="Extract LFP Features", style="Lfp.TButton", command=self.run_lfp_feature_extraction).pack(side="left")
        ttk.Button(buttons, text="Plot Complex LFP Comparisons", style="Plot.TButton", command=self.run_lfp_plotting).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Calculate Spike-LFP Coupling", style="Lfp.TButton", command=self.run_spike_lfp_coupling).pack(side="left", padx=(8, 0))
        self._refresh_lfp_recording_listbox()

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
        frame.rowconfigure(5, weight=1, minsize=320)
        self._tab_header(frame, 0, "4", "Comparison and plotting", STEP_COLORS["plot"])

        self.plot_input_var = tk.StringVar(value="")
        self.plot_out_dir_var = tk.StringVar(value="")
        self.plot_id_cols_var = tk.StringVar(value="channel")
        self.plot_group_col_var = tk.StringVar(value="")
        self.plot_time_course_var = tk.BooleanVar(value=True)
        self.continuous_plot_title_var = tk.StringVar(value="")
        self.continuous_boundary_var = tk.StringVar(value="")
        self.continuous_y_top_var = tk.StringVar(value="")
        self.continuous_trace_color_var = tk.StringVar(value="tab:blue")
        self.continuous_marker_var = tk.StringVar(value="o")
        self.continuous_linewidth_var = tk.StringVar(value="2")
        self.continuous_fig_width_var = tk.StringVar(value="16")
        self.continuous_fig_height_var = tk.StringVar(value="5.5")
        self.continuous_dpi_var = tk.StringVar(value="200")
        self.continuous_max_x_ticks_var = tk.StringVar(value="24")
        self.continuous_plot_bin_var = tk.StringVar(value="1 min")
        self.continuous_include_legend_var = tk.BooleanVar(value=True)
        self.epoch_title_vars = {
            key: tk.StringVar(value="")
            for key, _label, default in EPOCH_TITLE_FIELDS
        }
        self.perfusion_title_var = self.epoch_title_vars["perfusion"]
        self.perfusion_substance_var = tk.StringVar(value="")
        self.perfusion_concentration_var = tk.StringVar(value="")
        self.amp_overlay_var = tk.BooleanVar(value=True)
        self.amp_boundary_label_var = tk.StringVar(value="H2O2 treatment - not recorded")
        self.amp_polarity_var = tk.StringVar(value="both")
        for key in ("raster", "rate", "isi", "acg", "waveform"):
            setattr(self, f"amp_plot_{key}_var", tk.BooleanVar(value=False))

        self.phase_baseline_var = tk.BooleanVar(value=True)
        self.phase_stim_var = tk.BooleanVar(value=True)
        self.phase_post_var = tk.BooleanVar(value=True)
        self.comp_baseline_var = tk.BooleanVar(value=True)
        self.comp_stim_var = tk.BooleanVar(value=True)
        self.comp_post_var = tk.BooleanVar(value=True)
        self.pool_percent_change_var = tk.BooleanVar(value=False)
        self.middle_phase_split_var = tk.StringVar(value="Whole")
        self.post_phase_split_var = tk.StringVar(value="Whole")
        self.percent_show_channel_labels_var = tk.BooleanVar(value=True)
        self.pooled_percent_title_var = tk.StringVar(value="")
        self.percent_part_titles: dict[str, str] = {}
        self.percent_part_titles_summary_var = tk.StringVar(value="Default non-pooled titles")

        row = 1
        self._path_row(frame, row, "Spike-count file", self.plot_input_var, self._browse_plot_input)
        row += 1
        self._path_row(frame, row, "Plot output folder", self.plot_out_dir_var, self._browse_plot_out_dir)

        selector = ttk.LabelFrame(frame, text="Plot and analysis tools", padding=12)
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
            title="Percentage Change from Baseline",
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
        self._plot_tool_card(
            selector,
            key="amplitude_change",
            title="Spike amplitude change",
            description="Amplitude change from baseline, replotted from spike-amplitude time-course CSVs.",
            color=STEP_COLORS["spike"],
            row=2,
            column=0,
        )

        row += 1
        self.plot_detail_frame = ttk.Frame(frame)
        self.plot_detail_frame.grid(row=row, column=0, columnspan=3, sticky="nsew")
        self.plot_detail_frame.columnconfigure(0, weight=1)
        self.plot_detail_frame.rowconfigure(0, weight=1)

        self.plot_detail_panels = {
            "baseline_post": self._build_baseline_post_plot_panel(self.plot_detail_frame),
            "continuous_spike": self._build_continuous_spike_plot_panel(self.plot_detail_frame),
            "amplitude_change": self._build_amplitude_change_plot_panel(self.plot_detail_frame),
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
            bg=STEP_COLORS["panel_alt"],
            highlightbackground=STEP_COLORS["border"],
            highlightthickness=1,
            padx=14,
            pady=12,
            cursor="hand2",
        )
        card.grid(row=row, column=column, sticky="ew", padx=(0, 8) if column == 0 else (8, 0), pady=2)
        card.columnconfigure(1, weight=1)

        accent = tk.Frame(card, bg=color, width=6)
        accent.grid(row=0, column=0, rowspan=2, sticky="nsw", padx=(0, 10))
        title_label = tk.Label(card, text=title, bg=STEP_COLORS["panel_alt"], fg=STEP_COLORS["text"], font=("Segoe UI", 11, "bold"))
        title_label.grid(row=0, column=1, sticky="w")
        desc_label = tk.Label(
            card,
            text=description,
            bg=STEP_COLORS["panel_alt"],
            fg=STEP_COLORS["muted"],
            justify="left",
            wraplength=430,
            font=("Segoe UI", 9),
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
            "baseline_post": "Percentage Change from Baseline selected",
            "continuous_spike": "Continuous spike-count plots selected",
            "amplitude_change": "Spike amplitude change selected",
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
        panel = ttk.LabelFrame(parent, text="Percentage Change from Baseline parameters", padding=12)
        panel.columnconfigure(0, weight=1)

        row = 0
        phase_box = ttk.LabelFrame(panel, text="Recorded phases", padding=10, style="Subsection.TLabelframe")
        phase_box.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        ttk.Checkbutton(phase_box, text="Baseline", variable=self.phase_baseline_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(phase_box, text="Stimulation", variable=self.phase_stim_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(phase_box, text="Post", variable=self.phase_post_var).pack(side="left", padx=(0, 16))

        row += 1
        comp_box = ttk.LabelFrame(panel, text="Comparisons to output", padding=10, style="Subsection.TLabelframe")
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
        split_box = ttk.LabelFrame(panel, text="Phase split comparisons", padding=10, style="Subsection.TLabelframe")
        split_box.grid(row=row, column=0, sticky="ew", pady=6)
        split_box.columnconfigure(1, weight=1)
        split_box.columnconfigure(3, weight=1)
        ttk.Label(split_box, text="Treatment/perfusion").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Combobox(
            split_box,
            textvariable=self.middle_phase_split_var,
            values=("Whole", "Halves", "Thirds"),
            width=10,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=(8, 18), pady=4)
        ttk.Label(split_box, text="Post").grid(row=0, column=2, sticky="w", padx=(14, 4), pady=4)
        ttk.Combobox(
            split_box,
            textvariable=self.post_phase_split_var,
            values=("Whole", "Halves", "Thirds"),
            width=10,
            state="readonly",
        ).grid(row=0, column=3, sticky="w", padx=(8, 0), pady=4)

        row += 1
        ttk.Checkbutton(
            panel,
            text="Show channel name on Baseline/change plots",
            variable=self.percent_show_channel_labels_var,
        ).grid(
            row=row,
            column=0,
            sticky="w",
            pady=6,
        )

        row += 1
        title_box = ttk.LabelFrame(panel, text="Percent-change plot titles", padding=10, style="Subsection.TLabelframe")
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

    def _build_epoch_title_box(self, parent, row: int, columnspan: int = 4) -> int:
        epoch_box = ttk.LabelFrame(parent, text="Epoch display titles", padding=10, style="Subsection.TLabelframe")
        epoch_box.grid(row=row, column=0, columnspan=columnspan, sticky="ew", pady=(8, 6))
        epoch_box.columnconfigure(1, weight=1)
        epoch_box.columnconfigure(3, weight=1)
        for index, (key, label, _default) in enumerate(EPOCH_TITLE_FIELDS):
            field_row = index // 2
            field_col = (index % 2) * 2
            ttk.Label(epoch_box, text=label).grid(row=field_row, column=field_col, sticky="w", pady=4)
            ttk.Entry(epoch_box, textvariable=self.epoch_title_vars[key], width=26).grid(
                row=field_row,
                column=field_col + 1,
                sticky="ew",
                padx=(8, 20 if field_col == 0 else 0),
                pady=4,
            )

        perfusion_row = (len(EPOCH_TITLE_FIELDS) + 1) // 2
        ttk.Label(epoch_box, text="Perfusion substance").grid(row=perfusion_row, column=0, sticky="w", pady=(10, 4))
        ttk.Entry(epoch_box, textvariable=self.perfusion_substance_var, width=18).grid(
            row=perfusion_row,
            column=1,
            sticky="w",
            padx=(8, 20),
            pady=(10, 4),
        )
        ttk.Label(epoch_box, text="Perfusion legend concentration").grid(
            row=perfusion_row,
            column=2,
            sticky="w",
            pady=(10, 4),
        )
        ttk.Combobox(
            epoch_box,
            textvariable=self.perfusion_concentration_var,
            values=PERFUSION_CONCENTRATIONS,
            width=10,
        ).grid(row=perfusion_row, column=3, sticky="w", padx=(8, 0), pady=(10, 4))
        return row + 1

    def _build_continuous_spike_plot_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="Continuous spike-count plot parameters", padding=12)
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
        row = self._build_epoch_title_box(panel, row)
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
        ttk.Checkbutton(
            panel,
            text="Include legend",
            variable=self.continuous_include_legend_var,
        ).grid(row=row, column=1, sticky="w", pady=4)
        row += 1
        ttk.Button(panel, text="Run Plotting", style="Plot.TButton", command=self.run_continuous_spike_plot).grid(
            row=row,
            column=1,
            sticky="w",
            pady=(8, 0),
        )
        return panel

    def _build_amplitude_change_plot_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="Spike amplitude change parameters", padding=12)
        panel.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(
            panel,
            text="Uses the shared Plot output folder above. Leave it blank to save next to the input CSV.",
            style="Hint.TLabel",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1
        ttk.Label(panel, text="Polarity").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(
            panel,
            textvariable=self.amp_polarity_var,
            values=("both", "pos", "neg"),
            width=10,
            state="readonly",
        ).grid(row=row, column=1, sticky="w", pady=4)
        ttk.Label(panel, text="both = positive + negative combined", style="Hint.TLabel").grid(
            row=row, column=2, sticky="w", padx=(8, 0), pady=4
        )
        row += 1
        ttk.Label(panel, text="Baseline/Post boundary label").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(panel, textvariable=self.amp_boundary_label_var).grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        row += 1
        ttk.Label(panel, text="Diagnostic plots").grid(row=row, column=0, sticky="nw", pady=4)
        diag_frame = ttk.Frame(panel)
        diag_frame.grid(row=row, column=1, columnspan=2, sticky="w", pady=4)
        for key, label_text in (("raster", "Raster"), ("rate", "Rate"), ("isi", "ISI"), ("acg", "Autocorrelogram"), ("waveform", "Average waveforms")):
            ttk.Checkbutton(diag_frame, text=label_text, variable=getattr(self, f"amp_plot_{key}_var")).pack(side="left", padx=(0, 12))
        row += 1
        amp_buttons = ttk.Frame(panel)
        amp_buttons.grid(row=row, column=1, sticky="w", pady=(8, 0))
        ttk.Button(amp_buttons, text="Run Plotting", style="Plot.TButton", command=self.run_amplitude_change_plot).pack(side="left")
        return panel

    def _path_row(self, parent, row: int, label: str, var: tk.StringVar, command) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="Browse", style="Secondary.TButton", command=command).grid(row=row, column=2, sticky="w", padx=8, pady=4)

    def log(self, text: str) -> None:
        self.log_queue.put(text)

    def _append_to_text_widget(self, widget: tk.Text | None, text: str) -> None:
        if widget is None:
            return
        with contextlib.suppress(tk.TclError):
            widget.configure(state="normal")
            widget.insert("end", text)
            widget.see("end")
            widget.configure(state="disabled")

    def _drain_log_queue(self) -> None:
        try:
            while True:
                text = self.log_queue.get_nowait()
                self._append_to_text_widget(self.log_text, text)
                self._append_to_text_widget(self.processing_text, text)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log_queue)

    def _show_processing_terminal(self, title: str) -> None:
        self._close_processing_terminal()
        window = tk.Toplevel(self.root)
        window.title(f"{APP_SHORT_NAME} processing")
        window.configure(bg=STEP_COLORS["terminal"])
        window.geometry("900x460")
        window.minsize(700, 320)
        window.transient(self.root)
        if APP_ICON.exists():
            with contextlib.suppress(tk.TclError):
                window.iconbitmap(str(APP_ICON))

        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)
        tk.Label(
            window,
            text=f"Processing: {title}",
            bg=STEP_COLORS["terminal"],
            fg=STEP_COLORS["text"],
            font=("Segoe UI", 13, "bold"),
            padx=14,
            pady=10,
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        ttk.Button(window, text="Cancel", style="Secondary.TButton", command=self.cancel_current_worker).grid(
            row=0,
            column=1,
            sticky="e",
            padx=10,
            pady=8,
        )

        text = tk.Text(
            window,
            wrap="word",
            state="disabled",
            bg=STEP_COLORS["terminal"],
            fg=STEP_COLORS["terminal_text"],
            insertbackground=STEP_COLORS["terminal_text"],
            relief="flat",
            bd=0,
            padx=14,
            pady=10,
            font=("Consolas", 10),
        )
        scroll = ttk.Scrollbar(window, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.grid(row=1, column=0, sticky="nsew", padx=(10, 0), pady=(0, 10))
        scroll.grid(row=1, column=1, sticky="ns", padx=(0, 10), pady=(0, 10))

        self.processing_window = window
        self.processing_text = text

    def _close_processing_terminal(self) -> None:
        window = self.processing_window
        self.processing_window = None
        self.processing_text = None
        if window is not None:
            with contextlib.suppress(tk.TclError):
                window.destroy()

    def cancel_current_worker(self) -> None:
        if self.worker_cancel_event is not None:
            self.worker_cancel_event.set()
            self.log_queue.put("[cancel] Cancellation requested; the running calculation will stop at the next checkpoint.\n")

    def run_background(self, title: str, func, on_success=None, cancel_event: threading.Event | None = None) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Pipeline is running", "Wait for the current step to finish before starting another.")
            return
        self.worker_cancel_event = cancel_event or threading.Event()

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
                self.worker_cancel_event = None
                if failed:
                    self.status_var.set(f"{title} failed")
                else:
                    self.status_var.set(f"{title} finished")
                    self.log_queue.put(f"--- {title} finished ---\n")
                    if on_success:
                        on_success(result)
                    self.root.after(900, self._close_processing_terminal)

            self.root.after(0, finish)

        self.status_var.set(f"{title} running...")
        self._show_processing_terminal(title)
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

    def _browse_raw_chunk_input_dir(self) -> None:
        path = filedialog.askdirectory(
            title="Choose raw chunks folder",
            initialdir=self._initial_dir(self.raw_chunk_input_dir_var),
        )
        if path:
            self.raw_chunk_input_dir_var.set(path)
            self.raw_chunk_output_csv_var.set(str(default_output_path(path)))

    def _browse_raw_chunk_output_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Choose combined raw CSV",
            initialdir=self._initial_dir(self.raw_chunk_output_csv_var),
            initialfile=Path(self.raw_chunk_output_csv_var.get()).name if self.raw_chunk_output_csv_var.get().strip() else "combined_raw_chunks.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.raw_chunk_output_csv_var.set(path)

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

    def _browse_lfp_output_dir(self) -> None:
        self._browse_directory_into(self.lfp_output_dir_var)

    def _browse_lfp_feature_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose LFP feature CSV",
            initialdir=self._initial_dir(self.lfp_feature_csv_var),
            filetypes=[("LFP feature CSV", "lfp_features_by_window.csv"), ("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.lfp_feature_csv_var.set(path)

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

    def _new_recording_item(self, path: str, index: int) -> dict[str, str]:
        source = str(path)
        stem = Path(source).stem
        return {
            "path": source,
            "label": "",
            "epoch_label": "",
            "segment_id": stem,
            "experiment_id": "",
            "animal_id": "",
            "preparation_id": "",
            "recording_id": stem,
            "recording_order": str(index),
            "source_file": source,
            "phase": "",
            "treatment": "",
            "concentration_value": "",
            "concentration_unit": "",
            "application_onset_s": "",
            "washout_onset_s": "",
            "channel": "",
            "reference_scheme": "none",
            "electrode_type": "",
            "impedance_1khz_ohm": "",
            "sampling_rate_hz": "",
            "signal_unit": "uV",
            "reference_stability": "",
            "vehicle_perfusion_control": "",
            "no_tissue_h2o2_control": "",
        }

    def refresh_spike_files(self) -> None:
        csv_dir = self.spike_csv_dir_var.get().strip()
        pattern = self.spike_glob_var.get().strip() or "*.csv"
        paths = sorted(glob.glob(os.path.join(csv_dir, pattern)))
        self.recordings = [self._new_recording_item(path, index) for index, path in enumerate(paths, start=1)]
        if self.recordings:
            self.recordings[0]["label"] = "Baseline"
            self.recordings[0]["epoch_label"] = "Baseline"
            self.recordings[0]["phase"] = "baseline"
        self._refresh_recording_listbox()
        self._update_selection_label()
        self.log(f"Found {len(self.recordings)} CSV file(s) for spike counting.\n")

    def _refresh_recording_listbox(self, selected_indices: list[int] | None = None) -> None:
        self.recording_listbox.delete(0, "end")
        for i, item in enumerate(self.recordings, start=1):
            label = item.get("epoch_label") or item.get("label") or "unlabeled"
            phase = item.get("phase") or self._phase_from_label(label) or "unlabeled"
            prep = item.get("preparation_id") or "prep?"
            self.recording_listbox.insert("end", f"{i:>2}. {Path(item['path']).name}    epoch={label}    phase={phase}    prep={prep}")
            self.recording_listbox.itemconfig(
                i - 1,
                background=self._epoch_row_color(label),
                foreground=STEP_COLORS["text"],
                selectbackground=STEP_COLORS["plot"],
                selectforeground=STEP_COLORS["input"],
            )
        if selected_indices:
            self.recording_listbox.selection_clear(0, "end")
            last_index = len(self.recordings) - 1
            valid_indices = [index for index in selected_indices if 0 <= index <= last_index]
            for index in valid_indices:
                self.recording_listbox.selection_set(index)
            if valid_indices:
                self.recording_listbox.activate(valid_indices[0])
                self.recording_listbox.see(valid_indices[0])
        if hasattr(self, "lfp_recording_listbox"):
            self._refresh_lfp_recording_listbox(selected_indices)

    def _epoch_row_color(self, label: str) -> str:
        key = label.strip().lower()
        if "baseline" in key:
            return EPOCH_ROW_COLORS["baseline"]
        if "stim" in key or "current" in key or "during" in key or "perfusion" in key or "treatment" in key:
            return EPOCH_ROW_COLORS["stimulation"]
        if "post" in key or "after" in key or "recovery" in key:
            return EPOCH_ROW_COLORS["post"]
        return EPOCH_ROW_COLORS["unlabeled"]

    def _selected_recording_indices(self) -> list[int]:
        selection = self.recording_listbox.curselection()
        if not selection:
            return []
        return sorted(int(index) for index in selection)

    def _selected_recording_index(self) -> int | None:
        indices = self._selected_recording_indices()
        if not indices:
            return None
        return indices[0]

    def _selected_any_recording_indices(self) -> list[int]:
        return self._selected_recording_indices() or self._selected_lfp_recording_indices()
    def _selected_recording_indices(self) -> tuple[int, ...]:
        return tuple(int(i) for i in self.recording_listbox.curselection())

    def _update_selection_label(self) -> None:
        indices = self._selected_recording_indices()
        total = len(self.recordings)
        if hasattr(self, "selection_count_var"):
            if indices:
                self.selection_count_var.set(f"{len(indices)} of {total} selected")
            else:
                self.selection_count_var.set(f"0 of {total} selected")

    def on_recording_select(self, _event=None) -> None:
        indices = self._selected_recording_indices()
        if not indices:
            self._update_selection_label()
            return
        self._update_selection_label()
        first_label = self.recordings[indices[0]]["label"] or ""
        all_same = all(self.recordings[i]["label"] == first_label for i in indices)
        self.selected_epoch_var.set(first_label if all_same else "")

    def apply_epoch_label(self) -> None:
        indices = self._selected_any_recording_indices()
        if not indices:
            return
        label = self.selected_epoch_var.get().strip()
        for idx in indices:
            self.recordings[idx]["label"] = label
        self._refresh_recording_listbox()
        for idx in indices:
            self.recording_listbox.selection_set(idx)
        self._update_selection_label()

    def select_all_recordings(self) -> None:
        self.recording_listbox.select_set(0, "end")
        self.on_recording_select()

    def deselect_all_recordings(self) -> None:
        self.recording_listbox.selection_clear(0, "end")
        self.on_recording_select()

    def _phase_from_label(self, label: str) -> str:
        text = normalise_epoch_text(label)
        if any(term in text for term in ("baseline", "basal", "before", "pre", "control")):
            return "baseline"
        if any(term in text for term in ("post", "after", "recovery", "washout")):
            return "post"
        if any(term in text for term in ("during", "stim", "stimulation", "perfusion", "treatment", "current")):
            return "treatment"
        return label.strip().lower()

    def _selected_lfp_recording_indices(self) -> list[int]:
        if not hasattr(self, "lfp_recording_listbox"):
            return []
        selection = self.lfp_recording_listbox.curselection()
        return sorted(int(index) for index in selection)

    def _refresh_lfp_recording_listbox(self, selected_indices: list[int] | None = None) -> None:
        if not hasattr(self, "lfp_recording_listbox"):
            return
        self.lfp_recording_listbox.delete(0, "end")
        for i, item in enumerate(self.recordings, start=1):
            label = item.get("epoch_label") or item.get("label") or "unlabeled"
            phase = item.get("phase") or self._phase_from_label(label) or "unlabeled"
            prep = item.get("preparation_id") or "prep?"
            channel = item.get("channel") or item.get("path", "")
            channel = channel if channel and channel != item.get("path") else "all channels"
            self.lfp_recording_listbox.insert(
                "end",
                f"{i:>2}. {Path(item['path']).name}    epoch={label}    phase={phase}    prep={prep}    {channel}",
            )
            self.lfp_recording_listbox.itemconfig(
                i - 1,
                background=self._epoch_row_color(str(phase)),
                foreground=STEP_COLORS["text"],
                selectbackground=STEP_COLORS["lfp"],
                selectforeground=STEP_COLORS["input"],
            )
        if selected_indices:
            self.lfp_recording_listbox.selection_clear(0, "end")
            last_index = len(self.recordings) - 1
            valid_indices = [index for index in selected_indices if 0 <= index <= last_index]
            for index in valid_indices:
                self.lfp_recording_listbox.selection_set(index)
            if valid_indices:
                self.lfp_recording_listbox.activate(valid_indices[0])
                self.lfp_recording_listbox.see(valid_indices[0])

    def on_lfp_recording_select(self, _event=None) -> None:
        indices = self._selected_lfp_recording_indices()
        if not indices:
            return
        labels = [(self.recordings[index].get("epoch_label") or self.recordings[index].get("label", "")).strip() for index in indices]
        if labels and all(label == labels[0] for label in labels):
            self.selected_epoch_var.set(labels[0])
        prep_values = [str(self.recordings[index].get("preparation_id", "")).strip() for index in indices]
        if prep_values and all(value == prep_values[0] for value in prep_values):
            self.lfp_preparation_id_var.set(prep_values[0] or self.lfp_preparation_id_var.get())

    def apply_lfp_preparation_to_selection(self) -> None:
        indices = self._selected_any_recording_indices()
        if not indices:
            messagebox.showerror("No recording selected", "Select one or more ordered recordings first.")
            return
        self._apply_lfp_preparation(indices)

    def apply_lfp_preparation_to_all(self) -> None:
        if not self.recordings:
            self.refresh_spike_files()
        if not self.recordings:
            messagebox.showerror("No CSV files", "No ordered recording segments are available.")
            return
        self._apply_lfp_preparation(list(range(len(self.recordings))))

    def _apply_lfp_preparation(self, indices: list[int]) -> None:
        preparation_id = self.lfp_preparation_id_var.get().strip()
        if not preparation_id:
            messagebox.showerror("Missing preparation", "Enter a preparation ID before applying it.")
            return
        for index in indices:
            item = self.recordings[index]
            item["preparation_id"] = preparation_id
        self._refresh_recording_listbox(indices)
        self.log(f"Applied preparation {preparation_id!r} to {len(indices)} recording segment(s).\n")

    def move_recording(self, delta: int) -> None:
        indices = self._selected_any_recording_indices()
        if not indices:
            return
        first, last = indices[0], indices[-1]
        block = [self.recordings[i] for i in indices]
        remaining = [r for i, r in enumerate(self.recordings) if i not in indices]
        anchor = first + delta
        if anchor < 0 or anchor > len(remaining):
            return
        if delta > 0 and indices[-1] == len(self.recordings) - 1:
            return

        selected = set(indices)
        if delta < 0:
            for index in indices:
                if index - 1 in selected:
                    continue
                self.recordings[index - 1], self.recordings[index] = self.recordings[index], self.recordings[index - 1]
                selected.remove(index)
                selected.add(index - 1)
        elif delta > 0:
            for index in reversed(indices):
                if index + 1 in selected:
                    continue
                self.recordings[index + 1], self.recordings[index] = self.recordings[index], self.recordings[index + 1]
                selected.remove(index)
                selected.add(index + 1)
        else:
            return

        for order, item in enumerate(self.recordings, start=1):
            item["recording_order"] = str(order)
        self._refresh_recording_listbox(sorted(selected))
        for rec in block:
            remaining.insert(anchor, rec)
        self.recordings = remaining
        self._refresh_recording_listbox()
        new_first = anchor
        new_last = anchor + len(block) - 1
        self.recording_listbox.selection_set(new_first, new_last)
        self._update_selection_label()

    def _format_command(self, cmd: list[str]) -> str:
        return " ".join(f'"{part}"' if " " in part else part for part in cmd)

    def _perfusion_title(self) -> str:
        return self.perfusion_title_var.get().strip()

    def _perfusion_legend_label(self) -> str:
        substance = self.perfusion_substance_var.get().strip()
        concentration = self.perfusion_concentration_var.get().strip()
        return " ".join(part for part in (substance, concentration) if part).strip()

    def _epoch_title_mapping(self) -> dict[str, str]:
        titles: dict[str, str] = {}
        for key, var in self.epoch_title_vars.items():
            title = var.get().strip()
            if title and title != EPOCH_TITLE_DEFAULTS.get(key, ""):
                titles[key] = title
        return titles

    def _epoch_title_for_label(self, label: object) -> str:
        titles = self._epoch_title_mapping()
        if contains_perfusion_label(label):
            return titles.get("perfusion") or self._perfusion_title() or str(label).strip()

        label_text = str(label).strip()
        label_norm = normalise_epoch_text(label_text)
        for key, title in sorted(titles.items(), key=lambda item: len(normalise_epoch_text(item[0])), reverse=True):
            key_norm = normalise_epoch_text(key)
            if key_norm and (key_norm == label_norm or key_norm in label_norm):
                return title

        if any(term in label_norm for term in ("baseline", "basal", "before", "pre", "control")):
            return titles.get("baseline") or "Baseline"
        if any(term in label_norm for term in ("post", "after", "recovery", "washout")):
            return titles.get("post") or label_text
        if any(term in label_norm for term in ("during", "stim", "stimulation", "current")):
            return titles.get("stimulation") or label_text
        return label_text

    def _plot_title_for_epoch_label(self, label: object) -> str:
        return self._epoch_title_for_label(label)

    def _run_bundled_script(self, script_path: str, args: list[str], failure_label: str) -> None:
        script = Path(script_path)
        module_name = "spie_" + "".join(ch if ch.isalnum() else "_" for ch in script.stem)
        old_argv = sys.argv[:]
        old_cwd = Path.cwd()
        try:
            os.chdir(SCRIPT_DIR)
            sys.argv = [str(script), *args]
            with contextlib.redirect_stdout(QueueWriter(self.log_queue)), contextlib.redirect_stderr(QueueWriter(self.log_queue)):
                module = load_script_module(script, f"{module_name}_{threading.get_ident()}")
                module.main()
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else int(bool(exc.code))
            if code:
                raise RuntimeError(f"{failure_label} failed with exit code {code}.") from exc
        finally:
            sys.argv = old_argv
            os.chdir(old_cwd)

    def _run_logged_subprocess(self, cmd: list[str], failure_label: str) -> None:
        self.log_queue.put("Running command:\n  " + self._format_command(cmd) + "\n")
        if getattr(sys, "frozen", False) and len(cmd) >= 2:
            self._run_bundled_script(cmd[1], cmd[2:], failure_label)
            return
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

    def _parse_frequency_range(self, text: str, label: str) -> tuple[float, float]:
        parts = [part for part in re.split(r"\s*[-:,]\s*", text.strip()) if part]
        if len(parts) != 2:
            raise ValueError(f"{label} must look like 1-40.")
        lo, hi = float(parts[0]), float(parts[1])
        if lo <= 0 or hi <= lo:
            raise ValueError(f"{label} must have 0 < low < high.")
        return lo, hi

    def _optional_float(self, text: str) -> float | None:
        raw = str(text).strip()
        if not raw or raw.lower() in {"none", "nan"}:
            return None
        return float(raw)

    def _parse_manual_artifacts(self, text: str) -> tuple[list[tuple[float, float]], str | None]:
        raw = text.strip()
        if not raw:
            return [], None
        candidate = Path(raw)
        if candidate.exists() and candidate.suffix.lower() == ".csv":
            return [], str(candidate)
        intervals: list[tuple[float, float]] = []
        for item in raw.split(";"):
            item = item.strip()
            if not item:
                continue
            parts = [part for part in re.split(r"\s*[-:,]\s*", item) if part]
            if len(parts) != 2:
                raise ValueError("Manual artifact intervals must look like 10-15; 35-42 or point to a CSV.")
            start, end = float(parts[0]), float(parts[1])
            if start < 0 or end <= start:
                raise ValueError("Manual artifact intervals need 0 <= start < end.")
            intervals.append((start, end))
        return intervals, None

    def _lfp_recording_metadata(self) -> list[dict[str, str]]:
        if not self.recordings:
            self.refresh_spike_files()
        if not self.recordings:
            raise FileNotFoundError("No ordered raw CSV recordings are available for LFP analysis.")
        default_preparation = self.lfp_preparation_id_var.get().strip()
        records = []
        for order, item in enumerate(self.recordings, start=1):
            epoch_label = str(item.get("epoch_label") or item.get("label", "")).strip()
            if not epoch_label:
                raise ValueError(
                    f"Recording segment {order} ({Path(item['path']).name}) is unlabeled. "
                    "Assign an epoch label such as Baseline, During, Treatment, Post 1, or Washout."
                )
            preparation_id = str(item.get("preparation_id") or default_preparation).strip()
            if not preparation_id:
                raise ValueError(
                    f"Recording segment {order} ({Path(item['path']).name}) needs a preparation ID. "
                    "Apply one preparation ID to all phase files from the same animal/preparation."
                )
            record = dict(item)
            record["source_file"] = item["path"]
            record["recording_order"] = str(order)
            record["segment_id"] = item.get("segment_id") or Path(item["path"]).stem
            record["recording_id"] = record["segment_id"]
            record["epoch_label"] = epoch_label
            record["label"] = epoch_label
            record["phase"] = self._phase_from_label(epoch_label)
            record["preparation_id"] = preparation_id
            record["channel"] = item.get("channel") or self.lfp_channels_var.get().strip()
            record["signal_unit"] = item.get("signal_unit") or self.lfp_signal_unit_var.get().strip() or "uV"
            if not record.get("reference_scheme"):
                record["reference_scheme"] = self.lfp_reference_scheme_var.get().strip() or "none"
            records.append(record)
        return records

    def _lfp_config(self) -> LFPConfig:
        manual_intervals, manual_csv = self._parse_manual_artifacts(self.lfp_manual_artifacts_var.get())
        notch = self._optional_float(self.lfp_notch_var.get())
        downsample = self._optional_float(self.lfp_downsample_var.get())
        clipped_abs = self._optional_float(self.lfp_clipped_abs_var.get())
        return LFPConfig(
            output_dir=self.lfp_output_dir_var.get().strip() or str(DEFAULT_DATA_DIR / "lfp_analysis"),
            channels=parse_channel_list(self.lfp_channels_var.get()),
            reference_scheme=self.lfp_reference_scheme_var.get().strip() or "none",
            analysis_window_s=float(self.lfp_window_var.get()),
            welch_segment_s=float(self.lfp_welch_segment_var.get()),
            welch_overlap=float(self.lfp_welch_overlap_var.get()),
            exploratory_band_hz=self._parse_frequency_range(self.lfp_exploratory_band_var.get(), "Exploratory band"),
            primary_band_hz=self._parse_frequency_range(self.lfp_primary_band_var.get(), "Primary band"),
            marker_bands=parse_band_spec(self.lfp_marker_bands_var.get()),
            notch_frequency_hz=notch,
            downsample_hz=downsample,
            signal_unit=self.lfp_signal_unit_var.get().strip() or "uV",
            minimum_valid_fraction=float(self.lfp_min_valid_var.get()),
            clipped_abs_uv=clipped_abs,
            large_derivative_z_threshold=float(self.lfp_derivative_z_var.get()),
            derivative_warning_fraction_threshold=float(self.lfp_derivative_warning_fraction_var.get()),
            large_derivative_fraction_threshold=float(self.lfp_derivative_fraction_var.get()),
            derivative_spike_match_tolerance_s=float(self.lfp_derivative_spike_match_ms_var.get()) / 1000.0,
            derivative_coincidence_tolerance_s=float(self.lfp_derivative_coincidence_ms_var.get()) / 1000.0,
            flatline_min_duration_s=float(self.lfp_flatline_min_duration_var.get()),
            flatline_fraction_threshold=float(self.lfp_flatline_fraction_var.get()),
            line_noise_ratio_threshold=float(self.lfp_line_noise_ratio_var.get()),
            manual_artifact_intervals=manual_intervals,
            manual_artifact_csv=manual_csv,
            peak_prominence_db=float(self.lfp_peak_prominence_var.get()),
            spike_events_csv=self.spike_events_csv_var.get().strip() or None,
            write_simple_plots=bool(self.lfp_simple_plots_var.get()),
            simple_plot_bands=parse_band_spec(self.lfp_simple_plot_bands_var.get()) if self.lfp_simple_plot_bands_var.get().strip() else None,
        )

    def _coupling_config(self) -> SpikeLFPCouplingConfig:
        phase_bands = parse_band_spec(self.lfp_phase_band_var.get())
        coupling_downsample = self._optional_float(self.lfp_coupling_downsample_var.get())
        max_workload = int(float(self.lfp_workload_limit_var.get()))
        return SpikeLFPCouplingConfig(
            output_dir=str(Path(self.lfp_output_dir_var.get().strip() or str(DEFAULT_DATA_DIR / "lfp_analysis")) / "spike_lfp_coupling"),
            lfp_channels=parse_channel_list(self.lfp_channels_var.get()),
            phase_bands=phase_bands,
            window_s=float(self.lfp_window_var.get()),
            min_spikes=int(float(self.lfp_min_spikes_var.get())),
            surrogate_count=int(float(self.lfp_surrogates_var.get())),
            lfp_feature_csv=self.lfp_feature_csv_var.get().strip() or None,
            pairing_mode="same_channel",
            processing_mode=self.lfp_coupling_processing_mode_var.get().strip() or "complex",
            explicit_channel_pairs=[],
            coupling_downsample_hz=coupling_downsample,
            max_projected_ppc=int(max_workload),
            allow_large_all_to_all=False,
        )

    def _perform_lfp_feature_extraction(self, records: list[dict], cfg: LFPConfig) -> dict[str, str]:
        return process_lfp_recordings(records, cfg)

    def _lfp_plotting_config(self, feature_csv: str) -> dict[str, object]:
        cfg = {"output_dir": str(Path(feature_csv).parent / "lfp_plots"), "plot_mode": "complex"}
        if self.lfp_simple_plot_bands_var.get().strip():
            cfg["simple_plot_bands"] = parse_band_spec(self.lfp_simple_plot_bands_var.get())
        if self.plot_input_var.get().strip():
            cfg["spike_count_csv"] = self.plot_input_var.get().strip()
        if self.spike_events_csv_var.get().strip():
            cfg["spike_events_csv"] = self.spike_events_csv_var.get().strip()
        ppc_csv = Path(self.lfp_output_dir_var.get().strip()) / "spike_lfp_coupling" / "spike_lfp_ppc_by_window.csv"
        if ppc_csv.exists():
            cfg["ppc_csv"] = str(ppc_csv)
        return cfg

    def _perform_lfp_plotting(self, feature_csv: str, cfg: Mapping) -> dict[str, list[str]]:
        return plot_lfp_features(feature_csv, cfg)

    def _resolve_lfp_feature_csv(self) -> str:
        feature_csv = self.lfp_feature_csv_var.get().strip()
        if not feature_csv:
            candidate = Path(self.lfp_output_dir_var.get().strip()) / "lfp_features_by_window.csv"
            if candidate.exists():
                feature_csv = str(candidate)
        if not feature_csv:
            raise ValueError("Choose or generate lfp_features_by_window.csv first.")
        return feature_csv

    def _perform_spike_lfp_coupling(
        self,
        spike_request: SpikePipelineRequest,
        records: list[dict],
        cfg: SpikeLFPCouplingConfig,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, str]:
        print("[spike-lfp] Regenerating spike events with the authoritative Spike Analysis pipeline.")
        spike_result = run_shared_spike_pipeline(spike_request)
        cfg.spike_processing_provenance = spike_result["spike_processing_provenance"]
        coupling_paths = calculate_spike_lfp_coupling(
            spike_result["spike_events_csv"],
            records,
            cfg,
            progress_callback=lambda message: print(message),
            cancel_event=cancel_event,
        )
        return {
            **coupling_paths,
            "spike_count_csv": spike_result["spike_count_csv"],
            "spike_events_csv": spike_result["spike_events_csv"],
            "spike_waveform_summary_csv": spike_result.get("spike_waveform_summary_csv", ""),
            "spike_diagnostic_outputs_csv": spike_result.get("spike_diagnostic_outputs_csv", ""),
            "spike_amplitude_timecourse_csv": spike_result.get("spike_amplitude_timecourse_csv", ""),
            "spike_amplitude_plot_paths": spike_result.get("spike_amplitude_plot_paths", []),
            "spike_amplitude_qc_warnings": spike_result.get("spike_amplitude_qc_warnings", []),
            "spike_processing_provenance_json": spike_result["spike_processing_provenance_json"],
        }

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

    def _perform_raw_chunk_combine(self) -> str:
        input_dir = self.raw_chunk_input_dir_var.get().strip()
        if not input_dir:
            raise ValueError("Choose a raw chunks folder first.")

        out_csv = self.raw_chunk_output_csv_var.get().strip()
        if not out_csv:
            out_csv = str(default_output_path(input_dir))

        output_path = combine_raw_chunks(
            input_dir=input_dir,
            out_csv=out_csv,
            pattern=self.raw_chunk_pattern_var.get().strip() or "chunk_*.csv",
            strict_time=bool(self.raw_chunk_strict_time_var.get()),
            write_summary=True,
        )
        return str(output_path)

    def _spike_metadata_for_paths(
        self,
        paths: list[str],
        labels: dict[str, str],
        recording_metadata: list[dict] | None = None,
    ) -> list[dict]:
        if recording_metadata is not None:
            return [dict(item) for item in recording_metadata]
        by_path = {str(item.get("path")): dict(item) for item in self.recordings}
        metadata = []
        default_preparation = self.lfp_preparation_id_var.get().strip() or "Preparation 1"
        channels = self.lfp_channels_var.get().strip()
        reference_scheme = self.lfp_reference_scheme_var.get().strip() or "none"
        signal_unit = self.lfp_signal_unit_var.get().strip() or "uV"
        for idx, path in enumerate(paths, start=1):
            item = by_path.get(str(path), {})
            label = (
                labels.get(path)
                or labels.get(str(path))
                or str(item.get("epoch_label") or item.get("label") or "").strip()
                or self._default_full_pipeline_epoch_label()
            )
            metadata.append(
                {
                    **item,
                    "source_file": path,
                    "path": path,
                    "recording_order": str(idx),
                    "segment_id": item.get("segment_id") or Path(path).stem,
                    "recording_id": item.get("recording_id") or item.get("segment_id") or Path(path).stem,
                    "epoch_label": label,
                    "label": label,
                    "phase": item.get("phase") or self._phase_from_label(label),
                    "preparation_id": item.get("preparation_id") or default_preparation,
                    "channel": item.get("channel") or channels,
                    "reference_scheme": item.get("reference_scheme") or reference_scheme,
                    "signal_unit": item.get("signal_unit") or signal_unit,
                }
            )
        return metadata

    def _spike_pipeline_request(
        self,
        paths: list[str] | None = None,
        labels: dict[str, str] | None = None,
        out_dir: str | None = None,
        recording_metadata: list[dict] | None = None,
    ) -> SpikePipelineRequest:
        if paths is None:
            if not self.recordings:
                self.refresh_spike_files()
            paths = [item["path"] for item in self.recordings]
            labels = {
                item["path"]: str(item.get("epoch_label") or item.get("label") or "").strip()
                for item in self.recordings
                if str(item.get("epoch_label") or item.get("label") or "").strip()
            }
        if not paths:
            raise FileNotFoundError("No CSV files were found for spike counting.")
        labels = dict(labels or {})
        selected_out_dir = out_dir if out_dir is not None else self.spike_out_dir_var.get().strip()
        selected_out_dir = selected_out_dir or str(Path(paths[0]).parent / "spike_counts_per_min")
        spike_events_path = str(Path(selected_out_dir) / "spike_events.csv")
        metadata = self._spike_metadata_for_paths(paths, labels, recording_metadata)
        return SpikePipelineRequest(
            paths=list(paths),
            labels_by_path=labels,
            out_dir=selected_out_dir,
            window_sec=plot_bin_seconds(self.spike_window_var.get()),
            polarity=self.spike_polarity_var.get(),
            spike_events_path=spike_events_path,
            recording_metadata=metadata,
        )

    def _perform_spike_counting(self, request: SpikePipelineRequest) -> dict:
        return run_shared_spike_pipeline(request)

    def _combine_source_paths(self) -> list[str]:
        if not self.recordings:
            self.refresh_spike_files()
        indices = self._selected_recording_indices()
        if indices:
            return [self.recordings[index]["path"] for index in indices]
        return [item["path"] for item in self.recordings]

    def _prepare_spike_count_frame_for_combine(self, path: str, source_order: int):
        import numpy as np
        import pandas as pd

        csv_path = Path(path)
        df = pd.read_csv(csv_path)
        df = standardise_combine_columns(df)

        required = ["epoch_label", "channel", "spike_count"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"{csv_path.name} does not look like a spike-count output CSV. "
                f"Missing column(s): {missing}"
            )

        df = df.copy()
        if "recording_name" not in df.columns:
            df["recording_name"] = csv_path.stem
        else:
            df["recording_name"] = df["recording_name"].astype("string").fillna("").astype(str).str.strip()
            df.loc[df["recording_name"] == "", "recording_name"] = csv_path.stem

        if "recording_index" in df.columns:
            df["source_recording_index"] = pd.to_numeric(df["recording_index"], errors="coerce")
        else:
            df["source_recording_index"] = source_order

        if "window_index" not in df.columns:
            df["window_index"] = (
                df.groupby(["channel", "epoch_label"], dropna=False)
                .cumcount()
                .astype(int)
            )

        for col in ["window_index", "window_start_s", "window_end_s", "window_duration_s", "spike_count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if df["window_index"].isna().any():
            inferred_window_index = df.groupby(["channel", "epoch_label"], dropna=False).cumcount()
            df.loc[df["window_index"].isna(), "window_index"] = inferred_window_index[df["window_index"].isna()]

        fallback_window_sec = plot_bin_seconds(self.spike_window_var.get())
        if "window_duration_s" not in df.columns and {"window_start_s", "window_end_s"}.issubset(df.columns):
            df["window_duration_s"] = df["window_end_s"] - df["window_start_s"]
        if "window_duration_s" not in df.columns:
            df["window_duration_s"] = fallback_window_sec
        df["window_duration_s"] = df["window_duration_s"].fillna(fallback_window_sec)

        if "window_start_s" not in df.columns:
            df["window_start_s"] = df["window_index"].fillna(0).astype(float) * df["window_duration_s"].fillna(fallback_window_sec)
        else:
            missing_start = df["window_start_s"].isna()
            df.loc[missing_start, "window_start_s"] = (
                df.loc[missing_start, "window_index"].fillna(0).astype(float)
                * df.loc[missing_start, "window_duration_s"].fillna(fallback_window_sec)
            )
        if "window_end_s" not in df.columns:
            df["window_end_s"] = df["window_start_s"] + df["window_duration_s"].fillna(fallback_window_sec)
        else:
            missing_end = df["window_end_s"].isna()
            df.loc[missing_end, "window_end_s"] = (
                df.loc[missing_end, "window_start_s"].fillna(0).astype(float)
                + df.loc[missing_end, "window_duration_s"].fillna(fallback_window_sec)
            )

        if "window_label" not in df.columns:
            starts_min = df["window_start_s"].fillna(0).astype(float) / 60.0
            ends_min = df["window_end_s"].fillna(0).astype(float) / 60.0
            df["window_label"] = [
                f"Min {start:g}-{end:g}"
                for start, end in zip(starts_min, ends_min)
            ]

        for col in ["recording_name", "epoch_label", "channel", "window_label"]:
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace("", pd.NA)

        df = df.dropna(subset=["epoch_label", "channel", "spike_count"]).copy()
        if df.empty:
            raise ValueError(f"{csv_path.name} has no valid spike-count rows after cleaning.")

        df["_source_file_order"] = source_order
        df["_source_row_order"] = np.arange(len(df))
        df["_epoch_phase_order"] = df["epoch_label"].map(epoch_phase_order)
        df["source_file"] = csv_path.name
        return df

    def _perform_combine_spike_count_files(self, paths: list[str] | None = None, out_dir: str | None = None) -> str:
        import pandas as pd

        selected_paths = paths or self._combine_source_paths()
        if len(selected_paths) < 2:
            raise ValueError("Choose at least two spike-count CSV files to combine.")

        frames = [
            self._prepare_spike_count_frame_for_combine(path, source_order)
            for source_order, path in enumerate(selected_paths, start=1)
        ]
        combined = pd.concat(frames, ignore_index=True, sort=False)

        for col in ["source_recording_index", "window_index", "window_start_s", "window_end_s", "window_duration_s"]:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors="coerce")

        sort_cols = [
            "channel",
            "_epoch_phase_order",
            "epoch_label",
            "_source_file_order",
            "source_recording_index",
            "recording_name",
            "window_start_s",
            "window_index",
            "_source_row_order",
        ]
        combined = combined.sort_values([col for col in sort_cols if col in combined.columns], kind="stable").reset_index(drop=True)

        recording_key_cols = ["_epoch_phase_order", "epoch_label", "_source_file_order", "source_recording_index", "recording_name"]
        recording_keys = combined[recording_key_cols].astype(str).agg("|".join, axis=1)
        combined["recording_index"] = pd.factorize(recording_keys, sort=False)[0] + 1
        combined["window_index"] = (
            combined.groupby(["channel", "recording_index", "epoch_label"], dropna=False)
            .cumcount()
            .astype(int)
        )

        output_dir = out_dir or self.spike_out_dir_var.get().strip() or str(Path(selected_paths[0]).parent)
        os.makedirs(output_dir, exist_ok=True)
        out_csv = str(Path(output_dir) / "COMBINED__spike_counts_by_epoch.csv")

        preferred_cols = [
            "recording_index",
            "recording_name",
            "epoch_label",
            "channel",
            "window_index",
            "window_start_s",
            "window_end_s",
            "window_duration_s",
            "window_label",
            "spike_count",
            "source_file",
            "source_recording_index",
        ]
        output_cols = [col for col in preferred_cols if col in combined.columns]
        output_cols.extend(
            col
            for col in combined.columns
            if col not in output_cols and not str(col).startswith("_")
        )
        combined[output_cols].to_csv(out_csv, index=False)
        print(f"[combine] wrote {len(combined)} row(s) from {len(selected_paths)} file(s):\n  {out_csv}")
        return out_csv

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
        middle_phase_split = self.middle_phase_split_var.get().strip().lower() or "whole"
        post_phase_split = self.post_phase_split_var.get().strip().lower() or "whole"
        df = module.add_middle_phase_segment_labels(df, id_columns, group_column, middle_phase_split)
        df = module.add_post_phase_segment_labels(df, id_columns, group_column, post_phase_split)
        df = module.add_baseline_half_labels(df, id_columns, group_column)
        baseline_change = module.build_baseline_half_table(df, id_columns, group_column)
        part_table = module.build_experiment_part_table(
            df,
            baseline_change,
            id_columns,
            group_column,
            middle_phase_split=middle_phase_split,
            post_phase_split=post_phase_split,
        )
        if part_table.empty:
            return []

        phase_specs = module.phase_specs_for_comparisons(
            comparison_names,
            middle_phase_split=middle_phase_split,
            post_phase_split=post_phase_split,
        )
        specs: list[dict[str, str]] = []
        for (part_index, stimulation_label, part_label), part in part_table.groupby(
            ["part_index", "stimulation_label", "part_label"],
            dropna=False,
            sort=True,
        ):
            has_plot_value = False
            for phase_spec in phase_specs:
                percent_col = phase_spec["percent_col"]
                if percent_col in part.columns and part[percent_col].notna().any():
                    has_plot_value = True
                    break
            if not has_plot_value:
                continue
            index = int(part_index)
            default_title = self._plot_title_for_epoch_label(stimulation_label)
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
        dialog.configure(bg=STEP_COLORS["surface"])
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.columnconfigure(0, weight=1)
        if APP_ICON.exists():
            with contextlib.suppress(tk.TclError):
                dialog.iconbitmap(str(APP_ICON))

        ttk.Label(
            dialog,
            text="Set the title for each non-pooled percent-change plot.",
            style="Stage.TLabel",
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
            "--middle-phase-split",
            self.middle_phase_split_var.get().strip().lower() or "whole",
            "--post-phase-split",
            self.post_phase_split_var.get().strip().lower() or "whole",
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
        if not self.percent_show_channel_labels_var.get():
            cmd.append("--no-channel-labels")
        perfusion_title = self._perfusion_title()
        perfusion_label = self._perfusion_legend_label()
        if perfusion_title:
            cmd.extend(["--perfusion-title", perfusion_title])
        if perfusion_label:
            cmd.extend(["--perfusion-label", perfusion_label])
        epoch_titles = self._epoch_title_mapping()
        if epoch_titles:
            cmd.extend(["--epoch-titles", json.dumps(epoch_titles)])
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
            self.continuous_plot_title_var.get().strip(),
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
        perfusion_title = self._perfusion_title()
        perfusion_label = self._perfusion_legend_label()
        if perfusion_title:
            cmd.extend(["--perfusion-title", perfusion_title])
        if perfusion_label:
            cmd.extend(["--perfusion-label", perfusion_label])
        epoch_titles = self._epoch_title_mapping()
        if epoch_titles:
            cmd.extend(["--epoch-titles", json.dumps(epoch_titles)])
        if not self.continuous_include_legend_var.get():
            cmd.append("--no-legend")
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

    def run_raw_chunk_combine(self) -> None:
        def task():
            return self._perform_raw_chunk_combine()

        def success(path):
            output_path = Path(path)
            self.raw_chunk_output_csv_var.set(str(output_path))
            self.log_queue.put(f"Combined raw chunks ready:\n  {output_path}\n")
            self.log_queue.put(f"Summary:\n  {output_path.with_suffix(output_path.suffix + '.summary.txt')}\n")

        self.run_background("Combine raw chunks", task, on_success=success)

    def run_spike_counting(self) -> None:
        if not self.recordings:
            self.refresh_spike_files()
        if not self.recordings:
            messagebox.showerror("No CSV files", "No CSV files were found for spike counting.")
            return
        try:
            request = self._spike_pipeline_request()
        except Exception as exc:
            messagebox.showerror("Spike counting settings", str(exc))
            return

        def task():
            return self._perform_spike_counting(request)

        def success(result):
            if not result:
                return
            path = result["spike_count_csv"]
            self.plot_input_var.set(str(path))
            self.spike_events_csv_var.set(str(result.get("spike_events_csv", "")))
            if result.get("spike_waveform_summary_csv"):
                self.log_queue.put(f"Spike waveform summary:\n  {result.get('spike_waveform_summary_csv')}\n")
            if result.get("spike_amplitude_timecourse_csv"):
                self.log_queue.put(f"Spike amplitude time-course:\n  {result.get('spike_amplitude_timecourse_csv')}\n")
                if result.get("spike_amplitude_waveforms_csv"):
                    self.log_queue.put(f"Spike amplitude waveforms:\n  {result.get('spike_amplitude_waveforms_csv')}\n")
                self.log_queue.put("Use 'Spike amplitude change' in the Comparison and plotting tab to make figures.\n")
            for warning in result.get("spike_amplitude_qc_warnings") or []:
                self.log_queue.put(f"[spike amplitude warning] {warning}\n")
            if not self.plot_out_dir_var.get().strip():
                spike_path = Path(path)
                self.plot_out_dir_var.set(str(spike_path.with_name(f"{spike_path.stem}_baseline_post_plots")))

        self.run_background("Spike counting", task, on_success=success)

    def run_combine_spike_count_files(self) -> None:
        if not self.recordings:
            self.refresh_spike_files()
        paths = self._combine_source_paths()
        if len(paths) < 2:
            messagebox.showerror("Not enough files", "Choose at least two spike-count CSV files to combine.")
            return

        def task():
            return self._perform_combine_spike_count_files(paths=paths)

        def success(path):
            self.plot_input_var.set(str(path))
            spike_path = Path(path)
            if not self.plot_out_dir_var.get().strip():
                self.plot_out_dir_var.set(str(spike_path.with_name(f"{spike_path.stem}_baseline_post_plots")))

        self.run_background("Combine spike-count files", task, on_success=success)

    def run_lfp_feature_extraction(self) -> None:
        try:
            cfg = self._lfp_config()
            records = self._lfp_recording_metadata()
        except Exception as exc:
            messagebox.showerror("LFP settings", str(exc))
            return

        def task():
            return self._perform_lfp_feature_extraction(records, cfg)

        def success(paths):
            if not paths:
                return
            self.lfp_feature_csv_var.set(str(paths.get("features_csv", "")))
            simple_dir = paths.get("simple_lfp_plots_dir", "")
            simple_line = f"\nSimple LFP plots:\n  {simple_dir}" if simple_dir and self.lfp_simple_plots_var.get() else ""
            self.log_queue.put(f"LFP features ready:\n  {paths.get('features_csv')}{simple_line}\n")

        self.run_background("LFP feature extraction", task, on_success=success)

    def run_lfp_plotting(self) -> None:
        try:
            feature_csv = self._resolve_lfp_feature_csv()
            cfg = self._lfp_plotting_config(feature_csv)
        except Exception as exc:
            messagebox.showerror("LFP plotting", str(exc))
            return

        def task():
            return self._perform_lfp_plotting(feature_csv, cfg)

        self.run_background("LFP comparison plotting", task)

    def run_spike_lfp_coupling(self) -> None:
        try:
            records = self._lfp_recording_metadata()
            cfg = self._coupling_config()
            raw_paths = [str(record["source_file"]) for record in records]
            labels_by_path = {str(record["source_file"]): str(record.get("epoch_label") or record.get("label") or "") for record in records}
            spike_request = self._spike_pipeline_request(
                paths=raw_paths,
                labels=labels_by_path,
                out_dir=self.spike_out_dir_var.get().strip(),
                recording_metadata=records,
            )
        except Exception as exc:
            messagebox.showerror("Spike-LFP coupling settings", str(exc))
            return
        cancel_event = threading.Event()

        def task():
            return self._perform_spike_lfp_coupling(spike_request, records, cfg, cancel_event=cancel_event)

        def success(paths):
            if not paths:
                return
            self.plot_input_var.set(str(paths.get("spike_count_csv", "")))
            self.spike_events_csv_var.set(str(paths.get("spike_events_csv", "")))
            primary = paths.get("primary_ppc_plot")
            rate_power = paths.get("primary_rate_power_plot")
            primary_line = f"\nPrimary PPC figure:\n  {primary}" if primary else ""
            rate_power_line = f"\nSpike rate versus LFP power figure:\n  {rate_power}" if rate_power else ""
            waveform_line = f"\nSpike waveform summary:\n  {paths.get('spike_waveform_summary_csv')}" if paths.get("spike_waveform_summary_csv") else ""
            amplitude_line = f"\nSpike amplitude time-course:\n  {paths.get('spike_amplitude_timecourse_csv')}" if paths.get("spike_amplitude_timecourse_csv") else ""
            sta_line = f"\nSpike-triggered LFP STA summary:\n  {paths.get('sta_summary_csv')}" if cfg.processing_mode == "complex" and paths.get("sta_summary_csv") else ""
            self.log_queue.put(f"Spike-LFP PPC ready:\n  {paths.get('ppc_csv')}{rate_power_line}{primary_line}{waveform_line}{amplitude_line}{sta_line}\n")
            for warning in paths.get("spike_amplitude_qc_warnings") or []:
                self.log_queue.put(f"[spike amplitude warning] {warning}\n")

        self.run_background("Spike-LFP coupling", task, on_success=success, cancel_event=cancel_event)

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

    def run_amplitude_change_plot(self) -> None:
        input_path = self.plot_input_var.get().strip()
        if not input_path:
            messagebox.showerror("Missing input", "Choose the spike-amplitude time-course CSV first.")
            return
        sibling_amp_csv = str(Path(input_path).with_name("spike_amplitude_time_course.csv"))
        if Path(sibling_amp_csv).exists() and Path(input_path).name != "spike_amplitude_time_course.csv":
            input_path = sibling_amp_csv

        def task():
            return self._perform_amplitude_change_plot(input_path)

        self.run_background("Spike amplitude change plotting", task)

    def _perform_amplitude_change_plot(self, input_path: str) -> str:
        import matplotlib

        matplotlib.use("Agg", force=True)
        module = load_script_module(SPIKE_SCRIPT, f"spike_replot_amplitude_{threading.get_ident()}")
        waveform_csv = str(Path(input_path).with_name("spike_amplitude_waveforms.csv"))
        if not Path(waveform_csv).exists():
            waveform_csv = None
        out_dir = self.plot_out_dir_var.get().strip() or str(Path(input_path).parent)
        results = module.replot_spike_amplitude_timecourse(
            input_path,
            waveform_csv=waveform_csv,
            out_dir=out_dir,
            boundary_label=self.amp_boundary_label_var.get().strip() or "H2O2 treatment - not recorded",
            overlay=bool(self.amp_overlay_var.get()),
            channels=None,
            polarity=self.amp_polarity_var.get().strip() or "both",
        )
        for plot_path in results.get("plot_paths", []):
            self.log_queue.put(f"Spike amplitude change figure:\n  {plot_path}\n")
        for warning in results.get("warnings", []):
            self.log_queue.put(f"[spike amplitude warning] {warning}\n")
        if not results.get("plot_paths"):
            self.log_queue.put("[spike amplitude] No amplitude figures produced.\n")

        spike_events_csv = str(Path(input_path).with_name("spike_events.csv"))
        if Path(spike_events_csv).exists():
            plot_types = tuple(
                key
                for key in ("raster", "rate", "isi", "acg", "waveform")
                if getattr(self, f"amp_plot_{key}_var").get()
            )
            if plot_types:
                diag_results = module.replot_spike_diagnostic_outputs(
                    spike_events_csv,
                    out_dir=out_dir,
                    plot_types=plot_types,
                    channels=None,
                )
                for plot_path in diag_results.get("plot_paths", []):
                    self.log_queue.put(f"Spike diagnostic figure:\n  {plot_path}\n")
                for warning in diag_results.get("warnings", []):
                    self.log_queue.put(f"[spike diagnostics warning] {warning}\n")
        return out_dir

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
            self.log_queue.put(
                "Full pipeline sequence: raw conversion -> spike counts/events -> LFP feature extraction -> "
                "spike/LFP coupling -> LFP plots/statistics -> spike plots\n"
            )

            try:
                lfp_records = self._lfp_recording_metadata()
                raw_paths = [record["source_file"] for record in lfp_records]
                self.log_queue.put(f"\n[full pipeline] Using {len(raw_paths)} ordered recording segment CSV file(s).\n")
            except Exception:
                raw_csv = self._perform_raw_conversion()
                self.log_queue.put(f"\n[full pipeline] Raw CSV ready:\n  {raw_csv}\n")
                spike_label = self._default_full_pipeline_epoch_label()
                raw_paths = [raw_csv]
                lfp_records = [
                    {
                        "path": raw_csv,
                        "source_file": raw_csv,
                        "segment_id": Path(raw_csv).stem,
                        "recording_id": Path(raw_csv).stem,
                        "recording_order": "1",
                        "label": spike_label,
                        "epoch_label": spike_label,
                        "phase": self._phase_from_label(spike_label) or "baseline",
                        "preparation_id": self.lfp_preparation_id_var.get().strip() or "Preparation 1",
                        "reference_scheme": self.lfp_reference_scheme_var.get().strip() or "none",
                        "channel": self.lfp_channels_var.get().strip(),
                        "signal_unit": self.lfp_signal_unit_var.get().strip() or "uV",
                    }
                ]

            labels_by_path = {record["source_file"]: record.get("epoch_label", record.get("label", "")) for record in lfp_records}
            spike_request = self._spike_pipeline_request(
                paths=raw_paths,
                labels=labels_by_path,
                out_dir=self.spike_out_dir_var.get().strip(),
                recording_metadata=lfp_records,
            )
            spike_result = self._perform_spike_counting(spike_request)
            spike_csv = spike_result["spike_count_csv"]
            self.log_queue.put(f"\n[full pipeline] Spike-count CSV ready:\n  {spike_csv}\n")
            self.log_queue.put(f"[full pipeline] Spike waveform summary:\n  {spike_result.get('spike_waveform_summary_csv')}\n")
            if spike_result.get("spike_amplitude_timecourse_csv"):
                self.log_queue.put(f"[full pipeline] Spike amplitude time-course:\n  {spike_result.get('spike_amplitude_timecourse_csv')}\n")
            for warning in spike_result.get("spike_amplitude_qc_warnings") or []:
                self.log_queue.put(f"[spike amplitude warning] {warning}\n")
            events_csv = spike_result["spike_events_csv"]

            lfp_paths = process_lfp_recordings(lfp_records, self._lfp_config())
            self.log_queue.put(f"\n[full pipeline] LFP feature CSV ready:\n  {lfp_paths['features_csv']}\n")

            coupling_cfg = self._coupling_config()
            coupling_cfg.spike_processing_provenance = spike_result["spike_processing_provenance"]
            coupling_paths = calculate_spike_lfp_coupling(
                events_csv,
                lfp_records,
                coupling_cfg,
            )
            self.log_queue.put(f"\n[full pipeline] Spike-LFP PPC ready:\n  {coupling_paths['ppc_csv']}\n")
            if coupling_cfg.processing_mode == "complex":
                self.log_queue.put(f"[full pipeline] Spike-triggered LFP STA summary:\n  {coupling_paths.get('sta_summary_csv')}\n")
            if coupling_paths.get("primary_rate_power_plot"):
                self.log_queue.put(f"[full pipeline] Spike rate versus LFP power figure:\n  {coupling_paths['primary_rate_power_plot']}\n")
            if coupling_paths.get("primary_ppc_plot"):
                self.log_queue.put(f"[full pipeline] Primary PPC figure:\n  {coupling_paths['primary_ppc_plot']}\n")

            lfp_plot_outputs = plot_lfp_features(
                lfp_paths["features_csv"],
                {
                    "output_dir": str(Path(lfp_paths["features_csv"]).parent / "lfp_plots"),
                    "spike_count_csv": spike_csv,
                    "spike_events_csv": events_csv,
                    "ppc_csv": coupling_paths["ppc_csv"],
                    "simple_plot_bands": parse_band_spec(self.lfp_simple_plot_bands_var.get()) if self.lfp_simple_plot_bands_var.get().strip() else None,
                },
            )
            for plot_path in lfp_plot_outputs.get("plots", []):
                if Path(plot_path).name.startswith("PRIMARY_spike_lfp_aligned"):
                    self.log_queue.put(f"[full pipeline] Primary aligned spike/LFP figure:\n  {plot_path}\n")

            plot_phases = self.selected_recorded_phases()
            plot_comparisons = self.selected_comparisons()
            spike_label = next(iter(labels_by_path.values()), "")
            if len(raw_paths) == 1 and spike_label.lower().startswith("baseline") and (
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
                "raw_csv": raw_paths[0],
                "spike_csv": spike_csv,
                "spike_events_csv": events_csv,
                "lfp_features_csv": lfp_paths["features_csv"],
                "lfp_plot_outputs": lfp_plot_outputs,
                "spike_lfp_ppc_csv": coupling_paths["ppc_csv"],
                "spike_lfp_sta_summary_csv": coupling_paths.get("sta_summary_csv", ""),
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
            self.spike_events_csv_var.set(str(result.get("spike_events_csv", "")))
            self.lfp_feature_csv_var.set(str(result.get("lfp_features_csv", "")))
            self.plot_out_dir_var.set(str(result["baseline_plot_dir"]))

        self.run_background("Full pipeline", task, on_success=success)


def main() -> None:
    root = tk.Tk()
    app = LocustPipelineApp(root)
    app.refresh_spike_files()
    root.mainloop()


if __name__ == "__main__":
    main()
