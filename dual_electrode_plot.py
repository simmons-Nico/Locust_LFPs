#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-channel spike count plot with section highlighting.

Plots multiple channels in vertically stacked subplots (one column).
Sections (epochs) are highlighted with consistent colors across channels.
Channels that are missing a section show blank space for that section.

Expected input CSV columns (per channel file):
    recording_index, recording_name, epoch_label, channel, window_index, spike_count
    (optional: window_start_s, window_end_s)

Section alignment is position-based: section N in channel A aligns with
section N in channel B. Section order is chronological (by recording_index
then window_index within each channel). The unified section order across
all channels is determined by first appearance.
"""

import os
import re
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


#C:\Users\locadmin\AppData\Local\Programs\Python\Python313\python.exe c:/Users/locadmin/Documents/CEITEC/Locust_LFPs/dual_electrode_plot.py ""C:\Users\locadmin\Documents\CEITEC\Output\all\A-000_uV__spike_counts_per_min.csv"" ""C:\Users\locadmin\Documents\CEITEC\Output\all\A-001_uV__spike_counts_per_min.csv"" --channel-labels "150um Main" "Stainless steel Auxillary" --out-dir "C:\Users\locadmin\Documents\CEITEC\Output\all" --title "Double electrode - Galvanostatic (24/07/2026)"
#C:\Users\locadmin\AppData\Local\Programs\Python\Python313\python.exe c:/Users/locadmin/Documents/CEITEC/Locust_LFPs/dual_electrode_plot.py "C:\Users\locadmin\Documents\CEITEC\Output\all\A-005_uV__spike_counts_per_min.csv" "C:\Users\locadmin\Documents\CEITEC\Output\all\A-008_uV__spike_counts_per_min.csv" "C:\Users\locadmin\Documents\CEITEC\Output\all\A-009_uV__spike_counts_per_min.csv" --channel-labels "A-005" "A-008" "A-009" --out-dir "C:\Users\locadmin\Documents\CEITEC\Output\all" --title "ATLAS probe - 1mM drip @ 2,5ul/min (30/07/2026)"
# =========================
# USER SETTINGS
# =========================
EPOCH_COLORS = [
    "#929292FF", "#FFE4D6", "#8BD0F8", "#B6FCAF", "#E1C1FD",
    "#FFEFAA", "#FFBCBC", "#FFFD9F",
]

DEFAULT_PLOT_TITLE = "Multi-Channel Spike Counts"

TRACE_COLOR = "tab:blue"
TRACE_MARKER = "o"
TRACE_LINEWIDTH = 2

FIG_WIDTH = 16.0
FIG_SUBPLOT_HEIGHT = 4.0
FIG_DPI = 200

Y_LIM_TOP = None
MAX_X_TICKS = 24
PLOT_BIN_SEC = 60.0

BOUNDARY_DESCRIPTION = ""


# =========================
# HELPERS
# =========================

def clean_epoch_name(name):
    s = str(name).strip()
    low = s.lower()
    if "baseline" in low:
        return "Baseline"
    if "h2o2" in low:
        return "H2O2 at 0.25 Atm"
    match = re.search(r'(-?\d+)\s*n[aA]\b', low)
    if match:
        current_val = int(match.group(1))
        cleaned_current = f"\n-{abs(current_val)} nA"
        cleaned = re.sub(r'-?\d+\s*n[aA]\b', cleaned_current, s, flags=re.IGNORECASE)
        return cleaned
    return s


def build_epoch_color_map(epoch_labels):
    unique = list(dict.fromkeys(epoch_labels))
    return {ep: EPOCH_COLORS[i % len(EPOCH_COLORS)] for i, ep in enumerate(unique)}


def format_bin_label(bin_sec):
    if bin_sec < 60:
        return f"{bin_sec:g} sec"
    return f"{bin_sec / 60.0:g} min"


# =========================
# DATA LOADING
# =========================

def load_channel_csv(path):
    df = pd.read_csv(path, skip_blank_lines=False)

    required = ["epoch_label", "window_index", "spike_count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {os.path.basename(path)}: {missing}"
        )

    for col in ["epoch_label", "channel"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df["window_index"] = pd.to_numeric(df["window_index"], errors="coerce")
    df["spike_count"] = pd.to_numeric(df["spike_count"], errors="coerce")
    df["epoch_label_clean"] = df["epoch_label"].apply(clean_epoch_name)

    if "recording_index" in df.columns:
        df["recording_index"] = pd.to_numeric(df["recording_index"], errors="coerce")
    else:
        df["recording_index"] = 1

    if "window_start_s" in df.columns:
        df["window_start_s"] = pd.to_numeric(df["window_start_s"], errors="coerce")
    if "window_end_s" in df.columns:
        df["window_end_s"] = pd.to_numeric(df["window_end_s"], errors="coerce")

    df = df.dropna(subset=["window_index", "spike_count", "epoch_label_clean"])
    df = df.sort_values(
        ["recording_index", "window_index"], kind="stable"
    ).reset_index(drop=True)

    return df


# =========================
# SECTION EXTRACTION
# =========================

def extract_ordered_sections(df):
    """Extract ordered sections from a channel DataFrame.

    A new section starts whenever the epoch_label or recording_index changes.
    Returns a list of dicts: [{label, windows DataFrame}, ...]
    """
    section_id = (
        (df["epoch_label_clean"] != df["epoch_label_clean"].shift())
        | (df["recording_index"] != df["recording_index"].shift())
    ).cumsum()

    sections = []
    for _, block in df.groupby(section_id, sort=False):
        block = block.sort_values("window_index", kind="stable")
        label = block["epoch_label_clean"].iloc[0]
        sections.append({"label": label, "windows": block.reset_index(drop=True)})

    return sections


def build_unified_section_order(all_channel_sections):
    """Build a unified section label list using the channel with the most
    unique sections as the reference.  Other channels are aligned to this
    order with blank gaps for missing sections."""
    best_idx = 0
    best_count = 0
    for i, channel_sections in enumerate(all_channel_sections):
        unique_labels = list(dict.fromkeys(s["label"] for s in channel_sections))
        if len(unique_labels) > best_count:
            best_count = len(unique_labels)
            best_idx = i

    seen = list(dict.fromkeys(
        s["label"] for s in all_channel_sections[best_idx]
    ))

    for channel_sections in all_channel_sections:
        for sec in channel_sections:
            if sec["label"] not in seen:
                seen.append(sec["label"])

    # Baseline must always be first
    if "Baseline" in seen:
        seen = ["Baseline"] + [label for label in seen if label != "Baseline"]

    return seen


def compute_section_widths(all_channel_sections, unified_labels):
    """For each unified section label, compute the max window count across channels."""
    widths = {}
    for label in unified_labels:
        max_count = 0
        for channel_sections in all_channel_sections:
            for sec in channel_sections:
                if sec["label"] == label:
                    max_count = max(max_count, len(sec["windows"]))
        widths[label] = max(max_count, 1)
    return widths


def map_sections_to_x_positions(channel_sections, unified_labels, section_widths):
    """Map a channel's sections to global x-axis positions.

    Each unified section gets a band of width section_widths[label].
    Data is left-aligned within the band.  Missing sections produce no data
    points (blank space).

    Returns
    -------
    x_data, y_data : np.ndarray
        Plottable x/y coordinates.
    section_spans : list of (label, x_start, x_end)
        The x-extent of every section band (for shading).
    """
    x_data = []
    y_data = []
    section_spans = []

    x_offset = 0
    for label in unified_labels:
        width = section_widths[label]
        section_spans.append((label, x_offset, x_offset + width))

        for sec in channel_sections:
            if sec["label"] == label:
                n = len(sec["windows"])
                x_positions = np.arange(n, dtype=float) + x_offset
                y_values = sec["windows"]["spike_count"].to_numpy(dtype=float)
                x_data.extend(x_positions.tolist())
                y_data.extend(y_values.tolist())
                break

        x_offset += width

    return np.array(x_data), np.array(y_data), section_spans

# =========================
# PLOTTING
# =========================

def plot_multi_channel(
    channels,
    unified_labels,
    section_widths,
    color_map,
    title=DEFAULT_PLOT_TITLE,
    trace_color=TRACE_COLOR,
    trace_marker=TRACE_MARKER,
    trace_linewidth=TRACE_LINEWIDTH,
    fig_width=FIG_WIDTH,
    subplot_height=FIG_SUBPLOT_HEIGHT,
    dpi=FIG_DPI,
    y_lim_top=Y_LIM_TOP,
    max_x_ticks=MAX_X_TICKS,
    boundary_description=BOUNDARY_DESCRIPTION,
):
    n_channels = len(channels)
    global_x_max = sum(section_widths.values())

    fig_height = subplot_height * n_channels
    fig, axes = plt.subplots(
        n_channels, 1,
        figsize=(fig_width, fig_height),
        sharex=True,
        squeeze=False,
    )
    axes = axes.flatten()

    trace_marker = (
        None
        if str(trace_marker).strip().lower() in {"", "none", "no", "off"}
        else trace_marker
    )

    legend_handles = []
    seen_labels = set()

    for i, (ax, ch) in enumerate(zip(axes, channels)):
        x_data, y_data, section_spans = map_sections_to_x_positions(
            ch["sections"], unified_labels, section_widths,
        )

        # --- section shading ---
        for label, x_start, x_end in section_spans:
            color = color_map[label]
            ax.axvspan(
                x_start - 0.5, x_end - 0.5,
                color=color, alpha=0.35, zorder=0,
            )
            if label not in seen_labels:
                '''legend_handles.append(
                    Patch(facecolor=color, edgecolor="none", alpha=0.35, label=label)
                )'''
                seen_labels.add(label)

        # --- data trace (split per section so gaps stay blank) ---
        for label, x_start, x_end in section_spans:
            mask = (x_data >= x_start) & (x_data < x_end)
            if mask.any():
                ax.plot(
                    x_data[mask],
                    y_data[mask],
                    marker=trace_marker,
                    linewidth=trace_linewidth,
                    color=trace_color,
                    zorder=3,
                )

        # --- section boundary lines ---
        x_pos = 0
        for label in unified_labels:
            width = section_widths[label]
            if x_pos > 0:
                ax.axvline(
                    x_pos - 0.5,
                    linestyle="--", color="black", alpha=0.7, zorder=2,
                )
            x_pos += width

        # --- section text labels (skip Baseline / Post for cleaner look) ---
        '''x_pos = 0
        for label in unified_labels:
            width = section_widths[label]
            midpoint = x_pos + width / 2.0
            y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
            ax.text(
                midpoint, y_top * 0.94, label,
                ha="center", va="top", fontsize=12,
                fontweight="bold", color="black", zorder=4,
            )
            x_pos += width'''

        # --- y-axis ---
        ax.set_ylabel(f"{ch['label']}\nSpike count", rotation=90, labelpad=10)

        if y_lim_top is not None:
            ax.set_ylim(0, y_lim_top)
        elif len(y_data) > 0:
            ax.set_ylim(0, max(1, float(np.nanmax(y_data)) * 1.15))
        else:
            ax.set_ylim(0, 1)

        if i < n_channels - 1:
            plt.setp(ax.get_xticklabels(), visible=False)

    # --- x-axis tick labels (section names, centered) ---
    bottom_ax = axes[-1]
    tick_positions = []
    tick_labels_text = []
    x_pos = 0
    for label in unified_labels:
        width = section_widths[label]
        tick_positions.append(x_pos + width / 2.0)
        tick_labels_text.append(label)
        x_pos += width

    step = max(1, len(tick_positions) // max_x_ticks) if tick_positions else 1
    bottom_ax.set_xticks(tick_positions[::step])
    bottom_ax.set_xticklabels(tick_labels_text[::step], rotation=0)
    bottom_ax.set_xlim(-0.5, global_x_max - 0.5)
    bottom_ax.set_xlabel("Section [20 min each]")

    # --- shared legend ---
    if boundary_description:
        legend_handles.append(
            Line2D(
                [0], [0], color="black", linestyle="--",
                alpha=0.7, label=boundary_description,
            )
        )
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=min(len(legend_handles), 4),
            frameon=False,
        )

    fig.suptitle(title, fontsize=20, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 1])

    return fig, axes

# =========================
# CLI
# =========================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Multi-channel spike count plot with section highlighting.",
    )
    parser.add_argument(
        "csv_files", nargs="+",
        help="One CSV file per channel to plot.",
    )
    parser.add_argument(
        "--title", default=DEFAULT_PLOT_TITLE, help="Figure title.",
    )
    parser.add_argument(
        "--out-dir", default="",
        help="Output directory (default: directory of first CSV).",
    )
    parser.add_argument(
        "--out-name", default="multi_channel_spike_plot.png",
        help="Output filename.",
    )
    parser.add_argument(
        "--trace-color", default=TRACE_COLOR,
        help="Matplotlib color for traces.",
    )
    parser.add_argument(
        "--trace-marker", default=TRACE_MARKER,
        help="Marker style (use 'None' for none).",
    )
    parser.add_argument(
        "--trace-linewidth", type=float, default=TRACE_LINEWIDTH,
    )
    parser.add_argument("--fig-width", type=float, default=FIG_WIDTH)
    parser.add_argument("--subplot-height", type=float, default=FIG_SUBPLOT_HEIGHT)
    parser.add_argument("--dpi", type=int, default=FIG_DPI)
    parser.add_argument(
        "--y-lim-top", type=float, default=Y_LIM_TOP,
        help="Fixed y-axis max (None = auto).",
    )
    parser.add_argument("--max-x-ticks", type=int, default=MAX_X_TICKS)
    parser.add_argument(
        "--boundary-description", default=BOUNDARY_DESCRIPTION,
        help="Legend label for dashed boundary lines. Blank = hidden.",
    )
    parser.add_argument(
        "--channel-labels", nargs="+", default=None,
        help="Custom label for each channel (must match number of CSV files).",
    )
    parser.add_argument(
        "--plot-bin-sec", type=float, default=PLOT_BIN_SEC,
        help="Spike-count bin size in seconds (for axis labels).",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
    })

    csv_paths = args.csv_files
    n_channels = len(csv_paths)

    if args.channel_labels and len(args.channel_labels) != n_channels:
        raise ValueError(
            f"Number of channel labels ({len(args.channel_labels)}) "
            f"must match number of CSV files ({n_channels})."
        )

    # --- load all channels ---
    channels = []
    for i, path in enumerate(csv_paths):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        df = load_channel_csv(path)
        sections = extract_ordered_sections(df)
        label = (
            args.channel_labels[i]
            if args.channel_labels
            else os.path.splitext(os.path.basename(path))[0]
        )
        channels.append({
            "path": path, "df": df,
            "sections": sections, "label": label,
        })
        n_windows = len(df)
        sec_summary = ", ".join(
            f"{s['label']}({len(s['windows'])})" for s in sections
        )
        print(f"  {label}: {n_windows} windows, sections -> [{sec_summary}]")

    # --- build unified section order ---
    all_section_lists = [ch["sections"] for ch in channels]
    unified_labels = build_unified_section_order(all_section_lists)
    print(f"Unified section order: {unified_labels}")

    # --- compute section widths ---
    section_widths = compute_section_widths(all_section_lists, unified_labels)
    bin_label = format_bin_label(args.plot_bin_sec)
    for label in unified_labels:
        print(f"  Section '{label}': max {section_widths[label]} windows")

    # --- build color map ---
    color_map = build_epoch_color_map(unified_labels)

    # --- plot ---
    fig, axes = plot_multi_channel(
        channels,
        unified_labels,
        section_widths,
        color_map,
        title=args.title,
        trace_color=args.trace_color,
        trace_marker=args.trace_marker,
        trace_linewidth=args.trace_linewidth,
        fig_width=args.fig_width,
        subplot_height=args.subplot_height,
        dpi=args.dpi,
        y_lim_top=args.y_lim_top,
        max_x_ticks=args.max_x_ticks,
        boundary_description=args.boundary_description,
    )

    # --- save ---
    out_dir = args.out_dir
    if not out_dir:
        out_dir = os.path.dirname(os.path.abspath(csv_paths[0]))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out_name)
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
