#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spike count per 1-minute window (MULTIPLE CSVs, continuous time per channel)

What this script does
---------------------
- Loads multiple CSVs from CSV_DIR matching CSV_GLOB
- Prompts you to choose the processing order of the CSV recordings
- Ignores marker / digital / stimulation columns
- Processes spike counts exactly like the original one-CSV version:
    * band-pass filter
    * MAD z-threshold spike detection
    * polarity-aware peak detection
    * refractory enforcement
    * optional amplitude / width gating
    * per-minute spike counts
- Saves:
    1) one combined per-window spike count CSV for all recordings, without marker data
    2) one plot per channel, where recordings from the SAME channel are concatenated
       in your chosen order so the x-axis represents continuous time

Important change
----------------
If channel A-000 exists in multiple recordings, its spike counts are shown on ONE plot
as one continuous trace in time, in the order you specify.

"""

import os
import glob
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks
from matplotlib.lines import Line2D


# =========================
# USER SETTINGS
# =========================

CSV_DIR  = r"C:\Users\simmons\Desktop\Exploring PSDs"
CSV_GLOB = "*.csv"

OUT_DIR_NAME = "spike_counts_per_min"

TIME_COL = "time_s"

HP_SPIKE_BAND = (300.0, 5000.0)
SPIKE_Z_THR = 6.0
POLARITY = "both"   # "neg", "pos", "both"
REFRACTORY_MS = 1.0

AMP_MIN_UV = 40.0
AMP_MAX_UV = 700.0

W_MIN_MS = None
W_MAX_MS = None

WINDOW_MIN = 1.0
WINDOW_SEC = WINDOW_MIN * 60.0

# Columns containing any of these tokens are treated as marker/event data and skipped.
MARKER_COLUMN_TOKENS = (
    "marker",
    "ttl",
    "trigger",
    "event",
    "digital",
    "dig",
    "din",
    "dout",
    "stim",
    "sync",
)

TRACE_MARKER = "o"
TRACE_LINEWIDTH = 2.0
TRACE_COLOR = "tab:blue"


def format_window_label(window_sec: float) -> str:
    if window_sec < 60:
        return f"{window_sec:g}_sec"
    return f"{window_sec / 60.0:g}_min"


def format_window_axis_value(window_index: int, window_sec: float) -> str:
    minutes = window_index * (window_sec / 60.0)
    return f"{minutes:g}"


def folded_window_bounds(
    t0: float,
    t1: float,
    window_sec: float,
    sample_interval_s: float = 0.0,
) -> list[tuple[float, float]]:
    """Build windows while folding a trailing partial window into the previous one."""
    total_dur = max(0.0, float(t1) - float(t0) + max(0.0, float(sample_interval_s)))
    if total_dur <= 0:
        return []

    tolerance = max(1e-9, float(window_sec) * 1e-9)
    full_windows = int(math.floor(total_dur / float(window_sec)))
    exact_multiple = (
        full_windows > 0
        and math.isclose(total_dur, full_windows * float(window_sec), rel_tol=0.0, abs_tol=tolerance)
    )
    n_windows = full_windows if exact_multiple else max(1, full_windows)

    bounds: list[tuple[float, float]] = []
    for window_index in range(n_windows):
        start_s = float(t0) + window_index * float(window_sec)
        if window_index == n_windows - 1:
            end_s = float(t0) + total_dur
        else:
            end_s = float(t0) + (window_index + 1) * float(window_sec)
        bounds.append((start_s, end_s))
    return bounds


# =========================
# SIGNAL HELPERS
# =========================

def butter_bandpass(lo, hi, fs, order=3):
    ny = 0.5 * fs
    lo = max(lo / ny, 1e-6)
    hi = min(hi / ny, 0.999999)
    b, a = butter(order, [lo, hi], btype="bandpass")
    return b, a


def bandpass_filt(x, fs, band, order=3):
    b, a = butter_bandpass(band[0], band[1], fs, order=order)
    return filtfilt(b, a, x)


def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x), med, mad
    z = 0.6745 * (x - med) / mad
    return z, med, mad


def apply_refractory(peaks: np.ndarray, refractory_samp: int) -> np.ndarray:
    if peaks.size == 0:
        return peaks
    peaks = np.asarray(peaks, dtype=int)
    peaks.sort()
    kept = [peaks[0]]
    last = peaks[0]
    for p in peaks[1:]:
        if p - last >= refractory_samp:
            kept.append(p)
            last = p
    return np.asarray(kept, dtype=int)


def estimate_fs_from_time(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        raise ValueError("Cannot estimate sampling rate from time column.")
    med_dt = np.median(dt)
    if med_dt <= 0:
        raise ValueError("Non-positive median dt; time column may be invalid.")
    return 1.0 / med_dt


def width_gate_indices(x_hp: np.ndarray, peaks: np.ndarray, fs: float,
                       wmin_ms=None, wmax_ms=None, polarity="neg"):
    if peaks.size == 0:
        return peaks
    if wmin_ms is None and wmax_ms is None:
        return peaks

    wmin_samp = 0 if wmin_ms is None else int(round((wmin_ms / 1000.0) * fs))
    wmax_samp = int(round((wmax_ms / 1000.0) * fs)) if wmax_ms is not None else None

    if wmax_samp is None:
        search_max = int(round(0.002 * fs))
    else:
        search_max = max(wmax_samp, 1)

    kept = []
    N = len(x_hp)
    for p in peaks:
        a = p
        b = min(p + search_max, N - 1)
        if b <= a + 1:
            continue

        seg = x_hp[a:b]

        if polarity == "neg":
            j = np.argmax(seg)
        elif polarity == "pos":
            j = np.argmin(seg)
        else:
            if x_hp[p] < 0:
                j = np.argmax(seg)
            else:
                j = np.argmin(seg)

        width = j
        if width <= 0:
            continue
        if wmin_ms is not None and width < wmin_samp:
            continue
        if wmax_samp is not None and width > wmax_samp:
            continue
        kept.append(p)

    return np.asarray(kept, dtype=int)


def find_peaks_distance(z: np.ndarray, polarity: str, thr: float, distance: int) -> np.ndarray:
    peaks_all = []

    if polarity in ("pos", "both"):
        p_pos, _ = find_peaks(z, height=thr, distance=distance)
        peaks_all.append(p_pos)

    if polarity in ("neg", "both"):
        p_neg, _ = find_peaks(-z, height=thr, distance=distance)
        peaks_all.append(p_neg)

    if not peaks_all:
        return np.array([], dtype=int)

    peaks = np.unique(np.concatenate(peaks_all)).astype(int)
    peaks.sort()
    return peaks


def detect_spikes(x_uv: np.ndarray, t_s: np.ndarray):
    fs = estimate_fs_from_time(t_s)
    x_hp = bandpass_filt(x_uv.astype(float), fs, HP_SPIKE_BAND, order=3)
    z, _, _ = robust_z(x_hp)
    thr = float(SPIKE_Z_THR)

    refractory = int(round((REFRACTORY_MS / 1000.0) * fs))
    refractory = max(refractory, 1)

    peaks = find_peaks_distance(z, polarity=POLARITY, thr=thr, distance=refractory)
    peaks = apply_refractory(peaks, refractory)
    peaks = width_gate_indices(x_hp, peaks, fs, W_MIN_MS, W_MAX_MS, polarity=POLARITY)

    if AMP_MIN_UV is not None or AMP_MAX_UV is not None:
        amps = np.abs(x_hp[peaks]) if peaks.size else np.array([])
        keep = np.ones_like(peaks, dtype=bool)
        if AMP_MIN_UV is not None:
            keep &= (amps >= float(AMP_MIN_UV))
        if AMP_MAX_UV is not None:
            keep &= (amps <= float(AMP_MAX_UV))
        peaks = peaks[keep]

    spike_times = t_s[peaks] if peaks.size else np.array([], dtype=float)
    return peaks, spike_times, fs


# =========================
# RECORDING / CHANNEL HELPERS
# =========================

def choose_recording_order(csv_paths):
    print("\nFound the following CSV files:\n")
    for i, p in enumerate(csv_paths, start=1):
        print(f"  {i:>2}. {os.path.basename(p)}")

    print("\nEnter the recording order as comma-separated numbers.")
    print("Example: 3,1,2")
    print("Press Enter to keep the default alphabetical order.\n")

    while True:
        raw = input("Order: ").strip()
        if raw == "":
            return csv_paths

        try:
            idx = [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
        except ValueError:
            print("Invalid input. Use numbers separated by commas.")
            continue

        if len(idx) != len(csv_paths):
            print(f"Please provide exactly {len(csv_paths)} numbers.")
            continue
        if sorted(idx) != list(range(1, len(csv_paths) + 1)):
            print("Use each file number exactly once.")
            continue

        ordered = [csv_paths[i - 1] for i in idx]
        print("\nChosen order:")
        for j, p in enumerate(ordered, start=1):
            print(f"  {j:>2}. {os.path.basename(p)}")
        return ordered


def is_marker_column(column_name: str) -> bool:
    normalized = str(column_name).strip().lower()
    normalized = normalized.replace(" ", "_").replace("-", "_")
    return any(token in normalized for token in MARKER_COLUMN_TOKENS)


def get_channel_columns(df: pd.DataFrame, csv_name: str):
    chan_cols = []
    skipped_marker_cols = []

    for c in df.columns:
        if c == TIME_COL:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if is_marker_column(c):
            skipped_marker_cols.append(c)
            continue
        chan_cols.append(c)

    if skipped_marker_cols:
        print(f"[skip markers] {csv_name}: {', '.join(skipped_marker_cols)}")

    return chan_cols


# =========================
# MAIN
# =========================

def _normalise_label_lookup(epoch_labels_by_path=None):
    lookup = {}
    if not epoch_labels_by_path:
        return lookup
    for key, value in epoch_labels_by_path.items():
        label = "" if value is None else str(value).strip()
        lookup[str(key)] = label
        lookup[os.path.abspath(str(key))] = label
        lookup[os.path.basename(str(key))] = label
    return lookup


def _epoch_label_for_path(csv_path, lookup):
    return (
        lookup.get(os.path.abspath(csv_path))
        or lookup.get(csv_path)
        or lookup.get(os.path.basename(csv_path))
        or ""
    )


def process_csvs(csv_paths, out_dir=None, epoch_labels_by_path=None, window_sec=None):
    """Process an ordered list of raw CSV files and return the combined output CSV path."""
    ordered_csv_paths = list(csv_paths)
    if not ordered_csv_paths:
        raise FileNotFoundError("No CSV files were provided for spike counting.")

    selected_window_sec = float(window_sec if window_sec is not None else WINDOW_SEC)
    if selected_window_sec <= 0:
        raise ValueError("Spike-count window size must be greater than zero.")
    selected_window_min = selected_window_sec / 60.0
    selected_window_label = format_window_label(selected_window_sec)

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(ordered_csv_paths[0]) or ".", OUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    epoch_lookup = _normalise_label_lookup(epoch_labels_by_path)
    results_rows = []
    plot_payload = {}

    for rec_idx, csv_path in enumerate(ordered_csv_paths, start=1):
        rec_name = os.path.splitext(os.path.basename(csv_path))[0]
        epoch_label = _epoch_label_for_path(csv_path, epoch_lookup)

        print(f"\n[load] {csv_path}")
        if epoch_label:
            print(f"[label] {rec_name}: {epoch_label}")
        df = pd.read_csv(csv_path)

        if TIME_COL not in df.columns:
            raise ValueError(
                f"Time column '{TIME_COL}' not found in {os.path.basename(csv_path)}. "
                f"Columns: {list(df.columns)[:20]} ..."
            )

        t = df[TIME_COL].to_numpy(dtype=float)

        chan_cols = get_channel_columns(df, os.path.basename(csv_path))
        if not chan_cols:
            raise ValueError(
                f"No numeric signal channel columns found in {os.path.basename(csv_path)} besides time/markers."
            )

        for ch in chan_cols:
            x = df[ch].to_numpy(dtype=float)

            if np.any(~np.isfinite(x)):
                s = pd.Series(x)
                x = s.interpolate(limit_direction="both").to_numpy(dtype=float)

            peaks, spike_times, fs = detect_spikes(x, t)
            print(f"[{rec_name} | {ch}] detected spikes: {len(spike_times)} (fs≈{fs:.2f} Hz)")

            t0 = float(t[0])
            t1 = float(t[-1])
            sample_interval_s = 1.0 / float(fs) if np.isfinite(fs) and fs > 0 else 0.0
            window_bounds = folded_window_bounds(t0, t1, selected_window_sec, sample_interval_s)

            block_rows = []
            for w, (a, b) in enumerate(window_bounds):
                cnt = int(np.sum((spike_times >= a) & (spike_times < b)))
                start_min = (a - t0) / 60.0
                end_min = (b - t0) / 60.0

                row = {
                    "recording_index": rec_idx,
                    "recording_name": rec_name,
                    "epoch_label": epoch_label,
                    "channel": ch,
                    "window_index": w,
                    "window_start_s": a,
                    "window_end_s": b,
                    "window_duration_s": b - a,
                    "window_label": f"Min {start_min:g}-{end_min:g}",
                    "spike_count": cnt,
                    "fs_est_hz": float(fs),
                    "z_thr": float(SPIKE_Z_THR),
                    "polarity": POLARITY,
                    "refractory_ms": float(REFRACTORY_MS),
                    "hp_lo_hz": float(HP_SPIKE_BAND[0]),
                    "hp_hi_hz": float(HP_SPIKE_BAND[1]),
                    "amp_min_uv": AMP_MIN_UV,
                    "amp_max_uv": AMP_MAX_UV,
                    "w_min_ms": W_MIN_MS,
                    "w_max_ms": W_MAX_MS,
                }
                results_rows.append(row)
                block_rows.append(row)

            plot_block = sorted(block_rows, key=lambda r: r["window_index"])
            if len(plot_block) > 1:
                plot_block = plot_block[:-1]

            if ch not in plot_payload:
                plot_payload[ch] = []

            plot_payload[ch].append({
                "recording_index": rec_idx,
                "recording_name": rec_name,
                "rows": plot_block,
            })

    if not results_rows:
        raise RuntimeError("No spike-count results were generated.")

    out_table = pd.DataFrame(results_rows)
    out_name = (
        "ALL_RECORDINGS__spike_counts_per_min.csv"
        if math.isclose(selected_window_sec, 60.0)
        else f"ALL_RECORDINGS__spike_counts_per_{selected_window_label}.csv"
    )
    out_csv = os.path.join(out_dir, out_name)
    out_table.to_csv(out_csv, index=False)

    for ch, blocks in plot_payload.items():
        fig, ax = plt.subplots(figsize=(16, 5))

        x_cursor = 0
        x_all = []
        y_all = []
        xticks = []
        xticklabels = []
        boundary_x_positions = []
        recording_text_positions = []

        for block in blocks:
            rows = block["rows"]
            rec_name = block["recording_name"]
            n = len(rows)
            if n == 0:
                continue

            x_local = np.arange(x_cursor, x_cursor + n)
            y_local = [r["spike_count"] for r in rows]

            x_all.extend(x_local.tolist())
            y_all.extend(y_local)
            xticks.extend(x_local.tolist())
            xticklabels.extend([format_window_axis_value(int(r["window_index"]), selected_window_sec) for r in rows])
            recording_text_positions.append((x_cursor + (n - 1) / 2.0, rec_name))

            if x_cursor > 0:
                boundary_x_positions.append(x_cursor - 0.5)
            x_cursor += n

        if x_all:
            ax.plot(
                x_all,
                y_all,
                marker=TRACE_MARKER,
                linewidth=TRACE_LINEWIDTH,
                color=TRACE_COLOR,
                label=ch,
            )

        for bx in boundary_x_positions:
            ax.axvline(x=bx, linestyle="--", color="black", alpha=0.7)

        y_top_for_text = max(y_all) if y_all else 1
        y_top_for_text = max(y_top_for_text, 1)

        for xpos, rec_name in recording_text_positions:
            ax.text(
                xpos,
                y_top_for_text * 1.08,
                rec_name,
                ha="center",
                va="bottom",
                fontsize=9,
                weight="bold",
            )

        ax.set_title(f"Spike count per {selected_window_min:g}-minute window - {ch}")
        ax.set_ylabel("Spike count")
        ax.set_xlabel("Continuous time (min across ordered recordings)")
        ax.set_ylim(bottom=0)

        if xticks:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=45, ha="right")

        trace_handle = [Line2D([0], [0], color=TRACE_COLOR, marker=TRACE_MARKER, linewidth=TRACE_LINEWIDTH, label="Spike count")]
        boundary_handle = []
        if boundary_x_positions:
            boundary_handle = [Line2D([0], [0], color="black", linestyle="--", label="Recording boundary")]

        legend1 = ax.legend(handles=trace_handle, loc="upper left", title="Trace")
        ax.add_artist(legend1)
        if boundary_handle:
            legend2 = ax.legend(handles=boundary_handle, loc="upper right", title="Recording boundaries")
            ax.add_artist(legend2)

        plt.tight_layout()
        out_png = os.path.join(out_dir, f"ALL_RECORDINGS__{ch}__continuous_spike_count_curve.png")
        plt.savefig(out_png, dpi=200)
        plt.close(fig)

    print(f"\n[done] wrote:\n  {out_csv}\n  plots in: {out_dir}")
    return out_csv


def main():
    csv_paths = sorted(glob.glob(os.path.join(CSV_DIR, CSV_GLOB)))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {CSV_DIR} matching {CSV_GLOB}")

    ordered_csv_paths = choose_recording_order(csv_paths)

    out_dir = os.path.join(CSV_DIR, OUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    results_rows = []
    plot_payload = {}   # channel -> list of blocks in chosen order

    for rec_idx, csv_path in enumerate(ordered_csv_paths, start=1):
        rec_name = os.path.splitext(os.path.basename(csv_path))[0]

        print(f"\n[load] {csv_path}")
        df = pd.read_csv(csv_path)

        if TIME_COL not in df.columns:
            raise ValueError(
                f"Time column '{TIME_COL}' not found in {os.path.basename(csv_path)}. "
                f"Columns: {list(df.columns)[:20]} ..."
            )

        t = df[TIME_COL].to_numpy(dtype=float)

        chan_cols = get_channel_columns(df, os.path.basename(csv_path))
        if not chan_cols:
            raise ValueError(
                f"No numeric signal channel columns found in {os.path.basename(csv_path)} besides time/markers."
            )

        for ch in chan_cols:
            x = df[ch].to_numpy(dtype=float)

            if np.any(~np.isfinite(x)):
                s = pd.Series(x)
                x = s.interpolate(limit_direction="both").to_numpy(dtype=float)

            peaks, spike_times, fs = detect_spikes(x, t)
            print(f"[{rec_name} | {ch}] detected spikes: {len(spike_times)} (fs≈{fs:.2f} Hz)")

            t0 = float(t[0])
            t1 = float(t[-1])
            total_dur = max(0.0, t1 - t0)
            nwin = int(math.ceil(total_dur / WINDOW_SEC)) if total_dur > 0 else 0

            block_rows = []
            for w in range(nwin):
                a = t0 + w * WINDOW_SEC
                b = min(t0 + (w + 1) * WINDOW_SEC, t1 + 1e-12)
                cnt = int(np.sum((spike_times >= a) & (spike_times < b)))

                row = {
                    "recording_index": rec_idx,
                    "recording_name": rec_name,
                    "epoch_label": "",
                    "channel": ch,
                    "window_index": w,
                    "window_start_s": a,
                    "window_end_s": b,
                    "window_label": f"Min {int(w*WINDOW_MIN)}-{int((w+1)*WINDOW_MIN)}",
                    "spike_count": cnt,
                    "fs_est_hz": float(fs),
                    "z_thr": float(SPIKE_Z_THR),
                    "polarity": POLARITY,
                    "refractory_ms": float(REFRACTORY_MS),
                    "hp_lo_hz": float(HP_SPIKE_BAND[0]),
                    "hp_hi_hz": float(HP_SPIKE_BAND[1]),
                    "amp_min_uv": AMP_MIN_UV,
                    "amp_max_uv": AMP_MAX_UV,
                    "w_min_ms": W_MIN_MS,
                    "w_max_ms": W_MAX_MS,
                }
                results_rows.append(row)
                block_rows.append(row)

            # preserve your current behavior: drop last window from plotted trace
            plot_block = sorted(block_rows, key=lambda r: r["window_index"])
            if len(plot_block) > 1:
                plot_block = plot_block[:-1]

            if ch not in plot_payload:
                plot_payload[ch] = []

            plot_payload[ch].append({
                "recording_index": rec_idx,
                "recording_name": rec_name,
                "rows": plot_block,
            })

    if not results_rows:
        raise RuntimeError("No spike-count results were generated.")

    out_table = pd.DataFrame(results_rows)
    out_csv = os.path.join(out_dir, "ALL_RECORDINGS__spike_counts_per_min.csv")
    out_table.to_csv(out_csv, index=False)

    for ch, blocks in plot_payload.items():
        fig, ax = plt.subplots(figsize=(16, 5))

        x_cursor = 0
        x_all = []
        y_all = []
        xticks = []
        xticklabels = []

        boundary_x_positions = []
        recording_text_positions = []

        # First pass: build one continuous trace for this channel
        for i, block in enumerate(blocks):
            rows = block["rows"]
            rec_name = block["recording_name"]
            n = len(rows)

            if n == 0:
                continue

            x_local = np.arange(x_cursor, x_cursor + n)
            y_local = [r["spike_count"] for r in rows]

            x_all.extend(x_local.tolist())
            y_all.extend(y_local)

            xticks.extend(x_local.tolist())
            xticklabels.extend([int(r["window_index"] * WINDOW_MIN) for r in rows])

            recording_text_positions.append((x_cursor + (n - 1) / 2.0, rec_name))

            if x_cursor > 0:
                boundary_x_positions.append(x_cursor - 0.5)

            x_cursor += n

        # Plot as ONE continuous trace per channel
        if x_all:
            ax.plot(
                x_all,
                y_all,
                marker=TRACE_MARKER,
                linewidth=TRACE_LINEWIDTH,
                color=TRACE_COLOR,
                label=ch,
            )

        # Add boundary markers
        for bx in boundary_x_positions:
            ax.axvline(x=bx, linestyle="--", color="black", alpha=0.7)

        y_top_for_text = max(y_all) if y_all else 1
        y_top_for_text = max(y_top_for_text, 1)

        # Recording labels above each block
        for xpos, rec_name in recording_text_positions:
            ax.text(
                xpos,
                y_top_for_text * 1.08,
                rec_name,
                ha="center",
                va="bottom",
                fontsize=9,
                weight="bold",
            )

        ax.set_title(f"Spike count per {WINDOW_MIN:.0f}-minute window — {ch}")
        ax.set_ylabel("Spike count")
        ax.set_xlabel("Continuous time (minute windows across ordered recordings)")
        ax.set_ylim(bottom=0)

        if xticks:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=45, ha="right")

        trace_handle = [Line2D([0], [0], color=TRACE_COLOR, marker=TRACE_MARKER, linewidth=TRACE_LINEWIDTH, label="Spike count")]
        boundary_handle = []
        if boundary_x_positions:
            boundary_handle = [Line2D([0], [0], color="black", linestyle="--", label="Recording boundary")]

        legend1 = ax.legend(handles=trace_handle, loc="upper left", title="Trace")
        ax.add_artist(legend1)

        if boundary_handle:
            legend2 = ax.legend(handles=boundary_handle, loc="upper right", title="Recording boundaries")
            ax.add_artist(legend2)

        plt.tight_layout()

        out_png = os.path.join(out_dir, f"ALL_RECORDINGS__{ch}__continuous_spike_count_curve.png")
        plt.savefig(out_png, dpi=200)
        plt.close(fig)

    print(f"\n[done] wrote:\n  {out_csv}\n  plots in: {out_dir}")


if __name__ == "__main__":
    main()
