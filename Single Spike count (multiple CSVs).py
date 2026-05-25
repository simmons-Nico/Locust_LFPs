#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spike-count summary script for neuropil recordings.

FEATURES:
    - Only the **middle 3 minutes** (180 seconds) of each recording are used.
    - For each channel, a single line graph is generated:
        x-axis = Condition (PBS, Pre, Post1, Post2) that actually have data
        y-axis = SpikeCount (mean ± SD)
        raw data points also plotted.

Across files:
  - writes ALL_SPIKE_COUNTS.csv with columns:
      Animal, Condition, Channel, SpikeCount
  - generates one line plot per channel.
"""

import os
import glob
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, find_peaks, peak_widths

# ================== USER SETTINGS ==================

CSV_DIR      = r"C:\Users\simmons\Desktop\Exploring PSDs"
CSV_GLOB     = "*.csv"
OUT_DIR_NAME = "spike_counts_summary"

TIME_COLS    = ["time_s", "time", "t", "seconds"]

MIDDLE_WINDOW_SEC = 1200   # 3 minutes

# Spike detection parameters
HP_SPIKE_BAND   = (300.0, 3000.0)
SPIKE_Z_THR     = 6.0
SPIKE_POLARITY  = "both"
AMP_MIN_UV      = 40.0
AMP_MAX_UV      = 700.0
MIN_WIDTH_MS    = 0.2
MAX_WIDTH_MS    = 2.0
REFRACTORY_MS   = 1.0

# Preferred ordering on x-axis
CONDITIONS_ORDER = ["PBS", "Baseline", "Post(-70nA)", "Post(-1V)"]


# ================== HELPER FUNCTIONS ==================

def list_csvs(directory, pattern):
    return sorted(glob.glob(os.path.join(directory, pattern)))

def find_time_column(df: pd.DataFrame) -> str:
    """Find time column by name or by monotonic numeric values."""
    for c in TIME_COLS:
        if c in df.columns:
            return c
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.number):
            s = df[c].to_numpy()
            diffs = np.diff(s.astype(float))
            if np.all(diffs > 0):
                return c
    raise ValueError("No time column found.")

def infer_fs(t: np.ndarray) -> float:
    dt = np.median(np.diff(t))
    if dt <= 0:
        raise ValueError("Time column invalid.")
    return 1.0 / dt

def _safe_band_for_fs(low, high, fs, margin=0.95):
    ny = fs * 0.5
    hi = min(high, ny * margin)
    lo = max(low, 0.001)
    if hi <= lo:
        hi = lo * 1.2
    return lo, hi

def design_sos_bandpass(low, high, fs, order=4):
    low, high = _safe_band_for_fs(low, high, fs)
    wn = [low / (fs * 0.5), high / (fs * 0.5)]
    return butter(order, wn, btype="bandpass", output="sos")

def band_filter(x, fs, band, order=4):
    sos = design_sos_bandpass(band[0], band[1], fs, order=order)
    return sosfiltfilt(sos, x)

def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad)

def detect_spikes_advanced(xhp, fs,
                           zthr=SPIKE_Z_THR,
                           refr_ms=REFRACTORY_MS,
                           polarity=SPIKE_POLARITY,
                           amp_min_uv=AMP_MIN_UV,
                           amp_max_uv=AMP_MAX_UV,
                           min_width_ms=MIN_WIDTH_MS,
                           max_width_ms=MAX_WIDTH_MS):
    """Spike detector for neuropil recordings."""

    if polarity == "neg":
        x_for_peaks = -xhp
    elif polarity == "pos":
        x_for_peaks = xhp
    else:
        x_for_peaks = np.abs(xhp)

    z = robust_z(x_for_peaks)
    distance = int(round((refr_ms / 1000.0) * fs))

    idx, _ = find_peaks(z, height=zthr, distance=max(1, distance))
    if idx.size == 0:
        return np.array([], dtype=int)

    amps = np.abs(xhp[idx])
    keep = (amps >= amp_min_uv) & (amps <= amp_max_uv)
    idx = idx[keep]
    if idx.size == 0:
        return np.array([], dtype=int)

    widths_samples, _, _, _ = peak_widths(np.abs(xhp), idx, rel_height=0.5)
    widths_ms = widths_samples / fs * 1000.0
    keep2 = (widths_ms >= min_width_ms) & (widths_ms <= max_width_ms)
    return idx[keep2]


def extract_condition_from_filename(base: str) -> str:
    """
    Robust condition parser based on substrings in filename (case-insensitive).
    Adjust this mapping to match your actual naming scheme.
    """
    name = base.lower()

    if "pbs" in name or "vehicle" in name:
        return "PBS"
    if ("Baseline" in name or "base" in name):
        return "Baseline"
    if ("6" in name or "Post" in name):
        return "Post(-70nA)"
    if ("1" in name or "ablation1" in name):
        return "Post(-1V)"


    print(f"  [WARN] Could not infer condition from filename '{base}', labelling as 'Unknown'.")
    return "Unknown"

def extract_animal_from_filename(base: str) -> str:
    """Animal ID = substring before first underscore."""
    if "_" in base:
        return base.split("_")[0]
    return base


# ================== MAIN SCRIPT ==================

def main():
    out_root = os.path.join(CSV_DIR, OUT_DIR_NAME)
    os.makedirs(out_root, exist_ok=True)

    csv_files = list_csvs(CSV_DIR, CSV_GLOB)
    if not csv_files:
        raise SystemExit("No CSVs found.")

    spike_rows = []

    # ------------- PROCESS EACH CSV -------------
    for path in csv_files:
        base = os.path.splitext(os.path.basename(path))[0]
        print(f"Processing {base}")

        df = pd.read_csv(path)
        time_col = find_time_column(df)
        t = df[time_col].to_numpy(float)
        fs = infer_fs(t)

        # middle 3 minutes
        duration = t[-1] - t[0]
        if duration <= MIDDLE_WINDOW_SEC:
            mask = np.ones_like(t, bool)
        else:
            half_win = MIDDLE_WINDOW_SEC / 2
            mid = (t[0] + t[-1]) / 2
            mask = (t >= mid - half_win) & (t <= mid + half_win)

        channel_cols = [
            c for c in df.columns
            if c != time_col and np.issubdtype(df[c].dtype, np.number)
        ]

        condition = extract_condition_from_filename(base)
        animal_id = extract_animal_from_filename(base)

        for chan in channel_cols:
            x_full = df[chan].to_numpy(float)
            x = x_full[mask]

            if not np.isfinite(x).all():
                idx = np.arange(len(x))
                good = np.isfinite(x)
                x[~good] = np.interp(idx[~good], idx[good], x[good])

            x_hp = band_filter(x, fs, HP_SPIKE_BAND)
            spike_idx = detect_spikes_advanced(x_hp, fs)
            spike_count = int(len(spike_idx))

            spike_rows.append({
                "Animal": animal_id,
                "Condition": condition,
                "Channel": chan,
                "SpikeCount": spike_count
            })

    # ---------- SAVE SUMMARY CSV ----------
    df = pd.DataFrame(spike_rows)
    out_csv = os.path.join(out_root, "ALL_SPIKE_COUNTS.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved summary CSV: {out_csv}")

    # ---------- LINE PLOTS PER CHANNEL ----------
    channels = sorted(df["Channel"].unique())

    for ch in channels:
        df_ch = df[df["Channel"] == ch].copy()

        # what conditions actually exist for this channel?
        present = [c for c in CONDITIONS_ORDER if c in df_ch["Condition"].values]
        if not present:
            print(f"  [WARN] No recognized conditions for channel {ch}. Skipping plot.")
            continue

        df_ch = df_ch[df_ch["Condition"].isin(present)]

        # set categorical order to PRESENT conditions only
        df_ch["Condition"] = pd.Categorical(df_ch["Condition"],
                                            categories=present,
                                            ordered=True)

        means = df_ch.groupby("Condition")["SpikeCount"].mean()
        sds   = df_ch.groupby("Condition")["SpikeCount"].std()

        x = np.arange(len(present))

        plt.figure(figsize=(7, 5))

        # raw points
        for i, cond in enumerate(present):
            vals = df_ch[df_ch["Condition"] == cond]["SpikeCount"].values
            if vals.size:
                plt.scatter(np.full_like(vals, i), vals,
                            color="gray", alpha=0.6, s=50)

        # mean ± SD
        plt.errorbar(x, means[present], yerr=sds[present],
                     fmt="-o", color="blue",
                     capsize=5, linewidth=2, markersize=8)

        plt.xticks(x, present)
        plt.xlabel("Condition")
        plt.ylabel("Spike Count")
        plt.title(f"Spike Count Across Conditions — Channel {ch}")
        plt.tight_layout()

        out_plot = os.path.join(out_root, f"lineplot_spikecounts_{ch}.png")
        plt.savefig(out_plot, dpi=150)
        plt.close()
        print(f"  Line plot saved: {out_plot}")

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
