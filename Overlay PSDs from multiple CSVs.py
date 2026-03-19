#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Overlay Welch PSDs from the SAME channel across ALL CSV files.

- Each CSV is assumed to have:
    * One time column (in seconds) that can be inferred.
    * One or more numeric signal columns (channels).

- The script:
    * Finds all CSVs matching CSV_GLOB in CSV_DIR
    * For each CSV:
        - Infers the time column and sampling rate
        - Finds all numeric channel columns (besides time)
        - Computes Welch PSD for each channel
    * Groups PSDs by channel name
    * For each channel:
        - Interpolates all PSDs for that channel onto a common frequency grid
        - Produces ONE PNG figure with curves overlaid from all CSVs.

Curve labels: "<file_base>" (one curve per file containing that channel)
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # comment this line out if you want interactive windows
import matplotlib.pyplot as plt
from scipy.signal import welch


# ================== USER SETTINGS ==================
# Folder that contains your CSVs
CSV_DIR  = r"C:\Users\simmons\Desktop\Exploring PSDs"
CSV_GLOB = "*.csv"      # which CSVs to include (pattern)

# Output directory for per-channel overlay plots
OUT_DIR = "overlay_psd_by_channel"

# Time column inference candidates
TIME_COLS = ["time_s", "time", "t", "seconds"]

# Optional: only include or exclude certain channel names
# None  -> include all numeric channels
CHANNEL_INCLUDE = None        # e.g. ["A-001", "A-002"]
CHANNEL_EXCLUDE = []          # e.g. ["A-001-A-002"] for bipolar columns you don't want

# Welch PSD parameters
TARGET_NPERSEG_SEC = 2.0      # window length in seconds
NOVERLAP_RATIO     = 0.5      # 50% overlap
FMAX               = 150.0    # Hz; x-axis max (set None for full up to Nyquist)

# Plot dB range (None -> autoscale per channel)
DB_RANGE = None               # e.g. (-130, -60) or None
# ===================================================

CURVE_COLORS = {
    # "Baseline_01": "tab:blue",
    # "Post_01": "tab:orange",
    # "Mouse3": "#2ca02c",
}

# Optional fallback rules based on label text
COLOR_RULES = [
    ("Baseline", "tab:blue"),
    ("Post", "tab:orange"),
]
# ===================================================

def get_curve_color(label: str):
    """Return the plotting color for a curve label."""
    if label in CURVE_COLORS:
        return CURVE_COLORS[label]

    for token, color in COLOR_RULES:
        if token in label:
            return color

    return None
# ----------------- Helpers -----------------

def find_time_column(df: pd.DataFrame) -> str:
    """Heuristically find a 'time' column; otherwise use first strictly-increasing numeric col."""
    for c in TIME_COLS:
        if c in df.columns:
            return c

    # Fallback: first strictly increasing numeric column
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.number):
            s = df[c].to_numpy()
            diffs = np.diff(s.astype(float))
            if np.all(np.isfinite(diffs)) and np.all(diffs > 0):
                return c

    raise ValueError(
        "Could not find a time column. Add/rename your time column or update TIME_COLS."
    )


def infer_fs(t: np.ndarray) -> float:
    """Infer sampling frequency from time vector."""
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Time column invalid (nonpositive or non-finite diffs).")
    return 1.0 / dt


def list_signal_channels(df: pd.DataFrame, time_col: str) -> list[str]:
    """Return all numeric columns except time, applying include/exclude filters if set."""
    chans = []
    for c in df.columns:
        if c == time_col:
            continue
        if np.issubdtype(df[c].dtype, np.number):
            chans.append(c)

    if CHANNEL_INCLUDE is not None:
        include_set = set(CHANNEL_INCLUDE)
        chans = [c for c in chans if c in include_set]

    if CHANNEL_EXCLUDE:
        exclude_set = set(CHANNEL_EXCLUDE)
        chans = [c for c in chans if c not in set(CHANNEL_EXCLUDE)]

    return chans


def sanitize(name: str) -> str:
    """Safe label for filenames."""
    return re.sub(r'[^A-Za-z0-9_\-\.]+', '_', name)


def welch_psd(x: np.ndarray, fs: float, nperseg: int, noverlap: int, fmax: float | None):
    """
    Return (f, Pxx_linear, Pxx_dB) where
      - Pxx_linear is in units^2/Hz (e.g., µV²/Hz for Intan data)
      - Pxx_dB   is 10*log10(Pxx_linear)
    """
    f, Pxx = welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
        detrend="constant",
        scaling="density",
        average="mean",
    )
    if fmax is not None:
        keep = f <= fmax
        f, Pxx = f[keep], Pxx[keep]
    Pxx_db = 10.0 * np.log10(Pxx + np.finfo(float).eps)
    return f, Pxx, Pxx_db


def list_csvs(directory: str, pattern: str):
    return sorted(glob.glob(os.path.join(directory, pattern)))


# ----------------- Main -----------------
def main():
    # Gather CSVs
    csv_files = list_csvs(CSV_DIR, CSV_GLOB)
    if len(csv_files) < 2:
        print(f"[WARN] Only {len(csv_files)} CSV file(s) found. "
              f"Script will still run, but overlays per channel may have just one curve.")
    print(f"Found {len(csv_files)} CSV files. Processing...")

    # Dictionary mapping channel_name -> dict with lists of (freqs, psds, label)
    # channel_data[chan]["freqs"]   -> list of frequency arrays
    # channel_data[chan]["psds_db"] -> list of PSD(dB) arrays
    # channel_data[chan]["labels"]  -> list of file labels
    channel_data: dict[str, dict[str, list]] = {}

    for path in csv_files:
        base = os.path.splitext(os.path.basename(path))[0]
        print(f"  -> {base}")

        df = pd.read_csv(path)

        # Time vector & sampling rate
        time_col = find_time_column(df)
        t = df[time_col].to_numpy(dtype=float)
        fs = infer_fs(t)

        # Channel list
        chans = list_signal_channels(df, time_col)
        if not chans:
            print(f"     [WARN] No numeric channels in {base} (besides time). Skipping file.")
            continue

        # Welch parameters for this file
        nperseg = max(64, int(round(TARGET_NPERSEG_SEC * fs)))
        noverlap = int(round(nperseg * NOVERLAP_RATIO))

        for chan in chans:
            x = pd.to_numeric(df[chan], errors="coerce").to_numpy(float)

            # Handle NaNs by interpolation
            if not np.isfinite(x).all():
                idx = np.arange(len(x), dtype=float)
                good = np.isfinite(x)
                if good.sum() >= 2:
                    first, last = np.where(good)[0][0], np.where(good)[0][-1]
                    x[:first] = x[first]
                    x[last + 1:] = x[last]
                    x[~good] = np.interp(idx[~good], idx[good], x[good])
                else:
                    print(f"     [WARN] Skipping {base}::{chan}: insufficient finite samples.")
                    continue

            f, Pxx_lin, Pxx_db = welch_psd(x, fs, nperseg, noverlap, fmax=FMAX)

            if chan not in channel_data:
                channel_data[chan] = {"freqs": [], "psds_db": [], "labels": []}

            channel_data[chan]["freqs"].append(f)
            channel_data[chan]["psds_db"].append(Pxx_db)
            channel_data[chan]["labels"].append(base)

    if not channel_data:
        raise SystemExit("No PSDs calculated (no valid channels found in any file).")

    # Make output directory
    out_root = os.path.join(CSV_DIR, OUT_DIR)
    os.makedirs(out_root, exist_ok=True)

    print("\nCreating per-channel overlay plots...")

    # ---------- Per-channel overlay plots ----------
    for chan, data in channel_data.items():
        freqs_list = data["freqs"]
        psds_list = data["psds_db"]
        labels = data["labels"]

        if not freqs_list:
            continue

        # Choose a common frequency grid for this channel:
        # here we use the shortest frequency array to avoid extrapolation.
        lens = [len(f) for f in freqs_list]
        idx_min = int(np.argmin(lens))
        f_common = freqs_list[idx_min]

        # Determine dB range for this channel (if autoscaling)
        ch_db_min, ch_db_max = np.inf, -np.inf

        # Interpolate all PSDs to the common grid
        curves = []
        for f_i, P_i in zip(freqs_list, psds_list):
            if np.array_equal(f_i, f_common):
                y = P_i
            else:
                y = np.interp(f_common, f_i, P_i)
            curves.append(y)

            finite_db = y[np.isfinite(y)]
            if finite_db.size:
                ch_db_min = min(ch_db_min, float(finite_db.min()))
                ch_db_max = max(ch_db_max, float(finite_db.max()))

        # Plot this channel's overlay
        plt.figure(figsize=(10, 6))
        for y, label in zip(curves, labels):
            color = get_curve_color(label)
            plt.plot(f_common, y, lw=1.2, alpha=0.85, label=label, color=color)

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (dB(µV²/Hz))")
        plt.title(f"Welch PSD overlay — channel {chan}")
        plt.grid(True, alpha=0.3)

        if FMAX is not None:
            plt.xlim(0, FMAX)

        if DB_RANGE is None and np.isfinite(ch_db_min) and np.isfinite(ch_db_max):
            pad = 3.0
            plt.ylim(ch_db_min - pad, ch_db_max + pad)
        elif DB_RANGE is not None:
            plt.ylim(*DB_RANGE)

        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        

        out_name = f"PSD_overlay_dB__{sanitize(chan)}.png"
        out_path = os.path.join(out_root, out_name)
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"  Saved: {out_path}")

    print("\n✅ Done. Per-channel overlay PSDs saved under:")
    print(f"   {out_root}")


if __name__ == "__main__":
    main()
