#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # comment out for interactive
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# ---------------- USER SETTINGS ----------------

# Option B (optional): if CSV_PATH doesn't exist, use folder + glob
CSV_DIR  = r"C:\Users\simmons\Desktop\Exploring PSDs"
CSV_GLOB = "*.csv"

# Columns
TIME_COLS  = ["time_s", "time", "t", "seconds"]
SIGNAL_COL = None  # set e.g. "A-008" to force a channel; None = auto first numeric non-time

# Spike-band for estimating noise floor
HP_SPIKE_BAND = (300.0, 5000.0)  # Hz
FILTER_ORDER  = 4

# Histogram
BINS = 300
OUT_PNG = "noise_hist_hp_300_5000Hz.png"
# ------------------------------------------------


def find_time_column(df: pd.DataFrame) -> str:
    for c in TIME_COLS:
        if c in df.columns:
            return c
    # fallback: find numeric monotonic increasing column
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.number):
            s = df[c].to_numpy(dtype=float)
            diffs = np.diff(s)
            if np.all(np.isfinite(diffs)) and np.all(diffs > 0):
                return c
    raise ValueError("Could not find a time column. Add/rename or update TIME_COLS.")


def infer_fs(t: np.ndarray) -> float:
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Time column invalid (nonpositive/non-finite diffs).")
    return 1.0 / dt


def pick_signal_col(df: pd.DataFrame, time_col: str) -> str:
    if SIGNAL_COL is not None:
        if SIGNAL_COL not in df.columns:
            raise ValueError(f"SIGNAL_COL '{SIGNAL_COL}' not found in CSV.")
        return SIGNAL_COL
    for c in df.columns:
        if c != time_col and np.issubdtype(df[c].dtype, np.number):
            return c
    raise ValueError("No numeric signal column found besides time.")


def _safe_band_for_fs(low, high, fs, margin=0.95):
    ny = fs * 0.5
    hi = min(high, ny * margin) if high is not None else None
    lo = max(low, 0.001) if low is not None else None
    if hi is not None and lo is not None and hi <= lo:
        hi = min(lo * 1.2, ny * margin)
    return lo, hi


def design_sos_bandpass(low, high, fs, order=4):
    low, high = _safe_band_for_fs(low, high, fs)
    if low is None and high is None:
        raise ValueError("Both low and high are None.")
    if low is None:
        wn = high / (fs * 0.5)
        sos = butter(order, wn, btype="lowpass", output="sos")
    elif high is None:
        wn = low / (fs * 0.5)
        sos = butter(order, wn, btype="highpass", output="sos")
    else:
        wn = [low / (fs * 0.5), high / (fs * 0.5)]
        sos = butter(order, wn, btype="bandpass", output="sos")
    return sos


def band_filter(x, fs, band, order=4):
    sos = design_sos_bandpass(band[0], band[1], fs, order=order)
    return sosfiltfilt(sos, x)


def load_csv_path():
    if os.path.isfile(CSV_DIR):
        return CSV_DIR
    matches = sorted(glob.glob(os.path.join(CSV_DIR, CSV_GLOB)))
    if not matches:
        raise FileNotFoundError(f"No CSV found at CSV_PATH and none in {CSV_DIR} matching {CSV_GLOB}")
    return matches[0]


def main():
    path = load_csv_path()
    base = os.path.splitext(os.path.basename(path))[0]
    out_png = os.path.join(os.path.dirname(path), f"{base}_{OUT_PNG}")

    df = pd.read_csv(path)

    time_col = find_time_column(df)
    t = df[time_col].to_numpy(dtype=float)
    fs = infer_fs(t)

    sig_col = pick_signal_col(df, time_col)
    x = pd.to_numeric(df[sig_col], errors="coerce").to_numpy(dtype=float)

    # Handle NaNs (simple interpolate)
    if not np.isfinite(x).all():
        idx = np.arange(len(x), dtype=float)
        good = np.isfinite(x)
        if good.sum() < 2:
            raise ValueError("Signal column has <2 finite samples; cannot interpolate.")
        first, last = np.where(good)[0][0], np.where(good)[0][-1]
        x[:first] = x[first]
        x[last + 1:] = x[last]
        x[~good] = np.interp(idx[~good], idx[good], x[good])

    # High-pass/bandpass to spike-band
    x_hp = band_filter(x, fs, HP_SPIKE_BAND, order=FILTER_ORDER)

    # Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(x_hp, bins=BINS, density=True)
    plt.xlabel("Amplitude (µV)")
    plt.ylabel("Probability density")
    plt.title(f"HP noise distribution ({HP_SPIKE_BAND[0]}–{HP_SPIKE_BAND[1]} Hz)\n{base} | fs≈{fs:.2f} Hz | col={sig_col}")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("Saved:", out_png)
    print("Tip: send me this PNG and I can recommend AMP_MIN_UV based on the noise floor.")


if __name__ == "__main__":
    main()
