#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone script to compute and plot ONLY the LFP autocorrelogram
for every numeric channel in an Intan-exported CSV file.

Now includes a user-defined TIME WINDOW (in seconds) to extract the segment
used to compute the ACG.

No spike detection, no PSD, no STA.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# ================= USER SETTINGS ====================
CSV_DIR      = r"C:\Users\simmons\Desktop\Exploring PSDs"
CSV_GLOB     = "*.csv"
OUT_DIR_NAME = "lfp_acg_plots"

LFP_BAND       = (8.0, 10.0)   # Hz
LFP_MAX_LAG_MS = 250.0         # ±250 ms
TIME_COLS      = ["time_s", "time", "t", "seconds"]

# ---- TIME WINDOW OF INTEREST (seconds) ----
# Set both to None to use the full recording.
# Example: WINDOW_START_SEC = 30.0; WINDOW_END_SEC = 150.0
WINDOW_START_SEC = 30.0
WINDOW_END_SEC   = 90.0
# =====================================================


# ----------------- Helper Functions -----------------

def find_time_column(df: pd.DataFrame) -> str:
    for c in TIME_COLS:
        if c in df.columns:
            return c
    # fallback: first strictly increasing numeric column
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.number):
            arr = df[c].to_numpy()
            diffs = np.diff(arr.astype(float))
            if np.all(np.isfinite(diffs)) and np.all(diffs > 0):
                return c
    raise ValueError("No usable time column found.")

def infer_fs(t: np.ndarray) -> float:
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Could not infer sampling rate (non-positive or invalid dt).")
    return 1.0 / dt

def design_sos_bandpass(low, high, fs, order=4):
    ny = fs * 0.5
    low = max(low, 0.001)
    high = min(high, ny * 0.95)
    if low >= high:
        raise ValueError(f"Invalid bandpass after clamping: low={low}, high={high}, ny={ny}")
    wn = [low / ny, high / ny]
    return butter(order, wn, btype="bandpass", output="sos")

def band_filter(x, fs, band, order=4):
    sos = design_sos_bandpass(band[0], band[1], fs, order)
    return sosfiltfilt(sos, x)

def lfp_autocorrelogram(x, fs, max_lag_ms):
    x = x - np.mean(x)
    ac = np.correlate(x, x, mode="full")
    mid = len(ac) // 2

    max_lag_samp = int(round((max_lag_ms / 1000.0) * fs))
    max_lag_samp = min(max_lag_samp, mid)

    ac_clip = ac[mid - max_lag_samp : mid + max_lag_samp + 1]
    lags = np.arange(-max_lag_samp, max_lag_samp + 1) / fs

    # normalize to 0-lag
    if ac_clip[max_lag_samp] != 0:
        ac_clip = ac_clip / ac_clip[max_lag_samp]

    return lags, ac_clip

def interpolate_nans(x: np.ndarray) -> np.ndarray:
    if np.isfinite(x).all():
        return x
    idx = np.arange(len(x))
    good = np.isfinite(x)
    if good.sum() < 2:
        return np.nan_to_num(x, nan=0.0)
    return np.interp(idx, idx[good], x[good])

def apply_time_window(t: np.ndarray, start_s, end_s):
    """
    Returns a boolean mask selecting samples within [start_s, end_s].
    If start_s/end_s are None, uses full range.
    """
    if start_s is None and end_s is None:
        return np.ones_like(t, dtype=bool)

    t0 = float(t[0])
    t1 = float(t[-1])

    if start_s is None:
        start_s = t0
    if end_s is None:
        end_s = t1

    if end_s <= start_s:
        raise ValueError(f"Invalid window: end ({end_s}) must be > start ({start_s}).")

    # clamp to data range (with a warning print in main)
    start_c = max(start_s, t0)
    end_c   = min(end_s, t1)

    mask = (t >= start_c) & (t <= end_c)
    return mask, start_c, end_c, t0, t1

# ---------------------- MAIN ------------------------

def main():
    out_root = os.path.join(CSV_DIR, OUT_DIR_NAME)
    os.makedirs(out_root, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, CSV_GLOB)))
    if not csv_files:
        print("No CSV files found.")
        return

    for path in csv_files:
        base = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(out_root, base)
        os.makedirs(out_dir, exist_ok=True)

        df = pd.read_csv(path)
        time_col = find_time_column(df)
        t = df[time_col].to_numpy(float)

        # infer sampling rate from full time (more stable), then slice
        fs = infer_fs(t)

        # apply window
        mask, w_start, w_end, t0, t1 = apply_time_window(t, WINDOW_START_SEC, WINDOW_END_SEC)
        n_win = int(mask.sum())

        if (WINDOW_START_SEC is not None or WINDOW_END_SEC is not None):
            if w_start != (t0 if WINDOW_START_SEC is None else WINDOW_START_SEC) or w_end != (t1 if WINDOW_END_SEC is None else WINDOW_END_SEC):
                print(f"[{base}] ⚠️ Window clamped to data range: requested=({WINDOW_START_SEC},{WINDOW_END_SEC}) "
                      f"data=({t0:.3f},{t1:.3f}) used=({w_start:.3f},{w_end:.3f})")

        if n_win < max(10, int(fs)):  # at least ~1 second or 10 samples
            print(f"[{base}] ⚠️ Window too small ({n_win} samples). Skipping file.")
            continue

        t_win = t[mask]

        # all numeric channels except time
        channels = [c for c in df.columns
                    if c != time_col and np.issubdtype(df[c].dtype, np.number)]

        print(f"[{base}] fs={fs:.2f} Hz | {len(channels)} channels | window={w_start:.3f}–{w_end:.3f}s ({n_win} samples)")

        for chan in channels:
            x = pd.to_numeric(df[chan], errors="coerce").to_numpy(float)
            x = interpolate_nans(x)
            x_win = x[mask]

            # Filter to LFP band
            x_lfp = band_filter(x_win, fs, LFP_BAND)

            # Compute autocorrelogram
            lags, ac = lfp_autocorrelogram(x_lfp, fs, LFP_MAX_LAG_MS)

            # Plot
            plt.figure(figsize=(7, 4))
            plt.plot(lags * 1000, ac, lw=1.5)
            plt.xlabel("Lag (ms)")
            plt.ylabel("Autocorrelation (norm.)")
            plt.title(
                f"LFP autocorrelogram — {base} | {chan} "
                f"({LFP_BAND[0]}–{LFP_BAND[1]} Hz)\n"
                f"Window: {w_start:.2f}–{w_end:.2f} s"
            )
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{base}__{chan}_lfp_acg.png"), dpi=150)
            plt.close()

    print("\n✅ DONE — All LFP autocorrelograms saved in:")
    print(out_root)

if __name__ == "__main__":
    main()