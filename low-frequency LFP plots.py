#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # comment out if you want interactive windows
import matplotlib.pyplot as plt
from scipy.signal import welch

# ================== USER SETTINGS ==================
CSV_DIR      = r"C:\Users\simmons\Desktop\Exploring PSDs"  # folder with your CSV files
CSV_GLOB     = "*.csv"                                     # pattern to match
OUT_DIR_NAME = "Low Fq Plots"                                 # root output folder name

# Time & signal column inference
TIME_COLS    = ["time_s", "time", "t", "seconds"]          # candidates for time column
SIGNAL_COL   = None  # e.g. "amplifier_ch0"; if None => auto-pick first numeric non-time column

# Welch parameters
TARGET_NPERSEG_SEC = 2.0    # ~2s windows
NOVERLAP_RATIO     = 0.5    # 50% overlap
FMAX               = 100  # Hz; limit frequency axis (None => up to Nyquist)

# Band definitions (for bar chart)
LOW_BANDS = [(2,3), (3,4), (4,5), (5,8), (8,13), (13,20), (20,27), (27,40), (40,50)]  # Hz

# Plot y-limits in dB (set None for autoscale)
DB_RANGE = None
# ===================================================


# ----------------- Helpers -----------------
def find_time_column(df: pd.DataFrame) -> str:
    """Find a reasonable 'time' column; otherwise infer by monotonic numeric column."""
    for c in TIME_COLS:
        if c in df.columns:
            return c
    # fall back: first strictly-increasing numeric column
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.number):
            s = df[c].to_numpy()
            diffs = np.diff(s.astype(float))
            if np.all(np.isfinite(diffs)) and np.all(diffs > 0):
                return c
    raise ValueError("Could not find a time column. Add/rename your time column or update TIME_COLS.")

def infer_fs(t: np.ndarray) -> float:
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Time column invalid (nonpositive or non-finite diffs).")
    return 1.0 / dt

def pick_signal_col(df: pd.DataFrame, time_col: str) -> str:
    """Choose the first numeric column that's not the time column (or use SIGNAL_COL if set)."""
    if SIGNAL_COL is not None:
        if SIGNAL_COL not in df.columns:
            raise ValueError(f"Requested SIGNAL_COL '{SIGNAL_COL}' not found in CSV.")
        return SIGNAL_COL
    for c in df.columns:
        if c == time_col:
            continue
        if np.issubdtype(df[c].dtype, np.number):
            return c
    raise ValueError("No numeric signal column found besides time.")

def welch_psd(x: np.ndarray, fs: float, nperseg: int, noverlap: int, fmax: float | None):
    """Return (f, Pxx_linear[µV²/Hz], Pxx_dB[dB(µV²/Hz)]) with optional f<=fmax limit."""
    f, Pxx = welch(
        x, fs=fs, nperseg=nperseg, noverlap=noverlap,
        window="hann", detrend="constant", scaling="density", average="mean"
    )  # Pxx in units^2/Hz; for Intan LFP this is µV²/Hz
    if fmax is not None:
        keep = f <= fmax
        f, Pxx = f[keep], Pxx[keep]
    Pxx_db = 10.0*np.log10(Pxx + np.finfo(float).eps)
    return f, Pxx, Pxx_db

def bandpower_linear(f: np.ndarray, Pxx_linear: np.ndarray, band: tuple[float,float]) -> float:
    """Integrate linear PSD over [lo, hi) Hz to get band power in µV²."""
    lo, hi = band
    keep = (f >= lo) & (f < hi)
    if not np.any(keep):
        return np.nan
    return np.trapz(Pxx_linear[keep], f[keep])

def list_csvs(directory: str, pattern: str):
    return sorted(glob.glob(os.path.join(directory, pattern)))
# -------------------------------------------


def main():
    # Prepare output
    out_root = os.path.join(CSV_DIR, OUT_DIR_NAME)
    os.makedirs(out_root, exist_ok=True)

    # Gather files
    csv_files = list_csvs(CSV_DIR, CSV_GLOB)
    if not csv_files:
        raise SystemExit(f"No CSVs found in {CSV_DIR!r} with pattern {CSV_GLOB!r}")

    # For the combined overlay: store per-file (f, Pxx_dB) aligned later
    overlay_freqs = []
    overlay_psds_db = []
    overlay_labels = []

    # Global dB limits (for consistent per-file dB plots if DB_RANGE=None)
    global_db_min, global_db_max = np.inf, -np.inf

    # ---------- Per-file processing ----------
    for path in csv_files:
        base = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(out_root, base)
        os.makedirs(out_dir, exist_ok=True)

        # Load
        df = pd.read_csv(path)
        time_col = find_time_column(df)
        t = df[time_col].to_numpy(dtype=float)
        fs = infer_fs(t)
        sig_col = pick_signal_col(df, time_col)
        x = pd.to_numeric(df[sig_col], errors="coerce").to_numpy(float)

        # NaN handling: interpolate over NaNs in x
        if not np.isfinite(x).all():
            idx = np.arange(len(x), dtype=float)
            good = np.isfinite(x)
            if good.sum() >= 2:
                first, last = np.where(good)[0][0], np.where(good)[0][-1]
                x[:first] = x[first]
                x[last+1:] = x[last]
                x[~good] = np.interp(idx[~good], idx[good], x[good])
            else:
                print(f"Skipping {path}: insufficient finite samples.")
                continue

        # Welch params
        nperseg = max(64, int(round(TARGET_NPERSEG_SEC * fs)))
        noverlap = int(round(nperseg * NOVERLAP_RATIO))

        # PSD
        f, Pxx_lin, Pxx_db = welch_psd(x, fs, nperseg, noverlap, fmax=FMAX)

        # Track global dB range
        finite_db = Pxx_db[np.isfinite(Pxx_db)]
        if finite_db.size:
            global_db_min = min(global_db_min, float(finite_db.min()))
            global_db_max = max(global_db_max, float(finite_db.max()))

        # Save per-file PSD (dB)
        plt.figure(figsize=(9,5))
        plt.plot(f, Pxx_db, lw=1.5)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (dB(µV²/Hz))")
        plt.title(f"Welch PSD (dB) — {base}  |  fs≈{fs:.3f} Hz  |  col={sig_col}")
        plt.grid(True, alpha=0.3)
        if FMAX is not None:
            plt.xlim(0, FMAX)
        if DB_RANGE is not None:
            plt.ylim(*DB_RANGE)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base}_psd_db.png"), dpi=150)
        plt.close()

        # Save per-file PSD (linear)
        plt.figure(figsize=(9,5))
        plt.plot(f, Pxx_lin, lw=1.5)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (µV²/Hz)")
        plt.title(f"Welch PSD (linear) — {base}  |  fs≈{fs:.3f} Hz  |  col={sig_col}")
        plt.grid(True, alpha=0.3)
        if FMAX is not None:
            plt.xlim(0, FMAX)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base}_psd_linear.png"), dpi=150)
        plt.close()

        # Band-averaged power bar chart (using linear PSD integration)
        band_vals = [bandpower_linear(f, Pxx_lin, band) for band in LOW_BANDS]
        band_labels = [f"{lo}-{hi} Hz" for (lo,hi) in LOW_BANDS]

        plt.figure(figsize=(8,5))
        plt.bar(band_labels, band_vals)
        plt.ylabel("Band power (µV²)")
        plt.title(f"Band-averaged power — {base}")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base}_bandpower_bars.png"), dpi=150)
        plt.close()

        # Stash for combined overlay
        overlay_freqs.append(f)
        overlay_psds_db.append(Pxx_db)
        overlay_labels.append(base)

    # ---------- Combined overlay (all files on one PSD dB plot) ----------
    if overlay_freqs:
        # Choose common frequency grid = shortest of all
        f_common = overlay_freqs[np.argmin([len(fi) for fi in overlay_freqs])]
        plt.figure(figsize=(10,6))

        for f_i, P_i, label in zip(overlay_freqs, overlay_psds_db, overlay_labels):
            if np.array_equal(f_i, f_common):
                y = P_i
            else:
                y = np.interp(f_common, f_i, P_i)
            plt.plot(f_common, y, lw=1.5, label=label)

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (dB(µV²/Hz))")
        plt.title("Welch PSD overlay (all files)")
        plt.grid(True, alpha=0.3)
        if FMAX is not None:
            plt.xlim(0, FMAX)

        # If user didn't set DB_RANGE, auto from global min/max
        if DB_RANGE is None and np.isfinite(global_db_min) and np.isfinite(global_db_max):
            pad = 3.0
            plt.ylim(global_db_min - pad, global_db_max + pad)
        elif DB_RANGE is not None:
            plt.ylim(*DB_RANGE)

        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        # Save this at the root of output folder
        plt.savefig(os.path.join(out_root, "all_files_psd_overlay_dB.png"), dpi=150)
        plt.close()

    print(f"✅ Done. Plots saved under: {out_root}")

if __name__ == "__main__":
    main()
