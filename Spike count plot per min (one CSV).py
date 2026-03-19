#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spike count per 1-minute window (ONE CSV)
- Loads one CSV (time_s + one or more channel columns)
- Band-pass filters in HP_SPIKE_BAND for spikes
- Detects spikes using robust MAD z-threshold
- Uses scipy.signal.find_peaks(..., distance=...) for candidate peaks
- Enforces a global refractory period between accepted spikes
- Outputs:
  1) per-minute spike count CSV
  2) spike count curve PNG per channel (excluding the last window from plotting)

Assumes your CSV is from Intan conversion and contains:
  - a time column named TIME_COL (default: "time_s")
  - one or more voltage columns in microvolts (µV)
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks


# =========================
# USER SETTINGS
# =========================

# Input folder containing ONE csv (or multiple; the script will pick the first match)
CSV_DIR  = r"C:\Users\simmons\Desktop\Exploring PSDs"
CSV_GLOB = "*.csv"

# Output folder (created inside CSV_DIR)
OUT_DIR_NAME = "spike_counts_per_min"

# Time column
TIME_COL = "time_s"

# Spike detection band (Hz)
HP_SPIKE_BAND = (300.0, 5000.0)

# Threshold in robust z (MAD-based)
SPIKE_Z_THR = 6.0

# Detect peaks by polarity: "neg", "pos", "both"
POLARITY = "both"

# Refractory (minimum time between accepted spikes)
REFRACTORY_MS = 1.0

# Optional spike-shape / amplitude gating (set to None to disable)
AMP_MIN_UV = 40.0
AMP_MAX_UV = 700.0

# Width gate in ms (trough-to-peak or peak-to-trough proxy; optional)
W_MIN_MS = None
W_MAX_MS = None

# Windowing
WINDOW_MIN = 1.0
WINDOW_SEC = WINDOW_MIN * 60.0


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
    """Robust z-score using MAD."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x), med, mad
    z = 0.6745 * (x - med) / mad
    return z, med, mad


def apply_refractory(peaks: np.ndarray, refractory_samp: int) -> np.ndarray:
    """Keep first peak, then only peaks at least refractory_samp later."""
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
    """
    Optional width gate:
    crude proxy: after a peak, find opposite extremum within ~2 ms,
    measure sample distance.
    If disabled, returns peaks unchanged.
    """
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
            j = np.argmax(seg)  # max after trough
        elif polarity == "pos":
            j = np.argmin(seg)  # min after peak
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
    """
    Candidate peak finder using scipy.signal.find_peaks with distance.
    Works in z-space.
      - pos: peaks on +z above thr
      - neg: peaks on -z above thr (i.e., troughs on z below -thr)
      - both: union of both
    Returns sorted unique indices.
    """
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
    """
    Detect spikes on HP-filtered signal using MAD-z threshold + polarity + refractory + amp+width gates.
    Returns: spike sample indices, spike times (s), fs
    """
    fs = estimate_fs_from_time(t_s)

    # HP filter for spikes
    x_hp = bandpass_filt(x_uv.astype(float), fs, HP_SPIKE_BAND, order=3)

    # Robust z
    z, _, _ = robust_z(x_hp)
    thr = float(SPIKE_Z_THR)

    # Refractory in samples (used as find_peaks distance + global cleanup)
    refractory = int(round((REFRACTORY_MS / 1000.0) * fs))
    refractory = max(refractory, 1)

    # Candidate peaks using scipy find_peaks(distance=...)
    peaks = find_peaks_distance(z, polarity=POLARITY, thr=thr, distance=refractory)

    # IMPORTANT: if POLARITY="both", pos/neg were distance-filtered separately
    # This enforces a *global* refractory across the merged set.
    peaks = apply_refractory(peaks, refractory)

    # Optional width gate
    peaks = width_gate_indices(x_hp, peaks, fs, W_MIN_MS, W_MAX_MS, polarity=POLARITY)

    # Optional amplitude gate (use HP amplitude at peak)
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
# MAIN
# =========================

def main():
    # find CSV
    csv_paths = sorted(glob.glob(os.path.join(CSV_DIR, CSV_GLOB)))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {CSV_DIR} matching {CSV_GLOB}")
    csv_path = csv_paths[0]
    print(f"[load] {csv_path}")

    out_dir = os.path.join(CSV_DIR, OUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if TIME_COL not in df.columns:
        raise ValueError(f"Time column '{TIME_COL}' not found. Columns: {list(df.columns)[:20]} ...")

    t = df[TIME_COL].to_numpy(dtype=float)

    # Choose channel columns: all numeric columns except time
    chan_cols = []
    for c in df.columns:
        if c == TIME_COL:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            chan_cols.append(c)
    if not chan_cols:
        raise ValueError("No numeric channel columns found in CSV besides time.")

    results_rows = []

    for ch in chan_cols:
        x = df[ch].to_numpy(dtype=float)

        # Handle NaNs by interpolation
        if np.any(~np.isfinite(x)):
            s = pd.Series(x)
            x = s.interpolate(limit_direction="both").to_numpy(dtype=float)

        peaks, spike_times, fs = detect_spikes(x, t)
        print(f"[{ch}] detected spikes: {len(spike_times)} (fs≈{fs:.2f} Hz)")

        # Per-minute windows
        t0 = float(t[0])
        t1 = float(t[-1])
        total_dur = max(0.0, t1 - t0)
        nwin = int(math.ceil(total_dur / WINDOW_SEC)) if total_dur > 0 else 0

        for w in range(nwin):
            a = t0 + w * WINDOW_SEC
            b = min(t0 + (w + 1) * WINDOW_SEC, t1 + 1e-12)

            cnt = int(np.sum((spike_times >= a) & (spike_times < b)))

            results_rows.append({
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
            })

        # Plot spike count curve for this channel
        sub = [r for r in results_rows if r["channel"] == ch]
        sub = sorted(sub, key=lambda r: r["window_index"])

        # EXCLUDE LAST WINDOW FROM PLOTTING (always)
        if len(sub) > 1:
            sub = sub[:-1]

        labels = [r["window_label"] for r in sub]
        counts = [r["spike_count"] for r in sub]
        
        plt.figure(figsize=(14, 4))
        plt.plot(range(len(counts)), counts, marker="o")
        plt.title(f"Spike count per {WINDOW_MIN:.0f}-minute window — {ch}")
        plt.ylabel("Spike count")
        plt.xlabel("Time window")
        # X positions (one per window)
        x = np.arange(len(counts))
        # Label each window by its starting minute
        minute_labels = [int(i * WINDOW_MIN) for i in x]
        plt.xticks(x, minute_labels, rotation=45, ha="right")
        plt.ylim(bottom=0)
        
        
        plt.tight_layout()
        out_png = os.path.join(
            out_dir,
            f"{os.path.splitext(os.path.basename(csv_path))[0]}__{ch}__spike_count_curve.png"
        )
        
        plt.savefig(out_png, dpi=200)
        plt.close()


    # Save combined table
    out_table = pd.DataFrame(results_rows)
    out_csv = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(csv_path))[0]}__spike_counts_per_min.csv")
    out_table.to_csv(out_csv, index=False)
    print(f"[done] wrote:\n  {out_csv}\n  plots in: {out_dir}")


if __name__ == "__main__":
    main()
