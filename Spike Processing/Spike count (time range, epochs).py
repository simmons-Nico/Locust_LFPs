#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spike count (time range, epochs)

Processes raw Intan-style CSV files with:
- spike detection from time-series channels
- per-minute spike counts
- named epochs
- optional skipped time windows
- epoch shading + labels on plots
- per-minute and per-epoch CSV outputs
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


# =========================
# USER SETTINGS
# =========================

FOLDER_PATH = Path(r"C:\Users\simmons\Desktop\Exploring PSDs")
TIME_COL = "time_s"
OUT_DIR_NAME = "spike_counts_per_min"

HP_SPIKE_BAND = (300.0, 5000.0)
SPIKE_Z_THR = 6.0
POLARITY = "both"  # "neg", "pos", "both"
REFRACTORY_MS = 1.0

AMP_MIN_UV = 40.0
AMP_MAX_UV = 700.0
W_MIN_MS = None
W_MAX_MS = None

WINDOW_MIN = 1.0
WINDOW_SEC = WINDOW_MIN * 60.0

START_MIN = None
END_MIN = 50

# Use (start_min, end_min). Set to None to disable an epoch.
EPOCHS = {
    "baseline": (0, 4),
    "-70nA": (4, 22),
    "recovery": None,
}

EPOCH_COLORS = {
    "baseline": "blue",
    "-80nA": "red",
    "recovery": "gold",
}

# Use [(start_min, end_min), ...]
SKIP_WINDOWS = []

# Fixed plot y-limit. Set to None for automatic scaling.
PLOT_YMAX = 700


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
    n = len(x_hp)
    for p in peaks:
        a = p
        b = min(p + search_max, n - 1)
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

    refractory = max(int(round((REFRACTORY_MS / 1000.0) * fs)), 1)
    peaks = find_peaks_distance(z, polarity=POLARITY, thr=float(SPIKE_Z_THR), distance=refractory)
    peaks = apply_refractory(peaks, refractory)
    peaks = width_gate_indices(x_hp, peaks, fs, W_MIN_MS, W_MAX_MS, polarity=POLARITY)

    if AMP_MIN_UV is not None or AMP_MAX_UV is not None:
        amps = np.abs(x_hp[peaks]) if peaks.size else np.array([])
        keep = np.ones_like(peaks, dtype=bool)
        if AMP_MIN_UV is not None:
            keep &= amps >= float(AMP_MIN_UV)
        if AMP_MAX_UV is not None:
            keep &= amps <= float(AMP_MAX_UV)
        peaks = peaks[keep]

    spike_times = t_s[peaks] if peaks.size else np.array([], dtype=float)
    return peaks, spike_times, fs


# =========================
# TIME HELPERS
# =========================

def normalize_time_ranges(ranges_dict):
    normalized = {}
    if not ranges_dict:
        return normalized

    for name, value in ranges_dict.items():
        if value is None:
            continue
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(
                f"Epoch '{name}' must be (start_min, end_min) or None, got: {value!r}"
            )

        start_min, end_min = value
        if start_min is None or end_min is None:
            continue
        if float(end_min) <= float(start_min):
            raise ValueError(f"Epoch '{name}' has end <= start: {value!r}")

        normalized[name] = (float(start_min) * 60.0, float(end_min) * 60.0)

    return normalized


def normalize_skip_ranges(skip_windows):
    normalized = []
    for value in skip_windows:
        if value is None:
            continue
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(
                f"Each skip window must be (start_min, end_min) or None, got: {value!r}"
            )

        start_min, end_min = value
        if start_min is None or end_min is None:
            continue
        if float(end_min) <= float(start_min):
            raise ValueError(f"Skip window has end <= start: {value!r}")

        normalized.append((float(start_min) * 60.0, float(end_min) * 60.0))
    return normalized


def overlaps_any(a_start, a_end, ranges):
    for b_start, b_end in ranges:
        if a_start < b_end and a_end > b_start:
            return True
    return False


def epoch_name_for_window(a_rel, b_rel, epochs_sec):
    names = []
    for name, (e_start, e_end) in epochs_sec.items():
        if a_rel < e_end and b_rel > e_start:
            names.append(name)
    return "|".join(names) if names else ""


# =========================
# MAIN
# =========================

def main():
    csv_files = sorted(FOLDER_PATH.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {FOLDER_PATH}")

    print("\nAvailable CSV files:\n")
    for i, f in enumerate(csv_files, start=1):
        print(f"{i}: {f.name}")

    choice = int(input("\nSelect a file number: ").strip()) - 1
    if choice < 0 or choice >= len(csv_files):
        raise ValueError("Invalid file selection.")

    csv_path = csv_files[choice]
    print(f"\nLoading {csv_path.name}...")

    out_dir = FOLDER_PATH / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if TIME_COL not in df.columns:
        raise ValueError(f"Time column '{TIME_COL}' not found. Columns: {list(df.columns)}")

    t_full = df[TIME_COL].to_numpy(dtype=float)

    if START_MIN is not None or END_MIN is not None:
        rec_start_full = float(t_full[0])
        start_s_abs = rec_start_full if START_MIN is None else rec_start_full + float(START_MIN) * 60.0
        end_s_abs = float(t_full[-1]) if END_MIN is None else rec_start_full + float(END_MIN) * 60.0
        mask = (t_full >= start_s_abs) & (t_full <= end_s_abs)
        df = df.loc[mask].reset_index(drop=True)
        print(f"Applied global crop: {START_MIN} to {END_MIN} min")

    t = df[TIME_COL].to_numpy(dtype=float)
    if len(t) == 0:
        raise ValueError("No samples remain after cropping.")
    rec_start = float(t[0])

    epochs_sec = normalize_time_ranges(EPOCHS)
    skip_ranges_sec = normalize_skip_ranges(SKIP_WINDOWS)

    chan_cols = []
    for c in df.columns:
        if c == TIME_COL:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            chan_cols.append(c)

    if not chan_cols:
        raise ValueError("No numeric channel columns found besides the time column.")

    minute_rows = []
    epoch_rows = []

    for ch in chan_cols:
        x = df[ch].to_numpy(dtype=float)
        if np.any(~np.isfinite(x)):
            x = pd.Series(x).interpolate(limit_direction="both").to_numpy(dtype=float)

        peaks, spike_times, fs = detect_spikes(x, t)
        spike_times_rel = spike_times - rec_start
        print(f"[{ch}] detected spikes: {len(spike_times)} (fs≈{fs:.2f} Hz)")

        t_rel = t - rec_start
        total_dur = max(0.0, float(t_rel[-1]))
        nwin = int(math.ceil(total_dur / WINDOW_SEC)) if total_dur > 0 else 0

        channel_rows = []
        for w in range(nwin):
            a_rel = w * WINDOW_SEC
            b_rel = min((w + 1) * WINDOW_SEC, total_dur + 1e-12)

            if overlaps_any(a_rel, b_rel, skip_ranges_sec):
                continue

            cnt = int(np.sum((spike_times_rel >= a_rel) & (spike_times_rel < b_rel)))
            epoch_name = epoch_name_for_window(a_rel, b_rel, epochs_sec)

            row = {
                "channel": ch,
                "window_index": w,
                "window_start_min": a_rel / 60.0,
                "window_end_min": b_rel / 60.0,
                "window_label": f"Min {int(a_rel / 60.0)}-{int(b_rel / 60.0)}",
                "epoch": epoch_name,
                "spike_count": cnt,
                "fs_est_hz": float(fs),
            }
            minute_rows.append(row)
            channel_rows.append(row)

        for epoch_name, (e_start, e_end) in epochs_sec.items():
            usable_spikes = spike_times_rel[
                (spike_times_rel >= e_start) & (spike_times_rel < e_end)
            ]

            if usable_spikes.size:
                keep = np.ones_like(usable_spikes, dtype=bool)
                for s_start, s_end in skip_ranges_sec:
                    keep &= ~((usable_spikes >= s_start) & (usable_spikes < s_end))
                usable_spikes = usable_spikes[keep]

            effective_duration_s = e_end - e_start
            for s_start, s_end in skip_ranges_sec:
                overlap_start = max(e_start, s_start)
                overlap_end = min(e_end, s_end)
                if overlap_end > overlap_start:
                    effective_duration_s -= (overlap_end - overlap_start)

            if effective_duration_s <= 0:
                continue

            epoch_rows.append({
                "channel": ch,
                "epoch": epoch_name,
                "epoch_start_min": e_start / 60.0,
                "epoch_end_min": e_end / 60.0,
                "effective_duration_min": effective_duration_s / 60.0,
                "spike_count": int(len(usable_spikes)),
                "spikes_per_min": float(len(usable_spikes)) / (effective_duration_s / 60.0),
                "fs_est_hz": float(fs),
            })

        plot_rows = channel_rows[:-1] if len(channel_rows) > 1 else channel_rows
        counts = [r["spike_count"] for r in plot_rows]
        x_idx = np.arange(len(counts))
        minute_labels = [round(r["window_start_min"], 2) for r in plot_rows]

        plt.figure(figsize=(14, 4))
        plt.plot(x_idx, counts, marker="o", linewidth=1.5, markersize=4)

        y_text = max(counts) * 0.9 if counts and max(counts) > 0 else 1

        legend_handles = []
        for epoch_name, (e_start, e_end) in epochs_sec.items():
            start_min = e_start / 60.0
            end_min = e_end / 60.0

            start_idx = start_min / WINDOW_MIN
            end_idx = end_min / WINDOW_MIN

            color = EPOCH_COLORS.get(epoch_name, "gray")
            plt.axvspan(start_idx, end_idx, alpha=0.2, color=color)

            mid_idx = (start_idx + end_idx) / 2
            plt.text(
                mid_idx,
                y_text,
                epoch_name,
                ha="center",
                va="top",
                fontsize=9,
                color=color,
                fontweight="bold",
            )

            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.2, edgecolor="none", label=epoch_name)
            )

        if legend_handles:
            plt.legend(
                handles=legend_handles,
                title="Epochs",
                loc="upper right",
                frameon=True,
            )

        if PLOT_YMAX is None:
            y_max = max(counts) if counts else 1
            plt.ylim(0, max(1, y_max * 1.2))
        else:
            plt.ylim(0, PLOT_YMAX)

        plt.title(f"Spike count 1min bins")
        plt.xlabel("Window start (min)")
        plt.ylabel("Spike count")
        plt.xticks(x_idx, minute_labels, rotation=45, ha="right")
        plt.grid(False)
        plt.tight_layout()

        output_png = out_dir / f"{csv_path.stem}__{ch}__spike_count_curve.png"
        plt.savefig(output_png, dpi=300)
        plt.close()

    minute_df = pd.DataFrame(minute_rows)
    epoch_df = pd.DataFrame(epoch_rows)

    minute_csv = out_dir / f"{csv_path.stem}__spike_counts_per_min.csv"
    epoch_csv = out_dir / f"{csv_path.stem}__spike_counts_by_epoch.csv"

    minute_df.to_csv(minute_csv, index=False)
    epoch_df.to_csv(epoch_csv, index=False)

    print(f"\nSaved per-minute spike counts to:\n{minute_csv}")
    print(f"Saved per-epoch summary to:\n{epoch_csv}")
    print(f"Saved channel plots to:\n{out_dir}")


if __name__ == "__main__":
    main()
