#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # comment out for interactive
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, sosfiltfilt, find_peaks, spectrogram

# ================== USER SETTINGS ==================
CSV_DIR      = r"C:\Users\simmons\Desktop\Exploring PSDs"
CSV_GLOB     = "*.csv"
OUT_DIR_NAME = "psd_plots_4min"

# Time & signal column inference
TIME_COLS    = ["time_s", "time", "t", "seconds"]

# Channel selection:
# - Set CHANNEL_INCLUDE = None or [] to include all channels.
# - You MAY type comma-separated strings and it will split them for you.
CHANNEL_INCLUDE = None          # e.g. ["A-008", "A-012"] or "A-008, A-012"
CHANNEL_EXCLUDE = []            # e.g. ["A-003"] or "A-003, A-004"

# ---- Middle window ----
MIDDLE_WINDOW_SEC = 600.0   # 4 minutes

# ---- Spike amplitude gating ----
POLARITY    = "neg"      # "neg", "pos", or "both"
AMP_MIN_UV  = 40.0       # based on your noise histogram
AMP_MAX_UV  = 700.0      # artifact clamp; or set None

# ---- Welch PSD ----
TARGET_NPERSEG_SEC = 2.0
NOVERLAP_RATIO     = 0.5
FMAX               = 100  # Hz

# Band-power bars (linear PSD integration)
LOW_BANDS = [(1,4), (4,8), (8,13), (13,20)]  # Hz
DB_RANGE  = None

# ---- NEW: Sliding-window bandpower vs time ----
BP_WIN_SEC            = 10.0   # window length (s)
BP_STEP_SEC           = 1.0    # step size (s)
BP_TARGET_NPERSEG_SEC = 2.0    # Welch nperseg inside each window (s)
BP_NOVERLAP_RATIO     = 0.5
BP_FMAX               = FMAX
BP_NORMALIZE_TOTAL    = False  # if True: divide each band by total power in [0, BP_FMAX]

# ---- NEW: Time-frequency spectrogram ----
SPEC_NPERSEG_SEC      = 2.0
SPEC_NOVERLAP_RATIO   = 0.75
SPEC_FMAX             = FMAX
SPEC_DB_RANGE         = None   # e.g. (-120, -60) or None for auto

# -------- Spike analysis --------
HP_SPIKE_BAND   = (300.0, 5000.0)   # for spike detection / waveforms
SPIKE_Z_THR     = 6.0               # MAD-based robust z threshold
REFRACTORY_MS   = 1.0               # ms
PRE_MS          = 0.6               # ms for waveform extraction
POST_MS         = 1.0               # ms for waveform extraction
RATE_BIN_S      = 1.0               # firing-rate bin
# ================================================


# ----------------- Helpers -----------------
def _parse_channel_list(x):
    """
    Allow these forms:
      None
      []
      ["A-008", "A-012"]
      "A-008, A-012"
      ["A-008, A-012"]   (your current case)
    Returns: list[str] or None
    """
    if x is None:
        return None
    if isinstance(x, str):
        parts = [p.strip() for p in x.split(",") if p.strip()]
        return parts if parts else None
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            if item is None:
                continue
            if isinstance(item, str):
                # split each list entry by commas too
                out.extend([p.strip() for p in item.split(",") if p.strip()])
        return out if out else None
    return None

def find_time_column(df: pd.DataFrame) -> str:
    for c in TIME_COLS:
        if c in df.columns:
            return c
    # fallback: first strictly increasing numeric column
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
        raise ValueError("Time column invalid (nonpositive/non-finite diffs).")
    return 1.0 / dt

def list_csvs(directory, pattern):
    return sorted(glob.glob(os.path.join(directory, pattern)))

def list_signal_channels(df: pd.DataFrame, time_col: str) -> list[str]:
    """Return all numeric columns except time, applying include/exclude filters."""
    chans = []
    for c in df.columns:
        if c == time_col:
            continue
        if np.issubdtype(df[c].dtype, np.number):
            chans.append(c)

    include = _parse_channel_list(CHANNEL_INCLUDE)
    exclude = _parse_channel_list(CHANNEL_EXCLUDE)

    if include:
        include_set = set(include)
        chans = [c for c in chans if c in include_set]

    if exclude:
        exclude_set = set(exclude)
        chans = [c for c in chans if c not in exclude_set]

    return chans

def select_middle_window(t: np.ndarray, x: np.ndarray, window_sec: float):
    """Return t,x restricted to the middle window_sec of the recording."""
    t0, t1 = float(t[0]), float(t[-1])
    tmid = 0.5 * (t0 + t1)
    half = window_sec / 2.0
    mask = (t >= (tmid - half)) & (t <= (tmid + half))
    return t[mask], x[mask]

def welch_psd(x, fs, nperseg, noverlap, fmax):
    f, Pxx = welch(
        x, fs=fs, nperseg=nperseg, noverlap=noverlap,
        window="hann", detrend="constant", scaling="density", average="mean"
    )
    if fmax is not None:
        keep = f <= fmax
        f, Pxx = f[keep], Pxx[keep]
    Pxx_db = 10*np.log10(Pxx + np.finfo(float).eps)
    return f, Pxx, Pxx_db

def bandpower_linear(f, Pxx_linear, band):
    lo, hi = band
    keep = (f >= lo) & (f < hi)
    if not np.any(keep):
        return np.nan
    return np.trapz(Pxx_linear[keep], f[keep])

# ---------- NEW: sliding-window bandpower ----------
def sliding_bandpower_time(t, x, fs, bands, win_sec, step_sec,
                           target_nperseg_sec=2.0, noverlap_ratio=0.5,
                           fmax=100, normalize_total=False):
    """
    Compute bandpower vs time using Welch PSD within sliding windows.

    Returns:
      centers_t : (K,) window-center times in seconds (same units as t)
      bp        : (K, nbands) bandpowers (µV^2) or normalized fraction
    """
    n = len(x)
    win_n = int(round(win_sec * fs))
    step_n = int(round(step_sec * fs))
    if win_n < 8 or step_n < 1 or n < win_n:
        return np.array([]), np.zeros((0, len(bands)))

    # Welch params inside each window
    nperseg = max(16, int(round(target_nperseg_sec * fs)))
    nperseg = min(nperseg, win_n)
    noverlap = int(round(nperseg * noverlap_ratio))
    noverlap = min(noverlap, nperseg - 1) if nperseg > 1 else 0

    centers = []
    bp_rows = []
    for start in range(0, n - win_n + 1, step_n):
        seg = x[start:start + win_n]
        f, Pxx_lin, _ = welch_psd(seg, fs, nperseg, noverlap, fmax=fmax)

        band_vals = np.array([bandpower_linear(f, Pxx_lin, b) for b in bands], dtype=float)

        if normalize_total:
            # total power in [0, fmax] (or all f returned)
            total = np.trapz(Pxx_lin, f) if f.size > 1 else np.nan
            if np.isfinite(total) and total > 0:
                band_vals = band_vals / total
            else:
                band_vals[:] = np.nan

        center_idx = start + win_n // 2
        centers.append(float(t[center_idx]))
        bp_rows.append(band_vals)

    return np.asarray(centers), np.vstack(bp_rows) if bp_rows else np.zeros((0, len(bands)))

# ---------- filtering & spikes ----------
def _safe_band_for_fs(low, high, fs, margin=0.95):
    ny = fs * 0.5
    hi = min(high, ny*margin) if high is not None else None
    lo = max(low, 0.001) if low is not None else None
    if hi is not None and lo is not None and hi <= lo:
        hi = min(lo*1.2, ny*margin)
    return lo, hi

def design_sos_bandpass(low, high, fs, order=4):
    low, high = _safe_band_for_fs(low, high, fs)
    if low is None and high is None:
        raise ValueError("Both low and high are None.")
    if low is None:
        wn = high / (fs*0.5)
        sos = butter(order, wn, btype="lowpass", output="sos")
    elif high is None:
        wn = low / (fs*0.5)
        sos = butter(order, wn, btype="highpass", output="sos")
    else:
        wn = [low/(fs*0.5), high/(fs*0.5)]
        sos = butter(order, wn, btype="bandpass", output="sos")
    return sos

def band_filter(x, fs, band, order=4):
    sos = design_sos_bandpass(band[0], band[1], fs, order=order)
    return sosfiltfilt(sos, x)

def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad)

def detect_spikes(xhp, fs, zthr=SPIKE_Z_THR, refr_ms=REFRACTORY_MS):
    z = robust_z(xhp)
    distance = max(1, int(round((refr_ms/1000.0)*fs)))
    idx, _ = find_peaks(np.abs(z), height=zthr, distance=distance)
    return np.asarray(idx, dtype=int)

def extract_waveforms(xhp, idx, fs, pre_ms=PRE_MS, post_ms=POST_MS):
    pre  = int(round(pre_ms*1e-3*fs))
    post = int(round(post_ms*1e-3*fs))
    valid = idx[(idx - pre >= 0) & (idx + post < len(xhp))]
    if valid.size == 0:
        return valid, np.zeros((0, pre+post))
    W = np.stack([xhp[i-pre:i+post] for i in valid], axis=0)
    return valid, W

def gate_by_amplitude(valid_idx, W):
    """Return (valid_idx_gated, W_gated, amp_uv, keep_mask)."""
    if W.shape[0] == 0:
        return valid_idx, W, np.array([]), np.array([], dtype=bool)

    if POLARITY == "neg":
        amp_uv = np.abs(W.min(axis=1))
    elif POLARITY == "pos":
        amp_uv = np.abs(W.max(axis=1))
    else:  # "both"
        amp_uv = np.max(np.abs(W), axis=1)

    keep = amp_uv >= AMP_MIN_UV
    if AMP_MAX_UV is not None:
        keep &= amp_uv <= AMP_MAX_UV

    return valid_idx[keep], W[keep], amp_uv, keep

def autocorrelogram(times_s, bin_ms=1.0, max_lag_ms=100.0):
    if times_s.size < 2:
        return None, None
    t0, t1 = times_s.min(), times_s.max()
    edges = np.arange(t0, t1 + bin_ms/1000.0, bin_ms/1000.0)
    counts, _ = np.histogram(times_s, bins=edges)
    ac = np.correlate(counts - counts.mean(), counts - counts.mean(), mode='full')
    lags = np.arange(-len(counts)+1, len(counts)) * (bin_ms/1000.0)
    keep = np.abs(lags) <= (max_lag_ms/1000.0)
    return lags[keep], ac[keep]

def _nan_interpolate(x: np.ndarray) -> np.ndarray:
    x = x.astype(float, copy=True)
    if np.isfinite(x).all():
        return x
    idx = np.arange(len(x), dtype=float)
    good = np.isfinite(x)
    if good.sum() >= 2:
        first, last = np.where(good)[0][0], np.where(good)[0][-1]
        x[:first] = x[first]
        x[last+1:] = x[last]
        x[~good] = np.interp(idx[~good], idx[good], x[good])
        return x
    return x  # leave as-is; caller can skip


def main():
    out_root = os.path.join(CSV_DIR, OUT_DIR_NAME)
    os.makedirs(out_root, exist_ok=True)

    csv_files = list_csvs(CSV_DIR, CSV_GLOB)
    if not csv_files:
        raise SystemExit(f"No CSVs found in {CSV_DIR!r} with pattern {CSV_GLOB!r}")

    # Overlay across ALL (file,channel)
    overlay_freqs, overlay_psds_db, overlay_labels = [], [], []
    global_db_min, global_db_max = np.inf, -np.inf

    for path in csv_files:
        base = os.path.splitext(os.path.basename(path))[0]
        out_dir_file = os.path.join(out_root, base)
        os.makedirs(out_dir_file, exist_ok=True)

        df = pd.read_csv(path)
        time_col = find_time_column(df)
        t_full = df[time_col].to_numpy(dtype=float)

        channels = list_signal_channels(df, time_col)
        if not channels:
            print(f"[WARN] {base}: no numeric channels found after include/exclude filtering.")
            continue

        for ch in channels:
            label = f"{base}__{ch}"
            out_dir = os.path.join(out_dir_file, ch)
            os.makedirs(out_dir, exist_ok=True)

            x_full = pd.to_numeric(df[ch], errors="coerce").to_numpy(float)
            x_full = _nan_interpolate(x_full)
            if not np.isfinite(x_full).all():
                print(f"[WARN] {label}: insufficient finite samples; skipping.")
                continue

            # middle window
            t, x = select_middle_window(t_full, x_full, MIDDLE_WINDOW_SEC)
            if t.size < 2:
                print(f"[WARN] {label}: middle window empty")
                continue
            win_len = float(t[-1] - t[0])
            if win_len < 0.9 * MIDDLE_WINDOW_SEC:
                print(f"[WARN] {label}: recording too short for 4 min window ({win_len:.1f}s)")
                continue

            fs = infer_fs(t)

            # ================= PSD =================
            nperseg = max(64, int(round(TARGET_NPERSEG_SEC * fs)))
            noverlap = int(round(nperseg * NOVERLAP_RATIO))
            f, Pxx_lin, Pxx_db = welch_psd(x, fs, nperseg, noverlap, fmax=FMAX)

            finite_db = Pxx_db[np.isfinite(Pxx_db)]
            if finite_db.size:
                global_db_min = min(global_db_min, float(finite_db.min()))
                global_db_max = max(global_db_max, float(finite_db.max()))

            # PSD (dB)
            plt.figure(figsize=(9,5))
            plt.plot(f, Pxx_db, lw=1.5)
            plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD (dB(µV²/Hz))")
            plt.title(f"Welch PSD (dB) — {label} | middle {MIDDLE_WINDOW_SEC:.0f}s | fs≈{fs:.3f} Hz")
            plt.grid(True, alpha=0.3)
            if FMAX is not None: plt.xlim(0, FMAX)
            if DB_RANGE is not None: plt.ylim(*DB_RANGE)
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{label}_psd_db.png"), dpi=150); plt.close()

            # PSD (linear)
            plt.figure(figsize=(9,5))
            plt.plot(f, Pxx_lin, lw=1.5)
            plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD (µV²/Hz)")
            plt.title(f"Welch PSD (linear) — {label} | middle {MIDDLE_WINDOW_SEC:.0f}s | fs≈{fs:.3f} Hz")
            plt.grid(True, alpha=0.3)
            if FMAX is not None: plt.xlim(0, FMAX)
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{label}_psd_linear.png"), dpi=150); plt.close()

            # Band-power bars
            band_vals = [bandpower_linear(f, Pxx_lin, band) for band in LOW_BANDS]
            band_labels = [f"{lo}-{hi} Hz" for (lo,hi) in LOW_BANDS]
            plt.figure(figsize=(8,5))
            plt.bar(band_labels, band_vals)
            plt.ylabel("Band power (µV²)")
            plt.title(f"Band-averaged power — {label} | middle {MIDDLE_WINDOW_SEC:.0f}s")
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{label}_bandpower_bars.png"), dpi=150); plt.close()

            # ================= NEW: Sliding-window bandpower vs time =================
            t_bp, bp = sliding_bandpower_time(
                t, x, fs,
                bands=LOW_BANDS,
                win_sec=BP_WIN_SEC,
                step_sec=BP_STEP_SEC,
                target_nperseg_sec=BP_TARGET_NPERSEG_SEC,
                noverlap_ratio=BP_NOVERLAP_RATIO,
                fmax=BP_FMAX,
                normalize_total=BP_NORMALIZE_TOTAL
            )
            if t_bp.size:
                plt.figure(figsize=(10,4))
                for i, (lo, hi) in enumerate(LOW_BANDS):
                    plt.plot(t_bp, bp[:, i], lw=1.5, label=f"{lo}-{hi} Hz")
                plt.xlabel("Time (s)")
                plt.ylabel("Normalized band power" if BP_NORMALIZE_TOTAL else "Band power (µV²)")
                plt.title(f"Sliding-window bandpower — {label} | win={BP_WIN_SEC:.1f}s step={BP_STEP_SEC:.1f}s")
                plt.grid(True, alpha=0.3)
                plt.legend(loc="best", fontsize=8, ncol=2)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"{label}_bandpower_timeseries.png"), dpi=150)
                plt.close()

            # ================= NEW: Time-frequency spectrogram =================
            spec_nperseg = max(16, int(round(SPEC_NPERSEG_SEC * fs)))
            spec_nperseg = min(spec_nperseg, len(x))
            spec_noverlap = int(round(spec_nperseg * SPEC_NOVERLAP_RATIO))
            spec_noverlap = min(spec_noverlap, spec_nperseg - 1) if spec_nperseg > 1 else 0

            f_s, t_s, Sxx = spectrogram(
                x,
                fs=fs,
                window="hann",
                nperseg=spec_nperseg,
                noverlap=spec_noverlap,
                detrend="constant",
                scaling="density",
                mode="psd"
            )
            if SPEC_FMAX is not None:
                keep_f = f_s <= SPEC_FMAX
                f_s = f_s[keep_f]
                Sxx = Sxx[keep_f, :]

            # convert to dB (µV²/Hz)
            Sxx_db = 10.0 * np.log10(Sxx + np.finfo(float).eps)

            # Align spectrogram time axis to your actual time vector (offset by window start)
            t_s_abs = t_s + float(t[0])

            plt.figure(figsize=(10,5))
            plt.pcolormesh(t_s_abs, f_s, Sxx_db, shading="auto")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.title(f"Spectrogram (dB) — {label} | nperseg={spec_nperseg} samples | fs≈{fs:.3f} Hz")
            if SPEC_DB_RANGE is not None:
                plt.clim(*SPEC_DB_RANGE)
            plt.colorbar(label="PSD (dB(µV²/Hz))")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{label}_spectrogram_db.png"), dpi=150)
            plt.close()

            # ---------- SPIKE ANALYSIS ----------
            if fs < 2000:
                print(f"[WARN] {label}: fs≈{fs:.1f} Hz < 2 kHz: spike detection under-resolved.")

            x_hp = band_filter(x, fs, HP_SPIKE_BAND, order=4)

            spike_idx = detect_spikes(x_hp, fs, zthr=SPIKE_Z_THR, refr_ms=REFRACTORY_MS)
            valid_idx, W = extract_waveforms(x_hp, spike_idx, fs, PRE_MS, POST_MS)
            valid_idx, W, amp_uv, keep_mask = gate_by_amplitude(valid_idx, W)

            gated_times_s = t[valid_idx] if valid_idx.size else np.array([])

            print(f"[{label}] Kept {len(valid_idx)}/{len(spike_idx)} spikes (AMP_MIN_UV={AMP_MIN_UV}, AMP_MAX_UV={AMP_MAX_UV})")

            # Save spike times
            if gated_times_s.size:
                np.savetxt(os.path.join(out_dir, f"{label}_spike_times_s_GATED.txt"),
                           gated_times_s, fmt="%.6f")

            # Raster
            if gated_times_s.size:
                plt.figure(figsize=(10, 2.5))
                plt.eventplot(gated_times_s, lineoffsets=1, linelengths=0.8, colors='k')
                plt.yticks([]); plt.xlabel("Time (s)")
                plt.title(f"Spike raster — {label} (n={gated_times_s.size})")
                plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{label}_raster.png"), dpi=150); plt.close()

            # Firing rate
            if gated_times_s.size:
                bin_s = RATE_BIN_S
                edges = np.arange(gated_times_s.min(), gated_times_s.max() + bin_s, bin_s)
                counts, _ = np.histogram(gated_times_s, bins=edges)
                centers = (edges[:-1] + edges[1:]) / 2
                plt.figure(figsize=(10,3))
                plt.plot(centers, counts / bin_s, lw=1.5)
                plt.xlabel("Time (s)"); plt.ylabel("Rate (spikes/s)")
                plt.title(f"Firing rate — {label} (bin={bin_s:.1f}s)")
                plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{label}_rate.png"), dpi=150); plt.close()

            # ISI hist
            if gated_times_s.size >= 2:
                isi_ms = np.diff(gated_times_s) * 1000.0
                plt.figure(figsize=(7,4))
                plt.hist(isi_ms, bins=np.linspace(0, max(50, np.percentile(isi_ms, 99)), 60), edgecolor="none")
                plt.xlabel("ISI (ms)"); plt.ylabel("Count")
                plt.title(f"ISI histogram — {label}")
                plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{label}_isi.png"), dpi=150); plt.close()

                bins = np.logspace(np.log10(max(0.5, isi_ms.min())), np.log10(max(1000, isi_ms.max())), 60)
                plt.figure(figsize=(7,4))
                plt.hist(isi_ms, bins=bins, edgecolor="none")
                plt.xscale('log'); plt.xlabel("ISI (ms, log)"); plt.ylabel("Count")
                plt.title(f"ISI (log) — {label}")
                plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{label}_isi_log.png"), dpi=150); plt.close()

            # Autocorrelogram
            if gated_times_s.size:
                lags, ac = autocorrelogram(gated_times_s, bin_ms=1.0, max_lag_ms=100.0)
                if lags is not None:
                    plt.figure(figsize=(7,4))
                    plt.plot(lags*1000, ac, lw=1.2)
                    plt.xlabel("Lag (ms)"); plt.ylabel("Auto-correlation (a.u.)")
                    plt.title(f"Autocorrelogram — {label}")
                    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{label}_acg.png"), dpi=150); plt.close()

            # Waveform mean±SEM
            if W.shape[0] >= 1:
                t_wf_ms = (np.arange(W.shape[1]) / fs) * 1000.0
                mu = W.mean(0)
                sem = W.std(0) / max(np.sqrt(W.shape[0]), 1)
                plt.figure(figsize=(6,4))
                plt.plot(t_wf_ms, mu, lw=2)
                plt.fill_between(t_wf_ms, mu - sem, mu + sem, alpha=0.3)
                plt.xlabel("Time (ms)"); plt.ylabel("µV")
                plt.title(f"Spike waveform mean±SEM — {label} (n={W.shape[0]})")
                plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"{label}_waveform.png"), dpi=150); plt.close()

            # Collect for GLOBAL overlay
            overlay_freqs.append(f)
            overlay_psds_db.append(Pxx_db)
            overlay_labels.append(label)

    # ---------- ONE global overlay PSD (all channels across all files) ----------
    if overlay_freqs:
        f_common = overlay_freqs[np.argmin([len(fi) for fi in overlay_freqs])]
        plt.figure(figsize=(11,7))
        for f_i, P_i, label in zip(overlay_freqs, overlay_psds_db, overlay_labels):
            y = P_i if np.array_equal(f_i, f_common) else np.interp(f_common, f_i, P_i)
            plt.plot(f_common, y, lw=1.2, label=label)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (dB(µV²/Hz))")
        plt.title(f"GLOBAL Welch PSD overlay — all files × all channels (middle {MIDDLE_WINDOW_SEC:.0f}s)")
        plt.grid(True, alpha=0.3)
        if FMAX is not None: plt.xlim(0, FMAX)
        if DB_RANGE is None and np.isfinite(global_db_min) and np.isfinite(global_db_max):
            pad = 3.0
            plt.ylim(global_db_min - pad, global_db_max + pad)
        elif DB_RANGE is not None:
            plt.ylim(*DB_RANGE)
        plt.legend(loc="best", fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, "GLOBAL_all_channels_psd_overlay_dB.png"), dpi=150)
        plt.close()

    print(f"✅ Done. Plots saved under: {out_root}")

if __name__ == "__main__":
    main()