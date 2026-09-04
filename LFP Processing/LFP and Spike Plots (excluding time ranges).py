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
OUT_DIR_NAME = "psd_plots"

# Time & signal column inference
TIME_COLS    = ["time_s", "time", "t", "seconds"]
SIGNAL_COL   = None  # if None => auto-pick first numeric non-time column

POLARITY    = "neg"      # assuming extracellular spikes
AMP_MIN_UV  = 50.0       # <-- CHANGE THIS
AMP_MAX_UV  = 700.0      # or 500.0 if you prefer
SPIKE_Z_THR = 6.0        # keep as-is


# Welch PSD
TARGET_NPERSEG_SEC = 2.0
NOVERLAP_RATIO     = 0.5
FMAX               = 100  # Hz

# Band-power bars (linear PSD integration)
LOW_BANDS = [(1,4), (4,8), (8,13), (13,20)]  # Hz
DB_RANGE  = None

# -------- Spike analysis --------
HP_SPIKE_BAND   = (300.0, 5000.0)   # for spike detection / waveforms
LFP_BAND        = (0.1, 300.0)      # for STA
SPIKE_Z_THR     = 6.0               # MAD-based robust z threshold
REFRACTORY_MS   = 1.0               # ms
PRE_MS          = 0.6               # ms for waveform extraction
POST_MS         = 1.0               # ms for waveform extraction
RATE_BIN_S      = 1.0               # firing-rate bin
STA_WIN_SEC     = 0.200             # half-window for STA (±)
DO_ST_SPECTRO   = True              # spike-triggered spectrogram of LFP
SPECTRO_FMAX    = 200.0             # cap spectrogram to this frequency
# ================================================


# ----------------- Helpers -----------------
def find_time_column(df: pd.DataFrame) -> str:
    for c in TIME_COLS:
        if c in df.columns:
            return c
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

def pick_signal_col(df: pd.DataFrame, time_col: str) -> str:
    if SIGNAL_COL is not None:
        if SIGNAL_COL not in df.columns:
            raise ValueError(f"Requested SIGNAL_COL '{SIGNAL_COL}' not found.")
        return SIGNAL_COL
    for c in df.columns:
        if c != time_col and np.issubdtype(df[c].dtype, np.number):
            return c
    raise ValueError("No numeric signal column found besides time.")

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

def list_csvs(directory, pattern):
    return sorted(glob.glob(os.path.join(directory, pattern)))

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

def compute_sta(lfp, spike_idx, half_win_samp):
    snippets = []
    N = len(lfp)
    for idx in spike_idx:
        a = idx - half_win_samp
        b = idx + half_win_samp + 1
        if a < 0 or b > N:
            continue
        snippets.append(lfp[a:b])
    if not snippets:
        return None, 0
    stack = np.vstack(snippets)
    return np.nanmean(stack, axis=0), stack.shape[0]

def autocorrelogram(times_s, bin_ms=1.0, max_lag_ms=100.0):
    if times_s.size < 2:
        return None, None
    # Build spike train at 1 ms resolution (safe for LFP/ST analyses)
    t0, t1 = times_s.min(), times_s.max()
    edges = np.arange(t0, t1 + bin_ms/1000.0, bin_ms/1000.0)
    counts, _ = np.histogram(times_s, bins=edges)
    ac = np.correlate(counts - counts.mean(), counts - counts.mean(), mode='full')
    lags = np.arange(-len(counts)+1, len(counts)) * (bin_ms/1000.0)
    keep = np.abs(lags) <= (max_lag_ms/1000.0)
    return lags[keep], ac[keep]

def prompt_time_exclusion(t: np.ndarray, base: str):
    """
    Given the time vector t (in seconds) for one recording,
    ask the user which time ranges should be *ignored* (excluded)
    from all analyses.

    Returns:
        keep_mask (bool array) or None if everything is excluded.
    """
    if t.size == 0:
        return None

    t0 = float(t[0])
    t1 = float(t[-1])
    dur = t1 - t0

    print("\n" + "-"*60)
    print(f"File: {base}")
    print(f"Time span: {t0:.3f} s  →  {t1:.3f} s   (duration ≈ {dur:.3f} s)")
    print("If you want to EXCLUDE parts of this recording,")
    print("enter time ranges in seconds as:  start1-end1, start2-end2, ...")
    print("Example:  0-10, 55-60   (to ignore 0–10 s and 55–60 s)")
    print("Press ENTER with nothing typed to keep the whole recording.")
    resp = input("Time ranges to IGNORE (in seconds): ").strip()

    # Start with 'keep everything'
    keep = np.ones_like(t, dtype=bool)

    if resp:
        for chunk in resp.split(","):
            part = chunk.strip()
            if not part:
                continue
            try:
                s_str, e_str = part.split("-")
                s = float(s_str)
                e = float(e_str)
            except ValueError:
                print(f"  !! Could not parse '{part}' as 'start-end'; skipping that piece.")
                continue

            # Make sure s <= e
            if e < s:
                s, e = e, s

            bad = (t >= s) & (t <= e)
            if not np.any(bad):
                print(f"  (No samples fell into {s:.3f}–{e:.3f} s; nothing removed.)")
            else:
                n_bad = int(bad.sum())
                print(f"  Excluding {n_bad} samples in {s:.3f}–{e:.3f} s.")
                keep[bad] = False

    if not np.any(keep):
        print("  !! All samples were excluded for this file; skipping it.")
        return None

    n_keep = int(keep.sum())
    print(f"  Keeping {n_keep} samples out of {t.size}.")
    print("-"*60 + "\n")
    return keep


# -------------------------------------------

def main():
    out_root = os.path.join(CSV_DIR, OUT_DIR_NAME)
    os.makedirs(out_root, exist_ok=True)

    csv_files = list_csvs(CSV_DIR, CSV_GLOB)
    if not csv_files:
        raise SystemExit(f"No CSVs found in {CSV_DIR!r} with pattern {CSV_GLOB!r}")

    overlay_freqs, overlay_psds_db, overlay_labels = [], [], []
    global_db_min, global_db_max = np.inf, -np.inf

    for path in csv_files:
        base = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(out_root, base)
        os.makedirs(out_dir, exist_ok=True)

        # Load & infer
                # Load & infer
        df = pd.read_csv(path)
        time_col = find_time_column(df)
        t = df[time_col].to_numpy(dtype=float)
        fs = infer_fs(t)
        sig_col = pick_signal_col(df, time_col)
        x = pd.to_numeric(df[sig_col], errors="coerce").to_numpy(float)

        # NaN interpolate signal for PSD/LFP path
        if not np.isfinite(x).all():
            idx = np.arange(len(x), dtype=float)
            good = np.isfinite(x)
            if good.sum() >= 2:
                first, last = np.where(good)[0][0], np.where(good)[0][-1]
                x[:first] = x[first]; x[last+1:] = x[last]
                x[~good] = np.interp(idx[~good], idx[good], x[good])
            else:
                print(f"Skipping {path}: insufficient finite samples."); continue

        # -------- NEW: ask which time ranges to IGNORE --------
        keep_mask = prompt_time_exclusion(t, base)
        if keep_mask is None:
            # user excluded everything; skip this file
            continue

        t = t[keep_mask]
        x = x[keep_mask]
        # ------------------------------------------------------

        # Welch
        nperseg = max(64, int(round(TARGET_NPERSEG_SEC * fs)))
        noverlap = int(round(nperseg * NOVERLAP_RATIO))
        f, Pxx_lin, Pxx_db = welch_psd(x, fs, nperseg, noverlap, fmax=FMAX)

        # Track global dB span
        finite_db = Pxx_db[np.isfinite(Pxx_db)]
        if finite_db.size:
            global_db_min = min(global_db_min, float(finite_db.min()))
            global_db_max = max(global_db_max, float(finite_db.max()))

        # PSD (dB)
        plt.figure(figsize=(9,5))
        plt.plot(f, Pxx_db, lw=1.5)
        plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD (dB(µV²/Hz))")
        plt.title(f"Welch PSD (dB) — {base} | fs≈{fs:.3f} Hz | col={sig_col}")
        plt.grid(True, alpha=0.3)
        if FMAX is not None: plt.xlim(0, FMAX)
        if DB_RANGE is not None: plt.ylim(*DB_RANGE)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{base}_psd_db.png"), dpi=150); plt.close()

        # PSD (linear)
        plt.figure(figsize=(9,5))
        plt.plot(f, Pxx_lin, lw=1.5)
        plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD (µV²/Hz)")
        plt.title(f"Welch PSD (linear) — {base} | fs≈{fs:.3f} Hz | col={sig_col}")
        plt.grid(True, alpha=0.3)
        if FMAX is not None: plt.xlim(0, FMAX)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{base}_psd_linear.png"), dpi=150); plt.close()

        # Band-power bars
        band_vals = [bandpower_linear(f, Pxx_lin, band) for band in LOW_BANDS]
        band_labels = [f"{lo}-{hi} Hz" for (lo,hi) in LOW_BANDS]
        plt.figure(figsize=(8,5))
        plt.bar(band_labels, band_vals)
        plt.ylabel("Band power (µV²)"); plt.title(f"Band-averaged power — {base}")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{base}_bandpower_bars.png"), dpi=150); plt.close()

        # ---------- SPIKE ANALYSIS ----------
        # Choose path: use column directly if it's a spikeband column
        looks_like_spikeband = sig_col.endswith("_spike")
        if looks_like_spikeband:
            x_hp = x.copy()
        else:
            if fs < 2000:
                print(f"[{base}] fs≈{fs:.1f} Hz < 2 kHz: spike detection is under-resolved.")
            x_hp = band_filter(x, fs, HP_SPIKE_BAND, order=4)

        # Detect spikes
        spike_idx = detect_spikes(x_hp, fs, zthr=SPIKE_Z_THR, refr_ms=REFRACTORY_MS)
        spike_times_s = t[spike_idx] if spike_idx.size else np.array([])

        # Save spike times if any
        if spike_times_s.size:
            np.savetxt(os.path.join(out_dir, f"{base}_spike_times_s.txt"), spike_times_s, fmt="%.6f")

        # Waveforms (only meaningful if fs is high)
        valid_idx, W = extract_waveforms(x_hp, spike_idx, fs, PRE_MS, POST_MS)
        
        # ---------------- Amplitude gating ----------------
        # Determine peak amplitude per spike (in µV)
        if POLARITY == "neg":
            peak_uv = W.min(axis=1)          # negative peaks
            amp_uv = np.abs(peak_uv)
        elif POLARITY == "pos":
            peak_uv = W.max(axis=1)
            amp_uv = np.abs(peak_uv)
        else:  # "both"
            amp_uv = np.max(np.abs(W), axis=1)

        keep = amp_uv >= AMP_MIN_UV

        if AMP_MAX_UV is not None:
            keep &= amp_uv <= AMP_MAX_UV

        # Apply gate
            valid_idx = valid_idx[keep]
        W = W[keep]

        print(
            f"[Amplitude gate] Kept {keep.sum()}/{len(keep)} spikes "
            f"(AMP_MIN_UV={AMP_MIN_UV}, AMP_MAX_UV={AMP_MAX_UV})"
        )
        # --------------------------------------------------
       
        
        # LFP for STA
        x_lfp = band_filter(x, fs, LFP_BAND, order=4)
        half_win = int(round(STA_WIN_SEC * fs))
        sta, n_used = compute_sta(x_lfp, valid_idx, half_win)

        # ---- Plots: spike QC ----
        # Raster
        if spike_times_s.size:
            plt.figure(figsize=(10, 2.5))
            plt.eventplot(spike_times_s, lineoffsets=1, linelengths=0.8, colors='k')
            plt.yticks([]); plt.xlabel("Time (s)"); plt.title(f"Spike raster — {base} (n={spike_times_s.size})")
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{base}_raster.png"), dpi=150); plt.close()

        # Firing rate over time
        if spike_times_s.size:
            bin_s = RATE_BIN_S
            edges = np.arange(spike_times_s.min(), spike_times_s.max() + bin_s, bin_s)
            counts, _ = np.histogram(spike_times_s, bins=edges)
            centers = (edges[:-1] + edges[1:]) / 2
            plt.figure(figsize=(10,3))
            plt.plot(centers, counts / bin_s, lw=1.5)
            plt.xlabel("Time (s)"); plt.ylabel("Rate (spikes/s)")
            plt.title(f"Firing rate — {base} (bin={bin_s:.1f}s)")
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{base}_rate.png"), dpi=150); plt.close()

        # ISI histogram
        if spike_times_s.size >= 2:
            isi_ms = np.diff(spike_times_s) * 1000.0
            plt.figure(figsize=(7,4))
            plt.hist(isi_ms, bins=np.linspace(0, max(50, np.percentile(isi_ms, 99)), 60), edgecolor="none")
            plt.xlabel("ISI (ms)"); plt.ylabel("Count")
            plt.title(f"ISI histogram — {base}")
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{base}_isi.png"), dpi=150); plt.close()

            # Log-binned ISI (optional)
            bins = np.logspace(np.log10(max(0.5, isi_ms.min())), np.log10(max(1000, isi_ms.max())), 60)
            plt.figure(figsize=(7,4))
            plt.hist(isi_ms, bins=bins, edgecolor="none")
            plt.xscale('log'); plt.xlabel("ISI (ms, log)"); plt.ylabel("Count")
            plt.title(f"ISI (log) — {base}")
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{base}_isi_log.png"), dpi=150); plt.close()

        # Autocorrelogram
        if spike_times_s.size:
            lags, ac = autocorrelogram(spike_times_s, bin_ms=1.0, max_lag_ms=100.0)
            if lags is not None:
                plt.figure(figsize=(7,4))
                plt.plot(lags*1000, ac, lw=1.2)
                plt.xlabel("Lag (ms)"); plt.ylabel("Auto-correlation (a.u.)")
                plt.title(f"Autocorrelogram — {base}")
                plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{base}_acg.png"), dpi=150); plt.close()

        # Waveform mean±SEM
        if W.shape[0] >= 1:
            t_wf_ms = (np.arange(W.shape[1]) - 0) / fs * 1000.0
            mu = W.mean(0); sem = W.std(0) / max(np.sqrt(W.shape[0]), 1)
            plt.figure(figsize=(6,4))
            plt.plot(t_wf_ms, mu, lw=2)
            plt.fill_between(t_wf_ms, mu - sem, mu + sem, alpha=0.3)
            plt.xlabel("Time (ms)"); plt.ylabel("µV")
            plt.title(f"Spike waveform mean±SEM — {base} (n={W.shape[0]})")
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{base}_waveform.png"), dpi=150); plt.close()

        # STA (per PDF’s event-triggered average idea)
        if sta is not None and n_used > 0:
            wlen = sta.shape[0]
            t_rel = np.arange(-half_win, -half_win + wlen, dtype=float) / fs
            plt.figure(figsize=(8,4))
            plt.plot(t_rel, sta, lw=1.5)
            plt.axvline(0, color='k', lw=1, alpha=0.5)
            plt.xlabel("Time rel. to spike (s)"); plt.ylabel("LFP (µV)")
            plt.title(f"Spike-Triggered Average LFP — {base} (n={n_used})")
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{base}_STA.png"), dpi=150); plt.close()

            pd.DataFrame({"t_rel_s": t_rel, "lfp_sta_uV": sta}).to_csv(
                os.path.join(out_dir, f"{base}_STA.csv"), index=False
            )

            # Spike-triggered spectrogram (optional)
            if DO_ST_SPECTRO:
                # collect spectrogram around each spike, then average
                pad = half_win
                specs = []
                for i in valid_idx:
                    a = i - pad; b = i + pad + 1
                    if a < 0 or b > len(x_lfp): continue
                    fS, tS, Sxx = spectrogram(x_lfp[a:b], fs=fs, nperseg=max(64, int(0.25*fs)),
                                              noverlap=int(0.5*max(64, int(0.25*fs))), scaling='density', mode='psd')
                    if SPECTRO_FMAX is not None:
                        keep = fS <= SPECTRO_FMAX
                        fS, Sxx = fS[keep], Sxx[keep, :]
                    specs.append(10*np.log10(Sxx + np.finfo(float).eps))
                if specs:
                    Smean = np.mean(np.stack(specs, axis=0), axis=0)
                    # center time axis at 0 for display
                    tS_rel = (tS - tS.mean()) * ( (b-a)/fs / (tS[-1]-tS[0]+1e-12) )
                    plt.figure(figsize=(7,4))
                    plt.pcolormesh(tS_rel, fS, Smean, shading='auto')
                    plt.xlabel("Time rel. to spike (s)"); plt.ylabel("Frequency (Hz)")
                    plt.title(f"Spike-triggered spectrogram (mean) — {base}")
                    plt.colorbar(label="dB(µV²/Hz)")
                    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{base}_STA_spectrogram.png"), dpi=150); plt.close()

        else:
            print(f"[STA] {base}: no usable spikes/windows (detected {len(spike_idx)}).")

        # For overlay
        overlay_freqs.append(f); overlay_psds_db.append(Pxx_db); overlay_labels.append(base)

    # ---------- Combined PSD overlay (all files) ----------
    if overlay_freqs:
        f_common = overlay_freqs[np.argmin([len(fi) for fi in overlay_freqs])]
        plt.figure(figsize=(10,6))
        for f_i, P_i, label in zip(overlay_freqs, overlay_psds_db, overlay_labels):
            y = P_i if np.array_equal(f_i, f_common) else np.interp(f_common, f_i, P_i)
            plt.plot(f_common, y, lw=1.5, label=label)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (dB(µV²/Hz))")
        plt.title("Welch PSD overlay (all files)")
        plt.grid(True, alpha=0.3)
        if FMAX is not None: plt.xlim(0, FMAX)
        if DB_RANGE is None and np.isfinite(global_db_min) and np.isfinite(global_db_max):
            pad = 3.0; plt.ylim(global_db_min - pad, global_db_max + pad)
        elif DB_RANGE is not None:
            plt.ylim(*DB_RANGE)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, "all_files_psd_overlay_dB.png"), dpi=150)
        plt.close()

    print(f"✅ Done. Plots saved under: {out_root}")

if __name__ == "__main__":
    main()
