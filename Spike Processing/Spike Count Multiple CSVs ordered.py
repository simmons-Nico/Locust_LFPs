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
import re
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spike_shape_qc as shape_qc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks


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

SPIKE_WAVEFORM_PRE_MS = 0.6
SPIKE_WAVEFORM_POST_MS = 1.0
SPIKE_WAVEFORM_MAX_TRACES = 200
SPIKE_CLASSIFICATION_WINDOW_MS = 0.5
SPIKE_RATE_BIN_S = 1.0
SPIKE_ACG_BIN_MS = 1.0
SPIKE_ACG_MAX_LAG_MS = 100.0
SPIKE_AMPLITUDE_BIN_S = 60.0
SPIKE_AMPLITUDE_BASELINE_MINUTES = 10
SPIKE_AMPLITUDE_BASELINE_MIN_VALID_BINS = 8
SPIKE_AMPLITUDE_BASELINE_MIN_SPIKES = 20
SPIKE_AMPLITUDE_BASELINE_MAX_ABS_SLOPE_PERCENT_PER_MIN = 2.0
SPIKE_AMPLITUDE_BASELINE_MAX_CV_PERCENT = 20.0

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

POLARITY_ALIASES = {
    "all": "both",
    "all spikes": "both",
    "both": "both",
    "both polarities": "both",
    "neg": "neg",
    "negative": "neg",
    "negative only": "neg",
    "negative spikes": "neg",
    "negative spikes only": "neg",
    "pos": "pos",
    "positive": "pos",
    "positive only": "pos",
    "positive spikes": "pos",
    "positive spikes only": "pos",
}

POLARITY_LABELS = {
    "both": "all spikes",
    "neg": "negative spikes only",
    "pos": "positive spikes only",
}

EVENT_POLARITY_LABELS = {
    "pos": "positive",
    "neg": "negative",
}

EVENT_POLARITY_AXIS_LABELS = {
    "pos": "Time from positive peak (ms)",
    "neg": "Time from negative trough (ms)",
}

EVENT_POLARITY_COLORS = {
    "pos": "tab:red",
    "neg": "tab:blue",
}

AMPLITUDE_PHASE_LABELS = {
    "baseline": "Baseline",
    "treatment": "Treatment",
    "post": "Post/Washout",
}

AMPLITUDE_PHASE_LINESTYLES = {
    "baseline": "-",
    "treatment": "--",
    "post": "-.",
}

PHASE_UNKNOWN = "unknown"
DEFAULT_UNRECORDED_TREATMENT_LABEL = "H2O2 treatment - not recorded"

SPIKE_AMPLITUDE_TIMECOURSE_COLUMNS = [
    "recording_order",
    "recording_name",
    "epoch_label",
    "phase",
    "phase_group",
    "normalized_phase",
    "channel",
    "polarity",
    "polarity_short",
    "bin_index",
    "recording_bin_index",
    "recording_bin_start_min",
    "recording_bin_mid_min",
    "recording_bin_end_min",
    "continuous_bin_start_min",
    "continuous_bin_mid_min",
    "continuous_bin_end_min",
    "treatment_relative_bin_start_min",
    "treatment_relative_bin_mid_min",
    "treatment_relative_bin_end_min",
    "plotted_relative_bin_start_min",
    "plotted_relative_bin_mid_min",
    "plotted_relative_bin_end_min",
    "bin_duration_s",
    "n_spikes",
    "mean_amplitude_uv",
    "sd_amplitude_uv",
    "sem_across_spikes_uv",
    "baseline_interval_start_min",
    "baseline_interval_end_min",
    "baseline_interval_start_treatment_relative_min",
    "baseline_interval_end_treatment_relative_min",
    "baseline_requested_bins",
    "baseline_available_bins",
    "baseline_valid_bins",
    "baseline_spike_count",
    "baseline_mean_amplitude_uv",
    "baseline_sd_amplitude_uv",
    "baseline_bin_mean_amplitude_uv",
    "baseline_bin_sd_amplitude_uv",
    "baseline_cv_percent",
    "baseline_slope_uv_per_min",
    "baseline_slope_percent_per_min",
    "baseline_qc_pass",
    "baseline_qc_reason",
    "treatment_recorded",
    "treatment_duration_known",
    "treatment_boundary_label",
    "phase_warning_reason",
    "percent_of_baseline",
    "percent_sem",
]


def normalize_polarity(value: str | None) -> str:
    raw = POLARITY if value is None else str(value)
    key = " ".join(raw.strip().lower().replace("_", " ").replace("-", " ").split())
    polarity = POLARITY_ALIASES.get(key)
    if polarity is None:
        allowed = ", ".join(sorted({"both", "neg", "pos", "all", "negative", "positive"}))
        raise ValueError(f"Spike polarity must be one of: {allowed}. Got {value!r}.")
    return polarity


def polarity_label(value: str | None) -> str:
    return POLARITY_LABELS[normalize_polarity(value)]


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
    polarity = normalize_polarity(polarity)
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
    polarity = normalize_polarity(polarity)
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


def requested_event_polarities(polarity: str | None) -> tuple[str, ...]:
    selected = normalize_polarity(polarity)
    if selected == "both":
        return ("pos", "neg")
    return (selected,)


def find_threshold_candidates(z: np.ndarray, polarity: str | None, thr: float) -> list[dict]:
    selected = normalize_polarity(polarity)
    candidates: list[dict] = []

    if selected in ("pos", "both"):
        p_pos, _ = find_peaks(z, height=thr)
        candidates.extend(
            {"candidate_sample_index": int(sample_index), "candidate_polarity": "pos"}
            for sample_index in p_pos
        )

    if selected in ("neg", "both"):
        p_neg, _ = find_peaks(-z, height=thr)
        candidates.extend(
            {"candidate_sample_index": int(sample_index), "candidate_polarity": "neg"}
            for sample_index in p_neg
        )

    candidates.sort(key=lambda row: (row["candidate_sample_index"], 0 if row["candidate_polarity"] == "neg" else 1))
    return candidates


def classification_radius_samples(fs: float, window_ms: float | None = None) -> int:
    selected_ms = SPIKE_CLASSIFICATION_WINDOW_MS if window_ms is None else float(window_ms)
    if not np.isfinite(selected_ms) or selected_ms < 0:
        raise ValueError("Spike classification/alignment window must be a non-negative finite value in ms.")
    if selected_ms == 0:
        return 0
    return max(1, int(math.ceil(selected_ms * 1e-3 * float(fs))))


def classify_and_align_candidates(
    x_hp: np.ndarray,
    z: np.ndarray,
    candidates: list[dict],
    fs: float,
    requested_polarity: str | None,
    classification_window_ms: float | None = None,
) -> list[dict]:
    selected = normalize_polarity(requested_polarity)
    radius = classification_radius_samples(fs, classification_window_ms)
    n_samples = len(x_hp)
    events: list[dict] = []

    for candidate in candidates:
        candidate_index = int(candidate["candidate_sample_index"])
        if candidate_index < 0 or candidate_index >= n_samples:
            continue

        start = max(0, candidate_index - radius)
        stop = min(n_samples, candidate_index + radius + 1)
        local = np.asarray(x_hp[start:stop], dtype=float)
        finite = np.isfinite(local)
        if local.size == 0 or not finite.any():
            continue

        local_max_uv = float(np.nanmax(local))
        local_min_uv = float(np.nanmin(local))
        local_max_index = int(start + np.nanargmax(local))
        local_min_index = int(start + np.nanargmin(local))

        final_polarity = "pos" if local_max_uv > abs(local_min_uv) else "neg"
        if selected != "both" and final_polarity != selected:
            continue

        canonical_index = local_max_index if final_polarity == "pos" else local_min_index
        signed_amplitude_uv = float(x_hp[canonical_index])
        events.append(
            {
                "sample_index": int(canonical_index),
                "candidate_sample_index": int(candidate_index),
                "alignment_shift_samples": int(canonical_index - candidate_index),
                "candidate_polarity": str(candidate["candidate_polarity"]),
                "polarity": final_polarity,
                "polarity_label": EVENT_POLARITY_LABELS[final_polarity],
                "signed_amplitude_uv": signed_amplitude_uv,
                "amplitude_uv": float(abs(signed_amplitude_uv)),
                "robust_z": float(z[canonical_index]),
                "candidate_robust_z": float(z[candidate_index]),
                "local_max_uv": local_max_uv,
                "local_max_sample_index": local_max_index,
                "local_min_uv": local_min_uv,
                "local_min_sample_index": local_min_index,
            }
        )

    return events


def deduplicate_canonical_events(events: list[dict]) -> list[dict]:
    best_by_sample: dict[int, dict] = {}

    def dedup_key(event: dict) -> tuple:
        return (
            abs(int(event["alignment_shift_samples"])),
            int(event["candidate_sample_index"]),
            0 if event["polarity"] == "neg" else 1,
            0 if event["candidate_polarity"] == "neg" else 1,
        )

    for event in events:
        sample_index = int(event["sample_index"])
        existing = best_by_sample.get(sample_index)
        if existing is None or dedup_key(event) < dedup_key(existing):
            best_by_sample[sample_index] = event

    return sorted(best_by_sample.values(), key=lambda event: int(event["sample_index"]))


def suppress_refractory_events_by_amplitude(events: list[dict], refractory_samp: int) -> list[dict]:
    if not events:
        return []
    refractory_samp = max(1, int(refractory_samp))

    def priority_key(event: dict) -> tuple:
        # Exact amplitude ties are deterministic: negative extrema win, then earlier samples.
        return (
            -float(event["amplitude_uv"]),
            0 if event["polarity"] == "neg" else 1,
            int(event["sample_index"]),
        )

    kept: list[dict] = []
    # A refractory-sized bucket holds at most one accepted event. Only its
    # immediate neighbours can conflict, avoiding an all-pairs scan while
    # preserving the amplitude priority and exact boundary/tie rules.
    occupied: dict[int, int] = {}
    for event in sorted(events, key=priority_key):
        sample_index = int(event["sample_index"])
        bucket = sample_index // refractory_samp
        if all(
            neighbour not in occupied
            or abs(sample_index - occupied[neighbour]) >= refractory_samp
            for neighbour in (bucket - 1, bucket, bucket + 1)
        ):
            kept.append(event)
            occupied[bucket] = sample_index

    return sorted(kept, key=lambda event: int(event["sample_index"]))


def apply_width_gate_to_events(
    x_hp: np.ndarray,
    events: list[dict],
    fs: float,
    wmin_ms=None,
    wmax_ms=None,
) -> list[dict]:
    if not events or (wmin_ms is None and wmax_ms is None):
        return events
    kept: list[dict] = []
    for event in events:
        peak = np.array([int(event["sample_index"])], dtype=int)
        if width_gate_indices(x_hp, peak, fs, wmin_ms, wmax_ms, polarity=event["polarity"]).size:
            kept.append(event)
    return kept


def apply_amplitude_limits_to_events(events: list[dict]) -> list[dict]:
    if not events or (AMP_MIN_UV is None and AMP_MAX_UV is None):
        return events
    kept: list[dict] = []
    for event in events:
        amplitude = float(event["amplitude_uv"])
        if AMP_MIN_UV is not None and amplitude < float(AMP_MIN_UV):
            continue
        if AMP_MAX_UV is not None and amplitude > float(AMP_MAX_UV):
            continue
        kept.append(event)
    return kept


def detect_spikes_with_details(
    x_uv: np.ndarray,
    t_s: np.ndarray,
    polarity: str | None = None,
    return_filtered_trace: bool = False,
    classification_window_ms: float | None = None,
    shape_settings=None,
):
    selected_polarity = normalize_polarity(polarity)
    fs = estimate_fs_from_time(t_s)
    x_hp = bandpass_filt(x_uv.astype(float), fs, HP_SPIKE_BAND, order=3)
    z, _, _ = robust_z(x_hp)
    thr = float(SPIKE_Z_THR)

    refractory = int(round((REFRACTORY_MS / 1000.0) * fs))
    refractory = max(refractory, 1)

    candidates = find_threshold_candidates(z, selected_polarity, thr)
    events = classify_and_align_candidates(
        x_hp,
        z,
        candidates,
        fs,
        selected_polarity,
        classification_window_ms=classification_window_ms,
    )
    events = deduplicate_canonical_events(events)
    cfg = shape_qc.settings(shape_settings)
    # Evaluate all unique, amplitude/width-eligible events before suppression.
    eligible = apply_amplitude_limits_to_events(apply_width_gate_to_events(x_hp, events, fs, W_MIN_MS, W_MAX_MS)) if cfg["mode"] != "off" else []
    shape_reports = shape_qc.assess_events(x_hp, eligible, fs, cfg)
    baseline_events = apply_amplitude_limits_to_events(apply_width_gate_to_events(
        x_hp, suppress_refractory_events_by_amplitude(events, refractory), fs, W_MIN_MS, W_MAX_MS)) if cfg["mode"] != "off" else []
    if cfg["mode"] == "reject":
        rejected = {row["sample_index"] for row in shape_reports if row["shape_flagged"]}
        events = [event for event in events if event["sample_index"] not in rejected]
    events = suppress_refractory_events_by_amplitude(events, refractory)
    events = apply_width_gate_to_events(x_hp, events, fs, W_MIN_MS, W_MAX_MS)
    events = apply_amplitude_limits_to_events(events)
    if cfg["mode"] == "off":
        baseline_events = events

    peaks = np.asarray([event["sample_index"] for event in events], dtype=int)
    spike_times = t_s[peaks] if peaks.size else np.array([], dtype=float)
    signed_amp = np.asarray([event["signed_amplitude_uv"] for event in events], dtype=float)
    event_polarity = np.asarray([event["polarity"] for event in events], dtype=str)
    n_positive = int(np.sum(event_polarity == "pos")) if event_polarity.size else 0
    n_negative = int(np.sum(event_polarity == "neg")) if event_polarity.size else 0
    details = {
        "shape_settings": cfg,
        "shape_reports": shape_reports,
        "shape_before_peaks": np.asarray([e["sample_index"] for e in baseline_events], dtype=int),
        "peaks": peaks,
        "spike_times": spike_times,
        "fs": fs,
        "amplitude_uv": np.abs(signed_amp),
        "signed_amplitude_uv": signed_amp,
        "robust_z": np.asarray([event["robust_z"] for event in events], dtype=float),
        "event_polarity": event_polarity,
        "event_polarity_label": np.asarray(
            [event["polarity_label"] for event in events],
            dtype=str,
        ),
        "candidate_sample_index": np.asarray(
            [event["candidate_sample_index"] for event in events],
            dtype=int,
        ),
        "alignment_shift_samples": np.asarray(
            [event["alignment_shift_samples"] for event in events],
            dtype=int,
        ),
        "candidate_polarity": np.asarray(
            [event["candidate_polarity"] for event in events],
            dtype=str,
        ),
        "candidate_robust_z": np.asarray(
            [event["candidate_robust_z"] for event in events],
            dtype=float,
        ),
        "local_max_uv": np.asarray(
            [event["local_max_uv"] for event in events],
            dtype=float,
        ),
        "local_max_sample_index": np.asarray(
            [event["local_max_sample_index"] for event in events],
            dtype=int,
        ),
        "local_min_uv": np.asarray(
            [event["local_min_uv"] for event in events],
            dtype=float,
        ),
        "local_min_sample_index": np.asarray(
            [event["local_min_sample_index"] for event in events],
            dtype=int,
        ),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "requested_polarity": selected_polarity,
        "classification_window_ms": float(
            SPIKE_CLASSIFICATION_WINDOW_MS if classification_window_ms is None else classification_window_ms
        ),
    }
    if return_filtered_trace:
        details["filtered_trace_uv"] = x_hp
    return details


def detect_spikes(
    x_uv: np.ndarray,
    t_s: np.ndarray,
    polarity: str | None = None,
    classification_window_ms: float | None = None,
):
    details = detect_spikes_with_details(
        x_uv,
        t_s,
        polarity=polarity,
        classification_window_ms=classification_window_ms,
    )
    peaks = details["peaks"]
    spike_times = details["spike_times"]
    fs = details["fs"]
    return peaks, spike_times, fs


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_") or "value"


def extract_spike_waveforms(
    x_hp: np.ndarray,
    peaks: np.ndarray,
    fs: float,
    pre_ms: float = SPIKE_WAVEFORM_PRE_MS,
    post_ms: float = SPIKE_WAVEFORM_POST_MS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract spike-band snippets centered on detected threshold crossings."""
    if not np.isfinite(fs) or fs <= 0:
        return np.array([], dtype=int), np.empty((0, 0), dtype=float), np.array([], dtype=float)
    pre = int(round(float(pre_ms) * 1e-3 * fs))
    post = int(round(float(post_ms) * 1e-3 * fs))
    if pre < 1 or post < 1:
        return np.array([], dtype=int), np.empty((0, 0), dtype=float), np.array([], dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    valid = peaks[(peaks - pre >= 0) & (peaks + post + 1 <= len(x_hp))]
    if valid.size == 0:
        rel_ms = np.arange(-pre, post + 1, dtype=float) / fs * 1000.0
        return valid, np.empty((0, rel_ms.size), dtype=float), rel_ms
    waveforms = np.vstack([x_hp[idx - pre : idx + post + 1] for idx in valid])
    rel_ms = np.arange(-pre, post + 1, dtype=float) / fs * 1000.0
    return valid, waveforms, rel_ms


def spike_autocorrelogram(times_s: np.ndarray, bin_ms: float = SPIKE_ACG_BIN_MS, max_lag_ms: float = SPIKE_ACG_MAX_LAG_MS) -> tuple[np.ndarray, np.ndarray]:
    """Return a symmetric spike-time autocorrelogram with the zero-lag bin removed."""
    times = np.sort(np.asarray(times_s, dtype=float))
    times = times[np.isfinite(times)]
    if times.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    max_lag_s = float(max_lag_ms) / 1000.0
    lags = []
    for i, time in enumerate(times[:-1]):
        stop = int(np.searchsorted(times, time + max_lag_s, side="right"))
        diffs = times[i + 1 : stop] - time
        if diffs.size:
            lags.extend(diffs.tolist())
    if not lags:
        return np.array([], dtype=float), np.array([], dtype=float)
    lag_arr_ms = np.asarray(lags, dtype=float) * 1000.0
    both = np.concatenate([-lag_arr_ms[::-1], lag_arr_ms])
    edges = np.arange(-float(max_lag_ms), float(max_lag_ms) + float(bin_ms), float(bin_ms))
    if edges.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    counts, edges = np.histogram(both, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    zero_bin = np.abs(centers) < (float(bin_ms) * 0.5)
    counts[zero_bin] = 0
    return centers, counts.astype(float)


def write_spike_diagnostic_outputs(
    out_dir: str,
    rec_name: str,
    channel: str,
    details: dict,
) -> tuple[list[dict], list[dict]]:
    """Write per-recording average waveform CSVs and waveform summaries (figures are drawn on the 4th tab)."""
    diagnostic_dir = os.path.join(out_dir, "spike_diagnostics")
    dirs = {
        "waveform": os.path.join(diagnostic_dir, "average_waveforms"),
    }
    for folder in dirs.values():
        os.makedirs(folder, exist_ok=True)

    safe_base = _safe_name(rec_name)
    safe_ch = _safe_name(channel)
    prefix = f"{safe_base}__{safe_ch}"
    spike_times = np.asarray(details.get("spike_times", []), dtype=float)
    fs = float(details.get("fs", np.nan))
    peaks = np.asarray(details.get("peaks", []), dtype=int)
    outputs: list[dict] = []

    def add(kind: str, path: str) -> None:
        outputs.append(
            {
                "recording_name": rec_name,
                "channel": channel,
                "output_type": kind,
                "path": path,
            }
        )

    waveform_summary_rows: list[dict] = []
    x_hp = details.get("filtered_trace_uv")
    if x_hp is not None:
        x_hp = np.asarray(x_hp, dtype=float)
        event_polarities = np.asarray(details.get("event_polarity", []), dtype=str)
        n_positive = int(details.get("n_positive", int(np.sum(event_polarities == "pos"))))
        n_negative = int(details.get("n_negative", int(np.sum(event_polarities == "neg"))))
        output_polarities = requested_event_polarities(details.get("requested_polarity", "both"))

        for polarity_key in output_polarities:
            polarity_name = EVENT_POLARITY_LABELS[polarity_key]
            class_mask = event_polarities == polarity_key
            class_peaks = peaks[class_mask] if peaks.size else np.array([], dtype=int)
            valid_peaks, waveforms, rel_ms = extract_spike_waveforms(x_hp, class_peaks, fs)

            if waveforms.shape[0] >= 1:
                mean = np.nanmean(waveforms, axis=0)
                sem = (
                    np.nanstd(waveforms, axis=0, ddof=1) / math.sqrt(waveforms.shape[0])
                    if waveforms.shape[0] > 1
                    else np.zeros_like(mean)
                )
            else:
                mean = np.full(rel_ms.shape, np.nan, dtype=float)
                sem = np.full(rel_ms.shape, np.nan, dtype=float)

            csv_path = os.path.join(dirs["waveform"], f"{prefix}__{polarity_name}_average_waveform.csv")
            pd.DataFrame(
                {
                    "time_ms": rel_ms,
                    "mean_waveform_uv": mean,
                    "sem_waveform_uv": sem,
                    "n_spikes_used": int(waveforms.shape[0]),
                }
            ).to_csv(csv_path, index=False)
            add(f"{polarity_name}_average_waveform_csv", csv_path)

            zero_idx = int(np.argmin(np.abs(rel_ms))) if rel_ms.size else 0
            zero_mean = float(mean[zero_idx]) if rel_ms.size and np.isfinite(mean[zero_idx]) else np.nan
            waveform_summary_rows.append(
                {
                    "recording_name": rec_name,
                    "channel": channel,
                    "polarity": polarity_name,
                    "polarity_short": polarity_key,
                    "detected_spikes": int(spike_times.size),
                    "total_spikes": int(peaks.size),
                    "n_positive": n_positive,
                    "n_negative": n_negative,
                    "polarity_spike_count": int(class_peaks.size),
                    "waveform_spikes_used": int(waveforms.shape[0]),
                    "edge_excluded_spikes": int(max(0, class_peaks.size - valid_peaks.size)),
                    "sampling_rate_hz": fs,
                    "pre_ms": float(SPIKE_WAVEFORM_PRE_MS),
                    "post_ms": float(SPIKE_WAVEFORM_POST_MS),
                    "classification_window_ms": float(details.get("classification_window_ms", SPIKE_CLASSIFICATION_WINDOW_MS)),
                    "mean_peak_uv": zero_mean,
                    "mean_peak_abs_uv": float(abs(zero_mean)) if np.isfinite(zero_mean) else np.nan,
                    "mean_peak_to_peak_uv": float(np.nanmax(mean) - np.nanmin(mean)) if np.isfinite(mean).any() else np.nan,
                    "waveform_csv": csv_path,
                    "waveform_plot": "",
                }
            )

    return outputs, waveform_summary_rows


def _normalise_phase_text(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())


def normalize_amplitude_phase(phase: str | None = None, epoch_label: str | None = None) -> tuple[str, str]:
    label_text = _normalise_phase_text(epoch_label)
    metadata_text = _normalise_phase_text(phase)
    text = label_text or metadata_text

    if any(token in text for token in ("baseline", "basal", "before")) or re.search(r"(?<![a-z0-9])pre(?![a-z0-9])", text):
        return "baseline", ""
    if any(token in text for token in ("post", "after", "recovery", "washout")):
        return "post", ""
    if "during" in text:
        return "treatment", ""
    if metadata_text == "treatment":
        return "treatment", ""

    if text:
        return PHASE_UNKNOWN, f"unknown epoch label '{epoch_label or phase}'"
    return PHASE_UNKNOWN, "missing epoch label"


def amplitude_phase_group(phase: str | None, epoch_label: str | None = None) -> str:
    return normalize_amplitude_phase(phase=phase, epoch_label=epoch_label)[0]


def spike_amplitude_value_uv(polarity_short: str, signed_amplitude_uv: float) -> float:
    signed = float(signed_amplitude_uv)
    return signed if polarity_short == "pos" else abs(signed)


def _recording_duration_s(t_s: np.ndarray, fs: float) -> float:
    finite_t = np.asarray(t_s, dtype=float)
    finite_t = finite_t[np.isfinite(finite_t)]
    if finite_t.size == 0:
        return 0.0
    sample_interval_s = 1.0 / float(fs) if np.isfinite(fs) and fs > 0 else 0.0
    return max(0.0, float(finite_t[-1] - finite_t[0] + sample_interval_s))


def spike_amplitude_recording_row(
    recording_key: str,
    rec_idx: int,
    rec_name: str,
    channel: str,
    epoch_label: str,
    phase: str,
    phase_group: str,
    phase_warning_reason: str,
    t_s: np.ndarray,
    fs: float,
) -> dict:
    return {
        "recording_key": recording_key,
        "recording_order": int(rec_idx),
        "recording_name": rec_name,
        "channel": channel,
        "epoch_label": epoch_label,
        "phase": phase,
        "phase_group": phase_group,
        "normalized_phase": phase_group,
        "phase_warning_reason": phase_warning_reason,
        "recording_duration_s": _recording_duration_s(t_s, fs),
    }


def spike_amplitude_event_rows(
    recording_key: str,
    rec_idx: int,
    rec_name: str,
    channel: str,
    epoch_label: str,
    phase: str,
    phase_group: str,
    phase_warning_reason: str,
    t0: float,
    details: dict,
) -> list[dict]:
    peaks = np.asarray(details.get("peaks", []), dtype=int)
    if peaks.size == 0:
        return []

    rows: list[dict] = []
    spike_times = np.asarray(details.get("spike_times", []), dtype=float)
    event_polarities = np.asarray(details.get("event_polarity", []), dtype=str)
    signed_amplitudes = np.asarray(details.get("signed_amplitude_uv", []), dtype=float)
    robust_z_values = np.asarray(details.get("robust_z", []), dtype=float)
    candidate_indices = np.asarray(details.get("candidate_sample_index", []), dtype=int)
    alignment_shifts = np.asarray(details.get("alignment_shift_samples", []), dtype=int)

    for event_index, sample_index in enumerate(peaks, start=1):
        polarity_short = str(event_polarities[event_index - 1])
        signed_amplitude = float(signed_amplitudes[event_index - 1])
        rows.append(
            {
                "recording_key": recording_key,
                "recording_order": int(rec_idx),
                "recording_name": rec_name,
                "epoch_label": epoch_label,
                "phase": phase,
                "phase_group": phase_group,
                "normalized_phase": phase_group,
                "phase_warning_reason": phase_warning_reason,
                "channel": channel,
                "polarity": EVENT_POLARITY_LABELS.get(polarity_short, polarity_short),
                "polarity_short": polarity_short,
                "event_index": int(event_index),
                "sample_index": int(sample_index),
                "canonical_sample_index": int(sample_index),
                "candidate_sample_index": int(candidate_indices[event_index - 1]) if candidate_indices.size else int(sample_index),
                "alignment_shift_samples": int(alignment_shifts[event_index - 1]) if alignment_shifts.size else 0,
                "spike_time_s": float(spike_times[event_index - 1]),
                "recording_relative_spike_time_s": float(spike_times[event_index - 1] - t0),
                "signed_amplitude_uv": signed_amplitude,
                "amplitude_uv": spike_amplitude_value_uv(polarity_short, signed_amplitude),
                "robust_z": float(robust_z_values[event_index - 1]) if robust_z_values.size else np.nan,
            }
        )
    return rows


def spike_amplitude_waveform_rows(
    channel: str,
    phase_group: str,
    details: dict,
) -> list[dict]:
    x_hp = details.get("filtered_trace_uv")
    if x_hp is None:
        return []
    fs = float(details.get("fs", np.nan))
    peaks = np.asarray(details.get("peaks", []), dtype=int)
    event_polarities = np.asarray(details.get("event_polarity", []), dtype=str)
    if not np.isfinite(fs) or fs <= 0 or peaks.size == 0:
        return []

    rows: list[dict] = []
    x_hp = np.asarray(x_hp, dtype=float)
    for polarity_short in requested_event_polarities(details.get("requested_polarity", "both")):
        class_peaks = peaks[event_polarities == polarity_short]
        _valid_peaks, waveforms, rel_ms = extract_spike_waveforms(x_hp, class_peaks, fs)
        for waveform in waveforms:
            rows.append(
                {
                    "channel": channel,
                    "phase_group": phase_group,
                    "polarity_short": polarity_short,
                    "waveform": np.asarray(waveform, dtype=float),
                    "rel_ms": np.asarray(rel_ms, dtype=float),
                }
            )
    return rows


def _spike_amplitude_stats(values: np.ndarray) -> tuple[int, float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return 0, np.nan, np.nan, np.nan
    mean = float(np.nanmean(values))
    if n == 1:
        return n, mean, np.nan, np.nan
    sd = float(np.nanstd(values, ddof=1))
    return n, mean, sd, float(sd / math.sqrt(n))


def _timeline_recordings(recording_rows: list[dict]) -> pd.DataFrame:
    if not recording_rows:
        return pd.DataFrame()
    rec_df = pd.DataFrame(recording_rows).copy()
    for column in ("epoch_label", "phase", "phase_group", "normalized_phase", "phase_warning_reason"):
        if column not in rec_df.columns:
            rec_df[column] = ""
    for idx in rec_df.index:
        normalized_phase, warning_reason = normalize_amplitude_phase(
            phase=rec_df.at[idx, "phase"],
            epoch_label=rec_df.at[idx, "epoch_label"],
        )
        rec_df.at[idx, "phase_group"] = normalized_phase
        rec_df.at[idx, "normalized_phase"] = normalized_phase
        rec_df.at[idx, "phase_warning_reason"] = warning_reason
    rec_df["recording_start_continuous_s"] = np.nan
    rec_df["recording_end_continuous_s"] = np.nan
    for channel, sub in rec_df.groupby("channel", sort=True):
        cursor = 0.0
        for idx in sub.sort_values(["recording_order", "recording_name"]).index:
            duration_s = max(0.0, float(rec_df.at[idx, "recording_duration_s"]))
            rec_df.at[idx, "recording_start_continuous_s"] = cursor
            rec_df.at[idx, "recording_end_continuous_s"] = cursor + duration_s
            cursor += duration_s
    return rec_df


def _event_frame_with_continuous_time(event_rows: list[dict], rec_df: pd.DataFrame) -> pd.DataFrame:
    if event_rows:
        events = pd.DataFrame(event_rows).copy()
    else:
        events = pd.DataFrame(
            columns=[
                "recording_key",
                "recording_order",
                "recording_name",
                "phase",
                "phase_group",
                "channel",
                "polarity",
                "polarity_short",
                "recording_relative_spike_time_s",
                "amplitude_uv",
            ]
        )
    if rec_df.empty:
        events["continuous_spike_time_s"] = np.nan
        return events
    starts = rec_df.set_index("recording_key")["recording_start_continuous_s"].to_dict()
    events["continuous_spike_time_s"] = events["recording_key"].map(starts) + pd.to_numeric(
        events["recording_relative_spike_time_s"],
        errors="coerce",
    )
    return events


def _treatment_start_by_channel(rec_df: pd.DataFrame) -> dict[str, float]:
    starts: dict[str, float] = {}
    if rec_df.empty:
        return starts
    for channel, sub in rec_df.groupby("channel", sort=True):
        treatment = sub[sub["phase_group"] == "treatment"].sort_values("recording_start_continuous_s")
        starts[channel] = (
            float(treatment.iloc[0]["recording_start_continuous_s"])
            if not treatment.empty
            else np.nan
        )
    return starts


def _amplitude_plot_reference_by_channel(rec_df: pd.DataFrame) -> dict[str, dict[str, float | bool]]:
    refs: dict[str, dict[str, float | bool]] = {}
    if rec_df.empty:
        return refs
    for channel, sub in rec_df.groupby("channel", sort=True):
        ordered = sub.sort_values("recording_start_continuous_s")
        treatment = ordered[ordered["phase_group"] == "treatment"]
        if not treatment.empty:
            refs[channel] = {
                "reference_start_s": float(treatment.iloc[0]["recording_start_continuous_s"]),
                "treatment_recorded": True,
                "treatment_duration_known": True,
            }
            continue
        post = ordered[ordered["phase_group"] == "post"]
        if not post.empty:
            refs[channel] = {
                "reference_start_s": float(post.iloc[0]["recording_start_continuous_s"]),
                "treatment_recorded": False,
                "treatment_duration_known": False,
            }
            continue
        refs[channel] = {
            "reference_start_s": np.nan,
            "treatment_recorded": False,
            "treatment_duration_known": False,
        }
    return refs


def _complete_amplitude_bins(
    rec_df: pd.DataFrame,
    events: pd.DataFrame,
    output_polarities: tuple[str, ...],
    boundary_label: str = DEFAULT_UNRECORDED_TREATMENT_LABEL,
    bin_s: float = SPIKE_AMPLITUDE_BIN_S,
) -> pd.DataFrame:
    rows: list[dict] = []
    if rec_df.empty:
        return pd.DataFrame(rows, columns=SPIKE_AMPLITUDE_TIMECOURSE_COLUMNS)

    treatment_starts = _treatment_start_by_channel(rec_df)
    plot_refs = _amplitude_plot_reference_by_channel(rec_df)
    channel_bin_index: dict[str, int] = {}
    events = events.copy()
    if not events.empty:
        events["amplitude_uv"] = pd.to_numeric(events["amplitude_uv"], errors="coerce")
        events["continuous_spike_time_s"] = pd.to_numeric(events["continuous_spike_time_s"], errors="coerce")

    for rec in rec_df.sort_values(["channel", "recording_order", "recording_name"]).itertuples(index=False):
        channel = str(rec.channel)
        phase = str(rec.phase)
        phase_group = str(rec.phase_group)
        epoch_label = str(getattr(rec, "epoch_label", "") or "")
        phase_warning_reason = str(getattr(rec, "phase_warning_reason", "") or "")
        treatment_start = treatment_starts.get(channel, np.nan)
        plot_ref = plot_refs.get(channel, {})
        plot_reference_start_s = float(plot_ref.get("reference_start_s", np.nan))
        treatment_recorded = bool(plot_ref.get("treatment_recorded", False))
        treatment_duration_known = bool(plot_ref.get("treatment_duration_known", False))
        start_s = float(rec.recording_start_continuous_s)
        n_bins = int(math.floor(max(0.0, float(rec.recording_duration_s)) / float(bin_s)))
        for recording_bin_index in range(n_bins):
            bin_start_s = start_s + recording_bin_index * float(bin_s)
            bin_end_s = bin_start_s + float(bin_s)
            bin_mid_s = (bin_start_s + bin_end_s) / 2.0
            bin_index = channel_bin_index.get(channel, 0)
            channel_bin_index[channel] = bin_index + 1
            for polarity_short in output_polarities:
                polarity_name = EVENT_POLARITY_LABELS[polarity_short]
                if events.empty:
                    amps = np.array([], dtype=float)
                else:
                    mask = (
                        (events["channel"].astype(str) == channel)
                        & (events["polarity_short"].astype(str) == polarity_short)
                        & (events["continuous_spike_time_s"] >= bin_start_s)
                        & (events["continuous_spike_time_s"] < bin_end_s)
                    )
                    amps = events.loc[mask, "amplitude_uv"].to_numpy(dtype=float)
                n_spikes, mean_uv, sd_uv, sem_uv = _spike_amplitude_stats(amps)
                if np.isfinite(treatment_start):
                    rel_start_min = (bin_start_s - treatment_start) / 60.0
                    rel_mid_min = (bin_mid_s - treatment_start) / 60.0
                    rel_end_min = (bin_end_s - treatment_start) / 60.0
                else:
                    rel_start_min = rel_mid_min = rel_end_min = np.nan
                if np.isfinite(plot_reference_start_s):
                    plotted_start_min = (bin_start_s - plot_reference_start_s) / 60.0
                    plotted_mid_min = (bin_mid_s - plot_reference_start_s) / 60.0
                    plotted_end_min = (bin_end_s - plot_reference_start_s) / 60.0
                else:
                    plotted_start_min = bin_start_s / 60.0
                    plotted_mid_min = bin_mid_s / 60.0
                    plotted_end_min = bin_end_s / 60.0
                rows.append(
                    {
                        "recording_order": int(rec.recording_order),
                        "recording_name": str(rec.recording_name),
                        "epoch_label": epoch_label,
                        "phase": phase,
                        "phase_group": phase_group,
                        "normalized_phase": phase_group,
                        "channel": channel,
                        "polarity": polarity_name,
                        "polarity_short": polarity_short,
                        "bin_index": int(bin_index),
                        "recording_bin_index": int(recording_bin_index),
                        "recording_bin_start_min": recording_bin_index * float(bin_s) / 60.0,
                        "recording_bin_mid_min": (recording_bin_index + 0.5) * float(bin_s) / 60.0,
                        "recording_bin_end_min": (recording_bin_index + 1.0) * float(bin_s) / 60.0,
                        "continuous_bin_start_min": bin_start_s / 60.0,
                        "continuous_bin_mid_min": bin_mid_s / 60.0,
                        "continuous_bin_end_min": bin_end_s / 60.0,
                        "treatment_relative_bin_start_min": rel_start_min,
                        "treatment_relative_bin_mid_min": rel_mid_min,
                        "treatment_relative_bin_end_min": rel_end_min,
                        "plotted_relative_bin_start_min": plotted_start_min,
                        "plotted_relative_bin_mid_min": plotted_mid_min,
                        "plotted_relative_bin_end_min": plotted_end_min,
                        "bin_duration_s": float(bin_s),
                        "n_spikes": n_spikes,
                        "mean_amplitude_uv": mean_uv,
                        "sd_amplitude_uv": sd_uv,
                        "sem_across_spikes_uv": sem_uv,
                        "treatment_recorded": treatment_recorded,
                        "treatment_duration_known": treatment_duration_known,
                        "treatment_boundary_label": boundary_label if not treatment_recorded else "",
                        "phase_warning_reason": phase_warning_reason,
                        "percent_of_baseline": np.nan,
                        "percent_sem": np.nan,
                    }
                )

    df = pd.DataFrame(rows, columns=SPIKE_AMPLITUDE_TIMECOURSE_COLUMNS)
    df["baseline_qc_pass"] = pd.Series([pd.NA] * len(df), dtype="object")
    df["baseline_qc_reason"] = pd.Series([pd.NA] * len(df), dtype="object")
    return df


def _events_in_bin_interval(events: pd.DataFrame, selected_bins: pd.DataFrame, channel: str, polarity_short: str) -> pd.DataFrame:
    if events.empty or selected_bins.empty:
        return events.iloc[0:0].copy()
    event_mask = (
        (events["channel"].astype(str) == channel)
        & (events["polarity_short"].astype(str) == polarity_short)
    )
    interval_mask = pd.Series(False, index=events.index)
    for row in selected_bins.itertuples(index=False):
        start_s = float(row.continuous_bin_start_min) * 60.0
        end_s = float(row.continuous_bin_end_min) * 60.0
        interval_mask |= (
            (events["continuous_spike_time_s"] >= start_s)
            & (events["continuous_spike_time_s"] < end_s)
        )
    return events[event_mask & interval_mask].copy()


def _apply_spike_amplitude_baselines(
    bin_df: pd.DataFrame,
    events: pd.DataFrame,
    output_polarities: tuple[str, ...],
) -> tuple[pd.DataFrame, list[str]]:
    if bin_df.empty:
        return bin_df, ["No complete 60-second bins were available for spike-amplitude time-course plotting."]

    out = bin_df.copy()
    warnings: list[str] = []
    if "phase_warning_reason" in out.columns:
        phase_warnings = (
            out["phase_warning_reason"]
            .dropna()
            .astype(str)
            .str.strip()
        )
        for reason in sorted({value for value in phase_warnings if value}):
            warnings.append(f"Phase warning - {reason}.")
    requested_bins = int(SPIKE_AMPLITUDE_BASELINE_MINUTES)

    for channel in sorted(out["channel"].dropna().astype(str).unique()):
        channel_rows = out[out["channel"].astype(str) == channel]
        treatment_rel = pd.to_numeric(channel_rows["treatment_relative_bin_start_min"], errors="coerce")
        treatment_available = treatment_rel.notna().any()
        for polarity_short in output_polarities:
            polarity_name = EVENT_POLARITY_LABELS[polarity_short]
            mask = (
                (out["channel"].astype(str) == channel)
                & (out["polarity_short"].astype(str) == polarity_short)
            )
            sub = out[mask].sort_values("continuous_bin_start_min")
            before_treatment = (
                pd.to_numeric(sub["treatment_relative_bin_end_min"], errors="coerce") <= 0.0
                if treatment_available
                else pd.Series(True, index=sub.index)
            )
            baseline_bins = sub[(sub["phase_group"] == "baseline") & before_treatment].tail(requested_bins)
            baseline_events = _events_in_bin_interval(events, baseline_bins, channel, polarity_short)
            baseline_amps = pd.to_numeric(baseline_events.get("amplitude_uv", pd.Series(dtype=float)), errors="coerce")
            baseline_amps = baseline_amps[np.isfinite(baseline_amps)]
            baseline_spike_count = int(baseline_amps.size)

            baseline_mean = float(baseline_amps.mean()) if baseline_spike_count else np.nan
            baseline_sd = float(baseline_amps.std(ddof=1)) if baseline_spike_count > 1 else np.nan
            bin_means = pd.to_numeric(baseline_bins["mean_amplitude_uv"], errors="coerce").dropna()
            baseline_valid_bins = int(bin_means.size)
            baseline_available_bins = int(len(baseline_bins))
            bin_mean = float(bin_means.mean()) if baseline_valid_bins else np.nan
            bin_sd = float(bin_means.std(ddof=1)) if baseline_valid_bins > 1 else np.nan
            baseline_cv = float(100.0 * bin_sd / bin_mean) if np.isfinite(bin_mean) and bin_mean > 0 and np.isfinite(bin_sd) else np.nan

            if baseline_valid_bins >= 2:
                x = pd.to_numeric(
                    baseline_bins.loc[bin_means.index, "treatment_relative_bin_mid_min"],
                    errors="coerce",
                )
                if not x.notna().all():
                    x = pd.to_numeric(
                        baseline_bins.loc[bin_means.index, "continuous_bin_mid_min"],
                        errors="coerce",
                    )
                slope = float(np.polyfit(x.to_numpy(dtype=float), bin_means.to_numpy(dtype=float), 1)[0])
            else:
                slope = np.nan
            slope_percent = float(100.0 * slope / baseline_mean) if np.isfinite(slope) and np.isfinite(baseline_mean) and baseline_mean > 0 else np.nan

            reasons: list[str] = []
            if baseline_available_bins == 0:
                reasons.append("no complete baseline bins before treatment")
            if baseline_valid_bins < SPIKE_AMPLITUDE_BASELINE_MIN_VALID_BINS:
                reasons.append(
                    f"valid baseline bins {baseline_valid_bins}<"
                    f"{SPIKE_AMPLITUDE_BASELINE_MIN_VALID_BINS}"
                )
            if baseline_spike_count < SPIKE_AMPLITUDE_BASELINE_MIN_SPIKES:
                reasons.append(
                    f"baseline spikes {baseline_spike_count}<"
                    f"{SPIKE_AMPLITUDE_BASELINE_MIN_SPIKES}"
                )
            if not np.isfinite(baseline_mean) or baseline_mean <= 0:
                reasons.append("baseline mean is missing or nonpositive")
            if np.isfinite(slope_percent) and abs(slope_percent) > SPIKE_AMPLITUDE_BASELINE_MAX_ABS_SLOPE_PERCENT_PER_MIN:
                reasons.append(
                    f"baseline slope {slope_percent:.3g}%/min exceeds "
                    f"{SPIKE_AMPLITUDE_BASELINE_MAX_ABS_SLOPE_PERCENT_PER_MIN:g}%/min"
                )
            if np.isfinite(baseline_cv) and baseline_cv > SPIKE_AMPLITUDE_BASELINE_MAX_CV_PERCENT:
                reasons.append(
                    f"baseline-bin CV {baseline_cv:.3g}% exceeds "
                    f"{SPIKE_AMPLITUDE_BASELINE_MAX_CV_PERCENT:g}%"
                )

            qc_pass = len(reasons) == 0
            qc_reason = "pass" if qc_pass else "; ".join(reasons)
            normalization_available = (
                baseline_valid_bins >= SPIKE_AMPLITUDE_BASELINE_MIN_VALID_BINS
                and baseline_spike_count >= SPIKE_AMPLITUDE_BASELINE_MIN_SPIKES
                and np.isfinite(baseline_mean)
                and baseline_mean > 0
            )

            if not qc_pass:
                warnings.append(f"{channel} {polarity_name}: baseline QC warning - {qc_reason}.")

            if baseline_available_bins:
                baseline_start_min = float(baseline_bins["continuous_bin_start_min"].iloc[0])
                baseline_end_min = float(baseline_bins["continuous_bin_end_min"].iloc[-1])
                baseline_rel_start_min = float(baseline_bins["treatment_relative_bin_start_min"].iloc[0])
                baseline_rel_end_min = float(baseline_bins["treatment_relative_bin_end_min"].iloc[-1])
            else:
                baseline_start_min = baseline_end_min = baseline_rel_start_min = baseline_rel_end_min = np.nan

            out.loc[mask, "baseline_interval_start_min"] = baseline_start_min
            out.loc[mask, "baseline_interval_end_min"] = baseline_end_min
            out.loc[mask, "baseline_interval_start_treatment_relative_min"] = baseline_rel_start_min
            out.loc[mask, "baseline_interval_end_treatment_relative_min"] = baseline_rel_end_min
            out.loc[mask, "baseline_requested_bins"] = requested_bins
            out.loc[mask, "baseline_available_bins"] = baseline_available_bins
            out.loc[mask, "baseline_valid_bins"] = baseline_valid_bins
            out.loc[mask, "baseline_spike_count"] = baseline_spike_count
            out.loc[mask, "baseline_mean_amplitude_uv"] = baseline_mean
            out.loc[mask, "baseline_sd_amplitude_uv"] = baseline_sd
            out.loc[mask, "baseline_bin_mean_amplitude_uv"] = bin_mean
            out.loc[mask, "baseline_bin_sd_amplitude_uv"] = bin_sd
            out.loc[mask, "baseline_cv_percent"] = baseline_cv
            out.loc[mask, "baseline_slope_uv_per_min"] = slope
            out.loc[mask, "baseline_slope_percent_per_min"] = slope_percent
            out.loc[mask, "baseline_qc_pass"] = qc_pass
            out.loc[mask, "baseline_qc_reason"] = qc_reason

            if normalization_available:
                mean_values = pd.to_numeric(out.loc[mask, "mean_amplitude_uv"], errors="coerce")
                sem_values = pd.to_numeric(out.loc[mask, "sem_across_spikes_uv"], errors="coerce")
                out.loc[mask, "percent_of_baseline"] = 100.0 * mean_values / baseline_mean
                out.loc[mask, "percent_sem"] = 100.0 * sem_values / baseline_mean

    return out, warnings


def _waveform_bucket(records: list[dict]) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if not records:
        return None, None
    buckets: dict[tuple, list[np.ndarray]] = {}
    rel_by_key: dict[tuple, np.ndarray] = {}
    for record in records:
        rel_ms = np.asarray(record["rel_ms"], dtype=float)
        waveform = np.asarray(record["waveform"], dtype=float)
        if rel_ms.size != waveform.size:
            continue
        step = float(np.nanmedian(np.diff(rel_ms))) if rel_ms.size > 1 else 0.0
        key = (int(rel_ms.size), round(float(rel_ms[0]), 9), round(float(rel_ms[-1]), 9), round(step, 9))
        buckets.setdefault(key, []).append(waveform)
        rel_by_key[key] = rel_ms
    if not buckets:
        return None, None
    key = max(buckets, key=lambda candidate: len(buckets[candidate]))
    return rel_by_key[key], np.vstack(buckets[key])


def _plot_phase_waveforms(
    ax,
    waveform_rows: list[dict],
    channel: str,
    polarities: tuple[str, ...],
) -> None:
    plotted = False
    for polarity_short in polarities:
        color = EVENT_POLARITY_COLORS[polarity_short]
        polarity_name = EVENT_POLARITY_LABELS[polarity_short]
        for phase_group in ("baseline", "treatment", "post"):
            records = [
                row
                for row in waveform_rows
                if row["channel"] == channel
                and row["polarity_short"] == polarity_short
                and row["phase_group"] == phase_group
            ]
            rel_ms, waveforms = _waveform_bucket(records)
            if rel_ms is None or waveforms is None or waveforms.size == 0:
                continue
            mean = np.nanmean(waveforms, axis=0)
            label = f"{polarity_name} {AMPLITUDE_PHASE_LABELS.get(phase_group, phase_group)} (n={waveforms.shape[0]})"
            ax.plot(
                rel_ms,
                mean,
                color=color,
                linestyle=AMPLITUDE_PHASE_LINESTYLES.get(phase_group, "-"),
                linewidth=2.0,
                label=label,
            )
            plotted = True
    ax.axvline(0.0, color="black", alpha=0.35, linewidth=1)
    ax.set_ylabel("Spike-band voltage (uV)")
    if len(polarities) == 1:
        ax.set_xlabel(EVENT_POLARITY_AXIS_LABELS[polarities[0]])
    else:
        ax.set_xlabel("Time from canonical peak/trough (ms)")
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No edge-valid waveforms", transform=ax.transAxes, ha="center", va="center")


def _recording_axis_value(seconds: float, reference_start_s: float) -> float:
    if np.isfinite(reference_start_s):
        return (float(seconds) - reference_start_s) / 60.0
    return float(seconds) / 60.0


def _add_recording_boundaries_and_treatment_bar(
    ax,
    channel_rec_df: pd.DataFrame,
    reference_start_s: float,
    treatment_recorded: bool,
    boundary_label: str,
) -> None:
    if channel_rec_df.empty:
        return
    y0, y1 = ax.get_ylim()
    if not np.isfinite(y0) or not np.isfinite(y1) or math.isclose(y0, y1):
        y0, y1 = 80.0, 120.0
        ax.set_ylim(y0, y1)
    bar_y = y1 - 0.06 * (y1 - y0)
    has_baseline = (channel_rec_df["phase_group"].astype(str) == "baseline").any()
    has_post = (channel_rec_df["phase_group"].astype(str) == "post").any()
    first = True
    for rec in channel_rec_df.sort_values("recording_start_continuous_s").itertuples(index=False):
        start_x = _recording_axis_value(float(rec.recording_start_continuous_s), reference_start_s)
        end_x = _recording_axis_value(float(rec.recording_end_continuous_s), reference_start_s)
        mid_x = (start_x + end_x) / 2.0
        if not first:
            ax.axvline(start_x, color="black", linestyle="--", linewidth=0.8, alpha=0.45)
        first = False
        label = AMPLITUDE_PHASE_LABELS.get(str(rec.phase_group), str(rec.phase) or str(rec.recording_name))
        ax.text(mid_x, 0.98, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
        if str(rec.phase_group) == "treatment":
            ax.hlines(bar_y, start_x, end_x, color="black", linewidth=4, alpha=0.7)
            ax.text(mid_x, bar_y, " treatment ", ha="center", va="bottom", fontsize=8)
    if not treatment_recorded and has_baseline and has_post and np.isfinite(reference_start_s):
        boundary_text = (boundary_label or DEFAULT_UNRECORDED_TREATMENT_LABEL).strip()
        ax.axvline(0.0, color="tab:orange", linestyle="-", linewidth=1.4, alpha=0.9)
        ax.text(
            0.0,
            bar_y,
            boundary_text,
            ha="center",
            va="bottom",
            fontsize=8,
            color="tab:orange",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )


def _plot_amplitude_trajectory(
    ax,
    bin_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    channel: str,
    polarities: tuple[str, ...],
    boundary_label: str = DEFAULT_UNRECORDED_TREATMENT_LABEL,
) -> None:
    channel_rec_df = rec_df[rec_df["channel"].astype(str) == channel].copy()
    treatment = channel_rec_df[channel_rec_df["phase_group"] == "treatment"].sort_values("recording_start_continuous_s")
    treatment_start_s = float(treatment.iloc[0]["recording_start_continuous_s"]) if not treatment.empty else np.nan
    plot_refs = _amplitude_plot_reference_by_channel(channel_rec_df)
    plot_ref = plot_refs.get(channel, {})
    reference_start_s = float(plot_ref.get("reference_start_s", np.nan))
    treatment_recorded = bool(plot_ref.get("treatment_recorded", False))
    any_points = False
    for polarity_short in polarities:
        polarity_name = EVENT_POLARITY_LABELS[polarity_short]
        sub = bin_df[
            (bin_df["channel"].astype(str) == channel)
            & (bin_df["polarity_short"].astype(str) == polarity_short)
        ].sort_values("continuous_bin_mid_min")
        if sub.empty:
            continue
        x = pd.to_numeric(
            sub["plotted_relative_bin_mid_min"] if "plotted_relative_bin_mid_min" in sub else sub["continuous_bin_mid_min"],
            errors="coerce",
        ).to_numpy(dtype=float)
        y = pd.to_numeric(sub["percent_of_baseline"], errors="coerce").to_numpy(dtype=float)
        yerr = pd.to_numeric(sub["percent_sem"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        with_sem = valid & np.isfinite(yerr)
        color = EVENT_POLARITY_COLORS[polarity_short]
        if np.any(with_sem):
            ax.errorbar(
                x[with_sem],
                y[with_sem],
                yerr=yerr[with_sem],
                marker="o",
                linestyle="-",
                linewidth=1.5,
                capsize=3,
                color=color,
                label=f"{polarity_name} amplitude",
            )
            any_points = True
        point_only = valid & ~np.isfinite(yerr)
        if np.any(point_only):
            ax.plot(
                x[point_only],
                y[point_only],
                marker="o",
                linestyle="none",
                color=color,
                label=f"{polarity_name} amplitude (n=1)",
            )
            any_points = True
    ax.axhline(100.0, color="black", linestyle=":", linewidth=1.1, alpha=0.8)
    _add_recording_boundaries_and_treatment_bar(
        ax,
        channel_rec_df,
        reference_start_s,
        treatment_recorded=treatment_recorded,
        boundary_label=boundary_label,
    )
    ax.set_ylabel("Amplitude (% of baseline)")
    if np.isfinite(treatment_start_s):
        ax.set_xlabel("Time from treatment start (min)")
    elif np.isfinite(reference_start_s):
        ax.set_xlabel("Recorded time relative to post-treatment recording (min)")
    else:
        ax.set_xlabel("Continuous time (min)")
    ax.grid(True, alpha=0.25)
    if any_points:
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No normalized amplitude values", transform=ax.transAxes, ha="center", va="center")


def _qc_reasons_for_figure(bin_df: pd.DataFrame, channel: str, polarities: tuple[str, ...]) -> list[str]:
    reasons: list[str] = []
    for polarity_short in polarities:
        polarity_name = EVENT_POLARITY_LABELS[polarity_short]
        sub = bin_df[
            (bin_df["channel"].astype(str) == channel)
            & (bin_df["polarity_short"].astype(str) == polarity_short)
        ]
        if sub.empty:
            continue
        qc_pass = sub["baseline_qc_pass"].dropna()
        reason = str(sub["baseline_qc_reason"].dropna().iloc[0]) if sub["baseline_qc_reason"].dropna().size else ""
        if reason and not (qc_pass.astype(str).str.lower() == "true").all():
            reasons.append(f"{polarity_name}: {reason}")
    return reasons


def _plot_spike_amplitude_timecourse_figures(
    out_dir: str,
    bin_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    waveform_rows: list[dict],
    output_polarities: tuple[str, ...],
    overlay: bool,
    boundary_label: str = DEFAULT_UNRECORDED_TREATMENT_LABEL,
    channels: list[str] | None = None,
) -> list[str]:
    if bin_df.empty:
        return []
    figure_dir = os.path.join(out_dir, "spike_diagnostics", "amplitude_time_course")
    os.makedirs(figure_dir, exist_ok=True)
    plot_paths: list[str] = []

    selected_channels = [c for c in sorted(bin_df["channel"].dropna().astype(str).unique()) if channels is None or c in set(channels)]
    for channel in selected_channels:
        if overlay and len(output_polarities) > 1:
            figure_jobs = [(output_polarities, "overlay")]
        else:
            figure_jobs = [((polarity_short,), EVENT_POLARITY_LABELS[polarity_short]) for polarity_short in output_polarities]

        for polarities, suffix in figure_jobs:
            fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False, gridspec_kw={"height_ratios": [1.0, 1.35]})
            _plot_phase_waveforms(axes[0], waveform_rows, channel, polarities)
            _plot_amplitude_trajectory(axes[1], bin_df, rec_df, channel, polarities, boundary_label=boundary_label)
            reasons = _qc_reasons_for_figure(bin_df, channel, polarities)
            title = f"Spike amplitude change from baseline - {channel}"
            if suffix != "overlay":
                title += f" - {suffix}"
            else:
                title += " - positive and negative"
            if reasons:
                title += "\nBaseline QC warning: " + "; ".join(reasons[:2])
            fig.suptitle(title)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            plot_path = os.path.join(
                figure_dir,
                f"{_safe_name(channel)}__spike_amplitude_percent_baseline_{_safe_name(suffix)}.png",
            )
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)
            plot_paths.append(plot_path)
    return plot_paths


def build_spike_amplitude_timecourse_outputs(
    out_dir: str,
    recording_rows: list[dict],
    event_rows: list[dict],
    waveform_rows: list[dict],
    requested_polarity: str | None,
    overlay: bool = False,
    boundary_label: str | None = None,
    plot_figures: bool = True,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    output_polarities = requested_event_polarities(requested_polarity)
    boundary_label = (boundary_label or DEFAULT_UNRECORDED_TREATMENT_LABEL).strip() or DEFAULT_UNRECORDED_TREATMENT_LABEL
    rec_df = _timeline_recordings(recording_rows)
    events = _event_frame_with_continuous_time(event_rows, rec_df)
    bin_df = _complete_amplitude_bins(rec_df, events, output_polarities, boundary_label=boundary_label)
    bin_df, warnings = _apply_spike_amplitude_baselines(bin_df, events, output_polarities)

    csv_path = os.path.join(out_dir, "spike_amplitude_time_course.csv")
    pd.DataFrame(bin_df, columns=SPIKE_AMPLITUDE_TIMECOURSE_COLUMNS).to_csv(csv_path, index=False)
    waveform_csv_path = os.path.join(out_dir, "spike_amplitude_waveforms.csv")
    _write_amplitude_waveform_csv(waveform_csv_path, waveform_rows)
    plot_paths: list[str] = []
    if plot_figures:
        plot_paths = _plot_spike_amplitude_timecourse_figures(
            out_dir,
            bin_df,
            rec_df,
            waveform_rows,
            output_polarities,
            overlay=bool(overlay),
            boundary_label=boundary_label,
        )
    return {
        "csv_path": csv_path,
        "waveform_csv_path": waveform_csv_path,
        "plot_paths": plot_paths,
        "warnings": warnings,
    }


AMPLITUDE_WAVEFORM_CSV_COLUMNS = [
    "channel",
    "phase_group",
    "polarity_short",
    "n_spikes",
    "rel_ms",
    "mean_waveform_uv",
    "sem_waveform_uv",
]


def _write_amplitude_waveform_csv(path: str, waveform_rows: list[dict], include_sem: bool = True) -> str:
    """Persist per (channel, phase_group, polarity) mean waveforms for later replotting."""
    grouped: dict[str, dict] = {}
    for row in waveform_rows:
        channel = str(row.get("channel", ""))
        phase_group = str(row.get("phase_group", ""))
        polarity_short = str(row.get("polarity_short", ""))
        key = (channel, phase_group, polarity_short)
        grouped.setdefault(key, []).append(row)

    rows: list[dict] = []
    for (channel, phase_group, polarity_short), records in sorted(grouped.items()):
        rel_ms, waveforms = _waveform_bucket(records)
        if rel_ms is None or waveforms is None or waveforms.shape[0] == 0:
            continue
        mean = np.nanmean(waveforms, axis=0)
        if include_sem and waveforms.shape[0] > 1:
            sem = np.nanstd(waveforms, axis=0, ddof=1) / math.sqrt(waveforms.shape[0])
        else:
            sem = np.zeros_like(mean)
        rows.append(
            {
                "channel": channel,
                "phase_group": phase_group,
                "polarity_short": polarity_short,
                "n_spikes": int(waveforms.shape[0]),
                "rel_ms": ",".join(f"{v:.6g}" for v in rel_ms),
                "mean_waveform_uv": ",".join(f"{v:.6g}" for v in mean),
                "sem_waveform_uv": ",".join(f"{v:.6g}" for v in sem),
            }
        )
    pd.DataFrame(rows, columns=AMPLITUDE_WAVEFORM_CSV_COLUMNS).to_csv(path, index=False)
    return path


def replot_spike_amplitude_timecourse(
    input_csv: str,
    waveform_csv: str | None = None,
    out_dir: str | None = None,
    boundary_label: str | None = None,
    overlay: bool = False,
    channels: list[str] | None = None,
    polarity: str | None = None,
) -> dict:
    """Replot spike-amplitude change-from-baseline figures from persisted time-course CSVs.

    channels: restrict figures to these channel names; None plots all channels.
    polarity: "both" / "pos" / "neg"; None infers from the CSV contents.
    """
    import os

    bin_df = pd.read_csv(input_csv)
    if bin_df.empty:
        return {"plot_paths": [], "warnings": ["Empty amplitude time-course CSV."]}
    if channels:
        bin_df = bin_df[bin_df["channel"].astype(str).isin(set(channels))]
        if bin_df.empty:
            return {"plot_paths": [], "warnings": ["Selected channels not found in the amplitude time-course CSV."]}
    os.makedirs(out_dir or os.path.dirname(input_csv) or ".", exist_ok=True)
    out_dir = out_dir or os.path.dirname(input_csv) or "."
    boundary_label = (boundary_label or DEFAULT_UNRECORDED_TREATMENT_LABEL).strip() or DEFAULT_UNRECORDED_TREATMENT_LABEL
    rec_df = _recording_timeline_from_bins(bin_df)
    inferred_polarity = _polarity_from_amp_bins(bin_df)
    output_polarities = requested_event_polarities(polarity or inferred_polarity)
    waveform_rows: list[dict] = []
    if waveform_csv and os.path.exists(waveform_csv):
        waveform_rows = _waveform_rows_from_csv(waveform_csv)
    plot_paths = _plot_spike_amplitude_timecourse_figures(
        out_dir,
        bin_df,
        rec_df,
        waveform_rows,
        output_polarities,
        overlay=bool(overlay),
        boundary_label=boundary_label,
        channels=channels,
    )
    return {
        "csv_path": input_csv,
        "waveform_csv_path": waveform_csv or "",
        "plot_paths": plot_paths,
        "warnings": [],
    }


# =========================
# Spike diagnostic figure replot (tab-4)
# =========================


def _channels_from_events_csv(events_csv: str) -> list[str]:
    """Sorted unique channel names present in a spike_events.csv."""
    if not os.path.exists(events_csv):
        return []
    try:
        df = pd.read_csv(events_csv, usecols=["channel"])
    except (ValueError, KeyError):
        return []
    return sorted(str(value) for value in df["channel"].dropna().unique())


def replot_spike_diagnostic_outputs(
    spike_events_csv: str,
    out_dir: str | None = None,
    plot_types: tuple[str, ...] = ("raster", "rate", "isi", "acg", "waveform"),
    channels: list[str] | None = None,
) -> dict:
    """Replot raster/rate/ISI/ACG/average-waveform diagnostic figures from spike_events.csv.

    Reads the input-adjacent spike_events.csv plus the per-recording average-waveform
    CSVs written by the processing step. One figure per diagnostic type per selected
    channel, with recordings concatenated along continuous time (raster/rate) or
    pooled (ISI/ACG/average waveform).

    plot_types: subset of ("raster", "rate", "isi", "acg", "waveform").
    channels: restrict figures to these channel names; None plots all channels.
    """
    if not os.path.exists(spike_events_csv):
        return {"plot_paths": [], "warnings": [f"Spike events CSV not found: {spike_events_csv}"]}
    events_csv = spike_events_csv
    data_dir = os.path.dirname(events_csv) or "."
    out_dir = out_dir or data_dir
    diagnostic_dir = os.path.join(out_dir, "spike_diagnostics")
    event_dir = os.path.join(diagnostic_dir, "spike_diagnostics")
    os.makedirs(event_dir, exist_ok=True)

    events = pd.read_csv(events_csv)
    if events.empty:
        return {"plot_paths": [], "warnings": ["Empty spike events CSV."]}
    for column in ("recording_name", "channel", "spike_time_s", "polarity_short", "recording_relative_spike_time_s"):
        if column not in events:
            return {"plot_paths": [], "warnings": [f"spike_events.csv missing required column: {column}"]}
    events["recording_name"] = events["recording_name"].astype(str)
    events["channel"] = events["channel"].astype(str)
    events["spike_time_s"] = pd.to_numeric(events["spike_time_s"], errors="coerce")
    events["recording_relative_spike_time_s"] = pd.to_numeric(
        events["recording_relative_spike_time_s"], errors="coerce"
    )
    if "polarity_short" in events:
        events["polarity_short"] = events["polarity_short"].astype(str)

    selected_channels = _channels_from_events_csv(events_csv)
    if channels:
        selected_channels = [c for c in selected_channels if c in set(channels)]

    allowed = {t for t in plot_types}
    plot_paths: list[str] = []
    warnings: list[str] = []

    for channel in selected_channels:
        channel_events = events[events["channel"] == channel].copy()
        if channel_events.empty:
            continue
        for kind in ("raster", "rate", "isi", "acg"):
            if kind in allowed:
                path = _plot_diagnostic_channel_figure(
                    event_dir, kind, channel, channel_events
                )
                if path:
                    plot_paths.append(path)
        if "waveform" in allowed:
            path = _plot_average_waveform_channel_figure(data_dir, out_dir, channel)
            if path:
                plot_paths.append(path)

    if not plot_paths:
        warnings.append("No diagnostic figures generated for the selected channels/types.")
    return {"plot_paths": plot_paths, "warnings": warnings}


def _plot_diagnostic_channel_figure(
    fig_dir: str,
    kind: str,
    channel: str,
    events: pd.DataFrame,
) -> str:
    """Draw one diagnostic figure (raster/rate/isi/acg) for a channel into fig_dir."""
    polarity_shorts = sorted(
        str(p) for p in events["polarity_short"].dropna().unique() if p in EVENT_POLARITY_LABELS
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    cursor = 0.0
    recording_bounds: list[tuple[float, str]] = []
    event_row = 0
    for rec_name, sub in events.groupby("recording_name", sort=False):
        rel = sub["recording_relative_spike_time_s"].dropna().to_numpy(dtype=float)
        if rel.size == 0:
            continue
        dur = float(rel.max() - rel.min()) if rel.size > 1 else 0.0
        if dur <= 0:
            dur = 1.0
        recording_bounds.append((cursor, rec_name))
        start = cursor
        cursor += dur
        if kind == "raster":
            pol_arr = sub["polarity_short"].dropna().astype(str).to_numpy(dtype=str)
            rel_full = sub["recording_relative_spike_time_s"].dropna().to_numpy(dtype=float)
            for idx, (t, polarity_short) in enumerate(zip(rel_full, pol_arr)):
                color = EVENT_POLARITY_COLORS.get(polarity_short, "tab:gray")
                ax.vlines(start + float(t), event_row + 0.5, event_row + 1.5, color=color, linewidth=0.6)
                event_row += 1
        if kind == "rate":
            rate_bin_s = float(SPIKE_RATE_BIN_S)
            edges = np.arange(rel.min(), rel.max() + rate_bin_s, rate_bin_s)
            centers = edges[:-1] + rate_bin_s / 2.0
            for polarity_short in polarity_shorts:
                pol_rel = sub.loc[sub["polarity_short"].astype(str) == polarity_short, "recording_relative_spike_time_s"].to_numpy(dtype=float)
                pol_counts, _ = np.histogram(pol_rel, bins=edges)
                ax.plot(
                    start + centers,
                    pol_counts,
                    color=EVENT_POLARITY_COLORS.get(polarity_short, "tab:gray"),
                    label=f"{EVENT_POLARITY_LABELS[polarity_short]} (per {rate_bin_s:g}s)",
                    linewidth=1.4,
                )

    if kind == "raster":
        ax.set_ylabel("Event index")
        ax.set_ylim(0, event_row + 1)
        ax.set_title(f"Spike raster - {channel}")
    elif kind == "rate":
        ax.set_ylabel("Spike rate (Hz)")
        ax.set_title(f"Spike rate - {channel}")
        ax.legend(loc="best", fontsize=8)
    elif kind == "isi":
        rel = events["recording_relative_spike_time_s"].dropna().to_numpy(dtype=float)
        rel = np.sort(rel)
        d = np.diff(rel)
        d = d[d > 0] * 1000.0
        if d.size:
            ax.hist(d, bins=40, color="tab:blue", alpha=0.7)
            ax.set_xlabel("Inter-spike interval (ms)")
            ax.set_ylabel("Count")
        ax.set_title(f"Inter-spike interval - {channel}")
    elif kind == "acg":
        rel = events["recording_relative_spike_time_s"].dropna().to_numpy(dtype=float)
        centers, counts = spike_autocorrelogram(rel)
        if centers.size:
            ax.vlines(centers, 0, counts, color="tab:blue", linewidth=1.2)
            ax.set_xlabel("Lag (ms)")
            ax.set_ylabel("Count")
        ax.set_title(f"Autocorrelogram - {channel}")

    for start, rec_name in recording_bounds:
        ax.axvline(start, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
        if recording_bounds and start == recording_bounds[0][0]:
            continue
        ax.text(start, 0.98, rec_name, transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=7, rotation=90)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    safe_ch = _safe_name(channel)
    path = os.path.join(fig_dir, f"spike_{kind}_{safe_ch}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _plot_average_waveform_channel_figure(data_dir: str, out_dir: str, channel: str) -> str:
    """Pool per-recording average-waveform CSVs for a channel and draw mean+SEM."""
    import glob

    wave_dir = os.path.join(data_dir, "spike_diagnostics", "average_waveforms")
    pattern = os.path.join(wave_dir, f"*__{_safe_name(channel)}__*_average_waveform.csv")
    rel_ms: np.ndarray | None = None
    means: list[np.ndarray] = []
    sems: list[np.ndarray] = []
    ns: list[float] = []
    for path in glob.glob(pattern):
        df = pd.read_csv(path)
        if {"time_ms", "mean_waveform_uv"}.issubset(df.columns):
            rel_ms = df["time_ms"].to_numpy(dtype=float)
            mean = df["mean_waveform_uv"].to_numpy(dtype=float)
            sem = df["sem_waveform_uv"].to_numpy(dtype=float) if "sem_waveform_uv" in df else np.zeros_like(mean)
            n = float(int(df["n_spikes_used"].iloc[0])) if "n_spikes_used" in df else 1.0
            means.append(mean)
            sems.append(sem)
            ns.append(n)
    if rel_ms is None or not means:
        return ""
    x = rel_ms
    y = np.average(np.vstack(means), axis=0, weights=ns)
    pooled_sem = np.sqrt(
        np.average(np.square(np.vstack(sems)), axis=0, weights=ns)
    ) if ns else np.zeros_like(x)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y, color="tab:blue", linewidth=2.0)
    ax.fill_between(x, y - pooled_sem, y + pooled_sem, color="tab:blue", alpha=0.25)
    ax.axvline(0.0, color="black", alpha=0.35, linewidth=1)
    ax.axhline(0.0, color="black", alpha=0.35, linewidth=1)
    ax.set_xlabel("Time relative to peak/trough (ms)")
    ax.set_ylabel("Voltage (µV)")
    ax.set_title(f"Average spike waveform (mean±SEM) - {channel}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig_dir = os.path.join(out_dir, "spike_diagnostics", "spike_diagnostics")
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, f"spike_average_waveform_{_safe_name(channel)}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _channels_from_amp_csv(input_csv: str) -> list[str]:
    """Sorted unique channel names present in a spike-amplitude time-course CSV."""
    if not os.path.exists(input_csv):
        return []
    try:
        bin_df = pd.read_csv(input_csv, usecols=["channel"])
    except (ValueError, KeyError):
        return []
    return sorted(str(value) for value in bin_df["channel"].dropna().unique())


def _polarity_from_amp_bins(bin_df: pd.DataFrame) -> str | None:
    if bin_df.empty or "polarity_short" not in bin_df:
        return None
    present = sorted(str(value) for value in bin_df["polarity_short"].dropna().unique())
    if len(present) == 2 and "pos" in present and "neg" in present:
        return "both"
    return present[0] if present else None


def _recording_timeline_from_bins(bin_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct a recording timeline (continuous start/end per channel) from bin rows."""
    if bin_df.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    group_cols = [col for col in ("channel", "recording_order", "recording_name") if col in bin_df.columns]
    for _key, sub in bin_df.groupby(group_cols, sort=False):
        first = sub.sort_values("continuous_bin_start_min").iloc[0]
        start_min = pd.to_numeric(sub["continuous_bin_start_min"], errors="coerce").min()
        end_min = pd.to_numeric(sub["continuous_bin_end_min"], errors="coerce").max()
        rows.append(
            {
                "channel": str(first.get("channel", "")),
                "recording_order": int(first.get("recording_order", 1)),
                "recording_name": str(first.get("recording_name", "")),
                "epoch_label": str(first.get("epoch_label", "") or ""),
                "phase": str(first.get("phase", "") or ""),
                "phase_group": str(first.get("phase_group", "") or ""),
                "normalized_phase": str(first.get("normalized_phase", "") or ""),
                "recording_start_continuous_s": float(start_min) * 60.0 if np.isfinite(start_min) else np.nan,
                "recording_end_continuous_s": float(end_min) * 60.0 if np.isfinite(end_min) else np.nan,
                "recording_duration_s": (float(end_min) - float(start_min)) * 60.0 if np.isfinite(start_min) and np.isfinite(end_min) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _waveform_rows_from_csv(waveform_csv: str) -> list[dict]:
    """Load persisted mean waveforms back into per-group waveform rows for plotting."""
    df = pd.read_csv(waveform_csv)
    rows: list[dict] = []
    for row in df.itertuples(index=False):
        rel_ms = np.array([float(value) for value in str(getattr(row, "rel_ms", "")).split(",") if value])
        mean = np.array([float(value) for value in str(getattr(row, "mean_waveform_uv", "")).split(",") if value])
        if rel_ms.size == 0 or mean.size == 0 or rel_ms.size != mean.size:
            continue
        rows.append(
            {
                "channel": str(getattr(row, "channel", "")),
                "phase_group": str(getattr(row, "phase_group", "")),
                "polarity_short": str(getattr(row, "polarity_short", "")),
                "waveform": mean,
                "rel_ms": rel_ms,
            }
        )
    return rows


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


def _phase_from_epoch_label(label):
    phase_group, _warning_reason = normalize_amplitude_phase(epoch_label=label)
    return "" if phase_group == PHASE_UNKNOWN else phase_group


def _normalise_metadata_lookup(recording_metadata_by_path=None):
    lookup = {}
    if not recording_metadata_by_path:
        return lookup
    items = recording_metadata_by_path.values() if isinstance(recording_metadata_by_path, dict) else recording_metadata_by_path
    for item in items:
        if not isinstance(item, dict):
            continue
        path = item.get("source_file") or item.get("path")
        if not path:
            continue
        meta = dict(item)
        lookup[str(path)] = meta
        lookup[os.path.abspath(str(path))] = meta
        lookup[os.path.basename(str(path))] = meta
    return lookup


def _metadata_for_path(csv_path, lookup):
    return (
        lookup.get(os.path.abspath(csv_path))
        or lookup.get(csv_path)
        or lookup.get(os.path.basename(csv_path))
        or {}
    )


def process_csvs(
    csv_paths,
    out_dir=None,
    epoch_labels_by_path=None,
    window_sec=None,
    polarity=None,
    classification_window_ms=None,
    plot_spike_amplitude_change=False,
    overlay_spike_amplitudes=False,
    spike_amplitude_boundary_label=None,
    spike_events_path=None,
    recording_metadata_by_path=None,
    return_outputs=False,
    shape_settings=None,
):
    """Process an ordered list of raw CSV files and return the combined output CSV path."""
    ordered_csv_paths = list(csv_paths)
    if not ordered_csv_paths:
        raise FileNotFoundError("No CSV files were provided for spike counting.")

    selected_polarity = normalize_polarity(polarity)
    shape_config = shape_qc.settings(shape_settings)
    selected_window_sec = float(window_sec if window_sec is not None else WINDOW_SEC)
    if selected_window_sec <= 0:
        raise ValueError("Spike-count window size must be greater than zero.")
    selected_window_min = selected_window_sec / 60.0
    selected_window_label = format_window_label(selected_window_sec)

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(ordered_csv_paths[0]) or ".", OUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    epoch_lookup = _normalise_label_lookup(epoch_labels_by_path)
    metadata_lookup = _normalise_metadata_lookup(recording_metadata_by_path)
    results_rows = []
    spike_event_rows = []
    diagnostic_output_rows = []
    waveform_summary_rows = []
    shape_audit_dirs = []
    amplitude_recording_rows: list[dict] = []
    amplitude_event_rows: list[dict] = []
    amplitude_waveform_rows: list[dict] = []
    amplitude_outputs = {
        "csv_path": "",
        "plot_paths": [],
        "warnings": [],
    }

    print(f"[settings] spike polarity: {polarity_label(selected_polarity)} ({selected_polarity})")

    for rec_idx, csv_path in enumerate(ordered_csv_paths, start=1):
        rec_name = os.path.splitext(os.path.basename(csv_path))[0]
        rec_meta = _metadata_for_path(csv_path, metadata_lookup)
        epoch_label = _epoch_label_for_path(csv_path, epoch_lookup)
        if not epoch_label:
            epoch_label = str(rec_meta.get("epoch_label") or rec_meta.get("label") or "")
        phase = str(rec_meta.get("phase") or "").strip() or _phase_from_epoch_label(epoch_label)
        phase_group, phase_warning_reason = normalize_amplitude_phase(phase=phase, epoch_label=epoch_label)
        segment_id = str(rec_meta.get("segment_id") or rec_meta.get("recording_id") or rec_name)
        preparation_id = str(rec_meta.get("preparation_id") or "")

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
            print(f"[detect] {rec_name} | {ch}: {len(t):,} samples", flush=True)
            x = df[ch].to_numpy(dtype=float)

            if np.any(~np.isfinite(x)):
                s = pd.Series(x)
                x = s.interpolate(limit_direction="both").to_numpy(dtype=float)

            t0 = float(t[0])
            t1 = float(t[-1])
            details = detect_spikes_with_details(
                x,
                t,
                polarity=selected_polarity,
                return_filtered_trace=True,
                classification_window_ms=classification_window_ms,
                shape_settings=shape_config,
            )
            peaks = details["peaks"]
            spike_times = details["spike_times"]
            fs = details["fs"]
            event_polarities = np.asarray(details.get("event_polarity", []), dtype=str)
            positive_times = spike_times[event_polarities == "pos"] if spike_times.size else np.array([], dtype=float)
            negative_times = spike_times[event_polarities == "neg"] if spike_times.size else np.array([], dtype=float)
            n_positive_total = int(details.get("n_positive", positive_times.size))
            n_negative_total = int(details.get("n_negative", negative_times.size))
            recording_key = f"{rec_idx}::{ch}"
            amplitude_recording_rows.append(
                spike_amplitude_recording_row(
                    recording_key=recording_key,
                    rec_idx=rec_idx,
                    rec_name=rec_name,
                    channel=ch,
                    epoch_label=epoch_label,
                    phase=phase,
                    phase_group=phase_group,
                    phase_warning_reason=phase_warning_reason,
                    t_s=t,
                    fs=fs,
                )
            )
            amplitude_event_rows.extend(
                spike_amplitude_event_rows(
                    recording_key=recording_key,
                    rec_idx=rec_idx,
                    rec_name=rec_name,
                    channel=ch,
                    epoch_label=epoch_label,
                    phase=phase,
                    phase_group=phase_group,
                    phase_warning_reason=phase_warning_reason,
                    t0=t0,
                    details=details,
                )
            )
            amplitude_waveform_rows.extend(spike_amplitude_waveform_rows(ch, phase_group, details))
            outputs, waveform_summaries = write_spike_diagnostic_outputs(
                out_dir, rec_name, ch, details
            )
            diagnostic_output_rows.extend(outputs)
            for waveform_summary in waveform_summaries:
                waveform_summary.update(
                    {
                        "recording_index": rec_idx,
                        "recording_order": rec_idx,
                        "epoch_label": epoch_label,
                        "phase": phase,
                        "preparation_id": preparation_id,
                        "segment_id": segment_id,
                        "source_file": csv_path,
                    }
                )
                waveform_summary_rows.append(waveform_summary)

            for event_index, sample_index in enumerate(peaks, start=1):
                spike_time = float(details["spike_times"][event_index - 1])
                polarity_short = str(details["event_polarity"][event_index - 1])
                spike_event_rows.append(
                    {
                            "recording_index": rec_idx,
                            "recording_order": rec_idx,
                            "preparation_id": preparation_id,
                            "segment_id": segment_id,
                            "recording_id": rec_name,
                            "recording_name": rec_name,
                            "epoch_label": epoch_label,
                            "phase": phase,
                            "channel": ch,
                            "unit_id": "threshold_mua",
                            "event_index": event_index,
                            "spike_time_s": spike_time,
                            "segment_spike_time_s": float(spike_time - t0),
                            "recording_relative_spike_time_s": float(spike_time - t0),
                            "source_sampling_rate_hz": float(fs),
                            "sample_index": int(sample_index),
                            "canonical_sample_index": int(sample_index),
                            "original_sample_index": int(sample_index),
                            "candidate_sample_index": int(details["candidate_sample_index"][event_index - 1]),
                            "alignment_shift_samples": int(details["alignment_shift_samples"][event_index - 1]),
                            "candidate_polarity": str(details["candidate_polarity"][event_index - 1]),
                            "polarity": EVENT_POLARITY_LABELS.get(polarity_short, polarity_short),
                            "polarity_short": polarity_short,
                            "requested_polarity": selected_polarity,
                            "amplitude_uv": float(details["amplitude_uv"][event_index - 1]),
                            "signed_amplitude_uv": float(details["signed_amplitude_uv"][event_index - 1]),
                            "robust_z": float(details["robust_z"][event_index - 1]),
                            "candidate_robust_z": float(details["candidate_robust_z"][event_index - 1]),
                            "local_max_uv": float(details["local_max_uv"][event_index - 1]),
                            "local_max_sample_index": int(details["local_max_sample_index"][event_index - 1]),
                            "local_min_uv": float(details["local_min_uv"][event_index - 1]),
                            "local_min_sample_index": int(details["local_min_sample_index"][event_index - 1]),
                            "n_positive": n_positive_total,
                            "n_negative": n_negative_total,
                            "threshold_z": float(SPIKE_Z_THR),
                            "refractory_ms": float(REFRACTORY_MS),
                            "classification_window_ms": float(details.get("classification_window_ms", SPIKE_CLASSIFICATION_WINDOW_MS)),
                            "amp_min_uv": AMP_MIN_UV,
                            "amp_max_uv": AMP_MAX_UV,
                            "qc_pass": True,
                            "qc_reason": "threshold_crossing_mua",
                            "source_file": csv_path,
                        }
                    )
            print(
                f"[{rec_name} | {ch}] detected spikes: {len(spike_times)} "
                f"(positive={n_positive_total}, negative={n_negative_total}, fs~{fs:.2f} Hz)"
            )

            sample_interval_s = 1.0 / float(fs) if np.isfinite(fs) and fs > 0 else 0.0
            window_bounds = folded_window_bounds(t0, t1, selected_window_sec, sample_interval_s)
            if shape_config["mode"] != "off":
                shape_audit_dirs.append(shape_qc.write_audit(
                    Path(out_dir) / "spike_shape_qc" / shape_config["mode"] / f"{rec_idx:03d}_{_safe_name(rec_name)}" / _safe_name(ch),
                    rec_name, ch, details["filtered_trace_uv"], t, fs, details["shape_reports"],
                    details["shape_before_peaks"], peaks, window_bounds, shape_config))

            for w, (a, b) in enumerate(window_bounds):
                cnt = int(np.sum((spike_times >= a) & (spike_times < b)))
                start_min = (a - t0) / 60.0
                end_min = (b - t0) / 60.0

                row = {
                    "recording_index": rec_idx,
                    "recording_order": rec_idx,
                    "recording_name": rec_name,
                    "epoch_label": epoch_label,
                    "phase": phase,
                    "channel": ch,
                    "window_index": w,
                    "window_start_s": a,
                    "window_end_s": b,
                    "segment_window_start_s": a - t0,
                    "segment_window_end_s": b - t0,
                    "segment_window_mid_s": ((a + b) / 2.0) - t0,
                    "window_duration_s": b - a,
                    "window_label": f"Min {start_min:g}-{end_min:g}",
                    "spike_count": cnt,
                    "n_positive": int(np.sum((positive_times >= a) & (positive_times < b))),
                    "n_negative": int(np.sum((negative_times >= a) & (negative_times < b))),
                    "fs_est_hz": float(fs),
                    "source_sampling_rate_hz": float(fs),
                    "z_thr": float(SPIKE_Z_THR),
                    "polarity": selected_polarity,
                    "refractory_ms": float(REFRACTORY_MS),
                    "classification_window_ms": float(details.get("classification_window_ms", SPIKE_CLASSIFICATION_WINDOW_MS)),
                    "hp_lo_hz": float(HP_SPIKE_BAND[0]),
                    "hp_hi_hz": float(HP_SPIKE_BAND[1]),
                    "amp_min_uv": AMP_MIN_UV,
                    "amp_max_uv": AMP_MAX_UV,
                    "w_min_ms": W_MIN_MS,
                    "w_max_ms": W_MAX_MS,
                }
                if rec_meta:
                    row.update(
                        {
                            "preparation_id": preparation_id,
                            "segment_id": segment_id,
                            "recording_id": rec_name,
                            "source_file": csv_path,
                        }
                    )
                results_rows.append(row)

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

    events_csv = spike_events_path or os.path.join(out_dir, "spike_events.csv")
    event_columns = [
            "recording_index",
            "recording_order",
            "preparation_id",
            "segment_id",
            "recording_id",
            "recording_name",
            "epoch_label",
            "phase",
            "channel",
            "unit_id",
            "event_index",
            "spike_time_s",
            "segment_spike_time_s",
            "recording_relative_spike_time_s",
            "source_sampling_rate_hz",
            "sample_index",
            "canonical_sample_index",
            "original_sample_index",
            "candidate_sample_index",
            "alignment_shift_samples",
            "candidate_polarity",
            "polarity",
            "polarity_short",
            "requested_polarity",
            "amplitude_uv",
            "signed_amplitude_uv",
            "robust_z",
            "candidate_robust_z",
            "local_max_uv",
            "local_max_sample_index",
            "local_min_uv",
            "local_min_sample_index",
            "n_positive",
            "n_negative",
            "threshold_z",
            "refractory_ms",
            "classification_window_ms",
            "amp_min_uv",
            "amp_max_uv",
            "qc_pass",
            "qc_reason",
            "source_file",
        ]
    pd.DataFrame(spike_event_rows, columns=event_columns).to_csv(events_csv, index=False)
    print(f"[events] wrote threshold-crossing MUA spike events:\n  {events_csv}")

    amplitude_outputs = build_spike_amplitude_timecourse_outputs(
        out_dir,
        amplitude_recording_rows,
        amplitude_event_rows,
        amplitude_waveform_rows,
        selected_polarity,
        overlay=overlay_spike_amplitudes,
        boundary_label=spike_amplitude_boundary_label,
        plot_figures=False,
    )
    if amplitude_outputs.get("csv_path"):
        diagnostic_output_rows.append(
            {
                "recording_name": "ALL_RECORDINGS",
                "channel": "ALL_CHANNELS",
                "output_type": "spike_amplitude_time_course_csv",
                "path": amplitude_outputs["csv_path"],
            }
        )
        print(f"[amplitude] wrote spike-amplitude time-course CSV:\n  {amplitude_outputs['csv_path']}")
    if amplitude_outputs.get("waveform_csv_path"):
        diagnostic_output_rows.append(
            {
                "recording_name": "ALL_RECORDINGS",
                "channel": "ALL_CHANNELS",
                "output_type": "spike_amplitude_waveforms_csv",
                "path": amplitude_outputs["waveform_csv_path"],
            }
        )
    for plot_path in amplitude_outputs.get("plot_paths", []):
        diagnostic_output_rows.append(
            {
                "recording_name": "ALL_RECORDINGS",
                "channel": "ALL_CHANNELS",
                "output_type": "spike_amplitude_time_course_plot",
                "path": plot_path,
            }
        )
    for warning in amplitude_outputs.get("warnings", []):
        print(f"[amplitude warning] {warning}")

    diagnostics_csv = os.path.join(out_dir, "spike_diagnostic_outputs.csv")
    pd.DataFrame(
        diagnostic_output_rows,
        columns=["recording_name", "channel", "output_type", "path"],
    ).to_csv(diagnostics_csv, index=False)

    waveform_summary_csv = os.path.join(out_dir, "spike_waveform_summary.csv")
    pd.DataFrame(
        waveform_summary_rows,
        columns=[
            "recording_index",
            "recording_order",
            "preparation_id",
            "segment_id",
            "recording_name",
            "epoch_label",
            "phase",
            "channel",
            "polarity",
            "polarity_short",
            "detected_spikes",
            "total_spikes",
            "n_positive",
            "n_negative",
            "polarity_spike_count",
            "waveform_spikes_used",
            "edge_excluded_spikes",
            "sampling_rate_hz",
            "pre_ms",
            "post_ms",
            "classification_window_ms",
            "mean_peak_uv",
            "mean_peak_abs_uv",
            "mean_peak_to_peak_uv",
            "waveform_csv",
            "waveform_plot",
            "source_file",
        ],
    ).to_csv(waveform_summary_csv, index=False)

    print(f"\n[done] wrote:\n  {out_csv}\n  plots in: {out_dir}")
    outputs_info = {
        "spike_shape_audit_dirs": shape_audit_dirs,
        "spike_count_csv": out_csv,
        "spike_events_csv": events_csv,
        "spike_diagnostic_outputs_csv": diagnostics_csv,
        "spike_waveform_summary_csv": waveform_summary_csv,
        "spike_amplitude_timecourse_csv": str(amplitude_outputs.get("csv_path") or ""),
        "spike_amplitude_waveforms_csv": str(amplitude_outputs.get("waveform_csv_path") or ""),
        "spike_amplitude_plot_paths": list(amplitude_outputs.get("plot_paths", [])),
        "spike_amplitude_qc_warnings": list(amplitude_outputs.get("warnings", [])),
    }
    with open(os.path.join(out_dir, "spike_shape_provenance.json"), "w", encoding="utf-8") as handle:
        json.dump({"shape_settings": shape_config, "audit_dirs": shape_audit_dirs,
                   "method": "Bombcell-inspired per-event adaptation; not validated for locust spikes"}, handle, indent=2)
    return outputs_info if return_outputs else out_csv


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Count spikes across multiple CSV recordings and make continuous per-channel plots."
    )
    shape_qc.add_arguments(parser)
    parser.add_argument(
        "--polarity",
        default=POLARITY,
        help="Spike polarity to count: both/all, neg/negative, or pos/positive.",
    )
    parser.add_argument(
        "--classification-window-ms",
        type=float,
        default=SPIKE_CLASSIFICATION_WINDOW_MS,
        help="Half-window in ms around each threshold candidate for polarity classification and peak/trough alignment.",
    )
    parser.add_argument(
        "--plot-spike-amplitude-change",
        action="store_true",
        help="Write polarity-specific spike-amplitude change-from-baseline CSV and PNG figures.",
    )
    parser.add_argument(
        "--overlay-spike-amplitudes",
        action="store_true",
        help="Overlay positive and negative normalized amplitude trajectories in one figure per channel.",
    )
    parser.add_argument(
        "--spike-amplitude-boundary-label",
        default=DEFAULT_UNRECORDED_TREATMENT_LABEL,
        help="Label drawn at the Baseline/Post boundary when treatment was not recorded.",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    selected_polarity = normalize_polarity(args.polarity)

    csv_paths = sorted(glob.glob(os.path.join(CSV_DIR, CSV_GLOB)))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {CSV_DIR} matching {CSV_GLOB}")

    ordered_csv_paths = choose_recording_order(csv_paths)

    out_dir = os.path.join(CSV_DIR, OUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    process_csvs(
        ordered_csv_paths,
        out_dir=out_dir,
        polarity=selected_polarity,
        classification_window_ms=args.classification_window_ms,
        shape_settings=shape_qc.from_arguments(args),
        plot_spike_amplitude_change=args.plot_spike_amplitude_change,
        overlay_spike_amplitudes=args.overlay_spike_amplitudes,
        spike_amplitude_boundary_label=args.spike_amplitude_boundary_label,
        window_sec=WINDOW_SEC,
    )


if __name__ == "__main__":
    main()
