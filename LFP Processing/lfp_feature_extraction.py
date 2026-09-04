#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Windowed LFP feature extraction for SPIE.

This module is import-safe.  The public entry point is
``process_lfp_recordings(recordings, config)``; the CLI wrapper at the bottom is
only for batch testing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import signal, stats


EPS = np.finfo(float).eps
DEFAULT_LFP_DOWNSAMPLE_HZ = 2000.0
DEFAULT_LFP_EXPLORATORY_BAND_HZ = (0.5, 300.0)
DEFAULT_LFP_ANTI_ALIAS_PASS_HZ = 300.0
DEFAULT_LFP_ANTI_ALIAS_STOP_HZ = 850.0
DEFAULT_LFP_ANTI_ALIAS_RIPPLE_DB = 0.1
DEFAULT_LFP_ANTI_ALIAS_ATTENUATION_DB = 80.0

MANIFEST_COLUMNS = [
    "experiment_id",
    "animal_id",
    "preparation_id",
    "segment_id",
    "recording_id",
    "recording_order",
    "source_file",
    "epoch_label",
    "phase",
    "treatment",
    "concentration_value",
    "concentration_unit",
    "application_onset_s",
    "washout_onset_s",
    "channel",
    "reference_scheme",
    "electrode_type",
    "impedance_1khz_ohm",
    "sampling_rate_hz",
    "signal_unit",
    "reference_stability",
    "vehicle_perfusion_control",
    "no_tissue_h2o2_control",
]

BIOLOGICAL_N_COLUMNS = ["experiment_id", "animal_id", "preparation_id"]

WINDOW_BASE_COLUMNS = [
    "experiment_id",
    "animal_id",
    "preparation_id",
    "segment_id",
    "recording_id",
    "recording_order",
    "source_file",
    "epoch_label",
    "phase",
    "treatment",
    "concentration_value",
    "concentration_unit",
    "application_onset_s",
    "washout_onset_s",
    "channel",
    "reference_scheme",
    "electrode_type",
    "impedance_1khz_ohm",
    "reference_stability",
    "vehicle_perfusion_control",
    "no_tissue_h2o2_control",
    "window_index",
    "window_start_s",
    "window_end_s",
    "window_mid_s",
    "window_duration_s",
    "cumulative_window_start_s",
    "cumulative_window_end_s",
    "cumulative_window_mid_s",
    "fs_est_hz",
    "fs_used_hz",
    "signal_unit",
]

EVENT_COLUMNS = [
    *WINDOW_BASE_COLUMNS,
    "event_index",
    "event_start_s",
    "event_end_s",
    "event_duration_s",
    "event_peak_amplitude_uv",
    "event_peak_frequency_hz",
    "event_threshold_uv",
]

DERIVATIVE_EVENT_COLUMNS = [
    "experiment_id",
    "animal_id",
    "preparation_id",
    "segment_id",
    "recording_id",
    "recording_order",
    "source_file",
    "epoch_label",
    "phase",
    "channel",
    "window_index",
    "event_index",
    "event_start_s",
    "event_peak_s",
    "event_end_s",
    "event_duration_s",
    "max_abs_voltage_derivative_uv_per_s",
    "max_robust_derivative_z_score",
    "large_derivative_z_threshold",
    "large_derivative_fraction_threshold",
    "local_raw_voltage_amplitude_uv",
    "nearest_spike_time_s",
    "nearest_spike_abs_latency_s",
    "spike_match_tolerance_s",
    "matched_to_spike_within_tolerance",
    "coincident_derivative_channel_count",
    "coincident_derivative_channel_fraction",
    "coincidence_tolerance_s",
    "provisional_classification",
    "classification_note",
]

DEFAULT_MARKER_BANDS = [
    ("very_low_1_4", "Descriptive 1-4 Hz", (1.0, 4.0)),
    ("low_4_8", "Descriptive 4-8 Hz", (4.0, 8.0)),
    ("mid_8_13", "Descriptive 8-13 Hz", (8.0, 13.0)),
    ("high_13_30", "Descriptive 13-30 Hz", (13.0, 30.0)),
    ("high_40_100", "Descriptive 40-100 Hz", (40.0, 100.0)),
    ("high_100_150", "Descriptive 100-150 Hz", (100.0, 150.0)),
    ("high_150_300", "Descriptive 150-300 Hz", (150.0, 300.0)),
    ("total_0p5_300", "Descriptive total 0.5-300 Hz", (0.5, 300.0)),
    ("primary_1_40", "Primary 1-40 Hz", (1.0, 40.0)),
]

MASK_COLUMNS = [
    ("missing", "finite", True),
    ("filter_edge", "filter_edge", False),
    ("manual_artifact", "manual_artifact", False),
    ("clipped", "clipped", False),
    ("flatline", "flatline", False),
    ("large_derivative", "large_derivative", False),
    ("time_discontinuity", "time_discontinuity", False),
    ("reference_excursion", "reference_excursion", False),
]

H2O2_WARNING = (
    "H2O2 interpretation warning: an LFP amplitude/power effect that also "
    "appears in saline without tissue, no-tissue H2O2 control, vehicle/"
    "perfusion control, or that tracks impedance/reference/noise change cannot "
    "be assigned to neural physiology."
)


@dataclass
class LFPConfig:
    output_dir: str | Path = "lfp_analysis"
    time_column_candidates: tuple[str, ...] = ("time_s", "time", "t", "seconds")
    channels: list[str] | None = None
    reference_scheme: str = "none"
    signal_unit: str = "uV"
    scale_to_uv: float | None = None
    analysis_window_s: float = 60.0
    welch_segment_s: float = 4.0
    welch_overlap: float = 0.5
    welch_min_cycles: float = 3.0
    exploratory_band_hz: tuple[float, float] = DEFAULT_LFP_EXPLORATORY_BAND_HZ
    primary_band_hz: tuple[float, float] = (1.0, 40.0)
    marker_bands: list[tuple[str, str, tuple[float, float]]] = field(default_factory=lambda: list(DEFAULT_MARKER_BANDS))
    filter_order: int = 4
    notch_frequency_hz: float | None = 50.0
    notch_harmonics: int = 3
    notch_q: float = 30.0
    downsample_hz: float | None = DEFAULT_LFP_DOWNSAMPLE_HZ
    minimum_valid_fraction: float = 0.90
    clipped_abs_uv: float | None = None
    clipped_fraction_threshold: float = 0.02
    flatline_epsilon_uv: float = 1e-9
    flatline_min_duration_s: float = 0.05
    flatline_fraction_threshold: float = 0.05
    large_derivative_z_threshold: float = 8.0
    derivative_warning_fraction_threshold: float = 0.005
    large_derivative_fraction_threshold: float = 0.02
    large_derivative_count_threshold: int = 5
    derivative_spike_match_tolerance_s: float = 0.002
    derivative_coincidence_tolerance_s: float = 0.002
    derivative_sensitivity_thresholds: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05)
    spike_events_csv: str | Path | None = None
    line_noise_ratio_threshold: float = 0.25
    reference_score_threshold: float = 0.75
    max_robust_amplitude_uv: float | None = None
    filter_edge_s: float | None = None
    manual_artifact_intervals: list[tuple[float, float]] = field(default_factory=list)
    manual_artifact_csv: str | Path | None = None
    peak_prominence_db: float = 3.0
    event_envelope_mad_threshold: float = 3.0
    min_event_duration_s: float = 0.20
    streaming: bool = False
    write_psd_long: bool = True
    write_simple_plots: bool = True
    simple_plot_bands: list[tuple[str, str, tuple[float, float]]] | None = None


@dataclass(frozen=True)
class LFPStream:
    t_rel_s: np.ndarray
    x_uv: np.ndarray
    valid_mask: np.ndarray
    fs_hz: float
    metadata: dict


def _sha256_array(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def design_lfp_resample_filter(
    fs_in_hz: float,
    fs_out_hz: float = DEFAULT_LFP_DOWNSAMPLE_HZ,
    passband_edge_hz: float = DEFAULT_LFP_ANTI_ALIAS_PASS_HZ,
    stopband_edge_hz: float | None = DEFAULT_LFP_ANTI_ALIAS_STOP_HZ,
    ripple_db: float = DEFAULT_LFP_ANTI_ALIAS_RIPPLE_DB,
    attenuation_db: float = DEFAULT_LFP_ANTI_ALIAS_ATTENUATION_DB,
) -> tuple[np.ndarray, dict]:
    """Design and verify the explicit FIR anti-alias filter for LFP resampling."""
    fs_in = float(fs_in_hz)
    fs_out = float(fs_out_hz)
    if not np.isfinite(fs_in) or fs_in <= 0:
        raise ValueError("Input sampling rate must be finite and positive.")
    if not np.isfinite(fs_out) or fs_out <= 0:
        raise ValueError("Output sampling rate must be finite and positive.")
    nyq_out = 0.5 * fs_out
    pass_edge = float(passband_edge_hz)
    if pass_edge <= 0:
        raise ValueError("Anti-alias passband edge must be positive.")
    if pass_edge >= 0.80 * nyq_out:
        raise ValueError(
            f"Downsample target {fs_out:g} Hz is too low for a {pass_edge:g} Hz LFP passband "
            f"(new Nyquist is {nyq_out:g} Hz)."
        )
    if stopband_edge_hz is None:
        stop_edge = 0.85 * nyq_out
    else:
        stop_edge = min(float(stopband_edge_hz), 0.85 * nyq_out)
    if stop_edge <= pass_edge:
        stop_edge = pass_edge + max(25.0, 0.25 * (nyq_out - pass_edge))
    if stop_edge >= nyq_out:
        stop_edge = 0.95 * nyq_out
    transition_hz = float(stop_edge - pass_edge)
    if transition_hz <= 0:
        raise ValueError("Anti-alias stopband must be above the passband edge.")
    width = transition_hz / (0.5 * fs_in)
    numtaps, beta = signal.kaiserord(float(attenuation_db), width)
    numtaps = max(17, int(numtaps))
    if numtaps % 2 == 0:
        numtaps += 1
    cutoff = 0.5 * (pass_edge + stop_edge)
    taps = signal.firwin(numtaps, cutoff, window=("kaiser", beta), fs=fs_in, pass_zero="lowpass")
    freqs, response = signal.freqz(taps, worN=262144, fs=fs_in)
    mag = np.abs(response)
    db = 20.0 * np.log10(np.maximum(mag, EPS))
    pass_mask = freqs <= pass_edge
    stop_mask = freqs >= stop_edge
    measured_ripple = float(np.nanmax(db[pass_mask]) - np.nanmin(db[pass_mask])) if np.any(pass_mask) else float("nan")
    measured_attenuation = float(-np.nanmax(db[stop_mask])) if np.any(stop_mask) else float("nan")
    if np.isfinite(measured_ripple) and measured_ripple > float(ripple_db):
        raise RuntimeError(
            f"Anti-alias design failed ripple target: {measured_ripple:.4g} dB > {float(ripple_db):.4g} dB."
        )
    if np.isfinite(measured_attenuation) and measured_attenuation < min(60.0, float(attenuation_db)):
        raise RuntimeError(
            f"Anti-alias design failed attenuation target: {measured_attenuation:.4g} dB < 60 dB."
        )
    group_delay_input_samples = 0.5 * (numtaps - 1)
    metadata = {
        "method": "scipy.signal.resample_poly with explicit Kaiser-windowed linear-phase FIR anti-alias filter",
        "filter_family": "FIR Kaiser low-pass",
        "source_fs_hz": fs_in,
        "target_fs_hz": fs_out,
        "passband_edge_hz": pass_edge,
        "stopband_edge_hz": float(stop_edge),
        "transition_width_hz": transition_hz,
        "requested_passband_ripple_db": float(ripple_db),
        "requested_stopband_attenuation_db": float(attenuation_db),
        "measured_passband_ripple_db": measured_ripple,
        "measured_stopband_attenuation_db": measured_attenuation,
        "numtaps": int(numtaps),
        "kaiser_beta": float(beta),
        "cutoff_hz": float(cutoff),
        "coeff_sha256": _sha256_array(taps),
        "group_delay_input_samples": float(group_delay_input_samples),
        "group_delay_s": float(group_delay_input_samples / fs_in),
        "delay_compensation": "linear-phase delay compensated by resample_poly output centering",
        "padding": "padtype=line per finite continuous run",
    }
    return taps.astype(float), metadata


def _resample_ratio(fs_in_hz: float, fs_out_hz: float) -> tuple[int, int, float]:
    fs_in = float(fs_in_hz)
    fs_out = float(fs_out_hz)
    rounded_in = int(round(fs_in))
    rounded_out = int(round(fs_out))
    if rounded_in > 0 and rounded_out > 0 and abs(fs_in - rounded_in) / fs_in < 0.01:
        divisor = math.gcd(rounded_in, rounded_out)
        up = max(1, rounded_out // divisor)
        down = max(1, rounded_in // divisor)
        return up, down, float(fs_in * up / down)
    ratio = Fraction(fs_out / fs_in).limit_denominator(100_000)
    return int(ratio.numerator), int(ratio.denominator), float(fs_in * ratio.numerator / ratio.denominator)


def create_lfp_stream(
    t_rel_s: np.ndarray,
    x_uv: np.ndarray,
    valid_mask: np.ndarray | None,
    fs_hz: float,
    target_fs_hz: float | None = DEFAULT_LFP_DOWNSAMPLE_HZ,
    warnings_out: list[str] | None = None,
    passband_edge_hz: float = DEFAULT_LFP_ANTI_ALIAS_PASS_HZ,
    stopband_edge_hz: float | None = DEFAULT_LFP_ANTI_ALIAS_STOP_HZ,
) -> LFPStream:
    """Create the synchronized LFP stream without filtering across gaps or NaNs."""
    t = np.asarray(t_rel_s, dtype=float)
    x = np.asarray(x_uv, dtype=float)
    if t.size != x.size:
        raise ValueError("LFP time and signal arrays must have the same length.")
    if valid_mask is None:
        valid = np.isfinite(t) & np.isfinite(x)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.size != t.size:
            raise ValueError("LFP valid mask must match the signal length.")
        valid = valid & np.isfinite(t)
    fs_in = float(fs_hz)
    target = None if target_fs_hz in (None, "", "none", "None", 0) else float(target_fs_hz)
    finite_continuous = np.isfinite(t) & np.isfinite(x) & ~_time_discontinuity_mask(t)
    runs = _contiguous_true_runs(finite_continuous)
    base_meta = {
        "stream_role": "LFP",
        "source_fs_hz": fs_in,
        "target_fs_hz": target,
        "finite_continuous_runs": int(len(runs)),
        "input_samples": int(t.size),
        "anti_alias_passband_hz": float(passband_edge_hz),
        "anti_alias_stopband_hz": None if stopband_edge_hz is None else float(stopband_edge_hz),
        "never_filter_or_resample_across": ["NaN/non-finite samples", "timestamp gaps", "timestamp resets", "separate recording CSVs"],
    }
    if target is None or not np.isfinite(target) or fs_in <= target * 1.05:
        meta = {
            **base_meta,
            "resampling_applied": False,
            "output_fs_hz": fs_in,
            "resample_up": 1,
            "resample_down": 1,
            "actual_output_fs_hz": fs_in,
            "output_samples": int(t.size),
            "filter_edge_exclusion_s": 0.0,
        }
        return LFPStream(t.copy(), x.copy(), valid.copy(), fs_in, meta)

    taps, filter_meta = design_lfp_resample_filter(
        fs_in,
        target,
        passband_edge_hz=passband_edge_hz,
        stopband_edge_hz=stopband_edge_hz,
    )
    up, down, actual_fs = _resample_ratio(fs_in, target)
    if warnings_out is not None and abs(actual_fs - target) / target > 0.001:
        warnings_out.append(
            f"LFP resample ratio gives {actual_fs:g} Hz instead of requested {target:g} Hz for source fs {fs_in:g} Hz."
        )
    out_t: list[np.ndarray] = []
    out_x: list[np.ndarray] = []
    out_valid: list[np.ndarray] = []
    for start, end in runs:
        run_t = t[start:end]
        run_x = x[start:end]
        run_valid = valid[start:end] & np.isfinite(run_x)
        if run_t.size < 2:
            continue
        y = signal.resample_poly(run_x, up, down, window=taps, padtype="line")
        if y.size < 2:
            continue
        td = float(run_t[0]) + np.arange(y.size, dtype=float) / float(target)
        keep = td <= float(run_t[-1]) + (0.5 / float(target))
        td = td[keep]
        y = y[keep]
        if td.size == 0:
            continue
        valid_ds = np.interp(td, run_t, run_valid.astype(float), left=0.0, right=0.0) > 0.999
        out_t.append(td)
        out_x.append(y)
        out_valid.append(valid_ds)
    if out_t:
        t_ds = np.concatenate(out_t)
        x_ds = np.concatenate(out_x)
        valid_ds = np.concatenate(out_valid)
    else:
        t_ds = np.array([], dtype=float)
        x_ds = np.array([], dtype=float)
        valid_ds = np.array([], dtype=bool)
    meta = {
        **base_meta,
        **filter_meta,
        "resampling_applied": True,
        "resample_up": int(up),
        "resample_down": int(down),
        "resample_ratio": f"{up}/{down}",
        "actual_output_fs_hz": float(actual_fs),
        "output_fs_hz": float(target),
        "output_samples": int(t_ds.size),
        "filter_edge_exclusion_s": float(filter_meta["group_delay_s"]),
    }
    return LFPStream(t_ds, x_ds, valid_ds, float(target), meta)


def parse_band_spec(value: object) -> list[tuple[str, str, tuple[float, float]]]:
    """Parse GUI/CLI band definitions.

    Accepted examples:
        "slow=1-4; fast=20-40"
        [{"key": "slow", "label": "1-4 Hz", "lo": 1, "hi": 4}]
        [("slow", "Slow 1-4", (1, 4))]
    """
    if value is None or value == "":
        return list(DEFAULT_MARKER_BANDS)
    if isinstance(value, str):
        bands = []
        for item in value.split(";"):
            item = item.strip()
            if not item:
                continue
            item = item.replace("\u2013", "-").replace("\u2014", "-")
            if "=" in item:
                key, rng = item.split("=", 1)
                label = key.strip()
            else:
                key = ""
                rng = item
                label = ""
            parts = re.split(r"\s*[-:,]\s*", rng.strip())
            nums = [float(part) for part in parts if part]
            if len(nums) != 2:
                raise ValueError(f"Could not parse frequency band {item!r}. Use name=lo-hi.")
            lo, hi = nums
            if not key:
                key = f"band_{lo:g}_{hi:g}"
            safe_key = sanitize_name(key)
            bands.append((safe_key, label or f"{lo:g}-{hi:g} Hz", (float(lo), float(hi))))
        return bands or list(DEFAULT_MARKER_BANDS)
    bands = []
    for item in value:  # type: ignore[union-attr]
        if isinstance(item, Mapping):
            key = sanitize_name(item.get("key") or item.get("name") or item.get("label") or "band")
            label = str(item.get("label") or key)
            lo = float(item.get("lo", item.get("low", item.get("fmin"))))
            hi = float(item.get("hi", item.get("high", item.get("fmax"))))
        else:
            key, label, band = item
            lo, hi = band
            key = sanitize_name(key)
            label = str(label)
        bands.append((str(key), label, (float(lo), float(hi))))
    return bands or list(DEFAULT_MARKER_BANDS)


def as_lfp_config(config: LFPConfig | Mapping | None = None) -> LFPConfig:
    if isinstance(config, LFPConfig):
        return config
    cfg = LFPConfig()
    if not config:
        return cfg
    raw = dict(config)
    if "marker_bands" in raw:
        raw["marker_bands"] = parse_band_spec(raw["marker_bands"])
    if "simple_plot_bands" in raw:
        raw["simple_plot_bands"] = parse_band_spec(raw["simple_plot_bands"]) if raw["simple_plot_bands"] not in (None, "") else None
    if "channels" in raw:
        raw["channels"] = parse_channel_list(raw["channels"])
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.exploratory_band_hz = tuple(float(x) for x in cfg.exploratory_band_hz)  # type: ignore[assignment]
    cfg.primary_band_hz = tuple(float(x) for x in cfg.primary_band_hz)  # type: ignore[assignment]
    cfg.analysis_window_s = float(cfg.analysis_window_s)
    cfg.welch_segment_s = float(cfg.welch_segment_s)
    cfg.welch_overlap = float(cfg.welch_overlap)
    cfg.minimum_valid_fraction = float(cfg.minimum_valid_fraction)
    cfg.flatline_min_duration_s = float(cfg.flatline_min_duration_s)
    cfg.derivative_warning_fraction_threshold = float(cfg.derivative_warning_fraction_threshold)
    cfg.large_derivative_fraction_threshold = float(cfg.large_derivative_fraction_threshold)
    cfg.large_derivative_count_threshold = int(float(cfg.large_derivative_count_threshold))
    cfg.derivative_spike_match_tolerance_s = float(cfg.derivative_spike_match_tolerance_s)
    cfg.derivative_coincidence_tolerance_s = float(cfg.derivative_coincidence_tolerance_s)
    cfg.derivative_sensitivity_thresholds = tuple(float(x) for x in cfg.derivative_sensitivity_thresholds)
    if cfg.downsample_hz in ("", "none", "None", 0):
        cfg.downsample_hz = None
    if cfg.downsample_hz is not None:
        cfg.downsample_hz = float(cfg.downsample_hz)
    if cfg.notch_frequency_hz in ("", "none", "None", 0):
        cfg.notch_frequency_hz = None
    if cfg.notch_frequency_hz is not None:
        cfg.notch_frequency_hz = float(cfg.notch_frequency_hz)
    cfg.write_psd_long = _bool_from_any(cfg.write_psd_long)
    cfg.write_simple_plots = _bool_from_any(cfg.write_simple_plots)
    return cfg


def _bool_from_any(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off", "none"}
    return bool(value)


def sanitize_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_") or "value"


def parse_channel_list(value: object) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return parts or None
    if isinstance(value, Iterable):
        out = []
        for item in value:
            if item is None:
                continue
            out.extend(part.strip() for part in str(item).split(",") if part.strip())
        return out or None
    return [str(value).strip()]


def normalize_epoch_phase(label: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "", str(label or "").strip().lower())
    if not text:
        return ""
    if any(term in text for term in ("baseline", "basal", "before", "pre")):
        return "baseline"
    if any(term in text for term in ("post", "after", "recovery", "washout")):
        return "post"
    if any(term in text for term in ("during", "stim", "stimulation", "perfusion", "treatment")):
        return "treatment"
    return str(label).strip().lower()


def _is_marker_column(column_name: str) -> bool:
    key = str(column_name).strip().lower().replace(" ", "_").replace("-", "_")
    return any(token in key for token in ("marker", "ttl", "trigger", "event", "digital", "dig", "din", "dout", "stim", "sync"))


def find_time_column(columns: Sequence[str], candidates: Sequence[str]) -> str:
    lower = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return str(candidate)
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    raise ValueError(f"Could not find a time column. Tried: {', '.join(candidates)}")


def _list_signal_columns(header: pd.DataFrame, time_col: str, requested: list[str] | None = None) -> list[str]:
    columns = [str(col) for col in header.columns if str(col) != str(time_col) and not _is_marker_column(str(col))]
    if requested:
        wanted = set(requested)
        columns = [col for col in columns if col in wanted]
    return columns


def _normalise_recording(recording: str | Path | Mapping, order: int, cfg: LFPConfig) -> dict:
    if isinstance(recording, (str, Path)):
        rec = {"source_file": str(recording)}
    else:
        rec = dict(recording)
        if "source_file" not in rec and "path" in rec:
            rec["source_file"] = rec["path"]
    source = str(rec.get("source_file", "")).strip()
    if not source:
        raise ValueError("Each LFP recording needs source_file/path metadata.")
    rec.setdefault("recording_order", order)
    rec.setdefault("segment_id", rec.get("recording_id") or Path(source).stem)
    rec.setdefault("recording_id", rec.get("segment_id") or Path(source).stem)
    rec.setdefault("epoch_label", rec.get("label") or rec.get("phase", ""))
    epoch_label = str(rec.get("epoch_label", "")).strip()
    if not epoch_label:
        raise ValueError(
            f"LFP recording segment {order} ({Path(source).name}) is unlabeled; "
            "assign an explicit epoch_label such as Baseline, Treatment, Post 1, or Washout."
        )
    rec["epoch_label"] = epoch_label
    rec["phase"] = normalize_epoch_phase(epoch_label)
    rec.setdefault("treatment", "")
    rec.setdefault("concentration_value", "")
    rec.setdefault("concentration_unit", "")
    rec.setdefault("application_onset_s", "")
    rec.setdefault("washout_onset_s", "")
    rec.setdefault("channel", "")
    rec.setdefault("reference_scheme", cfg.reference_scheme)
    rec.setdefault("electrode_type", "")
    rec.setdefault("impedance_1khz_ohm", "")
    rec.setdefault("sampling_rate_hz", "")
    rec.setdefault("signal_unit", cfg.signal_unit)
    rec.setdefault("reference_stability", "")
    rec.setdefault("vehicle_perfusion_control", "")
    rec.setdefault("no_tissue_h2o2_control", "")
    for col in MANIFEST_COLUMNS:
        rec.setdefault(col, "")
    rec["source_file"] = source
    rec["recording_order"] = int(float(rec.get("recording_order") or order))
    rec["segment_id"] = str(rec.get("segment_id") or rec.get("recording_id") or Path(source).stem)
    rec["recording_id"] = str(rec.get("recording_id") or rec["segment_id"])
    return rec


def _unit_scale_to_uv(unit: object, explicit_scale: float | None, column_name: str = "") -> tuple[float, str]:
    if explicit_scale is not None:
        return float(explicit_scale), str(unit or "explicit_scale")
    raw = str(unit or "").strip()
    if not raw:
        lower = column_name.lower()
        if lower.endswith("_uv") or "uv" in lower:
            raw = "uV"
        elif lower.endswith("_mv") or "mv" in lower:
            raw = "mV"
        elif lower.endswith("_v") or lower == "v":
            raw = "V"
        else:
            raw = "uV"
    key = raw.replace("micro", "u").replace("µ", "u").lower()
    if key in {"uv", "uvolt", "uvolts", "microvolt", "microvolts"}:
        return 1.0, "uV"
    if key in {"mv", "millivolt", "millivolts"}:
        return 1000.0, "mV"
    if key in {"v", "volt", "volts"}:
        return 1_000_000.0, "V"
    raise ValueError(f"Unsupported LFP signal unit {unit!r}; use uV, mV, V, or an explicit scale_to_uv.")


def _to_float_or_nan(value: object) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "yes", "true", "y", "checked"}


def validate_timestamps(t: np.ndarray, declared_fs: object = "") -> dict:
    t = np.asarray(t, dtype=float)
    finite = np.isfinite(t)
    if finite.sum() < 3:
        raise ValueError("Time column has fewer than three finite samples.")
    diffs = np.diff(t)
    finite_diffs = diffs[np.isfinite(diffs)]
    if finite_diffs.size == 0:
        raise ValueError("Time column has no finite sample intervals.")
    positive_diffs = finite_diffs[finite_diffs > 0]
    if positive_diffs.size == 0:
        raise ValueError("Time column is not increasing.")
    median_dt = float(np.median(positive_diffs))
    fs = 1.0 / median_dt
    resets = int(np.sum(finite_diffs <= 0))
    gaps = int(np.sum(finite_diffs > 1.5 * median_dt))
    jitter_fraction = float(np.mean(np.abs(positive_diffs - median_dt) > 0.10 * median_dt))
    dropped = int(np.sum(np.maximum(0, np.round(positive_diffs / median_dt).astype(int) - 1)))
    declared = _to_float_or_nan(declared_fs)
    fs_mismatch_fraction = float(abs(declared - fs) / fs) if np.isfinite(declared) and fs > 0 else float("nan")
    return {
        "fs_est_hz": float(fs),
        "median_dt_s": median_dt,
        "timestamp_resets": resets,
        "timestamp_gaps": gaps,
        "timestamp_jitter_fraction": jitter_fraction,
        "dropped_samples_est": dropped,
        "declared_sampling_rate_hz": declared,
        "sampling_rate_mismatch_fraction": fs_mismatch_fraction,
    }


def _interp_fill(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float).copy()
    if np.isfinite(x).all():
        return x
    good = np.isfinite(x)
    if good.sum() == 0:
        return np.zeros_like(x, dtype=float)
    if good.sum() == 1:
        x[~good] = x[good][0]
        return x
    idx = np.arange(x.size, dtype=float)
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


def robust_mad(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def robust_amplitude(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanpercentile(arr, 95.0) - np.nanpercentile(arr, 5.0))


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return [(int(start), int(end)) for start, end in zip(starts, ends) if end > start]


def _time_discontinuity_mask(t_rel: np.ndarray) -> np.ndarray:
    discontinuity = np.zeros_like(t_rel, dtype=bool)
    if t_rel.size <= 1:
        return discontinuity
    dt = np.diff(t_rel)
    positive = dt[np.isfinite(dt) & (dt > 0)]
    if positive.size == 0:
        discontinuity[1:] = True
        return discontinuity
    med_dt = float(np.median(positive))
    bad = (~np.isfinite(dt)) | (dt <= 0) | (dt > 1.5 * med_dt)
    discontinuity[1:] |= bad
    return discontinuity


def _effective_filter_edge_s(t_rel: np.ndarray, cfg: LFPConfig) -> float:
    if cfg.filter_edge_s is not None:
        return max(0.0, float(cfg.filter_edge_s))
    edge_s = max(1.0, 3.0 / max(cfg.exploratory_band_hz[0], 0.1))
    if t_rel.size > 1:
        duration = float(np.nanmax(t_rel) - np.nanmin(t_rel))
        if np.isfinite(duration) and duration > 0 and duration <= 2.0 * edge_s:
            edge_s = min(edge_s, 0.10 * duration)
    return float(edge_s)


def _window_mask_budget(masks: Mapping[str, np.ndarray], sample_interval_s: float, prefix: str = "") -> dict[str, float]:
    valid = np.asarray(masks.get("valid", []), dtype=bool)
    n = max(1, valid.size)
    dt = float(sample_interval_s) if np.isfinite(sample_interval_s) and sample_interval_s > 0 else 0.0
    out: dict[str, float] = {}
    hard_union = np.zeros(n, dtype=bool)
    hard_sum = 0
    hard_mask_keys = {
        "finite",
        "filter_edge",
        "manual_artifact",
        "clipped",
        "flatline",
        "time_discontinuity",
        "reference_excursion",
    }
    for label, key, invert in MASK_COLUMNS:
        if key not in masks:
            mask = np.zeros(n, dtype=bool)
        else:
            raw = np.asarray(masks[key], dtype=bool)
            mask = ~raw if invert else raw
        count = int(np.count_nonzero(mask))
        out[f"{prefix}{label}_masked_s"] = float(count * dt)
        out[f"{prefix}{label}_masked_fraction"] = float(count / n)
        if key == "large_derivative":
            out[f"{prefix}large_derivative_candidate_s"] = float(count * dt)
            out[f"{prefix}large_derivative_candidate_fraction"] = float(count / n)
        if key in hard_mask_keys:
            hard_union |= mask
            hard_sum += count
    downsample_loss = np.asarray(masks.get("downsample_validity_loss", np.zeros(n, dtype=bool)), dtype=bool)
    out[f"{prefix}downsample_validity_loss_s"] = float(np.count_nonzero(downsample_loss) * dt)
    out[f"{prefix}downsample_validity_loss_fraction"] = float(np.count_nonzero(downsample_loss) / n)
    combined = hard_union
    valid_count = int(np.count_nonzero(~combined))
    out[f"{prefix}combined_masked_s"] = float(np.count_nonzero(combined) * dt)
    out[f"{prefix}combined_masked_fraction"] = float(np.count_nonzero(combined) / n)
    out[f"{prefix}hard_mask_sum_s"] = float(hard_sum * dt)
    out[f"{prefix}hard_mask_sum_fraction"] = float(hard_sum / n)
    out[f"{prefix}usable_duration_s"] = float(valid_count * dt)
    return out


def _band_sos(fs: float, band: tuple[float, float], order: int) -> np.ndarray:
    nyq = 0.5 * fs
    lo = max(float(band[0]) / nyq, 1e-6)
    hi = min(float(band[1]) / nyq, 0.999)
    if not 0 < lo < hi < 1:
        raise ValueError(f"Invalid band {band} for fs={fs:g} Hz.")
    return signal.butter(int(order), [lo, hi], btype="bandpass", output="sos")


def _lowpass_sos(fs: float, cutoff_hz: float, order: int = 6) -> np.ndarray:
    nyq = 0.5 * fs
    cutoff = min(max(float(cutoff_hz) / nyq, 1e-6), 0.999)
    return signal.butter(int(order), cutoff, btype="lowpass", output="sos")


def _apply_zero_phase_sos(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    if x.size < 4:
        return x.copy()
    try:
        return signal.sosfiltfilt(sos, x)
    except ValueError:
        padlen = max(0, min(x.size - 1, 3 * (2 * len(sos) + 1)))
        return signal.sosfiltfilt(sos, x, padlen=padlen)


def _apply_filters(x_uv: np.ndarray, fs: float, cfg: LFPConfig, warnings_out: list[str]) -> np.ndarray:
    y = _interp_fill(x_uv)
    y = y - float(np.nanmedian(y))
    nyq = 0.5 * fs
    if cfg.notch_frequency_hz and cfg.notch_frequency_hz > 0:
        for harmonic in range(1, int(cfg.notch_harmonics) + 1):
            freq = float(cfg.notch_frequency_hz) * harmonic
            if freq >= nyq * 0.98:
                continue
            b, a = signal.iirnotch(freq / nyq, Q=float(cfg.notch_q))
            y = _apply_zero_phase_sos(y, signal.tf2sos(b, a))
    y = _apply_zero_phase_sos(y, _band_sos(fs, cfg.exploratory_band_hz, cfg.filter_order))
    return y


def _apply_filters_by_runs(x_uv: np.ndarray, fs: float, cfg: LFPConfig, warnings_out: list[str]) -> np.ndarray:
    x = np.asarray(x_uv, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    min_len = max(16, int(round(2.0 * fs / max(cfg.exploratory_band_hz[0], 0.1))))
    for start, end in _contiguous_true_runs(np.isfinite(x)):
        run = x[start:end]
        if run.size < min_len:
            warnings_out.append(
                f"Skipped filtering a {run.size / fs:.3g}s valid run because it is too short for "
                f"the {cfg.exploratory_band_hz[0]:g}-{cfg.exploratory_band_hz[1]:g} Hz LFP filter."
            )
            continue
        out[start:end] = _apply_filters(run, fs, cfg, warnings_out)
    return out


def _maybe_downsample(
    t: np.ndarray,
    x: np.ndarray,
    valid_mask: np.ndarray,
    fs: float,
    cfg: LFPConfig,
    warnings_out: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    target = cfg.downsample_hz
    if target is None or target <= 0 or fs <= target * 1.05:
        return t, x, valid_mask, fs
    if cfg.exploratory_band_hz[1] >= 0.45 * target:
        warnings_out.append(
            f"Exploratory LFP high-pass edge {cfg.exploratory_band_hz[1]:g} Hz is near "
            f"the downsampled Nyquist frequency for {target:g} Hz."
        )
    anti_alias = _lowpass_sos(fs, 0.45 * target)
    x_aa = _apply_zero_phase_sos(_interp_fill(x), anti_alias)
    fs_i = max(1, int(round(fs)))
    target_i = max(1, int(round(target)))
    divisor = math.gcd(fs_i, target_i)
    up = target_i // divisor
    down = fs_i // divisor
    x_ds = signal.resample_poly(x_aa, up, down)
    valid_ds = signal.resample_poly(valid_mask.astype(float), up, down) > 0.95
    t0 = float(t[0])
    t_ds = t0 + np.arange(x_ds.size, dtype=float) / float(target)
    n = min(len(t_ds), len(x_ds), len(valid_ds))
    return t_ds[:n], x_ds[:n], valid_ds[:n], float(target)


def _manual_intervals_for_recording(cfg: LFPConfig, rec: Mapping) -> list[tuple[float, float]]:
    intervals = [(float(a), float(b)) for a, b in cfg.manual_artifact_intervals if float(b) > float(a)]
    path = cfg.manual_artifact_csv
    if path:
        df = pd.read_csv(path)
        for _idx, row in df.iterrows():
            if "recording_id" in df.columns and str(row.get("recording_id", "")).strip():
                if str(row.get("recording_id", "")).strip() != str(rec.get("recording_id", "")).strip():
                    continue
            if "source_file" in df.columns and str(row.get("source_file", "")).strip():
                if Path(str(row.get("source_file"))).name != Path(str(rec.get("source_file"))).name:
                    continue
            start = _to_float_or_nan(row.get("start_s", row.get("start")))
            end = _to_float_or_nan(row.get("end_s", row.get("end")))
            if np.isfinite(start) and np.isfinite(end) and end > start:
                intervals.append((float(start), float(end)))
    return intervals


def _artifact_masks(
    t_rel: np.ndarray,
    x_uv: np.ndarray,
    cfg: LFPConfig,
    rec: Mapping,
    reference_uv: np.ndarray | None,
    fs: float,
) -> dict[str, np.ndarray]:
    finite = np.isfinite(x_uv)
    mask_manual = np.zeros_like(finite, dtype=bool)
    for start, end in _manual_intervals_for_recording(cfg, rec):
        mask_manual |= (t_rel >= start) & (t_rel < end)
    edge_s = _effective_filter_edge_s(t_rel, cfg)
    edge = (t_rel < float(edge_s)) | (t_rel > (float(t_rel[-1]) - float(edge_s)))
    clipped = np.zeros_like(finite, dtype=bool)
    if cfg.clipped_abs_uv is not None:
        clipped |= np.abs(x_uv) >= float(cfg.clipped_abs_uv)
    else:
        values = x_uv[finite]
        if values.size:
            lo = float(np.min(values))
            hi = float(np.max(values))
            tol = max(1e-12, 1e-9 * max(abs(lo), abs(hi), 1.0))
            for rail in (lo, hi):
                near = np.abs(x_uv - rail) <= tol
                edges = np.diff(near.astype(int), prepend=0, append=0)
                starts = np.where(edges == 1)[0]
                ends = np.where(edges == -1)[0]
                if starts.size and np.max(ends - starts) >= 3:
                    clipped |= near
    discontinuity = _time_discontinuity_mask(t_rel)
    flat = np.zeros_like(finite, dtype=bool)
    large_derivative = np.zeros_like(finite, dtype=bool)
    derivative_uv_per_s = np.full_like(x_uv, np.nan, dtype=float)
    derivative_z_score = np.full_like(x_uv, np.nan, dtype=float)
    continuous_data = finite & ~discontinuity
    for start, end in _contiguous_true_runs(continuous_data):
        run = np.asarray(x_uv[start:end], dtype=float)
        run_t = np.asarray(t_rel[start:end], dtype=float)
        if run.size < 3:
            continue
        diffs = np.diff(run)
        dt = np.diff(run_t)
        good_dt = np.isfinite(dt) & (dt > 0)
        local_derivative = np.full_like(diffs, np.nan, dtype=float)
        local_derivative[good_dt] = diffs[good_dt] / dt[good_dt]
        derivative_uv_per_s[start + 1 : end] = local_derivative
        same = np.abs(diffs) <= float(cfg.flatline_epsilon_uv)
        if np.isfinite(fs) and fs > 0:
            min_flat_samples = max(2, int(math.ceil(float(cfg.flatline_min_duration_s) * fs)))
        else:
            min_flat_samples = 2
        for flat_start, flat_end in _contiguous_true_runs(same):
            flat_sample_count = (flat_end - flat_start) + 1
            if flat_sample_count >= min_flat_samples:
                flat[start + flat_start : start + flat_end + 1] = True
        diff_mad = robust_mad(diffs)
        if np.isfinite(diff_mad) and diff_mad > 0:
            signed_z = (diffs - np.median(diffs)) / (1.4826 * diff_mad + EPS)
            derivative_z_score[start + 1 : end] = signed_z
            bad_step = np.abs(signed_z) >= float(cfg.large_derivative_z_threshold)
            bad_idx = np.where(bad_step)[0] + 1
            large_derivative[start + bad_idx] = True
    reference_bad = np.zeros_like(finite, dtype=bool)
    if reference_uv is not None:
        ref_amp = np.abs(reference_uv - np.nanmedian(reference_uv))
        ref_mad = robust_mad(ref_amp)
        if np.isfinite(ref_mad) and ref_mad > 0:
            reference_bad = ref_amp > (np.nanmedian(ref_amp) + 8.0 * 1.4826 * ref_mad)
    hard_invalid = (~finite) | mask_manual | edge | clipped | flat | discontinuity | reference_bad
    return {
        "finite": finite,
        "manual_artifact": mask_manual,
        "filter_edge": edge,
        "clipped": clipped,
        "flatline": flat,
        "large_derivative": large_derivative,
        "derivative_uv_per_s": derivative_uv_per_s,
        "derivative_z_score": derivative_z_score,
        "time_discontinuity": discontinuity,
        "reference_excursion": reference_bad,
        "valid": ~hard_invalid,
    }


def _welch_window_psd(x: np.ndarray, fs: float, cfg: LFPConfig, warnings_out: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    requested_s = max(float(cfg.welch_segment_s), 0.5)
    min_freq = max(0.001, min(cfg.exploratory_band_hz[0], cfg.primary_band_hz[0]))
    needed_s = float(cfg.welch_min_cycles) / min_freq
    segment_s = requested_s
    if requested_s < needed_s:
        segment_s = needed_s
        warnings_out.append(
            f"Welch segment length increased from {requested_s:g}s to {segment_s:g}s "
            f"to include {cfg.welch_min_cycles:g} cycles at {min_freq:g} Hz."
        )
    spectra: list[tuple[np.ndarray, np.ndarray, int]] = []
    for start, end in _contiguous_true_runs(np.isfinite(x)):
        run = x[start:end]
        if run.size < 8:
            continue
        nperseg = int(round(segment_s * fs))
        nperseg = max(8, min(nperseg, run.size))
        actual_cycles = (nperseg / fs) * min_freq
        if actual_cycles < cfg.welch_min_cycles:
            warnings_out.append(
                f"Welch segment contains only {actual_cycles:.2f} cycles at {min_freq:g} Hz; "
                "low-frequency estimates are exploratory."
            )
        noverlap = int(round(nperseg * float(cfg.welch_overlap)))
        noverlap = max(0, min(noverlap, nperseg - 1))
        try:
            f, pxx = signal.welch(
                run,
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="linear",
                scaling="density",
                average="median",
            )
        except TypeError:
            f, pxx = signal.welch(
                run,
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="linear",
                scaling="density",
            )
        keep = (f >= cfg.exploratory_band_hz[0]) & (f <= cfg.exploratory_band_hz[1])
        if np.any(keep):
            spectra.append((f[keep], pxx[keep], int(run.size)))
    if not spectra:
        return np.array([]), np.array([])
    base_f = spectra[0][0]
    weighted = np.zeros_like(base_f, dtype=float)
    total_weight = 0
    for f, pxx, weight in spectra:
        p_interp = pxx if np.array_equal(f, base_f) else np.interp(base_f, f, pxx, left=np.nan, right=np.nan)
        good = np.isfinite(p_interp)
        if not np.any(good):
            continue
        weighted[good] += p_interp[good] * weight
        total_weight += weight
    if total_weight <= 0:
        return np.array([]), np.array([])
    return base_f, weighted / total_weight


def integrate_power(f: np.ndarray, pxx: np.ndarray, band: tuple[float, float]) -> float:
    keep = (f >= float(band[0])) & (f <= float(band[1])) & np.isfinite(pxx)
    if np.sum(keep) < 2:
        return float("nan")
    return float(np.trapezoid(pxx[keep], f[keep]))


def _line_noise_ratio(f: np.ndarray, pxx: np.ndarray, cfg: LFPConfig) -> float:
    if not cfg.notch_frequency_hz:
        return 0.0
    total = integrate_power(f, pxx, cfg.exploratory_band_hz)
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    line = 0.0
    for harmonic in range(1, int(cfg.notch_harmonics) + 1):
        freq = float(cfg.notch_frequency_hz) * harmonic
        if freq >= f[-1]:
            continue
        val = integrate_power(f, pxx, (freq - 1.0, freq + 1.0))
        if np.isfinite(val):
            line += val
    return float(line / total) if total > 0 else float("nan")


def _aperiodic_fit(f: np.ndarray, pxx: np.ndarray, band: tuple[float, float]) -> dict:
    keep = (f >= max(0.25, band[0])) & (f <= band[1]) & np.isfinite(pxx) & (pxx > 0)
    if np.sum(keep) < 4:
        return {"aperiodic_exponent": np.nan, "aperiodic_offset": np.nan, "aperiodic_r2": np.nan, "aperiodic_qc": "insufficient_positive_psd"}
    xf = np.log10(f[keep])
    yf = np.log10(pxx[keep])
    slope, intercept = np.polyfit(xf, yf, 1)
    pred = slope * xf + intercept
    ss_res = float(np.sum((yf - pred) ** 2))
    ss_tot = float(np.sum((yf - np.mean(yf)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "aperiodic_exponent": float(-slope),
        "aperiodic_offset": float(intercept),
        "aperiodic_r2": r2,
        "aperiodic_qc": "pass" if np.isfinite(r2) else "fit_quality_not_computable",
    }


def _peak_features(f: np.ndarray, pxx: np.ndarray, cfg: LFPConfig, fit: Mapping[str, float]) -> dict:
    keep = (f >= cfg.primary_band_hz[0]) & (f <= cfg.primary_band_hz[1]) & np.isfinite(pxx) & (pxx > 0)
    out = {
        "dominant_peak_frequency_hz": np.nan,
        "dominant_peak_power_db": np.nan,
        "peak_power_over_aperiodic_db": np.nan,
        "peak_prominence_db": np.nan,
        "peak_bandwidth_hz": np.nan,
        "peak_qc": "no_peak_above_prominence",
    }
    if np.sum(keep) < 3:
        out["peak_qc"] = "too_few_psd_bins"
        return out
    f_band = f[keep]
    p_band = pxx[keep]
    y_db = 10.0 * np.log10(p_band + EPS)
    peaks, props = signal.find_peaks(y_db, prominence=float(cfg.peak_prominence_db))
    if peaks.size == 0:
        return out
    best_local = int(peaks[np.argmax(props["prominences"])])
    peak_freq = float(f_band[best_local])
    peak_power_db = float(y_db[best_local])
    exponent = fit.get("aperiodic_exponent", np.nan)
    offset = fit.get("aperiodic_offset", np.nan)
    if np.isfinite(exponent) and np.isfinite(offset) and peak_freq > 0:
        bg_log10 = float(offset) - float(exponent) * math.log10(peak_freq)
        over_bg = peak_power_db - 10.0 * bg_log10
    else:
        over_bg = float("nan")
    widths = signal.peak_widths(y_db, [best_local], rel_height=0.5)
    if f_band.size > 1:
        df = float(np.median(np.diff(f_band)))
        bandwidth = float(widths[0][0] * df)
    else:
        bandwidth = float("nan")
    out.update(
        {
            "dominant_peak_frequency_hz": peak_freq,
            "dominant_peak_power_db": peak_power_db,
            "peak_power_over_aperiodic_db": float(over_bg),
            "peak_prominence_db": float(props["prominences"][np.argmax(props["prominences"])]),
            "peak_bandwidth_hz": bandwidth,
            "peak_qc": "pass",
        }
    )
    return out


def _autocorrelation_timescale(x: np.ndarray, fs: float) -> tuple[float, float]:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        return float("nan"), float("nan")
    arr = arr - np.mean(arr)
    if np.allclose(arr, 0):
        return float("nan"), float("nan")
    ac = signal.correlate(arr, arr, mode="full", method="auto")[arr.size - 1 :]
    ac = ac / (ac[0] + EPS)
    below = np.where(ac <= 1.0 / math.e)[0]
    tau = float(below[0] / fs) if below.size else float("nan")
    peaks, _ = signal.find_peaks(ac[1:])
    period = float((peaks[0] + 1) / fs) if peaks.size else float("nan")
    return tau, period


def _event_features(
    x: np.ndarray,
    t: np.ndarray,
    fs: float,
    cfg: LFPConfig,
    base_row: Mapping,
) -> tuple[dict, list[dict]]:
    def skipped(reason: str) -> tuple[dict, list[dict]]:
        return {
            "oscillatory_event_rate_per_min": np.nan,
            "oscillatory_event_duration_s": np.nan,
            "oscillatory_event_occupancy": np.nan,
            "oscillatory_event_peak_amplitude_uv": np.nan,
            "oscillatory_event_peak_frequency_hz": np.nan,
            "event_qc": reason,
        }, []

    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    finite = np.isfinite(x) & np.isfinite(t)
    if np.count_nonzero(finite) < max(8, int(0.5 * fs)):
        return skipped("not_enough_finite_filtered_samples")
    if not np.isfinite(fs) or fs <= 0:
        return skipped("invalid_sampling_rate")
    nyq = 0.5 * fs
    if cfg.primary_band_hz[0] <= 0 or cfg.primary_band_hz[1] >= nyq:
        return skipped("event_band_not_representable")
    finite_x = x[finite]
    if finite_x.size == 0 or not np.isfinite(np.nanstd(finite_x)) or np.nanstd(finite_x) <= EPS:
        return skipped("zero_or_nonfinite_filtered_variance")
    event_rows = []
    durations = []
    peak_amps = []
    peak_freqs = []
    min_len = max(1, int(round(float(cfg.min_event_duration_s) * fs)))
    event_index = 0
    valid_samples = 0
    for run_start, run_end in _contiguous_true_runs(finite):
        run_x = x[run_start:run_end]
        run_t = t[run_start:run_end]
        valid_samples += int(run_x.size)
        if run_x.size < max(8, int(0.5 * fs)):
            continue
        try:
            band_x = _apply_zero_phase_sos(run_x, _band_sos(fs, cfg.primary_band_hz, max(2, min(cfg.filter_order, 4))))
            envelope = np.abs(signal.hilbert(band_x))
        except Exception:
            continue
        med = float(np.nanmedian(envelope))
        mad = robust_mad(envelope)
        thresh = med + float(cfg.event_envelope_mad_threshold) * 1.4826 * (mad if np.isfinite(mad) and mad > 0 else np.nanstd(envelope))
        if not np.isfinite(thresh) or thresh <= 0:
            return skipped("event_threshold_zero_or_not_finite")
        active = envelope >= thresh
        if not np.any(active):
            continue
        edges = np.diff(active.astype(int), prepend=0, append=0)
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        for start, end in zip(starts, ends):
            if end - start < min_len:
                continue
            seg = band_x[start:end]
            event_t = run_t[start:end]
            duration = float((end - start) / fs)
            peak_amp = float(np.max(envelope[start:end]))
            if seg.size >= 8:
                f_evt, p_evt = _welch_window_psd(seg, fs, cfg, [])
                peak_freq = float(f_evt[np.argmax(p_evt)]) if f_evt.size and p_evt.size else float("nan")
            else:
                peak_freq = float("nan")
            event_index += 1
            durations.append(duration)
            peak_amps.append(peak_amp)
            peak_freqs.append(peak_freq)
            event_rows.append(
                {
                    **{k: base_row.get(k, "") for k in base_row.keys()},
                    "event_index": event_index,
                    "event_start_s": float(event_t[0]) if event_t.size else np.nan,
                    "event_end_s": float(event_t[-1]) if event_t.size else np.nan,
                    "event_duration_s": duration,
                    "event_peak_amplitude_uv": peak_amp,
                    "event_peak_frequency_hz": peak_freq,
                    "event_threshold_uv": float(thresh),
                }
            )
    if not durations:
        return {
            "oscillatory_event_rate_per_min": 0.0,
            "oscillatory_event_duration_s": np.nan,
            "oscillatory_event_occupancy": 0.0,
            "oscillatory_event_peak_amplitude_uv": np.nan,
            "oscillatory_event_peak_frequency_hz": np.nan,
            "event_qc": "pass_no_events_detected",
        }, event_rows
    valid_duration_min = max(valid_samples / fs / 60.0, EPS)
    occupancy = float(np.sum([d * fs for d in durations]) / max(valid_samples, 1))
    return {
        "oscillatory_event_rate_per_min": float(len(durations) / valid_duration_min),
        "oscillatory_event_duration_s": float(np.nanmedian(durations)) if durations else np.nan,
        "oscillatory_event_occupancy": occupancy,
        "oscillatory_event_peak_amplitude_uv": float(np.nanmedian(peak_amps)) if peak_amps else np.nan,
        "oscillatory_event_peak_frequency_hz": float(np.nanmedian(peak_freqs)) if peak_freqs else np.nan,
        "event_qc": "pass",
    }, event_rows


def _mask_run_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0
    starts = np.diff(mask.astype(int), prepend=0) == 1
    return int(np.count_nonzero(starts))


def _feature_qc_for_window(
    filt_x: np.ndarray,
    f: np.ndarray,
    pxx: np.ndarray,
    cfg: LFPConfig,
    line_ratio: float,
) -> dict:
    x = np.asarray(filt_x, dtype=float)
    f = np.asarray(f, dtype=float)
    pxx = np.asarray(pxx, dtype=float)
    n = max(1, x.size)
    finite_x = x[np.isfinite(x)]
    filtered_finite_fraction = float(finite_x.size / n)
    filtered_rms = float(np.sqrt(np.mean(finite_x**2))) if finite_x.size else np.nan
    filtered_std = float(np.nanstd(finite_x)) if finite_x.size else np.nan
    filtered_robust_amplitude = robust_amplitude(finite_x) if finite_x.size else np.nan
    finite_psd = np.isfinite(f) & np.isfinite(pxx)
    positive_psd = finite_psd & (pxx > 0)
    finite_psd_bins = int(np.count_nonzero(finite_psd))
    positive_psd_bins = int(np.count_nonzero(positive_psd))
    total_band_bins = int(np.count_nonzero(finite_psd & (f >= cfg.exploratory_band_hz[0]) & (f <= cfg.exploratory_band_hz[1])))
    primary_band_bins = int(np.count_nonzero(finite_psd & (f >= cfg.primary_band_hz[0]) & (f <= cfg.primary_band_hz[1])))
    total_power = integrate_power(f, pxx, cfg.exploratory_band_hz) if f.size and pxx.size else np.nan
    primary_power = integrate_power(f, pxx, cfg.primary_band_hz) if f.size and pxx.size else np.nan
    total_power_ok = bool(np.isfinite(total_power) and total_power > 0)
    primary_power_ok = bool(np.isfinite(primary_power) and primary_power > 0)
    line_noise_qc = "pass"
    if not f.size or finite_psd_bins < 2 or not np.isfinite(line_ratio):
        line_noise_qc = "not_computable"
    elif line_ratio > cfg.line_noise_ratio_threshold:
        line_noise_qc = "fail"
    reasons: list[str] = []
    details: list[str] = []

    def add(reason: str, detail: str) -> None:
        reasons.append(reason)
        details.append(detail)

    if filtered_finite_fraction < cfg.minimum_valid_fraction:
        add(
            "filtered_finite_fraction",
            f"filtered_finite_sample_fraction={filtered_finite_fraction:.4g} < minimum_valid_fraction={cfg.minimum_valid_fraction:.4g}",
        )
    if finite_x.size == 0:
        add("filtered_signal_all_nan", "filtered signal has no finite samples after hard masks and filtering")
    if finite_x.size > 0 and (
        not np.isfinite(filtered_rms)
        or not np.isfinite(filtered_std)
        or filtered_rms <= EPS
        or filtered_std <= EPS
    ):
        add(
            "filtered_signal_zero_variance",
            f"filtered_rms={filtered_rms:.4g}; filtered_std={filtered_std:.4g}; numerical zero/constant filtered signal",
        )
    if finite_psd_bins < 2 or positive_psd_bins < 1:
        add(
            "invalid_psd",
            f"finite_psd_bin_count={finite_psd_bins}; positive_psd_bin_count={positive_psd_bins}; no valid PSD was produced",
        )
    if total_band_bins < 2:
        add(
            "insufficient_psd_bins_for_total_power",
            f"total_band_psd_bin_count={total_band_bins}; requested_band={cfg.exploratory_band_hz}",
        )
    if primary_band_bins < 2:
        add(
            "insufficient_psd_bins_for_primary_power",
            f"primary_band_psd_bin_count={primary_band_bins}; requested_band={cfg.primary_band_hz}",
        )
    if not total_power_ok:
        add("total_power_invalid", f"total_power_uV2={total_power}")
    if not primary_power_ok:
        add("primary_power_invalid", f"primary_power_uV2={primary_power}")

    unique_reasons = list(dict.fromkeys(reasons))
    unique_details = list(dict.fromkeys(details))
    return {
        "filtered_finite_sample_fraction": filtered_finite_fraction,
        "filtered_rms_uv": filtered_rms,
        "filtered_std_uv": filtered_std,
        "filtered_robust_amplitude_uv": filtered_robust_amplitude,
        "finite_psd_bin_count": finite_psd_bins,
        "positive_psd_bin_count": positive_psd_bins,
        "total_band_psd_bin_count": total_band_bins,
        "primary_band_psd_bin_count": primary_band_bins,
        "total_power_finite_positive": total_power_ok,
        "primary_power_finite_positive": primary_power_ok,
        "line_noise_qc": line_noise_qc,
        "feature_qc_pass": len(unique_reasons) == 0,
        "feature_qc_reason": "pass" if not unique_reasons else ";".join(unique_reasons),
        "feature_qc_details": "" if not unique_details else " | ".join(unique_details),
    }


def _qc_row_for_window(
    base: Mapping,
    raw_x: np.ndarray,
    filt_x: np.ndarray,
    valid: np.ndarray,
    masks: Mapping[str, np.ndarray],
    f: np.ndarray,
    pxx: np.ndarray,
    cfg: LFPConfig,
    reference_score: float,
) -> dict:
    n = max(1, raw_x.size)
    fs_used = _to_float_or_nan(base.get("fs_used_hz", ""))
    sample_interval_s = 1.0 / fs_used if np.isfinite(fs_used) and fs_used > 0 else 0.0
    mask_budget = _window_mask_budget(masks, sample_interval_s)
    missing_fraction = float(np.mean(~masks["finite"]))
    clipped_fraction = float(np.mean(masks["clipped"]))
    flatline_fraction = float(np.mean(masks["flatline"]))
    valid_fraction = float(np.mean(valid))
    large_derivative_mask = np.asarray(masks["large_derivative"], dtype=bool)
    large_derivative_fraction = float(np.mean(large_derivative_mask))
    large_derivative_count = _mask_run_count(large_derivative_mask)
    window_duration_s = _to_float_or_nan(base.get("window_duration_s", ""))
    if not np.isfinite(window_duration_s) or window_duration_s <= 0:
        window_duration_s = n * sample_interval_s
    derivative_event_rate_per_min = float(large_derivative_count / max(window_duration_s / 60.0, EPS))
    valid_without_derivative = (
        np.asarray(masks.get("finite", np.ones(n, dtype=bool)), dtype=bool)
        & ~np.asarray(masks.get("manual_artifact", np.zeros(n, dtype=bool)), dtype=bool)
        & ~np.asarray(masks.get("filter_edge", np.zeros(n, dtype=bool)), dtype=bool)
        & ~np.asarray(masks.get("clipped", np.zeros(n, dtype=bool)), dtype=bool)
        & ~np.asarray(masks.get("flatline", np.zeros(n, dtype=bool)), dtype=bool)
        & ~np.asarray(masks.get("time_discontinuity", np.zeros(n, dtype=bool)), dtype=bool)
        & ~np.asarray(masks.get("reference_excursion", np.zeros(n, dtype=bool)), dtype=bool)
    )
    valid_fraction_without_derivative_mask = float(np.mean(valid_without_derivative))
    usable_duration_without_derivative_mask_s = float(np.count_nonzero(valid_without_derivative) * sample_interval_s)
    derivative_z = np.asarray(masks.get("derivative_z_score", np.full(n, np.nan)), dtype=float)
    if derivative_z.size:
        step_score = float(np.nanmax(np.abs(derivative_z))) if np.isfinite(derivative_z).any() else 0.0
    else:
        step_score = 0.0
    line_ratio = _line_noise_ratio(f, pxx, cfg) if f.size else float("nan")
    feature_stats = _feature_qc_for_window(filt_x, f, pxx, cfg, line_ratio)
    robust_amp = robust_amplitude(raw_x)
    signal_reasons = []
    signal_details = []

    def add_failure(reason: str, detail: str) -> None:
        signal_reasons.append(reason)
        signal_details.append(detail)

    if valid_fraction < cfg.minimum_valid_fraction:
        add_failure(
            "valid_fraction",
            (
                f"valid_fraction={valid_fraction:.4g} < minimum_valid_fraction={cfg.minimum_valid_fraction:.4g}; "
                f"usable_duration_s={mask_budget.get('usable_duration_s', 0.0):.4g}; "
                f"combined_masked_s={mask_budget.get('combined_masked_s', 0.0):.4g}; "
                "window has too few usable samples after missing/artifact/filter-edge/reference masks"
            ),
        )
    if clipped_fraction > cfg.clipped_fraction_threshold:
        add_failure(
            "clipping",
            f"clipped_fraction={clipped_fraction:.4g} > clipped_fraction_threshold={cfg.clipped_fraction_threshold:.4g}",
        )
    if flatline_fraction > cfg.flatline_fraction_threshold:
        add_failure(
            "flatline",
            f"flatline_fraction={flatline_fraction:.4g} > flatline_fraction_threshold={cfg.flatline_fraction_threshold:.4g}",
        )
    derivative_fraction_fail = large_derivative_fraction > cfg.large_derivative_fraction_threshold
    derivative_fraction_warning = (
        large_derivative_fraction >= cfg.derivative_warning_fraction_threshold
        and large_derivative_fraction <= cfg.large_derivative_fraction_threshold
    )
    if derivative_fraction_fail:
        derivative_qc_status = "fail"
    elif derivative_fraction_warning:
        derivative_qc_status = "warning"
    else:
        derivative_qc_status = "pass"
    if np.isfinite(step_score) and derivative_fraction_fail:
        add_failure(
            "large_derivative_artifact_burden",
            (
                f"large_derivative_fraction={large_derivative_fraction:.4g} > "
                f"large_derivative_fraction_threshold={cfg.large_derivative_fraction_threshold:.4g}; "
                f"large_derivative_count={large_derivative_count}; "
                f"large_derivative_event_rate_per_min={derivative_event_rate_per_min:.4g}; "
                f"large_derivative_or_step_score={step_score:.4g}; "
                f"usable_duration_s={mask_budget.get('usable_duration_s', 0.0):.4g}"
            ),
        )
    if np.isfinite(line_ratio) and line_ratio > cfg.line_noise_ratio_threshold:
        add_failure(
            "line_noise",
            f"line_noise_ratio={line_ratio:.4g} > line_noise_ratio_threshold={cfg.line_noise_ratio_threshold:.4g}",
        )
    if np.isfinite(reference_score) and reference_score > cfg.reference_score_threshold:
        add_failure(
            "reference_or_common_mode",
            f"reference_or_common_mode_score={reference_score:.4g} > reference_score_threshold={cfg.reference_score_threshold:.4g}",
        )
    if cfg.max_robust_amplitude_uv is not None and np.isfinite(robust_amp) and robust_amp > cfg.max_robust_amplitude_uv:
        add_failure(
            "robust_amplitude",
            f"robust_amplitude={robust_amp:.4g} uV > max_robust_amplitude_uv={float(cfg.max_robust_amplitude_uv):.4g}",
        )
    signal_qc_pass = len(signal_reasons) == 0
    unique_signal_reasons = list(dict.fromkeys(signal_reasons))
    unique_signal_details = list(dict.fromkeys(signal_details))
    feature_qc_pass = bool(feature_stats["feature_qc_pass"])
    combined_reasons = []
    if not signal_qc_pass:
        combined_reasons.extend(unique_signal_reasons)
    if not feature_qc_pass:
        combined_reasons.extend(str(feature_stats["feature_qc_reason"]).split(";"))
    combined_reasons = [reason for reason in dict.fromkeys(combined_reasons) if reason and reason != "pass"]
    combined_details = []
    if unique_signal_details:
        combined_details.extend(unique_signal_details)
    if str(feature_stats.get("feature_qc_details", "")).strip():
        combined_details.append(str(feature_stats["feature_qc_details"]))
    qc_pass = bool(signal_qc_pass and feature_qc_pass)
    return {
        **base,
        **mask_budget,
        "missing_fraction": missing_fraction,
        "clipped_fraction": clipped_fraction,
        "flatline_fraction": flatline_fraction,
        "valid_fraction": valid_fraction,
        "robust_amplitude": robust_amp,
        "large_derivative_or_step_score": step_score,
        "large_derivative_fraction": large_derivative_fraction,
        "large_derivative_count": large_derivative_count,
        "large_derivative_event_rate_per_min": derivative_event_rate_per_min,
        "large_derivative_z_threshold": float(cfg.large_derivative_z_threshold),
        "derivative_warning_fraction_threshold": float(cfg.derivative_warning_fraction_threshold),
        "large_derivative_fraction_threshold": float(cfg.large_derivative_fraction_threshold),
        "derivative_qc_status": derivative_qc_status,
        "large_derivative_count_threshold_deprecated": int(cfg.large_derivative_count_threshold),
        "large_derivative_count_threshold_note": "deprecated_not_used_for_window_rejection",
        "valid_fraction_without_derivative_mask": valid_fraction_without_derivative_mask,
        "usable_duration_without_derivative_mask_s": usable_duration_without_derivative_mask_s,
        "minimum_valid_fraction": float(cfg.minimum_valid_fraction),
        "line_noise_ratio": line_ratio,
        "reference_or_common_mode_score": reference_score,
        "signal_qc_pass": bool(signal_qc_pass),
        "signal_qc_reason": "pass" if signal_qc_pass else ";".join(unique_signal_reasons),
        "signal_qc_details": "" if signal_qc_pass else " | ".join(unique_signal_details),
        **feature_stats,
        "qc_pass": bool(qc_pass),
        "qc_reason": "pass" if qc_pass else ";".join(combined_reasons),
        "qc_failure_details": "" if qc_pass else " | ".join(combined_details),
        "qc_failure_count": 0 if qc_pass else len(combined_reasons),
        "qc_exclusion": "" if qc_pass else "excluded_from_baseline_normalization_plots_and_statistics",
    }


def _load_spike_events_for_derivative_matching(cfg: LFPConfig, warnings_out: list[str]) -> pd.DataFrame:
    path = cfg.spike_events_csv
    if not path:
        return pd.DataFrame()
    spike_path = Path(path)
    if not spike_path.exists():
        warnings_out.append(f"Derivative-spike matching skipped: spike_events_csv not found: {spike_path}")
        return pd.DataFrame()
    spikes = pd.read_csv(spike_path)
    if "spike_time_s" not in spikes.columns or "channel" not in spikes.columns:
        warnings_out.append("Derivative-spike matching skipped: spike event file needs spike_time_s and channel columns.")
        return pd.DataFrame()
    return spikes


def _candidate_spikes_for_event(spikes: pd.DataFrame, base: Mapping, channel: str) -> pd.DataFrame:
    if spikes.empty:
        return spikes
    out = spikes[spikes["channel"].astype(str) == str(channel)].copy()
    prep = str(base.get("preparation_id", "")).strip()
    if "preparation_id" not in out.columns or not prep:
        return out.iloc[0:0]
    out = out[out["preparation_id"].astype(str) == prep]
    segment = str(base.get("segment_id", "")).strip()
    recording_id = str(base.get("recording_id", "")).strip()
    if "segment_id" in out.columns and segment:
        out = out[out["segment_id"].astype(str) == segment]
    elif "recording_id" in out.columns and recording_id:
        out = out[out["recording_id"].astype(str).isin({recording_id, segment})]
    else:
        return out.iloc[0:0]
    return out


def _nearest_spike_match(spikes: pd.DataFrame, base: Mapping, channel: str, event_peak_s: float, tolerance_s: float) -> tuple[float, float, bool]:
    candidates = _candidate_spikes_for_event(spikes, base, channel)
    if candidates.empty or not np.isfinite(event_peak_s):
        return np.nan, np.nan, False
    times = pd.to_numeric(candidates["spike_time_s"], errors="coerce").dropna().to_numpy(dtype=float)
    if times.size == 0:
        return np.nan, np.nan, False
    idx = int(np.argmin(np.abs(times - event_peak_s)))
    nearest = float(times[idx])
    latency = float(abs(nearest - event_peak_s))
    return nearest, latency, bool(latency <= float(tolerance_s))


def _derivative_event_rows_for_window(
    base: Mapping,
    raw_x: np.ndarray,
    t: np.ndarray,
    masks: Mapping[str, np.ndarray],
    cfg: LFPConfig,
    spike_events: pd.DataFrame,
) -> list[dict]:
    derivative_mask = np.asarray(masks.get("large_derivative", []), dtype=bool)
    derivative_z = np.asarray(masks.get("derivative_z_score", np.full(derivative_mask.size, np.nan)), dtype=float)
    derivative_uv_per_s = np.asarray(masks.get("derivative_uv_per_s", np.full(derivative_mask.size, np.nan)), dtype=float)
    raw = np.asarray(raw_x, dtype=float)
    t = np.asarray(t, dtype=float)
    n = min(derivative_mask.size, derivative_z.size, derivative_uv_per_s.size, raw.size, t.size)
    if n == 0:
        return []
    derivative_mask = derivative_mask[:n]
    derivative_z = derivative_z[:n]
    derivative_uv_per_s = derivative_uv_per_s[:n]
    raw = raw[:n]
    t = t[:n]
    rows = []
    tolerance_s = float(cfg.derivative_spike_match_tolerance_s)
    local_pad_s = max(tolerance_s, 0.002)
    channel = str(base.get("channel", ""))
    for event_index, (start, end) in enumerate(_contiguous_true_runs(derivative_mask), start=1):
        event_slice = slice(start, end)
        z_event = derivative_z[event_slice]
        deriv_event = derivative_uv_per_s[event_slice]
        if np.isfinite(z_event).any():
            local_peak = int(np.nanargmax(np.abs(z_event)))
        elif np.isfinite(deriv_event).any():
            local_peak = int(np.nanargmax(np.abs(deriv_event)))
        else:
            local_peak = 0
        peak_idx = min(start + local_peak, n - 1)
        event_start_s = float(t[start])
        event_peak_s = float(t[peak_idx])
        event_end_s = float(t[end - 1])
        if n > 1:
            dt = float(np.nanmedian(np.diff(t[np.isfinite(t)])))
        else:
            dt = 0.0
        event_duration_s = float(max(dt, event_end_s - event_start_s + max(dt, 0.0)))
        local = (t >= event_start_s - local_pad_s) & (t <= event_end_s + local_pad_s) & np.isfinite(raw)
        local_amp = robust_amplitude(raw[local]) if np.any(local) else np.nan
        nearest, latency, matched = _nearest_spike_match(spike_events, base, channel, event_peak_s, tolerance_s)
        rows.append(
            {
                "experiment_id": base.get("experiment_id", ""),
                "animal_id": base.get("animal_id", ""),
                "preparation_id": base.get("preparation_id", ""),
                "segment_id": base.get("segment_id", ""),
                "recording_id": base.get("recording_id", ""),
                "recording_order": base.get("recording_order", ""),
                "source_file": base.get("source_file", ""),
                "epoch_label": base.get("epoch_label", ""),
                "phase": base.get("phase", ""),
                "channel": channel,
                "window_index": base.get("window_index", ""),
                "event_index": event_index,
                "event_start_s": event_start_s,
                "event_peak_s": event_peak_s,
                "event_end_s": event_end_s,
                "event_duration_s": event_duration_s,
                "max_abs_voltage_derivative_uv_per_s": float(np.nanmax(np.abs(deriv_event))) if np.isfinite(deriv_event).any() else np.nan,
                "max_robust_derivative_z_score": float(np.nanmax(np.abs(z_event))) if np.isfinite(z_event).any() else np.nan,
                "large_derivative_z_threshold": float(cfg.large_derivative_z_threshold),
                "large_derivative_fraction_threshold": float(cfg.large_derivative_fraction_threshold),
                "local_raw_voltage_amplitude_uv": local_amp,
                "nearest_spike_time_s": nearest,
                "nearest_spike_abs_latency_s": latency,
                "spike_match_tolerance_s": tolerance_s,
                "matched_to_spike_within_tolerance": bool(matched),
                "coincident_derivative_channel_count": np.nan,
                "coincident_derivative_channel_fraction": np.nan,
                "coincidence_tolerance_s": float(cfg.derivative_coincidence_tolerance_s),
                "provisional_classification": "candidate_transient_matched_to_spike" if matched else "candidate_derivative_transient_unmatched",
                "classification_note": "Automatic label is provisional; derivative events are not classified as neural spikes or artifacts without inspection.",
            }
        )
    return rows


def _add_derivative_channel_coincidence(events: pd.DataFrame, manifest: pd.DataFrame, tolerance_s: float) -> pd.DataFrame:
    if events.empty:
        return events
    out = events.copy()
    group_cols = [col for col in ("preparation_id", "segment_id", "source_file") if col in out.columns]
    if not group_cols:
        return out
    selected_counts = {}
    if not manifest.empty:
        for key, sub in manifest.groupby([col for col in group_cols if col in manifest.columns], dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            selected_counts[key] = max(1, int(sub["channel"].astype(str).nunique()))
    for key, sub in out.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        denom = selected_counts.get(key, max(1, int(sub["channel"].astype(str).nunique())))
        peaks = pd.to_numeric(sub["event_peak_s"], errors="coerce")
        for idx, peak in peaks.items():
            if not np.isfinite(peak):
                continue
            coincident = sub[np.abs(pd.to_numeric(sub["event_peak_s"], errors="coerce") - float(peak)) <= float(tolerance_s)]
            count = int(coincident["channel"].astype(str).nunique())
            out.loc[idx, "coincident_derivative_channel_count"] = count
            out.loc[idx, "coincident_derivative_channel_fraction"] = float(count / max(denom, 1))
    return out


def _derivative_sensitivity_report(qc: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    if qc.empty or "large_derivative_fraction" not in qc.columns:
        return pd.DataFrame()
    rows = []
    reasons = qc.get("qc_reason", pd.Series("pass", index=qc.index)).astype(str)
    def reason_parts(value: object) -> set[str]:
        text = str(value).strip()
        if text.lower() in {"", "nan", "none"}:
            return {"pass"}
        return {part.strip() for part in text.split(";") if part.strip()}

    parsed_reasons = reasons.apply(reason_parts)
    hard_failure_reasons = parsed_reasons.apply(lambda parts: parts - {"pass", "large_derivative_artifact_burden", "large_derivative_or_step"})
    non_derivative_ok = hard_failure_reasons.apply(lambda parts: len(parts) == 0)
    min_valid = pd.to_numeric(qc.get("minimum_valid_fraction", pd.Series(0.90, index=qc.index)), errors="coerce").fillna(0.90)
    valid_fraction = pd.to_numeric(qc.get("valid_fraction", pd.Series(np.nan, index=qc.index)), errors="coerce")
    for threshold in thresholds:
        derivative_ok = pd.to_numeric(qc["large_derivative_fraction"], errors="coerce") <= float(threshold)
        valid_ok = valid_fraction >= min_valid
        would_pass = non_derivative_ok & derivative_ok & valid_ok
        rows.append(
            {
                "scenario": "candidate_derivative_fraction_threshold",
                "large_derivative_fraction_threshold_candidate": float(threshold),
                "derivative_samples_masked_before_filtering": False,
                "windows_total": int(len(qc)),
                "windows_would_pass": int(would_pass.sum()),
                "windows_would_fail": int((~would_pass).sum()),
                "recorded_time_would_pass_s": float(pd.to_numeric(qc.loc[would_pass, "window_duration_s"], errors="coerce").sum()) if "window_duration_s" in qc.columns else np.nan,
                "usable_time_would_pass_s": float(pd.to_numeric(qc.loc[would_pass, "usable_duration_s"], errors="coerce").sum()) if "usable_duration_s" in qc.columns else np.nan,
                "selection_note": "Report only; candidate derivative detections are not sample-masked before filtering.",
            }
        )
    if "valid_fraction_without_derivative_mask" in qc.columns:
        valid_without_derivative = pd.to_numeric(qc["valid_fraction_without_derivative_mask"], errors="coerce")
        other_reasons_without_valid = hard_failure_reasons.apply(lambda parts: parts - {"valid_fraction"})
        would_pass_without_derivative_mask = (
            other_reasons_without_valid.apply(lambda parts: len(parts) == 0)
            & (valid_without_derivative >= min_valid)
        )
        rows.append(
            {
                "scenario": "derivative_masking_disabled",
                "large_derivative_fraction_threshold_candidate": np.nan,
                "derivative_samples_masked_before_filtering": False,
                "windows_total": int(len(qc)),
                "windows_would_pass": int(would_pass_without_derivative_mask.sum()),
                "windows_would_fail": int((~would_pass_without_derivative_mask).sum()),
                "recorded_time_would_pass_s": float(pd.to_numeric(qc.loc[would_pass_without_derivative_mask, "window_duration_s"], errors="coerce").sum()) if "window_duration_s" in qc.columns else np.nan,
                "usable_time_would_pass_s": float(
                    pd.to_numeric(qc.loc[would_pass_without_derivative_mask, "usable_duration_without_derivative_mask_s"], errors="coerce").sum()
                ) if "usable_duration_without_derivative_mask_s" in qc.columns else np.nan,
                "selection_note": (
                    "Sensitivity report only; production LFP features already leave candidate derivative samples in the filtering input."
                ),
            }
        )
    return pd.DataFrame(rows)


def _feature_row_for_window(
    base: Mapping,
    x: np.ndarray,
    t: np.ndarray,
    fs: float,
    f: np.ndarray,
    pxx: np.ndarray,
    cfg: LFPConfig,
) -> tuple[dict, list[dict]]:
    valid_x = x[np.isfinite(x)]
    if valid_x.size == 0:
        row = {**base}
        return row, []
    rms = float(np.sqrt(np.mean(valid_x**2)))
    mad = robust_mad(valid_x)
    p2p = robust_amplitude(valid_x)
    total_power = integrate_power(f, pxx, cfg.exploratory_band_hz)
    primary_power = integrate_power(f, pxx, cfg.primary_band_hz)
    fit = _aperiodic_fit(f, pxx, cfg.exploratory_band_hz)
    peak = _peak_features(f, pxx, cfg, fit)
    finite_runs = _contiguous_true_runs(np.isfinite(x))
    if finite_runs:
        longest = max(finite_runs, key=lambda item: item[1] - item[0])
        ac_input = x[longest[0] : longest[1]]
    else:
        ac_input = valid_x
    tau, period = _autocorrelation_timescale(ac_input, fs)
    row = {
        **base,
        "lfp_rms_uv": rms,
        "lfp_mad_uv": mad,
        "lfp_robust_peak_to_peak_uv": p2p,
        "total_power_uV2": total_power,
        "total_power_db": 10.0 * math.log10(total_power + EPS) if np.isfinite(total_power) and total_power > 0 else np.nan,
        "primary_power_uV2": primary_power,
        "primary_power_db": 10.0 * math.log10(primary_power + EPS) if np.isfinite(primary_power) and primary_power > 0 else np.nan,
        "autocorrelation_timescale_s": tau,
        "dominant_period_s": period,
        **fit,
        **peak,
    }
    for key, label, band in cfg.marker_bands:
        power = integrate_power(f, pxx, band)
        prefix = f"band_{sanitize_name(key)}"
        row[f"{prefix}_label"] = label
        row[f"{prefix}_lo_hz"] = float(band[0])
        row[f"{prefix}_hi_hz"] = float(band[1])
        row[f"{prefix}_power_uV2"] = power
        row[f"{prefix}_power_db"] = 10.0 * math.log10(power + EPS) if np.isfinite(power) and power > 0 else np.nan
        row[f"{prefix}_relative_power"] = float(power / total_power) if np.isfinite(power) and np.isfinite(total_power) and total_power > 0 else np.nan
    event_summary, events = _event_features(x, t, fs, cfg, base)
    row.update(event_summary)
    return row, events


def _window_bounds(t_rel: np.ndarray, window_s: float, usable_mask: np.ndarray | None = None) -> list[tuple[int, int, float, float]]:
    if t_rel.size == 0:
        return []
    if usable_mask is None:
        usable_mask = np.ones_like(t_rel, dtype=bool)
    else:
        usable_mask = np.asarray(usable_mask, dtype=bool) & np.isfinite(t_rel)
    dt = float(np.median(np.diff(t_rel))) if t_rel.size > 1 else 0.0
    bounds = []
    for run_start, run_end in _contiguous_true_runs(usable_mask):
        run_t = t_rel[run_start:run_end]
        if run_t.size < 2:
            continue
        start = float(run_t[0])
        end = float(run_t[-1] + dt)
        if end <= start:
            continue
        nwin = int(math.ceil((end - start) / float(window_s)))
        for idx in range(nwin):
            a = start + idx * float(window_s)
            b = min(start + (idx + 1) * float(window_s), end)
            sel = np.where((t_rel >= a) & (t_rel < b) & usable_mask)[0]
            if sel.size < 2:
                continue
            bounds.append((int(sel[0]), int(sel[-1]) + 1, float(a), float(b)))
    return bounds


def _reference_columns_for_scheme(scheme: str, channel: str) -> list[str]:
    raw = str(scheme or "none").strip()
    if not raw or raw.lower() == "none":
        return []
    lower = raw.lower()
    if lower.startswith("bipolar:"):
        ref = raw.split(":", 1)[1].strip()
        return [ref] if ref and ref != channel else []
    if "-" in raw and not lower.startswith("common"):
        parts = [p.strip() for p in raw.split("-", 1)]
        return [part for part in parts if part and part != channel]
    return []


def _apply_reference(raw_by_col: Mapping[str, np.ndarray], channel: str, scheme: str) -> tuple[np.ndarray, np.ndarray | None, str]:
    raw = np.asarray(raw_by_col[channel], dtype=float)
    clean_scheme = str(scheme or "none").strip() or "none"
    lower = clean_scheme.lower()
    if lower == "none":
        return raw, None, "none"
    if lower.startswith("bipolar:"):
        ref_col = clean_scheme.split(":", 1)[1].strip()
        if not ref_col:
            return raw, None, "none"
        ref = np.asarray(raw_by_col[ref_col], dtype=float)
        return raw - ref, ref, f"bipolar:{ref_col}"
    if lower.startswith("common"):
        stack = np.vstack([np.asarray(values, dtype=float) for values in raw_by_col.values()])
        ref = np.nanmedian(stack, axis=0)
        return raw - ref, ref, "common_average"
    if "-" in clean_scheme:
        left, right = [part.strip() for part in clean_scheme.split("-", 1)]
        if channel == left and right in raw_by_col:
            ref = np.asarray(raw_by_col[right], dtype=float)
            return raw - ref, ref, f"bipolar:{right}"
    return raw, None, clean_scheme


def _reference_score(raw: np.ndarray, ref: np.ndarray | None) -> float:
    if ref is None:
        return 0.0
    finite = np.isfinite(raw) & np.isfinite(ref)
    if np.sum(finite) < 4:
        return float("nan")
    raw_center = raw[finite] - np.nanmedian(raw[finite])
    ref_center = ref[finite] - np.nanmedian(ref[finite])
    if np.nanstd(raw_center) <= 0 or np.nanstd(ref_center) <= 0:
        return float("nan")
    return float(abs(np.corrcoef(raw_center, ref_center)[0, 1]))


def _metadata_warnings(manifest: pd.DataFrame) -> list[str]:
    warnings_out = []
    for col in ("animal_id", "preparation_id", "epoch_label", "phase"):
        missing = manifest[col].astype(str).str.strip().eq("").sum()
        if missing:
            warnings_out.append(
                f"{missing} recording/channel row(s) are missing {col}; statistics using biological n may be incomplete."
            )
    if "baseline" not in set(manifest["phase"].astype(str).str.strip().str.lower()):
        warnings_out.append("No explicitly labelled baseline phase was found; baseline normalization will be unavailable.")
    if any("h2o2" in str(value).lower() for value in manifest.get("treatment", pd.Series(dtype=str))):
        warnings_out.append(H2O2_WARNING)
    if any(_boolish(value) for value in manifest.get("vehicle_perfusion_control", pd.Series(dtype=str))):
        warnings_out.append(H2O2_WARNING)
    if any(_boolish(value) for value in manifest.get("no_tissue_h2o2_control", pd.Series(dtype=str))):
        warnings_out.append(H2O2_WARNING)
    return list(dict.fromkeys(warnings_out))


def _add_baseline_normalization(features: pd.DataFrame, warnings_out: list[str]) -> pd.DataFrame:
    if features.empty:
        return features
    df = features.copy()
    if "qc_pass" not in df.columns:
        df["qc_pass"] = True
    phase = df["phase"].astype(str).str.strip().str.lower()
    baseline = df[(phase == "baseline") & (df["qc_pass"].astype(bool))].copy()
    group_cols = [col for col in ("experiment_id", "animal_id", "preparation_id", "channel") if col in df.columns]
    if baseline.empty or not group_cols:
        warnings_out.append("Baseline normalization skipped: no valid explicit baseline windows.")
        return df
    if baseline[group_cols].astype(str).apply(lambda col: col.str.strip().eq("")).any().any():
        warnings_out.append("Some baseline rows have missing biological-replicate metadata; normalization falls back to available grouping keys.")
    power_cols = [col for col in df.columns if col.endswith("_power_uV2") or col in ("total_power_uV2", "primary_power_uV2")]
    amp_cols = [col for col in ("lfp_rms_uv", "lfp_mad_uv", "lfp_robust_peak_to_peak_uv") if col in df.columns]
    peak_cols = [col for col in ("dominant_peak_frequency_hz", "oscillatory_event_peak_frequency_hz") if col in df.columns]
    ref_cols = power_cols + amp_cols + peak_cols
    refs = baseline.groupby(group_cols, dropna=False)[ref_cols].median(numeric_only=True).reset_index()
    refs = refs.rename(columns={col: f"baseline_median_{col}" for col in ref_cols})
    df = df.merge(refs, on=group_cols, how="left")
    for col in power_cols:
        ref = f"baseline_median_{col}"
        if ref not in df:
            continue
        value = pd.to_numeric(df[col], errors="coerce")
        baseline_value = pd.to_numeric(df[ref], errors="coerce")
        ratio = value / baseline_value
        df[f"{col}_change_db"] = 10.0 * np.log10(ratio)
        df[f"{col}_baseline_ratio"] = ratio
        df[f"{col}_percent_change"] = 100.0 * (value - baseline_value) / baseline_value
        df[f"{col}_absolute_difference"] = value - baseline_value
    for col in amp_cols:
        ref = f"baseline_median_{col}"
        value = pd.to_numeric(df[col], errors="coerce")
        baseline_value = pd.to_numeric(df[ref], errors="coerce")
        ratio = value / baseline_value
        df[f"{col}_change_db"] = 20.0 * np.log10(ratio)
        df[f"{col}_baseline_ratio"] = ratio
        df[f"{col}_percent_change"] = 100.0 * (value - baseline_value) / baseline_value
        df[f"{col}_absolute_difference"] = value - baseline_value
    for col in peak_cols:
        ref = f"baseline_median_{col}"
        df[f"{col}_change_hz"] = pd.to_numeric(df[col], errors="coerce") - pd.to_numeric(df[ref], errors="coerce")
    return df


def _phase_summary(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    valid = features[features.get("qc_pass", True).astype(bool)].copy()
    if valid.empty:
        return pd.DataFrame()
    group_cols = [
        col
        for col in (
            "experiment_id",
            "animal_id",
            "preparation_id",
            "channel",
            "epoch_label",
            "phase",
            "treatment",
            "concentration_value",
            "concentration_unit",
        )
        if col in valid.columns
    ]
    numeric_candidates = [
        "lfp_rms_uv",
        "lfp_mad_uv",
        "lfp_robust_peak_to_peak_uv",
        "total_power_uV2",
        "primary_power_uV2",
        "total_power_uV2_change_db",
        "primary_power_uV2_change_db",
        "dominant_peak_frequency_hz",
        "peak_power_over_aperiodic_db",
        "oscillatory_event_rate_per_min",
        "oscillatory_event_duration_s",
        "oscillatory_event_occupancy",
    ]
    band_cols = [col for col in valid.columns if col.startswith("band_") and (col.endswith("_power_uV2") or col.endswith("_change_db"))]
    value_cols = [col for col in numeric_candidates + band_cols if col in valid.columns]
    if not value_cols:
        return pd.DataFrame()
    summary = valid.groupby(group_cols, dropna=False)[value_cols].agg(["median", "mean", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    summary["statistics_note"] = "Summaries use animal/preparation as the biological n; windows/channels are repeated observations."
    return summary


def _baseline_post_statistics(features: pd.DataFrame, warnings_out: list[str]) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    valid = features[features.get("qc_pass", True).astype(bool)].copy()
    if valid.empty:
        return pd.DataFrame()
    valid["phase_norm"] = valid["phase"].astype(str).str.strip().str.lower()
    group_cols = [col for col in ("experiment_id", "animal_id", "preparation_id", "channel", "phase_norm") if col in valid.columns]
    feature_cols = [
        col
        for col in (
            "lfp_rms_uv",
            "lfp_robust_peak_to_peak_uv",
            "total_power_uV2",
            "primary_power_uV2",
            "dominant_peak_frequency_hz",
            "oscillatory_event_rate_per_min",
        )
        if col in valid.columns
    ]
    if not feature_cols or not group_cols:
        return pd.DataFrame()
    channel_phase = valid.groupby(group_cols, dropna=False)[feature_cols].median(numeric_only=True).reset_index()
    rows = []
    channel_id_cols = [col for col in ("experiment_id", "animal_id", "preparation_id", "channel") if col in channel_phase.columns]
    prep_id_cols = [col for col in ("experiment_id", "animal_id", "preparation_id") if col in channel_phase.columns]
    baseline = channel_phase[channel_phase["phase_norm"] == "baseline"]
    if baseline.empty:
        warnings_out.append("Baseline/post statistics skipped: no valid baseline preparation summaries.")
        return pd.DataFrame()
    if baseline["preparation_id"].astype(str).str.strip().eq("").all():
        warnings_out.append(
            "Population-level paired statistics skipped: preparation_id is missing; descriptive channel-level normalization is still exported."
        )
        return pd.DataFrame()
    for phase_name in sorted(set(channel_phase["phase_norm"]) - {"baseline"}):
        other = channel_phase[channel_phase["phase_norm"] == phase_name]
        paired = baseline.merge(other, on=channel_id_cols, suffixes=("_baseline", "_phase"))
        if paired.empty:
            warnings_out.append(
                f"No complete matched baseline-to-{phase_name} channel pairs were available; "
                "channels missing from one phase were excluded from that paired comparison."
            )
            continue
        prep_pairs = paired.groupby(prep_id_cols, dropna=False).median(numeric_only=True).reset_index()
        for feature in feature_cols:
            b = pd.to_numeric(prep_pairs[f"{feature}_baseline"], errors="coerce")
            p = pd.to_numeric(prep_pairs[f"{feature}_phase"], errors="coerce")
            keep = b.notna() & p.notna()
            n = int(prep_pairs.loc[keep, "preparation_id"].astype(str).str.strip().replace("", np.nan).dropna().nunique()) if "preparation_id" in prep_pairs.columns else int(keep.sum())
            diff = p[keep].to_numpy(dtype=float) - b[keep].to_numpy(dtype=float)
            if n >= 2 and np.any(np.abs(diff) > 0):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        p_value = float(stats.wilcoxon(p[keep], b[keep], zero_method="wilcox").pvalue)
                        test_name = "paired_wilcoxon_by_preparation"
                    except ValueError:
                        p_value = float("nan")
                        test_name = "paired_wilcoxon_by_preparation_unavailable"
            else:
                p_value = float("nan")
                test_name = "insufficient_paired_preparations"
            rows.append(
                {
                    "phase": phase_name,
                    "feature": feature,
                    "biological_n_preparations": n,
                    "baseline_median": float(np.nanmedian(b[keep])) if n else np.nan,
                    "phase_median": float(np.nanmedian(p[keep])) if n else np.nan,
                    "paired_difference_median": float(np.nanmedian(diff)) if n else np.nan,
                    "test": test_name,
                    "p_value": p_value,
                    "statistics_note": "Windows/channels were collapsed to preparation-level medians before paired testing.",
                    "mixed_effects_note": "Mixed-effects model not run unless statsmodels and richer replicate metadata are available.",
                }
            )
    if len(set(valid["phase_norm"])) > 1:
        try:
            import statsmodels.api as _sm  # noqa: F401
            import statsmodels.formula.api as smf

            mixed_rows = []
            for feature in feature_cols:
                model_df = valid.dropna(subset=[feature, "preparation_id", "phase_norm"]).copy()
                if model_df["preparation_id"].nunique() >= 2 and model_df["phase_norm"].nunique() >= 2:
                    model = smf.mixedlm(f"{feature} ~ C(phase_norm)", model_df, groups=model_df["preparation_id"]).fit(reml=False)
                    mixed_rows.append(
                        {
                            "phase": "mixed_effects",
                            "feature": feature,
                            "biological_n_preparations": int(model_df["preparation_id"].nunique()),
                            "baseline_median": np.nan,
                            "phase_median": np.nan,
                            "paired_difference_median": np.nan,
                            "test": "mixedlm_phase_fixed_effect_preparation_random_intercept",
                            "p_value": np.nan,
                            "statistics_note": "Model uses preparation as grouping variable; inspect model summary separately.",
                            "mixed_effects_note": str(model.summary()).splitlines()[0],
                        }
                    )
            rows.extend(mixed_rows)
        except Exception:
            warnings_out.append("Mixed-effects model skipped because statsmodels is unavailable or metadata are insufficient.")
    return pd.DataFrame(rows)


def _matched_channel_availability(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty or "channel" not in features.columns:
        return pd.DataFrame()
    df = features.copy()
    if "qc_pass" in df.columns:
        df = df[df["qc_pass"].astype(bool)].copy()
    if df.empty:
        return pd.DataFrame()
    df["phase_norm"] = df["phase"].astype(str).str.strip().str.lower()
    id_cols = [col for col in ("experiment_id", "animal_id", "preparation_id", "channel") if col in df.columns]
    if not id_cols:
        return pd.DataFrame()
    rows = []
    for key, sub in df.groupby(id_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(id_cols, key))
        phases = sorted(str(value) for value in sub["phase_norm"].dropna().unique())
        epochs = sorted(str(value) for value in sub.get("epoch_label", sub["phase"]).dropna().unique())
        has_baseline = "baseline" in phases
        nonbaseline = [phase for phase in phases if phase != "baseline"]
        row.update(
            {
                "available_phases": ";".join(phases),
                "available_epoch_labels": ";".join(epochs),
                "has_valid_baseline": bool(has_baseline),
                "has_valid_nonbaseline_epoch": bool(nonbaseline),
                "paired_comparison_ready": bool(has_baseline and nonbaseline),
                "exclusion_reason": "" if has_baseline and nonbaseline else (
                    "missing_valid_baseline" if not has_baseline else "missing_valid_nonbaseline_epoch"
                ),
                "replication_note": "Channel is a repeated technical/spatial observation nested within preparation, not an independent biological replicate.",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _add_psd_baseline_change(psd: pd.DataFrame, warnings_out: list[str]) -> pd.DataFrame:
    if psd.empty or "psd_uV2_per_hz" not in psd.columns:
        return psd
    df = psd.copy()
    if "phase" not in df.columns:
        return df
    df["phase_norm"] = df["phase"].astype(str).str.strip().str.lower()
    id_cols = [col for col in ("experiment_id", "animal_id", "preparation_id", "channel", "frequency_hz") if col in df.columns]
    baseline = df[df["phase_norm"] == "baseline"].copy()
    if baseline.empty or not id_cols:
        warnings_out.append("PSD baseline-change table skipped: no valid matched baseline PSD rows.")
        return df
    refs = (
        baseline.groupby(id_cols, dropna=False)["psd_uV2_per_hz"]
        .median()
        .reset_index()
        .rename(columns={"psd_uV2_per_hz": "baseline_median_psd_uV2_per_hz"})
    )
    df = df.merge(refs, on=id_cols, how="left")
    value = pd.to_numeric(df["psd_uV2_per_hz"], errors="coerce")
    base = pd.to_numeric(df["baseline_median_psd_uV2_per_hz"], errors="coerce")
    ratio = value / base
    df["psd_change_db"] = 10.0 * np.log10(ratio)
    df["psd_baseline_ratio"] = ratio
    df["psd_percent_change"] = 100.0 * (value - base) / base
    return df


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def process_lfp_recordings(recordings, config: LFPConfig | Mapping | None = None) -> dict[str, str]:
    """Extract LFP features from wide raw-conversion CSV recordings.

    Parameters
    ----------
    recordings:
        Ordered iterable of dictionaries. ``source_file`` (or ``path``), phase,
        animal_id, preparation_id, and channel metadata are used directly; phase
        is never inferred from filenames.
    config:
        ``LFPConfig`` or a dict with matching keys.

    Returns
    -------
    dict
        Paths for the manifest, features, QC, events, summaries, PSD table, and
        processing-parameter JSON.
    """
    cfg = as_lfp_config(config)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings_out: list[str] = []
    if cfg.streaming:
        warnings_out.append("Streaming API selected; current implementation uses the same numerically stable batch backend.")

    normalized = [_normalise_recording(rec, order, cfg) for order, rec in enumerate(recordings, start=1)]
    if not normalized:
        raise FileNotFoundError("No LFP raw recordings were provided.")

    manifest_rows: list[dict] = []
    feature_rows: list[dict] = []
    qc_rows: list[dict] = []
    event_rows: list[dict] = []
    derivative_event_rows: list[dict] = []
    psd_rows: list[dict] = []
    lfp_stream_metadata: list[dict] = []
    cumulative_offsets_by_preparation: dict[str, float] = {}
    spike_events_for_derivatives = _load_spike_events_for_derivative_matching(cfg, warnings_out)

    for rec in sorted(normalized, key=lambda item: (str(item.get("preparation_id") or ""), int(item.get("recording_order") or 0))):
        path = Path(str(rec["source_file"]))
        if not path.exists():
            raise FileNotFoundError(f"LFP raw CSV not found: {path}")
        header = pd.read_csv(path, nrows=0)
        time_col = find_time_column(list(header.columns), cfg.time_column_candidates)
        requested_channels = parse_channel_list(rec.get("channel")) or cfg.channels
        if requested_channels:
            missing_channels = [channel for channel in requested_channels if channel not in header.columns]
            if missing_channels:
                warnings_out.append(
                    f"Recording segment {rec.get('segment_id', path.stem)} ({path.name}) is missing requested "
                    f"LFP channel(s) {missing_channels}; available channels are retained, and missing channels "
                    "are excluded from matched paired comparisons."
                )
        channels = _list_signal_columns(header, time_col, requested_channels)
        if not channels:
            warnings_out.append(f"No requested numeric LFP channels were available in segment {rec.get('segment_id', path.stem)} ({path.name}); segment skipped.")
            continue
        scheme = str(rec.get("reference_scheme") or cfg.reference_scheme or "none").strip() or "none"
        ref_cols = []
        for channel in channels:
            ref_cols.extend(_reference_columns_for_scheme(scheme, channel))
        usecols = list(dict.fromkeys([time_col, *channels, *ref_cols]))
        df = pd.read_csv(path, usecols=[col for col in usecols if col in header.columns])
        t_abs = df[time_col].to_numpy(dtype=float)
        t_info = validate_timestamps(t_abs, rec.get("sampling_rate_hz"))
        fs = float(t_info["fs_est_hz"])
        t_rel = t_abs - float(t_abs[0])
        recording_duration_s = float(t_rel[-1] - t_rel[0] + t_info["median_dt_s"])
        preparation_key = str(rec.get("preparation_id") or "__unknown_preparation__")
        cumulative_offset_s = cumulative_offsets_by_preparation.get(preparation_key, 0.0)
        raw_by_col = {}
        for col in [c for c in usecols if c != time_col and c in df.columns]:
            scale, unit_used = _unit_scale_to_uv(rec.get("signal_unit", cfg.signal_unit), cfg.scale_to_uv, col)
            raw_by_col[col] = df[col].to_numpy(dtype=float) * scale
        for channel in channels:
            scale, unit_used = _unit_scale_to_uv(rec.get("signal_unit", cfg.signal_unit), cfg.scale_to_uv, channel)
            rec_manifest = {col: rec.get(col, "") for col in MANIFEST_COLUMNS}
            rec_manifest.update(
                {
                    "segment_id": rec.get("segment_id", rec.get("recording_id", path.stem)),
                    "epoch_label": rec.get("epoch_label", rec.get("phase", "")),
                    "recording_order": int(rec.get("recording_order") or 0),
                    "source_file": str(path),
                    "channel": channel,
                    "reference_scheme": scheme,
                    "sampling_rate_hz": rec.get("sampling_rate_hz") or fs,
                    "signal_unit": unit_used,
                    **t_info,
                }
            )
            manifest_rows.append(rec_manifest)
            x_ref_uv, reference_uv, applied_scheme = _apply_reference(raw_by_col, channel, scheme)
            masks = _artifact_masks(t_rel, x_ref_uv, cfg, rec, reference_uv, fs)
            segment_budget = _window_mask_budget(masks, float(t_info["median_dt_s"]), prefix="segment_")
            segment_budget["segment_duration_s"] = recording_duration_s
            segment_budget["segment_recorded_start_s"] = 0.0
            segment_budget["segment_recorded_end_s"] = recording_duration_s
            filter_repair_mask = (
                (~masks["finite"])
                | masks["manual_artifact"]
                | masks["clipped"]
                | masks["flatline"]
                | masks["time_discontinuity"]
                | masks["reference_excursion"]
            )
            x_for_filter = np.array(x_ref_uv, dtype=float, copy=True)
            x_for_filter[filter_repair_mask] = np.nan
            lfp_stream = create_lfp_stream(t_rel, x_for_filter, masks["valid"], fs, cfg.downsample_hz, warnings_out)
            filtered_t = lfp_stream.t_rel_s
            fs_used = lfp_stream.fs_hz
            filtered = _apply_filters_by_runs(lfp_stream.x_uv, fs_used, cfg, warnings_out)
            raw_for_windows = np.interp(filtered_t, t_rel, x_ref_uv)
            lfp_stream_metadata.append(
                {
                    "experiment_id": rec.get("experiment_id", ""),
                    "animal_id": rec.get("animal_id", ""),
                    "preparation_id": rec.get("preparation_id", ""),
                    "segment_id": rec.get("segment_id", rec.get("recording_id", path.stem)),
                    "recording_id": rec.get("recording_id", path.stem),
                    "recording_order": int(rec.get("recording_order") or 0),
                    "epoch_label": rec.get("epoch_label", rec.get("phase", "")),
                    "phase": str(rec.get("phase", "")).strip(),
                    "source_file": str(path),
                    "channel": channel,
                    "source_sampling_rate_hz": fs,
                    "lfp_sampling_rate_hz": fs_used,
                    "analysis_filter_band_hz": list(cfg.exploratory_band_hz),
                    "deterministic_filter_edge_exclusion_s": _effective_filter_edge_s(t_rel, cfg),
                    "stream_metadata": lfp_stream.metadata,
                }
            )
            masks_ds = {}
            for name, mask in masks.items():
                arr = np.asarray(mask)
                if arr.dtype == bool:
                    masks_ds[name] = np.interp(filtered_t, t_rel, arr.astype(float)) > 0.5
                else:
                    filled = np.asarray(arr, dtype=float)
                    finite_values = np.isfinite(filled)
                    if np.count_nonzero(finite_values) >= 2:
                        masks_ds[name] = np.interp(filtered_t, t_rel[finite_values], filled[finite_values], left=np.nan, right=np.nan)
                    else:
                        masks_ds[name] = np.full_like(filtered_t, np.nan, dtype=float)
            hard_valid_ds = np.asarray(masks_ds["valid"], dtype=bool)
            masks_ds["downsample_validity_loss"] = hard_valid_ds & ~np.asarray(lfp_stream.valid_mask, dtype=bool)
            analysis_valid_mask = hard_valid_ds & np.asarray(lfp_stream.valid_mask, dtype=bool)
            ref_score = _reference_score(raw_for_windows, np.interp(filtered_t, t_rel, _interp_fill(reference_uv)) if reference_uv is not None else None)
            window_domain = (~masks_ds["filter_edge"]) & np.isfinite(filtered_t)
            window_bounds = _window_bounds(filtered_t, cfg.analysis_window_s, window_domain)
            if not window_bounds:
                warnings_out.append(
                    f"No usable LFP analysis windows remained for segment {rec.get('segment_id', path.stem)} "
                    f"channel {channel} after deterministic filter-edge and finite-run exclusion."
                )
            for window_index, (i0, i1, start_s, end_s) in enumerate(window_bounds):
                base = {
                    "experiment_id": rec.get("experiment_id", ""),
                    "animal_id": rec.get("animal_id", ""),
                    "preparation_id": rec.get("preparation_id", ""),
                    "segment_id": rec.get("segment_id", rec.get("recording_id", path.stem)),
                    "recording_id": rec.get("recording_id", path.stem),
                    "recording_order": int(rec.get("recording_order") or 0),
                    "source_file": str(path),
                    "epoch_label": rec.get("epoch_label", rec.get("phase", "")),
                    "phase": str(rec.get("phase", "")).strip(),
                    "treatment": rec.get("treatment", ""),
                    "concentration_value": rec.get("concentration_value", ""),
                    "concentration_unit": rec.get("concentration_unit", ""),
                    "application_onset_s": rec.get("application_onset_s", ""),
                    "washout_onset_s": rec.get("washout_onset_s", ""),
                    "channel": channel,
                    "reference_scheme": applied_scheme,
                    "electrode_type": rec.get("electrode_type", ""),
                    "impedance_1khz_ohm": rec.get("impedance_1khz_ohm", ""),
                    "reference_stability": rec.get("reference_stability", ""),
                    "vehicle_perfusion_control": rec.get("vehicle_perfusion_control", ""),
                    "no_tissue_h2o2_control": rec.get("no_tissue_h2o2_control", ""),
                    "window_index": window_index,
                    "window_start_s": start_s,
                    "window_end_s": end_s,
                    "window_mid_s": (start_s + end_s) / 2.0,
                    "window_duration_s": end_s - start_s,
                    "cumulative_window_start_s": cumulative_offset_s + start_s,
                    "cumulative_window_end_s": cumulative_offset_s + end_s,
                    "cumulative_window_mid_s": cumulative_offset_s + (start_s + end_s) / 2.0,
                    "fs_est_hz": fs,
                    "fs_used_hz": fs_used,
                    "signal_unit": unit_used,
                    "analysis_label": "LFP feature extraction from raw wide CSV",
                    "interpretation_warning": H2O2_WARNING,
                    **segment_budget,
                }
                xw = filtered[i0:i1]
                raww = raw_for_windows[i0:i1]
                tw = filtered_t[i0:i1]
                validw = hard_valid_ds[i0:i1]
                analysis_validw = analysis_valid_mask[i0:i1]
                xw_masked = np.array(xw, dtype=float, copy=True)
                xw_masked[~analysis_validw] = np.nan
                f, pxx = _welch_window_psd(xw_masked, fs_used, cfg, warnings_out)
                qc = _qc_row_for_window(
                    base,
                    raww,
                    xw_masked,
                    validw,
                    {name: mask[i0:i1] for name, mask in masks_ds.items()},
                    f,
                    pxx,
                    cfg,
                    ref_score,
                )
                qc_rows.append(qc)
                window_masks = {name: mask[i0:i1] for name, mask in masks_ds.items()}
                derivative_event_rows.extend(_derivative_event_rows_for_window(base, raww, tw, window_masks, cfg, spike_events_for_derivatives))
                if bool(qc["qc_pass"]):
                    features, events = _feature_row_for_window(base, xw_masked, tw, fs_used, f, pxx, cfg)
                    features.update(qc)
                    feature_rows.append(features)
                    event_rows.extend(events)
                    if cfg.write_psd_long:
                        for freq, power in zip(f, pxx):
                            psd_rows.append(
                                {
                                    **base,
                                    "frequency_hz": float(freq),
                                    "psd_uV2_per_hz": float(power),
                                    "psd_db": float(10.0 * math.log10(float(power) + EPS)),
                                }
                            )
                else:
                    failed_feature_row = {**qc}
                    failed_feature_row.setdefault("event_qc", "not_run_qc_failed")
                    feature_rows.append(failed_feature_row)
        cumulative_offsets_by_preparation[preparation_key] = cumulative_offset_s + recording_duration_s

    manifest = pd.DataFrame(manifest_rows)
    features = pd.DataFrame(feature_rows)
    qc = pd.DataFrame(qc_rows)
    if "qc_pass" in qc.columns:
        qc_failures = qc[~qc["qc_pass"].astype(bool)].copy()
    else:
        qc_failures = pd.DataFrame(columns=qc.columns)
    events = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    derivative_events = pd.DataFrame(derivative_event_rows, columns=DERIVATIVE_EVENT_COLUMNS)
    psd = pd.DataFrame(psd_rows)
    if psd.empty:
        psd = pd.DataFrame(columns=[*WINDOW_BASE_COLUMNS, "frequency_hz", "psd_uV2_per_hz", "psd_db"])

    warnings_out.extend(_metadata_warnings(manifest))
    derivative_events = _add_derivative_channel_coincidence(derivative_events, manifest, cfg.derivative_coincidence_tolerance_s)
    features = _add_baseline_normalization(features, warnings_out)
    psd = _add_psd_baseline_change(psd, warnings_out)
    matched_channel_availability = _matched_channel_availability(features)
    phase_summary = _phase_summary(features)
    statistics_table = _baseline_post_statistics(features, warnings_out)
    derivative_sensitivity = _derivative_sensitivity_report(qc, cfg.derivative_sensitivity_thresholds)
    if phase_summary.empty:
        phase_summary = pd.DataFrame(
            columns=[
                "experiment_id",
                "animal_id",
                "preparation_id",
                "channel",
                "epoch_label",
                "phase",
                "treatment",
                "concentration_value",
                "concentration_unit",
                "statistics_note",
            ]
        )
    if matched_channel_availability.empty:
        matched_channel_availability = pd.DataFrame(
            columns=[
                "experiment_id",
                "animal_id",
                "preparation_id",
                "channel",
                "available_phases",
                "available_epoch_labels",
                "has_valid_baseline",
                "has_valid_nonbaseline_epoch",
                "paired_comparison_ready",
                "exclusion_reason",
                "replication_note",
            ]
        )
    if statistics_table.empty:
        statistics_table = pd.DataFrame(
            columns=[
                "phase",
                "feature",
                "biological_n_preparations",
                "baseline_median",
                "phase_median",
                "paired_difference_median",
                "test",
                "p_value",
                "statistics_note",
                "mixed_effects_note",
            ]
        )
    if derivative_sensitivity.empty:
        derivative_sensitivity = pd.DataFrame(
            columns=[
                "scenario",
                "large_derivative_fraction_threshold_candidate",
                "derivative_samples_masked_before_filtering",
                "windows_total",
                "windows_would_pass",
                "windows_would_fail",
                "recorded_time_would_pass_s",
                "usable_time_would_pass_s",
                "selection_note",
            ]
        )

    paths = {
        "manifest_csv": str(out_dir / "lfp_recording_manifest.csv"),
        "features_csv": str(out_dir / "lfp_features_by_window.csv"),
        "qc_csv": str(out_dir / "lfp_qc_by_window.csv"),
        "qc_failures_csv": str(out_dir / "lfp_qc_failures.csv"),
        "events_csv": str(out_dir / "lfp_events.csv"),
        "derivative_events_csv": str(out_dir / "lfp_derivative_events.csv"),
        "phase_summary_csv": str(out_dir / "lfp_phase_summary.csv"),
        "baseline_post_statistics_csv": str(out_dir / "lfp_baseline_post_statistics.csv"),
        "matched_channel_availability_csv": str(out_dir / "lfp_matched_channel_availability.csv"),
        "derivative_sensitivity_csv": str(out_dir / "lfp_derivative_qc_sensitivity.csv"),
        "psd_long_csv": str(out_dir / "lfp_psd_long.csv"),
        "simple_lfp_plots_dir": str(out_dir / "lfp_plots"),
        "simple_lfp_plot_outputs_json": str(out_dir / "lfp_simple_plot_outputs.json"),
        "parameters_json": str(out_dir / "lfp_processing_parameters.json"),
    }
    manifest.to_csv(paths["manifest_csv"], index=False)
    features.to_csv(paths["features_csv"], index=False)
    qc.to_csv(paths["qc_csv"], index=False)
    qc_failures.to_csv(paths["qc_failures_csv"], index=False)
    events.to_csv(paths["events_csv"], index=False)
    derivative_events.to_csv(paths["derivative_events_csv"], index=False)
    phase_summary.to_csv(paths["phase_summary_csv"], index=False)
    statistics_table.to_csv(paths["baseline_post_statistics_csv"], index=False)
    matched_channel_availability.to_csv(paths["matched_channel_availability_csv"], index=False)
    derivative_sensitivity.to_csv(paths["derivative_sensitivity_csv"], index=False)
    if cfg.write_psd_long:
        psd.to_csv(paths["psd_long_csv"], index=False)
    else:
        Path(paths["psd_long_csv"]).write_text("", encoding="utf-8")

    simple_plot_outputs: dict[str, list[str] | str] = {"plots": [], "tables": []}
    if cfg.write_simple_plots:
        try:
            from lfp_plotting import plot_lfp_features

            simple_plot_outputs = plot_lfp_features(
                paths["features_csv"],
                {
                    "output_dir": paths["simple_lfp_plots_dir"],
                    "plot_mode": "simple",
                    "qc_csv": paths["qc_csv"],
                    "psd_long_csv": paths["psd_long_csv"],
                    "events_csv": paths["events_csv"],
                    "derivative_events_csv": paths["derivative_events_csv"],
                    "simple_plot_bands": cfg.simple_plot_bands,
                },
            )
        except Exception as exc:
            simple_plot_outputs = {"plots": [], "tables": [], "error": str(exc)}
            warnings_out.append(f"Simple LFP plot generation failed: {exc}")
    Path(paths["simple_lfp_plot_outputs_json"]).write_text(
        json.dumps(_json_ready(simple_plot_outputs), indent=2),
        encoding="utf-8",
    )

    params = {
        "config": _json_ready(asdict(cfg)),
        "output_paths": paths,
        "simple_lfp_plot_outputs": _json_ready(simple_plot_outputs),
        "warnings": list(dict.fromkeys(warnings_out)),
        "lfp_stream_preprocessing": _json_ready(lfp_stream_metadata),
        "lfp_stream_contract": (
            "Spike detection stays on the original recording clock and sampling rate. LFP features use a synchronized "
            "recording-relative stream created from the same raw CSV by explicit anti-alias filtering and per-run "
            "polyphase resampling before 0.5-300 Hz analysis filtering. Separate files, timestamp resets/gaps, and "
            "NaN/non-finite runs are never filtered or resampled across."
        ),
        "h2o2_safety_warning": H2O2_WARNING,
        "derivative_detection_note": (
            "Large-derivative candidates are detected on referenced raw extracellular voltage before LFP filtering/downsampling, "
            "inside contiguous finite time-continuous runs only. Candidate derivative samples are not converted to NaN, "
            "do not divide filtering runs, and are not automatically classified as neural spikes or artifacts."
        ),
        "schemas": {
            "lfp_recording_manifest.csv": list(manifest.columns),
            "lfp_features_by_window.csv": list(features.columns),
            "lfp_qc_by_window.csv": list(qc.columns),
            "lfp_qc_failures.csv": list(qc_failures.columns),
            "lfp_events.csv": list(events.columns),
            "lfp_derivative_events.csv": list(derivative_events.columns),
            "lfp_phase_summary.csv": list(phase_summary.columns),
            "lfp_baseline_post_statistics.csv": list(statistics_table.columns),
            "lfp_matched_channel_availability.csv": list(matched_channel_availability.columns),
            "lfp_derivative_qc_sensitivity.csv": list(derivative_sensitivity.columns),
            "lfp_psd_long.csv": list(psd.columns),
        },
    }
    Path(paths["parameters_json"]).write_text(json.dumps(params, indent=2), encoding="utf-8")
    for msg in params["warnings"]:
        print(f"[lfp warning] {msg}")
    if not qc_failures.empty:
        print("[lfp qc] failed windows by reason:")
        for reason, count in qc_failures["qc_reason"].value_counts(dropna=False).items():
            print(f"  {count}: {reason}")
        print(f"[lfp qc] details for failed windows:\n  {paths['qc_failures_csv']}")
    print(f"[lfp] wrote features: {paths['features_csv']}")
    return paths


def _load_recordings_from_manifest(path: str | Path) -> list[dict]:
    df = pd.read_csv(path)
    if "source_file" not in df.columns:
        raise ValueError("Manifest CSV must contain a source_file column.")
    return df.to_dict(orient="records")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract windowed LFP features from raw wide CSV recordings.")
    parser.add_argument("inputs", nargs="*", help="Raw wide CSV file(s), ignored if --manifest is supplied.")
    parser.add_argument("--manifest", help="Recording manifest CSV with explicit phase/animal/preparation metadata.")
    parser.add_argument("--out-dir", default="lfp_analysis", help="Output folder.")
    parser.add_argument("--channels", default="", help="Comma-separated LFP channel names. Blank means all numeric signal columns.")
    parser.add_argument("--phase", default="", help="Explicit phase to apply to positional inputs.")
    parser.add_argument("--animal-id", default="", help="Animal id for positional inputs.")
    parser.add_argument("--preparation-id", default="", help="Preparation id for positional inputs.")
    parser.add_argument("--window-sec", type=float, default=60.0)
    parser.add_argument("--primary-band", default="1-40")
    parser.add_argument("--exploratory-band", default="0.5-300")
    parser.add_argument("--marker-bands", default="")
    parser.add_argument("--welch-segment-sec", type=float, default=4.0)
    parser.add_argument("--notch-frequency", default="50")
    parser.add_argument("--downsample-hz", default=str(int(DEFAULT_LFP_DOWNSAMPLE_HZ)))
    parser.add_argument("--signal-unit", default="uV")
    parser.add_argument("--simple-plot-bands", default="", help="Optional simple plot bands, e.g. very_slow=0.5-2; slow_motor=2-5.")
    parser.add_argument("--simple-plots", dest="write_simple_plots", action="store_true", default=True, help="Write simple per-recording PSD and band-power plots after extraction.")
    parser.add_argument("--no-simple-plots", dest="write_simple_plots", action="store_false", help="Skip automatic simple LFP plots.")
    return parser


def _parse_pair(text: str) -> tuple[float, float]:
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    parts = [part for part in re.split(r"\s*[-:,]\s*", text.strip()) if part]
    if len(parts) != 2:
        raise ValueError(f"Expected lo-hi frequency pair, got {text!r}.")
    return float(parts[0]), float(parts[1])


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.manifest:
        recordings = _load_recordings_from_manifest(args.manifest)
    else:
        recordings = [
            {
                "source_file": path,
                "phase": args.phase,
                "animal_id": args.animal_id,
                "preparation_id": args.preparation_id,
                "recording_order": idx,
            }
            for idx, path in enumerate(args.inputs, start=1)
        ]
    cfg = LFPConfig(
        output_dir=args.out_dir,
        channels=parse_channel_list(args.channels),
        analysis_window_s=args.window_sec,
        primary_band_hz=_parse_pair(args.primary_band),
        exploratory_band_hz=_parse_pair(args.exploratory_band),
        marker_bands=parse_band_spec(args.marker_bands),
        welch_segment_s=args.welch_segment_sec,
        notch_frequency_hz=None if str(args.notch_frequency).strip().lower() in {"", "none", "0"} else float(args.notch_frequency),
        downsample_hz=None if str(args.downsample_hz).strip().lower() in {"", "none", "0"} else float(args.downsample_hz),
        signal_unit=args.signal_unit,
        write_simple_plots=bool(args.write_simple_plots),
        simple_plot_bands=parse_band_spec(args.simple_plot_bands) if args.simple_plot_bands else None,
    )
    process_lfp_recordings(recordings, cfg)


if __name__ == "__main__":
    main()
