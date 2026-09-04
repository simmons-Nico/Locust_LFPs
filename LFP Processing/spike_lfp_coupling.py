#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Spike-LFP coupling for SPIE.

Primary output is pairwise phase consistency (PPC) computed from LFP phase at
threshold-crossing spike times exported by the Spike Analysis pipeline.
Conventional spike-field coherence (SFC) is exported as a companion marker.
These outputs are descriptive and must not be interpreted as causal
connectivity or single-unit coupling.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

from lfp_feature_extraction import (
    DEFAULT_LFP_DOWNSAMPLE_HZ,
    DEFAULT_LFP_EXPLORATORY_BAND_HZ,
    H2O2_WARNING,
    _interp_fill,
    _unit_scale_to_uv,
    create_lfp_stream,
    find_time_column,
    normalize_epoch_phase,
    parse_channel_list,
    validate_timestamps,
)


EPS = np.finfo(float).eps
COUPLING_WARNING = (
    "PPC/SFC are MUA-LFP phase-coupling markers for threshold crossings and "
    "do not establish single-unit identity or causal connectivity."
)
SAME_CHANNEL_WARNING = (
    "Same-channel spike-LFP coupling is useful for longitudinal channel matching, "
    "but spike-waveform leakage can inflate PPC/SFC."
)
RATE_POWER_POSITIVE_INTERPRETATION = "Positive correlation: windows with greater firing tend to have greater LFP power."
RATE_POWER_NEGATIVE_INTERPRETATION = "Negative correlation: greater firing accompanies reduced power in that band."
RATE_POWER_NO_CORRELATION_INTERPRETATION = "No correlation: spike rate and band power do not consistently covary."
RATE_POWER_EPOCH_CAVEAT = (
    "Epoch separation without within-epoch correlation: both variables may respond independently to treatment, "
    "producing an apparent pooled correlation."
)
DEFAULT_COUPLING_PHASE_BANDS = [
    ("delta_1_4", "1-4 Hz", (1.0, 4.0)),
    ("theta_4_8", "4-8 Hz", (4.0, 8.0)),
    ("alpha_8_13", "8-13 Hz", (8.0, 13.0)),
    ("beta_13_30", "13-30 Hz", (13.0, 30.0)),
    ("low_gamma_30_60", "30-60 Hz", (30.0, 60.0)),
    ("high_gamma_60_100", "60-100 Hz", (60.0, 100.0)),
]
PPC_COLUMNS = [
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
    "lfp_channel",
    "spike_channel",
    "channel_pair",
    "pairing_mode",
    "same_channel",
    "band_key",
    "band_label",
    "phase_band_lo_hz",
    "phase_band_hi_hz",
    "window_index",
    "window_start_s",
    "window_end_s",
    "window_mid_s",
    "cumulative_window_mid_s",
    "spike_count",
    "valid_lfp_duration_s",
    "valid_oscillatory_cycles",
    "cycle_qc_pass",
    "primary_power_uV2",
    "ppc",
    "preferred_phase_rad",
    "preferred_phase_deg",
    "surrogate_mean",
    "surrogate_ci_low",
    "surrogate_ci_high",
    "surrogate_p_value",
    "surrogate_q_value",
    "surrogate_significant_fdr_0p05",
    "signal_qc_pass",
    "signal_qc_reason",
    "feature_qc_pass",
    "feature_qc_reason",
    "coupling_qc_pass",
    "coupling_qc_reason",
    "qc_pass",
    "qc_reason",
    "baseline_median_ppc",
    "ppc_change_from_baseline",
    "coupling_label",
    "interpretation_warning",
    "same_channel_warning",
    "high_frequency_same_channel_flag",
    "h2o2_safety_warning",
]
RATE_POWER_COLUMNS = [
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
    "lfp_channel",
    "spike_channel",
    "channel_pair",
    "pairing_mode",
    "same_channel",
    "window_index",
    "window_start_s",
    "window_end_s",
    "window_mid_s",
    "cumulative_window_mid_s",
    "spike_count",
    "valid_lfp_duration_s",
    "spike_rate_hz",
    "spike_rate_per_min",
    "lfp_power_uV2",
    "lfp_power_label",
    "signal_qc_pass",
    "feature_qc_pass",
    "matched_window_qc_pass",
    "matched_window_qc_reason",
]
RATE_POWER_SUMMARY_COLUMNS = [
    "scope",
    "experiment_id",
    "animal_id",
    "preparation_id",
    "lfp_channel",
    "spike_channel",
    "channel_pair",
    "epoch_label",
    "phase",
    "lfp_power_label",
    "n_matched_windows",
    "spearman_rho_s",
    "spearman_p_value",
    "interpretation",
    "epoch_separation_caveat",
]
SFC_COLUMNS = [
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
    "lfp_channel",
    "spike_channel",
    "channel_pair",
    "pairing_mode",
    "same_channel",
    "band_key",
    "band_label",
    "window_index",
    "window_start_s",
    "window_end_s",
    "window_mid_s",
    "cumulative_window_mid_s",
    "spike_count",
    "valid_lfp_duration_s",
    "frequency_hz",
    "sfc",
    "baseline_median_sfc",
    "sfc_change_from_baseline",
    "rate_matched_sfc",
    "signal_qc_pass",
    "feature_qc_pass",
    "coupling_qc_pass",
    "qc_pass",
]
STA_COLUMNS = [
    "experiment_id",
    "animal_id",
    "preparation_id",
    "segment_id",
    "recording_id",
    "recording_order",
    "source_file",
    "epoch_label",
    "phase",
    "lfp_channel",
    "spike_channel",
    "channel_pair",
    "pairing_mode",
    "same_channel",
    "sta_lfp_band_lo_hz",
    "sta_lfp_band_hi_hz",
    "sta_window_s",
    "spike_count_total",
    "spike_count_sampled",
    "spike_count_used",
    "source_sampling_rate_hz",
    "lfp_sampling_rate_hz",
    "sta_csv",
    "sta_plot",
    "sta_spectrogram_plot",
    "interpretation_warning",
    "same_channel_warning",
]


@dataclass
class SpikeLFPCouplingConfig:
    output_dir: str | Path = "spike_lfp_coupling"
    time_column_candidates: tuple[str, ...] = ("time_s", "time", "t", "seconds")
    lfp_channels: list[str] | None = None
    spike_channels: list[str] | None = None
    phase_band_hz: tuple[float, float] = (1.0, 100.0)
    phase_bands: list[tuple[str, str, tuple[float, float]]] | None = None
    window_s: float = 60.0
    min_spikes: int = 20
    min_oscillatory_cycles: float = 3.0
    surrogate_count: int = 200
    random_seed: int = 13
    signal_unit: str = "uV"
    scale_to_uv: float | None = None
    filter_order: int = 4
    notch_frequency_hz: float | None = 50.0
    notch_q: float = 30.0
    pairing_mode: str = "same_channel"
    processing_mode: str = "complex"
    explicit_channel_pairs: list[tuple[str, str]] = field(default_factory=list)
    allow_large_all_to_all: bool = False
    max_projected_ppc: int = 250_000
    coupling_downsample_hz: float | None = DEFAULT_LFP_DOWNSAMPLE_HZ
    surrogate_chunk_size: int = 64
    rate_match_sfc: bool = True
    max_sfc_frequency_hz: float = DEFAULT_LFP_EXPLORATORY_BAND_HZ[1]
    max_sfc_frequencies: int = 128
    sfc_flush_rows: int = 50_000
    sta_lfp_band_hz: tuple[float, float] = DEFAULT_LFP_EXPLORATORY_BAND_HZ
    sta_window_s: float = 0.200
    sta_min_spikes: int = 5
    sta_max_spikes_per_pair: int = 500
    sta_max_pairs: int = 24
    sta_spectrogram: bool = True
    sta_spectrogram_max_frequency_hz: float = 200.0
    lfp_feature_csv: str | Path | None = None
    spike_processing_provenance: Mapping | None = None


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_") or "plot"


def _as_bool(series: pd.Series, default: bool = False) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(default)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "pass"})


def _parse_frequency_pair(text: str) -> tuple[float, float]:
    parts = [part for part in re.split(r"\s*[-:,]\s*", str(text).strip()) if part]
    if len(parts) != 2:
        raise ValueError(f"Expected lo-hi frequency pair, got {text!r}.")
    lo, hi = float(parts[0]), float(parts[1])
    if not 0 < lo < hi:
        raise ValueError(f"Frequency band must have 0 < low < high, got {text!r}.")
    return lo, hi


def _parse_phase_bands(value: object) -> list[tuple[str, str, tuple[float, float]]] | None:
    if value is None or value == "":
        return None
    bands: list[tuple[str, str, tuple[float, float]]] = []
    if isinstance(value, str):
        for item in value.split(";"):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                key, rng = item.split("=", 1)
                label = key.strip()
            else:
                key = ""
                rng = item
                label = ""
            lo, hi = _parse_frequency_pair(rng)
            if not key:
                key = f"band_{lo:g}_{hi:g}"
            if not label:
                label = f"{lo:g}-{hi:g} Hz"
            bands.append((_safe_name(key).lower(), label, (lo, hi)))
        return bands
    for item in value:  # type: ignore[union-attr]
        if isinstance(item, Mapping):
            key = str(item.get("key") or item.get("name") or "")
            label = str(item.get("label") or key or "")
            if "band" in item:
                lo, hi = item["band"]
            else:
                lo, hi = item.get("lo"), item.get("hi")
            lo, hi = float(lo), float(hi)
        else:
            key, label, band = item
            lo, hi = band
            key = str(key)
            label = str(label)
            lo, hi = float(lo), float(hi)
        if not 0 < lo < hi:
            raise ValueError(f"Invalid phase band {item!r}.")
        if not key:
            key = f"band_{lo:g}_{hi:g}"
        if not label:
            label = f"{lo:g}-{hi:g} Hz"
        bands.append((_safe_name(key).lower(), label, (lo, hi)))
    return bands


def _parse_explicit_pairs(value: object) -> list[tuple[str, str]]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        pairs = []
        for item in value.split(";"):
            item = item.strip()
            if not item:
                continue
            if "->" in item:
                spike, lfp = [part.strip() for part in item.split("->", 1)]
            else:
                parts = [part.strip() for part in item.split(",") if part.strip()]
                if len(parts) != 2:
                    raise ValueError("Explicit pairs must look like spike_channel->lfp_channel or spike_channel,lfp_channel.")
                spike, lfp = parts
            pairs.append((spike, lfp))
        return pairs
    return [(str(a), str(b)) for a, b in value]  # type: ignore[misc]


def as_coupling_config(config: SpikeLFPCouplingConfig | Mapping | None = None) -> SpikeLFPCouplingConfig:
    if isinstance(config, SpikeLFPCouplingConfig):
        cfg = config
    else:
        cfg = SpikeLFPCouplingConfig()
        if config:
            raw = dict(config)
            if "lfp_channels" in raw:
                raw["lfp_channels"] = parse_channel_list(raw["lfp_channels"])
            if "spike_channels" in raw:
                raw["spike_channels"] = parse_channel_list(raw["spike_channels"])
            if "phase_bands" in raw:
                raw["phase_bands"] = _parse_phase_bands(raw["phase_bands"])
            if "explicit_channel_pairs" in raw:
                raw["explicit_channel_pairs"] = _parse_explicit_pairs(raw["explicit_channel_pairs"])
            if "sta_lfp_band_hz" in raw and isinstance(raw["sta_lfp_band_hz"], str):
                raw["sta_lfp_band_hz"] = _parse_frequency_pair(raw["sta_lfp_band_hz"])
            for key, value in raw.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
    cfg.phase_band_hz = tuple(float(x) for x in cfg.phase_band_hz)  # type: ignore[assignment]
    cfg.phase_bands = _parse_phase_bands(cfg.phase_bands)
    cfg.explicit_channel_pairs = _parse_explicit_pairs(cfg.explicit_channel_pairs)
    cfg.window_s = float(cfg.window_s)
    cfg.min_spikes = int(cfg.min_spikes)
    cfg.min_oscillatory_cycles = max(0.0, float(cfg.min_oscillatory_cycles))
    cfg.surrogate_count = int(cfg.surrogate_count)
    cfg.random_seed = int(cfg.random_seed)
    cfg.filter_order = int(cfg.filter_order)
    cfg.surrogate_chunk_size = max(1, int(cfg.surrogate_chunk_size))
    cfg.max_projected_ppc = int(cfg.max_projected_ppc)
    cfg.max_sfc_frequencies = max(1, int(cfg.max_sfc_frequencies))
    cfg.sfc_flush_rows = max(1, int(cfg.sfc_flush_rows))
    cfg.sta_lfp_band_hz = tuple(float(x) for x in cfg.sta_lfp_band_hz)  # type: ignore[assignment]
    cfg.sta_window_s = max(0.0, float(cfg.sta_window_s))
    cfg.sta_min_spikes = max(1, int(cfg.sta_min_spikes))
    cfg.sta_max_spikes_per_pair = max(1, int(cfg.sta_max_spikes_per_pair))
    cfg.sta_max_pairs = max(0, int(cfg.sta_max_pairs))
    cfg.sta_spectrogram = bool(cfg.sta_spectrogram)
    cfg.sta_spectrogram_max_frequency_hz = max(1.0, float(cfg.sta_spectrogram_max_frequency_hz))
    cfg.pairing_mode = str(cfg.pairing_mode or "same_channel").strip().lower().replace("-", "_")
    cfg.processing_mode = str(cfg.processing_mode or "complex").strip().lower().replace("-", "_")
    if cfg.processing_mode in {"basic", "quick", "screening"}:
        cfg.processing_mode = "simple"
    if cfg.processing_mode in {"full", "advanced", "detailed"}:
        cfg.processing_mode = "complex"
    if cfg.processing_mode not in {"simple", "complex"}:
        raise ValueError("processing_mode must be simple or complex.")
    if cfg.pairing_mode in {"same", "matched", "matched_channel"}:
        cfg.pairing_mode = "same_channel"
    if cfg.pairing_mode in {"all", "all-to-all"}:
        cfg.pairing_mode = "all_to_all"
    if cfg.pairing_mode not in {"same_channel", "explicit", "all_to_all"}:
        raise ValueError("pairing_mode must be same_channel, explicit, or all_to_all.")
    if cfg.notch_frequency_hz in ("", "none", "None", 0):
        cfg.notch_frequency_hz = None
    if cfg.notch_frequency_hz is not None:
        cfg.notch_frequency_hz = float(cfg.notch_frequency_hz)
    if cfg.coupling_downsample_hz in ("", "none", "None", 0):
        cfg.coupling_downsample_hz = None
    if cfg.coupling_downsample_hz is not None:
        cfg.coupling_downsample_hz = float(cfg.coupling_downsample_hz)
    cfg.max_sfc_frequency_hz = float(cfg.max_sfc_frequency_hz)
    return cfg


def _phase_band_defs(cfg: SpikeLFPCouplingConfig) -> list[tuple[str, str, tuple[float, float]]]:
    if cfg.phase_bands:
        return cfg.phase_bands
    lo, hi = cfg.phase_band_hz
    if math.isclose(lo, 1.0) and (math.isclose(hi, 40.0) or math.isclose(hi, 100.0)):
        return list(DEFAULT_COUPLING_PHASE_BANDS)
    return [(f"band_{lo:g}_{hi:g}", f"{lo:g}-{hi:g} Hz", (lo, hi))]


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def pairwise_phase_consistency(phases: np.ndarray) -> float:
    """Return unbiased PPC from spike phases in radians."""
    phi = np.asarray(phases, dtype=float)
    phi = phi[np.isfinite(phi)]
    n = phi.size
    if n < 2:
        return float("nan")
    vec = np.exp(1j * phi)
    return float((np.abs(np.sum(vec)) ** 2 - n) / (n * (n - 1)))


def _ppc_many(phases: np.ndarray) -> np.ndarray:
    arr = np.asarray(phases, dtype=float)
    if arr.ndim == 1:
        return np.asarray([pairwise_phase_consistency(arr)], dtype=float)
    finite = np.isfinite(arr)
    n = finite.sum(axis=1).astype(float)
    vec = np.where(finite, np.exp(1j * arr), 0.0)
    denom = n * (n - 1.0)
    out = (np.abs(vec.sum(axis=1)) ** 2 - n) / denom
    out[denom <= 0] = np.nan
    return out.astype(float)


def _bandpass_phase(
    x_uv: np.ndarray,
    fs: float,
    band: tuple[float, float],
    order: int = 4,
    notch_frequency_hz: float | None = 50.0,
    notch_q: float = 30.0,
) -> np.ndarray:
    y = _interp_fill(np.asarray(x_uv, dtype=float))
    y = y - float(np.nanmedian(y))
    nyq = 0.5 * fs
    if y.size < max(16, order * 8) or band[1] >= nyq:
        return np.full(y.shape, np.nan, dtype=float)
    if notch_frequency_hz and 0 < notch_frequency_hz < nyq * 0.98:
        b, a = signal.iirnotch(float(notch_frequency_hz) / nyq, Q=float(notch_q))
        y = signal.filtfilt(b, a, y)
    lo = max(float(band[0]) / nyq, 1e-6)
    hi = min(float(band[1]) / nyq, 0.999)
    sos = signal.butter(int(order), [lo, hi], btype="bandpass", output="sos")
    try:
        y = signal.sosfiltfilt(sos, y)
    except ValueError:
        y = signal.sosfiltfilt(sos, y, padlen=max(0, min(y.size - 1, 3 * (2 * len(sos) + 1))))
    return np.angle(signal.hilbert(y))


def _phase_at_times(t: np.ndarray, phase: np.ndarray, spike_times: np.ndarray) -> np.ndarray:
    if t.size < 2 or spike_times.size == 0:
        return np.array([], dtype=float)
    complex_phase = np.exp(1j * phase)
    real = np.interp(spike_times, t, complex_phase.real, left=np.nan, right=np.nan)
    imag = np.interp(spike_times, t, complex_phase.imag, left=np.nan, right=np.nan)
    both = np.isfinite(real) & np.isfinite(imag)
    out = np.full(spike_times.shape, np.nan, dtype=float)
    out[both] = np.angle(real[both] + 1j * imag[both])
    return out


def _window_bounds_from_time(t_rel: np.ndarray, window_s: float) -> pd.DataFrame:
    if t_rel.size < 2:
        return pd.DataFrame(columns=["window_index", "window_start_s", "window_end_s", "window_mid_s", "valid_lfp_duration_s"])
    dt = float(np.nanmedian(np.diff(t_rel)))
    duration = float(t_rel[-1] - t_rel[0] + dt)
    rows = []
    for idx in range(int(math.ceil(duration / float(window_s)))):
        start = float(t_rel[0] + idx * window_s)
        end = float(min(t_rel[0] + (idx + 1) * window_s, t_rel[0] + duration))
        rows.append(
            {
                "window_index": idx,
                "window_start_s": start,
                "window_end_s": end,
                "window_mid_s": (start + end) / 2.0,
                "valid_lfp_duration_s": end - start,
                "signal_qc_pass": True,
                "signal_qc_reason": "not_available",
                "feature_qc_pass": True,
                "feature_qc_reason": "not_available",
                "qc_pass": True,
            }
        )
    return pd.DataFrame(rows)


def _load_lfp_records(lfp_data, cfg: SpikeLFPCouplingConfig) -> list[dict]:
    if isinstance(lfp_data, Mapping):
        if "manifest_csv" in lfp_data:
            df = pd.read_csv(lfp_data["manifest_csv"])
            return df.to_dict(orient="records")
        if "source_file" in lfp_data or "path" in lfp_data:
            return [dict(lfp_data)]
    if isinstance(lfp_data, (str, Path)):
        path = Path(lfp_data)
        if path.name == "lfp_recording_manifest.csv":
            return pd.read_csv(path).to_dict(orient="records")
        return [{"source_file": str(path), "recording_id": path.stem, "phase": ""}]
    return [dict(item) if isinstance(item, Mapping) else {"source_file": str(item)} for item in lfp_data]


def _unique_segment_records(records: list[dict]) -> list[dict]:
    unique: list[dict] = []
    index_by_key: dict[tuple[str, str, str], int] = {}
    for rec in records:
        path = str(rec.get("source_file") or rec.get("path") or "")
        segment_id = str(rec.get("segment_id") or rec.get("recording_id") or Path(path).stem)
        prep = str(rec.get("preparation_id") or "")
        key = (path, segment_id, prep)
        channel_text = str(rec.get("channel") or "").strip()
        if key in index_by_key:
            if channel_text:
                existing = unique[index_by_key[key]]
                merged = parse_channel_list(existing.get("channel")) or []
                for channel in parse_channel_list(channel_text) or [channel_text]:
                    if channel not in merged:
                        merged.append(channel)
                existing["channel"] = ",".join(merged)
            continue
        rec = dict(rec)
        rec["segment_id"] = segment_id
        rec.setdefault("recording_id", segment_id)
        rec.setdefault("epoch_label", rec.get("label") or rec.get("phase", ""))
        if not str(rec.get("phase", "")).strip() and str(rec.get("epoch_label", "")).strip():
            rec["phase"] = normalize_epoch_phase(rec.get("epoch_label"))
        index_by_key[key] = len(unique)
        unique.append(rec)
    return unique


def _read_raw_channel(path: Path, channel: str, cfg: SpikeLFPCouplingConfig, rec: Mapping) -> tuple[np.ndarray, np.ndarray, float, float]:
    header = pd.read_csv(path, nrows=0)
    time_col = find_time_column(list(header.columns), cfg.time_column_candidates)
    if channel not in header.columns:
        raise ValueError(f"LFP channel {channel!r} not found in {path.name}.")
    df = pd.read_csv(path, usecols=[time_col, channel])
    t_abs = df[time_col].to_numpy(dtype=float)
    info = validate_timestamps(t_abs, rec.get("sampling_rate_hz", ""))
    t0 = float(t_abs[0]) if t_abs.size else 0.0
    t_rel = t_abs - t0
    scale, _unit = _unit_scale_to_uv(rec.get("signal_unit", cfg.signal_unit), cfg.scale_to_uv, channel)
    x = df[channel].to_numpy(dtype=float) * scale
    return t_abs, t_rel, x, float(info["fs_est_hz"])


def _available_lfp_channels(path: Path, rec: Mapping, cfg: SpikeLFPCouplingConfig) -> list[str]:
    header = pd.read_csv(path, nrows=0)
    requested = cfg.lfp_channels or parse_channel_list(rec.get("channel"))
    if requested:
        return [channel for channel in requested if channel in header.columns]
    time_names = {name.lower() for name in cfg.time_column_candidates}
    channels = []
    for col in header.columns:
        lower = str(col).lower()
        if lower in time_names or "marker" in lower or "ttl" in lower or "digital" in lower:
            continue
        channels.append(str(col))
    return channels


def _select_record_spikes(spikes: pd.DataFrame, rec: Mapping, path: Path) -> pd.DataFrame:
    recording_id = str(rec.get("recording_id") or rec.get("segment_id") or path.stem)
    segment_id = str(rec.get("segment_id") or recording_id)
    candidates = spikes.copy()
    if "preparation_id" in candidates.columns and str(rec.get("preparation_id") or "").strip():
        candidates = candidates[candidates["preparation_id"].astype(str) == str(rec.get("preparation_id"))]
    if "segment_id" in candidates.columns:
        subset = candidates[candidates["segment_id"].astype(str) == segment_id]
        if not subset.empty:
            return subset.copy()
    if "recording_id" in candidates.columns:
        subset = candidates[candidates["recording_id"].astype(str) == recording_id]
        if not subset.empty:
            return subset.copy()
    if "recording_name" in candidates.columns:
        subset = candidates[candidates["recording_name"].astype(str).isin({recording_id, path.stem})]
        if not subset.empty:
            return subset.copy()
    if "source_file" in candidates.columns:
        source = candidates["source_file"].astype(str)
        subset = candidates[(source == str(path)) | (source.map(lambda value: Path(value).name) == path.name)]
        if not subset.empty:
            return subset.copy()
    return candidates.iloc[0:0].copy()


def _channel_pairs(lfp_channels: list[str], spike_channels: list[str], cfg: SpikeLFPCouplingConfig) -> list[tuple[str, str]]:
    lfp_set = set(map(str, lfp_channels))
    spike_set = set(map(str, spike_channels))
    if cfg.pairing_mode == "same_channel":
        return [(channel, channel) for channel in lfp_channels if str(channel) in spike_set]
    if cfg.pairing_mode == "explicit":
        pairs = [(str(spike), str(lfp)) for spike, lfp in cfg.explicit_channel_pairs]
        missing = [(spike, lfp) for spike, lfp in pairs if spike not in spike_set or lfp not in lfp_set]
        if missing:
            raise ValueError(f"Explicit spike-LFP channel pair(s) are unavailable: {missing}")
        return pairs
    return [(spike, lfp) for lfp in lfp_channels for spike in spike_channels]


def _load_valid_lfp_windows(cfg: SpikeLFPCouplingConfig) -> pd.DataFrame:
    if not cfg.lfp_feature_csv:
        return pd.DataFrame()
    path = Path(cfg.lfp_feature_csv)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    if {"signal_qc_pass", "feature_qc_pass"}.issubset(df.columns):
        valid = _as_bool(df["signal_qc_pass"]) & _as_bool(df["feature_qc_pass"])
    elif "qc_pass" in df.columns:
        valid = _as_bool(df["qc_pass"])
        if "signal_qc_pass" not in df.columns:
            df["signal_qc_pass"] = valid
        if "feature_qc_pass" not in df.columns:
            df["feature_qc_pass"] = valid
    else:
        valid = pd.Series(True, index=df.index)
        df["signal_qc_pass"] = True
        df["feature_qc_pass"] = True
    out = df.loc[valid].copy()
    if "channel" in out.columns and "lfp_channel" not in out.columns:
        out["lfp_channel"] = out["channel"].astype(str)
    if "usable_duration_s" in out.columns:
        out["valid_lfp_duration_s"] = pd.to_numeric(out["usable_duration_s"], errors="coerce")
    elif "window_duration_s" in out.columns:
        out["valid_lfp_duration_s"] = pd.to_numeric(out["window_duration_s"], errors="coerce")
    else:
        out["valid_lfp_duration_s"] = np.nan
    return out


def _windows_for_record_channel(valid_windows: pd.DataFrame, rec: Mapping, channel: str, fallback: pd.DataFrame) -> pd.DataFrame:
    if valid_windows.empty:
        return fallback.copy()
    df = valid_windows.copy()
    for rec_col, rec_value in (
        ("preparation_id", rec.get("preparation_id", "")),
        ("segment_id", rec.get("segment_id", rec.get("recording_id", ""))),
        ("recording_id", rec.get("recording_id", "")),
        ("source_file", rec.get("source_file", rec.get("path", ""))),
    ):
        if rec_col not in df.columns or not str(rec_value).strip():
            continue
        if rec_col == "source_file":
            value = str(rec_value)
            subset = df[(df[rec_col].astype(str) == value) | (df[rec_col].astype(str).map(lambda item: Path(item).name) == Path(value).name)]
        else:
            subset = df[df[rec_col].astype(str) == str(rec_value)]
        if not subset.empty:
            df = subset
    if "lfp_channel" in df.columns:
        df = df[df["lfp_channel"].astype(str) == str(channel)]
    if df.empty:
        return df
    cols = [
        "window_index",
        "window_start_s",
        "window_end_s",
        "window_mid_s",
        "cumulative_window_mid_s",
        "valid_lfp_duration_s",
        "signal_qc_pass",
        "signal_qc_reason",
        "feature_qc_pass",
        "feature_qc_reason",
        "qc_pass",
        "primary_power_uV2",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[cols].sort_values(["window_start_s", "window_index"], na_position="last").reset_index(drop=True)


def _continuous_runs(t: np.ndarray, x: np.ndarray) -> list[tuple[int, int]]:
    finite = np.isfinite(t) & np.isfinite(x)
    if finite.size < 2 or not finite.any():
        return []
    diffs = np.diff(t)
    good_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    median_dt = float(np.nanmedian(good_diffs)) if good_diffs.size else 0.0
    if median_dt <= 0:
        median_dt = 1.0
    good_step = np.r_[True, (diffs > 0) & (diffs <= median_dt * 1.5)]
    usable = finite & good_step
    runs = []
    start = None
    for idx, ok in enumerate(usable):
        if ok and start is None:
            start = idx
        if (not ok or idx == usable.size - 1) and start is not None:
            end = idx if not ok else idx + 1
            if end - start >= 4:
                runs.append((start, end))
            start = None
    return runs


def _overlap_duration(w_start: float, w_end: float, runs: list[dict]) -> float:
    total = 0.0
    for run in runs:
        start = max(float(w_start), float(run["t"][0]))
        end = min(float(w_end), float(run["t"][-1]))
        if end > start:
            total += end - start
    return float(total)


def _prepare_lfp_stream_for_coupling(
    t_rel: np.ndarray,
    x: np.ndarray,
    fs: float,
    cfg: SpikeLFPCouplingConfig,
    warnings_out: list[str],
):
    valid = np.isfinite(t_rel) & np.isfinite(x)
    return create_lfp_stream(t_rel, x, valid, fs, cfg.coupling_downsample_hz, warnings_out)


def _phase_runs_for_band(
    t_rel: np.ndarray,
    x: np.ndarray,
    fs: float,
    cfg: SpikeLFPCouplingConfig,
    band: tuple[float, float],
) -> list[dict]:
    runs = []
    for i0, i1 in _continuous_runs(t_rel, x):
        tr = t_rel[i0:i1]
        xr = x[i0:i1]
        if tr.size < max(16, int(math.ceil(3.0 * fs / max(band[0], EPS)))):
            continue
        phase = _bandpass_phase(xr, fs, band, cfg.filter_order, cfg.notch_frequency_hz, cfg.notch_q)
        if np.isfinite(phase).any():
            runs.append({"t": tr, "x": xr, "fs": fs, "phase": phase})
    return runs


def _spike_times_relative(ch_spikes: pd.DataFrame, t_abs: np.ndarray, t_rel: np.ndarray) -> np.ndarray:
    if ch_spikes.empty:
        return np.array([], dtype=float)
    if "segment_spike_time_s" in ch_spikes.columns:
        values = pd.to_numeric(ch_spikes["segment_spike_time_s"], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size:
            return np.sort(values)
    values = pd.to_numeric(ch_spikes["spike_time_s"], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return values
    rel_min, rel_max = float(np.nanmin(t_rel)), float(np.nanmax(t_rel))
    abs_min, abs_max = float(np.nanmin(t_abs)), float(np.nanmax(t_abs))
    in_rel = np.count_nonzero((values >= rel_min - EPS) & (values <= rel_max + EPS))
    in_abs = np.count_nonzero((values >= abs_min - EPS) & (values <= abs_max + EPS))
    if in_abs > in_rel and abs_min != rel_min:
        values = values - abs_min
    return np.sort(values)


def _phases_from_runs(runs: list[dict], spike_times: np.ndarray) -> np.ndarray:
    if spike_times.size == 0:
        return np.array([], dtype=float)
    out = np.full(spike_times.shape, np.nan, dtype=float)
    for run in runs:
        t = run["t"]
        mask = (spike_times >= float(t[0])) & (spike_times <= float(t[-1]))
        if mask.any():
            out[mask] = _phase_at_times(t, run["phase"], spike_times[mask])
    return out[np.isfinite(out)]


def _bandpass_signal_for_sta(
    x_uv: np.ndarray,
    fs: float,
    cfg: SpikeLFPCouplingConfig,
) -> np.ndarray:
    y = _interp_fill(np.asarray(x_uv, dtype=float))
    y = y - float(np.nanmedian(y))
    nyq = 0.5 * float(fs)
    if y.size < max(16, int(cfg.filter_order) * 8) or nyq <= 0:
        return np.full(y.shape, np.nan, dtype=float)
    if cfg.notch_frequency_hz and 0 < cfg.notch_frequency_hz < nyq * 0.98:
        b, a = signal.iirnotch(float(cfg.notch_frequency_hz) / nyq, Q=float(cfg.notch_q))
        y = signal.filtfilt(b, a, y)
    lo, hi = cfg.sta_lfp_band_hz
    lo = max(float(lo), 0.001)
    hi = min(float(hi), nyq * 0.95)
    if hi <= lo:
        return np.full(y.shape, np.nan, dtype=float)
    sos = signal.butter(int(cfg.filter_order), [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    try:
        return signal.sosfiltfilt(sos, y)
    except ValueError:
        return signal.sosfiltfilt(sos, y, padlen=max(0, min(y.size - 1, 3 * (2 * len(sos) + 1))))


def _sta_lfp_runs(
    t_rel: np.ndarray,
    x_uv: np.ndarray,
    fs: float,
    cfg: SpikeLFPCouplingConfig,
) -> list[dict]:
    runs = []
    half = int(round(float(cfg.sta_window_s) * float(fs)))
    min_samples = max(16, 2 * half + 1)
    for i0, i1 in _continuous_runs(t_rel, x_uv):
        tr = np.asarray(t_rel[i0:i1], dtype=float)
        xr = np.asarray(x_uv[i0:i1], dtype=float)
        if tr.size < min_samples:
            continue
        filtered = _bandpass_signal_for_sta(xr, fs, cfg)
        if np.isfinite(filtered).any():
            runs.append({"t": tr, "x": filtered, "fs": float(fs)})
    return runs


def _sta_snippets_from_runs(
    runs: list[dict],
    spike_times: np.ndarray,
    cfg: SpikeLFPCouplingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    times = np.sort(np.asarray(spike_times, dtype=float))
    times = times[np.isfinite(times)]
    total = int(times.size)
    if total == 0 or not runs:
        return np.empty((0, 0), dtype=float), np.array([], dtype=float), np.array([], dtype=float), total
    if times.size > cfg.sta_max_spikes_per_pair:
        keep = np.unique(np.linspace(0, times.size - 1, cfg.sta_max_spikes_per_pair).round().astype(int))
        times = times[keep]
    fs = float(runs[0]["fs"])
    half = int(round(float(cfg.sta_window_s) * fs))
    if half < 1:
        return np.empty((0, 0), dtype=float), np.array([], dtype=float), times, total
    rel_s = np.arange(-half, half + 1, dtype=float) / fs
    snippets = []
    used_times = []
    for spike_time in times:
        for run in runs:
            t = np.asarray(run["t"], dtype=float)
            if spike_time < t[0] or spike_time > t[-1]:
                continue
            idx = int(np.searchsorted(t, spike_time))
            if idx >= t.size:
                idx = t.size - 1
            if idx > 0 and abs(t[idx - 1] - spike_time) < abs(t[idx] - spike_time):
                idx -= 1
            if idx - half < 0 or idx + half + 1 > t.size:
                continue
            snippet = np.asarray(run["x"][idx - half : idx + half + 1], dtype=float)
            if np.isfinite(snippet).all():
                snippets.append(snippet)
                used_times.append(float(spike_time))
            break
    if not snippets:
        return np.empty((0, rel_s.size), dtype=float), rel_s, np.asarray(used_times, dtype=float), total
    return np.vstack(snippets), rel_s, np.asarray(used_times, dtype=float), total


def _write_spike_triggered_lfp_outputs(
    rec: Mapping,
    path: Path,
    lfp_channel: str,
    spike_channel: str,
    cached: Mapping,
    spike_times: np.ndarray,
    cfg: SpikeLFPCouplingConfig,
    out_dir: Path,
) -> dict | None:
    if cfg.sta_window_s <= 0 or cfg.sta_max_pairs <= 0:
        return None
    sta_dir = out_dir / "spike_triggered_lfp"
    sta_dir.mkdir(parents=True, exist_ok=True)
    runs = _sta_lfp_runs(
        np.asarray(cached["lfp_t_rel"], dtype=float),
        np.asarray(cached["x"], dtype=float),
        float(cached["fs"]),
        cfg,
    )
    snippets, rel_s, used_times, total_spikes = _sta_snippets_from_runs(runs, spike_times, cfg)
    if snippets.shape[0] < int(cfg.sta_min_spikes):
        return None
    mean = np.nanmean(snippets, axis=0)
    sem = np.nanstd(snippets, axis=0, ddof=1) / math.sqrt(snippets.shape[0]) if snippets.shape[0] > 1 else np.zeros_like(mean)
    segment_id = str(rec.get("segment_id", rec.get("recording_id", path.stem)))
    prep = str(rec.get("preparation_id", ""))
    pair = f"{spike_channel}->{lfp_channel}"
    safe = "_".join(_safe_name(bit) for bit in (prep, segment_id, pair) if str(bit).strip())
    if not safe:
        safe = _safe_name(pair)

    csv_path = sta_dir / f"spike_lfp_sta_{safe}.csv"
    pd.DataFrame(
        {
            "time_rel_s": rel_s,
            "time_rel_ms": rel_s * 1000.0,
            "lfp_sta_uv": mean,
            "lfp_sta_sem_uv": sem,
            "n_spikes_used": int(snippets.shape[0]),
        }
    ).to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(rel_s * 1000.0, mean, color="tab:green", linewidth=2.0)
    ax.fill_between(rel_s * 1000.0, mean - sem, mean + sem, color="tab:green", alpha=0.22)
    ax.axvline(0.0, color="black", alpha=0.45, linewidth=1)
    ax.set_xlabel("Time from spike (ms)")
    ax.set_ylabel("LFP (uV)")
    ax.set_title(f"Spike-triggered average LFP | {segment_id} | {pair} (n={snippets.shape[0]})")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    plot_path = sta_dir / f"spike_lfp_sta_{safe}.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    spectro_path = ""
    if bool(cfg.sta_spectrogram):
        fs = float(cached["fs"])
        nperseg = min(snippets.shape[1], max(16, int(round(0.05 * fs))))
        noverlap = max(0, int(round(nperseg * 0.5)))
        specs = []
        f_spec = None
        t_spec = None
        for snippet in snippets:
            try:
                f_tmp, t_tmp, sxx = signal.spectrogram(
                    snippet,
                    fs=fs,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    scaling="density",
                    mode="psd",
                )
            except ValueError:
                continue
            keep = f_tmp <= float(cfg.sta_spectrogram_max_frequency_hz)
            if not np.any(keep):
                continue
            f_tmp = f_tmp[keep]
            sxx = sxx[keep, :]
            if f_spec is None:
                f_spec = f_tmp
                t_spec = t_tmp
            elif not (np.array_equal(f_spec, f_tmp) and np.array_equal(t_spec, t_tmp)):
                continue
            specs.append(10.0 * np.log10(sxx + EPS))
        if specs and f_spec is not None and t_spec is not None:
            smean = np.nanmean(np.stack(specs, axis=0), axis=0)
            fig, ax = plt.subplots(figsize=(7.4, 4.8))
            t_ms = (t_spec - float(cfg.sta_window_s)) * 1000.0
            mesh = ax.pcolormesh(t_ms, f_spec, smean, shading="auto")
            ax.axvline(0.0, color="white", alpha=0.7, linewidth=1)
            ax.set_xlabel("Time from spike (ms)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title(f"Spike-triggered LFP spectrogram | {segment_id} | {pair}")
            fig.colorbar(mesh, ax=ax, label="PSD (dB uV^2/Hz)")
            fig.tight_layout()
            spectro = sta_dir / f"spike_lfp_sta_spectrogram_{safe}.png"
            fig.savefig(spectro, dpi=200)
            plt.close(fig)
            spectro_path = str(spectro)

    same_channel = bool(str(spike_channel) == str(lfp_channel))
    return {
        "experiment_id": rec.get("experiment_id", ""),
        "animal_id": rec.get("animal_id", ""),
        "preparation_id": rec.get("preparation_id", ""),
        "segment_id": segment_id,
        "recording_id": rec.get("recording_id", segment_id),
        "recording_order": rec.get("recording_order", ""),
        "source_file": str(path),
        "epoch_label": rec.get("epoch_label", rec.get("phase", "")),
        "phase": rec.get("phase", ""),
        "lfp_channel": lfp_channel,
        "spike_channel": spike_channel,
        "channel_pair": pair,
        "pairing_mode": cfg.pairing_mode,
        "same_channel": same_channel,
        "sta_lfp_band_lo_hz": float(cfg.sta_lfp_band_hz[0]),
        "sta_lfp_band_hi_hz": float(min(cfg.sta_lfp_band_hz[1], 0.5 * float(cached["fs"]) * 0.95)),
        "sta_window_s": float(cfg.sta_window_s),
        "spike_count_total": int(total_spikes),
        "spike_count_sampled": int(min(total_spikes, cfg.sta_max_spikes_per_pair)),
        "spike_count_used": int(snippets.shape[0]),
        "source_sampling_rate_hz": float(cached.get("source_fs", np.nan)),
        "lfp_sampling_rate_hz": float(cached["fs"]),
        "sta_csv": str(csv_path),
        "sta_plot": str(plot_path),
        "sta_spectrogram_plot": spectro_path,
        "interpretation_warning": COUPLING_WARNING,
        "same_channel_warning": SAME_CHANNEL_WARNING if same_channel else "",
    }


def _surrogate_ppc(
    runs: list[dict],
    spike_times: np.ndarray,
    start_s: float,
    end_s: float,
    cfg: SpikeLFPCouplingConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    count = int(cfg.surrogate_count)
    if count <= 0 or spike_times.size < 2:
        return np.array([], dtype=float)
    duration = max(float(end_s - start_s), EPS)
    out = []
    done = 0
    while done < count:
        n = min(cfg.surrogate_chunk_size, count - done)
        shifts = rng.uniform(duration * 0.10, duration * 0.90, size=n)
        shifted = start_s + np.mod(spike_times[None, :] - start_s + shifts[:, None], duration)
        rows = []
        for row in shifted:
            rows.append(_phases_from_runs(runs, row))
        max_len = max((row.size for row in rows), default=0)
        if max_len < 2:
            out.extend([np.nan] * n)
        else:
            arr = np.full((n, max_len), np.nan, dtype=float)
            for idx, row in enumerate(rows):
                arr[idx, : row.size] = row
            out.extend(_ppc_many(arr).tolist())
        done += n
    return np.asarray(out, dtype=float)


def _sfc_for_window(
    run: dict,
    start_s: float,
    end_s: float,
    spike_times: np.ndarray,
    cfg: SpikeLFPCouplingConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(run["t"], dtype=float)
    x = np.asarray(run["x"], dtype=float)
    fs = float(run["fs"])
    in_win = (t >= start_s) & (t < end_s)
    if np.count_nonzero(in_win) < 16 or spike_times.size < 2:
        return np.array([]), np.array([])
    tw = t[in_win]
    xw = x[in_win]
    train = np.zeros_like(xw, dtype=float)
    local_spikes = spike_times[(spike_times >= float(tw[0])) & (spike_times <= float(tw[-1]))]
    idx = np.searchsorted(tw, local_spikes)
    idx = idx[(idx >= 0) & (idx < train.size)]
    idx = np.unique(idx)
    if idx.size < 2:
        return np.array([]), np.array([])
    if cfg.rate_match_sfc and idx.size > cfg.min_spikes:
        idx = np.sort(rng.choice(idx, size=cfg.min_spikes, replace=False))
    train[idx] = 1.0
    nperseg = min(xw.size, max(32, int(round(4.0 * fs))))
    noverlap = max(0, int(nperseg * 0.5))
    try:
        f, coh = signal.coherence(train, _interp_fill(xw), fs=fs, nperseg=nperseg, noverlap=noverlap)
    except ValueError:
        return np.array([]), np.array([])
    keep = f <= float(cfg.max_sfc_frequency_hz)
    f, coh = f[keep], coh[keep]
    if f.size > cfg.max_sfc_frequencies:
        idx_keep = np.unique(np.linspace(0, f.size - 1, cfg.max_sfc_frequencies).round().astype(int))
        f, coh = f[idx_keep], coh[idx_keep]
    return f, np.clip(coh, 0.0, 1.0)


class _CsvRowWriter:
    def __init__(self, path: Path, columns: list[str], flush_rows: int):
        self.path = path
        self.columns = columns
        self.flush_rows = int(flush_rows)
        self.rows: list[dict] = []
        self.wrote_header = False
        if path.exists():
            path.unlink()

    def add(self, row: dict) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        df = pd.DataFrame(self.rows)
        for col in self.columns:
            if col not in df.columns:
                df[col] = np.nan
        df[self.columns].to_csv(self.path, mode="a", header=not self.wrote_header, index=False)
        self.wrote_header = True
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        if not self.wrote_header:
            pd.DataFrame(columns=self.columns).to_csv(self.path, index=False)


def _base_row(rec: Mapping, rec_index: int, path: Path, lfp_channel: str, spike_channel: str, cfg: SpikeLFPCouplingConfig, window: Mapping, band: tuple[str, str, tuple[float, float]]) -> dict:
    band_key, band_label, (lo, hi) = band
    start_s = float(window["window_start_s"])
    end_s = float(window["window_end_s"])
    mid_s = float(window.get("window_mid_s", (start_s + end_s) / 2.0))
    cumulative_mid = window.get("cumulative_window_mid_s", np.nan)
    same_channel = bool(str(spike_channel) == str(lfp_channel))
    high_frequency_same_channel = bool(same_channel and float(hi) >= 40.0)
    same_channel_warning = SAME_CHANNEL_WARNING if same_channel else ""
    if high_frequency_same_channel:
        same_channel_warning = (
            same_channel_warning
            + " High-frequency same-channel coupling is flagged because residual spike-waveform leakage is most likely to affect faster bands."
        ).strip()
    return {
        "experiment_id": rec.get("experiment_id", ""),
        "animal_id": rec.get("animal_id", ""),
        "preparation_id": rec.get("preparation_id", ""),
        "segment_id": rec.get("segment_id", rec.get("recording_id", path.stem)),
        "recording_id": rec.get("recording_id", path.stem),
        "recording_order": rec.get("recording_order", rec_index),
        "source_file": str(path),
        "epoch_label": window.get("epoch_label", rec.get("epoch_label", rec.get("phase", ""))),
        "phase": window.get("phase", rec.get("phase", "")),
        "treatment": window.get("treatment", rec.get("treatment", "")),
        "concentration_value": window.get("concentration_value", rec.get("concentration_value", "")),
        "concentration_unit": window.get("concentration_unit", rec.get("concentration_unit", "")),
        "lfp_channel": lfp_channel,
        "spike_channel": spike_channel,
        "channel_pair": f"{spike_channel}->{lfp_channel}",
        "pairing_mode": cfg.pairing_mode,
        "same_channel": same_channel,
        "band_key": band_key,
        "band_label": band_label,
        "phase_band_lo_hz": float(lo),
        "phase_band_hi_hz": float(hi),
        "window_index": int(float(window.get("window_index", 0))) if pd.notna(window.get("window_index", np.nan)) else 0,
        "window_start_s": start_s,
        "window_end_s": end_s,
        "window_mid_s": mid_s,
        "cumulative_window_mid_s": float(cumulative_mid) if pd.notna(cumulative_mid) else mid_s,
        "valid_lfp_duration_s": float(window.get("valid_lfp_duration_s", end_s - start_s)),
        "primary_power_uV2": window.get("primary_power_uV2", np.nan),
        "signal_qc_pass": bool(_as_bool(pd.Series([window.get("signal_qc_pass", True)])).iloc[0]),
        "signal_qc_reason": window.get("signal_qc_reason", ""),
        "feature_qc_pass": bool(_as_bool(pd.Series([window.get("feature_qc_pass", True)])).iloc[0]),
        "feature_qc_reason": window.get("feature_qc_reason", ""),
        "coupling_label": "MUA-LFP phase coupling",
        "interpretation_warning": COUPLING_WARNING,
        "same_channel_warning": same_channel_warning,
        "high_frequency_same_channel_flag": high_frequency_same_channel,
        "h2o2_safety_warning": H2O2_WARNING,
    }


def _add_ppc_baseline_changes(ppc: pd.DataFrame) -> pd.DataFrame:
    if ppc.empty or "ppc" not in ppc.columns:
        return ppc
    df = ppc.copy()
    phase_norm = df.get("phase", df.get("epoch_label", "")).astype(str).str.strip().str.lower()
    baseline = df[(phase_norm == "baseline") & _as_bool(df.get("qc_pass", pd.Series(True, index=df.index)))].copy()
    keys = [col for col in ("experiment_id", "animal_id", "preparation_id", "lfp_channel", "spike_channel", "band_key", "window_index") if col in df.columns]
    if baseline.empty or not keys:
        df["baseline_median_ppc"] = np.nan
        df["ppc_change_from_baseline"] = np.nan
        return df
    refs = baseline.groupby(keys, dropna=False)["ppc"].median().reset_index().rename(columns={"ppc": "baseline_median_ppc"})
    if "baseline_median_ppc" in df.columns:
        df = df.drop(columns=["baseline_median_ppc"])
    df = df.merge(refs, on=keys, how="left")
    df["ppc_change_from_baseline"] = pd.to_numeric(df["ppc"], errors="coerce") - pd.to_numeric(df["baseline_median_ppc"], errors="coerce")
    return df


def _add_surrogate_fdr(ppc: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    if ppc.empty or "surrogate_p_value" not in ppc.columns:
        return ppc
    df = ppc.copy()
    p = pd.to_numeric(df["surrogate_p_value"], errors="coerce")
    eligible = p.notna() & _as_bool(df.get("qc_pass", pd.Series(True, index=df.index)))
    q = pd.Series(np.nan, index=df.index, dtype=float)
    if eligible.any():
        pvals = p[eligible].to_numpy(dtype=float)
        order = np.argsort(pvals)
        ranked = pvals[order]
        m = float(ranked.size)
        raw_q = ranked * m / np.arange(1, ranked.size + 1, dtype=float)
        adjusted = np.minimum.accumulate(raw_q[::-1])[::-1]
        adjusted = np.clip(adjusted, 0.0, 1.0)
        idx = p[eligible].index.to_numpy()[order]
        q.loc[idx] = adjusted
    df["surrogate_q_value"] = q
    df["surrogate_significant_fdr_0p05"] = q <= float(alpha)
    return df


def _add_sfc_baseline_changes(sfc_path: Path) -> pd.DataFrame:
    if not sfc_path.exists() or sfc_path.stat().st_size == 0:
        return pd.DataFrame(columns=SFC_COLUMNS)
    df = pd.read_csv(sfc_path)
    if df.empty or "sfc" not in df.columns:
        return df
    phase_norm = df.get("phase", df.get("epoch_label", "")).astype(str).str.strip().str.lower()
    baseline = df[(phase_norm == "baseline") & _as_bool(df.get("qc_pass", pd.Series(True, index=df.index)))].copy()
    keys = [col for col in ("experiment_id", "animal_id", "preparation_id", "lfp_channel", "spike_channel", "frequency_hz", "window_index") if col in df.columns]
    if baseline.empty or not keys:
        df["baseline_median_sfc"] = np.nan
        df["sfc_change_from_baseline"] = np.nan
        df.to_csv(sfc_path, index=False)
        return df
    refs = baseline.groupby(keys, dropna=False)["sfc"].median().reset_index().rename(columns={"sfc": "baseline_median_sfc"})
    if "baseline_median_sfc" in df.columns:
        df = df.drop(columns=["baseline_median_sfc"])
    df = df.merge(refs, on=keys, how="left")
    df["sfc_change_from_baseline"] = pd.to_numeric(df["sfc"], errors="coerce") - pd.to_numeric(df["baseline_median_sfc"], errors="coerce")
    for col in SFC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df[SFC_COLUMNS].to_csv(sfc_path, index=False)
    return df


def preflight_coupling_workload(spike_events, lfp_data, config: SpikeLFPCouplingConfig | Mapping | None = None) -> dict:
    cfg = as_coupling_config(config)
    spikes = pd.read_csv(spike_events) if isinstance(spike_events, (str, Path)) else pd.DataFrame(spike_events)
    records = _unique_segment_records(_load_lfp_records(lfp_data, cfg))
    valid_windows = _load_valid_lfp_windows(cfg)
    bands = _phase_band_defs(cfg)
    total_pairs = 0
    total_windows = 0
    total_pair_windows = 0
    lfp_channel_names: set[str] = set()
    spike_channel_names: set[str] = set()
    preparations: set[str] = set()
    rows = []
    for rec in records:
        path = Path(str(rec.get("source_file") or rec.get("path") or ""))
        if not path.exists():
            continue
        preparations.add(str(rec.get("preparation_id") or ""))
        rec_spikes = _select_record_spikes(spikes, rec, path)
        if rec_spikes.empty or "channel" not in rec_spikes.columns:
            continue
        spike_channels = cfg.spike_channels or sorted(rec_spikes["channel"].dropna().astype(str).unique())
        lfp_channels = _available_lfp_channels(path, rec, cfg)
        pairs = _channel_pairs(lfp_channels, spike_channels, cfg)
        t_abs, t_rel, x0, _fs = _read_raw_channel(path, lfp_channels[0], cfg, rec) if lfp_channels else (np.array([]), np.array([]), np.array([]), 0.0)
        fallback = _window_bounds_from_time(t_rel, cfg.window_s)
        windows_this_segment = 0
        for channel in lfp_channels:
            lfp_channel_names.add(channel)
            windows_this_segment = max(windows_this_segment, int(len(_windows_for_record_channel(valid_windows, rec, channel, fallback))))
        spike_channel_names.update(spike_channels)
        total_pairs += len(pairs)
        total_windows += windows_this_segment
        total_pair_windows += windows_this_segment * max(1, len(pairs))
        rows.append(
            {
                "segment_id": rec.get("segment_id", path.stem),
                "lfp_channels": len(lfp_channels),
                "spike_channels": len(spike_channels),
                "channel_pairs": len(pairs),
                "windows_per_pair": windows_this_segment,
            }
        )
    estimated_ppc = int(total_pair_windows * len(bands) * (cfg.surrogate_count + 1))
    estimated_sfc_rows = 0 if cfg.processing_mode == "simple" else int(total_pair_windows * len(bands) * min(cfg.max_sfc_frequencies, 128))
    return {
        "preparations": len({p for p in preparations if p.strip()}) or len(preparations),
        "segments": len(records),
        "lfp_channels": len(lfp_channel_names),
        "spike_channels": len(spike_channel_names),
        "channel_pairs": total_pairs,
        "windows": total_windows,
        "pair_windows": total_pair_windows,
        "phase_bands": len(bands),
        "surrogates": cfg.surrogate_count,
        "processing_mode": cfg.processing_mode,
        "estimated_ppc_computations": estimated_ppc,
        "estimated_sfc_rows": estimated_sfc_rows,
        "pairing_mode": cfg.pairing_mode,
        "segment_rows": rows,
    }


def _check_cancel(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Spike-LFP coupling cancelled.")


def calculate_spike_lfp_coupling(
    spike_events,
    lfp_data,
    config: SpikeLFPCouplingConfig | Mapping | None = None,
    progress_callback: Callable[[str], None] | None = None,
    cancel_event=None,
) -> dict[str, str]:
    """Calculate MUA-LFP PPC and conventional SFC from exported spike events."""
    cfg = as_coupling_config(config)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ppc_path = out_dir / "spike_lfp_ppc_by_window.csv"
    sfc_path = out_dir / "spike_field_coherence.csv"
    paths = {
        "ppc_csv": str(ppc_path),
        "sfc_csv": str(sfc_path),
        "rate_power_csv": str(out_dir / "spike_lfp_rate_power_by_window.csv"),
        "rate_power_summary_csv": str(out_dir / "spike_lfp_rate_power_summary.csv"),
        "summary_csv": str(out_dir / "spike_lfp_phase_summary.csv"),
        "ppc_change_csv": str(out_dir / "spike_lfp_ppc_change_from_baseline.csv"),
        "sfc_change_csv": str(out_dir / "spike_lfp_sfc_change_from_baseline.csv"),
        "sta_summary_csv": str(out_dir / "spike_lfp_sta_summary.csv"),
        "qc_summary_csv": str(out_dir / "spike_lfp_coupling_qc_summary.csv"),
        "workload_json": str(out_dir / "spike_lfp_coupling_workload.json"),
        "parameters_json": str(out_dir / "spike_lfp_coupling_parameters.json"),
    }
    progress = progress_callback or (lambda message: print(message))
    spikes = pd.read_csv(spike_events) if isinstance(spike_events, (str, Path)) else pd.DataFrame(spike_events)
    if spikes.empty:
        raise ValueError("Spike-event input is empty. Export spike_events.csv from spike analysis first.")
    if "spike_time_s" not in spikes.columns:
        raise ValueError("Spike events must contain spike_time_s; spike-count windows alone are insufficient.")
    records = _unique_segment_records(_load_lfp_records(lfp_data, cfg))
    if not records:
        raise FileNotFoundError("No LFP raw recordings were provided.")
    bands = _phase_band_defs(cfg)
    workload = preflight_coupling_workload(spikes, records, cfg)
    Path(paths["workload_json"]).write_text(json.dumps(_json_ready(workload), indent=2), encoding="utf-8")
    progress(
        "[spike-lfp workload] "
        f"mode={cfg.processing_mode}, "
        f"preparations={workload['preparations']}, segments={workload['segments']}, "
        f"LFP channels={workload['lfp_channels']}, spike channels={workload['spike_channels']}, "
        f"pairs={workload['channel_pairs']}, windows={workload['windows']}, bands={workload['phase_bands']}, "
        f"surrogates={workload['surrogates']}, estimated PPC computations={workload['estimated_ppc_computations']}, "
        f"estimated SFC rows={workload['estimated_sfc_rows']}."
    )
    if cfg.pairing_mode == "all_to_all" and workload["estimated_ppc_computations"] > cfg.max_projected_ppc and not cfg.allow_large_all_to_all:
        raise RuntimeError(
            "Projected all-to-all spike-LFP coupling workload exceeds the configured limit "
            f"({workload['estimated_ppc_computations']} > {cfg.max_projected_ppc}). "
            "Use same_channel, explicit pairs, or enable the all-to-all override."
        )
    valid_windows = _load_valid_lfp_windows(cfg)
    if cfg.lfp_feature_csv and valid_windows.empty:
        progress("[spike-lfp qc] No valid LFP feature/QC windows matched; coupling outputs will be empty.")
    ppc_rows: list[dict] = []
    sta_rows: list[dict] = []
    phase_samples_for_plot: list[dict] = []
    phase_sample_count = 0
    max_phase_plot_samples = 200_000
    sta_pairs_written = 0
    sta_pairs_attempted = 0
    sfc_writer = _CsvRowWriter(sfc_path, SFC_COLUMNS, cfg.sfc_flush_rows)
    rng = np.random.default_rng(cfg.random_seed)
    cumulative_offsets_by_preparation: dict[str, float] = {}
    warnings_out: list[str] = []
    lfp_stream_metadata: list[dict] = []

    for rec_index, rec in enumerate(records, start=1):
        _check_cancel(cancel_event)
        path = Path(str(rec.get("source_file") or rec.get("path") or ""))
        if not path.exists():
            progress(f"[spike-lfp skip] Missing raw LFP CSV: {path}")
            continue
        prep_key = str(rec.get("preparation_id") or "__unknown_preparation__")
        cumulative_offset = float(rec.get("cumulative_offset_s", cumulative_offsets_by_preparation.get(prep_key, 0.0)) or 0.0)
        rec_spikes = _select_record_spikes(spikes, rec, path)
        if rec_spikes.empty:
            progress(f"[spike-lfp skip] No exported spike events matched {path.name}.")
            continue
        lfp_channels = _available_lfp_channels(path, rec, cfg)
        spike_channels = cfg.spike_channels or sorted(rec_spikes["channel"].dropna().astype(str).unique())
        pairs = _channel_pairs(lfp_channels, spike_channels, cfg)
        if not pairs:
            progress(f"[spike-lfp skip] No channel pairs for {path.name} with pairing_mode={cfg.pairing_mode}.")
            continue
        progress(f"[spike-lfp segment] {rec.get('segment_id', path.stem)}: {len(pairs)} channel pair(s).")
        channel_cache: dict[str, dict] = {}
        recording_duration_s = 0.0
        for lfp_channel in lfp_channels:
            _check_cancel(cancel_event)
            if lfp_channel not in {lfp for _spike, lfp in pairs}:
                continue
            t_abs, t_rel, x, fs = _read_raw_channel(path, lfp_channel, cfg, rec)
            if t_rel.size > 1:
                recording_duration_s = max(recording_duration_s, float(t_rel[-1] - t_rel[0] + np.nanmedian(np.diff(t_rel))))
            lfp_stream = _prepare_lfp_stream_for_coupling(t_rel, x, fs, cfg, warnings_out)
            stream_x = np.asarray(lfp_stream.x_uv, dtype=float).copy()
            stream_x[~np.asarray(lfp_stream.valid_mask, dtype=bool)] = np.nan
            fallback_windows = _window_bounds_from_time(lfp_stream.t_rel_s, cfg.window_s)
            windows = _windows_for_record_channel(valid_windows, rec, lfp_channel, fallback_windows)
            if windows.empty:
                progress(f"[spike-lfp qc] {path.name} {lfp_channel}: no valid LFP QC windows.")
                continue
            lfp_stream_metadata.append(
                {
                    "preparation_id": rec.get("preparation_id", ""),
                    "segment_id": rec.get("segment_id", rec.get("recording_id", path.stem)),
                    "recording_id": rec.get("recording_id", path.stem),
                    "recording_order": rec.get("recording_order", rec_index),
                    "epoch_label": rec.get("epoch_label", rec.get("phase", "")),
                    "phase": rec.get("phase", ""),
                    "source_file": str(path),
                    "lfp_channel": lfp_channel,
                    "source_sampling_rate_hz": fs,
                    "lfp_sampling_rate_hz": lfp_stream.fs_hz,
                    "stream_metadata": lfp_stream.metadata,
                }
            )
            channel_cache[lfp_channel] = {
                "t_abs": t_abs,
                "t_rel": t_rel,
                "lfp_t_rel": lfp_stream.t_rel_s,
                "x": stream_x,
                "fs": lfp_stream.fs_hz,
                "source_fs": fs,
                "windows": windows,
                "phase_runs": {},
            }
            progress(f"[spike-lfp channel] {lfp_channel}: {len(windows)} valid window(s), source fs={fs:.2f} Hz, LFP fs={lfp_stream.fs_hz:.2f} Hz.")
            for band in bands:
                _check_cancel(cancel_event)
                band_key, band_label, band_range = band
                runs = _phase_runs_for_band(lfp_stream.t_rel_s, stream_x, lfp_stream.fs_hz, cfg, band_range)
                channel_cache[lfp_channel]["phase_runs"][band_key] = runs
                progress(f"[spike-lfp band] {lfp_channel} {band_label}: cached {len(runs)} contiguous LFP run(s).")
        for spike_channel, lfp_channel in pairs:
            _check_cancel(cancel_event)
            if lfp_channel not in channel_cache:
                continue
            cached = channel_cache[lfp_channel]
            ch_spikes = rec_spikes[rec_spikes["channel"].astype(str) == str(spike_channel)]
            spike_times_all = _spike_times_relative(ch_spikes, cached["t_abs"], cached["t_rel"])
            if cfg.processing_mode == "complex" and sta_pairs_attempted < cfg.sta_max_pairs:
                sta_pairs_attempted += 1
                sta_row = _write_spike_triggered_lfp_outputs(
                    rec,
                    path,
                    lfp_channel,
                    spike_channel,
                    cached,
                    spike_times_all,
                    cfg,
                    out_dir,
                )
                if sta_row is not None:
                    sta_rows.append(sta_row)
                    sta_pairs_written += 1
            for band in bands:
                band_key = band[0]
                phase_runs = cached["phase_runs"].get(band_key, [])
                if not phase_runs:
                    continue
                for window_counter, window in enumerate(cached["windows"].to_dict(orient="records"), start=1):
                    _check_cancel(cancel_event)
                    start_s = float(window["window_start_s"])
                    end_s = float(window["window_end_s"])
                    valid_duration = float(window.get("valid_lfp_duration_s", np.nan))
                    run_duration = _overlap_duration(start_s, end_s, phase_runs)
                    if not np.isfinite(valid_duration):
                        valid_duration = run_duration
                    else:
                        valid_duration = min(valid_duration, run_duration) if run_duration > 0 else valid_duration
                    spike_times = spike_times_all[(spike_times_all >= start_s) & (spike_times_all < end_s)]
                    n_spikes = int(spike_times.size)
                    base = _base_row(rec, rec_index, path, lfp_channel, spike_channel, cfg, window, band)
                    base["cumulative_window_mid_s"] = cumulative_offset + float(base["window_mid_s"])
                    base["valid_lfp_duration_s"] = valid_duration
                    valid_cycles = float(max(0.0, run_duration) * float(band[2][0]))
                    base["valid_oscillatory_cycles"] = valid_cycles
                    base["cycle_qc_pass"] = bool(valid_cycles >= cfg.min_oscillatory_cycles)
                    qc_reason = []
                    if valid_duration <= 0:
                        qc_reason.append("no_contiguous_valid_lfp")
                    if valid_cycles < cfg.min_oscillatory_cycles:
                        qc_reason.append(f"oscillatory_cycles<{cfg.min_oscillatory_cycles:g}")
                    if n_spikes < cfg.min_spikes:
                        qc_reason.append(f"spike_count<{cfg.min_spikes}")
                    if qc_reason:
                        ppc_rows.append(
                            {
                                **base,
                                "spike_count": n_spikes,
                                "ppc": np.nan,
                                "preferred_phase_rad": np.nan,
                                "preferred_phase_deg": np.nan,
                                "surrogate_mean": np.nan,
                                "surrogate_ci_low": np.nan,
                                "surrogate_ci_high": np.nan,
                                "surrogate_p_value": np.nan,
                                "coupling_qc_pass": False,
                                "coupling_qc_reason": ";".join(qc_reason),
                                "qc_pass": False,
                                "qc_reason": ";".join(qc_reason),
                                "baseline_median_ppc": np.nan,
                                "ppc_change_from_baseline": np.nan,
                            }
                        )
                        continue
                    spike_phase = _phases_from_runs(phase_runs, spike_times)
                    ppc = pairwise_phase_consistency(spike_phase)
                    preferred = np.angle(np.nanmean(np.exp(1j * spike_phase))) if spike_phase.size else np.nan
                    sur = _surrogate_ppc(phase_runs, spike_times, start_s, end_s, cfg, rng)
                    p_value = float((np.sum(sur >= ppc) + 1) / (np.sum(np.isfinite(sur)) + 1)) if np.isfinite(ppc) and np.isfinite(sur).any() else np.nan
                    if spike_phase.size and phase_sample_count < max_phase_plot_samples:
                        remaining = max_phase_plot_samples - phase_sample_count
                        sampled_phase = spike_phase[:remaining]
                        phase_samples_for_plot.append(
                            {
                                "band_key": band_key,
                                "band_label": band[1],
                                "epoch_label": base.get("epoch_label", ""),
                                "phase": base.get("phase", ""),
                                "channel_pair": base.get("channel_pair", ""),
                                "phase_rad": sampled_phase,
                            }
                        )
                        phase_sample_count += int(sampled_phase.size)
                    ppc_rows.append(
                        {
                            **base,
                            "spike_count": n_spikes,
                            "ppc": ppc,
                            "preferred_phase_rad": float(preferred) if np.isfinite(preferred) else np.nan,
                            "preferred_phase_deg": float(np.degrees(preferred) % 360.0) if np.isfinite(preferred) else np.nan,
                            "surrogate_mean": float(np.nanmean(sur)) if sur.size else np.nan,
                            "surrogate_ci_low": float(np.nanpercentile(sur, 2.5)) if sur.size and np.isfinite(sur).any() else np.nan,
                            "surrogate_ci_high": float(np.nanpercentile(sur, 97.5)) if sur.size and np.isfinite(sur).any() else np.nan,
                            "surrogate_p_value": p_value,
                            "coupling_qc_pass": bool(np.isfinite(ppc)),
                            "coupling_qc_reason": "pass" if np.isfinite(ppc) else "ppc_not_finite",
                            "qc_pass": bool(np.isfinite(ppc)),
                            "qc_reason": "pass" if np.isfinite(ppc) else "ppc_not_finite",
                            "baseline_median_ppc": np.nan,
                            "ppc_change_from_baseline": np.nan,
                        }
                    )
                    if cfg.processing_mode == "complex":
                        for run in phase_runs:
                            f, coh = _sfc_for_window(run, start_s, end_s, spike_times, cfg, rng)
                            for freq, value in zip(f, coh):
                                sfc_writer.add(
                                    {
                                        **base,
                                        "spike_count": n_spikes,
                                        "frequency_hz": float(freq),
                                        "sfc": float(value),
                                        "baseline_median_sfc": np.nan,
                                        "sfc_change_from_baseline": np.nan,
                                        "rate_matched_sfc": bool(cfg.rate_match_sfc),
                                        "coupling_qc_pass": bool(np.isfinite(value)),
                                        "qc_pass": bool(np.isfinite(value)),
                                    }
                                )
                    if window_counter == 1 or window_counter % 10 == 0 or window_counter == len(cached["windows"]):
                        progress(f"[spike-lfp window] {spike_channel}->{lfp_channel} {band[1]}: {window_counter}/{len(cached['windows'])}")
        if "cumulative_offset_s" not in rec:
            cumulative_offsets_by_preparation[prep_key] = cumulative_offset + recording_duration_s

    sfc_writer.close()
    ppc = pd.DataFrame(ppc_rows)
    for col in PPC_COLUMNS:
        if col not in ppc.columns:
            ppc[col] = np.nan
    ppc = _add_surrogate_fdr(ppc[PPC_COLUMNS])
    ppc = _add_ppc_baseline_changes(ppc[PPC_COLUMNS])
    ppc.to_csv(paths["ppc_csv"], index=False)
    sfc = _add_sfc_baseline_changes(sfc_path)
    ppc_change = ppc[ppc["ppc_change_from_baseline"].notna()].copy() if "ppc_change_from_baseline" in ppc.columns else pd.DataFrame()
    sfc_change = sfc[sfc["sfc_change_from_baseline"].notna()].copy() if not sfc.empty and "sfc_change_from_baseline" in sfc.columns else pd.DataFrame()
    ppc_change.to_csv(paths["ppc_change_csv"], index=False)
    sfc_change.to_csv(paths["sfc_change_csv"], index=False)
    rate_power = _rate_power_from_ppc(ppc)
    rate_power_summary = _rate_power_summary(rate_power)
    rate_power.to_csv(paths["rate_power_csv"], index=False)
    rate_power_summary.to_csv(paths["rate_power_summary_csv"], index=False)
    summary = _phase_summary(ppc)
    qc_summary = _coupling_qc_summary(ppc)
    sta_summary = pd.DataFrame(sta_rows)
    for col in STA_COLUMNS:
        if col not in sta_summary.columns:
            sta_summary[col] = np.nan
    summary.to_csv(paths["summary_csv"], index=False)
    qc_summary.to_csv(paths["qc_summary_csv"], index=False)
    sta_summary[STA_COLUMNS].to_csv(paths["sta_summary_csv"], index=False)
    _plot_spike_rate_vs_lfp_power(rate_power, rate_power_summary, out_dir)
    _plot_outputs(ppc, sfc, out_dir, include_complex=cfg.processing_mode == "complex", phase_samples=phase_samples_for_plot)
    primary_rate_power_plot = out_dir / "PRIMARY_spike_rate_vs_lfp_power.png"
    primary_ppc_plot = out_dir / "PRIMARY_spike_lfp_ppc_phase_coupling.png"
    if primary_rate_power_plot.exists():
        paths["primary_rate_power_plot"] = str(primary_rate_power_plot)
    if primary_ppc_plot.exists():
        paths["primary_ppc_plot"] = str(primary_ppc_plot)
    params = {
        "config": _json_ready(asdict(cfg)),
        "outputs": paths,
        "workload": workload,
        "warnings": list(dict.fromkeys(warnings_out)),
        "lfp_stream_preprocessing": _json_ready(lfp_stream_metadata),
        "spike_processing_provenance": _json_ready(cfg.spike_processing_provenance or {}),
        "interpretation_warning": COUPLING_WARNING,
        "same_channel_warning": SAME_CHANNEL_WARNING,
        "h2o2_safety_warning": H2O2_WARNING,
        "schemas": {
            "spike_lfp_ppc_by_window.csv": list(ppc.columns),
            "spike_field_coherence.csv": list(sfc.columns),
            "spike_lfp_rate_power_by_window.csv": list(rate_power.columns),
            "spike_lfp_rate_power_summary.csv": list(rate_power_summary.columns),
            "spike_lfp_phase_summary.csv": list(summary.columns),
            "spike_lfp_coupling_qc_summary.csv": list(qc_summary.columns),
            "spike_lfp_sta_summary.csv": STA_COLUMNS,
        },
    }
    Path(paths["parameters_json"]).write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")
    progress(f"[spike-lfp] wrote PPC: {paths['ppc_csv']}")
    return paths


def _phase_summary(ppc: pd.DataFrame) -> pd.DataFrame:
    if ppc.empty:
        return pd.DataFrame(
            columns=[
                "experiment_id",
                "animal_id",
                "preparation_id",
                "epoch_label",
                "phase",
                "lfp_channel",
                "spike_channel",
                "band_key",
                "n_valid_windows",
            ]
        )
    valid = ppc[_as_bool(ppc.get("qc_pass", pd.Series(True, index=ppc.index)))].copy()
    if valid.empty:
        return pd.DataFrame(columns=["n_valid_windows"])
    group_cols = [
        col
        for col in (
            "experiment_id",
            "animal_id",
            "preparation_id",
            "epoch_label",
            "phase",
            "lfp_channel",
            "spike_channel",
            "band_key",
            "band_label",
            "phase_band_lo_hz",
            "phase_band_hi_hz",
        )
        if col in valid.columns
    ]
    grouped = valid.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        ppc_median=("ppc", "median"),
        ppc_mean=("ppc", "mean"),
        preferred_phase_rad=("preferred_phase_rad", "median"),
        spike_count_total=("spike_count", "sum"),
        valid_lfp_duration_s=("valid_lfp_duration_s", "sum"),
        n_valid_windows=("ppc", "count"),
    ).reset_index()
    if "preparation_id" in group_cols:
        summary["biological_n_preparations"] = summary["preparation_id"].astype(str).str.strip().ne("").astype(int)
    elif "preparation_id" in valid.columns:
        prep_counts = grouped["preparation_id"].nunique().reset_index().rename(columns={"preparation_id": "biological_n_preparations"})
        summary = summary.merge(prep_counts, on=group_cols, how="left")
    else:
        summary["biological_n_preparations"] = np.nan
    summary["coupling_label"] = "MUA-LFP phase coupling"
    summary["interpretation_warning"] = COUPLING_WARNING
    return summary


def _coupling_qc_summary(ppc: pd.DataFrame) -> pd.DataFrame:
    if ppc.empty:
        return pd.DataFrame(columns=["qc_pass", "qc_reason", "rows"])
    return (
        ppc.groupby(["qc_pass", "qc_reason"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
    )


def _bounded_legend(ax, max_entries: int = 12) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if len(handles) > max_entries:
        ax.legend(handles[:max_entries], labels[:max_entries], loc="best", fontsize=7, title=f"First {max_entries} of {len(handles)}")
    else:
        ax.legend(loc="best", fontsize=7)


def _deduplicate_axis_legend(ax, max_entries: int = 8) -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen[label] = handle
    if not seen:
        return
    items = list(seen.items())
    shown = items[:max_entries]
    title = f"First {max_entries} of {len(items)}" if len(items) > max_entries else None
    ax.legend([handle for _label, handle in shown], [label for label, _handle in shown], loc="best", fontsize=7, title=title)


def _finite_xy(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x_num = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_num = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_num) & np.isfinite(y_num)
    return x_num[mask], y_num[mask]


def _safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    x_num, y_num = _finite_xy(x, y)
    n = int(x_num.size)
    if n < 3 or np.unique(x_num).size < 2 or np.unique(y_num).size < 2:
        return float("nan"), float("nan"), n
    result = stats.spearmanr(x_num, y_num)
    return float(result.statistic), float(result.pvalue), n


def _rate_power_interpretation(rho: float, p_value: float, n: int) -> str:
    if n < 3 or not np.isfinite(rho):
        return RATE_POWER_NO_CORRELATION_INTERPRETATION
    significant = not np.isfinite(p_value) or p_value < 0.05
    if not significant or abs(float(rho)) < 0.2:
        return RATE_POWER_NO_CORRELATION_INTERPRETATION
    if rho > 0:
        return RATE_POWER_POSITIVE_INTERPRETATION
    return RATE_POWER_NEGATIVE_INTERPRETATION


def _epoch_separation_caveat(rate_power: pd.DataFrame, rho: float, p_value: float) -> str:
    if rate_power.empty or "epoch_label" not in rate_power.columns or not np.isfinite(rho):
        return ""
    if np.isfinite(p_value) and p_value >= 0.05:
        return ""
    if abs(float(rho)) < 0.2:
        return ""
    labels = rate_power["epoch_label"].fillna("").astype(str).str.strip()
    if labels.nunique(dropna=True) < 2:
        return ""
    for _label, sub in rate_power.groupby(labels, dropna=False, sort=False):
        erho, ep, n = _safe_spearman(sub["spike_rate_hz"], sub["lfp_power_uV2"])
        if n < 3 or not np.isfinite(erho):
            continue
        if abs(float(erho)) >= 0.2 and (not np.isfinite(ep) or ep < 0.05):
            return ""
    return RATE_POWER_EPOCH_CAVEAT


def _rate_power_from_ppc(ppc: pd.DataFrame) -> pd.DataFrame:
    if ppc.empty:
        return pd.DataFrame(columns=RATE_POWER_COLUMNS)
    work = ppc.copy()
    dedupe_keys = [
        "experiment_id",
        "animal_id",
        "preparation_id",
        "segment_id",
        "recording_id",
        "lfp_channel",
        "spike_channel",
        "window_index",
        "window_start_s",
        "window_end_s",
    ]
    existing_keys = [col for col in dedupe_keys if col in work.columns]
    if existing_keys:
        work = work.sort_values(existing_keys).drop_duplicates(existing_keys, keep="first")

    out = pd.DataFrame(index=work.index)
    for col in RATE_POWER_COLUMNS:
        if col in work.columns:
            out[col] = work[col]
    valid_duration = pd.to_numeric(work.get("valid_lfp_duration_s", pd.Series(np.nan, index=work.index)), errors="coerce")
    fallback_duration = (
        pd.to_numeric(work.get("window_end_s", pd.Series(np.nan, index=work.index)), errors="coerce")
        - pd.to_numeric(work.get("window_start_s", pd.Series(np.nan, index=work.index)), errors="coerce")
    )
    duration = valid_duration.where(valid_duration > 0, fallback_duration)
    spike_count = pd.to_numeric(work.get("spike_count", pd.Series(np.nan, index=work.index)), errors="coerce")
    power = pd.to_numeric(work.get("primary_power_uV2", pd.Series(np.nan, index=work.index)), errors="coerce")
    out["valid_lfp_duration_s"] = duration
    out["spike_rate_hz"] = spike_count / duration.replace(0, np.nan)
    out["spike_rate_per_min"] = out["spike_rate_hz"] * 60.0
    out["lfp_power_uV2"] = power
    out["lfp_power_label"] = "primary_power_uV2"
    signal_ok = _as_bool(work.get("signal_qc_pass", pd.Series(True, index=work.index)), default=True)
    feature_ok = _as_bool(work.get("feature_qc_pass", pd.Series(True, index=work.index)), default=True)
    finite = out["spike_rate_hz"].notna() & out["lfp_power_uV2"].notna() & (duration > 0)
    matched = signal_ok & feature_ok & finite
    out["signal_qc_pass"] = signal_ok
    out["feature_qc_pass"] = feature_ok
    out["matched_window_qc_pass"] = matched
    reasons = []
    for idx in out.index:
        row_reasons = []
        if not bool(signal_ok.loc[idx]):
            row_reasons.append("signal_qc_fail")
        if not bool(feature_ok.loc[idx]):
            row_reasons.append("feature_qc_fail")
        if not bool(finite.loc[idx]):
            row_reasons.append("missing_rate_or_power")
        reasons.append(";".join(row_reasons) if row_reasons else "pass")
    out["matched_window_qc_reason"] = reasons
    for col in RATE_POWER_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[RATE_POWER_COLUMNS].reset_index(drop=True)


def _rate_power_summary(rate_power: pd.DataFrame) -> pd.DataFrame:
    if rate_power.empty:
        return pd.DataFrame(columns=RATE_POWER_SUMMARY_COLUMNS)
    valid = rate_power[_as_bool(rate_power.get("matched_window_qc_pass", pd.Series(False, index=rate_power.index)))].copy()
    valid = valid.dropna(subset=["spike_rate_hz", "lfp_power_uV2"])
    if valid.empty:
        return pd.DataFrame(columns=RATE_POWER_SUMMARY_COLUMNS)

    rows: list[dict] = []

    def add_summary(scope: str, sub: pd.DataFrame, fixed: Mapping[str, object] | None = None) -> None:
        rho, p_value, n = _safe_spearman(sub["spike_rate_hz"], sub["lfp_power_uV2"])
        caveat = _epoch_separation_caveat(sub, rho, p_value) if scope != "epoch" else ""
        row = {
            "scope": scope,
            "experiment_id": "",
            "animal_id": "",
            "preparation_id": "",
            "lfp_channel": "",
            "spike_channel": "",
            "channel_pair": "",
            "epoch_label": "",
            "phase": "",
            "lfp_power_label": "primary_power_uV2",
            "n_matched_windows": n,
            "spearman_rho_s": rho,
            "spearman_p_value": p_value,
            "interpretation": _rate_power_interpretation(rho, p_value, n),
            "epoch_separation_caveat": caveat,
        }
        if fixed:
            row.update(fixed)
        rows.append(row)

    add_summary("pooled", valid)
    pair_cols = [col for col in ("experiment_id", "animal_id", "preparation_id", "lfp_channel", "spike_channel", "channel_pair") if col in valid.columns]
    if pair_cols:
        for key, sub in valid.groupby(pair_cols, dropna=False, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            add_summary("channel_pair", sub, dict(zip(pair_cols, key)))
    epoch_cols = [col for col in ("epoch_label", "phase") if col in valid.columns]
    if epoch_cols:
        for key, sub in valid.groupby(epoch_cols, dropna=False, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            add_summary("epoch", sub, dict(zip(epoch_cols, key)))

    summary = pd.DataFrame(rows)
    for col in RATE_POWER_SUMMARY_COLUMNS:
        if col not in summary.columns:
            summary[col] = np.nan
    return summary[RATE_POWER_SUMMARY_COLUMNS]


def _phase_samples_to_frame(phase_samples: list[dict] | None) -> pd.DataFrame:
    if not phase_samples:
        return pd.DataFrame(columns=["band_label", "phase_rad"])
    rows = []
    for item in phase_samples:
        phase_values = np.asarray(item.get("phase_rad", []), dtype=float)
        phase_values = phase_values[np.isfinite(phase_values)]
        for value in phase_values:
            rows.append(
                {
                    "band_key": item.get("band_key", ""),
                    "band_label": item.get("band_label", ""),
                    "epoch_label": item.get("epoch_label", ""),
                    "phase": item.get("phase", ""),
                    "channel_pair": item.get("channel_pair", ""),
                    "phase_rad": float(value),
                }
            )
    return pd.DataFrame(rows)


def _plot_spike_rate_vs_lfp_power(rate_power: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    if rate_power.empty:
        return
    valid = rate_power[_as_bool(rate_power.get("matched_window_qc_pass", pd.Series(False, index=rate_power.index)))].copy()
    valid = valid.dropna(subset=["spike_rate_hz", "lfp_power_uV2", "cumulative_window_mid_s"])
    if valid.empty:
        return
    valid["time_min"] = pd.to_numeric(valid["cumulative_window_mid_s"], errors="coerce") / 60.0
    valid["epoch_plot_label"] = valid.get("epoch_label", pd.Series("", index=valid.index)).fillna("").astype(str).replace("", "Unlabeled")
    time = (
        valid.groupby(["epoch_plot_label", "segment_id", "time_min"], dropna=False)
        .agg(spike_rate_hz=("spike_rate_hz", "median"), lfp_power_uV2=("lfp_power_uV2", "median"))
        .reset_index()
        .sort_values("time_min")
    )
    labels = list(pd.Categorical(valid["epoch_plot_label"]).categories)
    cmap = plt.get_cmap("tab10")
    color_by_label = {label: cmap(idx % 10) for idx, label in enumerate(labels)}

    fig = plt.figure(figsize=(13.5, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1, 1])
    ax_rate = fig.add_subplot(gs[0, 0])
    ax_power = fig.add_subplot(gs[1, 0], sharex=ax_rate)
    ax_scatter = fig.add_subplot(gs[:, 1])

    for label, sub in time.groupby("epoch_plot_label", dropna=False, sort=False):
        color = color_by_label.get(label, "C0")
        for _segment, seg in sub.groupby("segment_id", dropna=False, sort=False):
            ax_rate.plot(seg["time_min"], seg["spike_rate_hz"], marker="o", linewidth=1.3, color=color, label=str(label))
            ax_power.plot(seg["time_min"], seg["lfp_power_uV2"], marker="o", linewidth=1.3, color=color, label=str(label))
    _add_ppc_segment_boundaries(ax_rate, valid)
    _add_ppc_segment_boundaries(ax_power, valid)
    ax_rate.set_ylabel("Spike rate (Hz)")
    ax_power.set_ylabel("LFP power (uV^2)")
    ax_power.set_xlabel("Cumulative recorded time (min)")
    ax_rate.set_title("Matched-window spike rate")
    ax_power.set_title("Matched-window LFP power")
    _deduplicate_axis_legend(ax_rate)

    for label, sub in valid.groupby("epoch_plot_label", dropna=False, sort=False):
        ax_scatter.scatter(
            pd.to_numeric(sub["spike_rate_hz"], errors="coerce"),
            pd.to_numeric(sub["lfp_power_uV2"], errors="coerce"),
            s=38,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.45,
            color=color_by_label.get(label, "C0"),
            label=str(label),
        )
    pooled = summary[summary["scope"] == "pooled"].iloc[0] if not summary.empty and (summary["scope"] == "pooled").any() else {}
    rho = float(pooled.get("spearman_rho_s", np.nan)) if isinstance(pooled, pd.Series) else float("nan")
    p_value = float(pooled.get("spearman_p_value", np.nan)) if isinstance(pooled, pd.Series) else float("nan")
    n = int(pooled.get("n_matched_windows", len(valid))) if isinstance(pooled, pd.Series) and pd.notna(pooled.get("n_matched_windows", np.nan)) else len(valid)
    interpretation = str(pooled.get("interpretation", RATE_POWER_NO_CORRELATION_INTERPRETATION)) if isinstance(pooled, pd.Series) else RATE_POWER_NO_CORRELATION_INTERPRETATION
    caveat = str(pooled.get("epoch_separation_caveat", "")) if isinstance(pooled, pd.Series) else ""
    ax_scatter.set_xlabel("Spike rate (Hz)")
    ax_scatter.set_ylabel("LFP power (uV^2)")
    ax_scatter.set_title("Spearman rank correlation")
    rho_text = "nan" if not np.isfinite(rho) else f"{rho:.3f}"
    p_text = "nan" if not np.isfinite(p_value) else f"{p_value:.3g}"
    note = f"rho_s = cor(rank(R), rank(P)) = {rho_text}\np = {p_text}, n = {n}\n{interpretation}"
    if caveat:
        note += "\n" + RATE_POWER_EPOCH_CAVEAT
    ax_scatter.text(
        0.03,
        0.97,
        note,
        transform=ax_scatter.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#888888", "alpha": 0.9},
    )
    _bounded_legend(ax_scatter, max_entries=8)
    fig.suptitle("PRIMARY Spike Rate Versus LFP Power")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "PRIMARY_spike_rate_vs_lfp_power.png", dpi=200)
    plt.close(fig)


def _add_ppc_segment_boundaries(ax, df: pd.DataFrame, x_scale: float = 60.0) -> None:
    needed = {"segment_id", "cumulative_window_mid_s"}
    if not needed.issubset(df.columns):
        return
    bounds = (
        df.groupby(["preparation_id", "segment_id"] if "preparation_id" in df.columns else ["segment_id"], dropna=False)
        .agg(start=("cumulative_window_mid_s", "min"))
        .sort_values("start")
        .reset_index()
    )
    for idx, row in bounds.iterrows():
        if idx == 0:
            continue
        x = pd.to_numeric(pd.Series([row["start"]]), errors="coerce").iloc[0]
        if np.isfinite(x):
            ax.axvline(float(x) / x_scale, color="black", linestyle="--", alpha=0.22, linewidth=1)


def _plot_primary_ppc_summary(ppc: pd.DataFrame, out_dir: Path, phase_samples: list[dict] | None = None) -> None:
    valid = ppc[_as_bool(ppc.get("qc_pass", pd.Series(True, index=ppc.index)))].copy() if not ppc.empty else pd.DataFrame()
    if valid.empty or "ppc" not in valid.columns:
        return
    valid["time_min"] = pd.to_numeric(valid.get("cumulative_window_mid_s", pd.Series(np.nan, index=valid.index)), errors="coerce") / 60.0
    valid["ppc_num"] = pd.to_numeric(valid["ppc"], errors="coerce")
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0])
    ax_time = fig.add_subplot(gs[0, 0])
    ax_change = fig.add_subplot(gs[0, 1])
    ax_phase = fig.add_subplot(gs[1, 0], projection="polar")
    ax_sig = fig.add_subplot(gs[1, 1])

    label_col = "band_label" if "band_label" in valid.columns else "band_key"
    grouped_time = (
        valid.groupby([label_col, "segment_id", "time_min"], dropna=False)["ppc_num"]
        .median()
        .reset_index()
        .sort_values("time_min")
    )
    for band, sub in grouped_time.groupby(label_col, dropna=False, sort=False):
        ax_time.plot(
            pd.to_numeric(sub["time_min"], errors="coerce"),
            pd.to_numeric(sub["ppc_num"], errors="coerce"),
            marker="o",
            linewidth=1.4,
            label=str(band),
        )
    ax_time.axhline(0, color="black", alpha=0.4, linewidth=1)
    _add_ppc_segment_boundaries(ax_time, valid)
    ax_time.set_title("PPC time course by phase band")
    ax_time.set_xlabel("Cumulative recorded time (min)")
    ax_time.set_ylabel("PPC")
    _bounded_legend(ax_time, max_entries=8)

    if "ppc_change_from_baseline" in valid.columns and valid["ppc_change_from_baseline"].notna().any():
        change = (
            valid.dropna(subset=["ppc_change_from_baseline"])
            .groupby([label_col, "epoch_label"] if "epoch_label" in valid.columns else [label_col], dropna=False)["ppc_change_from_baseline"]
            .median()
            .reset_index()
        )
        labels = [
            " | ".join(str(row[col]) for col in change.columns if col != "ppc_change_from_baseline")
            for _idx, row in change.iterrows()
        ]
        x = np.arange(len(change))
        ax_change.bar(x, pd.to_numeric(change["ppc_change_from_baseline"], errors="coerce"))
        ax_change.axhline(0, color="black", linewidth=1)
        ax_change.set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
        ax_change.set_ylabel("PPC change")
    else:
        ax_change.text(0.5, 0.5, "No matched baseline change available", ha="center", va="center", transform=ax_change.transAxes)
        ax_change.set_xticks([])
    ax_change.set_title("Baseline-to-epoch change")

    phase_df = _phase_samples_to_frame(phase_samples)
    if not phase_df.empty and "band_label" in phase_df.columns:
        counts_by_band = phase_df["band_label"].fillna("").astype(str).replace("", "Unlabeled").value_counts()
        chosen_band = counts_by_band.index[0]
        phase_values = pd.to_numeric(
            phase_df.loc[phase_df["band_label"].fillna("").astype(str).replace("", "Unlabeled") == chosen_band, "phase_rad"],
            errors="coerce",
        ).dropna()
        phase_values = ((phase_values.to_numpy(dtype=float) + np.pi) % (2.0 * np.pi)) - np.pi
        if phase_values.size:
            bins = np.linspace(-np.pi, np.pi, 25)
            counts, edges = np.histogram(phase_values, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2.0
            ax_phase.bar(centers, counts, width=np.diff(edges), alpha=0.72, color="#4C78A8")
            mean_vector = np.nanmean(np.exp(1j * phase_values))
            if np.isfinite(mean_vector):
                preferred = float(np.angle(mean_vector))
                ax_phase.plot([preferred, preferred], [0, max(1.0, float(np.nanmax(counts)))], color="#E45756", linewidth=2.0)
            ax_phase.set_title(f"Spike-time phase histogram | {chosen_band}")
    else:
        phase = pd.to_numeric(valid.get("preferred_phase_rad", pd.Series(np.nan, index=valid.index)), errors="coerce")
        radius = pd.to_numeric(valid["ppc_num"], errors="coerce").clip(lower=0)
        finite_phase = phase.notna() & radius.notna()
        if finite_phase.any():
            colors = pd.Categorical(valid.loc[finite_phase, label_col]).codes
            ax_phase.scatter(phase[finite_phase], radius[finite_phase], c=colors, cmap="tab10", s=28, alpha=0.75)
        ax_phase.set_title("Preferred phase by window")

    sig_col = "surrogate_q_value" if "surrogate_q_value" in valid.columns else "surrogate_p_value"
    sig = pd.to_numeric(valid.get(sig_col, pd.Series(np.nan, index=valid.index)), errors="coerce")
    if sig.notna().any():
        y = -np.log10(sig.clip(lower=np.nextafter(0.0, 1.0)))
        marker_size = np.where(_as_bool(valid.get("surrogate_significant_fdr_0p05", pd.Series(False, index=valid.index))), 52, 24)
        ax_sig.scatter(valid["time_min"], y, s=marker_size, alpha=0.75)
        ax_sig.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax_sig.set_ylabel(f"-log10({sig_col})")
    else:
        ax_sig.text(0.5, 0.5, "No finite surrogate values", ha="center", va="center", transform=ax_sig.transAxes)
        ax_sig.set_yticks([])
    _add_ppc_segment_boundaries(ax_sig, valid)
    ax_sig.set_title("Surrogate/FDR readout")
    ax_sig.set_xlabel("Cumulative recorded time (min)")

    flags = int(np.count_nonzero(_as_bool(valid.get("high_frequency_same_channel_flag", pd.Series(False, index=valid.index)))))
    note = f" | high-frequency same-channel flagged rows: {flags}" if flags else ""
    fig.suptitle(f"PRIMARY Spike-LFP PPC phase coupling | 20 kHz spikes + synchronized 2 kHz LFP phase{note}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "PRIMARY_spike_lfp_ppc_phase_coupling.png", dpi=200)
    plt.close(fig)


def _plot_outputs(ppc: pd.DataFrame, sfc: pd.DataFrame, out_dir: Path, *, include_complex: bool = True, phase_samples: list[dict] | None = None) -> None:
    valid_ppc = ppc[_as_bool(ppc.get("qc_pass", pd.Series(True, index=ppc.index)))].copy() if not ppc.empty else pd.DataFrame()
    if not valid_ppc.empty and "ppc" in valid_ppc.columns:
        _plot_primary_ppc_summary(valid_ppc, out_dir, phase_samples=phase_samples)
        if not include_complex:
            return
        for (prep, pair, band), sub in valid_ppc.groupby(["preparation_id", "channel_pair", "band_label"], dropna=False, sort=False):
            fig, ax = plt.subplots(figsize=(10, 4.6))
            label_col = "epoch_label" if "epoch_label" in sub.columns else "phase"
            for label, epoch_sub in sub.groupby(label_col, dropna=False, sort=False):
                for _seg, seg_sub in epoch_sub.groupby("segment_id", dropna=False, sort=False):
                    x = pd.to_numeric(seg_sub["cumulative_window_mid_s"], errors="coerce") / 60.0
                    y = pd.to_numeric(seg_sub["ppc"], errors="coerce")
                    ax.plot(x, y, marker="o", linewidth=1.5, label=str(label))
                    if {"surrogate_ci_low", "surrogate_ci_high"}.issubset(seg_sub.columns):
                        ax.fill_between(
                            x.to_numpy(dtype=float),
                            pd.to_numeric(seg_sub["surrogate_ci_low"], errors="coerce").to_numpy(dtype=float),
                            pd.to_numeric(seg_sub["surrogate_ci_high"], errors="coerce").to_numpy(dtype=float),
                            alpha=0.15,
                        )
            ax.axhline(0, color="black", alpha=0.45, linewidth=1)
            ax.set_title(f"PPC time course | prep={prep} | {pair} | {band}")
            ax.set_xlabel("Cumulative recorded time (min)")
            ax.set_ylabel("PPC")
            _bounded_legend(ax)
            fig.tight_layout()
            fig.savefig(out_dir / f"spike_lfp_ppc_timecourse_{_safe_name(prep)}_{_safe_name(pair)}_{_safe_name(band)}.png", dpi=200)
            plt.close(fig)
        _plot_ppc_paired(valid_ppc, out_dir)
        _plot_ppc_change(valid_ppc, out_dir)
        _plot_preferred_phase(valid_ppc, out_dir)
    if include_complex:
        valid_sfc = sfc[_as_bool(sfc.get("qc_pass", pd.Series(True, index=sfc.index)))].copy() if not sfc.empty else pd.DataFrame()
        if not valid_sfc.empty and "sfc" in valid_sfc.columns:
            _plot_sfc_spectrum(valid_sfc, out_dir, "sfc", "spike_field_coherence_spectrum.png", "SFC spectrum")
            if "sfc_change_from_baseline" in valid_sfc.columns and valid_sfc["sfc_change_from_baseline"].notna().any():
                _plot_sfc_spectrum(valid_sfc.dropna(subset=["sfc_change_from_baseline"]), out_dir, "sfc_change_from_baseline", "spike_lfp_sfc_change_from_baseline_spectrum.png", "SFC epoch minus baseline")
        _plot_spike_lfp_aligned(valid_ppc, out_dir)


def _plot_ppc_paired(ppc: pd.DataFrame, out_dir: Path) -> None:
    if "baseline_median_ppc" not in ppc.columns or not ppc["baseline_median_ppc"].notna().any():
        return
    sub = ppc.dropna(subset=["baseline_median_ppc", "ppc"]).copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for _key, row in sub.iterrows():
        label = str(row.get("epoch_label") or row.get("phase"))
        ax.plot([0, 1], [row["baseline_median_ppc"], row["ppc"]], marker="o", alpha=0.35, linewidth=1)
    ax.set_xticks([0, 1], ["Baseline median", "Epoch"])
    ax.set_ylabel("PPC")
    ax.set_title("Matched baseline-to-epoch PPC")
    fig.tight_layout()
    fig.savefig(out_dir / "spike_lfp_ppc_paired_baseline_to_epoch.png", dpi=200)
    plt.close(fig)


def _plot_ppc_change(ppc: pd.DataFrame, out_dir: Path) -> None:
    if "ppc_change_from_baseline" not in ppc.columns or not ppc["ppc_change_from_baseline"].notna().any():
        return
    sub = ppc.dropna(subset=["ppc_change_from_baseline"]).copy()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = []
    values = []
    for key, grp in sub.groupby(["channel_pair", "band_label", "epoch_label"], dropna=False, sort=False):
        labels.append(" | ".join(map(str, key)))
        values.append(float(pd.to_numeric(grp["ppc_change_from_baseline"], errors="coerce").median()))
    ax.bar(np.arange(len(values)), values)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("PPC change from matched baseline")
    ax.set_title("Epoch minus baseline PPC")
    ax.set_xticks(np.arange(len(values)), labels, rotation=35, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "spike_lfp_ppc_change_from_baseline.png", dpi=200)
    plt.close(fig)


def _plot_preferred_phase(ppc: pd.DataFrame, out_dir: Path) -> None:
    phases = pd.to_numeric(ppc.get("preferred_phase_rad", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    if phases.size == 0:
        return
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    bins = np.linspace(-np.pi, np.pi, 25)
    counts, edges = np.histogram(phases, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    ax.bar(centers, counts, width=np.diff(edges), alpha=0.75)
    ax.set_title("Preferred LFP phase")
    fig.tight_layout()
    fig.savefig(out_dir / "spike_lfp_preferred_phase_histogram.png", dpi=200)
    plt.close(fig)


def _plot_sfc_spectrum(sfc: pd.DataFrame, out_dir: Path, value_col: str, filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    label_col = "epoch_label" if "epoch_label" in sfc.columns else "phase"
    grouped = (
        sfc.groupby(["preparation_id", "channel_pair", label_col, "frequency_hz"], dropna=False)[value_col]
        .median()
        .reset_index()
    )
    for key, sub in grouped.groupby(["preparation_id", "channel_pair", label_col], dropna=False, sort=False):
        ax.plot(
            pd.to_numeric(sub["frequency_hz"], errors="coerce"),
            pd.to_numeric(sub[value_col], errors="coerce"),
            linewidth=1.5,
            label=" | ".join(map(str, key)),
        )
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(value_col.replace("_", " "))
    _bounded_legend(ax)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=200)
    plt.close(fig)


def _plot_spike_lfp_aligned(ppc: pd.DataFrame, out_dir: Path) -> None:
    if ppc.empty or "primary_power_uV2" not in ppc.columns or not ppc["primary_power_uV2"].notna().any():
        return
    sub = ppc.dropna(subset=["primary_power_uV2"]).copy()
    if sub.empty:
        return
    for (prep, pair), grp in sub.groupby(["preparation_id", "channel_pair"], dropna=False, sort=False):
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for _seg, seg in grp.groupby("segment_id", dropna=False, sort=False):
            x = pd.to_numeric(seg["cumulative_window_mid_s"], errors="coerce") / 60.0
            axes[0].plot(x, pd.to_numeric(seg["spike_count"], errors="coerce"), marker="o", linewidth=1.2)
            axes[1].plot(x, pd.to_numeric(seg["primary_power_uV2"], errors="coerce"), marker="o", linewidth=1.2)
        axes[0].set_ylabel("Spike count")
        axes[1].set_ylabel("LFP primary power")
        axes[1].set_xlabel("Cumulative recorded time (min)")
        axes[0].set_title(f"Aligned spike count and LFP power | prep={prep} | {pair}")
        fig.tight_layout()
        fig.savefig(out_dir / f"spike_lfp_count_lfp_power_{_safe_name(prep)}_{_safe_name(pair)}.png", dpi=200)
        plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate SPIE spike-LFP PPC and SFC.")
    parser.add_argument("spike_events_csv")
    parser.add_argument("--manifest", required=True, help="lfp_recording_manifest.csv or raw manifest.")
    parser.add_argument("--out-dir", default="spike_lfp_coupling")
    parser.add_argument("--lfp-channels", default="")
    parser.add_argument("--spike-channels", default="")
    parser.add_argument("--phase-band", default="", help="Optional single exploratory band such as 1-40.")
    parser.add_argument("--phase-bands", default="", help="Semicolon-separated bands such as delta=1-4; theta=4-8.")
    parser.add_argument("--window-sec", type=float, default=60.0)
    parser.add_argument("--min-spikes", type=int, default=20)
    parser.add_argument("--min-oscillatory-cycles", type=float, default=3.0)
    parser.add_argument("--surrogates", type=int, default=200)
    parser.add_argument("--coupling-downsample-hz", default=str(int(DEFAULT_LFP_DOWNSAMPLE_HZ)))
    parser.add_argument("--max-sfc-frequency-hz", type=float, default=DEFAULT_LFP_EXPLORATORY_BAND_HZ[1])
    parser.add_argument("--sta-lfp-band", default="0.5-300", help="Broad LFP band for spike-triggered averages, e.g. 0.5-300.")
    parser.add_argument("--sta-window-sec", type=float, default=0.200, help="Half-window around each spike for STA output.")
    parser.add_argument("--sta-min-spikes", type=int, default=5)
    parser.add_argument("--sta-max-pairs", type=int, default=24)
    parser.add_argument("--no-sta-spectrogram", action="store_true")
    parser.add_argument("--pairing-mode", choices=("same_channel", "explicit", "all_to_all"), default="same_channel")
    parser.add_argument("--processing-mode", choices=("simple", "complex"), default="complex")
    parser.add_argument("--explicit-pairs", default="", help="Pairs as spike->lfp; spike2->lfp2.")
    parser.add_argument("--lfp-feature-csv", default="")
    parser.add_argument("--allow-large-all-to-all", action="store_true")
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    phase_bands = _parse_phase_bands(args.phase_bands)
    if not phase_bands and args.phase_band:
        lo, hi = _parse_frequency_pair(args.phase_band)
        label = "Exploratory broad 1-40 Hz" if math.isclose(lo, 1.0) and math.isclose(hi, 40.0) else f"{lo:g}-{hi:g} Hz"
        phase_bands = [(f"exploratory_{lo:g}_{hi:g}", label, (lo, hi))]
    cfg = SpikeLFPCouplingConfig(
        output_dir=args.out_dir,
        lfp_channels=parse_channel_list(args.lfp_channels),
        spike_channels=parse_channel_list(args.spike_channels),
        phase_bands=phase_bands,
        window_s=args.window_sec,
        min_spikes=args.min_spikes,
        min_oscillatory_cycles=args.min_oscillatory_cycles,
        surrogate_count=args.surrogates,
        pairing_mode=args.pairing_mode,
        processing_mode=args.processing_mode,
        explicit_channel_pairs=_parse_explicit_pairs(args.explicit_pairs),
        lfp_feature_csv=args.lfp_feature_csv or None,
        allow_large_all_to_all=args.allow_large_all_to_all,
        coupling_downsample_hz=None if str(args.coupling_downsample_hz).strip().lower() in {"", "none", "0"} else float(args.coupling_downsample_hz),
        max_sfc_frequency_hz=args.max_sfc_frequency_hz,
        sta_lfp_band_hz=_parse_frequency_pair(args.sta_lfp_band),
        sta_window_s=args.sta_window_sec,
        sta_min_spikes=args.sta_min_spikes,
        sta_max_pairs=args.sta_max_pairs,
        sta_spectrogram=not args.no_sta_spectrogram,
    )
    calculate_spike_lfp_coupling(args.spike_events_csv, args.manifest, cfg)


if __name__ == "__main__":
    main()
