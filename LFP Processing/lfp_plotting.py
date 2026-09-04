#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot processed SPIE LFP feature outputs.

Plots in this module consume ``lfp_features_by_window.csv`` and companion
processed outputs. They do not recompute LFP features from raw traces.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats


DEFAULT_DPI = 200

DEFAULT_SIMPLE_LFP_BANDS = [
    (
        "very_slow",
        "Very slow",
        (0.5, 2.0),
        "Slow state changes, modulation, ventilation-related activity and preparation drift",
    ),
    (
        "slow_motor",
        "Slow motor",
        (2.0, 5.0),
        "Slow walking-like or discontinuous rhythmic activity",
    ),
    (
        "intermediate_motor",
        "Intermediate motor",
        (5.0, 9.0),
        "Slower fictive motor rhythms and transitions into rhythmic states",
    ),
    (
        "fictive_flight_range",
        "Fictive-flight range",
        (9.0, 15.0),
        "Deafferented or pharmacologically induced flight commonly occurs near 10-13 Hz",
    ),
    (
        "intact_flight_range",
        "Intact-flight range",
        (15.0, 25.0),
        "Natural locust wingbeat and flight motor rhythms are commonly around 20-25 Hz",
    ),
    (
        "motor_harmonic_fast_synaptic",
        "Motor harmonic/fast synaptic",
        (25.0, 40.0),
        "Harmonics of motor rhythms and faster coordinated population activity",
    ),
    (
        "fast_population_activity",
        "Fast population activity",
        (40.0, 80.0),
        "Exploratory narrowband or broadband synaptic/population activity",
    ),
]


def _safe_name(value: object) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_") or "plot"


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "pass"})


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def _resolve_inputs(feature_csv: str | Path, config: Mapping | None = None) -> dict[str, Path]:
    cfg = dict(config or {})
    feature_path = Path(feature_csv)
    base_dir = feature_path.parent
    out_dir = Path(cfg.get("output_dir") or base_dir / "lfp_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "features": feature_path,
        "qc": Path(cfg.get("qc_csv") or base_dir / "lfp_qc_by_window.csv"),
        "psd": Path(cfg.get("psd_long_csv") or base_dir / "lfp_psd_long.csv"),
        "events": Path(cfg.get("events_csv") or base_dir / "lfp_events.csv"),
        "derivative_events": Path(cfg.get("derivative_events_csv") or base_dir / "lfp_derivative_events.csv"),
        "spike_counts": Path(cfg["spike_count_csv"]) if cfg.get("spike_count_csv") else None,
        "spike_events": Path(cfg["spike_events_csv"]) if cfg.get("spike_events_csv") else None,
        "ppc": Path(cfg["ppc_csv"]) if cfg.get("ppc_csv") else None,
        "out_dir": out_dir,
    }


def _plot_mode(config: Mapping | None = None) -> str:
    raw = str((config or {}).get("plot_mode") or (config or {}).get("lfp_plot_mode") or "complex")
    mode = raw.strip().lower().replace("-", "_")
    if mode in {"basic", "simple_lfp", "simple_plots"}:
        return "simple"
    if mode in {"comparison", "comparisons", "full"}:
        return "complex"
    if mode not in {"simple", "complex"}:
        raise ValueError("plot_mode must be simple or complex.")
    return mode


def _parse_plot_bands(value: object) -> list[tuple[str, str, tuple[float, float], str]]:
    if value is None or value == "":
        return list(DEFAULT_SIMPLE_LFP_BANDS)
    bands = []
    if isinstance(value, str):
        for item in value.split(";"):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                label, rng = item.split("=", 1)
                label = label.strip()
            else:
                label = ""
                rng = item
            rng = rng.replace("\u2013", "-").replace("\u2014", "-")
            parts = re.split(r"\s*(?:-|:|,)\s*", rng.strip())
            nums = [float(part) for part in parts if part]
            if len(nums) != 2:
                raise ValueError(f"Could not parse simple LFP band {item!r}. Use label=lo-hi.")
            lo, hi = nums
            band_label = label or f"{lo:g}-{hi:g} Hz"
            bands.append((_safe_name(band_label).lower(), band_label, (float(lo), float(hi)), ""))
        return bands or list(DEFAULT_SIMPLE_LFP_BANDS)
    for item in value:  # type: ignore[union-attr]
        if isinstance(item, Mapping):
            label = str(item.get("label") or item.get("name") or item.get("key") or "band").strip()
            key = _safe_name(item.get("key") or item.get("name") or label).lower()
            lo = float(item.get("lo", item.get("low", item.get("fmin"))))
            hi = float(item.get("hi", item.get("high", item.get("fmax"))))
            reason = str(item.get("reason") or item.get("main_reason") or "")
        else:
            if len(item) >= 4:
                key, label, band, reason = item
            else:
                key, label, band = item
                reason = ""
            lo, hi = band
            key = _safe_name(key).lower()
            label = str(label)
            reason = str(reason)
        bands.append((str(key), label, (float(lo), float(hi)), reason))
    return bands or list(DEFAULT_SIMPLE_LFP_BANDS)


def _simple_plot_bands(config: Mapping | None = None) -> list[tuple[str, str, tuple[float, float], str]]:
    cfg = dict(config or {})
    return _parse_plot_bands(cfg.get("simple_plot_bands") or cfg.get("narrow_bands"))


def _band_display_label(band: tuple[str, str, tuple[float, float], str]) -> str:
    _key, label, (lo, hi), _reason = band
    return f"{label} ({lo:g}-{hi:g} Hz)"


def _write_simple_band_definitions(
    bands: list[tuple[str, str, tuple[float, float], str]],
    out_dir: Path,
) -> str:
    out = out_dir / "lfp_simple_narrow_band_definitions.csv"
    rows = [
        {
            "band_key": key,
            "suggested_label": label,
            "low_hz": float(lo),
            "high_hz": float(hi),
            "main_reason_to_examine": reason,
        }
        for key, label, (lo, hi), reason in bands
    ]
    pd.DataFrame(rows).to_csv(out, index=False)
    return str(out)


def _series_colors(labels) -> dict:
    unique = list(dict.fromkeys(str(label) for label in labels))
    if not unique:
        return {}
    if len(unique) == 1:
        return {unique[0]: plt.get_cmap("tab10")(0)}
    cmap = plt.get_cmap("turbo", len(unique))
    return {label: cmap(idx) for idx, label in enumerate(unique)}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> str:
    out = out_dir / name
    fig.tight_layout()
    fig.savefig(out, dpi=DEFAULT_DPI)
    plt.close(fig)
    return str(out)


def _phase_colors(phases) -> dict:
    palette = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("Set2").colors)
    return {phase: palette[i % len(palette)] for i, phase in enumerate(phases)}


def _label_col(df: pd.DataFrame) -> str:
    if "epoch_label" in df.columns and df["epoch_label"].astype(str).str.strip().ne("").any():
        return "epoch_label"
    return "phase"


def _segment_col(df: pd.DataFrame) -> str:
    if "segment_id" in df.columns:
        return "segment_id"
    if "recording_id" in df.columns:
        return "recording_id"
    return "source_file"


def _add_segment_boundaries(ax, df: pd.DataFrame, x_scale: float = 60.0) -> None:
    if "cumulative_window_start_s" not in df.columns or "cumulative_window_end_s" not in df.columns:
        return
    segment_col = _segment_col(df)
    if segment_col not in df.columns:
        return
    bounds = (
        df.groupby(["preparation_id", segment_col] if "preparation_id" in df.columns else [segment_col], dropna=False)
        .agg(start=("cumulative_window_start_s", "min"), end=("cumulative_window_end_s", "max"))
        .sort_values("start")
        .reset_index()
    )
    for idx, row in bounds.iterrows():
        if idx == 0:
            continue
        x = pd.to_numeric(pd.Series([row["start"]]), errors="coerce").iloc[0]
        if np.isfinite(x):
            ax.axvline(float(x) / x_scale, color="black", linestyle="--", alpha=0.22, linewidth=1)


def _valid_features(df: pd.DataFrame) -> pd.DataFrame:
    if {"signal_qc_pass", "feature_qc_pass"}.issubset(df.columns):
        return df[_as_bool(df["signal_qc_pass"]) & _as_bool(df["feature_qc_pass"])].copy()
    if "qc_pass" not in df.columns:
        return df.copy()
    return df[_as_bool(df["qc_pass"])].copy()


def _window_match_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    candidates = [
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
        "window_start_s",
        "window_end_s",
    ]
    keys = [key for key in candidates if key in left.columns and key in right.columns]
    if "channel" not in keys:
        return []
    if "window_index" not in keys and "window_start_s" not in keys:
        return []
    return keys


def _filter_psd_to_valid_windows(psd: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    if psd.empty or valid.empty:
        return psd.iloc[0:0].copy()
    keys = _window_match_keys(psd, valid)
    if not keys:
        return psd.copy()
    valid_keys = valid[keys].drop_duplicates()
    return psd.merge(valid_keys, on=keys, how="inner")


def _write_qc_failure_summary(qc: pd.DataFrame, out_dir: Path) -> str | None:
    if qc.empty:
        return None
    rows = []
    for col in ("qc_reason", "signal_qc_reason", "feature_qc_reason", "derivative_qc_status", "line_noise_qc"):
        if col not in qc.columns:
            continue
        counts = qc[col].astype(str).replace({"": "blank"}).value_counts(dropna=False)
        for value, count in counts.items():
            rows.append({"qc_field": col, "value": value, "window_count": int(count)})
    if not rows:
        return None
    out = out_dir / "lfp_qc_failure_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return str(out)


def _time_col(df: pd.DataFrame) -> str:
    for col in ("cumulative_window_mid_s", "cumulative_window_start_s", "window_mid_s", "window_start_s"):
        if col in df.columns:
            return col
    raise ValueError("Processed LFP feature CSV is missing window timing columns.")


def _find_time_column(columns) -> str:
    lower = {str(col).lower(): str(col) for col in columns}
    for candidate in ("time_s", "time", "t", "seconds"):
        if candidate in columns:
            return candidate
        if candidate in lower:
            return lower[candidate]
    raise ValueError("Raw CSV is missing a recognizable time column.")


def _read_raw_trace(path: str | Path, channel: str) -> tuple[np.ndarray, np.ndarray]:
    raw_path = Path(path)
    header = pd.read_csv(raw_path, nrows=0)
    time_col = _find_time_column(list(header.columns))
    if channel not in header.columns:
        raise ValueError(f"Channel {channel!r} not found in {raw_path.name}.")
    df = pd.read_csv(raw_path, usecols=[time_col, channel])
    t = df[time_col].to_numpy(dtype=float)
    t = t - float(t[np.isfinite(t)][0]) if np.isfinite(t).any() else t
    x = df[channel].to_numpy(dtype=float)
    return t, x


def _derivative_z_trace(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    z = np.full_like(x, np.nan, dtype=float)
    deriv = np.full_like(x, np.nan, dtype=float)
    finite = np.isfinite(t) & np.isfinite(x)
    if t.size > 1:
        dt_all = np.diff(t)
        positive = dt_all[np.isfinite(dt_all) & (dt_all > 0)]
        med_dt = np.median(positive) if positive.size else np.nan
    else:
        med_dt = np.nan
    continuous = finite.copy()
    if t.size > 1 and np.isfinite(med_dt):
        bad = np.zeros_like(finite)
        dt = np.diff(t)
        bad[1:] = (~np.isfinite(dt)) | (dt <= 0) | (dt > 1.5 * med_dt)
        continuous &= ~bad
    edges = np.diff(continuous.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    for start, end in zip(starts, ends):
        if end - start < 3:
            continue
        run = x[start:end]
        run_t = t[start:end]
        d = np.diff(run)
        dt = np.diff(run_t)
        good_dt = np.isfinite(dt) & (dt > 0)
        local_deriv = np.full_like(d, np.nan, dtype=float)
        local_deriv[good_dt] = d[good_dt] / dt[good_dt]
        deriv[start + 1 : end] = local_deriv
        mad = np.nanmedian(np.abs(d - np.nanmedian(d)))
        if np.isfinite(mad) and mad > 0:
            z[start + 1 : end] = (d - np.nanmedian(d)) / (1.4826 * mad + np.finfo(float).eps)
    return t, deriv, z


def _continuous_finite_runs(t: np.ndarray | None, x: np.ndarray) -> list[tuple[int, int]]:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if t is not None:
        t = np.asarray(t, dtype=float)
        finite &= np.isfinite(t)
        if t.size > 1:
            dt = np.diff(t)
            positive = dt[np.isfinite(dt) & (dt > 0)]
            if positive.size:
                med_dt = float(np.median(positive))
                discontinuity = np.zeros_like(finite)
                discontinuity[1:] = (~np.isfinite(dt)) | (dt <= 0) | (dt > 1.5 * med_dt)
                finite &= ~discontinuity
    edges = np.diff(finite.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return [(int(start), int(end)) for start, end in zip(starts, ends) if end > start]


def _plot_stride_indices(n: int, max_points: int = 200_000) -> slice:
    stride = max(1, int(math.ceil(max(1, n) / float(max_points))))
    return slice(None, None, stride)


def _safe_filter_trace(x: np.ndarray, fs: float, kind: str, t: np.ndarray | None = None) -> np.ndarray | None:
    if not np.isfinite(fs) or fs <= 0:
        return None
    raw = np.asarray(x, dtype=float)
    if not np.isfinite(raw).any() or raw.size < 32:
        return None
    nyq = 0.5 * fs
    try:
        if kind == "lfp":
            cutoff = min(100.0, nyq * 0.8)
            if cutoff <= 0:
                return None
            sos = signal.butter(4, cutoff / nyq, btype="lowpass", output="sos")
        else:
            if nyq <= 600.0:
                return None
            lo = 300.0 / nyq
            hi = min(3000.0 / nyq, 0.95)
            if not 0 < lo < hi < 1:
                return None
            sos = signal.butter(3, [lo, hi], btype="bandpass", output="sos")
        out = np.full_like(raw, np.nan, dtype=float)
        for start, end in _continuous_finite_runs(t, raw):
            run = raw[start:end]
            if run.size < 32:
                continue
            centered = run - np.nanmedian(run)
            out[start:end] = signal.sosfiltfilt(sos, centered)
        return out if np.isfinite(out).any() else None
    except Exception:
        return None


def plot_lfp_features(feature_csv: str | Path, config: Mapping | None = None) -> dict[str, list[str]]:
    """Create LFP plots from processed feature and QC CSVs."""
    paths = _resolve_inputs(feature_csv, config)
    out_dir = paths["out_dir"]
    mode = _plot_mode(config)
    df = pd.read_csv(paths["features"])
    if df.empty:
        raise ValueError(f"No rows found in {paths['features']}.")
    valid = _valid_features(df)
    outputs: dict[str, list[str]] = {"plots": [], "tables": []}
    qc = pd.DataFrame()
    if paths["qc"].exists():
        qc = pd.read_csv(paths["qc"])
        if not qc.empty:
            outputs["plots"].append(_plot_qc(qc, out_dir))
            summary_path = _write_qc_failure_summary(qc, out_dir)
            if summary_path:
                outputs["tables"].append(summary_path)
    if paths["derivative_events"].exists():
        derivative_plots = _plot_derivative_diagnostics(paths["derivative_events"], qc, out_dir, paths.get("spike_events"))
        outputs["plots"].extend(derivative_plots)
    if valid.empty:
        warning_path = out_dir / "lfp_feature_plots_skipped_all_windows_failed_qc.txt"
        warning_path.write_text(
            "All LFP windows failed QC, so physiological feature plots were skipped. "
            "QC summaries and derivative diagnostics were generated where inputs were available.\n",
            encoding="utf-8",
        )
        outputs["tables"].append(str(warning_path))
        print(f"[lfp plots warning] all feature windows failed QC; diagnostic outputs written to {out_dir}")
        return outputs
    valid_unscaled = valid.copy()
    xcol = _time_col(valid)
    valid[xcol] = _numeric(valid, xcol) / 60.0
    psd = pd.DataFrame()
    simple_bands = _simple_plot_bands(config)
    outputs["tables"].append(_write_simple_band_definitions(simple_bands, out_dir))
    simple_band_power = pd.DataFrame()
    if paths["psd"].exists() and paths["psd"].stat().st_size:
        psd = pd.read_csv(paths["psd"])
        psd = _filter_psd_to_valid_windows(psd, valid)
        simple_band_power = _simple_band_power_from_psd(psd, simple_bands)
        if not simple_band_power.empty:
            band_power_path = out_dir / "lfp_simple_narrow_band_power_by_window.csv"
            simple_band_power.to_csv(band_power_path, index=False)
            outputs["tables"].append(str(band_power_path))

    detail_dir = out_dir / "per_recording_psd"
    detail_dir.mkdir(parents=True, exist_ok=True)
    bandpower_timecourses = _plot_per_recording_bandpower_timecourses(valid_unscaled, detail_dir, config, band_power=simple_band_power)
    outputs["plots"].extend(bandpower_timecourses)
    if not psd.empty:
        detail_outputs = _plot_per_recording_psd_and_bandpower(psd, valid_unscaled, out_dir, config, band_power=simple_band_power)
        outputs["plots"].extend(detail_outputs.get("plots", []))
        outputs["tables"].extend(detail_outputs.get("tables", []))
        outputs["plots"].extend(_plot_simple_baseline_psd_overlay(psd, out_dir, config))
        outputs["plots"].extend(_plot_same_channel_baseline_post_psd(psd, out_dir, config))
    if mode == "simple":
        summary = _plot_summary_table(valid, out_dir)
        outputs["tables"].append(summary)
        print(f"[lfp simple plots] wrote {len(outputs['plots'])} plot(s) to {out_dir}")
        return outputs

    outputs["plots"].append(_plot_feature_time(valid, xcol, out_dir))
    outputs["plots"].append(_plot_broadband_change(valid, xcol, out_dir))
    band_plot = _plot_band_trajectories(valid, xcol, out_dir)
    if band_plot:
        outputs["plots"].append(band_plot)
    if not psd.empty:
        outputs["plots"].append(_plot_psd_overlay(psd, out_dir))
        diff_plot = _plot_treatment_minus_baseline_psd(psd, out_dir)
        if diff_plot:
            outputs["plots"].append(diff_plot)
    peak_plot = _plot_peak_features(valid, xcol, out_dir)
    if peak_plot:
        outputs["plots"].append(peak_plot)
    event_plot = _plot_event_features(valid, xcol, out_dir)
    if event_plot:
        outputs["plots"].append(event_plot)
    heatmap = _plot_channel_time_heatmap(valid, xcol, out_dir)
    if heatmap:
        outputs["plots"].append(heatmap)
    paired = _plot_paired_baseline_post(valid, out_dir)
    if paired:
        outputs["plots"].append(paired)
    if paths["spike_counts"] is not None and paths["spike_counts"].exists():
        aligned = _plot_aligned_spike_lfp(df, valid_unscaled, paths["spike_counts"], out_dir, paths.get("spike_events"), config)
        if aligned:
            outputs["plots"].append(aligned["plot"])
            outputs["tables"].append(aligned["table"])
    if paths["ppc"] is not None and paths["ppc"].exists():
        ppc_plot = _plot_ppc_timecourse(paths["ppc"], out_dir)
        if ppc_plot:
            outputs["plots"].append(ppc_plot)
    summary = _plot_summary_table(valid, out_dir)
    outputs["tables"].append(summary)
    print(f"[lfp plots] wrote {len(outputs['plots'])} plot(s) to {out_dir}")
    return outputs


def _plot_feature_time(df: pd.DataFrame, xcol: str, out_dir: Path) -> str:
    ycol = "primary_power_uV2"
    if ycol not in df.columns:
        ycol = "total_power_uV2"
    fig, ax = plt.subplots(figsize=(12, 4.8))
    label_col = _label_col(df)
    labels = list(dict.fromkeys(df[label_col].astype(str)))
    colors = _phase_colors(labels)
    group_cols = [col for col in ("preparation_id", "channel", label_col) if col in df.columns]
    for key, sub in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        label_value = str(key[-1])
        channel = sub["channel"].iloc[0] if "channel" in sub.columns else ""
        prep = sub["preparation_id"].iloc[0] if "preparation_id" in sub.columns else ""
        ax.plot(sub[xcol], _numeric(sub, ycol), marker="o", linewidth=1.6, markersize=3, color=colors.get(label_value), label=f"{prep} {channel} | {label_value}".strip())
    _add_segment_boundaries(ax, df)
    ax.set_title("LFP feature versus cumulative recorded time")
    ax.set_xlabel("Cumulative recorded time (min)")
    ax.set_ylabel(ycol.replace("_", " "))
    ax.legend(loc="best", fontsize=7, ncols=2)
    return _save(fig, out_dir, "lfp_feature_vs_experiment_time.png")


def _plot_broadband_change(df: pd.DataFrame, xcol: str, out_dir: Path) -> str:
    ycol = "primary_power_uV2_change_db" if "primary_power_uV2_change_db" in df.columns else "total_power_uV2_change_db"
    fig, ax = plt.subplots(figsize=(12, 4.6))
    label_col = _label_col(df)
    labels = list(dict.fromkeys(df[label_col].astype(str)))
    colors = _phase_colors(labels)
    for key, sub in df.groupby([col for col in ("preparation_id", "channel", label_col) if col in df.columns], dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        label_value = str(key[-1])
        channel = sub["channel"].iloc[0] if "channel" in sub.columns else ""
        prep = sub["preparation_id"].iloc[0] if "preparation_id" in sub.columns else ""
        ax.plot(sub[xcol], _numeric(sub, ycol), marker="o", linewidth=1.6, markersize=3, color=colors.get(label_value), label=f"{prep} {channel} | {label_value}".strip())
    ax.axhline(0, color="black", linewidth=1, alpha=0.45)
    _add_segment_boundaries(ax, df)
    ax.set_title("Baseline-normalized broadband LFP power")
    ax.set_xlabel("Cumulative recorded time (min)")
    ax.set_ylabel("Power change (dB)")
    ax.legend(loc="best", fontsize=7, ncols=2)
    return _save(fig, out_dir, "lfp_baseline_normalized_broadband_power.png")


def _plot_band_trajectories(df: pd.DataFrame, xcol: str, out_dir: Path) -> str | None:
    band_cols = [col for col in df.columns if col.startswith("band_") and col.endswith("_power_uV2_change_db")]
    ylabel = "Band power change (dB)"
    if not band_cols:
        band_cols = [col for col in df.columns if col.startswith("band_") and col.endswith("_power_uV2")]
        ylabel = "Band power (uV^2)"
    if not band_cols:
        return None
    fig, ax = plt.subplots(figsize=(12, 5.2))
    channel = str(df["channel"].dropna().iloc[0]) if "channel" in df.columns and not df["channel"].dropna().empty else ""
    plot_df = df[df["channel"].astype(str) == channel] if channel else df
    for col in band_cols[:8]:
        label = col.replace("band_", "").replace("_power_uV2_change_db", "").replace("_power_uV2", "")
        ax.plot(plot_df[xcol], _numeric(plot_df, col), marker="o", linewidth=1.4, markersize=2.8, label=label)
    if "change_db" in band_cols[0]:
        ax.axhline(0, color="black", linewidth=1, alpha=0.4)
    ax.set_title("Absolute and normalized band-power trajectories")
    _add_segment_boundaries(ax, plot_df)
    ax.set_xlabel("Cumulative recorded time (min)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=7, ncols=2)
    return _save(fig, out_dir, "lfp_band_power_trajectories.png")


def _prep_psd_summary(psd: pd.DataFrame) -> pd.DataFrame:
    psd = psd.copy()
    if "epoch_label" not in psd.columns:
        psd["epoch_label"] = psd.get("phase", "")
    psd["frequency_hz"] = _numeric(psd, "frequency_hz")
    psd["psd_db"] = _numeric(psd, "psd_db")
    group_cols = [col for col in ("experiment_id", "animal_id", "preparation_id", "channel", "epoch_label", "phase", "frequency_hz") if col in psd.columns]
    prep = psd.groupby(group_cols, dropna=False)["psd_db"].mean().reset_index()
    return prep


def _plot_psd_overlay(psd: pd.DataFrame, out_dir: Path) -> str:
    prep = _prep_psd_summary(psd)
    label_col = _label_col(prep)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    group_cols = [col for col in ("preparation_id", "channel", label_col) if col in prep.columns]
    groups = []
    for key, sub in prep.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        label_value = str(key[-1])
        prep_id = sub["preparation_id"].iloc[0] if "preparation_id" in sub.columns else ""
        channel = sub["channel"].iloc[0] if "channel" in sub.columns else ""
        trace_label = f"{prep_id} {channel} | {label_value}".strip()
        groups.append((trace_label, sub))
    colors = _series_colors([label for label, _sub in groups])
    for trace_label, sub in groups:
        freq = _numeric(sub, "frequency_hz").to_numpy(dtype=float)
        y = _numeric(sub, "psd_db").to_numpy(dtype=float)
        ax.plot(freq, y, color=colors.get(trace_label), alpha=0.78, linewidth=1.3, label=trace_label)
    ax.set_title("Absolute PSD by preparation, channel and epoch")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB uV^2/Hz)")
    ax.legend(loc="best", fontsize=7, ncols=2)
    return _save(fig, out_dir, "lfp_psd_overlays_by_phase.png")


def _plot_treatment_minus_baseline_psd(psd: pd.DataFrame, out_dir: Path) -> str | None:
    prep = _prep_psd_summary(psd)
    prep["phase_norm"] = prep["phase"].astype(str).str.strip().str.lower()
    if "baseline" not in set(prep["phase_norm"]):
        return None
    id_cols = [col for col in ("experiment_id", "animal_id", "preparation_id", "channel", "frequency_hz") if col in prep.columns]
    base = prep[prep["phase_norm"] == "baseline"][id_cols + ["psd_db"]].rename(columns={"psd_db": "baseline_psd_db"})
    rows = []
    for phase in sorted(set(prep["phase_norm"]) - {"baseline"}):
        other = prep[prep["phase_norm"] == phase]
        merged = other.merge(base, on=id_cols, how="inner")
        if merged.empty:
            continue
        merged["difference_db"] = merged["psd_db"] - merged["baseline_psd_db"]
        merged["comparison_phase"] = phase
        rows.append(merged)
    if not rows:
        return None
    diff = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    groups = []
    for key, sub in diff.groupby([col for col in ("preparation_id", "channel", "comparison_phase") if col in diff.columns]):
        if not isinstance(key, tuple):
            key = (key,)
        phase = str(key[-1])
        sub = sub.sort_values("frequency_hz")
        prep_id = sub["preparation_id"].iloc[0] if "preparation_id" in sub.columns else ""
        channel = sub["channel"].iloc[0] if "channel" in sub.columns else ""
        trace_label = f"{prep_id} {channel} | {phase}".strip()
        groups.append((trace_label, sub))
    colors = _series_colors([label for label, _sub in groups])
    for trace_label, sub in groups:
        ax.plot(_numeric(sub, "frequency_hz"), _numeric(sub, "difference_db"), linewidth=1.4, color=colors.get(trace_label), alpha=0.78, label=trace_label)
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax.set_title("Matched epoch-minus-baseline PSD")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Difference (dB)")
    ax.legend(loc="best")
    return _save(fig, out_dir, "lfp_treatment_minus_baseline_psd.png")


def _phase_norm_series(df: pd.DataFrame) -> pd.Series:
    if "phase" in df.columns:
        values = df["phase"]
    elif "epoch_label" in df.columns:
        values = df["epoch_label"]
    else:
        values = pd.Series("", index=df.index)
    return values.astype(str).str.strip().str.lower()


def _psd_trace_group_columns(psd: pd.DataFrame, *, include_channel: bool = True, include_phase: bool = True) -> list[str]:
    preferred = [
        "channel",
        "phase",
        "epoch_label",
        "segment_id",
        "recording_id",
        "source_file",
    ]
    if not include_channel:
        preferred = [col for col in preferred if col != "channel"]
    if not include_phase:
        preferred = [col for col in preferred if col not in {"phase", "epoch_label"}]
    return [col for col in preferred if col in psd.columns]


def _median_psd_spec(psd: pd.DataFrame) -> pd.DataFrame:
    if psd.empty:
        return pd.DataFrame()
    spec = psd.copy()
    spec["frequency_hz"] = _numeric(spec, "frequency_hz")
    spec["psd_uV2_per_hz"] = _numeric(spec, "psd_uV2_per_hz")
    spec = (
        spec.dropna(subset=["frequency_hz", "psd_uV2_per_hz"])
        .groupby("frequency_hz", dropna=False)["psd_uV2_per_hz"]
        .median()
        .reset_index()
        .sort_values("frequency_hz")
    )
    if spec.empty:
        return spec
    spec["psd_db"] = 10.0 * np.log10(pd.to_numeric(spec["psd_uV2_per_hz"], errors="coerce") + np.finfo(float).eps)
    return spec


def _scoped_psd_groups(psd: pd.DataFrame):
    scope_cols = [col for col in ("experiment_id", "animal_id", "preparation_id") if col in psd.columns]
    if not scope_cols:
        yield "all_recordings", "all_recordings", psd
        return
    for key, sub in psd.groupby(scope_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = pd.Series(dict(zip(scope_cols, key)))
        label = _plot_group_label(row, scope_cols) or "all_recordings"
        safe = "_".join(_safe_name(bit) for bit in key if str(bit).strip()) or "all_recordings"
        yield label, safe, sub


def _psd_trace_specs(psd: pd.DataFrame, trace_cols: list[str]) -> list[tuple[str, pd.DataFrame]]:
    specs = []
    grouped = [((), psd)] if not trace_cols else list(psd.groupby(trace_cols, dropna=False, sort=False))
    for trace_index, (key, sub) in enumerate(grouped, start=1):
        if not isinstance(key, tuple):
            key = (key,)
        row = sub.iloc[0] if not sub.empty else pd.Series(dtype=object)
        label = _plot_group_label(row, trace_cols) or "PSD trace"
        spec = _median_psd_spec(sub)
        if spec.empty:
            continue
        specs.append((f"{trace_index}. {label}", spec))
    return specs


def _plot_simple_baseline_psd_overlay(psd: pd.DataFrame, out_dir: Path, config: Mapping | None = None) -> list[str]:
    outputs = []
    if psd.empty or "frequency_hz" not in psd.columns or "psd_uV2_per_hz" not in psd.columns:
        return outputs
    baseline = psd[_phase_norm_series(psd).eq("baseline")].copy()
    if baseline.empty:
        return outputs
    compare_dir = out_dir / "simple_psd_comparisons"
    compare_dir.mkdir(parents=True, exist_ok=True)
    max_traces = int((config or {}).get("max_simple_psd_overlay_traces", 0) or 0)
    for scope_label, scope_safe, sub in _scoped_psd_groups(baseline):
        trace_cols = _psd_trace_group_columns(sub, include_channel=True, include_phase=True)
        specs = _psd_trace_specs(sub, trace_cols)
        if max_traces > 0:
            specs = specs[:max_traces]
        if not specs:
            continue
        colors = _series_colors([label for label, _spec in specs])
        fig, ax = plt.subplots(figsize=(9.8, 5.4))
        for label, spec in specs:
            ax.plot(_numeric(spec, "frequency_hz"), _numeric(spec, "psd_db"), linewidth=1.5, color=colors.get(label), label=label)
        ax.set_title(f"Simple baseline PSD overlay - {scope_label}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (dB uV^2/Hz)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=7, ncols=1)
        outputs.append(_save(fig, compare_dir, f"lfp_simple_baseline_psd_overlay_{scope_safe}.png"))
    return outputs


def _plot_same_channel_baseline_post_psd(psd: pd.DataFrame, out_dir: Path, config: Mapping | None = None) -> list[str]:
    outputs = []
    if psd.empty or "channel" not in psd.columns or "frequency_hz" not in psd.columns or "psd_uV2_per_hz" not in psd.columns:
        return outputs
    df = psd.copy()
    df["phase_norm"] = _phase_norm_series(df)
    df = df[df["phase_norm"].isin({"baseline", "post"})].copy()
    if df.empty:
        return outputs
    id_cols = [col for col in ("experiment_id", "animal_id", "preparation_id", "channel") if col in df.columns]
    if "channel" not in id_cols:
        return outputs
    compare_dir = out_dir / "simple_psd_comparisons"
    compare_dir.mkdir(parents=True, exist_ok=True)
    max_traces = int((config or {}).get("max_simple_psd_overlay_traces", 0) or 0)
    for key, sub in df.groupby(id_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        if not {"baseline", "post"}.issubset(set(sub["phase_norm"].astype(str))):
            continue
        title_bits = dict(zip(id_cols, key))
        title = " | ".join(str(title_bits.get(col, "")) for col in ("preparation_id", "channel") if str(title_bits.get(col, "")).strip())
        safe = "_".join(_safe_name(bit) for bit in key if str(bit).strip()) or "same_channel"
        trace_cols = ["phase_norm"] + [col for col in ("epoch_label", "segment_id", "recording_id", "source_file") if col in sub.columns]
        specs = _psd_trace_specs(sub, trace_cols)
        if max_traces > 0:
            specs = specs[:max_traces]
        if specs:
            colors = _series_colors([label for label, _spec in specs])
            fig, ax = plt.subplots(figsize=(9.8, 5.4))
            for label, spec in specs:
                ax.plot(_numeric(spec, "frequency_hz"), _numeric(spec, "psd_db"), linewidth=1.5, color=colors.get(label), label=label)
            ax.set_title(f"Same-channel baseline vs post PSD - {title}")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD (dB uV^2/Hz)")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=7, ncols=1)
            outputs.append(_save(fig, compare_dir, f"lfp_simple_same_channel_baseline_post_psd_{safe}.png"))

        baseline_spec = _median_psd_spec(sub[sub["phase_norm"] == "baseline"])
        post_specs = _psd_trace_specs(sub[sub["phase_norm"] == "post"], trace_cols)
        if max_traces > 0:
            post_specs = post_specs[:max_traces]
        diff_specs = []
        for label, post_spec in post_specs:
            merged = post_spec.merge(baseline_spec[["frequency_hz", "psd_db"]], on="frequency_hz", how="inner", suffixes=("_post", "_baseline"))
            if merged.empty:
                continue
            merged["difference_db"] = pd.to_numeric(merged["psd_db_post"], errors="coerce") - pd.to_numeric(merged["psd_db_baseline"], errors="coerce")
            diff_specs.append((label, merged))
        if diff_specs:
            colors = _series_colors([label for label, _spec in diff_specs])
            fig, ax = plt.subplots(figsize=(9.8, 5.4))
            for label, spec in diff_specs:
                ax.plot(_numeric(spec, "frequency_hz"), _numeric(spec, "difference_db"), linewidth=1.5, color=colors.get(label), label=f"{label} minus baseline")
            ax.axhline(0, color="black", linewidth=1, alpha=0.45)
            ax.set_title(f"Same-channel post-minus-baseline PSD - {title}")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD difference (dB)")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=7, ncols=1)
            outputs.append(_save(fig, compare_dir, f"lfp_simple_same_channel_post_minus_baseline_psd_{safe}.png"))
    return outputs


def _plot_group_label(row: pd.Series, columns: list[str]) -> str:
    values = []
    for col in columns:
        value = str(row.get(col, "")).strip()
        if value and value.lower() not in {"nan", "none"}:
            if col == "source_file":
                value = Path(value).name
            values.append(value)
    return " | ".join(dict.fromkeys(values))


def _single_recording_group_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["preparation_id", "segment_id", "recording_id", "source_file", "channel", "epoch_label", "phase"]
    cols = [col for col in preferred if col in df.columns]
    if "channel" not in cols and "channel" in df.columns:
        cols.append("channel")
    return cols


def _feature_group_subset(features: pd.DataFrame, row: pd.Series, group_cols: list[str]) -> pd.DataFrame:
    if features.empty:
        return features
    sub = features.copy()
    for col in group_cols:
        if col not in sub.columns or col not in row.index:
            continue
        value = row.get(col)
        text = str(value or "").strip()
        if not text or text.lower() in {"nan", "none"}:
            continue
        if col == "source_file":
            names = sub[col].astype(str).map(lambda item: Path(item).name)
            candidate = sub[(sub[col].astype(str) == text) | (names == Path(text).name)]
        else:
            candidate = sub[sub[col].astype(str) == text]
        if not candidate.empty:
            sub = candidate
    return sub


def _band_power_columns(features: pd.DataFrame) -> list[str]:
    excluded = {"total_power_uV2", "primary_power_uV2"}
    return [
        col
        for col in features.columns
        if col.startswith("band_") and col.endswith("_power_uV2") and col not in excluded
    ]


def _band_power_labels(features: pd.DataFrame, cols: list[str]) -> list[str]:
    labels = []
    for col in cols:
        label_col = col.replace("_power_uV2", "_label")
        label = ""
        if label_col in features.columns:
            values = features[label_col].astype(str).str.strip()
            values = values[values.ne("") & values.str.lower().ne("nan")]
            if not values.empty:
                label = str(values.iloc[0])
        if not label:
            label = col.replace("band_", "").replace("_power_uV2", "").replace("_", " ")
        labels.append(label)
    return labels


def _integrate_psd_band(freq: np.ndarray, power: np.ndarray, band: tuple[float, float]) -> float:
    freq = np.asarray(freq, dtype=float)
    power = np.asarray(power, dtype=float)
    keep = (
        np.isfinite(freq)
        & np.isfinite(power)
        & (freq >= float(band[0]))
        & (freq <= float(band[1]))
    )
    if np.count_nonzero(keep) < 2:
        return float("nan")
    return float(np.trapezoid(power[keep], freq[keep]))


def _psd_window_group_columns(psd: pd.DataFrame) -> list[str]:
    preferred = [
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
        "window_start_s",
        "window_end_s",
        "window_mid_s",
        "cumulative_window_start_s",
        "cumulative_window_end_s",
        "cumulative_window_mid_s",
    ]
    return [col for col in preferred if col in psd.columns]


def _simple_band_power_from_psd(
    psd: pd.DataFrame,
    bands: list[tuple[str, str, tuple[float, float], str]],
) -> pd.DataFrame:
    if psd.empty or "frequency_hz" not in psd.columns or "psd_uV2_per_hz" not in psd.columns:
        return pd.DataFrame()
    group_cols = _psd_window_group_columns(psd)
    if "channel" not in group_cols:
        return pd.DataFrame()
    rows = []
    for key, sub in psd.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))
        spec = sub.copy()
        spec["frequency_hz"] = _numeric(spec, "frequency_hz")
        spec["psd_uV2_per_hz"] = _numeric(spec, "psd_uV2_per_hz")
        spec = spec.dropna(subset=["frequency_hz", "psd_uV2_per_hz"]).sort_values("frequency_hz")
        if spec.empty:
            continue
        freq = spec["frequency_hz"].to_numpy(dtype=float)
        power = spec["psd_uV2_per_hz"].to_numpy(dtype=float)
        for band in bands:
            band_key, label, (lo, hi), reason = band
            band_power = _integrate_psd_band(freq, power, (lo, hi))
            rows.append(
                {
                    **base,
                    "band_key": band_key,
                    "band_label": label,
                    "band_display_label": _band_display_label(band),
                    "band_low_hz": float(lo),
                    "band_high_hz": float(hi),
                    "band_power_uV2": band_power,
                    "main_reason_to_examine": reason,
                }
            )
    return pd.DataFrame(rows)


def _plot_per_recording_bandpower_timecourses(
    features: pd.DataFrame,
    detail_dir: Path,
    config: Mapping | None = None,
    band_power: pd.DataFrame | None = None,
) -> list[str]:
    outputs: list[str] = []
    if band_power is not None:
        if band_power.empty:
            return outputs
        group_cols = _single_recording_group_columns(band_power)
        if not group_cols:
            return outputs
        time_col = "window_mid_s" if "window_mid_s" in band_power.columns else _time_col(band_power)
        max_groups = int((config or {}).get("max_per_recording_bandpower_plots", 0) or 0)
        grouped = list(band_power.groupby(group_cols, dropna=False, sort=False))
        skipped = 0
        for group_index, (_key, sub) in enumerate(grouped, start=1):
            if max_groups > 0 and group_index > max_groups:
                skipped += 1
                continue
            row = sub.iloc[0]
            label = _plot_group_label(row, group_cols)
            name_bits = [row.get(col, "") for col in ("preparation_id", "segment_id", "recording_id", "channel", "epoch_label") if col in row.index]
            safe = "_".join(_safe_name(bit) for bit in name_bits if str(bit).strip()) or f"group_{group_index:03d}"
            fig, ax = plt.subplots(figsize=(9.5, 5.2))
            band_labels = list(dict.fromkeys(sub["band_display_label"].astype(str)))
            colors = _series_colors(band_labels)
            plotted = 0
            for band_label, band_sub in sub.groupby("band_display_label", dropna=False, sort=False):
                band_sub = band_sub.sort_values(time_col)
                y = _numeric(band_sub, "band_power_uV2")
                if not y.notna().any():
                    continue
                display = str(band_label)
                ax.plot(_numeric(band_sub, time_col), y, marker="o", linewidth=1.3, markersize=3, color=colors.get(display), label=display)
                plotted += 1
            if not plotted:
                plt.close(fig)
                continue
            ax.set_xlabel("Recording time (s)")
            ax.set_ylabel("Band power (uV^2)")
            ax.set_title(f"Narrow-band LFP power - {label}")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=7, ncols=2)
            outputs.append(_save(fig, detail_dir, f"lfp_narrow_band_power_{safe}.png"))
        if skipped:
            note = detail_dir / "lfp_narrow_band_power_plots_skipped.txt"
            note.write_text(
                f"Skipped {skipped} band-power group(s) because max_per_recording_bandpower_plots={max_groups}.\n",
                encoding="utf-8",
            )
        return outputs

    if features.empty:
        return outputs
    band_cols = _band_power_columns(features)
    if not band_cols:
        return outputs
    time_col = "window_mid_s" if "window_mid_s" in features.columns else _time_col(features)
    group_cols = _single_recording_group_columns(features)
    if not group_cols:
        return outputs
    max_groups = int((config or {}).get("max_per_recording_bandpower_plots", 0) or 0)
    grouped = list(features.groupby(group_cols, dropna=False, sort=False))
    skipped = 0
    for group_index, (_key, sub) in enumerate(grouped, start=1):
        if max_groups > 0 and group_index > max_groups:
            skipped += 1
            continue
        row = sub.iloc[0]
        label = _plot_group_label(row, group_cols)
        name_bits = [row.get(col, "") for col in ("preparation_id", "segment_id", "recording_id", "channel", "epoch_label") if col in row.index]
        safe = "_".join(_safe_name(bit) for bit in name_bits if str(bit).strip()) or f"group_{group_index:03d}"
        sub = sub.sort_values(time_col).copy()
        x = _numeric(sub, time_col)
        labels = _band_power_labels(sub, band_cols)
        colors = _series_colors(labels)
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        plotted = 0
        for col, band_label in zip(band_cols, labels):
            y = _numeric(sub, col)
            if not y.notna().any():
                continue
            ax.plot(x, y, marker="o", linewidth=1.3, markersize=3, color=colors.get(str(band_label)), label=band_label)
            plotted += 1
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Recording time (s)")
        ax.set_ylabel("Band power (uV^2)")
        ax.set_title(f"Narrow-band LFP power - {label}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=7, ncols=2)
        outputs.append(_save(fig, detail_dir, f"lfp_narrow_band_power_{safe}.png"))
    if skipped:
        note = detail_dir / "lfp_narrow_band_power_plots_skipped.txt"
        note.write_text(
            f"Skipped {skipped} band-power group(s) because max_per_recording_bandpower_plots={max_groups}.\n",
            encoding="utf-8",
        )
    return outputs


def _plot_per_recording_psd_and_bandpower(
    psd: pd.DataFrame,
    features: pd.DataFrame,
    out_dir: Path,
    config: Mapping | None = None,
    band_power: pd.DataFrame | None = None,
) -> dict[str, list[str]]:
    outputs: dict[str, list[str]] = {"plots": [], "tables": []}
    if psd.empty or "frequency_hz" not in psd.columns or "psd_uV2_per_hz" not in psd.columns:
        return outputs
    detail_dir = out_dir / "per_recording_psd"
    detail_dir.mkdir(parents=True, exist_ok=True)
    group_cols = _single_recording_group_columns(psd)
    if not group_cols:
        return outputs
    max_groups = int((config or {}).get("max_per_recording_psd_plots", 0) or 0)
    grouped = list(psd.groupby(group_cols, dropna=False, sort=False))
    skipped = 0
    summary_rows = []
    for group_index, (key, sub) in enumerate(grouped, start=1):
        if max_groups > 0 and group_index > max_groups:
            skipped += 1
            continue
        sub = sub.copy()
        sub["frequency_hz"] = _numeric(sub, "frequency_hz")
        sub["psd_uV2_per_hz"] = _numeric(sub, "psd_uV2_per_hz")
        spec = (
            sub.dropna(subset=["frequency_hz", "psd_uV2_per_hz"])
            .groupby("frequency_hz", dropna=False)["psd_uV2_per_hz"]
            .median()
            .reset_index()
            .sort_values("frequency_hz")
        )
        if spec.empty:
            continue
        row = sub.iloc[0]
        label = _plot_group_label(row, group_cols)
        name_bits = [row.get(col, "") for col in ("preparation_id", "segment_id", "recording_id", "channel", "epoch_label") if col in row.index]
        safe = "_".join(_safe_name(bit) for bit in name_bits if str(bit).strip())
        if not safe:
            safe = f"group_{group_index:03d}"
        freq = pd.to_numeric(spec["frequency_hz"], errors="coerce")
        power = pd.to_numeric(spec["psd_uV2_per_hz"], errors="coerce")
        psd_db = 10.0 * np.log10(power + np.finfo(float).eps)

        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        ax.plot(freq, psd_db, linewidth=1.5)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (dB uV^2/Hz)")
        ax.set_title(f"Welch PSD (dB) - {label}")
        ax.grid(True, alpha=0.25)
        outputs["plots"].append(_save(fig, detail_dir, f"lfp_psd_db_{safe}.png"))

        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        ax.plot(freq, power, linewidth=1.5)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (uV^2/Hz)")
        ax.set_title(f"Welch PSD (linear) - {label}")
        ax.grid(True, alpha=0.25)
        outputs["plots"].append(_save(fig, detail_dir, f"lfp_psd_linear_{safe}.png"))

        labels = []
        values = []
        if band_power is not None:
            if not band_power.empty:
                band_sub = _feature_group_subset(band_power, row, group_cols)
                if not band_sub.empty and {"band_key", "band_display_label", "band_power_uV2"}.issubset(band_sub.columns):
                    grouped_bands = (
                        band_sub.groupby(["band_key", "band_display_label"], dropna=False, sort=False)["band_power_uV2"]
                        .median()
                        .reset_index()
                    )
                    labels = grouped_bands["band_display_label"].astype(str).tolist()
                    values = pd.to_numeric(grouped_bands["band_power_uV2"], errors="coerce").tolist()
        else:
            feature_sub = _feature_group_subset(features, row, group_cols)
            band_cols = _band_power_columns(feature_sub)
            if band_cols:
                labels = _band_power_labels(feature_sub, band_cols)
                values = [pd.to_numeric(feature_sub[col], errors="coerce").median() for col in band_cols]
        finite = [idx for idx, value in enumerate(values) if np.isfinite(value)]
        if finite:
            labels = [labels[idx] for idx in finite]
            values = [float(values[idx]) for idx in finite]
            colors = _series_colors(labels)
            fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(labels) + 4), 4.8))
            ax.bar(np.arange(len(values)), values, color=[colors.get(str(label)) for label in labels], alpha=0.85)
            ax.set_xticks(np.arange(len(values)), labels, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("Median band power (uV^2)")
            ax.set_title(f"Band-power bars - {label}")
            ax.grid(True, axis="y", alpha=0.25)
            outputs["plots"].append(_save(fig, detail_dir, f"lfp_bandpower_bars_{safe}.png"))

        summary_rows.append(
            {
                "group_index": group_index,
                "label": label,
                "n_psd_rows": int(len(sub)),
                "n_frequency_bins": int(spec["frequency_hz"].nunique()),
                "plots_written": int(len(outputs["plots"])),
            }
        )
    summary_path = detail_dir / "lfp_per_recording_psd_plot_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    outputs["tables"].append(str(summary_path))
    if skipped:
        note = detail_dir / "lfp_per_recording_psd_plots_skipped.txt"
        note.write_text(
            f"Skipped {skipped} PSD group(s) because max_per_recording_psd_plots={max_groups}.\n",
            encoding="utf-8",
        )
        outputs["tables"].append(str(note))
    return outputs


def _plot_peak_features(df: pd.DataFrame, xcol: str, out_dir: Path) -> str | None:
    cols = [col for col in ("dominant_peak_frequency_hz", "peak_power_over_aperiodic_db") if col in df.columns]
    if not cols:
        return None
    fig, axes = plt.subplots(len(cols), 1, figsize=(11, 3.4 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        for channel, sub in df.groupby("channel", dropna=False, sort=False):
            ax.plot(sub[xcol], _numeric(sub, col), marker="o", linewidth=1.4, markersize=3, label=str(channel))
        ax.set_ylabel(col.replace("_", " "))
        ax.legend(loc="best", fontsize=7, ncols=2)
    _add_segment_boundaries(axes[-1], df)
    axes[-1].set_xlabel("Cumulative recorded time (min)")
    axes[0].set_title("Peak frequency and peak-over-aperiodic power")
    return _save(fig, out_dir, "lfp_peak_frequency_and_peak_over_aperiodic.png")


def _plot_event_features(df: pd.DataFrame, xcol: str, out_dir: Path) -> str | None:
    cols = [
        col
        for col in ("oscillatory_event_rate_per_min", "oscillatory_event_duration_s", "oscillatory_event_occupancy")
        if col in df.columns
    ]
    if not cols:
        return None
    fig, axes = plt.subplots(len(cols), 1, figsize=(11, 3.2 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        for channel, sub in df.groupby("channel", dropna=False, sort=False):
            ax.plot(sub[xcol], _numeric(sub, col), marker="o", linewidth=1.4, markersize=3, label=str(channel))
        ax.set_ylabel(col.replace("_", " "))
    axes[0].legend(loc="best", fontsize=7, ncols=2)
    _add_segment_boundaries(axes[-1], df)
    axes[-1].set_xlabel("Cumulative recorded time (min)")
    axes[0].set_title("Oscillatory-event rate, duration and occupancy")
    return _save(fig, out_dir, "lfp_oscillatory_event_metrics.png")


def _plot_channel_time_heatmap(df: pd.DataFrame, xcol: str, out_dir: Path) -> str | None:
    value_col = "primary_power_uV2_change_db" if "primary_power_uV2_change_db" in df.columns else "primary_power_uV2"
    if value_col not in df.columns:
        return None
    pivot = df.pivot_table(index="channel", columns=xcol, values=value_col, aggfunc="median")
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, max(3, 0.35 * len(pivot) + 2.5)))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title("Channel-by-time LFP heat map")
    ax.set_xlabel("Cumulative recorded time (min)")
    ax.set_ylabel("Channel")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str))
    ticks = np.linspace(0, len(pivot.columns) - 1, min(8, len(pivot.columns))).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{pivot.columns[i]:.1f}" for i in ticks], rotation=45, ha="right")
    fig.colorbar(im, ax=ax, label=value_col.replace("_", " "))
    return _save(fig, out_dir, "lfp_channel_time_heatmap.png")


def _plot_paired_baseline_post(df: pd.DataFrame, out_dir: Path) -> str | None:
    feature = "primary_power_uV2"
    if feature not in df.columns:
        feature = "total_power_uV2"
    phase = df["phase"].astype(str).str.strip().str.lower()
    if "baseline" not in set(phase):
        return None
    id_cols = [col for col in ("experiment_id", "animal_id", "preparation_id", "channel") if col in df.columns]
    prep = df.assign(phase_norm=phase).groupby(id_cols + ["phase_norm"], dropna=False)[feature].median().reset_index()
    base = prep[prep["phase_norm"] == "baseline"][id_cols + [feature]].rename(columns={feature: "baseline"})
    rows = []
    for phase_name in sorted(set(prep["phase_norm"]) - {"baseline"}):
        other = prep[prep["phase_norm"] == phase_name][id_cols + [feature]].rename(columns={feature: "phase_value"})
        merged = base.merge(other, on=id_cols)
        merged["phase"] = phase_name
        rows.append(merged)
    if not rows:
        return None
    paired = pd.concat(rows, ignore_index=True)
    phases = sorted(paired["phase"].unique())
    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(phases) + 4), 5))
    x = 0
    xticks = []
    xticklabels = []
    for phase_name in phases:
        sub = paired[paired["phase"] == phase_name]
        for _idx, row in sub.iterrows():
            ax.plot([x, x + 1], [row["baseline"], row["phase_value"]], color="#6b7280", alpha=0.6, marker="o")
        xticks.extend([x, x + 1])
        xticklabels.extend(["Baseline", str(phase_name)])
        x += 3
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=35, ha="right")
    ax.set_ylabel(feature.replace("_", " "))
    ax.set_title("Per-preparation paired baseline/post plots")
    return _save(fig, out_dir, "lfp_paired_baseline_post_by_preparation.png")


def _plot_qc(qc: pd.DataFrame, out_dir: Path) -> str:
    qc = qc.copy()
    xcol = _time_col(qc)
    qc[xcol] = _numeric(qc, xcol) / 60.0
    cols = [
        "valid_fraction",
        "filtered_finite_sample_fraction",
        "large_derivative_fraction",
        "clipped_fraction",
        "flatline_fraction",
        "line_noise_ratio",
        "large_derivative_or_step_score",
    ]
    cols = [col for col in cols if col in qc.columns]
    fig, axes = plt.subplots(len(cols) + 1, 1, figsize=(11, max(4.2, 2.25 * (len(cols) + 1))), sharex=True)
    axes = np.atleast_1d(axes)
    signal_failed = ~_as_bool(qc["signal_qc_pass"]) if "signal_qc_pass" in qc.columns else (~_as_bool(qc["qc_pass"]) if "qc_pass" in qc.columns else pd.Series(False, index=qc.index))
    feature_failed = ~_as_bool(qc["feature_qc_pass"]) if "feature_qc_pass" in qc.columns else pd.Series(False, index=qc.index)
    derivative_warning = qc.get("derivative_qc_status", pd.Series("", index=qc.index)).astype(str).str.lower().eq("warning")
    derivative_failed = (
        qc.get("derivative_qc_status", pd.Series("", index=qc.index)).astype(str).str.lower().eq("fail")
        | qc.get("qc_reason", pd.Series("", index=qc.index)).astype(str).str.contains("large_derivative_artifact_burden", na=False)
    )
    feature_reason = qc.get("feature_qc_reason", pd.Series("", index=qc.index)).astype(str)
    filtered_failed = feature_reason.str.contains("filtered_signal|filtered_finite", regex=True, na=False)
    psd_failed = feature_reason.str.contains("psd|power", regex=True, na=False)
    for ax, col in zip(axes[:-1], cols):
        ax.plot(qc[xcol], _numeric(qc, col), marker="o", linestyle="-", markersize=2.8, linewidth=1)
        overlays = [
            (signal_failed, "tab:red", "signal QC fail", "x"),
            (feature_failed, "tab:purple", "feature QC fail", "s"),
            (derivative_warning, "tab:orange", "derivative warning", "^"),
            (derivative_failed, "darkred", "derivative burden fail", "D"),
            (filtered_failed, "black", "filtered signal fail", "v"),
            (psd_failed, "tab:brown", "PSD/power fail", "P"),
        ]
        seen = set()
        for mask, color, label, marker in overlays:
            if not bool(mask.any()):
                continue
            plot_label = label if label not in seen else None
            seen.add(label)
            ax.scatter(qc.loc[mask, xcol], _numeric(qc.loc[mask], col), color=color, marker=marker, s=24, label=plot_label)
        ax.set_ylabel(col.replace("_", " "))
    category_counts = {
        "signal fail": int(signal_failed.sum()),
        "feature fail": int(feature_failed.sum()),
        "deriv warn": int(derivative_warning.sum()),
        "deriv fail": int(derivative_failed.sum()),
        "filtered fail": int(filtered_failed.sum()),
        "PSD fail": int(psd_failed.sum()),
    }
    axes[-1].bar(list(category_counts.keys()), list(category_counts.values()), color=["tab:red", "tab:purple", "tab:orange", "darkred", "black", "tab:brown"])
    axes[-1].set_ylabel("Windows")
    axes[-1].set_title("QC decision categories")
    axes[-1].tick_params(axis="x", rotation=25)
    axes[0].set_title("LFP QC: signal failures, feature failures, derivative warnings and PSD/filter diagnostics")
    boundary_axis = axes[-2] if len(axes) > 1 else axes[-1]
    _add_segment_boundaries(boundary_axis, qc)
    boundary_axis.set_xlabel("Cumulative recorded time (min)")
    if any(mask.any() for mask in (signal_failed, feature_failed, derivative_warning, derivative_failed, filtered_failed, psd_failed)):
        axes[0].legend(loc="best", fontsize=8)
    return _save(fig, out_dir, "lfp_qc_overview.png")


def _plot_derivative_diagnostics(events_path: Path, qc: pd.DataFrame, out_dir: Path, spike_events_path: Path | None = None) -> list[str]:
    if not events_path.exists() or not events_path.stat().st_size:
        return []
    events = pd.read_csv(events_path)
    if events.empty:
        return []
    spike_events = pd.DataFrame()
    if spike_events_path is not None and Path(spike_events_path).exists() and Path(spike_events_path).stat().st_size:
        try:
            spike_events = pd.read_csv(spike_events_path)
        except Exception:
            spike_events = pd.DataFrame()
    diag_dir = out_dir / "derivative_qc"
    diag_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    outputs.extend(_plot_derivative_summary(events, qc, diag_dir))
    outputs.extend(_plot_derivative_rasters(events, diag_dir))
    group_cols = [col for col in ("preparation_id", "segment_id", "source_file", "channel") if col in events.columns]
    for key, sub in events.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = sub.iloc[0]
        source = row.get("source_file", "")
        channel = str(row.get("channel", ""))
        try:
            t, raw = _read_raw_trace(source, channel)
        except Exception:
            continue
        spike_sub = _matching_spikes_for_diagnostic(spike_events, row, channel)
        outputs.append(_plot_derivative_segment_overview(t, raw, sub, diag_dir, spike_sub))
        zoom = _plot_derivative_zoom(t, raw, sub, diag_dir, spike_sub)
        if zoom:
            outputs.append(zoom)
        wave = _plot_derivative_triggered_waveforms(t, raw, sub, diag_dir)
        if wave:
            outputs.append(wave)
    return outputs


def _matching_spikes_for_diagnostic(spikes: pd.DataFrame, row: pd.Series, channel: str) -> pd.DataFrame:
    if spikes.empty or "spike_time_s" not in spikes.columns or "channel" not in spikes.columns:
        return pd.DataFrame()
    out = spikes[spikes["channel"].astype(str) == str(channel)].copy()
    prep = str(row.get("preparation_id", "")).strip()
    if "preparation_id" not in out.columns or not prep:
        return pd.DataFrame()
    out = out[out["preparation_id"].astype(str) == prep]
    segment = str(row.get("segment_id", "")).strip()
    recording_id = str(row.get("recording_id", "")).strip()
    if "segment_id" in out.columns and segment:
        out = out[out["segment_id"].astype(str) == segment]
    elif "recording_id" in out.columns and recording_id:
        out = out[out["recording_id"].astype(str).isin({recording_id, segment})]
    else:
        return pd.DataFrame()
    return out


def _event_match_tolerance(events: pd.DataFrame) -> float:
    if "spike_match_tolerance_s" not in events.columns:
        return 0.002
    values = pd.to_numeric(events["spike_match_tolerance_s"], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty and np.isfinite(values.iloc[0]) else 0.002


def _spikes_without_derivative(events: pd.DataFrame, spikes: pd.DataFrame) -> np.ndarray:
    if spikes.empty or "spike_time_s" not in spikes.columns or "event_peak_s" not in events.columns:
        return np.asarray([], dtype=float)
    spike_times = pd.to_numeric(spikes["spike_time_s"], errors="coerce").dropna().to_numpy(dtype=float)
    peaks = pd.to_numeric(events["event_peak_s"], errors="coerce").dropna().to_numpy(dtype=float)
    if spike_times.size == 0:
        return spike_times
    if peaks.size == 0:
        return spike_times
    tolerance = _event_match_tolerance(events)
    unmatched = [time for time in spike_times if np.min(np.abs(peaks - time)) > tolerance]
    return np.asarray(unmatched, dtype=float)


def _event_style(event_row: pd.Series) -> tuple[str, str]:
    matched = _as_bool(pd.Series([event_row.get("matched_to_spike_within_tolerance", False)])).iloc[0]
    if bool(matched):
        return "tab:green", "matched derivative/spike"
    return "tab:orange", "derivative only"


def _plot_derivative_segment_overview(t: np.ndarray, raw: np.ndarray, events: pd.DataFrame, out_dir: Path, spikes: pd.DataFrame | None = None) -> str:
    plot_sel = _plot_stride_indices(t.size)
    tp = t[plot_sel]
    rawp = raw[plot_sel]
    _, deriv, z = _derivative_z_trace(tp, rawp)
    zthr_values = pd.to_numeric(events.get("large_derivative_z_threshold", pd.Series([8.0])), errors="coerce").dropna()
    zthr = float(zthr_values.iloc[0]) if not zthr_values.empty and np.isfinite(zthr_values.iloc[0]) else 8.0
    title_bits = [str(events[col].iloc[0]) for col in ("preparation_id", "segment_id", "channel") if col in events.columns]
    fig, axes = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True)
    axes[0].plot(tp, rawp, color="#111827", linewidth=0.8)
    axes[0].set_ylabel("Raw voltage")
    finite_raw = rawp[np.isfinite(rawp)]
    raw_fill = np.nan_to_num(rawp, nan=float(np.nanmedian(finite_raw)) if finite_raw.size else 0.0)
    seen = set()
    for _idx, row in events.iterrows():
        color, label = _event_style(row)
        label = label if label not in seen else None
        if label:
            seen.add(label)
        if np.isfinite(pd.to_numeric(pd.Series([row.get("event_start_s")]), errors="coerce").iloc[0]) and np.isfinite(pd.to_numeric(pd.Series([row.get("event_end_s")]), errors="coerce").iloc[0]):
            axes[0].axvspan(float(row["event_start_s"]), float(row["event_end_s"]), color=color, alpha=0.10)
        axes[0].axvline(float(row["event_peak_s"]), color=color, alpha=0.55, linewidth=1, label=label)
        matched = _as_bool(pd.Series([row.get("matched_to_spike_within_tolerance", False)])).iloc[0]
        if bool(matched) and np.isfinite(pd.to_numeric(pd.Series([row.get("nearest_spike_time_s")]), errors="coerce").iloc[0]):
            axes[0].scatter(float(row["nearest_spike_time_s"]), np.interp(float(row["nearest_spike_time_s"]), tp, raw_fill), marker="x", color="tab:green", s=30)
    spike_only = _spikes_without_derivative(events, spikes if spikes is not None else pd.DataFrame())
    if spike_only.size:
        axes[0].scatter(spike_only, np.interp(spike_only, tp, raw_fill), marker="v", color="tab:blue", s=18, label="spike only")
    axes[1].plot(tp, deriv, color="tab:blue", linewidth=0.75)
    axes[1].set_ylabel("dV/dt")
    axes[2].plot(tp, z, color="tab:purple", linewidth=0.75)
    axes[2].axhline(zthr, color="red", linestyle="--", linewidth=1, label="+/- robust z threshold")
    axes[2].axhline(-zthr, color="red", linestyle="--", linewidth=1)
    axes[2].scatter(pd.to_numeric(events["event_peak_s"], errors="coerce"), pd.to_numeric(events["max_robust_derivative_z_score"], errors="coerce"), color="tab:orange", s=12)
    axes[2].set_ylabel("Robust derivative z")
    axes[2].set_xlabel("Recording-segment time (s)")
    axes[0].legend(loc="best", fontsize=8)
    axes[2].legend(loc="best", fontsize=8)
    axes[0].set_title("Derivative QC overview: " + " | ".join(title_bits))
    name = "derivative_qc_overview_" + "_".join(_safe_name(bit) for bit in title_bits) + ".png"
    return _save(fig, out_dir, name)


def _plot_derivative_zoom(t: np.ndarray, raw: np.ndarray, events: pd.DataFrame, out_dir: Path, spikes: pd.DataFrame | None = None) -> str | None:
    if events.empty:
        return None
    peak = float(pd.to_numeric(events["event_peak_s"], errors="coerce").dropna().iloc[0])
    width = 1.0
    keep = (t >= peak - width / 2.0) & (t <= peak + width / 2.0)
    if np.count_nonzero(keep) < 4:
        return None
    tk = t[keep]
    rawk = raw[keep]
    fs = 1.0 / np.nanmedian(np.diff(t[np.isfinite(t)])) if np.count_nonzero(np.isfinite(t)) > 2 else np.nan
    lfp = _safe_filter_trace(rawk, fs, "lfp", t=tk)
    spike_band = _safe_filter_trace(rawk, fs, "spike", t=tk)
    _, _, z = _derivative_z_trace(tk, rawk)
    panels = 3 + int(spike_band is not None)
    fig, axes = plt.subplots(panels, 1, figsize=(12, 2.3 * panels), sharex=True)
    axes = np.atleast_1d(axes)
    axes[0].plot(tk, rawk, color="#111827", linewidth=0.8)
    axes[0].set_ylabel("Raw")
    axes[1].plot(tk, lfp if lfp is not None else rawk, color="tab:green", linewidth=0.85)
    axes[1].set_ylabel("LFP/low-pass")
    next_idx = 2
    if spike_band is not None:
        axes[next_idx].plot(tk, spike_band, color="tab:blue", linewidth=0.75)
        axes[next_idx].set_ylabel("Spike band")
        next_idx += 1
    axes[next_idx].plot(tk, z, color="tab:purple", linewidth=0.75)
    axes[next_idx].set_ylabel("Derivative z")
    for ax in axes:
        for _idx, row in events.iterrows():
            ep = float(row["event_peak_s"])
            if tk[0] <= ep <= tk[-1]:
                color, _label = _event_style(row)
                if np.isfinite(pd.to_numeric(pd.Series([row.get("event_start_s")]), errors="coerce").iloc[0]) and np.isfinite(pd.to_numeric(pd.Series([row.get("event_end_s")]), errors="coerce").iloc[0]):
                    ax.axvspan(float(row["event_start_s"]), float(row["event_end_s"]), color=color, alpha=0.10)
                ax.axvline(ep, color=color, alpha=0.55, linewidth=1)
        spike_only = _spikes_without_derivative(events, spikes if spikes is not None else pd.DataFrame())
        for st in spike_only:
            if tk[0] <= st <= tk[-1]:
                ax.axvline(float(st), color="tab:blue", linestyle=":", alpha=0.6, linewidth=1)
        ax.set_xlim(tk[0], tk[-1])
    axes[-1].set_xlabel("Recording-segment time (s)")
    title_bits = [str(events[col].iloc[0]) for col in ("preparation_id", "segment_id", "channel") if col in events.columns]
    axes[0].set_title("Aligned raw/LFP/spike-band derivative zoom")
    name = "derivative_qc_zoom_" + "_".join(_safe_name(bit) for bit in [*title_bits, f"{peak:.3f}s"]) + ".png"
    return _save(fig, out_dir, name)


def _plot_derivative_triggered_waveforms(t: np.ndarray, raw: np.ndarray, events: pd.DataFrame, out_dir: Path) -> str | None:
    peaks = pd.to_numeric(events["event_peak_s"], errors="coerce").dropna().to_numpy(dtype=float)
    if peaks.size == 0 or t.size < 8:
        return None
    fs = 1.0 / np.nanmedian(np.diff(t[np.isfinite(t)])) if np.count_nonzero(np.isfinite(t)) > 2 else np.nan
    if not np.isfinite(fs) or fs <= 0:
        return None
    half = int(round(0.01 * fs))
    if half < 2:
        return None
    rel = np.arange(-half, half + 1) / fs * 1000.0
    groups = {
        "all": events,
        "matched": events[_as_bool(events["matched_to_spike_within_tolerance"])] if "matched_to_spike_within_tolerance" in events.columns else events.iloc[0:0],
        "unmatched": events[~_as_bool(events["matched_to_spike_within_tolerance"])] if "matched_to_spike_within_tolerance" in events.columns else events,
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (label, sub) in zip(axes, groups.items()):
        waves = []
        for peak in pd.to_numeric(sub["event_peak_s"], errors="coerce").dropna():
            idx = int(np.searchsorted(t, float(peak)))
            if idx - half < 0 or idx + half + 1 > raw.size:
                continue
            wave = raw[idx - half : idx + half + 1]
            if np.isfinite(wave).all():
                waves.append(wave - np.nanmedian(wave))
        if waves:
            arr = np.vstack(waves)
            for wave in arr[:100]:
                ax.plot(rel, wave, color="#9ca3af", alpha=0.18, linewidth=0.7)
            med = np.nanmedian(arr, axis=0)
            q1 = np.nanpercentile(arr, 25, axis=0)
            q3 = np.nanpercentile(arr, 75, axis=0)
            ax.plot(rel, med, color="tab:red", linewidth=2)
            ax.fill_between(rel, q1, q3, color="tab:red", alpha=0.18)
        ax.axvline(0, color="black", alpha=0.4, linewidth=1)
        ax.set_title(f"{label} (n={len(waves)})")
        ax.set_xlabel("Time from derivative peak (ms)")
    axes[0].set_ylabel("Centered raw voltage")
    title_bits = [str(events[col].iloc[0]) for col in ("preparation_id", "segment_id", "channel") if col in events.columns]
    fig.suptitle("Derivative-triggered raw waveforms")
    name = "derivative_triggered_waveforms_" + "_".join(_safe_name(bit) for bit in title_bits) + ".png"
    return _save(fig, out_dir, name)


def _plot_derivative_rasters(events: pd.DataFrame, out_dir: Path) -> list[str]:
    outputs = []
    if events.empty or "event_peak_s" not in events.columns:
        return outputs
    group_cols = [col for col in ("preparation_id", "segment_id") if col in events.columns]
    if not group_cols or "channel" not in events.columns:
        return outputs
    for key, sub in events.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        channels = list(dict.fromkeys(sub["channel"].astype(str)))
        channel_to_y = {channel: idx for idx, channel in enumerate(channels)}
        fig, ax = plt.subplots(figsize=(12, max(3, 0.28 * len(channels) + 2)))
        y = sub["channel"].astype(str).map(channel_to_y)
        color = pd.to_numeric(sub.get("coincident_derivative_channel_fraction", pd.Series(0, index=sub.index)), errors="coerce")
        sc = ax.scatter(pd.to_numeric(sub["event_peak_s"], errors="coerce"), y, c=color, cmap="viridis", s=18, vmin=0, vmax=1)
        ax.set_yticks(np.arange(len(channels)))
        ax.set_yticklabels(channels)
        ax.set_xlabel("Recording-segment time (s)")
        ax.set_ylabel("Channel")
        ax.set_title("Multichannel derivative-event coincidence raster")
        fig.colorbar(sc, ax=ax, label="Coincident channel fraction")
        name = "derivative_multichannel_raster_" + "_".join(_safe_name(bit) for bit in key) + ".png"
        outputs.append(_save(fig, out_dir, name))
    return outputs


def _plot_derivative_summary(events: pd.DataFrame, qc: pd.DataFrame, out_dir: Path) -> list[str]:
    outputs = []
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    axes = axes.ravel()
    label_col = "epoch_label" if "epoch_label" in events.columns else "phase"
    if not qc.empty and "large_derivative_event_rate_per_min" in qc.columns:
        q = qc.copy()
        qlabel = "epoch_label" if "epoch_label" in q.columns else "phase"
        grouped = q.groupby([col for col in ("preparation_id", "channel", qlabel) if col in q.columns], dropna=False)["large_derivative_event_rate_per_min"].median().reset_index(name="event_rate_per_min")
        grouped["label"] = grouped[[col for col in ("preparation_id", "channel", qlabel) if col in grouped.columns]].astype(str).agg(" | ".join, axis=1)
        axes[0].bar(np.arange(len(grouped)), grouped["event_rate_per_min"])
        axes[0].set_xticks(np.arange(len(grouped)))
        axes[0].set_xticklabels(grouped["label"], rotation=60, ha="right", fontsize=7)
        axes[0].set_ylabel("Events/min")
        axes[0].set_title("Derivative-event rate by preparation/channel/epoch")
    elif label_col in events.columns:
        grouped = events.groupby([col for col in ("preparation_id", "channel", label_col) if col in events.columns], dropna=False).size().reset_index(name="event_count")
        grouped["label"] = grouped[[col for col in ("preparation_id", "channel", label_col) if col in grouped.columns]].astype(str).agg(" | ".join, axis=1)
        axes[0].bar(np.arange(len(grouped)), grouped["event_count"])
        axes[0].set_xticks(np.arange(len(grouped)))
        axes[0].set_xticklabels(grouped["label"], rotation=60, ha="right", fontsize=7)
        axes[0].set_ylabel("Derivative events")
        axes[0].set_title("Derivative-event count by preparation/channel/epoch")
    if not qc.empty and "large_derivative_fraction" in qc.columns:
        q = qc.copy()
        qlabel = "epoch_label" if "epoch_label" in q.columns else "phase"
        summary = q.groupby([col for col in ("preparation_id", "channel", qlabel) if col in q.columns], dropna=False)["large_derivative_fraction"].median().reset_index()
        summary["label"] = summary[[col for col in ("preparation_id", "channel", qlabel) if col in summary.columns]].astype(str).agg(" | ".join, axis=1)
        axes[1].bar(np.arange(len(summary)), summary["large_derivative_fraction"])
        axes[1].set_xticks(np.arange(len(summary)))
        axes[1].set_xticklabels(summary["label"], rotation=60, ha="right", fontsize=7)
        axes[1].set_ylabel("Derivative fraction")
        axes[1].set_title("Median derivative fraction")
    if "matched_to_spike_within_tolerance" in events.columns and label_col in events.columns:
        match_df = events.copy()
        match_df["matched_to_spike_within_tolerance"] = _as_bool(match_df["matched_to_spike_within_tolerance"])
        match = match_df.groupby(label_col)["matched_to_spike_within_tolerance"].mean().reset_index()
        axes[2].bar(match[label_col].astype(str), 100.0 * match["matched_to_spike_within_tolerance"])
        axes[2].set_ylabel("% matched to spikes")
        axes[2].set_title("Derivative/spike coincidence")
    axes[3].scatter(pd.to_numeric(events.get("event_duration_s", pd.Series(dtype=float)), errors="coerce"), pd.to_numeric(events.get("local_raw_voltage_amplitude_uv", pd.Series(dtype=float)), errors="coerce"), alpha=0.65)
    axes[3].set_xlabel("Duration (s)")
    axes[3].set_ylabel("Local raw amplitude")
    axes[3].set_title("Derivative duration versus amplitude")
    axes[4].hist(pd.to_numeric(events.get("coincident_derivative_channel_fraction", pd.Series(dtype=float)), errors="coerce").dropna(), bins=np.linspace(0, 1, 11))
    axes[4].set_xlabel("Coincident channel fraction")
    axes[4].set_title("Channel-coincidence distribution")
    if not qc.empty and "qc_reason" in qc.columns:
        counts = qc["qc_reason"].astype(str).value_counts()
        axes[5].bar(np.arange(len(counts)), counts.values)
        axes[5].set_xticks(np.arange(len(counts)))
        axes[5].set_xticklabels(counts.index, rotation=60, ha="right", fontsize=7)
        axes[5].set_title("QC pass/fail counts by reason")
    fig.suptitle("Derivative QC summary")
    outputs.append(_save(fig, out_dir, "derivative_qc_summary.png"))
    return outputs


def _standardise_spike_counts(path: Path) -> pd.DataFrame:
    spike = pd.read_csv(path)
    rename = {}
    for col in spike.columns:
        key = str(col).strip().lower()
        if key in {"epoch", "phase"}:
            rename[col] = "epoch_label"
        elif key in {"spikes", "spike_counts", "count", "n_spikes"}:
            rename[col] = "spike_count"
    return spike.rename(columns=rename)


def _source_matches(series: pd.Series, value: object) -> pd.Series:
    text = str(value or "").strip()
    if not text:
        return pd.Series(True, index=series.index)
    names = series.astype(str).map(lambda item: Path(item).name)
    return (series.astype(str) == text) | (names == Path(text).name)


def _refine_match(df: pd.DataFrame, col: str, value: object) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    text = str(value or "").strip()
    if not text:
        return df
    if col == "source_file":
        candidate = df[_source_matches(df[col], text)].copy()
    else:
        candidate = df[df[col].astype(str) == text].copy()
    return candidate if not candidate.empty else df


def _matching_spike_event_subset(events: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    sub = events.copy()
    if "channel" in sub.columns and "channel" in row:
        sub = sub[sub["channel"].astype(str) == str(row.get("channel"))].copy()
    for col in ("preparation_id", "segment_id", "source_file", "epoch_label", "phase"):
        sub = _refine_match(sub, col, row.get(col, ""))
    return sub


def _spike_event_relative_times(events: pd.DataFrame) -> np.ndarray:
    for col in ("segment_spike_time_s", "recording_relative_spike_time_s", "spike_time_relative_s"):
        if col in events.columns:
            values = pd.to_numeric(events[col], errors="coerce").dropna().to_numpy(dtype=float)
            if values.size:
                return np.sort(values)
    if "spike_time_s" not in events.columns:
        return np.array([], dtype=float)
    return np.sort(pd.to_numeric(events["spike_time_s"], errors="coerce").dropna().to_numpy(dtype=float))


def _choose_lfp_power_columns(df: pd.DataFrame, requested: object = None) -> tuple[str | None, str | None]:
    request = str(requested or "").strip()
    candidates = [request] if request else []
    candidates.extend(["total_power_uV2", "primary_power_uV2"])
    candidates.extend([col for col in df.columns if col.startswith("band_total_") and col.endswith("_power_uV2")])
    candidates.extend([col for col in df.columns if col.startswith("band_") and col.endswith("_power_uV2")])
    absolute = next((col for col in candidates if col and col in df.columns), None)
    if absolute is None:
        return None, None
    normalized = f"{absolute}_change_db"
    if normalized not in df.columns:
        normalized = None
    return absolute, normalized


def _aligned_from_spike_events(lfp: pd.DataFrame, events_path: Path, power_col: str, norm_col: str | None) -> pd.DataFrame:
    if not events_path.exists() or events_path.stat().st_size == 0:
        return pd.DataFrame()
    events = pd.read_csv(events_path)
    if events.empty or "channel" not in events.columns:
        return pd.DataFrame()
    rows = []
    for _idx, row in lfp.iterrows():
        start_s = pd.to_numeric(pd.Series([row.get("window_start_s")]), errors="coerce").iloc[0]
        end_s = pd.to_numeric(pd.Series([row.get("window_end_s")]), errors="coerce").iloc[0]
        duration_s = pd.to_numeric(pd.Series([row.get("window_duration_s", end_s - start_s)]), errors="coerce").iloc[0]
        if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
            continue
        sub = _matching_spike_event_subset(events, row)
        times = _spike_event_relative_times(sub)
        count = int(np.count_nonzero((times >= float(start_s)) & (times < float(end_s))))
        duration = float(duration_s) if np.isfinite(duration_s) and duration_s > 0 else float(end_s - start_s)
        rows.append(
            {
                "experiment_id": row.get("experiment_id", ""),
                "animal_id": row.get("animal_id", ""),
                "preparation_id": row.get("preparation_id", ""),
                "segment_id": row.get("segment_id", row.get("recording_id", "")),
                "recording_id": row.get("recording_id", ""),
                "recording_order": row.get("recording_order", ""),
                "source_file": row.get("source_file", ""),
                "epoch_label": row.get("epoch_label", row.get("phase", "")),
                "phase": row.get("phase", ""),
                "channel": row.get("channel", ""),
                "window_index": row.get("window_index", ""),
                "window_start_s": float(start_s),
                "window_end_s": float(end_s),
                "window_mid_s": row.get("window_mid_s", (start_s + end_s) / 2.0),
                "cumulative_window_start_s": row.get("cumulative_window_start_s", np.nan),
                "cumulative_window_end_s": row.get("cumulative_window_end_s", np.nan),
                "cumulative_window_mid_s": row.get("cumulative_window_mid_s", np.nan),
                "window_duration_s": duration,
                "spike_count": count,
                "spike_rate_hz": count / duration if duration > 0 else np.nan,
                "spike_rate_per_min": 60.0 * count / duration if duration > 0 else np.nan,
                "lfp_power_column": power_col,
                "lfp_power_uV2": row.get(power_col, np.nan),
                "lfp_power_change_db": row.get(norm_col, np.nan) if norm_col else np.nan,
                "lfp_qc_pass": True,
                "lfp_qc_reason": row.get("qc_reason", ""),
                "matched_spike_event_rows": int(len(sub)),
                "alignment_error_s": 0.0,
                "match_source": "spike_events_exact_lfp_window",
            }
        )
    return pd.DataFrame(rows)


def _aligned_from_spike_counts(lfp: pd.DataFrame, spike_path: Path, power_col: str, norm_col: str | None) -> pd.DataFrame:
    spike = _standardise_spike_counts(spike_path)
    required = {"channel", "spike_count"}
    if not required.issubset(spike.columns):
        return pd.DataFrame()
    for qc_col in ("spike_qc_pass", "spike_qc", "qc_pass"):
        if qc_col in spike.columns:
            spike = spike[_as_bool(spike[qc_col])].copy()
            break
    if spike.empty:
        return pd.DataFrame()
    if "window_start_s" not in spike.columns and "window_index" in spike.columns:
        spike["window_start_s"] = pd.to_numeric(spike["window_index"], errors="coerce") * 60.0
    lfp = lfp.copy()
    lfp["window_start_s"] = _numeric(lfp, "window_start_s")
    if "window_end_s" in spike.columns:
        spike["window_end_s"] = pd.to_numeric(spike["window_end_s"], errors="coerce")
    if "window_end_s" in lfp.columns:
        lfp["window_end_s"] = _numeric(lfp, "window_end_s")
    for table in (lfp, spike):
        if "epoch_label" not in table.columns and "phase" in table.columns:
            table["epoch_label"] = table["phase"]
    spike_start = "segment_window_start_s" if "segment_window_start_s" in spike.columns else "window_start_s"
    spike_end = "segment_window_end_s" if "segment_window_end_s" in spike.columns else "window_end_s"
    spike["_join_window_start"] = (pd.to_numeric(spike[spike_start], errors="coerce") * 2000.0).round().astype("Int64")
    spike["_join_window_end"] = (pd.to_numeric(spike[spike_end], errors="coerce") * 2000.0).round().astype("Int64")
    lfp["_join_window_start"] = (pd.to_numeric(lfp["window_start_s"], errors="coerce") * 2000.0).round().astype("Int64")
    lfp["_join_window_end"] = (pd.to_numeric(lfp["window_end_s"], errors="coerce") * 2000.0).round().astype("Int64")
    candidate_keys = [
        "preparation_id",
        "segment_id",
        "recording_order",
        "source_file",
        "epoch_label",
        "channel",
        "_join_window_start",
        "_join_window_end",
    ]
    keys = [key for key in candidate_keys if key in lfp.columns and key in spike.columns]
    if "channel" not in keys or "_join_window_start" not in keys:
        return pd.DataFrame()
    strong_keys = [key for key in keys if key not in {"channel", "window_start_s", "window_end_s"}]
    if not strong_keys:
        return pd.DataFrame()
    merged = lfp.merge(
        spike,
        on=keys,
        how="inner",
        suffixes=("_lfp", "_spike"),
    )
    if merged.empty:
        return pd.DataFrame()
    merged["lfp_power_column"] = power_col
    merged["lfp_power_uV2"] = pd.to_numeric(merged[power_col], errors="coerce")
    merged["lfp_power_change_db"] = pd.to_numeric(merged[norm_col], errors="coerce") if norm_col else np.nan
    merged["spike_rate_hz"] = pd.to_numeric(merged["spike_count"], errors="coerce") / pd.to_numeric(merged.get("window_duration_s_lfp", merged.get("window_duration_s", np.nan)), errors="coerce")
    merged["spike_rate_per_min"] = merged["spike_rate_hz"] * 60.0
    merged["alignment_error_s"] = np.maximum(
        np.abs(pd.to_numeric(merged.get("window_start_s_lfp", merged.get("window_start_s")), errors="coerce") - pd.to_numeric(merged.get(spike_start, merged.get("window_start_s_spike", merged.get("window_start_s"))), errors="coerce")),
        np.abs(pd.to_numeric(merged.get("window_end_s_lfp", merged.get("window_end_s")), errors="coerce") - pd.to_numeric(merged.get(spike_end, merged.get("window_end_s_spike", merged.get("window_end_s"))), errors="coerce")),
    )
    merged = merged[merged["alignment_error_s"] <= 0.0005].copy()
    if merged.empty:
        return pd.DataFrame()
    merged["match_source"] = "spike_count_window_exact_within_0p5_ms"
    return merged


def _plot_aligned_spike_lfp(all_lfp: pd.DataFrame, valid_lfp: pd.DataFrame, spike_path: Path, out_dir: Path, spike_events_path: Path | None = None, config: Mapping | None = None) -> dict[str, str] | None:
    power_col, norm_col = _choose_lfp_power_columns(valid_lfp, (config or {}).get("lfp_power_column") if config else None)
    if power_col is None:
        return None
    matched = pd.DataFrame()
    if spike_events_path is not None:
        matched = _aligned_from_spike_events(valid_lfp, spike_events_path, power_col, norm_col)
    if matched.empty:
        matched = _aligned_from_spike_counts(valid_lfp, spike_path, power_col, norm_col)
    if matched.empty:
        return None
    table_path = out_dir / "PRIMARY_spike_lfp_aligned_windows.csv"
    matched.to_csv(table_path, index=False)
    n_rows = 3 if matched["lfp_power_change_db"].notna().any() else 2
    fig = plt.figure(figsize=(13, 7.8 if n_rows == 3 else 6.4))
    gs = fig.add_gridspec(n_rows, 2, width_ratios=[3.2, 1.25])
    axes = []
    for idx in range(n_rows):
        axes.append(fig.add_subplot(gs[idx, 0], sharex=axes[0] if axes else None))
    ax_scatter = fig.add_subplot(gs[:, 1])
    x = pd.to_numeric(matched.get("cumulative_window_mid_s", matched.get("window_mid_s")), errors="coerce") / 60.0
    for _seg, sub in matched.groupby("segment_id", dropna=False, sort=False):
        sx = pd.to_numeric(sub.get("cumulative_window_mid_s", sub.get("window_mid_s")), errors="coerce") / 60.0
        axes[0].plot(sx, pd.to_numeric(sub["spike_count"], errors="coerce"), marker="o", linewidth=1.4)
        axes[1].plot(sx, pd.to_numeric(sub["lfp_power_uV2"], errors="coerce"), marker="o", linewidth=1.4, color="tab:green")
        if n_rows == 3:
            axes[2].plot(sx, pd.to_numeric(sub["lfp_power_change_db"], errors="coerce"), marker="o", linewidth=1.4, color="tab:purple")
    failed = all_lfp.loc[~all_lfp.index.isin(valid_lfp.index)].copy()
    if not failed.empty and {"cumulative_window_start_s", "cumulative_window_end_s"}.issubset(failed.columns):
        for ax in axes:
            for _idx, row in failed.iterrows():
                a = pd.to_numeric(pd.Series([row.get("cumulative_window_start_s")]), errors="coerce").iloc[0]
                b = pd.to_numeric(pd.Series([row.get("cumulative_window_end_s")]), errors="coerce").iloc[0]
                if np.isfinite(a) and np.isfinite(b) and b > a:
                    ax.axvspan(float(a) / 60.0, float(b) / 60.0, color="#9ca3af", alpha=0.12, linewidth=0)
    for ax in axes:
        _add_segment_boundaries(ax, matched)
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Spike count")
    axes[1].set_ylabel(power_col.replace("_", " "))
    if n_rows == 3:
        axes[2].set_ylabel("Baseline-normalized dB")
    axes[-1].set_xlabel("Cumulative recorded time (min)")
    scatter_y_col = "lfp_power_change_db" if matched["lfp_power_change_db"].notna().any() else "lfp_power_uV2"
    sx = pd.to_numeric(matched["spike_rate_per_min"], errors="coerce")
    sy = pd.to_numeric(matched[scatter_y_col], errors="coerce")
    finite = sx.notna() & sy.notna()
    ax_scatter.scatter(sx[finite], sy[finite], s=30, alpha=0.75)
    rho = float(stats.spearmanr(sx[finite], sy[finite]).statistic) if np.count_nonzero(finite) >= 3 else np.nan
    rho_text = f"Spearman rho={rho:.3g}" if np.isfinite(rho) else "Spearman rho=n/a"
    ax_scatter.set_title(f"Matched windows n={int(np.count_nonzero(finite))}\n{rho_text} descriptive")
    ax_scatter.set_xlabel("Spike rate (per min)")
    ax_scatter.set_ylabel(scatter_y_col.replace("_", " "))
    ax_scatter.grid(alpha=0.18)
    axes[0].set_title("PRIMARY aligned 20 kHz spike counts/rate and synchronized 2 kHz LFP power")
    fig.tight_layout()
    plot_path = out_dir / "PRIMARY_spike_lfp_aligned_spike_rate_lfp_power.png"
    fig.savefig(plot_path, dpi=DEFAULT_DPI)
    plt.close(fig)
    return {"plot": str(plot_path), "table": str(table_path)}


def _plot_ppc_timecourse(ppc_path: Path, out_dir: Path) -> str | None:
    ppc = pd.read_csv(ppc_path)
    if ppc.empty or "ppc" not in ppc.columns:
        return None
    if "qc_pass" in ppc.columns:
        ppc = ppc[_as_bool(ppc["qc_pass"])].copy()
    if "coupling_qc_pass" in ppc.columns:
        ppc = ppc[_as_bool(ppc["coupling_qc_pass"])].copy()
    if ppc.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 4.5))
    group_cols = [
        col
        for col in ("preparation_id", "segment_id", "channel_pair", "band_label", "epoch_label")
        if col in ppc.columns
    ]
    if not group_cols:
        group_cols = [col for col in ("lfp_channel", "spike_channel") if col in ppc.columns]
    plotted = 0
    max_lines = 24
    for key, sub in ppc.groupby(group_cols, dropna=False, sort=False):
        if plotted >= max_lines:
            break
        sx = _numeric(sub, "cumulative_window_mid_s") / 60.0 if "cumulative_window_mid_s" in sub.columns else _numeric(sub, "window_mid_s") / 60.0
        label = " | ".join(map(str, key if isinstance(key, tuple) else (key,)))
        ax.plot(sx, _numeric(sub, "ppc"), marker="o", linewidth=1.5, label=label)
        plotted += 1
    ax.axhline(0, color="black", linewidth=1, alpha=0.45)
    suffix = f" (first {max_lines} groups)" if len(ppc.groupby(group_cols, dropna=False)) > max_lines else ""
    ax.set_title(f"MUA-LFP phase coupling PPC time course{suffix}")
    _add_segment_boundaries(ax, ppc)
    ax.set_xlabel("Cumulative recorded time (min)")
    ax.set_ylabel("Pairwise phase consistency")
    if plotted <= 4:
        ax.legend(loc="best", fontsize=7, ncols=2)
    elif plotted:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=6)
    return _save(fig, out_dir, "spike_lfp_ppc_timecourse.png")


def _plot_summary_table(df: pd.DataFrame, out_dir: Path) -> str:
    cols = [col for col in ("total_power_uV2", "primary_power_uV2", "lfp_rms_uv", "dominant_peak_frequency_hz") if col in df.columns]
    group_cols = [col for col in ("animal_id", "preparation_id", "channel", "epoch_label", "phase") if col in df.columns]
    summary = df.groupby(group_cols, dropna=False)[cols].median(numeric_only=True).reset_index() if cols and group_cols else pd.DataFrame()
    out = out_dir / "lfp_plot_preparation_summary.csv"
    summary.to_csv(out, index=False)
    return str(out)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot processed SPIE LFP feature outputs.")
    parser.add_argument("feature_csv", help="lfp_features_by_window.csv")
    parser.add_argument("--out-dir", default="", help="Plot output folder.")
    parser.add_argument("--plot-mode", choices=("simple", "complex"), default="complex", help="Simple plots or full comparison plots.")
    parser.add_argument("--simple-plot-bands", default="", help="Optional simple narrow bands, e.g. very_slow=0.5-2; slow_motor=2-5.")
    parser.add_argument("--spike-count-csv", default="", help="Optional spike count CSV for aligned panels.")
    parser.add_argument("--ppc-csv", default="", help="Optional spike_lfp_ppc_by_window.csv for PPC panels.")
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = {}
    if args.out_dir:
        cfg["output_dir"] = args.out_dir
    cfg["plot_mode"] = args.plot_mode
    if args.simple_plot_bands:
        cfg["simple_plot_bands"] = args.simple_plot_bands
    if args.spike_count_csv:
        cfg["spike_count_csv"] = args.spike_count_csv
    if args.ppc_csv:
        cfg["ppc_csv"] = args.ppc_csv
    plot_lfp_features(args.feature_csv, cfg)


if __name__ == "__main__":
    main()
