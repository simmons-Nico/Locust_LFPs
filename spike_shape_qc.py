"""Bombcell-inspired per-event adaptation; thresholds require dataset validation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


DEFAULTS = dict(mode="off", separation_ms=1.0, depth_fraction=0.2,
                prominence_fraction=0.2, noise_multiplier=3.0, padding_ms=0.5)


def settings(values=None):
    cfg = {**DEFAULTS, **(values or {})}
    cfg["mode"] = str(cfg["mode"]).lower().replace(" ", "_")
    if cfg["mode"] not in ("off", "flag_only", "reject"):
        raise ValueError("Shape mode must be off, flag_only, or reject.")
    for key in (k for k in DEFAULTS if k != "mode"):
        cfg[key] = float(cfg[key])
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"Shape {key} must be finite and non-negative.")
    if cfg["separation_ms"] <= 0 or cfg["padding_ms"] <= 0:
        raise ValueError("Shape separation and padding must be positive.")
    return cfg


def add_arguments(parser):
    parser.add_argument("--shape-mode", choices=("off", "flag_only", "reject"), default="off")
    for key in (k for k in DEFAULTS if k != "mode"):
        parser.add_argument("--shape-" + key.replace("_", "-"), type=float, default=DEFAULTS[key])


def from_arguments(args):
    return settings({key: getattr(args, "shape_" + key) for key in DEFAULTS})


def assess_events(trace, events, fs, config):
    """Assess canonical events, without applying refractory suppression.

    Negative events use their aligned trough. Positive events use the deepest
    significant trough within separation_ms of their positive peak. A single
    negative lobe never fails; positive waveforms with no trough also pass.
    Depth is relative to the trace median. Prominence uses scipy's local bases.
    """
    cfg = settings(config)
    if cfg["mode"] == "off":
        return []
    trace = np.asarray(trace, dtype=float)
    baseline = float(np.median(trace))
    sigma = float(np.median(np.abs(trace - baseline)) / 0.6745)
    # Two separation windows cover the positive anchor-to-trough shift plus
    # the trough-to-secondary search. Extra padding establishes local bases.
    radius = int(np.ceil((2 * cfg["separation_ms"] + cfg["padding_ms"]) * fs / 1000))
    reports = []
    for event in events:
        anchor = int(event["sample_index"])
        lo, hi = max(0, anchor - radius), min(len(trace), anchor + radius + 1)
        depth_limit = max(cfg["depth_fraction"] * event["amplitude_uv"], cfg["noise_multiplier"] * sigma)
        prominence_limit = max(cfg["prominence_fraction"] * event["amplitude_uv"], cfg["noise_multiplier"] * sigma)
        indices, props = find_peaks(baseline - trace[lo:hi], height=depth_limit, prominence=prominence_limit)
        indices = indices + lo
        depths, prominences = props["peak_heights"], props["prominences"]
        within = np.abs(indices - anchor) * 1000 / fs <= cfg["separation_ms"] + 1e-9
        choices = np.flatnonzero(within)
        if event["polarity"] == "neg":
            choices = np.flatnonzero(indices == anchor)
        main = int(choices[np.argmax(depths[choices])]) if choices.size else None
        secondary = None
        if main is not None:
            close = np.flatnonzero((np.abs(indices - indices[main]) * 1000 / fs <= cfg["separation_ms"] + 1e-9)
                                   & (indices != indices[main]))
            if close.size:
                secondary = int(close[np.argmax(depths[close])])
        row = {**event, **{"shape_" + k: v for k, v in cfg.items()},
               "noise_sigma_uv": sigma, "baseline_uv": baseline,
               "depth_threshold_uv": depth_limit, "prominence_threshold_uv": prominence_limit,
               "shape_flagged": secondary is not None,
               "shape_reason": "multiple_significant_troughs" if secondary is not None else "",
               "inspection_complete": anchor - radius >= 0 and anchor + radius < len(trace)}
        for name, j in (("main", main), ("secondary", secondary)):
            row[name + "_trough_sample"] = int(indices[j]) if j is not None else None
            row[name + "_trough_depth_uv"] = float(depths[j]) if j is not None else None
            row[name + "_trough_prominence_uv"] = float(prominences[j]) if j is not None else None
        row["trough_separation_ms"] = float(abs(indices[secondary] - indices[main]) * 1000 / fs) if secondary is not None else None
        reports.append(row)
    return reports


def write_audit(out_dir, recording, channel, trace, times, fs, reports, before, after, bounds, config):
    """Save unique candidate audit, all rejected snippets, and grouped means.

    Before/after counts include the original refractory and amplitude/width
    gates. Candidate rejections need not equal net count reduction.
    """
    import matplotlib.pyplot as plt
    cfg = settings(config)
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    before, after = set(map(int, before)), set(map(int, after))
    pre = int(np.ceil((2 * cfg["separation_ms"] + cfg["padding_ms"]) * fs / 1000))
    post = pre
    rows, snippets, summaries = [], [], []
    prefix = "flagged" if cfg["mode"] == "flag_only" else "rejected"
    for n, original in enumerate(reports, 1):
        row = dict(original)
        p = row["sample_index"]
        row.update(recording=recording, channel=channel, event_id=n, event_time_s=float(times[p]),
                   sampling_rate_hz=float(fs), counted_before=p in before, counted_after=p in after,
                   rejected=cfg["mode"] == "reject" and row["shape_flagged"])
        row["complete_waveform"] = p - pre >= 0 and p + post < len(trace)
        rows.append(row)
        if row["shape_flagged"]:
            start, stop = max(0, p - pre), min(len(trace), p + post + 1)
            snippets.extend(dict(event_id=n, recording=recording, channel=channel, polarity=row["polarity"],
                                 event_time_s=row["event_time_s"], sampling_rate_hz=fs,
                                 rejection_reason=row["shape_reason"], complete_waveform=row["complete_waveform"],
                                 relative_time_ms=(i-p)*1000/fs, voltage_uv=float(trace[i]))
                            for i in range(start, stop))
    audit_columns = list(rows[0]) if rows else ["recording", "channel", "event_id", "sample_index", "event_time_s",
        "sampling_rate_hz", "polarity", "shape_flagged", "shape_reason", "rejected", "complete_waveform", "counted_before", "counted_after"]
    pd.DataFrame(rows, columns=audit_columns).to_csv(folder / "candidate_audit.csv", index=False)
    pd.DataFrame(snippets, columns=["event_id", "recording", "channel", "polarity", "event_time_s", "sampling_rate_hz",
                                   "rejection_reason", "complete_waveform", "relative_time_ms", "voltage_uv"]).to_csv(folder / f"{prefix}_waveforms.csv", index=False)
    rel = np.arange(-pre, post+1) * 1000/fs
    for polarity in ("pos", "neg"):
        rejected = [r for r in rows if r["polarity"] == polarity and r["shape_flagged"]]
        complete = [r for r in rejected if r["complete_waveform"]]
        summary = dict(recording=recording, channel=channel, polarity=polarity, sampling_rate_hz=fs,
                       flagged_total=len(rejected), rejected_total=sum(r["rejected"] for r in rejected),
                       waveforms_in_mean=len(complete), incomplete_waveforms=len(rejected)-len(complete), waveform_group=prefix)
        summaries.append(summary)
        if complete:
            waves = np.stack([trace[r["sample_index"]-pre:r["sample_index"]+post+1] for r in complete])
            mean, sd = waves.mean(axis=0), waves.std(axis=0)
            pd.DataFrame(dict(relative_time_ms=rel, mean_uv=mean, sd_uv=sd, n=len(complete))).to_csv(folder / f"{polarity}_{prefix}_mean.csv", index=False)
            fig, ax = plt.subplots(figsize=(8, 4))
            # A deterministic subset keeps the plot readable; all snippets are saved.
            for wave in waves[np.linspace(0, len(waves)-1, min(200, len(waves)), dtype=int)]:
                ax.plot(rel, wave, color="gray", alpha=0.12, linewidth=0.5)
            ax.fill_between(rel, mean-sd, mean+sd, alpha=0.25, label="±1 SD")
            ax.plot(rel, mean, label=f"Mean (N={len(complete)})")
            ax.set(xlabel="Time from event extremum (ms)", ylabel="Filtered voltage (µV)",
                   title=f"{recording} | {channel} | {polarity}\n{prefix.title()}={len(rejected)}, mean N={len(complete)}, incomplete={len(rejected)-len(complete)}")
            ax.legend(); fig.tight_layout(); fig.savefig(folder / f"{polarity}_{prefix}_mean.png", dpi=150); plt.close(fig)
        else:
            # Always overwrite known outputs: a rerun with zero contributors
            # must not leave a previous run's mean visible.
            pd.DataFrame(dict(relative_time_ms=rel, mean_uv=np.nan, sd_uv=np.nan, n=0)).to_csv(folder / f"{polarity}_{prefix}_mean.csv", index=False)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(.5, .5, f"No complete waveforms (N=0)\n{prefix.title()}={len(rejected)}, incomplete={len(rejected)}", ha="center", va="center", transform=ax.transAxes)
            ax.set(title=f"{recording} | {channel} | {polarity}", xlabel="Time from event extremum (ms)", ylabel="Filtered voltage (µV)")
            fig.tight_layout(); fig.savefig(folder / f"{polarity}_{prefix}_mean.png", dpi=150); plt.close(fig)
    pd.DataFrame(summaries).to_csv(folder / "waveform_summary.csv", index=False)
    counts = []
    for index, (a, b) in enumerate(bounds):
        candidates = [r for r in rows if a <= r["event_time_s"] < b]
        n_before = sum(a <= times[p] < b for p in before)
        n_after = sum(a <= times[p] < b for p in after)
        flagged = sum(r["shape_flagged"] for r in candidates)
        rejected = sum(r["rejected"] for r in candidates)
        counts.append(dict(recording=recording, channel=channel, window_index=index, window_start_s=a, window_end_s=b,
                           count_before=n_before, count_after=n_after, net_count_reduction=n_before-n_after,
                           unique_candidates=len(candidates), flagged_candidates=flagged, rejected_candidates=rejected,
                           rejected_candidate_percent=100*rejected/len(candidates) if candidates else 0,
                           rejected_multiple_significant_troughs=rejected))
    pd.DataFrame(counts).to_csv(folder / "count_comparison.csv", index=False)
    (folder / "settings.json").write_text(json.dumps(dict(settings=cfg, sampling_rate_hz=fs,
        method="Bombcell-inspired per-event adaptation; not validated for locust spikes",
        source="https://github.com/Julie-Fabre/bombcell/wiki/List-of-all-quality-metrics-and-parameters"), indent=2), encoding="utf-8")
    return str(folder)
