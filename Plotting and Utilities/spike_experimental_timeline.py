"""Experimental timing for the shared SPIE preparation analysis.

No resampling or filling occurs here. All displayed points are observed window
midpoints; group keys require identical epoch and experimental/display bounds.
"""
from __future__ import annotations

from pathlib import Path
import re
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def timeline_options(options=None):
    cfg = dict(timing_mode="auto", gap_min=20., show_treatment_duration=True,
               relative_to_treatment=True, baseline_display="reference",
               display_baseline_range_min=None, boundaries={}, epoch_starts_min={},
               compression_gap_min=1.)
    if options is not None:
        if not isinstance(options, dict) or set(options) - set(cfg):
            raise ValueError("Unknown experimental timeline settings.")
        cfg.update(options)
    if cfg["timing_mode"] not in ("auto", "reconstructed", "measured", "assigned"):
        raise ValueError("Timing mode must be auto, reconstructed, measured, or assigned.")
    if cfg["baseline_display"] not in ("reference", "full", "range"):
        raise ValueError("Baseline display must be reference, full, or range.")
    for key in ("gap_min", "compression_gap_min"):
        cfg[key] = float(cfg[key])
        if not np.isfinite(cfg[key]) or cfg[key] < 0 or (key == "compression_gap_min" and cfg[key] == 0):
            raise ValueError(f"{key} must be a finite nonnegative duration (visible break must be positive).")
    if cfg["baseline_display"] == "range":
        r = cfg["display_baseline_range_min"]
        if r is None or len(r) != 2 or not np.isfinite(r).all() or not 0 <= r[0] < r[1]:
            raise ValueError("Displayed baseline range must have 0 <= start < end in minutes.")
    if not isinstance(cfg["boundaries"], dict) or not isinstance(cfg["epoch_starts_min"], dict):
        raise ValueError("Boundary and epoch-start settings must be objects.")
    cfg["epoch_starts_min"] = {**cfg["epoch_starts_min"], **{str(Path(k).resolve()): v for k, v in cfg["epoch_starts_min"].items() if Path(k).is_absolute()}}
    for key, boundary in cfg["boundaries"].items():
        if not isinstance(boundary, dict) or set(boundary) - {"treatment", "duration_min"}:
            raise ValueError(f"Invalid boundary settings for {key}.")
        duration = float(boundary.get("duration_min", cfg["gap_min"]))
        if not np.isfinite(duration) or duration < 0:
            raise ValueError(f"Invalid treatment duration for {key}.")
    return cfg


def boundary_key(previous, following):
    return f"{previous} -> {following}"


def timeline_epoch_order(labels):
    from preparation_spike_analysis import epoch_order
    def key(label):
        if label == "Baseline":
            return (0, 0, 0)
        match = re.fullmatch(r"(Post|Treatment|Stimulation|During)\s*(\d*)", label, re.I)
        if match:
            return (1, int(match.group(2) or 1), int(match.group(1).lower() == "post"))
        return (2, 0, 0)
    return sorted(epoch_order(labels), key=key)


def boundary_setting(previous, following, cfg):
    default_treatment = (previous == "Baseline" or previous.lower().startswith("post")) and following.lower().startswith("post")
    setting = cfg["boundaries"].get(boundary_key(previous, following), {})
    enabled = setting.get("treatment", default_treatment)
    return bool(enabled), float(setting.get("duration_min", cfg["gap_min"])) if enabled else 0.


def build_experimental_timeline(data, options=None):
    from preparation_spike_analysis import epoch_order
    cfg = timeline_options(options)
    out = data.copy()
    treatments = []
    # Treat channel dropout as missing observations, not a different physical
    # treatment onset: determine each epoch's extent across this specimen only.
    all_epochs = timeline_epoch_order(data.epoch_label)
    for prep, group in data.groupby("preparation_id", sort=False):
        path = group.source_file.iloc[0]
        epochs = timeline_epoch_order(group.epoch_label)
        assigned = cfg["epoch_starts_min"].get(path, cfg["epoch_starts_min"].get(prep, {}))
        if not isinstance(assigned, dict):
            raise ValueError(f"Epoch starts for {prep} must map epoch labels to minutes.")
        has_measured = {"experimental_start_s", "experimental_end_s"}.issubset(group.columns)
        mode = "assigned" if assigned else cfg["timing_mode"]
        if has_measured and mode in ("auto", "measured"):
            epochs.sort(key=lambda e: pd.to_numeric(group.loc[group.epoch_label == e, "experimental_start_s"], errors="coerce").min())
        elif assigned and not set(epochs) - set(assigned):
            epochs.sort(key=lambda e: float(assigned[e]))
        if mode == "auto":
            if has_measured:
                mode = "measured"
            else:
                # Strictly increasing non-overlapping native epoch timestamps
                # could already contain gaps. Require an explicit interpretation.
                extents = [(group[group.epoch_label == e].window_start_s.min(), group[group.epoch_label == e].window_end_s.max()) for e in epochs]
                if len(extents) > 1 and all(b[0] >= a[1] for a, b in zip(extents, extents[1:])):
                    raise ValueError(f"{prep}: native times may already contain treatment gaps. Select measured or reconstructed timing explicitly.")
                mode = "reconstructed"
        if mode == "assigned" and set(epochs) - set(assigned):
            raise ValueError(f"{prep}: assign a start time to every observed epoch.")
        if "Baseline" not in epochs and mode == "reconstructed":
            raise ValueError(f"{prep}: experimental origin is unknown without Baseline. Use measured times or assign epoch starts.")
        if mode == "reconstructed":
            missing = set(all_epochs) - set(epochs)
            # Missing intermediate phases have unknown durations. Missing trailing
            # phases do not affect the timing of observations that do exist.
            before_last = set(all_epochs[:all_epochs.index(epochs[-1])])
            post_numbers = [int(m.group(1)) for e in epochs if (m := re.fullmatch(r"Post\s+(\d+)", e))]
            if missing & before_last or (post_numbers and set(range(1, max(post_numbers)+1)) - set(post_numbers)):
                raise ValueError(f"{prep}: missing intermediate epochs make elapsed timing ambiguous; supply epoch starts or measured timestamps.")
        previous = None
        previous_end = None
        baseline_end = None
        prep_treatments = []
        for epoch in epochs:
            rows = group[group.epoch_label == epoch]
            local_origin = rows.window_start_s.min()
            alignment_origin = rows.window_end_s.max() if epoch == "Baseline" else local_origin
            out.loc[rows.index, "epoch_relative_start_s"] = rows.window_start_s - alignment_origin
            out.loc[rows.index, "epoch_relative_end_s"] = rows.window_end_s - alignment_origin
            enabled, gap = boundary_setting(previous, epoch, cfg) if previous is not None else (False, 0.)
            if mode == "measured":
                start_col, end_col = ("experimental_start_s", "experimental_end_s") if has_measured else ("window_start_s", "window_end_s")
                starts = pd.to_numeric(rows[start_col], errors="coerce") / 60
                ends = pd.to_numeric(rows[end_col], errors="coerce") / 60
                if not np.isfinite(starts).all() or not np.isfinite(ends).all() or not np.allclose((ends-starts)*60, rows.duration_s, rtol=0, atol=1e-6):
                    raise ValueError(f"{prep}/{epoch}: measured times must be finite and preserve each window's duration.")
                for _, channel_rows in rows.groupby("channel"):
                    ordered = pd.DataFrame({"start": starts.loc[channel_rows.index], "end": ends.loc[channel_rows.index]}).sort_values("start")
                    if (ordered.start.iloc[1:].to_numpy() < ordered.end.iloc[:-1].to_numpy() - 1e-8).any():
                        raise ValueError(f"{prep}/{epoch}: measured windows overlap within a channel.")
            else:
                start = float(assigned[epoch]) if mode == "assigned" else (0. if previous_end is None else previous_end + gap)
                if not np.isfinite(start):
                    raise ValueError(f"{prep}/{epoch}: epoch start must be finite.")
                starts = start + (rows.window_start_s - local_origin) / 60
                ends = start + (rows.window_end_s - local_origin) / 60
            epoch_start, epoch_end = float(starts.min()), float(ends.max())
            if previous_end is not None:
                if epoch_start < previous_end - 1e-8:
                    raise ValueError(f"{prep}: experimental epochs overlap or run backwards.")
                if enabled and gap > epoch_start - previous_end + 1e-8:
                    raise ValueError(f"{prep}/{epoch}: configured treatment duration exceeds the measured/assigned gap. Correct the boundary duration; no extra gap is added.")
                if enabled and gap > 0:
                    prep_treatments.append(dict(previous_epoch=previous, next_epoch=epoch, start_min=previous_end, end_min=previous_end+gap, duration_min=gap, recorded=False))
            if re.fullmatch(r"(treatment|stimulation|during)\s*\d*", epoch, re.I):
                prep_treatments.append(dict(previous_epoch=epoch, next_epoch=epoch, start_min=epoch_start, end_min=epoch_end, duration_min=epoch_end-epoch_start, recorded=True))
            out.loc[rows.index, "experimental_start_min"] = starts
            out.loc[rows.index, "experimental_end_min"] = ends
            out.loc[rows.index, "timing_source"] = mode if mode != "assigned" else "assigned/reconstructed"
            out.loc[rows.index, "treatment_duration_before_min"] = gap
            if epoch == "Baseline":
                baseline_end = epoch_end
            previous, previous_end = epoch, epoch_end
        # Onset is the first explicitly present treatment, otherwise the end of
        # Baseline. This is recorded in exports rather than inferred from n bins.
        onset = prep_treatments[0]["start_min"] if prep_treatments else baseline_end
        if onset is None:
            if cfg["relative_to_treatment"]:
                raise ValueError(f"{prep}: treatment-relative origin unavailable; disable relative timing or supply Baseline/treatment.")
            onset = 0.
        origin = onset if cfg["relative_to_treatment"] else float(out.loc[group.index, "experimental_start_min"].min())
        def compressed(times):
            values = np.asarray(times, dtype=float).copy()
            if not cfg["show_treatment_duration"]:
                for treatment in prep_treatments:
                    if not treatment["recorded"]:
                        reduction = treatment["duration_min"] - min(cfg["compression_gap_min"], treatment["duration_min"])
                        values -= np.where(np.asarray(times) >= treatment["end_min"] - 1e-8, reduction, 0.)
            return values - origin
        out.loc[group.index, "plot_start_min"] = compressed(out.loc[group.index, "experimental_start_min"])
        out.loc[group.index, "plot_end_min"] = compressed(out.loc[group.index, "experimental_end_min"])
        out.loc[group.index, "time_origin_min"] = origin
        out.loc[group.index, "timing_compressed"] = not cfg["show_treatment_duration"]
        # Experimental times are expressed relative to the selected origin too;
        # retain unshifted measured clock coordinates in separate columns.
        for edge in ("start", "end"):
            col = f"experimental_{edge}_min"
            out.loc[group.index, f"unshifted_{col}"] = out.loc[group.index, col]
            out.loc[group.index, col] -= origin
        for treatment in prep_treatments:
            plot_start = float(compressed([treatment["start_min"]])[0])
            displayed_duration = treatment["duration_min"] if cfg["show_treatment_duration"] or treatment["recorded"] else min(cfg["compression_gap_min"], treatment["duration_min"])
            treatments.append(dict(treatment, preparation_id=prep, source_file=path,
                                   display_name=group.display_name.iloc[0], display_label=group.display_label.iloc[0],
                                   current_na=group.current_na.iloc[0], plot_start_min=plot_start,
                                   plot_end_min=plot_start+displayed_duration,
                                   experimental_start_min=treatment["start_min"]-origin,
                                   experimental_end_min=treatment["end_min"]-origin))
    out["experimental_midpoint_min"] = (out.experimental_start_min + out.experimental_end_min) / 2
    out["plot_midpoint_min"] = (out.plot_start_min + out.plot_end_min) / 2
    visible = pd.Series(True, index=out.index)
    for (_, _), baseline in out[out.epoch_label == "Baseline"].groupby(["preparation_id", "channel"]):
        if cfg["baseline_display"] == "reference":
            lo, hi = baseline.baseline_reference_start_s.iloc[0], baseline.baseline_reference_end_s.iloc[0]
        elif cfg["baseline_display"] == "range":
            lo, hi = [baseline.window_start_s.min() + x*60 for x in cfg["display_baseline_range_min"]]
        else:
            continue
        # No synthetic clipped observations: display original windows intersecting
        # the chosen range, at their actual midpoints.
        visible.loc[baseline.index] = (baseline.window_end_s > lo) & (baseline.window_start_s < hi)
    out["displayed"] = visible
    treatment_cols = ["preparation_id", "source_file", "display_name", "display_label", "current_na", "previous_epoch", "next_epoch", "start_min", "end_min", "duration_min", "recorded", "plot_start_min", "plot_end_min", "experimental_start_min", "experimental_end_min"]
    return out, pd.DataFrame(treatments, columns=treatment_cols)


TIME_KEYS = ["channel", "current_na", "epoch_label", "epoch_relative_start_s", "epoch_relative_end_s", "experimental_start_min", "experimental_end_min", "plot_start_min", "plot_end_min"]


def summarize_timeline(data):
    from preparation_spike_analysis import mean_sem
    rows = []
    for keys, group in data[data.displayed].groupby(TIME_KEYS, sort=False):
        if group.preparation_id.duplicated().any():
            raise ValueError("Duplicate windows for the same preparation in a group time point.")
        normalized = mean_sem(group.normalized_rate_pct)
        absolute = mean_sem(group.firing_rate_hz)
        rows.append(dict(zip(TIME_KEYS, keys), group_mean_normalized_pct=normalized["mean"],
                         group_sem_normalized_pct=normalized["sem"], n_preparations=normalized["n_preparations"],
                         group_mean_hz=absolute["mean"], group_sem_hz=absolute["sem"], n_preparations_hz=absolute["n_preparations"],
                         plot_midpoint_min=(keys[-2]+keys[-1])/2))
    return pd.DataFrame(rows, columns=TIME_KEYS+["group_mean_normalized_pct", "group_sem_normalized_pct", "n_preparations", "group_mean_hz", "group_sem_hz", "n_preparations_hz", "plot_midpoint_min"])


def contiguous_segments(group):
    group = group.sort_values("plot_start_min")
    breaks = ~np.isclose(group.plot_start_min, group.plot_end_min.shift(), rtol=0, atol=1e-8)
    for _, segment in group.groupby(breaks.cumsum()):
        yield segment


def timeline_figure(data, summary, treatments, title, normalized, styles, options):
    from preparation_spike_analysis import finish_figure, sample_label
    cfg = timeline_options(options)
    value = "normalized_rate_pct" if normalized else "firing_rate_hz"
    mean = "group_mean_normalized_pct" if normalized else "group_mean_hz"
    sem = "group_sem_normalized_pct" if normalized else "group_sem_hz"
    n = "n_preparations" if normalized else "n_preparations_hz"
    points = data[data.displayed & data[value].notna()]
    summary = summary[summary[n] > 0]
    fig, (ax, counts_ax, treatment_ax) = plt.subplots(3, 1, sharex=True, figsize=(14, 9), gridspec_kw={"height_ratios": [6, 1, max(1, points.preparation_id.nunique()*.65)]})
    seen = set()
    for (prep, epoch), group in points.groupby(["preparation_id", "epoch_label"], sort=False):
        for segment in contiguous_segments(group):
            ax.plot(segment.plot_midpoint_min, segment[value], color=styles[prep]["color"], marker=styles[prep]["marker"],
                    ms=5, lw=1, alpha=.65 if points.preparation_id.nunique() > 1 else 1,
                    markerfacecolor="none", zorder=4, label=group.display_label.iloc[0] if prep not in seen else None)
            seen.add(prep)
    for epoch, group in summary.groupby("epoch_label", sort=False):
        # Separate differing windows rather than connect interleaved group points.
        for _, same_duration in group.groupby(group.plot_end_min-group.plot_start_min):
            for segment in contiguous_segments(same_duration[same_duration[n] >= 2]):
                ax.plot(segment.plot_midpoint_min, segment[mean], "ko-", lw=2, ms=4, zorder=3,
                        label="Group mean" if "mean" not in seen else None)
                ax.errorbar(segment.plot_midpoint_min, segment[mean], yerr=segment[sem], fmt="none", color="black", capsize=3, lw=1,
                            label="Mean ± SEM" if "sem" not in seen else None)
                seen.update(("mean", "sem"))
            for segment in contiguous_segments(same_duration):
                counts_ax.plot(segment.plot_midpoint_min, segment[n], ".-", lw=.8, ms=3)
    counts_ax.set_ylabel("Contributing n", fontsize=9)
    counts_ax.set_yticks(sorted(set(summary[n])))
    counts_ax.set_ylim(0, max(1, summary[n].max() if len(summary) else 1)+1)
    preps = list(points.preparation_id.unique())
    for i, prep in enumerate(preps):
        for row in treatments[treatments.preparation_id == prep].itertuples():
            treatment_ax.plot([row.plot_start_min, row.plot_end_min], [i, i], color="black", lw=6, solid_capstyle="butt")
            compressed = not cfg["show_treatment_duration"] and not row.recorded
            treatment_ax.annotate(f"{row.current_na:g} nA, {row.duration_min:g} min" + (" (compressed)" if compressed else ""),
                                  ((row.plot_start_min+row.plot_end_min)/2, i), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    labels = points.drop_duplicates("preparation_id").set_index("preparation_id").display_label
    treatment_ax.set_yticks(range(len(preps)), [textwrap.fill(labels[p], 32) for p in preps], fontsize=8)
    treatment_ax.set_ylim(-.5, max(.5, len(preps)-.3))
    treatment_ax.set_ylabel("Treatment", fontsize=9)
    treatment_ax.set_xlabel("Time (min)")
    note = "Treatment intervals compressed; " if not cfg["show_treatment_duration"] else ""
    ax.text(0, 1.025, note + sample_label(len(preps)) + "; contributing n shown below", transform=ax.transAxes, fontsize=9)
    if normalized:
        ax.axhline(100, color="gray", ls="--", lw=1)
    ax.set_ylabel("Firing rate (% of baseline)" if normalized else "Firing rate (Hz)")
    for axis in (ax, counts_ax, treatment_ax):
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_facecolor("white")
    finish_figure(fig, ax, title, treatment_ax)
    return fig
