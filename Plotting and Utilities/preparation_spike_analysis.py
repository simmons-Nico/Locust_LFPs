"""Independent-preparation spike analysis shared by SPIE and standalone tools.

Rates are duration weighted. A bin crossing the baseline cutoff contributes
in proportion to its overlap (constant rate within that bin). Raw overlays
combine only identical aligned time windows, never interpolate missing counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
import tempfile
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_FOLDER = Path(r"C:\Users\simmons\Desktop\Exploring PSDs")
KEYS = ["preparation_id", "source_file", "channel", "current_na"]


def mean_sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    mean = float(values.mean()) if n else np.nan
    sem = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return {"mean": mean, "sem": sem, "n_preparations": n}


def load_preparations(paths, currents, display_names=None, epoch_aliases=None):
    display_names = validate_titles(display_names)
    display_names = {**display_names, **{str(Path(k).resolve()): v for k, v in display_names.items() if Path(k).is_absolute()}}
    paths = [Path(p).resolve() for p in paths]
    if not paths or len(paths) != len(currents):
        raise ValueError("Assign one current intensity to each input file.")
    if len(set(paths)) != len(paths):
        raise ValueError("The same preparation file cannot be selected twice.")
    if not np.isfinite(np.asarray(currents, dtype=float)).all():
        raise ValueError("Current intensities must be finite numbers in nA.")
    frames = []
    required = ["epoch_label", "channel", "window_start_s", "window_end_s", "spike_count"]
    for path, current in zip(paths, currents):
        df = pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        if df.empty or df[["epoch_label", "channel"]].isna().any().any():
            raise ValueError(f"{path.name}: empty data or missing epoch/channel labels.")
        if epoch_aliases is not None:
            df["original_epoch_label"] = df.epoch_label
            labels = df.epoch_label.astype(str).str.strip().str.casefold()
            unknown = sorted(set(labels) - set(epoch_aliases))
            if unknown:
                raise ValueError(f"{path.name}: unrecognized H2O2 epoch labels {unknown}; use baseline, during and post.")
            df["epoch_label"] = labels.map(epoch_aliases)
        for col in required[2:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if not np.isfinite(df[required[2:]].to_numpy()).all() or (df.spike_count < 0).any():
            raise ValueError(f"{path.name}: counts and window times must be finite; counts cannot be negative.")
        df["duration_s"] = df.window_end_s - df.window_start_s
        if (df.duration_s <= 0).any():
            raise ValueError(f"{path.name}: window duration must be positive.")
        df["source_file"] = str(path)
        df["preparation_id"] = path.stem + "_" + hashlib.sha256(str(path).encode()).hexdigest()[:8]
        df["display_name"] = display_names.get(str(path), display_names.get(df["preparation_id"].iloc[0], "")).strip() or path.stem
        if "recording_name" not in df:
            df["recording_name"] = path.stem
        if "window_index" not in df:
            df["window_index"] = df.groupby(["channel", "epoch_label"]).cumcount()
        df["current_na"] = float(current)
        df["channel"] = df.channel.astype(str)
        for (_, _), group in df.groupby(["channel", "epoch_label"], sort=False):
            group = group.sort_values("window_start_s")
            if np.any(group.window_start_s.to_numpy()[1:] < group.window_end_s.to_numpy()[:-1] - 1e-8):
                raise ValueError(f"{path.name}: overlapping/reset window times in {group.channel.iloc[0]} / {group.epoch_label.iloc[0]}. Supply epoch-relative continuous times.")
            origin = group.window_start_s.min()
            df.loc[group.index, "aligned_start_s"] = group.window_start_s - origin
            df.loc[group.index, "aligned_end_s"] = group.window_end_s - origin
        df["firing_rate_hz"] = df.spike_count / df.duration_s
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    names = data[["preparation_id", "display_name"]].drop_duplicates()
    duplicates = set(names.loc[names.display_name.duplicated(keep=False), "display_name"])
    data["display_label"] = data.apply(lambda r: f"{r.display_name} [{r.preparation_id[-8:]}]" if r.display_name in duplicates else r.display_name, axis=1)
    return data


def normalize(data, baseline_range_min=None):
    if baseline_range_min is not None:
        if len(baseline_range_min) != 2 or not np.isfinite(baseline_range_min).all() or not 0 <= baseline_range_min[0] < baseline_range_min[1]:
            raise ValueError("Baseline reference range must have 0 <= start < end (minutes from first Baseline window).")
    data = data.copy()
    notices, baselines = [], []
    for keys, group in data.groupby(KEYS, sort=False):
        base = group[group.epoch_label == "Baseline"]
        rate, duration = np.nan, 0.
        start, end = np.nan, np.nan
        status = "ok"
        if base.empty:
            status = "Missing Baseline epoch; normalization excluded."
        else:
            if baseline_range_min is None:
                end = base.window_end_s.max()
                start = end - 1200
            else:
                origin = base.window_start_s.min()
                start, end = [origin + value * 60 for value in baseline_range_min]
            overlap = np.maximum(0, np.minimum(base.window_end_s, end) - np.maximum(base.window_start_s, start))
            duration = float(overlap.sum())
            if duration > 0:
                rate = float(np.sum(base.firing_rate_hz * overlap) / duration)
            if not np.isfinite(rate) or rate <= 0:
                status = "Zero or invalid baseline mean; normalization excluded."
            elif duration < end - start - 1e-6:
                status = f"Only {duration / 60:g} minutes of baseline available in requested {(end-start)/60:g} minutes; using available data."
        if status != "ok":
            notices.append(f"{keys[0]} / {keys[2]}: {status}")
        valid = np.isfinite(rate) and rate > 0
        data.loc[group.index, "baseline_reference_start_s"] = start
        data.loc[group.index, "baseline_reference_end_s"] = end
        data.loc[group.index, "baseline_rate_hz"] = rate
        data.loc[group.index, "baseline_duration_s"] = duration
        data.loc[group.index, "normalized_rate_pct"] = 100 * group.firing_rate_hz / rate if valid else np.nan
        baselines.append(dict(zip(KEYS, keys), baseline_rate_hz=rate, baseline_duration_s=duration, normalization_valid=valid, baseline_reference_start_s=start, baseline_reference_end_s=end, display_name=group.display_name.iloc[0], display_label=group.display_label.iloc[0], notice=status))
    rows = []
    for keys, group in data.groupby(KEYS + ["epoch_label"], sort=False):
        whole_epoch_rate = group.spike_count.sum() / group.duration_s.sum()
        base = group.baseline_rate_hz.iloc[0]
        # The Baseline summary is the defined reference (100%); its complete
        # recorded trace remains unmodified in the time-window export/plot.
        is_baseline = keys[-1] == "Baseline"
        rate = base if is_baseline else whole_epoch_rate
        rows.append(dict(zip(KEYS + ["epoch_label"], keys), mean_rate_hz=rate, display_name=group.display_name.iloc[0], display_label=group.display_label.iloc[0],
                         baseline_reference_start_s=group.baseline_reference_start_s.iloc[0], baseline_reference_end_s=group.baseline_reference_end_s.iloc[0],
                         whole_epoch_mean_rate_hz=whole_epoch_rate,
                         summary_duration_s=group.baseline_duration_s.iloc[0] if is_baseline else group.duration_s.sum(),
                         epoch_duration_s=group.duration_s.sum(), baseline_rate_hz=base,
                         baseline_duration_s=group.baseline_duration_s.iloc[0],
                         normalized_rate_pct=100 * rate / base if np.isfinite(base) and base > 0 else np.nan))
    return data, pd.DataFrame(rows), pd.DataFrame(baselines), notices


def summarize(frame, keys, value):
    rows = []
    for key, group in frame.groupby(keys, sort=False):
        rows.append(dict(zip(keys, key if isinstance(key, tuple) else (key,)), **mean_sem(group[value])))
    return pd.DataFrame(rows, columns=keys + ["mean", "sem", "n_preparations"])


def epoch_order(labels):
    def key(label):
        return (label != "Baseline", [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", label)])
    return sorted(set(labels), key=key)


def sample_label(n):
    return f"n = {int(n)} preparation{'s' if n != 1 else ''}"


def plot_spec(kind, default_title, **selection):
    identity = json.dumps([kind, selection], sort_keys=True, ensure_ascii=True)
    plot_id = kind + "_" + hashlib.sha256(identity.encode()).hexdigest()[:20]
    return dict(plot_id=plot_id, kind=kind, default_title=default_title, **selection)


def plot_specs(data, epochs, mode):
    specs = []
    if mode in ("overlay", "both", "all"):
        for channel, current in data.groupby(["channel", "current_na"]).groups:
            specs.append(plot_spec("overlay", f"Raw spike counts: {channel} / {current:g} nA", channel=channel, current_na=float(current)))
    if mode in ("normalized", "both", "all"):
        for (prep, channel), group in data.groupby(["preparation_id", "channel"]):
            if group.normalized_rate_pct.notna().any():
                for kind, label in (("normalized", "Normalized time course"), ("epoch_means", "Epoch means")):
                    specs.append(plot_spec(kind, f"{label}: {group.display_label.iloc[0]} / {channel}", preparation_id=prep, channel=channel))
        for (channel, epoch), group in epochs.groupby(["channel", "epoch_label"]):
            if epoch.lower().startswith("post") and group.normalized_rate_pct.notna().any():
                specs.append(plot_spec("current", f"Current-intensity response: {channel} / {epoch}", channel=channel, epoch_label=epoch))
    if mode in ("timeline", "all"):
        for (channel, current), group in data.groupby(["channel", "current_na"]):
            specs.append(plot_spec("time_course_absolute_hz", f"Experimental firing-rate time course: {channel} / {current:g} nA", channel=channel, current_na=float(current)))
            if group.normalized_rate_pct.notna().any():
                specs.append(plot_spec("time_course_percent_baseline", f"Baseline-normalized experimental time course: {channel} / {current:g} nA", channel=channel, current_na=float(current)))
    return specs


def discover_plot_specs(paths, currents, mode="both", display_names=None, baseline_range_min=None):
    from h2o2_recordings import split_conditions
    numeric, controls = split_conditions(paths, currents)
    if controls:
        if not numeric:
            return []
        paths, currents = zip(*numeric)
    data, epochs, _, _ = normalize(load_preparations(paths, currents, display_names), baseline_range_min)
    return plot_specs(data, epochs, mode)


def validate_titles(titles):
    if titles is None:
        return {}
    if not isinstance(titles, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in titles.items()):
        raise ValueError("Plot titles must be a JSON object mapping plot IDs to title strings.")
    return titles


def resolved_title(spec, titles):
    return titles.get(spec["plot_id"], "").strip() or spec["default_title"]


RECORDING_SYMBOLS = {"Automatic": "", "Circle": "o", "Square": "s", "Triangle up": "^", "Triangle down": "v", "Diamond": "D", "Pentagon": "p", "Hexagon": "h", "Plus (filled)": "P", "Cross (filled)": "X", "Star": "*", "Triangle left": "<", "Triangle right": ">"}


def preparation_styles(data, symbols=None):
    # Stable across selection order, plots, epochs and current-intensity groups.
    markers = ("o", "s", "^", "v", "P", "X", "<", ">")
    styles = {}
    symbols = validate_titles(symbols)
    if any(v not in RECORDING_SYMBOLS.values() for v in symbols.values()):
        raise ValueError("Unsupported recording symbol. Use a documented marker code or blank for automatic.")
    symbols = {**symbols, **{str(Path(k).resolve()): v for k, v in symbols.items() if Path(k).is_absolute()}}
    for prep in data.preparation_id.unique():
        number = int(hashlib.sha256(prep.encode()).hexdigest()[:8], 16)
        styles[prep] = dict(color=plt.get_cmap("tab10")(number % 10), marker=markers[(number // 10) % len(markers)])
        source = data.loc[data.preparation_id == prep, "source_file"].iloc[0]
        custom = symbols.get(source, symbols.get(prep, ""))
        if custom:
            styles[prep]["marker"] = custom
    return styles


def finish_figure(fig, ax, title, legend_ax=None):
    ax.set_ylim(0, max(1, ax.get_ylim()[1], 105 if "% of baseline" in ax.get_ylabel() else 1))
    wrapped = "\n".join(textwrap.fill(line, width=max(45, int(fig.get_figwidth() * 8))) for line in title.splitlines())
    fig.set_figheight(fig.get_figheight() + max(0, wrapped.count("\n")) * .23)
    ax.set_title(wrapped, fontsize=12, pad=30)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        labels = [textwrap.fill(label, 80) for label in labels]
        (legend_ax or ax).legend(handles, labels, fontsize=7, loc="upper left", bbox_to_anchor=(0, -.3))
    fig.tight_layout()


def epoch_figure(data, title, value, ylabel, styles, stats=None, selected_n=None):
    epochs = epoch_order(data.epoch_label)
    if stats is not None:
        fig, (ax, counts_ax) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [5, 1]})
        counts_ax.set_ylabel("Contributing\npreparations", fontsize=9)
        counts_ax.set_yticks(sorted(set(stats.n_preparations)))
        counts_ax.set_ylim(0, max(2, stats.n_preparations.max()) + 1)
        counts_ax.grid(axis="y", alpha=.2)
        ax.text(0, 1.025, f"Selected in this current/channel group: {sample_label(data.preparation_id.nunique())}; selected overall: {sample_label(selected_n)}",
                transform=ax.transAxes, fontsize=9)
    else:
        fig, ax = plt.subplots(figsize=(14, 6))
        counts_ax = None
        ax.text(0, 1.025, sample_label(data.preparation_id.nunique()), transform=ax.transAxes, fontsize=9)
    offset = 0.
    seen = set()
    for i, epoch in enumerate(epochs):
        subset = data[data.epoch_label == epoch]
        width = subset.aligned_end_s.max() / 60
        ax.axvspan(offset, offset + width, color=("#eeeeee", "#fff2eb", "#eaf5fc")[i % 3], zorder=0)
        ax.text(offset + width / 2, .98, epoch, transform=ax.get_xaxis_transform(), ha="center", va="top")
        for prep, group in subset.groupby("preparation_id", sort=False):
            group = group.sort_values("aligned_start_s")
            x, y, previous = [], [], None
            for row in group.itertuples():
                if previous is not None and row.aligned_start_s > previous + 1e-8:
                    x.append(np.nan)
                    y.append(np.nan)
                x.append(offset + row.aligned_start_s / 60)
                y.append(getattr(row, value))
                previous = row.aligned_end_s
            ax.plot(x, y, ms=4, lw=1, alpha=.85, zorder=4, markerfacecolor="none", **styles[prep],
                    label=group.display_label.iloc[0] if prep not in seen else None)
            seen.add(prep)
        if stats is not None:
            s = stats[stats.epoch_label == epoch].sort_values("aligned_start_s")
            for duration, segment in s.groupby(s.aligned_end_s - s.aligned_start_s):
                segments = (segment.aligned_start_s > segment.aligned_end_s.shift() + 1e-8).cumsum()
                for _, part in segment.groupby(segments):
                    x = offset + part.aligned_start_s.to_numpy() / 60
                    ax.plot(x, part["mean"], color="black", marker="_", ms=8, lw=1.4, zorder=3,
                            label="Group mean" if "mean" not in seen else None)
                    seen.add("mean")
                    errors = part[part["sem"].notna()]
                    if not errors.empty:
                        ax.errorbar(offset + errors.aligned_start_s / 60, errors["mean"], yerr=errors["sem"],
                                    fmt="none", color="black", capsize=2, elinewidth=.9, zorder=2,
                                    label="Mean ± SEM" if "sem" not in seen else None)
                        seen.add("sem")
                    # One strip point per contributing window, preserving gaps.
                    counts_ax.plot(x, part.n_preparations, ".-", drawstyle="steps-post", lw=1, ms=3,
                                   label=f"{duration:g} s windows")
                    runs = (part.n_preparations != part.n_preparations.shift()).cumsum()
                    for _, run in part.groupby(runs):
                        mid = offset + (run.aligned_start_s.min() + run.aligned_end_s.max()) / 120
                        counts_ax.text(mid, run.n_preparations.iloc[0] + .18, sample_label(run.n_preparations.iloc[0]),
                                       ha="center", fontsize=7)
            if s.n_preparations.nunique() == 1:
                ax.text(offset + width / 2, .92, sample_label(s.n_preparations.iloc[0]), transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
        offset += width + 2
        if i < len(epochs) - 1:
            ax.axvline(offset - 1, ls="--", color="gray")
            ax.text(offset - 1, .6, f"{data.current_na.iloc[0]:g} nA", rotation=90, transform=ax.get_xaxis_transform(), ha="center")
    if value == "normalized_rate_pct":
        ax.axhline(100, color="gray", ls="--", lw=1)
    ax.set_ylabel(ylabel)
    (counts_ax or ax).set_xlabel("Aligned time within separate epochs (min)")
    finish_figure(fig, ax, title, counts_ax)
    return fig


def current_figure(points, summary, title, styles, publication=None, colors=None, plot_id=""):
    from current_response_publication import response_figure
    return response_figure(points, summary, title, styles, publication, colors, plot_id)


def epoch_window_statistics(data):
    """Within-recording weighted SEM and Welch tests; normalization is untouched.

    Windows are the observational units, not independent preparations. Unequal
    exposure uses effective sample size (a weighted Welch approximation).
    """
    from scipy.stats import ttest_ind_from_stats
    rows = []
    for keys, group in data.groupby(KEYS, sort=False):
        summaries = {}
        for epoch, windows in group.groupby("epoch_label", sort=False):
            weights = windows.duration_s.to_numpy(dtype=float)
            if epoch == "Baseline":
                weights = np.maximum(0., np.minimum(windows.window_end_s, windows.baseline_reference_end_s)
                                     - np.maximum(windows.window_start_s, windows.baseline_reference_start_s))
                weights = np.asarray(weights, dtype=float)
            rates = windows.firing_rate_hz.to_numpy(dtype=float)
            valid = (weights > 0) & np.isfinite(rates)
            rates, weights = rates[valid], weights[valid]
            n = len(rates)
            mean, sd, effective_n = np.nan, np.nan, 0.
            if n:
                mean = float(np.average(rates, weights=weights))
                effective_n = float(weights.sum() ** 2 / np.square(weights).sum())
                if n > 1:
                    variance = np.sum(weights * (rates - mean) ** 2) / (weights.sum() - np.square(weights).sum() / weights.sum())
                    sd = float(np.sqrt(max(0., variance)))
            summaries[epoch] = dict(window_mean_rate_hz=mean, window_sd_hz=sd,
                                    n_windows=n, effective_n_windows=effective_n)
        baseline = summaries.get("Baseline", {})
        reference = float(group.baseline_rate_hz.iloc[0])
        valid_reference = np.isfinite(reference) and reference > 0
        for epoch, summary in summaries.items():
            n = summary["effective_n_windows"]
            sem = summary["window_sd_hz"] / np.sqrt(n) if n > 1 else np.nan
            p, statistic = np.nan, np.nan
            status = "reference epoch" if epoch == "Baseline" else "insufficient windows"
            if epoch != "Baseline" and valid_reference and n > 1 and baseline.get("effective_n_windows", 0) > 1:
                if summary["window_sd_hz"] == 0 and baseline["window_sd_hz"] == 0:
                    status = "zero variance; t-test undefined"
                else:
                    result = ttest_ind_from_stats(summary["window_mean_rate_hz"], summary["window_sd_hz"], n,
                                                  baseline["window_mean_rate_hz"], baseline["window_sd_hz"],
                                                  baseline["effective_n_windows"], equal_var=False)
                    statistic, p = float(result.statistic), float(result.pvalue)
                    status = "ok" if np.isfinite(p) else "t-test undefined"
            if not valid_reference:
                status = "invalid baseline reference"
            rows.append(dict(zip(KEYS, keys), epoch_label=epoch, **summary,
                             baseline_n_windows=baseline.get("n_windows", 0),
                             baseline_effective_n_windows=baseline.get("effective_n_windows", 0.),
                             normalized_sem_pct=100 * sem / reference if valid_reference else np.nan,
                             t_statistic=statistic, p_value=p, test_status=status,
                             test="two-sided duration-weighted Welch t-test (within recording)",
                             error_bar="SEM across time windows; baseline denominator held fixed"))
    result = pd.DataFrame(rows)
    # Control the family of baseline comparisons separately per recording/channel.
    result["p_value_holm"] = np.nan
    for _, sub in result.groupby(KEYS, sort=False):
        ordered = sub[sub.p_value.notna()].sort_values("p_value")
        if not ordered.empty:
            adjusted = np.minimum(1., np.maximum.accumulate(ordered.p_value.to_numpy() * np.arange(len(ordered), 0, -1)))
            result.loc[ordered.index, "p_value_holm"] = adjusted
    return result


def baseline_change_figure(epochs, title, window_stats=None):
    """Bars originate at 100%; uncertainty and tests use recorded windows."""
    values = epochs.normalized_rate_pct.to_numpy(dtype=float)
    statistics = window_stats.set_index("epoch_label").reindex(epochs.index) if window_stats is not None else None
    errors = statistics.normalized_sem_pct.to_numpy(dtype=float) if statistics is not None else np.full(len(epochs), np.nan)
    x = np.arange(len(epochs))
    with plt.rc_context({"font.family": "Arial", "font.size": 12,
                         "axes.labelsize": 14, "axes.linewidth": 1.1}):
        fig, ax = plt.subplots(figsize=(max(4.4, .8 * len(epochs) + 2.2), 4.3))
        ax.bar(x, values - 100., bottom=100., width=.62,
               color=["white" if label == "Baseline" else "black" for label in epochs.index],
               edgecolor="black", linewidth=1.1, zorder=2)
        finite = np.isfinite(values)
        ax.scatter(x[finite], values[finite], s=18, color="#555555",
                   edgecolor="none", alpha=.75, zorder=3)
        error_mask = finite & np.isfinite(errors)
        if error_mask.any():
            ax.errorbar(x[error_mask], values[error_mask], yerr=errors[error_mask],
                        fmt="none", ecolor="black", elinewidth=1., capsize=4, capthick=1., zorder=4)
        ax.axhline(100, color="black", linewidth=1., zorder=1)
        ax.set_xticks(x, epochs.index)
        ax.set_xlim(-.45, len(epochs) - .55)
        error_extent = np.where(np.isfinite(errors), errors, 0.)
        low = min(100., float((values - error_extent)[finite].min())) if finite.any() else 100.
        high = max(100., float((values + error_extent)[finite].max())) if finite.any() else 100.
        span = max(high - low, 10.)
        comparisons = []
        if statistics is not None and "Baseline" in epochs.index:
            baseline_x = epochs.index.get_loc("Baseline")
            for xi, (epoch, stat) in enumerate(statistics.iterrows()):
                if epoch == "Baseline":
                    continue
                p = stat.p_value_holm
                label = "n/a" if not np.isfinite(p) else ("****" if p < .0001 else "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns")
                comparisons.append((xi, label))
            for bracket, (xi, label) in enumerate(comparisons):
                y = high + (.08 + .12 * bracket) * span
                h = .04 * span
                ax.plot([baseline_x, baseline_x, xi, xi], [y, y+h, y+h, y], color="black", lw=1.)
                ax.text((baseline_x+xi)/2, y+h, label, ha="center", va="bottom", weight="bold", fontsize=12)
        ax.set_ylim(low - .12 * span, high + max(.24, .16 + .12 * len(comparisons)) * span)
        ax.set_ylabel("Firing rate (% of baseline)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=4, width=1., pad=6)
        wrapped = "\n".join(textwrap.fill(line, width=48) for line in title.splitlines())
        extra_height = max(0, wrapped.count("\n")) * .23
        fig.set_figheight(fig.get_figheight() + extra_height)
        fig.suptitle(wrapped, y=.98, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 1 - (.35 + extra_height) / fig.get_figheight()))
    fig.epoch_window_statistics = window_stats
    return fig


def render_plot(spec, data, epochs, overlay, groups, titles, styles, publication=None):
    title = resolved_title(spec, titles)
    channel = spec["channel"]
    subset = data[data.channel == channel]
    if spec["kind"] == "overlay":
        subset = subset[subset.current_na == spec["current_na"]]
        stats = overlay[(overlay.channel == channel) & (overlay.current_na == spec["current_na"])]
        return epoch_figure(subset, title, "spike_count", "Spike count per window", styles, stats, data.preparation_id.nunique())
    if spec["kind"] in ("normalized", "epoch_means"):
        subset = subset[subset.preparation_id == spec["preparation_id"]]
        if spec["kind"] == "normalized":
            return epoch_figure(subset, title, "normalized_rate_pct", "Firing rate (% of baseline)", styles)
        e = epochs[(epochs.preparation_id == spec["preparation_id"]) & (epochs.channel == channel)].set_index("epoch_label").reindex(epoch_order(subset.epoch_label))
        return baseline_change_figure(e, title, epoch_window_statistics(subset))
    points = epochs[(epochs.channel == channel) & (epochs.epoch_label == spec["epoch_label"]) & epochs.normalized_rate_pct.notna()]
    summary = groups[(groups.channel == channel) & (groups.epoch_label == spec["epoch_label"])]
    from current_response_publication import current_colors
    return current_figure(points, summary, title, styles, publication, current_colors(data.current_na), spec["plot_id"])


MANIFEST_NAME = "preparation_plot_manifest.json"


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def managed_path(folder, relative):
    """Resolve before any replacement/deletion; never follow paths outside output."""
    path = folder / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.resolve().is_relative_to(folder.resolve()) or path.is_symlink():
        raise ValueError(f"Unsafe path in output manifest: {relative}")
    if not (relative == "analysis_notes.txt" or (len(Path(relative).parts) == 2 and Path(relative).parts[0] in ("CSVs", "Plots"))):
        raise ValueError(f"Unexpected managed output path: {relative}")
    return path


def publish_outputs(folder, staging, fingerprint, mode, specs):
    """Commit complete outputs, then remove stale, unchanged tool-owned files.

    The two modes can share a folder for an unchanged selection. A changed
    selection invalidates both families. Modified/unmanaged files are protected.
    """
    manifest_path = folder / MANIFEST_NAME
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    old_files = old.get("files", {})
    for relative, metadata in old_files.items():
        path = managed_path(folder, relative)
        if path.exists() and file_hash(path) != metadata["sha256"]:
            raise ValueError(f"Previously generated file was modified: {path}. Use another output folder to preserve it.")
    staged_files = [p for p in staging.rglob("*") if p.is_file()]
    for src in staged_files:
        relative = src.relative_to(staging).as_posix()
        dest = managed_path(folder, relative)
        if dest.exists() and relative not in old_files:
            # The previous SPIE version wrote this recognizable notes file at
            # the root before manifests existed. Only this signature is migrated.
            legacy_notes = relative == "analysis_notes.txt" and dest.read_text(encoding="utf-8").startswith("Rates are duration weighted. Baseline uses final 1200 seconds")
            if not legacy_notes:
                raise ValueError(f"Output would overwrite an unrelated file: {dest}. Choose another output folder.")
    same_selection = old.get("selection") == fingerprint
    active_kinds = {spec["kind"] for spec in specs}
    if mode in ("normalized", "both", "all"):
        active_kinds.update(("normalized", "epoch_means", "current"))
    if mode in ("timeline", "all"):
        active_kinds.update(("time_course_absolute_hz", "time_course_percent_baseline", "timeline_table"))
    if mode in ("overlay", "both", "all"):
        active_kinds.add("overlay")
    keep = {relative: meta for relative, meta in old_files.items()
            if same_selection and meta.get("kind") not in active_kinds and meta.get("kind") != "table"}
    # Keep the catalog consistent with compatible plots retained from the other
    # tool, including their saved custom titles.
    old_catalog = folder / "CSVs/plot_catalog.csv"
    if keep and "CSVs/plot_catalog.csv" in old_files and old_catalog.exists():
        previous_catalog = pd.read_csv(old_catalog)
        kept_ids = {Path(relative).stem for relative in keep if relative.startswith("Plots/")}
        previous_catalog = previous_catalog[previous_catalog.plot_id.isin(kept_ids)]
        catalog_path = staging / "CSVs/plot_catalog.csv"
        pd.concat([previous_catalog, pd.read_csv(catalog_path)], ignore_index=True).drop_duplicates("plot_id", keep="last").to_csv(catalog_path, index=False)
    kinds = {spec["plot_id"]: spec["kind"] for spec in specs}
    for src in staged_files:
        relative = src.relative_to(staging).as_posix()
        dest = managed_path(folder, relative)
        dest.parent.mkdir(parents=True, exist_ok=True)
        digest = file_hash(src)
        src.replace(dest)
        keep[relative] = {"sha256": digest, "kind": kinds.get(src.stem, "current" if src.stem == "current_response_presentation" else "timeline_table" if src.stem.startswith("experimental_") else "table")}
    for relative in old_files.keys() - keep.keys():
        path = managed_path(folder, relative)
        if path.exists():
            path.unlink()
    new_manifest = dict(version=1, selection=fingerprint, files=keep)
    temporary = folder / (MANIFEST_NAME + ".tmp")
    temporary.write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)


def run_analysis(paths, currents, out_dir, mode="normalized", titles=None, display_names=None, baseline_range_min=None, timeline_config=None, symbols=None, publication=None, h2o2_config=None):
    from h2o2_recordings import split_conditions, run_control_analysis
    numeric, controls = split_conditions(paths, currents)
    if controls:
        return run_control_analysis(paths, currents, out_dir, mode, titles, display_names,
                                    baseline_range_min, timeline_config, symbols, publication, h2o2_config)
    if mode not in ("normalized", "overlay", "both", "timeline", "all"):
        raise ValueError("Unknown plotting mode.")
    from current_response_publication import publication_options
    publication = publication_options(publication)
    titles = validate_titles(titles)
    raw = load_preparations(paths, currents, display_names)
    data, epochs, baselines, notices = normalize(raw, baseline_range_min)
    timeline_data = timeline_groups = treatments = None
    if mode in ("timeline", "all"):
        from spike_experimental_timeline import build_experimental_timeline, summarize_timeline
        timeline_data, treatments = build_experimental_timeline(data, timeline_config)
        timeline_groups = summarize_timeline(timeline_data)
    overlay = summarize(data, ["channel", "current_na", "epoch_label", "aligned_start_s", "aligned_end_s"], "spike_count")
    groups = summarize(epochs, ["channel", "current_na", "epoch_label"], "normalized_rate_pct")
    specs = plot_specs(data, epochs, mode)
    # Input content, path, and assigned intensity determine whether older plots
    # remain compatible. Selection ordering does not affect this fingerprint.
    fingerprint = hashlib.sha256(json.dumps([sorted((str(Path(p).resolve()), float(c), file_hash(p)) for p, c in zip(paths, currents)), baseline_range_min, display_names or {}, symbols or {}], sort_keys=True).encode()).hexdigest()
    folder = Path(out_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    for notice in notices:
        print("NOTICE: " + notice, flush=True)
    styles = preparation_styles(data, symbols)
    for frame in (data, epochs, baselines):
        frame["recording_symbol"] = frame.preparation_id.map(lambda prep: styles[prep]["marker"])
    if timeline_data is not None:
        timeline_data["recording_symbol"] = timeline_data.preparation_id.map(lambda prep: styles[prep]["marker"])
    with tempfile.TemporaryDirectory(prefix=".spie-staging-", dir=folder) as temporary:
        staging = Path(temporary).resolve()
        assert staging.parent == folder  # Bound automatic temporary cleanup.
        (staging / "CSVs").mkdir()
        (staging / "Plots").mkdir()
        for name, frame in (("time_windows", data), ("epoch_summaries", epochs), ("baseline_references", baselines), ("overlay_summary", overlay), ("current_intensity_summary", groups)):
            frame.to_csv(staging / "CSVs" / f"{name}.csv", index=False)
        if mode in ("normalized", "both", "all"):
            epoch_window_statistics(data).to_csv(staging / "CSVs/epoch_window_statistics.csv", index=False)
        if timeline_data is not None:
            from spike_experimental_timeline import TIME_KEYS
            used = timeline_data.merge(timeline_groups.drop(columns=["plot_midpoint_min"]), on=TIME_KEYS, how="left", validate="many_to_one")
            used.to_csv(staging / "CSVs/experimental_time_windows.csv", index=False)
            timeline_groups.to_csv(staging / "CSVs/experimental_group_time_points.csv", index=False)
            treatments.to_csv(staging / "CSVs/experimental_treatments.csv", index=False)
            settings = dict(baseline_range_min=baseline_range_min, timeline=timeline_config or {})
            pd.DataFrame([{"settings_json": json.dumps(settings)}]).to_csv(staging / "CSVs/experimental_settings.csv", index=False)
        catalog = []
        publication_rows = []
        for spec in specs:
            if spec["kind"].startswith("time_course_"):
                from spike_experimental_timeline import timeline_figure
                mask = (timeline_data.channel == spec["channel"]) & (timeline_data.current_na == spec["current_na"])
                summary = timeline_groups[(timeline_groups.channel == spec["channel"]) & (timeline_groups.current_na == spec["current_na"])]
                fig = timeline_figure(timeline_data[mask], summary, treatments, resolved_title(spec, titles), spec["kind"] == "time_course_percent_baseline", styles, timeline_config)
            else:
                fig = render_plot(spec, data, epochs, overlay, groups, titles, styles, publication)
            try:
                for suffix in (("png", "svg", "pdf") if spec["kind"] == "current" else ("png", "svg")):
                    if spec["kind"] == "current":
                        with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
                            fig.savefig(staging / "Plots" / f"{spec['plot_id']}.{suffix}", dpi=600, facecolor="white")
                    else:
                        fig.savefig(staging / "Plots" / f"{spec['plot_id']}.{suffix}", dpi=180, bbox_inches="tight", pad_inches=.18)
                if spec["kind"] == "current":
                    publication_rows.append(dict(plot_id=spec["plot_id"], settings_json=json.dumps(fig.publication_metadata, ensure_ascii=False)))
            finally:
                plt.close(fig)
            catalog.append(dict(spec, title=fig.publication_metadata["title"] if spec["kind"] == "current" else resolved_title(spec, titles), png=f"Plots/{spec['plot_id']}.png", svg=f"Plots/{spec['plot_id']}.svg",
                                presentation_csv="CSVs/current_response_presentation.csv" if spec["kind"] == "current" else ""))
        if mode in ("normalized", "both", "all"):
            pd.DataFrame(publication_rows, columns=["plot_id", "settings_json"]).to_csv(staging / "CSVs/current_response_presentation.csv", index=False)
        pd.DataFrame(catalog, columns=["plot_id", "kind", "channel", "preparation_id", "current_na", "epoch_label", "default_title", "title", "png", "svg", "presentation_csv"]).to_csv(staging / "CSVs/plot_catalog.csv", index=False)
        (staging / "analysis_notes.txt").write_text(
            "Rates are duration weighted. Baseline defaults to final 1200 seconds of labelled Baseline; any explicit range is recorded in baseline_references.csv. The denominator is shared by all Post epochs.\n"
            "Partial bins are weighted by overlap, assuming constant rate within a bin; gaps are not zero activity.\n"
            "Baseline summary is 100%; the complete baseline trace retains its variation.\n"
            "Epoch mean bars originate at 100%. Their error bars are duration-weighted SEM across time windows within each recording, holding the baseline denominator fixed; these are not preparation-level SEM.\n"
            "Epoch brackets compare each non-Baseline epoch with the selected Baseline reference using two-sided Welch t-tests. Unequal window exposures use weighted sample variance and effective n (an approximate weighted Welch test); equal exposures recover ordinary Welch tests.\n"
            "Holm adjustment covers all non-Baseline comparisons per preparation/channel; stars use adjusted p (<.05, .01, .001, .0001); ns means p >= .05; n/a means insufficient windows, invalid baseline or undefined test. Exact p, adjusted p, SEM and window counts are in CSVs/epoch_window_statistics.csv.\n"
            "Within-recording tests assume independent time windows and do not correct temporal autocorrelation; they do not support population inference across preparations. Baseline is the reference and is not tested against itself.\n"
            "Group error bars: Mean ± SEM. SEM = sample SD (ddof=1) / sqrt(n independent preparations).\n"
            "SEM is not a confidence interval or standard deviation. SEM is missing for n < 2.\n"
            "Individual observations remain visible. Channels and time windows never increase independent n.\n"
            "Raw summaries match aligned window boundaries exactly. The lower strip reports contributing n at each window.\n"
            "Only manifest-listed files are current managed outputs; unrelated pre-existing files are untouched.\n"
            "Experimental time courses use observed midpoints and each specimen's own epoch durations. No fabricated observations or interpolation.\n"
            "Group time-course keys require the same epoch and experimental/display window boundaries; channels are separate.\n"
            "Current-response figures show quartile boxes with median and 1.5-IQR whiskers overlaid with the individual preparation means, with descending signed current from left to right.\n"
            "Current-response publication figures use uniform filled circles without legends or sample annotations; effective settings are in CSVs/current_response_presentation.csv when generated.\n"
            "Publication PNGs use 600 dpi at the selected physical width; SVG text stays editable. Deterministic overlap offsets affect display only, are bounded to +/-5 nA and are recorded alongside true currents. PDF embeds TrueType fonts.\n"
            + "\n".join(notices), encoding="utf-8")
        publish_outputs(folder, staging, fingerprint, mode, specs)
    print(f"Saved {len(paths)} independent preparation(s). CSVs: {folder / 'CSVs'}; plots: {folder / 'Plots'}", flush=True)
    return dict(data=data, epochs=epochs, baselines=baselines, overlay=overlay, groups=groups, notices=notices, specs=specs, timeline_data=timeline_data, timeline_groups=timeline_groups, treatments=treatments)


def cli(argv=None, default_mode="normalized"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="One CSV per independent preparation")
    parser.add_argument("--currents", nargs="+", help="One current in nA or H2O2 condition per input, in the same order")
    parser.add_argument("--h2o2-json", help="Optional H2O2 concentration; baseline/during/post epochs are read from each preparation CSV")
    parser.add_argument("--out-dir")
    parser.add_argument("--mode", choices=["normalized", "overlay", "both", "timeline", "all"], default=default_mode)
    title_args = parser.add_mutually_exclusive_group()
    title_args.add_argument("--titles-json", help="JSON object mapping stable plot IDs to custom titles")
    title_args.add_argument("--titles-file", help="UTF-8 JSON file mapping stable plot IDs to custom titles")
    parser.add_argument("--write-title-template", help="Write editable JSON title mapping and exit without generating plots")
    name_args = parser.add_mutually_exclusive_group()
    name_args.add_argument("--names-json", help="Display-name mapping keyed by absolute source path or preparation ID")
    name_args.add_argument("--names-file", help="UTF-8 JSON display-name mapping")
    symbol_args = parser.add_mutually_exclusive_group()
    symbol_args.add_argument("--symbols-json", help="Marker codes keyed by absolute source path or preparation ID")
    symbol_args.add_argument("--symbols-file", help="UTF-8 JSON recording-symbol mapping")
    publication_args = parser.add_mutually_exclusive_group()
    publication_args.add_argument("--publication-json", help="Current-response presentation configuration JSON")
    publication_args.add_argument("--publication-file", help="UTF-8 current-response presentation configuration")
    parser.add_argument("--baseline-range-min", nargs=2, type=float, help="Reference start/end minutes from first Baseline window; default final 20 minutes")
    timing_args = parser.add_mutually_exclusive_group()
    timing_args.add_argument("--timeline-json", help="Experimental timing settings JSON")
    timing_args.add_argument("--timeline-file", help="UTF-8 JSON experimental timing settings")
    args = parser.parse_args(argv)
    if not args.inputs:
        from tkinter import Tk, filedialog, simpledialog
        root = Tk()
        root.withdraw()
        try:
            args.inputs = list(filedialog.askopenfilenames(initialdir=str(DEFAULT_FOLDER), filetypes=[("Spike counts", "*.csv")]))
            if not args.inputs:
                return
            args.currents = [simpledialog.askfloat("Current intensity", f"Current (nA) for {Path(p).name}", initialvalue=-200, parent=root) for p in args.inputs]
            if any(c is None for c in args.currents):
                return
        finally:
            root.destroy()
    if args.currents is None:
        parser.error("Supply --currents with one numeric current intensity per file.")
    output = args.out_dir or str(Path(args.inputs[0]).parent / f"{Path(args.inputs[0]).stem}_{args.mode}_plots")
    try:
        names = json.loads(Path(args.names_file).read_text(encoding="utf-8-sig") if args.names_file else args.names_json or "{}")
        symbols = json.loads(Path(args.symbols_file).read_text(encoding="utf-8-sig") if args.symbols_file else args.symbols_json or "{}")
        publication = json.loads(Path(args.publication_file).read_text(encoding="utf-8-sig") if args.publication_file else args.publication_json or "{}")
        timing = json.loads(Path(args.timeline_file).read_text(encoding="utf-8-sig") if args.timeline_file else args.timeline_json or "{}")
        if args.write_title_template:
            specs = discover_plot_specs(args.inputs, args.currents, args.mode, names, args.baseline_range_min)
            Path(args.write_title_template).write_text(json.dumps({s["plot_id"]: s["default_title"] for s in specs}, indent=2, ensure_ascii=False), encoding="utf-8")
            return
        titles = json.loads(Path(args.titles_file).read_text(encoding="utf-8-sig") if args.titles_file else args.titles_json or "{}")
        run_analysis(args.inputs, args.currents, output, args.mode, titles, names, args.baseline_range_min, timing, symbols, publication, json.loads(args.h2o2_json or "{}"))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    cli()


