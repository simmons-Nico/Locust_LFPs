"""One H2O2 CSV per preparation: baseline, during and post epochs."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

DURING = "During H₂O₂ perfusion"
AFTER = "After H₂O₂ perfusion"
H2O2 = "H\u2082O\u2082 positive control"
CONDITIONS = ("-100", "-200", "-300", H2O2)


def parse_condition(value):
    value = str(value).strip()
    if value.casefold() in (H2O2.casefold(), "h2o2", "h2o2 positive control"):
        return H2O2
    try:
        current = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Choose a current in nA or H2O2 positive control (one CSV with baseline, during and post epochs).") from exc
    if not np.isfinite(current):
        raise ValueError("Current intensities must be finite numbers in nA.")
    return current


def split_conditions(paths, conditions):
    if len(paths) != len(conditions) or not paths:
        raise ValueError("Assign one condition to each input file.")
    paths = [str(Path(p).resolve()) for p in paths]
    if len(paths) != len(set(paths)):
        raise ValueError("Select each recording file only once.")
    numeric, controls = [], []
    for path, condition in zip(paths, conditions):
        condition = parse_condition(condition)
        (controls if isinstance(condition, str) else numeric).append((path, condition))
    return numeric, controls


def control_rates(controls, config=None, baseline_range_min=None):
    """Read same-file epochs and pair by preparation/channel, never by user-entered IDs."""
    from preparation_spike_analysis import load_preparations, normalize
    rows, seen = [], set()
    for path, condition in controls:
        path = str(Path(path).resolve())
        if parse_condition(condition) != H2O2:
            raise ValueError("Select H2O2 positive control for a single preparation CSV.")
        if path in seen:
            raise ValueError("Select each H2O2 preparation file only once.")
        seen.add(path)
        # Reuse count/time validation; this temporary loader value is never
        # exported as a current or included in a cathodic group.
        data = load_preparations([path], [0], epoch_aliases={
            "baseline": "Baseline", "during": "During", "post": "Post"})
        _, _, baselines, _ = normalize(data, baseline_range_min)
        for channel, channel_data in data.groupby("channel"):
            match = baselines[baselines.channel == channel].iloc[0]
            if not match.normalization_valid:
                raise ValueError(f"{Path(path).name} / {channel}: missing or nonpositive pre-H2O2 baseline.")
            measurements = channel_data[channel_data.epoch_label != "Baseline"]
            if measurements.empty:
                raise ValueError(f"{Path(path).name} / {channel}: no during or post epoch found.")
            locust = "H2O2-" + channel_data.preparation_id.iloc[0].rsplit("_", 1)[-1]
            for epoch, condition in (("During", DURING), ("Post", AFTER)):
                group = measurements[measurements.epoch_label == epoch]
                if group.empty:
                    continue  # Missing observations are not imputed or paired.
                rate = group.spike_count.sum() / group.duration_s.sum()
                rows.append(dict(locust_id=locust, preparation_id=channel_data.preparation_id.iloc[0],
                                 channel=channel, condition=condition, source_file=path,
                                 measurement_epoch=epoch.casefold(),
                                 measurement_rate_hz=rate, baseline_file=path, baseline_epoch="baseline",
                                 pre_h2o2_baseline_hz=match.baseline_rate_hz,
                                 baseline_reference_start_s=match.baseline_reference_start_s,
                                 baseline_reference_end_s=match.baseline_reference_end_s,
                                 baseline_duration_s=match.baseline_duration_s,
                                 baseline_notice=match.notice,
                                 normalized_rate_pct=100 * rate / match.baseline_rate_hz))
    return pd.DataFrame(rows)


def run_control_analysis(paths, conditions, out_dir, mode, titles=None, display_names=None,
                         baseline_range_min=None, timeline_config=None, symbols=None,
                         publication=None, h2o2_config=None):
    from preparation_spike_analysis import run_analysis, load_preparations, normalize
    from plot_firing_rates_h2o2 import firing_rate_figure, normalize_h2o2
    import matplotlib.pyplot as plt
    numeric, controls = split_conditions(paths, conditions)
    if mode not in ("normalized", "both", "all"):
        raise ValueError("H₂O₂ recording conditions use Run normalized analysis. For current-only overlay or timelines, select cathodic recordings.")
    rates = control_rates(controls, h2o2_config, baseline_range_min)
    if numeric:
        if any(c not in (-100, -200, -300) for _, c in numeric):
            raise ValueError("The combined H₂O₂ figure supports −100, −200 and −300 nA cathodic conditions.")
        # Validate before writing outputs and preserve the standard numerical workflow.
        _, epochs, _, _ = normalize(load_preparations([p for p, _ in numeric], [c for _, c in numeric], display_names), baseline_range_min)
    else:
        epochs = pd.DataFrame(columns=["channel", "epoch_label", "preparation_id", "current_na", "normalized_rate_pct"])
    result = run_analysis([p for p, _ in numeric], [c for _, c in numeric], out_dir, mode,
                          titles, display_names, baseline_range_min, timeline_config, symbols,
                          publication) if numeric else {}
    folder = Path(out_dir) / "H2O2_positive_control"
    folder.mkdir(parents=True, exist_ok=True)
    rates.to_csv(folder / "h2o2_normalized_rates.csv", index=False)
    epochs.to_csv(folder / "cathodic_epoch_summaries.csv", index=False)
    (folder / "settings.json").write_text(json.dumps(dict(h2o2=h2o2_config or {},
        baseline_range_min=baseline_range_min, conditions=dict(zip(map(str, paths), conditions))),
        ensure_ascii=False, indent=2), encoding="utf-8")
    selected = epochs[epochs.epoch_label.str.lower().str.startswith("post") & epochs.normalized_rate_pct.notna()]
    catalog = []
    for channel in sorted(set(rates.channel) | set(selected.channel)):
        group = rates[rates.channel == channel]
        baselines = dict(zip(group.locust_id, group.pre_h2o2_baseline_hz))
        during = dict(zip(group[group.condition == DURING].locust_id, group[group.condition == DURING].measurement_rate_hz))
        after = dict(zip(group[group.condition == AFTER].locust_id, group[group.condition == AFTER].measurement_rate_hz))
        paired = normalize_h2o2(baselines, during, after)
        posts = selected[selected.channel == channel]
        for epoch in sorted(posts.epoch_label.unique()) or ["H2O2_only"]:
            points = posts[posts.epoch_label == epoch]
            fig = firing_rate_figure(points, paired, (h2o2_config or {}).get("concentration"))
            import hashlib
            stem = "firing_rates_" + hashlib.sha256(f"{channel}/{epoch}".encode()).hexdigest()[:12]
            try:
                with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
                    for suffix in ("png", "svg"):
                        fig.savefig(folder / f"{stem}.{suffix}", dpi=600, facecolor="white")
            finally:
                plt.close(fig)
            catalog.append(dict(channel=channel, cathodic_epoch=epoch, png=stem+".png", svg=stem+".svg"))
    pd.DataFrame(catalog).to_csv(folder / "plot_catalog.csv", index=False)
    result["h2o2_rates"] = rates
    print(f"H2O2 paired figures and data: {folder}")
    return result
