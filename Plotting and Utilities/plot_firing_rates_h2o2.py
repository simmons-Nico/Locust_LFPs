"""Extend the saved cathodic firing-rate plot with paired H2O2 controls.

Run: python "Plotting and Utilities/plot_firing_rates_h2o2.py"
Edit the inputs below and rerun. Rates are Hz, with one value per preparation,
not per recording window. Empty dictionaries produce an explicitly empty panel.
The existing Post 1 and Post 2 summaries are exported separately, never pooled.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATHODIC_EPOCH_SUMMARIES = ROOT / "Analysis Outputs/preparation_publication_validation/double/CSVs/epoch_summaries.csv"
OUTPUT_DIR = ROOT / "Analysis Outputs/firing_rates_h2o2"

# Supply the concentration with units, e.g. a string from your experiment log.
H2O2_CONCENTRATION = None
# Keys must be the actual, identical locust IDs across these dictionaries.
# Both measurements use the ONE pre-H2O2 reference for that same preparation.
PRE_H2O2_BASELINE_RATES_HZ = {}
DURING_H2O2_FIRING_RATES_HZ = {}
AFTER_H2O2_FIRING_RATES_HZ = {}
# A missing measurement may be omitted or set to None; it is never imputed.


def normalize_h2o2(baselines, during, after):
    """Return one row per locust, preserving missing measurements without pairing them."""
    ids = sorted(set(baselines) | set(during) | set(after))
    rows = []
    for locust in ids:
        if not isinstance(locust, str) or not locust.strip():
            raise ValueError("Each locust ID must be a nonempty string.")
        baseline = baselines.get(locust)
        if baseline is None or not np.isfinite(baseline) or baseline <= 0:
            raise ValueError(f"{locust}: pre-H2O2 baseline must be finite and > 0 Hz.")
        row = dict(locust_id=locust, pre_h2o2_baseline_hz=baseline)
        for phase, rates in (("during", during), ("after", after)):
            rate = rates.get(locust)
            if rate is not None and (not np.isfinite(rate) or rate < 0):
                raise ValueError(f"{locust}: {phase} firing rate must be finite and >= 0 Hz, or None.")
            row[f"{phase}_hz"] = rate
            row[f"{phase}_pct"] = np.nan if rate is None else 100 * rate / baseline
        rows.append(row)
    return pd.DataFrame(rows, columns=["locust_id", "pre_h2o2_baseline_hz", "during_hz", "during_pct", "after_hz", "after_pct"])


def firing_rate_figure(cathodic, h2o2, concentration=None):
    """Plot already-normalized cathodic observations and same-locust H2O2 pairs."""
    if cathodic.duplicated(["preparation_id", "current_na"]).any():
        raise ValueError("Select exactly one channel and post epoch per cathodic preparation/current.")
    if not np.isfinite(cathodic.normalized_rate_pct.to_numpy(dtype=float)).all() or (cathodic.normalized_rate_pct < 0).any():
        raise ValueError("Cathodic normalized rates must be finite and nonnegative.")
    if not cathodic.current_na.isin([-100, -200, -300]).all():
        raise ValueError("Expected only -100, -200 and -300 nA.")
    with plt.rc_context({"font.family": "Arial", "font.size": 12,
                         "svg.fonttype": "none", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(9, 4.8),
                                 gridspec_kw={"width_ratios": [3, 2.6]})
        a, b = axes
        for ax, title in zip(axes, ["A. Cathodic DC stimulation", "B. Exogenous H$_2$O$_2$ control"]):
            ax.set_title(title, loc="left", fontsize=14, pad=17)
            ax.axhline(100, color=".5", ls=(0, (4, 3)), lw=.8, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(direction="out", length=3, width=.7, labelsize=12)
        for x, current in enumerate([-100, -200, -300]):
            group = cathodic[cathodic.current_na == current].sort_values("preparation_id")
            values = group.normalized_rate_pct.to_numpy()
            if len(values):
                a.boxplot([values], positions=[x], widths=.28, showfliers=False,
                          manage_ticks=False, medianprops={"color": ".25"},
                          boxprops={"color": ".4"}, whiskerprops={"color": ".4"},
                          capprops={"color": ".4"})
                offsets = np.linspace(-.09, .09, len(group)) if len(group) > 1 else [0]
                a.scatter(x + np.asarray(offsets), values, s=25, c="#6c9fba",
                          edgecolors="#203040", linewidths=.55, zorder=4)
        a.set_xticks([0, 1, 2], ["−100 nA", "−200 nA", "−300 nA"])
        a.set_xlim(-.5, 2.5)
        a.set_ylabel("Firing rate (% of baseline)", fontsize=14)
        if cathodic.empty:
            a.text(.5, .80, "No cathodic measurements\nselected", transform=a.transAxes,
                   ha="center", va="center", color=".4")
        b.set_xticks([0, 1], ["During H$_2$O$_2$", "Post H$_2$O$_2$"])
        b.set_xlim(-.45, 1.65)
        for x, phase in enumerate(("during_pct", "after_pct")):
            values = h2o2[phase].dropna().to_numpy()
            if len(values):
                b.boxplot([values], positions=[x], widths=.28, showfliers=False,
                          manage_ticks=False, medianprops={"color": ".25"},
                          boxprops={"color": ".4"}, whiskerprops={"color": ".4"},
                          capprops={"color": ".4"})
        for i, row in enumerate(h2o2.itertuples()):
            offset = np.linspace(-.08, .08, len(h2o2))[i] if len(h2o2) > 1 else 0
            xs = np.array([0, 1]) + offset
            ys = np.array([row.during_pct, row.after_pct])
            # Each line contains only this locust's two observations. NaN breaks it.
            b.plot(xs, ys, color="#6c9fba", lw=.65, zorder=2)
            b.scatter(xs, ys, s=25, c="#6c9fba", edgecolors="#203040", linewidths=.55, zorder=4)
        if h2o2.empty or not h2o2[["during_pct", "after_pct"]].notna().any().any():
            b.text(.5, .80, "H$_2$O$_2$ measurements\nnot supplied", transform=b.transAxes,
                   ha="center", va="center", color=".4")
        b.text(.5, 1.015, "Concentration: " + (str(concentration) if concentration else "not supplied"),
               transform=b.transAxes, ha="center", fontsize=12, color=".4")
        all_values = [100., *cathodic.normalized_rate_pct.tolist(),
                      *h2o2.during_pct.dropna().tolist(), *h2o2.after_pct.dropna().tolist()]
        a.set_ylim(0, max(all_values) * 1.22)
        fig.subplots_adjust(left=.105, right=.96, bottom=.18, top=.86, wspace=.20)
    return fig


def main():
    epochs = pd.read_csv(CATHODIC_EPOCH_SUMMARIES)
    h2o2 = normalize_h2o2(PRE_H2O2_BASELINE_RATES_HZ, DURING_H2O2_FIRING_RATES_HZ,
                         AFTER_H2O2_FIRING_RATES_HZ)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    h2o2.to_csv(OUTPUT_DIR / "h2o2_normalized_rates.csv", index=False)
    selected = epochs[epochs.epoch_label.str.lower().str.startswith("post") &
                      epochs.current_na.isin([-100, -200, -300])]
    if selected.empty:
        raise ValueError("No cathodic post-stimulation observations found.")
    for (channel, epoch), points in selected.groupby(["channel", "epoch_label"]):
        fig = firing_rate_figure(points, h2o2, H2O2_CONCENTRATION)
        stem = f"firing_rates_h2o2_{channel}_{epoch.replace(' ', '_')}"
        with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
            for suffix in ("svg", "png"):
                fig.savefig(OUTPUT_DIR / f"{stem}.{suffix}", dpi=600, facecolor="white")
        points.to_csv(OUTPUT_DIR / f"{stem}_cathodic_data.csv", index=False)
        plt.close(fig)
        print(OUTPUT_DIR / stem)
    (OUTPUT_DIR / "provenance.json").write_text(json.dumps({
        "cathodic_source": str(CATHODIC_EPOCH_SUMMARIES),
        "cathodic_normalization": "Existing normalized_rate_pct preserved without recalculation",
        "h2o2_concentration": H2O2_CONCENTRATION,
        "h2o2_normalization": "100 * measurement_hz / same_locust_pre_h2o2_baseline_hz",
        "h2o2_status": "Inputs pending" if h2o2.empty else "User-supplied inputs",
        "png_dpi": 600}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
