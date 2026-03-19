#!/usr/bin/env python3
"""
Overlay spike-count graphs (PBS, Baseline, Post(-600mV)) for multiple sessions/dates
on one shared plot with a shared y-axis.

Input options:
1) Edit the `DATA` dict below (quickest).
2) Or provide a CSV file:
   columns: date,condition,count
   where condition is one of: PBS, Baseline, Post(-600mV)

Trend visibility options:
- Raw counts (default)
- Normalized-to-baseline (% of Baseline) via --normalize baseline
- Log y-scale via --logy (often helps when sessions span orders of magnitude)

Example:
  python overlay_spike_counts.py
  python overlay_spike_counts.py --normalize baseline
  python overlay_spike_counts.py --csv spikes.csv --logy
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---- Option A: hardcode your values here (replace these placeholders) ----
# Keys are labels (e.g., dates). Values are dicts for the conditions.
DATA = {
    "23-01-26": {"PBS": 0, "Baseline": 1420, "Post(-1V)": 90},
    "26-01-26": {"PBS": 0, "Baseline": 2700, "Post(-1V)": 1950},
    "29-01-26": {"PBS": 0, "Baseline": 5200, "Post(-1V)": 500},
    "27-02-26": {"PBS": 0, "Baseline": 1819, "Post(-1V)": 28},
}
# ------------------------------------------------------------------------


CONDITIONS = ["PBS", "Baseline", "Post(-1V)"]


def load_from_csv(csv_path: Path):
    """
    CSV format:
      date,condition,count
      09-12-25,PBS,0
      09-12-25,Baseline,5200
      09-12-25,Post(-1mV),7400
    """
    out = {}
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            date = row["date"].strip()
            cond = row["condition"].strip()
            count = float(row["count"])
            out.setdefault(date, {})[cond] = count

    # basic validation
    missing = []
    for d, vals in out.items():
        for c in CONDITIONS:
            if c not in vals:
                missing.append((d, c))
    if missing:
        msg = "\n".join([f"  - {d} missing {c}" for d, c in missing])
        raise ValueError(f"CSV is missing required condition rows:\n{msg}")

    return out


def transform_counts(counts_by_date, normalize=None):
    """
    normalize:
      None -> raw
      'baseline' -> percent of Baseline (Baseline=100)
    """
    transformed = {}
    for d, vals in counts_by_date.items():
        if normalize is None:
            transformed[d] = [vals[c] for c in CONDITIONS]
        elif normalize == "baseline":
            base = vals["Baseline"]
            if base == 0:
                # avoid divide-by-zero; keep NaNs so it’s obvious
                transformed[d] = [np.nan, np.nan, np.nan]
            else:
                transformed[d] = [100.0 * vals[c] / base for c in CONDITIONS]
        else:
            raise ValueError(f"Unknown normalize mode: {normalize}")
    return transformed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Optional CSV path with date,condition,count")
    ap.add_argument("--normalize", choices=["baseline"], default=None,
                    help="Normalize each session to Baseline (Baseline=100)")
    ap.add_argument("--logy", action="store_true", help="Use log scale on y-axis (raw mode recommended)")
    ap.add_argument("--title", type=str, default="Spike Count Across Conditions (Overlay)")
    ap.add_argument("--out", type=str, default=None, help="Optional output image path (e.g., overlay.png)")
    args = ap.parse_args()

    counts = DATA if args.csv is None else load_from_csv(Path(args.csv))
    series = transform_counts(counts, normalize=args.normalize)

    x = np.arange(len(CONDITIONS))

    fig, ax = plt.subplots(figsize=(9, 5))

    # Plot each date as a separate line on the same axis
    # (markers + slight alpha helps readability)
    for label in sorted(series.keys()):
        y = np.array(series[label], dtype=float)
        ax.plot(x, y, marker="o", linewidth=2, alpha=0.85, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS)
    ax.grid(True, axis="y", alpha=0.3)

    if args.normalize == "baseline":
        ax.set_ylabel("Spike Count (% of Baseline)")
        ax.axhline(100, linewidth=1, alpha=0.6)  # baseline reference
    else:
        ax.set_ylabel("Spike Count")
        if args.logy:
            ax.set_yscale("log")
            ax.set_ylabel("Spike Count (log scale)")

    ax.set_title(args.title)
    ax.legend(title="Session", frameon=True, ncols=2)

    # Give a bit of horizontal padding
    ax.set_xlim(-0.15, len(CONDITIONS) - 0.85)

    plt.tight_layout()


        # --- Automatic save location ---
    save_path = r"C:\Users\simmons\Desktop\Exploring PSDs\overlay.png"

    plt.savefig(save_path, dpi=300)
    print("Saved automatically to:", save_path)



if __name__ == "__main__":
    main()