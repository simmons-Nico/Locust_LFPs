#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot continuous spike counts per channel from a combined CSV summary.

Expected input CSV columns:
    recording_index, recording_name, epoch_label, channel, window_index, spike_count

Behavior:
- Keeps the user settings style:
      CSV_DIR
      CSV_GLOB
      OUT_DIR_NAME
- Automatically finds the combined spike-count CSV inside CSV_DIR
- Creates one continuous plot per channel
- Shades epochs
- Marks recording boundaries
- Uses cleaner, publication-style labels
- Saves all plots into OUT_DIR_NAME
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# =========================
# USER SETTINGS
# =========================
CSV_DIR      = r"C:\Users\simmons\Desktop\Exploring PSDs"
CSV_GLOB     = "*.csv"
OUT_DIR_NAME = "spike_counts_summary"

TRACE_COLOR = "tab:blue"
TRACE_MARKER = "o"
TRACE_LINEWIDTH = 2

Y_LIM_TOP = None
 # set to None for automatic scaling

EPOCH_COLORS = [
    "#929292FF", "#FFE4D6", "#8BD0F8", "#B6FCAF", "#E1C1FD", "#FFEFAA", "#FFBCBC",  "#FFFD9F"
]

PREFERRED_NAME_TOKENS = [
    "Spike_counts_per_min",
    "spike_counts_per_min",
]


def find_input_csv(csv_dir: str, csv_glob: str) -> str:
    """Find the best-matching spike-count summary CSV in csv_dir."""
    csv_files = sorted(glob.glob(os.path.join(csv_dir, csv_glob)))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {csv_dir}")

    for token in PREFERRED_NAME_TOKENS:
        for path in csv_files:
            if token.lower() in os.path.basename(path).lower():
                return path

    if len(csv_files) == 1:
        return csv_files[0]

    candidates = "\n".join(f" - {os.path.basename(p)}" for p in csv_files)
    raise FileNotFoundError(
        "Could not uniquely identify the spike-count summary CSV.\n"
        "Put the correct summary CSV in CSV_DIR or rename it to include one of:\n"
        f"  {PREFERRED_NAME_TOKENS}\n\n"
        f"Found CSV files:\n{candidates}"
    )


def build_epoch_color_map(epoch_series: pd.Series) -> dict:
    unique_epochs = list(dict.fromkeys(epoch_series.astype(str).tolist()))
    return {
        epoch: EPOCH_COLORS[i % len(EPOCH_COLORS)]
        for i, epoch in enumerate(unique_epochs)
    }


def clean_epoch_name(name: str) -> str:
    """Convert verbose epoch names into cleaner plot labels."""
    s = str(name).strip()
    low = s.lower()

    if "baseline" in low:
        return "Baseline"
    if "h2o2" in low:
        return "H2O2 at 0.25 Atm"
    if "81na" in low:
        return "-80 nA"
    if "90na" in low:
        return "-90 nA"
    if "70na" in low:
        return "-70 nA"
    if "60na" in low:
        return "-60 nA"
    if "50na" in low:
        return "-50 nA"

    return s


def main():
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 18,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
    })

    csv_path = find_input_csv(CSV_DIR, CSV_GLOB)
    out_dir = os.path.join(CSV_DIR, OUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Using input CSV:\n  {csv_path}")
    print(f"Saving plots to:\n  {out_dir}")

    df = pd.read_csv(csv_path)

    required_cols = [
        "recording_index",
        "recording_name",
        "epoch_label",
        "channel",
        "window_index",
        "spike_count",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["recording_index"] = pd.to_numeric(df["recording_index"], errors="coerce")
    df["window_index"] = pd.to_numeric(df["window_index"], errors="coerce")
    df["spike_count"] = pd.to_numeric(df["spike_count"], errors="coerce")
    df["recording_name"] = df["recording_name"].astype(str)
    df["epoch_label"] = df["epoch_label"].astype(str)
    df["channel"] = df["channel"].astype(str)

    df = df.dropna(subset=["recording_index", "window_index", "spike_count", "channel"])
    if df.empty:
        raise ValueError("Input CSV has no valid rows after cleaning.")

    df["recording_index"] = df["recording_index"].astype(int)
    df["window_index"] = df["window_index"].astype(int)
    df["epoch_label_clean"] = df["epoch_label"].apply(clean_epoch_name)

   # after:
    # df["epoch_label_clean"] = df["epoch_label"].apply(clean_epoch_name)

    epoch_color_map = build_epoch_color_map(df["epoch_label_clean"])

    for ch in sorted(df["channel"].unique()):
            sub = df[df["channel"] == ch].copy()
            sub = sub.sort_values(["recording_index", "window_index"], kind="stable").reset_index(drop=True)

            fig, ax = plt.subplots(figsize=(16, 5.5))

            x_all = np.arange(len(sub))
            y_all = sub["spike_count"].to_numpy()
            xticklabels = sub["window_index"].tolist()

            epoch_handles = []
            seen_epochs = set()
            boundary_positions = []
            label_positions = []

            # Split into contiguous segments whenever recording_index OR epoch changes
            segment_id = (
                (sub["recording_index"] != sub["recording_index"].shift()) |
                (sub["epoch_label_clean"] != sub["epoch_label_clean"].shift())
            ).cumsum()

            for _, block in sub.groupby(segment_id, sort=False):
                start = block.index[0]
                end = block.index[-1]
                epoch = block["epoch_label_clean"].iloc[0]
                color = epoch_color_map[epoch]

                # shaded background for this epoch segment
                ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.35)

                # legend entry once per epoch
                if epoch not in seen_epochs:
                    epoch_handles.append(
                        Patch(facecolor=color, edgecolor="none", alpha=0.35, label=epoch)
                    )
                    seen_epochs.add(epoch)

                # label position
                midpoint = (start + end) / 2.0
                label_positions.append((midpoint, epoch))

                # draw boundary between segments
                if start > 0:
                    boundary_positions.append(start - 0.5)

            if len(sub) == 0:
                plt.close(fig)
                print(f"Skipped empty channel: {ch}")
                continue

            ax.plot(
                x_all,
                y_all,
                marker=TRACE_MARKER,
                linewidth=TRACE_LINEWIDTH,
                color=TRACE_COLOR,
            )

            for bx in boundary_positions:
                ax.axvline(bx, linestyle="--", color="black", alpha=0.7, label="_nolegend_")

            y_max_data = np.nanmax(y_all) if len(y_all) else 1
            y_top = Y_LIM_TOP if Y_LIM_TOP is not None else max(1, y_max_data * 1.15)

            for xpos, epoch in label_positions:
                if epoch == "Baseline":
                    continue  # skip labeling Baseline on the plot
                if epoch == "Post":
                    continue  # skip labeling Baseline on the plot

                ax.text(
                    xpos,
                    y_top * 0.94,
                    epoch,
                    ha="center",
                    va="top",
                    fontsize=14,
                    weight="bold",
                    color="black",
                )

            ax.set_title(f"Surface Perfusion 50mM 25/05/26 (2)")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Spike count")
            ax.set_ylim(0, y_top)

            if len(x_all) > 0:
                step = max(1, len(x_all) // 24)
                ax.set_xticks(x_all[::step])
                ax.set_xticklabels(xticklabels[::step], rotation=0)

                ax.set_xlim(-0.5, len(x_all) - 0.5)
                ax.margins(x=0)

                ax.spines["top"].set_visible(True)
                ax.spines["right"].set_visible(True)
                ax.spines["left"].set_linewidth(1.2)
                ax.spines["bottom"].set_linewidth(1.2)


            handles2 = epoch_handles.copy()

            if handles2:
                legend2 = ax.legend(
                    handles=epoch_handles,
                    loc="upper left",
                    bbox_to_anchor=(1.01, 1),
                    borderaxespad=0
                )

            plt.tight_layout(pad=1.0)

            safe_ch = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in ch)
            out_path = os.path.join(out_dir, f"{safe_ch}_continuous_spike_plot.png")
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            print(f"Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
