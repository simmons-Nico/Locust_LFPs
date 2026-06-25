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
- Creates one continuous plot per channel for each data block
- Treats blank rows as gaps that start a new data block
- Prompts for each plot title in the terminal before saving
- Shades epochs
- Marks recording boundaries
- Uses cleaner, publication-style labels
- Saves all plots into OUT_DIR_NAME
"""

import os
import glob
import argparse
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

DEFAULT_PLOT_TITLE = "Single Electrode - Galvanostatic (16/06/26)"
PROMPT_FOR_TITLES = True

TRACE_COLOR = "tab:blue"
TRACE_MARKER = "o"
TRACE_LINEWIDTH = 2
FIG_WIDTH = 16.0
FIG_HEIGHT = 5.5
FIG_DPI = 200
MAX_X_TICKS = 24
PLOT_BIN_SEC = 60.0

# Legend label for the dashed vertical boundary lines.
# Set to "" or None to leave boundary lines out of the legend.
BOUNDARY_DESCRIPTION = "20 min (-200nA)"

Y_LIM_TOP = None
 # set to None for automatic scaling

EPOCH_COLORS = [
    "#929292FF", "#FFE4D6", "#8BD0F8", "#B6FCAF", "#E1C1FD", "#FFEFAA", "#FFBCBC",  "#FFFD9F"
]

PREFERRED_NAME_TOKENS = [
    "Spike_counts_per_min",
    "spike_counts_per_min",
    "spike_counts_per_",
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


def is_blank_value(value) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def find_gap_rows(df: pd.DataFrame, required_cols: list) -> pd.Series:
    """Return True for rows that are empty across the required data columns."""
    return df[required_cols].apply(
        lambda col: col.map(is_blank_value)
    ).all(axis=1)


def format_bin_label(bin_sec: float) -> str:
    if bin_sec < 60:
        return f"{bin_sec:g} sec"
    minutes = bin_sec / 60.0
    return f"{minutes:g} min"


def estimate_source_window_sec(df: pd.DataFrame) -> float:
    """Estimate existing spike-count window size from timing columns, falling back to 60 s."""
    if "window_duration_s" in df.columns:
        durations = pd.to_numeric(df["window_duration_s"], errors="coerce").dropna()
    elif {"window_start_s", "window_end_s"}.issubset(df.columns):
        starts = pd.to_numeric(df["window_start_s"], errors="coerce")
        ends = pd.to_numeric(df["window_end_s"], errors="coerce")
        durations = (ends - starts).dropna()
    else:
        durations = pd.Series(dtype="float64")

    durations = durations[durations > 0]
    if durations.empty:
        return 60.0
    return float(durations.median())


def add_timing_columns(df: pd.DataFrame, source_window_sec: float) -> pd.DataFrame:
    """Ensure rows have start/end timing so plot bins can be aggregated."""
    df = df.copy()
    if {"window_start_s", "window_end_s"}.issubset(df.columns):
        df["window_start_s"] = pd.to_numeric(df["window_start_s"], errors="coerce")
        df["window_end_s"] = pd.to_numeric(df["window_end_s"], errors="coerce")
    else:
        df["window_start_s"] = np.nan
        df["window_end_s"] = np.nan

    missing_time = df["window_start_s"].isna() | df["window_end_s"].isna()
    if missing_time.any():
        inferred_start = df["window_index"].astype(float) * float(source_window_sec)
        df.loc[missing_time, "window_start_s"] = inferred_start.loc[missing_time]
        df.loc[missing_time, "window_end_s"] = inferred_start.loc[missing_time] + float(source_window_sec)

    return df


def add_epoch_segment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Track contiguous epoch/condition blocks so rebinning never crosses them."""
    df = df.copy()
    if "_source_row_order" not in df.columns:
        df["_source_row_order"] = np.arange(len(df))

    df["_epoch_segment_index"] = pd.NA
    df["_segment_start_order"] = pd.NA

    segment_keys = ["epoch_label", "epoch_label_clean"]
    group_keys = ["data_set_index", "channel"]

    for _, unit in df.sort_values("_source_row_order", kind="stable").groupby(
        group_keys,
        sort=False,
        dropna=False,
    ):
        unit = unit.sort_values("_source_row_order", kind="stable")
        segment_start = pd.Series(False, index=unit.index)
        for key in segment_keys:
            segment_start |= unit[key] != unit[key].shift()
        if not segment_start.empty:
            segment_start.iloc[0] = True

        segment_index = segment_start.cumsum().astype(int)
        segment_start_order = unit["_source_row_order"].groupby(segment_index).transform("min")

        df.loc[unit.index, "_epoch_segment_index"] = segment_index.to_numpy()
        df.loc[unit.index, "_segment_start_order"] = segment_start_order.to_numpy()

    df["_epoch_segment_index"] = df["_epoch_segment_index"].astype(int)
    df["_segment_start_order"] = df["_segment_start_order"].astype(int)
    return df


def folded_bin_indices(n_rows: int, rows_per_bin: int) -> np.ndarray:
    """Assign source rows to plot bins, folding a trailing partial bin into the last full bin."""
    if n_rows <= 0:
        return np.array([], dtype=int)
    rows_per_bin = max(1, int(rows_per_bin))
    indices = np.arange(n_rows, dtype=int) // rows_per_bin
    remainder = n_rows % rows_per_bin
    n_full_bins = n_rows // rows_per_bin
    if remainder and n_full_bins > 0:
        indices[-remainder:] = n_full_bins - 1
    return indices


def rebin_spike_counts(df: pd.DataFrame, target_bin_sec: float) -> pd.DataFrame:
    """Sum existing spike-count rows into the requested plotting bin."""
    if target_bin_sec <= 0:
        raise ValueError("Plot bin size must be greater than zero.")

    source_window_sec = estimate_source_window_sec(df)
    tolerance = max(1e-6, source_window_sec * 0.01)
    if target_bin_sec + tolerance < source_window_sec:
        raise ValueError(
            f"Cannot plot {format_bin_label(target_bin_sec)} bins from a CSV whose existing "
            f"window size is about {format_bin_label(source_window_sec)}. Re-run spike counting "
            f"with {format_bin_label(target_bin_sec)} windows first."
        )

    df = add_timing_columns(df, source_window_sec)
    df = add_epoch_segment_columns(df)
    ratio = target_bin_sec / source_window_sec
    if abs(ratio - round(ratio)) > 0.02:
        raise ValueError(
            f"Plot bin size {format_bin_label(target_bin_sec)} must be a multiple of the existing "
            f"window size, about {format_bin_label(source_window_sec)}."
        )
    rows_per_plot_bin = max(1, int(round(ratio)))

    rebinned_rows = []
    group_cols = ["data_set_index", "channel", "_epoch_segment_index", "epoch_label", "epoch_label_clean"]

    for _, block in df.groupby(group_cols, sort=False, dropna=False):
        block = block.sort_values(["window_start_s", "window_index", "_source_row_order"], kind="stable").copy()
        block["_plot_bin_index"] = folded_bin_indices(len(block), rows_per_plot_bin)

        for bin_index, bin_df in block.groupby("_plot_bin_index", sort=True):
            first = bin_df.iloc[0]
            start_s = float(bin_df["window_start_s"].min())
            end_s = float(bin_df["window_end_s"].max())
            rebinned_rows.append(
                {
                    "data_set_index": int(first["data_set_index"]),
                    "recording_index": int(first["recording_index"]),
                    "recording_name": first["recording_name"],
                    "epoch_label": first["epoch_label"],
                    "epoch_label_clean": first["epoch_label_clean"],
                    "channel": first["channel"],
                    "window_index": int(bin_index),
                    "window_start_s": start_s,
                    "window_end_s": end_s,
                    "window_duration_s": end_s - start_s,
                    "spike_count": float(bin_df["spike_count"].sum()),
                    "plot_time_min": start_s / 60.0,
                    "_source_row_order": int(first["_source_row_order"]),
                    "_segment_start_order": int(first["_segment_start_order"]),
                    "_epoch_segment_index": int(first["_epoch_segment_index"]),
                    "_plot_bin_index": int(bin_index),
                }
            )

    out = pd.DataFrame(rebinned_rows)
    if out.empty:
        return df
    return out.reset_index(drop=True)


def prompt_for_plot_title(default_title: str, channel: str, data_set_id: int, show_data_set: bool) -> str:
    if not PROMPT_FOR_TITLES:
        return default_title

    print("\nPlot title")
    if show_data_set:
        print(f"  Data set: {data_set_id}")
    print(f"  Channel: {channel}")
    print(f"  Default: {default_title}")

    try:
        custom_title = input("Enter title, or press Enter to use the default: ").strip()
    except EOFError:
        print("No terminal input available; using default title.")
        return default_title

    return custom_title if custom_title else default_title


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot continuous spike counts per channel from a combined CSV summary.")
    parser.add_argument("input", nargs="?", default="", help="Combined spike-count CSV to plot.")
    parser.add_argument("--csv-dir", default=CSV_DIR, help="Folder to search if no input file is provided.")
    parser.add_argument("--csv-glob", default=CSV_GLOB, help="CSV glob used when searching the folder.")
    parser.add_argument("--out-dir", default="", help="Output folder. Default: <csv-dir>/spike_counts_summary.")
    parser.add_argument("--title", default=DEFAULT_PLOT_TITLE, help="Default title used for every channel plot.")
    parser.add_argument(
        "--no-prompt-titles",
        action="store_true",
        help="Use the default title without prompting. Useful from the GUI.",
    )
    parser.add_argument(
        "--boundary-description",
        default=BOUNDARY_DESCRIPTION or "",
        help="Legend label for dashed recording-boundary lines. Blank hides it.",
    )
    parser.add_argument("--y-lim-top", type=float, default=Y_LIM_TOP, help="Optional fixed y-axis maximum.")
    parser.add_argument("--trace-color", default=TRACE_COLOR, help="Matplotlib color for the spike-count trace.")
    parser.add_argument("--trace-marker", default=TRACE_MARKER, help="Matplotlib marker for data points. Use 'None' for no marker.")
    parser.add_argument("--trace-linewidth", type=float, default=TRACE_LINEWIDTH, help="Line width for the spike-count trace.")
    parser.add_argument("--fig-width", type=float, default=FIG_WIDTH, help="Figure width in inches.")
    parser.add_argument("--fig-height", type=float, default=FIG_HEIGHT, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=FIG_DPI, help="PNG output resolution.")
    parser.add_argument("--max-x-ticks", type=int, default=MAX_X_TICKS, help="Maximum approximate number of x-axis tick labels.")
    parser.add_argument(
        "--plot-bin-sec",
        type=float,
        default=PLOT_BIN_SEC,
        help="Spike-count bin size to plot in seconds, e.g. 30, 60, 120, 180, 240, 300, or 600.",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    global PROMPT_FOR_TITLES
    PROMPT_FOR_TITLES = not args.no_prompt_titles

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 18,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
    })

    csv_path = args.input.strip() if args.input else find_input_csv(args.csv_dir, args.csv_glob)
    base_dir = os.path.dirname(csv_path) if args.input else args.csv_dir
    out_dir = args.out_dir.strip() if args.out_dir else os.path.join(base_dir, OUT_DIR_NAME)
    default_plot_title = args.title
    boundary_description = args.boundary_description.strip()
    y_lim_top = args.y_lim_top
    trace_marker = None if str(args.trace_marker).strip().lower() in {"", "none", "no", "off"} else args.trace_marker
    max_x_ticks = max(1, int(args.max_x_ticks))
    plot_bin_sec = float(args.plot_bin_sec)
    plot_bin_label = format_bin_label(plot_bin_sec)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Using input CSV:\n  {csv_path}")
    print(f"Saving plots to:\n  {out_dir}")
    print(f"Plot bin size: {plot_bin_label}")

    df = pd.read_csv(csv_path, skip_blank_lines=False)

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
    gap_rows = find_gap_rows(df, required_cols)
    section_keys = gap_rows.cumsum()
    df["data_set_index"] = pd.NA

    if (~gap_rows).any():
        df.loc[~gap_rows, "data_set_index"] = (
            pd.factorize(section_keys.loc[~gap_rows])[0] + 1
        )

    df["recording_index"] = pd.to_numeric(df["recording_index"], errors="coerce")
    df["window_index"] = pd.to_numeric(df["window_index"], errors="coerce")
    df["spike_count"] = pd.to_numeric(df["spike_count"], errors="coerce")

    for text_col in ["recording_name", "epoch_label", "channel"]:
        df[text_col] = df[text_col].astype("string").str.strip()
        df[text_col] = df[text_col].replace("", pd.NA)

    df = df.dropna(subset=["recording_index", "window_index", "spike_count", "channel", "data_set_index"])
    if df.empty:
        raise ValueError("Input CSV has no valid rows after cleaning.")

    df["recording_name"] = df["recording_name"].astype(str)
    df["epoch_label"] = df["epoch_label"].astype(str)
    df["channel"] = df["channel"].astype(str)

    df["data_set_index"] = df["data_set_index"].astype(int)
    df["recording_index"] = df["recording_index"].astype(int)
    df["window_index"] = df["window_index"].astype(int)
    df["epoch_label_clean"] = df["epoch_label"].apply(clean_epoch_name)

   # after:
    # df["epoch_label_clean"] = df["epoch_label"].apply(clean_epoch_name)

    df = rebin_spike_counts(df, plot_bin_sec)

    epoch_color_map = build_epoch_color_map(df["epoch_label_clean"])

    data_set_ids = sorted(df["data_set_index"].unique())
    if len(data_set_ids) > 1:
        print(f"Detected {len(data_set_ids)} data sets separated by blank row(s).")

    for data_set_id in data_set_ids:
        data_set_df = df[df["data_set_index"] == data_set_id].copy()
        data_set_suffix = f"_set_{data_set_id:02d}" if len(data_set_ids) > 1 else ""
        title_suffix = f" - Set {data_set_id}" if len(data_set_ids) > 1 else ""

        for ch in sorted(data_set_df["channel"].unique()):
            sub = data_set_df[data_set_df["channel"] == ch].copy()
            sort_cols = [
                col
                for col in ["_segment_start_order", "_plot_bin_index", "window_start_s", "window_index"]
                if col in sub.columns
            ]
            if sort_cols:
                sub = sub.sort_values(sort_cols, kind="stable")
            sub = sub.reset_index(drop=True)
            default_title = f"{default_plot_title}{title_suffix}"
            plot_title = prompt_for_plot_title(
                default_title,
                ch,
                data_set_id,
                show_data_set=len(data_set_ids) > 1,
            )

            fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))

            x_all = np.arange(len(sub))
            y_all = sub["spike_count"].to_numpy()
            xticklabels = [f"{value:g}" for value in sub["plot_time_min"].astype(float).tolist()]

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

            y_max_data = np.nanmax(y_all) if len(y_all) else 1
            y_top = y_lim_top if y_lim_top is not None else max(1, y_max_data * 1.15)

            for _, block in sub.groupby(segment_id, sort=False):
                ax.plot(
                    block.index.to_numpy(),
                    block["spike_count"].to_numpy(),
                    marker=trace_marker,
                    linewidth=args.trace_linewidth,
                    color=args.trace_color,
                )

            for bx in boundary_positions:
                ax.axvline(bx, linestyle="--", color="black", alpha=0.7, label="_nolegend_")

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

            ax.set_title(plot_title)
            ax.set_xlabel("Time (min)")
            ax.set_ylabel(f"Spike count per {plot_bin_label}")
            ax.set_ylim(0, y_top)

            if len(x_all) > 0:
                step = max(1, len(x_all) // max_x_ticks)
                ax.set_xticks(x_all[::step])
                ax.set_xticklabels(xticklabels[::step], rotation=0)

                ax.set_xlim(-0.5, len(x_all) - 0.5)
                ax.margins(x=0)

                ax.spines["top"].set_visible(True)
                ax.spines["right"].set_visible(True)
                ax.spines["left"].set_linewidth(1.2)
                ax.spines["bottom"].set_linewidth(1.2)


            handles2 = [
                h for h in epoch_handles
                if h.get_label() != "Post"
            ] + [
                h for h in epoch_handles
                if h.get_label() == "Post"
            ]

            if boundary_positions and boundary_description:
                handles2.append(
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        linestyle="--",
                        alpha=0.7,
                        label=boundary_description,
                    )
                )

            if handles2:
                legend2 = ax.legend(
                    handles=handles2,
                    loc="upper left",
                    bbox_to_anchor=(1.01, 1),
                    borderaxespad=0
                )

            plt.tight_layout(pad=1.0)

            safe_ch = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in ch)
            bin_name = format_bin_label(plot_bin_sec).replace(" ", "_").replace(".", "p")
            out_path = os.path.join(out_dir, f"{safe_ch}{data_set_suffix}_{bin_name}_continuous_spike_plot.png")
            plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)

            print(f"Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
