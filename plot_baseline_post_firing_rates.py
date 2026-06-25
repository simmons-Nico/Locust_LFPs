#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Publication-style Baseline / stimulation / Post firing-rate analysis.

Expected input columns:
    recording_name, epoch_label, channel, window_index,
    window_start_s, window_end_s, window_label, spike_count

What this script does:
1. Loads a .csv, .xlsx, or .xls file.
2. Computes firing rate as spike_count / window duration in Hz.
3. Detects repeated stimulation + Post experiment parts after the initial Baseline.
4. Collapses windows to one mean firing rate per recording/channel/phase.
5. Splits Baseline into first/second halves for percent-change controls.
6. Makes spike-frequency and percent-change plots for each experiment part.
7. Runs paired tests vs Baseline and optional unpaired group tests.
8. Saves PNG/SVG figures and CSV summaries.

Run examples:
    python plot_baseline_post_firing_rates.py
    python plot_baseline_post_firing_rates.py --input-dir "C:\\path\\to\\folder"
    python plot_baseline_post_firing_rates.py "C:\\path\\to\\processed_spike_counts.csv"
    python plot_baseline_post_firing_rates.py "C:\\path\\to\\processed_spike_counts.xlsx"
    python plot_baseline_post_firing_rates.py "C:\\path\\to\\processed_spike_counts.csv" --group-col treatment
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import re
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# =========================
# EDITABLE USER SETTINGS
# =========================

# Leave INPUT_PATH blank to choose a file from INPUT_DIR automatically.
INPUT_PATH = ""
INPUT_DIR = r"C:\Users\simmons\Desktop\Exploring PSDs"
INPUT_PATTERNS = ("*.csv", "*.xlsx", "*.xls")
AUTO_SELECT_NEWEST_INPUT = True

OUT_DIR_NAME = ""  # blank means: <input file stem>_baseline_post_plots
EXCEL_SHEET = 0
CSV_SEPARATOR = None  # None lets pandas auto-detect comma, semicolon, tab, etc.

# The default assumes each input file is one experiment and pairs by channel.
# For combined multi-experiment files, use e.g. --id-cols experiment_id,channel.
ID_COLUMNS = ("channel",)

# Optional grouping column for unpaired percent-change tests, e.g. "treatment",
# "substrate", "drug", "genotype", or "group". Leave blank/None to skip.
GROUP_COLUMN = None

BASELINE_LABEL = "Baseline"
STIMULATION_LABEL = "Stimulation"
POST_LABEL = "Post"
POST_LABELS = ("Post 1", "Post 2")  # kept for older helper functions
BASELINE_FIRST_HALF_LABEL = "Baseline first half"
BASELINE_SECOND_HALF_LABEL = "Baseline second half"

COMPARISON_BASELINE = "baseline"
COMPARISON_STIMULATION = "stimulation"
COMPARISON_POST = "post"
DEFAULT_RECORDED_PHASES = (COMPARISON_BASELINE, COMPARISON_STIMULATION, COMPARISON_POST)
DEFAULT_COMPARISONS = (COMPARISON_BASELINE, COMPARISON_STIMULATION, COMPARISON_POST)

MAKE_TIME_COURSE = True
MAX_TIME_COURSE_UNITS = 80
POOL_PERCENT_CHANGE_COMPARISONS = False

PAIRED_TEST = "paired t-test"
UNPAIRED_TEST = "Welch t-test"
ERROR_BAR_TYPE = "sem"  # "sem" or "sd"; window SEM is used when there is only one unit.

FIG_DPI = 300


# =========================
# GENERAL HELPERS
# =========================

CANONICAL_COLUMNS = {
    "recording_name": ("recording_name", "recording", "recording_id", "file_name", "filename"),
    "epoch_label": ("epoch_label", "epoch", "condition", "condition_label"),
    "channel": ("channel", "chan", "electrode", "electrode_id"),
    "window_index": ("window_index", "window", "bin", "bin_index"),
    "window_start_s": ("window_start_s", "start_s", "window_start", "start_time_s"),
    "window_end_s": ("window_end_s", "end_s", "window_end", "end_time_s"),
    "window_label": ("window_label", "window_name", "bin_label"),
    "spike_count": ("spike_count", "spikes", "spike_counts", "count", "n_spikes"),
    "window_duration_s": ("window_duration_s", "duration_s", "duration", "window_length_s"),
}


def normalise_name(value: str) -> str:
    """Lowercase a column/label and remove punctuation used only as separators."""
    value = str(value).strip().lower()
    value = re.sub(r"[\s\-./()]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def safe_name(value: object, max_len: int = 80) -> str:
    """Make a readable filename component."""
    text = str(value).strip()
    text = re.sub(r"[^\w\-.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "unknown")[:max_len]


def split_option_list(value: object, default: tuple[str, ...]) -> list[str]:
    """Parse comma/semicolon-separated command-line or GUI option values."""
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            items.extend(split_option_list(item, ()))
        return items
    text = str(value).strip()
    if not text:
        return list(default)
    return [item.strip() for item in re.split(r"[,;]+", text) if item.strip()]


def parse_title_mapping(value: object = None) -> dict[str, str]:
    """Parse per-plot title mappings from JSON or key=value text."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key).strip(): str(title).strip() for key, title in value.items() if str(title).strip()}

    text = str(value).strip()
    if not text:
        return {}

    with contextlib.suppress(json.JSONDecodeError, TypeError, ValueError):
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return {
                str(key).strip(): str(title).strip()
                for key, title in parsed.items()
                if str(key).strip() and str(title).strip()
            }

    titles: dict[str, str] = {}
    for item in re.split(r";+", text):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("Percent plot titles must be JSON or semicolon-separated key=title entries.")
        key, title = item.split("=", 1)
        key = key.strip()
        title = title.strip()
        if key and title:
            titles[key] = title
    return titles


def parse_recorded_phases(value: object = None) -> tuple[str, ...]:
    aliases = {
        "baseline": BASELINE_LABEL,
        "base": BASELINE_LABEL,
        "pre": BASELINE_LABEL,
        "control": BASELINE_LABEL,
        "stimulation": STIMULATION_LABEL,
        "stim": STIMULATION_LABEL,
        "current": STIMULATION_LABEL,
        "post": POST_LABEL,
        "after": POST_LABEL,
        "recovery": POST_LABEL,
    }
    phases: list[str] = []
    for item in split_option_list(value, DEFAULT_RECORDED_PHASES):
        key = normalise_name(item)
        if key == "all":
            phases.extend([BASELINE_LABEL, STIMULATION_LABEL, POST_LABEL])
            continue
        if key not in aliases:
            raise ValueError(
                f"Unknown recorded phase '{item}'. Use baseline, stimulation, post, or all."
            )
        phases.append(aliases[key])
    return tuple(dict.fromkeys(phases))


def parse_comparisons(value: object = None) -> tuple[str, ...]:
    aliases = {
        "baseline": COMPARISON_BASELINE,
        "baseline_halves": COMPARISON_BASELINE,
        "baseline_split": COMPARISON_BASELINE,
        "control": COMPARISON_BASELINE,
        "stimulation": COMPARISON_STIMULATION,
        "stim": COMPARISON_STIMULATION,
        "current": COMPARISON_STIMULATION,
        "post": COMPARISON_POST,
        "after": COMPARISON_POST,
        "recovery": COMPARISON_POST,
    }
    comparisons: list[str] = []
    for item in split_option_list(value, DEFAULT_COMPARISONS):
        key = normalise_name(item)
        if key == "all":
            comparisons.extend(DEFAULT_COMPARISONS)
            continue
        if key not in aliases:
            raise ValueError(
                f"Unknown comparison '{item}'. Use baseline, stimulation, post, or all."
            )
        comparisons.append(aliases[key])
    return tuple(dict.fromkeys(comparisons))


def filter_comparisons_for_phases(
    comparisons: tuple[str, ...],
    recorded_phases: tuple[str, ...],
) -> tuple[str, ...]:
    allowed: set[str] = set()
    if BASELINE_LABEL in recorded_phases:
        allowed.add(COMPARISON_BASELINE)
    if STIMULATION_LABEL in recorded_phases:
        allowed.add(COMPARISON_STIMULATION)
    if POST_LABEL in recorded_phases:
        allowed.add(COMPARISON_POST)
    return tuple(comparison for comparison in comparisons if comparison in allowed)


def phase_specs_for_comparisons(comparisons: tuple[str, ...] | None = None) -> list[dict[str, str]]:
    comparison_set = set(comparisons or DEFAULT_COMPARISONS)
    specs: list[dict[str, str]] = []
    if COMPARISON_BASELINE in comparison_set:
        specs.append(
            {
                "phase": BASELINE_LABEL,
                "hz_col": "baseline_hz",
                "percent_col": "baseline_percent_change",
                "test_a": "baseline_first_half_hz",
                "test_b": "baseline_second_half_hz",
                "comparison": "Baseline first half vs Baseline second half",
            }
        )
    if COMPARISON_STIMULATION in comparison_set:
        specs.append(
            {
                "phase": STIMULATION_LABEL,
                "hz_col": "stimulation_hz",
                "percent_col": "stimulation_percent_change",
                "test_a": "baseline_hz",
                "test_b": "stimulation_hz",
                "comparison": "Baseline vs stimulation",
            }
        )
    if COMPARISON_POST in comparison_set:
        specs.append(
            {
                "phase": POST_LABEL,
                "hz_col": "post_hz",
                "percent_col": "post_percent_change",
                "test_a": "baseline_hz",
                "test_b": "post_hz",
                "comparison": "Baseline vs Post",
            }
        )
    return specs


def sem(values: Iterable[float]) -> float:
    arr = pd.Series(values, dtype="float64").dropna().to_numpy()
    if arr.size <= 1:
        return float("nan")
    return float(np.nanstd(arr, ddof=1) / math.sqrt(arr.size))


def sd(values: Iterable[float]) -> float:
    arr = pd.Series(values, dtype="float64").dropna().to_numpy()
    if arr.size <= 1:
        return float("nan")
    return float(np.nanstd(arr, ddof=1))


def error_bar(values: Iterable[float]) -> float:
    if str(ERROR_BAR_TYPE).lower() == "sd":
        return sd(values)
    return sem(values)


def p_to_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return "n/a"
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": FIG_DPI,
            "font.family": "Arial",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.linewidth": 1.1,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def use_zero_x_axis(ax: plt.Axes) -> None:
    """Draw a y=0 reference line while keeping category labels at the plot bottom."""
    ax.axhline(0, color="black", lw=0.9, zorder=1)
    ax.spines["bottom"].set_position(("outward", 0))
    ax.spines["bottom"].set_linewidth(1.0)
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="x", direction="out", length=4, width=1.0, pad=6)


def percent_axis_limits(y_min: float, y_max: float, top_pad: float = 0.18) -> tuple[float, float]:
    """Avoid negative percent-axis space unless values or error bars go below zero."""
    y_min = float(y_min) if np.isfinite(y_min) else 0.0
    y_max = float(y_max) if np.isfinite(y_max) else 1.0
    y_span = max(y_max - min(y_min, 0.0), 1.0)
    lower = 0.0 if y_min >= 0 else y_min - 0.12 * y_span
    upper = y_max + top_pad * y_span
    return lower, upper


def add_sig_label(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    text: str,
    line_height: float,
    fontsize: int = 12,
) -> None:
    """Draw a compact significance bracket."""
    if text == "n/a":
        return
    ax.plot([x1, x1, x2, x2], [y, y + line_height, y + line_height, y], color="black", lw=1.0)
    ax.text((x1 + x2) / 2, y + line_height, text, ha="center", va="bottom", fontsize=fontsize, weight="bold")


def save_figure(fig: plt.Figure, out_dir: Path, base_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{base_name}.png"
    svg_path = out_dir / f"{base_name}.svg"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def prepare_output_dir(out_dir: Path) -> None:
    """Remove this script's prior generated outputs so stale plots do not linger."""
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_dirs = [
        "paired_baseline_post",
        "percent_change",
        "percent_change_by_part",
        "spike_frequency_by_part",
        "time_course",
    ]
    for dirname in generated_dirs:
        path = out_dir / dirname
        if path.exists():
            shutil.rmtree(path)

    generated_files = [
        "baseline_post_summary.csv",
        "baseline_split_percent_changes.csv",
        "cleaned_window_level_firing_rates.csv",
        "experiment_part_summary.csv",
        "experiment_part_unit_values.csv",
        "unit_condition_mean_firing_rates.csv",
        "unit_percent_changes.csv",
        "unpaired_percent_change_tests.csv",
    ]
    for filename in generated_files:
        path = out_dir / filename
        if path.exists():
            path.unlink()


def should_ignore_candidate(path: Path) -> bool:
    """Skip temporary files and CSV summaries generated by this script."""
    if path.name.startswith("~$"):
        return True

    generated_tokens = (
        "baseline_post_summary",
        "baseline_split_percent_changes",
        "cleaned_window_level_firing_rates",
        "unit_condition_mean_firing_rates",
        "unit_percent_changes",
        "unpaired_percent_change_tests",
    )
    stem = path.stem.lower()
    return any(token in stem for token in generated_tokens)


def find_input_file(
    input_path: str | None,
    input_dir: str,
    patterns: tuple[str, ...],
    auto_select_newest: bool = True,
) -> Path:
    if input_path:
        path = Path(input_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        if path.is_dir():
            return find_input_file(None, str(path), patterns, auto_select_newest=auto_select_newest)
        return path

    base_dir = Path(input_dir).expanduser()
    if not base_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {base_dir}")

    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(base_dir.glob(pattern)))

    candidates = [p for p in candidates if p.is_file() and not should_ignore_candidate(p)]
    if not candidates:
        raise FileNotFoundError(
            f"No input files found in {base_dir.resolve()} matching: {', '.join(patterns)}"
        )

    if len(candidates) == 1:
        return candidates[0]

    if auto_select_newest:
        newest = max(candidates, key=lambda p: (p.stat().st_mtime, p.name.lower()))
        print(
            f"Multiple input files found in {base_dir.resolve()}; "
            f"using newest supported file: {newest.name}"
        )
        return newest

    preferred_tokens = ("processed", "spike", "spike_counts", "summary", "firing")
    preferred = [
        p for p in candidates if any(token in p.name.lower() for token in preferred_tokens)
    ]
    if len(preferred) == 1:
        return preferred[0]

    file_list = "\n".join(f"  - {p.name}" for p in candidates)
    raise FileNotFoundError(
        "Could not choose one input file automatically. Pass the file path explicitly.\n"
        f"Found:\n{file_list}"
    )


def load_csv(path: Path, sep: str | None = None) -> pd.DataFrame:
    """Read CSV files robustly, including UTF-8 BOMs and non-comma delimiters."""
    read_kwargs = {
        "sep": sep,
        "engine": "python" if sep is None else "c",
        "skip_blank_lines": True,
    }

    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding, **read_kwargs)
        except UnicodeDecodeError:
            continue

    return pd.read_csv(path, encoding="latin1", **read_kwargs)


def load_table(path: Path, sheet: str | int = 0, csv_sep: str | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv(path, sep=csv_sep)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet)
    raise ValueError(f"Unsupported input file type: {path.suffix}. Use .csv, .xlsx, or .xls.")


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common column spellings to the canonical names used below."""
    norm_to_original = {normalise_name(c): c for c in df.columns}
    rename_map = {}

    for canonical, aliases in CANONICAL_COLUMNS.items():
        for alias in aliases:
            normalised_alias = normalise_name(alias)
            if normalised_alias in norm_to_original:
                rename_map[norm_to_original[normalised_alias]] = canonical
                break

    return df.rename(columns=rename_map)


def classify_epoch_label(epoch_label: object) -> str | None:
    """Return Baseline, Stimulation, or Post for flexible epoch labels."""
    text = str(epoch_label).strip()
    if not text or text.lower() == "nan":
        return None

    compact = normalise_name(text)
    no_sep = compact.replace("_", "")

    baseline_terms = {"baseline", "basal", "before", "pre", "pretreatment", "control"}
    if any(term in no_sep for term in baseline_terms):
        return BASELINE_LABEL

    post_terms = {"post", "after", "recovery", "washout"}
    if any(term in no_sep for term in post_terms):
        return POST_LABEL

    return STIMULATION_LABEL


def clean_stimulation_label(epoch_label: object) -> str:
    """Clean stimulation labels without hiding the current intensity."""
    text = str(epoch_label).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?i)([-+]?\d+(?:\.\d+)?)\s*([munp]?a)\b", r"\1 \2", text)
    return text or STIMULATION_LABEL


def clean_post_label(epoch_label: object) -> str:
    """Keep Post labels such as Post 1/Post 2 readable for post-only experiments."""
    text = str(epoch_label).strip()
    text = re.sub(r"\s+", " ", text)
    return text or POST_LABEL


def validate_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = standardise_columns(df.copy())

    required = ["recording_name", "epoch_label", "channel", "spike_count"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    has_start_end = {"window_start_s", "window_end_s"}.issubset(df.columns)
    has_duration = "window_duration_s" in df.columns
    if not has_start_end and not has_duration:
        raise ValueError(
            "Need either window_start_s + window_end_s, or window_duration_s, "
            "to compute firing rate."
        )

    for col in ["spike_count", "window_start_s", "window_end_s", "window_duration_s", "window_index"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["recording_name", "epoch_label", "channel"]:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)

    if has_start_end:
        df["window_duration_s"] = df["window_end_s"] - df["window_start_s"]

    valid = (
        df["recording_name"].notna()
        & df["epoch_label"].notna()
        & df["channel"].notna()
        & df["spike_count"].notna()
        & df["window_duration_s"].notna()
        & (df["window_duration_s"] > 0)
    )
    dropped = int((~valid).sum())
    if dropped:
        print(f"Dropped {dropped} row(s) with missing/invalid identifiers, spike counts, or durations.")

    df = df.loc[valid].copy()
    if df.empty:
        raise ValueError("No valid rows remain after cleaning.")

    df["recording_name"] = df["recording_name"].astype(str)
    df["epoch_label"] = df["epoch_label"].astype(str)
    df["channel"] = df["channel"].astype(str)
    df["_row_order"] = np.arange(len(df))
    df["firing_rate_hz"] = df["spike_count"] / df["window_duration_s"]
    df["phase"] = df["epoch_label"].map(classify_epoch_label)
    df["condition"] = df["phase"]
    df["stimulation_label_raw"] = np.where(
        df["phase"] == STIMULATION_LABEL,
        df["epoch_label"].map(clean_stimulation_label),
        pd.NA,
    )
    df["post_label_raw"] = np.where(
        df["phase"] == POST_LABEL,
        df["epoch_label"].map(clean_post_label),
        pd.NA,
    )

    ignored = df["phase"].isna()
    if ignored.any():
        ignored_labels = sorted(df.loc[ignored, "epoch_label"].dropna().unique())
        print("Ignoring blank/unrecognized epoch labels:")
        for label in ignored_labels:
            print(f"  - {label}")

    df = df.loc[df["phase"].isin([BASELINE_LABEL, STIMULATION_LABEL, POST_LABEL])].copy()
    if df.empty:
        raise ValueError("No rows matched Baseline, stimulation, or Post after epoch-label classification.")

    if "window_index" not in df.columns:
        df["window_index"] = df.groupby(["recording_name", "channel", "phase"]).cumcount() + 1

    return df


def choose_group_column(df: pd.DataFrame, requested: str | None) -> str | None:
    normalised_columns = {normalise_name(col): col for col in df.columns}

    if requested:
        if requested in df.columns:
            return requested
        requested_norm = normalise_name(requested)
        if requested_norm in normalised_columns:
            return normalised_columns[requested_norm]
        else:
            raise ValueError(f"Requested group column '{requested}' was not found.")

    if GROUP_COLUMN and GROUP_COLUMN in df.columns:
        return GROUP_COLUMN
    if GROUP_COLUMN and normalise_name(GROUP_COLUMN) in normalised_columns:
        return normalised_columns[normalise_name(GROUP_COLUMN)]

    for candidate in ("group", "treatment", "substrate", "condition_group", "genotype", "drug"):
        candidate_norm = normalise_name(candidate)
        if candidate_norm in normalised_columns:
            return normalised_columns[candidate_norm]

    return None


def collapse_to_unit_means(
    df: pd.DataFrame,
    id_columns: tuple[str, ...],
    group_column: str | None,
) -> pd.DataFrame:
    missing_id_cols = [col for col in id_columns if col not in df.columns]
    if missing_id_cols:
        raise ValueError(f"ID column(s) not found in data: {missing_id_cols}")

    group_cols = list(id_columns)
    if group_column:
        group_cols.append(group_column)
        df[group_column] = df[group_column].astype("string").fillna("Unlabeled").astype(str)

    phase_cols = ["phase"]
    for col in ("part_index", "stimulation_label", "part_label"):
        if col in df.columns:
            phase_cols.append(col)

    unit_means = (
        df.groupby([*group_cols, *phase_cols], dropna=False)
        .agg(
            mean_firing_rate_hz=("firing_rate_hz", "mean"),
            sem_firing_rate_hz=("firing_rate_hz", sem),
            n_windows=("firing_rate_hz", "count"),
            mean_spike_count=("spike_count", "mean"),
            mean_window_duration_s=("window_duration_s", "mean"),
        )
        .reset_index()
    )

    return unit_means


def baseline_sort_columns(df: pd.DataFrame) -> list[str]:
    sort_cols = []
    for col in ("window_start_s", "window_end_s", "window_index"):
        if col in df.columns:
            sort_cols.append(col)
    return sort_cols


def split_baseline_rows(sub: pd.DataFrame) -> pd.Series:
    """Label baseline rows as first/second half within one paired unit."""
    sort_cols = baseline_sort_columns(sub)
    sub = sub.sort_values(sort_cols, kind="stable") if sort_cols else sub.copy()
    labels = pd.Series(pd.NA, index=sub.index, dtype="object")

    if len(sub) < 2:
        return labels

    starts = sub["window_start_s"]
    ends = sub["window_end_s"]
    centers = (starts + ends) / 2.0

    if starts.notna().all() and ends.notna().all() and centers.notna().all():
        midpoint = float(starts.min() + ((ends.max() - starts.min()) / 2.0))
        first_mask = centers < midpoint
        second_mask = ~first_mask

        if first_mask.any() and second_mask.any():
            labels.loc[sub.index[first_mask]] = BASELINE_FIRST_HALF_LABEL
            labels.loc[sub.index[second_mask]] = BASELINE_SECOND_HALF_LABEL
            return labels

    split_at = len(sub) // 2
    if split_at == 0 or split_at == len(sub):
        return labels

    labels.iloc[:split_at] = BASELINE_FIRST_HALF_LABEL
    labels.iloc[split_at:] = BASELINE_SECOND_HALF_LABEL
    return labels


def add_baseline_half_labels(
    df: pd.DataFrame,
    id_columns: tuple[str, ...],
    group_column: str | None,
) -> pd.DataFrame:
    """Add a baseline_half column for first/second half Baseline windows."""
    df = df.copy()
    df["baseline_half"] = pd.NA

    group_cols = list(id_columns)
    if group_column:
        group_cols.append(group_column)

    baseline = df[df["condition"] == BASELINE_LABEL]
    if baseline.empty:
        return df

    for _, sub in baseline.groupby(group_cols, dropna=False, sort=False):
        labels = split_baseline_rows(sub)
        df.loc[labels.index, "baseline_half"] = labels

    return df


def build_baseline_half_table(
    df: pd.DataFrame,
    id_columns: tuple[str, ...],
    group_column: str | None,
) -> pd.DataFrame:
    """Build unit-level first-half vs second-half Baseline percent changes."""
    group_cols = list(id_columns)
    if group_column:
        group_cols.append(group_column)

    baseline = df[
        (df["condition"] == BASELINE_LABEL)
        & df["baseline_half"].isin([BASELINE_FIRST_HALF_LABEL, BASELINE_SECOND_HALF_LABEL])
    ].copy()
    if baseline.empty:
        return pd.DataFrame(columns=group_cols)

    means = (
        baseline.groupby([*group_cols, "baseline_half"], dropna=False)
        .agg(
            mean_firing_rate_hz=("firing_rate_hz", "mean"),
            n_windows=("firing_rate_hz", "count"),
        )
        .reset_index()
    )

    mean_wide = means.pivot_table(
        index=group_cols,
        columns="baseline_half",
        values="mean_firing_rate_hz",
        aggfunc="first",
    ).reset_index()
    count_wide = means.pivot_table(
        index=group_cols,
        columns="baseline_half",
        values="n_windows",
        aggfunc="first",
    ).reset_index()

    rename_mean = {
        BASELINE_FIRST_HALF_LABEL: "baseline_first_half_hz",
        BASELINE_SECOND_HALF_LABEL: "baseline_second_half_hz",
    }
    rename_count = {
        BASELINE_FIRST_HALF_LABEL: "baseline_first_half_n_windows",
        BASELINE_SECOND_HALF_LABEL: "baseline_second_half_n_windows",
    }
    mean_wide = mean_wide.rename(columns=rename_mean)
    count_wide = count_wide.rename(columns=rename_count)

    out = mean_wide.merge(count_wide, on=group_cols, how="left")
    for col in [
        "baseline_first_half_hz",
        "baseline_second_half_hz",
        "baseline_first_half_n_windows",
        "baseline_second_half_n_windows",
    ]:
        if col not in out.columns:
            out[col] = np.nan

    out["post_condition"] = BASELINE_LABEL
    out["baseline_hz"] = out["baseline_first_half_hz"]
    out["post_hz"] = out["baseline_second_half_hz"]
    out["delta_hz"] = out["post_hz"] - out["baseline_hz"]
    out["percent_reference_hz"] = out["baseline_first_half_hz"]
    out["percent_reference_label"] = BASELINE_FIRST_HALF_LABEL
    out["percent_change"] = np.where(
        out["percent_reference_hz"] != 0,
        ((out["baseline_second_half_hz"] - out["baseline_first_half_hz"]) / out["percent_reference_hz"]) * 100.0,
        np.nan,
    )

    return out


def build_paired_tables(
    unit_means: pd.DataFrame,
    baseline_halves: pd.DataFrame,
    id_columns: tuple[str, ...],
    group_column: str | None,
) -> dict[str, pd.DataFrame]:
    index_cols = list(id_columns)
    if group_column:
        index_cols.append(group_column)

    paired_tables: dict[str, pd.DataFrame] = {}
    wide = unit_means.pivot_table(
        index=index_cols,
        columns="condition",
        values="mean_firing_rate_hz",
        aggfunc="first",
    ).reset_index()

    baseline_half_cols = [
        "baseline_first_half_hz",
        "baseline_second_half_hz",
        "baseline_first_half_n_windows",
        "baseline_second_half_n_windows",
    ]
    available_half_cols = [col for col in baseline_half_cols if col in baseline_halves.columns]
    if available_half_cols:
        wide = wide.merge(
            baseline_halves[[*index_cols, *available_half_cols]],
            on=index_cols,
            how="left",
        )

    for post_label in POST_LABELS:
        if BASELINE_LABEL not in wide.columns or post_label not in wide.columns:
            paired_tables[post_label] = pd.DataFrame()
            continue

        paired = wide.dropna(subset=[BASELINE_LABEL, post_label]).copy()
        paired = paired.rename(columns={BASELINE_LABEL: "baseline_hz", post_label: "post_hz"})
        other_post_cols = [label for label in POST_LABELS if label != post_label and label in paired.columns]
        if other_post_cols:
            paired = paired.drop(columns=other_post_cols)
        paired["post_condition"] = post_label
        paired["delta_hz"] = paired["post_hz"] - paired["baseline_hz"]
        paired["percent_reference_hz"] = paired.get("baseline_second_half_hz", np.nan)
        paired["percent_reference_label"] = BASELINE_SECOND_HALF_LABEL
        paired["percent_change"] = np.where(
            paired["percent_reference_hz"] != 0,
            ((paired["post_hz"] - paired["percent_reference_hz"]) / paired["percent_reference_hz"]) * 100.0,
            np.nan,
        )
        paired_tables[post_label] = paired

    return paired_tables


def experiment_sort_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in ("recording_index", "window_start_s", "window_end_s", "window_index", "_row_order"):
        if col in df.columns:
            cols.append(col)
    return cols


def assign_experiment_parts(
    df: pd.DataFrame,
    id_columns: tuple[str, ...],
    group_column: str | None,
    allow_post_without_stimulation: bool = False,
) -> pd.DataFrame:
    """Assign each stimulation block and following Post block to an experiment part."""
    df = df.copy()
    df["part_index"] = pd.NA
    df["stimulation_label"] = pd.NA
    df["part_label"] = pd.NA

    group_cols = list(id_columns)
    if group_column:
        group_cols.append(group_column)

    sort_cols = experiment_sort_columns(df)

    for _, unit in df.groupby(group_cols, dropna=False, sort=False):
        unit = unit.sort_values(sort_cols, kind="stable") if sort_cols else unit.copy()
        block_key = np.where(
            unit["phase"] == STIMULATION_LABEL,
            unit["stimulation_label_raw"].fillna(STIMULATION_LABEL).astype(str),
            np.where(
                unit["phase"] == POST_LABEL,
                unit["post_label_raw"].fillna(POST_LABEL).astype(str),
                unit["phase"].astype(str),
            ),
        )
        block_ids = pd.Series(block_key, index=unit.index).ne(pd.Series(block_key, index=unit.index).shift()).cumsum()

        current_part = 0
        current_stim_label: str | None = None
        current_part_has_stimulation = False

        for _, block in unit.groupby(block_ids, sort=False):
            phase = block["phase"].iloc[0]

            if phase == BASELINE_LABEL:
                continue

            if phase == STIMULATION_LABEL:
                current_part += 1
                current_stim_label = str(block["stimulation_label_raw"].dropna().iloc[0])
                part_label = f"Part {current_part}: {current_stim_label}"
                df.loc[block.index, "part_index"] = current_part
                df.loc[block.index, "stimulation_label"] = current_stim_label
                df.loc[block.index, "part_label"] = part_label
                current_part_has_stimulation = True
                continue

            if phase == POST_LABEL and current_part > 0 and current_part_has_stimulation:
                stim_label = current_stim_label or STIMULATION_LABEL
                part_label = f"Part {current_part}: {stim_label}"
                df.loc[block.index, "part_index"] = current_part
                df.loc[block.index, "stimulation_label"] = stim_label
                df.loc[block.index, "part_label"] = part_label
                continue

            if phase == POST_LABEL and allow_post_without_stimulation:
                current_part += 1
                post_label = str(block["post_label_raw"].dropna().iloc[0]) if block["post_label_raw"].notna().any() else f"Post {current_part}"
                part_label = f"Part {current_part}: {post_label}"
                df.loc[block.index, "part_index"] = current_part
                df.loc[block.index, "stimulation_label"] = post_label
                df.loc[block.index, "part_label"] = part_label
                current_stim_label = None
                current_part_has_stimulation = False

    return df


def build_experiment_part_table(
    df: pd.DataFrame,
    baseline_change: pd.DataFrame,
    id_columns: tuple[str, ...],
    group_column: str | None,
) -> pd.DataFrame:
    """Return one row per paired unit and experiment part."""
    group_cols = list(id_columns)
    if group_column:
        group_cols.append(group_column)

    baseline_means = (
        df[df["phase"] == BASELINE_LABEL]
        .groupby(group_cols, dropna=False)
        .agg(
            baseline_hz=("firing_rate_hz", "mean"),
            baseline_window_error_hz=("firing_rate_hz", error_bar),
            baseline_n_windows=("firing_rate_hz", "count"),
        )
        .reset_index()
    )

    part_rows = df[df["part_index"].notna() & df["phase"].isin([STIMULATION_LABEL, POST_LABEL])].copy()
    if part_rows.empty:
        return pd.DataFrame()

    part_rows["part_index"] = part_rows["part_index"].astype(int)

    phase_means = (
        part_rows.groupby([*group_cols, "part_index", "stimulation_label", "part_label", "phase"], dropna=False)
        .agg(
            mean_firing_rate_hz=("firing_rate_hz", "mean"),
            error_firing_rate_hz=("firing_rate_hz", error_bar),
            n_windows=("firing_rate_hz", "count"),
        )
        .reset_index()
    )

    index_cols = [*group_cols, "part_index", "stimulation_label", "part_label"]
    mean_wide = phase_means.pivot_table(
        index=index_cols,
        columns="phase",
        values="mean_firing_rate_hz",
        aggfunc="first",
    ).reset_index()
    count_wide = phase_means.pivot_table(
        index=index_cols,
        columns="phase",
        values="n_windows",
        aggfunc="first",
    ).reset_index()
    error_wide = phase_means.pivot_table(
        index=index_cols,
        columns="phase",
        values="error_firing_rate_hz",
        aggfunc="first",
    ).reset_index()

    mean_wide = mean_wide.rename(
        columns={
            STIMULATION_LABEL: "stimulation_hz",
            POST_LABEL: "post_hz",
        }
    )
    count_wide = count_wide.rename(
        columns={
            STIMULATION_LABEL: "stimulation_n_windows",
            POST_LABEL: "post_n_windows",
        }
    )
    error_wide = error_wide.rename(
        columns={
            STIMULATION_LABEL: "stimulation_window_error_hz",
            POST_LABEL: "post_window_error_hz",
        }
    )

    out = mean_wide.merge(count_wide, on=index_cols, how="left")
    out = out.merge(error_wide, on=index_cols, how="left")
    out = out.merge(baseline_means, on=group_cols, how="left")

    baseline_cols = [
        *group_cols,
        "baseline_first_half_hz",
        "baseline_second_half_hz",
        "baseline_first_half_n_windows",
        "baseline_second_half_n_windows",
        "percent_change",
    ]
    available_baseline_cols = [col for col in baseline_cols if col in baseline_change.columns]
    if available_baseline_cols:
        baseline_subset = baseline_change[available_baseline_cols].copy()
        baseline_subset = baseline_subset.rename(columns={"percent_change": "baseline_percent_change"})
        out = out.merge(baseline_subset, on=group_cols, how="left")

    for col in [
        "stimulation_hz",
        "post_hz",
        "stimulation_n_windows",
        "post_n_windows",
        "baseline_window_error_hz",
        "stimulation_window_error_hz",
        "post_window_error_hz",
        "baseline_first_half_hz",
        "baseline_second_half_hz",
        "baseline_first_half_n_windows",
        "baseline_second_half_n_windows",
        "baseline_percent_change",
    ]:
        if col not in out.columns:
            out[col] = np.nan

    out["percent_reference_hz"] = out["baseline_second_half_hz"].combine_first(out["baseline_hz"])
    out["percent_reference_label"] = np.where(
        out["baseline_second_half_hz"].notna(),
        BASELINE_SECOND_HALF_LABEL,
        BASELINE_LABEL,
    )
    out["stimulation_delta_hz"] = out["stimulation_hz"] - out["baseline_hz"]
    out["post_delta_hz"] = out["post_hz"] - out["baseline_hz"]
    out["stimulation_percent_change"] = np.where(
        out["percent_reference_hz"] != 0,
        ((out["stimulation_hz"] - out["percent_reference_hz"]) / out["percent_reference_hz"]) * 100.0,
        np.nan,
    )
    out["post_percent_change"] = np.where(
        out["percent_reference_hz"] != 0,
        ((out["post_hz"] - out["percent_reference_hz"]) / out["percent_reference_hz"]) * 100.0,
        np.nan,
    )

    for col in [
        "baseline_percent_change_error",
        "stimulation_percent_change_error",
        "post_percent_change_error",
    ]:
        out[col] = np.nan

    for row_index, row in out.iterrows():
        unit_mask = pd.Series(True, index=df.index)
        for col in group_cols:
            if pd.isna(row[col]):
                unit_mask &= df[col].isna()
            else:
                unit_mask &= df[col].astype(str) == str(row[col])

        baseline_first_ref = row.get("baseline_first_half_hz", np.nan)
        baseline_second_ref = row.get("percent_reference_hz", np.nan)

        if pd.notna(baseline_first_ref) and baseline_first_ref != 0:
            baseline_second_windows = df[
                unit_mask
                & (df["phase"] == BASELINE_LABEL)
                & (df["baseline_half"] == BASELINE_SECOND_HALF_LABEL)
            ]["firing_rate_hz"]
            baseline_pct_windows = ((baseline_second_windows - baseline_first_ref) / baseline_first_ref) * 100.0
            out.loc[row_index, "baseline_percent_change_error"] = error_bar(baseline_pct_windows)

        if pd.notna(baseline_second_ref) and baseline_second_ref != 0:
            part_mask = unit_mask & (df["part_index"] == row["part_index"])

            stim_windows = df[part_mask & (df["phase"] == STIMULATION_LABEL)]["firing_rate_hz"]
            stim_pct_windows = ((stim_windows - baseline_second_ref) / baseline_second_ref) * 100.0
            out.loc[row_index, "stimulation_percent_change_error"] = error_bar(stim_pct_windows)

            post_windows = df[part_mask & (df["phase"] == POST_LABEL)]["firing_rate_hz"]
            post_pct_windows = ((post_windows - baseline_second_ref) / baseline_second_ref) * 100.0
            out.loc[row_index, "post_percent_change_error"] = error_bar(post_pct_windows)

    return out.sort_values([*group_cols, "part_index"], kind="stable").reset_index(drop=True)


def make_experiment_part_summary(
    part_table: pd.DataFrame,
    df: pd.DataFrame,
    group_column: str | None,
    comparisons: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    unpaired_rows: list[dict[str, object]] = []

    if part_table.empty:
        return pd.DataFrame(), pd.DataFrame()

    phase_specs = phase_specs_for_comparisons(comparisons)

    for (part_index, stimulation_label, part_label), part in part_table.groupby(
        ["part_index", "stimulation_label", "part_label"],
        dropna=False,
        sort=True,
    ):
        groups = ["All units"]
        if group_column:
            groups = sorted(part[group_column].dropna().astype(str).unique())

        for group in groups:
            sub = part if group == "All units" else part[part[group_column].astype(str) == group]
            if sub.empty:
                continue

            for spec in phase_specs:
                hz = sub[spec["hz_col"]] if spec["hz_col"] in sub.columns else pd.Series(dtype=float)
                pct = sub[spec["percent_col"]] if spec["percent_col"] in sub.columns else pd.Series(dtype=float)
                if hz.notna().sum() == 0 and pct.notna().sum() == 0:
                    continue

                test_name, p_value = paired_test(sub[spec["test_a"]], sub[spec["test_b"]])
                paired_n = int((sub[spec["test_a"]].notna() & sub[spec["test_b"]].notna()).sum())
                if pd.isna(p_value):
                    test_name, p_value, paired_n = window_level_paired_test(
                        df,
                        int(part_index),
                        spec["phase"],
                        group,
                        group_column,
                    )

                summary_rows.append(
                    {
                        "part_index": int(part_index),
                        "part_label": part_label,
                        "stimulation_label": stimulation_label,
                        "phase": spec["phase"],
                        "group": group,
                        "n_units": int(hz.notna().sum()),
                        "mean_hz": float(hz.mean()) if hz.notna().any() else np.nan,
                        "sem_hz": sem(hz),
                        "percent_reference_label": (
                            BASELINE_FIRST_HALF_LABEL
                            if spec["phase"] == BASELINE_LABEL
                            else sub["percent_reference_label"].iloc[0]
                        ),
                        "percent_reference_mean_hz": (
                            float(sub["baseline_first_half_hz"].mean())
                            if spec["phase"] == BASELINE_LABEL
                            else float(sub["percent_reference_hz"].mean())
                        ),
                        "percent_change_mean": float(pct.mean()) if pct.notna().any() else np.nan,
                        "percent_change_sem": sem(pct),
                        "percent_change_n": int(pct.notna().sum()),
                        "paired_test_used": test_name,
                        "paired_comparison": spec["comparison"],
                        "paired_n": paired_n,
                        "paired_p_value": p_value,
                        "paired_significance": p_to_stars(p_value),
                    }
                )

        if group_column:
            group_names = sorted(part[group_column].dropna().astype(str).unique())
            for spec in phase_specs:
                pct_col = spec["percent_col"]
                if pct_col not in part.columns:
                    continue
                for g1, g2 in itertools.combinations(group_names, 2):
                    a = part.loc[part[group_column].astype(str) == g1, pct_col]
                    b = part.loc[part[group_column].astype(str) == g2, pct_col]
                    test_name, p_value = unpaired_test(a, b)
                    unpaired_rows.append(
                        {
                            "part_index": int(part_index),
                            "part_label": part_label,
                            "stimulation_label": stimulation_label,
                            "phase": spec["phase"],
                            "group_1": g1,
                            "group_2": g2,
                            "n_group_1": int(a.notna().sum()),
                            "n_group_2": int(b.notna().sum()),
                            "group_1_percent_change_mean": float(a.mean()) if a.notna().any() else np.nan,
                            "group_2_percent_change_mean": float(b.mean()) if b.notna().any() else np.nan,
                            "unpaired_test_used": test_name,
                            "unpaired_p_value": p_value,
                            "unpaired_significance": p_to_stars(p_value),
                        }
                    )

    return pd.DataFrame(summary_rows), pd.DataFrame(unpaired_rows)


def paired_test(baseline: pd.Series, post: pd.Series) -> tuple[str, float]:
    baseline = pd.Series(baseline, dtype="float64")
    post = pd.Series(post, dtype="float64")
    valid = baseline.notna() & post.notna()
    baseline = baseline[valid]
    post = post[valid]
    n = len(baseline)
    if n < 2:
        return PAIRED_TEST, float("nan")

    diff = post.to_numpy() - baseline.to_numpy()
    if np.allclose(diff, 0, equal_nan=False):
        return PAIRED_TEST, 1.0

    result = stats.ttest_rel(post, baseline, nan_policy="omit")
    return PAIRED_TEST, float(result.pvalue)


def paired_test_ordered_windows(
    baseline: pd.Series,
    comparison: pd.Series,
    label: str = "paired t-test on matched windows",
) -> tuple[str, float, int]:
    """Pair ordered windows up to the shortest condition length."""
    baseline = pd.Series(baseline, dtype="float64").dropna().reset_index(drop=True)
    comparison = pd.Series(comparison, dtype="float64").dropna().reset_index(drop=True)
    n = min(len(baseline), len(comparison))
    if n < 2:
        return label, float("nan"), n

    baseline = baseline.iloc[:n]
    comparison = comparison.iloc[:n]
    diff = comparison.to_numpy() - baseline.to_numpy()
    if np.allclose(diff, 0, equal_nan=False):
        return label, 1.0, n

    result = stats.ttest_rel(comparison, baseline, nan_policy="omit")
    return label, float(result.pvalue), n


def sorted_window_rates(df: pd.DataFrame) -> pd.Series:
    sort_cols = experiment_sort_columns(df)
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable")
    return df["firing_rate_hz"]


def window_level_paired_test(
    df: pd.DataFrame,
    part_index: int,
    phase: str,
    group: str,
    group_column: str | None,
) -> tuple[str, float, int]:
    """Fallback paired t-test for one-unit datasets using matched ordered windows."""
    sub = df.copy()
    if group_column and group != "All units":
        sub = sub[sub[group_column].astype(str) == str(group)]

    if phase == BASELINE_LABEL:
        baseline = sorted_window_rates(
            sub[(sub["phase"] == BASELINE_LABEL) & (sub["baseline_half"] == BASELINE_FIRST_HALF_LABEL)]
        )
        comparison = sorted_window_rates(
            sub[(sub["phase"] == BASELINE_LABEL) & (sub["baseline_half"] == BASELINE_SECOND_HALF_LABEL)]
        )
        return paired_test_ordered_windows(
            baseline,
            comparison,
            label="paired t-test on matched Baseline windows",
        )

    baseline = sorted_window_rates(sub[sub["phase"] == BASELINE_LABEL])
    comparison = sorted_window_rates(sub[(sub["part_index"] == int(part_index)) & (sub["phase"] == phase)])
    return paired_test_ordered_windows(
        baseline,
        comparison,
        label=f"paired t-test on matched Baseline/{phase} windows",
    )


def unpaired_test(group_a: pd.Series, group_b: pd.Series) -> tuple[str, float]:
    group_a = pd.Series(group_a, dtype="float64").dropna()
    group_b = pd.Series(group_b, dtype="float64").dropna()
    if len(group_a) < 2 or len(group_b) < 2:
        return UNPAIRED_TEST, float("nan")

    result = stats.ttest_ind(group_a, group_b, equal_var=False, nan_policy="omit")
    return UNPAIRED_TEST, float(result.pvalue)


def make_summary_tables(
    paired_tables: dict[str, pd.DataFrame],
    baseline_change: pd.DataFrame,
    group_column: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    unpaired_rows: list[dict[str, object]] = []

    comparison_tables = {BASELINE_LABEL: baseline_change, **paired_tables}

    for post_label, paired in comparison_tables.items():
        if paired is None or paired.empty:
            continue

        groups = ["All units"]
        if group_column:
            groups = sorted(paired[group_column].dropna().astype(str).unique())

        for group in groups:
            sub = paired if group == "All units" else paired[paired[group_column].astype(str) == group]
            if sub.empty:
                continue

            test_name, p_value = paired_test(sub["baseline_hz"], sub["post_hz"])
            summary_rows.append(
                {
                    "post_condition": post_label,
                    "percent_reference_label": (
                        sub["percent_reference_label"].iloc[0]
                        if "percent_reference_label" in sub.columns
                        else ""
                    ),
                    "group": group,
                    "n_pairs": int(len(sub)),
                    "baseline_mean_hz": float(sub["baseline_hz"].mean()),
                    "baseline_sem_hz": sem(sub["baseline_hz"]),
                    "post_mean_hz": float(sub["post_hz"].mean()),
                    "post_sem_hz": sem(sub["post_hz"]),
                    "delta_mean_hz": float(sub["delta_hz"].mean()),
                    "delta_sem_hz": sem(sub["delta_hz"]),
                    "percent_reference_mean_hz": (
                        float(sub["percent_reference_hz"].mean())
                        if "percent_reference_hz" in sub.columns
                        else np.nan
                    ),
                    "percent_change_mean": float(sub["percent_change"].mean()),
                    "percent_change_sem": sem(sub["percent_change"]),
                    "percent_change_n": int(sub["percent_change"].notna().sum()),
                    "paired_test_used": test_name,
                    "paired_p_value": p_value,
                    "paired_significance": p_to_stars(p_value),
                }
            )

        if group_column:
            group_names = sorted(paired[group_column].dropna().astype(str).unique())
            for g1, g2 in itertools.combinations(group_names, 2):
                a = paired.loc[paired[group_column].astype(str) == g1, "percent_change"]
                b = paired.loc[paired[group_column].astype(str) == g2, "percent_change"]
                test_name, p_value = unpaired_test(a, b)
                unpaired_rows.append(
                    {
                        "post_condition": post_label,
                        "group_1": g1,
                        "group_2": g2,
                        "n_group_1": int(a.notna().sum()),
                        "n_group_2": int(b.notna().sum()),
                        "group_1_percent_change_mean": float(a.mean()) if a.notna().any() else np.nan,
                        "group_2_percent_change_mean": float(b.mean()) if b.notna().any() else np.nan,
                        "unpaired_test_used": test_name,
                        "unpaired_p_value": p_value,
                        "unpaired_significance": p_to_stars(p_value),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    unpaired = pd.DataFrame(unpaired_rows)
    return summary, unpaired


# =========================
# PLOTTING
# =========================

def plot_paired(
    paired_tables: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    group_column: str | None,
    out_dir: Path,
) -> None:
    paired_dir = out_dir / "paired_baseline_post"

    for post_label, paired in paired_tables.items():
        if paired.empty:
            print(f"Skipping paired plot for {post_label}: no paired units.")
            continue

        groups = ["All units"]
        if group_column:
            groups = sorted(paired[group_column].dropna().astype(str).unique())

        n_groups = len(groups)
        fig_width = max(3.1 * n_groups, 3.8)
        fig, axes = plt.subplots(1, n_groups, figsize=(fig_width, 4.2), sharey=True)
        if n_groups == 1:
            axes = [axes]

        all_y = paired[["baseline_hz", "post_hz"]].to_numpy().ravel()
        y_max = np.nanmax(all_y) if np.isfinite(all_y).any() else 1.0
        y_min = min(0.0, np.nanmin(all_y) if np.isfinite(all_y).any() else 0.0)
        y_span = max(y_max - y_min, 1.0)

        for ax, group in zip(axes, groups):
            sub = paired if group == "All units" else paired[paired[group_column].astype(str) == group]
            x = np.array([0, 1], dtype=float)

            for _, row in sub.iterrows():
                ax.plot(
                    x,
                    [row["baseline_hz"], row["post_hz"]],
                    color="black",
                    lw=0.9,
                    marker="o",
                    markersize=4.2,
                    alpha=0.82,
                    zorder=2,
                )

            means = [sub["baseline_hz"].mean(), sub["post_hz"].mean()]
            ax.plot(
                x,
                means,
                color="#d7191c",
                lw=2.0,
                marker="o",
                markersize=7.0,
                markeredgecolor="#d7191c",
                markerfacecolor="#d7191c",
                zorder=4,
            )

            row = summary[(summary["post_condition"] == post_label) & (summary["group"] == group)]
            sig = row["paired_significance"].iloc[0] if not row.empty else "n/a"
            bracket_y = y_max + 0.10 * y_span
            add_sig_label(ax, 0, 1, bracket_y, sig, 0.035 * y_span)

            ax.set_xticks([0, 1])
            ax.set_xticklabels([BASELINE_LABEL, post_label])
            ax.set_xlim(-0.45, 1.45)
            ax.set_ylim(y_min - 0.02 * y_span, y_max + 0.24 * y_span)
            ax.set_title(str(group))
            ax.set_ylabel("Spike frequency (Hz)")
            despine(ax)

        fig.suptitle(f"{BASELINE_LABEL} vs {post_label}", y=1.02, fontsize=13)
        fig.tight_layout()
        save_figure(fig, paired_dir, f"paired_{safe_name(BASELINE_LABEL)}_vs_{safe_name(post_label)}")


def bar_colors(n: int) -> list[str]:
    base = ["black", "white", "#777777", "#d9d9d9", "#4d4d4d", "#bdbdbd"]
    return [base[i % len(base)] for i in range(n)]


def plot_percent_change(
    paired_tables: dict[str, pd.DataFrame],
    baseline_change: pd.DataFrame,
    unpaired: pd.DataFrame,
    group_column: str | None,
    out_dir: Path,
) -> None:
    bar_dir = out_dir / "percent_change"

    for post_label, paired in paired_tables.items():
        if paired.empty and baseline_change.empty:
            print(f"Skipping percent-change plot for {post_label}: no paired units.")
            continue

        if group_column:
            group_values = []
            for table in (baseline_change, paired):
                if table is not None and not table.empty and group_column in table.columns:
                    group_values.extend(table[group_column].dropna().astype(str).unique())
            categories = sorted(dict.fromkeys(group_values))

            series = []
            if baseline_change is not None and not baseline_change.empty:
                series.append((BASELINE_LABEL, baseline_change, "white"))
            if paired is not None and not paired.empty:
                series.append((post_label, paired, "black"))

            x = np.arange(len(categories), dtype=float)
            bar_width = min(0.34, 0.72 / max(len(series), 1))
            offsets = (np.arange(len(series), dtype=float) - ((len(series) - 1) / 2.0)) * bar_width

            fig_width = max(4.2, 1.55 * len(categories) + 2.0)
            fig, ax = plt.subplots(figsize=(fig_width, 4.2))
            plotted_values: list[pd.Series] = []
            means_for_ylim: list[float] = []

            for series_index, (series_label, table, color) in enumerate(series):
                values_by_cat = [
                    table.loc[table[group_column].astype(str) == group, "percent_change"].dropna()
                    for group in categories
                ]
                means = [vals.mean() if len(vals) else np.nan for vals in values_by_cat]
                errors = [sem(vals) if len(vals) else np.nan for vals in values_by_cat]
                means_for_ylim.extend([m for m in means if np.isfinite(m)])
                plotted_values.extend(values_by_cat)

                positions = x + offsets[series_index]
                ax.bar(
                    positions,
                    means,
                    yerr=[err if np.isfinite(err) else 0 for err in errors],
                    width=bar_width * 0.9,
                    color=color,
                    edgecolor="black",
                    linewidth=1.1,
                    capsize=4,
                    error_kw={"elinewidth": 1.0, "capthick": 1.0},
                    label=series_label,
                    zorder=2,
                )

                for xi, vals in zip(positions, values_by_cat):
                    arr = vals.to_numpy(dtype=float)
                    if arr.size:
                        jitter = np.linspace(-bar_width * 0.18, bar_width * 0.18, arr.size) if arr.size > 1 else np.array([0.0])
                        ax.scatter(
                            xi + jitter,
                            arr,
                            s=18,
                            color="#555555",
                            edgecolor="none",
                            alpha=0.75,
                            zorder=3,
                        )

            ax.axhline(0, color="black", lw=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.set_ylabel("Firing change (%)")
            ax.set_title(f"Firing change: {post_label}")
            ax.legend(loc="best")
            despine(ax)

            finite_values = np.array(
                [v for vals in plotted_values for v in vals.to_numpy(dtype=float) if np.isfinite(v)]
                + means_for_ylim
            )
            if finite_values.size:
                y_min = min(0.0, float(np.nanmin(finite_values)))
                y_max = max(0.0, float(np.nanmax(finite_values)))
            else:
                y_min, y_max = -1.0, 1.0
            y_span = max(y_max - y_min, 1.0)

            bracket_added = False
            if len(categories) == 2 and paired is not None and not paired.empty and len(series) >= 2:
                test_row = unpaired[
                    (unpaired["post_condition"] == post_label)
                    & (unpaired["group_1"].astype(str) == categories[0])
                    & (unpaired["group_2"].astype(str) == categories[1])
                ]
                sig = test_row["unpaired_significance"].iloc[0] if not test_row.empty else "n/a"
                post_series_index = [label for label, _, _ in series].index(post_label)
                x1 = x[0] + offsets[post_series_index]
                x2 = x[1] + offsets[post_series_index]
                add_sig_label(ax, x1, x2, y_max + 0.08 * y_span, sig, 0.04 * y_span)
                bracket_added = sig != "n/a"

            top_pad = 0.30 if bracket_added else 0.18
            ax.set_ylim(*percent_axis_limits(y_min, y_max, top_pad=top_pad))

            fig.tight_layout()
            save_figure(fig, bar_dir, f"percent_change_{safe_name(BASELINE_LABEL)}_and_{safe_name(post_label)}_by_{safe_name(group_column)}")
        else:
            categories = []
            values_by_cat = []
            colors = []
            if baseline_change is not None and not baseline_change.empty:
                categories.append(BASELINE_LABEL)
                values_by_cat.append(baseline_change["percent_change"].dropna())
                colors.append("white")
            if paired is not None and not paired.empty:
                categories.append(post_label)
                values_by_cat.append(paired["percent_change"].dropna())
                colors.append("black")

            means = [vals.mean() if len(vals) else np.nan for vals in values_by_cat]
            errors = [sem(vals) if len(vals) else np.nan for vals in values_by_cat]
            x = np.arange(len(categories), dtype=float)

            fig_width = max(3.8, 1.15 * len(categories) + 1.7)
            fig, ax = plt.subplots(figsize=(fig_width, 4.2))

            for xi, mean, err, color, vals in zip(x, means, errors, colors, values_by_cat):
                ax.bar(
                    xi,
                    mean,
                    yerr=err if np.isfinite(err) else None,
                    width=0.62,
                    color=color,
                    edgecolor="black",
                    linewidth=1.1,
                    capsize=4,
                    error_kw={"elinewidth": 1.0, "capthick": 1.0},
                    zorder=2,
                )

                arr = vals.to_numpy(dtype=float)
                if arr.size:
                    jitter = np.linspace(-0.08, 0.08, arr.size) if arr.size > 1 else np.array([0.0])
                    ax.scatter(
                        xi + jitter,
                        arr,
                        s=18,
                        color="#555555",
                        edgecolor="none",
                        alpha=0.75,
                        zorder=3,
                    )

            ax.axhline(0, color="black", lw=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.set_ylabel("Firing change (%)")
            ax.set_title(f"Firing change: {post_label}")
            despine(ax)

            finite_values = np.array(
                [v for vals in values_by_cat for v in vals.to_numpy(dtype=float) if np.isfinite(v)]
                + [v for v in means if np.isfinite(v)]
            )
            if finite_values.size:
                y_min = min(0.0, float(np.nanmin(finite_values)))
                y_max = max(0.0, float(np.nanmax(finite_values)))
            else:
                y_min, y_max = -1.0, 1.0
            y_span = max(y_max - y_min, 1.0)
            ax.set_ylim(*percent_axis_limits(y_min, y_max, top_pad=0.16))

            fig.tight_layout()
            save_figure(fig, bar_dir, f"percent_change_{safe_name(BASELINE_LABEL)}_and_{safe_name(post_label)}")


def plot_baseline_split_percent_change(
    baseline_change: pd.DataFrame,
    group_column: str | None,
    out_dir: Path,
) -> None:
    """Plot Baseline first-half vs second-half percent change on its own."""
    if baseline_change is None or baseline_change.empty or "percent_change" not in baseline_change.columns:
        print("Skipping Baseline split percent-change plot: no valid split-Baseline values.")
        return

    plot_data = baseline_change[baseline_change["percent_change"].notna()].copy()
    if plot_data.empty:
        print("Skipping Baseline split percent-change plot: no valid split-Baseline values.")
        return

    groups = ["All units"]
    if group_column and group_column in plot_data.columns:
        groups = sorted(plot_data[group_column].dropna().astype(str).unique())

    n_groups = len(groups)
    fig_width = max(3.4 * n_groups, 3.8)
    fig, axes = plt.subplots(1, n_groups, figsize=(fig_width, 4.1), sharey=True)
    if n_groups == 1:
        axes = [axes]

    all_values = plot_data["percent_change"].to_numpy(dtype=float)
    y_min = min(0.0, float(np.nanmin(all_values)))
    y_max = max(0.0, float(np.nanmax(all_values)))
    y_span = max(y_max - y_min, 1.0)

    for ax, group in zip(axes, groups):
        sub = plot_data if group == "All units" else plot_data[plot_data[group_column].astype(str) == group]
        vals = sub["percent_change"].dropna().to_numpy(dtype=float)
        mean = float(np.nanmean(vals)) if vals.size else np.nan
        err = error_bar(vals) if vals.size else np.nan

        ax.bar(
            [0],
            [mean],
            yerr=[err] if np.isfinite(err) else None,
            width=0.58,
            color="white",
            edgecolor="black",
            linewidth=1.1,
            capsize=4,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
            zorder=2,
        )
        if vals.size:
            jitter = np.linspace(-0.08, 0.08, vals.size) if vals.size > 1 else np.array([0.0])
            ax.scatter(
                jitter,
                vals,
                s=18,
                color="#555555",
                edgecolor="none",
                alpha=0.75,
                zorder=3,
            )

        test_name, p_value = paired_test(sub["baseline_hz"], sub["post_hz"])
        add_sig_label(ax, -0.18, 0.18, y_max + 0.08 * y_span, p_to_stars(p_value), 0.04 * y_span)

        ax.set_xticks([0])
        ax.set_xticklabels([BASELINE_LABEL])
        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(*percent_axis_limits(y_min, y_max, top_pad=0.24))
        ax.set_ylabel("Firing change from Baseline (%)")
        ax.set_title("" if group == "All units" and group_column is None else str(group))
        despine(ax)
        use_zero_x_axis(ax)

    fig.suptitle("Baseline first half vs second half", y=1.02, fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_dir / "percent_change", "percent_change_baseline_first_half_vs_second_half")


def part_groups(part: pd.DataFrame, group_column: str | None) -> list[str]:
    if group_column:
        return sorted(part[group_column].dropna().astype(str).unique())
    return ["All units"]


def summary_sig(
    summary: pd.DataFrame,
    part_index: int,
    phase: str,
    group: str,
) -> str:
    if summary.empty:
        return "n/a"
    rows = summary[
        (summary["part_index"].astype(int) == int(part_index))
        & (summary["phase"] == phase)
        & (summary["group"] == group)
    ]
    if rows.empty:
        return "n/a"
    return rows["paired_significance"].iloc[0]


def plot_errors_for_columns(
    sub: pd.DataFrame,
    value_cols: list[str],
    fallback_error_cols: list[str],
) -> list[float]:
    """Use SEM/SD across units; fall back to within-window error for single-unit data."""
    errors: list[float] = []
    for value_col, fallback_col in zip(value_cols, fallback_error_cols):
        values = sub[value_col].dropna() if value_col in sub.columns else pd.Series(dtype=float)
        if len(values) > 1:
            errors.append(error_bar(values))
        elif fallback_col in sub.columns and sub[fallback_col].notna().any():
            errors.append(float(sub[fallback_col].dropna().iloc[0]))
        else:
            errors.append(float("nan"))
    return errors


def title_for_part_plot(
    part_index: object,
    stimulation_label: object,
    part_label: object,
    part_titles: dict[str, str] | None = None,
) -> str:
    default_title = str(stimulation_label)
    if not part_titles:
        return default_title

    keys = []
    with contextlib.suppress(TypeError, ValueError):
        keys.extend([f"part:{int(part_index)}", str(int(part_index))])
    keys.extend([str(part_label), str(stimulation_label)])
    for key in keys:
        title = part_titles.get(key, "").strip()
        if title:
            return title
    return default_title


def plot_part_spike_frequency(
    part_table: pd.DataFrame,
    summary: pd.DataFrame,
    group_column: str | None,
    out_dir: Path,
    comparisons: tuple[str, ...] | None = None,
) -> None:
    """Plot absolute spike frequency for Baseline and selected recorded phases."""
    if part_table.empty:
        print("Skipping spike-frequency part plots: no experiment parts found.")
        return

    freq_dir = out_dir / "spike_frequency_by_part"
    comparison_set = set(comparisons or DEFAULT_COMPARISONS)

    for (part_index, stimulation_label, part_label), part in part_table.groupby(
        ["part_index", "stimulation_label", "part_label"],
        dropna=False,
        sort=True,
    ):
        value_specs = [
            {
                "phase": BASELINE_LABEL,
                "label": BASELINE_LABEL,
                "value_col": "baseline_hz",
                "error_col": "baseline_window_error_hz",
            }
        ]
        if COMPARISON_STIMULATION in comparison_set and part["stimulation_hz"].notna().any():
            value_specs.append(
                {
                    "phase": STIMULATION_LABEL,
                    "label": str(stimulation_label),
                    "value_col": "stimulation_hz",
                    "error_col": "stimulation_window_error_hz",
                }
            )
        if COMPARISON_POST in comparison_set and part["post_hz"].notna().any():
            post_label = POST_LABEL
            if "stimulation_hz" in part.columns and not part["stimulation_hz"].notna().any():
                post_label = str(stimulation_label)
            value_specs.append(
                {
                    "phase": POST_LABEL,
                    "label": post_label,
                    "value_col": "post_hz",
                    "error_col": "post_window_error_hz",
                }
            )
        if len(value_specs) < 2:
            print(f"Skipping spike-frequency plot for part {int(part_index)}: no selected non-Baseline data.")
            continue

        value_cols = [spec["value_col"] for spec in value_specs]
        error_cols = [spec["error_col"] for spec in value_specs]
        xticklabels = [spec["label"] for spec in value_specs]
        phase_to_x = {spec["phase"]: i for i, spec in enumerate(value_specs)}

        groups = part_groups(part, group_column)
        n_groups = len(groups)
        fig_width = max(4.3 * n_groups, 4.6)
        fig, axes = plt.subplots(1, n_groups, figsize=(fig_width, 4.3), sharey=True)
        if n_groups == 1:
            axes = [axes]

        all_values_matrix = part[value_cols].to_numpy(dtype=float)
        if all(col in part.columns for col in error_cols):
            all_errors_matrix = part[error_cols].to_numpy(dtype=float)
            all_y = np.concatenate(
                [
                    all_values_matrix.ravel(),
                    (all_values_matrix + all_errors_matrix).ravel(),
                ]
            )
        else:
            all_y = all_values_matrix.ravel()
        finite_y = all_y[np.isfinite(all_y)]
        if finite_y.size == 0:
            plt.close(fig)
            continue

        y_min = min(0.0, float(np.nanmin(finite_y)))
        y_max = float(np.nanmax(finite_y))
        y_span = max(y_max - y_min, 1.0)
        x = np.arange(len(value_specs), dtype=float)

        for ax, group in zip(axes, groups):
            sub = part if group == "All units" else part[part[group_column].astype(str) == group]

            for _, row in sub.iterrows():
                y = row[value_cols].to_numpy(dtype=float)
                valid = np.isfinite(y)
                if valid.any():
                    ax.plot(
                        x[valid],
                        y[valid],
                        color="black",
                        lw=0.9,
                        marker="o",
                        markersize=4.2,
                        alpha=0.82,
                        zorder=2,
                    )

            means = [sub[col].mean() for col in value_cols]
            errors = plot_errors_for_columns(sub, value_cols, error_cols)
            valid_means = np.isfinite(means)
            plot_yerr = np.array(errors, dtype=float)
            plot_yerr[~np.isfinite(plot_yerr)] = 0.0
            ax.errorbar(
                x[valid_means],
                np.array(means, dtype=float)[valid_means],
                yerr=plot_yerr[valid_means],
                color="#d7191c",
                lw=2.0,
                elinewidth=1.2,
                capsize=4,
                capthick=1.2,
                marker="o",
                markersize=7.0,
                markeredgecolor="#d7191c",
                markerfacecolor="#d7191c",
                    zorder=4,
            )

            bracket_i = 0
            for phase in (STIMULATION_LABEL, POST_LABEL):
                if phase not in phase_to_x:
                    continue
                bracket_i += 1
                add_sig_label(
                    ax,
                    0,
                    phase_to_x[phase],
                    y_max + (0.08 + 0.11 * (bracket_i - 1)) * y_span,
                    summary_sig(summary, int(part_index), phase, group),
                    0.035 * y_span,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(xticklabels)
            ax.set_xlim(-0.45, len(value_specs) - 0.55)
            ax.set_ylim(y_min - 0.03 * y_span, y_max + 0.34 * y_span)
            ax.set_ylabel("Spike frequency (Hz)")
            ax.set_title("" if group == "All units" and group_column is None else str(group))
            despine(ax)

        fig.suptitle(str(stimulation_label), y=1.02, fontsize=13)
        fig.tight_layout()
        save_figure(fig, freq_dir, f"spike_frequency_part_{int(part_index):02d}_{safe_name(stimulation_label)}")


def plot_part_percent_change(
    part_table: pd.DataFrame,
    summary: pd.DataFrame,
    group_column: str | None,
    out_dir: Path,
    comparisons: tuple[str, ...] | None = None,
    pool_comparisons: bool = False,
    df: pd.DataFrame | None = None,
    baseline_change: pd.DataFrame | None = None,
    id_columns: tuple[str, ...] | None = None,
    pooled_title: str | None = None,
    part_titles: dict[str, str] | None = None,
) -> None:
    """Plot selected percent-change comparisons for each experiment part."""
    percent_dir = out_dir / "percent_change_by_part"
    if pool_comparisons:
        if df is None or baseline_change is None or id_columns is None:
            raise ValueError("Pooled percent-change plotting needs window-level data and ID columns.")
        plot_pooled_part_percent_change(
            df,
            baseline_change,
            id_columns,
            group_column,
            percent_dir,
            comparisons,
            title=pooled_title,
        )
        return

    if part_table.empty:
        print("Skipping percent-change part plots: no experiment parts found.")
        return

    comparison_set = set(comparisons or DEFAULT_COMPARISONS)

    for (part_index, stimulation_label, part_label), part in part_table.groupby(
        ["part_index", "stimulation_label", "part_label"],
        dropna=False,
        sort=True,
    ):
        value_specs: list[dict[str, str]] = []
        if COMPARISON_BASELINE in comparison_set and part["baseline_percent_change"].notna().any():
            value_specs.append(
                {
                    "phase": BASELINE_LABEL,
                    "label": BASELINE_LABEL,
                    "value_col": "baseline_percent_change",
                    "error_col": "baseline_percent_change_error",
                    "color": "white",
                }
            )
        if COMPARISON_STIMULATION in comparison_set and part["stimulation_percent_change"].notna().any():
            value_specs.append(
                {
                    "phase": STIMULATION_LABEL,
                    "label": str(stimulation_label),
                    "value_col": "stimulation_percent_change",
                    "error_col": "stimulation_percent_change_error",
                    "color": "#777777",
                }
            )
        if COMPARISON_POST in comparison_set and part["post_percent_change"].notna().any():
            post_label = POST_LABEL
            if "stimulation_hz" in part.columns and not part["stimulation_hz"].notna().any():
                post_label = str(stimulation_label)
            value_specs.append(
                {
                    "phase": POST_LABEL,
                    "label": post_label,
                    "value_col": "post_percent_change",
                    "error_col": "post_percent_change_error",
                    "color": "black",
                }
            )
        if not value_specs:
            print(f"Skipping percent-change plot for part {int(part_index)}: no selected comparison data.")
            continue

        value_cols = [spec["value_col"] for spec in value_specs]
        error_cols = [spec["error_col"] for spec in value_specs]
        colors = [spec["color"] for spec in value_specs]
        xticklabels = [spec["label"] for spec in value_specs]
        phase_to_x = {spec["phase"]: i for i, spec in enumerate(value_specs)}

        groups = part_groups(part, group_column)
        n_groups = len(groups)
        fig_width = max(4.1 * n_groups, 4.4)
        fig, axes = plt.subplots(1, n_groups, figsize=(fig_width, 4.3), sharey=True)
        if n_groups == 1:
            axes = [axes]

        all_values_matrix = part[value_cols].to_numpy(dtype=float)
        if all(col in part.columns for col in error_cols):
            all_errors_matrix = part[error_cols].to_numpy(dtype=float)
            all_y = np.concatenate(
                [
                    all_values_matrix.ravel(),
                    (all_values_matrix + all_errors_matrix).ravel(),
                    (all_values_matrix - all_errors_matrix).ravel(),
                ]
            )
        else:
            all_y = all_values_matrix.ravel()
        finite_y = all_y[np.isfinite(all_y)]
        if finite_y.size == 0:
            plt.close(fig)
            continue

        y_min = min(0.0, float(np.nanmin(finite_y)))
        y_max = max(0.0, float(np.nanmax(finite_y)))
        y_span = max(y_max - y_min, 1.0)
        x = np.arange(len(value_specs), dtype=float)

        for ax, group in zip(axes, groups):
            sub = part if group == "All units" else part[part[group_column].astype(str) == group]
            means = [sub[col].mean() for col in value_cols]
            errors = plot_errors_for_columns(sub, value_cols, error_cols)

            for xi, mean, err, color, col in zip(x, means, errors, colors, value_cols):
                ax.bar(
                    xi,
                    mean,
                    yerr=err if np.isfinite(err) else None,
                    width=0.62,
                    color=color,
                    edgecolor="black",
                    linewidth=1.1,
                    capsize=4,
                    error_kw={"elinewidth": 1.0, "capthick": 1.0},
                    zorder=2,
                )

                vals = sub[col].dropna().to_numpy(dtype=float)
                if vals.size:
                    jitter = np.linspace(-0.08, 0.08, vals.size) if vals.size > 1 else np.array([0.0])
                    ax.scatter(
                        xi + jitter,
                        vals,
                        s=18,
                        color="#555555",
                        edgecolor="none",
                        alpha=0.75,
                        zorder=3,
                    )

            bracket_i = 0
            baseline_x = phase_to_x.get(BASELINE_LABEL)
            if baseline_x is not None:
                for phase in (STIMULATION_LABEL, POST_LABEL):
                    if phase not in phase_to_x:
                        continue
                    bracket_i += 1
                    add_sig_label(
                        ax,
                        baseline_x,
                        phase_to_x[phase],
                        y_max + (0.08 + 0.12 * (bracket_i - 1)) * y_span,
                        summary_sig(summary, int(part_index), phase, group),
                        0.04 * y_span,
                    )

            ax.set_xticks(x)
            ax.set_xticklabels(xticklabels)
            ax.set_xlim(-0.45, len(value_specs) - 0.55)
            ax.set_ylim(*percent_axis_limits(y_min, y_max, top_pad=0.36))
            ax.set_ylabel("Firing change from Baseline (%)")
            ax.set_title("" if group == "All units" and group_column is None else str(group))
            despine(ax)
            use_zero_x_axis(ax)

        fig.suptitle(title_for_part_plot(part_index, stimulation_label, part_label, part_titles), y=1.02, fontsize=13)
        fig.tight_layout()
        save_figure(fig, percent_dir, f"percent_change_part_{int(part_index):02d}_{safe_name(stimulation_label)}")


def plot_pooled_part_percent_change(
    df: pd.DataFrame,
    baseline_change: pd.DataFrame,
    id_columns: tuple[str, ...],
    group_column: str | None,
    percent_dir: Path,
    comparisons: tuple[str, ...] | None = None,
    title: str | None = None,
) -> None:
    """Plot all selected percent-change categories in one figure without merging labels."""
    comparison_set = set(comparisons or DEFAULT_COMPARISONS)
    group_cols = list(id_columns)
    if group_column:
        group_cols.append(group_column)

    def row_group_label(row: pd.Series) -> str:
        if not group_column:
            return "All units"
        value = row.get(group_column)
        return "Unlabeled" if pd.isna(value) else str(value)

    def group_labels(frame: pd.DataFrame) -> pd.Series:
        if not group_column:
            return pd.Series("All units", index=frame.index)
        return frame[group_column].map(lambda value: "Unlabeled" if pd.isna(value) else str(value))

    def filter_group(frame: pd.DataFrame, group: str) -> pd.DataFrame:
        if not group_column:
            return frame
        return frame[group_labels(frame) == str(group)]

    baseline_means = (
        df[df["phase"] == BASELINE_LABEL]
        .groupby(group_cols, dropna=False)
        .agg(baseline_hz=("firing_rate_hz", "mean"))
        .reset_index()
    )

    reference_cols = [*group_cols, "baseline_second_half_hz"]
    available_reference_cols = [col for col in reference_cols if col in baseline_change.columns]
    references = baseline_means.copy()
    if available_reference_cols:
        references = references.merge(
            baseline_change[available_reference_cols].drop_duplicates(group_cols),
            on=group_cols,
            how="left",
        )
    if "baseline_second_half_hz" not in references.columns:
        references["baseline_second_half_hz"] = np.nan
    references["baseline_hz"] = pd.to_numeric(references["baseline_hz"], errors="coerce")
    references["baseline_second_half_hz"] = pd.to_numeric(
        references["baseline_second_half_hz"],
        errors="coerce",
    )
    references["percent_reference_hz"] = references["baseline_second_half_hz"].where(
        references["baseline_second_half_hz"].notna(),
        references["baseline_hz"],
    )

    category_specs: list[dict[str, object]] = []
    pooled_rows: list[dict[str, object]] = []
    significance_by_group_key: dict[tuple[str, str], str] = {}
    fallback_error_by_group_key: dict[tuple[str, str], float] = {}

    def add_category(key: str, label: str, color: str) -> None:
        if key in {str(spec["key"]) for spec in category_specs}:
            return
        category_specs.append(
            {
                "key": key,
                "label": label,
                "color": color,
            }
        )

    def set_significance(group: str, key: str, p_value: float) -> None:
        significance_by_group_key[(str(group), str(key))] = p_to_stars(p_value)

    def set_fallback_error(group: str, key: str, values: Iterable[float]) -> None:
        err = error_bar(pd.Series(values, dtype="float64"))
        if np.isfinite(err):
            fallback_error_by_group_key[(str(group), str(key))] = err

    def row_mask(frame: pd.DataFrame, row: pd.Series, columns: list[str]) -> pd.Series:
        mask = pd.Series(True, index=frame.index)
        for col in columns:
            if col not in frame.columns or col not in row.index:
                continue
            if pd.isna(row[col]):
                mask &= frame[col].isna()
            else:
                mask &= frame[col].astype(str) == str(row[col])
        return mask

    if COMPARISON_BASELINE in comparison_set and baseline_change is not None and not baseline_change.empty:
        baseline_values = baseline_change[baseline_change["percent_change"].notna()].copy()
        if not baseline_values.empty:
            add_category("baseline", BASELINE_LABEL, "white")
            for _, row in baseline_values.iterrows():
                pooled_rows.append(
                    {
                        "category_key": "baseline",
                        "value": float(row["percent_change"]),
                        "group": row_group_label(row),
                    }
                )

            for group in sorted(group_labels(baseline_values).dropna().astype(str).unique()):
                sub = filter_group(baseline_values, group)
                _, p_value = paired_test(sub["baseline_hz"], sub["post_hz"])
                fallback_values: list[float] = []
                window_sub = filter_group(df, group)
                for _, unit_row in sub.iterrows():
                    ref = unit_row.get("baseline_first_half_hz", np.nan)
                    if pd.isna(ref) or ref == 0:
                        continue
                    unit_windows = window_sub[row_mask(window_sub, unit_row, group_cols)]
                    second_half_windows = unit_windows[
                        (unit_windows["phase"] == BASELINE_LABEL)
                        & (unit_windows["baseline_half"] == BASELINE_SECOND_HALF_LABEL)
                    ]["firing_rate_hz"]
                    fallback_values.extend(((second_half_windows - ref) / ref * 100.0).dropna().tolist())
                set_fallback_error(group, "baseline", fallback_values)
                if pd.isna(p_value):
                    first_half = sorted_window_rates(
                        window_sub[
                            (window_sub["phase"] == BASELINE_LABEL)
                            & (window_sub["baseline_half"] == BASELINE_FIRST_HALF_LABEL)
                        ]
                    )
                    second_half = sorted_window_rates(
                        window_sub[
                            (window_sub["phase"] == BASELINE_LABEL)
                            & (window_sub["baseline_half"] == BASELINE_SECOND_HALF_LABEL)
                        ]
                    )
                    _, p_value, _ = paired_test_ordered_windows(
                        first_half,
                        second_half,
                        label="paired t-test on matched Baseline windows",
                    )
                set_significance(group, "baseline", p_value)

    def add_phase_rows(
        phase: str,
        label_col: str,
        fallback_label: str,
        comparison_name: str,
        color: str,
    ) -> None:
        if comparison_name not in comparison_set:
            return
        phase_rows = df[df["phase"] == phase].copy()
        if phase_rows.empty:
            return

        phase_rows["_pooled_label"] = phase_rows[label_col].where(
            phase_rows[label_col].notna(),
            fallback_label,
        )
        phase_rows["_pooled_label"] = phase_rows["_pooled_label"].astype(str).str.strip()
        phase_rows.loc[phase_rows["_pooled_label"] == "", "_pooled_label"] = fallback_label
        if phase == POST_LABEL and "part_index" in phase_rows.columns:
            generic_post = phase_rows["_pooled_label"].map(normalise_name).eq("post")
            numbered_part = generic_post & phase_rows["part_index"].notna()
            phase_rows.loc[numbered_part, "_pooled_label"] = (
                "Post " + phase_rows.loc[numbered_part, "part_index"].astype(int).astype(str)
            )

        label_order = (
            phase_rows.sort_values("_row_order", kind="stable")["_pooled_label"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        phase_means = (
            phase_rows.groupby([*group_cols, "_pooled_label"], dropna=False)
            .agg(phase_hz=("firing_rate_hz", "mean"))
            .reset_index()
        )
        phase_means = phase_means.merge(
            references[[*group_cols, "baseline_hz", "percent_reference_hz"]],
            on=group_cols,
            how="left",
        )
        phase_means["percent_change"] = np.where(
            phase_means["percent_reference_hz"] != 0,
            ((phase_means["phase_hz"] - phase_means["percent_reference_hz"]) / phase_means["percent_reference_hz"]) * 100.0,
            np.nan,
        )

        for label in label_order:
            key = f"{phase}:{label}"
            label_means = phase_means[phase_means["_pooled_label"].astype(str) == label]
            if label_means["percent_change"].notna().any():
                add_category(key, label, color)
                for group in sorted(group_labels(label_means).dropna().astype(str).unique()):
                    sub = filter_group(label_means, group)
                    _, p_value = paired_test(sub["baseline_hz"], sub["phase_hz"])
                    fallback_values: list[float] = []
                    grouped_phase_rows = filter_group(phase_rows, group)
                    for _, unit_row in sub.iterrows():
                        ref = unit_row.get("percent_reference_hz", np.nan)
                        if pd.isna(ref) or ref == 0:
                            continue
                        unit_phase_rows = grouped_phase_rows[row_mask(grouped_phase_rows, unit_row, group_cols)]
                        unit_phase_rows = unit_phase_rows[unit_phase_rows["_pooled_label"].astype(str) == label]
                        fallback_values.extend(
                            ((unit_phase_rows["firing_rate_hz"] - ref) / ref * 100.0).dropna().tolist()
                        )
                    set_fallback_error(group, key, fallback_values)
                    if pd.isna(p_value):
                        grouped_windows = filter_group(df, group)
                        baseline_windows = sorted_window_rates(
                            grouped_windows[grouped_windows["phase"] == BASELINE_LABEL]
                        )
                        comparison_windows = sorted_window_rates(
                            grouped_phase_rows[grouped_phase_rows["_pooled_label"].astype(str) == label]
                        )
                        _, p_value, _ = paired_test_ordered_windows(
                            baseline_windows,
                            comparison_windows,
                            label=f"paired t-test on matched Baseline/{label} windows",
                        )
                    set_significance(group, key, p_value)

        for _, row in phase_means[phase_means["percent_change"].notna()].iterrows():
            label = str(row["_pooled_label"])
            pooled_rows.append(
                {
                    "category_key": f"{phase}:{label}",
                    "value": float(row["percent_change"]),
                    "group": row_group_label(row),
                }
            )

    add_phase_rows(
        STIMULATION_LABEL,
        "stimulation_label_raw",
        STIMULATION_LABEL,
        COMPARISON_STIMULATION,
        "#777777",
    )
    add_phase_rows(
        POST_LABEL,
        "post_label_raw",
        POST_LABEL,
        COMPARISON_POST,
        "black",
    )

    pooled = pd.DataFrame(pooled_rows)
    if pooled.empty:
        print("Skipping pooled percent-change plot: no selected percent-change values.")
        return

    groups = sorted(pooled["group"].dropna().astype(str).unique()) if group_column else ["All units"]
    category_specs = [
        spec
        for spec in category_specs
        if (pooled["category_key"] == spec["key"]).any()
    ]
    labels = [str(spec["label"]) for spec in category_specs]
    colors = [str(spec["color"]) for spec in category_specs]
    category_keys = [str(spec["key"]) for spec in category_specs]
    n_groups = len(groups)
    fig_width = max(4.4 * n_groups, (0.82 * len(category_specs) + 2.1) * n_groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(fig_width, 4.3), sharey=True)
    if n_groups == 1:
        axes = [axes]

    group_stats: dict[str, dict[str, object]] = {}
    plot_y_values: list[float] = []
    for group in groups:
        sub = pooled[pooled["group"] == group]
        means: list[float] = []
        errors: list[float] = []
        values_by_label: list[np.ndarray] = []
        for key in category_keys:
            vals = sub.loc[sub["category_key"] == key, "value"].dropna().to_numpy(dtype=float)
            values_by_label.append(vals)
            mean = float(np.nanmean(vals)) if vals.size else float("nan")
            if vals.size > 1:
                err = error_bar(vals)
            else:
                err = fallback_error_by_group_key.get((str(group), key), float("nan"))
            means.append(mean)
            errors.append(err)
            if np.isfinite(mean):
                plot_y_values.append(mean)
                if np.isfinite(err):
                    plot_y_values.extend([mean - err, mean + err])
        group_stats[group] = {
            "means": means,
            "errors": errors,
            "values_by_label": values_by_label,
        }

    finite_values = np.array([value for value in plot_y_values if np.isfinite(value)], dtype=float)
    if finite_values.size == 0:
        plt.close(fig)
        print("Skipping pooled percent-change plot: no finite values.")
        return
    y_min = min(0.0, float(np.nanmin(finite_values)))
    y_max = max(0.0, float(np.nanmax(finite_values)))
    y_span = max(y_max - y_min, 1.0)
    top_pad = min(0.95, 0.24 + 0.10 * max(1, len(category_specs)))
    x = np.arange(len(category_specs), dtype=float)
    baseline_x = category_keys.index("baseline") if "baseline" in category_keys else None

    for ax, group in zip(axes, groups):
        stats_for_group = group_stats[group]
        means = stats_for_group["means"]
        errors = stats_for_group["errors"]
        values_by_label = stats_for_group["values_by_label"]

        for xi, mean, err, color, vals in zip(x, means, errors, colors, values_by_label):
            if np.isfinite(mean):
                ax.bar(
                    xi,
                    mean,
                    yerr=err if np.isfinite(err) else None,
                    width=0.62,
                    color=color,
                    edgecolor="black",
                    linewidth=1.1,
                    capsize=4,
                    error_kw={"elinewidth": 1.0, "capthick": 1.0},
                    zorder=2,
                )

            if vals.size:
                jitter = np.linspace(-0.08, 0.08, vals.size) if vals.size > 1 else np.array([0.0])
                ax.scatter(
                    xi + jitter,
                    vals,
                    s=18,
                    color="#555555",
                    edgecolor="none",
                    alpha=0.75,
                    zorder=3,
                )

        if baseline_x is not None:
            bracket_index = 0
            baseline_sig = significance_by_group_key.get((str(group), "baseline"), "n/a")
            if baseline_sig != "n/a":
                add_sig_label(
                    ax,
                    baseline_x - 0.18,
                    baseline_x + 0.18,
                    y_max + 0.08 * y_span,
                    baseline_sig,
                    0.04 * y_span,
                )
                bracket_index += 1

            for xi, key in enumerate(category_keys):
                if xi == baseline_x:
                    continue
                sig = significance_by_group_key.get((str(group), key), "n/a")
                if sig == "n/a":
                    continue
                add_sig_label(
                    ax,
                    baseline_x,
                    xi,
                    y_max + (0.10 + 0.11 * bracket_index) * y_span,
                    sig,
                    0.04 * y_span,
                )
                bracket_index += 1

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        if len(labels) > 4:
            ax.tick_params(axis="x", labelrotation=30)
            for label in ax.get_xticklabels():
                label.set_ha("right")
        ax.set_xlim(-0.45, len(category_specs) - 0.55)
        ax.set_ylim(*percent_axis_limits(y_min, y_max, top_pad=top_pad))
        ax.set_ylabel("Firing change from Baseline (%)")
        ax.set_title("" if group == "All units" and group_column is None else str(group))
        despine(ax)
        use_zero_x_axis(ax)

    fig.suptitle((title or "Firing change from Baseline").strip() or "Firing change from Baseline", y=1.02, fontsize=13)
    fig.tight_layout()
    save_figure(fig, percent_dir, "percent_change_pooled_by_phase")


def plot_time_course(
    df: pd.DataFrame,
    id_columns: tuple[str, ...],
    group_column: str | None,
    out_dir: Path,
) -> None:
    if not MAKE_TIME_COURSE:
        return

    time_dir = out_dir / "time_course"
    group_cols = list(id_columns)
    if group_column:
        group_cols.append(group_column)

    units = list(df.groupby(group_cols, dropna=False))
    if len(units) > MAX_TIME_COURSE_UNITS:
        print(
            f"Skipping time-course plots because {len(units)} units were found "
            f"(MAX_TIME_COURSE_UNITS={MAX_TIME_COURSE_UNITS})."
        )
        return

    condition_colors = {
        BASELINE_LABEL: "black",
        STIMULATION_LABEL: "#777777",
        POST_LABEL: "#1f77b4",
        "Post 1": "#1f77b4",
        "Post 2": "#2ca02c",
    }

    for unit_key, sub in units:
        if not isinstance(unit_key, tuple):
            unit_key = (unit_key,)

        sub = sub.sort_values(["window_start_s", "window_index"], kind="stable").copy()
        x = sub["window_start_s"] / 60.0
        if x.isna().all():
            x = np.arange(len(sub), dtype=float)
            x_label = "Window"
        else:
            x_label = "Time (min)"

        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        for condition, cond_df in sub.groupby("condition", sort=False):
            cond_df = cond_df.sort_values(["window_start_s", "window_index"], kind="stable")
            cond_x = cond_df["window_start_s"] / 60.0 if x_label == "Time (min)" else np.arange(len(cond_df))
            ax.plot(
                cond_x,
                cond_df["firing_rate_hz"],
                color=condition_colors.get(condition, "#555555"),
                marker="o",
                markersize=3.4,
                lw=1.2,
                label=condition,
            )

        boundaries = (
            sub.sort_values(["window_start_s", "window_index"])
            .assign(prev_condition=lambda d: d["condition"].shift())
            .query("prev_condition.notna() and condition != prev_condition")
        )
        for _, row in boundaries.iterrows():
            if pd.notna(row.get("window_start_s")):
                ax.axvline(row["window_start_s"] / 60.0, color="#999999", lw=0.8, ls="--", zorder=1)

        title_parts = [str(value) for value in unit_key]
        ax.set_title(" | ".join(title_parts))
        ax.set_xlabel(x_label)
        ax.set_ylabel("Firing rate (Hz)")
        ax.legend(loc="best")
        despine(ax)
        fig.tight_layout()

        name = "_".join(safe_name(part) for part in unit_key)
        save_figure(fig, time_dir, f"time_course_{name}")


def write_outputs(
    df: pd.DataFrame,
    unit_means: pd.DataFrame,
    part_table: pd.DataFrame,
    baseline_change: pd.DataFrame,
    summary: pd.DataFrame,
    unpaired: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "cleaned_window_level_firing_rates.csv", index=False)
    unit_means.to_csv(out_dir / "unit_condition_mean_firing_rates.csv", index=False)
    baseline_change.to_csv(out_dir / "baseline_split_percent_changes.csv", index=False)
    part_table.to_csv(out_dir / "experiment_part_unit_values.csv", index=False)
    summary.to_csv(out_dir / "experiment_part_summary.csv", index=False)
    summary.to_csv(out_dir / "baseline_post_summary.csv", index=False)

    leading_id_cols = []
    if "part_index" in part_table.columns:
        leading_id_cols = list(part_table.columns[: part_table.columns.get_loc("part_index")])

    percent_cols = [
        col for col in [
            "part_index",
            "part_label",
            "stimulation_label",
            "baseline_percent_change",
            "stimulation_percent_change",
            "post_percent_change",
            "baseline_percent_change_error",
            "stimulation_percent_change_error",
            "post_percent_change_error",
            "percent_reference_hz",
            "percent_reference_label",
        ]
        if col in part_table.columns
    ]
    unit_percent_cols = [*leading_id_cols, *percent_cols]
    if unit_percent_cols:
        part_table[unit_percent_cols].to_csv(out_dir / "unit_percent_changes.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "unit_percent_changes.csv", index=False)

    if not unpaired.empty:
        unpaired.to_csv(out_dir / "unpaired_percent_change_tests.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-style Baseline vs Post electrophysiology firing-rate plots."
    )
    parser.add_argument("input", nargs="?", default=INPUT_PATH, help="Path to .csv, .xlsx, or .xls dataset.")
    parser.add_argument(
        "--input-dir",
        default=INPUT_DIR,
        help="Folder to scan when no input file is given. Default: INPUT_DIR setting.",
    )
    parser.add_argument("--out-dir", default=OUT_DIR_NAME, help="Output directory. Default: <input stem>_baseline_post_plots.")
    parser.add_argument("--sheet", default=EXCEL_SHEET, help="Excel sheet name/index. Default: first sheet.")
    parser.add_argument(
        "--csv-sep",
        default=CSV_SEPARATOR,
        help="CSV delimiter. Default auto-detects. Use ',' ';' or '\\t' if needed.",
    )
    parser.add_argument("--group-col", default=GROUP_COLUMN, help="Optional grouping column for unpaired tests.")
    parser.add_argument(
        "--id-cols",
        default=",".join(ID_COLUMNS),
        help="Comma-separated paired-unit ID columns. Default: recording_name,channel.",
    )
    parser.add_argument(
        "--no-time-course",
        action="store_true",
        help="Skip per-recording/channel time-course plots.",
    )
    parser.add_argument(
        "--recorded-phases",
        default=",".join(DEFAULT_RECORDED_PHASES),
        help="Comma-separated phases present in this experiment: baseline, stimulation, post, or all.",
    )
    parser.add_argument(
        "--comparisons",
        default=",".join(DEFAULT_COMPARISONS),
        help=(
            "Comma-separated comparisons to output: baseline, stimulation, post, or all. "
            "Use baseline,post for Baseline + Post-only experiments."
        ),
    )
    parser.add_argument(
        "--pool-percent-change-comparisons",
        action="store_true",
        default=POOL_PERCENT_CHANGE_COMPARISONS,
        help="Pool selected stimulation/Post percent-change values across experiment parts into one plot.",
    )
    parser.add_argument(
        "--pooled-percent-title",
        default="",
        help="Custom title for the pooled percent-change plot.",
    )
    parser.add_argument(
        "--percent-part-titles",
        default="",
        help=(
            "Custom titles for non-pooled percent-change part plots. "
            "Accepts JSON, e.g. {\"part:1\":\"Post 1\"}, or semicolon key=title entries."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    recorded_phases = parse_recorded_phases(args.recorded_phases)
    comparison_names = parse_comparisons(args.comparisons)
    comparison_names = filter_comparisons_for_phases(comparison_names, recorded_phases)
    if not comparison_names:
        raise ValueError("No comparisons remain after applying the selected recorded phases.")
    if BASELINE_LABEL not in recorded_phases:
        raise ValueError("Baseline must be included because all comparisons use it as the reference.")
    percent_part_titles = parse_title_mapping(args.percent_part_titles)

    input_path = find_input_file(
        args.input,
        args.input_dir,
        INPUT_PATTERNS,
        auto_select_newest=AUTO_SELECT_NEWEST_INPUT,
    )
    out_dir = Path(args.out_dir) if args.out_dir else input_path.with_name(f"{input_path.stem}_baseline_post_plots")
    prepare_output_dir(out_dir)

    global MAKE_TIME_COURSE
    if args.no_time_course:
        MAKE_TIME_COURSE = False

    sheet: str | int = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)

    csv_sep = args.csv_sep
    if csv_sep == r"\t":
        csv_sep = "\t"
    if csv_sep == "":
        csv_sep = None

    print(f"Loading: {input_path}")
    raw = load_table(input_path, sheet=sheet, csv_sep=csv_sep)
    df = validate_and_prepare_data(raw)
    df = df[df["phase"].isin(recorded_phases)].copy()
    if df.empty:
        raise ValueError("No rows remain after applying the selected recorded phases.")

    id_columns = tuple(col.strip() for col in args.id_cols.split(",") if col.strip())
    group_column = choose_group_column(df, args.group_col)
    if group_column:
        print(f"Using group column for unpaired tests: {group_column}")
    else:
        print("No group column found/requested; unpaired percent-change tests will be skipped.")

    allow_post_without_stimulation = (
        STIMULATION_LABEL not in recorded_phases
        or not (df["phase"] == STIMULATION_LABEL).any()
    )
    print("Recorded phases: " + ", ".join(recorded_phases))
    print("Comparisons: " + ", ".join(comparison_names))
    if args.pool_percent_change_comparisons:
        print("Percent-change part plots: pooled stimulation/Post values across experiment parts.")
    df = assign_experiment_parts(
        df,
        id_columns,
        group_column,
        allow_post_without_stimulation=allow_post_without_stimulation,
    )
    detected_parts = (
        df.loc[df["part_index"].notna(), ["part_index", "stimulation_label"]]
        .drop_duplicates()
        .sort_values("part_index")
    )
    if detected_parts.empty:
        if any(name in comparison_names for name in (COMPARISON_STIMULATION, COMPARISON_POST)):
            raise ValueError(
                "No selected stimulation/Post experiment parts were found. "
                "For Post-only experiments, use --recorded-phases baseline,post "
                "and label post recordings distinctly, such as Post 1 and Post 2."
            )
        print("No stimulation/Post experiment parts were found; writing Baseline-only outputs.")
    else:
        print("Detected experiment parts:")
        for _, row in detected_parts.iterrows():
            print(f"  Part {int(row['part_index'])}: {row['stimulation_label']}")

    df = add_baseline_half_labels(df, id_columns, group_column)
    baseline_change = build_baseline_half_table(df, id_columns, group_column)
    if baseline_change.empty or baseline_change["percent_change"].notna().sum() == 0:
        print(
            "No valid split-Baseline percent changes were found. "
            "Baseline needs at least two valid windows per paired unit."
        )

    unit_means = collapse_to_unit_means(df, id_columns, group_column)
    part_table = build_experiment_part_table(df, baseline_change, id_columns, group_column)
    summary, unpaired = make_experiment_part_summary(part_table, df, group_column, comparison_names)

    if part_table.empty and any(name in comparison_names for name in (COMPARISON_STIMULATION, COMPARISON_POST)):
        raise ValueError(
            "No complete experiment part values were found. Check that each paired unit has Baseline "
            "and at least one selected stimulation/Post segment."
        )

    if COMPARISON_BASELINE in comparison_names:
        plot_baseline_split_percent_change(baseline_change, group_column, out_dir)
    if any(name in comparison_names for name in (COMPARISON_STIMULATION, COMPARISON_POST)):
        plot_part_spike_frequency(part_table, summary, group_column, out_dir, comparison_names)
    plot_part_percent_change(
        part_table,
        summary,
        group_column,
        out_dir,
        comparison_names,
        pool_comparisons=args.pool_percent_change_comparisons,
        df=df,
        baseline_change=baseline_change,
        id_columns=id_columns,
        pooled_title=args.pooled_percent_title,
        part_titles=percent_part_titles,
    )
    plot_time_course(df, id_columns, group_column, out_dir)
    write_outputs(df, unit_means, part_table, baseline_change, summary, unpaired, out_dir)

    print("\nDone.")
    print(f"Saved outputs to: {out_dir.resolve()}")
    print("Main summary CSV: baseline_post_summary.csv")


if __name__ == "__main__":
    main()
