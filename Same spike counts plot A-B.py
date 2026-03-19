import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# =========================
# USER SETTINGS
# =========================

CSV_DIR = r"C:\Users\simmons\Desktop\Exploring PSDs"

LABEL_A = "Recovery Baseline"
LABEL_B = "Post -1V (600s)"

COLOR_A = "tab:blue"
COLOR_B = "tab:orange"

OUT_PNG = r"C:\Users\simmons\Desktop\Exploring PSDs\combined_spike_count_curve.png"

WINDOW_MIN = 1.0  # minutes per bin (used only for labeling)

# ---------------------------------
# MINUTES TO PLOT
# Use None to keep everything
# Use 1-based minute numbers
# Examples:
#   A_MINUTES = None
#   A_MINUTES = (1, 10)      -> plot minutes 1 to 10 inclusive
#   A_MINUTES = [1, 2, 5, 8] -> plot only those minutes
# ---------------------------------
A_MINUTES = (13, 23)
B_MINUTES = None

# Optional: label shown at the boundary line
BOUNDARY_TEXT = "[-1V 10mins]"


def find_condition_csv(csv_dir, keyword):
    matches = [
        os.path.join(csv_dir, f)
        for f in os.listdir(csv_dir)
        if f.lower().endswith(".csv") and keyword.lower() in f.lower()
    ]
    if len(matches) == 0:
        raise FileNotFoundError(f"No CSV found containing '{keyword}' in filename.")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple CSVs found containing '{keyword}':\n" + "\n".join(matches)
        )
    return matches[0]


def drop_last_incomplete_window(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Drop the final row (often incomplete minute)."""
    if len(df) < 2:
        raise ValueError(
            f"CSV {name} has <2 windows, cannot drop the last minute safely."
        )
    return df.iloc[:-1].reset_index(drop=True)

def select_minutes(df: pd.DataFrame, minutes, name: str) -> pd.DataFrame:
    """
    Select which minutes/windows to keep.
    Supports:
      - None: keep all rows
      - tuple(start, end): keep inclusive range, 1-based
      - list of minute numbers: keep only those rows, 1-based
    """
    if minutes is None:
        return df.copy()

    df = df.reset_index(drop=True).copy()
    df["plot_minute"] = np.arange(1, len(df) + 1)

    if isinstance(minutes, tuple) and len(minutes) == 2:
        start, end = minutes
        if start < 1 or end < start:
            raise ValueError(f"{name}: invalid minute range {minutes}")
        out = df[(df["plot_minute"] >= start) & (df["plot_minute"] <= end)].copy()

    elif isinstance(minutes, list):
        wanted = set(minutes)
        if any(m < 1 for m in wanted):
            raise ValueError(f"{name}: minute numbers must be >= 1")
        out = df[df["plot_minute"].isin(wanted)].copy()

    else:
        raise ValueError(
            f"{name}: minutes must be None, a tuple like (1,10), or a list like [1,2,5]"
        )

    if out.empty:
        raise ValueError(f"{name}: no rows left after minute selection")

    return out.reset_index(drop=True)

# =========================
# LOAD DATA
# =========================

CSV_A = find_condition_csv(CSV_DIR, "Baseline")
CSV_B = find_condition_csv(CSV_DIR, "Post")

print(f"[INFO] Baseline CSV: {os.path.basename(CSV_A)}")
print(f"[INFO] Post CSV: {os.path.basename(CSV_B)}")

df_a = pd.read_csv(CSV_A)
df_b = pd.read_csv(CSV_B)

# Safety checks
for name, df in zip(["A", "B"], [df_a, df_b]):
    if "window_index" not in df.columns or "spike_count" not in df.columns:
        raise ValueError(
            f"CSV {name} must contain 'window_index' and 'spike_count' columns"
        )

# Sort just in case
df_a = df_a.sort_values("window_index").reset_index(drop=True)
df_b = df_b.sort_values("window_index").reset_index(drop=True)

# =========================
# OPTIONAL MINUTE SELECTION
# =========================
df_a = select_minutes(df_a, A_MINUTES, "A (Baseline)")
df_b = select_minutes(df_b, B_MINUTES, "B (Post)")

# =========================
# DROP LAST MINUTE (INCOMPLETE) FOR EACH CSV
# =========================
df_a = drop_last_incomplete_window(df_a, "A (Baseline)")
df_b = drop_last_incomplete_window(df_b, "B (Post)")

print(f"[INFO] Baseline windows kept: {len(df_a)}")
print(f"[INFO] Post windows kept: {len(df_b)}")

# =========================
# BUILD STITCHED TIME AXIS (B AFTER A)
# =========================

# Ensure A runs 1..N regardless of whatever window_index is in the CSV
N_a = len(df_a)
N_b = len(df_b)

x_a = np.arange(1, N_a + 1)          # A: 1..N_a
y_a = df_a["spike_count"].to_numpy()

offset = x_a[-1]                      # last A minute number
x_b = offset + np.arange(1, N_b + 1)  # B plotted after A
y_b = df_b["spike_count"].to_numpy()

# Boundary line between last A and first B
boundary_x = offset + 0.5

# =========================
# PLOT
# =========================

plt.figure(figsize=(12, 5))

plt.plot(
    x_a, y_a,
    marker="o",
    color=COLOR_A,
    label=LABEL_A
)

plt.plot(
    x_b, y_b,
    marker="o",
    color=COLOR_B,
    label=LABEL_B
)

plt.ylim(bottom=0)
# Vertical line marking transition
plt.axvline(boundary_x, color="k", linestyle="--", alpha=0.5)



# Boundary label centered between the two lines
y_top = max(
    np.max(y_a) if len(y_a) else 0,
    np.max(y_b) if len(y_b) else 0
)

plt.text(
    boundary_x,
    y_top * 0.75 if y_top > 0 else 1.0,
    BOUNDARY_TEXT,
    rotation=90,
    va="top",
    ha="center",
    fontsize=10,
    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        edgecolor="none",
        alpha=0.8
    )
)


plt.xlabel("Time (minutes)")
plt.ylabel("Spike count")
plt.title("Spike counts - 2nd run")

# =========================
# RESET X TICK LABELS AT B (this is the key change)
# =========================
tick_step_a = 1   # baseline keeps every minute
tick_step_b = 1   # post shows every 5 minutes

ticks_a = x_a[::tick_step_a]
ticks_b = x_b[::tick_step_b]

labels_a = [f"{int(t)}" for t in ticks_a]
labels_b = [f"{int(t - offset)}" for t in ticks_b]

plt.xticks(list(ticks_a) + list(ticks_b), labels_a + labels_b, rotation=45, ha="right")
plt.legend()
plt.tight_layout()

plt.savefig(OUT_PNG, dpi=200)
plt.close()

print(f"[DONE] Combined spike count plot saved to:\n{OUT_PNG}")
