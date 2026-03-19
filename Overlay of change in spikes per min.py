import matplotlib.pyplot as plt
import numpy as np

OUT_PNG = r"C:\Users\simmons\Desktop\Exploring PSDs\overlay_spike_counts_H2O2_Pos_Control.png"
# =========================
# SPIKE COUNT DATA
# =========================

RECORDINGS = {
    "Rec 1": {
        "baseline": [1220, 1130, 1410, 1410, 1290, 850, 1170, 700, 1460, 1310, 1800, 2520],
        "post":     [3150, 3580, 4060, 1120, 20, 0, 140, 200, 220, 225, 30, 60],
    },
    "Rec 2": {
        "baseline": [635, 600, 620, 690, 750, 885, 890, 1020, 890, 1010, 1045, 1050],
        "post":     [2240, 650, 280, 150, 200, 250, 290, 360, 455, 500, 650, 705],
    },
    "Rec 3": {
        "baseline": [1120, 1290, 1340, 1180, 1850, 1500, 1250, 1590, 1220, 1470, 1300, 1370],
        "post":     [30, 25, 15, 20, 90, 220, 260, 820, 1200, 480, 600, 370],
    },
}

# =========================
# COLORS
# =========================

BASELINE_COLORS = {
    "Rec 1": "lightblue",
    "Rec 2": "lightcoral",
    "Rec 3": "lightgreen",
}

POST_COLORS = {
    "Rec 1": "blue",
    "Rec 2": "red",
    "Rec 3": "green",
}

# =========================
# PLOT
# =========================

plt.figure(figsize=(14,6))

baseline_handles = []
post_handles = []

max_baseline_len = max(len(v["baseline"]) for v in RECORDINGS.values())
max_post_len = max(len(v["post"]) for v in RECORDINGS.values())

offset = max_baseline_len + 1
boundary_x = max_baseline_len + 0.5

for name, data in RECORDINGS.items():

    baseline = data["baseline"]
    post = data["post"]

    x_base = np.arange(1, len(baseline) + 1)
    x_post = offset + np.arange(1, len(post) + 1)

    # Baseline plot
    line_base, = plt.plot(
        x_base,
        baseline,
        marker="o",
        linewidth=2,
        color=BASELINE_COLORS[name],
        label=f"{name} Baseline"
    )

    baseline_handles.append(line_base)

    # Post plot
    line_post, = plt.plot(
        x_post,
        post,
        marker="o",
        linewidth=2.5,
        color=POST_COLORS[name],
        label=f"{name} Post"
    )

    post_handles.append(line_post)

# =========================
# INJECTION LINE
# =========================

plt.axvline(boundary_x, linestyle="--", color="black", alpha=0.5)

plt.text(
    boundary_x,
    plt.ylim()[1] * 0.9,
    "Start of H2O2 Injection",
    rotation=90,
    ha="center",
    va="top"
)

# =========================
# AXES
# =========================

plt.xlabel("Time (minutes)")
plt.ylabel("Spike count")
plt.title("Spike count (1 min bins) — H2O2 Posotive Control")

# =========================
# LEGENDS
# =========================

baseline_legend = plt.legend(
    handles=baseline_handles,
    loc="upper left",
    title="Baseline"
)

post_legend = plt.legend(
    handles=post_handles,
    loc="upper right",
    title="Post"
)

plt.gca().add_artist(baseline_legend)

# =========================
# CLEANUP
# =========================

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)

print(f"[DONE] Figure saved to:\n{OUT_PNG}")
plt.show()