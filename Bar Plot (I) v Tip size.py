import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

file_path = r"C:\Users\simmons\Desktop\current densities.xlsx"

tip_col = "Tip size"
value_col = "uM in 100uL of PBS"

df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

def tip_to_um(value):
    value = str(value).strip().lower()
    number = float(re.findall(r"\d+\.?\d*", value)[0])

    if "mm" in value:
        return number * 1000
    elif "um" in value or "µm" in value:
        return number
    else:
        return number

df["Tip size (um)"] = df[tip_col].apply(tip_to_um)

summary = (
    df.groupby([tip_col, "Tip size (um)"], as_index=False)
      .agg(
          mean_value=(value_col, "mean"),
          std_value=(value_col, "std"),
          count=(value_col, "count")
      )
      .sort_values("Tip size (um)")
)

summary["std_value"] = summary["std_value"].fillna(0)

x = np.arange(len(summary))
labels = summary[tip_col].astype(str)

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    x,
    summary["mean_value"],
    yerr=summary["std_value"],
    capsize=6,
    color="grey",
    edgecolor="black",
    width=0.6,
    linewidth=0.6,
    label="Mean ± SD"
)

rng = np.random.default_rng(4)


for i, tip in enumerate(summary[tip_col]):
    values = df.loc[df[tip_col] == tip, value_col].dropna()
    jitter = rng.normal(0, 0.05, size=len(values))

    ax.scatter(
        np.full(len(values), x[i]) + jitter,
        values,
        s=35,
        marker="o",
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=3,
        label="Individual data points" if i == 0 else None
    )

ax.set_xlabel("Electrode PEDOT Tip size")
ax.set_ylabel("H2O2 Concentration (µM)")
ax.set_title(" H2O2 after 30 mins of -0.9V")
ax.set_xticks(x)
ax.set_xticklabels(labels)


fig.tight_layout()
plt.show()
