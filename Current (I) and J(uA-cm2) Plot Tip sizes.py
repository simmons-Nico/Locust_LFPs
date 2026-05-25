import pandas as pd
import matplotlib.pyplot as plt
import re

file_path = r"C:\Users\simmons\Desktop\current densities.xlsx"

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

df["Tip size (um)"] = df["Tip size"].apply(tip_to_um)

summary = (
    df.groupby(["Tip size", "Tip size (um)"], as_index=False)
      .agg(
          I_mean=("Current in (uA)", "mean"),
          I_std=("Current in (uA)", "std"),
          J_mean=("J(uA/cm2)", "mean"),
          J_std=("J(uA/cm2)", "std")
      )
      .sort_values("Tip size (um)")
)

summary["I_std"] = summary["I_std"].fillna(0)
summary["J_std"] = summary["J_std"].fillna(0)

fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.errorbar(
    summary["Tip size"],
    summary["J_mean"],
    yerr=summary["J_std"],
    fmt="o-",
    capsize=5,
    color="tab:blue",
    label="Current density J"
)

ax1.set_xlabel("Electrode tip size")
ax1.set_ylabel("Current density J (uA/cm2)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()

ax2.errorbar(
    summary["Tip size"],
    summary["I_mean"],
    yerr=summary["I_std"],
    fmt="s-",
    capsize=5,
    color="tab:red",
    label="Current I"
)

ax2.set_ylabel("Current I (uA)", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

plt.title("Current Density and Current vs Electrode Tip Size")
fig.tight_layout()
plt.show()
