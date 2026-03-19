import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Folder path
FOLDER_PATH = Path(r"C:\Users\simmons\Desktop\Exploring PSDs")

# Find CSV files
csv_files = list(FOLDER_PATH.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found in the folder.")

print("\nAvailable CSV files:\n")
for i, f in enumerate(csv_files):
    print(f"{i+1}: {f.name}")

choice = int(input("\nSelect a file number: ")) - 1
csv_path = csv_files[choice]

print(f"\nLoading {csv_path.name}...")

# Load CSV
df = pd.read_csv(csv_path)
df = df.iloc[:-1]

# X axis
if "window_index" in df.columns:
    x = df["window_index"]
else:
    x = range(len(df))

y = df["spike_count"]

# Plot
plt.figure(figsize=(14,4))
ax = plt.gca()



plt.plot(
    x,
    y,
    marker="o",
    linewidth=1.5,
    markersize=4
)

plt.title(f"Spike count per 1-minute window — {csv_path.stem}")
plt.xlabel("Time window")
plt.ylabel("Spike count")
plt.ylim(bottom=0)
plt.grid(False)

plt.tight_layout()

# Save output
output_file = FOLDER_PATH / f"{csv_path.stem}_plot.png"
plt.savefig(output_file, dpi=300)

print(f"\nSaved plot to:\n{output_file}")

plt.show()
