"""Plot independent-preparation spike counts in separate aligned epochs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Plotting and Utilities"))
from preparation_spike_analysis import cli

def main():
    cli(default_mode="overlay")


if __name__ == "__main__":
    main()
