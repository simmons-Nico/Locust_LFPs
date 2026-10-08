"""Keep refractory selection identical while making long recordings practical."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "Spike Processing" / "Spike Count Multiple CSVs ordered.py"
spec = importlib.util.spec_from_file_location("spike_scaling", PATH)
spike = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spike)


def original_selection(events, refractory):
    kept = []
    for event in sorted(events, key=lambda e: (-float(e["amplitude_uv"]), 0 if e["polarity"] == "neg" else 1, int(e["sample_index"]))):
        if all(abs(event["sample_index"] - other["sample_index"]) >= max(1, refractory) for other in kept):
            kept.append(event)
    return sorted(kept, key=lambda e: e["sample_index"])


class RefractorySelectionTests(unittest.TestCase):
    def test_matches_original_including_ties_duplicates_and_boundaries(self):
        rng = np.random.default_rng(42)
        for refractory in (0, 1, 2, 20, 100):
            events = [dict(sample_index=int(i), amplitude_uv=float(a), polarity=p)
                      for i, a, p in zip(rng.integers(-100, 1000, 700),
                                         rng.integers(1, 10, 700),
                                         rng.choice(["neg", "pos"], 700))]
            self.assertEqual(spike.suppress_refractory_events_by_amplitude(events, refractory),
                             original_selection(events, refractory))
        self.assertEqual(spike.suppress_refractory_events_by_amplitude([], 20), [])

    def test_long_recording_keeps_all_well_separated_events(self):
        # The former implementation would perform about five billion comparisons.
        events = [dict(sample_index=i * 20, amplitude_uv=100., polarity="neg")
                  for i in range(100_000)]
        self.assertEqual(spike.suppress_refractory_events_by_amplitude(events, 20), events)


if __name__ == "__main__":
    unittest.main()
