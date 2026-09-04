from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for module_dir in (
    ROOT / "Raw Conversion",
    ROOT / "LFP Processing",
    ROOT / "Spike Processing",
    ROOT / "Plotting and Utilities",
):
    sys.path.insert(0, str(module_dir))

from lfp_feature_extraction import LFPConfig, create_lfp_stream, process_lfp_recordings
from spike_lfp_coupling import SpikeLFPCouplingConfig, calculate_spike_lfp_coupling


def _tone_amplitude(t: np.ndarray, x: np.ndarray, freq_hz: float) -> float:
    basis = np.column_stack(
        [
            np.sin(2.0 * np.pi * freq_hz * t),
            np.cos(2.0 * np.pi * freq_hz * t),
            np.ones_like(t),
        ]
    )
    coef, *_ = np.linalg.lstsq(basis, x, rcond=None)
    return float(np.hypot(coef[0], coef[1]))


class LFPDualStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_two_khz_lfp_stream_preserves_passband_and_suppresses_aliases(self) -> None:
        fs = 20_000.0
        t = np.arange(0, 8.0, 1.0 / fs)
        x = np.zeros_like(t)
        for freq in (1.0, 50.0, 100.0, 200.0, 300.0):
            x += np.sin(2.0 * np.pi * freq * t)
        x += 6.0 * np.sin(2.0 * np.pi * 1500.0 * t)
        x += 6.0 * np.sin(2.0 * np.pi * 2800.0 * t)

        stream = create_lfp_stream(t, x, np.ones_like(t, dtype=bool), fs, 2000.0)
        meta = stream.metadata
        self.assertEqual(meta["resample_ratio"], "1/10")
        self.assertAlmostEqual(stream.fs_hz, 2000.0)
        self.assertLessEqual(meta["measured_passband_ripple_db"], 0.1)
        self.assertGreaterEqual(meta["measured_stopband_attenuation_db"], 60.0)
        self.assertEqual(len(meta["coeff_sha256"]), 64)
        self.assertAlmostEqual(float(np.median(np.diff(stream.t_rel_s))), 0.0005, places=7)

        center = (stream.t_rel_s >= 0.5) & (stream.t_rel_s <= 7.5)
        for freq in (1.0, 50.0, 100.0, 200.0, 300.0):
            amp = _tone_amplitude(stream.t_rel_s[center], stream.x_uv[center], freq)
            self.assertGreater(amp, 0.80, msg=f"{freq:g} Hz passband tone was attenuated too much")
            self.assertLess(amp, 1.20, msg=f"{freq:g} Hz passband tone changed too much")
        self.assertLess(_tone_amplitude(stream.t_rel_s[center], stream.x_uv[center], 500.0), 0.05)
        self.assertLess(_tone_amplitude(stream.t_rel_s[center], stream.x_uv[center], 800.0), 0.05)

    def test_feature_extraction_defaults_use_two_khz_and_three_hundred_hz_psd(self) -> None:
        fs = 20_000.0
        t = np.arange(0, 20.0, 1.0 / fs)
        x = 20.0 * np.sin(2.0 * np.pi * 10.0 * t) + 5.0 * np.sin(2.0 * np.pi * 220.0 * t)
        raw = self.root / "raw.csv"
        pd.DataFrame({"time_s": t, "A-001": x}).to_csv(raw, index=False)
        cfg = LFPConfig(
            output_dir=self.root / "lfp",
            analysis_window_s=10.0,
            welch_segment_s=4.0,
            notch_frequency_hz=None,
            filter_edge_s=0.0,
            write_simple_plots=False,
        )
        paths = process_lfp_recordings(
            [
                {
                    "source_file": str(raw),
                    "recording_id": "raw",
                    "segment_id": "raw",
                    "epoch_label": "Baseline",
                    "phase": "baseline",
                    "preparation_id": "prep1",
                    "signal_unit": "uV",
                }
            ],
            cfg,
        )
        features = pd.read_csv(paths["features_csv"])
        psd = pd.read_csv(paths["psd_long_csv"])
        valid = features[features["qc_pass"].astype(bool)]
        self.assertFalse(valid.empty)
        self.assertTrue(np.allclose(pd.to_numeric(valid["fs_used_hz"], errors="coerce"), 2000.0))
        self.assertLessEqual(float(pd.to_numeric(psd["frequency_hz"], errors="coerce").max()), 300.0)
        self.assertTrue(np.isfinite(pd.to_numeric(valid["band_high_150_300_power_uV2"], errors="coerce")).any())

    def test_known_phase_locked_spikes_recover_high_ppc(self) -> None:
        fs = 20_000.0
        freq = 8.0
        t = np.arange(0, 12.0, 1.0 / fs)
        x = 40.0 * np.sin(2.0 * np.pi * freq * t)
        raw = self.root / "phase_lock.csv"
        pd.DataFrame({"time_s": t, "A-001": x}).to_csv(raw, index=False)

        spike_times = np.arange(1.0 + 0.25 / freq, 11.0, 1.0 / freq)
        spikes = pd.DataFrame(
            {
                "preparation_id": "prep1",
                "segment_id": "phase_lock",
                "recording_id": "phase_lock",
                "source_file": str(raw),
                "epoch_label": "Baseline",
                "phase": "baseline",
                "channel": "A-001",
                "event_index": np.arange(1, spike_times.size + 1),
                "spike_time_s": spike_times,
                "segment_spike_time_s": spike_times,
                "recording_relative_spike_time_s": spike_times,
                "sample_index": np.rint(spike_times * fs).astype(int),
                "source_sampling_rate_hz": fs,
                "qc_pass": True,
            }
        )
        spike_csv = self.root / "spike_events.csv"
        spikes.to_csv(spike_csv, index=False)
        features = pd.DataFrame(
            [
                {
                    "preparation_id": "prep1",
                    "segment_id": "phase_lock",
                    "recording_id": "phase_lock",
                    "source_file": str(raw),
                    "epoch_label": "Baseline",
                    "phase": "baseline",
                    "channel": "A-001",
                    "window_index": 0,
                    "window_start_s": 1.0,
                    "window_end_s": 11.0,
                    "window_mid_s": 6.0,
                    "cumulative_window_mid_s": 6.0,
                    "window_duration_s": 10.0,
                    "usable_duration_s": 10.0,
                    "primary_power_uV2": 100.0,
                    "signal_qc_pass": True,
                    "feature_qc_pass": True,
                    "qc_pass": True,
                }
            ]
        )
        feature_csv = self.root / "features.csv"
        features.to_csv(feature_csv, index=False)
        cfg = SpikeLFPCouplingConfig(
            output_dir=self.root / "coupling",
            lfp_feature_csv=feature_csv,
            phase_bands=[("theta", "6-10 Hz", (6.0, 10.0))],
            window_s=10.0,
            min_spikes=20,
            surrogate_count=5,
            notch_frequency_hz=None,
            max_sfc_frequencies=8,
        )
        paths = calculate_spike_lfp_coupling(
            spike_csv,
            [
                {
                    "source_file": str(raw),
                    "recording_id": "phase_lock",
                    "segment_id": "phase_lock",
                    "epoch_label": "Baseline",
                    "phase": "baseline",
                    "preparation_id": "prep1",
                    "channel": "A-001",
                    "signal_unit": "uV",
                }
            ],
            cfg,
        )
        ppc = pd.read_csv(paths["ppc_csv"])
        valid = ppc[ppc["qc_pass"].astype(bool)]
        self.assertFalse(valid.empty)
        self.assertGreater(float(valid["ppc"].max()), 0.85)
        phase = float(valid["preferred_phase_rad"].dropna().iloc[0])
        self.assertLess(abs(np.angle(np.exp(1j * phase))), 0.5)
        self.assertTrue(Path(paths["primary_ppc_plot"]).exists())


if __name__ == "__main__":
    unittest.main()
