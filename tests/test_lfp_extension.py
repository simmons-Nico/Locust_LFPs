#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for module_dir in (
    ROOT / "Apps" / "SPIE",
    ROOT / "Raw Conversion",
    ROOT / "LFP Processing",
    ROOT / "Spike Processing",
    ROOT / "Plotting and Utilities",
):
    sys.path.insert(0, str(module_dir))

from lfp_feature_extraction import LFPConfig, process_lfp_recordings, _event_features
from lfp_plotting import plot_lfp_features
from spike_lfp_coupling import pairwise_phase_consistency


def write_sine_csv(path: Path, amp_uv: float = 10.0, freq_hz: float = 10.0, fs: float = 250.0, duration_s: float = 20.0) -> Path:
    t = np.arange(0, duration_s, 1.0 / fs)
    x = amp_uv * np.sin(2.0 * np.pi * freq_hz * t)
    pd.DataFrame({"time_s": t, "A-001": x}).to_csv(path, index=False)
    return path


def write_multi_sine_csv(path: Path, amplitudes: dict[str, float], freq_hz: float = 10.0, fs: float = 250.0, duration_s: float = 20.0) -> Path:
    t = np.arange(0, duration_s, 1.0 / fs)
    data = {"time_s": t}
    for channel, amp in amplitudes.items():
        data[channel] = amp * np.sin(2.0 * np.pi * freq_hz * t)
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def base_config(out_dir: Path, **overrides) -> LFPConfig:
    cfg = LFPConfig(
        output_dir=out_dir,
        analysis_window_s=10.0,
        welch_segment_s=4.0,
        exploratory_band_hz=(0.5, 100.0),
        primary_band_hz=(1.0, 40.0),
        marker_bands=[("primary", "Primary 1-40 Hz", (1.0, 40.0))],
        notch_frequency_hz=None,
        downsample_hz=None,
        filter_edge_s=0.0,
        minimum_valid_fraction=0.90,
        write_simple_plots=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class LFPExtensionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_known_sine_peak_and_band_power(self) -> None:
        path = write_sine_csv(self.root / "baseline.csv", amp_uv=10.0, freq_hz=10.0)
        records = [{"source_file": str(path), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "b1", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out"))
        features = pd.read_csv(paths["features_csv"])
        valid = features[features["qc_pass"].astype(bool)]
        self.assertFalse(valid.empty)
        self.assertAlmostEqual(float(valid["dominant_peak_frequency_hz"].median()), 10.0, delta=0.25)
        self.assertGreater(float(valid["primary_power_uV2"].median()), 40.0)

    def test_lfp_plotting_writes_per_recording_psd_and_bandpower(self) -> None:
        path = write_sine_csv(self.root / "baseline.csv", amp_uv=10.0, freq_hz=10.0)
        records = [{"source_file": str(path), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "b1", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out"))
        outputs = plot_lfp_features(paths["features_csv"], {"output_dir": self.root / "plots"})
        names = {Path(plot).name for plot in outputs["plots"]}
        self.assertTrue(any(name.startswith("lfp_psd_db_") for name in names))
        self.assertTrue(any(name.startswith("lfp_psd_linear_") for name in names))
        self.assertTrue(any(name.startswith("lfp_bandpower_bars_") for name in names))
        self.assertTrue(any(name.startswith("lfp_narrow_band_power_") for name in names))

    def test_feature_extraction_outputs_simple_lfp_plots_when_enabled(self) -> None:
        path = write_sine_csv(self.root / "baseline.csv", amp_uv=10.0, freq_hz=10.0)
        records = [{"source_file": str(path), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "b1", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out", write_simple_plots=True))
        manifest = json.loads(Path(paths["simple_lfp_plot_outputs_json"]).read_text(encoding="utf-8"))
        names = {Path(plot).name for plot in manifest["plots"]}
        self.assertTrue(any(name.startswith("lfp_psd_db_") for name in names))
        self.assertTrue(any(name.startswith("lfp_psd_linear_") for name in names))
        self.assertTrue(any(name.startswith("lfp_bandpower_bars_") for name in names))
        self.assertTrue(any(name.startswith("lfp_narrow_band_power_") for name in names))
        band_defs = pd.read_csv(Path(paths["simple_lfp_plots_dir"]) / "lfp_simple_narrow_band_definitions.csv")
        self.assertEqual(
            list(band_defs["suggested_label"]),
            [
                "Very slow",
                "Slow motor",
                "Intermediate motor",
                "Fictive-flight range",
                "Intact-flight range",
                "Motor harmonic/fast synaptic",
                "Fast population activity",
            ],
        )
        band_power = pd.read_csv(Path(paths["simple_lfp_plots_dir"]) / "lfp_simple_narrow_band_power_by_window.csv")
        self.assertEqual(set(band_power["band_key"]), set(band_defs["band_key"]))
        self.assertFalse((Path(paths["simple_lfp_plots_dir"]) / "lfp_paired_baseline_post_by_preparation.png").exists())

    def test_simple_psd_comparisons_include_baseline_overlay_and_same_channel_post(self) -> None:
        base1 = write_multi_sine_csv(self.root / "base1.csv", {"A-001": 10.0, "A-002": 8.0})
        base2 = write_multi_sine_csv(self.root / "base2.csv", {"A-001": 12.0, "A-002": 9.0})
        post = write_multi_sine_csv(self.root / "post.csv", {"A-001": 20.0, "A-002": 6.0})
        records = [
            {"source_file": str(base1), "phase": "baseline", "epoch_label": "Baseline 1", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "base1", "recording_order": 1, "signal_unit": "uV"},
            {"source_file": str(base2), "phase": "baseline", "epoch_label": "Baseline 2", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "base2", "recording_order": 2, "signal_unit": "uV"},
            {"source_file": str(post), "phase": "post", "epoch_label": "Post", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "post", "recording_order": 3, "signal_unit": "uV"},
        ]
        paths = process_lfp_recordings(records, base_config(self.root / "out", channels=["A-001", "A-002"], write_simple_plots=True))
        manifest = json.loads(Path(paths["simple_lfp_plot_outputs_json"]).read_text(encoding="utf-8"))
        names = {Path(plot).name for plot in manifest["plots"]}
        self.assertTrue(any(name.startswith("lfp_simple_baseline_psd_overlay_") for name in names))
        self.assertTrue(any(name.startswith("lfp_simple_same_channel_baseline_post_psd_") and "A-001" in name for name in names))
        self.assertTrue(any(name.startswith("lfp_simple_same_channel_baseline_post_psd_") and "A-002" in name for name in names))
        self.assertTrue(any(name.startswith("lfp_simple_same_channel_post_minus_baseline_psd_") and "A-001" in name for name in names))
        self.assertTrue(any(name.startswith("lfp_simple_same_channel_post_minus_baseline_psd_") and "A-002" in name for name in names))

    def test_doubling_sine_amplitude_gives_six_db_power_change(self) -> None:
        base = write_sine_csv(self.root / "base.csv", amp_uv=10.0)
        post = write_sine_csv(self.root / "post.csv", amp_uv=20.0)
        records = [
            {"source_file": str(base), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "base", "recording_order": 1, "signal_unit": "uV"},
            {"source_file": str(post), "phase": "post", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "post", "recording_order": 2, "signal_unit": "uV"},
        ]
        paths = process_lfp_recordings(records, base_config(self.root / "out"))
        features = pd.read_csv(paths["features_csv"])
        post_rows = features[(features["phase"] == "post") & (features["qc_pass"].astype(bool))]
        self.assertAlmostEqual(float(post_rows["primary_power_uV2_change_db"].median()), 6.0206, delta=0.15)

    def test_random_phase_spikes_have_ppc_near_zero(self) -> None:
        rng = np.random.default_rng(5)
        ppc = pairwise_phase_consistency(rng.uniform(-np.pi, np.pi, size=300))
        self.assertLess(abs(ppc), 0.05)

    def test_phase_locked_spikes_have_positive_ppc(self) -> None:
        ppc = pairwise_phase_consistency(np.zeros(60))
        self.assertGreater(ppc, 0.95)

    def test_artifact_contaminated_window_is_excluded(self) -> None:
        path = write_sine_csv(self.root / "artifact.csv", amp_uv=5000.0)
        records = [{"source_file": str(path), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "artifact", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out", clipped_abs_uv=1000.0))
        qc = pd.read_csv(paths["qc_csv"])
        self.assertTrue((~qc["qc_pass"].astype(bool)).any())
        self.assertTrue(qc["qc_reason"].astype(str).str.contains("clipping").any())
        self.assertIn("qc_failure_details", qc.columns)
        self.assertTrue(qc["qc_failure_details"].astype(str).str.contains("clipped_fraction").any())
        failures = pd.read_csv(paths["qc_failures_csv"])
        self.assertFalse(failures.empty)
        self.assertTrue(failures["qc_failure_details"].astype(str).str.contains("threshold").any())

    def test_filter_edge_regions_are_masked(self) -> None:
        path = write_sine_csv(self.root / "edge.csv", amp_uv=10.0, duration_s=10.0)
        records = [{"source_file": str(path), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "edge", "signal_unit": "uV"}]
        cfg = base_config(self.root / "out", analysis_window_s=2.0, filter_edge_s=1.0, minimum_valid_fraction=0.75)
        paths = process_lfp_recordings(records, cfg)
        qc = pd.read_csv(paths["qc_csv"])
        self.assertAlmostEqual(float(qc.iloc[0]["valid_fraction"]), 1.0, delta=1e-9)
        self.assertTrue(bool(qc.iloc[0]["qc_pass"]))
        self.assertGreater(float(qc.iloc[0]["segment_filter_edge_masked_s"]), 0.0)
        self.assertAlmostEqual(float(qc.iloc[0]["filter_edge_masked_fraction"]), 0.0, delta=1e-9)

    def test_clean_sixty_second_recording_uses_edge_excluded_interior(self) -> None:
        path = write_sine_csv(self.root / "sixty.csv", amp_uv=10.0, duration_s=60.0)
        records = [{"source_file": str(path), "epoch_label": "Baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "segment_id": "seg_base", "signal_unit": "uV"}]
        cfg = base_config(self.root / "out", analysis_window_s=60.0, filter_edge_s=None)
        paths = process_lfp_recordings(records, cfg)
        qc = pd.read_csv(paths["qc_csv"])
        self.assertEqual(len(qc), 1)
        self.assertTrue(bool(qc.iloc[0]["qc_pass"]))
        self.assertAlmostEqual(float(qc.iloc[0]["window_start_s"]), 6.0, delta=0.01)
        self.assertAlmostEqual(float(qc.iloc[0]["window_end_s"]), 54.0, delta=0.02)
        self.assertGreater(float(qc.iloc[0]["segment_filter_edge_masked_s"]), 11.5)

    def test_epoch_label_is_preserved_and_phase_is_normalized(self) -> None:
        path = write_sine_csv(self.root / "post1.csv", amp_uv=10.0)
        records = [{"source_file": str(path), "epoch_label": "Post 1", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "segment_id": "seg_post1", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out"))
        features = pd.read_csv(paths["features_csv"])
        self.assertEqual(set(features["epoch_label"].astype(str)), {"Post 1"})
        self.assertEqual(set(features["phase"].astype(str)), {"post"})
        self.assertEqual(set(features["segment_id"].astype(str)), {"seg_post1"})

    def test_baseline_normalization_is_matched_by_preparation_and_channel(self) -> None:
        base = write_multi_sine_csv(self.root / "base_multi.csv", {"A-001": 10.0, "A-002": 50.0})
        post = write_multi_sine_csv(self.root / "post_multi.csv", {"A-001": 20.0, "A-002": 50.0})
        records = [
            {"source_file": str(base), "epoch_label": "Baseline", "preparation_id": "prepA", "segment_id": "base", "recording_order": 1, "channel": "A-001,A-002", "signal_unit": "uV"},
            {"source_file": str(post), "epoch_label": "Treatment", "preparation_id": "prepA", "segment_id": "post", "recording_order": 2, "channel": "A-001,A-002", "signal_unit": "uV"},
        ]
        paths = process_lfp_recordings(records, base_config(self.root / "out"))
        features = pd.read_csv(paths["features_csv"])
        valid = features[features["qc_pass"].astype(bool)]
        treatment = valid[valid["phase"].astype(str).eq("treatment")]
        ch1 = treatment[treatment["channel"].astype(str).eq("A-001")]
        ch2 = treatment[treatment["channel"].astype(str).eq("A-002")]
        self.assertAlmostEqual(float(ch1["primary_power_uV2_change_db"].median()), 6.0206, delta=0.2)
        self.assertAlmostEqual(float(ch2["primary_power_uV2_change_db"].median()), 0.0, delta=0.2)

    def test_missing_channel_is_retained_as_availability_exclusion(self) -> None:
        base = write_multi_sine_csv(self.root / "base_missing.csv", {"A-001": 10.0, "A-002": 20.0})
        post = write_multi_sine_csv(self.root / "post_missing.csv", {"A-001": 20.0})
        records = [
            {"source_file": str(base), "epoch_label": "Baseline", "preparation_id": "prepA", "segment_id": "base", "recording_order": 1, "channel": "A-001,A-002", "signal_unit": "uV"},
            {"source_file": str(post), "epoch_label": "Post", "preparation_id": "prepA", "segment_id": "post", "recording_order": 2, "channel": "A-001,A-002", "signal_unit": "uV"},
        ]
        paths = process_lfp_recordings(records, base_config(self.root / "out"))
        features = pd.read_csv(paths["features_csv"])
        self.assertIn("A-002", set(features["channel"].astype(str)))
        self.assertFalse(((features["segment_id"].astype(str) == "post") & (features["channel"].astype(str) == "A-002")).any())
        availability = pd.read_csv(paths["matched_channel_availability_csv"])
        missing = availability[availability["channel"].astype(str).eq("A-002")]
        self.assertFalse(bool(missing.iloc[0]["paired_comparison_ready"]))
        self.assertIn("missing_valid_nonbaseline_epoch", str(missing.iloc[0]["exclusion_reason"]))

    def test_thirty_two_channel_preparation_preserves_channel_rows(self) -> None:
        channels = {f"A-{idx:03d}": 5.0 + idx for idx in range(1, 33)}
        path = write_multi_sine_csv(self.root / "thirty_two.csv", channels, duration_s=10.0)
        records = [{"source_file": str(path), "epoch_label": "Baseline", "preparation_id": "prep32", "segment_id": "seg32", "channel": ",".join(channels), "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out", analysis_window_s=10.0, filter_edge_s=0.0))
        features = pd.read_csv(paths["features_csv"])
        self.assertEqual(features["channel"].nunique(), 32)

    def test_nan_gap_does_not_create_derivative_artifact_at_gap_boundary(self) -> None:
        fs = 250.0
        t = np.arange(0, 20.0, 1.0 / fs)
        x = 10.0 * np.sin(2.0 * np.pi * 10.0 * t)
        x[(t >= 5.0) & (t < 5.5)] = np.nan
        path = self.root / "nan_gap.csv"
        pd.DataFrame({"time_s": t, "A-001": x}).to_csv(path, index=False)
        records = [{"source_file": str(path), "epoch_label": "Baseline", "preparation_id": "prepA", "segment_id": "gap", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out", analysis_window_s=20.0, filter_edge_s=0.0))
        qc = pd.read_csv(paths["qc_csv"])
        self.assertTrue(bool(qc.iloc[0]["qc_pass"]))
        self.assertEqual(int(qc.iloc[0]["large_derivative_count"]), 0)
        self.assertGreater(float(qc.iloc[0]["missing_masked_s"]), 0.0)

    def test_application_and_washout_onsets_do_not_mask_samples(self) -> None:
        path = write_sine_csv(self.root / "onsets.csv", amp_uv=10.0, duration_s=10.0)
        records = [
            {
                "source_file": str(path),
                "phase": "treatment",
                "experiment_id": "e1",
                "animal_id": "a1",
                "preparation_id": "p1",
                "recording_id": "onsets",
                "application_onset_s": 2.0,
                "washout_onset_s": 8.0,
                "signal_unit": "uV",
            }
        ]
        paths = process_lfp_recordings(records, base_config(self.root / "out", analysis_window_s=5.0, filter_edge_s=0.0))
        qc = pd.read_csv(paths["qc_csv"])
        self.assertTrue(np.allclose(qc["valid_fraction"], 1.0))
        self.assertTrue(qc["qc_pass"].astype(bool).all())

    def test_single_recoverable_step_is_masked_without_window_failure(self) -> None:
        fs = 250.0
        duration_s = 10.0
        t = np.arange(0, duration_s, 1.0 / fs)
        x = 10.0 * np.sin(2.0 * np.pi * 10.0 * t)
        x[int(5.0 * fs)] += 1000.0
        path = self.root / "recoverable_step.csv"
        pd.DataFrame({"time_s": t, "A-001": x}).to_csv(path, index=False)
        records = [{"source_file": str(path), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "recoverable", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out", analysis_window_s=10.0, filter_edge_s=0.0))
        qc = pd.read_csv(paths["qc_csv"])
        self.assertTrue(bool(qc.iloc[0]["qc_pass"]))
        self.assertGreater(float(qc.iloc[0]["large_derivative_fraction"]), 0.0)
        self.assertEqual(str(qc.iloc[0]["derivative_qc_status"]), "pass")
        self.assertEqual(int(qc.iloc[0]["large_derivative_count"]), 1)
        self.assertAlmostEqual(float(qc.iloc[0]["valid_fraction"]), 1.0, delta=1e-9)
        self.assertAlmostEqual(float(qc.iloc[0]["combined_masked_s"]), 0.0, delta=1e-9)
        derivative_events = pd.read_csv(paths["derivative_events_csv"])
        self.assertEqual(len(derivative_events), 1)
        features = pd.read_csv(paths["features_csv"])
        self.assertAlmostEqual(float(features.iloc[0]["dominant_peak_frequency_hz"]), 10.0, delta=0.5)

    def test_more_than_five_isolated_derivative_events_below_fraction_threshold_pass(self) -> None:
        fs = 250.0
        duration_s = 10.0
        t = np.arange(0, duration_s, 1.0 / fs)
        x = 10.0 * np.sin(2.0 * np.pi * 10.0 * t)
        for spike_s in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            x[int(spike_s * fs)] += 1000.0
        path = self.root / "repeated_steps_recoverable.csv"
        pd.DataFrame({"time_s": t, "A-001": x}).to_csv(path, index=False)
        records = [{"source_file": str(path), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "repeated", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out", analysis_window_s=10.0, filter_edge_s=0.0))
        qc = pd.read_csv(paths["qc_csv"])
        self.assertTrue(bool(qc.iloc[0]["qc_pass"]))
        self.assertTrue(bool(qc.iloc[0]["signal_qc_pass"]))
        self.assertTrue(bool(qc.iloc[0]["feature_qc_pass"]))
        self.assertEqual(str(qc.iloc[0]["qc_reason"]), "pass")
        self.assertGreaterEqual(int(qc.iloc[0]["large_derivative_count"]), 6)
        self.assertLess(float(qc.iloc[0]["large_derivative_fraction"]), 0.02)
        self.assertAlmostEqual(float(qc.iloc[0]["valid_fraction"]), 1.0, delta=1e-9)
        self.assertAlmostEqual(float(qc.iloc[0]["filtered_finite_sample_fraction"]), 1.0, delta=1e-9)
        self.assertIn("large_derivative_event_rate_per_min", qc.columns)
        derivative_events = pd.read_csv(paths["derivative_events_csv"])
        self.assertEqual(len(derivative_events), int(qc.iloc[0]["large_derivative_count"]))
        params = json.loads(Path(paths["parameters_json"]).read_text(encoding="utf-8"))
        self.assertFalse(any("too short" in warning.lower() for warning in params["warnings"]))

    def test_derivative_warning_fraction_does_not_fail_or_mask_signal(self) -> None:
        fs = 250.0
        duration_s = 10.0
        t = np.arange(0, duration_s, 1.0 / fs)
        x = 10.0 * np.sin(2.0 * np.pi * 10.0 * t)
        for spike_s in np.arange(1.0, 7.5, 0.5):
            x[int(spike_s * fs)] += 1000.0
        path = self.root / "warning_steps.csv"
        pd.DataFrame({"time_s": t, "A-001": x}).to_csv(path, index=False)
        records = [{"source_file": str(path), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "warning", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out", analysis_window_s=10.0, filter_edge_s=0.0))
        qc = pd.read_csv(paths["qc_csv"])
        self.assertTrue(bool(qc.iloc[0]["qc_pass"]))
        self.assertEqual(str(qc.iloc[0]["derivative_qc_status"]), "warning")
        self.assertGreaterEqual(float(qc.iloc[0]["large_derivative_fraction"]), 0.005)
        self.assertLessEqual(float(qc.iloc[0]["large_derivative_fraction"]), 0.02)
        self.assertAlmostEqual(float(qc.iloc[0]["combined_masked_s"]), 0.0, delta=1e-9)
        self.assertGreater(float(qc.iloc[0]["primary_power_finite_positive"]), 0)

    def test_derivative_fraction_above_threshold_fails_with_burden_reason(self) -> None:
        fs = 250.0
        duration_s = 10.0
        t = np.arange(0, duration_s, 1.0 / fs)
        x = 10.0 * np.sin(2.0 * np.pi * 10.0 * t)
        for spike_s in np.arange(0.25, 9.25, 0.15):
            x[int(spike_s * fs)] += 1000.0
        path = self.root / "many_steps.csv"
        pd.DataFrame({"time_s": t, "A-001": x}).to_csv(path, index=False)
        records = [{"source_file": str(path), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "many", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out", analysis_window_s=10.0, filter_edge_s=0.0))
        qc = pd.read_csv(paths["qc_csv"])
        self.assertFalse(bool(qc.iloc[0]["qc_pass"]))
        self.assertFalse(bool(qc.iloc[0]["signal_qc_pass"]))
        self.assertTrue(bool(qc.iloc[0]["feature_qc_pass"]))
        self.assertEqual(str(qc.iloc[0]["derivative_qc_status"]), "fail")
        self.assertIn("large_derivative_artifact_burden", str(qc.iloc[0]["qc_reason"]))
        self.assertGreater(float(qc.iloc[0]["large_derivative_fraction"]), 0.02)
        self.assertNotIn("large_derivative_or_step", str(qc.iloc[0]["qc_reason"]))
        self.assertIn("large_derivative_fraction", str(qc.iloc[0]["qc_failure_details"]))
        self.assertIn("large_derivative_count", str(qc.iloc[0]["qc_failure_details"]))

    def test_frequent_spike_like_derivatives_still_produce_nonzero_lfp_and_valid_psd(self) -> None:
        fs = 250.0
        duration_s = 10.0
        t = np.arange(0, duration_s, 1.0 / fs)
        x = 12.0 * np.sin(2.0 * np.pi * 8.0 * t)
        for spike_s in np.arange(1.0, 7.5, 0.5):
            x[int(spike_s * fs)] += 800.0
        path = self.root / "spike_like_lfp.csv"
        pd.DataFrame({"time_s": t, "A-001": x}).to_csv(path, index=False)
        records = [{"source_file": str(path), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "spike_like", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out", analysis_window_s=10.0, filter_edge_s=0.0))
        qc = pd.read_csv(paths["qc_csv"])
        features = pd.read_csv(paths["features_csv"])
        self.assertTrue(bool(qc.iloc[0]["feature_qc_pass"]))
        self.assertGreater(float(qc.iloc[0]["filtered_rms_uv"]), 0.0)
        self.assertGreater(int(qc.iloc[0]["finite_psd_bin_count"]), 2)
        self.assertGreater(float(features.iloc[0]["primary_power_uV2"]), 0.0)

    def test_time_discontinuity_does_not_create_derivative_artifact_at_boundary(self) -> None:
        fs = 250.0
        t = np.arange(0, 10.0, 1.0 / fs)
        t[int(5.0 * fs) :] += 1.0
        x = 10.0 * np.sin(2.0 * np.pi * 10.0 * t)
        x[int(5.0 * fs) :] += 1000.0
        path = self.root / "time_gap.csv"
        pd.DataFrame({"time_s": t, "A-001": x}).to_csv(path, index=False)
        records = [{"source_file": str(path), "epoch_label": "Baseline", "preparation_id": "prepA", "segment_id": "time_gap", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out", analysis_window_s=11.0, filter_edge_s=0.0, minimum_valid_fraction=0.75))
        qc = pd.read_csv(paths["qc_csv"])
        self.assertEqual(int(qc.iloc[0]["large_derivative_count"]), 0)
        self.assertGreater(float(qc.iloc[0]["time_discontinuity_masked_s"]), 0.0)

    def test_manual_artifacts_remain_hard_masks(self) -> None:
        path = write_sine_csv(self.root / "manual_artifact.csv", amp_uv=10.0, duration_s=10.0)
        records = [{"source_file": str(path), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "manual", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out", analysis_window_s=10.0, filter_edge_s=0.0, manual_artifact_intervals=[(2.0, 3.0)]))
        qc = pd.read_csv(paths["qc_csv"])
        self.assertGreater(float(qc.iloc[0]["manual_artifact_masked_s"]), 0.9)
        self.assertLess(float(qc.iloc[0]["valid_fraction"]), 1.0)
        self.assertGreater(float(qc.iloc[0]["combined_masked_s"]), 0.9)

    def test_short_repeated_samples_do_not_trigger_flatline_mask(self) -> None:
        fs = 250.0
        t = np.arange(0, 10.0, 1.0 / fs)
        x = 10.0 * np.sin(2.0 * np.pi * 10.0 * t)
        x[500:505] = x[500]
        path = self.root / "short_repeated.csv"
        pd.DataFrame({"time_s": t, "A-001": x}).to_csv(path, index=False)
        records = [{"source_file": str(path), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "short_repeat", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out", analysis_window_s=10.0, filter_edge_s=0.0))
        qc = pd.read_csv(paths["qc_csv"])
        self.assertEqual(float(qc.iloc[0]["flatline_fraction"]), 0.0)
        self.assertTrue(bool(qc.iloc[0]["qc_pass"]))

    def test_sustained_flatline_is_detected(self) -> None:
        fs = 250.0
        t = np.arange(0, 10.0, 1.0 / fs)
        x = 10.0 * np.sin(2.0 * np.pi * 10.0 * t)
        x[500:530] = x[500]
        path = self.root / "sustained_flatline.csv"
        pd.DataFrame({"time_s": t, "A-001": x}).to_csv(path, index=False)
        records = [{"source_file": str(path), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "flat", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out", analysis_window_s=10.0, filter_edge_s=0.0))
        qc = pd.read_csv(paths["qc_csv"])
        self.assertGreater(float(qc.iloc[0]["flatline_fraction"]), 0.0)
        self.assertGreater(float(qc.iloc[0]["flatline_masked_s"]), 0.05)

    def test_zero_filtered_signal_fails_feature_qc(self) -> None:
        fs = 250.0
        t = np.arange(0, 10.0, 1.0 / fs)
        path = self.root / "zeros.csv"
        pd.DataFrame({"time_s": t, "A-001": np.zeros_like(t)}).to_csv(path, index=False)
        records = [{"source_file": str(path), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "zeros", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out", analysis_window_s=10.0, filter_edge_s=0.0))
        qc = pd.read_csv(paths["qc_csv"])
        self.assertFalse(bool(qc.iloc[0]["feature_qc_pass"]))
        self.assertIn("filtered_signal", str(qc.iloc[0]["feature_qc_reason"]))
        features = pd.read_csv(paths["features_csv"])
        self.assertEqual(str(features.iloc[0]["event_qc"]), "not_run_qc_failed")

    def test_invalid_primary_band_psd_fails_feature_qc(self) -> None:
        path = write_sine_csv(self.root / "bad_primary_band.csv", amp_uv=10.0, duration_s=10.0)
        records = [{"source_file": str(path), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "bad_band", "signal_unit": "uV"}]
        cfg = base_config(self.root / "out", analysis_window_s=10.0, filter_edge_s=0.0, exploratory_band_hz=(0.5, 40.0), primary_band_hz=(80.0, 90.0))
        paths = process_lfp_recordings(records, cfg)
        qc = pd.read_csv(paths["qc_csv"])
        self.assertFalse(bool(qc.iloc[0]["feature_qc_pass"]))
        self.assertIn("insufficient_psd_bins_for_primary_power", str(qc.iloc[0]["feature_qc_reason"]))

    def test_peak_detection_failure_does_not_invalidate_broadband_power(self) -> None:
        path = write_sine_csv(self.root / "no_peak_required.csv", amp_uv=10.0, duration_s=10.0)
        records = [{"source_file": str(path), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "no_peak", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out", analysis_window_s=10.0, filter_edge_s=0.0, peak_prominence_db=1000.0))
        features = pd.read_csv(paths["features_csv"])
        self.assertTrue(bool(features.iloc[0]["qc_pass"]))
        self.assertIn("no_peak", str(features.iloc[0]["peak_qc"]))
        self.assertGreater(float(features.iloc[0]["primary_power_uV2"]), 0.0)

    def test_event_extraction_skips_nonpositive_threshold(self) -> None:
        fs = 250.0
        t = np.arange(0, 10.0, 1.0 / fs)
        x = 10.0 * np.sin(2.0 * np.pi * 10.0 * t)
        cfg = base_config(self.root / "out", event_envelope_mad_threshold=-1e9)
        summary, events = _event_features(x, t, fs, cfg, {"channel": "A-001"})
        self.assertEqual(summary["event_qc"], "event_threshold_zero_or_not_finite")
        self.assertEqual(events, [])

    def test_combined_mask_accounting_uses_only_hard_masks(self) -> None:
        path = write_sine_csv(self.root / "mask_budget.csv", amp_uv=10.0, duration_s=10.0)
        records = [{"source_file": str(path), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "budget", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out", analysis_window_s=10.0, filter_edge_s=0.0, manual_artifact_intervals=[(2.0, 3.0)]))
        qc = pd.read_csv(paths["qc_csv"])
        self.assertLessEqual(float(qc.iloc[0]["combined_masked_s"]), float(qc.iloc[0]["hard_mask_sum_s"]) + 1e-9)
        self.assertGreater(float(qc.iloc[0]["large_derivative_candidate_fraction"]), -1e-9)

    def test_derivative_events_match_spikes_only_within_same_segment_and_channel(self) -> None:
        fs = 250.0
        duration_s = 10.0
        t = np.arange(0, duration_s, 1.0 / fs)
        records = []
        for segment in ("segA", "segB"):
            x = 10.0 * np.sin(2.0 * np.pi * 10.0 * t)
            x[int(1.0 * fs)] += 1000.0
            path = self.root / f"{segment}.csv"
            pd.DataFrame({"time_s": t, "A-001": x, "A-002": x}).to_csv(path, index=False)
            records.append(
                {
                    "source_file": str(path),
                    "phase": "baseline",
                    "experiment_id": "e1",
                    "animal_id": "a1",
                    "preparation_id": "prepA",
                    "segment_id": segment,
                    "recording_id": segment,
                    "channel": "A-001",
                    "signal_unit": "uV",
                }
            )
        spike_events_csv = self.root / "spike_events.csv"
        pd.DataFrame(
            [
                {"preparation_id": "prepA", "segment_id": "segA", "channel": "A-001", "spike_time_s": 1.0},
                {"preparation_id": "prepA", "segment_id": "segB", "channel": "A-002", "spike_time_s": 1.0},
            ]
        ).to_csv(spike_events_csv, index=False)
        paths = process_lfp_recordings(records, base_config(self.root / "out", analysis_window_s=10.0, filter_edge_s=0.0, spike_events_csv=spike_events_csv))
        derivative_events = pd.read_csv(paths["derivative_events_csv"])
        seg_a = derivative_events[derivative_events["segment_id"].astype(str).eq("segA")]
        seg_b = derivative_events[derivative_events["segment_id"].astype(str).eq("segB")]
        self.assertTrue(seg_a["matched_to_spike_within_tolerance"].astype(bool).any())
        self.assertFalse(seg_b["matched_to_spike_within_tolerance"].astype(bool).any())

    def test_derivative_sensitivity_report_includes_thresholds_and_masking_disabled(self) -> None:
        fs = 250.0
        t = np.arange(0, 10.0, 1.0 / fs)
        x = 10.0 * np.sin(2.0 * np.pi * 10.0 * t)
        for spike_s in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            x[int(spike_s * fs)] += 1000.0
        path = self.root / "sensitivity.csv"
        pd.DataFrame({"time_s": t, "A-001": x}).to_csv(path, index=False)
        records = [{"source_file": str(path), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "sensitivity", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out", analysis_window_s=10.0, filter_edge_s=0.0))
        sensitivity = pd.read_csv(paths["derivative_sensitivity_csv"])
        self.assertEqual(set(sensitivity["large_derivative_fraction_threshold_candidate"].dropna().round(3)), {0.005, 0.01, 0.02, 0.05})
        self.assertIn("derivative_masking_disabled", set(sensitivity["scenario"].astype(str)))

    def test_diagnostic_plots_are_generated_even_when_all_feature_windows_fail(self) -> None:
        fs = 250.0
        t = np.arange(0, 10.0, 1.0 / fs)
        x = 10.0 * np.sin(2.0 * np.pi * 10.0 * t)
        for spike_s in np.arange(0.25, 9.25, 0.15):
            x[int(spike_s * fs)] += 1000.0
        path = self.root / "all_fail.csv"
        pd.DataFrame({"time_s": t, "A-001": x}).to_csv(path, index=False)
        records = [{"source_file": str(path), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "segment_id": "all_fail", "recording_id": "all_fail", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out", analysis_window_s=10.0, filter_edge_s=0.0))
        features = pd.read_csv(paths["features_csv"])
        self.assertFalse(features["qc_pass"].astype(bool).any())
        plot_outputs = plot_lfp_features(paths["features_csv"], {"output_dir": self.root / "plots"})
        self.assertTrue(any(Path(path).name == "lfp_qc_overview.png" for path in plot_outputs["plots"]))
        self.assertTrue(any("derivative_qc" in path for path in plot_outputs["plots"]))
        self.assertTrue(any(Path(path).name == "lfp_feature_plots_skipped_all_windows_failed_qc.txt" for path in plot_outputs["tables"]))

    def test_older_all_fail_qc_files_remain_readable_for_diagnostic_plotting(self) -> None:
        out_dir = self.root / "old_qc"
        out_dir.mkdir()
        feature_csv = out_dir / "lfp_features_by_window.csv"
        pd.DataFrame(
            [
                {
                    "preparation_id": "prepA",
                    "segment_id": "seg1",
                    "channel": "A-001",
                    "phase": "baseline",
                    "qc_pass": False,
                    "qc_reason": "large_derivative_or_step",
                    "cumulative_window_mid_s": 30.0,
                    "cumulative_window_start_s": 0.0,
                    "cumulative_window_end_s": 60.0,
                }
            ]
        ).to_csv(feature_csv, index=False)
        pd.DataFrame(
            [
                {
                    "preparation_id": "prepA",
                    "segment_id": "seg1",
                    "channel": "A-001",
                    "phase": "baseline",
                    "qc_pass": False,
                    "qc_reason": "large_derivative_or_step",
                    "valid_fraction": 0.98,
                    "clipped_fraction": 0.0,
                    "flatline_fraction": 0.0,
                    "line_noise_ratio": 0.01,
                    "large_derivative_or_step_score": 12.0,
                    "cumulative_window_mid_s": 30.0,
                    "cumulative_window_start_s": 0.0,
                    "cumulative_window_end_s": 60.0,
                }
            ]
        ).to_csv(out_dir / "lfp_qc_by_window.csv", index=False)
        plot_outputs = plot_lfp_features(feature_csv, {"output_dir": out_dir / "plots"})
        self.assertTrue(any(Path(path).name == "lfp_qc_overview.png" for path in plot_outputs["plots"]))
        self.assertTrue(any(Path(path).name == "lfp_feature_plots_skipped_all_windows_failed_qc.txt" for path in plot_outputs["tables"]))

    def test_plotting_uses_signal_and_feature_qc_for_new_files(self) -> None:
        out_dir = self.root / "new_qc_plot"
        out_dir.mkdir()
        feature_csv = out_dir / "lfp_features_by_window.csv"
        pd.DataFrame(
            [
                {
                    "preparation_id": "prepA",
                    "segment_id": "seg1",
                    "channel": "A-001",
                    "phase": "baseline",
                    "epoch_label": "Baseline",
                    "qc_pass": True,
                    "signal_qc_pass": True,
                    "feature_qc_pass": False,
                    "feature_qc_reason": "invalid_psd",
                    "primary_power_uV2": 100.0,
                    "cumulative_window_mid_s": 30.0,
                    "cumulative_window_start_s": 0.0,
                    "cumulative_window_end_s": 60.0,
                    "window_index": 0,
                }
            ]
        ).to_csv(feature_csv, index=False)
        pd.DataFrame(
            [
                {
                    "preparation_id": "prepA",
                    "segment_id": "seg1",
                    "channel": "A-001",
                    "phase": "baseline",
                    "qc_pass": True,
                    "signal_qc_pass": True,
                    "feature_qc_pass": False,
                    "feature_qc_reason": "invalid_psd",
                    "valid_fraction": 1.0,
                    "filtered_finite_sample_fraction": 0.0,
                    "large_derivative_fraction": 0.0,
                    "cumulative_window_mid_s": 30.0,
                    "cumulative_window_start_s": 0.0,
                    "cumulative_window_end_s": 60.0,
                }
            ]
        ).to_csv(out_dir / "lfp_qc_by_window.csv", index=False)
        plot_outputs = plot_lfp_features(feature_csv, {"output_dir": out_dir / "plots"})
        self.assertTrue(any(Path(path).name == "lfp_feature_plots_skipped_all_windows_failed_qc.txt" for path in plot_outputs["tables"]))

    def test_older_qc_pass_only_files_still_plot_when_valid(self) -> None:
        out_dir = self.root / "old_qc_pass"
        out_dir.mkdir()
        feature_csv = out_dir / "lfp_features_by_window.csv"
        pd.DataFrame(
            [
                {
                    "preparation_id": "prepA",
                    "segment_id": "seg1",
                    "channel": "A-001",
                    "phase": "baseline",
                    "epoch_label": "Baseline",
                    "qc_pass": True,
                    "primary_power_uV2": 100.0,
                    "total_power_uV2": 120.0,
                    "lfp_rms_uv": 8.0,
                    "cumulative_window_mid_s": 30.0,
                    "cumulative_window_start_s": 0.0,
                    "cumulative_window_end_s": 60.0,
                    "window_start_s": 0.0,
                    "window_end_s": 60.0,
                }
            ]
        ).to_csv(feature_csv, index=False)
        plot_outputs = plot_lfp_features(feature_csv, {"output_dir": out_dir / "plots"})
        self.assertTrue(any(Path(path).name == "lfp_feature_vs_experiment_time.png" for path in plot_outputs["plots"]))

    def test_streaming_and_nonstreaming_results_agree(self) -> None:
        path = write_sine_csv(self.root / "stream.csv", amp_uv=12.0)
        records = [{"source_file": str(path), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "stream", "signal_unit": "uV"}]
        p1 = process_lfp_recordings(records, base_config(self.root / "out1", streaming=False))
        p2 = process_lfp_recordings(records, base_config(self.root / "out2", streaming=True))
        f1 = pd.read_csv(p1["features_csv"])
        f2 = pd.read_csv(p2["features_csv"])
        self.assertTrue(np.allclose(f1["primary_power_uV2"], f2["primary_power_uV2"], rtol=1e-6, atol=1e-6, equal_nan=True))

    def test_baseline_uses_all_explicit_valid_baseline_windows(self) -> None:
        base1 = write_sine_csv(self.root / "base1.csv", amp_uv=10.0)
        base2 = write_sine_csv(self.root / "base2.csv", amp_uv=14.0)
        post = write_sine_csv(self.root / "post.csv", amp_uv=20.0)
        records = [
            {"source_file": str(base1), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "base1", "recording_order": 1, "signal_unit": "uV"},
            {"source_file": str(base2), "phase": "baseline", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "base2", "recording_order": 2, "signal_unit": "uV"},
            {"source_file": str(post), "phase": "post", "experiment_id": "e1", "animal_id": "a1", "preparation_id": "p1", "recording_id": "post", "recording_order": 3, "signal_unit": "uV"},
        ]
        paths = process_lfp_recordings(records, base_config(self.root / "out"))
        features = pd.read_csv(paths["features_csv"])
        post_rows = features[(features["phase"] == "post") & (features["qc_pass"].astype(bool))]
        baseline_ref = float(post_rows["baseline_median_primary_power_uV2"].median())
        self.assertAlmostEqual(baseline_ref, 74.0, delta=3.0)

    def test_missing_biological_metadata_produces_warning(self) -> None:
        path = write_sine_csv(self.root / "missing.csv")
        records = [{"source_file": str(path), "phase": "baseline", "recording_id": "missing", "signal_unit": "uV"}]
        paths = process_lfp_recordings(records, base_config(self.root / "out"))
        params = json.loads(Path(paths["parameters_json"]).read_text(encoding="utf-8"))
        self.assertTrue(any("animal_id" in warning or "preparation_id" in warning for warning in params["warnings"]))

    def test_existing_spike_count_output_is_unchanged_by_default(self) -> None:
        module_path = ROOT / "Spike Processing" / "Spike Count Multiple CSVs ordered.py"
        spec = importlib.util.spec_from_file_location("spike_ordered_test", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        fs = 20000.0
        t = np.arange(0, 2.0, 1.0 / fs)
        raw = np.zeros_like(t)
        csv_path = self.root / "spikes.csv"
        pd.DataFrame({"time_s": t, "A-001": raw}).to_csv(csv_path, index=False)
        out = module.process_csvs([str(csv_path)], out_dir=str(self.root / "spike_out"), window_sec=1.0, polarity="both")
        df = pd.read_csv(out)
        self.assertIn("spike_count", df.columns)
        self.assertNotIn("spike_time_s", df.columns)
        self.assertFalse((self.root / "spike_out" / "spike_events.csv").exists())


if __name__ == "__main__":
    unittest.main()
