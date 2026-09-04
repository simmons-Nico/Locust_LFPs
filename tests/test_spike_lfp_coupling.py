from __future__ import annotations

import json
import sys
import tempfile
import threading
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

from locust_lfp_gui import SpikePipelineRequest, run_shared_spike_pipeline
from spike_lfp_coupling import (
    SpikeLFPCouplingConfig,
    _channel_pairs,
    _continuous_runs,
    calculate_spike_lfp_coupling,
    preflight_coupling_workload,
)


def write_raw(path: Path, channels: list[str], start_s: float = 0.0, duration_s: float = 2.0, fs: float = 500.0) -> Path:
    t = start_s + np.arange(0, duration_s, 1.0 / fs)
    data = {"time_s": t}
    for idx, channel in enumerate(channels):
        data[channel] = 25.0 * np.sin(2.0 * np.pi * (8.0 + idx) * (t - start_s))
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def write_spike_events(path: Path, channels: list[str], segment_id: str, start_s: float = 0.0, preparation_id: str = "prep1", phase: str = "baseline") -> Path:
    rows = []
    for channel in channels:
        for idx, spike_time in enumerate(np.arange(start_s + 0.15, start_s + 1.85, 0.2), start=1):
            rows.append(
                {
                    "preparation_id": preparation_id,
                    "segment_id": segment_id,
                    "recording_id": segment_id,
                    "recording_name": segment_id,
                    "epoch_label": phase.title(),
                    "phase": phase,
                    "channel": channel,
                    "event_index": idx,
                    "spike_time_s": spike_time,
                    "source_file": str(path),
                    "qc_pass": True,
                }
            )
    out = path.with_name(f"{segment_id}_spike_events.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def feature_rows(raw_path: Path, channels: list[str], segment_id: str, phase: str = "baseline", pass_second: bool = True) -> pd.DataFrame:
    rows = []
    for channel in channels:
        for idx, (start, end, ok) in enumerate([(0.0, 1.0, True), (1.0, 2.0, pass_second)]):
            rows.append(
                {
                    "preparation_id": "prep1",
                    "segment_id": segment_id,
                    "recording_id": segment_id,
                    "source_file": str(raw_path),
                    "epoch_label": phase.title(),
                    "phase": phase,
                    "channel": channel,
                    "window_index": idx,
                    "window_start_s": start,
                    "window_end_s": end,
                    "window_mid_s": (start + end) / 2.0,
                    "cumulative_window_mid_s": (start + end) / 2.0,
                    "window_duration_s": end - start,
                    "usable_duration_s": end - start,
                    "primary_power_uV2": 10.0 + idx,
                    "signal_qc_pass": ok,
                    "feature_qc_pass": ok,
                    "qc_pass": ok,
                }
            )
    return pd.DataFrame(rows)


class SpikeLFPCouplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_shared_spike_pipeline_regenerates_events_and_provenance(self) -> None:
        fs = 20_000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        raw = np.zeros_like(t)
        for spike_time in (0.1, 0.25, 0.5, 0.75):
            raw[int(spike_time * fs)] = -250.0
        csv_path = self.root / "spikes.csv"
        pd.DataFrame({"time_s": t, "A-001": raw}).to_csv(csv_path, index=False)
        out_dir = self.root / "spike_out"
        stale_events = out_dir / "spike_events.csv"
        out_dir.mkdir()
        stale_events.write_text("stale\n", encoding="utf-8")
        request = SpikePipelineRequest(
            paths=[str(csv_path)],
            labels_by_path={str(csv_path): "Baseline"},
            out_dir=str(out_dir),
            window_sec=1.0,
            polarity="both",
            export_spike_events=True,
            spike_events_path=str(stale_events),
            recording_metadata=[
                {
                    "source_file": str(csv_path),
                    "segment_id": "spikes",
                    "recording_id": "spikes",
                    "epoch_label": "Baseline",
                    "preparation_id": "prep1",
                    "signal_unit": "uV",
                }
            ],
        )
        first = run_shared_spike_pipeline(request)
        second = run_shared_spike_pipeline(request)
        a = pd.read_csv(first["spike_events_csv"])
        b = pd.read_csv(second["spike_events_csv"])
        pd.testing.assert_frame_equal(a, b)
        provenance = json.loads(Path(first["spike_processing_provenance_json"]).read_text(encoding="utf-8"))
        self.assertEqual(provenance["pipeline"], "Spike Count Multiple CSVs ordered.py::process_csvs")
        self.assertIn("script_sha256", provenance)
        waveform_summary = pd.read_csv(first["spike_waveform_summary_csv"])
        self.assertFalse(waveform_summary.empty)
        self.assertTrue(Path(waveform_summary.iloc[0]["waveform_plot"]).exists())
        diagnostic_outputs = pd.read_csv(first["spike_diagnostic_outputs_csv"])
        self.assertIn("negative_average_waveform_plot", set(diagnostic_outputs["output_type"]))

    def test_pairing_modes_same_channel_and_all_to_all(self) -> None:
        cfg = SpikeLFPCouplingConfig(pairing_mode="same_channel")
        self.assertEqual(len(_channel_pairs([f"A-{i:03d}" for i in range(16)], [f"A-{i:03d}" for i in range(16)], cfg)), 16)
        self.assertEqual(len(_channel_pairs([f"A-{i:03d}" for i in range(32)], [f"A-{i:03d}" for i in range(32)], cfg)), 32)
        all_cfg = SpikeLFPCouplingConfig(pairing_mode="all_to_all")
        self.assertEqual(len(_channel_pairs(["A", "B"], ["A", "B", "C"], all_cfg)), 6)

    def test_preflight_rejects_large_all_to_all_without_override(self) -> None:
        raw = write_raw(self.root / "raw.csv", ["A", "B"])
        spikes = write_spike_events(raw, ["A", "B"], "raw")
        records = [{"source_file": str(raw), "segment_id": "raw", "recording_id": "raw", "preparation_id": "prep1", "phase": "baseline", "signal_unit": "uV"}]
        cfg = SpikeLFPCouplingConfig(
            output_dir=self.root / "out",
            pairing_mode="all_to_all",
            phase_bands=[("theta", "4-12 Hz", (4.0, 12.0))],
            window_s=1.0,
            surrogate_count=10,
            max_projected_ppc=1,
        )
        workload = preflight_coupling_workload(spikes, records, cfg)
        self.assertEqual(workload["channel_pairs"], 4)
        with self.assertRaises(RuntimeError):
            calculate_spike_lfp_coupling(spikes, records, cfg)

    def test_lfp_qc_excludes_failed_windows_and_relative_time_matches(self) -> None:
        raw = write_raw(self.root / "offset.csv", ["A-001"], start_s=100.0)
        spikes = write_spike_events(raw, ["A-001"], "offset", start_s=100.0)
        features = feature_rows(raw, ["A-001"], "offset", pass_second=False)
        feature_csv = self.root / "features.csv"
        features.to_csv(feature_csv, index=False)
        records = [{"source_file": str(raw), "segment_id": "offset", "recording_id": "offset", "preparation_id": "prep1", "phase": "baseline", "epoch_label": "Baseline", "signal_unit": "uV"}]
        cfg = SpikeLFPCouplingConfig(
            output_dir=self.root / "out",
            lfp_feature_csv=feature_csv,
            phase_bands=[("theta", "4-12 Hz", (4.0, 12.0))],
            window_s=1.0,
            min_spikes=2,
            surrogate_count=3,
            notch_frequency_hz=None,
        )
        paths = calculate_spike_lfp_coupling(spikes, records, cfg)
        ppc = pd.read_csv(paths["ppc_csv"])
        self.assertEqual(ppc["window_index"].nunique(), 1)
        self.assertGreater(int(ppc["spike_count"].max()), 0)
        self.assertTrue(ppc["ppc"].between(-1.0, 1.0).all())

    def test_no_interpolation_across_nan_gap_and_cancellation(self) -> None:
        t = np.arange(0, 2.0, 1 / 500.0)
        x = np.sin(2 * np.pi * 8 * t)
        x[(t > 0.8) & (t < 1.2)] = np.nan
        runs = _continuous_runs(t, x)
        self.assertGreaterEqual(len(runs), 2)
        raw = write_raw(self.root / "cancel.csv", ["A-001"])
        spikes = write_spike_events(raw, ["A-001"], "cancel")
        records = [{"source_file": str(raw), "segment_id": "cancel", "recording_id": "cancel", "preparation_id": "prep1", "phase": "baseline", "signal_unit": "uV"}]
        event = threading.Event()
        event.set()
        with self.assertRaises(RuntimeError):
            calculate_spike_lfp_coupling(spikes, records, SpikeLFPCouplingConfig(output_dir=self.root / "cancel_out"), cancel_event=event)

    def test_surrogates_reproducible_sfc_bounded_and_baseline_change(self) -> None:
        base = write_raw(self.root / "base.csv", ["A-001"])
        post = write_raw(self.root / "post.csv", ["A-001"])
        spike_frames = []
        for path, segment, phase in ((base, "base", "baseline"), (post, "post", "post")):
            spike_frames.append(pd.read_csv(write_spike_events(path, ["A-001"], segment, phase=phase)))
        spikes = self.root / "events.csv"
        pd.concat(spike_frames, ignore_index=True).to_csv(spikes, index=False)
        features = pd.concat(
            [feature_rows(base, ["A-001"], "base", "baseline"), feature_rows(post, ["A-001"], "post", "post")],
            ignore_index=True,
        )
        feature_csv = self.root / "features.csv"
        features.to_csv(feature_csv, index=False)
        records = [
            {"source_file": str(base), "segment_id": "base", "recording_id": "base", "recording_order": 1, "preparation_id": "prep1", "phase": "baseline", "epoch_label": "Baseline", "signal_unit": "uV"},
            {"source_file": str(post), "segment_id": "post", "recording_id": "post", "recording_order": 2, "preparation_id": "prep1", "phase": "post", "epoch_label": "Post", "signal_unit": "uV"},
        ]
        cfg = SpikeLFPCouplingConfig(
            output_dir=self.root / "out1",
            lfp_feature_csv=feature_csv,
            phase_bands=[("theta", "4-12 Hz", (4.0, 12.0))],
            window_s=1.0,
            min_spikes=2,
            surrogate_count=5,
            random_seed=99,
            max_sfc_frequencies=5,
            notch_frequency_hz=None,
        )
        first = calculate_spike_lfp_coupling(spikes, records, cfg)
        cfg.output_dir = self.root / "out2"
        second = calculate_spike_lfp_coupling(spikes, records, cfg)
        ppc1 = pd.read_csv(first["ppc_csv"])
        ppc2 = pd.read_csv(second["ppc_csv"])
        pd.testing.assert_series_equal(ppc1["surrogate_mean"], ppc2["surrogate_mean"], check_names=False)
        sfc = pd.read_csv(first["sfc_csv"])
        self.assertLessEqual(sfc.groupby(["segment_id", "window_index"]).size().max(), 5)
        self.assertTrue(ppc1.loc[ppc1["phase"] == "post", "ppc_change_from_baseline"].notna().any())
        sta = pd.read_csv(first["sta_summary_csv"])
        self.assertFalse(sta.empty)
        self.assertTrue(Path(sta.iloc[0]["sta_plot"]).exists())

    def test_simple_mode_outputs_rate_power_and_ppc_without_complex_plots(self) -> None:
        raw = write_raw(self.root / "simple.csv", ["A-001"], duration_s=4.0)
        spikes = write_spike_events(raw, ["A-001"], "simple")
        features = []
        for idx in range(4):
            features.append(
                {
                    "preparation_id": "prep1",
                    "segment_id": "simple",
                    "recording_id": "simple",
                    "source_file": str(raw),
                    "epoch_label": "Baseline" if idx < 2 else "Post",
                    "phase": "baseline" if idx < 2 else "post",
                    "channel": "A-001",
                    "window_index": idx,
                    "window_start_s": float(idx),
                    "window_end_s": float(idx + 1),
                    "window_mid_s": float(idx) + 0.5,
                    "cumulative_window_mid_s": float(idx) + 0.5,
                    "usable_duration_s": 1.0,
                    "primary_power_uV2": 10.0 + 5.0 * idx,
                    "signal_qc_pass": True,
                    "feature_qc_pass": True,
                    "qc_pass": True,
                }
            )
        feature_csv = self.root / "simple_features.csv"
        pd.DataFrame(features).to_csv(feature_csv, index=False)
        records = [
            {
                "source_file": str(raw),
                "segment_id": "simple",
                "recording_id": "simple",
                "recording_order": 1,
                "preparation_id": "prep1",
                "phase": "baseline",
                "epoch_label": "Baseline",
                "signal_unit": "uV",
            }
        ]
        cfg = SpikeLFPCouplingConfig(
            output_dir=self.root / "simple_out",
            lfp_feature_csv=feature_csv,
            processing_mode="simple",
            phase_bands=[("theta", "4-12 Hz", (4.0, 12.0))],
            window_s=1.0,
            min_spikes=2,
            surrogate_count=3,
            max_sfc_frequencies=5,
            notch_frequency_hz=None,
        )
        workload = preflight_coupling_workload(spikes, records, cfg)
        self.assertEqual(workload["processing_mode"], "simple")
        self.assertEqual(workload["estimated_sfc_rows"], 0)
        paths = calculate_spike_lfp_coupling(spikes, records, cfg)
        self.assertTrue(Path(paths["primary_rate_power_plot"]).exists())
        self.assertTrue(Path(paths["primary_ppc_plot"]).exists())
        rate_power = pd.read_csv(paths["rate_power_csv"])
        self.assertEqual(len(rate_power), 4)
        self.assertIn("spike_rate_hz", rate_power.columns)
        rate_summary = pd.read_csv(paths["rate_power_summary_csv"])
        self.assertIn("spearman_rho_s", rate_summary.columns)
        sfc = pd.read_csv(paths["sfc_csv"])
        self.assertTrue(sfc.empty)
        sta = pd.read_csv(paths["sta_summary_csv"])
        self.assertTrue(sta.empty)
        self.assertFalse((Path(paths["ppc_csv"]).parent / "spike_field_coherence_spectrum.png").exists())
        self.assertFalse((Path(paths["ppc_csv"]).parent / "spike_triggered_lfp").exists())


if __name__ == "__main__":
    unittest.main()
