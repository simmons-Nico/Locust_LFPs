from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "Spike Processing" / "Spike Count Multiple CSVs ordered.py"


def load_spike_module():
    spec = importlib.util.spec_from_file_location("spike_ordered_polarity_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.bandpass_filt = lambda x, fs, band, order=3: np.asarray(x, dtype=float)
    module.SPIKE_Z_THR = 6.0
    module.REFRACTORY_MS = 1.0
    module.AMP_MIN_UV = 40.0
    module.AMP_MAX_UV = 700.0
    module.W_MIN_MS = None
    module.W_MAX_MS = None
    module.SPIKE_CLASSIFICATION_WINDOW_MS = 0.5
    return module


def synthetic_trace(fs: float = 20_000.0, duration_s: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0.0, duration_s, 1.0 / fs)
    x = 0.45 * np.sin(2.0 * np.pi * 173.0 * t) + 0.25 * np.sin(2.0 * np.pi * 431.0 * t)
    return t, x


def add_recording(recordings: list[dict], key: str, order: int, name: str, channel: str, phase: str, duration_s: float) -> None:
    recordings.append(
        {
            "recording_key": key,
            "recording_order": order,
            "recording_name": name,
            "channel": channel,
            "epoch_label": name,
            "phase": phase,
            "phase_group": phase,
            "recording_duration_s": duration_s,
        }
    )


def add_amp_event(
    events: list[dict],
    key: str,
    order: int,
    name: str,
    channel: str,
    phase: str,
    minute: int,
    second: float,
    polarity_short: str,
    signed_amplitude_uv: float,
) -> None:
    events.append(
        {
            "recording_key": key,
            "recording_order": order,
            "recording_name": name,
            "epoch_label": name,
            "phase": phase,
            "phase_group": phase,
            "channel": channel,
            "polarity": "positive" if polarity_short == "pos" else "negative",
            "polarity_short": polarity_short,
            "sample_index": 0,
            "canonical_sample_index": 0,
            "spike_time_s": minute * 60.0 + second,
            "recording_relative_spike_time_s": minute * 60.0 + second,
            "signed_amplitude_uv": signed_amplitude_uv,
            "amplitude_uv": signed_amplitude_uv if polarity_short == "pos" else abs(signed_amplitude_uv),
            "robust_z": 10.0,
        }
    )


def add_amp_waveform(waveforms: list[dict], channel: str, phase: str, polarity_short: str, amplitude_uv: float, count: int = 3) -> None:
    rel_ms = np.array([-0.5, 0.0, 0.5], dtype=float)
    peak = amplitude_uv if polarity_short == "pos" else -amplitude_uv
    waveform = np.array([0.0, peak, 0.0], dtype=float)
    for _ in range(count):
        waveforms.append(
            {
                "channel": channel,
                "phase_group": phase,
                "polarity_short": polarity_short,
                "waveform": waveform.copy(),
                "rel_ms": rel_ms.copy(),
            }
        )


def amplitude_fixture() -> tuple[list[dict], list[dict], list[dict]]:
    recordings: list[dict] = []
    events: list[dict] = []
    waveforms: list[dict] = []

    for channel in ("A-001", "A-002"):
        add_recording(recordings, f"base::{channel}", 1, "baseline", channel, "baseline", 12 * 60.0)
        add_recording(recordings, f"treat::{channel}", 2, "treatment", channel, "treatment", 2 * 60.0)
        add_recording(recordings, f"post::{channel}", 3, "post", channel, "post", 60.0)

    for minute in range(12):
        neg_amp = 55.0 if minute < 2 else 100.0
        pos_amp = 25.0 if minute < 2 else 40.0
        for second in (5.0, 25.0, 45.0):
            add_amp_event(events, "base::A-001", 1, "baseline", "A-001", "baseline", minute, second, "neg", -neg_amp)
            add_amp_event(events, "base::A-001", 1, "baseline", "A-001", "baseline", minute, second + 2.0, "pos", pos_amp)
            add_amp_event(events, "base::A-002", 1, "baseline", "A-002", "baseline", minute, second, "neg", -200.0)
            add_amp_event(events, "base::A-002", 1, "baseline", "A-002", "baseline", minute, second + 2.0, "pos", 20.0)

    for amp, second in ((140.0, 10.0), (160.0, 40.0)):
        add_amp_event(events, "treat::A-001", 2, "treatment", "A-001", "treatment", 0, second, "neg", -amp)
    for amp, second in ((70.0, 12.0), (90.0, 42.0)):
        add_amp_event(events, "treat::A-001", 2, "treatment", "A-001", "treatment", 0, second, "pos", amp)
    add_amp_event(events, "treat::A-001", 2, "treatment", "A-001", "treatment", 1, 20.0, "pos", 60.0)
    add_amp_event(events, "post::A-001", 3, "post", "A-001", "post", 0, 20.0, "neg", -120.0)

    add_amp_event(events, "treat::A-002", 2, "treatment", "A-002", "treatment", 0, 20.0, "neg", -300.0)
    add_amp_event(events, "treat::A-002", 2, "treatment", "A-002", "treatment", 0, 25.0, "pos", 30.0)

    for channel, neg_amp, pos_amp in (("A-001", 100.0, 40.0), ("A-002", 200.0, 20.0)):
        for phase in ("baseline", "treatment", "post"):
            add_amp_waveform(waveforms, channel, phase, "neg", neg_amp)
            add_amp_waveform(waveforms, channel, phase, "pos", pos_amp)

    return recordings, events, waveforms


class SpikePolarityWaveformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_spike_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def detect(self, x: np.ndarray, t: np.ndarray, polarity: str = "both") -> dict:
        return self.module.detect_spikes_with_details(
            x,
            t,
            polarity=polarity,
            return_filtered_trace=True,
            classification_window_ms=0.5,
        )

    def test_negative_waveform_with_smaller_positive_rebound_is_negative_only(self) -> None:
        t, x = synthetic_trace()
        center = 600
        x[center] = -140.0
        x[center + 6] = 55.0

        details = self.detect(x, t)

        self.assertEqual(details["peaks"].tolist(), [center])
        self.assertEqual(details["event_polarity"].tolist(), ["neg"])
        self.assertEqual(details["n_positive"], 0)
        self.assertEqual(details["n_negative"], 1)
        self.assertLess(float(details["signed_amplitude_uv"][0]), 0.0)

    def test_positive_waveform_with_smaller_negative_rebound_is_positive_only(self) -> None:
        t, x = synthetic_trace()
        center = 700
        x[center] = 150.0
        x[center + 6] = -65.0

        details = self.detect(x, t)

        self.assertEqual(details["peaks"].tolist(), [center])
        self.assertEqual(details["event_polarity"].tolist(), ["pos"])
        self.assertEqual(details["n_positive"], 1)
        self.assertEqual(details["n_negative"], 0)
        self.assertGreater(float(details["signed_amplitude_uv"][0]), 0.0)

    def test_biphasic_later_lobe_with_larger_amplitude_aligns_to_later_lobe(self) -> None:
        t, x = synthetic_trace()
        first_lobe = 800
        later_lobe = first_lobe + 7
        x[first_lobe] = 80.0
        x[later_lobe] = -160.0

        details = self.detect(x, t)

        self.assertEqual(details["peaks"].tolist(), [later_lobe])
        self.assertEqual(details["event_polarity"].tolist(), ["neg"])
        self.assertEqual(len(details["spike_times"]), 1)
        self.assertAlmostEqual(float(details["spike_times"][0]), float(t[later_lobe]))

    def test_opposite_polarity_candidates_within_refractory_keep_largest_absolute_amplitude(self) -> None:
        t, x = synthetic_trace()
        positive = 900
        negative = positive + 15
        x[positive] = 95.0
        x[negative] = -155.0

        details = self.detect(x, t)

        self.assertEqual(details["peaks"].tolist(), [negative])
        self.assertEqual(details["event_polarity"].tolist(), ["neg"])
        self.assertAlmostEqual(float(details["amplitude_uv"][0]), 155.0)

    def test_amplitude_limits_apply_to_canonical_extremum_after_alignment(self) -> None:
        t, x = synthetic_trace()
        first_lobe = 1_000
        canonical = first_lobe + 6
        x[first_lobe] = 85.0
        x[canonical] = -130.0
        self.module.AMP_MAX_UV = 100.0

        details = self.detect(x, t)

        self.assertEqual(details["peaks"].size, 0)
        self.assertEqual(details["n_positive"], 0)
        self.assertEqual(details["n_negative"], 0)

    def test_waveform_outputs_are_split_and_aligned_at_zero(self) -> None:
        t, x = synthetic_trace(duration_s=0.2)
        positive = 1_000
        negative = 2_000
        x[positive] = 145.0
        x[positive + 5] = -50.0
        x[negative] = -155.0
        x[negative + 5] = 60.0
        csv_path = self.root / "spikes.csv"
        out_dir = self.root / "out"
        pd.DataFrame({"time_s": t, "A-001": x}).to_csv(csv_path, index=False)

        count_csv = self.module.process_csvs(
            [str(csv_path)],
            out_dir=str(out_dir),
            window_sec=0.2,
            polarity="both",
            export_spike_events=True,
            classification_window_ms=0.5,
        )

        counts = pd.read_csv(count_csv)
        self.assertEqual(int(counts["spike_count"].sum()), 2)
        self.assertEqual(int(counts["n_positive"].sum()), 1)
        self.assertEqual(int(counts["n_negative"].sum()), 1)

        events = pd.read_csv(out_dir / "spike_events.csv")
        self.assertEqual(set(events["polarity"]), {"positive", "negative"})
        self.assertEqual(set(events["polarity_short"]), {"pos", "neg"})
        self.assertTrue((events["sample_index"] == events["canonical_sample_index"]).all())
        self.assertEqual(set(events["n_positive"]), {1})
        self.assertEqual(set(events["n_negative"]), {1})

        summary = pd.read_csv(out_dir / "spike_waveform_summary.csv")
        self.assertEqual(set(summary["polarity"]), {"positive", "negative"})
        self.assertTrue((out_dir / "spike_diagnostics" / "average_waveforms" / "spikes__A-001__positive_average_waveform.png").exists())
        self.assertTrue((out_dir / "spike_diagnostics" / "average_waveforms" / "spikes__A-001__negative_average_waveform.png").exists())
        self.assertFalse((out_dir / "spike_diagnostics" / "average_waveforms" / "spikes__A-001__average_waveform.png").exists())

        pos_csv = summary.loc[summary["polarity"] == "positive", "waveform_csv"].iloc[0]
        neg_csv = summary.loc[summary["polarity"] == "negative", "waveform_csv"].iloc[0]
        pos_wf = pd.read_csv(pos_csv)
        neg_wf = pd.read_csv(neg_csv)
        pos_zero = pos_wf.loc[np.isclose(pos_wf["time_ms"], 0.0), "mean_waveform_uv"].iloc[0]
        neg_zero = neg_wf.loc[np.isclose(neg_wf["time_ms"], 0.0), "mean_waveform_uv"].iloc[0]
        self.assertAlmostEqual(float(pos_zero), 145.0)
        self.assertAlmostEqual(float(neg_zero), -155.0)

    def test_requested_single_polarity_outputs_only_that_class(self) -> None:
        t, x = synthetic_trace(duration_s=0.2)
        positive = 1_000
        negative = 2_000
        x[positive] = 135.0
        x[negative] = -145.0
        csv_path = self.root / "single_polarity.csv"
        out_dir = self.root / "pos_out"
        pd.DataFrame({"time_s": t, "A-001": x}).to_csv(csv_path, index=False)

        count_csv = self.module.process_csvs(
            [str(csv_path)],
            out_dir=str(out_dir),
            window_sec=0.2,
            polarity="pos",
            export_spike_events=True,
            classification_window_ms=0.5,
        )

        counts = pd.read_csv(count_csv)
        self.assertEqual(int(counts["spike_count"].sum()), 1)
        self.assertEqual(int(counts["n_positive"].sum()), 1)
        self.assertEqual(int(counts["n_negative"].sum()), 0)
        events = pd.read_csv(out_dir / "spike_events.csv")
        self.assertEqual(events["polarity"].tolist(), ["positive"])
        summary = pd.read_csv(out_dir / "spike_waveform_summary.csv")
        self.assertEqual(summary["polarity"].tolist(), ["positive"])
        self.assertTrue(summary["waveform_csv"].iloc[0].endswith("__positive_average_waveform.csv"))
        self.assertFalse((out_dir / "spike_diagnostics" / "average_waveforms" / "single_polarity__A-001__negative_average_waveform.csv").exists())

    def test_amplitude_timecourse_keeps_polarities_channels_and_bins_separate(self) -> None:
        recordings, events, waveforms = amplitude_fixture()
        out = self.module.build_spike_amplitude_timecourse_outputs(
            str(self.root / "amp"),
            recordings,
            events,
            waveforms,
            requested_polarity="both",
            overlay=False,
        )

        self.assertTrue(Path(out["csv_path"]).exists())
        self.assertEqual(len(out["plot_paths"]), 4)
        self.assertTrue(all(Path(path).exists() for path in out["plot_paths"]))
        df = pd.read_csv(out["csv_path"])
        self.assertTrue(set(self.module.SPIKE_AMPLITUDE_TIMECOURSE_COLUMNS).issubset(df.columns))

        a1_neg = df[(df["channel"] == "A-001") & (df["polarity"] == "negative")]
        a1_pos = df[(df["channel"] == "A-001") & (df["polarity"] == "positive")]
        a2_neg = df[(df["channel"] == "A-002") & (df["polarity"] == "negative")]

        self.assertEqual(set(a1_neg["baseline_mean_amplitude_uv"].dropna()), {100.0})
        self.assertEqual(set(a1_pos["baseline_mean_amplitude_uv"].dropna()), {40.0})
        self.assertEqual(set(a2_neg["baseline_mean_amplitude_uv"].dropna()), {200.0})

        self.assertEqual(set(a1_neg["baseline_interval_start_min"].dropna()), {2.0})
        self.assertEqual(set(a1_neg["baseline_interval_end_min"].dropna()), {12.0})
        self.assertEqual(set(a1_neg["baseline_spike_count"].dropna()), {30})
        self.assertTrue(a1_neg["baseline_qc_pass"].astype(bool).all())

        first_treatment_neg = a1_neg[(a1_neg["phase_group"] == "treatment") & (a1_neg["recording_bin_index"] == 0)].iloc[0]
        self.assertEqual(int(first_treatment_neg["n_spikes"]), 2)
        self.assertAlmostEqual(float(first_treatment_neg["mean_amplitude_uv"]), 150.0)
        self.assertAlmostEqual(float(first_treatment_neg["percent_of_baseline"]), 150.0)
        self.assertAlmostEqual(float(first_treatment_neg["treatment_relative_bin_mid_min"]), 0.5)

        first_treatment_pos = a1_pos[(a1_pos["phase_group"] == "treatment") & (a1_pos["recording_bin_index"] == 0)].iloc[0]
        self.assertAlmostEqual(float(first_treatment_pos["mean_amplitude_uv"]), 80.0)
        self.assertAlmostEqual(float(first_treatment_pos["percent_of_baseline"]), 200.0)

        second_treatment_pos = a1_pos[(a1_pos["phase_group"] == "treatment") & (a1_pos["recording_bin_index"] == 1)].iloc[0]
        self.assertEqual(int(second_treatment_pos["n_spikes"]), 1)
        self.assertAlmostEqual(float(second_treatment_pos["mean_amplitude_uv"]), 60.0)
        self.assertTrue(pd.isna(second_treatment_pos["sd_amplitude_uv"]))
        self.assertTrue(pd.isna(second_treatment_pos["sem_across_spikes_uv"]))
        self.assertTrue(pd.isna(second_treatment_pos["percent_sem"]))

        second_treatment_neg = a1_neg[(a1_neg["phase_group"] == "treatment") & (a1_neg["recording_bin_index"] == 1)].iloc[0]
        self.assertEqual(int(second_treatment_neg["n_spikes"]), 0)
        self.assertTrue(pd.isna(second_treatment_neg["mean_amplitude_uv"]))
        self.assertTrue(pd.isna(second_treatment_neg["sd_amplitude_uv"]))
        self.assertTrue(pd.isna(second_treatment_neg["sem_across_spikes_uv"]))
        self.assertTrue(pd.isna(second_treatment_neg["percent_of_baseline"]))

        last_baseline_neg = a1_neg[(a1_neg["phase_group"] == "baseline") & (a1_neg["recording_bin_index"] == 11)].iloc[0]
        self.assertAlmostEqual(float(last_baseline_neg["treatment_relative_bin_mid_min"]), -0.5)
        self.assertAlmostEqual(float(last_baseline_neg["percent_of_baseline"]), 100.0)

    def test_amplitude_overlay_mode_writes_one_figure_per_channel(self) -> None:
        recordings, events, waveforms = amplitude_fixture()
        out = self.module.build_spike_amplitude_timecourse_outputs(
            str(self.root / "overlay"),
            recordings,
            events,
            waveforms,
            requested_polarity="both",
            overlay=True,
        )

        self.assertTrue(Path(out["csv_path"]).exists())
        self.assertEqual(len(out["plot_paths"]), 2)
        self.assertTrue(all(path.endswith("_overlay.png") for path in out["plot_paths"]))
        self.assertTrue(all(Path(path).exists() for path in out["plot_paths"]))

    def test_phase_normalization_uses_epoch_labels_with_post_treatment_as_post(self) -> None:
        self.assertEqual(self.module.normalize_amplitude_phase(epoch_label="Baseline")[0], "baseline")
        self.assertEqual(self.module.normalize_amplitude_phase(epoch_label="Before recording")[0], "baseline")
        self.assertEqual(self.module.normalize_amplitude_phase(epoch_label="pre")[0], "baseline")
        self.assertEqual(self.module.normalize_amplitude_phase(epoch_label="Post-treatment")[0], "post")
        self.assertEqual(self.module.normalize_amplitude_phase(epoch_label="During 1")[0], "treatment")
        self.assertEqual(self.module.normalize_amplitude_phase(phase="treatment", epoch_label="Stimulation")[0], "treatment")

    def test_unknown_epoch_label_warns_without_position_based_phase_inference(self) -> None:
        recordings = [
            {
                "recording_key": "mystery::A-001",
                "recording_order": 1,
                "recording_name": "mystery",
                "channel": "A-001",
                "epoch_label": "Mystery interval",
                "phase": "",
                "phase_group": "",
                "recording_duration_s": 60.0,
            }
        ]
        out = self.module.build_spike_amplitude_timecourse_outputs(
            str(self.root / "unknown_label"),
            recordings,
            [],
            [],
            requested_polarity="neg",
        )
        df = pd.read_csv(out["csv_path"])
        self.assertEqual(df["phase_group"].tolist(), ["unknown"])
        self.assertEqual(df["normalized_phase"].tolist(), ["unknown"])
        self.assertTrue(any("unknown epoch label" in warning for warning in out["warnings"]))

    def test_amplitude_protocol_with_recorded_during_keeps_post_after_treatment(self) -> None:
        recordings: list[dict] = []
        events: list[dict] = []
        waveforms: list[dict] = []
        add_recording(recordings, "base::A-001", 1, "Baseline", "A-001", "baseline", 10 * 60.0)
        add_recording(recordings, "during::A-001", 2, "During", "A-001", "treatment", 2 * 60.0)
        add_recording(recordings, "post1::A-001", 3, "Post 1", "A-001", "post", 60.0)
        add_recording(recordings, "post2::A-001", 4, "Post 2", "A-001", "post", 60.0)
        for minute in range(10):
            for second in (10.0, 30.0, 50.0):
                add_amp_event(events, "base::A-001", 1, "Baseline", "A-001", "baseline", minute, second, "neg", -100.0)
        add_amp_event(events, "during::A-001", 2, "During", "A-001", "treatment", 0, 20.0, "neg", -150.0)
        add_amp_event(events, "post1::A-001", 3, "Post 1", "A-001", "post", 0, 20.0, "neg", -120.0)
        add_amp_event(events, "post2::A-001", 4, "Post 2", "A-001", "post", 0, 20.0, "neg", -130.0)
        for phase, amp in (("baseline", 100.0), ("treatment", 150.0), ("post", 120.0)):
            add_amp_waveform(waveforms, "A-001", phase, "neg", amp)

        out = self.module.build_spike_amplitude_timecourse_outputs(
            str(self.root / "recorded_during"),
            recordings,
            events,
            waveforms,
            requested_polarity="neg",
        )
        df = pd.read_csv(out["csv_path"])
        self.assertEqual(set(df[df["epoch_label"].isin(["Post 1", "Post 2"])]["phase_group"]), {"post"})
        self.assertTrue(df["treatment_recorded"].astype(bool).all())
        self.assertTrue(df["treatment_duration_known"].astype(bool).all())
        last_baseline = df[(df["phase_group"] == "baseline") & (df["recording_bin_index"] == 9)].iloc[0]
        first_during = df[(df["phase_group"] == "treatment") & (df["recording_bin_index"] == 0)].iloc[0]
        first_post = df[(df["epoch_label"] == "Post 1") & (df["recording_bin_index"] == 0)].iloc[0]
        self.assertAlmostEqual(float(last_baseline["plotted_relative_bin_mid_min"]), -0.5)
        self.assertAlmostEqual(float(first_during["plotted_relative_bin_mid_min"]), 0.5)
        self.assertAlmostEqual(float(first_post["plotted_relative_bin_mid_min"]), 2.5)
        self.assertAlmostEqual(float(first_post["percent_of_baseline"]), 120.0)

    def test_amplitude_protocol_without_during_uses_first_post_as_time_zero(self) -> None:
        recordings: list[dict] = []
        events: list[dict] = []
        waveforms: list[dict] = []
        add_recording(recordings, "base::A-001", 1, "Baseline", "A-001", "baseline", 10 * 60.0)
        add_recording(recordings, "post1::A-001", 2, "Post 1", "A-001", "post", 60.0)
        add_recording(recordings, "post2::A-001", 3, "Post 2", "A-001", "post", 60.0)
        for minute in range(10):
            for second in (10.0, 30.0, 50.0):
                add_amp_event(events, "base::A-001", 1, "Baseline", "A-001", "baseline", minute, second, "neg", -100.0)
        for key, order, name, amp in (("post1::A-001", 2, "Post 1", 125.0), ("post2::A-001", 3, "Post 2", 140.0)):
            for second in (10.0, 30.0, 50.0):
                add_amp_event(events, key, order, name, "A-001", "post", 0, second, "neg", -amp)
        add_amp_waveform(waveforms, "A-001", "baseline", "neg", 100.0)
        add_amp_waveform(waveforms, "A-001", "post", "neg", 125.0)

        out = self.module.build_spike_amplitude_timecourse_outputs(
            str(self.root / "unrecorded_treatment"),
            recordings,
            events,
            waveforms,
            requested_polarity="neg",
            boundary_label="Drug application - not recorded",
        )
        df = pd.read_csv(out["csv_path"])
        self.assertNotIn("treatment", set(df["phase_group"]))
        self.assertFalse(df["treatment_recorded"].astype(bool).any())
        self.assertFalse(df["treatment_duration_known"].astype(bool).any())
        self.assertEqual(set(df["treatment_boundary_label"].dropna()), {"Drug application - not recorded"})
        last_baseline = df[(df["phase_group"] == "baseline") & (df["recording_bin_index"] == 9)].iloc[0]
        post1 = df[(df["epoch_label"] == "Post 1") & (df["recording_bin_index"] == 0)].iloc[0]
        post2 = df[(df["epoch_label"] == "Post 2") & (df["recording_bin_index"] == 0)].iloc[0]
        self.assertAlmostEqual(float(last_baseline["plotted_relative_bin_mid_min"]), -0.5)
        self.assertAlmostEqual(float(post1["plotted_relative_bin_mid_min"]), 0.5)
        self.assertAlmostEqual(float(post2["plotted_relative_bin_mid_min"]), 1.5)
        self.assertAlmostEqual(float(post1["percent_of_baseline"]), 125.0)
        self.assertAlmostEqual(float(post2["percent_of_baseline"]), 140.0)

    def test_amplitude_overlay_and_separate_modes_have_identical_phase_values(self) -> None:
        recordings, events, waveforms = amplitude_fixture()
        separate = self.module.build_spike_amplitude_timecourse_outputs(
            str(self.root / "separate_compare"),
            recordings,
            events,
            waveforms,
            requested_polarity="both",
            overlay=False,
        )
        overlay = self.module.build_spike_amplitude_timecourse_outputs(
            str(self.root / "overlay_compare"),
            recordings,
            events,
            waveforms,
            requested_polarity="both",
            overlay=True,
        )
        sep_df = pd.read_csv(separate["csv_path"]).sort_values(["channel", "polarity_short", "bin_index"]).reset_index(drop=True)
        over_df = pd.read_csv(overlay["csv_path"]).sort_values(["channel", "polarity_short", "bin_index"]).reset_index(drop=True)
        compare_columns = [
            "epoch_label",
            "phase_group",
            "normalized_phase",
            "plotted_relative_bin_mid_min",
            "percent_of_baseline",
            "treatment_recorded",
            "treatment_duration_known",
        ]
        pd.testing.assert_frame_equal(sep_df[compare_columns], over_df[compare_columns])
        self.assertNotEqual(len(separate["plot_paths"]), len(overlay["plot_paths"]))

    def test_amplitude_baseline_stability_failure_is_reported_without_dropping_data(self) -> None:
        recordings: list[dict] = []
        events: list[dict] = []
        waveforms: list[dict] = []
        add_recording(recordings, "base::A-001", 1, "baseline", "A-001", "baseline", 10 * 60.0)
        add_recording(recordings, "treat::A-001", 2, "treatment", "A-001", "treatment", 60.0)
        for minute in range(10):
            amp = 80.0 + 6.0 * minute
            for second in (10.0, 30.0, 50.0):
                add_amp_event(events, "base::A-001", 1, "baseline", "A-001", "baseline", minute, second, "neg", -amp)
        add_amp_event(events, "treat::A-001", 2, "treatment", "A-001", "treatment", 0, 30.0, "neg", -140.0)
        add_amp_waveform(waveforms, "A-001", "baseline", "neg", 100.0)
        add_amp_waveform(waveforms, "A-001", "treatment", "neg", 140.0)

        out = self.module.build_spike_amplitude_timecourse_outputs(
            str(self.root / "unstable"),
            recordings,
            events,
            waveforms,
            requested_polarity="neg",
            overlay=False,
        )

        df = pd.read_csv(out["csv_path"])
        self.assertFalse(df["baseline_qc_pass"].astype(bool).all())
        self.assertTrue(any("slope" in warning for warning in out["warnings"]))
        self.assertTrue(df["percent_of_baseline"].notna().any())
        self.assertTrue(all(Path(path).exists() for path in out["plot_paths"]))

    def test_amplitude_feature_does_not_change_combined_spike_counts(self) -> None:
        fs = 1_000.0
        t = np.arange(0.0, 120.0, 1.0 / fs)
        raw = 0.2 * np.sin(2.0 * np.pi * 13.0 * t)
        for spike_time, amplitude in ((10.0, -130.0), (70.0, 140.0)):
            raw[int(spike_time * fs)] = amplitude
        csv_path = self.root / "count_regression.csv"
        pd.DataFrame({"time_s": t, "A-001": raw}).to_csv(csv_path, index=False)

        no_amp_csv = self.module.process_csvs(
            [str(csv_path)],
            out_dir=str(self.root / "no_amp"),
            window_sec=60.0,
            polarity="both",
            plot_spike_amplitude_change=False,
        )
        amp_csv = self.module.process_csvs(
            [str(csv_path)],
            out_dir=str(self.root / "with_amp"),
            window_sec=60.0,
            polarity="both",
            plot_spike_amplitude_change=True,
            overlay_spike_amplitudes=True,
        )

        no_amp = pd.read_csv(no_amp_csv)
        with_amp = pd.read_csv(amp_csv)
        pd.testing.assert_frame_equal(no_amp, with_amp)
        self.assertTrue((self.root / "with_amp" / "spike_amplitude_time_course.csv").exists())

    def test_amplitude_bin_membership_uses_complete_sixty_second_bins(self) -> None:
        recordings = []
        events = []
        waveforms = []
        add_recording(recordings, "base::A-001", 1, "baseline", "A-001", "baseline", 120.5)
        add_amp_event(events, "base::A-001", 1, "baseline", "A-001", "baseline", 0, 59.999, "neg", -100.0)
        add_amp_event(events, "base::A-001", 1, "baseline", "A-001", "baseline", 1, 0.0, "neg", -110.0)

        out = self.module.build_spike_amplitude_timecourse_outputs(
            str(self.root / "bins"),
            recordings,
            events,
            waveforms,
            requested_polarity="neg",
            overlay=False,
        )
        df = pd.read_csv(out["csv_path"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df["recording_bin_index"].tolist(), [0, 1])
        self.assertEqual(df["n_spikes"].tolist(), [1, 1])
        self.assertEqual(df["mean_amplitude_uv"].tolist(), [100.0, 110.0])


if __name__ == "__main__":
    unittest.main()
