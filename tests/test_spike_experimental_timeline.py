from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Plotting and Utilities"))
import preparation_spike_analysis as a
import spike_experimental_timeline as t


def row(epoch, start, end, rate, channel="A"):
    return dict(epoch_label=epoch, window_start_s=start, window_end_s=end, spike_count=rate*(end-start), channel=channel)


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def write(self, name, rows):
        path = self.folder / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def sample(self, name="one.csv", post1_length=2, baseline_rate=2):
        rows = [row("Baseline", i*60, (i+1)*60, baseline_rate) for i in range(10)]
        rows += [row("Post 1", i*60, (i+1)*60, 4) for i in range(post1_length)]
        rows += [row("Post 2", i*60, (i+1)*60, 6) for i in range(4)]
        return self.write(name, rows)

    def data(self, paths, currents=None, reference=None, names=None):
        return a.normalize(a.load_preparations(paths, currents or [-200]*len(paths), names), reference)

    def test_current_axis_points_only_and_names_preserve_identity(self):
        paths = [self.sample(f"p{i}.csv") for i in range(4)]
        currents = [-300, -100, -450, -200]
        original, *_ = self.data(paths, currents)
        names = {p.as_posix(): "Same display name" for p in paths}
        data, epochs, _, _ = self.data(paths, currents, names=names)
        self.assertEqual(original.preparation_id.tolist(), data.preparation_id.tolist())
        self.assertEqual(data.display_name.nunique(), 1)
        self.assertEqual(data.display_label.nunique(), 4)
        np.testing.assert_allclose(original.normalized_rate_pct, data.normalized_rate_pct)
        group = a.summarize(epochs, ["channel", "current_na", "epoch_label"], "normalized_rate_pct")
        points = epochs[epochs.epoch_label == "Post 1"]
        fig = a.current_figure(points, group[group.epoch_label == "Post 1"], "Title", a.preparation_styles(data))
        try:
            ax = fig.axes[0]
            self.assertGreater(ax.get_xlim()[0], ax.get_xlim()[1])
            self.assertEqual(ax.get_xticks().tolist(), [-100, -200, -300, -450])
            self.assertEqual(len(ax.collections), 4)
            self.assertFalse(ax.containers)
            self.assertNotIn("Group mean", ax.get_legend_handles_labels()[1])
            self.assertEqual(ax.get_ylim()[0], 0)
            self.assertGreaterEqual(ax.get_ylim()[1], 100)
            self.assertEqual([x.get_text() for x in ax.texts], ["n = 1"]*4)
        finally:
            a.plt.close(fig)
        specs = a.plot_specs(data, epochs, "normalized")
        self.assertTrue(any("Same display name" in s["default_title"] for s in specs))
        self.assertEqual(a.resolved_title(specs[0], {specs[0]["plot_id"]: "Explicit title"}), "Explicit title")

    def test_recording_symbols_and_current_publication_style(self):
        p = self.sample()
        data, epochs, *_ = self.data([p])
        styles = a.preparation_styles(data, {p.as_posix(): "D"})
        self.assertEqual(styles[data.preparation_id.iloc[0]]["marker"], "D")
        self.assertEqual(styles, a.preparation_styles(data.iloc[::-1], {p.as_posix(): "D"}))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            a.preparation_styles(data, {str(p): "invalid"})
        points = epochs[epochs.epoch_label == "Post 1"]
        summary = a.summarize(points, ["channel", "current_na", "epoch_label"], "normalized_rate_pct")
        fig = a.current_figure(points, summary, "Current response", styles)
        try:
            ax = fig.axes[0]
            self.assertEqual(fig._suptitle.get_fontsize(), 14)
            self.assertEqual(ax.yaxis.label.get_fontfamily(), ["DejaVu Sans"])
            self.assertGreater(ax.get_legend().get_bbox_to_anchor().x0, ax.get_window_extent().x1)
            self.assertLess(fig.get_figwidth()*ax.get_position().width, 5)
        finally:
            a.plt.close(fig)

    def test_reference_override_and_display_range_are_independent(self):
        p = self.write("p.csv", [row("Baseline", 0, 60, 2), row("Baseline", 60, 120, 4), row("Post 1", 0, 60, 6), row("Post 2", 0, 60, 12)])
        data, epochs, bases, _ = self.data([p], reference=[.5, 1.5])
        self.assertEqual(bases.baseline_rate_hz.iloc[0], 3)
        self.assertEqual(bases.baseline_duration_s.iloc[0], 60)
        self.assertEqual(bases.baseline_reference_start_s.iloc[0], 30)
        self.assertEqual(epochs.set_index("epoch_label").loc["Post 2", "normalized_rate_pct"], 400)
        full, _ = t.build_experimental_timeline(data, {"baseline_display": "full"})
        limited, _ = t.build_experimental_timeline(data, {"baseline_display": "range", "display_baseline_range_min": [1, 2]})
        np.testing.assert_allclose(full.normalized_rate_pct, limited.normalized_rate_pct)
        self.assertEqual(full.displayed.sum()-limited.displayed.sum(), 1)

    def test_reconstructed_midpoints_relative_onset_and_compression(self):
        data, *_ = self.data([self.sample()])
        timeline, treatments = t.build_experimental_timeline(data)
        base = timeline[timeline.epoch_label == "Baseline"]
        self.assertEqual(base.plot_midpoint_min.tolist(), list(np.arange(-9.5, 0, 1)))
        post = timeline[timeline.epoch_label == "Post 1"]
        self.assertEqual(post.plot_midpoint_min.tolist(), [20.5, 21.5])
        self.assertEqual(treatments.plot_start_min.tolist(), [0, 22])
        self.assertEqual(treatments.plot_end_min.tolist(), [20, 42])
        self.assertEqual(len(timeline), len(data))
        compressed, bars = t.build_experimental_timeline(data, {"show_treatment_duration": False})
        self.assertEqual(compressed[compressed.epoch_label == "Post 1"].plot_midpoint_min.tolist(), [1.5, 2.5])
        self.assertEqual(compressed[compressed.epoch_label == "Post 2"].plot_midpoint_min.iloc[0], 4.5)
        self.assertEqual(bars.duration_min.tolist(), [20, 20])
        self.assertTrue(compressed.timing_compressed.all())
        np.testing.assert_allclose(timeline.experimental_midpoint_min, compressed.experimental_midpoint_min)
        np.testing.assert_allclose(timeline.normalized_rate_pct, compressed.normalized_rate_pct)

    def test_boundary_overrides_and_no_treatment(self):
        data, *_ = self.data([self.sample()])
        config = {"boundaries": {"Baseline -> Post 1": {"treatment": False}, "Post 1 -> Post 2": {"duration_min": 7}}}
        timeline, treatments = t.build_experimental_timeline(data, config)
        self.assertEqual(len(treatments), 1)
        # Relative origin is the first actual configured treatment, after Post 1.
        self.assertEqual(timeline[timeline.epoch_label == "Post 1"].plot_midpoint_min.iloc[0], -1.5)
        self.assertEqual(treatments.duration_min.iloc[0], 7)

    def test_measured_times_are_preferred_without_double_gap(self):
        rows = [dict(row("Baseline", 0, 600, 2), experimental_start_s=0, experimental_end_s=600),
                dict(row("Post 1", 0, 60, 4), experimental_start_s=1800, experimental_end_s=1860)]
        data, *_ = self.data([self.write("measured.csv", rows)])
        timeline, treatments = t.build_experimental_timeline(data)
        self.assertTrue((timeline.timing_source == "measured").all())
        self.assertEqual(timeline[timeline.epoch_label == "Post 1"].plot_midpoint_min.iloc[0], 20.5)
        self.assertEqual(treatments.duration_min.iloc[0], 20)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            t.build_experimental_timeline(data, {"gap_min": 30})

    def test_recorded_treatment_order_and_measured_overlap_validation(self):
        p = self.write("recorded.csv", [row("Baseline", 0, 600, 2), row("Stimulation", 0, 1200, 3), row("Post 1", 0, 60, 4)])
        data, *_ = self.data([p])
        timeline, bars = t.build_experimental_timeline(data)
        self.assertEqual(timeline[timeline.epoch_label == "Post 1"].plot_midpoint_min.iloc[0], 20.5)
        self.assertEqual(bars.duration_min.tolist(), [20])
        self.assertTrue(bars.recorded.all())
        p = self.write("overlap.csv", [dict(row("Baseline", 0, 60, 2), experimental_start_s=0, experimental_end_s=60), dict(row("Baseline", 60, 120, 2), experimental_start_s=30, experimental_end_s=90)])
        data, *_ = self.data([p])
        with self.assertRaisesRegex(ValueError, "overlap within"):
            t.build_experimental_timeline(data)

    def test_ambiguous_times_missing_epochs_and_assignments(self):
        p = self.write("clock.csv", [row("Baseline", 0, 600, 2), row("Post 1", 1800, 1860, 4)])
        data, *_ = self.data([p])
        with self.assertRaisesRegex(ValueError, "already contain"):
            t.build_experimental_timeline(data)
        measured, _ = t.build_experimental_timeline(data, {"timing_mode": "measured"})
        self.assertEqual(measured[measured.epoch_label == "Post 1"].plot_midpoint_min.iloc[0], 20.5)
        missing = self.write("missing.csv", [row("Baseline", 0, 600, 2), row("Post 2", 0, 60, 4)])
        data, *_ = self.data([missing])
        with self.assertRaisesRegex(ValueError, "missing intermediate"):
            t.build_experimental_timeline(data)
        assigned, _ = t.build_experimental_timeline(data, {"epoch_starts_min": {str(missing): {"Baseline": 0, "Post 2": 60}}})
        self.assertEqual(assigned[assigned.epoch_label == "Post 2"].plot_midpoint_min.iloc[0], 50.5)

    def test_grouping_respects_epoch_elapsed_and_experimental_time(self):
        data, *_ = self.data([self.sample("a.csv", 2), self.sample("b.csv", 3)])
        timeline, _ = t.build_experimental_timeline(data)
        groups = t.summarize_timeline(timeline)
        self.assertTrue((groups[groups.epoch_label == "Baseline"].n_preparations == 2).all())
        self.assertEqual(groups[groups.epoch_label == "Post 1"].n_preparations.tolist(), [2, 2, 1])
        # Some Post 2 global times overlap, but they are different minutes since
        # the second treatment; they must not be averaged together.
        self.assertTrue((groups[groups.epoch_label == "Post 2"].n_preparations == 1).all())
        self.assertTrue(groups[groups.epoch_label == "Post 2"].group_sem_normalized_pct.isna().all())

    def test_zero_baseline_keeps_absolute_output_and_channels_do_not_inflate_n(self):
        p = self.sample(baseline_rate=0)
        rows = pd.read_csv(p)
        extra = rows.copy()
        extra["channel"] = "B"
        pd.concat([rows, extra]).to_csv(p, index=False)
        data, epochs, *_ = self.data([p])
        specs = a.plot_specs(data, epochs, "timeline")
        self.assertEqual([s["kind"] for s in specs], ["time_course_absolute_hz"]*2)
        timeline, bars = t.build_experimental_timeline(data)
        group = t.summarize_timeline(timeline)
        self.assertTrue((group.n_preparations_hz == 1).all())
        self.assertTrue((group.n_preparations == 0).all())
        fig = t.timeline_figure(timeline[timeline.channel == "A"], group[group.channel == "A"], bars, "Absolute rate", False, a.preparation_styles(data), {})
        try:
            self.assertEqual(fig.axes[0].get_ylim()[0], 0)
            self.assertFalse(fig.axes[0].containers)
        finally:
            a.plt.close(fig)

    def test_time_course_export_catalog_manifest_and_artists(self):
        paths = [self.sample("p1.csv"), self.sample("p2.csv", baseline_rate=4)]
        output = self.folder / "out"
        names = {str(paths[0]): "Locust A", str(paths[1]): "Locust B"}
        specs = a.discover_plot_specs(paths, [-200, -200], "timeline", names)
        title = "Long custom experimental title " * 7
        result = a.run_analysis(paths, [-200, -200], output, "timeline", {s["plot_id"]: title for s in specs}, names)
        self.assertEqual(len(list((output / "Plots").glob("*.png"))), 2)
        used = pd.read_csv(output / "CSVs/experimental_time_windows.csv")
        self.assertTrue({"display_name", "recording_name", "window_index", "group_mean_normalized_pct", "baseline_reference_start_s"}.issubset(used.columns))
        self.assertEqual(set(used.display_name), {"Locust A", "Locust B"})
        fig = t.timeline_figure(result["timeline_data"], result["timeline_groups"], result["treatments"], title, True, a.preparation_styles(result["data"]), {})
        try:
            ax = fig.axes[0]
            self.assertEqual(ax.get_ylim()[0], 0)
            self.assertGreaterEqual(ax.get_ylim()[1], 100)
            self.assertTrue(ax.containers)
            self.assertIn("Group mean", ax.get_legend_handles_labels()[1])
            # No line runs from pre-treatment to post-treatment observations.
            for line in ax.lines:
                x = np.asarray(line.get_xdata(), dtype=float)
                if len(x) > 2:
                    self.assertFalse(np.any(np.diff(x) > 1.1))
        finally:
            a.plt.close(fig)
        timeline_files = set(p.name for p in (output / "Plots").iterdir())
        a.run_analysis(paths, [-200, -200], output, "overlay", display_names=names)
        self.assertTrue(timeline_files.issubset({p.name for p in (output / "Plots").iterdir()}))
        self.assertTrue((output / "CSVs/experimental_time_windows.csv").exists())


if __name__ == "__main__":
    unittest.main()
