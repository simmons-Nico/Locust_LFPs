from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Plotting and Utilities"))
import preparation_spike_analysis as analysis


def row(epoch, start, end, rate, channel="A-000", recording="chunk"):
    return dict(epoch_label=epoch, window_start_s=start, window_end_s=end,
                spike_count=(end - start) * rate, channel=channel, recording_name=recording)


class PreparationAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def write(self, name, rows):
        path = self.folder / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_final_20_minutes_partial_bins_and_channels(self):
        # Cutoff is 300s: 600s at 2Hz plus 600s at 4Hz gives 3Hz.
        rows = [row("Baseline", 0, 900, 2), row("Baseline", 900, 1500, 4),
                row("Post 1", 0, 60, 6), row("Post 2", 0, 120, 1.5)]
        rows += [row("Baseline", 0, 1500, 10, "B"), row("Post 1", 0, 60, 5, "B")]
        path = self.write("prep.csv", rows)
        data, epochs, bases, _ = analysis.normalize(analysis.load_preparations([path], [-200]))
        self.assertEqual(bases.iloc[0].baseline_rate_hz, 3)
        self.assertEqual(bases.iloc[0].baseline_duration_s, 1200)
        a = epochs[epochs.channel == "A-000"].set_index("epoch_label")
        self.assertEqual(a.loc["Baseline", "normalized_rate_pct"], 100)
        self.assertEqual(a.loc["Baseline", "whole_epoch_mean_rate_hz"], 2.8)
        self.assertEqual(a.loc["Post 1", "normalized_rate_pct"], 200)
        self.assertEqual(a.loc["Post 2", "normalized_rate_pct"], 50)
        b = epochs[epochs.channel == "B"].set_index("epoch_label")
        self.assertEqual(b.loc["Post 1", "normalized_rate_pct"], 50)
        self.assertEqual(data.iloc[0].normalized_rate_pct, 200 / 3)

    def test_short_zero_missing_baseline(self):
        path = self.write("prep.csv", [row("Baseline", 0, 600, 2), row("Post 1", 0, 60, 4),
                                      row("Baseline", 0, 1200, 0, "zero"), row("Post 1", 0, 60, 2, "zero"),
                                      row("Post 1", 0, 60, 3, "missing")])
        data, epochs, bases, notices = analysis.normalize(analysis.load_preparations([path], [-100]))
        self.assertEqual(len(notices), 3)
        self.assertEqual(bases.set_index("channel").loc["A-000", "baseline_duration_s"], 600)
        self.assertTrue(data[data.channel.isin(["zero", "missing"])].normalized_rate_pct.isna().all())
        self.assertEqual(data[data.channel == "zero"].spike_count.sum(), 120)

    def test_epoch_means_are_weighted_not_medians_and_preparations_equal(self):
        p1 = self.write("p1.csv", [row("Baseline", 0, 1200, 2), row("Post 1", 0, 60, 0), row("Post 1", 60, 240, 4)])
        p2 = self.write("p2.csv", [row("Baseline", 0, 1200, 10), row("Post 1", 0, 600, 5)])
        p3 = self.write("p3.csv", [row("Baseline", 0, 1200, 10), row("Post 1", 0, 600, 30)])
        _, epochs, _, _ = analysis.normalize(analysis.load_preparations([p1, p2, p3], [-200, -200, -300]))
        posts = epochs[epochs.epoch_label == "Post 1"]
        self.assertEqual(posts.normalized_rate_pct.tolist(), [150, 50, 300])
        stats = analysis.summarize(posts, ["channel", "current_na", "epoch_label"], "normalized_rate_pct").set_index("current_na")
        self.assertEqual(stats.loc[-200, "mean"], 100)
        self.assertEqual(stats.loc[-200, "n_preparations"], 2)
        self.assertEqual(stats.loc[-300, "mean"], 300)
        self.assertTrue(np.isnan(stats.loc[-300, "sem"]))

    def test_literal_epochs_missing_lengths_chunks_and_bin_boundaries(self):
        p1 = self.write("p1.csv", [row("Baseline", 0, 60, 1, recording="one"), row("Baseline", 60, 120, 2, recording="two"), row("Post 1", 0, 60, 3)])
        p2 = self.write("p2.csv", [row("Baseline", 60, 120, 3), row("Post 2", 0, 120, 4)])
        data = analysis.load_preparations([p1, p2], [-200, -200])
        self.assertEqual(data.preparation_id.nunique(), 2)
        s = analysis.summarize(data, ["channel", "current_na", "epoch_label", "aligned_start_s", "aligned_end_s"], "spike_count")
        base = s[s.epoch_label == "Baseline"].sort_values("aligned_start_s")
        self.assertEqual(base.n_preparations.tolist(), [2, 1])
        self.assertEqual(base["mean"].tolist(), [120, 120])
        self.assertEqual(set(s.epoch_label), {"Baseline", "Post 1", "Post 2"})
        self.assertTrue(s[s.epoch_label != "Baseline"]["sem"].isna().all())

    def test_standard_error(self):
        s = analysis.mean_sem([1, 3, np.nan])
        self.assertEqual(s["mean"], 2)
        self.assertEqual(s["n_preparations"], 2)
        self.assertAlmostEqual(s["sem"], 1.0, places=6)
        self.assertTrue(np.isnan(analysis.mean_sem([1])["sem"]))
        self.assertEqual(analysis.mean_sem([5, 5])["sem"], 0)
        self.assertEqual(analysis.mean_sem([np.nan])["n_preparations"], 0)
        self.assertTrue(np.isnan(analysis.mean_sem([np.nan])["sem"]))

    def test_invalid_inputs(self):
        p = self.write("overlap.csv", [row("Baseline", 0, 120, 2), row("Baseline", 60, 180, 3)])
        with self.assertRaisesRegex(ValueError, "overlapping"):
            analysis.load_preparations([p], [-200])
        with self.assertRaisesRegex(ValueError, "twice"):
            analysis.load_preparations([p, p], [-200, -200])
        with self.assertRaisesRegex(ValueError, "finite"):
            analysis.load_preparations([p], [float("nan")])

    def test_sem_counts_exclude_invalid_normalization_and_do_not_mix_windows(self):
        p1 = self.write("p1.csv", [row("Baseline", 0, 1200, 0), row("Post 1", 0, 60, 2)])
        p2 = self.write("p2.csv", [row("Baseline", 0, 1200, 2), row("Post 1", 0, 120, 4)])
        data, epochs, _, _ = analysis.normalize(analysis.load_preparations([p1, p2], [-200, -200]))
        group = analysis.summarize(epochs, ["channel", "current_na", "epoch_label"], "normalized_rate_pct")
        self.assertEqual(group.n_preparations.tolist(), [1, 1])
        self.assertTrue(group["sem"].isna().all())
        raw = analysis.summarize(data, ["channel", "current_na", "epoch_label", "aligned_start_s", "aligned_end_s"], "spike_count")
        self.assertEqual(raw[raw.epoch_label == "Post 1"].n_preparations.tolist(), [1, 1])

    def test_plot_artists_show_sem_points_sample_sizes_and_long_titles(self):
        p1 = self.write("p1.csv", [row("Baseline", 0, 60, 2), row("Baseline", 60, 120, 3), row("Post 1", 0, 60, 4)])
        p2 = self.write("p2.csv", [row("Baseline", 0, 60, 4), row("Post 1", 0, 60, 3)])
        data, epochs, _, _ = analysis.normalize(analysis.load_preparations([p1, p2], [-200, -200]))
        overlay = analysis.summarize(data, ["channel", "current_na", "epoch_label", "aligned_start_s", "aligned_end_s"], "spike_count")
        groups = analysis.summarize(epochs, ["channel", "current_na", "epoch_label"], "normalized_rate_pct")
        styles = analysis.preparation_styles(data)
        specs = analysis.plot_specs(data, epochs, "both")
        from matplotlib.container import ErrorbarContainer
        from matplotlib.collections import PolyCollection
        title = "Long custom title: " + "baseline versus post response " * 15
        for spec in specs:
            fig = analysis.render_plot(spec, data, epochs, overlay, groups, {spec["plot_id"]: title}, styles)
            try:
                ax = fig.axes[0]
                labels = ax.get_legend_handles_labels()[1]
                self.assertNotIn("95% CI", labels)
                self.assertFalse(any(isinstance(c, PolyCollection) for c in ax.collections))
                if spec["kind"] == "overlay":
                    self.assertIn("Mean ± SEM", labels)
                    self.assertTrue(any(isinstance(c, ErrorbarContainer) for c in ax.containers))
                    self.assertTrue(set(data.display_label).issubset(labels))
                else:
                    self.assertFalse(any(isinstance(c, ErrorbarContainer) for c in ax.containers))
                all_text = " ".join(t.get_text() for a in fig.axes for t in a.texts)
                self.assertIn("n = 2 preparations" if spec["kind"] == "overlay" else ("n = 2" if spec["kind"] == "current" else "n = 1 preparation"), all_text)
                if spec["kind"] == "overlay":
                    self.assertIn("n = 1 preparation", all_text)
                    self.assertEqual(len(fig.axes), 2)
                self.assertIn("Long custom title", fig._suptitle.get_text() if spec["kind"] == "current" else ax.get_title())
                if spec["kind"] == "current":
                    fig.canvas.draw()
                    self.assertGreater(fig._suptitle.get_window_extent().y0, ax.get_window_extent().y1)
                else:
                    self.assertGreater(fig.get_figheight(), 6)
            finally:
                analysis.plt.close(fig)

    def test_titles_stable_ids_folder_layout_and_safe_stale_cleanup(self):
        p1 = self.write("p1.csv", [row("Baseline", 0, 1200, 2), row("Post 1", 0, 60, 4)])
        p2 = self.write("p2.csv", [row("Baseline", 0, 1200, 3), row("Post 1", 0, 60, 5)])
        specs = analysis.discover_plot_specs([p1, p2], [-200, -200], "both")
        reverse = analysis.discover_plot_specs([p2, p1], [-200, -200], "both")
        self.assertEqual({s["plot_id"] for s in specs}, {s["plot_id"] for s in reverse})
        title = '../Invalid/file:name? A custom title'
        titles = {s["plot_id"]: title for s in specs}
        self.assertEqual(analysis.resolved_title(specs[0], {specs[0]["plot_id"]: " "}), specs[0]["default_title"])
        output = self.folder / "out"
        analysis.run_analysis([p1, p2], [-200, -200], output, "both", titles)
        self.assertFalse(list(output.glob("*.csv")))
        self.assertFalse(list(output.glob("*.png")))
        catalog = pd.read_csv(output / "CSVs/plot_catalog.csv")
        self.assertEqual(set(catalog.title), {title})
        self.assertEqual(set(catalog.plot_id), {s["plot_id"] for s in specs})
        for path in (output / "Plots").iterdir():
            self.assertRegex(path.name, r"^[a-z_]+_[a-f0-9]{20}\.(png|svg)$")
        original_names = set(p.name for p in (output / "Plots").iterdir())
        # Same selection, different mode: keep compatible plots from other tool.
        analysis.run_analysis([p1, p2], [-200, -200], output, "overlay", titles)
        self.assertEqual(set(p.name for p in (output / "Plots").iterdir()), original_names)
        self.assertEqual(set(pd.read_csv(output / "CSVs/plot_catalog.csv").plot_id), {s["plot_id"] for s in specs})
        unrelated = output / "Plots/my_own_figure.png"
        unrelated.write_bytes(b"unrelated")
        analysis.run_analysis([p1], [-100], output, "overlay")
        current = set(p.name for p in (output / "Plots").iterdir())
        self.assertTrue(original_names.isdisjoint(current))
        self.assertEqual(unrelated.read_bytes(), b"unrelated")
        self.assertEqual(len(current), 3)
        generated = next(p for p in (output / "Plots").glob("overlay_*.png"))
        generated.write_bytes(b"user edit")
        with self.assertRaisesRegex(ValueError, "modified"):
            analysis.run_analysis([p1], [-100], output, "overlay")
        self.assertEqual(generated.read_bytes(), b"user edit")

    def test_manifest_refuses_paths_outside_output(self):
        output = self.folder / "out"
        output.mkdir()
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            analysis.managed_path(output, "../unrelated.csv")
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            analysis.managed_path(output, "my_notes.txt")

    def test_standalone_entry_points_and_exports(self):
        p = self.write("prep.csv", [row("Baseline", 0, 1200, 2), row("Post 1", 0, 60, 4)])
        for script, mode in [(ROOT / "Spike Processing/overlay_spike_counts_separate_epochs.py", "overlay"),
                             (ROOT / "Plotting and Utilities/plot_baseline_post_firing_rates.py", "normalized")]:
            output = self.folder / mode
            result = subprocess.run([sys.executable, str(script), str(p), "--currents", "-200", "--out-dir", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(list((output / "Plots").glob("*.png")))
            self.assertTrue(list((output / "Plots").glob("*.svg")))
            summary = pd.read_csv(output / "CSVs/current_intensity_summary.csv")
            self.assertTrue((summary.n_preparations == 1).all())
            self.assertTrue(summary["sem"].isna().all())

    def test_standalone_title_template_and_json_file(self):
        p = self.write("prep.csv", [row("Baseline", 0, 1200, 2), row("Post 1", 0, 60, 4)])
        script = ROOT / "Spike Processing/overlay_spike_counts_separate_epochs.py"
        template = self.folder / "titles.json"
        base = [sys.executable, str(script), str(p), "--currents", "-200"]
        result = subprocess.run(base + ["--write-title-template", str(template)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        titles = json.loads(template.read_text(encoding="utf-8"))
        self.assertEqual(len(titles), 1)
        title = 'Custom "response" title / baseline and post'
        titles[next(iter(titles))] = title
        template.write_text(json.dumps(titles), encoding="utf-8")
        output = self.folder / "title_outputs"
        result = subprocess.run(base + ["--titles-file", str(template), "--out-dir", str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        catalog = pd.read_csv(output / "CSVs/plot_catalog.csv")
        self.assertEqual(catalog.title.iloc[0], title)
        self.assertEqual(catalog.plot_id.iloc[0], next(iter(titles)))


class GUIWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tkinter as tk
        spec = importlib.util.spec_from_file_location("preparation_gui_test", ROOT / "Apps/SPIE/locust_lfp_gui.py")
        cls.gui = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.gui
        spec.loader.exec_module(cls.gui)
        cls.root = tk.Tk()
        cls.root.withdraw()
        cls.app = cls.gui.LocustPipelineApp(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def test_current_publication_editors_and_execution(self):
        import tkinter as tk
        from tkinter import ttk
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            paths = []
            for i in range(3):
                path = folder / f"publication{i}.csv"
                pd.DataFrame([row("Baseline", 0, 1200, 2), row("Post 1", 0, 60, 3+i), row("Post 2", 0, 60, 2+i)]).to_csv(path, index=False)
                paths.append(path)
            app = self.app
            old = app.current_publication
            app.current_publication = dict(size="double", legend_order=[], legend_note="", note_overrides={}, horizontal_offset_na=0)
            def descendants(widget):
                for child in widget.winfo_children():
                    yield child
                    yield from descendants(child)
            def button(dialog, label):
                return next(w for w in descendants(dialog) if isinstance(w, ttk.Button) and w.cget("text") == label)
            try:
                app._set_preparation_inputs(paths[:2])
                for path in paths:
                    app.preparation_display_names[str(path)] = "Duplicated name"
                dialog = app.configure_current_legend_order()
                listing = next(w for w in descendants(dialog) if isinstance(w, tk.Listbox))
                self.assertIn(str(paths[0]), listing.get(0))
                button(dialog, "Move down").invoke()
                button(dialog, "Save order").invoke()
                app._set_preparation_inputs(paths)
                dialog = app.configure_current_legend_order()
                listing = next(w for w in descendants(dialog) if isinstance(w, tk.Listbox))
                self.assertIn(str(paths[1]), listing.get(0))
                self.assertIn(str(paths[2]), listing.get(2))
                button(dialog, "Save order").invoke()
                specs = [s for s in analysis.discover_plot_specs(paths, [-200]*3) if s["kind"] == "current"]
                dialog = app._edit_current_publication(specs)
                editors = [w for w in descendants(dialog) if isinstance(w, tk.Text)]
                editors[0].insert("1.0", "Synthetic note\nDimension: 25 µm; test area: 3 µm²")
                checks = [w for w in descendants(dialog) if isinstance(w, ttk.Checkbutton)]
                checks[0].invoke()  # Blank override hides note on this plot.
                checks[1].invoke()
                editors[2].insert("1.0", "Per-plot note")
                button(dialog, "Save appearance").invoke()
                dialog = app._edit_current_publication(specs)
                editors = [w for w in descendants(dialog) if isinstance(w, tk.Text)]
                self.assertIn("µm²", editors[0].get("1.0", "end"))
                self.assertIn("Per-plot note", editors[2].get("1.0", "end"))
                dialog.destroy()
                def synchronous(title, task, on_success=None):
                    result = task()
                    if on_success:
                        on_success(result)
                for bundled in (False, True):
                    out = folder / ("bundled" if bundled else "source")
                    app.plot_out_dir_var.set(str(out))
                    with patch.object(app, "run_background", side_effect=synchronous), patch.object(sys, "frozen", bundled, create=True):
                        app.run_preparation_plotting("normalized")
                    metadata = pd.read_csv(out / "CSVs/current_response_presentation.csv")
                    actual = {r.plot_id: json.loads(r.settings_json) for r in metadata.itertuples()}
                    self.assertEqual(actual[specs[0]["plot_id"]]["legend_note"], "")
                    self.assertEqual(actual[specs[1]["plot_id"]]["legend_note"], "Per-plot note")
                    self.assertEqual(actual[specs[0]["plot_id"]]["settings"]["legend_order"][:2], [str(paths[1]), str(paths[0])])
                dialog = app.configure_current_legend_order()
                button(dialog, "Restore default").invoke()
                button(dialog, "Save order").invoke()
                self.assertEqual(app.current_publication["legend_order"], [str(p) for p in paths])
            finally:
                app.current_publication = old

    def test_single_and_multiple_file_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            paths = []
            for i in range(2):
                p = folder / f"prep{i}.csv"
                pd.DataFrame([row("Baseline", 0, 1200, 2 + i), row("Post 1", 0, 60, 4)]).to_csv(p, index=False)
                paths.append(p)
            app = self.app
            app._set_preparation_inputs([])
            app.plot_input_var.set(str(paths[0]))
            app.preparation_default_current.set("-100")
            def synchronous(title, task, on_success=None):
                result = task()
                if on_success:
                    on_success(result)
            with patch.object(app, "run_background", side_effect=synchronous):
                app.plot_out_dir_var.set(str(folder / "single"))
                app.run_plotting()
                summary = pd.read_csv(folder / "single/CSVs/epoch_summaries.csv")
                self.assertEqual(summary.current_na.unique().tolist(), [-100])
                app._set_preparation_inputs(paths)
                app.preparation_currents[str(paths[1].resolve())].set("-450")
                app.plot_out_dir_var.set(str(folder / "multi"))
                app.show_plot_panel("epoch_overlay")
                # The Windows bundle imports scripts and calls main(), whereas
                # source launches use subprocesses. Exercise both paths.
                with patch.object(sys, "frozen", True, create=True):
                    app.run_preparation_plotting("overlay")
                summary = pd.read_csv(folder / "multi/CSVs/overlay_summary.csv")
                self.assertEqual(set(summary.current_na), {-100, -450})
                self.assertTrue((summary.n_preparations == 1).all())
                self.assertEqual(len(list((folder / "multi/Plots").glob("*.png"))), 2)
                app.plot_out_dir_var.set(str(folder / "multi_normalized"))
                app.run_preparation_plotting("normalized")
                summary = pd.read_csv(folder / "multi_normalized/CSVs/epoch_summaries.csv")
                self.assertEqual(summary.preparation_id.nunique(), 2)

    def test_title_editor_persistence_and_baseline_notices(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            path = folder / "short.csv"
            pd.DataFrame([row("Baseline", 0, 60, 2), row("Post 1", 0, 60, 4)]).to_csv(path, index=False)
            app = self.app
            app._set_preparation_inputs([path])
            app.plot_out_dir_var.set(str(folder / "out"))
            specs = analysis.discover_plot_specs([path], [-100], "normalized")
            from tkinter import ttk
            def descendants(widget):
                for child in widget.winfo_children():
                    yield child
                    yield from descendants(child)
            dialog = app._edit_preparation_titles(specs)
            entries = [w for w in descendants(dialog) if isinstance(w, ttk.Entry)]
            entries[0].insert(0, "My saved title")
            next(w for w in descendants(dialog) if isinstance(w, ttk.Button) and w.cget("text") == "Save titles").invoke()
            app._set_preparation_inputs([])
            app._set_preparation_inputs([path])
            dialog = app._edit_preparation_titles(specs)
            self.assertEqual(next(w for w in descendants(dialog) if isinstance(w, ttk.Entry)).get(), "My saved title")
            dialog.destroy()
            self.assertEqual(app.preparation_plot_titles[specs[0]["plot_id"]], "My saved title")
            def synchronous(title, task, on_success=None):
                result = task()
                if on_success:
                    on_success(result)
            with patch.object(app, "run_background", side_effect=synchronous), patch.object(self.gui.messagebox, "showwarning") as warning:
                app.run_preparation_plotting("normalized")
                warning.assert_called_once()
                self.assertIn("Only 1 minutes", warning.call_args.args[1])
            self.assertTrue((folder / "out/CSVs/baseline_references.csv").is_file())
            catalog = pd.read_csv(folder / "out/CSVs/plot_catalog.csv")
            self.assertIn("My saved title", catalog.title.tolist())

    def test_recording_names_and_experimental_gui_bundled_workflow(self):
        from tkinter import ttk
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            paths = []
            for i in range(2):
                p = folder / f"timeline{i}.csv"
                pd.DataFrame([row("Baseline", minute*60, (minute+1)*60, 2+i) for minute in range(20)] +
                             [row("Post 1", 0, 60, 4)]).to_csv(p, index=False)
                paths.append(p)
            app = self.app
            app._set_preparation_inputs(paths)
            app.preparation_default_current.set("-200")
            for current in app.preparation_currents.values():
                current.set("-200")
            def descendants(widget):
                for child in widget.winfo_children():
                    yield child
                    yield from descendants(child)
            dialog = app.configure_recording_names()
            entries = [w for w in descendants(dialog) if isinstance(w, ttk.Entry)]
            entries[0].insert(0, "Animal Alpha")
            entries[1].insert(0, "Animal Beta")
            next(w for w in descendants(dialog) if isinstance(w, ttk.Button) and w.cget("text") == "Save names").invoke()
            app._set_preparation_inputs([])
            app._set_preparation_inputs(paths)
            self.assertEqual(app.preparation_display_names[str(paths[0])], "Animal Alpha")
            dialog = app.configure_recording_symbols()
            combos = [w for w in descendants(dialog) if isinstance(w, ttk.Combobox)]
            combos[0].set("Diamond")
            next(w for w in descendants(dialog) if isinstance(w, ttk.Button) and w.cget("text") == "Save symbols").invoke()
            app._set_preparation_inputs([])
            app._set_preparation_inputs(paths)
            dialog = app.configure_recording_symbols()
            self.assertEqual(next(w for w in descendants(dialog) if isinstance(w, ttk.Combobox)).get(), "Diamond")
            dialog.destroy()
            app.timeline_show_gap_var.set(False)
            app.timeline_gap_var.set("15")
            app.reference_start_var.set("10")
            app.reference_end_var.set("20")
            try:
                app.show_plot_panel("experimental_timeline")
                app.plot_out_dir_var.set(str(folder / "out"))
                def synchronous(title, task, on_success=None):
                    result = task()
                    if on_success:
                        on_success(result)
                with patch.object(app, "run_background", side_effect=synchronous), patch.object(sys, "frozen", True, create=True):
                    app.run_preparation_plotting("timeline")
                windows = pd.read_csv(folder / "out/CSVs/experimental_time_windows.csv")
                self.assertEqual(set(windows.display_name), {"Animal Alpha", "Animal Beta"})
                self.assertEqual(set(windows.loc[windows.display_name == "Animal Alpha", "recording_symbol"]), {"D"})
                self.assertTrue(windows.timing_compressed.all())
                self.assertTrue((windows.baseline_duration_s == 600).all())
                self.assertEqual(len(list((folder / "out/Plots").glob("time_course_*.png"))), 2)
            finally:
                app.timeline_show_gap_var.set(True)
                app.timeline_gap_var.set("20")
                app.reference_start_var.set("")
                app.reference_end_var.set("")


if __name__ == "__main__":
    unittest.main()
