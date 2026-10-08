from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Plotting and Utilities"))
import h2o2_recordings as h
import preparation_spike_analysis as analysis


class H2O2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.paths = []
        for name, epochs in (("control", [(" baseline ", 1200, 4), ("DURING", 60, 8), ("Post", 60, 2)]),
                             ("current", [("Baseline", 1200, 10), ("Post 1", 60, 5)])):
            path = self.folder / (name + ".csv")
            pd.DataFrame([dict(epoch_label=e, channel="A", window_start_s=0,
                               window_end_s=d, spike_count=d*r) for e, d, r in epochs]).to_csv(path, index=False)
            self.paths.append(str(path.resolve()))
        self.controls = [(self.paths[0], h.H2O2)]

    def test_case_insensitive_same_file_and_missing_measurement(self):
        rates = h.control_rates(self.controls)
        self.assertEqual(rates.normalized_rate_pct.tolist(), [200., 50.])
        self.assertEqual(rates.pre_h2o2_baseline_hz.tolist(), [4., 4.])
        self.assertEqual(rates.locust_id.nunique(), 1)
        self.assertEqual(rates.source_file.tolist(), rates.baseline_file.tolist())
        frame = pd.read_csv(self.paths[0])
        frame[frame.epoch_label != "Post"].to_csv(self.paths[0], index=False)
        self.assertEqual(len(h.control_rates(self.controls)), 1)

    def test_reject_missing_zero_baseline_unknown_labels_and_duplicate_files(self):
        with self.assertRaisesRegex(ValueError, "only once"):
            h.control_rates(self.controls * 2)
        frame = pd.read_csv(self.paths[0])
        frame.loc[0, "epoch_label"] = "unknown"
        frame.to_csv(self.paths[0], index=False)
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            h.control_rates(self.controls)
        frame.loc[0, "epoch_label"] = "baseline"
        frame.loc[0, "spike_count"] = 0
        frame.to_csv(self.paths[0], index=False)
        with self.assertRaisesRegex(ValueError, "nonpositive"):
            h.control_rates(self.controls)
        frame.iloc[1:].to_csv(self.paths[0], index=False)
        with self.assertRaisesRegex(ValueError, "missing"):
            h.control_rates(self.controls)

    def test_separate_preparations_and_baselines(self):
        second = self.folder / "second.csv"
        frame = pd.read_csv(self.paths[0])
        frame.loc[0, "spike_count"] *= 2
        frame.to_csv(second, index=False)
        rates = h.control_rates([*self.controls, (str(second), h.H2O2)])
        self.assertEqual(rates.locust_id.nunique(), 2)
        self.assertEqual(rates.normalized_rate_pct.tolist(), [200, 50, 100, 25])
        self.assertTrue((rates.groupby("locust_id").source_file.nunique() == 1).all())

    def test_mixed_cli_and_exports_without_pairing_config(self):
        output = self.folder / "output"
        analysis.cli([*self.paths, "--currents", "H2O2", "-100", "--out-dir", str(output)])
        epochs = pd.read_csv(output / "CSVs/epoch_summaries.csv")
        self.assertEqual(set(epochs.current_na), {-100})
        self.assertEqual(epochs.loc[epochs.epoch_label == "Post 1", "normalized_rate_pct"].tolist(), [50.])
        controls = output / "H2O2_positive_control"
        rates = pd.read_csv(controls / "h2o2_normalized_rates.csv")
        self.assertEqual(rates.normalized_rate_pct.tolist(), [200., 50.])
        catalog = pd.read_csv(controls / "plot_catalog.csv")
        with Image.open(controls / catalog.png.iloc[0]) as image:
            self.assertEqual(image.size, (5400, 2880))
            self.assertAlmostEqual(image.info["dpi"][0], 600, places=2)
        svg = (controls / catalog.svg.iloc[0]).read_text(encoding="utf-8")
        self.assertIn(h.H2O2, svg)
        self.assertNotIn(rates.locust_id.iloc[0], svg)

    def test_reference_range_and_channels_are_separate(self):
        frame = pd.read_csv(self.paths[0])
        frame = pd.concat([frame.iloc[1:], pd.DataFrame([
            dict(epoch_label="baseline", channel="A", window_start_s=0, window_end_s=60, spike_count=120),
            dict(epoch_label="BASELINE", channel="A", window_start_s=60, window_end_s=120, spike_count=360)])], ignore_index=True)
        second = frame.copy()
        second["channel"] = "B"
        second.loc[second.epoch_label.str.lower() == "baseline", "spike_count"] *= 2
        pd.concat([frame, second]).to_csv(self.paths[0], index=False)
        rates = h.control_rates(self.controls, baseline_range_min=[1, 2])
        np.testing.assert_allclose(rates.pre_h2o2_baseline_hz, [6, 6, 12, 12])
        np.testing.assert_allclose(rates.normalized_rate_pct, [100*8/6, 100*2/6, 100*8/12, 100*2/12])
        self.assertEqual(rates.baseline_duration_s.tolist(), [60]*4)

    def test_gui_single_preparation_and_bundled_completion(self):
        import tkinter as tk
        from tkinter import ttk
        spec = importlib.util.spec_from_file_location("h2o2_gui_test", ROOT / "Apps/SPIE/locust_lfp_gui.py")
        gui = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = gui
        spec.loader.exec_module(gui)
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        app = gui.LocustPipelineApp(root)
        app._set_preparation_inputs(self.paths[:1])
        app.preparation_currents[self.paths[0]].set(h.H2O2)
        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)
        combos = [w for w in descendants(app.preparation_rows) if isinstance(w, ttk.Combobox)]
        self.assertTrue(all(h.H2O2 in w.cget("values") and h.DURING not in w.cget("values") and h.AFTER not in w.cget("values") for w in combos))
        dialog = app.configure_h2o2_recordings()
        self.assertEqual(len([w for w in descendants(dialog) if isinstance(w, ttk.Entry)]), 1)
        next(w for w in descendants(dialog) if isinstance(w, ttk.Button) and w.cget("text") == "Save").invoke()
        command, _ = app._preparation_plot_command("normalized")
        self.assertIn(h.H2O2, command)
        self.assertEqual(json.loads(command[command.index("--h2o2-json") + 1]), {"concentration": ""})
        app._set_preparation_inputs(self.paths[:1])
        self.assertEqual(app._preparation_inputs()[1], [h.H2O2])
        app.plot_out_dir_var.set(str(self.folder / "gui_control_only"))
        app._run_logged_subprocess = lambda cmd, label: app._run_bundled_script(cmd[1], cmd[2:], label)
        app.run_background = lambda label, task, on_success: on_success(task())
        app.run_preparation_plotting("normalized")
        self.assertTrue((self.folder / "gui_control_only/H2O2_positive_control/plot_catalog.csv").exists())


if __name__ == "__main__":
    unittest.main()
