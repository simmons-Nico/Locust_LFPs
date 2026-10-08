"""Exercise manual epoch labels and file refresh using real Tk widgets."""
import sys
import tempfile
import tkinter as tk
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Apps" / "SPIE"))
from locust_lfp_gui import LocustPipelineApp


class SpikeFileLabelTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.app = LocustPipelineApp.__new__(LocustPipelineApp)
        self.app.recordings = []
        self.app.log = lambda text: None
        self.app.recording_listbox = tk.Listbox(self.root, selectmode="extended", exportselection=False)
        self.app.lfp_recording_listbox = tk.Listbox(self.root, selectmode="extended", exportselection=False)
        for name, value in {
            "spike_csv_dir": str(self.folder), "spike_glob": "*.csv",
            "selected_epoch": "", "selection_count": "", "spike_out_dir": "",
            "spike_window": "1 min", "spike_polarity": "both",
            "lfp_preparation_id": "Preparation 1", "lfp_channels": "",
            "lfp_reference_scheme": "none", "lfp_signal_unit": "uV",
        }.items():
            setattr(self.app, name + "_var", tk.StringVar(self.root, value=value))
        self.write("b.csv")
        self.write("c.csv")
        self.app.refresh_spike_files()

    def write(self, name):
        (self.folder / name).write_text("time,signal\n0,0\n")

    def label(self, index, label):
        self.app.recording_listbox.selection_clear(0, "end")
        self.app.recording_listbox.selection_set(index)
        self.app.on_recording_select()
        self.app.selected_epoch_var.set(label)
        self.app.apply_epoch_label()

    def test_initial_baseline_and_each_file_can_be_renamed(self):
        self.assertEqual(self.app.recordings[0]["epoch_label"], "Baseline")
        for index, label in enumerate(("Perfusion", "Post 2")):
            self.label(index, label)
            item = self.app.recordings[index]
            self.assertEqual(item["label"], label)
            self.assertEqual(item["epoch_label"], label)
            self.assertEqual(item["phase"], ("treatment", "post")[index])
            self.assertIn(f"epoch={label}", self.app.recording_listbox.get(index))
            self.assertIn(f"epoch={label}", self.app.lfp_recording_listbox.get(index))
        request = self.app._spike_pipeline_request()
        self.assertEqual(list(request.labels_by_path.values()), ["Perfusion", "Post 2"])
        self.assertEqual([item["epoch_label"] for item in request.recording_metadata], ["Perfusion", "Post 2"])

    def test_refresh_preserves_labels_metadata_order_and_selection(self):
        self.label(0, "Custom epoch")
        self.app.recordings[0]["preparation_id"] = "Locust 1"
        self.app.recordings.reverse()
        self.app._refresh_recording_listbox([1])
        self.write("a.csv")  # Sorts before the existing baseline file.
        self.app.refresh_spike_files()
        self.assertEqual([Path(item["path"]).name for item in self.app.recordings], ["c.csv", "b.csv", "a.csv"])
        self.assertEqual(self.app.recordings[1]["epoch_label"], "Custom epoch")
        self.assertEqual(self.app.recordings[1]["preparation_id"], "Locust 1")
        self.assertEqual(self.app.recordings[2]["epoch_label"], "")
        self.assertEqual(self.app.recording_listbox.curselection(), (1,))
        self.label(2, "Baseline 2")
        self.app.refresh_spike_files()
        self.assertEqual(self.app.recordings[2]["epoch_label"], "Baseline 2")
        (self.folder / "c.csv").unlink()
        self.app.refresh_spike_files()
        self.assertEqual([item["recording_order"] for item in self.app.recordings], ["1", "2"])

    def test_cleared_baseline_stays_cleared_after_refresh(self):
        self.label(0, "")
        self.app.refresh_spike_files()
        item = self.app.recordings[0]
        self.assertEqual((item["label"], item["epoch_label"], item["phase"]), ("", "", ""))


if __name__ == "__main__":
    unittest.main()
