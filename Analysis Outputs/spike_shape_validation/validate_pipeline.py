"""Reproducible source-GUI and real-filter export validation."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tkinter as tk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gui = load("shape_gui_validation", ROOT / "Apps/SPIE/locust_lfp_gui.py")
root = tk.Tk()
root.withdraw()
try:
    app = gui.LocustPipelineApp(root)
    root.update_idletasks()
    assert app.spike_shape_vars["mode"].get() == "off"
    fs = 20000
    t = np.arange(10000)/fs
    x = np.random.default_rng(11).normal(0, 1, len(t))
    for time, amplitude in ((.05, -180), (.0508, -150), (.15, -180), (.25, -170)):
        x += amplitude*np.exp(-.5*((t-time)/.00008)**2)
    source = OUT / "rec_Baseline_001.csv"
    pd.DataFrame(dict(time_s=t, A=x)).to_csv(source, index=False)
    totals = {}
    for mode in ("off", "flag_only", "reject"):
        app.spike_shape_vars["mode"].set(mode)
        app.spike_polarity_var.set("neg")
        request = app._spike_pipeline_request(paths=[str(source)], labels={str(source):"Baseline"},
            out_dir=str(OUT / mode), recording_metadata=[])
        assert request.shape_settings["mode"] == mode
        result = gui.run_shared_spike_pipeline(request)
        events = pd.read_csv(result["spike_events_csv"])
        counts = pd.read_csv(result["spike_count_csv"])
        assert counts.spike_count.sum() == len(events)
        assert result["spike_processing_provenance"]["shape_settings"]["mode"] == mode
        totals[mode] = len(events)
    assert totals["flag_only"] == totals["off"]
    assert totals["reject"] < totals["off"]
    waveform = load("shape_waveform_validation", ROOT / "waveform_analysis.py")
    waveform.CSV_DIR = str(OUT)
    waveform.CHANNELS = ["A"]
    waveform.main(["--shape-mode", "reject"])
    print("VALIDATION", totals)
    (OUT / "validation_summary.json").write_text(json.dumps(dict(gui_controls="passed", totals=totals,
        standalone_waveform_analysis="passed"), indent=2), encoding="utf-8")
finally:
    root.destroy()
