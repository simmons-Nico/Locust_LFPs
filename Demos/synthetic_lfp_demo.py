#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create and run a small synthetic SPIE LFP end-to-end demo."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for module_dir in (
    PROJECT_ROOT / "LFP Processing",
    PROJECT_ROOT / "Spike Processing",
):
    sys.path.insert(0, str(module_dir))

from lfp_feature_extraction import LFPConfig, process_lfp_recordings
from lfp_plotting import plot_lfp_features
from spike_lfp_coupling import SpikeLFPCouplingConfig, calculate_spike_lfp_coupling


def create_demo_dataset(root: str | Path = "synthetic_lfp_demo_dataset") -> dict[str, str]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    fs = 500.0
    duration_s = 60.0
    t = np.arange(0, duration_s, 1.0 / fs)
    rng = np.random.default_rng(22)
    outputs = {}
    for name, phase, amp, slow_amp in (
        ("baseline", "baseline", 12.0, 4.0),
        ("post_h2o2", "post", 18.0, 7.0),
    ):
        ch1 = amp * np.sin(2.0 * np.pi * 10.0 * t) + slow_amp * np.sin(2.0 * np.pi * 3.0 * t) + rng.normal(0, 1.2, size=t.size)
        ch2 = 0.65 * amp * np.sin(2.0 * np.pi * 16.0 * t + 0.4) + rng.normal(0, 1.0, size=t.size)
        path = root / f"{name}.csv"
        pd.DataFrame({"time_s": t, "A-001": ch1, "A-002": ch2}).to_csv(path, index=False)
        outputs[phase] = str(path)
    spike_rows = []
    for rec_name, phase in (("baseline", "baseline"), ("post_h2o2", "post")):
        spike_times = np.arange(0.025, duration_s, 0.10)
        if phase == "post":
            spike_times = np.sort(np.r_[spike_times, np.arange(0.055, duration_s, 0.20)])
        for idx, spike_time in enumerate(spike_times, start=1):
            spike_rows.append(
                {
                    "recording_index": 1 if phase == "baseline" else 2,
                    "recording_id": rec_name,
                    "recording_name": rec_name,
                    "epoch_label": phase,
                    "channel": "A-001",
                    "unit_id": "threshold_mua",
                    "event_index": idx,
                    "spike_time_s": float(spike_time),
                    "sample_index": int(round(spike_time * fs)),
                    "polarity": "pos",
                    "amplitude_uv": 90.0,
                    "qc_pass": True,
                    "qc_reason": "synthetic_threshold_mua",
                }
            )
    spike_path = root / "spike_events.csv"
    pd.DataFrame(spike_rows).to_csv(spike_path, index=False)
    outputs["spike_events"] = str(spike_path)
    return outputs


def run_demo(root: str | Path = "synthetic_lfp_demo_dataset", output_dir: str | Path = "synthetic_lfp_demo_output") -> dict[str, object]:
    files = create_demo_dataset(root)
    output_dir = Path(output_dir)
    records = [
        {
            "source_file": files["baseline"],
            "recording_order": 1,
            "recording_id": "baseline",
            "experiment_id": "synthetic_demo",
            "animal_id": "animal_001",
            "preparation_id": "prep_001",
            "phase": "baseline",
            "treatment": "saline",
            "channel": "A-001,A-002",
            "reference_scheme": "none",
            "electrode_type": "synthetic",
            "impedance_1khz_ohm": "500000",
            "sampling_rate_hz": "500",
            "signal_unit": "uV",
            "reference_stability": "stable",
            "vehicle_perfusion_control": "true",
        },
        {
            "source_file": files["post"],
            "recording_order": 2,
            "recording_id": "post_h2o2",
            "experiment_id": "synthetic_demo",
            "animal_id": "animal_001",
            "preparation_id": "prep_001",
            "phase": "post",
            "treatment": "H2O2 synthetic",
            "concentration_value": "1",
            "concentration_unit": "mM",
            "application_onset_s": "10",
            "washout_onset_s": "",
            "channel": "A-001,A-002",
            "reference_scheme": "none",
            "electrode_type": "synthetic",
            "impedance_1khz_ohm": "500000",
            "sampling_rate_hz": "500",
            "signal_unit": "uV",
            "reference_stability": "stable",
            "vehicle_perfusion_control": "",
        },
    ]
    lfp_paths = process_lfp_recordings(
        records,
        LFPConfig(
            output_dir=output_dir,
            channels=["A-001", "A-002"],
            analysis_window_s=30.0,
            welch_segment_s=6.0,
            exploratory_band_hz=(0.5, 100.0),
            primary_band_hz=(1.0, 40.0),
            notch_frequency_hz=None,
            filter_edge_s=1.0,
            minimum_valid_fraction=0.90,
        ),
    )
    coupling_paths = calculate_spike_lfp_coupling(
        files["spike_events"],
        records,
        SpikeLFPCouplingConfig(
            output_dir=output_dir / "spike_lfp_coupling",
            lfp_channels=["A-001"],
            phase_band_hz=(8.0, 12.0),
            window_s=30.0,
            min_spikes=20,
            surrogate_count=50,
            notch_frequency_hz=None,
        ),
    )
    plot_paths = plot_lfp_features(
        lfp_paths["features_csv"],
        {
            "output_dir": str(output_dir / "lfp_plots"),
            "ppc_csv": coupling_paths["ppc_csv"],
        },
    )
    return {"dataset": files, "lfp": lfp_paths, "coupling": coupling_paths, "plots": plot_paths}


def main() -> None:
    result = run_demo()
    print("Synthetic demo complete.")
    print(result)


if __name__ == "__main__":
    main()
