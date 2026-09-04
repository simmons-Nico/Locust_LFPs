#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compatibility wrapper for SPIE LFP analysis.

The old version of this file was a whole-file plotting script.  The LFP backend
now lives in import-safe modules:

    lfp_feature_extraction.py
    lfp_plotting.py
    spike_lfp_coupling.py

Use this wrapper for quick command-line extraction and plotting.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from lfp_feature_extraction import LFPConfig, parse_band_spec, parse_channel_list, process_lfp_recordings
from lfp_plotting import plot_lfp_features


def _parse_pair(text: str) -> tuple[float, float]:
    parts = [part for part in re.split(r"\s*[-:,]\s*", text.strip()) if part]
    if len(parts) != 2:
        raise ValueError(f"Expected frequency range like 1-40, got {text!r}.")
    return float(parts[0]), float(parts[1])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract and plot windowed SPIE LFP features.")
    parser.add_argument("inputs", nargs="*", help="Raw wide CSV file(s).")
    parser.add_argument("--manifest", default="", help="Optional explicit LFP manifest CSV.")
    parser.add_argument("--out-dir", default="lfp_analysis", help="Output folder.")
    parser.add_argument("--channels", default="", help="Comma-separated LFP channels; blank means all numeric signal channels.")
    parser.add_argument("--phase", default="", help="Explicit phase for positional inputs.")
    parser.add_argument("--animal-id", default="", help="Animal id for positional inputs.")
    parser.add_argument("--preparation-id", default="", help="Preparation id for positional inputs.")
    parser.add_argument("--window-sec", type=float, default=60.0)
    parser.add_argument("--primary-band", default="1-40")
    parser.add_argument("--exploratory-band", default="0.5-100")
    parser.add_argument("--marker-bands", default="")
    parser.add_argument("--welch-segment-sec", type=float, default=4.0)
    parser.add_argument("--notch-frequency", default="50")
    parser.add_argument("--downsample-hz", default="")
    parser.add_argument("--signal-unit", default="uV")
    parser.add_argument("--no-plots", action="store_true", help="Only extract features.")
    return parser


def _recordings_from_manifest(path: str) -> list[dict]:
    import pandas as pd

    return pd.read_csv(path).to_dict(orient="records")


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.manifest:
        recordings = _recordings_from_manifest(args.manifest)
    else:
        recordings = [
            {
                "source_file": str(Path(path)),
                "recording_order": index,
                "recording_id": Path(path).stem,
                "phase": args.phase,
                "animal_id": args.animal_id,
                "preparation_id": args.preparation_id,
            }
            for index, path in enumerate(args.inputs, start=1)
        ]
    cfg = LFPConfig(
        output_dir=args.out_dir,
        channels=parse_channel_list(args.channels),
        analysis_window_s=args.window_sec,
        primary_band_hz=_parse_pair(args.primary_band),
        exploratory_band_hz=_parse_pair(args.exploratory_band),
        marker_bands=parse_band_spec(args.marker_bands),
        welch_segment_s=args.welch_segment_sec,
        notch_frequency_hz=None if str(args.notch_frequency).strip().lower() in {"", "none", "0"} else float(args.notch_frequency),
        downsample_hz=None if str(args.downsample_hz).strip().lower() in {"", "none", "0"} else float(args.downsample_hz),
        signal_unit=args.signal_unit,
    )
    paths = process_lfp_recordings(recordings, cfg)
    if not args.no_plots:
        plot_lfp_features(paths["features_csv"], {"output_dir": str(Path(args.out_dir) / "lfp_plots")})


if __name__ == "__main__":
    main()
