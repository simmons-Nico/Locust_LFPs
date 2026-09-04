#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import numpy as np
import pandas as pd
from math import isclose
from load_intan_rhd_format import read_data

# ------------- options you may tweak -------------
INPUT_DIR   = r"C:\Users\simmons\Desktop\Exploring PSDs"
OUT_CSV     = r"C:\Users\simmons\Desktop\Exploring PSDs\all_channels_amplifier_raw.csv"
TARGET_CHANNELS = None         # None = all channels; or list like ["A-008","A-010"]
DIFF_PAIRS      = []           # e.g., [("A-008","A-010")] to add "A-008-A-010" column
CONTINUOUS_TIME = True         # accumulate time across multiple files
TIME_COL_NAME   = "time_s"
# --------------------------------------------------

def _normalize_amp_shape(amp: np.ndarray, channels: list[str]) -> np.ndarray:
    """Ensure (channels, samples) shape."""
    amp = np.asarray(amp)
    C = len(channels)
    if amp.ndim != 2:
        raise ValueError(f"Signal array must be 2D, got shape {amp.shape}")
    if amp.shape[0] == C:
        return amp
    if amp.shape[1] == C:
        return amp.T
    raise ValueError(f"Channel count ({C}) doesn't match shape {amp.shape}")

def _list_rhd_files(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".rhd")])

def _first_pass_union_channels_and_fs(input_dir, target_channels=None, diff_pairs=()):
    """Scan all files to get union of channel names and consistent fs."""
    files = _list_rhd_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No .rhd files found in: {input_dir}")
    fs_master = None
    union = set()
    for fname in files:
        fpath = os.path.join(input_dir, fname)
        d = read_data(fpath)
        fs = float(d["frequency_parameters"]["amplifier_sample_rate"])
        if fs_master is None:
            fs_master = fs
        elif not isclose(fs, fs_master, rel_tol=1e-9, abs_tol=1e-9):
            raise RuntimeError(f"Sample-rate mismatch: {fname} has fs={fs}, expected {fs_master}")
        ch_names = [ch["native_channel_name"] for ch in d["amplifier_channels"]]
        if target_channels is None:
            use_names = ch_names
        else:
            use_names = [n for n in ch_names if n in set(target_channels)]
        union.update(use_names)
    # add diff-pair names
    for a, b in diff_pairs:
        union.add(f"{a}-{b}")
    return fs_master, sorted(union)

def _open_csv_writer(path, header):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(header)
    return f, w

def combine_all_amplifier_to_one_csv(
    input_dir: str,
    out_csv: str,
    target_channels=None,
    diff_pairs=(),
    continuous_time=True,
    time_col_name="time_s",
):
    """
    Create a single CSV with columns:
        time_s, <ch1>, <ch2>, ..., <A-B> (optional)
    containing raw amplified traces (no filtering, no downsampling).
    """
    files = _list_rhd_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No .rhd files found in: {input_dir}")

    # First pass: union of channels and consistent fs
    fs, master_cols_no_time = _first_pass_union_channels_and_fs(
        input_dir, target_channels=target_channels, diff_pairs=diff_pairs
    )

    # Header: time + channels (including diff names)
    header = [time_col_name] + master_cols_no_time
    f, w = _open_csv_writer(out_csv, header)

    t_offset = 0.0
    try:
        for idx, fname in enumerate(files, 1):
            fpath = os.path.join(input_dir, fname)
            print(f"[{idx}/{len(files)}] Reading {fname} ...")
            d = read_data(fpath)
            fs_in = float(d["frequency_parameters"]["amplifier_sample_rate"])
            if not isclose(fs_in, fs, rel_tol=1e-9, abs_tol=1e-9):
                raise RuntimeError(f"{fname}: fs mismatch {fs_in} vs {fs}")

            ch_list = [ch["native_channel_name"] for ch in d["amplifier_channels"]]
            amp = _normalize_amp_shape(d["amplifier_data"], ch_list)  # (C x T)

            # Choose channels
            if target_channels is None:
                chosen = ch_list
                amp_idx = np.arange(len(ch_list))
            else:
                # keep only requested channels that exist in this file
                name_to_idx = {n:i for i,n in enumerate(ch_list)}
                chosen = [n for n in target_channels if n in name_to_idx]
                amp_idx = np.array([name_to_idx[n] for n in chosen], dtype=int)

            # raw samples for chosen channels
            X = amp[amp_idx, :] if len(chosen) else np.empty((0, amp.shape[1]))
            T = X.shape[1] if X.size else amp.shape[1]
            # build dict for current file => values per master column
            # start with zeros; we’ll fill by name
            # (we will stream row-by-row to keep memory usage reasonable)
            # Build a mapping for chosen single channels:
            single_map = {name: X[i, :] for i, name in enumerate(chosen)}

            # build bipolar if requested and both present
            diff_map = {}
            for a, b in diff_pairs:
                if a in single_map and b in single_map:
                    diff_map[f"{a}-{b}"] = single_map[a] - single_map[b]

            # time vector for this file
            t_local = np.arange(T, dtype=np.float64) / fs
            if continuous_time:
                t_local = t_local + t_offset

            # stream rows
            for j in range(T):
                row = [t_local[j]]
                for col_name in master_cols_no_time:
                    if col_name in single_map:
                        row.append(float(single_map[col_name][j]))
                    elif col_name in diff_map:
                        row.append(float(diff_map[col_name][j]))
                    else:
                        row.append("")  # keep CSV small; could also use np.nan
                w.writerow(row)

            # advance the global time offset
            if T > 0 and continuous_time:
                t_offset = float(t_local[-1] + 1.0/fs)

        print(f"✅ Combined raw amplifier CSV saved: {out_csv}")
        print(f"   Columns: {len(header)} (time + {len(header)-1} channels)")
        print(f"   Sample rate: {fs:.6f} Hz")

    finally:
        f.close()


# -------- entry point (run this file) --------
if __name__ == "__main__":
    combine_all_amplifier_to_one_csv(
        input_dir=INPUT_DIR,
        out_csv=OUT_CSV,
        target_channels=TARGET_CHANNELS,  # None = all channels
        diff_pairs=DIFF_PAIRS,            # optional bipolar columns
        continuous_time=CONTINUOUS_TIME,
        time_col_name=TIME_COL_NAME,
    )
