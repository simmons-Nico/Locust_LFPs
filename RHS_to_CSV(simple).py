#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import numpy as np
from math import isclose
from load_intan_rhs_format import read_data

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


def _list_rhs_files(folder: str) -> list[str]:
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".rhs")])


def _open_csv_writer(path: str, header: list[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(header)
    return f, w


def _summary_path_from_csv(out_csv: str) -> str:
    base, _ = os.path.splitext(out_csv)
    return base + "_summary.txt"


def _safe_get(dct, keys, default=None):
    """
    Safely fetch nested values:
      _safe_get(d, ["frequency_parameters","amplifier_sample_rate"])
    """
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _write_global_metadata(sf, input_dir, out_csv, continuous_time, time_col_name, target_channels, diff_pairs):
    sf.write("RHS → CSV conversion summary (with Intan metadata)\n")
    sf.write("=" * 52 + "\n\n")
    sf.write(f"Input directory:\n  {input_dir}\n\n")
    sf.write(f"Output CSV:\n  {out_csv}\n\n")
    sf.write(f"Summary TXT:\n  {_summary_path_from_csv(out_csv)}\n\n")
    sf.write(f"Time column:\n  {time_col_name}\n\n")
    sf.write(f"Continuous time:\n  {continuous_time}\n\n")
    sf.write("Target channels:\n")
    sf.write("  ALL\n\n" if target_channels is None else f"  {target_channels}\n\n")
    sf.write("Differential pairs:\n")
    sf.write("  None\n\n" if not diff_pairs else f"  {diff_pairs}\n\n")
    sf.write("-" * 60 + "\n\n")


def _write_intan_metadata(sf, d, prefix="  "):
    """
    Write out useful Intan metadata if present.
    The RHS reader returns a dict; keys can vary by recording settings/version.
    This function tries a set of common fields and prints whatever exists.
    """
    sf.write(f"{prefix}Intan metadata (if present):\n")

    # Frequency parameters (very common)
    fp = d.get("frequency_parameters", {})
    if isinstance(fp, dict) and fp:
        sf.write(f"{prefix}  frequency_parameters:\n")
        for k in sorted(fp.keys()):
            sf.write(f"{prefix}    {k}: {_fmt(fp.get(k))}\n")
    else:
        sf.write(f"{prefix}  frequency_parameters: N/A\n")

    # Notch filter info (varies by implementation)
    notch = (
        _safe_get(d, ["notch_filter_frequency"], None)
        or _safe_get(d, ["notch_filter_hz"], None)
        or _safe_get(fp, ["notch_filter_frequency"], None)
    )
    if notch is not None:
        sf.write(f"{prefix}  notch_filter_frequency: {_fmt(notch)}\n")

    # Board / recording info – may or may not exist depending on reader
    for k in [
        "board_mode",
        "reference_channel",
        "notes",
        "version",
        "sample_rate",
        "num_amplifier_channels",
    ]:
        if k in d:
            sf.write(f"{prefix}  {k}: {_fmt(d.get(k))}\n")

    # Channel metadata (amplifier channels)
    amp_ch = d.get("amplifier_channels", None)
    if isinstance(amp_ch, list) and amp_ch:
        sf.write(f"{prefix}  amplifier_channels: {len(amp_ch)}\n")

        # Print a small set of useful per-channel fields (first few only if many)
        fields_of_interest = [
            "native_channel_name",
            "custom_channel_name",
            "electrode_impedance_magnitude",
            "electrode_impedance_phase",
            "chip_channel",
            "port_name",
            "board_stream",
        ]
        preview_n = min(10, len(amp_ch))
        sf.write(f"{prefix}  amplifier_channels preview (first {preview_n}):\n")
        for i in range(preview_n):
            ch = amp_ch[i]
            if not isinstance(ch, dict):
                continue
            name = ch.get("native_channel_name", f"ch{i}")
            sf.write(f"{prefix}    - {name}\n")
            for f in fields_of_interest:
                if f in ch and f != "native_channel_name":
                    sf.write(f"{prefix}        {f}: {_fmt(ch.get(f))}\n")
    else:
        sf.write(f"{prefix}  amplifier_channels: N/A\n")

    sf.write("\n")


def _first_pass_union_channels_and_fs(input_dir, target_channels=None, diff_pairs=()):
    """Scan all files to get union of channel names and consistent fs."""
    files = _list_rhs_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No .rhs files found in: {input_dir}")

    fs_master = None
    union = set()

    for fname in files:
        fpath = os.path.join(input_dir, fname)
        d = read_data(fpath)

        fs = float(_safe_get(d, ["frequency_parameters", "amplifier_sample_rate"], None))
        if fs_master is None:
            fs_master = fs
        elif not isclose(fs, fs_master, rel_tol=1e-9, abs_tol=1e-9):
            raise RuntimeError(f"Sample-rate mismatch: {fname} has fs={fs}, expected {fs_master}")

        ch_names = [ch["native_channel_name"] for ch in d["amplifier_channels"]]

        if target_channels is None:
            use_names = ch_names
        else:
            want = set(target_channels)
            use_names = [n for n in ch_names if n in want]

        union.update(use_names)

    # add diff-pair names
    for a, b in diff_pairs:
        union.add(f"{a}-{b}")

    return fs_master, sorted(union)


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
    containing raw amplified traces (no filtering, no downsampling),
    plus a TXT summary file with per-file durations and Intan metadata.
    """
    files = _list_rhs_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No .rhs files found in: {input_dir}")

    # First pass: union of channels and consistent fs
    fs, master_cols_no_time = _first_pass_union_channels_and_fs(
        input_dir, target_channels=target_channels, diff_pairs=diff_pairs
    )

    # Header: time + channels (including diff names)
    header = [time_col_name] + master_cols_no_time
    f_csv, w = _open_csv_writer(out_csv, header)

    # Summary file
    summary_path = _summary_path_from_csv(out_csv)
    sf = open(summary_path, "w", encoding="utf-8")

    _write_global_metadata(sf, input_dir, out_csv, continuous_time, time_col_name, target_channels, diff_pairs)
    sf.write(f"Master sample rate (Hz): {fs}\n")
    sf.write(f"Master output columns (excluding time): {len(master_cols_no_time)}\n")
    sf.write("-" * 60 + "\n\n")

    t_offset = 0.0
    total_samples = 0
    total_files_ok = 0

    try:
        for idx, fname in enumerate(files, 1):
            fpath = os.path.join(input_dir, fname)
            print(f"[{idx}/{len(files)}] Reading {fname} ...")

            d = read_data(fpath)

            fs_in = float(_safe_get(d, ["frequency_parameters", "amplifier_sample_rate"], None))
            if not isclose(fs_in, fs, rel_tol=1e-9, abs_tol=1e-9):
                raise RuntimeError(f"{fname}: fs mismatch {fs_in} vs {fs}")

            ch_list = [ch["native_channel_name"] for ch in d["amplifier_channels"]]
            amp = _normalize_amp_shape(d["amplifier_data"], ch_list)  # (C x T)

            # Choose channels
            if target_channels is None:
                chosen = ch_list
                amp_idx = np.arange(len(ch_list))
            else:
                name_to_idx = {n: i for i, n in enumerate(ch_list)}
                chosen = [n for n in target_channels if n in name_to_idx]
                amp_idx = np.array([name_to_idx[n] for n in chosen], dtype=int)

            X = amp[amp_idx, :] if len(chosen) else np.empty((0, amp.shape[1]))
            T = X.shape[1] if X.size else amp.shape[1]
            total_samples += int(T)

            # Mapping name -> samples
            single_map = {name: X[i, :] for i, name in enumerate(chosen)}

            # Differential pairs
            diff_map = {}
            for a, b in diff_pairs:
                if a in single_map and b in single_map:
                    diff_map[f"{a}-{b}"] = single_map[a] - single_map[b]

            # Time vector for this file
            t_local = np.arange(T, dtype=np.float64) / fs
            if continuous_time:
                t_local = t_local + t_offset

            # ---- Write per-file summary block (includes Intan metadata) ----
            duration_s = (T / fs) if fs else float("nan")
            csv_start = float(t_local[0]) if T > 0 else float("nan")
            csv_end = float(t_local[-1]) if T > 0 else float("nan")

            sf.write(f"File {idx}/{len(files)}: {fname}\n")
            sf.write(f"  Full path:       {fpath}\n")
            sf.write(f"  File size (B):   {os.path.getsize(fpath)}\n")
            sf.write(f"  Samples:         {T}\n")
            sf.write(f"  Duration (s):    {duration_s:.6f}\n")
            sf.write(f"  Sample rate (Hz): {fs_in}\n")
            sf.write(f"  Channels in file: {len(ch_list)}\n")
            sf.write(f"  Channels exported: {len(chosen)}\n")
            sf.write(f"  Diff cols added:  {len(diff_map)}\n")
            sf.write(f"  CSV start (s):    {csv_start:.6f}\n")
            sf.write(f"  CSV end (s):      {csv_end:.6f}\n")
            sf.write(f"  Channel names exported: {', '.join(chosen) if chosen else 'None'}\n\n")

            _write_intan_metadata(sf, d, prefix="  ")

            sf.write("-" * 60 + "\n\n")

            # ---- Stream rows to CSV ----
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

            # advance global time offset
            if T > 0 and continuous_time:
                t_offset = float(t_local[-1] + 1.0 / fs)

            total_files_ok += 1

        # Footer summary
        sf.write("\n")
        sf.write("=" * 60 + "\n")
        sf.write("Totals\n")
        sf.write("=" * 60 + "\n")
        sf.write(f"Files processed OK: {total_files_ok}/{len(files)}\n")
        sf.write(f"Total samples:      {total_samples}\n")
        if continuous_time:
            sf.write(f"Total duration (s): {t_offset:.6f}\n")
        else:
            sf.write("Total duration (s): N/A (continuous_time=False)\n")
        sf.write(f"Output columns:     {len(header)} (time + {len(header)-1})\n")

        print(f"✅ Combined raw amplifier CSV saved: {out_csv}")
        print(f"📝 Summary saved: {summary_path}")
        print(f"   Columns: {len(header)} (time + {len(header)-1} channels)")
        print(f"   Sample rate: {fs:.6f} Hz")

    finally:
        f_csv.close()
        sf.close()


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
