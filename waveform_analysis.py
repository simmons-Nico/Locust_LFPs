#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Waveform analysis across experimental stages (baseline-relative).

For each CSV in CSV_DIR (raw Intan-style recordings with a time column and one
column per channel, voltage in uV) that matches CSV_GLOB:

- spikes are detected on the 300-5000 Hz band-passed signal using the same
  routines as 'LFP and Spike Plots (excluding time ranges).py'
  (order-4 SOS band-pass, MAD-based robust z threshold, refractory distance),
- individual spike waveforms are sliced from that high-passed signal,
- the stage label is parsed from the filename (chunk_LABEL_NUMBER.csv), and
  waveforms are grouped by stage per channel.

For every selected channel the BASELINE_STAGE (set in the USER SETTINGS block,
default token 'Baseline') is the reference and every other stage is compared
ONLY against it:

1) AMPLITUDE : per-spike max amplitude (polarity peak magnitude). Each stage is
   expressed as a percentage of the baseline mean amplitude so that baseline
   equals 100%. Statistical significance (Mann-Whitney U) is computed against
   the baseline per-spike amplitudes only.
2) SHAPE     : the baseline mean waveform is the template; every spike waveform
   of every stage is correlated (Pearson) with that template. Each stage's
   per-spike correlation distribution is compared with the baseline's own
   distribution by Mann-Whitney U.

Outputs are written under CSV_DIR/waveform_analysis/<channel>/.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.stats import gaussian_kde, ttest_rel

# ================== USER SETTINGS ==================
# Folder containing the raw CSV files (set this!)
CSV_DIR = r"C:\Users\locadmin\Documents\CEITEC\Final Store\drip - 1mm - 0,25ul (30.07.2026)\raw"
CSV_GLOB = "*.csv"

# Channels to analyse (set these - column names as they appear in the CSVs)
CHANNELS = ["A-005_uV", "A-008_uV", "A-009_uV"]

OUT_DIR_NAME = "waveform_analysis"

# -------- filename -> stage parsing --------
# Filenames look like:  chunk_LABEL_NUMBER.csv   e.g.  rec_Baseline_001.csv
# The stage label is matched (space/underscore-insensitive) against STAGE_LABELS
# or, failing that, taken as the token(s) immediately before the trailing number.
STAGE_LABELS = ["Baseline", "Thru", "Post"]
# The stage used as the 100% reference (must appear in STAGE_LABELS / filenames).
BASELINE_STAGE = "Baseline"
# Optional ordering of the non-baseline stages for display; leave empty for sorted order.
STAGE_ORDER = ["Baseline", "Thru", "Post"]

# -------- spike detection (project defaults) --------
HP_SPIKE_BAND = (300.0, 5000.0)
SPIKE_Z_THR = 6.0
POLARITY = "neg"        # "neg", "pos" or "both"
REFRACTORY_MS = 1.0
AMP_MIN_UV = 40.0
AMP_MAX_UV = 700.0

# -------- waveform extraction --------
PRE_MS = 0.6             # ms before spike peak
POST_MS = 1.0            # ms after spike peak

TIME_COLS = ["time_s", "time", "t", "seconds"]

SIG_ALPHA = 0.05         # significance threshold for the vs-baseline tests
DPI = 150
# ================================================

# ----------------- helpers (as in existing scripts) -----------------
def find_time_column(df: pd.DataFrame) -> str:
    for c in TIME_COLS:
        if c in df.columns:
            return c
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.number):
            s = df[c].to_numpy()
            diffs = np.diff(s.astype(float))
            if np.all(np.isfinite(diffs)) and np.all(diffs > 0):
                return c
    raise ValueError("Could not find a time column. Add/rename your time column or update TIME_COLS.")


def infer_fs(t: np.ndarray) -> float:
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Time column invalid (nonpositive/non-finite diffs).")
    return 1.0 / dt


def _safe_band_for_fs(low, high, fs, margin=0.95):
    ny = fs * 0.5
    hi = min(high, ny * margin) if high is not None else None
    lo = max(low, 0.001) if low is not None else None
    if hi is not None and lo is not None and hi <= lo:
        hi = min(lo * 1.2, ny * margin)
    return lo, hi


def design_sos_bandpass(low, high, fs, order=4):
    low, high = _safe_band_for_fs(low, high, fs)
    if low is None and high is None:
        raise ValueError("Both low and high are None.")
    if low is None:
        wn = high / (fs * 0.5)
        sos = butter(order, wn, btype="lowpass", output="sos")
    elif high is None:
        wn = low / (fs * 0.5)
        sos = butter(order, wn, btype="highpass", output="sos")
    else:
        wn = [low / (fs * 0.5), high / (fs * 0.5)]
        sos = butter(order, wn, btype="bandpass", output="sos")
    return sos


def band_filter(x, fs, band, order=4):
    sos = design_sos_bandpass(band[0], band[1], fs, order=order)
    return sosfiltfilt(sos, x)


def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad)


def detect_spikes(xhp, fs, zthr=SPIKE_Z_THR, refr_ms=REFRACTORY_MS):
    z = robust_z(xhp)
    distance = max(1, int(round((refr_ms / 1000.0) * fs)))
    idx, _ = find_peaks(np.abs(z), height=zthr, distance=distance)
    return np.asarray(idx, dtype=int)


def extract_waveforms(xhp, idx, fs, pre_ms=PRE_MS, post_ms=POST_MS):
    pre = int(round(pre_ms * 1e-3 * fs))
    post = int(round(post_ms * 1e-3 * fs))
    valid = idx[(idx - pre >= 0) & (idx + post < len(xhp))]
    if valid.size == 0:
        return valid, np.zeros((0, pre + post))
    W = np.stack([xhp[i - pre:i + post] for i in valid], axis=0)
    return valid, W


def fill_nans(x):
    if np.isfinite(x).all():
        return x
    good = np.isfinite(x)
    if good.sum() < 2:
        return x
    idx = np.arange(len(x), dtype=float)
    first, last = int(np.where(good)[0][0]), int(np.where(good)[0][-1])
    y = x.copy()
    y[:first] = y[first]
    y[last + 1:] = y[last]
    y[~good] = np.interp(idx[~good], idx[good], y[good])
    return y


# ----------------- filename / stage parsing -----------------
def parse_stage_label(path):
    """Return (stage_label, trailing_number_or_None) for a CSV path."""
    base = os.path.splitext(os.path.basename(path))[0]
    parts = [p for p in base.split("_") if p != ""]
    if not parts:
        return base, None

    number = None
    if parts[-1].isdigit():
        number = int(parts[-1])
        parts_n = parts[:-1]
    else:
        parts_n = parts[:]

    # space/underscore-normalised forms for robust matching
    fn_norm = re.sub(r"[\s_]+", "_", base).strip("_").lower()
    if STAGE_LABELS:
        best = None  # (original_label, norm_label)
        for lab in STAGE_LABELS:
            ln = re.sub(r"\s+", "_", lab.strip()).lower()
            if ln and ln in fn_norm and (best is None or len(ln) > len(best[1])):
                best = (lab, ln)
        if best is not None:
            return best[0], number

    if parts_n:
        return parts_n[-1], number
    return base, number


def stage_display_order(labels):
    """Baseline always first; remaining stages ordered by STAGE_ORDER, else sorted."""
    known = [s for s in labels if s in labels]
    if STAGE_ORDER:
        ordered = [s for s in STAGE_ORDER if s in known]
        ordered += [s for s in sorted(known) if s not in ordered]
    else:
        ordered = sorted(known)
    if BASELINE_STAGE in ordered:
        ordered.remove(BASELINE_STAGE)
        ordered.insert(0, BASELINE_STAGE)
    return ordered


# ----------------- waveform / shape helpers -----------------
def waveform_time_axis(fs):
    pre = int(round(PRE_MS * 1e-3 * fs))
    n = pre + int(round(POST_MS * 1e-3 * fs))
    return (np.arange(n) - pre) / fs * 1e3


def resample_to_axis(w, fs, t_ref):
    pre_i = int(round(PRE_MS * 1e-3 * fs))
    n_i = pre_i + int(round(POST_MS * 1e-3 * fs))
    if n_i == t_ref.size:
        return w
    t_i = (np.arange(n_i) - pre_i) / fs * 1e3
    return np.interp(t_ref, t_i, w)


def pearson_vs_template(W, tmpl):
    """Pearson r between every row of W and the 1-D template."""
    Wc = W - W.mean(axis=1, keepdims=True)
    tc = tmpl - tmpl.mean()
    den = np.sqrt((Wc ** 2).sum(axis=1) * (tc ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (Wc @ tc) / den
    r[~np.isfinite(r)] = np.nan
    return r


# ----------------- plotting helpers -----------------
def colors_for(n):
    base = plt.rcParams["axes.prop_cycle"].by_key().get("color",
                                                        ["#1f77b4"])
    return [base[i % len(base)] for i in range(n)]


def block_perm_p(stage_dict, base_dict, n_perm=1000, seed=0):
    """Two-sided p from a within-recording block permutation test.

    stage_dict/base_dict are dicts keyed by recording id (filename trailing
    number = position within the stage), each value a per-spike array. To
    avoid treating thousands of pooled spikes as independent (autocorrelation
    within a single session), spike labels are shuffled only within each
    matched recording, then the pooled mean(stage %% ) - mean(baseline %%)
    statistic is recomputed across the recording. p = two-sided tail fraction
    over n_perm shuffles. Recordings present in only one dict are dropped.
    """
    if not stage_dict or not base_dict:
        return np.nan
    pairs = []
    for k in set(stage_dict) & set(base_dict):
        a = np.asarray(stage_dict[k], dtype=float)
        b = np.asarray(base_dict[k], dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size and b.size:
            pairs.append((a, b))
    if sum(a.size + b.size for a, b in pairs) < 4:
        return np.nan

    def _stat(s_stage, s_base):
        ms = float(np.concatenate(s_stage).mean())
        mb = float(np.concatenate(s_base).mean())
        return ms - mb

    obs = _stat([a for a, _ in pairs], [b for _, b in pairs])

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        s_s, s_b = [], []
        for a, b in pairs:
            n_s = a.size
            sh = rng.permutation(np.concatenate([a, b]))
            s_s.append(sh[:n_s])
            s_b.append(sh[n_s:])
        null[i] = _stat(s_s, s_b)
    p = 2.0 * min(float((null <= obs).mean()), float((null >= obs).mean()))
    return float(min(p, 1.0))


def paired_split_p(pairs):
    """Baseline stationarity p: paired t-test on per-recording
    1st-half vs 2nd-half mean amplitudes (~100, ns if stationary)."""
    first = np.array([l for l, r in pairs if np.isfinite(l) and np.isfinite(r)],
                     dtype=float)
    second = np.array([r for l, r in pairs if np.isfinite(l) and np.isfinite(r)],
                      dtype=float)
    if first.size >= 2 and second.size >= 2 and not np.allclose(first, second):
        try:
            return float(ttest_rel(second, first).pvalue)
        except Exception:
            return np.nan
    return np.nan



# ----------------- raincloud plot -----------------
def _sig_bracket(ax, x1, x2, y, text, dy):
    """Compact significance bracket with bold label (firing-rate style)."""
    if text == "n/a":
        return
    ax.plot([x1, x1, x2, x2], [y, y + dy, y + dy, y], color="black", lw=1.0)
    ax.text((x1 + x2) / 2.0, y + dy, text, ha="center", va="bottom",
            fontsize=10, weight="bold")


def _p_stars(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < SIG_ALPHA:
        return "*"
    return "ns"


def raincloud_amplitude_figure(box_by, labels, colors, channel, ylabel,
                               title, save_path, baseline_label,
                               stage_base_p=None, base_pair_p=None,
                               hline=100.0):
    """Amplitude raincloud (box + half-violin, no dots) on the %-of-baseline
    scale, per spike.

    Every stage -- baseline included -- is per-spike 100 x amp / pooled baseline
    mean, so all boxes show raw per-spike spread on one common construct.
    Significance: stage_base_p (stage -> p) is a within-recording block
    permutation test vs baseline; base_pair_p is a paired 1st/2nd-half baseline
    stationarity control rendered as a paired star over the baseline box.
    """
    base_idx = labels.index(baseline_label)
    n = len(labels)
    pos = np.arange(n, dtype=float)
    fig, ax = plt.subplots(figsize=(0.9 * n + 2.2, 5.0))

    box_data = [box_by[st] for st in labels]

    # ---- box (centered) ----
    bp = ax.boxplot(box_data, positions=pos, widths=0.45, patch_artist=True,
                    showfliers=False, zorder=2,
                    medianprops=dict(color="black", lw=1.2),
                    whiskerprops=dict(color="black", lw=1.0),
                    capprops=dict(color="black", lw=1.0))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.45)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    # ---- half-violin (right of box) ----
    violin_offset = 0.22
    for xi, data, col in zip(pos, box_data, colors):
        if data.size < 2:
            continue
        try:
            kde = gaussian_kde(data, bw_method="scott")
        except Exception:
            continue
        lo = float(np.percentile(data, 1))
        hi = float(np.percentile(data, 99))
        if hi <= lo:
            lo, hi = float(data.min()), float(data.max())
        pad = (hi - lo) * 0.05 or 1.0
        grid = np.linspace(lo - pad, hi + pad, 200)
        density = kde(grid)
        density_max = density.max()
        if density_max <= 0:
            continue
        scale = 0.18 / density_max
        ax.fill_betweenx(grid, xi + violin_offset,
                          xi + violin_offset + density * scale,
                          color=col, alpha=0.35, linewidth=0, zorder=1)

    # ---- reference line ----
    if hline is not None:
        ax.axhline(hline, color="k", ls="--", lw=1.0, alpha=0.6)

    # ---- significance: block-permutation p per stage vs baseline ----
    star_pairs = []
    for k in range(n):
        if k == base_idx:
            continue
        star_pairs.append((k, stage_base_p[labels[k]] if stage_base_p
                           and labels[k] in stage_base_p else np.nan))

    pool = np.concatenate([d for d in box_data if d.size]) if any(d.size for d in box_data) else np.array([0.0])
    lo_all, hi_all = np.percentile(pool, [0.5, 99.5])
    span = (hi_all - lo_all) or 1.0

    finite_pairs = [(k, p) for k, p in star_pairs if np.isfinite(p)]
    top = hi_all + 0.10 * span + (len(finite_pairs) + 1) * 0.07 * span

    row = 0
    for k, p in reversed(finite_pairs):
        y = top - row * 0.07 * span
        _sig_bracket(ax, 0, k, y, _p_stars(p), 0.02 * span)
        row += 1

    # baseline split-half stationarity control (paired star over baseline box)
    if base_pair_p is not None:
        base_star = _p_stars(base_pair_p)
        if base_star != "n/a":
            _sig_bracket(ax, -0.18, 0.18, hi_all + 0.12 * span, base_star,
                         0.04 * span)
            top = max(top, hi_all + 0.12 * span + 0.06 * span)

    ax.set_xticks(pos)
    ax.set_xticklabels([f"{st}\n(n={int(d.size)})" for st, d in zip(labels, box_data)],
                       rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} -- {channel}")
    ax.set_ylim(lo_all - 0.1 * span, top + 0.10 * span)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)


# ======================================================================
def main():
    os.makedirs(CSV_DIR, exist_ok=True)
    out_root = os.path.join(CSV_DIR, OUT_DIR_NAME)
    os.makedirs(out_root, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, CSV_GLOB)))
    if not csv_files:
        raise SystemExit(f"No CSVs found in {CSV_DIR!r} with pattern {CSV_GLOB!r}")

    # ---- 1. detect spikes & extract waveforms per file/channel ----
    # results[channel][stage] = {"fs": [...], "amp": [arrays], "W": [gated
    #                            waveform matrices], "files": [...]}
    results = {}

    for path in csv_files:
        stage, rec_num = parse_stage_label(path)
        base = os.path.splitext(os.path.basename(path))[0]
        try:
            df = pd.read_csv(path)
            time_col = find_time_column(df)
            t = df[time_col].to_numpy(dtype=float)
            fs = infer_fs(t)
        except Exception as e:
            print(f"[skip] {base}: {e}")
            continue

        for ch in CHANNELS:
            if ch not in df.columns:
                print(f"[skip] {base}: channel '{ch}' not present")
                continue
            x = pd.to_numeric(df[ch], errors="coerce").to_numpy(float)
            x = fill_nans(x)
            if not np.isfinite(x).any():
                print(f"[skip] {base}/{ch}: no finite samples")
                continue

            xhp = band_filter(x, fs, HP_SPIKE_BAND, order=4)
            spike_idx = detect_spikes(xhp, fs, zthr=SPIKE_Z_THR,
                                      refr_ms=REFRACTORY_MS)
            valid_idx, W = extract_waveforms(xhp, spike_idx, fs, PRE_MS, POST_MS)
            if W.shape[0] == 0:
                print(f"[{base}/{ch}] stage '{stage}': 0 spikes")
                continue

            # per-spike max amplitude = polarity peak magnitude (gating metric)
            if POLARITY == "neg":
                amp_uv = np.abs(W.min(axis=1))
            elif POLARITY == "pos":
                amp_uv = np.abs(W.max(axis=1))
            else:  # "both"
                amp_uv = np.max(np.abs(W), axis=1)

            keep = amp_uv >= AMP_MIN_UV
            if AMP_MAX_UV is not None:
                keep &= amp_uv <= AMP_MAX_UV
            W = W[keep]
            amp_uv = amp_uv[keep]
            valid_idx = valid_idx[keep]
            if W.shape[0] == 0:
                print(f"[{base}/{ch}] stage '{stage}': all spikes failed amplitude gate")
                continue

            results.setdefault(ch, {}).setdefault(stage, {"fs": [], "amp": [],
                                                          "W": [], "files": []})
            dst = results[ch][stage]
            dst["fs"].append(fs)
            dst["amp"].append(amp_uv)
            dst["W"].append(W)
            dst["files"].append(rec_num)
            print(f"[{base}/{ch}] stage '{stage}': {W.shape[0]} spikes kept")

    if not results:
        raise SystemExit("No spikes extracted for any requested channel.")

    for ch, stages in results.items():
        if BASELINE_STAGE not in stages:
            print(f"[skip] {ch}: baseline stage '{BASELINE_STAGE}' not found "
                  f"(have: {sorted(stages)})")
            continue

        labels = stage_display_order(list(stages.keys()))
        if labels[0] != BASELINE_STAGE:
            labels.remove(BASELINE_STAGE)
            labels.insert(0, BASELINE_STAGE)
        colors = colors_for(len(labels))

        ch_dir = "".join(c if (c.isalnum() or c in "._-") else "_" for c in ch)
        out_dir = os.path.join(out_root, ch_dir)
        os.makedirs(out_dir, exist_ok=True)

        # ---- 2. common time axis, per-stage amplitude & waveform stacks ----
        fs_list = [f for v in stages.values() for f in v["fs"]]
        ref_fs_ch = fs_list[0]
        t_ms = waveform_time_axis(ref_fs_ch)

        amps, Ws_by, meanW, semW, n_spk = {}, {}, {}, {}, {}
        for st in labels:
            recs = stages[st]
            amps[st] = np.concatenate(recs["amp"]) if recs["amp"] else np.array([])
            Ws = [resample_to_axis(w, f, t_ms)
                  for w, f in zip(recs["W"], recs["fs"])]
            Ws_by[st] = np.vstack(Ws) if Ws else np.zeros((0, t_ms.size))
            n_spk[st] = Ws_by[st].shape[0]
            if n_spk[st]:
                meanW[st] = Ws_by[st].mean(axis=0)
                ddof = 1 if n_spk[st] > 1 else 0
                semW[st] = Ws_by[st].std(axis=0, ddof=ddof) / np.sqrt(n_spk[st])
            else:
                meanW[st] = np.full(t_ms.size, np.nan)
                semW[st] = np.full(t_ms.size, np.nan)

        if amps[BASELINE_STAGE].size < 2:
            print(f"[skip] {ch}: baseline stage has <2 spikes")
            continue

        base_mean_amp = float(amps[BASELINE_STAGE].mean())

        # baseline mean waveform = shape template
        template = meanW[BASELINE_STAGE]

        # per-spike correlation with baseline template
        r_by = {}
        for st in labels:
            if n_spk[st] and np.isfinite(template).all():
                r_by[st] = pearson_vs_template(Ws_by[st], template)
            else:
                r_by[st] = np.array([])

        # recording-level amplitude as % of pooled baseline mean (one value per
        # recording per stage; all stages same construct, incl. baseline)
        # per-spike amplitude as % of pooled baseline mean (all stages incl.
        # baseline), plus per-recording dicts keyed by rec_num (position)
        # for the block permutation tests
        pct_by = {st: 100.0 * amps[st] / base_mean_amp for st in labels}
        rec_pct_by = {
            st: {num: 100.0 * a / base_mean_amp
                 for num, a in zip(stages[st]["files"], stages[st]["amp"])
                 if num is not None}
            for st in labels}
        stage_base_p = {
            st: block_perm_p(rec_pct_by[st], rec_pct_by[BASELINE_STAGE])
            for st in labels if st != BASELINE_STAGE}

        # per-recording shape (Pearson r to template), keyed by rec_num
        rec_r_by = {}
        for st in labels:
            d = {}
            for num, w, f in zip(stages[st]["files"], stages[st]["W"],
                                 stages[st]["fs"]):
                if num is None or w.shape[0] == 0:
                    continue
                if not np.isfinite(template).all():
                    continue
                w_res = resample_to_axis(w, f, t_ms)
                d[num] = pearson_vs_template(w_res, template)
            rec_r_by[st] = d
        shape_perm_p = {
            st: block_perm_p(rec_r_by[st], rec_r_by[BASELINE_STAGE])
            for st in labels if st != BASELINE_STAGE}
        base_half = []
        for a in stages[BASELINE_STAGE]["amp"]:
            a = np.asarray(a, dtype=float)
            if a.size < 2:
                continue
            h = a.size // 2
            l, r = a[:h], a[h:]
            if l.size and r.size and np.isfinite(l).all() and np.isfinite(r).all():
                base_half.append((float(l.mean()), float(r.mean())))
        base_pair_p = paired_split_p(base_half)

        # ---- 3. mean +/- SEM figures (visual QC) ----
        # shared y-range across all subplots = the largest of the stages
        y_lo = min((float(meanW[st].min() - semW[st].max())
                    for st in labels if np.isfinite(meanW[st]).all()),
                   default=-1.0)
        y_hi = max((float(meanW[st].max() + semW[st].max())
                    for st in labels if np.isfinite(meanW[st]).all()),
                   default=1.0)
        fig, axes = plt.subplots(1, len(labels), figsize=(3.4 * len(labels), 3.2),
                                 squeeze=False)
        for ax, st, col in zip(axes[0], labels, colors):
            if n_spk[st]:
                ax.plot(t_ms, meanW[st], lw=2, color=col)
                ax.fill_between(t_ms, meanW[st] - semW[st],
                                meanW[st] + semW[st], alpha=0.3, color=col)
                ax.set_title(f"{st}\n(n={n_spk[st]})")
            else:
                ax.set_title(f"{st}\n(no spikes)")
            ax.axhline(0, color="k", lw=0.5, alpha=0.4)
            ax.set_xlabel("Time (ms)")
            ax.grid(True, alpha=0.3)
        axes[0][0].set_ylabel("uV")
        for ax in axes[0]:
            ax.set_ylim(y_lo, y_hi)
        fig.suptitle(f"Spike waveforms (mean +/- SEM) -- {ch}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{ch_dir}_mean_waveforms.png"), dpi=DPI)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for st, col in zip(labels, colors):
            if n_spk[st]:
                ax.plot(t_ms, meanW[st], lw=2, color=col,
                        label=f"{st} (n={n_spk[st]})")
                ax.fill_between(t_ms, meanW[st] - semW[st],
                                meanW[st] + semW[st], alpha=0.25, color=col)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("uV")
        ax.set_title(f"Stage overlay -- {ch}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{ch_dir}_stage_overlay.png"), dpi=DPI)
        plt.close(fig)

        # ---- 4. amplitude raincloud figure (per spike, block-perm sig) ----
        box_by = dict(pct_by)
        raincloud_amplitude_figure(
            box_by, labels, colors, ch,
            ylabel="Max amplitude (% of baseline mean)",
            title="Max amplitude vs baseline (baseline = 100%)",
            save_path=os.path.join(out_dir, f"{ch_dir}_amplitude_pct_baseline.png"),
            baseline_label=BASELINE_STAGE, stage_base_p=stage_base_p,
            base_pair_p=base_pair_p, hline=100.0)

        # ---- 5. tables (vs baseline) ----
        amp_rows, shape_rows = [], []
        for st in labels:
            a = amps[st]
            r_ = r_by[st]
            p_amp = (stage_base_p.get(st) if st != BASELINE_STAGE else np.nan)
            n_rec = int(len(rec_pct_by[st]))
            p_r = (shape_perm_p.get(st) if st != BASELINE_STAGE else np.nan)
            amp_rows.append({
                "channel": ch, "stage": st, "n_spikes": int(a.size),
                "n_recordings": n_rec,
                "baseline_mean_amp_uv": base_mean_amp,
                "mean_amp_uv": float(a.mean()) if a.size else np.nan,
                "std_amp_uv": float(a.std(ddof=1)) if a.size > 1 else np.nan,
                "median_amp_uv": float(np.median(a)) if a.size else np.nan,
                "mean_pct_of_baseline": (float(a.mean()) / base_mean_amp * 100.0
                                         if a.size else np.nan),
                "median_pct_of_baseline": (100.0 * float(np.median(a)) / base_mean_amp
                                           if a.size else np.nan),
                "perm_p_vs_baseline": p_amp,
            })
            if r_.size:
                ddof_r = 1 if r_.size > 1 else 0
                shape_rows.append({
                    "channel": ch, "stage": st, "n_spikes": int(r_.size),
                    "mean_r_to_baseline": float(np.nanmean(r_)),
                    "sem_r_to_baseline": float(np.nanstd(r_, ddof=ddof_r)
                                               / np.sqrt(r_.size)),
                    "median_r_to_baseline": float(np.nanmedian(r_)),
                    "q1_r_to_baseline": float(np.nanpercentile(r_, 25)),
                    "q3_r_to_baseline": float(np.nanpercentile(r_, 75)),
                    "perm_p_vs_baseline": p_r,
                })
            else:
                shape_rows.append({
                    "channel": ch, "stage": st, "n_spikes": 0,
                    "mean_r_to_baseline": np.nan, "sem_r_to_baseline": np.nan,
                    "median_r_to_baseline": np.nan, "q1_r_to_baseline": np.nan,
                    "q3_r_to_baseline": np.nan,
                    "perm_p_vs_baseline": np.nan,
                })

        pd.DataFrame(amp_rows).to_csv(
            os.path.join(out_dir, f"{ch_dir}_amplitude_vs_baseline.csv"), index=False)
        pd.DataFrame(shape_rows).to_csv(
            os.path.join(out_dir, f"{ch_dir}_shape_vs_baseline.csv"), index=False)

        # ---- 6. mean waveform export ----
        mw = {"t_ms": t_ms}
        for st in labels:
            mw[f"{st}_mean_uV"] = meanW[st]
            mw[f"{st}_sem_uV"] = semW[st]
        pd.DataFrame(mw).to_csv(os.path.join(out_dir,
                                             f"{ch_dir}_mean_waveforms.csv"),
                                index=False)

    print(f"\nDone. Plots and CSVs saved under: {out_root}")


if __name__ == "__main__":
    main()
