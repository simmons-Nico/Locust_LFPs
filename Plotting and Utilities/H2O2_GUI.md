# H2O2: one CSV per preparation

In SPIE's **4 Comparison and plotting** tab:

1. Select the recording CSV once, with all three epochs in the same file.
2. Choose **H₂O₂ positive control** in that file's condition selector, alongside the cathodic current options.
3. Run **Firing rate (% of baseline) > Run normalized analysis**.

The script reads the `epoch_label` column automatically:

| Epoch label | Use |
| --- | --- |
| baseline | Pre-H2O2 reference for this preparation |
| during | During-perfusion firing rate |
| post | After-perfusion firing rate |

Labels are case-insensitive and surrounding whitespace is ignored. For example,
`Baseline`, `DURING`, and `Post` are accepted. Other epoch names are reported as
unrecognized instead of silently pooled. Use the existing CSV columns
`epoch_label`, `channel`, `window_start_s`, `window_end_s`, and `spike_count`.

Each CSV is one preparation. Both measurements use that CSV's baseline from the
same channel: `100 * measurement firing rate / baseline firing rate`. Rates are
duration weighted. Baseline uses the final 20 minutes by default, or the baseline
reference range entered in the main panel. Channels are kept separate.

Pairing and stable preparation identifiers come from the source file. No manual
locust ID, external baseline file, or separate during/after assignment is needed.
The CSV export maps each plotted identifier to its source recording. Missing
measurements are left absent and are not connected; missing/nonpositive baselines
are rejected. No measurements are invented.

**H₂O₂ concentration...** is optional and accepts a concentration with units.
It is not necessary to open this dialog before running the analysis.

Combined cathodic/control figures and data are in **H2O2_positive_control/** under
the selected output folder. **plot_catalog.csv** identifies the current figures
by channel and cathodic post epoch. Figures have the shared percentage axis,
100% reference, individual observations and same-preparation H2O2 pair lines.
Exports are SVG and 600-dpi PNG. Normal cathodic outputs retain their existing
CSVs/ and Plots/ directories. The combined layout is 9 by 4.8 inches.

**h2o2_normalized_rates.csv** records preparation IDs, source file, epoch,
measurement and baseline rates, the baseline window, and normalized percentages.
**settings.json** records the concentration, condition assignments and baseline
range. GUI concentration settings remain in memory during the session.
