# Preparation spike analysis

In SPIE tab **4 Comparison and plotting**, choose preparation CSVs and assign
each file a current in nA or H2O2 positive control. The editable selectors offer
-100, -200, -300 and H2O2 positive control; other finite numeric values are accepted.
For H2O2, one CSV contains baseline, during and post epochs (case-insensitive);
baseline normalization and pairing are automatic within that preparation/channel.
See [H2O2 GUI instructions](H2O2_GUI.md). The selected preparation list takes
precedence over the single input for the normalization and overlay tools only.
Clear that list to use the single input again. Each channel gets separate plots.

**Epoch spike-count overlay** plots raw counts by current intensity, with
individual traces, equal-preparation means, and a vertical mean ? SEM error bar
at each matching aligned window. No shaded error band is drawn.
**Firing rate (% of baseline)** produces each preparation's normalized time
course and epoch means, plus individual Post-epoch responses versus current intensity.
The current axis descends numerically from left to right (-100, -200, -300 nA).
These response figures show individual points and `n = ...` only, without group
mean markers, connecting lines, or SEM. Raw overlays retain mean +/- SEM.

Each CSV is one specimen, regardless of its recording names or chunk count.
Required columns are `epoch_label`, `channel`, `window_start_s`, `window_end_s`,
and `spike_count`. Counts may be fractional. Times must be continuous within an
epoch/channel, with positive, nonoverlapping windows. Chunk-local times that
reset/overlap are rejected rather than silently concatenated. Epoch labels match
literally; the reference epoch must be labelled `Baseline`.

Firing rate is count divided by elapsed seconds. The baseline is the
duration-weighted mean over the final 1200 seconds of Baseline, separately per
file/channel. The GUI can override this with start/end minutes measured from
the first observed Baseline window of each channel. The explicit range applies
to all normalized outputs in the run and is recorded in the baseline table.
A bin crossing the cutoff contributes according to its overlap,
assuming a constant rate within that bin. Gaps are not treated as zero activity.
If less than 1200 seconds of valid exposure is available there, available data
are used with a notice. Missing/zero baseline excludes normalized values for
that file/channel and leaves raw counts available. Notices appear in the GUI,
processing log, baseline table, and analysis notes.

Every Post epoch uses the same reference. Normalization is `100 * rate / baseline`.
Epoch means use total counts divided by total observed duration, not medians.
The Baseline epoch summary uses the defined final-20-minute reference and is
100%. The full recorded baseline mean is also exported as
`whole_epoch_mean_rate_hz`. The reference line is always 100%; baseline
time-course samples retain variation.

Epochs align at their first observed window. Raw means combine only identical
aligned start/end boundaries. Missing epochs/windows are absent, not zero or
interpolated. Group summaries average one normalized epoch value per independent
preparation, separately by channel, epoch, and current. Channels and time bins
are never extra replicates. SEM is `sample SD (ddof=1) / sqrt(n)` across
independent preparations. Error bars show **Mean ? SEM**, not confidence
intervals or standard deviation. SEM is missing for n < 2, so a single
preparation has no error bar. Individual time courses and epoch summaries
never have preparation-level SEM. No paired tests between specimens are performed.

Sample sizes are calculated dynamically. The overlay shows total selected n
and a separate contributing-n strip for aligned windows, including changes
caused by unequal lengths or missing epochs. Current-response plots explicitly
show `n = 2` (or the actual n) at each intensity; invalid baseline
normalizations are excluded. Each preparation retains its color and marker
across plots. Overlay group means use black lines and horizontal markers, with hollow
individual markers drawn above the error bars.

Epoch-mean bars originate at the normalized 100% reference line: decreases
extend downward and increases extend upward. These figures show duration-weighted
mean +/- SEM across time windows within that preparation/channel, with the
baseline denominator held fixed. This error describes within-recording variation,
not SEM across independent preparations or uncertainty in the normalization ratio.
Baseline windows are restricted to the selected reference interval; partially
overlapping windows use their overlap duration. SEM uses weighted sample variance
and effective sample size `(sum weights)^2 / sum(weights^2)`.

Each non-Baseline epoch is compared with those baseline windows using a two-sided
Welch t-test. With unequal window exposures, weighted means, sample variances and
effective sample sizes give an approximate weighted Welch test; equal exposures
recover the ordinary test. Holm adjustment covers all comparisons in each
preparation/channel. Bracket stars use adjusted p-values: `*` < .05, `**` < .01,
`***` < .001, `****` < .0001; `ns` is not significant and `n/a` means the test
could not be calculated. No baseline-versus-itself test is performed. SEM requires
at least two windows; tests also require sufficient baseline windows and nonzero
variance in at least one epoch. These tests assume independent time windows and
do not account for temporal autocorrelation or establish effects across specimens.
Exact raw and adjusted p-values, test status, window counts and plotted SEM are
exported to `CSVs/epoch_window_statistics.csv`. Existing epoch means, normalization
and intensity-response calculations and figures remain unchanged.

Other firing-rate/count y-axes start at zero, and normalized axes include 100%.
See [EXPERIMENTAL_TIME_COURSES.md](EXPERIMENTAL_TIME_COURSES.md) for the additional
experimental-timeline tool, recording display names, timing configuration, and
baseline reference overrides.

Use **Set plot titles...** in either plotting panel to edit the title of each
expected plot. Labels identify the plot by preparation/channel, current group,
or Post epoch. Blank entries use the displayed default. Save retains edits
throughout the GUI session, including reopening the editor and changing the
selection. Titles are keyed to stable IDs, never list positions. Sample-size
annotations are independent of titles. Long titles wrap onto multiple lines.

Output layout:

```text
Selected output directory/
    CSVs/
        time_windows.csv
        epoch_summaries.csv
        baseline_references.csv
        overlay_summary.csv
        current_intensity_summary.csv
        plot_catalog.csv
    Plots/
        <stable_plot_id>.png
        <stable_plot_id>.svg
    analysis_notes.txt
    preparation_plot_manifest.json
```

Summary tables contain `mean`, `sem`, and `n_preparations`, plus their grouping
identifiers. The plot catalog links stable IDs, default/custom titles, plot
identifiers, and figure paths. Titles never become filenames.

The manifest records generated files and content hashes. A successful new run
replaces the requested plot family and removes stale tool-owned files. Running
the other tool with the same inputs/intensities retains compatible plots and
catalog entries from the first tool. Changing input content, files or currents
invalidates both previous plot families. Unrelated files are untouched. If you
edit a generated file, use a different output folder; SPIE protects the edited
file instead of overwriting/deleting it. Old untracked exports from pre-manifest
SPIE versions remain untouched; use a fresh output folder for a clean upgrade.
Current outputs are the files listed in the manifest and catalog.

Standalone examples (from the repository root):

```powershell
python "Spike Processing/overlay_spike_counts_separate_epochs.py" "C:/data/prep1.csv" "C:/data/prep2.csv" --currents -200 -200 --out-dir "C:/data/overlay"
python "Plotting and Utilities/plot_baseline_post_firing_rates.py" "C:/data/prep1.csv" "C:/data/prep2.csv" --currents -100 -300 --out-dir "C:/data/normalized"
```

Without input arguments, either script opens a multiple-file dialog and asks
for each file's current intensity. `--mode both` generates both plot families.
For standalone custom titles, first create a template using the same inputs,
currents and mode, with `--write-title-template "C:/data/titles.json"`. This writes
a JSON object mapping plot IDs to default titles and exits without plotting.
Edit the values, then rerun with `--titles-file "C:/data/titles.json"`.
Alternatively pass the JSON object using `--titles-json`. Unknown IDs are
retained/ignored, enabling one title file for multiple selections. Blank values
use defaults. Paths and title values may contain spaces and Unicode.

The baseline script's default entry point now uses this preparation analysis;
its historical helper functions are retained for existing importers, but the
old zero-centered CLI options are no longer used by SPIE.

Validation: `python -m unittest discover -s tests -p test_preparation_spike_analysis.py -v`
