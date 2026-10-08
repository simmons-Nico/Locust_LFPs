# Experimental firing-rate time courses

SPIE tab 4 now has an additional **Experimental firing-rate time courses** tool.
It produces `time_course_absolute_hz_<ID>` and
`time_course_percent_baseline_<ID>` PNG/SVG pairs per channel/current group.
Existing raw overlays, normalized epoch plots and summaries remain available.
All firing/count axes start at zero; normalized axes always include 100%.

The separate current-response figures now show individual preparation points
only, with `n = ...` at each intensity. Their numeric x-axis descends from left
to right (-100, -200, -300 nA and any additional entered values). No group mean,
median, SEM, or connecting line appears on those response figures. Raw overlays
and experimental time courses retain preparation-level means and SEM.

This follows the requested Katsuki-inspired presentation, not a verified
reproduction of Fig. 1B. A likely reference is Katsuki et al., *Biphasic effect of
hydrogen peroxide on field potentials in rat hippocampal slices* (1997),
https://pubmed.ncbi.nlm.nih.gov/9430416/ . The figure itself was not verified.

## Recording display names

Use **Set recording names...** beside the preparation selector. Each field is
keyed to a resolved source path. Blank uses the filename stem. Names persist
when reopening the editor and changing selections during the GUI session. They
appear in legends, individual-plot default titles, and treatment rows. Explicit
custom titles take precedence. Duplicate names get stable ID suffixes in figure
labels and `display_label`; `display_name` retains the entered name.

Names never change source files, preparation IDs, colors/markers, or biological
n. Each file remains one specimen regardless of internal recording chunks.
Exports retain source paths, original recording names, display names and IDs.
Standalone scripts accept `--names-file names.json` or `--names-json`, with a
JSON object mapping absolute source paths or exported preparation IDs to names.

## Reference versus displayed baseline

The reference defaults to the duration-weighted mean over the final 20 minutes
of labelled Baseline, separately for each preparation/channel. GUI start/end
fields or `--baseline-range-min 10 20` select another stable range in minutes
from the first observed Baseline window. The same denominator is used for all
Post epochs and all normalized outputs generated in that run.

The requested reference bounds in seconds, duration actually observed, and mean
Hz are exported. A partially overlapping bin contributes in proportion to its
overlap, assuming constant rate within the bin. Insufficient exposure uses the
available reference data with a notice. Missing/zero baseline excludes
normalization; valid absolute-Hz data remain available.

Timeline baseline display defaults to reference-overlapping windows. Choose
`full` to show the full baseline, or `range` with separate display bounds.
Original windows intersecting that display range appear at their actual
midpoints. No synthetic endpoint samples or clipped-count observations are
added. Display changes never change the baseline denominator. Baseline summary
points remain 100%, while individual baseline samples retain their variation.

## Experimental timing

The timing-source selector supports:

- `auto`: uses `experimental_start_s` and `experimental_end_s` columns when
  present. They must describe one experimental clock per specimen and preserve
  every window's duration. No configured gap is added to these times.
- Without those columns, auto reconstructs epoch-local recordings using each
  specimen's own epoch extents and configured treatment gaps. Extents are
  determined across that specimen's channels, so channel dropout does not
  produce a fictitious treatment onset.
- If native `window_start_s/end_s` already run continuously across epochs, auto
  requires an explicit interpretation. `measured` uses them directly;
  `reconstructed` explicitly treats them as local to each epoch.
- `assigned`: uses user-entered starts for every observed epoch, in minutes in
  one experimental clock per specimen. Original local offsets and window
  durations remain intact. Assignments for a file override its automatic source.

Missing intermediate epochs have unknown durations and require measured times
or explicit starts. Missing trailing epochs are allowed. Recording-name dates
are not assumed to be reliable timestamps and are not parsed automatically.

The default is a 20-minute treatment between Baseline and Post 1 and between
successive Post epochs. **Set treatment boundaries / epoch starts...** can
disable a treatment at a boundary, change its duration, or assign starts for
each file. Treatment onset is assumed to be the preceding observed epoch end.
With measured/assigned timing, treatment must fit within the actual gap. Extra
unobserved time remains blank; no extra gap is inserted. Explicit recorded
Treatment/Stimulation/During epochs are preserved and marked.

By default, the first configured treatment onset is t=0. Ten one-minute baseline
windows appear at -9.5 through -0.5 minutes. With no treatment, Baseline end is
the origin. Disabling treatment-relative timing uses the first observation as
zero. Unshifted and relative experimental coordinates are exported.

Turning off **Show treatment duration** compresses each unrecorded treatment to
a visible one-minute break, or its actual duration if shorter. Original
experimental times and firing rates remain unchanged. Displayed times are
exported separately, and both plot and treatment labels explicitly identify
compression. Recorded treatment data are never compressed away. Traces break
between epochs and across treatment/missing-data intervals.

## Group alignment and biological n

Group time points require the same channel, current, epoch, epoch-relative
window boundaries, experimental boundaries, and displayed boundaries. Baseline
equivalence is aligned to Baseline end; Post equivalence is aligned to that
epoch's first observation. Windows are not matched by row number or nearest
timestamp. No interpolation, resampling or padding occurs.

Two observations with the same global time but different minutes since their
Post epoch began are not averaged together. Different Post 1 durations can
therefore leave Post 2 at n=1 throughout, even where absolute time ranges
overlap. The contributing-n strip shows this explicitly.

Each preparation is normalized first. Group SEM is sample SD (`ddof=1`) divided
by sqrt(number of independent contributing preparations). Channels and time
bins are never additional specimens. There is no SEM or extra group curve for
a single preparation. Individual curves remain visible; group means use dark
circular-marker lines. Treatment bars occupy separate rows per preparation so
unequal treatment timings are explicit. No significance tests are added.

## Standalone use and configuration

Use `--mode timeline` for the two new plot families, or `--mode all` to generate
all old and new outputs. Existing title-file options apply to the new plots.

```powershell
python "Plotting and Utilities/plot_baseline_post_firing_rates.py" "C:/data/prep1.csv" "C:/data/prep2.csv" --currents -200 -200 --mode timeline --timeline-file "C:/data/timing.json" --names-file "C:/data/names.json" --out-dir "C:/data/results"
```

Example `timing.json` (omit `epoch_starts_min` for ordinary auto reconstruction):

```json
{
  "timing_mode": "auto",
  "gap_min": 20,
  "show_treatment_duration": true,
  "relative_to_treatment": true,
  "baseline_display": "reference",
  "boundaries": {
    "Baseline -> Post 1": {"treatment": true, "duration_min": 20},
    "Post 1 -> Post 2": {"treatment": true, "duration_min": 15}
  },
  "epoch_starts_min": {
    "C:/data/prep1.csv": {"Baseline": 0, "Post 1": 40, "Post 2": 65}
  }
}
```

Assignments must fit the actual epoch durations and configured treatments.
Also supported: `--timeline-json`, `baseline_display: "full"`, or
`baseline_display: "range"` with `display_baseline_range_min: [0, 10]`.
The GUI exposes these settings without requiring JSON editing.

## Additional exports

All tables go into `CSVs/` and figures into `Plots/`:

- `experimental_time_windows.csv`: original observations, names/IDs, baseline
  details, window index, unshifted/relative experimental and displayed
  start/end/midpoint minutes, epoch-relative boundaries, timing source,
  compression/displayed flags, treatment duration before the epoch, and
  matching group statistics. Undisplayed observations are retained and flagged.
- `experimental_group_time_points.csv`: normalized and absolute group means,
  SEMs and separate contributing n values. Normalization exclusions do not
  reduce absolute-Hz n.
- `experimental_treatments.csv`: per-preparation treatment bounds, duration,
  current, recorded flag and displayed/experimental coordinates.
- `experimental_settings.csv`: the explicit run configuration.

The new plots participate in the title editor, catalog and manifest. Their
figures and timeline tables survive a subsequent compatible overlay/normalized
run. Changed inputs, currents, display names, or reference ranges invalidate
incompatible previous outputs. Unrelated and user-edited files remain protected.

Validation commands:

```powershell
python -m unittest discover -s tests -p test_preparation_spike_analysis.py -v
python -m unittest discover -s tests -p test_spike_experimental_timeline.py -v
```

## Recording symbols and current-response styling

Use **Set recording symbols...** to choose a symbol per source file. Choices persist when reopening the editor or changing selections during the session. Automatic restores the stable default. Symbols apply consistently to preparation traces and current-response observations without changing their identity or statistics.

Standalone scripts accept `--symbols-file symbols.json` or `--symbols-json`, mapping absolute source paths or preparation IDs to marker codes: `o` circle, `s` square, `^`/`v` triangles, `D` diamond, `p` pentagon, `h` hexagon, `P` filled plus, `X` filled cross, `*` star, `<`/`>` sideways triangles; an empty string selects Automatic. Observation tables export `recording_symbol`.

Current-response figures use filled markers, sequential blue shades, publication-sized sans-serif typography, and an editable right-hand legend. Signed current distances remain proportional, in descending order. They retain individual points and sample sizes without mean/SEM artists. See [CURRENT_RESPONSE_PUBLICATION.md](CURRENT_RESPONSE_PUBLICATION.md) for legend notes/order, export sizes and reproducibility settings.
