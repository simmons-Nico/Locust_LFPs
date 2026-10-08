# Optional waveform-shape noise rejection

In SPIE's Spike Counting tab, expand **Optional waveform-shape noise rejection**.
The same settings are used when the full pipeline runs spike counting. The
accepted event list drives spike counts, waveform diagnostics, amplitude
analysis, and spike–LFP coupling.

Modes:

- `off` (default): preserves the previous detector and accepted events.
- `flag_only`: counts remain unchanged. Export flagged snippets and means for inspection.
- `reject`: removes flagged canonical candidates before refractory selection.

Defaults:

| Setting | Default | Meaning |
| --- | --- | --- |
| Maximum trough separation | 1.0 ms | Inclusive separation from the main trough, before or after |
| Depth fraction | 0.2 | Fraction of the aligned event's absolute filtered amplitude |
| Prominence fraction | 0.2 | Fraction of the aligned event's absolute filtered amplitude |
| Noise multiplier | 3.0 | Floor for both thresholds, in robust noise sigma |
| Inspection padding | 0.5 ms | Extra context to establish local trough prominence |

Depth = filtered-trace median minus trough voltage. Noise sigma = MAD/0.6745
over the recording/channel's filtered trace. Both minimum depth and minimum
prominence are the larger of their amplitude-relative threshold and the noise
floor. Prominence is calculated by `scipy.signal.find_peaks` on the inverted,
median-centered local trace. It measures how far the trough rises toward its
local bases; a shallow notch in a large trough does not qualify independently.

Negative events use their aligned main trough. Positive events use the deepest
significant negative trough within 1 ms (or the configured separation) of the
positive extremum. A positive event with no significant trough passes. One
negative trough plus a positive lobe passes: only two significant *negative*
troughs can trigger rejection. Ties and canonical candidate deduplication use
the existing detector rules.

The inspection radius and saved snippet half-width are
`2 × separation_ms + padding_ms`. This covers the shift from a positive peak
to its main trough and the search for a second trough. Sampling intervals are
used directly, with inclusive timing comparisons; unavailable temporal
resolution is not invented by interpolation.

## Outputs

Under the chosen spike output folder:

`spike_shape_qc/<mode>/<recording-order-and-name>/<channel>/`

- `candidate_audit.csv`: one row per unique amplitude/width-eligible canonical
  candidate before refractory selection; times, polarity, thresholds, depth,
  prominence, separation, flag/rejection reason, inspection completeness,
  before/after count membership, and snippet completeness.
- `rejected_waveforms.csv` (or `flagged_waveforms.csv`): all affected filtered
  snippets in long format, including partial snippets at recording edges.
  Event IDs join to the candidate audit.
- `pos_rejected_mean.csv/.png` and `neg_rejected_mean.csv/.png` (or `flagged`):
  mean, population SD, and N. The plot overlays at most 200 individual snippets;
  every snippet is exported. Only complete snippets contribute to the mean.
  N=0 exports explicitly show that no complete waveforms exist.
- `waveform_summary.csv`: affected totals, actual rejected totals, contributors
  to each mean, incomplete snippets, and sampling rate, separately by polarity.
- `count_comparison.csv`: counts before/after, net count reduction, unique
  candidate denominator, flagged/rejected counts, rejection percentage, and
  reason counts for each existing counting interval.
- `settings.json`: resolved settings, method label, sampling rate and source.

`spike_shape_provenance.json` records settings and audit folders even for off
mode. SPIE also records these in `spike_processing_provenance.json`.

Each recording/channel is exported separately, so rejected means never mix
sampling rates. Candidate rejections may exceed the change in counted spikes:
some candidates would already have been suppressed by the refractory rule.
Conversely, removal can let a nearby previously suppressed event survive.
The audit reports both quantities rather than equating them.

## Command line

```powershell
python "Spike Processing/Spike Count Multiple CSVs ordered.py" --shape-mode flag_only
python "Spike Processing/Spike Count Multiple CSVs ordered.py" --shape-mode reject --shape-separation-ms 1 --shape-depth-fraction 0.2 --shape-prominence-fraction 0.2 --shape-noise-multiplier 3 --shape-padding-ms 0.5
python waveform_analysis.py --shape-mode reject
```

The standalone waveform script retains its original detector in off mode.
When enabled it uses the authoritative order-3 spike counter and polarity
classification, then extracts waveforms from those accepted events. Its
existing band, amplitude, threshold and refractory settings still apply.
Thus opt-in mode also aligns its detector with SPIE; compare with off results
with this detector difference in mind. Events too close to a recording edge
to provide a full accepted waveform remain counted but cannot contribute a
complete waveform to analysis.

## Scientific basis and limits

This is a **Bombcell-inspired per-event adaptation**, not a validated universal
rule for locust spikes. Bombcell documents `maxNTroughs=1` and a relative
extremum threshold of 0.2:
<https://github.com/Julie-Fabre/bombcell/wiki/List-of-all-quality-metrics-and-parameters>.
The individual-event application, 1 ms cutoff, prominence requirement, noise
floor and inspection padding are project choices. Bombcell/SpikeInterface
ordinarily assess representative unit waveforms, not each threshold candidate.

Start with flag-only and inspect clean spikes and noise in your recordings.
Overlapping neural events can be flagged. Prominence depends on local context;
an inspection window truncated by a recording boundary may miss a trough or
underestimate prominence. `inspection_complete=False` identifies this case;
incomplete context alone never rejects an event. A minimum located at the very
first or last recorded sample cannot be established as a local trough.
