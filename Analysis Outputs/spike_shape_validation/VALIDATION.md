# Waveform-shape rejection validation

Validated with Python 3.13, NumPy, SciPy, pandas, matplotlib and Tkinter.

- 9 new shape-QC tests passed: clean/rippled waveforms, shallow prominence,
  before/after secondary troughs, <1 / =1 / >1 ms boundaries at 10/20/30 kHz,
  positive/biphasic waveforms, robust-noise floor, modes, actual band-pass
  filtering, incomplete edge snippets, audit counts, means and N=0 reruns.
- 18 existing polarity/waveform tests passed.
- 2 existing refractory-selection/scaling tests passed.
- GUI controls were instantiated with a withdrawn Tk root. Requests and all
  three processing modes were exercised through `run_shared_spike_pipeline`.
- Real-filter synthetic validation produced 3 counted events in off/flag-only
  and 2 in reject. Two significant-trough candidates were rejected, with
  N=2 in the rejected mean and net count reduction of 1.
- Standalone waveform analysis ran in reject mode on the same source and
  retained the same 2 accepted events.
- Inspected the generated rejected-mean PNG for readable axes, title, N, SD
  band and waveform alignment.
- Compilation and changed-file whitespace checks passed.
- Rebuilt the SPIE executable. Verified its embedded QC module, spike-counter
  source, and compiled GUI against the current source, then installed it in
  `Apps/SPIE`. Preserved the previous executable in this validation folder.

The broader `test_spike*.py` run initially passed 39 of 41 existing tests.
Two failures belong to `test_spike_experimental_timeline.py`: missing `n = 1`
annotations and a missing legend. That test imports the unchanged
`preparation_spike_analysis` and `spike_experimental_timeline` modules; these
failures do not involve the modified spike detector or QC module.

Reproduce the export/GUI validation with:

```powershell
python "Analysis Outputs/spike_shape_validation/validate_pipeline.py"
python -m unittest discover -s tests -p test_spike_shape_qc.py
```

These are software and synthetic-signal checks. Locust recording thresholds
still require inspection and scientific validation, preferably in flag-only
mode before excluding events.
