# SPIE experimental timeline validation

Validated on 2026-09-08.

- 15 preparation-analysis/GUI tests passed.
- 10 experimental-timeline tests passed.
- GUI tests exercised source subprocess execution and bundled in-process script dispatch, including display names, reference override, compressed timeline, titles and baseline notices.
- PyInstaller completed successfully. The four bundled analysis scripts were compared byte-for-byte with the current sources; the latest GUI timing-editor function was verified in the executable archive.
- The rebuilt executable opened a responding SPIE main window and closed normally. The installed executable hash matches the tested build.
- The previous executable is preserved as `SPIE_before_experimental_timeline.exe` in this folder.

## Reference data and outputs

Used the six original spike-count CSVs in `C:/Users/simmons/Desktop/Exploring PSDs`, assigning currents from their filenames for this validation only. The GUI still requires one editable current assignment per file. Generated summaries were not used as preparations.

- `reference/`: six independent preparations, two at each of -100, -200 and -300 nA; 363 original windows; 204 experimental group time points; 23 PNG and 23 SVG figures.
- `compressed/`: the two -200 nA preparations with compressed treatment intervals.
- `single/`: one -200 nA preparation, long custom plot titles, no SEM.

Each directory contains `CSVs/`, `Plots/`, analysis notes and the managed-output manifest. Visually checked representative current-response, normalized, absolute-Hz, single-preparation and compressed figures for axes, individual observations, error bars, gaps, sample sizes, names and title layout.

All three currents have two contributors during baseline. Post 1 contributes one or two preparations depending on window availability. Post 2 remains individual-only in the experimental timeline because the different Post 1 durations produce non-equivalent experimental/epoch-relative timing. No interpolation or forced alignment was applied. This does not prevent the separate epoch-summary current-response plots from showing both preparation observations.

## Interpretation and limitations

The denominator is the duration-weighted firing rate over the final 20 minutes of labelled Baseline by default. An explicit reference interval replaces this consistently within the run. Partially intersecting windows contribute by overlap duration, assuming constant rate within each window. Displaying additional baseline observations does not change the denominator.

Experimental group keys require identical channel, intensity, epoch, epoch-relative bounds and experimental/display bounds. SEM uses sample SD with ddof=1 divided by sqrt(number of contributing independent preparations); SEM is missing below two contributors. Absolute-Hz sample size is independent of normalization exclusions.

Ambiguous native timestamps or missing intermediate epochs require measured timing or explicit epoch-start assignments. Reconstructed times use each preparation's own durations and configured treatment gaps. Compressed displays retain true experimental times in the exports.

The Katsuki-inspired figure follows the specified presentation behavior. The likely 1997 paper was identified, but its Fig. 1B was not directly verified; no exact reproduction is claimed.
