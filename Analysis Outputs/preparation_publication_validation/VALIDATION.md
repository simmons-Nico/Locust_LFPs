# Publication styling validation

32 focused tests passed: 16 preparation-analysis/GUI tests, 11 experimental-timeline tests, and 5 publication tests.

GUI tests exercised legend reordering, restore default, duplicate display names, selection changes, multiline Unicode notes, blank overrides, and source subprocess/bundled in-process plotting dispatch. Publication tests cover filled markers, true coordinates, selected symbols, colour mapping, offsets, metadata, PNG dimensions, SVG text, manifest retention and long-text bounds.

Reference inputs were the six original previously used preparation CSVs with explicit current intensities in their names. Additional files with unknown current or potentially different protocols were not silently assigned or pooled. No generated summaries were used as preparations.

- C:\Users\simmons\Desktop\Exploring PSDs\-200nA 20 mins (ALL Recordings) ALL Spikes divided by 2.csv (-200 nA)
- C:\Users\simmons\Desktop\Exploring PSDs\All_Recordings_22.06.2026_Spike_Count (-300nA).csv (-300 nA)
- C:\Users\simmons\Desktop\Exploring PSDs\All_Recordings_23.06.2026_Spike_Count (-100nA).csv (-100 nA)
- C:\Users\simmons\Desktop\Exploring PSDs\All_Recordings_25.06.2026_Spike_Count (-100nA).csv (-100 nA)
- C:\Users\simmons\Desktop\Exploring PSDs\All_Recordings_25.06.2026_Spike_Count (-200nA).csv (-200 nA)
- C:\Users\simmons\Desktop\Exploring PSDs\All_Recordings_25.06.2026_Spike_Count (-300nA).csv (-300 nA)

Outputs:
- single/: 3.5-inch publication figures with long titles and long recording names.
- double/: 7.2-inch publication figures with long titles and long recording names.
- offset_example/: compact names, blank annotation, and explicitly labelled 4 nA display spacing.

Visually inspected representative figures in both sizes and the optional offset example. Numerical data, baseline references, sample sizes and raw/time-course styling are preserved. PNGs are 300 dpi; SVG text stays editable. Each output directory contains CSVs/, Plots/, the ownership manifest and analysis notes.

PEDOT annotation is intentionally not populated: confirmation of what 150 ?m measures and which preparations share it remains pending. Layout checks use clearly labelled placeholder or synthetic notes rather than asserting unconfirmed specimen properties.

Build: PyInstaller completed successfully. Bundled source files match the workspace and new GUI control functions are present in the executable archive. See build.log. Startup and backup/install results are recorded separately after the smoke check.

Remaining limits: exact/near-overlapping observations can obscure one another with filled symbols; optional offsets default off. Many currents can produce similar blue shades. Very long legends increase figure height; use concise aliases or double-column width. These general export presets are not a guarantee of compliance with a particular journal.

Guidance: https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/ and https://journals.plos.org/plosone/s/figures; see Plotting and Utilities/CURRENT_RESPONSE_PUBLICATION.md for details and CLI configuration.

Startup passed: rebuilt SPIE opened a responding main window and closed normally. The previous executable was backed up to SPIE_before_publication_styling.exe in this directory. The tested build was installed at Apps/SPIE/Signal Processing for Insect Electrophysiology (SPIE).exe; both backup and installation hashes were verified.
