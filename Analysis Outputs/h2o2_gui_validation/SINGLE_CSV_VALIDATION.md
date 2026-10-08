# H2O2 single-CSV correction

This supersedes the separate during/after recording workflow in VALIDATION.md.

One H2O2 positive-control selection now represents one preparation CSV. The loader recognizes baseline, during and post in epoch_label, ignoring case and surrounding whitespace. It validates windows after canonicalizing labels. Same-file/channel normalization and pairing are automatic. The GUI retains only an optional concentration dialog; manual locust IDs, measurement-epoch selectors and external baseline selection were removed.

Validation passed: 6 updated H2O2 tests, 6 cathodic loading/normalization regression tests, and 2 current-response publication tests (14 total). Coverage includes GUI selection and bundled completion, mixed current/H2O2 exports without pairing configuration, shared baseline values, different preparation identities, channels, custom reference windows, missing measurements, invalid baselines/labels, and SVG/600-dpi PNG output. Tests use temporary synthetic fixtures, not invented experimental data.

Build log: build_single_csv.log.

Build and packaged-source verification passed. The executable opened a responding main window and closed normally. Installed at Apps/SPIE/Signal Processing for Insect Electrophysiology (SPIE).exe; backup and installation hashes verified. Backup: C:\Users\simmons\Desktop\Locust_LFP_Python\Analysis Outputs\h2o2_gui_validation\SPIE_before_single_csv_20260909_205013.exe
