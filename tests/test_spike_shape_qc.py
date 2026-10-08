import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import spike_shape_qc as qc

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("shape_counter", ROOT / "Spike Processing" / "Spike Count Multiple CSVs ordered.py")
counter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(counter)


def event(p, amplitude=100, polarity="neg"):
    return dict(sample_index=p, amplitude_uv=amplitude, polarity=polarity)


class ShapeTests(unittest.TestCase):
    def test_separation_boundaries_and_rates(self):
        for fs in (10000, 20000, 30000):
            p = int(fs * .01)
            for delta, expected in ((int(fs*.001)-1, True), (int(fs*.001), True), (int(fs*.001)+1, False)):
                x = np.zeros(int(fs*.03)); x[p] = -100; x[p+delta] = -60
                row = qc.assess_events(x, [event(p)], fs, {"mode": "reject"})[0]
                self.assertEqual(row["shape_flagged"], expected)
                if expected:
                    self.assertAlmostEqual(row["trough_separation_ms"], delta*1000/fs)
            x = np.zeros(int(fs*.03)); x[p] = -100; x[p-int(fs*.001)] = -60
            self.assertTrue(qc.assess_events(x, [event(p)], fs, {"mode": "reject"})[0]["shape_flagged"])

    def test_clean_ripples_and_shallow_prominence(self):
        x = np.zeros(600); x[200] = -100; x[210] = -10
        self.assertFalse(qc.assess_events(x, [event(200)], 20000, {"mode": "reject"})[0]["shape_flagged"])
        # Deep second minimum, but only a 5 uV rise separates it from the first.
        x[200:212] = [-100, -90, -80, -70, -60, -55, -60, -55, -50, -40, -30, -20]
        self.assertFalse(qc.assess_events(x, [event(200)], 20000, {"mode": "reject"})[0]["shape_flagged"])

    def test_positive_anchor_and_single_biphasic_lobe(self):
        x = np.zeros(600); x[200] = 100; x[206] = -60
        e = event(200, polarity="pos")
        self.assertFalse(qc.assess_events(x, [e], 20000, {"mode": "reject"})[0]["shape_flagged"])
        x[222] = -50
        self.assertTrue(qc.assess_events(x, [e], 20000, {"mode": "reject"})[0]["shape_flagged"])
        x[206] = 0; x[222] = 0
        self.assertFalse(qc.assess_events(x, [e], 20000, {"mode": "reject"})[0]["shape_flagged"])

    def test_modes_and_refractory(self):
        fs = 20000; t = np.arange(2000)/fs
        x = np.random.default_rng(21).normal(0, .3, len(t))
        x[500] = -100; x[516] = -60; x[1000] = -90
        with patch.object(counter, "bandpass_filt", side_effect=lambda x, *a, **k: x):
            off = counter.detect_spikes_with_details(x, t, polarity="neg")
            explicit = counter.detect_spikes_with_details(x, t, polarity="neg", shape_settings={"mode": "off"})
            flag = counter.detect_spikes_with_details(x, t, polarity="neg", shape_settings={"mode": "flag_only"})
            reject = counter.detect_spikes_with_details(x, t, polarity="neg", shape_settings={"mode": "reject"})
        np.testing.assert_array_equal(off["peaks"], explicit["peaks"])
        np.testing.assert_array_equal(off["peaks"], flag["peaks"])
        np.testing.assert_array_equal(reject["peaks"], [1000])
        np.testing.assert_array_equal(reject["shape_before_peaks"], off["peaks"])
        self.assertEqual(sum(r["shape_flagged"] for r in flag["shape_reports"]), 2)

    def test_audit_incomplete_and_mean(self):
        fs = 20000; x = np.zeros(600); t = np.arange(len(x))/fs
        x[5] = -100; x[15] = -50; x[200] = -100; x[215] = -50
        cfg = {"mode": "reject"}
        reports = qc.assess_events(x, [event(5), event(200)], fs, cfg)
        with tempfile.TemporaryDirectory() as temp:
            qc.write_audit(temp, "rec", "A", x, t, fs, reports, [5, 200], [], [(0, .03)], cfg)
            audit = pd.read_csv(Path(temp)/"candidate_audit.csv")
            self.assertEqual(audit["rejected"].sum(), 2)
            summary = pd.read_csv(Path(temp)/"waveform_summary.csv").query("polarity == 'neg'").iloc[0]
            self.assertEqual(summary["waveforms_in_mean"], 1)
            self.assertEqual(summary["incomplete_waveforms"], 1)
            mean = pd.read_csv(Path(temp)/"neg_rejected_mean.csv")
            self.assertEqual(mean["n"].iloc[0], 1)
            self.assertEqual(mean.loc[np.isclose(mean.relative_time_ms, 0), "mean_uv"].iloc[0], -100)
            snippets = pd.read_csv(Path(temp)/"rejected_waveforms.csv")
            self.assertEqual(snippets.event_id.nunique(), 2)
            self.assertTrue((Path(temp)/"neg_rejected_mean.png").exists())
            counts = pd.read_csv(Path(temp)/"count_comparison.csv").iloc[0]
            self.assertEqual(counts["net_count_reduction"], 2)
            self.assertEqual(json.loads((Path(temp)/"settings.json").read_text())["settings"]["mode"], "reject")

    def test_pipeline_exports_match_accepted_counts(self):
        fs=20000; t=np.arange(4000)/fs
        x=np.random.default_rng(1).normal(0,.3,len(t)); x[500]=-100; x[516]=-60; x[2000]=-90
        with tempfile.TemporaryDirectory() as temp, patch.object(counter, "bandpass_filt", side_effect=lambda x,*a,**k:x):
            source=Path(temp)/"source.csv"; pd.DataFrame(dict(time_s=t, A=x)).to_csv(source,index=False)
            out=counter.process_csvs([str(source)], out_dir=str(Path(temp)/"out"), polarity="neg", window_sec=.1,
                                     shape_settings={"mode":"reject"}, return_outputs=True)
            counts=pd.read_csv(out["spike_count_csv"])
            events=pd.read_csv(out["spike_events_csv"])
            self.assertEqual(counts.spike_count.sum(),len(events))
            self.assertEqual(len(events),1)
            comparison=pd.read_csv(Path(out["spike_shape_audit_dirs"][0])/"count_comparison.csv")
            self.assertEqual(comparison.count_after.sum(),1)

    def test_invalid_settings(self):
        for cfg in ({"mode":"bad"}, {"separation_ms":0}, {"depth_fraction":-1}, {"padding_ms":float("nan")}):
            with self.assertRaises(ValueError): qc.settings(cfg)

    def test_noise_floor_and_flag_exports_empty_rerun(self):
        x = np.tile([-10., 10.], 300)
        x[200] = -100; x[210] = -25
        # A secondary trough that exceeds the relative threshold still fails
        # the robust noise floor (approximately 44.5 uV).
        self.assertFalse(qc.assess_events(x, [event(200)], 20000, {"mode":"reject"})[0]["shape_flagged"])
        x = np.zeros(600); x[200] = -100; x[210] = -50
        t = np.arange(600)/20000
        cfg={"mode":"flag_only"}
        with tempfile.TemporaryDirectory() as temp:
            reports=qc.assess_events(x,[event(200)],20000,cfg)
            qc.write_audit(temp,"rec","A",x,t,20000,reports,[200],[200],[(0,.03)],cfg)
            self.assertTrue((Path(temp)/"neg_flagged_mean.png").exists())
            counts=pd.read_csv(Path(temp)/"count_comparison.csv").iloc[0]
            self.assertEqual(counts.count_before,counts.count_after)
            self.assertEqual(counts.rejected_candidates,0)
            qc.write_audit(temp,"rec","A",x,t,20000,[],[],[],[(0,.03)],cfg)
            self.assertEqual(pd.read_csv(Path(temp)/"neg_flagged_mean.csv")["n"].max(),0)
            self.assertTrue(pd.read_csv(Path(temp)/"candidate_audit.csv").empty)

    def test_real_bandpass_preserves_clean_spike_and_rejects_double(self):
        fs=20000; t=np.arange(8000)/fs
        x=np.random.default_rng(11).normal(0,1,len(t))
        for time, amp in ((.05,-180),(.0508,-150),(.15,-180)):
            x += amp*np.exp(-.5*((t-time)/.00008)**2)
        details=counter.detect_spikes_with_details(x,t,polarity="neg",shape_settings={"mode":"reject"})
        self.assertTrue(np.any(np.abs(details["spike_times"]-.15)<.0002))
        self.assertFalse(np.any(np.abs(details["spike_times"]-.05)<.003))
        self.assertTrue(any(r["shape_flagged"] for r in details["shape_reports"]))


if __name__ == "__main__":
    unittest.main()
