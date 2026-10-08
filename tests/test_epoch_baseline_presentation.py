from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Plotting and Utilities"))
import preparation_spike_analysis as a


class EpochBaselineTests(unittest.TestCase):
    def data(self):
        rows = []
        for epoch, rates in (("Baseline", [8., 10., 12., 10.]),
                             ("Post 1", [3., 4., 5., 4.]),
                             ("Post 2", [8., 9., 10., 9.])):
            for i, rate in enumerate(rates):
                rows.append(dict(preparation_id="p", source_file="p.csv", channel="A", current_na=-200.,
                                 display_name="p", display_label="p", epoch_label=epoch,
                                 window_start_s=i*300., window_end_s=(i+1)*300.,
                                 duration_s=300., spike_count=rate*300., firing_rate_hz=rate))
        return a.normalize(pd.DataFrame(rows))[:2]

    def test_welch_sem_and_holm_match_reference(self):
        data, epochs = self.data()
        original_data, original_epochs = data.copy(deep=True), epochs.copy(deep=True)
        result = a.epoch_window_statistics(data).set_index("epoch_label")
        expected = stats.ttest_ind([3, 4, 5, 4], [8, 10, 12, 10], equal_var=False)
        self.assertAlmostEqual(result.loc["Post 1", "p_value"], expected.pvalue)
        self.assertAlmostEqual(result.loc["Post 1", "normalized_sem_pct"], stats.sem([3, 4, 5, 4])*10)
        self.assertAlmostEqual(result.loc["Post 1", "p_value_holm"], expected.pvalue*2)
        self.assertTrue(np.isnan(result.loc["Baseline", "p_value"]))
        pd.testing.assert_frame_equal(data, original_data)
        pd.testing.assert_frame_equal(epochs, original_epochs)

    def test_cutoff_weights_and_insufficient_windows(self):
        data, _ = self.data()
        data.loc[data.epoch_label == "Baseline", "baseline_reference_start_s"] = 450.
        data = data[(data.epoch_label != "Post 1") | (data.window_start_s == 0)]
        result = a.epoch_window_statistics(data).set_index("epoch_label")
        base = result.loc["Baseline"]
        self.assertEqual(base.n_windows, 3)
        self.assertAlmostEqual(base.window_mean_rate_hz, np.average([10, 12, 10], weights=[150, 300, 300]))
        self.assertAlmostEqual(base.effective_n_windows, 750**2 / (150**2+300**2+300**2))
        self.assertTrue(np.isnan(result.loc["Post 1", "normalized_sem_pct"]))
        self.assertTrue(np.isnan(result.loc["Post 1", "p_value"]))

    def test_bars_start_at_100_with_error_bars_and_brackets(self):
        from matplotlib.container import ErrorbarContainer
        data, epochs = self.data()
        e = epochs.set_index("epoch_label").reindex(["Baseline", "Post 1", "Post 2"])
        fig = a.baseline_change_figure(e, "Baseline comparison", a.epoch_window_statistics(data))
        try:
            ax = fig.axes[0]
            np.testing.assert_allclose([p.get_y() for p in ax.patches], 100)
            np.testing.assert_allclose([p.get_height()+p.get_y() for p in ax.patches], e.normalized_rate_pct)
            self.assertTrue(any(isinstance(c, ErrorbarContainer) for c in ax.containers))
            self.assertEqual(len(ax.texts), 2)
            self.assertIn("ns", [t.get_text() for t in ax.texts])
        finally:
            a.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
