from pathlib import Path
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Plotting and Utilities"))
import preparation_spike_analysis as a
import current_response_publication as p


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.paths = []
        for i in range(4):
            path = self.folder / f"prep{i}.csv"
            pd.DataFrame([dict(epoch_label=e, channel="A", window_start_s=0, window_end_s=d, spike_count=rate*d)
                          for e, d, rate in (("Baseline", 1200, 2), ("Post 1", 60, 3+i), ("Post 2", 60, 2+i))]).to_csv(path, index=False)
            self.paths.append(path)
        self.currents = [-100, -200, -300, -450]
        self.data, self.epochs, *_ = a.normalize(a.load_preparations(self.paths, self.currents))
        self.styles = a.preparation_styles(self.data, {str(self.paths[0]): "D"})
        self.points = self.epochs[self.epochs.epoch_label == "Post 1"]
        self.groups = a.summarize(self.points, ["channel", "current_na", "epoch_label"], "normalized_rate_pct")

    def figure(self, cfg=None, points=None, plot_id="test"):
        fig = a.current_figure(self.points if points is None else points, self.groups, "Current response", self.styles, cfg, p.current_colors(self.data.current_na), plot_id)
        self.addCleanup(a.plt.close, fig)
        return fig

    def test_uniform_markers_and_collision_jitter(self):
        points = self.points.copy()
        points["current_na"] = -100
        points["normalized_rate_pct"] = [130., 130., 130., 80.]
        original = points.copy(deep=True)
        fig = self.figure(points=points)
        again = self.figure(points=points.sample(frac=1, random_state=42))
        pd.testing.assert_frame_equal(points, original)
        ax = fig.axes[0]
        self.assertIsNone(ax.get_legend())
        self.assertFalse(ax.texts)
        self.assertEqual(ax.get_xlabel(), "Applied cathodic current (nA)")
        self.assertEqual(fig.publication_metadata["title"], "Current response")
        self.assertEqual(fig._suptitle.get_text(), "Current response")
        records = fig.publication_metadata["observations"]
        self.assertEqual(records, again.publication_metadata["observations"])
        for record in records:
            self.assertEqual(record["normalized_rate_pct"], points.loc[points.preparation_id == record["preparation_id"], "normalized_rate_pct"].iloc[0])
            self.assertLessEqual(abs(record["plotted_current_na"]-record["current_na"]), 5)
            if record["normalized_rate_pct"] == 80:
                self.assertEqual(record["plotted_current_na"], -100)
        self.assertEqual(len(set(r["plotted_current_na"] for r in records if r["normalized_rate_pct"] == 130)), 3)
        for artist in ax.collections:
            np.testing.assert_array_equal(artist.get_sizes(), [25])
            np.testing.assert_array_equal(artist.get_facecolors(), ax.collections[0].get_facecolors())
        for artist in (fig._suptitle, ax.xaxis.label, ax.yaxis.label):
            bounds = artist.get_window_extent()
            self.assertGreaterEqual(bounds.x0, 0)
            self.assertGreaterEqual(bounds.y0, 0)
            self.assertLessEqual(bounds.x1, fig.bbox.width)
            self.assertLessEqual(bounds.y1, fig.bbox.height)

    def test_sizes_exports_vector_text_and_manifest(self):
        for size, width in (("single", 3.5), ("double", 7.2)):
            out = self.folder / size
            specs = a.discover_plot_specs(self.paths, self.currents, "normalized")
            current_specs = [spec for spec in specs if spec["kind"] == "current"]
            titles = {current_specs[0]["plot_id"]: "Custom GUI title: Post 1", current_specs[1]["plot_id"]: "   "}
            expected_titles = {spec["plot_id"]: a.resolved_title(spec, titles) for spec in current_specs}
            result = a.run_analysis(self.paths, self.currents, out, "normalized", titles=titles, publication={"size": size, "legend_note": "Synthetic test note\nDimension: 25 µm"})
            metadata = pd.read_csv(out / "CSVs/current_response_presentation.csv")
            self.assertEqual(len(metadata), 2)
            catalog = pd.read_csv(out / "CSVs/plot_catalog.csv").set_index("plot_id")
            for record in metadata.itertuples():
                expected = expected_titles[record.plot_id]
                self.assertEqual(json.loads(record.settings_json)["title"], expected)
                self.assertEqual(catalog.loc[record.plot_id, "title"], expected)
                exported = ET.parse(out / "Plots" / (record.plot_id + ".svg"))
                texts = [node.text for node in exported.findall('.//{http://www.w3.org/2000/svg}text')]
                self.assertIn(expected, texts)
            meta = json.loads(metadata.settings_json.iloc[0])
            self.assertEqual(meta["width_in"], width)
            self.assertEqual(meta["dpi"], 600)
            image = out / "Plots" / (metadata.plot_id.iloc[0] + ".png")
            with Image.open(image) as img:
                self.assertEqual(img.width, int(width*600))
            self.assertTrue(image.with_suffix(".pdf").exists())
            svg = ET.parse(image.with_suffix(".svg"))
            self.assertTrue(svg.findall('.//{http://www.w3.org/2000/svg}text'))
            pd.testing.assert_frame_equal(result["epochs"].drop(columns="recording_symbol"), self.epochs)
            a.run_analysis(self.paths, self.currents, out, "overlay")
            self.assertTrue((out / "CSVs/current_response_presentation.csv").exists())
            self.assertTrue(image.exists())
            keep = out / "Plots/user-note.txt"
            keep.write_text("untouched")
            a.run_analysis(self.paths[:1], self.currents[:1], out, "normalized", publication={"size": size})
            self.assertEqual(keep.read_text(), "untouched")


if __name__ == "__main__":
    unittest.main()
