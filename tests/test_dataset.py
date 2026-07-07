import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from cropstate.constants import STAGE_TO_ID
from cropstate.dataset import RiceStageDataset, canonical_stage_label


class CanonicalStageLabelTests(unittest.TestCase):
    def test_normalizes_known_aliases(self):
        self.assertEqual(canonical_stage_label("S02_Tillering"), "tillering")
        self.assertEqual(canonical_stage_label("stem-booting"), "stem_booting")
        self.assertEqual(canonical_stage_label("Heading/Flowering"), "reproductive")

    def test_normalizes_whitespace_and_case(self):
        self.assertEqual(canonical_stage_label("  Grain Filling  "), "grain_filling")

    def test_raises_on_unknown_label(self):
        with self.assertRaises(KeyError):
            canonical_stage_label("not_a_real_stage")


class RiceStageDatasetTests(unittest.TestCase):
    def setUp(self):
        self._tmp = None

    def _make_dataset(self, tmp_path: Path) -> RiceStageDataset:
        image_path = tmp_path / "sample.jpg"
        Image.new("RGB", (8, 8), color=(0, 128, 0)).save(image_path)
        manifest = pd.DataFrame([
            {"image_id": "sample", "image_path": "sample.jpg", "macro_stage": "tillering"},
        ])
        return RiceStageDataset(manifest, tmp_path)

    def test_raises_on_missing_required_columns(self):
        manifest = pd.DataFrame([{"image_id": "sample"}])
        with self.assertRaises(ValueError):
            RiceStageDataset(manifest, ".")

    def test_len_matches_manifest_row_count(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._make_dataset(Path(directory))
            self.assertEqual(len(dataset), 1)

    def test_getitem_returns_image_label_and_id(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._make_dataset(Path(directory))
            image, label, image_id = dataset[0]
            self.assertEqual(image.size, (8, 8))
            self.assertEqual(int(label), STAGE_TO_ID["tillering"])
            self.assertEqual(image_id, "sample")

    def test_getitem_raises_on_missing_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            manifest = pd.DataFrame([
                {"image_id": "missing", "image_path": "missing.jpg", "macro_stage": "tillering"},
            ])
            dataset = RiceStageDataset(manifest, Path(directory))
            with self.assertRaises(FileNotFoundError):
                dataset[0]


if __name__ == "__main__":
    unittest.main()
