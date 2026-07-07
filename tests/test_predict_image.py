import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from cropstate.constants import STAGE_NAMES  # noqa: E402

CHECKPOINT = Path(__file__).resolve().parent.parent / "CROPSTATE_RESULTS" / "vision_final" / "best_checkpoint.pt"

try:
    from predict_image import predict_image  # noqa: E402
    IMPORT_ERROR = ""
except ImportError as error:  # pragma: no cover - exercised only when torch/timm are missing
    predict_image = None
    IMPORT_ERROR = str(error)


@unittest.skipUnless(CHECKPOINT.exists(), f"checkpoint not found at {CHECKPOINT}")
@unittest.skipUnless(predict_image is not None, IMPORT_ERROR)
class PredictImageTests(unittest.TestCase):
    def _predict(self, filename: str) -> dict:
        image_path = Path(__file__).resolve().parent / filename
        return predict_image(CHECKPOINT, image_path)

    def test_stage_belief_is_a_valid_probability_distribution_for_each_test_image(self):
        for filename in ["test1.jpg", "test2.jpg"]:
            with self.subTest(filename=filename):
                result = self._predict(filename)
                self.assertEqual(set(result["stage_belief"]), set(STAGE_NAMES))
                self.assertAlmostEqual(sum(result["stage_belief"].values()), 1.0, places=5)
                self.assertIn(result["predicted_stage"], STAGE_NAMES)
                self.assertEqual(
                    result["confidence"],
                    result["stage_belief"][result["predicted_stage"]],
                )
                self.assertEqual(
                    result["predicted_stage"],
                    max(result["stage_belief"], key=result["stage_belief"].get),
                )

    def test_low_confidence_flowering_photo_does_not_confidently_favor_early_growth(self):
        # test1.jpg shows visible anthesis (panicles with white anthers); early-vegetative belief should stay low.
        result = self._predict("test1.jpg")
        early_stage_mass = result["stage_belief"]["establishment"] + result["stage_belief"]["tillering"]
        self.assertLess(early_stage_mass, 0.5)


if __name__ == "__main__":
    unittest.main()
