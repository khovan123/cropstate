import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_stage_profiles import build_stage_profiles  # noqa: E402
from cropstate.knowledge import KnowledgeChunk  # noqa: E402


def make_chunk(chunk_id: str, stage_compatibility: list[float], facet: str, authority_score: float = 0.8) -> KnowledgeChunk:
    return KnowledgeChunk.from_mapping({
        "chunk_id": chunk_id,
        "text": f"Synthetic evidence text for {chunk_id} used only in unit tests for stage profiling.",
        "topic": "nutrient_management",
        "stage_compatibility": stage_compatibility,
        "authority_score": authority_score,
        "facet": facet,
    })


class StageProfilesTests(unittest.TestCase):
    def test_groups_chunks_by_dominant_stage_and_facet(self):
        chunks = [
            make_chunk("A1", [1.0, 0.55, 0, 0, 0, 0], "fertilizer"),
            make_chunk("A2", [1.0, 0.55, 0, 0, 0, 0], "conditions"),
            make_chunk("B1", [0.55, 1.0, 0, 0, 0, 0], "pest_disease_prevention"),
        ]
        profiles = build_stage_profiles(chunks, min_stage_score=0.8, max_items_per_bucket=20, max_lookahead_items=8)

        establishment = profiles["by_stage"]["establishment"]
        self.assertEqual([item["chunk_id"] for item in establishment["fertilizer"]], ["A1"])
        self.assertEqual([item["chunk_id"] for item in establishment["conditions"]], ["A2"])
        self.assertEqual(establishment["evidence_count"], 2)

        tillering = profiles["by_stage"]["tillering"]
        self.assertEqual([item["chunk_id"] for item in tillering["pest_disease_prevention"]], ["B1"])

    def test_next_stage_actions_uses_explicit_tag_and_lookahead_preview(self):
        chunks = [
            make_chunk("N1", [1.0, 0.55, 0, 0, 0, 0], "next_stage_action"),
            make_chunk("N2", [0.55, 1.0, 0, 0, 0, 0], "fertilizer"),
        ]
        profiles = build_stage_profiles(chunks, min_stage_score=0.8, max_items_per_bucket=20, max_lookahead_items=8)

        establishment = profiles["by_stage"]["establishment"]
        ids = [item["chunk_id"] for item in establishment["next_stage_actions"]]
        self.assertIn("N1", ids)
        self.assertIn("N2", ids)
        preview_item = next(item for item in establishment["next_stage_actions"] if item["chunk_id"] == "N2")
        self.assertIn("note", preview_item)

    def test_below_threshold_chunk_is_excluded_and_flagged_in_coverage(self):
        chunks = [make_chunk("Low", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "fertilizer")]
        profiles = build_stage_profiles(chunks, min_stage_score=0.8, max_items_per_bucket=20, max_lookahead_items=8)

        for stage in profiles["stages"]:
            self.assertEqual(stage["evidence_count"], 0)
        self.assertIn("establishment: no fertilizer evidence", profiles["coverage_warnings"])


if __name__ == "__main__":
    unittest.main()
