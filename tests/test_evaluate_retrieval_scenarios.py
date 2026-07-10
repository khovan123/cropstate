import sys
import unittest
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate_retrieval import normalize_stage, one_hot, parse_belief, parse_ids, read_scenarios  # noqa: E402
from cropstate.constants import STAGE_NAMES  # noqa: E402
from cropstate.knowledge import load_knowledge_chunks  # noqa: E402
from cropstate.metrics import precision_recall_at_k  # noqa: E402
from cropstate.retrieval import HybridRetriever, build_topic_query, hard_filter, minmax, reciprocal_rank_fusion, rerank, tokenize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATHS = [
    REPO_ROOT / "CROPSTATE_KNOWLEDGE_BASE" / "chunks" / "rice_knowledge_irri_en.jsonl",
]
SCENARIOS_PATH = REPO_ROOT / "data" / "retrieval_scenarios.csv"


class _HashingEncoder:
    # Deterministic bag-of-hashed-tokens stand-in for a sentence-transformers encoder;
    # avoids downloading a real model just to exercise the ranking pipeline in tests.
    def __init__(self, dims: int = 256):
        self.dims = dims

    def encode(self, texts, normalize_embeddings=True):
        vectors = np.zeros((len(texts), self.dims), dtype=float)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                vectors[row, zlib.crc32(token.encode("utf-8")) % self.dims] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


@unittest.skipUnless(all(path.exists() for path in CORPUS_PATHS), "knowledge base chunk files not built")
@unittest.skipUnless(SCENARIOS_PATH.exists(), "data/retrieval_scenarios.csv not found")
class ImageDerivedScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chunks = []
        for path in CORPUS_PATHS:
            chunks.extend(load_knowledge_chunks(path, mode="research"))
        cls.retriever = HybridRetriever(chunks, encoder=_HashingEncoder())
        cls.chunk_map = {chunk.chunk_id: chunk for chunk in cls.retriever.chunks}
        cls.scenarios = read_scenarios(SCENARIOS_PATH)

    def _rank(self, scenario, k=5):
        # Mirrors evaluate_retrieval.py: B0/B2/B3/P share one stage-free query (paper Table baselines).
        topic = scenario["topic"]
        ground_truth_stage = normalize_stage(scenario["ground_truth_stage"])
        predicted_stage = normalize_stage(scenario["predicted_stage"])
        belief = parse_belief(scenario.get("stage_belief"), predicted_stage)
        confidence = float(scenario.get("confidence") or belief.max())
        query = build_topic_query(topic)

        bm25_ranked, dense_ranked = self.retriever.retrieve(query, depth=50, topic=topic)
        base_scores = minmax(reciprocal_rank_fusion([bm25_ranked, dense_ranked]))
        rankings = {
            "hard_top1": hard_filter(base_scores, self.chunk_map, STAGE_NAMES.index(predicted_stage)),
            "adaptive_soft": rerank(base_scores, self.chunk_map, belief, confidence),
            "oracle": rerank(base_scores, self.chunk_map, one_hot(ground_truth_stage), 1.0),
        }
        return {method: ranking[:k] for method, ranking in rankings.items()}

    def test_every_scenario_has_two_real_test_images_and_matching_relevance_labels(self):
        self.assertEqual({scenario["scenario_id"] for scenario in self.scenarios}, {
            "img_test1_reproductive_pest", "img_test2_tillering_water",
        })
        for scenario in self.scenarios:
            image_path = REPO_ROOT / scenario["image_path"]
            self.assertTrue(image_path.exists(), f"missing fixture image: {image_path}")
            # Relevance labels must reference chunks that exist in the (IRRI) corpus,
            # otherwise the whole evaluation silently degenerates to all-zero scores.
            relevant = set(parse_ids(scenario.get("relevant_chunk_ids")))
            self.assertTrue(relevant, f"{scenario['scenario_id']} has no relevant chunks")
            self.assertTrue(
                relevant.issubset(self.chunk_map),
                f"{scenario['scenario_id']} references chunk ids absent from the corpus: "
                f"{sorted(relevant - set(self.chunk_map))}",
            )

    def test_hard_top1_filtering_misses_all_relevant_evidence_when_stage_is_misclassified(self):
        # Both scenarios are real, low-confidence misclassifications: predicted_stage != ground_truth_stage.
        for scenario in self.scenarios:
            with self.subTest(scenario_id=scenario["scenario_id"]):
                relevant = set(parse_ids(scenario.get("relevant_chunk_ids")))
                rankings = self._rank(scenario)
                precision, recall = precision_recall_at_k(rankings["hard_top1"], relevant, k=5)
                self.assertEqual(recall, 0.0)
                self.assertEqual(precision, 0.0)

    def test_oracle_stage_conditioning_recovers_relevant_evidence_that_hard_filtering_misses(self):
        # Aggregate over scenarios: true-stage conditioning must recover, on average,
        # relevant evidence that top-1 hard filtering drops on misclassified inputs.
        # (Per-scenario recovery is encoder-dependent; the mean is the stable claim.)
        hard_recalls, oracle_recalls = [], []
        for scenario in self.scenarios:
            relevant = set(parse_ids(scenario.get("relevant_chunk_ids")))
            rankings = self._rank(scenario)
            _, hard_recall = precision_recall_at_k(rankings["hard_top1"], relevant, k=5)
            _, oracle_recall = precision_recall_at_k(rankings["oracle"], relevant, k=5)
            hard_recalls.append(hard_recall)
            oracle_recalls.append(oracle_recall)
        self.assertGreater(float(np.mean(oracle_recalls)), float(np.mean(hard_recalls)))


if __name__ == "__main__":
    unittest.main()
