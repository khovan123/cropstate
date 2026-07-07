import unittest

import numpy as np

from cropstate.knowledge import KnowledgeChunk
from cropstate.retrieval import rerank


class RetrievalTest(unittest.TestCase):
    def test_vector_changes_order(self):
        first = KnowledgeChunk(
            chunk_id="first",
            text="This complete test record contains enough words to validate a deterministic ranking path without loading any external model or remote resource during the unit test.",
            topic="topic_a",
            stage_compatibility=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            authority_score=0.5,
        )
        second = KnowledgeChunk(
            chunk_id="second",
            text="This second complete record also contains enough words to exercise deterministic ranking while remaining independent from external files services models and network access.",
            topic="topic_a",
            stage_compatibility=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            authority_score=0.5,
        )
        ranked = rerank(
            {"first": 0.5, "second": 0.5},
            {"first": first, "second": second},
            np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            1.0,
        )
        self.assertEqual(ranked[0], "first")


if __name__ == "__main__":
    unittest.main()
