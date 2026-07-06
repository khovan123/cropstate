from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    chunk_id: str
    text: str
    topic: str
    stage_compatibility: list[float]
    authority_score: float


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.ids = [c.chunk_id for c in chunks]
        self.bm25 = BM25Okapi([c.text.lower().split() for c in chunks])
        self.encoder = SentenceTransformer(embedding_model)
        self.embeddings = self.encoder.encode([c.text for c in chunks], normalize_embeddings=True)

    def retrieve(self, query: str, depth: int = 50) -> tuple[list[str], list[str]]:
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_order = np.argsort(-bm25_scores)[:depth]
        query_vector = self.encoder.encode([query], normalize_embeddings=True)[0]
        dense_scores = self.embeddings @ query_vector
        dense_order = np.argsort(-dense_scores)[:depth]
        return [self.ids[i] for i in bm25_order], [self.ids[i] for i in dense_order]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k0: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (k0 + rank)
    return dict(scores)


def minmax(scores: dict[str, float], eps: float = 1e-12) -> dict[str, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    return {k: (v - low) / (high - low + eps) for k, v in scores.items()}


def rerank(
    base_scores: dict[str, float],
    chunks: dict[str, Chunk],
    belief: np.ndarray,
    confidence: float,
    beta_min: float = 0.05,
    beta_max: float = 0.35,
    source_weight: float = 0.10,
    fixed_beta: float | None = None,
) -> list[str]:
    beta = fixed_beta if fixed_beta is not None else beta_min + (beta_max - beta_min) * confidence
    alpha = 1.0 - beta - source_weight
    if alpha < 0:
        raise ValueError("Invalid weights")
    final = {}
    for doc_id, base in base_scores.items():
        chunk = chunks[doc_id]
        stage = float(np.dot(belief, np.asarray(chunk.stage_compatibility)))
        final[doc_id] = alpha * base + beta * stage + source_weight * chunk.authority_score
    return [doc_id for doc_id, _ in sorted(final.items(), key=lambda item: item[1], reverse=True)]


def hard_filter(base_scores: dict[str, float], chunks: dict[str, Chunk], predicted_stage: int) -> list[str]:
    eligible = {
        doc_id: score for doc_id, score in base_scores.items()
        if chunks[doc_id].stage_compatibility[predicted_stage] > 0
    }
    return [doc_id for doc_id, _ in sorted(eligible.items(), key=lambda item: item[1], reverse=True)]
