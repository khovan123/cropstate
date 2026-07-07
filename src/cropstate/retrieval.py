from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np
try:
    from rank_bm25 import BM25Okapi
except ImportError:  # Lightweight fallback used by tests/minimal environments.
    class BM25Okapi:
        def __init__(self, corpus):
            self.corpus = corpus
            import math
            self.avgdl = sum(len(doc) for doc in corpus) / max(1, len(corpus))
            df = {}
            for doc in corpus:
                for token in set(doc):
                    df[token] = df.get(token, 0) + 1
            n = len(corpus)
            self.idf = {token: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for token, freq in df.items()}

        def get_scores(self, query_tokens):
            scores = []
            k1, b = 1.5, 0.75
            for doc in self.corpus:
                counts = {}
                for token in doc:
                    counts[token] = counts.get(token, 0) + 1
                score = 0.0
                for token in query_tokens:
                    tf = counts.get(token, 0)
                    if tf == 0:
                        continue
                    denom = tf + k1 * (1 - b + b * len(doc) / max(self.avgdl, 1e-9))
                    score += self.idf.get(token, 0.0) * tf * (k1 + 1) / denom
                scores.append(score)
            return scores

from .constants import STAGE_BBCH_RANGES, STAGE_DISPLAY_NAMES, STAGE_NAMES
from .knowledge import KnowledgeChunk

Chunk = KnowledgeChunk

VI_TOPIC_LABELS = {
    "water_management": "quản lý nước",
    "nutrient_management": "quản lý dinh dưỡng và bón phân",
    "pest_risk": "nguy cơ và quản lý sâu hại",
    "disease_risk": "nguy cơ và quản lý bệnh hại",
    "weed_management": "quản lý cỏ dại và lúa cỏ",
    "harvest_readiness": "đánh giá chín và thời điểm thu hoạch",
    "residue_management": "quản lý rơm rạ sau thu hoạch",
    "climate_adaptation": "canh tác thích ứng biến đổi khí hậu",
    "general_crop_care": "kỹ thuật chăm sóc lúa",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", text.lower(), flags=re.UNICODE)


def build_topic_query(topic: str, stage_name: str, language: str = "vi") -> str:
    stage = stage_name.strip().lower()
    display = STAGE_DISPLAY_NAMES.get(stage, stage_name)
    bbch = STAGE_BBCH_RANGES.get(stage, "")
    if language == "vi":
        topic_label = VI_TOPIC_LABELS.get(topic, topic.replace("_", " "))
        return f"Bằng chứng {topic_label} phù hợp với lúa ở giai đoạn {display}, BBCH {bbch}."
    return f"Rice {topic.replace('_', ' ')} evidence applicable to {display}, BBCH {bbch}."


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        *,
        encoder: Any | None = None,
    ):
        if not chunks:
            raise ValueError("HybridRetriever requires at least one chunk")
        self.chunks = chunks
        self.ids = [chunk.chunk_id for chunk in chunks]
        self.bm25 = BM25Okapi([tokenize(chunk.text) for chunk in chunks])
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(embedding_model)
        self.encoder = encoder
        self.embeddings = np.asarray(
            self.encoder.encode([chunk.text for chunk in chunks], normalize_embeddings=True),
            dtype=float,
        )

    def eligible_indices(self, topic: str | None, include_general_fallback: bool = True) -> np.ndarray:
        if not topic:
            return np.arange(len(self.chunks))
        exact = [index for index, chunk in enumerate(self.chunks) if chunk.topic == topic]
        if exact:
            if include_general_fallback and topic != "general_crop_care":
                exact.extend(index for index, chunk in enumerate(self.chunks) if chunk.topic == "general_crop_care")
            return np.asarray(sorted(set(exact)), dtype=int)
        if include_general_fallback:
            fallback = [index for index, chunk in enumerate(self.chunks) if chunk.topic == "general_crop_care"]
            if fallback:
                return np.asarray(fallback, dtype=int)
        return np.arange(len(self.chunks))

    def retrieve(
        self,
        query: str,
        depth: int = 50,
        *,
        topic: str | None = None,
        include_general_fallback: bool = True,
    ) -> tuple[list[str], list[str]]:
        eligible = self.eligible_indices(topic, include_general_fallback)
        bm25_scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)
        bm25_order = eligible[np.argsort(-bm25_scores[eligible])[:depth]]
        query_vector = np.asarray(self.encoder.encode([query], normalize_embeddings=True)[0], dtype=float)
        dense_scores = self.embeddings @ query_vector
        dense_order = eligible[np.argsort(-dense_scores[eligible])[:depth]]
        return [self.ids[index] for index in bm25_order], [self.ids[index] for index in dense_order]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k0: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, document_id in enumerate(ranked, start=1):
            scores[document_id] += 1.0 / (k0 + rank)
    return dict(scores)


def minmax(scores: dict[str, float], eps: float = 1e-12) -> dict[str, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if abs(high - low) <= eps:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def rerank_scores(
    base_scores: dict[str, float],
    chunks: dict[str, Chunk],
    belief: np.ndarray,
    confidence: float,
    beta_min: float = 0.05,
    beta_max: float = 0.35,
    source_weight: float = 0.10,
    fixed_beta: float | None = None,
    *,
    allow_restricted: bool = False,
) -> dict[str, dict[str, float]]:
    belief = np.asarray(belief, dtype=float)
    if belief.shape != (len(STAGE_NAMES),):
        raise ValueError(f"belief must contain {len(STAGE_NAMES)} probabilities")
    if belief.sum() <= 0:
        raise ValueError("belief must have a positive sum")
    belief = belief / belief.sum()
    beta = fixed_beta if fixed_beta is not None else beta_min + (beta_max - beta_min) * float(confidence)
    alpha = 1.0 - beta - source_weight
    if alpha < 0:
        raise ValueError("Invalid weights: alpha is negative")
    result: dict[str, dict[str, float]] = {}
    for document_id, base in base_scores.items():
        chunk = chunks[document_id]
        if chunk.restricted_action and not allow_restricted:
            continue
        stage = float(np.dot(belief, np.asarray(chunk.stage_compatibility, dtype=float)))
        final = alpha * float(base) + beta * stage + source_weight * chunk.authority_score
        result[document_id] = {
            "final_score": final,
            "base_score": float(base),
            "stage_score": stage,
            "authority_score": chunk.authority_score,
            "alpha": alpha,
            "beta": beta,
            "source_weight": source_weight,
        }
    return result


def rerank(
    base_scores: dict[str, float],
    chunks: dict[str, Chunk],
    belief: np.ndarray,
    confidence: float,
    beta_min: float = 0.05,
    beta_max: float = 0.35,
    source_weight: float = 0.10,
    fixed_beta: float | None = None,
    *,
    allow_restricted: bool = False,
) -> list[str]:
    scores = rerank_scores(
        base_scores, chunks, belief, confidence, beta_min, beta_max,
        source_weight, fixed_beta, allow_restricted=allow_restricted,
    )
    return [document_id for document_id, _ in sorted(scores.items(), key=lambda item: item[1]["final_score"], reverse=True)]


def hard_filter(
    base_scores: dict[str, float],
    chunks: dict[str, Chunk],
    predicted_stage: int,
    *,
    allow_restricted: bool = False,
) -> list[str]:
    eligible = {
        document_id: score
        for document_id, score in base_scores.items()
        if chunks[document_id].stage_compatibility[predicted_stage] > 0
        and (allow_restricted or not chunks[document_id].restricted_action)
    }
    return [document_id for document_id, _ in sorted(eligible.items(), key=lambda item: item[1], reverse=True)]


def retrieve_and_rerank(
    retriever: HybridRetriever,
    query: str,
    topic: str,
    belief: np.ndarray,
    confidence: float,
    *,
    depth: int = 50,
    top_k: int = 5,
    fixed_beta: float | None = None,
    allow_restricted: bool = False,
) -> list[dict[str, Any]]:
    bm25_ranked, dense_ranked = retriever.retrieve(query, depth=depth, topic=topic)
    base_scores = minmax(reciprocal_rank_fusion([bm25_ranked, dense_ranked]))
    chunk_map = {chunk.chunk_id: chunk for chunk in retriever.chunks}
    component_scores = rerank_scores(
        base_scores,
        chunk_map,
        belief,
        confidence,
        fixed_beta=fixed_beta,
        allow_restricted=allow_restricted,
    )
    ranked_ids = sorted(component_scores, key=lambda document_id: component_scores[document_id]["final_score"], reverse=True)
    results: list[dict[str, Any]] = []
    for rank, document_id in enumerate(ranked_ids[:top_k], start=1):
        chunk = chunk_map[document_id]
        results.append({
            "rank": rank,
            "chunk_id": document_id,
            "text": chunk.text,
            "topic": chunk.topic,
            "source_id": chunk.source_id,
            "source_title": chunk.source_title,
            "source_url": chunk.source_url,
            "page_start": chunk.page_start,
            "review_status": chunk.review_status,
            **component_scores[document_id],
        })
    return results
