from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from cropstate.constants import STAGE_ALIASES, STAGE_NAMES
from cropstate.knowledge import load_knowledge_chunks
from cropstate.metrics import ndcg_at_k, precision_recall_at_k, sirr_at_k
from cropstate.retrieval import HybridRetriever, build_topic_query, hard_filter, minmax, reciprocal_rank_fusion, rerank


def parse_ids(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(item) for item in value}
    text = str(value).strip()
    if not text:
        return set()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return {str(item) for item in parsed}
    except json.JSONDecodeError:
        pass
    return {item.strip() for item in text.replace(",", "|").split("|") if item.strip()}


def read_scenarios(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_stage(stage: str) -> str:
    key = stage.strip().lower().replace("/", "_").replace("-", "_").replace(" ", "_")
    normalized = STAGE_ALIASES.get(key, key)
    if normalized not in STAGE_NAMES:
        raise ValueError(f"Unknown stage: {stage}")
    return normalized


def one_hot(stage: str) -> np.ndarray:
    normalized = normalize_stage(stage)
    vector = np.zeros(len(STAGE_NAMES), dtype=float)
    vector[STAGE_NAMES.index(normalized)] = 1.0
    return vector


def parse_belief(value: Any, fallback_stage: str) -> np.ndarray:
    if value in (None, ""):
        return one_hot(fallback_stage)
    if isinstance(value, list):
        belief = np.asarray(value, dtype=float)
    else:
        text = str(value).strip()
        try:
            belief = np.asarray(json.loads(text), dtype=float)
        except json.JSONDecodeError:
            belief = np.asarray([float(part) for part in text.split(",")], dtype=float)
    if belief.shape != (len(STAGE_NAMES),) or belief.sum() <= 0:
        raise ValueError("stage_belief must contain six non-negative values with a positive sum")
    return belief / belief.sum()


def parse_relevance(scenario: dict[str, Any]) -> dict[str, float]:
    raw = scenario.get("relevance_grades")
    if raw:
        parsed = raw if isinstance(raw, dict) else json.loads(str(raw))
        return {str(key): float(value) for key, value in parsed.items()}
    return {document_id: 1.0 for document_id in parse_ids(scenario.get("relevant_chunk_ids"))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CROPSTATE retrieval baselines.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--mode", choices=["research", "production", "all"], default="research")
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", default="results/retrieval_evaluation.json")
    args = parser.parse_args()

    chunks = load_knowledge_chunks(args.corpus, mode=args.mode)
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    retriever = HybridRetriever(chunks, embedding_model=args.embedding_model)
    scenarios = read_scenarios(Path(args.scenarios))
    rows: list[dict[str, Any]] = []

    for scenario_index, scenario in enumerate(scenarios, start=1):
        scenario_id = scenario.get("scenario_id") or f"scenario_{scenario_index:04d}"
        topic = str(scenario["topic"])
        true_stage = normalize_stage(str(scenario["ground_truth_stage"]))
        predicted_stage = normalize_stage(str(scenario.get("predicted_stage") or true_stage))
        belief = parse_belief(scenario.get("stage_belief"), predicted_stage)
        confidence = float(scenario.get("confidence") or belief.max())
        # B0/B2/B3/P share a stage-free query; only B1 (query expansion) gets a stage-bearing query.
        base_query = str(scenario.get("query") or build_topic_query(topic))
        expanded_query = build_topic_query(topic, predicted_stage)
        relevance = parse_relevance(scenario)
        relevant = {document_id for document_id, grade in relevance.items() if grade > 0}
        incompatible = parse_ids(scenario.get("incompatible_chunk_ids", scenario.get("nonmatching_chunk_ids")))

        bm25_ranked, dense_ranked = retriever.retrieve(base_query, depth=args.depth, topic=topic)
        base_scores = minmax(reciprocal_rank_fusion([bm25_ranked, dense_ranked]))
        expanded_bm25, expanded_dense = retriever.retrieve(expanded_query, depth=args.depth, topic=topic)
        expanded_scores = minmax(reciprocal_rank_fusion([expanded_bm25, expanded_dense]))
        methods = {
            "B0_ungated": sorted(base_scores, key=base_scores.get, reverse=True),
            "B1_query_expansion": sorted(expanded_scores, key=expanded_scores.get, reverse=True),
            "B2_hard_filter": hard_filter(base_scores, chunk_map, int(np.argmax(belief))),
            "B3_fixed_soft": rerank(base_scores, chunk_map, belief, confidence, fixed_beta=0.20),
            "P_adaptive_soft": rerank(base_scores, chunk_map, belief, confidence),
            "oracle_reference": rerank(base_scores, chunk_map, one_hot(true_stage), 1.0),
        }
        for method, ranking in methods.items():
            precision, recall = precision_recall_at_k(ranking, relevant, args.k)
            rows.append({
                "scenario_id": scenario_id,
                "topic": topic,
                "ground_truth_stage": true_stage,
                "predicted_stage": predicted_stage,
                "method": method,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "ndcg_at_k": ndcg_at_k(ranking, relevance, args.k),
                "sirr_at_k": sirr_at_k(ranking, incompatible, args.k),
                "top_ids": ranking[: args.k],
            })

    summary: dict[str, dict[str, float]] = {}
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        summary[method] = {
            metric: float(np.mean([row[metric] for row in subset]))
            for metric in ["precision_at_k", "recall_at_k", "ndcg_at_k", "sirr_at_k"]
        }
    payload = {"k": args.k, "scenario_count": len(scenarios), "summary": summary, "rows": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
