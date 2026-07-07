from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cropstate.constants import STAGE_NAMES
from cropstate.knowledge import load_knowledge_chunks
from cropstate.retrieval import HybridRetriever, build_topic_query, retrieve_and_rerank


def parse_belief(value: str | None, stage: str | None) -> np.ndarray:
    if value:
        belief = np.asarray([float(part) for part in value.split(",")], dtype=float)
        if belief.shape != (len(STAGE_NAMES),):
            raise ValueError(f"--belief must contain {len(STAGE_NAMES)} comma-separated values")
        return belief / belief.sum()
    if not stage:
        raise ValueError("Provide --belief or --stage")
    normalized = stage.strip().lower().replace("/", "_").replace("-", "_").replace(" ", "_")
    if normalized not in STAGE_NAMES:
        raise ValueError(f"Unknown stage: {stage}")
    belief = np.zeros(len(STAGE_NAMES), dtype=float)
    belief[STAGE_NAMES.index(normalized)] = 1.0
    return belief


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-topic CROPSTATE retrieval.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--stage")
    parser.add_argument("--belief")
    parser.add_argument("--confidence", type=float, default=1.0)
    parser.add_argument("--query")
    parser.add_argument("--mode", choices=["research", "production", "all"], default="research")
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fixed-beta", type=float)
    parser.add_argument("--allow-restricted", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    belief = parse_belief(args.belief, args.stage)
    stage_name = STAGE_NAMES[int(np.argmax(belief))]
    query = args.query or build_topic_query(args.topic, stage_name)
    chunks = load_knowledge_chunks(args.corpus, mode=args.mode)
    retriever = HybridRetriever(chunks, embedding_model=args.embedding_model)
    results = retrieve_and_rerank(
        retriever, query, args.topic, belief, args.confidence,
        depth=args.depth, top_k=args.top_k, fixed_beta=args.fixed_beta,
        allow_restricted=args.allow_restricted,
    )
    payload = {
        "topic": args.topic,
        "query": query,
        "belief": belief.tolist(),
        "confidence": args.confidence,
        "mode": args.mode,
        "results": results,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
