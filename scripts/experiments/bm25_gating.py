"""Closed-loop stage-gated retrieval, BM25-only base (Tier B#4, torch-free).

Same closed loop as evaluate_retrieval.py, but the base ranking uses BM25 alone so
the whole thing runs without sentence-transformers/torch (this box's CUDA driver
starves torch processes). The scientific claim — that a calibrated stage belief
driving soft gating lowers the stage-incompatible retrieval rate (SIRR) toward the
true-stage oracle — is independent of whether the base retriever is BM25 or hybrid.

Relevance is programmatic (from chunk stage_compatibility), so no agronomist is needed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cropstate.constants import STAGE_NAMES
from cropstate.knowledge import load_knowledge_chunks
from cropstate.metrics import ndcg_at_k, precision_recall_at_k, sirr_at_k
from cropstate.retrieval import BM25Okapi, hard_filter, minmax, rerank, tokenize, build_topic_query


def one_hot(stage_idx):
    v = np.zeros(len(STAGE_NAMES)); v[stage_idx] = 1.0
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="CROPSTATE_KNOWLEDGE_BASE/chunks/rice_knowledge_complete.jsonl")
    parser.add_argument("--scenarios", default="CROPSTATE_RESULTS/novelty/belief_scenarios.csv")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--use-calibrated", action="store_true",
                        help="Use the temperature-calibrated belief column for soft gating.")
    parser.add_argument("--output", default="CROPSTATE_RESULTS/novelty/retrieval_gating.json")
    args = parser.parse_args()

    chunks = load_knowledge_chunks(args.corpus, mode="research")
    chunk_map = {c.chunk_id: c for c in chunks}
    ids = [c.chunk_id for c in chunks]
    topic_of = {c.chunk_id: c.topic for c in chunks}
    bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    with open(args.scenarios, encoding="utf-8") as fh:
        scenarios = list(csv.DictReader(fh))

    belief_col = "calibrated_stage_belief" if args.use_calibrated else "stage_belief"
    methods = ["B0_ungated", "B2_hard_filter", "P_adaptive_soft", "oracle_reference"]
    agg = {m: {"precision_at_k": [], "recall_at_k": [], "ndcg_at_k": [], "sirr_at_k": []} for m in methods}
    rows = []

    for sc in scenarios:
        topic = sc["topic"]
        true_idx = STAGE_NAMES.index(sc["ground_truth_stage"])
        belief = np.array(json.loads(sc[belief_col]), dtype=float); belief /= belief.sum()
        confidence = float(sc["confidence"])
        pred_idx = int(belief.argmax())
        relevant = set(sc["relevant_chunk_ids"].split("|")) if sc["relevant_chunk_ids"] else set()
        incompatible = set(sc["incompatible_chunk_ids"].split("|")) if sc["incompatible_chunk_ids"] else set()
        relevance = {cid: 1.0 for cid in relevant}

        # BM25 base over topic-eligible chunks (+general fallback).
        eligible = [i for i, cid in enumerate(ids) if topic_of[cid] in (topic, "general_crop_care")]
        query = build_topic_query(topic)
        scores = np.asarray(bm25.get_scores(tokenize(query)), dtype=float)
        order = [ids[i] for i in sorted(eligible, key=lambda i: -scores[i])[:args.depth]]
        base_scores = minmax({cid: float(scores[ids.index(cid)]) for cid in order})

        rankings = {
            "B0_ungated": sorted(base_scores, key=base_scores.get, reverse=True),
            "B2_hard_filter": hard_filter(base_scores, chunk_map, pred_idx),
            "P_adaptive_soft": rerank(base_scores, chunk_map, belief, confidence),
            "oracle_reference": rerank(base_scores, chunk_map, one_hot(true_idx), 1.0),
        }
        for m, ranking in rankings.items():
            p, r = precision_recall_at_k(ranking, relevant, args.k)
            agg[m]["precision_at_k"].append(p)
            agg[m]["recall_at_k"].append(r)
            agg[m]["ndcg_at_k"].append(ndcg_at_k(ranking, relevance, args.k))
            agg[m]["sirr_at_k"].append(sirr_at_k(ranking, incompatible, args.k))

    summary = {m: {k: float(np.mean(v)) for k, v in metrics.items()} for m, metrics in agg.items()}
    payload = {"base_retriever": "bm25", "belief": belief_col, "k": args.k,
               "scenario_count": len(scenarios), "summary": summary}
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(f"base=BM25 belief={belief_col} scenarios={len(scenarios)}")
    for m in methods:
        s = summary[m]
        print(f"  {m:18} SIRR={s['sirr_at_k']:.3f}  P@k={s['precision_at_k']:.3f}  "
              f"R@k={s['recall_at_k']:.3f}  nDCG={s['ndcg_at_k']:.3f}")


if __name__ == "__main__":
    main()
