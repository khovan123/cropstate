from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cropstate.constants import STAGE_BBCH_RANGES, STAGE_DISPLAY_NAMES, STAGE_NAMES
from cropstate.knowledge import KnowledgeChunk, load_knowledge_chunks

FACET_BUCKETS = ["fertilizer", "conditions", "pest_disease_prevention", "general"]
LOOKAHEAD_FACETS = ["fertilizer", "conditions", "pest_disease_prevention"]


def load_corpora(paths: list[Path], mode: str) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    seen_ids: set[str] = set()
    for path in paths:
        for chunk in load_knowledge_chunks(path, mode=mode):
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            chunks.append(chunk)
    return chunks


def dominant_stage_index(chunk: KnowledgeChunk, min_stage_score: float) -> int | None:
    best_index = max(range(len(STAGE_NAMES)), key=lambda index: chunk.stage_compatibility[index])
    if chunk.stage_compatibility[best_index] < min_stage_score:
        return None
    return best_index


def as_evidence_item(chunk: KnowledgeChunk, note: str = "") -> dict[str, Any]:
    item = {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "topic": chunk.topic,
        "facet": chunk.facet,
        "source_title": chunk.source_title,
        "source_organization": chunk.source_organization,
        "source_url": chunk.source_url,
        "authority_score": chunk.authority_score,
        "language": chunk.language,
    }
    if note:
        item["note"] = note
    return item


def rank(chunks: list[KnowledgeChunk]) -> list[KnowledgeChunk]:
    return sorted(chunks, key=lambda chunk: (-chunk.authority_score, chunk.chunk_id))


def build_stage_profiles(
    chunks: list[KnowledgeChunk],
    min_stage_score: float,
    max_items_per_bucket: int,
    max_lookahead_items: int,
) -> dict[str, Any]:
    by_stage: dict[int, list[KnowledgeChunk]] = {index: [] for index in range(len(STAGE_NAMES))}
    for chunk in chunks:
        stage_index = dominant_stage_index(chunk, min_stage_score)
        if stage_index is not None:
            by_stage[stage_index].append(chunk)

    stages: list[dict[str, Any]] = []
    coverage_warnings: list[str] = []
    for index, stage_name in enumerate(STAGE_NAMES):
        stage_chunks = by_stage[index]
        buckets: dict[str, list[dict[str, Any]]] = {facet: [] for facet in FACET_BUCKETS}
        tagged_next = rank([chunk for chunk in stage_chunks if chunk.facet == "next_stage_action"])
        for chunk in rank(stage_chunks):
            facet = chunk.facet if chunk.facet in FACET_BUCKETS else "general"
            if len(buckets[facet]) < max_items_per_bucket:
                buckets[facet].append(as_evidence_item(chunk))

        next_stage_actions = [as_evidence_item(chunk) for chunk in tagged_next[:max_items_per_bucket]]
        if index < len(STAGE_NAMES) - 1:
            next_stage_name = STAGE_NAMES[index + 1]
            lookahead = rank([chunk for chunk in by_stage[index + 1] if chunk.facet in LOOKAHEAD_FACETS])
            seen_ids = {item["chunk_id"] for item in next_stage_actions}
            added = 0
            for chunk in lookahead:
                if added >= max_lookahead_items or len(next_stage_actions) >= max_items_per_bucket:
                    break
                if chunk.chunk_id in seen_ids:
                    continue
                next_stage_actions.append(as_evidence_item(chunk, note=f"preview of {next_stage_name}"))
                seen_ids.add(chunk.chunk_id)
                added += 1

        stage_profile = {
            "stage": stage_name,
            "display_name": STAGE_DISPLAY_NAMES[stage_name],
            "bbch_range": STAGE_BBCH_RANGES[stage_name],
            "fertilizer": buckets["fertilizer"],
            "conditions": buckets["conditions"],
            "pest_disease_prevention": buckets["pest_disease_prevention"],
            "next_stage_actions": next_stage_actions,
            "general": buckets["general"],
            "evidence_count": len(stage_chunks),
        }
        stages.append(stage_profile)
        for facet in ["fertilizer", "conditions", "pest_disease_prevention"]:
            if not stage_profile[facet]:
                coverage_warnings.append(f"{stage_name}: no {facet} evidence")
        if not stage_profile["next_stage_actions"]:
            coverage_warnings.append(f"{stage_name}: no next_stage_actions evidence")

    return {
        "stages": stages,
        "by_stage": {stage["stage"]: stage for stage in stages},
        "coverage_warnings": coverage_warnings,
        "total_chunks_considered": len(chunks),
        "min_stage_score": min_stage_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Roll up CROPSTATE knowledge chunks into a per-stage report: fertilizer, "
        "conditions, pest/disease prevention, and what to prepare for the next stage."
    )
    parser.add_argument("--corpus", action="append", required=True, help="JSONL knowledge corpus. Repeatable.")
    parser.add_argument("--mode", choices=["all", "research", "production"], default="research")
    parser.add_argument("--min-stage-score", type=float, default=0.8)
    parser.add_argument("--max-items-per-bucket", type=int, default=20)
    parser.add_argument("--max-lookahead-items", type=int, default=8)
    parser.add_argument("--output", default="stage_profiles.json")
    args = parser.parse_args()

    chunks = load_corpora([Path(path) for path in args.corpus], args.mode)
    profiles = build_stage_profiles(chunks, args.min_stage_score, args.max_items_per_bucket, args.max_lookahead_items)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "output": str(output_path),
        "total_chunks_considered": profiles["total_chunks_considered"],
        "evidence_count_by_stage": {stage["stage"]: stage["evidence_count"] for stage in profiles["stages"]},
        "coverage_warnings": profiles["coverage_warnings"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
