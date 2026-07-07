from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from cropstate.knowledge import KnowledgeChunk, knowledge_coverage, load_knowledge_chunks, write_knowledge_chunks

PREFERRED_NAMES = [
    "chunks/rice_knowledge_complete.jsonl",
    "rice_knowledge_complete.jsonl",
    "chunks/rice_knowledge_nonrestricted.jsonl",
    "CROPSTATE_Knowledge_Base_Complete.xlsx",
    "CROPSTATE_Sample_Knowledge_Base.xlsx",
    "Knowledge_Chunks.csv",
]


def resolve_input(input_path: str | None, knowledge_root: str | None) -> Path:
    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    if not knowledge_root:
        raise ValueError("Provide --input or --knowledge-root")
    root = Path(knowledge_root)
    for relative in PREFERRED_NAMES:
        candidate = root / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No supported knowledge-base file found under {root}. Tried: {PREFERRED_NAMES}")


def records_from_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        workbook = pd.ExcelFile(path)
        sheet = "Knowledge_Chunks" if "Knowledge_Chunks" in workbook.sheet_names else "Chunks"
        frame = pd.read_excel(path, sheet_name=sheet)
    else:
        frame = pd.read_csv(path)
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def convert_table(path: Path, mode: str, include_sample: bool) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    seen: set[str] = set()
    for index, record in enumerate(records_from_table(path), start=1):
        chunk = KnowledgeChunk.from_mapping(record, fallback_id=f"chunk_{index:05d}")
        chunk.validate()
        if chunk.chunk_id in seen:
            raise ValueError(f"Duplicate chunk_id: {chunk.chunk_id}")
        seen.add(chunk.chunk_id)
        if not include_sample and chunk.review_status == "sample_only_not_agronomic_ground_truth":
            continue
        if mode == "production" and not (
            chunk.review_status in {"reviewed", "domain_reviewed", "approved"}
            and chunk.production_eligible
            and not chunk.restricted_action
        ):
            continue
        if mode == "research" and chunk.review_status == "excluded":
            continue
        chunks.append(chunk)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert and validate a CROPSTATE knowledge base.")
    parser.add_argument("--input", help="JSONL, CSV or XLSX knowledge-base file")
    parser.add_argument("--knowledge-root", help="Knowledge-base folder")
    parser.add_argument("--output", default="data/knowledge_chunks.jsonl")
    parser.add_argument("--report", default="data/knowledge_coverage.json")
    parser.add_argument("--mode", choices=["all", "research", "production"], default="research")
    parser.add_argument("--include-sample", action="store_true")
    args = parser.parse_args()

    input_path = resolve_input(args.input, args.knowledge_root)
    if input_path.suffix.lower() == ".jsonl":
        chunks = load_knowledge_chunks(input_path, mode=args.mode, include_sample=args.include_sample)
    else:
        chunks = convert_table(input_path, args.mode, args.include_sample)
    if not chunks:
        raise RuntimeError(
            f"No chunks remain in mode={args.mode}. For production mode, approve records, set production_eligible=true, "
            "and keep restricted_action=false."
        )
    write_knowledge_chunks(chunks, args.output)
    report = knowledge_coverage(chunks)
    report["input"] = str(input_path)
    report["mode"] = args.mode
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
