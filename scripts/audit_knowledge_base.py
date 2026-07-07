from __future__ import annotations

import argparse
import json
from pathlib import Path

from cropstate.knowledge import knowledge_coverage, load_knowledge_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit canonical CROPSTATE JSONL knowledge chunks.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--mode", choices=["all", "research", "production"], default="all")
    parser.add_argument("--minimum-topic-count", type=int, default=5)
    parser.add_argument("--minimum-stage-count", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()

    chunks = load_knowledge_chunks(args.input, mode=args.mode, include_sample=True)
    report = knowledge_coverage(chunks)
    report["topic_warnings"] = {
        topic: count for topic, count in report["by_topic"].items() if count < args.minimum_topic_count
    }
    report["stage_warnings"] = {
        stage: count for stage, count in report["stage_high_compatibility"].items() if count < args.minimum_stage_count
    }
    report["ready_for_production"] = (
        report["total_chunks"] > 0
        and not report["topic_warnings"]
        and not report["stage_warnings"]
        and report["restricted_action_chunks"] == 0
        and report["production_eligible_chunks"] == report["total_chunks"]
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
