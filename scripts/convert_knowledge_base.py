from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cropstate.constants import STAGE_NAMES

DEFAULT_WORKBOOK_NAME = "CROPSTATE_Sample_Knowledge_Base.xlsx"


COMPATIBILITY_COLUMN_CANDIDATES = {
    "establishment": ["c_establishment", "compatibility_establishment", "stage_compatibility_establishment", "s01_score", "s01"],
    "tillering": ["c_tillering", "compatibility_tillering", "stage_compatibility_tillering", "s02_score", "s02"],
    "stem_booting": ["c_stem_booting", "compatibility_stem_booting", "stage_compatibility_stem_booting", "s03_score", "s03"],
    "reproductive": ["c_reproductive", "compatibility_reproductive", "stage_compatibility_reproductive", "s04_score", "s04"],
    "grain_filling": ["c_grain_filling", "compatibility_grain_filling", "stage_compatibility_grain_filling", "s05_score", "s05"],
    "ripening": ["c_ripening", "compatibility_ripening", "stage_compatibility_ripening", "s06_score", "s06"],
}


def first_present(row: pd.Series, names: list[str], default: str = "") -> str:
    for name in names:
        if name in row and pd.notna(row[name]) and str(row[name]).strip():
            return str(row[name]).strip()
    return default


def parse_float(value: object, default: float = 0.0) -> float:
    if pd.isna(value) or str(value).strip() == "":
        return default
    return float(value)


def parse_stage_compatibility(row: pd.Series) -> list[float]:
    raw = first_present(row, ["stage_compatibility", "compatibility_vector"])
    if raw:
        parsed = json.loads(raw)
        if len(parsed) != len(STAGE_NAMES):
            raise ValueError(f"stage_compatibility must have {len(STAGE_NAMES)} values: {raw}")
        return [float(value) for value in parsed]

    values = []
    for stage in STAGE_NAMES:
        column_names = COMPATIBILITY_COLUMN_CANDIDATES[stage]
        value = 0.0
        for column in column_names:
            if column in row and pd.notna(row[column]) and str(row[column]).strip():
                value = parse_float(row[column])
                break
        values.append(value)
    return values


def convert_chunks(input_path: Path) -> list[dict]:
    if input_path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(input_path, sheet_name="Knowledge_Chunks")
    else:
        df = pd.read_csv(input_path)
    chunks = []
    for index, row in df.iterrows():
        chunk_id = first_present(row, ["chunk_id", "id"], f"chunk_{index:05d}")
        text = first_present(row, ["text", "content", "chunk_text"])
        if not text:
            raise ValueError(f"Missing chunk text at row {index}")
        chunks.append({
            "chunk_id": chunk_id,
            "source_id": first_present(row, ["source_id"]),
            "text": text,
            "topic": first_present(row, ["topic", "care_topic"], "general_crop_care"),
            "stage_compatibility": parse_stage_compatibility(row),
            "authority_score": parse_float(first_present(row, ["authority_score"], "0.5"), default=0.5),
            "applicability_type": first_present(row, ["applicability_type"]),
            "contraindicated_stages": first_present(row, ["contraindicated_stages", "stage_contraindicated"]),
            "review_status": first_present(row, ["review_status"], "unreviewed"),
        })
    return chunks


def resolve_input(input_path: str | None, knowledge_root: str | None) -> Path:
    if input_path:
        return Path(input_path)
    if not knowledge_root:
        raise ValueError("Provide --input or --knowledge-root")

    root = Path(knowledge_root)
    workbook = root / DEFAULT_WORKBOOK_NAME
    if workbook.exists():
        return workbook

    csv_path = root / "Knowledge_Chunks.csv"
    if csv_path.exists():
        return csv_path

    raise FileNotFoundError(
        f"Could not find {DEFAULT_WORKBOOK_NAME} or Knowledge_Chunks.csv under {root}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="CSV export or XLSX workbook containing Knowledge_Chunks sheet.")
    parser.add_argument("--knowledge-root", help="Folder containing the knowledge-base workbook or exported CSV files.")
    parser.add_argument("--output", default="data/knowledge_chunks.jsonl")
    args = parser.parse_args()

    input_path = resolve_input(args.input, args.knowledge_root)
    chunks = convert_chunks(input_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Wrote {len(chunks)} chunks to {output}")


if __name__ == "__main__":
    main()
