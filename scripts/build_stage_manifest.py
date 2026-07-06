from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cropstate.constants import STAGE_BBCH_RANGES, STAGE_NAMES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parent_image_id(path: Path) -> str:
    return path.stem.split("_subset_overlap", 1)[0]


def stage_from_folder_name(folder_name: str) -> str | None:
    for index, stage in enumerate(STAGE_NAMES, start=1):
        if folder_name.strip() == f"{index:02d}_{stage}":
            return stage
    return None


def build_manifest(data_root: Path) -> pd.DataFrame:
    rows = []
    for stage_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        stage = stage_from_folder_name(stage_dir.name)
        if stage is None:
            continue
        for image_path in sorted(path for path in stage_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
            relative = image_path.relative_to(data_root)
            parent = parent_image_id(image_path)
            subdirs = relative.parts[1:-1]
            capture_session = "/".join(subdirs) if subdirs else "unknown"
            rows.append({
                "image_id": image_path.stem,
                "image_path": relative.as_posix(),
                "parent_image_id": parent,
                "field_id": f"{stage}:{capture_session}:{parent}",
                "capture_session": capture_session,
                "capture_date": "",
                "region": capture_session if capture_session != "unknown" else "",
                "season": "unknown",
                "variety": "unknown",
                "bbch_code": STAGE_BBCH_RANGES[stage],
                "macro_stage": stage,
                "source": "stage_folder",
                "license": "user_provided",
                "annotator_1": "",
                "annotator_2": "",
                "adjudicated_label": stage,
                "review_status": "folder_label",
                "split": "unassigned",
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="data/stage_folder_manifest.csv")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(Path(args.data_root))
    manifest.to_csv(output, index=False)
    print(f"Wrote {len(manifest)} rows to {output}")
    if not manifest.empty:
        print(manifest["macro_stage"].value_counts().to_string())


if __name__ == "__main__":
    main()
