from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd

from cropstate.constants import NON_TRAINING_STAGE_ALIASES, STAGE_ALIASES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUTPUT_COLUMNS = [
    "image_id",
    "image_path",
    "parent_image_id",
    "field_id",
    "capture_session",
    "capture_date",
    "region",
    "season",
    "variety",
    "macro_stage",
    "source",
    "source_url",
    "drive_url",
    "license",
    "annotator_1",
    "annotator_2",
    "adjudicated_label",
    "review_status",
    "split",
    "sha256",
]


def slug(value: object) -> str:
    text = str(value).strip().lower().replace("/", "_").replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    return "_".join(part for part in text.split("_") if part)


def canonical_stage_label(value: object) -> str:
    label_slug = slug(value)
    try:
        return STAGE_ALIASES[label_slug]
    except KeyError as exc:
        raise KeyError(f"Unknown macro_stage label: {value!r}") from exc


def first_present(row: pd.Series, names: list[str], default: str = "") -> str:
    for name in names:
        if name in row and pd.notna(row[name]) and str(row[name]).strip():
            return str(row[name]).strip()
    return default


def parse_bool(value: object, default: bool = True) -> bool:
    if pd.isna(value):
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "usable", "use", "ok"}:
        return True
    if text in {"0", "false", "no", "n", "unusable", "exclude", "bad"}:
        return False
    return default


def parent_from_name(name: str) -> str:
    stem = Path(name).stem
    return stem.split("_subset_overlap", 1)[0]


def index_images(data_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in data_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.name, []).append(path)
    return index


def resolve_image_path(file_name: str, data_root: Path, image_index: dict[str, list[Path]]) -> str:
    candidate = Path(file_name)
    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        try:
            return candidate.relative_to(data_root).as_posix()
        except ValueError:
            return candidate.as_posix()

    direct = data_root / candidate
    if direct.exists():
        return direct.relative_to(data_root).as_posix()

    matches = image_index.get(candidate.name, [])
    if len(matches) == 1:
        return matches[0].relative_to(data_root).as_posix()
    if len(matches) > 1:
        locations = ", ".join(path.relative_to(data_root).as_posix() for path in matches[:5])
        raise ValueError(f"Ambiguous file_name {file_name!r}; matches: {locations}")
    raise FileNotFoundError(f"Could not locate {file_name!r} under {data_root}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_or_exclusion(value: object) -> tuple[str | None, str]:
    label_slug = slug(value)
    if label_slug in NON_TRAINING_STAGE_ALIASES:
        return None, NON_TRAINING_STAGE_ALIASES[label_slug]
    try:
        return canonical_stage_label(value), ""
    except KeyError:
        return None, "unknown_label"


def convert_manifest(input_path: Path, data_root: Path, compute_checksum: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if input_path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        source = pd.read_excel(input_path, sheet_name="Image_Manifest_Template")
    else:
        source = pd.read_csv(input_path)
    image_index = index_images(data_root)
    training_rows = []
    excluded_rows = []

    for _, row in source.iterrows():
        file_name = first_present(row, ["image_path", "file_name", "filename", "image_file"])
        label = first_present(row, ["macro_stage", "final_label", "label", "stage"])
        usable = parse_bool(row.get("usable", True), default=True)
        stage, exclusion_reason = stage_or_exclusion(label)

        base = row.to_dict()
        base["original_label"] = label
        if not file_name:
            base["exclude_reason"] = "missing_file_name"
            excluded_rows.append(base)
            continue
        if not usable:
            base["exclude_reason"] = "usable_false"
            excluded_rows.append(base)
            continue
        if stage is None:
            base["exclude_reason"] = exclusion_reason
            excluded_rows.append(base)
            continue

        try:
            image_path = resolve_image_path(file_name, data_root, image_index)
        except (FileNotFoundError, ValueError) as exc:
            base["exclude_reason"] = str(exc)
            excluded_rows.append(base)
            continue
        absolute_image_path = data_root / image_path
        parent_image_id = first_present(row, ["parent_image_id"], parent_from_name(file_name))
        capture_session = first_present(row, ["capture_session"], first_present(row, ["source_name", "region"], "unknown"))
        field_id = first_present(row, ["field_id"], f"{capture_session}:{parent_image_id}")

        converted = {
            "image_id": first_present(row, ["image_id"], Path(file_name).stem),
            "image_path": image_path,
            "parent_image_id": parent_image_id,
            "field_id": field_id,
            "capture_session": capture_session,
            "capture_date": first_present(row, ["capture_date"]),
            "region": first_present(row, ["region"]),
            "season": first_present(row, ["season"], "unknown"),
            "variety": first_present(row, ["variety"], "unknown"),
            "macro_stage": stage,
            "source": first_present(row, ["source_name", "source"], "manifest"),
            "source_url": first_present(row, ["source_url"]),
            "drive_url": first_present(row, ["drive_url"]),
            "license": first_present(row, ["license"], "unknown"),
            "annotator_1": first_present(row, ["annotator_1"]),
            "annotator_2": first_present(row, ["annotator_2"]),
            "adjudicated_label": label,
            "review_status": first_present(row, ["review_status"], "unreviewed"),
            "split": first_present(row, ["split"], "unassigned"),
        }
        if compute_checksum:
            converted["sha256"] = sha256_file(absolute_image_path)
        training_rows.append(converted)

    train_df = pd.DataFrame(training_rows)
    if train_df.empty:
        train_df = pd.DataFrame(columns=OUTPUT_COLUMNS if compute_checksum else OUTPUT_COLUMNS[:-1])
    return train_df, pd.DataFrame(excluded_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV export or XLSX workbook containing Image_Manifest_Template.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="data/image_manifest.csv")
    parser.add_argument("--excluded-output", default="data/image_manifest_excluded.csv")
    parser.add_argument("--duplicate-report", default="data/image_manifest_duplicates.csv")
    parser.add_argument("--no-checksum", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output = Path(args.output)
    excluded_output = Path(args.excluded_output)
    duplicate_report = Path(args.duplicate_report)
    train_df, excluded_df = convert_manifest(Path(args.input), data_root, compute_checksum=not args.no_checksum)

    output.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output, index=False)
    if not excluded_df.empty:
        excluded_output.parent.mkdir(parents=True, exist_ok=True)
        excluded_df.to_csv(excluded_output, index=False)

    if "sha256" in train_df.columns:
        duplicates = train_df[train_df.duplicated("sha256", keep=False)].sort_values("sha256")
        if not duplicates.empty:
            duplicate_report.parent.mkdir(parents=True, exist_ok=True)
            duplicates.to_csv(duplicate_report, index=False)

    print(f"Wrote training manifest: {output} ({len(train_df)} rows)")
    print(f"Excluded rows: {len(excluded_df)}")
    if "sha256" in train_df.columns:
        duplicate_count = int(train_df.duplicated("sha256", keep=False).sum())
        print(f"Exact duplicate image rows by sha256: {duplicate_count}")


if __name__ == "__main__":
    main()
