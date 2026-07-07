from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd

from cropstate.constants import NON_TRAINING_STAGE_ALIASES, STAGE_ALIASES, STAGE_BBCH_RANGES, STAGE_NAMES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_WORKBOOK_NAME = "CROPSTATE_Sample_Knowledge_Base.xlsx"
WORKBOOK_CANDIDATES = [
    "CROPSTATE_Knowledge_Base_Complete.xlsx",
    "CROPSTATE_Sample_Knowledge_Base.xlsx",
    "CROPSTATE Sample Knowledge Base.xlsx",
]
STAGE_DIR_RE = re.compile(r"^\s*(?P<number>0?[1-6])[\s_-]+(?P<name>.+?)\s*$")
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


def normalize_stage_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def stage_from_folder_name(folder_name: str) -> str | None:
    match = STAGE_DIR_RE.match(folder_name)
    if not match:
        return None
    index = int(match.group("number")) - 1
    if not 0 <= index < len(STAGE_NAMES):
        return None
    expected = STAGE_NAMES[index]
    parsed_name = normalize_stage_name(match.group("name"))
    aliases = {
        "stem_booting": {"stem_booting", "stem", "booting", "stem_elongation_booting"},
        "grain_filling": {"grain_filling", "grain_development"},
    }
    if parsed_name and parsed_name not in aliases.get(expected, {expected}):
        return None
    return expected


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


def build_manifest_from_stage_folders(data_root: Path, compute_checksum: bool) -> pd.DataFrame:
    rows = []
    for stage_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        stage = stage_from_folder_name(stage_dir.name)
        if stage is None:
            continue
        for image_path in sorted(path for path in stage_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
            relative = image_path.relative_to(data_root)
            parent = parent_from_name(image_path.name)
            subdirs = relative.parts[1:-1]
            capture_session = "/".join(subdirs) if subdirs else "unknown"
            row = {
                "image_id": image_path.stem,
                "image_path": relative.as_posix(),
                "parent_image_id": f"{stage}:{capture_session}:{parent}",
                "field_id": f"{stage}:{capture_session}:{parent}",
                "capture_session": capture_session,
                "capture_date": "",
                "region": "" if capture_session == "unknown" else capture_session,
                "season": "unknown",
                "variety": "unknown",
                "macro_stage": stage,
                "source": "stage_folder_auto",
                "source_url": "",
                "drive_url": "",
                "license": "user_provided",
                "annotator_1": "",
                "annotator_2": "",
                "adjudicated_label": stage,
                "review_status": "folder_label_unreviewed",
                "split": "unassigned",
            }
            if compute_checksum:
                row["sha256"] = sha256_file(image_path)
            rows.append(row)
    columns = OUTPUT_COLUMNS if compute_checksum else OUTPUT_COLUMNS[:-1]
    return pd.DataFrame(rows, columns=columns)


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


def resolve_input(input_path: str | None, knowledge_root: str | None) -> Path:
    if input_path:
        return Path(input_path)
    if not knowledge_root:
        raise ValueError("Provide --input or --knowledge-root")

    root = Path(knowledge_root)
    for workbook_name in WORKBOOK_CANDIDATES:
        workbook = root / workbook_name
        if workbook.exists():
            return workbook

    csv_path = root / "Image_Manifest_Template.csv"
    if csv_path.exists():
        return csv_path

    raise FileNotFoundError(
        f"Could not find {', '.join(WORKBOOK_CANDIDATES)} or Image_Manifest_Template.csv under {root}"
    )


def convert_with_folder_fallback(
    input_path: Path | None,
    data_root: Path,
    compute_checksum: bool,
    allow_folder_fallback: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if input_path is not None:
        try:
            train_df, excluded_df = convert_manifest(input_path, data_root, compute_checksum=compute_checksum)
        except ValueError as exc:
            if not allow_folder_fallback or "Worksheet named" not in str(exc):
                raise
            train_df = pd.DataFrame()
            excluded_df = pd.DataFrame([{"exclude_reason": str(exc), "input_path": str(input_path)}])
            print(f"Could not read Image_Manifest_Template from {input_path}; falling back to stage folders.")
        if not train_df.empty or not allow_folder_fallback:
            return train_df, excluded_df, f"knowledge_manifest:{input_path}"
        print("No training rows found in Image_Manifest_Template; falling back to stage folders.")
    elif not allow_folder_fallback:
        raise ValueError("No input manifest found and folder fallback is disabled.")
    else:
        excluded_df = pd.DataFrame()
        print("No Image_Manifest_Template found; building manifest from stage folders.")

    train_df = build_manifest_from_stage_folders(data_root, compute_checksum=compute_checksum)
    if train_df.empty:
        raise ValueError(f"No stage images found under {data_root}")
    return train_df, excluded_df, "stage_folders"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="CSV export or XLSX workbook containing Image_Manifest_Template.")
    parser.add_argument("--knowledge-root", help="Folder containing the knowledge-base workbook or exported CSV files.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="data/image_manifest.csv")
    parser.add_argument("--excluded-output", default="data/image_manifest_excluded.csv")
    parser.add_argument("--duplicate-report", default="data/image_manifest_duplicates.csv")
    parser.add_argument("--no-checksum", action="store_true")
    parser.add_argument(
        "--no-folder-fallback",
        action="store_true",
        help="Fail instead of auto-building a folder-label manifest when the knowledge manifest is empty or missing.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output = Path(args.output)
    excluded_output = Path(args.excluded_output)
    duplicate_report = Path(args.duplicate_report)
    try:
        input_path = resolve_input(args.input, args.knowledge_root)
    except FileNotFoundError:
        if args.no_folder_fallback:
            raise
        input_path = None
    train_df, excluded_df, source_mode = convert_with_folder_fallback(
        input_path,
        data_root,
        compute_checksum=not args.no_checksum,
        allow_folder_fallback=not args.no_folder_fallback,
    )

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
    print(f"Source mode: {source_mode}")
    print(f"Excluded rows: {len(excluded_df)}")
    if "sha256" in train_df.columns:
        duplicate_count = int(train_df.duplicated("sha256", keep=False).sum())
        print(f"Exact duplicate image rows by sha256: {duplicate_count}")


if __name__ == "__main__":
    main()
