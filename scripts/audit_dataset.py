import argparse
import hashlib
from pathlib import Path
import pandas as pd

REQUIRED = ["image_id", "image_path", "parent_image_id", "field_id", "macro_stage", "split"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--checksum", action="store_true")
    args = parser.parse_args()
    path = Path(args.manifest)
    data_root = Path(args.data_root)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")
    print("Rows:", len(df))
    print("\nClass counts:\n", df["macro_stage"].value_counts(dropna=False))
    print("\nSplit counts:\n", df["split"].value_counts(dropna=False))
    for col in ["parent_image_id", "field_id", "capture_session"]:
        if col in df.columns:
            leakage = df.groupby(col)["split"].nunique()
            leakage = leakage[leakage > 1]
            print(f"\n{col} leakage groups:", len(leakage))
            if len(leakage):
                print(leakage.head(10))
    invalid = df[df["macro_stage"].isin(["", "UNLABELED"]) | df["macro_stage"].isna()]
    print("\nUnlabeled rows:", len(invalid))

    missing_files = []
    for image_path in df["image_path"]:
        if not (data_root / str(image_path)).exists():
            missing_files.append(str(image_path))
    print("\nMissing image files:", len(missing_files))
    if missing_files:
        print(pd.Series(missing_files).head(10).to_string(index=False))

    non_training_labels = df[df["macro_stage"].isin(["uncertain", "unusable", "S07", "S08"])]
    print("\nNon-training labels in manifest:", len(non_training_labels))

    if args.checksum:
        checksums = []
        for image_path in df["image_path"]:
            full_path = data_root / str(image_path)
            checksums.append(sha256_file(full_path) if full_path.exists() else "")
        df = df.copy()
        df["sha256"] = checksums
        duplicates = df[df["sha256"].ne("") & df.duplicated("sha256", keep=False)]
        print("\nExact duplicate rows by sha256:", len(duplicates))
        if len(duplicates):
            print(duplicates[["image_id", "image_path", "macro_stage", "split", "sha256"]].head(20))


if __name__ == "__main__":
    main()
