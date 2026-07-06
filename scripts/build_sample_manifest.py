from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
image_dir = ROOT / "data" / "sample_images"
rows = []
for path in sorted(image_dir.glob("*.jpg")):
    stem = path.stem
    parent = stem.split("_subset_overlap")[0]
    rows.append({
        "image_id": stem,
        "image_path": str(path.relative_to(ROOT / "data")),
        "parent_image_id": parent,
        "field_id": "UNKNOWN_FIELD",
        "capture_session": "UNKNOWN_SESSION",
        "capture_date": "",
        "season": "unknown",
        "variety": "unknown",
        "days_after_sowing": "",
        "bbch_code": "",
        "macro_stage": "UNLABELED",
        "source": "user_sample",
        "license": "user_provided",
        "annotator_1": "",
        "annotator_2": "",
        "adjudicated_label": "",
        "split": "unassigned",
    })
out = ROOT / "data" / "sample_manifest.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Wrote {len(rows)} rows to {out}")
