"""Manual BBCH stage labeller for the rice image dataset.

The stage folders were found to be unreliable, so this tool lets a human assign
the correct CROPSTATE macro-stage to each image using the BBCH scale as the rubric
(see the on-screen cheatsheet). It labels BY PARENT: overlapping crops of one scene
(``p<NNN>_subset_overlap_*``) share a scene, so you judge the scene once and the
label is applied to all its crops — ~150 decisions instead of ~550.

Keys (matplotlib window):
    1..6  assign stage   0/space  skip (unsure)   u  undo last   q  save & quit
    Stages: 1 establishment  2 tillering  3 stem_booting  4 reproductive
            5 grain_filling   6 ripening

Resumable: progress is autosaved to --output after every decision; re-running skips
already-labelled groups. Output is a manifest compatible with scripts/train_vision.py
(image_path, macro_stage, parent_image_id, field_id, split=unassigned), so the repo's
leak-free grouping + balanced sampler apply unchanged.

Deps: pandas, Pillow, matplotlib (GUI). The grouping / IO core is GUI-free and unit-tested.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

STAGE_NAMES = [
    "establishment", "tillering", "stem_booting",
    "reproductive", "grain_filling", "ripening",
]
KEY_TO_STAGE = {str(i + 1): s for i, s in enumerate(STAGE_NAMES)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# BBCH cheatsheet shown beside each image.
RUBRIC = [
    "1 establishment  (BBCH 00-19): ma, it la, lo nuoc/dat",
    "2 tillering      (20-29): de nhanh, tan phu dan, TOAN XANH",
    "3 stem_booting   (30-49): lam dong, tan kin, CHUA tro bong",
    "4 reproductive   (50-69): TRO bong/hoa, hat xanh",
    "5 grain_filling  (70-79): hat may, nga vang",
    "6 ripening       (80-92): hat cung, VANG/NAU, tan ua",
]


def parent_stem(path: Path) -> str:
    return path.stem.split("_subset_overlap", 1)[0]


def enumerate_groups(root: Path) -> list[dict]:
    """Group every image under ``root`` by (containing folder, parent stem).

    The folder is part of the key (not the label): per-folder ``p<NNN>`` numbering
    means the same stem in two folders is a different scene, so they stay separate.
    """
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    groups: dict[str, list[Path]] = {}
    for img in images:
        rel_dir = img.parent.relative_to(root).as_posix()
        key = f"{rel_dir}/{parent_stem(img)}" if rel_dir != "." else parent_stem(img)
        groups.setdefault(key, []).append(img)
    return [{"group": k, "images": v} for k, v in sorted(groups.items())]


def load_progress(output: Path) -> dict[str, str]:
    """Return {group_key: stage} already labelled, for resume."""
    if not output.exists():
        return {}
    df = pd.read_csv(output)
    if "parent_image_id" not in df.columns or "macro_stage" not in df.columns:
        return {}
    return dict(zip(df["parent_image_id"].astype(str), df["macro_stage"].astype(str)))


def build_rows(groups: list[dict], labels: dict[str, str], root: Path) -> list[dict]:
    rows = []
    for g in groups:
        stage = labels.get(g["group"])
        if stage is None:
            continue
        for img in g["images"]:
            rows.append({
                "image_id": img.stem,
                "image_path": img.relative_to(root).as_posix(),
                "macro_stage": stage,
                "parent_image_id": g["group"],
                "field_id": f"label:{g['group']}",
                "capture_session": img.parent.relative_to(root).as_posix(),
                "source": "manual_bbch",
                "split": "unassigned",
            })
    return rows


def save(groups: list[dict], labels: dict[str, str], root: Path, output: Path) -> int:
    rows = build_rows(groups, labels, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return len(rows)


def run_gui(groups: list[dict], labels: dict[str, str], root: Path, output: Path, show_crops: int) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    pending = [i for i, g in enumerate(groups) if g["group"] not in labels]
    if not pending:
        print("Tat ca group da co nhan. Khong con gi de gan.")
        return
    state = {"pos": 0, "history": []}
    fig = plt.figure(figsize=(12, 7))
    fig.subplots_adjust(left=0.02, right=0.78, top=0.92, bottom=0.05, wspace=0.05, hspace=0.15)

    def render():
        fig.clear()
        idx = pending[state["pos"]]
        g = groups[idx]
        imgs = g["images"][:show_crops]
        cols = min(len(imgs), 3) or 1
        rows_n = (len(imgs) + cols - 1) // cols
        for j, ip in enumerate(imgs):
            ax = fig.add_subplot(rows_n, cols, j + 1)
            try:
                ax.imshow(Image.open(ip).convert("RGB"))
            except Exception as exc:  # noqa: BLE001
                ax.text(0.5, 0.5, f"loi doc anh\n{exc}", ha="center")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(ip.name, fontsize=7)
        done = len(labels)
        fig.suptitle(f"[{done}/{len(groups)} da gan]  group: {g['group']}   "
                     f"({len(g['images'])} crop)", fontsize=11)
        fig.text(0.80, 0.90, "BBCH rubric", fontsize=11, weight="bold", va="top")
        fig.text(0.80, 0.85, "\n\n".join(RUBRIC), fontsize=8.5, va="top", family="monospace")
        fig.text(0.80, 0.16, "1-6 gan | 0/space skip\nu undo | q luu&thoat",
                 fontsize=9, va="top", color="tab:blue")
        fig.canvas.draw_idle()

    def advance():
        state["pos"] += 1
        if state["pos"] >= len(pending):
            n = save(groups, labels, root, output)
            print(f"Xong toan bo. Da luu {n} dong -> {output}")
            plt.close(fig)
        else:
            render()

    def on_key(event):
        idx = pending[state["pos"]]
        key = (event.key or "").lower()
        if key in KEY_TO_STAGE:
            labels[groups[idx]["group"]] = KEY_TO_STAGE[key]
            state["history"].append(groups[idx]["group"])
            save(groups, labels, root, output)
            advance()
        elif key in ("0", " ", "space"):
            advance()
        elif key == "u" and state["history"]:
            last = state["history"].pop()
            labels.pop(last, None)
            save(groups, labels, root, output)
            state["pos"] = max(0, state["pos"] - 1)
            render()
        elif key == "q":
            n = save(groups, labels, root, output)
            print(f"Luu & thoat. {n} dong -> {output}")
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    render()
    plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="Folder of images (may contain 0N_stage subfolders).")
    p.add_argument("--output", default="data/manual_stage_manifest.csv")
    p.add_argument("--show-crops", type=int, default=6, help="Max crops of a parent to display at once.")
    p.add_argument("--list-only", action="store_true", help="Print group/progress summary and exit (no GUI).")
    args = p.parse_args()

    root = Path(args.data_root)
    output = Path(args.output)
    groups = enumerate_groups(root)
    if not groups:
        raise SystemExit(f"No images found under {root}")
    labels = load_progress(output)

    n_imgs = sum(len(g["images"]) for g in groups)
    print(f"{len(groups)} group (scene) / {n_imgs} anh | da gan: {len(labels)} | con lai: {len(groups) - len(labels)}")
    if args.list_only:
        for g in groups[:20]:
            mark = labels.get(g["group"], "-")
            print(f"  [{mark:>13}] {g['group']}  ({len(g['images'])} crop)")
        return

    run_gui(groups, labels, root, output, args.show_crops)


if __name__ == "__main__":
    main()
