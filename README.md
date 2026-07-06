# CROPSTATE Full Research Package

Image-driven rice growth-stage recognition and confidence-aware stage-gated retrieval.

## Scope

- Input is a rice-field image; no free-form chatbot question is required.
- The vision component performs **six-class image classification**, not object detection.
- The six macro-stages are: Establishment, Tillering, Stem/Booting, Reproductive, Grain Filling, Ripening.
- Image patches that overlap or originate from one parent image must stay in the same train/validation/test split.
- Retrieval runs automatically for fixed agricultural topics and uses the full visual stage-belief distribution.

## Package contents

- `paper/`: updated IEEE LaTeX paper and PDF.
- `docs/IMPLEMENTATION_GUIDE_VI.md`: complete Vietnamese implementation guide.
- `docs/ANNOTATION_GUIDELINE.md`: image and knowledge annotation protocol.
- `src/cropstate/`: training, inference, confidence, retrieval, evaluation, and statistics code.
- `configs/`: example experiment configurations.
- `data/templates/`: CSV/JSONL templates.
- `data/sample_images/`: six user-provided sample patches for pipeline testing only.
- `scripts/`: command-line entry points.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/build_sample_manifest.py
python scripts/audit_dataset.py --manifest data/sample_manifest.csv
```

The six bundled images are **not sufficient to train a scientific six-stage model**. They are included only to test file loading, metadata validation, and leakage checks.

## Manifest and knowledge-base conversion

If the Google Sheet `Image_Manifest_Template` has been exported as CSV, convert it before training:

```bash
python scripts/convert_image_manifest.py \
  --input KNOWLEDGE_BASE_SAMPLE/Image_Manifest_Template.csv \
  --data-root data \
  --output data/image_manifest.csv

python scripts/audit_dataset.py \
  --manifest data/image_manifest.csv \
  --data-root data \
  --checksum
```

The converter keeps only usable S01-S06 samples for six-class training. S07 Uncertain and S08 Unusable rows are written to `data/image_manifest_excluded.csv`.

For a folder-only pilot manifest generated directly from the six stage folders:

```bash
python scripts/build_stage_manifest.py \
  --data-root data \
  --output data/stage_folder_manifest.csv

python scripts/audit_dataset.py \
  --manifest data/stage_folder_manifest.csv \
  --data-root data \
  --checksum
```

If the Google Sheet `Knowledge_Chunks` has been exported as CSV, convert it for retrieval experiments:

```bash
python scripts/convert_knowledge_base.py \
  --input KNOWLEDGE_BASE_SAMPLE/Knowledge_Chunks.csv \
  --output data/knowledge_chunks.jsonl
```

## Main research sequence

1. Collect and annotate images for all six stages.
2. Create parent-image, field, date, and season metadata.
3. Split data by group, never randomly by overlapping patch.
4. Train image-classification baselines.
5. Calibrate probabilities and export stage-belief vectors.
6. Build stage-annotated agricultural knowledge chunks.
7. Run oracle-stage and end-to-end retrieval experiments.
8. Calculate Macro-F1, MASD, ECE, nDCG, SIRR, degradation, ablation, and paired statistics.
9. Populate paper result tables only from saved experiment outputs.
