# Colab Workflow: GitHub Code + Drive Dataset

Use this notebook flow when the repository is cloned from GitHub while the image dataset and knowledge base live in Google Drive.

You can also import `notebooks/cropstate_colab_github_drive.ipynb` directly into Google Colab.

Expected Drive layout:

```text
MyDrive/CROPSTATE_DATASET/
  01_establishment/
  02_tillering/
  03_stem_booting/
  04_reproductive/
  05_grain_filling/
  06_ripening/
```

Expected knowledge-base layout:

```text
MyDrive/CROPSTATE_KNOWLEDGE_BASE/
  CROPSTATE_Knowledge_Base_Complete.xlsx
```

The knowledge-base folder should include:

```text
MyDrive/CROPSTATE_KNOWLEDGE_BASE/
  chunks/rice_knowledge_complete.jsonl
  chunks/rice_knowledge_nonrestricted.jsonl
  knowledge_chunks_complete.csv
  review_queue.csv
```

Results are written to:

```text
MyDrive/CROPSTATE_RESULTS/
```

## 1. Mount Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 2. Clone Or Update Repo

Replace `YOUR_USERNAME/YOUR_REPO` with the actual GitHub repo.

```bash
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /content/CROPSTATE_Full_Research_Package
```

If the repo already exists:

```bash
%cd /content/CROPSTATE_Full_Research_Package
!git pull
```

## 3. Install Dependencies

```bash
%cd /content/CROPSTATE_Full_Research_Package
!pip install -r requirements.txt
!pip install -e .
```

## 4. Set Paths

```python
DATA_ROOT = "/content/drive/MyDrive/CROPSTATE_DATASET"
KNOWLEDGE_ROOT = "/content/drive/MyDrive/CROPSTATE_KNOWLEDGE_BASE"
RESULTS_ROOT = "/content/drive/MyDrive/CROPSTATE_RESULTS"
```

## 5. Check Dataset Folders

```bash
!ls -lah "{DATA_ROOT}"
```

You should see the six stage folders listed above.

## 6. Check Knowledge Base Folder

```bash
!ls -lah "{KNOWLEDGE_ROOT}"
```

You should see `CROPSTATE_Knowledge_Base_Complete.xlsx` and the `chunks/` folder.

## 7. Build Manifest From Drive Dataset

```bash
!PYTHONPATH=src python scripts/build_stage_manifest.py \
  --data-root "{DATA_ROOT}" \
  --output data/stage_folder_manifest.csv
```

## 8. Convert Image Manifest From Knowledge Base

Run this after `Image_Manifest_Template` has been filled. If the sheet is still a placeholder, `data/image_manifest.csv` will be empty and excluded rows will be written to `data/image_manifest_excluded.csv`.

```bash
!PYTHONPATH=src python scripts/convert_image_manifest.py \
  --knowledge-root "{KNOWLEDGE_ROOT}" \
  --data-root "{DATA_ROOT}" \
  --output data/image_manifest.csv
```

Use `data/image_manifest.csv` for training only after it contains reviewed rows. Until then, use `data/stage_folder_manifest.csv` for pilot runs.

## 9. Convert Knowledge Chunks From Drive

```bash
!PYTHONPATH=src python scripts/convert_knowledge_base.py \
  --knowledge-root "{KNOWLEDGE_ROOT}" \
  --output data/knowledge_chunks.jsonl
```


## 10. Audit Knowledge Corpus

```bash
!mkdir -p "{RESULTS_ROOT}/retrieval"
!PYTHONPATH=src python scripts/audit_knowledge_base.py \
  --input "{KNOWLEDGE_ROOT}/chunks/rice_knowledge_complete.jsonl" \
  --mode research \
  --output "{RESULTS_ROOT}/retrieval/knowledge_audit_complete.json"

!PYTHONPATH=src python scripts/audit_knowledge_base.py \
  --input "{KNOWLEDGE_ROOT}/chunks/rice_knowledge_nonrestricted.jsonl" \
  --mode research \
  --output "{RESULTS_ROOT}/retrieval/knowledge_audit_nonrestricted.json"
```

## 11. Run Sample Retrieval

```bash
!PYTHONPATH=src python scripts/run_retrieval.py \
  --corpus "{KNOWLEDGE_ROOT}/chunks/rice_knowledge_nonrestricted.jsonl" \
  --topic water_management \
  --stage tillering \
  --mode research \
  --top-k 5 \
  --output "{RESULTS_ROOT}/retrieval/water_tillering.json"
```

## 12. Evaluate Retrieval Baselines

Run this only after `data/retrieval_scenarios.csv` exists.

```bash
!PYTHONPATH=src python scripts/evaluate_retrieval.py \
  --corpus "{KNOWLEDGE_ROOT}/chunks/rice_knowledge_nonrestricted.jsonl" \
  --scenarios data/retrieval_scenarios.csv \
  --mode research \
  --output "{RESULTS_ROOT}/retrieval/retrieval_evaluation.json"
```

## 13. Audit Manifest

For the folder-generated pilot manifest:

```bash
!PYTHONPATH=src python scripts/audit_dataset.py \
  --manifest data/stage_folder_manifest.csv \
  --data-root "{DATA_ROOT}" \
  --checksum
```

For the reviewed sheet manifest:

```bash
!PYTHONPATH=src python scripts/audit_dataset.py \
  --manifest data/image_manifest.csv \
  --data-root "{DATA_ROOT}" \
  --checksum
```

## 14. Train / Fine-Tune

This writes checkpoints and logs directly to Drive.

```bash
!PYTHONPATH=src python scripts/train_vision.py \
  --manifest data/stage_folder_manifest.csv \
  --data-root "{DATA_ROOT}" \
  --config configs/vision.yaml \
  --output "{RESULTS_ROOT}/vision_resnet18_finetune"
```

To train with the reviewed sheet manifest instead, replace `data/stage_folder_manifest.csv` with `data/image_manifest.csv`.

## 15. Continue Fine-Tuning

Use this for round 2 or later. Keep the same manifest so validation/test splits remain comparable.

```bash
!PYTHONPATH=src python scripts/train_vision.py \
  --manifest data/stage_folder_manifest.csv \
  --data-root "{DATA_ROOT}" \
  --config configs/vision.yaml \
  --resume-checkpoint "{RESULTS_ROOT}/vision_resnet18_finetune/best_checkpoint.pt" \
  --freeze-backbone-epochs 0 \
  --learning-rate 0.0001 \
  --backbone-learning-rate 0.00001 \
  --output "{RESULTS_ROOT}/vision_resnet18_finetune_round2"
```

## 16. Predict One Uploaded Image

```python
from google.colab import files

uploaded = files.upload()
image_path = next(iter(uploaded))
print("Uploaded:", image_path)
```

```bash
!PYTHONPATH=src python scripts/predict_image.py \
  --checkpoint "{RESULTS_ROOT}/vision_resnet18_finetune/best_checkpoint.pt" \
  --image "{image_path}"
```

## 17. Inspect Saved Results

```bash
!ls -lah "{RESULTS_ROOT}/vision_resnet18_finetune"
```

Expected files:

```text
best_checkpoint.pt
best_model.pt
history.json
class_counts.json
```

GitHub stores code and small metadata. Google Drive stores the dataset, knowledge base, checkpoints, and training outputs.
