# Colab Workflow: GitHub Code + Drive Dataset

Use this notebook flow when the repository is cloned from GitHub and the image dataset lives in Google Drive.

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
RESULTS_ROOT = "/content/drive/MyDrive/CROPSTATE_RESULTS"
```

## 5. Check Dataset Folders

```bash
!ls -lah "{DATA_ROOT}"
```

You should see the six stage folders listed above.

## 6. Build Manifest From Drive Dataset

```bash
!PYTHONPATH=src python scripts/build_stage_manifest.py \
  --data-root "{DATA_ROOT}" \
  --output data/stage_folder_manifest.csv
```

## 7. Audit Manifest

```bash
!PYTHONPATH=src python scripts/audit_dataset.py \
  --manifest data/stage_folder_manifest.csv \
  --data-root "{DATA_ROOT}" \
  --checksum
```

## 8. Train / Fine-Tune

This writes checkpoints and logs directly to Drive.

```bash
!PYTHONPATH=src python scripts/train_vision.py \
  --manifest data/stage_folder_manifest.csv \
  --data-root "{DATA_ROOT}" \
  --config configs/vision.yaml \
  --output "{RESULTS_ROOT}/vision_resnet18_finetune"
```

## 9. Continue Fine-Tuning

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

## 10. Predict One Uploaded Image

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

## 11. Inspect Saved Results

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

GitHub stores code and small metadata. Google Drive stores the dataset, checkpoints, and training outputs.
