#!/usr/bin/env bash
# Retrain experiments on ONE fixed split (CROPSTATE_RESULTS/vision_final/manifest.csv)
# so every run is directly comparable to the CE baseline (vision_final).
#   A#2 ordinal loss, C#6 focal loss, C#9 fixed-split multi-seed.
set -u
cd "$(dirname "$0")/../.."
export CROPSTATE_FORCE_CPU=1 PYTHONPATH=src OMP_NUM_THREADS=4
MANIFEST=CROPSTATE_RESULTS/vision_final/manifest.csv
DATA=CROPSTATE_DATASET
EPOCHS=${EPOCHS:-30}
PY=.venv/bin/python

run() {  # name seed loss extra...
  local name=$1 seed=$2 loss=$3; shift 3
  echo "=== [$(date +%H:%M:%S)] $name (seed=$seed loss=$loss) ==="
  $PY scripts/train_vision.py \
    --manifest "$MANIFEST" --data-root "$DATA" --config configs/vision.yaml \
    --model resnet18 --seed "$seed" --loss "$loss" --epochs "$EPOCHS" \
    --output "CROPSTATE_RESULTS/novelty/$name" "$@" \
    && echo "--- $name done ---" || echo "!!! $name FAILED !!!"
}

run resnet18_ordinal   42  ordinal
run resnet18_focal     42  focal
run resnet18_ce_seed7   7  ce
run resnet18_ce_seed123 123 ce
echo "=== ALL RETRAINS COMPLETE $(date +%H:%M:%S) ==="
