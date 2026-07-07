# CROPSTATE Research Package

Image-driven rice growth-stage recognition and confidence-aware stage-gated retrieval.

## Scope

- Input is a rice-field image; no free-form chatbot question is required.
- Vision is six-class image classification: Establishment, Tillering, Stem/Booting, Reproductive, Grain Filling, and Ripening.
- Overlapping patches from one parent image must remain in one train/validation/test split.
- Retrieval runs on fixed agricultural topics and uses the complete calibrated stage-belief distribution.
- Research mode and production mode are separated. Machine-curated chunks are not treated as production-approved recommendations.

## Repository contents

- `src/cropstate/`: vision, confidence, knowledge validation, retrieval, metrics, and statistics.
- `scripts/build_knowledge_base.py`: page-aware chunking from the registered PDF sources.
- `scripts/audit_knowledge_base.py`: schema, coverage, and production-readiness audit.
- `scripts/convert_knowledge_base.py`: canonical conversion from JSONL, XLSX, or CSV.
- `scripts/run_retrieval.py`: fixed-topic hybrid retrieval and stage-aware reranking.
- `scripts/evaluate_retrieval.py`: ungated, hard, fixed-soft, adaptive-soft, and oracle evaluation.
- `configs/knowledge_sources.json`: source scope, page ranges, authority, region, and variety metadata.
- `configs/retrieval.yaml`: retrieval configuration.
- `tests/`: knowledge-loader and retrieval tests.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Test

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Build the knowledge base from PDFs

```bash
PYTHONPATH=src python scripts/build_knowledge_base.py \
  --source-root /path/to/CROPSTATE_KNOWLEDGE_BASE/raw_sources \
  --registry configs/knowledge_sources.json \
  --output-dir /path/to/CROPSTATE_KNOWLEDGE_BASE/chunks
```

Generated files:

```text
rice_knowledge_complete.jsonl
rice_knowledge_nonrestricted.jsonl
knowledge_chunks_complete.csv
review_queue.csv
source_registry_complete.json
chunking_report.json
```

## Audit the corpus

```bash
PYTHONPATH=src python scripts/audit_knowledge_base.py \
  --input /path/to/chunks/rice_knowledge_complete.jsonl \
  --mode research \
  --output results/knowledge_audit.json
```

## Convert an external knowledge folder

The converter accepts canonical JSONL, XLSX, or CSV. It prefers complete canonical files over the old sample workbook.

```bash
PYTHONPATH=src python scripts/convert_knowledge_base.py \
  --knowledge-root /path/to/CROPSTATE_KNOWLEDGE_BASE \
  --mode research \
  --output data/knowledge_chunks.jsonl \
  --report results/knowledge_coverage.json
```

## Run fixed-topic retrieval

```bash
PYTHONPATH=src python scripts/run_retrieval.py \
  --corpus /path/to/chunks/rice_knowledge_complete.jsonl \
  --topic water_management \
  --stage tillering \
  --mode research \
  --top-k 5 \
  --output results/water_tillering.json
```

Supported topics include:

```text
water_management
nutrient_management
pest_risk
disease_risk
weed_management
harvest_readiness
residue_management
climate_adaptation
general_crop_care
```

## Evaluate retrieval baselines

```bash
PYTHONPATH=src python scripts/evaluate_retrieval.py \
  --corpus /path/to/chunks/rice_knowledge_complete.jsonl \
  --scenarios data/retrieval_scenarios.csv \
  --mode research \
  --output results/retrieval_evaluation.json
```

Metrics: P@k, R@k, nDCG@k, and SIRR@k.

## Safety and review status

The generated corpus is structurally validated and ready for research/pilot retrieval. Machine-curated chunks use `review_status=machine_curated_pending_domain_review` and `production_eligible=false` until a domain reviewer approves them.

Production mode loads a chunk only when all conditions hold:

```text
review_status in {reviewed, domain_reviewed, approved}
production_eligible = true
restricted_action = false
```

Commercial, chemical, product-dose, variety-specific, and regulation-sensitive chunks must be reviewed by an agronomy/domain reviewer before use outside research experiments.

## Vision workflow

```bash
PYTHONPATH=src python scripts/build_stage_manifest.py \
  --data-root /path/to/CROPSTATE_DATASET \
  --output data/stage_folder_manifest.csv

PYTHONPATH=src python scripts/audit_dataset.py \
  --manifest data/stage_folder_manifest.csv \
  --data-root /path/to/CROPSTATE_DATASET \
  --checksum

PYTHONPATH=src python scripts/train_vision.py \
  --manifest data/stage_folder_manifest.csv \
  --data-root /path/to/CROPSTATE_DATASET \
  --config configs/vision.yaml \
  --output results/vision_resnet18
```
