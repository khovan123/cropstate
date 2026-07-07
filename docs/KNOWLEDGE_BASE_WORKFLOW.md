# CROPSTATE Knowledge Base Workflow

## 1. Drive layout

```text
CROPSTATE_KNOWLEDGE_BASE/
  raw_sources/
  processed_documents/
  chunks/
  CROPSTATE_Knowledge_Base_Complete.xlsx
```

## 2. Build canonical chunks

```bash
PYTHONPATH=src python scripts/build_knowledge_base.py \
  --source-root /content/drive/MyDrive/CROPSTATE_KNOWLEDGE_BASE/raw_sources \
  --registry configs/knowledge_sources.json \
  --output-dir /content/drive/MyDrive/CROPSTATE_KNOWLEDGE_BASE/chunks
```

The builder performs:

1. page-aware PDF text extraction;
2. removal of front matter, tables of contents, figure captions, and irrelevant sections;
3. semantic chunking into complete passages;
4. topic assignment;
5. six-stage compatibility assignment;
6. source, region, variety, page, and authority metadata attachment;
7. restricted-action flagging;
8. duplicate removal;
9. research and non-restricted JSONL export;
10. coverage and review-queue generation.

## 3. Generated files

```text
rice_knowledge_complete.jsonl
rice_knowledge_nonrestricted.jsonl
knowledge_chunks_complete.csv
review_queue.csv
source_registry_complete.json
chunking_report.json
```

## 4. Audit

```bash
PYTHONPATH=src python scripts/audit_knowledge_base.py \
  --input /content/drive/MyDrive/CROPSTATE_KNOWLEDGE_BASE/chunks/rice_knowledge_complete.jsonl \
  --mode research \
  --output results/knowledge_audit.json
```

## 5. Domain review

Review every row in `Review_Queue` and record:

- approval decision;
- reviewer;
- corrected topic;
- corrected stage compatibility;
- region and variety limits;
- source-page confirmation;
- whether the content is safe for production use.

A chunk may be used in production mode only when:

```text
review_status = reviewed/domain_reviewed/approved
production_eligible = true
restricted_action = false
```

## 6. Convert or normalize an external file

```bash
PYTHONPATH=src python scripts/convert_knowledge_base.py \
  --knowledge-root /content/drive/MyDrive/CROPSTATE_KNOWLEDGE_BASE \
  --mode research \
  --output data/knowledge_chunks.jsonl \
  --report results/knowledge_coverage.json
```

## 7. Run retrieval

```bash
PYTHONPATH=src python scripts/run_retrieval.py \
  --corpus /content/drive/MyDrive/CROPSTATE_KNOWLEDGE_BASE/chunks/rice_knowledge_complete.jsonl \
  --topic water_management \
  --stage tillering \
  --mode research \
  --top-k 5 \
  --output results/sample_retrieval.json
```

## 8. Evaluate baselines

```bash
PYTHONPATH=src python scripts/evaluate_retrieval.py \
  --corpus /content/drive/MyDrive/CROPSTATE_KNOWLEDGE_BASE/chunks/rice_knowledge_complete.jsonl \
  --scenarios data/retrieval_scenarios.csv \
  --mode research \
  --output results/retrieval_evaluation.json
```

The evaluator reports P@k, R@k, nDCG@k, and SIRR@k for ungated, hard top-1, fixed-soft, adaptive-soft, and oracle-stage methods.

## 9. Safety boundary

Machine-curated records are suitable for research and internal pilot retrieval. They are not automatically approved as real-world agronomic recommendations. Production mode intentionally blocks records that have not passed domain review.

## 10. Stage-first output (IRRI Rice Knowledge Bank + stage profiles)

Every chunk (Vietnamese PDFs and the IRRI corpus below) now also carries a `facet`: `fertilizer`,
`conditions`, `pest_disease_prevention`, `next_stage_action`, or `general`. This is additive —
`topic` and `stage_compatibility` are unchanged, so `retrieval.py` and `evaluate_retrieval.py`
(and the P@k/R@k/nDCG@k/SIRR@k numbers already reported for the 302-chunk Vietnamese corpus in
`paper/cropstate_image_paper.tex`) are unaffected.

### 10.1 Crawl the IRRI Rice Knowledge Bank

```bash
python scripts/crawl_irri_rkb.py \
  --registry configs/knowledge_sources_irri.json \
  --output-dir CROPSTATE_KNOWLEDGE_BASE/raw_sources_irri
```

Caches the 14 registered `knowledgebank.irri.org/step-by-step-production/...` pages (English,
CC BY-NC-SA 3.0) as raw HTML. Re-run with `--force` to refresh.

### 10.2 Chunk it (facet-aware)

```bash
PYTHONPATH=src python scripts/build_knowledge_base.py \
  --output-dir CROPSTATE_KNOWLEDGE_BASE/chunks \
  --web-pages-dir CROPSTATE_KNOWLEDGE_BASE/raw_sources_irri \
  --web-registry configs/knowledge_sources_irri.json
```

Omitting `--source-root` skips the PDF rebuild entirely, so this only (re)writes the IRRI
outputs — `rice_knowledge_irri_en.jsonl`, `rice_knowledge_irri_en_nonrestricted.jsonl`,
`chunking_report_irri.json`, `knowledge_chunks_irri_en.csv`, `review_queue_irri.csv`,
`source_registry_irri.json` — and **never touches** `rice_knowledge_complete.jsonl`. Pass
`--source-root <pdf dir>` too only when you also want to rebuild the Vietnamese corpus from a
complete local copy of the registered PDFs (all 7 files in `configs/knowledge_sources.json`);
rebuilding it from a partial PDF set would change the paper's reported chunk counts.

### 10.3 Roll up into a stage-first report

```bash
PYTHONPATH=src python scripts/build_stage_profiles.py \
  --corpus CROPSTATE_KNOWLEDGE_BASE/chunks/rice_knowledge_complete.jsonl \
  --corpus CROPSTATE_KNOWLEDGE_BASE/chunks/rice_knowledge_irri_en.jsonl \
  --mode all \
  --output CROPSTATE_KNOWLEDGE_BASE/chunks/stage_profiles.json
```

Groups chunks from one or more corpora by their dominant growth stage (from
`stage_compatibility`) and by `facet`. Each of the 6 stages gets `fertilizer`, `conditions`,
`pest_disease_prevention`, and `next_stage_actions` (evidence tagged `next_stage_action`, plus a
look-ahead preview pulled from the following stage's chunks so the bucket isn't empty just
because sources rarely phrase things as "before the next stage"). The output also lists
`coverage_warnings` for any stage/facet combination with no evidence — current sources skew
toward `establishment` and `ripening`, so the middle stages (tillering, stem/booting,
reproductive, grain filling) are thin and will show up there until more sources are added.
