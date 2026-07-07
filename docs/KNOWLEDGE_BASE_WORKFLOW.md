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
