# ReliaGraph local ML models: why, how to enable, and how to evaluate

## Design decision

Do not train a new large model from scratch for the hackathon. The prototype uses a deterministic rule-based baseline and optional pretrained local models:

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` suggests semantically similar entity labels. Its model card lists Apache-2.0. Similarity is only a candidate signal, not proof that two entities are the same.
- **NLI model:** `cross-encoder/nli-deberta-v3-small` produces contradiction/entailment/neutral signals for an evidence passage paired with a normalized edge claim. Its model card lists Apache-2.0. It does not prove a claim true or false.

Model-card links:
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- https://huggingface.co/cross-encoder/nli-deberta-v3-small

The exact model revision, dependencies and licences should be checked at the time of any distribution. No model weight files are bundled in this package.

## Why we keep models optional

1. The rules-only pipeline runs without model downloads and gives a reproducible baseline.
2. Model weights download on first use, which can consume time and disk.
3. CPU model inference can be slow on modest laptops. Run one model at a time at first.
4. Model scores depend on domain, phrasing and thresholds; neither model is MUJ-specific or calibrated on MUJ content.
5. Optional model signals should add review suggestions, not silently rewrite or merge graph facts.

## How to enable

```powershell
pip install -r requirements-optional-models.txt
python -m examples.run_muj_demo --use-embeddings
python -m examples.run_muj_demo --use-nli
```

Run both only after measuring each separately:

```powershell
python -m examples.run_muj_demo --use-embeddings --use-nli
```

For first use, download/inference timing is recorded in `muj_model_diagnostics.json` and stage timing in `muj_runtime_report.json`. The embedding vectors are cached in a local SQLite database; NLI scores are cached in memory for the process. Do not commit `data/cache/`.

## Required evaluation before reporting model quality

The included facts are concise paraphrases designed to exercise supported patterns. The current exact-triple metric is a smoke-test on this curated fixture, not an independent generalization test. Never present its perfect or high score as real-world accuracy.

Before claiming a threshold or model is better:

1. Fill `examples/data/relation_review_template.csv` with a manually checked set of at least 30-50 candidate relations from representative documents, including true relationships, unrelated co-occurrences, negation, paraphrases, ambiguous statements and contradictions.
2. Split examples into a tuning set and a held-out test set. Do not tune thresholds on the held-out examples.
3. Annotate each candidate as `supported`, `contradicted`, or `insufficient/unclear`, and have another teammate review disagreements.
4. Report precision, recall, F1 for duplicate/relation findings, and a confusion matrix for NLI classes. Also report false-positive and false-negative examples.
5. Record hardware, cold-start model load time, warm inference time, number of chunks/nodes/edges, cache state and batch size.
6. Compare the rules-only baseline with the optional model. Keep the model in the prototype only if it improves the relevant metrics without unacceptable latency.
7. Use the same corpus/test set for before/after comparison and do not change the reference labels to fit the model output.

## Scaling limits and next steps

The embedding duplicate finder is deliberately bounded to a maximum of 500 non-topic nodes. For larger graphs, use a nearest-neighbour index (for example, an appropriately licensed vector index) rather than all-pairs similarity. Batch inference and SQLite cache help repeated runs, but this prototype is not a production-scale benchmark.

For production, add incremental re-embedding, source-authority/date resolution, reviewed feedback labels, threshold calibration, broader language support, stronger document-level provenance, and access controls. Keep a human-review path for official policies and other consequential claims.


## Optional learned review-priority model (not a relation extractor)

`core/evidence_calibrator.py` contains a deliberately small logistic model using evidence/process signals, including whether the quote is source-anchored, whether methods/sources corroborate, NLI signals when available, and flags for injection/domain shift. It is intended to **prioritize which candidate claims a human reviews first**: a higher score means review sooner. It does not extract new relations and its output is not a truth probability. It cannot bypass provenance checks or change a claim to `human_verified`.

### Collect labels properly

Use `examples/data/evidence_review_template.csv` as the schema. It currently contains only headers; it is not a dataset and no fitted model is distributed. Populate it with human-reviewed candidates from at least three domains, keeping one row per candidate and recording `doc_id`, `domain`, whether the claim is supported (`is_supported`), and each feature as observed by the system. Include both supported and unsupported/unclear candidates. Do not use synthetic corpus assumptions or the extractor's own score as the gold label. Hold out at the document/source level.

### Train with document-group holdout

```powershell
python -m examples.train_evidence_calibrator data/reviewed/evidence_review.csv --output data/reviewed/evidence_calibrator.json
```

The default training command requires at least 100 reviewed rows, at least three domains, both label classes, explicit supported/unsupported label values, valid finite feature values and a document-hash grouped holdout. Malformed labels, NaN/Infinity, invalid ranges or blank source/domain fields fail closed. The score is enabled in the extraction path only if the trained model has at least 60 training rows spanning at least three domains. If these checks fail, the pipeline silently *does not use the triage score* and the ordinary review/abstention policy remains active. Inspect the JSON result and report held-out precision/recall/F1/Brier score before using it operationally.

### Evaluate cross-domain transfer

```powershell
python -m examples.train_evidence_calibrator data/reviewed/evidence_review.csv --leave-one-domain-out --output data/reviewed/domain_transfer_report.json
```

This leaves out one domain at a time. If a domain has too little data or the training split lacks both classes, that fold is marked unevaluated. Do not tune on a held-out domain and then report that same domain as unseen. The calibrator can improve review prioritization only if the human-labelled evaluation establishes that it helps; it is not evidence of universal domain generalization.

### Example integration (Member 2 → Member 3)

```python
import json
from core.robust_extraction import extract_with_fallback

with open("data/reviewed/evidence_calibrator.json", encoding="utf-8") as f:
    reviewed_calibrator = json.load(f)

# `chunks` must use the shared models.Chunk schema or equivalent mappings.
result = extract_with_fallback(
    chunks,
    enable_gliner=True,  # optional; falls back if the adapter cannot load
    calibration_model=reviewed_calibrator,
)
# Pass result.candidate_decisions to build_graph(..., candidate_decisions=...).
# The triage score is attached to edge metadata; do not use it to auto-verify an edge.
```

If model dependencies or weights cannot load, extraction returns to the deterministic fallback. If the calibrator was not trained with enough human-labelled rows/domains, its triage score is omitted and all other evidence gates remain active. The multilingual retrieval module intentionally abstains on Hindi-to-English semantic retrieval when no cross-lingual model is available.
