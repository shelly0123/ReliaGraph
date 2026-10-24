# Member 1 review and unified integration notes

## What Member 1 supplied

The supplied `topicweave_member1(1).zip` contains:

- `core/ingestion.py`: handles file ingestion and text extraction.
- `core/chunking.py`: creates source-linked chunks.
- `core/registry.py`: tracks documents, hashes, chunks and add/edit/delete changes.
- `core/topics.py`: topic discovery using dense embeddings when available, otherwise TF-IDF/LSA and clustering; incremental topic assignment with refit logic.
- `core/retrieval.py`: lexical retrieval and optional dense retrieval adapters.
- `core/question_generator.py`: creates synthetic, source-backed test questions.
- `core/security.py`: Member 1's ingestion security helpers.
- `models.py` and tests: initial data contracts and unit tests.

The implementation is feasible for a hackathon prototype. The TF-IDF/LSA path means topic discovery and lexical retrieval do not require a downloaded model. Optional dense embeddings may help semantic matching but must be tested on the target machine.

## What was changed or added in the unified project

1. **Shared contracts:** `models.py` preserves compatibility with the Member 1 data layouts while including Member 2's source, evidence, status and graph metadata fields.
2. **Single orchestration layer:** `pipeline.py` exposes `TopicWeaveEngine` so Member 3 can call one API for ingestion, graph update, query routing, evaluation cases and metrics.
3. **Source-scoped graph updates:** an edit/delete modifies the claims contributed by that document instead of replacing all claims globally.
4. **Evidence gates and status propagation:** relation candidates are checked against their own source chunk; rejected provenance is not stored, and uncertain relationships remain review-required.
5. **Graph/query integration:** graph paths retain supporting chunk IDs, and topic-membership edges are filtered out of factual relation paths.
6. **Safe language-model failure handling:** Hindi queries over English-only collections abstain when the multilingual model is unavailable instead of relying on shared-token lexical overlap.
7. **General relation fallback improvement:** a small set of broadly applicable English predicate inflections (`has/have/had`, `offer/offers`, `follow/follows`, etc.) and object-phrase boundary handling improve recall for common language patterns. These are language-level patterns, not MUJ-specific subject/entity rules. Every fallback relation remains a candidate requiring review.
8. **Benchmarks and documentation:** the unified scale, MUJ corpus, domain-shift, injection and Hindi-English benchmark scripts now write reports into `examples/outputs/`.

## What the smoke tests show

- The full suite currently passes 99 tests.
- The default unified path is model-free and uses the same pipeline contract as the optional-model path.
- The 600-record synthetic run measures ingestion, topic fit, retrieval indexing, graph update and reporting. It demonstrates local throughput but not relation accuracy.
- On the small MUJ paraphrased development corpus, exact relation triple F1 was 0.9302 after improving generic predicate and noun-phrase handling (previous development run: 0.7027). Since the corpus/labels were used during development, this difference is a development feedback result, not a held-out performance gain.
- The small four-domain authored smoke set has F1 0.9375 with deterministic fallback, also not an independent generalization benchmark.

## Remaining limits

- GLiNER-ReLEx, spaCy parser and multilingual embedding inference could not be fully run in this environment because optional model dependencies/weights were unavailable. The adapters are opt-in and the fallback reports unavailability honestly.
- The 600 synthetic records are not 600 real documents. Actual arXiv or other licensed corpus ingestion has not been completed here.
- Hindi→English Hit@1/MRR and Phase 1-vs-Phase 2 answer quality are not measured. Do not claim those acceptance criteria are met until the actual multilingual model and same-query regression test run successfully.
- The generic fallback is English-cue-limited and prioritizes conservative candidate generation, not universal open information extraction. Do not interpret a confidence score as a truth probability.

## How Member 3 should integrate this

Use the shared API in `docs/INTEGRATION_CONTRACT.md`:

```python
from pipeline import TopicWeaveEngine

engine = TopicWeaveEngine("data/topicweave")
result = engine.upsert_files(files, source_metadata=metadata)
query_result = engine.query(question, top_k=5)
metrics = engine.metrics()
```

Display `result.snapshot`, `result.findings`, `result.topic_coverage`, `result.diff`, `query_result["hits"]`, `query_result["graph_paths"]`, and `query_result["diagnostics"]`. Resolve cited chunks back to `engine.registry.chunk_map()` so the UI opens the actual source text. The evaluator should use the same fixed questions and source references for paired comparisons.
