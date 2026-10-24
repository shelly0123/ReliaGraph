# ReliaGraph integration contract (Members 1, 2 and 3)

## Ownership and boundaries

- Member 1: `CorpusRegistry`, chunking, `TopicModel`, `Retriever`, synthetic question generation.
- Member 2: extraction candidates, incremental SQLite/NetworkX graph, evidence validation, conflicts, snapshot changes, prompt-injection controls.
- Member 3: `pipeline.py` central orchestration, answer generator callback, Streamlit UI, display/scoring integration.

No module should parse uploaded files a second time if a standardized `Chunk` object is already available. The stable contract is `models.py`; new code should construct `Node`, `Edge`, `Chunk`, `Topic`, and `GraphSnapshot` with keywords.

## Stable orchestration interface

`TopicWeaveEngine(data_dir, enable_topic_embeddings=False, enable_dense_retrieval=False, enable_crosslingual=False, enable_gliner=False, enable_spacy=False)`

- `upsert_files(files, source_metadata=None) -> PipelineResult`
  - `files`: paths, `(filename, bytes)` tuples or Streamlit UploadedFile objects accepted by Member 1's registry.
  - Adds/edits documents and returns a snapshot, optional diff, findings, coverage, timings and diagnostics.
- `delete_document(doc_ref) -> PipelineResult`
  - `doc_ref` may be the `doc_id` or original filename.
- `query(query, top_k=5, answerer=None, include_graph_paths=True) -> dict`
  - Uses query routing for thematic, factual and relationship-chain questions.
  - `answerer(question, evidence_context)` is an optional local generator callback.
  - If cross-lingual inference is disabled/unavailable for Hindi over an English-only corpus, the engine abstains rather than returning lexical noise.
- `generate_test_cases(n=40, seed=7, include_hindi=True)`
  - Deterministic reference/evidence test cases; Hindi cases must be evaluated with an available multilingual model.
- `evaluate_retrieval(cases, top_k=5)`
  - Reports Hit@k / MRR against supporting chunk IDs. It does not replace answer-accuracy/faithfulness scoring.
- `metrics()`
  - Counts docs, chunks, topics, nodes/edges, storage, retrieval state and last ingestion timing report.
- `save_snapshot(path=None)` and `close()`.

## Shared model contracts

- `Document`: stable `doc_id`, filename, content hash, source metadata.
- `Chunk`: stable `chunk_id`, `doc_id`, page, text, `topic_ids`, source metadata and quarantined-segment metadata.
- `Topic`: stable topic ID, human-readable label, representative keywords, chunk IDs.
- `Node`: ID, label, type, topic IDs and evidence chunk IDs.
- `Edge`: ID, source/target node IDs, relation, evidence chunk IDs, candidate/review status, optional score and attributes.
- `AuditFinding`: ID, category, severity, impacted IDs, evidence IDs and readable description.
- `GraphSnapshot`: snapshot ID, document hashes, topics, nodes, edges and metadata.
- `TestCase`: test ID, query, reference answer, supporting chunk IDs, answerable flag, kind and language.

## Status semantics

- `candidate`: proposed by extractor, not verified.
- `review_required`: source/quality gate says it requires human verification.
- `corroborated_candidate`: supported by multiple independent evidence items, but still not a truth guarantee.
- `derived`: structural topic-membership edge, not a factual relation.
- `conflict`: a conflict candidate requiring comparison; never silently select one claim.
- `human_verified`: only set after a documented human-review action.

## Invariants

1. Every stored factual edge must retain its source `doc_id`, `chunk_id`, evidence quote and source metadata when available.
2. A quote appearing in the source is provenance anchoring, not proof of factual truth.
3. Edits/deletes are scoped to the source document; an update may change topic assignments, but unchanged factual claim extraction should be reused where possible.
4. Deleting a source removes its live graph claims and retrieval chunks; prior audit events remain metadata-only according to the retention policy.
5. Never use model confidence alone to set `human_verified`.
6. Do not let retrieved document text act as system instructions.
7. If multilingual retrieval is unavailable, fail closed rather than claim Hindi-English retrieval succeeded.
8. Re-run the full tests after changes to `models.py`, extraction, security or retrieval.

## Member 3 dashboard mapping

- Overview KPI cards: `engine.metrics()`.
- Graph: `result.snapshot.nodes`, `result.snapshot.edges`.
- Source inspector: edge `attributes.evidence_sources` and `evidence_chunk_ids`; resolve the chunk through `engine.registry.chunk_map()`.
- Quality panel: `result.findings` and `result.topic_coverage`.
- Change tracking: `result.diff` and `result.previous_snapshot_id`.
- Query pane: `engine.query()` fields `answer`, `hits`, `evidence_context`, `graph_paths`, `diagnostics`.
- Evaluation pane: `engine.generate_test_cases()` + `engine.evaluate_retrieval()`; Member 3 should separately compute answer faithfulness/accuracy using the same labelled cases and actual generator output.
