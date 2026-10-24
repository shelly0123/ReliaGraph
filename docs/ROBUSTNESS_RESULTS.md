# ReliaGraph — current robustness and integration results

**Run date:** 2026-10-24 (development container)  
**Purpose:** record what has actually been executed, distinguish smoke tests from real-world evidence, and define the remaining release gates.

## Summary

| Check | Result | Interpretation |
|---|---:|---|
| Automated tests | 100 passed in 3.79 s (final full-suite rerun) | Unit/integration behavior in this environment; 12 non-fatal Python multiprocessing fork deprecation warnings; not a production certification |
| Unified API dashboard upload of 600 synthetic JSONL records | 5.101 s pipeline / 5.168 s API wall time in the final acceptance suite | Includes import parsing, topic discovery, retrieval indexing, graph mutation and snapshot/audit reporting; no PDF parsing or ML inference |
| Dashboard graph after 600-record fixture was uploaded on top of 36 synthetic records and the 20-row Kaggle excerpt | 656 records in workspace, 979 nodes, 1,437 edges, 21 topics | Graph structure/throughput demonstration, not evidence that every extracted edge is correct; a fresh isolated 600-record run measured 920 nodes, 1,356 edges and 16 topics |
| Dataset re-import edit after scale fixture | 5.266 s API wall time | Re-imported a 3-row CSV as 2 rows and removed one stale row while refreshing the larger workspace; within 60 seconds on this host |
| One-document delete after ingest | 4.151 s | Within 60 seconds for this synthetic text run |
| Hash-chain audit | Valid, 602 events | Detects tampering under the stated trusted-head assumptions; not a digital signature |
| MUJ paraphrased source-fact corpus | 22 input records, 3 topics, 28 nodes, 63 edges, 0.201 s | Small development smoke set; source facts are paraphrases and the labelled triples were available during development |
| MUJ exact relation triples | Precision 0.9524, recall 0.9091, F1 0.9302 (20 TP, 1 FP, 2 FN) | Exact normalized triple match on a small development set. Not held-out, not independent, not a claim of general domain accuracy. One miss is a relation-label synonym mismatch; this is not semantically adjudicated by the exact-match metric. |
| Four-domain relation smoke set | Precision/recall/F1 = 0.9375 (15 TP, 1 FP, 1 FN) | Small project-authored set, used during development; optional GLiNER was unavailable and the run used deterministic fallback |
| Planted injection tests | 36/36 detected, 0/20 benign controls flagged | Only for the constructed known-case set; no guarantee against unseen or model-specific injections |
| Hindi queries over English text | NOT MEASURED | Multilingual model weights were unavailable; fallback abstains, so Hit@1 and MRR are null rather than misleadingly calculated |
| Phase 1 vs Phase 2 answer quality | NOT MEASURED | Paired regression script exists, but the full production-like generator/model path and identical reference set must be run before claiming the <=3-point requirement passes |
| Bundled Kaggle-derived excerpt import | PASSED: 20 records in 0.248 s API time | CC BY 4.0 sustainable-energy excerpt from the public CSV mirror; subset only, not full dataset |
| Full Kaggle 500+ record ingestion | NOT YET MEASURED | The dashboard-path 600-record run uses synthetic text. Full Kaggle data files were unavailable; no full-corpus ingestion is claimed. |

## Dashboard acceptance test (latest release run)

`python -m examples.benchmark_dashboard_acceptance` runs the dashboard's actual FastAPI upload/query paths against an isolated temporary data directory. Latest report: `examples/outputs/dashboard_acceptance_report.json`.

- Three original synthetic domain fixtures loaded through the dashboard in 0.144–0.187 seconds per request.
- The bundled 20-row CC BY 4.0 Kaggle-derived energy excerpt loaded in 0.248 seconds through the dashboard sample-data API; the source attribution was included in the returned manifest.
- Uploading 600 synthetic JSONL rows through `/api/documents/upload` took 5.168 seconds end-to-end in the final acceptance run (5.101 seconds measured within the pipeline), below the 60-second target on this host.
- CSV import, shortening/re-import to remove one stale row, ZIP import containing JSONL/TXT, and combined PDF/Markdown/JSON/NDJSON import also passed in the same workspace. These small updates occur against the larger graph, so the API still refreshes an audit/snapshot for the active corpus.
- The optimization is in `core/generalized_audit.py`: safe string-similarity upper bounds are applied before expensive fuzzy duplicate checks. It retained the existing similarity threshold and never merges entities automatically. The 100-test suite passed after this change.
- Factual, thematic and relation-chain query routing returned results. Query engine latencies in this run were about 2.4–17.5 milliseconds; those are local deterministic baseline timings and exclude model generation.
- Authentication setup, blocked unauthenticated dashboard access, CSRF rejection, logout revocation and upload-path checks passed.
- Browser automation itself was unavailable because the execution environment blocked Chromium navigation. JavaScript syntax, static asset serving and the same web API interaction paths were tested; this is not a claim of cross-browser visual QA.

The acceptance run now loads a 20-row, attributed CC BY 4.0 excerpt from the public CSV mirror of the Kaggle sustainable-energy dataset through the actual dashboard API, as well as original synthetic project fixtures. The full original Kaggle files and optional multilingual model weights were not downloaded in this environment. Thus full-corpus Kaggle ingestion and Hindi→English model accuracy are still not claimed.

## Benchmark commands

Run from the repository root in the shared Python environment:

```bash
pytest -q
python -m examples.run_unified_demo
python -m examples.benchmark_unified_scale
python -m examples.benchmark_muj_corpus
python -m examples.run_domain_shift_benchmark --backend gliner
python -m examples.phase2_injection_benchmark
python -m examples.benchmark_hindi_english
```

The output reports are placed in `examples/outputs/`. The `--backend gliner` run attempts an optional adapter, but the current environment did not have its dependencies/weights and therefore fell back. Do not describe its result as GLiNER model performance.

## Important measurement details

### 600-record scale run

Dataset: `examples/data/synthetic_scale_600.jsonl`, an original multi-domain synthetic performance fixture. All optional ML adapters were disabled. The ingest path includes Member 1's registry/chunking/topic discovery and retrieval indexing, plus Member 2's evidence-gated graph contributions, audit and snapshot reporting. It is more representative than a graph-store-only benchmark, but it still does not include actual PDF parsing, cold model startup or real-world relation accuracy.

The edit and delete operations each stayed below 60 seconds on this host. The report-stage/audit work dominates those timings. The requirement must be rerun on the actual hackathon laptop and on the selected corpus, with model-load time reported separately from per-document inference. Do not promise the same timings on different hardware.

### MUJ source-fact corpus

Dataset: `examples/data/muj_paraphrased_corpus.json`. This is a short set of independently phrased summaries linked to public MUJ pages. It is not an official MUJ document, is not endorsed by MUJ, and is not a copy of full webpage passages. The pipeline was run without supplying a MUJ keyword list or MUJ-specific extraction rules. However, the corpus and its gold relation labels were available during development; the benchmark is therefore a development smoke test, not a held-out generalization result.

Exact triple matching is strict: it counts different relation labels as errors even where a human may regard two predicates as related synonyms. The false positives and false negatives in the JSON report are available for inspection.

### Four-domain relation smoke set

The project-authored benchmark contains 20 short examples across university operations, software, ecology and laboratory science, with manually labelled triples. It was used in development and is too small to support generalization claims. The reported result is produced by the deterministic fallback because optional model weights could not be loaded.

### Prompt injection

The benchmark contains 36 deliberately planted attack strings and 20 benign controls. The fallback scanner detects all 36 examples and flags none of the 20 controls. This is evidence only for those patterns. OWASP recommends layered controls across ingestion, retrieval context and output validation; pattern scanning alone cannot guarantee prompt-injection immunity.

### Multilingual retrieval

`examples/data` contains a small authored Hindi-query/English-passage fixture. The optional multilingual E5 and secondary multilingual embedding weights could not be loaded here. The system correctly reports that semantic cross-lingual retrieval is unavailable and abstains; the metrics remain null. Download and test the model on the target laptop before reporting Hit@1, MRR, answer quality or latency.

## Research interpretation

- BEIR shows that retrieval quality varies substantially across domains and tasks; retain a BM25/TF-IDF baseline and report per-domain results instead of claiming one retriever generalizes uniformly: https://arxiv.org/abs/2104.08663
- RAPTOR motivates hierarchical organization for broad corpus-level questions, but ReliaGraph's lightweight topic layer is an analogue, not a reproduction: https://arxiv.org/abs/2401.18059
- GraphRAG and LightRAG already explore graph-oriented retrieval; LightRAG also describes incremental graph updates, so the project must not claim that graph RAG or incremental updates alone are novel: https://arxiv.org/abs/2404.16130 and https://arxiv.org/abs/2410.05779
- The 2024 knowledge-graph construction survey discusses graph quality, provenance, entity resolution and incremental construction as continuing engineering challenges: https://www.mdpi.com/2078-2489/15/8/509
- Multilingual E5 model instructions require `query: ` and `passage: ` prefixes; similarity scores are ranking signals, not calibrated probabilities: https://huggingface.co/intfloat/multilingual-e5-small
- GLiNER-ReLEx is an optional zero-shot entity/relation candidate generator, not a truth oracle: https://huggingface.co/knowledgator/gliner-relex-multi-v1.0
- RAG security controls should be layered across the pipeline: https://cheatsheetseries.owasp.org/cheatsheets/RAG_Security_Cheat_Sheet.html

## Release gates still outstanding

1. Install and run optional models on the actual laptop; record model, revision, device, load time, inference time, memory and errors.
2. Run real 500+ document ingestion on a correctly licensed dataset. For arXiv, the Kaggle metadata is CC0 according to the dataset page, but this does not grant blanket redistribution rights for the linked paper PDFs: https://www.kaggle.com/dsv/6015950 and https://info.arxiv.org/help/bulk_data.html
3. Label relation candidates from at least three genuinely distinct domains. Split by document/source, tune thresholds only on training/validation data, and report held-out per-domain precision, recall, F1, review coverage and abstention quality.
4. Run the Hindi→English benchmark with the real multilingual model and inspect that every result carries the expected evidence chunk and graph path.
5. Run Phase 1 and Phase 2 on the exact same question set, generation settings and references. Report paired answer-quality change in percentage points and confirm whether degradation is no more than three points.
6. Add unseen injection variants and model-facing-context tests; keep benign controls to measure false positives.
7. Re-measure CRUD including actual PDF parsing and enabled per-document model inference on the target environment.

No result in this document guarantees a hackathon outcome or universal accuracy. The intended contribution is the integration of evidence-gated, source-scoped graph updates, preserved conflict claims, explicit uncertainty, query routing and fail-closed multilingual fallback, evaluated with transparent limits.
