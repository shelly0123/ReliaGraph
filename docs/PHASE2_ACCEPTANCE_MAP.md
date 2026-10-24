# ReliaGraph Phase 2 acceptance map

This table maps the organizer update to code and current verification. Read `ROBUSTNESS_RESULTS.md` for environment and benchmark caveats. Do not present synthetic smoke numbers as model generalization or industry benchmarks.

| Requirement | Implementation | Current verification | Status |
|---|---|---|---|
| Keep knowledge base trustworthy as documents change, conflict or mislead | `core/phase2/incremental_kb.py`, `core/phase2/robust_service.py`, `core/phase2/conflicts.py`, `core/phase2/injection_guard.py` | 99 tests pass; edits/deletes are source-scoped, conflict candidates keep source references, injection inputs are quarantined where matched | Implemented with known limitations; source-specific correctness needs evaluation |
| Broad themes and relation chains beyond flat vector search | `core/query_router.py`, `core/topics.py`, `core/phase2/crosslingual_retrieval.py`, `pipeline.py` | `examples/run_unified_demo.py` verifies thematic, factual and relation-chain routes; paths exclude topic-membership edges as factual predicates | Prototype path works in the deterministic English fallback; broad query quality needs a labelled corpus |
| Ingest 500+ documents and report time/graph size | `examples/benchmark_unified_scale.py`, `examples/import_kaggle_arxiv.py`, `examples/benchmark_arxiv_500.py` | 600 synthetic text records via dashboard API: 5.142 s pipeline / 5.207 s API time; 600 rows plus other fixtures created 979 nodes, 1,437 edges and 21 topics; a separate 20-row CC BY 4.0 Kaggle-derived excerpt imported in 0.215 s; full Kaggle data not present | Synthetic throughput verified; real corpus not yet measured |
| Add/edit/delete without full rebuild; complete update within 60 s | `TopicWeaveEngine.upsert_files`, `.delete_document`, `core/phase2/incremental_kb.py` | Synthetic run: one-document edit 4.372 s, delete 4.151 s; model inference/PDF parsing excluded | Meets on current synthetic fallback run; rerun on target laptop and actual inputs |
| When two sources contradict, show both claims with sources | `find_competing_claims`, `core/phase2/conflicts.py`, `core/phase2/incremental_kb.py` | Unit tests cover competing claims and source references | Implemented; source authority still requires human/domain policy |
| Ignore 20+ planted document injections | `core/phase2/injection_guard.py`; `examples/data/phase2_prompt_injection_cases.json`; `examples/phase2_injection_benchmark.py` | 36/36 crafted attacks detected, 0/20 crafted benign controls flagged; matched spans absent from model context in this test | Known-case test passes; not immunity to unseen attacks |
| Hindi questions over English docs, with supporting graph path | `core/phase2/crosslingual_retrieval.py`, `pipeline.py`, `examples/benchmark_hindi_english.py` | Current environment could not load multilingual E5 or secondary model; engine abstains rather than fake semantic match; Hit@1/MRR are null | NOT YET MEASURED; model weights and labelled test required |
| Phase 2 answer quality stays within 3 points of Phase 1 | `examples/check_phase1_regression.py`, `examples/data/phase1_phase2_regression_template.jsonl` | Script/template exist; same full answer-generation path and labelled reference set not yet run | NOT YET MEASURED |
| Source-grounded synthetic question generation, relevance/faithfulness/accuracy metrics, root-cause report | `core/question_generator.py`, `core/phase2/phase2_evaluation.py`, `pipeline.py`, `examples/` | Retrieval metrics and deterministic test generation are integrated; RAG answer evaluation depends on Member 3's actual answerer and a labelled set | Interfaces ready; end-to-end generator evaluation not yet certified |
| Use cross-domain extraction, not hand-coded MUJ vocabulary | `core/robust_extraction.py`, `core/domain_agnostic_extraction.py`, optional `GLiNER-ReLEx` adapter | Four-domain smoke set F1 0.9375 and MUJ development smoke F1 0.9302 with generic fallback. Both small and used during development; GLiNER weights unavailable | Useful regression checks, not independent generalization evidence |
| CIA controls | `core/cia_controls.py`, local SQLite storage, source hashes, injection guard and scoped deletion | Local-first policy blocks remote API use by default; tamper-evident hash-chain tests pass | Prototype safeguards; not a complete enterprise security control set |

## Commands to reproduce

```bash
pytest -q
python -m examples.run_unified_demo
python -m examples.benchmark_unified_scale
python -m examples.benchmark_muj_corpus
python -m examples.run_domain_shift_benchmark --backend gliner
python -m examples.phase2_injection_benchmark
python -m examples.benchmark_hindi_english
```

## Minimum acceptance protocol before final presentation

1. Run these commands on the target laptop and preserve the outputs in `examples/outputs/`.
2. Download/use the licensed metadata corpus locally for a real 500+ document throughput run; keep the full arXiv PDFs out of the distributed package unless rights are checked individually.
3. Validate on an independently labelled, document-held-out dataset with at least three domains. Report per-domain precision/recall/F1 alongside how many candidates were accepted, sent for review, or abstained.
4. Enable the actual cross-lingual model and measure Hindi→English Hit@1/MRR, response latency and graph-path evidence.
5. Run paired Phase 1/Phase 2 answer evaluation and report quality delta in percentage points.
6. Test unseen prompt-injection patterns, multilingual injection variants, false positives on benign text and model-facing contexts.
7. Measure document add/edit/delete with actual PDFs and the final selected inference models. The synthetic timings are not a performance promise for the final system.
