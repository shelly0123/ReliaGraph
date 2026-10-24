# ReliaGraph — Prototype acceptance report

**Run date:** 24 October 2026  
**Build:** unified Member 1 + Member 2 pipeline with the local browser dashboard  
**Mode measured:** deterministic baseline; optional model weights disabled  
**Scope:** engineering and interface acceptance, not a claim of general accuracy superiority

## What is included

- First-run account creation, sign-in, sign-out, server-side sessions and password hashing.
- Local browser dashboard with Overview, Sources & datasets, Interactive Knowledge Graph, Ask & trace, Evidence & trust, Evaluation & benchmarks, and Security & privacy sections.
- PDF/TXT/Markdown/CSV/JSON/JSONL/NDJSON/ZIP upload adapter; ZIP members are handled in memory under extension, size and compression-ratio limits.
- Interactive node/edge graph with search, type filters, pan/zoom, fit/reset, and a node inspector that exposes links to source evidence.
- Thematic/factual/relation-chain query routing; evidence/context, optional graph paths and explicit retrieval diagnostics.
- Per-source add/edit/delete, shorter-export stale-row removal, audit events, competing-claim review, injection scanning/quarantine, graph sizes and latency reports.
- Dataset catalogue with three ready-to-load original synthetic fixtures, one ready-to-load 20-row CC BY 4.0 excerpt from a public mirror of Kaggle sustainable-energy data, and links to the full external Kaggle datasets.
- Exportable JSON evaluation report and benchmark history visible through the dashboard.

## Recorded acceptance results

| Check | Measured result | Scope and caveat |
|---|---:|---|
| Automated tests | 100 passed in 3.79 s (final full-suite rerun) | Unit and API integration suite; 12 non-fatal Python fork warnings |
| Three synthetic domain fixtures | 12 records each; 0.144–0.187 s per API request | Cyber, biomedical and energy; original synthetic text, not Kaggle data |
| Bundled real-data excerpt | 20 records; 0.248 s API request | Selected rows/columns from the public mirror of Kaggle sustainable-energy data; CC BY 4.0; not the full data file |
| 600-record dashboard upload | 5.101 s pipeline / 5.168 s end-to-end API in the final acceptance run | Synthetic JSONL performance fixture; optional model inference and PDF parsing disabled |
| Graph after demo fixtures + 20-row excerpt + 600 rows | 656 records, 979 nodes, 1,437 edges, 21 topics | Structural throughput/counts, not extraction precision |
| Fresh isolated 600-row upload | 4.84 s in separate repeat run | 600 chunks, 920 nodes, 1,356 edges, 16 topics |
| Update smaller dataset after scale fixture | 5.338 s API request; one stale CSV row removed | Includes snapshot/audit against the larger active graph |
| ZIP with JSONL + TXT | 3.381 s API request | Three records created in the latest run |
| PDF + Markdown + JSON + NDJSON | 3.553 s API request | One PDF, one Markdown document, two JSON records and one NDJSON record all ingested successfully |
| Thematic query | 17.5 ms engine time; 8 hits, 8 topic summaries, 5 graph paths | Deterministic baseline; this is not LLM answer quality |
| Factual query | 2.4 ms engine time; 5 hits | Local deterministic retrieval only |
| Relationship-chain query | 7.1 ms engine time; 5 hits, 4 graph paths | Local deterministic graph/path retrieval |
| In-corpus retrieval smoke test | Hit@5 = 0.75, MRR = 0.75 over 8 answerable cases | Small, generated from the same workspace; not held-out evaluation and not cross-domain accuracy |
| Authentication/CSRF/logout | Passed | API tests; local single-user prototype security profile |
| JavaScript source check | Passed | `node --check webapp/static/app.js` |

Times are specific to the build host and test run; actual laptop hardware, PDFs, multilingual embedding inference and larger corpora will change results. `examples/outputs/dashboard_acceptance_report.json` is the machine-readable record.

## Dataset status

The dashboard bundles three original synthetic fixtures and one adapted 20-row excerpt of real Kaggle sustainable-energy rows for immediate offline demonstration. The excerpt is CC BY 4.0, selected from a public CSV mirror, and is not the full source file. The app also lists these current Kaggle dataset pages:

- arXiv metadata: https://www.kaggle.com/datasets/Cornell-University/arxiv (metadata licence stated on the Kaggle page as CC0; related papers have separate terms).
- Global sustainable energy: https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy (Kaggle lists CC BY 4.0).
- Daily Delhi climate: https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data (Kaggle lists CC0).

The full original Kaggle files were **not** downloaded in this environment because outbound DNS access was blocked. The sustainable-energy excerpt `examples/dashboard_datasets/kaggle-energy-sample.csv` was created from 20 source rows visible in the public CSV mirror and was actually loaded through the dashboard API in the acceptance run: 20 records in 0.248 seconds on this host. This validates the dashboard path for a real-data excerpt, not full-dataset throughput, complete metadata coverage, or extraction accuracy on the full corpus. The other cards are labelled “Download + upload”.

## Model status: multilingual language integration

The code includes an optional `intfloat/multilingual-e5-small` retrieval adapter (with a secondary multilingual embedding option), but the model weights could not be loaded in the build environment. Thus Hindi-query/English-passage Hit@1, MRR, graph-path evidence coverage and Hindi answer generation are **not measured**. The fallback is intentionally conservative and should abstain instead of falsely reporting successful semantic translation. To test it on a laptop with internet access, see `docs/USER_GUIDE.md` and run `examples/benchmark_hindi_english.py` after using `run_reliagraph_multilingual_windows.bat`.

## Security boundary

The app is deliberately bound to `127.0.0.1`. It uses salted PBKDF2-HMAC-SHA256 password hashes, server-side sessions in SQLite, an HttpOnly SameSite cookie, expiry and logout, CSRF tokens on mutations, login throttling, upload validation, upload/ZIP size bounds and restrictive browser security headers. The test suite confirms auth setup, unauthenticated blocking, CSRF rejection and logout revocation. This is not a production security audit: the prototype lacks MFA, recovery, roles, antivirus scanning, HTTPS deployment, managed secrets, backup/retention operations and public-hosting hardening.

## Research position and contribution

Established projects already provide graph-based RAG, graph/community retrieval and incremental updates. Official references include Microsoft GraphRAG (https://github.com/microsoft/graphrag; https://microsoft.github.io/graphrag/), LightRAG (https://github.com/HKUDS/LightRAG; https://lightrag.github.io/) and Neo4j GraphRAG (https://neo4j.com/docs/neo4j-graphrag-python/current/). ReliaGraph should not claim those ideas are individually unprecedented.

Our project-level differentiation is the combined, inspectable workflow: source-scoped graph updates; evidence-gated claim candidates; preservation of competing claims; query routing; explicit abstention when evidence/model capability is insufficient; multilingual retrieval diagnostics; and end-user visibility into graph paths, source records, updates and timings in one small local-first prototype. This is a defensible integration contribution, not proof of worldwide novelty or superiority. Controlled evaluation against baselines on held-out multi-domain data remains necessary.

## Release gates not yet passed

1. Load model weights and report real Hindi-English retrieval scores on manually verified questions.
2. Import and measure a full or 500+ record, correctly licensed Kaggle subset; the currently bundled 20-row excerpt does not satisfy this gate.
3. Run paired Phase 1 vs Phase 2 answer-quality comparisons on the same held-out question set and show whether quality loss is within 3 percentage points.
4. Evaluate relation extraction on unseen labelled samples from multiple domains.
5. Perform browser visual QA on the target Windows laptop and security review before any non-local deployment.
