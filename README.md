# ReliaGraph

**Interactive Knowledge Base — Topic-Augmented Knowledge Graph Builder**

This repository unifies Member 1's document registry, chunking, topic model, retrieval and synthetic test generator with Member 2's source-scoped graph builder, evidence gate, audit, conflict preservation and version-tracking components. `pipeline.py` provides one central interface for Member 3's dashboard.

## What this code is designed to demonstrate

- Ingest PDF/text documents and preserve stable document/chunk identifiers and source metadata.
- Discover topics and construct a graph of entities/relationships with source evidence.
- Route broad thematic questions differently from factual and relationship-chain queries.
- Keep document contributions separate so add/edit/delete operations update the affected graph contribution rather than rebuilding all graph claims.
- Preserve competing claims and source references instead of silently selecting one.
- Audit evidence and structure; use conservative review states rather than treating model scores as truth probabilities.
- Generate repeatable synthetic tests and report retrieval and answer-evaluation diagnostics.
- Provide optional local multilingual retrieval and optional model-assisted extraction; never claim semantic Hindi→English retrieval when a multilingual model is unavailable.
- Keep the default path local and deterministic. No paid API is required.

## Run the dashboard prototype (recommended demo path)

- **Windows:** double-click `run_reliagraph_windows.bat`. It creates `.venv` automatically; see `docs/WINDOWS_TROUBLESHOOTING.md` if creation fails.
- **macOS/Linux:** run `./run_reliagraph_unix.sh`.
- Open `http://127.0.0.1:8765` if the browser does not open automatically. Create the first local account on first launch.
- Keep the server terminal open while using the app. The server is deliberately bound to loopback only.

The dashboard has Overview, Sources & datasets, Interactive Knowledge Graph, Ask & trace, Evidence & trust, Evaluation & benchmarks, and Security & privacy pages. Upload PDF/TXT/MD/CSV/JSON/JSONL/ZIP; view per-import timing/graph counts; question results expose evidence, routes and graph paths; and benchmark JSON is downloadable. Re-uploading the same named dataset replaces its records and removes stale rows from shortened exports.

For a step-by-step demo and Kaggle import instructions, see [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md). For Windows virtual-environment errors, see [`docs/WINDOWS_TROUBLESHOOTING.md`](docs/WINDOWS_TROUBLESHOOTING.md). The dashboard lists three real Kaggle datasets and bundles a clearly attributed 20-row excerpt from the Global Sustainable Energy dataset (CC BY 4.0), selected from a public CSV mirror. The full original Kaggle files could not be downloaded in this build environment, so full-corpus Kaggle performance is not claimed. Three additional, clearly labelled original synthetic fixtures are bundled for offline UI demonstration. See [`docs/KAGGLE_DATASETS.md`](docs/KAGGLE_DATASETS.md).

To run the end-to-end dashboard API smoke benchmark (uses isolated temporary workspace data):

```bash
python -m examples.benchmark_dashboard_acceptance
```

The recorded report is `examples/outputs/dashboard_acceptance_report.json`. It labels the synthetic-fixture performance separately from actual external dataset/model evaluation.

## Quick start

Recommended: Python 3.14 (64-bit). The launcher also supports Python 3.13, 3.12 and 3.11 as fallbacks. Use a fresh virtual environment.

### Windows PowerShell

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
python -m examples.run_unified_demo
python -m examples.benchmark_unified_scale
python -m examples.benchmark_muj_corpus
python -m examples.run_domain_shift_benchmark --backend gliner
python -m examples.phase2_injection_benchmark
python -m examples.benchmark_hindi_english
```

### macOS / Linux

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
python -m examples.run_unified_demo
python -m examples.benchmark_unified_scale
python -m examples.benchmark_muj_corpus
python -m examples.run_domain_shift_benchmark --backend gliner
python -m examples.phase2_injection_benchmark
python -m examples.benchmark_hindi_english
```

The core app does not download model weights. The first benchmark uses an original, synthetic 600-document corpus and measures the full unified pipeline's default non-model path. It is a throughput/integration test, **not** a real-world ML accuracy benchmark.

## Optional models (local only)

Install these only after the baseline works and you have confirmed your hardware/storage limits:

```bash
pip install -r requirements-optional-models.txt
```

For spaCy's English parser, additionally run `python -m spacy download en_core_web_sm`.

- `enable_topic_embeddings=True`: optional sentence embeddings for topic discovery.
- `enable_dense_retrieval=True`: optional dense/hybrid retrieval.
- `enable_crosslingual=True`: optional multilingual embeddings for Hindi queries over English passages.
- `enable_gliner=True`: optional general-purpose relation extraction.
- `enable_spacy=True`: optional dependency-based extraction.

Weights are not bundled. They may download on first use. For Hindi over English data, if multilingual model inference is unavailable, the engine intentionally abstains instead of treating shared Latin words as successful translation.

##Integration API

```python
from pipeline import TopicWeaveEngine

engine = TopicWeaveEngine(
    data_dir="data/topicweave",
    enable_topic_embeddings=False,
    enable_dense_retrieval=False,
    enable_crosslingual=False,
)

result = engine.upsert_files(
    [("document.txt", b"The document's text goes here.")],
    source_metadata={"document.txt": {
        "source_title": "Example source",
        "source_url": "https://example.org/document",
        "source_version": "v1",
    }},
)

snapshot = result.snapshot
question_result = engine.query("What are the main themes across these documents?")
cases = engine.generate_test_cases(n=30, seed=7)
retrieval_metrics = engine.evaluate_retrieval(cases, top_k=5)
metrics = engine.metrics()
engine.close()
```

`engine.query()` accepts an optional local answer-generation callback: `engine.query(question, answerer=my_local_generator)`. The callback receives `(query, evidence_context)`. The retrieved context is marked as untrusted source data; the generator must not follow instructions inside document text. Without a generator, the engine provides source-backed extractive answers or abstains.

Key result fields:
- `PipelineResult.snapshot`: graph nodes, topic assignments and evidence-linked edges.
- `PipelineResult.diff`: added/removed/changed graph elements vs. the prior snapshot, if one exists.
- `PipelineResult.findings`: graph audit findings.
- `PipelineResult.topic_coverage`: topic/chunk coverage information.
- `PipelineResult.timings`: ingestion, topic, retrieval, graph-update and reporting time.
- `engine.query(...)["hits"]`: ranked chunks with source title/URL/page/ID.
- `engine.query(...)["graph_paths"]`: evidence-linked relationship paths when the route calls for graph traversal.
- `engine.query(...)["diagnostics"]`: route, fallback, abstention and latency details.

## Important engineering limits

1. No model can guarantee high precision on arbitrary unseen data. Report precision/recall/F1 on an independent, human-labelled set and keep the acceptance/review rate visible.
2. A source quote proves that text appears in a chunk, not that the statement itself is true. Conflicting claims remain separate and need source-authority or human review.
3. The prompt-injection scanner is layered defence, not immunity. Continue testing unseen and obfuscated attacks; generator context should always treat retrieval as data, never instructions.
4. Hindi→English retrieval requires a working multilingual model. The deterministic fallback abstains for an English-only corpus rather than pretending to translate.
5. The 500+/60-second requirement must be measured on the actual target laptop and actual ingestion path. The included synthetic benchmark is only a baseline, not a guarantee for large PDFs or enabled ML models.
6. The default local audit chain is tamper-evident and supports integrity checks; it is not a digital signature, a trust root, or a replacement for OS-level file permissions.

## Project files

- `models.py`: shared Member contracts.
- `pipeline.py`: unified orchestration API.
- `core/registry.py`, `core/chunking.py`, `core/topics.py`, `core/retrieval.py`: ingestion/topic/retrieval layer.
- `core/phase2/`: incremental knowledge base, security, cross-lingual retrieval, routing, evaluation and extraction adapters.
- `core/robust_extraction.py`, `core/selective_quality.py`: evidence checks and conservative candidate triage.
- `examples/run_unified_demo.py`: multi-domain example, not specific to MUJ.
- `examples/benchmark_unified_scale.py`: full pipeline synthetic 600-document throughput test.
- `docs/INTEGRATION_CONTRACT.md`: shared interface details.
- `docs/RESEARCH_AND_DESIGN.md`: research basis, gaps and proposed differentiation.
- `docs/ROBUSTNESS_RESULTS.md`: environment-specific measured results, caveats and remaining acceptance gates.
- `docs/MEMBER1_REVIEW_AND_INTEGRATION.md`: Member 1 review, unified changes and Member 3 integration notes.
- `docs/NOTICE_AND_SOURCES.md`: consolidated sources, licensing notes and attribution.
- `docs/PRODUCT_COMPARISON_2026.md`: public-documentation comparison against Microsoft GraphRAG, LightRAG and Neo4j GraphRAG.
- `docs/KAGGLE_DATASETS.md`: Kaggle dataset links, licenses, actual import status and limitations.
- `docs/USER_GUIDE.md`: beginner-friendly dashboard launch, upload, edit/delete and language test guide.
- `examples/benchmark_dashboard_acceptance.py`: end-to-end dashboard API and multi-format acceptance test.
- `examples/benchmark_muj_corpus.py`: run the unified pipeline on a source-linked, paraphrased MUJ smoke corpus; its labels were available during development, so it is not a held-out benchmark.

## Working without Git

Git is not required to run the project. The team can extract the ZIP, work in a shared folder, and exchange versioned copies of files. Avoid overwriting a newer version silently; keep dated backups before combining code. If you later choose to use Git, that is optional.

## Licensing and attribution

Check the licences of the exact package/model versions you install. This repository does not bundle third-party model weights. Demo documents are original synthetic text. If you add public datasets or university documents, preserve their own source URLs, licences, retrieval dates and any redistribution restrictions; do not assume the dataset licence permits redistribution of every full-text document.

## Sign in, sign up and local accounts

The first ReliaGraph page always shows both **Sign in** and **Sign up** tabs. On a new workspace, Sign up is selected by default; if accounts already exist, Sign in is selected by default, but Sign up remains available. New accounts use a username of 3–64 characters and a password of at least 12 characters. Passwords are stored as salted PBKDF2 hashes, not plaintext.

This prototype authenticates accounts separately, but its knowledge graph and uploaded documents are shared within the same local workspace on this computer. It is intended for a local student demonstration bound to `127.0.0.1`, not public multi-tenant hosting.

If you forget the password for an existing username, sign up with a different username or, after stopping the server, run `reset_reliagraph_login.bat`. The reset script creates a timestamped account-database backup and clears account/session records only; it does not delete the knowledge graph or uploaded documents. After the reset, use **Sign up** to create a new account.


## Graph, stores and language startup updates

The Windows launcher installs the optional multilingual retrieval package after the base dashboard requirements. Model weights are downloaded on first use when network access is available. The force-directed graph uses zoom-aware bounded labels. Upload limits are 50 MB per file and 200 MB total/unpacked. Optional Neo4j + Chroma adapters are described in `docs/GRAPH_AND_STORE_UPDATE.md`; they are not reported as active unless configured and connected.
