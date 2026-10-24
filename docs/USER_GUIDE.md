# ReliaGraph prototype user guide

## 1. Start the dashboard on Windows

1. Extract the entire ZIP into a folder such as `C:\ReliaGraph`.
2. Install 64-bit Python 3.14 if it is not already installed; the launcher prefers 3.14 and falls back to 3.13, 3.12 or 3.11.
3. Double-click `run_reliagraph_windows.bat`. The first run installs the dashboard dependencies; it needs an internet connection for package installation.
4. Open `http://127.0.0.1:8765` in your browser if it did not open automatically.
5. Create the first local workspace account. Use a strong password of at least 12 characters and keep it somewhere safe; the prototype does not provide account recovery.
6. Keep the command window open while using the app. Press `Ctrl+C` in that window to stop the local server.

The server binds to `127.0.0.1`, not the public network. Do not change it to `0.0.0.0` or expose the prototype to the internet.

## 2. Demo the main workflow

1. Go to **Evaluation & benchmarks** and load the Cybersecurity, Biomedical or Energy synthetic fixture, or load **Global sustainable energy — 20-row real-data excerpt**. The first three fixtures were created for this project; the energy excerpt is selected from a public mirror of the CC BY 4.0 Kaggle dataset and is not the full dataset.
2. Open **Knowledge graph**. Search and filter the graph, zoom, fit the view, select a node and inspect the selected node/source details. The table below the graph exposes source-linked relationships.
3. Open **Ask & trace**, ask a factual question such as `What is access control?`, a broad question such as `What are the key themes across the documents?`, or a relation question such as `How are identity and security logging related?`. Review the route, evidence passages, graph paths and diagnostics rather than trusting a generated sentence alone.
4. Open **Evidence & trust** to review competing claims and findings. Absence of a conflict card does not prove that the source corpus has no contradictions.
5. Open **Security & privacy** to see the local authentication, CSRF, upload-limit and security-header status. This is a prototype security profile, not a production audit.

## 3. Import a real Kaggle export

1. Open **Evaluation & benchmarks** and click **Open source** beside a Kaggle dataset. Download a manageable CSV, JSON or JSONL export from Kaggle using your own account/browser as needed.
2. For arXiv metadata, use `examples/import_kaggle_arxiv.py` to choose a bounded subset from the downloaded JSONL. Example:

   ```powershell
   python -m examples.import_kaggle_arxiv --input "C:\Data\arxiv-metadata-oai-snapshot.json" --output "C:\Data\arxiv_600.jsonl" --max-docs 600
   ```

3. Open **Sources & datasets**, choose the file, give it a dataset name, and paste the Kaggle URL into the source URL field. Import timing and graph counts appear after processing.
4. Re-uploading an updated export with the same dataset name replaces the prior dataset row set and removes stale rows. Use a distinct dataset name for a separate corpus.
5. Run a retrieval smoke test in **Evaluation & benchmarks** and download the JSON report. The smoke test generates questions from current records, so its scores are in-corpus and are not a publishable held-out benchmark.

The dashboard currently supports PDF, TXT, Markdown, CSV, JSON, JSONL/NDJSON and ZIP archives containing those file types. The ingestion adapter has 20 MB per-file and 50 MB total upload/unpacked limits, and tabular uploads are capped at 5,000 records per upload. Large datasets should be subset and imported in batches. PDF extraction quality depends on whether the PDF contains selectable text.

## 4. Edit and delete

- To edit a text document, upload the revised content again with the same filename and dataset name.
- To revise a CSV/JSONL dataset, upload the revised file with the same dataset name. The system removes stale rows that disappeared from that export.
- Use the delete control beside a source to remove one record/document and update the corresponding graph contribution.
- Check the returned timing for your machine; the 60-second requirement is not guaranteed for every PDF, dataset size or enabled model.

## 5. Hindi question over English sources

The multilingual adapter uses `intfloat/multilingual-e5-small` when enabled and successfully loaded. The ZIP does not include model weights. The default launcher runs without the optional model and is expected to abstain for Hindi questions over an English-only corpus instead of pretending keyword overlap is translation.

After the baseline is working, close the baseline server and run `run_reliagraph_multilingual_windows.bat`. First model use may download weights and needs internet, disk space and RAM. Then run the Hindi-English evaluation script:

```powershell
.venv\Scripts\activate
python -m examples.benchmark_hindi_english
```

Use the report's `evaluation_status`, `model_diagnostics`, `hit_at_1` and `mean_reciprocal_rank`. If model status is fallback/unavailable, metric values should stay null. The supplied set is small and project-authored; for final submission, add a separate human-reviewed test set across subject areas and query styles.

## 6. Security boundaries

Passwords use salted PBKDF2-HMAC-SHA256 hashes; sessions are server-side with an HttpOnly cookie, expiry and logout; mutations require a CSRF token; uploads have allow-listed types and size/ZIP limits; responses include restrictive browser security headers. The database is local to the workspace folder. Keep OS login security and filesystem permissions enabled.

This prototype does not include multi-user roles, password reset, MFA, antivirus scanning, remote TLS termination, managed secrets, disaster-recovery backups or a production security audit. It is not ready for public internet deployment. Treat user-uploaded documents as untrusted, and do not put sensitive or regulated documents in a demo without permission.


## Sign in and sign up

The login page always provides both **Sign in** and **Sign up** tabs. On an empty installation the Sign up tab is selected automatically; on an existing installation the Sign in tab is selected, while Sign up remains available to create another local account. Use a username of 3–64 characters and a password of at least 12 characters.

All accounts currently share the local knowledge workspace and its documents/graph. This prototype provides local authentication, not per-user data isolation or public multi-tenant hosting. If you forget an existing password, use a different username or follow the safe account-reset procedure in the README after stopping the server.

## Optional Neo4j + Chroma stores

The zero-configuration default remains embedded SQLite + NetworkX so the prototype runs without Docker or a remote database. To meet a submission rubric that explicitly asks for Neo4j and a vector database, the package now includes an optional mirror adapter and `docker-compose.stores.yml` for a local Neo4j Community instance, plus `requirements-stores.txt` for the Python Neo4j driver and ChromaDB.

1. Install Docker Desktop if you want the Neo4j option, then run `docker compose -f docker-compose.stores.yml up -d` from the project folder.
2. Set environment variables in the same CMD window before launching: `set RELIAGRAPH_NEO4J_URI=bolt://127.0.0.1:7687`, `set RELIAGRAPH_NEO4J_USER=neo4j`, and `set RELIAGRAPH_NEO4J_PASSWORD=ReliaGraphChangeThisPassword` (change this password first in a private local setup).
3. Install optional adapters with `.venv\Scripts\python.exe -m pip install -r requirements-stores.txt`.
4. For Chroma's local persistent vector collection, set `RELIAGRAPH_CHROMA=1`. It writes to `data/chroma` by default. Chroma must be installed from `requirements-stores.txt`.
5. Relaunch and inspect the workspace `/api/dashboard` response `stores` status. Do not claim Neo4j or Chroma is active unless its status says `connected: true`.

Neo4j is a separately running service; merely installing the Python driver does not create a database. See the official Neo4j Python driver documentation and Chroma persistent-client guide. The built-in SQLite/NetworkX and local retrieval remain available if either optional service is not configured.

### Source edits and resets

Re-upload a file or dataset using the same dataset/source name to replace its records (this is the edit workflow and recomputes the source's graph contribution). Individual source rows can be deleted. **Reset all sources** removes all uploaded source documents and rebuilds the graph but preserves the local login account. Export the source manifest first if you need a record of the current workspace.
