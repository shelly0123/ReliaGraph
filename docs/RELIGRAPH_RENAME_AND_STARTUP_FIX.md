# ReliaGraph final packaging and startup note

- Product name applied to the dashboard title, authentication screen, sidebar logo/name, answer header, footer, launchers, README and user-facing docs. Internal Python class identifiers such as `TopicWeaveEngine` remain unchanged for compatibility with the Member 1/2 integration API.
- Windows launcher now prefers Python 3.14 and falls back to 3.13, 3.12 and 3.11; create `.venv` if missing; detect and rename an incomplete `.venv`; and invoke `.venv\Scripts\python.exe` directly rather than relying on shell activation.
- The multilingual launcher now gives a clear first-run dependency message and expects the baseline environment to have been created first.
- Test run in this build environment: `100 passed` (3.90 seconds).
- Dashboard acceptance benchmark rerun: 600-record upload about 4.80 seconds via the API; 3 synthetic domains; 20-row attributed Kaggle-derived excerpt; PDF/TXT/Markdown/CSV/JSON/JSONL/NDJSON/ZIP paths; factual/thematic/relation-chain query paths.
- Actual Windows batch execution cannot be simulated in this Linux build environment. The batch files were inspected and packaged with CRLF line endings; run them on Windows to validate the local Python installation and network access.
- Hindi-English embedding weights remain unavailable in this environment, so multilingual retrieval accuracy is still unmeasured.
