"""ReliaGraph local prototype web server.

Security profile is intended for a single-user local demo on 127.0.0.1. Do not expose
this starter server to the public internet; use proper HTTPS/reverse-proxy and an audited
identity provider before production deployment.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).resolve().parent
STATIC = WEB_ROOT / "static"
DATA_HOME = Path(os.environ.get("RELIAGRAPH_DATA_DIR", os.environ.get("TOPICWEAVE_DATA_DIR", ROOT / "data" / "dashboard")))
DATA_HOME.mkdir(parents=True, exist_ok=True)
AUTH_DB = DATA_HOME / "auth.sqlite3"
MAX_REQUEST_BYTES = 210 * 1024 * 1024
SESSION_SECONDS = 8 * 60 * 60
PASSWORD_MIN_LENGTH = 12
COOKIE_SECURE = os.environ.get("RELIAGRAPH_HTTPS", os.environ.get("TOPICWEAVE_HTTPS", "0")) == "1"
LOGIN_FAILS: dict[str, list[float]] = {}
SIGNUP_ATTEMPTS: dict[str, list[float]] = {}
LAST_FINDINGS: list[dict[str, Any]] = []
LAST_IMPORTS: list[dict[str, Any]] = []
MODEL_STATUS = {
    "crosslingual_enabled": os.environ.get("TW_ENABLE_CROSSLINGUAL", "0") == "1",
    "gliner_enabled": os.environ.get("TW_ENABLE_GLINER", "0") == "1",
    "topic_embeddings_enabled": os.environ.get("TW_ENABLE_TOPIC_EMBEDDINGS", "0") == "1",
    "dense_retrieval_enabled": os.environ.get("TW_ENABLE_DENSE_RETRIEVAL", "0") == "1",
}

# Import project root modules when launched as `python -m uvicorn webapp.main:app`.
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline import TopicWeaveEngine
from webapp.dataset_adapter import prepare_uploads, MAX_FILE_BYTES, MAX_TOTAL_UNPACKED
from core.optional_stores import optional_store_status, sync_optional_stores, clear_optional_stores, check_neo4j


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(AUTH_DB, timeout=10, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def _init_auth() -> None:
    with _db() as con:
        con.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_salt BLOB NOT NULL, password_hash BLOB NOT NULL, created_at REAL NOT NULL)")
        con.execute("CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, username TEXT NOT NULL, csrf_token TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expiry ON sessions(expires_at)")
        con.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))


def _password_hash(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    salt = salt or secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000, dklen=32)
    return salt, derived


def _has_user() -> bool:
    with _db() as con:
        return con.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _session(request: Request) -> dict[str, str] | None:
    token = request.cookies.get("tw_session", "")
    if not token or len(token) > 200:
        return None
    with _db() as con:
        row = con.execute("SELECT username, csrf_token, expires_at FROM sessions WHERE token_hash=?", (_hash_token(token),)).fetchone()
        if not row:
            return None
        if row["expires_at"] < time.time():
            con.execute("DELETE FROM sessions WHERE token_hash=?", (_hash_token(token),))
            return None
        return {"username": row["username"], "csrf": row["csrf_token"]}


def _need_auth(request: Request, *, csrf: bool = False) -> dict[str, str]:
    sess = _session(request)
    if not sess:
        raise HTTPException(status_code=401, detail="Sign in to use this workspace.")
    if csrf:
        header = request.headers.get("x-csrf-token", "")
        if not header or not hmac.compare_digest(header, sess["csrf"]):
            raise HTTPException(status_code=403, detail="CSRF validation failed. Refresh the page and try again.")
    return sess


engine = TopicWeaveEngine(
    data_dir=DATA_HOME / "reliagraph",
    enable_topic_embeddings=MODEL_STATUS["topic_embeddings_enabled"],
    enable_dense_retrieval=MODEL_STATUS["dense_retrieval_enabled"],
    enable_crosslingual=MODEL_STATUS["crosslingual_enabled"],
    enable_gliner=MODEL_STATUS["gliner_enabled"],
    enable_spacy=False,
)
_init_auth()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    engine.close()

app = FastAPI(title="ReliaGraph", version="1.0.0-prototype", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    # Reject over-large bodies before application-level parsing; upload endpoint has tighter limits.
    content_len = request.headers.get("content-length")
    if content_len:
        try:
            if int(content_len) > MAX_REQUEST_BYTES:
                return JSONResponse({"detail": "Request exceeds the 210 MB limit."}, status_code=413)
        except ValueError:
            return JSONResponse({"detail": "Invalid Content-Length."}, status_code=400)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/auth/status")
def auth_status(request: Request):
    sess = _session(request)
    return {"setup_required": not _has_user(), "authenticated": bool(sess),
            "username": sess["username"] if sess else None, "csrf_token": sess["csrf"] if sess else None}


@app.post("/api/auth/setup")
def setup_account(request: Request, payload: dict[str, Any]):
    if _has_user():
        raise HTTPException(409, "Initial account has already been created. Sign in instead.")
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    if len(username) < 3 or len(username) > 64 or not all(c.isalnum() or c in "._-@" for c in username):
        raise HTTPException(400, "Username must be 3–64 characters and use letters, numbers, ., _, -, or @.")
    if len(password) < PASSWORD_MIN_LENGTH:
        raise HTTPException(400, f"Choose a password with at least {PASSWORD_MIN_LENGTH} characters.")
    salt, hashed = _password_hash(password)
    try:
        with _db() as con:
            con.execute("INSERT INTO users(username,password_salt,password_hash,created_at) VALUES(?,?,?,?)",
                        (username, salt, hashed, time.time()))
    except sqlite3.IntegrityError:
        raise HTTPException(409, "An account already exists.")
    return _create_session(username)


@app.post("/api/auth/signup")
def signup(request: Request, payload: dict[str, Any]):
    """Create a local account. Registration is always available on the local prototype."""
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    times = [t for t in SIGNUP_ATTEMPTS.get(ip, []) if now - t < 900]
    SIGNUP_ATTEMPTS[ip] = times
    if len(times) >= 10:
        raise HTTPException(429, "Too many account-creation attempts from this browser. Wait 15 minutes before retrying.")

    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    if len(username) < 3 or len(username) > 64 or not all(c.isalnum() or c in "._-@" for c in username):
        raise HTTPException(400, "Username must be 3–64 characters and use letters, numbers, ., _, -, or @.")
    if len(password) < PASSWORD_MIN_LENGTH:
        raise HTTPException(400, f"Choose a password with at least {PASSWORD_MIN_LENGTH} characters.")
    salt, hashed = _password_hash(password)
    try:
        with _db() as con:
            con.execute("INSERT INTO users(username,password_salt,password_hash,created_at) VALUES(?,?,?,?)",
                        (username, salt, hashed, now))
    except sqlite3.IntegrityError:
        SIGNUP_ATTEMPTS[ip].append(now)
        raise HTTPException(409, "That username already exists. Choose another username or sign in.")
    SIGNUP_ATTEMPTS[ip].append(now)
    return _create_session(username)


def _create_session(username: str) -> JSONResponse:
    raw = secrets.token_urlsafe(36)
    csrf = secrets.token_urlsafe(24)
    now = time.time()
    with _db() as con:
        con.execute("INSERT INTO sessions(token_hash,username,csrf_token,created_at,expires_at) VALUES(?,?,?,?,?)",
                    (_hash_token(raw), username, csrf, now, now + SESSION_SECONDS))
    resp = JSONResponse({"ok": True, "username": username, "csrf_token": csrf, "expires_in_seconds": SESSION_SECONDS})
    resp.set_cookie("tw_session", raw, max_age=SESSION_SECONDS, httponly=True, secure=COOKIE_SECURE,
                    samesite="lax", path="/")
    return resp


@app.post("/api/auth/login")
def login(request: Request, payload: dict[str, Any]):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    times = [t for t in LOGIN_FAILS.get(ip, []) if now - t < 900]
    LOGIN_FAILS[ip] = times
    if len(times) >= 5:
        raise HTTPException(429, "Too many attempts from this browser. Wait 15 minutes before retrying.")
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    with _db() as con:
        row = con.execute("SELECT username,password_salt,password_hash FROM users WHERE username=?", (username,)).fetchone()
    valid = False
    if row:
        _, candidate = _password_hash(password, bytes(row["password_salt"]))
        valid = hmac.compare_digest(candidate, bytes(row["password_hash"]))
    if not valid:
        LOGIN_FAILS[ip].append(now)
        # Delay a little to make rapid guessing less attractive.
        time.sleep(0.25)
        raise HTTPException(401, "Invalid username or password.")
    LOGIN_FAILS.pop(ip, None)
    return _create_session(username)


@app.post("/api/auth/logout")
def logout(request: Request):
    sess = _need_auth(request, csrf=True)
    token = request.cookies.get("tw_session", "")
    with _db() as con:
        con.execute("DELETE FROM sessions WHERE token_hash=?", (_hash_token(token),))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("tw_session", path="/")
    return resp


def _doc_rows() -> list[dict[str, Any]]:
    rows = []
    chunk_counts: dict[str, int] = {}
    for did, chunks in engine.registry.chunks_by_doc.items():
        chunk_counts[did] = len(chunks)
    for did, doc in engine.registry.docs.items():
        rows.append({"doc_id": did, "filename": doc.filename, "title": doc.source_title or doc.filename,
                     "source_url": doc.source_url or "", "status": doc.status, "error": doc.error,
                     "content_hash": doc.content_hash, "chunks": chunk_counts.get(did, 0),
                     "bytes_or_chars": sum(len(c.text) for c in engine.registry.chunks_by_doc.get(did, [])),
                     "metadata": doc.metadata or {}})
    rows.sort(key=lambda r: r["filename"].casefold())
    return rows


def _snapshot_dict() -> dict[str, Any]:
    snap = engine.last_snapshot
    if not snap:
        return {"snapshot_id": None, "nodes": [], "edges": [], "topics": [], "metadata": {}}
    return asdict(snap)


@app.get("/api/stores/check")
def stores_check(request: Request):
    _need_auth(request)
    check_neo4j()
    return optional_store_status()


@app.get("/api/dashboard")
def dashboard(request: Request):
    _need_auth(request)
    metrics = engine.metrics()
    snap = _snapshot_dict()
    return {"metrics": metrics, "documents": _doc_rows(), "snapshot": snap,
            "conflicts": engine.graph_service.competing_claims(), "findings": LAST_FINDINGS,
            "imports": LAST_IMPORTS[-10:], "models": _model_state(), "stores": optional_store_status()}


def _model_state() -> dict[str, Any]:
    cross = getattr(engine, "crosslingual_retriever", None)
    status = {"multilingual_enabled": bool(engine.crosslingual_enabled),
              "multilingual_available": bool((getattr(getattr(cross, "primary", None), "_model", None) is not None) or (getattr(getattr(cross, "secondary", None), "_model", None) is not None)) if cross else False,
              "gliner_enabled": MODEL_STATUS["gliner_enabled"],
              "dense_retrieval_enabled": bool(engine.retriever.dense_enabled),
              "topic_embeddings_enabled": MODEL_STATUS["topic_embeddings_enabled"],
              "answer_generation": "extractive / abstain by default; no generative LLM is bundled"}
    # Optional adapter versions can be different; expose warnings rather than a green check.
    try:
        status["crosslingual_diagnostics"] = {"active_model": getattr(getattr(cross, "_active", None), "model_name", None), "mode": getattr(cross, "_mode", "not_started"), "errors": getattr(cross, "_errors", []), "load_seconds": getattr(getattr(cross, "_active", None), "load_seconds", None)} if cross else {}
    except Exception:
        status["crosslingual_diagnostics"] = {}
    return status


@app.post("/api/documents/upload")
async def upload_documents(request: Request, files: list[UploadFile] = File(...),
                           dataset_name: str = Form(""),
                           source_url: str = Form("")):
    global LAST_FINDINGS
    _need_auth(request, csrf=True)
    dataset_name = dataset_name.strip()[:160]
    source_url = source_url.strip()[:1000]
    if source_url and not (source_url.startswith("https://") or source_url.startswith("http://")):
        raise HTTPException(400, "Source URL must begin with http:// or https://.")
    inputs = []
    total = 0
    for upload in files:
        name = upload.filename or "upload.txt"
        content = await upload.read(MAX_FILE_BYTES + 1)
        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(413, f"{name} exceeds the 50 MB per-file limit.")
        total += len(content)
        if total > MAX_TOTAL_UNPACKED:
            raise HTTPException(413, "Total upload exceeds 50 MB.")
        inputs.append((name, content))
    if not dataset_name:
        # Give unnamed uploads a stable, human-readable name without merging unrelated uploads
        # into one destructive default dataset.
        dataset_name = ", ".join(Path(name).name for name, _ in inputs)[:160] or "Uploaded dataset"
    try:
        prepared, manifest = prepare_uploads(inputs, dataset_name=dataset_name, source_url=source_url)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    metadata = {r.filename: r.as_metadata() for r in prepared}
    prepared_names = {r.filename for r in prepared}
    # Re-importing the same named dataset replaces it as a set. Capture stale rows before the
    # upsert, then remove rows that disappeared from the new export so shorter CSVs do not leave
    # outdated claims in the graph. Unnamed uploads receive filename-derived names above.
    stale_names = [doc.filename for doc in engine.registry.docs.values()
                   if (doc.metadata or {}).get("dataset_name") == dataset_name
                   and doc.filename not in prepared_names]
    start = time.perf_counter()
    try:
        result = engine.upsert_files([(r.filename, r.content) for r in prepared], source_metadata=metadata)
        stale_delete_seconds = 0.0
        last_delete_result = None
        for stale_name in stale_names:
            del_start = time.perf_counter()
            last_delete_result = engine.delete_document(stale_name)
            stale_delete_seconds += time.perf_counter() - del_start
        # Use the final graph snapshot and findings after any removed rows.
        snapshot = engine.last_snapshot or result.snapshot
        try:
            sync_optional_stores(snapshot, list(engine.registry.chunks.values()))
        except Exception:
            pass
        if last_delete_result is not None:
            LAST_FINDINGS = [asdict(f) for f in last_delete_result.findings]
    except Exception as exc:
        # Keep a diagnostic that is safe for the local demo; no stack trace to browser.
        raise HTTPException(500, f"Ingestion failed: {type(exc).__name__}. Check the local server log.") from exc
    wall = time.perf_counter() - start
    manifest["wall_clock_seconds"] = round(wall, 4)
    manifest["pipeline_timings"] = {k: round(float(v), 4) for k, v in result.timings.items()}
    manifest["changed_documents"] = len(result.changed_documents)
    manifest["stale_records_removed"] = len(stale_names)
    manifest["stale_removal_seconds"] = round(stale_delete_seconds, 4)
    manifest["changes"] = result.change_summary
    manifest["graph_counts"] = {"nodes": len(snapshot.nodes) if snapshot else 0,
                                 "edges": len(snapshot.edges) if snapshot else 0,
                                 "topics": len(snapshot.topics) if snapshot else 0}
    manifest["injection_segments_quarantined"] = result.diagnostics.get("blocked_injection_segments", 0)
    manifest["model_mode"] = "optional ML disabled unless enabled explicitly" if not any(MODEL_STATUS.values()) else "optional model mode requested; check model diagnostics"
    manifest["under_60_seconds"] = wall < 60
    LAST_IMPORTS.append(manifest)
    LAST_FINDINGS = [asdict(f) for f in result.findings]
    try:
        sync_optional_stores(result.snapshot, list(engine.registry.chunks.values()))
    except Exception:
        pass
    return {"ok": True, "manifest": manifest, "snapshot_id": snapshot.snapshot_id if snapshot else None,
            "findings": LAST_FINDINGS[:30], "models": _model_state()}


@app.delete("/api/documents/{doc_id}")
def delete_document(request: Request, doc_id: str):
    global LAST_FINDINGS
    _need_auth(request, csrf=True)
    doc = engine.registry.docs.get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    start = time.perf_counter()
    result = engine.delete_document(doc.filename)
    wall = time.perf_counter() - start
    LAST_FINDINGS = [asdict(f) for f in result.findings]
    try:
        sync_optional_stores(result.snapshot, list(engine.registry.chunks.values()))
    except Exception:
        pass
    return {"ok": True, "deleted": doc.filename, "seconds": round(wall, 4),
            "under_60_seconds": wall < 60, "metrics": engine.metrics(),
            "snapshot_id": result.snapshot.snapshot_id if result.snapshot else None}


@app.delete("/api/documents")
def reset_all_documents(request: Request):
    """Remove all indexed source documents and rebuild an empty graph; keep user accounts."""
    global LAST_FINDINGS, LAST_IMPORTS
    _need_auth(request, csrf=True)
    started = time.perf_counter()
    names = [doc.filename for doc in list(engine.registry.docs.values())]
    deleted = 0
    for name in names:
        try:
            engine.delete_document(name)
            deleted += 1
        except Exception:
            continue
    LAST_FINDINGS = []
    LAST_IMPORTS = []
    try:
        clear_optional_stores()
        sync_optional_stores(engine.last_snapshot, list(engine.registry.chunks.values()))
    except Exception:
        pass
    return {"ok": True, "deleted_documents": deleted, "seconds": round(time.perf_counter()-started, 4),
            "metrics": engine.metrics(), "snapshot_id": engine.last_snapshot.snapshot_id if engine.last_snapshot else None}


@app.post("/api/query")
def query(request: Request, payload: dict[str, Any]):
    _need_auth(request, csrf=True)
    q = str(payload.get("query", "")).strip()
    if not q or len(q) > 2000:
        raise HTTPException(400, "Enter a question of 1–2,000 characters.")
    top_k = max(1, min(10, int(payload.get("top_k", 5))))
    result = engine.query(q, top_k=top_k, include_graph_paths=True)
    # Keep response small and avoid exposing internal traceback/configuration.
    return result


@app.get("/api/conflicts")
def conflicts(request: Request):
    _need_auth(request)
    return {"conflicts": engine.graph_service.competing_claims(), "findings": LAST_FINDINGS}


@app.get("/api/benchmarks/catalog")
def benchmark_catalog(request: Request):
    _need_auth(request)
    return {"datasets": [
        {"id": "kaggle-arxiv", "name": "arXiv scholarly metadata (Kaggle)", "kind": "External Kaggle dataset", "status": "Download + upload", "url": "https://www.kaggle.com/datasets/Cornell-University/arxiv",
         "license": "CC0 for metadata; papers have separate terms", "description": "Large research corpus with titles, abstracts, authors and categories. Download a manageable CSV/JSON/JSONL subset, then upload it here.",
         "test_note": "The build environment could not resolve external hosts, so the actual dataset was not downloaded or benchmarked as part of this package."},
        {"id": "kaggle-energy", "name": "Global sustainable energy (Kaggle full dataset)", "kind": "External Kaggle dataset", "status": "Download + upload", "url": "https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy",
         "license": "CC BY 4.0", "description": "Full country-year sustainable-energy dataset. For your own full-corpus test, download the source CSV and upload it in Sources & datasets.",
         "test_note": "The full original file could not be fetched from the build host. A small, attributed 20-row excerpt from the public CSV mirror is bundled separately for a real-data dashboard smoke test."},
        {"id": "kaggle-energy-sample", "name": "Global sustainable energy — 20-row real-data excerpt", "kind": "Bundled Kaggle-derived excerpt", "status": "Ready to load", "url": "https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy",
         "license": "CC BY 4.0; selected rows/columns adapted", "description": "20 source-derived country-year rows across Afghanistan, Albania, Algeria and Angola. A real-data import sample, not the full dataset; source attribution is recorded with every imported row.",
         "test_note": "Selected rows/columns from the public mirror of the Kaggle dataset. The full original data file is not included."},
        {"id": "kaggle-delhi-climate", "name": "Daily Delhi climate (Kaggle)", "kind": "External Kaggle dataset", "status": "Download + upload", "url": "https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data",
         "license": "CC0 per Kaggle dataset card", "description": "Small daily weather time-series CSVs. Useful to check that tabular rows can become source-linked records outside research and cybersecurity domains.",
         "test_note": "External data is not bundled in this build; download the CSV and upload it to measure the actual pipeline."},
        {"id": "synthetic-cyber", "name": "Cybersecurity controls (synthetic demo)", "kind": "Bundled synthetic fixture", "status": "Ready to load", "url": "", "license": "Original project fixture", "description": "Original synthetic security-policy sentences; no personal data and not sourced from Kaggle."},
        {"id": "synthetic-biomed", "name": "Biomedical research (synthetic demo)", "kind": "Bundled synthetic fixture", "status": "Ready to load", "url": "", "license": "Original project fixture", "description": "Original synthetic biomedical examples designed to test topic and relation variation."},
        {"id": "synthetic-energy", "name": "Energy systems (synthetic demo)", "kind": "Bundled synthetic fixture", "status": "Ready to load", "url": "", "license": "Original project fixture", "description": "Original synthetic energy-system examples for cross-domain robustness checks."},
    ]}


@app.post("/api/benchmarks/load-demo")
def load_demo(request: Request, payload: dict[str, Any]):
    global LAST_FINDINGS
    _need_auth(request, csrf=True)
    name = str(payload.get("dataset", ""))
    bundled = {"synthetic-cyber", "synthetic-biomed", "synthetic-energy", "kaggle-energy-sample"}
    if name not in bundled:
        raise HTTPException(400, "Unknown bundled dataset.")
    fp = ROOT / "examples" / "dashboard_datasets" / f"{name}.csv"
    if not fp.is_file():
        raise HTTPException(500, "Bundled demo dataset file is missing.")
    data = fp.read_bytes()
    source_url = "https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy" if name == "kaggle-energy-sample" else ""
    prepared, manifest = prepare_uploads([(fp.name, data)], dataset_name=name, source_url=source_url)
    meta = {r.filename: r.as_metadata() for r in prepared}
    start = time.perf_counter()
    result = engine.upsert_files([(r.filename, r.content) for r in prepared], source_metadata=meta)
    wall = time.perf_counter() - start
    manifest["wall_clock_seconds"] = round(wall, 4)
    manifest["pipeline_timings"] = {k: round(float(v), 4) for k, v in result.timings.items()}
    manifest["changed_documents"] = len(result.changed_documents)
    manifest["changes"] = result.change_summary
    manifest["graph_counts"] = {"nodes": len(result.snapshot.nodes) if result.snapshot else 0,
                                 "edges": len(result.snapshot.edges) if result.snapshot else 0,
                                 "topics": len(result.snapshot.topics) if result.snapshot else 0}
    manifest["under_60_seconds"] = wall < 60
    if name == "kaggle-energy-sample":
        manifest["source_attribution"] = (
            "Kaggle-derived 20-row selected-column excerpt from Global Data on Sustainable Energy; "
            "dataset page: https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy; "
            "CC BY 4.0; adapted by selecting 20 country-year rows and 4 indicator columns. "
            "Source rows were inspected via the public CSV mirror; this is not the full downloaded source file."
        )
    else:
        manifest["source_attribution"] = "Original synthetic demonstration fixture, not a Kaggle download."
    LAST_IMPORTS.append(manifest)
    LAST_FINDINGS = [asdict(f) for f in result.findings]
    try:
        sync_optional_stores(result.snapshot, list(engine.registry.chunks.values()))
    except Exception:
        pass
    return {"ok": True, "manifest": manifest, "snapshot_id": result.snapshot.snapshot_id if result.snapshot else None}


@app.post("/api/benchmarks/run")
def run_benchmark(request: Request, payload: dict[str, Any]):
    _need_auth(request, csrf=True)
    if len(engine.registry.chunks) < 2:
        raise HTTPException(400, "Upload or load at least two text records before running retrieval evaluation.")
    n = max(5, min(100, int(payload.get("n", 30))))
    top_k = max(1, min(10, int(payload.get("top_k", 5))))
    start = time.perf_counter()
    cases = engine.generate_test_cases(n=n, seed=17, include_hindi=False)
    if not cases:
        raise HTTPException(400, "Could not generate evaluation questions from this corpus. Add more readable text.")
    metrics = engine.evaluate_retrieval(cases, top_k=top_k)
    wall = time.perf_counter() - start
    report = {"dataset_name": LAST_IMPORTS[-1].get("dataset_name", "current workspace") if LAST_IMPORTS else "current workspace",
              "cases_generated": len(cases), "top_k": top_k, "wall_clock_seconds": round(wall, 4),
              "retrieval_metrics": metrics,
              "methodology": "Questions are generated from the current corpus; treat scores as an in-corpus smoke test, not a held-out benchmark. For a publishable result, create independent, human-labelled query/supporting-chunk pairs.",
              "crosslingual": "Not included in this run. Use the dedicated Hindi-English benchmark after enabling and verifying multilingual model weights."}
    out_dir = ROOT / "examples" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "dashboard_retrieval_benchmark_latest.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


@app.get("/api/reports/{filename}")
def report_file(request: Request, filename: str):
    _need_auth(request)
    allowed = {"dashboard_retrieval_benchmark_latest.json", "unified_pipeline_600.json", "muj_corpus_benchmark.json",
               "hindi_english_retrieval_report.json", "injection_benchmark_report.json", "domain_shift_benchmark_report.json"}
    if filename not in allowed:
        raise HTTPException(404, "Report not found.")
    path = ROOT / "examples" / "outputs" / filename
    if not path.is_file():
        raise HTTPException(404, "Report has not been generated yet.")
    return FileResponse(path, media_type="application/json", filename=filename)


@app.get("/api/health")
def health():
    return {"ok": True, "app": "ReliaGraph local prototype", "version": app.version,
            "mode": "local-only prototype; bind to loopback"}
