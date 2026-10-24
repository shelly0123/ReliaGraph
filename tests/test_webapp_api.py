"""End-to-end API smoke test for the local dashboard and ingestion pipeline."""
from __future__ import annotations

import io
import os
import tempfile
import zipfile
from pathlib import Path

# Isolate the web app's SQLite session/auth DB and ReliaGraph state from the developer's data.
_TEST_DATA_DIR = tempfile.mkdtemp(prefix="topicweave-webtest-")
os.environ["TOPICWEAVE_DATA_DIR"] = _TEST_DATA_DIR

from fastapi.testclient import TestClient  # noqa: E402
from webapp.main import app, engine  # noqa: E402


def test_authenticated_dashboard_upload_update_query_benchmark_and_delete():
    client = TestClient(app)

    # Protected APIs are not accessible before the first-run account is created.
    assert client.get("/api/dashboard").status_code == 401
    status = client.get("/api/auth/status")
    assert status.status_code == 200 and status.json()["setup_required"] is True

    # The public sign-up UI uses the general signup endpoint even for the very first account.
    setup = client.post("/api/auth/signup", json={
        "username": "demo_researcher",
        "password": "a-long-demo-password-2026",
    })
    assert setup.status_code == 200
    csrf = setup.json()["csrf_token"]
    headers = {"x-csrf-token": csrf}

    # UI static document is served by the same application.
    page = client.get("/")
    assert page.status_code == 200 and "Knowledge landscape" in page.text
    assert 'data-auth-mode="login"' in page.text and 'data-auth-mode="signup"' in page.text
    assert client.get("/static/app.js").status_code == 200

    # Load three different cross-domain fixtures through the dashboard API.
    loaded = {}
    for dataset in ("synthetic-cyber", "synthetic-biomed", "synthetic-energy"):
        response = client.post("/api/benchmarks/load-demo", json={"dataset": dataset}, headers=headers)
        assert response.status_code == 200, response.text
        manifest = response.json()["manifest"]
        assert manifest["prepared_record_count"] >= 10
        assert manifest["wall_clock_seconds"] < 60
        loaded[dataset] = manifest

    dashboard = client.get("/api/dashboard").json()
    assert len(dashboard["documents"]) >= 30
    assert dashboard["snapshot"]["nodes"]
    assert dashboard["snapshot"]["edges"]
    assert dashboard["snapshot"]["topics"]

    # A broad question is routed to topic augmentation; a fact query returns source evidence.
    thematic = client.post("/api/query", json={"query": "What are the key themes across the documents?", "top_k": 5}, headers=headers)
    assert thematic.status_code == 200
    assert thematic.json()["route"]["route"] == "thematic"
    assert thematic.json()["topic_overview"]
    factual = client.post("/api/query", json={"query": "What is access control?", "top_k": 5}, headers=headers)
    assert factual.status_code == 200 and factual.json()["hits"]
    assert "Source chunk:" in factual.json()["answer"] or factual.json()["evidence_context"]

    # Test an ordinary Kaggle-shaped CSV import without claiming this fixture came from Kaggle.
    csv_v1 = ("id,title,abstract,category\n"
              "R1,Research memory,Memory systems retain source evidence for later retrieval,AI\n"
              "R2,Graph retrieval,Knowledge graphs connect concepts through source-linked relationships,AI\n"
              "R3,Version control,Document versioning keeps old evidence traceable,Data\n").encode()
    uploaded = client.post("/api/documents/upload", headers=headers,
                           data={"dataset_name": "Tiny research import", "source_url": "https://www.kaggle.com/datasets/Cornell-University/arxiv"},
                           files={"files": ("tiny_research.csv", csv_v1, "text/csv")})
    assert uploaded.status_code == 200, uploaded.text
    v1 = uploaded.json()["manifest"]
    assert v1["prepared_record_count"] == 3 and v1["under_60_seconds"]
    assert v1["graph_counts"]["nodes"] > 0

    # Re-importing the same named dataset with fewer rows must remove stale row records.
    csv_v2 = ("id,title,abstract,category\n"
              "R1,Research memory,Memory systems retain source evidence for later retrieval,AI\n"
              "R2,Graph retrieval,Knowledge graphs connect concepts through source-linked relationships,AI\n").encode()
    updated = client.post("/api/documents/upload", headers=headers,
                          data={"dataset_name": "Tiny research import", "source_url": "https://www.kaggle.com/datasets/Cornell-University/arxiv"},
                          files={"files": ("tiny_research.csv", csv_v2, "text/csv")})
    assert updated.status_code == 200, updated.text
    v2 = updated.json()["manifest"]
    assert v2["prepared_record_count"] == 2
    assert v2["stale_records_removed"] == 1
    assert v2["under_60_seconds"]
    assert not any(d["filename"] == "tiny_research_row_00003.txt" for d in client.get("/api/dashboard").json()["documents"])

    # ZIP with JSONL + text works without extracting arbitrary paths to disk.
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dataset/papers.jsonl", '{"id":"P1","title":"Topic graphs","abstract":"Topics connect source-backed knowledge."}\n'
                                             '{"id":"P2","title":"Evidence paths","abstract":"Evidence paths explain answer provenance."}\n')
        zf.writestr("notes/guide.txt", "Document updates should preserve the origin of every claim.")
        zf.writestr("../../ignored.exe", b"not allowed")
    zipped = client.post("/api/documents/upload", headers=headers,
                         data={"dataset_name": "Zip multi-format import"},
                         files={"files": ("multiformat.zip", zip_buffer.getvalue(), "application/zip")})
    assert zipped.status_code == 200, zipped.text
    assert zipped.json()["manifest"]["prepared_record_count"] == 3

    # Benchmark route and source catalogue both work after ingestion.
    catalog = client.get("/api/benchmarks/catalog")
    assert catalog.status_code == 200
    catalog_items = catalog.json()["datasets"]
    catalog_ids = {x["id"] for x in catalog_items}
    assert {"kaggle-arxiv", "kaggle-energy", "kaggle-energy-sample", "kaggle-delhi-climate", "synthetic-cyber"} <= catalog_ids
    # Bundled excerpt consists of actual values inspected from the CC BY 4.0 Kaggle dataset mirror.
    energy_sample = next(x for x in catalog_items if x["id"] == "kaggle-energy-sample")
    assert energy_sample["status"] == "Ready to load" and "CC BY 4.0" in energy_sample["license"]
    real_sample = client.post("/api/benchmarks/load-demo", json={"dataset": "kaggle-energy-sample"}, headers=headers)
    assert real_sample.status_code == 200, real_sample.text
    real_manifest = real_sample.json()["manifest"]
    assert real_manifest["prepared_record_count"] == 20
    assert "Kaggle-derived" in real_manifest["source_attribution"]
    assert "kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy" in real_manifest["source_attribution"]
    benchmark = client.post("/api/benchmarks/run", headers=headers, json={"n": 10, "top_k": 5})
    assert benchmark.status_code == 200, benchmark.text
    assert benchmark.json()["cases_generated"] >= 5
    assert benchmark.json()["wall_clock_seconds"] < 60

    # Delete a single document through the protected endpoint.
    docs = client.get("/api/dashboard").json()["documents"]
    target = next(d for d in docs if d["filename"] == "tiny_research_row_00002.txt")
    deleted = client.delete(f"/api/documents/{target['doc_id']}", headers=headers)
    assert deleted.status_code == 200 and deleted.json()["under_60_seconds"]

    # Logout revokes the server-side session.
    logout = client.post("/api/auth/logout", headers=headers, json={})
    assert logout.status_code == 200
    assert client.get("/api/dashboard").status_code == 401

    # Sign-up remains available after the first account exists, and a second account can log in.
    signup = client.post("/api/auth/signup", json={
        "username": "second_researcher",
        "password": "another-strong-password-2026",
    })
    assert signup.status_code == 200, signup.text
    assert signup.json()["username"] == "second_researcher"
    assert client.get("/api/dashboard").status_code == 200
    second_csrf = signup.json()["csrf_token"]
    assert client.post("/api/auth/logout", headers={"x-csrf-token": second_csrf}, json={}).status_code == 200
    assert client.post("/api/auth/login", json={
        "username": "second_researcher",
        "password": "another-strong-password-2026",
    }).status_code == 200
    client.close()
    engine.close()
