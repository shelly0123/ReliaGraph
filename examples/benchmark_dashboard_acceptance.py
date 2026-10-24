"""End-to-end dashboard acceptance smoke test with isolated temporary data.

Run: python -m examples.benchmark_dashboard_acceptance
The report distinguishes synthetic fixtures from external Kaggle data which is not bundled.
"""
from __future__ import annotations

import io
import json
import os
import platform
import tempfile
import time
import zipfile
from pathlib import Path

_tmp = tempfile.mkdtemp(prefix="topicweave-acceptance-")
os.environ["TOPICWEAVE_DATA_DIR"] = _tmp

from fastapi.testclient import TestClient  # noqa: E402
from webapp.main import app, engine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def timed_request(fn):
    start = time.perf_counter()
    response = fn()
    return response, round(time.perf_counter() - start, 5)


def main() -> int:
    report = {
        "title": "ReliaGraph dashboard acceptance smoke test",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "mode": "deterministic baseline; optional ML models disabled",
        "data_dir": "temporary isolated directory (removed by the OS later)",
        "external_kaggle_data": "Full Kaggle datasets were not downloaded because external DNS resolution failed. One clearly attributed 20-row excerpt was transcribed from a public CSV mirror and imported through the dashboard API.",
        "synthetic_fixtures_are_kaggle_data": False,
        "kaggle_full_dataset_downloaded": False,
        "operations": [],
        "queries": [],
        "security_checks": {},
    }
    client = TestClient(app)
    try:
        unauth, unauth_s = timed_request(lambda: client.get("/api/dashboard"))
        report["security_checks"]["unauthenticated_dashboard_blocked"] = unauth.status_code == 401
        report["security_checks"]["unauthenticated_dashboard_http_seconds"] = unauth_s
        setup, setup_s = timed_request(lambda: client.post("/api/auth/setup", json={
            "username": "acceptance_researcher", "password": "acceptance-password-2026"
        }))
        if setup.status_code != 200:
            raise RuntimeError(f"Test account setup failed: {setup.status_code} {setup.text}")
        headers = {"x-csrf-token": setup.json()["csrf_token"]}
        report["security_checks"]["first_run_account_setup"] = setup.status_code == 200
        report["security_checks"]["setup_http_seconds"] = setup_s

        for dataset in ("synthetic-cyber", "synthetic-biomed", "synthetic-energy"):
            resp, request_s = timed_request(lambda ds=dataset: client.post(
                "/api/benchmarks/load-demo", json={"dataset": ds}, headers=headers))
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to load {dataset}: {resp.status_code} {resp.text}")
            m = resp.json()["manifest"]
            report["operations"].append({
                "operation": "load_synthetic_fixture", "dataset": dataset,
                "dataset_origin": "original synthetic project fixture, not Kaggle",
                "records": m["prepared_record_count"],
                "pipeline_wall_seconds": m["wall_clock_seconds"],
                "api_request_seconds": request_s,
                "nodes": m["graph_counts"]["nodes"], "edges": m["graph_counts"]["edges"],
                "topics": m["graph_counts"]["topics"], "under_60_seconds": m["under_60_seconds"],
            })

        # Load a small, attributed excerpt of real Kaggle sustainable-energy rows through the dashboard's bundled-sample control.
        kaggle_resp, kaggle_request_s = timed_request(lambda: client.post(
            "/api/benchmarks/load-demo", json={"dataset": "kaggle-energy-sample"}, headers=headers))
        if kaggle_resp.status_code != 200:
            raise RuntimeError(f"Kaggle-derived sample load failed: {kaggle_resp.status_code} {kaggle_resp.text}")
        km = kaggle_resp.json()["manifest"]
        report["operations"].append({
            "operation": "load_kaggle_derived_energy_excerpt",
            "dataset_origin": "20 selected rows/four columns transcribed from the public CSV mirror of the Kaggle Global Data on Sustainable Energy dataset; CC BY 4.0; not the full file",
            "source_url": "https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy",
            "source_mirror_url": "https://github.com/AnshTanwar/Global-Data-on-Sustainable-Energy",
            "license": "CC BY 4.0; excerpt adapted by selecting rows and columns",
            "records": km["prepared_record_count"], "pipeline_wall_seconds": km["wall_clock_seconds"],
            "api_request_seconds": kaggle_request_s, "nodes": km["graph_counts"]["nodes"],
            "edges": km["graph_counts"]["edges"], "topics": km["graph_counts"]["topics"],
            "under_60_seconds": km["under_60_seconds"], "source_attribution": km["source_attribution"]
        })

        # Upload the full 600-record synthetic scale fixture through the dashboard's own file API.
        scale_file = ROOT / "examples" / "data" / "synthetic_scale_600.jsonl"
        scale_bytes = scale_file.read_bytes()
        scale_resp, scale_request_s = timed_request(lambda: client.post(
            "/api/documents/upload", headers=headers,
            data={"dataset_name": "Synthetic 600-record dashboard scale"},
            files={"files": ("synthetic_scale_600.jsonl", scale_bytes, "application/x-ndjson")}))
        if scale_resp.status_code != 200:
            raise RuntimeError(f"600-record dashboard import failed: {scale_resp.status_code} {scale_resp.text}")
        sm = scale_resp.json()["manifest"]
        report["operations"].append({
            "operation": "upload_600_record_scale_fixture_via_dashboard_api",
            "dataset_origin": "original synthetic scale fixture; not Kaggle and not used for accuracy claims",
            "records": sm["prepared_record_count"], "pipeline_wall_seconds": sm["wall_clock_seconds"],
            "api_request_seconds": scale_request_s, "nodes": sm["graph_counts"]["nodes"],
            "edges": sm["graph_counts"]["edges"], "topics": sm["graph_counts"]["topics"],
            "under_60_seconds": sm["under_60_seconds"],
        })

        d, dashboard_s = timed_request(lambda: client.get("/api/dashboard"))
        dash = d.json()
        report["workspace_after_fixtures"] = {
            "api_request_seconds": dashboard_s,
            "document_records": len(dash["documents"]),
            "nodes": len(dash["snapshot"]["nodes"]),
            "edges": len(dash["snapshot"]["edges"]),
            "topics": len(dash["snapshot"]["topics"]),
        }

        for query in (
            "What are the key themes across the documents?",
            "What is access control?",
            "How are identity and security logging related?",
        ):
            resp, request_s = timed_request(lambda q=query: client.post(
                "/api/query", json={"query": q, "top_k": 5}, headers=headers))
            body = resp.json()
            report["queries"].append({
                "query": query, "http_status": resp.status_code,
                "api_request_seconds": request_s,
                "engine_latency_seconds": body.get("diagnostics", {}).get("latency_seconds"),
                "route": body.get("route", {}).get("route"),
                "hits": len(body.get("hits", [])),
                "topic_summaries": len(body.get("topic_overview", [])),
                "graph_paths": len(body.get("graph_paths", [])),
                "abstained": body.get("diagnostics", {}).get("abstained"),
            })

        csv_v1 = ("id,title,abstract,category\n"
                  "R1,Research memory,Memory systems retain source evidence for later retrieval,AI\n"
                  "R2,Graph retrieval,Knowledge graphs connect concepts through source-linked relationships,AI\n"
                  "R3,Version control,Document versioning keeps old evidence traceable,Data\n").encode()
        resp, request_s = timed_request(lambda: client.post(
            "/api/documents/upload", headers=headers,
            data={"dataset_name": "Acceptance CSV", "source_url": "https://www.kaggle.com/datasets/Cornell-University/arxiv"},
            files={"files": ("acceptance_research.csv", csv_v1, "text/csv")}))
        if resp.status_code != 200:
            raise RuntimeError(f"CSV upload failed: {resp.status_code} {resp.text}")
        m1 = resp.json()["manifest"]
        report["operations"].append({"operation": "upload_kaggle_shaped_csv", "dataset_origin": "project-authored CSV fixture with Kaggle URL metadata; not a downloaded Kaggle file",
                                     "records": m1["prepared_record_count"], "pipeline_wall_seconds": m1["wall_clock_seconds"],
                                     "api_request_seconds": request_s, "under_60_seconds": m1["under_60_seconds"]})

        csv_v2 = ("id,title,abstract,category\n"
                  "R1,Research memory,Memory systems retain source evidence for later retrieval,AI\n"
                  "R2,Graph retrieval,Knowledge graphs connect concepts through source-linked relationships,AI\n").encode()
        resp, request_s = timed_request(lambda: client.post(
            "/api/documents/upload", headers=headers,
            data={"dataset_name": "Acceptance CSV", "source_url": "https://www.kaggle.com/datasets/Cornell-University/arxiv"},
            files={"files": ("acceptance_research.csv", csv_v2, "text/csv")}))
        if resp.status_code != 200:
            raise RuntimeError(f"CSV re-import failed: {resp.status_code} {resp.text}")
        m2 = resp.json()["manifest"]
        report["operations"].append({"operation": "reimport_shorter_csv_edit", "records": m2["prepared_record_count"],
                                     "stale_records_removed": m2["stale_records_removed"], "pipeline_wall_seconds": m2["wall_clock_seconds"],
                                     "api_request_seconds": request_s, "under_60_seconds": m2["under_60_seconds"]})

        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("papers.jsonl", '{"id":"P1","title":"Graph evidence","abstract":"Graph edges cite source chunks."}\n'
                                         '{"id":"P2","title":"Topic discovery","abstract":"Topic clusters organize related records."}\n')
            zf.writestr("notes.txt", "Source evidence should be inspectable.")
        resp, request_s = timed_request(lambda: client.post(
            "/api/documents/upload", headers=headers,
            data={"dataset_name": "Acceptance ZIP import"},
            files={"files": ("acceptance_bundle.zip", zbuf.getvalue(), "application/zip")}))
        if resp.status_code != 200:
            raise RuntimeError(f"ZIP upload failed: {resp.status_code} {resp.text}")
        zm = resp.json()["manifest"]
        report["operations"].append({"operation": "upload_zip_jsonl_and_text", "records": zm["prepared_record_count"],
                                     "pipeline_wall_seconds": zm["wall_clock_seconds"], "api_request_seconds": request_s,
                                     "under_60_seconds": zm["under_60_seconds"]})

        # Exercise PDF extraction plus Markdown, JSON, and NDJSON uploads through the same API.
        pdf_path = ROOT / "examples" / "dashboard_datasets" / "synthetic-lab-notes.pdf"
        multi_files = [
            ("synthetic-lab-notes.pdf", pdf_path.read_bytes(), "application/pdf"),
            ("operator-notes.md", b"# Synthetic operator notes\nThe audit report preserves source provenance after updates.", "text/markdown"),
            ("research-records.json", json.dumps({"records": [
                {"id": "J1", "title": "Evidence path", "abstract": "A graph path links a query to source evidence."},
                {"id": "J2", "title": "Source update", "abstract": "A source update replaces only its own claim contribution."}
            ]}).encode(), "application/json"),
            ("events.ndjson", b'{"id":"N1","title":"Safe import","text":"The importer rejects unsafe file types."}\n', "application/x-ndjson"),
        ]
        resp, request_s = timed_request(lambda: client.post(
            "/api/documents/upload", headers=headers, data={"dataset_name": "Acceptance multi-format files"},
            files=[("files", (name, body, mime)) for name, body, mime in multi_files]))
        if resp.status_code != 200:
            raise RuntimeError(f"Multi-format upload failed: {resp.status_code} {resp.text}")
        mm = resp.json()["manifest"]
        report["operations"].append({"operation": "upload_pdf_markdown_json_ndjson",
                                     "dataset_origin": "bundled original synthetic sample files; not Kaggle",
                                     "file_count": len(multi_files), "records": mm["prepared_record_count"],
                                     "file_manifest": mm["file_manifest"], "pipeline_wall_seconds": mm["wall_clock_seconds"],
                                     "api_request_seconds": request_s, "under_60_seconds": mm["under_60_seconds"]})

        bench, bench_s = timed_request(lambda: client.post("/api/benchmarks/run", headers=headers,
                                                           json={"n": 10, "top_k": 5}))
        if bench.status_code != 200:
            raise RuntimeError(f"Retrieval smoke benchmark failed: {bench.status_code} {bench.text}")
        report["retrieval_smoke_test"] = {**bench.json(), "api_request_seconds": bench_s,
                                          "methodology_limit": "In-corpus generated questions; not held-out evaluation and not cross-domain proof."}
        catalog = client.get("/api/benchmarks/catalog").json()["datasets"]
        report["dataset_catalog"] = [{"id": d["id"], "name": d["name"], "status": d["status"], "kind": d["kind"], "license": d.get("license", "")} for d in catalog]

        csrf_blocked = client.post("/api/benchmarks/load-demo", json={"dataset": "synthetic-energy"}).status_code == 403
        report["security_checks"]["missing_csrf_blocked_mutation"] = csrf_blocked
        logout = client.post("/api/auth/logout", headers=headers, json={})
        report["security_checks"]["logout_revokes_session"] = logout.status_code == 200 and client.get("/api/dashboard").status_code == 401
        report["summary"] = {
            "synthetic_domains_loaded": 3,
            "600_record_dashboard_import_passed": sm["prepared_record_count"] >= 600 and sm["under_60_seconds"],
            "formats_tested": ["PDF", "TXT", "Markdown", "CSV", "JSON", "JSONL/NDJSON", "ZIP"],
            "query_routes_tested": sorted({q.get("route") for q in report["queries"]}),
            "all_record_update_operations_under_60_seconds": all(x.get("under_60_seconds", True) for x in report["operations"]),
            "kaggle_derived_excerpt_ingested": any(x.get("operation") == "load_kaggle_derived_energy_excerpt" for x in report["operations"]),
            "kaggle_full_dataset_downloaded_or_ingested": False,
            "multilingual_model_weights_loaded": False,
            "browser_automation": "Chromium navigation was blocked by the execution environment policy; frontend JavaScript syntax and the same web API endpoints were tested independently.",
        }
        out = ROOT / "examples" / "outputs" / "dashboard_acceptance_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"report": str(out), "summary": report["summary"], "operations": report["operations"],
                          "queries": report["queries"], "workspace": report["workspace_after_fixtures"]}, indent=2, ensure_ascii=False))
        return 0
    finally:
        client.close()
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
