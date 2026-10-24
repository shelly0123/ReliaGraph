"""Measure full per-document extraction+CRUD latency with the robust service.

This uses original synthetic documents for repeatable throughput testing. It is not a
replacement for the 500+ real-text corpus run. Use --enable-gliner to include model
inference on the changed document if the local model is already available.
"""
from __future__ import annotations
import argparse, json, time, tempfile
from pathlib import Path
from core.phase2.robust_service import TopicWeaveRobustService


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--documents",type=int,default=600)
    p.add_argument("--enable-gliner",action="store_true")
    p.add_argument("--enable-spacy",action="store_true")
    p.add_argument("--output",default="examples/outputs/full_update_latency_report.json")
    a=p.parse_args()
    temp=Path(tempfile.mkdtemp(prefix="topicweave-update-bench-"))
    service=TopicWeaveRobustService(temp/"kb.sqlite3",audit_log_path=temp/"audit.jsonl",
                                    enable_gliner=a.enable_gliner,enable_spacy=a.enable_spacy)
    rows=[]; t0=time.perf_counter()
    for i in range(a.documents):
        txt=(f"The document store uses a local database. Record {i} describes a process that produces a result. "
             f"The service supports traceable evidence for item {i}. This is an original synthetic benchmark document.")
        rows.append(service.upsert_document(f"synthetic-{i:04d}",txt,title=f"Synthetic source {i}"))
    total_ingest=time.perf_counter()-t0
    edit_text="The document store uses a local database. Record 300 describes a process that produces an updated result."
    t0=time.perf_counter(); edited=service.upsert_document("synthetic-0300",edit_text,title="Synthetic source 300 revised"); edit_total=time.perf_counter()-t0
    t0=time.perf_counter(); deleted=service.delete_document("synthetic-0301"); delete_total=time.perf_counter()-t0
    report={"benchmark":"full update latency through TopicWeaveRobustService",
        "dataset":"original synthetic multi-relation documents; runtime stress only, not accuracy benchmark",
        "documents_added":len(rows),"total_add_seconds":round(total_ingest,4),"average_add_seconds":round(total_ingest/max(1,len(rows)),6),
        "edit_end_to_end_seconds":round(edit_total,4),"delete_seconds":round(delete_total,4),
        "edit_under_60s":edit_total<60,"delete_under_60s":delete_total<60,
        "updated_doc_diagnostics":edited.get("extraction_diagnostics",{}),"post_update_metrics":service.metrics(),
        "limitations":"Run on the target laptop. This synthetic corpus does not establish full PDF parsing/model latency for real documents. Report model cold-load separately from per-document update latency."}
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))
    service.close()
    return 0

if __name__ == "__main__": raise SystemExit(main())
