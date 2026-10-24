"""Reproducible storage/graph-update stress benchmark using 600 ORIGINAL synthetic records.

This measures incremental SQLite + NetworkX maintenance, not ML relation-extraction accuracy.
Use the Kaggle arXiv importer plus GLiNER batch extraction for a real-corpus throughput run.
"""
from __future__ import annotations
import json, tempfile, time, random
from pathlib import Path
from core.phase2.incremental_kb import IncrementalKnowledgeBase

DOMAINS = [
    ("software", ["service", "cache", "database", "queue", "API", "worker"], ["stores", "uses", "contains", "provides", "requires"]),
    ("ecology", ["wetland", "root system", "plant", "river", "microbe", "soil"], ["supports", "absorbs", "contains", "reduces", "requires"]),
    ("education", ["library", "course", "assessment", "student portal", "research centre", "curriculum"], ["provides", "contains", "supports", "requires", "records"]),
    ("science", ["sensor", "membrane", "experiment", "catalyst", "sample", "instrument"], ["measures", "protects", "produces", "contains", "affects"]),
    ("operations", ["workflow", "report", "team", "process", "registry", "schedule"], ["records", "requires", "contains", "updates", "supports"]),
    ("health_research", ["study", "cohort", "protocol", "survey", "dataset", "analysis"], ["uses", "measures", "includes", "reports", "supports"]),
]
OBJECTS = ["validated records", "recent observations", "research findings", "digital resources", "system events", "updated values", "quality checks", "student progress", "measured outcomes", "evidence summaries"]

def make_doc(i: int):
    domain, subjects, rels = DOMAINS[i % len(DOMAINS)]
    subj, rel, obj = subjects[(i*3) % len(subjects)], rels[(i*5) % len(rels)], OBJECTS[(i*7) % len(OBJECTS)]
    sentence = f"In the {domain} collection, the {subj} {rel} {obj}."
    claim = {"subject": f"the {subj}", "relation_type": rel, "object": obj, "evidence_text": sentence,
             "confidence": 0.7, "status": "candidate"}
    return {"doc_id": f"synthetic-{i:04d}", "title": f"Synthetic {domain} note {i:04d}", "text": sentence,
            "source_url": f"https://example.invalid/synthetic/{i:04d}", "claims": [claim], "domain": domain}

def main():
    docs = [make_doc(i) for i in range(600)]
    with tempfile.TemporaryDirectory(prefix="topicweave_scale_") as td:
        kb = IncrementalKnowledgeBase(Path(td)/"scale.sqlite3")
        t0 = time.perf_counter()
        for d in docs: kb.upsert_document(d["doc_id"], d["text"], title=d["title"], source_url=d["source_url"], claims=d["claims"])
        full_seconds = time.perf_counter()-t0
        before = kb.metrics()
        edit = make_doc(300); edit["text"] = edit["text"] + " The revised record also includes a follow-up quality check."
        edit["claims"] = edit["claims"] + [{"subject":"revised record", "relation_type":"includes", "object":"follow-up quality check",
                                               "evidence_text":"The revised record also includes a follow-up quality check.", "confidence":0.7}]
        t1=time.perf_counter(); edit_report=kb.upsert_document(edit["doc_id"], edit["text"], title=edit["title"], source_url=edit["source_url"], claims=edit["claims"]); edit_seconds=time.perf_counter()-t1
        t2=time.perf_counter(); delete_report=kb.delete_document("synthetic-0301"); delete_seconds=time.perf_counter()-t2
        report={"benchmark":"ReliaGraph synthetic 600-document graph-store stress test",
                "dataset_notice":"Original synthetic statements; this benchmark measures storage/update performance, not generalization or ML accuracy.",
                "initial_ingest_seconds":round(full_seconds,4), "initial_documents":before["documents"],
                "initial_claims":before["claims"], "initial_graph_nodes":before["graph_nodes"],
                "initial_graph_edges":before["graph_edges"], "estimated_graph_bytes":before["graph_bytes_estimate"],
                "edit_elapsed_seconds":round(edit_seconds,6), "delete_elapsed_seconds":round(delete_seconds,6),
                "edit_under_60_seconds":edit_seconds<60, "delete_under_60_seconds":delete_seconds<60,
                "edit_report":edit_report, "delete_report":delete_report,
                "edit_added_claims_detected": edit_report.get("added_claims", 0),
                "edit_rejected_claim_count": edit_report.get("rejected_claims", 0),
                "final_metrics":kb.metrics(),
                "qualification":"Run this script on the target laptop and use its own measurements in the presentation. It excludes model loading/inference and PDF parsing. The synthetic corpus is for storage/update speed only; use a separately labelled real sample for extraction accuracy."}
        kb.close()
        out=Path("examples/outputs/scale_600_report.json"); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps(report,indent=2))
if __name__ == "__main__": main()
