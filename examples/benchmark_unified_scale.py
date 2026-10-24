"""Full unified-pipeline benchmark on 600 original synthetic text records.

Measures ingestion + topics + retrieval index + graph updates + audit/snapshot reports,
then one edit and one deletion. It does NOT measure real-world extraction accuracy and
excludes optional model downloads/inference; use the same script with actual files to
measure those separately.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import time

from pipeline import TopicWeaveEngine

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data" / "synthetic_scale_600.jsonl"
OUT = ROOT / "examples" / "outputs" / "unified_pipeline_600.json"


def main() -> None:
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 500:
        raise SystemExit(f"Need >=500 rows; found {len(rows)} in {DATA}")
    files = [(f"{row['doc_id']}.txt", str(row["text"]).encode("utf-8")) for row in rows]
    metadata = {name: {"source_title": name, "source_kind": "original_synthetic_performance_fixture"}
                for name, _ in files}

    with TemporaryDirectory(prefix="topicweave_unified_scale_") as td:
        with TopicWeaveEngine(td, enable_topic_embeddings=False, enable_dense_retrieval=False,
                              enable_crosslingual=False, enable_gliner=False, enable_spacy=False) as engine:
            start = time.perf_counter()
            first = engine.upsert_files(files, source_metadata=metadata)
            total_batch = time.perf_counter() - start
            initial_metrics = engine.metrics()
            snapshot_one = first.snapshot.snapshot_id if first.snapshot else None

            edit_name, edit_body = files[min(250, len(files) - 1)]
            edit_text = edit_body.decode("utf-8") + " The revised record includes an additional verification step."
            start = time.perf_counter()
            edited = engine.upsert_files([(edit_name, edit_text.encode("utf-8"))],
                                         source_metadata={edit_name: metadata[edit_name]})
            edit_total = time.perf_counter() - start

            delete_name = files[min(251, len(files) - 1)][0]
            start = time.perf_counter()
            deleted = engine.delete_document(delete_name)
            delete_total = time.perf_counter() - start
            integrity = engine.graph_service.audit_log.verify()
            final_metrics = engine.metrics()

            report = {
                "benchmark": "ReliaGraph unified pipeline synthetic 600-document test",
                "dataset": {"path": "examples/data/synthetic_scale_600.jsonl", "records": len(rows),
                            "type": "original synthetic multi-domain records", "for_accuracy_claims": False},
                "configuration": {"topic_embeddings": False, "dense_retrieval": False,
                                   "crosslingual_model": False, "GLiNER": False, "spaCy_model": False,
                                   "paid_services": False},
                "initial_ingest_seconds_wall_clock": round(total_batch, 6),
                "initial_stage_timings": first.timings,
                "initial_graph": {"documents": initial_metrics["documents"], "chunks": initial_metrics["chunks"],
                                  "topics": initial_metrics["topics"], "nodes": initial_metrics["nodes"],
                                  "edges": initial_metrics["edges"],
                                  "estimated_graph_bytes": initial_metrics["graph_storage"].get("graph_bytes_estimate")},
                "edit_seconds_wall_clock": round(edit_total, 6),
                "edit_stage_timings": edited.timings,
                "edit_operation": edited.change_summary,
                "delete_seconds_wall_clock": round(delete_total, 6),
                "delete_stage_timings": deleted.timings,
                "delete_operation": deleted.change_summary,
                "under_60_seconds_for_full_batch": total_batch < 60,
                "edit_under_60_seconds": edit_total < 60,
                "delete_under_60_seconds": delete_total < 60,
                "audit_chain_full_verification": integrity,
                "final_counts": {"documents": final_metrics["documents"], "chunks": final_metrics["chunks"],
                                 "nodes": final_metrics["nodes"], "edges": final_metrics["edges"]},
                "snapshot_before_update": snapshot_one,
                "snapshot_after_update": edited.snapshot.snapshot_id if edited.snapshot else None,
                "limitations": [
                    "Original synthetic text only; do not use this to claim cross-domain relation precision.",
                    "No PDF parsing or ML inference in this run; report model load/inference separately.",
                    "Timings are host-specific and must be remeasured on the hackathon laptop.",
                    "Accurate extraction requires an independently human-labelled relation benchmark.",
                ],
            }
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
