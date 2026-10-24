"""Exercise the unified engine on the bundled paraphrased MUJ topic corpus.

This script deliberately passes text through the generic pipeline without supplying
MUJ-specific term lists or extraction rules. It evaluates exact triples against the
small bundled gold-edge list. This is a development smoke test, not an independent
or industry-wide benchmark; see docs/ROBUSTNESS_RESULTS.md.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import re
import time

from pipeline import TopicWeaveEngine

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data" / "muj_paraphrased_corpus.json"
OUT = ROOT / "examples" / "outputs" / "muj_corpus_benchmark.json"


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def main() -> None:
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    source_by_id = {str(s["source_id"]): s for s in raw["sources"]}

    # Keep each independently paraphrased fact as its own small input unit so
    # its source evidence can be evaluated without custom extraction rules.
    # This granularity is for this smoke corpus only; production PDF chunking
    # remains controlled by Member 1's ingestion/chunking module.
    files = []
    source_metadata = {}
    for fact in raw["facts"]:
        filename = f"{fact['chunk_id']}.txt"
        files.append((filename, fact["text"].encode("utf-8")))
        src = source_by_id[str(fact["source_id"])]
        source_metadata[filename] = {
            "source_title": src["source_title"],
            "source_url": src["source_url"],
            "source_version": src.get("accessed_on"),
            "source_kind": "official_web_paraphrase",
        }

    with TemporaryDirectory(prefix="topicweave_muj_smoke_") as temp_dir:
        with TopicWeaveEngine(
            temp_dir,
            enable_topic_embeddings=False,
            enable_dense_retrieval=False,
            enable_crosslingual=False,
            enable_gliner=False,
            enable_spacy=False,
        ) as engine:
            t0 = time.perf_counter()
            ingest_result = engine.upsert_files(files, source_metadata=source_metadata)
            wall = time.perf_counter() - t0
            metrics = engine.metrics()
            snapshot = ingest_result.snapshot
            node_labels = {n.node_id: n.label for n in snapshot.nodes}

            # Exclude derived topic membership links; score factual relationship
            # triples only. Match exact normalized labels, not semantic synonyms.
            predicted = set()
            predicted_rows = []
            for edge in snapshot.edges:
                if edge.status == "derived":
                    continue
                triple = (
                    norm(node_labels.get(edge.source_node_id, "")),
                    norm(edge.relation_type),
                    norm(node_labels.get(edge.target_node_id, "")),
                )
                predicted.add(triple)
                predicted_rows.append({
                    "subject": node_labels.get(edge.source_node_id, ""),
                    "relation": edge.relation_type,
                    "object": node_labels.get(edge.target_node_id, ""),
                    "evidence_chunk_ids": edge.evidence_chunk_ids,
                    "status": edge.status,
                })

            gold = {
                (norm(e["subject"]), norm(e["relation_type"]), norm(e["object"]))
                for e in raw.get("gold_edges", [])
            }
            tp_set, fp_set, fn_set = predicted & gold, predicted - gold, gold - predicted
            tp, fp, fn = len(tp_set), len(fp_set), len(fn_set)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            query_demos = []
            for question in (
                "What are the main themes across these documents?",
                "What types of programmes does Manipal University Jaipur offer?",
                "How are assessment and examinations connected?",
            ):
                result = engine.query(question, top_k=5)
                query_demos.append({
                    "question": question,
                    "route": result.get("route", {}).get("route"),
                    "answer": result.get("answer"),
                    "hit_count": len(result.get("hits", [])),
                    "graph_path_count": len(result.get("graph_paths", [])),
                    "source_chunk_ids": [h.get("chunk_id") for h in result.get("hits", [])[:5]],
                    "note": "Qualitative smoke outputs; not a human-rated answer-quality benchmark.",
                })

            report = {
                "benchmark": "ReliaGraph unified pipeline on paraphrased MUJ source-fact corpus",
                "notice": raw.get("notice", ""),
                "method": {
                    "algorithm_specific_muj_terms_supplied": False,
                    "optional_models_enabled": False,
                    "each_fact_as_input_record": True,
                    "source_provenance_preserved": True,
                    "gold_edge_match": "exact normalized subject-relation-object; semantic synonyms do not count as matches",
                },
                "dataset": {
                    "dataset_id": raw.get("dataset_id"),
                    "facts": len(raw.get("facts", [])),
                    "sources": len(raw.get("sources", [])),
                    "gold_relations": len(gold),
                    "type": "short, independently paraphrased public facts with a manually curated gold-edge list",
                    "independent_holdout": False,
                },
                "runtime": {
                    "wall_clock_seconds": round(wall, 6),
                    "stage_timings_seconds": ingest_result.timings,
                    "note": "Current development host; no model inference or full PDF parsing. Re-measure on target laptop.",
                },
                "graph": {
                    "documents": metrics.get("documents"),
                    "chunks": metrics.get("chunks"),
                    "topics": len(snapshot.topics),
                    "nodes": len(snapshot.nodes),
                    "edges_including_derived_topic_links": len(snapshot.edges),
                    "factual_relation_predictions": len(predicted),
                    "estimated_graph_bytes": metrics.get("graph_storage", {}).get("graph_bytes_estimate"),
                },
                "exact_relation_metrics": {
                    "tp": tp, "fp": fp, "fn": fn,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "true_positive_triples": [list(x) for x in sorted(tp_set)],
                    "false_positive_triples": [list(x) for x in sorted(fp_set)],
                    "false_negative_triples": [list(x) for x in sorted(fn_set)],
                    "caveat": "Development smoke-set score only. The corpus/labels were available during development, the set is small, and exact label matching treats paraphrase/synonym differences as errors. Do not call this held-out or industry-grade generalization.",
                },
                "query_demos": query_demos,
                "predicted_relations": predicted_rows,
                "limitations": [
                    "MUJ-themed corpus is a demo only; no official endorsement is implied.",
                    "Source facts are independently phrased summaries, not copied full webpage passages.",
                    "Small corpus is not sufficient to judge topic clustering quality statistically.",
                    "The same corpus has been used for development and is not a held-out test set.",
                    "Cross-lingual inference is not tested by this script.",
                ],
            }
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({
                "out": str(OUT),
                "documents": report["graph"]["documents"],
                "chunks": report["graph"]["chunks"],
                "topics": report["graph"]["topics"],
                "nodes": report["graph"]["nodes"],
                "edges": report["graph"]["edges_including_derived_topic_links"],
                "wall_clock_seconds": report["runtime"]["wall_clock_seconds"],
                "exact_relation_metrics": report["exact_relation_metrics"],
            }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
