"""Run ReliaGraph's MUJ-themed Member 2 demo with timing and optional local ML.

Base mode is deterministic and needs no model downloads. Optional modes:
  python -m examples.run_muj_demo --use-embeddings
  python -m examples.run_muj_demo --use-nli
  python -m examples.run_muj_demo --use-embeddings --use-nli
"""
from __future__ import annotations
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from models import Chunk, Topic
from core.pattern_extractor import extract_candidates
from core.graph_builder import build_graph, export_graphml, to_networkx
from core.graph_audit import audit_graph, calculate_topic_coverage
from core.graph_versions import save_snapshot, compare_snapshots
from core.reporting import to_jsonable, summarize_findings
from core.model_runtime import RuntimeProfiler, LocalSentenceEmbedder, LocalNLIReviewer
from core.optional_model_audit import semantic_duplicate_candidates, review_edge_evidence_with_nli
from core.quality_metrics import relation_extraction_metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data" / "muj_paraphrased_corpus.json"
OUT = ROOT / "examples" / "outputs"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_corpus() -> tuple[dict[str, Any], list[Chunk], list[Topic]]:
    raw = json.loads(DATA.read_text(encoding="utf-8"))
    source_by_id = {s["source_id"]: s for s in raw["sources"]}
    chunks = []
    for fact in raw["facts"]:
        src = source_by_id[fact["source_id"]]
        chunks.append(Chunk(
            chunk_id=fact["chunk_id"], doc_id=fact["doc_id"], text=fact["text"],
            page=fact.get("page"), topic_ids=list(fact.get("topic_ids", [])),
            source_url=src["source_url"], source_title=src["source_title"],
            source_version=src["accessed_on"], source_kind="official_web_paraphrase",
            metadata={"paraphrased_summary": True, "source_id": fact["source_id"]},
        ))
    topics = [Topic(
        topic_id=t["topic_id"], label=t["label"], keywords=list(t["keywords"]),
        chunk_ids=[c.chunk_id for c in chunks if t["topic_id"] in c.topic_ids],
        parent_topic_id=t.get("parent_topic_id"),
    ) for t in raw["topics"]]
    return raw, chunks, topics


def build_synthetic_version_diff() -> dict[str, Any]:
    """A clearly labelled fictional notice solely to test version-diff code."""
    from models import Chunk, Topic
    from core.pattern_extractor import extract_candidates
    from core.graph_builder import build_graph
    from core.graph_versions import compare_snapshots
    import hashlib

    terms = ["fictional research showcase deadline", "10 October", "12 October"]
    topic = Topic("t-synthetic-demo", "Synthetic version-diff test", ["deadline", "version"], [])
    old_text = "The fictional research showcase deadline is 10 October."
    new_text = "The fictional research showcase deadline is 12 October."
    old_chunk = Chunk("synthetic-v1-c1", "synthetic-notice", old_text, 1, [topic.topic_id],
                      source_title="SYNTHETIC TEST ONLY — not a MUJ document", source_kind="synthetic_test")
    new_chunk = Chunk("synthetic-v2-c1", "synthetic-notice", new_text, 1, [topic.topic_id],
                      source_title="SYNTHETIC TEST ONLY — not a MUJ document", source_kind="synthetic_test")
    old_topic = Topic(topic.topic_id, topic.label, topic.keywords, [old_chunk.chunk_id])
    new_topic = Topic(topic.topic_id, topic.label, topic.keywords, [new_chunk.chunk_id])
    old_e, old_r = extract_candidates([old_chunk], terms)
    new_e, new_r = extract_candidates([new_chunk], terms)
    old = build_graph([old_chunk], [old_topic], old_e, old_r,
                      {"synthetic-notice": hashlib.sha256(old_text.encode()).hexdigest()})
    new = build_graph([new_chunk], [new_topic], new_e, new_r,
                      {"synthetic-notice": hashlib.sha256(new_text.encode()).hexdigest()})
    diff = compare_snapshots(old, new)
    result = to_jsonable(diff)
    result["notice"] = "SYNTHETIC TEST ONLY. The dates and notice are fictional and are not MUJ policy or an official university date. This exists only to test change detection."
    result["old_text"] = old_text
    result["new_text"] = new_text
    return result


def build_synthetic_conflict_audit() -> dict[str, Any]:
    """Create a clearly fictional conflicting-claim case to exercise the auditor."""
    from models import Chunk, Topic
    from core.pattern_extractor import extract_candidates
    from core.graph_builder import build_graph
    from core.graph_audit import audit_graph
    fictional_topic = Topic("t-synthetic-conflict", "Synthetic audit test", ["demo", "deadline"],
                            ["synthetic-conflict-a", "synthetic-conflict-b"])
    chunks = [
        Chunk("synthetic-conflict-a", "synthetic-demo-only", "The fictional research showcase deadline is 10 October.",
              1, [fictional_topic.topic_id], source_title="SYNTHETIC TEST ONLY — not a MUJ document", source_kind="synthetic_test"),
        Chunk("synthetic-conflict-b", "synthetic-demo-only", "The fictional research showcase deadline is 12 October.",
              2, [fictional_topic.topic_id], source_title="SYNTHETIC TEST ONLY — not a MUJ document", source_kind="synthetic_test"),
    ]
    terms = ["fictional research showcase deadline", "10 October", "12 October"]
    entities, relations = extract_candidates(chunks, terms)
    topic = Topic(fictional_topic.topic_id, fictional_topic.label, fictional_topic.keywords,
                  [c.chunk_id for c in chunks])
    snap = build_graph(chunks, [topic], entities, relations)
    findings = audit_graph(snap, chunks)
    return {
        "notice": "SYNTHETIC TEST ONLY. Fictional deadlines are not MUJ policy, events or dates. This scenario tests whether the auditor flags two different values for one deadline relation.",
        "snapshot_id": snap.snapshot_id,
        "findings": to_jsonable(findings),
        "finding_summary": summarize_findings(findings),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-embeddings", action="store_true", help="Enable optional local semantic duplicate suggestions; first run may download model weights.")
    parser.add_argument("--use-nli", action="store_true", help="Enable optional local NLI evidence review; first run may download model weights.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--nli-max-edges", type=int, default=40)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    profiler = RuntimeProfiler()
    raw, chunks, topics = load_corpus()
    known_terms = raw["known_terms"]
    with profiler.measure("pattern_extraction", len(chunks), "Known-term and relation-cue extraction"):
        entities, relations = extract_candidates(chunks, known_terms)
    # Hash the complete ordered content of each document, not an incremental hash chain.
    texts_by_doc: dict[str, list[str]] = {}
    for chunk in chunks:
        texts_by_doc.setdefault(chunk.doc_id, []).append(chunk.text)
    document_hashes = {doc_id: sha("\n".join(texts)) for doc_id, texts in sorted(texts_by_doc.items())}
    with profiler.measure("graph_construction", len(relations), "Build graph, topic links and evidence metadata"):
        snapshot = build_graph(chunks, topics, entities, relations, document_hashes)
    snapshot.metadata.update({
        "dataset_id": raw["dataset_id"],
        "dataset_title": raw["dataset_title"],
        "dataset_notice": raw["notice"],
        "source_ids": [source["source_id"] for source in raw["sources"]],
        "source_data_style": "independently_paraphrased_factual_summaries_with_source_urls",
    })
    with profiler.measure("structural_audit", len(snapshot.edges), "Deterministic graph checks"):
        findings = audit_graph(snapshot, chunks)
    with profiler.measure("topic_coverage", len(topics), "Graph-based structural coverage counts"):
        coverage = calculate_topic_coverage(snapshot, chunks)
    predicted_triples = [{"subject": r.subject, "relation_type": r.relation_type, "object": r.object} for r in relations]
    # Negative, deliberately labelled test text is separate from the MUJ fact graph.
    # It checks that mere co-occurrence of concepts does not automatically create edges.
    negative_chunks = [Chunk(
        chunk_id=test["test_id"], doc_id="synthetic-negative-benchmark", text=test["text"],
        topic_ids=[], source_kind="synthetic_benchmark_only",
        source_title="Synthetic extraction test — not MUJ information",
    ) for test in raw.get("negative_test_contexts", [])]
    with profiler.measure("negative_relation_tests", len(negative_chunks), "Synthetic no-relation controls"):
        _, negative_relations = extract_candidates(negative_chunks, known_terms)
    predicted_triples.extend({"subject": r.subject, "relation_type": r.relation_type, "object": r.object} for r in negative_relations)
    with profiler.measure("gold_set_metrics", len(raw["gold_edges"]), "Curated positive facts plus negative co-occurrence controls"):
        edge_metrics = relation_extraction_metrics(predicted_triples, raw["gold_edges"])
    edge_metrics["negative_control_cases"] = len(negative_chunks)
    edge_metrics["relations_emitted_from_negative_controls"] = len(negative_relations)
    edge_metrics["negative_control_note"] = "These are deliberately written no-relation examples; they are a smoke test, not an independent benchmark."

    model_outputs: dict[str, Any] = {
        "optional_models_enabled": {"embeddings": args.use_embeddings, "nli": args.use_nli},
        "warnings": [],
    }
    if args.use_embeddings:
        embedder = LocalSentenceEmbedder(device=args.device, cache_db=ROOT / "data" / "cache" / "muj_embeddings.sqlite3")
        try:
            with profiler.measure("semantic_duplicate_suggestions", len(snapshot.nodes), "Optional local embedding model"):
                semantic_findings = semantic_duplicate_candidates(snapshot, embedder, threshold=0.88)
            findings += semantic_findings
            model_outputs["embedding_model"] = embedder.diagnostics()
            model_outputs["semantic_duplicate_suggestion_count"] = len(semantic_findings)
        except Exception as exc:
            warning = f"Embedding mode unavailable; deterministic baseline was kept. {type(exc).__name__}: {exc}"
            model_outputs["warnings"].append(warning)
            profiler.add("semantic_duplicate_suggestions_skipped", 0.0, len(snapshot.nodes), warning)
    if args.use_nli:
        reviewer = LocalNLIReviewer(device=args.device)
        try:
            with profiler.measure("nli_evidence_review", min(len(snapshot.edges), args.nli_max_edges), "Optional local cross-encoder; advisory only"):
                nli_results = review_edge_evidence_with_nli(snapshot, chunks, reviewer, max_edges=args.nli_max_edges)
            findings += nli_results["findings"]
            model_outputs["nli_model"] = nli_results["model"]
            model_outputs["nli_check_count"] = len(nli_results["checks"])
            (OUT / "muj_nli_checks.json").write_text(json.dumps(to_jsonable(nli_results["checks"]), indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            warning = f"NLI mode unavailable; deterministic baseline was kept. {type(exc).__name__}: {exc}"
            model_outputs["warnings"].append(warning)
            profiler.add("nli_evidence_review_skipped", 0.0, min(len(snapshot.edges), args.nli_max_edges), warning)

    with profiler.measure("graph_export", len(snapshot.edges), "GraphML export"):
        export_graphml(snapshot, str(OUT / "muj_topicweave_graph.graphml"))
    save_snapshot(snapshot, OUT / "muj_graph_snapshot.json")
    (OUT / "muj_source_manifest.json").write_text(json.dumps({"notice": raw["notice"], "sources": raw["sources"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "muj_audit_findings.json").write_text(json.dumps(to_jsonable(findings), indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "muj_topic_coverage.json").write_text(json.dumps(to_jsonable(coverage), indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "muj_relation_metrics.json").write_text(json.dumps(edge_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "muj_runtime_report.json").write_text(json.dumps(profiler.report(), indent=2), encoding="utf-8")
    (OUT / "muj_model_diagnostics.json").write_text(json.dumps(model_outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "synthetic_version_diff.json").write_text(json.dumps(build_synthetic_version_diff(), indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "synthetic_audit_scenario.json").write_text(json.dumps(build_synthetic_conflict_audit(), indent=2, ensure_ascii=False), encoding="utf-8")

    graph = to_networkx(snapshot)
    print("ReliaGraph MUJ-themed Member 2 demo complete")
    print(f"Corpus: {raw['dataset_title']}")
    print(f"Chunks: {len(chunks)} | topics: {len(topics)} | entity candidates: {len(entities)} | relation candidates: {len(relations)}")
    print(f"Graph: {graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges")
    print(f"Audit findings: {json.dumps(summarize_findings(findings), ensure_ascii=False)}")
    print(f"Curated relation smoke-test: precision={edge_metrics['precision']:.3f}, recall={edge_metrics['recall']:.3f}, F1={edge_metrics['f1']:.3f}; negative-control relations={edge_metrics['relations_emitted_from_negative_controls']}")
    print(f"Measured time (seconds): {profiler.report()['total_measured_seconds']:.3f}")
    print("Runtime and model diagnostics saved. Optional model loading/inference time is reported separately when enabled.")
    for warning in model_outputs.get("warnings", []):
        print(f"OPTIONAL MODEL WARNING: {warning}")
    print("MUJ fact statements are paraphrased summaries with source URLs; synthetic version diff is clearly labelled and not an official policy.")
    print(f"Outputs: {OUT}")

if __name__ == "__main__":
    main()
