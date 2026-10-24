import json
from pathlib import Path

from core.model_runtime import RuntimeProfiler, SQLiteVectorCache
from core.pattern_extractor import extract_candidates
from core.graph_builder import build_graph
from core.graph_audit import audit_graph, calculate_topic_coverage
from core.quality_metrics import relation_extraction_metrics
from examples.run_muj_demo import load_corpus, build_synthetic_version_diff, build_synthetic_conflict_audit


def test_muj_corpus_has_source_links_and_paraphrase_notice():
    raw, chunks, topics = load_corpus()
    assert "independently phrased" in raw["notice"].casefold()
    assert len(chunks) >= 15
    assert all(c.source_url and c.source_title for c in chunks)
    assert all(c.metadata.get("paraphrased_summary") for c in chunks)
    assert len(topics) >= 3


def test_muj_graph_edges_keep_evidence_and_official_source_urls():
    raw, chunks, topics = load_corpus()
    entities, relations = extract_candidates(chunks, raw["known_terms"])
    assert entities and relations
    snapshot = build_graph(chunks, topics, entities, relations)
    chunk_ids = {c.chunk_id for c in chunks}
    assert all(set(e.evidence_chunk_ids) <= chunk_ids for e in snapshot.edges if e.relation_type not in {"belongs_to_topic", "has_subtopic"})
    evidence_edges = [e for e in snapshot.edges if e.attributes.get("evidence_sources")]
    assert evidence_edges
    assert all(s.get("source_url", "").startswith("https://") for e in evidence_edges for s in e.attributes["evidence_sources"])


def test_muj_gold_edge_metrics_are_defined_and_bounded():
    raw, chunks, topics = load_corpus()
    _, relations = extract_candidates(chunks, raw["known_terms"])
    metrics = relation_extraction_metrics(relations, raw["gold_edges"])
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert metrics["gold_unique_triples"] == len(raw["gold_edges"])
    negative_chunks = [
        {"chunk_id": t["test_id"], "doc_id": "negative-test", "text": t["text"], "topic_ids": []}
        for t in raw.get("negative_test_contexts", [])
    ]
    _, negative_relations = extract_candidates(negative_chunks, raw["known_terms"])
    assert len(negative_relations) == 0


def test_synthetic_version_diff_is_clearly_marked_and_detected():
    diff = build_synthetic_version_diff()
    assert "SYNTHETIC TEST ONLY" in diff["notice"]
    assert diff["changed_claims"]
    claim = diff["changed_claims"][0]
    assert claim["old_object"] == "10 October"
    assert claim["new_object"] == "12 October"


def test_runtime_profiler_records_stages():
    profiler = RuntimeProfiler()
    with profiler.measure("unit_test", 3, "example"):
        sum(range(10))
    report = profiler.report()
    assert report["total_measured_seconds"] >= 0
    assert report["events"][0]["stage"] == "unit_test"
    assert report["events"][0]["item_count"] == 3


def test_sqlite_vector_cache_roundtrip(tmp_path):
    cache = SQLiteVectorCache(tmp_path / "vectors.sqlite3")
    cache.put_many("test-model", [("some sentence", [0.1, 0.2, 0.3])])
    found = cache.get_many("test-model", [" some   sentence "])
    assert len(found) == 1
    vector = next(iter(found.values()))
    assert vector == [0.1, 0.2, 0.3]


def test_auditor_and_coverage_run_on_muj_graph():
    raw, chunks, topics = load_corpus()
    entities, relations = extract_candidates(chunks, raw["known_terms"])
    snapshot = build_graph(chunks, topics, entities, relations)
    findings = audit_graph(snapshot, chunks)
    coverage = calculate_topic_coverage(snapshot, chunks)
    assert isinstance(findings, list)
    assert len(coverage) == len(topics)
    assert all(row.chunk_count >= 1 for row in coverage)


def test_synthetic_conflict_audit_flags_both_evidence_chunks():
    report = build_synthetic_conflict_audit()
    assert "SYNTHETIC TEST ONLY" in report["notice"]
    matches = [f for f in report["findings"] if f["category"] == "possible_functional_value_conflict"]
    assert matches
    assert set(matches[0]["evidence_chunk_ids"]) == {"synthetic-conflict-a", "synthetic-conflict-b"}
