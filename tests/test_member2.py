from models import Chunk, Topic, GraphSnapshot, Node, Edge
from core.pattern_extractor import extract_candidates
from core.graph_builder import build_graph, to_networkx, export_graphml
from core.graph_audit import audit_graph, calculate_topic_coverage
from core.graph_versions import compare_snapshots, save_snapshot, load_snapshot


def fixture():
    chunks = [
        Chunk("c1", "policy", "The scholarship requires a research internship.", 1, ["t1"]),
        Chunk("c2", "policy", "The scholarship does not require a research internship.", 2, ["t1"]),
        Chunk("c3", "policy", "The application deadline is 15 March.", 3, ["t1"]),
        Chunk("c4", "security", "Multi-factor authentication reduces account takeover risk.", 1, ["t2"]),
        Chunk("c5", "policy", "The scholarship requires academic eligibility.", 4, ["t1"]),
    ]
    topics = [
        Topic("t1", "Scholarship eligibility", ["scholarship", "internship", "deadline"], ["c1", "c2", "c3", "c5"]),
        Topic("t2", "Account security", ["authentication", "account security"], ["c4"]),
    ]
    terms = ["scholarship", "academic eligibility", "research internship", "application deadline", "15 March",
             "multi-factor authentication", "account takeover risk"]
    entities, relations = extract_candidates(chunks, terms)
    return chunks, topics, entities, relations


def test_pattern_extractor_captures_evidence_and_negation():
    chunks, _, entities, relations = fixture()
    assert entities
    assert any(r.relation_type == "requires" and r.chunk_id == "c1" for r in relations)
    assert any(r.relation_type == "does_not_require" and r.chunk_id == "c2" for r in relations)
    assert any(r.relation_type == "has_deadline" and r.object == "15 March" for r in relations)
    assert all(r.evidence_text and r.chunk_id for r in relations)


def test_graph_has_real_nodes_edges_and_valid_provenance():
    chunks, topics, entities, relations = fixture()
    snapshot = build_graph(chunks, topics, entities, relations, {"policy": "v1", "security": "v1"})
    node_ids = {n.node_id for n in snapshot.nodes}
    chunk_ids = {c.chunk_id for c in chunks}
    assert len(snapshot.nodes) >= len(topics) + 4
    assert snapshot.edges
    assert all(e.source_node_id in node_ids and e.target_node_id in node_ids for e in snapshot.edges)
    assert all(set(e.evidence_chunk_ids) <= chunk_ids for e in snapshot.edges)
    assert to_networkx(snapshot).number_of_edges() == len(snapshot.edges)


def test_auditor_flags_opposing_claims_and_functional_values():
    chunks, topics, entities, relations = fixture()
    chunks += [Chunk("c5", "faq", "The application deadline is 20 March.", 1, ["t1"])]
    more_entities, more_relations = extract_candidates(chunks[-1:], ["application deadline", "20 March"])
    snapshot = build_graph(chunks, topics, entities + more_entities, relations + more_relations)
    findings = audit_graph(snapshot, chunks)
    assert any(f.category == "possible_conflicting_claims" for f in findings)
    assert any(f.category == "possible_functional_value_conflict" for f in findings)


def test_bad_evidence_ref_is_flagged():
    chunks, topics, entities, relations = fixture()
    snapshot = build_graph(chunks, topics, entities, relations)
    edge = snapshot.edges[0]
    edge.evidence_chunk_ids = ["does-not-exist"]
    findings = audit_graph(snapshot, chunks)
    assert any(f.category == "invalid_evidence_reference" for f in findings)


def test_snapshot_round_trip_and_claim_diff(tmp_path):
    old_chunks = [Chunk("old-c", "policy", "The application deadline is 15 March.", 1, ["t1"])]
    new_chunks = [Chunk("new-c", "policy", "The application deadline is 20 March.", 1, ["t1"])]
    topics_old = [Topic("t1", "Applications", ["deadline"], ["old-c"])]
    topics_new = [Topic("t1", "Applications", ["deadline"], ["new-c"])]
    terms = ["application deadline", "15 March", "20 March"]
    e_old, r_old = extract_candidates(old_chunks, terms)
    e_new, r_new = extract_candidates(new_chunks, terms)
    old = build_graph(old_chunks, topics_old, e_old, r_old, {"policy": "hash1"})
    new = build_graph(new_chunks, topics_new, e_new, r_new, {"policy": "hash2"})
    path = tmp_path / "snapshot.json"
    save_snapshot(old, path)
    assert load_snapshot(path).snapshot_id == old.snapshot_id
    diff = compare_snapshots(old, new)
    assert "policy" in diff.changed_document_ids
    assert diff.changed_claims
    assert diff.changed_claims[0].old_object == "15 March"
    assert diff.changed_claims[0].new_object == "20 March"


def test_topic_coverage_counts_topics():
    chunks, topics, entities, relations = fixture()
    snapshot = build_graph(chunks, topics, entities, relations)
    rows = calculate_topic_coverage(snapshot, chunks)
    assert len(rows) == 2
    assert all(r.chunk_count >= 1 for r in rows)


def test_mapping_inputs_from_other_modules_are_accepted():
    chunks = [{"chunk_id": "c1", "doc_id": "d1", "text": "Password hashing protects account security.", "page": 1, "topic_ids": ["t1"]}]
    topics = [{"topic_id": "t1", "label": "Security", "keywords": ["password hashing"], "chunk_ids": ["c1"]}]
    entities, relations = extract_candidates(chunks, ["password hashing", "account security"])
    snapshot = build_graph(chunks, topics, entities, relations)
    assert snapshot.nodes and snapshot.topics[0].label == "Security"


def test_graphml_export_writes_file(tmp_path):
    chunks, topics, entities, relations = fixture()
    snapshot = build_graph(chunks, topics, entities, relations)
    path = tmp_path / "graph.graphml"
    export_graphml(snapshot, str(path))
    assert path.exists()
    assert "<graphml" in path.read_text(encoding="utf-8")
