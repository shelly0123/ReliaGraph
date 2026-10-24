from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path

from models import Chunk, Topic, Node, Edge, GraphSnapshot, RelationCandidate
from core.selective_quality import (
    adjudicate_candidates, choose_conservative_threshold, apply_calibrated_threshold,
    schema_shift_profile, precision_recall_f1,
)
from core.generalized_audit import audit_graph_general
from core.generalized_versioning import compare_snapshots_general
from core.domain_benchmark import evaluate_by_domain


def _chunk(chunk_id="c1", text="The service uses a local database.", doc_id="d1", source_url="https://example.org/a"):
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, text=text, source_url=source_url, metadata={})


def _rel(extractor="model_a", quote="The service uses a local database.", confidence=0.8, chunk_id="c1",
         subject="service", relation="uses", object_="local database"):
    return RelationCandidate(subject=subject, relation_type=relation, object=object_, chunk_id=chunk_id,
                             evidence_text=quote, confidence=confidence,
                             attributes={"extractor": extractor})


def test_source_gate_rejects_quote_not_found():
    chunks = [_chunk()]
    forged = _rel(quote="The service deletes all records.", subject="service", relation="deletes", object_="records")
    decisions = adjudicate_candidates([forged], chunks)
    assert len(decisions) == 1
    assert decisions[0].decision == "rejected_provenance"
    assert "quote_not_found_in_source_chunk" in decisions[0].evidence_reasons


def test_one_extractor_with_source_anchor_is_review_not_verified():
    decisions = adjudicate_candidates([_rel()], [_chunk()])
    assert decisions[0].evidence_valid
    assert decisions[0].decision == "review_required"
    assert "not a verified fact" in decisions[0].explanation


def test_independent_methods_corroborate_but_do_not_verify():
    candidates = [_rel(extractor="dependency_parser"), _rel(extractor="zero_shot_relation_model", confidence=0.78)]
    decisions = adjudicate_candidates(candidates, [_chunk()])
    assert decisions[0].decision == "corroborated_candidate"
    assert decisions[0].independent_extractor_count == 2
    assert "human verification is still required" in decisions[0].explanation


def test_same_source_duplicate_chunks_do_not_count_as_independent_documents():
    chunks = [_chunk("c1", doc_id="d1", source_url="https://example.org/a"),
              _chunk("c2", doc_id="d2", source_url="https://example.org/a")]
    candidates = [_rel(extractor="model_a", chunk_id="c1"), _rel(extractor="model_a", chunk_id="c2")]
    decisions = adjudicate_candidates(candidates, chunks)
    assert decisions[0].independent_source_count == 1
    assert decisions[0].decision == "review_required"


def test_different_sources_can_corroborate_but_same_text_is_not_auto_truth():
    chunks = [_chunk("c1", doc_id="d1", source_url="https://example.org/a"),
              _chunk("c2", doc_id="d2", source_url="https://example.net/b")]
    candidates = [_rel(extractor="model_a", chunk_id="c1"), _rel(extractor="model_a", chunk_id="c2")]
    decisions = adjudicate_candidates(candidates, chunks)
    assert decisions[0].decision == "corroborated_candidate"
    assert decisions[0].independent_source_count == 2


def test_calibration_abstains_when_labels_insufficient():
    result = choose_conservative_threshold([0.9, 0.2], [True, False], min_labeled=30)
    assert not result["auto_promotion_enabled"]
    assert result["threshold"] > 1.0


def test_calibration_selects_high_precision_policy_and_can_apply_it():
    scores = [0.9 + i * 0.001 for i in range(15)] + [0.1 + i * 0.001 for i in range(15)]
    labels = [True] * 15 + [False] * 15
    policy = choose_conservative_threshold(scores, labels, target_precision=0.79, min_accepted=8, min_labeled=30)
    assert policy["auto_promotion_enabled"]
    assert policy["precision_wilson_lb_95"] >= 0.79
    candidate = _rel()
    decision = adjudicate_candidates([candidate], [_chunk()])[0]
    # Use a synthetic policy from an independently labelled validation set; this test checks routing only.
    applied = apply_calibrated_threshold([decision], {"auto_promotion_enabled": True, "threshold": decision.evidence_quality_score - 0.01})
    assert applied[0].decision == "calibrated_candidate"


def test_shift_profile_flags_vocabulary_change_without_claiming_semantic_failure():
    ref = [_rel(relation="uses", quote="The service uses a local database.")]
    cur = [_rel(relation="absorbs", quote="The root system absorbs water.", subject="root system", object_="water")]
    report = schema_shift_profile(ref, cur)
    assert "relation_vocabulary_shift" in report["warnings"]
    assert "do not prove low accuracy" in report["interpretation"]


def test_domain_metrics_are_reported_separately():
    cases = [
        {"case_id": "a", "domain": "software", "gold_triples": [["service", "uses", "database"]]},
        {"case_id": "b", "domain": "ecology", "gold_triples": [["wetland", "supports", "birds"]]},
    ]
    predictions = {"a": [("service", "uses", "database")], "b": []}
    report = evaluate_by_domain(predictions, cases)
    assert report["overall"]["precision"] == 1.0
    assert report["overall"]["recall"] == 0.5
    assert report["per_domain"]["software"]["f1"] == 1.0
    assert report["per_domain"]["ecology"]["recall"] == 0.0


def test_general_audit_uses_open_world_competing_claim_label():
    topics = [Topic("t1", "topic", chunk_ids=["c1", "c2"])]
    nodes = [Node("s", "service", "entity", ["t1"], ["c1", "c2"]),
             Node("a", "database", "entity", ["t1"], ["c1"]),
             Node("b", "cache", "entity", ["t1"], ["c2"]),
             Node("topic:t1", "topic", ["t1"], ["c1", "c2"])]
    edges = [Edge("e1", "s", "a", "uses", ["c1"], 0.8, "candidate", {"evidence_quotes": ["The service uses a local database."]}),
             Edge("e2", "s", "b", "uses", ["c2"], 0.8, "candidate", {"evidence_quotes": ["The service uses a local cache."]})]
    snapshot = GraphSnapshot("snap", topics, nodes, edges)
    chunks = [_chunk("c1", "The service uses a local database.", "d1", "https://example.org/a"),
              _chunk("c2", "The service uses a local cache.", "d2", "https://example.net/b")]
    findings = audit_graph_general(snapshot, chunks)
    assert any(f.category == "competing_claims_requires_review" for f in findings)
    assert not any(f.category == "possible_functional_value_conflict" for f in findings)


def test_generic_version_diff_reports_any_predicate_changed_without_domain_rules():
    old_chunks = ["c1"]
    old = GraphSnapshot("old", [], [Node("s", "sensor", "entity"), Node("t1", "18 C", "value")],
                        [Edge("e_old", "s", "t1", "measures", old_chunks, 0.8, "candidate")], {"d1": "hash1"})
    new = GraphSnapshot("new", [], [Node("s", "sensor", "entity"), Node("t2", "20 C", "value")],
                        [Edge("e_new", "s", "t2", "measures", ["c2"], 0.8, "candidate")], {"d1": "hash2"})
    diff = compare_snapshots_general(old, new)
    assert len(diff.changed_claims) == 1
    assert diff.changed_claims[0].old_object == "18 C"
    assert diff.changed_claims[0].new_object == "20 C"


def test_cross_domain_benchmark_corpus_is_varied_and_original():
    path = Path(__file__).resolve().parents[1] / "examples" / "data" / "domain_shift_benchmark.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    domains = {d["domain"] for d in data["domains"]}
    cases = [c for d in data["domains"] for c in d["cases"]]
    assert len(domains) >= 4
    assert len(cases) >= 16
    assert all("gold_triples" in c for c in cases)
    assert "Original benchmark sentences authored" in data["notice"]


def test_gliner_adapter_batches_sentences_and_uses_sqlite_cache(tmp_path):
    from core.domain_agnostic_extraction import GLiNERRelexExtractor

    class FakeGLiNER:
        def __init__(self):
            self.calls = 0
        def inference(self, texts, labels, relations, threshold, relation_threshold, return_relations, flat_ner):
            self.calls += 1
            entity_batches, relation_batches = [], []
            for text in texts:
                if "database" in text:
                    entity_batches.append([
                        {"text": "service", "label": "organization", "score": 0.9},
                        {"text": "local database", "label": "concept", "score": 0.91},
                    ])
                    relation_batches.append([{"head": {"text": "service", "type": "organization"},
                                             "tail": {"text": "local database", "type": "concept"},
                                             "relation": "uses", "score": 0.85}])
                else:
                    entity_batches.append([
                        {"text": "cache", "label": "concept", "score": 0.9},
                        {"text": "recent results", "label": "concept", "score": 0.91},
                    ])
                    relation_batches.append([{"head": {"text": "cache", "type": "concept"},
                                             "tail": {"text": "recent results", "type": "concept"},
                                             "relation": "stores", "score": 0.86}])
            return entity_batches, relation_batches

    adapter = GLiNERRelexExtractor(cache_path=str(tmp_path / "extract.sqlite3"))
    fake = FakeGLiNER()
    adapter._model = fake  # inject a fake interface to test orchestration without downloading weights
    chunk = Chunk("c1", "d1", "The service uses a local database. The cache stores recent results.")
    entities, relations = adapter.extract_chunk(chunk)
    assert len(relations) == 2
    assert {r.relation_type for r in relations} == {"uses", "stores"}
    assert all(r.evidence_text in chunk.text for r in relations)
    assert fake.calls == 1  # batched: one inference call for two sentences
    adapter.extract_chunk(chunk)
    assert fake.calls == 1  # second call served from persistent cache
    assert adapter.cache_hits == 1


def test_graph_builder_reuses_typed_extracted_entities_for_relation_edges():
    from core.graph_builder import build_graph
    from models import EntityCandidate, RelationCandidate
    chunks = [Chunk("c1", "d1", "The service uses a local database.", topic_ids=["t1"])]
    topics = [Topic("t1", "Software systems", keywords=["service", "database"], chunk_ids=["c1"])]
    entities = [EntityCandidate("service", "organization", "c1", 0.9, {"extractor": "model"}),
                EntityCandidate("local database", "concept", "c1", 0.9, {"extractor": "model"})]
    relations = [RelationCandidate("service", "uses", "local database", "c1",
                                   "The service uses a local database.", 0.85,
                                   {"extractor": "model", "subject_type": "organization", "object_type": "concept"})]
    graph = build_graph(chunks, topics, entities, relations)
    service_nodes = [n for n in graph.nodes if n.label.casefold() == "service"]
    db_nodes = [n for n in graph.nodes if n.label.casefold() == "local database"]
    assert len(service_nodes) == 1
    assert service_nodes[0].node_type == "organization"
    assert len(db_nodes) == 1
    assert db_nodes[0].node_type == "concept"
