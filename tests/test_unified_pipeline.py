from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline import TopicWeaveEngine


class FakeUnavailableCrosslingual:
    def search(self, query, chunks, top_k=5):
        # Simulate a weak lexical fallback that finds a shared English word.
        class Hit:
            def to_dict(self):
                return {"chunk_id": chunks[0]["chunk_id"], "doc_id": chunks[0]["doc_id"],
                        "score": 0.1, "text": chunks[0]["text"], "source_title": "", "source_url": ""}
        return [Hit()]

    def diagnostics(self):
        return {"mode": "lexical_fallback_limited_crosslingual", "last_fallback_note": "No model available."}


def test_unified_pipeline_ingest_query_and_graph_evidence():
    with TemporaryDirectory() as tmp:
        engine = TopicWeaveEngine(tmp, enable_topic_embeddings=False,
                                  enable_dense_retrieval=False, enable_crosslingual=False)
        result = engine.upsert_files([
            ("research.txt", b"Research teams publish technical reports. Reports contain methods and results."),
            ("security.txt", b"Password hashing protects stored credentials. Multi-factor authentication reduces account takeover risk."),
        ], source_metadata={
            "research.txt": {"source_url": "https://example.org/research", "source_title": "Research Notes"},
            "security.txt": {"source_url": "https://example.org/security", "source_title": "Security Notes"},
        })
        assert len(engine.registry.docs) == 2
        assert result.snapshot is not None
        assert result.snapshot.nodes and result.snapshot.edges
        assert all(edge.edge_id for edge in result.snapshot.edges)
        factual = engine.query("What protects stored credentials?")
        assert factual["hits"]
        assert "Password hashing protects stored credentials" in factual["answer"]
        assert factual["evidence_context"]
        broad = engine.query("What are the main themes across these documents?")
        assert broad["route"]["route"] == "thematic"
        assert broad["topic_overview"]
        engine.close()


def test_unified_pipeline_edit_reuses_unchanged_chunk_claims_and_delete_is_scoped():
    with TemporaryDirectory() as tmp:
        engine = TopicWeaveEngine(tmp, enable_topic_embeddings=False, enable_dense_retrieval=False)
        original = ("Research teams publish technical reports. Reports contain methods and results. "
                    "The library provides online journals.")
        other = "Password hashing protects stored credentials. Multi-factor authentication reduces account takeover risk."
        engine.upsert_files([("research.txt", original.encode()), ("security.txt", other.encode())])
        before_claims = engine.graph_service.query_claims()
        assert any(c["doc_id"] == engine.registry.docs[next(k for k,v in engine.registry.docs.items() if v.filename == "security.txt")].doc_id for c in before_claims)
        updated = original + " A technical standard defines reporting requirements."
        changed = engine.upsert_files([("research.txt", updated.encode())])
        op = next(x for x in changed.diagnostics["document_operations"] if x["doc_id"] == next(k for k,v in engine.registry.docs.items() if v.filename == "research.txt"))
        assert op["incremental_extraction"]["fresh_chunks"] <= 2
        claims_after_edit = engine.graph_service.query_claims()
        assert any("Password hashing" in c["evidence_text"] for c in claims_after_edit)
        deleted = engine.delete_document("research.txt")
        assert deleted.deleted_documents
        remaining_doc_ids = {d.doc_id for d in engine.registry.docs.values()}
        assert remaining_doc_ids == {d.doc_id for d in engine.registry.docs.values() if d.filename == "security.txt"}
        assert all(c["doc_id"] in remaining_doc_ids for c in engine.graph_service.query_claims())
        engine.close()


def test_hindi_to_english_abstains_when_multilingual_model_is_unavailable():
    with TemporaryDirectory() as tmp:
        engine = TopicWeaveEngine(tmp, enable_topic_embeddings=False, enable_dense_retrieval=False,
                                  enable_crosslingual=True, crosslingual_retriever=FakeUnavailableCrosslingual())
        engine.upsert_files([("english.txt", b"Password hashing protects stored credentials.")])
        result = engine.query("कौन सा तरीका stored credentials की रक्षा करता है?")
        assert result["hits"] == []
        assert result["diagnostics"]["abstained"] is True
        assert result["diagnostics"]["retrieval"]["safe_abstention"] is True
        engine.close()


def test_metadata_change_updates_source_without_rebuilding_content():
    with TemporaryDirectory() as tmp:
        engine = TopicWeaveEngine(tmp, enable_topic_embeddings=False, enable_dense_retrieval=False)
        body = b"The research library provides online journals. Journals support academic research."
        engine.upsert_files([("a.txt", body)], source_metadata={"a.txt": {"source_title": "Old title", "source_url": "https://example.org/old"}})
        did = next(iter(engine.registry.docs))
        ids_before = [c.chunk_id for c in engine.registry.chunks]
        result = engine.upsert_files([("a.txt", body)], source_metadata={"a.txt": {"source_title": "New title", "source_url": "https://example.org/new"}})
        assert ids_before == [c.chunk_id for c in engine.registry.chunks]
        assert did in result.changed_documents
        claims = engine.graph_service.query_claims()
        assert all(c["source_title"] == "New title" for c in claims)
        engine.close()


def test_hindi_to_english_abstains_when_crosslingual_model_not_enabled():
    with TemporaryDirectory() as tmp:
        engine = TopicWeaveEngine(tmp, enable_topic_embeddings=False,
                                  enable_dense_retrieval=False, enable_crosslingual=False)
        engine.upsert_files([("english.txt", b"Password hashing protects stored credentials.")])
        result = engine.query("कौन सा तरीका stored credentials की रक्षा करता है?")
        assert result["hits"] == []
        assert result["diagnostics"]["abstained"] is True
        assert result["diagnostics"]["retrieval"]["safe_abstention"] is True
        engine.close()


def test_reopening_registry_with_empty_graph_database_rebuilds_claims(tmp_path):
    registry_dir = tmp_path / "persisted"
    first = TopicWeaveEngine(registry_dir, enable_topic_embeddings=False, enable_dense_retrieval=False)
    first.upsert_files([("a.txt", b"A server uses a local database.")])
    first.close()
    # Keep the registry but remove the graph/audit files to simulate graph DB loss.
    (registry_dir / "knowledge.sqlite3").unlink()
    audit = registry_dir / "audit_events.jsonl"
    if audit.exists():
        audit.unlink()
    second = TopicWeaveEngine(registry_dir, enable_topic_embeddings=False, enable_dense_retrieval=False)
    assert second.graph_service.kb.metrics()["documents"] == 1
    assert second.graph_service.query_claims(doc_id=next(iter(second.registry.docs)))
    second.close()


def test_relation_chain_path_uses_fact_edges_not_topic_membership_edges():
    with TemporaryDirectory() as tmp:
        engine = TopicWeaveEngine(tmp, enable_topic_embeddings=False, enable_dense_retrieval=False)
        engine.upsert_files([
            ("portal.txt", b"Student Portal provides enrollment records."),
            ("attendance.txt", b"Enrollment records support attendance reporting."),
            ("coral.txt", b"Coral reefs shelter juvenile fish."),
            ("battery.txt", b"Battery systems monitor charge cycles."),
        ])
        result = engine.query("How is Student Portal connected to attendance reporting?")
        assert result["route"]["route"] == "relation_chain"
        assert result["graph_paths"]
        relations = [step["relation"] for step in result["graph_paths"][0]["steps"]]
        assert "contains_concept" not in relations
        assert any(r in {"provides", "supports"} for r in relations)
        engine.close()


def test_hindi_thematic_question_does_not_bypass_crosslingual_abstention():
    with TemporaryDirectory() as tmp:
        engine = TopicWeaveEngine(tmp, enable_topic_embeddings=False,
                                  enable_dense_retrieval=False, enable_crosslingual=False)
        engine.upsert_files([
            ("a.txt", b"The library provides digital research resources."),
            ("b.txt", b"The application stores recent results."),
        ])
        result = engine.query("इन दस्तावेज़ों के मुख्य विषय क्या हैं?")
        assert result["hits"] == []
        assert result["topic_overview"] == []
        assert result["diagnostics"]["abstained"] is True
        engine.close()
