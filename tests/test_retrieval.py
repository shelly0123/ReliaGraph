"""Member 1 tests: retrieval + synthetic question generation."""
from core.chunking import chunk_documents
from core.ingestion import extract_documents
from core.question_generator import NOT_ANSWERABLE, generate_test_cases
from core.retrieval import Retriever, detect_lang, retrieve
from core.topics import discover_topics
from models import Edge, Node
from tests.helpers import CORPUS


def _chunks():
    return chunk_documents(extract_documents([(k, v.encode()) for k, v in CORPUS.items()]))


def test_known_query_retrieves_supporting_passage_with_valid_ids():
    chunks = _chunks()
    r = Retriever(use_dense=False).index(chunks)
    hits = r.search("Which drug is first-line for type 2 diabetes?", top_k=3)
    ids = {c.chunk_id for c in chunks}
    assert hits and all(h.chunk_id in ids and h.score > 0 for h in hits)
    assert "Metformin" in hits[0].chunk.text and hits[0].rank == 1
    assert [h.rank for h in hits] == [1, 2, 3][:len(hits)]


def test_agreed_signature_returns_chunks():
    out = retrieve("repo rate inflation Reserve Bank", _chunks(), top_k=2)
    assert out and "Reserve Bank" in out[0].text


def test_filter_by_topic_and_doc():
    topics, chunks = discover_topics(_chunks(), k=3, use_embeddings=False)
    r = Retriever(use_dense=False).index(chunks)
    med = next(c for c in chunks if "Metformin" in c.text).topic_ids[0]
    hits = r.search("interest rates", top_k=5, topic_id=med)
    assert all(med in h.chunk.topic_ids for h in hits)
    assert r.search("diabetes", doc_ids=["nope"]) == []


def test_incremental_update_add_and_remove():
    chunks = _chunks()
    r = Retriever(use_dense=False).index(chunks[:-3])
    assert len(r) == len(chunks) - 3
    r.update(add=chunks[-3:], remove_ids=[chunks[0].chunk_id])
    assert len(r) == len(chunks) - 1
    assert chunks[0].chunk_id not in {h.chunk_id for h in r.search(chunks[0].text, top_k=10)}


def test_hindi_query_without_multilingual_model_warns_instead_of_garbage():
    assert detect_lang("भारत में सौर ऊर्जा क्षमता कितनी है?") == "hi" and detect_lang("solar") == "en"
    r = Retriever(use_dense=False).index(_chunks())
    assert r.search("भारत में सौर ऊर्जा क्षमता कितनी है?") == [] and r.last_warning
    r2 = Retriever(use_dense=False, query_translator=lambda q: "solar power capacity in India").index(_chunks())
    assert "Solar" in r2.search("भारत में सौर ऊर्जा क्षमता कितनी है?")[0].chunk.text


def test_test_cases_are_reproducible_grounded_and_include_unanswerable():
    chunks = _chunks()
    topics, chunks = discover_topics(chunks, k=3, use_embeddings=False)
    a = generate_test_cases(chunks, topics=topics, n=20, seed=1)
    b = generate_test_cases(chunks, topics=topics, n=20, seed=1)
    assert [t.test_id for t in a] == [t.test_id for t in b] and len(a) >= 8
    cmap = {c.chunk_id: c for c in chunks}
    for t in a:
        if t.kind == "cloze":
            assert t.reference_answer in cmap[t.supporting_chunk_ids[0]].text
            assert t.reference_answer not in t.question.split("blank:")[1]
    un = [t for t in a if t.kind == "unanswerable"]
    assert un and all(not t.answerable and not t.supporting_chunk_ids and t.reference_answer == NOT_ANSWERABLE for t in un)
    assert any(t.lang == "hi" for t in a)


def test_cloze_question_retrieves_its_own_supporting_chunk():
    chunks = _chunks()
    r = Retriever(use_dense=False).index(chunks)
    cases = [t for t in generate_test_cases(chunks, n=30, seed=3) if t.kind == "cloze"]
    assert cases
    hit = sum(t.supporting_chunk_ids[0] in {h.chunk_id for h in r.search(t.question, top_k=3)} for t in cases)
    assert hit / len(cases) >= 0.8


def test_edge_based_and_multihop_cases():
    chunks = _chunks()
    nodes = [Node("n1", "Metformin", "DRUG"), Node("n2", "Type 2 Diabetes", "DISEASE"), Node("n3", "Blood Glucose", "METRIC")]
    edges = [Edge("e1", "n1", "n2", "TREATS", [chunks[0].chunk_id], "supported"),
             Edge("e2", "n2", "n3", "RAISES", [chunks[1].chunk_id], "supported")]
    cases = generate_test_cases(chunks, edges, nodes=nodes, n=24, seed=2)
    kinds = {t.kind for t in cases}
    assert {"relation", "multi_hop"} <= kinds
    hop = next(t for t in cases if t.kind == "multi_hop")
    assert hop.reference_answer == "Blood Glucose" and set(hop.supporting_chunk_ids) == {chunks[0].chunk_id, chunks[1].chunk_id}
