"""Member 1 tests: topic discovery (TF-IDF path; dense path is exercised when the model is installed)."""
from core.chunking import chunk_documents
from core.ingestion import extract_documents
from core.registry import CorpusRegistry
from core.topics import TopicModel, discover_topics
from tests.helpers import CORPUS


def _chunks():
    return chunk_documents(extract_documents([(k, v.encode()) for k, v in CORPUS.items()]))


def test_topics_cover_all_chunks_and_have_labels():
    chunks = _chunks()
    topics, out = discover_topics(chunks, k=3, use_embeddings=False)
    assert len(topics) == 3 and len(out) == len(chunks)
    assert all(c.topic_ids for c in out)
    assert all(t.label and t.keywords and t.chunk_ids for t in topics)
    ids = {c.chunk_id for c in chunks}
    assert all(set(t.chunk_ids) <= ids for t in topics)


def test_topics_separate_energy_medicine_finance():
    chunks = _chunks()
    topics, out = discover_topics(chunks, k=3, use_embeddings=False)
    dom = {}
    for c in out:
        dom.setdefault(c.doc_id, []).append(c.topic_ids[0])
    from core.ingestion import make_doc_id
    main = lambda f: max(set(dom[make_doc_id(f)]), key=dom[make_doc_id(f)].count)
    assert main("solar.txt") == main("wind.txt")
    assert main("medicine.txt") == main("vaccines.txt")
    assert main("finance.txt") == main("markets.txt")
    assert len({main("solar.txt"), main("medicine.txt"), main("finance.txt")}) == 3


def test_deterministic_topics():
    a, _ = discover_topics(_chunks(), k=3, use_embeddings=False)
    b, _ = discover_topics(_chunks(), k=3, use_embeddings=False)
    assert [(t.topic_id, t.keywords) for t in a] == [(t.topic_id, t.keywords) for t in b]


def test_degenerate_inputs():
    assert discover_topics([], use_embeddings=False) == ([], [])
    one = _chunks()[:2]
    topics, out = discover_topics(one, use_embeddings=False)
    assert len(topics) == 1 and all(c.topic_ids == ["t0"] for c in out)


def test_incremental_update_keeps_topic_ids_stable():
    reg = CorpusRegistry()
    reg.upsert([(k, v.encode()) for k, v in CORPUS.items()])
    tm = TopicModel(k=3, use_embeddings=False, refit_churn=0.9)
    topics, _ = tm.fit(reg.chunks)
    before = [(t.topic_id, t.label) for t in topics]
    cs = reg.upsert([("solar2.txt", b"Rooftop solar panels and photovoltaic batteries cut electricity bills for households in Rajasthan. "
                                     b"Solar storage with lithium batteries improves grid reliability across India.")])
    topics2, chunks2 = tm.update(reg.chunks, cs.new_chunk_ids, cs.removed_chunk_ids)
    assert [(t.topic_id, t.label) for t in topics2] == before                 # no refit, IDs stable
    new = [c for c in chunks2 if c.chunk_id in set(cs.new_chunk_ids)]
    assert new and new[0].topic_ids[0] == next(c for c in chunks2 if "Photovoltaic" in c.text).topic_ids[0]
