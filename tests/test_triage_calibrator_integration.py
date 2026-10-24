from models import Chunk, RelationCandidate
from core.evidence_calibrator import FEATURES, LogisticCalibrationModel
from core.selective_quality import adjudicate_candidates


def _chunk():
    return Chunk(
        chunk_id="c1",
        doc_id="d1",
        text="The sensor measures temperature.",
        source_url="https://example.org/sensor",
    )


def _candidate():
    return RelationCandidate(
        subject="sensor",
        relation_type="measures",
        object="temperature",
        chunk_id="c1",
        evidence_text="The sensor measures temperature.",
        confidence=0.8,
        attributes={"extractor": "test_extractor"},
    )


def _training_result():
    model = LogisticCalibrationModel(
        features=list(FEATURES),
        means=[0.0] * len(FEATURES),
        scales=[1.0] * len(FEATURES),
        weights=[0.1] * len(FEATURES),
        bias=0.0,
        trained_on_rows=80,
        trained_domains=["software", "ecology", "medicine"],
    )
    return {"trained": True, "model": model.to_dict()}


def test_learned_triage_score_is_added_but_never_verifies_claim():
    result = adjudicate_candidates([_candidate()], [_chunk()], calibration_model=_training_result())
    assert len(result) == 1
    decision = result[0]
    assert decision.triage_model_score is not None
    assert 0.0 <= decision.triage_model_score <= 1.0
    assert decision.triage_score_kind == "learned_review_priority_higher_means_review_first_not_truth_probability"
    # A model review score must not turn a single-source candidate into a verified fact.
    assert decision.decision == "review_required"
    assert "human verification" in decision.explanation.lower() or "not a verified fact" in decision.explanation.lower()


def test_untrained_or_insufficient_calibrator_fails_closed():
    raw = _training_result()
    raw["trained"] = False
    result = adjudicate_candidates([_candidate()], [_chunk()], calibration_model=raw)
    assert result[0].triage_model_score is None
    assert result[0].decision == "review_required"

    too_small = _training_result()
    too_small["model"]["trained_on_rows"] = 40
    result = adjudicate_candidates([_candidate()], [_chunk()], calibration_model=too_small)
    assert result[0].triage_model_score is None
    assert result[0].decision == "review_required"


def test_single_domain_calibrator_is_not_eligible_for_cross_domain_triage():
    raw = _training_result()
    raw["model"]["trained_domains"] = ["software"]
    result = adjudicate_candidates([_candidate()], [_chunk()], calibration_model=raw)
    assert result[0].triage_model_score is None
    assert result[0].triage_score_kind == "not_enabled_or_insufficient_reviewed_training_data"


def test_graph_builder_preserves_triage_signal_without_promoting_edge():
    from models import Topic
    from core.graph_builder import build_graph

    chunk = _chunk()
    candidate = _candidate()
    decisions = adjudicate_candidates([candidate], [chunk], calibration_model=_training_result())
    graph = build_graph(
        [chunk],
        [Topic(topic_id="t1", label="sensors", chunk_ids=["c1"])],
        [],
        [candidate],
        candidate_decisions=[decision.to_dict() for decision in decisions],
    )
    edge = next(edge for edge in graph.edges if edge.relation_type == "measures")
    assert edge.status == "review_required"
    assert edge.attributes["triage_model_score"] is not None
    assert edge.attributes["triage_score_kind"] == "learned_review_priority_higher_means_review_first_not_truth_probability"
    assert edge.attributes["human_verification_required"] is True
