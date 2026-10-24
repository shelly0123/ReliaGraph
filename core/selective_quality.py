"""Evidence-first fusion, selective promotion, score calibration and drift signals.

Design principle: maximize the precision of the *promoted/corroborated subset* by
rejecting provenance failures and abstaining on uncertain candidates. This trades
coverage for reliability and cannot guarantee performance on arbitrary data.
"""
from __future__ import annotations

from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from math import sqrt
from typing import Any, Iterable, Sequence, Mapping
import re

from models import Chunk, RelationCandidate, chunk_from
from core.domain_agnostic_extraction import norm_text, text_contains_span


def _norm_entity(value: str) -> str:
    value = norm_text(value)
    # Ignore only leading English articles for evaluation/entity matching. This does not
    # change display labels or merge graph nodes by itself.
    value = re.sub(r"^(?:the|a|an)\s+", "", value)
    return value


def canonical_triple(subject: str, relation: str, object_: str) -> tuple[str, str, str]:
    """Normalize formatting variants without guessing semantic synonyms.

    Underscores and whitespace are equivalent separators for relation labels (for
    example ``part_of`` and ``part of``). Unknown predicate synonyms are *not* merged.
    """
    relation_key = re.sub(r"[_\s]+", " ", norm_text(relation)).strip()
    return _norm_entity(subject), relation_key, _norm_entity(object_)


def _wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    denom = 1 + z * z / total
    center = p + z * z / (2 * total)
    spread = z * sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return max(0.0, (center - spread) / denom)


@dataclass
class CandidateDecision:
    subject: str
    relation_type: str
    object: str
    chunk_id: str
    evidence_text: str
    evidence_valid: bool
    evidence_reasons: list[str]
    extractor_names: list[str]
    independent_extractor_count: int
    independent_source_count: int
    evidence_quality_score: float
    decision: str
    explanation: str
    candidate_ids: list[str]
    triage_model_score: float | None = None
    triage_score_kind: str = "not_enabled"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_attr(candidate: Any, name: str, default: Any = None) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(name, default)
    return getattr(candidate, name, default)


def assess_candidate_evidence(candidate: Any, chunk_by_id: dict[str, Chunk]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    chunk_id = str(_candidate_attr(candidate, "chunk_id", ""))
    chunk = chunk_by_id.get(chunk_id)
    if chunk is None:
        reasons.append("unknown_chunk_id")
        return False, reasons
    quote = str(_candidate_attr(candidate, "evidence_text", "") or "").strip()
    if not quote:
        reasons.append("missing_evidence_quote")
    elif not text_contains_span(chunk.text, quote):
        reasons.append("quote_not_found_in_source_chunk")
    subject = str(_candidate_attr(candidate, "subject", "") or "").strip()
    object_ = str(_candidate_attr(candidate, "object", "") or "").strip()
    comparison_text = quote if quote and text_contains_span(chunk.text, quote) else chunk.text
    if not subject or not text_contains_span(comparison_text, subject):
        reasons.append("subject_not_anchored_to_evidence")
    if not object_ or not text_contains_span(comparison_text, object_):
        reasons.append("object_not_anchored_to_evidence")
    relation = str(_candidate_attr(candidate, "relation_type", "") or "").strip()
    if not relation:
        reasons.append("empty_relation_label")
    return len(reasons) == 0, reasons


def adjudicate_candidates(
    candidates: Sequence[RelationCandidate | dict[str, Any]],
    chunks: Sequence[Chunk | dict[str, Any]],
    min_score: float = 0.72,
    require_independent_support: bool = True,
    calibration_model: Mapping[str, Any] | Any | None = None,
) -> list[CandidateDecision]:
    """Merge duplicate triples and decide which are grounded/corroborated/review-only.

    `corroborated_candidate` is NOT a verified fact. It means evidence spans align and
    there are multiple extraction methods or distinct source documents. Human review
    remains a separate state in the graph workflow.
    """
    normalized_chunks = [chunk_from(c) for c in chunks]
    chunk_by_id = {c.chunk_id: c for c in normalized_chunks}

    # Optional locally trained triage model. Its output only orders human review; it
    # never changes a candidate into a verified fact and never bypasses provenance gates.
    triage_model = None
    triage_model_eligible = False
    if calibration_model is not None:
        try:
            from core.evidence_calibrator import LogisticCalibrationModel
            raw_model: Any = calibration_model
            if isinstance(raw_model, Mapping) and "model" in raw_model:
                if raw_model.get("trained") is not True:
                    raise ValueError("calibration training result is not marked trained")
                raw_model = raw_model["model"]
            triage_model = raw_model if hasattr(raw_model, "predict_proba") else LogisticCalibrationModel.from_dict(raw_model)
            trained_rows = int(getattr(triage_model, "trained_on_rows", 0))
            trained_domains = list(getattr(triage_model, "trained_domains", []) or [])
            triage_model_eligible = trained_rows >= 60 and len(set(trained_domains)) >= 3
        except Exception:
            triage_model = None
            triage_model_eligible = False
    groups: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for candidate in candidates:
        key = canonical_triple(
            str(_candidate_attr(candidate, "subject", "")),
            str(_candidate_attr(candidate, "relation_type", "")),
            str(_candidate_attr(candidate, "object", "")),
        )
        if all(key):
            groups[key].append(candidate)

    result: list[CandidateDecision] = []
    for key, group in sorted(groups.items()):
        reasons_all: list[str] = []
        valid_candidates = []
        invalid_candidates = []
        for candidate in group:
            valid, reasons = assess_candidate_evidence(candidate, chunk_by_id)
            if valid:
                valid_candidates.append(candidate)
            else:
                invalid_candidates.append(candidate)
                reasons_all.extend(reasons)
        valid = bool(valid_candidates)
        methods = sorted({str((_candidate_attr(c, "attributes", {}) or {}).get("extractor", "unknown"))
                          for c in valid_candidates})
        source_ids = set()
        evidence_ids = set()
        quotes = []
        confidences = []
        for candidate in valid_candidates:
            chunk_id = str(_candidate_attr(candidate, "chunk_id", ""))
            chunk = chunk_by_id.get(chunk_id)
            if chunk:
                # Distinct URLs are preferred over doc IDs to avoid counting multiple
                # chunks/versions of the same source as independent corroboration.
                source_ids.add((chunk.source_url or "").casefold().strip() or str(chunk.metadata.get("content_hash") or chunk.doc_id))
            evidence_ids.add(chunk_id)
            quote = str(_candidate_attr(candidate, "evidence_text", "") or "").strip()
            if quote:
                quotes.append(quote)
            conf = _candidate_attr(candidate, "confidence", None)
            if isinstance(conf, (int, float)) and 0 <= float(conf) <= 1:
                confidences.append(float(conf))
        # This is a ranking score, not probability: 45% evidence gate, 20% model score,
        # 20% method agreement, 15% independent source corroboration.
        evidence_component = 1.0 if valid else 0.0
        model_component = (sum(confidences) / len(confidences)) if confidences else 0.5
        agreement_component = 1.0 if len(methods) >= 2 else 0.0
        source_component = min(1.0, len(source_ids) / 2.0)
        score = (0.45 * evidence_component + 0.20 * model_component +
                 0.20 * agreement_component + 0.15 * source_component)
        independent_support = len(methods) >= 2 or len(source_ids) >= 2
        if not valid:
            decision = "rejected_provenance"
            explanation = "No candidate in this group passed source-anchor checks; do not add this claim to the graph."
        elif score >= min_score and (independent_support or not require_independent_support):
            decision = "corroborated_candidate"
            explanation = "Evidence is source-anchored and the candidate has independent-method or independent-source support; human verification is still required."
        else:
            decision = "review_required"
            explanation = "Evidence is anchored, but available support is insufficient for promotion at the current policy. This is not a verified fact; preserve it as a candidate and abstain from treating it as verified."
        representative = valid_candidates[0] if valid_candidates else group[0]
        triage_score: float | None = None
        if triage_model is not None and triage_model_eligible:
            nli_values: dict[str, list[float]] = {"nli_entailment": [], "nli_contradiction": [], "nli_neutral": []}
            injection_flag = 0
            domain_shift_flag = 0
            for candidate in valid_candidates:
                attrs = dict(_candidate_attr(candidate, "attributes", {}) or {})
                chunk = chunk_by_id.get(str(_candidate_attr(candidate, "chunk_id", "")))
                metadata = dict(chunk.metadata or {}) if chunk else {}
                injection_flag = max(injection_flag, int(bool(attrs.get("injection_flag") or metadata.get("injection_flag") or metadata.get("injection_findings"))))
                domain_shift_flag = max(domain_shift_flag, int(bool(attrs.get("domain_shift_flag") or metadata.get("domain_shift_flag"))))
                for name in nli_values:
                    value = attrs.get(name, metadata.get(name))
                    if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
                        nli_values[name].append(float(value))
            triage_features = {
                "source_anchor_valid": int(valid),
                "quote_present": int(bool(quotes)),
                "extractor_score": round(model_component, 6),
                "independent_extractor_count": len(methods),
                "independent_source_count": len(source_ids),
                "nli_entailment": sum(nli_values["nli_entailment"]) / max(1, len(nli_values["nli_entailment"])),
                "nli_contradiction": sum(nli_values["nli_contradiction"]) / max(1, len(nli_values["nli_contradiction"])),
                "nli_neutral": sum(nli_values["nli_neutral"]) / max(1, len(nli_values["nli_neutral"])),
                "injection_flag": injection_flag,
                "domain_shift_flag": domain_shift_flag,
            }
            try:
                support_signal = float(triage_model.predict_proba(triage_features))
                # The calibrator predicts the human-reviewed supported-label signal.
                # Convert it to review priority so HIGHER means a candidate should be
                # reviewed sooner. This remains an uncalibrated ranking signal, not truth.
                triage_score = round(1.0 - support_signal, 4)
            except Exception:
                triage_score = None
        result.append(CandidateDecision(
            subject=str(_candidate_attr(representative, "subject", key[0])),
            relation_type=str(_candidate_attr(representative, "relation_type", key[1])),
            object=str(_candidate_attr(representative, "object", key[2])),
            chunk_id=str(_candidate_attr(representative, "chunk_id", "")),
            evidence_text=(quotes[0] if quotes else str(_candidate_attr(representative, "evidence_text", ""))),
            evidence_valid=valid, evidence_reasons=sorted(set(reasons_all)),
            extractor_names=methods, independent_extractor_count=len(methods),
            independent_source_count=len(source_ids), evidence_quality_score=round(score, 4),
            decision=decision,
            explanation=(explanation + (" Optional triage-model score is a review-priority signal, not proof of truth." if triage_score is not None else "")),
            candidate_ids=sorted({str(_candidate_attr(c, "candidate_id", "") or _candidate_attr(c, "chunk_id", "")) for c in group}),
            triage_model_score=triage_score,
            triage_score_kind="learned_review_priority_higher_means_review_first_not_truth_probability" if triage_score is not None else "not_enabled_or_insufficient_reviewed_training_data",
        ))
    return result


def precision_recall_f1(predicted: Iterable[tuple[str, str, str]], gold: Iterable[tuple[str, str, str]]) -> dict[str, Any]:
    pred = {canonical_triple(*x) for x in predicted}
    ref = {canonical_triple(*x) for x in gold}
    tp = len(pred & ref)
    fp = len(pred - ref)
    fn = len(ref - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "predicted": len(pred), "gold": len(ref)}


def threshold_curve(scores: Sequence[float], labels: Sequence[bool], thresholds: Sequence[float] | None = None) -> list[dict[str, Any]]:
    """Compute precision/recall/coverage for thresholds on an annotated validation set."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    if thresholds is None:
        thresholds = [round(x / 20, 3) for x in range(0, 21)]
    positives = sum(bool(x) for x in labels)
    curve = []
    for threshold in thresholds:
        selected = [bool(label) for score, label in zip(scores, labels) if float(score) >= threshold]
        total = len(selected)
        tp = sum(selected)
        precision = tp / total if total else 0.0
        recall = tp / positives if positives else 0.0
        curve.append({"threshold": float(threshold), "accepted": total, "coverage": total / len(scores) if scores else 0.0,
                      "tp": tp, "precision": round(precision, 4), "precision_wilson_lb_95": round(_wilson_lower_bound(tp, total), 4),
                      "recall": round(recall, 4), "gold_positives": positives})
    return curve


def choose_conservative_threshold(
    scores: Sequence[float], labels: Sequence[bool], target_precision: float = 0.90,
    min_accepted: int = 8, min_labeled: int = 30, require_wilson_lower_bound: bool = True,
) -> dict[str, Any]:
    """Choose widest coverage satisfying a target precision on labelled calibration data.

    Uses a Wilson lower confidence bound to avoid treating a tiny all-correct sample as
    proof. If evidence is insufficient, the policy abstains from auto-promotion. Use a
    separate grouped-by-document holdout to report final metrics; do not score on the
    same labels used to choose this threshold.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    if len(scores) < min_labeled or sum(bool(v) for v in labels) == 0 or sum(not bool(v) for v in labels) == 0:
        return {"auto_promotion_enabled": False, "threshold": 1.000001, "reason": "insufficient_calibration_labels",
                "label_count": len(labels), "required_label_count": min_labeled,
                "instruction": "Keep candidates in review; gather labelled examples from this data source before tuning."}
    curve = threshold_curve(scores, labels)
    eligible = [r for r in curve if r["accepted"] >= min_accepted and
                ((r["precision_wilson_lb_95"] if require_wilson_lower_bound else r["precision"]) >= target_precision)]
    if not eligible:
        return {"auto_promotion_enabled": False, "threshold": 1.000001, "reason": "target_precision_not_supported_by_calibration_set",
                "label_count": len(labels), "target_precision": target_precision,
                "instruction": "Do not auto-promote candidates. Improve extraction or obtain more independent labels."}
    chosen = max(eligible, key=lambda r: (r["coverage"], -r["threshold"]))
    return {"auto_promotion_enabled": True, "threshold": chosen["threshold"], "target_precision": target_precision,
            "calibration_precision": chosen["precision"], "precision_wilson_lb_95": chosen["precision_wilson_lb_95"],
            "recall": chosen["recall"], "coverage": chosen["coverage"], "accepted": chosen["accepted"],
            "label_count": len(labels), "reason": "selected_on_labelled_calibration_set",
            "warning": "This threshold does not guarantee precision after domain shift; monitor a held-out set."}


def apply_calibrated_threshold(decisions: Sequence[CandidateDecision], policy: dict[str, Any]) -> list[CandidateDecision]:
    """Apply a policy learned from labelled calibration examples to candidate scores.

    If the calibrator does not have enough evidence to meet its target, every candidate
    remains under review. A threshold is not transferable automatically after domain shift.
    """
    if not policy.get("auto_promotion_enabled"):
        return list(decisions)
    threshold = float(policy.get("threshold", 1.000001))
    import copy
    output = []
    for item in decisions:
        clone = copy.deepcopy(item)
        if clone.evidence_valid and clone.evidence_quality_score >= threshold and clone.decision == "review_required":
            clone.decision = "calibrated_candidate"
            clone.explanation = ("This candidate meets a threshold selected on labelled calibration data and passes source anchoring. "
                                 "It is not verified; review the held-out evaluation results before using auto-promotion in production.")
        output.append(clone)
    return output


def schema_shift_profile(reference: Sequence[RelationCandidate | dict[str, Any]],
                         current: Sequence[RelationCandidate | dict[str, Any]]) -> dict[str, Any]:
    """Surface relation-label, evidence-vocabulary and extraction-yield shifts.

    These are warnings, not a reliable universal OOD detector. They can signal when a
    previously calibrated policy needs re-checking on freshly labelled examples.
    """
    def relation_set(rows: Sequence[Any]) -> set[str]:
        return {norm_text(str(_candidate_attr(x, "relation_type", ""))) for x in rows if norm_text(str(_candidate_attr(x, "relation_type", "")))}
    ref_labels = relation_set(reference)
    cur_labels = relation_set(current)
    union = ref_labels | cur_labels
    jaccard = len(ref_labels & cur_labels) / len(union) if union else 1.0
    new_relations = sorted(cur_labels - ref_labels)
    lost_relations = sorted(ref_labels - cur_labels)
    def evidence_vocab(rows: Sequence[Any]) -> set[str]:
        vocab: set[str] = set()
        for row in rows:
            text = str(_candidate_attr(row, "evidence_text", "") or "").casefold()
            vocab.update(re.findall(r"[a-z0-9]{3,}", text))
        return vocab
    ref_vocab, cur_vocab = evidence_vocab(reference), evidence_vocab(current)
    vocab_union = ref_vocab | cur_vocab
    vocab_jaccard = len(ref_vocab & cur_vocab) / len(vocab_union) if vocab_union else 1.0
    ref_yield = len(reference)
    cur_yield = len(current)
    ratio = cur_yield / ref_yield if ref_yield else None
    warnings = []
    if ref_labels and jaccard < 0.35:
        warnings.append("relation_vocabulary_shift")
    if ref_vocab and vocab_jaccard < 0.25:
        warnings.append("evidence_vocabulary_shift")
    if ratio is not None and (ratio < 0.5 or ratio > 2.0):
        warnings.append("candidate_yield_shift")
    return {"relation_label_jaccard": round(jaccard, 4), "evidence_vocabulary_jaccard": round(vocab_jaccard, 4),
            "new_relation_labels": new_relations, "missing_relation_labels": lost_relations,
            "reference_candidate_count": ref_yield, "current_candidate_count": cur_yield,
            "candidate_count_ratio": round(ratio, 4) if ratio is not None else None, "warnings": warnings,
            "interpretation": "Descriptive shift signals only. They indicate changed vocabulary or extraction behaviour; they do not prove low accuracy. Recalibrate and measure on labelled examples from the new data before enabling automatic promotion."}
