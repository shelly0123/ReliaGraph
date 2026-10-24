"""Quality regression, injection and retrieval metrics for Phase 2 acceptance tests."""
from __future__ import annotations
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence
import re

def token_f1(reference: str, prediction: str) -> float:
    ref = re.findall(r"[\w]+", str(reference or "").casefold(), flags=re.UNICODE)
    pred = re.findall(r"[\w]+", str(prediction or "").casefold(), flags=re.UNICODE)
    if not ref and not pred: return 1.0
    if not ref or not pred: return 0.0
    c1, c2 = Counter(ref), Counter(pred)
    common = sum((c1 & c2).values())
    if common == 0: return 0.0
    precision = common / len(pred); recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)

def exact_match(reference: str, prediction: str) -> float:
    norm = lambda x: " ".join(re.findall(r"[\w]+", str(x or "").casefold(), flags=re.UNICODE))
    return float(norm(reference) == norm(prediction))

def quality_report(rows: Sequence[Mapping[str, Any]], max_drop_points: float = 3.0) -> dict[str, Any]:
    """Compare Phase 1 and Phase 2 outputs on identical cases.

    `phase1_answer` and `phase2_answer` are checked against the same `reference_answer`.
    Scores are percentage points (0-100), not relative percent. The result is an honest
    regression check on this dataset, not a generalization guarantee.
    """
    if not rows: raise ValueError("At least one evaluation row is required")
    phase1, phase2 = [], []
    details = []
    for row in rows:
        ref = str(row.get("reference_answer", ""))
        a1 = str(row.get("phase1_answer", "")); a2 = str(row.get("phase2_answer", ""))
        m1, m2 = token_f1(ref, a1), token_f1(ref, a2)
        phase1.append(m1); phase2.append(m2)
        details.append({"test_id": row.get("test_id", ""), "phase1_token_f1": round(m1, 4),
                        "phase2_token_f1": round(m2, 4), "delta_points": round((m2-m1)*100, 2)})
    p1 = sum(phase1) / len(phase1); p2 = sum(phase2) / len(phase2)
    drop = max(0.0, (p1-p2)*100)
    return {"metric": "token_f1", "test_count": len(rows), "phase1_score": round(p1*100, 2),
            "phase2_score": round(p2*100, 2), "change_points": round((p2-p1)*100, 2),
            "degradation_points": round(drop, 2), "max_allowed_degradation_points": max_drop_points,
            "within_phase1_quality_budget": drop <= max_drop_points,
            "warning": "Only valid for the supplied paired test set; this does not prove quality on unseen data.",
            "details": details}

def retrieval_metrics(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int | None = None) -> dict[str, float | int]:
    ranked = list(map(str, retrieved_ids[:k] if k is not None else retrieved_ids))
    gold = set(map(str, gold_ids))
    found = set(ranked) & gold
    precision = len(found) / len(ranked) if ranked else 0.0
    recall = len(found) / len(gold) if gold else 0.0
    return {"precision_at_k": round(precision, 4), "recall_at_k": round(recall, 4),
            "f1": round(2*precision*recall/(precision+recall), 4) if precision+recall else 0.0,
            "hits": len(found), "retrieved": len(ranked), "gold": len(gold)}

def injection_metrics(expected_attacks: int, detected_attacks: int, benign_total: int, benign_flagged: int) -> dict[str, Any]:
    return {"attack_cases": expected_attacks, "detected_attacks": detected_attacks,
            "attack_detection_recall": round(detected_attacks/expected_attacks, 4) if expected_attacks else 0.0,
            "benign_cases": benign_total, "benign_false_positives": benign_flagged,
            "benign_false_positive_rate": round(benign_flagged/benign_total, 4) if benign_total else 0.0,
            "note": "Known-case detector metrics only; not proof of prompt-injection immunity."}
