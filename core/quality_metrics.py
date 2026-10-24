"""Small transparent metrics for a manually labelled relation-extraction set."""
from __future__ import annotations
from typing import Any, Iterable
import re


def norm(value: str) -> str:
    value = str(value or "").casefold().strip()
    value = re.sub(r"[^\w\s-]", "", value)
    return re.sub(r"\s+", " ", value)


def triple_key(subject: str, relation_type: str, object_: str) -> tuple[str, str, str]:
    return norm(subject), norm(relation_type), norm(object_)


def relation_extraction_metrics(predicted: Iterable[Any], gold: Iterable[dict[str, str]]) -> dict[str, Any]:
    """Exact normalized triple precision/recall/F1, deduplicated by triple.

    Use on a manually reviewed test set only. This metric says nothing about
    unlabelled relations or general performance outside this corpus.
    """
    pred_set = set()
    for item in predicted:
        if isinstance(item, dict):
            pred_set.add(triple_key(item["subject"], item["relation_type"], item["object"]))
        else:
            pred_set.add(triple_key(item.subject, item.relation_type, item.object))
    gold_set = {triple_key(item["subject"], item["relation_type"], item["object"]) for item in gold}
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "metric": "exact_normalized_relation_triple_precision_recall_f1",
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "predicted_unique_triples": len(pred_set),
        "gold_unique_triples": len(gold_set),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "scope_note": "Scores are valid only for this manually reviewed demo corpus and exact normalized triples; they are not a general accuracy guarantee.",
        "false_positive_examples": [list(t) for t in sorted(pred_set - gold_set)[:20]],
        "false_negative_examples": [list(t) for t in sorted(gold_set - pred_set)[:20]],
    }
