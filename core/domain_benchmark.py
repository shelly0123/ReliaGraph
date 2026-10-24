"""Cross-domain evaluation utilities. Scores are meaningful only for labelled test data."""
from __future__ import annotations
from collections import defaultdict
from typing import Any, Sequence

from core.selective_quality import canonical_triple, precision_recall_f1


def evaluate_by_domain(predictions_by_case: dict[str, Sequence[tuple[str, str, str]]],
                       cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate exact normalized triples overall and per domain; missing predictions are empty."""
    domain_gold: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    domain_pred: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    overall_gold: list[tuple[str, str, str]] = []
    overall_pred: list[tuple[str, str, str]] = []
    per_case = []
    for case in cases:
        cid, domain = str(case["case_id"]), str(case["domain"])
        gold = [tuple(x) for x in case.get("gold_triples", [])]
        pred = [tuple(x) for x in predictions_by_case.get(cid, [])]
        metrics = precision_recall_f1(pred, gold)
        per_case.append({"case_id": cid, "domain": domain, **metrics})
        domain_gold[domain].extend(gold); domain_pred[domain].extend(pred)
        overall_gold.extend(gold); overall_pred.extend(pred)
    per_domain = {domain: precision_recall_f1(domain_pred[domain], domain_gold[domain])
                  for domain in sorted(set(domain_gold) | set(domain_pred))}
    return {"overall": precision_recall_f1(overall_pred, overall_gold), "per_domain": per_domain,
            "per_case": per_case,
            "method": "exact_normalized_subject_predicate_object_triples",
            "scope_note": "Scores describe only this labelled set and extractor configuration. Domain-specific and held-out-source results must be reviewed; no arbitrary-domain guarantee is implied."}
