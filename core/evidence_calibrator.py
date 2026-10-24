"""Optional learned calibration for relation-candidate triage.

This is a small logistic model over evidence/process features, NOT a new entity/relation
extractor. It may only be trained on human-reviewed candidates with source-grouped splits.
By default the pipeline must abstain/review unless there is enough labelled evidence.
Pure Python implementation keeps the training utility dependency-light.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import csv, hashlib, json, math

FEATURES = (
    "source_anchor_valid",
    "quote_present",
    "extractor_score",
    "independent_extractor_count",
    "independent_source_count",
    "nli_entailment",
    "nli_contradiction",
    "nli_neutral",
    "injection_flag",
    "domain_shift_flag",
)


def _sigmoid(x: float) -> float:
    if x >= 35: return 1.0
    if x <= -35: return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _doc_bucket(doc_id: str, buckets: int = 5) -> int:
    digest = hashlib.sha256(str(doc_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % buckets


@dataclass
class LogisticCalibrationModel:
    features: list[str]
    means: list[float]
    scales: list[float]
    weights: list[float]
    bias: float
    label: str = "human_reviewed_supported_claim"
    trained_on_rows: int = 0
    trained_domains: list[str] | None = None
    model_kind: str = "logistic_calibration_not_relation_extraction"

    def predict_proba(self, row: Mapping[str, Any]) -> float:
        values = [float(row.get(name, 0.0) or 0.0) for name in self.features]
        z = self.bias
        for v, mean, scale, weight in zip(values, self.means, self.scales, self.weights):
            z += ((v - mean) / (scale or 1.0)) * weight
        return _sigmoid(z)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LogisticCalibrationModel":
        return cls(features=list(data["features"]), means=list(data["means"]), scales=list(data["scales"]),
                   weights=list(data["weights"]), bias=float(data["bias"]), label=str(data.get("label", "human_reviewed_supported_claim")),
                   trained_on_rows=int(data.get("trained_on_rows", 0)), trained_domains=list(data.get("trained_domains") or []),
                   model_kind=str(data.get("model_kind", "logistic_calibration_not_relation_extraction")))


def _features(row: Mapping[str, Any]) -> list[float]:
    return [float(row.get(k, 0.0) or 0.0) for k in FEATURES]


def _normalization(x_rows: list[list[float]]) -> tuple[list[float], list[float]]:
    if not x_rows: return [0.0] * len(FEATURES), [1.0] * len(FEATURES)
    means = [sum(row[j] for row in x_rows) / len(x_rows) for j in range(len(FEATURES))]
    scales = []
    for j, mean in enumerate(means):
        variance = sum((row[j] - mean) ** 2 for row in x_rows) / max(1, len(x_rows) - 1)
        sd = math.sqrt(variance)
        scales.append(sd if sd > 1e-9 else 1.0)
    return means, scales


def _fit_logistic(x_rows: list[list[float]], y: list[int], means: list[float], scales: list[float],
                  epochs: int = 800, learning_rate: float = 0.08, l2: float = 0.03) -> tuple[list[float], float]:
    n, d = len(x_rows), len(FEATURES)
    xs = [[(v - means[j]) / scales[j] for j, v in enumerate(row)] for row in x_rows]
    weights = [0.0] * d
    bias = 0.0
    for epoch in range(epochs):
        grad = [0.0] * d
        grad_b = 0.0
        for x, label in zip(xs, y):
            p = _sigmoid(sum(a*b for a, b in zip(x, weights)) + bias)
            err = p - label
            for j in range(d): grad[j] += err * x[j]
            grad_b += err
        # Diminishing step size to keep small dataset training stable.
        lr = learning_rate / (1.0 + epoch / 400.0)
        for j in range(d):
            weights[j] -= lr * (grad[j] / n + l2 * weights[j])
        bias -= lr * grad_b / n
    return weights, bias


def _metric(rows: list[Mapping[str, Any]], model: LogisticCalibrationModel, threshold: float = 0.5) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    brier_sum = 0.0
    for row in rows:
        prob = model.predict_proba(row)
        y = int(str(row.get("is_supported", "0")).casefold() in _POSITIVE_LABELS)
        pred = prob >= threshold
        tp += int(pred and y == 1); fp += int(pred and y == 0)
        tn += int((not pred) and y == 0); fn += int((not pred) and y == 1)
        brier_sum += (prob - y) ** 2
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"rows": len(rows), "threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2*precision*recall/max(1e-12, precision+recall), 4),
            "brier_score": round(brier_sum / max(1, len(rows)), 4)}


_POSITIVE_LABELS = {"1", "true", "yes", "supported", "accept"}
_NEGATIVE_LABELS = {"0", "false", "no", "unsupported", "contradicted", "unclear", "insufficient", "reject"}


def _read_review_csv(path: str | Path) -> tuple[list[dict[str, str]] | None, dict[str, Any] | None]:
    """Strictly validate human-reviewed rows; malformed data must fail closed."""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows: list[dict[str, str]] = list(csv.DictReader(f))
    required = {"doc_id", "domain", "is_supported", *FEATURES}
    if not rows or not required.issubset(rows[0].keys()):
        return None, {"reason": "missing_or_empty_required_columns", "required_columns": sorted(required)}
    bounded = {"source_anchor_valid", "quote_present", "extractor_score", "nli_entailment",
               "nli_contradiction", "nli_neutral", "injection_flag", "domain_shift_flag"}
    flags = {"source_anchor_valid", "quote_present", "injection_flag", "domain_shift_flag"}
    counts = {"independent_extractor_count", "independent_source_count"}
    for row_ix, row in enumerate(rows, start=2):
        if not str(row.get("doc_id", "")).strip():
            return None, {"reason": "blank_doc_id", "row": row_ix}
        if not str(row.get("domain", "")).strip():
            return None, {"reason": "blank_domain", "row": row_ix}
        label = str(row.get("is_supported", "")).strip().casefold()
        if label not in _POSITIVE_LABELS | _NEGATIVE_LABELS:
            return None, {"reason": "invalid_review_label", "row": row_ix,
                          "allowed_positive": sorted(_POSITIVE_LABELS),
                          "allowed_negative": sorted(_NEGATIVE_LABELS)}
        row["is_supported"] = "1" if label in _POSITIVE_LABELS else "0"
        for feature in FEATURES:
            try:
                value = float(row.get(feature, ""))
            except (TypeError, ValueError):
                return None, {"reason": "non_numeric_feature", "row": row_ix, "feature": feature}
            if not math.isfinite(value):
                return None, {"reason": "non_finite_feature", "row": row_ix, "feature": feature}
            if feature in bounded and not 0.0 <= value <= 1.0:
                return None, {"reason": "feature_out_of_range", "row": row_ix, "feature": feature, "expected": "0..1"}
            if feature in counts and value < 0.0:
                return None, {"reason": "negative_count_feature", "row": row_ix, "feature": feature}
            if feature in flags and value not in {0.0, 1.0}:
                return None, {"reason": "non_binary_flag", "row": row_ix, "feature": feature}
            row[feature] = str(value)
    return rows, None


def train_from_csv(path: str | Path, *, min_rows: int = 100, min_domains: int = 3,
                   test_bucket: int = 0, buckets: int = 5) -> dict[str, Any]:
    """Fit a review calibrator from a labeled CSV with independent source-group holdout.

    CSV fields: doc_id, domain, is_supported, and all columns in FEATURES. The split is
    by doc_id hash so chunks from one source cannot leak into both partitions. The
    function deliberately refuses tiny/single-domain data instead of inventing quality.
    """
    rows, validation_error = _read_review_csv(path)
    if validation_error:
        return {"trained": False, **validation_error}
    assert rows is not None
    domains = sorted({str(r["domain"]).strip() for r in rows})
    labels = [str(r["is_supported"]).casefold() in _POSITIVE_LABELS for r in rows]
    if len(rows) < min_rows:
        return {"trained": False, "reason": "insufficient_reviewed_rows", "rows": len(rows), "minimum_rows": min_rows}
    if len(domains) < min_domains:
        return {"trained": False, "reason": "insufficient_domain_diversity", "domains": domains, "minimum_domains": min_domains}
    if len(set(labels)) < 2:
        return {"trained": False, "reason": "labels_need_supported_and_unsupported_examples", "positive_count": sum(labels), "negative_count": len(labels)-sum(labels)}

    train_rows, test_rows = [], []
    for row in rows:
        (test_rows if _doc_bucket(row["doc_id"], buckets) == test_bucket else train_rows).append(row)
    train_domains = {str(r["domain"]).strip() for r in train_rows}
    test_domains = {str(r["domain"]).strip() for r in test_rows}
    train_labels = [int(str(r["is_supported"]).casefold() in {"1", "true", "yes", "supported", "accept"}) for r in train_rows]
    if not test_rows or len(set(train_labels)) < 2:
        return {"trained": False, "reason": "invalid_grouped_holdout_or_one_class_training", "train_rows": len(train_rows), "test_rows": len(test_rows)}
    x = [_features(r) for r in train_rows]
    means, scales = _normalization(x)
    weights, bias = _fit_logistic(x, train_labels, means, scales)
    model = LogisticCalibrationModel(list(FEATURES), means, scales, weights, bias,
                                     trained_on_rows=len(train_rows), trained_domains=sorted(train_domains))
    metrics = _metric(test_rows, model)
    return {"trained": True, "model": model.to_dict(),
            "split": {"train_rows": len(train_rows), "test_rows": len(test_rows),
                      "train_domains": sorted(train_domains), "test_domains": sorted(test_domains), "grouped_by_doc_id": True},
            "holdout_metrics": metrics,
            "warning": "This calibrates candidate triage; it does not extract relations. Holdout performance is evidence only for the tested source/domain distribution, not a guarantee on arbitrary new data."}



def evaluate_leave_one_domain_out(path: str | Path, *, min_rows: int = 100,
                                  min_train_rows: int = 30, min_domains: int = 3) -> dict[str, Any]:
    """Evaluate the *triage calibrator* on unseen domains, keeping domains disjoint.

    This is deliberately separate from `train_from_csv`, whose source-group split is
    useful when the intended deployment domains overlap. Leave-one-domain-out is a
    stricter test for domain transfer. Only manually reviewed labels should be used.
    It does not train a relation extractor and does not prove generalization beyond the
    domains represented in the CSV.
    """
    rows, validation_error = _read_review_csv(path)
    if validation_error:
        return {"evaluated": False, **validation_error}
    assert rows is not None
    domains = sorted({str(r["domain"]).strip() for r in rows})
    if len(rows) < min_rows:
        return {"evaluated": False, "reason": "insufficient_reviewed_rows", "rows": len(rows), "minimum_rows": min_rows}
    if len(domains) < min_domains:
        return {"evaluated": False, "reason": "insufficient_domain_diversity", "domains": domains, "minimum_domains": min_domains}

    folds = []
    for heldout_domain in domains:
        train_rows = [r for r in rows if str(r.get("domain", "")).strip() != heldout_domain]
        test_rows = [r for r in rows if str(r.get("domain", "")).strip() == heldout_domain]
        train_y = [int(str(r.get("is_supported", "0")).casefold() in _POSITIVE_LABELS) for r in train_rows]
        test_y = [int(str(r.get("is_supported", "0")).casefold() in _POSITIVE_LABELS) for r in test_rows]
        if len(train_rows) < min_train_rows or not test_rows or len(set(train_y)) < 2:
            folds.append({"heldout_domain": heldout_domain, "evaluated": False,
                          "reason": "insufficient_train_rows_or_training_classes",
                          "train_rows": len(train_rows), "test_rows": len(test_rows)})
            continue
        x = [_features(r) for r in train_rows]
        means, scales = _normalization(x)
        weights, bias = _fit_logistic(x, train_y, means, scales)
        model = LogisticCalibrationModel(list(FEATURES), means, scales, weights, bias,
                                         trained_on_rows=len(train_rows),
                                         trained_domains=sorted({str(r.get("domain", "")).strip() for r in train_rows}))
        metrics = _metric(test_rows, model)
        folds.append({"heldout_domain": heldout_domain, "evaluated": True,
                      "train_rows": len(train_rows), "test_rows": len(test_rows),
                      "train_domains": sorted({str(r.get("domain", "")).strip() for r in train_rows}),
                      "test_domain": heldout_domain, "metrics": metrics})
    valid_folds = [f for f in folds if f.get("evaluated")]
    macro = None
    if valid_folds:
        metric_names = ["precision", "recall", "f1", "brier_score"]
        macro = {name: round(sum(float(f["metrics"][name]) for f in valid_folds) / len(valid_folds), 4)
                 for name in metric_names}
    return {"evaluated": bool(valid_folds), "split_strategy": "leave_one_domain_out",
            "rows": len(rows), "domains": domains, "evaluated_folds": len(valid_folds),
            "folds": folds, "macro_metrics": macro,
            "warning": "This tests a logistic candidate-triage model on held-out domains using human-reviewed labels. It is not the relation extractor itself, and performance does not guarantee accuracy on arbitrary future domains."}

def save_training_result(result: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(result), indent=2, ensure_ascii=False), encoding="utf-8")
