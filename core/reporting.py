"""JSON-safe report helpers for the dashboard and demo."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from typing import Any


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return value


def summarize_findings(findings) -> dict[str, Any]:
    counts = {"high": 0, "medium": 0, "low": 0, "info": 0}
    categories: dict[str, int] = {}
    for item in findings:
        counts[item.severity] = counts.get(item.severity, 0) + 1
        categories[item.category] = categories.get(item.category, 0) + 1
    return {"total": len(findings), "by_severity": counts, "by_category": dict(sorted(categories.items()))}
