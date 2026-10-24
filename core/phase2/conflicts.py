"""Open-world conflict handling: preserve competing claims and every source."""
from __future__ import annotations
from collections import defaultdict
import re
from typing import Any, Mapping, Sequence

def _n(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.-]", "", str(value or "").casefold(), flags=re.UNICODE)).strip()

def detect_competing_claims(claims: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group same subject+predicate with different objects across sources.

    Never chooses one claim as correct. Since many predicates can be multi-valued, the
    output says `competing_claims_requires_review`, not automatically `contradiction`.
    """
    groups = defaultdict(list)
    for c in claims:
        s, p, o = _n(c.get("subject", "")), _n(c.get("relation_type", c.get("relation", ""))), _n(c.get("object", ""))
        if s and p and o:
            groups[(s, p)].append(dict(c))
    result = []
    for (s, p), rows in groups.items():
        objects = {_n(r.get("object", "")) for r in rows}
        source_ids = {str(r.get("doc_id", r.get("source_url", ""))) for r in rows}
        if len(objects) > 1 and len(source_ids) > 1:
            result.append({"category": "competing_claims_requires_review", "subject": rows[0].get("subject", s),
                           "relation_type": rows[0].get("relation_type", rows[0].get("relation", p)),
                           "claims": [{"object": r.get("object", ""), "doc_id": r.get("doc_id", ""),
                                       "source_title": r.get("source_title", ""), "source_url": r.get("source_url", ""),
                                       "chunk_id": r.get("chunk_id", ""), "evidence_text": r.get("evidence_text", ""),
                                       "timestamp": r.get("timestamp", ""), "status": r.get("status", "candidate")} for r in rows],
                           "recommendation": "Show all claims with their sources; compare dates, authority and scope. Never silently overwrite a claim."})
    return result
