"""Domain-agnostic graph checks that do not assume a fixed relation vocabulary."""
from __future__ import annotations
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any, Sequence

from models import Chunk, GraphSnapshot, AuditFinding, chunk_from
from core.graph_builder import stable_id, normalize_label, to_networkx
from core.domain_agnostic_extraction import text_contains_span

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


def _finding(category: str, severity: str, affected: list[str], evidence: list[str], message: str, recommendation: str) -> AuditFinding:
    return AuditFinding(finding_id=stable_id("finding", category, *sorted(affected)), category=category,
                        severity=severity, affected_ids=sorted(set(affected)),
                        evidence_chunk_ids=sorted(set(evidence)), message=message, recommendation=recommendation)


def audit_graph_general(snapshot: GraphSnapshot, chunks: Sequence[Chunk | dict[str, Any]], fuzzy_threshold: float = 0.92) -> list[AuditFinding]:
    """Run structural, provenance, duplicate-candidate and competing-claim checks.

    Unlike the older MUJ-focused audit, this does not hard-code predicates such as
    `has_deadline` or `has_threshold`. Different objects for a subject-predicate pair
    are called *competing claims*, not contradictions, since many relations are naturally
    multi-valued. Source conflicts require a human or an evidence model to assess scope,
    time and authority.
    """
    normalized_chunks = [chunk_from(c) for c in chunks]
    chunk_by_id = {c.chunk_id: c for c in normalized_chunks}
    valid_chunk_ids = set(chunk_by_id)
    node_by_id = {n.node_id: n for n in snapshot.nodes}
    graph = to_networkx(snapshot)
    findings: list[AuditFinding] = []

    for edge in snapshot.edges:
        missing = sorted(set(edge.evidence_chunk_ids) - valid_chunk_ids)
        is_structural = edge.relation_type in {"belongs_to_topic", "has_subtopic"} or edge.status == "derived"
        if not edge.evidence_chunk_ids and not is_structural:
            findings.append(_finding("missing_edge_evidence", "high", [edge.edge_id], [],
                f"Relationship '{edge.relation_type}' has no source-chunk reference.",
                "Do not promote this relationship. Attach valid source evidence or keep it in the candidate/review queue."))
        elif missing:
            findings.append(_finding("invalid_evidence_reference", "high", [edge.edge_id], missing,
                "Relationship refers to chunk IDs absent from the current source collection.",
                "Reconcile evidence references after document removal or re-chunking."))
        if edge.source_node_id not in node_by_id or edge.target_node_id not in node_by_id:
            findings.append(_finding("invalid_edge_endpoint", "high", [edge.edge_id], edge.evidence_chunk_ids,
                "Relationship endpoint is missing from the snapshot.", "Rebuild the edge using valid node IDs."))
        if edge.confidence is not None and not 0 <= float(edge.confidence) <= 1:
            findings.append(_finding("invalid_score", "medium", [edge.edge_id], edge.evidence_chunk_ids,
                "Extractor score is outside the expected 0–1 range.", "Inspect the extractor output; model scores are not necessarily calibrated probabilities."))

        # Validate provenance if a source quote was saved on the edge.
        quotes = (edge.attributes or {}).get("evidence_quotes", [])
        if quotes:
            valid_quotes = [q for q in quotes if any(text_contains_span(chunk_by_id[cid].text, q)
                            for cid in edge.evidence_chunk_ids if cid in chunk_by_id)]
            if not valid_quotes:
                findings.append(_finding("quote_not_aligned_to_source", "high", [edge.edge_id], edge.evidence_chunk_ids,
                    "Stored evidence quote could not be found in its referenced source chunks.",
                    "Keep the relationship unverified and inspect offsets/normalization or the extractor output."))

    for node in snapshot.nodes:
        missing = sorted(set(node.evidence_chunk_ids) - valid_chunk_ids)
        if missing:
            findings.append(_finding("invalid_node_evidence", "medium", [node.node_id], missing,
                f"Node '{node.label}' references absent source chunks.", "Refresh source references after chunk updates."))
        if node.node_type != "topic" and graph.degree(node.node_id) == 0:
            findings.append(_finding("isolated_node", "low", [node.node_id], node.evidence_chunk_ids,
                f"Concept '{node.label}' has no graph connections.",
                "Review whether a relationship was missed or the concept is legitimately standalone."))

    concepts = [(n, normalize_label(n.label)) for n in snapshot.nodes
                if n.node_type != "topic" and normalize_label(n.label)]
    for (left, a), (right, b) in combinations(concepts, 2):
        if a == b:
            similarity = 1.0
        else:
            # Safe upper-bound filters keep the O(n^2) candidate scan cheap for larger
            # corpora. SequenceMatcher's quick_ratio is an upper bound on ratio(), so
            # pairs rejected here could not have crossed fuzzy_threshold anyway.
            max_len = max(len(a), len(b), 1)
            if 2 * min(len(a), len(b)) / (len(a) + len(b) or 1) < fuzzy_threshold:
                continue
            matcher = SequenceMatcher(None, a, b, autojunk=False)
            if matcher.quick_ratio() < fuzzy_threshold:
                continue
            similarity = matcher.ratio()
        if similarity >= fuzzy_threshold:
            findings.append(_finding("possible_duplicate_entity", "low", [left.node_id, right.node_id],
                left.evidence_chunk_ids + right.evidence_chunk_ids,
                f"'{left.label}' and '{right.label}' may be duplicate mentions (surface similarity {similarity:.2f}).",
                "Compare source context, entity type and optional embedding similarity; never merge on string similarity alone."))

    # Open-world competing-claim detection: deliberately does not assert contradiction.
    node_label = {n.node_id: n.label for n in snapshot.nodes}
    by_sp: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for edge in snapshot.edges:
        if edge.status == "derived" or edge.relation_type in {"belongs_to_topic", "has_subtopic"}:
            continue
        if not edge.evidence_chunk_ids:
            continue
        subject = normalize_label(node_label.get(edge.source_node_id, ""))
        predicate = normalize_label(edge.relation_type)
        if subject and predicate:
            by_sp[(subject, predicate)].append(edge)
    for (subject, predicate), edges in by_sp.items():
        target_groups: dict[str, list[Any]] = defaultdict(list)
        for edge in edges:
            target_groups[normalize_label(node_label.get(edge.target_node_id, ""))].append(edge)
        if len(target_groups) > 1:
            all_edges = [e for group in target_groups.values() for e in group]
            # Do not flag simple multi-valued relations when all targets are supported in
            # one chunk and no cross-source/time/version evidence is present.
            sources = {chunk_by_id[cid].source_url or chunk_by_id[cid].doc_id
                       for e in all_edges for cid in e.evidence_chunk_ids if cid in chunk_by_id}
            versions = {chunk_by_id[cid].source_version for e in all_edges for cid in e.evidence_chunk_ids if cid in chunk_by_id}
            if len(sources) > 1 or len(versions) > 1:
                findings.append(_finding("competing_claims_requires_review", "medium", [e.edge_id for e in all_edges],
                    [cid for e in all_edges for cid in e.evidence_chunk_ids],
                    f"Multiple objects are connected to subject '{subject}' with relation '{predicate}'. This may be valid multi-valued knowledge or a source/version conflict.",
                    "Compare evidence, source authority, dates, version and scope. Do not automatically treat this as a contradiction."))

    return sorted({f.finding_id: f for f in findings}.values(),
                  key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.category, f.finding_id))
