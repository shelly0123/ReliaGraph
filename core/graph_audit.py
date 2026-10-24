"""Explainable structural/provenance checks for the ReliaGraph graph.

Findings are review suggestions, not automatically verified real-world facts.
"""
from __future__ import annotations
from collections.abc import Sequence
from difflib import SequenceMatcher
from itertools import combinations
from models import Chunk, GraphSnapshot, AuditFinding, TopicCoverage, chunk_from
from core.graph_builder import to_networkx, stable_id, normalize_label

OPPOSITE_RELATIONS = {
    frozenset(("requires", "does_not_require")),
    frozenset(("allows", "does_not_allow")),
    frozenset(("allows", "prohibits")),
    frozenset(("increases", "decreases")),
    frozenset(("supports", "contradicts")),
}
# These predicates normally describe one value per subject within a given scope.
# They are intentionally few; broader rules would create false conflict reports.
FUNCTIONAL_RELATIONS = {"has_deadline", "has_threshold", "has_effective_date", "policy_version_is"}
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


def _finding(category: str, severity: str, affected: list[str], evidence: list[str], message: str, recommendation: str) -> AuditFinding:
    return AuditFinding(
        finding_id=stable_id("finding", category, *sorted(affected)), category=category,
        severity=severity, affected_ids=sorted(set(affected)), evidence_chunk_ids=sorted(set(evidence)),
        message=message, recommendation=recommendation,
    )


def audit_graph(snapshot: GraphSnapshot, chunks: Sequence[Chunk | dict], fuzzy_threshold: float = 0.91) -> list[AuditFinding]:
    normalized_chunks = [chunk_from(c) for c in chunks]
    chunk_ids = {c.chunk_id for c in normalized_chunks}
    node_by_id = {n.node_id: n for n in snapshot.nodes}
    graph = to_networkx(snapshot)
    findings: list[AuditFinding] = []

    for edge in snapshot.edges:
        missing = sorted(set(edge.evidence_chunk_ids) - chunk_ids)
        if not edge.evidence_chunk_ids and edge.relation_type not in {"belongs_to_topic", "has_subtopic"}:
            findings.append(_finding("missing_edge_evidence", "high", [edge.edge_id], [],
                f"Relationship '{edge.relation_type}' has no source-chunk reference.",
                "Attach valid source evidence or keep the relationship marked as unreviewed."))
        elif missing:
            findings.append(_finding("invalid_evidence_reference", "high", [edge.edge_id], missing,
                "Relationship refers to chunk IDs not present in the current chunk collection.",
                "Refresh the edge's source references after re-chunking or removing documents."))
        if edge.source_node_id not in node_by_id or edge.target_node_id not in node_by_id:
            findings.append(_finding("invalid_edge_endpoint", "high", [edge.edge_id], edge.evidence_chunk_ids,
                "Relationship endpoint is missing from the graph snapshot.",
                "Rebuild the edge using valid node IDs before displaying or evaluating the graph."))
        if edge.confidence is not None and not 0.0 <= edge.confidence <= 1.0:
            findings.append(_finding("invalid_confidence_value", "medium", [edge.edge_id], edge.evidence_chunk_ids,
                "Extraction confidence is outside the expected 0–1 range.",
                "Inspect the extractor output. Confidence is a model/rule signal, not a calibrated truth probability."))

    for node in snapshot.nodes:
        missing = sorted(set(node.evidence_chunk_ids) - chunk_ids)
        if missing:
            findings.append(_finding("invalid_node_evidence", "medium", [node.node_id], missing,
                f"Node '{node.label}' references unavailable source chunks.",
                "Update evidence references when source chunks or document versions change."))
        if node.node_type != "topic" and graph.degree(node.node_id) == 0:
            findings.append(_finding("isolated_node", "low", [node.node_id], node.evidence_chunk_ids,
                f"Concept '{node.label}' is not connected to another graph node.",
                "Check whether a relationship was missed or whether the concept should remain standalone."))

    # Fuzzy duplicates are candidate pairs, never automatic merges.
    concepts = [n for n in snapshot.nodes if n.node_type != "topic" and normalize_label(n.label)]
    for left, right in combinations(concepts, 2):
        a, b = normalize_label(left.label), normalize_label(right.label)
        if a == b or SequenceMatcher(None, a, b).ratio() >= fuzzy_threshold:
            # Exact same normalized text but different node IDs is worth flagging too.
            findings.append(_finding("possible_duplicate_entity", "low", [left.node_id, right.node_id],
                left.evidence_chunk_ids + right.evidence_chunk_ids,
                f"'{left.label}' and '{right.label}' may refer to the same entity (string similarity {SequenceMatcher(None, a, b).ratio():.2f}).",
                "Review both source passages and entity types; do not merge solely because the labels look similar."))

    # Relation opposites: only same ordered subject-object pair and explicit cue pair.
    edge_by_pair_relation = {}
    for edge in snapshot.edges:
        key = (edge.source_node_id, edge.target_node_id, edge.relation_type.casefold())
        edge_by_pair_relation.setdefault(key, []).append(edge)
    seen_pairs = set()
    for edge in snapshot.edges:
        rel = edge.relation_type.casefold()
        for opposite_pair in OPPOSITE_RELATIONS:
            if rel not in opposite_pair:
                continue
            other_rel = next(r for r in opposite_pair if r != rel)
            for other in edge_by_pair_relation.get((edge.source_node_id, edge.target_node_id, other_rel), []):
                pair_key = tuple(sorted((edge.edge_id, other.edge_id)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                findings.append(_finding("possible_conflicting_claims", "high", list(pair_key),
                    edge.evidence_chunk_ids + other.evidence_chunk_ids,
                    f"The same entity pair has opposing relationship labels: '{edge.relation_type}' and '{other.relation_type}'.",
                    "Compare source passages, dates, scope and authority. Do not resolve the conflict automatically."))

    # Functional values, e.g. a policy's deadline, differ across evidence-backed claims.
    functional_groups = {}
    for edge in snapshot.edges:
        rel = edge.relation_type.casefold()
        if rel not in FUNCTIONAL_RELATIONS or not edge.evidence_chunk_ids:
            continue
        functional_groups.setdefault((edge.source_node_id, rel), []).append(edge)
    for (subject_id, rel), group in functional_groups.items():
        distinct_targets = {}
        for edge in group:
            distinct_targets.setdefault(edge.target_node_id, []).append(edge)
        if len(distinct_targets) > 1:
            edges = [e for subgroup in distinct_targets.values() for e in subgroup]
            ids = [e.edge_id for e in edges]
            evid = [cid for e in edges for cid in e.evidence_chunk_ids]
            subject_label = node_by_id.get(subject_id).label if subject_id in node_by_id else subject_id
            target_labels = sorted(node_by_id[e.target_node_id].label for e in edges if e.target_node_id in node_by_id)
            findings.append(_finding("possible_functional_value_conflict", "high", ids, evid,
                f"'{subject_label}' has multiple values for '{rel}': {', '.join(sorted(set(target_labels)))}.",
                "Check document versions, publication dates, jurisdiction and which source is authoritative."))

    unique = {f.finding_id: f for f in findings}
    return sorted(unique.values(), key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.category, f.finding_id))


def calculate_topic_coverage(snapshot: GraphSnapshot, chunks: Sequence[Chunk | dict]) -> list[TopicCoverage]:
    """Transparent structural coverage counters; not a semantic completeness score."""
    normalized_chunks = [chunk_from(c) for c in chunks]
    chunk_by_id = {c.chunk_id: c for c in normalized_chunks}
    entities = [n for n in snapshot.nodes if n.node_type != "topic"]
    result = []
    for topic in snapshot.topics:
        topic_chunk_ids = set(topic.chunk_ids) | {c.chunk_id for c in normalized_chunks if topic.topic_id in c.topic_ids}
        topic_chunk_ids &= set(chunk_by_id)
        topic_nodes = [n for n in entities if topic.topic_id in n.topic_ids]
        node_ids = {n.node_id for n in topic_nodes}
        # Count only actual semantic relations, not topic-membership/hierarchy edges.
        topic_edges = [e for e in snapshot.edges if e.relation_type not in {"belongs_to_topic", "has_subtopic"}
                       and (e.source_node_id in node_ids or e.target_node_id in node_ids)]
        if not topic_chunk_ids:
            status = "no_source_chunks"
        elif not topic_nodes:
            status = "no_extracted_entities"
        elif not topic_edges:
            status = "few_or_no_relationships"
        else:
            status = "has_basic_coverage"
        result.append(TopicCoverage(topic.topic_id, topic.label, len(topic_chunk_ids), len(topic_nodes), len(topic_edges), status))
    return result
