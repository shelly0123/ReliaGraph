"""Optional embedding-based duplicate suggestions and NLI evidence review."""
from __future__ import annotations
from itertools import combinations
from typing import Any

from models import AuditFinding, GraphSnapshot, Chunk
from core.graph_builder import stable_id, normalize_label
from core.model_runtime import LocalSentenceEmbedder, LocalNLIReviewer

RELATION_TO_CLAIM = {
    "is_a": "{subject} is a {object}.",
    "offers": "{subject} offers {object}.",
    "includes": "{subject} includes {object}.",
    "has": "{subject} has {object}.",
    "uses": "{subject} uses {object}.",
    "follows": "{subject} follows {object}.",
    "supports": "{subject} supports {object}.",
    "enables": "{subject} enables {object}.",
    "has_deadline": "The deadline for {subject} is {object}.",
    "has_threshold": "The threshold for {subject} is {object}.",
    "requires": "{subject} requires {object}.",
    "protects": "{subject} protects {object}.",
    "reduces": "{subject} reduces {object}.",
    "decreases": "{subject} decreases {object}.",
}


def semantic_duplicate_candidates(
    snapshot: GraphSnapshot,
    embedder: LocalSentenceEmbedder,
    threshold: float = 0.88,
    max_nodes: int = 500,
) -> list[AuditFinding]:
    """Return suggestions only; never merge entities automatically.

    Work is bounded to max_nodes to prevent accidental O(n^2) explosion. Exact
    label duplicates should already be merged by the deterministic graph builder.
    """
    candidates = [n for n in snapshot.nodes if n.node_type != "topic" and normalize_label(n.label)]
    if len(candidates) > max_nodes:
        candidates = candidates[:max_nodes]
    if len(candidates) < 2:
        return []
    vectors = embedder.encode([n.label for n in candidates])
    findings: list[AuditFinding] = []
    # Vectorize the pairwise calculation. The default cap bounds memory and time.
    import numpy as np
    matrix = np.asarray(vectors, dtype=np.float32)
    similarity_matrix = matrix @ matrix.T
    left_indices, right_indices = np.where(np.triu(similarity_matrix >= threshold, k=1))
    for i, j in zip(left_indices.tolist(), right_indices.tolist()):
        a, b = candidates[i], candidates[j]
        if a.node_type != b.node_type:
            continue
        if normalize_label(a.label) == normalize_label(b.label):
            continue
        similarity = float(similarity_matrix[i, j])
        findings.append(AuditFinding(
            finding_id=stable_id("finding_semantic_duplicate", a.node_id, b.node_id),
            category="semantic_duplicate_candidate",
            severity="low",
            affected_ids=sorted([a.node_id, b.node_id]),
            evidence_chunk_ids=sorted(set(a.evidence_chunk_ids + b.evidence_chunk_ids)),
            message=(f"'{a.label}' and '{b.label}' have embedding cosine similarity "
                     f"{similarity:.3f} (threshold {threshold:.2f}); they may be aliases or related terms."),
            recommendation="Review entity types and source passages. Similarity alone is not proof of identity; do not auto-merge.",
        ))
    return sorted(findings, key=lambda f: (f.category, f.affected_ids))


def _edge_claim(subject: str, relation_type: str, object_: str) -> str:
    template = RELATION_TO_CLAIM.get(relation_type.casefold())
    if template:
        return template.format(subject=subject, object=object_)
    readable = relation_type.replace("_", " ")
    return f"The relationship between {subject} and {object_} is described as {readable}."


def review_edge_evidence_with_nli(
    snapshot: GraphSnapshot,
    chunks: list[Chunk],
    reviewer: LocalNLIReviewer,
    max_edges: int = 50,
    max_premise_chars: int = 700,
    contradiction_threshold: float = 0.70,
    neutral_threshold: float = 0.72,
) -> dict[str, Any]:
    """Run a bounded, optional NLI evidence check for the first eligible edges.

    NLI results are advisory. Contradiction/neutral scores create review findings,
    not automatic rejection of claims.
    """
    node_by_id = {node.node_id: node for node in snapshot.nodes}
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    edges = [e for e in snapshot.edges
             if e.evidence_chunk_ids
             and e.source_node_id in node_by_id and e.target_node_id in node_by_id
             and e.relation_type not in {"belongs_to_topic", "has_subtopic"}
             and e.status != "derived"]
    edges = edges[:max(1, int(max_edges))]
    jobs: list[tuple[Any, str, str, str, str]] = []
    for edge in edges:
        subject = node_by_id[edge.source_node_id].label
        object_ = node_by_id[edge.target_node_id].label
        claim = _edge_claim(subject, edge.relation_type, object_)
        premise = " ".join(
            chunk_by_id[cid].text.strip() for cid in edge.evidence_chunk_ids
            if cid in chunk_by_id and chunk_by_id[cid].text.strip()
        )[:max_premise_chars]
        if premise:
            jobs.append((edge, premise, claim, subject, object_))
    scores = reviewer.score_pairs([(job[1], job[2]) for job in jobs])
    findings: list[dict[str, Any]] = []
    for (edge, premise, claim, subject, object_), score in zip(jobs, scores):
        label = max(score, key=score.get)
        review_state = "model_support_signal" if label == "entailment" else "needs_human_review"
        findings.append({
            "edge_id": edge.edge_id,
            "subject": subject,
            "relation_type": edge.relation_type,
            "object": object_,
            "claim_checked": claim,
            "evidence_chunk_ids": edge.evidence_chunk_ids,
            "scores": score,
            "top_label": label,
            "review_state": review_state,
        })
        if score.get("contradiction", 0.0) >= contradiction_threshold:
            findings.append(AuditFinding(
                finding_id=stable_id("finding_nli_contradiction", edge.edge_id),
                category="nli_potential_contradiction", severity="medium",
                affected_ids=[edge.edge_id], evidence_chunk_ids=edge.evidence_chunk_ids,
                message=(f"Local NLI model found a contradiction signal for edge '{subject} -{edge.relation_type}-> {object_}' "
                         f"(score {score.get('contradiction', 0.0):.3f})."),
                recommendation="Verify the original source passage and claim manually; NLI is an imperfect model signal.",
            ))
        elif score.get("neutral", 0.0) >= neutral_threshold:
            findings.append(AuditFinding(
                finding_id=stable_id("finding_nli_neutral", edge.edge_id),
                category="nli_insufficient_support_signal", severity="low",
                affected_ids=[edge.edge_id], evidence_chunk_ids=edge.evidence_chunk_ids,
                message=(f"Local NLI model could not confidently infer the claim from the source (neutral score "
                         f"{score.get('neutral', 0.0):.3f})."),
                recommendation="Keep the relationship as a candidate until a reviewer confirms its supporting evidence.",
            ))
    audit_findings = [entry for entry in findings if isinstance(entry, AuditFinding)]
    result_rows = [entry for entry in findings if isinstance(entry, dict)]
    return {"checks": result_rows, "findings": audit_findings, "model": reviewer.diagnostics()}
