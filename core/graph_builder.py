"""Construct a topic/entity multigraph with source-linked relationships."""
from __future__ import annotations
from dataclasses import asdict
from collections import defaultdict
from collections.abc import Sequence
import hashlib
import json
import re
from typing import Any
import networkx as nx
from models import Chunk, Topic, EntityCandidate, RelationCandidate, Node, Edge, GraphSnapshot, chunk_from, topic_from
from core.selective_quality import canonical_triple, assess_candidate_evidence


def normalize_label(label: str) -> str:
    value = re.sub(r"\s+", " ", str(label).strip().casefold())
    return re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)


def stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def build_graph(
    chunks: Sequence[Chunk | dict],
    topics: Sequence[Topic | dict],
    entity_candidates: Sequence[EntityCandidate] = (),
    relation_candidates: Sequence[RelationCandidate] = (),
    document_hashes: dict[str, str] | None = None,
    candidate_decisions: Sequence[dict[str, Any]] | None = None,
) -> GraphSnapshot:
    """Create graph nodes/edges from Member 1's chunks/topics and extracted candidates.

    Exact-normalized labels of the same entity type are merged. Similar labels are
    never silently merged: the auditor emits suggestions for human review.
    Every extracted relationship keeps its source chunk IDs and quote(s).
    """
    normalized_chunks = [chunk_from(c) for c in chunks]
    normalized_topics = [topic_from(t) for t in topics]
    chunk_by_id = {c.chunk_id: c for c in normalized_chunks}
    topic_by_id = {t.topic_id: t for t in normalized_topics}

    decisions_by_triple: dict[tuple[str, str, str], dict[str, Any]] = {}
    if candidate_decisions is not None:
        for decision in candidate_decisions:
            key = canonical_triple(str(decision.get("subject", "")),
                                   str(decision.get("relation_type", "")),
                                   str(decision.get("object", "")))
            if all(key):
                decisions_by_triple[key] = dict(decision)

    rejected_relation_candidates = 0
    # Use mutable accumulators, then freeze into sorted dataclasses.
    node_acc: dict[str, Node] = {}
    edge_acc: dict[str, Edge] = {}
    entity_ids: dict[tuple[str, str], str] = {}
    entity_ids_by_norm: dict[str, list[str]] = defaultdict(list)
    entity_ids_by_norm_chunk: dict[tuple[str, str], list[str]] = defaultdict(list)

    # Topic nodes are stable if the upstream topic_id remains stable.
    for topic in normalized_topics:
        tid = f"topic:{topic.topic_id}"
        evidence = sorted(set(topic.chunk_ids) | {c.chunk_id for c in normalized_chunks if topic.topic_id in c.topic_ids})
        node_acc[tid] = Node(
            node_id=tid, label=topic.label, node_type="topic", topic_ids=[topic.topic_id],
            evidence_chunk_ids=evidence,
            attributes={"keywords": list(topic.keywords), "parent_topic_id": topic.parent_topic_id},
        )

    # Build entity nodes from explicit candidate list.
    for candidate in entity_candidates:
        label = candidate.label.strip()
        norm = normalize_label(label)
        etype = (candidate.entity_type or "concept").strip().casefold()
        if not norm:
            continue
        nid = stable_id("entity", etype, norm)
        entity_ids[(norm, etype)] = nid
        if nid not in entity_ids_by_norm[norm]:
            entity_ids_by_norm[norm].append(nid)
        if candidate.chunk_id and nid not in entity_ids_by_norm_chunk[(norm, candidate.chunk_id)]:
            entity_ids_by_norm_chunk[(norm, candidate.chunk_id)].append(nid)
        if nid not in node_acc:
            node_acc[nid] = Node(node_id=nid, label=label, node_type=etype)
        node = node_acc[nid]
        if label and label not in node.aliases and label != node.label:
            node.aliases.append(label)
        if candidate.chunk_id and candidate.chunk_id not in node.evidence_chunk_ids:
            node.evidence_chunk_ids.append(candidate.chunk_id)
        chunk = chunk_by_id.get(candidate.chunk_id)
        if chunk:
            for topic_id in chunk.topic_ids:
                if topic_id in topic_by_id and topic_id not in node.topic_ids:
                    node.topic_ids.append(topic_id)
        node.attributes.setdefault("candidate_attributes", []).append(dict(candidate.attributes))
        confidences = node.attributes.setdefault("candidate_confidences", [])
        if candidate.confidence is not None:
            confidences.append(float(candidate.confidence))

    def get_or_create_entity(label: str, chunk_id: str, desired_type: str | None = None) -> str:
        norm = normalize_label(label)
        etype = (desired_type or "").strip().casefold()
        key = (norm, etype) if etype else None
        nid = entity_ids.get(key) if key else None
        if nid is None:
            same_mention = entity_ids_by_norm_chunk.get((norm, chunk_id), [])
            # If this exact mention has a single typed candidate in this evidence chunk,
            # reuse it. If several types disagree, do not silently merge them.
            if len(same_mention) == 1:
                nid = same_mention[0]
            elif not same_mention:
                all_matches = entity_ids_by_norm.get(norm, [])
                if len(all_matches) == 1:
                    nid = all_matches[0]
        if nid is None:
            # Ambiguous or unseen surface form: preserve a generic concept node rather
            # than choosing an arbitrary type. The duplicate/type ambiguity is auditable.
            key = (norm, "concept")
            nid = entity_ids.get(key)
            if nid is None:
                nid = stable_id("entity", "concept", norm)
                entity_ids[key] = nid
                node_acc.setdefault(nid, Node(node_id=nid, label=label.strip(), node_type="concept",
                                              attributes={"entity_type_inferred": desired_type is None,
                                                          "entity_type_ambiguous": bool(entity_ids_by_norm_chunk.get((norm, chunk_id)))}))
                if nid not in entity_ids_by_norm[norm]:
                    entity_ids_by_norm[norm].append(nid)
                if chunk_id and nid not in entity_ids_by_norm_chunk[(norm, chunk_id)]:
                    entity_ids_by_norm_chunk[(norm, chunk_id)].append(nid)
        node = node_acc[nid]
        if label.strip() and label.strip() != node.label and label.strip() not in node.aliases:
            node.aliases.append(label.strip())
        if chunk_id and chunk_id not in node.evidence_chunk_ids:
            node.evidence_chunk_ids.append(chunk_id)
        chunk = chunk_by_id.get(chunk_id)
        if chunk:
            for topic_id in chunk.topic_ids:
                if topic_id in topic_by_id and topic_id not in node.topic_ids:
                    node.topic_ids.append(topic_id)
        return nid

    # Topic hierarchy edges.
    for topic in normalized_topics:
        if topic.parent_topic_id and topic.parent_topic_id in topic_by_id:
            parent_id = f"topic:{topic.parent_topic_id}"
            child_id = f"topic:{topic.topic_id}"
            eid = stable_id("edge", parent_id, "has_subtopic", child_id)
            edge_acc[eid] = Edge(
                edge_id=eid, source_node_id=parent_id, target_node_id=child_id,
                relation_type="has_subtopic", evidence_chunk_ids=[], status="derived", confidence=None,
                attributes={"evidence_note": "Derived from topic hierarchy; not a source-text claim."},
            )

    # Connect entity nodes to topic nodes based on chunk membership.
    for node in list(node_acc.values()):
        if node.node_type == "topic":
            continue
        for topic_id in node.topic_ids:
            if topic_id not in topic_by_id:
                continue
            topic_node_id = f"topic:{topic_id}"
            eid = stable_id("edge", node.node_id, "belongs_to_topic", topic_node_id)
            existing = edge_acc.get(eid)
            ev = sorted(set(node.evidence_chunk_ids))
            if existing:
                existing.evidence_chunk_ids = sorted(set(existing.evidence_chunk_ids) | set(ev))
            else:
                edge_acc[eid] = Edge(
                    edge_id=eid, source_node_id=node.node_id, target_node_id=topic_node_id,
                    relation_type="belongs_to_topic", evidence_chunk_ids=ev, status="derived",
                    confidence=None, attributes={},
                )

    # Merge relation candidates for the same normalized ordered triple; preserve all
    # valid evidence. When the quality gate is supplied, invalid provenance candidates
    # are excluded before the graph is built, and review decisions remain visible.
    relation_groups: dict[tuple[str, str, str], list[RelationCandidate]] = defaultdict(list)
    for relation in relation_candidates:
        triple_key = canonical_triple(relation.subject, relation.relation_type, relation.object)
        if not all(triple_key):
            rejected_relation_candidates += 1
            continue
        if candidate_decisions is not None:
            decision = decisions_by_triple.get(triple_key)
            if decision is None or decision.get("decision") == "rejected_provenance" or decision.get("evidence_valid") is False:
                rejected_relation_candidates += 1
                continue
            # Validate each candidate separately, so one valid candidate does not
            # accidentally legitimize another candidate with an invalid quote.
            evidence_ok, _reasons = assess_candidate_evidence(relation, chunk_by_id)
            if not evidence_ok:
                rejected_relation_candidates += 1
                continue
        relation_groups[triple_key].append(relation)

    for (subject_norm, relation_type, object_norm), group in relation_groups.items():
        first = group[0]
        first_attrs = dict(first.attributes or {})
        # Keep a human-readable/original predicate in the graph while using the
        # canonical form only to deduplicate formatting variants.
        display_relation_type = str(first.relation_type).strip()
        source_id = get_or_create_entity(first.subject, first.chunk_id, first_attrs.get("subject_type"))
        target_id = get_or_create_entity(first.object, first.chunk_id, first_attrs.get("object_type"))
        eid = stable_id("edge", source_id, relation_type, target_id)
        ev_ids = sorted({r.chunk_id for r in group if r.chunk_id})
        quotes = sorted({r.evidence_text.strip() for r in group if r.evidence_text.strip()})
        confidences = [float(r.confidence) for r in group if r.confidence is not None]
        evidence_sources = []
        for chunk_id in ev_ids:
            source_chunk = chunk_by_id.get(chunk_id)
            if source_chunk and (source_chunk.source_url or source_chunk.source_title):
                evidence_sources.append({
                    "chunk_id": chunk_id,
                    "doc_id": source_chunk.doc_id,
                    "source_url": source_chunk.source_url,
                    "source_title": source_chunk.source_title,
                    "source_version": source_chunk.source_version,
                    "source_kind": source_chunk.source_kind,
                    "page": source_chunk.page,
                })
        decision = decisions_by_triple.get((subject_norm, relation_type, object_norm), {}) if candidate_decisions is not None else {}
        quality_status = str(decision.get("decision", "review_required")) if candidate_decisions is not None else "candidate"
        # Never treat model corroboration as verified truth; review status remains explicit.
        if quality_status not in {"candidate", "review_required", "corroborated_candidate"}:
            quality_status = "review_required"
        attrs = {
            "evidence_quotes": quotes,
            "evidence_sources": evidence_sources,
            "extractors": sorted({str((r.attributes or {}).get("extractor", "unknown")) for r in group}),
            "candidate_count": len(group),
            "candidate_details": [dict(r.attributes or {}) for r in group],
            "score_kind": "extractor_score_not_calibrated_probability" if confidences else "no_score",
            "quality_gate_applied": candidate_decisions is not None,
            "quality_decision": decision.get("decision", "not_run" if candidate_decisions is None else "review_required"),
            "quality_explanation": decision.get("explanation", "Quality gate not run; treat as an unverified candidate."),
            "evidence_valid": bool(decision.get("evidence_valid", candidate_decisions is None)),
            "evidence_quality_score": decision.get("evidence_quality_score"),
            "triage_model_score": decision.get("triage_model_score"),
            "triage_score_kind": decision.get("triage_score_kind", "not_enabled"),
            "independent_extractor_count": decision.get("independent_extractor_count", 0),
            "independent_source_count": decision.get("independent_source_count", 0),
            "human_verification_required": True,
        }
        edge_acc[eid] = Edge(
            edge_id=eid, source_node_id=source_id, target_node_id=target_id,
            relation_type=display_relation_type, evidence_chunk_ids=ev_ids,
            confidence=(sum(confidences) / len(confidences)) if confidences else None,
            status=quality_status, attributes=attrs,
        )

    # Normalize all list attributes for deterministic serialization.
    nodes = sorted(node_acc.values(), key=lambda n: n.node_id)
    for n in nodes:
        n.topic_ids = sorted(set(n.topic_ids))
        n.evidence_chunk_ids = sorted(set(n.evidence_chunk_ids))
        n.aliases = sorted(set(n.aliases))
    edges = sorted(edge_acc.values(), key=lambda e: e.edge_id)
    for e in edges:
        e.evidence_chunk_ids = sorted(set(e.evidence_chunk_ids))

    hashes = dict(sorted((document_hashes or {}).items()))
    content = {
        "topics": [asdict(t) for t in sorted(normalized_topics, key=lambda t: t.topic_id)],
        "nodes": [asdict(n) for n in nodes],
        "edges": [asdict(e) for e in edges],
        "document_hashes": hashes,
    }
    snapshot_id = stable_id("snapshot", _canonical(content))
    return GraphSnapshot(snapshot_id=snapshot_id, topics=sorted(normalized_topics, key=lambda t: t.topic_id),
                         nodes=nodes, edges=edges, document_hashes=hashes,
                         metadata={"builder": "topicweave_member2_domain_agnostic_ready", "node_count": len(nodes), "edge_count": len(edges),
                                   "source_links_preserved": True,
                                   "quality_gate_applied": candidate_decisions is not None,
                                   "relation_candidates_rejected": rejected_relation_candidates,
                                   "unverified_relation_edges": sum(1 for e in edges if e.relation_type not in {"has_subtopic", "belongs_to_topic"} and e.status != "human_verified")})


def to_networkx(snapshot: GraphSnapshot) -> nx.MultiDiGraph:
    """Convert snapshot into a directed multigraph for graph algorithms/UI use."""
    graph = nx.MultiDiGraph(snapshot_id=snapshot.snapshot_id)
    for node in snapshot.nodes:
        graph.add_node(node.node_id, label=node.label, node_type=node.node_type,
                       topic_ids=list(node.topic_ids), evidence_chunk_ids=list(node.evidence_chunk_ids),
                       aliases=list(node.aliases), **{f"attr_{k}": v for k, v in node.attributes.items()})
    for edge in snapshot.edges:
        graph.add_edge(edge.source_node_id, edge.target_node_id, key=edge.edge_id,
                       edge_id=edge.edge_id, relation_type=edge.relation_type,
                       evidence_chunk_ids=list(edge.evidence_chunk_ids), confidence=edge.confidence,
                       status=edge.status, **{f"attr_{k}": v for k, v in edge.attributes.items()})
    return graph


def export_graphml(snapshot: GraphSnapshot, path: str) -> None:
    """Export a GraphML-compatible copy; complex values are JSON-encoded strings."""
    graph = nx.MultiDiGraph(snapshot_id=snapshot.snapshot_id)
    for node in snapshot.nodes:
        data = {"label": node.label, "node_type": node.node_type,
                "topic_ids": json.dumps(node.topic_ids),
                "evidence_chunk_ids": json.dumps(node.evidence_chunk_ids),
                "aliases": json.dumps(node.aliases)}
        for k, v in node.attributes.items():
            data[f"attr_{k}"] = json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (list, dict)) else ("" if v is None else str(v))
        graph.add_node(node.node_id, **data)
    for edge in snapshot.edges:
        data = {"edge_id": edge.edge_id, "relation_type": edge.relation_type,
                "evidence_chunk_ids": json.dumps(edge.evidence_chunk_ids),
                "confidence": "" if edge.confidence is None else str(edge.confidence), "status": edge.status}
        for k, v in edge.attributes.items():
            data[f"attr_{k}"] = json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (list, dict)) else ("" if v is None else str(v))
        graph.add_edge(edge.source_node_id, edge.target_node_id, key=edge.edge_id, **data)
    nx.write_graphml(graph, path)
