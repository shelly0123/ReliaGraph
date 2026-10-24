"""Serialize graph snapshots and compare them, including conservative changed-claim hints."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
from typing import Any
from models import (Chunk, Topic, Node, Edge, GraphSnapshot, GraphDiff, ChangedClaim,
                    utc_now_iso)
from core.graph_builder import stable_id, normalize_label
from core.graph_audit import FUNCTIONAL_RELATIONS


def snapshot_to_dict(snapshot: GraphSnapshot) -> dict[str, Any]:
    return asdict(snapshot)


def snapshot_from_dict(data: dict[str, Any]) -> GraphSnapshot:
    return GraphSnapshot(
        snapshot_id=data["snapshot_id"],
        topics=[Topic(**t) for t in data.get("topics", [])],
        nodes=[Node(**n) for n in data.get("nodes", [])],
        edges=[Edge(**e) for e in data.get("edges", [])],
        document_hashes=dict(data.get("document_hashes", {})),
        created_at=data.get("created_at", utc_now_iso()),
        metadata=dict(data.get("metadata", {})),
    )


def save_snapshot(snapshot: GraphSnapshot, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot_to_dict(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")


def load_snapshot(path: str | Path) -> GraphSnapshot:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return snapshot_from_dict(data)


def _canonical(obj: Any) -> str:
    return json.dumps(asdict(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _node_label_map(snapshot: GraphSnapshot) -> dict[str, str]:
    return {n.node_id: n.label for n in snapshot.nodes}


def compare_snapshots(old: GraphSnapshot, new: GraphSnapshot) -> GraphDiff:
    old_nodes = {n.node_id: n for n in old.nodes}
    new_nodes = {n.node_id: n for n in new.nodes}
    old_edges = {e.edge_id: e for e in old.edges}
    new_edges = {e.edge_id: e for e in new.edges}
    old_labels, new_labels = _node_label_map(old), _node_label_map(new)

    changed_docs = sorted(doc for doc in (set(old.document_hashes) | set(new.document_hashes))
                          if old.document_hashes.get(doc) != new.document_hashes.get(doc))
    added_edges = set(new_edges) - set(old_edges)
    removed_edges = set(old_edges) - set(new_edges)

    # A changed claim is a *hint*: for a small, explicit set of one-value predicates,
    # report when the same subject/predicate now points to a different object.
    old_groups: dict[tuple[str, str], list[Edge]] = {}
    new_groups: dict[tuple[str, str], list[Edge]] = {}
    for edge in old.edges:
        rel = edge.relation_type.casefold()
        if rel in FUNCTIONAL_RELATIONS:
            subject = normalize_label(old_labels.get(edge.source_node_id, edge.source_node_id))
            old_groups.setdefault((subject, rel), []).append(edge)
    for edge in new.edges:
        rel = edge.relation_type.casefold()
        if rel in FUNCTIONAL_RELATIONS:
            subject = normalize_label(new_labels.get(edge.source_node_id, edge.source_node_id))
            new_groups.setdefault((subject, rel), []).append(edge)

    changed_claims: list[ChangedClaim] = []
    matched_old: set[str] = set()
    matched_new: set[str] = set()
    for key in sorted(set(old_groups) & set(new_groups)):
        old_group, new_group = old_groups[key], new_groups[key]
        if len(old_group) == 1 and len(new_group) == 1:
            before, after = old_group[0], new_group[0]
            old_obj = normalize_label(old_labels.get(before.target_node_id, before.target_node_id))
            new_obj = normalize_label(new_labels.get(after.target_node_id, after.target_node_id))
            if old_obj != new_obj:
                changed_claims.append(ChangedClaim(
                    subject=new_labels.get(after.source_node_id, after.source_node_id),
                    relation_type=after.relation_type,
                    old_object=old_labels.get(before.target_node_id, before.target_node_id),
                    new_object=new_labels.get(after.target_node_id, after.target_node_id),
                    old_edge_id=before.edge_id, new_edge_id=after.edge_id,
                    old_evidence_chunk_ids=list(before.evidence_chunk_ids),
                    new_evidence_chunk_ids=list(after.evidence_chunk_ids),
                ))
                matched_old.add(before.edge_id)
                matched_new.add(after.edge_id)

    changed_edges = {eid for eid in set(old_edges) & set(new_edges) if _canonical(old_edges[eid]) != _canonical(new_edges[eid])}
    return GraphDiff(
        old_snapshot_id=old.snapshot_id, new_snapshot_id=new.snapshot_id,
        added_node_ids=sorted(set(new_nodes) - set(old_nodes)),
        removed_node_ids=sorted(set(old_nodes) - set(new_nodes)),
        changed_node_ids=sorted(nid for nid in set(old_nodes) & set(new_nodes) if _canonical(old_nodes[nid]) != _canonical(new_nodes[nid])),
        added_edge_ids=sorted(added_edges), removed_edge_ids=sorted(removed_edges),
        changed_edge_ids=sorted(changed_edges), changed_document_ids=changed_docs,
        changed_claims=sorted(changed_claims, key=lambda c: (c.subject, c.relation_type, c.old_object, c.new_object)),
    )
