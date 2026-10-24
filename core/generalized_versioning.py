"""Domain-agnostic graph snapshots and candidate changed-claim detection."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any
import json

from models import GraphSnapshot, Topic, Node, Edge, GraphDiff, ChangedClaim, utc_now_iso
from core.graph_builder import normalize_label, stable_id


def snapshot_to_dict(snapshot: GraphSnapshot) -> dict[str, Any]:
    return asdict(snapshot)


def snapshot_from_dict(data: dict[str, Any]) -> GraphSnapshot:
    return GraphSnapshot(
        snapshot_id=data["snapshot_id"], topics=[Topic(**t) for t in data.get("topics", [])],
        nodes=[Node(**n) for n in data.get("nodes", [])], edges=[Edge(**e) for e in data.get("edges", [])],
        document_hashes=dict(data.get("document_hashes", {})), created_at=data.get("created_at", utc_now_iso()),
        metadata=dict(data.get("metadata", {})),
    )


def save_snapshot(snapshot: GraphSnapshot, path: str | Path) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot_to_dict(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")


def load_snapshot(path: str | Path) -> GraphSnapshot:
    return snapshot_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _edge_key(edge: Edge, node_labels: dict[str, str]) -> tuple[str, str, str]:
    return (normalize_label(node_labels.get(edge.source_node_id, edge.source_node_id)),
            normalize_label(edge.relation_type),
            normalize_label(node_labels.get(edge.target_node_id, edge.target_node_id)))


def compare_snapshots_general(old: GraphSnapshot, new: GraphSnapshot) -> GraphDiff:
    """Compare any predicates without assuming specific relation types are functional.

    `changed_claims` are hints only: same subject and predicate map to one target in the
    old snapshot and one different target in the new snapshot. Multi-target relations
    are left as additions/removals because they may be valid. The diff never asserts
    that any claim is false or that the latest source is authoritative.
    """
    old_nodes = {n.node_id: n for n in old.nodes}; new_nodes = {n.node_id: n for n in new.nodes}
    old_edges = {e.edge_id: e for e in old.edges}; new_edges = {e.edge_id: e for e in new.edges}
    old_labels = {n.node_id: n.label for n in old.nodes}; new_labels = {n.node_id: n.label for n in new.nodes}
    changed_docs = sorted(doc for doc in set(old.document_hashes) | set(new.document_hashes)
                          if old.document_hashes.get(doc) != new.document_hashes.get(doc))

    old_by_sp: dict[tuple[str, str], list[Edge]] = {}
    new_by_sp: dict[tuple[str, str], list[Edge]] = {}
    for e in old.edges:
        if e.relation_type not in {"belongs_to_topic", "has_subtopic"} and e.status != "derived" and e.evidence_chunk_ids:
            k = _edge_key(e, old_labels)[:2]; old_by_sp.setdefault(k, []).append(e)
    for e in new.edges:
        if e.relation_type not in {"belongs_to_topic", "has_subtopic"} and e.status != "derived" and e.evidence_chunk_ids:
            k = _edge_key(e, new_labels)[:2]; new_by_sp.setdefault(k, []).append(e)

    changed_claims: list[ChangedClaim] = []
    matched_old: set[str] = set(); matched_new: set[str] = set()
    for key in sorted(set(old_by_sp) & set(new_by_sp)):
        old_group, new_group = old_by_sp[key], new_by_sp[key]
        old_targets = {_edge_key(e, old_labels)[2] for e in old_group}
        new_targets = {_edge_key(e, new_labels)[2] for e in new_group}
        if len(old_group) == 1 and len(new_group) == 1 and old_targets != new_targets:
            before, after = old_group[0], new_group[0]
            changed_claims.append(ChangedClaim(
                subject=new_labels.get(after.source_node_id, after.source_node_id),
                relation_type=after.relation_type,
                old_object=old_labels.get(before.target_node_id, before.target_node_id),
                new_object=new_labels.get(after.target_node_id, after.target_node_id),
                old_edge_id=before.edge_id, new_edge_id=after.edge_id,
                old_evidence_chunk_ids=list(before.evidence_chunk_ids), new_evidence_chunk_ids=list(after.evidence_chunk_ids),
            ))
            matched_old.add(before.edge_id); matched_new.add(after.edge_id)

    changed_edges = {eid for eid in set(old_edges) & set(new_edges) if asdict(old_edges[eid]) != asdict(new_edges[eid])}
    return GraphDiff(
        old_snapshot_id=old.snapshot_id, new_snapshot_id=new.snapshot_id,
        added_node_ids=sorted(set(new_nodes) - set(old_nodes)), removed_node_ids=sorted(set(old_nodes) - set(new_nodes)),
        changed_node_ids=sorted(nid for nid in set(old_nodes) & set(new_nodes) if asdict(old_nodes[nid]) != asdict(new_nodes[nid])),
        added_edge_ids=sorted(set(new_edges) - set(old_edges)), removed_edge_ids=sorted(set(old_edges) - set(new_edges)),
        changed_edge_ids=sorted(changed_edges), changed_document_ids=changed_docs,
        changed_claims=sorted(changed_claims, key=lambda c: (c.subject, c.relation_type, c.old_object, c.new_object)),
    )
