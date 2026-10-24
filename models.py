"""Unified ReliaGraph contracts shared by Member 1, Member 2 and Member 3.

Field order for Document/Chunk/Topic/Node/Edge/GraphSnapshot preserves the original
Member 1 pipeline where practical. New metadata fields are optional. All modules should
use keyword construction for Node/Edge/GraphSnapshot to avoid positional ambiguity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Document:
    doc_id: str
    filename: str
    content_hash: str
    pages: list[str] = field(default_factory=list)
    status: str = "ok"  # ok | empty | error | duplicate
    error: str = ""
    duplicate_of: str = ""
    source_url: str | None = None
    source_title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(init=False)
class Chunk:
    # Keep Member 1's ordering: chunk_id, doc_id, page, text, topic_ids.
    chunk_id: str
    doc_id: str
    page: int | None
    text: str
    topic_ids: list[str] = field(default_factory=list)
    chunk_index: int = 0
    quarantined: list[str] = field(default_factory=list)
    # Optional provenance supported by Member 2's graph/evaluation modules.
    source_url: str | None = None
    source_title: str | None = None
    source_version: str | None = None
    source_kind: str = "uploaded_document"
    metadata: dict[str, Any] = field(default_factory=dict)


    def __init__(self, chunk_id: str, doc_id: str, *args, page: int | None = None,
                 text: str | None = None, topic_ids: list[str] | None = None,
                 chunk_index: int = 0, quarantined: list[str] | None = None,
                 source_url: str | None = None, source_title: str | None = None,
                 source_version: str | None = None, source_kind: str = "uploaded_document",
                 metadata: dict[str, Any] | None = None):
        """Accept both team interfaces: M1 (id, doc, page, text, topics) and
        M2 (id, doc, text, page, topics). New code should use keyword arguments.
        """
        rest = list(args)
        if rest:
            # M1's third positional argument is a page number followed by text.
            if isinstance(rest[0], int) or (rest[0] is None and len(rest) >= 2 and isinstance(rest[1], str)):
                if page is not None:
                    raise TypeError("page was supplied both positionally and by keyword")
                page = rest.pop(0)
                if rest:
                    if text is not None:
                        raise TypeError("text was supplied both positionally and by keyword")
                    text = rest.pop(0)
                if rest:
                    if topic_ids is not None:
                        raise TypeError("topic_ids was supplied both positionally and by keyword")
                    topic_ids = rest.pop(0)
            else:
                # M2's third positional argument is the text.
                if text is not None:
                    raise TypeError("text was supplied both positionally and by keyword")
                text = rest.pop(0)
                if rest:
                    if page is not None:
                        raise TypeError("page was supplied both positionally and by keyword")
                    page = rest.pop(0)
                if rest:
                    if topic_ids is not None:
                        raise TypeError("topic_ids was supplied both positionally and by keyword")
                    topic_ids = rest.pop(0)
            if rest:
                # Remaining legacy positions, if any, correspond to M1's chunk index
                # and quarantine list. New provenance fields must be keywords.
                if chunk_index == 0:
                    chunk_index = rest.pop(0)
                if rest and quarantined is None:
                    quarantined = rest.pop(0)
                if rest:
                    raise TypeError("too many positional arguments for Chunk")
        if text is None:
            text = ""
        self.chunk_id = str(chunk_id)
        self.doc_id = str(doc_id)
        self.page = page
        self.text = str(text)
        self.topic_ids = list(topic_ids or [])
        self.chunk_index = int(chunk_index or 0)
        self.quarantined = list(quarantined or [])
        self.source_url = source_url
        self.source_title = source_title
        self.source_version = source_version
        self.source_kind = str(source_kind or "uploaded_document")
        self.metadata = dict(metadata or {})


@dataclass
class Topic:
    topic_id: str
    label: str
    keywords: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    parent_topic_id: str | None = None


@dataclass
class EntityCandidate:
    label: str
    entity_type: str
    chunk_id: str
    confidence: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationCandidate:
    subject: str
    relation_type: str
    object: str
    chunk_id: str
    evidence_text: str
    confidence: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:
    node_id: str
    label: str
    node_type: str
    topic_ids: list[str] = field(default_factory=list)
    evidence_chunk_ids: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(init=False)
class Edge:
    # Keep Member 1's positional status field at index 6. Prefer keyword construction.
    edge_id: str
    source_node_id: str
    target_node_id: str
    relation_type: str
    evidence_chunk_ids: list[str] = field(default_factory=list)
    status: str = "candidate"  # candidate | review_required | corroborated_candidate | derived | conflict | human_verified
    confidence: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


    def __init__(self, edge_id: str, source_node_id: str, target_node_id: str,
                 relation_type: str, evidence_chunk_ids: list[str] | None = None,
                 *args, status: str | None = None, confidence: float | None = None,
                 attributes: dict[str, Any] | None = None):
        """Backward compatible constructor for M1/M2 positional Edge layouts.

        M1 uses the sixth positional value as status; M2 used it as confidence.
        Prefer keywords for all new callers.
        """
        rest = list(args)
        if rest:
            first = rest.pop(0)
            if isinstance(first, str):
                if status is not None:
                    raise TypeError("status supplied more than once")
                status = first
            elif first is None or isinstance(first, (int, float)):
                if confidence is not None:
                    raise TypeError("confidence supplied more than once")
                confidence = None if first is None else float(first)
            else:
                raise TypeError("unrecognized positional Edge argument")
        if rest:
            second = rest.pop(0)
            if isinstance(second, str):
                if status is not None:
                    raise TypeError("status supplied more than once")
                status = second
            elif isinstance(second, dict):
                if attributes is not None:
                    raise TypeError("attributes supplied more than once")
                attributes = second
            else:
                raise TypeError("unrecognized positional Edge argument")
        if rest:
            third = rest.pop(0)
            if isinstance(third, dict):
                if attributes is not None:
                    raise TypeError("attributes supplied more than once")
                attributes = third
            else:
                raise TypeError("unrecognized positional Edge argument")
        if rest:
            raise TypeError("too many positional arguments for Edge")
        self.edge_id = str(edge_id)
        self.source_node_id = str(source_node_id)
        self.target_node_id = str(target_node_id)
        self.relation_type = str(relation_type)
        self.evidence_chunk_ids = list(evidence_chunk_ids or [])
        self.status = str(status or "candidate")
        self.confidence = confidence
        self.attributes = dict(attributes or {})


@dataclass
class TestCase:
    __test__ = False  # prevent pytest collecting this as a test class
    test_id: str
    question: str
    reference_answer: str
    supporting_chunk_ids: list[str]
    answerable: bool = True
    kind: str = "factual"
    lang: str = "en"


@dataclass
class AuditFinding:
    finding_id: str
    category: str
    severity: str
    affected_ids: list[str]
    evidence_chunk_ids: list[str] = field(default_factory=list)
    message: str = ""
    recommendation: str = ""


@dataclass
class TopicCoverage:
    topic_id: str
    topic_label: str
    chunk_count: int
    entity_count: int
    relationship_count: int
    coverage_status: str


@dataclass(init=False)
class GraphSnapshot:
    # Keep Member 1's field order; use keywords when constructing.
    snapshot_id: str
    document_hashes: dict[str, str] = field(default_factory=dict)
    topics: list[Topic] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)


    def __init__(self, snapshot_id: str, *args, document_hashes: dict[str, str] | None = None,
                 topics: list[Topic] | None = None, nodes: list[Node] | None = None,
                 edges: list[Edge] | None = None, created_at: str | None = None,
                 metadata: dict[str, Any] | None = None):
        """Accept M1 positional order (hashes, topics, nodes, edges) and M2's
        historical order (topics, nodes, edges, hashes). New code should use keywords.
        """
        rest = list(args)
        if rest:
            if isinstance(rest[0], dict):
                # Member 1 ordering
                if document_hashes is not None: raise TypeError("document_hashes supplied twice")
                document_hashes = rest.pop(0)
                if rest and topics is None: topics = rest.pop(0)
                if rest and nodes is None: nodes = rest.pop(0)
                if rest and edges is None: edges = rest.pop(0)
            else:
                # Member 2 ordering
                if topics is None: topics = rest.pop(0)
                if rest and nodes is None: nodes = rest.pop(0)
                if rest and edges is None: edges = rest.pop(0)
                if rest and document_hashes is None and isinstance(rest[0], dict): document_hashes = rest.pop(0)
            if rest:
                if created_at is None: created_at = rest.pop(0)
            if rest:
                if metadata is None: metadata = rest.pop(0)
            if rest:
                raise TypeError("too many positional arguments for GraphSnapshot")
        self.snapshot_id = str(snapshot_id)
        self.document_hashes = dict(document_hashes or {})
        self.topics = list(topics or [])
        self.nodes = list(nodes or [])
        self.edges = list(edges or [])
        self.created_at = created_at or utc_now_iso()
        self.metadata = dict(metadata or {})


@dataclass
class ChangedClaim:
    subject: str
    relation_type: str
    old_object: str
    new_object: str
    old_edge_id: str
    new_edge_id: str
    old_evidence_chunk_ids: list[str] = field(default_factory=list)
    new_evidence_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class GraphDiff:
    old_snapshot_id: str
    new_snapshot_id: str
    added_node_ids: list[str] = field(default_factory=list)
    removed_node_ids: list[str] = field(default_factory=list)
    changed_node_ids: list[str] = field(default_factory=list)
    added_edge_ids: list[str] = field(default_factory=list)
    removed_edge_ids: list[str] = field(default_factory=list)
    changed_edge_ids: list[str] = field(default_factory=list)
    changed_document_ids: list[str] = field(default_factory=list)
    changed_claims: list[ChangedClaim] = field(default_factory=list)


@dataclass
class EvaluationResult:
    test_id: str
    retrieved_chunk_ids: list[str]
    answer: str
    scores: dict[str, Any] = field(default_factory=dict)
    diagnosis: str = ""


def chunk_from(value: Chunk | Mapping[str, Any]) -> Chunk:
    """Normalize a Member 1 Chunk or mapping into the unified contract."""
    if isinstance(value, Chunk):
        return value
    return Chunk(
        chunk_id=str(value["chunk_id"]),
        doc_id=str(value["doc_id"]),
        page=value.get("page"),
        text=str(value.get("text", "")),
        topic_ids=list(value.get("topic_ids") or []),
        chunk_index=int(value.get("chunk_index", 0) or 0),
        quarantined=list(value.get("quarantined") or []),
        source_url=value.get("source_url"),
        source_title=value.get("source_title"),
        source_version=value.get("source_version"),
        source_kind=str(value.get("source_kind", "uploaded_document")),
        metadata=dict(value.get("metadata") or {}),
    )


def topic_from(value: Topic | Mapping[str, Any]) -> Topic:
    if isinstance(value, Topic):
        return value
    return Topic(
        topic_id=str(value["topic_id"]),
        label=str(value.get("label", value["topic_id"])),
        keywords=list(value.get("keywords") or []),
        chunk_ids=list(value.get("chunk_ids") or []),
        parent_topic_id=value.get("parent_topic_id"),
    )
