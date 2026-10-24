"""SQLite-backed incremental knowledge graph with per-document provenance.

Adding/editing/deleting a document processes only that document's extraction output and
updates an in-memory MultiDiGraph incrementally. Claims from different documents are
kept as separate evidence-bearing edges; competing claims are never overwritten.
"""
from __future__ import annotations

import hashlib, json, re, sqlite3, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import networkx as nx
from collections import defaultdict
from core.phase2.injection_guard import scan_and_sanitize


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.-]", "", str(s or "").casefold(), flags=re.UNICODE)).strip()

def _stable_id(prefix: str, *parts: str) -> str:
    joined = "\0".join([prefix, *map(str, parts)])
    return prefix + "_" + hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

def _contains_quote(haystack: str, quote: str) -> bool:
    normalize = lambda x: re.sub(r"\s+", " ", str(x or "")).casefold().strip()
    return bool(normalize(quote)) and normalize(quote) in normalize(haystack)

def split_chunks(doc_id: str, text: str, max_chars: int = 900, overlap: int = 120) -> list[dict[str, Any]]:
    """Simple deterministic chunking for the standalone module; callers can supply their own IDs."""
    value = str(text or "").strip()
    if not value: return []
    # Prefer paragraph/sentence boundaries but avoid chunks much larger than max_chars.
    units = [x.strip() for x in re.split(r"(?<=[.!?।！？])\s+|\n{2,}", value) if x.strip()]
    chunks: list[str] = []
    current = ""
    for unit in units or [value]:
        if len(current) + len(unit) + 1 <= max_chars:
            current = (current + " " + unit).strip()
        else:
            if current: chunks.append(current)
            if len(unit) <= max_chars:
                current = unit
            else:
                for pos in range(0, len(unit), max(1, max_chars - overlap)):
                    piece = unit[pos:pos + max_chars].strip()
                    if piece: chunks.append(piece)
                current = ""
    if current: chunks.append(current)
    out = []
    for ix, body in enumerate(chunks):
        cid = _stable_id("chunk", doc_id, str(ix), hashlib.sha256(body.encode("utf-8")).hexdigest())
        out.append({"chunk_id": cid, "doc_id": doc_id, "text": body, "page": None})
    return out

class IncrementalKnowledgeBase:
    """Document-level CRUD with incremental graph mutations and persistent SQLite storage.

    The extractor callback signature is extractor(safe_text, chunks) -> sequence of claim dicts.
    Each claim has subject, relation_type/relation, object, evidence_text and optional chunk_id,
    confidence, status, metadata. The evidence quote must occur in the original safe source text.
    """
    def __init__(self, db_path: str | Path = "data/phase2_kb.sqlite3",
                 extractor: Callable[[str, Sequence[dict[str, Any]]], Sequence[Mapping[str, Any]]] | None = None,
                 *, retain_history_text: bool = False):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.extractor = extractor
        # CIA confidentiality default: archive hashes and extracted claim evidence, not
        # whole prior documents. Set True only if a documented retention policy requires it.
        self.retain_history_text = bool(retain_history_text)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        self.graph = nx.MultiDiGraph()
        # Index live graph edges by owning document. This avoids scanning all E edges
        # on every update; normal edit/delete cost depends primarily on the changed doc.
        self._edge_keys_by_doc: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
        self._load_graph_once()

    def close(self): self.conn.close()

    def _init_schema(self):
        with self.conn:
            self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents(
              doc_id TEXT PRIMARY KEY, title TEXT NOT NULL, source_url TEXT NOT NULL DEFAULT '',
              raw_text TEXT NOT NULL, safe_text TEXT NOT NULL, content_hash TEXT NOT NULL,
              version INTEGER NOT NULL DEFAULT 1, updated_at TEXT NOT NULL, injection_findings_json TEXT NOT NULL DEFAULT '[]');
            CREATE TABLE IF NOT EXISTS document_history(
              history_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, version INTEGER NOT NULL,
              title TEXT NOT NULL, source_url TEXT NOT NULL, raw_text TEXT NOT NULL, safe_text TEXT NOT NULL,
              content_hash TEXT NOT NULL, updated_at TEXT NOT NULL, claims_json TEXT NOT NULL,
              content_length_chars INTEGER NOT NULL DEFAULT 0,
              UNIQUE(doc_id, version));
            CREATE TABLE IF NOT EXISTS chunks(
              chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, text TEXT NOT NULL, page INTEGER,
              FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE);
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            CREATE TABLE IF NOT EXISTS claims(
              claim_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, chunk_id TEXT NOT NULL,
              subject TEXT NOT NULL, relation_type TEXT NOT NULL, object TEXT NOT NULL,
              evidence_text TEXT NOT NULL, confidence REAL, status TEXT NOT NULL DEFAULT 'candidate',
              source_url TEXT NOT NULL DEFAULT '', source_title TEXT NOT NULL DEFAULT '',
              metadata_json TEXT NOT NULL DEFAULT '{}',
              FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE);
            CREATE INDEX IF NOT EXISTS idx_claims_doc ON claims(doc_id);
            CREATE INDEX IF NOT EXISTS idx_claims_slot ON claims(subject, relation_type);
            CREATE TABLE IF NOT EXISTS topic_links(
              link_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, chunk_id TEXT NOT NULL, topic_id TEXT NOT NULL,
              topic_label TEXT NOT NULL, evidence_text TEXT NOT NULL,
              FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE);
            CREATE INDEX IF NOT EXISTS idx_topic_links_doc ON topic_links(doc_id);
            """)
        # Backward-compatible schema migration for databases created by earlier versions.
        history_cols = {str(row[1]) for row in self.conn.execute("PRAGMA table_info(document_history)")}
        if "content_length_chars" not in history_cols:
            self.conn.execute("ALTER TABLE document_history ADD COLUMN content_length_chars INTEGER NOT NULL DEFAULT 0")
            self.conn.commit()

    def _load_graph_once(self):
        """Initial materialization on startup; subsequent CRUD calls update graph incrementally."""
        self.graph.clear()
        self._edge_keys_by_doc.clear()
        for row in self.conn.execute("SELECT * FROM claims ORDER BY claim_id"):
            self._graph_add_claim(dict(row))
        for row in self.conn.execute("SELECT * FROM topic_links ORDER BY link_id"):
            self._graph_add_topic_link(dict(row))
        self._prune_orphan_nodes()

    def _node_id(self, label: str, kind: str = "entity") -> str:
        return _stable_id(kind, _norm(label))

    def _graph_add_claim(self, claim: Mapping[str, Any]):
        s, o = str(claim["subject"]), str(claim["object"])
        pred = str(claim["relation_type"])
        sid, oid = self._node_id(s), self._node_id(o)
        for nid, label in ((sid, s), (oid, o)):
            if nid not in self.graph:
                self.graph.add_node(nid, label=label, node_type="entity", support_doc_ids=[], evidence_chunk_ids=[])
        evidence = {"doc_id": claim["doc_id"], "chunk_id": claim["chunk_id"],
                    "evidence_text": claim["evidence_text"], "source_url": claim.get("source_url", ""),
                    "source_title": claim.get("source_title", ""), "status": claim.get("status", "candidate")}
        edge_key = str(claim["claim_id"])
        self.graph.add_edge(sid, oid, key=edge_key, claim_id=edge_key, doc_id=str(claim["doc_id"]),
                            subject=s, relation_type=pred, object=o, evidence_chunk_ids=[claim["chunk_id"]],
                            evidence=[evidence], confidence=claim.get("confidence"), status=claim.get("status", "candidate"),
                            source_url=claim.get("source_url", ""), source_title=claim.get("source_title", ""))
        self._edge_keys_by_doc[str(claim["doc_id"])].add((sid, oid, edge_key))
        for nid in (sid, oid):
            self._refresh_node_support(nid)

    def _graph_add_topic_link(self, row: Mapping[str, Any]):
        topic_id = str(row["topic_id"])
        topic_label = str(row["topic_label"])
        cid = str(row["chunk_id"])
        # Link the topic to entities explicitly mentioned in claims grounded in this chunk.
        topic_node_id = self._node_id(topic_label, "topic")
        if topic_node_id not in self.graph:
            self.graph.add_node(topic_node_id, label=topic_label, node_type="topic", topic_id=topic_id,
                                support_doc_ids=[], evidence_chunk_ids=[])
        entity_labels = set()
        for claim_row in self.conn.execute("SELECT subject, object FROM claims WHERE chunk_id=?", (cid,)):
            entity_labels.add(str(claim_row["subject"])); entity_labels.add(str(claim_row["object"]))
        doc_meta = self.conn.execute("SELECT title, source_url FROM documents WHERE doc_id=?", (str(row["doc_id"]),)).fetchone()
        source_title = str(doc_meta["title"] if doc_meta else topic_label)
        source_url = str(doc_meta["source_url"] if doc_meta else "")
        for entity_label in sorted(x for x in entity_labels if x):
            entity_id = self._node_id(entity_label)
            if entity_id in self.graph:
                edge_key = _stable_id("topic_link", str(row["link_id"]), entity_id)
                self.graph.add_edge(topic_node_id, entity_id, key=edge_key, claim_id=edge_key,
                                    doc_id=row["doc_id"], subject=topic_label,
                                    relation_type="contains_concept", object=entity_label,
                                    evidence_chunk_ids=[cid], evidence=[{"doc_id": row["doc_id"], "chunk_id": cid,
                                    "evidence_text": row["evidence_text"], "source_url": source_url, "source_title": source_title,
                                    "status": "derived_topic_link"}], status="derived_topic_link", source_url=source_url, source_title=source_title)
                self._edge_keys_by_doc[str(row["doc_id"])].add((topic_node_id, entity_id, edge_key))
        self._refresh_node_support(topic_node_id)

    def _refresh_node_support(self, node_id: str):
        if node_id not in self.graph: return
        docs, chunks = set(), set()
        for _, _, _, data in self.graph.in_edges(node_id, keys=True, data=True):
            for ev in data.get("evidence", []):
                docs.add(str(ev.get("doc_id", ""))); chunks.add(str(ev.get("chunk_id", "")))
        for _, _, _, data in self.graph.out_edges(node_id, keys=True, data=True):
            for ev in data.get("evidence", []):
                docs.add(str(ev.get("doc_id", ""))); chunks.add(str(ev.get("chunk_id", "")))
        self.graph.nodes[node_id]["support_doc_ids"] = sorted(x for x in docs if x)
        self.graph.nodes[node_id]["evidence_chunk_ids"] = sorted(x for x in chunks if x)

    def _remove_doc_from_graph(self, doc_id: str):
        """Remove only this document's indexed graph edges, then refresh affected nodes.

        Older versions scanned every graph edge twice for every edit/delete. The doc index
        changes this to O(edges owned by the changed document + affected-node neighbourhood).
        """
        affected: set[str] = set()
        owned_edges = self._edge_keys_by_doc.pop(str(doc_id), set())
        for u, v, key in owned_edges:
            if not self.graph.has_edge(u, v, key):
                continue
            data = self.graph.get_edge_data(u, v, key) or {}
            evidence = [ev for ev in data.get("evidence", []) if str(ev.get("doc_id", "")) != str(doc_id)]
            affected.update([u, v])
            if evidence:
                data["evidence"] = evidence
                data["evidence_chunk_ids"] = sorted({str(ev.get("chunk_id", "")) for ev in evidence if ev.get("chunk_id")})
                # This path is defensive: current claim/topic edges are source-owned, but
                # if older serialized data combined sources, preserve remaining evidence.
                remaining_docs = sorted({str(ev.get("doc_id", "")) for ev in evidence if ev.get("doc_id")})
                if remaining_docs:
                    self._edge_keys_by_doc[remaining_docs[0]].add((u, v, key))
            else:
                self.graph.remove_edge(u, v, key)
        for node_id in affected:
            if node_id in self.graph:
                self._refresh_node_support(node_id)
        self._prune_orphan_nodes(affected)

    def _prune_orphan_nodes(self, candidates: Iterable[str] | None = None):
        # Topics can remain as useful empty topic containers; remove only isolated concepts.
        check = list(candidates) if candidates is not None else list(self.graph.nodes())
        isolates = [n for n in check if n in self.graph and self.graph.degree(n) == 0
                    and self.graph.nodes[n].get("node_type") != "topic"]
        self.graph.remove_nodes_from(isolates)

    def _claim_rows(self, doc_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM claims WHERE doc_id=?", (doc_id,))]

    def _insert_claims_graph(self, doc_id: str):
        for row in self.conn.execute("SELECT * FROM claims WHERE doc_id=? ORDER BY claim_id", (doc_id,)):
            self._graph_add_claim(dict(row))
        for row in self.conn.execute("SELECT * FROM topic_links WHERE doc_id=? ORDER BY link_id", (doc_id,)):
            self._graph_add_topic_link(dict(row))

    def _prepare_claims(self, doc_id: str, claims: Sequence[Mapping[str, Any]], chunks: Sequence[Mapping[str, Any]], raw_text: str, safe_text: str, source_url: str, title: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        chunk_by_id = {str(c["chunk_id"]): c for c in chunks}
        accepted, rejected = [], []
        for ix, c in enumerate(claims):
            subject = str(c.get("subject", "")).strip()
            relation = str(c.get("relation_type", c.get("relation", ""))).strip()
            obj = str(c.get("object", c.get("object_", ""))).strip()
            evidence = str(c.get("evidence_text", c.get("quote", ""))).strip()
            if not (subject and relation and obj and evidence):
                rejected.append({"reason": "missing_required_field", "excerpt": evidence[:100]}); continue
            if not _contains_quote(safe_text, evidence):
                rejected.append({"reason": "evidence_not_in_safe_source", "excerpt": evidence[:100]}); continue
            if not _contains_quote(evidence, subject):
                rejected.append({"reason": "subject_not_anchored_to_evidence", "excerpt": evidence[:100]}); continue
            if not _contains_quote(evidence, obj):
                rejected.append({"reason": "object_not_anchored_to_evidence", "excerpt": evidence[:100]}); continue
            chunk_id = str(c.get("chunk_id", ""))
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None or not _contains_quote(str(chunk.get("text", "")), evidence):
                # Find the chunk that anchors the full evidence quote.
                chunk = next((item for item in chunks if _contains_quote(str(item.get("text", "")), evidence)), None)
                if chunk is None:
                    rejected.append({"reason": "evidence_not_anchored_to_chunk", "excerpt": evidence[:100]}); continue
                chunk_id = str(chunk["chunk_id"])
            conf = c.get("confidence")
            try: conf = float(conf) if conf is not None else None
            except (TypeError, ValueError): conf = None
            if conf is not None and not 0 <= conf <= 1: conf = None
            meta = dict(c.get("metadata") or c.get("attributes") or {})
            claim_id = _stable_id("claim", doc_id, chunk_id, _norm(subject), _norm(relation), _norm(obj))
            accepted.append({
                "claim_id": claim_id, "doc_id": doc_id, "chunk_id": chunk_id,
                "subject": subject, "relation_type": relation, "object": obj,
                "evidence_text": evidence, "confidence": conf,
                "status": str(c.get("status", "candidate")), "source_url": source_url,
                "source_title": title, "metadata_json": json.dumps(meta, ensure_ascii=False, sort_keys=True),
            })
        return accepted, rejected

    def upsert_document(self, doc_id: str, text: str, *, title: str = "", source_url: str = "",
                        claims: Sequence[Mapping[str, Any]] | None = None,
                        topic_assignments: Mapping[str, Sequence[Mapping[str, str] | str]] | None = None,
                        chunks: Sequence[Mapping[str, Any]] | None = None,
                        max_chars: int = 900, overlap: int = 120,
                        force_refresh: bool = False) -> dict[str, Any]:
        """Add/edit one document. Existing claims from this document are replaced only.

        `topic_assignments` maps chunk_id to strings or {topic_id,label} dicts. The update
        does not re-run extraction over other documents or rebuild the full graph.
        """
        started = time.perf_counter()
        doc_id = str(doc_id).strip()
        if not doc_id: raise ValueError("doc_id must be non-empty")
        raw_text = str(text or "")
        if not raw_text.strip(): raise ValueError("document text must be non-empty")
        content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        old = self.conn.execute("SELECT content_hash, version, title, source_url FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        metadata_unchanged = bool(old and str(old["title"] or "") == str(title or "")
                                  and str(old["source_url"] or "") == str(source_url or ""))
        if old and old["content_hash"] == content_hash and metadata_unchanged and not force_refresh:
            return {"operation": "unchanged", "doc_id": doc_id, "version": old["version"], "elapsed_seconds": time.perf_counter()-started,
                    "graph_nodes": self.graph.number_of_nodes(), "graph_edges": self.graph.number_of_edges(), "rejected_claims": 0}
        sanit = scan_and_sanitize(raw_text)
        safe_text = sanit.safe_text
        if chunks is not None:
            # Preserve Member 1's stable chunk IDs, but replace model-facing chunk text with
            # its own sanitized copy so an unsafe line cannot reach extraction/generation.
            safe_chunks = []
            for raw_chunk in chunks:
                c = dict(raw_chunk)
                raw_chunk_text = str(c.get("text", ""))
                chunk_sanitization = scan_and_sanitize(raw_chunk_text)
                if chunk_sanitization.findings:
                    sanit.findings.extend(chunk_sanitization.findings)
                c["doc_id"] = str(c.get("doc_id", doc_id)); c["chunk_id"] = str(c["chunk_id"])
                c["text"] = chunk_sanitization.safe_text
                c["page"] = c.get("page")
                safe_chunks.append(c)
            chunks = safe_chunks
        else:
            chunks = split_chunks(doc_id, safe_text, max_chars=max_chars, overlap=overlap)
        if claims is None:
            if self.extractor is None:
                raise ValueError("No extractor configured. Pass `claims=` or initialize with an extractor callback.")
            claims = self.extractor(safe_text, chunks)
        accepted, rejected = self._prepare_claims(doc_id, claims, chunks, raw_text, safe_text, source_url, title)
        old_doc = None
        old_claims = []
        if old:
            old_doc_row = self.conn.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
            old_doc = dict(old_doc_row) if old_doc_row else None
            old_claims = self._claim_rows(doc_id)
        version = int(old["version"] + 1) if old else 1
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # Write all new document data in a single DB transaction. If it fails, in-memory
        # graph is untouched. After commit, mutate only this document's graph contribution.
        with self.conn:
            if old_doc is not None:
                history_id = _stable_id("history", doc_id, str(old_doc.get("version", 1)), str(old_doc.get("content_hash", "")))
                historical_raw = old_doc.get("raw_text", "") if self.retain_history_text else ""
                historical_safe = old_doc.get("safe_text", "") if self.retain_history_text else ""
                self.conn.execute("INSERT OR IGNORE INTO document_history(history_id,doc_id,version,title,source_url,raw_text,safe_text,content_hash,updated_at,claims_json,content_length_chars) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                                  (history_id, doc_id, old_doc.get("version", 1), old_doc.get("title", ""),
                                   old_doc.get("source_url", ""), historical_raw, historical_safe,
                                   old_doc.get("content_hash", ""), old_doc.get("updated_at", now),
                                   json.dumps(old_claims, ensure_ascii=False, sort_keys=True),
                                   len(str(old_doc.get("raw_text", "")))))
            self.conn.execute("INSERT OR REPLACE INTO documents(doc_id,title,source_url,raw_text,safe_text,content_hash,version,updated_at,injection_findings_json) VALUES(?,?,?,?,?,?,?,?,?)",
                              (doc_id,title,source_url,raw_text,safe_text,content_hash,version,now,json.dumps([f.to_dict() for f in sanit.findings], ensure_ascii=False)))
            self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self.conn.executemany("INSERT INTO chunks(chunk_id,doc_id,text,page) VALUES(?,?,?,?)",
                                  [(c["chunk_id"], doc_id, c["text"], c.get("page")) for c in chunks])
            self.conn.execute("DELETE FROM claims WHERE doc_id=?", (doc_id,))
            self.conn.executemany("INSERT INTO claims(claim_id,doc_id,chunk_id,subject,relation_type,object,evidence_text,confidence,status,source_url,source_title,metadata_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                                  [(c["claim_id"], c["doc_id"], c["chunk_id"], c["subject"], c["relation_type"], c["object"], c["evidence_text"], c["confidence"], c["status"], c["source_url"], c["source_title"], c["metadata_json"]) for c in accepted])
            self.conn.execute("DELETE FROM topic_links WHERE doc_id=?", (doc_id,))
            topic_rows = []
            for chunk_id, values in (topic_assignments or {}).items():
                chunk = next((x for x in chunks if x["chunk_id"] == str(chunk_id)), None)
                if chunk is None: continue
                for tv in values:
                    topic_id = str(tv.get("topic_id", "") if isinstance(tv, Mapping) else tv)
                    label = str(tv.get("label", topic_id) if isinstance(tv, Mapping) else tv)
                    if not topic_id or not label: continue
                    link_id = _stable_id("topiclink", doc_id, str(chunk_id), topic_id)
                    topic_rows.append((link_id, doc_id, str(chunk_id), topic_id, label, str(chunk["text"])))
            self.conn.executemany("INSERT OR REPLACE INTO topic_links(link_id,doc_id,chunk_id,topic_id,topic_label,evidence_text) VALUES(?,?,?,?,?,?)", topic_rows)
        # Update in-memory graph contribution. No full rebuild for normal CRUD operations.
        self._remove_doc_from_graph(doc_id)
        self._insert_claims_graph(doc_id)
        elapsed = time.perf_counter() - started
        old_triples = {(str(c.get("subject", "")).casefold(), str(c.get("relation_type", "")).casefold(), str(c.get("object", "")).casefold()) for c in old_claims}
        new_triples = {(str(c.get("subject", "")).casefold(), str(c.get("relation_type", "")).casefold(), str(c.get("object", "")).casefold()) for c in accepted}
        return {"operation": "added" if not old else "edited", "doc_id": doc_id, "version": version,
                "elapsed_seconds": round(elapsed, 6), "graph_nodes": self.graph.number_of_nodes(),
                "graph_edges": self.graph.number_of_edges(), "accepted_claims": len(accepted),
                "rejected_claims": len(rejected), "rejections": rejected,
                "added_claims": len(new_triples - old_triples), "removed_claims": len(old_triples - new_triples),
                "injection_findings": len(sanit.findings), "document_quarantined_for_model_context": sanit.quarantined,
                "under_60_seconds": elapsed < 60.0}

    def delete_document(self, doc_id: str) -> dict[str, Any]:
        """Delete one document's live contribution while archiving its last version.

        The archive preserves metadata and extracted claims for change audit. Normal graph
        mutation still removes only this document's edges and leaves other sources intact.
        """
        started = time.perf_counter()
        row = self.conn.execute("SELECT * FROM documents WHERE doc_id=?", (str(doc_id),)).fetchone()
        if not row:
            return {"operation": "not_found", "doc_id": str(doc_id), "elapsed_seconds": round(time.perf_counter()-started, 6),
                    "graph_nodes": self.graph.number_of_nodes(), "graph_edges": self.graph.number_of_edges(), "under_60_seconds": True}
        document = dict(row)
        current_claims = self._claim_rows(str(doc_id))
        history_id = _stable_id("history", str(doc_id), str(document.get("version", 1)), str(document.get("content_hash", "")))
        with self.conn:
            historical_raw = document.get("raw_text", "") if self.retain_history_text else ""
            historical_safe = document.get("safe_text", "") if self.retain_history_text else ""
            self.conn.execute("INSERT OR IGNORE INTO document_history(history_id,doc_id,version,title,source_url,raw_text,safe_text,content_hash,updated_at,claims_json,content_length_chars) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                              (history_id, doc_id, document.get("version", 1), document.get("title", ""),
                               document.get("source_url", ""), historical_raw, historical_safe,
                               document.get("content_hash", ""), document.get("updated_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
                               json.dumps(current_claims, ensure_ascii=False, sort_keys=True), len(str(document.get("raw_text", "")))))
            self.conn.execute("DELETE FROM documents WHERE doc_id=?", (str(doc_id),))
            # Foreign-key cascades delete the document's live chunks, claims and topic links.
        self._remove_doc_from_graph(str(doc_id))
        elapsed = time.perf_counter() - started
        return {"operation": "deleted", "doc_id": str(doc_id), "archived_version": document.get("version", 1),
                "elapsed_seconds": round(elapsed, 6), "graph_nodes": self.graph.number_of_nodes(),
                "graph_edges": self.graph.number_of_edges(), "under_60_seconds": elapsed < 60.0}

    def get_claims(self, doc_id: str | None = None) -> list[dict[str, Any]]:
        """Return claims, optionally scoped to one document using the document index."""
        if doc_id is None:
            cursor = self.conn.execute("SELECT * FROM claims ORDER BY doc_id, claim_id")
        else:
            cursor = self.conn.execute("SELECT * FROM claims WHERE doc_id=? ORDER BY claim_id", (str(doc_id),))
        rows = []
        for r in cursor:
            d = dict(r); d["metadata"] = json.loads(d.pop("metadata_json", "{}")); rows.append(d)
        return rows

    def find_competing_claims(self) -> list[dict[str, Any]]:
        """Group alternative objects by subject/predicate but preserve all sources.

        Label as 'competing claims' because many predicates are naturally multi-valued;
        the result intentionally does not choose the winning source.
        """
        from collections import defaultdict
        groups = defaultdict(list)
        for c in self.get_claims():
            key = (_norm(c["subject"]), _norm(c["relation_type"]))
            groups[key].append(c)
        findings = []
        for key, rows in groups.items():
            objects = {_norm(c["object"]) for c in rows}
            docs = {c["doc_id"] for c in rows}
            if len(objects) > 1 and len(docs) > 1:
                findings.append({"category": "competing_claims_requires_review", "subject": rows[0]["subject"],
                                 "relation_type": rows[0]["relation_type"], "claims": [
                                     {"object": c["object"], "doc_id": c["doc_id"], "source_title": c["source_title"],
                                      "source_url": c["source_url"], "chunk_id": c["chunk_id"],
                                      "evidence_text": c["evidence_text"], "status": c["status"]} for c in rows]})
        return findings

    def get_document_history(self, doc_id: str) -> list[dict[str, Any]]:
        """Return prior-version metadata and claims, not whole raw documents by default."""
        out = []
        for row in self.conn.execute("SELECT * FROM document_history WHERE doc_id=? ORDER BY version", (str(doc_id),)):
            item = dict(row)
            claims = json.loads(item.pop("claims_json", "[]"))
            item.pop("raw_text", None)
            item.pop("safe_text", None)
            item["claims"] = claims
            item["content_length_chars"] = int(row["content_length_chars"] or len(str(row["raw_text"] or "")))
            out.append(item)
        return out

    def metrics(self) -> dict[str, Any]:
        return {"documents": self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                "chunks": self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
                "claims": self.conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0],
                "topic_links": self.conn.execute("SELECT COUNT(*) FROM topic_links").fetchone()[0],
                "document_versions_archived": self.conn.execute("SELECT COUNT(*) FROM document_history").fetchone()[0],
                "graph_nodes": self.graph.number_of_nodes(), "graph_edges": self.graph.number_of_edges(),
                "graph_bytes_estimate": len(json.dumps(nx.node_link_data(self.graph), default=str).encode("utf-8"))}
