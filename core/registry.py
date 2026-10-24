"""Incremental corpus state: add / edit / delete a document WITHOUT a rebuild.

upsert(files) and remove(...) return a ChangeSet that says exactly which chunks are new
and which chunk IDs disappeared. Downstream modules only touch the delta:

    Retriever.update(add=cs.new_chunks, remove_ids=cs.removed_chunk_ids)
    TopicModel.update(registry.chunks, cs.new_chunk_ids, cs.removed_chunk_ids)
    graph_builder  -> re-extract only cs.new_chunks, drop evidence for cs.removed_chunk_ids

Because chunk IDs are content-addressed, an edit that changes one paragraph yields a
handful of new chunks, not a re-chunk of the whole document.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from models import Chunk, Document
from core.chunking import ChunkConfig, DEFAULT_CONFIG, chunk_document
from core.ingestion import extract_documents, make_doc_id


@dataclass
class ChangeSet:
    added: list[str] = field(default_factory=list)          # doc_ids
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    duplicates: list[Document] = field(default_factory=list)
    errors: list[Document] = field(default_factory=list)    # status error / empty
    new_chunks: list[Chunk] = field(default_factory=list)
    removed_chunk_ids: list[str] = field(default_factory=list)
    injections_blocked: int = 0
    elapsed_s: float = 0.0

    @property
    def new_chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.new_chunks]

    @property
    def is_noop(self) -> bool:
        return not (self.new_chunks or self.removed_chunk_ids)

    def summary(self) -> dict[str, Any]:
        return {"added": len(self.added), "updated": len(self.updated), "removed": len(self.removed),
                "unchanged": len(self.unchanged), "duplicates": len(self.duplicates),
                "errors": [(d.filename, d.error) for d in self.errors],
                "new_chunks": len(self.new_chunks), "removed_chunks": len(self.removed_chunk_ids),
                "injections_blocked": self.injections_blocked, "elapsed_s": round(self.elapsed_s, 3)}


class CorpusRegistry:
    def __init__(self, path: str | Path | None = None, cfg: ChunkConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        self.docs: dict[str, Document] = {}              # pages dropped after chunking (memory)
        self.chunks_by_doc: dict[str, list[Chunk]] = {}
        self._flat: list[Chunk] | None = None
        if self.path and self.path.exists():
            self.load()

    # ------------------------------------------------------------------ views
    @property
    def chunks(self) -> list[Chunk]:
        with self._lock:
            if self._flat is None:
                self._flat = [c for d in self.chunks_by_doc.values() for c in d]
            return self._flat

    def chunk_map(self) -> dict[str, Chunk]:
        return {c.chunk_id: c for c in self.chunks}

    def apply_topics(self, chunks: Iterable[Chunk]) -> None:
        """Write topic_ids from TopicModel output back into the stored chunks."""
        with self._lock:
            tmap = {c.chunk_id: c.topic_ids for c in chunks}
            for c in self.chunks:
                if c.chunk_id in tmap:
                    c.topic_ids = list(tmap[c.chunk_id])

    def document_hashes(self) -> dict[str, str]:
        with self._lock:
            return {d.doc_id: d.content_hash for d in self.docs.values()}

    # ------------------------------------------------------------------ mutate
    def upsert(self, files: Iterable[Any]) -> ChangeSet:
        t0 = time.perf_counter()
        cs = ChangeSet()
        incoming = extract_documents(files)                 # heavy work OUTSIDE the lock
        prepared: list[tuple[Document, list[Chunk]]] = []
        for d in incoming:
            if d.status == "duplicate":
                cs.duplicates.append(d)
            elif d.status != "ok":
                cs.errors.append(d)
            else:
                prepared.append((d, chunk_document(d, self.cfg)))

        with self._lock:
            hash_owner = {x.content_hash: x.doc_id for x in self.docs.values()}
            for d, new in prepared:
                owner = hash_owner.get(d.content_hash)
                if owner and owner != d.doc_id:             # same bytes already stored under another name
                    d.status, d.duplicate_of, d.pages = "duplicate", owner, []
                    d.error = f"identical content to {owner}"
                    cs.duplicates.append(d)
                    continue
                old_doc = self.docs.get(d.doc_id)
                if old_doc and old_doc.content_hash == d.content_hash:
                    cs.unchanged.append(d.doc_id)
                    continue
                old_chunks = self.chunks_by_doc.get(d.doc_id, [])
                old_ids = {c.chunk_id for c in old_chunks}
                new_ids = {c.chunk_id for c in new}
                cs.new_chunks += [c for c in new if c.chunk_id not in old_ids]
                cs.removed_chunk_ids += sorted(old_ids - new_ids)
                (cs.updated if old_doc else cs.added).append(d.doc_id)
                if old_doc:
                    hash_owner.pop(old_doc.content_hash, None)
                hash_owner[d.content_hash] = d.doc_id
                d.pages = []
                self.docs[d.doc_id] = d
                self.chunks_by_doc[d.doc_id] = new
            self._flat = None
            cs.injections_blocked = sum(len(c.quarantined) for c in cs.new_chunks)
            if self.path and not cs.is_noop:
                self.save()
        cs.elapsed_s = time.perf_counter() - t0
        return cs

    def remove(self, doc_ref: str) -> ChangeSet:
        """doc_ref = doc_id or filename."""
        t0 = time.perf_counter()
        cs = ChangeSet()
        with self._lock:
            did = doc_ref if doc_ref in self.docs else make_doc_id(doc_ref)
            if did in self.docs:
                cs.removed.append(did)
                cs.removed_chunk_ids = [c.chunk_id for c in self.chunks_by_doc.pop(did, [])]
                del self.docs[did]
                self._flat = None
                if self.path:
                    self.save()
        cs.elapsed_s = time.perf_counter() - t0
        return cs

    # ------------------------------------------------------------------ persistence
    def save(self) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"docs": [asdict(d) for d in self.docs.values()],
                   "chunks": {k: [asdict(c) for c in v] for k, v in self.chunks_by_doc.items()}}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)                              # atomic: no half-written registry

    def load(self) -> None:
        assert self.path is not None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.docs = {d["doc_id"]: Document(**d) for d in data["docs"]}
        self.chunks_by_doc = {k: [Chunk(**c) for c in v] for k, v in data["chunks"].items()}
        self._flat = None
