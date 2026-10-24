"""Sentence-aware chunking with page provenance, injection quarantine and stable IDs.

Design choices
  * Chunks never cross page boundaries -> every chunk has exactly one valid page number.
  * Boundaries fall on sentence ends; overlap is carried as whole trailing sentences.
  * chunk_id is CONTENT-ADDRESSED (doc_id + page + text). If a document is edited, every
    unchanged chunk keeps its ID, so graph extraction / embeddings are redone only for
    chunks that really changed (this is what makes the 60-second update target feasible).
  * Instruction-like sentences are removed from text and kept in Chunk.quarantined.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from models import Chunk, Document
from core.security import is_injection


@dataclass(frozen=True)
class ChunkConfig:
    max_chars: int = 700        # target upper bound (spec: ~500-800)
    min_chars: int = 150        # smaller tails are merged into the previous chunk
    overlap_chars: int = 100    # trailing sentences re-used at the start of the next chunk
    hard_max: int = 1000        # any single sentence longer than this is window-split


DEFAULT_CONFIG = ChunkConfig()

_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "inc", "ltd", "co", "corp",
           "fig", "eq", "no", "al", "e.g", "i.e", "u.s", "u.k", "approx", "dept", "est", "vol", "pp"}
_SENT_SPLIT = re.compile(r'(?<=[.!?।])["\')\]]*\s+(?=["(\[]?[A-Z0-9\u0900-\u097F])')


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        parts = _SENT_SPLIT.split(para)
        buf = ""
        for p in parts:
            buf = f"{buf} {p}".strip() if buf else p
            last = re.findall(r"([A-Za-z.]+)[.!?]*$", buf.rstrip('"\')]'))
            tail = last[-1].lower().rstrip(".") if last else ""
            if tail in _ABBREV or re.search(r"(?:^|\s)[A-Z]\.$", buf):
                continue          # false split after an abbreviation / initial: keep accumulating
            out.append(buf)
            buf = ""
        if buf:
            out.append(buf)
    return out


def _window_split(s: str, size: int) -> list[str]:
    words, cur, parts = s.split(), [], []
    n = 0
    for w in words:
        if n + len(w) + 1 > size and cur:
            parts.append(" ".join(cur))
            cur, n = [], 0
        cur.append(w)
        n += len(w) + 1
    if cur:
        parts.append(" ".join(cur))
    return parts


def _cid(doc_id: str, page: int, text: str, seen: dict[str, int]) -> str:
    h = hashlib.sha1(f"{doc_id}|{page}|{text}".encode("utf-8")).hexdigest()[:12]
    cid = f"c_{h}"
    n = seen.get(cid, 0)
    seen[cid] = n + 1
    return cid if n == 0 else f"{cid}-{n + 1}"      # identical repeated passages stay unique


def _pack(sentences: list[str], cfg: ChunkConfig) -> list[str]:
    units: list[str] = []
    for s in sentences:
        units.extend(_window_split(s, cfg.hard_max) if len(s) > cfg.hard_max else [s])

    chunks: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for u in units:
        if cur and size + len(u) + 1 > cfg.max_chars:
            chunks.append(cur)
            carry, c = [], 0
            for prev in reversed(cur):               # overlap = whole trailing sentences
                if c + len(prev) > cfg.overlap_chars * 1.5 and carry:
                    break
                carry.insert(0, prev)
                c += len(prev) + 1
                if c >= cfg.overlap_chars:
                    break
            cur = carry if sum(map(len, carry)) < cfg.max_chars * 0.5 else []
            size = sum(len(x) + 1 for x in cur)
        cur.append(u)
        size += len(u) + 1
    if cur:
        chunks.append(cur)

    texts = [" ".join(c) for c in chunks]
    # merge a too-small tail into its predecessor (it is mostly overlap anyway)
    if len(texts) >= 2 and len(texts[-1]) < cfg.min_chars:
        tail_unique = " ".join(chunks[-1][len([x for x in chunks[-1] if x in chunks[-2]]):])
        if len(texts[-2]) + len(tail_unique) <= cfg.max_chars * 1.4:
            texts[-2] = (texts[-2] + " " + tail_unique).strip()
            texts.pop()
    return texts


def chunk_document(doc: Document, cfg: ChunkConfig = DEFAULT_CONFIG) -> list[Chunk]:
    if doc.status != "ok":
        return []
    out: list[Chunk] = []
    seen: dict[str, int] = {}
    pending_q: list[str] = []            # quarantined sentences waiting for a chunk to attach to
    idx = 0
    for page_no, page_text in enumerate(doc.pages, start=1):
        if not page_text.strip():
            continue
        clean: list[str] = []
        for s in split_sentences(page_text):
            (pending_q if is_injection(s) else clean).append(s)
        for t in (_pack(clean, cfg) if clean else []):
            t = t.strip()
            if len(t) < 3:
                continue
            out.append(Chunk(chunk_id=_cid(doc.doc_id, page_no, t, seen), doc_id=doc.doc_id,
                             page=page_no, text=t, chunk_index=idx, quarantined=pending_q))
            pending_q = []               # attached once, to the first chunk at/after the injection
            idx += 1
    if pending_q and out:                # injections after the last clean chunk
        out[-1].quarantined = out[-1].quarantined + pending_q
    return out


def chunk_documents(documents: list[Document], cfg: ChunkConfig = DEFAULT_CONFIG) -> list[Chunk]:
    """Agreed interface. Skips duplicate / error / empty documents. Output order is deterministic."""
    result: list[Chunk] = []
    for d in documents:
        result.extend(c for c in chunk_document(d, cfg))
    return result
