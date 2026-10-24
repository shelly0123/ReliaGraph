"""Document ingestion: bytes/paths -> clean, traceable Document objects.

Guarantees
  * Never raises for bad input: unreadable / empty / encrypted files come back with
    status="error"|"empty" and a human-readable reason.
  * doc_id is derived from the FILENAME  -> editing a file keeps its identity, so the
    registry can detect "updated" vs "added".
  * content_hash is SHA-256 of the BYTES -> identical uploads are detected as duplicates
    even under different names.
  * Scales: PyMuPDF if installed (10-50x faster), else pypdf; process pool for big batches.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import logging

from models import Document

logging.getLogger("pypdf").setLevel(logging.ERROR)   # corrupt PDFs are handled, not shouted about

MAX_BYTES = 80 * 1024 * 1024
MIN_TEXT_CHARS = 20
TEXT_EXT = {".txt", ".md", ".text", ".rst", ".csv", ".log"}

_ZW = dict.fromkeys(
    map(ord, "\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e"
             "\u2060\u2061\u2062\u2063\u2064\ufeff\u00ad"), None)
_CTRL = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")


# --------------------------------------------------------------------------- ids
def make_doc_id(filename: str) -> str:
    base = os.path.basename(filename).strip().lower()
    return "doc_" + hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- input
def _read_input(f: Any) -> tuple[str, bytes]:
    """Accept path | (name, bytes) | Streamlit UploadedFile | any file-like with .name."""
    if isinstance(f, (str, Path)):
        p = Path(f)
        return p.name, p.read_bytes()
    if isinstance(f, tuple) and len(f) == 2:
        name, data = f
        return str(name), bytes(data)
    name = getattr(f, "name", None) or "unnamed"
    if hasattr(f, "getvalue"):
        return os.path.basename(name), f.getvalue()
    if hasattr(f, "read"):
        try:
            f.seek(0)
        except Exception:
            pass
        return os.path.basename(name), f.read()
    raise TypeError(f"unsupported input type: {type(f)!r}")


# --------------------------------------------------------------------------- cleaning
def normalize_text(text: str) -> str:
    """Unicode-normalise, strip invisible/control chars, rejoin hyphenation and hard-wrapped lines."""
    t = unicodedata.normalize("NFKC", text).translate(_ZW)
    t = _CTRL.sub(" ", t).replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)               # de-hyphenate across lines
    t = re.sub(r"[ \t\f\v]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in t.split("\n\n")]
    return "\n\n".join(p for p in paras if p)


def _strip_running_headers(pages: list[str]) -> list[str]:
    """Drop lines (page numbers, running headers/footers) that repeat on >=60% of pages."""
    if len(pages) < 4:
        return pages
    sig = lambda ln: re.sub(r"\d+", "#", ln.strip().lower())
    counts: Counter[str] = Counter()
    for p in pages:
        lines = [ln for ln in p.split("\n") if ln.strip()]
        counts.update({sig(ln) for ln in lines[:2] + lines[-2:]})
    cut = {s for s, c in counts.items() if c >= 0.6 * len(pages) and len(s) < 120}
    if not cut:
        return pages
    out = []
    for p in pages:
        lines = p.split("\n")
        keep = [ln for i, ln in enumerate(lines)
                if not (sig(ln) in cut and (i < 2 or i >= len(lines) - 2))]
        out.append("\n".join(keep))
    return out


# --------------------------------------------------------------------------- extraction
def _pdf_pages(data: bytes) -> list[str]:
    try:  # fast path (optional dependency)
        import pymupdf  # type: ignore
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass and not doc.authenticate(""):
                raise ValueError("PDF is password-protected")
            return [pg.get_text("text") for pg in doc]
    except ImportError:
        pass
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data), strict=False)
    if reader.is_encrypted:
        try:
            if not reader.decrypt(""):
                raise ValueError("PDF is password-protected")
        except Exception as e:  # missing crypto lib or wrong password
            raise ValueError(f"PDF is encrypted ({e})")
    pages = []
    for pg in reader.pages:
        try:
            pages.append(pg.extract_text() or "")
        except Exception:
            pages.append("")   # one bad page must not lose the document
    return pages


def _decode(data: bytes) -> str:
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def _extract_one(item: tuple[str, bytes]) -> Document:
    """Module-level so it can run in a process pool."""
    name, data = item
    doc = Document(doc_id=make_doc_id(name), filename=os.path.basename(name), content_hash=content_hash(data))
    if not data:
        doc.status, doc.error = "empty", "file is empty"
        return doc
    if len(data) > MAX_BYTES:
        doc.status, doc.error = "error", f"file larger than {MAX_BYTES // 2**20} MB"
        return doc
    ext = Path(name).suffix.lower()
    try:
        if ext == ".pdf" or data[:5] == b"%PDF-":
            raw_pages = _pdf_pages(data)
        elif ext in TEXT_EXT or ext == "":
            raw_pages = _decode(data).split("\f")
        else:
            doc.status, doc.error = "error", f"unsupported file type '{ext}'"
            return doc
    except Exception as e:
        doc.status, doc.error = "error", f"could not read file: {type(e).__name__}: {e}"
        return doc

    # strip repeated headers BEFORE line-joining, then normalise
    raw_pages = _strip_running_headers([p.replace("\r", "") for p in raw_pages])
    pages = [normalize_text(p) for p in raw_pages]
    if sum(len(p) for p in pages) < MIN_TEXT_CHARS:
        doc.status = "empty"
        doc.error = ("no extractable text (scanned/image-only PDF? OCR is not supported)"
                     if ext == ".pdf" else "no text content")
        return doc
    doc.pages = pages
    return doc


def extract_documents(files: Iterable[Any], *, max_workers: int | None = None,
                      parallel_threshold: int = 12) -> list[Document]:
    """Read every input into a Document. Order preserved. Never raises per-file.

    Within one call, a file whose bytes match an earlier one becomes status="duplicate"
    (duplicate_of=<doc_id of first>) and carries no pages.
    """
    items: list[tuple[str, bytes]] = []
    for f in files:
        try:
            items.append(_read_input(f))
        except Exception as e:
            name = str(getattr(f, "name", f))
            items.append((name, b""))
            # surfaced below as an empty/err doc
            _ = e

    docs: list[Document]
    workers = max_workers if max_workers is not None else min(os.cpu_count() or 1, 8)
    if workers > 1 and len(items) >= parallel_threshold:
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                docs = list(ex.map(_extract_one, items, chunksize=max(1, len(items) // (workers * 4))))
        except Exception:      # spawn problems (Windows/Streamlit), pickling, etc.
            docs = [_extract_one(i) for i in items]
    else:
        docs = [_extract_one(i) for i in items]

    seen: dict[str, str] = {}
    for d in docs:
        if d.status != "ok":
            continue
        first = seen.setdefault(d.content_hash, d.doc_id)
        if first != d.doc_id:
            d.status, d.duplicate_of, d.pages = "duplicate", first, []
            d.error = f"identical content to {first}"
    return docs
