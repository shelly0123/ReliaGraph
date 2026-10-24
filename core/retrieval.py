"""Hybrid retrieval: TF-IDF (lexical) + multilingual dense embeddings, fused with RRF.

  * Works with ZERO optional dependencies (TF-IDF only) -> app always runs.
  * If sentence-transformers + a local model exist, dense retrieval is added. The default
    model (paraphrase-multilingual-MiniLM-L12-v2) embeds Hindi and English into one space,
    which is what makes Hindi questions over English documents work (Phase 2).
  * Embeddings are cached by chunk_id (content-addressed) -> an edited document re-embeds
    only its changed chunks.
  * Retriever.update() is incremental; ChromaMirror keeps a persistent ChromaDB copy in sync.
  * search() returns hits with scores; retrieve() keeps the agreed list[Chunk] signature.

PRE-DOWNLOAD the model before the event (no network at venue):
    python -c "from sentence_transformers import SentenceTransformer as S; S('paraphrase-multilingual-MiniLM-L12-v2')"
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from models import Chunk

DEFAULT_MODEL = os.environ.get("TOPICWEAVE_EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
RRF_K = 60
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def detect_lang(text: str) -> str:
    """'hi' if mostly Devanagari, else 'en' (good enough to route Hindi vs English)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    return "hi" if sum(bool(_DEVANAGARI.match(c)) for c in letters) / len(letters) > 0.3 else "en"


# ============================================================================ embeddings
class Embedder:
    """Lazy, cached, optional. `available` is False when the library/model is missing."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 64, max_seq_length: int = 256):
        self.model_name, self.batch_size, self.max_seq_length = model_name, batch_size, max_seq_length
        self._model = None
        self._tried = False
        self._lock = threading.Lock()
        self.cache: dict[str, np.ndarray] = {}

    @property
    def available(self) -> bool:
        if not self._tried:
            with self._lock:
                if not self._tried:
                    try:
                        from sentence_transformers import SentenceTransformer
                        self._model = SentenceTransformer(self.model_name)
                        self._model.max_seq_length = self.max_seq_length
                    except Exception:
                        self._model = None
                    self._tried = True
        return self._model is not None

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        assert self._model is not None
        v = self._model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False)
        return v.astype(np.float32)

    def encode_chunks(self, chunks: Sequence[Chunk]) -> np.ndarray:
        """(n, d) matrix, computing only chunks missing from the cache."""
        todo = [c for c in chunks if c.chunk_id not in self.cache]
        if todo:
            vecs = self.encode_texts([c.text for c in todo])
            for c, v in zip(todo, vecs):
                self.cache[c.chunk_id] = v
        if not chunks:
            return np.zeros((0, 1), dtype=np.float32)
        return np.stack([self.cache[c.chunk_id] for c in chunks])

    def forget(self, chunk_ids: Iterable[str]) -> None:
        for i in chunk_ids:
            self.cache.pop(i, None)

    def save(self, path: str) -> None:
        ids = list(self.cache)
        if ids:
            np.savez_compressed(path, ids=np.array(ids), vecs=np.stack([self.cache[i] for i in ids]))

    def load(self, path: str) -> None:
        if os.path.exists(path):
            z = np.load(path, allow_pickle=False)
            self.cache.update({str(i): v for i, v in zip(z["ids"], z["vecs"])})


_EMBEDDER: Embedder | None = None


def get_embedder() -> Embedder:
    """Process-wide singleton so topics + retrieval share one model and one cache."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER


# ============================================================================ chroma
class ChromaMirror:
    """Persistent ChromaDB copy of the dense index (project deliverable #2). Optional."""

    def __init__(self, path: str = "data/chroma", name: str = "topicweave_chunks"):
        import chromadb
        self.col = chromadb.PersistentClient(path=path).get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"})

    def sync(self, add: Sequence[Chunk], vectors: np.ndarray | None, remove_ids: Iterable[str]) -> None:
        rm = list(remove_ids)
        if rm:
            self.col.delete(ids=rm)
        if add and vectors is not None:
            for i in range(0, len(add), 500):
                part = add[i:i + 500]
                self.col.upsert(
                    ids=[c.chunk_id for c in part], embeddings=vectors[i:i + 500].tolist(),
                    documents=[c.text for c in part],
                    metadatas=[{"doc_id": c.doc_id, "page": int(c.page),
                                "topic_ids": ",".join(c.topic_ids)} for c in part])

    def query(self, vector: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        r = self.col.query(query_embeddings=[vector.tolist()], n_results=k)
        return [(i, 1.0 - d) for i, d in zip(r["ids"][0], r["distances"][0])]


# ============================================================================ retriever
@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float                                   # fused (RRF) score, higher = better
    rank: int
    scores: dict[str, float] = field(default_factory=dict)   # raw per-method scores (cosine)
    method: str = "tfidf"

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class Retriever:
    def __init__(self, embedder: Embedder | None = None, use_dense: bool | str = "auto",
                 query_translator: Callable[[str], str] | None = None,
                 mirror: ChromaMirror | None = None):
        self.embedder = embedder if embedder is not None else get_embedder()
        self.use_dense = use_dense
        self.translator = query_translator
        self.mirror = mirror
        self.last_warning = ""
        self._lock = threading.RLock()
        self._chunks: dict[str, Chunk] = {}
        self._ids: list[str] = []
        self._vec: TfidfVectorizer | None = None
        self._X = None
        self._D: np.ndarray | None = None

    # ------------------------------------------------------------------ state
    @property
    def dense_enabled(self) -> bool:
        if self.use_dense is False:
            return False
        return self.embedder.available

    def __len__(self) -> int:
        return len(self._ids)

    def _rebuild(self) -> None:
        self._ids = list(self._chunks)
        texts = [self._chunks[i].text for i in self._ids]
        if not texts:
            self._vec, self._X, self._D = None, None, None
            return
        n = len(texts)
        self._vec = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", ngram_range=(1, 2),
                                    stop_words="english", min_df=1, max_df=0.85 if n > 20 else 1.0,
                                    max_features=300_000, dtype=np.float32)
        try:
            self._X = self._vec.fit_transform(texts)
        except ValueError:                         # empty vocabulary (e.g. all stop-words)
            self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), dtype=np.float32)
            self._X = self._vec.fit_transform(texts)
        if self.dense_enabled:
            self._D = self.embedder.encode_chunks([self._chunks[i] for i in self._ids])

    # ------------------------------------------------------------------ index API
    def index(self, chunks: Sequence[Chunk]) -> "Retriever":
        with self._lock:
            self._chunks = {c.chunk_id: c for c in chunks if c.text.strip()}
            self._rebuild()
            if self.mirror and self._D is not None:
                self.mirror.sync(list(self._chunks.values()), self._D, [])
        return self

    def update(self, add: Sequence[Chunk] = (), remove_ids: Iterable[str] = ()) -> "Retriever":
        """Incremental: only NEW chunks are embedded; TF-IDF (cheap) is refit over the corpus."""
        with self._lock:
            rm = [i for i in remove_ids if i in self._chunks]
            for i in rm:
                del self._chunks[i]
            self.embedder.forget(rm)
            new = [c for c in add if c.text.strip()]
            for c in new:
                self._chunks[c.chunk_id] = c
            self._rebuild()
            if self.mirror:
                vecs = (np.stack([self.embedder.cache[c.chunk_id] for c in new])
                        if new and self.dense_enabled else None)
                self.mirror.sync(new, vecs, rm)
        return self

    # ------------------------------------------------------------------ search
    @staticmethod
    def _top(scores: np.ndarray, k: int, mask: np.ndarray | None) -> np.ndarray:
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        k = min(k, scores.shape[0])
        if k <= 0:
            return np.array([], dtype=int)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return idx[np.isfinite(scores[idx])]

    def search(self, query: str, top_k: int = 5, *, doc_ids: Iterable[str] | None = None,
               topic_id: str | None = None, chunk_ids: Iterable[str] | None = None,
               min_score: float = 0.0) -> list[RetrievalHit]:
        """Restrict by doc_ids / topic_id / chunk_ids to retrieve INSIDE a topic community."""
        self.last_warning = ""
        with self._lock:
            if not self._ids or not query.strip():
                return []
            n = len(self._ids)
            mask = None
            if doc_ids is not None or topic_id is not None or chunk_ids is not None:
                dset = set(doc_ids) if doc_ids is not None else None
                cset = set(chunk_ids) if chunk_ids is not None else None
                mask = np.fromiter(
                    ((dset is None or self._chunks[i].doc_id in dset) and
                     (cset is None or i in cset) and
                     (topic_id is None or topic_id in self._chunks[i].topic_ids) for i in self._ids),
                    dtype=bool, count=n)
                if not mask.any():
                    return []

            lang = detect_lang(query)
            q_sparse = query
            if lang == "hi" and self.translator:
                try:
                    q_sparse = self.translator(query) or query
                except Exception:
                    q_sparse = query
            sparse_ok = not (lang == "hi" and q_sparse == query)   # TF-IDF is useless on raw Hindi vs English docs

            pool = max(50, top_k * 10)
            ranked: dict[str, list[int]] = {}
            raw: dict[str, np.ndarray] = {}

            if sparse_ok and self._vec is not None:
                qv = self._vec.transform([q_sparse])
                s = (self._X @ qv.T).toarray().ravel()
                raw["tfidf"] = s
                ranked["tfidf"] = [i for i in self._top(s, pool, mask) if s[i] > 0]

            if self.dense_enabled and self._D is not None and self._D.shape[0] == n:
                qd = self.embedder.encode_texts([query])[0]
                s = self._D @ qd
                raw["dense"] = s
                ranked["dense"] = list(self._top(s, pool, mask))
            elif lang == "hi" and not sparse_ok:
                self.last_warning = ("Hindi query but no multilingual embedder or translator is available; "
                                     "install sentence-transformers or pass query_translator.")

            if not ranked:
                return []
            fused: dict[int, float] = {}
            for lst in ranked.values():
                for r, i in enumerate(lst):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
            order = sorted(fused, key=lambda i: (-fused[i], i))
            hits: list[RetrievalHit] = []
            for i in order:
                per = {m: float(raw[m][i]) for m in raw}
                best = max(per.values())
                if best < min_score:
                    continue
                hits.append(RetrievalHit(self._chunks[self._ids[i]], fused[i], len(hits) + 1, per,
                                         "+".join(ranked)))
                if len(hits) >= top_k:
                    break
            return hits


# ============================================================================ agreed interface
_CACHE: dict[str, Retriever] = {}


def _fingerprint(chunks: Sequence[Chunk]) -> str:
    h = hashlib.sha1()
    for i in sorted(c.chunk_id for c in chunks):
        h.update(i.encode())
    return h.hexdigest()


def retrieve(query: str, chunks: Sequence[Chunk], top_k: int = 5) -> list[Chunk]:
    """Agreed signature. Index is cached per chunk-set, so repeated calls are cheap."""
    key = _fingerprint(chunks)
    r = _CACHE.get(key)
    if r is None:
        if len(_CACHE) > 4:
            _CACHE.pop(next(iter(_CACHE)))
        r = _CACHE[key] = Retriever().index(chunks)
    return [h.chunk for h in r.search(query, top_k)]
