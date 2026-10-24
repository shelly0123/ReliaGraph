"""Scalable topic discovery + incremental topic assignment.

Pipeline
  features : multilingual embeddings (if available)  else  TF-IDF -> TruncatedSVD (LSA)
  cluster  : k chosen by silhouette on a sample; KMeans (<=5k chunks) / MiniBatchKMeans
  clean-up : tiny clusters are folded into their nearest large cluster
  labels   : class-based TF-IDF (c-TF-IDF, as in BERTopic) -> readable keywords per topic
  soft     : a chunk may belong to a 2nd topic when it is nearly as close (bridges for graph)
  update() : new chunks are assigned to existing centroids (no refit, topic IDs stay stable);
             a refit is triggered only when the corpus has drifted (>25% churn / poor fit).

discover_topics(chunks) keeps the agreed signature.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from models import Chunk, Topic

SEED = 42


class TopicModel:
    def __init__(self, k: int | None = None, min_k: int = 3, max_k: int = 25, min_cluster_frac: float = 0.01,
                 use_embeddings: bool | str = "auto", max_topics_per_chunk: int = 2,
                 soft_ratio: float = 0.92, refit_churn: float = 0.25, embedder=None):
        self.k, self.min_k, self.max_k = k, min_k, max_k
        self.min_cluster_frac = min_cluster_frac
        self.use_embeddings = use_embeddings
        self.max_topics_per_chunk, self.soft_ratio, self.refit_churn = max_topics_per_chunk, soft_ratio, refit_churn
        self._embedder = embedder
        self.topics: list[Topic] = []
        self.centroids: np.ndarray | None = None
        self.assign: dict[str, list[str]] = {}      # chunk_id -> topic_ids
        self._tfidf: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self._mode = "tfidf"
        self._n_fit = 0
        self._churn = 0
        self._fit_sim = 1.0

    # ------------------------------------------------------------------ features
    def _get_embedder(self):
        if self._embedder is None:
            from core.retrieval import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def _dense_ok(self) -> bool:
        if self.use_embeddings is False:
            return False
        try:
            return self._get_embedder().available
        except Exception:
            return False

    def _features_fit(self, chunks: Sequence[Chunk]) -> np.ndarray:
        n = len(chunks)
        if self._dense_ok():
            self._mode = "dense"
            return self._get_embedder().encode_chunks(chunks)
        self._mode = "tfidf"
        self._tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english", ngram_range=(1, 2),
                                      min_df=2 if n >= 50 else 1, max_df=0.5 if n >= 20 else 1.0,
                                      max_features=50_000, dtype=np.float32)
        try:
            X = self._tfidf.fit_transform([c.text for c in chunks])
        except ValueError:
            self._tfidf = TfidfVectorizer(max_features=50_000, dtype=np.float32)
            X = self._tfidf.fit_transform([c.text for c in chunks])
        comps = max(2, min(100, X.shape[1] - 1, n - 1))
        self._svd = TruncatedSVD(n_components=comps, random_state=SEED)
        return normalize(self._svd.fit_transform(X))

    def _features_new(self, chunks: Sequence[Chunk]) -> np.ndarray:
        if self._mode == "dense":
            return self._get_embedder().encode_chunks(chunks)
        return normalize(self._svd.transform(self._tfidf.transform([c.text for c in chunks])))

    # ------------------------------------------------------------------ fit
    def _choose_k(self, F: np.ndarray) -> int:
        n = len(F)
        hi = max(self.min_k, min(self.max_k, n // 8))
        if hi <= self.min_k:
            return min(self.min_k, max(1, n // 2))
        cands = sorted({int(round(x)) for x in np.linspace(self.min_k, hi, num=min(6, hi - self.min_k + 1))})
        rng = np.random.RandomState(SEED)
        idx = rng.choice(n, size=min(2000, n), replace=False)
        best_k, best = cands[0], -2.0
        for k in cands:
            lab = MiniBatchKMeans(k, n_init=3, random_state=SEED, batch_size=1024).fit_predict(F[idx])
            if len(set(lab)) < 2:
                continue
            s = silhouette_score(F[idx], lab)
            if s > best + 0.005:                        # prefer smaller k unless clearly better
                best_k, best = k, s
        return best_k

    def fit(self, chunks: Sequence[Chunk]) -> tuple[list[Topic], list[Chunk]]:
        chunks = [c for c in chunks if c.text.strip()]
        n = len(chunks)
        self.topics, self.assign, self._churn, self._n_fit = [], {}, 0, n
        if n == 0:
            return [], []
        if n < 6:                                       # too small to cluster
            labels = np.zeros(n, dtype=int)
            F = np.ones((n, 1), dtype=np.float32)
            self._mode = "single"
            self.centroids = F[:1].copy()
        else:
            F = self._features_fit(chunks)
            k = self.k or self._choose_k(F)
            km_cls = KMeans if n <= 5000 else MiniBatchKMeans
            kw = {"n_init": 5} if km_cls is KMeans else {"n_init": 5, "batch_size": 2048}
            km = km_cls(n_clusters=k, random_state=SEED, **kw).fit(F)
            labels, cent = km.labels_.copy(), normalize(km.cluster_centers_)
            labels, cent = self._fold_small(F, labels, cent)
            self.centroids = cent

        # order topics by size so t0 is the largest
        uniq, counts = np.unique(labels, return_counts=True)
        order = [u for _, u in sorted(zip(-counts, uniq))]
        remap = {old: new for new, old in enumerate(order)}
        labels = np.array([remap[x] for x in labels])
        self.centroids = self.centroids[order]
        keywords = self._ctfidf_keywords(chunks, labels, len(order))
        sims = F @ self.centroids.T if self._mode != "single" else np.ones((n, 1))
        self._fit_sim = float(np.mean(sims[np.arange(n), labels]))
        for ci, ch in enumerate(chunks):
            self.assign[ch.chunk_id] = self._topics_for(sims[ci], int(labels[ci]))
        self.topics = [Topic(topic_id=f"t{i}", label=_label(keywords[i]), keywords=keywords[i]) for i in range(len(order))]
        return self._finalise(chunks)

    def _fold_small(self, F, labels, cent):
        n, k = len(labels), len(cent)
        min_size = max(3, int(self.min_cluster_frac * n))
        sizes = np.bincount(labels, minlength=k)
        small = {i for i in range(k) if sizes[i] < min_size}
        if not small or len(small) == k:
            return labels, cent
        keep = [i for i in range(k) if i not in small]
        sims = F @ cent[keep].T
        for i in np.where(np.isin(labels, list(small)))[0]:
            labels[i] = keep[int(np.argmax(sims[i]))]
        remap = {old: new for new, old in enumerate(keep)}
        return np.array([remap[x] for x in labels]), cent[keep]

    def _topics_for(self, sim_row: np.ndarray, best: int) -> list[str]:
        out = [f"t{best}"]
        if self.max_topics_per_chunk > 1 and len(sim_row) > 1:
            order = np.argsort(-sim_row)
            for j in order[1:self.max_topics_per_chunk]:
                if sim_row[best] > 0 and sim_row[j] >= self.soft_ratio * sim_row[best] and sim_row[j] > 0.2:
                    out.append(f"t{int(j)}")
        return out

    def _ctfidf_keywords(self, chunks, labels, k, top_n: int = 8) -> list[list[str]]:
        cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1,
                             token_pattern=r"(?u)\b[^\W\d_]{3,}\b", max_features=100_000)
        try:
            C = cv.fit_transform([c.text for c in chunks])
        except ValueError:
            return [["general"] for _ in range(k)]
        import scipy.sparse as sp
        ind = sp.csr_matrix((np.ones(len(labels)), (labels, np.arange(len(labels)))), shape=(k, len(labels)))
        T = (ind @ C).astype(np.float64)                          # class x term counts
        tf = normalize(T, norm="l1", axis=1)
        avg_words = T.sum() / max(k, 1)
        freq = np.asarray(T.sum(axis=0)).ravel()
        idf = np.log(1 + avg_words / np.maximum(freq, 1))
        scores = tf.multiply(idf).tocsr()
        vocab = cv.get_feature_names_out()
        out = []
        for i in range(k):
            row = scores.getrow(i)
            if row.nnz == 0:
                out.append(["general"])
                continue
            idx = row.indices[np.argsort(-row.data)][:40]
            picked: list[str] = []
            for t in vocab[idx]:
                words = set(t.split())
                if any(words <= set(p.split()) or set(p.split()) <= words for p in picked):
                    continue                                    # drop "learning" if "machine learning" is in
                picked.append(t)
                if len(picked) == top_n:
                    break
            out.append(picked)
        return out

    def _finalise(self, chunks: Sequence[Chunk]) -> tuple[list[Topic], list[Chunk]]:
        members: dict[str, list[str]] = {t.topic_id: [] for t in self.topics}
        out = []
        for c in chunks:
            tids = [t for t in self.assign.get(c.chunk_id, []) if t in members]
            for t in tids:
                members[t].append(c.chunk_id)
            out.append(dataclasses.replace(c, topic_ids=tids))
        for t in self.topics:
            t.chunk_ids = members[t.topic_id]
        return self.topics, out

    # ------------------------------------------------------------------ incremental
    def needs_refit(self, current: Sequence[Chunk]) -> bool:
        if self.centroids is None or self._mode == "single":
            return True
        return self._churn / max(self._n_fit, 1) > self.refit_churn or len(current) > 2 * max(self._n_fit, 1)

    def update(self, chunks: Sequence[Chunk], new_ids: Sequence[str] = (), removed_ids: Sequence[str] = ()
               ) -> tuple[list[Topic], list[Chunk]]:
        """`chunks` = FULL current chunk list. Cheap path: assign only `new_ids` to old centroids."""
        chunks = [c for c in chunks if c.text.strip()]
        self._churn += len(new_ids) + len(removed_ids)
        if self.needs_refit(chunks):
            return self.fit(chunks)
        for r in removed_ids:
            self.assign.pop(r, None)
        newset = set(new_ids)
        fresh = [c for c in chunks if c.chunk_id in newset or c.chunk_id not in self.assign]
        if fresh:
            F = self._features_new(fresh)
            sims = F @ self.centroids.T
            if float(np.mean(sims.max(axis=1))) < 0.6 * self._fit_sim:      # new material doesn't fit
                return self.fit(chunks)
            for i, c in enumerate(fresh):
                self.assign[c.chunk_id] = self._topics_for(sims[i], int(np.argmax(sims[i])))
        return self._finalise(chunks)


def discover_topics(chunks: Sequence[Chunk], **kw) -> tuple[list[Topic], list[Chunk]]:
    """Agreed interface: (topics, chunks-with-topic_ids)."""
    return TopicModel(**kw).fit(chunks)


def _label(keywords: list[str]) -> str:
    return " / ".join(k.title() for k in keywords[:3]) or "General"
