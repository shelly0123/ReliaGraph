"""Optional zero-cost local multilingual semantic retrieval (e.g. Hindi -> English).

Model: intfloat/multilingual-e5-small; weights are downloaded only after the caller
explicitly enables the adapter. E5 expects `query: ` and `passage: ` prefixes for
asymmetric retrieval. Model scores are rankings, not calibrated probabilities.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
import time
import re
from typing import Any, Mapping, Sequence

@dataclass
class RetrievalHit:
    chunk_id: str
    doc_id: str
    score: float
    text: str
    source_title: str = ""
    source_url: str = ""
    def to_dict(self): return asdict(self)

class MultilingualE5Retriever:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", device: str = "cpu", cache_path: str = "data/cache/e5_chunks.sqlite3"):
        self.model_name, self.device, self.cache_path = model_name, device, str(cache_path)
        self._model = None
        self.load_seconds = None
        self.last_query_seconds = 0.0
        self.model_load_attempted = False
        self._cache = None

    def _load(self):
        if self._model is not None: return
        self.model_load_attempted = True
        started = time.perf_counter()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install optional model dependencies from requirements-optional-models.txt") from exc
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self.load_seconds = time.perf_counter() - started
        from core.extraction_cache import SQLiteJSONCache
        self._cache = SQLiteJSONCache(self.cache_path)

    def _encode_passages(self, chunks: Sequence[Mapping[str, Any]]):
        self._load()
        import numpy as np
        vectors, missing_indices, missing_texts = [None] * len(chunks), [], []
        cache_keys = []
        for i, chunk in enumerate(chunks):
            text = str(chunk.get("text", ""))
            key = hashlib.sha256(f"{self.model_name}\0passage:\0{text}".encode("utf-8")).hexdigest()
            cache_keys.append(key)
            cached = self._cache.get(key) if self._cache else None
            if cached is not None:
                vectors[i] = np.asarray(cached, dtype=np.float32)
            else:
                missing_indices.append(i)
                # E5 is trained with query:/passage: prefixes for asymmetric retrieval.
                # The multilingual MiniLM fallback is not E5 and takes raw text.
                use_e5_prefix = "e5" in self.model_name.casefold()
                missing_texts.append(("passage: " + text) if use_e5_prefix else text)
        if missing_texts:
            encoded = self._model.encode(missing_texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
            for ix, vector in zip(missing_indices, encoded):
                vectors[ix] = np.asarray(vector, dtype=np.float32)
                if self._cache: self._cache.put(cache_keys[ix], vectors[ix].tolist())
        if not vectors: return np.empty((0, 0), dtype=np.float32)
        return np.vstack(vectors)

    def search(self, query: str, chunks: Sequence[Mapping[str, Any]], top_k: int = 5) -> list[RetrievalHit]:
        if not str(query).strip() or not chunks: return []
        import numpy as np
        started = time.perf_counter()
        self._load()
        passage_vectors = self._encode_passages(chunks)
        query_text = str(query).strip()
        if "e5" in self.model_name.casefold():
            query_text = "query: " + query_text
        query_vec = self._model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
        scores = passage_vectors @ np.asarray(query_vec, dtype=np.float32)
        indexes = np.argsort(-scores)[:max(1, int(top_k))]
        hits = []
        for ix in indexes:
            c = chunks[int(ix)]
            hits.append(RetrievalHit(chunk_id=str(c.get("chunk_id", f"chunk-{ix}")), doc_id=str(c.get("doc_id", "")),
                score=float(scores[int(ix)]), text=str(c.get("text", "")), source_title=str(c.get("source_title", "")),
                source_url=str(c.get("source_url", ""))))
        self.last_query_seconds = time.perf_counter() - started
        return hits

    def diagnostics(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "device": self.device, "load_seconds": self.load_seconds,
                "last_query_seconds": self.last_query_seconds, "cache_path": self.cache_path,
                "model_load_attempted": self.model_load_attempted}


def supporting_graph_paths(graph, retrieved_chunk_ids: Sequence[str], max_paths: int = 5, max_hops: int = 4) -> list[dict[str, Any]]:
    """Return source-grounded multi-edge paths containing at least one retrieved-evidence edge.

    All path steps must carry source evidence. At least one step must be anchored to a
    chunk retrieved for the query; adjacent steps may use additional graph evidence to
    complete a relationship chain. Paths are explanations, not proof that inference is valid.
    Supports NetworkX Graph/DiGraph/MultiGraph/MultiDiGraph. `max_hops` and output count
    are bounded to prevent graph traversal from becoming unbounded on large collections.
    """
    wanted = set(map(str, retrieved_chunk_ids))
    if not wanted or graph is None or graph.number_of_edges() == 0:
        return []
    max_paths, max_hops = max(1, int(max_paths)), max(1, min(int(max_hops), 5))
    import hashlib

    edge_rows: list[dict[str, Any]] = []
    if graph.is_multigraph():
        iterator = graph.edges(keys=True, data=True)
    else:
        iterator = ((u, v, str(data.get("claim_id", f"{u}->{v}")), data) for u, v, data in graph.edges(data=True))
    for u, v, key, data in iterator:
        # Topic-membership links organize navigation but are not factual claims.
        # Excluding them avoids explaining a factual relation chain with arbitrary
        # topic labels inserted as if they were real-world predicates.
        if str(data.get("relation_type", "")).casefold() in {"contains_concept", "topic_membership", "belongs_to_topic"} or str(data.get("status", "")) == "derived_topic_link":
            continue
        ev_ids = set(map(str, data.get("evidence_chunk_ids", [])))
        if not ev_ids:
            ev_ids = {str(x.get("chunk_id")) for x in data.get("evidence", []) if x.get("chunk_id")}
        if not ev_ids:
            continue
        ev_rows = [dict(x) for x in data.get("evidence", [])]
        eid = str(data.get("claim_id", key))
        edge_rows.append({
            "edge_id": eid, "source_node_id": str(u), "source": str(graph.nodes[u].get("label", u)),
            "relation": str(data.get("relation_type", "related_to")), "target_node_id": str(v),
            "target": str(graph.nodes[v].get("label", v)), "evidence_chunk_ids": sorted(ev_ids),
            "retrieved_evidence_chunk_ids": sorted(ev_ids & wanted),
            "source_url": str(data.get("source_url", "")), "source_title": str(data.get("source_title", "")),
            "doc_id": str(data.get("doc_id", "")), "evidence": ev_rows,
        })

    by_id = {e["edge_id"]: e for e in edge_rows}
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seeded: list[dict[str, Any]] = []
    for e in edge_rows:
        outgoing[e["source_node_id"]].append(e)
        incoming[e["target_node_id"]].append(e)
        if e["retrieved_evidence_chunk_ids"]:
            seeded.append(e)
    sort_edge = lambda e: (0 if e["retrieved_evidence_chunk_ids"] else 1, e["relation"].casefold(), e["target"].casefold(), e["edge_id"])
    for rows in outgoing.values(): rows.sort(key=sort_edge)
    for rows in incoming.values(): rows.sort(key=lambda e: (0 if e["retrieved_evidence_chunk_ids"] else 1, e["relation"].casefold(), e["source"].casefold(), e["edge_id"]))

    # Collect bounded prefixes/suffixes around each seed edge; never repeat a node in a path.
    candidates: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    explored = 0
    max_explored = max(500, max_paths * 250)

    def backward(node: str, seen: set[str], remaining: int) -> list[list[dict[str, Any]]]:
        nonlocal explored
        found = [[]]
        if remaining <= 0 or explored >= max_explored: return found
        for e in incoming.get(node, []):
            explored += 1
            prev = e["source_node_id"]
            if prev in seen: continue
            for prefix in backward(prev, seen | {prev}, remaining - 1):
                found.append(prefix + [e])
                if explored >= max_explored: break
        return found

    def forward(node: str, seen: set[str], remaining: int) -> list[list[dict[str, Any]]]:
        nonlocal explored
        found = [[]]
        if remaining <= 0 or explored >= max_explored: return found
        for e in outgoing.get(node, []):
            explored += 1
            nxt = e["target_node_id"]
            if nxt in seen: continue
            for suffix in forward(nxt, seen | {nxt}, remaining - 1):
                found.append([e] + suffix)
                if explored >= max_explored: break
        return found

    # Prefer multi-hop explanatory paths, then fall back to a single evidence edge.
    for seed in sorted(seeded, key=sort_edge):
        lefts = backward(seed["source_node_id"], {seed["source_node_id"]}, max_hops - 1)
        rights = forward(seed["target_node_id"], {seed["source_node_id"], seed["target_node_id"]}, max_hops - 1)
        for left in lefts:
            for right in rights:
                steps = left + [seed] + right
                if len(steps) > max_hops: continue
                # Ensure node continuity and no cycles, including interactions across left/right.
                labels = [steps[0]["source_node_id"]] + [s["target_node_id"] for s in steps]
                if len(set(labels)) != len(labels): continue
                if len(steps) == 1: continue
                key = tuple(s["edge_id"] for s in steps)
                candidates.setdefault(key, steps)
                if len(candidates) >= max_explored: break
            if len(candidates) >= max_explored: break
        if len(candidates) >= max_explored: break

    if not candidates:
        for seed in seeded:
            candidates[(seed["edge_id"],)] = [seed]
    ranked = sorted(candidates.items(), key=lambda kv: (-len(kv[1]), -sum(bool(s["retrieved_evidence_chunk_ids"]) for s in kv[1]), kv[0]))
    result = []
    for edge_ids, steps in ranked[:max_paths]:
        steps_out = [{k: v for k, v in step.items()} for step in steps]
        path_id = "path_" + hashlib.sha256("\0".join(edge_ids).encode("utf-8")).hexdigest()[:16]
        result.append({"path_id": path_id, "start_node_id": steps[0]["source_node_id"], "start": steps[0]["source"],
            "end_node_id": steps[-1]["target_node_id"], "end": steps[-1]["target"], "hop_count": len(steps),
            "seed_chunk_ids": sorted({cid for s in steps for cid in s["retrieved_evidence_chunk_ids"]}),
            "all_evidence_chunk_ids": sorted({cid for s in steps for cid in s["evidence_chunk_ids"]}),
            "steps": steps_out, "explanation_note": "Each step has source evidence; at least one step matches the query's retrieved chunks. This path is explanatory and does not prove the inference is correct."})
    return result


class ResilientMultilingualRetriever:
    """Cross-lingual retriever with local-model fallback and a safe lexical last resort.

    Model order:
      1. intfloat/multilingual-e5-small (E5 prefixes applied)
      2. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (raw text)
      3. lexical overlap fallback (does NOT provide general Hindi-to-English semantics)

    If both multilingual models fail, the fallback returns only passages with lexical
    overlap; for Hindi queries over English-only text, this will often return no results.
    That is intentional abstention, not successful cross-lingual retrieval. The caller
    must surface the degraded mode and should ask the user to re-enable a local model.
    """
    DEFAULT_PRIMARY = "intfloat/multilingual-e5-small"
    DEFAULT_SECONDARY = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, *, cache_dir: str = "data/cache", device: str = "cpu",
                 primary_model_name: str = DEFAULT_PRIMARY,
                 fallback_model_name: str = DEFAULT_SECONDARY,
                 enable_secondary_model: bool = True):
        self.cache_dir = cache_dir
        self.device = device
        self.primary = MultilingualE5Retriever(primary_model_name, device=device,
                                               cache_path=f"{cache_dir}/e5_chunks.sqlite3")
        self.secondary = MultilingualE5Retriever(fallback_model_name, device=device,
                                                 cache_path=f"{cache_dir}/multilingual_minilm_chunks.sqlite3") if enable_secondary_model else None
        self.enable_secondary_model = bool(enable_secondary_model)
        self._failed_models: set[str] = set()
        self._active: MultilingualE5Retriever | None = None
        self._mode = "not_started"
        self._errors: list[dict[str, str]] = []
        self._last_fallback_note = ""
        self.last_query_seconds = 0.0

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {x.casefold() for x in re.findall(r"[\w]+", str(text or ""), flags=re.UNICODE) if len(x) > 1}

    @staticmethod
    def _contains_devanagari(text: str) -> bool:
        return bool(re.search(r"[\u0900-\u097f]", str(text or "")))

    def _lexical_search(self, query: str, chunks: Sequence[Mapping[str, Any]], top_k: int) -> list[RetrievalHit]:
        qtokens = self._tokens(query)
        rows: list[tuple[float, Mapping[str, Any]]] = []
        for chunk in chunks:
            text = str(chunk.get("text", ""))
            dtokens = self._tokens(text)
            shared = qtokens & dtokens
            if not shared:
                continue
            # Jaccard is only a lexical overlap ranking signal, not semantic relevance.
            score = len(shared) / max(1, len(qtokens | dtokens))
            rows.append((score, chunk))
        rows.sort(key=lambda row: (-row[0], str(row[1].get("chunk_id", ""))))
        return [RetrievalHit(chunk_id=str(c.get("chunk_id", f"chunk-{i}")),
                             doc_id=str(c.get("doc_id", "")), score=float(score),
                             text=str(c.get("text", "")), source_title=str(c.get("source_title", "")),
                             source_url=str(c.get("source_url", "")))
                for i, (score, c) in enumerate(rows[:max(1, int(top_k))])]

    def search(self, query: str, chunks: Sequence[Mapping[str, Any]], top_k: int = 5) -> list[RetrievalHit]:
        started = time.perf_counter()
        if not str(query).strip() or not chunks:
            self.last_query_seconds = time.perf_counter() - started
            self._mode = "empty_input"
            return []
        adapters = [self.primary] + ([self.secondary] if self.secondary is not None else [])
        for adapter in adapters:
            if adapter is None or adapter.model_name in self._failed_models:
                continue
            try:
                hits = adapter.search(query, chunks, top_k=top_k)
                self._active = adapter
                self._mode = "multilingual_model"
                self._last_fallback_note = ""
                self.last_query_seconds = time.perf_counter() - started
                return hits
            except Exception as exc:
                self._failed_models.add(adapter.model_name)
                self._errors.append({"model": adapter.model_name, "error_type": type(exc).__name__,
                                     "message": str(exc)[:240]})
        # Last resort: lexical overlap only. Never describe this as cross-lingual semantic retrieval.
        hits = self._lexical_search(query, chunks, top_k)
        is_hi_to_en = self._contains_devanagari(query) and all(not self._contains_devanagari(str(c.get("text", ""))) for c in chunks)
        self._mode = "lexical_fallback_limited_crosslingual" if is_hi_to_en else "lexical_fallback_same_language_only"
        self._last_fallback_note = (
            "Multilingual models are unavailable. No general Hindi-to-English semantic capability is active; "
            "empty results are an intentional safe abstention. Enable a working local multilingual model."
            if is_hi_to_en else
            "Multilingual models are unavailable. Results use lexical overlap only and may miss paraphrases."
        )
        self.last_query_seconds = time.perf_counter() - started
        return hits

    def diagnostics(self) -> dict[str, Any]:
        return {"mode": self._mode, "active_model": self._active.model_name if self._active else None,
                "crosslingual_model_enabled": self._active is not None,
                "enable_secondary_model": self.enable_secondary_model,
                "failed_models": sorted(self._failed_models), "model_errors": list(self._errors),
                "last_fallback_note": self._last_fallback_note,
                "last_query_seconds": round(self.last_query_seconds, 4),
                "primary": self.primary.diagnostics(),
                "secondary": self.secondary.diagnostics() if self.secondary else None,
                "important_limit": "Lexical fallback is not a replacement for cross-lingual embeddings."}
