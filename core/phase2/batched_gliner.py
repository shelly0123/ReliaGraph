"""Batched domain-agnostic GLiNER-ReLEx adapter.

Uses broad generic entity/relation descriptions and batches sentence inference to avoid
one model call per document. Scores are ranking signals, not calibrated probabilities.
The model weights are downloaded only if this adapter is explicitly selected.
"""
from __future__ import annotations
import hashlib, json, time
from typing import Any, Mapping, Sequence
from core.extraction_cache import SQLiteJSONCache
from core.domain_agnostic_extraction import (
    GENERIC_ENTITY_LABELS, GENERIC_RELATION_LABELS, sentence_spans, text_contains_span,
)
from models import Chunk, EntityCandidate, RelationCandidate, chunk_from

class BatchedGLiNERRelexExtractor:
    MODEL_NAME = "knowledgator/gliner-relex-multi-v1.0"
    name = "gliner_relex_batched_zero_shot"

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu", batch_size: int = 24,
                 entity_threshold: float = 0.4, relation_threshold: float = 0.7,
                 cache_path: str = "data/cache/gliner_batched.sqlite3"):
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.entity_threshold = float(entity_threshold)
        self.relation_threshold = float(relation_threshold)
        self.cache = SQLiteJSONCache(cache_path)
        self._model = None
        self.load_seconds = None
        self.inference_seconds = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.batch_calls = 0

    def _load(self):
        if self._model is not None: return
        started = time.perf_counter()
        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise RuntimeError("Install optional extraction model dependencies from requirements-optional-models.txt") from exc
        self._model = GLiNER.from_pretrained(self.model_name)
        self.load_seconds = time.perf_counter() - started

    def extract_chunks(self, chunks: Sequence[Chunk | Mapping[str, Any]]) -> tuple[list[EntityCandidate], list[RelationCandidate]]:
        norm_chunks = [chunk_from(c) for c in chunks]
        sentence_rows = []
        for chunk in norm_chunks:
            for start, end, sentence in sentence_spans(chunk.text):
                if sentence.strip():
                    cache_payload = {"model": self.model_name, "sentence": sentence,
                                     "entities": GENERIC_ENTITY_LABELS, "relations": GENERIC_RELATION_LABELS,
                                     "entity_threshold": self.entity_threshold, "relation_threshold": self.relation_threshold}
                    key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    sentence_rows.append({"chunk": chunk, "start": start, "end": end, "sentence": sentence, "cache_key": key})
        if not sentence_rows: return [], []
        # Gather cache misses, infer in batches, and cache each sentence independently.
        work_texts, work_row_indices = [], []
        cached_by_row: dict[int, dict[str, Any]] = {}
        for i, row in enumerate(sentence_rows):
            cached = self.cache.get(row["cache_key"])
            if cached is not None:
                self.cache_hits += 1
                cached_by_row[i] = cached
            else:
                self.cache_misses += 1
                work_texts.append(row["sentence"]); work_row_indices.append(i)
        if work_texts:
            self._load()
            for start in range(0, len(work_texts), self.batch_size):
                batch_texts = work_texts[start:start + self.batch_size]
                batch_ids = work_row_indices[start:start + self.batch_size]
                t0 = time.perf_counter()
                try:
                    entity_batches, relation_batches = self._model.inference(
                        texts=batch_texts, labels=GENERIC_ENTITY_LABELS, relations=GENERIC_RELATION_LABELS,
                        threshold=self.entity_threshold, relation_threshold=self.relation_threshold,
                        return_relations=True, flat_ner=False)
                except TypeError:
                    entity_batches, relation_batches = self._model.inference(
                        texts=batch_texts, labels=GENERIC_ENTITY_LABELS, relations=GENERIC_RELATION_LABELS,
                        threshold=self.entity_threshold, return_relations=True, flat_ner=False)
                self.inference_seconds += time.perf_counter() - t0
                self.batch_calls += 1
                for local_i, global_i in enumerate(batch_ids):
                    ents_raw = (entity_batches[local_i] if local_i < len(entity_batches) else [])
                    rels_raw = (relation_batches[local_i] if local_i < len(relation_batches) else [])
                    cached_value = {
                        "entities": [{"text": str(e.get("text", "")), "label": str(e.get("label", "entity")), "score": float(e.get("score", 0.0))} for e in ents_raw],
                        "relations": [{"head": {"text": str((r.get("head") or {}).get("text", "")), "type": str((r.get("head") or {}).get("type", "entity"))},
                                       "tail": {"text": str((r.get("tail") or {}).get("text", "")), "type": str((r.get("tail") or {}).get("type", "entity"))},
                                       "relation": str(r.get("relation", "")), "score": float(r.get("score", 0.0))} for r in rels_raw]
                    }
                    cached_by_row[global_i] = cached_value
                    self.cache.put(sentence_rows[global_i]["cache_key"], cached_value)
        entities, relations = {}, {}
        for i, row in enumerate(sentence_rows):
            chunk, sentence = row["chunk"], row["sentence"]
            cached = cached_by_row.get(i, {"entities": [], "relations": []})
            for e in cached["entities"]:
                label = str(e.get("text", "")).strip()
                if not label or not text_contains_span(sentence, label): continue
                key = (chunk.chunk_id, label.casefold())
                entities.setdefault(key, EntityCandidate(label=label, entity_type=str(e.get("label", "entity")),
                    chunk_id=chunk.chunk_id, confidence=float(e.get("score", 0.0)),
                    attributes={"extractor": self.name, "score_kind": "model_rank_not_calibrated_probability"}))
            for r in cached["relations"]:
                head = str((r.get("head") or {}).get("text", "")).strip()
                tail = str((r.get("tail") or {}).get("text", "")).strip()
                rel = str(r.get("relation", "")).strip()
                if not head or not tail or not rel or head.casefold() == tail.casefold(): continue
                if not text_contains_span(sentence, head) or not text_contains_span(sentence, tail): continue
                key = (chunk.chunk_id, head.casefold(), rel.casefold(), tail.casefold())
                relations.setdefault(key, RelationCandidate(subject=head, relation_type=rel, object=tail,
                    chunk_id=chunk.chunk_id, evidence_text=sentence, confidence=float(r.get("score", 0.0)),
                    attributes={"extractor": self.name, "score_kind": "model_rank_not_calibrated_probability",
                                "subject_type": str((r.get("head") or {}).get("type", "entity")),
                                "object_type": str((r.get("tail") or {}).get("type", "entity"))}))
        return list(entities.values()), list(relations.values())

    def diagnostics(self):
        return {"model_name": self.model_name, "device": self.device, "batch_size": self.batch_size,
                "load_seconds": self.load_seconds, "inference_seconds": round(self.inference_seconds, 4),
                "cache_hits": self.cache_hits, "cache_misses": self.cache_misses, "batch_calls": self.batch_calls}
