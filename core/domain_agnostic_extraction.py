"""Domain-agnostic relation extraction adapters for ReliaGraph.

The module does not use MUJ-specific terms or relation patterns. It provides:
- a generic spaCy dependency/OpenIE-like extractor (optional model package),
- an optional zero-shot GLiNER-ReLEx adapter,
- dynamically discovered predicate labels from the parsed text,
- evidence span validation before candidates enter the graph.

Scores returned by extractors are ranking signals, NOT calibrated probabilities of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Any, Protocol, Sequence
import re
import hashlib
import json

from core.extraction_cache import SQLiteJSONCache
from models import Chunk, EntityCandidate, RelationCandidate, chunk_from

# Generic relation labels only. These are not tailored to a specific corpus. The
# spaCy adapter adds predicate labels discovered from the current text at runtime.
GENERIC_RELATION_LABELS = [
    "related to", "associated with", "part of", "type of", "instance of",
    "located in", "works for", "member of", "created by", "created",
    "founded by", "author of", "published by", "requires", "depends on",
    "causes", "contributes to", "leads to", "prevents", "produces", "uses",
    "contains", "includes", "supports", "provides", "measured by", "measures",
    "stores", "absorbs", "protects", "regulates", "converts", "connects", "organizes",
    "maintains", "records", "analyzes", "controls", "affects", "flows through",
    "occurs before", "occurs after", "replaced by", "differs from",
    "contradicts", "derived from", "owned by", "developed by", "connected to",
]

GENERIC_ENTITY_LABELS = {
    "person": "a human individual or named person",
    "organization": "an organization, institution, company, team or group",
    "location": "a geographical, physical or virtual location",
    "date": "a date, time, duration or temporal expression",
    "event": "an event, activity, occurrence or process",
    "work": "a document, book, work, product or named artifact",
    "concept": "a concept, method, property, process, object or abstract idea",
    "quantity": "a number, measurement, amount, threshold or unit",
    "other": "any other entity involved in a meaningful relation",
}


def norm_text(value: str) -> str:
    value = str(value or "").casefold().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s%.-]", "", value, flags=re.UNICODE)
    return value.strip()


def text_contains_span(text: str, span: str) -> bool:
    """Whitespace/case-insensitive source-span check; deliberately conservative."""
    t = re.sub(r"\s+", " ", str(text or "")).casefold()
    s = re.sub(r"\s+", " ", str(span or "")).casefold().strip()
    return bool(s) and s in t


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """Simple sentence segmentation fallback. Offsets refer to the original text."""
    result = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)|[^\n]+", text or ""):
        value = match.group(0).strip()
        if value:
            start = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
            result.append((start, start + len(value), value))
    return result


class RelationExtractor(Protocol):
    name: str
    def extract_chunk(self, chunk: Chunk, relation_hints: Sequence[str] | None = None) -> tuple[list[EntityCandidate], list[RelationCandidate]]: ...


@dataclass
class ExtractionRun:
    entities: list[EntityCandidate]
    relations: list[RelationCandidate]
    diagnostics: dict[str, Any]


class SpaCyDependencyOpenIE:
    """Generic syntax-driven subject/predicate/object extraction.

    Requires `spacy` and its English dependency model. It uses grammatical
    dependencies, not a manually curated domain vocabulary. It only emits a
    triple when subject, predicate and object can be identified in a sentence.
    """
    name = "spacy_dependency_openie"
    SUBJECT_DEPS = {"nsubj", "nsubjpass", "nsubj:pass", "csubj"}
    OBJECT_DEPS = {"dobj", "obj", "attr", "oprd", "dative", "acomp", "pobj"}

    def __init__(self, model_name: str = "en_core_web_sm", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._nlp = None
        self._relation_hint_cache: dict[str, list[str]] = {}
        self.load_seconds: float | None = None
        self.last_seconds: float = 0.0

    def _load(self) -> None:
        if self._nlp is not None:
            return
        import time
        start = time.perf_counter()
        try:
            import spacy
            self._nlp = spacy.load(self.model_name)
        except Exception as exc:
            raise RuntimeError(
                f"spaCy parser model '{self.model_name}' is unavailable. Install spaCy and run "
                f"`python -m spacy download {self.model_name}`. This adapter is optional; the "
                "GLiNER-ReLEx adapter or deterministic review queue can be used instead."
            ) from exc
        self.load_seconds = time.perf_counter() - start

    @staticmethod
    def _phrase(token: Any, noun_chunks: Sequence[Any]) -> str:
        for span in noun_chunks:
            if span.start <= token.i < span.end:
                return span.text.strip()
        # Fallback to a bounded subtree, avoiding entire sentence subtrees.
        toks = [t for t in token.subtree if t.pos_ not in {"PUNCT", "SPACE"}]
        if not toks:
            return token.text.strip()
        start, end = min(t.i for t in toks), max(t.i for t in toks) + 1
        return token.doc[start:end].text.strip()

    @staticmethod
    def _predicate(token: Any, obj_token: Any) -> str:
        """Build a predicate from grammatical words around a predicate head."""
        parts = [token.lemma_.casefold() or token.text.casefold()]
        # Particles and prepositions carrying this object are part of the relation.
        prep_parts = []
        for child in token.children:
            if child.dep_ in {"prt", "prep", "agent"}:
                if child.dep_ == "prep":
                    has_obj = any(gc == obj_token or gc.i == obj_token.i for gc in child.children)
                    if not has_obj and not any(gc.dep_ == "pobj" and gc.i == obj_token.i for gc in child.children):
                        continue
                prep_parts.append(child.text.casefold())
        parts.extend(prep_parts)
        negated = any(c.dep_ == "neg" for c in token.children)
        label = " ".join(dict.fromkeys(parts)).strip()
        return f"not {label}" if negated else label

    @staticmethod
    def _sentence_for_token(token: Any) -> str:
        try:
            return token.sent.text.strip()
        except Exception:
            return token.doc.text.strip()

    def extract_chunk(self, chunk: Chunk, relation_hints: Sequence[str] | None = None) -> tuple[list[EntityCandidate], list[RelationCandidate]]:
        import time
        self._load()
        start_time = time.perf_counter()
        doc = self._nlp(chunk.text)
        try:
            noun_chunks = list(doc.noun_chunks)
        except Exception:
            noun_chunks = []
        entities: dict[str, EntityCandidate] = {}
        relations: dict[tuple[str, str, str], RelationCandidate] = {}

        for ent in getattr(doc, "ents", []):
            label = ent.text.strip()
            if label:
                entities.setdefault(norm_text(label), EntityCandidate(
                    label=label, entity_type=(ent.label_ or "entity").casefold(),
                    chunk_id=chunk.chunk_id, confidence=0.7,
                    attributes={"extractor": self.name, "span_start": ent.start_char, "span_end": ent.end_char},
                ))

        for predicate in doc:
            # Main verbs and copular predicate heads (e.g., "X is a method").
            is_predicate = predicate.pos_ in {"VERB", "AUX"} or any(c.dep_ == "cop" for c in predicate.children)
            if not is_predicate or predicate.dep_ == "aux":
                continue
            children = list(predicate.children)
            subjects = [c for c in children if c.dep_ in self.SUBJECT_DEPS]
            objects = [c for c in children if c.dep_ in self.OBJECT_DEPS and c.dep_ != "pobj"]
            # Prepositional objects are linked through a child preposition.
            for prep in [c for c in children if c.dep_ in {"prep", "agent"}]:
                objects.extend([gc for gc in prep.children if gc.dep_ == "pobj"])
            # Copular constructions may use the predicate head as the object.
            if not subjects and any(c.dep_ == "cop" for c in children):
                subjects = [c for c in children if c.dep_ in self.SUBJECT_DEPS]
            if not subjects or not objects:
                continue
            for subject_token in subjects:
                subject = self._phrase(subject_token, noun_chunks)
                for object_token in objects:
                    object_ = self._phrase(object_token, noun_chunks)
                    if not subject or not object_ or norm_text(subject) == norm_text(object_):
                        continue
                    sentence = self._sentence_for_token(predicate)
                    rel_type = self._predicate(predicate, object_token)
                    if not rel_type:
                        continue
                    # Keep every span strictly grounded in the original sentence.
                    if not text_contains_span(sentence, subject) or not text_contains_span(sentence, object_):
                        continue
                    key = (norm_text(subject), norm_text(rel_type), norm_text(object_))
                    relations.setdefault(key, RelationCandidate(
                        subject=subject, relation_type=rel_type, object=object_,
                        chunk_id=chunk.chunk_id, evidence_text=sentence, confidence=0.58,
                        attributes={"extractor": self.name, "score_kind": "heuristic_rank_not_probability",
                                    "sentence_start": getattr(predicate.sent, "start_char", None),
                                    "sentence_end": getattr(predicate.sent, "end_char", None)},
                    ))
                    entities.setdefault(norm_text(subject), EntityCandidate(
                        label=subject, entity_type="entity", chunk_id=chunk.chunk_id, confidence=0.58,
                        attributes={"extractor": self.name},
                    ))
                    entities.setdefault(norm_text(object_), EntityCandidate(
                        label=object_, entity_type="entity", chunk_id=chunk.chunk_id, confidence=0.58,
                        attributes={"extractor": self.name},
                    ))
        hints = []
        for tok in doc:
            if tok.pos_ in {"VERB", "AUX"} or any(c.dep_ == "cop" for c in tok.children):
                label = tok.lemma_.casefold().strip()
                if label and label not in {"be", "do", "have"}:
                    prep = [c.text.casefold() for c in tok.children if c.dep_ in {"prep", "prt"}]
                    label = " ".join([label, *prep]).strip()
                    hints.append(label)
        self._relation_hint_cache[hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()] = list(dict.fromkeys(hints))
        self.last_seconds = time.perf_counter() - start_time
        return list(entities.values()), list(relations.values())

    def relation_hints(self, text: str) -> list[str]:
        """Expose predicates discovered in text, for dynamic zero-shot label candidates."""
        key = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
        if key in self._relation_hint_cache:
            return list(self._relation_hint_cache[key])
        self._load()
        doc = self._nlp(text)
        hints = []
        for tok in doc:
            if tok.pos_ in {"VERB", "AUX"} or any(c.dep_ == "cop" for c in tok.children):
                label = tok.lemma_.casefold().strip()
                if label and label not in {"be", "do", "have"}:
                    prep = [c.text.casefold() for c in tok.children if c.dep_ in {"prep", "prt"}]
                    label = " ".join([label, *prep]).strip()
                    hints.append(label)
        self._relation_hint_cache[key] = list(dict.fromkeys(hints))
        return list(self._relation_hint_cache[key])

    def diagnostics(self) -> dict[str, Any]:
        return {"adapter": self.name, "model": self.model_name, "load_seconds": self.load_seconds,
                "last_inference_seconds": self.last_seconds}


class GLiNERRelexExtractor:
    """Optional local, zero-shot entity + relation extraction.

    Hugging Face model card: knowledgator/gliner-relex-multi-v1.0 (Apache-2.0).
    It needs no training on the incoming domain. It does require candidate relation
    labels, so the adapter combines a broad generic vocabulary with predicate hints
    discovered from the current text when the spaCy adapter is available.
    """
    name = "gliner_relex_zero_shot"
    MODEL_NAME = "knowledgator/gliner-relex-multi-v1.0"

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu", max_relation_labels: int = 48,
                 entity_threshold: float = 0.4, relation_threshold: float = 0.7,
                 cache_path: str = "data/cache/gliner_relex.sqlite3") -> None:
        self.model_name = model_name
        self.device = device
        self.max_relation_labels = max(16, int(max_relation_labels))
        self.entity_threshold = float(entity_threshold)
        self.relation_threshold = float(relation_threshold)
        self._model = None
        self.cache = SQLiteJSONCache(cache_path)
        self.load_seconds: float | None = None
        self.last_seconds = 0.0
        self.last_model_seconds = 0.0
        self.calls = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def _load(self) -> None:
        if self._model is not None:
            return
        import time
        start = time.perf_counter()
        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise RuntimeError("Install optional model dependencies with `pip install -r requirements-optional-models.txt`.") from exc
        self._model = GLiNER.from_pretrained(self.model_name)
        self.load_seconds = time.perf_counter() - start

    def extract_chunk(self, chunk: Chunk, relation_hints: Sequence[str] | None = None) -> tuple[list[EntityCandidate], list[RelationCandidate]]:
        import time
        self._load()
        start_time = time.perf_counter()
        hints = [re.sub(r"\s+", " ", str(x).casefold()).strip() for x in (relation_hints or []) if str(x).strip()]
        # Reserve roughly half the labels for general relations, even when a long document
        # produces many verb hints. Dynamic hints keep the system adaptable; the generic
        # portion avoids overfitting to just the first sentences in a chunk.
        dynamic_budget = max(1, self.max_relation_labels // 2)
        relation_labels = list(dict.fromkeys(hints[:dynamic_budget] + GENERIC_RELATION_LABELS))[:self.max_relation_labels]
        sentences = sentence_spans(chunk.text)
        texts = [sentence for _, _, sentence in sentences if sentence.strip()]
        if not texts:
            return [], []
        cache_payload = {"model": self.model_name, "text": chunk.text, "relations": relation_labels,
                         "entity_labels": GENERIC_ENTITY_LABELS, "entity_threshold": self.entity_threshold,
                         "relation_threshold": self.relation_threshold}
        cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            raw_entities, raw_relations = cached["entities"], cached["relations"]
            model_seconds = 0.0
        else:
            self.cache_misses += 1
            model_start = time.perf_counter()
            try:
                entity_batches, relation_batches = self._model.inference(
                    texts=texts, labels=GENERIC_ENTITY_LABELS, relations=relation_labels,
                    threshold=self.entity_threshold, relation_threshold=self.relation_threshold,
                    return_relations=True, flat_ner=False,
                )
            except TypeError:
                # Backward-compatible fallback for model versions without relation_threshold.
                entity_batches, relation_batches = self._model.inference(
                    texts=texts, labels=GENERIC_ENTITY_LABELS, relations=relation_labels,
                    threshold=self.entity_threshold, return_relations=True, flat_ner=False,
                )
            model_seconds = time.perf_counter() - model_start
            raw_entities = []
            raw_relations = []
            for i in range(len(texts)):
                raw_entities.append([{"text": str(e.get("text", "")), "label": str(e.get("label", "entity")),
                                      "score": float(e.get("score", 0.0))}
                                     for e in (entity_batches[i] if i < len(entity_batches) else [])])
                raw_relations.append([{"head": {"text": str((r.get("head") or {}).get("text", "")),
                                                "type": str((r.get("head") or {}).get("type", "entity"))},
                                       "tail": {"text": str((r.get("tail") or {}).get("text", "")),
                                                "type": str((r.get("tail") or {}).get("type", "entity"))},
                                       "relation": str(r.get("relation", "")), "score": float(r.get("score", 0.0))}
                                      for r in (relation_batches[i] if i < len(relation_batches) else [])])
            self.cache.put(cache_key, {"entities": raw_entities, "relations": raw_relations})
        self.last_model_seconds = model_seconds
        # Reattach evidence and source chunk IDs only after cache lookup so the cache is
        # reusable for identical text imported from another document/version.
        entities_out: dict[tuple[str, str], EntityCandidate] = {}
        relations_out: dict[tuple[str, str, str, str], RelationCandidate] = {}
        sentence_index = 0
        for _, _, sentence in sentences:
            if not sentence.strip():
                continue
            ents_for_sentence = raw_entities[sentence_index] if sentence_index < len(raw_entities) else []
            rels_for_sentence = raw_relations[sentence_index] if sentence_index < len(raw_relations) else []
            for ent in ents_for_sentence:
                label = str(ent.get("text", "")).strip()
                if not label:
                    continue
                entity_type = str(ent.get("label", "entity")).casefold()
                key = (norm_text(label), entity_type)
                entities_out.setdefault(key, EntityCandidate(
                    label=label, entity_type=entity_type, chunk_id=chunk.chunk_id,
                    confidence=max(0.0, min(1.0, float(ent.get("score", 0.0)))),
                    attributes={"extractor": self.name, "score_kind": "model_score_not_calibrated_probability"},
                ))
            for rel in rels_for_sentence:
                head_obj = rel.get("head") or {}; tail_obj = rel.get("tail") or {}
                head = str(head_obj.get("text", "")).strip(); tail = str(tail_obj.get("text", "")).strip()
                predicate = str(rel.get("relation", "")).strip()
                score = float(rel.get("score", 0.0))
                if not head or not tail or not predicate or norm_text(head) == norm_text(tail):
                    continue
                if not text_contains_span(sentence, head) or not text_contains_span(sentence, tail):
                    continue
                key = (norm_text(head), norm_text(predicate), norm_text(tail), chunk.chunk_id)
                relations_out.setdefault(key, RelationCandidate(
                    subject=head, relation_type=predicate, object=tail, chunk_id=chunk.chunk_id,
                    evidence_text=sentence, confidence=max(0.0, min(1.0, score)),
                    attributes={"extractor": self.name, "score_kind": "model_score_not_calibrated_probability",
                                "subject_type": str(head_obj.get("type", "entity")),
                                "object_type": str(tail_obj.get("type", "entity"))},
                ))
            sentence_index += 1
        self.calls += 1
        self.last_seconds = time.perf_counter() - start_time
        return list(entities_out.values()), list(relations_out.values())

    def diagnostics(self) -> dict[str, Any]:
        return {"adapter": self.name, "model": self.model_name, "device": self.device,
                "load_seconds": self.load_seconds, "last_inference_seconds": self.last_seconds,
                "last_model_inference_seconds": self.last_model_seconds, "calls": self.calls,
                "cache_hits": self.cache_hits, "cache_misses": self.cache_misses, "cache_path": str(self.cache.path)}


def extract_domain_agnostically(
    chunks: Sequence[Chunk | dict],
    use_spacy: bool = True,
    use_gliner: bool = False,
    spacy_model: str = "en_core_web_sm",
    gliner_model: str = GLiNERRelexExtractor.MODEL_NAME,
    max_chunks: int | None = None,
) -> ExtractionRun:
    """Run available general-purpose extractors without domain-specific vocabularies.

    The model adapters are lazy. If an explicitly requested adapter is unavailable,
    the exception is surfaced to the caller; callers must show an honest warning rather
    than silently pretending the model ran.
    """
    normalized = [chunk_from(x) for x in chunks]
    if max_chunks is not None:
        normalized = normalized[:max(0, int(max_chunks))]
    entities: list[EntityCandidate] = []
    relations: list[RelationCandidate] = []
    diagnostics: dict[str, Any] = {"domain_specific_lexicon_used": False, "chunk_count": len(normalized), "adapters": []}
    spacy_adapter = None
    if use_spacy:
        spacy_adapter = SpaCyDependencyOpenIE(spacy_model)
        for chunk in normalized:
            ens, rels = spacy_adapter.extract_chunk(chunk)
            entities.extend(ens); relations.extend(rels)
        diagnostics["adapters"].append(spacy_adapter.diagnostics())
    if use_gliner:
        model_adapter = GLiNERRelexExtractor(gliner_model)
        for chunk in normalized:
            hints = []
            if spacy_adapter is not None:
                try:
                    hints = spacy_adapter.relation_hints(chunk.text)
                except Exception as exc:
                    diagnostics.setdefault("warnings", []).append(f"Predicate-hint discovery unavailable for {chunk.chunk_id}: {type(exc).__name__}")
            ens, rels = model_adapter.extract_chunk(chunk, relation_hints=hints)
            entities.extend(ens); relations.extend(rels)
        diagnostics["adapters"].append(model_adapter.diagnostics())
    diagnostics["entity_candidate_count"] = len(entities)
    diagnostics["relation_candidate_count"] = len(relations)
    diagnostics["notes"] = [
        "Model scores are not calibrated truth probabilities.",
        "No domain-specific term list or MUJ-specific relation rules were used.",
        "Candidates must pass source alignment and selective promotion before being treated as corroborated.",
    ]
    return ExtractionRun(entities=entities, relations=relations, diagnostics=diagnostics)
