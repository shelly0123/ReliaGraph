"""Domain-general extraction orchestrator with explicit fallbacks and evidence gating.

Primary model extractors are opt-in and lazy. If an optional model or parser fails,
ReliaGraph falls back to a conservative, language-limited generic pattern extractor.
That fallback favours precision over recall and does not claim universal extraction.
All candidates stay candidates until separately reviewed/calibrated.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Any, Sequence, Mapping
import re
import time

from models import Chunk, EntityCandidate, RelationCandidate, chunk_from
from core.domain_agnostic_extraction import (
    ExtractionRun,
    extract_domain_agnostically,
    norm_text,
    text_contains_span,
    sentence_spans,
)
from core.phase2.injection_guard import scan_and_sanitize
from core.selective_quality import adjudicate_candidates


# Generic predicate cues rather than domain terms. These are intentionally limited:
# an unknown relation is not invented by the fallback; it is left for a model/reviewer.
# Longer/negative predicates are ordered first to preserve meaning.
_GENERIC_CUES: tuple[tuple[str, str], ...] = (
    ("does not depend on", "not_depends_on"),
    ("does not require", "not_requires"),
    ("does not contain", "not_contains"),
    ("does not support", "not_supports"),
    ("is not part of", "not_part_of"),
    ("is not associated with", "not_associated_with"),
    ("is associated with", "associated_with"),
    ("is related to", "related_to"),
    ("is connected to", "connected_to"),
    ("is linked to", "linked_to"),
    ("is based on", "based_on"),
    ("is derived from", "derived_from"),
    ("is composed of", "composed_of"),
    ("consists of", "consists_of"),
    ("part of", "part_of"),
    ("belongs to", "belongs_to"),
    ("depends on", "depends_on"),
    ("leads to", "leads_to"),
    ("results in", "results_in"),
    ("causes", "causes"),
    ("caused by", "caused_by"),
    ("prevents", "prevents"),
    ("reduces", "reduces"),
    ("increases", "increases"),
    ("supports", "supports"),
    ("offers", "offers"),
    ("has", "has"),
    ("follows", "follows"),
    ("describes", "describes"),
    ("defines", "defines"),
    ("represents", "represents"),
    ("lists", "lists"),
    ("covers", "covers"),
    ("manages", "manages"),
    ("provides", "provides"),
    ("requires", "requires"),
    ("contains", "contains"),
    ("includes", "includes"),
    ("uses", "uses"),
    ("stores", "stores"),
    ("measures", "measures"),
    ("protects", "protects"),
    ("absorbs", "absorbs"),
    ("produces", "produces"),
    ("creates", "creates"),
    ("develops", "develops"),
    ("created by", "created_by"),
    ("developed by", "developed_by"),
    ("published by", "published_by"),
    ("located in", "located_in"),
    ("enables", "enables"),
    ("affects", "affects"),
    ("replaces", "replaces"),
    ("contradicts", "contradicts"),
    ("converts", "converts"),
    ("identifies", "identifies"),
    ("stores", "stores"),
    ("is a", "is_a"),
    ("is an", "is_a"),
    ("are a", "is_a"),
    ("are an", "is_a"),
)

_ARTICLE_RE = re.compile(r"^(?:(?:the|a|an|this|these|those|each|every|some|many|several|one)\s+)+", re.I)
_LEADING_CONNECTIVE_RE = re.compile(r"^(?:(?:and|but|while|whereas|because|therefore|however|also)\s+)+", re.I)
_TRAILING_PUNCT_RE = re.compile(r"[\s,;:.!?]+$")


def _cue_variants(cue: str) -> set[str]:
    """Return modest English inflection variants for a single-word predicate cue.

    This is language-level morphology, not a domain vocabulary. It deliberately does
    not infer irregular verb forms or generate variants for multiword predicates.
    """
    cue = cue.strip().casefold()
    if not cue or " " in cue or cue in {"is", "as"}:
        return {cue}
    # Common irregular English predicates are language-level morphology, not
    # corpus-specific relation rules. Keeping these explicit avoids generating
    # nonsense forms such as "haed" for the generic predicate "has".
    irregular = {
        "has": {"has", "have", "had"},
        "does": {"does", "do", "did"},
        "is": {"is", "are", "was", "were", "be", "been", "being"},
        "are": {"are", "is", "was", "were", "be", "been", "being"},
    }
    if cue in irregular:
        return irregular[cue]
    if not cue.endswith("s"):
        return {cue}
    if cue.endswith("ies") and len(cue) > 3:
        base = cue[:-3] + "y"  # identifies -> identify
    else:
        base = cue[:-1]        # provides -> provide; supports -> support
    variants = {cue, base}
    if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        variants.add(base[:-1] + "ied")
        variants.add(base[:-1] + "ies")
        variants.add(base + "ing")
    elif base.endswith("e"):
        variants.add(base + "d")
        variants.add(base[:-1] + "ing")
        variants.add(base + "s")
    else:
        variants.add(base + "ed")
        variants.add(base + "ing")
        variants.add(base + "s")
    return {v for v in variants if v}


def _clean_phrase(value: str, *, side: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n,;:.!?()[]{}\"'")
    value = _LEADING_CONNECTIVE_RE.sub("", value)
    if side == "left":
        value = _ARTICLE_RE.sub("", value)
        # If the sentence begins with a weak introductory clause, use only the final
        # simple clause before the predicate. This is a conservative heuristic.
        if ", " in value and len(value.split(", ")) > 1:
            value = value.rsplit(", ", 1)[-1].strip()
        value = _ARTICLE_RE.sub("", value)
        # In copular constructions such as "The research centre is part of ...",
        # "is" belongs to the predicate, not to the subject node.
        value = re.sub(r"\s+(?:is|are|was|were|has|have|had)$", "", value, flags=re.I).strip()
    else:
        value = _ARTICLE_RE.sub("", value)
        value = re.split(r"\s+(?:because|although|whereas|however|but|and then)\s+", value, maxsplit=1, flags=re.I)[0]
        value = _TRAILING_PUNCT_RE.sub("", value)
        # Trim common trailing adjuncts (e.g. recipient/time/location) but preserve
        # prepositions that frequently belong inside a noun phrase such as "policy on
        # plagiarism" or "support for research". This is a conservative heuristic,
        # not a full syntactic parse.
        value = re.split(r"\s+(?:to|during|within|inside|outside|under|over|at|via|through)\s+", value, maxsplit=1, flags=re.I)[0].strip()
    # Avoid extremely long left/right phrases; a low-confidence fallback should abstain.
    if len(value) > 120:
        value = value[:120].rsplit(" ", 1)[0].strip()
    return value


class GenericPatternExtractor:
    """Model-free English S-P-O fallback that uses no corpus-specific entity terms.

    It detects a limited set of common relation cues. Its scores are ranking signals,
    not probabilities. It is purposely low recall and emits `review_required` candidates.
    """
    name = "generic_pattern_fallback"

    def extract_chunk(self, chunk: Chunk) -> tuple[list[EntityCandidate], list[RelationCandidate]]:
        entities: dict[tuple[str, str], EntityCandidate] = {}
        relations: dict[tuple[str, str, str, str], RelationCandidate] = {}
        for start, end, sentence in sentence_spans(chunk.text):
            if not sentence or len(sentence.split()) < 4:
                continue
            # Suppress relations mentioned only as examples or explicitly denied.
            # This is a language-level safeguard, not a domain-specific word list.
            meta_negation = re.search(
                r"\b(?:no (?:stated |explicit |direct )?(?:relation|relationship|link|connection)|"
                r"without (?:claiming|stating|asserting|implying|establishing)|"
                r"does not (?:state|claim|assert|establish|imply)|"
                r"did not (?:state|claim|assert|establish|imply)|"
                r"cannot conclude|no evidence that|not evidence that|"
                r"mentions? .*? but (?:does not|doesn't|no ) )\b",
                sentence, flags=re.I,
            )
            if meta_negation:
                continue
            matches: list[tuple[int, int, str, str]] = []
            for cue, label in _GENERIC_CUES:
                variants = _cue_variants(cue)
                patterns = sorted(variants, key=len, reverse=True)
                pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(v) for v in patterns) + r")(?!\w)", re.I)
                for m in pattern.finditer(sentence):
                    matches.append((m.start(), m.end(), cue, label))
            if not matches:
                continue
            # Use the earliest cue only in a sentence to avoid accidentally making
            # long-distance claims from conjunction-heavy prose. Multiple claims can be
            # obtained by chunking at sentence boundaries; this fallback never guesses.
            matches.sort(key=lambda m: (m[0], -(m[1] - m[0])))
            cue_start, cue_end, cue, relation = matches[0]
            subject = _clean_phrase(sentence[:cue_start], side="left")
            object_ = _clean_phrase(sentence[cue_end:], side="right")
            if not subject or not object_ or len(subject) < 2 or len(object_) < 2:
                continue
            if norm_text(subject) == norm_text(object_):
                continue
            # Basic safety against cue words being swallowed into endpoint strings.
            if not text_contains_span(sentence, subject) or not text_contains_span(sentence, object_):
                continue
            evidence = sentence.strip()
            ent_sub_key = (norm_text(subject), chunk.chunk_id)
            ent_obj_key = (norm_text(object_), chunk.chunk_id)
            entities.setdefault(ent_sub_key, EntityCandidate(
                label=subject, entity_type="concept", chunk_id=chunk.chunk_id, confidence=0.45,
                attributes={"extractor": self.name, "score_kind": "heuristic_rank_not_probability", "needs_review": True}))
            entities.setdefault(ent_obj_key, EntityCandidate(
                label=object_, entity_type="concept", chunk_id=chunk.chunk_id, confidence=0.45,
                attributes={"extractor": self.name, "score_kind": "heuristic_rank_not_probability", "needs_review": True}))
            key = (norm_text(subject), relation, norm_text(object_), chunk.chunk_id)
            relations.setdefault(key, RelationCandidate(
                subject=subject, relation_type=relation, object=object_, chunk_id=chunk.chunk_id,
                evidence_text=evidence, confidence=0.42,
                attributes={"extractor": self.name, "score_kind": "heuristic_rank_not_probability",
                            "needs_review": True, "cue_text": cue, "language_scope": "English heuristic fallback"}))
        return list(entities.values()), list(relations.values())


@dataclass
class RobustExtractionResult:
    entities: list[EntityCandidate]
    relations: list[RelationCandidate]
    safe_chunks: list[Chunk]
    candidate_decisions: list[dict[str, Any]]
    sanitization_reports: list[dict[str, Any]]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": [asdict(x) for x in self.entities],
            "relations": [asdict(x) for x in self.relations],
            "safe_chunks": [asdict(x) for x in self.safe_chunks],
            "candidate_decisions": self.candidate_decisions,
            "sanitization_reports": self.sanitization_reports,
            "diagnostics": self.diagnostics,
        }


def extract_with_fallback(
    chunks: Sequence[Chunk | dict[str, Any]],
    *,
    enable_gliner: bool = False,
    enable_spacy: bool = False,
    max_chunks: int | None = None,
    require_independent_support: bool = True,
    calibration_model: Mapping[str, Any] | None = None,
) -> RobustExtractionResult:
    """Run opted-in local models when available, then deterministic generic fallback.

    Security properties:
    - document text is scanned before model-facing use;
    - fully quarantined chunks are never sent to extractors;
    - any unavailable optional dependency/model is recorded and fallback proceeds;
    - no candidate is labelled a verified fact.

    Model downloads are not triggered unless the caller explicitly enables that adapter.
    """
    started = time.perf_counter()
    normalized = [chunk_from(c) for c in chunks]
    if max_chunks is not None:
        normalized = normalized[:max(0, int(max_chunks))]

    safe_chunks: list[Chunk] = []
    sanitization_reports: list[dict[str, Any]] = []
    quarantined_count = 0
    for chunk in normalized:
        sanit = scan_and_sanitize(chunk.text)
        sanitization_reports.append({
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "original_sha256": sanit.original_sha256,
            "removed_segments": sanit.removed_segments,
            "finding_count": len(sanit.findings),
            "categories": sorted({x.category for x in sanit.findings}),
            "quarantined": sanit.quarantined,
        })
        if sanit.quarantined or not sanit.safe_text.strip():
            quarantined_count += 1
            continue
        safe_metadata = dict(chunk.metadata)
        safe_metadata["original_sha256"] = sanit.original_sha256
        safe_metadata["sanitized_for_model"] = sanit.safe_text != chunk.text
        safe_metadata["injection_flag"] = bool(sanit.findings)
        safe_metadata["injection_finding_count"] = len(sanit.findings)
        safe_chunks.append(replace(chunk, text=sanit.safe_text, metadata=safe_metadata))

    entities: list[EntityCandidate] = []
    relations: list[RelationCandidate] = []
    adapter_log: list[dict[str, Any]] = []

    # Each adapter is isolated. One model failing does not prevent use of another.
    if safe_chunks and enable_gliner:
        t0 = time.perf_counter()
        try:
            output = extract_domain_agnostically(safe_chunks, use_spacy=False, use_gliner=True, max_chunks=max_chunks)
            entities.extend(output.entities); relations.extend(output.relations)
            adapter_log.append({"adapter": "gliner_relex", "status": "available", "seconds": round(time.perf_counter()-t0, 4),
                                "entities": len(output.entities), "relations": len(output.relations), "diagnostics": output.diagnostics})
        except Exception as exc:
            adapter_log.append({"adapter": "gliner_relex", "status": "fallback", "error_type": type(exc).__name__,
                                "message": str(exc)[:240], "seconds": round(time.perf_counter()-t0, 4)})

    if safe_chunks and enable_spacy:
        t0 = time.perf_counter()
        try:
            output = extract_domain_agnostically(safe_chunks, use_spacy=True, use_gliner=False, max_chunks=max_chunks)
            entities.extend(output.entities); relations.extend(output.relations)
            adapter_log.append({"adapter": "spacy_dependency", "status": "available", "seconds": round(time.perf_counter()-t0, 4),
                                "entities": len(output.entities), "relations": len(output.relations), "diagnostics": output.diagnostics})
        except Exception as exc:
            adapter_log.append({"adapter": "spacy_dependency", "status": "fallback", "error_type": type(exc).__name__,
                                "message": str(exc)[:240], "seconds": round(time.perf_counter()-t0, 4)})

    fallback = GenericPatternExtractor()
    fb_entities = 0
    fb_relations = 0
    t0 = time.perf_counter()
    for chunk in safe_chunks:
        e, r = fallback.extract_chunk(chunk)
        entities.extend(e); relations.extend(r)
        fb_entities += len(e); fb_relations += len(r)
    adapter_log.append({"adapter": fallback.name, "status": "always_available", "seconds": round(time.perf_counter()-t0, 4),
                        "entities": fb_entities, "relations": fb_relations})

    decisions = adjudicate_candidates(relations, safe_chunks,
                                      require_independent_support=require_independent_support,
                                      calibration_model=calibration_model)
    elapsed = time.perf_counter() - started
    return RobustExtractionResult(
        entities=entities,
        relations=relations,
        safe_chunks=safe_chunks,
        candidate_decisions=[x.to_dict() for x in decisions],
        sanitization_reports=sanitization_reports,
        diagnostics={
            "mode": "hybrid_with_fallback" if (enable_gliner or enable_spacy) else "deterministic_fallback_only",
            "input_chunks": len(normalized), "safe_chunks": len(safe_chunks),
            "quarantined_chunks": quarantined_count,
            "entity_candidates": len(entities), "relation_candidates": len(relations),
            "elapsed_seconds": round(elapsed, 4), "adapters": adapter_log,
            "candidate_decision_counts": _count_decisions(decisions),
            "accuracy_note": "Fallback is conservative and English-cue-limited. Model scores are not truth probabilities. Use a labelled held-out set to measure per-domain precision/recall and review coverage.",
        },
    )


def _count_decisions(decisions: Sequence[Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in decisions:
        result[item.decision] = result.get(item.decision, 0) + 1
    return result
