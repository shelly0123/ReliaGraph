"""Optional spaCy entity-term suggestion helper.

This module is NOT required by the core graph pipeline. Install spaCy and its
English model only if the team wants more entity suggestions than a curated
term list supplies. Relationship extraction remains rule-based and evidence-
linked, so the optional model does not become a hard runtime dependency.
"""
from __future__ import annotations
from collections.abc import Sequence
from models import Chunk, chunk_from

DEFAULT_LABELS = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "NORP", "FAC"}


def suggest_entity_terms(
    chunks: Sequence[Chunk | dict], model_name: str = "en_core_web_sm",
    allowed_labels: set[str] | None = None,
) -> list[str]:
    """Return entity phrase candidates from spaCy; raises a clear optional-dependency error.

    Review the suggestions before using them as relation-extraction terms; generic
    NER models may miss technical concepts or label domain-specific terms poorly.
    """
    try:
        import spacy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Optional spaCy support is not installed. Run: pip install '.[nlp]' then python -m spacy download en_core_web_sm") from exc
    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(f"spaCy model '{model_name}' is not installed. Run: python -m spacy download {model_name}") from exc
    labels = allowed_labels or DEFAULT_LABELS
    found: dict[str, str] = {}
    for raw_chunk in chunks:
        chunk = chunk_from(raw_chunk)
        doc = nlp(chunk.text)
        for ent in doc.ents:
            label = ent.text.strip()
            if ent.label_ in labels and len(label) >= 2:
                found.setdefault(" ".join(label.casefold().split()), label)
    return sorted(found.values(), key=lambda s: s.casefold())
