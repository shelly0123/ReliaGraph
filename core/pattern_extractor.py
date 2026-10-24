"""Transparent rule-based candidate extraction for a controlled prototype.

This is intentionally not presented as a universal relation-extraction model.
It only creates candidates when known terms occur in a sentence and a supported
relation cue appears between adjacent known-term mentions. Every candidate keeps
its source chunk and quote so later auditing can check provenance.
"""
from __future__ import annotations
import re
from collections.abc import Iterable, Sequence
from models import Chunk, EntityCandidate, RelationCandidate, chunk_from

# More specific / negated cues must precede their positive forms.
RELATION_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("does_not_require", ("does not require", "doesn't require", "is not required for", "is not required")),
    ("does_not_allow", ("does not allow", "doesn't allow", "does not permit")),
    ("prohibits", ("prohibits", "forbids", "is forbidden", "is prohibited", "must not")),
    ("requires", ("requires", "require", "must complete", "must have", "needs", "is required for")),
    ("allows", ("allows", "allow", "permits", "permit", "is allowed")),
    ("contradicts", ("contradicts", "conflicts with", "is inconsistent with")),
    ("increases", ("increases", "increased", "raises", "elevates")),
    ("decreases", ("decreases", "decreased", "reduces", "reduced", "lowers", "mitigates")),
    ("protects", ("protects", "protect", "safeguards")),
    ("prevents", ("prevents", "prevent", "blocks")),
    ("causes", ("causes", "caused", "leads to", "results in")),
    ("depends_on", ("depends on", "relies on")),
    ("includes", ("includes", "include", "contains", "contain", "lists", "list", "covers", "cover", "spans")),
    ("offers", ("offers", "offer", "provides", "provide")),
    ("has", ("has", "have", "maintains", "maintain")),
    ("follows", ("follows", "follow")),
    ("uses", ("uses", "use", "employs", "employ")),
    ("enables", ("enables", "enable", "supports", "support", "facilitates", "promotes", "promote")),
    ("has_deadline", ("deadline is", "deadline: ", "due date is", "due by")),
    ("has_threshold", ("threshold is", "minimum is", "requires a minimum of")),
    ("is_a", (" is a ", " is an ", " are a ", " are an ")),
)


def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term.strip())
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


def _find_non_overlapping_terms(text: str, terms: Sequence[str]) -> list[tuple[int, int, str]]:
    """Find phrases, preferring longer matches to avoid 'access' inside 'access control'."""
    matches: list[tuple[int, int, str]] = []
    seen = {t.strip() for t in terms if t and t.strip()}
    for term in sorted(seen, key=lambda v: (-len(v), v.casefold())):
        for match in _term_pattern(term).finditer(text):
            if any(match.start() < end and match.end() > start for start, end, _ in matches):
                continue
            matches.append((match.start(), match.end(), term.strip()))
    return sorted(matches, key=lambda item: (item[0], item[1]))


def _relation_in_gap(gap: str, subject: str = "") -> str | None:
    """Map a short verb phrase to a normalized relation label.

    Context-sensitive rules (for example, 'deadline is 20 March') are narrow by
    design; unsupported sentences are left unconnected instead of guessed.
    """
    lowered = f" {gap.casefold()} "
    subj = subject.casefold()
    # Special case for functional values; used for conservative version diffs.
    if re.search(r"\bis\b|:\s*$", gap, re.IGNORECASE) and any(
        word in subj for word in ("deadline", "due date", "effective date")
    ):
        return "has_deadline"
    if re.search(r"\bis\b|:\s*$", gap, re.IGNORECASE) and any(
        word in subj for word in ("threshold", "minimum score", "minimum attendance")
    ):
        return "has_threshold"
    for relation_type, cues in RELATION_CUES:
        for cue in cues:
            if re.search(rf"(?<!\w){re.escape(cue.strip())}(?!\w)", lowered, re.IGNORECASE):
                return relation_type
    return None


def extract_candidates(
    chunks: Sequence[Chunk | dict], known_terms: Iterable[str]
) -> tuple[list[EntityCandidate], list[RelationCandidate]]:
    """Extract entity mentions and relation candidates from known-term pairs.

    Relation extraction is intentionally limited to adjacent recognized terms in
    a sentence. It is predictable and auditable, but users must review candidates.
    """
    normalized_chunks = [chunk_from(c) for c in chunks]
    terms = [str(t).strip() for t in known_terms if str(t).strip()]
    entities: dict[tuple[str, str], EntityCandidate] = {}
    relations: dict[tuple[str, str, str, str], RelationCandidate] = {}

    for chunk in normalized_chunks:
        # Keep each source quote whole for traceability; semicolon/newline boundaries help.
        sentences = re.split(r"(?<=[.!?])\s+|\n+", chunk.text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            mentions = _find_non_overlapping_terms(sentence, terms)
            for _, _, label in mentions:
                entities.setdefault(
                    (label.casefold(), chunk.chunk_id),
                    EntityCandidate(label=label, entity_type="concept", chunk_id=chunk.chunk_id,
                                    confidence=0.95, attributes={"extractor": "known_term_match"}),
                )
            # Only adjacent terms are linked; this limits long-distance false links.
            for i in range(len(mentions) - 1):
                _, sub_end, subject = mentions[i]
                obj_start, _, object_label = mentions[i + 1]
                if subject.casefold() == object_label.casefold():
                    continue
                gap = sentence[sub_end:obj_start]
                relation_type = _relation_in_gap(gap, subject)
                if relation_type is None:
                    continue
                key = (subject.casefold(), relation_type, object_label.casefold(), chunk.chunk_id)
                relations.setdefault(
                    key,
                    RelationCandidate(
                        subject=subject,
                        relation_type=relation_type,
                        object=object_label,
                        chunk_id=chunk.chunk_id,
                        evidence_text=sentence,
                        confidence=0.75,
                        attributes={"extractor": "adjacent_known_term_pattern", "cue_text": gap.strip()},
                    ),
                )
    return list(entities.values()), list(relations.values())
