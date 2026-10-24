"""Reproducible synthetic test questions with reference answers and gold evidence.

Template-based (no LLM) => the same corpus + seed always yields the same TestCases, and the
reference answer is exactly recoverable from the supporting chunks. Kinds:

  cloze        blank out a number / proper noun in one sentence         (retrieval + exact answer)
  relation     "How is A related to B?"  from a graph Edge              (graph path + evidence)
  multi_hop    A -r1-> B -r2-> C : "what do you reach?"                  (chains of related facts)
  thematic     "What are the main themes ...?"  answer = topic labels    (community retrieval)
  unanswerable fabricated entity absent from corpus, answerable=False    (hallucination / refusal)

Hindi (lang="hi") variants are generated for relation / thematic / unanswerable so the
Phase-2 cross-lingual requirement can be scored with the same machinery.
"""
from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Sequence

from models import Chunk, TestCase
from core.chunking import split_sentences

NOT_ANSWERABLE = "Not answerable from the provided documents."
_NUM = re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?(?:%|percent|million|billion|thousand|crore|lakh|years?|months?|days?|km|kg|MW|GW|USD|EUR|INR)?\b")
_PROPER = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2})\b")
_STOP_CAPS = {"The", "This", "That", "These", "Those", "However", "In", "On", "At", "For", "And", "But", "Our", "Its", "His", "Her"}

_UNANS_EN = ["What is the annual budget of {x}?", "Who founded {x}?", "Where is the headquarters of {x}?",
             "What products does {x} manufacture?"]
_UNANS_HI = ["{x} का वार्षिक बजट कितना है?", "{x} की स्थापना किसने की?", "{x} का मुख्यालय कहाँ है?"]
_THEME_EN = "What are the main themes across these documents?"
_THEME_HI = "इन दस्तावेज़ों के मुख्य विषय क्या हैं?"
_SYL = ["ka", "vo", "ren", "mi", "zul", "tor", "ex", "lan", "qui", "bra", "thy", "nor", "ved", "sar", "pho"]


def _tid(kind: str, q: str, a: str) -> str:
    return f"{kind[:2]}_" + hashlib.sha1(f"{kind}|{q}|{a}".encode()).hexdigest()[:10]


def _rel(r: str) -> str:
    return re.sub(r"[_\-]+", " ", str(r)).strip().lower()


def _g(o: Any, name: str, default: Any = None) -> Any:
    return o.get(name, default) if isinstance(o, dict) else getattr(o, name, default)


def _cloze(chunks: Sequence[Chunk], rng: random.Random, n: int) -> list[TestCase]:
    sent_count: dict[str, int] = {}
    cands: list[tuple[str, str, str]] = []
    for c in chunks:
        for s in split_sentences(c.text):
            if 50 <= len(s) <= 240:
                sent_count[s] = sent_count.get(s, 0) + 1
                cands.append((c.chunk_id, s, ""))
    order = list(range(len(cands)))
    rng.shuffle(order)
    out: list[TestCase] = []
    for i in order:
        cid, s, _ = cands[i]
        if sent_count[s] != 1:
            continue                                  # ambiguous: same sentence in several places
        nums = [m.group(0).strip() for m in _NUM.finditer(s) if len(m.group(0).strip()) > 1 or m.group(0).strip().isdigit()]
        props = [m.group(1) for m in _PROPER.finditer(s) if m.group(1) not in _STOP_CAPS]
        pool = [t for t in nums + props if s.count(t) == 1]
        if not pool:
            continue
        ans = sorted(pool)[rng.randrange(len(pool))]  # sorted -> deterministic given rng state
        q = f"According to the documents, fill in the blank: \"{s.replace(ans, '_____')}\""
        out.append(TestCase(_tid("cloze", q, ans), q, ans, [cid], True, "cloze", "en"))
        if len(out) >= n:
            break
    return out


def _from_edges(edges: Sequence[Any], labels: dict[str, str], rng: random.Random, n_rel: int, n_hop: int,
                hindi: bool) -> list[TestCase]:
    good = [e for e in edges if _g(e, "evidence_chunk_ids") and _g(e, "source_node_id") in labels
            and _g(e, "target_node_id") in labels and _g(e, "status", "candidate") != "conflict"]
    good.sort(key=lambda e: _g(e, "edge_id"))
    rng.shuffle(good)
    out: list[TestCase] = []
    for e in good[:n_rel]:
        a, b = labels[_g(e, "source_node_id")], labels[_g(e, "target_node_id")]
        ans = f"{a} {_rel(_g(e, 'relation_type'))} {b}"
        ev = list(_g(e, "evidence_chunk_ids"))
        q = f"According to the documents, how is {a} related to {b}?"
        out.append(TestCase(_tid("relation", q, ans), q, ans, ev, True, "relation", "en"))
        if hindi:
            qh = f"दस्तावेज़ों के अनुसार, {a} और {b} के बीच क्या संबंध है?"
            out.append(TestCase(_tid("relation", qh, ans), qh, ans, ev, True, "relation", "hi"))
    by_src: dict[str, list[Any]] = {}
    for e in good:
        by_src.setdefault(_g(e, "source_node_id"), []).append(e)
    hops = [(e1, e2) for e1 in good for e2 in by_src.get(_g(e1, "target_node_id"), [])
            if _g(e2, "target_node_id") != _g(e1, "source_node_id")]
    rng.shuffle(hops)
    for e1, e2 in hops[:n_hop]:
        a, c = labels[_g(e1, "source_node_id")], labels[_g(e2, "target_node_id")]
        r1, r2 = _rel(_g(e1, "relation_type")), _rel(_g(e2, "relation_type"))
        q = f"Starting from {a}, follow the relation '{r1}' and then the relation '{r2}'. What do you reach?"
        ev = list(dict.fromkeys(list(_g(e1, "evidence_chunk_ids")) + list(_g(e2, "evidence_chunk_ids"))))
        out.append(TestCase(_tid("multi_hop", q, c), q, c, ev, True, "multi_hop", "en"))
    return out


def _thematic(topics: Sequence[Any], hindi: bool) -> list[TestCase]:
    top = sorted(topics, key=lambda t: -len(_g(t, "chunk_ids", [])))[:5]
    if not top:
        return []
    ans = "; ".join(_g(t, "label") for t in top)
    ev = [c for t in top for c in _g(t, "chunk_ids", [])[:2]]
    out = [TestCase(_tid("thematic", _THEME_EN, ans), _THEME_EN, ans, ev, True, "thematic", "en")]
    if hindi:
        out.append(TestCase(_tid("thematic", _THEME_HI, ans), _THEME_HI, ans, ev, True, "thematic", "hi"))
    return out


def _fake_names(chunks: Sequence[Chunk], rng: random.Random, n: int) -> list[str]:
    blob = " ".join(c.text for c in chunks).lower()
    names: list[str] = []
    for _ in range(n * 20):
        nm = "".join(rng.choice(_SYL) for _ in range(3)).title() + rng.choice([" Industries", " Labs", " Group", " Institute"])
        if nm.lower() not in blob and nm not in names:
            names.append(nm)
        if len(names) == n:
            break
    return names


def _unanswerable(chunks: Sequence[Chunk], rng: random.Random, n: int, hindi: bool) -> list[TestCase]:
    out: list[TestCase] = []
    for i, x in enumerate(_fake_names(chunks, rng, n)):
        q = _UNANS_EN[i % len(_UNANS_EN)].format(x=x)
        out.append(TestCase(_tid("unanswerable", q, ""), q, NOT_ANSWERABLE, [], False, "unanswerable", "en"))
        if hindi:
            qh = _UNANS_HI[i % len(_UNANS_HI)].format(x=x)
            out.append(TestCase(_tid("unanswerable", qh, ""), qh, NOT_ANSWERABLE, [], False, "unanswerable", "hi"))
    return out


def generate_test_cases(chunks: Sequence[Chunk], edges: Sequence[Any] | None = None, *,
                        nodes: Sequence[Any] | None = None, topics: Sequence[Any] | None = None,
                        n: int = 40, seed: int = 7, include_hindi: bool = True) -> list[TestCase]:
    """Agreed interface + optional nodes/topics. Same inputs + seed -> identical output.

    `nodes` (node_id,label) is needed to verbalise edges; without it only cloze / thematic /
    unanswerable cases are produced, so this works before Member 2's graph exists.
    """
    rng = random.Random(seed)
    chunks = [c for c in chunks if c.text.strip()]
    if not chunks:
        return []
    labels = {_g(nd, "node_id"): _g(nd, "label") for nd in (nodes or [])}
    n_un = max(2, n // 5)
    n_rel = n // 4 if edges and labels else 0
    n_hop = n // 6 if edges and labels else 0
    cases: list[TestCase] = []
    cases += _from_edges(edges or [], labels, rng, n_rel, n_hop, include_hindi) if n_rel else []
    cases += _thematic(topics or [], include_hindi)
    cases += _unanswerable(chunks, rng, n_un, include_hindi)
    cases += _cloze(chunks, rng, max(0, n - len([c for c in cases if c.lang == "en"])))
    seen, uniq = set(), []
    for c in cases:
        if c.test_id not in seen:
            seen.add(c.test_id)
            uniq.append(c)
    return uniq
