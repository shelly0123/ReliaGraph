"""Prompt-injection screening for document text (Phase 2: 20+ planted injections).

Policy: instruction-like sentences found INSIDE documents are quarantined -- removed
from Chunk.text (so they never reach retrieval, triple extraction or the LLM
context) but kept in Chunk.quarantined so the UI can show "N injections blocked".

This is layer 1 of 3. Layers 2/3 (Member 3): wrap retrieved context in delimiters
and tell the generator it is untrusted DATA; never execute instructions from it.
Detection is regex-based: O(n), deterministic, offline, auditable.
"""
from __future__ import annotations

import re
import unicodedata

_ZW = dict.fromkeys(
    map(ord, "\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e"
             "\u2060\u2061\u2062\u2063\u2064\ufeff\u00ad"), None)

# Leetspeak folding so "1gn0re previous instructi0ns" cannot slip through.
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def fold(text: str) -> str:
    """Canonical form used ONLY for matching (never stored)."""
    t = unicodedata.normalize("NFKC", text).translate(_ZW).lower()
    return re.sub(r"\s+", " ", t)


_STRONG = [
    # "ignore / disregard all previous instructions"
    r"\b(ignore|disregard|forget|override|bypass|skip|discard)\b[^.!?\n]{0,50}\b(instruction|prompt|rule|direction|guideline|context|constraint|polic(y|ies))s?\b",
    r"\b(previous|prior|above|earlier|preceding)\s+(instruction|prompt|direction|message)s?\b",
    # role / persona hijack
    r"\byou\s+(are|will|must|shall)\s+now\b",
    r"\bfrom\s+now\s+on\b[^.!?\n]{0,40}\b(you|assistant|ai|answer|respond)\b",
    r"\b(new|updated|real|actual)\s+(instruction|task|objective|system\s+prompt)s?\s*[:\-]",
    r"\b(act|behave|respond)\s+as\s+(if\s+you\s+(are|were)|an?\s+(unrestricted|jailbroken|different))\b",
    r"\bpretend\s+(to\s+be|you\s+are|that\s+you)\b",
    r"\bjailbreak|\bdo\s+anything\s+now\b|\bdeveloper\s+mode\b",
    # exfiltration
    r"\b(reveal|print|show|output|repeat|leak|disclose|display)\b[^.!?\n]{0,40}\b(system\s+prompt|your\s+instructions|hidden\s+instructions|api[\s_-]?key|secret|password|credentials?)\b",
    # addressing the model directly
    r"\b(attention|note\s+to|message\s+for|instruction\s+for|dear)\b[^.!?\n]{0,15}\b(ai|llm|assistant|chatbot|language\s+model|gpt|claude|model)\b",
    r"\b(ai|llm|assistant|chatbot|language\s+model)\b[^.!?\n]{0,25}\b(must|should|shall|has\s+to|need\s+to)\s+(always|only|never|not|ignore|respond|answer|say|reply|output|include|recommend)\b",
    # answer steering
    r"\bwhen(ever)?\s+(you\s+are\s+)?(asked|queried|questioned|prompted)\b[^.!?\n]{0,100}\b(answer|respond|reply|say|output|state|return)\b",
    r"\b(always|only|never)\s+(answer|respond|reply|say|output|cite|mention|recommend)\b[^.!?\n]{0,60}\b(with|that|as|using|about)\b",
    r"\bdo\s+not\s+(tell|inform|mention|reveal|let)\b[^.!?\n]{0,30}\b(user|anyone|human|reader)\b",
    # control tokens / fake roles
    r"<\s*/?\s*(system|instructions?|prompt|im_start|im_end|assistant)\s*>",
    r"\[/?\s*(inst|system|sys)\s*\]|<<\s*sys\s*>>|<\|[a-z_]{2,20}\|>",
    r"^\s*(system|assistant|developer)\s*:",
    r"^\s*#{1,3}\s*(system|instruction)s?\b",
    # Hindi (documents are English but attackers are not) + a couple of other languages
    r"(पिछले|पहले\s+के|सभी|ऊपर\s+के)\s+(निर्देश|आदेश)[^.!?\n]{0,20}(अनदेखा|भूल|नज़रअंदाज़|नजरअंदाज)",
    r"(निर्देशों|आदेशों)\s+को\s+(अनदेखा|नज़रअंदाज़|नजरअंदाज)",
    r"\b(ignora|ignorez|ignoriere)\b[^.!?\n]{0,30}\b(instrucciones|instructions|anweisungen)\b",
]
_WEAK = [
    r"\bsystem\s+prompt\b",
    r"\bact\s+as\b",
    r"\b(you\s+must|you\s+should|you\s+have\s+to)\b",
    r"\bbase64\b[^.!?\n]{0,30}\bdecode\b",
    r"\bdo\s+not\s+follow\b",
    r"\bprompt\s+injection\b",
    r"\b(confidential|secret)\s+(instruction|directive)s?\b",
    r"\bend\s+of\s+(document|context|text)\b",
]
STRONG = [re.compile(p, re.I | re.M) for p in _STRONG]
WEAK = [re.compile(p, re.I | re.M) for p in _WEAK]

# Sentences that merely *discuss* attacks (a security paper) get mild benefit of the doubt.
_DISCUSSION = re.compile(r"\b(attack|vulnerab|defen[cs]e|mitigat|detect|research|paper|study|example\s+of|known\s+as)\w*\b", re.I)


def injection_score(sentence: str) -> float:
    """0 = clean. >= 2 is quarantined. Strong pattern = 2 each, weak = 1 each."""
    f = fold(sentence)
    leet = f.translate(_LEET)
    s = 0.0
    for pat in STRONG:
        if pat.search(f) or pat.search(leet):
            s += 2.0
    for pat in WEAK:
        if pat.search(f):
            s += 1.0
    if 0 < s < 4 and _DISCUSSION.search(f):
        s -= 1.0
    return s


def is_injection(sentence: str, threshold: float = 2.0) -> bool:
    return injection_score(sentence) >= threshold


def sanitize_for_llm(text: str) -> str:
    """Belt-and-braces for text that did not pass through the chunker (e.g. chat input)."""
    from core.chunking import split_sentences
    return " ".join(s for s in split_sentences(text) if not is_injection(s))
