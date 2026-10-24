"""Lightweight query routing for Phase 1: thematic vs factual vs relation-chain queries."""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict

THEMATIC_CUES = (
    "main themes", "key themes", "overall themes", "common themes", "main topics", "key topics",
    "summarize these documents", "summarise these documents", "summarize the documents", "summarise the documents",
    "summarize this collection", "summarise this collection", "summarize the collection", "summarise the collection",
    "overall picture", "overall findings", "overall trends", "what is this collection about", "across these documents",
    "across the documents", "across all documents", "across the corpus", "across this corpus", "across the collection",
    "in all these documents", "common patterns", "common ideas", "what are the main themes", "what are the key themes",
    "what are the main topics", "what are the key topics", "give me an overview", "high-level overview",
    "मुख्य विषय", "मुख्य थीम", "इन दस्तावेज़ों का सार", "कुल मिलाकर", "प्रमुख विषय",
    "इन दस्तावेजों के मुख्य विषय", "सभी दस्तावेज़ों का सार", "दस्तावेजों में क्या समान है", "कौन से विषय",
    "सभी दस्तावेजों में", "दस्तावेजों के मुख्य विषय", "मुख्य विषय क्या हैं"
)
CHAIN_CUES = (
    "how is", "how are", "relationship between", "connect", "chain of", "path from", "linked to", "related through",
    "कैसे जुड़ा", "कैसे जुड़े", "कैसे जुड़ी", "कैसे संबंधित", "के बीच संबंध", "किस प्रकार जुड़ा", "किस तरह जुड़े",
    "जुड़ाव", "रिश्ता", "संबंध", "कनेक्शन", "के माध्यम से जुड़ा", "के माध्यम से जुड़े"
)

@dataclass
class QueryRoute:
    route: str
    explanation: str
    use_topic_summaries: bool
    use_graph_traversal: bool
    use_vector_retrieval: bool
    def to_dict(self): return asdict(self)

def route_query(query: str) -> QueryRoute:
    q = re.sub(r"\s+", " ", str(query or "").casefold()).strip()
    if not q:
        return QueryRoute("empty", "No query was supplied.", False, False, False)
    if any(cue in q for cue in THEMATIC_CUES):
        return QueryRoute("thematic", "Broad/collection-level question detected; summarize across topic clusters.", True, True, True)
    if any(cue in q for cue in CHAIN_CUES):
        return QueryRoute("relation_chain", "Question asks how concepts are related; retrieve evidence and traverse graph paths.", False, True, True)
    return QueryRoute("factual", "Default to specific factual retrieval and source-backed evidence.", False, False, True)
