# Research and design rationale for ReliaGraph

## The problem we are targeting

ReliaGraph targets the combination of (a) topic-level exploration across a corpus, (b) relation-chain exploration through evidence-linked graph edges, and (c) safe document lifecycle changes that preserve conflicting claims and provenance. The project should not claim that individual components such as GraphRAG, multilingual embeddings, source citations or incremental graph updates are themselves novel.

## Research basis

1. **BEIR — heterogeneous zero-shot IR evaluation**: 18 datasets across tasks/domains found substantial out-of-distribution variability; BM25 is a strong robust baseline, while reranking can improve average quality at additional compute cost. This motivates measuring per-domain results and retaining a lexical baseline instead of assuming one embedding model generalizes. https://arxiv.org/abs/2104.08663
2. **RAPTOR (ICLR 2024)**: recursively clusters and summarizes chunks into a tree to help broad questions that a flat chunk retrieval approach may miss. ReliaGraph uses topic coverage and source-backed representative passages as a lighter prototype-level analogue, not a full RAPTOR implementation. https://arxiv.org/abs/2401.18059
3. **Multilingual E5**: model card instructs using `query: ` and `passage: ` prefixes for asymmetric retrieval; scores are rankings, not calibrated probabilities. The adapter should use the prefixes and abstain when weights are unavailable. https://huggingface.co/intfloat/multilingual-e5-small
4. **GLiNER-ReLEx**: proposes joint zero-shot NER and relation extraction with configurable entity/relation types. It is an optional candidate generator, not a truth oracle. https://arxiv.org/abs/2605.10108
5. **OWASP RAG Security Cheat Sheet**: recommends provenance/integrity, context delimiters, treating retrieved text as untrusted data, scanning for prompt injection, output validation and fail-closed behaviour. ReliaGraph provides a prototype defence layer but cannot claim immunity. https://cheatsheetseries.owasp.org/cheatsheets/RAG_Security_Cheat_Sheet.html
6. **Construction of Knowledge Graphs: Current State and Challenges (2024)**: observes that incremental KG construction, provenance, entity resolution/fusion and quality assurance are difficult and are not always systematically integrated. This motivates the integration focus, but does not prove our combination is unprecedented. https://www.mdpi.com/2078-2489/15/8/509

## Proposed contribution (testable, not an unsupported novelty claim)

**Evidence-gated delta graph maintenance**: only update the graph contribution of an added/edited/deleted document; retain claim instances separately with source provenance; validate each claim instance's endpoints and evidence before storage; preserve conflicting claims; route broad/factual/relationship-chain queries differently; and expose a fail-closed path for multilingual model loss. This integrated design is intended to trade unsafe coverage for explicit review/abstention.

What we should evaluate:

- Relation precision/recall/F1 on independently annotated held-out document domains.
- Accepted-edge coverage, review-required rate, and abstention correctness (a high precision with almost no accepted edges is not a successful system).
- Retrieval Hit@k and MRR by query language and domain.
- Relation-chain validity: every step must cite an evidence chunk and topic-membership edges must not be misrepresented as factual predicates.
- Add/edit/delete latency at 500+ documents, including PDF extraction and model inference measured separately.
- Injection detection recall and benign false-positive rate on both known and held-out attacks.
- Paired Phase 1/Phase 2 answer quality on the same fixed test set; report the absolute percentage-point change and its uncertainty.

## Data independence

No universal extractor is equally accurate on every domain or writing style. We avoid domain-specific rules as the main approach, use generic extractors/embeddings as optional candidate generators, preserve the original source span, and hold a model-independent deterministic fallback. All reported quality metrics must be measured on a separate labelled set; the synthetic data is for smoke/performance testing only.
