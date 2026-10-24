# ReliaGraph Phase 2 — Research, design rationale, and prototype plan

## Problem definition

**Interactive Knowledge Base — Topic-Augmented Knowledge Graph Builder** means the system should not only extract facts. It should organize document passages into topics, represent concepts and typed relationships in a graph, preserve the source evidence behind each relationship, and route questions according to intent:

- **Thematic questions** (e.g., “What are the main themes across these documents?”): topic summaries and topic-level graph communities.
- **Factual questions**: retrieve specific evidence chunks and answer only when support is sufficient.
- **Relation-chain questions**: find short paths through source-backed graph edges and show the supporting evidence for each step.
- **Hindi query over English documents**: use multilingual passage retrieval, then show the English evidence and graph path.

Phase 2 adds: 500+ document scale reporting; add/edit/delete without full rebuild and under 60 seconds; preserve contradictory claims with both sources; defend against at least 20 planted prompt injections; and keep Phase 1 answer quality within 3 percentage points using a paired regression test.

## Research foundation

1. **OpenIE survey (Pai et al., Findings of EMNLP 2024)** describes rule-based, neural, and pretrained language-model approaches to relation extraction and their trade-offs. It motivates a hybrid extractor rather than a hand-written vocabulary for each dataset: https://aclanthology.org/2024.findings-emnlp.560/
2. **GLiNER-ReLEx model card** describes zero-shot joint named-entity and relation extraction with customizable label descriptions. ReliaGraph uses it as an optional candidate generator; model scores are not interpreted as truth probabilities: https://huggingface.co/knowledgator/gliner-relex-multi-v1.0
3. **Multilingual E5 Small** supports multilingual semantic retrieval when the trained `query: ` and `passage: ` prefixes are used. ReliaGraph uses it as an optional local Hindi-to-English retrieval component: https://huggingface.co/intfloat/multilingual-e5-small
4. **OWASP RAG Security Cheat Sheet** recommends provenance, scanning for hidden instructions, delimiting retrieved context as untrusted data, output validation, deletion propagation and defence in depth. It does not imply that prompt injection can be eliminated by a regex scanner: https://cheatsheetseries.owasp.org/cheatsheets/RAG_Security_Cheat_Sheet.html
5. **GraphRAG** provides research context for entity graphs and thematic/community summaries across documents: https://arxiv.org/abs/2404.16130
6. **LightRAG** explores entity-level and higher-level graph-assisted retrieval: https://arxiv.org/abs/2410.05779
7. **RAGAS** provides reference methods for assessing retrieval-augmented generation: https://arxiv.org/abs/2309.15217 and https://docs.ragas.io/

These works establish that many ingredients are known. ReliaGraph's contribution is the prototype integration of topic-aware graph construction, claim-level evidence provenance, selective review, incremental updates, conflict preservation, injection-resilient model context, cross-lingual evidence retrieval, and a paired Phase 1/Phase 2 regression gate. Do not claim this integration is globally first without a deeper literature search.

## Why model scores alone are insufficient

Relation-extraction scores and NLI labels are signals, not calibrated probabilities that a statement is true. Results change with domain, phrasing, entity type, language, source quality, and model version. Consequently:

- Preserve the exact quote and stable chunk/document ID for every candidate edge.
- Reject a candidate if source evidence or its subject/object cannot be anchored in that source.
- Use distinct extractors or independent sources as corroboration, but never treat agreement as proof.
- Use a manually labelled calibration set to choose thresholds; keep a separate held-out test set.
- Report **precision, recall, F1, review/abstention coverage, false-positive examples, and per-domain results** rather than only one aggregate accuracy score.
- If the evidence is weak or the domain drifts, preserve the claim as `review_required` or abstain. Higher precision may mean lower automatic coverage, which must be reported honestly.

A universal guarantee of high precision across arbitrary unseen domains is not realistic. A defensible target is controlled quality with measured uncertainty and safe abstention.

## Data plan and licensing

### 1. Main scale corpus: Kaggle arXiv metadata

Dataset: https://www.kaggle.com/dsv/6015950

Use 500–600 title + abstract records from multiple categories such as `cs.AI`, `cs.CL`, `cs.IR`, `cs.LG`, `cs.CY`, and `cs.HC`. The dataset page states that the metadata is CC0. Each record is linked back to its arXiv abstract page. The project importer deliberately does not package the PDFs. Individual paper full-text licences vary; check an individual paper's licence before downloading, adapting, or redistributing its PDF. This corpus is useful for scale and topical variety, but it is **not** a gold relation-extraction benchmark.

### 2. Cohesive presentation corpus: MUJ public sources

Use independently paraphrased summaries with source URLs for a coherent live demo. Do not represent paraphrases as official policy text or MUJ endorsement. A synthetic conflicting version must be labelled fictional.

### 3. Accuracy evaluation corpus

Create a separately labelled collection across at least four domains. Human reviewers label whether candidate triples are `supported`, `contradicted`, or `insufficient/unclear`; include negative examples where two entities co-occur but no relationship is stated, as well as negation and temporal changes. Split by document/source—not random sentence—so near-duplicate passages cannot leak into both tuning and test sets. Aim for at least 100 reviewed candidates for initial evidence, more if time permits. Publish the size and limits of the set.

### 4. Prompt injection corpus

Use original authored attack-like examples and benign controls. Current local test has 36 attacks and 20 benign examples. This only tests these known examples, not unknown adversarial prompts. Add unseen cases, unicode/obfuscation cases, multi-chunk attacks, benign mentions of security terminology, and verify that unsafe text is not sent to model context.

## Incremental architecture

Represent document-derived claims separately, with `doc_id`, `version`, `chunk_id`, subject, predicate, object, evidence text, source URL/title, and status. For an edit/delete, remove or deactivate only the changed document's active claim contributions, retain an audit/version record, and rebuild only the affected shared graph projection. Delete associated chunks, cached embeddings and derived data from active retrieval, while retaining only version history intended by policy. Keep competing values as separate provenance-backed claims; do not globally overwrite one value or assume that any two different values are a contradiction without checking the predicate and context.

## Query strategy

A lightweight intent router chooses one or more routes:

- **Thematic:** topic labels, community summaries and representative chunks.
- **Fact:** multilingual/vector retrieval plus direct source evidence.
- **Relation chain:** retrieve relevant seeds, traverse a bounded graph path, and return every edge with its source chunk and link.
- **Hindi-English:** use the multilingual model only if loaded; otherwise clearly warn that cross-lingual semantic retrieval is unavailable rather than silently promise translation capability.

Paths explain how the system connected evidence; they do not by themselves prove that all edges are true.

## Evaluation protocol

1. Record the baseline Phase 1 outputs on a fixed test set.
2. Run Phase 2 on the exact same test questions and reference answers.
3. Use the same answer-quality metric for both versions and report the difference in percentage points. The Phase 2 gate is satisfied only if the real paired run drops no more than three percentage points.
4. For relation extraction, report exact/canonical triple precision, recall and F1 by domain, plus review coverage. Keep calibration and test sources separate.
5. For Hindi-to-English retrieval, use labelled pairs and report Hit@1, MRR, and query latency with model cold-load separately.
6. For 500+ scale, report total ingestion time, number of records, nodes, edges, storage size and hardware. Separate cold model load, PDF/text parsing, model inference, and database/graph work.
7. For edits/deletes, measure end-to-end time on the target laptop including the document's own parse/extraction, excluding only one-time model download. The included synthetic benchmark isolates database/graph updates and is not an end-to-end ML ingestion result.
8. For injection, report detection recall and benign false-positive rate on the known test set, plus tests of whether suspicious text was excluded from model-facing context. Never say “prompt injection proof.”

## Originality claims we can defend

A defensible claim: “ReliaGraph integrates source-anchored, domain-general candidate extraction with selective review, claim-level provenance, incremental document updates, contradiction preservation, prompt-injection context filtering and multilingual graph-path retrieval, and evaluates these behaviours on fixed, reproducible test sets.”

Avoid saying: “first in the world,” “100% accurate,” “works for all data,” “completely prevents prompt injection,” or “industry-level accuracy” unless independently demonstrated by a suitable benchmark.
