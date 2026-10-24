# ReliaGraph product comparison and differentiation (checked 24 October 2026)

This is a focused capability comparison based on public official documentation. It is not a complete market survey, benchmark shootout or claim that ReliaGraph is better overall.

| System | Documented strengths | Practical trade-offs / context | ReliaGraph angle for the hackathon |
|---|---|---|---|
| Microsoft GraphRAG | Entity, relationship and claim extraction; graph communities and hierarchical summaries; local, global, DRIFT and basic search | Its indexing can be expensive and the official project describes itself as a research project in maintenance mode; model/configuration choices affect cost and results | A simpler local-first student prototype with one UI for imports, source traces, graph updates, conflicts, review signals and latency evidence |
| LightRAG | Graph-enhanced entity/relationship retrieval, dual-level retrieval and incremental insertion/update | It already covers incremental graph retrieval, so incremental updating alone is not a novel claim; deployment and model/storage configuration still matter | Emphasize source-scoped claim auditing, inspectable conflicts and fail-closed multilingual behavior as a combined workflow |
| Neo4j GraphRAG | First-party Python package, graph + vector/hybrid retrieval, graph traversal and graph-query integrations | Can leverage Neo4j's database ecosystem; a complete deployment entails a graph database and model configuration | A lightweight SQLite/local prototype to demo and evaluate without requiring a separately operated graph database |

## Research sources

- Microsoft GraphRAG documentation: https://github.com/microsoft/graphrag and https://microsoft.github.io/graphrag/index/overview/
- LightRAG project site: https://lightrag.github.io/ and paper: https://arxiv.org/abs/2410.05779
- Neo4j GraphRAG overview: https://neo4j.com/labs/genai-ecosystem/graphrag/
- Neo4j GraphRAG Python docs: https://neo4j.com/docs/neo4j-graphrag-python/current/

## What should be called our contribution

The defensible contribution is the **evidence-first workflow**, not the graph itself: (1) document-scoped incremental graph contribution, (2) preserving conflicting source claims for side-by-side inspection, (3) routing thematic versus factual versus relation-chain queries, (4) returning evidence and paths with explicit abstention when support or model capability is missing, and (5) exposing system and evaluation results in one approachable local dashboard.

These parts are not globally unprecedented as isolated ideas. We have not established worldwide novelty, patentability, or industry-leading quality. Any claim of superiority must be supported by a controlled comparison on the same independently labelled datasets, including latency, recall, precision and answer quality.
