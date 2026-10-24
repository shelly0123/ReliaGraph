# ReliaGraph graph and storage update

- Graph display now uses a force-directed relaxation layout with cached coordinates, reduced edge opacity at low zoom, labels hidden while zoomed out, and bounded text cards when labels are visible. This is inspired by general literature-map usability patterns, not a copy of ResearchRabbit's proprietary interface.
- Node labels are line-wrapped and bounded; users can select nodes, pan, zoom, filter and inspect the node inspector.
- Upload limit is 50 MB per file and 200 MB total/unpacked, subject to server request and ZIP safety caps. This is not an unlimited upload promise.
- Added a destructive Reset all sources action; it preserves accounts. Individual source deletion remains available. Re-importing the same named dataset is the replace/edit workflow.
- Added optional Neo4j graph mirror and Chroma persistent vector collection adapter. Default remains SQLite/NetworkX + local retrieval so no external service is falsely reported as running. Configure and verify status before claiming Neo4j/Chroma acceptance.
- Official references: Neo4j Python Driver manual https://neo4j.com/docs/python-manual/current/install/ ; Chroma persistent client guide https://cookbook.chromadb.dev/core/clients/ ; ResearchRabbit graph overview https://libguides.trinity.edu/researchrabbit/papers
