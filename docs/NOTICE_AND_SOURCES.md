# ReliaGraph — consolidated notices, attribution and data sources

## Project source

The unified repository incorporates and adapts code supplied in the team's Member 1 and Member 2 packages. Preserve the team's authorship/attribution agreed by its members. Before public redistribution, confirm that every contributor has approved the release and add the team's chosen repository licence. No licence is silently assigned by this document.

## Demo corpus: MUJ public information

`examples/data/muj_paraphrased_corpus.json` contains 22 short independently phrased factual summaries and a small manually curated gold-edge list. It does not reproduce full webpage passages or images and is not official MUJ wording, policy, endorsement or affiliation. Its references are:

- About MUJ: https://www.jaipur.manipal.edu/about-muj.php
- Academic Rules & Regulations: https://jaipur.manipal.edu/academic-rules-regulations.php
- Examination information: https://jaipur.manipal.edu/examination.php
- MUJ home: https://www.jaipur.manipal.edu/

The source records include title, URL and access date. Website content can change. Cite the official page and date in the presentation; don't imply that a paraphrase is a current official policy. The synthetic version-change example is explicitly fictional and is not a MUJ deadline/event.

## Scale dataset

- `examples/data/synthetic_scale_600.jsonl`: original synthetic multi-domain text records. It is for throughput and integration testing only, not a source of accuracy claims.
- Optional arXiv metadata: https://www.kaggle.com/dsv/6015950. The Kaggle page states CC0 for metadata. The current package does not bundle the full arXiv Kaggle data.
- Bundled dataset excerpt: `examples/dashboard_datasets/kaggle-energy-sample.csv` is an adapted 20-row/selected-column excerpt of the “Global Data on Sustainable Energy” dataset: Kaggle page https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy; public CSV mirror https://github.com/AnshTanwar/Global-Data-on-Sustainable-Energy. The Kaggle card states CC BY 4.0. Attribution: dataset page and dataset author Ansh Tanwar; licence https://creativecommons.org/licenses/by/4.0/. Changes made: selected 20 country-year rows across Afghanistan, Albania, Algeria and Angola, and retained six columns (entity, year and four indicators). This is an excerpt, not the full dataset; missing values are preserved as empty CSV cells. Full original file was not downloaded by the build environment.
- arXiv licensing guidance: https://info.arxiv.org/help/bulk_data.html. Do not infer that CC0 metadata grants permission to redistribute every full-text PDF. Per-paper licensing varies; avoid bundling papers unless individual rights are verified.

## Project-created evaluation fixtures

- `examples/data/domain_shift_benchmark.json`: 20 short manually labelled sentences across four domains. This small fixture was used during development and is a smoke test, not a held-out industry benchmark.
- `examples/data/phase2_prompt_injection_cases.json`: 36 original planted attack strings and 20 benign controls. These are test inputs; successful detection is not proof against unseen attacks.
- `examples/data/muj_paraphrased_corpus.json`: small development corpus with curated gold edges; results are development smoke metrics, not independent generalization evidence.
- `examples/data/phase1_phase2_regression_template.jsonl`: template only; not a measured answer-quality benchmark.
- `examples/data/domain_candidate_review_template.csv`, `examples/data/evidence_review_template.csv`, `examples/data/relation_review_template.csv`: blank human-review templates; they are not training data and do not produce a trained calibrator until real labels are provided.

## Research resources studied

- Edge et al., GraphRAG, 2024: https://arxiv.org/abs/2404.16130. Research inspiration only; no Microsoft code or figures are copied into this repository.
- Guo et al., LightRAG, 2024 preprint / later publication: https://arxiv.org/abs/2410.05779. Research inspiration only; no LightRAG source implementation is bundled.
- Sarthi et al., RAPTOR, ICLR 2024: https://arxiv.org/abs/2401.18059. Motivates hierarchical organization for broad questions; ReliaGraph is not a reproduction of RAPTOR.
- BEIR, heterogeneous information-retrieval benchmark: https://arxiv.org/abs/2104.08663. Motivates per-domain evaluation and keeping a lexical baseline.
- Pai et al., open information extraction survey, Findings of EMNLP 2024: https://aclanthology.org/2024.findings-emnlp.560/.
- Construction of Knowledge Graphs: Current State and Challenges, 2024: https://www.mdpi.com/2078-2489/15/8/509.
- RAGAS, automated RAG evaluation: https://arxiv.org/abs/2309.15217.
- OWASP RAG Security Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/RAG_Security_Cheat_Sheet.html.

These works are established prior art. ReliaGraph should not claim that graph RAG, incremental updates, topic modelling, source citations, prompt-injection scanning or multilingual embeddings are individually new. The project's proposed contribution is the measured integration of source-scoped evidence-gated graph changes, preserved competing claims, query routing and explicit safe abstention under model failure. Whether that combination is globally novel has not been established.

## Optional models and their sources

The package does not bundle pretrained model weights. When enabled, these model IDs may download weights from the model host; first-run internet, disk and RAM are required. Check the current model card and dependency licence before use/redistribution.

- `knowledgator/gliner-relex-multi-v1.0`: https://huggingface.co/knowledgator/gliner-relex-multi-v1.0 — optional zero-shot relation/entity candidate generation; not a truth oracle.
- `intfloat/multilingual-e5-small`: https://huggingface.co/intfloat/multilingual-e5-small — multilingual retrieval. Model usage expects `query: ` and `passage: ` prefixes; similarity values are ranking signals, not calibrated truth probabilities.
- `sentence-transformers/all-MiniLM-L6-v2`: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 — optional English embeddings.
- `cross-encoder/nli-deberta-v3-small`: https://huggingface.co/cross-encoder/nli-deberta-v3-small — optional NLI review signal, not a final fact checker.
- `en_core_web_sm`, if installed separately: https://huggingface.co/spacy/en_core_web_sm.

No model inference for GLiNER or multilingual E5 was measured in the development environment because optional dependencies/weights were unavailable. Any results produced after enabling them must name the exact model revision, device, inference/load times and dataset.

## Third-party libraries

The `requirements.txt` and `requirements-optional-models.txt` enumerate dependencies. Their licenses are maintained by their authors; inspect installed package metadata and model cards for the exact versions used. The project does not bundle third-party graph diagrams or copy implementations from GraphRAG, LightRAG, RAGAS or BERTopic. Documentation links are provided for context and credit.

## Metrics disclaimer

The result metrics are explicitly limited in `docs/ROBUSTNESS_RESULTS.md`. Synthetic performance fixtures establish only run-time behaviour on the measured host. Small development relation sets cannot establish generalization. Injection detection is pattern-specific. Hindi→English quality and Phase 1 versus Phase 2 quality are not yet measured with actual optional model inference and a paired held-out dataset.
