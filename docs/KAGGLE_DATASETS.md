# Kaggle dataset catalogue and responsible import

The dashboard lists the following publicly documented Kaggle datasets. The build environment could not download the full original files because external DNS resolution failed. To give the dashboard one real-data example that works offline, the ZIP includes `examples/dashboard_datasets/kaggle-energy-sample.csv`: a small, 20-row/selected-column excerpt transcribed from a public CSV mirror of the Global Sustainable Energy dataset. It is explicitly labelled as an adapted excerpt, not as the full source file. The other full dataset cards correctly say “Download + upload”.

| Dataset | Dataset page | Licence reported by Kaggle | Why it is useful | Import guidance |
|---|---|---|---|---|
| arXiv metadata | https://www.kaggle.com/datasets/Cornell-University/arxiv | CC0 for metadata; paper/full-text terms vary | Research titles, abstracts and categories can test themes and linked concepts across papers | The full corpus is extremely large. Use `examples/import_kaggle_arxiv.py` to make a 500–600 record metadata-only JSONL subset from a local download, then upload that file or its resulting `.jsonl` through Sources & datasets. |
| Global data on sustainable energy (2000–2020) | https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy | CC BY 4.0 (Kaggle page) | A structured non-NLP domain that checks whether CSV rows become source-linked records without topic-specific logic | Load the bundled 20-row excerpt now. For full-corpus testing, download the original CSV, then upload it through Sources & datasets with source URL attribution. |
| Daily climate time-series data | https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data | CC0 (Kaggle page) | Small weather time-series rows test a non-research domain and varied column structures | Download a CSV file and upload it. Use a distinct dataset name. |

## Current bundled test data

- `kaggle-energy-sample.csv`: 20 selected country-year rows and four indicators from the public CSV mirror of the Kaggle Global Sustainable Energy dataset. This is a small adapted excerpt, CC BY 4.0; it is not the full dataset. The UI records the Kaggle source link and attribution when loaded.
- `synthetic-cyber.csv`, `synthetic-biomed.csv`, and `synthetic-energy.csv`: original project-created fixtures, not Kaggle data. These demonstrate multiple-domain workflows but do not prove accuracy on real domain datasets.

## What was actually tested for this release

- Three bundled synthetic fixtures imported through the dashboard API into one workspace.
- User-uploaded CSV, JSONL/TXT inside ZIP, retrieval questions, source-linked graph creation, benchmark execution, edit by re-importing a shorter dataset, and one-document deletion.
- The bundled 20-row sustainable-energy excerpt was loaded through the same API used by the dashboard and generated 20 source-linked records in about 0.248 seconds on the build host. This is a real-data excerpt import, not a full raw-file ingestion benchmark. A full licensed Kaggle file still needs a local acceptance run.

For full external dataset imports, capture the dataset version/page, licence, date downloaded, selected file, row count, graph counts, processing time, model state, and report JSON. Do not upload a multi-gigabyte source file blindly; use a bounded subset first.
