"""Run the generalized, batched relation extractor on 500+ arXiv metadata documents.

This uses only title+abstract metadata (not full paper PDFs), and groups records by
arXiv categories as seed topics for the scale run. It requires the local metadata file
and optional GLiNER weights. No quality score is fabricated without a labelled gold set.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from models import Chunk, Topic
from core.phase2.batched_gliner import BatchedGLiNERRelexExtractor
from core.graph_builder import build_graph

ROOT = Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input", default="data/phase2/arxiv_500.jsonl")
    p.add_argument("--limit", type=int, default=600)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--cache", default="data/cache/arxiv_gliner.sqlite3")
    p.add_argument("--output", default="examples/outputs/arxiv_500_report.json")
    args=p.parse_args()
    all_start=time.perf_counter()
    src=Path(args.input)
    if not src.exists(): raise SystemExit(f"Missing {src}. First run examples/import_kaggle_arxiv.py")
    load_start=time.perf_counter()
    records=[]
    with src.open(encoding="utf-8") as f:
        for line in f:
            try: records.append(json.loads(line))
            except json.JSONDecodeError: continue
            if len(records)>=args.limit: break
    metadata_load_seconds=time.perf_counter()-load_start
    prep_start=time.perf_counter()
    chunks=[]; topics_by_id={}
    for r in records:
        cid=f"{r['doc_id']}::abstract"
        cats=list(r.get("categories", []))
        topic_ids=[]
        for cat in cats:
            tid=f"arxiv-category:{cat}"; topic_ids.append(tid)
            topics_by_id.setdefault(tid, {"topic_id":tid,"label":cat,"keywords":[cat],"chunk_ids":[]})
            topics_by_id[tid]["chunk_ids"].append(cid)
        chunks.append(Chunk(chunk_id=cid, doc_id=r["doc_id"], text=r["text"], topic_ids=topic_ids,
                            source_url=r.get("source_url"), source_title=r.get("title"), source_kind="arxiv_metadata",
                            metadata={"categories":cats,"metadata_license":r.get("metadata_license","CC0 metadata; verify source") }))
    prep_seconds=time.perf_counter()-prep_start
    extract_start=time.perf_counter()
    extractor=BatchedGLiNERRelexExtractor(device=args.device,batch_size=args.batch_size,cache_path=args.cache)
    entities, relations=extractor.extract_chunks(chunks)
    extraction_seconds=time.perf_counter()-extract_start
    topics=[Topic(topic_id=v["topic_id"],label=v["label"],keywords=v["keywords"],chunk_ids=v["chunk_ids"]) for v in topics_by_id.values()]
    graph_start=time.perf_counter(); graph=build_graph(chunks,topics,entities,relations); graph_build_seconds=time.perf_counter()-graph_start
    total_pipeline_seconds=time.perf_counter()-all_start
    report={"benchmark":"arXiv metadata subset / GLiNER-ReLEx batch extraction",
            "documents_requested":args.limit,"documents_loaded":len(records),"chunks":len(chunks),
            "topic_count":len(topics),"entity_candidates":len(entities),"relation_candidates":len(relations),
            "graph_nodes":len(graph.nodes),"graph_edges":len(graph.edges),
            "metadata_load_seconds":round(metadata_load_seconds,3),"chunk_and_topic_setup_seconds":round(prep_seconds,3),
            "extraction_seconds_including_model_first_load":round(extraction_seconds,3),
            "graph_build_seconds":round(graph_build_seconds,3),"total_pipeline_seconds":round(total_pipeline_seconds,3),
            "model_diagnostics":extractor.diagnostics(),"quality_metrics":None,
            "quality_note":"No precision/recall is reported without a manually labelled gold set. Use a separate source-grouped test set and core.selective_quality.precision_recall_f1.",
            "topic_note":"arXiv categories are used as topic labels for this scale benchmark; this run does not evaluate unsupervised topic quality.",
            "license_note":"This script expects Kaggle arXiv metadata and uses title+abstract only; each record links to its arXiv page. Do not bulk-copy full papers without checking each paper's license.",
            "hardware_note":"Record machine, Python version, cold/warm cache and whether model weight loading was already complete. Model download is not part of this elapsed time unless the library itself downloads weights during initialization.",
            "scope_note":"Measures metadata loading + chunk/topic setup + extraction + in-memory graph build. It does not represent PDF parsing or production concurrency."}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
