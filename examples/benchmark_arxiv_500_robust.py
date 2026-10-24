"""Generalized 500+ record extraction/graph benchmark for JSONL document corpora.

Input records need `doc_id` and `text`; optional fields: title, source_url, categories.
The Kaggle arXiv importer produces this shape, but the code is not arXiv-specific.
It reports performance and graph size; no accuracy is claimed without gold labels.
"""
from __future__ import annotations
import argparse, json, time, platform, os
from pathlib import Path
from models import Chunk, Topic
from core.robust_extraction import extract_with_fallback
from core.graph_builder import build_graph, export_graphml

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try: row = json.loads(line)
            except json.JSONDecodeError: continue
            if row.get("doc_id") and str(row.get("text", "")).strip(): rows.append(row)
            if len(rows) >= limit: break
    return rows


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--input", default="data/phase2/arxiv_500.jsonl")
    p.add_argument("--limit", type=int, default=600)
    p.add_argument("--enable-gliner", action="store_true")
    p.add_argument("--enable-spacy", action="store_true")
    p.add_argument("--output", default="examples/outputs/generalized_600_report.json")
    p.add_argument("--graphml", default="examples/outputs/generalized_600.graphml")
    a=p.parse_args(); total0=time.perf_counter()
    source=Path(a.input)
    if not source.exists():
        raise SystemExit(f"Input not found: {source}. For arXiv metadata, first run examples/import_kaggle_arxiv.py with your local Kaggle JSONL file.")
    t=time.perf_counter(); rows=load_jsonl(source,a.limit); read_s=time.perf_counter()-t
    if len(rows)<500: print(f"WARNING: loaded only {len(rows)} records; the 500+ requirement is not yet met.")
    topics_map: dict[str, dict] = {}
    chunks=[]
    for row in rows:
        did=str(row["doc_id"]); txt=str(row["text"]); cats=row.get("categories") or []
        topic_ids=[]
        for cat in cats:
            tid="topic:"+str(cat)
            topic_ids.append(tid)
            topics_map.setdefault(tid,{"topic_id":tid,"label":str(cat),"keywords":[str(cat)],"chunk_ids":[]})["chunk_ids"].append(did+"::chunk0")
        chunks.append(Chunk(chunk_id=did+"::chunk0", doc_id=did, text=txt, topic_ids=topic_ids,
                            source_url=row.get("source_url"), source_title=row.get("title"),
                            source_kind=str(row.get("source_kind", "uploaded_text")), metadata={"categories":cats}))
    topics=[Topic(topic_id=v["topic_id"],label=v["label"],keywords=v["keywords"],chunk_ids=v["chunk_ids"]) for v in topics_map.values()]
    t=time.perf_counter(); result=extract_with_fallback(chunks, enable_gliner=a.enable_gliner, enable_spacy=a.enable_spacy); extraction_s=time.perf_counter()-t
    t=time.perf_counter(); graph=build_graph(result.safe_chunks,topics,result.entities,result.relations, candidate_decisions=result.candidate_decisions); graph_s=time.perf_counter()-t
    graphml=Path(a.graphml); graphml.parent.mkdir(parents=True,exist_ok=True); export_graphml(graph,str(graphml))
    serialized={"topics":[vars(x) for x in graph.topics],"nodes":[vars(x) for x in graph.nodes],"edges":[vars(x) for x in graph.edges],"document_hashes":graph.document_hashes}
    edge_status_counts = {}
    for edge in graph.edges:
        edge_status_counts[edge.status] = edge_status_counts.get(edge.status, 0) + 1
    graph_bytes=len(json.dumps(serialized,ensure_ascii=False,default=str).encode("utf-8"))
    report={"benchmark":"Generalized JSONL document ingestion + extraction + graph build",
        "input_file":str(source),"documents_loaded":len(rows),"chunks":len(chunks),"topics":len(topics),
        "entities":len(result.entities),"relation_candidates":len(result.relations),"graph_nodes":len(graph.nodes),"graph_edges":len(graph.edges),
        "graph_edge_status_counts":edge_status_counts,
        "quality_gate": {"applied": graph.metadata.get("quality_gate_applied", False),
                         "relation_candidates_rejected": graph.metadata.get("relation_candidates_rejected", 0),
                         "unverified_relation_edges": graph.metadata.get("unverified_relation_edges", 0),
                         "note": "corroborated_candidate remains unverified and requires human review"},
        "graph_serialized_estimate_bytes":graph_bytes,"graphml_path":str(graphml),
        "timings_seconds":{"read_records":round(read_s,4),"extract_including_optional_model_cold_load":round(extraction_s,4),
                           "graph_build":round(graph_s,4),"total":round(time.perf_counter()-total0,4)},
        "environment":{"python":platform.python_version(),"os":platform.system(),"machine":platform.machine(),"cpu_count":os.cpu_count()},
        "diagnostics":result.diagnostics,
        "accuracy":{"reported":False,"reason":"The corpus lacks manually labelled gold relation triples. This is a scale/runtime report, not a quality score."},
        "scope_note":"Categories are reused as topics only when the input supplies them. This benchmark does not measure unsupervised topic quality or full PDF parsing; it measures records supplied as text.",
        "licensing_note":"Use only corpus fields permitted by the source licence; arXiv metadata and a full paper do not necessarily share the same licence."}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 0

if __name__ == "__main__": raise SystemExit(main())
