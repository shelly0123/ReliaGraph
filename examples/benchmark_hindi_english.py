"""Optional Hindi-query / English-evidence retrieval benchmark.

All question/passage pairs are originally written for this project. This benchmark is
small and synthetic; it is a smoke test for wiring and local model behavior, not a
claim about real-world multilingual retrieval quality. Requires optional E5 weights.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from core.phase2.crosslingual_retrieval import ResilientMultilingualRetriever, supporting_graph_paths
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    {"id":"hi_01","query":"पुस्तकालय के डिजिटल संसाधनों में क्या शामिल है?","gold":["c1","c2"]},
    {"id":"hi_02","query":"प्रमाणीकरण सेवा के लिए क्या आवश्यक है?","gold":["c3"]},
    {"id":"hi_03","query":"आर्द्रभूमि किन पक्षियों को सहारा देती है?","gold":["c4"]},
    {"id":"hi_04","query":"सेंसर क्या मापता है?","gold":["c5"]},
    {"id":"hi_05","query":"अनुप्रयोग किस डेटाबेस का उपयोग करता है?","gold":["c6"]},
    {"id":"hi_06","query":"प्रयोग क्या उत्पन्न करता है?","gold":["c7"]},
    {"id":"hi_07","query":"जड़ प्रणाली क्या अवशोषित करती है?","gold":["c8"]},
    {"id":"hi_08","query":"समय-सारणी में क्या होता है?","gold":["c9"]},
    {"id":"hi_09","query":"कैश में क्या रखा जाता है?","gold":["c10"]},
    {"id":"hi_10","query":"झिल्ली कोशिका के लिए क्या करती है?","gold":["c11"]},
]
CHUNKS = [
    {"chunk_id":"c1","doc_id":"d1","text":"The university library provides digital resources to students.","source_title":"Original benchmark sentence"},
    {"chunk_id":"c2","doc_id":"d1","text":"The digital resources include electronic books and research journals.","source_title":"Original benchmark sentence"},
    {"chunk_id":"c3","doc_id":"d2","text":"The authentication service requires a valid access token.","source_title":"Original benchmark sentence"},
    {"chunk_id":"c4","doc_id":"d3","text":"The wetland supports migratory birds during seasonal movement.","source_title":"Original benchmark sentence"},
    {"chunk_id":"c5","doc_id":"d4","text":"The sensor measures temperature in the laboratory.","source_title":"Original benchmark sentence"},
    {"chunk_id":"c6","doc_id":"d5","text":"The application uses a local database to store records.","source_title":"Original benchmark sentence"},
    {"chunk_id":"c7","doc_id":"d6","text":"The experiment produces heat under the stated conditions.","source_title":"Original benchmark sentence"},
    {"chunk_id":"c8","doc_id":"d7","text":"The root system absorbs water from the surrounding soil.","source_title":"Original benchmark sentence"},
    {"chunk_id":"c9","doc_id":"d8","text":"The examination timetable contains examination dates and times.","source_title":"Original benchmark sentence"},
    {"chunk_id":"c10","doc_id":"d9","text":"The cache stores recent results to speed up repeated requests.","source_title":"Original benchmark sentence"},
    {"chunk_id":"c11","doc_id":"d10","text":"The membrane protects the cell from selected environmental changes.","source_title":"Original benchmark sentence"},
    {"chunk_id":"c12","doc_id":"d11","text":"The research report describes a separate procedure for updating records.","source_title":"Original benchmark sentence"},
]

def make_path_graph():
    g=nx.MultiDiGraph()
    g.add_node("library",label="University library")
    g.add_node("resources",label="Digital resources")
    g.add_node("journals",label="Research journals")
    g.add_edge("library","resources",key="rel1",claim_id="rel1",relation_type="provides",evidence_chunk_ids=["c1"],
                evidence=[{"doc_id":"d1","chunk_id":"c1","evidence_text":CHUNKS[0]["text"],"source_title":"Original benchmark sentence"}])
    g.add_edge("resources","journals",key="rel2",claim_id="rel2",relation_type="include",evidence_chunk_ids=["c2"],
                evidence=[{"doc_id":"d1","chunk_id":"c2","evidence_text":CHUNKS[1]["text"],"source_title":"Original benchmark sentence"}])
    return g

def main():
    out_path=ROOT/"examples"/"outputs"/"hindi_english_retrieval_report.json"
    retriever=ResilientMultilingualRetriever(cache_dir="data/cache/hindi_english_benchmark")
    results=[]; all_ranks=[]; path_graph=make_path_graph(); path_rows=[]
    t0=time.perf_counter()
    for case in CASES:
        hits=retriever.search(case["query"],CHUNKS,top_k=3)
        adapter_diag = retriever.diagnostics()
        retrieval_mode = adapter_diag["mode"]
        hit_ids=[h.chunk_id for h in hits]
        ranks=[ix+1 for ix,cid in enumerate(hit_ids) if cid in case["gold"]]
        all_ranks.append(min(ranks) if ranks else None)
        paths=supporting_graph_paths(path_graph,hit_ids,max_paths=3,max_hops=3) if case["id"]=="hi_01" else []
        if paths: path_rows.extend(paths)
        results.append({"id":case["id"],"query_hi":case["query"],"gold_chunk_ids":case["gold"],
                       "retrieved_chunk_ids":hit_ids,"retrieval_mode":retrieval_mode,
                       "top1_hit":bool(hit_ids and hit_ids[0] in case["gold"]) if retrieval_mode == "multilingual_model" else None,
                       "first_gold_rank":min(ranks) if ranks and retrieval_mode == "multilingual_model" else None,
                       "top_k_evidence":[h.to_dict() for h in hits],"graph_paths":paths})
    elapsed=time.perf_counter()-t0
    all_model = all(row["retrieval_mode"] == "multilingual_model" for row in results)
    hit_metric = round(sum(bool(r["top1_hit"]) for r in results)/len(results),4) if all_model else None
    mrr_metric = round(sum((1/r if r else 0.0) for r in all_ranks)/len(all_ranks),4) if all_model else None
    report={"benchmark":"Original Hindi queries over English passages",
        "notice":"Small manually labelled synthetic smoke set, not a general multilingual accuracy guarantee.",
        "cases":len(CASES),"model_evaluation_complete":all_model,
        "evaluation_status":"measured_model_run" if all_model else "not_measured_model_unavailable_or_fallback_used",
        "hit_at_1":hit_metric,"mean_reciprocal_rank":mrr_metric,
        "elapsed_seconds_including_model_load":round(elapsed,4),"model_diagnostics":retriever.diagnostics(),
        "path_demo_found":bool(path_rows),"path_demo":path_rows[:2],"results":results,
        "next_step":"Run after at least one multilingual model loads successfully, then evaluate 30+ manually checked Hindi/English pairs. When only lexical fallback is active, metrics are intentionally null rather than misreporting same-language overlap as cross-lingual performance."}
    out_path.parent.mkdir(parents=True,exist_ok=True);out_path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ["benchmark","cases","model_evaluation_complete","evaluation_status","hit_at_1","mean_reciprocal_rank","elapsed_seconds_including_model_load","model_diagnostics","path_demo_found","next_step"]},indent=2,ensure_ascii=False))
    print("Full report:",out_path)
if __name__=="__main__": main()
