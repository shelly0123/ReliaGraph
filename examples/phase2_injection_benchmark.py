"""Evaluate known injection strings and verify the model-facing context is screened.

This benchmark measures a small, original known-case suite. Passing it is not proof that
unseen prompt injections are blocked; document content remains untrusted even after scan.
"""
from __future__ import annotations
import json
from pathlib import Path
from core.phase2.injection_guard import scan_and_sanitize, build_untrusted_context
from core.phase2.phase2_evaluation import injection_metrics

ROOT = Path(__file__).resolve().parents[1]

def main():
    path=ROOT/"examples"/"data"/"phase2_prompt_injection_cases.json"
    data=json.loads(path.read_text(encoding="utf-8"))
    detections=[]
    for case in data["attacks"]:
        chunk={"chunk_id":case["id"],"doc_id":"planted-attack-corpus","text":case["text"],
               "source_title":"Original synthetic injection test case"}
        result=scan_and_sanitize(case["text"])
        context=build_untrusted_context([chunk])
        safe_context=context["safe_context"].casefold()
        excerpts=[str(f.get("excerpt","")) for f in context["findings"] if f.get("excerpt")]
        excerpt_residual=any(e.casefold() in safe_context for e in excerpts if len(e.strip())>=12)
        # A detected attack is considered blocked in the test harness only if no matched
        # instruction excerpt remains in model-facing context. Unseen attacks remain unknown.
        blocked=bool(result.findings) and not excerpt_residual
        detections.append({"id":case["id"],"detected":bool(result.findings),"blocked_from_context":blocked,
                           "excerpt_residual_in_context":excerpt_residual,"quarantined":context["quarantined_chunk_ids"],
                           "findings":[f.to_dict() for f in result.findings],
                           "model_context_nonempty":bool(context["safe_context"])})
    benign=[]
    for case in data["benign"]:
        result=scan_and_sanitize(case["text"])
        benign.append({"id":case["id"],"flagged":bool(result.findings),"findings":[f.to_dict() for f in result.findings]})
    metrics=injection_metrics(len(data["attacks"]),sum(x["detected"] for x in detections),
                              len(data["benign"]),sum(x["flagged"] for x in benign))
    metrics["blocked_from_model_context"] = sum(x["blocked_from_context"] for x in detections)
    metrics["model_context_residual_count"] = sum(x["excerpt_residual_in_context"] for x in detections)
    metrics["note"] += " The `blocked_from_model_context` field applies only to these crafted examples; it does not prove immunity to unknown attack patterns or model-specific prompt injection."
    report={"notice":data["notice"],"metrics":metrics,"attacks":detections,"benign":benign}
    out=ROOT/"examples"/"outputs"/"injection_benchmark_report.json"
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(metrics,indent=2));print(f"Full report: {out}")
if __name__ == "__main__": main()
