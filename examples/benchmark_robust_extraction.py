"""Cross-domain extraction benchmark that degrades gracefully when optional models are absent.

Uses original, manually labelled examples across four domains. This is a smoke benchmark,
not an industrial generalization claim. Use --enable-gliner/--enable-spacy to opt into local
model attempts; if they fail to load, the generic fallback still runs and records the failure.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from collections import defaultdict
from models import Chunk
from core.robust_extraction import extract_with_fallback
from core.domain_benchmark import evaluate_by_domain
from core.selective_quality import canonical_triple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "examples/data/domain_shift_benchmark.json"
DEFAULT_OUT = ROOT / "examples/outputs/robust_extraction_benchmark.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_DATA), help="Manually labelled cross-domain JSON benchmark")
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--enable-gliner", action="store_true", help="Attempt local GLiNER-ReLEx; can require a large first download")
    parser.add_argument("--enable-spacy", action="store_true", help="Attempt spaCy dependency parser; requires en_core_web_sm")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    cases = []
    chunks = []
    for domain in data["domains"]:
        for case in domain["cases"]:
            if args.max_chunks and len(chunks) >= args.max_chunks:
                break
            row = dict(case)
            row["domain"] = domain["domain"]
            cases.append(row)
            chunks.append(Chunk(chunk_id=str(case["case_id"]), doc_id=str(case["case_id"]), text=str(case["text"]),
                                source_title=f"Original test sentence: {domain['domain']}", source_kind="original_test_fixture"))
        if args.max_chunks and len(chunks) >= args.max_chunks:
            break
    started = time.perf_counter()
    run = extract_with_fallback(chunks, enable_gliner=args.enable_gliner, enable_spacy=args.enable_spacy)
    elapsed = time.perf_counter() - started
    by_case = defaultdict(list)
    for rel in run.relations:
        by_case[rel.chunk_id].append((rel.subject, rel.relation_type, rel.object))
    metrics = evaluate_by_domain(by_case, cases)
    decision_counts = run.diagnostics.get("candidate_decision_counts", {})
    reviewed_candidates = len(run.candidate_decisions)
    promoted_candidates = [d for d in run.candidate_decisions if d["decision"] in {"corroborated_candidate", "calibrated_candidate"}]
    promoted_by_case = defaultdict(list)
    for decision in promoted_candidates:
        promoted_by_case[str(decision.get("chunk_id", ""))].append((decision["subject"], decision["relation_type"], decision["object"]))
    promoted_eval = evaluate_by_domain(promoted_by_case, cases)
    report = {
        "benchmark": data.get("title", "Cross-domain relation extraction"),
        "notice": data.get("notice", "Manually labelled smoke benchmark; not representative of industry-scale data."),
        "cases": len(cases), "domains": sorted({x["domain"] for x in cases}),
        "requested_models": {"gliner": args.enable_gliner, "spacy": args.enable_spacy},
        "runtime_seconds_including_model_load": round(elapsed, 4),
        "extraction_diagnostics": run.diagnostics,
        "metrics_all_candidates": metrics,
        "decision_counts": decision_counts,
        "selective_promotion": {
            "candidate_groups": reviewed_candidates,
            "groups_marked_corroborated_or_calibrated": len(promoted_candidates),
            "promotion_coverage": round(len(promoted_candidates) / max(1, reviewed_candidates), 4),
            "promoted_subset_metrics": promoted_eval["overall"] if promoted_candidates else None,
            "warning": "A promoted/corroborated candidate is still not a verified fact; human verification is required. Zero promotion means the system abstained from auto-promotion rather than demonstrating zero error."
        },
        "quality_warning": "Exact triple metrics are limited by this small, project-authored set and the fallback's English relation-cue coverage. They do not establish domain independence.",
        "next_evaluation": "Use a source-grouped holdout dataset with human-reviewed triples, including negative co-occurrence examples, negation, temporal changes, cross-language cases and unfamiliar relation labels."
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out), "runtime_seconds": round(elapsed, 4),
                      "overall": metrics["overall"], "per_domain": metrics["per_domain"],
                      "fallback_only_or_hybrid": run.diagnostics["mode"],
                      "decision_counts": decision_counts,
                      "selective_promotion": report["selective_promotion"],
                      "warnings": [a for a in run.diagnostics["adapters"] if a["status"] != "available" and a["status"] != "always_available"]}, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
