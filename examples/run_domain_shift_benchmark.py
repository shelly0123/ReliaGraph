"""Run the same domain-agnostic extractor on four different text domains.

Example (zero-shot model; downloads weights on first use):
  python -m examples.run_domain_shift_benchmark --backend gliner

For parser-discovered predicate labels (requires en_core_web_sm):
  python -m examples.run_domain_shift_benchmark --backend both

The benchmark corpus is original and manually labelled. It is a smoke benchmark,
not a representative industrial benchmark or a guarantee of generalization.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from collections import defaultdict

from models import Chunk
from core.robust_extraction import extract_with_fallback
from core.domain_benchmark import evaluate_by_domain
from core.selective_quality import schema_shift_profile

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "examples" / "data" / "domain_shift_benchmark.json"
OUT_PATH = ROOT / "examples" / "outputs" / "domain_shift_benchmark_report.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["gliner", "spacy", "both"], default="gliner")
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all cases")
    args = parser.parse_args()
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    cases = []
    chunks = []
    for domain in data["domains"]:
        for case in domain["cases"]:
            if args.max_cases and len(cases) >= args.max_cases:
                break
            normalized = {**case, "domain": domain["domain"]}
            cases.append(normalized)
            chunks.append(Chunk(chunk_id=case["case_id"], doc_id=case["case_id"], text=case["text"],
                                source_title=f"Original test sentence — {domain['domain']}",
                                source_kind="original_benchmark_sentence"))
        if args.max_cases and len(cases) >= args.max_cases:
            break

    start = time.perf_counter()
    result = extract_with_fallback(chunks, enable_spacy=args.backend in {"spacy", "both"},
                                   enable_gliner=args.backend in {"gliner", "both"})
    elapsed = time.perf_counter() - start
    predictions_by_case: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for rel in result.relations:
        predictions_by_case[rel.chunk_id].append((rel.subject, rel.relation_type, rel.object))
    metrics = evaluate_by_domain(predictions_by_case, cases)
    decision_counts = defaultdict(int)
    for decision in result.candidate_decisions:
        decision_counts[str(decision.get("decision", "unknown"))] += 1
    # Compare candidate predicate vocabulary across domains to surface shift in the
    # extractor's output, not as a direct measure of semantic correctness.
    domain_profiles = {}
    relations_by_domain = defaultdict(list)
    chunk_domain = {c["case_id"]: c["domain"] for c in cases}
    for relation in result.relations:
        relations_by_domain[chunk_domain.get(relation.chunk_id, "unknown")].append(relation)
    domain_names = sorted({c["domain"] for c in cases})
    for i, domain_name in enumerate(domain_names):
        reference = [r for prior in domain_names[:i] for r in relations_by_domain[prior]]
        domain_profiles[domain_name] = schema_shift_profile(reference, relations_by_domain[domain_name]) if reference else {
            "relation_label_jaccard": None, "new_relation_labels": sorted({r.relation_type.casefold() for r in relations_by_domain[domain_name]}),
            "warnings": [], "interpretation": "First domain in this run; used as an initial reference, not a gold standard."}
    report = {
        "benchmark_title": data["title"], "notice": data["notice"], "backend": args.backend,
        "elapsed_seconds_including_model_load": round(elapsed, 4),
        "case_count": len(cases), "entity_candidate_count": len(result.entities),
        "relation_candidate_count": len(result.relations), "candidate_decision_counts": dict(decision_counts),
        "extraction_diagnostics": result.diagnostics, "metrics": metrics, "domain_shift_profiles": domain_profiles,
        "interpretation": [
            "Use the per-domain metrics; do not rely only on the pooled score.",
            "This is a small project-authored smoke benchmark, not independent external validation. It measures only these sentences and settings.",
            "If an optional model or parser is unavailable, its error is recorded and the deterministic fallback runs; inspect extraction_diagnostics.adapters.",
            "A candidate that passes source alignment is grounded in a passage, not necessarily semantically true.",
            "No result from this dataset alone proves generalization to arbitrary domains; validate on held-out documents and an external cross-domain benchmark.",
        ]
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(OUT_PATH), "backend": args.backend, "elapsed_seconds_including_model_load": round(elapsed, 4),
                      "metrics": metrics["overall"], "per_domain": metrics["per_domain"],
                      "candidate_decision_counts": dict(decision_counts)}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
