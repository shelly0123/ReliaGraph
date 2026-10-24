"""Generate an original synthetic multi-domain JSONL corpus for performance smoke tests.

These records are not real research data and must never be used to claim model accuracy.
They exercise the generic JSONL import path and 500+ document performance workflow without
requiring a paid or restricted dataset. For research/accuracy evaluation use labelled sources.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

TEMPLATES = [
    ("computing", "The cache stores recent results for request batch {i}.", "Computing synthetic record {i}"),
    ("computing", "The service requires a valid access token for operation {i}.", "Computing synthetic record {i}"),
    ("energy", "The battery stores chemical energy in test unit {i}.", "Energy synthetic record {i}"),
    ("energy", "The converter converts stored energy into motion in test unit {i}.", "Energy synthetic record {i}"),
    ("ecology", "The root system absorbs water from soil plot {i}.", "Ecology synthetic record {i}"),
    ("ecology", "The wetland supports migratory birds in survey area {i}.", "Ecology synthetic record {i}"),
    ("laboratory_science", "The sensor measures temperature in chamber {i}.", "Science synthetic record {i}"),
    ("laboratory_science", "The membrane protects cell sample {i}.", "Science synthetic record {i}"),
    ("education", "The institute provides digital resources for cohort {i}.", "Education synthetic record {i}"),
    ("education", "The curriculum includes laboratory work for course {i}.", "Education synthetic record {i}"),
    ("business", "The contract requires written approval for case {i}.", "Business synthetic record {i}"),
    ("business", "The database contains transaction records for batch {i}.", "Business synthetic record {i}"),
    ("agriculture", "The irrigation system distributes water to field {i}.", "Agriculture synthetic record {i}"),
    ("transport", "The route connects station {i} with the regional network.", "Transport synthetic record {i}"),
    ("research", "The research group publishes findings for project {i}.", "Research synthetic record {i}"),
    ("public_services", "The application records service requests from region {i}.", "Public services synthetic record {i}"),
]

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--docs", type=int, default=600)
    p.add_argument("--output", default="examples/data/synthetic_cross_domain_600.jsonl")
    args = p.parse_args()
    count = max(1, int(args.docs))
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i in range(count):
            domain, template, title = TEMPLATES[i % len(TEMPLATES)]
            # Each record has one short source sentence by design. This is a throughput
            # smoke corpus; repeated templates make it unsuitable for accuracy claims.
            row = {"doc_id": f"synthetic-{i:06d}", "title": title.format(i=i),
                   "text": template.format(i=i), "categories": [domain],
                   "source_kind": "original_synthetic_performance_fixture"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out), "documents": count,
                      "domains": sorted({row[0] for row in TEMPLATES}),
                      "purpose": "runtime/scale smoke test only; not a real-world accuracy dataset"}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
