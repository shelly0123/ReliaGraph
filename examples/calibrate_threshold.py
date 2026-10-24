"""Tune a conservative promotion threshold from human-labelled candidate scores.

CSV must contain columns `evidence_quality_score` and `is_correct` (yes/no, true/false,
1/0). Use a calibration set separate from the final test set and group splits by source
or document to avoid near-duplicate leakage.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

from core.selective_quality import choose_conservative_threshold, threshold_curve

TRUE_VALUES = {"1", "true", "yes", "y", "correct", "accept"}
FALSE_VALUES = {"0", "false", "no", "n", "incorrect", "reject"}


def parse_bool(value: str) -> bool:
    normalized = str(value or "").strip().casefold()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"Unrecognized is_correct value: {value!r}; use yes/no or true/false.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Reviewed candidate CSV with scores and correctness labels")
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--min-accepted", type=int, default=8)
    parser.add_argument("--min-labelled", type=int, default=30)
    parser.add_argument("--out", default="examples/outputs/calibrated_promotion_policy.json")
    args = parser.parse_args()
    scores, labels = [], []
    with open(args.csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"evidence_quality_score", "is_correct"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise SystemExit(f"CSV must have columns: {', '.join(sorted(required))}")
        for row in reader:
            if not str(row.get("is_correct", "")).strip():
                continue
            scores.append(float(row["evidence_quality_score"]))
            labels.append(parse_bool(row["is_correct"]))
    policy = choose_conservative_threshold(scores, labels, target_precision=args.target_precision,
                                           min_accepted=args.min_accepted, min_labeled=args.min_labelled)
    policy["calibration_csv"] = str(Path(args.csv_path))
    policy["note"] = ("This policy is only as reliable as the reviewed calibration set. Keep a separate, source-grouped held-out set. "
                      "A dataset shift should disable auto-promotion until re-evaluated.")
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(policy, indent=2))
    if not policy.get("auto_promotion_enabled"):
        print("\nSAFE MODE: automatic promotion is disabled; candidates should remain in review.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
