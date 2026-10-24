"""Train a tiny selective-review calibration model from human-reviewed rows.

No domain-specific rules are learned from the MUJ demo. The calibrator learns how generic
provenance/model features correlate with reviewed support labels. The script refuses to
train on small or single-domain CSV files and evaluates on document-grouped holdout.
"""
from __future__ import annotations
import argparse, json
from core.evidence_calibrator import train_from_csv, evaluate_leave_one_domain_out, save_training_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--output", default="examples/outputs/evidence_calibrator.json")
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--min-domains", type=int, default=3)
    parser.add_argument("--leave-one-domain-out", action="store_true",
                        help="Evaluate cross-domain transfer only; does not save a final deployable calibration model")
    args = parser.parse_args()
    if args.leave_one_domain_out:
        result = evaluate_leave_one_domain_out(args.input_csv, min_rows=args.min_rows, min_domains=args.min_domains)
    else:
        result = train_from_csv(args.input_csv, min_rows=args.min_rows, min_domains=args.min_domains)
    save_training_result(result, args.output)
    print(json.dumps({k:v for k,v in result.items() if k != "model"}, indent=2, ensure_ascii=False))
    if args.leave_one_domain_out:
        if not result.get("evaluated"):
            print("SAFE MODE: cross-domain evaluation could not run; candidates remain review-required.")
            return 2
        print("Evaluation only: no deployable model was produced by leave-one-domain-out mode.")
        return 0
    if not result.get("trained"):
        print("SAFE MODE: calibrator was not trained; candidates remain review-required.")
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
