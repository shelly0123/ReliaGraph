"""Check the Phase 2 requirement: do not lose >3 percentage points of Phase 1 quality.
Input JSONL rows: test_id, reference_answer, phase1_answer, phase2_answer.
Use identical queries/references and the same answer metric before making claims.
"""
import argparse, json
from pathlib import Path
from core.phase2.phase2_evaluation import quality_report

def main():
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--max-drop-points",type=float,default=3.0);p.add_argument("--output",default="examples/outputs/phase1_regression_report.json");a=p.parse_args()
    rows=[]
    for line in Path(a.input).read_text(encoding="utf-8").splitlines():
        if line.strip(): rows.append(json.loads(line))
    report=quality_report(rows,a.max_drop_points)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2))
    if not report["within_phase1_quality_budget"]: raise SystemExit(2)
if __name__=="__main__": main()
