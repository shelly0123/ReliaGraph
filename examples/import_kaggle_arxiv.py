"""Stream 500+ records from Kaggle's arXiv metadata JSONL into original ReliaGraph JSONL.

Input is the local file downloaded by the user from Kaggle. This script uses metadata
(title, abstract, category, identifier) only; it does NOT copy the full PDF or source files.
ArXiv metadata dataset is marked CC0 on Kaggle, but individual paper/full-text licences vary.
Check the source record/licence before copying full text into a public repository.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

DEFAULT_CATEGORIES = {"cs.AI", "cs.CL", "cs.IR", "cs.LG", "cs.CY", "cs.HC"}

def parse_categories(raw) -> set[str]:
    if not raw: return set(DEFAULT_CATEGORIES)
    return set(raw)

def convert_record(r: dict) -> dict | None:
    arxiv_id = str(r.get("id", "")).strip()
    title = re.sub(r"\s+", " ", str(r.get("title", ""))).strip()
    abstract = re.sub(r"\s+", " ", str(r.get("abstract", ""))).strip()
    cats = str(r.get("categories", "")).split()
    if not arxiv_id or not title or not abstract: return None
    url = f"https://arxiv.org/abs/{arxiv_id}"
    return {"doc_id": f"arxiv:{arxiv_id}", "title": title, "text": f"{title}. {abstract}",
            "source_url": url, "source_title": title, "categories": cats,
            "source_kind": "arxiv_metadata", "metadata_license": "CC0 metadata (per Kaggle dataset page)",
            "full_text_included": False}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to arxiv-metadata-oai-snapshot.json downloaded from Kaggle")
    parser.add_argument("--output", default="data/phase2/arxiv_500.jsonl")
    parser.add_argument("--max-docs", type=int, default=600)
    parser.add_argument("--max-scan", type=int, default=2_000_000)
    parser.add_argument("--categories", nargs="*", default=sorted(DEFAULT_CATEGORIES))
    parser.add_argument("--all-categories", action="store_true")
    args = parser.parse_args()
    input_path, output_path = Path(args.input), Path(args.output)
    if not input_path.exists(): raise SystemExit(f"Input file not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wanted = set() if args.all_categories else parse_categories(args.categories)
    count = scanned = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            scanned += 1
            if scanned > args.max_scan: break
            try: r = json.loads(line)
            except json.JSONDecodeError: continue
            cats = set(str(r.get("categories", "")).split())
            if wanted and not cats.intersection(wanted): continue
            item = convert_record(r)
            if item is None: continue
            dst.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
            if count >= args.max_docs: break
    print(json.dumps({"input": str(input_path), "output": str(output_path), "documents_written": count,
                      "records_scanned": scanned, "categories": sorted(wanted) if wanted else "all",
                      "content_type": "title+abstract metadata only; not full-text papers",
                      "note": "If fewer than 500 were written, increase --max-scan or relax --categories."}, indent=2))
    if count < 500: print("WARNING: fewer than 500 documents were written; increase max-scan or relax category filter.", file=sys.stderr)
if __name__ == "__main__": main()
