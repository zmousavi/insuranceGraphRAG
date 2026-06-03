"""
evaluation/evaluate.py
======================
Runs all 25 benchmark questions through the retrieval pipeline and saves results.

Usage:
  python evaluation/evaluate.py --mode rag
  python evaluation/evaluate.py --mode graph_rag
  python evaluation/evaluate.py --mode rag --prompt-version v1

Output:
  results/eval_{mode}_{timestamp}.json   — full results with answers + latency
  Printed to stdout as each question completes (so you can watch progress).

Scoring is manual for now — compare actual_answer to expected_answer by eye.
The JSON output is structured for a future LLM-as-judge scoring pass.
"""

import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))  # so "from retrieval.retrieve import retrieve" resolves
GROUND_TRUTH   = ROOT / "evaluation" / "ground_truth.json"
RESULTS_DIR    = ROOT / "results"


def main():
    parser = argparse.ArgumentParser(description="Run benchmark questions through retrieval pipeline.")
    parser.add_argument("--mode", choices=["rag", "graph_rag"], default="rag")
    parser.add_argument("--prompt-version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks retrieved (RAG mode)")
    parser.add_argument("--questions", nargs="+", help="Subset of question IDs to run, e.g. Q01 Q05 Q16")
    args = parser.parse_args()

    # Import here so module-level startup (FAISS, cross-encoder, Neo4j) runs once.
    from retrieval.retrieve import retrieve

    questions = json.loads(GROUND_TRUTH.read_text())

    # Filter to subset if requested.
    if args.questions:
        questions = [q for q in questions if q["id"] in args.questions]
        print(f"Running {len(questions)} selected questions: {[q['id'] for q in questions]}")
    else:
        print(f"Running all {len(questions)} questions in mode={args.mode} prompt={args.prompt_version} top_k={args.top_k}")

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"eval_{args.mode}_k{args.top_k}_{timestamp}.json"

    results = []

    for i, q in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(questions)}] {q['id']} ({q['traversal_type']}, {q['difficulty']})")
        print(f"Q: {q['question']}")
        print(f"Expected: {q['expected_answer']}")

        try:
            result = retrieve(q["question"], mode=args.mode, prompt_version=args.prompt_version, top_k=args.top_k)
            print(f"\nActual:\n{result.answer}")
            print(f"\nLatency: {result.latency_breakdown}")

            record = {
                "id":               q["id"],
                "traversal_type":   q["traversal_type"],
                "difficulty":       q["difficulty"],
                "question":         q["question"],
                "expected_answer":  q["expected_answer"],
                "required_sources": q["required_sources"],
                "actual_answer":    result.answer,
                "mode":             result.mode,
                "latency":          result.latency_breakdown,
                "clusters_used":    result.clusters_used,
                "num_paths":        len(result.supporting_paths),
                "error":            None,
            }
        except Exception as e:
            print(f"\nERROR: {e}")
            record = {
                "id":               q["id"],
                "traversal_type":   q["traversal_type"],
                "difficulty":       q["difficulty"],
                "question":         q["question"],
                "expected_answer":  q["expected_answer"],
                "required_sources": q["required_sources"],
                "actual_answer":    None,
                "mode":             args.mode,
                "latency":          {},
                "clusters_used":    [],
                "num_paths":        0,
                "error":            str(e),
            }

        results.append(record)

        # Save after every question — don't lose work if something fails mid-run.
        output_file.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}")
    print(f"Done. {len(results)} questions answered.")
    errors = [r for r in results if r["error"]]
    if errors:
        print(f"Errors: {len(errors)} — {[e['id'] for e in errors]}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
