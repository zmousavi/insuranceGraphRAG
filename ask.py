"""
Usage:
  python ask.py "your question"
  python ask.py "your question" --mode rag
  python ask.py "your question" --mode graph_rag
  python ask.py "your question" --mode agent
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from retrieval.retrieve import retrieve

parser = argparse.ArgumentParser()
parser.add_argument("question")
parser.add_argument("--mode", default="agent", choices=["rag", "graph_rag", "agent"])
args = parser.parse_args()

result = retrieve(args.question, mode=args.mode)
print(f"\n=== {args.mode} ===")
print(result.answer)
print(f"\nLatency: {result.latency_breakdown.get('total_ms', 0):,} ms")
if args.mode == "agent":
    print(f"Tools: {[p['tool'] for p in result.supporting_paths]}")
