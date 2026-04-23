#!/usr/bin/env python3
from __future__ import annotations

import argparse

from knowledge_hub.application.eval_gate import export_document_memory_eval_template
from knowledge_hub.infrastructure.persistence import SQLiteDatabase


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a manual evaluation template for labs document-memory search."
    )
    parser.add_argument("--db", default="data/knowledge.db", help="SQLite database path")
    parser.add_argument(
        "--queries",
        default="docs/research/document-memory-eval-queries-v1.txt",
        help="Text file with one query per line",
    )
    parser.add_argument(
        "--out",
        default="docs/experiments/document_memory_eval_template.csv",
        help="Output CSV path",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-k grouped documents per query")
    args = parser.parse_args()

    db = SQLiteDatabase(args.db)
    payload = export_document_memory_eval_template(
        db,
        db_path=args.db,
        queries_path=args.queries,
        out_path=args.out,
        top_k=max(1, int(args.top_k)),
    )
    print(
        f"Wrote document-memory eval template: {payload['outPath']} "
        f"({payload['queryCount']} queries, top{payload['topK']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
