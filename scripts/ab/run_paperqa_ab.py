#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from uuid import UUID

from paperqa import Docs
from paperqa.llms import OpenAILLMModel


QUERY_SET = {
    "q06": "What are the main contributions of this paper?",
    "q07": "What are the main limitations of this paper?",
    "q08": "How does this paper differ from prior approaches?",
    "q09": "How is the proposed method evaluated in this paper?",
    "q10": "What follow-up research is discussed or implied by this paper?",
}


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (Path, UUID)):
        return str(value)
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)


def run(args: argparse.Namespace) -> None:
    source_path = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    index_path = Path(args.index_path).expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("OPENAI_API_KEY", args.api_key)
    os.environ.setdefault("OPENAI_BASE_URL", args.base_url)

    llm = OpenAILLMModel(
        name=args.model,
        config={"model": args.model, "temperature": args.temperature},
    )
    docs = Docs(
        llm_model=llm,
        summary_llm_model=llm,
        embedding=args.embedding_model,
        index_path=index_path,
    )
    added = docs.add(source_path, citation=args.citation)

    for query_id, question in QUERY_SET.items():
        answer = docs.query(
            question,
            k=args.k,
            max_sources=args.max_sources,
            length_prompt=args.answer_length,
        )
        payload = {
            "system": "paperqa",
            "source_path": str(source_path),
            "citation": args.citation,
            "added_docname": added,
            "query_id": query_id,
            "question": question,
            "config": {
                "model": args.model,
                "embedding_model": args.embedding_model,
                "base_url": args.base_url,
                "k": args.k,
                "max_sources": args.max_sources,
                "answer_length": args.answer_length,
            },
            "answer": _to_jsonable(answer),
        }
        output_path = output_dir / f"paperqa_{query_id}.json"
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the PaperQA q06-q10 A/B query set against a single source."
    )
    parser.add_argument("--source", required=True, help="Path to the paper source file")
    parser.add_argument(
        "--citation",
        required=True,
        help="Citation label to attach when adding the source to PaperQA",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where per-query JSON outputs will be written",
    )
    parser.add_argument(
        "--index-path",
        required=True,
        help="PaperQA index directory for this run",
    )
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-sources", type=int, default=3)
    parser.add_argument("--answer-length", default="about 120 words")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
