#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from knowledge_hub.ai.rag import RAGSearcher
from knowledge_hub.core.config import Config
from knowledge_hub.core.database import VectorDatabase
from knowledge_hub.providers.pplx_st import PPLXSentenceTransformerEmbedder


def _read_queries(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _build_searcher(cfg: Config, collection: str, model: str, device: str, batch_size: int) -> RAGSearcher:
    db = VectorDatabase(cfg.vector_db_path, collection)
    embedder = PPLXSentenceTransformerEmbedder(
        model=model,
        device=device,
        batch_size=batch_size,
        torch_num_threads=1,
        disable_tokenizers_parallelism=True,
        max_chars_per_chunk=1000,
        chunk_overlap_chars=200,
    )
    return RAGSearcher(embedder, db, llm=None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate manual Precision@K labeling template for two embedding collections.")
    parser.add_argument("--queries", default="docs/eval_queries_ko_20.txt")
    parser.add_argument("--out", default="docs/eval_precision_template.csv")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=["semantic", "hybrid", "keyword"], default="hybrid")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--a-collection", default="knowledge_hub_pplx_060b")
    parser.add_argument("--a-model", default="perplexity-ai/pplx-embed-v1-0.6b")
    parser.add_argument("--a-label", default="v1")
    parser.add_argument("--b-collection", default="knowledge_hub_pplx_context_060b")
    parser.add_argument("--b-model", default="perplexity-ai/pplx-embed-context-v1-0.6b")
    parser.add_argument("--b-label", default="context")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    cfg = Config.get()
    queries = _read_queries(Path(args.queries))
    if not queries:
        raise SystemExit("No queries found")

    search_a = _build_searcher(cfg, args.a_collection, args.a_model, args.device, args.batch_size)
    search_b = _build_searcher(cfg, args.b_collection, args.b_model, args.device, args.batch_size)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query",
                "rank",
                f"{args.a_label}_id",
                f"{args.a_label}_title",
                f"{args.a_label}_source",
                f"{args.a_label}_score",
                f"{args.b_label}_id",
                f"{args.b_label}_title",
                f"{args.b_label}_source",
                f"{args.b_label}_score",
                f"label_{args.a_label}",
                f"label_{args.b_label}",
                "notes",
            ],
        )
        writer.writeheader()

        for q in queries:
            ra = search_a.search(q, top_k=args.k, retrieval_mode=args.mode, alpha=args.alpha)
            rb = search_b.search(q, top_k=args.k, retrieval_mode=args.mode, alpha=args.alpha)

            for i in range(args.k):
                a = ra[i] if i < len(ra) else None
                b = rb[i] if i < len(rb) else None
                writer.writerow(
                    {
                        "query": q,
                        "rank": i + 1,
                        f"{args.a_label}_id": a.document_id if a else "",
                        f"{args.a_label}_title": (a.metadata.get("title", "") if a else ""),
                        f"{args.a_label}_source": (a.metadata.get("source_type", "") if a else ""),
                        f"{args.a_label}_score": (f"{a.score:.4f}" if a else ""),
                        f"{args.b_label}_id": b.document_id if b else "",
                        f"{args.b_label}_title": (b.metadata.get("title", "") if b else ""),
                        f"{args.b_label}_source": (b.metadata.get("source_type", "") if b else ""),
                        f"{args.b_label}_score": (f"{b.score:.4f}" if b else ""),
                        f"label_{args.a_label}": "",
                        f"label_{args.b_label}": "",
                        "notes": "",
                    }
                )

    print(f"Wrote template: {out_path} ({len(queries)} queries x top{args.k})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
