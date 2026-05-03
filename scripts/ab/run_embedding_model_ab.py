#!/usr/bin/env python3
"""Run embedding-model A/B retrieval experiments on a minimal shared corpus.

This runner reuses an existing retrieval comparison CSV to derive a compact
corpus, rebuilds separate vector collections for multiple embedding models, and
executes the same query set across each collection.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge_hub.ai.rag import RAGSearcher
from knowledge_hub.core.chunking import chunk_text_with_offsets, infer_content_type
from knowledge_hub.core.vector_db import VectorDatabase
from knowledge_hub.infrastructure.config import Config
from knowledge_hub.infrastructure.persistence import SQLiteDatabase
from knowledge_hub.interfaces.cli.commands.index_cmd import _get_paper_keywords
from knowledge_hub.providers.registry import get_embedder


DEFAULT_MODEL_SPECS = [
    {
        "label": "openai_small",
        "provider": "openai",
        "model": "text-embedding-3-small",
    },
    {
        "label": "nomic_ollama",
        "provider": "ollama",
        "model": "nomic-embed-text:latest",
    },
    {
        "label": "bge_m3_ollama",
        "provider": "ollama",
        "model": "bge-m3:latest",
    },
    {
        "label": "bge_m3_st",
        "provider": "pplx-st",
        "model": "BAAI/bge-m3",
    },
    {
        "label": "pplx_4b",
        "provider": "pplx-st",
        "model": "perplexity-ai/pplx-embed-v1-4b",
    },
    {
        "label": "gte_qwen2_15b",
        "provider": "pplx-st",
        "model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    },
]


@dataclass(frozen=True)
class CorpusItem:
    doc_id: str
    title: str
    source_type: str
    text: str
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run embedding-model A/B retrieval experiments.")
    parser.add_argument(
        "--baseline-csv",
        default="docs/eval_precision_template.csv",
        help="Existing comparison CSV used to derive the minimal corpus.",
    )
    parser.add_argument(
        "--queries",
        default="docs/eval_queries_ko_20.txt",
        help="Query set to replay against each rebuilt collection.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/ab/embedding_models",
        help="Directory to store manifests, result JSON, and comparison CSV.",
    )
    parser.add_argument(
        "--collection-prefix",
        default="ab_embed",
        help="Prefix for rebuilt Chroma collection names.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["semantic", "hybrid", "keyword"], default="hybrid")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--device", default="mps", help="Device override for local sentence-transformers models")
    parser.add_argument("--batch-size", type=int, default=4, help="Embed batch size override for local models")
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear per-model collections before rebuilding them.",
    )
    parser.add_argument(
        "--reuse-collections",
        action="store_true",
        help="Skip rebuilding a model collection when it already contains documents.",
    )
    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        help="JSON spec: {'label':'name','provider':'openai|ollama|pplx-st','model':'...'}",
    )
    return parser.parse_args()


def _load_queries(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _comparison_variants(rows: list[dict[str, str]]) -> list[str]:
    variants: set[str] = set()
    for row in rows:
        for key in row:
            if key.endswith("_id"):
                variants.add(key[:-3])
    return sorted(variants)


def _parse_doc_key(doc_id: str) -> tuple[str, int]:
    match = re.match(r"^(.*)_(\d+)$", doc_id.strip())
    if not match:
        return doc_id.strip(), 0
    return match.group(1), int(match.group(2))


def _existing_document_lookup(config: Config) -> dict[str, tuple[str, dict[str, Any]]]:
    found: dict[str, tuple[str, dict[str, Any]]] = {}
    known_collections = list(
        dict.fromkeys(
            [
                str(config.collection_name or "").strip(),
                "knowledge_hub",
                "knowledge_hub_pplx_context_060b_v2",
            ]
        )
    )
    for name in known_collections:
        if not name:
            continue
        try:
            db = VectorDatabase(config.vector_db_path, name)
            ids = db.collection.get(include=[]).get("ids", [])
            for doc_id in ids:
                if doc_id in found:
                    continue
                payload = db.collection.get(ids=[doc_id], include=["documents", "metadatas"])
                documents = payload.get("documents") or []
                metadatas = payload.get("metadatas") or []
                if documents:
                    found[str(doc_id)] = (str(documents[0] or ""), dict(metadatas[0] or {}) if metadatas else {})
        except Exception:
            continue
    return found


def _resolve_vault_text(
    *,
    doc_id: str,
    title: str,
    config: Config,
) -> tuple[str, dict[str, Any]] | None:
    vault_root = Path(str(config.vault_path or "")).expanduser()
    if not vault_root.exists():
        return None
    rel_path, chunk_index = _parse_doc_key(doc_id)
    abs_path = vault_root / rel_path
    if not abs_path.exists():
        return None
    try:
        body = abs_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        body = abs_path.read_text(encoding="utf-8", errors="ignore")
    content_type = infer_content_type(text=body, file_path=abs_path)
    chunks = chunk_text_with_offsets(
        body,
        content_type=content_type,
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap,
    )
    if not chunks:
        return None
    selected = chunks[chunk_index] if chunk_index < len(chunks) else chunks[0]
    metadata = {
        "title": title or abs_path.stem,
        "source_type": "vault",
        "file_path": rel_path,
        "chunk_index": int(selected.get("chunk_index", chunk_index)),
        "section_title": str(selected.get("section_title") or ""),
        "section_path": str(selected.get("section_path") or ""),
    }
    return str(selected.get("text") or ""), metadata


def _resolve_paper_text(
    *,
    doc_id: str,
    title: str,
    sqlite_db: SQLiteDatabase,
    keyword_map: dict[str, list[str]],
    config: Config,
) -> tuple[str, dict[str, Any]] | None:
    match = re.match(r"^paper_([^_]+)_(\d+)$", doc_id.strip())
    if not match:
        return None
    paper_id = match.group(1)
    chunk_index = int(match.group(2))
    paper = sqlite_db.get_paper(paper_id)
    if not paper:
        return None
    text_path = str(paper.get("text_path") or "").strip()
    if text_path:
        path = Path(text_path)
        if path.exists():
            try:
                body = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                body = path.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_text_with_offsets(
                body,
                content_type="plain",
                chunk_size=1000,
                overlap=200,
            )
            if chunks:
                selected = chunks[chunk_index] if chunk_index < len(chunks) else chunks[0]
                metadata = {
                    "title": paper.get("title") or title or paper_id,
                    "source_type": "paper",
                    "arxiv_id": paper_id,
                    "field": str(paper.get("field") or ""),
                    "chunk_index": int(selected.get("chunk_index", chunk_index)),
                    "file_path": str(path),
                }
                return str(selected.get("text") or ""), metadata

    keywords = keyword_map.get(paper_id, [])
    text = f"Title: {paper.get('title') or title or paper_id}"
    if keywords:
        text += f"\nKeywords: {', '.join(keywords[:10])}"
    if paper.get("notes"):
        text += f"\n\n{paper['notes']}"
    metadata = {
        "title": paper.get("title") or title or paper_id,
        "source_type": "paper",
        "arxiv_id": paper_id,
        "field": str(paper.get("field") or ""),
        "chunk_index": 0,
    }
    return text, metadata


def _resolve_concept_text(*, title: str, config: Config) -> tuple[str, dict[str, Any]] | None:
    vault_root = Path(str(config.vault_path or "")).expanduser()
    if not vault_root.exists():
        return None
    candidates = [
        vault_root / "AI" / "AI_Papers" / "Concepts" / f"{title}.md",
        vault_root / "Papers" / "Concepts" / f"{title}.md",
        vault_root / "Projects" / "AI" / "AI_Papers" / "Concepts" / f"{title}.md",
        vault_root / "Concepts" / f"{title}.md",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            body = path.read_text(encoding="utf-8", errors="ignore")
        return body, {"title": title, "source_type": "concept", "file_path": str(path)}
    return (title, {"title": title, "source_type": "concept"})


def _derive_corpus(
    *,
    baseline_csv: Path,
    config: Config,
    sqlite_db: SQLiteDatabase,
) -> tuple[list[CorpusItem], dict[str, Any]]:
    rows = list(csv.DictReader(baseline_csv.open("r", encoding="utf-8")))
    variants = _comparison_variants(rows)
    existing_docs = _existing_document_lookup(config)
    keyword_map = _get_paper_keywords(config.vault_path) if config.vault_path else {}
    seen: set[str] = set()
    items: list[CorpusItem] = []
    missing: list[dict[str, Any]] = []

    for row in rows:
        for variant in variants:
            doc_id = str(row.get(f"{variant}_id", "") or "").strip()
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            title = str(row.get(f"{variant}_title", "") or "").strip()
            source_type = str(row.get(f"{variant}_source", "") or "").strip().lower()

            existing = existing_docs.get(doc_id)
            if existing is not None:
                text, metadata = existing
                items.append(
                    CorpusItem(
                        doc_id=doc_id,
                        title=title or str(metadata.get("title") or doc_id),
                        source_type=source_type or str(metadata.get("source_type") or ""),
                        text=text,
                        metadata=dict(metadata),
                    )
                )
                continue

            resolved: tuple[str, dict[str, Any]] | None = None
            if source_type == "vault":
                resolved = _resolve_vault_text(doc_id=doc_id, title=title, config=config)
            elif source_type == "paper":
                resolved = _resolve_paper_text(
                    doc_id=doc_id,
                    title=title,
                    sqlite_db=sqlite_db,
                    keyword_map=keyword_map,
                    config=config,
                )
            elif source_type == "concept":
                resolved = _resolve_concept_text(title=title or doc_id.removeprefix("concept_"), config=config)

            if resolved is None:
                missing.append({"doc_id": doc_id, "title": title, "source_type": source_type})
                continue

            text, metadata = resolved
            items.append(
                CorpusItem(
                    doc_id=doc_id,
                    title=title or str(metadata.get("title") or doc_id),
                    source_type=source_type or str(metadata.get("source_type") or ""),
                    text=text,
                    metadata=dict(metadata),
                )
            )

    manifest = {
        "baselineCsv": str(baseline_csv),
        "variantColumns": variants,
        "documentCount": len(items),
        "missingCount": len(missing),
        "missing": missing[:200],
    }
    return items, manifest


def _model_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.model_spec:
        return list(DEFAULT_MODEL_SPECS)
    specs: list[dict[str, str]] = []
    for raw in args.model_spec:
        payload = json.loads(raw)
        specs.append(
            {
                "label": str(payload["label"]).strip(),
                "provider": str(payload["provider"]).strip(),
                "model": str(payload["model"]).strip(),
            }
        )
    return specs


def _build_embedder(config: Config, spec: dict[str, str], *, device: str, batch_size: int):
    provider = spec["provider"]
    model = spec["model"]
    provider_cfg = dict(config.get_provider_config(provider))
    if provider == "pplx-st":
        provider_cfg["device"] = str(device or "mps")
        provider_cfg["batch_size"] = max(1, int(batch_size))
        provider_cfg.setdefault("torch_num_threads", 1)
        provider_cfg.setdefault("disable_tokenizers_parallelism", True)
    return get_embedder(provider, model=model, **provider_cfg)


def _collection_name(prefix: str, label: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", label).strip("_").lower()
    return f"{prefix}_{safe}"


def _index_collection(
    *,
    config: Config,
    sqlite_db: SQLiteDatabase,
    spec: dict[str, str],
    corpus: list[CorpusItem],
    collection_name: str,
    clear_existing: bool,
    reuse_collections: bool,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    db = VectorDatabase(config.vector_db_path, collection_name)
    if db.count() and reuse_collections:
        return {
            "label": spec["label"],
            "provider": spec["provider"],
            "model": spec["model"],
            "collection": collection_name,
            "status": "reused",
            "documentCount": db.count(),
        }
    if clear_existing or db.count():
        db.clear_collection()

    embedder = _build_embedder(config, spec, device=device, batch_size=batch_size)
    documents = [item.text for item in corpus]
    metadatas = []
    ids = []
    for item in corpus:
        metadata = dict(item.metadata)
        metadata.setdefault("title", item.title)
        metadata.setdefault("source_type", item.source_type)
        metadatas.append(metadata)
        ids.append(item.doc_id)

    started = time.perf_counter()
    embeddings = embedder.embed_batch(documents, show_progress=False)
    elapsed = time.perf_counter() - started
    valid = [
        (doc, emb, meta, doc_id)
        for doc, emb, meta, doc_id in zip(documents, embeddings, metadatas, ids)
        if emb is not None
    ]
    if valid:
        v_docs, v_embs, v_meta, v_ids = zip(*valid)
        db.add_documents(
            documents=list(v_docs),
            embeddings=list(v_embs),
            metadatas=list(v_meta),
            ids=list(v_ids),
        )
    return {
        "label": spec["label"],
        "provider": spec["provider"],
        "model": spec["model"],
        "collection": collection_name,
        "status": "indexed",
        "documentCount": len(valid),
        "embedSec": round(elapsed, 3),
        "failedDocs": len(corpus) - len(valid),
    }


def _run_queries(
    *,
    config: Config,
    sqlite_db: SQLiteDatabase,
    spec: dict[str, str],
    collection_name: str,
    queries: list[str],
    top_k: int,
    mode: str,
    alpha: float,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    db = VectorDatabase(config.vector_db_path, collection_name)
    embedder = _build_embedder(config, spec, device=device, batch_size=batch_size)
    searcher = RAGSearcher(embedder=embedder, database=db, llm=None, sqlite_db=sqlite_db, config=config)
    results_by_query: list[dict[str, Any]] = []
    latencies_ms: list[float] = []

    for query in queries:
        started = time.perf_counter()
        hits = searcher.search(query, top_k=top_k, retrieval_mode=mode, alpha=alpha)
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(latency_ms)
        results_by_query.append(
            {
                "query": query,
                "latencyMs": round(latency_ms, 3),
                "results": [
                    {
                        "rank": idx + 1,
                        "id": str(item.document_id or ""),
                        "title": str((item.metadata or {}).get("title") or ""),
                        "source": str((item.metadata or {}).get("source_type") or ""),
                        "score": round(float(item.score or 0.0), 6),
                    }
                    for idx, item in enumerate(hits)
                ],
            }
        )
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    return {
        "label": spec["label"],
        "provider": spec["provider"],
        "model": spec["model"],
        "collection": collection_name,
        "queryCount": len(queries),
        "avgLatencyMs": round(avg_latency, 3),
        "resultsByQuery": results_by_query,
    }


def _write_comparison_csv(
    *,
    out_path: Path,
    runs: list[dict[str, Any]],
    top_k: int,
) -> None:
    fieldnames = ["query", "rank"]
    labels = [str(run["label"]) for run in runs]
    for label in labels:
        fieldnames.extend(
            [
                f"{label}_id",
                f"{label}_title",
                f"{label}_source",
                f"{label}_score",
                f"{label}_latency_ms",
            ]
        )
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if not runs:
            return
        query_order = [entry["query"] for entry in runs[0]["resultsByQuery"]]
        per_label = {str(run["label"]): {item["query"]: item for item in run["resultsByQuery"]} for run in runs}
        for query in query_order:
            for rank in range(1, top_k + 1):
                row: dict[str, Any] = {"query": query, "rank": rank}
                for label in labels:
                    payload = per_label[label].get(query, {})
                    results = list(payload.get("results") or [])
                    item = results[rank - 1] if rank - 1 < len(results) else {}
                    row[f"{label}_id"] = str(item.get("id") or "")
                    row[f"{label}_title"] = str(item.get("title") or "")
                    row[f"{label}_source"] = str(item.get("source") or "")
                    row[f"{label}_score"] = item.get("score", "")
                    row[f"{label}_latency_ms"] = payload.get("latencyMs", "")
                writer.writerow(row)


def main() -> int:
    args = parse_args()
    config = Config.get()
    sqlite_db = SQLiteDatabase(config.sqlite_path)
    queries = _load_queries(Path(args.queries))
    if not queries:
        raise SystemExit("No queries found")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus, manifest = _derive_corpus(
        baseline_csv=Path(args.baseline_csv),
        config=config,
        sqlite_db=sqlite_db,
    )
    if not corpus:
        raise SystemExit("No corpus items could be resolved from the baseline CSV")

    (output_dir / "corpus_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    specs = _model_specs(args)
    index_runs: list[dict[str, Any]] = []
    query_runs: list[dict[str, Any]] = []
    for spec in specs:
        collection_name = _collection_name(args.collection_prefix, spec["label"])
        index_info = _index_collection(
            config=config,
            sqlite_db=sqlite_db,
            spec=spec,
            corpus=corpus,
            collection_name=collection_name,
            clear_existing=bool(args.clear_existing),
            reuse_collections=bool(args.reuse_collections),
            device=args.device,
            batch_size=args.batch_size,
        )
        index_runs.append(index_info)
        query_info = _run_queries(
            config=config,
            sqlite_db=sqlite_db,
            spec=spec,
            collection_name=collection_name,
            queries=queries,
            top_k=args.top_k,
            mode=args.mode,
            alpha=args.alpha,
            device=args.device,
            batch_size=args.batch_size,
        )
        query_runs.append(query_info)
        (output_dir / f"{spec['label']}_results.json").write_text(
            json.dumps(query_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    _write_comparison_csv(
        out_path=output_dir / "embedding_model_comparison.csv",
        runs=query_runs,
        top_k=args.top_k,
    )
    summary = {
        "queries": str(args.queries),
        "baselineCsv": str(args.baseline_csv),
        "outputDir": str(output_dir),
        "collectionPrefix": args.collection_prefix,
        "corpus": {
            "documentCount": len(corpus),
            "sourceTypes": {
                key: sum(1 for item in corpus if item.source_type == key)
                for key in sorted({item.source_type for item in corpus})
            },
        },
        "indexRuns": index_runs,
        "queryRuns": [
            {
                "label": run["label"],
                "provider": run["provider"],
                "model": run["model"],
                "collection": run["collection"],
                "queryCount": run["queryCount"],
                "avgLatencyMs": run["avgLatencyMs"],
            }
            for run in query_runs
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
