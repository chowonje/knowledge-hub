#!/usr/bin/env python3
"""Rebuild and compare multiple embedding-model collections on a shared corpus.

This runner keeps the experiment outside the product core:
- reuse an existing source collection when available
- otherwise build a bounded corpus from local vault/paper files
- rebuild one vector collection per embedding model
- run the same query set across each collection
- persist per-query JSON and a wide comparison CSV under runs/ab/
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge_hub.ai.rag import RAGSearcher
from knowledge_hub.core.chunking import chunk_text_with_offsets, infer_content_type
from knowledge_hub.infrastructure.config import Config
from knowledge_hub.infrastructure.persistence import SQLiteDatabase, VectorDatabase
from knowledge_hub.infrastructure.providers import get_embedder

DEFAULT_QUERY_FILE = "docs/eval_queries_ko_20.txt"
DEFAULT_OUTPUT_DIR = "runs/ab/embedding_models"
DEFAULT_VAULT_PREFIXES = (
    "Projects/AI",
    "Projects/ai engineering",
    "Papers",
    "AI",
)
DEFAULT_EXCLUDES = {".obsidian", ".trash", ".local-rag", "node_modules", "templates", ".git", "__pycache__"}
DEFAULT_MODELS = (
    "nomic|ollama|nomic-embed-text:latest",
    "bge_m3_ollama|ollama|bge-m3:latest",
    "pplx06b|pplx-st|perplexity-ai/pplx-embed-v1-0.6b",
    "bge_m3|pplx-st|BAAI/bge-m3",
    "pplx4b|pplx-st|perplexity-ai/pplx-embed-v1-4b",
    "gte_qwen2_1_5b|pplx-st|Alibaba-NLP/gte-Qwen2-1.5B-instruct",
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    provider: str
    model: str
    collection: str


def _slug(value: str) -> str:
    safe = []
    for ch in str(value or "").strip().lower():
        if ch.isalnum():
            safe.append(ch)
        else:
            safe.append("_")
    token = "".join(safe).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or "model"


def _parse_model_spec(raw: str, source_collection: str) -> ModelSpec:
    parts = [part.strip() for part in str(raw or "").split("|")]
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"invalid --model spec: {raw!r}; expected 'label|provider|model'")
    label, provider, model = parts
    return ModelSpec(
        label=label,
        provider=provider.lower(),
        model=model,
        collection=f"{source_collection}_{_slug(label)}",
    )


def _read_queries(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _iter_collection_documents(
    vector_db: VectorDatabase,
    *,
    page_size: int = 200,
) -> list[dict[str, Any]]:
    include = ["documents", "metadatas"]
    offset = 0
    count = int(vector_db.count())
    records: list[dict[str, Any]] = []
    while offset < count:
        payload = vector_db.collection.get(limit=page_size, offset=offset, include=include)
        ids = list(payload.get("ids") or [])
        docs = list(payload.get("documents") or [])
        metas = list(payload.get("metadatas") or [])
        for doc_id, document, metadata in zip(ids, docs, metas):
            records.append(
                {
                    "id": str(doc_id),
                    "document": str(document or ""),
                    "metadata": dict(metadata or {}),
                }
            )
        offset += page_size
    return records


def _eligible_markdown(path: Path) -> bool:
    return path.suffix.lower() == ".md" and not any(token in path.parts for token in DEFAULT_EXCLUDES)


def _collect_filesystem_documents(
    cfg: Config,
    *,
    max_files: int = 0,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    vault_root = Path(str(cfg.vault_path or "")).expanduser()
    file_candidates: list[tuple[str, Path]] = []
    if vault_root.exists():
        for prefix in DEFAULT_VAULT_PREFIXES:
            target = vault_root / prefix
            if not target.exists():
                continue
            for md_path in target.rglob("*.md"):
                if _eligible_markdown(md_path):
                    file_candidates.append(("vault", md_path))

    papers_root = Path(str(cfg.papers_dir or "")).expanduser()
    if papers_root.exists():
        for txt_path in sorted(papers_root.glob("*.txt")):
            file_candidates.append(("paper", txt_path))

    file_candidates.sort(key=lambda item: str(item[1]))
    if max_files > 0:
        file_candidates = file_candidates[:max_files]

    for source_type, file_path in file_candidates:
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if not text.strip():
            continue
        content_type = infer_content_type(text=text, file_path=file_path)
        chunks = chunk_text_with_offsets(
            text,
            content_type=content_type,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        rel_path = str(file_path)
        title = file_path.stem
        for chunk in chunks:
            chunk_index = int(chunk.get("chunk_index", 0))
            raw_id = f"{source_type}:{rel_path}:{chunk_index}"
            doc_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            records.append(
                {
                    "id": doc_id,
                    "document": str(chunk.get("text") or ""),
                    "metadata": {
                        "title": title,
                        "file_path": rel_path,
                        "source_type": source_type,
                        "chunk_index": chunk_index,
                        "section_title": str(chunk.get("section_title") or ""),
                        "section_path": str(chunk.get("section_path") or ""),
                        "content_type": str(chunk.get("content_type") or content_type),
                    },
                }
            )
    return records


def _build_corpus(
    cfg: Config,
    *,
    source_db_path: str,
    source_collection: str,
    max_files: int,
    source_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if source_mode in {"auto", "collection"}:
        source_db = VectorDatabase(source_db_path, source_collection)
        count = source_db.count()
        if count > 0:
            records = _iter_collection_documents(source_db)
            return records, {"source": "collection", "collection": source_collection, "documentCount": len(records)}
        if source_mode == "collection":
            return [], {"source": "collection", "collection": source_collection, "documentCount": 0}

    records = _collect_filesystem_documents(cfg, max_files=max_files)
    return records, {"source": "filesystem", "documentCount": len(records), "maxFiles": max_files}


def _build_embedder(
    cfg: Config,
    *,
    provider: str,
    model: str,
    device: str,
    batch_size: int,
) -> Any:
    provider_cfg = dict(cfg.get_provider_config(provider) or {})
    if provider == "pplx-st":
        provider_cfg.setdefault("device", device)
        provider_cfg.setdefault("batch_size", batch_size)
        provider_cfg.setdefault("torch_num_threads", 1)
        provider_cfg.setdefault("disable_tokenizers_parallelism", True)
        provider_cfg.setdefault("max_chars_per_chunk", 1000)
        provider_cfg.setdefault("chunk_overlap_chars", 200)
    return get_embedder(provider, model=model, **provider_cfg)


def _release_model_memory() -> None:
    gc.collect()
    try:
        import torch

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _rebuild_collection(
    cfg: Config,
    *,
    db_path: str,
    model: ModelSpec,
    records: list[dict[str, Any]],
    device: str,
    batch_size: int,
    rebuild: bool,
) -> dict[str, Any]:
    vector_db = VectorDatabase(db_path, model.collection)
    print(f"[rebuild:start] {model.label} provider={model.provider} model={model.model} docs={len(records)}", flush=True)
    if rebuild:
        vector_db.clear_collection()
    elif vector_db.count() > 0:
        print(f"[rebuild:reuse] {model.label} collection={model.collection} count={vector_db.count()}", flush=True)
        return {"collection": model.collection, "documentCount": vector_db.count(), "reused": True, "elapsedSec": 0.0}

    embedder = _build_embedder(cfg, provider=model.provider, model=model.model, device=device, batch_size=batch_size)
    started = time.perf_counter()
    inserted = 0
    step = max(1, int(batch_size))
    for start in range(0, len(records), step):
        batch = records[start : start + step]
        texts = [str(item["document"]) for item in batch]
        metadatas = [dict(item["metadata"]) for item in batch]
        ids = [str(item["id"]) for item in batch]
        raw_embeddings = embedder.embed_batch(texts, show_progress=False)
        valid_docs: list[str] = []
        valid_embeddings: list[list[float]] = []
        valid_metas: list[dict[str, Any]] = []
        valid_ids: list[str] = []
        for doc_text, embedding, metadata, doc_id in zip(texts, raw_embeddings, metadatas, ids):
            if embedding is None:
                continue
            valid_docs.append(doc_text)
            valid_embeddings.append(embedding)
            valid_metas.append(metadata)
            valid_ids.append(doc_id)
        if valid_docs:
            vector_db.add_documents(
                documents=valid_docs,
                embeddings=valid_embeddings,
                metadatas=valid_metas,
                ids=valid_ids,
            )
            inserted += len(valid_docs)
        if (start // step) % 100 == 0:
            print(
                f"[rebuild:progress] {model.label} batch={(start // step) + 1} inserted={inserted}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    print(f"[rebuild:done] {model.label} inserted={inserted} elapsed={elapsed:.2f}s", flush=True)
    return {
        "collection": model.collection,
        "documentCount": inserted,
        "reused": False,
        "elapsedSec": round(elapsed, 3),
    }


def _run_queries(
    cfg: Config,
    *,
    db_path: str,
    model: ModelSpec,
    queries: list[str],
    output_dir: Path,
    top_k: int,
    mode: str,
    alpha: float,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    sqlite_db = SQLiteDatabase(cfg.sqlite_path)
    vector_db = VectorDatabase(db_path, model.collection)
    embedder = _build_embedder(cfg, provider=model.provider, model=model.model, device=device, batch_size=batch_size)
    searcher = RAGSearcher(embedder, vector_db, llm=None, sqlite_db=sqlite_db, config=cfg)
    model_dir = output_dir / model.label
    model_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    print(f"[search:start] {model.label} queries={len(queries)} collection={model.collection}", flush=True)
    for idx, query in enumerate(queries, start=1):
        started = time.perf_counter()
        results = searcher.search(query, top_k=top_k, retrieval_mode=mode, alpha=alpha)
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)
        payload = {
            "model": model.model,
            "provider": model.provider,
            "collection": model.collection,
            "query": query,
            "topK": top_k,
            "retrievalMode": mode,
            "alpha": alpha,
            "latencyMs": round(latency_ms, 3),
            "results": [
                {
                    "rank": rank,
                    "documentId": item.document_id,
                    "score": round(float(item.score or 0.0), 6),
                    "semanticScore": round(float(item.semantic_score or 0.0), 6),
                    "lexicalScore": round(float(item.lexical_score or 0.0), 6),
                    "title": str((item.metadata or {}).get("title", "")),
                    "sourceType": str((item.metadata or {}).get("source_type", "")),
                    "filePath": str((item.metadata or {}).get("file_path", "")),
                }
                for rank, item in enumerate(results, start=1)
            ],
        }
        (model_dir / f"q{idx:02d}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_rows.append(
            {
                "query": query,
                "top1_id": payload["results"][0]["documentId"] if payload["results"] else "",
                "top1_title": payload["results"][0]["title"] if payload["results"] else "",
                "top1_source": payload["results"][0]["sourceType"] if payload["results"] else "",
                "latency_ms": round(latency_ms, 3),
            }
        )
        print(f"[search:query] {model.label} q{idx:02d} latency_ms={latency_ms:.1f} top1={(payload['results'][0]['title'] if payload['results'] else '')[:80]}", flush=True)
    return {
        "queries": len(queries),
        "avgLatencyMs": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "rows": summary_rows,
    }


def _write_comparison_csv(
    output_path: Path,
    *,
    top_k: int,
    queries: list[str],
    per_model_results: dict[str, list[dict[str, Any]]],
) -> None:
    labels = list(per_model_results.keys())
    fieldnames = ["query", "rank"]
    for label in labels:
        fieldnames.extend(
            [
                f"{label}_id",
                f"{label}_title",
                f"{label}_source",
                f"{label}_score",
                f"label_{label}",
            ]
        )
    fieldnames.append("notes")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for query_index, query in enumerate(queries):
            for rank in range(top_k):
                row: dict[str, Any] = {"query": query, "rank": rank + 1, "notes": ""}
                for label in labels:
                    entries = per_model_results[label][query_index]["results"]
                    item = entries[rank] if rank < len(entries) else {}
                    row[f"{label}_id"] = item.get("documentId", "")
                    row[f"{label}_title"] = item.get("title", "")
                    row[f"{label}_source"] = item.get("sourceType", "")
                    row[f"{label}_score"] = item.get("score", "")
                    row[f"label_{label}"] = ""
                writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-model embedding A/B collection rebuild and retrieval comparison.")
    parser.add_argument("--queries", default=DEFAULT_QUERY_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-db-path", default="")
    parser.add_argument("--target-db-path", default="")
    parser.add_argument("--source-collection", default="")
    parser.add_argument("--model", action="append", default=None, help="label|provider|model")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["semantic", "keyword", "hybrid"], default="hybrid")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--max-files", type=int, default=0, help="filesystem corpus cap when source collection is empty")
    parser.add_argument("--source-mode", choices=["auto", "collection", "filesystem"], default="auto")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    cfg = Config.get()
    source_db_path = str(args.source_db_path or cfg.vector_db_path)
    target_db_path = str(args.target_db_path or cfg.vector_db_path)
    source_collection = str(args.source_collection or cfg.collection_name)
    queries = _read_queries(Path(args.queries))
    if not queries:
        raise SystemExit("no queries found")

    raw_model_specs = list(args.model or DEFAULT_MODELS)
    models = [_parse_model_spec(raw, source_collection) for raw in raw_model_specs]
    records, corpus_meta = _build_corpus(
        cfg,
        source_db_path=source_db_path,
        source_collection=source_collection,
        max_files=max(0, int(args.max_files)),
        source_mode=str(args.source_mode),
    )
    if not records:
        raise SystemExit("no corpus records found from source collection or filesystem fallback")
    print(f"[corpus] source={corpus_meta.get('source')} documents={len(records)}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "queriesPath": str(Path(args.queries)),
        "queryCount": len(queries),
        "sourceDbPath": source_db_path,
        "targetDbPath": target_db_path,
        "sourceCollection": source_collection,
        "corpus": corpus_meta,
        "models": [],
    }

    per_model_payloads: dict[str, list[dict[str, Any]]] = {}
    for model in models:
        rebuild_meta = _rebuild_collection(
            cfg,
            db_path=target_db_path,
            model=model,
            records=records,
            device=str(args.device),
            batch_size=max(1, int(args.batch_size)),
            rebuild=bool(args.rebuild),
        )
        run_meta = _run_queries(
            cfg,
            db_path=target_db_path,
            model=model,
            queries=queries,
            output_dir=output_dir,
            top_k=max(1, int(args.top_k)),
            mode=str(args.mode),
            alpha=float(args.alpha),
            device=str(args.device),
            batch_size=max(1, int(args.batch_size)),
        )
        model_dir = output_dir / model.label
        query_payloads: list[dict[str, Any]] = []
        for idx in range(1, len(queries) + 1):
            payload_path = model_dir / f"q{idx:02d}.json"
            query_payloads.append(json.loads(payload_path.read_text(encoding="utf-8")))
        per_model_payloads[model.label] = query_payloads
        manifest["models"].append(
            {
                "label": model.label,
                "provider": model.provider,
                "model": model.model,
                "collection": model.collection,
                "rebuild": rebuild_meta,
                "queryRun": run_meta,
            }
        )
        _release_model_memory()

    _write_comparison_csv(
        output_dir / "comparison.csv",
        top_k=max(1, int(args.top_k)),
        queries=queries,
        per_model_results=per_model_payloads,
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "outputDir": str(output_dir), "modelCount": len(models), "corpus": corpus_meta}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
