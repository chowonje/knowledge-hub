#!/usr/bin/env python3
"""A/B benchmark for pplx context embedding models (0.6b vs 4b).

Decision rule (user-agreed):
- quality gain >= +1% (top1 accuracy)
- time increase <= 1.6x
- no OOM/crash
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from knowledge_hub.providers.pplx_st import PPLXSentenceTransformerEmbedder


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        return v
    return v / norm


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_l2_normalize(a), _l2_normalize(b)))


@dataclass
class BenchResult:
    model: str
    ok: bool
    error: str
    total_sec: float
    corpus_embed_sec: float
    query_embed_sec: float
    top1_accuracy: float
    hit_count: int
    query_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "ok": self.ok,
            "error": self.error,
            "totalSec": round(self.total_sec, 3),
            "corpusEmbedSec": round(self.corpus_embed_sec, 3),
            "queryEmbedSec": round(self.query_embed_sec, 3),
            "top1Accuracy": round(self.top1_accuracy, 6),
            "hitCount": self.hit_count,
            "queryCount": self.query_count,
        }


def _build_benchmark_set() -> tuple[list[str], list[tuple[str, int]]]:
    positives = [
        "Retrieval-augmented generation combines external document retrieval with LLM generation to reduce hallucinations.",
        "Transformer self-attention computes token interactions in parallel and scales with sequence length.",
        "Vector databases store dense embeddings and support nearest-neighbor similarity search.",
        "Fine-tuning updates model weights on task-specific data to specialize behavior.",
        "LoRA adapts large models by training low-rank update matrices instead of full weights.",
        "Quantization reduces model memory footprint by representing weights with lower precision.",
        "Chain-of-thought prompting can improve reasoning accuracy on multi-step tasks.",
        "Agent frameworks use tool-calling loops for plan, act, verify, and writeback flows.",
        "Knowledge graphs represent entities and relations as typed edges with evidence.",
        "Embedding models map text into vectors where semantic similarity corresponds to distance.",
        "GPU batch size strongly influences embedding throughput and memory pressure.",
        "Incremental indexing avoids re-embedding unchanged documents by content hashes.",
        "Schema validation prevents malformed payloads from entering downstream pipelines.",
        "Policy engines classify payloads and block outbound P0 sensitive data.",
        "Event sourcing records immutable changes and reconstructs snapshots by replay.",
        "Contrastive learning pulls positive pairs together and pushes negatives apart.",
        "Cross-encoder reranking improves retrieval precision by scoring query-document pairs jointly.",
        "Top-k retrieval and hybrid search combine lexical and semantic ranking signals.",
        "Context window limits require chunking long documents into manageable segments.",
        "Deduplication by canonical URL and content hash reduces redundant crawl data.",
        "MCP tools expose typed interfaces for agent actions over local knowledge stores.",
        "Batch backoff lowers embedding batch size after OOM to preserve stability.",
        "MPS acceleration on Apple Silicon can speed up local embedding workloads.",
        "CUDA enables GPU acceleration for dense vector encoding on NVIDIA devices.",
        "Grounded answers should cite evidence from retrieved sources instead of pure prior.",
        "Ontology extraction can map free text into normalized concepts and relations.",
        "Pending approval queues are useful for low-confidence relation candidates.",
        "Feature stores transform raw events into reusable, queryable features for apps.",
        "Sparse BM25 retrieval complements embeddings for exact keyword matching.",
        "Hallucination mitigation requires better retrieval coverage and evidence checks.",
    ]
    negatives = [
        "Recipe for sourdough bread fermentation and baking timings.",
        "Travel tips for visiting alpine ski resorts during winter holidays.",
        "History of baroque architecture in central Europe.",
        "Beginner guitar chord transitions and metronome practice methods.",
        "Gardening guide for tomato pest management in humid climates.",
        "Running hydration strategy for marathon race week.",
        "Photography basics about aperture, shutter speed, and ISO triangle.",
        "Home coffee brewing methods including pour-over and espresso shots.",
        "Cycling tire pressure recommendations by terrain and rider weight.",
        "Furniture woodworking joinery techniques and sanding grits.",
    ]

    corpus = positives + negatives

    queries = [
        ("How does RAG reduce hallucination by retrieving sources?", 0),
        ("What does self-attention do in transformers?", 1),
        ("Why use vector DB for semantic search?", 2),
        ("Task specialization via updating model weights", 3),
        ("Low-rank adapters instead of full fine-tuning", 4),
        ("Lower precision weights to cut memory usage", 5),
        ("Reasoning prompts with explicit intermediate steps", 6),
        ("Agent loop with tool calls and verification", 7),
        ("Graph of entities and typed relations with provenance", 8),
        ("Sentence embedding space and semantic distance", 9),
        ("Batch size vs throughput trade-off on GPU", 10),
        ("Skip unchanged docs using content hash", 11),
        ("Reject malformed payloads using schema checks", 12),
        ("Block outbound sensitive data by policy class", 13),
        ("Rebuild current state from immutable events", 14),
        ("Train embeddings with positive and negative pairs", 15),
        ("Use cross encoder to rerank retrieval results", 16),
        ("Combine lexical and dense retrieval signals", 17),
        ("Split long text due context window constraints", 18),
        ("Remove duplicate crawl pages by canonical url", 19),
        ("Expose tools for agent runtime over MCP", 20),
        ("Reduce batch automatically on OOM", 21),
        ("Apple silicon GPU acceleration for embeddings", 22),
        ("NVIDIA CUDA for local embedding acceleration", 23),
        ("Ground responses with retrieved evidence", 24),
        ("Normalize text into ontology concepts and relations", 25),
        ("Queue low confidence relations for review", 26),
        ("Transform events into reusable features", 27),
        ("BM25 exact keyword retrieval", 28),
        ("Improve hallucination by stronger retrieval", 29),
    ]
    return corpus, queries


def run_single_model(
    model: str,
    *,
    device: str,
    batch_size: int,
    torch_num_threads: int,
    max_chars_per_chunk: int,
    chunk_overlap_chars: int,
) -> BenchResult:
    started = time.perf_counter()
    corpus, queries = _build_benchmark_set()
    try:
        embedder = PPLXSentenceTransformerEmbedder(
            model=model,
            device=device,
            batch_size=batch_size,
            torch_num_threads=torch_num_threads,
            max_chars_per_chunk=max_chars_per_chunk,
            chunk_overlap_chars=chunk_overlap_chars,
            auto_batch_backoff=True,
            min_batch_size=1,
        )

        t0 = time.perf_counter()
        corpus_vecs_raw = embedder.embed_batch(corpus, show_progress=False)
        corpus_embed_sec = time.perf_counter() - t0
        corpus_vecs = [
            _l2_normalize(np.asarray(v, dtype=np.float32))
            for v in corpus_vecs_raw
            if v is not None
        ]
        if len(corpus_vecs) != len(corpus):
            raise RuntimeError("corpus embeddings include None values")

        query_texts = [q for q, _ in queries]
        t1 = time.perf_counter()
        query_vecs_raw = embedder.embed_batch(query_texts, show_progress=False)
        query_embed_sec = time.perf_counter() - t1
        query_vecs = [
            _l2_normalize(np.asarray(v, dtype=np.float32))
            for v in query_vecs_raw
            if v is not None
        ]
        if len(query_vecs) != len(queries):
            raise RuntimeError("query embeddings include None values")

        hits = 0
        for vec, (_, expected_idx) in zip(query_vecs, queries):
            sims = [float(np.dot(vec, doc_vec)) for doc_vec in corpus_vecs]
            top_idx = int(np.argmax(sims))
            if top_idx == expected_idx:
                hits += 1

        total_sec = time.perf_counter() - started
        return BenchResult(
            model=model,
            ok=True,
            error="",
            total_sec=total_sec,
            corpus_embed_sec=corpus_embed_sec,
            query_embed_sec=query_embed_sec,
            top1_accuracy=(hits / len(queries)),
            hit_count=hits,
            query_count=len(queries),
        )
    except Exception as error:
        total_sec = time.perf_counter() - started
        return BenchResult(
            model=model,
            ok=False,
            error=str(error),
            total_sec=total_sec,
            corpus_embed_sec=0.0,
            query_embed_sec=0.0,
            top1_accuracy=0.0,
            hit_count=0,
            query_count=len(_build_benchmark_set()[1]),
        )


class _ModelTimeout(Exception):
    pass


def _run_with_timeout(
    model: str,
    *,
    timeout_sec: int,
    device: str,
    batch_size: int,
    torch_num_threads: int,
    max_chars_per_chunk: int,
    chunk_overlap_chars: int,
) -> BenchResult:
    def _handler(_signum, _frame):
        raise _ModelTimeout(f"model_timeout({timeout_sec}s)")

    prev = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(max(1, int(timeout_sec)))
    try:
        return run_single_model(
            model,
            device=device,
            batch_size=batch_size,
            torch_num_threads=torch_num_threads,
            max_chars_per_chunk=max_chars_per_chunk,
            chunk_overlap_chars=chunk_overlap_chars,
        )
    except _ModelTimeout as error:
        return BenchResult(
            model=model,
            ok=False,
            error=str(error),
            total_sec=float(timeout_sec),
            corpus_embed_sec=0.0,
            query_embed_sec=0.0,
            top1_accuracy=0.0,
            hit_count=0,
            query_count=len(_build_benchmark_set()[1]),
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def decide(base: BenchResult, candidate: BenchResult) -> dict[str, Any]:
    if not base.ok:
        return {
            "adopt4b": False,
            "reason": "baseline_failed",
            "qualityGainPct": 0.0,
            "timeFactor": math.inf,
        }
    if not candidate.ok:
        return {
            "adopt4b": False,
            "reason": "candidate_failed",
            "qualityGainPct": 0.0,
            "timeFactor": math.inf,
        }

    base_acc = max(1e-9, base.top1_accuracy)
    gain_pct = ((candidate.top1_accuracy - base.top1_accuracy) / base_acc) * 100.0
    time_factor = candidate.total_sec / max(1e-9, base.total_sec)

    adopt = gain_pct >= 1.0 and time_factor <= 1.6
    reason = "pass" if adopt else "did_not_meet_threshold"
    return {
        "adopt4b": adopt,
        "reason": reason,
        "qualityGainPct": round(gain_pct, 4),
        "timeFactor": round(time_factor, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--max-chars-per-chunk", type=int, default=800)
    parser.add_argument("--chunk-overlap-chars", type=int, default=120)
    parser.add_argument("--model-timeout-sec", type=int, default=300)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    model_06 = "perplexity-ai/pplx-embed-context-v1-0.6b"
    model_4b = "perplexity-ai/pplx-embed-context-v1-4b"

    base = _run_with_timeout(
        model_06,
        timeout_sec=args.model_timeout_sec,
        device=args.device,
        batch_size=args.batch_size,
        torch_num_threads=args.torch_num_threads,
        max_chars_per_chunk=args.max_chars_per_chunk,
        chunk_overlap_chars=args.chunk_overlap_chars,
    )
    cand = _run_with_timeout(
        model_4b,
        timeout_sec=args.model_timeout_sec,
        device=args.device,
        batch_size=args.batch_size,
        torch_num_threads=args.torch_num_threads,
        max_chars_per_chunk=args.max_chars_per_chunk,
        chunk_overlap_chars=args.chunk_overlap_chars,
    )
    decision = decide(base, cand)

    payload = {
        "schema": "knowledge-hub.embedding.abtest.result.v1",
        "device": args.device,
        "batchSize": args.batch_size,
        "torchNumThreads": args.torch_num_threads,
        "models": [base.to_dict(), cand.to_dict()],
        "decision": decision,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
