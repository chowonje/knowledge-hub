#!/usr/bin/env python3
"""Run the Khoj q01-q05 vault retrieval A/B on a minimal shared corpus.

This runner reuses the existing `knowledge_hub_q01.json` ~ `q05.json` baselines
to derive a compact comparison corpus, uploads those markdown files to a running
Khoj server, and stores Khoj search results under `runs/ab/khoj/`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import requests


QSET = {
    "q01": "강화 학습",
    "q02": "Transformer architecture",
    "q03": "RAG implementation",
    "q04": "agent retrieval",
    "q05": "safety evaluation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Khoj q01-q05 A/B on a minimal shared corpus.")
    parser.add_argument("--base-url", default="http://127.0.0.1:42111", help="Khoj base URL")
    parser.add_argument(
        "--vault-root",
        default=os.environ.get("KHUB_AB_VAULT_ROOT", ""),
        help="Absolute path to the Obsidian vault root, or KHUB_AB_VAULT_ROOT",
    )
    parser.add_argument(
        "--baseline-dir",
        default="runs/ab/khoj",
        help="Directory containing knowledge_hub_q01.json ~ q05.json",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/ab/khoj",
        help="Directory to write khoj_q01.json ~ q05.json",
    )
    parser.add_argument(
        "--top-k-per-query",
        type=int,
        default=5,
        help="How many baseline results per query to use when deriving the minimal corpus",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="How many Khoj search results to request per query",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=1.0,
        help="Khoj max_distance parameter for search",
    )
    parser.add_argument(
        "--clear-computer-source",
        action="store_true",
        help="Delete any previously uploaded computer-source data before indexing",
    )
    return parser.parse_args()


def load_minimal_corpus(baseline_dir: Path, vault_root: Path, top_k_per_query: int) -> list[tuple[str, Path]]:
    file_paths: list[str] = []
    for qid in QSET:
        baseline_file = baseline_dir / f"knowledge_hub_{qid}.json"
        data = json.loads(baseline_file.read_text())
        for result in data.get("results", [])[:top_k_per_query]:
            rel_path = ((result.get("metadata") or {}).get("file_path")) or ""
            if rel_path:
                file_paths.append(rel_path)

    unique_rel_paths: list[str] = []
    seen: set[str] = set()
    for rel_path in file_paths:
        if rel_path not in seen:
            seen.add(rel_path)
            unique_rel_paths.append(rel_path)

    corpus: list[tuple[str, Path]] = []
    for rel_path in unique_rel_paths:
        abs_path = vault_root / rel_path
        if abs_path.exists():
            corpus.append((rel_path, abs_path))
    return corpus


def assert_server_ready(base_url: str) -> None:
    response = requests.get(f"{base_url}/api/health", timeout=10)
    response.raise_for_status()


def clear_existing_content(base_url: str) -> None:
    response = requests.delete(f"{base_url}/api/content/source/computer", timeout=30)
    response.raise_for_status()


def upload_markdown_batch(base_url: str, files: Iterable[tuple[str, Path]]) -> requests.Response:
    multipart = []
    handles = []
    try:
        for rel_path, abs_path in files:
            handle = abs_path.open("rb")
            handles.append(handle)
            multipart.append(("files", (rel_path, handle, "text/markdown")))
        response = requests.put(f"{base_url}/api/content", params={"t": "markdown"}, files=multipart, timeout=600)
        response.raise_for_status()
        return response
    finally:
        for handle in handles:
            handle.close()


def fetch_indexed_files(base_url: str) -> dict:
    response = requests.get(
        f"{base_url}/api/content/files",
        params={"truncated": "true", "page": 0},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def run_searches(base_url: str, output_dir: Path, max_results: int, max_distance: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for qid, query in QSET.items():
        response = requests.get(
            f"{base_url}/api/search",
            params={
                "q": query,
                "n": max_results,
                "t": "markdown",
                "max_distance": max_distance,
                "dedupe": "false",
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = {
            "query_id": qid,
            "query": query,
            "results": response.json(),
        }
        (output_dir / f"khoj_{qid}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir)
    if not args.vault_root:
        raise SystemExit("--vault-root or KHUB_AB_VAULT_ROOT is required.")
    vault_root = Path(args.vault_root).expanduser()

    assert_server_ready(base_url)

    corpus = load_minimal_corpus(baseline_dir, vault_root, args.top_k_per_query)
    if not corpus:
        raise SystemExit("No valid markdown files found for Khoj comparison corpus.")

    manifest = {
        "vault_root": str(vault_root),
        "baseline_dir": str(baseline_dir),
        "file_count": len(corpus),
        "files": [rel_path for rel_path, _ in corpus],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "khoj_corpus_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    if args.clear_computer_source:
        clear_existing_content(base_url)

    upload_response = upload_markdown_batch(base_url, corpus)
    indexed_files = fetch_indexed_files(base_url)
    (output_dir / "khoj_upload_response.json").write_text(
        json.dumps(
            {
                "status_code": upload_response.status_code,
                "body": upload_response.text,
                "indexed_files": indexed_files,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    run_searches(base_url, output_dir, args.max_results, args.max_distance)
    print(f"Uploaded {len(corpus)} files and wrote Khoj search outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
