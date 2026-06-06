from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from knowledge_hub.application.remote_index_contract import JsonObject, sha256_text, write_checksums, write_json, write_jsonl
from knowledge_hub.application.remote_query_embedding import EXPECTED_QUERY_DIMS

QUERY_EXPORT_SCHEMA: Final = "knowledge-hub.labs.paper-harness.query-embedding-request.v1"
DEFAULT_QUERY_MODEL: Final = "qwen3-embedding:8b"


@dataclass(frozen=True, slots=True)
class QueryRow:
    query_id: str
    query: str
    model: str
    expected_dim: int

    def to_json(self) -> JsonObject:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "query_text_hash": sha256_text(self.query),
            "embedding_model": self.model,
            "expected_embedding_dim": self.expected_dim,
        }


def _query_rows(queries: tuple[str, ...], model: str, expected_dim: int) -> list[QueryRow]:
    clean_queries = [query.strip() for query in queries if query.strip()]
    return [
        QueryRow(
            query_id=f"query:{index:04d}",
            query=query,
            model=model,
            expected_dim=expected_dim,
        )
        for index, query in enumerate(clean_queries, start=1)
    ]


def _hermes_prompt(model: str, expected_dim: int) -> str:
    return f"""You are the Oracle remote query-embedding worker for KnowledgeOS.

Mode: report_only query embedding.

Goal:
Given a sanitized query bundle, generate query embeddings with:
1. {model}

Hard boundaries:
- Do not scan any vault.
- Do not read or request private local paths outside the provided bundle.
- Do not run khub commands.
- Do not mutate any canonical SQLite DB, Chroma DB, vector index, repository file, or Obsidian vault.
- Do not download papers or parse PDFs.
- Do not change query ids, query text, or query_text_hash values.
- Output only derived query embedding artifacts, validation logs, and a report.

Input bundle expected:
- input_manifest.json
- queries.jsonl
- checksums.sha256
- run_instructions.md

Tasks:
1. Validate that input_manifest.json and queries.jsonl exist.
2. Verify checksums.sha256.
3. Confirm Ollama is available.
4. Pull or verify the model: {model}
5. Generate one embedding per query row.
6. Write outputs:
   - query_embeddings.qwen3-8b.jsonl
   - run_report.json
   - run_report.md
   - checksums.output.sha256
7. Each query embedding row must carry:
   - query_id
   - query
   - query_text_hash
   - embedding_model
   - embedding_dim
   - embedding
8. The expected embedding_dim is {expected_dim}.
9. If a query fails, keep successful rows and mark failures clearly in run_report.json.
10. After outputs are complete, do not import them anywhere. The local operator will fetch and review.
"""


def _run_instructions(model: str) -> str:
    return f"""# KnowledgeOS qwen8 query embedding request

This bundle contains sanitized query text only. Generate query embeddings with `{model}` and return the output artifacts without importing them into any local store.

Expected output file: `query_embeddings.qwen3-8b.jsonl`
"""


def export_query_embedding_bundle(*, queries: tuple[str, ...], out_dir: Path, model: str = DEFAULT_QUERY_MODEL) -> JsonObject:
    expected_dim = EXPECTED_QUERY_DIMS.get(model)
    if expected_dim is None:
        return {
            "schema": QUERY_EXPORT_SCHEMA,
            "status": "blocked",
            "blockers": ["unsupported_query_embedding_model"],
            "queryCount": 0,
            "targetModel": model,
        }

    rows = _query_rows(queries, model, expected_dim)
    if not rows:
        return {
            "schema": QUERY_EXPORT_SCHEMA,
            "status": "blocked",
            "blockers": ["query_required"],
            "queryCount": 0,
            "targetModel": model,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    query_payloads = [row.to_json() for row in rows]
    manifest: JsonObject = {
        "schema": QUERY_EXPORT_SCHEMA,
        "status": "ready",
        "queryCount": len(rows),
        "queryIds": [row.query_id for row in rows],
        "targetModel": model,
        "expectedEmbeddingDim": expected_dim,
        "inputFiles": ["queries.jsonl", "run_instructions.md", "hermes_prompt.md"],
        "expectedOutputFiles": ["query_embeddings.qwen3-8b.jsonl", "run_report.json", "run_report.md", "checksums.output.sha256"],
    }
    write_json(out_dir / "input_manifest.json", manifest)
    write_jsonl(out_dir / "queries.jsonl", query_payloads)
    (out_dir / "run_instructions.md").write_text(_run_instructions(model), encoding="utf-8")
    (out_dir / "hermes_prompt.md").write_text(_hermes_prompt(model, expected_dim), encoding="utf-8")
    write_checksums(out_dir, ["input_manifest.json", "queries.jsonl", "run_instructions.md", "hermes_prompt.md"])
    return {
        **manifest,
        "blockers": [],
        "files": ["input_manifest.json", "queries.jsonl", "run_instructions.md", "hermes_prompt.md", "checksums.sha256"],
    }


__all__ = ["DEFAULT_QUERY_MODEL", "QUERY_EXPORT_SCHEMA", "export_query_embedding_bundle"]
