from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from knowledge_hub.application.paper_query_export import DEFAULT_QUERY_MODEL, QUERY_EXPORT_SCHEMA
from knowledge_hub.application.remote_index_contract import JsonObject, read_json_object, read_jsonl_objects, verify_checksums
from knowledge_hub.application.remote_query_embedding import load_query_embedding_artifacts, query_row_id, query_row_text

QUERY_EMBEDDING_OUTPUTS = {
    "qwen3-embedding:4b": "query_embeddings.qwen3-4b.jsonl",
    "qwen3-embedding:8b": "query_embeddings.qwen3-8b.jsonl",
}


@dataclass(frozen=True, slots=True)
class QueryRunValidation:
    run_dir: Path
    model: str
    status: str
    blockers: tuple[str, ...]
    query_rows: tuple[JsonObject, ...]
    query_embedding_path: Path

    def to_json(self) -> JsonObject:
        return {
            "schema": QUERY_EXPORT_SCHEMA,
            "status": self.status,
            "blockers": list(self.blockers),
            "queryCount": len(self.query_rows),
            "queryIds": [query_row_id(row, index) for index, row in enumerate(self.query_rows, start=1)],
            "targetModel": self.model,
            "runDir": str(self.run_dir),
            "queryEmbeddingPath": str(self.query_embedding_path),
        }


def query_embedding_output_path(run_dir: Path, model: str = DEFAULT_QUERY_MODEL) -> Path:
    return run_dir / QUERY_EMBEDDING_OUTPUTS.get(model, "query_embeddings.unknown.jsonl")


def validate_query_embedding_run(run_dir: Path, model: str = DEFAULT_QUERY_MODEL) -> QueryRunValidation:
    output_path = query_embedding_output_path(run_dir, model)
    blockers: list[str] = []
    query_rows: list[JsonObject] = []

    if model not in QUERY_EMBEDDING_OUTPUTS:
        blockers.append("unsupported_query_embedding_model")
    if not (run_dir / "input_manifest.json").exists():
        blockers.append("input_manifest_missing")
    if not (run_dir / "queries.jsonl").exists():
        blockers.append("queries_missing")
    if not output_path.exists():
        blockers.append("query_embedding_file_missing")

    if not blockers:
        manifest = read_json_object(run_dir / "input_manifest.json")
        if str(manifest.get("schema") or "") != QUERY_EXPORT_SCHEMA:
            blockers.append("input_manifest_schema_mismatch")
        if str(manifest.get("targetModel") or "") != model:
            blockers.append("input_manifest_model_mismatch")
        query_rows = read_jsonl_objects(run_dir / "queries.jsonl")
        blockers.extend(verify_checksums(run_dir))
        blockers.extend(verify_checksums(run_dir, "checksums.output.sha256"))
        if not blockers:
            artifacts = load_query_embedding_artifacts(
                query_rows=query_rows,
                query_embeddings_path=output_path,
                model=model,
            )
            blockers.extend(artifacts.blockers)

    unique_blockers = tuple(sorted(set(blockers)))
    return QueryRunValidation(
        run_dir=run_dir,
        model=model,
        status="blocked" if unique_blockers else "ready",
        blockers=unique_blockers,
        query_rows=tuple(query_rows),
        query_embedding_path=output_path,
    )


def query_text_for_id(query_rows: tuple[JsonObject, ...], query_id: str) -> str:
    for index, row in enumerate(query_rows, start=1):
        if query_row_id(row, index) == query_id:
            return query_row_text(row)
    return ""


__all__ = [
    "QueryRunValidation",
    "query_embedding_output_path",
    "query_text_for_id",
    "validate_query_embedding_run",
]
