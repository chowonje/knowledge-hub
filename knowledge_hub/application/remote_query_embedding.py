from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from knowledge_hub.application.remote_index_contract import (
    JsonObject,
    read_jsonl_objects,
    sha256_text,
)

EXPECTED_QUERY_DIMS: Final = {
    "qwen3-embedding:4b": 2560,
    "qwen3-embedding:8b": 4096,
}


@dataclass(frozen=True, slots=True)
class QueryEmbeddingArtifacts:
    status: str
    blockers: tuple[str, ...]
    embeddings_by_query_id: dict[str, tuple[float, ...]]
    dimensions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SingleQueryEmbeddingArtifact:
    status: str
    blockers: tuple[str, ...]
    embedding: tuple[float, ...]
    dimension: int
    query_text_hash: str = ""


def query_row_id(row: JsonObject, index: int) -> str:
    raw_query_id = str(row.get("query_id") or row.get("queryId") or "").strip()
    return raw_query_id or f"query:{index:04d}"


def query_row_text(row: JsonObject) -> str:
    return str(row.get("query") or "").strip()


def query_row_text_hash(row: JsonObject) -> str:
    return sha256_text(query_row_text(row))


def _numeric_embedding(row: JsonObject) -> tuple[float, ...] | None:
    raw = row.get("embedding")
    if not isinstance(raw, list) or not raw:
        return None
    values: list[float] = []
    for item in raw:
        if not isinstance(item, (int, float)):
            return None
        values.append(float(item))
    return tuple(values)


def load_query_embedding_artifacts(
    *,
    query_rows: list[JsonObject],
    query_embeddings_path: str | Path,
    model: str,
) -> QueryEmbeddingArtifacts:
    path = Path(query_embeddings_path).expanduser()
    blockers: list[str] = []
    if not path.exists():
        return QueryEmbeddingArtifacts(
            status="blocked",
            blockers=("query_embedding_file_missing",),
            embeddings_by_query_id={},
            dimensions=(),
        )

    expected_by_id: dict[str, JsonObject] = {}
    for index, query_row in enumerate(query_rows, start=1):
        query_id = query_row_id(query_row, index)
        if query_id in expected_by_id:
            blockers.append("duplicate_query_id")
        expected_by_id[query_id] = query_row

    rows = read_jsonl_objects(path)
    embeddings_by_id: dict[str, tuple[float, ...]] = {}
    dimensions: set[int] = set()
    seen_ids: set[str] = set()
    expected_dim = EXPECTED_QUERY_DIMS.get(model)

    for row in rows:
        query_id = str(row.get("query_id") or row.get("queryId") or "").strip()
        if not query_id:
            blockers.append("query_id_missing")
            continue
        if query_id in seen_ids:
            blockers.append("duplicate_query_id")
        seen_ids.add(query_id)
        expected_query = expected_by_id.get(query_id)
        if expected_query is None:
            blockers.append("query_embedding_unexpected_query_id")
            continue
        if str(row.get("embedding_model") or "").strip() != model:
            blockers.append("query_embedding_model_mismatch")
        embedding = _numeric_embedding(row)
        if embedding is None:
            blockers.append("query_embedding_non_numeric")
            continue
        dim = int(row.get("embedding_dim") or 0)
        if len(embedding) != dim:
            blockers.append("query_embedding_dim_mismatch")
        if expected_dim is not None and dim != expected_dim:
            blockers.append("query_embedding_dim_mismatch")
        if str(row.get("query_text_hash") or "").strip() != query_row_text_hash(expected_query):
            blockers.append("query_text_hash_mismatch")
        if dim > 0:
            dimensions.add(dim)
        embeddings_by_id[query_id] = embedding

    for query_id in expected_by_id:
        if query_id not in seen_ids:
            blockers.append("query_embedding_missing")

    sorted_blockers = tuple(sorted(set(blockers)))
    return QueryEmbeddingArtifacts(
        status="blocked" if sorted_blockers else "ready",
        blockers=sorted_blockers,
        embeddings_by_query_id=embeddings_by_id if not sorted_blockers else {},
        dimensions=tuple(sorted(dimensions)),
    )


def load_query_embedding_for_id(
    *,
    query_embeddings_path: str | Path,
    model: str,
    query_id: str,
    expected_query_text_hash: str = "",
) -> SingleQueryEmbeddingArtifact:
    path = Path(query_embeddings_path).expanduser()
    if not path.exists():
        return SingleQueryEmbeddingArtifact("blocked", ("query_embedding_file_missing",), (), 0)
    expected_dim = EXPECTED_QUERY_DIMS.get(model)
    blockers: list[str] = []
    found: tuple[float, ...] = ()
    found_dim = 0
    found_query_text_hash = ""
    match_count = 0
    for row in read_jsonl_objects(path):
        if str(row.get("query_id") or row.get("queryId") or "").strip() != query_id:
            continue
        match_count += 1
        found_query_text_hash = str(row.get("query_text_hash") or "").strip()
        if str(row.get("embedding_model") or "").strip() != model:
            blockers.append("query_embedding_model_mismatch")
        if expected_query_text_hash and found_query_text_hash != expected_query_text_hash:
            blockers.append("query_text_hash_mismatch")
        embedding = _numeric_embedding(row)
        if embedding is None:
            blockers.append("query_embedding_non_numeric")
            continue
        dim = int(row.get("embedding_dim") or 0)
        if len(embedding) != dim:
            blockers.append("query_embedding_dim_mismatch")
        if expected_dim is not None and dim != expected_dim:
            blockers.append("query_embedding_dim_mismatch")
        found = embedding
        found_dim = dim
    if match_count == 0:
        blockers.append("query_embedding_missing")
    if match_count > 1:
        blockers.append("duplicate_query_id")
    sorted_blockers = tuple(sorted(set(blockers)))
    return SingleQueryEmbeddingArtifact(
        status="blocked" if sorted_blockers else "ready",
        blockers=sorted_blockers,
        embedding=() if sorted_blockers else found,
        dimension=found_dim,
        query_text_hash=found_query_text_hash,
    )
