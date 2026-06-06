from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from knowledge_hub.application.remote_index_contract import JsonObject, sha256_text
from knowledge_hub.application.remote_query_embedding import load_query_embedding_for_id
from knowledge_hub.infrastructure.config import Config
from knowledge_hub.infrastructure.persistence.vector import VectorDatabase

DEFAULT_QWEN8_NAMESPACE: Final = "qwen3_8b_full_candidate"
DEFAULT_QWEN8_MODEL: Final = "qwen3-embedding:8b"
PAPER_FILTER: Final[JsonObject] = {"source_type": "paper"}


class EmbedderLike(Protocol):
    def embed_text(self, text: str) -> list[float]: ...


class KhubLike(Protocol):
    config: Config

    def build_embedder(self, provider: str, model: str | None = None) -> EmbedderLike: ...


@dataclass(frozen=True, slots=True)
class HarnessArm:
    name: str
    status: str
    namespace: str
    model: str
    result_count: int
    blockers: tuple[str, ...] = ()
    query_embedding_source: str = "local"

    def to_json(self) -> JsonObject:
        return {
            "name": self.name,
            "status": self.status,
            "namespace": self.namespace,
            "model": self.model,
            "resultCount": self.result_count,
            "blockers": list(self.blockers),
            "queryEmbeddingSource": self.query_embedding_source,
        }


def _paper_id(metadata: JsonObject) -> str:
    for key in ("paper_id", "arxiv_id", "source_id", "document_id"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return ""


def _chunk_id(metadata: JsonObject, doc_id: str) -> str:
    for key in ("chunk_id", "prepared_segment_id", "document_id"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return str(doc_id or "")


def _title(metadata: JsonObject) -> str:
    for key in ("title", "parent_title", "section_title"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return ""


def _snippet(document: str, limit: int = 720) -> str:
    return " ".join(str(document or "").split())[: max(80, int(limit))]


def _semantic_items(results: JsonObject, arm_name: str) -> list[JsonObject]:
    ids = list((results.get("ids") or [[]])[0] or [])
    documents = list((results.get("documents") or [[]])[0] or [])
    metadatas = list((results.get("metadatas") or [[]])[0] or [])
    distances = list((results.get("distances") or [[]])[0] or [])
    items: list[JsonObject] = []
    for index, doc_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        document = str(documents[index] if index < len(documents) else "")
        distance = float(distances[index] if index < len(distances) else 0.0)
        items.append(_evidence_seed(arm_name, index + 1, str(doc_id), document, metadata, distance))
    return items


def _lexical_items(results: list[JsonObject], arm_name: str) -> list[JsonObject]:
    items: list[JsonObject] = []
    for index, row in enumerate(results):
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        document = str(row.get("document") or "")
        score = float(row.get("score") or 0.0)
        items.append(_evidence_seed(arm_name, index + 1, str(row.get("id") or ""), document, metadata, score))
    return items


def _evidence_seed(arm_name: str, rank: int, doc_id: str, document: str, metadata: JsonObject, score: float) -> JsonObject:
    return {
        "id": doc_id,
        "paperId": _paper_id(metadata),
        "chunkId": _chunk_id(metadata, doc_id),
        "title": _title(metadata),
        "sourceHash": str(metadata.get("source_hash") or ""),
        "chunkTextHash": str(metadata.get("chunk_text_hash") or ""),
        "snippet": _snippet(document),
        "retrievalArms": [arm_name],
        "rankSignals": {arm_name: {"rank": rank, "score": score}},
    }


def _merge_items(item_groups: list[list[JsonObject]], limit: int) -> list[JsonObject]:
    merged: dict[str, JsonObject] = {}
    for items in item_groups:
        for item in items:
            key = str(item.get("chunkId") or item.get("id") or "")
            if key not in merged:
                merged[key] = dict(item)
                continue
            existing = merged[key]
            arms = list(existing.get("retrievalArms") or [])
            for arm in list(item.get("retrievalArms") or []):
                if arm not in arms:
                    arms.append(str(arm))
            existing["retrievalArms"] = arms
            rank_signals = dict(existing.get("rankSignals") or {})
            rank_signals.update(dict(item.get("rankSignals") or {}))
            existing["rankSignals"] = rank_signals
    ranked = sorted(
        merged.values(),
        key=lambda item: (
            -len(list(item.get("retrievalArms") or [])),
            min(int(signal.get("rank") or 9999) for signal in dict(item.get("rankSignals") or {}).values()),
            str(item.get("paperId") or ""),
        ),
    )
    return ranked[: max(1, int(limit))]


def _overall_status(arms: list[HarnessArm], evidence: list[JsonObject]) -> str:
    blocked = [arm for arm in arms if arm.status == "blocked"]
    ok = [arm for arm in arms if arm.status == "ok"]
    if not evidence and not ok:
        return "blocked"
    if blocked:
        return "partial"
    return "ok"


def retrieve_paper_evidence_pack(
    *,
    khub: KhubLike,
    query: str,
    use_bge: bool = True,
    use_keyword: bool = True,
    use_qwen8: bool = True,
    qwen_query_embeddings_path: Path | None = None,
    qwen_query_id: str = "",
    qwen_namespace: str = DEFAULT_QWEN8_NAMESPACE,
    qwen_model: str = DEFAULT_QWEN8_MODEL,
    top_k: int = 5,
) -> JsonObject:
    arms: list[HarnessArm] = []
    groups: list[list[JsonObject]] = []
    config = khub.config
    if use_bge:
        embedder = khub.build_embedder(config.embedding_provider, config.embedding_model)
        vector_db = VectorDatabase(config.vector_db_path, config.collection_name, repair_on_init=False)
        results = vector_db.search(embedder.embed_text(query), top_k=top_k, filter_dict=PAPER_FILTER, include_stale=True)
        items = _semantic_items(results, "bge")
        groups.append(items)
        arms.append(HarnessArm("bge", "ok", config.collection_name, config.embedding_model, len(items)))
    if use_keyword:
        vector_db = VectorDatabase(config.vector_db_path, config.collection_name, repair_on_init=False)
        items = _lexical_items(
            vector_db.lexical_search(query, top_k=top_k, filter_dict=PAPER_FILTER, include_stale=True),
            "keyword",
        )
        groups.append(items)
        arms.append(HarnessArm("keyword", "ok", config.collection_name, "fts5", len(items), query_embedding_source="none"))
    if use_qwen8:
        if qwen_query_embeddings_path is None or not qwen_query_id:
            arms.append(HarnessArm("qwen8", "blocked", qwen_namespace, qwen_model, 0, ("qwen_query_embedding_required",), "artifact"))
        else:
            artifact = load_query_embedding_for_id(
                query_embeddings_path=qwen_query_embeddings_path,
                model=qwen_model,
                query_id=qwen_query_id,
                expected_query_text_hash=sha256_text(query),
            )
            if artifact.status == "blocked":
                arms.append(HarnessArm("qwen8", "blocked", qwen_namespace, qwen_model, 0, tuple(artifact.blockers), "artifact"))
            else:
                vector_db = VectorDatabase(config.vector_db_path, qwen_namespace, repair_on_init=False)
                results = vector_db.search(list(artifact.embedding), top_k=top_k, filter_dict=PAPER_FILTER, include_stale=True)
                items = _semantic_items(results, "qwen8")
                groups.append(items)
                arms.append(HarnessArm("qwen8", "ok", qwen_namespace, qwen_model, len(items), query_embedding_source="artifact"))
    evidence = _merge_items(groups, top_k)
    return {
        "schema": "knowledge-hub.labs.paper-retrieval-harness.v1",
        "status": _overall_status(arms, evidence),
        "query": query,
        "topK": int(top_k),
        "arms": [arm.to_json() for arm in arms],
        "evidence": evidence,
        "runtimeDiagnostics": {
            "canonicalNamespace": config.collection_name,
            "canonicalModel": config.embedding_model,
            "vectorDbPath": config.vector_db_path,
            "qwenNamespace": qwen_namespace,
        },
    }
