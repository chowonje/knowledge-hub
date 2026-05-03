"""Canonical vector database implementation."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from knowledge_hub.infrastructure.persistence.stores.derivative_lifecycle import source_hash_from_content

log = logging.getLogger("khub.database")

try:
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None

SQLITE_BUSY_TIMEOUT_MS = 5000


def _safe_sqlite_count(db_path: Path, query: str, params: tuple[Any, ...] = ()) -> int:
    if not db_path.exists():
        return 0
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1)
        row = conn.execute(query, params).fetchone()
        if not row:
            return 0
        return int(row[0] or 0)
    except sqlite3.Error:
        return 0
    finally:
        if conn is not None:
            conn.close()


def _safe_sqlite_rows(
    db_path: Path,
    query: str,
    params: tuple[Any, ...] = (),
) -> list[sqlite3.Row]:
    if not db_path.exists():
        return []
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1)
        conn.row_factory = sqlite3.Row
        return list(conn.execute(query, params).fetchall() or [])
    except sqlite3.Error:
        return []
    finally:
        if conn is not None:
            conn.close()


def _inspect_vector_root(root: Path, collection_name: str = "") -> dict[str, Any]:
    lexical_db_path = root / "_lexical.sqlite3"
    chroma_sqlite_path = root / "chroma.sqlite3"
    lexical_count = _safe_sqlite_count(lexical_db_path, "SELECT count(*) FROM lexical_documents_fts")
    collection_token = str(collection_name or "").strip()
    if collection_token:
        chroma_count = _safe_sqlite_count(
            chroma_sqlite_path,
            """
            SELECT count(*)
            FROM embeddings AS e
            JOIN segments AS s ON e.segment_id = s.id
            JOIN collections AS c ON s.collection = c.id
            WHERE c.name = ?
            """,
            (collection_token,),
        )
    else:
        chroma_count = _safe_sqlite_count(chroma_sqlite_path, "SELECT count(*) FROM embeddings")
    total_documents = chroma_count

    reasons: list[str] = []
    if chroma_count <= 0:
        reasons.append("vector_corpus_empty")
    if lexical_count > 0 and chroma_count <= 0:
        reasons.append("vector_embeddings_missing")
    if not str(collection_name or "").strip():
        reasons.append("vector_collection_missing")

    payload: dict[str, Any] = {
        "available": chroma_count > 0 and bool(str(collection_name or "").strip()),
        "degraded": bool(reasons),
        "reasons": reasons,
        "collection_name": str(collection_name or ""),
        "total_documents": total_documents,
        "db_path": str(root),
        "lexical_db_path": str(lexical_db_path),
        "lexical_documents": lexical_count,
        "chroma_embeddings": chroma_count,
    }
    return payload


def list_vector_documents(db_path: str | Path, limit: int = 5000) -> list[dict[str, Any]]:
    root = Path(db_path).expanduser()
    lexical_db_path = root / "_lexical.sqlite3"
    rows = _safe_sqlite_rows(
        lexical_db_path,
        """
        SELECT f.doc_id, f.title, m.metadata_json
        FROM lexical_documents_fts AS f
        LEFT JOIN lexical_documents_meta AS m ON m.doc_id = f.doc_id
        ORDER BY f.doc_id
        LIMIT ?
        """,
        (max(1, int(limit)),),
    )
    items: list[dict[str, Any]] = []
    for row in rows:
        metadata_json = str(row["metadata_json"] or "").strip()
        metadata = {}
        if metadata_json:
            try:
                metadata = dict(json.loads(metadata_json) or {})
            except Exception:
                metadata = {}
        items.append(
            {
                "doc_id": str(row["doc_id"] or ""),
                "title": str(row["title"] or metadata.get("title") or ""),
                "source_type": str(metadata.get("source_type") or ""),
                "file_path": str(metadata.get("file_path") or ""),
                "arxiv_id": str(metadata.get("arxiv_id") or ""),
            }
        )
    return items


def _publisher_heuristic(source_type: str, file_path: str) -> str:
    if str(source_type or "").strip() != "web":
        return ""
    token = Path(str(file_path or "")).stem.lower()
    for publisher in ("anthropic", "openai", "openreview"):
        if token.endswith(f"-{publisher}") or f"-{publisher}-" in token:
            return publisher
    return "other_web"


def _mtime_range(file_paths: set[str]) -> dict[str, str]:
    mtimes: list[float] = []
    for raw_path in sorted(file_paths):
        candidate = Path(str(raw_path or "")).expanduser()
        try:
            if candidate.exists():
                mtimes.append(float(candidate.stat().st_mtime))
        except Exception:
            continue
    if not mtimes:
        return {}
    return {
        "oldest": datetime.fromtimestamp(min(mtimes)).isoformat(),
        "newest": datetime.fromtimestamp(max(mtimes)).isoformat(),
    }


def _document_group_summary(items: list[dict[str, Any]], *, top_n: int = 5) -> dict[str, Any]:
    source_counts = Counter(str(item.get("source_type") or "<empty>") for item in items)
    publisher_counts = Counter()
    file_counts = Counter()
    file_source_type: dict[str, str] = {}
    nonempty_paths: set[str] = set()

    for item in items:
        file_path = str(item.get("file_path") or "")
        source_type = str(item.get("source_type") or "")
        publisher = _publisher_heuristic(source_type, file_path)
        if publisher:
            publisher_counts[publisher] += 1
        if file_path:
            nonempty_paths.add(file_path)
            file_counts[file_path] += 1
            file_source_type.setdefault(file_path, source_type)

    top_files = []
    for file_path, count in file_counts.most_common(max(1, int(top_n))):
        top_files.append(
            {
                "file_path": file_path,
                "count": int(count),
                "source_type": str(file_source_type.get(file_path) or ""),
                "publisher": _publisher_heuristic(str(file_source_type.get(file_path) or ""), file_path),
            }
        )

    return {
        "documentCount": len(items),
        "sourceTypeCounts": dict(source_counts.most_common()),
        "publisherCounts": dict(publisher_counts.most_common()),
        "uniqueFileCount": len(nonempty_paths),
        "missingFilePathCount": max(0, len(items) - sum(file_counts.values())),
        "fileMtimeRange": _mtime_range(nonempty_paths),
        "topFiles": top_files,
    }


def _compare_decision_hint(
    *,
    active_only_count: int,
    backup_only_count: int,
    shared_count: int,
    backup_only_summary: dict[str, Any],
) -> dict[str, Any]:
    source_counts = dict(backup_only_summary.get("sourceTypeCounts") or {})
    publisher_counts = dict(backup_only_summary.get("publisherCounts") or {})
    reason_codes: list[str] = []

    if active_only_count == 0 and shared_count > 0:
        reason_codes.append("active_fully_contained_in_backup")
    if backup_only_count > 0 and set(source_counts) == {"web"}:
        reason_codes.append("backup_only_all_web")
    if backup_only_count > 0 and publisher_counts:
        dominant_publisher, dominant_count = next(iter(publisher_counts.items()))
        if int(dominant_count) > 0 and (int(dominant_count) / max(1, backup_only_count)) >= 0.5:
            reason_codes.append(f"backup_only_dominant_publisher:{dominant_publisher}")

    recommendation = "manual_review"
    summary = "backup and active differ; review document provenance before changing the active corpus."
    if active_only_count == 0 and backup_only_count == 0:
        recommendation = "restore_equivalent"
        summary = "backup and active contain the same document ids; restore would not change retrieval coverage."
    elif active_only_count == 0 and backup_only_count > 0 and set(source_counts) == {"web"}:
        recommendation = "review_before_restore"
        summary = (
            "backup fully contains the active corpus, but its extra documents are dominated by web sources; "
            "review provenance before restore."
        )
    elif active_only_count == 0 and backup_only_count > 0:
        recommendation = "review_before_restore"
        summary = "backup fully contains the active corpus and adds extra documents; inspect source mix before restore."
    elif active_only_count > 0 and backup_only_count == 0:
        recommendation = "keep_active"
        summary = "active contains documents missing from the backup; avoid restore unless those losses are intentional."

    return {
        "kind": "metadata_only_heuristic",
        "recommendedAction": recommendation,
        "summary": summary,
        "reasonCodes": reason_codes,
    }


def compare_vector_stores(
    active_db_path: str | Path,
    backup_db_path: str | Path,
    *,
    collection_name: str = "",
    sample_limit: int = 10,
    document_limit: int = 10000,
) -> dict[str, Any]:
    active_docs = list_vector_documents(active_db_path, limit=document_limit)
    backup_docs = list_vector_documents(backup_db_path, limit=document_limit)
    active_by_id = {str(item.get("doc_id") or ""): item for item in active_docs if str(item.get("doc_id") or "")}
    backup_by_id = {str(item.get("doc_id") or ""): item for item in backup_docs if str(item.get("doc_id") or "")}
    active_ids = set(active_by_id)
    backup_ids = set(backup_by_id)
    active_only_ids = sorted(active_ids - backup_ids)
    backup_only_ids = sorted(backup_ids - active_ids)
    shared_ids = sorted(active_ids & backup_ids)
    changed_ids: list[str] = []
    for doc_id in shared_ids:
        active_item = active_by_id.get(doc_id) or {}
        backup_item = backup_by_id.get(doc_id) or {}
        if (
            str(active_item.get("title") or "") != str(backup_item.get("title") or "")
            or str(active_item.get("file_path") or "") != str(backup_item.get("file_path") or "")
            or str(active_item.get("arxiv_id") or "") != str(backup_item.get("arxiv_id") or "")
        ):
            changed_ids.append(doc_id)

    def _sample(items_by_id: dict[str, dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
        return [dict(items_by_id[item_id]) for item_id in ids[: max(1, int(sample_limit))]]

    active_only_items = [dict(active_by_id[item_id]) for item_id in active_only_ids]
    backup_only_items = [dict(backup_by_id[item_id]) for item_id in backup_only_ids]
    shared_items = [dict(backup_by_id[item_id]) for item_id in shared_ids if item_id in backup_by_id]
    provenance = {
        "activeOnly": _document_group_summary(active_only_items),
        "backupOnly": _document_group_summary(backup_only_items),
        "shared": _document_group_summary(shared_items),
    }
    provenance["decisionHint"] = _compare_decision_hint(
        active_only_count=len(active_only_ids),
        backup_only_count=len(backup_only_ids),
        shared_count=len(shared_ids),
        backup_only_summary=provenance["backupOnly"],
    )

    return {
        "documentLimit": int(document_limit),
        "sharedCount": len(shared_ids),
        "activeOnlyCount": len(active_only_ids),
        "backupOnlyCount": len(backup_only_ids),
        "changedSharedCount": len(changed_ids),
        "activeOnlySample": _sample(active_by_id, active_only_ids),
        "backupOnlySample": _sample(backup_by_id, backup_only_ids),
        "changedSharedSample": [
            {
                "doc_id": doc_id,
                "active": dict(active_by_id.get(doc_id) or {}),
                "backup": dict(backup_by_id.get(doc_id) or {}),
            }
            for doc_id in changed_ids[: max(1, int(sample_limit))]
        ],
        "provenance": provenance,
    }


def list_vector_backups(db_path: str | Path, collection_name: str = "") -> list[dict[str, Any]]:
    root = Path(db_path).expanduser()
    backups: list[dict[str, Any]] = []
    pattern = f"{root.name}.corrupt.*"
    for candidate in sorted(root.parent.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True):
        inspection = _inspect_vector_root(candidate, collection_name)
        if int(inspection.get("lexical_documents", 0) or 0) <= 0 and int(inspection.get("chroma_embeddings", 0) or 0) <= 0:
            continue
        inspection["path"] = str(candidate)
        inspection["restorable"] = bool(inspection.get("available"))
        backups.append(inspection)
    return backups


def inspect_vector_store(db_path: str | Path, collection_name: str = "") -> dict[str, Any]:
    """Inspect vector-store readiness without creating or repairing a Chroma client."""

    root = Path(db_path).expanduser()
    payload = _inspect_vector_root(root, collection_name)
    latest_backup = next(iter(list_vector_backups(root, collection_name)), None)
    if latest_backup is not None:
        probe = probe_vector_store_openability(latest_backup.get("path", ""), collection_name)
        latest_backup = dict(latest_backup)
        latest_backup["openable"] = bool(probe.get("openable"))
        latest_backup["openProbeError"] = str(probe.get("error") or "")
        latest_backup["restorable"] = bool(latest_backup.get("available")) and bool(latest_backup.get("openable"))
        payload["recovery_backup"] = latest_backup
    return payload


class VectorDatabaseInitError(RuntimeError):
    """Raised when a vector store cannot be opened without repair."""


class _NoOpEmbeddingFunction:
    """Prevent Chroma from auto-loading ONNX embedding models."""

    def __init__(self) -> None:
        return

    def __call__(self, input):
        return [[0.0]] * len(input)

    def is_legacy(self) -> bool:
        return False

    @staticmethod
    def name():
        return "noop"

    @staticmethod
    def build_from_config(config):
        return _NoOpEmbeddingFunction()

    def get_config(self):
        return {}


class VectorDatabase:
    """ChromaDB + FTS5-backed vector database."""

    def __init__(
        self,
        db_path: str,
        collection_name: str = "knowledge_hub",
        *,
        repair_on_init: bool = True,
        create_collection: bool = True,
        init_lexical: bool = True,
    ):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.repair_on_init = bool(repair_on_init)
        self.create_collection = bool(create_collection)
        self.init_lexical = bool(init_lexical)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.lexical_db_path = self.db_path / "_lexical.sqlite3"

        self.client = self._create_client_with_repair()
        self._ef = _NoOpEmbeddingFunction()
        self.collection = self._get_or_create_collection()
        self.lexical_conn: sqlite3.Connection | None = None
        if self.init_lexical:
            self.lexical_conn = sqlite3.connect(str(self.lexical_db_path), timeout=SQLITE_BUSY_TIMEOUT_MS / 1000)
            self.lexical_conn.row_factory = sqlite3.Row
            self._init_lexical_index()

    def _init_lexical_index(self) -> None:
        if self.lexical_conn is None:
            return
        self.lexical_conn.execute("PRAGMA journal_mode=WAL")
        self.lexical_conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        self.lexical_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lexical_documents_meta (
                doc_id TEXT PRIMARY KEY,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        self.lexical_conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS lexical_documents_fts
            USING fts5(
                doc_id UNINDEXED,
                title,
                section_title,
                contextual_summary,
                keywords,
                field,
                document,
                searchable_text
            )
            """
        )
        self.lexical_conn.commit()

    def _is_backup_path(self) -> bool:
        return any(".corrupt." in part for part in self.db_path.parts)

    def _create_client_with_repair(self):
        settings = Settings(anonymized_telemetry=False)
        try:
            return chromadb.PersistentClient(path=str(self.db_path), settings=settings)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if not self.repair_on_init or self._is_backup_path():
                raise VectorDatabaseInitError(f"ChromaDB init failed for {self.db_path}: {exc}") from exc
            log.warning("Primary ChromaDB init failed for %s, attempting repair", self.db_path)
            return self._repair_corrupted_db_and_recreate(settings)

    def _backup_corrupted_path(self, db_path: Path) -> Path | None:
        if not db_path.exists():
            return None
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = db_path.with_name(f"{db_path.name}.corrupt.{stamp}")
        idx = 0
        while backup_path.exists():
            idx += 1
            backup_path = db_path.with_name(f"{db_path.name}.corrupt.{stamp}.{idx}")
        try:
            if db_path.is_dir():
                shutil.move(str(db_path), str(backup_path))
            else:
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(db_path), str(backup_path))
            log.warning("Moved corrupted ChromaDB path to backup: %s", backup_path)
            return backup_path
        except Exception as error:
            log.warning("Failed to backup corrupted ChromaDB path %s: %s", db_path, error)
            return None

    def _repair_corrupted_db_and_recreate(self, settings: Settings):
        if self.db_path.exists():
            self._backup_corrupted_path(self.db_path)
            if self.db_path.exists():
                try:
                    shutil.rmtree(self.db_path, ignore_errors=True)
                except Exception as error:
                    log.warning("Failed to remove corrupted ChromaDB path %s: %s", self.db_path, error)
        return chromadb.PersistentClient(path=str(self.db_path), settings=settings)

    def _get_or_create_collection(self):
        try:
            if self.create_collection:
                return self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=self._ef,
                )
            return self.client.get_collection(
                name=self.collection_name,
                embedding_function=self._ef,
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if not self.repair_on_init or self._is_backup_path():
                raise VectorDatabaseInitError(f"ChromaDB collection init failed for {self.collection_name}: {exc}") from exc
            log.warning("Failed to open existing collection '%s': %s", self.collection_name, exc)
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception as error:
                log.warning("Failed to delete collection '%s': %s", self.collection_name, error)
            try:
                return self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=self._ef,
                )
            except BaseException as retry_error:
                log.warning("Retry collection init failed for '%s': %s", self.collection_name, retry_error)
                self.client = self._repair_corrupted_db_and_recreate(Settings(anonymized_telemetry=False))
                return self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=self._ef,
                )

    @staticmethod
    def _decode_metadata_value(value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if (stripped.startswith("{") and stripped.endswith("}")) or (
                stripped.startswith("[") and stripped.endswith("]")
            ):
                try:
                    return json.loads(value)
                except Exception:
                    return value
        return value

    @staticmethod
    def _encode_metadata_value(value: Any) -> Any:
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False, default=str)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (str, int, float, bool)):
            return value
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _normalize_metadatas(metadatas: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        if not metadatas:
            return []
        normalized: list[dict[str, Any]] = []
        for raw in metadatas:
            item: dict[str, Any] = {}
            for key, value in raw.items():
                item[key] = VectorDatabase._decode_metadata_value(value)
            normalized.append(item)
        return normalized

    @staticmethod
    def _build_lexical_row(document: str, metadata: dict[str, Any]) -> dict[str, str]:
        title = str(metadata.get("title", "") or "")
        section_title = str(metadata.get("section_title", "") or "")
        contextual_summary = str(metadata.get("contextual_summary", "") or "")
        keywords = metadata.get("keywords", "")
        if isinstance(keywords, list):
            keywords = " ".join(str(item) for item in keywords if str(item).strip())
        field = str(metadata.get("field", "") or "")
        searchable_parts = [
            title,
            section_title,
            contextual_summary,
            str(keywords or ""),
            field,
            str(document or ""),
        ]
        return {
            "title": title,
            "section_title": section_title,
            "contextual_summary": contextual_summary,
            "keywords": str(keywords or ""),
            "field": field,
            "document": str(document or ""),
            "searchable_text": " ".join(part for part in searchable_parts if part).strip(),
        }

    @staticmethod
    def _metadata_matches_filter(metadata: dict[str, Any], filter_dict: dict[str, Any] | None) -> bool:
        if not filter_dict:
            return True
        if "$and" in filter_dict:
            clauses = list(filter_dict.get("$and") or [])
            return all(VectorDatabase._metadata_matches_filter(metadata, clause) for clause in clauses if isinstance(clause, dict))
        for key, expected in filter_dict.items():
            actual = metadata.get(key)
            if isinstance(expected, dict) and "$eq" in expected:
                expected = expected["$eq"]
            if actual != expected:
                return False
        return True

    @staticmethod
    def _metadata_is_stale(metadata: dict[str, Any] | None) -> bool:
        if not isinstance(metadata, dict):
            return False
        value = metadata.get("stale")
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "stale"}
        return bool(value)

    @staticmethod
    def _filter_query_results(
        results: dict[str, Any],
        *,
        top_k: int,
        include_stale: bool,
    ) -> dict[str, Any]:
        if include_stale:
            return results
        metadatas = results.get("metadatas")
        if not metadatas:
            return results

        filtered: dict[str, Any] = dict(results)
        row_count = len(metadatas)
        for key, value in list(results.items()):
            if isinstance(value, list) and len(value) == row_count:
                filtered[key] = []

        for row_index, metadata_row in enumerate(metadatas):
            metadata_items = list(metadata_row or [])
            keep_indexes = [
                item_index
                for item_index, metadata in enumerate(metadata_items)
                if not VectorDatabase._metadata_is_stale(metadata)
            ][: max(1, int(top_k))]
            for key, value in list(results.items()):
                if not (isinstance(value, list) and len(value) == row_count):
                    continue
                row = value[row_index] or []
                filtered[key].append([row[item_index] for item_index in keep_indexes if item_index < len(row)])
        return filtered

    @staticmethod
    def _normalize_where_filter(filter_dict: dict[str, Any] | None) -> dict[str, Any] | None:
        if not filter_dict:
            return None
        if "$and" in filter_dict:
            clauses = [
                VectorDatabase._normalize_where_filter(clause)
                for clause in list(filter_dict.get("$and") or [])
                if isinstance(clause, dict)
            ]
            clauses = [clause for clause in clauses if clause]
            if not clauses:
                return None
            if len(clauses) == 1:
                return clauses[0]
            return {"$and": clauses}
        normalized: list[dict[str, Any]] = []
        for key, value in filter_dict.items():
            if isinstance(value, dict):
                normalized.append({key: value})
            else:
                normalized.append({key: {"$eq": value}})
        if not normalized:
            return None
        if len(normalized) == 1:
            return normalized[0]
        return {"$and": normalized}

    @contextmanager
    def _collection_lock(self):
        if fcntl is None:
            yield
            return

        lock_path = self.db_path.with_name(f".{self.db_path.name}.write.lock")
        lock_handle = None
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            lock_handle = os.fdopen(fd, "a+")
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if lock_handle is not None:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                lock_handle.close()
            elif fd is not None:
                os.close(fd)

    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        ids: list[str] | None = None,
    ):
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        clean_metadatas: list[dict[str, Any]] = []
        for metadata in metadatas:
            clean = {}
            for key, value in metadata.items():
                clean[key] = VectorDatabase._encode_metadata_value(value)
            clean_metadatas.append(clean)

        with self._collection_lock():
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=clean_metadatas,
            )
            if self.lexical_conn is not None:
                lexical_rows = []
                for doc_id, document, metadata in zip(ids, documents, metadatas):
                    lexical_rows.append(
                        (
                            doc_id,
                            json.dumps(metadata or {}, ensure_ascii=False, default=str),
                            self._build_lexical_row(document or "", metadata or {}),
                        )
                    )
                for doc_id, metadata_json, lexical_row in lexical_rows:
                    self.lexical_conn.execute("DELETE FROM lexical_documents_fts WHERE doc_id = ?", (doc_id,))
                    self.lexical_conn.execute(
                        """INSERT INTO lexical_documents_fts
                           (doc_id, title, section_title, contextual_summary, keywords, field, document, searchable_text)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            doc_id,
                            lexical_row["title"],
                            lexical_row["section_title"],
                            lexical_row["contextual_summary"],
                            lexical_row["keywords"],
                            lexical_row["field"],
                            lexical_row["document"],
                            lexical_row["searchable_text"],
                        ),
                    )
                    self.lexical_conn.execute(
                        """INSERT INTO lexical_documents_meta(doc_id, metadata_json)
                           VALUES (?, ?)
                           ON CONFLICT(doc_id) DO UPDATE SET metadata_json = excluded.metadata_json""",
                        (doc_id, metadata_json),
                    )
                self.lexical_conn.commit()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_dict: dict[str, Any] | None = None,
        include_stale: bool = False,
    ) -> dict[str, Any]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, int(top_k)) if include_stale else max(1, int(top_k) * 4),
            where=self._normalize_where_filter(filter_dict),
        )
        if results.get("metadatas"):
            normalized = [
                VectorDatabase._normalize_metadatas(metadata_list)
                for metadata_list in results["metadatas"]
                if metadata_list
            ]
            if normalized:
                results["metadatas"] = normalized
        return self._filter_query_results(results, top_k=top_k, include_stale=include_stale)

    def get_documents(
        self,
        filter_dict: dict[str, Any] | None = None,
        limit: int = 500,
        offset: int = 0,
        include_ids: bool = True,
        include_documents: bool = True,
        include_metadatas: bool = True,
        include_embeddings: bool = False,
    ) -> dict[str, Any]:
        include: list[str] = []
        if include_documents:
            include.append("documents")
        if include_metadatas:
            include.append("metadatas")
        if include_embeddings:
            include.append("embeddings")

        request = {
            "where": self._normalize_where_filter(filter_dict),
            "limit": max(1, limit),
            "offset": max(0, int(offset)),
            "include": include,
        }
        request = {key: value for key, value in request.items() if value}
        results = self.collection.get(**request)

        if include_documents and results.get("documents"):
            results["documents"] = [(doc or "") for doc in (results.get("documents") or [])]
        if include_metadatas and results.get("metadatas"):
            results["metadatas"] = self._normalize_metadatas(results.get("metadatas") or [])
        if include_embeddings:
            embeddings = results.get("embeddings")
            if embeddings is not None:
                results["embeddings"] = list(embeddings)
        if not include_ids:
            results.pop("ids", None)
        return results

    @staticmethod
    def _source_metadata_identity(*, doc_id: str, metadata: dict[str, Any]) -> str:
        for key in ("document_id", "file_path", "arxiv_id", "url", "canonical_url", "title"):
            token = str((metadata or {}).get(key) or "").strip()
            if token:
                return token
        return str(doc_id or "").strip()

    def source_metadata_rows(self, *, limit: int = 10000, offset: int = 0) -> list[dict[str, Any]]:
        results = self.get_documents(
            limit=max(1, int(limit)),
            offset=max(0, int(offset)),
            include_ids=True,
            include_documents=True,
            include_metadatas=True,
            include_embeddings=False,
        )
        ids = [str(item) for item in (results.get("ids") or [])]
        documents = list(results.get("documents") or [])
        metadatas = [dict(item or {}) for item in (results.get("metadatas") or [])]
        rows: list[dict[str, Any]] = []
        for index, doc_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            document = str(documents[index] if index < len(documents) else "")
            source_hash = str(metadata.get("source_content_hash") or metadata.get("content_hash") or "").strip()
            computed_hash = source_hash or source_hash_from_content(
                content=document,
                metadata=metadata,
                identity=self._source_metadata_identity(doc_id=doc_id, metadata=metadata),
            )
            rows.append(
                {
                    "id": doc_id,
                    "document": document,
                    "metadata": metadata,
                    "source_content_hash": source_hash,
                    "computed_source_content_hash": computed_hash,
                    "has_source_content_hash": bool(source_hash),
                    "has_stale": "stale" in metadata,
                    "source_type": str(metadata.get("source_type") or ""),
                    "title": str(metadata.get("title") or ""),
                    "document_id": str(metadata.get("document_id") or ""),
                    "file_path": str(metadata.get("file_path") or ""),
                }
            )
        return rows

    def update_metadata_by_id(self, metadata_by_id: dict[str, dict[str, Any]]) -> int:
        clean_items = {
            str(doc_id): dict(metadata or {})
            for doc_id, metadata in dict(metadata_by_id or {}).items()
            if str(doc_id or "").strip()
        }
        if not clean_items:
            return 0
        ids = list(clean_items.keys())
        metadatas = [
            {key: VectorDatabase._encode_metadata_value(value) for key, value in metadata.items()}
            for metadata in clean_items.values()
        ]
        with self._collection_lock():
            self.collection.update(ids=ids, metadatas=metadatas)
            if self.lexical_conn is not None:
                for doc_id, metadata in clean_items.items():
                    self.lexical_conn.execute(
                        """INSERT INTO lexical_documents_meta(doc_id, metadata_json)
                           VALUES (?, ?)
                           ON CONFLICT(doc_id) DO UPDATE SET metadata_json = excluded.metadata_json""",
                        (doc_id, json.dumps(metadata or {}, ensure_ascii=False, default=str)),
                    )
                self.lexical_conn.commit()
        return len(ids)

    def has_metadata(self, filter_dict: dict[str, Any]) -> bool:
        if not filter_dict:
            return False
        normalized = self._normalize_where_filter(filter_dict)
        where_variants = [where for where in [filter_dict, normalized] if where]
        for where in where_variants:
            try:
                result = self.collection.get(where=where, limit=1)
                ids = result.get("ids") if isinstance(result, dict) else None
                if ids:
                    return True
            except Exception:
                continue
        return False

    def lexical_search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: dict[str, Any] | None = None,
        include_stale: bool = False,
    ) -> list[dict[str, Any]]:
        token = self._build_fts5_match_query(query)
        if not token:
            return []
        try:
            if self.lexical_conn is None:
                return []
            rows = self.lexical_conn.execute(
                """
                SELECT f.doc_id, f.document, m.metadata_json, bm25(lexical_documents_fts) AS rank
                FROM lexical_documents_fts AS f
                JOIN lexical_documents_meta AS m ON m.doc_id = f.doc_id
                WHERE lexical_documents_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (token, max(1, int(top_k * 8))),
            ).fetchall()
        except sqlite3.OperationalError as error:
            log.warning("FTS5 lexical search failed for '%s': %s", self.collection_name, error)
            return []

        transformed_scores: list[float] = []
        for row in rows:
            raw_rank = float(row["rank"] or 0.0)
            transformed_scores.append(max(0.0, -raw_rank))
        max_score = max(transformed_scores, default=0.0)
        min_score = min(transformed_scores, default=0.0)

        results: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except Exception:
                metadata = {}
            if not include_stale and self._metadata_is_stale(metadata):
                continue
            if not self._metadata_matches_filter(metadata, filter_dict):
                continue
            raw_rank = float(row["rank"] or 0.0)
            transformed = transformed_scores[idx] if idx < len(transformed_scores) else max(0.0, -raw_rank)
            if max_score > min_score:
                score = (transformed - min_score) / (max_score - min_score)
            else:
                score = 1.0 if transformed > 0.0 else 0.0
            results.append(
                {
                    "id": str(row["doc_id"]),
                    "document": str(row["document"] or ""),
                    "metadata": metadata,
                    "score": score,
                    "rank": raw_rank,
                }
            )
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _build_fts5_match_query(query: str) -> str:
        raw = str(query or "").strip().lower()
        if not raw:
            return ""
        normalized = re.sub(r'["]+', " ", raw)
        normalized = re.sub(r"[-:]+", " ", normalized)
        tokens = re.findall(r"[0-9a-zA-Z가-힣]+", normalized)
        if not tokens:
            return ""
        return " AND ".join(f'"{token}"' for token in tokens)

    def delete_by_id(self, doc_ids: list[str]):
        self.collection.delete(ids=doc_ids)
        with self._collection_lock():
            if self.lexical_conn is not None:
                for doc_id in doc_ids:
                    self.lexical_conn.execute("DELETE FROM lexical_documents_fts WHERE doc_id = ?", (doc_id,))
                    self.lexical_conn.execute("DELETE FROM lexical_documents_meta WHERE doc_id = ?", (doc_id,))
                self.lexical_conn.commit()

    def delete_by_metadata(self, filter_dict: dict[str, Any], *, limit: int = 100000) -> int:
        if not filter_dict:
            return 0
        try:
            results = self.collection.get(
                where=self._normalize_where_filter(filter_dict),
                limit=max(1, int(limit)),
                include=[],
            )
        except Exception:
            return 0
        ids = [str(item) for item in (results.get("ids") or []) if str(item)]
        if not ids:
            return 0
        self.delete_by_id(ids)
        return len(ids)

    def clear_collection(self):
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()
        if self.lexical_conn is not None:
            self.lexical_conn.execute("DELETE FROM lexical_documents_fts")
            self.lexical_conn.execute("DELETE FROM lexical_documents_meta")
            self.lexical_conn.commit()

    def count(self) -> int:
        return self.collection.count()

    def get_stats(self) -> dict[str, Any]:
        count = self.count()
        stats = {
            "collection_name": self.collection_name,
            "total_documents": count,
            "db_path": str(self.db_path),
            "lexical_db_path": str(self.lexical_db_path),
        }
        if count > 0:
            try:
                sample = self.collection.peek(limit=1)
                if sample and sample["metadatas"]:
                    stats["metadata_keys"] = list(sample["metadatas"][0].keys())
            except Exception as error:
                log.warning("Failed to peek sample metadata for '%s': %s", self.collection_name, error)
        return stats


__all__ = [
    "VectorDatabase",
    "VectorDatabaseInitError",
    "inspect_vector_store",
    "list_vector_backups",
    "list_vector_documents",
    "compare_vector_stores",
    "probe_vector_store_openability",
]


def probe_vector_store_openability(db_path: str | Path, collection_name: str = "") -> dict[str, Any]:
    """Verify that a vector store can be opened without repair or collection creation."""

    root = Path(db_path).expanduser()
    collection = str(collection_name or "").strip()
    if not root.exists():
        return {
            "openable": False,
            "error": f"vector store does not exist: {root}",
        }
    if not collection:
        return {
            "openable": False,
            "error": "vector collection missing",
        }

    probe_code = """
import sys
import chromadb
from chromadb.config import Settings

path = sys.argv[1]
collection = sys.argv[2]
try:
    client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
    client.get_collection(name=collection)
except BaseException as exc:
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        raise
    sys.stderr.write(str(exc))
    sys.exit(1)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe_code, str(root), collection],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        return {
            "openable": False,
            "error": f"timed out while probing vector store openability: {root}",
        }
    except Exception as error:
        return {
            "openable": False,
            "error": str(error),
        }
    if result.returncode != 0:
        error = str((result.stderr or result.stdout).strip() or f"failed to open vector store: {root}")
        lines = [line.strip() for line in error.splitlines() if line.strip()]
        if lines:
            error = lines[-1]
        return {
            "openable": False,
            "error": error,
        }

    return {
        "openable": True,
        "error": "",
    }
