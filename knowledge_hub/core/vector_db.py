"""Vector database implementation extracted from sqlite_db facade."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

log = logging.getLogger("khub.database")

try:
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None

SQLITE_BUSY_TIMEOUT_MS = 5000


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

    def __init__(self, db_path: str, collection_name: str = "knowledge_hub"):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.lexical_db_path = self.db_path / "_lexical.sqlite3"
        self.lexical_conn = sqlite3.connect(str(self.lexical_db_path), timeout=SQLITE_BUSY_TIMEOUT_MS / 1000)
        self.lexical_conn.row_factory = sqlite3.Row

        self.client = self._create_client_with_repair()
        self._ef = _NoOpEmbeddingFunction()
        self.collection = self._get_or_create_collection()
        self._init_lexical_index()

    def _init_lexical_index(self) -> None:
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

    def _create_client_with_repair(self):
        settings = Settings(anonymized_telemetry=False)
        try:
            return chromadb.PersistentClient(path=str(self.db_path), settings=settings)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
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
            return self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._ef,
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
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
    ) -> dict[str, Any]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
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
        return results

    def get_documents(
        self,
        filter_dict: dict[str, Any] | None = None,
        limit: int = 500,
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

        request = {"where": self._normalize_where_filter(filter_dict), "limit": max(1, limit), "include": include}
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
    ) -> list[dict[str, Any]]:
        token = self._build_fts5_match_query(query)
        if not token:
            return []
        try:
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
            for doc_id in doc_ids:
                self.lexical_conn.execute("DELETE FROM lexical_documents_fts WHERE doc_id = ?", (doc_id,))
                self.lexical_conn.execute("DELETE FROM lexical_documents_meta WHERE doc_id = ?", (doc_id,))
            self.lexical_conn.commit()

    def clear_collection(self):
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()
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


__all__ = ["VectorDatabase"]
