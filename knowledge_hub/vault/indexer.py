"""
Vault 인덱서

Obsidian vault를 파싱하고 벡터 DB + SQLite에 인덱싱합니다.
"""

import time
from pathlib import Path
from typing import Optional
from rich.console import Console

from knowledge_hub.infrastructure.config import Config
from knowledge_hub.infrastructure.persistence import VectorDatabase, SQLiteDatabase
from knowledge_hub.infrastructure.persistence.stores.derivative_lifecycle import (
    mark_derivatives_stale_for_document,
    source_hash_from_content,
)
from knowledge_hub.providers.base import BaseEmbedder
from knowledge_hub.core.models import SourceType
from knowledge_hub.vault.parser import ObsidianParser

console = Console()


class VaultIndexer:
    """Obsidian vault 인덱싱"""

    def __init__(
        self,
        config: Config,
        vector_db: VectorDatabase,
        sqlite_db: SQLiteDatabase,
        embedder: BaseEmbedder,
    ):
        self.config = config
        self.vector_db = vector_db
        self.sqlite_db = sqlite_db
        self.embedder = embedder

    @staticmethod
    def _split_parts(batch: list[dict]) -> list[list[dict]]:
        n = len(batch)
        if n <= 1:
            return [batch]
        if n > 25:
            return [batch[i : i + 25] for i in range(0, n, 25)]
        if n > 10:
            return [batch[i : i + 10] for i in range(0, n, 10)]
        return [[item] for item in batch]

    def _index_chunk_batch(
        self,
        batch: list[dict],
        failures: list[dict],
        warnings: list[dict] | None = None,
    ) -> tuple[int, dict[str, int]]:
        if not batch:
            return 0, {
                "providerRetries": 0,
                "batchRetries": 0,
                "isolatedRetries": 0,
                "totalAdaptiveRetries": 0,
            }

        warning_list = warnings if warnings is not None else []

        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        ids = [
            f"{c['metadata']['file_path']}_{c['chunk_index']}" for c in batch
        ]

        try:
            embeddings = self.embedder.embed_batch(texts, show_progress=False)
            provider_status = {}
            if hasattr(self.embedder, "get_last_status"):
                try:
                    provider_status = self.embedder.get_last_status() or {}
                except Exception:
                    provider_status = {}

            retries = int(provider_status.get("retries", 0)) if isinstance(provider_status, dict) else 0
            provider_failures = provider_status.get("failures", []) if isinstance(provider_status, dict) else []
            for item in provider_failures if isinstance(provider_failures, list) else []:
                warning_list.append(
                    {
                        "stage": "vault.embedder",
                        "errorCode": str(item.get("errorCode", "EMBEDDER_WARNING")),
                        "message": str(item.get("message", "")),
                        "file": "",
                    }
                )

            valid = [
                (t, e, m, doc_id)
                for t, e, m, doc_id in zip(texts, embeddings, metadatas, ids)
                if e is not None
            ]
            for metadata, emb in zip(metadatas, embeddings):
                if emb is None:
                    failures.append(
                        {
                            "stage": "vault.embed",
                            "errorCode": "EMBEDDING_NONE",
                            "message": "embedder returned None",
                            "file": str(metadata.get("file_path", "")),
                        }
                    )
            if valid:
                v_texts, v_embeds, v_metas, v_ids = zip(*valid)
                self.vector_db.add_documents(
                    documents=list(v_texts),
                    embeddings=list(v_embeds),
                    metadatas=list(v_metas),
                    ids=list(v_ids),
                )
            return len(valid), {
                "providerRetries": retries,
                "batchRetries": 0,
                "isolatedRetries": 0,
                "totalAdaptiveRetries": 0,
            }
        except Exception as error:
            if len(batch) <= 1:
                metadata = batch[0].get("metadata", {}) if batch else {}
                failures.append(
                    {
                        "stage": "vault.embed",
                        "errorCode": "EMBEDDING_FAILED",
                        "message": str(error),
                        "file": str(metadata.get("file_path", "")),
                    }
                )
                return 0, {
                    "providerRetries": 0,
                    "batchRetries": 0,
                    "isolatedRetries": 1,
                    "totalAdaptiveRetries": 1,
                }

            split = self._split_parts(batch)
            total_success = 0
            retry_diagnostics = {
                "providerRetries": 0,
                "batchRetries": 1,
                "isolatedRetries": 0,
                "totalAdaptiveRetries": 1,
            }
            for part in split:
                part_success, part_retries = self._index_chunk_batch(part, failures, warning_list)
                total_success += part_success
                for key, value in part_retries.items():
                    retry_diagnostics[key] = retry_diagnostics.get(key, 0) + int(value or 0)
            return total_success, retry_diagnostics

    def _clear_existing_vault_vectors(self) -> int:
        existing = self.vector_db.collection.get(where={"source_type": "vault"}, include=[])
        ids = existing.get("ids", []) if isinstance(existing, dict) else []
        if not ids:
            return 0
        self.vector_db.delete_by_id(list(ids))
        return len(ids)

    @staticmethod
    def _normalize_link_key(value: str) -> str:
        token = str(value or "").strip()
        if not token:
            return ""
        token = token.split("|", 1)[0].split("#", 1)[0].split("^", 1)[0].strip()
        token = Path(token).as_posix()
        if token.lower().endswith(".md"):
            token = token[:-3]
        return token.lower().strip()

    def _build_link_lookup(self, documents: list) -> dict[str, str]:
        candidates: dict[str, set[str]] = {}

        def register(key: str, note_id: str) -> None:
            clean = self._normalize_link_key(key)
            if not clean or not note_id:
                return
            candidates.setdefault(clean, set()).add(note_id)

        for doc in documents:
            note_id = str(doc.file_path or "").strip()
            if not note_id:
                continue
            register(note_id, note_id)
            register(Path(note_id).with_suffix("").as_posix(), note_id)
            register(Path(note_id).stem, note_id)
            register(doc.title, note_id)

        return {
            key: next(iter(values))
            for key, values in candidates.items()
            if len(values) == 1
        }

    def _resolve_links(self, links: list[str], link_lookup: dict[str, str]) -> list[str]:
        resolved: list[str] = []
        seen: set[str] = set()
        for raw in links or []:
            clean = str(raw or "").strip()
            if not clean:
                continue
            target = link_lookup.get(self._normalize_link_key(clean), clean)
            if target in seen:
                continue
            seen.add(target)
            resolved.append(target)
        return resolved

    def _prune_stale_vault_notes(self, live_note_ids: set[str]) -> int:
        rows = self.sqlite_db.list_notes(source_type=SourceType.VAULT.value, limit=1_000_000)
        stale_ids = [
            str(row.get("id") or "").strip()
            for row in rows
            if str(row.get("id") or "").strip() and str(row.get("id") or "").strip() not in live_note_ids
        ]
        return self.sqlite_db.delete_notes(stale_ids)

    def index(self, vault_path: Optional[str] = None, clear: bool = False, authoritative: bool = False) -> dict:
        """vault 전체를 인덱싱"""
        started = time.time()
        failures: list[dict] = []
        warnings: list[dict] = []
        path = vault_path or self.config.vault_path
        if not path:
            console.print("[red]vault 경로가 설정되지 않았습니다. config.yaml을 확인하세요.[/red]")
            return {
                "status": "error",
                "processed": 0,
                "succeeded": 0,
                "failed": 0,
                "retries": 0,
                "warnings": [],
                "retryDiagnostics": {
                    "providerRetries": 0,
                    "batchRetries": 0,
                    "isolatedRetries": 0,
                    "totalAdaptiveRetries": 0,
                },
                "failures": [{"stage": "vault.path", "errorCode": "MISSING_VAULT_PATH", "message": "vault path missing", "file": ""}],
                "durationSec": round(time.time() - started, 3),
                "throughputChunksPerMin": 0.0,
            }

        parser = ObsidianParser(
            vault_path=path,
            exclude_folders=self.config.vault_excludes,
        )

        cleared_vectors = 0
        if clear:
            console.print("[yellow]벡터 DB 초기화 중...[/yellow]")
            self.vector_db.clear_collection()
        elif authoritative:
            cleared_vectors = self._clear_existing_vault_vectors()
            if cleared_vectors > 0:
                console.print(f"[dim]기존 vault 벡터 {cleared_vectors}개 삭제[/dim]")

        console.print(f"[cyan]vault 파싱 중: {path}[/cyan]")
        documents = parser.parse_vault()
        replaced_vectors = 0
        for doc in documents:
            source_hash = source_hash_from_content(
                content=doc.content,
                metadata=dict(doc.metadata or {}),
                identity=doc.file_path,
            )
            if source_hash:
                doc.metadata = {**dict(doc.metadata or {}), "source_content_hash": source_hash, "stale": 0}
                mark_derivatives_stale_for_document(
                    self.sqlite_db.conn,
                    document_id=str(doc.file_path or "").strip(),
                    source_content_hash=source_hash,
                    source_type=SourceType.VAULT.value,
                )
            delete_by_metadata = getattr(self.vector_db, "delete_by_metadata", None)
            if callable(delete_by_metadata) and str(doc.file_path or "").strip():
                replaced_vectors += int(
                    delete_by_metadata(
                        {
                            "source_type": SourceType.VAULT.value,
                            "file_path": str(doc.file_path or "").strip(),
                        }
                    )
                    or 0
                )
        for parse_error in getattr(parser, "last_errors", []) or []:
            failures.append(
                {
                    "stage": "vault.parse",
                    "errorCode": "PARSE_FAILED",
                    "message": str(parse_error.get("message", "")),
                    "file": str(parse_error.get("file_path", "")),
                }
            )
        console.print(f"[green]{len(documents)}개 문서 발견[/green]")

        live_note_ids = {str(doc.file_path or "").strip() for doc in documents if str(doc.file_path or "").strip()}
        link_lookup = self._build_link_lookup(documents)
        pruned_notes = self._prune_stale_vault_notes(live_note_ids) if authoritative else 0

        if not documents:
            console.print("[yellow]인덱싱할 문서가 없습니다.[/yellow]")
            return {
                "status": "ok",
                "processed": 0,
                "succeeded": 0,
                "failed": len(failures),
                "retries": 0,
                "warnings": warnings,
                "retryDiagnostics": {
                    "providerRetries": 0,
                    "batchRetries": 0,
                    "isolatedRetries": 0,
                    "totalAdaptiveRetries": 0,
                },
                "failures": failures,
                "durationSec": round(time.time() - started, 3),
                "throughputChunksPerMin": 0.0,
                "clearedVectors": cleared_vectors,
                "replacedVectors": replaced_vectors,
                "prunedNotes": pruned_notes,
            }

        all_chunks = []
        for doc in documents:
            chunks = ObsidianParser.chunk_document(
                doc,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            all_chunks.extend(chunks)

        console.print(f"[cyan]총 {len(all_chunks)}개 청크 임베딩 생성 중...[/cyan]")

        batch_size = 50
        succeeded_chunks = 0
        retries = 0
        retry_diagnostics = {
            "providerRetries": 0,
            "batchRetries": 0,
            "isolatedRetries": 0,
            "totalAdaptiveRetries": 0,
        }
        processed_chunks = 0
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            batch_success, batch_retry_info = self._index_chunk_batch(batch, failures, warnings)
            succeeded_chunks += batch_success
            retries += sum(int(batch_retry_info.get(key, 0) or 0) for key in batch_retry_info)
            for key, value in batch_retry_info.items():
                retry_diagnostics[key] = retry_diagnostics.get(key, 0) + int(value or 0)
            processed_chunks += len(batch)
            failed_chunks = processed_chunks - succeeded_chunks

            elapsed = max(0.001, time.time() - started)
            chunks_per_min = (processed_chunks / elapsed) * 60.0
            remaining = max(0, len(all_chunks) - processed_chunks)
            eta_sec = (remaining / max(0.001, chunks_per_min / 60.0)) if chunks_per_min > 0 else 0.0
            console.print(
                f"  [{processed_chunks}/{len(all_chunks)}] "
                f"성공={succeeded_chunks} 실패={failed_chunks} "
                f"속도={chunks_per_min:.1f}chunk/min ETA={eta_sec/60.0:.1f}m"
            )

        # SQLite에도 문서 메타데이터 저장
        for doc in documents:
            resolved_links = self._resolve_links(list(doc.links), link_lookup)
            self.sqlite_db.upsert_note(
                note_id=doc.file_path,
                title=doc.title,
                content=doc.content[:500],
                file_path=doc.file_path,
                source_type=SourceType.VAULT.value,
                metadata=doc.metadata,
            )
            self.sqlite_db.replace_note_tags(doc.file_path, list(doc.tags))
            self.sqlite_db.replace_links_for_source(doc.file_path, resolved_links, "wiki_link")

        total = self.vector_db.count()
        console.print(f"\n[bold green]인덱싱 완료! 총 {total}개 청크가 벡터 DB에 저장됨[/bold green]")
        duration = max(0.001, time.time() - started)
        failed_chunks = max(0, len(all_chunks) - succeeded_chunks)
        status = "ok" if failed_chunks == 0 else "partial"
        return {
            "status": status,
            "processed": len(all_chunks),
            "succeeded": succeeded_chunks,
            "failed": failed_chunks,
            "retries": retries,
            "warnings": warnings,
            "retryDiagnostics": retry_diagnostics,
            "failures": failures,
            "durationSec": round(duration, 3),
            "throughputChunksPerMin": round((len(all_chunks) / duration) * 60.0, 3),
            "documents": len(documents),
            "parseErrors": len(getattr(parser, "last_errors", []) or []),
            "clearedVectors": cleared_vectors,
            "replacedVectors": replaced_vectors,
            "prunedNotes": pruned_notes,
        }
