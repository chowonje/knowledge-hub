"""Audit and backfill source lifecycle metadata on vector rows."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from knowledge_hub.infrastructure.persistence import VectorDatabase


def _metadata_has_stale(metadata: dict[str, Any]) -> bool:
    return "stale" in dict(metadata or {})


def _source_hash(metadata: dict[str, Any]) -> str:
    for key in ("source_content_hash", "content_hash", "contentHash", "content_sha1", "content_sha256"):
        token = str(dict(metadata or {}).get(key) or "").strip()
        if token:
            return token
    return ""


def audit_vector_source_metadata(
    *,
    vector_db: VectorDatabase,
    limit: int = 10000,
    sample_limit: int = 10,
    apply: bool = False,
) -> dict[str, Any]:
    row_limit = max(1, int(limit))
    sample_cap = max(0, int(sample_limit))
    rows = vector_db.source_metadata_rows(limit=row_limit)
    updates: dict[str, dict[str, Any]] = {}
    missing_samples: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    missing_hash_count = 0
    missing_stale_count = 0
    unable_to_hash_count = 0

    for row in rows:
        metadata = dict(row.get("metadata") or {})
        doc_id = str(row.get("id") or "")
        source_type = str(row.get("source_type") or "<empty>")
        source_counts[source_type] += 1

        has_hash = bool(_source_hash(metadata))
        has_stale = _metadata_has_stale(metadata)
        if not has_hash:
            missing_hash_count += 1
        if not has_stale:
            missing_stale_count += 1

        computed_hash = str(row.get("computed_source_content_hash") or "").strip()
        if not has_hash and not computed_hash:
            unable_to_hash_count += 1

        patch_needed = False
        next_metadata = dict(metadata)
        if not has_hash and computed_hash:
            next_metadata["source_content_hash"] = computed_hash
            patch_needed = True
        if not has_stale:
            next_metadata["stale"] = 0
            patch_needed = True
        if patch_needed and doc_id:
            updates[doc_id] = next_metadata
            if len(missing_samples) < sample_cap:
                missing_samples.append(
                    {
                        "id": doc_id,
                        "sourceType": source_type if source_type != "<empty>" else "",
                        "title": str(row.get("title") or ""),
                        "documentId": str(row.get("document_id") or ""),
                        "filePath": str(row.get("file_path") or ""),
                        "missingSourceContentHash": not has_hash,
                        "missingStale": not has_stale,
                    }
                )

    updated_count = 0
    if apply and updates:
        updated_count = int(vector_db.update_metadata_by_id(updates) or 0)

    warnings: list[str] = []
    if len(rows) >= row_limit:
        warnings.append("scan_limit_reached")
    if unable_to_hash_count:
        warnings.append("some_rows_could_not_compute_source_content_hash")

    status = "ok"
    if unable_to_hash_count and apply:
        status = "partial"

    return {
        "schema": "knowledge-hub.vector-source-metadata.result.v1",
        "status": status,
        "dryRun": not bool(apply),
        "applied": bool(apply),
        "scannedCount": len(rows),
        "updateCandidateCount": len(updates),
        "updatedCount": updated_count,
        "missingSourceContentHashCount": missing_hash_count,
        "missingStaleCount": missing_stale_count,
        "unableToHashCount": unable_to_hash_count,
        "sourceTypeCounts": dict(source_counts.most_common()),
        "sampleMissing": missing_samples,
        "warnings": warnings,
        "checkedAt": datetime.now(timezone.utc).isoformat(),
    }


def audit_vector_source_metadata_for_config(
    *,
    config: Any,
    limit: int = 10000,
    sample_limit: int = 10,
    apply: bool = False,
) -> dict[str, Any]:
    vector_db = VectorDatabase(
        config.vector_db_path,
        config.collection_name,
        repair_on_init=False,
    )
    return audit_vector_source_metadata(
        vector_db=vector_db,
        limit=limit,
        sample_limit=sample_limit,
        apply=apply,
    )


__all__ = ["audit_vector_source_metadata", "audit_vector_source_metadata_for_config"]
