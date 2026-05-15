"""Application helpers for persisted evidence substrate URI records."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Iterable


REGISTRY_LOOKUP_SCHEMA = "knowledge-hub.evidence-registry.lookup.result.v1"
REGISTRY_WRITE_SCHEMA = "knowledge-hub.evidence-registry.write.result.v1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_json_bytes(value)).hexdigest()}"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _walk_objects(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_objects(child)


def _first_key(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _source_id_from_item(item: dict[str, Any]) -> str:
    direct = _first_key(item, "source_id", "sourceId", "source_ref", "sourceRef")
    if direct:
        return direct
    target = _first_key(item, "target")
    if target and any(
        key in item
        for key in (
            "label",
            "kind",
            "source_type",
            "sourceType",
            "content_hash",
            "contentHash",
            "source_content_hash",
            "sourceContentHash",
        )
    ):
        return target
    return ""


def build_lineage(payload: dict[str, Any]) -> dict[str, Any]:
    source_ids: set[str] = set()
    document_ids: set[str] = set()
    chunk_ids: set[str] = set()
    claim_ids: set[str] = set()
    evidence_span_ids: set[str] = set()
    citation_labels: set[str] = set()

    for item in _walk_objects(payload):
        source_id = _source_id_from_item(item)
        if source_id:
            source_ids.add(source_id)
        document_id = _first_key(item, "document_id", "documentId", "doc_id", "docId")
        if document_id:
            document_ids.add(document_id)
        chunk_id = _first_key(item, "chunk_id", "chunkId", "retrieval_unit_id", "retrievalUnitId")
        if chunk_id:
            chunk_ids.add(chunk_id)
        claim_id = _first_key(item, "claim_id", "claimId")
        if claim_id:
            claim_ids.add(claim_id)
        span_id = _first_key(item, "span_id", "spanId", "spanRef", "span_ref")
        if span_id:
            evidence_span_ids.add(span_id)
        label = _first_key(item, "label", "citation_label", "citationLabel")
        if label:
            citation_labels.add(label)

    return {
        "sourceIds": sorted(source_ids),
        "documentIds": sorted(document_ids),
        "chunkIds": sorted(chunk_ids),
        "claimIds": sorted(claim_ids),
        "evidenceSpanIds": sorted(evidence_span_ids),
        "citationLabels": sorted(citation_labels),
    }


def extract_source_refs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for item in _walk_objects(payload):
        source_id = _source_id_from_item(item)
        if not source_id:
            continue
        content_hash = _first_key(
            item,
            "source_content_hash",
            "sourceContentHash",
            "content_hash",
            "contentHash",
            "source_revision",
            "sourceRevision",
        )
        existing = refs.setdefault(
            source_id,
            {
                "sourceId": source_id,
                "sourceContentHash": content_hash,
                "sourceType": _first_key(item, "source_type", "sourceType", "kind"),
                "documentId": _first_key(item, "document_id", "documentId", "doc_id", "docId"),
                "chunkId": _first_key(item, "chunk_id", "chunkId", "retrieval_unit_id", "retrievalUnitId"),
                "spanId": _first_key(item, "span_id", "spanId", "spanRef", "span_ref"),
            },
        )
        if content_hash and not existing.get("sourceContentHash"):
            existing["sourceContentHash"] = content_hash
    return list(refs.values())


def source_revision_hash(source_refs: list[dict[str, Any]]) -> str:
    bound_refs = [
        {
            "sourceId": _first_key(ref, "sourceId", "source_id"),
            "sourceContentHash": _first_key(ref, "sourceContentHash", "source_content_hash", "contentHash", "content_hash"),
        }
        for ref in source_refs
        if _first_key(ref, "sourceId", "source_id") and _first_key(ref, "sourceContentHash", "source_content_hash", "contentHash", "content_hash")
    ]
    if not bound_refs:
        return ""
    return _sha256_json(sorted(bound_refs, key=lambda item: item["sourceId"]))


def default_registry_authority() -> dict[str, Any]:
    return {
        "role": "derived_lookup_registry",
        "sourceOfTruth": "source ledger, normalized documents, and source-backed chunks",
        "payloadIsSourceOfTruth": False,
        "embeddingIsSourceOfTruth": False,
        "requiresSourceRevisionMatch": True,
        "deletionPolicy": "Deleting a registry record removes only the lookup projection; source records and indexes remain untouched.",
    }


def _estimate_token_count(payload: dict[str, Any]) -> int:
    # Conservative, dependency-free estimate for a registry guard. Runtime pack
    # builders still own precise context selection and character budgets.
    return max(1, len(_json_bytes(payload)) // 4)


def _record_id_from_payload(payload: dict[str, Any], *keys: str, prefix: str) -> str:
    for key in keys:
        value = _clean_text(payload.get(key))
        if value:
            return value
    return f"{prefix}_{hashlib.sha256(_json_bytes(payload)).hexdigest()[:16]}"


def _store(sqlite_db: Any) -> Any:
    store = getattr(sqlite_db, "evidence_registry_store", None)
    if store is not None:
        return store
    registry = getattr(sqlite_db, "registry", None)
    if registry is not None:
        store = getattr(registry, "evidence_registry_store", None)
        if store is not None:
            return store
    return sqlite_db


def register_packet(
    sqlite_db: Any,
    packet_payload: dict[str, Any],
    *,
    registry_id: str = "",
    expires_at: str = "",
    status: str = "ok",
) -> dict[str, Any]:
    packet_id = registry_id or _record_id_from_payload(packet_payload, "packet_id", "packetId", "id", prefix="epkt")
    refs = extract_source_refs(packet_payload)
    lineage = build_lineage(packet_payload)
    return _store(sqlite_db).upsert_record(
        registry_id=packet_id,
        record_kind="packet",
        payload=packet_payload,
        payload_schema=_clean_text(packet_payload.get("schema")),
        status=status,
        source_revision_hash=source_revision_hash(refs),
        source_refs=refs,
        lineage=lineage,
        authority=default_registry_authority(),
        token_count=_estimate_token_count(packet_payload),
        expires_at=expires_at,
        deletion_policy=default_registry_authority()["deletionPolicy"],
    )


def register_context_pack(
    sqlite_db: Any,
    context_pack: dict[str, Any],
    *,
    registry_id: str = "",
    expires_at: str = "",
    max_tokens: int | None = None,
) -> dict[str, Any]:
    context_id = registry_id or _record_id_from_payload(
        context_pack,
        "contextPackId",
        "context_pack_id",
        "context_id",
        "id",
        prefix="ctx",
    )
    token_count = _estimate_token_count(context_pack)
    if max_tokens is not None and token_count > int(max_tokens):
        raise ValueError(f"context pack exceeds token budget: estimated={token_count} max={int(max_tokens)}")
    refs = extract_source_refs(context_pack)
    return _store(sqlite_db).upsert_record(
        registry_id=context_id,
        record_kind="context",
        payload=context_pack,
        payload_schema=_clean_text(context_pack.get("schema")),
        status="ok",
        source_revision_hash=source_revision_hash(refs),
        source_refs=refs,
        lineage=build_lineage(context_pack),
        authority=default_registry_authority(),
        token_count=token_count,
        expires_at=expires_at,
        deletion_policy=default_registry_authority()["deletionPolicy"],
    )


def register_answer_trace(
    sqlite_db: Any,
    *,
    answer_payload: dict[str, Any],
    trace_payload: dict[str, Any],
    expires_at: str = "",
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    evidence_packet = dict(answer_payload.get("evidencePacketContract") or {})
    if evidence_packet:
        records.append(register_packet(sqlite_db, evidence_packet, expires_at=expires_at))
    else:
        warnings.append("answer payload has no evidencePacketContract to persist")

    trace_id = _clean_text(trace_payload.get("answerId")) or _record_id_from_payload(
        trace_payload,
        "answerId",
        "answer_id",
        "id",
        prefix="atr",
    )
    trace_refs = extract_source_refs(trace_payload)
    records.append(
        _store(sqlite_db).upsert_record(
            registry_id=trace_id,
            record_kind="trace",
            payload=trace_payload,
            payload_schema=_clean_text(trace_payload.get("schema")),
            status=_clean_text(trace_payload.get("status")) or "ok",
            source_revision_hash=source_revision_hash(trace_refs),
            source_refs=trace_refs,
            lineage=build_lineage(trace_payload),
            authority=default_registry_authority(),
            token_count=_estimate_token_count(trace_payload),
            expires_at=expires_at,
            deletion_policy=default_registry_authority()["deletionPolicy"],
        )
    )
    return {
        "schema": REGISTRY_WRITE_SCHEMA,
        "status": "ok" if records else "not_found",
        "records": records,
        "warnings": warnings,
        "createdAt": _now_iso(),
    }


def _current_ref_map(source_refs: list[dict[str, Any]] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for ref in source_refs or []:
        source_id = _first_key(ref, "sourceId", "source_id")
        content_hash = _first_key(ref, "sourceContentHash", "source_content_hash", "contentHash", "content_hash")
        if source_id and content_hash:
            result[source_id] = content_hash
    return result


def _revision_mismatch(record: dict[str, Any], current_source_refs: list[dict[str, Any]] | None) -> bool:
    current = _current_ref_map(current_source_refs)
    if not current:
        return False
    for ref in list(record.get("sourceRefs") or []):
        source_id = _first_key(ref, "sourceId", "source_id")
        recorded_hash = _first_key(ref, "sourceContentHash", "source_content_hash", "contentHash", "content_hash")
        if source_id and recorded_hash and current.get(source_id) and current[source_id] != recorded_hash:
            return True
    return False


def resolve_registry_lookup(
    sqlite_db: Any,
    record_kind: str,
    registry_id: str,
    *,
    current_source_refs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    kind = _clean_text(record_kind).lower()
    identifier = _clean_text(registry_id)
    getter = getattr(_store(sqlite_db), "get_record", None)
    record = getter(kind, identifier) if callable(getter) else None
    if not record:
        return {
            "schema": REGISTRY_LOOKUP_SCHEMA,
            "status": "not_found",
            "resourceKind": kind,
            "identifier": identifier,
            "reason": f"{kind} registry record not found",
            "registryRecord": {},
            "payload": {},
            "authority": default_registry_authority(),
            "lineage": {},
            "sourceRefs": [],
            "warnings": [],
        }

    warnings = []
    status = str(record.get("status") or "ok")
    if _revision_mismatch(record, current_source_refs):
        status = "stale"
        record = dict(record)
        record["status"] = "stale"
        record["stale"] = True
        record["staleReason"] = "source_revision_mismatch"
        warnings.append("registry record source revision does not match current source refs")
    elif bool(record.get("stale")) and record.get("staleReason"):
        warnings.append(str(record.get("staleReason")))

    return {
        "schema": REGISTRY_LOOKUP_SCHEMA,
        "status": status,
        "resourceKind": kind,
        "identifier": identifier,
        "reason": "registry_record_found",
        "registryRecord": record,
        "payload": dict(record.get("payload") or {}),
        "authority": dict(record.get("authority") or default_registry_authority()),
        "lineage": dict(record.get("lineage") or {}),
        "sourceRefs": list(record.get("sourceRefs") or []),
        "warnings": warnings,
    }
