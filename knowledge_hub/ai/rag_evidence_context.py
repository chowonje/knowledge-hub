from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from knowledge_hub.core.models import SearchResult


def collect_claim_context(
    results: list[SearchResult],
    *,
    sqlite_db: Any,
    note_id_for_result_fn: Callable[[SearchResult], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not sqlite_db:
        return [], [], [], []

    claims: list[dict[str, Any]] = []
    claim_ids: list[str] = []
    seen_claims: set[str] = set()
    for result in results:
        note_id = note_id_for_result_fn(result)
        if note_id:
            for claim in sqlite_db.list_claims_by_note(note_id, limit=20):
                claim_id = str(claim.get("claim_id", "")).strip()
                if claim_id and claim_id not in seen_claims:
                    seen_claims.add(claim_id)
                    claim_ids.append(claim_id)
                    claims.append(claim)

        record_id = str((result.metadata or {}).get("record_id", "") or "").strip()
        if record_id:
            for claim in sqlite_db.list_claims_by_record(record_id, limit=20):
                claim_id = str(claim.get("claim_id", "")).strip()
                if claim_id and claim_id not in seen_claims:
                    seen_claims.add(claim_id)
                    claim_ids.append(claim_id)
                    claims.append(claim)

    beliefs = sqlite_db.list_beliefs_by_claim_ids(claim_ids, limit=200) if claim_ids else []
    supporting = [
        item
        for item in beliefs
        if str(item.get("status", "")).strip() != "rejected" and not item.get("contradiction_ids")
    ]
    contradicting = [
        item
        for item in beliefs
        if item.get("contradiction_ids") or str(item.get("status", "")).strip() in {"rejected", "stale"}
    ]
    known_claim_ids = {
        claim_id
        for belief in beliefs
        for claim_id in belief.get("derived_from_claim_ids", [])
        if str(claim_id).strip()
    }
    suggested = [
        {
            "belief_id": f"suggested_{str(claim.get('claim_id', ''))}",
            "statement": str(claim.get("claim_text", "")).strip(),
            "derived_from_claim_ids": [str(claim.get("claim_id", "")).strip()],
            "scope": "query",
            "status": "proposed",
            "confidence": float(claim.get("confidence", 0.5) or 0.5),
        }
        for claim in claims
        if str(claim.get("claim_id", "")).strip() and str(claim.get("claim_id", "")).strip() not in known_claim_ids
    ][:5]
    return claims, supporting[:10], contradicting[:10], suggested


def resolve_result_note_row(
    result: SearchResult,
    *,
    sqlite_db: Any,
    note_id_for_result_fn: Callable[[SearchResult], str],
) -> dict[str, Any] | None:
    if not sqlite_db:
        return None

    getter = getattr(sqlite_db, "get_note", None)
    if not callable(getter):
        return None

    note_id = note_id_for_result_fn(result)
    if note_id:
        try:
            note = getter(note_id)
        except Exception:
            note = None
        if isinstance(note, dict):
            return note
    return None


def resolve_result_quality(
    result: SearchResult,
    *,
    sqlite_db: Any,
    config: Any,
    resolve_result_note_row_fn: Callable[[SearchResult], dict[str, Any] | None],
    json_load_dict_fn: Callable[[Any], dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if not sqlite_db:
        return "unscored", {}

    note_row = resolve_result_note_row_fn(result)
    note_meta = json_load_dict_fn((note_row or {}).get("metadata"))
    direct_quality = json_load_dict_fn(note_meta.get("quality"))
    if direct_quality:
        flag = str(direct_quality.get("flag") or "").strip().lower() or "unscored"
        return flag, direct_quality

    finder = getattr(sqlite_db, "find_ko_note_item_by_final_path", None)
    if not callable(finder):
        return "unscored", {}

    metadata = result.metadata or {}
    file_path = str(metadata.get("file_path") or "").strip()
    candidates: list[str] = []
    if file_path:
        candidates.append(file_path)
        vault_root = str(getattr(config, "vault_path", "") or "").strip() if config else ""
        if vault_root:
            candidates.append(str((Path(vault_root).expanduser() / file_path).resolve()))

    seen: set[str] = set()
    for final_path in candidates:
        token = str(final_path or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        try:
            item = finder(final_path=token)
        except Exception:
            item = None
        if not isinstance(item, dict):
            continue
        payload = dict(item.get("payload_json") or {})
        quality = json_load_dict_fn(payload.get("quality"))
        if quality:
            flag = str(quality.get("flag") or "").strip().lower() or "unscored"
            return flag, quality
    return "unscored", {}
