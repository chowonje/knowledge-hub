from __future__ import annotations

from collections import Counter
from typing import Any

from knowledge_hub.ai.rag_support import json_load_dict, normalize_source_type, note_id_for_result, result_paper_id


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _item_metadata(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item.get("metadata") or {})
    return dict(getattr(item, "metadata", {}) or {})


def _item_source_type(item: Any) -> str:
    metadata = _item_metadata(item)
    if isinstance(item, dict):
        direct = _clean_text(item.get("normalized_source_type"))
        if direct:
            return _clean_text(normalize_source_type(direct) or direct).lower()
    source_type = _clean_text(metadata.get("source_type"))
    return _clean_text(normalize_source_type(source_type) or source_type).lower()


def _item_file_path(item: Any) -> str:
    metadata = _item_metadata(item)
    if isinstance(item, dict):
        return _clean_text(item.get("file_path") or metadata.get("file_path"))
    return _clean_text(metadata.get("file_path"))


def _item_url(item: Any) -> str:
    metadata = _item_metadata(item)
    if isinstance(item, dict):
        return _clean_text(item.get("source_url") or metadata.get("url") or metadata.get("canonical_url"))
    return _clean_text(metadata.get("url") or metadata.get("canonical_url"))


def _item_record_id(item: Any) -> str:
    metadata = _item_metadata(item)
    return _clean_text(metadata.get("record_id"))


def _item_note_hint(item: Any) -> str:
    metadata = _item_metadata(item)
    for key in ("note_id", "resolved_parent_id", "parent_id"):
        token = _clean_text(metadata.get(key))
        if token:
            return token
    if isinstance(item, dict):
        for key in ("note_id", "local_source_id", "stable_scope_id", "document_scope_id"):
            token = _clean_text(item.get(key))
            if token:
                return token
    try:
        token = _clean_text(note_id_for_result(item))
        if token:
            return token
    except Exception:
        pass
    return ""


def _item_paper_id(item: Any) -> str:
    if isinstance(item, dict):
        metadata = _item_metadata(item)
        for key in ("arxiv_id", "paper_id"):
            token = _clean_text(metadata.get(key) or item.get(key))
            if token:
                return token
        local_source_id = _clean_text(item.get("local_source_id"))
        if local_source_id:
            return local_source_id
        return ""
    try:
        return _clean_text(result_paper_id(item))
    except Exception:
        return ""


class _NoteIndex:
    def __init__(self, sqlite_db: Any):
        self.sqlite_db = sqlite_db
        self.loaded = False
        self.by_id: dict[str, str] = {}
        self.by_file_path: dict[str, str] = {}
        self.by_url: dict[str, str] = {}
        self.by_record_id: dict[str, str] = {}

    def resolve(
        self,
        *,
        note_hint: str = "",
        file_path: str = "",
        url: str = "",
        record_id: str = "",
    ) -> str:
        candidates = [note_hint, file_path, url, record_id]
        get_note = getattr(self.sqlite_db, "get_note", None)
        if callable(get_note):
            for candidate in candidates:
                token = _clean_text(candidate)
                if not token:
                    continue
                try:
                    note = get_note(token)
                except Exception:
                    note = None
                if note:
                    return _clean_text(note.get("id"))
        self._load()
        for token in candidates:
            key = _clean_text(token)
            if not key:
                continue
            resolved = (
                self.by_id.get(key)
                or self.by_file_path.get(key)
                or self.by_url.get(key)
                or self.by_record_id.get(key)
            )
            if resolved:
                return resolved
        return ""

    def _load(self) -> None:
        if self.loaded:
            return
        self.loaded = True
        list_notes = getattr(self.sqlite_db, "list_notes", None)
        if not callable(list_notes):
            return
        offset = 0
        page_size = 500
        max_rows = 5000
        while offset < max_rows:
            try:
                rows = list(list_notes(limit=page_size, offset=offset))
            except Exception:
                break
            if not rows:
                break
            for row in rows:
                note_id = _clean_text(row.get("id"))
                if not note_id:
                    continue
                self.by_id.setdefault(note_id, note_id)
                file_path = _clean_text(row.get("file_path"))
                if file_path:
                    self.by_file_path.setdefault(file_path, note_id)
                metadata = json_load_dict(row.get("metadata"))
                url = _clean_text(metadata.get("canonical_url") or metadata.get("url"))
                if url:
                    self.by_url.setdefault(url, note_id)
                record_id = _clean_text(metadata.get("record_id"))
                if record_id:
                    self.by_record_id.setdefault(record_id, note_id)
            offset += len(rows)
            if len(rows) < page_size:
                break


class ClaimSignalBuilder:
    def __init__(self, sqlite_db: Any):
        self.sqlite_db = sqlite_db
        self._note_index = _NoteIndex(sqlite_db)

    def build_for_items(self, items: list[Any]) -> dict[str, Any]:
        entries = [self._entry_for_item(item) for item in items]
        return {
            "summary": self._summarize_entries(entries),
            "items": entries,
        }

    def _entry_for_item(self, item: Any) -> dict[str, Any]:
        source_type = _item_source_type(item)
        scope = self._resolve_scope(item, source_type=source_type)
        claims = self._load_claims(scope)
        normalizations = self._load_normalizations(claims)
        metrics = Counter(
            _clean_text(row.get("metric"))
            for row in normalizations.values()
            if _clean_text(row.get("metric"))
        )
        datasets = Counter(
            _clean_text(row.get("dataset"))
            for row in normalizations.values()
            if _clean_text(row.get("dataset"))
        )
        conflict_count = self._conflict_candidate_count(normalizations)
        claim_ids = [
            _clean_text(row.get("claim_id"))
            for row in claims
            if _clean_text(row.get("claim_id"))
        ]
        return {
            "available": bool(self.sqlite_db),
            "sourceType": source_type,
            "resolution": scope["resolution"],
            "scopeId": scope["scope_id"],
            "scopeType": scope["scope_type"],
            "claimCount": len(claim_ids),
            "normalizedClaimCount": len(normalizations),
            "claimIds": claim_ids[:5],
            "topMetrics": self._counter_payload(metrics),
            "topDatasets": self._counter_payload(datasets),
            "conflictCandidateCount": conflict_count,
        }

    def _resolve_scope(self, item: Any, *, source_type: str) -> dict[str, str]:
        note_hint = _item_note_hint(item)
        file_path = _item_file_path(item)
        url = _item_url(item)
        record_id = _item_record_id(item)
        paper_id = _item_paper_id(item) if source_type == "paper" else ""
        note_id = self._note_index.resolve(
            note_hint=note_hint,
            file_path=file_path,
            url=url,
            record_id=record_id,
        )
        if source_type == "paper" and paper_id and note_id:
            return {
                "resolution": "note+record",
                "scope_id": paper_id,
                "scope_type": "paper",
                "note_id": note_id,
                "record_id": paper_id,
            }
        if source_type == "paper" and paper_id:
            return {
                "resolution": "record",
                "scope_id": paper_id,
                "scope_type": "paper",
                "note_id": "",
                "record_id": paper_id,
            }
        if note_id:
            return {
                "resolution": "note",
                "scope_id": note_id,
                "scope_type": source_type or "note",
                "note_id": note_id,
                "record_id": "",
            }
        if record_id:
            return {
                "resolution": "record",
                "scope_id": record_id,
                "scope_type": source_type or "record",
                "note_id": "",
                "record_id": record_id,
            }
        return {
            "resolution": "unresolved",
            "scope_id": "",
            "scope_type": source_type or "unknown",
            "note_id": "",
            "record_id": "",
        }

    def _load_claims(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        claims: list[dict[str, Any]] = []
        seen: set[str] = set()
        list_claims_by_note = getattr(self.sqlite_db, "list_claims_by_note", None)
        note_id = _clean_text(scope.get("note_id"))
        if note_id and callable(list_claims_by_note):
            try:
                rows = list_claims_by_note(note_id, limit=100)
            except Exception:
                rows = []
            for row in rows or []:
                claim_id = _clean_text(row.get("claim_id"))
                if claim_id and claim_id not in seen:
                    seen.add(claim_id)
                    claims.append(dict(row))
        list_claims_by_record = getattr(self.sqlite_db, "list_claims_by_record", None)
        record_id = _clean_text(scope.get("record_id"))
        if record_id and callable(list_claims_by_record):
            try:
                rows = list_claims_by_record(record_id, limit=100)
            except Exception:
                rows = []
            for row in rows or []:
                claim_id = _clean_text(row.get("claim_id"))
                if claim_id and claim_id not in seen:
                    seen.add(claim_id)
                    claims.append(dict(row))
        claims.sort(
            key=lambda item: (
                float(item.get("confidence", 0.0) or 0.0),
                _clean_text(item.get("created_at")),
            ),
            reverse=True,
        )
        return claims[:20]

    def _load_normalizations(self, claims: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        claim_ids = [_clean_text(item.get("claim_id")) for item in claims if _clean_text(item.get("claim_id"))]
        if not claim_ids:
            return {}
        list_claim_normalizations = getattr(self.sqlite_db, "list_claim_normalizations", None)
        if not callable(list_claim_normalizations):
            return {}
        try:
            rows = list_claim_normalizations(claim_ids=claim_ids, status="normalized", limit=max(20, len(claim_ids) * 2))
        except Exception:
            return {}
        normalizations: dict[str, dict[str, Any]] = {}
        for row in rows or []:
            claim_id = _clean_text(row.get("claim_id"))
            if claim_id and claim_id not in normalizations:
                normalizations[claim_id] = dict(row)
        return normalizations

    def _conflict_candidate_count(self, normalizations: dict[str, dict[str, Any]]) -> int:
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in normalizations.values():
            key = (
                _clean_text(row.get("dataset")),
                _clean_text(row.get("metric")),
                _clean_text(row.get("comparator")),
            )
            if any(key):
                grouped.setdefault(key, []).append(row)
        count = 0
        for rows in grouped.values():
            directions = {
                _clean_text(item.get("result_direction"))
                for item in rows
                if _clean_text(item.get("result_direction")) not in {"", "unknown"}
            }
            numeric_values = {
                item.get("result_value_numeric")
                for item in rows
                if item.get("result_value_numeric") is not None
            }
            if len(directions) > 1 or len(numeric_values) > 1:
                count += 1
        return count

    def _counter_payload(self, counter: Counter[str], *, limit: int = 3) -> list[dict[str, Any]]:
        return [
            {"name": name, "count": count}
            for name, count in counter.most_common(max(1, int(limit)))
            if _clean_text(name)
        ]

    def _summarize_entries(self, entries: list[dict[str, Any]]) -> dict[str, Any]:
        total_claim_count = sum(int(item.get("claimCount") or 0) for item in entries)
        total_normalized_claim_count = sum(int(item.get("normalizedClaimCount") or 0) for item in entries)
        metric_counter = Counter()
        dataset_counter = Counter()
        for item in entries:
            for metric in item.get("topMetrics") or []:
                metric_counter[_clean_text(metric.get("name"))] += int(metric.get("count") or 0)
            for dataset in item.get("topDatasets") or []:
                dataset_counter[_clean_text(dataset.get("name"))] += int(dataset.get("count") or 0)
        return {
            "available": bool(self.sqlite_db),
            "resultCount": len(entries),
            "resultsWithResolvedScope": sum(1 for item in entries if _clean_text(item.get("resolution")) != "unresolved"),
            "resultsWithClaims": sum(1 for item in entries if int(item.get("claimCount") or 0) > 0),
            "totalClaimCount": total_claim_count,
            "totalNormalizedClaimCount": total_normalized_claim_count,
            "conflictCandidateCount": sum(int(item.get("conflictCandidateCount") or 0) for item in entries),
            "topMetrics": self._counter_payload(metric_counter, limit=5),
            "topDatasets": self._counter_payload(dataset_counter, limit=5),
        }


def build_claim_signal_payload(sqlite_db: Any, items: list[Any]) -> dict[str, Any]:
    builder = ClaimSignalBuilder(sqlite_db)
    return builder.build_for_items(list(items or []))


__all__ = ["build_claim_signal_payload", "ClaimSignalBuilder"]
