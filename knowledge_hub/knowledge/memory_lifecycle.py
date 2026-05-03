"""Internal memory lifecycle boundary for paper/document/claim memory forms."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


_SUPPORTED_KINDS = {"paper", "document", "claim"}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


class MemoryLifecycleService:
    """Thin service boundary over existing stores.

    The service does not introduce a new database. It delegates `add/query/update`
    to the current SQLite/artifact stores and keeps archive state in a small
    sidecar manifest so archive semantics stay reversible.
    """

    def __init__(self, sqlite_db: Any, config: Any | None = None):
        self.sqlite_db = sqlite_db
        self.config = config

    def _normalize_kind(self, kind: str) -> str:
        token = str(kind or "").strip().lower()
        if token not in _SUPPORTED_KINDS:
            raise ValueError(f"unsupported memory kind: {kind}")
        return token

    def _archive_root(self) -> Path:
        papers_dir = Path(str(getattr(self.config, "papers_dir", "") or "")).expanduser()
        base = papers_dir if str(papers_dir) else Path.cwd()
        return base / "memory-lifecycle"

    def _archive_path(self) -> Path:
        return self._archive_root() / "archive-state.json"

    def _load_archive_state(self) -> dict[str, Any]:
        path = self._archive_path()
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _write_archive_state(self, payload: dict[str, Any]) -> None:
        path = self._archive_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _archive_key(self, *, kind: str, identifier: str) -> str:
        return f"{kind}:{identifier}"

    def _merge_archive_status(self, *, kind: str, identifier: str, payload: Any) -> Any:
        state = self._load_archive_state()
        archive_info = dict(state.get(self._archive_key(kind=kind, identifier=identifier)) or {})
        if not archive_info:
            return payload
        if isinstance(payload, dict):
            return {**payload, "archiveStatus": archive_info}
        if isinstance(payload, list):
            return [{"item": item, "archiveStatus": archive_info} for item in payload]
        return {"payload": payload, "archiveStatus": archive_info}

    def add(self, *, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        token = self._normalize_kind(kind)
        if token == "paper":
            stored = self.sqlite_db.upsert_paper_memory_card(card=dict(payload or {}))
            paper_id = _clean_text((payload or {}).get("paper_id") or (payload or {}).get("paperId"))
            return self._merge_archive_status(kind=token, identifier=paper_id, payload=stored or {})
        if token == "document":
            document_id = _clean_text((payload or {}).get("document_id") or (payload or {}).get("documentId"))
            units = list((payload or {}).get("units") or [])
            stored = self.sqlite_db.replace_document_memory_units(document_id=document_id, units=units)
            return self._merge_archive_status(kind=token, identifier=document_id, payload={"documentId": document_id, "units": stored})
        claim_id = _clean_text((payload or {}).get("claim_id") or (payload or {}).get("claimId"))
        if _clean_text((payload or {}).get("normalization_version")):
            self.sqlite_db.upsert_claim_normalization(**dict(payload or {}))
            stored = self.sqlite_db.get_claim_normalization(
                claim_id,
                _clean_text((payload or {}).get("normalization_version")),
            )
        else:
            self.sqlite_db.upsert_claim(**dict(payload or {}))
            stored = self.sqlite_db.get_claim(claim_id)
        return self._merge_archive_status(kind=token, identifier=claim_id, payload=stored or {})

    def query(self, *, kind: str, identifier: str = "", query: str = "", limit: int = 20) -> dict[str, Any]:
        token = self._normalize_kind(kind)
        if token == "paper":
            paper_id = _clean_text(identifier)
            if paper_id:
                item = self.sqlite_db.get_paper_memory_card(paper_id) or {}
                return {"kind": token, "items": [self._merge_archive_status(kind=token, identifier=paper_id, payload=item)] if item else []}
            items = list(self.sqlite_db.search_paper_memory_cards(_clean_text(query), limit=max(1, int(limit))) or [])
            return {"kind": token, "items": [self._merge_archive_status(kind=token, identifier=_clean_text(item.get("paper_id")), payload=item) for item in items]}
        if token == "document":
            document_id = _clean_text(identifier)
            if document_id:
                summary = self.sqlite_db.get_document_memory_summary(document_id) or {}
                units = list(self.sqlite_db.list_document_memory_units(document_id, limit=max(1, int(limit))) or [])
                merged = self._merge_archive_status(kind=token, identifier=document_id, payload={"summary": summary, "units": units})
                return {"kind": token, "items": [merged] if summary or units else []}
            return {"kind": token, "items": list(self.sqlite_db.search_document_memory_units(_clean_text(query), limit=max(1, int(limit))) or [])}
        claim_id = _clean_text(identifier)
        if claim_id:
            item = self.sqlite_db.get_claim(claim_id) or {}
            return {"kind": token, "items": [self._merge_archive_status(kind=token, identifier=claim_id, payload=item)] if item else []}
        items = list(self.sqlite_db.list_claims(limit=max(1, int(limit))) or [])
        if query:
            needle = _clean_text(query).casefold()
            items = [item for item in items if needle in _clean_text(item.get("claim_text")).casefold()]
        return {"kind": token, "items": [self._merge_archive_status(kind=token, identifier=_clean_text(item.get("claim_id")), payload=item) for item in items[: max(1, int(limit))]]}

    def update(self, *, kind: str, identifier: str, payload: dict[str, Any]) -> dict[str, Any]:
        token = self._normalize_kind(kind)
        merged = dict(payload or {})
        if token == "paper":
            merged.setdefault("paper_id", identifier)
        elif token == "document":
            merged.setdefault("document_id", identifier)
        else:
            merged.setdefault("claim_id", identifier)
        return self.add(kind=token, payload=merged)

    def archive(self, *, kind: str, identifier: str, reason: str = "") -> dict[str, Any]:
        token = self._normalize_kind(kind)
        item_id = _clean_text(identifier)
        state = self._load_archive_state()
        archive_info = {
            "status": "archived",
            "reason": _clean_text(reason),
            "archivedAt": datetime.now(timezone.utc).isoformat(),
        }
        state[self._archive_key(kind=token, identifier=item_id)] = archive_info
        self._write_archive_state(state)
        return {"kind": token, "identifier": item_id, "archiveStatus": archive_info}


__all__ = ["MemoryLifecycleService"]
