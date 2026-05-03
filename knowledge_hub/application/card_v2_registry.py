"""Shared CardV2 builder registry for source-aware runtime surfaces."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import json

from knowledge_hub.document_memory import DocumentMemoryBuilder
from knowledge_hub.papers.card_v2_builder import PaperCardV2Builder
from knowledge_hub.vault.card_v2_builder import VaultCardV2Builder
from knowledge_hub.web.card_v2_builder import WebCardV2Builder
from knowledge_hub.web.ingest import make_web_note_id


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalized_source_kind(value: Any) -> str:
    token = _clean_text(value).casefold()
    if token == "note":
        return "vault"
    return token


def _parse_note_metadata(row: dict[str, Any] | None) -> dict[str, Any]:
    raw = (row or {}).get("metadata")
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(raw or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_timestamp(value: Any) -> datetime | None:
    token = _clean_text(value)
    if not token:
        return None
    try:
        if token.endswith("Z"):
            return datetime.fromisoformat(token.replace("Z", "+00:00")).astimezone(timezone.utc)
        parsed = datetime.fromisoformat(token)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _has_newer_upstream(card_updated_at: Any, *upstream_updated_at: Any) -> bool:
    baseline = _parse_timestamp(card_updated_at)
    if baseline is None:
        return False
    for candidate in upstream_updated_at:
        parsed = _parse_timestamp(candidate)
        if parsed is not None and parsed > baseline:
            return True
    return False


class _CardV2SourceHandler:
    def __init__(self, sqlite_db, builder):
        self.sqlite_db = sqlite_db
        self.builder = builder

    def get_existing(self, source_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def needs_rebuild(self, source_id: str, existing: dict[str, Any] | None) -> bool:
        raise NotImplementedError

    def prepare_build(self, source_id: str) -> None:
        _ = source_id

    def build_and_store(self, source_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def ensure_card(self, source_id: str) -> dict[str, Any] | None:
        existing = self.get_existing(source_id)
        if not self.needs_rebuild(source_id, existing):
            return dict(existing or {}) if existing else None
        self.prepare_build(source_id)
        try:
            stored = self.build_and_store(source_id)
            return dict(stored or {}) if stored else None
        except Exception:
            return dict(existing or {}) if existing else None

    def resolve_scoped_card(
        self,
        *,
        source_id: str | None = None,
        document_id: str | None = None,
        file_path: str | None = None,
    ) -> dict[str, Any] | None:
        token = _clean_text(source_id)
        if not token:
            return None
        return self.ensure_card(token)

    def resolve_scoped_cards(
        self,
        *,
        source_ids: list[str] | None = None,
        document_ids: list[str] | None = None,
        file_paths: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for source_id in list(source_ids or []):
            item = self.resolve_scoped_card(source_id=source_id)
            if item:
                result.append(item)
        return result


class _PaperCardV2SourceHandler(_CardV2SourceHandler):
    def get_existing(self, source_id: str) -> dict[str, Any] | None:
        existing = self.sqlite_db.get_paper_card_v2(source_id)
        return dict(existing or {}) if existing else None

    def needs_rebuild(self, source_id: str, existing: dict[str, Any] | None) -> bool:
        if not existing:
            return True
        card_id = _clean_text(existing.get("card_id"))
        if not self.sqlite_db.list_evidence_anchors_v2(card_id=card_id):
            return True
        if not self.sqlite_db.list_paper_card_claim_refs_v2(card_id=card_id):
            note_id = f"paper:{source_id}"
            if list(self.sqlite_db.list_claims_by_note(note_id, limit=1)) or list(self.sqlite_db.list_claims_by_entity(note_id, limit=1)):
                return True
        memory_card = self.sqlite_db.get_paper_memory_card(source_id) or {}
        document_summary = self.sqlite_db.get_document_memory_summary(f"paper:{source_id}") or {}
        return _has_newer_upstream(existing.get("updated_at"), memory_card.get("updated_at"), document_summary.get("updated_at"))

    def build_and_store(self, source_id: str) -> dict[str, Any] | None:
        return self.builder.build_and_store(paper_id=source_id)


class _WebCardV2SourceHandler(_CardV2SourceHandler):
    def get_existing(self, source_id: str) -> dict[str, Any] | None:
        existing = self.sqlite_db.get_web_card_v2_by_url(source_id)
        return dict(existing or {}) if existing else None

    def needs_rebuild(self, source_id: str, existing: dict[str, Any] | None) -> bool:
        if not existing:
            return True
        card_id = _clean_text(existing.get("card_id"))
        if not self.sqlite_db.list_web_evidence_anchors_v2(card_id=card_id):
            return True
        note_id = _clean_text(existing.get("document_id"))
        note = self.sqlite_db.get_note(note_id) or {}
        document_summary = self.sqlite_db.get_document_memory_summary(note_id) or {}
        if not self.sqlite_db.list_web_card_claim_refs_v2(card_id=card_id):
            metadata = _parse_note_metadata(note)
            for key in ("record_id", "source_item_id", "canonical_url", "url"):
                token = _clean_text(metadata.get(key))
                if token and list(self.sqlite_db.list_claims_by_record(token, limit=1)):
                    return True
            if list(self.sqlite_db.list_claims_by_note(note_id, limit=1)):
                return True
        return _has_newer_upstream(existing.get("updated_at"), note.get("updated_at"), document_summary.get("updated_at"))

    def prepare_build(self, source_id: str) -> None:
        document_id = make_web_note_id(source_id)
        if not self.sqlite_db.get_document_memory_summary(document_id):
            DocumentMemoryBuilder(self.sqlite_db).build_and_store_web(canonical_url=source_id)

    def build_and_store(self, source_id: str) -> dict[str, Any] | None:
        return self.builder.build_and_store(canonical_url=source_id)

    def resolve_scoped_card(
        self,
        *,
        source_id: str | None = None,
        document_id: str | None = None,
        file_path: str | None = None,
    ) -> dict[str, Any] | None:
        token = _clean_text(source_id)
        if token:
            return self.ensure_card(token)
        doc_id = _clean_text(document_id)
        if not doc_id:
            return None
        scoped = self.sqlite_db.get_web_card_v2(doc_id)
        if scoped:
            return dict(scoped)
        note = self.sqlite_db.get_note(doc_id) or {}
        metadata = _parse_note_metadata(note)
        canonical_url = _clean_text(metadata.get("canonical_url") or metadata.get("source_url") or metadata.get("url"))
        if not canonical_url:
            return None
        return self.ensure_card(canonical_url)

    def resolve_scoped_cards(
        self,
        *,
        source_ids: list[str] | None = None,
        document_ids: list[str] | None = None,
        file_paths: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for source_id in list(source_ids or []):
            item = self.resolve_scoped_card(source_id=source_id)
            if item:
                result.append(item)
        for document_id in list(document_ids or []):
            item = self.resolve_scoped_card(document_id=document_id)
            if item:
                result.append(item)
        return result


class _VaultCardV2SourceHandler(_CardV2SourceHandler):
    def get_existing(self, source_id: str) -> dict[str, Any] | None:
        existing = self.sqlite_db.get_vault_card_v2(source_id)
        return dict(existing or {}) if existing else None

    def needs_rebuild(self, source_id: str, existing: dict[str, Any] | None) -> bool:
        if not existing:
            return True
        card_id = _clean_text(existing.get("card_id"))
        if not self.sqlite_db.list_vault_evidence_anchors_v2(card_id=card_id):
            return True
        if not self.sqlite_db.list_vault_card_claim_refs_v2(card_id=card_id):
            if list(self.sqlite_db.list_claims_by_note(source_id, limit=1)):
                return True
        note = self.sqlite_db.get_note(source_id) or {}
        document_summary = self.sqlite_db.get_document_memory_summary(source_id) or {}
        return _has_newer_upstream(existing.get("updated_at"), note.get("updated_at"), document_summary.get("updated_at"))

    def build_and_store(self, source_id: str) -> dict[str, Any] | None:
        return self.builder.build_and_store(note_id=source_id)

    def resolve_scoped_card(
        self,
        *,
        source_id: str | None = None,
        document_id: str | None = None,
        file_path: str | None = None,
    ) -> dict[str, Any] | None:
        token = _clean_text(source_id) or _clean_text(document_id)
        if token:
            return self.ensure_card(token)
        scoped_path = _clean_text(file_path)
        if not scoped_path:
            return None
        for row in self.sqlite_db.list_notes(limit=1_000):
            if _normalized_source_kind(row.get("source_type")) != "vault":
                continue
            if _clean_text(row.get("file_path")) != scoped_path:
                continue
            note_id = _clean_text(row.get("id"))
            if note_id:
                return self.ensure_card(note_id)
        return None

    def resolve_scoped_cards(
        self,
        *,
        source_ids: list[str] | None = None,
        document_ids: list[str] | None = None,
        file_paths: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for source_id in list(source_ids or []):
            item = self.resolve_scoped_card(source_id=source_id)
            if item:
                result.append(item)
        for document_id in list(document_ids or []):
            item = self.resolve_scoped_card(document_id=document_id)
            if item:
                result.append(item)
        for file_path in list(file_paths or []):
            item = self.resolve_scoped_card(file_path=file_path)
            if item:
                result.append(item)
        return result


class CardV2BuilderRegistry:
    def __init__(self, sqlite_db):
        self.sqlite_db = sqlite_db
        self._builders = {
            "paper": PaperCardV2Builder(sqlite_db),
            "web": WebCardV2Builder(sqlite_db),
            "vault": VaultCardV2Builder(sqlite_db),
        }
        self._handlers = {
            "paper": _PaperCardV2SourceHandler(sqlite_db, self._builders["paper"]),
            "web": _WebCardV2SourceHandler(sqlite_db, self._builders["web"]),
            "vault": _VaultCardV2SourceHandler(sqlite_db, self._builders["vault"]),
        }

    def get(self, source_kind: str) -> Any:
        token = str(source_kind or "").strip().lower()
        builder = self._builders.get(token)
        if builder is None:
            raise ValueError(f"unsupported CardV2 source_kind: {source_kind}")
        return builder

    def get_handler(self, source_kind: str) -> _CardV2SourceHandler:
        token = str(source_kind or "").strip().lower()
        handler = self._handlers.get(token)
        if handler is None:
            raise ValueError(f"unsupported CardV2 source_kind: {source_kind}")
        return handler

    def get_existing(self, *, source_kind: str, source_id: str) -> dict[str, Any] | None:
        normalized_source_id = str(source_id or "").strip()
        if not normalized_source_id:
            raise ValueError("source_id is required for CardV2 access")
        return self.get_handler(source_kind).get_existing(normalized_source_id)

    def needs_rebuild(self, *, source_kind: str, source_id: str, existing: dict[str, Any] | None = None) -> bool:
        normalized_source_id = str(source_id or "").strip()
        if not normalized_source_id:
            raise ValueError("source_id is required for CardV2 rebuild check")
        handler = self.get_handler(source_kind)
        current = dict(existing or {}) if existing else handler.get_existing(normalized_source_id)
        return handler.needs_rebuild(normalized_source_id, current)

    def build_and_store(self, *, source_kind: str, source_id: str) -> dict[str, Any] | None:
        normalized_source_id = str(source_id or "").strip()
        if not normalized_source_id:
            raise ValueError("source_id is required for CardV2 build")
        return self.get_handler(source_kind).build_and_store(normalized_source_id)

    def ensure_card(self, *, source_kind: str, source_id: str) -> dict[str, Any] | None:
        token = str(source_kind or "").strip().lower()
        normalized_source_id = str(source_id or "").strip()
        if not normalized_source_id:
            raise ValueError("source_id is required for CardV2 ensure")
        return self.get_handler(token).ensure_card(normalized_source_id)

    def ensure_cards(self, *, source_kind: str, source_ids: list[str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for source_id in source_ids:
            item = self.ensure_card(source_kind=source_kind, source_id=source_id)
            if item:
                result.append(item)
        return result

    def resolve_scoped_card(
        self,
        *,
        source_kind: str,
        source_id: str | None = None,
        document_id: str | None = None,
        file_path: str | None = None,
    ) -> dict[str, Any] | None:
        return self.get_handler(source_kind).resolve_scoped_card(
            source_id=source_id,
            document_id=document_id,
            file_path=file_path,
        )

    def resolve_scoped_cards(
        self,
        *,
        source_kind: str,
        source_ids: list[str] | None = None,
        document_ids: list[str] | None = None,
        file_paths: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return self.get_handler(source_kind).resolve_scoped_cards(
            source_ids=source_ids,
            document_ids=document_ids,
            file_paths=file_paths,
        )


__all__ = ["CardV2BuilderRegistry"]
