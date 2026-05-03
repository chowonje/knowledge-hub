from __future__ import annotations

import re
from typing import Any


_VAULT_NOTE_ID_RE = re.compile(r"\bvault:[^\s,;]+", re.IGNORECASE)
_VAULT_MARKDOWN_EXT_RE = r"\.(?:md|markdown)(?=$|[^\w./-]|[가-힣])"
_VAULT_PATH_WITH_DIR_RE = re.compile(
    rf"(?<![\w가-힣])(?:[\w가-힣._-]+/)+[\w가-힣._ -]+{_VAULT_MARKDOWN_EXT_RE}",
    re.IGNORECASE,
)
_VAULT_ROOT_PATH_RE = re.compile(
    rf"(?<![\w가-힣/.-])[\w가-힣._-]+(?:\s+[\w가-힣._-]+){{0,6}}{_VAULT_MARKDOWN_EXT_RE}",
    re.IGNORECASE,
)
_VAULT_SCOPE_PREFIX_WORDS = {
    "please",
    "summarize",
    "summary",
    "show",
    "tell",
    "explain",
    "read",
    "open",
    "latest",
    "recent",
    "newest",
    "updated",
    "최근",
    "최신",
    "요약",
    "요약해줘",
    "설명",
    "설명해줘",
    "보여줘",
    "알려줘",
}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _strip_scope_prefix(candidate: str) -> str:
    parts = _clean_text(candidate).split()
    while len(parts) > 1 and parts[0].casefold().strip(":,;") in _VAULT_SCOPE_PREFIX_WORDS:
        parts = parts[1:]
    return _clean_text(" ".join(parts))


def _clean_note_id(value: str) -> str:
    return _clean_text(value).rstrip("?!.,)]}\"'”’")


def vault_scope_from_filter(metadata_filter: dict[str, Any] | None) -> str:
    scoped = dict(metadata_filter or {})
    return _clean_text(scoped.get("note_id") or scoped.get("file_path"))


def vault_scope_from_query(query: str) -> str:
    body = str(query or "")
    match = _VAULT_NOTE_ID_RE.search(body)
    if match:
        return _clean_note_id(match.group(0))
    match = _VAULT_PATH_WITH_DIR_RE.search(body)
    if match:
        return _clean_text(match.group(0))
    match = _VAULT_ROOT_PATH_RE.search(body)
    if not match:
        return ""
    return _strip_scope_prefix(match.group(0))


def explicit_vault_scope(query: str, metadata_filter: dict[str, Any] | None = None) -> str:
    return vault_scope_from_filter(metadata_filter) or vault_scope_from_query(query)


__all__ = [
    "explicit_vault_scope",
    "vault_scope_from_filter",
    "vault_scope_from_query",
]
