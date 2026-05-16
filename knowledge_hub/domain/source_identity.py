from __future__ import annotations

from pathlib import Path
import re
from typing import Any


_ARXIV_RE = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", re.IGNORECASE)
_HEX_SUFFIX_RE = re.compile(r"(?:[_-][0-9a-f]{6,})+$", re.IGNORECASE)
_HASH_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{32,64}$", re.IGNORECASE)
_PAPER_CHUNK_RE = re.compile(r"^paper[_:](\d{4}\.\d{4,5})(?:[_:#-]\d+)?$", re.IGNORECASE)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalized_text(value: Any) -> str:
    token = _clean_text(value)
    if not token:
        return ""
    token = Path(token).stem if "/" in token or token.lower().endswith((".pdf", ".txt", ".md")) else token
    token = token.replace("_", " ")
    token = re.sub(r"[-:/#]+", " ", token)
    token = re.sub(r"[^0-9A-Za-z가-힣.]+", " ", token)
    token = _HEX_SUFFIX_RE.sub("", token.strip())
    return _clean_text(token).casefold()


def source_identity_aliases(value: Any) -> set[str]:
    token = _clean_text(value)
    if not token:
        return set()

    aliases: set[str] = set()
    lowered = token.casefold()
    if _HASH_RE.fullmatch(lowered):
        aliases.add(f"hash:{lowered.removeprefix('sha256:')}")

    for arxiv_id in _ARXIV_RE.findall(token):
        aliases.add(f"arxiv:{arxiv_id.casefold()}")

    chunk_match = _PAPER_CHUNK_RE.fullmatch(lowered)
    if chunk_match:
        aliases.add(f"arxiv:{chunk_match.group(1)}")

    normalized = _normalized_text(token)
    if normalized:
        aliases.add(f"text:{normalized}")
        stripped = _HEX_SUFFIX_RE.sub("", normalized).strip()
        if stripped and stripped != normalized:
            aliases.add(f"text:{stripped}")
    return aliases


def source_item_aliases(item: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for key in (
        "source_id",
        "sourceId",
        "source_ref",
        "sourceRef",
        "target",
        "id",
        "arxiv_id",
        "paper_id",
        "paperId",
        "document_id",
        "documentId",
        "chunk_id",
        "chunkId",
        "citation_target",
        "citationTarget",
        "title",
        "parent_label",
        "parentLabel",
        "section_path",
        "sectionPath",
        "file_path",
        "filePath",
    ):
        aliases.update(source_identity_aliases(item.get(key)))
    for key in ("sourceContentHash", "source_content_hash"):
        value = _clean_text(item.get(key))
        if value:
            aliases.add(f"hash:{value.casefold().removeprefix('sha256:')}")
    return aliases


def source_refs_match(expected: Any, item: dict[str, Any]) -> bool:
    expected_aliases = source_identity_aliases(expected)
    if not expected_aliases:
        return False
    return bool(expected_aliases & source_item_aliases(item))


def alias_groups_for_items(items: list[dict[str, Any]]) -> list[set[str]]:
    groups: list[set[str]] = []
    for item in items:
        aliases = source_item_aliases(item)
        if not aliases:
            continue
        matched_indexes = [idx for idx, group in enumerate(groups) if aliases & group]
        if not matched_indexes:
            groups.append(set(aliases))
            continue
        first = matched_indexes[0]
        groups[first].update(aliases)
        for idx in reversed(matched_indexes[1:]):
            groups[first].update(groups[idx])
            del groups[idx]
    return groups


def aliases_with_groups(item: dict[str, Any], groups: list[set[str]]) -> set[str]:
    aliases = source_item_aliases(item)
    expanded = set(aliases)
    for group in groups:
        if aliases & group:
            expanded.update(group)
    return expanded


__all__ = [
    "alias_groups_for_items",
    "aliases_with_groups",
    "source_identity_aliases",
    "source_item_aliases",
    "source_refs_match",
]
