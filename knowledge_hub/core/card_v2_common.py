"""Shared low-level helpers for source-specific CardV2 builders."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def first_nonempty(*values: Any) -> str:
    for value in values:
        token = clean_text(value)
        if token:
            return token
    return ""


def coverage_status(value: Any) -> str:
    return "complete" if clean_text(value) else "missing"


def snippet_hash(*parts: Any) -> str:
    payload = "||".join(clean_text(part) for part in parts if clean_text(part))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20] if payload else ""


def stable_anchor_id(card_id: str, role: str, label: str, section_path: str, excerpt: str) -> str:
    digest = hashlib.sha1(f"{card_id}|{role}|{label}|{section_path}|{excerpt}".encode("utf-8")).hexdigest()[:16]
    return f"anchor:{digest}"


def slot_excerpt(unit: dict[str, Any]) -> str:
    return first_nonempty(unit.get("contextual_summary"), unit.get("source_excerpt"), unit.get("document_thesis"))


def unit_matches(unit: dict[str, Any], pattern: re.Pattern[str]) -> bool:
    haystack = " ".join(
        [
            str(unit.get("unit_type") or ""),
            str(unit.get("title") or ""),
            str(unit.get("section_path") or ""),
            str(unit.get("contextual_summary") or ""),
        ]
    )
    return bool(pattern.search(haystack))


def best_unit(units: list[dict[str, Any]], pattern: re.Pattern[str]) -> dict[str, Any] | None:
    for unit in units:
        if unit_matches(unit, pattern):
            return unit
    return None


def parse_note_metadata(row: dict[str, Any] | None) -> dict[str, Any]:
    raw = (row or {}).get("metadata")
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(raw or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def claim_is_accepted(claim: dict[str, Any]) -> bool:
    ptrs = claim.get("evidence_ptrs") if isinstance(claim.get("evidence_ptrs"), list) else []
    decisions = {
        str(ptr.get("claim_decision") or "").strip().lower()
        for ptr in ptrs
        if isinstance(ptr, dict) and str(ptr.get("claim_decision") or "").strip()
    }
    if not decisions:
        return True
    return "accepted" in decisions


__all__ = [
    "best_unit",
    "claim_is_accepted",
    "clean_text",
    "coverage_status",
    "first_nonempty",
    "parse_note_metadata",
    "slot_excerpt",
    "snippet_hash",
    "stable_anchor_id",
    "unit_matches",
]
