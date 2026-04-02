"""Canonical payload helpers for document-memory units."""

from __future__ import annotations

from typing import Any

from knowledge_hub.document_memory.models import DocumentMemoryUnit
from knowledge_hub.knowledge.semantic_units import document_semantic_units_payload


def unit_payload(value: dict[str, Any] | DocumentMemoryUnit | None) -> dict[str, Any]:
    if isinstance(value, DocumentMemoryUnit):
        return value.to_payload()
    if isinstance(value, dict):
        unit = DocumentMemoryUnit.from_row(value)
        if unit is not None:
            return unit.to_payload()
    return {}


def semantic_units_payload(document: dict[str, Any] | None) -> dict[str, Any]:
    return document_semantic_units_payload(document)
