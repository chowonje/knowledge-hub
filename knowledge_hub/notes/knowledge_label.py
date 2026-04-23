"""Helpers for compact Obsidian-facing AI knowledge labels."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


_KNOWLEDGE_KIND_ORDER = (
    "algorithm",
    "software_system",
    "hardware_system",
    "theory",
    "data",
    "evaluation",
    "operation",
    "application",
)

_LIFECYCLE_STAGE_ORDER = (
    "inference",
    "training",
    "evaluation",
    "operation",
    "research",
)


def _order_index(values: Sequence[str], token: str) -> int:
    try:
        return values.index(token)
    except ValueError:
        return len(values)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_values = value
    else:
        raw_values = [value]
    items: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        token = str(raw or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        items.append(token)
    return items


def summarize_ai_knowledge_label(
    entities: Sequence[Mapping[str, Any]] | None,
    *,
    max_kinds: int = 2,
) -> str:
    kind_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()

    for entity in entities or []:
        properties = entity.get("properties") if isinstance(entity, Mapping) else {}
        if not isinstance(properties, Mapping):
            continue
        knowledge_kind = str(properties.get("knowledge_kind") or "").strip().lower()
        if knowledge_kind in _KNOWLEDGE_KIND_ORDER:
            kind_counts[knowledge_kind] += 1
        facets = properties.get("facets")
        if not isinstance(facets, Mapping):
            continue
        for stage in _as_list(facets.get("lifecycle_stage")):
            if stage in _LIFECYCLE_STAGE_ORDER:
                stage_counts[stage] += 1

    if not kind_counts:
        return ""

    ordered_kinds = sorted(
        kind_counts,
        key=lambda token: (-kind_counts[token], _order_index(_KNOWLEDGE_KIND_ORDER, token), token),
    )
    parts = ordered_kinds[: max(1, int(max_kinds))]

    if stage_counts:
        ordered_stages = sorted(
            stage_counts,
            key=lambda token: (-stage_counts[token], _order_index(_LIFECYCLE_STAGE_ORDER, token), token),
        )
        parts.append(ordered_stages[0])

    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part in seen:
            continue
        seen.add(part)
        deduped.append(part)
    return " / ".join(deduped[:3])
