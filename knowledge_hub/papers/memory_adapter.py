"""Bridge paper-specific memory cards into shared section-style views."""

from __future__ import annotations

from typing import Any

from knowledge_hub.papers.memory_payloads import shared_slot_payload


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _clean_list(values: Any, *, limit: int | None = None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        items = [values]
    else:
        try:
            items = list(values)
        except Exception:
            items = [values]
    result: list[str] = []
    seen: set[str] = set()
    for raw in items:
        token = _clean_text(raw)
        if not token:
            continue
        lowered = token.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(token)
        if limit is not None and len(result) >= limit:
            break
    return result


def paper_memory_card_to_section_cards(
    memory_payload: dict[str, Any] | None,
    *,
    source_card_id: str,
    paper_id: str,
    title: str,
) -> list[dict[str, Any]]:
    payload = shared_slot_payload(memory_payload)
    memory_id = _clean_text(payload.get("memory_id")) or f"paper-memory:{_clean_text(paper_id)}"
    normalized_paper_id = _clean_text(paper_id)
    document_id = f"paper:{normalized_paper_id}"
    document_title = _clean_text(title) or _clean_text(payload.get("title")) or normalized_paper_id
    document_thesis = _clean_text(payload.get("paper_core"))
    concepts = _clean_list(payload.get("concept_links"), limit=12)
    claims = _clean_list(payload.get("claim_refs"), limit=12)

    slot_specs = [
        ("problem", _clean_text(payload.get("problem_core") or payload.get("paper_core")), "summary", 0, "Overview"),
        ("method", _clean_text(payload.get("method_core")), "method", 1, "Method"),
        ("results", _clean_text(payload.get("evidence_core")), "result", 2, "Evidence"),
        ("limitations", _clean_text(payload.get("limitations_core")), "limitation", 3, "Limitations"),
    ]
    result: list[dict[str, Any]] = []
    for role, summary, unit_type, order_index, label in slot_specs:
        if not summary:
            continue
        result.append(
            {
                "section_card_id": f"paper-memory-section-card:{normalized_paper_id}:{role}",
                "source_kind": "paper",
                "source_card_id": _clean_text(source_card_id),
                "source_id": normalized_paper_id,
                "paper_id": normalized_paper_id,
                "document_id": document_id,
                "unit_id": f"{memory_id}:{role}",
                "title": f"{document_title} {label}".strip(),
                "section_path": f"Paper Memory > {label}",
                "unit_type": unit_type,
                "role": role,
                "order_index": order_index,
                "contextual_summary": summary,
                "source_excerpt": summary,
                "document_thesis": document_thesis,
                "confidence": 0.56 if role == "problem" else 0.58,
                "claims": list(claims),
                "concepts": list(concepts),
                "provenance": {
                    "source": "paper_memory_card",
                    "memory_id": memory_id,
                    "published_at": _clean_text(payload.get("published_at")),
                },
                "appendix_like": False,
                "search_text": _clean_text(f"{document_title} {role} {summary} {document_thesis} {' '.join(concepts)}"),
                "origin": "paper_memory_adapter_v1",
            }
        )
    return result


__all__ = ["paper_memory_card_to_section_cards"]
